//! GitHub Copilot client implementation.

use crate::{
    CopilotError,
    constant::{COPILOT_BASE_URL, COPILOT_INTEGRATION_ID, DEFAULT_MODEL, EDITOR_VERSION},
};
use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest, Message, Role, ToolCall, Usage,
        model::{
            Ability, OpenAIPromptCacheRetention, Parameters, Profile as ModelProfile, ToolChoice,
        },
        tool::ToolDefinition,
    },
};
use async_io::Timer;
use futures_core::Stream;
use futures_lite::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};
use zenwave::{Client, client, header};

/// GitHub Copilot language model client.
///
/// Uses the OpenAI-compatible chat completions API at `api.githubcopilot.com`.
#[derive(Clone, Debug)]
pub struct Copilot {
    inner: Arc<Config>,
}

impl Copilot {
    /// Create a new Copilot client with the given OAuth token.
    pub fn new(token: impl Into<String>) -> Self {
        Self::builder(token).build()
    }

    /// Create a builder for configuring the Copilot client.
    #[must_use]
    pub fn builder(token: impl Into<String>) -> Builder {
        Builder::new(token)
    }

    /// Override the default chat model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).model = model.into().trim().to_string();
        self
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn with_oauth_token(mut self, token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).oauth_token = Some(token.into());
        self
    }
}

impl LanguageModel for Copilot {
    type Error = CopilotError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let (messages, parameters, tool_defs) = request.into_parts();
        let tool_defs = filter_tool_definitions(tool_defs, &parameters.tool_choice);

        async_stream::stream! {
            if parameters.cache.claude.is_some() || parameters.cache.gemini.is_some() {
                yield Err(CopilotError::Api(
                    "Copilot provider only accepts cache.openai settings".to_string(),
                ));
                return;
            }

            let payload_messages = to_chat_messages(&messages);
            let openai_tools = if tool_defs.is_empty() {
                None
            } else {
                Some(convert_tools(&tool_defs))
            };

            let mut events = chat_completions_stream(cfg, payload_messages, &parameters, openai_tools);

            while let Some(event) = events.next().await {
                yield event;
            }
        }
    }

    fn profile(&self) -> impl std::future::Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to get context window from models database
            let context_length = aither_models::lookup(&cfg.model)
                .and_then(aither_models::ModelEntry::max_input_tokens)
                .unwrap_or_else(|| {
                    const DEFAULT_CONTEXT_LENGTH: u32 = 8192;
                    tracing::warn!(
                        model = %cfg.model,
                        fallback_context_length = DEFAULT_CONTEXT_LENGTH,
                        "Copilot model missing context metadata in aither-models, using fallback",
                    );
                    DEFAULT_CONTEXT_LENGTH
                });

            ModelProfile::new(
                cfg.model.clone(),
                "GitHub Copilot",
                cfg.model.clone(),
                "GitHub Copilot model",
                context_length,
            )
            .with_ability(Ability::ToolUse)
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    token: String,
    base_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

/// Builder for [`Copilot`] clients.
#[derive(Debug)]
pub struct Builder {
    token: String,
    base_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

impl Builder {
    fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base_url: COPILOT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            editor_version: EDITOR_VERSION.to_string(),
            integration_id: COPILOT_INTEGRATION_ID.to_string(),
            oauth_token: None,
        }
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into().trim().to_string();
        self
    }

    /// Set the editor version header.
    #[must_use]
    pub fn editor_version(mut self, version: impl Into<String>) -> Self {
        self.editor_version = version.into();
        self
    }

    /// Set the integration ID header.
    #[must_use]
    pub fn integration_id(mut self, id: impl Into<String>) -> Self {
        self.integration_id = id.into();
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into());
        self
    }

    /// Build the Copilot client.
    #[must_use]
    pub fn build(self) -> Copilot {
        Copilot {
            inner: Arc::new(Config {
                token: self.token,
                base_url: self.base_url,
                model: self.model,
                editor_version: self.editor_version,
                integration_id: self.integration_id,
                oauth_token: self.oauth_token,
            }),
        }
    }
}

// === Request/Response Types ===

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessagePayload>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoicePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize, Clone)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize, Clone)]
struct ChatMessagePayload {
    role: &'static str,
    content: ContentPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum ContentPayload {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPayload },
}

#[derive(Debug, Clone, Serialize)]
struct ImageUrlPayload {
    url: String,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallPayload {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ChatToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Clone)]
struct ToolPayload {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoicePayload {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    usage: Option<ChatCompletionUsage>,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct ChatCompletionUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokenDetails>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct CompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<ChunkToolFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// === Conversion Functions ===

fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessagePayload> {
    messages
        .iter()
        .map(|msg| {
            let (role, tool_calls, tool_call_id) = match msg.role() {
                Role::System => ("system", None, None),
                Role::User => ("user", None, None),
                Role::Assistant => {
                    let calls = if msg.tool_calls().is_empty() {
                        None
                    } else {
                        Some(
                            msg.tool_calls()
                                .iter()
                                .map(|tc| ChatToolCallPayload {
                                    id: tc.id.clone(),
                                    kind: "function",
                                    function: ChatToolCallFunction {
                                        name: tc.name.clone(),
                                        arguments: tc.arguments.to_string(),
                                    },
                                })
                                .collect(),
                        )
                    };
                    ("assistant", calls, None)
                }
                Role::Tool => (
                    "tool",
                    None,
                    Some(msg.tool_call_id().unwrap_or_default().to_string()),
                ),
            };
            ChatMessagePayload {
                role,
                content: build_content(msg),
                tool_calls,
                tool_call_id,
            }
        })
        .collect()
}

fn build_content(message: &Message) -> ContentPayload {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return ContentPayload::Text(message.content().to_owned());
    }

    let mut parts = Vec::new();

    for attachment in attachments {
        if let Some(data_url) = url_to_data_url(attachment) {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrlPayload { url: data_url },
            });
        }
    }

    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    ContentPayload::Parts(parts)
}

fn url_to_data_url(url: &url::Url) -> Option<String> {
    match url.scheme() {
        // Data URLs are already in the wire format, and Copilot fetches http(s)
        // itself, so both go across untouched.
        "data" | "http" | "https" => Some(url.as_str().to_string()),
        "file" => read_file_to_data_url(url),
        _ => {
            tracing::warn!("Unsupported attachment URL scheme: {}", url.scheme());
            None
        }
    }
}

fn read_file_to_data_url(url: &url::Url) -> Option<String> {
    use base64::Engine;

    let path = url.to_file_path().ok()?;
    let data = std::fs::read(&path).ok()?;
    let mime_type = mime_from_path(&path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    Some(format!("data:{mime_type};base64,{base64_data}"))
}

fn mime_from_path(path: &std::path::Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase()
        .as_str()
    {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn convert_tools(tool_defs: &[ToolDefinition]) -> Vec<ToolPayload> {
    tool_defs
        .iter()
        .map(|def| ToolPayload {
            r#type: "function",
            function: ToolFunction {
                name: def.name().to_string(),
                description: def.description().to_string(),
                parameters: def.arguments_openai_schema(),
            },
        })
        .collect()
}

fn tool_choice(params: &Parameters, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => Some(ToolChoicePayload::Mode("auto")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn filter_tool_definitions(defs: Vec<ToolDefinition>, choice: &ToolChoice) -> Vec<ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => defs,
    }
}

fn prompt_cache_retention(params: &Parameters) -> Option<&'static str> {
    params
        .cache
        .openai
        .as_ref()
        .and_then(|cache| cache.retention)
        .map(OpenAIPromptCacheRetention::as_str)
}

// === Streaming ===

const SSE_FIRST_EVENT_TIMEOUT: Duration = Duration::from_secs(90);
const SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

async fn open_sse_stream(
    cfg: &Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    let endpoint = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();

    let build_result = backend
        .post(&endpoint)
        .and_then(|b| {
            b.header(
                header::AUTHORIZATION.as_str(),
                format!("Bearer {}", cfg.token),
            )
        })
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-copilot/0.1"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"))
        .and_then(|b| b.header("editor-version", cfg.editor_version.clone()))
        .and_then(|b| b.header("Copilot-Integration-Id", cfg.integration_id.clone()));

    let builder = build_result.map_err(CopilotError::Http)?;

    match builder
        .json_body(request)
        .map_err(CopilotError::Http)?
        .sse()
        .await
    {
        Ok(stream) => Ok(stream),
        Err(zenwave::Error::Timeout) => Err(CopilotError::Timeout),
        Err(e) => Err(CopilotError::Http(e)),
    }
}

const fn is_unauthorized(err: &CopilotError) -> bool {
    matches!(
        err,
        CopilotError::Http(zenwave::Error::Http { status, .. }) if status.as_u16() == 401
    )
}

async fn refresh_session_config(cfg: &Config) -> Result<Option<Config>, CopilotError> {
    let Some(oauth_token) = cfg.oauth_token.as_deref() else {
        return Ok(None);
    };

    let session = crate::auth::get_session_token(oauth_token).await?;

    let mut refreshed = cfg.clone();
    refreshed.token = session.token;
    refreshed.base_url = session.api_endpoint;

    Ok(Some(refreshed))
}

fn chat_completions_stream(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: &Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send + Unpin {
    let params = params.clone();
    Box::pin(chat_completions_stream_inner(
        cfg,
        payload_messages,
        params,
        tools,
    ))
}

/// Assembles the streaming chat-completion request.
fn build_chat_request(
    cfg: &Config,
    messages: Vec<ChatMessagePayload>,
    params: &Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> ChatCompletionRequest {
    let has_tools = tools.as_ref().is_some_and(|t| !t.is_empty());
    ChatCompletionRequest {
        model: cfg.model.clone(),
        messages,
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: true,
        }),
        temperature: params.temperature,
        top_p: params.top_p,
        max_tokens: params.max_tokens,
        tools,
        tool_choice: tool_choice(params, has_tools),
        prompt_cache_key: params
            .cache
            .openai
            .as_ref()
            .and_then(|cache| cache.key.clone()),
        prompt_cache_retention: prompt_cache_retention(params),
    }
}

/// Opens the SSE stream, minting a fresh session token if the current one has
/// expired.
///
/// Copilot session tokens are short-lived, so a 401 on an otherwise valid
/// request is routine rather than a real authorization failure; `cfg` is
/// updated in place so the caller keeps the refreshed token.
///
/// # Errors
///
/// Returns the original error if the token cannot be refreshed, and the retry's
/// error if the refreshed token is also rejected.
async fn open_sse_stream_refreshing(
    cfg: &mut Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    let err = match open_sse_stream(cfg, request).await {
        Ok(stream) => return Ok(stream),
        Err(err) if is_unauthorized(&err) => err,
        Err(err) => return Err(err),
    };

    match refresh_session_config(cfg).await? {
        Some(refreshed) => {
            *cfg = refreshed;
            open_sse_stream(cfg, request).await
        }
        None => Err(err),
    }
}

/// Reads an `{"error": ...}` body that arrived in place of a stream chunk.
fn sse_payload_error(data: &str) -> Option<CopilotError> {
    let value = serde_json::from_str::<Value>(data).ok()?;
    let error = value.get("error")?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Unknown API error");
    Some(CopilotError::Api(message.to_string()))
}

/// What one chat-completion chunk contributed to the stream.
struct ChunkOutcome {
    /// Events to hand to the caller, in wire order.
    events: Vec<Event>,
    /// Whether the chunk carried content, tool-call deltas, or usage. These are
    /// the signals that the connection is alive, so they reset the idle timer.
    made_progress: bool,
    /// Whether every choice in the chunk reported a finish reason.
    finished: bool,
}

/// Folds one chunk into the accumulated tool calls and usage.
fn apply_chunk(
    chunk: &ChatCompletionChunk,
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
    usage: &mut Option<Usage>,
) -> ChunkOutcome {
    let mut outcome = ChunkOutcome {
        events: Vec::new(),
        made_progress: false,
        finished: false,
    };

    if let Some(chunk_usage) = &chunk.usage {
        *usage = Some(usage_from_chat_completion(chunk_usage));
        outcome.made_progress = true;
    }

    outcome.finished = !chunk.choices.is_empty()
        && chunk
            .choices
            .iter()
            .all(|choice| choice.finish_reason.is_some());

    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            outcome.made_progress = true;
            outcome.events.push(Event::Text(content.clone()));
        }

        if let Some(calls) = &choice.delta.tool_calls
            && !calls.is_empty()
        {
            outcome.made_progress = true;
            accumulate_tool_call_deltas(tool_calls, calls);
        }
    }

    outcome
}

/// Folds one chunk's tool-call deltas into the per-index accumulators.
///
/// The id and name arrive in the first delta for an index and the arguments in
/// pieces after it, so every field is optional on every delta.
fn accumulate_tool_call_deltas(
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
    calls: &[ChunkToolCall],
) {
    for call in calls {
        let acc = tool_calls.entry(call.index.unwrap_or(0)).or_default();
        if let Some(id) = &call.id {
            acc.id = Some(id.clone());
        }
        let Some(function) = &call.function else {
            continue;
        };
        if let Some(name) = &function.name {
            acc.name = Some(name.clone());
        }
        if let Some(args) = &function.arguments {
            acc.arguments.push_str(args);
        }
    }
}

/// Turns the accumulators into tool calls, in the index order the API used.
///
/// An accumulator that never received both an id and a name is incomplete and
/// is dropped rather than emitted as a call the caller cannot answer.
fn drain_chat_tool_calls(tool_calls: HashMap<usize, ToolCallAccumulator>) -> Vec<ToolCall> {
    let mut sorted: Vec<_> = tool_calls.into_iter().collect();
    sorted.sort_by_key(|(index, _)| *index);
    sorted
        .into_iter()
        .filter_map(|(_, acc)| {
            let (Some(id), Some(name)) = (acc.id, acc.name) else {
                return None;
            };
            let arguments = serde_json::from_str(&acc.arguments)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::default()));
            Some(ToolCall {
                id,
                name,
                arguments,
            })
        })
        .collect()
}

/// The next thing to happen on an SSE stream that is also being timed.
enum NextEvent {
    /// An event arrived, or the stream ended (`None`).
    Event(Option<Result<zenwave::sse::Event, zenwave::sse::ParseError>>),
    /// The idle deadline passed first.
    Timeout,
}

/// Waits for the next SSE event, giving up after `timeout`.
async fn next_event_or_timeout(
    sse_stream: &mut Pin<&mut zenwave::sse::SseStream>,
    timeout: Duration,
) -> NextEvent {
    futures_lite::future::race(async { NextEvent::Event(sse_stream.next().await) }, async {
        Timer::after(timeout).await;
        NextEvent::Timeout
    })
    .await
}

fn chat_completions_stream_inner(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    async_stream::stream! {
        let mut cfg = cfg.as_ref().clone();
        let request = build_chat_request(&cfg, payload_messages, &params, tools);

        tracing::debug!(
            request = %serde_json::to_string_pretty(&request).unwrap_or_default(),
            "Sending Copilot chat completion request"
        );

        let sse_stream = match open_sse_stream_refreshing(&mut cfg, &request).await {
            Ok(stream) => stream,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        futures_lite::pin!(sse_stream);

        // Stream SSE events, stopping on [DONE], a finish_reason, or an idle timeout.
        let mut event_count = 0usize;
        let mut saw_payload = false;
        let mut last_progress = Instant::now();
        let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
        let mut usage: Option<Usage> = None;

        loop {
            // Before the first payload the model may still be thinking, so it
            // gets a longer grace period than a stream that has gone quiet.
            let timeout = if saw_payload {
                SSE_IDLE_TIMEOUT
            } else {
                SSE_FIRST_EVENT_TIMEOUT
            };
            let remaining =
                timeout.saturating_sub(Instant::now().saturating_duration_since(last_progress));

            let next = if remaining.is_zero() {
                NextEvent::Timeout
            } else {
                next_event_or_timeout(&mut sse_stream, remaining).await
            };

            let event = match next {
                NextEvent::Event(Some(Ok(event))) => event,
                NextEvent::Event(Some(Err(e))) => {
                    yield Err(CopilotError::Stream(e));
                    return;
                }
                NextEvent::Event(None) => break,
                NextEvent::Timeout => {
                    // Going quiet after real output is an ended stream; going
                    // quiet before any is a request that never started.
                    if saw_payload {
                        tracing::warn!("Copilot SSE idle timeout; ending stream");
                        break;
                    }
                    yield Err(CopilotError::Timeout);
                    return;
                }
            };

            let data = event.text_data();
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                break;
            }
            tracing::trace!(sse_event = %data, "Received Copilot SSE event");
            event_count += 1;

            if let Some(err) = sse_payload_error(data) {
                yield Err(err);
                return;
            }

            let chunk = match serde_json::from_str::<ChatCompletionChunk>(data) {
                Ok(chunk) => chunk,
                Err(e) => {
                    yield Err(CopilotError::Json(e));
                    return;
                }
            };

            let outcome = apply_chunk(&chunk, &mut tool_calls, &mut usage);
            if outcome.made_progress {
                saw_payload = true;
                last_progress = Instant::now();
            }
            for event in outcome.events {
                yield Ok(event);
            }
            if outcome.finished {
                break;
            }
        }

        tracing::debug!(event_count, "Processed Copilot SSE events");

        // Copilot only reports tool calls in deltas, so they are all emitted
        // once the stream has ended.
        for tool_call in drain_chat_tool_calls(tool_calls) {
            yield Ok(Event::ToolCall(tool_call));
        }
        if let Some(final_usage) = usage {
            yield Ok(Event::Usage(final_usage));
        }
    }
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn usage_from_chat_completion(raw: &ChatCompletionUsage) -> Usage {
    Usage {
        prompt_tokens: raw.prompt_tokens,
        completion_tokens: raw.completion_tokens,
        total_tokens: raw.total_tokens,
        reasoning_tokens: raw
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        cache_read_tokens: raw
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        cache_write_tokens: None,
        cost_usd: None,
        stop_reason: None,
    }
}
