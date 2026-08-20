//! Claude API client implementation.

use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest,
        model::{Ability, Profile as ModelProfile, ToolChoice},
        oneshot,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use tracing::debug;
use zenwave::{Client, client, header};

use crate::{
    constant::{ANTHROPIC_VERSION, CLAUDE_BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_MODEL},
    error::ClaudeError,
    request::{
        MessagesRequest, ParameterSnapshot, apply_cache_strategy, convert_tools,
        filter_tool_definitions, to_claude_messages, tool_choice_payload,
    },
    response::{StreamState, parse_event, should_skip_event},
};

/// Claude chat model client for the Anthropic Messages API.
///
/// # Example
///
/// ```ignore
/// use aither_claude::Claude;
/// use aither_core::{LanguageModel, llm::oneshot};
///
/// let client = Claude::new(std::env::var("ANTHROPIC_API_KEY")?);
///
/// let response = client.respond(oneshot(
///     "You are a helpful assistant.",
///     "What is the capital of France?"
/// )).await?;
///
/// println!("{response}");
/// ```
#[derive(Clone, Debug)]
pub struct Claude {
    inner: Arc<Config>,
}

impl Claude {
    /// Create a new client using the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder(api_key).build()
    }

    /// Start building a Claude client with custom configuration.
    #[must_use]
    pub fn builder(api_key: impl Into<String>) -> Builder {
        Builder::new(api_key)
    }

    /// Override the default model in-place.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).model = sanitize_model(model);
        self
    }

    /// Override the base URL (useful for proxies or local deployments).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Override the default `max_tokens`.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        Arc::make_mut(&mut self.inner).default_max_tokens = max_tokens;
        self
    }

    pub(crate) fn config(&self) -> Arc<Config> {
        self.inner.clone()
    }
}

/// Turns a provider-agnostic request into an Anthropic Messages request.
///
/// # Errors
///
/// Returns [`ClaudeError::Api`] if the request carries another provider's cache
/// settings, names an exact tool that was not supplied, or asks for a cache
/// strategy this request cannot satisfy.
fn build_messages_request(
    cfg: &Config,
    request: LLMRequest,
) -> Result<MessagesRequest, ClaudeError> {
    let (core_messages, parameters, tool_definitions) = request.into_parts();

    if parameters.cache.openai.is_some() || parameters.cache.gemini.is_some() {
        return Err(ClaudeError::Api(
            "Claude provider only accepts cache.claude settings".to_string(),
        ));
    }

    let (mut system_prompt, mut claude_messages) = to_claude_messages(&core_messages);
    let snapshot = ParameterSnapshot::from(&parameters);
    let tool_definitions = filter_tool_definitions(tool_definitions, &snapshot.tool_choice);

    if let ToolChoice::Exact(name) = &snapshot.tool_choice
        && tool_definitions.is_empty()
    {
        return Err(ClaudeError::Api(format!(
            "Exact tool choice '{name}' is not present in tool definitions"
        )));
    }

    let has_tools = !tool_definitions.is_empty();
    let mut claude_tools = has_tools.then(|| convert_tools(&tool_definitions));
    let claude_tool_choice = tool_choice_payload(&snapshot.tool_choice, has_tools);

    let cache_control = match snapshot.cache {
        Some(cache) => apply_cache_strategy(
            &mut system_prompt,
            &mut claude_messages,
            &mut claude_tools,
            cache,
        )
        .map_err(ClaudeError::Api)?,
        None => None,
    };

    Ok(MessagesRequest {
        model: cfg.model.clone(),
        max_tokens: snapshot.max_tokens.unwrap_or(cfg.default_max_tokens),
        messages: claude_messages,
        system: system_prompt,
        stream: true,
        temperature: snapshot.temperature,
        top_p: snapshot.top_p,
        top_k: snapshot.top_k,
        stop_sequences: snapshot.stop_sequences,
        tools: claude_tools,
        tool_choice: claude_tool_choice,
        cache_control,
    })
}

/// Opens the Messages SSE stream for an already-built request body.
///
/// # Errors
///
/// Returns [`ClaudeError::Http`] if the request cannot be built or sent.
async fn open_message_stream(
    cfg: &Config,
    body: &MessagesRequest,
) -> Result<zenwave::sse::SseStream, ClaudeError> {
    let mut backend = client();
    let builder = backend
        .post(cfg.request_url("/v1/messages"))
        .and_then(|b| b.header("x-api-key", cfg.api_key.clone()))
        .and_then(|b| b.header("anthropic-version", ANTHROPIC_VERSION))
        .and_then(|b| b.header(header::CONTENT_TYPE.as_str(), "application/json"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"))
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-claude/0.1"))
        .and_then(|b| b.json_body(body))
        .map_err(ClaudeError::Http)?;

    builder.sse().await.map_err(ClaudeError::Http)
}

impl LanguageModel for Claude {
    type Error = ClaudeError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.config();
        // Built before the stream so a malformed request fails on its first
        // poll rather than after the connection has been opened.
        let prepared = build_messages_request(&cfg, request);

        async_stream::stream! {
            let request_body = match prepared {
                Ok(body) => body,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            debug!("Claude request: {:?}", request_body);

            let sse_stream = match open_message_stream(&cfg, &request_body).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            futures_lite::pin!(sse_stream);

            let mut state = StreamState::new();

            while let Some(event) = sse_stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(e) => {
                        yield Err(ClaudeError::from(e));
                        return;
                    }
                };
                if should_skip_event(&event) {
                    continue;
                }
                match parse_event(&event, &mut state) {
                    Ok(llm_events) => {
                        for llm_event in llm_events {
                            yield Ok(llm_event);
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }

            // Tool calls are reported, never executed: the consumer decides.
            for call in state.tool_calls {
                yield Ok(Event::ToolCall(aither_core::llm::ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.input,
                }));
            }

            debug!("Claude response complete, stop_reason: {:?}", state.stop_reason);
        }
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.respond(oneshot(
            "Continue the user provided text without additional commentary.",
            prefix,
        ))
    }

    fn profile(&self) -> impl core::future::Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to fetch context window from proxy API (e.g., OpenRouter)
            // Native Anthropic API doesn't expose this
            let context_length = match fetch_model_context_length(&cfg).await {
                Ok(len) => len,
                Err(e) => {
                    tracing::debug!("API did not return context_length: {e}");
                    // Fallback to models database
                    aither_models::lookup(&cfg.model)
                        .and_then(aither_models::ModelEntry::max_input_tokens)
                        .unwrap_or_else(|| {
                            panic!(
                                "Claude model '{}' missing context metadata from provider and aither-models",
                                cfg.model
                            )
                        })
                }
            };

            let mut profile = ModelProfile::new(
                cfg.model.clone(),
                "Anthropic",
                cfg.model.clone(),
                "Claude model by Anthropic",
                context_length,
            )
            .with_abilities([Ability::ToolUse, Ability::Vision]);

            for ability in &cfg.native_abilities {
                if !profile.abilities.contains(ability) {
                    profile.abilities.push(*ability);
                }
            }
            profile
        }
    }
}

/// Try to fetch context length from the models endpoint.
async fn fetch_model_context_length(cfg: &Config) -> Result<u32, ClaudeError> {
    #[derive(serde::Deserialize)]
    struct ModelsResponse {
        data: Vec<ModelInfo>,
    }
    #[derive(serde::Deserialize)]
    struct ModelInfo {
        id: String,
        /// Anthropic uses `max_tokens`, `OpenRouter` uses `context_length`
        #[serde(default, alias = "max_tokens")]
        context_length: Option<u32>,
    }

    let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();
    let mut req = backend.get(&url).map_err(ClaudeError::Http)?;

    // Anthropic uses x-api-key header, proxies use Bearer token
    if cfg.base_url.contains("anthropic.com") {
        req = req
            .header("x-api-key", cfg.api_key.clone())
            .map_err(ClaudeError::Http)?
            .header("anthropic-version", ANTHROPIC_VERSION)
            .map_err(ClaudeError::Http)?;
    } else {
        req = req
            .header(
                header::AUTHORIZATION.as_str(),
                format!("Bearer {}", cfg.api_key),
            )
            .map_err(ClaudeError::Http)?;
    }

    let response: ModelsResponse = req.json().await.map_err(ClaudeError::Http)?;

    for model in response.data {
        if model.id == cfg.model {
            if let Some(ctx) = model.context_length {
                return Ok(ctx);
            }
        }
    }

    Err(ClaudeError::Api(format!(
        "Model '{}' not found or missing context_length",
        cfg.model
    )))
}

/// Builder for Claude clients.
#[derive(Debug)]
pub struct Builder {
    api_key: String,
    base_url: String,
    model: String,
    default_max_tokens: u32,
    native_abilities: Vec<Ability>,
}

impl Builder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CLAUDE_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            native_abilities: Vec::new(),
        }
    }

    /// Set a custom API base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Select a model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = sanitize_model(model);
        self
    }

    /// Set the default `max_tokens` for requests.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Declare extra native capabilities supported by the model.
    #[must_use]
    pub fn native_capabilities(mut self, abilities: impl IntoIterator<Item = Ability>) -> Self {
        for ability in abilities {
            if !self.native_abilities.contains(&ability) {
                self.native_abilities.push(ability);
            }
        }
        self
    }

    /// Mark this model as having built-in PDF understanding.
    #[must_use]
    pub fn enable_native_pdf(self) -> Self {
        self.native_capabilities([Ability::Pdf])
    }

    /// Consume the builder and create a Claude client.
    #[must_use]
    pub fn build(self) -> Claude {
        Claude {
            inner: Arc::new(Config {
                api_key: self.api_key,
                base_url: self.base_url,
                model: self.model,
                default_max_tokens: self.default_max_tokens,
                native_abilities: self.native_abilities,
            }),
        }
    }
}

/// Internal configuration for the Claude client.
#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) default_max_tokens: u32,
    pub(crate) native_abilities: Vec<Ability>,
}

impl Config {
    pub(crate) fn request_url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}

fn sanitize_model(model: impl Into<String>) -> String {
    model.into().trim().to_string()
}
