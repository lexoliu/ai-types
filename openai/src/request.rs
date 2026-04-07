use aither_core::llm::{
    Message, Role,
    model::{OpenAIPromptCacheRetention, Parameters, ReasoningEffort, ToolChoice},
    tool::ToolDefinition,
};
use url::Url;

use schemars::Schema;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use async_fs;

use crate::attachments::parse_openai_file_url;
use crate::error::OpenAIError;
#[derive(Clone)]
pub struct ParameterSnapshot {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) logit_bias: Option<HashMap<String, f32>>,
    pub(crate) seed: Option<u32>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u8>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) include_reasoning: bool,
    pub(crate) structured_outputs: bool,
    pub(crate) response_format: Option<Schema>,
    pub(crate) websearch: bool,
    pub(crate) code_execution: bool,
    pub(crate) legacy_max_tokens: bool,
    pub(crate) prompt_cache_key: Option<String>,
    pub(crate) prompt_cache_retention: Option<OpenAIPromptCacheRetention>,
}

impl From<&Parameters> for ParameterSnapshot {
    fn from(value: &Parameters) -> Self {
        Self {
            temperature: value.temperature,
            top_p: value.top_p,
            max_tokens: value.max_tokens,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop: value.stop.clone(),
            logit_bias: value
                .logit_bias
                .as_ref()
                .map(|pairs| pairs.iter().cloned().collect()),
            seed: value.seed,
            tool_choice: value.tool_choice.clone(),
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
            reasoning_effort: value.reasoning_effort,
            include_reasoning: value.include_reasoning,
            structured_outputs: value.structured_outputs,
            response_format: value.response_format.clone(),
            websearch: value.websearch,
            code_execution: value.code_execution,
            legacy_max_tokens: false,
            prompt_cache_key: value
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.key.clone()),
            prompt_cache_retention: value
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.retention),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
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
    #[serde(rename = "max_completion_tokens")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoicePayload>,
    /// Enable parallel tool calls (default: true when tools provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl ChatCompletionRequest {
    pub(crate) fn new(
        model: String,
        messages: Vec<ChatMessagePayload>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ToolPayload>>,
        stream: bool,
    ) -> Self {
        let has_tools = tools.as_ref().is_some_and(|t| !t.is_empty());
        Self {
            model,
            messages,
            stream,
            stream_options: stream.then_some(StreamOptions {
                include_usage: true,
            }),
            temperature: params.temperature,
            top_p: params.top_p,
            max_completion_tokens: params.max_tokens,
            max_tokens: if params.legacy_max_tokens {
                params.max_tokens
            } else {
                None
            },
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            stop: params.stop.clone(),
            logit_bias: params.logit_bias.clone(),
            seed: params.seed,
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice: tool_choice(params, has_tools),
            parallel_tool_calls: if has_tools { Some(false) } else { None },
            response_format: response_format(params),
            reasoning: reasoning(params),
            prompt_cache_key: params.prompt_cache_key.clone(),
            prompt_cache_retention: prompt_cache_retention(params),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessagePayload {
    role: &'static str,
    content: ContentPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Message content - either simple string or array of content parts (for vision).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ContentPayload {
    /// Simple text content.
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

/// Content part for multimodal messages.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content part.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content part.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPayload },
}

/// Image URL payload for vision.
#[derive(Debug, Clone, Serialize)]
pub struct ImageUrlPayload {
    /// URL to the image (can be data URL with base64).
    url: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolPayload {
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
pub struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormatPayload {
    JsonSchema { json_schema: JsonSchemaPayload },
    JsonObject,
}

#[derive(Debug, Serialize)]
struct JsonSchemaPayload {
    name: String,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ReasoningPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
}

pub async fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessagePayload> {
    let mut payloads = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.role() {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };

        let tool_call_id = message.tool_call_id().map(String::from);
        let tool_calls = if message.tool_calls().is_empty() {
            None
        } else {
            Some(
                message
                    .tool_calls()
                    .iter()
                    .map(|tc| ChatToolCallPayload {
                        id: tc.id.clone(),
                        kind: "function",
                        function: ChatToolFunctionPayload {
                            name: tc.name.clone(),
                            arguments: tc.arguments.to_string(),
                        },
                    })
                    .collect(),
            )
        };

        let content = build_content(message).await;
        payloads.push(ChatMessagePayload {
            role,
            content,
            tool_calls,
            tool_call_id,
        });
    }
    payloads
}

/// Build content payload for a message.
///
/// Returns simple text for messages without attachments,
/// or multimodal content parts for messages with attachments.
async fn build_content(message: &Message) -> ContentPayload {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return ContentPayload::Text(message.content().to_owned());
    }

    let mut parts = Vec::new();

    // Add image parts first
    for attachment in attachments {
        if let Some(data_url) = url_to_data_url(attachment).await {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrlPayload { url: data_url },
            });
        }
    }

    // Add text content
    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    ContentPayload::Parts(parts)
}

/// Flatten message content to a simple string.
///
/// For non-vision contexts (like Responses API), just returns the text content.
fn flatten_content(message: &Message) -> String {
    message.content().to_owned()
}

/// Convert a URL to a data URL suitable for `OpenAI` vision.
///
/// Handles:
/// - `data:...` URLs - passed through as-is
/// - `file:///path` URLs - reads file and converts to base64 data URL
/// - HTTP/HTTPS URLs - passed through as-is (`OpenAI` can fetch them)
async fn url_to_data_url(url: &url::Url) -> Option<String> {
    match url.scheme() {
        "data" => Some(url.as_str().to_string()),
        "http" | "https" => Some(url.as_str().to_string()),
        "file" => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                read_file_to_data_url(url).await
            }
            #[cfg(target_arch = "wasm32")]
            {
                tracing::warn!("file:// attachments are not supported on wasm32");
                None
            }
        }
        _ => {
            tracing::warn!("Unsupported attachment URL scheme: {}", url.scheme());
            None
        }
    }
}

/// Read a file:// URL and convert to a data URL.
#[cfg(not(target_arch = "wasm32"))]
async fn read_file_to_data_url(url: &url::Url) -> Option<String> {
    use base64::Engine;

    let path = url.to_file_path().ok()?;
    let data = async_fs::read(&path).await.ok()?;
    let mime_type = mime_from_path(&path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    Some(format!("data:{mime_type};base64,{base64_data}"))
}

#[cfg(target_arch = "wasm32")]
fn read_file_to_data_url(_url: &url::Url) -> Option<String> {
    None
}

/// Get MIME type from file path extension.
fn mime_from_path(path: &std::path::Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase()
        .as_str()
    {
        // Images
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        // Video (for providers that support it)
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        // Audio (for providers that support it)
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        // Documents
        "pdf" => Some("application/pdf"),
        _ => None,
    }
}

pub fn convert_tools(definitions: Vec<ToolDefinition>) -> Vec<ToolPayload> {
    definitions
        .into_iter()
        .map(|tool| ToolPayload {
            r#type: "function",
            function: ToolFunction {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.arguments_openai_schema(),
            },
        })
        .collect()
}

fn schema_to_value(schema: &Schema) -> Value {
    serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Map::new()))
}

fn tool_choice(params: &ParameterSnapshot, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => Some(ToolChoicePayload::Mode("auto")),
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn response_format(params: &ParameterSnapshot) -> Option<ResponseFormatPayload> {
    params
        .response_format
        .as_ref()
        .map(|schema| ResponseFormatPayload::JsonSchema {
            json_schema: JsonSchemaPayload {
                name: "aither.response".into(),
                schema: schema_to_value(schema),
                strict: Some(params.structured_outputs),
            },
        })
        .or_else(|| {
            if params.structured_outputs {
                Some(ResponseFormatPayload::JsonObject)
            } else {
                None
            }
        })
}

fn reasoning(params: &ParameterSnapshot) -> Option<ReasoningPayload> {
    params.reasoning_effort.map(|effort| ReasoningPayload {
        effort: Some(effort.as_str()),
    })
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatToolCallPayload {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: ChatToolFunctionPayload,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatToolFunctionPayload {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[allow(dead_code)]
impl ChatMessagePayload {
    pub(crate) const fn tool_output(call_id: String, output: String) -> Self {
        Self {
            role: "tool",
            content: ContentPayload::Text(output),
            tool_calls: None,
            tool_call_id: Some(call_id),
        }
    }

    pub(crate) const fn assistant_tool_calls(
        content: String,
        tool_calls: Vec<ChatToolCallPayload>,
    ) -> Self {
        Self {
            role: "assistant",
            content: ContentPayload::Text(content),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesInputItem {
    Message {
        role: String,
        content: ResponsesMessageContent,
    },
    FunctionCall {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesMessageContent {
    Text(String),
    Parts(Vec<ResponsesInputContent>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputContent {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(flatten)]
        source: InputImageSource,
    },
    InputFile {
        file_id: String,
    },
}

#[derive(Debug, Serialize, Clone)]
struct InputImageSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
}

impl InputImageSource {
    const fn from_url(url: String) -> Self {
        Self {
            image_url: Some(url),
            file_id: None,
        }
    }

    const fn from_file_id(file_id: String) -> Self {
        Self {
            image_url: None,
            file_id: Some(file_id),
        }
    }
}

#[allow(dead_code)]
impl ResponsesInputItem {
    pub(crate) fn message(role: impl Into<String>, content: ResponsesMessageContent) -> Self {
        Self::Message {
            role: role.into(),
            content,
        }
    }

    pub(crate) fn function_call_output(call_id: impl Into<String>, output: String) -> Self {
        Self::FunctionCallOutput {
            kind: "function_call_output",
            call_id: call_id.into(),
            output,
        }
    }
}

fn require_non_empty_call_id<'a>(
    call_id: Option<&'a str>,
    message_kind: &str,
) -> Result<&'a str, OpenAIError> {
    let Some(call_id) = call_id else {
        return Err(OpenAIError::Api(format!(
            "{message_kind} is missing tool_call_id"
        )));
    };
    if call_id.trim().is_empty() {
        return Err(OpenAIError::Api(format!(
            "{message_kind} has an empty tool_call_id"
        )));
    }
    Ok(call_id)
}

#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

impl ResponsesRequest {
    pub(crate) fn new(
        model: String,
        input: Vec<ResponsesInputItem>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ResponsesTool>>,
        tool_choice: Option<ResponsesToolChoice>,
        stream: bool,
    ) -> Self {
        let has_tools = tools.as_ref().is_some_and(|items| !items.is_empty());
        Self {
            model,
            input,
            stream,
            temperature: params.temperature,
            top_p: params.top_p,
            max_output_tokens: params.max_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            logit_bias: params.logit_bias.clone(),
            seed: params.seed,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice,
            parallel_tool_calls: if has_tools { Some(false) } else { None },
            text: responses_text(params),
            reasoning: reasoning(params),
            include: responses_include(params),
            prompt_cache_key: params.prompt_cache_key.clone(),
            prompt_cache_retention: prompt_cache_retention(params),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseTextConfig {
    format: ResponseTextFormat,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    JsonSchema {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    JsonObject,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function {
        name: String,
        description: String,
        parameters: Value,
    },
    WebSearch,
    CodeInterpreter,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
    },
}

pub fn to_responses_input(messages: &[Message]) -> Result<Vec<ResponsesInputItem>, OpenAIError> {
    let mut items = Vec::new();

    for message in messages {
        match message.role() {
            Role::User => {
                let attachments = message.attachments();
                if attachments.is_empty() {
                    items.push(ResponsesInputItem::message(
                        "user",
                        ResponsesMessageContent::Text(flatten_content(message)),
                    ));
                } else {
                    let mut parts = Vec::new();
                    for attachment in attachments {
                        parts.push(attachment_to_responses_part(attachment)?);
                    }
                    if !message.content().is_empty() {
                        parts.push(ResponsesInputContent::InputText {
                            text: message.content().to_owned(),
                        });
                    }
                    items.push(ResponsesInputItem::message(
                        "user",
                        ResponsesMessageContent::Parts(parts),
                    ));
                }
            }
            Role::System => {
                items.push(ResponsesInputItem::message(
                    "developer",
                    ResponsesMessageContent::Text(flatten_content(message)),
                ));
            }
            Role::Assistant => {
                let tool_calls = message.tool_calls();
                if tool_calls.is_empty() {
                    // Regular text response
                    items.push(ResponsesInputItem::message(
                        "assistant",
                        ResponsesMessageContent::Text(flatten_content(message)),
                    ));
                } else {
                    // Assistant message with function calls
                    // First add text content if present
                    if !message.content().is_empty() {
                        items.push(ResponsesInputItem::message(
                            "assistant",
                            ResponsesMessageContent::Text(message.content().to_string()),
                        ));
                    }
                    // Add function call items
                    for tc in tool_calls {
                        let call_id =
                            require_non_empty_call_id(Some(tc.id.as_str()), "assistant tool call")?;
                        items.push(ResponsesInputItem::FunctionCall {
                            kind: "function_call",
                            call_id: call_id.to_string(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.to_string(),
                        });
                    }
                }
            }
            Role::Tool => {
                // Tool results must be sent as FunctionCallOutput
                let call_id = require_non_empty_call_id(message.tool_call_id(), "tool result")?;
                items.push(ResponsesInputItem::function_call_output(
                    call_id,
                    message.content().to_string(),
                ));
            }
        }
    }

    validate_responses_input(&items)?;
    Ok(items)
}

fn validate_responses_input(items: &[ResponsesInputItem]) -> Result<(), OpenAIError> {
    for (index, item) in items.iter().enumerate() {
        match item {
            ResponsesInputItem::FunctionCall { call_id, name, .. } => {
                if call_id.trim().is_empty() {
                    return Err(OpenAIError::Api(format!(
                        "responses input[{index}] function_call '{name}' has an empty call_id"
                    )));
                }
            }
            ResponsesInputItem::FunctionCallOutput { call_id, .. } => {
                if call_id.trim().is_empty() {
                    return Err(OpenAIError::Api(format!(
                        "responses input[{index}] function_call_output has an empty call_id"
                    )));
                }
            }
            ResponsesInputItem::Message { .. } => {}
        }
    }
    Ok(())
}

fn attachment_to_responses_part(url: &Url) -> Result<ResponsesInputContent, OpenAIError> {
    if let Some((kind, id)) = parse_openai_file_url(url) {
        if kind.is_image() {
            return Ok(ResponsesInputContent::InputImage {
                source: InputImageSource::from_file_id(id),
            });
        }
        return Ok(ResponsesInputContent::InputFile { file_id: id });
    }

    match url.scheme() {
        "http" | "https" | "data" => Ok(ResponsesInputContent::InputImage {
            source: InputImageSource::from_url(url.as_str().to_string()),
        }),
        "file" => Err(OpenAIError::Api(
            "file:// attachments must be uploaded via Files API".to_string(),
        )),
        other => Err(OpenAIError::Api(format!(
            "Unsupported attachment URL scheme: {other}"
        ))),
    }
}

pub fn convert_responses_tools(definitions: Vec<ToolDefinition>) -> Vec<ResponsesTool> {
    definitions
        .into_iter()
        .map(|tool| ResponsesTool::Function {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.arguments_openai_schema(),
        })
        .collect()
}

pub fn responses_tool_choice(
    params: &ParameterSnapshot,
    has_tools: bool,
) -> Option<ResponsesToolChoice> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(ResponsesToolChoice::Mode("none")),
        ToolChoice::Required => Some(ResponsesToolChoice::Mode("required")),
        ToolChoice::Exact(name) => Some(ResponsesToolChoice::Function {
            kind: "function",
            name: name.clone(),
        }),
    }
}

fn responses_text(params: &ParameterSnapshot) -> Option<ResponseTextConfig> {
    params
        .response_format
        .as_ref()
        .map(|schema| ResponseTextConfig {
            format: ResponseTextFormat::JsonSchema {
                name: Some("aither.response".into()),
                schema: schema_to_value(schema),
                strict: Some(params.structured_outputs),
            },
        })
        .or_else(|| {
            if params.structured_outputs {
                Some(ResponseTextConfig {
                    format: ResponseTextFormat::JsonObject,
                })
            } else {
                None
            }
        })
}

fn responses_include(params: &ParameterSnapshot) -> Option<Vec<&'static str>> {
    let mut include = Vec::new();
    if params.logprobs.unwrap_or(false) {
        include.push("message.output_text.logprobs");
    }
    if params.include_reasoning {
        include.push("reasoning.encrypted_content");
    }
    if include.is_empty() {
        None
    } else {
        Some(include)
    }
}

fn prompt_cache_retention(params: &ParameterSnapshot) -> Option<&'static str> {
    params
        .prompt_cache_retention
        .map(OpenAIPromptCacheRetention::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::{OpenAIPromptCacheRetention, Parameters, ToolChoice};
    use aither_core::llm::{Message, ToolCall};

    #[test]
    fn chat_json_object_when_structured_outputs_without_schema() {
        let params = Parameters {
            structured_outputs: true,
            ..Parameters::default()
        };
        let snapshot = ParameterSnapshot::from(&params);
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, false);
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value["response_format"]["type"], "json_object");
    }

    #[test]
    fn responses_json_object_when_structured_outputs_without_schema() {
        let params = Parameters {
            structured_outputs: true,
            ..Parameters::default()
        };
        let snapshot = ParameterSnapshot::from(&params);
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            None,
            responses_tool_choice(&snapshot, false),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["text"]["format"]["type"], "json_object");
    }

    #[test]
    fn chat_stream_request_includes_usage_option() {
        let snapshot = ParameterSnapshot::from(&Parameters::default());
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, true);
        let value = serde_json::to_value(&req).expect("serialize stream chat request");
        assert_eq!(value["stream_options"]["include_usage"], true);
    }

    #[test]
    fn chat_request_disables_parallel_tool_calls_when_tools_exist() {
        let snapshot = ParameterSnapshot::from(&Parameters::default());
        let req = ChatCompletionRequest::new(
            "gpt-5".into(),
            Vec::new(),
            &snapshot,
            Some(vec![ToolPayload {
                r#type: "function",
                function: ToolFunction {
                    name: "lookup".to_string(),
                    description: "lookup data".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
            }]),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value["parallel_tool_calls"], false);
    }

    #[test]
    fn responses_request_disables_parallel_tool_calls_when_tools_exist() {
        let snapshot = ParameterSnapshot::from(&Parameters::default());
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            Some(vec![ResponsesTool::Function {
                name: "lookup".to_string(),
                description: "lookup data".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }]),
            responses_tool_choice(&snapshot, true),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["parallel_tool_calls"], false);
    }

    #[test]
    fn responses_tool_choice_exact_serializes_function_choice() {
        let params = Parameters::default().tool_choice(ToolChoice::Exact("lookup".to_string()));
        let snapshot = ParameterSnapshot::from(&params);
        let choice = responses_tool_choice(&snapshot, true).expect("tool choice should exist");
        let json = serde_json::to_value(choice).expect("serialize tool choice");
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "lookup");
    }

    #[test]
    fn responses_request_serializes_prompt_cache_fields() {
        let params = Parameters::default()
            .prompt_cache_key("session:alpha")
            .prompt_cache_retention(OpenAIPromptCacheRetention::Hours24);
        let snapshot = ParameterSnapshot::from(&params);
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            None,
            responses_tool_choice(&snapshot, false),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["prompt_cache_key"], "session:alpha");
        assert_eq!(value["prompt_cache_retention"], "24h");
    }

    #[test]
    fn chat_request_serializes_prompt_cache_fields() {
        let params = Parameters::default()
            .prompt_cache_key("session:beta")
            .prompt_cache_retention(OpenAIPromptCacheRetention::InMemory);
        let snapshot = ParameterSnapshot::from(&params);
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, false);
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value["prompt_cache_key"], "session:beta");
        assert_eq!(value["prompt_cache_retention"], "in-memory");
    }

    #[test]
    fn responses_input_rejects_empty_assistant_tool_call_id() {
        let messages = vec![Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new("", "lookup", serde_json::json!({"id": 1}))],
        )];
        let error = to_responses_input(&messages).expect_err("empty tool call id must fail");
        assert_eq!(
            error.to_string(),
            "assistant tool call has an empty tool_call_id"
        );
    }

    #[test]
    fn responses_input_rejects_empty_tool_result_call_id() {
        let messages = vec![Message::tool("", "done")];
        let error = to_responses_input(&messages).expect_err("empty tool result id must fail");
        assert_eq!(error.to_string(), "tool result has an empty tool_call_id");
    }

    #[test]
    fn responses_input_validation_rejects_empty_function_call_item() {
        let error = validate_responses_input(&[ResponsesInputItem::FunctionCall {
            kind: "function_call",
            call_id: String::new(),
            name: "lookup".to_string(),
            arguments: "{}".to_string(),
        }])
        .expect_err("empty function call id must fail");
        assert_eq!(
            error.to_string(),
            "responses input[0] function_call 'lookup' has an empty call_id"
        );
    }
}
