//! Request building and message conversion for the Claude API.

use aither_core::llm::{
    Message, Role,
    model::{
        ClaudeCacheBreakpointTarget, ClaudeExplicitCacheBreakpoints, ClaudePromptCache,
        ClaudePromptCacheStrategy, ClaudePromptCacheTtl, Parameters, ToolChoice,
    },
    tool::ToolDefinition,
};
use base64::Engine;
use serde::Serialize;
use serde_json::Value;

/// Claude Messages API request body.
#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    /// Model identifier.
    pub model: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Conversation messages.
    pub messages: Vec<MessagePayload>,
    /// System prompt (extracted from messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPayload>,
    /// Enable streaming.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Available tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolPayload>>,
    /// Tool choice policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoicePayload>,
    /// Prompt cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
}

/// Individual message in Claude format.
#[derive(Debug, Clone, Serialize)]
pub struct MessagePayload {
    /// Role: "user" or "assistant".
    pub role: &'static str,
    /// Message content.
    pub content: ContentPayload,
}

/// Message content - either a simple string or array of content blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ContentPayload {
    /// Simple text content.
    Text(String),
    /// Array of content blocks (for multimodal or tool results).
    Blocks(Vec<ContentBlock>),
}

/// System prompt payload in Claude format.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SystemPayload {
    /// Simple system text.
    Text(String),
    /// System content blocks.
    Blocks(Vec<SystemTextBlock>),
}

/// System text block.
#[derive(Debug, Clone, Serialize)]
pub struct SystemTextBlock {
    /// Claude text block kind.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Block text.
    pub text: String,
    /// Optional explicit cache control for this block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
}

/// Content block types for multimodal messages.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content block.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Image content block.
    #[serde(rename = "image")]
    Image {
        /// Image source (base64 or URL).
        source: ImageSource,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Tool use block (in assistant responses).
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: Value,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Tool result block (in user messages).
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the `tool_use` this is responding to.
        tool_use_id: String,
        /// Tool output content.
        content: String,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
}

/// Image source for vision requests.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    /// Base64-encoded image data.
    #[serde(rename = "base64")]
    Base64 {
        /// MIME type (image/jpeg, image/png, image/gif, image/webp).
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
    /// URL-referenced image.
    #[serde(rename = "url")]
    Url {
        /// Full URL to the image.
        url: String,
    },
}

/// Tool definition in Claude format.
#[derive(Debug, Clone, Serialize)]
pub struct ToolPayload {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON schema for tool input.
    pub input_schema: Value,
    /// Optional explicit cache control for this tool block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
}

/// Claude prompt cache control payload.
#[derive(Debug, Clone, Serialize)]
pub struct CacheControlPayload {
    /// Cache type.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Optional TTL override (`1h`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<&'static str>,
}

impl From<ClaudePromptCache> for CacheControlPayload {
    fn from(cache: ClaudePromptCache) -> Self {
        let ttl = match cache.ttl {
            ClaudePromptCacheTtl::FiveMinutes => None,
            ClaudePromptCacheTtl::OneHour => Some(ClaudePromptCacheTtl::OneHour.as_str()),
        };
        Self {
            kind: "ephemeral",
            ttl,
        }
    }
}

/// Tool choice payload for Claude Messages API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoicePayload {
    /// Let Claude decide when to call a tool.
    Auto,
    /// Require Claude to call at least one tool.
    Any,
    /// Restrict tool calling to a single tool by name.
    Tool {
        /// Name of the tool Claude is allowed to call.
        name: String,
    },
}

/// Snapshot of parameters for request building.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct ParameterSnapshot {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to include reasoning/thinking.
    pub include_reasoning: bool,
    /// Tool choice policy.
    pub tool_choice: ToolChoice,
    /// Claude-specific cache controls.
    pub cache: Option<ClaudePromptCache>,
}

impl From<&Parameters> for ParameterSnapshot {
    fn from(params: &Parameters) -> Self {
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            max_tokens: params.max_tokens,
            stop_sequences: params.stop.clone(),
            include_reasoning: params.include_reasoning,
            tool_choice: params.tool_choice.clone(),
            cache: params.cache.claude,
        }
    }
}

/// Convert aither messages to Claude format, extracting system messages.
///
/// Returns (`system_prompt`, messages). A single system message is emitted as
/// plain text, while multiple system messages are preserved as system blocks.
pub fn to_claude_messages(messages: &[Message]) -> (Option<SystemPayload>, Vec<MessagePayload>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut claude_messages: Vec<MessagePayload> = Vec::new();

    for message in messages {
        match message.role() {
            Role::System => {
                system_parts.push(message.content().to_string());
            }
            Role::User | Role::Tool => {
                let content = if matches!(message.role(), Role::Tool) {
                    build_tool_result_content(message)
                } else {
                    build_user_content(message)
                };
                claude_messages.push(MessagePayload {
                    role: "user",
                    content,
                });
            }
            Role::Assistant => {
                claude_messages.push(MessagePayload {
                    role: "assistant",
                    content: build_assistant_content(message),
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else if system_parts.len() == 1 {
        Some(SystemPayload::Text(system_parts.remove(0)))
    } else {
        Some(SystemPayload::Blocks(
            system_parts
                .into_iter()
                .map(|text| SystemTextBlock {
                    kind: "text",
                    text,
                    cache_control: None,
                })
                .collect(),
        ))
    };

    (system, claude_messages)
}

/// Build content for a user message, handling vision attachments.
fn build_user_content(message: &Message) -> ContentPayload {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return ContentPayload::Text(flatten_content(message));
    }

    let mut blocks: Vec<ContentBlock> = Vec::new();

    // Process image attachments
    for attachment in attachments {
        let url_str = attachment.as_str();
        if let Some(source) = parse_image_source(url_str) {
            blocks.push(ContentBlock::Image {
                source,
                cache_control: None,
            });
        }
    }

    // Add text content
    let text = flatten_content(message);
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text,
            cache_control: None,
        });
    }

    // Optimize: if only one text block, use simple string
    if blocks.len() == 1 {
        if let Some(ContentBlock::Text {
            text,
            cache_control: None,
        }) = blocks.pop()
        {
            return ContentPayload::Text(text);
        }
    }

    ContentPayload::Blocks(blocks)
}

fn build_assistant_content(message: &Message) -> ContentPayload {
    let tool_calls = message.tool_calls();
    if tool_calls.is_empty() {
        return ContentPayload::Text(flatten_content(message));
    }

    let mut blocks = Vec::new();
    let text = flatten_content(message);
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text,
            cache_control: None,
        });
    }

    for call in tool_calls {
        blocks.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
            cache_control: None,
        });
    }

    ContentPayload::Blocks(blocks)
}

fn build_tool_result_content(message: &Message) -> ContentPayload {
    let tool_use_id = message.tool_call_id().unwrap_or_else(|| {
        panic!("Tool message missing tool_call_id required by Claude tool_result payload")
    });
    let content = flatten_content(message);
    ContentPayload::Blocks(vec![ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content,
        cache_control: None,
    }])
}

/// Apply Claude cache strategy to request payload sections.
///
/// Returns top-level `cache_control` when automatic caching is selected.
/// Explicit mode mutates blocks in-place and returns `None`.
pub fn apply_cache_strategy(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    cache: ClaudePromptCache,
) -> Result<Option<CacheControlPayload>, String> {
    let default_ttl = cache.ttl;
    match cache.strategy {
        ClaudePromptCacheStrategy::Automatic => {
            let automatic =
                maybe_automatic_cache_control(system, messages, tools, default_ttl, None)?;
            Ok(automatic.map(|(cache_control, _)| cache_control))
        }
        ClaudePromptCacheStrategy::Explicit(breakpoints) => {
            let applied =
                apply_explicit_breakpoints(system, messages, tools, breakpoints, default_ttl)?;
            validate_mixed_ttl_ordering(&applied)?;
            Ok(None)
        }
        ClaudePromptCacheStrategy::AutomaticAndExplicit(breakpoints) => {
            let mut applied =
                apply_explicit_breakpoints(system, messages, tools, breakpoints, default_ttl)?;
            let automatic = maybe_automatic_cache_control(
                system,
                messages,
                tools,
                default_ttl,
                Some(breakpoints),
            )?;
            if let Some((cache_control, automatic_breakpoint)) = automatic {
                applied.push(automatic_breakpoint);
                validate_mixed_ttl_ordering(&applied)?;
                return Ok(Some(cache_control));
            }
            validate_mixed_ttl_ordering(&applied)?;
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PromptPosition {
    section: u8,
    major: usize,
    minor: usize,
}

impl PromptPosition {
    const fn tool(index: usize) -> Self {
        Self {
            section: 0,
            major: index,
            minor: 0,
        }
    }

    const fn system(index: usize) -> Self {
        Self {
            section: 1,
            major: index,
            minor: 0,
        }
    }

    const fn message(message_index: usize, block_index: usize) -> Self {
        Self {
            section: 2,
            major: message_index,
            minor: block_index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AppliedBreakpoint {
    position: PromptPosition,
    ttl: ClaudePromptCacheTtl,
}

fn apply_explicit_breakpoints(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    breakpoints: ClaudeExplicitCacheBreakpoints,
    default_ttl: ClaudePromptCacheTtl,
) -> Result<Vec<AppliedBreakpoint>, String> {
    let mut applied = Vec::new();
    for breakpoint in breakpoints.iter() {
        let ttl = breakpoint.effective_ttl(default_ttl);
        let cache_control = cache_control_payload_for_ttl(ttl);
        let position =
            mark_target_cache_control(system, messages, tools, breakpoint.target, &cache_control)?;
        applied.push(AppliedBreakpoint { position, ttl });
    }
    Ok(applied)
}

fn maybe_automatic_cache_control(
    system: &Option<SystemPayload>,
    messages: &[MessagePayload],
    tools: &Option<Vec<ToolPayload>>,
    ttl: ClaudePromptCacheTtl,
    explicit_breakpoints: Option<ClaudeExplicitCacheBreakpoints>,
) -> Result<Option<(CacheControlPayload, AppliedBreakpoint)>, String> {
    let Some((position, existing_control)) = find_last_cacheable_block(system, messages, tools)
    else {
        // No cacheable block exists: skip automatic caching as Anthropic does.
        return Ok(None);
    };

    if let Some(existing_control) = existing_control {
        if cache_control_matches_ttl(&existing_control, ttl) {
            // Anthropic behavior: automatic is a no-op when last block already has same TTL.
            return Ok(None);
        }
        return Err(
            "Claude automatic cache_control conflicts with explicit cache_control on the last cacheable block".to_string(),
        );
    }

    if let Some(explicit_breakpoints) = explicit_breakpoints {
        if explicit_breakpoints.is_full() {
            return Err(
                "Claude supports at most 4 cache breakpoints; automatic caching needs one free slot"
                    .to_string(),
            );
        }
    }

    let cache_control = cache_control_payload_for_ttl(ttl);
    Ok(Some((cache_control, AppliedBreakpoint { position, ttl })))
}

fn find_last_cacheable_block(
    system: &Option<SystemPayload>,
    messages: &[MessagePayload],
    tools: &Option<Vec<ToolPayload>>,
) -> Option<(PromptPosition, Option<CacheControlPayload>)> {
    for (message_index, message) in messages.iter().enumerate().rev() {
        match &message.content {
            ContentPayload::Text(text) => {
                if !text.is_empty() {
                    return Some((PromptPosition::message(message_index, 0), None));
                }
            }
            ContentPayload::Blocks(blocks) => {
                if let Some(block_index) = last_cacheable_block_index(blocks) {
                    return Some((
                        PromptPosition::message(message_index, block_index),
                        block_cache_control(&blocks[block_index]).clone(),
                    ));
                }
            }
        }
    }

    if let Some(system) = system {
        match system {
            SystemPayload::Text(text) => {
                if !text.is_empty() {
                    return Some((PromptPosition::system(0), None));
                }
            }
            SystemPayload::Blocks(blocks) => {
                if let Some(system_index) = blocks.iter().rposition(|block| !block.text.is_empty())
                {
                    return Some((
                        PromptPosition::system(system_index),
                        blocks[system_index].cache_control.clone(),
                    ));
                }
            }
        }
    }

    if let Some(tools) = tools {
        if let Some((tool_index, tool)) = tools.iter().enumerate().next_back() {
            return Some((PromptPosition::tool(tool_index), tool.cache_control.clone()));
        }
    }

    None
}

const fn cache_control_payload_for_ttl(ttl: ClaudePromptCacheTtl) -> CacheControlPayload {
    CacheControlPayload {
        kind: "ephemeral",
        ttl: match ttl {
            ClaudePromptCacheTtl::FiveMinutes => None,
            ClaudePromptCacheTtl::OneHour => Some(ClaudePromptCacheTtl::OneHour.as_str()),
        },
    }
}

fn cache_control_matches_ttl(
    cache_control: &CacheControlPayload,
    ttl: ClaudePromptCacheTtl,
) -> bool {
    let expected = cache_control_payload_for_ttl(ttl);
    cache_control.kind == expected.kind && cache_control.ttl == expected.ttl
}

fn validate_mixed_ttl_ordering(applied: &[AppliedBreakpoint]) -> Result<(), String> {
    if applied.is_empty() {
        return Ok(());
    }
    let mut ordered = applied.to_vec();
    ordered.sort_by_key(|breakpoint| breakpoint.position);
    let mut seen_five_minute = false;
    for breakpoint in ordered {
        match breakpoint.ttl {
            ClaudePromptCacheTtl::OneHour => {
                if seen_five_minute {
                    return Err(
                        "Claude mixed TTL ordering requires 1h cache breakpoints to appear before 5m breakpoints".to_string(),
                    );
                }
            }
            ClaudePromptCacheTtl::FiveMinutes => {
                seen_five_minute = true;
            }
        }
    }
    Ok(())
}

fn mark_target_cache_control(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    target: ClaudeCacheBreakpointTarget,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    match target {
        ClaudeCacheBreakpointTarget::LastTool => mark_last_tool_cache_control(tools, cache_control),
        ClaudeCacheBreakpointTarget::Tool(index) => {
            mark_tool_cache_control(tools, index, cache_control)
        }
        ClaudeCacheBreakpointTarget::LastSystem => {
            mark_last_system_cache_control(system, cache_control)
        }
        ClaudeCacheBreakpointTarget::System(index) => {
            mark_system_cache_control(system, index, cache_control)
        }
        ClaudeCacheBreakpointTarget::LastMessage => {
            mark_last_message_cache_control(messages, cache_control)
        }
        ClaudeCacheBreakpointTarget::Message {
            message_index,
            block_index,
        } => mark_message_cache_control(messages, message_index, block_index, cache_control),
    }
}

fn mark_last_tool_cache_control(
    tools: &mut Option<Vec<ToolPayload>>,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let tools = tools.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    let index = tools.len().checked_sub(1).ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    let tool = tools.get_mut(index).ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    set_cache_control(
        &mut tool.cache_control,
        cache_control,
        "tool breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::tool(index))
}

fn mark_tool_cache_control(
    tools: &mut Option<Vec<ToolPayload>>,
    index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let tools = tools.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    let tool = tools
        .get_mut(index)
        .ok_or_else(|| format!("Claude explicit tool breakpoint index {index} is out of range"))?;
    set_cache_control(
        &mut tool.cache_control,
        cache_control,
        "tool breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::tool(index))
}

fn mark_last_system_cache_control(
    system: &mut Option<SystemPayload>,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let system_payload = system.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for system but no system prompt exists"
            .to_string()
    })?;
    let blocks = ensure_system_blocks(system_payload);
    let index = blocks
        .iter()
        .rposition(|block| !block.text.is_empty())
        .ok_or_else(|| {
            "Claude explicit cache breakpoint requested for system but all system text blocks are empty"
                .to_string()
        })?;
    set_cache_control(
        &mut blocks[index].cache_control,
        cache_control,
        "system breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::system(index))
}

fn mark_system_cache_control(
    system: &mut Option<SystemPayload>,
    index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let system_payload = system.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for system but no system prompt exists"
            .to_string()
    })?;
    let blocks = ensure_system_blocks(system_payload);
    let block = blocks.get_mut(index).ok_or_else(|| {
        format!("Claude explicit system breakpoint index {index} is out of range")
    })?;
    if block.text.is_empty() {
        return Err(format!(
            "Claude explicit system breakpoint index {index} targets an empty text block"
        ));
    }
    set_cache_control(
        &mut block.cache_control,
        cache_control,
        "system breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::system(index))
}

fn ensure_system_blocks(system: &mut SystemPayload) -> &mut Vec<SystemTextBlock> {
    if let SystemPayload::Text(text) = system {
        *system = SystemPayload::Blocks(vec![SystemTextBlock {
            kind: "text",
            text: core::mem::take(text),
            cache_control: None,
        }]);
    }
    match system {
        SystemPayload::Blocks(blocks) => blocks,
        SystemPayload::Text(_) => panic!("System payload conversion to blocks failed"),
    }
}

fn mark_last_message_cache_control(
    messages: &mut [MessagePayload],
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    for (message_index, message) in messages.iter_mut().enumerate().rev() {
        match &mut message.content {
            ContentPayload::Text(text) => {
                if text.is_empty() {
                    continue;
                }
                let text = core::mem::take(text);
                message.content = ContentPayload::Blocks(vec![ContentBlock::Text {
                    text,
                    cache_control: Some(cache_control.clone()),
                }]);
                return Ok(PromptPosition::message(message_index, 0));
            }
            ContentPayload::Blocks(blocks) => {
                if let Some(index) = last_cacheable_block_index(blocks) {
                    set_cache_control(
                        block_cache_control_mut(&mut blocks[index]),
                        cache_control,
                        "message breakpoint already has conflicting cache_control",
                    )?;
                    return Ok(PromptPosition::message(message_index, index));
                }
            }
        }
    }
    Err("Claude explicit cache breakpoint requested for messages but no cacheable message block exists".to_string())
}

fn mark_message_cache_control(
    messages: &mut [MessagePayload],
    message_index: usize,
    block_index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let message = messages.get_mut(message_index).ok_or_else(|| {
        format!("Claude explicit message breakpoint message_index {message_index} is out of range")
    })?;
    if let ContentPayload::Text(text) = &mut message.content {
        if block_index != 0 {
            return Err(format!(
                "Claude explicit message breakpoint block_index {block_index} is invalid for text-only message at index {message_index}"
            ));
        }
        if text.is_empty() {
            return Err(format!(
                "Claude explicit message breakpoint targets empty text message at index {message_index}"
            ));
        }
        let text = core::mem::take(text);
        message.content = ContentPayload::Blocks(vec![ContentBlock::Text {
            text,
            cache_control: None,
        }]);
    }

    let blocks = match &mut message.content {
        ContentPayload::Blocks(blocks) => blocks,
        ContentPayload::Text(_) => panic!("message content conversion to blocks failed"),
    };
    let block = blocks.get_mut(block_index).ok_or_else(|| {
        format!(
            "Claude explicit message breakpoint block_index {block_index} is out of range for message index {message_index}"
        )
    })?;
    if !is_cacheable_block(block) {
        return Err(format!(
            "Claude explicit message breakpoint targets a non-cacheable block at message index {message_index}, block index {block_index}"
        ));
    }
    set_cache_control(
        block_cache_control_mut(block),
        cache_control,
        "message breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::message(message_index, block_index))
}

fn last_cacheable_block_index(blocks: &[ContentBlock]) -> Option<usize> {
    blocks.iter().rposition(|block| match block {
        ContentBlock::Text { text, .. } => !text.is_empty(),
        ContentBlock::Image { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. } => true,
    })
}

const fn is_cacheable_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { text, .. } => !text.is_empty(),
        ContentBlock::Image { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. } => true,
    }
}

const fn block_cache_control_mut(block: &mut ContentBlock) -> &mut Option<CacheControlPayload> {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => cache_control,
    }
}

const fn block_cache_control(block: &ContentBlock) -> &Option<CacheControlPayload> {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => cache_control,
    }
}

fn set_cache_control(
    slot: &mut Option<CacheControlPayload>,
    expected: &CacheControlPayload,
    mismatch_error: &str,
) -> Result<(), String> {
    if let Some(current) = slot {
        if current.kind != expected.kind || current.ttl != expected.ttl {
            return Err(mismatch_error.to_string());
        }
        return Ok(());
    }
    *slot = Some(expected.clone());
    Ok(())
}

/// Parse a URL string into an image source.
///
/// Handles:
/// - `data:image/...;base64,...` - already base64 encoded
/// - `file:///path/to/file` - reads file and converts to base64
/// - `http://` or `https://` URLs - passed through as URL source
fn parse_image_source(url: &str) -> Option<ImageSource> {
    if url.starts_with("data:image/") {
        // Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
        let after_data = url.strip_prefix("data:")?;
        let (header, data) = after_data.split_once(',')?;
        let media_type = header.strip_suffix(";base64")?;
        Some(ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        })
    } else if url.starts_with("file://") {
        // Read local file and convert to base64
        read_file_to_base64_source(url)
    } else if is_image_url(url) {
        Some(ImageSource::Url {
            url: url.to_string(),
        })
    } else {
        None
    }
}

/// Read a file:// URL and convert to base64 image source.
fn read_file_to_base64_source(file_url: &str) -> Option<ImageSource> {
    let url = url::Url::parse(file_url).ok()?;
    let path = url.to_file_path().ok()?;

    // Read the file
    let data = std::fs::read(&path).ok()?;

    // Determine media type from extension
    let media_type = mime_from_path(&path)?;

    // Encode to base64
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    Some(ImageSource::Base64 {
        media_type: media_type.to_string(),
        data: base64_data,
    })
}

/// Get MIME type from file path extension.
///
/// Supports images, video, audio, and PDFs.
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
        "heic" => Some("image/heic"),
        "heif" => Some("image/heif"),
        // Video
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        "mov" => Some("video/quicktime"),
        "avi" => Some("video/x-msvideo"),
        // Audio
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "ogg" => Some("audio/ogg"),
        "m4a" => Some("audio/mp4"),
        "flac" => Some("audio/flac"),
        // Documents
        "pdf" => Some("application/pdf"),
        _ => None,
    }
}

/// Check if a URL appears to be an image.
fn is_image_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".png")
        || lower.ends_with(".gif")
        || lower.ends_with(".webp")
        || lower.contains("/image")
}

/// Flatten message content.
fn flatten_content(message: &Message) -> String {
    message.content().to_owned()
}

/// Convert aither tool definitions to Claude format.
pub fn convert_tools(definitions: &[ToolDefinition]) -> Vec<ToolPayload> {
    definitions
        .iter()
        .map(|tool| ToolPayload {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.arguments_openai_schema(),
            cache_control: None,
        })
        .collect()
}

pub fn filter_tool_definitions(
    definitions: Vec<ToolDefinition>,
    choice: &ToolChoice,
) -> Vec<ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => definitions
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => definitions,
    }
}

pub fn tool_choice_payload(choice: &ToolChoice, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match choice {
        ToolChoice::None => None,
        ToolChoice::Auto => Some(ToolChoicePayload::Auto),
        ToolChoice::Required => Some(ToolChoicePayload::Any),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Tool { name: name.clone() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{
        ToolCall,
        model::{
            ClaudeCacheBreakpointTarget, ClaudeExplicitCacheBreakpoint,
            ClaudeExplicitCacheBreakpoints, ClaudePromptCache, ClaudePromptCacheStrategy,
            ClaudePromptCacheTtl, Parameters, ToolChoice,
        },
    };

    #[test]
    fn assistant_tool_calls_are_encoded_as_tool_use_blocks() {
        let messages = vec![Message::assistant_with_tool_calls(
            "Working on it",
            vec![ToolCall {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                arguments: serde_json::json!({"q":"rust"}),
            }],
        )];
        let (_, encoded) = to_claude_messages(&messages);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].role, "assistant");
        match &encoded[0].content {
            ContentPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(blocks[0], ContentBlock::Text { .. }));
                assert!(matches!(blocks[1], ContentBlock::ToolUse { .. }));
            }
            other => panic!("expected assistant blocks payload, got: {other:?}"),
        }
    }

    #[test]
    fn tool_message_is_encoded_as_tool_result_block() {
        let messages = vec![Message::tool("call_9", "{\"ok\":true}")];
        let (_, encoded) = to_claude_messages(&messages);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].role, "user");
        match &encoded[0].content {
            ContentPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        assert_eq!(tool_use_id, "call_9");
                        assert_eq!(content, "{\"ok\":true}");
                    }
                    other => panic!("expected tool_result block, got: {other:?}"),
                }
            }
            other => panic!("expected user blocks payload, got: {other:?}"),
        }
    }

    #[test]
    fn required_tool_choice_maps_to_any() {
        let payload = tool_choice_payload(&ToolChoice::Required, true)
            .expect("required should create payload");
        let json = serde_json::to_value(payload).expect("serialize tool choice");
        assert_eq!(json["type"], "any");
    }

    #[test]
    fn exact_tool_choice_maps_to_named_tool() {
        let payload = tool_choice_payload(&ToolChoice::Exact("search".to_string()), true)
            .expect("exact should create payload");
        let json = serde_json::to_value(payload).expect("serialize tool choice");
        assert_eq!(json["type"], "tool");
        assert_eq!(json["name"], "search");
    }

    #[test]
    fn parameter_snapshot_preserves_claude_cache_setting() {
        let params = Parameters::default()
            .claude_prompt_cache(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour));
        let snapshot = ParameterSnapshot::from(&params);
        assert_eq!(
            snapshot.cache,
            Some(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour))
        );
    }

    #[test]
    fn cache_control_payload_serializes_expected_shape() {
        let one_hour =
            CacheControlPayload::from(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour));
        let one_hour_json = serde_json::to_value(one_hour).expect("serialize one hour payload");
        assert_eq!(one_hour_json["type"], "ephemeral");
        assert_eq!(one_hour_json["ttl"], "1h");

        let short =
            CacheControlPayload::from(ClaudePromptCache::new(ClaudePromptCacheTtl::FiveMinutes));
        let short_json = serde_json::to_value(short).expect("serialize five minute payload");
        assert_eq!(short_json["type"], "ephemeral");
        assert!(short_json.get("ttl").is_none());
    }

    #[test]
    fn multiple_system_messages_are_preserved_as_system_blocks() {
        let messages = vec![
            Message::system("Instruction A"),
            Message::system("Instruction B"),
            Message::user("Hello"),
        ];
        let (system, encoded) = to_claude_messages(&messages);
        assert_eq!(encoded.len(), 1);
        let system = system.expect("system payload should exist");
        match system {
            SystemPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].text, "Instruction A");
                assert_eq!(blocks[1].text, "Instruction B");
            }
            other => panic!("expected system blocks payload, got: {other:?}"),
        }
    }

    #[test]
    fn automatic_cache_strategy_returns_top_level_cache_control() {
        let mut system = Some(SystemPayload::Text("You are helpful".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Hello".to_string()),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::automatic(ClaudePromptCacheTtl::FiveMinutes);
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("automatic cache strategy should succeed");
        let top_level = top_level.expect("automatic strategy should return top-level cache");
        assert_eq!(top_level.kind, "ephemeral");
        assert!(top_level.ttl.is_none());
    }

    #[test]
    fn explicit_message_breakpoint_marks_last_message_block() {
        let mut system = None;
        let mut messages = vec![MessagePayload {
            role: "assistant",
            content: ContentPayload::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "search".to_string(),
                input: serde_json::json!({"q":"mars"}),
                cache_control: None,
            }]),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour).with_strategy(
            ClaudePromptCacheStrategy::Explicit(ClaudeExplicitCacheBreakpoints::messages_only()),
        );
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("explicit strategy should succeed");
        assert!(top_level.is_none());

        match &messages[0].content {
            ContentPayload::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolUse { cache_control, .. } => {
                    let cache_control =
                        cache_control.as_ref().expect("cache control should be set");
                    assert_eq!(cache_control.kind, "ephemeral");
                    assert_eq!(cache_control.ttl, Some("1h"));
                }
                other => panic!("expected tool_use block, got: {other:?}"),
            },
            other => panic!("expected block payload, got: {other:?}"),
        }
    }

    #[test]
    fn explicit_breakpoint_target_out_of_range_fails_fast() {
        let mut system = None;
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("hello".to_string()),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::explicit(
            ClaudePromptCacheTtl::FiveMinutes,
            ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                ClaudeCacheBreakpointTarget::Message {
                    message_index: 0,
                    block_index: 1,
                },
            )),
        );
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("explicit strategy with out-of-range block index must fail");
        assert!(err.contains("block_index"));
    }

    #[test]
    fn automatic_and_explicit_strategy_sets_both_levels() {
        let mut system = Some(SystemPayload::Text("You are helpful".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Tell me about Mars".to_string()),
        }];
        let mut tools = Some(vec![ToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        }]);

        let cache =
            ClaudePromptCache::automatic_with_explicit(ClaudePromptCacheTtl::FiveMinutes, {
                ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                    ClaudeCacheBreakpointTarget::LastTool,
                ))
                .with_second(ClaudeExplicitCacheBreakpoint::new(
                    ClaudeCacheBreakpointTarget::LastSystem,
                ))
            });
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("combined strategy should succeed");
        assert!(top_level.is_some());

        let tools = tools.expect("tools should stay present");
        assert!(
            tools
                .last()
                .and_then(|tool| tool.cache_control.as_ref())
                .is_some()
        );
    }

    #[test]
    fn mixed_ttl_requires_one_hour_before_five_minutes() {
        let mut system = Some(SystemPayload::Text("System".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Hello".to_string()),
        }];
        let mut tools = Some(vec![ToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        }]);

        let breakpoints = ClaudeExplicitCacheBreakpoints::new(
            ClaudeExplicitCacheBreakpoint::new(ClaudeCacheBreakpointTarget::LastTool)
                .with_ttl(ClaudePromptCacheTtl::FiveMinutes),
        )
        .with_second(
            ClaudeExplicitCacheBreakpoint::new(ClaudeCacheBreakpointTarget::LastMessage)
                .with_ttl(ClaudePromptCacheTtl::OneHour),
        );
        let cache = ClaudePromptCache::explicit(ClaudePromptCacheTtl::FiveMinutes, breakpoints);
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("5m before 1h should fail mixed TTL ordering validation");
        assert!(err.contains("1h"));
    }

    #[test]
    fn automatic_with_full_explicit_slots_fails_when_no_overlap() {
        let mut system = Some(SystemPayload::Blocks(vec![
            SystemTextBlock {
                kind: "text",
                text: "S0".to_string(),
                cache_control: None,
            },
            SystemTextBlock {
                kind: "text",
                text: "S1".to_string(),
                cache_control: None,
            },
        ]));
        let mut messages = vec![
            MessagePayload {
                role: "assistant",
                content: ContentPayload::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q":"mars"}),
                    cache_control: None,
                }]),
            },
            MessagePayload {
                role: "user",
                content: ContentPayload::Text("Final user message".to_string()),
            },
        ];
        let mut tools = Some(vec![ToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        }]);

        let breakpoints = ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::Tool(0),
        ))
        .with_second(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::System(0),
        ))
        .with_third(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::System(1),
        ))
        .with_fourth(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::Message {
                message_index: 0,
                block_index: 0,
            },
        ));
        let cache = ClaudePromptCache::automatic_with_explicit(
            ClaudePromptCacheTtl::FiveMinutes,
            breakpoints,
        );
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("automatic caching should fail when all 4 explicit slots are occupied");
        assert!(err.contains("at most 4 cache breakpoints"));
    }
}
