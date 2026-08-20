//! Shared conversion logic for `LiteLLM` and `OpenRouter` JSON.
//!
//! This module is included via `#[path]` in build.rs so the same parsing
//! logic runs at compile time and runtime. It depends only on `serde`,
//! `serde_json`, and sibling types.

use serde::Deserialize;

/// Raw `LiteLLM` entry as it appears in the JSON.
#[derive(Debug, Deserialize)]
pub struct RawLiteLLMEntry {
    #[serde(default)]
    pub litellm_provider: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub input_cost_per_token: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token: Option<f64>,
    #[serde(default)]
    pub cache_read_input_token_cost: Option<f64>,
    #[serde(default)]
    pub cache_creation_input_token_cost: Option<f64>,
    #[serde(default)]
    pub output_cost_per_reasoning_token: Option<f64>,
    #[serde(default)]
    pub input_cost_per_audio_token: Option<f64>,
    #[serde(default)]
    pub deprecation_date: Option<String>,
    // supports_* flags
    #[serde(default)]
    pub supports_function_calling: Option<bool>,
    #[serde(default)]
    pub supports_vision: Option<bool>,
    #[serde(default)]
    pub supports_audio_input: Option<bool>,
    #[serde(default)]
    pub supports_audio_output: Option<bool>,
    #[serde(default)]
    pub supports_video_input: Option<bool>,
    #[serde(default)]
    pub supports_web_search: Option<bool>,
    #[serde(default)]
    pub supports_pdf_input: Option<bool>,
    #[serde(default)]
    pub supports_reasoning: Option<bool>,
    #[serde(default)]
    pub supports_computer_use: Option<bool>,
    #[serde(default)]
    pub supports_prompt_caching: Option<bool>,
    #[serde(default)]
    pub supports_assistant_prefill: Option<bool>,
}

/// Known provider prefix → canonical provider mapping.
const PROVIDER_PREFIXES: &[(&str, &str)] = &[
    ("openai/", "openai"),
    ("anthropic/", "anthropic"),
    ("gemini/", "gemini"),
    ("deepseek/", "deepseek"),
    ("xai/", "xai"),
    ("mistral/", "mistral"),
    ("meta-llama/", "meta"),
    ("bedrock/", "bedrock"),
    ("azure/", "azure"),
    ("azure_ai/", "azure"),
    ("vertex_ai/", "vertex_ai"),
    ("groq/", "groq"),
    ("together_ai/", "together_ai"),
    ("fireworks_ai/", "fireworks_ai"),
    ("replicate/", "replicate"),
    ("cohere/", "cohere"),
    ("cohere_chat/", "cohere"),
    ("perplexity/", "perplexity"),
    ("github_copilot/", "github_copilot"),
    ("ollama/", "ollama"),
    ("nvidia_nim/", "nvidia_nim"),
    ("sambanova/", "sambanova"),
    ("cerebras/", "cerebras"),
    ("dashscope/", "alibaba"),
];

/// Strip known provider prefixes to get the canonical model ID.
#[must_use]
pub fn strip_provider_prefix(litellm_key: &str) -> &str {
    for &(prefix, _) in PROVIDER_PREFIXES {
        if let Some(rest) = litellm_key.strip_prefix(prefix) {
            return rest;
        }
    }
    litellm_key
}

/// Map a `LiteLLM` `litellm_provider` string to a provider tag.
#[must_use]
pub fn parse_provider(provider_str: &str) -> String {
    let mapped = match provider_str {
        "openai" | "text-completion-openai" | "chatgpt" => "openai",
        "anthropic" => "anthropic",
        "gemini" | "palm" => "gemini",
        "deepseek" => "deepseek",
        "xai" => "xai",
        "mistral" | "codestral" | "text-completion-codestral" => "mistral",
        "meta_llama" => "meta",
        s if s.starts_with("bedrock") => "bedrock",
        "azure" | "azure_ai" | "azure_text" => "azure",
        s if s.starts_with("vertex_ai") => "vertex_ai",
        "groq" => "groq",
        "together_ai" => "together_ai",
        s if s.starts_with("fireworks_ai") => "fireworks_ai",
        "replicate" => "replicate",
        "cohere" | "cohere_chat" => "cohere",
        "perplexity" => "perplexity",
        "github_copilot" => "github_copilot",
        "dashscope" => "alibaba",
        other => return other.to_string(),
    };
    mapped.to_string()
}

/// Map a `LiteLLM` `mode` string to a mode tag.
#[must_use]
pub fn parse_mode(mode_str: &str) -> Option<&'static str> {
    match mode_str {
        "chat" | "responses" => Some("chat"),
        "embedding" => Some("embedding"),
        "image_generation" | "image_edit" => Some("image_generation"),
        "audio_transcription" => Some("audio_transcription"),
        "audio_speech" => Some("audio_speech"),
        "rerank" => Some("rerank"),
        "moderation" => Some("moderation"),
        "completion" => Some("completion"),
        "video_generation" => Some("video_generation"),
        "search" | "vector_store" => Some("search"),
        "ocr" => Some("ocr"),
        _ => None,
    }
}

/// Intermediate representation used by both build.rs and runtime.
#[derive(Debug, Clone)]
pub struct ConvertedEntry {
    pub litellm_id: String,
    pub canonical_id: String,
    pub provider: String,
    pub mode: String,
    pub max_input_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub input_cost_per_token: f64,
    pub output_cost_per_token: f64,
    pub cache_read_per_token: Option<f64>,
    pub cache_write_per_token: Option<f64>,
    pub reasoning_per_token: Option<f64>,
    pub image_per_token: Option<f64>,
    pub abilities: Vec<&'static str>,
    pub deprecation_date: Option<String>,
}

/// Convert a single `LiteLLM` JSON entry into our intermediate representation.
#[must_use]
pub fn convert_litellm_entry(key: &str, raw: &RawLiteLLMEntry) -> Option<ConvertedEntry> {
    let provider_str = raw.litellm_provider.as_deref()?;
    let mode_str = raw.mode.as_deref()?;
    let mode = parse_mode(mode_str)?;
    let provider = parse_provider(provider_str);
    let canonical_id = strip_provider_prefix(key).to_string();

    let max_input = raw.max_input_tokens;
    let max_output = raw.max_output_tokens.or(raw.max_tokens);

    let mut abilities = Vec::new();
    if raw.supports_function_calling == Some(true) {
        abilities.push("ToolUse");
    }
    if raw.supports_vision == Some(true) {
        abilities.push("Vision");
    }
    if raw.supports_audio_input == Some(true) {
        abilities.push("Audio");
    }
    if raw.supports_audio_output == Some(true) {
        abilities.push("AudioOutput");
    }
    if raw.supports_video_input == Some(true) {
        abilities.push("Video");
    }
    if raw.supports_web_search == Some(true) {
        abilities.push("WebSearch");
    }
    if raw.supports_pdf_input == Some(true) {
        abilities.push("Pdf");
    }
    if raw.supports_reasoning == Some(true) {
        abilities.push("Reasoning");
    }
    if raw.supports_computer_use == Some(true) {
        abilities.push("ComputerUse");
    }
    if raw.supports_prompt_caching == Some(true) {
        abilities.push("PromptCaching");
    }
    if raw.supports_assistant_prefill == Some(true) {
        abilities.push("AssistantPrefill");
    }
    if mode == "image_generation" {
        abilities.push("ImageGeneration");
    }

    Some(ConvertedEntry {
        litellm_id: key.to_string(),
        canonical_id,
        provider,
        mode: mode.to_string(),
        max_input_tokens: max_input,
        max_output_tokens: max_output,
        input_cost_per_token: raw.input_cost_per_token.unwrap_or(0.0),
        output_cost_per_token: raw.output_cost_per_token.unwrap_or(0.0),
        cache_read_per_token: raw.cache_read_input_token_cost,
        cache_write_per_token: raw.cache_creation_input_token_cost,
        reasoning_per_token: raw.output_cost_per_reasoning_token,
        image_per_token: raw.input_cost_per_audio_token,
        abilities,
        deprecation_date: raw.deprecation_date.clone(),
    })
}

/// Parse the full `LiteLLM` JSON blob and return converted entries.
#[must_use]
pub fn parse_litellm_json(json_bytes: &[u8]) -> Vec<ConvertedEntry> {
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(json_bytes).expect("invalid LiteLLM JSON");

    let mut entries = Vec::new();
    for (key, value) in &map {
        if key == "sample_spec" {
            continue;
        }
        let Ok(raw) = serde_json::from_value::<RawLiteLLMEntry>(value.clone()) else {
            continue;
        };
        if let Some(entry) = convert_litellm_entry(key, &raw) {
            entries.push(entry);
        }
    }
    entries
}
