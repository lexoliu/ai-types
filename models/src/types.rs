//! Owned model metadata types sourced from LiteLLM and OpenRouter.

use std::borrow::Cow;

use aither_core::llm::model::Ability;

/// The operational mode of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelMode {
    /// Chat / conversational completion.
    Chat,
    /// Text embedding / vectorization.
    Embedding,
    /// Image generation.
    ImageGeneration,
    /// Audio transcription (speech-to-text).
    AudioTranscription,
    /// Audio speech (text-to-speech).
    AudioSpeech,
    /// Reranking.
    Rerank,
    /// Content moderation.
    Moderation,
    /// Legacy text completion.
    Completion,
    /// Video generation.
    VideoGeneration,
    /// Search.
    Search,
    /// Optical character recognition.
    Ocr,
}

/// Known model providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Provider {
    /// OpenAI.
    OpenAI,
    /// Anthropic.
    Anthropic,
    /// Google (Gemini).
    Google,
    /// DeepSeek.
    DeepSeek,
    /// xAI (Grok).
    XAI,
    /// Mistral AI.
    Mistral,
    /// Meta (LLaMA).
    Meta,
    /// Alibaba (Qwen / DashScope).
    Alibaba,
    /// AWS Bedrock.
    Bedrock,
    /// Azure OpenAI.
    Azure,
    /// Google Vertex AI.
    VertexAI,
    /// Groq.
    Groq,
    /// Together AI.
    Together,
    /// Fireworks AI.
    Fireworks,
    /// Replicate.
    Replicate,
    /// Cohere.
    Cohere,
    /// Perplexity.
    Perplexity,
    /// GitHub Copilot.
    Copilot,
    /// Any other provider not explicitly listed.
    Other(Cow<'static, str>),
}

impl Provider {
    /// Returns the canonical string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "gemini",
            Self::DeepSeek => "deepseek",
            Self::XAI => "xai",
            Self::Mistral => "mistral",
            Self::Meta => "meta",
            Self::Alibaba => "alibaba",
            Self::Bedrock => "bedrock",
            Self::Azure => "azure",
            Self::VertexAI => "vertex_ai",
            Self::Groq => "groq",
            Self::Together => "together_ai",
            Self::Fireworks => "fireworks_ai",
            Self::Replicate => "replicate",
            Self::Cohere => "cohere",
            Self::Perplexity => "perplexity",
            Self::Copilot => "github_copilot",
            Self::Other(s) => s,
        }
    }
}

impl core::fmt::Display for Provider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-token pricing information (USD).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pricing {
    input_per_token: f64,
    output_per_token: f64,
    cache_read_per_token: Option<f64>,
    cache_write_per_token: Option<f64>,
    reasoning_per_token: Option<f64>,
    image_per_token: Option<f64>,
}

impl Pricing {
    /// Creates a new pricing entry.
    #[must_use]
    pub const fn new(input_per_token: f64, output_per_token: f64) -> Self {
        Self {
            input_per_token,
            output_per_token,
            cache_read_per_token: None,
            cache_write_per_token: None,
            reasoning_per_token: None,
            image_per_token: None,
        }
    }

    /// Input cost per token.
    #[must_use]
    pub const fn input_per_token(&self) -> f64 {
        self.input_per_token
    }

    /// Output cost per token.
    #[must_use]
    pub const fn output_per_token(&self) -> f64 {
        self.output_per_token
    }

    /// Cache read cost per token, if applicable.
    #[must_use]
    pub const fn cache_read_per_token(&self) -> Option<f64> {
        self.cache_read_per_token
    }

    /// Cache write cost per token, if applicable.
    #[must_use]
    pub const fn cache_write_per_token(&self) -> Option<f64> {
        self.cache_write_per_token
    }

    /// Reasoning output cost per token, if applicable.
    #[must_use]
    pub const fn reasoning_per_token(&self) -> Option<f64> {
        self.reasoning_per_token
    }

    /// Image input cost per token, if applicable.
    #[must_use]
    pub const fn image_per_token(&self) -> Option<f64> {
        self.image_per_token
    }

    /// Sets cache read cost per token.
    #[must_use]
    pub const fn with_cache_read(mut self, cost: f64) -> Self {
        self.cache_read_per_token = Some(cost);
        self
    }

    /// Sets cache write cost per token.
    #[must_use]
    pub const fn with_cache_write(mut self, cost: f64) -> Self {
        self.cache_write_per_token = Some(cost);
        self
    }

    /// Sets reasoning output cost per token.
    #[must_use]
    pub const fn with_reasoning(mut self, cost: f64) -> Self {
        self.reasoning_per_token = Some(cost);
        self
    }

    /// Sets image input cost per token.
    #[must_use]
    pub const fn with_image(mut self, cost: f64) -> Self {
        self.image_per_token = Some(cost);
        self
    }
}

/// A single model entry from the upstream registry.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEntry {
    litellm_id: Cow<'static, str>,
    canonical_id: Cow<'static, str>,
    provider: Provider,
    mode: ModelMode,
    max_input_tokens: Option<u32>,
    max_output_tokens: Option<u32>,
    pricing: Pricing,
    abilities: Cow<'static, [Ability]>,
    deprecation_date: Option<Cow<'static, str>>,
}

impl ModelEntry {
    /// Creates a new model entry with owned strings.
    #[must_use]
    pub fn new(
        litellm_id: String,
        canonical_id: String,
        provider: Provider,
        mode: ModelMode,
    ) -> Self {
        Self {
            litellm_id: Cow::Owned(litellm_id),
            canonical_id: Cow::Owned(canonical_id),
            provider,
            mode,
            max_input_tokens: None,
            max_output_tokens: None,
            pricing: Pricing::default(),
            abilities: Cow::Borrowed(&[]),
            deprecation_date: None,
        }
    }

    /// Construct from all fields (used by generated code).
    #[must_use]
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new_static(
        litellm_id: &'static str,
        canonical_id: &'static str,
        provider: Provider,
        mode: ModelMode,
        max_input_tokens: Option<u32>,
        max_output_tokens: Option<u32>,
        pricing: Pricing,
        abilities: &'static [Ability],
        deprecation_date: Option<&'static str>,
    ) -> Self {
        Self {
            litellm_id: Cow::Borrowed(litellm_id),
            canonical_id: Cow::Borrowed(canonical_id),
            provider,
            mode,
            max_input_tokens,
            max_output_tokens,
            pricing,
            abilities: Cow::Borrowed(abilities),
            deprecation_date: match deprecation_date {
                Some(s) => Some(Cow::Borrowed(s)),
                None => None,
            },
        }
    }

    /// The canonical model ID (provider prefix stripped).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.canonical_id
    }

    /// The original LiteLLM key (e.g. `"gemini/gemini-2.5-flash"`).
    #[must_use]
    pub fn litellm_id(&self) -> &str {
        &self.litellm_id
    }

    /// The model provider.
    #[must_use]
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    /// The operational mode.
    #[must_use]
    pub fn mode(&self) -> ModelMode {
        self.mode
    }

    /// Maximum input tokens (context window), if known.
    #[must_use]
    pub const fn max_input_tokens(&self) -> Option<u32> {
        self.max_input_tokens
    }

    /// Maximum output tokens, if known.
    #[must_use]
    pub const fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    /// Pricing information.
    #[must_use]
    pub fn pricing(&self) -> &Pricing {
        &self.pricing
    }

    /// Supported abilities.
    #[must_use]
    pub fn abilities(&self) -> &[Ability] {
        &self.abilities
    }

    /// Whether this model has a specific ability.
    #[must_use]
    pub fn has_ability(&self, ability: Ability) -> bool {
        self.abilities.contains(&ability)
    }

    /// Deprecation date string (YYYY-MM-DD), if set.
    #[must_use]
    pub fn deprecation_date(&self) -> Option<&str> {
        self.deprecation_date.as_deref()
    }

    /// Whether this model is past its deprecation date.
    #[must_use]
    pub fn is_deprecated(&self) -> bool {
        self.deprecation_date.as_deref().is_some_and(|date| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let days = now / 86400;
            let (year, month, day) = days_to_ymd(days);
            let today = format!("{year:04}-{month:02}-{day:02}");
            date <= today.as_str()
        })
    }

    /// Sets max input tokens.
    #[must_use]
    pub const fn with_max_input_tokens(mut self, tokens: u32) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Sets max output tokens.
    #[must_use]
    pub const fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Sets pricing.
    #[must_use]
    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = pricing;
        self
    }

    /// Sets abilities from a vec.
    #[must_use]
    pub fn with_abilities(mut self, abilities: Vec<Ability>) -> Self {
        self.abilities = Cow::Owned(abilities);
        self
    }

    /// Sets deprecation date.
    #[must_use]
    pub fn with_deprecation_date(mut self, date: String) -> Self {
        self.deprecation_date = Some(Cow::Owned(date));
        self
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm from Howard Hinnant
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
