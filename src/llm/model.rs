use alloc::{string::String, vec::Vec};

use crate::llm::tool::Tools;

#[derive(Debug)]
/// Parameters for configuring the behavior of a language model.
pub struct Parameters {
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus sampling probability.
    pub top_p: f32,
    /// Top-k sampling parameter.
    pub top_k: u32,
    /// Frequency penalty to reduce repetition.
    pub frequency_penalty: f32,
    /// Presence penalty to encourage new tokens.
    pub presence_penalty: f32,
    /// Repetition penalty to penalize repeated tokens.
    pub repetition_penalty: f32,
    /// Minimum probability for nucleus sampling.
    pub min_p: f32,
    /// Top-a sampling parameter.
    pub top_a: f32,
    /// Random seed for reproducibility.
    pub seed: u32,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Biases for specific logits.
    pub logit_bias: Option<Vec<(String, f32)>>,
    /// Whether to return log probabilities.
    pub logprobs: bool,
    /// Number of top log probabilities to return.
    pub top_logprobs: u8,
    /// Stop sequences to end generation.
    pub stop: Option<Vec<String>>,
    /// Tools available to the model.
    pub tools: Tools,
    /// Tool choices available to the model.
    pub tool_choice: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a language model's profile, including its name, description, abilities, context length, and optional pricing.
pub struct Profile {
    /// The name of the model.
    pub name: String,
    /// A description of the model.
    pub description: String,
    /// The abilities supported by the model.
    pub abilities: Vec<Ability>,
    /// The maximum context length supported by the model.
    pub context_length: u32,
    /// Optional pricing information for the model.
    pub pricing: Option<Pricing>,
}

/// Pricing information for a model's various capabilities (unit: USD).
#[derive(Debug, Clone, PartialEq)]
pub struct Pricing {
    /// Price per prompt token.
    pub prompt: f64,
    /// Price per completion token.
    pub completion: f64,
    /// Price per request.
    pub request: f64,
    /// Price per image processed.
    pub image: f64,
    /// Price per web search.
    pub web_search: f64,
    /// Price for internal reasoning.
    pub internal_reasoning: f64,
    /// Price for reading from input cache.
    pub input_cache_read: f64,
    /// Price for writing to input cache.
    pub input_cache_write: f64,
}

/// Indicates which parameters are supported by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportedParameters {
    /// Whether tools are supported.
    pub tools: bool,
    /// Whether tool choice is supported.
    pub tool_choice: bool,
    /// Whether max_tokens is supported.
    pub max_tokens: bool,
    /// Whether temperature is supported.
    pub temperature: bool,
    /// Whether top_p is supported.
    pub top_p: bool,
    /// Whether reasoning is supported.
    pub reasoning: bool,
    /// Whether including reasoning is supported.
    pub include_reasoning: bool,
    /// Whether structured outputs are supported.
    pub structured_outputs: bool,
    /// Whether response format is supported.
    pub response_format: bool,
    /// Whether stop sequences are supported.
    pub stop: bool,
    /// Whether frequency penalty is supported.
    pub frequency_penalty: bool,
    /// Whether presence penalty is supported.
    pub presence_penalty: bool,
    /// Whether seed is supported.
    pub seed: bool,
}

impl Profile {
    /// Creates a new `Profile` with the given name, description, and context length.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        context_length: u32,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            abilities: Vec::new(),
            context_length,
            pricing: None,
        }
    }

    /// Adds a single ability to the profile.
    pub fn with_ability(self, ability: Ability) -> Self {
        self.with_abilities([ability])
    }

    /// Adds multiple abilities to the profile.
    pub fn with_abilities(mut self, abilities: impl IntoIterator<Item = Ability>) -> Self {
        self.abilities.extend(abilities);
        self
    }

    /// Sets the pricing information for the profile.
    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }
}

/// Represents the capabilities that a language model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ability {
    /// The model can use external tools/functions.
    ToolUse,
    /// The model can process and understand images.
    Vision,
    /// The model can process and understand audio.
    Audio,
    /// The model can perform web searches naitvely.
    WebSearch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_profile_creation() {
        let profile = Profile::new("test-model", "A test model", 4096);

        assert_eq!(profile.name, "test-model");
        assert_eq!(profile.description, "A test model");
        assert_eq!(profile.context_length, 4096);
        assert!(profile.abilities.is_empty());
        assert!(profile.pricing.is_none());
    }

    #[test]
    fn test_profile_with_single_ability() {
        let profile =
            Profile::new("vision-model", "A vision model", 8192).with_ability(Ability::Vision);

        assert_eq!(profile.abilities.len(), 1);
        assert_eq!(profile.abilities[0], Ability::Vision);
    }

    #[test]
    fn test_profile_with_multiple_abilities() {
        let abilities = vec![Ability::ToolUse, Ability::Vision, Ability::Audio];
        let profile = Profile::new("multimodal-model", "A multimodal model", 16384)
            .with_abilities(abilities.clone());

        assert_eq!(profile.abilities.len(), 3);
        assert_eq!(profile.abilities, abilities);
    }

    #[test]
    fn test_profile_with_pricing() {
        let pricing = Pricing {
            prompt: 0.0001,
            completion: 0.0002,
            request: 0.001,
            image: 0.01,
            web_search: 0.005,
            internal_reasoning: 0.0003,
            input_cache_read: 0.00005,
            input_cache_write: 0.0001,
        };

        let profile =
            Profile::new("paid-model", "A paid model", 2048).with_pricing(pricing.clone());

        assert!(profile.pricing.is_some());
        let profile_pricing = profile.pricing.unwrap();
        assert_eq!(profile_pricing.prompt, 0.0001);
        assert_eq!(profile_pricing.completion, 0.0002);
        assert_eq!(profile_pricing.request, 0.001);
        assert_eq!(profile_pricing.image, 0.01);
        assert_eq!(profile_pricing.web_search, 0.005);
        assert_eq!(profile_pricing.internal_reasoning, 0.0003);
        assert_eq!(profile_pricing.input_cache_read, 0.00005);
        assert_eq!(profile_pricing.input_cache_write, 0.0001);
    }

    #[test]
    fn test_profile_builder_pattern() {
        let pricing = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let profile = Profile::new("full-model", "A full-featured model", 32768)
            .with_ability(Ability::ToolUse)
            .with_ability(Ability::Vision)
            .with_abilities(vec![Ability::Audio, Ability::WebSearch])
            .with_pricing(pricing);

        assert_eq!(profile.name, "full-model");
        assert_eq!(profile.description, "A full-featured model");
        assert_eq!(profile.context_length, 32768);
        assert_eq!(profile.abilities.len(), 4);
        assert!(profile.abilities.contains(&Ability::ToolUse));
        assert!(profile.abilities.contains(&Ability::Vision));
        assert!(profile.abilities.contains(&Ability::Audio));
        assert!(profile.abilities.contains(&Ability::WebSearch));
        assert!(profile.pricing.is_some());
    }

    #[test]
    fn test_ability_equality() {
        assert_eq!(Ability::ToolUse, Ability::ToolUse);
        assert_eq!(Ability::Vision, Ability::Vision);
        assert_eq!(Ability::Audio, Ability::Audio);
        assert_eq!(Ability::WebSearch, Ability::WebSearch);

        assert_ne!(Ability::ToolUse, Ability::Vision);
        assert_ne!(Ability::Audio, Ability::WebSearch);
    }

    #[test]
    fn test_ability_debug() {
        let ability = Ability::ToolUse;
        let debug_str = alloc::format!("{ability:?}");
        assert!(debug_str.contains("ToolUse"));
    }

    #[test]
    fn test_profile_debug() {
        let profile = Profile::new("debug-model", "A debug model", 1024);
        let debug_str = alloc::format!("{profile:?}");
        assert!(debug_str.contains("debug-model"));
        assert!(debug_str.contains("A debug model"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn test_profile_clone() {
        let original =
            Profile::new("original", "Original model", 2048).with_ability(Ability::Vision);
        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.description, cloned.description);
        assert_eq!(original.context_length, cloned.context_length);
        assert_eq!(original.abilities, cloned.abilities);
    }

    #[test]
    fn test_pricing_debug() {
        let pricing = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let debug_str = alloc::format!("{pricing:?}");
        assert!(debug_str.contains("0.001"));
        assert!(debug_str.contains("0.002"));
    }

    #[test]
    fn test_pricing_clone() {
        let original = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };
        let cloned = original.clone();

        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.completion, cloned.completion);
        assert_eq!(original.request, cloned.request);
        assert_eq!(original.image, cloned.image);
        assert_eq!(original.web_search, cloned.web_search);
        assert_eq!(original.internal_reasoning, cloned.internal_reasoning);
        assert_eq!(original.input_cache_read, cloned.input_cache_read);
        assert_eq!(original.input_cache_write, cloned.input_cache_write);
    }

    #[test]
    fn test_pricing_equality() {
        let pricing1 = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let pricing2 = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let pricing3 = Pricing {
            prompt: 0.002, // Different value
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        assert_eq!(pricing1, pricing2);
        assert_ne!(pricing1, pricing3);
    }

    #[test]
    fn test_supported_parameters() {
        let params = SupportedParameters {
            tools: true,
            tool_choice: false,
            max_tokens: true,
            temperature: true,
            top_p: false,
            reasoning: true,
            include_reasoning: false,
            structured_outputs: true,
            response_format: false,
            stop: true,
            frequency_penalty: false,
            presence_penalty: true,
            seed: false,
        };

        assert!(params.tools);
        assert!(!params.tool_choice);
        assert!(params.max_tokens);
        assert!(params.temperature);
        assert!(!params.top_p);
    }

    #[test]
    fn test_parameters_debug() {
        let params = Parameters {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_p: 0.05,
            top_a: 0.0,
            seed: 42,
            max_tokens: 1000,
            logit_bias: None,
            logprobs: false,
            top_logprobs: 0,
            stop: None,
            tools: Tools::new(),
            tool_choice: None,
        };

        let debug_str = alloc::format!("{params:?}");
        assert!(debug_str.contains("0.7"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("1000"));
    }
}
