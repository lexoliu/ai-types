//! Model registry with capabilities, context windows, and metadata for popular LLMs.
//!
//! This crate provides a database of known AI models sourced from LiteLLM upstream,
//! with their capabilities, context window sizes, pricing, and provider information.
//!
//! # Example
//!
//! ```rust
//! use aither_models::{lookup, Ability};
//!
//! if let Some(entry) = lookup("gpt-4o") {
//!     assert!(entry.max_input_tokens().unwrap_or(0) > 0);
//!     assert!(entry.has_ability(Ability::Vision));
//! }
//! ```

pub mod convert;
mod registry;
#[cfg(feature = "remote")]
pub mod remote;
mod tier;
pub mod types;

pub use aither_core::llm::model::Ability;
pub use registry::ModelRegistry;
pub use tier::ModelTier;
pub use types::{ModelEntry, ModelMode, Pricing, Provider};

/// Look up a model by ID or alias.
///
/// Delegates to the bundled [`ModelRegistry`].
#[must_use]
pub fn lookup(model_id: &str) -> Option<&'static ModelEntry> {
    ModelRegistry::bundled().lookup(model_id)
}

/// Get all models for a provider.
pub fn models_for_provider(provider: &Provider) -> impl Iterator<Item = &'static ModelEntry> + '_ {
    ModelRegistry::bundled().models_for_provider(provider)
}

/// Get all models with a specific ability.
pub fn models_with_ability(ability: Ability) -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().models_with_ability(ability)
}

/// Get all models matching a mode.
pub fn models_by_mode(mode: ModelMode) -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().models_by_mode(mode)
}

/// Get all known models.
pub fn all_models() -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().all()
}

/// Capability/cost tier of a model in the bundled registry.
///
/// Unknown models classify as [`ModelTier::Balanced`].
#[must_use]
pub fn tier(model_id: &str) -> ModelTier {
    ModelRegistry::bundled().tier(model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_gpt4o() {
        let entry = lookup("gpt-4o").expect("gpt-4o should exist");
        assert_eq!(entry.id(), "gpt-4o");
        assert!(entry.max_input_tokens().unwrap_or(0) >= 128_000);
        assert!(entry.has_ability(Ability::Vision));
        assert!(entry.has_ability(Ability::ToolUse));
    }

    #[test]
    fn test_lookup_claude() {
        let entry = lookup("claude-sonnet-4-20250514").expect("claude-sonnet-4 should exist");
        assert!(entry.max_input_tokens().unwrap_or(0) >= 200_000);
        assert!(entry.has_ability(Ability::ToolUse));
    }

    #[test]
    fn test_lookup_gemini() {
        let entry = lookup("gemini/gemini-2.5-flash").expect("gemini-2.5-flash should exist");
        assert!(entry.max_input_tokens().unwrap_or(0) >= 1_000_000);
    }

    #[test]
    fn test_lookup_case_insensitive() {
        let entry = lookup("GPT-4O").expect("case-insensitive lookup should work");
        assert_eq!(entry.id(), "gpt-4o");
    }

    #[test]
    fn test_lookup_prefix_match() {
        // A dated variant should match the base model via prefix
        let entry = lookup("gpt-4o-2024-05-13");
        assert!(entry.is_some());
    }

    #[test]
    fn test_models_with_ability() {
        let vision_models: Vec<_> = models_with_ability(Ability::Vision).collect();
        assert!(!vision_models.is_empty());
        assert!(vision_models.iter().all(|m| m.has_ability(Ability::Vision)));
    }

    #[test]
    fn test_models_by_mode() {
        let chat_models: Vec<_> = models_by_mode(ModelMode::Chat).collect();
        assert!(!chat_models.is_empty());
        assert!(chat_models.iter().all(|m| m.mode() == ModelMode::Chat));
    }

    #[test]
    fn test_registry_not_empty() {
        let registry = ModelRegistry::bundled();
        assert!(
            registry.len() > 100,
            "Expected 100+ models, got {}",
            registry.len()
        );
    }

    #[test]
    fn test_pricing_populated() {
        let entry = lookup("gpt-4o").expect("gpt-4o should exist");
        assert!(entry.pricing().input_per_token() > 0.0);
        assert!(entry.pricing().output_per_token() > 0.0);
    }
}
