use core::future::Future;

use alloc::{string::String, vec::Vec};

use crate::LanguageModel;

/// Trait for AI service providers that can list and provide language models.
pub trait LanguageModelProvider {
    /// The type of language model this provider creates.
    type Model: LanguageModel;

    /// Lists all available models from this provider.
    fn list_models(&self) -> impl Future<Output = Vec<String>> + Send;

    /// Gets a specific model by name from this provider.
    fn get_model(&self, name: &str) -> impl Future<Output = Self::Model> + Send;

    /// Returns the provider's profile information.
    fn profile() -> Profile;
}

/// Provider profile information.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Profile {
    name: String,
    description: String,
}

impl Profile {
    /// Creates a new profile with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }

    /// Returns the provider's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the provider's description.
    pub fn description(&self) -> &str {
        &self.description
    }
}
