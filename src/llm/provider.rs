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

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;
    use crate::llm::Message;
    use alloc::{string::ToString, vec};
    use futures_core::Stream;
    use futures_lite::StreamExt;

    struct MockModel {
        name: String,
    }

    impl LanguageModel for MockModel {
        type Error = Infallible;
        fn respond(
            &self,
            _request: crate::llm::Request,
        ) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin {
            futures_lite::stream::iter(Some(Ok("Mock response".to_string())))
        }

        fn complete(
            &self,
            _prefix: &str,
        ) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin {
            futures_lite::stream::iter(Some(Ok("Mock completion".to_string())))
        }

        fn profile(&self) -> crate::llm::model::Profile {
            crate::llm::model::Profile::new(&self.name, "A mock model", 2048)
        }
    }

    struct MockProvider;

    impl LanguageModelProvider for MockProvider {
        type Model = MockModel;

        async fn list_models(&self) -> Vec<String> {
            vec![
                "model-1".to_string(),
                "model-2".to_string(),
                "model-3".to_string(),
            ]
        }

        async fn get_model(&self, name: &str) -> Self::Model {
            MockModel {
                name: name.to_string(),
            }
        }

        fn profile() -> Profile {
            Profile {
                name: "MockProvider".to_string(),
                description: "A mock AI provider for testing".to_string(),
            }
        }
    }

    #[tokio::test]
    async fn provider_list_models() {
        let provider = MockProvider;
        let models = provider.list_models().await;

        assert_eq!(models.len(), 3);
        assert_eq!(models[0], "model-1");
        assert_eq!(models[1], "model-2");
        assert_eq!(models[2], "model-3");
    }

    #[tokio::test]
    async fn provider_get_model() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        assert_eq!(model.name, "test-model");

        let profile = model.profile();
        assert_eq!(profile.name, "test-model");
        assert_eq!(profile.description, "A mock model");
        assert_eq!(profile.context_length, 2048);
    }

    #[test]
    fn provider_profile() {
        let profile = MockProvider::profile();

        assert_eq!(profile.name, "MockProvider");
        assert_eq!(profile.description, "A mock AI provider for testing");
    }

    #[tokio::test]
    async fn model_respond() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        let messages = vec![Message::user("Hello")];
        let request = crate::llm::Request::new(messages);
        let mut response_stream = model.respond(request);

        let response = response_stream.next().await.unwrap().unwrap();
        assert_eq!(response, "Mock response");
    }

    #[tokio::test]
    async fn model_complete() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        let mut completion_stream = model.complete("Hello");
        let completion = completion_stream.next().await.unwrap().unwrap();
        assert_eq!(completion, "Mock completion");
    }

    #[test]
    fn profile_creation() {
        let profile = Profile {
            name: "TestProvider".to_string(),
            description: "A test provider".to_string(),
        };

        assert_eq!(profile.name, "TestProvider");
        assert_eq!(profile.description, "A test provider");
    }
}
