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
pub struct Profile {
    /// The name of the provider.
    pub name: String,
    /// A description of the provider.
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::model::Profile as ModelProfile;
    use crate::llm::{Message, Messages};
    use alloc::{string::ToString, vec};
    use futures_core::Stream;
    use futures_lite::StreamExt;

    struct MockModel {
        name: String,
    }

    impl LanguageModel for MockModel {
        fn respond(
            &self,
            _messages: impl Messages,
        ) -> impl Stream<Item = crate::Result> + Send + Unpin {
            futures_lite::stream::iter(vec![Ok("Mock response".to_string())])
        }

        fn complete(&self, _prefix: &str) -> impl Stream<Item = crate::Result> + Send + Unpin {
            futures_lite::stream::iter(vec![Ok("Mock completion".to_string())])
        }

        fn profile(&self) -> ModelProfile {
            ModelProfile::new(&self.name, "A mock model", 2048)
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
    async fn test_provider_list_models() {
        let provider = MockProvider;
        let models = provider.list_models().await;

        assert_eq!(models.len(), 3);
        assert_eq!(models[0], "model-1");
        assert_eq!(models[1], "model-2");
        assert_eq!(models[2], "model-3");
    }

    #[tokio::test]
    async fn test_provider_get_model() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        assert_eq!(model.name, "test-model");

        let profile = model.profile();
        assert_eq!(profile.name, "test-model");
        assert_eq!(profile.description, "A mock model");
        assert_eq!(profile.context_length, 2048);
    }

    #[test]
    fn test_provider_profile() {
        let profile = MockProvider::profile();

        assert_eq!(profile.name, "MockProvider");
        assert_eq!(profile.description, "A mock AI provider for testing");
    }

    #[tokio::test]
    async fn test_model_respond() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        let messages = vec![Message::user("Hello")];
        let mut response_stream = model.respond(messages);

        let response = response_stream.next().await.unwrap().unwrap();
        assert_eq!(response, "Mock response");
    }

    #[tokio::test]
    async fn test_model_complete() {
        let provider = MockProvider;
        let model = provider.get_model("test-model").await;

        let mut completion_stream = model.complete("Hello");
        let completion = completion_stream.next().await.unwrap().unwrap();
        assert_eq!(completion, "Mock completion");
    }

    #[test]
    fn test_profile_creation() {
        let profile = Profile {
            name: "TestProvider".to_string(),
            description: "A test provider".to_string(),
        };

        assert_eq!(profile.name, "TestProvider");
        assert_eq!(profile.description, "A test provider");
    }
}
