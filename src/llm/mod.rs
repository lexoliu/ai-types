/// Message types and conversation handling.
pub mod message;
/// Model profiles, capabilities, and configuration.
pub mod model;
mod provider;
/// Tool system for function calling and external integrations.
pub mod tool;
use alloc::{format, string::String};
use core::future::Future;
use futures_core::Stream;
use futures_lite::StreamExt;
pub use message::{Message, Role};
pub use provider::LanguageModelProvider;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
pub use tool::Tool;

use crate::llm::{model::Profile, tool::json};

/// Type alias for results returned by language model operations.
pub type Result<T = String> = anyhow::Result<T>;

/// Creates a simple two-message conversation with a system prompt and user message.
pub fn oneshot(system: impl Into<String>, user: impl Into<String>) -> [Message; 2] {
    [Message::system(system.into()), Message::user(user.into())]
}

/// Trait for types that can be converted into an iterator of messages.
pub trait Messages: IntoIterator<Item = Message, IntoIter: Send> + Send {}

impl<T> Messages for T where T: IntoIterator<Item = Message, IntoIter: Send> + Send {}

/// Trait for language models that can generate text and handle conversations.
pub trait LanguageModel: Sized + Send + Sync + 'static {
    /// Generates a streaming response to a conversation.
    fn respond(&self, messages: impl Messages) -> impl Stream<Item = Result> + Send + Unpin;

    /// Generates structured output conforming to a JSON schema.
    fn generate<T: JsonSchema + DeserializeOwned>(
        &self,
        messages: impl Messages,
    ) -> impl Future<Output = Result<T>> + Send {
        generate(self, messages)
    }

    /// Completes a given text prefix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The text to complete
    ///
    /// # Returns
    ///
    /// A stream of text chunks representing the completion.
    fn complete(&self, prefix: &str) -> impl Stream<Item = Result> + Send + Unpin;

    /// Summarizes the given text.
    fn summarize(&self, text: &str) -> impl Stream<Item = Result> + Send + Unpin {
        summarize(self, text)
    }

    /// Categorizes text according to a provided JSON schema.
    fn categorize<T: JsonSchema + DeserializeOwned>(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<T>> + Send {
        categorize(self, text)
    }

    /// Returns the model's profile including capabilities and limitations.
    fn profile(&self) -> Profile;
}

async fn generate<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    messages: impl Messages,
) -> Result<T> {
    let schema = json(&schema_for!(T));

    let prompt = format!(
        r#"You must respond with valid JSON that strictly conforms to the following JSON schema:

{schema}

Requirements:
- Your response must be ONLY valid JSON, no additional text, explanations, or markdown
- The JSON must exactly match the schema structure and types
- All required fields must be present
- Use appropriate data types (strings, numbers, booleans, arrays, objects)
- Ensure proper JSON syntax with correct quotes, brackets, and commas
- Do not include any text before or after the JSON

Example format: {{"field1": "value1", "field2": 123}}

Generate the JSON response now:"#
    );
    let messages = messages.into_iter().chain(Some(Message::system(prompt)));
    let response: String = model
        .respond(messages)
        .try_fold(String::new(), |state, new| Ok(state + &new))
        .await?;

    let value: T = serde_json::from_str(&response)?;

    Ok(value)
}

fn summarize<'a, M: LanguageModel>(
    model: &'a M,
    text: &str,
) -> impl Stream<Item = Result<String>> + Send + Unpin + 'a {
    let messages = oneshot("Summarize text:", text);
    model.respond(messages)
}

async fn categorize<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    text: &str,
) -> Result<T> {
    model
        .generate(oneshot("Categorize text by provided schema", text))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::model::Ability;
    use alloc::{string::ToString, vec, vec::Vec};
    use futures_lite::StreamExt;

    struct MockLanguageModel;

    impl LanguageModel for MockLanguageModel {
        fn respond(&self, messages: impl Messages) -> impl Stream<Item = Result> + Send + Unpin {
            let message_count = messages.into_iter().count();
            futures_lite::stream::iter(vec![
                Ok("Mock".to_string()),
                Ok(" response".to_string()),
                Ok(format!(" (from {message_count} messages)")),
            ])
        }

        fn complete(&self, prefix: &str) -> impl Stream<Item = Result> + Send + Unpin {
            futures_lite::stream::iter(vec![Ok(format!("{prefix} completion"))])
        }

        fn profile(&self) -> Profile {
            Profile::new("mock-model", "A mock language model", 2048).with_ability(Ability::ToolUse)
        }
    }

    #[test]
    fn test_oneshot_message_creation() {
        let messages = oneshot("You are helpful", "Hello");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are helpful");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");
    }

    #[test]
    fn test_oneshot_with_string_types() {
        let system = "System prompt".to_string();
        let user = "User message".to_string();
        let messages = oneshot(system.clone(), user.clone());

        assert_eq!(messages[0].content, system);
        assert_eq!(messages[1].content, user);
    }

    #[test]
    fn test_messages_trait_vec() {
        let messages: Vec<Message> = vec![
            Message::system("System"),
            Message::user("User"),
            Message::assistant("Assistant"),
        ];

        // Test that Vec<Message> implements Messages
        fn accepts_messages(_messages: impl Messages) {}
        accepts_messages(messages);
    }

    #[test]
    fn test_messages_trait_array() {
        let messages = [Message::system("System"), Message::user("User")];

        // Test that array implements Messages
        fn accepts_messages(_messages: impl Messages) {}
        accepts_messages(messages);
    }

    #[tokio::test]
    async fn test_language_model_respond() {
        let model = MockLanguageModel;
        let messages = oneshot("You are helpful", "Hello");

        let mut response_stream = model.respond(messages);
        let mut response_parts = Vec::new();

        while let Some(part) = response_stream.next().await {
            response_parts.push(part.unwrap());
        }

        assert_eq!(response_parts.len(), 3);
        assert_eq!(response_parts[0], "Mock");
        assert_eq!(response_parts[1], " response");
        assert_eq!(response_parts[2], " (from 2 messages)");
    }

    #[tokio::test]
    async fn test_language_model_complete() {
        let model = MockLanguageModel;

        let mut completion_stream = model.complete("Hello");
        let completion = completion_stream.next().await.unwrap().unwrap();

        assert_eq!(completion, "Hello completion");
    }

    #[test]
    fn test_language_model_profile() {
        let model = MockLanguageModel;
        let profile = model.profile();

        assert_eq!(profile.name, "mock-model");
        assert_eq!(profile.description, "A mock language model");
        assert_eq!(profile.context_length, 2048);
        assert_eq!(profile.abilities.len(), 1);
        assert_eq!(profile.abilities[0], Ability::ToolUse);
    }

    #[tokio::test]
    async fn test_language_model_summarize() {
        let model = MockLanguageModel;

        let mut summary_stream = model.summarize("This is a long text to summarize");
        let mut summary_parts = Vec::new();

        while let Some(part) = summary_stream.next().await {
            summary_parts.push(part.unwrap());
        }

        // Should get response from oneshot messages (system + user)
        assert_eq!(summary_parts.len(), 3);
        assert_eq!(summary_parts[2], " (from 2 messages)");
    }

    #[tokio::test]
    async fn test_generate_function() {
        use schemars::JsonSchema;
        use serde::{Deserialize, Serialize};

        #[derive(JsonSchema, Deserialize, Serialize, Debug, PartialEq)]
        struct TestResponse {
            message: String,
            count: u32,
        }

        // Create a mock model that returns valid JSON
        struct JsonMockModel;

        impl LanguageModel for JsonMockModel {
            fn respond(
                &self,
                _messages: impl Messages,
            ) -> impl Stream<Item = Result> + Send + Unpin {
                futures_lite::stream::iter(vec![Ok(
                    r#"{"message": "test", "count": 42}"#.to_string()
                )])
            }

            fn complete(&self, _prefix: &str) -> impl Stream<Item = Result> + Send + Unpin {
                futures_lite::stream::iter(vec![Ok("completion".to_string())])
            }

            fn profile(&self) -> Profile {
                Profile::new("json-mock", "JSON mock model", 1024)
            }
        }

        let model = JsonMockModel;
        let messages = oneshot("Generate", "test data");

        let result: TestResponse = model.generate(messages).await.unwrap();
        assert_eq!(result.message, "test");
        assert_eq!(result.count, 42);
    }

    #[tokio::test]
    async fn test_categorize_function() {
        use schemars::JsonSchema;
        use serde::{Deserialize, Serialize};

        #[derive(JsonSchema, Deserialize, Serialize, Debug, PartialEq)]
        struct Category {
            name: String,
            confidence: f32,
        }

        // Create a mock model that returns valid JSON
        struct CategoryMockModel;

        impl LanguageModel for CategoryMockModel {
            fn respond(
                &self,
                _messages: impl Messages,
            ) -> impl Stream<Item = Result> + Send + Unpin {
                futures_lite::stream::iter(vec![Ok(
                    r#"{"name": "positive", "confidence": 0.95}"#.to_string()
                )])
            }

            fn complete(&self, _prefix: &str) -> impl Stream<Item = Result> + Send + Unpin {
                futures_lite::stream::iter(vec![Ok("completion".to_string())])
            }

            fn profile(&self) -> Profile {
                Profile::new("category-mock", "Category mock model", 1024)
            }
        }

        let model = CategoryMockModel;
        let result: Category = model.categorize("This is great!").await.unwrap();

        assert_eq!(result.name, "positive");
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_result_type_alias() {
        let success: Result = Ok("success".to_string());
        let error: Result = Err(anyhow::Error::msg("error"));

        assert!(success.is_ok());
        assert!(error.is_err());

        if let Ok(value) = success {
            assert_eq!(value, "success");
        }
        if let Err(err) = error {
            assert_eq!(err.to_string(), "error");
        }
    }

    #[test]
    fn test_result_default_type() {
        let result: Result<String> = Ok("test".to_string());
        if let Ok(value) = result {
            assert_eq!(value, "test");
        }

        let result: Result<i32> = Ok(42);
        if let Ok(value) = result {
            assert_eq!(value, 42);
        }
    }
}
