/// Message types and conversation handling.
pub mod message;
/// Model profiles and capabilities.
pub mod model;
mod provider;
/// Tool system for function calling.
pub mod tool;
use crate::{
    Result,
    llm::{model::Parameters, tool::Tools},
};
use alloc::{format, string::String, vec::Vec};
use core::future::Future;
use futures_core::Stream;
use futures_lite::StreamExt;
pub use message::{Annotation, Message, Role, UrlAnnotation};
pub use provider::LanguageModelProvider;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
pub use tool::Tool;

use crate::llm::{model::Profile, tool::json};

/// Creates a two-message conversation with system and user prompts.
///
/// Returns an array containing a [`Message`] with [`Role::System`] and a [`Message`] with [`Role::User`].
fn oneshot(system: impl Into<String>, user: impl Into<String>) -> [Message; 2] {
    [Message::system(system.into()), Message::user(user.into())]
}

/// Types that can be converted into message iterators.
///
/// Implemented for [`Vec<Message>`](alloc::vec::Vec), arrays, and other iterables of [`Message`].
pub trait Messages: IntoIterator<Item = Message, IntoIter: Send> + Send {}

impl<T> Messages for T where T: IntoIterator<Item = Message, IntoIter: Send> + Send {}

/// A request to a language model.
///
/// Contains the conversation messages, available tools, and generation parameters.
/// This is the primary input structure for language model interactions.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::{Request, Message, model::Parameters};
///
/// // Simple request with messages
/// let messages = vec![
///     Message::system("You are a helpful assistant"),
///     Message::user("Hello!")
/// ];
/// let request = Request::new(messages);
///
/// // Request with custom parameters - create new messages
/// let other_messages = vec![Message::user("Another message")];
/// let request = Request::new(other_messages)
///     .parameters(Parameters::default().temperature(0.7));
/// ```
pub struct Request {
    /// The conversation messages to send to the model.
    pub messages: Vec<Message>,
    /// Available tools that the model can call.
    pub tools: Tools,
    /// Parameters controlling model behavior and generation.
    pub parameters: Parameters,
}

impl Request {
    /// Creates a new request with the given messages.
    ///
    /// The request is initialized with default tools and parameters.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages (can be `Vec<Message>`, array, etc.)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::{Request, Message};
    ///
    /// let messages = vec![Message::user("Hello, world!")];
    /// let request = Request::new(messages);
    /// ```
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            tools: Tools::default(),
            parameters: Parameters::default(),
        }
    }

    /// Creates a request for a simple system/user conversation.
    ///
    /// This is a convenience method that creates a two-message conversation
    /// with a system prompt and user message.
    ///
    /// # Arguments
    ///
    /// * `system` - The system message content
    /// * `user` - The user message content
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::Request;
    ///
    /// let request = Request::oneshot(
    ///     "You are a helpful assistant",
    ///     "What is the capital of France?"
    /// );
    /// ```
    pub fn oneshot(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self::new(oneshot(system, user))
    }

    /// Sets the generation parameters for this request.
    ///
    /// # Arguments
    ///
    /// * `parameters` - The model parameters to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::{Request, Message, model::Parameters};
    ///
    /// let request = Request::new(vec![Message::user("Hello")])
    ///     .parameters(Parameters::default().temperature(0.7));
    /// ```
    #[must_use]
    pub fn parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Adds a tool that the model can use.
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to add to this request
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::{Request, Message, Tool};
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(JsonSchema, Deserialize)]
    /// struct MyToolArgs {
    ///     input: String,
    /// }
    ///
    /// struct MyTool;
    ///
    /// impl Tool for MyTool {
    ///     const NAME: &'static str = "my_tool";
    ///     const DESCRIPTION: &'static str = "A test tool";
    ///     type Arguments = MyToolArgs;
    ///     
    ///     async fn call(&mut self, args: Self::Arguments) -> ai_types::Result {
    ///         Ok(format!("Processed: {}", args.input))
    ///     }
    /// }
    ///
    /// let tool = MyTool;
    /// let request = Request::new(vec![Message::user("Hello")])
    ///     .tool(tool);
    /// ```
    #[must_use]
    pub fn tool(mut self, tool: impl Tool) -> Self {
        self.tools.register(tool);
        self
    }
}

/// Language models for text generation and conversation.
pub trait LanguageModel: Sized + Send + Sync + 'static {
    /// Generates streaming response to conversation.
    ///
    /// Takes any type implementing [`Messages`] and returns a stream of text chunks.
    fn respond(&self, request: Request) -> impl Stream<Item = Result> + Send + Unpin;

    /// Generates structured output conforming to JSON schema.
    ///
    /// Uses [`schemars::JsonSchema`] to define the expected output structure.
    fn generate<T: JsonSchema + DeserializeOwned>(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<T>> + Send {
        generate(self, request)
    }

    /// Completes given text prefix.
    fn complete(&self, prefix: &str) -> impl Stream<Item = Result> + Send + Unpin;

    /// Summarizes text.
    fn summarize(&self, text: &str) -> impl Stream<Item = Result> + Send + Unpin {
        summarize(self, text)
    }

    /// Categorizes text according to JSON schema.
    ///
    /// Uses structured generation internally with a categorization prompt.
    fn categorize<T: JsonSchema + DeserializeOwned>(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<T>> + Send {
        categorize(self, text)
    }

    /// Returns model profile and capabilities.
    ///
    /// See [`Profile`] for details on model metadata.
    fn profile(&self) -> Profile;
}

async fn generate<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    mut request: Request,
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

    request.messages.push(Message::system(prompt));
    let response: String = model
        .respond(request)
        .try_fold(String::new(), |state, new| Ok(state + &new))
        .await?;

    let value: T = serde_json::from_str(&response)?;

    Ok(value)
}

fn summarize<'a, M: LanguageModel>(
    model: &'a M,
    text: &str,
) -> impl Stream<Item = Result<String>> + Send + Unpin + 'a {
    let messages = Request::oneshot("Summarize text:", text);
    model.respond(messages)
}

async fn categorize<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    text: &str,
) -> Result<T> {
    model
        .generate(Request::oneshot("Categorize text by provided schema", text))
        .await
}
