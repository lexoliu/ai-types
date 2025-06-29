//! # Language Models and Conversation Management
//!
//! This module provides everything you need to work with language models in a provider-agnostic way.
//! Build chat applications, generate structured output, and integrate tools without being tied to any specific AI service.
//!
//! ## Core Components
//!
//! - **[`LanguageModel`]** - The main trait for text generation and conversation
//! - **[`Request`]** - Encapsulates messages, tools, and parameters for model calls
//! - **[`Message`]** - Represents individual messages in a conversation
//! - **[`Tool`]** - Function calling interface for extending model capabilities
//!
//! ## Quick Start
//!
//! ### Basic Conversation
//!
//! ```rust
//! use ai_types::llm::{LanguageModel, Request, Message};
//! use futures_lite::StreamExt;
//!
//! async fn chat_with_model(model: impl LanguageModel) -> Result<String, Box<dyn std::error::Error>> {
//!     // Create a simple conversation
//!     let request = Request::oneshot(
//!         "You are a helpful assistant",
//!         "What's the capital of Japan?"
//!     );
//!
//!     // Stream the response
//!     let mut response = model.respond(request);
//!     let mut full_text = String::new();
//!     
//!     while let Some(chunk) = response.next().await {
//!         full_text.push_str(&chunk?);
//!     }
//!     
//!     Ok(full_text)
//! }
//! ```
//!
//! ### Multi-turn Conversation
//!
//! ```rust
//! use ai_types::llm::{Request, Message};
//!
//! let messages = [
//!     Message::system("You are a helpful coding assistant"),
//!     Message::user("How do I create a vector in Rust?"),
//!     Message::assistant("You can create a vector using `Vec::new()` or the `vec!` macro..."),
//!     Message::user("Can you show me an example?"),
//! ];
//!
//! let request = Request::new(messages);
//! ```
//!
//! ### Structured Output Generation
//!
//! ```rust
//! use ai_types::llm::{LanguageModel, Request, Message};
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct WeatherResponse {
//!     temperature: f32,
//!     condition: String,
//!     humidity: i32,
//! }
//!
//! async fn get_weather_data(model: impl LanguageModel) -> ai_types::Result<WeatherResponse> {
//!     let request = Request::oneshot(
//!         "Extract weather information from the following text",
//!         "It's 22°C and sunny with 65% humidity today"
//!     );
//!
//!     model.generate::<WeatherResponse>(request).await
//! }
//! ```
//!
//! ### Function Calling with Tools
//!
//! ```rust
//! use ai_types::llm::{Request, Message, Tool};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct CalculatorArgs {
//!     operation: String,  // "add", "subtract", "multiply", "divide"
//!     x: f64,
//!     y: f64,
//! }
//!
//! struct Calculator;
//!
//! impl Tool for Calculator {
//!     const NAME: &str = "calculator";
//!     const DESCRIPTION: &str = "Performs basic arithmetic operations";
//!     type Arguments = CalculatorArgs;
//!
//!     async fn call(&mut self, args: Self::Arguments) -> ai_types::Result {
//!         let result = match args.operation.as_str() {
//!             "add" => args.x + args.y,
//!             "subtract" => args.x - args.y,
//!             "multiply" => args.x * args.y,
//!             "divide" => args.x / args.y,
//!             _ => return Err(anyhow::anyhow!("Unknown operation")),
//!         };
//!         Ok(result.to_string())
//!     }
//! }
//!
//! // Usage
//! let request = Request::new([
//!     Message::user("What's 15 multiplied by 23?")
//! ]).with_tool(Calculator);
//! ```
//!
//! ### Model Configuration
//!
//! ```rust
//! use ai_types::llm::{Request, Message, model::Parameters};
//!
//! let request = Request::new([
//!     Message::user("Write a creative story")
//! ]).with_parameters(
//!     Parameters::default()
//!         .temperature(0.8)        // More creative
//!         .top_p(0.9)             // Nucleus sampling
//!         .frequency_penalty(0.5)  // Reduce repetition
//! );
//! ```
//!
//! ## Advanced Features
//!
//! ### Text Summarization
//!
//! ```rust
//! use ai_types::llm::LanguageModel;
//! use futures_lite::StreamExt;
//!
//! async fn summarize_text(model: impl LanguageModel, text: &str) -> Result<String, Box<dyn std::error::Error>> {
//!     let mut summary_stream = model.summarize(text);
//!     let mut summary = String::new();
//!     
//!     while let Some(chunk) = summary_stream.next().await {
//!         summary.push_str(&chunk?);
//!     }
//!     
//!     Ok(summary)
//! }
//! ```
//!
//! ### Text Categorization
//!
//! ```rust
//! use ai_types::llm::LanguageModel;
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! enum DocumentCategory {
//!     Technical,
//!     Marketing,
//!     Legal,
//!     Support,
//!     Internal,
//! }
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct ClassificationResult {
//!     category: DocumentCategory,
//!     confidence: f32,
//!     reasoning: String,
//! }
//!
//! async fn categorize_document(model: impl LanguageModel, text: &str) -> ai_types::Result<ClassificationResult> {
//!     model.categorize::<ClassificationResult>(text).await
//! }
//! ```
//!
//! ## Message Types and Annotations
//!
//! Messages support rich content including file attachments and URL annotations:
//!
//! ```rust
//! use ai_types::llm::{Message, UrlAnnotation, Annotation};
//! use url::Url;
//!
//! let message = Message::user("Check this documentation")
//!     .with_attachment("file:///path/to/doc.pdf")
//!     .with_annotation(
//!         Annotation::url(
//!             "https://docs.rs/ai-types",
//!             "AI Types Documentation",
//!             "Rust crate for AI model abstractions",
//!             0,
//!             25,
//!         )
//!    );
//! ```
/// Message types and conversation handling.
pub mod message;
/// Model profiles and capabilities.
pub mod model;
mod provider;
/// Tool system for function calling.
pub mod tool;
use crate::llm::{model::Parameters, tool::Tools};
use alloc::{boxed::Box, format, string::String, sync::Arc, vec::Vec};
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
/// let messages = [
///     Message::system("You are a helpful assistant"),
///     Message::user("Hello!")
/// ];
/// let request = Request::new(messages);
///
/// // Request with custom parameters - create new messages
/// let other_messages = [Message::user("Another message")];
/// let request = Request::new(other_messages)
///     .with_parameters(Parameters::default().temperature(0.7));
/// ```
#[derive(Debug, Default)]
pub struct Request {
    messages: Vec<Message>,
    tools: Tools,
    parameters: Parameters,
}

impl Request {
    /// Return available tools that the model can call.
    #[must_use]
    pub const fn tools(&self) -> &Tools {
        &self.tools
    }

    /// Return parameters controlling model behavior and generation.
    #[must_use]
    pub const fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Return the conversation messages to send to the model.
    #[must_use]
    pub const fn messages(&self) -> &[Message] {
        self.messages.as_slice()
    }
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
    /// let messages = [Message::user("Hello, world!")];
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
    /// let request = Request::new([Message::user("Hello")])
    ///     .with_parameters(Parameters::default().temperature(0.7));
    /// ```
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
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
    ///     const NAME: &str = "my_tool";
    ///     const DESCRIPTION: &str = "A test tool";
    ///     type Arguments = MyToolArgs;
    ///     
    ///     async fn call(&mut self, args: Self::Arguments) -> ai_types::Result {
    ///         Ok(format!("Processed: {}", args.input))
    ///     }
    /// }
    ///
    /// let tool = MyTool;
    /// let request = Request::new([Message::user("Hello")])
    ///     .with_tool(tool);
    /// ```
    #[must_use]
    pub fn with_tool(mut self, tool: impl Tool) -> Self {
        self.tools.register(tool);
        self
    }
}

/// Language models for text generation and conversation.
///
/// See the [module documentation](crate::llm) for examples and usage patterns.
pub trait LanguageModel: Sized + Send + Sync + 'static {
    /// The error type returned by this language model.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Generates streaming response to conversation.
    fn respond(
        &self,
        request: Request,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin;

    /// Generates structured output conforming to JSON schema.
    fn generate<T: JsonSchema + DeserializeOwned>(
        &self,
        request: Request,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        generate(self, request)
    }

    /// Completes given text prefix.
    fn complete(
        &self,
        prefix: &str,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin;

    /// Summarizes text.
    fn summarize(
        &self,
        text: &str,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin {
        summarize(self, text)
    }

    /// Categorizes text.
    fn categorize<T: JsonSchema + DeserializeOwned>(
        &self,
        text: &str,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        categorize(self, text)
    }

    /// Returns model profile and capabilities.
    ///
    /// See [`Profile`] for details on model metadata.
    fn profile(&self) -> Profile;
}

macro_rules! impl_language_model {
    ($($name:ident),*) => {
        $(
            impl<T: LanguageModel> LanguageModel for $name<T> {
                type Error = T::Error;
                fn respond(&self, request: Request) -> impl Stream<Item = Result<String, Self::Error>> + Send + Unpin{
                    T::respond(self, request)
                }

                fn generate<U: JsonSchema + DeserializeOwned>(
                    &self,
                    request: Request,
                ) -> impl Future<Output = crate::Result<U>> + Send {
                    T::generate(self, request)
                }

                fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String,Self::Error>> + Send + Unpin {
                    T::complete(self, prefix)
                }

                fn summarize(&self, text: &str) -> impl Stream<Item = Result<String,Self::Error>> + Send + Unpin {
                    T::summarize(self, text)
                }

                fn categorize<U: JsonSchema + DeserializeOwned>(
                    &self,
                    text: &str,
                ) -> impl Future<Output = crate::Result<U>> + Send {
                    T::categorize(self, text)
                }

                fn profile(&self) -> Profile {
                    T::profile(self)
                }
            }
        )*
    };
}

impl_language_model!(Arc, Box);

async fn generate<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    mut request: Request,
) -> crate::Result<T> {
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
) -> impl Stream<Item = Result<String, M::Error>> + Send + Unpin + 'a {
    let messages = Request::oneshot("Summarize text:", text);
    model.respond(messages)
}

async fn categorize<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    text: &str,
) -> crate::Result<T> {
    model
        .generate(Request::oneshot("Categorize text by provided schema", text))
        .await
}
