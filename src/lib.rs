//! # ai-types
//!
//! **Write AI applications that work with any provider** 🚀
//!
//! This crate provides unified trait abstractions for AI models, letting you write code once
//! and switch between providers (`OpenAI`, `Anthropic`, local models, etc.) without changing your application logic.
//!
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Your App      │───▶│    ai-types      │◀───│   Providers     │
//! │                 │    │   (this crate)   │    │                 │
//! │ - Chat bots     │    │                  │    │ - openai        │
//! │ - Search        │    │ - LanguageModel  │    │ - anthropic     │
//! │ - Content gen   │    │ - EmbeddingModel │    │ - llama.cpp     │
//! │ - Voice apps    │    │ - ImageGenerator │    │ - whisper       │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

//!
//! ## Supported AI Capabilities
//!
//! | Capability | Trait | Description |
//! |------------|-------|-------------|
//! | **Language Models** | [`LanguageModel`] | Text generation, conversations, structured output |
//! | **Embeddings** | [`EmbeddingModel`] | Convert text to vectors for semantic search |
//! | **Image Generation** | [`ImageGenerator`] | Create and edit images from text prompts |
//! | **Text-to-Speech** | [`AudioGenerator`] | Generate speech audio from text |
//! | **Speech-to-Text** | [`AudioTranscriber`] | Transcribe audio to text |
//!
//! ## Examples
//!
//! ### Basic Chat Bot
//!
//! ```rust
//! use ai_types::{LanguageModel, llm::{Message, Request}};
//! use futures_lite::StreamExt;
//!
//! async fn chat_example(model: impl LanguageModel) -> ai_types::Result {
//!     let messages = vec![
//!         Message::system("You are a helpful assistant"),
//!         Message::user("What's the capital of France?")
//!     ];
//!     
//!     let request = Request::new(messages);
//!     let mut response = model.respond(request);
//!     
//!     let mut full_response = String::new();
//!     while let Some(chunk) = response.next().await {
//!         full_response.push_str(&chunk?);
//!     }
//!     
//!     Ok(full_response)
//! }
//! ```
//!
//! ### Structured Output with Tools
//!
//! ```rust
//! use ai_types::{LanguageModel, llm::{Message, Request, Tool}};
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct WeatherQuery {
//!     location: String,
//!     units: Option<String>,
//! }
//!
//! struct WeatherTool;
//!
//! impl Tool for WeatherTool {
//!     const NAME: &'static str = "get_weather";
//!     const DESCRIPTION: &'static str = "Get current weather for a location";
//!     type Arguments = WeatherQuery;
//!     
//!     async fn call(&mut self, args: Self::Arguments) -> ai_types::Result {
//!         Ok(format!("Weather in {}: 22°C, sunny", args.location))
//!     }
//! }
//!
//! async fn weather_bot(model: impl LanguageModel) -> ai_types::Result {
//!     let request = Request::new(vec![
//!         Message::user("What's the weather like in Tokyo?")
//!     ]).tool(WeatherTool);
//!     
//!     // Model can now call the weather tool automatically
//!     let response: String = model.generate(request).await?;
//!     Ok(response)
//! }
//! ```
//!
//! ### Semantic Search with Embeddings
//!
//! ```rust
//! use ai_types::EmbeddingModel;
//!
//! async fn find_similar_docs(
//!     model: impl EmbeddingModel,
//!     query: &str,
//!     documents: &[&str]
//! ) -> ai_types::Result<Vec<f32>> {
//!     // Convert query to vector
//!     let query_embedding = model.embed(query).await?;
//!     
//!     // In a real app, you'd compare with document embeddings
//!     // and find the most similar ones using cosine similarity
//!     
//!     Ok(query_embedding)
//! }
//! ```
//!
//! ### Image Generation
//!
//! ```rust
//! use ai_types::{ImageGenerator, image::{Prompt, Size}};
//! use futures_lite::StreamExt;
//!
//! async fn generate_image(generator: impl ImageGenerator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let prompt = Prompt::new("A beautiful sunset over mountains");
//!     let size = Size::Square1024;
//!     
//!     let mut image_stream = generator.create(prompt, size);
//!     let mut image_data = Vec::new();
//!     
//!     while let Some(chunk) = image_stream.next().await {
//!         image_data.extend_from_slice(&chunk?);
//!     }
//!     
//!     Ok(image_data)
//! }
//! ```
//!
//!

#![no_std]
extern crate alloc;
/// Audio generation and transcription.
///
/// Contains [`AudioGenerator`] and [`AudioTranscriber`] traits.
pub mod audio;
/// Text and multimodal embeddings.
///
/// Contains [`EmbeddingModel`] trait for vector representations.
pub mod embedding;
/// Text-to-image generation.
///
/// Contains [`ImageGenerator`] trait for creating images from text.
pub mod image;
/// Language models, messages, and tools.
///
/// Contains [`LanguageModel`] trait, [`llm::Message`] types, and [`llm::Tool`] system.
pub mod llm;

/// Content moderation utilities.
///
/// Contains traits and types for detecting and handling unsafe or inappropriate content.
pub mod moderation;

use alloc::string::String;

#[doc(inline)]
pub use audio::{AudioGenerator, AudioTranscriber};
#[doc(inline)]
pub use embedding::EmbeddingModel;
#[doc(inline)]
pub use image::ImageGenerator;
#[doc(inline)]
pub use llm::LanguageModel;

/// Result type used throughout the crate.
///
/// Type alias for [`anyhow::Result<T>`](anyhow::Result) with [`String`] as default success type.
pub type Result<T = String> = anyhow::Result<T>;

pub use anyhow::Error;
