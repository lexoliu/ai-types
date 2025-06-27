#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

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
