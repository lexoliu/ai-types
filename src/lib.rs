#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

extern crate alloc;

/// Audio generation and transcription traits and types.
pub mod audio;
/// Embedding model traits and types.
pub mod embedding;
/// Image generation traits and types.
pub mod image;
/// Language model traits, message types, and tool system.
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

/// Convenient result type used throughout the crate, defaulting to `String` as the success type.
pub type Result<T = String> = anyhow::Result<T>;
