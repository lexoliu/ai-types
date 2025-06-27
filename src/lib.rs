#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

extern crate alloc;

/// Embedding model traits and types.
pub mod embedding;
/// Image generation traits and types.
pub mod image;
/// Language model traits, message types, and tool system.
pub mod llm;

pub use embedding::EmbeddingModel;
pub use image::ImageGenerator;
pub use llm::LanguageModel;
