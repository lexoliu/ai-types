//! Error types for the memory layer.

use thiserror::Error;

/// Something went wrong while storing or recalling a memory.
#[derive(Debug, Error)]
pub enum Mem0Error {
    /// The language model call used to extract or reconcile facts failed.
    #[error("LLM error: {0}")]
    Llm(anyhow::Error),

    /// Embedding the text to store or search for failed.
    #[error("Embedding error: {0}")]
    Embedding(anyhow::Error),

    /// The backing store rejected a read or write.
    #[error("Store error: {0}")]
    Store(String),

    /// The model answered, but not with facts we could use.
    #[error("Extraction failed: {0}")]
    Extraction(String),
}

/// Result alias for memory operations.
pub type Result<T> = core::result::Result<T, Mem0Error>;
