use alloc::vec::Vec;

/// Trait for content moderation services.
pub trait Moderation {
    /// The error type returned by moderation operations.
    type Error: core::error::Error + Send + Sync;

    /// Moderates the provided content and returns a result asynchronously.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to be moderated.
    fn moderate(
        &self,
        content: &str,
    ) -> impl Future<Output = Result<ModerationResult, Self::Error>> + Send;
}

/// The result of a moderation operation.
pub struct ModerationResult {
    /// Indicates whether the content was flagged.
    pub flagged: bool,
    /// The categories that were detected in the content.
    pub categories: Vec<ModerationCategory>,
}

/// Categories of content moderation.
pub enum ModerationCategory {
    /// Hate category with a confidence score.
    Hate {
        /// Confidence score for hate content.
        score: f32,
    },
    /// Harassment category with a confidence score.
    Harassment {
        /// Confidence score for harassment content.
        score: f32,
    },
    /// Sexual category with a confidence score.
    Sexual {
        /// Confidence score for sexual content.
        score: f32,
    },
    /// Violence category with a confidence score.
    Violence {
        /// Confidence score for violence content.
        score: f32,
    },
    /// Self-harm category with a confidence score.
    SelfHarm {
        /// Confidence score for self-harm content.
        score: f32,
    },
}
