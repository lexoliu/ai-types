use alloc::{string::String, vec::Vec};
use futures_core::Stream;

/// Audio data type alias for raw audio bytes.
pub type Data = Vec<u8>;

/// Trait for AI models that can generate audio from text prompts.
///
/// This trait provides a unified interface for text-to-speech and audio generation
/// models across different AI service providers.
///
/// # Examples
///
/// ```rust
/// use ai_types::AudioGenerator;
/// use futures_core::Stream;
///
/// struct MyAudioGenerator;
///
/// impl AudioGenerator for MyAudioGenerator {
///     fn generate(&self, prompt: &str) -> impl Stream<Item = ai_types::audio::Data> + Send {
///         // Implementation would stream audio data
///         futures_lite::stream::iter(vec![vec![0u8; 1024]]) // Dummy data
///     }
/// }
/// ```
pub trait AudioGenerator {
    /// Generates audio from a text prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to convert to audio
    ///
    /// # Returns
    ///
    /// A stream of audio data chunks as byte vectors.
    fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send;
}

/// Trait for AI models that can transcribe audio to text.
///
/// This trait provides a unified interface for speech-to-text and audio transcription
/// models across different AI service providers.
///
/// # Examples
///
/// ```rust
/// use ai_types::AudioTranscriber;
/// use futures_core::Stream;
///
/// struct MyAudioTranscriber;
///
/// impl AudioTranscriber for MyAudioTranscriber {
///     fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send {
///         // Implementation would transcribe audio data
///         futures_lite::stream::iter(vec!["Hello world".to_string()]) // Dummy transcription
///     }
/// }
/// ```
pub trait AudioTranscriber {
    /// Transcribes audio data to text.
    ///
    /// # Arguments
    ///
    /// * `audio` - The raw audio data to transcribe
    ///
    /// # Returns
    ///
    /// A stream of transcribed text chunks.
    fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send;
}
