use alloc::{string::String, vec::Vec};
use futures_core::Stream;

/// Audio data as bytes.
///
/// Type alias for [`Vec<u8>`] representing raw audio data.
pub type Data = Vec<u8>;

/// Generates audio from text prompts.
/// # Example
///
/// ```rust
/// use ai_types::AudioGenerator;
/// use futures_core::Stream;
///
/// struct MyAudioGen;
///
/// impl AudioGenerator for MyAudioGen {
///     fn generate(&self, prompt: &str) -> impl Stream<Item = ai_types::audio::Data> + Send {
///         futures_lite::stream::iter(vec![vec![0u8; 1024]])
///     }
/// }
/// ```
pub trait AudioGenerator {
    /// Generates audio from text prompt.
    ///
    /// Returns a [`Stream`] of [`Data`] chunks.
    fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send;
}

/// Transcribes audio to text.
///
/// # Example
///
/// ```rust
/// use ai_types::AudioTranscriber;
/// use futures_core::Stream;
///
/// struct MyTranscriber;
///
/// impl AudioTranscriber for MyTranscriber {
///     fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send {
///         futures_lite::stream::iter(vec!["Hello world".to_string()])
///     }
/// }
/// ```
pub trait AudioTranscriber {
    /// Transcribes audio data to text.
    ///
    /// Returns a [`Stream`] of transcribed text chunks.
    fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send;
}
