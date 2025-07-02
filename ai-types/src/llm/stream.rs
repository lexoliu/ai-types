use core::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Poll, ready},
};

use alloc::string::String;
use futures_core::Stream;
use pin_project_lite::pin_project;

/// A trait for streaming text responses from language models.
///
/// `TextStream` provides a unified interface for handling streaming text data from AI models.
/// It combines the functionality of `Stream` (for processing chunks as they arrive) and
/// `IntoFuture` (for collecting the complete response into a single string).
///
/// ## Key Features
///
/// - **Dual Interface**: Both streaming (`Stream`) and batch (`IntoFuture`) processing
/// - **Error Handling**: Type-safe error propagation throughout the stream
/// - **Composable**: Can be easily integrated with other async/streaming patterns
/// - **Provider Agnostic**: Works with any text streaming implementation
///
/// ## Usage Patterns
///
/// ### Real-time Processing (Stream Interface)
///
/// Process text chunks as they arrive, useful for real-time display or immediate processing:
///
/// ```rust
/// use ai_types::llm::{TextStream, LanguageModel};
/// use futures_lite::StreamExt;
///
/// async fn display_as_generated<S: TextStream>(mut stream: S) -> Result<String, S::Error> {
///     let mut complete_text = String::new();
///     while let Some(chunk) = stream.next().await {
///         let text = chunk?;
///         print!("{}", text); // Display immediately
///         complete_text.push_str(&text);
///     }
///     Ok(complete_text)
/// }
/// ```
///
/// ### Batch Collection (`IntoFuture` Interface)
///
/// Collect the complete response when you need the full text:
///
/// ```rust
/// use ai_types::llm::{TextStream, LanguageModel, Request, Message};
///
/// async fn get_complete_answer<M: LanguageModel>(model: M) -> ai_types::Result {
///     let request = Request::new([Message::user("What is Rust?")]);
///     let stream = model.respond(request);
///     
///     // Collect everything into a single string
///     let answer = stream.await?;
///     Ok(answer)
/// }
/// ```
///
/// ## Implementation Notes
///
/// Types implementing `TextStream` should ensure that:
/// - Text chunks are delivered in order
/// - Empty chunks are handled gracefully  
/// - The stream terminates properly on completion or error
/// - Buffer management is efficient for memory usage
pub trait TextStream:
    Stream<Item = Result<String, Self::Error>>
    + Send
    + Unpin
    + IntoFuture<Output = Self::Item, IntoFuture: Send>
{
    /// The error type produced by this text stream.
    type Error: core::error::Error + Send + Sync + 'static;
}

impl<T, E> TextStream for T
where
    T: Stream<Item = Result<String, E>>
        + Send
        + Unpin
        + IntoFuture<Output = Self::Item, IntoFuture: Send>,
    E: core::error::Error + Send + Sync + 'static,
{
    type Error = E;
}

pin_project! {
    struct TextStreamAdapter<S> {
        #[pin]
        stream:  S,
    }
}

impl<S, E> Stream for TextStreamAdapter<S>
where
    S: Stream<Item = Result<String, E>> + Send + Unpin,
    E: core::error::Error + Send + Sync + 'static,
{
    type Item = Result<String, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().stream.poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

impl<S, E> IntoFuture for TextStreamAdapter<S>
where
    S: Stream<Item = Result<String, E>> + Send + Unpin,
    E: core::error::Error + Send + Sync + 'static,
{
    type Output = Result<String, E>;
    type IntoFuture = TextStreamAdapterFuture<S>;

    fn into_future(self) -> Self::IntoFuture {
        TextStreamAdapterFuture {
            stream: self.stream,
            buffer: String::new(),
        }
    }
}

pin_project! {
    struct TextStreamAdapterFuture<S> {
        #[pin]
        stream: S,
        buffer: String,
    }
}

impl<S, E> Future for TextStreamAdapterFuture<S>
where
    S: Stream<Item = Result<String, E>> + Send + Unpin,
    E: core::error::Error + Send + Sync + 'static,
{
    type Output = Result<String, E>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> core::task::Poll<Self::Output> {
        loop {
            let this = self.as_mut().project();
            let result = ready!(this.stream.poll_next(cx));
            match result {
                Some(Ok(chunk)) => {
                    this.buffer.push_str(&chunk);
                    // Continue polling to collect more chunks
                }
                Some(Err(e)) => {
                    return Poll::Ready(Err(e));
                }
                None => {
                    // Stream ended, return accumulated buffer
                    let this = self.project();
                    return Poll::Ready(Ok(core::mem::take(this.buffer)));
                }
            }
        }
    }
}

/// Converts any `Stream<Item = Result<String, E>>` into a `TextStream`.
///
/// This utility function wraps an existing stream to provide the `TextStream` interface,
/// enabling both chunk-by-chunk processing and complete text collection via `IntoFuture`.
///
/// # Parameters
///
/// - `stream`: Any stream that yields `Result<String, E>` items
///
/// # Returns
///
/// A `TextStream` implementation that can be used with both streaming and batch patterns.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::stream::text_stream;
/// use futures_lite::{stream, StreamExt};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let chunks = vec!["Hello, ", "world!"];
/// let chunk_stream = stream::iter(chunks)
///     .map(|s| Ok::<String, std::io::Error>(s.to_string()));
///
/// let text_stream = text_stream(chunk_stream);
/// let complete_text = text_stream.await?;
/// assert_eq!(complete_text, "Hello, world!");
/// # Ok(())
/// # }
/// ```
pub fn text_stream<S, E>(stream: S) -> impl TextStream<Error = E>
where
    S: Stream<Item = Result<String, E>> + Send + Unpin + 'static,
    E: core::error::Error + Send + Sync + 'static,
{
    TextStreamAdapter { stream }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString, vec, vec::Vec};
    use futures_lite::{StreamExt, stream};

    // Simple error type for testing
    #[derive(Debug, Clone, PartialEq)]
    struct TestError(&'static str);

    impl core::fmt::Display for TestError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl core::error::Error for TestError {}

    #[tokio::test]
    async fn test_text_stream_adapter_streaming() {
        // Test streaming interface - process chunks one by one
        let chunks = vec!["Hello, ", "world", "!"];
        let chunk_stream =
            stream::iter(chunks.clone()).map(|s| Ok::<String, TestError>(s.to_string()));

        let mut text_stream = text_stream(chunk_stream);
        let mut collected_chunks = Vec::new();

        while let Some(chunk_result) = text_stream.next().await {
            let chunk = chunk_result.unwrap();
            collected_chunks.push(chunk);
        }

        assert_eq!(collected_chunks, chunks);
    }

    #[tokio::test]
    async fn test_text_stream_adapter_into_future() {
        // Test IntoFuture interface - collect complete response
        let chunks = vec!["Hello, ", "world", "!"];
        let expected = "Hello, world!";

        let chunk_stream = stream::iter(chunks).map(|s| Ok::<String, TestError>(s.to_string()));

        let text_stream = text_stream(chunk_stream);
        let complete_text = text_stream.await.unwrap();

        assert_eq!(complete_text, expected);
    }

    #[tokio::test]
    async fn test_text_stream_empty() {
        // Test empty stream
        let empty_stream =
            stream::iter(Vec::<&str>::new()).map(|s| Ok::<String, TestError>(s.to_string()));

        let text_stream = text_stream(empty_stream);
        let complete_text = text_stream.await.unwrap();

        assert_eq!(complete_text, "");
    }

    #[tokio::test]
    async fn test_text_stream_single_chunk() {
        // Test single chunk
        let chunks = vec!["Single chunk"];
        let chunk_stream = stream::iter(chunks).map(|s| Ok::<String, TestError>(s.to_string()));

        let text_stream = text_stream(chunk_stream);
        let complete_text = text_stream.await.unwrap();

        assert_eq!(complete_text, "Single chunk");
    }

    #[tokio::test]
    async fn test_text_stream_error_propagation() {
        // Test error handling
        let chunks = vec![
            Ok("Good chunk".to_string()),
            Err(TestError("Error occurred")),
        ];
        let chunk_stream = stream::iter(chunks);

        let mut text_stream = text_stream(chunk_stream);

        // First chunk should succeed
        let first_chunk = text_stream.next().await.unwrap().unwrap();
        assert_eq!(first_chunk, "Good chunk");

        // Second chunk should return error
        let error_result = text_stream.next().await.unwrap();
        assert!(error_result.is_err());
        assert_eq!(error_result.unwrap_err(), TestError("Error occurred"));
    }

    #[tokio::test]
    async fn test_text_stream_error_in_future() {
        // Test error handling with IntoFuture
        let chunks = vec![Ok("Good".to_string()), Err(TestError("Bad"))];
        let chunk_stream = stream::iter(chunks);

        let text_stream = text_stream(chunk_stream);
        let result = text_stream.await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TestError("Bad"));
    }

    #[tokio::test]
    async fn test_text_stream_large_chunks() {
        // Test with larger text chunks
        let large_chunk = "A".repeat(1000);
        let small_chunk = "B".repeat(500);

        let chunk_stream =
            stream::iter([large_chunk.clone(), small_chunk.clone()]).map(Ok::<String, TestError>);

        let text_stream = text_stream(chunk_stream);
        let complete_text = text_stream.await.unwrap();

        let expected = format!("{large_chunk}{small_chunk}");
        assert_eq!(complete_text, expected);
        assert_eq!(complete_text.len(), 1500);
    }

    #[tokio::test]
    async fn test_text_stream_unicode() {
        // Test with Unicode characters
        let chunks = vec!["Hello ", "世界", "! 🦀"];
        let expected = "Hello 世界! 🦀";

        let chunk_stream = stream::iter(chunks).map(|s| Ok::<String, TestError>(s.to_string()));

        let text_stream = text_stream(chunk_stream);
        let complete_text = text_stream.await.unwrap();

        assert_eq!(complete_text, expected);
    }

    #[tokio::test]
    async fn test_text_stream_multiple_awaits() {
        // Test that multiple streams can be created and awaited independently
        let create_stream = || {
            let chunks = vec!["test", "data"];
            let chunk_stream = stream::iter(chunks).map(|s| Ok::<String, TestError>(s.to_string()));
            text_stream(chunk_stream)
        };

        let stream1 = create_stream();
        let stream2 = create_stream();

        let result1 = stream1.await;
        let result2 = stream2.await;

        assert_eq!(result1.unwrap(), "testdata");
        assert_eq!(result2.unwrap(), "testdata");
    }
}
