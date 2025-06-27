use alloc::vec::Vec;
use futures_core::Stream;

/// Type alias for image data as a vector of bytes.
pub type Data = Vec<u8>;

/// Trait for image generation models that can create images from text prompts.
///
/// # Example
///
/// ```rust
/// use ai_types::ImageGenerator;
/// use futures_core::Stream;
///
/// struct MyImageGen;
///
/// impl ImageGenerator for MyImageGen {
///     fn generate(&self, prompt: &str) -> impl Stream<Item = ai_types::image::Data> + Send {
///         // Implementation would call actual image generation service
///         futures_lite::stream::iter(vec![vec![0u8; 1024]]) // Mock data
///     }
/// }
/// ```
pub trait ImageGenerator {
    /// Generates an image from a text prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text description of the image to generate
    ///
    /// # Returns
    ///
    /// A stream of image data chunks. The stream may yield multiple chunks
    /// as the image is being generated, allowing for progressive loading.
    fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use futures_lite::StreamExt;

    struct MockImageGenerator;

    impl ImageGenerator for MockImageGenerator {
        fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send {
            // Create mock image data based on prompt
            let prompt_bytes = prompt.as_bytes();
            let chunk1 = prompt_bytes.to_vec();
            let chunk2 = vec![0xFF, 0xD8, 0xFF, 0xE0]; // Mock JPEG header
            let chunk3 = vec![0x00; 100]; // Mock image data

            futures_lite::stream::iter(vec![chunk1, chunk2, chunk3])
        }
    }

    #[tokio::test]
    async fn test_image_generation() {
        let generator = MockImageGenerator;
        let mut stream = generator.generate("a cat");

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"a cat".to_vec());
        assert_eq!(chunks[1], vec![0xFF, 0xD8, 0xFF, 0xE0]);
        assert_eq!(chunks[2], vec![0x00; 100]);
    }

    #[tokio::test]
    async fn test_image_generation_empty_prompt() {
        let generator = MockImageGenerator;
        let mut stream = generator.generate("");

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"".to_vec());
        assert_eq!(chunks[1], vec![0xFF, 0xD8, 0xFF, 0xE0]);
        assert_eq!(chunks[2], vec![0x00; 100]);
    }

    #[tokio::test]
    async fn test_image_generation_long_prompt() {
        let generator = MockImageGenerator;
        let long_prompt = "a very detailed and elaborate description of a beautiful landscape with mountains, rivers, and forests";
        let mut stream = generator.generate(long_prompt);

        let mut total_bytes = 0;
        while let Some(chunk) = stream.next().await {
            total_bytes += chunk.len();
        }

        // Should have prompt bytes + header bytes + 100 mock data bytes
        assert_eq!(total_bytes, long_prompt.len() + 4 + 100);
    }

    #[tokio::test]
    async fn test_data_type_alias() {
        let data: Data = vec![1, 2, 3, 4];
        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 1);
        assert_eq!(data[3], 4);
    }

    #[test]
    fn test_data_operations() {
        let mut data: Data = vec![0xFF; 1024];
        assert_eq!(data.len(), 1024);

        data.push(0x00);
        assert_eq!(data.len(), 1025);
        assert_eq!(data[1024], 0x00);

        data.extend_from_slice(&[0x01, 0x02]);
        assert_eq!(data.len(), 1027);
        assert_eq!(data[1025], 0x01);
        assert_eq!(data[1026], 0x02);
    }
}
