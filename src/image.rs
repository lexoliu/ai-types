use alloc::{string::String, vec::Vec};
use futures_core::Stream;

/// Image data as bytes.
///
/// Type alias for [`Vec<u8>`] representing image data.
pub type Data = Vec<u8>;

/// Trait for generating and editing images from prompts and masks.
pub trait ImageGenerator {
    /// The error type returned by the image generator.
    type Error: core::error::Error + Send + Sync;

    /// Create an image from a prompt and a specified size.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt containing text and optional images.
    /// * `size` - The desired size of the generated image.
    ///
    /// # Returns
    ///
    /// A stream of image data chunks or errors.
    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send;

    /// Edit an image using a prompt and a mask.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt containing text and optional images.
    /// * `mask` - The mask to apply to the image data.
    ///
    /// # Returns
    ///
    /// A stream of edited image data chunks or errors.
    fn edit(
        &self,
        prompt: Prompt,
        mask: &[u8],
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send;
}

/// Represents a prompt for image generation, including text and optional images.
pub struct Prompt {
    /// The text description for the image generation.
    pub text: String,
    /// Optional images to guide the generation process.
    pub image: Vec<Data>,
}

impl Prompt {
    /// Creates a new `Prompt` with the given text
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            image: Vec::new(),
        }
    }

    /// Adds an image to the prompt and returns the updated `Prompt`.
    ///
    /// # Arguments
    ///
    /// * `image` - The image data to add to the prompt.
    #[must_use]
    pub fn with_image(mut self, image: Data) -> Self {
        self.image.push(image);
        self
    }
}

impl From<String> for Prompt {
    /// Converts a `String` into a `Prompt`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to use for the prompt.
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for Prompt {
    /// Converts a `&str` into a `Prompt`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to use for the prompt.
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

/// Represents the size (width and height) of an image.
pub struct Size {
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
}

impl Size {
    /// Creates a new `Size` with the given width and height.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the image in pixels.
    /// * `height` - The height of the image in pixels.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Creates a new `Size` with equal width and height (a square).
    ///
    /// # Arguments
    ///
    /// * `size` - The width and height of the square image in pixels.
    #[must_use]
    pub const fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;
    use alloc::vec;
    use futures_lite::StreamExt;

    struct MockImageGenerator;

    impl ImageGenerator for MockImageGenerator {
        type Error = Infallible;
        fn create(
            &self,
            prompt: Prompt,
            _size: Size,
        ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
            // Create mock image data based on prompt
            let prompt_bytes = prompt.text.as_bytes();
            let chunk1 = prompt_bytes.to_vec();
            let chunk2 = vec![0xFF, 0xD8, 0xFF, 0xE0]; // Mock JPEG header
            let chunk3 = vec![0x00; 100]; // Mock image data

            futures_lite::stream::iter(vec![chunk1, chunk2, chunk3].into_iter().map(Ok))
        }

        fn edit(
            &self,
            prompt: Prompt,
            _mask: &[u8],
        ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
            // Create mock image data based on prompt
            let prompt_bytes = prompt.text.as_bytes();
            let chunk1 = prompt_bytes.to_vec();
            let chunk2 = vec![0xFF, 0xD8, 0xFF, 0xE0]; // Mock JPEG header
            let chunk3 = vec![0x00; 100]; // Mock image data

            futures_lite::stream::iter(vec![chunk1, chunk2, chunk3].into_iter().map(Ok))
        }
    }

    #[tokio::test]
    async fn test_image_generation() {
        let generator = MockImageGenerator;
        let mut stream = generator.create(Prompt::new("a cat"), Size::square(256));

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"a cat".to_vec());
        assert_eq!(chunks[1], vec![0xFF, 0xD8, 0xFF, 0xE0]);
        assert_eq!(chunks[2], vec![0x00; 100]);
    }

    #[tokio::test]
    async fn test_image_generation_empty_prompt() {
        let generator = MockImageGenerator;
        let mut stream = generator.create(Prompt::new(""), Size::square(256));

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
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
        let mut stream = generator.create(Prompt::new(long_prompt), Size::square(512));

        let mut total_bytes = 0;
        while let Some(chunk) = stream.next().await {
            total_bytes += chunk.unwrap().len();
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
