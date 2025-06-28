use alloc::vec::Vec;
use core::future::Future;

/// Converts text to vector representations.
///
/// # Example
///
/// ```rust
/// use ai_types::EmbeddingModel;
///
/// struct MyEmbedding;
///
/// impl EmbeddingModel for MyEmbedding {
///     fn dim(&self) -> usize { 768 }
///     
///     async fn embed(&self, text: &str) -> Vec<f32> {
///         vec![0.0; self.dim()] // Implementation calls embedding service
///     }
/// }
/// ```
pub trait EmbeddingModel {
    /// Embedding vector dimension.
    fn dim(&self) -> usize;

    /// Converts text to embedding vector.
    ///
    /// Returns a [`Vec<f32>`] with length equal to [`Self::dim`].
    fn embed(&self, text: &str) -> impl Future<Output = crate::Result<Vec<f32>>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    struct MockEmbeddingModel {
        dimension: usize,
    }

    impl EmbeddingModel for MockEmbeddingModel {
        fn dim(&self) -> usize {
            self.dimension
        }

        async fn embed(&self, text: &str) -> crate::Result<Vec<f32>> {
            // Create a simple mock embedding based on text length
            let mut embedding = vec![0.0; self.dimension];
            let text_len = text.len();

            for (i, value) in embedding.iter_mut().enumerate() {
                *value = (text_len + i) as f32 * 0.01;
            }

            Ok(embedding)
        }
    }

    #[tokio::test]
    async fn test_embedding_model_dimension() {
        let model = MockEmbeddingModel { dimension: 768 };
        assert_eq!(model.dim(), 768);
    }

    #[tokio::test]
    async fn test_embedding_generation() {
        let model = MockEmbeddingModel { dimension: 4 };
        let embedding = model.embed("test").await.unwrap();

        assert_eq!(embedding.len(), 4);
        assert!((embedding[0] - 0.04).abs() < f32::EPSILON); // text length 4 + index 0 = 4 * 0.01
        assert!((embedding[1] - 0.05).abs() < f32::EPSILON); // text length 4 + index 1 = 5 * 0.01
        assert!((embedding[2] - 0.06).abs() < f32::EPSILON); // text length 4 + index 2 = 6 * 0.01
        assert!((embedding[3] - 0.07).abs() < f32::EPSILON); // text length 4 + index 3 = 7 * 0.01
    }

    #[tokio::test]
    async fn test_embedding_different_texts() {
        let model = MockEmbeddingModel { dimension: 2 };

        let embedding1 = model.embed("a").await.unwrap();
        let embedding2 = model.embed("ab").await.unwrap();

        assert_eq!(embedding1.len(), 2);
        assert_eq!(embedding2.len(), 2);

        // Different text lengths should produce different embeddings
        assert_ne!(embedding1[0], embedding2[0]);
        assert_ne!(embedding1[1], embedding2[1]);
    }

    #[tokio::test]
    async fn test_embedding_empty_text() {
        let model = MockEmbeddingModel { dimension: 3 };
        let embedding = model.embed("").await.unwrap();

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0], 0.00); // length 0 + index 0 = 0 * 0.01
        assert_eq!(embedding[1], 0.01); // length 0 + index 1 = 1 * 0.01
        assert_eq!(embedding[2], 0.02); // length 0 + index 2 = 2 * 0.01
    }

    #[tokio::test]
    async fn test_embedding_large_dimension() {
        let model = MockEmbeddingModel { dimension: 1536 }; // Common OpenAI dimension
        let embedding = model.embed("test text").await.unwrap();

        assert_eq!(embedding.len(), 1536);
        assert!((embedding[0] - 0.09).abs() < f32::EPSILON); // text length 9 + index 0 = 9 * 0.01
        assert!((embedding[1535] - 15.44).abs() < 0.01); // text length 9 + index 1535 = 1544 * 0.01
    }
}
