//! Message types for AI language model conversations.
//!
//! This module provides types for representing messages in conversations with AI language models.
//! Messages can have different roles (User, Assistant, System, Tool), contain text content,
//! and optionally include attachments and annotations.
//!
//! # Examples
//!
//! ## Creating basic messages
//!
//! ```rust
//! use ai_types::llm::{Message, Role};
//!
//! // Using convenience constructors
//! let user_msg = Message::user("Hello, how are you?");
//! let assistant_msg = Message::assistant("I'm doing well, thank you!");
//! let system_msg = Message::system("You are a helpful assistant.");
//!
//! // Using the general constructor
//! let tool_msg = Message::new(Role::Tool, "Tool executed successfully".into());
//! ```
//!
//! ## Adding attachments
//!
//! ```rust
//! use ai_types::llm::Message;
//! use url::Url;
//!
//! let message = Message::user("Check out this image")
//!     .with_attachment("https://example.com/image.jpg".parse::<Url>().unwrap());
//!
//! let urls = vec![
//!     "https://example.com/doc1.pdf".parse::<Url>().unwrap(),
//!     "https://example.com/doc2.pdf".parse::<Url>().unwrap(),
//! ];
//! let message_with_multiple = Message::user("Review these documents")
//!     .with_attachments(urls);
//! ```
//!
//! ## Working with annotations
//!
//! ```rust
//! use ai_types::llm::{Message, Annotation, UrlAnnotation};
//! use url::Url;
//!
//! let url_annotation = UrlAnnotation::new(
//!     "https://example.com".parse().unwrap(),
//!     "Example Site".into(),
//!     "A useful example website".into(),
//!     6,  // start index of URL in content
//!     25, // end index of URL in content
//! );
//!
//! let message = Message::user("Visit https://example.com for examples")
//!     .with_annotation(Annotation::url(url_annotation));
//! ```

use core::fmt::Debug;

use alloc::{string::String, vec::Vec};
use url::Url;

/// Conversation participant role.
///
/// Defines the role of a message sender in a conversation.
/// Each role has specific semantics and is typically handled differently
/// by AI language models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// User message.
    ///
    /// Represents input from a human user or client application.
    User,
    /// AI assistant message.
    ///
    /// Represents responses from the AI assistant.
    Assistant,
    /// System message for context/instructions.
    ///
    /// Provides context, instructions, or system-level information
    /// that guides the AI's behavior.
    System,
    /// Tool/function call message.
    ///
    /// Represents output from external tools or function calls.
    Tool,
}

/// A message in a conversation.
///
/// Contains a [`Role`], text content, and optional attachments and annotations.
/// Messages form the building blocks of conversations with AI language models.
///
/// # Fields
///
/// * `role` - The role of the message sender (User, Assistant, System, or Tool)
/// * `content` - The text content of the message
/// * `attachments` - Optional URLs to external resources (images, documents, etc.)
/// * `annotation` - Optional metadata annotations for URLs mentioned in the content
///
/// # Example
///
/// ```rust
/// use ai_types::llm::{Message, Role};
/// use url::Url;
///
/// // Create messages using convenience constructors
/// let user_msg = Message::user("Hello, how are you?");
/// let system_msg = Message::system("You are a helpful assistant.");
/// let custom_msg = Message::new(Role::Assistant, "I'm doing well!".to_string());
///
/// // Add attachments to a message
/// let msg_with_attachment = Message::user("Check out this image")
///     .with_attachment("https://example.com/image.jpg".parse::<Url>().unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct Message {
    /// Message sender role.
    pub role: Role,
    /// Message text content.
    pub content: String,
    /// Attachment URLs.
    ///
    /// URLs to external resources like images, documents, or other media
    /// that are referenced by this message.
    pub attachments: Vec<Url>,
    /// Message annotations. See [`Annotation`] for details.
    ///
    /// Metadata annotations for URLs mentioned in the message content,
    /// providing additional context like titles and descriptions.
    pub annotation: Vec<Annotation>,
}

/// URL annotation metadata.
///
/// Contains metadata about a [`url::Url`] referenced in a [`Message`].
/// This provides additional context about URLs mentioned in message content,
/// such as their title, description, and position within the text.
///
/// # Fields
///
/// * `url` - The annotated URL
/// * `title` - Human-readable title of the URL resource
/// * `content` - Description or summary of the URL content  
/// * `start` - Start character index of the URL in the message content
/// * `end` - End character index of the URL in the message content
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlAnnotation {
    /// The annotated URL.
    pub url: Url,
    /// URL title.
    pub title: String,
    /// URL content/description.
    pub content: String,
    /// Start index in message content.
    pub start: usize,
    /// End index in message content.
    pub end: usize,
}

impl UrlAnnotation {
    /// Creates a new URL annotation.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL being annotated
    /// * `title` - Human-readable title for the URL
    /// * `content` - Description or summary of the URL content
    /// * `start` - Start character index in the message content
    /// * `end` - End character index in the message content
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_types::llm::UrlAnnotation;
    /// use url::Url;
    ///
    /// let annotation = UrlAnnotation::new(
    ///     "https://example.com".parse().unwrap(),
    ///     "Example Site".into(),
    ///     "An example website".into(),
    ///     0,
    ///     10
    /// );
    /// ```
    pub fn new(url: Url, title: String, content: String, start: usize, end: usize) -> Self {
        Self {
            url,
            title,
            content,
            start,
            end,
        }
    }
}

/// Message annotation.
///
/// Provides additional metadata for [`Message`] content.
/// Currently supports URL annotations, but can be extended
/// to support other types of annotations in the future.
///
/// # Variants
///
/// * `Url` - Annotation for a URL mentioned in the message content
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Annotation {
    /// URL annotation. See [`UrlAnnotation`].
    Url(UrlAnnotation),
}

impl Message {
    /// Creates a new message with the specified role and content.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender
    /// * `content` - The text content of the message
    pub const fn new(role: Role, content: String) -> Self {
        Self {
            role,
            content,
            attachments: Vec::new(),
            annotation: Vec::new(),
        }
    }

    /// Creates a new user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content.into())
    }

    /// Creates a new assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content.into())
    }

    /// Creates a new system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content.into())
    }

    /// Creates a new tool message.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(Role::Tool, content.into())
    }

    /// Adds an attachment URL to the message.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to attach
    pub fn with_attachment<U: TryInto<Url, Error: Debug>>(mut self, url: U) -> Self {
        self.attachments.push(url.try_into().unwrap());
        self
    }

    /// Adds multiple attachment URLs to the message.
    ///
    /// # Arguments
    ///
    /// * `urls` - An iterable of URLs to attach
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_types::llm::Message;
    /// use url::Url;
    ///
    /// let urls = vec![
    ///     "https://example.com".parse::<Url>().unwrap(),
    ///     "https://example.org".parse::<Url>().unwrap(),
    /// ];
    /// let message = Message::user("Check these links").with_attachments(urls);
    /// ```
    pub fn with_attachments<U: TryInto<Url, Error: Debug>>(
        mut self,
        urls: impl IntoIterator<Item = U>,
    ) -> Self {
        self.attachments
            .extend(urls.into_iter().map(|url| url.try_into().unwrap()));
        self
    }

    /// Adds an annotation to the message.
    ///
    /// # Arguments
    ///
    /// * `annotation` - The annotation to add
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_types::llm::{Message, Annotation, UrlAnnotation};
    /// use url::Url;
    ///
    /// let url_annotation = UrlAnnotation {
    ///     url: "https://example.com".parse().unwrap(),
    ///     title: "Example Site".into(),
    ///     content: "An example website".into(),
    ///     start: 0,
    ///     end: 10,
    /// };
    ///
    /// let message = Message::user("Visit https://example.com")
    ///     .with_annotation(Annotation::Url(url_annotation));
    /// ```
    pub fn with_annotation(mut self, annotation: Annotation) -> Self {
        self.annotation.push(annotation);
        self
    }

    /// Adds multiple annotations to the message.
    ///
    /// # Arguments
    ///
    /// * `annotations` - An iterable of annotations to add
    pub fn with_annotations(mut self, annotations: impl IntoIterator<Item = Annotation>) -> Self {
        self.annotation.extend(annotations);
        self
    }
}

impl Annotation {
    /// Creates a new URL annotation from a `UrlAnnotation`.
    ///
    /// # Arguments
    ///
    /// * `url_annotation` - The URL annotation to wrap
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_types::llm::{Annotation, UrlAnnotation};
    /// use url::Url;
    ///
    /// let url_annotation = UrlAnnotation::new(
    ///     "https://example.com".parse().unwrap(),
    ///     "Example".into(),
    ///     "Example content".into(),
    ///     0,
    ///     10
    /// );
    ///
    /// let annotation = Annotation::url(url_annotation);
    /// ```
    pub fn url(url_annotation: UrlAnnotation) -> Self {
        Self::Url(url_annotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_eq!(Role::Assistant, Role::Assistant);
        assert_eq!(Role::System, Role::System);
        assert_eq!(Role::Tool, Role::Tool);

        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::System, Role::Tool);
    }

    #[test]
    fn test_message_creation() {
        let message = Message::new(Role::User, "Hello".into());
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content, "Hello");
        assert!(message.attachments.is_empty());
        assert!(message.annotation.is_empty());
    }

    #[test]
    fn test_message_convenience_constructors() {
        let user_msg = Message::user("User message");
        assert_eq!(user_msg.role, Role::User);
        assert_eq!(user_msg.content, "User message");

        let assistant_msg = Message::assistant("Assistant message");
        assert_eq!(assistant_msg.role, Role::Assistant);
        assert_eq!(assistant_msg.content, "Assistant message");

        let system_msg = Message::system("System message");
        assert_eq!(system_msg.role, Role::System);
        assert_eq!(system_msg.content, "System message");

        let tool_msg = Message::tool("Tool message");
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.content, "Tool message");
    }

    #[test]
    fn test_message_with_attachment() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let message = Message::user("Hello").with_attachment(url.clone());

        assert_eq!(message.attachments.len(), 1);
        assert_eq!(message.attachments[0], url);
    }

    #[test]
    fn test_message_with_multiple_attachments() {
        let urls = vec![
            "https://example.com".parse::<Url>().unwrap(),
            "https://example.org".parse::<Url>().unwrap(),
        ];

        let message = Message::user("Hello").with_attachments(urls.clone());

        assert_eq!(message.attachments.len(), 2);
        assert_eq!(message.attachments, urls);
    }

    #[test]
    fn test_url_annotation() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let annotation = UrlAnnotation {
            url: url.clone(),
            title: "Example".into(),
            content: "Example content".into(),
            start: 0,
            end: 10,
        };

        assert_eq!(annotation.url, url);
        assert_eq!(annotation.title, "Example");
        assert_eq!(annotation.content, "Example content");
        assert_eq!(annotation.start, 0);
        assert_eq!(annotation.end, 10);
    }

    #[test]
    fn test_annotation_enum() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let url_annotation = UrlAnnotation {
            url,
            title: "Example".into(),
            content: "Example content".into(),
            start: 0,
            end: 10,
        };

        let annotation = Annotation::Url(url_annotation.clone());

        match annotation {
            Annotation::Url(url_anno) => {
                assert_eq!(url_anno.title, url_annotation.title);
                assert_eq!(url_anno.content, url_annotation.content);
            }
        }
    }

    #[test]
    fn test_message_debug() {
        let message = Message::user("Test message");
        let debug_str = alloc::format!("{message:?}");
        assert!(debug_str.contains("User"));
        assert!(debug_str.contains("Test message"));
    }

    #[test]
    fn test_message_clone() {
        let original = Message::user("Original message");
        let cloned = original.clone();

        assert_eq!(original.role, cloned.role);
        assert_eq!(original.content, cloned.content);
        assert_eq!(original.attachments.len(), cloned.attachments.len());
        assert_eq!(original.annotation.len(), cloned.annotation.len());
    }

    #[test]
    fn test_message_with_annotation() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let url_annotation = UrlAnnotation::new(
            url.clone(),
            "Example".into(),
            "Example content".into(),
            0,
            10,
        );

        let message = Message::user("Visit https://example.com")
            .with_annotation(Annotation::url(url_annotation.clone()));

        assert_eq!(message.annotation.len(), 1);
        match &message.annotation[0] {
            Annotation::Url(annotation) => {
                assert_eq!(annotation.url, url);
                assert_eq!(annotation.title, "Example");
                assert_eq!(annotation.content, "Example content");
                assert_eq!(annotation.start, 0);
                assert_eq!(annotation.end, 10);
            }
        }
    }

    #[test]
    fn test_message_with_multiple_annotations() {
        let url1 = "https://example.com".parse::<Url>().unwrap();
        let url2 = "https://example.org".parse::<Url>().unwrap();

        let annotations = vec![
            Annotation::url(UrlAnnotation::new(
                url1,
                "Example 1".into(),
                "First example".into(),
                0,
                10,
            )),
            Annotation::url(UrlAnnotation::new(
                url2,
                "Example 2".into(),
                "Second example".into(),
                20,
                30,
            )),
        ];

        let message = Message::user("Visit these sites").with_annotations(annotations);

        assert_eq!(message.annotation.len(), 2);
    }

    #[test]
    fn test_url_annotation_constructor() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let annotation = UrlAnnotation::new(
            url.clone(),
            "Test Title".into(),
            "Test Content".into(),
            5,
            15,
        );

        assert_eq!(annotation.url, url);
        assert_eq!(annotation.title, "Test Title");
        assert_eq!(annotation.content, "Test Content");
        assert_eq!(annotation.start, 5);
        assert_eq!(annotation.end, 15);
    }

    #[test]
    fn test_annotation_url_constructor() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let url_annotation =
            UrlAnnotation::new(url.clone(), "Test".into(), "Test content".into(), 0, 5);

        let annotation = Annotation::url(url_annotation.clone());

        match annotation {
            Annotation::Url(inner) => {
                assert_eq!(inner.url, url_annotation.url);
                assert_eq!(inner.title, url_annotation.title);
            }
        }
    }
}
