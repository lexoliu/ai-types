use core::fmt::Debug;

use alloc::{string::String, vec::Vec};
use url::Url;

/// Represents the role of a participant in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A message from the user/human.
    User,
    /// A message from the AI assistant.
    Assistant,
    /// A system message providing context or instructions.
    System,
    /// A message from a tool or function call.
    Tool,
}

/// Represents a single message in a conversation.
///
/// A message contains the role of the sender, text content, and optionally attached images.
///
/// # Example
///
/// ```rust
/// use ai_types::llm::{Message, Role};
///
/// let user_msg = Message::user("Hello, how are you?".to_string());
/// let system_msg = Message::system("You are a helpful assistant.".to_string());
///
/// // Message with custom role
/// let custom_msg = Message::new(Role::Assistant, "I'm doing well, thank you!".to_string());
/// ```
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
    /// The list of attachment URLs associated with the message.
    pub attachments: Vec<Url>,
    /// The list of annotations associated with the message.
    pub anotation: Vec<Anotation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Represents an annotation for a URL within a message.
///
/// This struct contains metadata about a URL, including its title, content, and the range in the message it refers to.
pub struct UrlAnotation {
    /// The URL being annotated.
    pub url: Url,
    /// The title of the annotated URL.
    pub title: String,
    /// The content or description associated with the URL.
    pub content: String,
    /// The start index in the message content where the annotation applies.
    pub start: usize,
    /// The end index in the message content where the annotation applies.
    pub end: usize,
}

/// Represents an annotation within a message.
///
/// An annotation can provide additional metadata or context, such as a URL annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anotation {
    /// An annotation for a URL within the message.
    Url(UrlAnotation),
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
            anotation: Vec::new(),
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
    pub fn with_with_attachments<U: TryInto<Url, Error: Debug>>(
        mut self,
        urls: impl IntoIterator<Item = U>,
    ) -> Self {
        self.attachments
            .extend(urls.into_iter().map(|url| url.try_into().unwrap()));
        self
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
        assert!(message.anotation.is_empty());
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

        let message = Message::user("Hello").with_with_attachments(urls.clone());

        assert_eq!(message.attachments.len(), 2);
        assert_eq!(message.attachments, urls);
    }

    #[test]
    fn test_url_annotation() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let annotation = UrlAnotation {
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
        let url_annotation = UrlAnotation {
            url,
            title: "Example".into(),
            content: "Example content".into(),
            start: 0,
            end: 10,
        };

        let annotation = Anotation::Url(url_annotation.clone());

        match annotation {
            Anotation::Url(url_anno) => {
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
        assert_eq!(original.anotation.len(), cloned.anotation.len());
    }
}
