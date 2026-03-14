//! Unified context manager for agent conversation state.
//!
//! The [`Context`] struct owns the entire context window state: persistent
//! system blocks (stable cacheable prefix), ephemeral reminders, compaction
//! handoff, and recent conversation messages.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use aither_core::llm::{Message, Role};

use crate::handoff::HandoffDocument;

#[derive(Serialize)]
struct TextBlock<'a> {
    #[serde(rename = "$text")]
    content: &'a str,
}

/// The entire context window state. Fully serializable for persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// Persistent system blocks keyed by snake_case type name.
    ///
    /// These survive compaction and are rendered as a single cacheable system
    /// prefix message.
    system_blocks: IndexMap<String, String>,

    /// Ephemeral reminder blocks rendered as separate system messages after the
    /// stable system prefix.
    reminders: Vec<String>,

    /// Compaction handoff document from previous context compaction.
    ///
    /// This is rendered after reminders and before recent conversation.
    handoff: Option<String>,

    /// Recent conversation messages.
    recent: Vec<Message>,
}

impl Context {
    /// Creates a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── System blocks (stable, cacheable prefix) ──────────────────────

    /// Inserts or replaces a persistent system block.
    ///
    /// Identity is derived from the snake_case type name of `T`.
    pub fn insert_system<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.system_blocks.insert(tag, xml);
    }

    /// Inserts or replaces a persistent system block with an explicit tag.
    pub fn insert_system_named(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        let tag = tag.into();
        let content = content.into();
        let xml = serialize_xml(&tag, &TextBlock { content: &content });
        self.system_blocks.insert(tag, xml);
    }

    /// Removes a persistent system block by type.
    pub fn remove_system<T>(&mut self) {
        let tag = snake_case_type_name::<T>();
        self.system_blocks.shift_remove(&tag);
    }

    /// Removes a persistent system block by explicit tag.
    pub fn remove_system_named(&mut self, tag: &str) {
        self.system_blocks.shift_remove(tag);
    }

    /// Returns the number of persistent system blocks.
    #[must_use]
    pub fn system_block_count(&self) -> usize {
        self.system_blocks.len()
    }

    /// Returns whether a persistent system block with `tag` exists.
    #[must_use]
    pub fn has_system_block(&self, tag: &str) -> bool {
        self.system_blocks.contains_key(tag)
    }

    /// Returns an immutable view of persistent system blocks.
    #[must_use]
    pub fn system_blocks(&self) -> &IndexMap<String, String> {
        &self.system_blocks
    }

    /// Returns a mutable view of persistent system blocks.
    #[must_use]
    pub fn system_blocks_mut(&mut self) -> &mut IndexMap<String, String> {
        &mut self.system_blocks
    }

    // ── Reminders (ephemeral) ─────────────────────────────────────────

    /// Inserts an ephemeral reminder.
    pub fn insert_reminder<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.reminders.push(xml);
    }

    /// Clears all ephemeral reminders.
    pub fn clear_reminders(&mut self) {
        self.reminders.clear();
    }

    /// Returns current reminders.
    #[must_use]
    pub fn reminders(&self) -> &[String] {
        &self.reminders
    }

    // ── Handoff ───────────────────────────────────────────────────────

    /// Sets or replaces the compaction handoff document.
    pub fn set_handoff(&mut self, handoff: impl Into<String>) {
        let handoff = handoff.into();
        self.handoff = Some(serialize_xml("handoff", &TextBlock { content: &handoff }));
    }

    /// Sets the compaction handoff document from a structured handoff artifact.
    pub fn set_handoff_document(&mut self, handoff: &HandoffDocument) {
        self.handoff = Some(serialize_xml("handoff", handoff));
    }

    /// Clears the compaction handoff document.
    pub fn clear_handoff(&mut self) {
        self.handoff = None;
    }

    /// Returns the current compaction handoff document.
    #[must_use]
    pub fn handoff(&self) -> Option<&str> {
        self.handoff.as_deref()
    }

    // ── Recent conversation ───────────────────────────────────────────

    /// Appends a message to recent conversation.
    pub fn push(&mut self, message: Message) {
        self.recent.push(message);
    }

    /// Extends recent conversation with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.recent.extend(messages);
    }

    /// Returns number of recent conversation messages.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.recent.len()
    }

    /// Returns the number of recent system messages.
    #[must_use]
    pub fn count_recent_system_messages(&self) -> usize {
        self.recent
            .iter()
            .filter(|message| message.role() == Role::System)
            .count()
    }

    /// Returns recent conversation messages.
    #[must_use]
    pub fn recent(&self) -> &[Message] {
        &self.recent
    }

    /// Returns mutable recent conversation messages.
    #[must_use]
    pub fn recent_mut(&mut self) -> &mut Vec<Message> {
        &mut self.recent
    }

    /// Returns the latest recent message.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.recent.last()
    }

    /// Returns whether recent conversation is empty.
    #[must_use]
    pub fn is_conversation_empty(&self) -> bool {
        self.recent.is_empty()
    }

    /// Drains oldest recent messages while keeping the latest `keep` messages.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        if keep >= self.recent.len() {
            return Vec::new();
        }
        self.recent.drain(..self.recent.len() - keep).collect()
    }

    /// Returns recent conversation messages cloned.
    #[must_use]
    pub fn conversation_messages(&self) -> Vec<Message> {
        self.recent.clone()
    }

    // ── Message assembly ───────────────────────────────────────────────

    /// Builds the full LLM message list.
    ///
    /// Layout:
    /// 1. Persistent system blocks (single cacheable prefix)
    /// 2. Ephemeral reminders (one system message each)
    /// 3. Compaction handoff (optional system message)
    /// 4. Recent conversation
    #[must_use]
    pub fn build_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if !self.system_blocks.is_empty() {
            let system_xml = self
                .system_blocks
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(Message::system(system_xml));
        }

        for reminder in &self.reminders {
            messages.push(Message::system(reminder));
        }

        if let Some(handoff) = &self.handoff {
            messages.push(Message::system(handoff));
        }

        messages.extend(self.recent.iter().cloned());
        messages
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Clears only recent conversation messages.
    pub fn clear_recent(&mut self) {
        self.recent.clear();
    }

    /// Clears all non-persistent context: reminders, handoff, recent conversation.
    pub fn clear_history(&mut self) {
        self.reminders.clear();
        self.handoff = None;
        self.recent.clear();
    }

    /// Clears everything, including persistent system blocks.
    pub fn clear_all(&mut self) {
        self.system_blocks.clear();
        self.clear_history();
    }

    /// Clones the context.
    #[must_use]
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Creates a checkpoint that can be restored later.
    #[must_use]
    pub fn checkpoint(&self) -> ContextCheckpoint {
        ContextCheckpoint {
            reminders: self.reminders.clone(),
            handoff: self.handoff.clone(),
            recent: self.recent.clone(),
        }
    }

    /// Restores reminders, handoff, and recent conversation from checkpoint.
    pub fn restore(&mut self, checkpoint: ContextCheckpoint) {
        self.reminders = checkpoint.reminders;
        self.handoff = checkpoint.handoff;
        self.recent = checkpoint.recent;
    }

    /// Removes recent system messages and returns how many were pruned.
    pub fn prune_recent_system_messages(&mut self) -> usize {
        let before = self.recent.len();
        self.recent.retain(|message| message.role() != Role::System);
        before - self.recent.len()
    }

    /// Restores runtime-managed state from another serialized context while
    /// preserving the current persistent system blocks.
    pub fn restore_runtime_state(&mut self, restored: Self) {
        self.reminders = restored.reminders;
        self.handoff = restored.handoff;
        self.recent = restored.recent;
    }
}

/// A snapshot of non-persistent context state that can be restored.
#[derive(Debug, Clone)]
pub struct ContextCheckpoint {
    reminders: Vec<String>,
    handoff: Option<String>,
    recent: Vec<Message>,
}

impl ContextCheckpoint {
    /// Returns the number of recent messages in this checkpoint.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.recent.len()
    }

    /// Returns whether this checkpoint has no reminders, no handoff, and no recent messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reminders.is_empty() && self.handoff.is_none() && self.recent.is_empty()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn snake_case_type_name<T>() -> String {
    let full = std::any::type_name::<T>();
    let short = full.rsplit("::").next().unwrap_or(full);
    heck::AsSnakeCase(short).to_string()
}

pub(crate) fn serialize_xml<T: Serialize>(tag: &str, value: &T) -> String {
    let mut buffer = String::new();
    let serializer = quick_xml::se::Serializer::with_root(&mut buffer, Some(tag))
        .unwrap_or_else(|error| panic!("failed to create XML serializer for tag '{tag}': {error}"));

    value
        .serialize(serializer)
        .unwrap_or_else(|error| panic!("failed to serialize XML for tag '{tag}': {error}"));

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Memory {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct Reminder {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct Workspace {
        sandbox: String,
        cwd: String,
    }

    #[test]
    fn snake_case_type_name_uses_short_struct_name() {
        assert_eq!(snake_case_type_name::<Memory>(), "memory");
    }

    #[test]
    fn serialize_xml_renders_text_content() {
        let xml = serialize_xml(
            "memory",
            &Memory {
                content: "remember this".to_string(),
            },
        );
        assert!(xml.contains("<memory>"));
        assert!(xml.contains("remember this"));
        assert!(xml.contains("</memory>"));
    }

    #[test]
    fn serialize_xml_renders_struct_fields() {
        let xml = serialize_xml(
            "workspace",
            &Workspace {
                sandbox: "/tmp/sandbox".to_string(),
                cwd: "/home/user".to_string(),
            },
        );
        assert!(xml.contains("<workspace>"));
        assert!(xml.contains("<sandbox>"));
        assert!(xml.contains("/tmp/sandbox"));
        assert!(xml.contains("</workspace>"));
    }

    #[test]
    fn insert_system_replaces_existing_block() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "old".to_string(),
        });
        context.insert_system(&Memory {
            content: "new".to_string(),
        });

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content().contains("new"));
        assert!(!messages[0].content().contains("old"));
    }

    #[test]
    fn build_messages_has_expected_layout() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "base".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "todo".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::user("hello"));

        let messages = context.build_messages();
        assert_eq!(messages.len(), 4);
        assert!(messages[0].content().contains("base"));
        assert!(messages[1].content().contains("todo"));
        assert!(messages[2].content().contains("handoff"));
        assert!(messages[3].content().contains("hello"));
    }

    #[test]
    fn clear_history_preserves_system_blocks() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "persist".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::assistant("recent"));

        context.clear_history();

        assert_eq!(context.system_block_count(), 1);
        assert!(context.reminders().is_empty());
        assert!(context.handoff().is_none());
        assert_eq!(context.len_recent(), 0);
    }

    #[test]
    fn checkpoint_restore_roundtrip() {
        let mut context = Context::new();
        context.insert_reminder(&Reminder {
            content: "before".to_string(),
        });
        context.set_handoff("before");
        context.push(Message::user("first"));

        let checkpoint = context.checkpoint();

        context.clear_history();
        context.insert_reminder(&Reminder {
            content: "after".to_string(),
        });
        context.push(Message::user("second"));

        context.restore(checkpoint);

        assert_eq!(context.reminders().len(), 1);
        assert!(context.handoff().is_some());
        assert_eq!(context.len_recent(), 1);
        assert_eq!(context.recent()[0].content(), "first");
    }

    #[test]
    fn restore_runtime_state_preserves_system_blocks() {
        let mut current = Context::new();
        current.insert_system(&Memory {
            content: "fresh".to_string(),
        });

        let mut restored = Context::new();
        restored.insert_system(&Memory {
            content: "stale".to_string(),
        });
        restored.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        restored.set_handoff("summary");
        restored.push(Message::user("hello"));

        current.restore_runtime_state(restored);

        let messages = current.build_messages();
        assert!(messages[0].content().contains("fresh"));
        assert!(!messages[0].content().contains("stale"));
        assert_eq!(current.reminders().len(), 1);
        assert!(current.handoff().is_some());
        assert_eq!(current.len_recent(), 1);
    }

    #[test]
    fn prune_recent_system_messages_removes_only_system_entries() {
        let mut context = Context::new();
        context.push(Message::user("user"));
        context.push(Message::system("system"));
        context.push(Message::assistant("assistant"));
        context.push(Message::system("system-2"));

        let removed = context.prune_recent_system_messages();

        assert_eq!(removed, 2);
        assert_eq!(context.len_recent(), 2);
        assert_eq!(context.recent()[0].content(), "user");
        assert_eq!(context.recent()[1].content(), "assistant");
    }

    #[test]
    fn serde_roundtrip_preserves_message_layout() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "persist".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::user("hello"));
        context.push(Message::assistant("world"));

        let expected_messages = context.build_messages();
        let encoded = serde_json::to_string(&context)
            .unwrap_or_else(|error| panic!("failed to serialize context: {error}"));
        let decoded: Context = serde_json::from_str(&encoded)
            .unwrap_or_else(|error| panic!("failed to deserialize context: {error}"));
        let actual_messages = decoded.build_messages();

        assert_eq!(actual_messages, expected_messages);
    }
}
