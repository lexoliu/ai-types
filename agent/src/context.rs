//! Unified context manager for agent conversation state.
//!
//! The [`Context`] struct owns the entire context window state: persistent
//! system blocks (stable cacheable prefix), ephemeral reminders, compaction
//! handoff, and recent conversation messages.

use std::hash::{Hash, Hasher};

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

    /// Compaction handoff document from previous context compaction.
    ///
    /// This is rendered immediately after the stable system prefix.
    handoff: Option<String>,

    /// Ordered runtime items rendered after the optional handoff anchor.
    runtime_items: Vec<ContextRuntimeItem>,
}

/// Ordered runtime items stored after the fixed handoff anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextRuntimeItem {
    /// A reminder rendered as a system message.
    Reminder(String),
    /// A regular conversation message.
    Message(Message),
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

    /// Inserts or replaces a persistent system block with raw text.
    ///
    /// Unlike [`insert_system`](Self::insert_system) and
    /// [`insert_system_named`](Self::insert_system_named), the `content` is
    /// stored verbatim without any XML wrapping. This is the preferred entry
    /// point for prose system blocks such as workspace descriptions, runtime
    /// metadata, and environment hints, where XML structure adds tokens
    /// without providing semantic value.
    ///
    /// Use [`insert_system`](Self::insert_system) only when the block is a
    /// structured payload that genuinely needs XML delimiters to separate
    /// context from instructions (for example, attached documents).
    pub fn insert_system_text(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        self.system_blocks.insert(tag.into(), content.into());
    }

    /// Returns a stable fingerprint of the current system blocks in order.
    ///
    /// The fingerprint changes if and only if the ordered set of
    /// `(tag, content)` pairs changes. Hosts can use this to detect
    /// cache-invalidating modifications to the stable system prefix
    /// (for example, to track KV-cache hit rate over time).
    #[must_use]
    pub fn stable_prefix_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (tag, content) in &self.system_blocks {
            tag.hash(&mut hasher);
            content.hash(&mut hasher);
        }
        hasher.finish()
    }

    // ── Reminders (ephemeral) ─────────────────────────────────────────

    /// Inserts an ephemeral reminder.
    pub fn insert_reminder<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.runtime_items.push(ContextRuntimeItem::Reminder(xml));
    }

    /// Clears all ephemeral reminders.
    pub fn clear_reminders(&mut self) {
        self.runtime_items
            .retain(|item| !matches!(item, ContextRuntimeItem::Reminder(_)));
    }

    /// Returns current reminders.
    #[must_use]
    pub fn reminders(&self) -> Vec<&str> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextRuntimeItem::Reminder(content) => Some(content.as_str()),
                ContextRuntimeItem::Message(_) => None,
            })
            .collect()
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
        self.runtime_items
            .push(ContextRuntimeItem::Message(message));
    }

    /// Extends recent conversation with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.runtime_items
            .extend(messages.into_iter().map(ContextRuntimeItem::Message));
    }

    /// Returns number of recent conversation messages.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.runtime_items
            .iter()
            .filter(|item| matches!(item, ContextRuntimeItem::Message(_)))
            .count()
    }

    /// Returns the number of recent system messages.
    #[must_use]
    pub fn count_recent_system_messages(&self) -> usize {
        self.runtime_items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ContextRuntimeItem::Message(message) if message.role() == Role::System
                )
            })
            .count()
    }

    /// Returns recent conversation messages.
    #[must_use]
    pub fn recent(&self) -> Vec<&Message> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextRuntimeItem::Reminder(_) => None,
                ContextRuntimeItem::Message(message) => Some(message),
            })
            .collect()
    }

    /// Returns the latest recent message.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.runtime_items.iter().rev().find_map(|item| match item {
            ContextRuntimeItem::Reminder(_) => None,
            ContextRuntimeItem::Message(message) => Some(message),
        })
    }

    /// Returns whether recent conversation is empty.
    #[must_use]
    pub fn is_conversation_empty(&self) -> bool {
        self.len_recent() == 0
    }

    /// Drains oldest recent messages while keeping the latest `keep` messages.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        let remove_count = self.len_recent().saturating_sub(keep);
        if remove_count == 0 {
            return Vec::new();
        }
        let mut removed = Vec::with_capacity(remove_count);
        let mut removed_messages = 0usize;
        let mut retained = Vec::with_capacity(self.runtime_items.len());
        for item in self.runtime_items.drain(..) {
            match item {
                ContextRuntimeItem::Message(message) if removed_messages < remove_count => {
                    removed.push(message);
                    removed_messages += 1;
                }
                other => retained.push(other),
            }
        }
        self.runtime_items = retained;
        removed
    }

    /// Returns recent conversation messages cloned.
    #[must_use]
    pub fn conversation_messages(&self) -> Vec<Message> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextRuntimeItem::Reminder(_) => None,
                ContextRuntimeItem::Message(message) => Some(message.clone()),
            })
            .collect()
    }

    // ── Message assembly ───────────────────────────────────────────────

    /// Builds the full LLM message list.
    ///
    /// Layout:
    /// 1. Persistent system blocks (single cacheable prefix)
    /// 2. Compaction handoff (optional system message)
    /// 3. Additional transient system messages for the current request
    /// 4. Ordered runtime items (reminders and conversation messages)
    #[must_use]
    pub fn build_messages(&self) -> Vec<Message> {
        self.build_messages_with_transient_system(&[])
    }

    /// Builds the full LLM message list with additional transient system messages.
    ///
    /// Transient system messages are appended immediately after the fixed handoff
    /// anchor and before ordered runtime items. They are not
    /// stored inside the context and therefore are not checkpointed.
    #[must_use]
    pub fn build_messages_with_transient_system(
        &self,
        transient_system: &[String],
    ) -> Vec<Message> {
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

        if let Some(handoff) = &self.handoff {
            messages.push(Message::system(handoff));
        }

        for content in transient_system {
            messages.push(Message::system(content.clone()));
        }

        messages.extend(self.runtime_items.iter().cloned().map(|item| match item {
            ContextRuntimeItem::Reminder(content) => Message::system(content),
            ContextRuntimeItem::Message(message) => message,
        }));
        messages
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Clears only recent conversation messages.
    pub fn clear_recent(&mut self) {
        self.runtime_items
            .retain(|item| !matches!(item, ContextRuntimeItem::Message(_)));
    }

    /// Clears all non-persistent context: reminders, handoff, recent conversation.
    pub fn clear_history(&mut self) {
        self.handoff = None;
        self.runtime_items.clear();
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
            handoff: self.handoff.clone(),
            items: self
                .runtime_items
                .iter()
                .cloned()
                .map(|item| match item {
                    ContextRuntimeItem::Reminder(content) => {
                        ContextCheckpointItem::Reminder(content)
                    }
                    ContextRuntimeItem::Message(message) => ContextCheckpointItem::Message(message),
                })
                .collect(),
        }
    }

    /// Restores reminders, handoff, and recent conversation from checkpoint.
    pub fn restore(&mut self, checkpoint: ContextCheckpoint) {
        self.handoff = checkpoint.handoff;
        self.runtime_items = checkpoint
            .items
            .into_iter()
            .map(|item| match item {
                ContextCheckpointItem::Reminder(content) => ContextRuntimeItem::Reminder(content),
                ContextCheckpointItem::Message(message) => ContextRuntimeItem::Message(message),
            })
            .collect();
    }

    /// Removes recent system messages and returns how many were pruned.
    pub fn prune_recent_system_messages(&mut self) -> usize {
        let before = self.len_recent();
        self.runtime_items.retain(|item| {
            !matches!(
                item,
                ContextRuntimeItem::Message(message) if message.role() == Role::System
            )
        });
        before - self.len_recent()
    }

    /// Restores runtime-managed state from another serialized context while
    /// preserving the current persistent system blocks.
    pub fn restore_runtime_state(&mut self, restored: Self) {
        self.handoff = restored.handoff;
        self.runtime_items = restored.runtime_items;
    }
}

/// A snapshot of non-persistent context state that can be restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    handoff: Option<String>,
    items: Vec<ContextCheckpointItem>,
}

/// Ordered runtime checkpoint items stored after the fixed handoff anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextCheckpointItem {
    /// A reminder rendered as a system message.
    Reminder(String),
    /// A regular conversation message.
    Message(Message),
}

impl ContextCheckpoint {
    /// Creates a structured runtime checkpoint.
    #[must_use]
    pub const fn new(handoff: Option<String>, items: Vec<ContextCheckpointItem>) -> Self {
        Self { handoff, items }
    }

    /// Returns the number of recent messages in this checkpoint.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, ContextCheckpointItem::Message(_)))
            .count()
    }

    /// Returns the current handoff document, if present.
    #[must_use]
    pub fn handoff(&self) -> Option<&str> {
        self.handoff.as_deref()
    }

    /// Returns ordered runtime checkpoint items.
    #[must_use]
    pub fn items(&self) -> &[ContextCheckpointItem] {
        &self.items
    }

    /// Returns whether this checkpoint has no handoff and no runtime items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handoff.is_none() && self.items.is_empty()
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
    fn insert_system_text_stores_raw_content() {
        let mut context = Context::new();
        context.insert_system_text("workspace", "Environment: local\nCWD: /tmp");

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert_eq!(messages.len(), 1);
        let rendered = messages[0].content();
        assert!(rendered.contains("Environment: local"));
        assert!(rendered.contains("CWD: /tmp"));
        assert!(
            !rendered.contains("<workspace>"),
            "insert_system_text must not add XML tags: {rendered}"
        );
    }

    #[test]
    fn insert_system_text_replaces_by_tag() {
        let mut context = Context::new();
        context.insert_system_text("workspace", "old");
        context.insert_system_text("workspace", "new");

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert!(messages[0].content().contains("new"));
        assert!(!messages[0].content().contains("old"));
    }

    #[test]
    fn stable_prefix_fingerprint_reflects_block_changes() {
        let mut context = Context::new();
        let empty = context.stable_prefix_fingerprint();

        context.insert_system_text("workspace", "a");
        let after_insert = context.stable_prefix_fingerprint();
        assert_ne!(empty, after_insert);

        context.insert_system_text("workspace", "b");
        let after_edit = context.stable_prefix_fingerprint();
        assert_ne!(after_insert, after_edit);

        context.remove_system_named("workspace");
        assert_eq!(context.stable_prefix_fingerprint(), empty);
    }

    #[test]
    fn stable_prefix_fingerprint_is_order_sensitive() {
        let mut a = Context::new();
        a.insert_system_text("one", "1");
        a.insert_system_text("two", "2");

        let mut b = Context::new();
        b.insert_system_text("two", "2");
        b.insert_system_text("one", "1");

        assert_ne!(
            a.stable_prefix_fingerprint(),
            b.stable_prefix_fingerprint(),
            "insertion order must be part of the fingerprint to reflect KV-cache prefix stability"
        );
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
        assert!(messages[1].content().contains("handoff"));
        assert!(messages[2].content().contains("todo"));
        assert!(messages[3].content().contains("hello"));
    }

    #[test]
    fn transient_system_messages_are_inserted_before_recent_conversation() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "base".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "todo".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::user("hello"));

        let messages = context.build_messages_with_transient_system(&["hidden".to_string()]);
        assert_eq!(messages.len(), 5);
        assert!(messages[0].content().contains("base"));
        assert!(messages[1].content().contains("handoff"));
        assert_eq!(messages[2].content(), "hidden");
        assert!(messages[3].content().contains("todo"));
        assert!(messages[4].content().contains("hello"));
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
