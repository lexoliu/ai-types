//! Structured types the language model produces during fact extraction.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
/// Facts the model pulled out of a conversation.
pub struct ExtractedFacts {
    /// Short, self-contained statements worth remembering.
    pub facts: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
/// What to do with a memory in light of a newly extracted fact.
pub enum Action {
    /// Store the fact as a new memory.
    Add,
    /// Replace an existing memory's content.
    Update,
    /// Remove an existing memory the fact contradicts.
    Delete,
    /// Do nothing; the fact adds no new information.
    Noop,
}

/// The model's decision about one extracted fact.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryDecision {
    /// The operation to perform.
    pub action: Action,
    /// The ID of the existing memory to update or delete. Required for UPDATE and DELETE.
    pub memory_id: Option<String>,
    /// The new content for the memory. Required for UPDATE.
    pub new_content: Option<String>,
    /// Brief reasoning for the decision.
    pub reasoning: String,
}
