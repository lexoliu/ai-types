//! Agent checkpoint bundles for restart-safe persistence.

use serde::{Deserialize, Serialize};

use crate::{ContextWindowSnapshot, TodoItem};

/// Checkpoint payload exported by the agent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    /// Serialized runtime context managed by the agent.
    pub context_json: String,
    /// Current todo list state.
    pub todo_items: Vec<TodoItem>,
    /// Optional external history anchor supplied by the host runtime.
    pub history_anchor: Option<usize>,
    /// Hash of the active tool surface for compatibility checks.
    pub tool_surface_hash: String,
    /// Structured snapshot of the current context window.
    pub context_window: ContextWindowSnapshot,
    /// Whether background work was still active when the checkpoint was exported.
    pub has_background_tasks: bool,
}
