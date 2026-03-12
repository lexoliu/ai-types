//! Structured snapshots of the currently assembled context window.

use aither_core::llm::Message;
use serde::{Deserialize, Serialize};

/// High-level lifecycle phase of the assembled context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindowPhase {
    /// Regular operating state.
    Stable,
    /// Full-window usage is high enough that compaction should be considered.
    CompressionDue,
    /// Usage is close enough to the hard limit that the model should hand off.
    HandoffDue,
    /// A handoff document is already active in the context.
    HandoffActive,
}

/// Metrics describing the currently assembled context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowMetrics {
    /// Estimated fraction of the effective context window currently in use.
    pub usage_fraction: f32,
    /// Context window of the currently selected reasoning model.
    pub selected_model_context_window: Option<u32>,
    /// Context window of the fast model used for compaction.
    pub fast_model_context_window: Option<u32>,
    /// Effective window used for budgeting (minimum of selected and fast).
    pub effective_context_window: usize,
    /// Number of persistent system blocks in the structured context.
    pub system_block_count: usize,
    /// Number of ephemeral reminders currently assembled.
    pub reminder_count: usize,
    /// Number of recent conversation messages.
    pub recent_message_count: usize,
    /// Whether a persisted handoff document is present.
    pub has_handoff: bool,
}

/// Fully assembled LLM request window plus lifecycle metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowSnapshot {
    /// Lifecycle phase derived from current budgeting rules.
    pub phase: ContextWindowPhase,
    /// Metrics for observability and persistence.
    pub metrics: ContextWindowMetrics,
    /// Exact message list that would be sent to the model for the next turn.
    pub messages: Vec<Message>,
}
