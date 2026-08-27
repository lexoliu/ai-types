//! Working document utilities for long-task execution.
//!
//! `tasks.md` is a regular markdown file in the sandbox that the framework
//! treats specially by loading into context and supervising progress.

use std::path::Path;

use async_fs as fs;
use similar::{ChangeTag, TextDiff};

/// Snapshot of working documents currently present in the sandbox.
#[derive(Debug, Clone, Default)]
pub struct WorkingDocsSnapshot {
    /// Contents of `tasks.md`, when present.
    pub tasks_md: Option<String>,
}

impl WorkingDocsSnapshot {
    /// Returns true when `tasks.md` has unchecked markdown tasks.
    #[must_use]
    pub fn has_unchecked_items(&self) -> bool {
        self.tasks_md
            .as_deref()
            .is_some_and(has_unchecked_markdown_tasks)
    }
}

/// Reads `tasks.md` from the given sandbox directory.
pub async fn read_snapshot(sandbox_dir: &Path) -> WorkingDocsSnapshot {
    let tasks_path = sandbox_dir.join("tasks.md");

    let tasks_md = read_file_if_exists(&tasks_path).await;

    WorkingDocsSnapshot { tasks_md }
}

/// Returns a compact changed-line diff for `tasks.md` when it changed.
#[must_use]
pub fn tasks_md_diff(
    previous: &WorkingDocsSnapshot,
    current: &WorkingDocsSnapshot,
) -> Option<String> {
    if previous.tasks_md == current.tasks_md {
        return None;
    }

    let diff = TextDiff::from_lines(
        previous.tasks_md.as_deref().unwrap_or_default(),
        current.tasks_md.as_deref().unwrap_or_default(),
    );

    let mut lines = Vec::new();
    for change in diff.iter_all_changes() {
        let prefix = match change.tag() {
            ChangeTag::Delete => "- ",
            ChangeTag::Insert => "+ ",
            ChangeTag::Equal => continue,
        };
        let value = change.value().trim_end_matches('\n');
        if value.is_empty() {
            continue;
        }
        let mut line = String::from(prefix);
        line.push_str(value);
        lines.push(line);
        if lines.len() == 48 {
            lines.push("...".to_string());
            break;
        }
    }

    if lines.is_empty() {
        return Some("tasks.md changed.".to_string());
    }

    Some(lines.join("\n"))
}

/// Detects unfinished markdown checklist items (`- [ ]`).
#[must_use]
pub fn has_unchecked_markdown_tasks(content: &str) -> bool {
    content.lines().any(is_unchecked_todo_line)
}

fn is_unchecked_todo_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    (trimmed.starts_with("- [ ]") || trimmed.starts_with("* [ ]"))
        || (trimmed.starts_with("- [	]") || trimmed.starts_with("* [	]"))
}

async fn read_file_if_exists(path: &Path) -> Option<String> {
    match fs::read_to_string(path).await {
        Ok(content) => Some(content),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            tracing::warn!(path = %path.display(), error = %err, "failed to read working doc");
            None
        }
    }
}
