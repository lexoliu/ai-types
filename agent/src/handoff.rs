use askama::Template;
use serde::{Deserialize, Serialize};

use aither_core::llm::Message;

/// Structured handoff document used to recover work after compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffDocument {
    /// Current task being pursued.
    pub current_task: String,
    /// Key decisions that should be preserved.
    pub key_decisions: Vec<String>,
    /// Relevant files and paths.
    pub files_paths: Vec<String>,
    /// Errors encountered and fixes applied.
    pub errors_fixes: Vec<String>,
    /// Work that remains pending.
    pub pending_work: Vec<String>,
    /// Hints for recovering context.
    pub recovery_hints: Vec<String>,
    /// Mechanical index of on-disk materials produced during the compacted
    /// conversation (`outputs/`, `artifacts/`), each annotated with the
    /// originating command's description.
    ///
    /// Built by the framework, not the LLM: it tells the model after
    /// compaction exactly which results already live on disk so it re-reads
    /// files instead of re-running the work.
    #[serde(default)]
    pub file_index: Vec<String>,
}

/// Builds the mechanical file index for a handoff from conversation messages.
///
/// Scans tool results for `outputs/` URLs and `artifacts/` paths and pairs
/// each with the description of the tool call that produced it. Later
/// mentions of the same path win.
#[must_use]
pub fn build_file_index(messages: &[Message]) -> Vec<String> {
    use std::collections::HashMap;

    let mut call_description: HashMap<&str, &str> = HashMap::new();
    for message in messages {
        for call in message.tool_calls() {
            if let Some(description) = call.arguments.get("description").and_then(|v| v.as_str()) {
                call_description.insert(call.id.as_str(), description);
            }
        }
    }

    let mut index: indexmap::IndexMap<String, Option<&str>> = indexmap::IndexMap::new();
    for message in messages {
        let Message::Tool {
            content,
            tool_call_id,
        } = message
        else {
            continue;
        };
        let description = call_description.get(tool_call_id.as_str()).copied();
        for url in crate::compression::extract_referenced_urls(content) {
            index.insert(url, description);
        }
        for word in content.split_whitespace() {
            let word = word.trim_matches(|c: char| {
                c == '"' || c == '\'' || c == '`' || c == ',' || c == ')' || c == ']' || c == ';'
            });
            if word.starts_with("artifacts/") && word.len() > "artifacts/".len() {
                index.insert(word.to_string(), description);
            }
        }
    }

    index
        .into_iter()
        .map(|(path, description)| match description {
            Some(description) => format!("{path} — {description}"),
            None => path,
        })
        .collect()
}

#[derive(Template)]
#[template(path = "handoff_document.md", escape = "none")]
struct HandoffDocumentTemplate<'a> {
    handoff: &'a HandoffDocument,
}

impl HandoffDocument {
    /// Validate required handoff fields.
    ///
    /// # Errors
    ///
    /// Returns an error when required handoff fields are empty.
    pub fn validate(self) -> Result<Self, String> {
        if self.current_task.trim().is_empty() {
            return Err("handoff.current_task must not be empty".to_string());
        }
        Ok(self)
    }

    /// Render this handoff as Markdown.
    ///
    /// # Panics
    ///
    /// Panics if the bundled handoff template cannot be rendered.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        HandoffDocumentTemplate { handoff: self }
            .render()
            .expect("handoff document template must render")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::ToolCall;

    #[test]
    fn build_file_index_pairs_outputs_with_call_descriptions() {
        let messages = vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({
                        "description": "Fetch WebGPU status pages",
                        "script": "webfetch https://caniuse.com"
                    }),
                )],
            ),
            Message::tool(
                "call_1",
                "Output saved to outputs/amber-oak-swift-river.txt (100 lines, 5000 bytes)",
            ),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_2",
                    "terminal",
                    serde_json::json!({"description": "Write deck", "script": "tee artifacts/slides.md"}),
                )],
            ),
            Message::tool("call_2", "wrote artifacts/slides.md"),
        ];

        let index = build_file_index(&messages);
        assert_eq!(index.len(), 2);
        assert!(
            index[0].contains("outputs/amber-oak-swift-river.txt")
                && index[0].contains("Fetch WebGPU status pages"),
            "index entry should pair path with description: {index:?}"
        );
        assert!(index[1].contains("artifacts/slides.md") && index[1].contains("Write deck"));
    }

    #[test]
    fn build_file_index_keeps_latest_description_per_path() {
        let messages = vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({"description": "first pass"}),
                )],
            ),
            Message::tool("call_1", "see artifacts/report.md"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_2",
                    "terminal",
                    serde_json::json!({"description": "final revision"}),
                )],
            ),
            Message::tool("call_2", "updated artifacts/report.md"),
        ];

        let index = build_file_index(&messages);
        assert_eq!(index.len(), 1);
        assert!(index[0].contains("final revision"));
    }

    #[test]
    fn handoff_markdown_renders_file_index_section() {
        let handoff = HandoffDocument {
            current_task: "task".to_string(),
            key_decisions: Vec::new(),
            files_paths: Vec::new(),
            errors_fixes: Vec::new(),
            pending_work: Vec::new(),
            recovery_hints: Vec::new(),
            file_index: vec!["outputs/a-b-c-d.txt — research notes".to_string()],
        };
        let markdown = handoff.render_markdown();
        assert!(markdown.contains("Materials Already On Disk"));
        assert!(markdown.contains("outputs/a-b-c-d.txt — research notes"));
        assert!(markdown.contains("Read the file instead of redoing the work"));
    }
}
