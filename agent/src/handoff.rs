use askama::Template;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffDocument {
    pub current_task: String,
    pub key_decisions: Vec<String>,
    pub files_paths: Vec<String>,
    pub errors_fixes: Vec<String>,
    pub pending_work: Vec<String>,
    pub recovery_hints: Vec<String>,
}

#[derive(Template)]
#[template(path = "handoff_document.md", escape = "none")]
struct HandoffDocumentTemplate<'a> {
    handoff: &'a HandoffDocument,
}

impl HandoffDocument {
    pub fn validate(self) -> Result<Self, String> {
        if self.current_task.trim().is_empty() {
            return Err("handoff.current_task must not be empty".to_string());
        }
        Ok(self)
    }

    #[must_use]
    pub fn render_markdown(&self) -> String {
        HandoffDocumentTemplate { handoff: self }
            .render()
            .expect("handoff document template must render")
    }
}
