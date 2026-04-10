//! Managed browser tool protocol and tool implementation.

use std::borrow::Cow;

use aither_agent::tool_request::{
    ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};
use aither_core::llm::{IntoToolResult, Tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserServiceStatus {
    pub running: bool,
    pub cdp_endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserReadDetail {
    Brief,
    Standard,
    Full,
}

impl Default for BrowserReadDetail {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserScreenshotMode {
    Off,
    Viewport,
    FullPage,
}

impl Default for BrowserScreenshotMode {
    fn default() -> Self {
        Self::Viewport
    }
}

#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq,
)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserReadFormat {
    #[default]
    Json,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BrowserReadOptions {
    #[serde(default)]
    pub detail: BrowserReadDetail,
    #[serde(default)]
    pub screenshot: BrowserScreenshotMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserInteractiveElement {
    pub role: String,
    pub tag_name: String,
    pub text: Option<String>,
    pub accessible_name: Option<String>,
    pub href: Option<String>,
    pub input_type: Option<String>,
    pub disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilityProperty {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilityNode {
    pub role: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub value: Option<String>,
    pub depth: usize,
    pub child_count: usize,
    pub ignored: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<BrowserAccessibilityProperty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilitySnapshot {
    pub total_nodes: usize,
    pub interesting_nodes: usize,
    pub interactive_nodes: usize,
    pub heading_nodes: usize,
    pub landmark_nodes: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<BrowserAccessibilityNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserCapturedScreenshot {
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub data_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserArtifactScreenshot {
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub artifact_path: String,
    pub api_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageSummary {
    pub url: String,
    pub title: String,
    pub main_heading: Option<String>,
    pub headings: Vec<String>,
    pub text_excerpt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interactive_elements: Vec<BrowserInteractiveElement>,
    pub accessibility: BrowserAccessibilitySnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageCapture {
    pub page: BrowserPageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<BrowserCapturedScreenshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageSnapshot {
    #[serde(flatten)]
    pub page: BrowserPageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<BrowserArtifactScreenshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum BrowserServiceRequest {
    Status,
    EnsureStarted,
    Stop,
    Read {
        url: String,
        #[serde(default)]
        options: BrowserReadOptions,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BrowserServiceResponse {
    Ok { payload: BrowserServiceStatus },
    ReadOk { payload: BrowserPageCapture },
    Stopped,
    Error { message: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum BrowserArgs {
    /// Show managed browser runtime status.
    Status,
    /// Start the managed browser runtime.
    #[serde(rename = "open")]
    Start,
    /// Stop the managed browser runtime.
    #[serde(rename = "close")]
    Stop,
    /// Return the managed browser CDP endpoint. Requires `open` first.
    CdpEndpoint,
    /// Open a URL in the managed browser and inspect it with structured text,
    /// accessibility output, and an optional screenshot artifact.
    Read {
        url: String,
        #[serde(default)]
        detail: BrowserReadDetail,
        #[serde(default)]
        screenshot: BrowserScreenshotMode,
        #[serde(default)]
        format: BrowserReadFormat,
    },
}

impl BrowserArgs {
    #[must_use]
    pub fn read_options(
        detail: BrowserReadDetail,
        screenshot: BrowserScreenshotMode,
    ) -> BrowserReadOptions {
        BrowserReadOptions { detail, screenshot }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BrowserRequest {
    pub session_id: String,
    pub args: BrowserArgs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserResult {
    pub running: bool,
    pub cdp_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<BrowserPageSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum BrowserToolError {
    #[error("browser command is unavailable in this runtime")]
    UnavailableInRuntime,
    #[error("{message}")]
    RequestFailed { message: String },
}

impl BrowserToolError {
    #[must_use]
    pub fn request_failed(message: impl Into<String>) -> Self {
        Self::RequestFailed {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BrowserToolOutput(Result<BrowserResult, BrowserToolError>);

impl From<Result<BrowserResult, BrowserToolError>> for BrowserToolOutput {
    fn from(value: Result<BrowserResult, BrowserToolError>) -> Self {
        Self(value)
    }
}

impl IntoToolResult for BrowserToolOutput {
    fn into_tool_result(self) -> aither_core::Result<ToolResult> {
        match self.0 {
            Ok(result) => ToolResult::json(&result),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

pub type BrowserBroker = ToolRequestBroker<BrowserRequest, Result<BrowserResult, BrowserToolError>>;
pub type BrowserQueue = ToolRequestQueue<BrowserRequest, Result<BrowserResult, BrowserToolError>>;

#[must_use]
pub fn channel() -> (BrowserBroker, BrowserQueue) {
    tool_request_channel()
}

#[derive(Debug, Clone)]
pub struct BrowserTool {
    session_id: String,
    broker: Option<BrowserBroker>,
}

impl BrowserTool {
    #[must_use]
    pub const fn new(session_id: String, broker: Option<BrowserBroker>) -> Self {
        Self { session_id, broker }
    }
}

impl Tool for BrowserTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("browser")
    }

    type Arguments = BrowserArgs;
    type Res = BrowserToolOutput;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let Some(broker) = self.broker.as_ref() else {
            return Ok(BrowserToolOutput::from(Err(
                BrowserToolError::UnavailableInRuntime,
            )));
        };
        let response = broker
            .request(BrowserRequest {
                session_id: self.session_id.clone(),
                args,
            })
            .await?;
        Ok(BrowserToolOutput::from(response))
    }
}
