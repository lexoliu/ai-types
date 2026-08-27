//! Gemini Files API client for file uploads.
//!
//! This module provides a client for the Gemini Files API, which allows uploading
//! files for use in multimodal requests. Files are stored for 48 hours before expiration.
//!
//! See: <https://ai.google.dev/api/files>

use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
use async_fs;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use zenwave::{Client, client, header};

use crate::config::{AuthMode, GeminiConfig, USER_AGENT};
use crate::error::GeminiError;

/// The base URL for the Files API upload endpoint.
const UPLOAD_BASE_URL: &str = "https://generativelanguage.googleapis.com/upload/v1beta/files";

/// Gemini file state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FileState {
    /// File is being processed.
    Processing,
    /// File is ready for use.
    Active,
    /// File processing failed.
    Failed,
    /// Unknown state.
    #[serde(other)]
    Unknown,
}

/// A file uploaded to the Gemini Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFile {
    /// File resource name (e.g., "files/abc123").
    pub name: String,
    /// Display name.
    #[serde(default)]
    pub display_name: String,
    /// MIME type.
    pub mime_type: String,
    /// File size in bytes.
    #[serde(default)]
    pub size_bytes: String,
    /// File creation time.
    #[serde(default)]
    pub create_time: String,
    /// File update time.
    #[serde(default)]
    pub update_time: String,
    /// Expiration time (RFC 3339 format).
    #[serde(default)]
    pub expiration_time: String,
    /// URI to use in requests.
    #[serde(default)]
    pub uri: String,
    /// Current file state.
    pub state: FileState,
    /// SHA-256 hash of the file.
    #[serde(default)]
    pub sha256_hash: String,
}

impl GeminiFile {
    /// Parse the expiration time to `SystemTime`.
    ///
    /// Returns `None` if parsing fails.
    #[must_use]
    pub fn expiration(&self) -> Option<SystemTime> {
        parse_rfc3339(&self.expiration_time)
    }

    /// Check if the file is ready for use.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state == FileState::Active
    }
}

/// Response wrapper for file upload.
#[derive(Debug, Deserialize)]
pub struct UploadFileResponse {
    /// The uploaded file.
    pub file: GeminiFile,
}

/// Response wrapper for file list.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListFilesResponse {
    /// The files.
    #[serde(default)]
    pub files: Vec<GeminiFile>,
    /// Next page token.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

/// Upload raw bytes to the Gemini Files API.
///
/// This is the cross-platform upload entry point (including wasm).
///
/// # Errors
///
/// Returns an error if the upload request fails or the provider rejects
/// the file.
pub async fn upload_bytes(
    cfg: &GeminiConfig,
    file_name: &str,
    mime_type: &str,
    data: Vec<u8>,
) -> Result<GeminiFile, GeminiError> {
    // Build upload URL
    let upload_url = build_upload_url(cfg);

    // For simplicity, we use the simple (non-resumable) upload for files < 20MB
    // Resumable uploads would require more complex handling
    let metadata = serde_json::json!({
        "file": {
            "displayName": file_name
        }
    });

    let mut backend = client();
    let mut builder = backend
        .post(&upload_url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?
        .header("X-Goog-Upload-Protocol", "multipart")
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    // Build multipart body
    let boundary = format!(
        "----aither{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    let mut body = Vec::new();

    // Part 1: Metadata (JSON)
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
    let metadata_json = serde_json::to_string(&metadata).map_err(GeminiError::Json)?;
    body.extend_from_slice(metadata_json.as_bytes());
    body.extend_from_slice(b"\r\n");

    // Part 2: File content
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(&data);
    body.extend_from_slice(b"\r\n");

    // End boundary
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/related; boundary={boundary}");
    let builder = builder
        .header(header::CONTENT_TYPE.as_str(), &content_type)
        .map_err(GeminiError::from_http)?
        .bytes_body(body);

    let response: UploadFileResponse = builder.json().await.map_err(GeminiError::from_http)?;

    Ok(response.file)
}

/// Upload a local file path to the Gemini Files API.
///
/// This is a native-only convenience API. For wasm, use [`upload_bytes`].
///
/// # Errors
///
/// Returns an error when the file cannot be read or upload fails.
#[cfg(not(target_arch = "wasm32"))]
///
/// # Errors
///
/// Returns an error if the file cannot be read, the upload fails, or the
/// provider rejects the file.
pub async fn upload_file(cfg: &GeminiConfig, path: &Path) -> Result<GeminiFile, GeminiError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let mime_type = crate::mime::mime_from_path(path).unwrap_or("application/octet-stream");
    let data = async_fs::read(path)
        .await
        .map_err(|e| GeminiError::Parse(format!("Failed to read file: {e}")))?;

    upload_bytes(cfg, file_name, mime_type, data).await
}

/// Delete a file from the Gemini Files API.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn delete_file(cfg: &GeminiConfig, name: &str) -> Result<(), GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .delete(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    let _response = builder.await.map_err(GeminiError::from_http)?;
    Ok(())
}

/// Get file metadata.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn get_file(cfg: &GeminiConfig, name: &str) -> Result<GeminiFile, GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    builder.json().await.map_err(GeminiError::from_http)
}

/// List uploaded files.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `page_size` - Maximum number of files to return
/// * `page_token` - Page token from previous response
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn list_files(
    cfg: &GeminiConfig,
    page_size: Option<u32>,
    page_token: Option<&str>,
) -> Result<ListFilesResponse, GeminiError> {
    let mut url = cfg.endpoint("files");

    if let Some(size) = page_size {
        write!(url, "&pageSize={size}").expect("write to String cannot fail");
    }
    if let Some(token) = page_token {
        write!(url, "&pageToken={token}").expect("write to String cannot fail");
    }

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    builder.json().await.map_err(GeminiError::from_http)
}

/// Build the upload URL with authentication.
fn build_upload_url(cfg: &GeminiConfig) -> String {
    let mut url = UPLOAD_BASE_URL.to_string();
    if cfg.auth == AuthMode::Query {
        url.push_str("?key=");
        url.push_str(&cfg.api_key);
    }
    url
}

/// Parse an RFC 3339 timestamp to `SystemTime`.
///
/// Returns `None` for an empty or malformed timestamp — including the empty
/// string the API sends for a file that never expires.
fn parse_rfc3339(s: &str) -> Option<SystemTime> {
    let timestamp: jiff::Timestamp = s.trim().parse().ok()?;
    Some(SystemTime::from(timestamp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rfc3339_expiration() {
        // 2024-01-15T10:30:00Z is 1_705_314_600 seconds after the epoch.
        let parsed = parse_rfc3339("2024-01-15T10:30:00Z").expect("should parse");
        let secs = parsed
            .duration_since(std::time::UNIX_EPOCH)
            .expect("after the epoch")
            .as_secs();
        assert_eq!(secs, 1_705_314_600);

        // Fractional seconds and a numeric offset are both valid RFC 3339.
        assert!(parse_rfc3339("2024-01-15T10:30:00.123456Z").is_some());
        assert!(parse_rfc3339("  2024-01-15T11:30:00+01:00  ").is_some());

        // What the API sends when a file never expires, and outright garbage.
        assert!(parse_rfc3339("").is_none());
        assert!(parse_rfc3339("not a timestamp").is_none());
    }

    #[test]
    fn test_mime_from_path() {
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/image.png")),
            Some("image/png")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/video.mp4")),
            Some("video/mp4")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/audio.mp3")),
            Some("audio/mpeg")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/doc.pdf")),
            Some("application/pdf")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/unknown.aither_unknown")),
            None
        );
    }
}
