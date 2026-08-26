//! Constants for GitHub Copilot API integration.

/// Base URL for GitHub Copilot API.
pub const COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";

/// GitHub OAuth device authorization endpoint.
pub const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";

/// GitHub OAuth token endpoint.
pub const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

/// GitHub API endpoint to exchange OAuth token for Copilot session token.
pub const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// OAuth client ID for GitHub Copilot (VS Code's client ID).
pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// Default model for chat completions.
///
/// GitHub designated GPT-5.3-Codex the base and long-term-support model on
/// 2026-03-18: it is what Copilot falls back to when no other model is enabled,
/// which makes it the one identifier most likely to resolve for any given
/// account. Models here come and go on GitHub's schedule, not this crate's, so
/// the default tracks the LTS designation rather than the newest release.
pub const DEFAULT_MODEL: &str = GPT5_3_CODEX;

// Models GitHub Copilot currently serves. Availability varies by plan and by
// enterprise policy, so a constant here means GitHub lists the model, not that
// a given account can reach it.

/// GPT-5.3-Codex — the base and long-term-support model.
pub const GPT5_3_CODEX: &str = "gpt-5.3-codex";
/// GPT-5.6 Sol — frontier capability.
pub const GPT5_6_SOL: &str = "gpt-5.6-sol";
/// GPT-5.6 Terra — strong performance at lower cost.
pub const GPT5_6_TERRA: &str = "gpt-5.6-terra";
/// GPT-5.6 Luna — efficient, for high-volume work.
pub const GPT5_6_LUNA: &str = "gpt-5.6-luna";
/// GPT-5.5.
pub const GPT5_5: &str = "gpt-5.5";
/// GPT-5.4.
pub const GPT5_4: &str = "gpt-5.4";
/// GPT-5.4 mini.
pub const GPT5_4_MINI: &str = "gpt-5.4-mini";
/// GPT-5 mini.
pub const GPT5_MINI: &str = "gpt-5-mini";

/// Claude Opus 5.
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";
/// Claude Sonnet 5.
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";
/// Claude Fable 5.
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";
/// Claude Opus 4.8.
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// Claude Opus 4.7.
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
/// Claude Haiku 4.5.
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

/// Editor version header value (identifies as VS Code).
pub const EDITOR_VERSION: &str = "vscode/1.96.0";

/// Copilot integration ID header value.
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
