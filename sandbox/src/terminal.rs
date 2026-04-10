//! Terminal command tool implementation with heel sandboxing.
//!
//! Provides the main `terminal` tool that LLMs use to execute terminal work.
//! Despite the historical tool name, execution does not have to go through a
//! literal `terminal` process: simple commands may be executed directly, while shell
//! syntax falls back to a real shell when needed.
//! Commands run in a sandbox with configurable permission modes.
//!
//! Each `TerminalTool` creates a shared working directory with four random words
//! (e.g., `amber-forest-thunder-pearl/`). All terminal executions share this
//! directory, but each execution gets a fresh sandbox (new TTY/process).

#[cfg(unix)]
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::{
    borrow::Cow,
    io::{Read, Write},
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aither_core::llm::{Tool, ToolResult};
use askama::Template;
use async_channel::{Receiver, Sender};
#[cfg(unix)]
use async_io::Async;
use executor_core::{Executor, Task};
#[cfg(unix)]
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    AllowAll, DenyAll, DomainRequest, IpcRouter, NetworkPolicy, Sandbox, SandboxConfig,
    SecurityConfig, StdioConfig, WorkingDir,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

use crate::{
    builtin::builtin_router,
    command::ToolRegistry,
    job_registry::{JobRegistry, job_registry_channel},
    output::{
        Content, INLINE_OUTPUT_LIMIT, MediaResolution, OutputEntry, OutputFormat, OutputStore,
        save_raw_to_file,
    },
    permission::{PermissionError, PermissionHandler, TerminalMode},
    shell_session::{
        ContainerShellRuntime, ShellBackend, ShellSessionRegistry, SshRuntimeProfile,
        bootstrap_ssh_runtime,
    },
    stdin_watch::{
        STDIN_WATCH_INTERVAL, TERMINAL_STDIN_BLOCKED_NOTICE, detect_stdin_blocked_for_local_pid,
    },
};

/// Generate a random four-word ID (e.g., "amber-forest-thunder-pearl").
fn random_task_id() -> String {
    crate::naming::random_word_slug(4)
}

fn supports_stdin_blocked_notice(backend: ShellBackend) -> bool {
    matches!(backend, ShellBackend::Container)
        || (cfg!(target_os = "linux") && matches!(backend, ShellBackend::Local))
}

#[derive(Clone)]
struct PermissionNetworkPolicy<P> {
    permission_handler: Arc<P>,
}

impl<P: PermissionHandler + 'static> NetworkPolicy for PermissionNetworkPolicy<P> {
    async fn check(&self, request: &DomainRequest) -> bool {
        self.permission_handler
            .check_domain(request.target(), request.port())
            .await
    }
}

fn parse_direct_command(script: &str) -> Result<(String, Vec<String>), TerminalError> {
    let parts = split_shell_words(script)
        .map_err(|error| TerminalError::Execution(format!("failed to parse command: {error}")))?;
    let Some((program, args)) = parts.split_first() else {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    };
    Ok((program.clone(), args.to_vec()))
}

enum TerminalLaunch {
    Direct { program: String, args: Vec<String> },
    Shell { program: String, args: Vec<String> },
}

fn script_requires_shell(script: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in script.chars() {
        if escaped {
            escaped = false;
            continue;
        }

        match ch {
            '\\' if !in_single => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '\n' | '\r' | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '*' | '?' | '['
            | ']' | '{' | '}' | '!' | '~'
                if !in_single && !in_double =>
            {
                return true;
            }
            _ => {}
        }
    }

    false
}

fn looks_like_env_assignment(token: &str) -> bool {
    let Some((name, _value)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty() && name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn shell_launch(script: &str) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let shell = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string());
        return (shell, vec!["/C".to_string(), script.to_string()]);
    }

    #[cfg(not(windows))]
    {
        if let Some(shell) = std::env::var_os("SHELL").and_then(|value| value.into_string().ok()) {
            let shell_name = std::path::Path::new(&shell)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if shell_name.contains("pwsh") || shell_name.contains("powershell") {
                return (
                    shell,
                    vec![
                        "-NoLogo".to_string(),
                        "-NoProfile".to_string(),
                        "-Command".to_string(),
                        script.to_string(),
                    ],
                );
            }
            return (shell, vec!["-c".to_string(), script.to_string()]);
        }

        (
            "/bin/sh".to_string(),
            vec!["-c".to_string(), script.to_string()],
        )
    }
}

fn build_terminal_launch(script: &str) -> Result<TerminalLaunch, TerminalError> {
    let script = script.trim();
    if script.is_empty() {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    }

    if script_requires_shell(script) {
        let (program, args) = shell_launch(script);
        return Ok(TerminalLaunch::Shell { program, args });
    }

    let (program, args) = parse_direct_command(script)?;
    if looks_like_env_assignment(&program) {
        let (program, args) = shell_launch(script);
        return Ok(TerminalLaunch::Shell { program, args });
    }

    Ok(TerminalLaunch::Direct { program, args })
}

async fn ensure_mode_allowed<P: PermissionHandler>(
    permission_handler: &P,
    mode: TerminalMode,
    script: &str,
) -> Result<(), TerminalError> {
    if !mode.requires_approval() {
        return Ok(());
    }

    let allowed = permission_handler.check(mode, script).await?;
    if allowed {
        Ok(())
    } else {
        Err(TerminalError::PermissionDenied(mode))
    }
}

/// Execute terminal commands in a sandboxed environment.
///
/// The primary interface to all system capabilities. Scripts run in a sandbox
/// with full read access to the host filesystem but writes contained to a
/// dedicated working directory.
///
/// ## Output Handling
///
/// By default, stdout is **compressed** before being returned. Compression
/// is semantics-preserving: the meaning is identical to the raw output, but
/// the representation may differ. Specifically:
///
/// - **JSON** outputs (arrays of objects or single objects) are automatically
///   converted to **TSV** with dot-notation flattened column headers, but
///   only when the TSV is smaller than the original JSON.
/// - **Source code** outputs (detected via content analysis) are folded
///   using syntax-aware block folding with line numbers. The full raw
///   output is saved to a file whose URL is included in the result.
/// - **Empty lines** and **invisible/control characters** are stripped.
///
/// If you need the exact verbatim output (e.g., for checksums, binary
/// protocols, or diffing), either set `raw: true`, pipe through further
/// processing in the script, or redirect to a file within the script.
///
/// Large outputs are automatically saved to file to manage context. When this
/// happens, you receive the file path and can process it using standard Unix
/// tools (head, tail, grep, less) or pipe through `ask` for summarization.
///
/// ## Built-in Commands
///
/// The sandbox provides these commands without network access:
/// - `websearch "query"` - search the web, returns titles/URLs/snippets
/// - `webfetch "url"` - fetch URL content as markdown
/// - `ask "prompt"` - query a fast LLM about piped content (saves context)
/// - `subagent --subagent "<type-or-path>" --prompt "<prompt>"` - launch specialized subagents
/// - `todo` - manage task list
///
/// ## Execution Modes
///
/// `terminal` is stateless. Each call selects its own runtime mode.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminalArgs {
    /// Short human-readable explanation of what this command is doing.
    /// This is shown in the UI instead of echoing the shell source.
    pub description: String,

    /// The terminal command payload to execute.
    pub script: String,

    /// Runtime execution mode.
    /// - "default": local runtime execution with network enabled
    /// - "sandboxed": local runtime execution with network denied
    /// - "unsafe": direct host execution (heel profile only)
    /// - "ssh": execute on a preconfigured SSH server (`ssh_server_id` required when multiple are configured)
    #[serde(default)]
    pub mode: TerminalExecutionMode,

    /// SSH server identifier used when `mode` is `ssh`.
    /// Optional when exactly one SSH server is configured.
    #[serde(default)]
    pub ssh_server_id: Option<String>,

    /// Expected output format for proper handling.
    /// - "text" (default): plain text
    /// - "image": image data (auto-loaded to context)
    /// - "video": video data
    /// - "binary": binary data
    /// - "auto": auto-detect
    #[serde(default)]
    pub expect: OutputFormat,

    /// Requested delivery resolution for media returned to the model.
    /// - "auto" (default): preserve the source resolution
    /// - "low": downscale to a 512px bounding box
    /// - "medium": downscale to a 1024px bounding box
    /// - "high": downscale to a 2048px bounding box
    /// - "native": preserve original media bytes without resizing
    #[serde(default)]
    pub resolution: MediaResolution,

    /// Per-command timeout in seconds.
    /// - 0: run in background immediately
    /// - >0: run foreground up to timeout, then move to background on timeout
    pub timeout: u64,

    /// Maximum number of output lines to include inline.
    /// For foreground-complete executions, this is the inline line budget
    /// before output offloads to file.
    /// For timeout-promoted background executions, only the first `max_lines`
    /// are returned immediately, while full output is redirected to file.
    /// Clamped to 800 max.
    /// Default: 200.
    #[serde(default = "default_max_lines")]
    pub max_lines: u32,

    /// When true, skip all output compression and return the verbatim
    /// stdout bytes. Use this when you need exact byte-level fidelity
    /// (checksums, binary protocols, diff inputs). Default: false.
    #[serde(default)]
    pub raw: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TerminalExecutionMode {
    #[default]
    Default,
    Sandboxed,
    Unsafe,
    Ssh,
}

const fn default_max_lines() -> u32 {
    200
}

const MAX_LINES_CEILING: u32 = 800;

/// Result of a terminal execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalResult {
    /// stdout output.
    pub stdout: OutputEntry,

    /// stderr output (if non-empty).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<OutputEntry>,

    /// Exit code of the script.
    pub exit_code: i32,

    /// Task ID for background execution.
    /// Four random words like "amber-forest-thunder-pearl".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,

    /// Status for background tasks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

enum ForegroundDecision {
    Completed(Result<TerminalResult, String>),
    PromoteToBackground(Option<String>),
}

#[derive(Clone)]
struct BackgroundStartup {
    tx: async_channel::Sender<Result<(), String>>,
    reported: Arc<AtomicBool>,
}

impl BackgroundStartup {
    fn new(tx: async_channel::Sender<Result<(), String>>) -> Self {
        Self {
            tx,
            reported: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn report_ready(&self) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Ok(())).await;
        }
    }

    async fn report_failure(&self, error: &TerminalError) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Err(error.to_string())).await;
        }
    }
}

/// A completed background task result.
#[derive(Debug)]
pub struct CompletedTask {
    /// The task ID (four random words like "amber-forest-thunder-pearl").
    pub task_id: String,
    /// The original script that was executed.
    pub script: String,
    /// The result of execution.
    pub result: Result<TerminalResult, TerminalError>,
}

/// Stage of a permission-lifecycle event for `terminal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionEventStage {
    Waiting,
    Resolved,
}

/// Permission-lifecycle event emitted by `terminal` while waiting for approval.
#[derive(Debug, Clone)]
pub struct PermissionEvent {
    pub mode: TerminalMode,
    pub script: String,
    pub stage: PermissionEventStage,
}

/// Receiver for completed background tasks.
///
/// This can be cloned and used independently of the `TerminalTool` to poll for
/// completed background tasks. Multiple receivers share the same channel,
/// so completed tasks are distributed among receivers.
#[derive(Clone)]
pub struct BackgroundTaskReceiver {
    rx: Receiver<CompletedTask>,
    running_tasks: Arc<AtomicUsize>,
}

impl BackgroundTaskReceiver {
    /// Takes all completed background tasks without blocking.
    ///
    /// Returns an empty vector if no tasks have completed.
    #[must_use]
    pub fn take_completed(&self) -> Vec<CompletedTask> {
        let mut completed = Vec::new();
        while let Ok(task) = self.rx.try_recv() {
            completed.push(task);
        }
        completed
    }

    /// Checks if there are any completed tasks waiting.
    #[must_use]
    pub fn has_completed(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Checks if there are any background tasks still running.
    ///
    /// This works by checking the sender count - `TerminalTool` holds one sender,
    /// and each running background task holds a cloned sender.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.running_tasks.load(Ordering::Acquire) > 0
    }

    /// Waits for the next completed task.
    ///
    /// Returns `None` if the channel is closed (all senders dropped).
    pub async fn recv(&self) -> Option<CompletedTask> {
        self.rx.recv().await.ok()
    }

    /// Waits for a completed task with a timeout.
    ///
    /// Returns `None` if no task completes within the timeout or if the channel is closed.
    pub async fn recv_timeout(&self, duration: std::time::Duration) -> Option<CompletedTask> {
        futures_lite::future::or(async { self.rx.recv().await.ok() }, async {
            futures_lite::future::yield_now().await;
            async_io::Timer::after(duration).await;
            None
        })
        .await
    }
}

impl std::fmt::Debug for BackgroundTaskReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundTaskReceiver")
            .field("pending", &self.rx.len())
            .finish()
    }
}

/// Receiver for permission wait/resume events.
#[derive(Clone)]
pub struct PermissionEventReceiver {
    rx: Receiver<PermissionEvent>,
}

impl PermissionEventReceiver {
    #[must_use]
    pub fn take_pending(&self) -> Vec<PermissionEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub async fn recv(&self) -> Option<PermissionEvent> {
        self.rx.recv().await.ok()
    }
}

impl std::fmt::Debug for PermissionEventReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermissionEventReceiver")
            .field("pending", &self.rx.len())
            .finish()
    }
}

/// Factory for creating child terminal tools asynchronously.
#[derive(Clone)]
pub struct TerminalToolFactory {
    tx: Sender<Sender<crate::command::DynTerminalTool>>,
}

/// Receiver that serves terminal tool creation requests.
pub struct TerminalToolFactoryReceiver {
    rx: Receiver<Sender<crate::command::DynTerminalTool>>,
}

/// Errors that can occur when requesting a child terminal tool.
#[derive(Debug, thiserror::Error)]
pub enum TerminalToolFactoryError {
    /// The factory service is not running.
    #[error("terminal tool factory is not available")]
    Unavailable,
    /// The factory failed to return a tool.
    #[error("terminal tool factory failed to return a tool")]
    NoResponse,
}

/// Creates a factory channel pair for spawning child terminal tools.
#[must_use]
pub fn terminal_tool_factory_channel() -> (TerminalToolFactory, TerminalToolFactoryReceiver) {
    let (tx, rx) = async_channel::unbounded();
    (
        TerminalToolFactory { tx },
        TerminalToolFactoryReceiver { rx },
    )
}

impl TerminalToolFactory {
    /// Requests a new child terminal tool from the factory service.
    pub async fn create(
        &self,
    ) -> Result<crate::command::DynTerminalTool, TerminalToolFactoryError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(reply_tx)
            .await
            .map_err(|_| TerminalToolFactoryError::Unavailable)?;
        reply_rx
            .recv()
            .await
            .map_err(|_| TerminalToolFactoryError::NoResponse)
    }
}

impl TerminalToolFactoryReceiver {
    async fn serve<P, E>(self, terminal_tool: TerminalTool<P, E, Configured>)
    where
        P: PermissionHandler + 'static,
        E: Executor + Clone + 'static,
    {
        while let Ok(reply_tx) = self.rx.recv().await {
            let tool = terminal_tool.child().to_dyn();
            if reply_tx.send(tool).await.is_err() {
                warn!("terminal tool factory response channel dropped");
            }
        }
    }
}

/// The terminal tool for executing scripts in a sandbox.
///
/// Creates a shared working directory with four random words that persists
/// across all executions. Each execution creates a fresh sandbox but shares
/// the working directory.
///
/// The executor type `E` determines how async tasks are spawned for the IPC server.
/// Use `TokioExecutor` when running in a tokio runtime.
/// Marker type for a terminal tool without a configured registry.
#[derive(Clone, Debug)]
pub struct Unconfigured;

/// Marker type for a terminal tool with a configured registry.
#[derive(Clone)]
pub struct Configured {
    registry: Arc<ToolRegistry>,
}

/// The terminal tool for executing scripts in a sandbox.
pub struct TerminalTool<P, E, State = Unconfigured> {
    /// Shared working directory (four random words, e.g., `amber-forest-thunder-pearl/`)
    working_dir: PathBuf,
    /// Runtime registry for container and ssh execution metadata.
    shell_sessions: ShellSessionRegistry,
    /// Permission handler wrapped in Arc for sharing between parent and child tools.
    permission_handler: Arc<P>,
    executor: E,
    output_store: Arc<OutputStore>,
    job_registry: JobRegistry,
    /// Channel for receiving completed background tasks.
    completed_rx: Receiver<CompletedTask>,
    /// Channel for sending completed background tasks (cloned for each background task).
    completed_tx: Sender<CompletedTask>,
    /// Number of still-running spawned executions associated with this tool.
    running_tasks: Arc<AtomicUsize>,
    /// Channel for permission lifecycle events.
    permission_rx: Receiver<PermissionEvent>,
    /// Channel sender for permission lifecycle events.
    permission_tx: Sender<PermissionEvent>,
    /// Additional paths that should be writable in the sandbox.
    writable_paths: Vec<PathBuf>,
    /// Additional paths that should be readable (but not writable) in the sandbox.
    readable_paths: Vec<PathBuf>,
    /// Tool registry state.
    registry: State,
}

// Manual Clone impl because P doesn't need to be Clone (we use Arc<P>)
impl<P, E: Clone, State: Clone> Clone for TerminalTool<P, E, State> {
    fn clone(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self.permission_handler.clone(),
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            completed_rx: self.completed_rx.clone(),
            completed_tx: self.completed_tx.clone(),
            running_tasks: self.running_tasks.clone(),
            permission_rx: self.permission_rx.clone(),
            permission_tx: self.permission_tx.clone(),
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<P, E: Executor + Clone + 'static> TerminalTool<P, E, Unconfigured> {
    /// Injects the shared runtime registry used by `terminal` execution mode resolution.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        self.shell_sessions = sessions;
        self
    }

    /// Sets dynamic runtime availability for terminal execution.
    #[must_use]
    pub fn with_shell_runtime_availability(
        mut self,
        availability: crate::shell_session::ShellRuntimeAvailability,
    ) -> Self {
        self.shell_sessions = self.shell_sessions.with_availability(availability);
        self
    }

    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.shell_sessions = self.shell_sessions.with_container_runtime(runtime);
        self
    }

    /// Creates a new terminal tool with permission handler and executor.
    ///
    /// Creates a random four-word working directory under the specified parent directory.
    /// The executor is used to spawn async tasks for the IPC server.
    pub async fn new_in(
        parent_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, TerminalError> {
        let parent_dir = parent_dir.as_ref();

        // Ensure parent directory exists
        async_fs::create_dir_all(parent_dir).await?;

        // Create random four-word working directory
        let working_dir = WorkingDir::random_in(parent_dir).map_err(|e| {
            TerminalError::SandboxSetup(format!("failed to create working dir: {e}"))
        })?;
        let working_dir_path = working_dir.path().to_path_buf();

        info!(working_dir = %working_dir_path.display(), "created terminal tool working directory");

        // Create outputs directory inside working dir
        let outputs_dir = working_dir_path.join("outputs");
        async_fs::create_dir_all(&outputs_dir).await?;

        // Create output store
        let output_store = Arc::new(OutputStore::new(&outputs_dir).await?);

        // Start job registry service
        let (job_registry, job_registry_service) = job_registry_channel();
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(Default::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            registry: Unconfigured,
        })
    }

    /// Creates a new terminal tool with permission handler and executor,
    /// using the provided directory directly as the working directory.
    ///
    /// Unlike `new_in` which creates a random subdirectory, this method
    /// uses the exact path provided. Use this when you want explicit control
    /// over the working directory location.
    pub async fn new_exact(
        working_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, TerminalError> {
        let working_dir_path = working_dir.as_ref().to_path_buf();

        // Ensure working directory exists
        async_fs::create_dir_all(&working_dir_path).await?;

        // Create outputs directory inside working dir
        let outputs_dir = working_dir_path.join("outputs");
        async_fs::create_dir_all(&outputs_dir).await?;

        // Create output store
        let output_store = Arc::new(OutputStore::new(&outputs_dir).await?);

        // Start job registry service
        let (job_registry, job_registry_service) = job_registry_channel();
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(Default::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            registry: Unconfigured,
        })
    }

    /// Attaches a tool registry to this terminal tool, enabling IPC command dispatch.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> TerminalTool<P, E, Configured> {
        TerminalTool {
            working_dir: self.working_dir,
            shell_sessions: self.shell_sessions,
            permission_handler: self.permission_handler,
            executor: self.executor,
            output_store: self.output_store,
            job_registry: self.job_registry,
            completed_rx: self.completed_rx,
            completed_tx: self.completed_tx,
            running_tasks: self.running_tasks,
            permission_rx: self.permission_rx,
            permission_tx: self.permission_tx,
            writable_paths: self.writable_paths,
            readable_paths: self.readable_paths,
            registry: Configured { registry },
        }
    }
}

impl<P, E, State> TerminalTool<P, E, State>
where
    E: Executor + Clone + 'static,
    State: Clone,
{
    /// Adds additional writable paths to the sandbox configuration.
    ///
    /// These paths will be writable in sandboxed and network modes.
    pub fn with_writable_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        self.writable_paths
            .extend(paths.into_iter().map(Into::into));
        self
    }

    /// Adds additional readable (but not writable) paths to the sandbox configuration.
    ///
    /// These paths will be readable in all sandbox modes, even in strict
    /// filesystem mode where reads outside the sandbox are normally denied.
    pub fn with_readable_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        self.readable_paths
            .extend(paths.into_iter().map(Into::into));
        self
    }

    /// Creates a child `TerminalTool` that shares the same sandbox and permission handler
    /// but has independent background task tracking.
    ///
    /// Use this to create terminal tools for subagents that:
    /// - Share the same working directory and output store
    /// - Share the same permission handler (security policies enforced consistently)
    /// - Have independent completion channels (no message mixup)
    pub fn child(&self) -> Self {
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self.permission_handler.clone(), // Arc clone - shares handler
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            registry: self.registry.clone(),
        }
    }

    /// Returns the working directory path.
    pub const fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Returns the outputs directory path.
    pub fn outputs_dir(&self) -> PathBuf {
        self.working_dir.join("outputs")
    }

    /// Returns the output store.
    pub const fn output_store(&self) -> &Arc<OutputStore> {
        &self.output_store
    }

    /// Returns the job registry handle.
    pub fn job_registry(&self) -> JobRegistry {
        self.job_registry.clone()
    }

    /// Returns a receiver for completed background tasks.
    ///
    /// The returned `BackgroundTaskReceiver` can be used independently of the `TerminalTool`
    /// to poll for completed background tasks. This is useful for integrating with
    /// the Agent's main loop.
    pub fn background_receiver(&self) -> BackgroundTaskReceiver {
        BackgroundTaskReceiver {
            rx: self.completed_rx.clone(),
            running_tasks: self.running_tasks.clone(),
        }
    }

    /// Returns a receiver for permission lifecycle events.
    pub fn permission_receiver(&self) -> PermissionEventReceiver {
        PermissionEventReceiver {
            rx: self.permission_rx.clone(),
        }
    }

    /// Takes all completed background tasks without blocking.
    ///
    /// Returns an empty vector if no tasks have completed.
    /// The agent should call this after each turn to check for completed background tasks.
    pub fn take_completed(&self) -> Vec<CompletedTask> {
        let mut completed = Vec::new();
        while let Ok(task) = self.completed_rx.try_recv() {
            completed.push(task);
        }
        completed
    }

    /// Checks if there are any pending background tasks.
    pub fn has_pending_tasks(&self) -> bool {
        !self.completed_rx.is_empty() || self.running_tasks.load(Ordering::Acquire) > 0
    }
}

impl<P, E> TerminalTool<P, E, Configured>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    const fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry.registry
    }

    /// Starts a background service that produces child terminal tools on demand.
    pub fn start_factory_service(&self, receiver: TerminalToolFactoryReceiver) {
        let tool = self.clone();
        self.executor
            .spawn(async move { receiver.serve(tool).await })
            .detach();
    }

    /// Converts this `TerminalTool` into a type-erased `DynTerminalTool`.
    ///
    /// This is useful for creating child terminal tools for subagents where
    /// the concrete type cannot be known at compile time.
    pub fn to_dyn(self) -> crate::command::DynTerminalTool {
        use crate::command::{DynTerminalTool, DynToolHandler};
        use aither_core::llm::tool::ToolDefinition;
        use serde_json::Value;
        use std::sync::Arc;

        // Create the definition - description comes from TerminalArgs rustdoc
        let schema = schemars::schema_for!(TerminalArgs);
        let schema_value: Value = schema.to_value();
        let description = schema_value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let definition =
            ToolDefinition::from_parts("terminal".into(), description.into(), schema_value);

        // Create the handler
        let tool = Arc::new(self);
        let handler: DynToolHandler = Arc::new(move |args: &str| {
            let tool = tool.clone();
            let args_str = args.to_string();
            Box::pin(async move {
                match serde_json::from_str::<TerminalArgs>(&args_str) {
                    Ok(parsed) => match tool.call(parsed).await {
                        Ok(output) => output,
                        Err(error) => ToolResult::error(format!("Error: {error}")),
                    },
                    Err(error) => ToolResult::error(format!("Parse error: {error}")),
                }
            })
        });

        DynTerminalTool {
            definition,
            handler,
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool
    for TerminalTool<P, E, Configured>
{
    fn name(&self) -> Cow<'static, str> {
        "terminal".into()
    }

    type Arguments = TerminalArgs;
    type Res = ToolResult;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<Self::Res> {
        if arguments.description.trim().is_empty() {
            return Err(anyhow::anyhow!("terminal description must not be empty").into());
        }
        let task_id = random_task_id();
        let script = arguments.script.clone();
        let execution_id = format!("exec-{task_id}");
        let expect = arguments.expect;
        let resolution = arguments.resolution;
        let max_lines = arguments.max_lines.min(MAX_LINES_CEILING) as usize;
        let raw = arguments.raw;
        let timeout = arguments.timeout;

        let (backend, mode, ssh_target, ssh_runtime, container_runtime) = match arguments.mode {
            TerminalExecutionMode::Default => match self.shell_sessions.resolve_local_backend() {
                Ok(backend) => {
                    let container_runtime = if matches!(backend, ShellBackend::Container) {
                        Some(
                            self.shell_sessions
                                .container_runtime()
                                .cloned()
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "missing container runtime for container backend"
                                    )
                                })?,
                        )
                    } else {
                        None
                    };
                    (
                        backend,
                        TerminalMode::Network,
                        None,
                        None,
                        container_runtime,
                    )
                }
                Err(local_error) => {
                    self.shell_sessions
                        .ensure_ssh_available()
                        .map_err(anyhow::Error::msg)?;
                    let server = self
                        .shell_sessions
                        .default_ssh_server()
                        .map_err(|ssh_error| anyhow::anyhow!("{local_error}; {ssh_error}"))?;
                    let runtime = bootstrap_ssh_runtime(
                        &server.target,
                        &self.shell_sessions.ssh_authorizer(),
                    )
                    .await?;
                    (
                        ShellBackend::Ssh,
                        TerminalMode::Network,
                        Some(server.target),
                        Some(runtime),
                        None,
                    )
                }
            },
            TerminalExecutionMode::Sandboxed => {
                let backend = self
                    .shell_sessions
                    .resolve_local_backend()
                    .map_err(anyhow::Error::msg)?;
                let container_runtime = if matches!(backend, ShellBackend::Container) {
                    Some(
                        self.shell_sessions
                            .container_runtime()
                            .cloned()
                            .ok_or_else(|| {
                                anyhow::anyhow!("missing container runtime for container backend")
                            })?,
                    )
                } else {
                    None
                };
                (
                    backend,
                    TerminalMode::Sandboxed,
                    None,
                    None,
                    container_runtime,
                )
            }
            TerminalExecutionMode::Unsafe => {
                let backend = self
                    .shell_sessions
                    .resolve_local_backend()
                    .map_err(anyhow::Error::msg)?;
                if !matches!(backend, ShellBackend::Local) {
                    return Err(anyhow::anyhow!(
                        "unsafe mode is only available in heel local runtime"
                    ));
                }
                (backend, TerminalMode::Unsafe, None, None, None)
            }
            TerminalExecutionMode::Ssh => {
                self.shell_sessions
                    .ensure_ssh_available()
                    .map_err(anyhow::Error::msg)?;
                let server = match arguments.ssh_server_id.as_deref() {
                    Some(server_id) => self
                        .shell_sessions
                        .resolve_ssh_server(server_id)
                        .map_err(anyhow::Error::msg)?,
                    None => self
                        .shell_sessions
                        .default_ssh_server()
                        .map_err(anyhow::Error::msg)?,
                };
                let runtime =
                    bootstrap_ssh_runtime(&server.target, &self.shell_sessions.ssh_authorizer())
                        .await?;
                (
                    ShellBackend::Ssh,
                    TerminalMode::Network,
                    Some(server.target),
                    Some(runtime),
                    None,
                )
            }
        };

        let permission_waits = self
            .permission_handler
            .will_wait_for_approval(mode, &script)
            .await;
        if permission_waits {
            let _ = self
                .permission_tx
                .send(PermissionEvent {
                    mode,
                    script: script.clone(),
                    stage: PermissionEventStage::Waiting,
                })
                .await;
        }
        let permission_result =
            ensure_mode_allowed(self.permission_handler.as_ref(), mode, &script).await;
        if permission_waits {
            let _ = self
                .permission_tx
                .send(PermissionEvent {
                    mode,
                    script: script.clone(),
                    stage: PermissionEventStage::Resolved,
                })
                .await;
        }
        permission_result.map_err(anyhow::Error::new)?;

        if matches!(backend, ShellBackend::Container) && container_runtime.is_none() {
            return Err(anyhow::anyhow!(
                "missing container runtime for container backend"
            ));
        }
        let working_dir = self.working_dir.clone();
        let writable_paths = self.writable_paths.clone();
        let readable_paths = self.readable_paths.clone();
        let executor = self.executor.clone();
        let registry = self.registry().clone();
        let permission_handler = self.permission_handler.clone();
        let store_dir = self.output_store.dir().to_path_buf();
        let store_dir_for_spawn = store_dir.clone();
        let completed_tx = self.completed_tx.clone();
        let job_registry = self.job_registry.clone();
        let running_tasks = self.running_tasks.clone();
        let background_mode = Arc::new(AtomicBool::new(timeout == 0));
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (startup_tx, startup_rx) = async_channel::bounded(1);
        let background_startup = BackgroundStartup::new(startup_tx);
        let (stdin_blocked_tx, stdin_blocked_rx) = async_channel::bounded(1);
        let stdin_blocked_notice = if supports_stdin_blocked_notice(backend) {
            Some(stdin_blocked_tx)
        } else {
            None
        };

        let task_id_for_spawn = task_id.clone();
        let background_mode_for_spawn = background_mode.clone();
        let background_mode_for_completion = background_mode.clone();
        let background_startup_for_spawn = background_startup.clone();
        running_tasks.fetch_add(1, Ordering::AcqRel);
        self.executor
            .spawn(async move {
                let result = execute_script_standalone(
                    &working_dir,
                    &writable_paths,
                    &readable_paths,
                    executor,
                    registry,
                    permission_handler,
                    job_registry,
                    &task_id_for_spawn,
                    &execution_id,
                    background_mode_for_spawn,
                    &script,
                    mode,
                    backend,
                    ssh_target.as_deref(),
                    ssh_runtime.clone(),
                    container_runtime.clone(),
                    stdin_blocked_notice,
                    background_startup_for_spawn,
                    expect,
                    resolution,
                    &store_dir_for_spawn,
                    max_lines,
                    raw,
                )
                .await;

                let quick_result = match &result {
                    Ok(ok) => Ok(ok.clone()),
                    Err(err) => Err(err.to_string()),
                };
                let _ = result_tx.send(quick_result).await;
                if background_mode_for_completion.load(Ordering::Acquire) {
                    let _ = completed_tx
                        .send(CompletedTask {
                            task_id: task_id_for_spawn,
                            script,
                            result,
                        })
                        .await;
                }
                running_tasks.fetch_sub(1, Ordering::AcqRel);
            })
            .detach();

        if timeout == 0 {
            match startup_rx.recv().await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(anyhow::anyhow!(err)),
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "background startup channel dropped before registration"
                    ));
                }
            }
            let stdout = start_background_output_redirect(
                &self.job_registry,
                &store_dir,
                &task_id,
                max_lines,
                None,
            )
            .await?;
            let running = TerminalResult {
                stdout,
                stderr: None,
                exit_code: 0,
                task_id: Some(task_id),
                status: Some("running".to_string()),
            };
            return ToolResult::json(&running);
        }

        let timeout = std::time::Duration::from_secs(timeout);
        let immediate = futures_lite::future::or(
            async {
                result_rx
                    .recv()
                    .await
                    .ok()
                    .map(ForegroundDecision::Completed)
            },
            async {
                futures_lite::future::or(
                    async {
                        stdin_blocked_rx
                            .recv()
                            .await
                            .ok()
                            .map(|reason| ForegroundDecision::PromoteToBackground(Some(reason)))
                    },
                    async {
                        async_io::Timer::after(timeout).await;
                        Some(ForegroundDecision::PromoteToBackground(None))
                    },
                )
                .await
            },
        )
        .await;

        match immediate {
            Some(ForegroundDecision::Completed(Ok(mut result))) => {
                result.task_id = None;
                result.status = None;

                let failed = result.exit_code != 0;
                let json = serde_json::to_string(&result).map_err(|e| anyhow::anyhow!(e))?;

                if failed {
                    return Err(anyhow::anyhow!(format!("terminal command failed: {json}")));
                }

                ToolResult::json(&result)
            }
            Some(ForegroundDecision::Completed(Err(err))) => Err(anyhow::anyhow!(err)),
            Some(ForegroundDecision::PromoteToBackground(reason)) => {
                background_mode.store(true, Ordering::Release);
                match startup_rx.recv().await {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(anyhow::anyhow!(err)),
                    Err(_) => {
                        return Err(anyhow::anyhow!(
                            "background startup channel dropped before registration"
                        ));
                    }
                }
                let stdout = start_background_output_redirect(
                    &self.job_registry,
                    &store_dir,
                    &task_id,
                    max_lines,
                    reason.as_deref(),
                )
                .await?;
                let running = TerminalResult {
                    stdout,
                    stderr: None,
                    exit_code: 0,
                    task_id: Some(task_id),
                    status: Some("running".to_string()),
                };
                ToolResult::json(&running)
            }
            None => {
                background_mode.store(true, Ordering::Release);
                match startup_rx.recv().await {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(anyhow::anyhow!(err)),
                    Err(_) => {
                        return Err(anyhow::anyhow!(
                            "background startup channel dropped before registration"
                        ));
                    }
                }
                let stdout = start_background_output_redirect(
                    &self.job_registry,
                    &store_dir,
                    &task_id,
                    max_lines,
                    None,
                )
                .await?;
                let running = TerminalResult {
                    stdout,
                    stderr: None,
                    exit_code: 0,
                    task_id: Some(task_id),
                    status: Some("running".to_string()),
                };
                ToolResult::json(&running)
            }
        }
    }
}

async fn start_background_output_redirect(
    job_registry: &JobRegistry,
    store_dir: &PathBuf,
    task_id: &str,
    max_lines: usize,
    promotion_reason: Option<&str>,
) -> Result<OutputEntry, anyhow::Error> {
    let url = save_raw_to_file(store_dir, &[]).await?;
    let output_path = store_dir.join(url.strip_prefix("outputs/").unwrap_or(&url));
    let output_path_string = output_path.display().to_string();
    let snapshot = job_registry
        .start_output_redirect(task_id, output_path)
        .await
        .map_err(anyhow::Error::msg)?;
    let (preview, truncated) = preview_first_lines(&snapshot, max_lines);
    let text = match (promotion_reason, preview.is_empty()) {
        (Some(reason), true) => reason.to_string(),
        (Some(reason), false) => format!("{reason}\n{preview}"),
        (None, true) => "(no output yet)".to_string(),
        (None, false) => preview,
    };
    Ok(OutputEntry::Stored {
        path: output_path_string,
        url,
        content: Some(Content::Text { text, truncated }),
    })
}

/// Standalone script execution that can be spawned in a background task.
async fn execute_script_standalone<P, E>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    permission_handler: Arc<P>,
    job_registry: JobRegistry,
    task_id: &str,
    execution_id: &str,
    background_mode: Arc<AtomicBool>,
    script: &str,
    mode: TerminalMode,
    backend: ShellBackend,
    ssh_target: Option<&str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_runtime: Option<ContainerShellRuntime>,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    background_startup: BackgroundStartup,
    expect: OutputFormat,
    resolution: MediaResolution,
    store_dir: &PathBuf,
    max_lines: usize,
    raw: bool,
) -> Result<TerminalResult, TerminalError>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    info!(
        script_len = script.len(),
        ?mode,
        "executing background terminal command"
    );
    debug!(script = %script, "script content");

    let ipc_commands = registry.registered_tool_names();

    let start_result = if matches!(backend, ShellBackend::Container) {
        execute_container_background(
            executor.clone(),
            registry.clone(),
            task_id,
            execution_id,
            script,
            mode,
            container_runtime.as_ref(),
            &ipc_commands,
            &job_registry,
            stdin_blocked_notice.clone(),
            &background_startup,
        )
        .await
    } else if matches!(backend, ShellBackend::Ssh) {
        execute_ssh_background(
            task_id,
            execution_id,
            script,
            mode,
            ssh_target,
            ssh_runtime,
            &job_registry,
            &background_startup,
        )
        .await
    } else {
        match mode {
            TerminalMode::Network => {
                execute_sandboxed_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor.clone(),
                    registry.clone(),
                    task_id,
                    execution_id,
                    script,
                    mode,
                    PermissionNetworkPolicy { permission_handler },
                    &job_registry,
                    stdin_blocked_notice.clone(),
                    &background_startup,
                )
                .await
            }
            TerminalMode::Sandboxed => {
                execute_sandboxed_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor.clone(),
                    registry.clone(),
                    task_id,
                    execution_id,
                    script,
                    mode,
                    DenyAll,
                    &job_registry,
                    stdin_blocked_notice.clone(),
                    &background_startup,
                )
                .await
            }
            TerminalMode::Unsafe => {
                execute_unsafe_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor,
                    registry,
                    task_id,
                    execution_id,
                    script,
                    mode,
                    &job_registry,
                    stdin_blocked_notice.clone(),
                    &background_startup,
                )
                .await
            }
        }
    };
    let (pid, output) = match start_result {
        Ok(started) => started,
        Err(error) => {
            background_startup.report_failure(&error).await;
            return Err(error);
        }
    };

    let background_output = background_mode.load(Ordering::Acquire);
    let byte_limit = Some(INLINE_OUTPUT_LIMIT);
    let stdout = if background_output {
        match crate::output::save_text_with_line_limit(
            store_dir,
            &output.stdout,
            expect,
            resolution,
            max_lines,
            byte_limit,
        )
        .await
        {
            Ok(entry) => entry,
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(TerminalError::Io(err));
            }
        }
    } else {
        let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
        let compressed = if !raw && is_text && !output.stdout.is_empty() {
            if let Ok(text) = std::str::from_utf8(&output.stdout) {
                crate::output_compress::compress_text(text)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref c) = compressed {
            if let Some(ref raw_text) = c.raw_for_file {
                if let Err(err) =
                    crate::output::save_raw_to_file(store_dir, raw_text.as_bytes()).await
                {
                    warn!(error = %err, "failed to save raw source code output");
                }
            }
        }

        let data_to_save = compressed
            .as_ref()
            .map_or(&output.stdout[..], |c| c.text.as_bytes());

        match crate::output::save_text_with_line_limit(
            store_dir,
            data_to_save,
            expect,
            resolution,
            max_lines,
            byte_limit,
        )
        .await
        {
            Ok(entry) => entry,
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(TerminalError::Io(err));
            }
        }
    };

    // Save stderr if non-empty
    let stderr = if output.stderr.is_empty() {
        None
    } else {
        match OutputStore::save_to_dir_with_limit(
            store_dir,
            &output.stderr,
            OutputFormat::Text,
            MediaResolution::Auto,
            byte_limit,
        )
        .await
        {
            Ok(entry) => Some(entry),
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(TerminalError::Io(err));
            }
        }
    };

    let exit_code = output.status.code().unwrap_or(-1);
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        debug!(
            exit_code,
            signal = output.status.signal(),
            success = output.status.success(),
            "background script completed"
        );
    }
    #[cfg(not(unix))]
    {
        debug!(
            exit_code,
            success = output.status.success(),
            "background script completed"
        );
    }

    let output_path = stdout.stored_path(store_dir);
    job_registry.complete(pid, exit_code, output_path).await;

    Ok(TerminalResult {
        stdout,
        stderr,
        exit_code,
        task_id: None,
        status: None,
    })
}

async fn execute_sandboxed_background<E, N>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: TerminalMode,
    policy: N,
    job_registry: &JobRegistry,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    background_startup: &BackgroundStartup,
) -> Result<(u32, std::process::Output), TerminalError>
where
    E: Executor + Clone + 'static,
    N: NetworkPolicy + 'static,
{
    let script = wrap_script_with_session_runtime_env(working_dir, script);
    let router = create_ipc_router(registry);
    let config = SandboxConfig::builder()
        .network(policy)
        .working_dir(working_dir)
        .writable_paths(writable_paths)
        .readable_paths(readable_paths)
        .security(SecurityConfig::interactive())
        .ipc(router)
        .build()
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let sandbox = Sandbox::with_config_and_executor(config, executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(&script)?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let child = sandbox
        .command(program)
        .args(args)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, &script, mode, None)
        .await;
    background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        executor,
        job_registry,
        pid,
        stdin_blocked_notice,
        "missing stdout pipe for sandbox process",
        "missing stderr pipe for sandbox process",
    )
    .await?;

    Ok((pid, output))
}

async fn execute_unsafe_background<E: Executor + Clone + 'static>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: TerminalMode,
    job_registry: &JobRegistry,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    background_startup: &BackgroundStartup,
) -> Result<(u32, std::process::Output), TerminalError> {
    let script = wrap_script_with_session_runtime_env(working_dir, script);
    let router = create_ipc_gateway_router(registry);
    let config = SandboxConfig::builder()
        .network(AllowAll)
        .working_dir(working_dir)
        .writable_paths(writable_paths)
        .readable_paths(readable_paths)
        .security(SecurityConfig::interactive())
        .ipc(router)
        .build()
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let sandbox = Sandbox::with_config_and_executor(config, executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(&script)?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let child = sandbox
        .command(program)
        .args(args)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, &script, mode, None)
        .await;
    background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        executor,
        job_registry,
        pid,
        stdin_blocked_notice,
        "missing stdout pipe for unsafe process",
        "missing stderr pipe for unsafe process",
    )
    .await?;

    Ok((pid, output))
}

async fn collect_local_process_output<E: Executor + Clone + 'static>(
    mut child: heel::Child,
    executor: E,
    job_registry: &JobRegistry,
    pid: u32,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    missing_stdout_message: &'static str,
    missing_stderr_message: &'static str,
) -> Result<std::process::Output, TerminalError> {
    if let Some(stdin) = child.take_stdin() {
        let input_tx = spawn_terminal_stdin_writer(stdin);
        job_registry.attach_terminal_input(pid, input_tx).await;
    }

    let stdout = child
        .take_stdout()
        .ok_or_else(|| TerminalError::Execution(missing_stdout_message.into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| TerminalError::Execution(missing_stderr_message.into()))?;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    spawn_terminal_reader(stdout, chunk_tx.clone(), false);
    spawn_terminal_reader(stderr, chunk_tx.clone(), true);
    drop(chunk_tx);
    executor
        .spawn(drain_terminal_chunks(job_registry.clone(), pid, chunk_rx))
        .detach();

    let status = match wait_for_local_process_exit(&mut child, pid, stdin_blocked_notice).await {
        Ok(status) => status,
        Err(error) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(error);
        }
    };
    wait_for_terminal_stream_close(job_registry, pid).await;
    let (stdout, stderr) = job_registry.terminal_output(pid).await.ok_or_else(|| {
        TerminalError::Execution(format!("missing terminal output for pid {pid}"))
    })?;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };

    Ok(output)
}

async fn wait_for_local_process_exit(
    child: &mut heel::Child,
    pid: u32,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
) -> Result<std::process::ExitStatus, TerminalError> {
    if stdin_blocked_notice.is_none() {
        return child
            .wait()
            .await
            .map_err(|error| TerminalError::Execution(error.to_string()));
    }

    let mut notice_sent = false;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| TerminalError::Execution(error.to_string()))?
        {
            return Ok(status);
        }

        if !notice_sent {
            match detect_stdin_blocked_for_local_pid(pid).await {
                Ok(true) => {
                    if let Some(notice_tx) = stdin_blocked_notice.as_ref() {
                        let _ = notice_tx.try_send(TERMINAL_STDIN_BLOCKED_NOTICE.to_string());
                    }
                    notice_sent = true;
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::debug!(error = %error, pid, "stdin watchdog probe failed");
                }
            }
        }

        async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
    }
}

async fn execute_container_background<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: TerminalMode,
    container_runtime: Option<&ContainerShellRuntime>,
    ipc_commands: &[String],
    job_registry: &JobRegistry,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    background_startup: &BackgroundStartup,
) -> Result<(u32, std::process::Output), TerminalError> {
    let container_runtime = container_runtime.ok_or_else(|| {
        TerminalError::Execution("missing container runtime for container backend".into())
    })?;
    let exec = container_runtime.exec();

    // Use lower 32 bits of a UUID as a synthetic PID for job tracking.
    let pid = uuid::Uuid::new_v4().as_u128() as u32;
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    let (input_tx, input_rx) = async_channel::unbounded::<Vec<u8>>();
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;
    job_registry.attach_terminal_input(pid, input_tx).await;
    background_startup.report_ready().await;
    job_registry.attach_kill_switch(pid, kill_tx).await;

    let ipc_bridge = if ipc_commands.is_empty() {
        None
    } else {
        Some(start_container_ipc_bridge(executor, registry)?)
    };
    let wrapped_script = wrap_container_script(
        script,
        ipc_commands,
        ipc_bridge.as_ref().map(|bridge| ContainerIpcEndpoint {
            host: container_runtime.ipc_host(),
            port: bridge.port(),
        }),
    )?;

    let execution = exec
        .exec_boxed(
            container_runtime.container_id(),
            &wrapped_script,
            "/workspace",
            kill_rx,
            input_rx,
            stdin_blocked_notice,
        )
        .await;

    if let Some(bridge) = ipc_bridge {
        bridge.stop().await;
    }

    match execution {
        Ok(crate::shell_session::ContainerExecOutcome::Completed(output)) => {
            if !output.stdout.is_empty() {
                job_registry.append_stdout(pid, output.stdout.clone()).await;
            }
            if !output.stderr.is_empty() {
                job_registry.append_stderr(pid, output.stderr.clone()).await;
            }
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Ok(crate::shell_session::ContainerExecOutcome::Killed) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution("container job killed".to_string()))
        }
        Err(err) => {
            job_registry.fail(pid, &err, None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err))
        }
    }
}

async fn execute_ssh_background(
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: TerminalMode,
    ssh_target: Option<&str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    job_registry: &JobRegistry,
    background_startup: &BackgroundStartup,
) -> Result<(u32, std::process::Output), TerminalError> {
    let target =
        ssh_target.ok_or_else(|| TerminalError::Execution("missing ssh target".to_string()))?;
    let runtime = ssh_runtime
        .ok_or_else(|| TerminalError::Execution("missing ssh runtime profile".to_string()))?;

    let remote_cmd = match (runtime, mode) {
        (SshRuntimeProfile::Heel { binary }, TerminalMode::Network) => {
            let (program, args) = shell_launch(script);
            let escaped_args = args
                .iter()
                .map(|arg| shell_escape(arg))
                .collect::<Vec<_>>()
                .join(" ");
            if escaped_args.is_empty() {
                format!(
                    "{} run --network allow -- {}",
                    shell_escape(&binary),
                    shell_escape(&program),
                )
            } else {
                format!(
                    "{} run --network allow -- {} {}",
                    shell_escape(&binary),
                    shell_escape(&program),
                    escaped_args,
                )
            }
        }
        (SshRuntimeProfile::Heel { .. }, TerminalMode::Unsafe) => {
            return Err(TerminalError::Execution(
                "unsafe mode is not supported for ssh backend".to_string(),
            ));
        }
        (SshRuntimeProfile::Heel { .. }, TerminalMode::Sandboxed) => {
            return Err(TerminalError::Execution(
                "sandboxed mode is not supported for ssh backend".to_string(),
            ));
        }
    };

    let child = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg(remote_cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;
    background_startup.report_ready().await;

    match child.output().await {
        Ok(output) => {
            if !output.stdout.is_empty() {
                job_registry.append_stdout(pid, output.stdout.clone()).await;
            }
            if !output.stderr.is_empty() {
                job_registry.append_stderr(pid, output.stderr.clone()).await;
            }
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err.to_string()))
        }
    }
}

fn shell_escape(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    let mut output = String::with_capacity(escaped.len() + 2);
    output.push('\'');
    output.push_str(&escaped);
    output.push('\'');
    output
}

fn wrap_script_with_session_runtime_env(working_dir: &std::path::Path, script: &str) -> String {
    let session_tmp = working_dir.join("tmp");
    let session_cache = working_dir.join(".cache");
    let bun_cache = session_cache.join("bun");
    let session_config = working_dir.join(".config");
    let playwright_cache = session_cache.join("ms-playwright");
    let python_path = working_dir.join("skills").join("python");

    SessionRuntimeWrapperTemplate {
        tmp_dir: shell_escape(&session_tmp.display().to_string()),
        cache_dir: shell_escape(&session_cache.display().to_string()),
        bun_cache_dir: shell_escape(&bun_cache.display().to_string()),
        config_dir: shell_escape(&session_config.display().to_string()),
        playwright_cache_dir: shell_escape(&playwright_cache.display().to_string()),
        python_path: shell_escape(&python_path.display().to_string()),
        home_dir: shell_escape(&working_dir.display().to_string()),
        script,
    }
    .render()
    .unwrap_or_else(|error| panic!("failed to render session runtime wrapper template: {error}"))
}

#[derive(Template)]
#[template(path = "session_runtime_wrapper.sh", escape = "none")]
struct SessionRuntimeWrapperTemplate<'a> {
    tmp_dir: String,
    cache_dir: String,
    bun_cache_dir: String,
    config_dir: String,
    playwright_cache_dir: String,
    python_path: String,
    home_dir: String,
    script: &'a str,
}

#[derive(Template)]
#[template(path = "container_ipc_wrapper.sh", escape = "none")]
struct ContainerIpcWrapperTemplate<'a> {
    escaped_commands: &'a str,
    ipc_host: &'a str,
    ipc_port: u16,
    script: &'a str,
}

struct ContainerIpcEndpoint<'a> {
    host: &'a str,
    port: u16,
}

fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
) -> Result<String, TerminalError> {
    if ipc_commands.is_empty() {
        return Ok(script.to_string());
    }

    let endpoint = ipc_endpoint.ok_or_else(|| {
        TerminalError::Execution(
            "missing container IPC endpoint for wrapped script execution".to_string(),
        )
    })?;

    for name in ipc_commands {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            let mut message = String::from("invalid ipc command name for container wrapper: ");
            message.push_str(name);
            return Err(TerminalError::Execution(message));
        }
    }

    let escaped_commands = ipc_commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");
    ContainerIpcWrapperTemplate {
        escaped_commands: &escaped_commands,
        ipc_host: endpoint.host,
        ipc_port: endpoint.port,
        script,
    }
    .render()
    .map_err(|error| {
        let error_text = error.to_string();
        let mut message = String::from("failed to render container IPC wrapper: ");
        message.push_str(error_text.as_str());
        TerminalError::Execution(message)
    })
}

struct ContainerIpcBridge {
    shutdown_tx: Sender<()>,
    port: u16,
}

impl ContainerIpcBridge {
    fn port(&self) -> u16 {
        self.port
    }

    async fn stop(self) {
        tracing::debug!("stopping container IPC bridge");
        let _ = self.shutdown_tx.send(()).await;
    }
}

#[cfg(unix)]
fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
) -> Result<ContainerIpcBridge, TerminalError> {
    let listener = StdTcpListener::bind("127.0.0.1:0").map_err(|e| {
        TerminalError::Execution(format!("failed to bind container IPC tcp port: {e}"))
    })?;
    let local_addr = listener.local_addr().map_err(|e| {
        TerminalError::Execution(format!("failed to resolve container IPC tcp endpoint: {e}"))
    })?;
    tracing::debug!(
        bind = %local_addr,
        "starting container IPC bridge"
    );
    listener.set_nonblocking(true).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to set IPC tcp listener nonblocking mode: {e}"
        ))
    })?;
    let listener = Async::new(listener).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to register IPC tcp listener with async reactor: {e}"
        ))
    })?;

    let (shutdown_tx, shutdown_rx) = async_channel::bounded::<()>(1);
    let bridge_executor = executor.clone();
    executor
        .spawn(async move {
            tracing::debug!(bind = %local_addr, "container IPC bridge listening");
            if let Err(error) =
                run_container_ipc_bridge(listener, registry, shutdown_rx, bridge_executor).await
            {
                tracing::debug!(error = %error, "container IPC bridge stopped with error");
            }
            tracing::debug!(bind = %local_addr, "container IPC bridge stopped");
        })
        .detach();

    Ok(ContainerIpcBridge {
        shutdown_tx,
        port: local_addr.port(),
    })
}

#[cfg(not(unix))]
fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    _executor: E,
    _registry: Arc<ToolRegistry>,
) -> Result<ContainerIpcBridge, TerminalError> {
    Err(TerminalError::Execution(
        "container IPC bridge is only supported on unix hosts".to_string(),
    ))
}

#[cfg(unix)]
enum ContainerIpcBridgeEvent {
    Accept(std::io::Result<(Async<StdTcpStream>, std::net::SocketAddr)>),
    Shutdown,
}

#[cfg(unix)]
async fn run_container_ipc_bridge<E: Executor + Clone + 'static>(
    listener: Async<StdTcpListener>,
    registry: Arc<ToolRegistry>,
    shutdown_rx: Receiver<()>,
    executor: E,
) -> Result<(), String> {
    loop {
        let event = futures_lite::future::or(
            async { ContainerIpcBridgeEvent::Accept(listener.accept().await) },
            async {
                let _ = shutdown_rx.recv().await;
                ContainerIpcBridgeEvent::Shutdown
            },
        )
        .await;

        match event {
            ContainerIpcBridgeEvent::Shutdown => break,
            ContainerIpcBridgeEvent::Accept(Ok((stream, _addr))) => {
                tracing::debug!("container IPC bridge accepted connection");
                let registry = registry.clone();
                executor
                    .spawn(async move {
                        if let Err(error) = handle_container_ipc_connection(stream, registry).await
                        {
                            tracing::debug!(error = %error, "container IPC connection failed");
                        }
                    })
                    .detach();
            }
            ContainerIpcBridgeEvent::Accept(Err(error))
                if error.kind() == std::io::ErrorKind::Interrupted =>
            {
                continue;
            }
            ContainerIpcBridgeEvent::Accept(Err(error)) => {
                return Err(format!("container IPC accept failed: {error}"));
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
async fn handle_container_ipc_connection(
    mut stream: Async<StdTcpStream>,
    registry: Arc<ToolRegistry>,
) -> Result<(), String> {
    loop {
        let mut length_bytes = [0_u8; 4];
        match stream.read_exact(&mut length_bytes).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(format!("failed to read IPC request length: {error}")),
        }
        let body_length = u32::from_be_bytes(length_bytes) as usize;
        if body_length == 0 || body_length > 16 * 1024 * 1024 {
            return Err(format!("invalid IPC request length: {body_length}"));
        }

        let mut body = vec![0_u8; body_length];
        stream
            .read_exact(&mut body)
            .await
            .map_err(|e| format!("failed to read IPC request body: {e}"))?;

        if body.is_empty() {
            write_container_ipc_error(&mut stream, "empty IPC request body").await?;
            continue;
        }

        let method_length = body[0] as usize;
        if method_length == 0 {
            write_container_ipc_error(&mut stream, "empty IPC method name").await?;
            continue;
        }
        if body.len() < 1 + method_length {
            write_container_ipc_error(&mut stream, "invalid IPC request framing").await?;
            continue;
        }

        let method = std::str::from_utf8(&body[1..1 + method_length])
            .map_err(|e| format!("invalid IPC method name utf8: {e}"))?;
        let params = &body[1 + method_length..];
        let cli_args = decode_container_ipc_args(params)?;
        let response = registry.query_tool_handler(method, &cli_args).await;
        write_container_ipc_success(&mut stream, &response).await?;
    }
}

#[cfg(unix)]
fn decode_container_ipc_args(params: &[u8]) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        heel::rmp_serde::from_slice(params).map_err(|e| format!("invalid IPC params: {e}"))?;
    let args = parsed
        .as_object()
        .ok_or_else(|| "container IPC params must be a map".to_string())?;
    let args = args
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    Ok(crate::command::flatten_args_to_cli(&args))
}

#[cfg(unix)]
async fn write_container_ipc_success(
    stream: &mut Async<StdTcpStream>,
    response: &crate::command::CommandEnvelope,
) -> Result<(), String> {
    let payload = heel::rmp_serde::to_vec(response)
        .map_err(|e| format!("failed to encode IPC success payload: {e}"))?;
    write_container_ipc_response(stream, true, &payload).await
}

#[cfg(unix)]
async fn write_container_ipc_error(
    stream: &mut Async<StdTcpStream>,
    message: &str,
) -> Result<(), String> {
    let payload = heel::rmp_serde::to_vec(&message.to_string())
        .map_err(|e| format!("failed to encode IPC error payload: {e}"))?;
    write_container_ipc_response(stream, false, &payload).await
}

#[cfg(unix)]
async fn write_container_ipc_response(
    stream: &mut Async<StdTcpStream>,
    success: bool,
    payload: &[u8],
) -> Result<(), String> {
    let body_length = 1usize
        .checked_add(payload.len())
        .ok_or_else(|| "IPC response payload length overflow".to_string())?;
    let response_length = u32::try_from(body_length)
        .map_err(|_| format!("IPC response body too large: {body_length} bytes"))?;

    stream
        .write_all(&response_length.to_be_bytes())
        .await
        .map_err(|e| format!("failed to write IPC response length: {e}"))?;
    stream
        .write_all(&[if success { 1 } else { 0 }])
        .await
        .map_err(|e| format!("failed to write IPC success flag: {e}"))?;
    stream
        .write_all(payload)
        .await
        .map_err(|e| format!("failed to write IPC payload: {e}"))?;
    stream
        .flush()
        .await
        .map_err(|e| format!("failed to flush IPC payload: {e}"))
}

fn preview_first_lines(data: &[u8], max_lines: usize) -> (String, bool) {
    if data.is_empty() || max_lines == 0 {
        return (String::new(), !data.is_empty() && max_lines == 0);
    }

    let text = String::from_utf8_lossy(data);
    let mut preview = String::new();
    let mut total_lines = 0usize;

    for line in text.lines() {
        total_lines += 1;
        if total_lines <= max_lines {
            if !preview.is_empty() {
                preview.push('\n');
            }
            preview.push_str(line);
        }
    }

    let truncated = total_lines > max_lines;
    (preview, truncated)
}

enum TerminalChunk {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    StdoutClosed,
    StderrClosed,
}

fn spawn_terminal_reader<R>(
    mut reader: R,
    tx: async_channel::Sender<TerminalChunk>,
    is_stderr: bool,
) where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buffer = vec![0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::StderrClosed
                    } else {
                        TerminalChunk::StdoutClosed
                    });
                    break;
                }
                Ok(count) => {
                    let chunk = buffer[..count].to_vec();
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::Stderr(chunk)
                    } else {
                        TerminalChunk::Stdout(chunk)
                    });
                }
                Err(error) => {
                    warn!(error = %error, is_stderr, "terminal reader failed");
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::StderrClosed
                    } else {
                        TerminalChunk::StdoutClosed
                    });
                    break;
                }
            }
        }
    });
}

fn spawn_terminal_stdin_writer<W>(mut writer: W) -> async_channel::Sender<Vec<u8>>
where
    W: Write + Send + 'static,
{
    let (tx, rx) = async_channel::unbounded::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = rx.recv_blocking() {
            if bytes.is_empty() {
                continue;
            }
            if let Err(error) = writer.write_all(&bytes) {
                warn!(error = %error, "terminal stdin write failed");
                break;
            }
            if let Err(error) = writer.flush() {
                warn!(error = %error, "terminal stdin flush failed");
                break;
            }
        }
    });
    tx
}

async fn drain_terminal_chunks(
    job_registry: JobRegistry,
    pid: u32,
    rx: async_channel::Receiver<TerminalChunk>,
) {
    while let Ok(chunk) = rx.recv().await {
        match chunk {
            TerminalChunk::Stdout(bytes) => job_registry.append_stdout(pid, bytes).await,
            TerminalChunk::Stderr(bytes) => job_registry.append_stderr(pid, bytes).await,
            TerminalChunk::StdoutClosed => job_registry.close_stdout(pid).await,
            TerminalChunk::StderrClosed => job_registry.close_stderr(pid).await,
        }
    }
}

async fn wait_for_terminal_stream_close(job_registry: &JobRegistry, pid: u32) {
    while !job_registry.terminal_streams_closed(pid).await {
        async_io::Timer::after(Duration::from_millis(10)).await;
    }
}

/// Creates the IPC router with built-in and tool commands (standalone version).
fn create_ipc_router(registry: Arc<ToolRegistry>) -> IpcRouter {
    let mut router = builtin_router();

    // Register all configured tools as IPC commands
    let tool_names = registry.registered_tool_names();
    tracing::info!(tools = ?tool_names, "Creating IPC router with registered tools");
    for name in tool_names {
        router = crate::register_tool_command(router, registry.clone(), &name);
    }

    router
}

fn create_ipc_gateway_router(registry: Arc<ToolRegistry>) -> IpcRouter {
    let mut router = crate::register_ipc_gateway_command(IpcRouter::new(), registry.clone());

    // In unsafe mode, keep tool commands usable (websearch/webfetch/ask/task/todo...),
    // but never override native shell task/process commands like kill/jobs.
    let blocked = ["kill", "jobs"];
    let tool_names = registry.registered_tool_names();
    for name in tool_names {
        if blocked.contains(&name.as_str()) {
            continue;
        }
        router = crate::register_tool_command(router, registry.clone(), &name);
    }

    router
}

/// Errors that can occur during terminal execution.
#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    /// Permission denied for the requested mode.
    #[error(
        "sandbox blocked {mode} execution; approval is required to escalate this terminal command",
        mode = .0.description()
    )]
    PermissionDenied(TerminalMode),

    /// Permission check failed.
    #[error("permission error: {0}")]
    Permission(#[from] PermissionError),

    /// Sandbox setup failed.
    #[error("sandbox setup failed: {0}")]
    SandboxSetup(String),

    /// Script execution failed.
    #[error("execution failed: {0}")]
    Execution(String),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl<P, E, State> std::fmt::Debug for TerminalTool<P, E, State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalTool")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use heel::ConnectionDirection;

    use super::*;
    use crate::ToolRegistryBuilder;
    use crate::builtin::{InputTerminalArgs, InputTerminalTool};

    fn parse_terminal_tool_result(result: &ToolResult) -> TerminalResult {
        serde_json::from_str(
            &result
                .render_for_model()
                .expect("terminal result should render"),
        )
        .expect("terminal tool output should decode")
    }

    fn output_text(output: &OutputEntry) -> Option<&str> {
        match output {
            OutputEntry::Inline { content } | OutputEntry::Loaded { content, .. } => {
                match content {
                    Content::Text { text, .. } => Some(text.as_str()),
                    Content::Image { .. } => None,
                }
            }
            OutputEntry::Stored { content, .. } => match content.as_ref() {
                Some(Content::Text { text, .. }) => Some(text.as_str()),
                Some(Content::Image { .. }) | None => None,
            },
            OutputEntry::Empty => None,
        }
    }

    async fn wait_for_completed_task(
        receiver: &BackgroundTaskReceiver,
        task_id: &str,
    ) -> CompletedTask {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(task) = receiver
                    .take_completed()
                    .into_iter()
                    .find(|task| task.task_id == task_id)
                {
                    return task;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("background task should complete after terminal input")
    }

    #[derive(Default)]
    struct TestPermissionHandler {
        mode_checks: AtomicUsize,
        domain_checks: AtomicUsize,
        allow_network: bool,
        allow_domain: bool,
    }

    impl PermissionHandler for TestPermissionHandler {
        async fn check(&self, mode: TerminalMode, _script: &str) -> Result<bool, PermissionError> {
            self.mode_checks.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(match mode {
                TerminalMode::Sandboxed => true,
                TerminalMode::Network => self.allow_network,
                TerminalMode::Unsafe => false,
            })
        }

        async fn check_domain(&self, _domain: &str, _port: u16) -> bool {
            self.domain_checks.fetch_add(1, AtomicOrdering::Relaxed);
            self.allow_domain
        }
    }

    #[tokio::test]
    async fn test_terminal_args_defaults() {
        let args: TerminalArgs = serde_json::from_str(
            r#"{"description":"print a greeting","script":"echo hello","timeout":1}"#,
        )
        .unwrap();
        assert_eq!(args.expect, OutputFormat::Text);
        assert_eq!(args.resolution, MediaResolution::Auto);
        assert_eq!(args.timeout, 1);
        assert_eq!(args.mode, TerminalExecutionMode::Default);
    }

    #[tokio::test]
    async fn test_terminal_result_serialization() {
        use crate::output::Content;

        let result = TerminalResult {
            stdout: OutputEntry::Stored {
                url: "outputs/bold-oak-calm-river.txt".to_string(),
                path: "/tmp/outputs/bold-oak-calm-river.txt".to_string(),
                content: Some(Content::Text {
                    text: "preview text".to_string(),
                    truncated: true,
                }),
            },
            stderr: None,
            exit_code: 0,
            task_id: None,
            status: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("bold-oak-calm-river.txt"));
        assert!(!json.contains("stderr")); // should be skipped when None
        assert!(!json.contains("task_id")); // should be skipped when None

        // Test inline output (no URL)
        let result_inline = TerminalResult {
            stdout: OutputEntry::Inline {
                content: Content::Text {
                    text: "done".to_string(),
                    truncated: false,
                },
            },
            stderr: None,
            exit_code: 0,
            task_id: None,
            status: None,
        };
        let json_inline = serde_json::to_string(&result_inline).unwrap();
        assert!(!json_inline.contains("\"url\""));
        assert!(json_inline.contains("\"content\""));

        // Test background task result
        let result_background = TerminalResult {
            stdout: OutputEntry::Empty,
            stderr: None,
            exit_code: 0,
            task_id: Some("amber-forest-thunder-pearl".to_string()),
            status: Some("running".to_string()),
        };
        let json_bg = serde_json::to_string(&result_background).unwrap();
        assert!(json_bg.contains("\"task_id\":\"amber-forest-thunder-pearl\""));
        assert!(json_bg.contains("\"status\":\"running\""));
    }

    #[tokio::test]
    async fn test_terminal_args_timeout() {
        let args: TerminalArgs = serde_json::from_str(
            r#"{"description":"print a greeting","script":"echo hello","timeout":0}"#,
        )
        .unwrap();
        assert_eq!(args.timeout, 0);

        let args_default = serde_json::from_str::<TerminalArgs>(
            r#"{"description":"print a greeting","script":"echo hello"}"#,
        )
        .unwrap_err();
        assert!(args_default.to_string().contains("timeout"));

        let missing_description =
            serde_json::from_str::<TerminalArgs>(r#"{"script":"echo hello","timeout":1}"#)
                .unwrap_err();
        assert!(missing_description.to_string().contains("description"));
    }

    #[test]
    fn wrap_container_script_refreshes_command_hash_table() {
        let wrapped = wrap_container_script(
            "websearch \"gold price\"",
            &[String::from("websearch")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("MAY_IPC_DIR"));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent --subagent .skills/slide/subagents/art_direction.md --prompt 'x'";
        let wrapped = wrap_container_script(
            script,
            &[String::from("subagent")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains(script));
        assert!(!wrapped.contains("set -euo pipefail"));
    }

    #[test]
    fn wrap_container_script_rejects_invalid_ipc_command_names() {
        let error = wrap_container_script(
            "echo ok",
            &[String::from("bad name")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect_err("invalid command name must fail");
        assert!(
            error
                .to_string()
                .contains("invalid ipc command name for container wrapper")
        );
    }

    #[test]
    fn wrap_container_script_renders_expected_endpoint() {
        let wrapped = wrap_container_script(
            "environment --query deployment",
            &[String::from("environment"), String::from("session")],
            Some(ContainerIpcEndpoint {
                host: "host.containers.internal",
                port: 43123,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains("tcp://host.containers.internal:43123"));
        assert!(wrapped.contains("for cmd in 'environment' 'session'; do"));
    }

    #[tokio::test]
    async fn ensure_mode_allowed_requires_network_approval() {
        let handler = TestPermissionHandler {
            allow_network: false,
            allow_domain: true,
            ..Default::default()
        };
        let err = ensure_mode_allowed(&handler, TerminalMode::Network, "curl https://example.com")
            .await
            .expect_err("network mode should be denied");
        assert!(matches!(
            err,
            TerminalError::PermissionDenied(TerminalMode::Network)
        ));
        assert_eq!(handler.mode_checks.load(AtomicOrdering::Relaxed), 1);
    }

    #[tokio::test]
    async fn permission_network_policy_delegates_domain_checks() {
        let handler = Arc::new(TestPermissionHandler {
            allow_network: true,
            allow_domain: true,
            ..Default::default()
        });
        let policy = PermissionNetworkPolicy {
            permission_handler: handler.clone(),
        };
        let request = DomainRequest::new(
            "example.com".to_string(),
            443,
            ConnectionDirection::Outbound,
            1234,
        );
        assert!(policy.check(&request).await);
        assert_eq!(handler.domain_checks.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn sandboxed_execution_mode_deserializes() {
        let mode: TerminalExecutionMode =
            serde_json::from_str("\"sandboxed\"").expect("sandboxed mode must deserialize");
        assert_eq!(mode, TerminalExecutionMode::Sandboxed);
    }

    #[tokio::test]
    async fn sandboxed_command_output_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(dir.path())
            .security(SecurityConfig::interactive())
            .build()
            .expect("sandbox config should build");
        let sandbox = Sandbox::with_config_and_executor(config, executor_core::tokio::TokioGlobal)
            .await
            .expect("sandbox should initialize");

        let output = sandbox
            .command("/bin/pwd")
            .output()
            .await
            .expect("sandbox command output should succeed");
        eprintln!("PWD OUTPUT STATUS: {:?}", output.status);
        eprintln!(
            "PWD OUTPUT STDOUT: {:?}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(output.status.code(), Some(0));
    }

    #[tokio::test]
    async fn sandboxed_terminal_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize");
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(
            tool.working_dir(),
            &[],
            &[],
            executor_core::tokio::TokioGlobal,
            registry,
            Arc::new(TestPermissionHandler::default()),
            job_registry,
            "task-test",
            "exec-test",
            Arc::new(AtomicBool::new(false)),
            "/bin/pwd",
            TerminalMode::Sandboxed,
            ShellBackend::Local,
            None,
            None,
            None,
            None,
            BackgroundStartup::new(startup_tx),
            OutputFormat::Text,
            MediaResolution::Auto,
            &tool.outputs_dir(),
            50,
            false,
        )
        .await;
        match result {
            Ok(ok) => {
                eprintln!("PWD RESULT: {:?}", ok);
                assert_eq!(ok.exit_code, 0, "unexpected terminal result: {:?}", ok)
            }
            Err(error) => panic!("sandboxed pwd should succeed, got error: {error}"),
        }
    }

    #[tokio::test]
    async fn background_terminal_accepts_terminal_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        let result = tool
            .call(TerminalArgs {
                description: "wait for terminal input and echo it back".to_string(),
                script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\"".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 0,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should succeed");

        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.status.as_deref(), Some("running"));
        let task_id = payload
            .task_id
            .clone()
            .expect("timeout=0 should return a background task id");

        input_tool
            .call(InputTerminalArgs {
                task_id: task_id.clone(),
                input: "lexo".to_string(),
                append_newline: true,
            })
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background task should complete successfully after stdin input");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(
            stdout.contains("hello lexo"),
            "stdin input should reach the terminal process, got: {stdout}"
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn sandboxed_terminal_auto_promotes_stdin_wait_and_accepts_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        let result = tool
            .call(TerminalArgs {
                description: "wait for terminal input and echo it back".to_string(),
                script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\"".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 30,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should succeed");

        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.status.as_deref(), Some("running"));
        let task_id = payload
            .task_id
            .clone()
            .expect("stdin-blocked process should promote to background");
        let notice = output_text(&payload.stdout).expect("background preview should include text");
        assert!(
            notice.starts_with(TERMINAL_STDIN_BLOCKED_NOTICE),
            "unexpected promotion notice: {notice}"
        );

        input_tool
            .call(InputTerminalArgs {
                task_id: task_id.clone(),
                input: "lexo".to_string(),
                append_newline: true,
            })
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background task should complete successfully after stdin input");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(
            stdout.contains("hello lexo"),
            "stdin input should reach the terminal process, got: {stdout}"
        );
    }
}
