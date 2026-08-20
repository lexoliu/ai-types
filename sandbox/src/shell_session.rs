use std::{
    borrow::Cow,
    collections::HashSet,
    future::Future,
    pin::Pin,
    sync::{Arc, OnceLock, RwLock},
};

use aither_core::llm::{Tool, ToolOutput};
use async_channel::{Receiver, Sender};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Outcome of a container execution request.
#[derive(Debug)]
pub enum ContainerExecOutcome {
    /// The command ran to completion; carries its exit status and output.
    Completed(std::process::Output),
    /// The command was terminated before it finished.
    Killed,
}

/// Trait for executing commands inside a container.
///
/// Implementations provide the bridge between aither-sandbox and a specific
/// container runtime (e.g., Docker via bollard). The framework defines this trait;
/// the application (may) provides the implementation.
pub trait ContainerExec: Send + Sync {
    /// Execute a bash script inside the container, returning stdout/stderr and exit code.
    ///
    /// Implementations should send a human-readable message to
    /// `stdin_blocked_notice` when they detect the process is blocked on stdin
    /// (for example via `/proc/<pid>/syscall`), allowing callers to auto-promote
    /// foreground execution into background mode.
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;
}

pub trait ContainerExecObject: Send + Sync {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;
}

/// Backend-agnostic container executor handle.
///
/// This wraps any [`ContainerExec`] implementation into a cloneable concrete
/// type that can be passed around without exposing runtime-specific concrete
/// executor types.
#[derive(Clone)]
pub struct ContainerExecHandle {
    inner: Arc<dyn ContainerExecObject>,
}

impl std::fmt::Debug for ContainerExecHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerExecHandle")
            .finish_non_exhaustive()
    }
}

impl ContainerExecHandle {
    /// Creates a new handle from a concrete container executor implementation.
    #[must_use]
    pub fn new<T>(inner: Arc<T>) -> Self
    where
        T: ContainerExec + 'static,
    {
        Self { inner }
    }
}

impl ContainerExec for ContainerExecHandle {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let inner = Arc::clone(&self.inner);
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();

        async move {
            inner
                .exec_boxed(
                    &container_id,
                    &script,
                    &working_dir,
                    kill_rx,
                    stdin_blocked_notice,
                )
                .await
        }
    }
}

impl<T: ContainerExec> ContainerExecObject for T {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec(
            container_id,
            script,
            working_dir,
            kill_rx,
            stdin_blocked_notice,
        ))
    }
}

/// Where a shell command is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackend {
    /// On this machine, inside the local sandbox.
    Local,
    /// Inside a container.
    Container,
    /// On a remote host over SSH.
    Ssh,
}

/// A named SSH destination the agent may run commands on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshServer {
    /// Short name the agent uses to select this server.
    pub name: String,
    /// SSH target, as passed to `ssh` (for example `user@host`).
    pub target: String,
}

impl SshServer {
    /// The name used to address this server.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.name
    }
}

/// How the sandbox runtime is available on a remote host.
#[derive(Debug, Clone)]
pub enum SshRuntimeProfile {
    /// The `heel` sandbox binary is installed at the given path.
    Heel {
        /// Absolute path to the `heel` binary on the remote host.
        binary: String,
    },
}

/// Which shell backends are usable in the current deployment.
///
/// Reported to the model so it does not offer a backend that is not configured.
#[derive(Debug, Clone, Serialize)]
pub struct ShellRuntimeAvailability {
    /// Whether local execution is available. Always true in practice.
    pub local: bool,
    /// Whether a container runtime has been wired up.
    pub container: bool,
    /// Whether any SSH servers are registered.
    pub ssh: bool,
}

impl Default for ShellRuntimeAvailability {
    fn default() -> Self {
        Self {
            local: true,
            container: false,
            ssh: false,
        }
    }
}

/// Approves remote actions before the sandbox performs them.
///
/// Connecting to a host and installing software on it are both decisions a user
/// should make, so they are routed through this trait rather than assumed.
pub trait SshSessionAuthorizer: Send + Sync {
    /// Asks whether the agent may open an SSH connection to `target`.
    fn authorize_connect(
        &self,
        target: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;

    /// Asks whether the agent may install the `heel` runtime on `target`.
    ///
    /// `details` describes the remote host so the user can judge the request.
    fn authorize_heel_install(
        &self,
        target: &str,
        details: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;
}

/// Tracks which shell backends are configured and how to reach them.
///
/// Cloning shares the same underlying state, so registering a container or SSH
/// server is visible to every holder.
#[derive(Clone)]
pub struct ShellSessionRegistry {
    availability: Arc<RwLock<ShellRuntimeAvailability>>,
    ssh_servers: Arc<RwLock<Vec<SshServer>>>,
    ssh_authorizer: Arc<RwLock<Option<Arc<dyn SshSessionAuthorizer>>>>,
    /// Container executor — set once, shared across all clones via `OnceLock`.
    container_exec: Arc<OnceLock<Arc<dyn ContainerExecObject>>>,
    /// Default container ID — set once, used by container backend.
    container_id: Arc<OnceLock<String>>,
}

impl std::fmt::Debug for ShellSessionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSessionRegistry")
            .finish_non_exhaustive()
    }
}

impl ShellSessionRegistry {
    /// Creates a registry advertising the given backends.
    #[must_use]
    pub fn new(availability: ShellRuntimeAvailability) -> Self {
        Self {
            availability: Arc::new(RwLock::new(availability)),
            ssh_servers: Arc::new(RwLock::new(Vec::new())),
            ssh_authorizer: Arc::new(RwLock::new(None)),
            container_exec: Arc::new(OnceLock::new()),
            container_id: Arc::new(OnceLock::new()),
        }
    }

    /// Set the container executor (can only be called once; subsequent calls are no-ops).
    pub fn set_container_exec<T>(&self, exec: Arc<T>)
    where
        T: ContainerExec + 'static,
    {
        let exec: Arc<dyn ContainerExecObject> = exec;
        let _ = self.container_exec.set(exec);
    }

    /// Set the default container ID for container backend commands.
    pub fn set_container_id(&self, id: String) {
        let _ = self.container_id.set(id);
    }

    /// Replaces the set of advertised backends.
    ///
    /// # Errors
    ///
    /// Returns an error if the availability lock was poisoned by a panic in
    /// another thread.
    pub fn set_availability(&self, availability: ShellRuntimeAvailability) -> Result<(), String> {
        *self
            .availability
            .write()
            .map_err(|_| "shell availability lock poisoned".to_string())? = availability;
        Ok(())
    }

    /// Installs the authorizer consulted before any remote action.
    ///
    /// # Errors
    ///
    /// Returns an error if the authorizer lock was poisoned.
    pub fn set_ssh_authorizer(
        &self,
        authorizer: Arc<dyn SshSessionAuthorizer>,
    ) -> Result<(), String> {
        *self
            .ssh_authorizer
            .write()
            .map_err(|_| "ssh authorizer lock poisoned".to_string())? = Some(authorizer);
        Ok(())
    }

    /// The currently advertised backends.
    #[must_use]
    pub fn availability(&self) -> ShellRuntimeAvailability {
        self.availability
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Get the container executor (if set).
    pub(crate) fn container_exec(&self) -> Option<Arc<dyn ContainerExecObject>> {
        self.container_exec.get().cloned()
    }

    /// Get the configured container ID (if set).
    #[must_use]
    pub fn container_id(&self) -> Option<String> {
        self.container_id.get().cloned()
    }

    /// Replaces the registered SSH servers.
    ///
    /// # Errors
    ///
    /// Returns an error if any entry has an empty name or target, if two
    /// entries share a name, or if the server lock was poisoned.
    pub fn set_ssh_servers(&self, servers: Vec<SshServer>) -> Result<(), String> {
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for server in servers {
            let id = server.id().trim().to_string();
            let target = server.target.trim().to_string();
            if id.is_empty() || target.is_empty() {
                return Err("ssh server entries require non-empty name and target".to_string());
            }
            if !seen.insert(id.clone()) {
                return Err(format!("duplicate ssh server id: {id}"));
            }
            deduped.push(SshServer { name: id, target });
        }

        *self
            .ssh_servers
            .write()
            .map_err(|_| "ssh server lock poisoned".to_string())? = deduped;
        Ok(())
    }

    /// Every registered SSH server.
    #[must_use]
    pub fn list_ssh_servers(&self) -> Vec<SshServer> {
        self.ssh_servers
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Looks up a registered server by name.
    ///
    /// # Errors
    ///
    /// Returns an error if `server_id` is blank, names no registered server, or
    /// the server lock was poisoned.
    pub fn resolve_ssh_server(&self, server_id: &str) -> Result<SshServer, String> {
        let wanted = server_id.trim();
        if wanted.is_empty() {
            return Err("ssh_server_id is required for ssh mode".to_string());
        }
        self.ssh_servers
            .read()
            .map_err(|_| "ssh server lock poisoned".to_string())?
            .iter()
            .find(|s| s.id() == wanted)
            .cloned()
            .ok_or_else(|| format!("unknown ssh_server_id: {wanted}"))
    }

    /// Picks the backend to use for a command that did not name one.
    ///
    /// Prefers a container when one is configured, since it isolates better
    /// than running on the host.
    ///
    /// # Errors
    ///
    /// Returns an error if neither a container nor local execution is available.
    pub fn resolve_local_backend(&self) -> Result<ShellBackend, String> {
        let availability = self.availability();
        if availability.container {
            return Ok(ShellBackend::Container);
        }
        if availability.local {
            return Ok(ShellBackend::Local);
        }
        Err("no local backend available".to_string())
    }

    /// Confirms the SSH backend is configured.
    ///
    /// # Errors
    ///
    /// Returns an error if no SSH servers are registered.
    pub fn ensure_ssh_available(&self) -> Result<(), String> {
        if self.availability().ssh {
            Ok(())
        } else {
            Err("ssh backend is not available".to_string())
        }
    }

    /// The installed authorizer, if any.
    #[must_use]
    pub fn ssh_authorizer(&self) -> Option<Arc<dyn SshSessionAuthorizer>> {
        self.ssh_authorizer.read().ok()?.clone()
    }
}

/// Lists the SSH servers the agent may connect to.
#[derive(Debug, Clone)]
pub struct ListSshTool {
    registry: ShellSessionRegistry,
}

impl ListSshTool {
    /// Creates the tool over the given registry.
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry) -> Self {
        Self { registry }
    }
}

impl Tool for ListSshTool {
    fn name(&self) -> Cow<'static, str> {
        "list_ssh".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "List the SSH servers this agent is allowed to connect to. Returns each \
         server's id and target. Call this before open_ssh to discover valid ids."
            .into()
    }

    type Arguments = ();

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let payload = serde_json::json!({
            "servers": self.registry.list_ssh_servers(),
        });
        Ok(ToolOutput::text(payload.to_string()))
    }
}

/// Open an SSH session to a registered server.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OpenSshArgs {
    /// Name of a server returned by `list_ssh`.
    pub ssh_server_id: String,
}

/// Opens an SSH session to one of the registered servers.
#[derive(Debug, Clone)]
pub struct OpenSshTool {
    registry: ShellSessionRegistry,
}

impl OpenSshTool {
    /// Creates the tool over the given registry.
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry) -> Self {
        Self { registry }
    }
}

impl Tool for OpenSshTool {
    fn name(&self) -> Cow<'static, str> {
        "open_ssh".into()
    }

    type Arguments = OpenSshArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        self.registry
            .ensure_ssh_available()
            .map_err(anyhow::Error::msg)?;
        let server = self
            .registry
            .resolve_ssh_server(&args.ssh_server_id)
            .map_err(anyhow::Error::msg)?;
        let runtime =
            bootstrap_ssh_runtime(&server.target, &self.registry.ssh_authorizer()).await?;
        let payload = serde_json::json!({
            "ssh_server_id": server.id(),
            "target": server.target,
            "runtime": match runtime {
                SshRuntimeProfile::Heel { .. } => "heel",
            }
        });
        Ok(ToolOutput::text(payload.to_string()))
    }
}

/// Prepares a remote host to run sandboxed commands.
///
/// Confirms the connection is authorized, probes for the `heel` runtime, and —
/// with the user's approval — installs it if it is missing.
///
/// # Errors
///
/// Returns an error if the authorizer denies the connection, the SSH probe
/// fails, or `heel` is absent and either installation was declined or did not
/// succeed.
pub async fn bootstrap_ssh_runtime(
    target: &str,
    authorizer: &Option<Arc<dyn SshSessionAuthorizer>>,
) -> Result<SshRuntimeProfile, anyhow::Error> {
    if let Some(auth) = authorizer {
        let allowed = auth
            .authorize_connect(target)
            .await
            .map_err(anyhow::Error::msg)?;
        if !allowed {
            return Err(anyhow::anyhow!("user denied ssh connection authorization"));
        }
    }

    let remote = detect_remote(target).await?;
    if remote.heel_found {
        return Ok(SshRuntimeProfile::Heel {
            binary: remote.heel_path,
        });
    }

    if let Some(auth) = authorizer {
        let details = format!(
            "Remote {} ({}) does not have heel installed.",
            remote.os, remote.arch
        );
        let approve_install = auth
            .authorize_heel_install(target, &details)
            .await
            .map_err(anyhow::Error::msg)?;
        if approve_install && install_heel(target, &remote).await? {
            let verified = detect_remote(target).await?;
            if verified.heel_found {
                return Ok(SshRuntimeProfile::Heel {
                    binary: verified.heel_path,
                });
            }
        }
    }

    Err(anyhow::anyhow!(
        "remote heel runtime unavailable; ssh mode requires heel on the remote host"
    ))
}

struct RemoteInfo {
    os: String,
    arch: String,
    heel_found: bool,
    heel_path: String,
}

async fn detect_remote(target: &str) -> Result<RemoteInfo, anyhow::Error> {
    let probe = "uname -s; uname -m; if command -v heel >/dev/null 2>&1; then command -v heel; elif [ -x \"$HOME/.local/bin/heel\" ]; then printf '%s\\n' \"$HOME/.local/bin/heel\"; else echo __NO_HEEL__; fi";
    let output = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg(probe)
        .stdin(async_process::Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("ssh probe failed: {e}"))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "ssh probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    parse_remote_probe_output(&output.stdout)
}

fn parse_remote_probe_output(stdout: &[u8]) -> Result<RemoteInfo, anyhow::Error> {
    let lines = String::from_utf8_lossy(stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    if lines.len() < 3 {
        return Err(anyhow::anyhow!("ssh probe returned unexpected output"));
    }

    let heel_line = lines[2].clone();
    let heel_found = heel_line != "__NO_HEEL__";
    if !heel_found {
        return Ok(RemoteInfo {
            os: lines[0].clone(),
            arch: lines[1].clone(),
            heel_found,
            heel_path: String::new(),
        });
    }

    Ok(RemoteInfo {
        os: lines[0].clone(),
        arch: lines[1].clone(),
        heel_found,
        heel_path: heel_line,
    })
}

async fn install_heel(target: &str, remote: &RemoteInfo) -> Result<bool, anyhow::Error> {
    let local_os = std::env::consts::OS;
    let remote_os = remote.os.to_lowercase();
    let os_match = (local_os == "macos" && remote_os.contains("darwin"))
        || (local_os == "linux" && remote_os.contains("linux"));
    if !os_match {
        return Ok(false);
    }

    let local_arch = std::env::consts::ARCH;
    if normalize_arch(local_arch) != normalize_arch(&remote.arch) {
        return Ok(false);
    }

    let local_heel = find_local_heel().await?;
    if local_heel.is_empty() {
        return Ok(false);
    }

    let mkdir_status = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg("mkdir -p ~/.local/bin")
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("ssh mkdir failed: {e}"))?;
    if !mkdir_status.success() {
        return Ok(false);
    }

    let dest = format!("{target}:~/.local/bin/heel");
    let scp_status = async_process::Command::new("scp")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(&local_heel)
        .arg(&dest)
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("scp heel failed: {e}"))?;
    if !scp_status.success() {
        return Ok(false);
    }

    let verify_status = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg("chmod +x ~/.local/bin/heel && ~/.local/bin/heel --version")
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("verify heel failed: {e}"))?;

    Ok(verify_status.success())
}

async fn find_local_heel() -> Result<String, anyhow::Error> {
    let out = async_process::Command::new("sh")
        .arg("-c")
        .arg("command -v heel || true")
        .stdin(async_process::Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("failed to locate local heel: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn normalize_arch(raw: &str) -> &str {
    match raw {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "arm64",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_remote_probe_output;

    #[test]
    fn parse_remote_probe_detects_local_bin_heel_path() {
        let stdout = b"Linux\nx86_64\n/home/test/.local/bin/heel\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert_eq!(remote.os, "Linux");
        assert_eq!(remote.arch, "x86_64");
        assert!(remote.heel_found);
        assert_eq!(remote.heel_path, "/home/test/.local/bin/heel");
    }

    #[test]
    fn parse_remote_probe_handles_missing_heel() {
        let stdout = b"Linux\naarch64\n__NO_HEEL__\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert!(!remote.heel_found);
        assert!(remote.heel_path.is_empty());
    }
}
