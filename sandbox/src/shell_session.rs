use std::{collections::HashSet, future::Future, pin::Pin, sync::Arc};

use async_channel::{Receiver, Sender};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Outcome of a container execution request.
#[derive(Debug)]
pub enum ContainerExecOutcome {
    Completed(std::process::Output),
    Killed,
}

/// Trait for executing commands inside a container.
///
/// Implementations provide the bridge between aither-sandbox and a specific
/// container runtime (e.g., Docker via bollard). The framework defines this trait;
/// the application (may) provides the implementation.
pub trait ContainerExec: Send + Sync {
    /// Execute a terminal command payload inside the container, returning stdout/stderr and exit code.
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

pub(crate) trait ContainerExecObject: Send + Sync {
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

#[derive(Clone)]
pub struct ContainerShellRuntime {
    exec: Arc<dyn ContainerExecObject>,
    container_id: String,
    ipc_host: String,
}

impl std::fmt::Debug for ContainerShellRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerShellRuntime")
            .field("container_id", &self.container_id)
            .field("ipc_host", &self.ipc_host)
            .finish_non_exhaustive()
    }
}

impl ContainerShellRuntime {
    #[must_use]
    pub fn new(
        container_id: impl Into<String>,
        ipc_host: impl Into<String>,
        exec: ContainerExecHandle,
    ) -> Self {
        let container_id = container_id.into();
        let ipc_host = ipc_host.into();
        assert!(
            !container_id.trim().is_empty(),
            "container runtime requires a non-empty container_id"
        );
        assert!(
            !ipc_host.trim().is_empty(),
            "container runtime requires a non-empty ipc_host"
        );
        Self {
            exec: Arc::clone(&exec.inner),
            container_id,
            ipc_host,
        }
    }

    #[must_use]
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    #[must_use]
    pub fn ipc_host(&self) -> &str {
        &self.ipc_host
    }

    pub(crate) fn exec(&self) -> Arc<dyn ContainerExecObject> {
        Arc::clone(&self.exec)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackend {
    Local,
    Container,
    Ssh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshServer {
    pub name: String,
    pub target: String,
}

impl SshServer {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub enum SshRuntimeProfile {
    Heel { binary: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ShellRuntimeAvailability {
    pub local: bool,
    pub container: bool,
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

pub trait SshSessionAuthorizer: Send + Sync {
    fn authorize_connect(
        &self,
        target: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;

    fn authorize_heel_install(
        &self,
        target: &str,
        details: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;
}

#[derive(Clone)]
pub struct ShellSessionRegistry {
    availability: ShellRuntimeAvailability,
    ssh_servers: Vec<SshServer>,
    ssh_authorizer: Option<Arc<dyn SshSessionAuthorizer>>,
    container_runtime: Option<ContainerShellRuntime>,
}

impl std::fmt::Debug for ShellSessionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSessionRegistry")
            .finish_non_exhaustive()
    }
}

impl ShellSessionRegistry {
    #[must_use]
    pub fn new(availability: ShellRuntimeAvailability) -> Self {
        Self {
            availability,
            ssh_servers: Vec::new(),
            ssh_authorizer: None,
            container_runtime: None,
        }
    }

    #[must_use]
    pub fn with_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.availability = availability;
        self
    }

    #[must_use]
    pub fn with_ssh_authorizer(mut self, authorizer: Arc<dyn SshSessionAuthorizer>) -> Self {
        self.ssh_authorizer = Some(authorizer);
        self
    }

    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.container_runtime = Some(runtime);
        self
    }

    #[must_use]
    pub fn availability(&self) -> ShellRuntimeAvailability {
        self.availability.clone()
    }

    pub fn with_ssh_servers(mut self, servers: Vec<SshServer>) -> Result<Self, String> {
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

        self.ssh_servers = deduped;
        Ok(self)
    }

    #[must_use]
    pub fn list_ssh_servers(&self) -> Vec<SshServer> {
        self.ssh_servers.clone()
    }

    pub(crate) fn container_runtime(&self) -> Option<&ContainerShellRuntime> {
        self.container_runtime.as_ref()
    }

    pub fn default_ssh_server(&self) -> Result<SshServer, String> {
        match self.ssh_servers.as_slice() {
            [] => Err("no ssh servers are configured".to_string()),
            [server] => Ok(server.clone()),
            _ => Err(
                "ssh_server_id is required because multiple ssh servers are configured".to_string(),
            ),
        }
    }

    pub fn resolve_ssh_server(&self, server_id: &str) -> Result<SshServer, String> {
        let wanted = server_id.trim();
        if wanted.is_empty() {
            return Err("ssh_server_id is required for ssh mode".to_string());
        }
        self.ssh_servers
            .iter()
            .find(|s| s.id() == wanted)
            .cloned()
            .ok_or_else(|| format!("unknown ssh_server_id: {wanted}"))
    }

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

    pub fn ensure_ssh_available(&self) -> Result<(), String> {
        if self.availability().ssh {
            Ok(())
        } else {
            Err("ssh backend is not available".to_string())
        }
    }

    pub fn ssh_authorizer(&self) -> Option<Arc<dyn SshSessionAuthorizer>> {
        self.ssh_authorizer.clone()
    }
}

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
    use super::{
        ShellRuntimeAvailability, ShellSessionRegistry, SshServer, parse_remote_probe_output,
    };

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

    #[test]
    fn default_ssh_server_uses_single_configured_target() {
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: false,
            ssh: true,
        })
        .with_ssh_servers(vec![SshServer {
            name: "prod".to_string(),
            target: "root@example.com".to_string(),
        }])
        .expect("ssh server should configure");

        let server = registry
            .default_ssh_server()
            .expect("single ssh server should become default");
        assert_eq!(server.id(), "prod");
        assert_eq!(server.target, "root@example.com");
    }

    #[test]
    fn default_ssh_server_requires_disambiguation_when_multiple_exist() {
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: false,
            ssh: true,
        })
        .with_ssh_servers(vec![
            SshServer {
                name: "prod".to_string(),
                target: "root@example.com".to_string(),
            },
            SshServer {
                name: "staging".to_string(),
                target: "root@staging.example.com".to_string(),
            },
        ])
        .expect("ssh servers should configure");

        let error = registry
            .default_ssh_server()
            .expect_err("multiple ssh servers must require explicit selection");
        assert!(
            error.contains("ssh_server_id is required"),
            "unexpected error: {error}"
        );
    }
}
