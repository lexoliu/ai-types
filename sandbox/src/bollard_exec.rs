//! Container exec implementation backed by bollard.

use async_channel::{Receiver, Sender};
use bollard::Docker;
use bollard::container::LogOutput;
use bollard::exec::CreateExecOptions;
use futures_lite::StreamExt;
use std::future::Future;
use std::process::{ExitStatus, Output};
use std::sync::Arc;

use crate::shell_session::{ContainerExec, ContainerExecOutcome};
use crate::stdin_watch::{
    STDIN_WATCH_INTERVAL, TERMINAL_STDIN_BLOCKED_NOTICE, is_waiting_on_stdin,
};

/// Executes commands in a running container via Docker exec.
#[derive(Debug, Clone)]
pub struct BollardContainerExec {
    client: Arc<Docker>,
}

impl BollardContainerExec {
    /// Create a new exec adapter from a shared Docker client.
    #[must_use]
    pub const fn new(client: Arc<Docker>) -> Self {
        Self { client }
    }
}

async fn detect_stdin_blocked_inside_container(
    client: &Docker,
    container_id: &str,
    pid: i64,
) -> Result<bool, String> {
    let probe_script = format!("cat /proc/{pid}/syscall 2>/dev/null || true");
    let probe = client
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(vec!["sh", "-c", probe_script.as_str()]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to create syscall probe exec: {e}"))?;

    let started = client
        .start_exec(&probe.id, None)
        .await
        .map_err(|e| format!("failed to start syscall probe exec: {e}"))?;

    let mut stdout = String::new();
    match started {
        bollard::exec::StartExecResults::Attached { mut output, .. } => {
            while let Some(chunk) = output.next().await {
                match chunk {
                    Ok(LogOutput::StdOut { message } | LogOutput::Console { message }) => {
                        stdout.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(LogOutput::StdErr { .. } | LogOutput::StdIn { .. }) => {}
                    Err(e) => return Err(format!("syscall probe stream error: {e}")),
                }
            }
        }
        bollard::exec::StartExecResults::Detached => {
            return Err("syscall probe exec detached unexpectedly".to_string());
        }
    }

    Ok(is_waiting_on_stdin(&stdout))
}

async fn kill_exec_pid_inside_container(
    client: &Docker,
    container_id: &str,
    exec_id: &str,
) -> Result<(), String> {
    let inspect = client
        .inspect_exec(exec_id)
        .await
        .map_err(|e| format!("failed to inspect exec for kill: {e}"))?;
    if !inspect.running.unwrap_or(false) {
        return Ok(());
    }
    let Some(pid) = inspect.pid else {
        return Ok(());
    };
    if pid <= 0 {
        return Ok(());
    }

    let kill_script = format!("kill -KILL {pid} >/dev/null 2>&1 || true");
    let kill_exec = client
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(vec!["sh", "-c", kill_script.as_str()]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to create in-container kill exec: {e}"))?;
    match client
        .start_exec(&kill_exec.id, None)
        .await
        .map_err(|e| format!("failed to start in-container kill exec: {e}"))?
    {
        bollard::exec::StartExecResults::Attached { mut output, .. } => {
            while let Some(chunk) = output.next().await {
                if let Err(error) = chunk {
                    return Err(format!("in-container kill stream error: {error}"));
                }
            }
        }
        bollard::exec::StartExecResults::Detached => {
            return Err("in-container kill exec detached unexpectedly".to_string());
        }
    }

    Ok(())
}

enum StreamEvent {
    Output(Option<Result<LogOutput, bollard::errors::Error>>),
    Stdin(Result<Vec<u8>, async_channel::RecvError>),
    WatchdogTick,
    Cancel,
}

fn append_log_output(output: LogOutput, stdout: &mut Vec<u8>, stderr: &mut Vec<u8>) {
    match output {
        LogOutput::StdOut { message } | LogOutput::Console { message } => {
            stdout.extend_from_slice(&message);
        }
        LogOutput::StdErr { message } => {
            stderr.extend_from_slice(&message);
        }
        LogOutput::StdIn { .. } => {}
    }
}

async fn write_container_stdin<W>(input: &mut W, bytes: &[u8]) -> Result<(), String>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;

    input
        .write_all(bytes)
        .await
        .map_err(|error| format!("container exec stdin write failed: {error}"))?;
    input
        .flush()
        .await
        .map_err(|error| format!("container exec stdin flush failed: {error}"))
}

async fn shutdown_container_stdin<W>(input: &mut W) -> Result<(), String>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;

    input
        .shutdown()
        .await
        .map_err(|error| format!("container exec stdin shutdown failed: {error}"))
}

struct WatchdogTick<'a, W> {
    client: &'a Docker,
    container_id: &'a str,
    exec_id: &'a str,
    input: &'a mut W,
    stdin_open: &'a mut bool,
    watchdog_active: &'a mut bool,
    notice_sent: &'a mut bool,
    stdin_blocked_notice: Option<&'a Sender<String>>,
}

async fn handle_container_watchdog_tick<W>(tick: WatchdogTick<'_, W>) -> Result<(), String>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let inspect = tick
        .client
        .inspect_exec(tick.exec_id)
        .await
        .map_err(|e| format!("failed to inspect running exec for stdin watchdog: {e}"))?;

    if !inspect.running.unwrap_or(false) {
        if *tick.stdin_open {
            let _ = shutdown_container_stdin(tick.input).await;
            *tick.stdin_open = false;
        }
        *tick.watchdog_active = false;
        return Ok(());
    }

    let Some(pid) = inspect.pid else {
        return Ok(());
    };
    if pid <= 0 {
        return Ok(());
    }

    match detect_stdin_blocked_inside_container(tick.client, tick.container_id, pid).await {
        Ok(true) => {
            if let Some(notice_tx) = tick.stdin_blocked_notice {
                let _ = notice_tx.try_send(TERMINAL_STDIN_BLOCKED_NOTICE.to_string());
            }
            *tick.notice_sent = true;
            *tick.watchdog_active = false;
        }
        Ok(false) => {}
        Err(error) => {
            tracing::debug!(error = %error, pid, "stdin watchdog probe failed");
        }
    }
    Ok(())
}

async fn inspect_exec_exit_status(client: &Docker, exec_id: &str) -> Result<ExitStatus, String> {
    let inspect = client
        .inspect_exec(exec_id)
        .await
        .map_err(|e| format!("failed to inspect exec: {e}"))?;
    let exit_code = i32::try_from(inspect.exit_code.unwrap_or(-1))
        .map_err(|_| "container exec exit code is outside i32 range".to_string())?;
    Ok(ExitStatusExt::from_raw(exit_code))
}

impl ContainerExec for BollardContainerExec {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();
        let client = self.client.clone();

        async move {
            let config = CreateExecOptions {
                cmd: Some(vec!["sh", "-c", &script]),
                attach_stdin: Some(true),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                working_dir: Some(working_dir.as_str()),
                ..Default::default()
            };

            let exec_instance = client
                .create_exec(&container_id, config)
                .await
                .map_err(|e| format!("failed to create exec: {e}"))?;
            let exec_id = exec_instance.id.clone();

            let start = client
                .start_exec(&exec_id, None)
                .await
                .map_err(|e| format!("failed to start exec: {e}"))?;

            let (mut output, mut input) = match start {
                bollard::exec::StartExecResults::Attached { output, input } => (output, input),
                bollard::exec::StartExecResults::Detached => {
                    return Err("exec started in detached mode unexpectedly".to_string());
                }
            };

            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut watchdog_active = stdin_blocked_notice.is_some();
            let mut notice_sent = false;
            let mut stdin_open = true;

            loop {
                let event = tokio::select! {
                    output = async { output.next().await } => StreamEvent::Output(output),
                    () = async {
                        let _ = kill_rx.recv().await;
                    } => StreamEvent::Cancel,
                    bytes = async { stdin_rx.recv().await }, if stdin_open => StreamEvent::Stdin(bytes),
                    _ = async_io::Timer::after(STDIN_WATCH_INTERVAL), if watchdog_active && !notice_sent => StreamEvent::WatchdogTick,
                };

                match event {
                    StreamEvent::Output(Some(Ok(output))) => {
                        append_log_output(output, &mut stdout, &mut stderr);
                    }
                    StreamEvent::Output(Some(Err(e))) => {
                        return Err(format!("exec stream error: {e}"));
                    }
                    StreamEvent::Output(None) => {
                        break;
                    }
                    StreamEvent::Stdin(Ok(bytes)) => {
                        write_container_stdin(&mut input, &bytes).await?;
                    }
                    StreamEvent::Stdin(Err(_)) => {
                        shutdown_container_stdin(&mut input).await?;
                        stdin_open = false;
                    }
                    StreamEvent::Cancel => {
                        if stdin_open {
                            let _ = shutdown_container_stdin(&mut input).await;
                        }
                        if let Err(error) =
                            kill_exec_pid_inside_container(&client, &container_id, &exec_id).await
                        {
                            tracing::warn!(
                                error = %error,
                                exec_id = %exec_id,
                                "failed to kill container exec after cancellation request"
                            );
                        }
                        return Ok(ContainerExecOutcome::Killed);
                    }
                    StreamEvent::WatchdogTick => {
                        handle_container_watchdog_tick(WatchdogTick {
                            client: &client,
                            container_id: &container_id,
                            exec_id: &exec_id,
                            input: &mut input,
                            stdin_open: &mut stdin_open,
                            watchdog_active: &mut watchdog_active,
                            notice_sent: &mut notice_sent,
                            stdin_blocked_notice: stdin_blocked_notice.as_ref(),
                        })
                        .await?;
                    }
                }
            }

            let status = inspect_exec_exit_status(&client, &exec_id).await?;
            Ok(ContainerExecOutcome::Completed(Output {
                status,
                stdout,
                stderr,
            }))
        }
    }
}

#[cfg(unix)]
struct ExitStatusExt;

#[cfg(unix)]
impl ExitStatusExt {
    fn from_raw(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt as _;
        ExitStatus::from_raw(code << 8)
    }
}

#[cfg(not(unix))]
struct ExitStatusExt;

#[cfg(not(unix))]
impl ExitStatusExt {
    fn from_raw(code: i32) -> ExitStatus {
        use std::os::windows::process::ExitStatusExt as _;
        ExitStatus::from_raw(code as u32)
    }
}
