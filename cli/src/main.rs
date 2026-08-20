//! Interactive CLI for testing aither agents.
//!
//! A REPL-style interface for testing agents with `OpenAI`, Claude, or Gemini.
//!
//! # Bash-First Design
//!
//! The agent has a single `bash` tool. All other capabilities are exposed as
//! terminal commands that can be run within bash scripts:
//!
//! - `websearch "query"` - Search the web
//! - `webfetch "url"` - Fetch and read web pages
//! - `todo ...` - Track tasks
//!
//! Standard bash commands (ls, cat, grep, find, etc.) work normally.
//!
//! # Usage
//!
//! ```bash
//! # Auto-detect provider from available API keys
//! cargo run -p aither-cli
//!
//! # Explicit provider
//! OPENAI_API_KEY=xxx cargo run -p aither-cli -- --provider openai
//! ANTHROPIC_API_KEY=xxx cargo run -p aither-cli -- --provider claude
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --provider gemini
//!
//! # Headless mode (single query, useful for testing/scripting)
//! cargo run -p aither-cli -- --prompt "What is 2+2?"
//! cargo run -p aither-cli -- --prompt "List files" --quiet
//!
//! # With additional MCP servers
//! cargo run -p aither-cli -- --mcp servers.json
//!
//! # Disable Context7 (documentation lookup)
//! cargo run -p aither-cli -- --no-context7
//! ```

mod hook;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use aither_agent::sandbox::{
    ToolRegistryBuilder, cli_to_json,
    permission::{BashMode, PermissionError, PermissionHandler, StatefulPermissionHandler},
    schema_to_help,
};
use aither_agent::specialized::SubagentTool;
use aither_agent::{Agent, BashAgentBuilder, Hook};
use aither_core::LanguageModel;
use aither_core::llm::Role;
use aither_mcp::{McpConnection, McpServersConfig};
use anyhow::{Context, Result};
use async_lock::Mutex as AsyncMutex;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use executor_core::tokio::TokioGlobal;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

mod provider;

use crate::hook::DebugHook;
use crate::provider::Provider;
use aither_cloud::CloudProvider;

/// Default whitelist of common domains that don't need explicit approval.
const DEFAULT_DOMAIN_WHITELIST: &[&str] = &[
    // Package registries
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "pypi.org",
    "files.pythonhosted.org",
    "crates.io",
    "static.crates.io",
    "rubygems.org",
    // CDNs
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "esm.sh",
    // Git hosts
    "github.com",
    "raw.githubusercontent.com",
    "gitlab.com",
    "bitbucket.org",
    // Common APIs
    "api.github.com",
];

/// Interactive permission handler that prompts the user.
///
/// - Sandboxed: always allow (no prompt)
/// - Network: ask once, then auto-approve all domains
/// - Unsafe: always ask for each script
#[derive(Debug, Default)]
struct InteractivePermissionHandler {
    /// Whether network mode has been approved (auto-approves all domains).
    network_approved: std::sync::atomic::AtomicBool,
    /// Cache of approved domains for cases where network mode isn't blanket approved.
    approved_domains: std::sync::RwLock<std::collections::HashSet<String>>,
}

impl InteractivePermissionHandler {
    fn new() -> Self {
        Self {
            network_approved: std::sync::atomic::AtomicBool::new(false),
            approved_domains: std::sync::RwLock::new(std::collections::HashSet::new()),
        }
    }
}

impl PermissionHandler for InteractivePermissionHandler {
    async fn check(&self, mode: BashMode, script: &str) -> Result<bool, PermissionError> {
        // This will be wrapped in StatefulPermissionHandler for network mode
        match mode {
            BashMode::Sandboxed => Ok(true), // Always allow
            BashMode::Network | BashMode::Unsafe => {
                // Display script and ask for permission (show full script, no truncation)
                let mode_desc = mode.description();
                let display_script = script.trim().replace('\n', "; ");
                eprintln!("\n\x1b[33m⚠ Permission request: {mode_desc}\x1b[0m");
                eprintln!("\x1b[2mScript:\x1b[22m {display_script}");
                eprint!("\x1b[33mAllow? [y/N]: \x1b[0m");
                io::stderr().flush().ok();

                // Read single char in raw mode
                let approved = read_yes_no().unwrap_or(false);
                eprintln!();

                if approved {
                    // Mark network as approved to skip domain prompts
                    if mode == BashMode::Network {
                        self.network_approved
                            .store(true, std::sync::atomic::Ordering::Release);
                    }
                    Ok(true)
                } else {
                    Err(PermissionError::Denied("user declined".to_string()))
                }
            }
        }
    }

    async fn check_domain(&self, domain: &str, _port: u16) -> bool {
        // If network mode was approved, allow all domains
        if self
            .network_approved
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return true;
        }

        // Check default whitelist
        if DEFAULT_DOMAIN_WHITELIST.contains(&domain) {
            return true;
        }

        // Check if already approved
        {
            let approved = self.approved_domains.read().expect("domain cache lock");
            if approved.contains(domain) {
                return true;
            }
        }

        // Prompt user for new domain
        eprintln!("\n\x1b[33m⚠ Network access: {domain}\x1b[0m");
        eprint!("\x1b[33mAllow? [y/N]: \x1b[0m");
        io::stderr().flush().ok();

        let approved = read_yes_no().unwrap_or(false);
        eprintln!();

        if approved {
            // Cache approval for this domain
            let mut cache = self.approved_domains.write().expect("domain cache lock");
            cache.insert(domain.to_string());
            true
        } else {
            false
        }
    }
}

/// Expand ~ to home directory in a path.
fn expand_tilde(path: &std::path::Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    path.to_path_buf()
}

async fn path_exists(path: &std::path::Path) -> Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// Read y/n response from stdin.
fn read_yes_no() -> Result<bool> {
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    loop {
        if event::poll(Duration::from_secs(30))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('y' | 'Y') => {
                        eprint!("y");
                        return Ok(true);
                    }
                    KeyCode::Char('n' | 'N') | KeyCode::Enter | KeyCode::Esc => {
                        eprint!("n");
                        return Ok(false);
                    }
                    _ => {}
                },
                _ => {}
            }
        } else {
            // Timeout - default to no
            eprint!("n (timeout)");
            return Ok(false);
        }
    }
}

/// Register MCP tools as bash IPC commands.
///
/// This makes MCP tools available as bash commands (e.g., `resolve-library-id "tokio"`)
/// instead of direct LLM tool calls.
fn register_mcp_tools(conn: McpConnection, registry: &mut ToolRegistryBuilder) {
    let tools: Vec<_> = conn
        .mcp_definitions()
        .iter()
        .map(|def| {
            let name = def.name.clone();
            let description = def.description.clone().unwrap_or_default();
            let schema = def.input_schema.clone();
            (name, description, schema)
        })
        .collect();

    let conn = Arc::new(AsyncMutex::new(conn));

    for (name, description, schema) in tools {
        let conn = conn.clone();
        let tool_name = name.clone();

        // Build help text from schema
        let help = if description.is_empty() {
            schema_to_help(&schema)
        } else {
            format!("{description}\n\n{}", schema_to_help(&schema))
        };

        // Detect positional args from schema (all required fields in order)
        let positional_args = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        registry.configure_raw_handler(name, help, positional_args, move |args| {
            let conn = conn.clone();
            let tool_name = tool_name.clone();
            let schema = schema.clone();
            Box::pin(async move {
                // Parse CLI args into JSON object
                let arguments = match parse_cli_args_to_json(&schema, &args) {
                    Ok(value) => value,
                    Err(e) => return serde_json::json!({ "error": e.to_string() }).to_string(),
                };

                // Call the MCP tool
                let mut conn = conn.lock().await;
                match conn.call(&tool_name, arguments).await {
                    Ok(result) => {
                        // Extract text content from result
                        result
                            .content
                            .into_iter()
                            .filter_map(|c| {
                                if let aither_mcp::Content::Text(text_content) = c {
                                    Some(text_content.text)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                    Err(e) => format!("{{\"error\": \"{e}\"}}"),
                }
            })
        });
    }
}

/// Parse CLI-style arguments into a JSON object using the MCP schema.
fn parse_cli_args_to_json(
    schema: &serde_json::Value,
    args: &[String],
) -> anyhow::Result<serde_json::Value> {
    cli_to_json(schema, args)
}

/// Interactive CLI for testing aither agents.
#[derive(Parser, Debug)]
#[command(name = "aither", version, about)]
struct Args {
    /// Model to use. Auto-detects provider from model name prefix.
    #[arg(short, long)]
    model: Option<String>,

    /// Provider to use (openai, claude, gemini). Auto-detected if not specified.
    #[arg(short, long)]
    provider: Option<Provider>,

    /// Custom API base URL (for OpenAI-compatible endpoints, proxies, etc.)
    #[arg(short, long)]
    base_url: Option<String>,

    /// Path to MCP servers configuration file (JSON).
    #[arg(long)]
    mcp: Option<PathBuf>,

    /// System prompt for the agent.
    #[arg(short, long)]
    system: Option<String>,

    /// Disable Context7 MCP server (documentation lookup).
    #[arg(long)]
    no_context7: bool,

    /// Single prompt to run (headless mode). Runs query and exits.
    #[arg(long)]
    prompt: Option<String>,

    /// Quiet mode. Only output the response (useful with --prompt for scripting).
    #[arg(short, long)]
    quiet: bool,

    /// Path to skills directory. Skills are loaded from SKILL.md files in subdirectories.
    #[arg(long)]
    skills: Option<PathBuf>,

    /// Path to subagents directory. Subagents are markdown files invoked via `task <path> <prompt>`.
    #[arg(long)]
    subagents: Option<PathBuf>,

    /// Run as ACP server (for integration with Zed and other editors).
    #[arg(long)]
    acp: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Create cloud provider
    let base_url = args.base_url.as_deref();
    let (cloud, model, provider_name) = if let Some(provider) = args.provider {
        let model = args
            .model
            .as_deref()
            .unwrap_or_else(|| provider.default_model());
        (
            provider.create(model, base_url)?,
            model.to_string(),
            provider.to_string(),
        )
    } else {
        let (cloud, model) = provider::auto_detect(args.model.as_deref(), base_url)?;
        let provider_name = match &cloud {
            CloudProvider::OpenAI(_) => "OpenAI",
            CloudProvider::Claude(_) => "Claude",
            CloudProvider::Gemini(_) => "Gemini",
            CloudProvider::Copilot(_) => "Copilot",
        };
        (cloud, model, provider_name.to_string())
    };

    if !args.quiet {
        println!("Aither Agent CLI");
        println!("Provider: {provider_name}");
        println!("Model: {model}");
        if args.prompt.is_none() {
            println!("Commands: /quit, /clear, /history, /compact");
        }
        println!();
    }

    // ACP server mode: serve editors over stdio rather than running a REPL.
    if args.acp {
        return run_acp_server(cloud).await;
    }

    // Headless mode: run single prompt and exit. Both of these futures hold a
    // whole agent, so they are boxed rather than left on `main`'s stack frame.
    if let Some(ref prompt) = args.prompt {
        return Box::pin(run_headless(cloud, &args, prompt)).await;
    }

    Box::pin(run_repl(cloud, &args)).await
}

/// Run as ACP server for editor integration.
///
/// Each session the editor opens gets its own agent, sandboxed to the working
/// directory the editor supplied for that session.
async fn run_acp_server(cloud: CloudProvider) -> Result<()> {
    use aither_acp::AcpServer;

    let mut server = AcpServer::stdio("aither", env!("CARGO_PKG_VERSION"), move |cwd| {
        let cloud = cloud.clone();
        async move { acp_session_agent(cloud, cwd).await }
    })?;
    server.run().await?;
    Ok(())
}

/// Build the agent backing a single ACP session.
async fn acp_session_agent(
    cloud: CloudProvider,
    cwd: std::path::PathBuf,
) -> std::result::Result<
    Agent<CloudProvider, CloudProvider, CloudProvider>,
    aither_acp::protocol::AcpError,
> {
    let to_acp = |err: anyhow::Error| aither_acp::protocol::AcpError::Transport(err.to_string());

    let permission_handler = StatefulPermissionHandler::new(InteractivePermissionHandler::new());
    let bash_tool = aither_agent::sandbox::BashTool::new_in(&cwd, permission_handler, TokioGlobal)
        .await
        .map_err(|err| to_acp(err.into()))?;

    BashAgentBuilder::new(cloud.clone(), bash_tool)
        .tool(aither_agent::websearch::WebSearchTool::default())
        .tool(aither_agent::webfetch::WebFetchTool::new())
        .tool(aither_agent::TodoTool::new())
        .tool(aither_agent::sandbox::builtin::AskCommand::new(cloud))
        .with_default_prompt()
        .build()
        .map_err(|err| to_acp(anyhow::anyhow!("{err}")))
}

/// The concrete builder `build_agent` threads through its setup stages.
type CliAgentBuilder = BashAgentBuilder<
    CloudProvider,
    StatefulPermissionHandler<InteractivePermissionHandler>,
    TokioGlobal,
>;

async fn build_agent(
    cloud: CloudProvider,
    args: &Args,
) -> Result<Agent<CloudProvider, CloudProvider, CloudProvider, aither_agent::HCons<DebugHook, ()>>>
{
    // Create bash tool (creates random four-word working dir under system temp)
    // Uses interactive permission handler:
    // - Sandboxed: always allow (no prompt)
    // - Network: ask once, then remember
    // - Unsafe: always ask for each script
    let workdir_parent = std::env::temp_dir().join("aither");
    let permission_handler = StatefulPermissionHandler::new(InteractivePermissionHandler::new());
    let bash_tool =
        aither_agent::sandbox::BashTool::new_in(&workdir_parent, permission_handler, TokioGlobal)
            .await?;

    // Create bash-centric agent builder.
    // All tools become IPC commands accessible via bash.
    let mut builder = BashAgentBuilder::new(cloud.clone(), bash_tool)
        .tool(aither_agent::websearch::WebSearchTool::default())
        .tool(aither_agent::webfetch::WebFetchTool::new())
        .tool(aither_agent::TodoTool::new())
        .tool(aither_agent::sandbox::builtin::AskCommand::new(
            cloud.clone(),
        ));

    builder = attach_skills_and_subagents(builder, args).await?;
    builder = attach_mcp_servers(builder, args).await?;
    builder = attach_subagent_tool(builder, cloud);

    // Add system prompt
    let builder = if let Some(ref system) = args.system {
        builder.system_prompt_raw(system)
    } else {
        builder.with_default_prompt()
    };

    builder
        .hook(DebugHook)
        .build()
        .map_err(|err| anyhow::anyhow!("{err}"))
}

/// Loads the skills and subagent directories the user pointed at.
///
/// A missing directory is a warning rather than an error: the agent still runs,
/// just without those extras.
async fn attach_skills_and_subagents(
    mut builder: CliAgentBuilder,
    args: &Args,
) -> Result<CliAgentBuilder> {
    if let Some(ref skills_path) = args.skills {
        let expanded = expand_tilde(skills_path);
        if path_exists(&expanded).await? {
            builder = builder.with_skills(&expanded).await?;
            if !args.quiet {
                println!("Loaded skills from: {}", expanded.display());
            }
        } else if !args.quiet {
            eprintln!(
                "Warning: Skills directory not found: {}",
                expanded.display()
            );
        }
    }

    if let Some(ref subagents_path) = args.subagents {
        let expanded = expand_tilde(subagents_path);
        if path_exists(&expanded).await? {
            builder = builder.with_subagents(&expanded).await?;
            if !args.quiet {
                println!("Subagents directory: {}", expanded.display());
            }
        } else if !args.quiet {
            eprintln!(
                "Warning: Subagents directory not found: {}",
                expanded.display()
            );
        }
    }

    Ok(builder)
}

/// Connects the MCP servers this run should have: Context7 unless opted out,
/// plus anything in the user's MCP config file.
///
/// Context7 failing to connect is a warning — it is a convenience, not a
/// requirement — but a config file that cannot be read or parsed is an error,
/// because the user asked for those servers by name.
async fn attach_mcp_servers(mut builder: CliAgentBuilder, args: &Args) -> Result<CliAgentBuilder> {
    if !args.no_context7 {
        match McpConnection::http("https://mcp.context7.com/mcp").await {
            Ok(conn) => {
                if !args.quiet {
                    println!("Connected to Context7 MCP server");
                }
                builder = attach_mcp_connection(builder, conn);
            }
            Err(e) => {
                if !args.quiet {
                    eprintln!("Warning: Failed to connect to Context7: {e}");
                }
            }
        }
    }

    let Some(ref mcp_path) = args.mcp else {
        return Ok(builder);
    };
    let config_str = tokio::fs::read_to_string(mcp_path)
        .await
        .with_context(|| format!("failed to read MCP config from {}", mcp_path.display()))?;
    let config: McpServersConfig =
        serde_json::from_str(&config_str).with_context(|| "failed to parse MCP config")?;

    for (name, conn) in McpConnection::from_configs(&config).await? {
        if !args.quiet {
            println!("Connected to MCP server: {name}");
        }
        builder = attach_mcp_connection(builder, conn);
    }

    Ok(builder)
}

/// Registers one MCP connection's tools as bash commands.
///
/// The bash command listing only has room for a one-line summary, so each
/// tool's description is cut back to its first sentence.
fn attach_mcp_connection(mut builder: CliAgentBuilder, conn: McpConnection) -> CliAgentBuilder {
    for def in conn.mcp_definitions() {
        let desc = def.description.clone().unwrap_or_default();
        let desc = desc.split('.').next().unwrap_or(&desc).trim().to_string();
        builder = builder.tool_description(def.name.clone(), desc);
    }
    register_mcp_tools(conn, builder.tool_registry_mut());
    builder
}

/// Registers the subagent spawner as a bash IPC command.
///
/// Its base directory is the sandbox, so relative paths such as `.subagents/`
/// resolve the way they do for every other command the agent runs.
fn attach_subagent_tool(builder: CliAgentBuilder, cloud: CloudProvider) -> CliAgentBuilder {
    let subagent_tool = SubagentTool::new(cloud)
        .with_builtins()
        .with_base_dir(builder.sandbox_dir().to_string())
        .with_bash_tool_factory(builder.bash_tool_factory());

    let names: Vec<_> = subagent_tool
        .type_descriptions()
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let description = format!(
        "Spawn subagent for complex tasks (types: {})",
        names.join(", ")
    );

    builder.tool_with_desc(subagent_tool, description)
}

/// Run a single prompt and exit (headless mode).
async fn run_headless(cloud: CloudProvider, args: &Args, prompt: &str) -> Result<()> {
    let mut agent = build_agent(cloud, args).await?;

    match Box::pin(agent.query(prompt)).await {
        Ok(response) => {
            println!("{response}");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

async fn run_repl(cloud: CloudProvider, args: &Args) -> Result<()> {
    let mut agent = build_agent(cloud, args).await?;

    println!("Agent ready. Type your message or a command.\n");

    // REPL loop
    loop {
        let Some(input) = read_line("You> ")? else {
            break;
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => break,
                "/clear" => {
                    agent.clear_history();
                    println!("History cleared.");
                    continue;
                }
                "/history" => {
                    print_history(&agent);
                    continue;
                }
                "/compact" => {
                    match agent.compact(None).await {
                        Ok(Some(result)) => {
                            println!(
                                "\x1b[2mCompacted {} messages into summary, starting fresh session\x1b[22m",
                                result.messages_compacted
                            );
                        }
                        Ok(None) => {
                            println!("Nothing to compact (no conversation history).");
                        }
                        Err(e) => {
                            println!("\x1b[31mCompaction failed: {e}\x1b[0m");
                        }
                    }
                    continue;
                }
                cmd => {
                    println!("Unknown command: {cmd}");
                    println!("Available: /quit, /clear, /history, /compact");
                    continue;
                }
            }
        }

        // Query the agent
        println!("\nAgent>");
        match Box::pin(agent.query(input)).await {
            Ok(response) => {
                println!("{response}");
            }
            Err(e) => {
                println!("\x1b[31mError: {e}\x1b[0m");
            }
        }
        println!();
    }

    println!("Goodbye!");
    Ok(())
}

fn print_history<Advanced, Balanced, Fast, H>(agent: &Agent<Advanced, Balanced, Fast, H>)
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    let history = agent.history();
    if history.is_empty() {
        println!("No conversation history.");
    } else {
        println!("Conversation history ({} messages):", history.len());
        for msg in &history {
            let role = match msg.role() {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Tool => "Tool",
            };
            let content = truncate(msg.content(), 100);
            println!("  [{role}] {content}");
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn read_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut buffer = String::new();
    loop {
        if event::poll(Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Esc => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(None);
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(Some(buffer));
                    }
                    KeyCode::Backspace => {
                        if buffer.pop().is_some() {
                            print!("\u{8} \u{8}");
                            io::stdout().flush().ok();
                        }
                    }
                    KeyCode::Char(c) => {
                        buffer.push(c);
                        print!("{c}");
                        io::stdout().flush().ok();
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}

struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}
