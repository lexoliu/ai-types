//! MCP connections for agent integration.
//!
//! Provides a unified interface for connecting to MCP servers
//! via different transports (child process, HTTP, stdio).

use std::borrow::Cow;
use std::collections::HashMap;

use aither_core::llm::tool::ToolDefinition;
use async_channel::{Receiver, Sender};
use serde::Deserialize;

use crate::protocol::{CallToolResult, McpError, McpToolDefinition};
use crate::transport::{ChildProcessTransport, HttpTransport, StdioTransport};

use super::McpClient;

/// Configuration for a single MCP server.
///
/// This matches the format used by Claude Desktop and other MCP clients.
///
/// # Example JSON
///
/// ```json
/// {
///   "command": "npx",
///   "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    /// The command to run (for process-based servers).
    pub command: Option<String>,

    /// Arguments to pass to the command.
    #[serde(default)]
    pub args: Vec<String>,

    /// URL for HTTP-based servers.
    pub url: Option<String>,

    /// Optional environment variables for the process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Configuration for multiple MCP servers.
///
/// # Example JSON
///
/// ```json
/// {
///   "filesystem": {
///     "command": "npx",
///     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
///   },
///   "github": {
///     "command": "npx",
///     "args": ["-y", "@modelcontextprotocol/server-github"]
///   }
/// }
/// ```
pub type McpServersConfig = HashMap<String, McpServerConfig>;

/// A single MCP connection with its cached tools.
///
/// This enum handles all transport types internally, hiding the
/// transport abstraction from users. Use the constructor methods
/// ([`spawn`](Self::spawn), [`http`](Self::http), [`stdio`](Self::stdio))
/// to create connections.
#[non_exhaustive]
#[allow(missing_docs)]
pub enum McpConnection {
    /// Connection via spawned child process.
    Process {
        client: McpClient<ChildProcessTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
    /// Connection via HTTP.
    Http {
        client: McpClient<HttpTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
    /// Connection via stdio (for use as a subprocess).
    Stdio {
        client: McpClient<StdioTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
}

/// Service wrapper that serializes MCP tool calls through a command channel.
#[derive(Clone, Debug)]
pub struct McpToolService {
    tx: Sender<McpCommand>,
    tools: Vec<McpToolDefinition>,
}

#[derive(Debug)]
enum McpCommand {
    Call {
        name: String,
        arguments: serde_json::Value,
        reply: Sender<Result<CallToolResult, McpError>>,
    },
}

impl std::fmt::Debug for McpConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Process")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
            Self::Http {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Http")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
            Self::Stdio {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Stdio")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
        }
    }
}

impl McpConnection {
    /// Connect to an MCP server using configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = McpServerConfig {
    ///     command: Some("npx".to_string()),
    ///     args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
    ///     url: None,
    ///     env: HashMap::new(),
    /// };
    /// let conn = McpConnection::from_config(&config).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or connection fails.
    pub async fn from_config(config: &McpServerConfig) -> Result<Self, McpError> {
        if let Some(ref url) = config.url {
            // HTTP-based server
            Self::http(url).await
        } else if let Some(ref command) = config.command {
            // Process-based server
            let args: Vec<&str> = config
                .args
                .iter()
                .map(std::string::String::as_str)
                .collect();
            Self::spawn(command, &args).await
        } else {
            Err(McpError::InvalidConfig(
                "Config must have either 'command' or 'url'".to_string(),
            ))
        }
    }

    /// Connect to multiple MCP servers from a configuration map.
    ///
    /// Returns a vector of (name, connection) pairs for all connections.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config: McpServersConfig = serde_json::from_str(r#"{
    ///     "filesystem": {
    ///         "command": "npx",
    ///         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    ///     }
    /// }"#)?;
    ///
    /// let connections = McpConnection::from_configs(&config).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any connection fails.
    pub async fn from_configs(configs: &McpServersConfig) -> Result<Vec<(String, Self)>, McpError> {
        let mut connections = Vec::new();

        for (name, config) in configs {
            let conn = Self::from_config(config).await.map_err(|e| {
                McpError::InvalidConfig(format!("Failed to connect to MCP server '{name}': {e}"))
            })?;
            tracing::info!("Connected to MCP server: {name}");
            connections.push((name.clone(), conn));
        }

        Ok(connections)
    }

    /// Connect to an MCP server via a spawned child process.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conn = McpConnection::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned or connection fails.
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self, McpError> {
        let transport = ChildProcessTransport::spawn(program, args)?;
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Process {
            client,
            tools,
            server_name,
        })
    }

    /// Connect to an MCP server via HTTP.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conn = McpConnection::http("http://localhost:3000/mcp").await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP connection fails.
    pub async fn http(url: &str) -> Result<Self, McpError> {
        let transport = HttpTransport::new(url);
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Http {
            client,
            tools,
            server_name,
        })
    }

    /// Connect to an MCP server via HTTP with authentication.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP connection fails.
    pub async fn http_with_auth(url: &str, auth: &str) -> Result<Self, McpError> {
        let transport = HttpTransport::new(url).with_auth(auth);
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Http {
            client,
            tools,
            server_name,
        })
    }

    /// Connect via stdio (when running as a subprocess).
    ///
    /// # Errors
    ///
    /// Returns an error if stdio cannot be initialized.
    pub async fn stdio() -> Result<Self, McpError> {
        let transport = StdioTransport::new().map_err(|e| McpError::Transport(e.to_string()))?;
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Stdio {
            client,
            tools,
            server_name,
        })
    }

    /// Returns the server name if available.
    #[must_use]
    pub fn server_name(&self) -> Option<&str> {
        match self {
            Self::Process { server_name, .. }
            | Self::Http { server_name, .. }
            | Self::Stdio { server_name, .. } => server_name.as_deref(),
        }
    }

    /// Returns the MCP tool definitions.
    #[must_use]
    pub fn mcp_definitions(&self) -> &[McpToolDefinition] {
        match self {
            Self::Process { tools, .. } | Self::Http { tools, .. } | Self::Stdio { tools, .. } => {
                tools
            }
        }
    }

    /// Returns aither-compatible tool definitions.
    ///
    /// A tool whose schema the server sent in a form JSON Schema does not allow
    /// is dropped with a warning: it is unusable, and one bad entry should not
    /// cost the caller every other tool on the server.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        convert_definitions(self.mcp_definitions())
    }

    /// Check if this connection has a tool with the given name.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.mcp_definitions().iter().any(|d| d.name == name)
    }

    /// Call a tool on this MCP server.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        match self {
            Self::Process { client, .. } => client.call_tool(name, arguments).await,
            Self::Http { client, .. } => client.call_tool(name, arguments).await,
            Self::Stdio { client, .. } => client.call_tool(name, arguments).await,
        }
    }

    /// Close the connection.
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails.
    pub async fn close(&mut self) -> Result<(), McpError> {
        match self {
            Self::Process { client, .. } => client.close().await,
            Self::Http { client, .. } => client.close().await,
            Self::Stdio { client, .. } => client.close().await,
        }
    }
}

impl McpToolService {
    /// Creates a new MCP tool service and starts its background worker.
    #[must_use]
    pub fn new(conn: McpConnection) -> Self {
        let tools = conn.mcp_definitions().to_vec();
        let (tx, rx) = async_channel::unbounded();
        std::thread::spawn(move || run_service(&rx, conn));
        Self { tx, tools }
    }

    /// Returns the MCP tool definitions.
    #[must_use]
    pub fn mcp_definitions(&self) -> &[McpToolDefinition] {
        &self.tools
    }

    /// Returns aither-compatible tool definitions.
    ///
    /// See [`McpConnection::definitions`] for how an unusable schema is handled.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        convert_definitions(&self.tools)
    }

    /// Check if this service has a tool with the given name.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|d| d.name == name)
    }

    /// Call a tool on this MCP service.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(McpCommand::Call {
                name: name.to_string(),
                arguments,
                reply: reply_tx,
            })
            .await
            .map_err(|_| McpError::ConnectionClosed)?;
        reply_rx
            .recv()
            .await
            .map_err(|_| McpError::ConnectionClosed)?
    }
}

/// Converts the server's tool list into aither definitions.
///
/// The schemas come off the wire from a third-party server, so one that is not
/// a JSON schema at all is a rejection to log, not a reason to fail: that tool
/// is skipped and the rest are returned.
fn convert_definitions(defs: &[McpToolDefinition]) -> Vec<ToolDefinition> {
    defs.iter()
        .filter_map(|def| {
            let name: Cow<'static, str> = Cow::Owned(def.name.clone());
            let description: Cow<'static, str> =
                Cow::Owned(def.description.clone().unwrap_or_default());
            match ToolDefinition::from_parts(name, description, def.input_schema.clone()) {
                Ok(definition) => Some(definition),
                Err(err) => {
                    tracing::warn!(error = %err, "skipping MCP tool with an unusable schema");
                    None
                }
            }
        })
        .collect()
}

/// Drives one MCP connection until every command sender has gone away.
///
/// The caller's closure owns the receiver, so the loop ends when the last
/// [`McpToolService`] handle is dropped; `conn` is owned here so the transport
/// closes when this worker returns.
fn run_service(rx: &Receiver<McpCommand>, mut conn: McpConnection) {
    async_io::block_on(async {
        while let Ok(cmd) = rx.recv().await {
            match cmd {
                McpCommand::Call {
                    name,
                    arguments,
                    reply,
                } => {
                    let result = conn.call(&name, arguments).await;
                    let _ = reply.send(result).await;
                }
            }
        }
    });
}
