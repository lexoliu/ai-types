//! MCP client implementation.
//!
//! The client connects to MCP servers and provides methods to
//! list and call tools, read resources, etc.

#[path = "client.rs"]
mod client_impl;
mod toolset;

pub use client_impl::McpClient;
pub use toolset::{
    McpConnection, McpServerConfig, McpServersConfig, McpToolService, register_terminal_commands,
};
