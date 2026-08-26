//! MCP server implementation.
//!
//! The server exposes aither tools as an MCP server for external clients.

#[path = "server.rs"]
mod server_impl;

pub use server_impl::McpServer;
