# AI Types

[![Crates.io](https://img.shields.io/crates/v/ai_types.svg)](https://crates.io/crates/ai_types)
[![Documentation](https://docs.rs/ai_types/badge.svg)](https://docs.rs/ai_types)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A library providing unified trait abstractions for AI models including language models, embedding models, and image generators. This library enables you to write AI-powered applications that are agnostic to specific AI service providers.

## Features

- **🚫 `no_std` compatible** - Works in embedded and resource-constrained environments
- **🔧 Provider agnostic** - Unified traits for different AI model types
- **⚡ Async/await support** - Built on modern Rust async primitives
- **🛠️ Tool system** - Support for function calling and tool use
- **📸 Multimodal** - Support for text, images, and embeddings

## Supported Model Types
- LLM
- Embedding model
- Text-to-image model

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai_types = "0.0.1"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.