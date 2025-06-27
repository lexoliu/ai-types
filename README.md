# AI Types

[![Crates.io](https://img.shields.io/crates/v/ai-types.svg)](https://crates.io/crates/ai-types)
[![Documentation](https://docs.rs/ai_types/badge.svg)](https://docs.rs/ai_types)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Unified trait abstractions for AI models. Write provider-agnostic AI applications.

## Features

- **🔧 Provider agnostic** - Unified traits for different AI models
- **⚡ Async/await** - Built on modern Rust async primitives
- **🛠️ Tool system** - Function calling support
- **📸 Multimodal** - Text, images, embeddings, and audio
- **🎵 Audio processing** - Generation and transcription

## Supported Models

- **LLM** - Text generation and conversation
- **Embedding** - Text and multimodal embeddings
- **Image generation** - Text-to-image synthesis
- **Audio generation** - Text-to-speech
- **Audio transcription** - Speech-to-text

## Quick Start

```toml
[dependencies]
ai_types = "0.0.1"
```