# AI Types

[![Crates.io](https://img.shields.io/crates/v/ai_types.svg)](https://crates.io/crates/ai_types)
[![Documentation](https://docs.rs/ai_types/badge.svg)](https://docs.rs/ai_types)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A `no_std` Rust library providing unified trait abstractions for AI models including language models, embedding models, and image generators. This library enables you to write AI-powered applications that are agnostic to specific AI service providers.

## Features

- **🚫 `no_std` compatible** - Works in embedded and resource-constrained environments
- **🔧 Provider agnostic** - Unified traits for different AI model types
- **⚡ Async/await support** - Built on modern Rust async primitives
- **🛠️ Tool system** - Support for function calling and tool use
- **📸 Multimodal** - Support for text, images, and embeddings
- **🔒 Type-safe** - Leverages Rust's type system for robust AI applications

## Supported Model Types

### Language Models (`LanguageModel`)
- Text generation and completion
- Conversation management with structured messages
- Tool/function calling capabilities
- Structured output generation with JSON schemas
- Built-in text summarization and categorization

### Embedding Models (`EmbeddingModel`)
- Text-to-vector conversion
- Configurable embedding dimensions
- Async-first design

### Image Generators (`ImageGenerator`)
- Text-to-image generation
- Streaming image data support
- Progressive loading capabilities

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai_types = "0.1.0"
```

### Basic Language Model Usage

```rust
use ai_types::{LanguageModel, llm::{Message, Role, model::Profile}};
use futures_lite::StreamExt;
use futures_core::Stream;

// Your language model implementation
struct MyLanguageModel;

impl LanguageModel for MyLanguageModel {
    fn respond(&self, messages: impl ai_types::llm::Messages) -> impl Stream<Item = ai_types::llm::Result> + Send + Unpin {
        // Implementation would call your AI service
        futures_lite::stream::iter(vec![Ok("Hello!".to_string())])
    }
    
    fn complete(&self, prefix: &str) -> impl Stream<Item = ai_types::llm::Result> + Send + Unpin {
        // Implementation for text completion
        futures_lite::stream::iter(vec![Ok(" world!".to_string())])
    }
    
    fn profile(&self) -> Profile {
        Profile::new("my-model", "A sample language model", 4096)
    }
}

#[tokio::main]
async fn main() {
    let model = MyLanguageModel;
    let messages = vec![
        Message::system("You are a helpful assistant"),
        Message::user("Hello, how are you?")
    ];
    
    let mut response_stream = model.respond(messages);
    while let Some(chunk) = response_stream.next().await {
        print!("{}", chunk.unwrap());
    }
}
```

### Tool Usage

```rust
use ai_types::llm::{Tool, tool::Tools};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize)]
struct CalculatorArgs {
    operation: String,
    a: f64,
    b: f64,
}

struct Calculator;

impl Tool for Calculator {
    const NAME: &str = "calculator";
    const DESCRIPTION: &str = "Performs basic mathematical operations";
    type Arguments = CalculatorArgs;
    
    async fn call(&mut self, args: Self::Arguments) -> ai_types::llm::tool::Result {
        match args.operation.as_str() {
            "add" => Ok((args.a + args.b).to_string()),
            "subtract" => Ok((args.a - args.b).to_string()),
            _ => Err(anyhow::Error::msg("Unknown operation")),
        }
    }
}

#[tokio::main]
async fn main() {
    let mut tools = Tools::new();
    tools.register(Calculator);
    
    let result = tools.call("calculator", 
        r#"{"operation": "add", "a": 5, "b": 3}"#.to_string()).await;
    println!("Result: {}", result.unwrap()); // "8"
}
```

### Embedding Model Usage

```rust
use ai_types::EmbeddingModel;

struct MyEmbeddingModel;

impl EmbeddingModel for MyEmbeddingModel {
    fn dim(&self) -> usize {
        768  // Common embedding dimension
    }
    
    async fn embed(&self, text: &str) -> Vec<f32> {
        // Implementation would call your embedding service
        vec![0.1, 0.2, 0.3] // Mock embedding
    }
}
```

### Image Generation Usage

```rust
use ai_types::ImageGenerator;
use futures_core::Stream;

struct MyImageGenerator;

impl ImageGenerator for MyImageGenerator {
    fn generate_image(&self, prompt: &str) -> impl Stream<Item = ai_types::image::Data> + Send {
        // Implementation would call your image generation service
        futures_lite::stream::iter(vec![vec![0u8; 1024]]) // Mock image data
    }
}
```

## Architecture

The library is organized into three main modules:

- **`llm`** - Language model traits, message types, and tool system
- **`embedding`** - Embedding model traits and utilities
- **`image`** - Image generation traits and types

### Message System

The message system supports structured conversations with different participant roles:

- `Role::User` - Messages from the human user
- `Role::Assistant` - Messages from the AI assistant
- `Role::System` - System prompts and instructions
- `Role::Tool` - Messages from tool/function calls

Messages can include attachments (URLs) and annotations for rich content support.

### Tool System

The tool system enables language models to call external functions:

1. **Define tools** using the `Tool` trait with typed arguments
2. **Register tools** in a `Tools` collection
3. **Generate schemas** automatically for language model integration
4. **Execute tools** with type-safe argument parsing

### Model Profiles

Each model can provide a `Profile` containing:

- Model name and description
- Supported abilities (tool use, vision, audio, web search)
- Context length limits
- Pricing information
- Supported parameters

## `no_std` Support

This library is `no_std` compatible and only requires `alloc` for dynamic allocations. It uses:

- `alloc::vec::Vec` instead of `std::vec::Vec`
- `alloc::string::String` instead of `std::string::String`
- `core::future::Future` instead of `std::future::Future`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Examples

For more examples and usage patterns, check out the `tests/` directory in this repository.