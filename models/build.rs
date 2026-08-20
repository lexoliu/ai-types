//! Build script that fetches `LiteLLM` model data and generates a static registry.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

#[path = "src/convert.rs"]
mod convert;

fn main() {
    println!("cargo::rerun-if-changed=data/litellm_snapshot.json");
    println!("cargo::rerun-if-changed=src/convert.rs");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let snapshot_path = Path::new(&manifest_dir).join("data/litellm_snapshot.json");

    // Read the committed snapshot only. Fetching at build time would make the
    // generated registry depend on when the build ran rather than on the
    // commit, and would fail in sandboxed or offline builds. The snapshot is
    // refreshed by the `update-models` workflow, which opens a pull request.
    let json_bytes = fs::read(&snapshot_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read LiteLLM snapshot at {}: {e}",
            snapshot_path.display()
        )
    });

    let entries = convert::parse_litellm_json(&json_bytes);

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("generated.rs");

    let code = generate_registry_code(&entries);
    fs::write(&out_path, code).expect("Failed to write generated.rs");
}

fn generate_registry_code(entries: &[convert::ConvertedEntry]) -> String {
    let mut code = String::with_capacity(entries.len() * 512);

    // No `use` statements — this file is include!'d into registry.rs
    // which already imports the needed types.
    //
    // Generated text cannot satisfy style lints without changing the generator
    // itself, so silence them here rather than in the crate that includes it.
    code.push_str(
        "#[allow(clippy::all, clippy::pedantic, clippy::nursery, missing_docs)]\n         const _GENERATED: () = ();\n",
    );

    // Generate the static entries array
    code.push_str("#[allow(clippy::unreadable_literal, clippy::too_many_lines)]\n");
    code.push_str("static ENTRIES: &[ModelEntry] = &[\n");
    for entry in entries {
        let provider_expr = provider_to_expr(&entry.provider);
        let mode_expr = mode_to_expr(&entry.mode);

        code.push_str("    ModelEntry::new_static(\n");
        let _ = writeln!(code, "        {:?},", entry.litellm_id);
        let _ = writeln!(code, "        {:?},", entry.canonical_id);
        let _ = writeln!(code, "        {provider_expr},");
        let _ = writeln!(code, "        {mode_expr},");
        let _ = writeln!(code, "        {:?},", entry.max_input_tokens);
        let _ = writeln!(code, "        {:?},", entry.max_output_tokens);

        // Pricing
        let _ = write!(
            code,
            "        Pricing::new({:?}, {:?})",
            entry.input_cost_per_token, entry.output_cost_per_token
        );
        if let Some(cr) = entry.cache_read_per_token {
            let _ = write!(code, ".with_cache_read({cr:?})");
        }
        if let Some(cw) = entry.cache_write_per_token {
            let _ = write!(code, ".with_cache_write({cw:?})");
        }
        if let Some(r) = entry.reasoning_per_token {
            let _ = write!(code, ".with_reasoning({r:?})");
        }
        if let Some(i) = entry.image_per_token {
            let _ = write!(code, ".with_image({i:?})");
        }
        code.push_str(",\n");

        // Abilities
        let abilities_str: Vec<String> = entry
            .abilities
            .iter()
            .map(|a| format!("Ability::{a}"))
            .collect();
        let _ = writeln!(code, "        &[{}],", abilities_str.join(", "));

        // Deprecation date
        match &entry.deprecation_date {
            Some(d) => {
                let _ = writeln!(code, "        Some({d:?}),");
            }
            None => code.push_str("        None,\n"),
        }

        code.push_str("    ),\n");
    }
    code.push_str("];\n");

    code
}

fn provider_to_expr(provider: &str) -> String {
    match provider {
        "openai" => "Provider::OpenAI".to_string(),
        "anthropic" => "Provider::Anthropic".to_string(),
        "gemini" => "Provider::Google".to_string(),
        "deepseek" => "Provider::DeepSeek".to_string(),
        "xai" => "Provider::XAI".to_string(),
        "mistral" => "Provider::Mistral".to_string(),
        "meta" => "Provider::Meta".to_string(),
        "alibaba" => "Provider::Alibaba".to_string(),
        "bedrock" => "Provider::Bedrock".to_string(),
        "azure" => "Provider::Azure".to_string(),
        "vertex_ai" => "Provider::VertexAI".to_string(),
        "groq" => "Provider::Groq".to_string(),
        "together_ai" => "Provider::Together".to_string(),
        "fireworks_ai" => "Provider::Fireworks".to_string(),
        "replicate" => "Provider::Replicate".to_string(),
        "cohere" => "Provider::Cohere".to_string(),
        "perplexity" => "Provider::Perplexity".to_string(),
        "github_copilot" => "Provider::Copilot".to_string(),
        other => format!("Provider::Other(std::borrow::Cow::Borrowed({other:?}))"),
    }
}

fn mode_to_expr(mode: &str) -> &'static str {
    match mode {
        "embedding" => "ModelMode::Embedding",
        "image_generation" => "ModelMode::ImageGeneration",
        "audio_transcription" => "ModelMode::AudioTranscription",
        "audio_speech" => "ModelMode::AudioSpeech",
        "rerank" => "ModelMode::Rerank",
        "moderation" => "ModelMode::Moderation",
        "completion" => "ModelMode::Completion",
        "video_generation" => "ModelMode::VideoGeneration",
        "search" => "ModelMode::Search",
        "ocr" => "ModelMode::Ocr",
        // "chat" and anything unrecognised both map to the chat mode.
        _ => "ModelMode::Chat",
    }
}
