use std::str::FromStr;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;
use aither_core::llm::{LLMRequest, LanguageModel, Message, Tool, ToolOutput};
use anyhow::Context;
use llm::{Action, ExtractedFacts, MemoryDecision};
use store::{MemoryStore, SearchFilters};
use tracing::debug;
use uuid::Uuid;

pub mod error;
pub mod llm;
pub mod store;

pub use error::{Mem0Error, Result};
pub use store::{InMemoryStore, Memory, SearchResult};

pub struct SearchTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for SearchTool<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    type Arguments = String;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "search_memories".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let result = self
            .inner
            .retrieve_formatted(&arguments, 50)
            .await
            .context("Fail to retrive memory")?;
        Ok(ToolOutput::text(result))
    }
}

pub struct AddFactTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for AddFactTool<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    type Arguments = Vec<String>;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "add_fact".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        self.inner
            .add_fact(arguments)
            .await
            .context("Fail to add fact")?;
        Ok(ToolOutput::Done)
    }
}

/// Configuration for Mem0.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of similar memories to retrieve for update context.
    pub retrieve_count: usize,
    /// User ID to associate with memories.
    pub user_id: Option<String>,
    /// Agent ID to associate with memories.
    pub agent_id: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            retrieve_count: 5,
            user_id: None,
            agent_id: None,
        }
    }
}

/// Mem0 memory manager.
struct Mem0Inner<L, E, S> {
    llm: L,
    runtime: async_lock::Mutex<Mem0Runtime<E, S>>,
    config: Config,
}

struct Mem0Runtime<E, S> {
    pending_facts: Vec<String>,
    embedder: E,
    store: S,
}

pub struct Mem0<L, E, S> {
    inner: Arc<Mem0Inner<L, E, S>>,
}

impl<L, E, S> Clone for Mem0<L, E, S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<L, E, S> Mem0<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    /// Create a new Mem0 instance.
    pub fn new(llm: L, embedder: E, store: S, config: Config) -> Self {
        Self {
            inner: Arc::new(Mem0Inner {
                llm,
                runtime: async_lock::Mutex::new(Mem0Runtime {
                    pending_facts: Vec::new(),
                    embedder,
                    store,
                }),
                config,
            }),
        }
    }

    /// Add a new interaction to memory.
    ///
    /// This triggers the extraction and update pipeline:
    /// 1. Extract facts from the messages.
    /// 2. For each fact, retrieve similar memories.
    /// 3. Decide on an operation (Add, Update, Delete, Noop).
    /// 4. Execute the operation.
    pub async fn add(&self, messages: &[Message]) -> Result<()> {
        // 1. Extract facts
        let facts = self.extract_facts(messages).await?;

        self.add_fact(facts).await?;

        Ok(())
    }

    /// Add new facts to memory.
    /// Tip: This method batches fact additions for efficiency and accuracy.
    /// Put it simply, only one fact extraction task is running at a time. Facts added at this moment will be queued and processed together.
    /// And the caller have to wait if you `.await` this method.
    ///
    /// So if you doesn't mind the result of adding facts, you can spawn a task to call this method.
    pub async fn add_fact(&self, facts: Vec<String>) -> Result<()> {
        let mut runtime = self.inner.runtime.lock().await;
        runtime.pending_facts.extend(facts);
        let facts = std::mem::take(&mut runtime.pending_facts);

        for fact in facts {
            let embedding = runtime
                .embedder
                .embed(&fact)
                .await
                .map_err(Mem0Error::Embedding)?;

            debug!("Embedding generated for fact: {}", fact);

            let filters = SearchFilters {
                user_id: self.inner.config.user_id.clone(),
                agent_id: self.inner.config.agent_id.clone(),
            };
            let existing_memories = runtime
                .store
                .search(&embedding, self.inner.config.retrieve_count, filters)
                .await?;

            debug!(
                "Found {} similar existing memories for fact.",
                existing_memories.len()
            );

            let decision = self.decide_operation(&fact, &existing_memories).await?;

            match decision.action {
                Action::Add => {
                    let mut memory = Memory::new(fact, embedding);
                    if let Some(uid) = &self.inner.config.user_id {
                        memory = memory.with_user_id(uid);
                    }
                    if let Some(aid) = &self.inner.config.agent_id {
                        memory = memory.with_agent_id(aid);
                    }

                    runtime.store.add(memory).await?;
                }
                Action::Update => {
                    if let (Some(id_str), Some(content)) =
                        (decision.memory_id, decision.new_content)
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        let new_embedding = runtime
                            .embedder
                            .embed(&content)
                            .await
                            .map_err(Mem0Error::Llm)?;

                        if let Some(mut existing) = runtime.store.get(id).await? {
                            existing.content = content;
                            existing.embedding = new_embedding;
                            existing.updated_at = time::OffsetDateTime::now_utc();
                            runtime.store.update(existing).await?;
                        }
                    }
                }
                Action::Delete => {
                    if let Some(id_str) = decision.memory_id
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        runtime.store.delete(id).await?;
                    }
                }
                Action::Noop => {}
            }
        }

        Ok(())
    }

    /// Search for relevant memories.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<store::SearchResult>> {
        let runtime = self.inner.runtime.lock().await;
        let embedding = runtime
            .embedder
            .embed(query)
            .await
            .map_err(Mem0Error::Llm)?;
        let filters = SearchFilters {
            user_id: self.inner.config.user_id.clone(),
            agent_id: self.inner.config.agent_id.clone(),
        };
        runtime.store.search(&embedding, limit, filters).await
    }

    pub fn add_fact_tool(&self) -> AddFactTool<L, E, S> {
        AddFactTool {
            inner: self.clone(),
        }
    }

    pub fn search_tool(&self) -> SearchTool<L, E, S> {
        SearchTool {
            inner: self.clone(),
        }
    }

    /// Return all stored memories.
    pub async fn memories(&self) -> Result<Vec<Memory>> {
        let runtime = self.inner.runtime.lock().await;
        runtime.store.all().await
    }

    /// Retrieve relevant memories and format them for context injection.
    pub async fn retrieve_formatted(&self, query: &str, limit: usize) -> Result<String> {
        let results = self.search(query, limit).await?;
        if results.is_empty() {
            return Ok(String::new());
        }

        let formatted = results
            .into_iter()
            .map(|r| format!("- {}", r.memory.content))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!("Relevant Memories:\n{}", formatted))
    }

    async fn extract_facts(&self, messages: &[Message]) -> Result<Vec<String>> {
        // Format messages for the prompt
        let context = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role(), m.content()))
            .collect::<Vec<_>>()
            .join("\n");

        let system_prompt = include_str!("../prompts/extractor.txt");

        let request = LLMRequest::new(vec![
            Message::system(system_prompt),
            Message::user(format!(
                "Extract facts from the following conversation:\n\n{}",
                context
            )),
        ]);

        let extracted: ExtractedFacts = self
            .inner
            .llm
            .generate(request)
            .await
            .map_err(Mem0Error::Llm)?;
        Ok(extracted.facts)
    }

    async fn decide_operation(
        &self,
        fact: &str,
        existing_memories: &[store::SearchResult],
    ) -> Result<MemoryDecision> {
        let memories_context = existing_memories
            .iter()
            .map(|r| format!("ID: {}\nContent: {}\n", r.memory.id, r.memory.content))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let system_prompt = include_str!("../prompts/manager.txt");

        let user_prompt = format!(
            "New Fact: {}\n\nExisting Memories:\n{}\n\nDecide the operation.",
            fact, memories_context
        );

        let request = LLMRequest::new(vec![
            Message::system(system_prompt),
            Message::user(user_prompt),
        ]);

        debug!("Deciding operation for fact: {}", fact);

        let decision: MemoryDecision = self
            .inner
            .llm
            .generate(request)
            .await
            .map_err(Mem0Error::Llm)?;

        Ok(decision)
    }
}
