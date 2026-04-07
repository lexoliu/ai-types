use std::str::FromStr;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;
use aither_core::llm::{LLMRequest, LanguageModel, Message, Tool, ToolResult};
use anyhow::Context;
use async_channel::Sender;
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
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    type Arguments = String;
    type Res = ToolResult;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "search_memories".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<Self::Res> {
        let result = self
            .inner
            .retrieve_formatted(&arguments, 50)
            .await
            .context("Fail to retrive memory")?;
        Ok(ToolResult::text(result))
    }
}

pub struct AddFactTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for AddFactTool<L, E, S>
where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    type Arguments = Vec<String>;
    type Res = ToolResult;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "add_fact".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<Self::Res> {
        self.inner
            .add_fact(arguments)
            .await
            .context("Fail to add fact")?;
        Ok(ToolResult::Done)
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

enum Mem0Command {
    AddMessages {
        messages: Vec<Message>,
        response_tx: Sender<Result<()>>,
    },
    AddFacts {
        facts: Vec<String>,
        response_tx: Sender<Result<()>>,
    },
    Search {
        query: String,
        limit: usize,
        response_tx: Sender<Result<Vec<SearchResult>>>,
    },
    Memories {
        response_tx: Sender<Result<Vec<Memory>>>,
    },
}

struct Mem0Actor<L, E, S> {
    llm: L,
    embedder: E,
    store: S,
    config: Config,
}

struct Mem0Inner {
    command_tx: Sender<Mem0Command>,
}

pub struct Mem0<L, E, S> {
    inner: Arc<Mem0Inner>,
    _marker: std::marker::PhantomData<(L, E, S)>,
}

impl<L, E, S> Clone for Mem0<L, E, S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<L, E, S> Mem0<L, E, S>
where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    /// Create a new Mem0 instance.
    pub fn new(llm: L, embedder: E, store: S, config: Config) -> Self {
        let (command_tx, command_rx) = async_channel::unbounded();
        let actor = Mem0Actor {
            llm,
            embedder,
            store,
            config,
        };
        std::thread::Builder::new()
            .name("mem0-runtime".to_string())
            .spawn(move || futures_lite::future::block_on(run_mem0_actor(actor, command_rx)))
            .expect("mem0 runtime thread must spawn");
        Self {
            inner: Arc::new(Mem0Inner { command_tx }),
            _marker: std::marker::PhantomData,
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
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.inner
            .command_tx
            .send(Mem0Command::AddMessages {
                messages: messages.to_vec(),
                response_tx,
            })
            .await
            .expect("mem0 runtime command receiver must be alive");
        response_rx
            .recv()
            .await
            .expect("mem0 runtime response must be delivered")
    }

    /// Add new facts to memory.
    /// Tip: This method batches fact additions for efficiency and accuracy.
    /// Put it simply, only one fact extraction task is running at a time. Facts added at this moment will be queued and processed together.
    /// And the caller have to wait if you `.await` this method.
    ///
    /// So if you doesn't mind the result of adding facts, you can spawn a task to call this method.
    pub async fn add_fact(&self, facts: Vec<String>) -> Result<()> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.inner
            .command_tx
            .send(Mem0Command::AddFacts { facts, response_tx })
            .await
            .expect("mem0 runtime command receiver must be alive");
        response_rx
            .recv()
            .await
            .expect("mem0 runtime response must be delivered")
    }

    /// Search for relevant memories.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<store::SearchResult>> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.inner
            .command_tx
            .send(Mem0Command::Search {
                query: query.to_string(),
                limit,
                response_tx,
            })
            .await
            .expect("mem0 runtime command receiver must be alive");
        response_rx
            .recv()
            .await
            .expect("mem0 runtime response must be delivered")
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
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.inner
            .command_tx
            .send(Mem0Command::Memories { response_tx })
            .await
            .expect("mem0 runtime command receiver must be alive");
        response_rx
            .recv()
            .await
            .expect("mem0 runtime response must be delivered")
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
}

async fn run_mem0_actor<L, E, S>(
    mut actor: Mem0Actor<L, E, S>,
    command_rx: async_channel::Receiver<Mem0Command>,
) where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    while let Ok(command) = command_rx.recv().await {
        match command {
            Mem0Command::AddMessages {
                messages,
                response_tx,
            } => {
                let _ = response_tx.send(actor.add_messages(messages).await).await;
            }
            Mem0Command::AddFacts { facts, response_tx } => {
                let _ = response_tx.send(actor.add_facts(facts).await).await;
            }
            Mem0Command::Search {
                query,
                limit,
                response_tx,
            } => {
                let _ = response_tx
                    .send(actor.search(query.as_str(), limit).await)
                    .await;
            }
            Mem0Command::Memories { response_tx } => {
                let _ = response_tx.send(actor.memories().await).await;
            }
        }
    }
}

impl<L, E, S> Mem0Actor<L, E, S>
where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        let facts = extract_facts(&self.llm, &messages).await?;
        self.add_facts(facts).await
    }

    async fn add_facts(&mut self, facts: Vec<String>) -> Result<()> {
        for fact in facts {
            let embedding = self
                .embedder
                .embed(&fact)
                .await
                .map_err(Mem0Error::Embedding)?;

            debug!("Embedding generated for fact: {}", fact);

            let filters = SearchFilters {
                user_id: self.config.user_id.clone(),
                agent_id: self.config.agent_id.clone(),
            };
            let existing_memories = self
                .store
                .search(&embedding, self.config.retrieve_count, filters)
                .await?;

            debug!(
                "Found {} similar existing memories for fact.",
                existing_memories.len()
            );

            let decision = decide_operation(&self.llm, &fact, &existing_memories).await?;

            match decision.action {
                Action::Add => {
                    let mut memory = Memory::new(fact, embedding);
                    if let Some(uid) = &self.config.user_id {
                        memory = memory.with_user_id(uid);
                    }
                    if let Some(aid) = &self.config.agent_id {
                        memory = memory.with_agent_id(aid);
                    }
                    self.store.add(memory).await?;
                }
                Action::Update => {
                    if let (Some(id_str), Some(content)) =
                        (decision.memory_id, decision.new_content)
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        let new_embedding = self
                            .embedder
                            .embed(&content)
                            .await
                            .map_err(Mem0Error::Llm)?;

                        if let Some(mut existing) = self.store.get(id).await? {
                            existing.content = content;
                            existing.embedding = new_embedding;
                            existing.updated_at = time::OffsetDateTime::now_utc();
                            self.store.update(existing).await?;
                        }
                    }
                }
                Action::Delete => {
                    if let Some(id_str) = decision.memory_id
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        self.store.delete(id).await?;
                    }
                }
                Action::Noop => {}
            }
        }
        Ok(())
    }

    async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let embedding = self.embedder.embed(query).await.map_err(Mem0Error::Llm)?;
        let filters = SearchFilters {
            user_id: self.config.user_id.clone(),
            agent_id: self.config.agent_id.clone(),
        };
        self.store.search(&embedding, limit, filters).await
    }

    async fn memories(&mut self) -> Result<Vec<Memory>> {
        self.store.all().await
    }
}

async fn extract_facts<L: LanguageModel>(llm: &L, messages: &[Message]) -> Result<Vec<String>> {
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

    let extracted: ExtractedFacts = llm.generate(request).await.map_err(Mem0Error::Llm)?;
    Ok(extracted.facts)
}

async fn decide_operation<L: LanguageModel>(
    llm: &L,
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

    llm.generate(request).await.map_err(Mem0Error::Llm)
}
