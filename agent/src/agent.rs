//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages conversation memory, applies context compression, and
//! handles tool execution in an agent-controlled loop.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::{Duration, Instant};

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, Message, model::Profile as ModelProfile},
};
use futures_core::Stream;
use futures_lite::StreamExt;

use crate::{
    compression::{ContextStrategy, estimate_context_usage},
    config::{AgentConfig, AgentKind},
    context::Context,
    error::AgentError,
    event::AgentEvent,
    hook::{
        Hook, PostToolAction, PreToolAction, StopContext, StopReason, ToolResultContext,
        ToolUseContext,
    },
    todo::{TodoItem, TodoList, TodoStatus},
    tools::AgentTools,
    transcript::Transcript,
    working_docs,
};

use aither_sandbox::{BackgroundTaskReceiver, JobRegistry, OutputStore};
use std::sync::Arc;

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactResult {
    /// Number of messages that were compacted.
    pub messages_compacted: usize,
    /// Number of messages remaining (preserved).
    pub messages_remaining: usize,
    /// The generated summary.
    pub summary: String,
}

#[derive(serde::Serialize)]
struct SystemReminder {
    #[serde(rename = "$text")]
    content: String,
}

#[derive(serde::Serialize)]
struct Plan {
    #[serde(rename = "$text")]
    content: String,
}

#[derive(serde::Serialize)]
struct Todo {
    #[serde(rename = "$text")]
    content: String,
}

/// Which model tier to use for the agent's main reasoning loop.
///
/// This allows creating agents that use different capability levels:
/// - Main agent: typically uses `Advanced`
/// - Explore subagent: uses `Balanced` (cheaper, still capable)
/// - Quick tasks: uses `Fast`
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelTier {
    /// Use the most capable model (default for main agent).
    #[default]
    Advanced,
    /// Use the balanced model (good for subagents).
    Balanced,
    /// Use the fast model (for quick, simple tasks).
    Fast,
}

/// An autonomous agent that processes tasks using tiered language models.
///
/// The agent manages conversation context, handles tool execution in a loop,
/// and applies context compression when needed. Unlike the raw `LanguageModel`
/// trait which only emits events, the agent handles the full tool execution
/// cycle.
///
/// # Type Parameters
///
/// - `Advanced`: The primary LLM for main reasoning (most capable)
/// - `Balanced`: LLM for moderate tasks like subagents (defaults to Advanced)
/// - `Fast`: LLM for quick tasks like compaction (defaults to Balanced)
/// - `H`: Composed hooks for customizing behavior (defaults to `()`)
///
/// # Model Tier Selection
///
/// Each agent uses one tier for its main reasoning loop (set via `tier()`):
/// - Main agent: typically `ModelTier::Advanced`
/// - Explore subagent: `ModelTier::Balanced` (cheaper but capable)
/// - Quick tasks: `ModelTier::Fast`
///
/// The `Fast` tier is always used for compaction, regardless of main tier.
///
/// # Example
///
/// ```rust,ignore
/// // Simple usage (all tiers use same model)
/// let agent = Agent::new(claude);
/// let response = agent.query("What is 2+2?").await?;
///
/// // With tiered models
/// let agent = Agent::builder(opus)
///     .balanced_model(sonnet)
///     .fast_model(haiku)
///     .build();
///
/// // Subagent using balanced tier
/// let explore_agent = Agent::builder(opus)
///     .balanced_model(sonnet)
///     .fast_model(haiku)
///     .tier(ModelTier::Balanced)  // Use sonnet for reasoning
///     .build();
/// ```
#[derive(Debug)]
pub struct Agent<Advanced, Balanced = Advanced, Fast = Balanced, H = ()> {
    /// Primary LLM for main reasoning (most capable).
    pub(crate) advanced: Advanced,

    /// Balanced LLM for moderate tasks (subagents).
    pub(crate) balanced: Balanced,

    /// Fast LLM for quick tasks (compaction, ask command).
    pub(crate) fast: Fast,

    /// Which model tier to use for main reasoning.
    pub(crate) tier: ModelTier,

    /// Registered tools.
    pub(crate) tools: AgentTools,

    /// Composed hooks.
    pub(crate) hooks: H,

    /// Agent configuration.
    pub(crate) config: AgentConfig,

    /// Unified context manager.
    pub(crate) context: Context,

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// Cached fast model profile (for compression decisions).
    pub(crate) fast_profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Todo list for tracking long tasks.
    pub(crate) todo_list: Option<TodoList>,

    /// Output store for lazy URL allocation during compression.
    ///
    /// Held so the store outlives the bash tool that writes into it; the agent
    /// itself only needs to keep it alive.
    pub(crate) _output_store: Option<Arc<OutputStore>>,

    /// Receiver for completed background bash tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,
    /// Registry for running background bash tasks.
    pub(crate) job_registry: Option<JobRegistry>,

    /// Optional readable transcript for long-context recovery.
    pub(crate) transcript: Option<Transcript>,

    /// Optional sandbox directory for working-doc supervision (TODO.md/PLAN.md).
    pub(crate) sandbox_dir: Option<PathBuf>,
}

/// A finished tool call: its id, the tool's name, and either its output or the
/// error text it failed with.
type ToolOutcome = (String, String, Result<String, String>);

/// What one provider event asks the run loop to do.
enum TurnStep {
    /// The event only advanced accumulated state; nothing to emit.
    Continue,
    /// Model text: tell the hooks, then emit it.
    EmitText(String),
    /// Emit this event as-is.
    Emit(AgentEvent),
    /// Stop consuming this turn's stream.
    Stop,
}

/// What one turn accumulates from the model's stream.
///
/// Kept outside the `run` stream so the per-tier arms are the same short loop
/// rather than three copies of the same event handling.
#[derive(Default)]
struct Turn {
    text_chunks: Vec<String>,
    tool_calls: Vec<aither_core::llm::ToolCall>,
    /// The model emitted a tool call the provider could not parse.
    malformed_function_call: bool,
    /// The provider failed outright.
    error: Option<String>,
}

impl Turn {
    /// Folds one provider event into the turn.
    fn apply<E: core::fmt::Display>(&mut self, event: Result<Event, E>) -> TurnStep {
        let event = match event {
            Ok(event) => event,
            Err(e) => {
                let error_msg = e.to_string();
                // A malformed call is worth another attempt; anything else is
                // the provider telling us the turn is over.
                if error_msg.contains("malformed function call") {
                    tracing::warn!("Model generated malformed function call, retrying...");
                    self.malformed_function_call = true;
                } else {
                    self.error = Some(error_msg);
                }
                return TurnStep::Stop;
            }
        };

        match event {
            Event::Text(text) => {
                self.text_chunks.push(text.clone());
                TurnStep::EmitText(text)
            }
            Event::Reasoning(r) => TurnStep::Emit(AgentEvent::Reasoning(r)),
            Event::ToolCall(call) => {
                self.tool_calls.push(call);
                TurnStep::Continue
            }
            // Built-in results are the provider's own tools reporting back, so
            // they read as model output rather than as a tool call of ours.
            Event::BuiltInToolResult { tool, result } => {
                let formatted = format!("[{tool}] {result}");
                self.text_chunks.push(formatted.clone());
                TurnStep::Emit(AgentEvent::Text(formatted))
            }
            Event::Usage(u) => TurnStep::Emit(AgentEvent::Usage(u)),
        }
    }
}

/// How long to wait for outstanding background bash tasks before giving up.
const MAX_WAIT: Duration = Duration::from_secs(300);
/// How often to check whether a background task has finished.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
    /// Creates a new agent with default configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    #[must_use]
    pub fn new(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    /// Creates a new agent with the specified configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    #[must_use]
    pub fn with_config(llm: LLM, config: AgentConfig) -> Self {
        Self {
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config,
            context: Context::default(),
            profile: None,
            fast_profile: None,
            initialized: false,
            todo_list: None,
            _output_store: None,
            background_receiver: None,
            job_registry: None,
            transcript: None,
            sandbox_dir: None,
        }
    }
}

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
    /// Returns a builder for more complex agent construction.
    pub fn builder(llm: LLM) -> crate::builder::AgentBuilder<LLM, LLM, LLM, ()> {
        crate::builder::AgentBuilder::new(llm)
    }
}

impl<Advanced, Balanced, Fast, H> Agent<Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    /// Performs a one-shot query and returns the final response.
    ///
    /// This is the simplest way to use the agent. The agent handles tool
    /// execution internally and returns the final text response.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM returns an error
    /// - A hook aborts the operation
    /// - Tool execution fails
    pub async fn query(&mut self, prompt: &str) -> Result<String, AgentError> {
        use futures_lite::StreamExt;

        let stream = self.run(prompt, std::iter::empty());
        futures_lite::pin!(stream);

        let mut final_text = String::new();
        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::Text(chunk) => final_text.push_str(&chunk),
                AgentEvent::Complete {
                    final_text: text, ..
                } => {
                    return Ok(text);
                }
                _ => {}
            }
        }
        Ok(final_text)
    }

    /// Runs the agent with streaming events.
    ///
    /// Returns a stream of `AgentEvent`s that can be consumed to observe
    /// the agent's progress in real-time. Text chunks are yielded as they
    /// arrive from the LLM for true streaming display.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = agent.run("Implement the feature", std::iter::empty());
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         AgentEvent::Text(t) => print!("{}", t),
    ///         AgentEvent::Complete { .. } => break,
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn run(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = url::Url>,
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        let prompt = prompt.to_string();
        let attachments: Vec<url::Url> = attachments.into_iter().collect();

        async_stream::try_stream! {
            self.ensure_initialized().await;

            // Apply context compression if needed
            self.maybe_compress().await?;

            // Add user message with attachments
            let user_msg = Message::user(&prompt).with_attachments(attachments);
            self.context.push(user_msg);
            if let Some(transcript) = &self.transcript {
                transcript.write_user_message(&prompt).await;
            }

            // Run the tool loop
            let mut iteration = 0;

            let final_text = loop {
                iteration += 1;
                if iteration > self.config.max_iterations {
                    Err(AgentError::MaxIterations {
                        limit: self.config.max_iterations,
                    })?;
                }

                let messages = self.build_request_messages().await;
                let tool_defs = self.tools.active_definitions();
                let request = LLMRequest::new(messages).with_tool_definitions(tool_defs);

                let mut stream = self.respond_on_tier(request);
                let mut turn = Turn::default();
                while let Some(event) = stream.next().await {
                    match turn.apply(event) {
                        TurnStep::Continue => {}
                        TurnStep::EmitText(text) => {
                            self.hooks.on_text(&text).await;
                            yield AgentEvent::Text(text);
                        }
                        TurnStep::Emit(event) => yield event,
                        TurnStep::Stop => break,
                    }
                }
                drop(stream);

                if let Some(e) = turn.error {
                    Err(AgentError::Llm(e))?;
                }

                // A malformed call is the model's mistake, not the caller's:
                // spend another iteration rather than fail the run.
                if turn.malformed_function_call {
                    continue;
                }

                let response_text = turn.text_chunks.join("");

                // No tool calls means the model is done, unless working-doc
                // supervision says there is more to do.
                if turn.tool_calls.is_empty() {
                    if !response_text.is_empty() {
                        self.context.push(Message::assistant(&response_text));
                        if let Some(transcript) = &self.transcript {
                            transcript.write_assistant_text(&response_text).await;
                        }
                    }
                    if self.inject_working_doc_continue_reminder().await {
                        continue;
                    }
                    break response_text;
                }

                self.context.push(Message::assistant_with_tool_calls(
                    &response_text,
                    turn.tool_calls.clone(),
                ));

                for event in self.run_tool_calls(&turn.tool_calls, iteration).await? {
                    yield event;
                }
            };

            // Handle background tasks before completing
            if self.drain_background_tasks().await {
                // Continue processing with background results
                let continuation = self.continue_after_background_streaming().await;
                for event in continuation {
                    yield event?;
                }
                return;
            }

            // Notify hooks
            let stop_ctx = StopContext {
                final_text: &final_text,
                turns: iteration,
                reason: StopReason::Complete,
            };

            if let Some(reason) = self.hooks.on_stop(&stop_ctx).await {
                Err(AgentError::HookRejected {
                    hook: "on_stop",
                    reason,
                })?;
            }

            // Yield completion event
            yield AgentEvent::Complete {
                final_text,
                turns: iteration,
            };
        }
    }

    /// Streams a response from whichever tier is active.
    ///
    /// The tier picks the model at run time, so the three streams are boxed to
    /// a common type rather than driving three copies of the caller's loop.
    /// Their errors only ever get displayed, so they collapse to a string at
    /// the same boundary.
    fn respond_on_tier(
        &self,
        request: LLMRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, String>> + Send + '_>> {
        match self.tier {
            ModelTier::Advanced => Box::pin(
                self.advanced
                    .respond(request)
                    .map(|event| event.map_err(|e| e.to_string())),
            ),
            ModelTier::Balanced => Box::pin(
                self.balanced
                    .respond(request)
                    .map(|event| event.map_err(|e| e.to_string())),
            ),
            ModelTier::Fast => Box::pin(
                self.fast
                    .respond(request)
                    .map(|event| event.map_err(|e| e.to_string())),
            ),
        }
    }

    /// Runs one turn's tool calls and returns the events the caller should emit.
    ///
    /// Every call runs concurrently; results are folded back into the context in
    /// call order so the transcript stays deterministic regardless of which
    /// finished first.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::HookRejected`] if a hook aborts the turn.
    async fn run_tool_calls(
        &mut self,
        tool_calls: &[aither_core::llm::ToolCall],
        iteration: usize,
    ) -> Result<Vec<AgentEvent>, AgentError> {
        // Snapshot todo state BEFORE executing tool calls
        let old_todo_items: Vec<TodoItem> = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();
        let todo_tool_called = tool_calls.iter().any(|call| call.name == "todo");

        let mut events = Vec::with_capacity(tool_calls.len() * 2);
        for call in tool_calls {
            let args = call.arguments.to_string();
            if let Some(transcript) = &self.transcript {
                transcript.write_tool_call(&call.name, &args).await;
            }
            events.push(AgentEvent::ToolCallStart {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: args,
            });
        }

        let results = self.dispatch_tool_calls(tool_calls, iteration).await;

        let mut has_tool_error = false;
        for result in results {
            let (call_id, call_name, tool_result) = result?;

            if let Some(transcript) = &self.transcript {
                transcript.write_tool_result(&call_name, &tool_result).await;
            }
            events.push(AgentEvent::ToolCallEnd {
                id: call_id.clone(),
                name: call_name.clone(),
                result: tool_result.clone(),
            });

            let content = match &tool_result {
                Ok(content) | Err(content) => content,
            };
            if Self::is_tool_failure(&tool_result, content) {
                has_tool_error = true;
            }

            self.context.push(Message::tool(
                &call_id,
                Self::process_reload_marker(content),
            ));
            if call_name == "bash"
                && tool_result.is_ok()
                && let Some(reminder) = Self::format_background_started_reminder(content)
            {
                self.context.push(Message::system(reminder));
            }
        }

        if has_tool_error {
            self.context.push(Message::system(include_str!(
                "prompts/tool_error_reminder.txt"
            )));
        }
        if todo_tool_called {
            self.inject_todo_reminder(&old_todo_items);
        }

        // Poll for completed background tasks
        if let Some(ref receiver) = self.background_receiver {
            for task in receiver.take_completed() {
                tracing::info!(task_id = %task.task_id, "background task completed");
                let result_msg = Self::format_background_task_result(&task);
                self.context.push(Message::system(&result_msg));
            }
        }

        Ok(events)
    }

    /// Runs every tool call concurrently, applying the hooks around each one.
    async fn dispatch_tool_calls(
        &self,
        tool_calls: &[aither_core::llm::ToolCall],
        iteration: usize,
    ) -> Vec<Result<ToolOutcome, AgentError>> {
        let tools = &self.tools;
        let hooks = &self.hooks;
        let message_count = self.context.len_recent();

        let tool_futures = tool_calls.iter().map(|call| {
            let args_json = call.arguments.to_string();

            async move {
                let tool_ctx = ToolUseContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    turn: iteration,
                    message_count,
                };

                let (result, duration) = match hooks.pre_tool_use(&tool_ctx).await {
                    PreToolAction::Abort(reason) => {
                        return Err(AgentError::HookRejected {
                            hook: "pre_tool_use",
                            reason,
                        });
                    }
                    PreToolAction::Deny(reason) => (Err(anyhow::anyhow!(reason)), Duration::ZERO),
                    PreToolAction::Allow => {
                        let start = Instant::now();
                        let result = tools.call(&call.name, &args_json).await;
                        let result = result.map(|output| output.as_str().unwrap_or("").to_string());
                        (result, start.elapsed())
                    }
                };

                let result_ref = result
                    .as_ref()
                    .map(String::as_str)
                    .map_err(ToString::to_string);
                let result_ctx = ToolResultContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    result: result_ref.as_ref().map(|s| *s).map_err(String::as_str),
                    duration,
                };

                let tool_result = match hooks.post_tool_use(&result_ctx).await {
                    PostToolAction::Abort(reason) => {
                        return Err(AgentError::HookRejected {
                            hook: "post_tool_use",
                            reason,
                        });
                    }
                    PostToolAction::Replace(replacement) => {
                        if result.is_ok() {
                            Ok(replacement)
                        } else {
                            Err(replacement)
                        }
                    }
                    PostToolAction::Keep => result.map_err(|e| format!("Error: {e}")),
                };

                Ok((call.id.clone(), call.name.clone(), tool_result))
            }
        });

        futures::future::join_all(tool_futures).await
    }

    /// Whether a tool result should count as a failure worth reminding about.
    ///
    /// Some tools report their own misuse in an `Ok` string rather than an
    /// error, so the text is checked as well as the result.
    fn is_tool_failure(tool_result: &Result<String, String>, content: &str) -> bool {
        tool_result.is_err()
            || content.contains("ssh_server_id is required")
            || content.contains("unknown ssh_server_id")
            || content.contains("not found")
            || content.contains("Invalid arguments")
    }

    /// Reminds the model what to do next after it edited the todo list.
    ///
    /// Finishing a task earns a pointer at the next one; anything else gets the
    /// list back, so the model always sees the state it just wrote.
    fn inject_todo_reminder(&mut self, old_todo_items: &[TodoItem]) {
        let new_items = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();

        let newly_completed = new_items.iter().find(|new_item| {
            new_item.status == TodoStatus::Completed
                && old_todo_items.iter().any(|old| {
                    old.content == new_item.content && old.status != TodoStatus::Completed
                })
        });

        let reminder = newly_completed.map_or_else(
            || self.format_todo_reminder(),
            |completed| self.format_next_task_reminder(&completed.content),
        );
        if let Some(reminder) = reminder {
            self.context.push(Message::system(&reminder));
        }
    }

    /// Collects background tasks that finished during the run, waiting briefly
    /// for any still running.
    ///
    /// Returns whether anything completed, which tells the caller whether the
    /// model needs another turn to react to the results.
    async fn drain_background_tasks(&mut self) -> bool {
        let Some(receiver) = self.background_receiver.clone() else {
            return false;
        };

        let completed_tasks = receiver.take_completed();
        let mut had_completed = !completed_tasks.is_empty();
        for task in completed_tasks {
            tracing::info!(task_id = %task.task_id, "background task completed (final check)");
            let result_msg = Self::format_background_task_result(&task);
            self.context.push(Message::system(&result_msg));
        }

        let start = Instant::now();
        while start.elapsed() < MAX_WAIT {
            if let Some(task) = receiver.recv_timeout(POLL_INTERVAL).await {
                tracing::info!(task_id = %task.task_id, "background task completed (waiting)");
                let result_msg = Self::format_background_task_result(&task);
                self.context.push(Message::system(&result_msg));
                had_completed = true;
            } else if !receiver.has_running() {
                break;
            }
        }

        had_completed
    }

    /// Registers a tool for the agent to use.
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError`](aither_core::llm::tool::RegisterError) if the
    /// name collides with a registered tool, or the tool has no description.
    pub fn register_tool<T: aither_core::llm::Tool + 'static>(
        &mut self,
        tool: T,
    ) -> Result<(), aither_core::llm::tool::RegisterError> {
        self.tools.register(tool)
    }

    /// Returns a reference to the unified context manager.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns a mutable reference to the unified context manager.
    ///
    /// Use this to insert/update system blocks, push messages, etc.
    pub const fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Adds a message to the conversation history.
    pub fn push_message(&mut self, message: Message) {
        self.context.push(message);
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.context.clear_history();
    }

    /// Compacts the conversation by generating a structured handoff and starting fresh.
    ///
    /// # Errors
    ///
    /// Returns an error if handoff generation fails.
    pub async fn compact(
        &mut self,
        focus: Option<&str>,
    ) -> Result<Option<CompactResult>, AgentError> {
        self.ensure_initialized().await;

        let messages = self.context.conversation_messages();
        if messages.is_empty() {
            return Ok(None);
        }

        let messages_compacted = messages.len();
        let summary = self.generate_handoff_summary(focus).await?;

        if let Some(transcript) = &self.transcript {
            transcript.write_compact_marker().await;
        }

        self.context.clear_recent();
        self.context.clear_reminders();
        self.context.set_handoff(summary.clone());

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: 0,
            summary,
        }))
    }

    /// Generates a structured handoff summary using the current tier model.
    async fn generate_handoff_summary(&self, focus: Option<&str>) -> Result<String, AgentError> {
        let focus_instruction = focus.map(str::trim).filter(|s| !s.is_empty()).map_or_else(
            || "No additional focus hint was provided.".to_string(),
            |f| format!("Focus the handoff on: {f}"),
        );

        let transcript_path = self
            .transcript
            .as_ref()
            .map(|t| t.path().display().to_string())
            .or_else(|| self.config.transcript_path.clone())
            .unwrap_or_else(|| "transcript.md".to_string());

        // The braces below are template placeholders for `str::replace`, not
        // format arguments.
        #[allow(clippy::literal_string_with_formatting_args)]
        let handoff_prompt = include_str!("prompts/compact_handoff.txt")
            .replace("{focus_instruction}", &focus_instruction)
            .replace("{transcript_path}", &transcript_path);

        let mut messages = self.context.conversation_messages();
        messages.push(Message::user(handoff_prompt));

        let mut chunks = Vec::new();
        match self.tier {
            ModelTier::Advanced => {
                let stream = self.advanced.respond(LLMRequest::new(messages.clone()));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format!("[{tool}] {result}"));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
            ModelTier::Balanced => {
                let stream = self.balanced.respond(LLMRequest::new(messages.clone()));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format!("[{tool}] {result}"));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
            ModelTier::Fast => {
                let stream = self.fast.respond(LLMRequest::new(messages));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format!("[{tool}] {result}"));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
        }

        let summary = chunks.join("").trim().to_string();
        if summary.is_empty() {
            return Err(AgentError::Llm(
                "Compaction failed to generate handoff summary".to_string(),
            ));
        }
        Ok(summary)
    }

    /// Injects a continuation reminder when working documents still have pending tasks.
    async fn inject_working_doc_continue_reminder(&mut self) -> bool {
        let Some(sandbox_dir) = self.sandbox_dir.as_deref() else {
            return false;
        };

        let docs = working_docs::read_snapshot(sandbox_dir).await;
        if !docs.has_unchecked_items() {
            return false;
        }

        self.context.push(Message::system(
            "<system-reminder>TODO.md or PLAN.md still has unchecked items. Continue working through the checklist. If user input is required, call ask_user and then proceed.</system-reminder>",
        ));
        true
    }

    /// Returns the current conversation history.
    #[must_use]
    pub fn history(&self) -> Vec<Message> {
        self.context.conversation_messages()
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Ensures the agent is initialized (profiles fetched, static blocks set up).
    async fn ensure_initialized(&mut self) {
        if self.initialized {
            return;
        }

        // Fetch profile for the selected tier (for context window decisions)
        self.profile = Some(match self.tier {
            ModelTier::Advanced => self.advanced.profile().await,
            ModelTier::Balanced => self.balanced.profile().await,
            ModelTier::Fast => self.fast.profile().await,
        });

        // Fetch fast model profile (for compression decisions)
        // We always need this because compression uses the fast model
        self.fast_profile = Some(self.fast.profile().await);

        // Populate static system blocks in Context from AgentConfig.
        // These form the stable, cacheable prefix.
        self.populate_system_blocks();

        self.initialized = true;
    }

    /// Populates the Context's system blocks from `AgentConfig`.
    ///
    /// Called once during initialization. These blocks form the stable
    /// cacheable prefix of the system message.
    fn populate_system_blocks(&mut self) {
        if let Some(ref system_prompt) = self.config.system_prompt {
            self.context
                .insert_system_named("base_system", system_prompt);
        }

        if let Some(ref persona_prompt) = self.config.persona_prompt {
            self.context.insert_system_named("persona", persona_prompt);
        }

        if self.config.agent_kind == AgentKind::Coding {
            self.context.insert_system_named(
                "workspace_facts",
                "When discovering workspace guidance, load AGENT.md first. If AGENT.md is missing, load CLAUDE.md. Treat these files as repository policy for coding tasks; this behavior is not required for chatbot-style sessions.",
            );
        }

        self.context.insert_system_named(
            "knowledge_and_time",
            include_str!("prompts/knowledge_and_time.txt"),
        );

        self.context
            .insert_system_named("permissions", include_str!("prompts/permissions.txt"));

        if let Some(ref path) = self.config.transcript_path {
            self.context.insert_system_named(
                "transcript_memory",
                format!(
                    "Compressed memory may only keep a summary, but the full transcript remains available at {path}. If details are missing, recover them by searching/reading transcript content before making irreversible changes."
                ),
            );
        }

        let tool_hints = self.format_tool_hints_block();
        if !tool_hints.is_empty() {
            self.context.insert_system_named("tool_hints", &tool_hints);
        }
    }

    /// Builds the message list for an LLM request.
    ///
    /// Uses `context.build_messages()` for the stable system prefix + conversation,
    /// then prepends per-turn ephemeral context (todo, working docs, background
    /// jobs, context usage) as system messages inserted before conversation messages.
    async fn build_request_messages(&mut self) -> Vec<Message> {
        self.context.clear_reminders();

        if let Some(todo_ctx) = self.format_todo_context() {
            self.context
                .insert_reminder(&SystemReminder { content: todo_ctx });
        }

        if let Some(sandbox_dir) = self.sandbox_dir.as_deref() {
            let docs = working_docs::read_snapshot(sandbox_dir).await;
            if let Some(plan_md) = docs.plan_md {
                self.context.insert_reminder(&Plan { content: plan_md });
            }
            if let Some(todo_md) = docs.todo_md {
                self.context.insert_reminder(&Todo { content: todo_md });
            }
        }

        if let Some(job_registry) = &self.job_registry {
            let running = job_registry.format_running_jobs().await;
            if !running.is_empty() {
                self.context.insert_reminder(&SystemReminder {
                    content: format!(
                        "Running background terminals:\n{running}Read redirected output files via bash (head/tail/grep/cat). Use input_terminal for stdin and kill_terminal to stop when needed."
                    ),
                });
            }
        }

        // Context usage estimation uses conversation messages
        let usage = self.estimate_usage_for_messages(self.context.recent());

        if let Some(handoff_ctx) = self.format_handoff_context(usage) {
            self.context.insert_reminder(&SystemReminder {
                content: handoff_ctx,
            });
        }

        self.context.build_messages()
    }

    /// Estimates current context usage using the active and fast model windows.
    fn estimate_usage_for_messages(&self, messages: &[Message]) -> f32 {
        let tier_context = self
            .profile
            .as_ref()
            .map_or(100_000, |p| p.context_length as usize);

        let fast_context = self
            .fast_profile
            .as_ref()
            .map_or(100_000, |p| p.context_length as usize);

        let context_length = tier_context.min(fast_context);
        estimate_context_usage(messages, context_length)
    }

    /// Returns the exact request message sequence that would be sent for the next turn.
    ///
    /// This is intended for observability/debug UIs and does not mutate agent memory.
    /// Temporarily adds the prompt to a forked context, builds the messages, then discards.
    pub async fn preview_request_messages(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = url::Url>,
    ) -> Vec<Message> {
        self.ensure_initialized().await;
        // Fork context, add the user message, build messages
        let checkpoint = self.context.checkpoint();
        self.context
            .push(Message::user(prompt).with_attachments(attachments));
        let messages = self.build_request_messages().await;
        self.context.restore(checkpoint);
        messages
    }

    /// Injects a handoff instruction when context usage approaches the threshold.
    fn format_handoff_context(&self, usage: f32) -> Option<String> {
        if usage < self.config.context_assembler.handoff_threshold {
            return None;
        }

        let mut note = String::new();
        note.push_str(&self.config.context_assembler.handoff_instruction);
        if let Some(path) = &self.config.transcript_path {
            let _ = write!(note, " Transcript source: {path}.");
        }
        Some(note)
    }

    fn format_tool_hints_block(&self) -> String {
        let defs = self.tools.active_definitions();
        let mut lines = Vec::new();

        for def in defs {
            let desc = first_paragraph(def.description());
            if desc.is_empty() {
                continue;
            }
            lines.push(format!("{}: {}", def.name(), desc));
        }

        lines.join("\n")
    }

    /// Continues agent processing after background tasks complete (streaming version).
    async fn continue_after_background_streaming(&mut self) -> Vec<Result<AgentEvent, AgentError>> {
        let mut events = Vec::new();
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                events.push(Err(AgentError::MaxIterations {
                    limit: self.config.max_iterations,
                }));
                return events;
            }

            let messages = self.build_request_messages().await;
            let tool_defs = self.tools.active_definitions();
            let request = LLMRequest::new(messages).with_tool_definitions(tool_defs);

            let mut stream = self.respond_on_tier(request);
            let mut turn = Turn::default();
            while let Some(event) = stream.next().await {
                match turn.apply(event) {
                    TurnStep::Continue => {}
                    TurnStep::EmitText(text) => {
                        self.hooks.on_text(&text).await;
                        events.push(Ok(AgentEvent::Text(text)));
                    }
                    TurnStep::Emit(event) => events.push(Ok(event)),
                    TurnStep::Stop => break,
                }
            }
            drop(stream);

            // As in the main loop, a malformed call is worth another attempt.
            if turn.malformed_function_call {
                continue;
            }

            if let Some(e) = turn.error {
                events.push(Err(AgentError::Llm(e)));
                return events;
            }

            let response_text = turn.text_chunks.join("");

            if turn.tool_calls.is_empty() {
                if !response_text.is_empty() {
                    self.context.push(Message::assistant(&response_text));
                }
                events.push(Ok(AgentEvent::Complete {
                    final_text: response_text,
                    turns: iteration,
                }));
                return events;
            }

            self.context.push(Message::assistant_with_tool_calls(
                &response_text,
                turn.tool_calls.clone(),
            ));

            // Execute tool calls
            let tools = &self.tools;
            let tool_futures = turn.tool_calls.iter().map(|call| {
                let args_json = call.arguments.to_string();
                async move {
                    let result = tools
                        .call(&call.name, &args_json)
                        .await
                        .map(|output| output.as_str().unwrap_or("").to_string())
                        .map_err(|e| format!("Error: {e}"));
                    (call.id.clone(), call.name.clone(), result)
                }
            });

            let results: Vec<(String, String, Result<String, String>)> =
                futures::future::join_all(tool_futures).await;

            for (call_id, call_name, tool_result) in results {
                let is_bash_call = call_name == "bash";
                events.push(Ok(AgentEvent::ToolCallEnd {
                    id: call_id.clone(),
                    name: call_name,
                    result: tool_result.clone(),
                }));
                let content = match &tool_result {
                    Ok(content) => content,
                    Err(error) => error,
                };
                let processed_content = Self::process_reload_marker(content);
                self.context
                    .push(Message::tool(&call_id, processed_content));
                if is_bash_call
                    && tool_result.is_ok()
                    && let Some(reminder) = Self::format_background_started_reminder(content)
                {
                    self.context.push(Message::system(reminder));
                }
            }

            if let Some(ref receiver) = self.background_receiver {
                let completed_tasks = receiver.take_completed();
                for task in completed_tasks {
                    let result_msg = Self::format_background_task_result(&task);
                    self.context.push(Message::system(&result_msg));
                }
            }
        }
    }

    /// Compresses context if needed.
    ///
    /// Considers BOTH the selected tier's context window AND the fast model's
    /// context window. Compression is triggered based on the most constrained
    /// of the two windows, ensuring the fast LLM can actually see the content
    /// during compaction.
    ///
    /// Uses `effective_trigger()` which accounts for a 20% context reservation
    /// to ensure there's room for the fast LLM during compaction.
    ///
    /// When an `OutputStore` is available, uses lazy URL allocation:
    /// 1. Allocates URLs for large tool outputs before compression
    /// 2. Lets the fast LLM decide which URLs to reference in the summary
    /// 3. Only writes files for URLs actually referenced in the summary
    async fn maybe_compress(&mut self) -> Result<(), AgentError> {
        // Use the minimum of both context windows (most constrained)
        // This ensures:
        // 1. The selected tier doesn't run out of context for reasoning
        // 2. The fast model can see all content during compression
        let tier_context = self
            .profile
            .as_ref()
            .map_or(100_000, |p| p.context_length as usize);

        let fast_context = self
            .fast_profile
            .as_ref()
            .map_or(100_000, |p| p.context_length as usize);

        let context_length = tier_context.min(fast_context);
        let usage = estimate_context_usage(&self.context.conversation_messages(), context_length);

        match &self.config.context {
            ContextStrategy::Unlimited => Ok(()),
            ContextStrategy::Smart(config) => {
                // Use effective_trigger which reserves context for compaction process.
                if usage >= config.effective_trigger() {
                    let preserve_recent = config.preserve_recent;
                    if self.context.len_recent() > preserve_recent {
                        let _ = self.compact(None).await?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Processes a tool result (currently passthrough).
    ///
    /// Previously handled reload markers, now just returns the content as-is.
    fn process_reload_marker(result: &str) -> String {
        result.to_string()
    }

    /// Formats the todo list as a system reminder.
    ///
    /// Returns None if there's no todo list or it's empty.
    fn format_todo_reminder(&self) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();
        if items.is_empty() {
            return None;
        }
        let items_json = format_todo_items_json(&items);
        Some(format!(
            "<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n{items_json}. Continue on with the tasks at hand if applicable.\n</system-reminder>"
        ))
    }

    /// Formats the current todo list for context injection before each request.
    fn format_todo_context(&self) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();
        if items.is_empty() {
            return None;
        }

        let items_json = format_todo_items_json(&items);
        Some(format!(
            "Current todo list (do not mention this explicitly to the user):\n\n{items_json}"
        ))
    }

    /// Formats a reminder when `bash` has been auto-promoted to background.
    fn format_background_started_reminder(tool_content: &str) -> Option<String> {
        let payload: serde_json::Value = serde_json::from_str(tool_content).ok()?;
        let status = payload.get("status")?.as_str()?;
        if status != "running" {
            return None;
        }

        let task_id = payload.get("task_id")?.as_str()?.trim();
        if task_id.is_empty() {
            return None;
        }

        let output_preview = payload
            .get("stdout")
            .and_then(|stdout| stdout.get("content"))
            .and_then(|content| content.get("text"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("(no output yet)");

        let output_file = payload
            .get("stdout")
            .and_then(|stdout| stdout.get("url"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(missing output file)");

        Some(format!(
            "<system-reminder>\nA bash command is running in background (task_id={task_id}).\nCurrent output snapshot (first max_lines):\n{output_preview}\nFull redirected output file: {output_file}\nRead the file via bash when needed. If the command waits for stdin, use input_terminal. Use kill_terminal to stop it.\n</system-reminder>"
        ))
    }

    /// Formats a completed background task result as a system message.
    fn format_background_task_result(task: &aither_sandbox::CompletedTask) -> String {
        let mut msg = format!(
            "<background-bash-result task_id=\"{}\">\nScript: {}\n",
            task.task_id,
            truncate_script(&task.script, 100)
        );

        match &task.result {
            Ok(result) => {
                let _ = writeln!(msg, "Exit code: {}", result.exit_code);
                // Format stdout
                let stdout_str = result.stdout.to_string();
                if !stdout_str.is_empty() {
                    msg.push_str("Output:\n");
                    msg.push_str(&stdout_str);
                    if !stdout_str.ends_with('\n') {
                        msg.push('\n');
                    }
                }
                // Format stderr if present
                if let Some(ref stderr) = result.stderr {
                    let stderr_str = stderr.to_string();
                    if !stderr_str.is_empty() {
                        msg.push_str("Stderr:\n");
                        msg.push_str(&stderr_str);
                        if !stderr_str.ends_with('\n') {
                            msg.push('\n');
                        }
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(msg, "Error: {e}");
            }
        }

        msg.push_str("</background-bash-result>");
        msg
    }

    /// Generates a reminder about the next task after a task was completed.
    fn format_next_task_reminder(&self, completed_task: &str) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();

        // Find the next pending or in_progress task
        let next_task = items
            .iter()
            .find(|item| matches!(item.status, TodoStatus::Pending | TodoStatus::InProgress));

        next_task.map_or_else(
            || {
                items.iter().all(|i| i.status == TodoStatus::Completed).then(|| {
                    format!(
                        "<system-reminder>\nTask \"{completed_task}\" completed. All tasks in the todo list are now complete!\n</system-reminder>"
                    )
                })
            },
            |task| {
                Some(format!(
                    "<system-reminder>\nTask \"{}\" completed. Next task: {} ({})\n</system-reminder>",
                    completed_task, task.content, task.active_form
                ))
            },
        )
    }
}

/// Formats todo items into the JSON-ish list used in system reminders.
fn format_todo_items_json(items: &[TodoItem]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_string())
}

fn first_paragraph(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if let Some((first, _)) = trimmed.split_once("\n\n") {
        return first.split_whitespace().collect::<Vec<_>>().join(" ");
    }

    trimmed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Truncates a script for display in messages.
fn truncate_script(script: &str, max_chars: usize) -> &str {
    let script = script.trim();
    // Find byte index at max_chars boundary
    match script.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &script[..byte_idx],
        None => script, // String is shorter than max_chars
    }
}
