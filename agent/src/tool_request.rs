//! Generic request/response channel helpers for UI-mediated tools.

use std::sync::atomic::{AtomicU64, Ordering};

use async_channel::{Receiver, Sender};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;

const ERR_TOOL_REQUEST_UNAVAILABLE: &str = include_str!("texts/tool_request_unavailable.txt");
const ERR_TOOL_REQUEST_CANCELLED: &str = include_str!("texts/tool_request_cancelled.txt");

/// A UI-bound request carrying tool arguments and a response sender.
#[derive(Debug)]
pub struct ToolRequest<Args, Response> {
    /// Tool arguments provided by the agent.
    pub args: Args,
    response_tx: Sender<Response>,
}

impl<Args, Response> ToolRequest<Args, Response> {
    /// Create a new request with its response channel.
    #[must_use]
    pub const fn new(args: Args, response_tx: Sender<Response>) -> Self {
        Self { args, response_tx }
    }

    /// Respond to the request without blocking the UI thread.
    pub fn respond(self, response: Response) -> Result<(), async_channel::TrySendError<Response>> {
        self.response_tx.try_send(response)
    }
}

/// Request broker for UI-mediated tools.
#[derive(Debug, Clone)]
pub struct ToolRequestBroker<Args, Response> {
    tx: Sender<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestBroker<Args, Response> {
    /// Send a request and await the response.
    pub async fn request(&self, args: Args) -> anyhow::Result<Response> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.tx
            .send(ToolRequest::new(args, response_tx))
            .await
            .map_err(|_| anyhow::anyhow!(ERR_TOOL_REQUEST_UNAVAILABLE.trim()))?;

        response_rx
            .recv()
            .await
            .map_err(|_| anyhow::anyhow!(ERR_TOOL_REQUEST_CANCELLED.trim()))
    }
}

/// Queue for pending tool requests.
#[derive(Debug)]
pub struct ToolRequestQueue<Args, Response> {
    rx: Receiver<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestQueue<Args, Response> {
    /// Await the next request.
    pub async fn next(&self) -> Option<ToolRequest<Args, Response>> {
        self.rx.recv().await.ok()
    }

    /// Returns true if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Create a new request broker/queue pair.
#[must_use]
pub fn channel<Args, Response>() -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    let (tx, rx) = async_channel::unbounded();
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}

/// Create a bounded request broker/queue pair with explicit backpressure.
#[must_use]
pub fn bounded_channel<Args, Response>(
    capacity: usize,
) -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    assert!(capacity > 0, "tool request queue capacity must be > 0");
    let (tx, rx) = async_channel::bounded(capacity);
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}

// ── RequestApprover ─────────────────────────────────────────

struct PendingEntry<P, R> {
    order: u64,
    payload: P,
    tx: Sender<R>,
}

/// An ID-tracked, multi-request approval queue for server contexts.
///
/// Each request is assigned a unique ID, stored in FIFO order, and awaits
/// a response from the UI/caller side. Designed for web-server polling patterns
/// where the UI fetches pending requests and responds asynchronously.
///
/// Use [`event_listener::EventListener`] from [`listen`](Self::listen) to
/// efficiently wait for state changes without polling.
///
/// # Type Parameters
///
/// - `P`: Payload type (cloneable request data visible to the UI).
/// - `R`: Response type sent back to the requester.
pub struct RequestApprover<P, R> {
    order: SkipMap<u64, String>,
    pending: DashMap<String, PendingEntry<P, R>>,
    event: event_listener::Event,
    next_id: AtomicU64,
}

impl<P, R> std::fmt::Debug for RequestApprover<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending_count = self.pending.len();
        f.debug_struct("RequestApprover")
            .field("pending_count", &pending_count)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<P: Clone, R> Default for RequestApprover<P, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone, R> RequestApprover<P, R> {
    /// Create a new empty approver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            order: SkipMap::new(),
            pending: DashMap::new(),
            event: event_listener::Event::new(),
            next_id: AtomicU64::new(0),
        }
    }

    /// Enqueue a request and return its ID and a receiver for the response.
    ///
    /// The returned [`Receiver`] yields the response once [`respond`](Self::respond)
    /// is called with the matching ID.
    pub fn enqueue(&self, payload: P) -> (String, Receiver<R>) {
        let order = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = order.to_string();
        let (tx, rx) = async_channel::bounded(1);
        self.pending
            .insert(id.clone(), PendingEntry { order, payload, tx });
        self.order.insert(order, id.clone());
        self.event.notify(usize::MAX);
        (id, rx)
    }

    /// Respond to a pending request by ID.
    ///
    /// Returns `true` if the request was found and the response was sent.
    pub fn respond(&self, id: &str, response: R) -> bool {
        let found = if let Some((_, entry)) = self.pending.remove(id) {
            self.order.remove(&entry.order);
            entry.tx.try_send(response).is_ok()
        } else {
            false
        };
        self.event.notify(usize::MAX);
        found
    }

    /// Return the oldest pending request (ID + cloned payload) without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<(String, P)> {
        loop {
            let front = self.order.front()?;
            let order = *front.key();
            let id = front.value().clone();
            drop(front);
            if let Some(entry) = self.pending.get(id.as_str()) {
                return Some((id, entry.payload.clone()));
            }
            self.order.remove(&order);
        }
    }

    /// Return the oldest pending request matching a predicate.
    #[must_use]
    pub fn peek_filtered(&self, predicate: impl Fn(&P) -> bool) -> Option<(String, P)> {
        for item in self.order.iter() {
            let order = *item.key();
            let id = item.value().clone();
            let Some(entry) = self.pending.get(id.as_str()) else {
                self.order.remove(&order);
                continue;
            };
            if predicate(&entry.payload) {
                return Some((id, entry.payload.clone()));
            }
        }
        None
    }

    /// Get a cloned payload by request ID.
    #[must_use]
    pub fn get_payload(&self, id: &str) -> Option<P> {
        self.pending.get(id).map(|entry| entry.payload.clone())
    }

    /// Returns `true` if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Create a listener that resolves when the approver state changes.
    ///
    /// Use this for efficient async polling:
    /// ```rust,ignore
    /// loop {
    ///     if let Some((id, payload)) = approver.peek() {
    ///         // handle request
    ///     }
    ///     approver.listen().await;
    /// }
    /// ```
    pub fn listen(&self) -> event_listener::EventListener {
        self.event.listen()
    }
}

#[cfg(test)]
mod tests {
    use super::RequestApprover;

    #[test]
    fn peek_returns_requests_in_fifo_order() {
        let approver = RequestApprover::<&'static str, ()>::new();
        let _first = approver.enqueue("first");
        let _second = approver.enqueue("second");

        let peeked = approver.peek().expect("first request should be visible");
        assert_eq!(peeked, ("0".to_string(), "first"));
    }

    #[test]
    fn peek_skips_resolved_requests() {
        let approver = RequestApprover::<&'static str, ()>::new();
        let (first_id, first_rx) = approver.enqueue("first");
        let _second = approver.enqueue("second");

        assert!(approver.respond(&first_id, ()));
        drop(first_rx);

        let peeked = approver.peek().expect("second request should remain");
        assert_eq!(peeked, ("1".to_string(), "second"));
    }
}
