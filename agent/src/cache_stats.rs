//! Rolling KV-cache hit-rate statistics for a running agent.
//!
//! Models like Claude expose `cache_read_tokens` and `cache_write_tokens`
//! in their [`aither_core::llm::Usage`] payloads. Prompt caching is
//! dramatically cheaper than uncached input (Anthropic pricing: ~0.30 vs
//! 3.00 USD/MTok, a 10× difference), so a production-grade agent host
//! should track the cumulative hit rate to catch regressions where a
//! stable system prefix is accidentally invalidated between requests.
//!
//! [`CacheStats`] is a pure accumulator: hosts call
//! [`CacheStats::record`] with each observed `Usage` (typically when
//! handling an [`AgentEvent::Usage`](crate::AgentEvent::Usage) event)
//! and read the current rolling metrics back via the getters. The
//! accumulator stores raw totals so hosts can trivially diff two
//! snapshots if they want per-turn ratios.

use aither_core::llm::Usage;

/// Aggregate prompt-caching statistics collected across one agent session.
///
/// The counters are cumulative totals, not per-turn deltas. Hosts wanting
/// per-turn metrics should take snapshots via [`CacheStats::snapshot`]
/// and diff two snapshots; hosts wanting a single rolling rate can read
/// [`CacheStats::hit_rate`] directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of usage payloads recorded so far.
    requests: u64,
    /// Cumulative prompt tokens billed (both cached and fresh).
    prompt_tokens: u64,
    /// Cumulative prompt tokens that were served from the KV cache.
    cache_read_tokens: u64,
    /// Cumulative prompt tokens newly written to the KV cache.
    cache_write_tokens: u64,
}

impl CacheStats {
    /// Creates an empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            requests: 0,
            prompt_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }

    /// Records a single `Usage` observation.
    ///
    /// Missing fields on `usage` are treated as zero so providers that
    /// do not report cache metrics simply contribute nothing to the
    /// cache numerator. The request counter increments on every call
    /// regardless of field availability, so [`Self::requests`] is a
    /// faithful count of observed usage payloads.
    pub fn record(&mut self, usage: &Usage) {
        self.requests = self.requests.saturating_add(1);
        if let Some(prompt) = usage.prompt_tokens {
            self.prompt_tokens = self.prompt_tokens.saturating_add(u64::from(prompt));
        }
        if let Some(read) = usage.cache_read_tokens {
            self.cache_read_tokens = self.cache_read_tokens.saturating_add(u64::from(read));
        }
        if let Some(write) = usage.cache_write_tokens {
            self.cache_write_tokens = self.cache_write_tokens.saturating_add(u64::from(write));
        }
    }

    /// Resets all counters.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the number of usage payloads recorded.
    #[must_use]
    pub const fn requests(&self) -> u64 {
        self.requests
    }

    /// Returns cumulative prompt tokens (cached + fresh).
    #[must_use]
    pub const fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens
    }

    /// Returns cumulative tokens served from the KV cache.
    #[must_use]
    pub const fn cache_read_tokens(&self) -> u64 {
        self.cache_read_tokens
    }

    /// Returns cumulative tokens freshly written to the KV cache.
    #[must_use]
    pub const fn cache_write_tokens(&self) -> u64 {
        self.cache_write_tokens
    }

    /// Returns the rolling cache-read share of prompt tokens.
    ///
    /// Computed as `cache_read_tokens / prompt_tokens`. Returns `None`
    /// when no prompt tokens have been observed yet — attempting to
    /// answer "what's our hit rate?" before any request has run would
    /// be misleading.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        if self.prompt_tokens == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = self.cache_read_tokens as f64 / self.prompt_tokens as f64;
        Some(rate)
    }

    /// Returns a copy of the current counters as a plain value.
    ///
    /// Useful for comparing two moments in time without taking a
    /// mutable reference to the live accumulator.
    #[must_use]
    pub const fn snapshot(&self) -> Self {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, cache_read: u32, cache_write: u32) -> Usage {
        Usage::new(prompt, 0).with_cache_tokens(cache_read, cache_write)
    }

    #[test]
    fn empty_stats_have_no_hit_rate() {
        let stats = CacheStats::new();
        assert_eq!(stats.requests(), 0);
        assert_eq!(stats.hit_rate(), None);
    }

    #[test]
    fn record_aggregates_prompt_and_cache_tokens() {
        let mut stats = CacheStats::new();
        stats.record(&usage(1_000, 800, 200));
        stats.record(&usage(500, 450, 50));

        assert_eq!(stats.requests(), 2);
        assert_eq!(stats.prompt_tokens(), 1_500);
        assert_eq!(stats.cache_read_tokens(), 1_250);
        assert_eq!(stats.cache_write_tokens(), 250);
        assert!((stats.hit_rate().unwrap() - (1_250.0 / 1_500.0)).abs() < 1e-9);
    }

    #[test]
    fn missing_fields_do_not_poison_totals() {
        let mut stats = CacheStats::new();
        // Provider that reports no cache fields at all.
        let bare = Usage {
            prompt_tokens: Some(400),
            ..Usage::default()
        };
        stats.record(&bare);

        assert_eq!(stats.requests(), 1);
        assert_eq!(stats.prompt_tokens(), 400);
        assert_eq!(stats.cache_read_tokens(), 0);
        assert_eq!(stats.hit_rate(), Some(0.0));
    }

    #[test]
    fn reset_clears_all_counters() {
        let mut stats = CacheStats::new();
        stats.record(&usage(100, 50, 50));
        stats.reset();
        assert_eq!(stats, CacheStats::new());
    }

    #[test]
    fn snapshot_is_independent_of_the_live_accumulator() {
        let mut stats = CacheStats::new();
        stats.record(&usage(100, 50, 50));
        let snap = stats.snapshot();
        stats.record(&usage(100, 100, 0));
        assert_eq!(snap.prompt_tokens(), 100);
        assert_eq!(snap.cache_read_tokens(), 50);
        assert_eq!(stats.prompt_tokens(), 200);
        assert_eq!(stats.cache_read_tokens(), 150);
    }
}
