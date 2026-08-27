//! Persistence backends for RAG indexes.
//!
//! This module provides the [`Persistence`] trait and implementations for
//! saving and loading index data.

#[cfg(feature = "lancedb-persistence")]
mod lancedb_backend;
mod redb_backend;
mod rkyv_backend;

#[cfg(feature = "lancedb-persistence")]
pub use lancedb_backend::LanceDbPersistence;
pub use redb_backend::RedbPersistence;
pub use rkyv_backend::RkyvPersistence;

use crate::error::Result;
use crate::types::IndexEntry;
use std::future::Future;
use std::path::Path;

/// Trait for persistence backends.
///
/// Persistence backends handle saving and loading index entries to/from storage.
pub trait Persistence: Send + Sync {
    /// Saves all index entries to storage.
    ///
    /// # Errors
    /// Returns an error when the backend cannot write or serialize entries.
    fn save<'a>(
        &'a self,
        entries: &'a [IndexEntry],
    ) -> impl Future<Output = Result<()>> + Send + 'a;

    /// Loads all index entries from storage.
    ///
    /// Returns an empty vector if no data exists.
    ///
    /// # Errors
    /// Returns an error when the backend cannot read or deserialize entries.
    fn load(&self) -> impl Future<Output = Result<Vec<IndexEntry>>> + Send + '_;

    /// Returns the file extension used by this backend.
    fn extension(&self) -> &'static str;

    /// Returns the storage path.
    fn path(&self) -> &Path;
}
