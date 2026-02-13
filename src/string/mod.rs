//! Persistent full-text string index with BM25 scoring.
//!
//! Stores per-document term positions and field lengths. Supports search queries
//! that return scored results using BM25F. Safe for concurrent use.
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::string::{StringStorage, Threshold, Bm25Params, IndexedValue, TermData, SearchParams};
//! use std::collections::HashMap;
//! use std::path::PathBuf;
//!
//! let index = StringStorage::new(
//!     PathBuf::from("/tmp/my_string_index"),
//!     Threshold::default(),
//! ).unwrap();
//!
//! // Insert a document
//! let mut terms = HashMap::new();
//! terms.insert("hello".to_string(), TermData {
//!     exact_positions: vec![0],
//!     stemmed_positions: vec![],
//! });
//! index.insert(1, IndexedValue { field_length: 1, terms });
//!
//! // Search
//! let tokens = vec!["hello".to_string()];
//! let result = index.search::<oramacore_fields::string::NoFilter>(&SearchParams {
//!     tokens: &tokens,
//!     ..Default::default()
//! }, None);
//! assert_eq!(result.docs.len(), 1);
//! ```

mod compacted;
mod config;
mod error;
mod indexer;
mod info;
#[doc(hidden)]
pub mod io;
mod iterator;
mod live;
mod merge;
mod platform;
mod scoring;
mod storage;

pub use config::{Bm25Params, Threshold};
pub use error::Error;
pub use indexer::{IndexedValue, StringIndexer, TermData, Tokenizer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{ScoredDoc, SearchParams, SearchResult};
pub use storage::StringStorage;

pub trait DocumentFilter {
    fn contains(&self, doc_id: u64) -> bool;
}

/// No-op filter that accepts all documents. Used as default when no filtering is needed.
pub struct NoFilter;
impl DocumentFilter for NoFilter {
    #[inline]
    fn contains(&self, _doc_id: u64) -> bool {
        true
    }
}
