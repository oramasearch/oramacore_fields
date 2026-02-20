//! Persistent full-text string index with BM25 scoring.
//!
//! Stores per-document term positions and field lengths. Supports search queries
//! that return scored results using BM25F. Safe for concurrent use.
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::string::{StringStorage, Threshold, Bm25Params, IndexedValue, TermData, SearchParams, BM25Scorer};
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
//! terms.insert("hello".to_string(), TermData::new(vec![0], vec![]));
//! index.insert(1, IndexedValue::new(1, terms));
//!
//! // Search
//! let tokens = vec!["hello".to_string()];
//! let mut scorer = BM25Scorer::new();
//! index.search::<oramacore_fields::string::NoFilter>(&SearchParams {
//!     tokens: &tokens,
//!     ..Default::default()
//! }, None, &mut scorer).unwrap();
//! let result = scorer.into_search_result();
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
mod scorer;
mod scoring;
mod storage;

pub use config::{Bm25Params, Threshold};
pub use error::Error;
pub use indexer::{IndexedValue, StringIndexer, TermData, Tokenizer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{ScoredDoc, SearchParams, SearchResult};
pub use scorer::BM25Scorer;
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
