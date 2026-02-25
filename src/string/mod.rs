//! Persistent full-text string index with BM25 scoring.
//!
//! Stores per-document term positions and field lengths. Supports search queries
//! that return scored results using BM25F. Safe for concurrent use.
//!
//! # On-disk representation
//!
//! ```text
//! base_path/
//! ├── CURRENT                              # text: "<format_version>\n<version>\n"
//! └── versions/
//!     └── <version>/
//!         ├── keys.fst                     # FST map: term -> byte offset in postings.dat
//!         ├── postings.dat                 # per-term posting lists
//!         ├── doc_lengths.dat              # per-doc field lengths (for BM25)
//!         ├── deleted.bin                  # sorted deleted doc_ids
//!         └── global_info.bin              # aggregate stats for BM25
//!
//! postings.dat                             (concatenated per-term blocks)
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Per-term header                                             │
//! │ ┌────────────┬─────────┐                                    │
//! │ │ doc_count  │ padding │                                    │
//! │ │ u32  4B    │ u32  4B │                                    │
//! │ └────────────┴─────────┘                                    │
//! │ Per-doc entry (repeated doc_count times)                    │
//! │ ┌────────┬─────────────┬───────────────┬────────┬────────┐  │
//! │ │ doc_id │ exact_count │ stemmed_count │ exact  │stemmed │  │
//! │ │ u64 8B │ u32  4B     │ u32  4B       │positions│positions│
//! │ └────────┴─────────────┴───────────────┴────────┴────────┘  │
//! │  positions: u32 array, 4 bytes each                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │ next term block ...                                         │
//! └─────────────────────────────────────────────────────────────┘
//!
//! doc_lengths.dat                          (array of 12-byte entries)
//! ┌──────────┬──────────────┐
//! │ doc_id   │ field_length │  repeated, sorted by doc_id
//! │ u64  8B  │ u32  4B      │
//! └──────────┴──────────────┘
//!
//! global_info.bin                          (fixed 16 bytes)
//! ┌────────────────────────┬──────────────────┐
//! │ total_document_length  │ total_documents  │
//! │ u64  8B                │ u64  8B          │
//! └────────────────────────┴──────────────────┘
//!
//! deleted.bin
//! ┌──────────┬──────────┬─────┬──────────┐
//! │ doc_id   │ doc_id   │ ... │ doc_id   │
//! │ u64      │ u64      │     │ u64      │
//! └──────────┴──────────┴─────┴──────────┘
//!
//! All values are native-endian. Files are memory-mapped read-only.
//! ```
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
//! index.search(&SearchParams {
//!     tokens: &tokens,
//!     ..Default::default()
//! }, &mut scorer).unwrap();
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
