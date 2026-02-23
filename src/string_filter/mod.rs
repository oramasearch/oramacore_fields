//! Persistent string filter index with exact-match lookups.
//!
//! Maps string keys to sets of doc_ids (u64). Supports inserting, deleting,
//! filtering by exact key, and compacting data to disk. Safe for concurrent use.
//!
//! # On-disk representation
//!
//! ```text
//! base_path/
//! ├── CURRENT                              # text: "<format_version>\n<version>\n"
//! └── versions/
//!     └── <version>/
//!         ├── keys.fst                     # FST map: key -> byte offset in postings.dat
//!         ├── postings.dat                 # per-key doc_id lists
//!         └── deleted.bin                  # sorted deleted doc_ids
//!
//! postings.dat                             (concatenated per-key blocks)
//! ┌──────────────────────────────────────────────────────┐
//! │ Per-key block (offset recorded in FST)               │
//! │ ┌───────────┬──────────┬──────────┬─────┬──────────┐ │
//! │ │ count     │ doc_id   │ doc_id   │ ... │ doc_id   │ │
//! │ │ u64  8B   │ u64  8B  │ u64  8B  │     │ u64  8B  │ │
//! │ └───────────┴──────────┴──────────┴─────┴──────────┘ │
//! ├──────────────────────────────────────────────────────┤
//! │ next key block ...                                   │
//! └──────────────────────────────────────────────────────┘
//!
//! deleted.bin
//! ┌──────────┬──────────┬─────┬──────────┐
//! │ doc_id   │ doc_id   │ ... │ doc_id   │
//! │ u64      │ u64      │     │ u64      │
//! └──────────┴──────────┴─────┴──────────┘
//!
//! All u64 values are native-endian, doc_ids sorted ascending.
//! Files are memory-mapped read-only after creation.
//! ```
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::string_filter::{IndexedValue, StringFilterStorage, Threshold};
//! use std::path::PathBuf;
//!
//! let index = StringFilterStorage::new(PathBuf::from("/tmp/my_index"), Threshold::default()).unwrap();
//!
//! // Insert documents
//! index.insert(&IndexedValue::Plain("hello".to_string()), 1);
//! index.insert(&IndexedValue::Plain("hello".to_string()), 5);
//! index.insert(&IndexedValue::Plain("world".to_string()), 2);
//!
//! // Query documents
//! let hello_docs: Vec<u64> = index.filter("hello").iter().collect();
//! assert_eq!(hello_docs, vec![1, 5]);
//!
//! // Delete a document
//! index.delete(1);
//!
//! // Compact to persist
//! index.compact(1).unwrap();
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
mod storage;

pub use config::Threshold;
pub use error::Error;
pub use indexer::{IndexedValue, StringIndexer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{FilterData, FilterIterator};
pub use storage::StringFilterStorage;
