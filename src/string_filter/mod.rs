//! Persistent string filter index with exact-match lookups.
//!
//! Maps string keys to sets of doc_ids (u64). Supports inserting, deleting,
//! filtering by exact key, and compacting data to disk. Safe for concurrent use.
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
