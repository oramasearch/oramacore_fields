//! Boolean index that maps documents to true/false values.
//!
//! Stores doc_ids (u64) partitioned into TRUE and FALSE sets, with full
//! persistence and support for concurrent readers and writers.
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::bool::{BoolStorage, DeletionThreshold, IndexedValue};
//! use std::path::PathBuf;
//!
//! let index = BoolStorage::new(PathBuf::from("/tmp/my_index"), DeletionThreshold::default()).unwrap();
//!
//! // Insert documents
//! index.insert(&IndexedValue::Plain(true), 1);
//! index.insert(&IndexedValue::Plain(true), 5);
//! index.insert(&IndexedValue::Plain(false), 2);
//!
//! // Query documents
//! let true_docs: Vec<u64> = index.filter(true).iter().collect();
//! assert_eq!(true_docs, vec![1, 5]);
//!
//! // Delete a document (deletes from both TRUE and FALSE sets)
//! index.delete(1);
//!
//! // Compact to persist
//! index.compact(1).unwrap();
//! ```

mod info;
#[doc(hidden)]
pub mod io;
mod iterator;
mod live;
mod merge;
mod platform;
mod storage;
#[doc(hidden)]
pub mod version;

mod indexer;

pub use indexer::{BoolIndexer, IndexedValue};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{DescendingIterator, FilterData, FilterIterator, SortOrder, SortedIterator};
pub use storage::{BoolStorage, DeletionThreshold};
