//! A number index supporting range queries over u64 and f64 values.
//!
//! This module provides a persistent, thread-safe number index with:
//! - Support for both u64 and f64 value types
//! - Range queries (eq, gt, gte, lt, lte, between)
//! - JSON indexing via [`NumberIndexer`] (plain values and arrays)
//! - On-disk persistence with compaction
//!
//! # On-disk representation
//!
//! ```text
//! base_path/
//! ├── CURRENT                              # text: "<format_version>\n<version>\n"
//! └── versions/
//!     └── <version>/
//!         ├── header.idx                   # sparse index into data buckets
//!         ├── data_0000.dat                # first data bucket
//!         ├── data_0001.dat                # second data bucket (if needed)
//!         ├── ...
//!         └── deleted.bin                  # sorted deleted doc_ids
//!
//! header.idx                               (array of 24-byte entries)
//! ┌──────────┬──────────────┬───────────────┐
//! │ key      │ bucket_index │ bucket_offset │  repeated
//! │ u64/f64  │ u64          │ u64           │
//! └──────────┴──────────────┴───────────────┘
//!  8 bytes    8 bytes        8 bytes
//!
//! One entry emitted every ~1000 cumulative doc_ids.
//! Points to the (bucket file, byte offset) of a data entry.
//!
//! data_NNNN.dat                            (chain of variable-size entries)
//! ┌──────────────────────────────────────────────────────────┐
//! │ DataEntryHeader                                          │
//! │ ┌──────┬──────────────────┬──────────────────┬─────────┐ │
//! │ │ key  │ next_entry_offset│ prev_entry_offset│ count   │ │
//! │ │ 8B   │ u64  8B          │ u64  8B          │ u64  8B │ │
//! │ └──────┴──────────────────┴──────────────────┴─────────┘ │
//! │ ┌──────────┬──────────┬─────┬──────────┐                 │
//! │ │ doc_id   │ doc_id   │ ... │ doc_id   │  (count items)  │
//! │ │ u64  8B  │ u64  8B  │     │ u64  8B  │                 │
//! │ └──────────┴──────────┴─────┴──────────┘                 │
//! ├──────────────────────────────────────────────────────────┤
//! │ next entry ...                                           │
//! └──────────────────────────────────────────────────────────┘
//!
//! next/prev offsets are byte offsets within the same bucket file.
//! 0 = no next/prev. Offsets never cross bucket boundaries.
//! Entries are sorted by key; doc_ids within an entry are sorted ascending.
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
//! Use [`NumberIndexer`] to extract values from JSON and [`NumberStorage::insert`]
//! to index them:
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use oramacore_fields::number::{NumberIndexer, NumberStorage, FilterOp, Threshold};
//! use serde_json::json;
//!
//! // Create an index and an indexer for plain u64 values
//! let dir = tempfile::tempdir()?;
//! let index: NumberStorage<u64> = NumberStorage::new(
//!     dir.path().to_path_buf(),
//!     Threshold::default(),
//! )?;
//! let indexer = NumberIndexer::<u64>::new(false);
//!
//! // Index JSON values
//! let value = indexer.index_json(&json!(10)).unwrap();
//! index.insert(&value, 1)?;
//!
//! let value = indexer.index_json(&json!(20)).unwrap();
//! index.insert(&value, 2)?;
//!
//! let value = indexer.index_json(&json!(30)).unwrap();
//! index.insert(&value, 3)?;
//!
//! // Query for values >= 15
//! let results: Vec<u64> = index.filter(FilterOp::Gte(15)).iter().collect();
//! assert_eq!(results, vec![2, 3]);
//!
//! // Delete a document and compact
//! index.delete(2);
//! index.compact(1)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Array Fields
//!
//! Documents with array-valued fields (e.g., `tags: [10, 20, 30]`) are indexed
//! once per element, so a query matching any element finds the document:
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use oramacore_fields::number::{NumberIndexer, NumberStorage, FilterOp, Threshold};
//! use serde_json::json;
//!
//! let dir = tempfile::tempdir()?;
//! let index: NumberStorage<u64> = NumberStorage::new(
//!     dir.path().to_path_buf(),
//!     Threshold::default(),
//! )?;
//! let array_indexer = NumberIndexer::<u64>::new(true);
//!
//! let value = array_indexer.index_json(&json!([10, 20, 30])).unwrap();
//! index.insert(&value, 1)?;
//!
//! // Doc 1 is found when querying for any of its values
//! let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
//! assert_eq!(results, vec![1]);
//! # Ok(())
//! # }
//! ```

mod compacted;
mod config;
mod error;
mod indexer;
mod info;
mod io;
mod iterator;
mod key;
mod live;
mod merge;
mod platform;
mod storage;

// Re-exports
pub use compacted::{CompactionMeta, DEFAULT_BUCKET_TARGET_BYTES, DEFAULT_INDEX_STRIDE};
pub use config::Threshold;
pub use error::Error;
pub use indexer::{IndexedValue, NumberIndexer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{FilterHandle, FilterIterator, FilterOp, SortHandle, SortIterator, SortOrder};
pub use key::IndexableNumber;
pub use storage::{F64Storage, NumberStorage, U64Storage};
