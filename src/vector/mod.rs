//! Vector index with approximate nearest neighbor search via HNSW graphs.
//!
//! Stores high-dimensional vectors with doc_id associations, supporting ANN
//! search using quantized HNSW graphs backed by memory-mapped files.
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::vector::{VectorStorage, VectorConfig, DistanceMetric, SegmentConfig};
//! use std::path::PathBuf;
//!
//! let config = VectorConfig::new(3, DistanceMetric::Cosine).unwrap();
//! let index = VectorStorage::new(PathBuf::from("/tmp/my_vectors"), config, SegmentConfig::default()).unwrap();
//!
//! // Insert vectors
//! index.insert(1, &[0.1, 0.2, 0.3]).unwrap();
//! index.insert(2, &[0.4, 0.5, 0.6]).unwrap();
//!
//! // Search for nearest neighbors
//! let results = index.search(&[0.1, 0.2, 0.3], 2, None).unwrap();
//! assert_eq!(results[0].0, 1); // closest doc_id
//!
//! // Compact to persist
//! index.compact(1).unwrap();
//! ```

mod config;
mod distance;
mod error;
mod hnsw;
mod indexer;
mod info;
#[doc(hidden)]
pub mod io;
mod live;
mod platform;
mod quantization;
pub(crate) mod segment;
mod storage;

pub use config::{DeletionThreshold, DistanceMetric, SegmentConfig, VectorConfig};
pub use error::Error;
pub use indexer::{IndexedValue, VectorIndexer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use storage::VectorStorage;
