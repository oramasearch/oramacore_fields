//! Embedding index with approximate nearest neighbor search via HNSW graphs.
//!
//! Stores high-dimensional embeddings with doc_id associations, supporting ANN
//! search using quantized HNSW graphs backed by memory-mapped files.
//!
//! # Example
//!
//! ```no_run
//! use oramacore_fields::embedding::{EmbeddingStorage, EmbeddingConfig, DistanceMetric, SegmentConfig, EmbeddingIndexer};
//! use std::path::PathBuf;
//!
//! let config = EmbeddingConfig::new(3, DistanceMetric::Cosine).unwrap();
//! let index = EmbeddingStorage::new(PathBuf::from("/tmp/my_embeddings"), config, SegmentConfig::default()).unwrap();
//! let indexer = EmbeddingIndexer::new(3);
//!
//! // Insert embeddings
//! index.insert(1, indexer.index_vec(&[0.1, 0.2, 0.3]).unwrap());
//! index.insert(2, indexer.index_vec(&[0.4, 0.5, 0.6]).unwrap());
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

pub use config::{DeletionThreshold, DistanceMetric, SegmentConfig, EmbeddingConfig};
pub use error::Error;
pub use indexer::{Embedding, IndexedValue, EmbeddingIndexer};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use storage::EmbeddingStorage;
