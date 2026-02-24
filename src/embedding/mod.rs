//! Embedding index with approximate nearest neighbor search via HNSW graphs.
//!
//! Stores high-dimensional embeddings with doc_id associations, supporting ANN
//! search using quantized HNSW graphs backed by memory-mapped files.
//!
//! # On-disk representation
//!
//! ```text
//! base_path/
//! ├── CURRENT                              # text: "<format_version>\n<version>\n"
//! ├── versions/
//! │   └── <version>/
//! │       ├── manifest.json                # segment metadata array (JSON)
//! │       ├── seg_0.del                    # sorted deleted doc_ids for segment 0
//! │       ├── seg_1.del                    # (only if segment has deletions)
//! │       └── ...
//! └── segments/                            # shared across versions
//!     └── seg_<id>/
//!         ├── hnsw.meta                    # JSON: dimensions, metric, m, m0, ...
//!         ├── vectors.raw                  # raw f32 embeddings
//!         ├── vectors.quantized            # quantized i8 embeddings
//!         ├── hnsw.graph                   # HNSW graph structure
//!         ├── doc_ids.bin                  # sorted doc_id array
//!         ├── levels.bin                   # per-node HNSW level (u8 each)
//!         └── quantization.bin             # per-dimension min/max for quantization
//!
//! vectors.raw                              (flat f32 buffer, D = dimensions)
//! ┌──────────────────────────────────────────────┐
//! │ vec_0: [f32; D] │ vec_1: [f32; D] │ ...      │
//! └──────────────────────────────────────────────┘
//!  D * 4 bytes per vector
//!
//! vectors.quantized                        (flat i8 buffer)
//! ┌──────────────────────────────────────────────┐
//! │ vec_0: [i8; D]  │ vec_1: [i8; D]  │ ...      │
//! └──────────────────────────────────────────────┘
//!  D bytes per vector, values in [-127, 127]
//!
//! hnsw.graph
//! ┌───────────────────────────────────────────────────────────┐
//! │ Header (16 bytes)                                         │
//! │ ┌───────────┬──────┬──────┬───────────┬────────────┬─────┐│
//! │ │ num_nodes │ m    │ m0   │ max_level │ entry_point│ pad ││
//! │ │ u32  4B   │u16 2B│u16 2B│ u16  2B   │ u32  4B    │2B   ││
//! │ └───────────┴──────┴──────┴───────────┴────────────┴─────┘│
//! │ Per-node block (repeated num_nodes times)                 │
//! │ ┌──────────────────────────────────────────────────┐      │
//! │ │ layer 0 neighbors: [u32; m0]                     │      │
//! │ │ layer 1 neighbors: [u32; m]                      │      │
//! │ │ ...                                              │      │
//! │ │ layer max_level-1 neighbors: [u32; m]            │      │
//! │ └──────────────────────────────────────────────────┘      │
//! │  block_size = m0*4 + (max_level-1)*m*4 bytes              │
//! │  unused slots filled with 0xFFFFFFFF (sentinel)           │
//! └───────────────────────────────────────────────────────────┘
//!
//! quantization.bin
//! ┌────────────┬─────────────────────┬─────────────────────┐
//! │ dimensions │ mins: [f32; D]      │ maxs: [f32; D]      │
//! │ u32  4B    │ D * 4 bytes         │ D * 4 bytes         │
//! └────────────┴─────────────────────┴─────────────────────┘
//!
//! doc_ids.bin                              (sorted u64 array)
//! ┌──────────┬──────────┬─────┬──────────┐
//! │ doc_id   │ doc_id   │ ... │ doc_id   │
//! │ u64  8B  │ u64  8B  │     │ u64  8B  │
//! └──────────┴──────────┴─────┴──────────┘
//!
//! seg_<id>.del                             (sorted u64 array)
//! ┌──────────┬──────────┬─────┬──────────┐
//! │ doc_id   │ doc_id   │ ... │ doc_id   │
//! │ u64  8B  │ u64  8B  │     │ u64  8B  │
//! └──────────┴──────────┴─────┴──────────┘
//!
//! All numeric values are native-endian. Segment data is shared
//! across versions; only manifest.json and .del files are per-version.
//! ```
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
mod simd;
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

pub use config::{DeletionThreshold, DistanceMetric, EmbeddingConfig, SegmentConfig};
pub use error::Error;
pub use indexer::{Embedding, EmbeddingIndexer, IndexedValue};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use storage::EmbeddingStorage;
