//! Geographic point spatial index with bounding-box and radius queries.
//!
//! Maps doc_ids (u64) to latitude/longitude coordinates, stored in a BKD tree
//! for efficient spatial lookups. Supports full persistence with compaction
//! and concurrent readers and writers.
//!
//! # On-disk representation
//!
//! ```text
//! base_path/
//! ├── CURRENT                              # text: "<format_version>\n<version>\n"
//! └── versions/
//!     └── <version>/
//!         ├── manifest.bin                 # segment count + per-segment point counts
//!         ├── deleted.bin                  # sorted deleted doc_ids
//!         ├── segment_0/
//!         │   ├── inner.idx               # BKD tree inner nodes + leaf offsets
//!         │   └── leaves.dat              # BKD tree leaf data
//!         ├── segment_1/
//!         │   ├── inner.idx
//!         │   └── leaves.dat
//!         └── ...
//!
//! manifest.bin
//! ┌──────────────┬──────────┬──────────┬─────┬──────────┐
//! │ num_segments │ count_0  │ count_1  │ ... │ count_N  │
//! │ u64  8B      │ u64  8B  │ u64  8B  │     │ u64  8B  │
//! └──────────────┴──────────┴──────────┴─────┴──────────┘
//!
//! inner.idx
//! ┌─────────────────────────────────────────┬────────────────────────────┐
//! │ inner nodes (num_leaves - 1)            │ leaf offsets (num_leaves)  │
//! │ ┌─────────────┬───────────┬──────────┐  │ ┌────────────┐            │
//! │ │ split_value │ split_dim │ _padding │  │ │ offset     │  repeated  │
//! │ │ i32  4B     │ u8  1B    │ 3B       │  │ │ u64  8B    │            │
//! │ └─────────────┴───────────┴──────────┘  │ └────────────┘            │
//! │  8 bytes per node, repeated             │  8 bytes each             │
//! └─────────────────────────────────────────┴────────────────────────────┘
//!
//! leaves.dat                               (concatenated leaf blocks)
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Leaf header                                                 │
//! │ ┌──────────────┬─────────┐                                  │
//! │ │ entry_count  │ padding │                                  │
//! │ │ u32  4B      │ u32  4B │                                  │
//! │ └──────────────┴─────────┘                                  │
//! │ Entries (repeated entry_count times)                        │
//! │ ┌──────────┬──────────┬──────────┐                          │
//! │ │ lat      │ lon      │ doc_id   │                          │
//! │ │ i32  4B  │ i32  4B  │ u64  8B  │                          │
//! │ └──────────┴──────────┴──────────┘                          │
//! │  16 bytes per entry                                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │ next leaf ...                                               │
//! └─────────────────────────────────────────────────────────────┘
//!
//! lat/lon encoding: round(degrees * (2^31 - 1) / 180) for lat,
//!                   round(degrees * (2^31 - 1) / 360) for lon.
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
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use oramacore_fields::geopoint::{
//!     GeoFilterOp, GeoPoint, GeoPointIndexer, GeoPointStorage, Threshold,
//! };
//!
//! let dir = tempfile::tempdir()?;
//! let index = GeoPointStorage::new(
//!     dir.path().to_path_buf(),
//!     Threshold::default(),
//!     10,
//! )?;
//! let indexer = GeoPointIndexer::new(false);
//!
//! // Insert documents with geographic coordinates
//! let rome = indexer.index_json(&serde_json::json!({"lat": 41.9028, "lon": 12.4964})).unwrap();
//! index.insert(rome, 1);
//!
//! let london = indexer.index_json(&serde_json::json!({"lat": 51.5074, "lon": -0.1278})).unwrap();
//! index.insert(london, 2);
//!
//! // Query by bounding box
//! let results: Vec<u64> = index.filter(GeoFilterOp::BoundingBox {
//!     min_lat: 40.0, max_lat: 45.0,
//!     min_lon: 10.0, max_lon: 15.0,
//! }).iter().collect();
//! assert_eq!(results, vec![1]); // Only Rome
//!
//! // Query by radius
//! let center = GeoPoint::new(41.9028, 12.4964)?;
//! let results: Vec<u64> = index.filter(GeoFilterOp::Radius {
//!     center,
//!     radius_meters: 100_000.0,
//! }).iter().collect();
//!
//! // Delete a document and compact
//! index.delete(1);
//! index.compact(1)?;
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
mod live;
mod mmap_vec;
mod platform;
mod point;
mod storage;

pub use config::Threshold;
pub use error::Error;
pub use indexer::{GeoPointIndexer, IndexedValue};
pub use info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
pub use iterator::{FilterData, FilterIterator, GeoFilterOp};
pub use point::{GeoPoint, GeoPolygon};
pub use storage::GeoPointStorage;
