//! Geographic point spatial index with bounding-box and radius queries.
//!
//! Maps doc_ids (u64) to latitude/longitude coordinates, stored in a BKD tree
//! for efficient spatial lookups. Supports full persistence with compaction
//! and concurrent readers and writers.
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
pub use point::GeoPoint;
pub use storage::GeoPointStorage;
