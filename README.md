# oramacore_fields

On-disk field indexes for search engines. Provides six specialized index types with a shared architecture: in-memory live layer for fast writes, memory-mapped compacted storage for efficient reads, and periodic compaction with concurrent access.

## Modules

| Module | Type | Query Operations |
|--------|------|-----------------|
| `bool` | Boolean (true/false) | Filter by value |
| `number` | Numeric (u64, f64) | eq, gt, gte, lt, lte, between |
| `string` | Full-text (BM25 scoring) | Search by tokens (exact, fuzzy, prefix) |
| `string_filter` | String (exact match) | Filter by key |
| `geopoint` | Geographic (lat/lon) | Bounding box, radius |
| `embedding` | Embedding (f32 vectors) | Approximate nearest neighbor (HNSW) |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oramacore_fields = "*"
```

Each index type provides an **Indexer** that extracts values from JSON and a **Storage** that persists and queries them. The typical flow is:

1. Create a `*Indexer` to parse JSON values
2. Create a `*Storage` for on-disk persistence
3. Use `indexer.index_json(json)` to extract an `IndexedValue`
4. Insert the `IndexedValue` into the storage with a document ID
5. Query the storage with filter operations

### Bool Index

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::bool::{BoolIndexer, BoolStorage, DeletionThreshold};
use serde_json::json;
# let dir = tempdir()?;

// Create an indexer for plain boolean values (use `true` for array fields)
let indexer = BoolIndexer::new(false);
let storage = BoolStorage::new(dir.path().to_path_buf(), DeletionThreshold::default()).unwrap();

// Index JSON values
let value = indexer.index_json(&json!(true)).unwrap();
storage.insert(&value, 1);

let value = indexer.index_json(&json!(false)).unwrap();
storage.insert(&value, 2);

let value = indexer.index_json(&json!(true)).unwrap();
storage.insert(&value, 3);

// Query: find all documents with `true`
let true_docs: Vec<u64> = storage.filter(true).iter().collect();
assert_eq!(true_docs, vec![1, 3]);

// Delete a document and compact to disk
storage.delete(1);
storage.compact(1).unwrap();
# Ok(())
# }
```

Array fields are supported by passing `is_array: true`:

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
# use oramacore_fields::bool::{BoolIndexer, BoolStorage, DeletionThreshold};
# use serde_json::json;
# let dir = tempdir()?;
# let storage = BoolStorage::new(dir.path().to_path_buf(), DeletionThreshold::default()).unwrap();
let array_indexer = BoolIndexer::new(true);
let value = array_indexer.index_json(&json!([true, false, true])).unwrap();
storage.insert(&value, 4);
# Ok(())
# }
```

### Number Index

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::number::{NumberIndexer, NumberStorage, Threshold, FilterOp};
use serde_json::json;
# let dir = tempdir()?;

// Create an indexer and storage for u64 values
let indexer = NumberIndexer::<u64>::new(false);
let storage: NumberStorage<u64> = NumberStorage::new(
    dir.path().to_path_buf(),
    Threshold::default(),
).unwrap();

// Index JSON values
let value = indexer.index_json(&json!(10)).unwrap();
storage.insert(&value, 1).unwrap();

let value = indexer.index_json(&json!(20)).unwrap();
storage.insert(&value, 2).unwrap();

let value = indexer.index_json(&json!(30)).unwrap();
storage.insert(&value, 3).unwrap();

// Query: find values >= 15
let results: Vec<u64> = storage.filter(FilterOp::Gte(15)).iter().collect();
assert_eq!(results, vec![2, 3]);

// Delete and compact
storage.delete(2);
storage.compact(1).unwrap();
# Ok(())
# }
```

Also supports `f64` and array fields:

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
# use oramacore_fields::number::{NumberIndexer, NumberStorage, Threshold, FilterOp};
# use serde_json::json;
# let dir = tempdir()?;
# let storage: NumberStorage<u64> = NumberStorage::new(dir.path().to_path_buf(), Threshold::default()).unwrap();
// f64 index
let f64_indexer = NumberIndexer::<f64>::new(false);
let value = f64_indexer.index_json(&json!(3.14)).unwrap();

// Array field: document matches if any element matches
let array_indexer = NumberIndexer::<u64>::new(true);
let value = array_indexer.index_json(&json!([10, 20, 30])).unwrap();
storage.insert(&value, 4).unwrap();

let results: Vec<u64> = storage.filter(FilterOp::Eq(20)).iter().collect();
assert!(results.contains(&4));
# Ok(())
# }
```

### String Index (Full-Text Search)

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::string::{StringStorage, SegmentConfig, IndexedValue, TermData, SearchParams, BM25u64Scorer};
use std::collections::HashMap;
# let dir = tempdir()?;

let storage = StringStorage::new(
    dir.path().to_path_buf(),
    SegmentConfig::default(),
).unwrap();

// Insert a document with term positions
let mut terms = HashMap::new();
terms.insert("hello".to_string(), TermData::new(vec![0], vec![]));
terms.insert("world".to_string(), TermData::new(vec![1], vec![]));
storage.insert(1, IndexedValue::new(2, terms));

// Search for documents matching a token
let tokens = vec!["hello".to_string()];
let mut scorer = BM25u64Scorer::new();
storage.search(&SearchParams {
    tokens: &tokens,
    ..Default::default()
}, &mut scorer).unwrap();
let result = scorer.into_search_result();
assert_eq!(result.docs.len(), 1);

// Delete and compact
storage.delete(1);
storage.compact(1).unwrap();
# Ok(())
# }
```

### String Filter Index

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::string_filter::{StringIndexer, StringFilterStorage, Threshold};
use serde_json::json;
# let dir = tempdir()?;

let indexer = StringIndexer::new(false, |_| true);
let storage = StringFilterStorage::new(dir.path().to_path_buf(), Threshold::default()).unwrap();

// Index JSON strings
let value = indexer.index_json(&json!("hello")).unwrap();
storage.insert(&value, 1);

let value = indexer.index_json(&json!("world")).unwrap();
storage.insert(&value, 2);

let value = indexer.index_json(&json!("hello")).unwrap();
storage.insert(&value, 3);

// Query: exact match
let docs: Vec<u64> = storage.filter("hello").iter().collect();
assert_eq!(docs, vec![1, 3]);
# Ok(())
# }
```

Array fields are supported:

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
# use oramacore_fields::string_filter::{StringIndexer, StringFilterStorage, Threshold};
# use serde_json::json;
# let dir = tempdir()?;
# let storage = StringFilterStorage::new(dir.path().to_path_buf(), Threshold::default()).unwrap();
let array_indexer = StringIndexer::new(true, |_| true);
let value = array_indexer.index_json(&json!(["rust", "search", "index"])).unwrap();
storage.insert(&value, 4);
# Ok(())
# }
```

### GeoPoint Index

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::geopoint::{GeoPointIndexer, GeoPointStorage, Threshold, GeoFilterOp, GeoPoint};
use serde_json::json;
# let dir = tempdir()?;

let indexer = GeoPointIndexer::new(false);
let storage = GeoPointStorage::new(dir.path().to_path_buf(), Threshold::default(), 10).unwrap();

// Index JSON geopoints (objects with "lat" and "lon" fields)
let value = indexer.index_json(&json!({"lat": 41.9028, "lon": 12.4964})).unwrap(); // Rome
storage.insert(value, 1);

let value = indexer.index_json(&json!({"lat": 51.5074, "lon": -0.1278})).unwrap(); // London
storage.insert(value, 2);

// Query: find points within a radius
let results: Vec<u64> = storage.filter(GeoFilterOp::Radius {
    center: GeoPoint::new(41.9, 12.5).unwrap(),
    radius_meters: 10_000.0,
}).iter().collect();
assert_eq!(results, vec![1]);

// Query: find points within a bounding box
let results: Vec<u64> = storage.filter(GeoFilterOp::BoundingBox {
    min_lat: 40.0,
    max_lat: 55.0,
    min_lon: -5.0,
    max_lon: 15.0,
}).iter().collect();
# Ok(())
# }
```

### Embedding Index

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
use oramacore_fields::embedding::{EmbeddingStorage, EmbeddingConfig, DistanceMetric, SegmentConfig, EmbeddingIndexer};
# let dir = tempdir()?;

let config = EmbeddingConfig::new(3, DistanceMetric::Cosine).unwrap();
let storage = EmbeddingStorage::new(dir.path().to_path_buf(), config, SegmentConfig::default()).unwrap();
let indexer = EmbeddingIndexer::new(3);

// Insert vectors
storage.insert(1, indexer.index_vec(&[0.1, 0.2, 0.3]).unwrap());
storage.insert(2, indexer.index_vec(&[0.4, 0.5, 0.6]).unwrap());
storage.insert(3, indexer.index_vec(&[0.9, 0.1, 0.0]).unwrap());

// Search for nearest neighbors (returns Vec<(doc_id, distance)>)
let results = storage.search(&[0.1, 0.2, 0.3], 2, None).unwrap();
assert_eq!(results[0].0, 1); // closest doc_id

// Delete and compact
storage.delete(3);
storage.compact(1).unwrap();
# Ok(())
# }
```

## CLI

Build the CLI with:

```sh
cargo build --features cli
```

Commands:

```text
oramacore_fields bool check|info|show|filter <path>
oramacore_fields number check|info|filter <args>
oramacore_fields geopoint check|info|filter <args>
oramacore_fields string check|info|search <args>
oramacore_fields string-filter check|info|filter <args>
oramacore_fields embedding check|info|search <args>
```

## Architecture

All six modules share a common two-layer design:

- **LiveLayer**: In-memory storage for fast inserts and deletes. Thread-safe via `RwLock`.
- **CompactedVersion**: Memory-mapped disk storage loaded via `memmap2`. Swapped atomically using `ArcSwap` for lock-free reads during compaction.
- **Compaction**: Periodically merges live data into a new on-disk version, enabling persistence without blocking reads or writes.

## License

AGPL-3.0