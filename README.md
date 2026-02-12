# oramacore_fields

On-disk field indexes for search engines. Provides four specialized index types with a shared architecture: in-memory live layer for fast writes, memory-mapped compacted storage for efficient reads, and periodic compaction with concurrent access.

## Modules

| Module | Type | Query Operations |
|--------|------|-----------------|
| `bool` | Boolean (true/false) | Filter by value |
| `number` | Numeric (u64, f64) | eq, gt, gte, lt, lte, between |
| `string_filter` | String (exact match) | Filter by key |
| `geopoint` | Geographic (lat/lon) | Bounding box, radius |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oramacore_fields = "*"
```

### Bool Index

```rust
use oramacore_fields::bool::{BoolStorage, DeletionThreshold, IndexedValue};
use std::path::PathBuf;

let index = BoolStorage::new(PathBuf::from("/tmp/bool_idx"), DeletionThreshold::default()).unwrap();

index.insert(&IndexedValue::Plain(true), 1);
index.insert(&IndexedValue::Plain(false), 2);

let true_docs: Vec<u64> = index.filter(true).iter().collect();
assert_eq!(true_docs, vec![1]);

index.compact(1).unwrap();
```

### Number Index

```rust
use oramacore_fields::number::{NumberStorage, Threshold, FilterOp, IndexedValue};
use std::path::PathBuf;

let index: NumberStorage<u64> = NumberStorage::new(
    PathBuf::from("/tmp/num_idx"),
    Threshold::default(),
).unwrap();

index.insert(&IndexedValue::Plain(42), 1).unwrap();
index.insert(&IndexedValue::Plain(100), 2).unwrap();

let results: Vec<u64> = index.filter(FilterOp::Gte(50)).iter().collect();
assert_eq!(results, vec![2]);
```

### String Filter Index

```rust
use oramacore_fields::string_filter::{StringFilterStorage, Threshold, IndexedValue};
use std::path::PathBuf;

let index = StringFilterStorage::new(PathBuf::from("/tmp/str_idx"), Threshold::default()).unwrap();

index.insert(&IndexedValue::Plain("hello".to_string()), 1);
index.insert(&IndexedValue::Plain("world".to_string()), 2);

let docs: Vec<u64> = index.filter("hello").iter().collect();
assert_eq!(docs, vec![1]);
```

### GeoPoint Index

```rust
use oramacore_fields::geopoint::{GeoPointStorage, Threshold, GeoFilterOp, GeoPoint, IndexedValue};
use std::path::PathBuf;

let index = GeoPointStorage::new(PathBuf::from("/tmp/geo_idx"), Threshold::default(), 10).unwrap();

let point = GeoPoint::new(41.9028, 12.4964).unwrap(); // Rome
index.insert(&IndexedValue::Plain(point), 1).unwrap();

let results: Vec<u64> = index.filter(GeoFilterOp::Radius {
    center: GeoPoint::new(41.9, 12.5).unwrap(),
    radius_meters: 10_000.0,
}).iter().collect();
```

## CLI

Build the CLI with:

```sh
cargo build --features cli
```

Commands:

```
oramacore_fields bool check|info|show <path>
oramacore_fields number check|info|search <args>
oramacore_fields geopoint check|info|search <args>
```

## Architecture

All four modules share a common two-layer design:

- **LiveLayer**: In-memory storage for fast inserts and deletes. Thread-safe via `RwLock`.
- **CompactedVersion**: Memory-mapped disk storage loaded via `memmap2`. Swapped atomically using `ArcSwap` for lock-free reads during compaction.
- **Compaction**: Periodically merges live data into a new on-disk version, enabling persistence without blocking reads or writes.

## License

AGPL-3.0