# oramacore_number_index

A high-performance, thread-safe numerical index for range queries over `u64` and `f64` values.

## Overview

This crate provides a concurrent number index with:

- Support for both `u64` and `f64` value types
- Range queries (eq, gt, gte, lt, lte, between)
- JSON indexing via `NumberIndexer` (plain values and arrays)
- Two-layer architecture (LiveLayer + CompactedVersion)
- Lock-free reads during compaction
- Memory-mapped disk storage

## Features

- **JSON Indexing**: Extract values from JSON with `NumberIndexer`, supporting both scalar and array fields
- **Range Queries**: Filter by equality, greater/less than, or between values
- **Sorting**: Retrieve doc_ids sorted by their indexed values (ascending or descending)
- **Concurrency**: Multiple readers with single writer, lock-free version swaps
- **Persistence**: Memory-mapped files for efficient disk storage
- **Compaction**: Background merging with configurable garbage collection threshold

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oramacore_number_index = "0.1.0"
```

## Quick Start

Use `NumberIndexer` to extract values from JSON and `NumberStorage::insert` to index them:

```rust
use oramacore_number_index::{NumberIndexer, NumberStorage, FilterOp, Threshold};
use serde_json::json;
use std::path::PathBuf;

// Create an index and an indexer for plain u64 values
let index: NumberStorage<u64> = NumberStorage::new(
    PathBuf::from("/tmp/my_index"),
    Threshold::default(),
)?;
let indexer = NumberIndexer::<u64>::new(false);

// Index JSON values
let val = indexer.index_json(&json!(10)).unwrap();
index.insert(&val, 1)?;

let val = indexer.index_json(&json!(20)).unwrap();
index.insert(&val, 2)?;

let val = indexer.index_json(&json!(30)).unwrap();
index.insert(&val, 3)?;

// Query for values >= 15
let results: Vec<u64> = index.filter(FilterOp::Gte(15)).iter().collect();
assert_eq!(results, vec![2, 3]);

// Delete a document and compact
index.delete(2);
index.compact(1)?;
```

### Array Fields

Documents with array-valued fields (e.g., `tags: [10, 20, 30]`) are indexed once per
element, so a query matching any element finds the document:

```rust
use oramacore_number_index::{NumberIndexer, NumberStorage, FilterOp, Threshold};
use serde_json::json;
use std::path::PathBuf;

let index: NumberStorage<u64> = NumberStorage::new(
    PathBuf::from("/tmp/my_index"),
    Threshold::default(),
)?;

// Use is_array=true for array fields
let indexer = NumberIndexer::<u64>::new(true);

// Doc 1 has tags [10, 20, 30]
let val = indexer.index_json(&json!([10, 20, 30])).unwrap();
index.insert(&val, 1)?;

// Doc 2 has tags [20, 40]
let val = indexer.index_json(&json!([20, 40])).unwrap();
index.insert(&val, 2)?;

// Query: find docs with tag == 20
let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
assert_eq!(results, vec![1, 2]); // Both docs have 20

// Query: find docs with tag == 10
let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
assert_eq!(results, vec![1]); // Only doc 1 has 10
```

### f64 Index

```rust
use oramacore_number_index::{NumberIndexer, NumberStorage, FilterOp, Threshold};
use serde_json::json;
use std::path::PathBuf;

let index: NumberStorage<f64> = NumberStorage::new(
    PathBuf::from("/tmp/my_f64_index"),
    Threshold::default(),
)?;
let indexer = NumberIndexer::<f64>::new(false);

let val = indexer.index_json(&json!(3.14)).unwrap();
index.insert(&val, 1)?;

let val = indexer.index_json(&json!(-1.5)).unwrap();
index.insert(&val, 2)?;

// Query: values >= 0.0
let results: Vec<u64> = index.filter(FilterOp::Gte(0.0)).iter().collect();
assert_eq!(results, vec![1]);
```

### Sorting

```rust
use oramacore_number_index::{NumberIndexer, NumberStorage, SortOrder, Threshold};
use serde_json::json;
use std::path::PathBuf;

let index: NumberStorage<u64> = NumberStorage::new(
    PathBuf::from("/tmp/my_index"),
    Threshold::default(),
)?;
let indexer = NumberIndexer::<u64>::new(false);

for (value, doc_id) in [(30, 3), (10, 1), (20, 2)] {
    let val = indexer.index_json(&json!(value)).unwrap();
    index.insert(&val, doc_id)?;
}

let ascending: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
assert_eq!(ascending, vec![1, 2, 3]); // doc_ids ordered by value: 10, 20, 30

let descending: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();
assert_eq!(descending, vec![3, 2, 1]); // doc_ids ordered by value: 30, 20, 10
```

## API Reference

### Key Types

| Type | Description |
|------|-------------|
| `NumberStorage<T>` | Main index type, generic over `u64` or `f64` |
| `U64Storage` | Type alias for `NumberStorage<u64>` |
| `F64Storage` | Type alias for `NumberStorage<f64>` |
| `NumberIndexer<T>` | Extracts values from JSON (plain or array) |
| `IndexedValue<T>` | Extracted value: `Plain(T)` or `Array(Vec<T>)` |
| `Threshold` | Compaction threshold (0.0-1.0, default 0.1) |
| `FilterOp<T>` | Filter operation enum for queries |
| `FilterHandle<T>` | Query result wrapper |
| `SortHandle<T>` | Sort result wrapper |

### NumberIndexer

```rust
// Plain scalar fields (e.g., "price": 9.99)
let indexer = NumberIndexer::<f64>::new(false);
let value = indexer.index_json(&json!(9.99)); // Some(IndexedValue::Plain(9.99))

// Array fields (e.g., "tags": [1, 2, 3])
let indexer = NumberIndexer::<u64>::new(true);
let value = indexer.index_json(&json!([1, 2, 3])); // Some(IndexedValue::Array([1, 2, 3]))

// Invalid input returns None
let indexer = NumberIndexer::<u64>::new(false);
let value = indexer.index_json(&json!("hello")); // None
```

### NumberStorage Methods

```rust
// Construction
NumberStorage::new(path, threshold) -> Result<Self, Error>
NumberStorage::new_with_config(path, threshold, index_stride, bucket_target_bytes) -> Result<Self, Error>

// Indexing (preferred)
index.insert(&indexed_value, doc_id) -> Result<(), Error>

// Low-level insert
index.insert(value, doc_id) -> Result<(), Error>

// Deletion
index.delete(doc_id)

// Queries
index.filter(FilterOp<T>) -> FilterHandle<T>
index.sort(SortOrder) -> SortHandle<T>

// Maintenance
index.compact(offset) -> Result<(), Error>
index.cleanup()
index.current_offset() -> u64
index.info() -> IndexInfo
index.integrity_check() -> IntegrityCheckResult
```

### FilterOp Variants

```rust
pub enum FilterOp<T> {
    Eq(T),                    // value == target
    Gt(T),                    // value > target
    Gte(T),                   // value >= target
    Lt(T),                    // value < target
    Lte(T),                   // value <= target
    BetweenInclusive(T, T),   // min <= value <= max
}
```

### Threshold

Controls when compaction applies deletions vs. carrying them forward:

```rust
// Default: 10% - apply deletions when delete ratio >= 10%
let threshold = Threshold::default();

// Custom: 50% threshold
let threshold = Threshold::try_new(0.5)?;
```

## Architecture

The index uses a two-layer design:

```
┌─────────────────────────────────────────┐
│              NumberStorage                │
├─────────────────────────────────────────┤
│  RwLock<LiveLayer>                      │  ← In-memory inserts/deletes
│    - inserts: Vec<(T, u64)>             │
│    - deletes: HashSet<u64>              │
│    - snapshot: Arc<LiveSnapshot>        │
├─────────────────────────────────────────┤
│  ArcSwap<CompactedVersion>              │  ← Memory-mapped disk storage
│    - header.idx (binary search)         │
│    - data_XXXX.dat (entries)            │
│    - deleted.bin (deleted doc_ids)      │
└─────────────────────────────────────────┘
```

**LiveLayer**: Fast in-memory storage for new inserts and deletes. Uses lazy snapshot caching to amortize sorting costs.

**CompactedVersion**: Memory-mapped disk storage for efficient range queries. Uses fixed 16-byte entries (8 bytes value + 8 bytes doc_id).

## On-Disk Format

```
<base_path>/
├── CURRENT              # Points to current version offset
└── versions/
    └── <offset>/
        ├── header.idx   # Sparse index for binary search
        ├── data_0000.dat  # Entry data (grouped by key)
        ├── data_0001.dat  # Additional buckets (1GB each)
        └── deleted.bin  # Sorted deleted doc_ids
```

### File Formats

**header.idx**: Array of `HeaderEntry` (24 bytes each)
- `key: T` (8 bytes) - indexed value
- `bucket_index: u64` - which data file
- `bucket_offset: u64` - byte offset in data file

**data_XXXX.dat**: Sequence of entries with navigation pointers
- `DataEntryHeader` (32 bytes): key, next_offset, prev_offset, count
- `doc_ids`: array of `u64` values

**deleted.bin**: Sorted array of deleted `u64` doc_ids

## Concurrency Model

The index achieves thread safety through:

- `RwLock<LiveLayer>`: Multiple concurrent readers OR one exclusive writer
- `ArcSwap<CompactedVersion>`: Lock-free atomic version swap for readers
- `Mutex<()>`: Prevents concurrent compactions

Key patterns:

- **Double-check pattern**: Efficient snapshot refresh during queries
- **Clear-after-swap**: Preserves writes during compaction

## Notes

- **f64 NaN rejection**: Inserting `f64::NAN` returns an error
- **f64 ordering**: Uses `total_cmp` for consistent ordering (treats `-0.0 < +0.0`)
- **Multi-value docs**: Array fields index one entry per value; a doc may appear multiple times in filter results (once per matching value)
- **Compaction offset**: Caller provides the version offset; typically incrementing
- **Cleanup**: Call `cleanup()` to remove old version directories after compaction
