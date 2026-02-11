# oramacore_bool_index

On-disk + in-memory boolean postings index with periodic compaction for Rust.

This library provides a high-performance boolean postings index that stores document IDs (u64) partitioned into TRUE and FALSE sets. It's designed for search engine use cases where you need to filter documents by boolean attributes.

## Features

- **Zero-allocation iteration** - Filter iterators borrow directly from source slices with no heap allocations during iteration
- **Hybrid storage** - Combines in-memory LiveLayer for fast writes with memory-mapped CompactedVersion for efficient reads
- **Lock-free version swaps** - Uses ArcSwap for atomic version transitions without blocking readers
- **Streaming compaction** - Intelligent merge strategies with copy+append optimization when possible
- **Concurrent read/write support** - RwLock-protected live layer allows concurrent reads with exclusive writes
- **Crash-safe persistence** - Atomic CURRENT file updates ensure consistent recovery
- **Version cleanup** - Remove old version directories to reclaim disk space

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oramacore_bool_index = "0.1.0"
```

## Usage

```rust
use oramacore_bool_index::{BoolStorage, BoolIndexer, DeletionThreshold};
use std::path::PathBuf;

// Create or open an index with default delete ratio threshold (0.1)
let index = BoolStorage::new(PathBuf::from("/tmp/my_index"), DeletionThreshold::default()).unwrap();

// Use BoolIndexer to extract boolean values from JSON
let indexer = BoolIndexer::new(false);

let value = serde_json::json!(true);
if let Some(indexed) = indexer.index_json(&value) {
    index.update(&indexed, 1);
}

let value = serde_json::json!(true);
if let Some(indexed) = indexer.index_json(&value) {
    index.update(&indexed, 5);
}

let value = serde_json::json!(false);
if let Some(indexed) = indexer.index_json(&value) {
    index.update(&indexed, 2);
}

// For array boolean fields, use BoolIndexer with is_array=true.
// A doc_id appears in TRUE if any element is true,
// and in FALSE if any element is false.
let array_indexer = BoolIndexer::new(true);
let value = serde_json::json!([true, false, true]);
if let Some(indexed) = array_indexer.index_json(&value) {
    index.update(&indexed, 3);
}

// Query documents matching a boolean value
let true_docs: Vec<u64> = index.filter(true).iter().collect();
assert_eq!(true_docs, vec![1, 3, 5]);

// Delete a document (deletes from both TRUE and FALSE sets)
index.delete(1);

// Persist to disk with compaction
index.compact(1).unwrap();

// Clean up old version directories to reclaim disk space
index.cleanup();
```

### Iteration

The `filter()` method returns a `FilterData` struct that produces zero-allocation iterators:

```rust
let filter_data = index.filter(true);

// Create multiple iterators from the same FilterData
for doc_id in filter_data.iter() {
    println!("Doc: {}", doc_id);
}

// Or use IntoIterator
for doc_id in &filter_data {
    println!("Doc: {}", doc_id);
}
```

## Architecture

### Components

```
BoolStorage
├── LiveLayer (RwLock<...>)
│   ├── true_inserts: Vec<u64>    (unsorted)
│   ├── false_inserts: Vec<u64>   (unsorted)
│   ├── deletes: Vec<u64>         (unsorted)
│   └── cached_snapshot: Arc<LiveSnapshot>  (sorted, deduplicated)
│
└── CompactedVersion (ArcSwap<...>)
    ├── true_postings: Option<Mmap>   (memory-mapped, sorted)
    ├── false_postings: Option<Mmap>  (memory-mapped, sorted)
    └── deletes: Option<Mmap>         (memory-mapped, sorted)
```

### LiveLayer

The in-memory layer accepts writes in any order. Data is stored unsorted for O(1) inserts. A cached snapshot is maintained and refreshed lazily when needed for iteration. The snapshot sorts and deduplicates the data.

### CompactedVersion

Represents a persisted version with memory-mapped binary files. Files are mapped read-only and accessed as `&[u64]` slices. Uses `madvise(MADV_SEQUENTIAL)` on Unix for optimized sequential reads.

### Filtering

When you call `filter(value)`, the index:
1. Merges compacted postings with live inserts (sorted merge)
2. Merges compacted deletes with live deletes (sorted merge)
3. Subtracts merged deletes from merged postings (sorted subtract)

All operations are streaming with no intermediate allocations.

### Compaction

Compaction merges the live layer into a new compacted version on disk. It uses two strategies:

1. **No deletions path**: Copies existing files and appends new inserts (fast path when new inserts are all greater than existing max)
2. **With deletions path**: Full merge with optional delete application based on a configurable delete ratio threshold (default: 10%, configurable via `DeletionThreshold`)

### Cleanup

After multiple compactions, old version directories accumulate on disk. Call `cleanup()` to remove all version directories except the current one:

```rust
index.cleanup();
```

Errors during cleanup are logged via `tracing::error!` but do not cause the method to fail, ensuring partial cleanup still succeeds.

## On-Disk Format

```
base_path/
├── CURRENT           # Text file containing format version and version number
└── versions/
    └── {version_number}/
        ├── true.bin      # Little-endian u64 array of TRUE doc_ids
        ├── false.bin     # Little-endian u64 array of FALSE doc_ids
        └── deleted.bin   # Little-endian u64 array of deleted doc_ids
```

### Binary Format

Each `.bin` file contains a sequence of little-endian encoded `u64` values:

```
[doc_id_1: 8 bytes][doc_id_2: 8 bytes][doc_id_3: 8 bytes]...
```

Values within each file are sorted in ascending order.

### CURRENT File

A text file containing the format version and version number of the active version:

```
1
42
```

Where `1` is the format version and `42` is the version number.

Version transitions are atomic via rename: `CURRENT.tmp` -> `CURRENT`.

## Building & Testing

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture
```

## License

MIT
