# CLAUDE.md — oramacore_fields

## What This Project Is

On-disk field indexes for search engines. Six specialized index types (bool, number, string, string_filter, geopoint, vector) supporting full persistence via memory-mapped reads and concurrent access without blocking readers.

License: AGPL-3.0-or-later. Rust 1.88.0, edition 2021.

## Build & Test Commands

```bash
cargo build                          # Build library
cargo test                           # Run all tests
cargo test --test bool_integration   # Run specific test file
cargo clippy                         # Lint
cargo fmt                            # Format
cargo build --features cli           # Build CLI tool
cargo test --features alloc-tests    # Run allocation profiling tests
```

## Project Structure

```
src/
  bool/           # Boolean (true/false) postings index
  number/         # Numeric range index (u64, f64)
  string/         # Full-text string index (BM25 scoring)
  string_filter/  # Exact-match string postings index (FST-backed)
  geopoint/       # Geographic point spatial index (BKD tree)
  vector/         # Vector ANN index (HNSW, quantized)
  bin/cli.rs      # Optional CLI tool (feature-gated)
tests/            # Integration, concurrency, and allocation tests
examples/         # Usage demonstrations per module
```

Each module follows the same internal layout:
- `mod.rs` — Public API (Storage struct, IndexedValue, FilterOp)
- `storage.rs` — Core storage engine (insert/delete/filter/compact)
- `live.rs` — In-memory mutable layer (LiveLayer, LiveSnapshot)
- `compacted.rs` — On-disk immutable layer (CompactedVersion, mmap)
- `error.rs` — Module-specific error types
- `io.rs` — File I/O helpers (version dirs, CURRENT file, atomic writes)
- `indexer.rs` — JSON-to-IndexedValue conversion
- `config.rs` — Threshold and module-specific configuration
- `info.rs` — Info/integrity-check reporting
- `iterator.rs` — Lazy result iterators
- `merge.rs` — Merge logic for compaction
- `platform.rs` — Platform-specific optimizations (madvise, etc.)

## Document ID Invariants

These are fundamental invariants that hold across ALL modules. Never write code that violates them:

1. **Strictly monotonically increasing**: Document IDs are inserted in strictly increasing order. A new doc ID is always greater than all previously inserted doc IDs.
2. **No re-insertion**: Once a document ID is deleted, it is **never** re-added. Deletion is permanent and irreversible.
3. **No updates**: A document ID cannot be updated in place. There is no update operation.
4. **Random deletion order**: Deletions can happen in any order, not necessarily the order of insertion.
5. **Mutual exclusivity between layers**: A document ID exists in either the live layer OR the compacted layer, **never both**. The two layers are mutually exclusive with respect to document IDs.

## Architecture: Shared Two-Layer Pattern

All six modules implement the same concurrency architecture:

```
Storage {
    version: ArcSwap<CompactedVersion>,   // Lock-free pointer to disk data
    live: RwLock<LiveLayer>,              // In-memory mutation buffer
    compaction_lock: Mutex<()>,           // Serialize compactions
}
```

**Read path**: `filter()` merges results from compacted (mmap) + live (snapshot) layers.
**Write path**: `insert()`/`delete()` append to `LiveLayer.ops` under write lock.
**Compaction**: Merges live snapshot into compacted version, swaps atomically.

## Concurrency Model

No async/tokio. Pure synchronous with `std::thread`. Key primitives:

### ArcSwap (lock-free reads)
`ArcSwap<CompactedVersion>` enables readers to call `.load()` without any lock. Writers call `.store()` to atomically publish new compacted versions. Readers holding old `Arc<CompactedVersion>` continue safely — data lives until refcount drops to 0.

### RwLock (live layer protection)
`RwLock<LiveLayer>` allows concurrent readers for filter operations. Writers (insert/delete) take exclusive lock. Contention is low because filter operations only need read lock in steady state.

### Double-Check Locking (snapshot refresh)
All modules use this pattern to minimize write lock acquisition:
```rust
// Fast path: read lock
{ let live = self.live.read().unwrap();
  if !live.is_snapshot_dirty() { return live.get_snapshot(); } }
// Slow path: write lock, re-check
let mut live = self.live.write().unwrap();
if live.is_snapshot_dirty() { live.refresh_snapshot(); }
```
In steady state (no mutations), `filter()` never acquires write lock.

### Compaction isolation
`Mutex<()>` serializes compactions but does NOT block reads or writes. Sequence:
1. Acquire compaction lock
2. Take snapshot under write lock (brief)
3. Release write lock — I/O happens lock-free
4. Build new compacted version on disk
5. `version.store()` atomically swaps in new version
6. Drain compacted ops from live layer (position-based, not value-based)

**Critical invariant**: `ops.drain(..snapshot.ops_len)` preserves concurrent inserts/deletes that arrived during compaction (only the ops captured in the snapshot are drained).

## Edge Cases to Be Aware Of

- **Empty compaction**: If no live changes, compaction is skipped and version number is NOT incremented
- **NaN rejection**: `f64` values are validated; `NaN` is rejected at insert time
- **GeoPoint validation**: Latitude must be -90..=90, longitude must be -180..=180
- **Deletion threshold**: Controls whether compaction applies deletes (rewrites data) or carries them forward (cheaper but slower reads). Ratio = deletes / total entries
- **Vector dimension validation**: Dimensions must be 1..=4096; dimension mismatch at insert time is rejected
- **Vector multi-segment**: Uses segment-based architecture with configurable max nodes per segment; segments are rebuilt when deletion or insertion thresholds are exceeded
- **Format versioning**: CURRENT file stores `(format_version, version_number)`. Mismatched format_version errors immediately

## Key Dependencies

- `arc-swap` — Lock-free atomic pointer swap (central to compaction)
- `memmap2` — Memory-mapped file I/O for zero-copy reads
- `fst` — Finite State Transducer for string_filter key lookup
- `xtri` — Trie for string module prefix/fuzzy matching
- `anyhow` — Error propagation (bool module)
- `serde_json` — JSON helpers for indexing
- `tracing` — Structured logging
- `rand` — Random number generation (HNSW level selection, etc.)
- `libc` — Unix-specific madvise calls (unix only)

## Conventions

- All query results are sorted (binary-searchable)
- No `unwrap()` in public APIs — all errors are `Result` types
- Doc IDs are `u64`
- Persistence uses append-only version directories with atomic CURRENT file swaps
- All modules implement `.integrity_check()` for validation
- Tests follow pattern: `tests/{module}_integration_tests.rs`, `tests/{module}_concurrency_tests.rs`, `tests/{module}_alloc_count.rs`
