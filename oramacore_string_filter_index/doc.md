# oramacore_string_filter_index

Exact-match string postings index that maps `(key: String, doc_id: u64)` pairs into sorted sets. Two-layer design: a fast in-memory live layer and a compacted on-disk layer backed by an FST (finite state transducer) and memory-mapped postings.

## Data Model

All posting lists are sorted, deduplicated arrays of `u64`. On disk, keys are compressed into an FST (`fst::Map`) mapping each string key to a byte offset into a packed postings file. Postings and deletes are stored as native-endian byte arrays for zero-cost mmap reads.

- **LiveLayer** — append-only `Vec<LiveOp>` (Insert/Delete ops) preserving temporal order. A cached `LiveSnapshot` collapses the log into a columnar layout on demand: sorted `keys`, a `ranges` vector for O(log n) key lookup via binary search, and a flat `doc_ids` array grouped by key.
- **CompactedVersion** — immutable mmap'd files (`keys.fst`, `postings.dat`, `deleted.bin`). Never modified after creation; new versions are written to a new directory.

## Insertion

O(1) amortized. Acquires write lock on the live layer, pushes a `LiveOp::Insert(key, doc_id)` to the ops vec, marks the snapshot dirty. For array-typed fields, one insert per element. No sorting happens here.

Deletion works the same way: pushes `LiveOp::Delete(doc_id)`. A delete removes the doc_id from all keys it appeared under, not just a specific key.

## Snapshot Refresh

When the snapshot is dirty, `refresh_snapshot()` replays the entire ops log in a single forward pass:

- Build a map of `doc_id -> Vec<key>` for inserts and a set of deleted doc_ids, tracking op indices.
- An insert is kept only if its index is greater than the latest delete for that doc_id (or no delete exists). This makes delete-then-reinsert resolve correctly.
- Flatten surviving inserts into `(key, doc_id)` pairs, sort by `(key, doc_id)`, deduplicate.
- Build the columnar layout: sorted unique `keys`, a `ranges` vector (`ranges[i]..ranges[i+1]` indexes into `doc_ids` for key `i`), and the flat `doc_ids` array.
- Collect all ever-deleted doc_ids into both a `HashSet` (for O(1) membership) and a sorted `Vec` (for merge operations).
- Record `ops_len` for use during compaction drain.

## Filter

`filter(key)` returns a `FilterData` holding `Arc<CompactedVersion>` + `Arc<LiveSnapshot>` + the key. The iterator pipeline is:

```
result = subtract(
    merge(compacted_postings, live_inserts),
    merge(compacted_deletes,  live_deletes),
)
```

Compacted postings are found via FST lookup: the key maps to a byte offset into `postings.dat`, where the count and doc_ids are read directly from the mmap. Live inserts are found via binary search on the snapshot's sorted keys vector, then slicing into the doc_ids range.

Both merge and subtract are two-pointer streaming iterators, O(n+m) time, O(1) extra space. Merge deduplicates on the fly via a `last_emitted` tracker. Subtract advances the right pointer to catch up, skipping matched values.

### Snapshot Acquisition (double-check locking)

- Acquire read lock. If snapshot is clean, `Arc::clone` it — O(1).
- If dirty: drop read lock, acquire write lock, re-check dirty flag (another thread may have refreshed), refresh if still dirty.
- Load `CompactedVersion` from ArcSwap — lock-free.

The returned `FilterData` owns Arcs to both layers, so active iterators are immune to concurrent compactions or writes.

## Compacted Version On-Disk Format

Each version is a directory containing:

- **keys.fst**: FST binary mapping string keys to byte offsets into postings.dat. Keys must be inserted in lexicographic order during construction. Provides O(log n) lookup.
- **postings.dat**: contiguous packed posting lists. Each list is `[count: u64 LE][doc_id_0: u64 LE]...[doc_id_{count-1}: u64 LE]`. The FST value for a key is the byte offset of its count field.
- **deleted.bin**: sorted array of deleted doc_ids (8 bytes each). Empty when deletes are applied; populated when carried forward.
- **CURRENT**: text file at the base path pointing to the active version directory (format version on line 1, version number on line 2). Updated atomically via temp file + rename + fsync.

All integers are little-endian for zero-cost mmap access.

## Compaction

Merges live layer into the compacted version and writes new on-disk files. The flow:

- Acquire `compaction_lock` (Mutex, serializes compactions)
- Snapshot the live layer (records `ops_len`)
- Early return if nothing to compact (drain ops, shrink vec, refresh)
- **I/O phase** — write new version files to `versions/{n}/`. No locks held, reads and writes continue freely.
- Fsync, atomically update CURRENT file (write-to-temp + rename)
- Acquire write lock on LiveLayer:
  - `ArcSwap::store` the new version (lock-free swap for readers)
  - `ops.drain(..snapshot.ops_len)` — position-based, not value-based
  - Refresh snapshot
- Release locks

### Position-based drain

The op log is append-only. During the I/O phase, concurrent writes append new ops at indices `ops_len..`. Draining by position (`..ops_len`) instead of by value ensures concurrent inserts of the same doc_id are preserved. This is the core correctness guarantee for concurrent writes during compaction.

### Builder: `build_from_sorted_sources`

Streaming three-way merge on keys from the compacted iterator and live iterator (both in lexicographic order):

1. Peek both iterators, compare current keys.
2. Smaller key: emit that entry and advance. Equal keys: merge their doc_id lists and advance both.
3. For each entry: if a deletion set is provided, filter out deleted doc_ids. Skip entries that become empty.
4. Write to FST builder (`key -> current_offset`) and append `[count][doc_ids...]` to postings file.
5. `merge_sorted_u64_into` handles doc_id list merging: two-pointer merge with dedup into a reusable buffer, O(n+m), zero allocation after warmup.

### Compaction Strategies

Strategy chosen by deletion ratio (`merged_deletes / total_postings`):

- **Strategy A** (ratio > threshold): apply deletions. The builder receives the merged deletion set and physically filters doc_ids out of postings. Writes empty `deleted.bin`. More expensive but cleans up disk and speeds up future queries.
- **Strategy B** (ratio <= threshold): carry forward. The builder writes all doc_ids to postings unfiltered. Writes merged deletes to `deleted.bin`. Faster compaction but reads pay the subtract cost.

## Optimizations

- **FST compression**: String keys compressed into a finite state transducer — much smaller than storing raw strings, with O(log n) lookup.
- **Lock-free reads** (`ArcSwap`): version swaps don't block readers. Readers load via atomic operation.
- **Double-check locking**: snapshot refresh minimizes write lock contention on the read path.
- **Columnar snapshot**: keys and doc_ids separated; binary search on keys gives O(log k) key lookup, then direct slice into doc_ids.
- **Zero-allocation queries**: merge and subtract iterators work on borrowed slices with no intermediate vecs.
- **Memory-mapped I/O**: postings.dat and deleted.bin are never loaded into heap. OS page cache is used. Sequential prefetch hints on Linux via `madvise(MADV_SEQUENTIAL)`.
- **Reusable buffers**: `merge_sorted_u64_into` reuses its output buffer across entries during compaction, avoiding repeated allocation.
- **Lazy snapshot**: inserts are O(1) appends; sorting is deferred until a query actually needs it, amortizing the cost across bursts of writes.

## Concurrency

Three synchronization primitives:

- `RwLock<LiveLayer>`: multiple concurrent readers or one writer for the live ops log.
- `ArcSwap<CompactedVersion>`: lock-free atomic pointer swap; readers never block on version changes.
- `Mutex<()>`: serializes compactions.

Key properties:

- Queries hold `Arc` references to both the snapshot and the compacted version. Even if a new compaction swaps the version or new writes arrive, active iterators keep using their captured state. The old version is dropped only when all iterators complete.
- Compaction releases all locks before doing I/O, so reads and writes proceed concurrently.
- After compaction, ops are drained by position (`ops.drain(..ops_len)`), not by value, preserving any writes that arrived during the I/O phase.
