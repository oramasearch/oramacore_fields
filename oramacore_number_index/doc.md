# oramacore_number_index

A two-layer numeric index (u64 / f64) designed for concurrent reads and writes. Live data sits in memory; compacted data lives on disk via mmap. Compaction merges live into compacted atomically.

## Architecture

- **LiveLayer** (in-memory, behind `RwLock`): append-only ops log (`Vec<LiveOp>`) recording inserts and deletes in chronological order. A cached snapshot (sorted, deduplicated) is rebuilt lazily only when a query needs it.
- **CompactedVersion** (on disk, behind `ArcSwap`): memory-mapped files containing sorted entries grouped by key, plus a sparse header for binary search and a sorted deleted-doc list.
- **NumberIndexer**: parses a JSON value into `IndexedValue::Plain(T)` or `IndexedValue::Array(Vec<T>)`. Array mode creates one entry per element, all sharing the same doc_id.

## Insertion

- Acquires write lock on LiveLayer, appends `LiveOp::Insert { value, doc_id }` (or one per element for arrays), marks snapshot dirty, releases lock. O(1) per op.
- For f64, NaN is rejected at insert time; ordering uses `total_cmp` (so `-0.0 < +0.0`).
- Deletions work the same way: append `LiveOp::Delete { doc_id }`, mark dirty.

## Snapshot Refresh

Triggered lazily before a query when the snapshot is dirty. Single forward pass over the ops log:

- Track the latest op index for each `(value_bytes, doc_id)` insert and each `doc_id` delete.
- Keep an insert only if its index is greater than the latest delete for that doc_id (or no delete exists). This makes delete-then-reinsert work correctly.
- Collect all ever-deleted doc_ids into a set (needed to filter compacted entries).
- Sort surviving inserts by `(value, doc_id)`.

Uses double-check locking: try read lock first; if dirty, drop it, acquire write lock, re-check (another thread may have already refreshed).

## Filter

Given a `FilterOp` (Eq, Gt, Gte, Lt, Lte, BetweenInclusive):

- Extract min/max bounds from the op.
- From the compacted version, iterate entries in `[min, max]` range and filter out deleted doc_ids (from both live deletes and compacted deletes) **before** merging.
  - Filtering deletes before merge ensures that a doc_id re-inserted in the live layer is not incorrectly excluded.
- From the snapshot, take the sorted live inserts.
- Two-pointer sorted merge of both streams in O(N+M), deduplicating equal `(value, doc_id)` pairs.
- Apply the filter op's `matches()` predicate and yield doc_ids.

Sort queries follow the same pattern but iterate all entries (ascending or descending) instead of a range.

## Compacted Version On-Disk Format

Each version is a directory containing:

- **header.idx**: array of `(key, bucket_index, bucket_offset)` entries (24 bytes each). One header entry every ~1000 cumulative doc_ids (sparse index).
- **data_XXXX.dat**: doubly-linked entries. Each entry has a 32-byte header `(key, next_offset, prev_offset, doc_id_count)` followed by the doc_id array. New bucket file created when exceeding target size.
- **deleted.bin**: sorted array of deleted doc_ids (8 bytes each). Pre-loaded into a `HashSet` at open time for O(1) lookup.
- **meta.bin**: `(changes_since_rebuild, total_at_rebuild)` — two u64s driving the incremental-vs-full decision.
- **CURRENT**: text file at the base path pointing to the active version directory. Updated atomically via temp file + rename + fsync.

All integers are native-endian for zero-cost mmap access.

## Compaction

Acquires a dedicated compaction mutex (only one compaction at a time). Takes a snapshot under write lock, then releases all locks — I/O happens without blocking reads or writes.

### Strategy selection

- Compute **dirty ratio** = `changes_since_rebuild / (total_at_rebuild + changes_since_rebuild)`.
- Compute **delete ratio** = `deleted_count / approx_total`.

Three paths:

- **Deletes-only fast path**: no live inserts and delete ratio below threshold. Copies all data files and header as-is; only rewrites `deleted.bin` and `meta.bin`.
- **Incremental compaction**: dirty ratio below threshold, no deletes to apply, and inserts don't span all buckets.
  - Binary-search the header to find the first and last affected buckets.
  - Copy unaffected bucket files verbatim.
  - For each affected bucket: read old entries, filter out entries superseded by live inserts or live deletes, sorted-merge with the relevant slice of live inserts, write the new bucket.
  - For the first affected bucket, the live-insert slice starts at index 0 (not at the bucket's min key) so that inserts before all existing keys are included.
  - Rebuild the header from old (unaffected) + new (affected) entries.
- **Full rebuild** (default fallback): sorted-merge all compacted entries (minus superseded/deleted) with all live inserts. If delete ratio >= threshold, deletes are applied inline (entries physically removed); otherwise deletes are carried forward in `deleted.bin`. Resets compaction metadata.

### Finalization

- Write new version files to disk.
- Sync the version directory.
- Atomically update CURRENT via temp + rename.
- `ArcSwap::store()` the new version (lock-free, readers instantly see it).
- Under write lock, drain `ops[..snapshot.ops_len]` from the live layer. Ops appended concurrently during I/O sit at indices >= `ops_len` and are preserved.

## Optimizations

- **Sparse header + binary search**: O(log H) to locate the starting entry, where H is much smaller than total entry count.
- **mmap with advisory hints**: `MADV_SEQUENTIAL` on data files (kernel read-ahead), `MADV_RANDOM` on header (no unnecessary prefetch).
- **Lazy snapshot**: inserts are O(1) appends; sorting is deferred until a query actually needs it, amortizing the cost across bursts of writes.
- **Two-pointer merge**: O(N+M) with O(1) extra space, deduplicates on the fly.
- **HashSet for deletes**: O(1) per-doc lookup during iteration instead of scanning a sorted list.
- **Incremental compaction**: rewrites only the affected buckets and copies the rest, avoiding a full scan of large indexes.
- **Doubly-linked data entries**: enables bidirectional traversal from any header-located position, so both ascending and descending scans start close to the target.

## Concurrency

Three synchronization primitives:

- `RwLock<LiveLayer>`: multiple concurrent readers or one writer for the live ops log.
- `ArcSwap<CompactedVersion>`: lock-free atomic pointer swap; readers never block on version changes.
- `Mutex<()>`: serializes compactions.

Key properties:

- Queries hold `Arc` references to both the snapshot and the compacted version. Even if a new compaction swaps the version or new writes arrive, active iterators keep using their captured state. The old version is dropped only when all iterators complete.
- Compaction releases all locks before doing I/O, so reads and writes proceed concurrently.
- After compaction, ops are drained by position (`ops.drain(..ops_len)`), not by value, preserving any writes that arrived during the I/O phase.
