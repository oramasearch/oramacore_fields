# oramacore_bool_index

Boolean postings index that maps `doc_id: u64` into TRUE and FALSE sorted sets. Two-layer design: a fast in-memory live layer and a compacted on-disk layer served via mmap.

## Data Model

All posting lists (TRUE set, FALSE set, deletes) are sorted, deduplicated arrays of `u64`. On disk they're stored as native-endian byte arrays for zero-cost mmap reads.

- **LiveLayer** — append-only `Vec<LiveOp>` (Insert/Delete ops) preserving temporal order. A cached `LiveSnapshot` collapses the log into sorted vecs on demand.
- **CompactedVersion** — immutable mmap'd files (`true.bin`, `false.bin`, `deleted.bin`). Never modified after creation; new versions are written to a new directory.

## Insertion

O(1) amortized. Acquires write lock on the live layer, pushes a `LiveOp::Insert(bool, doc_id)` to the ops vec, marks the snapshot dirty. No sorting happens here.

Deletion works the same way: pushes `LiveOp::Delete(doc_id)`.

## Snapshot Refresh

When the snapshot is dirty, `refresh_snapshot()` replays the entire ops log through three HashSets to compute the final state:

- `Insert(value, id)` — adds to target set (true/false), removes from delete_set
- `Delete(id)` — removes from both true/false sets, adds to delete_set

This preserves temporal ordering: insert, delete, re-insert of the same id correctly resolves to insert. After replay, HashSets are converted to sorted Vecs. The snapshot also records `ops_len` (the number of ops at the time of creation) for use during compaction drain.

## Filter

`filter(value)` returns a `FilterData` holding `Arc<CompactedVersion>` + `Arc<LiveSnapshot>`. The iterator pipeline is:

```
result = subtract(
    merge(compacted_postings, live_inserts),
    merge(compacted_deletes,  live_deletes),
)
```

Both merge and subtract are two-pointer streaming iterators, O(n+m) time, O(1) extra space. Merge deduplicates on the fly via a `last_emitted` tracker. Subtract advances the right pointer to catch up, skipping matched values. Descending variants mirror the logic with reversed comparisons.

### Snapshot Acquisition (double-check locking)

- Acquire read lock. If snapshot is clean, `Arc::clone` it — O(1).
- If dirty: drop read lock, acquire write lock, re-check dirty flag (another thread may have refreshed), refresh if still dirty.
- Load `CompactedVersion` from ArcSwap — lock-free.

The returned `FilterData` owns Arcs to both layers, so active iterators are immune to concurrent compactions or writes.

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

### Compaction Strategies

**No new deletes** — optimized copy+append path:
- If all new doc_ids > existing max: copy existing file + append new bytes (fast, O(m))
- Otherwise: full two-pointer merge of existing + new (O(n+m))
- Existing `deleted.bin` is copied as-is

**With new deletes** — strategy chosen by deletion ratio (`merged_deletes / total_postings`):
- **Strategy A** (ratio > threshold): apply deletions. Writes `subtract(merge(existing, new), merge(existing_deletes, new_deletes))` for each set. Writes empty `deleted.bin`. More expensive but cleans up disk.
- **Strategy B** (ratio <= threshold): carry forward. Same copy/merge logic as the no-delete path for postings. Writes merged deletes to `deleted.bin`. Faster compaction but reads pay the subtract cost.
- **Edge case**: when `total_postings == 0` and deletes exist, Strategy A is forced. Without this, stale deletes persist and filter out future re-inserts.

## Concurrency Summary

- `RwLock<LiveLayer>` — protects the op log. Writes are exclusive, reads are shared. Double-check locking minimizes write lock contention on the read path.
- `ArcSwap<CompactedVersion>` — lock-free version reads. Compaction atomically swaps in the new version. Readers holding old Arcs continue safely until they drop.
- `Mutex<()>` — prevents concurrent compactions. Held for the entire `compact()` call, but the LiveLayer write lock is only held briefly at the start (snapshot) and end (drain+swap), not during I/O.
