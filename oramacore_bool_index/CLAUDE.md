# CLAUDE.md — oramacore_bool_index

A boolean postings index with layered compaction and memory-mapped storage.

## Build & Test

```bash
cargo build              # dev build
cargo test               # all tests
cargo build --features cli  # includes CLI binary (clap + serde)
```

## Architecture

Two layers serve reads together, compaction merges them periodically:

```
┌─────────────────────────────────┐
│         BoolStorage             │
│                                 │
│  ArcSwap<CompactedVersion>      │  ← mmap'd sorted u64 arrays (immutable after creation)
│  RwLock<LiveLayer>              │  ← in-memory append-only op log
│  Mutex<()>                      │  ← serializes compaction
│  DeletionThreshold              │
└─────────────────────────────────┘
```

**LiveLayer** — a `Vec<LiveOp>` (Insert/Delete) preserving temporal order. `refresh_snapshot()` collapses ops into a `LiveSnapshot` with sorted, deduplicated `true_inserts`, `false_inserts`, `deletes` vecs. The snapshot caches `ops_len` to support position-based drain during compaction.

**CompactedVersion** — read-only mmap'd files (`true.bin`, `false.bin`, `deleted.bin`) containing sorted native-endian `u64` arrays. Versions are immutable: files are never modified after creation.

**FilterData** — combines both layers to answer queries. Merges compacted postings with live inserts, subtracting deletes from both layers.

## Concurrency Model

This is the critical part of the codebase. Three primitives, each with a distinct role:

### `RwLock<LiveLayer>` — read/write access to the op log

- **Writes** (`insert`, `delete`): acquire write lock, push to `ops`, mark `snapshot_dirty`.
- **Reads** (`filter`): use double-check locking pattern:
  1. Acquire read lock. If snapshot is clean → `Arc::clone` the cached snapshot (O(1)).
  2. If dirty → drop read lock, acquire write lock, re-check dirty flag (another thread may have refreshed), refresh if still dirty.

### `ArcSwap<CompactedVersion>` — lock-free version swap

- Readers call `version.load()` to get an `Arc` to the current version — no lock contention.
- Compaction atomically swaps in the new version via `version.store(Arc::new(...))`.

### `Mutex<()>` — compaction serialization

Ensures only one compaction runs at a time. The compaction flow:

1. **Acquire compaction lock.**
2. **Snapshot the live layer** (using the same double-check pattern as `filter()`).
3. **Release all LiveLayer locks** — the op log is free for concurrent inserts/deletes during I/O.
4. **I/O phase** — write new version files to `versions/{n}/`, fsync, atomic-rename CURRENT.
5. **Acquire write lock on LiveLayer.**
6. **Swap version** via `ArcSwap::store`.
7. **Drain compacted ops**: `live.ops.drain(..snapshot.ops_len)`.
8. **Refresh snapshot**, release locks.

### Position-based drain (step 7)

The op log is append-only (`Vec::push`). When the snapshot was taken, it recorded `ops_len`. During the I/O phase (step 4), concurrent writes append new ops at indices `ops_len..`. Draining by position (`..ops_len`) instead of by value ensures concurrent inserts of the *same* doc_id are preserved. This is the core correctness guarantee for concurrent writes during compaction.

## Compaction Strategies

When new deletes exist, compaction chooses between two strategies based on the deletion ratio:

**Strategy A — Apply deletions**: Streams merged postings through `sorted_subtract` to remove deleted IDs. Writes an empty `deleted.bin`. Used when `merged_deletes / total_postings > threshold`.

**Strategy B — Carry forward**: Copies existing postings and appends new inserts. Writes merged deletes to `deleted.bin`. Used when deletion ratio is below threshold.

**Edge case**: When `total_postings == 0` and `merged_deletes > 0`, Strategy A is forced. Without this, stale deletes persist in `deleted.bin` and filter out future re-inserts of the same doc_ids.

When no new deletes exist, compaction uses an optimized copy+append path (no merge needed).

## On-Disk Format

```
{base_path}/
├── CURRENT              # 2-line ASCII: format_version (u32), version_number (u64)
└── versions/
    └── {version_number}/
        ├── true.bin     # sorted native-endian u64 doc_ids
        ├── false.bin    # sorted native-endian u64 doc_ids
        └── deleted.bin  # sorted native-endian u64 doc_ids
```

- All `.bin` files are arrays of `u64::to_ne_bytes()` — native endian for zero-cost mmap reads.
- CURRENT is updated via atomic write-to-temp + rename.
- On Unix, mmap'd files use `MADV_SEQUENTIAL` for kernel read-ahead optimization.

## Key Invariants

1. **All posting lists are sorted and deduplicated** — both in `LiveSnapshot` vecs and on-disk `.bin` files.
2. **Version files are immutable** — once written and fsynced, never modified. Old versions are only deleted after swap.
3. **Op log preserves temporal order** — Insert→Delete→Insert of the same ID must resolve to Insert. The HashSet-based collapsing in `refresh_snapshot()` handles this correctly because it replays ops in order.
4. **`ops_len` in snapshot matches drain count** — the snapshot records exactly how many ops were present when it was taken; compaction drains exactly that many.
5. **Compaction lock scope** — held for the entire compact() call, but LiveLayer write lock is only held briefly at start (snapshot) and end (drain+swap), not during I/O.

## Collapsing Semantics

The `refresh_snapshot()` replay rules:

- `Insert(value, id)` → add to target set (true/false), remove from `delete_set`
- `Delete(id)` → remove from both true and false sets, add to `delete_set`

This means a re-insert after delete correctly "resurrects" the doc_id.
