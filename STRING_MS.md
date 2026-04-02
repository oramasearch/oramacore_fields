# Multi-Segment Architecture for String Module

## Context

The string module's posting data (`postings.dat`) grows huge as documents accumulate. Currently, every compaction rewrites the entire FST + postings + doc_lengths into a single set of files per version. This is expensive and doesn't scale.

The embedding module already implements multi-segment successfully. We replicate a similar pattern: split data into multiple immutable segments (each with its own FST + postings + doc_lengths), stored in a shared `segments/` directory, referenced by a manifest per version. New data is only merged into the last segment during compaction. One global tomb file (`deleted.bin`) per version. Global stats (`global_info.bin`) per version.

## Target On-Disk Layout

```
base_path/
├── CURRENT                          # format_version(2) + version_number
├── segments/seg_<id>/               # IMMUTABLE, shared across versions
│   ├── keys.fst                     # Term → byte offset in this segment's postings.dat
│   ├── postings.dat                 # Posting lists for this segment
│   └── doc_lengths.dat              # (doc_id, field_length) for this segment's docs
└── versions/<version>/
    ├── manifest.json                # [{segment_id, num_postings, num_deletes, min_doc_id, max_doc_id, total_doc_length, total_documents}]
    ├── deleted.bin                  # ONE global sorted delete list
    └── global_info.bin              # (total_document_length, total_documents) — surviving docs only
```

## Implementation Plan

### Step 1: `config.rs` — Add SegmentConfig

Add `SegmentConfig` struct:
```rust
pub struct SegmentConfig {
    pub max_postings_per_segment: usize,  // default: 5_000_000
    pub deletion_threshold: Threshold,     // reuse existing Threshold
}
```

Keep existing `Threshold` and `Bm25Params` unchanged.

### Step 2: `io.rs` — Segment I/O and manifest

- Bump `FORMAT_VERSION` from `1` to `2`
- Add `segment_data_dir(base_path, segment_id) -> PathBuf` — `base_path/segments/seg_{id}`
- Add `ensure_segment_dir(base_path, segment_id) -> Result<PathBuf>`
- Add `list_segment_dirs(base_path) -> Result<Vec<u64>>`
- Add `ManifestEntry` struct:
  ```rust
  pub struct ManifestEntry {
      pub segment_id: u64,
      pub num_postings: usize,
      pub num_deletes: usize,
      pub min_doc_id: u64,
      pub max_doc_id: u64,
      pub total_doc_length: u64,
      pub total_documents: u64,
  }
  ```
- Add `write_manifest(version_dir, &[ManifestEntry])` and `read_manifest(version_dir) -> Vec<ManifestEntry>`
  - **Invariant:** Manifest entries MUST be ordered by `min_doc_id` (ascending). `write_manifest` asserts this ordering at write time. `read_manifest` validates it at load time and returns an error if violated. This ordering is critical for correctness: search cursors (delete cursors, live delete cursors) advance monotonically across segments and must never go backwards.
- Keep all existing functions unchanged (`write_deleted`, `write_doc_lengths`, `write_global_info`, etc.)

### Step 3: `compacted.rs` — Extract Segment, create SegmentList

**New `Segment` struct** — extracted from `CompactedVersion`, holds one (FST, postings, doc_lengths) triple:
```rust
pub struct Segment {
    pub segment_id: u64,
    fst_map: Map<Mmap>,
    postings_mmap: Mmap,
    doc_lengths_mmap: Mmap,
    pub num_postings: usize,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
}
```

`Segment` gets all query methods currently on `CompactedVersion`:
- `lookup_postings()`, `for_each_term_match()`, `field_length_galloping()`
- `iter_terms()`, `iter_doc_lengths()`, `total_postings()`, `term_count()`
- `Segment::load(base_path, &ManifestEntry)` — loads from `segments/seg_{id}/`

**New `SegmentList`** — replaces `CompactedVersion`:
```rust
pub struct SegmentList {
    pub segments: Vec<Segment>,
    pub version_number: u64,
    deleted_mmap: Option<Mmap>,
    pub total_document_length: u64,
    pub total_documents: u64,
}
```

Methods: `empty()`, `load(base_path, version_number)`, `deletes_slice()`, `has_data()`

**Refactor `build_from_sorted_sources()` → `build_segment_data()`** — Do NOT create a separate function. The current `build_from_sorted_sources()` (lines 366-648) contains ~250 lines of complex streaming sorted merge logic across terms and doc_lengths. Duplicating it would be a maintenance hazard. Instead, refactor as follows:

1. Extract the core merge logic into `build_segment_data()`:
   ```rust
   pub struct SegmentBuildResult {
       pub num_postings: usize,
       pub total_doc_length: u64,
       pub total_documents: u64,
       pub min_doc_id: u64,
       pub max_doc_id: u64,
   }

   pub fn build_segment_data<'a, P: Deref<Target = [u32]>>(
       compacted_terms: &mut CompactedTermIterator<'a>,
       live_terms: &[(&str, &[(u64, P, P)])],
       compacted_doc_lengths: &mut DocLengthIterator<'_>,
       live_doc_lengths: &[(u64, u16)],
       deleted_set: Option<&[u64]>,
       segment_dir: &Path,
   ) -> Result<SegmentBuildResult>
   ```
   This writes `keys.fst`, `postings.dat`, `doc_lengths.dat` to `segment_dir` and returns per-segment metadata. The `count_exclude_set` parameter is dropped: for a segment build, stats always match what's written (docs excluded by `deleted_set` are excluded from both output and stats).

2. `merge_and_write_doc_lengths()` gains return of `(min_doc_id, max_doc_id)` in addition to `(total_doc_length, total_documents)`.

3. The old `build_from_sorted_sources()` is removed entirely. Callers in `storage.rs` invoke `build_segment_data()` for each segment, then write `deleted.bin` and `global_info.bin` separately at the version level. Global stats are computed from manifest entry sums minus global deletes (see Step 4, point 7).

**Why this works:** The `count_exclude_set` parameter currently exists to decouple "what gets written" from "what gets counted" during carry-forward compaction (postings carry deleted docs, but stats exclude them). In multi-segment, this decoupling moves up: segments always have accurate per-segment stats, and the version-level `global_info.bin` is computed from manifest metadata minus global deletes — no need for dual cursors inside the merge loop.

**Keep:** `PostingsReader`, `PostingEntryRef`, `CompactedTermIterator`, `DocLengthIterator`, `SortedDeleteCursor`, `write_entry_to_buf`, `flush_term_buf`, `merge_and_write_doc_lengths` (with extended return type)

**Remove:** `CompactedVersion` struct (replaced by `SegmentList` + `Segment`), `build_from_sorted_sources()` (replaced by `build_segment_data()`)

### Step 4: `storage.rs` — Multi-segment compaction

**Struct change:**
```rust
pub struct StringStorage {
    base_path: PathBuf,
    segments: ArcSwap<SegmentList>,      // was: version: ArcSwap<CompactedVersion>
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    segment_config: SegmentConfig,
}
```

**Constructor:** `new(path, segment_config: SegmentConfig)` — loads `SegmentList::load()` instead of `CompactedVersion::load()`. Detects v1 format and auto-migrates (see Migration section below).

**Compaction algorithm:**
1. Take live snapshot
2. Load current SegmentList
3. `next_seg_id = max(existing segment_ids) + 1`
4. Compute merged global deletes = union(current.deletes + snapshot.deletes)
5. For each existing segment:
   - Count deletes in this segment's doc_id range
   - `deletion_ratio = segment_deletes / segment.num_postings`
   - **Last segment + live data exists + combined fits in max_postings**: REBUILD with merge (old segment data + live data), filter deletes → new segment_id. Mark `live_absorbed = true`
   - **Last segment + live data exists + combined exceeds max_postings**: REBUILD last segment filtering deletes only → new segment_id if deletion threshold exceeded, else carry forward. Then create NEW segment from live data alone
   - **deletion_ratio > threshold**: REBUILD segment filtering deletes → new segment_id
   - **Otherwise**: CARRY FORWARD (same segment_id in manifest, update num_deletes)
6. If live not absorbed: create new segment from live data only
7. Compute global stats: scan all segments' doc_lengths, skip global deletes, accumulate totals
8. Write manifest.json, deleted.bin, global_info.bin to new version dir
9. Atomically swap SegmentList, drain live ops

**Cleanup:** Extended to remove unreferenced segment directories from `segments/` (same pattern as embedding module).

### Step 5: `iterator.rs` — Search across segments

**SearchHandle change:**
```rust
pub struct SearchHandle {
    segments: Arc<SegmentList>,  // was: version: Arc<CompactedVersion>
    snapshot: Arc<LiveSnapshot>,
}
```

**`corpus_stats()`:** Uses `self.segments.total_documents` and `self.segments.total_document_length` (same logic, different field name).

**`accumulate_token()`:** Loop over `self.segments.segments` instead of single version:
```rust
for segment in &self.segments.segments {
    segment.for_each_term_match(token, tolerance, |is_exact, reader| {
        // Same inner loop, but:
        // - compacted_del_cursor uses self.segments.deletes_slice() (global)
        // - fl_cursor resets per segment (each has own doc_lengths)
        // - live_del_cursor does NOT reset between segments (doc_ids increase across segments)
        // - compacted_del_cursor does NOT reset between segments
    })?;
}
```

Key invariant: segments are ordered by min_doc_id (monotonically increasing doc_ids). Delete cursors advance across segments without resetting. But `fl_cursor` for `field_length_galloping` resets per segment.

**`execute_with_phrase_boost()`:** Same pattern — loop over segments for the compacted layer search. The `curr_raw` position buffer collects from all segments. Everything else (live layer search, adjacency check, scoring) stays the same.

### Step 6: `info.rs` — Update IndexInfo

```rust
pub struct IndexInfo {
    pub format_version: u32,
    pub current_version_number: u64,
    pub version_dir: PathBuf,
    pub num_segments: usize,              // NEW
    pub unique_terms_count: usize,        // sum across segments (may overcount shared terms)
    pub total_postings_count: usize,      // sum across segments
    pub total_documents: u64,
    pub avg_field_length: f64,
    pub deleted_count: usize,
    pub total_segments_size_bytes: u64,   // NEW: sum of all segment file sizes
    pub deleted_size_bytes: u64,
    pub global_info_size_bytes: u64,
    pub pending_ops: usize,
}
```

Remove old per-file size fields (`fst_size_bytes`, `postings_size_bytes`, `doc_lengths_size_bytes`).

### Step 7: `mod.rs` — Update public API

- Export `SegmentConfig` from config
- `StringStorage::new` takes `SegmentConfig` instead of `Threshold`
- Keep backward compat: could add `StringStorage::new_with_threshold()` that wraps into default SegmentConfig, or just change the signature (breaking change)

### Step 8: No changes needed

- `live.rs` — entirely in-memory, unaware of segments
- `merge.rs` — generic `SortedMerge` utility, still used for delete set merging
- `bm25.rs`, `scoring.rs`, `simd.rs`, `indexer.rs`, `platform.rs` — unaffected

## Files to Modify

| File | Change |
|------|--------|
| `src/string/config.rs` | Add `SegmentConfig` |
| `src/string/io.rs` | Bump FORMAT_VERSION, add segment dirs + manifest I/O |
| `src/string/compacted.rs` | Extract `Segment` from `CompactedVersion`, create `SegmentList`, add `build_segment_from_sorted_sources` |
| `src/string/storage.rs` | Replace `ArcSwap<CompactedVersion>` with `ArcSwap<SegmentList>`, rewrite compaction, update cleanup/info/integrity_check, add v1→v2 migration |
| `src/string/iterator.rs` | Update `SearchHandle` to use `SegmentList`, loop over segments in `accumulate_token` and `execute_with_phrase_boost` |
| `src/string/info.rs` | Update `IndexInfo` fields |
| `src/string/mod.rs` | Update exports |

## V1 → V2 Migration (existing data)

When `StringStorage::new()` reads `CURRENT` and finds `format_version == 1`, it must transparently migrate existing data to the v2 multi-segment layout. This happens once, in-place, before the storage becomes usable.

**Migration steps** (in a new `migrate_v1_to_v2` function in `storage.rs`):

1. Read CURRENT → get `(1, version_number)`
2. Determine the v1 version dir: `versions/{version_number}/`
3. Verify v1 files exist: `keys.fst`, `postings.dat`, `doc_lengths.dat`, `deleted.bin`, `global_info.bin`
4. Create segment directory: `segments/seg_0/`
5. **Move** (rename) the three data files into the segment:
   - `versions/{v}/keys.fst` �� `segments/seg_0/keys.fst`
   - `versions/{v}/postings.dat` → `segments/seg_0/postings.dat`
   - `versions/{v}/doc_lengths.dat` → `segments/seg_0/doc_lengths.dat`
6. Compute segment metadata by scanning the moved files:
   - `num_postings`: count total posting entries (scan FST, sum doc_counts)
   - `min_doc_id` / `max_doc_id`: read first and last entry from `doc_lengths.dat`
   - `total_doc_length` / `total_documents`: read from existing `global_info.bin`
   - `num_deletes`: count entries in `deleted.bin` (file size / 8)
7. Write `manifest.json` to `versions/{version_number}/` with the single segment entry
8. `deleted.bin` and `global_info.bin` stay in the version dir (they're already in the right place for v2)
9. Update CURRENT atomically: write `format_version=2` + same `version_number`
10. Sync directories

**Key properties:**
- Uses `fs::rename` (atomic on same filesystem) for the data files — no data copying needed
- If migration fails midway, the original v1 files may be partially moved. On retry, the migration function should detect partial state (e.g., files already in `segments/seg_0/` but no manifest) and resume
- After migration, the index opens normally as v2 with a single segment
- Future compactions will create additional segments as data grows

**Fallback:** If `fs::rename` fails (cross-filesystem), fall back to copy + delete.

## Verification

### Existing test suites (must remain green)

All 70 existing tests across three files need constructor signature updates (`Threshold` → `SegmentConfig`) but must otherwise pass unchanged:

1. `cargo test --test string_integration` — 47 tests covering basic CRUD, compaction, persistence, search modes (exact/prefix/fuzzy), phrase boost, scorer thresholds, document filters, edge cases (unicode, large doc_ids, doc_id 0), info, integrity check, cleanup, delete-then-reinsert
2. `cargo test --test string_concurrency` — 12 tests covering concurrent reads, concurrent writes, reads during compaction, writes during compaction, compaction serialization, mixed stress, snapshot isolation
3. `cargo test --test string_score_consistency` — 4 tests covering total_documents correctness after carry-forward, score equivalence across compaction strategies, score stability across insert/delete/compact cycles

### New tests to add

All new tests go in `tests/string_integration_tests.rs` unless noted otherwise.

**A. Multi-segment creation and layout**

1. `test_single_compaction_creates_one_segment` — Insert docs, compact once. Verify manifest has exactly one entry. Verify `segments/seg_0/` contains keys.fst, postings.dat, doc_lengths.dat.
2. `test_multiple_compactions_create_multiple_segments` — Use a small `max_postings_per_segment` (e.g., 50). Insert enough data across multiple compact cycles to force segment creation. Verify manifest has >1 entries. Verify each segment dir exists on disk.
3. `test_manifest_ordered_by_min_doc_id` — After multiple compactions, read manifest and assert entries are strictly ordered by `min_doc_id`.

**B. Cross-segment search correctness**

4. `test_same_term_across_segments_returns_all_docs` — Insert docs containing the same term across two compaction cycles (forcing two segments). Search for that term. Verify all docs from both segments appear in results.
5. `test_search_equivalence_single_vs_multi_segment` — Insert the same dataset twice: once with `max_postings = u64::MAX` (single segment) and once with a small limit (multi-segment). Verify search results (doc_ids and scores) are identical for exact, prefix, and fuzzy queries.
6. `test_prefix_search_across_segments` — Terms with shared prefix split across segments. Prefix search returns the union.
7. `test_fuzzy_search_across_segments` — Fuzzy matches distributed across segments all appear in results.
8. `test_phrase_boost_across_segment_boundary` — Two adjacent tokens where one doc's postings are in segment 0 and consecutive positions span the query. Verify phrase boost is applied correctly. (Note: a single doc's postings live in exactly one segment, so phrase boost within a doc is always intra-segment. This test confirms no regression.)
9. `test_exact_match_flag_multi_segment` — Verify `exact_match: true` works correctly when segments contain both exact and stemmed positions.

**C. Deletion and compaction strategies**

10. `test_deletion_threshold_triggers_segment_rebuild` — Insert docs, compact (segment 0). Delete enough docs to exceed threshold. Compact again. Verify segment 0 is replaced by a new segment_id (old segment_id absent from manifest).
11. `test_carry_forward_preserves_segment_id` — Insert docs, compact (segment 0). Delete few docs (below threshold). Compact again. Verify segment 0's segment_id is unchanged in manifest, but `num_deletes` is updated.
12. `test_last_segment_absorbs_live_data` — Insert docs, compact (segment 0). Insert more docs (fitting within max_postings). Compact again. Verify manifest still has one segment (live data absorbed into last segment, new segment_id).
13. `test_live_data_creates_new_segment_when_last_full` — Insert docs, compact (segment 0 near max_postings). Insert more docs. Compact. Verify manifest has two segments.
14. `test_empty_compaction_skipped_multi_segment` — No live changes after multi-segment state. Verify compaction is a no-op (version number unchanged).

**D. Global stats correctness**

15. `test_bm25_scores_correct_multi_segment` — Same data, single segment vs. multi-segment. BM25 scores must be identical. (Extends existing `test_bm25_scores_identical_across_compaction_strategies` to multi-segment.)
16. `test_global_stats_exclude_deleted_docs_multi_segment` — Delete docs spanning multiple segments. Verify `global_info.bin` reflects only surviving documents. Search scores must use correct corpus stats.
17. `test_total_documents_stable_through_segment_splits` — Insert/delete/compact through multiple segment splits. Verify `total_documents` in global_info always equals (inserted - deleted).

**E. Cleanup**

18. `test_cleanup_removes_unreferenced_segments` — Compact multiple times (creating segments 0, 1, 2; then rebuilding 0→3). Cleanup. Verify segment 0's directory is removed. Verify segments referenced by current manifest survive.
19. `test_cleanup_removes_old_versions_multi_segment` — Multiple compaction versions exist. Cleanup removes all except current. Segment dirs shared across versions survive if referenced by current.
20. `test_cleanup_safe_during_concurrent_reads` — A reader holds an old `Arc<SegmentList>`. Cleanup runs. Reader can still access mmapped data (OS keeps file alive until mmap is dropped).

**F. Persistence**

21. `test_persistence_multi_segment_reopen` — Create multi-segment index, close, reopen. Verify all segments load correctly, search returns correct results, manifest is intact.
22. `test_persistence_reopen_then_compact` — Reopen multi-segment index, insert new data, compact. Verify new segment is created correctly alongside persisted segments.

**G. V1 → V2 migration** (new test file: `tests/string_migration_tests.rs`)

23. `test_v1_to_v2_migration_basic` — Create a v1 index by writing v1-format files directly (keys.fst, postings.dat, doc_lengths.dat, deleted.bin, global_info.bin in version dir, CURRENT with format_version=1). Open with v2 code. Verify: CURRENT now has format_version=2, `segments/seg_0/` exists with the three data files, `manifest.json` exists in version dir, original v1 files are gone from version dir, search returns all expected results.
24. `test_v1_to_v2_migration_with_deletes` — Same as above but with non-empty `deleted.bin`. Verify deleted.bin remains in version dir, deleted docs excluded from search, manifest `num_deletes` matches.
25. `test_v1_to_v2_migration_then_compact` — Migrate v1 index, insert new data, compact. Verify second segment created, search returns both old and new data.
26. `test_v1_to_v2_migration_empty_index` — Migrate an empty v1 index (no data files). Verify v2 opens with empty state (no segments, no errors).
27. `test_v1_to_v2_migration_idempotent` — Simulate partial migration (files already in seg_0 but no manifest). Reopen. Migration resumes and completes correctly.

**H. Concurrency with multi-segment** (add to `tests/string_concurrency_tests.rs`)

28. `test_concurrent_reads_multi_segment` — Build multi-segment index. Spawn 8 reader threads. Verify all return consistent, correct results.
29. `test_search_during_compaction_multi_segment` — Continuous reads while compaction creates/rebuilds segments. Readers never see partial state or crash.
30. `test_writes_preserved_during_compaction_multi_segment` — Inserts during multi-segment compaction are preserved in next compaction.
31. `test_concurrent_compaction_serialization_multi_segment` — Multiple threads attempt compaction on multi-segment index. Exactly one proceeds at a time, no corruption.

**I. Edge cases**

32. `test_segment_with_all_docs_deleted` — All docs in a segment are deleted. Segment is rebuilt as empty (or removed from manifest). Search still works.
33. `test_many_small_segments` — Force 20+ segments with small `max_postings_per_segment`. Search still returns correct results. (Stress test for cursor management across many segments.)
34. `test_info_multi_segment` — Verify `IndexInfo` fields: `num_segments`, `total_segments_size_bytes`, `total_postings_count` (sum across segments).
35. `test_integrity_check_multi_segment` — Verify integrity check validates manifest, all segment dirs exist, all segment files present.

### Verification commands

```bash
cargo build                            # Compilation
cargo test --test string_integration   # Integration + multi-segment tests
cargo test --test string_concurrency   # Concurrency tests
cargo test --test string_score_consistency  # Scoring regression
cargo test --test string_migration     # V1→V2 migration tests
cargo clippy                           # Lint clean
```
