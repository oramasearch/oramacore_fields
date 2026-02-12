# String Module (BM25 Scoring Search) - Investigation & Design Notes

## Goal

Create a new `string` module in `oramacore_fields` that supports full-text search with BM25 scoring. This differs from `string_filter` (exact-match filtering returning doc_id sets) by assigning relevance scores to matching documents.

## Data Model

Each insert provides:
- `doc_id: u64`
- `field_length: u16` (number of tokens in the field)
- `terms: HashMap<String, TermData>` where `TermData = { exact_positions: Vec<u32>, stemmed_positions: Vec<u32> }`

This mirrors `UncommittedStringField::insert` in oramacore (`src/collection_manager/sides/read/index/uncommitted_field/string.rs`), which takes `(document_id, field_length, terms: InsertStringTerms)`.

## BM25 Algorithm (from oramacore/src/collection_manager/bm25.rs)

Three core functions:

```
normalized_tf(tf, field_len, avg_field_len, b) = tf / (1 - b + b * (field_len / avg_field_len))
idf(total_docs, corpus_df) = ln(1 + (total_docs - df + 0.5) / (df + 0.5))   [Lucene-style, always positive]
bm25f_score(S_t, k, idf) = idf * (k + 1) * S_t / (k + S_t)
```

Default parameters: k = 1.2, b = 0.75

Search flow (from oramacore committed_field/string.rs):
1. For each query token → search FST (exact or prefix match)
2. For each matching term → iterate posting list (doc_id, positions)
3. For each doc → compute `term_occurrence = len(exact_positions) + len(stemmed_positions)` (or just exact if `exact_match`)
4. Get `field_length` from doc_lengths storage
5. Compute `normalized_tf` → accumulate into scorer
6. After all terms processed → compute final score with IDF

For phrase matching (multi-token): track positions per document, boost score for sequential position occurrences.

## Concurrency Model (shared across all modules)

```rust
struct StringStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,      // Lock-free reads via load()
    live: RwLock<LiveLayer>,                  // Protects in-memory mutations
    compaction_lock: Mutex<()>,               // Serializes compaction operations
    threshold: Threshold,
}
```

- **Read path**: Double-check locking on `live` (read lock first, upgrade to write if snapshot dirty), then `ArcSwap::load()` for compacted version. Returns handle holding `Arc` refs.
- **Write path**: Acquire write lock on `live`, append op, mark dirty. O(1).
- **Compaction**: Lock `compaction_lock`, refresh snapshot, perform I/O (no locks), then write lock `live` to swap version + drain ops.

Key insight: `live.ops.drain(..snapshot.ops_len)` preserves any ops appended during compaction I/O.

## On-Disk Format

Version directory (`versions/{N}/`):

### keys.fst
FST (via `fst` crate) mapping term string → byte offset in `postings.dat`.
Supports O(key_length) exact lookup and prefix/automaton search.

### postings.dat
Variable-length posting lists with positions. Per term at byte offset from FST:

```
[doc_count: u32][_pad: u32]                     // 8 bytes header
For each doc (sorted by doc_id ascending):
  [doc_id: u64]                                  // 8 bytes
  [exact_pos_count: u32][stemmed_pos_count: u32] // 8 bytes
  [exact_positions: u32 * exact_pos_count]       // variable
  [stemmed_positions: u32 * stemmed_pos_count]   // variable
```

Unlike `string_filter/postings.dat` (fixed 8-byte entries), this is variable-length because each document can have different numbers of positions.

### doc_lengths.dat
Sorted array of 12-byte entries: `(doc_id: u64, field_length: u32)`.
Binary search for per-doc field length during BM25 normalization.

### deleted.bin
Sorted u64 doc_ids in little-endian (same as all other modules).

### global_info.bin
16 bytes: `(total_document_length: u64, total_documents: u64)`.
Needed for `avg_field_length = total_document_length / total_documents`.

### CURRENT
Two-line text file: format version (line 1), version number (line 2).
Written atomically via temp+rename (same as all modules).

## Compaction Strategies (same as string_filter)

**Strategy A (apply deletes)**: When `merged_deletes / total_postings > threshold`:
- Build HashSet of all deletes
- Merge compacted + live entries, filtering out deleted doc_ids
- Write empty `deleted.bin`

**Strategy B (carry forward)**: When below threshold:
- Merge compacted + live entries without filtering
- Write merged deletes to `deleted.bin`

During compaction merge:
- Terms from compacted and live are merged in lexicographic order (two-pointer merge)
- For same term: posting lists merged by doc_id; live version wins on conflict
- Doc lengths merged similarly

## LiveLayer Design

**LiveOp**:
```rust
enum LiveOp {
    Insert { doc_id: u64, field_length: u16, terms: HashMap<String, TermData> },
    Delete(u64),
}
```

**LiveSnapshot** (materialized from ops replay):
- `term_postings: HashMap<String, Vec<(u64, Vec<u32>, Vec<u32>)>>` - per-term sorted posting lists
- `doc_lengths: HashMap<u64, u16>` - per-doc field length
- `deletes: Arc<HashSet<u64>>` - for O(1) membership test
- `deletes_sorted: Vec<u64>` - for merge operations
- `total_field_length: u64` - sum of all field lengths (for avg computation)
- `total_documents: u64` - count of unique docs
- `ops_len: usize` - for drain after compaction

Snapshot refresh replays all ops: Insert adds to maps (removes from deletes), Delete removes from maps (adds to deletes).

## Search API

```rust
pub fn search(&self, params: &SearchParams) -> SearchResult
```

Where:
```rust
struct SearchParams {
    tokens: Vec<String>,     // Pre-tokenized query terms
    exact_match: bool,       // Only count exact positions
    boost: f32,              // Field weight for BM25F
    bm25_params: Bm25Params, // k, b parameters
}

struct SearchResult {
    docs: Vec<ScoredDoc>,    // Sorted by score descending
}

struct ScoredDoc {
    doc_id: u64,
    score: f32,
}
```

Search returns eagerly-computed scored results (unlike string_filter's lazy iterator), because BM25 requires global statistics (IDF, avg_field_length) before scores can be finalized.

## Key Differences from string_filter

| Aspect | string_filter | string (new) |
|--------|--------------|-------------|
| Purpose | Exact-match filtering | BM25 scored search |
| Returns | Lazy doc_id iterator | Eager scored results |
| Postings format | Fixed: `[count][doc_ids...]` | Variable: positions per doc |
| Extra files | — | `doc_lengths.dat`, `global_info.bin` |
| LiveOp::Insert | `(String, u64)` | `{ doc_id, field_length, terms }` |
| LiveSnapshot | Columnar (keys+ranges+doc_ids) | Per-term HashMap |
| Query | Exact string match | Token prefix/exact + BM25 scoring |

## Existing Dependencies (no additions needed)

- `fst 0.4` - FST for term lookup
- `memmap2 0.9` - mmap for zero-copy file access
- `arc-swap 1.8` - lock-free version swapping
- `anyhow 1.0` - error handling
- `serde_json 1.0` - JSON value extraction
- `tracing 0.1` - logging
- `libc 0.2` (unix) - madvise

## Module Structure

```
src/string/
  mod.rs        - module exports
  storage.rs    - main API (StringStorage): new, insert, delete, search, compact, cleanup
  indexer.rs    - IndexedValue, TermData types
  live.rs       - LiveOp, LiveLayer, LiveSnapshot
  compacted.rs  - CompactedVersion with FST + mmap
  iterator.rs   - SearchParams, SearchResult, ScoredDoc, SearchHandle
  merge.rs      - merge primitives for postings and doc_lengths
  scoring.rs    - BM25 scoring functions (copied from oramacore bm25.rs)
  io.rs         - file I/O utilities
  config.rs     - Threshold + Bm25Params
  platform.rs   - madvise wrappers (copy from string_filter)
  error.rs      - error types
  info.rs       - diagnostics
```

## Implementation Order

1. `config.rs`, `error.rs`, `platform.rs` - no dependencies
2. `scoring.rs` - pure math, no dependencies
3. `indexer.rs` - type definitions only
4. `io.rs` - file I/O utilities
5. `merge.rs` - sorted merge/subtract primitives
6. `info.rs` - diagnostic types
7. `live.rs` - depends on indexer, merge
8. `compacted.rs` - depends on io, platform, merge
9. `iterator.rs` - depends on compacted, live, scoring, config (search logic)
10. `storage.rs` - depends on all above
11. `mod.rs` - module declarations
12. Update `src/lib.rs` - add `pub mod string;`
