# Compaction Investigation: oramacore_geopoint_index

## 1. What This Project Does

`oramacore_geopoint_index` is a high-performance, thread-safe geographic point indexing library in Rust. It supports bounding-box and radius queries on geo-coordinates using a **BKD tree** (Block K-D tree) spatial data structure with a two-layer architecture:

- **Live layer**: In-memory append-only operation log (`Vec<LiveOp>`) for fast O(1) inserts/deletes
- **Compacted layer**: Memory-mapped BKD tree segments on disk for efficient O(log N + k) spatial queries

The `compact()` operation is the bridge: it merges the live layer into the compacted layer, rebuilding the BKD tree.

---

## 2. How the Current Compact Process Works

### 2.1 Two Compaction Paths

The system chooses between **hot compact** (incremental) and **full compact** (complete rebuild):

```
if delete_ratio > threshold OR segment_count >= max_segments OR segment_count == 0:
    Full Compact
else:
    Hot Compact
```

### 2.2 Full Compact

1. Refreshes the live snapshot under write lock
2. Streams ALL existing points from ALL segments + live inserts into a mmap'd temp file
3. Sorts all points by (lat, lon, doc_id)
4. Deduplicates exact (point, doc_id) pairs
5. Prunes stale deletes (only keeps deletes for doc_ids that exist)
6. If delete_ratio > threshold: physically removes deleted points, writes empty deleted.bin
7. If delete_ratio <= threshold: keeps all points, writes merged deleted.bin
8. Builds a single new BKD tree from the final point set
9. Atomically updates the CURRENT file (via temp + rename)
10. Swaps the version pointer (lock-free via ArcSwap)
11. Drains compacted ops from the live layer by position

**Cost: O(N log N)** where N = total points across all segments + live inserts.

### 2.3 Hot Compact

1. Refreshes the live snapshot under write lock
2. Hardlinks all existing segments into a new version directory (cheap, no data copy)
3. Writes only live inserts to a new mmap'd temp file
4. Sorts and deduplicates live inserts
5. Builds a BKD tree for the new segment only
6. Writes updated manifest and merged deleted.bin
7. Atomically updates CURRENT and swaps version

**Cost: O(n log n)** where n = live inserts since last compaction.

### 2.4 Concurrency Model

- I/O happens without holding any locks (reads and writes can continue)
- Version swap uses lock-free ArcSwap (readers are never blocked)
- Position-based drain (`ops.drain(..ops_len)`) preserves concurrent writes that arrived during compaction
- A single `compaction_lock` mutex serializes compactions

---

## 3. How Other Databases Handle Compaction

### 3.1 LSM-Tree Strategies (RocksDB, LevelDB, Cassandra)

#### Level-Based Compaction (RocksDB default)

Data organized into levels L0..Ln, each level T times larger than the previous. When a level exceeds its target size, one SST file is merged with overlapping files in the next level.

| Metric | Value |
|--------|-------|
| Write amplification | High: ~10-50x (T * L where T=fan-out, L=levels) |
| Read amplification | Low: 1 SST per level with Bloom filters |
| Space amplification | Low: ~10% overhead |
| Best for | Read-heavy, space-constrained workloads |

#### Size-Tiered Compaction (Cassandra default)

SSTables of similar size are grouped; when enough accumulate (typically 4), they merge into one larger SSTable.

| Metric | Value |
|--------|-------|
| Write amplification | Low: ~5x |
| Read amplification | High: up to T-1 runs per level |
| Space amplification | High: up to 4x, needs 50% free disk |
| Best for | Write-heavy workloads |

#### Universal Compaction (RocksDB variant)

A more controlled size-tiered approach with four evaluation steps: age-based, space amplification check, size ratio check, and fallback.

| Metric | Value |
|--------|-------|
| Write amplification | Low-medium |
| Read amplification | Medium |
| Space amplification | Medium-high (temporary 2x during full compaction) |
| Best for | Write-heavy with configurable trade-offs |

### 3.2 Spatial Index Approaches

#### PostGIS (GiST/R-Tree)

- No background compaction; relies on PostgreSQL's VACUUM and REINDEX
- R-tree bounding boxes degrade over time with updates/deletes (overly large bounding boxes cause false positives)
- REINDEX drops and rebuilds the entire index from scratch
- VACUUM FULL rewrites entire table + indexes with an exclusive lock
- Recommended: periodic REINDEX for heavily updated spatial tables

#### SQLite R*-Tree

- No separate compaction process at all
- Self-maintaining via standard R*-tree algorithms (insert, split, condense)
- Updates = delete + re-insert, which triggers local tree reorganization
- Node splits use quadratic algorithm for spatial quality
- Trade-off: simpler but page utilization degrades over time (~50-70%)

#### Lucene/Elasticsearch (BKD Trees)

**This is the closest comparison to oramacore_geopoint_index.**

- Uses BKD trees (same data structure) for spatial data since Lucene 6.0
- Segments are immutable; new data goes to new segments
- **TieredMergePolicy** (default): merges segments of approximately equal size
- During merge, BKD trees are **fully rebuilt** (no incremental merge) -- same approach as this project
- Maximum segment size defaults to 5GB
- Deleted documents consume space until their segment is merged

| Metric | Value |
|--------|-------|
| Write amplification | Similar to size-tiered LSM |
| Read amplification | Proportional to segment count |
| Space amplification | Temporary 2x during merges |

### 3.3 Copy-on-Write B+ Tree Approaches

#### LMDB

- Copy-on-write: modified pages written to new locations, root pointer atomically updated
- **No compaction, no WAL, no garbage collection needed**
- Free page tracking: freed pages reused by new transactions
- Write amplification: moderate (path-copy from leaf to root, ~3-4 pages), but zero background amplification
- Space amplification: database file never shrinks automatically (high-water mark)
- Single-writer limitation

#### WiredTiger (MongoDB)

- B-tree with copy-on-write checkpoints
- Compaction is explicit and on-demand, runs in 10% increments with three checkpoints each
- Relocates leaf pages from file end to free blocks near the beginning
- "Best-effort" operation that may not reduce file size

### 3.4 Modern/Novel Approaches

#### Dostoevsky (Lazy Leveling)

Apply tiering at ALL levels except the largest (which uses leveling). Achieves:
- Same point lookup & space amplification as leveling
- Same update cost as tiering
- Optimal for mixed workloads via closed-form performance model

#### PebblesDB (Fragmented LSM)

Data from level i is NOT merged with existing data at level i+1; instead added to the correct "guard" range. Write amplification 2.4-3x lower than RocksDB, write throughput 6.7x higher. Trade-off: range queries more expensive.

#### ScyllaDB Incremental Compaction (ICS)

Replaces large SSTables with fixed-size fragments (~1GB). During compaction, space from input fragments is immediately released. Worst-case temporary space drops from full SSTable size to just 2x fragment size (~5% space overhead vs STCS's 50%).

#### LSM RUM-Tree (Spatial-Specific)

Augments R-tree with an in-memory Update Memo hash map. Achieves 6.6x speedup on updates and up to 249,292x speedup on queries over naive LSM R-tree implementations.

---

## 4. Analysis of the Current Approach

### 4.1 What the Current Approach Is

The current design is essentially a **hybrid of Lucene-style segment merging and size-tiered compaction**, specialized for BKD trees:

- Hot compact = add a new segment (like Lucene adding a new segment)
- Full compact = merge all segments into one (like Lucene's force-merge or size-tiered compaction's full merge)
- Delete handling via a separate deleted.bin with threshold-based physical removal

### 4.2 Pros

| Pro | Detail |
|-----|--------|
| **Simple mental model** | Two clear paths: hot (incremental) or full (rebuild). Easy to reason about correctness. |
| **Lock-free reads** | ArcSwap pointer swap means readers are never blocked during compaction. Active iterators hold Arc to old version. |
| **Crash safety** | Atomic CURRENT file update via temp+rename ensures always-valid state on disk. |
| **Fast hot path** | Hardlinking old segments is O(1) per segment. Only live inserts are sorted/indexed. |
| **Efficient BKD rebuild** | Using select_nth_unstable (quickselect) for median partitioning is O(N) per level, O(N log N) total. |
| **Good delete semantics** | Threshold-based decision avoids unnecessary full rebuilds. Stale delete pruning prevents permanent shadowing of re-inserted docs. |
| **Low read amplification (single segment)** | After full compact, queries traverse exactly one BKD tree. O(log N + k) is optimal. |
| **Memory-mapped I/O** | Zero-copy reads with platform-specific madvise hints (WILLNEED, RANDOM, SEQUENTIAL). |
| **Concurrent writes during compaction** | Position-based ops drain preserves writes that arrive during the I/O phase. |

### 4.3 Cons

| Con | Detail | Severity |
|-----|--------|----------|
| **Full compact is O(N log N) and reads ALL data** | Every full compact re-reads, re-sorts, and re-writes the entire dataset. For 100M points (1.6 GB), this means reading 1.6 GB + sorting + writing 1.6 GB on every full compaction. This is the #1 scaling bottleneck. | **High** |
| **No incremental/partial full compact** | Unlike Lucene's TieredMergePolicy which selects a subset of similarly-sized segments to merge, or ScyllaDB's ICS which processes in fixed-size fragments, the current full compact is all-or-nothing. | **High** |
| **Temporary space doubling during full compact** | Writing all points to a temp mmap file while old version still exists means ~2x disk usage during compaction. Same issue as STCS. | **Medium** |
| **Write amplification grows linearly with data size** | Each full compaction rewrites every point. If compaction frequency is proportional to insert rate, write amplification = O(N/batch_size). With 100M points and 10K inserts per batch, that's 10,000x write amplification per point over its lifetime. | **High** |
| **Read amplification degrades with segment count** | Hot compacts add segments. Queries must scan all segments sequentially (MultiSegmentQueryIterator). With 10 segments, read amplification is 10x worse than single-segment. | **Medium** |
| **No partial tree merge for BKD** | Unlike B-tree variants (LMDB, WiredTiger) where updates modify only affected pages, BKD trees must be fully rebuilt. This is a fundamental limitation of the BKD tree design (also shared by Lucene). | **Medium** |
| **deleted.bin loaded entirely into HashSet on startup** | For millions of deletes, this is a significant memory overhead and startup cost. A Bloom filter or sorted array with binary search could be more memory-efficient. | **Low-Medium** |
| **No merge policy / segment selection** | When full compact triggers, ALL segments are merged. A smarter policy (like Lucene's TieredMergePolicy) could merge only a subset of similarly-sized segments, reducing work per compaction. | **Medium** |
| **Compaction is synchronous and caller-driven** | The caller must call `compact(version_id)` explicitly. There's no automatic background compaction thread, no scheduling, no priority management. The caller also provides the version_id externally, which is fragile. | **Low** |
| **No write throttling or backpressure** | If compaction falls behind the write rate, the live layer grows unboundedly. No mechanism to slow writes when compaction is needed. | **Low-Medium** |
| **Hardlinks for hot compact may not work across filesystems** | Minor portability concern. Also, hardlinks share the underlying inode, so disk space for old segments is not freed until the last hardlink is removed. | **Low** |

### 4.4 Comparison with Closest Analogues

#### vs. Lucene/Elasticsearch BKD

| Aspect | This Project | Lucene |
|--------|-------------|--------|
| BKD rebuild on merge | Full rebuild | Full rebuild (same) |
| Segment selection | All-or-nothing | TieredMergePolicy selects subset |
| Max segment size | No limit | 5 GB default |
| Merge concurrency | Single compaction thread | Multiple concurrent merges |
| Delete handling | Separate deleted.bin | Per-segment delete bitset |
| Segment count control | max_segments parameter | segmentsPerTier + maxMergeAtOnce |

The key difference: **Lucene's TieredMergePolicy is more granular**. It merges a selected subset of segments at a time, not all of them. This keeps compaction cost proportional to the segments being merged, not the entire dataset.

#### vs. RocksDB Level Compaction

| Aspect | This Project | RocksDB |
|--------|-------------|---------|
| Merge unit | Entire BKD tree | Individual SST files within a level |
| Write amplification | O(N) per full compact | O(T) per level transition |
| Read amplification | O(segments) | O(levels) with Bloom filters |
| Incremental merge | No (full rebuild) | Yes (one SST at a time) |
| Space amplification | 2x during full compact | ~10% steady-state |

The fundamental difference: RocksDB can merge individual SST files, keeping per-compaction cost bounded. BKD trees cannot be partially merged.

---

## 5. Potential Improvements

### 5.1 Tiered Segment Merge Policy (Recommended, High Impact)

**Instead of all-or-nothing full compact, merge subsets of similarly-sized segments.**

How it would work:
- Group segments into tiers by size (e.g., 1x, 10x, 100x)
- When a tier accumulates enough segments, merge only that tier's segments
- This bounds per-compaction I/O to the tier size, not the total dataset size

Example: With 1M points in segment A, 1M in B, 1K in C, and 1K in D:
- Current: full compact reads/writes all 2.002M points
- Tiered: merge C+D into a 2K segment; A and B are untouched

**Expected improvement:** Write amplification drops from O(N) to O(tier_size). This is what Lucene does with TieredMergePolicy.

### 5.2 Per-Segment Delete Bitsets (Medium Impact)

**Replace the global deleted.bin with per-segment delete bitmaps.**

Why:
- Currently, deleted.bin is global and loaded into a HashSet (O(D) memory, O(1) lookup)
- With per-segment bitmaps, each segment carries its own delete tracking
- When a segment is fully merged/rebuilt, its deletes are physically applied
- Reduces memory: only need bitmaps for segments that have deletions

Additional benefit: enables deleting only from specific segments during partial merges.

### 5.3 Background Compaction with Scheduling (Medium Impact)

**Add automatic compaction scheduling instead of caller-driven `compact()`.**

Options:
- Trigger compaction when live layer exceeds a size threshold
- Use a dedicated background thread with configurable concurrency
- Add write backpressure when live layer grows too large
- Auto-generate version_ids internally

### 5.4 Bloom Filters for Delete Set (Low-Medium Impact)

**Replace HashSet<u64> for deleted_set with a Bloom filter for the common case.**

- Most lookups against the delete set are negative (doc_id is NOT deleted)
- A Bloom filter would reduce memory from ~50 bytes/entry (HashSet) to ~10 bits/entry
- False positives cause an unnecessary skip; verify with a sorted array on disk
- This is what the Monkey paper recommends for LSM-tree Bloom filter optimization

### 5.5 Incremental BKD Merge (High Impact, High Complexity)

**Support partial BKD tree updates without full rebuild.**

This is a research-level change. Current BKD trees are static and must be rebuilt entirely. Options:
- **Buffer tree approach**: Attach insert/delete buffers to inner nodes; flush lazily
- **LSM-BKD hybrid**: Maintain multiple BKD trees at different levels (similar to what we already do with segments, but more structured)
- **Dynamic KD-tree**: Use a self-balancing variant instead of a static BKD tree

This is the most impactful but also the most complex change. Lucene also does not support incremental BKD updates.

### 5.6 Fragment-Based Compaction (Inspired by ScyllaDB ICS)

**Process compaction in fixed-size chunks to reduce peak disk usage.**

Instead of writing all N points to a temp file before building the BKD:
- Process in 100MB fragments
- Release input fragment space as each output fragment is written
- Reduces peak temporary disk usage from 2x to fragment_size * 2

---

## 6. Summary Table

| Approach | Write Amp | Read Amp | Space Amp | Complexity | Recommended |
|----------|-----------|----------|-----------|------------|-------------|
| Current (all-or-nothing) | High (O(N)) | Low-Medium (O(segments)) | 2x during full | Low | Current |
| Tiered Merge Policy | Medium (O(tier)) | Low-Medium | 2x per tier | Medium | **Yes** |
| Per-Segment Deletes | Same | Slightly better | Slightly better | Medium | Yes |
| Background Compaction | Same | Same | Same | Medium | Yes |
| Bloom Filter Deletes | Same | Same | Better memory | Low | Optional |
| Incremental BKD | Very Low | Low | Low | Very High | Future research |
| Fragment-Based | Same | Same | Much lower peak | Medium | Optional |

---

## 7. Conclusion

The current compaction design is **correct, crash-safe, and well-engineered for concurrency**. The hot compact path with hardlinks is an efficient optimization for the common case of small batches of inserts.

However, the **full compact path is the critical weakness**: it scales linearly with total data size (O(N log N)), reads and rewrites every point, and temporarily doubles disk usage. This is acceptable for small-to-medium datasets (under ~10M points) but becomes a significant bottleneck at scale.

The single highest-impact improvement would be adopting a **tiered segment merge policy** (similar to Lucene's TieredMergePolicy), which bounds per-compaction work to a subset of segments rather than the entire dataset. This change is architecturally compatible with the existing multi-segment design and would dramatically reduce write amplification at scale.

---

## Sources

- [RocksDB Compaction Wiki](https://github.com/facebook/rocksdb/wiki/Compaction)
- [RocksDB Universal Compaction](https://github.com/facebook/rocksdb/wiki/universal-compaction)
- [RocksDB Leveled Compaction](https://github.com/facebook/rocksdb/wiki/Leveled-Compaction)
- [Lucene Points/BKD in 6.0](https://www.elastic.co/blog/lucene-points-6-0)
- [BKD-backed geo_shapes in Elasticsearch](https://www.elastic.co/blog/bkd-backed-geo-shapes-in-elasticsearch-precision-efficiency-speed)
- [Elasticsearch Segment Merges Explained](https://medium.com/@shivam.agarwal.in/elasticsearch-navigating-lucene-segment-merges-9ed775bd45cb)
- [PostGIS Spatial Indexing](https://postgis.net/workshops/postgis-intro/indexing.html)
- [ScyllaDB Incremental Compaction](https://www.scylladb.com/2021/04/28/incremental-compaction-2-0-a-revolutionary-space-and-write-optimized-compaction-strategy/)
- [Cassandra Compaction Strategies](https://www.scylladb.com/glossary/cassandra-compaction-strategy/)
- [TiKV SST Compaction Guard](https://medium.com/@siddontang/how-we-optimize-rocksdb-in-tikv-sst-compaction-guard-6c2d2431a7c5)
- [CockroachDB Pebble Storage](https://www.cockroachlabs.com/blog/pebble-rocksdb-kv-store/)
- [CockroachDB Value Separation](https://www.cockroachlabs.com/blog/value-separation-pebble-optimization/)
- [LMDB Architecture](http://www.lmdb.tech/doc/)
- [WiredTiger Compaction](https://source.wiredtiger.com/11.0.0/arch-compact.html)
- [Dostoevsky: Lazy Leveling (SIGMOD 2018)](https://nivdayan.github.io/dostoevsky.pdf)
- [Monkey: Optimal Bloom Filters (ACM TODS 2018)](https://nivdayan.github.io/monkey-journal.pdf)
- [PebblesDB (SOSP 2017)](https://www.cs.utexas.edu/~vijay/papers/sosp17-pebblesdb.pdf)
- [LSM RUM-Tree (ICDE 2021)](https://www.cs.purdue.edu/homes/csjgwang/pubs/ICDE21_LSM.pdf)
- [An In-depth Discussion on LSM Compaction (Alibaba Cloud)](https://www.alibabacloud.com/blog/an-in-depth-discussion-on-the-lsm-compaction-mechanism_596780)
- [SQLite R*-Tree Module](https://www.sqlite.org/rtree.html)
