# oramacore_geopoint_index Crate Analysis

A comprehensive analysis of the `oramacore_geopoint_index` crate, documenting its structure, algorithms, and edge cases.

## 1. File Structure and Organization

```
oramacore_geopoint_index/
├── Cargo.toml           # Package manifest
├── src/
│   ├── lib.rs           # Public API exports (~19 lines)
│   ├── storage.rs       # Main GeoPointStorage implementation (~866 lines)
│   ├── live.rs          # In-memory ops-log data layer (~346 lines)
│   ├── iterator.rs      # Filter iteration logic with GeoFilterOp enum (~258 lines)
│   ├── point.rs         # GeoPoint, EncodedPoint, f64↔i32 encoding (~161 lines)
│   ├── compacted/
│   │   ├── mod.rs       # CompactedVersion, mmap loading, global bounds (~531 lines)
│   │   ├── build.rs     # BKD tree construction (~300 lines)
│   │   └── query.rs     # BKD tree traversal (bbox, radius, haversine) (~640 lines)
│   ├── indexer.rs       # GeoPointIndexer and IndexedValue for JSON extraction (~235 lines)
│   ├── config.rs        # Threshold configuration (~72 lines)
│   ├── error.rs         # Error types (~37 lines)
│   ├── io.rs            # Atomic disk I/O operations (~180 lines)
│   ├── info.rs          # IndexInfo and IntegrityCheck types (~51 lines)
│   ├── platform.rs      # Platform-specific madvise optimizations (~35 lines)
│   └── bin/
│       └── cli.rs       # CLI binary (requires "cli" feature) (~243 lines)
└── tests/
    ├── integration_tests.rs   # Integration tests (~674 lines)
    └── concurrency_tests.rs   # Concurrency tests (~1627 lines)
```

## 2. Main Types and Their Purposes

### Core Types

| Type | Module | Purpose |
|------|--------|---------|
| `GeoPointStorage` | storage.rs | Thread-safe geopoint index with concurrent read/write support |
| `GeoPoint` | point.rs | Validated (lat, lon) pair; lat in [-90, 90], lon in [-180, 180] |
| `EncodedPoint` | point.rs | Integer-encoded (i32, i32) representation for disk storage and comparison |
| `GeoPointIndexer` | indexer.rs | Extracts geopoints from JSON for indexing (plain or array mode) |
| `IndexedValue` | indexer.rs | Enum: `Plain(GeoPoint)` or `Array(Vec<GeoPoint>)` — value extracted by GeoPointIndexer |
| `Threshold` | config.rs | Configurable delete ratio threshold (0.0-1.0, default 0.1) |
| `LiveLayer` | live.rs | In-memory chronological ops log for new inserts/deletes |
| `LiveOp` | live.rs | Enum: `Insert(GeoPoint, u64)` or `Delete(u64)` — single mutation op |
| `LiveSnapshot` | live.rs | Sorted, deduplicated snapshot of live data for iteration |
| `CompactedVersion` | compacted/mod.rs | Memory-mapped BKD tree version with leaf data and deleted set |
| `FilterData` | iterator.rs | Query result wrapper that owns references to version/snapshot |
| `FilterIterator` | iterator.rs | Iterator that yields matching doc_ids from both live and compacted layers |
| `GeoFilterOp` | iterator.rs | Enum for spatial query operations (BoundingBox, Radius) |

### Algorithm Types

| Type | Module | Purpose |
|------|--------|---------|
| `CompactedQueryIterator` | compacted/query.rs | Stack-based BKD tree traversal iterator with leaf scanning |
| `InnerNodesView` | compacted/mod.rs | View into mmap'd inner nodes (split_value + split_dim) |
| `LeafOffsetsView` | compacted/mod.rs | View into mmap'd leaf offset table |
| `InnerNode` | compacted/build.rs | On-disk inner node structure (split_value: i32, split_dim: u8, padding: [u8; 3]) |
| `StackFrame` | compacted/query.rs | Enum: `Traverse { node_id, cell_bounds }` or `CollectAll { node_id }` |
| `LeafScan` | compacted/query.rs | Active leaf scan state (entries_start, count, current, check_bounds) |
| `QueryKind` | compacted/query.rs | Enum: `BBox { encoded bounds }` or `Radius { center, radius, bbox bounds }` |
| `Relation` | compacted/query.rs | Enum: `Outside`, `Inside`, `Crosses` — bbox classification result |

### On-Disk Types

| Type | Module | Purpose |
|------|--------|---------|
| `InnerNode` | compacted/build.rs | 8-byte inner node: split_value (i32) + split_dim (u8) + padding (3 bytes) |
| Leaf entry | compacted/build.rs | 16-byte entry: lat (i32) + lon (i32) + doc_id (u64) |
| Leaf header | compacted/build.rs | 8-byte header: count (u32) + padding (u32) |

### Info Types

| Type | Module | Purpose |
|------|--------|---------|
| `IndexInfo` | info.rs | Metadata and statistics about an index |
| `IntegrityCheckResult` | info.rs | Result of an integrity check (passed flag + individual checks) |
| `IntegrityCheck` | info.rs | Single integrity check (name, status, message) |
| `CheckStatus` | info.rs | Enum: `Ok`, `Fail`, `Skip` |

### Config Types

| Type | Module | Purpose |
|------|--------|---------|
| `Threshold` | config.rs | Delete ratio threshold for compaction strategy (0.0-1.0) |
| `Error` | error.rs | Error enum (InvalidLatitude, InvalidLongitude, UnsupportedVersion, Io) |

## 3. Architecture Overview

```
GeoPointStorage
├── LiveLayer (RwLock) - Chronological ops log in memory
│   ├── ops: Vec<LiveOp> (append-only Insert/Delete operations)
│   ├── cached_snapshot: Arc<LiveSnapshot> - sorted, deduplicated
│   └── snapshot_dirty: bool - whether snapshot needs refresh
│
└── CompactedVersion (ArcSwap) - Lock-free atomic version swap
    ├── inner_mmap: Option<Mmap> - BKD inner nodes + leaf offsets
    ├── leaves_mmap: Option<Mmap> - leaf entries (lat, lon, doc_id)
    ├── deleted_mmap: Option<Mmap> - sorted deleted doc_ids
    ├── deleted_set: Arc<HashSet<u64>> - precomputed for O(1) lookup
    ├── num_inner / num_leaves - tree dimensions
    └── global_min/max_lat/lon - precomputed bounding box
```

### Key Design Characteristics

1. **BKD tree spatial index** — points stored in a balanced k-d tree for efficient spatial queries
2. **Two query types** — `GeoFilterOp` supports BoundingBox and Radius queries
3. **f64→i32 encoding** — coordinates encoded to i32 for compact storage and integer comparison
4. **Multi-point support** — a single doc_id can have multiple geopoints (array indexing)
5. **Memory-mapped I/O** — inner nodes, leaves, and deleted set are mmap'd for zero-copy access
6. **madvise hints** — inner.idx: WILLNEED, leaves.dat: RANDOM, deleted.bin: SEQUENTIAL
7. **JSON indexing** — `GeoPointIndexer` extracts `{lat, lon}` objects from JSON
8. **Global bounds caching** — precomputed bounding box avoids scanning all leaves on every query

## 4. Concurrency Model

The index uses three synchronization primitives:

| Primitive | Purpose |
|-----------|---------|
| `RwLock<LiveLayer>` | Multiple concurrent readers OR one writer for the live layer |
| `ArcSwap<CompactedVersion>` | Lock-free atomic pointer swap; readers never block |
| `Mutex<()>` | Ensures only one compaction runs at a time |

### Read Path (`filter`)

1. Acquire read lock on LiveLayer
2. If snapshot is clean, clone Arc (O(1))
3. If dirty: drop read lock, acquire write lock, double-check, refresh if needed
4. Load CompactedVersion from ArcSwap (lock-free)
5. Return FilterData holding both Arcs

The double-check pattern avoids unnecessary snapshot refreshes when multiple threads race to upgrade the lock.

### Write Path (`insert` / `delete`)

1. Acquire write lock on LiveLayer
2. For insert: push `LiveOp::Insert(point, doc_id)` to ops Vec (array inserts push multiple ops)
3. For delete: push `LiveOp::Delete(doc_id)` to ops Vec
4. Mark snapshot as dirty
5. Release lock

Writes are fast O(1) appends. Sorting is deferred to snapshot refresh.

### Compaction Path

1. Acquire compaction_lock (prevents concurrent compactions)
2. Refresh snapshot under write lock
3. Load current version (lock-free)
4. Collect all existing points from the BKD tree
5. Merge with live inserts, deduplicate exact (point, doc_id) pairs
6. Decide compaction strategy based on delete ratio vs threshold
7. Build new BKD tree and write to disk (no locks held during I/O)
8. Sync directory and atomically update CURRENT file
9. Swap version and drain compacted ops via `ops.drain(..snapshot.ops_len)`
10. Release compaction_lock

Key insight: I/O happens without holding locks, allowing reads and writes to continue during disk operations.

### Iterator Stability

FilterData holds `Arc<CompactedVersion>` and `Arc<LiveSnapshot>`. Even if compaction swaps the version or new writes arrive, active iterators continue using their captured references. The old version is deallocated only when all iterators complete.

### Concurrent Operation Matrix

| Operation | Concurrent with Read | Concurrent with Write | Concurrent with Compact |
|-----------|---------------------|----------------------|------------------------|
| **Read**  | Yes (multiple readers) | Yes (readers block writers) | Yes (uses Arc snapshots) |
| **Write** | Yes (waits for readers) | No (exclusive lock) | Yes (preserved after swap) |
| **Compact** | Yes (no lock during I/O) | Yes (no lock during I/O) | No (compaction_lock) |

### Handling Writes During Compaction

When compaction completes, it drains ops by position (`ops.drain(..snapshot.ops_len)`), not by value. Items at indices 0..ops_len were present when the snapshot was taken, while items at indices ops_len.. were inserted concurrently during the I/O phase. This ensures concurrent inserts of the same doc_id are preserved.

## 5. On-Disk Format

```
base_path/
├── CURRENT           # Text file: format version + version_id (atomic write via rename)
└── versions/
    └── {version_id}/
        ├── inner.idx     # BKD inner nodes + leaf offset table
        ├── leaves.dat    # Leaf entries: (lat, lon, doc_id) packed sequentially
        └── deleted.bin   # Sorted deleted doc_ids (8 bytes each, native-endian)
```

### CURRENT File Format

Two lines: format version (currently `1`) and version_id pointing to the active directory.

```
1
42
```

On load, the format version is validated. If unsupported, an error is returned.

### inner.idx Format

The file is divided into two contiguous sections:

```
┌──────────────────────────────────────────────┐
│  Inner Nodes:  num_inner × 8 bytes each      │
│  ┌──────────────────────────────────────────┐ │
│  │ split_value: i32 (4 bytes, native-endian)│ │
│  │ split_dim:   u8  (1 byte: 0=lat, 1=lon) │ │
│  │ _padding:    [u8; 3]                     │ │
│  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│  Leaf Offsets: num_leaves × 8 bytes each     │
│  ┌──────────────────────────────────────────┐ │
│  │ offset: u64 (byte offset into leaves.dat)│ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

Total entries = `2 × num_inner + 1` (always odd). `num_leaves = num_inner + 1` (complete binary tree).

Inner nodes use 1-indexed BKD tree convention: node_id 1 is the root, children of node_id N are 2N (left) and 2N+1 (right). A node_id is a leaf when `node_id >= num_leaves`. Leaf index = `node_id - num_leaves`.

### leaves.dat Format

Each leaf block starts at the offset recorded in inner.idx:

```
┌───────────────────────────────────────────┐
│ Header (8 bytes):                         │
│   count:   u32 (native-endian)            │
│   padding: u32                            │
├───────────────────────────────────────────┤
│ Entries (count × 16 bytes each):          │
│   lat:    i32 (native-endian)             │
│   lon:    i32 (native-endian)             │
│   doc_id: u64 (native-endian)             │
└───────────────────────────────────────────┘
```

Leaf entries are sorted by (lat, lon, doc_id) within each leaf. Maximum 512 points per leaf (`MAX_POINTS_PER_LEAF`).

### deleted.bin Format

Sorted array of native-endian u64 doc_ids (8 bytes each). Empty file (0 bytes) when no deletions are carried forward.

### Coordinate Encoding (f64 → i32)

Coordinates are encoded to 31-bit signed integers for compact storage:

```
LAT_SCALE = (2^31 - 1) / 180.0
LON_SCALE = (2^31 - 1) / 360.0

encode_lat(lat) = round(lat × LAT_SCALE) as i32
encode_lon(lon) = round(lon × LON_SCALE) as i32

decode_lat(enc) = enc as f64 / LAT_SCALE
decode_lon(enc) = enc as f64 / LON_SCALE
```

This provides sub-centimeter precision (180° / 2^31 ≈ 8.4×10⁻⁸ degrees). The encoding preserves ordering: `lat1 < lat2 ⟹ encode_lat(lat1) < encode_lat(lat2)`.

## 6. Core Algorithms

### Algorithm 1: BKD Tree Build (Recursive Partitioning)

The `build_bkd` function constructs a complete binary tree from an unsorted set of `(EncodedPoint, doc_id)` pairs.

1. **Compute tree size**: `num_leaves = ceil(N / MAX_POINTS_PER_LEAF).next_power_of_two()`, `num_inner = num_leaves - 1`
2. **Recursive partition** (`build_recursive`):
   - At each inner node, choose split dimension by comparing lat spread vs lon spread (`choose_split_dim`)
   - Partition points around the median using `select_nth_unstable_by` (O(n) quickselect)
   - Record `split_value` (median point's coordinate on split dimension) and `split_dim` in the inner node
   - Recurse on left half `[from, mid)` and right half `[mid, to)`
3. **Leaf packing**: When `node_id > num_inner`, sort the leaf's point slice by (lat, lon, doc_id) and write header + entries to `leaf_data`
4. **Empty partitions**: If `from >= to`, still recurse to both children to ensure all leaves get written (empty leaves with count=0)
5. **Write files**: `write_inner_idx` writes inner nodes then leaf offsets; `write_leaf_data` writes the packed leaf buffer

### Algorithm 2: BKD Tree Query (Bbox Classification)

The `CompactedQueryIterator` uses an explicit stack to traverse the BKD tree without recursion.

For each stack frame:
1. **Classify** the cell's bounding box against the query bbox using `classify_bbox`:
   - `Outside` → skip entirely (prune subtree)
   - `Inside` → switch to `CollectAll` mode (skip per-point checks)
   - `Crosses` → recurse into children, scan leaves with bounds checking
2. **Inner node traversal**: Read `(split_value, split_dim)`, push right child then left child (LIFO: left processed first), splitting the cell's bounding box along the split dimension
3. **CollectAll mode**: Recurse into children without classification; at leaves, scan without bounds checking (only deleted set filtering)
4. **Leaf scanning** (`start_leaf_scan` / `next_from_leaf_scan`):
   - If `check_bounds = false`: yield all non-deleted doc_ids
   - If `check_bounds = true`: decode each point and test against the query bounds

### Algorithm 3: BKD Tree Query (Radius → Bbox + Haversine)

Radius queries use a two-stage filter:

1. **Bounding box approximation** (`bounding_box_for_radius`):
   - Compute angular distance: `radius_meters / EARTH_RADIUS_METERS`
   - Latitude range: `center_lat ± angular_distance_deg`, clamped to [-90, 90]
   - Longitude range: expand by `asin(sin(angular_dist) / cos(center_lat_rad))` degrees
   - Near poles or antimeridian: fall back to full longitude span [-180, 180]
2. **Tree traversal**: Uses the bounding box for cell pruning (same as bbox query)
3. **Haversine refinement**: For `Inside` cells, additionally checks if the cell is fully inside the radius circle via `cell_fully_inside_radius` (all 4 corners within `effective_radius = radius - cell_diagonal/2`). If not, falls through to per-point checking
4. **Per-point check**: Decodes each point's (lat, lon), computes haversine distance, yields only if within radius

### Algorithm 4: Snapshot Refresh (Ops Replay)

Replays the chronological ops log to compute the final state:

1. Maintain `insert_map: HashMap<u64, Vec<GeoPoint>>` and `delete_set: HashSet<u64>`
2. For each op:
   - `Insert(point, doc_id)` → remove doc_id from delete_set, add point to insert_map
   - `Delete(doc_id)` → remove doc_id from insert_map, add to delete_set
3. Flatten insert_map into `Vec<(GeoPoint, u64)>` (one entry per point, doc_id may repeat for multi-point)
4. Sort by (encoded_lat, encoded_lon, doc_id)
5. Record `ops_len` for position-based drain during compaction

### Algorithm 5: Filter (Double-Check Pattern)

```rust
pub fn filter(&self, op: GeoFilterOp) -> FilterData {
    let snapshot = {
        let live = self.live.read().unwrap();
        if !live.is_snapshot_dirty() {
            live.get_snapshot()             // Fast path: O(1) Arc clone
        } else {
            drop(live);
            let mut live = self.live.write().unwrap();
            if live.is_snapshot_dirty() {   // Double-check
                live.refresh_snapshot();
            }
            live.get_snapshot()
        }
    };
    let version = self.version.load();
    FilterData::new(Arc::clone(&version), snapshot, op)
}
```

### Algorithm 6: Filter Iterator (Two-Phase)

The `FilterIterator` yields doc_ids in two phases:

1. **Phase 1 — Live inserts**: Linear scan through `snapshot.inserts`, testing each `(point, doc_id)` against `GeoFilterOp::matches()`. Yields matching doc_ids immediately.
2. **Phase 2 — Compacted data**: Streams doc_ids from `CompactedQueryIterator` (BKD tree traversal). Skips doc_ids present in `snapshot.deletes` (the live delete set).

This is not a sorted merge — live results come first, then compacted results. The compacted iterator handles its own deleted set (`deleted_set`) internally via the `CompactedQueryIterator`, while the `FilterIterator` applies the live delete set to compacted results.

### Algorithm 7: Compaction Strategy (Two Paths)

**Decision logic:**
1. Merge all existing points (from BKD tree via `collect_all_points`) with live inserts
2. Deduplicate exact `(EncodedPoint, doc_id)` pairs via sort + dedup
3. Combine delete sets (live deletes + compacted deletes)
4. Filter stale deletes: retain only doc_ids that exist in the merged point set
5. Compute delete ratio: `all_deletes.len() / total_points`

**Path A — Apply deletions** (delete ratio > threshold):
- Remove deleted doc_ids from the point set
- Build new BKD tree from surviving points
- Write empty `deleted.bin`

**Path B — Carry forward deletions** (delete ratio <= threshold):
- Build new BKD tree from all points (including those marked deleted)
- Write merged delete set to `deleted.bin` (sorted)

### Algorithm 8: Atomic Version Swap

1. Write new version files (inner.idx, leaves.dat, deleted.bin) to `versions/{version_id}/`
2. Sync version directory (`fsync`)
3. Atomically write CURRENT file via `CURRENT.tmp` + rename
4. Acquire write lock on LiveLayer:
   a. Load and store new CompactedVersion via ArcSwap
   b. Drain compacted ops: `ops.drain(..snapshot.ops_len)`
   c. Refresh snapshot
5. Release compaction_lock

### Algorithm 9: Haversine Distance

Standard spherical distance formula using Earth radius = 6,371,000 meters:

```
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × asin(√a)
distance = EARTH_RADIUS × c
```

### Algorithm 10: Bounding Box for Radius

Converts a circle on the sphere to a bounding box:

1. `angular_distance = radius / EARTH_RADIUS`
2. `lat_range = center_lat ± angular_distance_deg` (clamped to ±90°)
3. `lon_delta = asin(sin(angular_distance) / cos(center_lat_rad))` (in degrees)
4. **Pole handling**: if lat range touches ±90°, use full longitude span
5. **asin overflow**: if `ratio ≥ 1.0` (near poles with large radius), use full longitude span
6. **Antimeridian**: if `center_lon ± delta_lon` exceeds ±180°, use full longitude span

## 7. Edge Cases Handled

### Spatial Query Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Poles (lat = ±90) | Supported; bbox queries work with full lon span |
| Antimeridian crossing | `bounding_box_for_radius` falls back to full lon span [-180, 180] |
| High-latitude radius query | `asin()` NaN prevention: if ratio ≥ 1.0, use full lon span |
| Zero-area bbox (point query) | Works correctly (min == max) |
| Whole-globe bbox | Works correctly (-90..90, -180..180) |
| Radius at pole | Lat clamped to ±90, full lon span used |
| Large cell near radius boundary | `cell_fully_inside_radius` uses safety margin (half diagonal) to prevent false positives |
| Empty leaves in BKD tree | Handled by count=0 leaf headers |
| NaN/Infinity coordinates | Rejected by `GeoPoint::new()` validation |

### Numeric Edge Cases

| Edge Case | Handling |
|-----------|----------|
| `doc_id = 0` | Fully supported, no special meaning |
| Large doc_ids near `u64::MAX` | Full u64 range supported |
| `GeoPoint::new(91.0, _)` | Returns `Error::InvalidLatitude` |
| `GeoPoint::new(_, 181.0)` | Returns `Error::InvalidLongitude` |
| `GeoPoint::new(NaN, _)` | Returns `Error::InvalidLatitude` |
| `GeoPoint::new(Infinity, _)` | Returns `Error::InvalidLatitude` |
| Encoding roundtrip precision | < 1e-5 degrees (~1 meter) |
| Same doc_id, multiple points | Both points kept (multi-point support) |
| Same (point, doc_id) duplicates | Deduplicated during compaction via sort + dedup |

### Concurrency Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Lock promotion race | Double-check pattern prevents lost updates |
| Version swap during iteration | `FilterData` holds `Arc` to old version; remains valid |
| Snapshot dirty during compaction | Position-based drain (`ops.drain(..ops_len)`) preserves concurrent writes |
| Concurrent deletes during iteration | Snapshot isolation via `Arc<LiveSnapshot>` |
| Remove non-existent doc_id | No-op (adds to ops log, filtered at query time) |
| Delete then re-insert | Snapshot refresh resolves via op replay; re-insert wins if later |
| Multiple concurrent compactions | compaction_lock serializes; all succeed sequentially |
| Concurrent inserts of same doc_id | All ops appended; multi-point semantics after refresh |

### Persistence Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Crash safety | Atomic CURRENT file updates via temp + rename |
| Version recovery | Loads CURRENT; creates empty version if missing |
| Unsupported format version | Returns error with message on `GeoPointStorage::new()` |
| Missing CURRENT file | Creates empty CompactedVersion (version_id = 0) |
| Corrupted deleted.bin | Rejected on load if size not multiple of 8 |
| Empty inner.idx | Treated as no compacted data |
| Stale deletes after compaction | Filtered: only doc_ids present in the point set are retained |

## 8. Public API

### Construction
- `GeoPointStorage::new(base_path: PathBuf, threshold: Threshold) -> Result<Self>` — Create new or open existing index

### JSON Indexing
- `GeoPointIndexer::new(is_array: bool) -> Self` — Create indexer (plain or array mode)
- `GeoPointIndexer::index_json(&self, value: &Value) -> Option<IndexedValue>` — Extract geopoints from JSON `{"lat": f64, "lon": f64}`

### Threshold
- `Threshold::default()` — Returns 0.1
- `Threshold::value(&self) -> f64` — Get threshold value
- `Threshold::try_from(f64) -> Result<Threshold, &'static str>` — Create with custom value (0.0-1.0)
- `Threshold::try_from(f32) -> Result<Threshold, &'static str>` — Create from f32

### GeoPoint
- `GeoPoint::new(lat: f64, lon: f64) -> Result<Self, Error>` — Validated construction
- `lat(&self) -> f64` — Get latitude
- `lon(&self) -> f64` — Get longitude
- `encode(&self) -> EncodedPoint` — Encode to i32 pair

### Writes
- `insert(&self, value: IndexedValue, doc_id: u64)` — Insert geopoint(s) for a doc_id
- `delete(&self, doc_id: u64)` — Mark doc_id as deleted

### Reads
- `filter(&self, op: GeoFilterOp) -> FilterData` — Query with a spatial filter
- `current_version_id(&self) -> u64` — Get current version id

### GeoFilterOp Enum
- `BoundingBox { min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64 }` — Rectangular region query
- `Radius { center: GeoPoint, radius_meters: f64 }` — Circle query (haversine distance)

### GeoFilterOp Methods
- `matches(&self, point: &GeoPoint) -> bool` — Test if a point satisfies the filter

### FilterData
- `iter(&self) -> FilterIterator<'_>` — Returns iterator yielding doc_ids
- Implements `IntoIterator` for `&FilterData`

### IndexedValue Enum
- `Plain(GeoPoint)` — Single geopoint; creates one index entry
- `Array(Vec<GeoPoint>)` — Multiple geopoints; creates one entry per element

### Diagnostics
- `info(&self) -> Result<IndexInfo>` — Get metadata and statistics about the index
- `integrity_check(&self) -> IntegrityCheckResult` — Check integrity of index files

### Maintenance
- `compact(&self, version_id: u64) -> Result<()>` — Merge live layer into compacted BKD tree
- `cleanup(&self)` — Remove old version directories

## 9. Usage Example

```rust
use oramacore_geopoint_index::{
    GeoPointStorage, GeoPoint, GeoFilterOp, IndexedValue, Threshold,
};
use std::path::PathBuf;

let index = GeoPointStorage::new(
    PathBuf::from("/tmp/my_geo_index"),
    Threshold::default(),
)?;

// Insert geopoints
let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
index.insert(IndexedValue::Plain(rome), 1);
index.insert(IndexedValue::Plain(paris), 2);

// Bounding box query
let op = GeoFilterOp::BoundingBox {
    min_lat: 40.0, max_lat: 50.0,
    min_lon: 0.0,  max_lon: 15.0,
};
let results: Vec<u64> = index.filter(op).iter().collect();
// results contains [1, 2]

// Radius query (500 km from Rome)
let op = GeoFilterOp::Radius {
    center: rome,
    radius_meters: 500_000.0,
};
let results: Vec<u64> = index.filter(op).iter().collect();
// results contains [1] (only Rome within 500km)

// Delete and compact
index.delete(2);
index.compact(1)?;

// Cleanup old versions
index.cleanup();
```

## 10. Dependencies

### Production
- `anyhow` — Error handling with context
- `arc-swap` — Lock-free atomic Arc swapping for version pointer
- `memmap2` — Memory-mapped file I/O for efficient disk access
- `serde_json` — JSON parsing for GeoPointIndexer
- `tracing` — Structured logging
- `libc` (Unix only) — madvise optimization for memory mapping

### Optional (cli feature)
- `clap` — CLI argument parsing
- `serde` — Serialization for IndexInfo/IntegrityCheck types

### Development
- `tempfile` — Temporary directories for tests
- `rand` — Random number generation for stress tests

## 11. Key Design Decisions

1. **Two-layer architecture**: LiveLayer (fast writes) + CompactedVersion (BKD tree for efficient spatial reads)
2. **BKD tree over R-tree**: Complete binary tree stored as flat array enables simple mmap layout; no rebalancing needed since tree is rebuilt each compaction
3. **Lock-free version swap**: ArcSwap allows readers to continue during compaction
4. **Ops-log live layer**: Chronological `Vec<LiveOp>` preserves operation ordering for correct delete/re-insert semantics
5. **Lazy snapshot refresh**: Only replays ops and sorts when needed for reads
6. **Position-based drain**: `ops.drain(..ops_len)` after compaction preserves concurrent writes by index, not value
7. **f64→i32 coordinate encoding**: Compact 8-byte representation (vs 16 bytes for two f64), enables integer comparison in BKD tree traversal, sub-centimeter precision is more than sufficient for geographic queries
8. **Power-of-two leaf count**: Ensures complete binary tree shape, simplifying node-to-leaf index mapping (`leaf_index = node_id - num_leaves`)
9. **Widest-spread split dimension**: `choose_split_dim` selects the axis with the largest coordinate range, producing balanced partitions regardless of point distribution
10. **Radius → bbox + haversine**: Coarse bbox pruning avoids expensive haversine computation for distant subtrees; exact haversine only applied at leaf level
11. **Conservative `cell_fully_inside_radius`**: Subtracts half the cell diagonal from the radius before testing corners, preventing false positives from spherical distortion on large cells
12. **Stale delete pruning**: During compaction, deletes for doc_ids not in the point set are discarded, preventing permanent shadowing of future re-inserts
13. **Multi-point support**: A doc_id can have multiple geopoints (e.g., a restaurant with multiple locations); all points are kept independently
14. **Native-endian encoding**: All on-disk integers use native byte order for zero-cost mmap access
15. **madvise hints per file**: inner.idx (WILLNEED — small, read once), leaves.dat (RANDOM — accessed at scattered offsets during query), deleted.bin (SEQUENTIAL — loaded once into HashSet)
16. **Full tree rebuild on compaction**: Simpler than incremental BKD updates; built without locks holding
17. **HashSet for deletions**: O(1) doc_id lookup during query result filtering
18. **Threshold-based compaction**: Avoids expensive garbage collection when delete ratio is low

## 12. Test Coverage Summary

### Unit Tests (in src/ modules)
- **config.rs**: Threshold validation (default, valid range, invalid, f32 conversion)
- **io.rs**: CURRENT file read/write, atomic file operations, version directory helpers
- **point.rs**: GeoPoint validation, encode/decode roundtrip, ordering preservation, EncodedPoint::dim
- **live.rs**: Insert, delete, snapshot refresh, multi-point per doc_id, sorted output, dirty flag, insert-delete-reinsert semantics, ops_len tracking, deletes tracked
- **iterator.rs**: GeoFilterOp::matches for bbox/radius, FilterData with empty/live-only/deleted data, IntoIterator
- **indexer.rs**: GeoPointIndexer plain/array, invalid JSON, invalid coordinates, edge coordinates, missing/null/non-number fields, extra fields, empty object, integer values
- **compacted/mod.rs**: Empty version, load and query single point, load with deletes, collect_all_points, corrupted deleted.bin rejection
- **compacted/build.rs**: Empty input, single point, many points, choose_split_dim lat/lon spread
- **compacted/query.rs**: classify_bbox (Outside/Inside/Crosses), haversine same point, haversine known distance, bounding_box_for_radius (equator, high latitude, antimeridian), cell_fully_inside_radius conservative check
- **storage.rs**: New empty index, insert and filter bbox, delete and filter, compact basic, compact with deletes, persistence, ops during compaction preserved, cleanup, multiple compactions, radius query, incompatible format version

### Integration Tests (tests/integration_tests.rs)
- **Basic CRUD**: Insert + bbox query, insert + radius query, delete removes from results, delete nonexistent is noop
- **Compaction**: Preserves data, applies deletes, multiple rounds, delete after compact
- **Persistence**: Survives close/reopen
- **Edge cases**: doc_id=0, poles, empty index query, single point (zero-area bbox), geopoint validation, whole-globe bbox
- **Multi-point**: Same doc_id with multiple points (live only), same doc_id with other docs across compactions
- **Spatial queries**: Partial overlap bbox, no results bbox, radius after compaction
- **Cleanup**: Preserves current version
- **Concurrency**: Concurrent reads/writes, iterator stability across compaction
- **Stress**: 10K random points with compaction and deletes, 5K+5K incremental compaction

### Concurrency Tests (tests/concurrency_tests.rs)
- **Double-Check Pattern**: Dirty snapshot race (8 readers + 1 writer), snapshot dirty/clean cycle (4 readers + 1 writer)
- **Compaction Serialization**: 4 concurrent compactions, high write rate with compact, interleaved write/compact
- **Version Swap Atomicity**: Filter during version swap (8 readers), FilterData Arc stability
- **Iterator Stability**: Iterator under stress (2 writers + 1 compactor), multiple concurrent iterators (8 threads)
- **Clear-After-Swap**: Writes preserved during compaction, deletes preserved during compaction
- **Multi-Thread Inserts/Deletes**: 8-thread disjoint inserts, alternating hemisphere inserts, same doc_id race, 8-thread disjoint deletes, deletes during snapshot refresh
- **Read Consistency**: 16 readers on static index, 8 readers during dirty snapshot
- **Spatial-Query-Specific**: Concurrent bbox + radius queries, disjoint quadrant queries (no cross-contamination)
- **High Contention Stress**: Mixed ops (4 inserters + 2 deleters + 4 readers + 2 compactors), high delete contention (8 overlapping deleters), insert-delete cycle
- **Edge Cases**: Empty index concurrent queries, doc_id=0 concurrent, large doc_ids near u64::MAX, delete nonexistent concurrent
- **Cleanup**: Cleanup during reads (Arc keeps mmaps alive)
- **Long-Running Stability**: 3-second stress (2 inserters + 1 deleter + 4 readers + 1 compactor)
- **Filter During Compact**: 4 readers during two compaction rounds
