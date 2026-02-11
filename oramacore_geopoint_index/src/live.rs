//! Live layer for in-memory unsorted data and snapshot for sorted iteration.
//!
//! The live layer buffers recent writes in memory before they're compacted to disk.
//! It provides two views of the data:
//!
//! - **Ordered operation log**: Fast O(1) appends for `insert()` and `delete()`
//! - **Sorted snapshot**: Cached sorted+deduplicated view for `filter()` iteration
//!
//! # Snapshot Lifecycle
//!
//! The snapshot is lazily maintained via a dirty flag:
//! 1. Any mutation (`insert`/`delete`) sets `snapshot_dirty = true`
//! 2. `filter()` checks the dirty flag and refreshes if needed
//! 3. `refresh_snapshot()` collapses the op log and clears the dirty flag
//!
//! This avoids sorting on every mutation while ensuring reads see consistent data.
//! Multiple `filter()` calls without intervening mutations share the same `Arc<LiveSnapshot>`.

use crate::point::GeoPoint;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A single operation in the live layer's ordered log.
#[derive(Debug, Clone, Copy)]
pub enum LiveOp {
    Insert(GeoPoint, u64),
    Delete(u64),
}

/// Live layer holding an ordered operation log in memory.
///
/// # Fields
///
/// The `pub` field (`ops`) is exposed for direct manipulation during compaction
/// cleanup, but normal usage should go through the `insert()`, `delete()`, and
/// `get_snapshot()` methods.
///
/// # Thread Safety
///
/// `LiveLayer` itself is not `Sync` - it must be protected by external synchronization
/// (the `GeoPointStorage` uses `RwLock<LiveLayer>`). The `cached_snapshot` is shared via
/// `Arc` and can be safely read from multiple threads after extraction.
pub struct LiveLayer {
    /// Ordered log of insert/delete operations. Preserves temporal ordering.
    pub ops: Vec<LiveOp>,
    /// Cached sorted, deduplicated snapshot. Shared via Arc for cheap cloning.
    cached_snapshot: Arc<LiveSnapshot>,
    /// True if ops have been modified since last `refresh_snapshot()`.
    snapshot_dirty: bool,
}

impl Default for LiveLayer {
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            cached_snapshot: Arc::new(LiveSnapshot::empty()),
            snapshot_dirty: false,
        }
    }
}

impl LiveLayer {
    /// Create an empty live layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cheap Arc clone of the cached snapshot.
    ///
    /// Returns the cached snapshot without modification. Caller should check
    /// `is_snapshot_dirty()` and call `refresh_snapshot()` if needed.
    ///
    /// # Complexity
    ///
    /// O(1) - just an atomic reference count increment.
    pub fn get_snapshot(&self) -> Arc<LiveSnapshot> {
        Arc::clone(&self.cached_snapshot)
    }

    /// Check if the snapshot needs refreshing.
    ///
    /// Returns `true` if any mutations have occurred since the last `refresh_snapshot()`.
    pub fn is_snapshot_dirty(&self) -> bool {
        self.snapshot_dirty
    }

    /// Refresh the cached snapshot by collapsing the operation log.
    ///
    /// Replays the ordered ops to compute the final state, producing sorted
    /// inserts and a delete set.
    ///
    /// Collapsing semantics:
    /// - `Insert(point, id)` → add to insert map (keyed by doc_id), remove from delete_set
    /// - `Delete(id)` → remove from insert map, add to delete_set
    ///
    /// # Complexity
    ///
    /// O(n log n) where n is the number of ops (replay is O(n), sorting is O(m log m)
    /// for inserts of size m).
    ///
    /// # Memory
    ///
    /// Allocates new collections for the snapshot. The old snapshot is dropped when
    /// all `Arc` references to it are released.
    pub fn refresh_snapshot(&mut self) {
        let mut insert_map: HashMap<u64, Vec<GeoPoint>> = HashMap::new();
        let mut delete_set: HashSet<u64> = HashSet::new();

        for op in &self.ops {
            match op {
                LiveOp::Insert(point, doc_id) => {
                    delete_set.remove(doc_id);
                    insert_map.entry(*doc_id).or_default().push(*point);
                }
                LiveOp::Delete(doc_id) => {
                    insert_map.remove(doc_id);
                    delete_set.insert(*doc_id);
                }
            }
        }

        let mut inserts: Vec<(GeoPoint, u64)> = insert_map
            .into_iter()
            .flat_map(|(id, points)| points.into_iter().map(move |p| (p, id)))
            .collect();

        inserts.sort_unstable_by(|a, b| {
            let a_enc = a.0.encode();
            let b_enc = b.0.encode();
            a_enc
                .lat
                .cmp(&b_enc.lat)
                .then(a_enc.lon.cmp(&b_enc.lon))
                .then(a.1.cmp(&b.1))
        });

        self.cached_snapshot = Arc::new(LiveSnapshot {
            inserts,
            deletes: delete_set,
            ops_len: self.ops.len(),
        });
        self.snapshot_dirty = false;
    }

    /// Insert a doc_id with the given point.
    /// Invalidates the cached snapshot (will be refreshed lazily on next get_snapshot call).
    pub fn insert(&mut self, point: GeoPoint, doc_id: u64) {
        self.ops.push(LiveOp::Insert(point, doc_id));
        self.snapshot_dirty = true;
    }

    /// Mark a doc_id for deletion.
    /// Invalidates the cached snapshot (will be refreshed lazily on next get_snapshot call).
    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete(doc_id));
        self.snapshot_dirty = true;
    }
}

/// Sorted snapshot of live data for iteration.
///
/// Contains sorted, deduplicated copies of the live layer data at a point in time.
/// Immutable after creation - shared via `Arc` for zero-copy access from multiple readers.
///
/// # Invariants
///
/// - `inserts` is sorted by (encoded_lat, encoded_lon, doc_id)
/// - A doc_id may appear multiple times in `inserts` (multi-point support)
/// - `deletes` contains doc_ids that were deleted and NOT re-inserted
/// - `deletes` and `inserts` doc_ids are disjoint
#[derive(Clone)]
pub struct LiveSnapshot {
    /// Sorted inserts as (point, doc_id) pairs. A doc_id may appear multiple times.
    pub inserts: Vec<(GeoPoint, u64)>,
    /// Doc_ids deleted from the compacted layer (not re-inserted in this batch).
    pub deletes: HashSet<u64>,
    /// Number of ops in the live layer when this snapshot was built.
    /// Used by compaction to drain exactly the right number of ops.
    pub ops_len: usize,
}

impl LiveSnapshot {
    /// Create an empty snapshot.
    pub fn empty() -> Self {
        Self {
            inserts: Vec::new(),
            deletes: HashSet::new(),
            ops_len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_layer_insert_delete() {
        let mut layer = LiveLayer::new();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap();

        layer.insert(p1, 1);
        layer.insert(p2, 2);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Delete removes from inserts; only doc 2 remains
        assert_eq!(snapshot.inserts.len(), 1);
        assert_eq!(snapshot.inserts[0].1, 2);
        assert!(snapshot.deletes.contains(&1));
    }

    #[test]
    fn test_snapshot_multi_point() {
        let mut layer = LiveLayer::new();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        layer.insert(p1, 1);
        layer.insert(p1, 1); // duplicate: same point + same doc_id → kept (dedup at compaction)
        layer.insert(p1, 2);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // doc_id=1 inserted twice (same point) + doc_id=2 once = 3 entries
        assert_eq!(snapshot.inserts.len(), 3);
    }

    #[test]
    fn test_snapshot_sorted() {
        let mut layer = LiveLayer::new();

        let p1 = GeoPoint::new(30.0, 20.0).unwrap();
        let p2 = GeoPoint::new(10.0, 20.0).unwrap();
        let p3 = GeoPoint::new(20.0, 20.0).unwrap();

        layer.insert(p1, 1);
        layer.insert(p2, 2);
        layer.insert(p3, 3);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.inserts[0].1, 2); // lat 10
        assert_eq!(snapshot.inserts[1].1, 3); // lat 20
        assert_eq!(snapshot.inserts[2].1, 1); // lat 30
    }

    #[test]
    fn test_multi_point_per_doc_id() {
        // With multi-point support, same doc_id with different points keeps both.
        let mut layer = LiveLayer::new();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap();

        layer.insert(p1, 1);
        layer.insert(p2, 1); // same doc_id, different point - both kept

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.inserts.len(), 2);
        // Both entries are for doc_id=1
        assert!(snapshot.inserts.iter().all(|(_, id)| *id == 1));
    }

    #[test]
    fn test_dirty_flag() {
        let mut layer = LiveLayer::new();
        assert!(!layer.is_snapshot_dirty());

        layer.insert(GeoPoint::new(0.0, 0.0).unwrap(), 1);
        assert!(layer.is_snapshot_dirty());

        layer.refresh_snapshot();
        assert!(!layer.is_snapshot_dirty());

        layer.delete(1);
        assert!(layer.is_snapshot_dirty());
    }

    #[test]
    fn test_ordering_insert_delete_reinsert() {
        let mut layer = LiveLayer::new();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap();

        // Insert doc 1 at p1, delete it, then re-insert at p2
        layer.insert(p1, 1);
        layer.delete(1);
        layer.insert(p2, 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Doc 1 should be in inserts at p2 (re-inserted after delete)
        assert_eq!(snapshot.inserts.len(), 1);
        assert_eq!(snapshot.inserts[0].1, 1);
        assert!((snapshot.inserts[0].0.lat() - 30.0).abs() < 1e-6);
        assert!(snapshot.deletes.is_empty());
    }

    #[test]
    fn test_deletes_tracked() {
        let mut layer = LiveLayer::new();

        layer.delete(5);
        layer.delete(3);
        layer.delete(10);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert!(snapshot.deletes.contains(&1));
        assert!(snapshot.deletes.contains(&3));
        assert!(snapshot.deletes.contains(&5));
        assert!(snapshot.deletes.contains(&10));
        assert_eq!(snapshot.deletes.len(), 4);
    }

    #[test]
    fn test_ops_len_tracking() {
        let mut layer = LiveLayer::new();

        layer.insert(GeoPoint::new(10.0, 20.0).unwrap(), 1);
        layer.insert(GeoPoint::new(20.0, 30.0).unwrap(), 2);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.ops_len, 3);

        // Add more ops
        layer.insert(GeoPoint::new(30.0, 40.0).unwrap(), 3);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.ops_len, 4);
    }
}
