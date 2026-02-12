//! In-memory buffer for recent writes before they are compacted to disk.
//!
//! Provides an operation log for inserts and deletes, and a lazily-refreshed
//! sorted snapshot for filter queries.

use super::point::GeoPoint;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A single insert or delete operation.
#[derive(Debug, Clone, Copy)]
pub enum LiveOp {
    Insert(GeoPoint, u64),
    Delete(u64),
}

/// In-memory buffer that records insert and delete operations and caches a sorted snapshot.
pub struct LiveLayer {
    /// Ordered log of insert/delete operations.
    pub ops: Vec<LiveOp>,
    /// Cached sorted snapshot of the current state.
    cached_snapshot: Arc<LiveSnapshot>,
    /// Whether ops have changed since the last snapshot refresh.
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
    /// Creates an empty live layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a shared reference to the cached snapshot.
    pub fn get_snapshot(&self) -> Arc<LiveSnapshot> {
        Arc::clone(&self.cached_snapshot)
    }

    /// Returns `true` if mutations have occurred since the last snapshot refresh.
    pub fn is_snapshot_dirty(&self) -> bool {
        self.snapshot_dirty
    }

    /// Rebuilds the cached snapshot from the operation log.
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

    /// Adds an insert operation for the given point and doc_id.
    pub fn insert(&mut self, point: GeoPoint, doc_id: u64) {
        self.ops.push(LiveOp::Insert(point, doc_id));
        self.snapshot_dirty = true;
    }

    /// Adds a delete operation for the given doc_id.
    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete(doc_id));
        self.snapshot_dirty = true;
    }
}

/// Point-in-time view of live data, with sorted inserts and a set of deleted doc_ids.
#[derive(Clone)]
pub struct LiveSnapshot {
    /// Sorted inserts as (point, doc_id) pairs.
    pub inserts: Vec<(GeoPoint, u64)>,
    /// Doc_ids marked for deletion.
    pub deletes: HashSet<u64>,
    /// Number of ops in the live layer when this snapshot was built.
    pub ops_len: usize,
}

impl LiveSnapshot {
    /// Creates an empty snapshot.
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
