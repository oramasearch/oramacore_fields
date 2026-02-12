//! In-memory buffer for recent writes, with a lazily refreshed sorted snapshot
//! for read operations.

use std::collections::HashSet;
use std::sync::Arc;

/// A single insert or delete operation.
#[derive(Debug, Clone, Copy)]
pub enum LiveOp {
    Insert(bool, u64),
    Delete(u64),
}

/// In-memory operation log with a cached sorted snapshot.
pub struct LiveLayer {
    /// Ordered log of insert/delete operations.
    pub ops: Vec<LiveOp>,
    /// Cached sorted, deduplicated snapshot.
    cached_snapshot: Arc<LiveSnapshot>,
    /// True if ops have changed since the last snapshot refresh.
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

    /// Get a reference-counted clone of the cached snapshot.
    pub fn get_snapshot(&self) -> Arc<LiveSnapshot> {
        Arc::clone(&self.cached_snapshot)
    }

    /// Returns `true` if mutations have occurred since the last snapshot refresh.
    pub fn is_snapshot_dirty(&self) -> bool {
        self.snapshot_dirty
    }

    /// Rebuild the cached snapshot from the operation log.
    ///
    /// Replays all ops to produce sorted, deduplicated vectors of inserts
    /// and deletes.
    pub fn refresh_snapshot(&mut self) {
        let mut true_set: HashSet<u64> = HashSet::new();
        let mut false_set: HashSet<u64> = HashSet::new();
        let mut delete_set: HashSet<u64> = HashSet::new();

        for op in &self.ops {
            match op {
                LiveOp::Insert(true, id) => {
                    true_set.insert(*id);
                    delete_set.remove(id);
                }
                LiveOp::Insert(false, id) => {
                    false_set.insert(*id);
                    delete_set.remove(id);
                }
                LiveOp::Delete(id) => {
                    true_set.remove(id);
                    false_set.remove(id);
                    delete_set.insert(*id);
                }
            }
        }

        let mut true_inserts: Vec<u64> = true_set.into_iter().collect();
        let mut false_inserts: Vec<u64> = false_set.into_iter().collect();
        let mut deletes: Vec<u64> = delete_set.into_iter().collect();

        true_inserts.sort_unstable();
        false_inserts.sort_unstable();
        deletes.sort_unstable();

        self.cached_snapshot = Arc::new(LiveSnapshot {
            true_inserts,
            false_inserts,
            deletes,
            ops_len: self.ops.len(),
        });
        self.snapshot_dirty = false;
    }

    /// Insert a doc_id with the given boolean value.
    /// Invalidates the cached snapshot (will be refreshed lazily on next get_snapshot call).
    pub fn insert(&mut self, value: bool, doc_id: u64) {
        self.ops.push(LiveOp::Insert(value, doc_id));
        self.snapshot_dirty = true;
    }

    /// Mark a doc_id for deletion from both true and false sets.
    /// Invalidates the cached snapshot (will be refreshed lazily on next get_snapshot call).
    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete(doc_id));
        self.snapshot_dirty = true;
    }
}

/// Point-in-time snapshot of the live layer data.
///
/// Contains sorted, deduplicated doc_id vectors. Immutable after creation.
#[derive(Clone)]
pub struct LiveSnapshot {
    /// Sorted, deduplicated doc_ids with value=true.
    pub true_inserts: Vec<u64>,
    /// Sorted, deduplicated doc_ids with value=false.
    pub false_inserts: Vec<u64>,
    /// Sorted, deduplicated doc_ids to delete from both sets.
    pub deletes: Vec<u64>,
    /// Number of ops in the live layer when this snapshot was built.
    /// Used by compaction to drain exactly the right number of ops.
    pub ops_len: usize,
}

impl LiveSnapshot {
    /// Create an empty snapshot.
    pub fn empty() -> Self {
        Self {
            true_inserts: Vec::new(),
            false_inserts: Vec::new(),
            deletes: Vec::new(),
            ops_len: 0,
        }
    }

    /// Get the inserts for a given boolean value.
    pub fn inserts(&self, value: bool) -> &[u64] {
        if value {
            &self.true_inserts
        } else {
            &self.false_inserts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_layer_insert_delete() {
        let mut layer = LiveLayer::new();

        layer.insert(true, 1);
        layer.insert(true, 2);
        layer.insert(false, 10);
        layer.delete(1);
        layer.delete(10);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.true_inserts, vec![2]);
        assert_eq!(snapshot.false_inserts, Vec::<u64>::new());
        assert_eq!(snapshot.deletes, vec![1, 10]);
    }

    #[test]
    fn test_snapshot_deduplication() {
        let mut layer = LiveLayer::new();

        layer.insert(true, 1);
        layer.insert(true, 1);
        layer.insert(true, 2);
        layer.insert(true, 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.true_inserts, vec![1, 2]);
    }

    #[test]
    fn test_deletes_sorted() {
        let mut layer = LiveLayer::new();

        layer.delete(5);
        layer.delete(3);
        layer.delete(10);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.deletes, vec![1, 3, 5, 10]);
    }

    #[test]
    fn test_ordering_insert_delete_reinsert() {
        let mut layer = LiveLayer::new();

        // Insert doc 1 as true, delete it, then re-insert as false
        layer.insert(true, 1);
        layer.delete(1);
        layer.insert(false, 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Doc 1 should be in false_inserts only (re-inserted after delete)
        assert_eq!(snapshot.true_inserts, Vec::<u64>::new());
        assert_eq!(snapshot.false_inserts, vec![1]);
        assert_eq!(snapshot.deletes, Vec::<u64>::new());
    }
}
