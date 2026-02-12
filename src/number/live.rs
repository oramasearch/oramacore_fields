//! In-memory buffer for pending inserts and deletes before compaction.

use super::error::Error;
use super::key::IndexableNumber;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A single operation in the live layer's ordered log.
#[derive(Debug, Clone, Copy)]
pub enum LiveOp<T: IndexableNumber> {
    Insert { value: T, doc_id: u64 },
    Delete { doc_id: u64 },
}

/// Immutable, shareable snapshot of pending inserts and deletes.
#[derive(Debug, Clone)]
pub struct LiveSnapshot<T: IndexableNumber> {
    /// Sorted (value, doc_id) pairs.
    pub inserts: Vec<(T, u64)>,
    /// Set of all deleted doc_ids (needed for compacted filtering).
    pub deletes: Arc<HashSet<u64>>,
    /// Number of ops in the live layer when this snapshot was built.
    /// Used by compaction to drain exactly the right number of ops.
    pub ops_len: usize,
}

impl<T: IndexableNumber> Default for LiveSnapshot<T> {
    fn default() -> Self {
        Self {
            inserts: Vec::new(),
            deletes: Arc::new(HashSet::new()),
            ops_len: 0,
        }
    }
}

/// In-memory buffer for new inserts and deletes before compaction.
pub struct LiveLayer<T: IndexableNumber> {
    /// Chronologically ordered mutation operations.
    pub ops: Vec<LiveOp<T>>,
    /// Cached sorted snapshot.
    cached_snapshot: Arc<LiveSnapshot<T>>,
    /// Whether the snapshot needs to be refreshed.
    snapshot_dirty: bool,
}

impl<T: IndexableNumber> LiveLayer<T> {
    /// Create a new empty LiveLayer.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            cached_snapshot: Arc::new(LiveSnapshot::default()),
            snapshot_dirty: false,
        }
    }

    /// Insert a (value, doc_id) pair.
    ///
    /// The value is validated before insertion.
    pub fn insert(&mut self, value: T, doc_id: u64) -> Result<(), Error> {
        value.validate()?;
        self.ops.push(LiveOp::Insert { value, doc_id });
        self.snapshot_dirty = true;
        Ok(())
    }

    /// Mark a doc_id as deleted.
    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete { doc_id });
        self.snapshot_dirty = true;
    }

    /// Check if the snapshot needs to be refreshed.
    pub fn is_snapshot_dirty(&self) -> bool {
        self.snapshot_dirty
    }

    /// Get the cached snapshot without refreshing.
    ///
    /// Returns the current cached snapshot, which may be stale if
    /// `is_snapshot_dirty()` returns true.
    pub fn get_snapshot(&self) -> Arc<LiveSnapshot<T>> {
        Arc::clone(&self.cached_snapshot)
    }

    /// Rebuild the cached snapshot from the current operations.
    pub fn refresh_snapshot(&mut self) {
        // Forward pass: track latest op index for each insert key and delete key.
        let mut latest_insert: HashMap<([u8; 8], u64), usize> = HashMap::new();
        let mut latest_delete: HashMap<u64, usize> = HashMap::new();

        for (idx, op) in self.ops.iter().enumerate() {
            match *op {
                LiveOp::Insert { value, doc_id } => {
                    latest_insert.insert((value.to_bytes(), doc_id), idx);
                }
                LiveOp::Delete { doc_id } => {
                    latest_delete.insert(doc_id, idx);
                }
            }
        }

        // Keep inserts where insert_idx > delete_idx (or no delete exists).
        let mut inserts: Vec<(T, u64)> = Vec::with_capacity(latest_insert.len());
        for (&(_, doc_id), &insert_idx) in &latest_insert {
            if let Some(&delete_idx) = latest_delete.get(&doc_id) {
                if delete_idx > insert_idx {
                    continue;
                }
            }
            if let LiveOp::Insert { value, doc_id } = self.ops[insert_idx] {
                inserts.push((value, doc_id));
            }
        }

        // Sort by (value, doc_id) for merge compatibility
        inserts.sort_by(|a, b| match T::compare(a.0, b.0) {
            std::cmp::Ordering::Equal => a.1.cmp(&b.1),
            other => other,
        });

        // Snapshot deletes = ALL ever-deleted doc_ids (needed for compacted entry filtering)
        let deletes = Arc::new(latest_delete.keys().copied().collect::<HashSet<u64>>());

        self.cached_snapshot = Arc::new(LiveSnapshot {
            inserts,
            deletes,
            ops_len: self.ops.len(),
        });
        self.snapshot_dirty = false;
    }

    /// Get the number of pending inserts.
    pub fn inserts_len(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, LiveOp::Insert { .. }))
            .count()
    }

    /// Get the number of pending deletes.
    pub fn deletes_len(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, LiveOp::Delete { .. }))
            .count()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl<T: IndexableNumber> Default for LiveLayer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_live_layer() {
        let layer: LiveLayer<u64> = LiveLayer::new();
        assert!(layer.is_empty());
        assert!(!layer.is_snapshot_dirty());
    }

    #[test]
    fn test_insert() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.insert(42, 1).unwrap();
        assert!(!layer.is_empty());
        assert!(layer.is_snapshot_dirty());
        assert_eq!(layer.inserts_len(), 1);
    }

    #[test]
    fn test_insert_nan_rejected() {
        let mut layer: LiveLayer<f64> = LiveLayer::new();

        let result = layer.insert(f64::NAN, 1);
        assert!(result.is_err());
        assert!(layer.is_empty());
    }

    #[test]
    fn test_delete() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.delete(1);
        assert!(!layer.is_empty());
        assert!(layer.is_snapshot_dirty());
        assert_eq!(layer.deletes_len(), 1);
    }

    #[test]
    fn test_refresh_snapshot() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.insert(30, 3).unwrap();
        layer.insert(10, 1).unwrap();
        layer.insert(20, 2).unwrap();
        layer.delete(100);

        layer.refresh_snapshot();

        assert!(!layer.is_snapshot_dirty());

        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.inserts.len(), 3);
        // Should be sorted by (value, doc_id)
        assert_eq!(snapshot.inserts[0], (10, 1));
        assert_eq!(snapshot.inserts[1], (20, 2));
        assert_eq!(snapshot.inserts[2], (30, 3));
        assert!(snapshot.deletes.contains(&100));
    }

    #[test]
    fn test_refresh_snapshot_dedup() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.insert(10, 1).unwrap();
        layer.insert(10, 1).unwrap(); // Duplicate
        layer.insert(20, 2).unwrap();

        layer.refresh_snapshot();

        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.inserts.len(), 2); // Deduplicated
    }

    #[test]
    fn test_drain_compacted() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.insert(10, 1).unwrap();
        layer.insert(20, 2).unwrap();
        layer.delete(100);
        layer.refresh_snapshot();

        let snapshot = layer.get_snapshot();

        // Add new items that shouldn't be cleared
        layer.insert(30, 3).unwrap();
        layer.delete(200);

        // Drain compacted ops by position
        layer.ops.drain(..snapshot.ops_len);

        // Original items should be removed
        assert_eq!(layer.inserts_len(), 1); // Only (30, 3) remains
        assert_eq!(layer.deletes_len(), 1); // Only 200 remains

        // Verify through snapshot that the correct items remain
        layer.refresh_snapshot();
        let snap = layer.get_snapshot();
        assert_eq!(snap.inserts.len(), 1);
        assert_eq!(snap.inserts[0], (30, 3));
        assert!(snap.deletes.contains(&200));
    }

    #[test]
    fn test_snapshot_arc_sharing() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        layer.insert(10, 1).unwrap();
        layer.refresh_snapshot();

        let snapshot1 = layer.get_snapshot();
        let snapshot2 = layer.get_snapshot();

        // Both should point to the same Arc
        assert!(Arc::ptr_eq(&snapshot1, &snapshot2));
    }

    #[test]
    fn test_f64_ordering() {
        let mut layer: LiveLayer<f64> = LiveLayer::new();

        layer.insert(std::f64::consts::PI, 1).unwrap();
        layer.insert(-1.0, 2).unwrap();
        layer.insert(0.0, 3).unwrap();
        layer.insert(std::f64::consts::E, 4).unwrap();

        layer.refresh_snapshot();

        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.inserts[0], (-1.0, 2));
        assert_eq!(snapshot.inserts[1], (0.0, 3));
        assert_eq!(snapshot.inserts[2], (std::f64::consts::E, 4));
        assert_eq!(snapshot.inserts[3], (std::f64::consts::PI, 1));
    }

    #[test]
    fn test_delete_then_reinsert_snapshot() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        // Insert doc 1 at value 10
        layer.insert(10, 1).unwrap();
        // Delete doc 1
        layer.delete(1);
        // Re-insert doc 1 at value 20
        layer.insert(20, 1).unwrap();

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // The re-inserted value should be in inserts
        assert_eq!(snapshot.inserts.len(), 1);
        assert_eq!(snapshot.inserts[0], (20, 1));
        // doc_id 1 should still be in deletes (needed to filter old compacted entries)
        assert!(snapshot.deletes.contains(&1));
    }

    #[test]
    fn test_insert_then_delete_snapshot() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        // Insert then delete
        layer.insert(10, 1).unwrap();
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Inserts should be empty (delete came after insert)
        assert!(snapshot.inserts.is_empty());
        // doc_id should be in deletes
        assert!(snapshot.deletes.contains(&1));
    }

    #[test]
    fn test_multiple_reinserts_snapshot() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        // insert → delete → insert → delete → insert
        layer.insert(10, 1).unwrap();
        layer.delete(1);
        layer.insert(20, 1).unwrap();
        layer.delete(1);
        layer.insert(30, 1).unwrap();

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Should keep latest value (30)
        assert_eq!(snapshot.inserts.len(), 1);
        assert_eq!(snapshot.inserts[0], (30, 1));
        assert!(snapshot.deletes.contains(&1));
    }

    #[test]
    fn test_drain_compacted_preserves_post_snapshot_ops() {
        let mut layer: LiveLayer<u64> = LiveLayer::new();

        // Build up: insert(10,1), insert(20,2), insert(30,3), delete(99)
        layer.insert(10, 1).unwrap();
        layer.insert(20, 2).unwrap();
        layer.insert(30, 3).unwrap();
        layer.delete(99);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Add new items after snapshot
        layer.insert(40, 4).unwrap();
        layer.delete(200);

        // Drain compacted ops by position
        layer.ops.drain(..snapshot.ops_len);

        // Only (40,4) should remain in inserts
        assert_eq!(layer.inserts_len(), 1);
        // Only delete(200) should remain
        assert_eq!(layer.deletes_len(), 1);

        // Verify through snapshot that post-compaction state is correct
        layer.refresh_snapshot();
        let snap = layer.get_snapshot();
        assert_eq!(snap.inserts.len(), 1);
        assert_eq!(snap.inserts[0], (40, 4));
        assert!(snap.deletes.contains(&200));
        // Insert(40,4) should NOT be affected by delete(200)
        assert!(!snap.deletes.contains(&4));
    }
}
