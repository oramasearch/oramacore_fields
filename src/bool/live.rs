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

use std::collections::HashSet;
use std::sync::Arc;

/// A single operation in the live layer's ordered log.
#[derive(Debug, Clone, Copy)]
pub enum LiveOp {
    Insert(bool, u64),
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
/// (the `BoolStorage` uses `RwLock<LiveLayer>`). The `cached_snapshot` is shared via
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
    /// Replays the ordered ops to compute the final state, producing sorted,
    /// deduplicated vectors for each category.
    ///
    /// Collapsing semantics (additive inserts):
    /// - `Insert(value, id)` → add to target set, remove from delete_set
    /// - `Delete(id)` → remove from both true/false sets, add to delete_set
    ///
    /// # Complexity
    ///
    /// O(n log n) where n is the number of ops (replay is O(n), sorting is O(m log m)
    /// for each output set of size m).
    ///
    /// # Memory
    ///
    /// Allocates new vectors for the snapshot. The old snapshot is dropped when
    /// all `Arc` references to it are released.
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

/// Sorted snapshot of live data for iteration.
///
/// Contains sorted, deduplicated copies of the live layer data at a point in time.
/// Immutable after creation - shared via `Arc` for zero-copy access from multiple readers.
///
/// # Invariants
///
/// - All vectors are sorted in ascending order
/// - All vectors are deduplicated (no repeated values)
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
