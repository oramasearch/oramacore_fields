use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A single insert or delete operation.
#[derive(Debug, Clone)]
pub enum LiveOp {
    Insert(String, u64), // (key, doc_id)
    Delete(u64),         // doc_id
}

/// In-memory buffer of pending inserts and deletes, not yet persisted to disk.
pub struct LiveLayer {
    /// Ordered log of insert and delete operations.
    pub ops: Vec<LiveOp>,
    /// Cached snapshot of the current state.
    cached_snapshot: Arc<LiveSnapshot>,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_snapshot(&self) -> Arc<LiveSnapshot> {
        Arc::clone(&self.cached_snapshot)
    }

    pub fn is_snapshot_dirty(&self) -> bool {
        self.snapshot_dirty
    }

    /// Rebuild the cached snapshot from the current operations log.
    pub fn refresh_snapshot(&mut self) {
        let mut insert_map: HashMap<u64, Vec<String>> = HashMap::new();
        let mut delete_set: HashSet<u64> = HashSet::new();

        for op in &self.ops {
            match op {
                LiveOp::Insert(key, doc_id) => {
                    delete_set.remove(doc_id);
                    insert_map.entry(*doc_id).or_default().push(key.clone());
                }
                LiveOp::Delete(doc_id) => {
                    insert_map.remove(doc_id);
                    delete_set.insert(*doc_id);
                }
            }
        }

        // Flatten insert_map into (key, doc_id) pairs
        let mut pairs: Vec<(String, u64)> = Vec::new();
        for (doc_id, keys) in &insert_map {
            for key in keys {
                pairs.push((key.clone(), *doc_id));
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        pairs.dedup();

        // Build columnar format from sorted (key, doc_id) pairs
        let mut keys = Vec::new();
        let mut ranges = vec![0usize];
        let mut doc_ids = Vec::with_capacity(pairs.len());

        for (key, doc_id) in &pairs {
            if keys.last() != Some(key) {
                if !keys.is_empty() {
                    ranges.push(doc_ids.len());
                }
                keys.push(key.clone());
            }
            doc_ids.push(*doc_id);
        }
        ranges.push(doc_ids.len());

        let mut deletes_sorted: Vec<u64> = delete_set.iter().copied().collect();
        deletes_sorted.sort_unstable();

        self.cached_snapshot = Arc::new(LiveSnapshot {
            keys,
            ranges,
            doc_ids,
            deletes: Arc::new(delete_set),
            deletes_sorted,
            ops_len: self.ops.len(),
        });
        self.snapshot_dirty = false;
    }

    pub fn insert(&mut self, value: &str, doc_id: u64) {
        self.ops.push(LiveOp::Insert(value.to_string(), doc_id));
        self.snapshot_dirty = true;
    }

    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete(doc_id));
        self.snapshot_dirty = true;
    }
}

/// A point-in-time, read-only view of the live layer's inserts and deletes.
#[derive(Clone)]
pub struct LiveSnapshot {
    /// Unique keys in sorted order.
    keys: Vec<String>,
    /// Index boundaries into `doc_ids` for each key.
    ranges: Vec<usize>,
    /// All doc_ids, grouped by key.
    doc_ids: Vec<u64>,
    /// Set of deleted doc_ids.
    pub deletes: Arc<HashSet<u64>>,
    /// Deleted doc_ids in sorted order.
    pub deletes_sorted: Vec<u64>,
    /// Number of operations included in this snapshot.
    pub ops_len: usize,
}

impl LiveSnapshot {
    pub fn empty() -> Self {
        Self {
            keys: Vec::new(),
            ranges: vec![0],
            doc_ids: Vec::new(),
            deletes: Arc::new(HashSet::new()),
            deletes_sorted: Vec::new(),
            ops_len: 0,
        }
    }

    /// Return the sorted doc_ids for a given key, or an empty slice if not found.
    pub fn doc_ids_for_key(&self, key: &str) -> &[u64] {
        match self.keys.binary_search_by(|k| k.as_str().cmp(key)) {
            Ok(idx) => &self.doc_ids[self.ranges[idx]..self.ranges[idx + 1]],
            Err(_) => &[],
        }
    }

    /// Return the total number of doc_ids across all keys.
    pub fn total_doc_ids(&self) -> usize {
        self.doc_ids.len()
    }

    /// Iterate over all entries as `(key, doc_ids)` pairs.
    pub fn iter_entries(&self) -> impl Iterator<Item = (&str, &[u64])> {
        self.keys.iter().enumerate().map(move |(i, key)| {
            (
                key.as_str(),
                &self.doc_ids[self.ranges[i]..self.ranges[i + 1]],
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_layer_insert_delete() {
        let mut layer = LiveLayer::new();

        layer.insert("hello", 1);
        layer.insert("hello", 2);
        layer.insert("world", 10);
        layer.delete(1);
        layer.delete(10);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // After replay: insert(hello,1), insert(hello,2), insert(world,10), delete(1), delete(10)
        // Only (hello, 2) remains in inserts; deletes = {1, 10}
        assert_eq!(snapshot.total_doc_ids(), 1);
        assert_eq!(snapshot.deletes_sorted, vec![1, 10]);
    }

    #[test]
    fn test_snapshot_deduplication() {
        let mut layer = LiveLayer::new();

        layer.insert("hello", 1);
        layer.insert("hello", 1);
        layer.insert("hello", 2);
        layer.insert("hello", 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // After dedup, should have doc_ids [1, 2] for "hello"
        assert_eq!(snapshot.total_doc_ids(), 2);
    }

    #[test]
    fn test_doc_ids_for_key() {
        let mut layer = LiveLayer::new();

        layer.insert("apple", 1);
        layer.insert("banana", 2);
        layer.insert("apple", 3);
        layer.insert("cherry", 4);
        layer.insert("banana", 5);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.doc_ids_for_key("apple"), &[1, 3]);
        assert_eq!(snapshot.doc_ids_for_key("banana"), &[2, 5]);
        assert!(snapshot.doc_ids_for_key("missing").is_empty());
    }

    #[test]
    fn test_inserts_sorted() {
        let mut layer = LiveLayer::new();

        layer.insert("cherry", 5);
        layer.insert("apple", 3);
        layer.insert("banana", 1);
        layer.insert("apple", 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let entries: Vec<(&str, &[u64])> = snapshot.iter_entries().collect();
        assert_eq!(
            entries,
            vec![
                ("apple", &[1, 3][..]),
                ("banana", &[1][..]),
                ("cherry", &[5][..]),
            ]
        );
    }

    #[test]
    fn test_insert_delete_reinsert() {
        let mut layer = LiveLayer::new();

        layer.insert("foo", 1);
        layer.delete(1);
        layer.insert("bar", 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // After replay: insert(foo,1) -> delete(1) -> insert(bar,1)
        // Only (bar, 1) should be in inserts, deletes should be empty
        assert_eq!(snapshot.doc_ids_for_key("bar"), &[1]);
        assert!(snapshot.doc_ids_for_key("foo").is_empty());
        assert!(snapshot.deletes_sorted.is_empty());
    }

    #[test]
    fn test_ops_len_in_snapshot() {
        let mut layer = LiveLayer::new();

        layer.insert("a", 1);
        layer.insert("b", 2);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.ops_len, 3);

        // Add more ops, refresh again
        layer.insert("c", 3);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.ops_len, 4);
    }

    #[test]
    fn test_multiple_keys_for_same_doc_id() {
        let mut layer = LiveLayer::new();

        layer.insert("color", 1);
        layer.insert("shape", 1);
        layer.insert("size", 1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.doc_ids_for_key("color"), &[1]);
        assert_eq!(snapshot.doc_ids_for_key("shape"), &[1]);
        assert_eq!(snapshot.doc_ids_for_key("size"), &[1]);
        assert_eq!(snapshot.total_doc_ids(), 3);
    }

    #[test]
    fn test_delete_removes_from_all_keys() {
        let mut layer = LiveLayer::new();

        layer.insert("color", 1);
        layer.insert("shape", 1);
        layer.insert("color", 2);
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.doc_ids_for_key("color"), &[2]);
        assert!(snapshot.doc_ids_for_key("shape").is_empty());
        assert_eq!(snapshot.deletes_sorted, vec![1]);
    }

    #[test]
    fn test_empty_snapshot() {
        let layer = LiveLayer::new();
        let snapshot = layer.get_snapshot();

        assert!(snapshot.doc_ids_for_key("anything").is_empty());
        assert_eq!(snapshot.total_doc_ids(), 0);
        assert!(snapshot.deletes.is_empty());
        assert!(snapshot.deletes_sorted.is_empty());
        assert_eq!(snapshot.ops_len, 0);
    }

    #[test]
    fn test_snapshot_dirty_flag() {
        let mut layer = LiveLayer::new();
        assert!(!layer.is_snapshot_dirty());

        layer.insert("hello", 1);
        assert!(layer.is_snapshot_dirty());

        layer.refresh_snapshot();
        assert!(!layer.is_snapshot_dirty());

        layer.delete(1);
        assert!(layer.is_snapshot_dirty());

        layer.refresh_snapshot();
        assert!(!layer.is_snapshot_dirty());
    }

    #[test]
    fn test_delete_all_then_reinsert() {
        let mut layer = LiveLayer::new();

        layer.insert("hello", 1);
        layer.insert("hello", 2);
        layer.delete(1);
        layer.delete(2);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert!(snapshot.doc_ids_for_key("hello").is_empty());
        assert_eq!(snapshot.total_doc_ids(), 0);
        assert_eq!(snapshot.deletes_sorted, vec![1, 2]);

        layer.insert("hello", 3);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.doc_ids_for_key("hello"), &[3]);
    }

    #[test]
    fn test_iter_entries_empty() {
        let layer = LiveLayer::new();
        let snapshot = layer.get_snapshot();
        let entries: Vec<_> = snapshot.iter_entries().collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_many_operations() {
        let mut layer = LiveLayer::new();

        for i in 0..100u64 {
            layer.insert(&format!("key_{:03}", i % 10), i);
        }
        // Delete every third
        for i in (0..100u64).step_by(3) {
            layer.delete(i);
        }

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // Verify keys are sorted
        let entries: Vec<_> = snapshot.iter_entries().collect();
        for w in entries.windows(2) {
            assert!(w[0].0 < w[1].0, "keys not sorted");
        }

        // Verify total: 100 inserts - 34 deletes = 66 remaining
        // But delete removes the doc from inserts entirely, so count the inserts
        let expected_inserts: usize = (0..100u64).filter(|i| i % 3 != 0).count();
        assert_eq!(snapshot.total_doc_ids(), expected_inserts);
    }

    #[test]
    fn test_snapshot_doc_ids_sorted_within_key() {
        let mut layer = LiveLayer::new();

        // Insert doc_ids in reverse order
        layer.insert("hello", 10);
        layer.insert("hello", 5);
        layer.insert("hello", 1);
        layer.insert("hello", 8);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let ids = snapshot.doc_ids_for_key("hello");
        assert_eq!(ids, &[1, 5, 8, 10], "doc_ids should be sorted ascending");
    }
}
