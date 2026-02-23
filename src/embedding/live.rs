use super::distance::DistanceFn;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum LiveOp {
    Insert {
        doc_id: u64,
        vectors: Vec<Vec<f32>>,
    },
    Delete {
        doc_id: u64,
    },
}

pub struct LiveLayer {
    pub ops: Vec<LiveOp>,
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

    pub fn refresh_snapshot(&mut self) {
        let mut entries_map: HashMap<u64, Vec<Vec<f32>>> = HashMap::new();
        let mut delete_set: HashSet<u64> = HashSet::new();

        for op in &self.ops {
            match op {
                LiveOp::Insert { doc_id, vectors } => {
                    entries_map.insert(*doc_id, vectors.clone());
                    // If previously deleted in this batch, un-delete
                    delete_set.remove(doc_id);
                }
                LiveOp::Delete { doc_id } => {
                    entries_map.remove(doc_id);
                    delete_set.insert(*doc_id);
                }
            }
        }

        // Flatten multi-embeddings: each vector gets its own entry with repeated doc_id
        let mut entries: Vec<(u64, Vec<f32>)> = entries_map
            .into_iter()
            .flat_map(|(doc_id, vectors)| {
                vectors.into_iter().map(move |v| (doc_id, v))
            })
            .collect();
        entries.sort_unstable_by_key(|(id, _)| *id);
        let mut deletes: Vec<u64> = delete_set.into_iter().collect();
        deletes.sort_unstable();

        self.cached_snapshot = Arc::new(LiveSnapshot {
            entries,
            deletes,
            ops_len: self.ops.len(),
        });
        self.snapshot_dirty = false;
    }

    pub fn insert(&mut self, doc_id: u64, vectors: Vec<Vec<f32>>) {
        self.ops.push(LiveOp::Insert { doc_id, vectors });
        self.snapshot_dirty = true;
    }

    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete { doc_id });
        self.snapshot_dirty = true;
    }
}

#[derive(Clone)]
pub struct LiveSnapshot {
    pub entries: Vec<(u64, Vec<f32>)>,
    pub deletes: Vec<u64>,
    pub ops_len: usize,
}

impl LiveSnapshot {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            deletes: Vec::new(),
            ops_len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.deletes.is_empty()
    }

    /// Brute-force search over live entries.
    /// Returns (doc_id, distance) pairs sorted by distance ascending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        distance_fn: DistanceFn,
        excluded: &[u64],
    ) -> Vec<(u64, f32)> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }

        // Max-heap: we keep the k closest, evicting the farthest
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();

        for (doc_id, vector) in &self.entries {
            if excluded.binary_search(doc_id).is_ok() {
                continue;
            }

            let dist = distance_fn(query, vector);
            if heap.len() < k {
                heap.push(HeapItem {
                    doc_id: *doc_id,
                    distance: dist,
                });
            } else if let Some(top) = heap.peek() {
                if dist < top.distance {
                    heap.pop();
                    heap.push(HeapItem {
                        doc_id: *doc_id,
                        distance: dist,
                    });
                }
            }
        }

        let mut results: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|item| (item.doc_id, item.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[derive(Debug, Clone)]
struct HeapItem {
    doc_id: u64,
    distance: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by distance (largest distance at top)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::distance::l2_distance;

    #[test]
    fn test_live_layer_insert_delete() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![1.0, 0.0]]);
        layer.insert(2, vec![vec![0.0, 1.0]]);
        layer.delete(1);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].0, 2);
        assert_eq!(snapshot.deletes, vec![1]);
    }

    #[test]
    fn test_snapshot_deduplication() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![1.0]]);
        layer.insert(1, vec![vec![2.0]]); // overwrites
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].1, vec![2.0]);
    }

    #[test]
    fn test_reinsert_after_delete() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![1.0]]);
        layer.delete(1);
        layer.insert(1, vec![vec![2.0]]);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].1, vec![2.0]);
        assert!(snapshot.deletes.is_empty());
    }

    #[test]
    fn test_multi_embedding_insert() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![1.0], vec![2.0], vec![3.0]]);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        // All 3 embeddings should survive with the same doc_id
        assert_eq!(snapshot.entries.len(), 3);
        assert!(snapshot.entries.iter().all(|(id, _)| *id == 1));
    }

    #[test]
    fn test_brute_force_search() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![0.0, 0.0]]);
        layer.insert(2, vec![vec![1.0, 0.0]]);
        layer.insert(3, vec![vec![10.0, 0.0]]);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let query = [0.1, 0.0];
        let results = snapshot.search(&query, 2, l2_distance, &[]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // closest: (0,0) dist=0.01
        assert_eq!(results[1].0, 2); // next: (1,0) dist=0.81
    }

    #[test]
    fn test_search_with_exclusion() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![0.0]]);
        layer.insert(2, vec![vec![0.5]]);
        layer.insert(3, vec![vec![1.0]]);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let results = snapshot.search(&[0.0], 3, l2_distance, &[1]);
        assert_eq!(results.len(), 2);
        // Doc 1 excluded
        assert!(results.iter().all(|(id, _)| *id != 1));
    }

    #[test]
    fn test_search_empty() {
        let snapshot = LiveSnapshot::empty();
        let results = snapshot.search(&[1.0], 5, l2_distance, &[]);
        assert!(results.is_empty());
    }
}
