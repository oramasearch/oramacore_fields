use super::distance::Distance;
use super::DocumentFilter;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum LiveOp {
    Insert { doc_id: u64, vectors: Vec<Vec<f32>> },
    Delete { doc_id: u64 },
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
        // Sort by doc_id first, then flatten into contiguous memory
        let mut sorted_entries: Vec<(u64, Vec<Vec<f32>>)> = entries_map.into_iter().collect();
        sorted_entries.sort_unstable_by_key(|(id, _)| *id);

        // Determine dimensions from first vector (0 if no entries)
        let dimensions = sorted_entries
            .first()
            .and_then(|(_, vecs)| vecs.first())
            .map(|v| v.len())
            .unwrap_or(0);

        let total_entries: usize = sorted_entries.iter().map(|(_, vecs)| vecs.len()).sum();
        let mut vectors = Vec::with_capacity(total_entries * dimensions);
        let mut doc_ids = Vec::with_capacity(total_entries);

        for (doc_id, vecs) in sorted_entries {
            for v in vecs {
                doc_ids.push(doc_id);
                vectors.extend_from_slice(&v);
            }
        }

        let mut deletes: Vec<u64> = delete_set.into_iter().collect();
        deletes.sort_unstable();

        self.cached_snapshot = Arc::new(LiveSnapshot {
            entries: FlatEntries {
                vectors,
                doc_ids,
                dimensions,
            },
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

/// Contiguous flat storage for live snapshot entries.
/// Stores all vectors in a single `Vec<f32>` and doc_ids in a parallel `Vec<u64>`,
/// giving cache-friendly sequential access during brute-force scan.
#[derive(Clone)]
pub struct FlatEntries {
    pub vectors: Vec<f32>,
    pub doc_ids: Vec<u64>,
    pub dimensions: usize,
}

impl FlatEntries {
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    pub fn last(&self) -> Option<(&u64, &[f32])> {
        if self.doc_ids.is_empty() {
            return None;
        }
        let i = self.doc_ids.len() - 1;
        let start = i * self.dimensions;
        let end = start + self.dimensions;
        Some((&self.doc_ids[i], &self.vectors[start..end]))
    }

    pub fn iter(&self) -> FlatEntriesIter<'_> {
        FlatEntriesIter {
            entries: self,
            pos: 0,
        }
    }

    pub fn vectors_slice(&self) -> &[f32] {
        &self.vectors
    }

    pub fn doc_ids_slice(&self) -> &[u64] {
        &self.doc_ids
    }
}

impl FlatEntries {
    pub fn get(&self, i: usize) -> (&u64, &[f32]) {
        let start = i * self.dimensions;
        let end = start + self.dimensions;
        (&self.doc_ids[i], &self.vectors[start..end])
    }
}

pub struct FlatEntriesIter<'a> {
    entries: &'a FlatEntries,
    pos: usize,
}

impl<'a> Iterator for FlatEntriesIter<'a> {
    type Item = (&'a u64, &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.entries.doc_ids.len() {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        let start = i * self.entries.dimensions;
        let end = start + self.entries.dimensions;
        Some((&self.entries.doc_ids[i], &self.entries.vectors[start..end]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.doc_ids.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for FlatEntriesIter<'a> {}

impl<'a> IntoIterator for &'a FlatEntries {
    type Item = (&'a u64, &'a [f32]);
    type IntoIter = FlatEntriesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone)]
pub struct LiveSnapshot {
    pub entries: FlatEntries,
    pub deletes: Vec<u64>,
    pub ops_len: usize,
}

impl LiveSnapshot {
    pub fn empty() -> Self {
        Self {
            entries: FlatEntries {
                vectors: Vec::new(),
                doc_ids: Vec::new(),
                dimensions: 0,
            },
            deletes: Vec::new(),
            ops_len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.deletes.is_empty()
    }

    /// Brute-force search reusing caller-provided buffers.
    /// Results are left in `ctx.live_results` (sorted by distance ascending).
    pub fn search_with_context<D: Distance, F: DocumentFilter>(
        &self,
        query: &[f32],
        k: usize,
        excluded: &[u64],
        filter: Option<&F>,
        ctx: &mut super::search_context::LiveSearchBuffers,
    ) {
        if self.entries.is_empty() || k == 0 {
            ctx.live_results.clear();
            return;
        }

        // Max-heap: we keep the k closest, evicting the farthest
        ctx.live_heap.clear();

        for (doc_id, vector) in &self.entries {
            if excluded.binary_search(doc_id).is_ok() {
                continue;
            }
            if let Some(f) = filter {
                if !f.contains(*doc_id) {
                    continue;
                }
            }

            let dist = D::distance(query, vector);
            if ctx.live_heap.len() < k {
                ctx.live_heap.push(HeapItem {
                    doc_id: *doc_id,
                    distance: dist,
                });
            } else if let Some(top) = ctx.live_heap.peek() {
                if dist < top.distance {
                    ctx.live_heap.pop();
                    ctx.live_heap.push(HeapItem {
                        doc_id: *doc_id,
                        distance: dist,
                    });
                }
            }
        }

        ctx.live_results.clear();
        while let Some(item) = ctx.live_heap.pop() {
            ctx.live_results.push((item.doc_id, item.distance));
        }
        ctx.live_results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HeapItem {
    pub(crate) doc_id: u64,
    pub(crate) distance: f32,
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
    use crate::embedding::distance::L2;

    #[test]
    fn test_live_layer_insert_delete() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![1.0, 0.0]]);
        layer.insert(2, vec![vec![0.0, 1.0]]);
        layer.delete(1);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(*snapshot.entries.get(0).0, 2);
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
        assert_eq!(snapshot.entries.get(0).1, vec![2.0]);
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
        assert_eq!(snapshot.entries.get(0).1, vec![2.0]);
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
        let mut ctx = crate::embedding::SearchContext::new();
        snapshot.search_with_context::<L2, crate::embedding::NoFilter>(
            &query,
            2,
            &[],
            None,
            &mut ctx.inner.live,
        );
        assert_eq!(ctx.inner.live.live_results.len(), 2);
        assert_eq!(ctx.inner.live.live_results[0].0, 1); // closest: (0,0) dist=0.01
        assert_eq!(ctx.inner.live.live_results[1].0, 2); // next: (1,0) dist=0.81
    }

    #[test]
    fn test_search_with_exclusion() {
        let mut layer = LiveLayer::new();
        layer.insert(1, vec![vec![0.0]]);
        layer.insert(2, vec![vec![0.5]]);
        layer.insert(3, vec![vec![1.0]]);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let mut ctx = crate::embedding::SearchContext::new();
        snapshot.search_with_context::<L2, crate::embedding::NoFilter>(
            &[0.0],
            3,
            &[1],
            None,
            &mut ctx.inner.live,
        );
        assert_eq!(ctx.inner.live.live_results.len(), 2);
        // Doc 1 excluded
        assert!(ctx.inner.live.live_results.iter().all(|(id, _)| *id != 1));
    }

    #[test]
    fn test_search_empty() {
        let snapshot = LiveSnapshot::empty();
        let mut ctx = crate::embedding::SearchContext::new();
        snapshot.search_with_context::<L2, crate::embedding::NoFilter>(
            &[1.0],
            5,
            &[],
            None,
            &mut ctx.inner.live,
        );
        assert!(ctx.inner.live.live_results.is_empty());
    }
}
