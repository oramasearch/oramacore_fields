use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;
use xtri::RadixTree;

use super::indexer::{IndexedValue, TermData};

/// A posting entry: (doc_id, exact_positions, stemmed_positions).
pub type PostingTuple = (u64, SmallVec<[u32; 4]>, SmallVec<[u32; 4]>);

/// A single insert or delete operation.
#[derive(Debug, Clone)]
pub enum LiveOp {
    Insert {
        doc_id: u64,
        field_length: u16,
        terms: HashMap<String, TermData>,
    },
    Delete(u64),
}

/// In-memory buffer of pending inserts and deletes, not yet persisted to disk.
pub struct LiveLayer {
    /// Ordered log of insert and delete operations.
    pub ops: Vec<LiveOp>,
    /// Cached snapshot of the current state.
    cached_snapshot: Arc<LiveSnapshot>,
    snapshot_dirty: bool,
    // Incremental refresh state: maintained across refreshes, reset on compaction drain.
    replay_doc_terms: HashMap<u64, (u16, HashMap<String, TermData>)>,
    replay_delete_set: HashSet<u64>,
    replay_compacted_deletes_count: u64,
    replay_applied_ops: usize,
}

impl Default for LiveLayer {
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            cached_snapshot: Arc::new(LiveSnapshot::empty()),
            snapshot_dirty: false,
            replay_doc_terms: HashMap::new(),
            replay_delete_set: HashSet::new(),
            replay_compacted_deletes_count: 0,
            replay_applied_ops: 0,
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
    /// Uses incremental replay: only processes ops added since the last refresh.
    pub fn refresh_snapshot(&mut self) {
        // Phase 1: Incrementally process only new operations into replay state
        for op in &self.ops[self.replay_applied_ops..] {
            match op {
                LiveOp::Insert {
                    doc_id,
                    field_length,
                    terms,
                } => {
                    self.replay_delete_set.remove(doc_id);
                    self.replay_doc_terms
                        .insert(*doc_id, (*field_length, terms.clone()));
                }
                LiveOp::Delete(doc_id) => {
                    let was_live = self.replay_doc_terms.remove(doc_id).is_some();
                    self.replay_delete_set.insert(*doc_id);
                    if !was_live {
                        self.replay_compacted_deletes_count += 1;
                    }
                }
            }
        }
        self.replay_applied_ops = self.ops.len();

        // Phase 2: Build per-term postings sorted by doc_id
        let mut term_postings: HashMap<String, Vec<PostingTuple>> = HashMap::new();
        let mut doc_lengths: HashMap<u64, u16> = HashMap::new();
        let mut total_field_length: u64 = 0;

        for (doc_id, (field_length, terms)) in &self.replay_doc_terms {
            doc_lengths.insert(*doc_id, *field_length);
            total_field_length += *field_length as u64;

            for (term, data) in terms {
                term_postings.entry(term.clone()).or_default().push((
                    *doc_id,
                    SmallVec::from_slice(&data.exact_positions),
                    SmallVec::from_slice(&data.stemmed_positions),
                ));
            }
        }

        // Sort each term's posting list by doc_id
        for postings in term_postings.values_mut() {
            postings.sort_unstable_by_key(|(doc_id, _, _)| *doc_id);
        }

        let mut deletes_sorted: Vec<u64> =
            self.replay_delete_set.iter().copied().collect();
        deletes_sorted.sort_unstable();

        let total_documents = self.replay_doc_terms.len() as u64;

        let mut term_tree = RadixTree::new();
        for key in term_postings.keys() {
            term_tree.insert(key, ());
        }

        self.cached_snapshot = Arc::new(LiveSnapshot {
            term_postings,
            doc_lengths,
            deletes: Arc::new(self.replay_delete_set.clone()),
            deletes_sorted,
            total_field_length,
            total_documents,
            compacted_deletes_count: self.replay_compacted_deletes_count,
            ops_len: self.ops.len(),
            term_tree,
        });
        self.snapshot_dirty = false;
    }

    /// Drain ops that have been compacted to disk and reset incremental state.
    /// Must be followed by `refresh_snapshot()` to rebuild from remaining ops.
    pub fn drain_compacted_ops(&mut self, count: usize) {
        self.ops.drain(..count);
        self.replay_doc_terms.clear();
        self.replay_delete_set.clear();
        self.replay_compacted_deletes_count = 0;
        self.replay_applied_ops = 0;
        self.snapshot_dirty = true;
    }

    pub fn insert(&mut self, doc_id: u64, value: IndexedValue) {
        self.ops.push(LiveOp::Insert {
            doc_id,
            field_length: value.field_length,
            terms: value.terms,
        });
        self.snapshot_dirty = true;
    }

    pub fn delete(&mut self, doc_id: u64) {
        self.ops.push(LiveOp::Delete(doc_id));
        self.snapshot_dirty = true;
    }
}

/// A point-in-time, read-only view of the live layer's inserts and deletes.
pub struct LiveSnapshot {
    /// Per-term sorted posting lists: (doc_id, exact_positions, stemmed_positions).
    pub term_postings: HashMap<String, Vec<PostingTuple>>,
    /// Per-doc field length.
    pub doc_lengths: HashMap<u64, u16>,
    /// Set of deleted doc_ids.
    pub deletes: Arc<HashSet<u64>>,
    /// Deleted doc_ids in sorted order.
    pub deletes_sorted: Vec<u64>,
    /// Sum of all field lengths of live documents.
    pub total_field_length: u64,
    /// Number of unique live documents.
    pub total_documents: u64,
    /// Number of deletes targeting docs not inserted in the live layer (i.e., compacted docs).
    /// Used to correct total_documents at search time.
    pub compacted_deletes_count: u64,
    /// Number of operations included in this snapshot.
    pub ops_len: usize,
    /// Radix tree for prefix key lookup (keys mirror term_postings).
    term_tree: RadixTree<()>,
}

impl LiveSnapshot {
    pub fn empty() -> Self {
        Self {
            term_postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            deletes: Arc::new(HashSet::new()),
            deletes_sorted: Vec::new(),
            total_field_length: 0,
            total_documents: 0,
            compacted_deletes_count: 0,
            ops_len: 0,
            term_tree: RadixTree::new(),
        }
    }

    /// Callback-based term search: invokes `f(is_exact, postings)` for each matching term.
    ///
    /// - `Some(0)`: exact match only
    /// - `None`: prefix search via RadixTree
    /// - `Some(n)`: fuzzy search via RadixTree with Levenshtein tolerance
    pub fn for_each_term_match<F>(&self, token: &str, tolerance: Option<u8>, mut f: F)
    where
        F: FnMut(bool, &[PostingTuple]),
    {
        match tolerance {
            Some(0) => {
                if let Some(postings) = self.term_postings.get(token) {
                    f(true, postings.as_slice());
                }
            }
            None => {
                for (key_bytes, _) in self.term_tree.search_iter(token, xtri::SearchMode::Prefix) {
                    if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                        if let Some((stored_key, postings)) =
                            self.term_postings.get_key_value(key_str)
                        {
                            let is_exact = stored_key == token;
                            f(is_exact, postings.as_slice());
                        }
                    }
                }
            }
            Some(n) => {
                for (key_bytes, _, _distance) in self.term_tree.search_with_tolerance(token, n) {
                    if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                        if let Some((stored_key, postings)) =
                            self.term_postings.get_key_value(key_str)
                        {
                            let is_exact = stored_key == token;
                            f(is_exact, postings.as_slice());
                        }
                    }
                }
            }
        }
    }

    /// Iterate over all terms and their postings in sorted order.
    pub fn iter_terms_sorted(&self) -> impl Iterator<Item = (&str, &[PostingTuple])> {
        self.term_tree
            .search_iter("", xtri::SearchMode::Prefix)
            .filter_map(|(key_bytes, _)| {
                let key_str = std::str::from_utf8(&key_bytes).ok()?;
                let (stored_key, postings) = self.term_postings.get_key_value(key_str)?;
                Some((stored_key.as_str(), postings.as_slice()))
            })
    }

    /// Iterate over all doc_lengths sorted by doc_id.
    pub fn iter_doc_lengths_sorted(&self) -> Vec<(u64, u16)> {
        let mut entries: Vec<_> = self.doc_lengths.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
        let mut term_map = HashMap::new();
        for (term, exact, stemmed) in terms {
            term_map.insert(
                term.to_string(),
                TermData {
                    exact_positions: exact,
                    stemmed_positions: stemmed,
                },
            );
        }
        IndexedValue {
            field_length,
            terms: term_map,
        }
    }

    #[test]
    fn test_empty_snapshot() {
        let layer = LiveLayer::new();
        let snapshot = layer.get_snapshot();

        let mut count = 0;
        snapshot.for_each_term_match("anything", Some(0), |_, _| count += 1);
        assert_eq!(count, 0);
        assert!(snapshot.doc_lengths.is_empty());
        assert!(snapshot.deletes.is_empty());
        assert_eq!(snapshot.total_field_length, 0);
        assert_eq!(snapshot.total_documents, 0);
        assert_eq!(snapshot.ops_len, 0);
    }

    #[test]
    fn test_insert_and_snapshot() {
        let mut layer = LiveLayer::new();

        layer.insert(
            1,
            make_value(
                3,
                vec![("hello", vec![0], vec![]), ("world", vec![1, 2], vec![])],
            ),
        );
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![1])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.total_documents, 2);
        assert_eq!(snapshot.total_field_length, 5); // 3 + 2

        let mut hello = Vec::new();
        snapshot.for_each_term_match("hello", Some(0), |_, postings| {
            hello = postings.to_vec();
        });
        assert_eq!(hello.len(), 2);
        assert_eq!(hello[0].0, 1); // doc_id 1
        assert_eq!(hello[1].0, 2); // doc_id 2

        let mut world = Vec::new();
        snapshot.for_each_term_match("world", Some(0), |_, postings| {
            world = postings.to_vec();
        });
        assert_eq!(world.len(), 1);
        assert_eq!(world[0].0, 1);

        assert_eq!(*snapshot.doc_lengths.get(&1).unwrap(), 3);
        assert_eq!(*snapshot.doc_lengths.get(&2).unwrap(), 2);
    }

    #[test]
    fn test_delete_removes_doc() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.total_documents, 1);
        assert_eq!(snapshot.total_field_length, 2);
        assert_eq!(snapshot.deletes_sorted, vec![1]);

        let mut hello = Vec::new();
        snapshot.for_each_term_match("hello", Some(0), |_, postings| {
            hello = postings.to_vec();
        });
        assert_eq!(hello.len(), 1);
        assert_eq!(hello[0].0, 2);
    }

    #[test]
    fn test_insert_delete_reinsert() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.delete(1);
        layer.insert(1, make_value(5, vec![("world", vec![0, 1], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.total_documents, 1);
        assert_eq!(snapshot.total_field_length, 5);
        assert!(snapshot.deletes.is_empty());

        let mut hello_count = 0;
        snapshot.for_each_term_match("hello", Some(0), |_, _| hello_count += 1);
        assert_eq!(hello_count, 0);

        let mut world = Vec::new();
        snapshot.for_each_term_match("world", Some(0), |_, postings| {
            world = postings.to_vec();
        });
        assert_eq!(world.len(), 1);
        assert_eq!(world[0].0, 1);
    }

    #[test]
    fn test_snapshot_dirty_flag() {
        let mut layer = LiveLayer::new();
        assert!(!layer.is_snapshot_dirty());

        layer.insert(1, make_value(1, vec![("a", vec![0], vec![])]));
        assert!(layer.is_snapshot_dirty());

        layer.refresh_snapshot();
        assert!(!layer.is_snapshot_dirty());

        layer.delete(1);
        assert!(layer.is_snapshot_dirty());

        layer.refresh_snapshot();
        assert!(!layer.is_snapshot_dirty());
    }

    #[test]
    fn test_ops_len_in_snapshot() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(1, vec![("a", vec![0], vec![])]));
        layer.insert(2, make_value(1, vec![("b", vec![0], vec![])]));
        layer.delete(1);

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();
        assert_eq!(snapshot.ops_len, 3);
    }

    #[test]
    fn test_iter_terms_sorted() {
        let mut layer = LiveLayer::new();

        layer.insert(
            1,
            make_value(
                3,
                vec![
                    ("cherry", vec![2], vec![]),
                    ("apple", vec![0], vec![]),
                    ("banana", vec![1], vec![]),
                ],
            ),
        );

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let keys: Vec<&str> = snapshot.iter_terms_sorted().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_iter_doc_lengths_sorted() {
        let mut layer = LiveLayer::new();

        layer.insert(10, make_value(5, vec![("a", vec![0], vec![])]));
        layer.insert(1, make_value(3, vec![("b", vec![0], vec![])]));
        layer.insert(5, make_value(7, vec![("c", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let doc_lengths = snapshot.iter_doc_lengths_sorted();
        assert_eq!(doc_lengths, vec![(1, 3), (5, 7), (10, 5)]);
    }

    #[test]
    fn test_postings_sorted_by_doc_id() {
        let mut layer = LiveLayer::new();

        layer.insert(10, make_value(1, vec![("term", vec![0], vec![])]));
        layer.insert(1, make_value(1, vec![("term", vec![0], vec![])]));
        layer.insert(5, make_value(1, vec![("term", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let mut postings = Vec::new();
        snapshot.for_each_term_match("term", Some(0), |_, p| {
            postings = p.to_vec();
        });
        let doc_ids: Vec<u64> = postings.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(doc_ids, vec![1, 5, 10]);
    }

    #[test]
    fn test_for_each_term_match_exact() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("apple", Some(0), |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].0); // is_exact
        assert_eq!(results[0].1.len(), 1);
        assert_eq!(results[0].1[0].0, 1); // doc_id

        // Non-existent term
        let mut count = 0;
        snapshot.for_each_term_match("missing", Some(0), |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_for_each_term_match_prefix() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("app", None, |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 2);
        for (is_exact, _) in &results {
            assert!(!is_exact); // not exact
        }

        // Prefix that matches exactly one term
        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("apple", None, |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].0); // is_exact

        // Prefix that matches nothing
        let mut count = 0;
        snapshot.for_each_term_match("xyz", None, |_, _| count += 1);
        assert_eq!(count, 0);

        // Levenshtein with tolerance 1: "apple" matches "apple" (exact, distance 0)
        // and "application" (fuzzy prefix: "appli" is distance 1 from "apple")
        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("apple", Some(1), |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 2);
        let exact_count = results.iter().filter(|(exact, _)| *exact).count();
        assert_eq!(exact_count, 1); // only "apple" is exact
    }

    #[test]
    fn test_for_each_term_match_levenshtein() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("apply", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        // "apple" with tolerance 1 should match "apple" (exact) and "apply" (distance 1)
        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("apple", Some(1), |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 2);

        let exact_count = results.iter().filter(|(is_exact, _)| *is_exact).count();
        assert_eq!(exact_count, 1); // only "apple" is exact

        let all_doc_ids: Vec<u64> = results
            .iter()
            .flat_map(|(_, postings)| postings.iter().map(|(id, _, _)| *id))
            .collect();
        assert!(all_doc_ids.contains(&1)); // "apple"
        assert!(all_doc_ids.contains(&2)); // "apply"
        assert!(!all_doc_ids.contains(&3)); // "banana" is too far

        // "banana" with tolerance 1 should not match "apple" or "apply"
        let mut results: Vec<(bool, Vec<PostingTuple>)> = Vec::new();
        snapshot.for_each_term_match("banana", Some(1), |is_exact, postings| {
            results.push((is_exact, postings.to_vec()));
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].0); // exact match
        assert_eq!(results[0].1[0].0, 3);
    }
}
