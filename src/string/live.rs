use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xtri::RadixTree;

use super::indexer::{IndexedValue, TermData};

/// A posting entry: (doc_id, exact_positions, stemmed_positions).
pub type PostingTuple = (u64, Vec<u32>, Vec<u32>);

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
        // Replay ops to build current state
        // Per doc: latest field_length, per-term positions
        let mut doc_terms: HashMap<u64, (u16, HashMap<String, TermData>)> = HashMap::new();
        let mut delete_set: HashSet<u64> = HashSet::new();

        for op in &self.ops {
            match op {
                LiveOp::Insert {
                    doc_id,
                    field_length,
                    terms,
                } => {
                    delete_set.remove(doc_id);
                    doc_terms.insert(*doc_id, (*field_length, terms.clone()));
                }
                LiveOp::Delete(doc_id) => {
                    doc_terms.remove(doc_id);
                    delete_set.insert(*doc_id);
                }
            }
        }

        // Build per-term postings sorted by doc_id
        let mut term_postings: HashMap<String, Vec<PostingTuple>> = HashMap::new();
        let mut doc_lengths: HashMap<u64, u16> = HashMap::new();
        let mut total_field_length: u64 = 0;

        for (doc_id, (field_length, terms)) in &doc_terms {
            doc_lengths.insert(*doc_id, *field_length);
            total_field_length += *field_length as u64;

            for (term, data) in terms {
                term_postings
                    .entry(term.clone())
                    .or_default()
                    .push((*doc_id, data.exact_positions.clone(), data.stemmed_positions.clone()));
            }
        }

        // Sort each term's posting list by doc_id
        for postings in term_postings.values_mut() {
            postings.sort_unstable_by_key(|(doc_id, _, _)| *doc_id);
        }

        let mut deletes_sorted: Vec<u64> = delete_set.iter().copied().collect();
        deletes_sorted.sort_unstable();

        let total_documents = doc_terms.len() as u64;

        let mut term_tree = RadixTree::new();
        for key in term_postings.keys() {
            term_tree.insert(key, ());
        }

        self.cached_snapshot = Arc::new(LiveSnapshot {
            term_postings,
            doc_lengths,
            deletes: Arc::new(delete_set),
            deletes_sorted,
            total_field_length,
            total_documents,
            ops_len: self.ops.len(),
            term_tree,
        });
        self.snapshot_dirty = false;
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
            ops_len: 0,
            term_tree: RadixTree::new(),
        }
    }

    /// Return the postings for a term, or an empty slice if not found.
    pub fn postings_for_term(&self, term: &str) -> &[PostingTuple] {
        self.term_postings.get(term).map_or(&[], |v| v.as_slice())
    }

    /// Search for terms matching the given token and tolerance.
    ///
    /// Returns `(matched_term, is_exact, postings)` triples.
    /// - `Some(0)`: exact match only
    /// - `None`: prefix search via RadixTree
    /// - `Some(n)`: not supported in live layer, returns empty
    pub fn search_terms<'a>(&'a self, token: &'a str, tolerance: Option<u8>) -> Vec<(&'a str, bool, &'a [PostingTuple])> {
        match tolerance {
            Some(0) => {
                // Exact match
                if let Some(postings) = self.term_postings.get(token) {
                    vec![(token, true, postings.as_slice())]
                } else {
                    vec![]
                }
            }
            None => {
                // Prefix search
                let mut results = Vec::new();
                for (key_bytes, _) in self.term_tree.search_iter(token, xtri::SearchMode::Prefix) {
                    if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                        if let Some((stored_key, postings)) = self.term_postings.get_key_value(key_str) {
                            let is_exact = stored_key == token;
                            results.push((stored_key.as_str(), is_exact, postings.as_slice()));
                        }
                    }
                }
                results
            }
            Some(_) => {
                // Levenshtein not supported in live layer
                vec![]
            }
        }
    }

    /// Iterate over all terms and their postings in sorted order.
    pub fn iter_terms_sorted(&self) -> Vec<(&str, &[PostingTuple])> {
        let mut entries: Vec<_> = self
            .term_postings
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        entries.sort_unstable_by_key(|(k, _)| *k);
        entries
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

        assert!(snapshot.postings_for_term("anything").is_empty());
        assert!(snapshot.doc_lengths.is_empty());
        assert!(snapshot.deletes.is_empty());
        assert_eq!(snapshot.total_field_length, 0);
        assert_eq!(snapshot.total_documents, 0);
        assert_eq!(snapshot.ops_len, 0);
    }

    #[test]
    fn test_insert_and_snapshot() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![]), ("world", vec![1, 2], vec![])]));
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![1])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        assert_eq!(snapshot.total_documents, 2);
        assert_eq!(snapshot.total_field_length, 5); // 3 + 2

        let hello = snapshot.postings_for_term("hello");
        assert_eq!(hello.len(), 2);
        assert_eq!(hello[0].0, 1); // doc_id 1
        assert_eq!(hello[1].0, 2); // doc_id 2

        let world = snapshot.postings_for_term("world");
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

        let hello = snapshot.postings_for_term("hello");
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

        assert!(snapshot.postings_for_term("hello").is_empty());
        let world = snapshot.postings_for_term("world");
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

        layer.insert(1, make_value(3, vec![
            ("cherry", vec![2], vec![]),
            ("apple", vec![0], vec![]),
            ("banana", vec![1], vec![]),
        ]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let terms = snapshot.iter_terms_sorted();
        let keys: Vec<&str> = terms.iter().map(|(k, _)| *k).collect();
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

        let postings = snapshot.postings_for_term("term");
        let doc_ids: Vec<u64> = postings.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(doc_ids, vec![1, 5, 10]);
    }

    #[test]
    fn test_search_terms_exact() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let results = snapshot.search_terms("apple", Some(0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
        assert!(results[0].1); // is_exact
        assert_eq!(results[0].2.len(), 1);
        assert_eq!(results[0].2[0].0, 1); // doc_id

        // Non-existent term
        let results = snapshot.search_terms("missing", Some(0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_terms_prefix() {
        let mut layer = LiveLayer::new();

        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let mut results = snapshot.search_terms("app", None);
        results.sort_by_key(|(k, _, _)| k.to_string());
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "apple");
        assert!(!results[0].1); // not exact
        assert_eq!(results[1].0, "application");
        assert!(!results[1].1); // not exact

        // Prefix that matches exactly one term
        let results = snapshot.search_terms("apple", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
        assert!(results[0].1); // is_exact

        // Prefix that matches nothing
        let results = snapshot.search_terms("xyz", None);
        assert!(results.is_empty());

        // Levenshtein not supported in live layer
        let results = snapshot.search_terms("apple", Some(1));
        assert!(results.is_empty());
    }
}
