use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::iterator::{ScoredDoc, SearchResult};

/// Private wrapper for BinaryHeap ordering.
/// Orders by score descending, then doc_id ascending (consistent with sort_by_score).
struct HeapEntry {
    doc_id: u64,
    score: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.doc_id.cmp(&self.doc_id))
    }
}

/// Accumulates BM25 scores across one or more string fields.
///
/// Pass the same `BM25Scorer` to multiple `StringStorage::search` calls to
/// collect cross-field scores. Call [`into_search_result`](BM25Scorer::into_search_result)
/// when done to get a sorted result set.
pub struct BM25Scorer {
    document_scores: HashMap<u64, f32>,
    threshold: Option<ThresholdState>,
}

struct ThresholdState {
    document_token_masks: HashMap<u64, u32>,
    min_tokens: u32,
}

impl BM25Scorer {
    /// Create a scorer with no threshold filtering.
    pub fn new() -> Self {
        Self {
            document_scores: HashMap::new(),
            threshold: None,
        }
    }

    /// Create a scorer that filters out documents matching fewer than `min_tokens` distinct tokens.
    pub fn with_threshold(min_tokens: u32) -> Self {
        Self {
            document_scores: HashMap::new(),
            threshold: Some(ThresholdState {
                document_token_masks: HashMap::new(),
                min_tokens,
            }),
        }
    }

    /// Multiply an existing document's score by `factor`. No-op if the document hasn't been added.
    pub fn multiply_score(&mut self, doc_id: u64, factor: f32) {
        if let Some(score) = self.document_scores.get_mut(&doc_id) {
            *score *= factor;
        }
    }

    /// Add a score contribution for a document. `token_mask` is a bitmask indicating which
    /// query tokens this document matched (bit *i* set ⇒ token *i* matched).
    pub fn add(&mut self, doc_id: u64, score: f32, token_mask: u32) {
        *self.document_scores.entry(doc_id).or_insert(0.0) += score;
        if let Some(ref mut state) = self.threshold {
            *state.document_token_masks.entry(doc_id).or_insert(0) |= token_mask;
        }
    }

    /// Consume the scorer and return per-document scores (filtered by threshold if set).
    pub fn get_scores(self) -> HashMap<u64, f32> {
        match self.threshold {
            None => self.document_scores,
            Some(state) => self
                .document_scores
                .into_iter()
                .filter(|(doc_id, _)| {
                    state
                        .document_token_masks
                        .get(doc_id)
                        .map(|mask| mask.count_ones() >= state.min_tokens)
                        .unwrap_or(false)
                })
                .collect(),
        }
    }

    /// Consume the scorer and return a sorted [`SearchResult`].
    pub fn into_search_result(self) -> SearchResult {
        let scores = self.get_scores();
        let mut result = SearchResult {
            docs: scores
                .into_iter()
                .map(|(doc_id, score)| ScoredDoc { doc_id, score })
                .collect(),
        };
        result.sort_by_score();
        result
    }

    /// Consume the scorer and return the top-`k` results sorted by score descending.
    ///
    /// Uses a min-heap of size `k` to avoid sorting all scored documents.
    /// Complexity is `O(N log k)` instead of `O(N log N)`.
    pub fn into_search_result_top_k(self, k: usize) -> SearchResult {
        let scores = self.get_scores();

        if k == 0 {
            return SearchResult { docs: vec![] };
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k);

        for (doc_id, score) in scores {
            let entry = HeapEntry { doc_id, score };
            if heap.len() < k {
                heap.push(Reverse(entry));
            } else if entry.cmp(&heap.peek().unwrap().0) == std::cmp::Ordering::Greater {
                heap.pop();
                heap.push(Reverse(entry));
            }
        }

        let mut docs: Vec<ScoredDoc> = heap
            .into_iter()
            .map(|Reverse(e)| ScoredDoc {
                doc_id: e.doc_id,
                score: e.score,
            })
            .collect();
        docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.doc_id.cmp(&b.doc_id))
        });

        SearchResult { docs }
    }
}

impl Default for BM25Scorer {
    fn default() -> Self {
        Self::new()
    }
}
