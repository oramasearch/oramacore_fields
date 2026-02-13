use std::collections::HashMap;
use std::sync::Arc;

use super::compacted::{CompactedVersion, PostingEntry};
use super::config::Bm25Params;
use super::live::LiveSnapshot;
use super::scorer::BM25Scorer;
use super::scoring::{bm25f_normalized_tf, bm25f_score, calculate_idf};
use super::DocumentFilter;

/// Parameters for a search query.
#[derive(Debug, Clone)]
pub struct SearchParams<'a> {
    /// Pre-tokenized query terms.
    pub tokens: &'a [String],
    /// If true, only count exact (unstemmed) positions.
    pub exact_match: bool,
    /// Field weight multiplier for BM25F scoring.
    pub boost: f32,
    /// BM25 tuning parameters.
    pub bm25_params: Bm25Params,
    /// Tolerance for fuzzy matching.
    /// - `Some(0)`: exact match (current behavior)
    /// - `None`: prefix search
    /// - `Some(n)`: Levenshtein distance <= n (compacted layer only)
    pub tolerance: Option<u8>,
    /// Score multiplier per consecutive token pair. None = disabled.
    pub phrase_boost: Option<f32>,
    /// Exact-match boost in fuzzy/prefix mode. None = default 3.0.
    pub exact_match_boost: Option<f32>,
}

impl Default for SearchParams<'_> {
    fn default() -> Self {
        Self {
            tokens: &[],
            exact_match: false,
            boost: 1.0,
            bm25_params: Bm25Params::default(),
            tolerance: Some(0),
            phrase_boost: None,
            exact_match_boost: None,
        }
    }
}

/// Score multiplier applied to exact matches when fuzzy/prefix mode is active.
const EXACT_MATCH_BOOST_MULTIPLIER: f32 = 3.0;

/// A document with its BM25 relevance score.
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    pub doc_id: u64,
    pub score: f32,
}

/// The result of a search operation.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub docs: Vec<ScoredDoc>,
}

impl SearchResult {
    /// Sort results by score descending (highest first), then by doc_id ascending for ties.
    pub fn sort_by_score(&mut self) {
        self.docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.doc_id.cmp(&b.doc_id))
        });
    }
}

/// Holds references to compacted + live data and executes searches.
pub struct SearchHandle {
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
}

impl SearchHandle {
    pub fn new(version: Arc<CompactedVersion>, snapshot: Arc<LiveSnapshot>) -> Self {
        Self { version, snapshot }
    }

    pub fn execute<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
        scorer: &mut BM25Scorer,
    ) {
        // Compute global stats by combining compacted + live
        let total_documents = self.version.total_documents + self.snapshot.total_documents;
        let total_document_length =
            self.version.total_document_length + self.snapshot.total_field_length;

        let avg_field_length = if total_documents > 0 {
            total_document_length as f32 / total_documents as f32
        } else {
            0.0
        };

        let deletes = &self.snapshot.deletes;
        let compacted_deletes = self.version.deletes_slice();

        let exact_match_boost_multiplier = params
            .exact_match_boost
            .unwrap_or(EXACT_MATCH_BOOST_MULTIPLIER);

        // Whether to collect positions for phrase matching
        let collect_positions =
            params.phrase_boost.is_some_and(|b| b > 0.0) && params.tokens.len() >= 2;

        // Per-doc accumulator: doc_id -> (score, token_mask)
        let mut field_scores: HashMap<u64, (f32, u32)> = HashMap::new();

        // Per-token per-doc positions (only allocated when phrase boost is active)
        let mut token_doc_positions: Vec<HashMap<u64, Vec<u32>>> = Vec::new();

        for (token_idx, token) in params.tokens.iter().enumerate() {
            let token_bit = 1u32 << (token_idx as u32);
            let mut per_doc_ntf: HashMap<u64, f32> = HashMap::new();
            let mut corpus_df: u64 = 0;
            let mut positions_map: HashMap<u64, Vec<u32>> = HashMap::new();

            // Search compacted layer
            let compacted_matches = self.version.search_terms(token, params.tolerance);
            for (_term, is_exact, reader) in compacted_matches {
                let exact_boost = if is_exact {
                    exact_match_boost_multiplier
                } else {
                    1.0
                };

                for entry in reader {
                    if is_deleted(entry.doc_id, deletes, compacted_deletes) {
                        continue;
                    }
                    if let Some(filter) = filter {
                        if !filter.contains(entry.doc_id) {
                            continue;
                        }
                    }
                    corpus_df += 1;

                    let tf = term_occurrence(&entry, params.exact_match);
                    if tf == 0 {
                        continue;
                    }

                    if collect_positions {
                        let pos = positions_map.entry(entry.doc_id).or_default();
                        if params.exact_match {
                            pos.extend_from_slice(&entry.exact_positions);
                        } else {
                            pos.extend_from_slice(&entry.exact_positions);
                            pos.extend_from_slice(&entry.stemmed_positions);
                        }
                    }

                    let field_length = self.version.field_length(entry.doc_id).unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(entry.doc_id).or_insert(0.0) +=
                        ntf * params.boost * exact_boost;
                }
            }

            // Search live layer
            let live_matches = self.snapshot.search_terms(token, params.tolerance);
            for (_term, is_exact, postings) in live_matches {
                let exact_boost = if is_exact {
                    exact_match_boost_multiplier
                } else {
                    1.0
                };

                for (doc_id, exact, stemmed) in postings {
                    if is_deleted(*doc_id, deletes, compacted_deletes) {
                        continue;
                    }
                    if let Some(filter) = filter {
                        if !filter.contains(*doc_id) {
                            continue;
                        }
                    }
                    corpus_df += 1;

                    let tf = if params.exact_match {
                        exact.len()
                    } else {
                        exact.len() + stemmed.len()
                    };

                    if tf == 0 {
                        continue;
                    }

                    if collect_positions {
                        let pos = positions_map.entry(*doc_id).or_default();
                        if params.exact_match {
                            pos.extend_from_slice(exact);
                        } else {
                            pos.extend_from_slice(exact);
                            pos.extend_from_slice(stemmed);
                        }
                    }

                    let field_length =
                        self.snapshot.doc_lengths.get(doc_id).copied().unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(*doc_id).or_insert(0.0) += ntf * params.boost * exact_boost;
                }
            }

            if collect_positions {
                // Sort and dedup positions per doc
                for positions in positions_map.values_mut() {
                    positions.sort_unstable();
                    positions.dedup();
                }
                token_doc_positions.push(positions_map);
            }

            // Compute IDF for this token and fold into field_scores
            let idf = calculate_idf(total_documents, corpus_df);
            for (doc_id, aggregated_ntf) in per_doc_ntf {
                let score = bm25f_score(aggregated_ntf, params.bm25_params.k, idf);
                let entry = field_scores.entry(doc_id).or_insert((0.0, 0));
                entry.0 += score;
                entry.1 |= token_bit;
            }
        }

        // Apply phrase boost
        if collect_positions {
            let phrase_multiplier = params.phrase_boost.unwrap_or(0.0);
            for (&doc_id, (score, _)) in field_scores.iter_mut() {
                let consecutive_count = count_consecutive_pairs(&token_doc_positions, doc_id);
                if consecutive_count > 0 {
                    *score *= 1.0 + consecutive_count as f32 * phrase_multiplier;
                }
            }
        }

        // Feed results into scorer
        for (doc_id, (score, token_mask)) in field_scores {
            scorer.add(doc_id, score, token_mask);
        }
    }
}

#[inline]
fn term_occurrence(entry: &PostingEntry, exact_match: bool) -> usize {
    if exact_match {
        entry.exact_positions.len()
    } else {
        entry.exact_positions.len() + entry.stemmed_positions.len()
    }
}

/// Count consecutive query token pairs at adjacent positions in a document.
fn count_consecutive_pairs(token_doc_positions: &[HashMap<u64, Vec<u32>>], doc_id: u64) -> usize {
    let mut count = 0;
    for i in 0..token_doc_positions.len().saturating_sub(1) {
        if let (Some(a), Some(b)) = (
            token_doc_positions[i].get(&doc_id),
            token_doc_positions[i + 1].get(&doc_id),
        ) {
            if has_adjacent_positions(a, b) {
                count += 1;
            }
        }
    }
    count
}

/// Check if any position in `a` + 1 exists in `b`. Both must be sorted.
/// O(n+m) two-pointer scan.
fn has_adjacent_positions(a: &[u32], b: &[u32]) -> bool {
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        let target = a[ai] + 1;
        match target.cmp(&b[bi]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => ai += 1,
            std::cmp::Ordering::Greater => bi += 1,
        }
    }
    false
}

#[inline]
fn is_deleted(
    doc_id: u64,
    live_deletes: &std::collections::HashSet<u64>,
    compacted_deletes: &[u64],
) -> bool {
    live_deletes.contains(&doc_id) || compacted_deletes.binary_search(&doc_id).is_ok()
}

#[cfg(test)]
mod tests {
    use super::super::indexer::{IndexedValue, TermData};
    use super::super::live::LiveLayer;
    use super::super::NoFilter;
    use super::*;
    use std::collections::HashMap as StdHashMap;

    fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
        let mut term_map = StdHashMap::new();
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

    fn execute_search(handle: &SearchHandle, params: &SearchParams<'_>) -> SearchResult {
        let mut scorer = BM25Scorer::new();
        handle.execute::<NoFilter>(params, None, &mut scorer);
        scorer.into_search_result()
    }

    #[test]
    fn test_search_empty() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot::empty());
        let handle = SearchHandle::new(version, snapshot);

        let tokens = vec!["hello".to_string()];
        let params = SearchParams {
            tokens: &tokens,
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert!(result.docs.is_empty());
    }

    #[test]
    fn test_search_live_only() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![1, 2])]));
        layer.insert(2, make_value(5, vec![("hello", vec![0, 3], vec![])]));
        layer.insert(3, make_value(2, vec![("world", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
        assert!(result.docs[0].score > 0.0);
        assert!(result.docs[1].score > 0.0);
    }

    #[test]
    fn test_search_exact_match_mode() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![1, 2])]));
        layer.insert(2, make_value(2, vec![("hello", vec![], vec![0, 1])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string()],
            exact_match: true,
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 1);
    }

    #[test]
    fn test_search_with_deletes() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.delete(1);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);
    }

    #[test]
    fn test_search_result_sort() {
        let mut result = SearchResult {
            docs: vec![
                ScoredDoc {
                    doc_id: 1,
                    score: 1.0,
                },
                ScoredDoc {
                    doc_id: 2,
                    score: 3.0,
                },
                ScoredDoc {
                    doc_id: 3,
                    score: 2.0,
                },
            ],
        };

        result.sort_by_score();
        assert_eq!(result.docs[0].doc_id, 2);
        assert_eq!(result.docs[1].doc_id, 3);
        assert_eq!(result.docs[2].doc_id, 1);
    }

    #[test]
    fn test_search_multiple_tokens() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_search_boost() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(Arc::clone(&version), Arc::clone(&snapshot));

        let params_normal = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let params_boosted = SearchParams {
            tokens: &["hello".to_string()],
            boost: 2.0,
            ..Default::default()
        };

        let result_normal = execute_search(&handle, &params_normal);
        let result_boosted = execute_search(&handle, &params_boosted);

        assert!(result_boosted.docs[0].score > result_normal.docs[0].score);
    }

    #[test]
    fn test_search_prefix() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["app".to_string()],
            tolerance: None,
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));
    }

    #[test]
    fn test_search_prefix_exact_boost() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("app", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["app".to_string()],
            tolerance: None,
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_search_tolerance_zero_is_exact() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["apple".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 1);
    }

    // ---- Phrase boost tests ----

    #[test]
    fn test_phrase_boost_adjacent() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Doc 1: "hello" at pos 0, "world" at pos 1 (adjacent)
        layer.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        // Doc 2: "hello" at pos 0, "world" at pos 5 (not adjacent)
        layer.insert(
            2,
            make_value(
                6,
                vec![("hello", vec![0], vec![]), ("world", vec![5], vec![])],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            phrase_boost: Some(2.0),
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
        // Doc 1 should score higher due to phrase boost
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_phrase_boost_non_adjacent() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Tokens are not adjacent
        layer.insert(
            1,
            make_value(
                6,
                vec![("hello", vec![0], vec![]), ("world", vec![5], vec![])],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params_with_boost = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            phrase_boost: Some(2.0),
            ..Default::default()
        };
        let params_without_boost = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            ..Default::default()
        };

        let result_with = execute_search(&handle, &params_with_boost);
        let result_without = execute_search(&handle, &params_without_boost);

        // Non-adjacent: no boost applied, scores should be equal
        assert_eq!(result_with.docs[0].score, result_without.docs[0].score);
    }

    #[test]
    fn test_phrase_boost_three_tokens() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // All 3 tokens at consecutive positions: 2 adjacent pairs
        layer.insert(
            1,
            make_value(
                6,
                vec![
                    ("the", vec![0], vec![]),
                    ("quick", vec![1], vec![]),
                    ("fox", vec![2], vec![]),
                ],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["the".to_string(), "quick".to_string(), "fox".to_string()],
            phrase_boost: Some(1.0),
            ..Default::default()
        };

        let result_boosted = execute_search(&handle, &params);

        let params_no_boost = SearchParams {
            tokens: &["the".to_string(), "quick".to_string(), "fox".to_string()],
            ..Default::default()
        };

        let result_base = execute_search(&handle, &params_no_boost);

        // With 2 consecutive pairs and phrase_boost=1.0: score *= 1 + 2*1.0 = 3.0
        let expected_ratio = 3.0;
        let actual_ratio = result_boosted.docs[0].score / result_base.docs[0].score;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Expected ratio ~{expected_ratio}, got {actual_ratio}"
        );
    }

    #[test]
    fn test_phrase_boost_disabled() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params_none = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            phrase_boost: None,
            ..Default::default()
        };
        let params_default = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            ..Default::default()
        };

        let result_none = execute_search(&handle, &params_none);
        let result_default = execute_search(&handle, &params_default);

        assert_eq!(result_none.docs[0].score, result_default.docs[0].score);
    }

    #[test]
    fn test_phrase_boost_with_exact_match() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // "hello" has exact pos 0, stemmed pos 5
        // "world" has exact pos 1, stemmed pos 3
        // Only exact positions adjacent: 0->1
        layer.insert(
            1,
            make_value(
                6,
                vec![("hello", vec![0], vec![5]), ("world", vec![1], vec![3])],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            exact_match: true,
            phrase_boost: Some(2.0),
            ..Default::default()
        };

        let result_boosted = execute_search(&handle, &params);

        let params_no_boost = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            exact_match: true,
            ..Default::default()
        };

        let result_base = execute_search(&handle, &params_no_boost);

        // Should be boosted (exact positions 0,1 are adjacent)
        assert!(result_boosted.docs[0].score > result_base.docs[0].score);
    }

    // ---- Threshold tests ----

    #[test]
    fn test_threshold_filters_partial_match() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Doc 1 matches all 3 tokens
        layer.insert(
            1,
            make_value(
                6,
                vec![
                    ("hello", vec![0], vec![]),
                    ("world", vec![1], vec![]),
                    ("foo", vec![2], vec![]),
                ],
            ),
        );
        // Doc 2 matches only 1 of 3 tokens
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string(), "foo".to_string()],
            ..Default::default()
        };

        // threshold=1.0 with 3 tokens => floor(3 * 1.0) = 3, need all 3
        let mut scorer = BM25Scorer::with_threshold(3);
        handle.execute::<NoFilter>(&params, None, &mut scorer);
        let result = scorer.into_search_result();
        // Only doc 1 matches all 3 tokens
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 1);
    }

    #[test]
    fn test_threshold_allows_partial_match() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Doc 1 matches all 3 tokens
        layer.insert(
            1,
            make_value(
                6,
                vec![
                    ("hello", vec![0], vec![]),
                    ("world", vec![1], vec![]),
                    ("foo", vec![2], vec![]),
                ],
            ),
        );
        // Doc 2 matches 2 of 3 tokens
        layer.insert(
            2,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        // Doc 3 matches only 1 of 3 tokens
        layer.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        // threshold=0.5 => floor(3 * 0.5) = 1, so docs matching >= 1 token pass
        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string(), "foo".to_string()],
            ..Default::default()
        };

        let mut scorer = BM25Scorer::with_threshold(1);
        handle.execute::<NoFilter>(&params, None, &mut scorer);
        let result = scorer.into_search_result();
        assert_eq!(result.docs.len(), 3);
    }

    #[test]
    fn test_threshold_none_no_filter() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string(), "world".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 2);
    }

    #[test]
    fn test_threshold_single_token() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        // threshold=1.0 with 1 token => need >= 1
        let mut scorer = BM25Scorer::with_threshold(1);
        handle.execute::<NoFilter>(&params, None, &mut scorer);
        let result = scorer.into_search_result();
        assert_eq!(result.docs.len(), 1);
    }

    // ---- Configurable exact-match boost tests ----

    #[test]
    fn test_custom_exact_match_boost() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // "app" exact match and "application" prefix match
        layer.insert(1, make_value(3, vec![("app", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        // Default boost (3.0)
        let params_default = SearchParams {
            tokens: &["app".to_string()],
            tolerance: None,
            ..Default::default()
        };
        let result_default = execute_search(&handle, &params_default);

        // Custom boost (10.0)
        let params_custom = SearchParams {
            tokens: &["app".to_string()],
            tolerance: None,
            exact_match_boost: Some(10.0),
            ..Default::default()
        };
        let result_custom = execute_search(&handle, &params_custom);

        // Both should have doc 1 first
        assert_eq!(result_default.docs[0].doc_id, 1);
        assert_eq!(result_custom.docs[0].doc_id, 1);

        // Higher exact_match_boost should give doc 1 an even bigger advantage
        let ratio_default = result_default.docs[0].score / result_default.docs[1].score;
        let ratio_custom = result_custom.docs[0].score / result_custom.docs[1].score;
        assert!(ratio_custom > ratio_default);
    }

    #[test]
    fn test_exact_match_boost_applied_in_exact_mode() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        // tolerance=Some(0) with default boost (3.0)
        let params_default = SearchParams {
            tokens: &["hello".to_string()],
            tolerance: Some(0),
            ..Default::default()
        };
        // tolerance=Some(0) with custom boost (100.0)
        let params_boosted = SearchParams {
            tokens: &["hello".to_string()],
            tolerance: Some(0),
            exact_match_boost: Some(100.0),
            ..Default::default()
        };

        let result_default = execute_search(&handle, &params_default);
        let result_boosted = execute_search(&handle, &params_boosted);

        // Higher exact_match_boost should give a higher score
        assert!(result_boosted.docs[0].score > result_default.docs[0].score);
    }

    // ---- Combined tests ----

    #[test]
    fn test_phrase_boost_with_threshold() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Doc 1: matches all 3 tokens, consecutive
        layer.insert(
            1,
            make_value(
                6,
                vec![
                    ("the", vec![0], vec![]),
                    ("quick", vec![1], vec![]),
                    ("fox", vec![2], vec![]),
                ],
            ),
        );
        // Doc 2: matches 2 of 3 tokens, consecutive
        layer.insert(
            2,
            make_value(
                4,
                vec![("the", vec![0], vec![]), ("quick", vec![1], vec![])],
            ),
        );
        // Doc 3: matches only 1 token
        layer.insert(3, make_value(2, vec![("the", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["the".to_string(), "quick".to_string(), "fox".to_string()],
            phrase_boost: Some(1.0),
            ..Default::default()
        };

        // threshold=0.7 with 3 tokens => floor(3 * 0.7) = 2, need >= 2 tokens
        let mut scorer = BM25Scorer::with_threshold(2);
        handle.execute::<NoFilter>(&params, None, &mut scorer);
        let result = scorer.into_search_result();
        // Doc 3 filtered out (matches only 1 token, need >= 2)
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));
        // Doc 1 should score higher (more phrase matches + more token matches)
        assert_eq!(result.docs[0].doc_id, 1);
    }

    // ---- Helper function unit tests ----

    #[test]
    fn test_has_adjacent_positions_basic() {
        assert!(has_adjacent_positions(&[0, 2, 5], &[1, 3, 6])); // 0+1=1 found
        assert!(has_adjacent_positions(&[2], &[3])); // 2+1=3 found
        assert!(!has_adjacent_positions(&[0, 2], &[4, 5])); // no match
        assert!(!has_adjacent_positions(&[], &[1])); // empty a
        assert!(!has_adjacent_positions(&[0], &[])); // empty b
    }

    #[test]
    fn test_count_consecutive_pairs_basic() {
        let token_positions = vec![
            HashMap::from([(1u64, vec![0u32])]),
            HashMap::from([(1u64, vec![1u32])]),
            HashMap::from([(1u64, vec![2u32])]),
        ];
        assert_eq!(count_consecutive_pairs(&token_positions, 1), 2);
        assert_eq!(count_consecutive_pairs(&token_positions, 99), 0);
    }

    // ---- DocumentFilter tests ----

    struct SortedDocumentFilter {
        doc_ids: Vec<u64>,
    }
    impl super::DocumentFilter for SortedDocumentFilter {
        fn contains(&self, doc_id: u64) -> bool {
            self.doc_ids.binary_search(&doc_id).is_ok()
        }
    }

    #[test]
    fn test_document_filter_skips_non_matching() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let filter = SortedDocumentFilter {
            doc_ids: vec![1, 3],
        };
        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let mut scorer = BM25Scorer::new();
        handle.execute(&params, Some(&filter), &mut scorer);
        let result = scorer.into_search_result();
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&3));
        assert!(!doc_ids.contains(&2));
    }

    #[test]
    fn test_document_filter_none_returns_all() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 3);
    }

    #[test]
    fn test_document_filter_empty_returns_nothing() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(2, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.insert(3, make_value(3, vec![("hello", vec![0], vec![])]));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let filter = SortedDocumentFilter { doc_ids: vec![] };
        let params = SearchParams {
            tokens: &["hello".to_string()],
            ..Default::default()
        };

        let mut scorer = BM25Scorer::new();
        handle.execute(&params, Some(&filter), &mut scorer);
        let result = scorer.into_search_result();
        assert!(result.docs.is_empty());
    }
}
