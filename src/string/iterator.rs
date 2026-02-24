use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use super::compacted::CompactedVersion;
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
    /// - `Some(n)`: Levenshtein distance <= n
    pub tolerance: Option<u8>,
    /// Score multiplier per consecutive token pair. None = disabled.
    pub phrase_boost: Option<f32>,
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
        }
    }
}

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
    ) -> Result<()> {
        let collect_positions =
            params.phrase_boost.is_some_and(|b| b > 0.0) && params.tokens.len() >= 2;

        if collect_positions {
            self.execute_with_phrase_boost(params, filter, scorer)
        } else {
            self.execute_simple(params, filter, scorer)
        }
    }

    /// Fast path: no phrase boost needed.
    ///
    /// Algorithm (per token):
    /// 1. Scan compacted + live layers for matching terms (exact, prefix, or fuzzy
    ///    depending on `tolerance`).
    /// 2. For each matching (term, doc) pair, compute the BM25F normalized TF:
    ///    `ntf = tf / (1 - b + b * field_len / avg_field_len)`, weighted by field
    ///    boost and exact-match boost. Multiple term matches for the same doc are
    ///    summed (BM25F multi-field aggregation with a single field).
    /// 3. Compute Lucene-style IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))`.
    /// 4. Final per-token score: `idf * (k+1) * ntf / (k + ntf)` — the standard
    ///    BM25 saturation formula that caps the TF contribution.
    /// 5. Scores are fed into `BM25Scorer` which sums across tokens and optionally
    ///    enforces a minimum-token-match threshold.
    ///
    /// A single `per_doc_ntf` HashMap is reused across tokens to avoid allocations.
    fn execute_simple<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
        scorer: &mut BM25Scorer,
    ) -> Result<()> {
        let total_documents = (self.version.total_documents + self.snapshot.total_documents)
            .saturating_sub(self.snapshot.compacted_deletes_count);
        // NOTE: total_document_length (and thus avg_field_length) is not corrected for
        // live-layer deletes of compacted docs because the live layer does not know their
        // field lengths (stored in the compacted mmap). Looking them up here would cost
        // O(d × log n) per search. The inaccuracy is minor (affects only length normalization,
        // not IDF) and is fully resolved on the next compaction.
        let total_document_length =
            self.version.total_document_length + self.snapshot.total_field_length;

        let avg_field_length = if total_documents > 0 {
            total_document_length as f32 / total_documents as f32
        } else {
            0.0
        };

        let deletes = &self.snapshot.deletes;
        let compacted_deletes = self.version.deletes_slice();

        let exact_match_boost_multiplier = params.bm25_params.exact_match_boost;

        let mut per_doc_ntf: HashMap<u64, f32> = HashMap::new();

        for (token_idx, token) in params.tokens.iter().enumerate() {
            let token_bit = token_bit(token_idx);
            per_doc_ntf.clear();
            let mut corpus_df: u64 = 0;

            // Search compacted layer (zero-copy)
            let version = &self.version;
            version.for_each_term_match(token, params.tolerance, |is_exact, mut reader| {
                let exact_boost = if is_exact {
                    exact_match_boost_multiplier
                } else {
                    1.0
                };

                let mut fl_cursor: usize = 0;
                while let Some(entry) = reader.next_ref() {
                    if is_deleted(entry.doc_id, deletes, compacted_deletes) {
                        continue;
                    }
                    if let Some(filter) = filter {
                        if !filter.contains(entry.doc_id) {
                            continue;
                        }
                    }
                    corpus_df += 1;

                    let tf = if params.exact_match {
                        entry.exact_positions.len()
                    } else {
                        entry.exact_positions.len() + entry.stemmed_positions.len()
                    };
                    if tf == 0 {
                        continue;
                    }

                    let field_length =
                        version.field_length_galloping(entry.doc_id, &mut fl_cursor).unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(entry.doc_id).or_insert(0.0) +=
                        ntf * params.boost * exact_boost;
                }
            })?;

            // Search live layer
            let snapshot = &self.snapshot;
            snapshot.for_each_term_match(token, params.tolerance, |is_exact, postings| {
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

                    let field_length =
                        snapshot.doc_lengths.get(doc_id).copied().unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(*doc_id).or_insert(0.0) += ntf * params.boost * exact_boost;
                }
            });

            // Compute IDF for this token and feed directly into scorer
            let idf = calculate_idf(total_documents, corpus_df);
            for (&doc_id, &aggregated_ntf) in per_doc_ntf.iter() {
                let score = bm25f_score(aggregated_ntf, params.bm25_params.k, idf);
                scorer.add(doc_id, score, token_bit);
            }
        }

        Ok(())
    }

    /// Phrase boost path: same BM25F scoring as `execute_simple`, but additionally
    /// collects token positions to detect consecutive token pairs (phrase matches).
    ///
    /// After all tokens are scored, each document's score is multiplied by
    /// `1 + count * phrase_boost`, where `count` is the number of adjacent token
    /// pairs found. Adjacency is checked via a merge-join on sorted (doc_id, pos)
    /// buffers — O(n+m) with zero extra allocations.
    fn execute_with_phrase_boost<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
        scorer: &mut BM25Scorer,
    ) -> Result<()> {
        let total_documents = (self.version.total_documents + self.snapshot.total_documents)
            .saturating_sub(self.snapshot.compacted_deletes_count);
        // NOTE: total_document_length (and thus avg_field_length) is not corrected for
        // live-layer deletes of compacted docs because the live layer does not know their
        // field lengths (stored in the compacted mmap). Looking them up here would cost
        // O(d × log n) per search. The inaccuracy is minor (affects only length normalization,
        // not IDF) and is fully resolved on the next compaction.
        let total_document_length =
            self.version.total_document_length + self.snapshot.total_field_length;

        let avg_field_length = if total_documents > 0 {
            total_document_length as f32 / total_documents as f32
        } else {
            0.0
        };

        let deletes = &self.snapshot.deletes;
        let compacted_deletes = self.version.deletes_slice();

        let exact_match_boost_multiplier = params.bm25_params.exact_match_boost;

        let mut per_doc_ntf: HashMap<u64, f32> = HashMap::new();
        let mut prev_raw: Vec<(u64, u32)> = Vec::new();
        let mut curr_raw: Vec<(u64, u32)> = Vec::new();
        let mut consecutive_counts: HashMap<u64, usize> = HashMap::new();

        for (token_idx, token) in params.tokens.iter().enumerate() {
            let token_bit = token_bit(token_idx);
            per_doc_ntf.clear();
            curr_raw.clear();
            let mut corpus_df: u64 = 0;

            // Search compacted layer (zero-copy)
            let version = &self.version;
            version.for_each_term_match(token, params.tolerance, |is_exact, mut reader| {
                let exact_boost = if is_exact {
                    exact_match_boost_multiplier
                } else {
                    1.0
                };

                let mut fl_cursor: usize = 0;
                while let Some(entry) = reader.next_ref() {
                    if is_deleted(entry.doc_id, deletes, compacted_deletes) {
                        continue;
                    }
                    if let Some(filter) = filter {
                        if !filter.contains(entry.doc_id) {
                            continue;
                        }
                    }
                    corpus_df += 1;

                    let tf = if params.exact_match {
                        entry.exact_positions.len()
                    } else {
                        entry.exact_positions.len() + entry.stemmed_positions.len()
                    };
                    if tf == 0 {
                        continue;
                    }

                    if params.exact_match {
                        for &p in entry.exact_positions {
                            curr_raw.push((entry.doc_id, p));
                        }
                    } else {
                        for &p in entry.exact_positions {
                            curr_raw.push((entry.doc_id, p));
                        }
                        for &p in entry.stemmed_positions {
                            curr_raw.push((entry.doc_id, p));
                        }
                    }

                    let field_length =
                        version.field_length_galloping(entry.doc_id, &mut fl_cursor).unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(entry.doc_id).or_insert(0.0) +=
                        ntf * params.boost * exact_boost;
                }
            })?;

            // Search live layer
            let snapshot = &self.snapshot;
            snapshot.for_each_term_match(token, params.tolerance, |is_exact, postings| {
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

                    if params.exact_match {
                        for &p in exact.iter() {
                            curr_raw.push((*doc_id, p));
                        }
                    } else {
                        for &p in exact.iter() {
                            curr_raw.push((*doc_id, p));
                        }
                        for &p in stemmed.iter() {
                            curr_raw.push((*doc_id, p));
                        }
                    }

                    let field_length =
                        snapshot.doc_lengths.get(doc_id).copied().unwrap_or(0) as f32;

                    let ntf = bm25f_normalized_tf(
                        tf as f32,
                        field_length,
                        avg_field_length,
                        params.bm25_params.b,
                    );

                    *per_doc_ntf.entry(*doc_id).or_insert(0.0) += ntf * params.boost * exact_boost;
                }
            });

            // Sort and dedup positions, then check adjacency with previous token
            curr_raw.sort_unstable();
            curr_raw.dedup();
            if token_idx > 0 {
                check_adjacency_pairs(&prev_raw, &curr_raw, &mut consecutive_counts);
            }
            prev_raw.clear();
            prev_raw.append(&mut curr_raw);

            // Compute IDF for this token and feed directly into scorer
            let idf = calculate_idf(total_documents, corpus_df);
            for (&doc_id, &aggregated_ntf) in per_doc_ntf.iter() {
                let score = bm25f_score(aggregated_ntf, params.bm25_params.k, idf);
                scorer.add(doc_id, score, token_bit);
            }

            if token_idx == 0 {
                consecutive_counts.reserve(per_doc_ntf.len());
            }
        }

        // Apply phrase boost directly to scorer
        let phrase_multiplier = params.phrase_boost.unwrap_or(0.0);
        for (&doc_id, &count) in consecutive_counts.iter() {
            if count > 0 {
                scorer.multiply_score(doc_id, 1.0 + count as f32 * phrase_multiplier);
            }
        }

        Ok(())
    }
}

/// Merge-join two sorted `(doc_id, position)` buffers and increment
/// `consecutive_counts` for each doc that has any position `p` in `prev`
/// where `(doc_id, p+1)` exists in `curr`. O(n+m), zero allocations.
fn check_adjacency_pairs(
    prev: &[(u64, u32)],
    curr: &[(u64, u32)],
    consecutive_counts: &mut HashMap<u64, usize>,
) {
    let mut pi = 0;
    let mut ci = 0;
    while pi < prev.len() && ci < curr.len() {
        let (pd, pp) = prev[pi];
        let (cd, cp) = curr[ci];
        let target_doc = pd;
        let target_pos = pp + 1;
        match (target_doc, target_pos).cmp(&(cd, cp)) {
            std::cmp::Ordering::Equal => {
                *consecutive_counts.entry(pd).or_insert(0) += 1;
                // Skip remaining pairs for this doc — we only need one adjacent pair per token pair
                let skip_doc = pd;
                while pi < prev.len() && prev[pi].0 == skip_doc {
                    pi += 1;
                }
                while ci < curr.len() && curr[ci].0 == skip_doc {
                    ci += 1;
                }
            }
            std::cmp::Ordering::Less => pi += 1,
            std::cmp::Ordering::Greater => ci += 1,
        }
    }
}

#[inline]
fn token_bit(token_idx: usize) -> u32 {
    if token_idx >= 32 {
        return 0;
    }
    1u32 << (token_idx as u32)
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
        handle
            .execute::<NoFilter>(params, None, &mut scorer)
            .unwrap();
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
        handle
            .execute::<NoFilter>(&params, None, &mut scorer)
            .unwrap();
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
        handle
            .execute::<NoFilter>(&params, None, &mut scorer)
            .unwrap();
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
        handle
            .execute::<NoFilter>(&params, None, &mut scorer)
            .unwrap();
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
            bm25_params: Bm25Params {
                exact_match_boost: 10.0,
                ..Default::default()
            },
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
            bm25_params: Bm25Params {
                exact_match_boost: 100.0,
                ..Default::default()
            },
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
        handle
            .execute::<NoFilter>(&params, None, &mut scorer)
            .unwrap();
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
    fn test_check_adjacency_pairs_basic() {
        let mut counts = HashMap::new();
        // Token 0→1: doc 1 has pos 0→1 (adjacent)
        let prev = vec![(1u64, 0u32)];
        let curr = vec![(1u64, 1u32)];
        check_adjacency_pairs(&prev, &curr, &mut counts);
        assert_eq!(*counts.get(&1).unwrap(), 1);

        // Token 1→2: doc 1 has pos 1→2 (adjacent)
        let prev2 = vec![(1u64, 1u32)];
        let curr2 = vec![(1u64, 2u32)];
        check_adjacency_pairs(&prev2, &curr2, &mut counts);
        assert_eq!(*counts.get(&1).unwrap(), 2);

        // Doc 99 never appeared
        assert!(!counts.contains_key(&99));
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
        handle.execute(&params, Some(&filter), &mut scorer).unwrap();
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
        handle.execute(&params, Some(&filter), &mut scorer).unwrap();
        let result = scorer.into_search_result();
        assert!(result.docs.is_empty());
    }

    // ---- Coverage: phrase boost against compacted layer ----

    #[allow(clippy::type_complexity)]
    fn build_compacted_simple(
        entries: &[(&str, Vec<(u64, Vec<u32>, Vec<u32>)>)],
        doc_lengths: &[(u64, u16)],
        deleted: &[u64],
        path: &std::path::Path,
    ) {
        let empty = CompactedVersion::empty();
        let mut compacted_terms = empty.iter_terms();
        let live: Vec<_> = entries.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let mut compacted_dl = empty.iter_doc_lengths();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &live,
            &mut compacted_dl,
            doc_lengths,
            None,
            None,
            deleted,
            path,
        )
        .unwrap();
    }

    #[test]
    fn test_phrase_boost_compacted_layer() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = super::super::io::ensure_version_dir(base_path, 1).unwrap();

        // Doc 1: "hello" at pos 0, "world" at pos 1 (adjacent)
        // Doc 2: "hello" at pos 0, "world" at pos 5 (not adjacent)
        build_compacted_simple(
            &[
                ("hello", vec![(1, vec![0], vec![]), (2, vec![0], vec![])]),
                ("world", vec![(1, vec![1], vec![]), (2, vec![5], vec![])]),
            ],
            &[(1, 4), (2, 6)],
            &[],
            &version_path,
        );

        let version = Arc::new(CompactedVersion::load(base_path, 1).unwrap());
        let snapshot = Arc::new(LiveSnapshot::empty());
        let handle = SearchHandle::new(version, snapshot);

        let tokens = vec!["hello".to_string(), "world".to_string()];
        let params = SearchParams {
            tokens: &tokens,
            phrase_boost: Some(2.0),
            ..Default::default()
        };

        let mut scorer = BM25Scorer::new();
        handle
            .execute::<NoFilter>(&params, None, &mut scorer)
            .unwrap();
        let mut result = scorer.into_search_result();
        result.sort_by_score();

        assert_eq!(result.docs.len(), 2);
        // Doc 1 should score higher due to phrase boost (adjacent positions)
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_phrase_boost_with_filter() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // All 3 docs: "hello" at pos 0, "world" at pos 1 (adjacent)
        layer.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.insert(
            2,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.insert(
            3,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let filter = SortedDocumentFilter {
            doc_ids: vec![1, 3],
        };
        let tokens = vec!["hello".to_string(), "world".to_string()];
        let params = SearchParams {
            tokens: &tokens,
            phrase_boost: Some(2.0),
            ..Default::default()
        };

        let mut scorer = BM25Scorer::new();
        handle.execute(&params, Some(&filter), &mut scorer).unwrap();
        let result = scorer.into_search_result();

        // Only docs 1 and 3 should be returned; doc 2 is excluded by filter
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&3));
        assert!(!doc_ids.contains(&2));
    }

    #[test]
    fn test_check_adjacency_greater_branch() {
        let mut counts = HashMap::new();
        // prev: doc 2 pos 0, doc 3 pos 0 → targets: (2, 1), (3, 1)
        // curr: doc 1 pos 5, doc 2 pos 1, doc 3 pos 1
        //
        // pi=0 target=(2,1) vs curr[0]=(1,5) → Greater (ci advances)
        // pi=0 target=(2,1) vs curr[1]=(2,1) → Equal (match doc 2, skip doc 2 in both)
        // pi=1 target=(3,1) vs curr[2]=(3,1) → Equal (match doc 3)
        let prev = vec![(2u64, 0u32), (3u64, 0u32)];
        let curr = vec![(1u64, 5u32), (2u64, 1u32), (3u64, 1u32)];
        check_adjacency_pairs(&prev, &curr, &mut counts);

        assert_eq!(*counts.get(&2).unwrap(), 1);
        assert_eq!(*counts.get(&3).unwrap(), 1);
        // Doc 1 never appeared in prev, so no entry
        assert!(!counts.contains_key(&1));
    }

    #[test]
    fn test_token_bit_helper() {
        assert_eq!(token_bit(0), 1);
        assert_eq!(token_bit(1), 2);
        assert_eq!(token_bit(31), 1u32 << 31);
        assert_eq!(token_bit(32), 0);
        assert_eq!(token_bit(100), 0);
    }

    #[test]
    fn test_many_tokens_no_panic() {
        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        // Insert a doc that matches a few terms
        let mut terms = Vec::new();
        for i in 0..40u32 {
            terms.push((format!("token{i}"), vec![i], vec![]));
        }
        let term_refs: Vec<(&str, Vec<u32>, Vec<u32>)> = terms
            .iter()
            .map(|(s, e, st)| (s.as_str(), e.clone(), st.clone()))
            .collect();
        layer.insert(1, make_value(40, term_refs));
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let handle = SearchHandle::new(version, snapshot);

        let token_strings: Vec<String> = (0..40).map(|i| format!("token{i}")).collect();
        let params = SearchParams {
            tokens: &token_strings,
            ..Default::default()
        };

        // Should not panic even with 40 tokens (> 32 bits)
        let result = execute_search(&handle, &params);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > 0.0);
    }
}
