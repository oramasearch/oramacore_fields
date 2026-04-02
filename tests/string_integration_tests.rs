//! Integration tests for the string (full-text search) module.
//!
//! Covers basic CRUD, compaction lifecycle, persistence, search modes,
//! phrase boost, scorer threshold, document filter, edge cases, info,
//! integrity checks, cleanup, and delete-then-reinsert scenarios.

use std::collections::{HashMap, HashSet};

use tempfile::TempDir;

use oramacore_fields::string::{
    BM25u64Scorer, CheckStatus, DocumentFilter, IndexedValue, SearchParams, SearchResult,
    SegmentConfig, StringStorage, TermData, Threshold,
};

// ============================================================================
// Helpers
// ============================================================================

fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
    let mut term_map = HashMap::new();
    for (term, exact, stemmed) in terms {
        term_map.insert(term.to_string(), TermData::new(exact, stemmed));
    }
    IndexedValue::new(field_length, term_map)
}

/// Search with default params (exact match via tolerance=Some(0)) and return doc_ids sorted.
fn search_doc_ids(index: &StringStorage, token: &str) -> Vec<u64> {
    let tokens = vec![token.to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    ids
}

/// Search and return score map.
fn search_scores(index: &StringStorage, token: &str) -> HashMap<u64, f32> {
    let tokens = vec![token.to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    scorer.get_scores()
}

/// Search with custom params and return SearchResult.
fn search_with_params(index: &StringStorage, params: &SearchParams<'_>) -> SearchResult {
    let mut scorer = BM25u64Scorer::new();
    index.search(params, &mut scorer).unwrap();
    scorer.into_search_result()
}

/// Search with custom params and a scorer with threshold.
fn search_with_threshold(
    index: &StringStorage,
    params: &SearchParams<'_>,
    min_tokens: u32,
) -> SearchResult {
    let mut scorer = BM25u64Scorer::with_threshold(min_tokens);
    index.search(params, &mut scorer).unwrap();
    scorer.into_search_result()
}

/// Search with a document filter.
fn search_with_filter<F: DocumentFilter>(
    index: &StringStorage,
    token: &str,
    filter: &F,
) -> Vec<u64> {
    let tokens = vec![token.to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search_with_filter(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            filter,
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    ids
}

// ============================================================================
// A. Basic CRUD
// ============================================================================

#[test]
fn test_basic_insert_and_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(3, vec![("hello", vec![0], vec![1, 2])]));
    index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
    index.insert(
        3,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![1, 3], "hello should match docs 1 and 3");

    let ids = search_doc_ids(&index, "world");
    assert_eq!(ids, vec![2, 3], "world should match docs 2 and 3");
}

#[test]
fn test_search_returns_no_results_for_missing_term() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));

    let ids = search_doc_ids(&index, "nonexistent");
    assert!(
        ids.is_empty(),
        "search for missing term should return empty"
    );
}

#[test]
fn test_delete_removes_doc_from_results() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
    index.delete(2);

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![1, 3], "deleted doc 2 should not appear");
}

#[test]
fn test_delete_nonexistent_is_noop() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.delete(999); // does not exist

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![1], "existing doc should be unaffected");
}

#[test]
fn test_duplicate_inserts() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(1, make_value(3, vec![("hello", vec![0, 1], vec![])]));

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![1], "duplicate insert should appear once");
}

#[test]
fn test_doc_id_zero() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(0, make_value(2, vec![("hello", vec![0], vec![])]));
    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![0], "doc_id 0 should be searchable");

    index.compact(1).unwrap();
    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![0], "doc_id 0 should survive compaction");

    index.delete(0);
    let ids = search_doc_ids(&index, "hello");
    assert!(ids.is_empty(), "doc_id 0 should be deletable");

    index.compact(2).unwrap();
    let ids = search_doc_ids(&index, "hello");
    assert!(
        ids.is_empty(),
        "deletion of doc_id 0 should survive compaction"
    );
}

// ============================================================================
// B. Compaction
// ============================================================================

#[test]
fn test_compaction_preserves_data() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
    index.insert(
        3,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );

    let before_hello = search_doc_ids(&index, "hello");
    let before_world = search_doc_ids(&index, "world");

    index.compact(1).unwrap();

    let after_hello = search_doc_ids(&index, "hello");
    let after_world = search_doc_ids(&index, "world");

    assert_eq!(
        before_hello, after_hello,
        "compaction should preserve hello results"
    );
    assert_eq!(
        before_world, after_world,
        "compaction should preserve world results"
    );
}

#[test]
fn test_compaction_applies_deletes() {
    let tmp = TempDir::new().unwrap();
    // Low threshold → apply-deletes mode
    let index = StringStorage::new(
        tmp.path().to_path_buf(),
        SegmentConfig {
            deletion_threshold: Threshold::try_from(0.01f64).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();

    for i in 1..=10u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    index.delete(5);
    index.compact(2).unwrap();

    let ids = search_doc_ids(&index, "term");
    assert!(
        !ids.contains(&5),
        "deleted doc should be gone after apply-deletes compaction"
    );
    assert_eq!(ids.len(), 9);

    // With apply-deletes, deleted.bin should be empty
    let info = index.info();
    assert_eq!(
        info.deleted_count, 0,
        "apply-deletes should clear deleted.bin"
    );
}

#[test]
fn test_compaction_carries_forward_deletes() {
    let tmp = TempDir::new().unwrap();
    // High threshold → carry-forward mode
    let index = StringStorage::new(
        tmp.path().to_path_buf(),
        SegmentConfig {
            deletion_threshold: Threshold::try_from(0.99f64).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();

    for i in 1..=10u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    index.delete(5);
    index.compact(2).unwrap();

    let ids = search_doc_ids(&index, "term");
    assert!(!ids.contains(&5), "deleted doc should not appear in search");
    assert_eq!(ids.len(), 9);

    // With carry-forward, deleted.bin should have the tombstone
    let info = index.info();
    assert!(
        info.deleted_count > 0,
        "carry-forward should have tombstones in deleted.bin"
    );
}

#[test]
fn test_multiple_compactions() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("alpha", vec![0], vec![])]));
    index.compact(1).unwrap();
    assert_eq!(index.current_version_number(), 1);

    index.insert(2, make_value(2, vec![("beta", vec![0], vec![])]));
    index.compact(2).unwrap();
    assert_eq!(index.current_version_number(), 2);

    index.insert(3, make_value(2, vec![("gamma", vec![0], vec![])]));
    index.compact(3).unwrap();
    assert_eq!(index.current_version_number(), 3);

    assert_eq!(search_doc_ids(&index, "alpha"), vec![1]);
    assert_eq!(search_doc_ids(&index, "beta"), vec![2]);
    assert_eq!(search_doc_ids(&index, "gamma"), vec![3]);
}

#[test]
fn test_empty_compaction_skipped() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();
    assert_eq!(index.current_version_number(), 1);

    // No changes since last compaction
    index.compact(2).unwrap();
    // Version should remain 1 because empty compaction is skipped
    assert_eq!(
        index.current_version_number(),
        1,
        "empty compaction should not increment version"
    );
}

// ============================================================================
// C. Persistence
// ============================================================================

#[test]
fn test_persistence_across_reopen() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    {
        let index = StringStorage::new(path.clone(), SegmentConfig::default()).unwrap();
        index.insert(1, make_value(3, vec![("hello", vec![0], vec![1])]));
        index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
        index.compact(1).unwrap();
    }

    {
        let index = StringStorage::new(path, SegmentConfig::default()).unwrap();
        assert_eq!(index.current_version_number(), 1);

        let ids = search_doc_ids(&index, "hello");
        assert_eq!(ids, vec![1]);

        let ids = search_doc_ids(&index, "world");
        assert_eq!(ids, vec![2]);
    }
}

#[test]
fn test_persistence_multiple_compactions() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    {
        let index = StringStorage::new(path.clone(), SegmentConfig::default()).unwrap();
        index.insert(1, make_value(2, vec![("alpha", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.insert(2, make_value(2, vec![("beta", vec![0], vec![])]));
        index.compact(2).unwrap();
    }

    {
        let index = StringStorage::new(path, SegmentConfig::default()).unwrap();
        assert_eq!(index.current_version_number(), 2);
        assert_eq!(search_doc_ids(&index, "alpha"), vec![1]);
        assert_eq!(search_doc_ids(&index, "beta"), vec![2]);
    }
}

// ============================================================================
// D. Search Modes
// ============================================================================

#[test]
fn test_exact_match_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("apple", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("application", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["apple".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            tolerance: Some(0), // exact match only
            ..Default::default()
        },
    );

    let ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    assert_eq!(ids, vec![1], "tolerance=0 should find only exact 'apple'");
}

#[test]
fn test_prefix_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("apple", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("application", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("banana", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["app".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            tolerance: None, // prefix search
            ..Default::default()
        },
    );

    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![1, 2],
        "prefix 'app' should match apple and application"
    );
}

#[test]
fn test_levenshtein_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("apple", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("apply", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("banana", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["apple".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            tolerance: Some(1), // edit distance 1
            ..Default::default()
        },
    );

    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![1, 2],
        "levenshtein(1) from 'apple' should find 'apple' and 'apply'"
    );
}

#[test]
fn test_exact_match_flag() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1: has both exact and stemmed positions
    index.insert(1, make_value(3, vec![("run", vec![0], vec![1, 2])]));
    // Doc 2: has only stemmed positions (no exact)
    index.insert(2, make_value(3, vec![("run", vec![], vec![0, 1])]));
    index.compact(1).unwrap();

    // exact_match=true should only use exact_positions
    let tokens = vec!["run".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            exact_match: true,
            tolerance: None,
            ..Default::default()
        },
    );

    let ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    assert_eq!(
        ids,
        vec![1],
        "exact_match=true should only find doc with exact positions"
    );
}

#[test]
fn test_stemmed_match() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Only stemmed positions
    index.insert(1, make_value(3, vec![("running", vec![], vec![0, 1])]));
    index.compact(1).unwrap();

    // exact_match=false should find docs via stemmed positions
    let tokens = vec!["running".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            exact_match: false,
            ..Default::default()
        },
    );

    assert_eq!(result.docs.len(), 1);
    assert_eq!(result.docs[0].doc_id, 1);

    // exact_match=true should NOT find it (no exact positions)
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            exact_match: true,
            ..Default::default()
        },
    );

    assert!(
        result.docs.is_empty(),
        "exact_match=true with stemmed-only should return nothing"
    );
}

#[test]
fn test_multi_token_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1 matches both tokens
    index.insert(
        1,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    // Doc 2 matches only "hello"
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    // Doc 3 matches only "world"
    index.insert(3, make_value(2, vec![("world", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            ..Default::default()
        },
    );

    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "multi-token search returns docs matching any token"
    );

    // Doc matching both should score highest
    assert_eq!(
        result.docs[0].doc_id, 1,
        "doc matching both tokens should rank first"
    );
}

// ============================================================================
// E. Phrase Boost
// ============================================================================

#[test]
fn test_phrase_boost_increases_score() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1: adjacent tokens (positions 0 and 1)
    index.insert(
        1,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    // Doc 2: non-adjacent tokens (positions 0 and 5)
    index.insert(
        2,
        make_value(
            6,
            vec![("hello", vec![0], vec![]), ("world", vec![5], vec![])],
        ),
    );
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            phrase_boost: Some(2.0),
            ..Default::default()
        },
    );

    assert_eq!(result.docs.len(), 2);
    assert_eq!(
        result.docs[0].doc_id, 1,
        "adjacent tokens should score higher with phrase boost"
    );
    assert!(result.docs[0].score > result.docs[1].score);
}

#[test]
fn test_phrase_boost_disabled_by_default() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Both docs have same terms, same field lengths, only differ in positions
    index.insert(
        1,
        make_value(
            3,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.insert(
        2,
        make_value(
            3,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            phrase_boost: None, // disabled
            ..Default::default()
        },
    );

    assert_eq!(result.docs.len(), 2);
    // With identical data, scores should be equal
    let score_diff = (result.docs[0].score - result.docs[1].score).abs();
    assert!(
        score_diff < 1e-6,
        "identical docs without phrase boost should have equal scores, diff={score_diff}"
    );
}

// ============================================================================
// F. BM25 Scorer Threshold
// ============================================================================

#[test]
fn test_scorer_threshold_filters_partial_matches() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1 matches both tokens
    index.insert(
        1,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    // Doc 2 matches only "hello"
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let result = search_with_threshold(
        &index,
        &SearchParams {
            tokens: &tokens,
            ..Default::default()
        },
        2, // require both tokens
    );

    assert_eq!(result.docs.len(), 1);
    assert_eq!(
        result.docs[0].doc_id, 1,
        "only doc matching both tokens survives threshold"
    );
}

#[test]
fn test_scorer_no_threshold_includes_all() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1 matches both
    index.insert(
        1,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    // Doc 2 matches only "hello"
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let result = search_with_params(
        &index,
        &SearchParams {
            tokens: &tokens,
            ..Default::default()
        },
    );

    assert_eq!(
        result.docs.len(),
        2,
        "no threshold should return all matching docs"
    );
}

// ============================================================================
// F2. Scorer top-K
// ============================================================================

#[test]
fn test_scorer_top_k_returns_highest_scores() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Insert docs with different field lengths so BM25 produces distinct scores.
    // Shorter field length → higher BM25 score for the same term.
    index.insert(1, make_value(10, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(3, make_value(5, vec![("hello", vec![0], vec![])]));
    index.insert(4, make_value(20, vec![("hello", vec![0], vec![])]));
    index.insert(5, make_value(1, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    // Get the full result for reference
    let tokens = vec!["hello".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let full = scorer.into_search_result();
    assert_eq!(full.docs.len(), 5);

    // Now get top-2
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let top2 = scorer.into_search_result_top_k(2);
    assert_eq!(top2.docs.len(), 2);

    // The top-2 from top_k must match the first 2 from full sort
    assert_eq!(top2.docs[0].doc_id, full.docs[0].doc_id);
    assert_eq!(top2.docs[1].doc_id, full.docs[1].doc_id);
    assert_eq!(top2.docs[0].score, full.docs[0].score);
    assert_eq!(top2.docs[1].score, full.docs[1].score);
}

#[test]
fn test_scorer_top_k_larger_than_total() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(5, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result_top_k(100);

    assert_eq!(result.docs.len(), 2, "k > total should return all docs");
}

#[test]
fn test_scorer_top_k_zero() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result_top_k(0);

    assert!(result.docs.is_empty(), "k=0 should return empty result");
}

#[test]
fn test_scorer_top_k_with_threshold() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Doc 1 matches both tokens
    index.insert(
        1,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    // Doc 2 matches only "hello"
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    // Doc 3 matches both tokens
    index.insert(
        3,
        make_value(
            8,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];
    let mut scorer = BM25u64Scorer::with_threshold(2);
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    // Only docs matching both tokens survive threshold (docs 1 and 3).
    // top_k(1) should return the highest scoring one.
    let result = scorer.into_search_result_top_k(1);

    assert_eq!(result.docs.len(), 1);

    // Verify it's the same as the full result's top-1
    let mut scorer = BM25u64Scorer::with_threshold(2);
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let full = scorer.into_search_result();
    assert_eq!(result.docs[0].doc_id, full.docs[0].doc_id);
}

#[test]
fn test_scorer_top_k_ordering_matches_full_sort() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Insert many docs with varied field lengths for score diversity
    for i in 1..=50u64 {
        index.insert(
            i,
            make_value((i as u16) * 2 + 1, vec![("search", vec![0], vec![])]),
        );
    }
    index.compact(1).unwrap();

    let tokens = vec!["search".to_string()];

    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let full = scorer.into_search_result();

    for k in [1, 5, 10, 25, 50] {
        let mut scorer = BM25u64Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let top = scorer.into_search_result_top_k(k);

        assert_eq!(top.docs.len(), k);
        for (i, doc) in top.docs.iter().enumerate() {
            assert_eq!(
                doc.doc_id, full.docs[i].doc_id,
                "top_k({k}) doc at position {i} should match full sort"
            );
            assert_eq!(
                doc.score, full.docs[i].score,
                "top_k({k}) score at position {i} should match full sort"
            );
        }
    }
}

// ============================================================================
// G. Document Filter
// ============================================================================

struct AllowList(HashSet<u64>);

impl DocumentFilter for AllowList {
    fn contains(&self, doc_id: u64) -> bool {
        self.0.contains(&doc_id)
    }
}

#[test]
fn test_document_filter_excludes_docs() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let allowed = AllowList(HashSet::from([1, 3]));
    let ids = search_with_filter(&index, "hello", &allowed);
    assert_eq!(ids, vec![1, 3], "filter should exclude doc 2");
}

#[test]
fn test_no_filter_includes_all() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids, vec![1, 2, 3], "NoFilter should include all docs");
}

// ============================================================================
// H. Edge Cases
// ============================================================================

#[test]
fn test_large_doc_ids() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    let large_id = u64::MAX - 1;
    index.insert(large_id, make_value(2, vec![("term", vec![0], vec![])]));
    index.insert(1, make_value(2, vec![("term", vec![0], vec![])]));

    let ids = search_doc_ids(&index, "term");
    assert!(ids.contains(&large_id));
    assert!(ids.contains(&1));

    index.compact(1).unwrap();

    let ids = search_doc_ids(&index, "term");
    assert!(
        ids.contains(&large_id),
        "large doc_id should survive compaction"
    );
    assert!(ids.contains(&1));
}

#[test]
fn test_unicode_terms() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("日本語", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("café", vec![0], vec![])]));
    index.insert(3, make_value(2, vec![("über", vec![0], vec![])]));
    index.compact(1).unwrap();

    assert_eq!(search_doc_ids(&index, "日本語"), vec![1]);
    assert_eq!(search_doc_ids(&index, "café"), vec![2]);
    assert_eq!(search_doc_ids(&index, "über"), vec![3]);
}

#[test]
fn test_many_terms_per_doc() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    let terms: Vec<(&str, Vec<u32>, Vec<u32>)> = (0..50)
        .map(|i| {
            // Leak strings to get &str with 'static lifetime for the test
            let s: &str = Box::leak(format!("term{i}").into_boxed_str());
            (s, vec![i as u32], vec![])
        })
        .collect();

    index.insert(1, make_value(50, terms));
    index.compact(1).unwrap();

    for i in 0..50 {
        let ids = search_doc_ids(&index, &format!("term{i}"));
        assert_eq!(ids, vec![1], "term{i} should be searchable");
    }
}

#[test]
fn test_many_docs_same_term() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    for i in 1..=1000u64 {
        index.insert(i, make_value(2, vec![("hello", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    let ids = search_doc_ids(&index, "hello");
    assert_eq!(ids.len(), 1000, "all 1000 docs should be returned");
}

#[test]
fn test_empty_index_search() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    let ids = search_doc_ids(&index, "anything");
    assert!(ids.is_empty(), "empty index should return no results");
}

// ============================================================================
// I. Info
// ============================================================================

#[test]
fn test_info_empty_index() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    let info = index.info();
    assert_eq!(info.current_version_number, 0);
    assert_eq!(info.unique_terms_count, 0);
    assert_eq!(info.total_postings_count, 0);
    assert_eq!(info.total_documents, 0);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.pending_ops, 0);
}

#[test]
fn test_info_with_pending_ops() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));

    let info = index.info();
    assert!(
        info.pending_ops > 0,
        "should have pending ops after inserts"
    );
}

#[test]
fn test_info_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(
        1,
        make_value(
            5,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.insert(2, make_value(3, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let info = index.info();
    assert_eq!(info.current_version_number, 1);
    assert_eq!(info.unique_terms_count, 2); // "hello" and "world"
    assert_eq!(info.total_documents, 2);
    assert_eq!(info.pending_ops, 0);
    assert!(info.avg_field_length > 0.0);
    assert!(info.num_segments > 0);
    assert!(info.total_segments_size_bytes > 0);
}

#[test]
fn test_info_with_deletes() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.delete(1);

    let info = index.info();
    assert!(info.pending_ops > 0, "delete should create pending op");

    // Compact with low threshold to apply deletes
    let index2 = StringStorage::new(
        tmp.path().to_path_buf(),
        SegmentConfig {
            deletion_threshold: Threshold::try_from(0.01f64).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    index2.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index2.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index2.delete(1);
    index2.compact(1).unwrap();

    let info = index2.info();
    assert_eq!(
        info.deleted_count, 0,
        "apply-deletes should clear deleted_count"
    );
    assert_eq!(info.total_documents, 1);
}

// ============================================================================
// J. Integrity Check
// ============================================================================

#[test]
fn test_integrity_check_before_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    let result = index.integrity_check();
    assert!(
        !result.passed,
        "integrity check should fail without CURRENT file"
    );
}

#[test]
fn test_integrity_check_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "integrity check should pass after compaction: {:?}",
        result.checks
    );
}

#[test]
fn test_integrity_check_corrupted_current_file() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    // Corrupt the CURRENT file
    std::fs::write(tmp.path().join("CURRENT"), "garbage").unwrap();

    let result = index.integrity_check();
    assert!(
        !result.passed,
        "integrity check should fail with corrupted CURRENT"
    );
}

#[test]
fn test_integrity_check_missing_required_file() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();

    // Remove keys.fst from the segment directory
    std::fs::remove_file(tmp.path().join("segments/seg_0/keys.fst")).unwrap();

    let result = index.integrity_check();
    assert!(
        !result.passed,
        "integrity check should fail with missing keys.fst"
    );

    // Verify the failure mentions index files
    let failed_check = result
        .checks
        .iter()
        .find(|c| matches!(c.status, CheckStatus::Failed));
    assert!(failed_check.is_some(), "should have a failed check");
}

#[test]
fn test_integrity_check_with_deletes_compacted() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
    index.delete(1);
    index.compact(1).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "integrity check should pass with deletes compacted: {:?}",
        result.checks
    );
}

// ============================================================================
// K. Cleanup
// ============================================================================

#[test]
fn test_cleanup_removes_old_versions() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
    index.compact(1).unwrap();
    assert!(tmp.path().join("versions/1").exists());

    index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
    index.compact(2).unwrap();
    assert!(tmp.path().join("versions/2").exists());

    index.insert(3, make_value(2, vec![("foo", vec![0], vec![])]));
    index.compact(3).unwrap();
    assert!(tmp.path().join("versions/3").exists());

    index.cleanup();

    assert!(
        !tmp.path().join("versions/1").exists(),
        "old version 1 should be removed"
    );
    assert!(
        !tmp.path().join("versions/2").exists(),
        "old version 2 should be removed"
    );
    assert!(
        tmp.path().join("versions/3").exists(),
        "current version 3 should remain"
    );

    // Data should still be accessible
    assert_eq!(search_doc_ids(&index, "hello"), vec![1]);
    assert_eq!(search_doc_ids(&index, "world"), vec![2]);
    assert_eq!(search_doc_ids(&index, "foo"), vec![3]);
}

// ============================================================================
// L. Delete-then-Reinsert
// ============================================================================

#[test]
fn test_delete_reinsert_live_only() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    index.insert(1, make_value(2, vec![("oldterm", vec![0], vec![])]));
    index.delete(1);
    index.insert(1, make_value(2, vec![("newterm", vec![0], vec![])]));

    let ids = search_doc_ids(&index, "newterm");
    assert_eq!(
        ids,
        vec![1],
        "re-inserted doc should be found under new term"
    );

    let ids = search_doc_ids(&index, "oldterm");
    assert!(
        ids.is_empty(),
        "old term should not match after delete+reinsert"
    );
}

#[test]
fn test_delete_reinsert_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Insert doc with "oldterm" and compact to disk
    index.insert(1, make_value(2, vec![("oldterm", vec![0], vec![])]));
    index.compact(1).unwrap();

    // Delete and re-insert with same term (updated positions)
    index.delete(1);
    index.insert(1, make_value(3, vec![("oldterm", vec![0, 1], vec![])]));
    index.compact(2).unwrap();

    let ids = search_doc_ids(&index, "oldterm");
    assert_eq!(ids, vec![1], "re-inserted doc should survive compaction");

    // Verify scores exist
    let scores = search_scores(&index, "oldterm");
    assert!(scores.contains_key(&1));
    assert!(scores[&1] > 0.0);
}
