//! Tests verifying that carry-forward compaction correctly excludes deleted
//! docs from total_documents in global_info.bin.
//!
//! Previously (Bug 5), carry-forward compaction inflated total_documents
//! because merge_and_write_doc_lengths was called with deleted_set=None.
//! This was fixed by passing count_exclude_set (commit f720e23), so deleted
//! documents are now properly excluded from total_documents and
//! total_document_length regardless of compaction strategy.

use std::collections::HashMap;
use tempfile::TempDir;

use oramacore_fields::string::{
    BM25Scorer, IndexedValue, SearchParams, StringStorage, TermData, Threshold,
};

fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
    let mut term_map = HashMap::new();
    for (term, exact, stemmed) in terms {
        term_map.insert(term.to_string(), TermData::new(exact, stemmed));
    }
    IndexedValue::new(field_length, term_map)
}

fn search_scores(index: &StringStorage, token: &str) -> HashMap<u64, f32> {
    let tokens = vec![token.to_string()];
    let mut scorer = BM25Scorer::new();
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

/// After carry-forward compaction with deletes, total_documents should
/// exclude deleted documents. It doesn't — deleted docs are still counted.
#[test]
fn test_total_documents_inflated_after_carry_forward() {
    let tmp = TempDir::new().unwrap();
    // High threshold → carry-forward mode
    let index = StringStorage::new(
        tmp.path().to_path_buf(),
        Threshold::try_from(0.99f64).unwrap(),
    )
    .unwrap();

    for i in 1..=10u64 {
        index.insert(i, make_value(5, vec![("hello", vec![0], vec![])]));
    }
    index.compact(1).unwrap();
    assert_eq!(index.info().total_documents, 10);

    index.delete(5);
    index.compact(2).unwrap(); // carry-forward (1/10 = 0.1 < 0.99)

    // total_documents should be 9 after deleting 1 of 10 docs
    assert_eq!(
        index.info().total_documents,
        9,
        "total_documents should exclude deleted doc after carry-forward compaction"
    );
}

/// Two identical indexes with different compaction strategies should produce
/// the same total_documents after the same delete.
#[test]
fn test_total_documents_matches_across_compaction_strategies() {
    let tmp_cf = TempDir::new().unwrap();
    let cf_index = StringStorage::new(
        tmp_cf.path().to_path_buf(),
        Threshold::try_from(0.99f64).unwrap(), // carry-forward
    )
    .unwrap();

    let tmp_ad = TempDir::new().unwrap();
    let ad_index = StringStorage::new(
        tmp_ad.path().to_path_buf(),
        Threshold::try_from(0.01f64).unwrap(), // apply-deletes
    )
    .unwrap();

    for i in 1..=10u64 {
        let value = make_value(5, vec![("hello", vec![0], vec![])]);
        cf_index.insert(i, value.clone());
        ad_index.insert(i, value);
    }
    cf_index.compact(1).unwrap();
    ad_index.compact(1).unwrap();

    cf_index.delete(5);
    ad_index.delete(5);

    cf_index.compact(2).unwrap();
    ad_index.compact(2).unwrap();

    assert_eq!(
        cf_index.info().total_documents,
        ad_index.info().total_documents,
        "carry-forward total_documents={} != apply-deletes total_documents={}",
        cf_index.info().total_documents,
        ad_index.info().total_documents,
    );
}

/// BM25 scores should be identical regardless of compaction strategy.
/// Inflated total_documents changes both IDF and avg_field_length, producing
/// different scores for the same surviving documents.
#[test]
fn test_bm25_scores_identical_across_compaction_strategies() {
    let tmp_cf = TempDir::new().unwrap();
    let cf_index = StringStorage::new(
        tmp_cf.path().to_path_buf(),
        Threshold::try_from(0.99f64).unwrap(),
    )
    .unwrap();

    let tmp_ad = TempDir::new().unwrap();
    let ad_index = StringStorage::new(
        tmp_ad.path().to_path_buf(),
        Threshold::try_from(0.01f64).unwrap(),
    )
    .unwrap();

    // Varying field lengths to make avg_field_length sensitive to deletion
    for i in 1..=10u64 {
        let field_len = (i * 2) as u16;
        let value = make_value(field_len, vec![("hello", vec![0], vec![])]);
        cf_index.insert(i, value.clone());
        ad_index.insert(i, value);
    }

    cf_index.compact(1).unwrap();
    ad_index.compact(1).unwrap();

    // Delete doc with longest field (doc 10, field_length=20)
    cf_index.delete(10);
    ad_index.delete(10);

    cf_index.compact(2).unwrap();
    ad_index.compact(2).unwrap();

    let cf_scores = search_scores(&cf_index, "hello");
    let ad_scores = search_scores(&ad_index, "hello");

    assert_eq!(cf_scores.len(), ad_scores.len());

    let cf_score = cf_scores[&1];
    let ad_score = ad_scores[&1];

    assert!(
        (cf_score - ad_score).abs() < 1e-6,
        "doc 1 score differs: carry-forward={cf_score:.10} vs apply-deletes={ad_score:.10}"
    );
}

fn assert_scores_map_equal(
    expected: &HashMap<u64, f32>,
    actual: &HashMap<u64, f32>,
    context: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected {} docs but got {}",
        expected.len(),
        actual.len()
    );
    for (&doc_id, &expected_score) in expected {
        let actual_score = actual.get(&doc_id).unwrap_or_else(|| {
            panic!("{context}: doc_id {doc_id} missing from results");
        });
        assert!(
            (expected_score - actual_score).abs() < 1e-7,
            "{context}: doc_id {doc_id} score differs: expected={expected_score:.10} actual={actual_score:.10}"
        );
    }
}

/// BM25 scores for the original documents must remain identical through
/// insert/delete/compact cycles. Inserting and then deleting documents
/// restores N, avg_field_length, corpus_df, and per-doc TF/field_length
/// to their original values, so scores must not drift.
#[test]
fn test_scores_stable_across_insert_delete_compact_cycles() {
    let tmp = TempDir::new().unwrap();
    let index = StringStorage::new(
        tmp.path().to_path_buf(),
        Threshold::try_from(0.5f64).unwrap(),
    )
    .unwrap();

    // Step 1: Insert 10 documents with varying field lengths, terms, and positions
    index.insert(
        1,
        make_value(
            2,
            vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.insert(
        2,
        make_value(
            5,
            vec![("hello", vec![0, 2, 4], vec![]), ("world", vec![1], vec![])],
        ),
    );
    index.insert(
        3,
        make_value(
            10,
            vec![("hello", vec![0], vec![]), ("foo", vec![5], vec![])],
        ),
    );
    index.insert(4, make_value(3, vec![("world", vec![0, 1, 2], vec![])]));
    index.insert(
        5,
        make_value(
            4,
            vec![("hello", vec![0], vec![]), ("bar", vec![2, 3], vec![])],
        ),
    );
    index.insert(
        6,
        make_value(
            6,
            vec![
                ("hello", vec![0, 3], vec![1]),
                ("world", vec![2, 5], vec![]),
            ],
        ),
    );
    index.insert(7, make_value(3, vec![("hello", vec![0], vec![2])]));
    index.insert(
        8,
        make_value(
            8,
            vec![
                ("hello", vec![0, 1, 2, 3], vec![]),
                ("world", vec![4, 5, 6, 7], vec![]),
            ],
        ),
    );
    index.insert(9, make_value(1, vec![("hello", vec![0], vec![])]));
    index.insert(
        10,
        make_value(
            7,
            vec![
                ("hello", vec![0, 6], vec![]),
                ("baz", vec![1, 2, 3, 4, 5], vec![]),
            ],
        ),
    );

    // Step 2: Search — baseline scores
    let s1 = search_scores(&index, "hello");
    assert_eq!(s1.len(), 9, "9 of 10 docs contain 'hello'");

    // Step 3: Insert 2 more documents (IDs 11-12)
    index.insert(
        11,
        make_value(
            4,
            vec![("hello", vec![0, 1], vec![]), ("world", vec![2, 3], vec![])],
        ),
    );
    index.insert(
        12,
        make_value(
            3,
            vec![("hello", vec![0], vec![]), ("foo", vec![1, 2], vec![])],
        ),
    );

    // Step 4: Delete those 2 documents
    index.delete(11);
    index.delete(12);

    // Step 5: Search — must match baseline
    let s2 = search_scores(&index, "hello");
    assert_scores_map_equal(&s1, &s2, "after insert+delete of 2 docs");

    // Step 6: Compact
    index.compact(1).unwrap();

    // Step 7: Search — must match baseline
    let s3 = search_scores(&index, "hello");
    assert_scores_map_equal(&s1, &s3, "after first compaction");

    // Step 8: Insert 10 more documents (IDs 13-22)
    for i in 13..=22u64 {
        index.insert(i, make_value(5, vec![("hello", vec![0, 2], vec![1, 3, 4])]));
    }

    // Step 9: Delete those 10 documents
    for i in 13..=22u64 {
        index.delete(i);
    }

    // Step 10: Search — must match baseline
    let s4 = search_scores(&index, "hello");
    assert_scores_map_equal(&s1, &s4, "after insert+delete of 10 docs");

    // Step 11: Compact again
    index.compact(2).unwrap();

    // Step 12: Search — must match baseline
    let s5 = search_scores(&index, "hello");
    assert_scores_map_equal(&s1, &s5, "after second compaction");

    // Step 13: Delete a compacted document via the live layer
    index.delete(3);

    // Step 14: Search — scores for remaining docs must be consistent
    let s6 = search_scores(&index, "hello");
    assert_eq!(s6.len(), 8, "8 of remaining 9 docs contain 'hello'");
    assert!(!s6.contains_key(&3), "deleted doc 3 should not appear");

    // Step 15: Compact (materializes the compacted-doc delete)
    index.compact(3).unwrap();

    // Step 16: Search — scores should closely match pre-compaction scores.
    // avg_field_length is not corrected for live-layer deletes of compacted docs
    // (their field lengths are stored in the compacted mmap and not accessible to the
    // live layer). This causes a small score difference that is fully resolved on
    // compaction. Use a wider tolerance to account for this known gap.
    let s7 = search_scores(&index, "hello");
    assert_eq!(
        s6.len(),
        s7.len(),
        "same number of docs before and after compaction"
    );
    for (&doc_id, &score_before) in &s6 {
        let score_after = s7[&doc_id];
        assert!(
            (score_before - score_after).abs() < 0.02,
            "after compacting a compacted-doc delete: doc_id {doc_id} score differs too much: \
             before={score_before:.10} after={score_after:.10}"
        );
    }
}
