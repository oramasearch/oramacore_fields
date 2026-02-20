//! Tests demonstrating Bug 5: carry-forward compaction writes inflated
//! total_documents to global_info.bin (includes deleted docs), causing
//! incorrect BM25 scores.
//!
//! When carry-forward compaction is used (delete ratio below threshold),
//! merge_and_write_doc_lengths is called with deleted_set=None, so deleted
//! documents are included in total_documents and total_document_length in
//! global_info.bin. This inflates BM25 parameters (avg_field_length and IDF),
//! producing incorrect scores.

use std::collections::HashMap;
use tempfile::TempDir;

use oramacore_fields::string::{
    BM25Scorer, IndexedValue, NoFilter, SearchParams, StringStorage, TermData, Threshold,
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
        .search::<NoFilter>(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            None,
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
