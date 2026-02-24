//! Score regression tests for the embedding module.
//!
//! These tests assert on the exact score values returned by search operations
//! to ensure future changes do not alter the scoring behavior.
//! Scores are always computed using the exact f32 distance functions (even for
//! compacted segments, which rescore with raw vectors after quantized HNSW navigation).

use oramacore_fields::embedding::{
    DeletionThreshold, DistanceMetric, EmbeddingConfig, EmbeddingIndexer, EmbeddingStorage,
    SegmentConfig,
};
use tempfile::TempDir;

fn make_storage(dimensions: usize, metric: DistanceMetric) -> (TempDir, EmbeddingStorage) {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(dimensions, metric).unwrap();
    let storage =
        EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();
    (tmp, storage)
}

fn make_storage_with_segment_config(
    dimensions: usize,
    metric: DistanceMetric,
    segment_config: SegmentConfig,
) -> (TempDir, EmbeddingStorage) {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(dimensions, metric).unwrap();
    let storage = EmbeddingStorage::new(tmp.path().to_path_buf(), config, segment_config).unwrap();
    (tmp, storage)
}

fn iv(dims: usize, v: &[f32]) -> oramacore_fields::embedding::IndexedValue {
    EmbeddingIndexer::new(dims).index_vec(v).unwrap()
}

/// Assert a score equals the expected value within a small tolerance.
fn assert_score(actual: f32, expected: f32, label: &str) {
    let eps = 1e-6;
    assert!(
        (actual - expected).abs() < eps,
        "{label}: expected {expected}, got {actual} (diff = {})",
        (actual - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// L2 distance: sum of squared differences
// ---------------------------------------------------------------------------

#[test]
fn test_l2_scores_live_only() {
    // L2 distance = sum((a_i - b_i)^2)
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    // doc 1: origin
    storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
    // doc 2: unit along x => L2 to origin = 1.0
    storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
    // doc 3: (1,1,0) => L2 to origin = 2.0
    storage.insert(3, iv(3, &[1.0, 1.0, 0.0]));
    // doc 4: (1,1,1) => L2 to origin = 3.0
    storage.insert(4, iv(3, &[1.0, 1.0, 1.0]));
    // doc 5: (3,4,0) => L2 to origin = 25.0
    storage.insert(5, iv(3, &[3.0, 4.0, 0.0]));

    let results = storage.search(&[0.0, 0.0, 0.0], 5, None).unwrap();

    assert_eq!(results.len(), 5);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1 L2");

    assert_eq!(results[1].0, 2);
    assert_score(results[1].1, 1.0, "doc2 L2");

    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 2.0, "doc3 L2");

    assert_eq!(results[3].0, 4);
    assert_score(results[3].1, 3.0, "doc4 L2");

    assert_eq!(results[4].0, 5);
    assert_score(results[4].1, 25.0, "doc5 L2");
}

#[test]
fn test_l2_scores_non_origin_query() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    // Query will be (3, 4)
    // doc 1: (0,0) => L2 = 9 + 16 = 25
    storage.insert(1, iv(2, &[0.0, 0.0]));
    // doc 2: (3,4) => L2 = 0
    storage.insert(2, iv(2, &[3.0, 4.0]));
    // doc 3: (4,4) => L2 = 1
    storage.insert(3, iv(2, &[4.0, 4.0]));
    // doc 4: (3,3) => L2 = 1
    storage.insert(4, iv(2, &[3.0, 3.0]));

    let results = storage.search(&[3.0, 4.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 2);
    assert_score(results[0].1, 0.0, "exact match");

    // doc 3 and 4 both have L2 = 1.0
    assert_score(results[1].1, 1.0, "second closest");
    assert_score(results[2].1, 1.0, "third closest");

    assert_eq!(results[3].0, 1);
    assert_score(results[3].1, 25.0, "origin");
}

#[test]
fn test_l2_scores_after_compaction() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
    storage.insert(3, iv(3, &[0.0, 2.0, 0.0]));
    storage.insert(4, iv(3, &[0.0, 0.0, 3.0]));

    storage.compact(1).unwrap();

    // Scores should be identical after compaction (Phase 2 rescores with raw f32)
    let results = storage.search(&[0.0, 0.0, 0.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    // doc 1: distance 0
    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1 L2 compacted");

    // doc 2: distance 1
    assert_eq!(results[1].0, 2);
    assert_score(results[1].1, 1.0, "doc2 L2 compacted");

    // doc 3: distance 4
    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 4.0, "doc3 L2 compacted");

    // doc 4: distance 9
    assert_eq!(results[3].0, 4);
    assert_score(results[3].1, 9.0, "doc4 L2 compacted");
}

#[test]
fn test_l2_scores_mixed_live_and_compacted() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    // Insert and compact first batch
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    // Insert second batch (live layer)
    storage.insert(3, iv(2, &[0.0, 1.0]));
    storage.insert(4, iv(2, &[2.0, 0.0]));

    // Query from origin: results merge live + compacted
    let results = storage.search(&[0.0, 0.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1 compacted origin");

    // doc 2 and doc 3 both have L2 = 1.0
    assert_score(results[1].1, 1.0, "second");
    assert_score(results[2].1, 1.0, "third");

    assert_eq!(results[3].0, 4);
    assert_score(results[3].1, 4.0, "doc4 live");
}

#[test]
fn test_l2_scores_after_deletion() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0])); // distance 0 to origin
    storage.insert(2, iv(2, &[1.0, 0.0])); // distance 1
    storage.insert(3, iv(2, &[2.0, 0.0])); // distance 4

    storage.delete(2); // remove the middle one

    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();

    assert_eq!(results.len(), 2);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1 after delete");

    assert_eq!(results[1].0, 3);
    assert_score(results[1].1, 4.0, "doc3 after delete");
}

// ---------------------------------------------------------------------------
// Dot product distance: -sum(a_i * b_i)
// ---------------------------------------------------------------------------

#[test]
fn test_dot_product_scores_live_only() {
    // DotProduct distance = -dot(a, b)
    let (_tmp, storage) = make_storage(3, DistanceMetric::DotProduct);

    // doc 1: (1,0,0), query=(1,0,0) => dot=1, dist=-1
    storage.insert(1, iv(3, &[1.0, 0.0, 0.0]));
    // doc 2: (0,1,0), query=(1,0,0) => dot=0, dist=0
    storage.insert(2, iv(3, &[0.0, 1.0, 0.0]));
    // doc 3: (2,0,0), query=(1,0,0) => dot=2, dist=-2
    storage.insert(3, iv(3, &[2.0, 0.0, 0.0]));
    // doc 4: (-1,0,0), query=(1,0,0) => dot=-1, dist=1
    storage.insert(4, iv(3, &[-1.0, 0.0, 0.0]));
    // doc 5: (3,4,0), query=(1,0,0) => dot=3, dist=-3
    storage.insert(5, iv(3, &[3.0, 4.0, 0.0]));

    let results = storage.search(&[1.0, 0.0, 0.0], 5, None).unwrap();

    assert_eq!(results.len(), 5);

    // Sorted ascending by distance: -3, -2, -1, 0, 1
    assert_eq!(results[0].0, 5);
    assert_score(results[0].1, -3.0, "doc5 dot");

    assert_eq!(results[1].0, 3);
    assert_score(results[1].1, -2.0, "doc3 dot");

    assert_eq!(results[2].0, 1);
    assert_score(results[2].1, -1.0, "doc1 dot");

    assert_eq!(results[3].0, 2);
    assert_score(results[3].1, 0.0, "doc2 dot");

    assert_eq!(results[4].0, 4);
    assert_score(results[4].1, 1.0, "doc4 dot");
}

#[test]
fn test_dot_product_scores_after_compaction() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::DotProduct);

    // query will be (1, 2)
    // doc 1: (1,2) => dot=1+4=5, dist=-5
    storage.insert(1, iv(2, &[1.0, 2.0]));
    // doc 2: (0,0) => dot=0, dist=0
    storage.insert(2, iv(2, &[0.0, 0.0]));
    // doc 3: (-1,-2) => dot=-1-4=-5, dist=5
    storage.insert(3, iv(2, &[-1.0, -2.0]));
    // doc 4: (2,1) => dot=2+2=4, dist=-4
    storage.insert(4, iv(2, &[2.0, 1.0]));

    storage.compact(1).unwrap();

    let results = storage.search(&[1.0, 2.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, -5.0, "doc1 dot compacted");

    assert_eq!(results[1].0, 4);
    assert_score(results[1].1, -4.0, "doc4 dot compacted");

    assert_eq!(results[2].0, 2);
    assert_score(results[2].1, 0.0, "doc2 dot compacted");

    assert_eq!(results[3].0, 3);
    assert_score(results[3].1, 5.0, "doc3 dot compacted");
}

// ---------------------------------------------------------------------------
// Cosine distance: 1 - dot(a,b) / (|a| * |b|)
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_scores_live_only() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::Cosine);

    // doc 1: same direction as query (1,0) => cosine_dist = 0
    storage.insert(1, iv(2, &[5.0, 0.0]));
    // doc 2: orthogonal => cosine_dist = 1
    storage.insert(2, iv(2, &[0.0, 3.0]));
    // doc 3: opposite direction => cosine_dist = 2
    storage.insert(3, iv(2, &[-2.0, 0.0]));
    // doc 4: 45 degrees => cosine_dist = 1 - cos(45) = 1 - 1/sqrt(2)
    storage.insert(4, iv(2, &[1.0, 1.0]));

    let results = storage.search(&[1.0, 0.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    // Same direction: distance = 0
    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "same direction cosine");

    // 45 degrees: 1 - 1/sqrt(2) ≈ 0.29289
    assert_eq!(results[1].0, 4);
    assert_score(results[1].1, 1.0 - 1.0 / 2.0_f32.sqrt(), "45 deg cosine");

    // Orthogonal: distance = 1
    assert_eq!(results[2].0, 2);
    assert_score(results[2].1, 1.0, "orthogonal cosine");

    // Opposite: distance = 2
    assert_eq!(results[3].0, 3);
    assert_score(results[3].1, 2.0, "opposite cosine");
}

#[test]
fn test_cosine_scores_after_compaction() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::Cosine);

    // query: (1, 0, 0)
    // doc 1: (1,0,0) => cos = 1, dist = 0
    storage.insert(1, iv(3, &[1.0, 0.0, 0.0]));
    // doc 2: (1,1,0) => cos = 1/sqrt(2), dist = 1 - 1/sqrt(2)
    storage.insert(2, iv(3, &[1.0, 1.0, 0.0]));
    // doc 3: (1,1,1) => cos = 1/sqrt(3), dist = 1 - 1/sqrt(3)
    storage.insert(3, iv(3, &[1.0, 1.0, 1.0]));
    // doc 4: (0,0,1) => cos = 0, dist = 1
    storage.insert(4, iv(3, &[0.0, 0.0, 1.0]));

    storage.compact(1).unwrap();

    let results = storage.search(&[1.0, 0.0, 0.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "aligned cosine compacted");

    assert_eq!(results[1].0, 2);
    assert_score(
        results[1].1,
        1.0 - 1.0 / 2.0_f32.sqrt(),
        "45 deg cosine compacted",
    );

    assert_eq!(results[2].0, 3);
    assert_score(
        results[2].1,
        1.0 - 1.0 / 3.0_f32.sqrt(),
        "54.7 deg cosine compacted",
    );

    assert_eq!(results[3].0, 4);
    assert_score(results[3].1, 1.0, "orthogonal cosine compacted");
}

#[test]
fn test_cosine_zero_vector_score() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::Cosine);

    // Zero vector should have cosine distance = 1.0 (special case in implementation)
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));

    let results = storage.search(&[1.0, 0.0], 2, None).unwrap();

    assert_eq!(results.len(), 2);

    assert_eq!(results[0].0, 2);
    assert_score(results[0].1, 0.0, "non-zero aligned");

    assert_eq!(results[1].0, 1);
    assert_score(results[1].1, 1.0, "zero vector cosine");
}

// ---------------------------------------------------------------------------
// Multi-segment score consistency
// ---------------------------------------------------------------------------

#[test]
fn test_l2_scores_across_multiple_segments() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 5,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(2, DistanceMetric::L2, seg_config);

    // Insert 5 docs and compact → segment 1
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.insert(4, iv(2, &[3.0, 0.0]));
    storage.insert(5, iv(2, &[4.0, 0.0]));
    storage.compact(1).unwrap();

    // Insert 5 more docs and compact → segment 2
    storage.insert(6, iv(2, &[5.0, 0.0]));
    storage.insert(7, iv(2, &[6.0, 0.0]));
    storage.insert(8, iv(2, &[7.0, 0.0]));
    storage.insert(9, iv(2, &[8.0, 0.0]));
    storage.insert(10, iv(2, &[9.0, 0.0]));
    storage.compact(2).unwrap();

    // Query from origin: scores should be L2 distances along x-axis
    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();

    assert_eq!(results.len(), 10);

    // Expected: doc_id=i, distance = (i-1)^2 for i=1..10
    for (idx, (doc_id, score)) in results.iter().enumerate() {
        let expected_doc_id = (idx + 1) as u64;
        let expected_distance = (idx as f32) * (idx as f32);
        assert_eq!(
            *doc_id, expected_doc_id,
            "position {idx}: expected doc {expected_doc_id}, got {doc_id}"
        );
        assert_score(
            *score,
            expected_distance,
            &format!("doc{expected_doc_id} L2 multi-seg"),
        );
    }
}

#[test]
fn test_l2_scores_multi_segment_with_live_layer() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 3,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(2, DistanceMetric::L2, seg_config);

    // Segment 1
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(1).unwrap();

    // Segment 2
    storage.insert(4, iv(2, &[3.0, 0.0]));
    storage.insert(5, iv(2, &[4.0, 0.0]));
    storage.insert(6, iv(2, &[5.0, 0.0]));
    storage.compact(2).unwrap();

    // Live layer (not compacted)
    storage.insert(7, iv(2, &[0.5, 0.0])); // L2 = 0.25

    let results = storage.search(&[0.0, 0.0], 7, None).unwrap();

    assert_eq!(results.len(), 7);

    // doc 1: distance 0
    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1");

    // doc 7 (live): distance 0.25
    assert_eq!(results[1].0, 7);
    assert_score(results[1].1, 0.25, "doc7 live");

    // doc 2: distance 1
    assert_eq!(results[2].0, 2);
    assert_score(results[2].1, 1.0, "doc2");

    // doc 3: distance 4
    assert_eq!(results[3].0, 3);
    assert_score(results[3].1, 4.0, "doc3");

    // doc 4: distance 9
    assert_eq!(results[4].0, 4);
    assert_score(results[4].1, 9.0, "doc4");

    // doc 5: distance 16
    assert_eq!(results[5].0, 5);
    assert_score(results[5].1, 16.0, "doc5");

    // doc 6: distance 25
    assert_eq!(results[6].0, 6);
    assert_score(results[6].1, 25.0, "doc6");
}

// ---------------------------------------------------------------------------
// Score stability across compaction cycles
// ---------------------------------------------------------------------------

#[test]
fn test_scores_stable_across_multiple_compactions() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, iv(3, &[1.0, 2.0, 3.0]));
    storage.insert(2, iv(3, &[4.0, 5.0, 6.0]));
    storage.insert(3, iv(3, &[7.0, 8.0, 9.0]));

    let query = [2.0, 3.0, 4.0];

    // Scores from live layer
    let live_results = storage.search(&query, 3, None).unwrap();

    // Expected distances from (2,3,4):
    // doc 1: (1-2)^2 + (2-3)^2 + (3-4)^2 = 1+1+1 = 3
    // doc 2: (4-2)^2 + (5-3)^2 + (6-4)^2 = 4+4+4 = 12
    // doc 3: (7-2)^2 + (8-3)^2 + (9-4)^2 = 25+25+25 = 75
    assert_eq!(live_results.len(), 3);
    assert_eq!(live_results[0].0, 1);
    assert_score(live_results[0].1, 3.0, "doc1 live");
    assert_eq!(live_results[1].0, 2);
    assert_score(live_results[1].1, 12.0, "doc2 live");
    assert_eq!(live_results[2].0, 3);
    assert_score(live_results[2].1, 75.0, "doc3 live");

    // Compact and verify scores remain identical
    storage.compact(1).unwrap();
    let compacted_results = storage.search(&query, 3, None).unwrap();

    assert_eq!(compacted_results.len(), 3);
    for i in 0..3 {
        assert_eq!(compacted_results[i].0, live_results[i].0);
        assert_score(
            compacted_results[i].1,
            live_results[i].1,
            &format!("compaction 1 doc {}", live_results[i].0),
        );
    }

    // Add more data, compact again, verify original scores unchanged
    storage.insert(4, iv(3, &[0.0, 0.0, 0.0]));
    storage.compact(2).unwrap();
    let results_v2 = storage.search(&query, 4, None).unwrap();

    // doc 4: (0-2)^2 + (0-3)^2 + (0-4)^2 = 4+9+16 = 29
    // Order: doc1(3), doc2(12), doc4(29), doc3(75)
    assert_eq!(results_v2.len(), 4);
    assert_eq!(results_v2[0].0, 1);
    assert_score(results_v2[0].1, 3.0, "doc1 v2");
    assert_eq!(results_v2[1].0, 2);
    assert_score(results_v2[1].1, 12.0, "doc2 v2");
    assert_eq!(results_v2[2].0, 4);
    assert_score(results_v2[2].1, 29.0, "doc4 v2");
    assert_eq!(results_v2[3].0, 3);
    assert_score(results_v2[3].1, 75.0, "doc3 v2");
}

// ---------------------------------------------------------------------------
// Score values with fractional / negative components
// ---------------------------------------------------------------------------

#[test]
fn test_l2_scores_fractional_vectors() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.5, 0.5]));
    storage.insert(2, iv(2, &[1.5, 2.5]));

    // query: (0.5, 0.5)
    // doc 1: distance = 0
    // doc 2: (1.5-0.5)^2 + (2.5-0.5)^2 = 1.0 + 4.0 = 5.0
    let results = storage.search(&[0.5, 0.5], 2, None).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "doc1 fractional");
    assert_eq!(results[1].0, 2);
    assert_score(results[1].1, 5.0, "doc2 fractional");
}

#[test]
fn test_dot_product_scores_negative_vectors() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::DotProduct);

    // query: (1, -1)
    // doc 1: (1, 1) => dot = 1*1 + (-1)*1 = 0, dist = 0
    storage.insert(1, iv(2, &[1.0, 1.0]));
    // doc 2: (1, -1) => dot = 1 + 1 = 2, dist = -2
    storage.insert(2, iv(2, &[1.0, -1.0]));
    // doc 3: (-1, 1) => dot = -1 + (-1) = -2, dist = 2
    storage.insert(3, iv(2, &[-1.0, 1.0]));

    let results = storage.search(&[1.0, -1.0], 3, None).unwrap();

    assert_eq!(results.len(), 3);

    assert_eq!(results[0].0, 2);
    assert_score(results[0].1, -2.0, "aligned dot neg");

    assert_eq!(results[1].0, 1);
    assert_score(results[1].1, 0.0, "orthogonal dot neg");

    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 2.0, "anti-aligned dot neg");
}

// ---------------------------------------------------------------------------
// Higher-dimensional score correctness
// ---------------------------------------------------------------------------

#[test]
fn test_l2_scores_higher_dimensions() {
    let (_tmp, storage) = make_storage(8, DistanceMetric::L2);

    let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    storage.insert(1, iv(8, &v1));
    storage.insert(2, iv(8, &v2));
    storage.insert(3, iv(8, &v3));

    // query: all zeros
    // doc 1: L2 = 1
    // doc 2: L2 = 1
    // doc 3: L2 = 8 (1^2 * 8)
    let results = storage.search(&[0.0; 8], 3, None).unwrap();

    assert_eq!(results.len(), 3);

    assert_score(results[0].1, 1.0, "v1 8dim");
    assert_score(results[1].1, 1.0, "v2 8dim");
    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 8.0, "v3 8dim");
}

#[test]
fn test_cosine_scores_higher_dimensions() {
    let (_tmp, storage) = make_storage(4, DistanceMetric::Cosine);

    // query: (1, 0, 0, 0)
    // doc 1: (1,0,0,0) => cos_dist = 0
    storage.insert(1, iv(4, &[1.0, 0.0, 0.0, 0.0]));
    // doc 2: (1,1,0,0) => cos = 1/sqrt(2), dist = 1 - 1/sqrt(2)
    storage.insert(2, iv(4, &[1.0, 1.0, 0.0, 0.0]));
    // doc 3: (1,1,1,0) => cos = 1/sqrt(3), dist = 1 - 1/sqrt(3)
    storage.insert(3, iv(4, &[1.0, 1.0, 1.0, 0.0]));
    // doc 4: (1,1,1,1) => cos = 1/sqrt(4) = 0.5, dist = 0.5
    storage.insert(4, iv(4, &[1.0, 1.0, 1.0, 1.0]));

    storage.compact(1).unwrap();

    let results = storage.search(&[1.0, 0.0, 0.0, 0.0], 4, None).unwrap();

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 0.0, "aligned 4d");

    assert_eq!(results[1].0, 2);
    assert_score(results[1].1, 1.0 - 1.0 / 2.0_f32.sqrt(), "2-comp 4d");

    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 1.0 - 1.0 / 3.0_f32.sqrt(), "3-comp 4d");

    assert_eq!(results[3].0, 4);
    assert_score(results[3].1, 1.0 - 1.0 / 4.0_f32.sqrt(), "4-comp 4d");
}

// ---------------------------------------------------------------------------
// K truncation preserves correct scores
// ---------------------------------------------------------------------------

#[test]
fn test_k_truncation_returns_correct_scores() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    for i in 1..=10u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }

    // Query from origin, only ask for top 3
    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();

    assert_eq!(results.len(), 3);

    // Closest: doc1(1), doc2(4), doc3(9)
    assert_eq!(results[0].0, 1);
    assert_score(results[0].1, 1.0, "top1");

    assert_eq!(results[1].0, 2);
    assert_score(results[1].1, 4.0, "top2");

    assert_eq!(results[2].0, 3);
    assert_score(results[2].1, 9.0, "top3");
}
