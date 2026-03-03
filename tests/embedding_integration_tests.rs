use oramacore_fields::embedding::{
    DeletionThreshold, DistanceMetric, EmbeddingConfig, EmbeddingIndexer, EmbeddingStorage,
    SegmentConfig,
};
use std::sync::Arc;
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

/// Helper: build an IndexedValue from a float slice.
fn iv(dims: usize, v: &[f32]) -> oramacore_fields::embedding::IndexedValue {
    EmbeddingIndexer::new(dims).index_vec(v).unwrap()
}

#[test]
fn test_insert_and_search_live_only() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
    storage.insert(3, iv(3, &[10.0, 10.0, 10.0]));

    let results = storage.search(&[0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest
    assert_eq!(results[1].0, 2);
}

#[test]
fn test_insert_compact_search() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
    storage.insert(3, iv(3, &[10.0, 10.0, 10.0]));

    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_insert_delete_search() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));

    storage.delete(1);

    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();
    assert!(results.iter().all(|(id, _)| *id != 1));
}

#[test]
fn test_compact_then_delete() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));

    storage.compact(1).unwrap();

    storage.delete(1);

    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();
    assert!(results.iter().all(|(id, _)| *id != 1));
}

#[test]
fn test_multiple_compactions() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    // Round 1
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    // Round 2
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(2).unwrap();

    // Round 3
    storage.insert(4, iv(2, &[3.0, 0.0]));
    storage.compact(3).unwrap();

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].0, 1); // closest to origin
}

#[test]
fn test_persistence() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();

    // Create and populate
    {
        let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
        let storage =
            EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default()).unwrap();
        storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
        storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
        storage.insert(3, iv(3, &[0.0, 1.0, 0.0]));
        storage.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = EmbeddingStorage::new(base_path, config, SegmentConfig::default()).unwrap();
        assert_eq!(storage.current_version_number(), 1);

        let results = storage.search(&[0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1); // closest
    }
}

#[test]
fn test_dimension_mismatch() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    // Search with wrong dimensions
    assert!(storage.search(&[1.0, 2.0], 1, None).is_err());
}

#[test]
fn test_empty_index_search() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);
    let results = storage.search(&[0.0, 0.0, 0.0], 5, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_k_zero() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    let results = storage.search(&[0.0, 0.0], 0, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_non_finite_value_rejected() {
    let indexer = EmbeddingIndexer::new(2);
    assert!(indexer.index_vec(&[f32::NAN, 0.0]).is_none());
    assert!(indexer.index_vec(&[f32::INFINITY, 0.0]).is_none());
    assert!(indexer.index_vec(&[f32::NEG_INFINITY, 0.0]).is_none());
}

#[test]
fn test_empty_vector_rejected() {
    let indexer = EmbeddingIndexer::new(2);
    assert!(indexer.index_vec(&[]).is_none());
}

#[test]
fn test_cosine_metric() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::Cosine);

    storage.insert(1, iv(3, &[1.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[0.0, 1.0, 0.0]));
    storage.insert(3, iv(3, &[0.9, 0.1, 0.0]));

    let results = storage.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
    assert_eq!(results[0].0, 1); // exact match is closest
}

#[test]
fn test_dot_product_metric() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::DotProduct);

    storage.insert(1, iv(3, &[1.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[0.0, 1.0, 0.0]));
    storage.insert(3, iv(3, &[5.0, 0.0, 0.0])); // highest dot product with query

    let results = storage.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
    // For dot product distance = -dot, so highest dot product = lowest distance
    assert_eq!(results[0].0, 3);
}

#[test]
fn test_delete_and_reinsert() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.compact(1).unwrap();

    storage.delete(1);
    storage.insert(1, iv(2, &[5.0, 5.0])); // reinsert with different vector

    let results = storage.search(&[5.0, 5.0], 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01); // should be very close to query
}

#[test]
fn test_compact_with_deletes() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(1).unwrap();

    storage.delete(2);
    storage.compact(2).unwrap();

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(id, _)| *id != 2));
}

#[test]
fn test_compact_empty_index() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    // Compact with nothing to do
    storage.compact(1).unwrap();
    assert_eq!(storage.current_version_number(), 0);

    // Insert and compact
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.compact(2).unwrap();
    assert_eq!(storage.current_version_number(), 2);
}

#[test]
fn test_cleanup() {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
    let storage =
        EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.compact(1).unwrap();

    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(2).unwrap();

    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(3).unwrap();

    assert!(tmp.path().join("versions/1").exists());
    assert!(tmp.path().join("versions/2").exists());
    assert!(tmp.path().join("versions/3").exists());

    storage.cleanup();

    assert!(!tmp.path().join("versions/1").exists());
    assert!(!tmp.path().join("versions/2").exists());
    assert!(tmp.path().join("versions/3").exists()); // current preserved
}

#[test]
fn test_integrity_check() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.compact(1).unwrap();

    let result = storage.integrity_check();
    assert!(result.passed);
}

#[test]
fn test_info() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::Cosine);

    storage.insert(1, iv(3, &[0.0, 0.0, 0.0]));
    storage.insert(2, iv(3, &[1.0, 0.0, 0.0]));
    storage.compact(1).unwrap();

    let info = storage.info();
    assert_eq!(info.num_embeddings, 2);
    assert_eq!(info.dimensions, 3);
    assert_eq!(info.current_version_number, 1);
}

#[test]
fn test_large_scale_recall() {
    let (_tmp, storage) = make_storage(32, DistanceMetric::L2);

    // Insert 1000 random vectors
    let mut rng = rand::rng();
    let mut all_vectors: Vec<(u64, Vec<f32>)> = Vec::new();

    for i in 0..1000u64 {
        let vec: Vec<f32> = (0..32)
            .map(|_| {
                use rand::RngExt;
                rng.random::<f32>()
            })
            .collect();
        storage.insert(i, iv(32, &vec));
        all_vectors.push((i, vec));
    }

    storage.compact(1).unwrap();

    // Query with a few random vectors and check recall
    let query: Vec<f32> = (0..32)
        .map(|_| {
            use rand::RngExt;
            rng.random::<f32>()
        })
        .collect();

    // Brute force ground truth
    let mut ground_truth: Vec<(u64, f32)> = all_vectors
        .iter()
        .map(|(id, v)| {
            let dist: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (*id, dist)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let top10_truth: Vec<u64> = ground_truth.iter().take(10).map(|(id, _)| *id).collect();

    // HNSW search with high ef for better recall
    let results = storage.search(&query, 10, Some(200)).unwrap();
    let result_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

    // Check recall: at least 9 out of 10 should match
    let recall = top10_truth
        .iter()
        .filter(|id| result_ids.contains(id))
        .count();
    assert!(
        recall >= 9,
        "Recall too low: {recall}/10. Ground truth: {top10_truth:?}, Results: {result_ids:?}"
    );
}

#[test]
fn test_concurrent_search() {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
    let storage = Arc::new(
        EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap(),
    );

    for i in 0..100u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    let mut handles = Vec::new();
    for _ in 0..4 {
        let s = Arc::clone(&storage);
        handles.push(std::thread::spawn(move || {
            for _ in 0..100 {
                let results = s.search(&[50.0, 0.0, 0.0], 5, None).unwrap();
                assert_eq!(results.len(), 5);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ──────────────────────────────────────────────────
// Multi-segment specific tests
// ──────────────────────────────────────────────────

#[test]
fn test_multiple_segments_created() {
    // Use small max_nodes_per_segment to force multiple segments
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 10,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Insert 15 vectors and compact — should create 1 segment
    for i in 0..15u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    // Insert 10 more and compact — should create second segment since first is full
    for i in 15..25u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    // Verify all 25 vectors are searchable
    let results = storage
        .search(&[0.0, 0.0, 0.0, 0.0], 25, Some(200))
        .unwrap();
    assert_eq!(results.len(), 25);

    // Verify the segment structure on disk
    let manifest_path = _tmp.path().join("versions/2/manifest.json");
    assert!(manifest_path.exists());
    let manifest_content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    let segments = manifest["segments"].as_array().unwrap();
    assert!(
        segments.len() >= 2,
        "Expected at least 2 segments, got {}",
        segments.len()
    );
}

#[test]
fn test_incremental_insert() {
    // Set thresholds to favour incremental insert
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3, // 30% threshold
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Round 1: create segment with 50 nodes
    for i in 0..50u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    // Round 2: insert 10 more (20% of 50 = below 30% threshold => incremental insert)
    for i in 50..60u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    // Verify manifest has insertions_since_rebuild > 0
    let manifest_path = _tmp.path().join("versions/2/manifest.json");
    let manifest_content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    let seg = &manifest["segments"][0];
    assert_eq!(seg["num_nodes"].as_u64().unwrap(), 60);
    assert_eq!(seg["insertions_since_rebuild"].as_u64().unwrap(), 10);
    assert_eq!(seg["nodes_at_last_rebuild"].as_u64().unwrap(), 50);

    // All 60 vectors should be searchable
    let results = storage
        .search(&[0.0, 0.0, 0.0, 0.0], 60, Some(200))
        .unwrap();
    assert_eq!(results.len(), 60);
}

#[test]
fn test_full_rebuild_on_insertion_threshold() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Round 1: create segment with 50 nodes
    for i in 0..50u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    // Round 2: incremental insert 10 (20% < 30%)
    for i in 50..60u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    // Round 3: insert 8 more. insertions_since_rebuild=10+8=18, nodes_at_last_rebuild=50.
    // ratio = 18/50 = 0.36 > 0.3 -> FULL REBUILD
    for i in 60..68u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(3).unwrap();

    // After full rebuild, counters should be reset
    let manifest_path = _tmp.path().join("versions/3/manifest.json");
    let manifest_content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    let seg = &manifest["segments"][0];
    assert_eq!(seg["num_nodes"].as_u64().unwrap(), 68);
    assert_eq!(seg["insertions_since_rebuild"].as_u64().unwrap(), 0); // Reset
    assert_eq!(seg["nodes_at_last_rebuild"].as_u64().unwrap(), 68); // Reset

    // All vectors searchable
    let results = storage
        .search(&[0.0, 0.0, 0.0, 0.0], 68, Some(200))
        .unwrap();
    assert_eq!(results.len(), 68);
}

#[test]
fn test_full_rebuild_on_deletion_threshold() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: 0.1f64.try_into().unwrap(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Insert 20 vectors and compact
    for i in 0..20u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    // Delete 3 out of 20 (15% > 10% threshold) -> should trigger full rebuild
    storage.delete(5);
    storage.delete(10);
    storage.delete(15);
    storage.compact(2).unwrap();

    // After rebuild, deletes should be 0
    let manifest_path = _tmp.path().join("versions/2/manifest.json");
    let manifest_content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    let seg = &manifest["segments"][0];
    assert_eq!(seg["num_deletes"].as_u64().unwrap(), 0);
    assert_eq!(seg["num_nodes"].as_u64().unwrap(), 17); // 20 - 3

    // Deleted vectors should not be searchable
    let results = storage
        .search(&[5.0, 0.0, 0.0, 0.0], 20, Some(200))
        .unwrap();
    assert!(results
        .iter()
        .all(|(id, _)| *id != 5 && *id != 10 && *id != 15));
    assert_eq!(results.len(), 17);
}

#[test]
fn test_carry_forward_deletes() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 1000,
        deletion_threshold: 0.5f64.try_into().unwrap(), // High threshold — won't trigger rebuild
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Insert 20 vectors and compact
    for i in 0..20u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    // Delete 1 out of 20 (5% < 50% threshold) → deletes carried forward, same segment_id
    storage.delete(5);
    storage.compact(2).unwrap();

    let manifest_path = _tmp.path().join("versions/2/manifest.json");
    let manifest_content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    let seg = &manifest["segments"][0];
    assert_eq!(seg["num_deletes"].as_u64().unwrap(), 1);
    assert_eq!(seg["num_nodes"].as_u64().unwrap(), 20); // Nodes unchanged — just delete carried

    // But the deleted vector should not appear in results
    let results = storage
        .search(&[5.0, 0.0, 0.0, 0.0], 20, Some(200))
        .unwrap();
    assert!(results.iter().all(|(id, _)| *id != 5));
}

#[test]
fn test_persistence_multi_segment() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 10,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };

    // Create with multiple segments
    {
        let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = EmbeddingStorage::new(base_path.clone(), config, seg_config.clone()).unwrap();

        // Fill first segment
        for i in 0..12u64 {
            storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
        }
        storage.compact(1).unwrap();

        // Create second segment
        for i in 12..24u64 {
            storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
        }
        storage.compact(2).unwrap();
    }

    // Reopen and verify
    {
        let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = EmbeddingStorage::new(base_path, config, seg_config).unwrap();
        assert_eq!(storage.current_version_number(), 2);

        let results = storage.search(&[0.0, 0.0, 0.0], 24, Some(200)).unwrap();
        assert_eq!(results.len(), 24);
    }
}

#[test]
fn test_cleanup_removes_old_segments() {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.0, // Always full rebuild for predictability
    };
    let storage = EmbeddingStorage::new(tmp.path().to_path_buf(), config, seg_config).unwrap();

    // Round 1: creates seg_1
    for i in 0..5u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    assert!(tmp.path().join("segments/seg_1").exists());

    // Round 2: full rebuild creates seg_2 (since insertion_rebuild_threshold=0.0)
    for i in 5..8u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();
    // seg_1 should still exist (not cleaned up yet), seg_2 is new
    assert!(tmp.path().join("segments/seg_2").exists());

    storage.cleanup();
    // After cleanup: old segments not in current manifest removed
    assert!(!tmp.path().join("segments/seg_1").exists());
    assert!(tmp.path().join("segments/seg_2").exists());
}

#[test]
fn test_worked_example() {
    // Implements section 11 of MULTI_SEGMENTS_4.md
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: 0.1f64.try_into().unwrap(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage) = make_storage_with_segment_config(4, DistanceMetric::L2, seg_config);

    // Round 1: Insert 50 vectors (doc_ids 0-49), compact
    // Expected: CREATE NEW segment (seg_1)
    for i in 0..50u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    let m1 = read_manifest(_tmp.path(), 1);
    assert_eq!(m1.len(), 1);
    assert_eq!(m1[0]["num_nodes"].as_u64().unwrap(), 50);
    assert_eq!(m1[0]["insertions_since_rebuild"].as_u64().unwrap(), 0);
    assert_eq!(m1[0]["nodes_at_last_rebuild"].as_u64().unwrap(), 50);

    // Round 2: Insert 10 vectors (doc_ids 50-59), compact
    // insertion_ratio = 10/50 = 0.2 < 0.3 → INCREMENTAL INSERT
    for i in 50..60u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    let m2 = read_manifest(_tmp.path(), 2);
    assert_eq!(m2.len(), 1);
    assert_eq!(m2[0]["num_nodes"].as_u64().unwrap(), 60);
    assert_eq!(m2[0]["insertions_since_rebuild"].as_u64().unwrap(), 10);
    assert_eq!(m2[0]["nodes_at_last_rebuild"].as_u64().unwrap(), 50);
    // Segment id should be different (copy-on-modify)
    let seg2_id = m2[0]["segment_id"].as_u64();
    let seg1_id = m1[0]["segment_id"].as_u64();
    assert_ne!(seg2_id, seg1_id);

    // Round 3: Insert 8 vectors (doc_ids 60-67), compact
    // insertion_ratio = (10+8)/50 = 0.36 > 0.3 → FULL REBUILD
    for i in 60..68u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(3).unwrap();

    let m3 = read_manifest(_tmp.path(), 3);
    assert_eq!(m3.len(), 1);
    assert_eq!(m3[0]["num_nodes"].as_u64().unwrap(), 68);
    assert_eq!(m3[0]["insertions_since_rebuild"].as_u64().unwrap(), 0); // Reset
    assert_eq!(m3[0]["nodes_at_last_rebuild"].as_u64().unwrap(), 68); // Reset

    // Round 4: Insert 40 vectors (doc_ids 68-107), compact
    // 68 + 40 = 108 > 100 (can't absorb) → CARRY FORWARD seg_3 + CREATE NEW seg_4
    for i in 68..108u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(4).unwrap();

    let m4 = read_manifest(_tmp.path(), 4);
    assert_eq!(m4.len(), 2, "Expected 2 segments after overflow");
    assert_eq!(m4[0]["num_nodes"].as_u64().unwrap(), 68);
    assert_eq!(m4[1]["num_nodes"].as_u64().unwrap(), 40);

    // Round 5: Delete 8 from seg_3's range + insert 5 vectors (doc_ids 108-112)
    // seg_3 (non-last): deletion_ratio = 8/68 = 0.118 > 0.1 → FULL REBUILD
    // seg_4 (last): 40 + 5 = 45 ≤ 100, insertion_ratio = 5/40 = 0.125 < 0.3 → INCREMENTAL INSERT
    for &doc_id in &[5u64, 10, 15, 20, 25, 30, 35, 40] {
        storage.delete(doc_id);
    }
    for i in 108..113u64 {
        storage.insert(i, iv(4, &[i as f32, 0.0, 0.0, 0.0]));
    }
    storage.compact(5).unwrap();

    let m5 = read_manifest(_tmp.path(), 5);
    assert_eq!(m5.len(), 2, "Expected 2 segments after round 5");
    // First segment: rebuilt from 60 surviving vectors
    assert_eq!(m5[0]["num_nodes"].as_u64().unwrap(), 60);
    assert_eq!(m5[0]["num_deletes"].as_u64().unwrap(), 0);
    assert_eq!(m5[0]["insertions_since_rebuild"].as_u64().unwrap(), 0);
    // Second segment: incremental insert from 40 + 5
    assert_eq!(m5[1]["num_nodes"].as_u64().unwrap(), 45);
    assert_eq!(m5[1]["insertions_since_rebuild"].as_u64().unwrap(), 5);
    assert_eq!(m5[1]["nodes_at_last_rebuild"].as_u64().unwrap(), 40);

    // Verify all surviving vectors are searchable
    let results = storage
        .search(&[0.0, 0.0, 0.0, 0.0], 200, Some(400))
        .unwrap();
    assert_eq!(results.len(), 105); // 113 total - 8 deleted = 105

    // Verify deleted docs are not in results
    let deleted_ids = [5u64, 10, 15, 20, 25, 30, 35, 40];
    for &del_id in &deleted_ids {
        assert!(
            !results.iter().any(|(id, _)| *id == del_id),
            "Deleted doc_id {del_id} should not appear in results"
        );
    }
}

// ============================================================================
// Additional edge-case, info, and integrity-check tests
// ============================================================================

#[test]
fn test_doc_id_zero() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(0, iv(2, &[0.0, 0.0]));
    storage.insert(1, iv(2, &[1.0, 0.0]));

    let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(results[0].0, 0);

    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(results[0].0, 0);

    storage.delete(0);
    storage.compact(2).unwrap();

    let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
    assert!(results.iter().all(|(id, _)| *id != 0));
}

#[test]
fn test_large_doc_ids() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    let large_id = u64::MAX - 1;
    storage.insert(large_id, iv(2, &[0.0, 0.0]));
    storage.insert(large_id - 1, iv(2, &[1.0, 0.0]));

    let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, large_id);

    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, large_id);
}

#[test]
fn test_delete_nonexistent_noop() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    storage.delete(999); // nonexistent

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 2);

    storage.compact(2).unwrap();

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_duplicate_inserts_same_doc_id() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(1, iv(2, &[5.0, 5.0])); // overwrite

    let results = storage.search(&[5.0, 5.0], 2, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01); // should be near the second insertion point
}

#[test]
fn test_info_empty_index() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    let info = storage.info();
    assert_eq!(info.num_embeddings, 0);
    assert_eq!(info.dimensions, 3);
    assert_eq!(info.current_version_number, 0);
    assert_eq!(info.pending_ops, 0);
}

#[test]
fn test_info_with_pending_ops() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));

    let info = storage.info();
    assert_eq!(info.pending_ops, 2);
    assert_eq!(info.num_embeddings, 2); // not yet compacted

    storage.compact(1).unwrap();

    let info = storage.info();
    assert_eq!(info.pending_ops, 0);
    assert_eq!(info.num_embeddings, 2);
}

#[test]
fn test_info_with_deletes() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    storage.delete(1);
    let info = storage.info();
    assert_eq!(info.pending_ops, 1);
}

#[test]
fn test_integrity_check_before_compaction() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    let result = storage.integrity_check();
    assert!(
        !result.passed,
        "integrity check should fail before any compaction"
    );
}

#[test]
fn test_integrity_check_corrupted_current_file() {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
    let storage =
        EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.compact(1).unwrap();

    // Corrupt the CURRENT file
    std::fs::write(tmp.path().join("CURRENT"), "garbage").unwrap();

    // Recreate storage (will fail to open due to corrupt CURRENT)
    let config2 = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
    let result = EmbeddingStorage::new(tmp.path().to_path_buf(), config2, SegmentConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_integrity_check_missing_segment_file() {
    let tmp = TempDir::new().unwrap();
    let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
    let storage =
        EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    assert!(storage.integrity_check().passed);

    // Remove a required segment file
    let seg_dir = tmp.path().join("segments/seg_1");
    std::fs::remove_file(seg_dir.join("hnsw.graph")).unwrap();

    let result = storage.integrity_check();
    assert!(
        !result.passed,
        "integrity check should fail with missing segment file"
    );
}

#[test]
fn test_persistence_multiple_compactions() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();

    {
        let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
        let storage =
            EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default()).unwrap();

        storage.insert(1, iv(2, &[0.0, 0.0]));
        storage.insert(2, iv(2, &[1.0, 0.0]));
        storage.compact(1).unwrap();

        storage.insert(3, iv(2, &[2.0, 0.0]));
        storage.compact(2).unwrap();
    }

    {
        let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
        let storage = EmbeddingStorage::new(base_path, config, SegmentConfig::default()).unwrap();
        assert_eq!(storage.current_version_number(), 2);

        let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 3);
    }
}

#[test]
fn test_delete_all_docs() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    storage.delete(1);
    storage.delete(2);
    storage.compact(2).unwrap();

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert!(results.is_empty());

    // Can still insert after full deletion
    storage.insert(3, iv(2, &[0.0, 0.0]));
    storage.compact(3).unwrap();

    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 3);
}

#[test]
fn test_search_returns_distances_sorted_ascending() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[3.0, 0.0]));
    storage.insert(3, iv(2, &[1.0, 0.0]));
    storage.insert(4, iv(2, &[5.0, 0.0]));
    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 4, None).unwrap();
    for w in results.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "distances not sorted: {} > {}",
            w[0].1,
            w[1].1
        );
    }
}

#[test]
fn test_search_k_larger_than_total() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 100, None).unwrap();
    assert_eq!(results.len(), 2);
}

// ──────────────────────────────────────────────────
// DocumentFilter tests
// ──────────────────────────────────────────────────

struct AllowSet(std::collections::HashSet<u64>);
impl oramacore_fields::embedding::DocumentFilter for AllowSet {
    fn contains(&self, doc_id: u64) -> bool {
        self.0.contains(&doc_id)
    }
}

struct RejectAll;
impl oramacore_fields::embedding::DocumentFilter for RejectAll {
    fn contains(&self, _doc_id: u64) -> bool {
        false
    }
}

struct AllowAll;
impl oramacore_fields::embedding::DocumentFilter for AllowAll {
    fn contains(&self, _doc_id: u64) -> bool {
        true
    }
}

#[test]
fn test_filter_live_only() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));

    let allowed: AllowSet = AllowSet([1, 3].into_iter().collect());
    let results = storage
        .search_with_filter(&[0.0, 0.0], 10, None, &allowed)
        .unwrap();
    assert_eq!(results.len(), 2);
    let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
    assert!(!ids.contains(&2));
}

#[test]
fn test_filter_after_compact() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(1).unwrap();

    let allowed: AllowSet = AllowSet([1, 3].into_iter().collect());
    let results = storage
        .search_with_filter(&[0.0, 0.0], 10, None, &allowed)
        .unwrap();
    assert_eq!(results.len(), 2);
    let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
    assert!(!ids.contains(&2));
}

#[test]
fn test_filter_excludes_all() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.compact(1).unwrap();

    let results = storage
        .search_with_filter(&[0.0, 0.0], 10, None, &RejectAll)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_filter_with_deletes() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.insert(4, iv(2, &[3.0, 0.0]));
    storage.compact(1).unwrap();

    // Delete doc 1, filter out doc 3 => only 2 and 4 remain
    storage.delete(1);
    let allowed: AllowSet = AllowSet([1, 2, 4].into_iter().collect());
    let results = storage
        .search_with_filter(&[0.0, 0.0], 10, None, &allowed)
        .unwrap();
    let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    assert!(!ids.contains(&1), "deleted doc should not appear");
    assert!(!ids.contains(&3), "filtered doc should not appear");
    assert!(ids.contains(&2));
    assert!(ids.contains(&4));
}

#[test]
fn test_filter_does_not_break_graph_traversal() {
    // Insert many docs forming an HNSW graph, filter out intermediate nodes,
    // verify that docs reachable only *through* filtered nodes are still found.
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    // Insert 50 docs with spread-out embeddings
    for i in 1..=50u64 {
        let v = vec![i as f32, 0.0, 0.0];
        storage.insert(i, iv(3, &v));
    }
    storage.compact(1).unwrap();

    // Filter out all "middle" docs (10..=40), only allow 1..=9 and 41..=50
    let allowed: AllowSet = AllowSet((1..=9).chain(41..=50).collect());
    let results = storage
        .search_with_filter(&[50.0, 0.0, 0.0], 5, Some(200), &allowed)
        .unwrap();

    // Should find docs near 50 even though middle nodes are filtered
    assert!(!results.is_empty());
    // All returned docs should be in the allowed set
    for (id, _) in &results {
        assert!(
            allowed.0.contains(id),
            "returned doc {id} should be in allowed set"
        );
    }
    // Doc 50 should be the closest
    assert_eq!(results[0].0, 50);
}

#[test]
fn test_filter_with_multi_segment() {
    let seg_cfg = SegmentConfig {
        max_nodes_per_segment: 10,
        ..SegmentConfig::default()
    };
    let (_tmp, storage) = make_storage_with_segment_config(2, DistanceMetric::L2, seg_cfg);

    // First batch → segment 1
    for i in 1..=10u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    // Second batch → segment 2
    for i in 11..=20u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(2).unwrap();

    // Filter to allow only docs from second segment
    let allowed: AllowSet = AllowSet((11..=20).collect());
    let results = storage
        .search_with_filter(&[15.0, 0.0], 5, Some(200), &allowed)
        .unwrap();

    assert!(!results.is_empty());
    for (id, _) in &results {
        assert!(
            *id >= 11 && *id <= 20,
            "returned doc {id} should be from segment 2"
        );
    }
}

#[test]
fn test_nofilter_returns_same_as_none() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(1).unwrap();

    let results_none = storage.search(&[0.0, 0.0], 3, None).unwrap();
    let no_filter = oramacore_fields::embedding::NoFilter;
    let results_nofilter = storage
        .search_with_filter(&[0.0, 0.0], 3, None, &no_filter)
        .unwrap();

    assert_eq!(results_none.len(), results_nofilter.len());
    for (a, b) in results_none.iter().zip(results_nofilter.iter()) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6);
    }
}

#[test]
fn test_filter_allow_all() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    storage.insert(1, iv(2, &[0.0, 0.0]));
    storage.insert(2, iv(2, &[1.0, 0.0]));
    storage.insert(3, iv(2, &[2.0, 0.0]));
    storage.compact(1).unwrap();

    let results_none = storage.search(&[0.0, 0.0], 3, None).unwrap();
    let results_allow_all = storage
        .search_with_filter(&[0.0, 0.0], 3, None, &AllowAll)
        .unwrap();

    assert_eq!(results_none.len(), results_allow_all.len());
    for (a, b) in results_none.iter().zip(results_allow_all.iter()) {
        assert_eq!(a.0, b.0);
    }
}

#[test]
fn test_filter_after_incremental_insert() {
    let seg_cfg = SegmentConfig {
        max_nodes_per_segment: 100,
        insertion_rebuild_threshold: 10.0, // high threshold to force incremental inserts
        ..SegmentConfig::default()
    };
    let (_tmp, storage) = make_storage_with_segment_config(2, DistanceMetric::L2, seg_cfg);

    // Initial batch
    for i in 1..=5u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    // Incremental insert (absorbed into existing segment)
    for i in 6..=10u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(2).unwrap();

    // Filter: only allow docs 1,2,8,9,10
    let allowed: AllowSet = AllowSet([1, 2, 8, 9, 10].into_iter().collect());
    let results = storage
        .search_with_filter(&[9.0, 0.0], 5, Some(200), &allowed)
        .unwrap();

    assert!(!results.is_empty());
    for (id, _) in &results {
        assert!(
            allowed.0.contains(id),
            "returned doc {id} should be in allowed set"
        );
    }
    // Doc 9 should be closest
    assert_eq!(results[0].0, 9);
}

// Helper to read manifest from disk
fn read_manifest(base_path: &std::path::Path, version_number: u64) -> Vec<serde_json::Value> {
    let path = base_path
        .join("versions")
        .join(version_number.to_string())
        .join("manifest.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    v["segments"].as_array().unwrap().clone()
}

/// Scan `base_path/segments/` and return sorted list of segment IDs found on disk.
fn list_segment_dirs_on_disk(base_path: &std::path::Path) -> Vec<u64> {
    let segments_dir = base_path.join("segments");
    if !segments_dir.exists() {
        return Vec::new();
    }
    let mut ids: Vec<u64> = std::fs::read_dir(&segments_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            name.strip_prefix("seg_")?.parse::<u64>().ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn manifest_segment_ids(base_path: &std::path::Path, version_number: u64) -> Vec<u64> {
    read_manifest(base_path, version_number)
        .iter()
        .map(|s| s["segment_id"].as_u64().unwrap())
        .collect()
}

// ============================================================================
// Cleanup safety tests
// ============================================================================

#[test]
fn test_cleanup_preserves_carried_forward_segment() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 1000,
        deletion_threshold: 0.5f64.try_into().unwrap(),
        insertion_rebuild_threshold: 10.0,
    };
    let (tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // Insert 20 docs, compact → seg_1
    for i in 0..20u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    let m1_ids = manifest_segment_ids(tmp.path(), 1);
    assert_eq!(m1_ids.len(), 1);
    let seg_v1 = m1_ids[0];

    // Delete 1 doc (1/20=0.05 < 0.5), no new inserts → carry forward
    storage.delete(5);
    storage.compact(2).unwrap();
    let m2_ids = manifest_segment_ids(tmp.path(), 2);
    assert_eq!(m2_ids.len(), 1);
    assert_eq!(
        m2_ids[0], seg_v1,
        "segment should be carried forward with same id"
    );

    // cleanup: seg_v1 must survive, versions/1 removed
    storage.cleanup();
    assert!(!tmp.path().join("versions/1").exists());
    assert!(tmp.path().join("versions/2").exists());
    let on_disk = list_segment_dirs_on_disk(tmp.path());
    assert!(
        on_disk.contains(&seg_v1),
        "carried-forward segment must survive cleanup"
    );

    // Search returns 19 results
    let results = storage.search(&[0.0, 0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), 19);
}

#[test]
fn test_cleanup_multi_segment_mixed_carry_forward_and_rebuild() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 20,
        deletion_threshold: 0.1f64.try_into().unwrap(),
        insertion_rebuild_threshold: 0.3,
    };
    let (tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // Insert 20, compact(1) → seg_1
    for i in 0..20u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    let m1_ids = manifest_segment_ids(tmp.path(), 1);
    assert_eq!(m1_ids.len(), 1);
    let seg_1 = m1_ids[0];

    // Insert 20 more, compact(2) → seg_1 carried forward + new seg for overflow
    for i in 20..40u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();
    let m2_ids = manifest_segment_ids(tmp.path(), 2);
    assert_eq!(m2_ids.len(), 2);
    assert_eq!(m2_ids[0], seg_1, "seg_1 should be carried forward");
    let seg_2 = m2_ids[1];

    // Delete 5 from seg_1 range (5/20=0.25 > 0.1) → seg_1 rebuilt, seg_2 carried forward
    for &doc_id in &[0u64, 2, 4, 6, 8] {
        storage.delete(doc_id);
    }
    storage.compact(3).unwrap();
    let m3_ids = manifest_segment_ids(tmp.path(), 3);
    assert_eq!(m3_ids.len(), 2);
    assert_ne!(
        m3_ids[0], seg_1,
        "seg_1 should have been rebuilt with new id"
    );
    let seg_3 = m3_ids[0];
    assert_eq!(m3_ids[1], seg_2, "seg_2 should be carried forward");

    // cleanup → seg_1 removed, seg_2 and seg_3 survive
    storage.cleanup();
    let on_disk = list_segment_dirs_on_disk(tmp.path());
    assert!(!on_disk.contains(&seg_1), "old seg_1 should be removed");
    assert!(
        on_disk.contains(&seg_2),
        "carried-forward seg_2 must survive"
    );
    assert!(on_disk.contains(&seg_3), "rebuilt seg_3 must survive");

    // Search returns 35 results
    let results = storage.search(&[0.0, 0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), 35);
}

#[test]
fn test_cleanup_after_all_segments_deleted() {
    let (tmp, storage) = make_storage(3, DistanceMetric::L2);

    // Insert 5, compact(1) → seg_1
    for i in 0..5u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    let seg_1 = manifest_segment_ids(tmp.path(), 1)[0];

    // Delete all 5, compact(2) → empty manifest
    for i in 0..5u64 {
        storage.delete(i);
    }
    storage.compact(2).unwrap();
    let m2 = read_manifest(tmp.path(), 2);
    assert!(
        m2.is_empty(),
        "manifest should be empty after deleting all docs"
    );

    // cleanup → seg_1 removed
    storage.cleanup();
    let on_disk = list_segment_dirs_on_disk(tmp.path());
    assert!(
        !on_disk.contains(&seg_1),
        "seg_1 should be removed after all docs deleted"
    );

    // Search returns 0
    let results = storage.search(&[0.0, 0.0, 0.0], 10, None).unwrap();
    assert!(results.is_empty());

    // Insert new docs, compact, search works
    for i in 100..103u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(3).unwrap();
    let results = storage.search(&[0.0, 0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_cleanup_multiple_compactions_accumulating_old_segments() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 1_000_000,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.0, // forces full rebuild every compaction
    };
    let (tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // 4 rounds: each insert+compact produces a new segment via full rebuild
    let mut expected_total = 0u64;
    for round in 1..=4u64 {
        let base = (round - 1) * 5;
        for i in base..base + 3 + round - 1 {
            storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
            expected_total += 1;
        }
        storage.compact(round).unwrap();
    }

    // Before cleanup: multiple segment dirs on disk
    let before = list_segment_dirs_on_disk(tmp.path());
    assert!(
        before.len() >= 2,
        "should have accumulated old segment dirs"
    );

    // Get current manifest segment id
    let current_ids = manifest_segment_ids(tmp.path(), 4);
    assert_eq!(current_ids.len(), 1);
    let current_seg = current_ids[0];

    // cleanup → only current segment survives
    storage.cleanup();
    let after = list_segment_dirs_on_disk(tmp.path());
    assert_eq!(
        after,
        vec![current_seg],
        "only current segment should survive"
    );

    // Search returns all docs
    let results = storage.search(&[0.0, 0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), expected_total as usize);

    // Integrity check passes
    assert!(storage.integrity_check().passed);
}

#[test]
fn test_cleanup_preserves_search_correctness_multi_segment() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 15,
        deletion_threshold: 0.5f64.try_into().unwrap(),
        insertion_rebuild_threshold: 10.0,
    };
    let (_tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // Create segments via overflow: 15 + 15 + 5 docs
    for i in 0..15u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    for i in 15..30u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    for i in 30..35u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    // Delete a few (low ratio, should carry forward)
    storage.delete(3);
    storage.delete(18);
    storage.compact(3).unwrap();

    // Record search results before cleanup
    let results_before = storage.search(&[10.0, 0.0, 0.0], 50, Some(200)).unwrap();
    assert_eq!(results_before.len(), 33); // 35 - 2 deleted

    // Cleanup
    storage.cleanup();

    // Search results must be identical after cleanup
    let results_after = storage.search(&[10.0, 0.0, 0.0], 50, Some(200)).unwrap();
    assert_eq!(results_after.len(), results_before.len());
    for (a, b) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(a.0, b.0, "doc_id mismatch after cleanup");
        assert!(
            (a.1 - b.1).abs() < 1e-6,
            "distance mismatch after cleanup for doc {}",
            a.0
        );
    }
}

#[test]
fn test_cleanup_carried_forward_across_three_versions() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: 0.5f64.try_into().unwrap(),
        insertion_rebuild_threshold: 10.0,
    };
    let (tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // Insert 20, compact(1) → seg_1
    for i in 0..20u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    let seg_1 = manifest_segment_ids(tmp.path(), 1)[0];

    // Delete-only compactions for versions 2, 3, 4 (low ratio each time)
    // Each deletes 1 doc: 1/20=0.05, 2/20=0.10, 3/20=0.15 — all < 0.5
    storage.delete(0);
    storage.compact(2).unwrap();
    assert_eq!(manifest_segment_ids(tmp.path(), 2)[0], seg_1);

    storage.delete(1);
    storage.compact(3).unwrap();
    assert_eq!(manifest_segment_ids(tmp.path(), 3)[0], seg_1);

    storage.delete(2);
    storage.compact(4).unwrap();
    assert_eq!(
        manifest_segment_ids(tmp.path(), 4)[0],
        seg_1,
        "seg_1 should still be carried forward"
    );

    // cleanup → seg_1 survives, only versions/4 remains
    storage.cleanup();
    let on_disk = list_segment_dirs_on_disk(tmp.path());
    assert!(on_disk.contains(&seg_1), "seg_1 must survive cleanup");

    let version_dirs: Vec<_> = std::fs::read_dir(tmp.path().join("versions"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str()?.parse::<u64>().ok())
        .collect();
    assert_eq!(version_dirs, vec![4], "only version 4 should remain");

    // Search returns 17 results (20 - 3 deleted)
    let results = storage.search(&[0.0, 0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), 17);
}

#[test]
fn test_cleanup_incremental_insert_then_carry_forward() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: 0.5f64.try_into().unwrap(),
        insertion_rebuild_threshold: 0.3,
    };
    let (tmp, storage) = make_storage_with_segment_config(3, DistanceMetric::L2, seg_config);

    // Insert 50, compact(1) → seg_1
    for i in 0..50u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();
    let seg_1 = manifest_segment_ids(tmp.path(), 1)[0];

    // Insert 10, compact(2) → incremental insert → new seg_2 (10/50=0.2 < 0.3), seg_1 orphaned
    for i in 50..60u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();
    let m2_ids = manifest_segment_ids(tmp.path(), 2);
    assert_eq!(m2_ids.len(), 1);
    let seg_2 = m2_ids[0];
    assert_ne!(seg_2, seg_1, "incremental insert creates new segment id");

    // Delete 1 doc, compact(3) → seg_2 carried forward (1/60 < 0.5)
    storage.delete(0);
    storage.compact(3).unwrap();
    let m3_ids = manifest_segment_ids(tmp.path(), 3);
    assert_eq!(m3_ids.len(), 1);
    assert_eq!(m3_ids[0], seg_2, "seg_2 should be carried forward");

    // cleanup → seg_1 removed, seg_2 survives
    storage.cleanup();
    let on_disk = list_segment_dirs_on_disk(tmp.path());
    assert!(!on_disk.contains(&seg_1), "old seg_1 should be removed");
    assert!(
        on_disk.contains(&seg_2),
        "carried-forward seg_2 must survive"
    );

    // Search returns 59 results (60 - 1 deleted)
    let results = storage.search(&[0.0, 0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), 59);
}
