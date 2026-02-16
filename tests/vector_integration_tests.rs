use oramacore_fields::vector::{
    DeletionThreshold, DistanceMetric, SegmentConfig, VectorConfig, VectorStorage,
};
use std::sync::Arc;
use tempfile::TempDir;

fn make_storage(dimensions: usize, metric: DistanceMetric) -> (TempDir, VectorStorage) {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(dimensions, metric).unwrap();
    let storage =
        VectorStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();
    (tmp, storage)
}

fn make_storage_with_segment_config(
    dimensions: usize,
    metric: DistanceMetric,
    segment_config: SegmentConfig,
) -> (TempDir, VectorStorage) {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(dimensions, metric).unwrap();
    let storage = VectorStorage::new(tmp.path().to_path_buf(), config, segment_config).unwrap();
    (tmp, storage)
}

#[test]
fn test_insert_and_search_live_only() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0, 0.0]).unwrap();
    storage.insert(3, &[10.0, 10.0, 10.0]).unwrap();

    let results = storage.search(&[0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest
    assert_eq!(results[1].0, 2);
}

#[test]
fn test_insert_compact_search() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0, 0.0]).unwrap();
    storage.insert(3, &[10.0, 10.0, 10.0]).unwrap();

    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_insert_delete_search() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0]).unwrap();
    storage.insert(3, &[2.0, 0.0]).unwrap();

    storage.delete(1);

    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();
    assert!(results.iter().all(|(id, _)| *id != 1));
}

#[test]
fn test_compact_then_delete() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0]).unwrap();
    storage.insert(3, &[2.0, 0.0]).unwrap();

    storage.compact(1).unwrap();

    storage.delete(1);

    let results = storage.search(&[0.0, 0.0], 3, None).unwrap();
    assert!(results.iter().all(|(id, _)| *id != 1));
}

#[test]
fn test_multiple_compactions() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    // Round 1
    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0]).unwrap();
    storage.compact(1).unwrap();

    // Round 2
    storage.insert(3, &[2.0, 0.0]).unwrap();
    storage.compact(2).unwrap();

    // Round 3
    storage.insert(4, &[3.0, 0.0]).unwrap();
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
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage =
            VectorStorage::new(base_path.clone(), config, SegmentConfig::default()).unwrap();
        storage.insert(1, &[0.0, 0.0, 0.0]).unwrap();
        storage.insert(2, &[1.0, 0.0, 0.0]).unwrap();
        storage.insert(3, &[0.0, 1.0, 0.0]).unwrap();
        storage.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(base_path, config, SegmentConfig::default()).unwrap();
        assert_eq!(storage.current_version_number(), 1);

        let results = storage.search(&[0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1); // closest
    }
}

#[test]
fn test_dimension_mismatch() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::L2);

    // Insert with wrong dimensions
    assert!(storage.insert(1, &[1.0, 2.0]).is_err());

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
    storage.insert(1, &[0.0, 0.0]).unwrap();
    let results = storage.search(&[0.0, 0.0], 0, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_non_finite_value_rejected() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    assert!(storage.insert(1, &[f32::NAN, 0.0]).is_err());
    assert!(storage.insert(2, &[f32::INFINITY, 0.0]).is_err());
    assert!(storage.insert(3, &[f32::NEG_INFINITY, 0.0]).is_err());
}

#[test]
fn test_empty_vector_rejected() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);
    assert!(storage.insert(1, &[]).is_err());
}

#[test]
fn test_cosine_metric() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::Cosine);

    storage.insert(1, &[1.0, 0.0, 0.0]).unwrap();
    storage.insert(2, &[0.0, 1.0, 0.0]).unwrap();
    storage.insert(3, &[0.9, 0.1, 0.0]).unwrap();

    let results = storage.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
    assert_eq!(results[0].0, 1); // exact match is closest
}

#[test]
fn test_dot_product_metric() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::DotProduct);

    storage.insert(1, &[1.0, 0.0, 0.0]).unwrap();
    storage.insert(2, &[0.0, 1.0, 0.0]).unwrap();
    storage.insert(3, &[5.0, 0.0, 0.0]).unwrap(); // highest dot product with query

    let results = storage.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
    // For dot product distance = -dot, so highest dot product = lowest distance
    assert_eq!(results[0].0, 3);
}

#[test]
fn test_delete_and_reinsert() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.compact(1).unwrap();

    storage.delete(1);
    storage.insert(1, &[5.0, 5.0]).unwrap(); // reinsert with different vector

    let results = storage.search(&[5.0, 5.0], 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01); // should be very close to query
}

#[test]
fn test_compact_with_deletes() {
    let (_tmp, storage) = make_storage(2, DistanceMetric::L2);

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0]).unwrap();
    storage.insert(3, &[2.0, 0.0]).unwrap();
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
    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.compact(2).unwrap();
    assert_eq!(storage.current_version_number(), 2);
}

#[test]
fn test_cleanup() {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(2, DistanceMetric::L2).unwrap();
    let storage =
        VectorStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap();

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.compact(1).unwrap();

    storage.insert(2, &[1.0, 0.0]).unwrap();
    storage.compact(2).unwrap();

    storage.insert(3, &[2.0, 0.0]).unwrap();
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

    storage.insert(1, &[0.0, 0.0]).unwrap();
    storage.compact(1).unwrap();

    let result = storage.integrity_check();
    assert!(result.passed);
}

#[test]
fn test_info() {
    let (_tmp, storage) = make_storage(3, DistanceMetric::Cosine);

    storage.insert(1, &[0.0, 0.0, 0.0]).unwrap();
    storage.insert(2, &[1.0, 0.0, 0.0]).unwrap();
    storage.compact(1).unwrap();

    let info = storage.info();
    assert_eq!(info.num_vectors, 2);
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
        storage.insert(i, &vec).unwrap();
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
    let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
    let storage = Arc::new(
        VectorStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap(),
    );

    for i in 0..100u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    storage.compact(1).unwrap();

    // Insert 10 more and compact — should create second segment since first is full
    for i in 15..25u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    storage.compact(1).unwrap();

    // Round 2: insert 10 more (20% of 50 = below 30% threshold => incremental insert)
    for i in 50..60u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    storage.compact(1).unwrap();

    // Round 2: incremental insert 10 (20% < 30%)
    for i in 50..60u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    storage.compact(2).unwrap();

    // Round 3: insert 8 more. insertions_since_rebuild=10+8=18, nodes_at_last_rebuild=50.
    // ratio = 18/50 = 0.36 > 0.3 -> FULL REBUILD
    for i in 60..68u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(base_path.clone(), config, seg_config.clone()).unwrap();

        // Fill first segment
        for i in 0..12u64 {
            storage.insert(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        storage.compact(1).unwrap();

        // Create second segment
        for i in 12..24u64 {
            storage.insert(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        storage.compact(2).unwrap();
    }

    // Reopen and verify
    {
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(base_path, config, seg_config).unwrap();
        assert_eq!(storage.current_version_number(), 2);

        let results = storage.search(&[0.0, 0.0, 0.0], 24, Some(200)).unwrap();
        assert_eq!(results.len(), 24);
    }
}

#[test]
fn test_cleanup_removes_old_segments() {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 100,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.0, // Always full rebuild for predictability
    };
    let storage = VectorStorage::new(tmp.path().to_path_buf(), config, seg_config).unwrap();

    // Round 1: creates seg_1
    for i in 0..5u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0]).unwrap();
    }
    storage.compact(1).unwrap();
    assert!(tmp.path().join("segments/seg_1").exists());

    // Round 2: full rebuild creates seg_2 (since insertion_rebuild_threshold=0.0)
    for i in 5..8u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    storage.compact(2).unwrap();

    let m2 = read_manifest(_tmp.path(), 2);
    assert_eq!(m2.len(), 1);
    assert_eq!(m2[0]["num_nodes"].as_u64().unwrap(), 60);
    assert_eq!(m2[0]["insertions_since_rebuild"].as_u64().unwrap(), 10);
    assert_eq!(m2[0]["nodes_at_last_rebuild"].as_u64().unwrap(), 50);
    // Segment id should be different (copy-on-modify)
    let seg2_id = m2[0]["segment_id"].as_u64().unwrap();
    let seg1_id = m1[0]["segment_id"].as_u64().unwrap();
    assert_ne!(seg2_id, seg1_id);

    // Round 3: Insert 8 vectors (doc_ids 60-67), compact
    // insertion_ratio = (10+8)/50 = 0.36 > 0.3 → FULL REBUILD
    for i in 60..68u64 {
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
        storage.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
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
