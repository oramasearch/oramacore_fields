use oramacore_fields::vector::{DistanceMetric, VectorConfig, VectorStorage};
use std::sync::Arc;
use tempfile::TempDir;

fn make_storage(dimensions: usize, metric: DistanceMetric) -> (TempDir, VectorStorage) {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(dimensions, metric).unwrap();
    let storage = VectorStorage::new(tmp.path().to_path_buf(), config).unwrap();
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
        let storage = VectorStorage::new(base_path.clone(), config).unwrap();
        storage.insert(1, &[0.0, 0.0, 0.0]).unwrap();
        storage.insert(2, &[1.0, 0.0, 0.0]).unwrap();
        storage.insert(3, &[0.0, 1.0, 0.0]).unwrap();
        storage.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(base_path, config).unwrap();
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
    let storage = VectorStorage::new(tmp.path().to_path_buf(), config).unwrap();

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
    let storage = Arc::new(VectorStorage::new(tmp.path().to_path_buf(), config).unwrap());

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
