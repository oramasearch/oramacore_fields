use oramacore_fields::geopoint::{GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold};
use std::sync::Arc;
use tempfile::TempDir;

fn make_index() -> (TempDir, GeoPointStorage) {
    let tmp = TempDir::new().unwrap();
    let index = GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 10).unwrap();
    (tmp, index)
}

// --- Basic CRUD ---

#[test]
fn test_insert_and_query_bbox() {
    let (_tmp, index) = make_index();

    let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
    let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
    let london = GeoPoint::new(51.5074, -0.1278).unwrap();

    index.insert(IndexedValue::Plain(rome), 1);
    index.insert(IndexedValue::Plain(paris), 2);
    index.insert(IndexedValue::Plain(london), 3);

    let op = GeoFilterOp::BoundingBox {
        min_lat: 40.0,
        max_lat: 55.0,
        min_lon: -5.0,
        max_lon: 15.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2, 3]);
}

#[test]
fn test_insert_and_query_radius() {
    let (_tmp, index) = make_index();

    let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
    let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
    let london = GeoPoint::new(51.5074, -0.1278).unwrap();

    index.insert(IndexedValue::Plain(rome), 1);
    index.insert(IndexedValue::Plain(paris), 2);
    index.insert(IndexedValue::Plain(london), 3);

    // 500 km from Rome - only Rome
    let op = GeoFilterOp::Radius {
        center: rome,
        radius_meters: 500_000.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![1]);
}

#[test]
fn test_delete_removes_from_results() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(15.0, 25.0).unwrap()), 2);
    index.insert(IndexedValue::Plain(GeoPoint::new(20.0, 30.0).unwrap()), 3);

    index.delete(2);

    let op = GeoFilterOp::BoundingBox {
        min_lat: 0.0,
        max_lat: 30.0,
        min_lon: 0.0,
        max_lon: 40.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_delete_nonexistent_is_noop() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.delete(999); // doesn't exist

    let op = GeoFilterOp::BoundingBox {
        min_lat: 0.0,
        max_lat: 30.0,
        min_lon: 0.0,
        max_lon: 30.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![1]);
}

// --- Compaction ---

#[test]
fn test_compact_preserves_data() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);

    index.compact(1).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2, 3]);
}

#[test]
fn test_compact_applies_deletes() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
    index.delete(2);

    index.compact(1).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_multiple_compaction_rounds() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.compact(1).unwrap();

    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.compact(2).unwrap();

    index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
    index.compact(3).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2, 3]);
}

#[test]
fn test_compact_delete_after_compact() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.compact(1).unwrap();

    // Delete after compaction
    index.delete(1);

    let results: Vec<u64> = index
        .filter(GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        })
        .iter()
        .collect();
    assert_eq!(results, vec![2]);

    // Compact again to persist the delete
    index.compact(2).unwrap();

    let results: Vec<u64> = index
        .filter(GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        })
        .iter()
        .collect();
    assert_eq!(results, vec![2]);
}

// --- Persistence ---

#[test]
fn test_persistence_survives_close_reopen() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();

    {
        let index = GeoPointStorage::new(base_path.clone(), Threshold::default(), 10).unwrap();
        index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
        index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
        index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
        index.compact(1).unwrap();
    }

    {
        let index = GeoPointStorage::new(base_path, Threshold::default(), 10).unwrap();
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
        assert_eq!(index.current_version_id(), 1);
    }
}

// --- Edge cases ---

#[test]
fn test_doc_id_zero() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 0);

    let op = GeoFilterOp::BoundingBox {
        min_lat: 0.0,
        max_lat: 30.0,
        min_lon: 0.0,
        max_lon: 30.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![0]);

    index.compact(1).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: 0.0,
        max_lat: 30.0,
        min_lon: 0.0,
        max_lon: 30.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![0]);
}

#[test]
fn test_poles() {
    let (_tmp, index) = make_index();

    let north_pole = GeoPoint::new(90.0, 0.0).unwrap();
    let south_pole = GeoPoint::new(-90.0, 0.0).unwrap();

    index.insert(IndexedValue::Plain(north_pole), 1);
    index.insert(IndexedValue::Plain(south_pole), 2);

    let op = GeoFilterOp::BoundingBox {
        min_lat: 85.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![1]);

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: -85.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![2]);
}

#[test]
fn test_empty_index_query() {
    let (_tmp, index) = make_index();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_single_point() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 42);

    // Point query (zero-area bbox)
    let op = GeoFilterOp::BoundingBox {
        min_lat: 10.0,
        max_lat: 10.0,
        min_lon: 20.0,
        max_lon: 20.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![42]);
}

#[test]
fn test_geopoint_validation() {
    assert!(GeoPoint::new(91.0, 0.0).is_err());
    assert!(GeoPoint::new(-91.0, 0.0).is_err());
    assert!(GeoPoint::new(0.0, 181.0).is_err());
    assert!(GeoPoint::new(0.0, -181.0).is_err());
    assert!(GeoPoint::new(f64::NAN, 0.0).is_err());
    assert!(GeoPoint::new(0.0, f64::NAN).is_err());
    assert!(GeoPoint::new(f64::INFINITY, 0.0).is_err());
}

// --- Concurrency ---

#[test]
fn test_concurrent_reads_writes() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 10).unwrap());

    // Insert initial data
    for i in 0..100 {
        let lat = (i as f64 * 1.0) - 50.0;
        let lon = (i as f64 * 1.0) - 50.0;
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
    }

    let stop = Arc::new(AtomicBool::new(false));

    // Reader thread
    let idx_read = Arc::clone(&index);
    let stop_read = Arc::clone(&stop);
    let reader = thread::spawn(move || {
        while !stop_read.load(Ordering::Relaxed) {
            let op = GeoFilterOp::BoundingBox {
                min_lat: -90.0,
                max_lat: 90.0,
                min_lon: -180.0,
                max_lon: 180.0,
            };
            let results: Vec<u64> = idx_read.filter(op).iter().collect();
            // Should always see at least the initial 100
            assert!(
                results.len() >= 100,
                "Expected at least 100, got {}",
                results.len()
            );
        }
    });

    // Writer thread
    let idx_write = Arc::clone(&index);
    let stop_write = Arc::clone(&stop);
    let writer = thread::spawn(move || {
        let mut i = 100u64;
        while !stop_write.load(Ordering::Relaxed) && i < 200 {
            let lat = ((i % 180) as f64) - 90.0;
            let lon = ((i % 360) as f64) - 180.0;
            idx_write.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
            i += 1;
        }
    });

    // Compact while reads/writes happen
    index.compact(1).unwrap();

    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();
    writer.join().unwrap();
}

#[test]
fn test_iterator_stability_across_compaction() {
    let (_tmp, index) = make_index();

    for i in 0..50 {
        let lat = (i as f64 * 1.0) - 25.0;
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, 0.0).unwrap()), i);
    }

    // Get an iterator
    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let filter = index.filter(op);

    // Compact while holding the filter
    index.compact(1).unwrap();

    // Iterator should still work and see the original data
    let results: Vec<u64> = filter.iter().collect();
    assert_eq!(results.len(), 50);
}

// --- Stress test ---

#[test]
fn test_stress_many_points() {
    use rand::Rng;

    let (_tmp, index) = make_index();
    let mut rng = rand::thread_rng();

    // Insert 10K random points
    for i in 0..10_000u64 {
        let lat = rng.gen_range(-90.0..=90.0);
        let lon = rng.gen_range(-180.0..=180.0);
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
    }

    let make_op = || GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };

    // Query all
    let results: Vec<u64> = index.filter(make_op()).iter().collect();
    assert_eq!(results.len(), 10_000);

    // Compact and query again
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(make_op()).iter().collect();
    assert_eq!(results.len(), 10_000);

    // Delete some and compact
    for i in 0..1000u64 {
        index.delete(i);
    }
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(make_op()).iter().collect();
    assert_eq!(results.len(), 9_000);
}

#[test]
fn test_stress_compact_with_many_points() {
    use rand::Rng;

    let (_tmp, index) = make_index();
    let mut rng = rand::thread_rng();

    // Insert 5K points, compact, insert 5K more, compact
    for i in 0..5_000u64 {
        let lat = rng.gen_range(-90.0..=90.0);
        let lon = rng.gen_range(-180.0..=180.0);
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
    }
    index.compact(1).unwrap();

    for i in 5_000..10_000u64 {
        let lat = rng.gen_range(-90.0..=90.0);
        let lon = rng.gen_range(-180.0..=180.0);
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
    }
    index.compact(2).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results.len(), 10_000);
}

#[test]
fn test_bbox_partial_overlap() {
    let (_tmp, index) = make_index();

    // Points at specific locations
    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 10.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(20.0, 20.0).unwrap()), 2);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 30.0).unwrap()), 3);
    index.insert(IndexedValue::Plain(GeoPoint::new(40.0, 40.0).unwrap()), 4);
    index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 50.0).unwrap()), 5);

    // Query partial range
    let op = GeoFilterOp::BoundingBox {
        min_lat: 15.0,
        max_lat: 35.0,
        min_lon: 15.0,
        max_lon: 35.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![2, 3]);
}

#[test]
fn test_bbox_no_results() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 10.0).unwrap()), 1);

    let op = GeoFilterOp::BoundingBox {
        min_lat: 50.0,
        max_lat: 60.0,
        min_lon: 50.0,
        max_lon: 60.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_radius_after_compaction() {
    let (_tmp, index) = make_index();

    let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
    let paris = GeoPoint::new(48.8566, 2.3522).unwrap();

    index.insert(IndexedValue::Plain(rome), 1);
    index.insert(IndexedValue::Plain(paris), 2);

    index.compact(1).unwrap();

    // 1200 km from Rome - should get both Rome and Paris
    let op = GeoFilterOp::Radius {
        center: rome,
        radius_meters: 1_200_000.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn test_cleanup_preserves_current() {
    let tmp = TempDir::new().unwrap();
    let index = GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 10).unwrap();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.compact(1).unwrap();
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.compact(2).unwrap();

    index.cleanup();

    // Current version should still work
    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn test_same_doc_id_multiple_points_live_only() {
    let (_tmp, index) = make_index();

    // Insert same doc_id with two different geo-points
    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 1);

    // Global query: doc_id=1 appears once per matching point (duplicates allowed)
    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|&id| id == 1));

    // Query covering only the first point
    let op = GeoFilterOp::BoundingBox {
        min_lat: 5.0,
        max_lat: 15.0,
        min_lon: 15.0,
        max_lon: 25.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![1]);

    // Query covering only the second point
    let op = GeoFilterOp::BoundingBox {
        min_lat: 25.0,
        max_lat: 35.0,
        min_lon: 35.0,
        max_lon: 45.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert_eq!(results, vec![1]);

    // Query covering neither point
    let op = GeoFilterOp::BoundingBox {
        min_lat: 60.0,
        max_lat: 70.0,
        min_lon: 60.0,
        max_lon: 70.0,
    };
    let results: Vec<u64> = index.filter(op).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_same_doc_id_with_other_docs() {
    let (_tmp, index) = make_index();

    // Insert several documents, some with multiple points
    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
    index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
    index.compact(1).unwrap();

    // Add another point for doc_id=2 (multi-point: both points kept)
    index.insert(IndexedValue::Plain(GeoPoint::new(15.0, 25.0).unwrap()), 2);
    index.compact(2).unwrap();

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2, 2, 3]);
}

#[test]
fn test_whole_globe_bbox() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(0.0, 0.0).unwrap()), 1);
    index.insert(IndexedValue::Plain(GeoPoint::new(89.0, 179.0).unwrap()), 2);
    index.insert(
        IndexedValue::Plain(GeoPoint::new(-89.0, -179.0).unwrap()),
        3,
    );

    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let mut results: Vec<u64> = index.filter(op).iter().collect();
    results.sort_unstable();
    assert_eq!(results, vec![1, 2, 3]);
}
