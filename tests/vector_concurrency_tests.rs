use oramacore_fields::vector::{
    DeletionThreshold, DistanceMetric, SegmentConfig, VectorConfig, VectorIndexer, VectorStorage,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

fn make_storage(dimensions: usize) -> (TempDir, Arc<VectorStorage>, VectorIndexer) {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(dimensions, DistanceMetric::L2).unwrap();
    let storage = Arc::new(
        VectorStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default()).unwrap(),
    );

    let indexer = VectorIndexer::new(dimensions);

    (tmp, storage, indexer)
}

fn make_storage_with_config(
    dimensions: usize,
    seg_config: SegmentConfig,
) -> (TempDir, Arc<VectorStorage>, VectorIndexer) {
    let tmp = TempDir::new().unwrap();
    let config = VectorConfig::new(dimensions, DistanceMetric::L2).unwrap();
    let storage =
        Arc::new(VectorStorage::new(tmp.path().to_path_buf(), config, seg_config).unwrap());
    let indexer = VectorIndexer::new(dimensions);
    (tmp, storage, indexer)
}

/// Helper: build an IndexedValue from a float slice.
fn iv(dims: usize, v: &[f32]) -> oramacore_fields::vector::IndexedValue {
    VectorIndexer::new(dims).index_vec(v).unwrap()
}

fn search_count(storage: &VectorStorage, query: &[f32], k: usize) -> usize {
    storage.search(query, k, Some(200)).unwrap().len()
}

// ============================================================================
// A. Concurrent Reads
// ============================================================================

#[test]
fn test_concurrent_reads_consistency() {
    let (_tmp, storage, _indexer) = make_storage(3);

    for i in 0..100u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            for _ in 0..100 {
                let results = s.search(&[50.0, 0.0, 0.0], 10, None).unwrap();
                assert_eq!(results.len(), 10);
                // Verify distances are sorted
                for w in results.windows(2) {
                    assert!(w[0].1 <= w[1].1);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_concurrent_reads_during_dirty_snapshot() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..50u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(9)); // 8 readers + 1 writer

    let mut handles = Vec::new();

    // 8 reader threads
    for _ in 0..8 {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            while !st.load(Ordering::Relaxed) {
                let results = s.search(&[0.0, 0.0], 100, Some(200)).unwrap();
                // Count should never decrease below baseline
                assert!(results.len() >= 50);
                // Results should be sorted by distance
                for w in results.windows(2) {
                    assert!(w[0].1 <= w[1].1);
                }
                thread::yield_now();
            }
        }));
    }

    // 1 writer thread
    {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            for i in 50..100u64 {
                s.insert(i, iv(2, &[i as f32, 0.0]));
                thread::yield_now();
            }
            st.store(true, Ordering::Relaxed);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ============================================================================
// B. Concurrent Writes
// ============================================================================

#[test]
fn test_concurrent_multi_thread_inserts() {
    let (_tmp, storage, _indexer) = make_storage(2);
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for t in 0..4u64 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let base = t * 100;
            for i in 0..100u64 {
                let doc_id = base + i;
                s.insert(doc_id, iv(2, &[doc_id as f32, 0.0]));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 400, Some(600)).unwrap();
    assert_eq!(results.len(), 400);
}

#[test]
fn test_concurrent_inserts_same_doc_id() {
    let (_tmp, storage, _indexer) = make_storage(2);
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for t in 0..4u64 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            // All threads insert the same doc_id with different vectors
            s.insert(1, iv(2, &[t as f32, 0.0]));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Should have exactly 1 result (last write wins)
    let results = storage.search(&[0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

// ============================================================================
// C. Reads During Compaction
// ============================================================================

#[test]
fn test_search_during_compaction() {
    let (_tmp, storage, _indexer) = make_storage(3);

    for i in 0..100u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(5)); // 4 readers + 1 compactor

    let mut handles = Vec::new();

    // 4 reader threads
    for _ in 0..4 {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            while !st.load(Ordering::Relaxed) {
                let results = s.search(&[50.0, 0.0, 0.0], 10, None).unwrap();
                // Always get some results (index is never empty)
                assert!(!results.is_empty());
                assert!(results.len() <= 10);
                thread::yield_now();
            }
        }));
    }

    // Compactor thread: inserts more and runs multiple compactions
    {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            for round in 0..5u64 {
                let base = 100 + round * 20;
                for i in 0..20u64 {
                    s.insert(base + i, iv(3, &[(base + i) as f32, 0.0, 0.0]));
                }
                s.compact(2 + round).unwrap();
                thread::yield_now();
            }
            st.store(true, Ordering::Relaxed);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Final state should have all 200 vectors
    let results = storage.search(&[0.0, 0.0, 0.0], 300, Some(400)).unwrap();
    assert_eq!(results.len(), 200);
}

#[test]
fn test_search_result_stability_across_version_swap() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..50u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    // Capture search result before compaction
    let results_before = storage.search(&[0.0, 0.0], 50, Some(200)).unwrap();
    assert_eq!(results_before.len(), 50);

    // Insert more and compact (swaps version)
    for i in 50..100u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(2).unwrap();

    // The captured results_before still valid (it's just data, not a live reference)
    assert_eq!(results_before.len(), 50);

    // New search should see all 100
    let results_after = storage.search(&[0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results_after.len(), 100);
}

// ============================================================================
// D. Writes During Compaction
// ============================================================================

#[test]
fn test_writes_preserved_during_compaction() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..50u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Writer thread
    let writer = {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            b.wait();
            for i in 50..100u64 {
                s.insert(i, iv(2, &[i as f32, 0.0]));
                thread::yield_now();
            }
        })
    };

    // Compactor in main thread
    {
        let b = Arc::clone(&barrier);
        b.wait();
        // Run compaction while writer is inserting
        storage.compact(2).unwrap();
    }

    writer.join().unwrap();

    // Final compaction to capture any remaining live data
    storage.compact(3).unwrap();

    let results = storage.search(&[0.0, 0.0], 200, Some(300)).unwrap();
    assert_eq!(results.len(), 100, "all 100 writes should be preserved");
}

#[test]
fn test_deletes_preserved_during_compaction() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..100u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Deleter thread
    let deleter = {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            b.wait();
            // Delete even-numbered docs
            for i in (0..100u64).step_by(2) {
                s.delete(i);
                thread::yield_now();
            }
        })
    };

    // Compactor in main thread
    {
        let b = Arc::clone(&barrier);
        b.wait();
        storage.compact(2).unwrap();
    }

    deleter.join().unwrap();

    // Final compaction to apply remaining deletes
    storage.compact(3).unwrap();

    let results = storage.search(&[0.0, 0.0], 200, Some(300)).unwrap();
    assert_eq!(results.len(), 50, "only odd docs should remain");
    for (id, _) in &results {
        assert!(id % 2 == 1, "even doc_id {id} should have been deleted");
    }
}

// ============================================================================
// E. Compaction Serialization
// ============================================================================

#[test]
fn test_concurrent_compaction_serialization() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..50u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }

    let barrier = Arc::new(Barrier::new(4));
    let version_counter = Arc::new(AtomicU64::new(1));
    let mut handles = Vec::new();

    for _ in 0..4 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        let vc = Arc::clone(&version_counter);
        handles.push(thread::spawn(move || {
            b.wait();
            let version = vc.fetch_add(1, Ordering::SeqCst);
            s.compact(version).unwrap();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Data should be intact after concurrent compactions
    let results = storage.search(&[0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(results.len(), 50);
}

// ============================================================================
// F. Mixed Operations Stress
// ============================================================================

#[test]
fn test_mixed_insert_delete_search() {
    let (_tmp, storage, _indexer) = make_storage(3);

    // Pre-populate
    for i in 0..50u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let insert_count = Arc::new(AtomicU64::new(0));
    let search_count_atomic = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = Vec::new();

    // 2 inserter threads
    for t in 0..2u64 {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let ic = Arc::clone(&insert_count);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let base = 1000 + t * 1000;
            let mut i = 0u64;
            while !st.load(Ordering::Relaxed) {
                s.insert(base + i, iv(3, &[(base + i) as f32, 0.0, 0.0]));
                ic.fetch_add(1, Ordering::Relaxed);
                i += 1;
                thread::yield_now();
            }
        }));
    }

    // 1 deleter thread
    {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let mut i = 0u64;
            while !st.load(Ordering::Relaxed) {
                s.delete(i % 50); // delete from pre-populated range
                i += 1;
                thread::yield_now();
            }
        }));
    }

    // 2 searcher threads
    for _ in 0..2 {
        let s = Arc::clone(&storage);
        let st = Arc::clone(&stop);
        let sc = Arc::clone(&search_count_atomic);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            while !st.load(Ordering::Relaxed) {
                let _results = s.search(&[0.0, 0.0, 0.0], 10, None).unwrap();
                sc.fetch_add(1, Ordering::Relaxed);
                thread::yield_now();
            }
        }));
    }

    // Let it run for a while
    thread::sleep(std::time::Duration::from_millis(500));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    assert!(insert_count.load(Ordering::Relaxed) > 0);
    assert!(search_count_atomic.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_high_write_contention() {
    let (_tmp, storage, _indexer) = make_storage(2);
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();

    for t in 0..8u64 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let base = t * 50;
            for i in 0..50u64 {
                s.insert(base + i, iv(2, &[(base + i) as f32, 0.0]));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // All 400 unique inserts should be present
    storage.compact(1).unwrap();

    let results = storage.search(&[0.0, 0.0], 400, Some(600)).unwrap();
    assert_eq!(results.len(), 400);
}

// ============================================================================
// G. Snapshot Isolation During Compaction
// ============================================================================

#[test]
fn test_snapshot_isolation_during_compaction() {
    let (_tmp, storage, _indexer) = make_storage(2);

    for i in 0..50u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    // Search before new inserts
    let count_before = search_count(&storage, &[0.0, 0.0], 100);
    assert_eq!(count_before, 50);

    // Insert more
    for i in 50..100u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }

    // Compact in a separate thread
    let s = Arc::clone(&storage);
    let compaction_handle = thread::spawn(move || {
        s.compact(2).unwrap();
    });

    compaction_handle.join().unwrap();

    // After compaction, all 100 should be visible
    let count_after = search_count(&storage, &[0.0, 0.0], 200);
    assert_eq!(count_after, 100);
}

// ============================================================================
// H. Multi-Segment Concurrency
// ============================================================================

#[test]
fn test_concurrent_search_multi_segment() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 20,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage, _indexer) = make_storage_with_config(3, seg_config);

    // Create 2 segments
    for i in 0..20u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(1).unwrap();

    for i in 20..40u64 {
        storage.insert(i, iv(3, &[i as f32, 0.0, 0.0]));
    }
    storage.compact(2).unwrap();

    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            for _ in 0..50 {
                let results = s.search(&[10.0, 0.0, 0.0], 10, Some(100)).unwrap();
                assert_eq!(results.len(), 10);
                // Results should be sorted by distance
                for w in results.windows(2) {
                    assert!(w[0].1 <= w[1].1);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_concurrent_writes_during_multi_segment_compaction() {
    let seg_config = SegmentConfig {
        max_nodes_per_segment: 30,
        deletion_threshold: DeletionThreshold::default(),
        insertion_rebuild_threshold: 0.3,
    };
    let (_tmp, storage, _indexer) = make_storage_with_config(2, seg_config);

    // Create initial segment with 10 vectors (small for fast HNSW builds in debug)
    for i in 0..10u64 {
        storage.insert(i, iv(2, &[i as f32, 0.0]));
    }
    storage.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Writer thread: insert a bounded number of vectors
    let writer = {
        let s = Arc::clone(&storage);
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            b.wait();
            for i in 100..115u64 {
                s.insert(i, iv(2, &[i as f32, 0.0]));
                thread::yield_now();
            }
        })
    };

    // Compactor in main thread
    {
        let b = Arc::clone(&barrier);
        b.wait();
        storage.compact(2).unwrap();
    }

    writer.join().unwrap();

    // Final compaction to capture remaining live data
    storage.compact(3).unwrap();

    // Should have all initial + writer inserts
    let results = storage.search(&[0.0, 0.0], 100, Some(200)).unwrap();
    assert_eq!(
        results.len(),
        25,
        "should have initial 10 + writer 15 = 25 vectors, got {}",
        results.len()
    );
}
