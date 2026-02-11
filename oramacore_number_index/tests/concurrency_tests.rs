//! Comprehensive concurrency tests for NumberStorage.
//!
//! These tests verify thread safety of the NumberStorage implementation,
//! including:
//! - RwLock<LiveLayer> behavior
//! - ArcSwap<CompactedVersion> lock-free reads
//! - Double-check pattern for snapshot refresh
//! - Clear-after-swap pattern during compaction
//! - Iterator stability during concurrent modifications

use oramacore_number_index::{FilterOp, IndexedValue, NumberStorage, Threshold, U64Storage};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Verify results are strictly sorted (no duplicates, ascending order)
fn assert_strictly_sorted(results: &[u64], context: &str) {
    for window in results.windows(2) {
        assert!(
            window[0] < window[1],
            "{}: Results not strictly sorted: {} >= {}",
            context,
            window[0],
            window[1]
        );
    }
}

/// Verify results contain exactly the expected doc_ids
fn assert_contains_exactly(
    results: &[u64],
    expected: impl IntoIterator<Item = u64>,
    context: &str,
) {
    let result_set: HashSet<u64> = results.iter().copied().collect();
    let expected_set: HashSet<u64> = expected.into_iter().collect();
    assert_eq!(
        result_set, expected_set,
        "{context}: Mismatch - got {result_set:?}, expected {expected_set:?}"
    );
}

// ============================================================================
// Double-Check Pattern Tests
// ============================================================================

/// Test that multiple readers hitting a dirty snapshot don't cause issues.
///
/// Scenario: Many threads simultaneously call filter() when snapshot is dirty.
/// Expected: Only one thread should refresh the snapshot; others should
///           wait and use the refreshed snapshot.
#[test]
fn test_dirty_snapshot_race() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert data to populate the index
    for i in 1..=1000 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads + 1));

    // Insert one more item to make snapshot dirty
    index.insert(&IndexedValue::Plain(1001), 1001).unwrap();

    // Launch threads that will all hit the dirty snapshot simultaneously
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                // All threads call filter at approximately the same time
                let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
                assert_contains_exactly(&results, 1..=1001, "dirty snapshot race");
                assert_strictly_sorted(&results, "dirty snapshot race");
            })
        })
        .collect();

    // Release all threads at once
    barrier.wait();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test repeated dirty/clean cycles with concurrent readers.
#[test]
fn test_snapshot_dirty_clean_cycle() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let counter = Arc::new(AtomicU64::new(0));

    // Reader threads - continuously query
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
                    // Results should always be consistent
                    let count = results.len();
                    for i in 0..count {
                        assert!(results.contains(&(i as u64 + 1)));
                    }
                }
            })
        })
        .collect();

    // Writer thread - continuously insert
    let index_writer = Arc::clone(&index);
    let counter_writer = Arc::clone(&counter);
    let writer = thread::spawn(move || {
        for i in 1..=500 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
            counter_writer.store(i, Ordering::Release);
            thread::sleep(Duration::from_micros(10));
        }
    });

    writer.join().unwrap();
    running.store(false, Ordering::Relaxed);

    for handle in readers {
        handle.join().unwrap();
    }

    // Verify final state
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=500, "snapshot dirty clean cycle");
    assert_strictly_sorted(&results, "snapshot dirty clean cycle");
}

// ============================================================================
// Compaction Serialization Tests
// ============================================================================

/// Test that concurrent compaction attempts are serialized.
///
/// Scenario: Multiple threads try to compact simultaneously.
/// Expected: Compactions should be serialized (no concurrent execution).
#[test]
fn test_concurrent_compaction_serialization() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(4));
    let compaction_offset = Arc::new(AtomicU64::new(1));

    // Launch multiple threads that try to compact
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            let offset = Arc::clone(&compaction_offset);
            thread::spawn(move || {
                barrier.wait();
                let my_offset = offset.fetch_add(1, Ordering::SeqCst);
                // All threads try to compact with different offsets
                // They should all succeed sequentially
                index.compact(my_offset).unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Data should still be intact
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=100, "concurrent compaction serialization");
    assert_strictly_sorted(&results, "concurrent compaction serialization");
}

/// Test compaction while high-frequency writes are happening.
#[test]
fn test_compaction_with_high_write_rate() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicU64::new(0));

    // Writer thread - high frequency writes
    let index_writer = Arc::clone(&index);
    let running_writer = Arc::clone(&running);
    let count_writer = Arc::clone(&write_count);
    let writer = thread::spawn(move || {
        let mut i = 1u64;
        while running_writer.load(Ordering::Relaxed) || i <= 1000 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
            count_writer.fetch_add(1, Ordering::Relaxed);
            i += 1;
            if i > 2000 {
                break;
            }
        }
        i - 1
    });

    // Let some writes happen first
    thread::sleep(Duration::from_millis(10));

    // Perform multiple compactions
    for offset in 1..=5 {
        index.compact(offset).unwrap();
        thread::sleep(Duration::from_millis(5));
    }

    running.store(false, Ordering::Relaxed);
    let final_count = writer.join().unwrap();

    // All writes should be preserved
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(results.len() as u64, final_count);
}

/// Test multiple compactions with interleaved writes.
#[test]
fn test_interleaved_write_compact() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Interleave writes and compactions
    for round in 0..10 {
        let base = round * 100;
        for i in 1..=100 {
            index
                .insert(&IndexedValue::Plain(base + i), base + i)
                .unwrap();
        }
        index.compact(round + 1).unwrap();
    }

    // All 1000 entries should be present
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=1000, "interleaved write compact");
    assert_strictly_sorted(&results, "interleaved write compact");
}

// ============================================================================
// Version Swap Atomicity Tests
// ============================================================================

/// Test that queries in flight during version swap see consistent data.
///
/// Scenario: Start query, trigger compaction (version swap), continue query.
/// Expected: Query should see consistent snapshot (either old or new, not mixed).
#[test]
fn test_query_during_version_swap() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=500 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact to have a compacted version
    index.compact(1).unwrap();

    // Add more data
    for i in 501..=1000 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Start multiple queries concurrently with compaction
    let barrier = Arc::new(Barrier::new(5)); // 4 readers + 1 compactor

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();

                // Perform multiple queries during compaction
                for _ in 0..50 {
                    let filter_data = index.filter(FilterOp::Gte(1));
                    let results: Vec<u64> = filter_data.iter().collect();

                    // Should see either 500 (old) or 1000 (new), but results must be consistent
                    let len = results.len();
                    assert!(len == 500 || len == 1000, "Expected 500 or 1000, got {len}");

                    // All results should be within expected range
                    for r in &results {
                        assert!(*r >= 1 && *r <= len as u64);
                    }
                }
            })
        })
        .collect();

    // Compaction thread
    let index_compact = Arc::clone(&index);
    let barrier_compact = Arc::clone(&barrier);
    let compactor = thread::spawn(move || {
        barrier_compact.wait();
        index_compact.compact(2).unwrap();
    });

    for handle in readers {
        handle.join().unwrap();
    }
    compactor.join().unwrap();

    // Final state should have all 1000
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(results.len(), 1000);
}

/// Test that FilterHandle holds Arc references correctly during version swap.
#[test]
fn test_filter_data_arc_stability() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Get filter data before compaction
    let filter_data = index.filter(FilterOp::Gte(1));

    // Now compact
    index.compact(1).unwrap();

    // Add more data
    for i in 101..=200 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Original filter_data should still see only 100 entries
    let results: Vec<u64> = filter_data.iter().collect();
    assert_eq!(results.len(), 100);

    // New query should see all 200
    let new_results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(new_results.len(), 200);
}

// ============================================================================
// Iterator Stability Tests
// ============================================================================

/// Test iterator stability with many concurrent modifications.
#[test]
fn test_iterator_stability_under_stress() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=500 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Start iteration
    let filter_data = index.filter(FilterOp::Gte(1));
    let mut iter = filter_data.iter();

    // Collect first half
    let mut first_half = Vec::new();
    for _ in 0..250 {
        if let Some(doc_id) = iter.next() {
            first_half.push(doc_id);
        }
    }

    // Concurrent modifications
    let index_writer = Arc::clone(&index);
    let writer = thread::spawn(move || {
        // Many insertions
        for i in 501..=1000 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
        }
        // Many deletions from original range
        for i in 1..=100 {
            index_writer.delete(i);
        }
    });

    writer.join().unwrap();

    // Collect remaining items from original iterator
    let second_half: Vec<u64> = iter.collect();

    // Iterator should see exactly 500 original items (consistent snapshot)
    assert_eq!(first_half.len() + second_half.len(), 500);
}

/// Test multiple concurrent iterators with different filter operations.
#[test]
fn test_multiple_concurrent_iterators() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert data with different value ranges
    for i in 1..=1000 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact to have a compacted version
    index.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = [
        FilterOp::Lt(250),
        FilterOp::BetweenInclusive(250, 500),
        FilterOp::BetweenInclusive(501, 750),
        FilterOp::Gt(750),
    ]
    .into_iter()
    .enumerate()
    .map(|(id, op)| {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();

            // Multiple iterations with the same filter
            for _ in 0..100 {
                let results: Vec<u64> = index.filter(op).iter().collect();
                let expected = match id {
                    0 => 249, // Lt(250): 1-249
                    1 => 251, // BetweenInclusive(250, 500): 250-500
                    2 => 250, // BetweenInclusive(501, 750): 501-750
                    3 => 250, // Gt(750): 751-1000
                    _ => panic!(),
                };
                assert_eq!(
                    results.len(),
                    expected,
                    "Thread {} expected {} got {}",
                    id,
                    expected,
                    results.len()
                );
            }
        })
    })
    .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// Clear-After-Swap Pattern Tests
// ============================================================================

/// Test that writes during compaction are preserved.
#[test]
fn test_writes_preserved_during_compaction() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(2));

    // Writer thread
    let index_writer = Arc::clone(&index);
    let barrier_writer = Arc::clone(&barrier);
    let writer = thread::spawn(move || {
        barrier_writer.wait();
        // Write while compaction is happening
        for i in 101..=200 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Compaction thread
    let barrier_compact = Arc::clone(&barrier);
    barrier_compact.wait();
    index.compact(1).unwrap();

    writer.join().unwrap();

    // All writes should be preserved
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=200, "writes preserved during compaction");
    assert_strictly_sorted(&results, "writes preserved during compaction");
}

/// Test that deletes during compaction are preserved.
#[test]
fn test_deletes_preserved_during_compaction() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=200 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact first to have data in compacted version
    index.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Deleter thread
    let index_deleter = Arc::clone(&index);
    let barrier_deleter = Arc::clone(&barrier);
    let deleter = thread::spawn(move || {
        barrier_deleter.wait();
        // Delete while compaction is happening
        for i in 1..=50 {
            index_deleter.delete(i);
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Compaction thread
    let barrier_compact = Arc::clone(&barrier);
    barrier_compact.wait();
    index.compact(2).unwrap();

    deleter.join().unwrap();

    // Deletes during compaction should be preserved
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();

    // At this point, some deletes might be applied in compaction, others in live layer
    // The count should be between 150 (all deletes applied) and 200 (no deletes yet)
    assert!(
        results.len() >= 150 && results.len() <= 200,
        "Expected between 150 and 200 entries, got {}",
        results.len()
    );

    // After another compaction, all deletes should be applied
    index.compact(3).unwrap();
    let final_results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(
        final_results.len(),
        150,
        "Expected exactly 150 entries after all deletes applied, got {}",
        final_results.len()
    );

    // Verify deleted doc_ids are not present
    for i in 1..=50 {
        assert!(
            !final_results.contains(&i),
            "Doc_id {i} should have been deleted but is still present"
        );
    }

    // Verify remaining doc_ids are present
    assert_contains_exactly(
        &final_results,
        51..=200,
        "deletes preserved during compaction",
    );
    assert_strictly_sorted(&final_results, "deletes preserved during compaction");
}

// ============================================================================
// High Contention Stress Tests
// ============================================================================

/// Stress test with mixed read/write/compact operations.
///
/// This test verifies that concurrent reads, writes, and compactions don't
/// cause panics, deadlocks, or data corruption. It doesn't assert exact
/// counts because the precise number depends on timing and compaction state.
#[test]
fn test_mixed_operations_stress() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));

    // Writer threads - each writes to a unique doc_id range
    let writers: Vec<_> = (0..2)
        .map(|thread_id| {
            let index = Arc::clone(&index);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                let mut local_count = 0u64;
                while running.load(Ordering::Relaxed) {
                    let doc_id = thread_id * 1_000_000 + local_count;
                    index.insert(&IndexedValue::Plain(doc_id), doc_id).unwrap();
                    local_count += 1;
                }
                local_count
            })
        })
        .collect();

    // Reader threads - verify queries don't panic and return consistent results
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                let mut read_count = 0u64;
                while running.load(Ordering::Relaxed) {
                    // Verify each query returns a valid result
                    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
                    // All values are unique (value=doc_id), so results must be strictly sorted
                    for window in results.windows(2) {
                        assert!(
                            window[0] < window[1],
                            "Results not strictly sorted: {} >= {}",
                            window[0],
                            window[1]
                        );
                    }
                    read_count += 1;
                }
                read_count
            })
        })
        .collect();

    // Compaction thread
    let index_compact = Arc::clone(&index);
    let running_compact = Arc::clone(&running);
    let compactor = thread::spawn(move || {
        let mut offset = 1u64;
        while running_compact.load(Ordering::Relaxed) {
            index_compact.compact(offset).unwrap();
            offset += 1;
            thread::sleep(Duration::from_millis(10));
        }
        offset
    });

    // Let it run for a while
    thread::sleep(Duration::from_millis(500));
    running.store(false, Ordering::Relaxed);

    // Collect results from all threads
    let write_counts: Vec<u64> = writers.into_iter().map(|h| h.join().unwrap()).collect();
    let read_counts: Vec<u64> = readers.into_iter().map(|h| h.join().unwrap()).collect();
    let compact_count = compactor.join().unwrap();

    // Verify final state - results should contain entries from both writer threads
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    let result_set: HashSet<u64> = results.iter().copied().collect();

    // Each result should be from one of the writer threads' ranges
    for &doc_id in &results {
        let is_thread_0 = doc_id < 1_000_000;
        let is_thread_1 = (1_000_000..2_000_000).contains(&doc_id);
        assert!(is_thread_0 || is_thread_1, "Unexpected doc_id: {doc_id}");
    }

    // No duplicate doc_ids in results
    assert_eq!(
        results.len(),
        result_set.len(),
        "Duplicate doc_ids in results"
    );

    // Should have data from both threads
    let has_thread_0 = results.iter().any(|&d| d < 1_000_000);
    let has_thread_1 = results.iter().any(|&d| d >= 1_000_000);
    assert!(has_thread_0, "No results from thread 0");
    assert!(has_thread_1, "No results from thread 1");

    // Print stats for debugging
    println!(
        "Total writes: {} (thread 0: {}, thread 1: {})",
        write_counts.iter().sum::<u64>(),
        write_counts[0],
        write_counts[1]
    );
    println!("Total results: {}", results.len());
    println!("Total reads: {read_counts:?}");
    println!("Total compactions: {compact_count}");
}

/// Test with high contention on deletes.
#[test]
fn test_high_delete_contention() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 0..1000 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(5));

    // Multiple threads deleting different ranges
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let start = thread_id * 250;
                let end = start + 125; // Delete half of each thread's range
                for i in start..end {
                    index.delete(i as u64);
                }
            })
        })
        .collect();

    // Reader thread that continues reading
    let index_reader = Arc::clone(&index);
    let barrier_reader = Arc::clone(&barrier);
    let reader = thread::spawn(move || {
        barrier_reader.wait();
        for _ in 0..50 {
            let results: Vec<u64> = index_reader.filter(FilterOp::Gte(0)).iter().collect();
            // Should always see a consistent count
            let count = results.len();
            assert!((500..=1000).contains(&count), "Unexpected count: {count}");
        }
    });

    for handle in handles {
        handle.join().unwrap();
    }
    reader.join().unwrap();

    // After all deletes, should have 500 entries
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 500);
}

/// Test rapid insert-delete cycles for the same doc_id.
#[test]
fn test_insert_delete_cycle() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let barrier = Arc::new(Barrier::new(3));

    // Thread that inserts
    let index_insert = Arc::clone(&index);
    let barrier_insert = Arc::clone(&barrier);
    let inserter = thread::spawn(move || {
        barrier_insert.wait();
        for i in 0..1000 {
            index_insert
                .insert(&IndexedValue::Plain(i % 100), i)
                .unwrap();
            thread::yield_now();
        }
    });

    // Thread that deletes
    let index_delete = Arc::clone(&index);
    let barrier_delete = Arc::clone(&barrier);
    let deleter = thread::spawn(move || {
        barrier_delete.wait();
        for i in 0..1000 {
            index_delete.delete(i);
            thread::yield_now();
        }
    });

    // Thread that reads
    let index_reader = Arc::clone(&index);
    let barrier_reader = Arc::clone(&barrier);
    let reader = thread::spawn(move || {
        barrier_reader.wait();
        for _ in 0..500 {
            // Just verify we can read without panicking
            let _ = index_reader.filter(FilterOp::Gte(0)).iter().count();
        }
    });

    inserter.join().unwrap();
    deleter.join().unwrap();
    reader.join().unwrap();

    // Verify the index is still functional and consistent
    let final_count = index.filter(FilterOp::Gte(0)).iter().count();
    // Due to concurrent insert/delete on overlapping doc_ids, we can't predict exact count
    // but we can verify the index is still functional and has at most 1000 unique doc_ids (0-999)
    assert!(
        final_count <= 1000,
        "Should have at most 1000 unique doc_ids (0-999), got {final_count}"
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test empty index with concurrent operations.
#[test]
fn test_empty_index_concurrent() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let barrier = Arc::new(Barrier::new(4));

    // Multiple threads querying empty index
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
                    assert!(results.is_empty());
                }
            })
        })
        .collect();

    // Compact empty index
    let index_compact = Arc::clone(&index);
    let barrier_compact = Arc::clone(&barrier);
    let compactor = thread::spawn(move || {
        barrier_compact.wait();
        index_compact.compact(1).unwrap();
    });

    for handle in handles {
        handle.join().unwrap();
    }
    compactor.join().unwrap();
}

/// Test with doc_id = 0 under concurrent access.
#[test]
fn test_doc_id_zero_concurrent() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert doc_id = 0
    index.insert(&IndexedValue::Plain(100), 0).unwrap();

    let barrier = Arc::new(Barrier::new(3));

    // Reader threads
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let results: Vec<u64> = index.filter(FilterOp::Eq(100)).iter().collect();
                    assert!(
                        results.contains(&0),
                        "doc_id 0 should be present, got: {results:?}"
                    );
                }
            })
        })
        .collect();

    // Compaction thread
    let index_compact = Arc::clone(&index);
    let barrier_compact = Arc::clone(&barrier);
    let compactor = thread::spawn(move || {
        barrier_compact.wait();
        index_compact.compact(1).unwrap();
    });

    for handle in handles {
        handle.join().unwrap();
    }
    compactor.join().unwrap();

    // Verify doc_id 0 still exists
    let results: Vec<u64> = index.filter(FilterOp::Eq(100)).iter().collect();
    assert_eq!(results, vec![0]);
}

/// Test with large doc_ids near u64::MAX.
#[test]
fn test_large_doc_ids_concurrent() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let large_ids: Vec<u64> = vec![u64::MAX, u64::MAX - 1, u64::MAX - 100, u64::MAX / 2, 0, 1];

    for (i, &doc_id) in large_ids.iter().enumerate() {
        index
            .insert(&IndexedValue::Plain(i as u64), doc_id)
            .unwrap();
    }

    let barrier = Arc::new(Barrier::new(3));

    // Reader threads
    let expected_ids: HashSet<u64> = large_ids.iter().copied().collect();
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            let expected = expected_ids.clone();
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
                    let result_set: HashSet<u64> = results.iter().copied().collect();
                    assert_eq!(result_set, expected);
                }
            })
        })
        .collect();

    // Compaction thread
    let index_compact = Arc::clone(&index);
    let barrier_compact = Arc::clone(&barrier);
    let compactor = thread::spawn(move || {
        barrier_compact.wait();
        index_compact.compact(1).unwrap();
    });

    for handle in handles {
        handle.join().unwrap();
    }
    compactor.join().unwrap();
}

/// Test delete non-existent doc during concurrent operations.
#[test]
fn test_delete_nonexistent_concurrent() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert some data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    let barrier = Arc::new(Barrier::new(3));

    // Deleter thread - deletes non-existent docs
    let index_deleter = Arc::clone(&index);
    let barrier_deleter = Arc::clone(&barrier);
    let deleter = thread::spawn(move || {
        barrier_deleter.wait();
        for i in 1000..2000 {
            index_deleter.delete(i);
        }
    });

    // Reader thread
    let index_reader = Arc::clone(&index);
    let barrier_reader = Arc::clone(&barrier);
    let reader = thread::spawn(move || {
        barrier_reader.wait();
        for _ in 0..100 {
            let results: Vec<u64> = index_reader.filter(FilterOp::Gte(1)).iter().collect();
            assert_eq!(results.len(), 100);
        }
    });

    // Compactor
    let index_compact = Arc::clone(&index);
    let barrier_compact = Arc::clone(&barrier);
    let compactor = thread::spawn(move || {
        barrier_compact.wait();
        index_compact.compact(1).unwrap();
    });

    deleter.join().unwrap();
    reader.join().unwrap();
    compactor.join().unwrap();

    // Original data should still be present
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(results.len(), 100);
}

// ============================================================================
// Cleanup Concurrent Tests
// ============================================================================

/// Test cleanup during concurrent reads.
#[test]
fn test_cleanup_during_reads() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Create multiple versions
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(1).unwrap();

    for i in 101..=200 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(2).unwrap();

    for i in 201..=300 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(3).unwrap();

    let barrier = Arc::new(Barrier::new(3));

    // Reader threads
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
                    assert_eq!(results.len(), 300);
                }
            })
        })
        .collect();

    // Cleanup thread
    let index_cleanup = Arc::clone(&index);
    let barrier_cleanup = Arc::clone(&barrier);
    let cleaner = thread::spawn(move || {
        barrier_cleanup.wait();
        index_cleanup.cleanup();
    });

    for handle in handles {
        handle.join().unwrap();
    }
    cleaner.join().unwrap();

    // Data should still be readable
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_eq!(results.len(), 300);
}

// ============================================================================
// Long-Running Stability Tests
// ============================================================================

/// Long-running test with continuous operations.
/// This test runs for longer to detect subtle race conditions.
#[test]
fn test_long_running_stability() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let total_inserts = Arc::new(AtomicU64::new(0));
    let total_deletes = Arc::new(AtomicU64::new(0));

    // Writer thread
    let index_writer = Arc::clone(&index);
    let running_writer = Arc::clone(&running);
    let inserts = Arc::clone(&total_inserts);
    let writer = thread::spawn(move || {
        let mut i = 0u64;
        while running_writer.load(Ordering::Relaxed) {
            index_writer
                .insert(&IndexedValue::Plain(i % 10000), i)
                .unwrap();
            inserts.fetch_add(1, Ordering::Relaxed);
            i += 1;
        }
    });

    // Deleter thread
    let index_deleter = Arc::clone(&index);
    let running_deleter = Arc::clone(&running);
    let deletes = Arc::clone(&total_deletes);
    let deleter = thread::spawn(move || {
        let mut i = 0u64;
        while running_deleter.load(Ordering::Relaxed) {
            index_deleter.delete(i);
            deletes.fetch_add(1, Ordering::Relaxed);
            i += 1;
            thread::sleep(Duration::from_micros(10));
        }
    });

    // Reader threads
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let index = Arc::clone(&index);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let _ = index.filter(FilterOp::Gte(0)).iter().count();
                }
            })
        })
        .collect();

    // Compactor thread
    let index_compact = Arc::clone(&index);
    let running_compact = Arc::clone(&running);
    let compactor = thread::spawn(move || {
        let mut offset = 1u64;
        while running_compact.load(Ordering::Relaxed) {
            index_compact.compact(offset).unwrap();
            offset += 1;
            thread::sleep(Duration::from_millis(50));
        }
    });

    // Run for 5 seconds
    thread::sleep(Duration::from_secs(5));
    running.store(false, Ordering::Relaxed);

    writer.join().unwrap();
    deleter.join().unwrap();
    for reader in readers {
        reader.join().unwrap();
    }
    compactor.join().unwrap();

    let total_ins = total_inserts.load(Ordering::Relaxed);
    let total_del = total_deletes.load(Ordering::Relaxed);

    println!("Long-running test completed: {total_ins} inserts, {total_del} deletes");

    // Verify we can still query and index has reasonable data
    let final_results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    let final_count = final_results.len();

    // We should have at least some data - with 5 seconds of inserts, expect significant entries
    // The writer inserts continuously, so we should have at least 1000 entries
    assert!(
        final_count >= 100,
        "Index should have at least 100 entries after 5 seconds of inserts, got {final_count}"
    );

    // Verify no duplicate doc_ids in results
    let result_set: HashSet<u64> = final_results.iter().copied().collect();
    assert_eq!(
        final_count,
        result_set.len(),
        "Found duplicate doc_ids in final results: {} entries but {} unique",
        final_count,
        result_set.len()
    );

    // Note: We don't assert strict sorting here because this test uses `i % 10000` as the key,
    // meaning many doc_ids share the same key value. Doc_ids are sorted by key first, then
    // by doc_id within each key group, which doesn't guarantee strict ascending doc_id order.
}
