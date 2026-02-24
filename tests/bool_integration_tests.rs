//! Integration tests for the boolean index.

use oramacore_fields::bool::{
    BoolStorage, CheckStatus, DeletionThreshold, IndexedValue, SortOrder,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

/// Test basic insert -> filter -> verify flow.
#[test]
fn test_basic_flow() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 0); // Edge case: doc_id 0
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.insert(&IndexedValue::Plain(false), 20);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![0, 1, 2, 3],
        "True filter should return doc_ids 0, 1, 2, 3 (including edge case 0)"
    );
    assert_eq!(
        false_results,
        vec![10, 20],
        "False filter should return doc_ids 10, 20"
    );
}

/// Test doc_id = 0 edge case explicitly.
/// Verifies that doc_id 0 (a common edge case for off-by-one errors) is handled correctly.
#[test]
fn test_doc_id_zero() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert doc_id 0 into both sets
    index.insert(&IndexedValue::Plain(true), 0);
    index.insert(&IndexedValue::Plain(false), 0);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(true_results, vec![0], "doc_id 0 should be in true set");
    assert_eq!(false_results, vec![0], "doc_id 0 should be in false set");

    // Test compaction with doc_id 0
    index.compact(1).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![0],
        "doc_id 0 should persist after compaction in true set"
    );
    assert_eq!(
        false_results,
        vec![0],
        "doc_id 0 should persist after compaction in false set"
    );

    // Test removal of doc_id 0
    index.delete(0);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert!(
        true_results.is_empty(),
        "doc_id 0 should be removed from true set"
    );
    assert!(
        false_results.is_empty(),
        "doc_id 0 should be removed from false set"
    );

    // Test persistence of removal
    index.compact(2).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert!(
        true_results.is_empty(),
        "doc_id 0 removal should persist after compaction"
    );
    assert!(
        false_results.is_empty(),
        "doc_id 0 removal should persist after compaction"
    );
}

/// Test with sparse/holey doc_ids.
#[test]
fn test_sparse_doc_ids() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert doc_ids with large gaps
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(true), 100);
    index.insert(&IndexedValue::Plain(true), 1000);
    index.insert(&IndexedValue::Plain(true), 10000);

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 10, 100, 1000, 10000],
        "Sparse doc_ids should be returned in sorted order"
    );

    // Compact and verify
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 10, 100, 1000, 10000],
        "Sparse doc_ids should persist after compaction"
    );
}

/// Test deletes exclude removed doc_ids from both sets.
#[test]
fn test_deletes() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(true), 4);
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(false), 2); // Same doc_id in false
    index.insert(&IndexedValue::Plain(false), 4); // Same doc_id in false

    // Remove some - removes from BOTH true and false
    index.delete(2);
    index.delete(4);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![1, 3, 5],
        "True set should exclude removed doc_ids 2 and 4"
    );
    assert!(
        false_results.is_empty(),
        "False set should be empty after removing doc_ids 2 and 4"
    );
}

/// Test that compaction preserves data.
#[test]
fn test_compaction_preserves_data() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(false), 2);
    index.insert(&IndexedValue::Plain(false), 6);

    // Capture state before compaction
    let true_before: Vec<u64> = index.filter(true).iter().collect();
    let false_before: Vec<u64> = index.filter(false).iter().collect();

    // Compact
    index.compact(1).unwrap();

    // Verify state after compaction
    let true_after: Vec<u64> = index.filter(true).iter().collect();
    let false_after: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_before, true_after,
        "True set should be preserved after compaction"
    );
    assert_eq!(
        false_before, false_after,
        "False set should be preserved after compaction"
    );
}

/// Test Strategy A: high delete ratio triggers delete application.
#[test]
fn test_strategy_a_high_delete_ratio() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert 10 items
    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(true), i);
    }

    // Delete 2 items (20% ratio, above 10% threshold)
    index.delete(3);
    index.delete(7);

    // Compact - should apply deletions (Strategy A)
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 2, 4, 5, 6, 8, 9, 10],
        "Strategy A (high delete ratio) should apply deletions during compaction"
    );
}

/// Test Strategy B: low delete ratio carries forward deletes.
#[test]
fn test_strategy_b_low_delete_ratio() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert 100 items
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(true), i);
    }

    // Delete 5 items (5% ratio, below 10% threshold)
    index.delete(10);
    index.delete(20);
    index.delete(30);
    index.delete(40);
    index.delete(50);

    // Compact - should carry forward deletes (Strategy B)
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();

    // Verify exact expected values: all items 1..=100 except deleted ones
    let deleted = [10, 20, 30, 40, 50];
    let expected: Vec<u64> = (1..=100).filter(|x| !deleted.contains(x)).collect();
    assert_eq!(
        results, expected,
        "Strategy B should exclude deleted items while carrying forward deletes"
    );
}

/// Test concurrent filter during compact.
#[test]
fn test_concurrent_filter_during_compact() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    // Insert initial data and compact first to have data in the version
    for i in 1..=1000 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    let index_clone = Arc::clone(&index);

    // Spawn filter threads
    let filter_handles: Vec<_> = (0..4)
        .map(|_| {
            let idx = Arc::clone(&index_clone);
            thread::spawn(move || {
                for _ in 0..100 {
                    let results: Vec<u64> = idx.filter(true).iter().collect();
                    // After compaction, we should always see 1000 items
                    assert_eq!(results.len(), 1000);
                }
            })
        })
        .collect();

    // Compact again in the main thread
    index.compact(2).unwrap();

    // Wait for filter threads
    for handle in filter_handles {
        handle.join().unwrap();
    }

    // Verify final state
    let results: Vec<u64> = index.filter(true).iter().collect();
    let expected: Vec<u64> = (1..=1000).collect();
    assert_eq!(
        results, expected,
        "All 1000 doc_ids should be present and sorted"
    );
}

/// Test persistence: create, compact, drop, reopen.
#[test]
fn test_persistence() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();

    // Create and populate index
    {
        let index = BoolStorage::new(base_path.clone(), DeletionThreshold::default()).unwrap();
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(false), 2);
        index.insert(&IndexedValue::Plain(false), 6);
        index.delete(5); // Removes from both
        index.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let index = BoolStorage::new(base_path, DeletionThreshold::default()).unwrap();

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(
            true_results,
            vec![1, 10],
            "True set should persist correctly after reopen"
        );
        assert_eq!(
            false_results,
            vec![2, 6],
            "False set should persist correctly after reopen"
        );
        assert_eq!(
            index.current_version_number(),
            1,
            "Offset should persist correctly after reopen"
        );
    }
}

/// Test multiple compactions.
#[test]
fn test_multiple_compactions() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // First batch
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.compact(1).unwrap();

    // Second batch
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(true), 4);
    index.compact(2).unwrap();

    // Third batch with deletes
    index.delete(2);
    index.delete(4);
    index.compact(3).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 3],
        "Multiple compactions should correctly apply all deletes"
    );
    assert_eq!(
        index.current_version_number(),
        3,
        "Offset should track multiple compactions"
    );
}

/// Test empty index operations.
#[test]
fn test_empty_index() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert!(
        true_results.is_empty(),
        "Empty index should return empty true set"
    );
    assert!(
        false_results.is_empty(),
        "Empty index should return empty false set"
    );

    // Compact empty index
    index.compact(1).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert!(
        true_results.is_empty(),
        "Empty index should remain empty after compaction"
    );
    assert!(
        false_results.is_empty(),
        "Empty index should remain empty after compaction"
    );
}

/// Test removing non-existent doc_id (should be no-op).
#[test]
fn test_remove_nonexistent() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.delete(100); // Non-existent

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 2],
        "Removing non-existent doc_id should be a no-op"
    );
}

/// Test duplicate inserts (should deduplicate).
#[test]
fn test_duplicate_inserts() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 2],
        "Duplicate inserts should be deduplicated"
    );

    // After compaction
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![1, 2],
        "Deduplication should persist after compaction"
    );
}

/// Test large doc_ids (full u64 range now supported).
#[test]
fn test_large_doc_ids() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Use large doc_ids including values with high bits set
    let large_id = u64::MAX - 1;
    index.insert(&IndexedValue::Plain(true), large_id);
    index.insert(&IndexedValue::Plain(true), large_id - 1);
    index.insert(&IndexedValue::Plain(false), large_id - 2);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![large_id - 1, large_id],
        "Large doc_ids near u64::MAX should be stored correctly"
    );
    assert_eq!(
        false_results,
        vec![large_id - 2],
        "Large doc_ids near u64::MAX should be stored correctly"
    );

    // After compaction
    index.compact(1).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![large_id - 1, large_id],
        "Large doc_ids should persist after compaction"
    );
    assert_eq!(
        false_results,
        vec![large_id - 2],
        "Large doc_ids should persist after compaction"
    );
}

/// Test interleaved operations: insert, delete, insert, compact, delete, etc.
#[test]
fn test_interleaved_operations() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.delete(1);
    index.insert(&IndexedValue::Plain(true), 3);

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![2, 3],
        "Interleaved insert-delete should reflect correct state"
    );

    index.compact(1).unwrap();

    index.insert(&IndexedValue::Plain(true), 4);
    index.delete(2);

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![3, 4],
        "Post-compaction operations should work correctly"
    );

    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results,
        vec![3, 4],
        "Second compaction should preserve correct state"
    );
}

/// Test that operations during compaction are preserved.
#[test]
fn test_ops_during_compaction_preserved() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    // Insert initial data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let index_clone = Arc::clone(&index);

    // Start insert thread that continuously adds new items
    let insert_handle = thread::spawn(move || {
        for i in 101..=200 {
            index_clone.insert(&IndexedValue::Plain(true), i);
            thread::sleep(std::time::Duration::from_micros(10));
        }
    });

    // Compact in main thread
    index.compact(1).unwrap();

    // Wait for inserts to complete
    insert_handle.join().unwrap();

    // All items should be visible (both original and concurrent inserts)
    let results: Vec<u64> = index.filter(true).iter().collect();

    // Verify ALL 200 items are present - operations during compaction must be preserved
    for i in 1..=200 {
        assert!(
            results.binary_search(&i).is_ok(),
            "Missing item {i} - operations during compaction may have been lost"
        );
    }
    assert_eq!(
        results.len(),
        200,
        "All 200 items should be present (100 original + 100 concurrent)"
    );
}

/// Test that delete affects both true and false sets.
#[test]
fn test_remove_affects_both_sets() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert same doc_id in both sets
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(false), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(false), 3);

    // Delete doc_id 1 - should delete from BOTH sets
    index.delete(1);

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![2],
        "Remove should affect both sets - true set"
    );
    assert_eq!(
        false_results,
        vec![3],
        "Remove should affect both sets - false set"
    );

    // Compact and verify persistence
    index.compact(1).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    assert_eq!(
        true_results,
        vec![2],
        "Remove from both sets should persist after compaction"
    );
    assert_eq!(
        false_results,
        vec![3],
        "Remove from both sets should persist after compaction"
    );
}

// ============================================================================
// CONCURRENT INSERTIONS TESTS
// ============================================================================

/// Test multiple threads inserting different doc_ids concurrently.
/// Verifies no data loss under concurrent write pressure.
#[test]
fn test_concurrent_multi_thread_inserts() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_THREADS: usize = 8;
    const INSERTS_PER_THREAD: u64 = 1000;

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(true), start + i);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all 8000 doc_ids are present
    let results: Vec<u64> = index.filter(true).iter().collect();
    let expected_count = NUM_THREADS as u64 * INSERTS_PER_THREAD;
    assert_eq!(
        results.len() as u64,
        expected_count,
        "Expected {} items, got {}",
        expected_count,
        results.len()
    );

    // Verify sorted and deduplicated
    for i in 1..results.len() {
        assert!(
            results[i] > results[i - 1],
            "Results not sorted/deduplicated at position {}: {} <= {}",
            i,
            results[i],
            results[i - 1]
        );
    }

    // Verify all expected doc_ids are present (use binary_search since results are sorted)
    for i in 0..expected_count {
        assert!(
            results.binary_search(&i).is_ok(),
            "Missing doc_id {i} in results"
        );
    }
}

/// Test threads inserting alternating true/false values concurrently.
/// Verifies correct partitioning with no overlap between sets.
#[test]
fn test_concurrent_inserts_alternating_values() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_TRUE_THREADS: usize = 4;
    const NUM_FALSE_THREADS: usize = 4;
    const INSERTS_PER_THREAD: u64 = 500;

    let barrier = Arc::new(Barrier::new(NUM_TRUE_THREADS + NUM_FALSE_THREADS));

    // True inserters use even doc_ids: 0, 2, 4, ...
    let true_handles: Vec<_> = (0..NUM_TRUE_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * INSERTS_PER_THREAD * 2;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(true), start + i * 2); // Even numbers
                }
            })
        })
        .collect();

    // False inserters use odd doc_ids: 1, 3, 5, ...
    let false_handles: Vec<_> = (0..NUM_FALSE_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * INSERTS_PER_THREAD * 2 + 1;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(false), start + i * 2); // Odd numbers
                }
            })
        })
        .collect();

    for handle in true_handles.into_iter().chain(false_handles) {
        handle.join().unwrap();
    }

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    // Verify correct counts
    let expected_true = NUM_TRUE_THREADS as u64 * INSERTS_PER_THREAD;
    let expected_false = NUM_FALSE_THREADS as u64 * INSERTS_PER_THREAD;

    assert_eq!(
        true_results.len() as u64,
        expected_true,
        "Expected {} true items, got {}",
        expected_true,
        true_results.len()
    );
    assert_eq!(
        false_results.len() as u64,
        expected_false,
        "Expected {} false items, got {}",
        expected_false,
        false_results.len()
    );

    // Verify no overlap
    let true_set: HashSet<u64> = true_results.iter().copied().collect();
    let false_set: HashSet<u64> = false_results.iter().copied().collect();
    let overlap: Vec<_> = true_set.intersection(&false_set).collect();
    assert!(
        overlap.is_empty(),
        "Unexpected overlap between true and false sets: {overlap:?}"
    );

    // Verify all true results are even
    for &id in &true_results {
        assert!(id % 2 == 0, "True set contains odd id: {id}");
    }

    // Verify all false results are odd
    for &id in &false_results {
        assert!(id % 2 == 1, "False set contains even id: {id}");
    }
}

/// Test same doc_id being inserted with different values from different threads.
///
/// Expected behavior: The index allows a doc_id to exist in both the true and false sets
/// simultaneously (the sets are not mutually exclusive). This test verifies that concurrent
/// inserts of the same doc_id with different boolean values don't cause panics or data
/// corruption. The doc_id may end up in one or both sets depending on timing.
///
/// Design note: The BoolStorage does not enforce mutual exclusion between sets - a document
/// can be tagged as both true and false. If mutual exclusion is needed, it should be
/// enforced at a higher level.
#[test]
fn test_concurrent_inserts_same_doc_id() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_ITERATIONS: usize = 1000;

    let barrier = Arc::new(Barrier::new(2));

    let idx1 = Arc::clone(&index);
    let b1 = Arc::clone(&barrier);
    let handle1 = thread::spawn(move || {
        b1.wait();
        for _ in 0..NUM_ITERATIONS {
            idx1.insert(&IndexedValue::Plain(true), 1);
        }
    });

    let idx2 = Arc::clone(&index);
    let b2 = Arc::clone(&barrier);
    let handle2 = thread::spawn(move || {
        b2.wait();
        for _ in 0..NUM_ITERATIONS {
            idx2.insert(&IndexedValue::Plain(false), 1);
        }
    });

    handle1.join().unwrap();
    handle2.join().unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    // Doc_id 1 should be in at least one set (may be in both since sets are not mutually exclusive)
    let in_true = true_results.contains(&1);
    let in_false = false_results.contains(&1);

    assert!(
        in_true || in_false,
        "Doc_id 1 not found in either set - concurrent inserts failed to persist"
    );

    // Both should be deduplicated (only one occurrence of doc_id 1 per set)
    assert!(
        true_results.iter().filter(|&&x| x == 1).count() <= 1,
        "Doc_id 1 duplicated in true set"
    );
    assert!(
        false_results.iter().filter(|&&x| x == 1).count() <= 1,
        "Doc_id 1 duplicated in false set"
    );
}

// ============================================================================
// CONCURRENT DELETIONS TESTS
// ============================================================================

/// Test multiple threads deleting different doc_ids concurrently.
/// Verifies all deletes are applied correctly.
#[test]
fn test_concurrent_multi_thread_deletes() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_THREADS: usize = 8;
    const DELETES_PER_THREAD: u64 = 500;
    const TOTAL_DOCS: u64 = NUM_THREADS as u64 * DELETES_PER_THREAD * 2;

    // Insert all documents first
    for i in 0..TOTAL_DOCS {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Each thread deletes every other doc_id in its range
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let range_start = (thread_id as u64) * DELETES_PER_THREAD * 2;
                for i in 0..DELETES_PER_THREAD {
                    idx.delete(range_start + i * 2); // Delete even positions in range
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let results: Vec<u64> = index.filter(true).iter().collect();

    // Verify deleted items are absent (use binary_search since results are sorted)
    for thread_id in 0..NUM_THREADS {
        let range_start = (thread_id as u64) * DELETES_PER_THREAD * 2;
        for i in 0..DELETES_PER_THREAD {
            let deleted_id = range_start + i * 2;
            assert!(
                results.binary_search(&deleted_id).is_err(),
                "Deleted doc_id {deleted_id} still present"
            );
        }
    }

    // Verify remaining items are present
    for thread_id in 0..NUM_THREADS {
        let range_start = (thread_id as u64) * DELETES_PER_THREAD * 2;
        for i in 0..DELETES_PER_THREAD {
            let remaining_id = range_start + i * 2 + 1;
            assert!(
                results.binary_search(&remaining_id).is_ok(),
                "Non-deleted doc_id {remaining_id} missing"
            );
        }
    }

    let expected_remaining = TOTAL_DOCS / 2;
    assert_eq!(
        results.len() as u64,
        expected_remaining,
        "Expected {} remaining items, got {}",
        expected_remaining,
        results.len()
    );
}

/// Test deletes occurring while snapshot refresh happens.
/// Verifies deletes are applied correctly even during filter operations.
#[test]
fn test_concurrent_deletes_during_snapshot_refresh() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_DOCS: u64 = 1000;
    const NUM_DELETE_THREADS: usize = 3;
    const DELETES_PER_THREAD: u64 = 100;

    // Insert documents
    for i in 0..NUM_DOCS {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(NUM_DELETE_THREADS + 1));
    let stop = Arc::new(AtomicBool::new(false));

    // Delete threads
    let delete_handles: Vec<_> = (0..NUM_DELETE_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * DELETES_PER_THREAD;
                for i in 0..DELETES_PER_THREAD {
                    idx.delete(start + i);
                }
            })
        })
        .collect();

    // Filter thread that triggers snapshot refreshes
    let idx_filter = Arc::clone(&index);
    let b_filter = Arc::clone(&barrier);
    let stop_filter = Arc::clone(&stop);
    let filter_handle = thread::spawn(move || {
        b_filter.wait();
        while !stop_filter.load(Ordering::Relaxed) {
            let _results: Vec<u64> = idx_filter.filter(true).iter().collect();
        }
    });

    // Wait for delete threads
    for handle in delete_handles {
        handle.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    filter_handle.join().unwrap();

    // Verify all deletes were applied (use binary_search since results are sorted)
    let results: Vec<u64> = index.filter(true).iter().collect();

    let total_deleted = NUM_DELETE_THREADS as u64 * DELETES_PER_THREAD;
    for i in 0..total_deleted {
        assert!(
            results.binary_search(&i).is_err(),
            "Deleted doc_id {i} still present"
        );
    }

    let expected_remaining = NUM_DOCS - total_deleted;
    assert_eq!(
        results.len() as u64,
        expected_remaining,
        "Expected {} remaining items, got {}",
        expected_remaining,
        results.len()
    );
}

// ============================================================================
// CONCURRENT FILTERS TESTS
// ============================================================================

/// Test multiple readers see consistent snapshots.
/// Verifies read consistency with no mutations.
#[test]
fn test_concurrent_filters_read_consistency() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_DOCS: u64 = 1000;
    const NUM_READER_THREADS: usize = 16;

    // Insert documents
    for i in 0..NUM_DOCS {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(NUM_READER_THREADS));
    // Use u64::MAX as sentinel for "not yet set"
    let expected_count = Arc::new(AtomicU64::new(u64::MAX));

    let handles: Vec<_> = (0..NUM_READER_THREADS)
        .map(|_| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let exp = Arc::clone(&expected_count);
            thread::spawn(move || {
                b.wait();
                for _ in 0..100 {
                    let results: Vec<u64> = idx.filter(true).iter().collect();
                    let count = results.len() as u64;

                    // Atomically set expected count if not yet set, or get the existing value
                    let expected = match exp.compare_exchange(
                        u64::MAX,
                        count,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => count,            // We were first, we set it
                        Err(existing) => existing, // Someone else set it, use their value
                    };
                    assert_eq!(
                        count, expected,
                        "Inconsistent read: got {count} but expected {expected}"
                    );
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test lock promotion race when multiple threads detect dirty snapshot.
/// Verifies no panics and data remains consistent.
#[test]
fn test_concurrent_filters_during_dirty_snapshot() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_READER_THREADS: usize = 8;
    const NUM_ITERATIONS: usize = 1000;

    let barrier = Arc::new(Barrier::new(NUM_READER_THREADS + 1));
    let stop = Arc::new(AtomicBool::new(false));
    let readers_started = Arc::new(AtomicUsize::new(0));

    // Reader threads
    let reader_handles: Vec<_> = (0..NUM_READER_THREADS)
        .map(|_| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&stop);
            let rs = Arc::clone(&readers_started);
            thread::spawn(move || {
                b.wait();
                let mut count = 0u64;
                while !s.load(Ordering::Acquire) {
                    let results: Vec<u64> = idx.filter(true).iter().collect();
                    // Results should always be sorted and deduplicated
                    for i in 1..results.len() {
                        assert!(
                            results[i] > results[i - 1],
                            "Results not sorted at position {i}"
                        );
                    }
                    if count == 0 {
                        rs.fetch_add(1, Ordering::Release);
                    }
                    count += 1;
                }
                count
            })
        })
        .collect();

    // Writer thread that continuously dirties the snapshot
    let idx_writer = Arc::clone(&index);
    let b_writer = Arc::clone(&barrier);
    let s_writer = Arc::clone(&stop);
    let rs_writer = Arc::clone(&readers_started);
    let writer_handle = thread::spawn(move || {
        b_writer.wait();
        let mut doc_id = 0u64;
        for i in 0..NUM_ITERATIONS {
            idx_writer.insert(&IndexedValue::Plain(true), doc_id);
            doc_id += 1;
            // After the first insert (which dirties the snapshot), wait until
            // all readers have completed at least one filter() call. This
            // guarantees readers observe dirty snapshots while the remaining
            // inserts continue concurrently.
            if i == 0 {
                while rs_writer.load(Ordering::Acquire) < NUM_READER_THREADS {
                    thread::yield_now();
                }
            }
            thread::yield_now();
        }
        s_writer.store(true, Ordering::Release);
        doc_id
    });

    let final_doc_id = writer_handle.join().unwrap();

    let mut total_reads = 0;
    for handle in reader_handles {
        total_reads += handle.join().unwrap();
    }

    // Verify final state
    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results.len() as u64,
        final_doc_id,
        "Final count mismatch: {} vs {}",
        results.len(),
        final_doc_id
    );
    assert!(total_reads > 0, "No reads completed");
}

/// Test FilterData holds old Arc after version swap.
/// Verifies iterator stability during compaction.
#[test]
fn test_filter_iterator_stability_during_version_swap() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    // Insert initial data and compact
    for i in 0..100 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    // Get filter data BEFORE adding more and compacting
    let old_filter = index.filter(true);

    // Add more data and compact
    for i in 100..200 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(2).unwrap();

    // Old filter should still see only original 100 items
    let old_results: Vec<u64> = old_filter.iter().collect();
    let expected_old: Vec<u64> = (0..100).collect();
    assert_eq!(
        old_results, expected_old,
        "Old filter should see exactly doc_ids 0..100"
    );

    // New filter should see all 200 items
    let new_filter = index.filter(true);
    let new_results: Vec<u64> = new_filter.iter().collect();
    let expected_new: Vec<u64> = (0..200).collect();
    assert_eq!(
        new_results, expected_new,
        "New filter should see exactly doc_ids 0..200"
    );
}

// ============================================================================
// MIXED OPERATIONS TESTS
// ============================================================================

/// Test all operations running concurrently.
/// Verifies inserts are present and deletes are absent.
#[test]
fn test_mixed_insert_delete_filter() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_INSERT_THREADS: usize = 4;
    const NUM_DELETE_THREADS: usize = 4;
    const NUM_FILTER_THREADS: usize = 4;
    const INSERTS_PER_THREAD: u64 = 500;

    // Insert range: 0..2000 for inserts, 2000..4000 for pre-inserted docs to delete
    let pre_insert_start = 2000u64;
    let pre_insert_count = NUM_DELETE_THREADS as u64 * INSERTS_PER_THREAD;

    // Pre-insert documents to be deleted
    for i in 0..pre_insert_count {
        index.insert(&IndexedValue::Plain(true), pre_insert_start + i);
    }

    let barrier = Arc::new(Barrier::new(
        NUM_INSERT_THREADS + NUM_DELETE_THREADS + NUM_FILTER_THREADS,
    ));
    let stop = Arc::new(AtomicBool::new(false));

    // Insert threads
    let insert_handles: Vec<_> = (0..NUM_INSERT_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(true), start + i);
                }
            })
        })
        .collect();

    // Delete threads
    let delete_handles: Vec<_> = (0..NUM_DELETE_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = pre_insert_start + (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.delete(start + i);
                }
            })
        })
        .collect();

    // Filter threads
    let filter_handles: Vec<_> = (0..NUM_FILTER_THREADS)
        .map(|_| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&stop);
            thread::spawn(move || {
                b.wait();
                while !s.load(Ordering::Relaxed) {
                    let results: Vec<u64> = idx.filter(true).iter().collect();
                    // Verify sorted
                    for i in 1..results.len() {
                        assert!(results[i] > results[i - 1], "Results not sorted");
                    }
                }
            })
        })
        .collect();

    // Wait for insert and delete threads
    for handle in insert_handles {
        handle.join().unwrap();
    }
    for handle in delete_handles {
        handle.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    for handle in filter_handles {
        handle.join().unwrap();
    }

    // Verify final state (use binary_search since results are sorted)
    let results: Vec<u64> = index.filter(true).iter().collect();

    // All inserts should be present
    let total_inserts = NUM_INSERT_THREADS as u64 * INSERTS_PER_THREAD;
    for i in 0..total_inserts {
        assert!(
            results.binary_search(&i).is_ok(),
            "Inserted doc_id {i} missing"
        );
    }

    // All deletes should be absent
    for i in 0..pre_insert_count {
        assert!(
            results.binary_search(&(pre_insert_start + i)).is_err(),
            "Deleted doc_id {} still present",
            pre_insert_start + i
        );
    }
}

/// Test rapid insert-delete cycles on same doc_id.
///
/// This test verifies the following invariants that must hold regardless of race order:
/// 1. No panics occur during concurrent insert/delete on the same doc_id
/// 2. Results are always sorted and deduplicated (no corruption)
/// 3. The index remains queryable after the concurrent operations complete
///
/// Note: The final state of doc_id 1 is intentionally non-deterministic - it may or may
/// not be present depending on the interleaving of operations. This test focuses on
/// verifying safety invariants, not functional correctness of the race outcome.
#[test]
fn test_interleaved_insert_delete_same_doc_id() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_ITERATIONS: usize = 1000;

    let barrier = Arc::new(Barrier::new(2));

    let idx1 = Arc::clone(&index);
    let b1 = Arc::clone(&barrier);
    let handle1 = thread::spawn(move || {
        b1.wait();
        for _ in 0..NUM_ITERATIONS {
            idx1.insert(&IndexedValue::Plain(true), 1);
        }
    });

    let idx2 = Arc::clone(&index);
    let b2 = Arc::clone(&barrier);
    let handle2 = thread::spawn(move || {
        b2.wait();
        for _ in 0..NUM_ITERATIONS {
            idx2.delete(1);
        }
    });

    handle1.join().unwrap();
    handle2.join().unwrap();

    // Invariant 1: No panic occurred - we reached this point
    // Invariant 2: Index is still queryable
    let results: Vec<u64> = index.filter(true).iter().collect();

    // Invariant 3: Results are valid (sorted, deduplicated) regardless of race outcome
    // Note: Doc_id 1 may or may not be present depending on operation interleaving
    for i in 1..results.len() {
        assert!(
            results[i] > results[i - 1],
            "Results not sorted at position {i} - possible corruption from concurrent operations"
        );
    }

    // Verify deduplication - each doc_id should appear at most once
    let unique_count = results.iter().collect::<HashSet<_>>().len();
    assert_eq!(
        results.len(),
        unique_count,
        "Results contain duplicates - deduplication failed"
    );
}

// ============================================================================
// COMPACTION UNDER LOAD TESTS
// ============================================================================

/// Test multiple threads attempting compaction simultaneously.
///
/// Expected behavior: Multiple concurrent compactions with different version numbers can all succeed
/// because each compaction writes to a different version file (determined by the version number).
/// The index's current_version_number will be set to whichever compaction completed last.
///
/// Key invariant: Data integrity must be maintained - all originally inserted documents
/// must be present regardless of how many compactions succeed.
#[test]
fn test_multiple_concurrent_compactions() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_COMPACTION_THREADS: usize = 4;
    const DOCS_PER_ROUND: u64 = 100;

    // Insert initial data
    for i in 0..DOCS_PER_ROUND {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(NUM_COMPACTION_THREADS));
    let success_count = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..NUM_COMPACTION_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let sc = Arc::clone(&success_count);
            thread::spawn(move || {
                b.wait();
                // Each thread tries to compact with a different version number
                if idx.compact((thread_id + 1) as u64).is_ok() {
                    sc.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Multiple compactions can succeed since each uses a different version number/file
    // At minimum, one should succeed; in practice, all may succeed with different offsets
    let successes = success_count.load(Ordering::SeqCst);
    assert!(
        successes >= 1,
        "No compaction succeeded - unexpected failure"
    );

    // Note: We don't assert exactly how many succeed because the outcome depends on
    // timing and lock contention. The important invariant is data integrity below.

    // Data integrity check
    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(
        results.len() as u64,
        DOCS_PER_ROUND,
        "Data lost: expected {} items, got {}",
        DOCS_PER_ROUND,
        results.len()
    );
}

/// Stress test: all operations plus compaction running concurrently.
/// Verifies no data loss under maximum concurrent pressure.
#[test]
fn test_compaction_with_concurrent_all_ops() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_INSERT_THREADS: usize = 2;
    const NUM_DELETE_THREADS: usize = 2;
    const NUM_FILTER_THREADS: usize = 4;
    const INSERTS_PER_THREAD: u64 = 200;

    // Pre-insert docs to be deleted (0-400)
    let delete_base = 0u64;
    let delete_count = NUM_DELETE_THREADS as u64 * INSERTS_PER_THREAD;
    for i in 0..delete_count {
        index.insert(&IndexedValue::Plain(true), delete_base + i);
    }

    let barrier = Arc::new(Barrier::new(
        NUM_INSERT_THREADS + NUM_DELETE_THREADS + NUM_FILTER_THREADS + 1,
    ));
    let stop = Arc::new(AtomicBool::new(false));

    // Insert threads (10000-10400)
    let insert_base = 10000u64;
    let insert_handles: Vec<_> = (0..NUM_INSERT_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = insert_base + (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(true), start + i);
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Delete threads (0-400)
    let delete_handles: Vec<_> = (0..NUM_DELETE_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let start = delete_base + (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.delete(start + i);
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Filter threads
    let filter_handles: Vec<_> = (0..NUM_FILTER_THREADS)
        .map(|_| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&stop);
            thread::spawn(move || {
                b.wait();
                while !s.load(Ordering::Relaxed) {
                    let _: Vec<u64> = idx.filter(true).iter().collect();
                }
            })
        })
        .collect();

    // Compact thread
    let idx_compact = Arc::clone(&index);
    let b_compact = Arc::clone(&barrier);
    let compact_handle = thread::spawn(move || {
        b_compact.wait();
        // Perform multiple compactions
        for version_number in 1..=3 {
            thread::sleep(std::time::Duration::from_millis(10));
            let _ = idx_compact.compact(version_number);
        }
    });

    // Wait for writers
    for handle in insert_handles {
        handle.join().unwrap();
    }
    for handle in delete_handles {
        handle.join().unwrap();
    }
    compact_handle.join().unwrap();

    stop.store(true, Ordering::Relaxed);
    for handle in filter_handles {
        handle.join().unwrap();
    }

    // Final compaction to ensure all ops are persisted
    let final_offset = index.current_version_number() + 1;
    index.compact(final_offset).unwrap();

    // Verify final state (use binary_search since results are sorted)
    let results: Vec<u64> = index.filter(true).iter().collect();

    // All inserts should be present (10000-10400)
    let total_inserts = NUM_INSERT_THREADS as u64 * INSERTS_PER_THREAD;
    for i in 0..total_inserts {
        let doc_id = insert_base + i;
        assert!(
            results.binary_search(&doc_id).is_ok(),
            "Inserted doc_id {doc_id} missing"
        );
    }

    // All deletes should be absent (0-400)
    for i in 0..delete_count {
        let doc_id = delete_base + i;
        assert!(
            results.binary_search(&doc_id).is_err(),
            "Deleted doc_id {doc_id} still present"
        );
    }
}

/// Test operations added during disk I/O phase are preserved.
/// Verifies inserts during compaction I/O are visible after compact.
#[test]
fn test_ops_during_compaction_io_phase() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const INITIAL_DOCS: u64 = 100;
    const CONCURRENT_INSERTS: u64 = 200;

    // Insert initial data
    for i in 0..INITIAL_DOCS {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(2));
    let inserts_done = Arc::new(AtomicU64::new(0));

    let idx_insert = Arc::clone(&index);
    let b_insert = Arc::clone(&barrier);
    let done = Arc::clone(&inserts_done);
    let insert_handle = thread::spawn(move || {
        b_insert.wait();
        for i in 0..CONCURRENT_INSERTS {
            idx_insert.insert(&IndexedValue::Plain(true), INITIAL_DOCS + i);
            done.fetch_add(1, Ordering::SeqCst);
        }
    });

    let idx_compact = Arc::clone(&index);
    let b_compact = Arc::clone(&barrier);
    let compact_handle = thread::spawn(move || {
        b_compact.wait();
        idx_compact.compact(1).unwrap();
    });

    insert_handle.join().unwrap();
    compact_handle.join().unwrap();

    // All inserts should be visible (use binary_search since results are sorted)
    let results: Vec<u64> = index.filter(true).iter().collect();

    // Initial docs should be present
    for i in 0..INITIAL_DOCS {
        assert!(
            results.binary_search(&i).is_ok(),
            "Initial doc_id {i} missing"
        );
    }

    // Concurrent inserts should be present
    for i in 0..CONCURRENT_INSERTS {
        assert!(
            results.binary_search(&(INITIAL_DOCS + i)).is_ok(),
            "Concurrent insert doc_id {} missing",
            INITIAL_DOCS + i
        );
    }

    let total_expected = INITIAL_DOCS + CONCURRENT_INSERTS;
    assert_eq!(
        results.len() as u64,
        total_expected,
        "Expected {} items, got {}",
        total_expected,
        results.len()
    );
}

// ============================================================================
// EDGE CASES TESTS
// ============================================================================

/// Stress write lock under extreme contention.
/// Verifies no deadlock and all inserts complete.
#[test]
fn test_high_write_contention() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_THREADS: usize = 32;
    const INSERTS_PER_THREAD: u64 = 10000;

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let completed = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let c = Arc::clone(&completed);
            thread::spawn(move || {
                b.wait();
                let start = (thread_id as u64) * INSERTS_PER_THREAD;
                for i in 0..INSERTS_PER_THREAD {
                    idx.insert(&IndexedValue::Plain(true), start + i);
                }
                c.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all threads completed (no deadlock)
    assert_eq!(
        completed.load(Ordering::SeqCst) as usize,
        NUM_THREADS,
        "Not all threads completed - possible deadlock"
    );

    // Verify all data present
    let results: Vec<u64> = index.filter(true).iter().collect();
    let expected_count = NUM_THREADS as u64 * INSERTS_PER_THREAD;
    assert_eq!(
        results.len() as u64,
        expected_count,
        "Expected {} items, got {}",
        expected_count,
        results.len()
    );
}

/// Target the double-check pattern in filter() for snapshot refresh.
/// Verifies snapshot is refreshed correctly under high concurrency.
#[test]
fn test_snapshot_refresh_race() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_THREADS: usize = 100;
    const NUM_ITERATIONS: usize = 100;

    // Insert initial data
    for i in 0..100 {
        index.insert(&IndexedValue::Plain(true), i);
    }

    let barrier = Arc::new(Barrier::new(NUM_THREADS + 1));

    // All threads call filter() simultaneously
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for _ in 0..NUM_ITERATIONS {
                    let results: Vec<u64> = idx.filter(true).iter().collect();
                    assert!(
                        results.len() >= 100,
                        "Snapshot not refreshed correctly: only {} items",
                        results.len()
                    );
                }
            })
        })
        .collect();

    // Writer that continuously dirties the snapshot
    let idx_writer = Arc::clone(&index);
    let b_writer = Arc::clone(&barrier);
    let writer_handle = thread::spawn(move || {
        b_writer.wait();
        for i in 100..200 {
            idx_writer.insert(&IndexedValue::Plain(true), i);
        }
    });

    for handle in handles {
        handle.join().unwrap();
    }
    writer_handle.join().unwrap();

    // Final verification
    let results: Vec<u64> = index.filter(true).iter().collect();
    assert_eq!(results.len(), 200, "Final count mismatch");
}

/// Test ArcSwap provides atomic visibility.
///
/// This test verifies that filter() never observes partial/inconsistent state during
/// version swaps caused by compaction. The key invariants being tested:
///
/// 1. Monotonic count increase: Once we observe N items, we should never see fewer
///    (assuming no concurrent deletes). A count decrease would indicate we saw an
///    older version after seeing a newer one - violating atomic swap semantics.
///
/// 2. Results are always sorted: Unsorted results would indicate corruption or
///    partial state visibility.
///
/// Note: The count may temporarily appear lower than expected if the live layer hasn't
/// been snapshotted yet, but it should never decrease from a previously observed value.
#[test]
fn test_version_swap_atomicity() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const DOCS_PER_BATCH: u64 = 500;
    const NUM_COMPACTIONS: u64 = 5;

    let stop = Arc::new(AtomicBool::new(false));
    let partial_state_detected = Arc::new(AtomicBool::new(false));

    // Insert initial batch
    for i in 0..DOCS_PER_BATCH {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    let idx_filter = Arc::clone(&index);
    let s_filter = Arc::clone(&stop);
    let partial = Arc::clone(&partial_state_detected);
    let filter_handle = thread::spawn(move || {
        let mut last_count = DOCS_PER_BATCH as usize;
        while !s_filter.load(Ordering::Relaxed) {
            let results: Vec<u64> = idx_filter.filter(true).iter().collect();
            let count = results.len();

            // Count should only increase (or stay same) - never decrease temporarily
            // and should always be a multiple of DOCS_PER_BATCH (no partial batches)
            if count < last_count {
                partial.store(true, Ordering::SeqCst);
            }
            last_count = count;

            // Verify sorted
            for i in 1..results.len() {
                if results[i] <= results[i - 1] {
                    partial.store(true, Ordering::SeqCst);
                }
            }
        }
    });

    // Compact thread that adds batches
    for batch in 2..=NUM_COMPACTIONS {
        let start = (batch - 1) * DOCS_PER_BATCH;
        for i in 0..DOCS_PER_BATCH {
            index.insert(&IndexedValue::Plain(true), start + i);
        }
        index.compact(batch).unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    filter_handle.join().unwrap();

    assert!(
        !partial_state_detected.load(Ordering::SeqCst),
        "Partial state detected during version swap"
    );

    // Final verification
    let results: Vec<u64> = index.filter(true).iter().collect();
    let expected = DOCS_PER_BATCH * NUM_COMPACTIONS;
    assert_eq!(
        results.len() as u64,
        expected,
        "Final count mismatch: expected {}, got {}",
        expected,
        results.len()
    );
}

/// Test deletes don't corrupt active iterator.
/// Verifies iterator completes correctly and sees old snapshot.
#[test]
fn test_delete_during_active_iteration() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_DOCS: u64 = 1000;

    // Insert documents
    for i in 0..NUM_DOCS {
        index.insert(&IndexedValue::Plain(true), i);
    }

    // Get filter data (captures snapshot)
    let filter_data = index.filter(true);

    // Spawn thread that deletes while we iterate
    let idx_delete = Arc::clone(&index);
    let delete_handle = thread::spawn(move || {
        for i in 0..NUM_DOCS / 2 {
            idx_delete.delete(i);
        }
    });

    // Iterate slowly while deletes happen
    let mut collected = Vec::new();
    for doc_id in filter_data.iter() {
        collected.push(doc_id);
        // Small yield to give delete thread a chance
        if collected.len() % 100 == 0 {
            thread::yield_now();
        }
    }

    delete_handle.join().unwrap();

    // Iterator should have seen all original documents (snapshot isolation)
    assert_eq!(
        collected.len() as u64,
        NUM_DOCS,
        "Iterator corrupted: expected {} items, got {}",
        NUM_DOCS,
        collected.len()
    );

    // Verify sorted
    for i in 1..collected.len() {
        assert!(
            collected[i] > collected[i - 1],
            "Iterator results not sorted"
        );
    }

    // New filter should see the deletes
    let new_results: Vec<u64> = index.filter(true).iter().collect();
    let expected_remaining: Vec<u64> = (NUM_DOCS / 2..NUM_DOCS).collect();
    assert_eq!(
        new_results,
        expected_remaining,
        "After deleting 0..{}, only {}..{} should remain",
        NUM_DOCS / 2,
        NUM_DOCS / 2,
        NUM_DOCS
    );
}

// ============================================================================
// ADDITIONAL STRESS TESTS
// ============================================================================

/// Randomized stress test with random operations.
/// Uses a seeded RNG for reproducibility. Set TEST_SEED env var to reproduce specific failures.
#[test]
fn test_randomized_operations() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const NUM_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 1000;

    // Use seeded RNG for reproducibility - can override with TEST_SEED env var
    let base_seed: u64 = std::env::var("TEST_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    eprintln!("test_randomized_operations using base seed: {base_seed}");

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let inserted = Arc::new(std::sync::Mutex::new(HashSet::new()));
    let deleted = Arc::new(std::sync::Mutex::new(HashSet::new()));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            let b = Arc::clone(&barrier);
            let ins = Arc::clone(&inserted);
            let del = Arc::clone(&deleted);
            // Each thread gets a deterministic seed derived from base_seed and thread_id
            let thread_seed = base_seed.wrapping_add(thread_id as u64);
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(thread_seed);
                b.wait();

                let base = (thread_id as u64) * (OPS_PER_THREAD as u64);
                for i in 0..OPS_PER_THREAD {
                    let doc_id = base + (i as u64);
                    let op: u8 = rng.random_range(0..10);

                    match op {
                        0..=6 => {
                            // 70% insert
                            let value = rng.random_bool(0.5);
                            idx.insert(&IndexedValue::Plain(value), doc_id);
                            ins.lock().unwrap().insert((doc_id, value));
                        }
                        7..=8 => {
                            // 20% delete
                            idx.delete(doc_id);
                            del.lock().unwrap().insert(doc_id);
                        }
                        _ => {
                            // 10% filter
                            let _: Vec<u64> = idx.filter(rng.random_bool(0.5)).iter().collect();
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify no panics occurred and data is consistent
    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();

    // Verify sorted
    for i in 1..true_results.len() {
        assert!(
            true_results[i] > true_results[i - 1],
            "True results not sorted"
        );
    }
    for i in 1..false_results.len() {
        assert!(
            false_results[i] > false_results[i - 1],
            "False results not sorted"
        );
    }

    // Verify deleted items are not present
    let deleted_set = deleted.lock().unwrap();
    let true_set: HashSet<u64> = true_results.iter().copied().collect();
    let false_set: HashSet<u64> = false_results.iter().copied().collect();

    for &doc_id in deleted_set.iter() {
        assert!(
            !true_set.contains(&doc_id),
            "Deleted doc_id {doc_id} found in true set"
        );
        assert!(
            !false_set.contains(&doc_id),
            "Deleted doc_id {doc_id} found in false set"
        );
    }
}

// ============================================================================
// SORTED ITERATOR TESTS
// ============================================================================

/// Test sorted ascending iteration returns doc_ids in ascending order.
#[test]
fn test_sorted_ascending_integration() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(true), 100);
    index.insert(&IndexedValue::Plain(true), 50);

    let results: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    assert_eq!(results, vec![1, 5, 10, 50, 100]);
}

/// Test sorted descending iteration returns doc_ids in descending order.
#[test]
fn test_sorted_descending_integration() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(true), 100);
    index.insert(&IndexedValue::Plain(true), 50);

    let results: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();
    assert_eq!(results, vec![100, 50, 10, 5, 1]);
}

/// Test sorted iteration with compacted data.
#[test]
fn test_sorted_with_compacted_data() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert and compact first batch
    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(true), 100);
    index.compact(1).unwrap();

    // Insert second batch (live layer)
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(true), 50);

    // Verify ascending merges compacted and live data correctly
    let asc: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    assert_eq!(asc, vec![1, 5, 10, 50, 100]);

    // Verify descending merges compacted and live data correctly
    let desc: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();
    assert_eq!(desc, vec![100, 50, 10, 5, 1]);
}

/// Test sorted iteration with deletes.
#[test]
fn test_sorted_with_deletes() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 5);
    index.insert(&IndexedValue::Plain(true), 10);
    index.insert(&IndexedValue::Plain(true), 15);
    index.insert(&IndexedValue::Plain(true), 20);

    // Delete some items
    index.delete(5);
    index.delete(15);

    let asc: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    let desc: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();

    assert_eq!(asc, vec![1, 10, 20]);
    assert_eq!(desc, vec![20, 10, 1]);

    // Test after compaction
    index.compact(1).unwrap();

    let asc: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    let desc: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();

    assert_eq!(asc, vec![1, 10, 20]);
    assert_eq!(desc, vec![20, 10, 1]);
}

/// Test sorted iteration on empty filter result.
#[test]
fn test_sorted_empty_results() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    let asc: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    let desc: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();

    assert_eq!(
        asc,
        Vec::<u64>::new(),
        "Ascending sort of empty should be empty"
    );
    assert_eq!(
        desc,
        Vec::<u64>::new(),
        "Descending sort of empty should be empty"
    );
}

/// Test sorted iteration on false values.
#[test]
fn test_sorted_false_values() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(false), 100);
    index.insert(&IndexedValue::Plain(false), 10);
    index.insert(&IndexedValue::Plain(false), 50);

    let asc: Vec<u64> = index.filter(false).sorted(SortOrder::Ascending).collect();
    let desc: Vec<u64> = index.filter(false).sorted(SortOrder::Descending).collect();

    assert_eq!(asc, vec![10, 50, 100]);
    assert_eq!(desc, vec![100, 50, 10]);
}

/// Test sorted iteration with single element.
#[test]
fn test_sorted_single_element() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 42);

    let asc: Vec<u64> = index.filter(true).sorted(SortOrder::Ascending).collect();
    let desc: Vec<u64> = index.filter(true).sorted(SortOrder::Descending).collect();

    assert_eq!(asc, vec![42]);
    assert_eq!(desc, vec![42]);
}

/// Test compaction during heavy concurrent load with verification.
#[test]
fn test_compaction_verification_under_load() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap());

    const BATCH_SIZE: u64 = 100;
    const NUM_BATCHES: u64 = 10;

    let stop = Arc::new(AtomicBool::new(false));
    let shared_max_seen = Arc::new(AtomicU64::new(0));

    // Reader thread that continuously verifies data
    let idx_reader = Arc::clone(&index);
    let s_reader = Arc::clone(&stop);
    let sms = Arc::clone(&shared_max_seen);
    let reader_handle = thread::spawn(move || {
        while !s_reader.load(Ordering::Relaxed) {
            let results: Vec<u64> = idx_reader.filter(true).iter().collect();

            let count = results.len() as u64;
            sms.fetch_max(count, Ordering::Release);

            // Verify sorted
            if !results.is_empty() {
                for i in 1..results.len() {
                    assert!(results[i] > results[i - 1], "Results not sorted");
                }
            }
        }
    });

    // Writer that adds batches and compacts
    for batch in 0..NUM_BATCHES {
        let start = batch * BATCH_SIZE;
        for i in 0..BATCH_SIZE {
            index.insert(&IndexedValue::Plain(true), start + i);
        }
        index.compact(batch + 1).unwrap();
    }

    // Retry: wait for the reader to observe nearly all items before stopping.
    let threshold = BATCH_SIZE * NUM_BATCHES - BATCH_SIZE;
    let mut max_seen = shared_max_seen.load(Ordering::Acquire);
    for _ in 0..100 {
        if max_seen >= threshold {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(10));
        max_seen = shared_max_seen.load(Ordering::Acquire);
    }

    stop.store(true, Ordering::Relaxed);
    reader_handle.join().unwrap();

    // Reader should have eventually seen all items
    assert!(
        max_seen >= threshold,
        "Reader didn't see enough items: max was {max_seen}, expected at least {threshold}"
    );

    // Final verification
    let results: Vec<u64> = index.filter(true).iter().collect();
    let expected = BATCH_SIZE * NUM_BATCHES;
    assert_eq!(
        results.len() as u64,
        expected,
        "Final count mismatch: expected {}, got {}",
        expected,
        results.len()
    );
}

// =====================================================================
// info() tests
// =====================================================================

#[test]
fn test_info_empty_index() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    let info = index.info();
    assert_eq!(info.format_version, 1);
    assert_eq!(info.current_version_number, 0);
    assert_eq!(info.true_count, 0);
    assert_eq!(info.false_count, 0);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.true_size_bytes, 0);
    assert_eq!(info.false_size_bytes, 0);
    assert_eq!(info.deleted_size_bytes, 0);
    assert_eq!(info.pending_ops, 0);
}

#[test]
fn test_info_with_pending_inserts() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.insert(&IndexedValue::Plain(false), 20);

    let info = index.info();
    assert_eq!(info.pending_ops, 5);
    // Nothing compacted yet
    assert_eq!(info.true_count, 0);
    assert_eq!(info.false_count, 0);
    assert_eq!(info.current_version_number, 0);
}

#[test]
fn test_info_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.insert(&IndexedValue::Plain(false), 20);
    index.compact(1).unwrap();

    let info = index.info();
    assert_eq!(info.true_count, 3);
    assert_eq!(info.false_count, 2);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.pending_ops, 0);
    assert!(info.true_size_bytes > 0);
    assert!(info.false_size_bytes > 0);
    assert_eq!(info.current_version_number, 1);
}

#[test]
fn test_info_with_pending_deletes() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.compact(1).unwrap();

    index.delete(1);
    index.delete(10);

    let info = index.info();
    assert_eq!(info.pending_ops, 2);
    assert_eq!(info.true_count, 3);
    assert_eq!(info.false_count, 1);
}

#[test]
fn test_info_after_multiple_compactions() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.compact(1).unwrap();

    let info = index.info();
    assert_eq!(info.current_version_number, 1);
    assert_eq!(info.true_count, 2);

    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.compact(2).unwrap();

    let info = index.info();
    assert_eq!(info.current_version_number, 2);
    assert_eq!(info.true_count, 3);
    assert_eq!(info.false_count, 1);
    assert_eq!(info.pending_ops, 0);
}

#[test]
fn test_info_total_documents_and_total_size() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(false), 10);
    index.compact(1).unwrap();

    let info = index.info();
    assert_eq!(info.total_documents(), info.true_count + info.false_count);
    assert_eq!(info.total_documents(), 3);
    assert_eq!(
        info.total_size_bytes(),
        info.true_size_bytes + info.false_size_bytes + info.deleted_size_bytes
    );
    assert!(info.total_size_bytes() > 0);
}

// =====================================================================
// integrity_check() tests
// =====================================================================

#[test]
fn test_integrity_check_before_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    let result = index.integrity_check();
    assert!(!result.passed);
    assert!(!result.checks.is_empty());
    assert_eq!(result.checks[0].name, "CURRENT");
    assert_eq!(result.checks[0].status, CheckStatus::Failed);
}

#[test]
fn test_integrity_check_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(false), 10);
    index.compact(1).unwrap();

    let result = index.integrity_check();
    assert!(result.passed, "Integrity check should pass: {result:?}");
    for check in &result.checks {
        assert_eq!(
            check.status,
            CheckStatus::Ok,
            "Check '{}' should be Ok: {:?}",
            check.name,
            check.details
        );
    }
}

#[test]
fn test_integrity_check_with_deletes_compacted() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.insert(&IndexedValue::Plain(true), 3);
    index.insert(&IndexedValue::Plain(false), 10);
    index.compact(1).unwrap();

    index.delete(1);
    index.delete(10);
    index.compact(2).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "Integrity check should pass after compacting deletes: {result:?}"
    );
}

#[test]
fn test_integrity_check_corrupted_current_file() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.compact(1).unwrap();

    // Overwrite CURRENT with garbage
    std::fs::write(tmp.path().join("CURRENT"), b"garbage data here").unwrap();

    let result = index.integrity_check();
    assert!(!result.passed, "Should fail with corrupted CURRENT file");
}

#[test]
fn test_integrity_check_missing_binary_file() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.compact(1).unwrap();

    // Delete true.bin
    let true_bin = tmp.path().join("versions").join("1").join("true.bin");
    std::fs::remove_file(&true_bin).unwrap();

    let result = index.integrity_check();
    assert!(!result.passed, "Should fail with missing binary file");
    let binary_check = result
        .checks
        .iter()
        .find(|c| c.name == "binary files")
        .expect("Should have a 'binary files' check");
    assert_eq!(binary_check.status, CheckStatus::Failed);
}

#[test]
fn test_integrity_check_corrupted_binary_file() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.compact(1).unwrap();

    // Drop the index to release mmaps (required on Windows where mmap holds file locks)
    drop(index);

    // Overwrite true.bin with bytes not a multiple of 8
    let true_bin = tmp.path().join("versions").join("1").join("true.bin");
    std::fs::write(&true_bin, b"12345").unwrap();

    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();
    let result = index.integrity_check();
    assert!(
        !result.passed,
        "Should fail with corrupted binary file (bad size)"
    );
    let valid_check = result
        .checks
        .iter()
        .find(|c| c.name == "binary files valid")
        .expect("Should have a 'binary files valid' check");
    assert_eq!(valid_check.status, CheckStatus::Failed);
}

#[test]
fn test_integrity_check_unsorted_binary_file() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(true), 1);
    index.insert(&IndexedValue::Plain(true), 2);
    index.compact(1).unwrap();

    // Drop the index to release mmaps (required on Windows where mmap holds file locks)
    drop(index);

    // Overwrite true.bin with valid-length but unsorted u64 values
    let true_bin = tmp.path().join("versions").join("1").join("true.bin");
    let unsorted: Vec<u8> = [10u64, 5u64].iter().flat_map(|v| v.to_ne_bytes()).collect();
    std::fs::write(&true_bin, &unsorted).unwrap();

    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();
    let result = index.integrity_check();
    assert!(
        !result.passed,
        "Should fail with unsorted binary file values"
    );
    let valid_check = result
        .checks
        .iter()
        .find(|c| c.name == "binary files valid")
        .expect("Should have a 'binary files valid' check");
    assert_eq!(valid_check.status, CheckStatus::Failed);
}

#[test]
fn test_integrity_check_after_strategy_b_compaction() {
    let tmp = TempDir::new().unwrap();
    // DeletionThreshold(0.9) means Strategy B is used when delete ratio < 90%
    let index = BoolStorage::new(
        tmp.path().to_path_buf(),
        DeletionThreshold::try_from(0.9).unwrap(),
    )
    .unwrap();

    // Insert 100 items and compact
    for i in 0..100u64 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    // Delete 5 items (5% < 90% threshold → Strategy B)
    for i in 0..5u64 {
        index.delete(i);
    }
    index.compact(2).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "Integrity check should pass after Strategy B compaction: {result:?}"
    );

    let deleted_check = result
        .checks
        .iter()
        .find(|c| c.name == "deleted not in postings")
        .expect("Should have a 'deleted not in postings' check");
    assert_eq!(deleted_check.status, CheckStatus::Ok);
    let details = deleted_check.details.as_deref().unwrap_or("");
    assert!(
        details.contains("Strategy B"),
        "Expected details to mention 'Strategy B', got: {details}"
    );
}

#[test]
fn test_integrity_check_after_strategy_b_then_insert_only_compact() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(
        tmp.path().to_path_buf(),
        DeletionThreshold::try_from(0.9).unwrap(),
    )
    .unwrap();

    // Insert 100 items and compact
    for i in 0..100u64 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    // Delete 5 items (5% < 90% → Strategy B)
    for i in 0..5u64 {
        index.delete(i);
    }
    index.compact(2).unwrap();

    // Insert 10 more items (no deletes) and compact again (PATH 1 carries forward deleted.bin)
    for i in 200..210u64 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(3).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "Integrity check should pass after PATH 1 carry-forward: {result:?}"
    );
}

#[test]
fn test_integrity_check_strategy_a_still_works() {
    let tmp = TempDir::new().unwrap();
    // Default threshold (0.1) means Strategy A is used when delete ratio > 10%
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert 10 items and compact
    for i in 0..10u64 {
        index.insert(&IndexedValue::Plain(true), i);
    }
    index.compact(1).unwrap();

    // Delete 3 items (30% > 10% → Strategy A)
    for i in 0..3u64 {
        index.delete(i);
    }
    index.compact(2).unwrap();

    let result = index.integrity_check();
    assert!(
        result.passed,
        "Integrity check should pass after Strategy A compaction: {result:?}"
    );

    let deleted_check = result
        .checks
        .iter()
        .find(|c| c.name == "deleted not in postings")
        .expect("Should have a 'deleted not in postings' check");
    assert_eq!(deleted_check.status, CheckStatus::Ok);
    let details = deleted_check.details.as_deref().unwrap_or("");
    assert_eq!(
        details, "OK",
        "Strategy A should produce 'OK' details, got: {details}"
    );
}

#[test]
fn test_insert_delete_reinsert_ordering() {
    let tmp = TempDir::new().unwrap();
    let index = BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

    // Insert doc 1 as true, delete it, then re-insert as false
    index.insert(&IndexedValue::Plain(true), 1);
    index.delete(1);
    index.insert(&IndexedValue::Plain(false), 1);

    // After collapsing, doc 1 should only be in false set
    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();
    assert_eq!(
        true_results,
        Vec::<u64>::new(),
        "true set should be empty after insert(true,1) → delete(1) → insert(false,1)"
    );
    assert_eq!(
        false_results,
        vec![1u64],
        "false set should contain only doc 1 after re-insert as false"
    );

    // Compact and verify ordering is preserved
    index.compact(1).unwrap();

    let true_results: Vec<u64> = index.filter(true).iter().collect();
    let false_results: Vec<u64> = index.filter(false).iter().collect();
    assert_eq!(
        true_results,
        Vec::<u64>::new(),
        "true set should remain empty after compaction"
    );
    assert_eq!(
        false_results,
        vec![1u64],
        "false set should still contain doc 1 after compaction"
    );
}
