//! Integration tests for NumberStorage.

use oramacore_number_index::{
    CompactionMeta, F64Storage, FilterOp, IndexedValue, NumberIndexer, NumberStorage, SortOrder,
    Threshold, U64Storage,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::{f64, thread};
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
// Basic Operations
// ============================================================================

#[test]
fn test_basic_crud_u64() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.insert(&IndexedValue::Plain(40), 4).unwrap();
    index.insert(&IndexedValue::Plain(50), 5).unwrap();

    // Filter all
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=5, "basic crud filter all");
    assert_strictly_sorted(&results, "basic crud filter all");

    // Delete
    index.delete(3);
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![1, 2, 4, 5]);
    assert_strictly_sorted(&results, "basic crud after delete");
}

#[test]
fn test_basic_crud_f64() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert
    index.insert(&IndexedValue::Plain(-10.5), 1).unwrap();
    index.insert(&IndexedValue::Plain(0.0), 2).unwrap();
    index
        .insert(&IndexedValue::Plain(f64::consts::PI), 3)
        .unwrap();
    index.insert(&IndexedValue::Plain(100.0), 4).unwrap();

    // Filter
    let results: Vec<u64> = index.filter(FilterOp::Gte(0.0)).iter().collect();
    assert_eq!(results, vec![2, 3, 4]);
}

// ============================================================================
// Range Queries
// ============================================================================

#[test]
fn test_filter_eq() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.insert(&IndexedValue::Plain(20), 3).unwrap(); // Same value, different doc
    index.insert(&IndexedValue::Plain(30), 4).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(results, vec![2, 3]);
    assert_strictly_sorted(&results, "filter eq");
}

#[test]
fn test_filter_gt() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    let results: Vec<u64> = index.filter(FilterOp::Gt(50)).iter().collect();
    assert_eq!(results, vec![6, 7, 8, 9, 10]);
    assert_strictly_sorted(&results, "filter gt");
}

#[test]
fn test_filter_lt() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    let results: Vec<u64> = index.filter(FilterOp::Lt(50)).iter().collect();
    assert_eq!(results, vec![1, 2, 3, 4]);
    assert_strictly_sorted(&results, "filter lt");
}

#[test]
fn test_filter_between() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    let results: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(30, 70))
        .iter()
        .collect();
    assert_eq!(results, vec![3, 4, 5, 6, 7]);
    assert_strictly_sorted(&results, "filter between");
}

#[test]
fn test_f64_range_with_negatives() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(-100.0), 1).unwrap();
    index.insert(&IndexedValue::Plain(-50.0), 2).unwrap();
    index.insert(&IndexedValue::Plain(-10.0), 3).unwrap();
    index.insert(&IndexedValue::Plain(0.0), 4).unwrap();
    index.insert(&IndexedValue::Plain(10.0), 5).unwrap();
    index.insert(&IndexedValue::Plain(50.0), 6).unwrap();

    let results: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(-50.0, 10.0))
        .iter()
        .collect();
    assert_eq!(results, vec![2, 3, 4, 5]);
    assert_strictly_sorted(&results, "f64 range with negatives");
}

// ============================================================================
// Compaction
// ============================================================================

#[test]
fn test_compaction_preserves_data() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact
    index.compact(1).unwrap();

    // Verify all data is still present
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=100, "after compaction");
    assert_strictly_sorted(&results, "after compaction");
}

#[test]
fn test_compaction_applies_deletes() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert data
    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    // Delete some
    index.delete(2);
    index.delete(4);
    index.delete(6);

    // Compact
    index.compact(1).unwrap();

    // Verify deleted items are gone
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![1, 3, 5, 7, 8, 9, 10]);
}

#[test]
fn test_multiple_compactions() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // First batch
    for i in 1..=10 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(1).unwrap();

    // Second batch
    for i in 11..=20 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(2).unwrap();

    // Third batch
    for i in 21..=30 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }
    index.compact(3).unwrap();

    // Verify all data
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=30, "after multiple compactions");
    assert_strictly_sorted(&results, "after multiple compactions");
}

// ============================================================================
// Persistence
// ============================================================================

#[test]
fn test_persistence_across_open() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().to_path_buf();

    // Create and populate
    {
        let index: U64Storage = NumberStorage::new(path.clone(), Threshold::default()).unwrap();
        for i in 1..=50 {
            index.insert(&IndexedValue::Plain(i), i).unwrap();
        }
        index.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let index: U64Storage = NumberStorage::new(path, Threshold::default()).unwrap();
        let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
        assert_contains_exactly(&results, 1..=50, "after reopen");
        assert_strictly_sorted(&results, "after reopen");
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_doc_id_zero() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 0).unwrap(); // doc_id = 0
    index.insert(&IndexedValue::Plain(20), 1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [0, 1], "doc_id zero");
}

#[test]
fn test_large_doc_ids() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    let large_id = u64::MAX - 10;
    index.insert(&IndexedValue::Plain(10), large_id).unwrap();
    index.insert(&IndexedValue::Plain(20), u64::MAX).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [large_id, u64::MAX], "large doc_ids");
}

#[test]
fn test_empty_index() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert!(results.is_empty());

    // Compact empty index should work
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_delete_nonexistent() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.delete(999); // Non-existent doc

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![1]);
}

#[test]
fn test_duplicate_inserts() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(10), 1).unwrap(); // Duplicate
    index.insert(&IndexedValue::Plain(10), 1).unwrap(); // Another duplicate

    let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert_eq!(results, vec![1]); // Should be deduplicated
}

#[test]
fn test_f64_special_values() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index
        .insert(&IndexedValue::Plain(f64::NEG_INFINITY), 1)
        .unwrap();
    index.insert(&IndexedValue::Plain(-0.0), 2).unwrap();
    index.insert(&IndexedValue::Plain(0.0), 3).unwrap();
    index
        .insert(&IndexedValue::Plain(f64::INFINITY), 4)
        .unwrap();

    // NaN should be rejected
    let result = index.insert(&IndexedValue::Plain(f64::NAN), 5);
    assert!(
        matches!(result, Err(oramacore_number_index::Error::NaNNotAllowed)),
        "Expected NaNNotAllowed error for NaN value, got {result:?}"
    );

    // All non-NaN values should be present
    let results: Vec<u64> = index
        .filter(FilterOp::Gte(f64::NEG_INFINITY))
        .iter()
        .collect();
    assert_contains_exactly(&results, [1, 2, 3, 4], "f64 special values");
}

// ============================================================================
// Concurrency
// ============================================================================

#[test]
fn test_concurrent_reads() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Populate
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Concurrent reads
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for _ in 0..100 {
                    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
                    assert_eq!(
                        results.len(),
                        100,
                        "Expected 100 results, got {}",
                        results.len()
                    );
                    assert_strictly_sorted(&results, "concurrent reads");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_writes() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Concurrent writes from multiple threads
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..100 {
                    let doc_id = (t * 1000 + i) as u64;
                    let value = doc_id;
                    index.insert(&IndexedValue::Plain(value), doc_id).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all entries exist - 4 threads * 100 entries each
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(
        results.len(),
        400,
        "Expected 400 results from 4 threads * 100 entries, got {}",
        results.len()
    );

    // Verify uniqueness - no duplicates
    let result_set: HashSet<u64> = results.iter().copied().collect();
    assert_eq!(
        results.len(),
        result_set.len(),
        "Found duplicate doc_ids in results"
    );

    // Verify all expected doc_ids are present
    let expected: HashSet<u64> = (0..4)
        .flat_map(|t| (0..100).map(move |i| (t * 1000 + i) as u64))
        .collect();
    assert_eq!(
        result_set, expected,
        "Missing or unexpected doc_ids in results"
    );

    assert_strictly_sorted(&results, "concurrent writes");
}

#[test]
fn test_read_during_write() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Writer thread
    let index_writer = Arc::clone(&index);
    let writer = thread::spawn(move || {
        for i in 0..1000 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
        }
    });

    // Reader thread
    let index_reader = Arc::clone(&index);
    let reader = thread::spawn(move || {
        for _ in 0..100 {
            let _ = index_reader.filter(FilterOp::Gte(0)).iter().count();
        }
    });

    writer.join().unwrap();
    reader.join().unwrap();

    // Verify final state
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 0..1000, "read during write final state");
    assert_strictly_sorted(&results, "read during write final state");
}

#[test]
fn test_iterator_stability() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Start iteration
    let filter_data = index.filter(FilterOp::Gte(1));
    let mut iter = filter_data.iter();

    // Read first 10 items
    let mut first_10 = Vec::new();
    for _ in 0..10 {
        if let Some(doc_id) = iter.next() {
            first_10.push(doc_id);
        }
    }
    assert_eq!(first_10.len(), 10);

    // Modify index while iterator is active
    index.insert(&IndexedValue::Plain(200), 200).unwrap();
    index.delete(50);

    // Continue iteration - should see consistent view
    let remaining: Vec<u64> = iter.collect();

    // Total should be 100 (original count)
    assert_eq!(
        first_10.len() + remaining.len(),
        100,
        "Iterator should see exactly 100 items"
    );

    // Verify no duplicates
    let mut all_items: Vec<u64> = first_10.clone();
    all_items.extend(remaining);
    let unique_count = all_items
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert_eq!(unique_count, 100, "Iterator should return unique doc_ids");
}

#[test]
fn test_read_during_compaction() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert data
    for i in 1..=1000 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Start reader thread
    let index_reader = Arc::clone(&index);
    let reader = thread::spawn(move || {
        for _ in 0..50 {
            let results: Vec<u64> = index_reader.filter(FilterOp::Gte(1)).iter().collect();
            // Should always see all 1000 entries (compaction preserves data)
            assert_eq!(
                results.len(),
                1000,
                "Read during compaction should see all 1000 entries, got {}",
                results.len()
            );
            assert_strictly_sorted(&results, "read during compaction");
        }
    });

    // Compact in main thread
    index.compact(1).unwrap();

    reader.join().unwrap();

    // Verify final state
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=1000, "after compaction final state");
    assert_strictly_sorted(&results, "after compaction final state");
}

#[test]
fn test_write_during_compaction() {
    let temp = TempDir::new().unwrap();
    let index = Arc::new(
        NumberStorage::<u64>::new(temp.path().to_path_buf(), Threshold::default()).unwrap(),
    );

    // Insert initial data
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact while writing
    let index_writer = Arc::clone(&index);
    let writer = thread::spawn(move || {
        for i in 101..=200 {
            index_writer.insert(&IndexedValue::Plain(i), i).unwrap();
        }
    });

    index.compact(1).unwrap();

    writer.join().unwrap();

    // All writes should be preserved
    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 1..=200, "write during compaction");
    assert_strictly_sorted(&results, "write during compaction");
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_large_dataset() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert 10K entries
    for i in 0..10_000u64 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Compact
    index.compact(1).unwrap();

    // Verify all data
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(
        results.len(),
        10_000,
        "Expected 10,000 entries after large dataset insert, got {}",
        results.len()
    );
    assert_strictly_sorted(&results, "large dataset");

    // Spot-check: verify some expected values are present
    assert!(results.contains(&0), "Should contain doc_id 0");
    assert!(results.contains(&5000), "Should contain doc_id 5000");
    assert!(results.contains(&9999), "Should contain doc_id 9999");

    // Range query
    let range: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(5000, 6000))
        .iter()
        .collect();
    assert_eq!(
        range.len(),
        1001,
        "Range query [5000, 6000] should return 1001 entries, got {}",
        range.len()
    );
    assert_contains_exactly(&range, 5000..=6000, "large dataset range query");
    assert_strictly_sorted(&range, "large dataset range query");
}

#[test]
fn test_high_delete_ratio() {
    let temp = TempDir::new().unwrap();
    // Low threshold so deletes get applied during compaction
    let threshold = Threshold::try_new(0.05).unwrap();
    let index: U64Storage = NumberStorage::new(temp.path().to_path_buf(), threshold).unwrap();

    // Insert
    for i in 1..=100 {
        index.insert(&IndexedValue::Plain(i), i).unwrap();
    }

    // Delete 20% (above 5% threshold)
    for i in 1..=20 {
        index.delete(i);
    }

    // Compact - should apply deletes
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(1)).iter().collect();
    assert_contains_exactly(&results, 21..=100, "high delete ratio");
}

// ============================================================================
// Bucket Split Tests (Integration)
// ============================================================================

/// Helper to count bucket files in a version directory
fn count_bucket_files(base_path: &std::path::Path, offset: u64) -> usize {
    let version_dir = base_path.join("versions").join(offset.to_string());
    let mut count = 0;
    for i in 0u32.. {
        let data_path = version_dir.join(format!("data_{i:04}.dat"));
        if data_path.exists() {
            count += 1;
        } else {
            break;
        }
    }
    count
}

#[test]
fn test_bucket_split_via_numberindex() {
    let temp = TempDir::new().unwrap();

    // Use NumberStorage::new_with_config with small bucket size
    let index: NumberStorage<u64> = NumberStorage::new_with_config(
        temp.path().to_path_buf(),
        Threshold::default(),
        1,  // index_stride
        50, // bucket_target_bytes - small to force splits
    )
    .unwrap();

    // Insert entries
    for i in 1..=10u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    // Compact to trigger bucket splitting
    index.compact(1).unwrap();

    // Verify multiple buckets were created
    let bucket_count = count_bucket_files(temp.path(), 1);
    assert!(
        bucket_count > 1,
        "Expected multiple buckets, got {bucket_count}"
    );

    // Verify all data is accessible
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=10, "bucket split all data");

    // Verify exact key lookup works across buckets
    let result = index.filter(FilterOp::Eq(50)).iter().collect::<Vec<_>>();
    assert_eq!(result, vec![5]);

    // Verify range query works across buckets
    let range: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(30, 70))
        .iter()
        .collect();
    assert_eq!(range, vec![3, 4, 5, 6, 7]);
    assert_strictly_sorted(&range, "bucket split range query");
}

#[test]
fn test_bucket_split_with_deletes() {
    let temp = TempDir::new().unwrap();

    // Low delete threshold so deletes are applied
    let threshold = Threshold::try_new(0.05).unwrap();
    let index: NumberStorage<u64> = NumberStorage::new_with_config(
        temp.path().to_path_buf(),
        threshold,
        1,  // index_stride
        80, // bucket_target_bytes
    )
    .unwrap();

    // Insert entries
    for i in 1..=20u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    // Delete some entries
    index.delete(5);
    index.delete(10);
    index.delete(15);

    // Compact
    index.compact(1).unwrap();

    // Verify buckets exist
    let bucket_count = count_bucket_files(temp.path(), 1);
    assert!(bucket_count >= 1);

    // Verify deleted entries are not returned
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert!(!results.contains(&5));
    assert!(!results.contains(&10));
    assert!(!results.contains(&15));

    // Verify remaining entries are accessible
    assert!(results.contains(&1));
    assert!(results.contains(&20));
    assert_eq!(results.len(), 17); // 20 - 3 deleted
}

#[test]
fn test_bucket_split_multiple_compactions() {
    let temp = TempDir::new().unwrap();

    let index: NumberStorage<u64> = NumberStorage::new_with_config(
        temp.path().to_path_buf(),
        Threshold::default(),
        1,
        60, // Small bucket
    )
    .unwrap();

    // First batch
    for i in 1..=5u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    let bucket_count_v1 = count_bucket_files(temp.path(), 1);
    assert!(bucket_count_v1 >= 1);

    // Second batch
    for i in 6..=10u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(2).unwrap();

    let bucket_count_v2 = count_bucket_files(temp.path(), 2);
    assert!(bucket_count_v2 >= bucket_count_v1);

    // Verify all data is accessible
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=10, "bucket split multiple compactions");
}

#[test]
fn test_bucket_split_persistence() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().to_path_buf();

    // Create index with small buckets and insert data
    {
        let index: NumberStorage<u64> = NumberStorage::new_with_config(
            path.clone(),
            Threshold::default(),
            1,
            50, // Small bucket
        )
        .unwrap();

        for i in 1..=10u64 {
            index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
        }
        index.compact(1).unwrap();

        // Verify buckets created
        let bucket_count = count_bucket_files(&path, 1);
        assert!(bucket_count > 1);
    }

    // Reopen and verify data
    {
        let index: NumberStorage<u64> =
            NumberStorage::new_with_config(path.clone(), Threshold::default(), 1, 50).unwrap();

        let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
        assert_contains_exactly(&results, 1..=10, "bucket split persistence");

        // Range query should still work
        let range: Vec<u64> = index
            .filter(FilterOp::BetweenInclusive(40, 70))
            .iter()
            .collect();
        assert_eq!(range, vec![4, 5, 6, 7]);
        assert_strictly_sorted(&range, "bucket split persistence range query");
    }
}

#[test]
fn test_bucket_split_f64_via_numberindex() {
    let temp = TempDir::new().unwrap();

    let index: F64Storage =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 1, 50)
            .unwrap();

    // Insert f64 values
    index.insert(&IndexedValue::Plain(-100.0), 1).unwrap();
    index.insert(&IndexedValue::Plain(-50.5), 2).unwrap();
    index.insert(&IndexedValue::Plain(0.0), 3).unwrap();
    index.insert(&IndexedValue::Plain(50.5), 4).unwrap();
    index.insert(&IndexedValue::Plain(100.0), 5).unwrap();

    index.compact(1).unwrap();

    // Verify buckets
    let bucket_count = count_bucket_files(temp.path(), 1);
    assert!(bucket_count > 1, "Expected multiple buckets for f64 index");

    // Verify all data
    let results: Vec<u64> = index
        .filter(FilterOp::Gte(f64::NEG_INFINITY))
        .iter()
        .collect();
    assert_contains_exactly(&results, [1, 2, 3, 4, 5], "bucket split f64 all data");

    // Verify range queries
    let range: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(-60.0, 60.0))
        .iter()
        .collect();
    assert_eq!(range, vec![2, 3, 4]);
    assert_strictly_sorted(&range, "bucket split f64 range query");
}

// ============================================================================
// Sort Tests
// ============================================================================

#[test]
fn test_sort_ascending_basic() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert values in random order
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();

    let results: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    // doc_ids should be ordered by value: 10 -> 1, 20 -> 2, 30 -> 3
    assert_eq!(results, vec![1, 2, 3]);
}

#[test]
fn test_sort_descending_basic() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert values in random order
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();

    let results: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();
    // doc_ids should be ordered by value descending: 30 -> 3, 20 -> 2, 10 -> 1
    assert_eq!(results, vec![3, 2, 1]);
}

#[test]
fn test_sort_empty_index() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    assert!(asc.is_empty());
    assert!(desc.is_empty());
}

#[test]
fn test_sort_with_deletes() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.delete(2);

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    assert_eq!(asc, vec![1, 3]);
    assert_eq!(desc, vec![3, 1]);
}

#[test]
fn test_sort_merges_live_and_compacted() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert and compact
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.compact(1).unwrap();

    // Insert more to live layer
    index.insert(&IndexedValue::Plain(20), 2).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    assert_eq!(asc, vec![1, 2, 3]);
    assert_eq!(desc, vec![3, 2, 1]);
}

#[test]
fn test_sort_same_value_multiple_docs() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Multiple docs with same value
    index.insert(&IndexedValue::Plain(100), 1).unwrap();
    index.insert(&IndexedValue::Plain(100), 2).unwrap();
    index.insert(&IndexedValue::Plain(100), 3).unwrap();
    index.insert(&IndexedValue::Plain(200), 4).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    // Same value: ordered by doc_id in ascending, reversed in descending
    assert_eq!(asc, vec![1, 2, 3, 4]);
    assert_eq!(desc, vec![4, 3, 2, 1]);
}

#[test]
fn test_sort_f64_with_negatives() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(-10.0), 1).unwrap();
    index.insert(&IndexedValue::Plain(0.0), 2).unwrap();
    index.insert(&IndexedValue::Plain(10.0), 3).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    // -10.0 < 0.0 < 10.0
    assert_eq!(asc, vec![1, 2, 3]);
    assert_eq!(desc, vec![3, 2, 1]);
}

#[test]
fn test_sort_after_compaction_with_deletes() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.insert(&IndexedValue::Plain(30), 3).unwrap();
    index.delete(2);
    index.compact(1).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    assert_eq!(asc, vec![1, 3]);
    assert_eq!(desc, vec![3, 1]);
}

#[test]
fn test_sort_large_dataset() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert 1000 values in random-ish order
    for i in (0..1000).rev() {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    assert_eq!(asc.len(), 1000);
    assert_eq!(desc.len(), 1000);

    // Verify ascending order
    for i in 0..1000 {
        assert_eq!(asc[i as usize], i);
    }

    // Verify descending order
    for i in 0..1000 {
        assert_eq!(desc[i as usize], 999 - i);
    }
}

#[test]
fn test_sort_across_multiple_buckets() {
    let temp = TempDir::new().unwrap();

    // Use small bucket size to force multiple buckets
    let index: U64Storage =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 1, 50)
            .unwrap();

    // Insert enough values to span multiple buckets
    for i in 0..20 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();

    // Verify order is correct across bucket boundaries
    let expected_asc: Vec<u64> = (0..20).collect();
    let expected_desc: Vec<u64> = (0..20).rev().collect();

    assert_eq!(asc, expected_asc);
    assert_eq!(desc, expected_desc);
}

// ============================================================================
// Incremental Compaction Tests
// ============================================================================

/// Helper to read meta.bin from a version directory
fn read_meta(base_path: &std::path::Path, offset: u64) -> CompactionMeta {
    let meta_path = base_path
        .join("versions")
        .join(offset.to_string())
        .join("meta.bin");
    if !meta_path.exists() {
        return CompactionMeta::default();
    }
    let data = std::fs::read(&meta_path).unwrap();
    if data.len() < 16 {
        return CompactionMeta::default();
    }
    CompactionMeta {
        changes_since_rebuild: u64::from_ne_bytes(data[0..8].try_into().unwrap()),
        total_at_rebuild: u64::from_ne_bytes(data[8..16].try_into().unwrap()),
    }
}

#[test]
fn test_incremental_single_insert_on_large_index() {
    let temp = TempDir::new().unwrap();
    // Use small buckets to ensure multiple buckets and ability to do incremental
    let index: NumberStorage<u64> = NumberStorage::new_with_config(
        temp.path().to_path_buf(),
        Threshold::default(),
        100,  // index_stride
        4096, // bucket_target_bytes - moderate size
    )
    .unwrap();

    // Insert 10K entries and compact (full rebuild)
    for i in 0..10_000u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    // Verify full rebuild wrote meta
    let meta1 = read_meta(temp.path(), 1);
    assert_eq!(meta1.changes_since_rebuild, 0);
    assert!(meta1.total_at_rebuild > 0);

    let bucket_count_v1 = count_bucket_files(temp.path(), 1);
    assert!(
        bucket_count_v1 > 1,
        "Need multiple buckets for incremental test, got {bucket_count_v1}"
    );

    // Insert 1 entry in the middle and compact (should take incremental path)
    index.insert(&IndexedValue::Plain(50005), 99999).unwrap();
    index.compact(2).unwrap();

    // Verify metadata shows incremental (changes > 0)
    let meta2 = read_meta(temp.path(), 2);
    assert_eq!(meta2.changes_since_rebuild, 1);
    assert_eq!(meta2.total_at_rebuild, meta1.total_at_rebuild);

    // Verify same bucket count (incremental doesn't change bucket count)
    let bucket_count_v2 = count_bucket_files(temp.path(), 2);
    assert_eq!(
        bucket_count_v1, bucket_count_v2,
        "Incremental compaction should preserve bucket count"
    );

    // Verify data integrity: all 10001 entries queryable
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 10_001);
    assert!(results.contains(&99999));

    // Verify the new entry is findable via filter
    let eq_results: Vec<u64> = index.filter(FilterOp::Eq(50005)).iter().collect();
    assert!(eq_results.contains(&99999));

    // Verify sort works
    let sorted: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    assert_eq!(sorted.len(), 10_001);
}

#[test]
fn test_incremental_compaction_matches_full_rewrite() {
    let temp_inc = TempDir::new().unwrap();
    let temp_full = TempDir::new().unwrap();

    // Create two identical indexes
    let index_inc: NumberStorage<u64> = NumberStorage::new_with_config(
        temp_inc.path().to_path_buf(),
        Threshold::default(),
        100,
        4096,
    )
    .unwrap();

    let index_full: NumberStorage<u64> = NumberStorage::new_with_config(
        temp_full.path().to_path_buf(),
        Threshold::default(),
        100,
        4096,
    )
    .unwrap();

    // Populate both with same data
    for i in 0..5000u64 {
        index_inc.insert(&IndexedValue::Plain(i * 10), i).unwrap();
        index_full.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index_inc.compact(1).unwrap();
    index_full.compact(1).unwrap();

    // Insert same data into both
    for i in 5000..5010u64 {
        index_inc.insert(&IndexedValue::Plain(i * 10), i).unwrap();
        index_full.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }

    // Compact incremental (index_inc) - should take incremental path
    index_inc.compact(2).unwrap();

    // Force full rewrite on index_full by setting dirty threshold to 0
    // Actually we can't easily force full rewrite from outside,
    // but the full index with same data should produce the same query results.
    index_full.compact(2).unwrap();

    // Compare query results
    for filter_op in [
        FilterOp::Gte(0),
        FilterOp::Eq(50000),
        FilterOp::BetweenInclusive(25000, 25100),
        FilterOp::Lt(100),
        FilterOp::Gt(49900),
    ] {
        let results_inc: Vec<u64> = index_inc.filter(filter_op).iter().collect();
        let results_full: Vec<u64> = index_full.filter(filter_op).iter().collect();
        assert_eq!(
            results_inc, results_full,
            "Mismatch for filter op {filter_op:?}"
        );
    }

    // Compare sort results
    let sort_inc: Vec<u64> = index_inc.sort(SortOrder::Ascending).iter().collect();
    let sort_full: Vec<u64> = index_full.sort(SortOrder::Ascending).iter().collect();
    assert_eq!(sort_inc, sort_full, "Sort results differ");
}

#[test]
fn test_dirty_header_threshold_triggers_full_rebuild() {
    let temp = TempDir::new().unwrap();

    let mut index: NumberStorage<u64> =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 100, 4096)
            .unwrap();

    // Set a very low dirty threshold so full rebuild triggers quickly
    index.set_dirty_header_threshold(0.01); // 1%

    // Initial data + compact (full rebuild)
    for i in 0..1000u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    let meta1 = read_meta(temp.path(), 1);
    assert_eq!(meta1.changes_since_rebuild, 0);
    assert_eq!(meta1.total_at_rebuild, 1000);

    // Insert 20 entries (2% > 1% threshold). This should trigger full rebuild.
    for i in 1000..1020u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(2).unwrap();

    // Full rebuild should reset metadata
    let meta2 = read_meta(temp.path(), 2);
    assert_eq!(
        meta2.changes_since_rebuild, 0,
        "Expected full rebuild to reset changes_since_rebuild"
    );
    assert_eq!(meta2.total_at_rebuild, 1020);

    // Verify data integrity
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 1020);
}

#[test]
fn test_deletes_only_fast_path() {
    let temp = TempDir::new().unwrap();
    let index: NumberStorage<u64> = NumberStorage::new_with_config(
        temp.path().to_path_buf(),
        Threshold::default(), // High threshold so deletes are carried forward
        100,
        4096,
    )
    .unwrap();

    // Insert data and compact
    for i in 0..100u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    let bucket_count_v1 = count_bucket_files(temp.path(), 1);

    // Delete some entries (no inserts)
    index.delete(5);
    index.delete(10);
    index.delete(15);

    // Compact again (should take deletes-only fast path)
    index.compact(2).unwrap();

    // Bucket count should be the same (files were copied)
    let bucket_count_v2 = count_bucket_files(temp.path(), 2);
    assert_eq!(bucket_count_v1, bucket_count_v2);

    // Compare data file contents (should be identical since they were copied)
    for i in 0..bucket_count_v1 {
        let path_v1 = temp
            .path()
            .join("versions")
            .join("1")
            .join(format!("data_{i:04}.dat"));
        let path_v2 = temp
            .path()
            .join("versions")
            .join("2")
            .join(format!("data_{i:04}.dat"));
        let data_v1 = std::fs::read(&path_v1).unwrap();
        let data_v2 = std::fs::read(&path_v2).unwrap();
        assert_eq!(data_v1, data_v2, "Data file {i} content differs");
    }

    // Header should also be identical
    let header_v1 = std::fs::read(temp.path().join("versions/1/header.idx")).unwrap();
    let header_v2 = std::fs::read(temp.path().join("versions/2/header.idx")).unwrap();
    assert_eq!(header_v1, header_v2, "Header content differs");

    // Deleted entries should not appear in results
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 97);
    assert!(!results.contains(&5));
    assert!(!results.contains(&10));
    assert!(!results.contains(&15));
}

#[test]
fn test_meta_bin_persistence() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().to_path_buf();

    // Create index, compact, verify meta
    {
        let index: NumberStorage<u64> =
            NumberStorage::new_with_config(path.clone(), Threshold::default(), 100, 4096).unwrap();

        for i in 0..1000u64 {
            index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
        }
        index.compact(1).unwrap();

        let meta = read_meta(&path, 1);
        assert_eq!(meta.changes_since_rebuild, 0);
        assert_eq!(meta.total_at_rebuild, 1000);
    }

    // Reopen and verify meta is loaded correctly
    {
        let index: NumberStorage<u64> =
            NumberStorage::new_with_config(path.clone(), Threshold::default(), 100, 4096).unwrap();

        // Insert a few entries and compact (should take incremental path)
        index.insert(&IndexedValue::Plain(5005), 9001).unwrap();
        index.compact(2).unwrap();

        let meta = read_meta(&path, 2);
        assert_eq!(meta.changes_since_rebuild, 1);
        assert_eq!(meta.total_at_rebuild, 1000);

        // Verify data integrity
        let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
        assert_eq!(results.len(), 1001);
        assert!(results.contains(&9001));
    }
}

#[test]
fn test_incremental_compaction_multiple_rounds() {
    let temp = TempDir::new().unwrap();
    let index: NumberStorage<u64> =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 100, 4096)
            .unwrap();

    // Initial data
    for i in 0..5000u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    let bucket_count = count_bucket_files(temp.path(), 1);

    // Multiple incremental rounds
    for round in 0..5u64 {
        let doc_id = 10_000 + round;
        index
            .insert(&IndexedValue::Plain((2500 + round) * 10), doc_id)
            .unwrap();
        index.compact(2 + round).unwrap();

        // Verify bucket count stays the same
        let new_bucket_count = count_bucket_files(temp.path(), 2 + round);
        assert_eq!(bucket_count, new_bucket_count);

        // Verify data integrity
        let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
        assert_eq!(results.len(), 5001 + round as usize);
    }

    // Verify meta tracks cumulative changes
    let meta = read_meta(temp.path(), 6);
    assert_eq!(meta.changes_since_rebuild, 5);
    assert_eq!(meta.total_at_rebuild, 5000);
}

#[test]
fn test_incremental_insert_at_beginning() {
    let temp = TempDir::new().unwrap();
    let index: NumberStorage<u64> =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 100, 4096)
            .unwrap();

    // Insert data starting from 1000
    for i in 100..5000u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    // Insert at the very beginning (key=5, below all existing keys)
    // This affects bucket 0 but not others
    index.insert(&IndexedValue::Plain(5), 99999).unwrap();
    index.compact(2).unwrap();

    // Verify data integrity
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 4901);
    assert!(results.contains(&99999));

    // Verify the new entry is findable
    let eq_results: Vec<u64> = index.filter(FilterOp::Eq(5)).iter().collect();
    assert_eq!(eq_results, vec![99999]);
}

#[test]
fn test_incremental_insert_at_end() {
    let temp = TempDir::new().unwrap();
    let index: NumberStorage<u64> =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 100, 4096)
            .unwrap();

    // Insert data up to 50000
    for i in 0..5000u64 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    // Insert at the very end (key beyond all existing)
    index.insert(&IndexedValue::Plain(999999), 88888).unwrap();
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results.len(), 5001);
    assert!(results.contains(&88888));

    let eq_results: Vec<u64> = index.filter(FilterOp::Eq(999999)).iter().collect();
    assert_eq!(eq_results, vec![88888]);
}

#[test]
fn test_incremental_with_f64() {
    let temp = TempDir::new().unwrap();
    let index: NumberStorage<f64> =
        NumberStorage::new_with_config(temp.path().to_path_buf(), Threshold::default(), 100, 4096)
            .unwrap();

    // Insert f64 data
    for i in 0..5000u64 {
        index
            .insert(&IndexedValue::Plain(i as f64 * 0.1), i)
            .unwrap();
    }
    index.compact(1).unwrap();

    let bucket_count_v1 = count_bucket_files(temp.path(), 1);

    // Insert one f64 in the middle
    index.insert(&IndexedValue::Plain(250.05), 99999).unwrap();
    index.compact(2).unwrap();

    let bucket_count_v2 = count_bucket_files(temp.path(), 2);
    assert_eq!(bucket_count_v1, bucket_count_v2);

    let results: Vec<u64> = index
        .filter(FilterOp::Gte(f64::NEG_INFINITY))
        .iter()
        .collect();
    assert_eq!(results.len(), 5001);
    assert!(results.contains(&99999));
}

// ============================================================================
// Delete-then-Reinsert Tests
// ============================================================================

#[test]
fn test_delete_reinsert_filter_live_only() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert doc 1 at value 10
    index.insert(&IndexedValue::Plain(10), 1).unwrap();

    // Delete doc 1, then re-insert at value 20
    index.delete(1);
    index.insert(&IndexedValue::Plain(20), 1).unwrap();

    // Filter should find doc 1 at value 20 (not at value 10)
    let results_eq_10: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert!(
        !results_eq_10.contains(&1),
        "doc 1 should NOT appear at old value 10"
    );

    let results_eq_20: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(
        results_eq_20,
        vec![1],
        "doc 1 should appear at new value 20"
    );

    let results_all: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results_all, vec![1], "doc 1 should appear exactly once");
}

#[test]
fn test_delete_reinsert_filter_after_compaction() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert and compact
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.compact(1).unwrap();

    // Delete doc 1 and re-insert at a new value
    index.delete(1);
    index.insert(&IndexedValue::Plain(30), 1).unwrap();

    // Before compaction: should see doc 1 at value 30
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [1, 2], "before second compaction");

    let eq_30: Vec<u64> = index.filter(FilterOp::Eq(30)).iter().collect();
    assert_eq!(eq_30, vec![1]);

    let eq_10: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert!(eq_10.is_empty(), "doc 1 should not appear at old value 10");

    // After compaction: should still see doc 1 at value 30
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [1, 2], "after second compaction");

    let eq_30: Vec<u64> = index.filter(FilterOp::Eq(30)).iter().collect();
    assert_eq!(eq_30, vec![1]);

    let eq_10: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert!(
        eq_10.is_empty(),
        "doc 1 should not appear at old value 10 after compaction"
    );
}

#[test]
fn test_delete_reinsert_sort() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert docs
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.insert(&IndexedValue::Plain(30), 3).unwrap();

    // Delete doc 1 (value 10) and re-insert at value 25 (between doc 2 and doc 3)
    index.delete(1);
    index.insert(&IndexedValue::Plain(25), 1).unwrap();

    // Ascending sort: should be doc 2 (20), doc 1 (25), doc 3 (30)
    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    assert_eq!(asc, vec![2, 1, 3]);

    // Descending sort: doc 3 (30), doc 1 (25), doc 2 (20)
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();
    assert_eq!(desc, vec![3, 1, 2]);
}

#[test]
fn test_delete_reinsert_survives_two_compactions() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert and compact
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.compact(1).unwrap();

    // Delete + re-insert + compact
    index.delete(1);
    index.insert(&IndexedValue::Plain(30), 1).unwrap();
    index.compact(2).unwrap();

    // Verify after first compact
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [1, 2], "after compact 2");

    // Compact again (no changes) — should be stable
    index.compact(3).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, [1, 2], "after compact 3");

    let eq_30: Vec<u64> = index.filter(FilterOp::Eq(30)).iter().collect();
    assert_eq!(
        eq_30,
        vec![1],
        "doc 1 should be at value 30 after 2 compactions"
    );

    let eq_10: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert!(eq_10.is_empty(), "doc 1 should not appear at old value 10");
}

#[test]
fn test_delete_reinsert_multiple_doc_ids() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert several docs
    for i in 1..=5 {
        index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
    }
    index.compact(1).unwrap();

    // Delete and re-insert docs 1, 3, 5 at new values
    index.delete(1);
    index.insert(&IndexedValue::Plain(100), 1).unwrap();
    index.delete(3);
    index.insert(&IndexedValue::Plain(200), 3).unwrap();
    index.delete(5);
    index.insert(&IndexedValue::Plain(300), 5).unwrap();

    // Before compaction
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=5, "before compaction with re-inserts");

    // After compaction
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_contains_exactly(&results, 1..=5, "after compaction with re-inserts");

    // Verify new positions
    assert_eq!(
        index.filter(FilterOp::Eq(100)).iter().collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(
        index.filter(FilterOp::Eq(200)).iter().collect::<Vec<_>>(),
        vec![3]
    );
    assert_eq!(
        index.filter(FilterOp::Eq(300)).iter().collect::<Vec<_>>(),
        vec![5]
    );

    // Old positions should be empty
    assert!(index
        .filter(FilterOp::Eq(10))
        .iter()
        .collect::<Vec<_>>()
        .is_empty());
    assert!(index
        .filter(FilterOp::Eq(30))
        .iter()
        .collect::<Vec<_>>()
        .is_empty());
    assert!(index
        .filter(FilterOp::Eq(50))
        .iter()
        .collect::<Vec<_>>()
        .is_empty());
}

#[test]
fn test_insert_then_delete_no_reinsert() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    // Simple insert + delete (no re-insert)
    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 2).unwrap();
    index.delete(1);

    // Doc 1 should be gone
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![2]);

    // After compaction, still gone
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![2]);

    // After another compaction
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![2]);
}

// ============================================================================
// NumberIndexer + NumberStorage integration tests
// ============================================================================

#[test]
fn test_indexer_plain_u64_update() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(false);

    let val = indexer.index_json(&serde_json::json!(42)).unwrap();
    index.insert(&val, 1).unwrap();

    let val = indexer.index_json(&serde_json::json!(100)).unwrap();
    index.insert(&val, 2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Eq(42)).iter().collect();
    assert_eq!(results, vec![1]);

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn test_indexer_plain_f64_update() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<f64>::new(false);

    let val = indexer
        .index_json(&serde_json::json!(f64::consts::PI))
        .unwrap();
    index.insert(&val, 1).unwrap();

    let val = indexer.index_json(&serde_json::json!(-1.5)).unwrap();
    index.insert(&val, 2).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0.0)).iter().collect();
    assert_eq!(results, vec![1]);

    let results: Vec<u64> = index.filter(FilterOp::Lte(0.0)).iter().collect();
    assert_eq!(results, vec![2]);
}

#[test]
fn test_same_doc_id_multiple_values_filter() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&IndexedValue::Plain(10), 1).unwrap();
    index.insert(&IndexedValue::Plain(20), 1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert_eq!(results, vec![1]);

    let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(results, vec![1]);

    let results: Vec<u64> = index.filter(FilterOp::Eq(30)).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_indexer_array_u64_update() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(true);

    // Doc 1 has values [10, 20, 30]
    let val = indexer
        .index_json(&serde_json::json!([10, 20, 30]))
        .unwrap();
    index.insert(&val, 1).unwrap();

    // Doc 2 has values [20, 40]
    let val = indexer.index_json(&serde_json::json!([20, 40])).unwrap();
    index.insert(&val, 2).unwrap();

    // Eq 10 → only doc 1
    let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert_eq!(results, vec![1]);

    // Eq 20 → both docs
    let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(results, vec![1, 2]);

    // Eq 40 → only doc 2
    let results: Vec<u64> = index.filter(FilterOp::Eq(40)).iter().collect();
    assert_eq!(results, vec![2]);

    // Gte 30 → doc 1 (has 30) and doc 2 (has 40)
    let results: Vec<u64> = index.filter(FilterOp::Gte(30)).iter().collect();
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn test_indexer_array_f64_update() {
    let temp = TempDir::new().unwrap();
    let index: F64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<f64>::new(true);

    let val = indexer
        .index_json(&serde_json::json!([1.5, -0.5, 3.0]))
        .unwrap();
    index.insert(&val, 1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Eq(-0.5)).iter().collect();
    assert_eq!(results, vec![1]);

    let results: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(1.0, 2.0))
        .iter()
        .collect();
    assert_eq!(results, vec![1]);
}

#[test]
fn test_indexer_update_then_compact() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(true);

    // Insert array values via indexer
    let val = indexer
        .index_json(&serde_json::json!([10, 20, 30]))
        .unwrap();
    index.insert(&val, 1).unwrap();

    let val = indexer.index_json(&serde_json::json!([20, 40])).unwrap();
    index.insert(&val, 2).unwrap();

    // Compact
    index.compact(1).unwrap();

    // Data should survive compaction
    let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(results, vec![1, 2]);

    let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert_eq!(results, vec![1]);

    // Add more after compaction
    let val = indexer.index_json(&serde_json::json!([50, 60])).unwrap();
    index.insert(&val, 3).unwrap();

    index.compact(2).unwrap();

    // Gte(40): doc 2 has one value >= 40, doc 3 has two values >= 40
    // Multi-value docs appear once per matching entry
    let results: Vec<u64> = index.filter(FilterOp::Gte(40)).iter().collect();
    assert_eq!(results, vec![2, 3, 3]);
}

#[test]
fn test_indexer_update_then_delete() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(true);

    let val = indexer
        .index_json(&serde_json::json!([10, 20, 30]))
        .unwrap();
    index.insert(&val, 1).unwrap();

    let val = indexer.index_json(&serde_json::json!([20, 40])).unwrap();
    index.insert(&val, 2).unwrap();

    // Delete doc 1 → should remove all its entries (10, 20, 30)
    index.delete(1);

    let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
    assert_eq!(results, vec![2]);

    let results: Vec<u64> = index.filter(FilterOp::Eq(10)).iter().collect();
    assert!(results.is_empty());

    // Compact and verify — doc 2 has two entries (20 and 40), both match Gte(0)
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert_eq!(results, vec![2, 2]);
}

#[test]
fn test_indexer_empty_array() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(true);

    let val = indexer.index_json(&serde_json::json!([])).unwrap();
    index.insert(&val, 1).unwrap();

    // Empty array should produce no entries
    let results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_indexer_invalid_json_returns_none() {
    let indexer = NumberIndexer::<u64>::new(false);

    assert!(indexer.index_json(&serde_json::json!("hello")).is_none());
    assert!(indexer.index_json(&serde_json::json!(true)).is_none());
    assert!(indexer.index_json(&serde_json::json!(null)).is_none());

    let array_indexer = NumberIndexer::<u64>::new(true);
    assert!(array_indexer.index_json(&serde_json::json!(42)).is_none());
    assert!(array_indexer.index_json(&serde_json::json!("hi")).is_none());
}

#[test]
fn test_indexer_sort_with_array_values() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let indexer = NumberIndexer::<u64>::new(true);

    // Doc 1 appears at values 10 and 30
    let val = indexer.index_json(&serde_json::json!([10, 30])).unwrap();
    index.insert(&val, 1).unwrap();

    // Doc 2 appears at value 20
    let val = indexer.index_json(&serde_json::json!([20])).unwrap();
    index.insert(&val, 2).unwrap();

    // Ascending sort: doc 1 (at 10), doc 2 (at 20), doc 1 (at 30)
    let asc: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    assert_eq!(asc, vec![1, 2, 1]);

    // Descending sort: doc 1 (at 30), doc 2 (at 20), doc 1 (at 10)
    let desc: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();
    assert_eq!(desc, vec![1, 2, 1]);
}

#[test]
fn test_indexer_mixed_plain_and_array_workflow() {
    let temp = TempDir::new().unwrap();
    let index: U64Storage =
        NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
    let plain_indexer = NumberIndexer::<u64>::new(false);
    let array_indexer = NumberIndexer::<u64>::new(true);

    // Some docs have plain values, some have arrays
    let val = plain_indexer.index_json(&serde_json::json!(50)).unwrap();
    index.insert(&val, 1).unwrap();

    let val = array_indexer
        .index_json(&serde_json::json!([10, 50, 90]))
        .unwrap();
    index.insert(&val, 2).unwrap();

    let val = plain_indexer.index_json(&serde_json::json!(90)).unwrap();
    index.insert(&val, 3).unwrap();

    // Eq 50 → docs 1 and 2
    let results: Vec<u64> = index.filter(FilterOp::Eq(50)).iter().collect();
    assert_eq!(results, vec![1, 2]);

    // Eq 90 → docs 2 and 3
    let results: Vec<u64> = index.filter(FilterOp::Eq(90)).iter().collect();
    assert_eq!(results, vec![2, 3]);

    // Compact and verify
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter(FilterOp::Eq(50)).iter().collect();
    assert_eq!(results, vec![1, 2]);
}
