//! Concurrency tests for the string (full-text search) module.
//!
//! Covers concurrent reads, concurrent writes, reads/writes during compaction,
//! compaction serialization, mixed operations stress, and snapshot isolation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use tempfile::TempDir;

use oramacore_fields::string::{
    BM25Scorer, IndexedValue, NoFilter, SearchParams, StringStorage, TermData, Threshold,
};

// ============================================================================
// Helpers
// ============================================================================

fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
    let mut term_map = HashMap::new();
    for (term, exact, stemmed) in terms {
        term_map.insert(term.to_string(), TermData::new(exact, stemmed));
    }
    IndexedValue::new(field_length, term_map)
}

fn search_count(index: &StringStorage, token: &str) -> usize {
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
    scorer.into_search_result().docs.len()
}

fn search_doc_ids_sorted(index: &StringStorage, token: &str) -> Vec<u64> {
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
    let result = scorer.into_search_result();
    let mut ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
    ids.sort();
    ids
}

// ============================================================================
// A. Concurrent Reads
// ============================================================================

#[test]
fn test_concurrent_reads_consistency() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Populate
    for i in 1..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    let num_readers = 8;
    let barrier = Arc::new(Barrier::new(num_readers));

    let handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let count = search_count(&index, "term");
                    assert_eq!(count, 100, "concurrent reader should see all 100 docs");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_reads_during_dirty_snapshot() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=50u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    let barrier = Arc::new(Barrier::new(5)); // 4 readers + 1 writer

    // 4 reader threads
    let mut handles: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let count = search_count(&index, "term");
                    // Must always see at least initial 50
                    assert!(
                        count >= 50,
                        "reader should always see >= 50 docs, got {count}"
                    );
                }
            })
        })
        .collect();

    // 1 writer thread
    {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 51..=100u64 {
                index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All 100 should be present at the end
    assert_eq!(search_count(&index, "term"), 100);
}

// ============================================================================
// B. Concurrent Writes
// ============================================================================

#[test]
fn test_concurrent_multi_thread_inserts() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    let num_threads = 4;
    let docs_per_thread = 250;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let base = (t * docs_per_thread) as u64 + 1;
                for i in 0..docs_per_thread as u64 {
                    index.insert(base + i, make_value(2, vec![("shared", vec![0], vec![])]));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let count = search_count(&index, "shared");
    assert_eq!(
        count,
        (num_threads * docs_per_thread),
        "all {0} docs should be present",
        num_threads * docs_per_thread,
    );
}

#[test]
fn test_concurrent_inserts_same_doc_id() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    let barrier = Arc::new(Barrier::new(2));

    let handles: Vec<_> = (0..2)
        .map(|t| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                // Both threads insert doc_id=1 with different terms
                let term = if t == 0 { "alpha" } else { "beta" };
                index.insert(1, make_value(2, vec![(term, vec![0], vec![])]));
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Doc should be present (no panic, no corruption)
    // It may match alpha, beta, or both depending on race
    let alpha_count = search_count(&index, "alpha");
    let beta_count = search_count(&index, "beta");
    assert!(
        alpha_count + beta_count >= 1,
        "doc_id=1 should be found under at least one term"
    );
}

// ============================================================================
// C. Reads During Compaction
// ============================================================================

#[test]
fn test_search_during_compaction() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    let pre_count = search_count(&index, "term");
    assert_eq!(pre_count, 100);

    let running = Arc::new(AtomicBool::new(true));

    // Reader thread: continuously reads during compaction
    let reader_handle = {
        let index = Arc::clone(&index);
        let running = Arc::clone(&running);
        thread::spawn(move || {
            let mut reads = 0u64;
            while running.load(Ordering::Relaxed) {
                let count = search_count(&index, "term");
                assert!(
                    count >= 100,
                    "reader should always see >= 100 docs during compaction, got {count}"
                );
                reads += 1;
            }
            reads
        })
    };

    // Compact on main thread
    index.compact(1).unwrap();

    running.store(false, Ordering::Relaxed);
    let reads = reader_handle.join().unwrap();
    assert!(reads > 0, "reader should have executed at least one search");

    // Post compaction
    assert_eq!(search_count(&index, "term"), 100);
}

#[test]
fn test_search_result_stability_across_version_swap() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    for i in 1..=50u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    // Capture result before compact
    let pre_result = {
        let tokens = vec!["term".to_string()];
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
        scorer.into_search_result()
    };
    assert_eq!(pre_result.docs.len(), 50);

    // Add more data and compact
    for i in 51..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    // Pre-captured result should still be valid (owned data)
    assert_eq!(
        pre_result.docs.len(),
        50,
        "pre-captured SearchResult should be unaffected by compaction"
    );

    // New search should see all 100
    assert_eq!(search_count(&index, "term"), 100);
}

// ============================================================================
// D. Writes During Compaction
// ============================================================================

#[test]
fn test_writes_preserved_during_compaction() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=50u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    let write_count = Arc::new(AtomicU64::new(0));

    // Writer thread: adds docs during compaction
    let writer_handle = {
        let index = Arc::clone(&index);
        let write_count = Arc::clone(&write_count);
        thread::spawn(move || {
            for i in 51..=150u64 {
                index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
                write_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    };

    // Compact while writes are happening
    index.compact(1).unwrap();
    writer_handle.join().unwrap();

    // Second compaction to capture all writes
    index.compact(2).unwrap();

    let count = search_count(&index, "term");
    assert_eq!(
        count, 150,
        "all 150 docs should be present after writes during compaction"
    );
}

#[test]
fn test_deletes_preserved_during_compaction() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    // Deleter thread: removes docs while compaction runs
    let deleter_handle = {
        let index = Arc::clone(&index);
        thread::spawn(move || {
            for i in 1..=50u64 {
                index.delete(i);
            }
        })
    };

    // Compact concurrently with deletes
    index.compact(1).unwrap();
    deleter_handle.join().unwrap();

    // Second compaction to apply the deletes
    index.compact(2).unwrap();

    let count = search_count(&index, "term");
    assert_eq!(count, 50, "50 docs should remain after concurrent deletes");
}

// ============================================================================
// E. Compaction Serialization
// ============================================================================

#[test]
fn test_concurrent_compaction_serialization() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    let version_counter = Arc::new(AtomicU64::new(1));
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            let version_counter = Arc::clone(&version_counter);
            thread::spawn(move || {
                barrier.wait();
                let v = version_counter.fetch_add(1, Ordering::SeqCst);
                // Some will succeed, some may get "same version" errors
                // if they see the same current version
                let _ = index.compact(v);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Data should be intact
    let count = search_count(&index, "term");
    assert_eq!(
        count, 100,
        "all docs should be present after concurrent compaction"
    );
}

// ============================================================================
// F. Mixed Operations Stress
// ============================================================================

#[test]
fn test_mixed_insert_delete_search() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    let barrier = Arc::new(Barrier::new(4)); // 1 inserter + 1 deleter + 2 searchers
    let mut handles = Vec::new();

    // Inserter
    {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 101..=200u64 {
                index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
            }
        }));
    }

    // Deleter (deletes some of the original docs)
    {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 1..=50u64 {
                index.delete(i);
            }
        }));
    }

    // 2 Searchers
    for _ in 0..2 {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..50 {
                let count = search_count(&index, "term");
                // Should always see a non-negative result count (no crash)
                assert!(count > 0, "search should return results during mixed ops");
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Final state: 51..=100 (survived deletes) + 101..=200 (inserted) = 150
    let final_ids = search_doc_ids_sorted(&index, "term");
    let expected: Vec<u64> = (51..=200).collect();
    assert_eq!(final_ids, expected, "final state should have docs 51..=200");
}

#[test]
fn test_high_write_contention() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    let num_threads = 16;
    let inserts_per_thread = 1000;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let base = (t * inserts_per_thread) as u64 + 1;
                for i in 0..inserts_per_thread as u64 {
                    index.insert(
                        base + i,
                        make_value(2, vec![("contention", vec![0], vec![])]),
                    );
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let count = search_count(&index, "contention");
    assert_eq!(
        count,
        num_threads * inserts_per_thread,
        "all {0} docs should be present after high-contention writes",
        num_threads * inserts_per_thread,
    );
}

// ============================================================================
// G. Snapshot Isolation
// ============================================================================

#[test]
fn test_snapshot_isolation_during_compaction() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Phase 1: Insert initial data and compact
    for i in 1..=50u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }
    index.compact(1).unwrap();

    // Phase 2: Insert more data (live layer only)
    for i in 51..=100u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    // Search sees both compacted + live
    assert_eq!(search_count(&index, "term"), 100);

    // Phase 3: Compact again
    index.compact(2).unwrap();

    // After compaction, search should still see all 100
    assert_eq!(
        search_count(&index, "term"),
        100,
        "after compaction all docs should still be visible"
    );

    // Phase 4: Insert even more, check snapshot isolation
    for i in 101..=150u64 {
        index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
    }

    // Should see all 150 (100 compacted + 50 live)
    assert_eq!(search_count(&index, "term"), 150);

    // Compact only captures up to snapshot point
    index.compact(3).unwrap();
    assert_eq!(search_count(&index, "term"), 150);
}
