#![cfg(feature = "alloc-tests")]
//! Allocation counting tests for the string search module.
//!
//! Uses a custom global allocator with thread-local counters to measure
//! the exact number of heap allocations during search calls.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static DEALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ACTIVE.try_with(|active| {
            if active.get() {
                let _ = ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = ACTIVE.try_with(|active| {
            if active.get() {
                let _ = DEALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn start_counting() -> (usize, usize) {
    let a = ALLOC_COUNT.with(|c| c.get());
    let d = DEALLOC_COUNT.with(|c| c.get());
    ACTIVE.with(|a| a.set(true));
    (a, d)
}

fn stop_counting(baseline: (usize, usize)) -> (usize, usize) {
    ACTIVE.with(|a| a.set(false));
    let a = ALLOC_COUNT.with(|c| c.get()) - baseline.0;
    let d = DEALLOC_COUNT.with(|c| c.get()) - baseline.1;
    (a, d)
}

use oramacore_fields::string::{
    BM25u64Scorer, IndexedValue, SearchParams, SegmentConfig, StringStorage, TermData,
};
use std::collections::HashMap;
use std::path::Path;

fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
    let mut term_map = HashMap::new();
    for (term, exact, stemmed) in terms {
        term_map.insert(term.to_string(), TermData::new(exact, stemmed));
    }
    IndexedValue::new(field_length, term_map)
}

fn setup_index(dir: &Path, n: usize) -> StringStorage {
    let index = StringStorage::new(dir.to_path_buf(), SegmentConfig::default()).unwrap();
    for i in 0..n {
        let term = if i % 3 == 0 {
            "apple"
        } else if i % 3 == 1 {
            "banana"
        } else {
            "cherry"
        };
        index.insert(i as u64, make_value(3, vec![(term, vec![0], vec![1, 2])]));
    }
    index
}

/// After compaction, search should only allocate for BM25u64Scorer's internal HashMap
/// (not for PostingEntry Vec<u32> or search_terms Vec/String).
#[test]
fn test_search_alloc_after_compact() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 100);
    index.compact(1).unwrap();

    // Warm up snapshot so the first search doesn't trigger snapshot rebuild
    let tokens = vec!["apple".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    drop(scorer);

    // Now measure allocations for a real search
    let baseline = start_counting();
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(result.docs.len(), 34, "Expected 34 apple docs");
    // Only BM25u64Scorer HashMap + per_doc_ntf HashMap + result Vec should allocate.
    // With 34 matching docs this should be very few allocations (HashMap internals).
    // The key improvement: no 2*N PostingEntry Vec<u32> allocations.
    assert!(
        allocs < 20,
        "search() after compaction allocated too much: {allocs} (expected < 20 for 34 docs)"
    );
}

/// Live-only search (no compaction). per_doc_ntf HashMap + scorer are the only allocators.
#[test]
fn test_search_alloc_clean_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 100);

    // Warm up snapshot
    let tokens = vec!["apple".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    drop(scorer);

    let baseline = start_counting();
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(result.docs.len(), 34, "Expected 34 apple docs");
    assert!(
        allocs < 20,
        "search() on clean snapshot allocated too much: {allocs} (expected < 20 for 34 docs)"
    );
}

/// Prefix search uses callback path and should not allocate Vec/String per match.
#[test]
fn test_search_alloc_prefix() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    // Insert terms that share a prefix
    for i in 0..50u64 {
        index.insert(i, make_value(3, vec![("apple", vec![0], vec![1, 2])]));
    }
    for i in 50..100u64 {
        index.insert(i, make_value(3, vec![("application", vec![0], vec![1, 2])]));
    }
    index.compact(1).unwrap();

    // Warm up
    let tokens = vec!["app".to_string()];
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                tolerance: None,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    drop(scorer);

    let baseline = start_counting();
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                tolerance: None,
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(result.docs.len(), 100, "Expected 100 docs matching 'app*'");
    // Should not allocate String per FST match or Vec per PostingEntry
    assert!(
        allocs < 30,
        "prefix search allocated too much: {allocs} (expected < 30 for 100 docs)"
    );
}

/// Phrase boost path allocates position Vecs but NOT PostingEntry Vec<u32>.
#[test]
fn test_search_alloc_phrase_boost() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = StringStorage::new(tmp.path().to_path_buf(), SegmentConfig::default()).unwrap();

    for i in 0..50u64 {
        index.insert(
            i,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
    }
    index.compact(1).unwrap();

    let tokens = vec!["hello".to_string(), "world".to_string()];

    // Warm up
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                phrase_boost: Some(2.0),
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    drop(scorer);

    let baseline = start_counting();
    let mut scorer = BM25u64Scorer::new();
    index
        .search(
            &SearchParams {
                tokens: &tokens,
                phrase_boost: Some(2.0),
                ..Default::default()
            },
            &mut scorer,
        )
        .unwrap();
    let result = scorer.into_search_result();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(result.docs.len(), 50, "Expected 50 docs");
    // Phrase boost needs position maps (Vec<u32> per doc per token), plus
    // HashMap internals. Should still be much less than the old 2*N*tokens
    // PostingEntry allocations.
    assert!(
        allocs < 25,
        "phrase boost search allocated too much: {allocs} (expected < 25 for 50 docs, 2 tokens)"
    );
}
