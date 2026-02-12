#![cfg(feature = "alloc-tests")]
//! Allocation counting tests for the string_filter module.
//!
//! Uses a custom global allocator with thread-local counters to measure
//! the exact number of heap allocations during filter calls and iteration.

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

use oramacore_fields::string_filter::{
    FilterData, FilterIterator, IndexedValue, StringFilterStorage, Threshold,
};
use std::path::Path;

fn setup_index(dir: &Path, n: usize) -> StringFilterStorage {
    let index = StringFilterStorage::new(dir.to_path_buf(), Threshold::default()).unwrap();
    for i in 0..n {
        let key = if i % 3 == 0 {
            "apple"
        } else if i % 3 == 1 {
            "banana"
        } else {
            "cherry"
        };
        index.insert(&IndexedValue::Plain(key.to_string()), i as u64);
    }
    // Warm up snapshot
    let _ = index.filter("apple").iter().count();
    index
}

#[test]
fn test_alloc_count_filter_clean_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_data = index.filter("apple");
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(
        allocs, 0,
        "filter() on clean snapshot should be zero-alloc, got {allocs}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_count_iteration() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let filter_data = index.filter("apple");

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs, _) = stop_counting(baseline);

    // n=1000, every 3rd is apple: indices 0,3,6,...,999 = 334 items
    assert_eq!(count, 334, "Expected 334 apple items");
    assert_eq!(
        allocs, 0,
        "iter() should be zero-alloc (concrete types, no Box), got {allocs}"
    );
}

#[test]
fn test_alloc_count_filter_dirty_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Make snapshot dirty
    index.insert(&IndexedValue::Plain("apple".to_string()), 9999);

    let baseline = start_counting();
    let filter_data = index.filter("apple");
    let (allocs, _) = stop_counting(baseline);

    assert!(
        allocs > 0,
        "filter() on dirty snapshot should allocate (snapshot rebuild)"
    );
    assert!(
        allocs < 5000,
        "filter() on dirty snapshot allocated too much: {allocs}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_count_after_compaction() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();

    let baseline = start_counting();
    let filter_data = index.filter("apple");
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 334, "Expected 334 apple items after compaction");
    assert_eq!(
        allocs_filter, 0,
        "filter() after compaction should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 0,
        "iter() after compaction should be zero-alloc, got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_mixed_live_and_compacted() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();
    for i in 1000..1100 {
        let key = if i % 3 == 0 {
            "apple"
        } else if i % 3 == 1 {
            "banana"
        } else {
            "cherry"
        };
        index.insert(&IndexedValue::Plain(key.to_string()), i as u64);
    }
    // Warm up snapshot
    let _ = index.filter("apple").iter().count();

    let baseline = start_counting();
    let filter_data = index.filter("apple");
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let _count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(
        allocs_filter, 0,
        "filter() on clean mixed snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 0,
        "iter() on mixed data should be zero-alloc, got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_missing_key() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_data = index.filter("nonexistent");
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 0, "Expected 0 items for missing key");
    assert_eq!(
        allocs_filter, 0,
        "filter() for missing key should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 0,
        "iter() for missing key should be zero-alloc, got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_iterator_directly() {
    let baseline = start_counting();
    let count = FilterIterator::new(&[], &[], &[], &[]).count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 0, "Expected 0 items from empty slices");
    assert_eq!(
        allocs, 0,
        "FilterIterator::new from empty slices should be zero-alloc, got {allocs}"
    );
}

#[test]
fn test_alloc_count_iterator_large_slices() {
    let compacted: Vec<u64> = (0..500).map(|i| i * 2).collect();
    let live: Vec<u64> = (0..500).map(|i| i * 2 + 1).collect();

    let baseline = start_counting();
    let count = FilterIterator::new(&compacted, &live, &[], &[]).count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items from large slices");
    assert_eq!(
        allocs, 0,
        "FilterIterator::new from large slices should be zero-alloc, got {allocs}"
    );
}

#[test]
fn test_alloc_count_iterator_sizes() {
    let filter_size = std::mem::size_of::<FilterIterator>();
    let filter_data_size = std::mem::size_of::<FilterData>();

    assert!(
        filter_size <= 256,
        "FilterIterator is unexpectedly large: {filter_size} bytes"
    );
    assert!(
        filter_data_size <= 256,
        "FilterData is unexpectedly large: {filter_data_size} bytes"
    );
}
