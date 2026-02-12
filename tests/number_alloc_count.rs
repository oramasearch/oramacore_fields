#![cfg(feature = "alloc-tests")]
//! Allocation counting tests for the number module.
//!
//! Uses a custom global allocator with thread-local counters to measure
//! the exact number of heap allocations during filter/sort calls and iteration.

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

use oramacore_fields::number::{
    FilterIterator, FilterOp, IndexedValue, NumberStorage, SortIterator, SortOrder, Threshold,
};
use std::path::Path;

fn setup_index(dir: &Path, n: usize) -> NumberStorage<u64> {
    let index = NumberStorage::new(dir.to_path_buf(), Threshold::default()).unwrap();
    for i in 0..n {
        index
            .insert(&IndexedValue::Plain(i as u64), i as u64)
            .unwrap();
    }
    // Warm up snapshot
    let _ = index.filter(FilterOp::Gte(0)).iter().count();
    index
}

#[test]
fn test_alloc_count_filter_clean_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Gte(0));
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(
        allocs, 0,
        "filter() on clean snapshot should be zero-alloc, got {allocs}"
    );
    drop(filter_handle);
}

#[test]
fn test_alloc_count_filter_iteration() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let filter_handle = index.filter(FilterOp::Gte(0));

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items");
    assert_eq!(
        allocs, 1,
        "iter() should allocate exactly 1 (Box), got {allocs}"
    );
}

#[test]
fn test_alloc_count_sort_ascending() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let sort_handle = index.sort(SortOrder::Ascending);

    let baseline = start_counting();
    let count = sort_handle.iter().count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items");
    assert_eq!(
        allocs, 1,
        "sort(Ascending).iter() should allocate exactly 1 (Box), got {allocs}"
    );
}

#[test]
fn test_alloc_count_sort_descending() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let sort_handle = index.sort(SortOrder::Descending);

    let baseline = start_counting();
    let count = sort_handle.iter().count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items");
    assert_eq!(
        allocs, 1,
        "sort(Descending).iter() should allocate exactly 1 (Box), got {allocs}"
    );
}

#[test]
fn test_alloc_count_filter_dirty_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Make snapshot dirty
    index.insert(&IndexedValue::Plain(9999), 9999).unwrap();

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Gte(0));
    let (allocs, _) = stop_counting(baseline);

    assert!(
        allocs > 0,
        "filter() on dirty snapshot should allocate (snapshot rebuild)"
    );
    assert!(
        allocs < 200,
        "filter() on dirty snapshot allocated too much: {allocs}"
    );
    drop(filter_handle);
}

#[test]
fn test_alloc_count_after_compaction_filter() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();

    // After compaction, filter reads from mmap + empty live layer
    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Gte(0));
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items after compaction");
    assert_eq!(
        allocs_filter, 0,
        "filter() after compaction should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 1,
        "iter() after compaction should be exactly 1 alloc (Box), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_mixed_live_and_compacted() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();
    for i in 1000..1100 {
        index
            .insert(&IndexedValue::Plain(i as u64), i as u64)
            .unwrap();
    }
    // Warm up snapshot
    let _ = index.filter(FilterOp::Gte(0)).iter().count();

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Gte(0));
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 1100, "Expected 1100 items in mixed mode");
    assert_eq!(
        allocs_filter, 0,
        "filter() on clean mixed snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 1,
        "iter() on mixed data should be exactly 1 alloc (Box), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_filter_ops_eq() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Eq(500));
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 1, "Expected 1 item for Eq(500)");
    assert_eq!(
        allocs_filter, 0,
        "filter(Eq) on clean snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 1,
        "iter() for Eq should be exactly 1 alloc (Box), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_filter_ops_between() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::BetweenInclusive(100, 200));
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(
        count, 101,
        "Expected 101 items for BetweenInclusive(100, 200)"
    );
    assert_eq!(
        allocs_filter, 0,
        "filter(BetweenInclusive) on clean snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 1,
        "iter() for BetweenInclusive should be exactly 1 alloc (Box), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_f64_filter() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index: NumberStorage<f64> =
        NumberStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();
    for i in 0..100 {
        index
            .insert(
                &oramacore_fields::number::IndexedValue::Plain(i as f64),
                i as u64,
            )
            .unwrap();
    }
    // Warm up snapshot
    let _ = index.filter(FilterOp::Gte(0.0)).iter().count();

    let baseline = start_counting();
    let filter_handle = index.filter(FilterOp::Gte(0.0));
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_handle.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 100, "Expected 100 f64 items");
    assert_eq!(
        allocs_filter, 0,
        "f64 filter() on clean snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 1,
        "f64 iter() should be exactly 1 alloc (Box), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_iterator_sizes() {
    let filter_size = std::mem::size_of::<FilterIterator<u64>>();
    let sort_size = std::mem::size_of::<SortIterator<u64>>();

    assert!(
        filter_size <= 256,
        "FilterIterator<u64> is unexpectedly large: {filter_size} bytes"
    );
    assert!(
        sort_size <= 256,
        "SortIterator<u64> is unexpectedly large: {sort_size} bytes"
    );
}
