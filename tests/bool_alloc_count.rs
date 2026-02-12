#![cfg(feature = "alloc-tests")]
//! Allocation counting test for filter operations.
//!
//! Uses a custom global allocator with thread-local counters to measure
//! the exact number of heap allocations during filter() calls and iteration.
//! Thread-local counting ensures only the test thread's allocations are
//! measured, so tests can run in parallel without cross-thread interference.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// A counting allocator that wraps the system allocator.
/// Uses thread-local counters so only the test thread's allocations are measured,
/// eliminating noise from the test runner and other threads.
struct CountingAlloc;

thread_local! {
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static DEALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // try_with avoids panicking if TLS is destroyed during thread shutdown.
        // const-init thread locals don't allocate on access, so no re-entrancy risk.
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

/// Start counting allocations. Returns (allocs, deallocs) at the start.
fn start_counting() -> (usize, usize) {
    let a = ALLOC_COUNT.with(|c| c.get());
    let d = DEALLOC_COUNT.with(|c| c.get());
    ACTIVE.with(|a| a.set(true));
    (a, d)
}

/// Stop counting and return (allocs, deallocs) that happened during the counted region.
fn stop_counting(baseline: (usize, usize)) -> (usize, usize) {
    ACTIVE.with(|a| a.set(false));
    let a = ALLOC_COUNT.with(|c| c.get()) - baseline.0;
    let d = DEALLOC_COUNT.with(|c| c.get()) - baseline.1;
    (a, d)
}

use oramacore_fields::bool::{BoolStorage, DeletionThreshold, IndexedValue};
use std::path::PathBuf;

fn setup_index(dir: &std::path::Path, n: usize) -> BoolStorage {
    let index = BoolStorage::new(PathBuf::from(dir), DeletionThreshold::default()).unwrap();
    for i in 0..n {
        index.insert(&IndexedValue::Plain(i as u64 % 2 == 0), i as u64);
    }
    // First filter to warm up snapshot
    let _ = index.filter(true).iter().count();
    index
}

#[test]
fn test_alloc_count_filter_clean_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Snapshot is clean after warm-up, so filter() should be zero-alloc.
    let baseline = start_counting();
    let filter_data = index.filter(true);
    let (allocs, _deallocs) = stop_counting(baseline);

    assert_eq!(
        allocs, 0,
        "filter() on clean snapshot should be zero-alloc, got {allocs}"
    );

    // Use filter_data so it's not dropped in the counted region
    drop(filter_data);
}

#[test]
fn test_alloc_count_iteration() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let filter_data = index.filter(true);

    // Count allocations during iteration only
    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs, _deallocs) = stop_counting(baseline);

    assert_eq!(
        count, 500,
        "Expected 500 true items from 1000 alternating inserts"
    );
    assert!(
        allocs <= 5,
        "iter() should need at most a few allocations, got {allocs}"
    );
}

#[test]
fn test_alloc_count_iteration_descending() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let filter_data = index.filter(true);

    let baseline = start_counting();
    let count = filter_data
        .sorted(oramacore_fields::bool::SortOrder::Descending)
        .count();
    let (allocs, _deallocs) = stop_counting(baseline);

    assert_eq!(
        count, 500,
        "Expected 500 true items from 1000 alternating inserts"
    );
    assert!(
        allocs <= 5,
        "sorted(Descending) iteration should need at most a few allocations, got {allocs}"
    );
}

#[test]
fn test_alloc_count_filter_dirty_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Make snapshot dirty by inserting
    index.insert(&IndexedValue::Plain(true), 9999);

    let baseline = start_counting();
    let filter_data = index.filter(true);
    let (allocs, _deallocs) = stop_counting(baseline);

    // Dirty snapshot requires rebuilding HashSets → Vecs, so allocations are expected.
    // But they should stay bounded relative to the number of ops.
    assert!(
        allocs > 0,
        "filter() on dirty snapshot should allocate (snapshot rebuild)"
    );
    assert!(
        allocs < 200,
        "filter() on dirty snapshot (1001 ops) allocated too much: {allocs}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_count_after_compaction() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();

    // After compaction, filter reads from mmap + empty live layer
    let baseline = start_counting();
    let filter_data = index.filter(true);
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 500, "Expected 500 true items after compaction");
    assert_eq!(
        allocs_filter, 0,
        "filter() after compaction (empty live layer) should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 0,
        "iter() over mmap-only data should be zero-alloc, got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_mixed_live_and_compacted() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Compact, then add more live data
    index.compact(1).unwrap();
    for i in 1000..1100 {
        index.insert(&IndexedValue::Plain(i as u64 % 2 == 0), i as u64);
    }
    // Warm up snapshot
    let _ = index.filter(true).iter().count();

    // Now measure filter + iteration with both compacted and live data
    let baseline = start_counting();
    let filter_data = index.filter(true);
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 550, "Expected 500 + 50 true items in mixed mode");
    assert_eq!(
        allocs_filter, 0,
        "filter() on clean mixed snapshot should be zero-alloc, got {allocs_filter}"
    );
    assert_eq!(
        allocs_iter, 0,
        "iter() over mixed data should be zero-alloc, got {allocs_iter}"
    );
}

// --- Targeted descending investigation ---

#[test]
fn test_alloc_descending_sorted_ascending_via_sorted() {
    // Does calling sorted(Ascending) allocate? (to isolate the enum wrapper)
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);
    let filter_data = index.filter(true);

    let baseline = start_counting();
    let count = filter_data
        .sorted(oramacore_fields::bool::SortOrder::Ascending)
        .count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 500, "Expected 500 true items");
    assert!(
        allocs <= 5,
        "sorted(Ascending) should need at most a few allocations, got {allocs}"
    );
}

#[test]
fn test_alloc_descending_iterator_directly() {
    // Use DescendingIterator::new directly, bypassing the SortedIterator enum

    use oramacore_fields::bool::DescendingIterator;

    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);
    let filter_data = index.filter(true);

    // We can't access the internals of FilterData, so use the enum path.
    // Instead, construct a DescendingIterator from raw slices.
    let compacted = [1u64, 3, 5, 7, 9];
    let live = [2u64, 4, 6, 8, 10];
    let cdel: [u64; 0] = [];
    let ldel: [u64; 0] = [];

    let baseline = start_counting();
    let count = DescendingIterator::new(&compacted, &live, &cdel, &ldel).count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 10, "Expected 10 items from raw slices");
    assert_eq!(
        allocs, 0,
        "DescendingIterator from raw slices should be zero-alloc, got {allocs}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_descending_large_slices() {
    // DescendingIterator with large slices
    use oramacore_fields::bool::DescendingIterator;

    let compacted: Vec<u64> = (0..500).map(|i| i * 2).collect();
    let live: Vec<u64> = (0..500).map(|i| i * 2 + 1).collect();
    let cdel: [u64; 0] = [];
    let ldel: [u64; 0] = [];

    let baseline = start_counting();
    let count = DescendingIterator::new(&compacted, &live, &cdel, &ldel).count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items from large slices");
    assert!(
        allocs <= 10,
        "DescendingIterator from large slices should be near-zero-alloc, got {allocs}"
    );
}

#[test]
fn test_alloc_descending_print_sizes() {
    use oramacore_fields::bool::{DescendingIterator, FilterIterator, SortedIterator};

    // Verify iterator types are stack-sized (no heap-allocated inner state).
    // These sizes should not grow unexpectedly.
    let filter_size = std::mem::size_of::<FilterIterator>();
    let descending_size = std::mem::size_of::<DescendingIterator>();
    let sorted_size = std::mem::size_of::<SortedIterator>();

    assert!(
        filter_size <= 256,
        "FilterIterator is unexpectedly large: {filter_size} bytes"
    );
    assert!(
        descending_size <= 256,
        "DescendingIterator is unexpectedly large: {descending_size} bytes"
    );
    assert!(
        sorted_size <= 256,
        "SortedIterator is unexpectedly large: {sorted_size} bytes"
    );
}

#[test]
fn test_alloc_descending_via_filter_data_only_construction() {
    // Measure JUST the construction of the descending iterator (no iteration)
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);
    let filter_data = index.filter(true);

    let baseline = start_counting();
    let iter = filter_data.sorted(oramacore_fields::bool::SortOrder::Descending);
    let (allocs_construct, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = iter.count();
    let (allocs_iterate, _) = stop_counting(baseline);

    assert_eq!(count, 500, "Expected 500 true items");
    assert!(
        allocs_construct <= 5,
        "sorted(Descending) construction should be near-zero-alloc, got {allocs_construct}"
    );
    assert!(
        allocs_iterate <= 5,
        "sorted(Descending) iteration should be near-zero-alloc, got {allocs_iterate}"
    );
}
