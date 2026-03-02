#![cfg(feature = "alloc-tests")]
//! Allocation counting tests for the geopoint module.
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

use oramacore_fields::geopoint::{
    FilterData, FilterIterator, GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold,
};
use std::path::Path;

fn bbox_all() -> GeoFilterOp {
    GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    }
}

fn setup_index(dir: &Path, n: usize) -> GeoPointStorage {
    let index = GeoPointStorage::new(dir.to_path_buf(), Threshold::default(), 10).unwrap();
    for i in 0..n {
        let lat = -80.0 + (i as f64) * 160.0 / n as f64;
        let lon = -170.0 + (i as f64) * 340.0 / n as f64;
        index.insert(
            IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
            i as u64,
        );
    }
    // Warm up snapshot
    let _ = index.filter(bbox_all()).iter().count();
    index
}

#[test]
fn test_alloc_count_filter_clean_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let baseline = start_counting();
    let filter_data = index.filter(bbox_all());
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(
        allocs, 0,
        "filter() on clean snapshot should be zero-alloc, got {allocs}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_count_iteration_live_only() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    let filter_data = index.filter(bbox_all());

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items");
    assert_eq!(
        allocs, 0,
        "iter() over live-only data should be zero-alloc, got {allocs}"
    );
}

#[test]
fn test_alloc_count_filter_dirty_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    // Make snapshot dirty
    index.insert(IndexedValue::Plain(GeoPoint::new(0.0, 0.0).unwrap()), 9999);

    let baseline = start_counting();
    let filter_data = index.filter(bbox_all());
    let (allocs, _) = stop_counting(baseline);

    assert!(
        allocs > 0,
        "filter() on dirty snapshot should allocate (snapshot rebuild)"
    );
    assert!(
        allocs < 2000,
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
    let filter_data = index.filter(bbox_all());
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(count, 1000, "Expected 1000 items after compaction");
    assert_eq!(
        allocs_filter, 0,
        "filter() after compaction should be zero-alloc, got {allocs_filter}"
    );
    // 1 segment = 1 alloc (Vec::with_capacity(32) for BKD stack)
    assert_eq!(
        allocs_iter, 1,
        "iter() with 1 compacted segment should be exactly 1 alloc (BKD stack Vec), got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_mixed_live_and_compacted() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();
    for i in 1000..1100 {
        let lat = -80.0 + (i as f64) * 160.0 / 1100.0;
        let lon = -170.0 + (i as f64) * 340.0 / 1100.0;
        index.insert(
            IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
            i as u64,
        );
    }
    // Warm up snapshot
    let _ = index.filter(bbox_all()).iter().count();

    let baseline = start_counting();
    let filter_data = index.filter(bbox_all());
    let (allocs_filter, _) = stop_counting(baseline);

    assert_eq!(
        allocs_filter, 0,
        "filter() on clean mixed snapshot should be zero-alloc, got {allocs_filter}"
    );
    drop(filter_data);
}

#[test]
fn test_alloc_count_radius_query() {
    let tmp = tempfile::TempDir::new().unwrap();
    let index = setup_index(tmp.path(), 1000);

    index.compact(1).unwrap();

    let center = GeoPoint::new(0.0, 0.0).unwrap();
    let op = GeoFilterOp::Radius {
        center,
        radius_meters: 20_000_000.0, // large enough to include most points
    };

    let baseline = start_counting();
    let filter_data = index.filter(op.clone());
    let (allocs_filter, _) = stop_counting(baseline);

    let baseline = start_counting();
    let _count = filter_data.iter().count();
    let (allocs_iter, _) = stop_counting(baseline);

    assert_eq!(
        allocs_filter, 0,
        "filter() for radius on clean snapshot should be zero-alloc, got {allocs_filter}"
    );
    // 1 segment = 1 alloc for BKD stack Vec
    assert_eq!(
        allocs_iter, 1,
        "iter() for radius with 1 segment should be exactly 1 alloc, got {allocs_iter}"
    );
}

#[test]
fn test_alloc_count_iterator_sizes() {
    let filter_iter_size = std::mem::size_of::<FilterIterator>();
    let filter_data_size = std::mem::size_of::<FilterData>();

    assert!(
        filter_iter_size <= 264,
        "FilterIterator is unexpectedly large: {filter_iter_size} bytes"
    );
    assert!(
        filter_data_size <= 264,
        "FilterData is unexpectedly large: {filter_data_size} bytes"
    );
}
