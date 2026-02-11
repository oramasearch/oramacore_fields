use oramacore_geopoint_index::{GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_index() -> (TempDir, Arc<GeoPointStorage>) {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 10).unwrap());
    (tmp, index)
}

fn bbox_all() -> GeoFilterOp {
    GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    }
}

/// Deterministic point from doc_id: spreads across the globe on a grid.
fn point_for_id(id: u64) -> GeoPoint {
    let lat = ((id % 179) as f64) - 89.0; // -89 .. 89
    let lon = ((id % 359) as f64) - 179.0; // -179 .. 179
    GeoPoint::new(lat, lon).unwrap()
}

/// Northern hemisphere point: lat in 1..89.
fn north_point(id: u64) -> GeoPoint {
    let lat = 1.0 + (id % 88) as f64; // 1 .. 88
    let lon = ((id % 359) as f64) - 179.0;
    GeoPoint::new(lat, lon).unwrap()
}

/// Southern hemisphere point: lat in -89..-1.
fn south_point(id: u64) -> GeoPoint {
    let lat = -89.0 + (id % 88) as f64; // -89 .. -2
    let lon = ((id % 359) as f64) - 179.0;
    GeoPoint::new(lat, lon).unwrap()
}

fn collect_sorted(index: &GeoPointStorage, op: GeoFilterOp) -> Vec<u64> {
    let mut v: Vec<u64> = index.filter(op).iter().collect();
    v.sort_unstable();
    v
}

// ===========================================================================
// Category 1: Double-Check Pattern
// ===========================================================================

/// 8 readers hit the dirty snapshot simultaneously while the main thread
/// inserts a doc, triggering the read → dirty check → write lock → double-check
/// path in filter().
#[test]
fn test_dirty_snapshot_race() {
    let (_tmp, index) = make_index();

    // Seed with one point so snapshot exists
    index.insert(IndexedValue::Plain(point_for_id(0)), 0);

    let barrier = Arc::new(Barrier::new(9)); // 8 readers + 1 main

    let mut handles = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let results: Vec<u64> = idx.filter(bbox_all()).iter().collect();
                // Must always see at least doc_id 0
                assert!(!results.is_empty(), "reader saw empty results");
            }
        }));
    }

    // Main thread: insert a new doc to dirty the snapshot, then release barrier
    barrier.wait();
    index.insert(IndexedValue::Plain(point_for_id(1)), 1);

    for h in handles {
        h.join().unwrap();
    }

    // Final state: both 0 and 1 present
    let results = collect_sorted(&index, bbox_all());
    assert!(results.contains(&0));
    assert!(results.contains(&1));
}

/// 4 readers + 1 writer. Writer inserts 200 items with yields.
/// Readers observe monotonically non-decreasing counts.
#[test]
fn test_snapshot_dirty_clean_cycle() {
    let (_tmp, index) = make_index();

    let stop = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicU64::new(0));

    // Writer
    let idx_w = Arc::clone(&index);
    let stop_w = Arc::clone(&stop);
    let written_w = Arc::clone(&written);
    let writer = thread::spawn(move || {
        for i in 0u64..200 {
            idx_w.insert(IndexedValue::Plain(point_for_id(i)), i);
            written_w.store(i + 1, Ordering::Release);
            thread::yield_now();
        }
        stop_w.store(true, Ordering::Release);
    });

    // 4 readers — each tracks that count never decreases
    let mut readers = Vec::new();
    for _ in 0..4 {
        let idx_r = Arc::clone(&index);
        let stop_r = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            let mut prev_count = 0usize;
            while !stop_r.load(Ordering::Acquire) {
                let count = idx_r.filter(bbox_all()).iter().count();
                assert!(
                    count >= prev_count,
                    "count went backwards: {prev_count} -> {count}"
                );
                prev_count = count;
            }
        }));
    }

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }

    // Final: all 200 present
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 200);
}

// ===========================================================================
// Category 2: Compaction Serialization
// ===========================================================================

/// 4 threads attempt compaction simultaneously. The compaction_lock serializes
/// them internally. Data integrity must be maintained.
#[test]
fn test_concurrent_compactions() {
    let (_tmp, index) = make_index();

    for i in 0u64..50 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(4));
    let success_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let sc = Arc::clone(&success_count);
        handles.push(thread::spawn(move || {
            bar.wait();
            // Each thread uses a different version_id
            if idx.compact(t + 1).is_ok() {
                sc.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // At least 1 compaction succeeded (all should, since they're serialized)
    assert!(success_count.load(Ordering::SeqCst) >= 1);

    // All 50 points intact
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 50);
}

/// 1 writer inserts continuously while main thread compacts multiple times.
/// All written doc_ids must be present after final compaction.
#[test]
fn test_high_write_rate_with_compact() {
    let (_tmp, index) = make_index();

    // Seed initial data
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let total_written = Arc::new(AtomicU64::new(100));

    let idx_w = Arc::clone(&index);
    let stop_w = Arc::clone(&stop);
    let tw = Arc::clone(&total_written);
    let writer = thread::spawn(move || {
        let mut i = 100u64;
        while !stop_w.load(Ordering::Relaxed) && i < 500 {
            idx_w.insert(IndexedValue::Plain(point_for_id(i)), i);
            tw.store(i + 1, Ordering::Release);
            i += 1;
            thread::yield_now();
        }
    });

    // Main: compact 5 times
    for version_id in 1..=5u64 {
        index.compact(version_id).unwrap();
        thread::yield_now();
    }

    stop.store(true, Ordering::Release);
    writer.join().unwrap();

    // Final compaction to capture everything
    index.compact(6).unwrap();

    let tw_final = total_written.load(Ordering::Acquire) as usize;
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), tw_final);
}

/// Single-threaded: 5 rounds of (insert 20 + compact). All 100 present.
#[test]
fn test_interleaved_write_compact() {
    let (_tmp, index) = make_index();

    for round in 0u64..5 {
        let base = round * 20;
        for i in 0..20 {
            index.insert(IndexedValue::Plain(point_for_id(base + i)), base + i);
        }
        index.compact(round + 1).unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 100);
    for i in 0u64..100 {
        assert!(results.contains(&i), "missing doc_id {i}");
    }
}

// ===========================================================================
// Category 3: Version Swap Atomicity
// ===========================================================================

/// 8 readers query continuously while main compacts. Readers must always see
/// at least the pre-compaction count — never a partial/torn state.
#[test]
fn test_filter_during_version_swap() {
    let (_tmp, index) = make_index();

    // Insert 100 items and compact
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(1).unwrap();

    // Insert 100 more (will be in live layer)
    for i in 100u64..200 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let count = idx.filter(bbox_all()).iter().count();
                // Must always see at least the 100 compacted items
                assert!(
                    count >= 100,
                    "reader saw only {count} items, expected >= 100"
                );
            }
        }));
    }

    // Compact — this triggers version swap
    index.compact(2).unwrap();

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }

    // Final: all 200
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 200);
}

/// FilterData captured before compaction still returns original results even
/// after version swap. Proves snapshot isolation via Arc.
#[test]
fn test_filter_data_arc_stability() {
    let (_tmp, index) = make_index();

    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    // Capture filter data BEFORE compaction
    let filter_before = index.filter(bbox_all());

    // Compact and add more
    index.compact(1).unwrap();
    for i in 100u64..200 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    // Old filter still sees 100
    let old_results: Vec<u64> = filter_before.iter().collect();
    assert_eq!(old_results.len(), 100);

    // New filter sees 200
    let new_results: Vec<u64> = index.filter(bbox_all()).iter().collect();
    assert_eq!(new_results.len(), 200);
}

// ===========================================================================
// Category 4: Iterator Stability
// ===========================================================================

/// 1 consumer holds a FilterData, 2 writers and 1 compactor mutate the index.
/// The consumer's iterator must yield a consistent snapshot.
#[test]
fn test_iterator_under_stress() {
    let (_tmp, index) = make_index();

    // Insert baseline
    for i in 0u64..200 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    // Capture filter data (snapshot)
    let filter = index.filter(bbox_all());

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(4)); // 2 writers + 1 compactor + 1 signal

    // 2 writers
    let mut handles = Vec::new();
    for t in 0..2u64 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let base = 1000 + t * 500;
            for i in 0..500 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                idx.insert(IndexedValue::Plain(point_for_id(base + i)), base + i);
                thread::yield_now();
            }
        }));
    }

    // 1 compactor
    {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for version_id in 1..=3u64 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                let _ = idx.compact(version_id);
                thread::yield_now();
            }
        }));
    }

    // Release all mutators
    barrier.wait();

    // Consume the snapshot iterator while mutations happen
    let snapshot_results: Vec<u64> = filter.iter().collect();

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // Snapshot should see exactly 200 (the baseline)
    assert_eq!(
        snapshot_results.len(),
        200,
        "iterator snapshot count mismatch"
    );
}

/// 8 threads each create a FilterData and consume it. All must see >= baseline.
#[test]
fn test_multiple_concurrent_iterators() {
    let (_tmp, index) = make_index();

    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let count = idx.filter(bbox_all()).iter().count();
                assert!(count >= 100, "expected >= 100, got {count}");
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ===========================================================================
// Category 5: Clear-After-Swap
// ===========================================================================

/// Writer inserts new items during compaction. Those items must survive in
/// the live layer after compaction completes.
#[test]
fn test_writes_preserved_during_compaction() {
    let (_tmp, index) = make_index();

    // Baseline: 100 items
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let idx_w = Arc::clone(&index);
    let stop_w = Arc::clone(&stop);

    let writer = thread::spawn(move || {
        let mut i = 100u64;
        while !stop_w.load(Ordering::Relaxed) && i < 200 {
            idx_w.insert(IndexedValue::Plain(point_for_id(i)), i);
            i += 1;
            thread::yield_now();
        }
        i // return how many were written total
    });

    // Compact while writer is active
    index.compact(1).unwrap();

    stop.store(true, Ordering::Relaxed);
    let total = writer.join().unwrap() as usize;

    // A second compaction to pick up everything
    index.compact(2).unwrap();

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), total, "expected {total} items in index");
}

/// Deleter removes items during compaction. Those deletes must be effective.
#[test]
fn test_deletes_preserved_during_compaction() {
    let (_tmp, index) = make_index();

    // Insert 200 items and compact
    for i in 0u64..200 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let deleted_count = Arc::new(AtomicU64::new(0));

    let idx_d = Arc::clone(&index);
    let stop_d = Arc::clone(&stop);
    let dc = Arc::clone(&deleted_count);
    let deleter = thread::spawn(move || {
        // Delete even ids 0..100
        for i in (0u64..100).step_by(2) {
            if stop_d.load(Ordering::Relaxed) {
                break;
            }
            idx_d.delete(i);
            dc.fetch_add(1, Ordering::Relaxed);
            thread::yield_now();
        }
    });

    // Compact while deleter runs
    index.compact(2).unwrap();

    stop.store(true, Ordering::Relaxed);
    deleter.join().unwrap();

    // Second compaction to finalize
    index.compact(3).unwrap();

    let results = collect_sorted(&index, bbox_all());
    let dc_final = deleted_count.load(Ordering::Relaxed) as usize;
    assert_eq!(results.len(), 200 - dc_final);

    // Verify deleted items are absent
    for i in (0u64..100).step_by(2) {
        if i < (dc_final as u64 * 2) {
            assert!(!results.contains(&i), "doc_id {i} should be deleted");
        }
    }
}

// ===========================================================================
// Category 6: Multi-Thread Inserts/Deletes
// ===========================================================================

/// 8 threads insert 100 docs each into disjoint ranges. All 800 must be present.
#[test]
fn test_concurrent_multi_thread_inserts() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for t in 0..8u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let base = t * 100;
            for i in 0..100 {
                let id = base + i;
                idx.insert(IndexedValue::Plain(point_for_id(id)), id);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 800);
    for i in 0u64..800 {
        assert!(results.contains(&i), "missing doc_id {i}");
    }
}

/// 4 threads insert northern-hemisphere points (even ids), 4 threads insert
/// southern-hemisphere points (odd ids). Spatial queries must partition correctly.
#[test]
fn test_concurrent_inserts_alternating_hemispheres() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();

    // 4 north writers (even ids: 0, 2, 4, ... 198)
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..50u64 {
                let id = t * 100 + i * 2; // even
                idx.insert(IndexedValue::Plain(north_point(id)), id);
            }
        }));
    }

    // 4 south writers (odd ids: 1, 3, 5, ... 199)
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..50u64 {
                let id = t * 100 + i * 2 + 1; // odd
                idx.insert(IndexedValue::Plain(south_point(id)), id);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // North bbox: lat 0.5..90 should contain only even ids
    let north_op = GeoFilterOp::BoundingBox {
        min_lat: 0.5,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let north_results = collect_sorted(&index, north_op);
    for &id in &north_results {
        assert_eq!(id % 2, 0, "north bbox got odd id {id}");
    }

    // South bbox: lat -90..-0.5 should contain only odd ids
    let south_op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: -0.5,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let south_results = collect_sorted(&index, south_op);
    for &id in &south_results {
        assert_eq!(id % 2, 1, "south bbox got even id {id}");
    }

    // Total should be 400 (200 north + 200 south)
    let total = collect_sorted(&index, bbox_all());
    assert_eq!(total.len(), 400);
}

/// 8 threads all race to insert the same doc_id with different points.
/// The doc_id must appear at least once. No panics.
#[test]
fn test_concurrent_inserts_same_doc_id() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for t in 0..8u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let lat = 1.0 + t as f64;
                let lon = 1.0 + t as f64;
                idx.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), 42);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert!(
        results.contains(&42),
        "doc_id 42 must be present at least once"
    );
}

/// 800 docs pre-inserted. 8 threads delete 50 each (disjoint). 400 remain.
#[test]
fn test_concurrent_multi_thread_deletes() {
    let (_tmp, index) = make_index();

    for i in 0u64..800 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for t in 0..8u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            // Each thread deletes 50 items from its range of 100
            let base = t * 100;
            for i in 0..50 {
                idx.delete(base + i);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 400, "expected 400 remaining");

    // Verify deleted items are absent
    for t in 0..8u64 {
        let base = t * 100;
        for i in 0..50 {
            assert!(!results.contains(&(base + i)));
        }
        // Remaining items present
        for i in 50..100 {
            assert!(results.contains(&(base + i)));
        }
    }
}

/// 4 deleters + 4 readers concurrent. Readers see count ≤ initial.
#[test]
fn test_concurrent_deletes_during_snapshot_refresh() {
    let (_tmp, index) = make_index();

    for i in 0u64..400 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    // Force a clean snapshot before starting threads
    let _ = index.filter(bbox_all());

    let barrier = Arc::new(Barrier::new(8));
    let stop = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // 4 deleters
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let base = t * 100;
            for i in 0..100 {
                idx.delete(base + i);
                thread::yield_now();
            }
        }));
    }

    // 4 readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            while !s.load(Ordering::Relaxed) {
                let count = idx.filter(bbox_all()).iter().count();
                assert!(count <= 400, "count exceeded initial: {count}");
            }
        }));
    }

    // Wait for deleters to finish (they're bounded)
    // First 4 handles are deleters
    for h in handles.drain(..4) {
        h.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // Final: 0 remaining (all 400 deleted)
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 0);
}

// ===========================================================================
// Category 7: Read Consistency
// ===========================================================================

/// 16 readers on a static index. All must see exactly 100 every time.
#[test]
fn test_concurrent_filters_read_consistency() {
    let (_tmp, index) = make_index();

    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    // Force snapshot refresh
    let _ = index.filter(bbox_all());

    let barrier = Arc::new(Barrier::new(16));

    let mut handles = Vec::new();
    for _ in 0..16 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let count = idx.filter(bbox_all()).iter().count();
                assert_eq!(count, 100, "inconsistent read: expected 100, got {count}");
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

/// 8 readers + 1 writer. Writer inserts with yield_now(). Readers see >= 100.
#[test]
fn test_concurrent_filters_during_dirty_snapshot() {
    let (_tmp, index) = make_index();

    // Baseline: 100 items
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let stop = Arc::new(AtomicBool::new(false));

    let idx_w = Arc::clone(&index);
    let stop_w = Arc::clone(&stop);
    let writer = thread::spawn(move || {
        for i in 100u64..1100 {
            if stop_w.load(Ordering::Relaxed) {
                break;
            }
            idx_w.insert(IndexedValue::Plain(point_for_id(i)), i);
            thread::yield_now();
        }
    });

    let mut readers = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let count = idx.filter(bbox_all()).iter().count();
                assert!(count >= 100, "expected >= 100, got {count}");
            }
        }));
    }

    writer.join().unwrap();
    stop.store(true, Ordering::Relaxed);

    for r in readers {
        r.join().unwrap();
    }
}

// ===========================================================================
// Category 8: Spatial-Query-Specific (geopoint unique)
// ===========================================================================

/// 4 bbox readers + 4 radius readers hit the index concurrently with a writer.
/// Exercises both CompactedQueryIterator code paths. No panics.
#[test]
fn test_concurrent_bbox_and_radius_queries() {
    let (_tmp, index) = make_index();

    // Insert 200 points around the equator/prime meridian
    for i in 0u64..200 {
        let lat = (i as f64 * 0.1) - 10.0; // -10 .. 10
        let lon = (i as f64 * 0.1) - 10.0;
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i);
    }
    index.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();

    // 4 bbox readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            while !s.load(Ordering::Relaxed) {
                let op = GeoFilterOp::BoundingBox {
                    min_lat: -15.0,
                    max_lat: 15.0,
                    min_lon: -15.0,
                    max_lon: 15.0,
                };
                let count = idx.filter(op).iter().count();
                assert!(count >= 200, "bbox: expected >= 200, got {count}");
            }
        }));
    }

    // 4 radius readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            let center = GeoPoint::new(0.0, 0.0).unwrap();
            while !s.load(Ordering::Relaxed) {
                let op = GeoFilterOp::Radius {
                    center,
                    radius_meters: 2_000_000.0, // ~2000km, should cover all points
                };
                let count = idx.filter(op).iter().count();
                assert!(count > 0, "radius: expected > 0, got 0");
            }
        }));
    }

    // Let readers run briefly
    thread::sleep(std::time::Duration::from_millis(200));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

/// 4 readers each query a different quadrant (NE/NW/SE/SW). Points placed
/// in specific quadrants must not cross-contaminate.
#[test]
fn test_concurrent_disjoint_spatial_queries() {
    let (_tmp, index) = make_index();

    // Insert points in 4 quadrants:
    // NE (lat 10-20, lon 10-20): ids 0-49
    // NW (lat 10-20, lon -20..-10): ids 50-99
    // SE (lat -20..-10, lon 10-20): ids 100-149
    // SW (lat -20..-10, lon -20..-10): ids 150-199
    for i in 0u64..50 {
        let lat = 10.0 + (i as f64 * 0.2);
        let lon = 10.0 + (i as f64 * 0.2);
        index.insert(IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()), i); // NE
    }
    for i in 0u64..50 {
        let lat = 10.0 + (i as f64 * 0.2);
        let lon = -20.0 + (i as f64 * 0.2);
        index.insert(
            IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
            50 + i,
        ); // NW
    }
    for i in 0u64..50 {
        let lat = -20.0 + (i as f64 * 0.2);
        let lon = 10.0 + (i as f64 * 0.2);
        index.insert(
            IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
            100 + i,
        ); // SE
    }
    for i in 0u64..50 {
        let lat = -20.0 + (i as f64 * 0.2);
        let lon = -20.0 + (i as f64 * 0.2);
        index.insert(
            IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
            150 + i,
        ); // SW
    }
    index.compact(1).unwrap();

    let barrier = Arc::new(Barrier::new(4));

    struct Quadrant {
        op: GeoFilterOp,
        expected_range: std::ops::Range<u64>,
    }

    let quadrants = vec![
        Quadrant {
            op: GeoFilterOp::BoundingBox {
                min_lat: 9.0,
                max_lat: 21.0,
                min_lon: 9.0,
                max_lon: 21.0,
            },
            expected_range: 0..50,
        },
        Quadrant {
            op: GeoFilterOp::BoundingBox {
                min_lat: 9.0,
                max_lat: 21.0,
                min_lon: -21.0,
                max_lon: -9.0,
            },
            expected_range: 50..100,
        },
        Quadrant {
            op: GeoFilterOp::BoundingBox {
                min_lat: -21.0,
                max_lat: -9.0,
                min_lon: 9.0,
                max_lon: 21.0,
            },
            expected_range: 100..150,
        },
        Quadrant {
            op: GeoFilterOp::BoundingBox {
                min_lat: -21.0,
                max_lat: -9.0,
                min_lon: -21.0,
                max_lon: -9.0,
            },
            expected_range: 150..200,
        },
    ];

    let mut handles = Vec::new();
    for q in quadrants {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..50 {
                let results = {
                    let mut v: Vec<u64> = idx.filter(q.op.clone()).iter().collect();
                    v.sort_unstable();
                    v
                };
                let expected: HashSet<u64> = q.expected_range.clone().collect();
                for &id in &results {
                    assert!(
                        expected.contains(&id),
                        "cross-contamination: unexpected id {id} in quadrant {:?}",
                        q.expected_range
                    );
                }
                assert_eq!(
                    results.len(),
                    50,
                    "quadrant {:?} missing points",
                    q.expected_range
                );
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ===========================================================================
// Category 9: High Contention Stress
// ===========================================================================

/// Stress test: 4 inserters + 2 deleters + 4 readers + 2 compactors for 500ms.
/// No panics, no deadlocks, no duplicates.
#[test]
fn test_mixed_ops_stress() {
    let (_tmp, index) = make_index();

    // Seed with initial data
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(12));
    let stop = Arc::new(AtomicBool::new(false));
    let next_id = Arc::new(AtomicU64::new(100));

    let mut handles = Vec::new();

    // 4 inserters
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        let nid = Arc::clone(&next_id);
        handles.push(thread::spawn(move || {
            bar.wait();
            while !s.load(Ordering::Relaxed) {
                let id = nid.fetch_add(1, Ordering::Relaxed);
                idx.insert(IndexedValue::Plain(point_for_id(id)), id);
                thread::yield_now();
            }
        }));
    }

    // 2 deleters (delete from the initial 100)
    for t in 0..2u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            let base = t * 50;
            let mut i = 0u64;
            while !s.load(Ordering::Relaxed) {
                idx.delete(base + (i % 50));
                i += 1;
                thread::yield_now();
            }
        }));
    }

    // 4 readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            while !s.load(Ordering::Relaxed) {
                let results: Vec<u64> = idx.filter(bbox_all()).iter().collect();
                // Check no duplicates
                let unique: HashSet<u64> = results.iter().copied().collect();
                assert_eq!(
                    results.len(),
                    unique.len(),
                    "duplicates detected in filter results"
                );
            }
        }));
    }

    // 2 compactors
    for t in 0..2u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let s = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut version_id = 1 + t * 1000;
            while !s.load(Ordering::Relaxed) {
                let _ = idx.compact(version_id);
                version_id += 1;
                thread::yield_now();
            }
        }));
    }

    // Let it run for 500ms
    thread::sleep(std::time::Duration::from_millis(500));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    // Final check: no duplicates
    let results = collect_sorted(&index, bbox_all());
    let unique: HashSet<u64> = results.iter().copied().collect();
    assert_eq!(results.len(), unique.len());
}

/// 8 deleters with overlapping targets. All targeted ids absent, untargeted present.
#[test]
fn test_high_delete_contention() {
    let (_tmp, index) = make_index();

    // Insert 800 items
    for i in 0u64..800 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(8));

    // All 8 threads try to delete ids 0..400 (overlapping)
    let mut handles = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0u64..400 {
                idx.delete(i);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 400, "expected 400 remaining");

    for i in 0u64..400 {
        assert!(!results.contains(&i), "deleted id {i} still present");
    }
    for i in 400u64..800 {
        assert!(results.contains(&i), "non-deleted id {i} missing");
    }
}

/// 4 threads each cycle: insert their doc_id, then delete it, 100 times.
/// After compaction clears sticky deletes, a final insert makes all ids present.
#[test]
fn test_insert_delete_cycle() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(4));

    let mut handles = Vec::new();
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let id = t;
            for _ in 0..100 {
                idx.insert(IndexedValue::Plain(point_for_id(id)), id);
                thread::yield_now();
                idx.delete(id);
                thread::yield_now();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Compact to clear sticky deletes from the live layer, then re-insert
    index.compact(1).unwrap();
    for t in 0..4u64 {
        index.insert(IndexedValue::Plain(point_for_id(t)), t);
    }

    let results = collect_sorted(&index, bbox_all());
    for t in 0..4u64 {
        assert!(
            results.contains(&t),
            "doc_id {t} missing after final insert"
        );
    }
}

// ===========================================================================
// Category 10: Edge Cases
// ===========================================================================

/// 8 readers on an empty index. Always 0 results. Both bbox and radius.
#[test]
fn test_empty_index_concurrent_queries() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for t in 0..8u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                if t % 2 == 0 {
                    let count = idx.filter(bbox_all()).iter().count();
                    assert_eq!(count, 0, "non-empty on empty index (bbox)");
                } else {
                    let op = GeoFilterOp::Radius {
                        center: GeoPoint::new(0.0, 0.0).unwrap(),
                        radius_meters: 1_000_000.0,
                    };
                    let count = idx.filter(op).iter().count();
                    assert_eq!(count, 0, "non-empty on empty index (radius)");
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

/// doc_id=0 survives concurrent reads + compaction.
#[test]
fn test_doc_id_zero_concurrent() {
    let (_tmp, index) = make_index();

    index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 0);

    let barrier = Arc::new(Barrier::new(3));

    let mut handles = Vec::new();

    // 2 readers
    for _ in 0..2 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let results: Vec<u64> = idx.filter(bbox_all()).iter().collect();
                assert!(results.contains(&0), "doc_id 0 disappeared");
            }
        }));
    }

    // 1 compactor
    {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            idx.compact(1).unwrap();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // After compaction, doc_id 0 still present
    let results = collect_sorted(&index, bbox_all());
    assert!(results.contains(&0));
}

/// doc_ids near u64::MAX inserted concurrently. All present.
#[test]
fn test_large_doc_ids_concurrent() {
    let (_tmp, index) = make_index();
    let barrier = Arc::new(Barrier::new(4));

    let mut handles = Vec::new();
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..25u64 {
                let id = u64::MAX - (t * 25 + i);
                idx.insert(IndexedValue::Plain(point_for_id(id)), id);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 100);

    for t in 0..4u64 {
        for i in 0..25u64 {
            let id = u64::MAX - (t * 25 + i);
            assert!(results.contains(&id), "missing large doc_id {id}");
        }
    }
}

/// Delete non-existent ids while reading. Original 100 docs must be unaffected.
#[test]
fn test_delete_nonexistent_concurrent() {
    let (_tmp, index) = make_index();

    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }

    let barrier = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();

    // 4 deleters — delete ids 1000..2000 (non-existent)
    for t in 0..4u64 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let base = 1000 + t * 250;
            for i in 0..250 {
                idx.delete(base + i);
            }
        }));
    }

    // 4 readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..100 {
                let count = idx.filter(bbox_all()).iter().count();
                assert_eq!(count, 100, "original data affected: got {count}");
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 100);
}

// ===========================================================================
// Category 11: Cleanup
// ===========================================================================

/// 3 versions created. Cleanup removes old ones while 4 readers run.
/// Readers still get correct results (Arc keeps mmaps alive).
#[test]
fn test_cleanup_during_reads() {
    let (_tmp, index) = make_index();

    // Create 3 versions
    for i in 0u64..50 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(1).unwrap();

    for i in 50u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(2).unwrap();

    for i in 100u64..150 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(3).unwrap();

    let stop = Arc::new(AtomicBool::new(false));

    // 4 readers
    let mut readers = Vec::new();
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let count = idx.filter(bbox_all()).iter().count();
                assert!(
                    count >= 150,
                    "expected >= 150 after 3 compactions, got {count}"
                );
            }
        }));
    }

    // Cleanup while readers run
    index.cleanup();

    // Let readers continue briefly after cleanup
    thread::sleep(std::time::Duration::from_millis(50));
    stop.store(true, Ordering::Relaxed);

    for r in readers {
        r.join().unwrap();
    }

    // Final check
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 150);
}

// ===========================================================================
// Category 12: Long-Running Stability
// ===========================================================================

/// 3-second stress: 2 inserters, 1 deleter, 4 readers, 1 compactor.
/// No deadlocks, no panics. Consistent final state.
#[test]
fn test_long_running_mixed_ops() {
    let (_tmp, index) = make_index();

    let stop = Arc::new(AtomicBool::new(false));
    let next_insert_id = Arc::new(AtomicU64::new(0));
    let insert_count = Arc::new(AtomicU64::new(0));
    let delete_count = Arc::new(AtomicU64::new(0));
    let compaction_count = Arc::new(AtomicU64::new(0));
    let read_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    // 2 inserters
    for _ in 0..2 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let nid = Arc::clone(&next_insert_id);
        let ic = Arc::clone(&insert_count);
        handles.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let id = nid.fetch_add(1, Ordering::Relaxed);
                idx.insert(IndexedValue::Plain(point_for_id(id)), id);
                ic.fetch_add(1, Ordering::Relaxed);
                thread::yield_now();
            }
        }));
    }

    // 1 deleter (deletes from the lower end)
    {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let dc = Arc::clone(&delete_count);
        handles.push(thread::spawn(move || {
            let mut next_del = 0u64;
            while !s.load(Ordering::Relaxed) {
                idx.delete(next_del);
                dc.fetch_add(1, Ordering::Relaxed);
                next_del += 1;
                thread::yield_now();
            }
        }));
    }

    // 4 readers
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let rc = Arc::clone(&read_count);
        handles.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let results: Vec<u64> = idx.filter(bbox_all()).iter().collect();
                // Check no duplicates
                let unique: HashSet<u64> = results.iter().copied().collect();
                assert_eq!(
                    results.len(),
                    unique.len(),
                    "duplicates in long-running test"
                );
                rc.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // 1 compactor
    {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        let cc = Arc::clone(&compaction_count);
        handles.push(thread::spawn(move || {
            let mut version_id = 1u64;
            while !s.load(Ordering::Relaxed) {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = idx.compact(version_id);
                }))
                .is_ok()
                {
                    cc.fetch_add(1, Ordering::Relaxed);
                }
                version_id += 1;
                thread::sleep(std::time::Duration::from_millis(50));
            }
        }));
    }

    // Run for 3 seconds
    thread::sleep(std::time::Duration::from_secs(3));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    // Verify final state is consistent
    let results = collect_sorted(&index, bbox_all());
    let unique: HashSet<u64> = results.iter().copied().collect();
    assert_eq!(results.len(), unique.len(), "duplicates in final state");

    // Verify operation counts are reasonable
    let ic = insert_count.load(Ordering::Relaxed);
    let dc = delete_count.load(Ordering::Relaxed);
    let cc = compaction_count.load(Ordering::Relaxed);
    let rc = read_count.load(Ordering::Relaxed);

    assert!(ic > 0, "no inserts happened");
    assert!(dc > 0, "no deletes happened");
    assert!(cc > 0, "no compactions happened");
    assert!(rc > 0, "no reads happened");

    // Final count should be reasonable
    assert!(
        !results.is_empty(),
        "index should have some entries (inserts: {ic}, deletes: {dc})"
    );
}

// ===========================================================================
// Category 13: Filter During Compact
// ===========================================================================

/// 4 readers query continuously during two compaction rounds.
/// Count must always be >= baseline.
#[test]
fn test_filter_during_compact() {
    let (_tmp, index) = make_index();

    // Baseline: 100 items, compacted
    for i in 0u64..100 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(1).unwrap();

    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let s = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let count = idx.filter(bbox_all()).iter().count();
                assert!(count >= 100, "count dropped below baseline: {count}");
            }
        }));
    }

    // Two more compaction rounds with added data
    for i in 100u64..200 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(2).unwrap();

    for i in 200u64..300 {
        index.insert(IndexedValue::Plain(point_for_id(i)), i);
    }
    index.compact(3).unwrap();

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }

    // Final: 300 items
    let results = collect_sorted(&index, bbox_all());
    assert_eq!(results.len(), 300);
}
