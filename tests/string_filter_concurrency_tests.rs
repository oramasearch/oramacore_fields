use oramacore_fields::string_filter::{IndexedValue, StringFilterStorage, Threshold};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

fn p(s: &str) -> IndexedValue {
    IndexedValue::Plain(s.to_string())
}

#[test]
fn test_concurrent_reads_during_writes() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100 {
        index.insert(&p("key"), i);
    }

    let stop = Arc::new(AtomicBool::new(false));

    // Spawn reader threads
    let mut handles = Vec::new();
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let results: Vec<u64> = idx.filter("key").iter().collect();
                assert!(
                    results.len() >= 100,
                    "Reader saw only {} items, expected >= 100",
                    results.len()
                );
            }
        }));
    }

    // Spawn writer thread
    let idx = Arc::clone(&index);
    let writer = thread::spawn(move || {
        for i in 101..=200 {
            idx.insert(&p("key"), i);
        }
    });

    writer.join().unwrap();
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_reads_during_compaction() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=100 {
        index.insert(&p("key"), i);
    }

    let stop = Arc::new(AtomicBool::new(false));

    // Spawn reader thread
    let idx = Arc::clone(&index);
    let stop_clone = Arc::clone(&stop);
    let reader = thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            let results: Vec<u64> = idx.filter("key").iter().collect();
            assert!(
                results.len() >= 100,
                "Data lost during compaction! Saw {} items, expected >= 100",
                results.len()
            );
        }
    });

    // Compact in main thread
    index.compact(1).unwrap();

    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();
}

#[test]
fn test_iterator_stability_across_version_swap() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Insert initial data
    for i in 1..=50 {
        index.insert(&p("key"), i);
    }

    // Get a filter handle before compaction
    let filter = index.filter("key");

    // Compact (swaps version)
    index.compact(1).unwrap();

    // Add more data
    for i in 51..=100 {
        index.insert(&p("key"), i);
    }

    // The old filter should still work with its snapshot
    let results: Vec<u64> = filter.iter().collect();
    assert_eq!(results.len(), 50);
    assert_eq!(results[0], 1);
    assert_eq!(results[49], 50);
}

#[test]
fn test_stress_concurrent_operations() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    let num_writers = 4;
    let inserts_per_writer = 100;
    let mut handles = Vec::new();

    // Spawn writer threads - each writes to a different key
    for t in 0..num_writers {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let key = format!("writer_{t}");
            for i in 0..inserts_per_writer {
                idx.insert(&p(&key), (t * inserts_per_writer + i) as u64);
            }
        }));
    }

    // Spawn reader thread
    let idx = Arc::clone(&index);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);
    handles.push(thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            let _results: Vec<u64> = idx.filter("writer_0").iter().collect();
            thread::yield_now();
        }
    }));

    // Wait for writers
    for h in handles.drain(..num_writers as usize) {
        h.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // Verify all data is present
    for t in 0..num_writers {
        let key = format!("writer_{t}");
        let results: Vec<u64> = index.filter(&key).iter().collect();
        assert_eq!(
            results.len(),
            inserts_per_writer as usize,
            "Writer {t} data incomplete"
        );
    }
}

#[test]
fn test_concurrent_compaction_and_writes() {
    let tmp = TempDir::new().unwrap();
    let index =
        Arc::new(StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

    // Pre-populate
    for i in 1..=50 {
        index.insert(&p("key"), i);
    }

    // Compact in a thread
    let idx = Arc::clone(&index);
    let compactor = thread::spawn(move || {
        idx.compact(1).unwrap();
    });

    // Write concurrently
    for i in 51..=100 {
        index.insert(&p("key"), i);
    }

    compactor.join().unwrap();

    // All data should be present
    let results: Vec<u64> = index.filter("key").iter().collect();
    assert!(
        results.len() >= 50,
        "Should have at least the pre-populated data, got {}",
        results.len()
    );
}
