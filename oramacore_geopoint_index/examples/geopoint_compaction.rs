//! Demonstrates compaction and persistence across reopens.

use oramacore_geopoint_index::{GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold};

fn query_all(index: &GeoPointStorage) -> Vec<u64> {
    let op = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    index.filter(op).iter().collect()
}

fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let base_path = dir.path().to_path_buf();

    // Phase 1: Insert initial data and compact
    {
        let index = GeoPointStorage::new(base_path.clone(), Threshold::default(), 10)?;

        index.insert(IndexedValue::Plain(GeoPoint::new(41.9028, 12.4964)?), 1); // Rome
        index.insert(IndexedValue::Plain(GeoPoint::new(48.8566, 2.3522)?), 2); // Paris
        index.insert(IndexedValue::Plain(GeoPoint::new(51.5074, -0.1278)?), 3); // London

        println!("Before compaction: {:?}", query_all(&index));

        // Compact flushes the live layer into a BKD tree on disk
        index.compact(1)?;
        println!("After compaction (version_id=1): {:?}", query_all(&index));
    }

    // Phase 2: Reopen the index — data persists from disk
    {
        let index = GeoPointStorage::new(base_path.clone(), Threshold::default(), 10)?;
        println!(
            "Reopened at version {}: {:?}",
            index.current_version_id(),
            query_all(&index)
        );

        // Add more data and compact again
        index.insert(IndexedValue::Plain(GeoPoint::new(52.5200, 13.4050)?), 4); // Berlin
        index.compact(2)?;
        println!(
            "After second compaction (version_id=2): {:?}",
            query_all(&index)
        );

        // Delete a document and compact
        index.delete(2); // Remove Paris
        index.compact(3)?;
        println!(
            "After delete + compact (version_id=3): {:?}",
            query_all(&index)
        );

        // Cleanup removes old version directories from disk
        index.cleanup();
        println!(
            "Cleanup done. Current version: {}",
            index.current_version_id()
        );
    }

    // Phase 3: Final reopen confirms everything survived
    {
        let index = GeoPointStorage::new(base_path, Threshold::default(), 10)?;
        println!(
            "Final reopen at version {}: {:?}",
            index.current_version_id(),
            query_all(&index)
        );
    }

    Ok(())
}
