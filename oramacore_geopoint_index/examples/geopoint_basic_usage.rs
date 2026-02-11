//! Basic usage of the geopoint index: insert, query, and delete.

use oramacore_geopoint_index::{GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold};

fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let index = GeoPointStorage::new(dir.path().to_path_buf(), Threshold::default(), 10)?;

    // Insert some European cities
    let rome = GeoPoint::new(41.9028, 12.4964)?;
    let paris = GeoPoint::new(48.8566, 2.3522)?;
    let london = GeoPoint::new(51.5074, -0.1278)?;
    let berlin = GeoPoint::new(52.5200, 13.4050)?;
    let madrid = GeoPoint::new(40.4168, -3.7038)?;

    index.insert(IndexedValue::Plain(rome), 1);
    index.insert(IndexedValue::Plain(paris), 2);
    index.insert(IndexedValue::Plain(london), 3);
    index.insert(IndexedValue::Plain(berlin), 4);
    index.insert(IndexedValue::Plain(madrid), 5);

    // --- Bounding box query ---
    // Find cities in central/western Europe (lat 40-55, lon -5 to 15)
    let bbox = GeoFilterOp::BoundingBox {
        min_lat: 40.0,
        max_lat: 55.0,
        min_lon: -5.0,
        max_lon: 15.0,
    };
    let results: Vec<u64> = index.filter(bbox).iter().collect();
    println!("Cities in bounding box (40-55°N, 5°W-15°E): {results:?}");
    // Expected: [1, 2, 3, 4] (Rome, Paris, London, Berlin — Madrid is at -3.7 lon so included too)

    // --- Radius query ---
    // Find cities within 500 km of Paris
    let radius = GeoFilterOp::Radius {
        center: paris,
        radius_meters: 500_000.0,
    };
    let results: Vec<u64> = index.filter(radius).iter().collect();
    println!("Cities within 500 km of Paris: {results:?}");
    // London (~340 km) is within range; Rome (~1100 km), Berlin (~880 km), Madrid (~1050 km) are not

    // --- Delete a document ---
    index.delete(3); // Remove London
    let bbox_all = GeoFilterOp::BoundingBox {
        min_lat: -90.0,
        max_lat: 90.0,
        min_lon: -180.0,
        max_lon: 180.0,
    };
    let results: Vec<u64> = index.filter(bbox_all).iter().collect();
    println!("All cities after deleting London (doc 3): {results:?}");

    Ok(())
}
