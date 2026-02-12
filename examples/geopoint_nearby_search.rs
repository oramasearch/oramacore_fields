//! A realistic "find nearby places" scenario using radius queries.

use oramacore_fields::geopoint::{GeoFilterOp, GeoPoint, GeoPointStorage, IndexedValue, Threshold};

struct Place {
    id: u64,
    name: &'static str,
    lat: f64,
    lon: f64,
}

const PLACES: &[Place] = &[
    Place {
        id: 1,
        name: "Colosseum",
        lat: 41.8902,
        lon: 12.4922,
    },
    Place {
        id: 2,
        name: "Trevi Fountain",
        lat: 41.9009,
        lon: 12.4833,
    },
    Place {
        id: 3,
        name: "Vatican Museums",
        lat: 41.9065,
        lon: 12.4536,
    },
    Place {
        id: 4,
        name: "Pantheon",
        lat: 41.8986,
        lon: 12.4769,
    },
    Place {
        id: 5,
        name: "Spanish Steps",
        lat: 41.9060,
        lon: 12.4828,
    },
    Place {
        id: 6,
        name: "Piazza Navona",
        lat: 41.8992,
        lon: 12.4731,
    },
    Place {
        id: 7,
        name: "Trastevere",
        lat: 41.8867,
        lon: 12.4700,
    },
    Place {
        id: 8,
        name: "Ostia Antica",
        lat: 41.7558,
        lon: 12.2914,
    },
    Place {
        id: 9,
        name: "Villa Adriana (Tivoli)",
        lat: 41.9424,
        lon: 12.7744,
    },
    Place {
        id: 10,
        name: "Castel Gandolfo",
        lat: 41.7473,
        lon: 12.6508,
    },
];

fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let index = GeoPointStorage::new(dir.path().to_path_buf(), Threshold::default(), 10)?;

    // Index all places
    for place in PLACES {
        let point = GeoPoint::new(place.lat, place.lon)?;
        index.insert(IndexedValue::Plain(point), place.id);
    }

    // Compact to build BKD tree on disk
    index.compact(1)?;

    // Scenario: user is at the Trevi Fountain, find places within various radii
    let user_location = GeoPoint::new(41.9009, 12.4833)?;

    for radius_km in [1.0, 2.0, 5.0, 25.0] {
        let op = GeoFilterOp::Radius {
            center: user_location,
            radius_meters: radius_km * 1000.0,
        };
        let ids: Vec<u64> = index.filter(op).iter().collect();
        let names: Vec<&str> = ids
            .iter()
            .filter_map(|id| PLACES.iter().find(|p| p.id == *id).map(|p| p.name))
            .collect();
        println!("Within {radius_km} km of Trevi Fountain: {names:?}");
    }

    // Scenario: bounding box query over central Rome
    let op = GeoFilterOp::BoundingBox {
        min_lat: 41.885,
        max_lat: 41.910,
        min_lon: 12.460,
        max_lon: 12.500,
    };
    let ids: Vec<u64> = index.filter(op).iter().collect();
    let names: Vec<&str> = ids
        .iter()
        .filter_map(|id| PLACES.iter().find(|p| p.id == *id).map(|p| p.name))
        .collect();
    println!("\nPlaces in central Rome bounding box: {names:?}");

    Ok(())
}
