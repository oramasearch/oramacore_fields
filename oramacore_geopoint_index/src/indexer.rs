use crate::point::GeoPoint;
use serde_json::Value;

pub struct GeoPointIndexer {
    is_array: bool,
}

impl GeoPointIndexer {
    pub fn new(is_array: bool) -> Self {
        GeoPointIndexer { is_array }
    }

    pub fn index_json(&self, value: &Value) -> Option<IndexedValue> {
        if self.is_array {
            self.index_json_array(value)
        } else {
            self.index_json_plain(value)
        }
    }

    fn index_json_plain(&self, value: &Value) -> Option<IndexedValue> {
        parse_geopoint(value).map(IndexedValue::Plain)
    }

    fn index_json_array(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Array(arr) => {
                let points: Vec<_> = arr.iter().filter_map(parse_geopoint).collect();
                Some(IndexedValue::Array(points))
            }
            _ => None,
        }
    }
}

fn parse_geopoint(value: &Value) -> Option<GeoPoint> {
    match value {
        Value::Object(map) => {
            let lon = map.get("lon");
            let lat = map.get("lat");

            lon.zip(lat)
                .and_then(|(lon, lat)| lon.as_f64().zip(lat.as_f64()))
                .and_then(|(lon, lat)| GeoPoint::new(lat, lon).ok())
        }
        _ => None,
    }
}

pub enum IndexedValue {
    Plain(GeoPoint),
    Array(Vec<GeoPoint>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_json_plain() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": 41.9028, "lon": 12.4964});
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Plain(p) => {
                assert!((p.lat() - 41.9028).abs() < 1e-6);
                assert!((p.lon() - 12.4964).abs() < 1e-6);
            }
            _ => panic!("Expected Plain"),
        }
    }

    #[test]
    fn test_index_json_array() {
        let indexer = GeoPointIndexer::new(true);
        let value = json!([
            {"lat": 41.9028, "lon": 12.4964},
            {"lat": 51.5074, "lon": -0.1278}
        ]);
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Array(points) => {
                assert_eq!(points.len(), 2);
                assert!((points[0].lat() - 41.9028).abs() < 1e-6);
                assert!((points[0].lon() - 12.4964).abs() < 1e-6);
                assert!((points[1].lat() - 51.5074).abs() < 1e-6);
                assert!((points[1].lon() - (-0.1278)).abs() < 1e-6);
            }
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!(123);
        assert!(indexer.index_json(&value).is_none());

        let value = json!("not a geopoint");
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_index_json_invalid_coordinates() {
        let indexer = GeoPointIndexer::new(false);
        // Invalid latitude
        let value = json!({"lat": 91.0, "lon": 12.0});
        assert!(indexer.index_json(&value).is_none());

        // Missing field
        let value = json!({"lat": 41.0});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_index_json_array_filters_invalid() {
        let indexer = GeoPointIndexer::new(true);
        let value = json!([
            {"lat": 41.9028, "lon": 12.4964},
            {"lat": 91.0, "lon": 0.0},
            "invalid",
            {"lat": 51.5074, "lon": -0.1278}
        ]);
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Array(points) => {
                assert_eq!(points.len(), 2);
            }
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_index_json_array_not_array() {
        let indexer = GeoPointIndexer::new(true);
        let value = json!({"lat": 41.0, "lon": 12.0});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_missing_lon() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lon": 12.0});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_non_number_fields() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": "41.0", "lon": 12.0});
        assert!(indexer.index_json(&value).is_none());

        let value = json!({"lat": 41.0, "lon": "12.0"});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_null_fields() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": null, "lon": 12.0});
        assert!(indexer.index_json(&value).is_none());

        let value = json!({"lat": 41.0, "lon": null});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_invalid_longitude() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": 41.0, "lon": 181.0});
        assert!(indexer.index_json(&value).is_none());

        let value = json!({"lat": 41.0, "lon": -181.0});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_edge_coordinates() {
        let indexer = GeoPointIndexer::new(false);
        let cases = [
            (90.0, 180.0),
            (-90.0, -180.0),
            (0.0, 0.0),
            (90.0, -180.0),
            (-90.0, 180.0),
        ];
        for (lat, lon) in cases {
            let value = json!({"lat": lat, "lon": lon});
            let indexed = indexer.index_json(&value).unwrap();
            match indexed {
                IndexedValue::Plain(p) => {
                    assert!((p.lat() - lat).abs() < 1e-6);
                    assert!((p.lon() - lon).abs() < 1e-6);
                }
                _ => panic!("Expected Plain for ({lat}, {lon})"),
            }
        }
    }

    #[test]
    fn test_parse_geopoint_extra_fields_ignored() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": 41.0, "lon": 12.0, "alt": 100.0});
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Plain(p) => {
                assert!((p.lat() - 41.0).abs() < 1e-6);
                assert!((p.lon() - 12.0).abs() < 1e-6);
            }
            _ => panic!("Expected Plain"),
        }
    }

    #[test]
    fn test_parse_geopoint_empty_object() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({});
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_parse_geopoint_integer_values() {
        let indexer = GeoPointIndexer::new(false);
        let value = json!({"lat": 41, "lon": 12});
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Plain(p) => {
                assert!((p.lat() - 41.0).abs() < 1e-6);
                assert!((p.lon() - 12.0).abs() < 1e-6);
            }
            _ => panic!("Expected Plain"),
        }
    }
}
