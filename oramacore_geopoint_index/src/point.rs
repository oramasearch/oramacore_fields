use crate::error::Error;

const LAT_SCALE: f64 = ((1u64 << 31) - 1) as f64 / 180.0;
const LON_SCALE: f64 = ((1u64 << 31) - 1) as f64 / 360.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    lat: f64,
    lon: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lon: f64) -> Result<Self, Error> {
        if !(-90.0..=90.0).contains(&lat) || lat.is_nan() {
            return Err(Error::InvalidLatitude(lat));
        }
        if !(-180.0..=180.0).contains(&lon) || lon.is_nan() {
            return Err(Error::InvalidLongitude(lon));
        }
        Ok(Self { lat, lon })
    }

    #[inline]
    pub fn lat(&self) -> f64 {
        self.lat
    }

    #[inline]
    pub fn lon(&self) -> f64 {
        self.lon
    }

    pub fn encode(&self) -> EncodedPoint {
        EncodedPoint {
            lat: encode_lat(self.lat),
            lon: encode_lon(self.lon),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedPoint {
    pub lat: i32,
    pub lon: i32,
}

impl EncodedPoint {
    pub fn decode(&self) -> GeoPoint {
        GeoPoint {
            lat: decode_lat(self.lat),
            lon: decode_lon(self.lon),
        }
    }

    #[inline]
    pub fn dim(&self, dim: u8) -> i32 {
        if dim == 0 { self.lat } else { self.lon }
    }
}

#[inline]
pub fn encode_lat(lat: f64) -> i32 {
    (lat * LAT_SCALE).round() as i32
}

#[inline]
pub fn encode_lon(lon: f64) -> i32 {
    (lon * LON_SCALE).round() as i32
}

#[inline]
pub fn decode_lat(encoded: i32) -> f64 {
    encoded as f64 / LAT_SCALE
}

#[inline]
pub fn decode_lon(encoded: i32) -> f64 {
    encoded as f64 / LON_SCALE
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointEntry {
    pub point: EncodedPoint,
    pub doc_id: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geopoint_valid() {
        let p = GeoPoint::new(41.9028, 12.4964).unwrap();
        assert!((p.lat() - 41.9028).abs() < 1e-6);
        assert!((p.lon() - 12.4964).abs() < 1e-6);
    }

    #[test]
    fn test_geopoint_invalid_lat() {
        assert!(GeoPoint::new(91.0, 0.0).is_err());
        assert!(GeoPoint::new(-91.0, 0.0).is_err());
        assert!(GeoPoint::new(f64::NAN, 0.0).is_err());
        assert!(GeoPoint::new(f64::INFINITY, 0.0).is_err());
    }

    #[test]
    fn test_geopoint_invalid_lon() {
        assert!(GeoPoint::new(0.0, 181.0).is_err());
        assert!(GeoPoint::new(0.0, -181.0).is_err());
        assert!(GeoPoint::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn test_geopoint_edge_values() {
        assert!(GeoPoint::new(90.0, 180.0).is_ok());
        assert!(GeoPoint::new(-90.0, -180.0).is_ok());
        assert!(GeoPoint::new(0.0, 0.0).is_ok());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let cases = [
            (41.9028, 12.4964),
            (0.0, 0.0),
            (90.0, 180.0),
            (-90.0, -180.0),
            (51.5074, -0.1278),
        ];

        for (lat, lon) in cases {
            let p = GeoPoint::new(lat, lon).unwrap();
            let encoded = p.encode();
            let decoded = encoded.decode();
            assert!(
                (decoded.lat() - lat).abs() < 1e-5,
                "lat roundtrip: {lat} -> {} -> {}",
                encoded.lat,
                decoded.lat()
            );
            assert!(
                (decoded.lon() - lon).abs() < 1e-5,
                "lon roundtrip: {lon} -> {} -> {}",
                encoded.lon,
                decoded.lon()
            );
        }
    }

    #[test]
    fn test_encoded_point_dim() {
        let p = GeoPoint::new(10.0, 20.0).unwrap().encode();
        assert_eq!(p.dim(0), p.lat);
        assert_eq!(p.dim(1), p.lon);
    }

    #[test]
    fn test_encoding_ordering() {
        let a = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let b = GeoPoint::new(20.0, 20.0).unwrap().encode();
        assert!(a.lat < b.lat);

        let c = GeoPoint::new(10.0, 10.0).unwrap().encode();
        let d = GeoPoint::new(10.0, 30.0).unwrap().encode();
        assert!(c.lon < d.lon);
    }
}
