use super::compacted::query::MultiSegmentQueryIterator;
use super::compacted::CompactedVersion;
use super::live::LiveSnapshot;
use super::point::GeoPoint;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum GeoFilterOp {
    BoundingBox {
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
    },
    Radius {
        center: GeoPoint,
        radius_meters: f64,
    },
}

impl GeoFilterOp {
    pub fn matches(&self, point: &GeoPoint) -> bool {
        match self {
            GeoFilterOp::BoundingBox {
                min_lat,
                max_lat,
                min_lon,
                max_lon,
            } => {
                point.lat() >= *min_lat
                    && point.lat() <= *max_lat
                    && point.lon() >= *min_lon
                    && point.lon() <= *max_lon
            }
            GeoFilterOp::Radius {
                center,
                radius_meters,
            } => {
                let dist = super::compacted::query::haversine_distance(
                    center.lat(),
                    center.lon(),
                    point.lat(),
                    point.lon(),
                );
                dist <= *radius_meters
            }
        }
    }
}

pub struct FilterData {
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
    op: GeoFilterOp,
}

impl FilterData {
    pub(crate) fn new(
        version: Arc<CompactedVersion>,
        snapshot: Arc<LiveSnapshot>,
        op: GeoFilterOp,
    ) -> Self {
        Self {
            version,
            snapshot,
            op,
        }
    }

    pub fn iter(&self) -> FilterIterator<'_> {
        let compacted_iter = self.version.query_iter(&self.op);

        FilterIterator {
            live_inserts: &self.snapshot.inserts,
            live_cursor: 0,
            op: &self.op,
            compacted_iter,
            live_deleted: &self.snapshot.deletes,
        }
    }
}

impl<'a> IntoIterator for &'a FilterData {
    type Item = u64;
    type IntoIter = FilterIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct FilterIterator<'a> {
    live_inserts: &'a [(GeoPoint, u64)],
    live_cursor: usize,
    op: &'a GeoFilterOp,
    compacted_iter: Option<MultiSegmentQueryIterator<'a>>,
    live_deleted: &'a HashSet<u64>,
}

impl Iterator for FilterIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        // Phase 1: Scan live inserts lazily
        while self.live_cursor < self.live_inserts.len() {
            let (point, doc_id) = &self.live_inserts[self.live_cursor];
            self.live_cursor += 1;
            if self.op.matches(point) {
                return Some(*doc_id);
            }
        }

        // Phase 2: Stream from compacted iterator
        let iter = self.compacted_iter.as_mut()?;
        loop {
            let doc_id = iter.next()?;
            if self.live_deleted.contains(&doc_id) {
                continue;
            }
            return Some(doc_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_matches() {
        let op = GeoFilterOp::BoundingBox {
            min_lat: 0.0,
            max_lat: 50.0,
            min_lon: 0.0,
            max_lon: 50.0,
        };

        let inside = GeoPoint::new(25.0, 25.0).unwrap();
        let outside = GeoPoint::new(60.0, 60.0).unwrap();
        let edge = GeoPoint::new(0.0, 0.0).unwrap();

        assert!(op.matches(&inside));
        assert!(!op.matches(&outside));
        assert!(op.matches(&edge));
    }

    #[test]
    fn test_radius_matches() {
        let center = GeoPoint::new(0.0, 0.0).unwrap();
        let op = GeoFilterOp::Radius {
            center,
            radius_meters: 200_000.0,
        };

        let near = GeoPoint::new(0.5, 0.5).unwrap();
        let far = GeoPoint::new(45.0, 45.0).unwrap();

        assert!(op.matches(&near));
        assert!(!op.matches(&far));
    }

    #[test]
    fn test_filter_data_empty() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot::empty());
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };

        let filter = FilterData::new(version, snapshot, op);
        let results: Vec<u64> = filter.iter().collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_data_live_only() {
        let version = Arc::new(CompactedVersion::empty());

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(50.0, 60.0).unwrap();

        let snapshot = Arc::new(LiveSnapshot {
            inserts: vec![(p1, 1), (p2, 2)],
            deletes: std::collections::HashSet::new(),

            ops_len: 0,
        });

        let op = GeoFilterOp::BoundingBox {
            min_lat: 0.0,
            max_lat: 30.0,
            min_lon: 0.0,
            max_lon: 30.0,
        };

        let filter = FilterData::new(version, snapshot, op);
        let results: Vec<u64> = filter.iter().collect();
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn test_filter_data_with_deletes() {
        let version = Arc::new(CompactedVersion::empty());

        let p2 = GeoPoint::new(15.0, 25.0).unwrap();

        // doc_id=1 is deleted (not in inserts), doc_id=2 is inserted
        let mut deletes = std::collections::HashSet::new();
        deletes.insert(1u64);

        let snapshot = Arc::new(LiveSnapshot {
            inserts: vec![(p2, 2)],
            deletes,

            ops_len: 0,
        });

        let op = GeoFilterOp::BoundingBox {
            min_lat: 0.0,
            max_lat: 30.0,
            min_lon: 0.0,
            max_lon: 30.0,
        };

        let filter = FilterData::new(version, snapshot, op);
        let results: Vec<u64> = filter.iter().collect();
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn test_filter_data_into_iter() {
        let version = Arc::new(CompactedVersion::empty());
        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let snapshot = Arc::new(LiveSnapshot {
            inserts: vec![(p1, 42)],
            deletes: std::collections::HashSet::new(),

            ops_len: 0,
        });

        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };

        let filter = FilterData::new(version, snapshot, op);
        let mut results = Vec::new();
        for doc_id in &filter {
            results.push(doc_id);
        }
        assert_eq!(results, vec![42]);
    }
}
