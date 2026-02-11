use super::{InnerNodesView, LEAF_ENTRY_SIZE, LEAF_HEADER_SIZE, LeafOffsetsView, Segment};
use crate::iterator::GeoFilterOp;
use std::collections::HashSet;

enum QueryKind {
    BBox {
        query_min_lat: i32,
        query_max_lat: i32,
        query_min_lon: i32,
        query_max_lon: i32,
    },
    Radius {
        center_lat_f64: f64,
        center_lon_f64: f64,
        radius_meters: f64,
        bbox_min_lat: i32,
        bbox_max_lat: i32,
        bbox_min_lon: i32,
        bbox_max_lon: i32,
    },
}

enum StackFrame {
    Traverse {
        node_id: usize,
        cell_min_lat: i32,
        cell_max_lat: i32,
        cell_min_lon: i32,
        cell_max_lon: i32,
    },
    CollectAll {
        node_id: usize,
    },
}

struct LeafScan {
    entries_start: usize,
    count: usize,
    current: usize,
    check_bounds: bool,
}

pub struct CompactedQueryIterator<'a> {
    inner_nodes: InnerNodesView<'a>,
    num_leaves: usize,
    leaf_offsets: LeafOffsetsView<'a>,
    leaf_data: &'a [u8],
    deleted_set: &'a HashSet<u64>,
    kind: QueryKind,
    stack: Vec<StackFrame>,
    leaf_scan: Option<LeafScan>,
}

impl<'a> CompactedQueryIterator<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_bbox(
        inner_nodes: InnerNodesView<'a>,
        num_leaves: usize,
        leaf_offsets: LeafOffsetsView<'a>,
        leaf_data: &'a [u8],
        query_min_lat: i32,
        query_max_lat: i32,
        query_min_lon: i32,
        query_max_lon: i32,
        global_min_lat: i32,
        global_max_lat: i32,
        global_min_lon: i32,
        global_max_lon: i32,
        deleted_set: &'a HashSet<u64>,
    ) -> Self {
        let mut stack = Vec::with_capacity(32);
        stack.push(StackFrame::Traverse {
            node_id: 1,
            cell_min_lat: global_min_lat,
            cell_max_lat: global_max_lat,
            cell_min_lon: global_min_lon,
            cell_max_lon: global_max_lon,
        });

        Self {
            inner_nodes,
            num_leaves,
            leaf_offsets,
            leaf_data,
            deleted_set,
            kind: QueryKind::BBox {
                query_min_lat,
                query_max_lat,
                query_min_lon,
                query_max_lon,
            },
            stack,
            leaf_scan: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_radius(
        inner_nodes: InnerNodesView<'a>,
        num_leaves: usize,
        leaf_offsets: LeafOffsetsView<'a>,
        leaf_data: &'a [u8],
        center_lat_f64: f64,
        center_lon_f64: f64,
        radius_meters: f64,
        global_min_lat: i32,
        global_max_lat: i32,
        global_min_lon: i32,
        global_max_lon: i32,
        deleted_set: &'a HashSet<u64>,
    ) -> Self {
        let (bb_min_lat, bb_max_lat, bb_min_lon, bb_max_lon) =
            bounding_box_for_radius(center_lat_f64, center_lon_f64, radius_meters);

        let bbox_min_lat = crate::point::encode_lat(bb_min_lat);
        let bbox_max_lat = crate::point::encode_lat(bb_max_lat);
        let bbox_min_lon = crate::point::encode_lon(bb_min_lon);
        let bbox_max_lon = crate::point::encode_lon(bb_max_lon);

        let mut stack = Vec::with_capacity(32);
        stack.push(StackFrame::Traverse {
            node_id: 1,
            cell_min_lat: global_min_lat,
            cell_max_lat: global_max_lat,
            cell_min_lon: global_min_lon,
            cell_max_lon: global_max_lon,
        });

        Self {
            inner_nodes,
            num_leaves,
            leaf_offsets,
            leaf_data,
            deleted_set,
            kind: QueryKind::Radius {
                center_lat_f64,
                center_lon_f64,
                radius_meters,
                bbox_min_lat,
                bbox_max_lat,
                bbox_min_lon,
                bbox_max_lon,
            },
            stack,
            leaf_scan: None,
        }
    }

    fn start_leaf_scan(&mut self, leaf_index: usize, check_bounds: bool) -> bool {
        if leaf_index >= self.leaf_offsets.len() {
            return false;
        }
        let offset = self.leaf_offsets.get(leaf_index) as usize;
        if offset + LEAF_HEADER_SIZE > self.leaf_data.len() {
            return false;
        }

        let count =
            u32::from_ne_bytes(self.leaf_data[offset..offset + 4].try_into().unwrap()) as usize;

        if count == 0 {
            return false;
        }

        let entries_start = offset + LEAF_HEADER_SIZE;
        self.leaf_scan = Some(LeafScan {
            entries_start,
            count,
            current: 0,
            check_bounds,
        });
        true
    }

    fn next_from_leaf_scan(&mut self) -> Option<u64> {
        let scan = self.leaf_scan.as_mut()?;

        while scan.current < scan.count {
            let entry_offset = scan.entries_start + scan.current * LEAF_ENTRY_SIZE;
            scan.current += 1;

            if entry_offset + LEAF_ENTRY_SIZE > self.leaf_data.len() {
                self.leaf_scan = None;
                return None;
            }

            let doc_id = u64::from_ne_bytes(
                self.leaf_data[entry_offset + 8..entry_offset + 16]
                    .try_into()
                    .unwrap(),
            );

            if self.deleted_set.contains(&doc_id) {
                continue;
            }

            if !scan.check_bounds {
                return Some(doc_id);
            }

            // Need to check bounds
            let lat = i32::from_ne_bytes(
                self.leaf_data[entry_offset..entry_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            let lon = i32::from_ne_bytes(
                self.leaf_data[entry_offset + 4..entry_offset + 8]
                    .try_into()
                    .unwrap(),
            );

            match &self.kind {
                QueryKind::BBox {
                    query_min_lat,
                    query_max_lat,
                    query_min_lon,
                    query_max_lon,
                } => {
                    if lat >= *query_min_lat
                        && lat <= *query_max_lat
                        && lon >= *query_min_lon
                        && lon <= *query_max_lon
                    {
                        return Some(doc_id);
                    }
                }
                QueryKind::Radius {
                    center_lat_f64,
                    center_lon_f64,
                    radius_meters,
                    ..
                } => {
                    let pt_lat = crate::point::decode_lat(lat);
                    let pt_lon = crate::point::decode_lon(lon);
                    let dist = haversine_distance(*center_lat_f64, *center_lon_f64, pt_lat, pt_lon);
                    if dist <= *radius_meters {
                        return Some(doc_id);
                    }
                }
            }
        }

        self.leaf_scan = None;
        None
    }
}

impl Iterator for CompactedQueryIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            // Phase 1: drain active leaf scan
            if self.leaf_scan.is_some()
                && let Some(doc_id) = self.next_from_leaf_scan()
            {
                return Some(doc_id);
            }
            // Scan exhausted, fall through to pop stack

            // Phase 2: pop stack frame
            let frame = self.stack.pop()?;

            match frame {
                StackFrame::Traverse {
                    node_id,
                    cell_min_lat,
                    cell_max_lat,
                    cell_min_lon,
                    cell_max_lon,
                } => {
                    let (q_min_lat, q_max_lat, q_min_lon, q_max_lon) = match &self.kind {
                        QueryKind::BBox {
                            query_min_lat,
                            query_max_lat,
                            query_min_lon,
                            query_max_lon,
                        } => (
                            *query_min_lat,
                            *query_max_lat,
                            *query_min_lon,
                            *query_max_lon,
                        ),
                        QueryKind::Radius {
                            bbox_min_lat,
                            bbox_max_lat,
                            bbox_min_lon,
                            bbox_max_lon,
                            ..
                        } => (*bbox_min_lat, *bbox_max_lat, *bbox_min_lon, *bbox_max_lon),
                    };

                    let relation = classify_bbox(
                        cell_min_lat,
                        cell_max_lat,
                        cell_min_lon,
                        cell_max_lon,
                        q_min_lat,
                        q_max_lat,
                        q_min_lon,
                        q_max_lon,
                    );

                    match relation {
                        Relation::Outside => continue,
                        Relation::Inside => {
                            match &self.kind {
                                QueryKind::BBox { .. } => {
                                    self.stack.push(StackFrame::CollectAll { node_id });
                                    continue;
                                }
                                QueryKind::Radius {
                                    center_lat_f64,
                                    center_lon_f64,
                                    radius_meters,
                                    ..
                                } => {
                                    if cell_fully_inside_radius(
                                        cell_min_lat,
                                        cell_max_lat,
                                        cell_min_lon,
                                        cell_max_lon,
                                        *center_lat_f64,
                                        *center_lon_f64,
                                        *radius_meters,
                                    ) {
                                        self.stack.push(StackFrame::CollectAll { node_id });
                                        continue;
                                    }
                                    // Fall through to Crosses handling
                                }
                            }
                        }
                        Relation::Crosses => {
                            // Fall through to Crosses handling below
                        }
                    }

                    // Crosses (or Radius Inside but not fully inside radius)
                    if node_id >= self.num_leaves {
                        // Leaf node
                        let leaf_index = node_id - self.num_leaves;
                        self.start_leaf_scan(leaf_index, true);
                        continue;
                    }

                    // Inner node: push right then left (LIFO: left processed first)
                    let (split_value, split_dim) = self.inner_nodes.get(node_id);
                    let left = 2 * node_id;
                    let right = 2 * node_id + 1;

                    if split_dim == 0 {
                        // Split on latitude
                        self.stack.push(StackFrame::Traverse {
                            node_id: right,
                            cell_min_lat: split_value,
                            cell_max_lat,
                            cell_min_lon,
                            cell_max_lon,
                        });
                        self.stack.push(StackFrame::Traverse {
                            node_id: left,
                            cell_min_lat,
                            cell_max_lat: split_value,
                            cell_min_lon,
                            cell_max_lon,
                        });
                    } else {
                        // Split on longitude
                        self.stack.push(StackFrame::Traverse {
                            node_id: right,
                            cell_min_lat,
                            cell_max_lat,
                            cell_min_lon: split_value,
                            cell_max_lon,
                        });
                        self.stack.push(StackFrame::Traverse {
                            node_id: left,
                            cell_min_lat,
                            cell_max_lat,
                            cell_min_lon,
                            cell_max_lon: split_value,
                        });
                    }
                }
                StackFrame::CollectAll { node_id } => {
                    if node_id >= self.num_leaves {
                        // Leaf node
                        let leaf_index = node_id - self.num_leaves;
                        self.start_leaf_scan(leaf_index, false);
                        continue;
                    }

                    // Inner node: push right then left (LIFO: left processed first)
                    let left = 2 * node_id;
                    let right = 2 * node_id + 1;
                    self.stack.push(StackFrame::CollectAll { node_id: right });
                    self.stack.push(StackFrame::CollectAll { node_id: left });
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Relation {
    Outside,
    Inside,
    Crosses,
}

#[allow(clippy::too_many_arguments)]
fn classify_bbox(
    cell_min_lat: i32,
    cell_max_lat: i32,
    cell_min_lon: i32,
    cell_max_lon: i32,
    query_min_lat: i32,
    query_max_lat: i32,
    query_min_lon: i32,
    query_max_lon: i32,
) -> Relation {
    // Outside: no overlap
    if cell_max_lat < query_min_lat || cell_min_lat > query_max_lat {
        return Relation::Outside;
    }
    if cell_max_lon < query_min_lon || cell_min_lon > query_max_lon {
        return Relation::Outside;
    }
    // Inside: cell fully contained in query
    if query_min_lat <= cell_min_lat
        && cell_max_lat <= query_max_lat
        && query_min_lon <= cell_min_lon
        && cell_max_lon <= query_max_lon
    {
        return Relation::Inside;
    }
    Relation::Crosses
}

// --- Radius query support ---

const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

#[allow(clippy::too_many_arguments)]
fn cell_fully_inside_radius(
    cell_min_lat: i32,
    cell_max_lat: i32,
    cell_min_lon: i32,
    cell_max_lon: i32,
    center_lat: f64,
    center_lon: f64,
    radius_meters: f64,
) -> bool {
    let min_lat_f = crate::point::decode_lat(cell_min_lat);
    let max_lat_f = crate::point::decode_lat(cell_max_lat);
    let min_lon_f = crate::point::decode_lon(cell_min_lon);
    let max_lon_f = crate::point::decode_lon(cell_max_lon);

    // Safety margin: half the cell diagonal accounts for max corner-to-edge deviation on a sphere
    let cell_diagonal = haversine_distance(min_lat_f, min_lon_f, max_lat_f, max_lon_f);
    let effective_radius = radius_meters - cell_diagonal / 2.0;
    if effective_radius <= 0.0 {
        return false;
    }

    let corners = [
        (cell_min_lat, cell_min_lon),
        (cell_min_lat, cell_max_lon),
        (cell_max_lat, cell_min_lon),
        (cell_max_lat, cell_max_lon),
    ];

    for (lat_enc, lon_enc) in corners {
        let lat = crate::point::decode_lat(lat_enc);
        let lon = crate::point::decode_lon(lon_enc);
        if haversine_distance(center_lat, center_lon, lat, lon) > effective_radius {
            return false;
        }
    }
    true
}

pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_METERS * c
}

pub fn bounding_box_for_radius(
    center_lat: f64,
    center_lon: f64,
    radius_meters: f64,
) -> (f64, f64, f64, f64) {
    let angular_distance = radius_meters / EARTH_RADIUS_METERS;
    let angular_distance_deg = angular_distance.to_degrees();

    let min_lat = (center_lat - angular_distance_deg).max(-90.0);
    let max_lat = (center_lat + angular_distance_deg).min(90.0);

    // If we're at a pole, the bbox covers all longitudes
    if min_lat <= -90.0 || max_lat >= 90.0 {
        return (min_lat, max_lat, -180.0, 180.0);
    }

    // Longitude range expands near poles (latitude distortion)
    let ratio = angular_distance.sin() / center_lat.to_radians().cos();
    if ratio.abs() >= 1.0 {
        return (min_lat, max_lat, -180.0, 180.0);
    }
    let delta_lon = ratio.asin().to_degrees();

    let raw_min_lon = center_lon - delta_lon;
    let raw_max_lon = center_lon + delta_lon;
    let (min_lon, max_lon) = if raw_min_lon < -180.0 || raw_max_lon > 180.0 {
        (-180.0, 180.0)
    } else {
        (raw_min_lon, raw_max_lon)
    };

    (min_lat, max_lat, min_lon, max_lon)
}

pub struct MultiSegmentQueryIterator<'a> {
    segments: &'a [Segment],
    current_segment_idx: usize,
    current_iter: Option<CompactedQueryIterator<'a>>,
    op: &'a GeoFilterOp,
    deleted_set: &'a HashSet<u64>,
}

impl<'a> MultiSegmentQueryIterator<'a> {
    pub fn new(
        segments: &'a [Segment],
        op: &'a GeoFilterOp,
        deleted_set: &'a HashSet<u64>,
    ) -> Self {
        let mut iter = Self {
            segments,
            current_segment_idx: 0,
            current_iter: None,
            op,
            deleted_set,
        };
        iter.advance_to_next_segment();
        iter
    }

    fn advance_to_next_segment(&mut self) {
        while self.current_segment_idx < self.segments.len() {
            let segment = &self.segments[self.current_segment_idx];
            self.current_segment_idx += 1;
            if let Some(it) = segment.query_iter(self.op, self.deleted_set) {
                self.current_iter = Some(it);
                return;
            }
        }
        self.current_iter = None;
    }
}

impl Iterator for MultiSegmentQueryIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if let Some(ref mut iter) = self.current_iter {
                if let Some(doc_id) = iter.next() {
                    return Some(doc_id);
                }
            }
            // Current segment exhausted, try next
            if self.current_segment_idx >= self.segments.len() {
                return None;
            }
            self.advance_to_next_segment();
            self.current_iter.as_ref()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_bbox() {
        // Outside
        assert_eq!(
            classify_bbox(0, 10, 0, 10, 20, 30, 0, 10),
            Relation::Outside
        );
        // Inside
        assert_eq!(classify_bbox(5, 8, 5, 8, 0, 10, 0, 10), Relation::Inside);
        // Crosses
        assert_eq!(classify_bbox(0, 10, 0, 10, 5, 15, 5, 15), Relation::Crosses);
    }

    #[test]
    fn test_haversine_same_point() {
        let dist = haversine_distance(0.0, 0.0, 0.0, 0.0);
        assert!(dist.abs() < 1.0); // should be ~0
    }

    #[test]
    fn test_haversine_known_distance() {
        // Rome to Paris is approximately 1105 km
        let dist = haversine_distance(41.9028, 12.4964, 48.8566, 2.3522);
        assert!((dist - 1_105_000.0).abs() < 50_000.0);
    }

    #[test]
    fn test_bounding_box_for_radius() {
        let (min_lat, max_lat, min_lon, max_lon) = bounding_box_for_radius(0.0, 0.0, 111_000.0);
        // ~1 degree at equator
        assert!((max_lat - min_lat - 2.0).abs() < 0.1);
        assert!((max_lon - min_lon - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bounding_box_high_latitude() {
        // Bug 1 regression: asin() NaN at high latitudes with large radii
        let (min_lat, max_lat, min_lon, max_lon) = bounding_box_for_radius(80.0, 0.0, 2_000_000.0);
        assert!(!min_lat.is_nan(), "min_lat should not be NaN");
        assert!(!max_lat.is_nan(), "max_lat should not be NaN");
        assert!(!min_lon.is_nan(), "min_lon should not be NaN");
        assert!(!max_lon.is_nan(), "max_lon should not be NaN");
        // At 80° with 2000km radius, ratio > 1 so full lon span expected
        assert_eq!(min_lon, -180.0);
        assert_eq!(max_lon, 180.0);
    }

    #[test]
    fn test_bounding_box_antimeridian() {
        // Bug 2 regression: antimeridian crossing should give full lon span
        let (_min_lat, _max_lat, min_lon, max_lon) = bounding_box_for_radius(0.0, 179.0, 500_000.0);
        // 500km at the equator is ~4.5° delta_lon, so 179 + 4.5 > 180 → full span
        assert_eq!(min_lon, -180.0);
        assert_eq!(max_lon, 180.0);
    }

    #[test]
    fn test_cell_fully_inside_radius_conservative() {
        // Bug 3 regression: a cell whose corners are inside the radius
        // but has edge midpoints outside should return false.
        //
        // Construct a scenario: center at equator, small radius, cell at high latitude
        // where spherical distortion makes edges bulge outward.
        // Use a cell with large longitude span at high latitude.
        let cell_min_lat = crate::point::encode_lat(60.0);
        let cell_max_lat = crate::point::encode_lat(61.0);
        let cell_min_lon = crate::point::encode_lon(-10.0);
        let cell_max_lon = crate::point::encode_lon(10.0);

        // Center directly above the cell. All 4 corners at specific distances.
        let center_lat = 60.5;
        let center_lon = 0.0;

        // Distance to corners (approx):
        let d_corner = haversine_distance(
            center_lat,
            center_lon,
            crate::point::decode_lat(cell_min_lat),
            crate::point::decode_lon(cell_max_lon),
        );

        // With effective_radius reduction, the function should be more conservative
        // The cell diagonal is large (~1200km for this cell), so effective_radius
        // is significantly smaller than the original radius.
        let radius = d_corner + 1.0; // barely includes corners

        // Without the fix, this would return true (corners barely inside).
        // With the safety margin fix, the effective_radius is reduced by half the diagonal,
        // so corners will exceed effective_radius → returns false.
        assert!(
            !cell_fully_inside_radius(
                cell_min_lat,
                cell_max_lat,
                cell_min_lon,
                cell_max_lon,
                center_lat,
                center_lon,
                radius,
            ),
            "cell_fully_inside_radius should be conservative for large cells"
        );
    }
}
