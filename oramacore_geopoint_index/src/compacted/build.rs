#[cfg(test)]
use crate::point::EncodedPoint;
use crate::point::PointEntry;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAX_POINTS_PER_LEAF: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InnerNode {
    pub split_value: i32,
    pub split_dim: u8,
    pub _padding: [u8; 3],
}

pub fn build_bkd(points: &mut [PointEntry], version_dir: &Path) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }

    let raw_leaves = points.len().div_ceil(MAX_POINTS_PER_LEAF);
    let num_leaves = raw_leaves.next_power_of_two();
    let num_inner = num_leaves - 1;

    let mut inner_nodes: Vec<InnerNode> = vec![
        InnerNode {
            split_value: 0,
            split_dim: 0,
            _padding: [0; 3],
        };
        num_inner
    ];
    let mut leaf_offsets: Vec<u64> = vec![0u64; num_leaves];

    let leaves_path = version_dir.join("leaves.dat");
    let file = File::create(&leaves_path)
        .with_context(|| format!("Failed to create leaves.dat: {leaves_path:?}"))?;
    let mut leaf_writer = BufWriter::new(file);
    let mut leaf_offset: u64 = 0;

    build_iterative(
        points,
        &mut inner_nodes,
        num_leaves,
        &mut leaf_offsets,
        &mut leaf_writer,
        &mut leaf_offset,
    )?;

    leaf_writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush leaves.dat: {leaves_path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync leaves.dat: {leaves_path:?}"))?;

    write_inner_idx(&version_dir.join("inner.idx"), &inner_nodes, &leaf_offsets)?;

    Ok(())
}

fn build_iterative(
    points: &mut [PointEntry],
    inner_nodes: &mut [InnerNode],
    num_leaves: usize,
    leaf_offsets: &mut [u64],
    leaf_writer: &mut BufWriter<File>,
    leaf_offset: &mut u64,
) -> Result<()> {
    struct StackItem {
        node_id: usize,
        from: usize,
        to: usize,
    }

    let mut stack = Vec::with_capacity(64);
    stack.push(StackItem {
        node_id: 1,
        from: 0,
        to: points.len(),
    });

    while let Some(StackItem { node_id, from, to }) = stack.pop() {
        if node_id > inner_nodes.len() {
            // This is a leaf node
            let leaf_index = node_id - num_leaves;
            leaf_offsets[leaf_index] = *leaf_offset;

            let slice = &mut points[from..to];
            let count = slice.len() as u32;

            // Sort leaf entries by (lat, lon, doc_id)
            slice.sort_unstable_by(|a, b| {
                a.point
                    .lat
                    .cmp(&b.point.lat)
                    .then(a.point.lon.cmp(&b.point.lon))
                    .then(a.doc_id.cmp(&b.doc_id))
            });

            // Write header: count (u32) + padding (u32)
            leaf_writer.write_all(&count.to_ne_bytes())?;
            leaf_writer.write_all(&0u32.to_ne_bytes())?; // padding
            *leaf_offset += 8;

            // Write entries: (lat: i32, lon: i32, doc_id: u64) = 16 bytes each
            for entry in slice.iter() {
                leaf_writer.write_all(&entry.point.lat.to_ne_bytes())?;
                leaf_writer.write_all(&entry.point.lon.to_ne_bytes())?;
                leaf_writer.write_all(&entry.doc_id.to_ne_bytes())?;
                *leaf_offset += 16;
            }
            continue;
        }

        let left = 2 * node_id;
        let right = 2 * node_id + 1;

        if from >= to {
            // Empty partition — push children with empty range
            inner_nodes[node_id - 1] = InnerNode {
                split_value: 0,
                split_dim: 0,
                _padding: [0; 3],
            };

            // Push right first, then left, so left is processed first
            stack.push(StackItem {
                node_id: right,
                from,
                to,
            });
            stack.push(StackItem {
                node_id: left,
                from,
                to,
            });
            continue;
        }

        let split_dim = choose_split_dim(&points[from..to]);
        let mid = (from + to) / 2;

        // Partition around median using nth_element
        points[from..to].select_nth_unstable_by(mid - from, |a, b| {
            a.point.dim(split_dim).cmp(&b.point.dim(split_dim))
        });

        let split_value = points[mid].point.dim(split_dim);

        inner_nodes[node_id - 1] = InnerNode {
            split_value,
            split_dim,
            _padding: [0; 3],
        };

        // Push right first, then left, so left is processed first
        stack.push(StackItem {
            node_id: right,
            from: mid,
            to,
        });
        stack.push(StackItem {
            node_id: left,
            from,
            to: mid,
        });
    }

    Ok(())
}

fn choose_split_dim(points: &[PointEntry]) -> u8 {
    if points.is_empty() {
        return 0;
    }

    let mut lat_min = i32::MAX;
    let mut lat_max = i32::MIN;
    let mut lon_min = i32::MAX;
    let mut lon_max = i32::MIN;

    for entry in points {
        lat_min = lat_min.min(entry.point.lat);
        lat_max = lat_max.max(entry.point.lat);
        lon_min = lon_min.min(entry.point.lon);
        lon_max = lon_max.max(entry.point.lon);
    }

    let lat_spread = (lat_max as i64) - (lat_min as i64);
    let lon_spread = (lon_max as i64) - (lon_min as i64);

    if lat_spread >= lon_spread { 0 } else { 1 }
}

fn write_inner_idx(path: &Path, inner_nodes: &[InnerNode], leaf_offsets: &[u64]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create inner.idx: {path:?}"))?;
    let mut writer = BufWriter::new(file);

    // Write inner nodes (each 8 bytes)
    for node in inner_nodes {
        writer.write_all(&node.split_value.to_ne_bytes())?;
        writer.write_all(&[node.split_dim])?;
        writer.write_all(&node._padding)?;
    }

    // Write leaf offsets (each 8 bytes)
    for &offset in leaf_offsets {
        writer.write_all(&offset.to_ne_bytes())?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush inner.idx: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync inner.idx: {path:?}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::GeoPoint;
    use tempfile::TempDir;

    #[test]
    fn test_build_bkd_empty() {
        let tmp = TempDir::new().unwrap();
        let mut points: Vec<PointEntry> = vec![];
        build_bkd(&mut points, tmp.path()).unwrap();
        // No files should be created for empty input
        assert!(!tmp.path().join("inner.idx").exists());
    }

    #[test]
    fn test_build_bkd_single_point() {
        let tmp = TempDir::new().unwrap();
        let p = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let mut points = vec![PointEntry {
            point: p,
            doc_id: 1,
        }];
        build_bkd(&mut points, tmp.path()).unwrap();
        assert!(tmp.path().join("inner.idx").exists());
        assert!(tmp.path().join("leaves.dat").exists());
    }

    #[test]
    fn test_build_bkd_many_points() {
        let tmp = TempDir::new().unwrap();
        let mut points: Vec<PointEntry> = (0..1000)
            .map(|i| {
                let lat = -90.0 + (i as f64 * 180.0 / 1000.0);
                let lon = -180.0 + (i as f64 * 360.0 / 1000.0);
                let p = GeoPoint::new(lat, lon).unwrap().encode();
                PointEntry {
                    point: p,
                    doc_id: i as u64,
                }
            })
            .collect();
        build_bkd(&mut points, tmp.path()).unwrap();
        assert!(tmp.path().join("inner.idx").exists());
        assert!(tmp.path().join("leaves.dat").exists());
    }

    #[test]
    fn test_choose_split_dim_lat_spread() {
        let points: Vec<PointEntry> = vec![
            PointEntry {
                point: EncodedPoint { lat: -1000, lon: 0 },
                doc_id: 0,
            },
            PointEntry {
                point: EncodedPoint { lat: 1000, lon: 10 },
                doc_id: 1,
            },
        ];
        assert_eq!(choose_split_dim(&points), 0); // lat spread = 2000 > lon spread = 10
    }

    #[test]
    fn test_choose_split_dim_lon_spread() {
        let points: Vec<PointEntry> = vec![
            PointEntry {
                point: EncodedPoint { lat: 0, lon: -1000 },
                doc_id: 0,
            },
            PointEntry {
                point: EncodedPoint { lat: 10, lon: 1000 },
                doc_id: 1,
            },
        ];
        assert_eq!(choose_split_dim(&points), 1); // lon spread = 2000 > lat spread = 10
    }
}
