pub mod build;
pub mod query;

use super::io::{read_manifest, segment_subdir, version_dir};
use super::iterator::GeoFilterOp;
use super::platform::{advise_random, advise_sequential, advise_willneed};
use super::point::{encode_lat, encode_lon, EncodedPoint};
use anyhow::{Context, Result};
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub(crate) const LEAF_HEADER_SIZE: usize = 8;
pub(crate) const LEAF_ENTRY_SIZE: usize = 16;

pub(crate) struct InnerNodesView<'a> {
    data: &'a [u8],
}

impl InnerNodesView<'_> {
    /// node_id is 1-indexed (BKD tree convention). Reads from mmap offset (node_id - 1) * 8.
    #[inline]
    pub(crate) fn get(&self, node_id: usize) -> (i32, u8) {
        let pos = (node_id - 1) * 8;
        let split_value = i32::from_ne_bytes(self.data[pos..pos + 4].try_into().unwrap());
        let split_dim = self.data[pos + 4];
        (split_value, split_dim)
    }
}

pub(crate) struct LeafOffsetsView<'a> {
    data: &'a [u8],
    num_leaves: usize,
}

impl LeafOffsetsView<'_> {
    #[inline]
    pub(crate) fn get(&self, index: usize) -> u64 {
        let pos = index * 8;
        u64::from_ne_bytes(self.data[pos..pos + 8].try_into().unwrap())
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.num_leaves
    }
}

pub(crate) struct Segment {
    pub point_count: u64,
    inner_mmap: Option<Mmap>,
    leaves_mmap: Option<Mmap>,
    num_inner: usize,
    num_leaves: usize,
    global_min_lat: i32,
    global_max_lat: i32,
    global_min_lon: i32,
    global_max_lon: i32,
}

impl Segment {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let inner_mmap = Self::load_mmap(&dir.join("inner.idx"))?;
        let leaves_mmap = Self::load_mmap(&dir.join("leaves.dat"))?;

        // Apply madvise hints
        if let Some(ref m) = inner_mmap {
            advise_willneed(m);
        }
        if let Some(ref m) = leaves_mmap {
            advise_random(m);
        }

        // Parse inner.idx to determine tree structure
        let (num_inner, num_leaves) = if let Some(ref mmap) = inner_mmap {
            Self::parse_tree_dimensions(mmap)?
        } else {
            (0, 0)
        };

        // Compute global bounds from the tree
        let (global_min_lat, global_max_lat, global_min_lon, global_max_lon) =
            if let (Some(leaves), Some(_)) = (&leaves_mmap, &inner_mmap) {
                if num_leaves > 0 {
                    Self::compute_global_bounds(leaves, num_leaves, &inner_mmap)
                } else {
                    (i32::MIN, i32::MAX, i32::MIN, i32::MAX)
                }
            } else {
                (i32::MIN, i32::MAX, i32::MIN, i32::MAX)
            };

        // Count total points
        let point_count = Self::count_points(&leaves_mmap, num_leaves, &inner_mmap);

        Ok(Self {
            point_count,
            inner_mmap,
            leaves_mmap,
            num_inner,
            num_leaves,
            global_min_lat,
            global_max_lat,
            global_min_lon,
            global_max_lon,
        })
    }

    fn load_mmap(path: &Path) -> Result<Option<Mmap>> {
        if !path.exists() {
            return Ok(None);
        }

        let file =
            File::open(path).with_context(|| format!("Failed to open file for mmap: {path:?}"))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get file metadata: {path:?}"))?;

        if metadata.len() == 0 {
            return Ok(None);
        }

        let mmap =
            unsafe { Mmap::map(&file).with_context(|| format!("Failed to mmap file: {path:?}"))? };

        Ok(Some(mmap))
    }

    fn parse_tree_dimensions(inner_mmap: &Mmap) -> Result<(usize, usize)> {
        let total_bytes = inner_mmap.len();

        if !total_bytes.is_multiple_of(8) || total_bytes < 8 {
            if total_bytes == 0 {
                return Ok((0, 0));
            }
            anyhow::bail!("inner.idx has invalid size {total_bytes}, must be a multiple of 8");
        }

        let total_entries = total_bytes / 8;
        if total_entries.is_multiple_of(2) {
            anyhow::bail!(
                "inner.idx has {total_entries} entries, expected odd number (2*num_inner + 1)"
            );
        }

        let num_inner = (total_entries - 1) / 2;
        let num_leaves = num_inner + 1;

        Ok((num_inner, num_leaves))
    }

    fn compute_global_bounds(
        leaves: &Mmap,
        num_leaves: usize,
        inner_mmap: &Option<Mmap>,
    ) -> (i32, i32, i32, i32) {
        let mmap = inner_mmap.as_ref().unwrap();
        let num_inner = num_leaves - 1;
        let offsets_start = num_inner * 8;
        let offsets_data = &mmap.as_ref()[offsets_start..offsets_start + num_leaves * 8];
        let leaf_offsets = LeafOffsetsView {
            data: offsets_data,
            num_leaves,
        };
        let leaf_data = leaves.as_ref();

        let mut min_lat = i32::MAX;
        let mut max_lat = i32::MIN;
        let mut min_lon = i32::MAX;
        let mut max_lon = i32::MIN;

        for leaf_idx in 0..num_leaves {
            let offset = leaf_offsets.get(leaf_idx) as usize;
            if offset + LEAF_HEADER_SIZE > leaf_data.len() {
                continue;
            }
            let count =
                u32::from_ne_bytes(leaf_data[offset..offset + 4].try_into().unwrap()) as usize;
            if count == 0 {
                continue;
            }
            let entries_start = offset + LEAF_HEADER_SIZE;
            for i in 0..count {
                let entry_offset = entries_start + i * LEAF_ENTRY_SIZE;
                if entry_offset + LEAF_ENTRY_SIZE > leaf_data.len() {
                    break;
                }
                let lat = i32::from_ne_bytes(
                    leaf_data[entry_offset..entry_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let lon = i32::from_ne_bytes(
                    leaf_data[entry_offset + 4..entry_offset + 8]
                        .try_into()
                        .unwrap(),
                );
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
            }
        }

        if min_lat > max_lat {
            (i32::MIN, i32::MAX, i32::MIN, i32::MAX)
        } else {
            (min_lat, max_lat, min_lon, max_lon)
        }
    }

    fn count_points(
        leaves_mmap: &Option<Mmap>,
        num_leaves: usize,
        inner_mmap: &Option<Mmap>,
    ) -> u64 {
        let Some(inner) = inner_mmap.as_ref() else {
            return 0;
        };
        let Some(leaves) = leaves_mmap.as_ref() else {
            return 0;
        };
        if num_leaves == 0 {
            return 0;
        }

        let num_inner = num_leaves - 1;
        let offsets_start = num_inner * 8;
        let offsets_data = &inner.as_ref()[offsets_start..offsets_start + num_leaves * 8];
        let leaf_offsets = LeafOffsetsView {
            data: offsets_data,
            num_leaves,
        };
        let leaf_data = leaves.as_ref();

        let mut total = 0u64;

        for leaf_idx in 0..num_leaves {
            let offset = leaf_offsets.get(leaf_idx) as usize;
            if offset + LEAF_HEADER_SIZE > leaf_data.len() {
                continue;
            }
            let count =
                u32::from_ne_bytes(leaf_data[offset..offset + 4].try_into().unwrap()) as u64;
            total += count;
        }
        total
    }

    pub(crate) fn inner_nodes_view(&self) -> Option<InnerNodesView<'_>> {
        self.inner_mmap.as_ref().map(|mmap| {
            let data = &mmap.as_ref()[..self.num_inner * 8];
            InnerNodesView { data }
        })
    }

    pub(crate) fn leaf_offsets_view(&self) -> Option<LeafOffsetsView<'_>> {
        self.inner_mmap.as_ref().map(|mmap| {
            let offsets_start = self.num_inner * 8;
            let data = &mmap.as_ref()[offsets_start..offsets_start + self.num_leaves * 8];
            LeafOffsetsView {
                data,
                num_leaves: self.num_leaves,
            }
        })
    }

    pub(crate) fn get_leaf_data(&self) -> &[u8] {
        match &self.leaves_mmap {
            None => &[],
            Some(mmap) => mmap.as_ref(),
        }
    }

    pub fn has_data(&self) -> bool {
        self.inner_mmap.is_some() && self.leaves_mmap.is_some()
    }

    pub fn query_iter<'a>(
        &'a self,
        op: &GeoFilterOp,
        deleted_set: &'a HashSet<u64>,
    ) -> Option<query::CompactedQueryIterator<'a>> {
        if !self.has_data() {
            return None;
        }

        let inner_nodes = self.inner_nodes_view()?;
        let leaf_offsets = self.leaf_offsets_view()?;
        let leaf_data = self.get_leaf_data();

        match op {
            GeoFilterOp::BoundingBox {
                min_lat,
                max_lat,
                min_lon,
                max_lon,
            } => {
                let enc_min_lat = encode_lat(*min_lat);
                let enc_max_lat = encode_lat(*max_lat);
                let enc_min_lon = encode_lon(*min_lon);
                let enc_max_lon = encode_lon(*max_lon);

                Some(query::CompactedQueryIterator::new_bbox(
                    inner_nodes,
                    self.num_leaves,
                    leaf_offsets,
                    leaf_data,
                    enc_min_lat,
                    enc_max_lat,
                    enc_min_lon,
                    enc_max_lon,
                    self.global_min_lat,
                    self.global_max_lat,
                    self.global_min_lon,
                    self.global_max_lon,
                    deleted_set,
                ))
            }
            GeoFilterOp::Radius {
                center,
                radius_meters,
            } => Some(query::CompactedQueryIterator::new_radius(
                inner_nodes,
                self.num_leaves,
                leaf_offsets,
                leaf_data,
                center.lat(),
                center.lon(),
                *radius_meters,
                self.global_min_lat,
                self.global_max_lat,
                self.global_min_lon,
                self.global_max_lon,
                deleted_set,
            )),
        }
    }

    pub fn iter_all_points(&self) -> Option<SegmentPointIter<'_>> {
        if !self.has_data() {
            return None;
        }

        let leaf_offsets = self.leaf_offsets_view()?;
        let leaf_data = self.get_leaf_data();
        let num_leaves = self.num_leaves;

        Some(SegmentPointIter {
            leaf_offsets,
            leaf_data,
            num_leaves,
            current_leaf: 0,
            entries_start: 0,
            entry_count: 0,
            current_entry: 0,
        })
    }

    #[allow(dead_code)]
    pub fn collect_all_points(&self) -> Vec<(EncodedPoint, u64)> {
        self.iter_all_points()
            .map_or_else(Vec::new, |it| it.collect())
    }
}

pub(crate) struct SegmentPointIter<'a> {
    leaf_offsets: LeafOffsetsView<'a>,
    leaf_data: &'a [u8],
    num_leaves: usize,
    current_leaf: usize,
    entries_start: usize,
    entry_count: usize,
    current_entry: usize,
}

impl Iterator for SegmentPointIter<'_> {
    type Item = (EncodedPoint, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Yield next entry from current leaf
            if self.current_entry < self.entry_count {
                let entry_offset = self.entries_start + self.current_entry * LEAF_ENTRY_SIZE;
                self.current_entry += 1;

                if entry_offset + LEAF_ENTRY_SIZE > self.leaf_data.len() {
                    // Truncated data, skip rest of this leaf
                    self.current_entry = self.entry_count;
                    continue;
                }

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
                let doc_id = u64::from_ne_bytes(
                    self.leaf_data[entry_offset + 8..entry_offset + 16]
                        .try_into()
                        .unwrap(),
                );
                return Some((EncodedPoint { lat, lon }, doc_id));
            }

            // Advance to next non-empty leaf
            while self.current_leaf < self.num_leaves {
                let leaf_idx = self.current_leaf;
                self.current_leaf += 1;

                let offset = self.leaf_offsets.get(leaf_idx) as usize;
                if offset + LEAF_HEADER_SIZE > self.leaf_data.len() {
                    continue;
                }

                let count =
                    u32::from_ne_bytes(self.leaf_data[offset..offset + 4].try_into().unwrap())
                        as usize;
                if count == 0 {
                    continue;
                }

                self.entries_start = offset + LEAF_HEADER_SIZE;
                self.entry_count = count;
                self.current_entry = 0;
                break;
            }

            // No more leaves
            if self.current_entry >= self.entry_count {
                return None;
            }
        }
    }
}

pub(crate) struct CompactedPointIter<'a> {
    segments: &'a [Segment],
    current_segment_idx: usize,
    current_iter: Option<SegmentPointIter<'a>>,
}

impl<'a> CompactedPointIter<'a> {
    fn advance_to_next_segment(&mut self) {
        while self.current_segment_idx < self.segments.len() {
            let segment = &self.segments[self.current_segment_idx];
            self.current_segment_idx += 1;
            if let Some(it) = segment.iter_all_points() {
                self.current_iter = Some(it);
                return;
            }
        }
        self.current_iter = None;
    }
}

impl Iterator for CompactedPointIter<'_> {
    type Item = (EncodedPoint, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut iter) = self.current_iter {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }
            if self.current_segment_idx >= self.segments.len() {
                return None;
            }
            self.advance_to_next_segment();
            self.current_iter.as_ref()?;
        }
    }
}

pub struct CompactedVersion {
    pub version_id: u64,
    pub(crate) segments: Vec<Segment>,
    deleted_mmap: Option<Mmap>,
    pub deleted_set: Arc<HashSet<u64>>,
}

impl CompactedVersion {
    pub fn empty() -> Self {
        Self {
            version_id: 0,
            segments: Vec::new(),
            deleted_mmap: None,
            deleted_set: Arc::new(HashSet::new()),
        }
    }

    pub fn load(base_path: &Path, version_id: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_id);

        // Try to read manifest (FORMAT_VERSION 2)
        let manifest_path = dir.join("manifest.bin");
        if manifest_path.exists() {
            return Self::load_v2(base_path, version_id);
        }

        // Legacy FORMAT_VERSION 1: single segment with files directly in version dir
        Self::load_v1(base_path, version_id)
    }

    fn load_v1(base_path: &Path, version_id: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_id);

        let segment = Segment::load_from_dir(&dir)?;

        let deleted_mmap = Self::load_mmap(&dir.join("deleted.bin"))?;

        if let Some(ref m) = deleted_mmap {
            advise_sequential(m);
        }

        let deleted_set = Self::load_deleted_set(deleted_mmap.as_ref())?;

        Ok(Self {
            version_id,
            segments: vec![segment],
            deleted_mmap,
            deleted_set: Arc::new(deleted_set),
        })
    }

    fn load_v2(base_path: &Path, version_id: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_id);

        let point_counts = read_manifest(&dir)?;

        let mut segments = Vec::with_capacity(point_counts.len());
        for (i, &_count) in point_counts.iter().enumerate() {
            let seg_dir = segment_subdir(&dir, i);
            let segment = Segment::load_from_dir(&seg_dir)
                .with_context(|| format!("Failed to load segment_{i}"))?;
            segments.push(segment);
        }

        let deleted_mmap = Self::load_mmap(&dir.join("deleted.bin"))?;

        if let Some(ref m) = deleted_mmap {
            advise_sequential(m);
        }

        let deleted_set = Self::load_deleted_set(deleted_mmap.as_ref())?;

        Ok(Self {
            version_id,
            segments,
            deleted_mmap,
            deleted_set: Arc::new(deleted_set),
        })
    }

    fn load_mmap(path: &Path) -> Result<Option<Mmap>> {
        if !path.exists() {
            return Ok(None);
        }

        let file =
            File::open(path).with_context(|| format!("Failed to open file for mmap: {path:?}"))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get file metadata: {path:?}"))?;

        if metadata.len() == 0 {
            return Ok(None);
        }

        let mmap =
            unsafe { Mmap::map(&file).with_context(|| format!("Failed to mmap file: {path:?}"))? };

        Ok(Some(mmap))
    }

    fn load_deleted_set(deleted: Option<&Mmap>) -> Result<HashSet<u64>> {
        match deleted {
            Some(m) => {
                if !m.len().is_multiple_of(8) {
                    anyhow::bail!(
                        "deleted.bin has invalid size {}, must be a multiple of 8",
                        m.len()
                    );
                }
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                Ok(slice.iter().copied().collect())
            }
            None => Ok(HashSet::new()),
        }
    }

    pub fn has_data(&self) -> bool {
        self.segments.iter().any(|s| s.has_data())
    }

    pub fn query_iter<'a>(
        &'a self,
        op: &'a GeoFilterOp,
    ) -> Option<query::MultiSegmentQueryIterator<'a>> {
        if !self.has_data() {
            return None;
        }

        Some(query::MultiSegmentQueryIterator::new(
            &self.segments,
            op,
            &self.deleted_set,
        ))
    }

    pub fn iter_all_points(&self) -> CompactedPointIter<'_> {
        let mut iter = CompactedPointIter {
            segments: &self.segments,
            current_segment_idx: 0,
            current_iter: None,
        };
        iter.advance_to_next_segment();
        iter
    }

    #[allow(dead_code)]
    pub fn collect_all_points(&self) -> Vec<(EncodedPoint, u64)> {
        self.iter_all_points().collect()
    }

    pub fn total_point_count(&self) -> u64 {
        self.segments.iter().map(|s| s.point_count).sum()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn deleted_slice(&self) -> &[u64] {
        match &self.deleted_mmap {
            Some(m) => {
                if !m.len().is_multiple_of(8) {
                    tracing::warn!(
                        "deleted.bin has invalid size {}, not a multiple of 8; trailing bytes ignored",
                        m.len()
                    );
                }
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::io::{ensure_version_dir, write_u64_slice};
    use super::super::point::{GeoPoint, PointEntry};
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_empty_version() {
        let version = CompactedVersion::empty();
        assert_eq!(version.version_id, 0);
        assert!(!version.has_data());
        assert!(version.deleted_set.is_empty());
    }

    #[test]
    fn test_load_and_query_single_point() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let p = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let mut points = vec![PointEntry {
            point: p,
            doc_id: 42,
        }];
        build::build_bkd(&mut points, &ver_dir).unwrap();
        write_u64_slice(&ver_dir.join("deleted.bin"), &[]).unwrap();

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert!(version.has_data());

        let results: Vec<u64> = version
            .query_iter(&GeoFilterOp::BoundingBox {
                min_lat: 5.0,
                max_lat: 15.0,
                min_lon: 15.0,
                max_lon: 25.0,
            })
            .map_or_else(Vec::new, |it| it.collect());
        assert_eq!(results, vec![42]);
    }

    #[test]
    fn test_load_and_query_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let p2 = GeoPoint::new(11.0, 21.0).unwrap().encode();
        let mut points = vec![
            PointEntry {
                point: p1,
                doc_id: 1,
            },
            PointEntry {
                point: p2,
                doc_id: 2,
            },
        ];
        build::build_bkd(&mut points, &ver_dir).unwrap();
        write_u64_slice(&ver_dir.join("deleted.bin"), &[1]).unwrap();

        let version = CompactedVersion::load(base_path, 1).unwrap();
        let results: Vec<u64> = version
            .query_iter(&GeoFilterOp::BoundingBox {
                min_lat: 5.0,
                max_lat: 15.0,
                min_lon: 15.0,
                max_lon: 25.0,
            })
            .map_or_else(Vec::new, |it| it.collect());
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn test_collect_all_points() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap().encode();
        let mut points = vec![
            PointEntry {
                point: p1,
                doc_id: 1,
            },
            PointEntry {
                point: p2,
                doc_id: 2,
            },
        ];
        build::build_bkd(&mut points, &ver_dir).unwrap();
        write_u64_slice(&ver_dir.join("deleted.bin"), &[]).unwrap();

        let version = CompactedVersion::load(base_path, 1).unwrap();
        let all = version.collect_all_points();
        assert_eq!(all.len(), 2);
        let doc_ids: Vec<u64> = all.iter().map(|(_, id)| *id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));
    }

    #[test]
    fn test_corrupted_deleted_bin_rejected() {
        // Bug 5 regression: deleted.bin with non-multiple-of-8 size should error
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let p = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let mut points = vec![PointEntry {
            point: p,
            doc_id: 42,
        }];
        build::build_bkd(&mut points, &ver_dir).unwrap();

        // Write a corrupted deleted.bin (5 bytes, not a multiple of 8)
        std::fs::write(ver_dir.join("deleted.bin"), [0u8; 5]).unwrap();

        let result = CompactedVersion::load(base_path, 1);
        match result {
            Ok(_) => panic!("should reject deleted.bin with invalid size"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("multiple of 8"),
                    "error should mention alignment: {err_msg}"
                );
            }
        }
    }

    #[test]
    fn test_iter_all_points() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap().encode();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap().encode();
        let mut points = vec![
            PointEntry {
                point: p1,
                doc_id: 1,
            },
            PointEntry {
                point: p2,
                doc_id: 2,
            },
        ];
        build::build_bkd(&mut points, &ver_dir).unwrap();
        write_u64_slice(&ver_dir.join("deleted.bin"), &[]).unwrap();

        let version = CompactedVersion::load(base_path, 1).unwrap();

        let collected = version.collect_all_points();
        let iterated: Vec<(EncodedPoint, u64)> = version.iter_all_points().collect();

        assert_eq!(collected.len(), iterated.len());
        // Both should contain the same elements (order may differ across methods)
        for item in &collected {
            assert!(iterated.contains(item));
        }
        for item in &iterated {
            assert!(collected.contains(item));
        }
    }

    #[test]
    fn test_iter_empty_segment() {
        let version = CompactedVersion::empty();
        let items: Vec<(EncodedPoint, u64)> = version.iter_all_points().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_iter_all_points_many() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();

        let n = 1000;
        let mut points: Vec<PointEntry> = (0..n)
            .map(|i| {
                let lat = -80.0 + (i as f64) * 0.15;
                let lon = -170.0 + (i as f64) * 0.33;
                let p = GeoPoint::new(lat, lon).unwrap().encode();
                PointEntry {
                    point: p,
                    doc_id: i as u64,
                }
            })
            .collect();
        build::build_bkd(&mut points, &ver_dir).unwrap();
        write_u64_slice(&ver_dir.join("deleted.bin"), &[]).unwrap();

        let version = CompactedVersion::load(base_path, 1).unwrap();

        let iterated: Vec<(EncodedPoint, u64)> = version.iter_all_points().collect();
        assert_eq!(iterated.len(), n);

        let mut doc_ids: Vec<u64> = iterated.iter().map(|(_, id)| *id).collect();
        doc_ids.sort_unstable();
        doc_ids.dedup();
        assert_eq!(doc_ids.len(), n);
    }
}
