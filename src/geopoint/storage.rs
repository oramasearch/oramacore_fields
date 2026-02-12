use super::compacted::build::build_bkd;
use super::compacted::CompactedVersion;
use super::config::Threshold;
use super::indexer::IndexedValue;
use super::info::{CheckStatus, IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    copy_dir_contents, ensure_segment_subdir, ensure_version_dir, list_version_dirs, read_current,
    read_manifest, remove_version_dir, segment_subdir, sync_dir, version_dir, write_current_atomic,
    write_manifest, write_u64_slice, FORMAT_VERSION,
};
use super::iterator::{FilterData, GeoFilterOp};
use super::live::{LiveLayer, LiveOp};
use super::mmap_vec::{dedup_sorted, retain_in_place, MmapVecWriter};
use super::point::PointEntry;
use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

pub struct GeoPointStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    threshold: Threshold,
    max_segments: usize,
}

fn merge_sorted_unique(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Selects which segments to merge by choosing the smallest ones.
fn select_segments_to_merge(point_counts: &[u64]) -> Vec<usize> {
    let n = point_counts.len();
    if n <= 1 {
        return (0..n).collect();
    }

    let merge_count = n.div_ceil(2).max(2).min(n);

    // Create (index, count) pairs, sort by count ascending
    let mut indexed: Vec<(usize, u64)> = point_counts.iter().copied().enumerate().collect();
    indexed.sort_by_key(|&(_, count)| count);

    let mut indices: Vec<usize> = indexed[..merge_count].iter().map(|&(i, _)| i).collect();
    indices.sort_unstable();
    indices
}

impl GeoPointStorage {
    pub fn new(base_path: PathBuf, threshold: Threshold, max_segments: usize) -> Result<Self> {
        std::fs::create_dir_all(&base_path)
            .with_context(|| format!("Failed to create base directory: {base_path:?}"))?;

        let version = match read_current(&base_path)? {
            Some((format_version, version_id)) => {
                if format_version != FORMAT_VERSION && format_version != 1 {
                    return Err(anyhow!(
                        "Unsupported format version {format_version}, expected {FORMAT_VERSION}"
                    ));
                }
                CompactedVersion::load(&base_path, version_id)
                    .with_context(|| format!("Failed to load version {version_id}"))?
            }
            None => CompactedVersion::empty(),
        };

        Ok(Self {
            base_path,
            version: ArcSwap::new(Arc::new(version)),
            live: RwLock::new(LiveLayer::new()),
            compaction_lock: Mutex::new(()),
            threshold,
            max_segments,
        })
    }

    pub fn insert(&self, value: IndexedValue, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        match value {
            IndexedValue::Plain(point) => {
                live.insert(point, doc_id);
            }
            IndexedValue::Array(points) => {
                for point in points {
                    live.insert(point, doc_id);
                }
            }
        }
    }

    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    pub fn filter(&self, op: GeoFilterOp) -> FilterData {
        let snapshot = {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                live.get_snapshot()
            } else {
                drop(live);
                let mut live = self.live.write().unwrap();
                if live.is_snapshot_dirty() {
                    live.refresh_snapshot();
                }
                live.get_snapshot()
            }
        };
        let version = self.version.load();
        FilterData::new(Arc::clone(&version), snapshot, op)
    }

    pub fn compact(&self, version_id: u64) -> Result<()> {
        let compaction_guard = self.compaction_lock.lock().unwrap();

        // Refresh snapshot under write lock
        let snapshot = {
            let mut live = self.live.write().unwrap();
            if live.is_snapshot_dirty() {
                live.refresh_snapshot();
            }
            live.get_snapshot()
        };

        let current = self.version.load();

        // Combine all delete sets using sorted Vec merge
        let mut snapshot_deletes: Vec<u64> = snapshot.deletes.iter().copied().collect();
        snapshot_deletes.sort_unstable();
        snapshot_deletes.dedup();
        let all_deletes = merge_sorted_unique(&snapshot_deletes, current.deleted_slice());

        // Remove from deletes any doc_ids in live inserts (re-insert supersedes delete)
        let mut live_doc_ids: Vec<u64> = snapshot.inserts.iter().map(|(_, id)| *id).collect();
        live_doc_ids.sort_unstable();
        live_doc_ids.dedup();
        let all_deletes: Vec<u64> = all_deletes
            .into_iter()
            .filter(|id| live_doc_ids.binary_search(id).is_err())
            .collect();

        // Decide: hot path or full compact?
        let segment_count = current.segment_count();
        let total_existing_points = current.total_point_count();
        let live_insert_count = snapshot.inserts.len() as u64;
        let total_points = total_existing_points + live_insert_count;

        let delete_ratio = if total_points > 0 {
            all_deletes.len() as f64 / total_points as f64
        } else {
            0.0
        };

        if delete_ratio > self.threshold.value() || segment_count == 0 {
            self.full_compact(
                version_id,
                &current,
                &snapshot,
                all_deletes,
                delete_ratio > self.threshold.value(),
            )?;
        } else if segment_count >= self.max_segments {
            self.partial_merge_compact(version_id, &current, &snapshot, all_deletes)?;
        } else {
            self.hot_compact(version_id, &current, &snapshot, all_deletes)?;
        }

        // Atomic update: swap version AND clear compacted items
        {
            let mut live = self.live.write().unwrap();

            let new_version = CompactedVersion::load(&self.base_path, version_id)?;
            self.version.store(Arc::new(new_version));

            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
        }

        drop(compaction_guard);

        Ok(())
    }

    /// Compacts by writing only new inserts as a new segment, keeping existing segments.
    fn hot_compact(
        &self,
        version_id: u64,
        current: &CompactedVersion,
        snapshot: &super::live::LiveSnapshot,
        all_deletes: Vec<u64>,
    ) -> Result<()> {
        let new_version_dir = ensure_version_dir(&self.base_path, version_id)?;
        let current_dir = version_dir(&self.base_path, current.version_id);

        // Read current manifest to get existing segment info
        let old_point_counts: Vec<u64> = if current_dir.join("manifest.bin").exists() {
            read_manifest(&current_dir)?
        } else {
            // Legacy v1 format: one segment in version dir root
            current.segments.iter().map(|s| s.point_count).collect()
        };

        let old_segment_count = old_point_counts.len();

        // Copy old segment dirs into new version dir
        for i in 0..old_segment_count {
            let old_seg_dir = if current_dir.join("manifest.bin").exists() {
                segment_subdir(&current_dir, i)
            } else {
                // v1 format: files are directly in version dir
                current_dir.clone()
            };
            let new_seg_dir = ensure_segment_subdir(&new_version_dir, i)?;
            copy_dir_contents(&old_seg_dir, &new_seg_dir)?;
        }

        // Build BKD from live inserts only
        let new_segment_point_count;
        if !snapshot.inserts.is_empty() {
            let new_seg_dir = ensure_segment_subdir(&new_version_dir, old_segment_count)?;
            let mut writer = MmapVecWriter::new(&new_seg_dir)?;
            for (p, id) in snapshot.inserts.iter() {
                writer.push(&PointEntry {
                    point: p.encode(),
                    doc_id: *id,
                })?;
            }
            let mut mmap_vec = writer.finish()?;
            let slice = mmap_vec.as_mut_slice();
            slice.sort_unstable_by(|a, b| {
                a.point
                    .lat
                    .cmp(&b.point.lat)
                    .then(a.point.lon.cmp(&b.point.lon))
                    .then(a.doc_id.cmp(&b.doc_id))
            });
            let new_len = dedup_sorted(slice);
            mmap_vec.set_len(new_len);

            new_segment_point_count = mmap_vec.len() as u64;
            if !mmap_vec.is_empty() {
                build_bkd(mmap_vec.as_mut_slice(), &new_seg_dir)?;
            }
        } else {
            new_segment_point_count = 0;
        }

        // Write manifest
        let mut point_counts = old_point_counts;
        if new_segment_point_count > 0 {
            point_counts.push(new_segment_point_count);
        }
        write_manifest(&new_version_dir, &point_counts)?;

        // Write deleted.bin (already sorted)
        write_u64_slice(&new_version_dir.join("deleted.bin"), &all_deletes)?;

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_id)?;

        Ok(())
    }

    /// Merges the smallest segments together while copying the rest unchanged.
    fn partial_merge_compact(
        &self,
        version_id: u64,
        current: &CompactedVersion,
        snapshot: &super::live::LiveSnapshot,
        all_deletes: Vec<u64>,
    ) -> Result<()> {
        let new_version_dir = ensure_version_dir(&self.base_path, version_id)?;
        let current_dir = version_dir(&self.base_path, current.version_id);

        // Read current manifest to get existing segment info
        let old_point_counts: Vec<u64> = if current_dir.join("manifest.bin").exists() {
            read_manifest(&current_dir)?
        } else {
            current.segments.iter().map(|s| s.point_count).collect()
        };

        // Select which segments to merge (smallest ones)
        let merge_indices = select_segments_to_merge(&old_point_counts);

        // Phase A: Copy kept segments into new version dir at sequential slot indices
        let mut new_point_counts: Vec<u64> = Vec::new();
        let mut slot = 0usize;
        for (i, &count) in old_point_counts.iter().enumerate() {
            if merge_indices.binary_search(&i).is_ok() {
                continue; // Will be merged
            }
            let old_seg_dir = if current_dir.join("manifest.bin").exists() {
                segment_subdir(&current_dir, i)
            } else {
                current_dir.clone()
            };
            let new_seg_dir = ensure_segment_subdir(&new_version_dir, slot)?;
            copy_dir_contents(&old_seg_dir, &new_seg_dir)?;
            new_point_counts.push(count);
            slot += 1;
        }

        // Phase B: Merge selected segments + live inserts into one new segment
        let merge_seg_dir = ensure_segment_subdir(&new_version_dir, slot)?;
        let mut writer = MmapVecWriter::new(&merge_seg_dir)?;

        // Read points from selected segments
        for &idx in &merge_indices {
            if let Some(iter) = current.segments[idx].iter_all_points() {
                for (pt, doc_id) in iter {
                    writer.push(&PointEntry { point: pt, doc_id })?;
                }
            }
        }

        // Add live inserts
        for (p, id) in snapshot.inserts.iter() {
            writer.push(&PointEntry {
                point: p.encode(),
                doc_id: *id,
            })?;
        }

        let mut mmap_vec = writer.finish()?;
        let slice = mmap_vec.as_mut_slice();
        slice.sort_unstable_by(|a, b| {
            a.point
                .lat
                .cmp(&b.point.lat)
                .then(a.point.lon.cmp(&b.point.lon))
                .then(a.doc_id.cmp(&b.doc_id))
        });
        let new_len = dedup_sorted(slice);
        mmap_vec.set_len(new_len);

        let merged_point_count = mmap_vec.len() as u64;
        if !mmap_vec.is_empty() {
            build_bkd(mmap_vec.as_mut_slice(), &merge_seg_dir)?;
        }

        if merged_point_count > 0 {
            new_point_counts.push(merged_point_count);
        }

        // Phase C: Write metadata
        write_manifest(&new_version_dir, &new_point_counts)?;
        write_u64_slice(&new_version_dir.join("deleted.bin"), &all_deletes)?;

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_id)?;

        Ok(())
    }

    /// Rebuilds all data into a single segment, optionally applying deletes.
    fn full_compact(
        &self,
        version_id: u64,
        current: &CompactedVersion,
        snapshot: &super::live::LiveSnapshot,
        all_deletes: Vec<u64>,
        should_apply_deletes: bool,
    ) -> Result<()> {
        let new_version_dir = ensure_version_dir(&self.base_path, version_id)?;

        // Stream all existing points + live inserts into mmap'd temp file
        let mut writer = MmapVecWriter::new(&new_version_dir)?;
        for (pt, doc_id) in current.iter_all_points() {
            writer.push(&PointEntry { point: pt, doc_id })?;
        }
        for (p, id) in snapshot.inserts.iter() {
            writer.push(&PointEntry {
                point: p.encode(),
                doc_id: *id,
            })?;
        }
        let mut mmap_vec = writer.finish()?;

        // Deduplicate exact (point, doc_id) pairs
        let slice = mmap_vec.as_mut_slice();
        slice.sort_unstable_by(|a, b| {
            a.point
                .lat
                .cmp(&b.point.lat)
                .then(a.point.lon.cmp(&b.point.lon))
                .then(a.doc_id.cmp(&b.doc_id))
        });
        let new_len = dedup_sorted(slice);
        mmap_vec.set_len(new_len);

        // Prune stale deletes: only keep deletes for doc_ids that exist
        let mut point_doc_ids: Vec<u64> =
            mmap_vec.as_mut_slice().iter().map(|e| e.doc_id).collect();
        point_doc_ids.sort_unstable();
        point_doc_ids.dedup();
        let all_deletes: Vec<u64> = all_deletes
            .into_iter()
            .filter(|id| point_doc_ids.binary_search(id).is_ok())
            .collect();

        let seg_dir = ensure_segment_subdir(&new_version_dir, 0)?;

        if should_apply_deletes {
            // Strategy A: Apply deletions
            let retained = retain_in_place(mmap_vec.as_mut_slice(), |e| {
                all_deletes.binary_search(&e.doc_id).is_err()
            });
            mmap_vec.set_len(retained);
            if !mmap_vec.is_empty() {
                build_bkd(mmap_vec.as_mut_slice(), &seg_dir)?;
            }
            write_u64_slice(&new_version_dir.join("deleted.bin"), &[])?;
            let point_count = mmap_vec.len() as u64;
            if point_count > 0 {
                write_manifest(&new_version_dir, &[point_count])?;
            } else {
                write_manifest(&new_version_dir, &[])?;
            }
        } else {
            // Strategy B: Carry forward deletions
            if !mmap_vec.is_empty() {
                let point_count = mmap_vec.len() as u64;
                build_bkd(mmap_vec.as_mut_slice(), &seg_dir)?;
                write_manifest(&new_version_dir, &[point_count])?;
            } else {
                write_manifest(&new_version_dir, &[])?;
            }
            write_u64_slice(&new_version_dir.join("deleted.bin"), &all_deletes)?;
        }

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_id)?;

        Ok(())
    }

    pub fn current_version_id(&self) -> u64 {
        self.version.load().version_id
    }

    pub fn cleanup(&self) {
        let current_version_id = self.version.load().version_id;

        let version_ids = match list_version_dirs(&self.base_path) {
            Ok(ids) => ids,
            Err(e) => {
                tracing::error!("Failed to list version directories: {e}");
                return;
            }
        };

        for version_id in version_ids {
            if version_id != current_version_id {
                if let Err(e) = remove_version_dir(&self.base_path, version_id) {
                    tracing::error!("Failed to remove old version {version_id}: {e}");
                }
            }
        }
    }

    pub fn integrity_check(&self) -> IntegrityCheckResult {
        let mut checks = Vec::new();
        let mut failed = false;

        // 1. CURRENT file exists and parses correctly
        let current_path = self.base_path.join("CURRENT");
        let (format_version, version_id) = match read_current(&self.base_path) {
            Ok(Some((fv, vid))) => {
                checks.push(IntegrityCheck {
                    name: "CURRENT file".to_string(),
                    status: CheckStatus::Ok,
                    message: format!("format_version={fv}, version_id={vid}"),
                });
                (fv, vid)
            }
            Ok(None) => {
                checks.push(IntegrityCheck {
                    name: "CURRENT file".to_string(),
                    status: CheckStatus::Fail,
                    message: format!("CURRENT file not found at {}", current_path.display()),
                });
                return IntegrityCheckResult {
                    checks,
                    passed: false,
                };
            }
            Err(e) => {
                checks.push(IntegrityCheck {
                    name: "CURRENT file".to_string(),
                    status: CheckStatus::Fail,
                    message: format!("Failed to parse CURRENT: {e}"),
                });
                return IntegrityCheckResult {
                    checks,
                    passed: false,
                };
            }
        };

        // 2. Format version matches
        if format_version == FORMAT_VERSION || format_version == 1 {
            checks.push(IntegrityCheck {
                name: "Format version".to_string(),
                status: CheckStatus::Ok,
                message: format!("version {format_version} is supported"),
            });
        } else {
            checks.push(IntegrityCheck {
                name: "Format version".to_string(),
                status: CheckStatus::Fail,
                message: format!(
                    "version {format_version} does not match expected {FORMAT_VERSION}"
                ),
            });
            failed = true;
        }

        // 3. Version directory exists
        let ver_dir = version_dir(&self.base_path, version_id);
        if ver_dir.is_dir() {
            checks.push(IntegrityCheck {
                name: "Version directory".to_string(),
                status: CheckStatus::Ok,
                message: format!("{}", ver_dir.display()),
            });
        } else {
            checks.push(IntegrityCheck {
                name: "Version directory".to_string(),
                status: CheckStatus::Fail,
                message: format!("directory not found: {}", ver_dir.display()),
            });
            return IntegrityCheckResult {
                checks,
                passed: false,
            };
        }

        // 4. Check manifest.bin (v2 format)
        let manifest_path = ver_dir.join("manifest.bin");
        if manifest_path.exists() {
            match read_manifest(&ver_dir) {
                Ok(point_counts) => {
                    checks.push(IntegrityCheck {
                        name: "manifest.bin".to_string(),
                        status: CheckStatus::Ok,
                        message: format!(
                            "{} segments, point_counts: {:?}",
                            point_counts.len(),
                            point_counts
                        ),
                    });

                    // Check each segment directory
                    for (i, &_count) in point_counts.iter().enumerate() {
                        let seg_dir = segment_subdir(&ver_dir, i);
                        if !seg_dir.is_dir() {
                            checks.push(IntegrityCheck {
                                name: format!("segment_{i}"),
                                status: CheckStatus::Fail,
                                message: format!("directory not found: {}", seg_dir.display()),
                            });
                            failed = true;
                            continue;
                        }

                        // Check inner.idx and leaves.dat exist
                        let inner_path = seg_dir.join("inner.idx");
                        let leaves_path = seg_dir.join("leaves.dat");
                        let inner_ok = inner_path.exists();
                        let leaves_ok = leaves_path.exists();

                        if inner_ok && leaves_ok {
                            checks.push(IntegrityCheck {
                                name: format!("segment_{i}"),
                                status: CheckStatus::Ok,
                                message: "inner.idx and leaves.dat present".to_string(),
                            });
                        } else {
                            let mut missing = Vec::new();
                            if !inner_ok {
                                missing.push("inner.idx");
                            }
                            if !leaves_ok {
                                missing.push("leaves.dat");
                            }
                            checks.push(IntegrityCheck {
                                name: format!("segment_{i}"),
                                status: CheckStatus::Fail,
                                message: format!("missing: {}", missing.join(", ")),
                            });
                            failed = true;
                        }
                    }
                }
                Err(e) => {
                    checks.push(IntegrityCheck {
                        name: "manifest.bin".to_string(),
                        status: CheckStatus::Fail,
                        message: format!("Failed to parse: {e}"),
                    });
                    failed = true;
                }
            }
        } else {
            // Legacy v1: check inner.idx, leaves.dat directly
            checks.push(IntegrityCheck {
                name: "manifest.bin".to_string(),
                status: CheckStatus::Skip,
                message: "not present (legacy v1 format)".to_string(),
            });

            let inner_path = ver_dir.join("inner.idx");
            match std::fs::metadata(&inner_path) {
                Ok(m) => {
                    let size = m.len();
                    if size == 0 {
                        checks.push(IntegrityCheck {
                            name: "inner.idx".to_string(),
                            status: CheckStatus::Ok,
                            message: "empty (no compacted data)".to_string(),
                        });
                    } else if !size.is_multiple_of(8) {
                        checks.push(IntegrityCheck {
                            name: "inner.idx".to_string(),
                            status: CheckStatus::Fail,
                            message: format!("size {size} is not a multiple of 8"),
                        });
                        failed = true;
                    } else {
                        checks.push(IntegrityCheck {
                            name: "inner.idx".to_string(),
                            status: CheckStatus::Ok,
                            message: format!("{size} bytes"),
                        });
                    }
                }
                Err(e) => {
                    checks.push(IntegrityCheck {
                        name: "inner.idx".to_string(),
                        status: CheckStatus::Fail,
                        message: format!("not found: {e}"),
                    });
                    failed = true;
                }
            }

            let leaves_path = ver_dir.join("leaves.dat");
            match std::fs::metadata(&leaves_path) {
                Ok(m) => {
                    checks.push(IntegrityCheck {
                        name: "leaves.dat".to_string(),
                        status: CheckStatus::Ok,
                        message: format!("{} bytes", m.len()),
                    });
                }
                Err(e) => {
                    checks.push(IntegrityCheck {
                        name: "leaves.dat".to_string(),
                        status: CheckStatus::Fail,
                        message: format!("not found: {e}"),
                    });
                    failed = true;
                }
            }
        }

        // Check deleted.bin in version directory
        let deleted_path = ver_dir.join("deleted.bin");
        match std::fs::metadata(&deleted_path) {
            Ok(m) => {
                let size = m.len();
                if !size.is_multiple_of(8) {
                    checks.push(IntegrityCheck {
                        name: "deleted.bin".to_string(),
                        status: CheckStatus::Fail,
                        message: format!("size {size} is not a multiple of 8"),
                    });
                    failed = true;
                } else {
                    let count = size / 8;
                    checks.push(IntegrityCheck {
                        name: "deleted.bin".to_string(),
                        status: CheckStatus::Ok,
                        message: format!("{size} bytes, {count} deleted doc_ids"),
                    });
                }
            }
            Err(e) => {
                checks.push(IntegrityCheck {
                    name: "deleted.bin".to_string(),
                    status: CheckStatus::Fail,
                    message: format!("not found: {e}"),
                });
                failed = true;
            }
        }

        IntegrityCheckResult {
            checks,
            passed: !failed,
        }
    }

    pub fn info(&self) -> Result<IndexInfo> {
        let (format_version, version_id) = read_current(&self.base_path)?
            .ok_or_else(|| anyhow!("No CURRENT file found; index has no compacted data"))?;

        let ver_dir = version_dir(&self.base_path, version_id);

        let version = self.version.load();
        let total_points = version.total_point_count() as usize;
        let segment_count = version.segment_count();

        let deleted_size = std::fs::metadata(ver_dir.join("deleted.bin"))
            .map(|m| m.len())
            .unwrap_or(0);

        let deleted_count = if deleted_size.is_multiple_of(8) {
            (deleted_size / 8) as usize
        } else {
            0
        };

        let live = self.live.read().unwrap();
        let pending_inserts = live
            .ops
            .iter()
            .filter(|op| matches!(op, LiveOp::Insert(..)))
            .count();
        let pending_deletes = live
            .ops
            .iter()
            .filter(|op| matches!(op, LiveOp::Delete(..)))
            .count();

        Ok(IndexInfo {
            format_version,
            current_version_id: version_id,
            version_dir: ver_dir,
            segment_count,
            deleted_count,
            deleted_size_bytes: deleted_size,
            total_points,
            pending_inserts,
            pending_deletes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::indexer::IndexedValue;
    use super::super::point::GeoPoint;
    use super::*;
    use tempfile::TempDir;

    fn make_index() -> (TempDir, GeoPointStorage) {
        let tmp = TempDir::new().unwrap();
        let index =
            GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 10).unwrap();
        (tmp, index)
    }

    #[test]
    fn test_new_empty_index() {
        let (_tmp, index) = make_index();
        assert_eq!(index.current_version_id(), 0);
    }

    #[test]
    fn test_insert_and_filter_bbox() {
        let (_tmp, index) = make_index();

        let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
        let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
        let london = GeoPoint::new(51.5074, -0.1278).unwrap();

        index.insert(IndexedValue::Plain(rome), 1);
        index.insert(IndexedValue::Plain(paris), 2);
        index.insert(IndexedValue::Plain(london), 3);

        let op = GeoFilterOp::BoundingBox {
            min_lat: 40.0,
            max_lat: 55.0,
            min_lon: -5.0,
            max_lon: 15.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_delete_and_filter() {
        let (_tmp, index) = make_index();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(15.0, 25.0).unwrap();

        index.insert(IndexedValue::Plain(p1), 1);
        index.insert(IndexedValue::Plain(p2), 2);
        index.delete(1);

        let op = GeoFilterOp::BoundingBox {
            min_lat: 0.0,
            max_lat: 30.0,
            min_lon: 0.0,
            max_lon: 30.0,
        };
        let results: Vec<u64> = index.filter(op).iter().collect();
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn test_compact_basic() {
        let (_tmp, index) = make_index();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap();

        index.insert(IndexedValue::Plain(p1), 1);
        index.insert(IndexedValue::Plain(p2), 2);

        index.compact(1).unwrap();

        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2]);
        assert_eq!(index.current_version_id(), 1);
    }

    #[test]
    fn test_compact_with_deletes() {
        let (_tmp, index) = make_index();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        let p2 = GeoPoint::new(30.0, 40.0).unwrap();
        let p3 = GeoPoint::new(50.0, 60.0).unwrap();

        index.insert(IndexedValue::Plain(p1), 1);
        index.insert(IndexedValue::Plain(p2), 2);
        index.insert(IndexedValue::Plain(p3), 3);
        index.delete(2);

        index.compact(1).unwrap();

        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = GeoPointStorage::new(base_path.clone(), Threshold::default(), 10).unwrap();
            let p1 = GeoPoint::new(10.0, 20.0).unwrap();
            let p2 = GeoPoint::new(30.0, 40.0).unwrap();
            index.insert(IndexedValue::Plain(p1), 1);
            index.insert(IndexedValue::Plain(p2), 2);
            index.compact(1).unwrap();
        }

        {
            let index = GeoPointStorage::new(base_path, Threshold::default(), 10).unwrap();
            let op = GeoFilterOp::BoundingBox {
                min_lat: -90.0,
                max_lat: 90.0,
                min_lon: -180.0,
                max_lon: 180.0,
            };
            let mut results: Vec<u64> = index.filter(op).iter().collect();
            results.sort_unstable();
            assert_eq!(results, vec![1, 2]);
            assert_eq!(index.current_version_id(), 1);
        }
    }

    #[test]
    fn test_ops_during_compaction_preserved() {
        let (_tmp, index) = make_index();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        index.insert(IndexedValue::Plain(p1), 1);
        index.compact(1).unwrap();

        let p2 = GeoPoint::new(30.0, 40.0).unwrap();
        index.insert(IndexedValue::Plain(p2), 2);

        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn test_cleanup_removes_old_versions() {
        let (_tmp, index) = make_index();

        let p = GeoPoint::new(10.0, 20.0).unwrap();

        index.insert(IndexedValue::Plain(p), 1);
        index.compact(1).unwrap();

        index.insert(IndexedValue::Plain(p), 2);
        index.compact(2).unwrap();

        index.insert(IndexedValue::Plain(p), 3);
        index.compact(3).unwrap();

        assert!(_tmp.path().join("versions/1").exists());
        assert!(_tmp.path().join("versions/2").exists());
        assert!(_tmp.path().join("versions/3").exists());

        index.cleanup();

        assert!(!_tmp.path().join("versions/1").exists());
        assert!(!_tmp.path().join("versions/2").exists());
        assert!(_tmp.path().join("versions/3").exists());
    }

    #[test]
    fn test_multiple_compactions() {
        let (_tmp, index) = make_index();

        let p1 = GeoPoint::new(10.0, 20.0).unwrap();
        index.insert(IndexedValue::Plain(p1), 1);
        index.compact(1).unwrap();

        let p2 = GeoPoint::new(30.0, 40.0).unwrap();
        index.insert(IndexedValue::Plain(p2), 2);
        index.compact(2).unwrap();

        let p3 = GeoPoint::new(50.0, 60.0).unwrap();
        index.insert(IndexedValue::Plain(p3), 3);
        index.compact(3).unwrap();

        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_radius_query() {
        let (_tmp, index) = make_index();

        let rome = GeoPoint::new(41.9028, 12.4964).unwrap();
        let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
        let london = GeoPoint::new(51.5074, -0.1278).unwrap();

        index.insert(IndexedValue::Plain(rome), 1);
        index.insert(IndexedValue::Plain(paris), 2);
        index.insert(IndexedValue::Plain(london), 3);

        // 500 km from Rome - should only get Rome
        let op = GeoFilterOp::Radius {
            center: rome,
            radius_meters: 500_000.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1]);

        // 2000 km from Rome - should get all three
        let op = GeoFilterOp::Radius {
            center: rome,
            radius_meters: 2_000_000.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_incompatible_format_version() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = GeoPointStorage::new(base_path.clone(), Threshold::default(), 10).unwrap();
            index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
            index.compact(1).unwrap();
        }

        let current_path = base_path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        let result = GeoPointStorage::new(base_path, Threshold::default(), 10);
        match result {
            Ok(_) => panic!("Expected error for incompatible format version"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("Unsupported format version"),
                    "Unexpected error: {err_msg}"
                );
            }
        }
    }

    #[test]
    fn test_multi_segment_hot_compact() {
        let (_tmp, index) = make_index();

        // 5 sequential compacts with max_segments=10 -> 5 segments
        for round in 0u64..5 {
            let lat = 10.0 + round as f64 * 10.0;
            index.insert(
                IndexedValue::Plain(GeoPoint::new(lat, 20.0).unwrap()),
                round,
            );
            index.compact(round + 1).unwrap();
        }

        // Verify all 5 points accessible
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);

        // Verify version dir has segment subdirs
        let ver_dir = version_dir(_tmp.path(), 5);
        assert!(ver_dir.join("manifest.bin").exists());
        // Should have 5 segment dirs
        let manifest = read_manifest(&ver_dir).unwrap();
        assert_eq!(manifest.len(), 5);
    }

    #[test]
    fn test_partial_merge_on_max_segments() {
        let tmp = TempDir::new().unwrap();
        let index =
            GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 2).unwrap();

        // First compact: full (segment_count=0)
        index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
        index.compact(1).unwrap();

        // Second compact: hot (segment_count=1 < max_segments=2)
        index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
        index.compact(2).unwrap();

        // Third compact: partial merge (segment_count=2 >= max_segments=2)
        index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
        index.compact(3).unwrap();

        // After partial merge, segment count should be reduced but not necessarily to 1
        let ver_dir = version_dir(tmp.path(), 3);
        let manifest = read_manifest(&ver_dir).unwrap();
        assert!(
            manifest.len() < 3,
            "segment count should decrease after partial merge"
        );
        assert!(!manifest.is_empty(), "should have at least 1 segment");

        // All data present
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_delete_ratio_triggers_full_compact() {
        let tmp = TempDir::new().unwrap();
        // threshold=0.0 means any deletes trigger full compact
        let threshold: Threshold = 0.0f64.try_into().unwrap();
        let index = GeoPointStorage::new(tmp.path().to_path_buf(), threshold, 10).unwrap();

        index.insert(IndexedValue::Plain(GeoPoint::new(10.0, 20.0).unwrap()), 1);
        index.insert(IndexedValue::Plain(GeoPoint::new(30.0, 40.0).unwrap()), 2);
        index.compact(1).unwrap();

        // Delete one point and compact - with threshold=0.0, any deletes trigger full compact
        index.delete(1);
        index.insert(IndexedValue::Plain(GeoPoint::new(50.0, 60.0).unwrap()), 3);
        index.compact(2).unwrap();

        // After full compact with deletes applied, doc 1 should be gone
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![2, 3]);

        // Should be single segment (full compact)
        let ver_dir = version_dir(tmp.path(), 2);
        let manifest = read_manifest(&ver_dir).unwrap();
        assert_eq!(manifest.len(), 1);
    }

    #[test]
    fn test_select_segments_to_merge() {
        // 4 segments: picks ceil(4/2)=2, the two smallest
        let counts = vec![100, 50, 200, 30];
        let indices = select_segments_to_merge(&counts);
        assert_eq!(indices.len(), 2);
        // Smallest are index 3 (30) and index 1 (50)
        assert_eq!(indices, vec![1, 3]);

        // 3 segments: picks ceil(3/2)=2, the two smallest
        let counts = vec![100, 50, 200];
        let indices = select_segments_to_merge(&counts);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices, vec![0, 1]);

        // 2 segments: picks min(ceil(2/2), 2)=2 (all of them)
        let counts = vec![100, 50];
        let indices = select_segments_to_merge(&counts);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices, vec![0, 1]);

        // 5 segments: picks ceil(5/2)=3, the three smallest
        let counts = vec![500, 100, 200, 50, 300];
        let indices = select_segments_to_merge(&counts);
        assert_eq!(indices.len(), 3);
        // Smallest: 50 (idx 3), 100 (idx 1), 200 (idx 2)
        assert_eq!(indices, vec![1, 2, 3]);

        // 1 segment: returns it
        let counts = vec![100];
        let indices = select_segments_to_merge(&counts);
        assert_eq!(indices, vec![0]);

        // 0 segments: returns empty
        let counts: Vec<u64> = vec![];
        let indices = select_segments_to_merge(&counts);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_partial_merge_preserves_deletes() {
        let tmp = TempDir::new().unwrap();
        let index =
            GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), 3).unwrap();

        // Build up 3 segments via hot compacts
        for round in 0u64..3 {
            let lat = 10.0 + round as f64 * 10.0;
            index.insert(
                IndexedValue::Plain(GeoPoint::new(lat, 20.0).unwrap()),
                round,
            );
            index.compact(round + 1).unwrap();
        }

        // Delete doc 1, then trigger partial merge (segment_count=3 >= max_segments=3)
        index.delete(1);
        index.insert(IndexedValue::Plain(GeoPoint::new(70.0, 20.0).unwrap()), 10);
        index.compact(4).unwrap();

        // Doc 1 should be invisible
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert!(!results.contains(&1), "deleted doc should be invisible");
        assert!(results.contains(&0));
        assert!(results.contains(&2));
        assert!(results.contains(&10));
    }

    #[test]
    fn test_repeated_partial_merges() {
        let tmp = TempDir::new().unwrap();
        let max_segments = 4;
        let index =
            GeoPointStorage::new(tmp.path().to_path_buf(), Threshold::default(), max_segments)
                .unwrap();

        let mut all_doc_ids: Vec<u64> = Vec::new();

        for round in 0u64..20 {
            let lat = -80.0 + (round as f64) * 7.0;
            let lon = -170.0 + (round as f64) * 15.0;
            let doc_id = round + 1;
            index.insert(
                IndexedValue::Plain(GeoPoint::new(lat, lon).unwrap()),
                doc_id,
            );
            all_doc_ids.push(doc_id);

            index.compact(round + 1).unwrap();

            // Verify segment count stays bounded
            let ver_dir = version_dir(tmp.path(), round + 1);
            if ver_dir.join("manifest.bin").exists() {
                let manifest = read_manifest(&ver_dir).unwrap();
                assert!(
                    manifest.len() <= max_segments + 1,
                    "segment count {} exceeded max_segments {} + 1 at round {}",
                    manifest.len(),
                    max_segments,
                    round
                );
            }
        }

        // Verify all data is accessible
        let op = GeoFilterOp::BoundingBox {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        let mut results: Vec<u64> = index.filter(op).iter().collect();
        results.sort_unstable();
        assert_eq!(results, all_doc_ids);
    }
}
