use super::compacted::{
    build_segment_data, CompactedTermIterator, DocLengthIterator, Segment, SegmentList,
};
use super::config::SegmentConfig;
use super::indexer::IndexedValue;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    ensure_segment_dir, ensure_version_dir, list_segment_dirs, list_version_dirs,
    read_current, read_manifest, remove_segment_dir, remove_version_dir, segment_data_dir,
    sync_dir, version_dir, write_current_atomic, write_manifest, ManifestEntry, FORMAT_VERSION,
    FORMAT_VERSION_V1,
};
use super::iterator::{ContributionsResult, SearchHandle, SearchParams};
use super::live::{LiveLayer, LiveSnapshot};
use super::merge::sorted_merge;
use super::BM25u64Scorer;
use super::{DocumentFilter, NoFilter};
use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// Persistent full-text string index with BM25 scoring, concurrent reads and writes,
/// and multi-segment compaction to disk.
pub struct StringStorage {
    base_path: PathBuf,
    segments: ArcSwap<SegmentList>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    segment_config: SegmentConfig,
}

impl StringStorage {
    /// Open or create a string index at the given path.
    pub fn new(path: impl Into<PathBuf>, segment_config: SegmentConfig) -> Result<Self> {
        let base_path = path.into();
        fs::create_dir_all(&base_path)
            .with_context(|| format!("Failed to create base directory: {base_path:?}"))?;

        let segments = match read_current(&base_path)? {
            Some((format_version, version_number)) => {
                if format_version == FORMAT_VERSION_V1 {
                    // Migrate v1 → v2
                    migrate_v1_to_v2(&base_path, version_number)?;
                    SegmentList::load(&base_path, version_number).with_context(|| {
                        format!("Failed to load after v1→v2 migration at version {version_number}")
                    })?
                } else if format_version == FORMAT_VERSION {
                    SegmentList::load(&base_path, version_number).with_context(|| {
                        format!("Failed to load version at version_number {version_number}")
                    })?
                } else {
                    return Err(anyhow!(
                        "Unsupported format version {format_version}, expected {FORMAT_VERSION}"
                    ));
                }
            }
            None => SegmentList::empty(),
        };

        Ok(Self {
            base_path,
            segments: ArcSwap::new(Arc::new(segments)),
            live: RwLock::new(LiveLayer::new()),
            compaction_lock: Mutex::new(()),
            segment_config,
        })
    }

    /// Insert a document's indexed terms into the index.
    pub fn insert(&self, doc_id: u64, indexed_value: IndexedValue) {
        let mut live = self.live.write().unwrap();
        live.insert(doc_id, indexed_value);
        drop(live);
    }

    /// Mark a doc_id as deleted.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
        drop(live);
    }

    /// Search the index without document filtering.
    pub fn search(&self, params: &SearchParams<'_>, scorer: &mut BM25u64Scorer) -> Result<()> {
        self.search_filtered::<NoFilter>(params, None, scorer)
    }

    /// Search the index, filtering results by document ID.
    pub fn search_with_filter<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: &F,
        scorer: &mut BM25u64Scorer,
    ) -> Result<()> {
        self.search_filtered(params, Some(filter), scorer)
    }

    fn search_filtered<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
        scorer: &mut BM25u64Scorer,
    ) -> Result<()> {
        let snapshot = self.get_fresh_snapshot();
        let segments = self.segments.load();
        let handle = SearchHandle::new(Arc::clone(&segments), snapshot);
        handle.execute(params, filter, scorer)
    }

    /// Collect raw per-token contributions without computing IDF or BM25F saturation.
    pub fn collect_contributions(&self, params: &SearchParams<'_>) -> Result<ContributionsResult> {
        self.collect_contributions_filtered::<NoFilter>(params, None)
    }

    /// Collect raw per-token contributions, filtering results by document ID.
    pub fn collect_contributions_with_filter<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: &F,
    ) -> Result<ContributionsResult> {
        self.collect_contributions_filtered(params, Some(filter))
    }

    fn collect_contributions_filtered<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
    ) -> Result<ContributionsResult> {
        let snapshot = self.get_fresh_snapshot();
        let segments = self.segments.load();
        let handle = SearchHandle::new(Arc::clone(&segments), snapshot);
        handle.execute_collect(params, filter)
    }

    /// Get a fresh snapshot, refreshing if dirty (double-check locking pattern).
    fn get_fresh_snapshot(&self) -> Arc<LiveSnapshot> {
        {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                let snapshot = live.get_snapshot();
                drop(live);
                return snapshot;
            }
        }
        let mut live = self.live.write().unwrap();
        if live.is_snapshot_dirty() {
            live.refresh_snapshot();
        }
        let snapshot = live.get_snapshot();
        drop(live);
        snapshot
    }

    /// Persist pending changes to disk at the given version number using multi-segment compaction.
    pub fn compact(&self, version_number: u64) -> Result<()> {
        let compaction_guard = self.compaction_lock.lock().unwrap();

        let snapshot = self.get_fresh_snapshot();

        // Nothing to compact — free memory and return early
        if snapshot.term_postings.is_empty() && snapshot.deletes.is_empty() {
            let mut live = self.live.write().unwrap();
            live.drain_compacted_ops(snapshot.ops_len);
            if live.ops.is_empty() {
                live.ops.shrink_to_fit();
            }
            live.refresh_snapshot();
            drop(live);
            return Ok(());
        }

        let current = self.segments.load();

        if version_number == current.version_number && current.has_data() {
            return Err(anyhow!(
                "Cannot compact to version {version_number}: same as current active version. \
                 Use a different version number to avoid corrupting active mmaps."
            ));
        }

        let new_version_dir = ensure_version_dir(&self.base_path, version_number)?;

        // Compute merged global deletes
        let merged_deletes: Vec<u64> = sorted_merge(
            current.deletes_slice().iter().copied(),
            snapshot.deletes.iter().copied(),
        )
        .collect();

        let has_live_data = !snapshot.term_postings.is_empty();

        // Determine next segment_id
        let next_seg_id = current
            .segments
            .iter()
            .map(|s| s.segment_id + 1)
            .max()
            .unwrap_or(0);

        let mut new_manifest: Vec<ManifestEntry> = Vec::new();
        let mut live_absorbed = false;
        let threshold_value = self.segment_config.deletion_threshold.value();
        let max_postings = self.segment_config.max_postings_per_segment;
        let mut seg_id_counter = next_seg_id;

        for (seg_idx, segment) in current.segments.iter().enumerate() {
            let is_last = seg_idx == current.segments.len() - 1;

            // Count deletes in this segment's doc_id range
            let segment_deletes = count_deletes_in_range(
                &merged_deletes,
                segment.min_doc_id,
                segment.max_doc_id,
            );
            let deletion_ratio = if segment.num_postings > 0 {
                segment_deletes as f64 / segment.num_postings as f64
            } else {
                0.0
            };

            if is_last && has_live_data {
                // Estimate combined postings
                let live_postings: usize =
                    snapshot.term_postings.values().map(|v| v.len()).sum();
                let combined_postings = segment.num_postings + live_postings;

                if combined_postings <= max_postings {
                    // REBUILD with merge: old segment data + live data
                    let new_seg_id = seg_id_counter;
                    seg_id_counter += 1;
                    let seg_dir = ensure_segment_dir(&self.base_path, new_seg_id)?;

                    let mut compacted_terms = segment.iter_terms();
                    let live_terms: Vec<_> = snapshot.iter_terms_sorted().collect();
                    let mut compacted_dl = segment.iter_doc_lengths();
                    let live_dl = snapshot.iter_doc_lengths_sorted();

                    let result = build_segment_data(
                        &mut compacted_terms,
                        &live_terms,
                        &mut compacted_dl,
                        &live_dl,
                        Some(&merged_deletes),
                        &seg_dir,
                    )?;

                    if result.total_documents > 0 {
                        new_manifest.push(ManifestEntry {
                            segment_id: new_seg_id,
                            num_postings: result.num_postings,
                            num_deletes: 0, // Deletes applied
                            min_doc_id: result.min_doc_id,
                            max_doc_id: result.max_doc_id,
                            total_doc_length: result.total_doc_length,
                            total_documents: result.total_documents,
                        });
                    }
                    live_absorbed = true;
                } else {
                    // Last segment exceeds max_postings when combined
                    // Rebuild or carry forward last segment
                    if deletion_ratio > threshold_value {
                        let entry =
                            rebuild_segment(segment, &merged_deletes, &self.base_path, seg_id_counter)?;
                        seg_id_counter += 1;
                        if let Some(e) = entry {
                            new_manifest.push(e);
                        }
                    } else {
                        // Carry forward
                        new_manifest.push(ManifestEntry {
                            segment_id: segment.segment_id,
                            num_postings: segment.num_postings,
                            num_deletes: segment_deletes,
                            min_doc_id: segment.min_doc_id,
                            max_doc_id: segment.max_doc_id,
                            total_doc_length: segment.num_postings as u64, // will be recomputed
                            total_documents: segment.num_postings as u64, // will be recomputed
                        });
                        // Fix: read actual stats from the segment's doc_lengths
                        let last = new_manifest.last_mut().unwrap();
                        let (td, tl) = scan_segment_stats(segment, &merged_deletes);
                        last.total_documents = td;
                        last.total_doc_length = tl;
                    }
                    // Create NEW segment from live data alone
                    let new_seg_id = seg_id_counter;
                    seg_id_counter += 1;
                    let seg_dir = ensure_segment_dir(&self.base_path, new_seg_id)?;

                    let mut empty_terms = CompactedTermIterator::empty();
                    let live_terms: Vec<_> = snapshot.iter_terms_sorted().collect();
                    let mut empty_dl = DocLengthIterator::empty();
                    let live_dl = snapshot.iter_doc_lengths_sorted();

                    let result = build_segment_data(
                        &mut empty_terms,
                        &live_terms,
                        &mut empty_dl,
                        &live_dl,
                        Some(&merged_deletes),
                        &seg_dir,
                    )?;

                    if result.total_documents > 0 {
                        new_manifest.push(ManifestEntry {
                            segment_id: new_seg_id,
                            num_postings: result.num_postings,
                            num_deletes: 0,
                            min_doc_id: result.min_doc_id,
                            max_doc_id: result.max_doc_id,
                            total_doc_length: result.total_doc_length,
                            total_documents: result.total_documents,
                        });
                    }
                    live_absorbed = true;
                }
            } else if deletion_ratio > threshold_value {
                // REBUILD segment filtering deletes
                let entry =
                    rebuild_segment(segment, &merged_deletes, &self.base_path, seg_id_counter)?;
                seg_id_counter += 1;
                if let Some(e) = entry {
                    new_manifest.push(e);
                }
            } else {
                // CARRY FORWARD
                let (td, tl) = scan_segment_stats(segment, &merged_deletes);
                new_manifest.push(ManifestEntry {
                    segment_id: segment.segment_id,
                    num_postings: segment.num_postings,
                    num_deletes: segment_deletes,
                    min_doc_id: segment.min_doc_id,
                    max_doc_id: segment.max_doc_id,
                    total_doc_length: tl,
                    total_documents: td,
                });
            }
        }

        // If live not absorbed: create new segment from live data only
        if !live_absorbed && has_live_data {
            let new_seg_id = seg_id_counter;
            let seg_dir = ensure_segment_dir(&self.base_path, new_seg_id)?;

            let mut empty_terms = CompactedTermIterator::empty();
            let live_terms: Vec<_> = snapshot.iter_terms_sorted().collect();
            let mut empty_dl = DocLengthIterator::empty();
            let live_dl = snapshot.iter_doc_lengths_sorted();

            let result = build_segment_data(
                &mut empty_terms,
                &live_terms,
                &mut empty_dl,
                &live_dl,
                Some(&merged_deletes),
                &seg_dir,
            )?;

            if result.total_documents > 0 {
                new_manifest.push(ManifestEntry {
                    segment_id: new_seg_id,
                    num_postings: result.num_postings,
                    num_deletes: 0,
                    min_doc_id: result.min_doc_id,
                    max_doc_id: result.max_doc_id,
                    total_doc_length: result.total_doc_length,
                    total_documents: result.total_documents,
                });
            }
        }

        // Compute global stats from manifest
        let global_total_documents: u64 = new_manifest.iter().map(|e| e.total_documents).sum();
        let global_total_doc_length: u64 = new_manifest.iter().map(|e| e.total_doc_length).sum();

        // Write manifest, deleted.bin, global_info.bin
        write_manifest(&new_version_dir, &new_manifest)?;

        // If all segments have been rebuilt (no carry-forward), deletes are fully applied
        let any_carried_forward = new_manifest.iter().any(|e| e.num_deletes > 0);
        let deletes_to_write = if any_carried_forward {
            &merged_deletes
        } else {
            &[] as &[u64]
        };
        super::io::write_deleted(
            &new_version_dir.join("deleted.bin"),
            deletes_to_write,
        )?;
        super::io::write_global_info(
            &new_version_dir.join("global_info.bin"),
            global_total_doc_length,
            global_total_documents,
        )?;

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_number)?;

        // Atomic update: swap segments AND clear compacted items
        let new_segments = SegmentList::load(&self.base_path, version_number)?;
        {
            let mut live = self.live.write().unwrap();
            self.segments.store(Arc::new(new_segments));

            live.drain_compacted_ops(snapshot.ops_len);
            live.refresh_snapshot();
            drop(live);
        }

        drop(compaction_guard);
        Ok(())
    }

    /// Return the current compacted version number.
    pub fn current_version_number(&self) -> u64 {
        self.segments.load().version_number
    }

    /// Delete old version directories and unreferenced segment directories.
    pub fn cleanup(&self) {
        let compaction_guard = self.compaction_lock.lock().unwrap();
        let current = self.segments.load();
        let current_version = current.version_number;

        // Remove old version directories
        let version_numbers = match list_version_dirs(&self.base_path) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to list version directories: {e}");
                return;
            }
        };

        for version_number in version_numbers {
            if version_number != current_version {
                if let Err(e) = remove_version_dir(&self.base_path, version_number) {
                    tracing::error!("Failed to remove old version {version_number}: {e}");
                }
            }
        }

        // Remove unreferenced segment directories
        let referenced_ids: std::collections::HashSet<u64> =
            current.segments.iter().map(|s| s.segment_id).collect();

        let all_seg_ids = match list_segment_dirs(&self.base_path) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to list segment directories: {e}");
                return;
            }
        };

        for seg_id in all_seg_ids {
            if !referenced_ids.contains(&seg_id) {
                if let Err(e) = remove_segment_dir(&self.base_path, seg_id) {
                    tracing::error!("Failed to remove unreferenced segment {seg_id}: {e}");
                }
            }
        }

        drop(compaction_guard);
    }

    /// Return metadata and statistics about the index.
    pub fn info(&self) -> IndexInfo {
        let current = self.segments.load();
        let live = self.live.read().unwrap();
        let pending_ops = live.ops.len();
        drop(live);

        let ver_dir = version_dir(&self.base_path, current.version_number);

        let avg_field_length = if current.total_documents > 0 {
            current.total_document_length as f64 / current.total_documents as f64
        } else {
            0.0
        };

        let mut unique_terms_count = 0;
        let mut total_postings_count = 0;
        let mut total_segments_size_bytes: u64 = 0;

        for segment in &current.segments {
            unique_terms_count += segment.term_count();
            total_postings_count += segment.num_postings;

            let seg_dir = segment_data_dir(&self.base_path, segment.segment_id);
            total_segments_size_bytes += file_size(&seg_dir.join("keys.fst"));
            total_segments_size_bytes += file_size(&seg_dir.join("postings.dat"));
            total_segments_size_bytes += file_size(&seg_dir.join("doc_lengths.dat"));
        }

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: current.version_number,
            version_dir: ver_dir.clone(),
            num_segments: current.segments.len(),
            unique_terms_count,
            total_postings_count,
            total_documents: current.total_documents,
            avg_field_length,
            deleted_count: current.deletes_slice().len(),
            total_segments_size_bytes,
            deleted_size_bytes: file_size(&ver_dir.join("deleted.bin")),
            global_info_size_bytes: file_size(&ver_dir.join("global_info.bin")),
            pending_ops,
        }
    }

    /// Verify that the on-disk index files are valid and consistent.
    pub fn integrity_check(&self) -> IntegrityCheckResult {
        let mut checks = Vec::new();

        let current_path = self.base_path.join("CURRENT");
        if !current_path.exists() {
            checks.push(IntegrityCheck::failed(
                "CURRENT",
                Some("File does not exist".to_string()),
            ));
            return IntegrityCheckResult::new(checks);
        }

        match read_current(&self.base_path) {
            Ok(Some((format_version, version_number))) => {
                checks.push(IntegrityCheck::ok(
                    "CURRENT",
                    Some(format!(
                        "version: {format_version}, version_number: {version_number}"
                    )),
                ));

                if format_version != FORMAT_VERSION {
                    checks.push(IntegrityCheck::failed(
                        "format version",
                        Some(format!("Expected {FORMAT_VERSION}, found {format_version}")),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "format version",
                    Some(format!("{FORMAT_VERSION}")),
                ));

                let ver_dir = version_dir(&self.base_path, version_number);
                if !ver_dir.exists() {
                    checks.push(IntegrityCheck::failed(
                        "version directory",
                        Some(format!("Does not exist: {}", ver_dir.display())),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                if !ver_dir.is_dir() {
                    checks.push(IntegrityCheck::failed(
                        "version directory",
                        Some(format!("Not a directory: {}", ver_dir.display())),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "version directory",
                    Some(ver_dir.display().to_string()),
                ));

                // Check manifest
                let manifest_path = ver_dir.join("manifest.json");
                if !manifest_path.exists() {
                    checks.push(IntegrityCheck::failed(
                        "manifest.json",
                        Some("File does not exist".to_string()),
                    ));
                    return IntegrityCheckResult::new(checks);
                }

                match read_manifest(&ver_dir) {
                    Ok(manifest) => {
                        checks.push(IntegrityCheck::ok(
                            "manifest.json",
                            Some(format!("{} segments", manifest.len())),
                        ));

                        // Check each segment directory
                        for entry in &manifest {
                            let seg_dir =
                                segment_data_dir(&self.base_path, entry.segment_id);
                            if !seg_dir.exists() {
                                checks.push(IntegrityCheck::failed(
                                    format!("segment {}", entry.segment_id),
                                    Some(format!("Directory does not exist: {}", seg_dir.display())),
                                ));
                                continue;
                            }

                            let required_files = ["keys.fst", "postings.dat", "doc_lengths.dat"];
                            let missing: Vec<&str> = required_files
                                .iter()
                                .filter(|f| !seg_dir.join(f).exists())
                                .copied()
                                .collect();

                            if !missing.is_empty() {
                                checks.push(IntegrityCheck::failed(
                                    format!("segment {}", entry.segment_id),
                                    Some(format!("Missing files: {}", missing.join(", "))),
                                ));
                            } else {
                                checks.push(IntegrityCheck::ok(
                                    format!("segment {}", entry.segment_id),
                                    Some("All files present".to_string()),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        checks.push(IntegrityCheck::failed(
                            "manifest.json",
                            Some(format!("Failed to read: {e}")),
                        ));
                    }
                }

                // Check version-level files
                let required_version_files = ["deleted.bin", "global_info.bin"];
                let missing_version: Vec<&str> = required_version_files
                    .iter()
                    .filter(|f| !ver_dir.join(f).exists())
                    .copied()
                    .collect();
                if !missing_version.is_empty() {
                    checks.push(IntegrityCheck::failed(
                        "version files",
                        Some(format!("Missing: {}", missing_version.join(", "))),
                    ));
                } else {
                    checks.push(IntegrityCheck::ok(
                        "version files",
                        Some("deleted.bin, global_info.bin present".to_string()),
                    ));
                }

                // Validate deleted.bin
                match validate_deleted_file(&ver_dir.join("deleted.bin")) {
                    Ok(()) => {
                        checks.push(IntegrityCheck::ok(
                            "deleted.bin valid",
                            Some("Size and sorting OK".to_string()),
                        ));
                    }
                    Err(e) => {
                        checks.push(IntegrityCheck::failed(
                            "deleted.bin valid",
                            Some(format!("{e}")),
                        ));
                    }
                }

                // Validate global_info.bin
                match super::io::read_global_info(&ver_dir.join("global_info.bin")) {
                    Ok((total_len, total_docs)) => {
                        checks.push(IntegrityCheck::ok(
                            "global_info.bin",
                            Some(format!(
                                "total_document_length: {total_len}, total_documents: {total_docs}"
                            )),
                        ));
                    }
                    Err(e) => {
                        checks.push(IntegrityCheck::failed(
                            "global_info.bin",
                            Some(format!("{e}")),
                        ));
                    }
                }
            }
            Ok(None) => {
                checks.push(IntegrityCheck::failed(
                    "CURRENT",
                    Some("File is empty or invalid".to_string()),
                ));
            }
            Err(e) => {
                checks.push(IntegrityCheck::failed(
                    "CURRENT",
                    Some(format!("Failed to read: {e}")),
                ));
            }
        }

        IntegrityCheckResult::new(checks)
    }
}

/// Rebuild a segment by re-writing its data, filtering out deleted doc_ids.
fn rebuild_segment(
    segment: &Segment,
    deleted_set: &[u64],
    base_path: &std::path::Path,
    new_seg_id: u64,
) -> Result<Option<ManifestEntry>> {
    let seg_dir = ensure_segment_dir(base_path, new_seg_id)?;

    let mut compacted_terms = segment.iter_terms();
    let mut compacted_dl = segment.iter_doc_lengths();

    let empty_live: &[(&str, &[(u64, Vec<u32>, Vec<u32>)])] = &[];
    let result = build_segment_data(
        &mut compacted_terms,
        empty_live,
        &mut compacted_dl,
        &[],
        Some(deleted_set),
        &seg_dir,
    )?;

    if result.total_documents == 0 {
        // Segment is now empty after applying deletes
        let _ = fs::remove_dir_all(&seg_dir);
        return Ok(None);
    }

    Ok(Some(ManifestEntry {
        segment_id: new_seg_id,
        num_postings: result.num_postings,
        num_deletes: 0,
        min_doc_id: result.min_doc_id,
        max_doc_id: result.max_doc_id,
        total_doc_length: result.total_doc_length,
        total_documents: result.total_documents,
    }))
}

/// Scan segment doc_lengths to compute stats excluding deleted docs.
fn scan_segment_stats(segment: &Segment, deleted_set: &[u64]) -> (u64, u64) {
    let mut total_documents: u64 = 0;
    let mut total_doc_length: u64 = 0;

    let mut del_idx = 0;
    for (doc_id, field_len) in segment.iter_doc_lengths() {
        // Advance delete cursor
        while del_idx < deleted_set.len() && deleted_set[del_idx] < doc_id {
            del_idx += 1;
        }
        let is_deleted = del_idx < deleted_set.len() && deleted_set[del_idx] == doc_id;
        if !is_deleted {
            total_documents += 1;
            total_doc_length += field_len as u64;
        }
    }

    (total_documents, total_doc_length)
}

/// Count how many entries in `deleted_set` fall within [min_doc_id, max_doc_id].
fn count_deletes_in_range(deleted_set: &[u64], min_doc_id: u64, max_doc_id: u64) -> usize {
    if deleted_set.is_empty() {
        return 0;
    }
    let start = deleted_set.partition_point(|&d| d < min_doc_id);
    let end = deleted_set.partition_point(|&d| d <= max_doc_id);
    end - start
}

/// Migrate a v1 format index to v2 multi-segment layout.
fn migrate_v1_to_v2(base_path: &std::path::Path, version_number: u64) -> Result<()> {
    let ver_dir = version_dir(base_path, version_number);
    let seg_dir = ensure_segment_dir(base_path, 0)?;

    // Check if already partially migrated
    let manifest_exists = ver_dir.join("manifest.json").exists();
    let seg_fst_exists = seg_dir.join("keys.fst").exists();

    if manifest_exists && seg_fst_exists {
        // Already migrated (or partially), just need to update CURRENT
        super::io::write_current_atomic_versioned(base_path, FORMAT_VERSION, version_number)?;
        return Ok(());
    }

    // Move the three data files into the segment directory
    let files_to_move = ["keys.fst", "postings.dat", "doc_lengths.dat"];
    for file_name in &files_to_move {
        let src = ver_dir.join(file_name);
        let dst = seg_dir.join(file_name);
        if src.exists() && !dst.exists() {
            // Try rename first (fast, atomic on same filesystem)
            if fs::rename(&src, &dst).is_err() {
                // Fallback: copy + delete
                fs::copy(&src, &dst).with_context(|| {
                    format!("Failed to copy {file_name} during v1→v2 migration")
                })?;
                fs::remove_file(&src).ok();
            }
        }
    }

    // Compute segment metadata
    let global_info_path = ver_dir.join("global_info.bin");
    let (total_doc_length, total_documents) = super::io::read_global_info(&global_info_path)?;

    // Read min/max doc_id from doc_lengths.dat
    let doc_lengths_path = seg_dir.join("doc_lengths.dat");
    let (min_doc_id, max_doc_id) = if doc_lengths_path.exists() {
        let metadata = fs::metadata(&doc_lengths_path)?;
        let size = metadata.len();
        if size >= 12 {
            let bytes = fs::read(&doc_lengths_path)?;
            let min = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
            let last_offset = (size as usize / 12 - 1) * 12;
            let max = u64::from_ne_bytes(bytes[last_offset..last_offset + 8].try_into().unwrap());
            (min, max)
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    // Count postings by scanning FST
    let num_postings = {
        let fst_path = seg_dir.join("keys.fst");
        if fst_path.exists() {
            let entry = ManifestEntry {
                segment_id: 0,
                num_postings: 0,
                num_deletes: 0,
                min_doc_id,
                max_doc_id,
                total_doc_length,
                total_documents,
            };
            let segment = Segment::load(base_path, &entry)?;
            segment.total_postings()
        } else {
            0
        }
    };

    // Count deletes
    let deleted_path = ver_dir.join("deleted.bin");
    let num_deletes = if deleted_path.exists() {
        let metadata = fs::metadata(&deleted_path)?;
        metadata.len() as usize / 8
    } else {
        0
    };

    // Write manifest
    let manifest = vec![ManifestEntry {
        segment_id: 0,
        num_postings,
        num_deletes,
        min_doc_id,
        max_doc_id,
        total_doc_length,
        total_documents,
    }];

    // Only write manifest if there's actual data
    if total_documents > 0 || num_postings > 0 {
        write_manifest(&ver_dir, &manifest)?;
    } else {
        write_manifest(&ver_dir, &[])?;
    }

    // Update CURRENT to format version 2
    super::io::write_current_atomic_versioned(base_path, FORMAT_VERSION, version_number)?;

    // Sync directories
    sync_dir(&ver_dir)?;
    sync_dir(&seg_dir)?;

    Ok(())
}

fn file_size(path: &std::path::Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn validate_deleted_file(path: &std::path::Path) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| "Failed to get metadata for deleted.bin".to_string())?;

    let size = metadata.len();
    if size % 8 != 0 {
        return Err(anyhow!("file size ({size} bytes) is not a multiple of 8"));
    }

    if size == 0 {
        return Ok(());
    }

    let bytes = fs::read(path).with_context(|| "Failed to read deleted.bin")?;

    let values: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();

    for i in 1..values.len() {
        if values[i] <= values[i - 1] {
            return Err(anyhow!(
                "values not strictly sorted at index {i} ({} <= {})",
                values[i],
                values[i - 1]
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::indexer::TermData;
    use super::super::iterator::SearchResult;
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn make_value(field_length: u16, terms: Vec<(&str, Vec<u32>, Vec<u32>)>) -> IndexedValue {
        let mut term_map = HashMap::new();
        for (term, exact, stemmed) in terms {
            term_map.insert(
                term.to_string(),
                TermData {
                    exact_positions: exact,
                    stemmed_positions: stemmed,
                },
            );
        }
        IndexedValue {
            field_length,
            terms: term_map,
        }
    }

    fn search_default(index: &StringStorage, tokens: &[&str]) -> SearchResult {
        let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        let mut scorer = BM25u64Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &owned,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        scorer.into_search_result()
    }

    fn default_config() -> SegmentConfig {
        SegmentConfig::default()
    }

    #[test]
    fn test_new_empty_index() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();
        assert_eq!(index.current_version_number(), 0);
    }

    #[test]
    fn test_insert_and_search() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(3, vec![("hello", vec![0], vec![1, 2])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(3, make_value(4, vec![("world", vec![0, 1], vec![])]));

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));

        let result = search_default(&index, &["world"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 3);

        let result = search_default(&index, &["missing"]);
        assert!(result.docs.is_empty());
    }

    #[test]
    fn test_delete_and_search() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.delete(1);

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);
    }

    #[test]
    fn test_compact_basic() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(3, make_value(4, vec![("world", vec![0], vec![])]));

        index.compact(1).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);

        let result = search_default(&index, &["world"]);
        assert_eq!(result.docs.len(), 1);

        assert_eq!(index.current_version_number(), 1);
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringStorage::new(base_path.clone(), default_config()).unwrap();
            index.insert(1, make_value(3, vec![("hello", vec![0], vec![1])]));
            index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
            index.compact(1).unwrap();
        }

        {
            let index = StringStorage::new(base_path, default_config()).unwrap();
            let result = search_default(&index, &["hello"]);
            assert_eq!(result.docs.len(), 1);
            assert_eq!(result.docs[0].doc_id, 1);
        }
    }

    #[test]
    fn test_compact_with_deletes_carry_forward() {
        let tmp = TempDir::new().unwrap();
        // High threshold → carry forward
        let config = SegmentConfig {
            deletion_threshold: 0.99f64.try_into().unwrap(),
            ..Default::default()
        };
        let index = StringStorage::new(tmp.path().to_path_buf(), config).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.delete(1);
        index.compact(2).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);
    }

    #[test]
    fn test_compact_with_deletes_apply() {
        let tmp = TempDir::new().unwrap();
        // Low threshold → apply deletes
        let config = SegmentConfig {
            deletion_threshold: 0.01f64.try_into().unwrap(),
            ..Default::default()
        };
        let index = StringStorage::new(tmp.path().to_path_buf(), config).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.delete(1);
        index.compact(2).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);
    }

    #[test]
    fn test_empty_compaction_skipped() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        // No new changes, compaction should be a no-op
        index.compact(2).unwrap();
        // Version stays at 1 since nothing was written
        assert_eq!(index.current_version_number(), 1);
    }

    #[test]
    fn test_cleanup() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
        index.compact(2).unwrap();

        index.cleanup();

        // Version 1 dir should be gone
        assert!(!version_dir(tmp.path(), 1).exists());
        // Version 2 dir should still exist
        assert!(version_dir(tmp.path(), 2).exists());
    }

    #[test]
    fn test_info() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(info.format_version, FORMAT_VERSION);
        assert_eq!(info.current_version_number, 1);
        assert_eq!(info.total_documents, 2);
        assert_eq!(info.num_segments, 1);
        assert!(info.total_segments_size_bytes > 0);
    }

    #[test]
    fn test_integrity_check_valid() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), default_config()).unwrap();

        index.insert(1, make_value(3, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        let result = index.integrity_check();
        assert!(result.passed);
    }
}
