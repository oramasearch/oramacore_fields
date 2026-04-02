use crate::embedding::{IndexedValue, NoFilter};

use super::config::DistanceMetric;
use super::config::{EmbeddingConfig, SegmentConfig};
use super::distance::Distance;
use super::error::Error;
use super::hnsw::HnswBuilder;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    ensure_segment_dir, ensure_version_dir, list_segment_dirs, list_version_dirs, read_current,
    remove_version_dir, segment_data_dir, sync_dir, version_dir, write_current_atomic,
    write_delete_file, write_manifest, ManifestEntry, FORMAT_VERSION,
};
use super::live::{LiveLayer, LiveSnapshot};
use super::quantization::{
    compute_min_max, element_wise_max, element_wise_min, range_contains, QuantizationParams,
};
use super::segment::SegmentList;
use arc_swap::ArcSwap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

pub struct EmbeddingStorage {
    base_path: PathBuf,
    segments: ArcSwap<SegmentList>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    config: EmbeddingConfig,
    segment_config: SegmentConfig,
}

impl EmbeddingStorage {
    pub fn new(
        base_path: PathBuf,
        config: EmbeddingConfig,
        segment_config: SegmentConfig,
    ) -> Result<Self, Error> {
        std::fs::create_dir_all(&base_path)?;

        let segment_list = match read_current(&base_path)? {
            Some((format_version, version_number)) => {
                if format_version != FORMAT_VERSION {
                    return Err(Error::FormatVersionMismatch {
                        expected: FORMAT_VERSION,
                        found: format_version,
                    });
                }
                SegmentList::load(&base_path, version_number)?
            }
            None => SegmentList::empty(),
        };

        Ok(Self {
            base_path,
            segments: ArcSwap::new(Arc::new(segment_list)),
            live: RwLock::new(LiveLayer::new()),
            compaction_lock: Mutex::new(()),
            config,
            segment_config,
        })
    }

    /// Insert an embedding for a doc_id.
    pub fn insert(&self, doc_id: u64, indexed_value: IndexedValue) {
        let mut live = self.live.write().unwrap();
        let vectors = match indexed_value {
            IndexedValue::Single(v) => vec![v.embedding],
            IndexedValue::Array(v) => v.into_iter().map(|e| e.embedding).collect(),
        };
        live.insert(doc_id, vectors);
        drop(live);
    }

    /// Delete a doc_id.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
        drop(live);
    }

    /// Get a fresh snapshot (double-check locking).
    fn fresh_snapshot(&self) -> Arc<LiveSnapshot> {
        let live = self.live.read().unwrap();
        if !live.is_snapshot_dirty() {
            return live.get_snapshot();
        }
        drop(live);
        let mut live = self.live.write().unwrap();
        if live.is_snapshot_dirty() {
            live.refresh_snapshot();
        }
        let snapshot = live.get_snapshot();
        drop(live);
        snapshot
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, Error> {
        let mut ctx = super::SearchContext::new();
        self.search_with_filter_and_context::<NoFilter>(query, k, ef_search, None, &mut ctx)?;
        Ok(std::mem::take(&mut ctx.inner.all_results))
    }

    /// Search for k nearest neighbors, filtering results by document ID.
    ///
    /// The filter is applied only to final results — HNSW graph traversal
    /// still visits filtered-out nodes to preserve graph connectivity.
    pub fn search_with_filter<F: super::DocumentFilter>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: &F,
    ) -> Result<Vec<(u64, f32)>, Error> {
        let mut ctx = super::SearchContext::new();
        self.search_with_filter_and_context(query, k, ef_search, Some(filter), &mut ctx)?;
        Ok(std::mem::take(&mut ctx.inner.all_results))
    }

    /// Search for k nearest neighbors, reusing caller-provided buffers.
    pub fn search_with_context<'a>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        ctx: &'a mut super::SearchContext,
    ) -> Result<&'a [(u64, f32)], Error> {
        self.search_with_filter_and_context::<NoFilter>(query, k, ef_search, None, ctx)
    }

    /// Search for k nearest neighbors with filter, reusing caller-provided buffers.
    pub fn search_with_filter_and_context<'a, F: super::DocumentFilter>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&F>,
        ctx: &'a mut super::SearchContext,
    ) -> Result<&'a [(u64, f32)], Error> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        if k == 0 {
            ctx.inner.all_results.clear();
            return Ok(&ctx.inner.all_results);
        }

        match self.config.metric {
            DistanceMetric::L2 => {
                self.search_inner_ctx::<super::distance::L2, F>(
                    query,
                    k,
                    ef_search,
                    filter,
                    &mut ctx.inner,
                )?;
            }
            DistanceMetric::DotProduct => {
                self.search_inner_ctx::<super::distance::DotProduct, F>(
                    query,
                    k,
                    ef_search,
                    filter,
                    &mut ctx.inner,
                )?;
            }
            DistanceMetric::Cosine => {
                let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm == 0.0 {
                    self.search_inner_ctx::<super::distance::Cosine, F>(
                        query,
                        k,
                        ef_search,
                        filter,
                        &mut ctx.inner,
                    )?;
                } else {
                    ctx.normalized.clear();
                    ctx.normalized.extend(query.iter().map(|x| x / norm));
                    self.search_inner_ctx::<super::distance::CosineNorm, F>(
                        &ctx.normalized,
                        k,
                        ef_search,
                        filter,
                        &mut ctx.inner,
                    )?;
                }
            }
        }

        Ok(&ctx.inner.all_results)
    }

    fn search_inner_ctx<D: Distance, F: super::DocumentFilter>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&F>,
        ctx: &mut super::search_context::SearchBuffers,
    ) -> Result<(), Error> {
        let snapshot = self.fresh_snapshot();
        let segment_list = self.segments.load();
        let ef = ef_search.unwrap_or(self.config.ef_search);

        ctx.all_results.clear();
        ctx.query_quantized.resize(self.config.dimensions, 0);

        for segment in &segment_list.segments {
            // Per-segment quantization: quantize query with THIS segment's params
            ctx.query_quantized.fill(0);
            segment
                .quantization_params
                .quantize(query, &mut ctx.query_quantized);

            // Extract live deletes targeting this segment's doc_id range (zero-alloc)
            let start = snapshot
                .deletes
                .partition_point(|&d| d < segment.min_doc_id);
            let end = snapshot
                .deletes
                .partition_point(|&d| d <= segment.max_doc_id);
            let targeted_live_deletes = &snapshot.deletes[start..end];

            segment.search_with_context::<D, F>(
                query,
                &ctx.query_quantized,
                k,
                ef,
                segment.deletes_slice(),
                targeted_live_deletes,
                filter,
                &mut ctx.segment,
            );
            ctx.all_results.extend_from_slice(&ctx.segment.scored);
        }

        // Search live layer (brute force)
        snapshot.search_with_context::<D, F>(query, k, &snapshot.deletes, filter, &mut ctx.live);
        ctx.all_results.extend_from_slice(&ctx.live.live_results);

        // Merge: sort by distance, deduplicate by doc_id, take top-k
        ctx.all_results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ctx.seen.clear();
        ctx.all_results
            .retain(|(doc_id, _)| ctx.seen.insert(*doc_id));
        ctx.all_results.truncate(k);
        Ok(())
    }

    /// Compact the index using multi-segment architecture.
    pub fn compact(&self, version_number: u64) -> Result<(), Error> {
        let compaction_guard = self.compaction_lock.lock().unwrap();

        let snapshot = self.fresh_snapshot();
        let current = self.segments.load();

        // Early return if nothing to do
        if snapshot.is_empty() {
            let mut live = self.live.write().unwrap();
            live.ops.drain(..snapshot.ops_len);
            if live.ops.is_empty() {
                live.ops.shrink_to_fit();
            }
            live.refresh_snapshot();
            drop(live);
            return Ok(());
        }

        match self.config.metric {
            DistanceMetric::L2 => {
                self.compact_inner::<super::distance::L2>(version_number, &snapshot, &current)?;
            }
            DistanceMetric::DotProduct => {
                self.compact_inner::<super::distance::DotProduct>(
                    version_number,
                    &snapshot,
                    &current,
                )?;
            }
            DistanceMetric::Cosine => {
                self.compact_inner::<super::distance::Cosine>(version_number, &snapshot, &current)?;
            }
        }

        drop(compaction_guard);
        Ok(())
    }

    fn compact_inner<D: Distance>(
        &self,
        version_number: u64,
        snapshot: &LiveSnapshot,
        current: &SegmentList,
    ) -> Result<(), Error> {
        let new_version_dir = ensure_version_dir(&self.base_path, version_number)?;
        let mut new_manifest_entries: Vec<ManifestEntry> = Vec::new();
        let mut next_seg_id = current
            .segments
            .iter()
            .map(|s| s.segment_id)
            .max()
            .unwrap_or(0)
            + 1;

        // Step 1: Distribute live deletes to segments
        let per_segment_new_deletes = distribute_deletes(&current.segments, &snapshot.deletes);

        let last_idx = current.segments.len().checked_sub(1);
        let dimensions = self.config.dimensions;

        // Track whether live entries were absorbed
        let mut live_absorbed = false;

        // Step 2: Process each segment
        for (i, segment) in current.segments.iter().enumerate() {
            let new_deletes = &per_segment_new_deletes[i];
            let updated_deletes = merge_sorted_u64(segment.deletes_slice(), new_deletes);
            let updated_num_deletes = updated_deletes.len();

            let is_last = Some(i) == last_idx;
            let surviving =
                surviving_count(segment.num_nodes, segment.doc_ids_slice(), &updated_deletes);
            let actual_deleted_nodes = segment.num_nodes - surviving;
            let deletion_ratio = if segment.num_nodes > 0 {
                actual_deleted_nodes as f64 / segment.num_nodes as f64
            } else {
                0.0
            };
            let can_absorb = is_last
                && !snapshot.entries.is_empty()
                && (surviving + snapshot.entries.len())
                    <= self.segment_config.max_nodes_per_segment as usize;

            if can_absorb {
                live_absorbed = true;

                // Determine: incremental insert or full rebuild?
                let future_insertion_ratio = if segment.nodes_at_last_rebuild > 0 {
                    (segment.insertions_since_rebuild + snapshot.entries.len()) as f64
                        / segment.nodes_at_last_rebuild as f64
                } else {
                    f64::INFINITY
                };

                let needs_full_rebuild = future_insertion_ratio
                    > self.segment_config.insertion_rebuild_threshold
                    || deletion_ratio > self.segment_config.deletion_threshold.value();

                let seg_id = next_seg_id;
                next_seg_id += 1;

                if needs_full_rebuild {
                    // FULL REBUILD with absorb
                    let (mut all_vecs, mut all_ids) =
                        collect_surviving(segment, &updated_deletes, dimensions);

                    all_vecs.extend_from_slice(snapshot.entries.vectors_slice());
                    all_ids.extend_from_slice(snapshot.entries.doc_ids_slice());

                    if all_ids.is_empty() {
                        continue;
                    }

                    let params = QuantizationParams::calibrate(&all_vecs, dimensions);
                    build_and_write_segment::<D>(
                        &self.base_path,
                        seg_id,
                        &all_vecs,
                        &all_ids,
                        &self.config,
                        &params,
                    )?;
                    write_delete_file(&new_version_dir, seg_id, &[])?;

                    new_manifest_entries.push(ManifestEntry {
                        segment_id: seg_id,
                        num_nodes: all_ids.len(),
                        num_deletes: 0,
                        min_doc_id: all_ids[0],
                        max_doc_id: *all_ids.last().unwrap(),
                        nodes_at_last_rebuild: all_ids.len(),
                        insertions_since_rebuild: 0,
                    });
                } else {
                    // INCREMENTAL INSERT
                    let new_num_nodes = segment.num_nodes + snapshot.entries.len();
                    incremental_insert_segment::<D>(
                        &self.base_path,
                        segment,
                        seg_id,
                        &snapshot.entries,
                        &self.config,
                    )?;

                    // Carry forward deletes (not applied, just kept)
                    write_delete_file(&new_version_dir, seg_id, &updated_deletes)?;

                    let new_max_doc_id = snapshot
                        .entries
                        .last()
                        .map(|(id, _)| *id)
                        .unwrap_or(segment.max_doc_id);

                    new_manifest_entries.push(ManifestEntry {
                        segment_id: seg_id,
                        num_nodes: new_num_nodes,
                        num_deletes: updated_num_deletes,
                        min_doc_id: segment.min_doc_id,
                        max_doc_id: new_max_doc_id,
                        nodes_at_last_rebuild: segment.nodes_at_last_rebuild,
                        insertions_since_rebuild: segment.insertions_since_rebuild
                            + snapshot.entries.len(),
                    });
                }
            } else if deletion_ratio > self.segment_config.deletion_threshold.value() {
                // FULL REBUILD (non-absorb): apply deletes
                let (vectors, doc_ids) = collect_surviving(segment, &updated_deletes, dimensions);

                if doc_ids.is_empty() {
                    continue;
                }

                let seg_id = next_seg_id;
                next_seg_id += 1;

                let params = QuantizationParams::calibrate(&vectors, dimensions);
                build_and_write_segment::<D>(
                    &self.base_path,
                    seg_id,
                    &vectors,
                    &doc_ids,
                    &self.config,
                    &params,
                )?;
                write_delete_file(&new_version_dir, seg_id, &[])?;

                new_manifest_entries.push(ManifestEntry {
                    segment_id: seg_id,
                    num_nodes: doc_ids.len(),
                    num_deletes: 0,
                    min_doc_id: doc_ids[0],
                    max_doc_id: *doc_ids.last().unwrap(),
                    nodes_at_last_rebuild: doc_ids.len(),
                    insertions_since_rebuild: 0,
                });
            } else {
                // CARRY FORWARD: keep segment data, write updated delete file
                write_delete_file(&new_version_dir, segment.segment_id, &updated_deletes)?;

                new_manifest_entries.push(ManifestEntry {
                    segment_id: segment.segment_id,
                    num_nodes: segment.num_nodes,
                    num_deletes: updated_num_deletes,
                    min_doc_id: segment.min_doc_id,
                    max_doc_id: segment.max_doc_id,
                    nodes_at_last_rebuild: segment.nodes_at_last_rebuild,
                    insertions_since_rebuild: segment.insertions_since_rebuild,
                });
            }
        }

        // Step 3: If live entries weren't absorbed, create new segment
        if !live_absorbed && !snapshot.entries.is_empty() {
            let live_vecs = snapshot.entries.vectors_slice();
            let live_ids = snapshot.entries.doc_ids_slice();

            if !live_ids.is_empty() {
                let seg_id = next_seg_id;

                let params = QuantizationParams::calibrate(live_vecs, dimensions);
                build_and_write_segment::<D>(
                    &self.base_path,
                    seg_id,
                    live_vecs,
                    live_ids,
                    &self.config,
                    &params,
                )?;
                write_delete_file(&new_version_dir, seg_id, &[])?;

                new_manifest_entries.push(ManifestEntry {
                    segment_id: seg_id,
                    num_nodes: live_ids.len(),
                    num_deletes: 0,
                    min_doc_id: live_ids[0],
                    max_doc_id: *live_ids.last().unwrap(),
                    nodes_at_last_rebuild: live_ids.len(),
                    insertions_since_rebuild: 0,
                });
            }
        }

        // Handle edge case: all segments fully deleted and no live entries
        if new_manifest_entries.is_empty() {
            // Write an empty manifest
            write_manifest(&new_version_dir, &new_manifest_entries)?;
            sync_dir(&new_version_dir)?;
            let new_segment_list = SegmentList::load(&self.base_path, version_number)?;
            write_current_atomic(&self.base_path, version_number)?;

            let mut live = self.live.write().unwrap();
            self.segments.store(Arc::new(new_segment_list));
            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
            drop(live);
            return Ok(());
        }

        // Step 4: Write manifest and finalize
        write_manifest(&new_version_dir, &new_manifest_entries)?;
        sync_dir(&new_version_dir)?;
        let new_segment_list = SegmentList::load(&self.base_path, version_number)?;
        write_current_atomic(&self.base_path, version_number)?;

        {
            let mut live = self.live.write().unwrap();
            self.segments.store(Arc::new(new_segment_list));
            live.ops.drain(..snapshot.ops_len);
            live.ops.shrink_to_fit();
            live.refresh_snapshot();
            drop(live);
        }

        Ok(())
    }

    pub fn current_version_number(&self) -> u64 {
        self.segments.load().version_number
    }

    pub fn cleanup(&self) {
        let compaction_guard = self.compaction_lock.lock().unwrap();
        let current = self.segments.load();
        let current_version = current.version_number;

        // Collect segment_ids referenced by the current manifest
        let referenced_seg_ids: std::collections::HashSet<u64> =
            current.segments.iter().map(|s| s.segment_id).collect();

        // Clean up old version directories
        let version_numbers = match list_version_dirs(&self.base_path) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to list version directories: {e}");
                return;
            }
        };
        for vn in version_numbers {
            if vn != current_version {
                if let Err(e) = remove_version_dir(&self.base_path, vn) {
                    tracing::error!("Failed to remove old version {vn}: {e}");
                }
            }
        }

        // Clean up old segment directories not referenced by current manifest
        let all_seg_ids = match list_segment_dirs(&self.base_path) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to list segment directories: {e}");
                return;
            }
        };
        for seg_id in all_seg_ids {
            if !referenced_seg_ids.contains(&seg_id) {
                let seg_dir = segment_data_dir(&self.base_path, seg_id);
                if let Err(e) = std::fs::remove_dir_all(&seg_dir) {
                    tracing::error!("Failed to remove old segment {seg_id}: {e}");
                }
            }
        }
        drop(compaction_guard);
    }

    pub fn info(&self) -> IndexInfo {
        let current = self.segments.load();
        // Get a fresh snapshot to include live (pending) entries in the count.
        let snapshot = self.fresh_snapshot();
        let live = self.live.read().unwrap();
        let pending_ops = live.ops.len();
        drop(live);
        let ver_dir = version_dir(&self.base_path, current.version_number);

        // Count embeddings from both compacted segments and live (pending) entries.
        // After compaction the live layer is drained, so there is no double-counting.
        let num_embeddings = current.total_embeddings() + snapshot.entries.len();

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: current.version_number,
            version_dir: ver_dir,
            num_embeddings,
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            pending_ops,
        }
    }

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

                let ver_dir = version_dir(&self.base_path, version_number);
                if !ver_dir.exists() || !ver_dir.is_dir() {
                    checks.push(IntegrityCheck::failed(
                        "version directory",
                        Some(format!("Missing or invalid: {}", ver_dir.display())),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "version directory",
                    Some(ver_dir.display().to_string()),
                ));

                // Check manifest.json
                let manifest_path = ver_dir.join("manifest.json");
                if !manifest_path.exists() {
                    checks.push(IntegrityCheck::failed(
                        "manifest.json",
                        Some("File does not exist".to_string()),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "manifest.json",
                    Some("Present".to_string()),
                ));

                // Check segment directories
                let current = self.segments.load();
                let mut all_seg_ok = true;
                for segment in &current.segments {
                    let seg_dir = segment_data_dir(&self.base_path, segment.segment_id);
                    let required = [
                        "hnsw.meta",
                        "vectors.raw",
                        "vectors.quantized",
                        "hnsw.graph",
                        "doc_ids.bin",
                        "levels.bin",
                        "quantization.bin",
                    ];
                    let missing: Vec<&str> = required
                        .iter()
                        .filter(|f| !seg_dir.join(f).exists())
                        .copied()
                        .collect();
                    if !missing.is_empty() {
                        checks.push(IntegrityCheck::failed(
                            format!("seg_{}", segment.segment_id),
                            Some(format!("Missing: {}", missing.join(", "))),
                        ));
                        all_seg_ok = false;
                    }
                }
                if all_seg_ok {
                    checks.push(IntegrityCheck::ok(
                        "segments",
                        Some(format!(
                            "{} segment(s), all files present",
                            current.segments.len()
                        )),
                    ));
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

// ──────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────

/// Distribute live deletes to segments by doc_id range using binary search.
fn distribute_deletes(segments: &[super::segment::Segment], deletes: &[u64]) -> Vec<Vec<u64>> {
    let mut per_segment: Vec<Vec<u64>> = vec![Vec::new(); segments.len()];
    for &doc_id in deletes {
        let idx = segments.partition_point(|seg| seg.max_doc_id < doc_id);
        if idx < segments.len()
            && doc_id >= segments[idx].min_doc_id
            && doc_id <= segments[idx].max_doc_id
        {
            per_segment[idx].push(doc_id);
        }
    }
    // Each per-segment list is already sorted because deletes is sorted
    per_segment
}

/// Count surviving nodes in a segment (total - nodes that are deleted).
/// Uses a two-pointer merge scan (O(N + D)) instead of binary search per element (O(N log D)).
fn surviving_count(num_nodes: usize, doc_ids: &[u64], deletes: &[u64]) -> usize {
    if deletes.is_empty() {
        return num_nodes;
    }
    let mut deleted = 0;
    let mut d = 0;
    for &doc_id in doc_ids {
        while d < deletes.len() && deletes[d] < doc_id {
            d += 1;
        }
        if d < deletes.len() && deletes[d] == doc_id {
            deleted += 1;
            d += 1;
        }
    }
    num_nodes - deleted
}

/// Collect surviving vectors from a segment (skip deleted doc_ids).
///
/// Note: no check against live snapshot entries is needed because doc_ids are
/// strictly monotonically increasing — live entries always have doc_ids greater
/// than any segment doc_id, so they can never overlap.
fn collect_surviving(
    segment: &super::segment::Segment,
    deletes: &[u64],
    dimensions: usize,
) -> (Vec<f32>, Vec<u64>) {
    let surviving = surviving_count(segment.num_nodes, segment.doc_ids_slice(), deletes);
    let mut vectors = Vec::with_capacity(surviving * dimensions);
    let mut doc_ids = Vec::with_capacity(surviving);

    let mut d = 0;
    for i in 0..segment.num_nodes {
        let doc_id = segment.doc_id_at_unchecked(i as u32);
        while d < deletes.len() && deletes[d] < doc_id {
            d += 1;
        }
        if d < deletes.len() && deletes[d] == doc_id {
            d += 1;
            continue;
        }
        let raw_vec = segment.raw_vector_unchecked(i as u32, dimensions);
        vectors.extend_from_slice(raw_vec);
        doc_ids.push(doc_id);
    }

    (vectors, doc_ids)
}

/// Build an HNSW graph and write all segment files to disk.
fn build_and_write_segment<D: Distance>(
    base_path: &std::path::Path,
    seg_id: u64,
    vectors: &[f32],
    doc_ids: &[u64],
    config: &EmbeddingConfig,
    quant_params: &QuantizationParams,
) -> Result<(), Error> {
    let seg_dir = ensure_segment_dir(base_path, seg_id)?;
    let dimensions = config.dimensions;
    let num_nodes = doc_ids.len();

    // Build HNSW graph
    let mut builder = HnswBuilder::<D>::new(config);
    builder.build(vectors, doc_ids, dimensions)?;

    // Write raw vectors
    write_raw_vectors(&seg_dir.join("vectors.raw"), vectors)?;

    // Quantize and write quantized vectors
    let quantized = quant_params.quantize_all(vectors, dimensions);
    write_quantized_vectors(&seg_dir.join("vectors.quantized"), &quantized)?;
    quant_params.write_to_file(&seg_dir.join("quantization.bin"))?;

    // Write graph
    builder.write_graph(&seg_dir.join("hnsw.graph"))?;

    // Write doc_ids
    write_doc_ids(&seg_dir.join("doc_ids.bin"), doc_ids)?;

    // Write levels
    builder.write_levels(&seg_dir.join("levels.bin"))?;

    // Write meta
    write_meta(
        &seg_dir,
        config,
        num_nodes,
        config.max_level,
        builder.entry_point(),
    )?;

    Ok(())
}

/// Incremental insert: append live entries to an existing segment, producing a new segment.
fn incremental_insert_segment<D: Distance>(
    base_path: &std::path::Path,
    old_segment: &super::segment::Segment,
    new_seg_id: u64,
    live_entries: &super::live::FlatEntries,
    config: &EmbeddingConfig,
) -> Result<(), Error> {
    let seg_dir = ensure_segment_dir(base_path, new_seg_id)?;
    let dimensions = config.dimensions;
    let old_num_nodes = old_segment.num_nodes;
    let new_count = live_entries.len();
    let total_nodes = old_num_nodes + new_count;

    // 1. Write raw vectors: old ++ new
    let old_raw = old_segment.raw_vectors_slice();
    let new_raw = live_entries.vectors_slice();
    write_raw_vectors_concat(&seg_dir.join("vectors.raw"), old_raw, new_raw)?;

    // 2. Write doc_ids: old ++ new
    let old_doc_ids = old_segment.doc_ids_slice();
    let new_doc_ids = live_entries.doc_ids_slice();
    write_doc_ids_concat(&seg_dir.join("doc_ids.bin"), old_doc_ids, new_doc_ids)?;

    // 3. Assign levels to new nodes
    let old_levels = old_segment.levels_slice();
    let new_levels = assign_random_levels(new_count, config.max_level, config.m);

    // 4. Handle quantization (fast path or extend path)
    let old_params = &old_segment.quantization_params;
    let (new_mins, new_maxs) = compute_min_max(new_raw, dimensions);
    let params_fit = range_contains(old_params, &new_mins, &new_maxs);

    // All raw vectors (old + new) — allocated once and reused for both quantization and HNSW
    let all_raw: Vec<f32> = old_raw.iter().chain(new_raw.iter()).copied().collect();

    if params_fit {
        // FAST PATH: quantize only new vectors with existing params, append
        let old_quantized = old_segment.quantized_vectors_slice();
        let new_quantized = old_params.quantize_all(new_raw, dimensions);
        write_quantized_vectors_concat(
            &seg_dir.join("vectors.quantized"),
            old_quantized,
            &new_quantized,
        )?;
        old_params.write_to_file(&seg_dir.join("quantization.bin"))?;
    } else {
        // EXTEND PATH: widen params, re-quantize everything
        let extended_params = QuantizationParams::new(
            element_wise_min(&old_params.mins, &new_mins),
            element_wise_max(&old_params.maxs, &new_maxs),
            dimensions,
        );
        let quantized = extended_params.quantize_all(&all_raw, dimensions);
        write_quantized_vectors(&seg_dir.join("vectors.quantized"), &quantized)?;
        extended_params.write_to_file(&seg_dir.join("quantization.bin"))?;
    }

    // 5. Load old graph, insert new nodes, write updated graph
    let mut builder = HnswBuilder::<D>::load_from_graph(
        old_segment.graph_bytes(),
        old_levels,
        config,
        old_segment.config.node_block_size,
    )?;

    builder.insert_nodes(old_num_nodes, &new_levels, &all_raw, dimensions)?;

    builder.write_graph(&seg_dir.join("hnsw.graph"))?;

    // 6. Write levels: old ++ new
    write_levels_concat(
        &seg_dir.join("levels.bin"),
        old_levels,
        &new_levels.iter().map(|&l| l as u8).collect::<Vec<u8>>(),
    )?;

    // 7. Write metadata
    write_meta(
        &seg_dir,
        config,
        total_nodes,
        config.max_level,
        builder.entry_point(),
    )?;

    Ok(())
}

/// Assign random levels to new nodes (same distribution as HnswBuilder).
fn assign_random_levels(count: usize, max_level: usize, m: usize) -> Vec<usize> {
    use rand::RngExt;
    let mut rng = rand::rng();
    let ml = 1.0 / (m as f64).ln();
    (0..count)
        .map(|_| {
            let r: f64 = rng.random();
            ((-r.ln() * ml).floor() as usize).min(max_level - 1)
        })
        .collect()
}

// ──────────────────────────────────────────────────
// I/O helper functions
// ──────────────────────────────────────────────────

fn write_raw_vectors(path: &std::path::Path, vectors: &[f32]) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for &v in vectors {
        writer.write_all(&v.to_ne_bytes())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_raw_vectors_concat(path: &std::path::Path, old: &[f32], new: &[f32]) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for &v in old {
        writer.write_all(&v.to_ne_bytes())?;
    }
    for &v in new {
        writer.write_all(&v.to_ne_bytes())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_quantized_vectors(path: &std::path::Path, vectors: &[i8]) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len()) };
    writer.write_all(bytes)?;
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_quantized_vectors_concat(
    path: &std::path::Path,
    old: &[i8],
    new: &[i8],
) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    let old_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(old.as_ptr() as *const u8, old.len()) };
    let new_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(new.as_ptr() as *const u8, new.len()) };
    writer.write_all(old_bytes)?;
    writer.write_all(new_bytes)?;
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_doc_ids(path: &std::path::Path, doc_ids: &[u64]) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for &id in doc_ids {
        writer.write_all(&id.to_ne_bytes())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_doc_ids_concat(path: &std::path::Path, old: &[u64], new: &[u64]) -> Result<(), Error> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for &id in old {
        writer.write_all(&id.to_ne_bytes())?;
    }
    for &id in new {
        writer.write_all(&id.to_ne_bytes())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

fn write_levels_concat(path: &std::path::Path, old: &[u8], new: &[u8]) -> Result<(), Error> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(old)?;
    file.write_all(new)?;
    file.sync_all()?;
    Ok(())
}

fn write_meta(
    dir: &std::path::Path,
    config: &EmbeddingConfig,
    num_nodes: usize,
    max_level: usize,
    entry_point: u32,
) -> Result<(), Error> {
    let meta = serde_json::json!({
        "dimensions": config.dimensions,
        "metric": config.metric.to_string(),
        "m": config.m,
        "m0": config.m0,
        "ef_construction": config.ef_construction,
        "num_nodes": num_nodes,
        "max_level": max_level,
        "entry_point": entry_point,
        "node_block_size": config.node_block_size(),
    });
    let json = serde_json::to_string_pretty(&meta)
        .map_err(|e| Error::CorruptedFile(format!("failed to serialize meta: {e}")))?;
    let mut file = std::fs::File::create(dir.join("hnsw.meta"))?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

/// Merge two sorted u64 slices into a sorted Vec.
fn merge_sorted_u64(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            result.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{config::DistanceMetric, EmbeddingIndexer};
    use tempfile::TempDir;

    #[test]
    fn test_new_empty() {
        let tmp = TempDir::new().unwrap();
        let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
        let storage =
            EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default())
                .unwrap();
        assert_eq!(storage.current_version_number(), 0);
    }

    #[test]
    fn test_insert_validation() {
        let tmp = TempDir::new().unwrap();
        let config = EmbeddingConfig::new(3, DistanceMetric::L2).unwrap();
        let storage =
            EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default())
                .unwrap();

        let indexer = EmbeddingIndexer::new(3);

        if let Some(v) = indexer.index_vec(&[1.0, 2.0, 3.0]) {
            storage.insert(1, v);
        }
    }

    #[test]
    fn test_search_live_only() {
        let tmp = TempDir::new().unwrap();
        let config = EmbeddingConfig::new(2, DistanceMetric::L2).unwrap();
        let storage =
            EmbeddingStorage::new(tmp.path().to_path_buf(), config, SegmentConfig::default())
                .unwrap();

        let indexer = EmbeddingIndexer::new(2);

        let values = [
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![10.0, 10.0]),
        ];
        for (id, vec) in &values {
            if let Some(v) = indexer.index_vec(vec) {
                storage.insert(*id, v);
            }
        }

        let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // closest
    }

    #[test]
    fn test_merge_sorted_u64() {
        assert_eq!(
            merge_sorted_u64(&[1, 3, 5], &[2, 4, 6]),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(merge_sorted_u64(&[1, 2], &[2, 3]), vec![1, 2, 3]);
        assert_eq!(merge_sorted_u64(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(merge_sorted_u64(&[1, 2], &[]), vec![1, 2]);
    }
}
