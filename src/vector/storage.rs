use super::compacted::CompactedVersion;
use super::config::VectorConfig;
use super::distance::{resolve_distance_fn, resolve_quantized_distance_fn, DistanceFn, QuantizedDistanceFn};
use super::error::Error;
use super::hnsw::HnswBuilder;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    ensure_version_dir, list_version_dirs, read_current, remove_version_dir, sync_dir,
    version_dir, write_current_atomic, write_postings, FORMAT_VERSION,
};
use super::live::{LiveLayer, LiveSnapshot};
use super::quantization::QuantizationParams;
use arc_swap::ArcSwap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

pub struct VectorStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    config: VectorConfig,
    distance_fn: DistanceFn,
    quantized_distance_fn: QuantizedDistanceFn,
}

impl VectorStorage {
    pub fn new(base_path: PathBuf, config: VectorConfig) -> Result<Self, Error> {
        std::fs::create_dir_all(&base_path)?;

        let distance_fn = resolve_distance_fn(config.metric);
        let quantized_distance_fn = resolve_quantized_distance_fn(config.metric);

        let version = match read_current(&base_path)? {
            Some((format_version, version_number)) => {
                if format_version != FORMAT_VERSION {
                    return Err(Error::FormatVersionMismatch {
                        expected: FORMAT_VERSION,
                        found: format_version,
                    });
                }
                CompactedVersion::load(&base_path, version_number)?
            }
            None => CompactedVersion::empty(),
        };

        Ok(Self {
            base_path,
            version: ArcSwap::new(Arc::new(version)),
            live: RwLock::new(LiveLayer::new()),
            compaction_lock: Mutex::new(()),
            config,
            distance_fn,
            quantized_distance_fn,
        })
    }

    /// Insert a vector for a doc_id.
    pub fn insert(&self, doc_id: u64, vector: &[f32]) -> Result<(), Error> {
        if vector.is_empty() {
            return Err(Error::EmptyVector);
        }
        if vector.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        for &v in vector {
            if !v.is_finite() {
                return Err(Error::NonFiniteValue);
            }
        }
        let mut live = self.live.write().unwrap();
        live.insert(doc_id, vector.to_vec());
        Ok(())
    }

    /// Delete a doc_id.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
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
        live.get_snapshot()
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, Error> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let snapshot = self.fresh_snapshot();
        let compacted = self.version.load();
        let ef = ef_search.unwrap_or(self.config.ef_search);

        // Merge deletes from both layers
        let compacted_deletes = compacted.deletes_slice();
        let merged_deletes = merge_sorted_u64(compacted_deletes, &snapshot.deletes);

        // Search compacted HNSW
        let mut compacted_results = if compacted.config.is_some() {
            let mut query_quantized = vec![0i8; self.config.dimensions];
            if let Some(ref params) = compacted.quantization_params {
                params.quantize(query, &mut query_quantized);
            }
            compacted.search(
                query,
                &query_quantized,
                k,
                ef,
                self.distance_fn,
                self.quantized_distance_fn,
                &merged_deletes,
            )
        } else {
            Vec::new()
        };

        // Search live layer (brute force)
        let live_results =
            snapshot.search(query, k, self.distance_fn, &merged_deletes);

        // Merge results
        compacted_results.extend(live_results);
        compacted_results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate by doc_id (keep closest)
        let mut seen = std::collections::HashSet::new();
        compacted_results.retain(|(doc_id, _)| seen.insert(*doc_id));

        compacted_results.truncate(k);
        Ok(compacted_results)
    }

    /// Compact the index.
    pub fn compact(&self, version_number: u64) -> Result<(), Error> {
        let _compaction_guard = self.compaction_lock.lock().unwrap();

        let snapshot = self.fresh_snapshot();

        // Early return if nothing to do
        if snapshot.is_empty() {
            let current = self.version.load();
            if current.config.is_none() {
                let mut live = self.live.write().unwrap();
                live.ops.drain(..snapshot.ops_len);
                live.ops.shrink_to_fit();
                live.refresh_snapshot();
                return Ok(());
            }
        }

        let current = self.version.load();

        let new_version_dir = ensure_version_dir(&self.base_path, version_number)?;

        // Collect all vectors
        let dimensions = self.config.dimensions;
        let mut all_vectors: Vec<f32> = Vec::new();
        let mut all_doc_ids: Vec<u64> = Vec::new();

        // From compacted: skip deleted
        if let Some(ref config) = current.config {
            let compacted_deletes = current.deletes_slice();
            let merged_deletes = merge_sorted_u64(compacted_deletes, &snapshot.deletes);

            for i in 0..config.num_nodes {
                let doc_id = current.doc_id_at_unchecked(i as u32);
                if merged_deletes.binary_search(&doc_id).is_ok() {
                    continue;
                }
                // Also skip if the doc is re-inserted in live (we'll use the live version)
                if snapshot
                    .entries
                    .binary_search_by_key(&doc_id, |(id, _)| *id)
                    .is_ok()
                {
                    continue;
                }
                let raw_vec = current.raw_vector_unchecked(i as u32, dimensions);
                all_vectors.extend_from_slice(raw_vec);
                all_doc_ids.push(doc_id);
            }
        }

        // From live snapshot: entries (already excludes live deletes)
        for (doc_id, vector) in &snapshot.entries {
            if snapshot.deletes.binary_search(doc_id).is_ok() {
                continue;
            }
            all_vectors.extend_from_slice(vector);
            all_doc_ids.push(*doc_id);
        }

        let num_nodes = all_doc_ids.len();

        if num_nodes == 0 {
            // Write empty files
            write_postings(&new_version_dir.join("deleted.bin"), &[])?;
            Self::write_meta(&new_version_dir, &self.config, 0, 0, 0)?;

            sync_dir(&new_version_dir)?;
            write_current_atomic(&self.base_path, version_number)?;

            let mut live = self.live.write().unwrap();
            let new_version = CompactedVersion::load(&self.base_path, version_number)?;
            self.version.store(Arc::new(new_version));
            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
            return Ok(());
        }

        if num_nodes > u32::MAX as usize {
            return Err(Error::TooManyNodes {
                count: num_nodes,
                max: u32::MAX as usize,
            });
        }

        // Build HNSW graph
        let mut builder = HnswBuilder::new(&self.config, self.distance_fn);
        builder.build(&all_vectors, &all_doc_ids, dimensions)?;

        // Write raw vectors
        Self::write_raw_vectors(&new_version_dir.join("vectors.raw"), &all_vectors)?;

        // Calibrate quantization and write quantized vectors
        let quant_params = QuantizationParams::calibrate(&all_vectors, dimensions);
        let quantized = quant_params.quantize_all(&all_vectors, dimensions);
        Self::write_quantized_vectors(&new_version_dir.join("vectors.quantized"), &quantized)?;
        quant_params.write_to_file(&new_version_dir.join("quantization.bin"))?;

        // Write graph
        builder.write_graph(&new_version_dir.join("hnsw.graph"))?;

        // Write doc_ids
        Self::write_doc_ids(&new_version_dir.join("doc_ids.bin"), &all_doc_ids)?;

        // Write levels
        builder.write_levels(&new_version_dir.join("levels.bin"))?;

        // Write empty deleted.bin
        write_postings(&new_version_dir.join("deleted.bin"), &[])?;

        // Write meta (use config.max_level, not builder.max_level(),
        // because the graph is written with config.max_level for block sizing)
        Self::write_meta(
            &new_version_dir,
            &self.config,
            num_nodes,
            self.config.max_level,
            builder.entry_point(),
        )?;

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_number)?;

        // Atomic swap
        {
            let mut live = self.live.write().unwrap();
            let new_version = CompactedVersion::load(&self.base_path, version_number)?;
            self.version.store(Arc::new(new_version));
            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
        }

        Ok(())
    }

    fn write_raw_vectors(path: &std::path::Path, vectors: &[f32]) -> Result<(), Error> {
        let mut file = std::fs::File::create(path)?;
        for &v in vectors {
            file.write_all(&v.to_ne_bytes())?;
        }
        file.sync_all()?;
        Ok(())
    }

    fn write_quantized_vectors(path: &std::path::Path, vectors: &[i8]) -> Result<(), Error> {
        let mut file = std::fs::File::create(path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len())
        };
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn write_doc_ids(path: &std::path::Path, doc_ids: &[u64]) -> Result<(), Error> {
        let mut file = std::fs::File::create(path)?;
        for &id in doc_ids {
            file.write_all(&id.to_ne_bytes())?;
        }
        file.sync_all()?;
        Ok(())
    }

    fn write_meta(
        dir: &std::path::Path,
        config: &VectorConfig,
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
        });
        let json = serde_json::to_string_pretty(&meta)
            .map_err(|e| Error::CorruptedFile(format!("failed to serialize meta: {e}")))?;
        let mut file = std::fs::File::create(dir.join("hnsw.meta"))?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        Ok(())
    }

    pub fn current_version_number(&self) -> u64 {
        self.version.load().version_number
    }

    pub fn cleanup(&self) {
        let _compaction_guard = self.compaction_lock.lock().unwrap();
        let current_version = self.version.load().version_number;
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
    }

    pub fn info(&self) -> IndexInfo {
        let version = self.version.load();
        let live = self.live.read().unwrap();
        let ver_dir = version_dir(&self.base_path, version.version_number);

        let num_vectors = version
            .config
            .as_ref()
            .map(|c| c.num_nodes)
            .unwrap_or(0);

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: version.version_number,
            version_dir: ver_dir,
            num_vectors,
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            pending_ops: live.ops.len(),
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
                    Some(format!("version: {format_version}, version_number: {version_number}")),
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

                // Check required files
                let required = [
                    "hnsw.meta",
                    "vectors.raw",
                    "vectors.quantized",
                    "hnsw.graph",
                    "doc_ids.bin",
                    "levels.bin",
                    "deleted.bin",
                    "quantization.bin",
                ];
                let missing: Vec<&str> = required
                    .iter()
                    .filter(|f| !ver_dir.join(f).exists())
                    .copied()
                    .collect();

                if missing.is_empty() {
                    checks.push(IntegrityCheck::ok(
                        "data files",
                        Some("All present".to_string()),
                    ));
                } else {
                    checks.push(IntegrityCheck::failed(
                        "data files",
                        Some(format!("Missing: {}", missing.join(", "))),
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

impl CompactedVersion {
    /// Unchecked doc_id access (for internal use during compaction).
    pub(crate) fn doc_id_at_unchecked(&self, node_idx: u32) -> u64 {
        let mmap = self.doc_ids.as_ref().expect("doc_ids mmap missing");
        let offset = node_idx as usize * 8;
        let bytes: [u8; 8] = mmap[offset..offset + 8].try_into().unwrap();
        u64::from_ne_bytes(bytes)
    }

    /// Unchecked raw vector access (for internal use during compaction).
    pub(crate) fn raw_vector_unchecked(&self, node_idx: u32, dimensions: usize) -> &[f32] {
        let mmap = self.raw_vectors.as_ref().expect("raw_vectors mmap missing");
        let offset = node_idx as usize * dimensions * 4;
        let ptr = mmap[offset..].as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, dimensions) }
    }
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
    use crate::vector::config::DistanceMetric;
    use tempfile::TempDir;

    #[test]
    fn test_new_empty() {
        let tmp = TempDir::new().unwrap();
        let config = VectorConfig::new(2, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(tmp.path().to_path_buf(), config).unwrap();
        assert_eq!(storage.current_version_number(), 0);
    }

    #[test]
    fn test_insert_validation() {
        let tmp = TempDir::new().unwrap();
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(tmp.path().to_path_buf(), config).unwrap();

        // Valid insert
        assert!(storage.insert(1, &[1.0, 2.0, 3.0]).is_ok());

        // Wrong dimensions
        assert!(storage.insert(2, &[1.0, 2.0]).is_err());

        // Empty vector
        assert!(storage.insert(3, &[]).is_err());

        // Non-finite value
        assert!(storage.insert(4, &[1.0, f32::NAN, 3.0]).is_err());
        assert!(storage.insert(5, &[1.0, f32::INFINITY, 3.0]).is_err());
    }

    #[test]
    fn test_search_live_only() {
        let tmp = TempDir::new().unwrap();
        let config = VectorConfig::new(2, DistanceMetric::L2).unwrap();
        let storage = VectorStorage::new(tmp.path().to_path_buf(), config).unwrap();

        storage.insert(1, &[0.0, 0.0]).unwrap();
        storage.insert(2, &[1.0, 0.0]).unwrap();
        storage.insert(3, &[10.0, 10.0]).unwrap();

        let results = storage.search(&[0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // closest
    }

    #[test]
    fn test_merge_sorted_u64() {
        assert_eq!(merge_sorted_u64(&[1, 3, 5], &[2, 4, 6]), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(merge_sorted_u64(&[1, 2], &[2, 3]), vec![1, 2, 3]);
        assert_eq!(merge_sorted_u64(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(merge_sorted_u64(&[1, 2], &[]), vec![1, 2]);
    }
}
