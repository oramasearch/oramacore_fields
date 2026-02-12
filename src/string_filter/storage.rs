use super::compacted::CompactedVersion;
use super::config::Threshold;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    ensure_version_dir, list_version_dirs, read_current, remove_version_dir, sync_dir, version_dir,
    write_current_atomic, FORMAT_VERSION,
};
use super::iterator::FilterData;
use super::live::{LiveLayer, LiveSnapshot};
use super::merge::sorted_merge;
use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// String filter storage with exact-match filtering via FST-based compacted versions.
///
/// # Thread Safety
///
/// `StringFilterStorage` is `Send + Sync`:
/// - `version` (`ArcSwap<CompactedVersion>`): Lock-free reads via `load()`.
/// - `live` (`RwLock<LiveLayer>`): Protects in-memory mutations.
/// - `compaction_lock` (`Mutex<()>`): Serializes compaction operations.
pub struct StringFilterStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    threshold: Threshold,
}

impl StringFilterStorage {
    /// Create a new StringFilterStorage at the given path.
    ///
    /// If a previous version exists (detected via CURRENT file), it will be loaded.
    /// Otherwise, starts with an empty index.
    pub fn new(path: impl Into<PathBuf>, threshold: Threshold) -> Result<Self> {
        let base_path = path.into();
        fs::create_dir_all(&base_path)
            .with_context(|| format!("Failed to create base directory: {base_path:?}"))?;

        let version = match read_current(&base_path)? {
            Some((format_version, version_number)) => {
                if format_version != FORMAT_VERSION {
                    return Err(anyhow!(
                        "Unsupported format version {format_version}, expected {FORMAT_VERSION}"
                    ));
                }
                CompactedVersion::load(&base_path, version_number).with_context(|| {
                    format!("Failed to load version at version_number {version_number}")
                })?
            }
            None => CompactedVersion::empty(),
        };

        Ok(Self {
            base_path,
            version: ArcSwap::new(Arc::new(version)),
            live: RwLock::new(LiveLayer::new()),
            compaction_lock: Mutex::new(()),
            threshold,
        })
    }

    /// Insert a doc_id based on an IndexedValue extracted by StringIndexer.
    pub fn insert(&self, indexed_value: &super::indexer::IndexedValue, doc_id: u64) {
        match indexed_value {
            super::indexer::IndexedValue::Plain(s) => {
                let mut live = self.live.write().unwrap();
                live.insert(s, doc_id);
            }
            super::indexer::IndexedValue::Array(strings) => {
                let mut live = self.live.write().unwrap();
                for s in strings {
                    live.insert(s, doc_id);
                }
            }
        }
    }

    /// Delete a doc_id from all string value sets.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    /// Return filter data for zero-allocation iteration over doc_ids matching the exact string value.
    ///
    /// Uses double-check locking to minimize contention on the live layer.
    pub fn filter<'a>(&self, value: &'a str) -> FilterData<'a> {
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
        FilterData::new(Arc::clone(&version), snapshot, value)
    }

    /// Compact the index at the given version number.
    ///
    /// Merges the live layer into the compacted version, building a new FST + postings.
    pub fn compact(&self, version_number: u64) -> Result<()> {
        let _compaction_guard = self.compaction_lock.lock().unwrap();

        // Refresh snapshot (ops_len is captured inside the snapshot)
        let snapshot = {
            let mut live = self.live.write().unwrap();
            if live.is_snapshot_dirty() {
                live.refresh_snapshot();
            }
            live.get_snapshot()
        };

        let current = self.version.load();
        let new_version_dir = ensure_version_dir(&self.base_path, version_number)?;

        let should_apply_deletes = self.should_apply_deletes(&snapshot, &current);

        if should_apply_deletes {
            self.compact_apply_deletes(&snapshot, &current, &new_version_dir)?;
        } else {
            self.compact_carry_forward(&snapshot, &current, &new_version_dir)?;
        }

        sync_dir(&new_version_dir)?;
        write_current_atomic(&self.base_path, version_number)?;

        // Atomic update: swap version AND clear compacted items
        {
            let mut live = self.live.write().unwrap();
            let new_version = CompactedVersion::load(&self.base_path, version_number)?;
            self.version.store(Arc::new(new_version));

            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
        }

        Ok(())
    }

    /// Determine if deletions should be applied based on threshold.
    fn should_apply_deletes(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
    ) -> bool {
        if snapshot.deletes.is_empty() && current.deletes_slice().is_empty() {
            return false;
        }

        let merged_deletes_count = current.deletes_slice().len() + snapshot.deletes.len();

        // Estimate total postings: count all doc_ids across all compacted keys + live inserts
        let compacted_postings = current.total_postings();
        let total_postings = compacted_postings + snapshot.total_doc_ids();

        if total_postings == 0 {
            // Deletes exist (checked above) but no postings — apply deletes to clear them out.
            return true;
        }

        merged_deletes_count as f64 / total_postings as f64 > self.threshold.value()
    }

    /// Strategy A: Apply deletions — rebuild FST + postings with deletions subtracted.
    fn compact_apply_deletes(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
        new_version_dir: &std::path::Path,
    ) -> Result<()> {
        let deleted_set: std::collections::HashSet<u64> = sorted_merge(
            current.deletes_slice().iter().copied(),
            snapshot.deletes_sorted.iter().copied(),
        )
        .collect();

        let mut compacted_iter = current.iter_all();
        let mut live_iter = snapshot.iter_entries().peekable();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            Some(&deleted_set),
            std::iter::empty(),
            new_version_dir,
        )?;

        Ok(())
    }

    /// Strategy B: Carry forward deletions — merge new data, carry forward deleted.bin.
    fn compact_carry_forward(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
        new_version_dir: &std::path::Path,
    ) -> Result<()> {
        let mut compacted_iter = current.iter_all();
        let mut live_iter = snapshot.iter_entries().peekable();

        let deletes_iter = sorted_merge(
            current.deletes_slice().iter().copied(),
            snapshot.deletes_sorted.iter().copied(),
        );

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            None,
            deletes_iter,
            new_version_dir,
        )?;

        Ok(())
    }

    /// Get the current version number.
    pub fn current_version_number(&self) -> u64 {
        self.version.load().version_number
    }

    /// Remove all old version directories except the current one.
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

    /// Get metadata and statistics about the index.
    pub fn info(&self) -> IndexInfo {
        let version = self.version.load();
        let live = self.live.read().unwrap();
        let ver_dir = version_dir(&self.base_path, version.version_number);

        let compacted_postings = version.total_postings();

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: version.version_number,
            version_dir: ver_dir.clone(),
            unique_keys_count: version.key_count(),
            total_postings_count: compacted_postings,
            deleted_count: version.deletes_slice().len(),
            fst_size_bytes: file_size(&ver_dir.join("keys.fst")),
            postings_size_bytes: file_size(&ver_dir.join("postings.dat")),
            deleted_size_bytes: file_size(&ver_dir.join("deleted.bin")),
            pending_ops: live.ops.len(),
        }
    }

    /// Check the integrity of the index files.
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

                // Check required files
                let fst_path = ver_dir.join("keys.fst");
                let postings_path = ver_dir.join("postings.dat");
                let deleted_path = ver_dir.join("deleted.bin");

                if !fst_path.exists() || !postings_path.exists() || !deleted_path.exists() {
                    let mut missing = Vec::new();
                    if !fst_path.exists() {
                        missing.push("keys.fst");
                    }
                    if !postings_path.exists() {
                        missing.push("postings.dat");
                    }
                    if !deleted_path.exists() {
                        missing.push("deleted.bin");
                    }
                    checks.push(IntegrityCheck::failed(
                        "index files",
                        Some(format!("Missing: {}", missing.join(", "))),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "index files",
                    Some("All present".to_string()),
                ));

                // Check deleted.bin is valid (size % 8 == 0, values strictly sorted)
                match validate_deleted_file(&deleted_path) {
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
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
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
    use super::super::indexer::IndexedValue;
    use super::*;
    use tempfile::TempDir;

    fn p(s: &str) -> IndexedValue {
        IndexedValue::Plain(s.to_string())
    }

    #[test]
    fn test_new_empty_index() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();
        assert_eq!(index.current_version_number(), 0);
    }

    #[test]
    fn test_insert_and_filter() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("hello"), 10);
        index.insert(&p("world"), 2);
        index.insert(&p("world"), 6);

        let hello_results: Vec<u64> = index.filter("hello").iter().collect();
        let world_results: Vec<u64> = index.filter("world").iter().collect();
        let missing_results: Vec<u64> = index.filter("missing").iter().collect();

        assert_eq!(hello_results, vec![1, 5, 10]);
        assert_eq!(world_results, vec![2, 6]);
        assert!(missing_results.is_empty());
    }

    #[test]
    fn test_delete_and_filter() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("hello"), 10);
        index.insert(&p("world"), 5);
        index.delete(5);

        let hello_results: Vec<u64> = index.filter("hello").iter().collect();
        let world_results: Vec<u64> = index.filter("world").iter().collect();

        assert_eq!(hello_results, vec![1, 10]);
        assert!(world_results.is_empty());
    }

    #[test]
    fn test_compact_basic() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("world"), 2);

        index.compact(1).unwrap();

        let hello_results: Vec<u64> = index.filter("hello").iter().collect();
        let world_results: Vec<u64> = index.filter("world").iter().collect();

        assert_eq!(hello_results, vec![1, 5]);
        assert_eq!(world_results, vec![2]);
        assert_eq!(index.current_version_number(), 1);
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringFilterStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(&p("hello"), 1);
            index.insert(&p("hello"), 5);
            index.insert(&p("world"), 2);
            index.compact(1).unwrap();
        }

        {
            let index = StringFilterStorage::new(base_path, Threshold::default()).unwrap();
            let hello_results: Vec<u64> = index.filter("hello").iter().collect();
            let world_results: Vec<u64> = index.filter("world").iter().collect();

            assert_eq!(hello_results, vec![1, 5]);
            assert_eq!(world_results, vec![2]);
            assert_eq!(index.current_version_number(), 1);
        }
    }

    #[test]
    fn test_compact_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("hello"), 10);
        index.delete(5);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 10]);
    }

    #[test]
    fn test_ops_during_compaction_preserved() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);

        index.compact(1).unwrap();

        index.insert(&p("hello"), 10);
        index.insert(&p("hello"), 20);

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 5, 10, 20]);
    }

    #[test]
    fn test_compact_multiple_rounds() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.compact(1).unwrap();

        index.insert(&p("hello"), 10);
        index.compact(2).unwrap();

        index.insert(&p("hello"), 20);
        index.compact(3).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 5, 10, 20]);
    }

    #[test]
    fn test_compact_carries_forward_deletes() {
        let tmp = TempDir::new().unwrap();
        // High threshold so deletes are carried forward
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), 0.9f64.try_into().unwrap()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("hello"), 10);
        index.delete(5);
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 10]);

        index.insert(&p("hello"), 20);
        index.insert(&p("hello"), 30);
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 10, 20, 30]);
    }

    #[test]
    fn test_compact_empty_to_non_empty() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 10);
        index.insert(&p("hello"), 20);
        index.insert(&p("world"), 15);

        index.compact(1).unwrap();

        let hello_results: Vec<u64> = index.filter("hello").iter().collect();
        let world_results: Vec<u64> = index.filter("world").iter().collect();

        assert_eq!(hello_results, vec![10, 20]);
        assert_eq!(world_results, vec![15]);
    }

    #[test]
    fn test_cleanup_removes_old_versions() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.compact(1).unwrap();
        assert!(tmp.path().join("versions/1").exists());

        index.insert(&p("hello"), 2);
        index.compact(2).unwrap();
        assert!(tmp.path().join("versions/1").exists());
        assert!(tmp.path().join("versions/2").exists());

        index.insert(&p("hello"), 3);
        index.compact(3).unwrap();

        index.cleanup();

        assert!(!tmp.path().join("versions/1").exists());
        assert!(!tmp.path().join("versions/2").exists());
        assert!(tmp.path().join("versions/3").exists());
    }

    #[test]
    fn test_empty_string_key() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p(""), 1);
        index.insert(&p(""), 2);

        let results: Vec<u64> = index.filter("").iter().collect();
        assert_eq!(results, vec![1, 2]);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("").iter().collect();
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn test_unicode_keys() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("日本語"), 1);
        index.insert(&p("中文"), 2);
        index.insert(&p("한국어"), 3);
        index.insert(&p("emoji🎉"), 4);

        let results: Vec<u64> = index.filter("日本語").iter().collect();
        assert_eq!(results, vec![1]);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("中文").iter().collect();
        assert_eq!(results, vec![2]);

        let results: Vec<u64> = index.filter("emoji🎉").iter().collect();
        assert_eq!(results, vec![4]);
    }

    #[test]
    fn test_doc_id_zero() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 0);
        index.insert(&p("hello"), 1);

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![0, 1]);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![0, 1]);
    }

    #[test]
    fn test_many_unique_keys() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        for i in 0..1000 {
            index.insert(&p(&format!("key_{i:04}")), i);
        }

        let results: Vec<u64> = index.filter("key_0500").iter().collect();
        assert_eq!(results, vec![500]);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("key_0500").iter().collect();
        assert_eq!(results, vec![500]);

        let results: Vec<u64> = index.filter("key_0999").iter().collect();
        assert_eq!(results, vec![999]);

        let results: Vec<u64> = index.filter("key_1000").iter().collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_compact_reinsert_same_doc_id() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 5);
        index.delete(5);
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert!(!results.contains(&5));

        index.insert(&p("hello"), 5);
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert!(
            results.contains(&5),
            "doc 5 should be present after re-insert + compact, got: {results:?}"
        );
    }

    #[test]
    fn test_open_incompatible_format_version() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringFilterStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(&p("hello"), 1);
            index.compact(1).unwrap();
        }

        let current_path = base_path.join("CURRENT");
        fs::write(&current_path, "999\n1").unwrap();

        let result = StringFilterStorage::new(base_path, Threshold::default());
        match result {
            Ok(_) => panic!("Expected error for incompatible format version"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("Unsupported format version"),
                    "Expected 'Unsupported format version' error, got: {err_msg}"
                );
            }
        }
    }

    #[test]
    fn test_filter_data_into_iter() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 5);
        index.insert(&p("hello"), 10);

        let filter_data = index.filter("hello");

        let mut results = Vec::new();
        for doc_id in &filter_data {
            results.push(doc_id);
        }

        assert_eq!(results, vec![1, 5, 10]);
    }
}
