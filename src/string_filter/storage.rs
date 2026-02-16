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

/// Persistent string filter index supporting exact-match lookups, concurrent reads and writes,
/// and compaction to disk.
pub struct StringFilterStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    threshold: Threshold,
}

impl StringFilterStorage {
    /// Open or create a string filter index at the given path.
    ///
    /// Loads an existing index if one is found, otherwise starts empty.
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

    /// Insert a doc_id for the given indexed string value(s).
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

    /// Mark a doc_id as deleted across all keys.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    /// Return filter data for iterating over doc_ids matching the exact string value.
    pub fn filter<'a>(&self, value: &'a str) -> FilterData<'a> {
        let (snapshot, version) = {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                (live.get_snapshot(), self.version.load())
            } else {
                drop(live);
                let mut live = self.live.write().unwrap();
                if live.is_snapshot_dirty() {
                    live.refresh_snapshot();
                }
                (live.get_snapshot(), self.version.load())
            }
        };
        FilterData::new(Arc::clone(&version), snapshot, value)
    }

    /// Persist pending changes to disk at the given version number.
    pub fn compact(&self, version_number: u64) -> Result<()> {
        let _compaction_guard = self.compaction_lock.lock().unwrap();

        // Take snapshot (double-check locking to avoid blocking readers)
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

        // Nothing to compact — free memory and return early
        if snapshot.is_empty() {
            let mut live = self.live.write().unwrap();
            live.ops.drain(..snapshot.ops_len);
            live.ops.shrink_to_fit();
            live.refresh_snapshot();
            return Ok(());
        }

        let current = self.version.load();

        if version_number == current.version_number && current.has_data() {
            return Err(anyhow!(
                "Cannot compact to version {version_number}: same as current active version. \
                 Use a different version number to avoid corrupting active mmaps."
            ));
        }

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

    /// Check whether deletions exceed the threshold for physical removal.
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

    /// Compact by physically removing deleted doc_ids from the data.
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

    /// Compact by merging new data while carrying deletions forward for later removal.
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

    /// Return the current compacted version number.
    pub fn current_version_number(&self) -> u64 {
        self.version.load().version_number
    }

    /// Delete old version directories, keeping only the current one.
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

    /// Return metadata and statistics about the index.
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

    // ──────────────────────────────────────────────────────────────
    //  Array values
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_array_values() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        let arr = IndexedValue::Array(vec!["red".into(), "blue".into(), "green".into()]);
        index.insert(&arr, 1);
        index.insert(&IndexedValue::Array(vec!["red".into(), "yellow".into()]), 2);

        let red: Vec<u64> = index.filter("red").iter().collect();
        let blue: Vec<u64> = index.filter("blue").iter().collect();
        let yellow: Vec<u64> = index.filter("yellow").iter().collect();
        let green: Vec<u64> = index.filter("green").iter().collect();

        assert_eq!(red, vec![1, 2]);
        assert_eq!(blue, vec![1]);
        assert_eq!(yellow, vec![2]);
        assert_eq!(green, vec![1]);
    }

    #[test]
    fn test_insert_array_values_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        let arr = IndexedValue::Array(vec!["red".into(), "blue".into()]);
        index.insert(&arr, 1);
        index.compact(1).unwrap();

        let red: Vec<u64> = index.filter("red").iter().collect();
        let blue: Vec<u64> = index.filter("blue").iter().collect();
        assert_eq!(red, vec![1]);
        assert_eq!(blue, vec![1]);

        // Add more after compaction
        index.insert(&IndexedValue::Array(vec!["red".into(), "green".into()]), 2);
        let red: Vec<u64> = index.filter("red").iter().collect();
        assert_eq!(red, vec![1, 2]);

        index.compact(2).unwrap();
        let red: Vec<u64> = index.filter("red").iter().collect();
        assert_eq!(red, vec![1, 2]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Delete-then-reinsert to different key
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_then_reinsert_different_key_separate_batches() {
        // When delete and re-insert happen in separate compaction rounds,
        // the old key is properly cleaned up.
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("color_red"), 1);
        index.compact(1).unwrap();

        index.delete(1);
        index.compact(2).unwrap();

        index.insert(&p("color_blue"), 1);
        index.compact(3).unwrap();

        let red: Vec<u64> = index.filter("color_red").iter().collect();
        let blue: Vec<u64> = index.filter("color_blue").iter().collect();
        assert!(red.is_empty(), "doc 1 should no longer be under color_red");
        assert_eq!(blue, vec![1], "doc 1 should now be under color_blue");
    }

    #[test]
    fn test_delete_then_reinsert_different_key_same_batch() {
        // When delete + re-insert happen in the same live batch, the live
        // layer's replay consumes the delete (since the re-insert brings the
        // doc_id back). This means the old compacted key entry is preserved.
        // This is a known characteristic of the architecture.
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("color_red"), 1);
        index.compact(1).unwrap();

        index.delete(1);
        index.insert(&p("color_blue"), 1);
        index.compact(2).unwrap();

        let red: Vec<u64> = index.filter("color_red").iter().collect();
        let blue: Vec<u64> = index.filter("color_blue").iter().collect();
        // Doc 1 appears under both keys because the delete was consumed by re-insert
        assert_eq!(red, vec![1]);
        assert_eq!(blue, vec![1]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact with only deletes (no inserts in live layer)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_only_deletes() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.insert(&p("hello"), 3);
        index.compact(1).unwrap();

        // Only deletes in the live layer
        index.delete(2);
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Multiple keys across multiple compaction rounds
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_keys_across_compaction_rounds() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Round 1: insert into "alpha" and "beta"
        index.insert(&p("alpha"), 1);
        index.insert(&p("beta"), 2);
        index.compact(1).unwrap();

        // Round 2: add more to "alpha", introduce "gamma"
        index.insert(&p("alpha"), 3);
        index.insert(&p("gamma"), 4);
        index.compact(2).unwrap();

        // Round 3: add to "beta" and "gamma"
        index.insert(&p("beta"), 5);
        index.insert(&p("gamma"), 6);
        index.compact(3).unwrap();

        let alpha: Vec<u64> = index.filter("alpha").iter().collect();
        let beta: Vec<u64> = index.filter("beta").iter().collect();
        let gamma: Vec<u64> = index.filter("gamma").iter().collect();

        assert_eq!(alpha, vec![1, 3]);
        assert_eq!(beta, vec![2, 5]);
        assert_eq!(gamma, vec![4, 6]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Delete doc_id that was already compacted
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_compacted_doc_id() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.insert(&p("hello"), 3);
        index.compact(1).unwrap();

        // Delete a compacted doc_id, verify it's filtered before re-compaction
        index.delete(2);
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 3]);

        // Now compact and verify it persists
        index.compact(2).unwrap();
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Delete non-existent doc_id
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_nonexistent_doc_id() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.delete(999); // doesn't exist

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1]);

        index.compact(1).unwrap();
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Interleaved insert/delete/compact
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_interleaved_insert_delete_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("a"), 1);
        index.insert(&p("a"), 2);
        index.insert(&p("b"), 3);
        index.compact(1).unwrap();

        index.delete(1);
        index.insert(&p("a"), 4);
        index.insert(&p("c"), 5);
        index.compact(2).unwrap();

        index.delete(3);
        index.delete(5);
        index.insert(&p("a"), 6);
        index.compact(3).unwrap();

        let a: Vec<u64> = index.filter("a").iter().collect();
        let b: Vec<u64> = index.filter("b").iter().collect();
        let c: Vec<u64> = index.filter("c").iter().collect();

        assert_eq!(a, vec![2, 4, 6]);
        assert!(b.is_empty());
        assert!(c.is_empty());
    }

    // ──────────────────────────────────────────────────────────────
    //  Same doc_id under multiple keys
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_same_doc_id_multiple_keys() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("color"), 1);
        index.insert(&p("shape"), 1);
        index.insert(&p("size"), 1);
        index.compact(1).unwrap();

        let color: Vec<u64> = index.filter("color").iter().collect();
        let shape: Vec<u64> = index.filter("shape").iter().collect();
        let size: Vec<u64> = index.filter("size").iter().collect();

        assert_eq!(color, vec![1]);
        assert_eq!(shape, vec![1]);
        assert_eq!(size, vec![1]);

        // Delete should remove from all keys
        index.delete(1);
        let color: Vec<u64> = index.filter("color").iter().collect();
        let shape: Vec<u64> = index.filter("shape").iter().collect();
        assert!(color.is_empty());
        assert!(shape.is_empty());
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact when everything is deleted
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_all_deleted() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.delete(1);
        index.delete(2);
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert!(results.is_empty());

        // Re-insert after all deleted
        index.insert(&p("hello"), 3);
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![3]);

        index.compact(2).unwrap();
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Threshold: apply-deletes strategy (low threshold)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_apply_deletes_low_threshold() {
        let tmp = TempDir::new().unwrap();
        // threshold = 0.0 means always apply deletes
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), 0.0f64.try_into().unwrap()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.insert(&p("hello"), 3);
        index.delete(2);
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 3]);

        // After apply-deletes, deleted.bin should be empty
        let info = index.info();
        assert_eq!(
            info.deleted_count, 0,
            "deletes should be applied, not carried"
        );
    }

    // ──────────────────────────────────────────────────────────────
    //  Threshold: carry-forward strategy (high threshold)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_carry_forward_high_threshold() {
        let tmp = TempDir::new().unwrap();
        // threshold = 1.0 means never apply deletes (always carry forward)
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), 1.0f64.try_into().unwrap()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.insert(&p("hello"), 3);
        index.delete(2);
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1, 3]);

        // Deletes should be carried forward
        let info = index.info();
        assert!(
            info.deleted_count > 0,
            "deletes should be carried forward, not applied"
        );
    }

    // ──────────────────────────────────────────────────────────────
    //  Multiple compaction rounds with deletes
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_compaction_rounds_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("x"), 1);
        index.insert(&p("x"), 2);
        index.insert(&p("x"), 3);
        index.compact(1).unwrap();

        index.delete(1);
        index.insert(&p("x"), 4);
        index.compact(2).unwrap();

        index.delete(3);
        index.insert(&p("x"), 5);
        index.compact(3).unwrap();

        let results: Vec<u64> = index.filter("x").iter().collect();
        assert_eq!(results, vec![2, 4, 5]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Persistence with deletes
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_persistence_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringFilterStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(&p("hello"), 1);
            index.insert(&p("hello"), 2);
            index.insert(&p("hello"), 3);
            index.delete(2);
            index.compact(1).unwrap();
        }

        {
            let index = StringFilterStorage::new(base_path, Threshold::default()).unwrap();
            let results: Vec<u64> = index.filter("hello").iter().collect();
            assert_eq!(results, vec![1, 3]);
        }
    }

    // ──────────────────────────────────────────────────────────────
    //  Persistence with carry-forward deletes
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_persistence_carry_forward_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            // Use high threshold so deletes are carried forward
            let index =
                StringFilterStorage::new(base_path.clone(), 0.9f64.try_into().unwrap()).unwrap();
            index.insert(&p("hello"), 1);
            index.insert(&p("hello"), 2);
            index.insert(&p("hello"), 3);
            index.delete(2);
            index.compact(1).unwrap();
        }

        {
            // Reopen and verify deletes are still effective
            let index = StringFilterStorage::new(base_path, 0.9f64.try_into().unwrap()).unwrap();
            let results: Vec<u64> = index.filter("hello").iter().collect();
            assert_eq!(results, vec![1, 3]);

            // Add more and compact again - carried deletes should merge correctly
            index.insert(&p("hello"), 4);
            index.compact(2).unwrap();
            let results: Vec<u64> = index.filter("hello").iter().collect();
            assert_eq!(results, vec![1, 3, 4]);
        }
    }

    // ──────────────────────────────────────────────────────────────
    //  info() correctness
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_info_after_operations() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        let info = index.info();
        assert_eq!(info.current_version_number, 0);
        assert_eq!(info.unique_keys_count, 0);
        assert_eq!(info.total_postings_count, 0);

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 2);
        index.insert(&p("world"), 3);
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(info.current_version_number, 1);
        assert_eq!(info.unique_keys_count, 2);
        assert_eq!(info.total_postings_count, 3);
        assert_eq!(info.pending_ops, 0);
        assert!(info.fst_size_bytes > 0);
        assert!(info.postings_size_bytes > 0);

        index.insert(&p("hello"), 4);
        let info = index.info();
        assert_eq!(info.pending_ops, 1);
    }

    // ──────────────────────────────────────────────────────────────
    //  integrity_check()
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_integrity_check_valid() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.compact(1).unwrap();

        let result = index.integrity_check();
        assert!(
            result.passed,
            "integrity check should pass: {:?}",
            result.checks
        );
    }

    #[test]
    fn test_integrity_check_missing_files() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.compact(1).unwrap();

        // Remove a required file
        fs::remove_file(tmp.path().join("versions/1/postings.dat")).unwrap();

        let result = index.integrity_check();
        assert!(
            !result.passed,
            "integrity check should fail with missing file"
        );
    }

    // ──────────────────────────────────────────────────────────────
    //  Filter results are always sorted
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_results_sorted() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Insert in non-sequential order across compaction rounds
        index.insert(&p("key"), 10);
        index.insert(&p("key"), 3);
        index.insert(&p("key"), 7);
        index.compact(1).unwrap();

        index.insert(&p("key"), 1);
        index.insert(&p("key"), 15);
        index.insert(&p("key"), 5);

        let results: Vec<u64> = index.filter("key").iter().collect();
        let mut sorted = results.clone();
        sorted.sort();
        assert_eq!(results, sorted, "filter results must be sorted");
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact merges live and compacted same key
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_merges_live_and_compacted_same_key() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("shared"), 1);
        index.insert(&p("shared"), 3);
        index.compact(1).unwrap();

        index.insert(&p("shared"), 2);
        index.insert(&p("shared"), 4);
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter("shared").iter().collect();
        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Large scale
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_large_scale_insert_delete_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Insert 500 docs across 50 keys
        for i in 0u64..500 {
            let key = format!("key_{:03}", i % 50);
            index.insert(&p(&key), i);
        }
        index.compact(1).unwrap();

        // Delete even doc_ids
        for i in (0u64..500).step_by(2) {
            index.delete(i);
        }
        index.compact(2).unwrap();

        // Verify only odd doc_ids remain
        for key_idx in 0..50u64 {
            let key = format!("key_{key_idx:03}");
            let results: Vec<u64> = index.filter(&key).iter().collect();
            let expected: Vec<u64> = (0..500u64)
                .filter(|i| i % 50 == key_idx && i % 2 != 0)
                .collect();
            assert_eq!(results, expected, "mismatch for {key}");
        }
    }

    // ──────────────────────────────────────────────────────────────
    //  Duplicate insert same key+doc_id
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_insert_same_key_doc() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 1);
        index.insert(&p("hello"), 1);

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1], "duplicates should be deduplicated");

        index.compact(1).unwrap();
        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact empty live layer
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_empty_live_layer() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.compact(1).unwrap();

        // Compact again with no changes in live layer
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter("hello").iter().collect();
        assert_eq!(results, vec![1]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Multi-threaded reads while inserting
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_concurrent_reads_writes() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let index = Arc::new(
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap(),
        );

        // Pre-populate
        for i in 0..100u64 {
            index.insert(&p("concurrent"), i);
        }

        let index_clone = Arc::clone(&index);
        let writer = thread::spawn(move || {
            for i in 100..200u64 {
                index_clone.insert(&p("concurrent"), i);
            }
        });

        let index_clone = Arc::clone(&index);
        let reader = thread::spawn(move || {
            for _ in 0..50 {
                let results: Vec<u64> = index_clone.filter("concurrent").iter().collect();
                // Should always get at least the initial 100
                assert!(results.len() >= 100);
                // Results should always be sorted
                for w in results.windows(2) {
                    assert!(w[0] < w[1], "results not sorted");
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        let final_results: Vec<u64> = index.filter("concurrent").iter().collect();
        assert_eq!(final_results.len(), 200);
    }

    // ──────────────────────────────────────────────────────────────
    //  Delete before any compaction, then compact
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_before_first_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("a"), 1);
        index.insert(&p("a"), 2);
        index.insert(&p("b"), 3);
        index.delete(1);
        index.delete(3);

        let a: Vec<u64> = index.filter("a").iter().collect();
        let b: Vec<u64> = index.filter("b").iter().collect();
        assert_eq!(a, vec![2]);
        assert!(b.is_empty());

        index.compact(1).unwrap();
        let a: Vec<u64> = index.filter("a").iter().collect();
        let b: Vec<u64> = index.filter("b").iter().collect();
        assert_eq!(a, vec![2]);
        assert!(b.is_empty());
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact with new key not in compacted version
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_introduces_new_keys() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("first"), 1);
        index.compact(1).unwrap();

        // "second" is a brand new key
        index.insert(&p("second"), 2);
        index.compact(2).unwrap();

        let first: Vec<u64> = index.filter("first").iter().collect();
        let second: Vec<u64> = index.filter("second").iter().collect();
        assert_eq!(first, vec![1]);
        assert_eq!(second, vec![2]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Compact where a key exists only in compacted (no live data for it)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_key_only_in_compacted() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("old_key"), 1);
        index.insert(&p("old_key"), 2);
        index.compact(1).unwrap();

        // Live layer only has a different key
        index.insert(&p("new_key"), 3);
        index.compact(2).unwrap();

        let old: Vec<u64> = index.filter("old_key").iter().collect();
        let new: Vec<u64> = index.filter("new_key").iter().collect();
        assert_eq!(old, vec![1, 2]);
        assert_eq!(new, vec![3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Lexicographic key ordering in compacted version
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_keys_stored_in_lexicographic_order() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Insert in reverse lexicographic order
        index.insert(&p("zebra"), 1);
        index.insert(&p("apple"), 2);
        index.insert(&p("mango"), 3);
        index.compact(1).unwrap();

        // All should still be found
        assert_eq!(index.filter("zebra").iter().collect::<Vec<_>>(), vec![1]);
        assert_eq!(index.filter("apple").iter().collect::<Vec<_>>(), vec![2]);
        assert_eq!(index.filter("mango").iter().collect::<Vec<_>>(), vec![3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Delete doc_id shared across keys after compaction
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_shared_doc_id_across_keys_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("key_a"), 1);
        index.insert(&p("key_b"), 1);
        index.insert(&p("key_a"), 2);
        index.insert(&p("key_b"), 3);
        index.compact(1).unwrap();

        index.delete(1);
        index.compact(2).unwrap();

        let a: Vec<u64> = index.filter("key_a").iter().collect();
        let b: Vec<u64> = index.filter("key_b").iter().collect();
        assert_eq!(a, vec![2]);
        assert_eq!(b, vec![3]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Insert after compact, delete, compact again
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_compact_delete_compact() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("k"), 1);
        index.compact(1).unwrap();

        index.insert(&p("k"), 2);
        index.compact(2).unwrap();

        index.delete(1);
        index.compact(3).unwrap();

        let results: Vec<u64> = index.filter("k").iter().collect();
        assert_eq!(results, vec![2]);

        // Persistence check
        drop(index);
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();
        let results: Vec<u64> = index.filter("k").iter().collect();
        assert_eq!(results, vec![2]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Verify total_size_bytes in info
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_info_total_size() {
        let tmp = TempDir::new().unwrap();
        let index =
            StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&p("hello"), 1);
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(
            info.total_size_bytes(),
            info.fst_size_bytes + info.postings_size_bytes + info.deleted_size_bytes
        );
        assert!(info.total_size_bytes() > 0);
    }
}
