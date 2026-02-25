use super::compacted::CompactedVersion;
use super::config::Threshold;
use super::indexer::IndexedValue;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    ensure_version_dir, list_version_dirs, read_current, remove_version_dir, sync_dir, version_dir,
    write_current_atomic, FORMAT_VERSION,
};
use super::iterator::{SearchHandle, SearchParams};
use super::live::{LiveLayer, LiveSnapshot};
use super::merge::sorted_merge;
use super::scorer::BM25Scorer;
use super::{DocumentFilter, NoFilter};
use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// Persistent full-text string index with BM25 scoring, concurrent reads and writes,
/// and compaction to disk.
pub struct StringStorage {
    base_path: PathBuf,
    version: ArcSwap<CompactedVersion>,
    live: RwLock<LiveLayer>,
    compaction_lock: Mutex<()>,
    threshold: Threshold,
}

impl StringStorage {
    /// Open or create a string index at the given path.
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

    /// Insert a document's indexed terms into the index.
    pub fn insert(&self, doc_id: u64, indexed_value: IndexedValue) {
        let mut live = self.live.write().unwrap();
        live.insert(doc_id, indexed_value);
    }

    /// Mark a doc_id as deleted.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    /// Search the index for documents matching the given tokens, accumulating scores into `scorer`.
    ///
    /// Uses BM25F scoring: for each query token, computes a length-normalized term frequency
    /// (ntf) per document, then combines it with inverse document frequency (IDF) to produce
    /// a relevance score. Scores from multiple tokens are summed per document.
    ///
    /// Both the compacted (mmap) and live (in-memory) layers are searched and merged, so
    /// results reflect all inserts/deletes without requiring a compaction first.
    /// Search the index without document filtering.
    pub fn search(
        &self,
        params: &SearchParams<'_>,
        scorer: &mut BM25Scorer,
    ) -> Result<()> {
        self.search_filtered::<NoFilter>(params, None, scorer)
    }

    /// Search the index, filtering results by document ID.
    pub fn search_with_filter<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: &F,
        scorer: &mut BM25Scorer,
    ) -> Result<()> {
        self.search_filtered(params, Some(filter), scorer)
    }

    fn search_filtered<F: DocumentFilter>(
        &self,
        params: &SearchParams<'_>,
        filter: Option<&F>,
        scorer: &mut BM25Scorer,
    ) -> Result<()> {
        let snapshot = self.get_fresh_snapshot();
        let version = self.version.load();
        let handle = SearchHandle::new(Arc::clone(&version), snapshot);
        handle.execute(params, filter, scorer)
    }

    /// Get a fresh snapshot, refreshing if dirty (double-check locking pattern).
    fn get_fresh_snapshot(&self) -> Arc<LiveSnapshot> {
        {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                return live.get_snapshot();
            }
        }
        let mut live = self.live.write().unwrap();
        if live.is_snapshot_dirty() {
            live.refresh_snapshot();
        }
        live.get_snapshot()
    }

    /// Persist pending changes to disk at the given version number.
    pub fn compact(&self, version_number: u64) -> Result<()> {
        let _compaction_guard = self.compaction_lock.lock().unwrap();

        let snapshot = self.get_fresh_snapshot();

        // Nothing to compact — free memory and return early
        if snapshot.term_postings.is_empty() && snapshot.deletes.is_empty() {
            let mut live = self.live.write().unwrap();
            live.ops.drain(..snapshot.ops_len);
            if live.ops.is_empty() {
                live.ops.shrink_to_fit();
            }
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

            live.drain_compacted_ops(snapshot.ops_len);
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

        let compacted_postings = current.total_postings();
        // Estimate live postings as number of unique (term, doc_id) pairs
        let live_postings: usize = snapshot.term_postings.values().map(|v| v.len()).sum();
        let total_postings = compacted_postings + live_postings;

        if total_postings == 0 {
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
        let deleted_set: Vec<u64> = sorted_merge(
            current.deletes_slice().iter().copied(),
            snapshot.deletes.iter().copied(),
        )
        .collect();

        let mut compacted_terms = current.iter_terms();
        let live_terms: Vec<_> = snapshot.iter_terms_sorted().collect();
        let mut compacted_dl = current.iter_doc_lengths();
        let live_dl = snapshot.iter_doc_lengths_sorted();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &live_terms,
            &mut compacted_dl,
            &live_dl,
            Some(&deleted_set),
            Some(&deleted_set),
            &[],
            new_version_dir,
        )?;

        Ok(())
    }

    /// Compact by merging new data while carrying deletions forward.
    fn compact_carry_forward(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
        new_version_dir: &std::path::Path,
    ) -> Result<()> {
        let mut compacted_terms = current.iter_terms();
        let live_terms: Vec<_> = snapshot.iter_terms_sorted().collect();
        let mut compacted_dl = current.iter_doc_lengths();
        let live_dl = snapshot.iter_doc_lengths_sorted();

        let deletes_merged: Vec<u64> = sorted_merge(
            current.deletes_slice().iter().copied(),
            snapshot.deletes.iter().copied(),
        )
        .collect();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &live_terms,
            &mut compacted_dl,
            &live_dl,
            None,
            Some(&deletes_merged),
            &deletes_merged,
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

        let avg_field_length = if version.total_documents > 0 {
            version.total_document_length as f64 / version.total_documents as f64
        } else {
            0.0
        };

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: version.version_number,
            version_dir: ver_dir.clone(),
            unique_terms_count: version.term_count(),
            total_postings_count: version.total_postings(),
            total_documents: version.total_documents,
            avg_field_length,
            deleted_count: version.deletes_slice().len(),
            fst_size_bytes: file_size(&ver_dir.join("keys.fst")),
            postings_size_bytes: file_size(&ver_dir.join("postings.dat")),
            doc_lengths_size_bytes: file_size(&ver_dir.join("doc_lengths.dat")),
            deleted_size_bytes: file_size(&ver_dir.join("deleted.bin")),
            global_info_size_bytes: file_size(&ver_dir.join("global_info.bin")),
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

                let required_files = [
                    "keys.fst",
                    "postings.dat",
                    "doc_lengths.dat",
                    "deleted.bin",
                    "global_info.bin",
                ];
                let missing: Vec<&str> = required_files
                    .iter()
                    .filter(|f| !ver_dir.join(f).exists())
                    .copied()
                    .collect();

                if !missing.is_empty() {
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
    use assert_approx_eq::assert_approx_eq;
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
        let mut scorer = BM25Scorer::new();
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

    #[test]
    fn test_new_empty_index() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();
        assert_eq!(index.current_version_number(), 0);
    }

    #[test]
    fn test_insert_and_search() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

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
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

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
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

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
            let index = StringStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(1, make_value(3, vec![("hello", vec![0], vec![1])]));
            index.insert(2, make_value(2, vec![("world", vec![0], vec![])]));
            index.compact(1).unwrap();
        }

        {
            let index = StringStorage::new(base_path, Threshold::default()).unwrap();
            let result = search_default(&index, &["hello"]);
            assert_eq!(result.docs.len(), 1);
            assert_eq!(result.docs[0].doc_id, 1);

            let result = search_default(&index, &["world"]);
            assert_eq!(result.docs.len(), 1);
            assert_eq!(result.docs[0].doc_id, 2);

            assert_eq!(index.current_version_number(), 1);
        }
    }

    #[test]
    fn test_compact_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        index.delete(2);

        index.compact(1).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&3));
    }

    #[test]
    fn test_ops_during_compaction_preserved() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);
    }

    #[test]
    fn test_compact_multiple_rounds() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(2).unwrap();

        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(3).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 3);
    }

    #[test]
    fn test_compact_carries_forward_deletes() {
        let tmp = TempDir::new().unwrap();
        // High threshold so deletes are carried forward
        let index =
            StringStorage::new(tmp.path().to_path_buf(), 0.9f64.try_into().unwrap()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        index.delete(2);
        index.compact(1).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);

        // Insert more and compact again
        index.insert(4, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(2).unwrap();

        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 3);
    }

    #[test]
    fn test_cleanup_removes_old_versions() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();
        assert!(tmp.path().join("versions/1").exists());

        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(2).unwrap();

        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(3).unwrap();

        index.cleanup();

        assert!(!tmp.path().join("versions/1").exists());
        assert!(!tmp.path().join("versions/2").exists());
        assert!(tmp.path().join("versions/3").exists());
    }

    #[test]
    fn test_info_after_operations() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        let info = index.info();
        assert_eq!(info.current_version_number, 0);
        assert_eq!(info.unique_terms_count, 0);

        index.insert(1, make_value(5, vec![("hello", vec![0], vec![])]));
        index.insert(
            2,
            make_value(
                3,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(info.current_version_number, 1);
        assert_eq!(info.unique_terms_count, 2);
        assert_eq!(info.total_documents, 2);
        assert!(info.avg_field_length > 0.0);
        assert_eq!(info.pending_ops, 0);
        assert!(info.fst_size_bytes > 0);
        assert!(info.postings_size_bytes > 0);
    }

    #[test]
    fn test_integrity_check_valid() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
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
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        fs::remove_file(tmp.path().join("versions/1/postings.dat")).unwrap();

        let result = index.integrity_check();
        assert!(
            !result.passed,
            "integrity check should fail with missing file"
        );
    }

    #[test]
    fn test_open_incompatible_format_version() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
            index.compact(1).unwrap();
        }

        let current_path = base_path.join("CURRENT");
        fs::write(&current_path, "999\n1").unwrap();

        let result = StringStorage::new(base_path, Threshold::default());
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
    fn test_delete_compacted_doc_id() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.delete(2);
        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);

        index.compact(2).unwrap();
        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 2);
    }

    #[test]
    fn test_concurrent_reads_writes() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let index =
            Arc::new(StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap());

        // Pre-populate
        for i in 0..50u64 {
            index.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
        }

        let index_clone = Arc::clone(&index);
        let writer = thread::spawn(move || {
            for i in 50..100u64 {
                index_clone.insert(i, make_value(2, vec![("term", vec![0], vec![])]));
            }
        });

        let index_clone = Arc::clone(&index);
        let reader = thread::spawn(move || {
            for _ in 0..20 {
                let tokens = vec!["term".to_string()];
                let mut scorer = BM25Scorer::new();
                index_clone
                    .search(
                        &SearchParams {
                            tokens: &tokens,
                            ..Default::default()
                        },
                        &mut scorer,
                    )
                    .unwrap();
                let result = scorer.into_search_result();
                assert!(result.docs.len() >= 50);
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        let final_result = search_default(&index, &["term"]);
        assert_eq!(final_result.docs.len(), 100);
    }

    #[test]
    fn test_persistence_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        {
            let index = StringStorage::new(base_path.clone(), Threshold::default()).unwrap();
            index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
            index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
            index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
            index.delete(2);
            index.compact(1).unwrap();
        }

        {
            let index = StringStorage::new(base_path, Threshold::default()).unwrap();
            let result = search_default(&index, &["hello"]);
            assert_eq!(result.docs.len(), 2);
            let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
            assert!(doc_ids.contains(&1));
            assert!(doc_ids.contains(&3));
        }
    }

    #[test]
    fn test_compact_all_deleted() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.delete(1);
        index.delete(2);
        index.compact(1).unwrap();

        let result = search_default(&index, &["hello"]);
        assert!(result.docs.is_empty());

        // Re-insert after all deleted
        index.insert(3, make_value(2, vec![("hello", vec![0], vec![])]));
        let result = search_default(&index, &["hello"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 3);
    }

    #[test]
    fn test_search_multiple_tokens_combined() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Doc 1 matches both "hello" and "world"
        index.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        // Doc 2 matches only "hello"
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));

        index.compact(1).unwrap();

        let result = search_default(&index, &["hello", "world"]);
        assert_eq!(result.docs.len(), 2);
        // Doc 1 should rank higher (matches both tokens)
        assert_eq!(result.docs[0].doc_id, 1);
    }

    #[test]
    fn test_insert_compact_delete_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("k", vec![0], vec![])]));
        index.compact(1).unwrap();

        index.insert(2, make_value(2, vec![("k", vec![0], vec![])]));
        index.compact(2).unwrap();

        index.delete(1);
        index.compact(3).unwrap();

        let result = search_default(&index, &["k"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);

        // Persistence check
        drop(index);
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();
        let result = search_default(&index, &["k"]);
        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 2);
    }

    #[test]
    fn test_total_size_bytes() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(
            info.total_size_bytes(),
            info.fst_size_bytes
                + info.postings_size_bytes
                + info.doc_lengths_size_bytes
                + info.deleted_size_bytes
                + info.global_info_size_bytes
        );
        assert!(info.total_size_bytes() > 0);
    }

    #[test]
    fn test_prefix_search_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        index.insert(2, make_value(3, vec![("application", vec![0], vec![])]));
        index.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));
        index.compact(1).unwrap();

        let tokens = vec!["app".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let result = scorer.into_search_result();

        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));
    }

    #[test]
    fn test_levenshtein_search_before_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        index.insert(2, make_value(3, vec![("apply", vec![0], vec![])]));
        index.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        // Fuzzy search before any compaction
        let tokens = vec!["apple".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    tolerance: Some(1),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let result = scorer.into_search_result();

        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1)); // "apple" exact
        assert!(doc_ids.contains(&2)); // "apply" distance 1
    }

    #[test]
    fn test_levenshtein_parity_before_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        index.insert(2, make_value(3, vec![("apply", vec![0], vec![])]));
        index.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));

        // Search before compaction
        let tokens = vec!["apple".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    tolerance: Some(1),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let before = scorer.into_search_result();

        // Compact
        index.compact(1).unwrap();

        // Search after compaction
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    tolerance: Some(1),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let after = scorer.into_search_result();

        // Same doc IDs should be returned
        let before_ids: Vec<u64> = before.docs.iter().map(|d| d.doc_id).collect();
        let after_ids: Vec<u64> = after.docs.iter().map(|d| d.doc_id).collect();
        assert_eq!(before_ids, after_ids);
    }

    #[test]
    fn test_levenshtein_search_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(3, vec![("apple", vec![0], vec![])]));
        index.insert(2, make_value(3, vec![("apply", vec![0], vec![])]));
        index.insert(3, make_value(3, vec![("banana", vec![0], vec![])]));
        index.compact(1).unwrap();

        // Levenshtein distance 1 from "apple"
        let tokens = vec!["apple".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    tolerance: Some(1),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let result = scorer.into_search_result();

        assert_eq!(result.docs.len(), 2);
        let doc_ids: Vec<u64> = result.docs.iter().map(|d| d.doc_id).collect();
        assert!(doc_ids.contains(&1)); // "apple" exact
        assert!(doc_ids.contains(&2)); // "apply" distance 1

        // Exact match should score higher due to exact boost
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_phrase_boost_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Doc 1: adjacent tokens
        index.insert(
            1,
            make_value(
                4,
                vec![("hello", vec![0], vec![]), ("world", vec![1], vec![])],
            ),
        );
        // Doc 2: non-adjacent tokens
        index.insert(
            2,
            make_value(
                6,
                vec![("hello", vec![0], vec![]), ("world", vec![5], vec![])],
            ),
        );
        index.compact(1).unwrap();

        let tokens = vec!["hello".to_string(), "world".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    phrase_boost: Some(2.0),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let result = scorer.into_search_result();

        assert_eq!(result.docs.len(), 2);
        // Doc 1 should score higher due to phrase boost on adjacent positions
        assert_eq!(result.docs[0].doc_id, 1);
        assert!(result.docs[0].score > result.docs[1].score);
    }

    #[test]
    fn test_threshold_after_compact() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Doc 1 matches all 3 tokens
        index.insert(
            1,
            make_value(
                6,
                vec![
                    ("hello", vec![0], vec![]),
                    ("world", vec![1], vec![]),
                    ("foo", vec![2], vec![]),
                ],
            ),
        );
        // Doc 2 matches only 1 of 3 tokens
        index.insert(2, make_value(2, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        let tokens = vec!["hello".to_string(), "world".to_string(), "foo".to_string()];
        // threshold=1.0 with 3 tokens => need all 3
        let mut scorer = BM25Scorer::with_threshold(3);
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let result = scorer.into_search_result();

        assert_eq!(result.docs.len(), 1);
        assert_eq!(result.docs[0].doc_id, 1);
    }

    // ---- Scoring parity tests ----
    // These tests verify that BM25 scores from oramacore_fields match those from oramacore
    // when given identical data and parameters.
    //
    // IMPORTANT: oramacore derives per-doc field_length as max(stemmed_positions) + 1,
    // while oramacore_fields uses the explicit IndexedValue.field_length.
    // To ensure parity, all test data satisfies: field_length = max(stemmed_positions) + 1.
    //
    // Run corresponding oramacore tests:
    //   cd /path/to/oramacore && cargo test test_scoring_parity -- --nocapture

    #[test]
    fn test_scoring_parity_basic_bm25() {
        // Test 1: Single token, two docs
        // Doc1: field_length=5, "term1" exact=[0,1] stemmed=[0,4]
        //   → tf(exact_match=false) = 2+2 = 4, oramacore doc_length = max(4)+1 = 5
        // Doc2: field_length=3, "term1" exact=[0] stemmed=[2]
        //   → tf = 1+1 = 2, oramacore doc_length = max(2)+1 = 3
        // tolerance=None (prefix search) → 3x exact_match_boost applied
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(5, vec![("term1", vec![0, 1], vec![0, 4])]));
        index.insert(2, make_value(3, vec![("term1", vec![0], vec![2])]));
        index.compact(1).unwrap();

        let tokens = vec!["term1".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: false,
                    boost: 1.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];
        let score2 = scores[&2];

        println!("test_scoring_parity_basic_bm25:");
        println!("  doc1 score = {score1:.10}");
        println!("  doc2 score = {score2:.10}");

        assert!(
            score1 > score2,
            "Doc1 (tf=4) should rank higher than Doc2 (tf=2)"
        );
        // Cross-repo parity: these values must match oramacore's test_scoring_parity_basic_bm25
        assert_approx_eq!(score1, 0.358_531_8, 1e-6);
        assert_approx_eq!(score2, 0.34503868, 1e-6);
    }

    #[test]
    fn test_scoring_parity_exact_match() {
        // Test 2: exact_match=true with tolerance=Some(0)
        // Same data as Test 1
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(5, vec![("term1", vec![0, 1], vec![0, 4])]));
        index.insert(2, make_value(3, vec![("term1", vec![0], vec![2])]));
        index.compact(1).unwrap();

        let tokens = vec!["term1".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: true,
                    boost: 1.0,
                    tolerance: Some(0),
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];
        let score2 = scores[&2];

        println!("test_scoring_parity_exact_match:");
        println!("  doc1 score = {score1:.10}");
        println!("  doc2 score = {score2:.10}");

        // Doc1: tf=2 (exact only), Doc2: tf=1 (exact only), with 3x exact_match_boost
        assert!(score1 > score2);
        // Cross-repo parity: these values must match oramacore's test_scoring_parity_exact_match
        assert_approx_eq!(score1, 0.32412723, 1e-6);
        assert_approx_eq!(score2, 0.302_722_6, 1e-6);
    }

    #[test]
    fn test_scoring_parity_field_boost() {
        // Test 3: Same data as Test 1, but boost=2.0
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(5, vec![("term1", vec![0, 1], vec![0, 4])]));
        index.insert(2, make_value(3, vec![("term1", vec![0], vec![2])]));
        index.compact(1).unwrap();

        let tokens = vec!["term1".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: false,
                    boost: 2.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];
        let score2 = scores[&2];

        println!("test_scoring_parity_field_boost:");
        println!("  doc1 score = {score1:.10}");
        println!("  doc2 score = {score2:.10}");

        assert!(score1 > score2);
        // Cross-repo parity: these values must match oramacore's test_scoring_parity_field_boost
        assert_approx_eq!(score1, 0.37862647, 1e-6);
        assert_approx_eq!(score2, 0.37096643, 1e-6);
    }

    #[test]
    fn test_scoring_parity_single_doc() {
        // Test 4: Single doc, single token (simplest case)
        // Doc1: field_length=3, "hello" exact=[0] stemmed=[2]
        //   → tf = 1+1 = 2, oramacore doc_length = max(2)+1 = 3
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(3, vec![("hello", vec![0], vec![2])]));
        index.compact(1).unwrap();

        let tokens = vec!["hello".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: false,
                    boost: 1.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];

        println!("test_scoring_parity_single_doc:");
        println!("  doc1 score = {score1:.10}");

        // Cross-repo parity: must match oramacore's test_scoring_parity_single_doc
        assert_approx_eq!(score1, 0.527_417_2, 1e-6);
    }

    #[test]
    fn test_scoring_parity_stemmed_only() {
        // Test 5: Stemmed-only positions
        // Doc1: field_length=4, "run" exact=[] stemmed=[0,1,3]
        //   → tf(exact_match=false) = 3, oramacore doc_length = max(3)+1 = 4
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(4, vec![("run", vec![], vec![0, 1, 3])]));
        index.compact(1).unwrap();

        // exact_match=false → tf=3 (stemmed only)
        let tokens = vec!["run".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: false,
                    boost: 1.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];
        println!("test_scoring_parity_stemmed_only (exact_match=false):");
        println!("  doc1 score = {score1:.10}");
        // Cross-repo parity: must match oramacore's test_scoring_parity_stemmed_only
        assert_approx_eq!(score1, 0.558_441_7, 1e-6);

        // exact_match=true → tf=0, no results
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: true,
                    boost: 1.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        println!("test_scoring_parity_stemmed_only (exact_match=true):");
        println!("  scores = {scores:?}");

        assert!(
            scores.is_empty(),
            "exact_match=true with stemmed-only should yield no results"
        );
    }

    #[test]
    fn test_scoring_parity_varying_lengths() {
        // Test 6: Multiple documents with varying field lengths
        // Doc1: field_length=10, "word" exact=[0,1,2] stemmed=[9]
        //   → tf = 3+1 = 4, oramacore doc_length = max(9)+1 = 10
        // Doc2: field_length=2,  "word" exact=[0] stemmed=[1]
        //   → tf = 1+1 = 2, oramacore doc_length = max(1)+1 = 2
        // Doc3: field_length=20, "word" exact=[0,1] stemmed=[0,19]
        //   → tf = 2+2 = 4, oramacore doc_length = max(19)+1 = 20
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(1, make_value(10, vec![("word", vec![0, 1, 2], vec![9])]));
        index.insert(2, make_value(2, vec![("word", vec![0], vec![1])]));
        index.insert(3, make_value(20, vec![("word", vec![0, 1], vec![0, 19])]));
        index.compact(1).unwrap();

        let tokens = vec!["word".to_string()];
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &SearchParams {
                    tokens: &tokens,
                    exact_match: false,
                    boost: 1.0,
                    tolerance: None,
                    ..Default::default()
                },
                &mut scorer,
            )
            .unwrap();
        let scores = scorer.get_scores();

        let score1 = scores[&1];
        let score2 = scores[&2];
        let score3 = scores[&3];

        println!("test_scoring_parity_varying_lengths:");
        println!("  doc1 (fl=10, tf=4) score = {score1:.10}");
        println!("  doc2 (fl=2,  tf=2) score = {score2:.10}");
        println!("  doc3 (fl=20, tf=4) score = {score3:.10}");

        // Cross-repo parity: must match oramacore's test_scoring_parity_varying_lengths
        assert_approx_eq!(score1, 0.268_205_7, 1e-6);
        assert_approx_eq!(score2, 0.27248147, 1e-6);
        assert_approx_eq!(score3, 0.252_027_1, 1e-6);
    }

    #[test]
    fn test_score_consistency_after_live_delete_of_compacted_doc() {
        let tmp = TempDir::new().unwrap();
        let index = StringStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

        // Insert two docs with different field lengths so avg_field_length
        // changes noticeably when one is removed.
        index.insert(1, make_value(5, vec![("hello", vec![0], vec![])]));
        index.insert(2, make_value(15, vec![("hello", vec![0], vec![])]));
        index.compact(1).unwrap();

        // Delete doc 2 in the live layer (no compaction yet).
        // Logical state: only doc 1 is alive.
        index.delete(2);
        let score_before_compact = {
            let result = search_default(&index, &["hello"]);
            assert_eq!(result.docs.len(), 1, "only doc 1 should be returned");
            assert_eq!(result.docs[0].doc_id, 1);
            result.docs[0].score
        };

        // Compact to materialize the delete.
        // Logical state is still: only doc 1 is alive.
        index.compact(2).unwrap();
        let score_after_compact = {
            let result = search_default(&index, &["hello"]);
            assert_eq!(result.docs.len(), 1, "only doc 1 should be returned");
            assert_eq!(result.docs[0].doc_id, 1);
            result.docs[0].score
        };

        // IDF is now correct (total_documents accounts for compacted-doc deletes),
        // but avg_field_length is still slightly off because the live layer doesn't know
        // the deleted doc's field length (stored in the compacted mmap). This causes a
        // score difference proportional to the field-length variance. The gap is fully
        // resolved on compaction. We use a wider tolerance to document this known trade-off.
        assert_approx_eq!(score_before_compact, score_after_compact, 0.1);
    }
}
