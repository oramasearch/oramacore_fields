//! Main NumberStorage implementation.

use super::compacted::{
    copy_bucket_file, copy_header_file, version_dir as compacted_version_dir, write_deleted_file,
    write_header_file, write_meta_file, write_single_bucket, write_version_with_config,
    CompactedVersion, CompactionMeta, HeaderEntry, DEFAULT_BUCKET_TARGET_BYTES,
    DEFAULT_INDEX_STRIDE,
};
use super::config::Threshold;
use super::error::Error;
use super::indexer::IndexedValue;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{ensure_version_dir, read_current, sync_dir, write_current_atomic};
use super::iterator::{FilterHandle, FilterOp, SortHandle, SortOrder};
use super::key::IndexableNumber;
use super::live::{LiveLayer, LiveSnapshot};
use super::merge::{sorted_merge, sorted_merge_doc_ids};
use arc_swap::ArcSwap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// Default dirty header threshold for deciding when to do a full rebuild during compaction.
const DEFAULT_DIRTY_HEADER_THRESHOLD: f64 = 0.1;

/// Type alias for a NumberStorage over u64 values.
pub type U64Storage = NumberStorage<u64>;

/// Type alias for a NumberStorage over f64 values.
pub type F64Storage = NumberStorage<f64>;

/// A thread-safe, persistent number index supporting range queries.
///
/// Supports inserting, deleting, filtering, and sorting documents by numeric
/// value. Data is persisted to disk through compaction.
///
/// # Example
///
/// Use [`NumberIndexer`](super::NumberIndexer) to extract values from JSON, then
/// call [`insert`](Self::insert) to index them:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use oramacore_fields::number::{NumberIndexer, NumberStorage, FilterOp, Threshold};
/// use serde_json::json;
///
/// let dir = tempfile::tempdir()?;
/// let index: NumberStorage<u64> = NumberStorage::new(
///     dir.path().to_path_buf(),
///     Threshold::default(),
/// )?;
/// let indexer = NumberIndexer::<u64>::new(false);
///
/// // Index JSON values
/// let val = indexer.index_json(&json!(10)).unwrap();
/// index.insert(&val, 1)?;
/// let val = indexer.index_json(&json!(20)).unwrap();
/// index.insert(&val, 2)?;
/// let val = indexer.index_json(&json!(30)).unwrap();
/// index.insert(&val, 3)?;
///
/// // Query for values >= 15
/// let results: Vec<u64> = index.filter(FilterOp::Gte(15)).iter().collect();
/// assert_eq!(results, vec![2, 3]);
///
/// // Delete a document
/// index.delete(2);
///
/// // Compact the index
/// index.compact(1)?;
/// # Ok(())
/// # }
/// ```
pub struct NumberStorage<T: IndexableNumber> {
    base_path: PathBuf,
    threshold: Threshold,
    index_stride: u32,
    bucket_target_bytes: usize,
    dirty_header_threshold: f64,
    live: RwLock<LiveLayer<T>>,
    version: ArcSwap<CompactedVersion<T>>,
    compaction_lock: Mutex<()>,
}

impl<T: IndexableNumber> NumberStorage<T> {
    /// Create a new NumberStorage at the given path.
    ///
    /// If the directory doesn't exist, it will be created.
    pub fn new(base_path: PathBuf, threshold: Threshold) -> Result<Self, Error> {
        Self::new_with_config(
            base_path,
            threshold,
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
    }

    /// Create a new NumberStorage with configurable index stride and bucket size.
    ///
    /// - `index_stride`: Controls how often header entries are created (every N doc_ids).
    /// - `bucket_target_bytes`: Controls the maximum size of each data bucket file.
    pub fn new_with_config(
        base_path: PathBuf,
        threshold: Threshold,
        index_stride: u32,
        bucket_target_bytes: usize,
    ) -> Result<Self, Error> {
        // Create directory structure
        fs::create_dir_all(&base_path)?;

        // Check if CURRENT exists (resuming existing index)
        let version = if let Some((format_version, offset)) = read_current(&base_path)? {
            if format_version != super::io::FORMAT_VERSION {
                return Err(Error::UnsupportedVersion {
                    version: format_version,
                });
            }
            CompactedVersion::load(&base_path, offset)?
        } else {
            // New index - create initial version
            let version_dir = ensure_version_dir(&base_path, 0)?;
            write_version_with_config::<T>(
                &version_dir,
                std::iter::empty(),
                &[],
                index_stride,
                bucket_target_bytes,
            )?;
            sync_dir(&version_dir)?;
            write_current_atomic(&base_path, 0)?;
            CompactedVersion::load(&base_path, 0)?
        };

        Ok(Self {
            base_path,
            threshold,
            index_stride,
            bucket_target_bytes,
            dirty_header_threshold: DEFAULT_DIRTY_HEADER_THRESHOLD,
            live: RwLock::new(LiveLayer::new()),
            version: ArcSwap::from_pointee(version),
            compaction_lock: Mutex::new(()),
        })
    }

    /// Set the dirty header threshold for incremental compaction.
    ///
    /// When the ratio of changes to total entries exceeds this threshold,
    /// a full rebuild is triggered instead of incremental compaction.
    pub fn set_dirty_header_threshold(&mut self, threshold: f64) {
        self.dirty_header_threshold = threshold;
    }

    /// Insert a doc_id based on an IndexedValue.
    ///
    /// - `Plain(value)`: inserts a single entry for the doc_id
    /// - `Array(values)`: inserts one entry per value for the same doc_id
    pub fn insert(&self, indexed_value: &IndexedValue<T>, doc_id: u64) -> Result<(), Error> {
        let mut live = self.live.write().unwrap();
        match indexed_value {
            IndexedValue::Plain(v) => live.insert(*v, doc_id),
            IndexedValue::Array(values) => {
                for v in values {
                    live.insert(*v, doc_id)?;
                }
                Ok(())
            }
        }
    }

    /// Delete a document from the index.
    ///
    /// The deletion is recorded but not immediately applied. Call `compact()`
    /// to apply deletions to the compacted version.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    /// Query the index with a filter operation.
    ///
    /// Returns a `FilterHandle` that can be iterated to get matching document IDs.
    pub fn filter(&self, op: FilterOp<T>) -> FilterHandle<T> {
        // Fast path: try with read lock
        {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                let snapshot = live.get_snapshot();
                let version = self.version.load();
                return FilterHandle::new(Arc::clone(&version), snapshot, op);
            }
        }

        // Slow path: need write lock to refresh snapshot
        let mut live = self.live.write().unwrap();

        // Double-check: another thread may have refreshed while we waited
        if live.is_snapshot_dirty() {
            live.refresh_snapshot();
        }

        let snapshot = live.get_snapshot();
        let version = self.version.load();
        FilterHandle::new(Arc::clone(&version), snapshot, op)
    }

    /// Return an iterator over all doc_ids sorted by their associated values.
    ///
    /// # Arguments
    /// * `order` - The sort direction (Ascending or Descending)
    ///
    /// # Returns
    /// A `SortHandle` that can be iterated to get doc_ids in sorted order.
    ///
    /// # Example
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oramacore_fields::number::{NumberIndexer, NumberStorage, Threshold, SortOrder};
    /// use serde_json::json;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let index: NumberStorage<u64> = NumberStorage::new(
    ///     dir.path().to_path_buf(),
    ///     Threshold::default(),
    /// )?;
    /// let indexer = NumberIndexer::<u64>::new(false);
    ///
    /// for (value, doc_id) in [(30, 3), (10, 1), (20, 2)] {
    ///     let val = indexer.index_json(&json!(value)).unwrap();
    ///     index.insert(&val, doc_id)?;
    /// }
    ///
    /// // Ascending order (smallest values first)
    /// let ascending: Vec<u64> = index.sort(SortOrder::Ascending).iter().collect();
    /// assert_eq!(ascending, vec![1, 2, 3]); // doc_ids ordered by value: 10, 20, 30
    ///
    /// // Descending order (largest values first)
    /// let descending: Vec<u64> = index.sort(SortOrder::Descending).iter().collect();
    /// assert_eq!(descending, vec![3, 2, 1]); // doc_ids ordered by value: 30, 20, 10
    /// # Ok(())
    /// # }
    /// ```
    pub fn sort(&self, order: SortOrder) -> SortHandle<T> {
        // Fast path: try with read lock
        {
            let live = self.live.read().unwrap();
            if !live.is_snapshot_dirty() {
                let snapshot = live.get_snapshot();
                let version = self.version.load();
                return SortHandle::new(Arc::clone(&version), snapshot, order);
            }
        }

        // Slow path: need write lock to refresh snapshot
        let mut live = self.live.write().unwrap();

        // Double-check: another thread may have refreshed while we waited
        if live.is_snapshot_dirty() {
            live.refresh_snapshot();
        }

        let snapshot = live.get_snapshot();
        let version = self.version.load();
        SortHandle::new(Arc::clone(&version), snapshot, order)
    }

    /// Get the current version offset.
    pub fn current_offset(&self) -> u64 {
        self.version.load().offset
    }

    /// Compact the index by merging pending changes into a new on-disk version.
    ///
    /// The threshold controls whether deletions are applied immediately
    /// or carried forward to the next compaction.
    pub fn compact(&self, offset: u64) -> Result<(), Error> {
        // Step 1: Acquire compaction lock
        let compaction_guard = self.compaction_lock.lock().unwrap();

        // Step 2: Take snapshot (double-check locking to avoid blocking readers)
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
        if snapshot.inserts.is_empty() && snapshot.deletes.is_empty() {
            let mut live = self.live.write().unwrap();
            live.ops.drain(..snapshot.ops_len);
            live.ops.shrink_to_fit();
            live.refresh_snapshot();
            return Ok(());
        }

        // Step 3: Load current version (lock-free)
        let old_version = self.version.load();

        if offset == old_version.offset && old_version.has_data() {
            return Err(Error::VersionConflict { version: offset });
        }

        // Step 4: Build new version (no locks held during I/O!)
        let version_dir = ensure_version_dir(&self.base_path, offset)?;
        self.build_new_version(&version_dir, &old_version, &snapshot)?;

        // Step 5: Sync and atomically update CURRENT
        sync_dir(&version_dir)?;
        write_current_atomic(&self.base_path, offset)?;

        // Step 6: Swap version and clear live layer
        {
            let mut live = self.live.write().unwrap();
            let new_version = CompactedVersion::load(&self.base_path, offset)?;
            self.version.store(Arc::new(new_version));
            // Remove compacted ops by position, not by value.
            // Vec::push is append-only: items at indices 0..ops_len were present
            // when the snapshot was taken, while items at indices ops_len.. were
            // inserted concurrently during the I/O phase (when the write lock was
            // released). Using drain(..len) instead of value-based retain ensures
            // that concurrent inserts of the same doc_id are preserved.
            live.ops.drain(..snapshot.ops_len);
            live.refresh_snapshot();
        }

        drop(compaction_guard);

        Ok(())
    }

    /// Build a new compacted version by merging old version with live snapshot.
    fn build_new_version(
        &self,
        version_dir: &std::path::Path,
        old_version: &CompactedVersion<T>,
        snapshot: &LiveSnapshot<T>,
    ) -> Result<(), Error> {
        // 1. Merge deleted doc_ids (cheap)
        let compacted_deleted = old_version.deleted_slice();
        let live_deleted: Vec<u64> = snapshot.deletes.iter().copied().collect();
        let mut live_deletes_sorted = live_deleted;
        live_deletes_sorted.sort_unstable();
        let merged_deleted: Vec<u64> = sorted_merge_doc_ids(
            compacted_deleted.into_iter(),
            live_deletes_sorted.into_iter(),
        )
        .collect();

        // 2. Compute effective deleted: exclude doc_ids that were re-inserted in live layer
        let live_insert_doc_ids: HashSet<u64> = snapshot.inserts.iter().map(|(_, d)| *d).collect();
        let live_insert_pairs: HashSet<([u8; 8], u64)> = snapshot
            .inserts
            .iter()
            .map(|(v, d)| (v.to_bytes(), *d))
            .collect();
        let effective_deleted: Vec<u64> = merged_deleted
            .iter()
            .filter(|doc_id| !live_insert_doc_ids.contains(doc_id))
            .copied()
            .collect();

        // 3. Read old metadata and compute dirty ratio
        let old_meta = &old_version.meta;
        let new_inserts_count = snapshot.inserts.len() as u64;
        let new_changes = old_meta.changes_since_rebuild + new_inserts_count;
        let approx_total = old_meta.total_at_rebuild + new_changes;

        let dirty_ratio = if approx_total > 0 {
            new_changes as f64 / approx_total as f64
        } else {
            0.0
        };
        let needs_full_rebuild = dirty_ratio >= self.dirty_header_threshold;

        // 4. Approximate delete ratio for threshold check
        let should_apply_deletes = if approx_total > 0 {
            merged_deleted.len() as f64 / approx_total as f64 >= self.threshold.value()
        } else {
            false
        };

        // 5. Deletes-only fast path: no inserts, just carry forward
        if snapshot.inserts.is_empty() && !should_apply_deletes && old_version.data_file_count() > 0
        {
            return self.build_new_version_deletes_only(
                version_dir,
                old_version,
                &merged_deleted,
                old_meta,
            );
        }

        // 6. Try incremental path
        if !needs_full_rebuild
            && !should_apply_deletes
            && !snapshot.inserts.is_empty()
            && old_version.data_file_count() > 0
        {
            let min_key = snapshot.inserts.first().unwrap().0;
            let max_key = snapshot.inserts.last().unwrap().0;

            if let Some((first, last)) = old_version.find_affected_bucket_range(min_key, max_key) {
                return self.build_new_version_incremental(
                    version_dir,
                    old_version,
                    snapshot,
                    &effective_deleted,
                    first,
                    last,
                    new_changes,
                    &live_insert_pairs,
                );
            }
        }

        // 7. Full rewrite (fallback)
        self.build_new_version_full(
            version_dir,
            old_version,
            snapshot,
            &effective_deleted,
            &live_insert_pairs,
            should_apply_deletes,
        )
    }

    /// Deletes-only fast path: copy all data files + header, write only deleted.bin and meta.bin.
    fn build_new_version_deletes_only(
        &self,
        version_dir: &std::path::Path,
        old_version: &CompactedVersion<T>,
        merged_deleted: &[u64],
        old_meta: &CompactionMeta,
    ) -> Result<(), Error> {
        let old_version_dir = compacted_version_dir(&self.base_path, old_version.offset);

        // Copy all bucket files
        for i in 0..old_version.data_file_count() {
            copy_bucket_file(&old_version_dir, i, version_dir, i)?;
        }

        // Copy header
        copy_header_file(&old_version_dir, version_dir)?;

        // Write deleted.bin
        write_deleted_file(version_dir, merged_deleted)?;

        // Compute how many new deletes were added in this compaction
        let old_deleted_count = old_version.deleted_count() as u64;
        let new_deletes_count = (merged_deleted.len() as u64).saturating_sub(old_deleted_count);

        // Write meta.bin (adjust total_at_rebuild to reflect newly deleted entries)
        write_meta_file(
            version_dir,
            &CompactionMeta {
                changes_since_rebuild: old_meta.changes_since_rebuild,
                total_at_rebuild: old_meta.total_at_rebuild.saturating_sub(new_deletes_count),
            },
        )?;

        Ok(())
    }

    /// Incremental compaction: copy unchanged buckets, rewrite only affected ones.
    #[allow(clippy::too_many_arguments)]
    fn build_new_version_incremental(
        &self,
        version_dir: &std::path::Path,
        old_version: &CompactedVersion<T>,
        snapshot: &LiveSnapshot<T>,
        merged_deleted: &[u64],
        first_affected: usize,
        last_affected: usize,
        changes_since_rebuild: u64,
        live_insert_pairs: &HashSet<([u8; 8], u64)>,
    ) -> Result<(), Error> {
        let old_version_dir = compacted_version_dir(&self.base_path, old_version.offset);
        let total_buckets = old_version.data_file_count();

        // A. Copy unchanged bucket files
        for i in 0..total_buckets {
            if i >= first_affected && i <= last_affected {
                continue;
            }
            copy_bucket_file(&old_version_dir, i, version_dir, i)?;
        }

        // B. Rewrite affected bucket(s)
        let mut new_affected_headers: Vec<HeaderEntry<T>> = Vec::new();

        for bucket_idx in first_affected..=last_affected {
            let old_entries = old_version.iter_bucket_range(bucket_idx, bucket_idx);

            // Filter compacted entries: remove exact (value, doc_id) duplicates from live,
            // and entries whose doc_id was deleted in the live layer.
            let old_entries_vec: Vec<(T, u64)> = old_entries
                .filter(|(value, doc_id)| {
                    !live_insert_pairs.contains(&(value.to_bytes(), *doc_id))
                        && !snapshot.deletes.contains(doc_id)
                })
                .collect();

            // Determine the bucket's key range from its entries
            let (bucket_min, bucket_max) = if old_entries_vec.is_empty() {
                // Empty bucket: use the full live insert range
                (
                    snapshot.inserts.first().unwrap().0,
                    snapshot.inserts.last().unwrap().0,
                )
            } else {
                (
                    old_entries_vec.first().unwrap().0,
                    old_entries_vec.last().unwrap().0,
                )
            };

            // Binary search to find the relevant slice of live inserts for this bucket.
            // For the first affected bucket, include all inserts up to bucket_max (no lower bound).
            // For the last affected bucket, include all remaining inserts (no upper bound).
            let live_start = if bucket_idx == first_affected {
                0
            } else {
                snapshot.inserts.partition_point(|&(k, _)| {
                    T::compare(k, bucket_min) == std::cmp::Ordering::Less
                })
            };

            let live_end = if bucket_idx == last_affected {
                snapshot.inserts.len()
            } else {
                snapshot.inserts.partition_point(|&(k, _)| {
                    T::compare(k, bucket_max) != std::cmp::Ordering::Greater
                })
            };

            let relevant_live = &snapshot.inserts[live_start..live_end];

            // Merge old bucket entries with relevant live inserts
            let merged = sorted_merge(old_entries_vec.into_iter(), relevant_live.iter().copied());

            let bucket_headers =
                write_single_bucket(version_dir, merged, self.index_stride, bucket_idx as u64)?;
            new_affected_headers.extend(bucket_headers);
        }

        // C. Partial header update
        let old_headers = old_version.header_entries();
        let mut final_headers: Vec<HeaderEntry<T>> = Vec::new();

        // Keep entries for non-affected buckets
        for e in &old_headers {
            let bucket_idx = e.bucket_index as usize;
            if bucket_idx < first_affected || bucket_idx > last_affected {
                final_headers.push(*e);
            }
        }
        // Add new entries for rewritten bucket(s)
        final_headers.extend(new_affected_headers);
        // Sort by key to maintain header ordering
        final_headers.sort_by(|a, b| T::compare(a.key, b.key));

        write_header_file::<T>(version_dir, &final_headers)?;
        write_deleted_file(version_dir, merged_deleted)?;

        // D. Write metadata
        write_meta_file(
            version_dir,
            &CompactionMeta {
                changes_since_rebuild,
                total_at_rebuild: old_version.meta.total_at_rebuild,
            },
        )?;

        Ok(())
    }

    /// Full rewrite: merge all entries, optionally apply deletes. Resets metadata.
    fn build_new_version_full(
        &self,
        version_dir: &std::path::Path,
        old_version: &CompactedVersion<T>,
        snapshot: &LiveSnapshot<T>,
        merged_deleted: &[u64],
        live_insert_pairs: &HashSet<([u8; 8], u64)>,
        should_apply_deletes: bool,
    ) -> Result<(), Error> {
        // Filter compacted entries: remove exact (value, doc_id) duplicates from live,
        // and entries whose doc_id was deleted in the live layer.
        let compacted_entries = old_version.iter().filter(|(value, doc_id)| {
            !live_insert_pairs.contains(&(value.to_bytes(), *doc_id))
                && !snapshot.deletes.contains(doc_id)
        });
        let live_entries = snapshot.inserts.iter().copied();
        let merged_entries = sorted_merge(compacted_entries, live_entries);

        let actual_entry_count = if should_apply_deletes {
            let delete_set: HashSet<u64> = merged_deleted.iter().copied().collect();
            write_version_with_config::<T>(
                version_dir,
                merged_entries.filter(move |(_, doc_id)| !delete_set.contains(doc_id)),
                &[],
                self.index_stride,
                self.bucket_target_bytes,
            )?
        } else {
            write_version_with_config::<T>(
                version_dir,
                merged_entries,
                merged_deleted,
                self.index_stride,
                self.bucket_target_bytes,
            )?
        };

        // Full rebuild resets metadata
        write_meta_file(
            version_dir,
            &CompactionMeta {
                changes_since_rebuild: 0,
                total_at_rebuild: actual_entry_count,
            },
        )?;

        Ok(())
    }

    /// Get metadata and statistics about the index.
    pub fn info(&self) -> IndexInfo {
        let version = self.version.load();
        let stats = version.stats();
        let live = self.live.read().unwrap();

        IndexInfo {
            format_version: super::io::FORMAT_VERSION,
            current_offset: version.offset,
            version_dir: self
                .base_path
                .join("versions")
                .join(version.offset.to_string()),
            header_entry_count: version.header_entry_count(),
            deleted_count: version.deleted_count(),
            data_file_count: stats.data_file_count,
            header_size_bytes: stats.header_size_bytes,
            deleted_size_bytes: stats.deleted_size_bytes,
            data_total_bytes: stats.data_total_bytes,
            pending_inserts: live.inserts_len(),
            pending_deletes: live.deletes_len(),
        }
    }

    /// Check the integrity of the index files.
    pub fn integrity_check(&self) -> IntegrityCheckResult {
        let mut checks = Vec::new();

        // Check CURRENT file
        let current_path = self.base_path.join("CURRENT");
        if !current_path.exists() {
            checks.push(IntegrityCheck::failed(
                "CURRENT",
                Some("File does not exist".to_string()),
            ));
            return IntegrityCheckResult::new(checks);
        }

        match read_current(&self.base_path) {
            Ok(Some((format_version, offset))) => {
                checks.push(IntegrityCheck::ok(
                    "CURRENT",
                    Some(format!("version: {format_version}, offset: {offset}")),
                ));

                // Check format version
                if format_version != super::io::FORMAT_VERSION {
                    checks.push(IntegrityCheck::failed(
                        "format version",
                        Some(format!(
                            "Expected {}, found {format_version}",
                            super::io::FORMAT_VERSION
                        )),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "format version",
                    Some(format!("{}", super::io::FORMAT_VERSION)),
                ));

                // Check version directory
                let version_dir = self.base_path.join("versions").join(offset.to_string());
                if !version_dir.exists() {
                    checks.push(IntegrityCheck::failed(
                        "version directory",
                        Some(format!("Does not exist: {}", version_dir.display())),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "version directory",
                    Some(version_dir.display().to_string()),
                ));

                // Check header.idx
                let header_path = version_dir.join("header.idx");
                if !header_path.exists() {
                    checks.push(IntegrityCheck::failed(
                        "header.idx",
                        Some("File does not exist".to_string()),
                    ));
                } else if let Ok(metadata) = fs::metadata(&header_path) {
                    let size = metadata.len();
                    // Header entry is 24 bytes (8 bytes key + 8 bytes bucket_index + 8 bytes bucket_offset)
                    if !size.is_multiple_of(24) {
                        checks.push(IntegrityCheck::failed(
                            "header.idx",
                            Some(format!("{size} bytes (not divisible by 24)")),
                        ));
                    } else {
                        let entries = size / 24;
                        checks.push(IntegrityCheck::ok(
                            "header.idx",
                            Some(format!("{size} bytes ({entries} entries)")),
                        ));
                    }
                } else {
                    checks.push(IntegrityCheck::failed(
                        "header.idx",
                        Some("Cannot read file metadata".to_string()),
                    ));
                }

                // Check deleted.bin
                let deleted_path = version_dir.join("deleted.bin");
                if !deleted_path.exists() {
                    checks.push(IntegrityCheck::failed(
                        "deleted.bin",
                        Some("File does not exist".to_string()),
                    ));
                } else if let Ok(metadata) = fs::metadata(&deleted_path) {
                    let size = metadata.len();
                    if !size.is_multiple_of(8) {
                        checks.push(IntegrityCheck::failed(
                            "deleted.bin",
                            Some(format!("{size} bytes (not divisible by 8)")),
                        ));
                    } else {
                        let count = size / 8;
                        checks.push(IntegrityCheck::ok(
                            "deleted.bin",
                            Some(format!("{size} bytes ({count} entries)")),
                        ));
                    }
                } else {
                    checks.push(IntegrityCheck::failed(
                        "deleted.bin",
                        Some("Cannot read file metadata".to_string()),
                    ));
                }

                // Check data files
                let mut data_file_count = 0;
                let mut total_data_bytes: u64 = 0;
                for i in 0u32.. {
                    let data_path = version_dir.join(format!("data_{i:04}.dat"));
                    if !data_path.exists() {
                        break;
                    }
                    if let Ok(metadata) = fs::metadata(&data_path) {
                        total_data_bytes += metadata.len();
                        data_file_count += 1;
                    }
                }

                if data_file_count == 0 {
                    checks.push(IntegrityCheck::ok(
                        "data files",
                        Some("No data files (empty index)".to_string()),
                    ));
                } else {
                    checks.push(IntegrityCheck::ok(
                        "data files",
                        Some(format!(
                            "{data_file_count} file(s), total: {total_data_bytes} bytes"
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

    /// Clean up old version directories.
    ///
    /// Removes all version directories except the current one.
    pub fn cleanup(&self) {
        let current_offset = self.current_offset();
        let versions_dir = self.base_path.join("versions");

        if let Ok(entries) = fs::read_dir(&versions_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Ok(offset) = name.parse::<u64>() {
                        if offset != current_offset {
                            let _ = fs::remove_dir_all(entry.path());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::info::CheckStatus;
    use super::FilterOp;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_index() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        assert_eq!(index.current_offset(), 0);
    }

    #[test]
    fn test_insert_and_filter() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.insert(&IndexedValue::Plain(30), 3).unwrap();

        // Filter eq
        let results: Vec<u64> = index.filter(FilterOp::Eq(20)).iter().collect();
        assert_eq!(results, vec![2]);

        // Filter gte
        let results: Vec<u64> = index.filter(FilterOp::Gte(20)).iter().collect();
        assert_eq!(results, vec![2, 3]);

        // Filter lte
        let results: Vec<u64> = index.filter(FilterOp::Lte(20)).iter().collect();
        assert_eq!(results, vec![1, 2]);

        // Filter between
        let results: Vec<u64> = index
            .filter(FilterOp::BetweenInclusive(15, 25))
            .iter()
            .collect();
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn test_delete() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.insert(&IndexedValue::Plain(30), 3).unwrap();

        // Delete doc 2
        index.delete(2);

        // Should not appear in results
        let results: Vec<u64> = index.filter(FilterOp::Gte(10)).iter().collect();
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_compact() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.insert(&IndexedValue::Plain(30), 3).unwrap();

        // Compact
        index.compact(1).unwrap();
        assert_eq!(index.current_offset(), 1);

        // Data should still be queryable
        let results: Vec<u64> = index.filter(FilterOp::Gte(10)).iter().collect();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_compact_with_deletes() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.insert(&IndexedValue::Plain(30), 3).unwrap();
        index.delete(2);

        // Compact
        index.compact(1).unwrap();

        // Deleted doc should not appear
        let results: Vec<u64> = index.filter(FilterOp::Gte(10)).iter().collect();
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_open_existing() {
        let temp = TempDir::new().unwrap();

        // Create and populate index
        {
            let index: NumberStorage<u64> =
                NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();
            index.insert(&IndexedValue::Plain(10), 1).unwrap();
            index.insert(&IndexedValue::Plain(20), 2).unwrap();
            index.compact(1).unwrap();
        }

        // Reopen
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        let results: Vec<u64> = index.filter(FilterOp::Gte(10)).iter().collect();
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn test_f64_index() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<f64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(1.5), 1).unwrap();
        index.insert(&IndexedValue::Plain(-0.5), 2).unwrap();
        index
            .insert(&IndexedValue::Plain(std::f64::consts::PI), 3)
            .unwrap();

        // Filter gte 0.0
        let results: Vec<u64> = index.filter(FilterOp::Gte(0.0)).iter().collect();
        assert_eq!(results, vec![1, 3]);

        // Filter between -1.0 and 2.0
        let results: Vec<u64> = index
            .filter(FilterOp::BetweenInclusive(-1.0, 2.0))
            .iter()
            .collect();
        assert_eq!(results, vec![2, 1]);
    }

    #[test]
    fn test_nan_rejected() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<f64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        let result = index.insert(&IndexedValue::Plain(f64::NAN), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_cleanup() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.compact(1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.compact(2).unwrap();

        // Should have versions 0, 1, 2
        let versions_dir = temp.path().join("versions");
        assert!(versions_dir.join("0").exists());
        assert!(versions_dir.join("1").exists());
        assert!(versions_dir.join("2").exists());

        // Cleanup
        index.cleanup();

        // Only current version (2) should remain
        assert!(!versions_dir.join("0").exists());
        assert!(!versions_dir.join("1").exists());
        assert!(versions_dir.join("2").exists());
    }

    #[test]
    fn test_info_empty_index() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        let info = index.info();
        assert_eq!(info.format_version, super::super::io::FORMAT_VERSION);
        assert_eq!(info.current_offset, 0);
        assert_eq!(info.header_entry_count, 0);
        assert_eq!(info.deleted_count, 0);
        assert_eq!(info.data_file_count, 0);
        assert_eq!(info.pending_inserts, 0);
        assert_eq!(info.pending_deletes, 0);
        assert_eq!(info.version_dir, temp.path().join("versions").join("0"));
    }

    #[test]
    fn test_info_with_live_data() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.insert(&IndexedValue::Plain(30), 3).unwrap();

        let info = index.info();
        assert_eq!(info.pending_inserts, 3);
        assert_eq!(info.pending_deletes, 0);
        // Not yet compacted, so on-disk counts are still 0
        assert_eq!(info.header_entry_count, 0);
        assert_eq!(info.data_file_count, 0);
    }

    #[test]
    fn test_info_after_compact() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.compact(1).unwrap();

        let info = index.info();
        assert_eq!(info.current_offset, 1);
        assert!(info.header_entry_count > 0);
        assert_eq!(info.pending_inserts, 0);
        assert_eq!(info.pending_deletes, 0);
        assert!(info.data_file_count > 0);
        assert!(info.header_size_bytes > 0);
        assert!(info.data_total_bytes > 0);
    }

    #[test]
    fn test_info_with_pending_deletes() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.compact(1).unwrap();

        index.delete(1);

        let info = index.info();
        assert_eq!(info.pending_deletes, 1);
    }

    #[test]
    fn test_info_deleted_count_after_compact() {
        let temp = TempDir::new().unwrap();
        // Use a high threshold so deletes are carried forward, not applied
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::try_new(0.99).unwrap())
                .unwrap();

        // Insert enough entries so that delete ratio stays below threshold
        for i in 0..20 {
            index.insert(&IndexedValue::Plain(i * 10), i).unwrap();
        }
        index.compact(1).unwrap();

        index.delete(1);
        index.compact(2).unwrap();

        let info = index.info();
        assert_eq!(info.deleted_count, 1);
        assert_eq!(info.pending_deletes, 0);
    }

    #[test]
    fn test_integrity_check_healthy_empty_index() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        let result = index.integrity_check();
        assert!(result.passed);
        assert!(!result.checks.is_empty());

        // All checks should be Ok
        for check in &result.checks {
            assert_eq!(
                check.status,
                CheckStatus::Ok,
                "check '{}' failed: {:?}",
                check.name,
                check.details
            );
        }
    }

    #[test]
    fn test_integrity_check_healthy_with_data() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(10), 1).unwrap();
        index.insert(&IndexedValue::Plain(20), 2).unwrap();
        index.compact(1).unwrap();

        let result = index.integrity_check();
        assert!(result.passed);

        // Should have checks for CURRENT, version directory, header.idx, deleted.bin, data files
        let names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"CURRENT"));
        assert!(names.contains(&"version directory"));
        assert!(names.contains(&"header.idx"));
        assert!(names.contains(&"deleted.bin"));
        assert!(names.contains(&"data files"));
    }

    #[test]
    fn test_integrity_check_missing_current() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Remove the CURRENT file
        fs::remove_file(temp.path().join("CURRENT")).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);
        assert_eq!(result.checks.len(), 1);
        assert_eq!(result.checks[0].name, "CURRENT");
        assert_eq!(result.checks[0].status, CheckStatus::Failed);
    }

    #[test]
    fn test_integrity_check_missing_version_dir() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Remove the version directory
        fs::remove_dir_all(temp.path().join("versions").join("0")).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let failed: Vec<_> = result
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Failed)
            .collect();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].name, "version directory");
    }

    #[test]
    fn test_integrity_check_missing_header() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Remove the header file
        fs::remove_file(temp.path().join("versions").join("0").join("header.idx")).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let header_check = result
            .checks
            .iter()
            .find(|c| c.name == "header.idx")
            .unwrap();
        assert_eq!(header_check.status, CheckStatus::Failed);
    }

    #[test]
    fn test_integrity_check_corrupted_header() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Write invalid data to header (not divisible by 24)
        let header_path = temp.path().join("versions").join("0").join("header.idx");
        fs::write(&header_path, [0u8; 25]).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let header_check = result
            .checks
            .iter()
            .find(|c| c.name == "header.idx")
            .unwrap();
        assert_eq!(header_check.status, CheckStatus::Failed);
        assert!(header_check
            .details
            .as_ref()
            .unwrap()
            .contains("not divisible by 24"));
    }

    #[test]
    fn test_integrity_check_missing_deleted_bin() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Remove the deleted.bin file
        fs::remove_file(temp.path().join("versions").join("0").join("deleted.bin")).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let deleted_check = result
            .checks
            .iter()
            .find(|c| c.name == "deleted.bin")
            .unwrap();
        assert_eq!(deleted_check.status, CheckStatus::Failed);
    }

    #[test]
    fn test_integrity_check_corrupted_deleted_bin() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Write invalid data to deleted.bin (not divisible by 8)
        let deleted_path = temp.path().join("versions").join("0").join("deleted.bin");
        fs::write(&deleted_path, [0u8; 13]).unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let deleted_check = result
            .checks
            .iter()
            .find(|c| c.name == "deleted.bin")
            .unwrap();
        assert_eq!(deleted_check.status, CheckStatus::Failed);
        assert!(deleted_check
            .details
            .as_ref()
            .unwrap()
            .contains("not divisible by 8"));
    }

    #[test]
    fn test_integrity_check_invalid_current_content() {
        let temp = TempDir::new().unwrap();
        let index: NumberStorage<u64> =
            NumberStorage::new(temp.path().to_path_buf(), Threshold::default()).unwrap();

        // Overwrite CURRENT with empty content
        fs::write(temp.path().join("CURRENT"), "").unwrap();

        let result = index.integrity_check();
        assert!(!result.passed);

        let current_check = result.checks.iter().find(|c| c.name == "CURRENT").unwrap();
        assert_eq!(current_check.status, CheckStatus::Failed);
    }
}
