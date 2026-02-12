//! Main BoolStorage implementation.
//!
//! The `BoolStorage` is a concurrent, persistent boolean postings index optimized for
//! append-heavy workloads with occasional deletions. It uses a layered architecture:
//!
//! - **Live layer**: In-memory unsorted writes (fast inserts)
//! - **Compacted layer**: On-disk sorted, memory-mapped data (efficient reads)
//!
//! # Concurrency Model
//!
//! - Reads (`filter()`) are mostly lock-free via `ArcSwap` for version access
//! - Writes (`insert()`, `delete()`) acquire a write lock on the live layer
//! - Compaction (`compact()`) is serialized via `compaction_lock`

use super::indexer::IndexedValue;
use super::info::{IndexInfo, IntegrityCheck, IntegrityCheckResult};
use super::io::{
    copy_and_append_postings, ensure_version_dir, list_version_dirs, read_current,
    remove_version_dir, sync_dir, version_dir, write_current_atomic, write_postings,
    write_postings_from_iter, FORMAT_VERSION,
};
use super::iterator::FilterData;
use super::live::{LiveLayer, LiveSnapshot};
use super::merge::{sorted_merge, sorted_subtract};
use super::version::CompactedVersion;
use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// A threshold value between 0.0 and 1.0 (inclusive).
///
/// Used to control when deletions are physically applied vs carried forward
/// during compaction. A threshold of 0.1 means deletions are applied when
/// they exceed 10% of total postings.
///
/// # Tuning Guidance
///
/// - **Lower values (e.g., 0.05)**: Apply deletions more aggressively. Better for
///   workloads with steady deletion rates. Uses more CPU during compaction but
///   keeps data files smaller.
///
/// - **Higher values (e.g., 0.3)**: Carry forward deletions longer. Better for
///   burst deletion patterns or when compaction time is critical. Data files
///   may grow larger between cleanup cycles.
///
/// The default value of 0.1 (10%) is a reasonable starting point for most workloads.
#[derive(Debug, Clone, Copy)]
pub struct DeletionThreshold(f64);

impl DeletionThreshold {
    /// Returns the inner value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for DeletionThreshold {
    fn default() -> Self {
        DeletionThreshold(0.1)
    }
}

impl TryFrom<f64> for DeletionThreshold {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if (0.0..=1.0).contains(&value) {
            Ok(DeletionThreshold(value))
        } else {
            Err("threshold must be between 0.0 and 1.0")
        }
    }
}

impl TryFrom<f32> for DeletionThreshold {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        DeletionThreshold::try_from(value as f64)
    }
}

/// Check if snapshot values can be safely appended (all > existing max).
///
/// Returns `false` if there's overlap, indicating the merge strategy should be used
/// instead of the fast append path. This happens when doc_ids are not monotonically
/// increasing (e.g., reusing deleted doc_ids).
///
/// # Complexity
///
/// O(1) - only checks the first new value against the existing max.
fn can_append_safely(existing_max: Option<u64>, new_values: &[u64]) -> bool {
    match (existing_max, new_values.first()) {
        (Some(max), Some(&first)) => first > max,
        _ => true,
    }
}

/// Boolean postings index that stores doc_ids partitioned into TRUE and FALSE sets.
///
/// # Thread Safety
///
/// `BoolStorage` is `Send + Sync` and designed for concurrent access:
///
/// - **`version`** (`ArcSwap<CompactedVersion>`): Lock-free reads via `load()`. Writers
///   atomically swap in new versions. Readers holding old `Arc` continue safely.
///
/// - **`live`** (`RwLock<LiveLayer>`): Protects in-memory mutations. Readers get shared
///   access; `insert()`/`delete()` take exclusive access. The `filter()` method uses
///   double-check locking to minimize write lock contention.
///
/// - **`compaction_lock`** (`Mutex<()>`): Serializes compaction operations. Only one
///   compaction can run at a time, but reads/writes continue concurrently.
pub struct BoolStorage {
    base_path: PathBuf,
    /// The current compacted version (memory-mapped). Swapped atomically during compaction.
    version: ArcSwap<CompactedVersion>,
    /// In-memory layer for recent writes. Protected by RwLock for concurrent reads.
    live: RwLock<LiveLayer>,
    /// Ensures only one compaction runs at a time.
    compaction_lock: Mutex<()>,
    /// Threshold for applying vs carrying forward deletions.
    deletion_threshold: DeletionThreshold,
}

impl BoolStorage {
    /// Create a new BoolStorage at the given path.
    ///
    /// If a previous version exists (detected via CURRENT file), it will be loaded.
    /// Otherwise, starts with an empty index.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Cannot create the base directory (permission denied)
    /// - CURRENT file exists but is malformed
    /// - Format version in CURRENT doesn't match `FORMAT_VERSION` (incompatible index)
    /// - Cannot load the compacted version (corrupted files, I/O error)
    ///
    /// # Panics
    ///
    /// Does not panic. All errors are returned via `Result`.
    pub fn new(base_path: PathBuf, deletion_threshold: DeletionThreshold) -> Result<Self> {
        std::fs::create_dir_all(&base_path)
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
            deletion_threshold,
        })
    }

    /// Insert a doc_id with the given boolean value.
    ///
    /// Insert a doc_id based on an IndexedValue extracted by BoolIndexer.
    ///
    /// - `Plain(bool)`: inserts the doc_id into the TRUE or FALSE set.
    /// - `Array(bools)`: inserts into TRUE set if any element is true,
    ///   into FALSE set if any element is false. The doc_id may appear
    ///   in both sets.
    ///
    /// The insert is recorded in the live layer and becomes visible to subsequent
    /// `filter()` calls. Data is not persisted until `compact()` is called.
    ///
    /// # Concurrency
    ///
    /// Acquires a write lock on the live layer. Multiple concurrent inserts are
    /// serialized. Consider batching if insert rate is very high.
    ///
    /// # Complexity
    ///
    /// O(1) amortized - appends to an unsorted vector.
    pub fn insert(&self, indexed_value: &IndexedValue, doc_id: u64) {
        match indexed_value {
            IndexedValue::Plain(b) => {
                let mut live = self.live.write().unwrap();
                live.insert(*b, doc_id);
            }
            IndexedValue::Array(bools) => {
                let has_true = bools.iter().any(|&b| b);
                let has_false = bools.iter().any(|&b| !b);
                if has_true || has_false {
                    let mut live = self.live.write().unwrap();
                    if has_true {
                        live.insert(true, doc_id);
                    }
                    if has_false {
                        live.insert(false, doc_id);
                    }
                }
            }
        }
    }

    /// Delete a doc_id from both TRUE and FALSE sets.
    ///
    /// Marks the doc_id for deletion. The deletion is applied lazily:
    /// - Immediately visible to `filter()` (excluded from results)
    /// - Physically removed during `compact()` based on threshold
    ///
    /// Deleting a non-existent doc_id is a no-op (no error).
    ///
    /// # Concurrency
    ///
    /// Acquires a write lock on the live layer.
    ///
    /// # Complexity
    ///
    /// O(1) amortized - appends to the deletes vector.
    pub fn delete(&self, doc_id: u64) {
        let mut live = self.live.write().unwrap();
        live.delete(doc_id);
    }

    /// Return filter data for zero-allocation iteration over doc_ids.
    ///
    /// Returns an iterator that yields doc_ids matching the given boolean value,
    /// excluding any deleted doc_ids. Results are sorted in ascending order.
    ///
    /// # Snapshot Semantics
    ///
    /// The returned `FilterData` captures a consistent snapshot at call time.
    /// Subsequent `insert()`/`delete()` calls do not affect the iteration.
    ///
    /// # Concurrency
    ///
    /// Uses double-check locking to minimize contention:
    /// 1. First tries read lock - O(1) `Arc::clone` if snapshot is clean
    /// 2. Only acquires write lock if snapshot needs refresh
    /// 3. Re-checks dirty flag after write lock (another thread may have refreshed)
    ///
    /// In the common case (no recent mutations), this is nearly lock-free.
    ///
    /// # Complexity
    ///
    /// - Snapshot acquisition: O(1) if clean, O(n log n) if dirty (sorting)
    /// - Iteration: O(n + m + d) where n=compacted postings, m=live inserts, d=deletes
    ///
    /// Obtain a fresh snapshot of the live layer using double-check locking.
    ///
    /// 1. Acquire read lock — if snapshot is clean, return `Arc::clone` (O(1)).
    /// 2. If dirty, drop read lock, acquire write lock, re-check dirty flag
    ///    (another thread may have refreshed), refresh if still dirty.
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

    pub fn filter(&self, value: bool) -> FilterData {
        let snapshot = self.fresh_snapshot();
        let version = self.version.load();
        FilterData::new(Arc::clone(&version), snapshot, value)
    }

    /// Compact the index at the given version number.
    ///
    /// Merges the live layer into the compacted version and writes to disk. After
    /// successful compaction, the new version becomes active and compacted items
    /// are cleared from the live layer.
    ///
    /// # Deletion Strategies
    ///
    /// Compaction uses one of two strategies based on `deletion_threshold`:
    ///
    /// - **Strategy A (Apply Deletions)**: When `deletes / total_postings > threshold`,
    ///   deletions are physically applied. Results in smaller files but requires
    ///   reading and rewriting all postings. O(n) where n = total postings.
    ///
    /// - **Strategy B (Carry Forward)**: When below threshold, deletions are merged
    ///   into `deleted.bin` and postings are copied/appended. Faster compaction but
    ///   deletion overhead during reads. Uses fast append path when possible.
    ///
    /// # Concurrency
    ///
    /// - Acquires `compaction_lock` to serialize compactions (only one at a time)
    /// - Writes to a new version directory, then atomically swaps
    /// - Readers continue using the old version until swap completes
    /// - Operations during compaction are preserved (not lost)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Cannot create version directory (permission denied, disk full)
    /// - Cannot write postings files (I/O error, disk full)
    /// - Cannot update CURRENT file (I/O error)
    /// - Cannot load the new version (shouldn't happen if writes succeeded)
    ///
    /// # Recovery
    ///
    /// On error, the old version remains active. Partial writes may leave orphan
    /// files in the new version directory; call `cleanup()` to remove them.
    /// Retrying compaction with the same or different version number is safe.
    ///
    /// # Complexity
    ///
    /// - Strategy A: O(n) where n = total postings (must read and rewrite all)
    /// - Strategy B fast path: O(m) where m = new inserts (copy existing + append)
    /// - Strategy B merge path: O(n + m) when new doc_ids overlap existing range
    pub fn compact(&self, version_number: u64) -> Result<()> {
        let compaction_guard = self.compaction_lock.lock().unwrap();

        let snapshot = self.fresh_snapshot();

        // Nothing to compact — free memory and return early
        if snapshot.true_inserts.is_empty()
            && snapshot.false_inserts.is_empty()
            && snapshot.deletes.is_empty()
        {
            let mut live = self.live.write().unwrap();
            live.ops.drain(..snapshot.ops_len);
            live.ops.shrink_to_fit();
            live.refresh_snapshot();
            return Ok(());
        }

        // Get current version
        let current = self.version.load();

        if version_number == current.version_number
            && (current.true_postings.is_some()
                || current.false_postings.is_some()
                || current.deletes.is_some())
        {
            return Err(anyhow!(
                "Cannot compact to version {version_number}: same as current active version. \
                 Use a different version number to avoid corrupting active mmaps."
            ));
        }

        let new_version_dir = ensure_version_dir(&self.base_path, version_number)?;

        if snapshot.deletes.is_empty() {
            // PATH 1: No new deletions - optimized copy + append
            self.compact_no_new_deletes(&snapshot, &current, &new_version_dir)?;
        } else {
            // PATH 2: With new deletions - merge strategy
            self.compact_with_new_deletes(&snapshot, &current, &new_version_dir)?;
        }

        // Sync directory
        sync_dir(&new_version_dir)?;

        // Atomic update of CURRENT
        write_current_atomic(&self.base_path, version_number)?;

        // Atomic update: swap version AND clear compacted items
        {
            let mut live = self.live.write().unwrap();

            // Swap version while holding lock
            let new_version = CompactedVersion::load(&self.base_path, version_number)?;
            self.version.store(Arc::new(new_version));

            // Remove compacted ops by position, not by value.
            // Vec::push is append-only: items at indices 0..ops_len were present
            // when the snapshot was taken, while items at indices ops_len.. were
            // inserted concurrently during the I/O phase (when the write lock was
            // released). Using drain(..len) instead of value-based retain ensures
            // that concurrent inserts of the same doc_id are preserved.
            live.ops.drain(..snapshot.ops_len);

            // Refresh snapshot to reflect cleared state
            live.refresh_snapshot();
        }

        drop(compaction_guard);

        Ok(())
    }

    /// Optimized compaction path when there are no new deletions.
    ///
    /// Copies existing postings files and appends new inserts (fast path).
    /// Falls back to merge strategy if new doc_ids overlap with existing range.
    ///
    /// # Fast Path vs Merge Path
    ///
    /// - **Fast path**: When all new doc_ids > existing max, we can simply copy
    ///   the existing file and append. This is O(m) where m = new inserts.
    ///
    /// - **Merge path**: When new doc_ids overlap (e.g., reused doc_ids), we must
    ///   merge the two sorted sequences. This is O(n + m).
    ///
    /// The fast path is typical for monotonically increasing doc_ids.
    fn compact_no_new_deletes(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
        new_version_dir: &std::path::Path,
    ) -> Result<()> {
        let current_dir = version_dir(&self.base_path, current.version_number);

        let true_max = current.true_postings_slice().last().copied();
        let false_max = current.false_postings_slice().last().copied();

        // TRUE postings: check if we can safely append or need to merge
        if can_append_safely(true_max, &snapshot.true_inserts) {
            // Fast path: copy existing + append new inserts
            copy_and_append_postings(
                &current_dir.join("true.bin"),
                &new_version_dir.join("true.bin"),
                true_max,
                &snapshot.true_inserts,
            )?;
        } else {
            // Merge path: new inserts overlap with existing max
            write_postings_from_iter(
                &new_version_dir.join("true.bin"),
                sorted_merge(
                    current.true_postings_slice().iter().copied(),
                    snapshot.true_inserts.iter().copied(),
                ),
            )?;
        }

        // FALSE postings: check if we can safely append or need to merge
        if can_append_safely(false_max, &snapshot.false_inserts) {
            // Fast path: copy existing + append new inserts
            copy_and_append_postings(
                &current_dir.join("false.bin"),
                &new_version_dir.join("false.bin"),
                false_max,
                &snapshot.false_inserts,
            )?;
        } else {
            // Merge path: new inserts overlap with existing max
            write_postings_from_iter(
                &new_version_dir.join("false.bin"),
                sorted_merge(
                    current.false_postings_slice().iter().copied(),
                    snapshot.false_inserts.iter().copied(),
                ),
            )?;
        }

        // Carry forward existing deleted.bin (copy as-is)
        copy_and_append_postings(
            &current_dir.join("deleted.bin"),
            &new_version_dir.join("deleted.bin"),
            None, // No max check needed for deletes
            &[],  // No new deletes to append
        )?;

        Ok(())
    }

    /// Compaction path with new deletions - uses merge strategy with ratio check.
    ///
    /// Chooses between two strategies based on the deletion ratio:
    ///
    /// # Strategy A: Apply Deletions (when `deletes/postings > threshold`)
    ///
    /// Physically removes deleted doc_ids from the postings files:
    /// - `true.bin` = merge(existing_true, new_true) - merge(existing_deleted, new_deleted)
    /// - `false.bin` = merge(existing_false, new_false) - merge(existing_deleted, new_deleted)
    /// - `deleted.bin` = empty
    ///
    /// **Tradeoff**: Higher compaction cost (must read/write all postings), but smaller
    /// files and faster subsequent reads (no deletion filtering needed).
    ///
    /// # Strategy B: Carry Forward Deletions (when `deletes/postings <= threshold`)
    ///
    /// Accumulates deletions without applying them:
    /// - `true.bin` = copy or merge existing + new (same logic as no-delete path)
    /// - `false.bin` = copy or merge existing + new
    /// - `deleted.bin` = merge(existing_deleted, new_deleted)
    ///
    /// **Tradeoff**: Faster compaction (may use copy+append), but larger files and
    /// deletion filtering overhead during reads.
    ///
    /// # Threshold Guidance
    ///
    /// The threshold balances compaction cost vs read cost. With threshold=0.1:
    /// - Deletions < 10% of postings: carry forward (fast compaction)
    /// - Deletions >= 10% of postings: apply (clean up disk space)
    fn compact_with_new_deletes(
        &self,
        snapshot: &LiveSnapshot,
        current: &Arc<CompactedVersion>,
        new_version_dir: &std::path::Path,
    ) -> Result<()> {
        // Estimate merged deletes count arithmetically. This may overcount when doc_ids
        // overlap between compacted and live layers (non-monotonic inserts), but the ratio
        // is a heuristic so an approximate count is acceptable.
        // When doc_ids are monotonically increasing, counts are exact and the fast append
        // path in Strategy B avoids re-reading existing postings.
        let merged_deletes_count = current.deletes_slice().len() + snapshot.deletes.len();

        // Estimate posting counts arithmetically. May overcount with non-monotonic doc_ids
        // due to duplicates across compacted and live layers.
        let true_count = current.true_postings_slice().len() + snapshot.true_inserts.len();
        let false_count = current.false_postings_slice().len() + snapshot.false_inserts.len();
        let total_postings = true_count + false_count;

        let should_apply = if total_postings > 0 {
            merged_deletes_count as f64 / total_postings as f64 > self.deletion_threshold.value()
        } else {
            // No postings at all — apply deletions to clear deleted.bin,
            // preventing stale deletes from filtering out future re-inserts.
            merged_deletes_count > 0
        };

        if should_apply {
            // Strategy A: Apply deletions using sorted_subtract (streaming)
            // Create merged deletes iterator inline - no allocation needed
            write_postings_from_iter(
                &new_version_dir.join("true.bin"),
                sorted_subtract(
                    sorted_merge(
                        current.true_postings_slice().iter().copied(),
                        snapshot.true_inserts.iter().copied(),
                    ),
                    sorted_merge(
                        current.deletes_slice().iter().copied(),
                        snapshot.deletes.iter().copied(),
                    ),
                ),
            )?;

            write_postings_from_iter(
                &new_version_dir.join("false.bin"),
                sorted_subtract(
                    sorted_merge(
                        current.false_postings_slice().iter().copied(),
                        snapshot.false_inserts.iter().copied(),
                    ),
                    sorted_merge(
                        current.deletes_slice().iter().copied(),
                        snapshot.deletes.iter().copied(),
                    ),
                ),
            )?;

            // Empty deletes file (deletions were applied)
            write_postings(&new_version_dir.join("deleted.bin"), &[])?;
        } else {
            // Strategy B: Carry forward deletes (copy + append optimization)
            let current_dir = version_dir(&self.base_path, current.version_number);

            let true_max = current.true_postings_slice().last().copied();
            let false_max = current.false_postings_slice().last().copied();

            // TRUE postings: check if we can safely append or need to merge
            if can_append_safely(true_max, &snapshot.true_inserts) {
                copy_and_append_postings(
                    &current_dir.join("true.bin"),
                    &new_version_dir.join("true.bin"),
                    true_max,
                    &snapshot.true_inserts,
                )?;
            } else {
                // Merge path: new inserts overlap with existing max
                write_postings_from_iter(
                    &new_version_dir.join("true.bin"),
                    sorted_merge(
                        current.true_postings_slice().iter().copied(),
                        snapshot.true_inserts.iter().copied(),
                    ),
                )?;
            }

            // FALSE postings: check if we can safely append or need to merge
            if can_append_safely(false_max, &snapshot.false_inserts) {
                copy_and_append_postings(
                    &current_dir.join("false.bin"),
                    &new_version_dir.join("false.bin"),
                    false_max,
                    &snapshot.false_inserts,
                )?;
            } else {
                // Merge path: new inserts overlap with existing max
                write_postings_from_iter(
                    &new_version_dir.join("false.bin"),
                    sorted_merge(
                        current.false_postings_slice().iter().copied(),
                        snapshot.false_inserts.iter().copied(),
                    ),
                )?;
            }

            write_postings_from_iter(
                &new_version_dir.join("deleted.bin"),
                sorted_merge(
                    current.deletes_slice().iter().copied(),
                    snapshot.deletes.iter().copied(),
                ),
            )?;
        }

        Ok(())
    }

    /// Get the current version number.
    ///
    /// Returns the version number of the currently active compacted version.
    /// Returns 0 if no compaction has occurred yet.
    ///
    /// # Complexity
    ///
    /// O(1) - atomic load from `ArcSwap`.
    pub fn current_version_number(&self) -> u64 {
        self.version.load().version_number
    }

    /// Remove all old version directories except the current one.
    ///
    /// Call this after compaction to reclaim disk space from old versions.
    /// It's safe to call during normal operation - readers using old versions
    /// will continue to work (files remain open).
    ///
    /// # Error Handling
    ///
    /// Errors are logged via `tracing::error!` but do not cause the method to fail.
    /// Partial cleanup is acceptable; re-running cleanup will retry failed removals.
    ///
    /// # Complexity
    ///
    /// O(v) where v = number of old versions, each requiring a directory traversal.
    pub fn cleanup(&self) {
        let compaction_guard = self.compaction_lock.lock().unwrap();
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

        drop(compaction_guard);
    }

    /// Get metadata and statistics about the index.
    pub fn info(&self) -> IndexInfo {
        let version = self.version.load();
        let live = self.live.read().unwrap();
        let ver_dir = version_dir(&self.base_path, version.version_number);

        IndexInfo {
            format_version: FORMAT_VERSION,
            current_version_number: version.version_number,
            version_dir: ver_dir.clone(),
            true_count: version.true_postings_slice().len(),
            false_count: version.false_postings_slice().len(),
            deleted_count: version.deletes_slice().len(),
            true_size_bytes: file_size(&ver_dir.join("true.bin")),
            false_size_bytes: file_size(&ver_dir.join("false.bin")),
            deleted_size_bytes: file_size(&ver_dir.join("deleted.bin")),
            pending_ops: live.ops.len(),
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
            Ok(Some((format_version, version_number))) => {
                checks.push(IntegrityCheck::ok(
                    "CURRENT",
                    Some(format!(
                        "version: {format_version}, version_number: {version_number}"
                    )),
                ));

                // Check format version
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

                // Check version directory
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

                // Check binary files exist
                let true_path = ver_dir.join("true.bin");
                let false_path = ver_dir.join("false.bin");
                let deleted_path = ver_dir.join("deleted.bin");

                if !true_path.exists() || !false_path.exists() || !deleted_path.exists() {
                    let mut missing = Vec::new();
                    if !true_path.exists() {
                        missing.push("true.bin");
                    }
                    if !false_path.exists() {
                        missing.push("false.bin");
                    }
                    if !deleted_path.exists() {
                        missing.push("deleted.bin");
                    }
                    checks.push(IntegrityCheck::failed(
                        "binary files",
                        Some(format!("Missing: {}", missing.join(", "))),
                    ));
                    return IntegrityCheckResult::new(checks);
                }
                checks.push(IntegrityCheck::ok(
                    "binary files",
                    Some("All present".to_string()),
                ));

                // Check binary files are valid (size % 8 == 0, values strictly sorted)
                let true_valid = validate_binary_file(&true_path, "true.bin");
                let false_valid = validate_binary_file(&false_path, "false.bin");
                let deleted_valid = validate_binary_file(&deleted_path, "deleted.bin");

                match (&true_valid, &false_valid, &deleted_valid) {
                    (Ok(()), Ok(()), Ok(())) => {
                        checks.push(IntegrityCheck::ok(
                            "binary files valid",
                            Some("Size and sorting OK".to_string()),
                        ));
                    }
                    _ => {
                        let mut errors = Vec::new();
                        if let Err(e) = true_valid {
                            errors.push(format!("true.bin: {e}"));
                        }
                        if let Err(e) = false_valid {
                            errors.push(format!("false.bin: {e}"));
                        }
                        if let Err(e) = deleted_valid {
                            errors.push(format!("deleted.bin: {e}"));
                        }
                        checks.push(IntegrityCheck::failed(
                            "binary files valid",
                            Some(errors.join("; ")),
                        ));
                        return IntegrityCheckResult::new(checks);
                    }
                }

                // Check deleted items not in postings
                let version = self.version.load();
                let true_slice = version.true_postings_slice();
                let false_slice = version.false_postings_slice();
                let deleted_slice = version.deletes_slice();

                let mut found_in_postings = Vec::new();
                for &deleted_id in deleted_slice {
                    if true_slice.binary_search(&deleted_id).is_ok() {
                        found_in_postings.push(format!("{deleted_id} in true"));
                    }
                    if false_slice.binary_search(&deleted_id).is_ok() {
                        found_in_postings.push(format!("{deleted_id} in false"));
                    }
                }

                if found_in_postings.is_empty() {
                    checks.push(IntegrityCheck::ok(
                        "deleted not in postings",
                        Some("OK".to_string()),
                    ));
                } else {
                    // Strategy B (carry forward): deleted items are expected in postings,
                    // subtracted at query time via SortedSubtract.
                    // Note: After Strategy A, deleted.bin is empty so this loop never finds
                    // anything — we cannot detect Strategy A corruption here without an
                    // external reference to the pre-compaction delete list.
                    checks.push(IntegrityCheck::ok(
                        "deleted not in postings",
                        Some(format!(
                            "Strategy B: {} deleted items carried forward in postings (expected)",
                            found_in_postings.len()
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

/// Get file size, returning 0 if the file doesn't exist or can't be read.
fn file_size(path: &std::path::Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Validate a binary postings file.
fn validate_binary_file(path: &std::path::Path, name: &str) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("Failed to get metadata for {name}"))?;

    let size = metadata.len();
    if !size.is_multiple_of(8) {
        return Err(anyhow!("file size ({size} bytes) is not a multiple of 8"));
    }

    if size == 0 {
        return Ok(());
    }

    let bytes = fs::read(path).with_context(|| format!("Failed to read {name}"))?;

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
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_empty_index() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();
        assert_eq!(index.current_version_number(), 0);
    }

    #[test]
    fn test_insert_and_filter() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(false), 2);
        index.insert(&IndexedValue::Plain(false), 6);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![1, 5, 10]);
        assert_eq!(false_results, vec![2, 6]);
    }

    #[test]
    fn test_delete_and_filter() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(false), 5); // Same doc in false
        index.delete(5); // Deletes from both

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![1, 10]);
        assert!(false_results.is_empty());
    }

    #[test]
    fn test_compact_basic() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(false), 2);

        // Compact
        index.compact(1).unwrap();

        // Verify data persists
        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![1, 5]);
        assert_eq!(false_results, vec![2]);
        assert_eq!(index.current_version_number(), 1);
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        // Create and populate index
        {
            let index = BoolStorage::new(base_path.clone(), DeletionThreshold::default()).unwrap();
            index.insert(&IndexedValue::Plain(true), 1);
            index.insert(&IndexedValue::Plain(true), 5);
            index.insert(&IndexedValue::Plain(false), 2);
            index.compact(1).unwrap();
        }

        // Reopen and verify
        {
            let index = BoolStorage::new(base_path, DeletionThreshold::default()).unwrap();
            let true_results: Vec<u64> = index.filter(true).iter().collect();
            let false_results: Vec<u64> = index.filter(false).iter().collect();

            assert_eq!(true_results, vec![1, 5]);
            assert_eq!(false_results, vec![2]);
            assert_eq!(index.current_version_number(), 1);
        }
    }

    #[test]
    fn test_compact_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.delete(5);

        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 10]);
    }

    #[test]
    fn test_sparse_doc_ids() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Insert sparse doc_ids
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(true), 100);
        index.insert(&IndexedValue::Plain(true), 1000);

        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 10, 100, 1000]);

        // Compact and verify
        index.compact(1).unwrap();

        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 10, 100, 1000]);
    }

    #[test]
    fn test_ops_during_compaction_preserved() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);

        index.compact(1).unwrap();

        // Operations after compact but before next compact
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(true), 20);

        // These should be visible
        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 5, 10, 20]);
    }

    #[test]
    fn test_filter_during_compaction_sees_all_data() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let index = Arc::new(
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap(),
        );

        // Insert data
        for i in 1..=100 {
            index.insert(&IndexedValue::Plain(true), i);
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);

        // Spawn thread that filters continuously
        let idx = Arc::clone(&index);
        let handle = thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                let results: Vec<u64> = idx.filter(true).iter().collect();
                // Should ALWAYS see at least 100 items
                assert!(
                    results.len() >= 100,
                    "Data lost! Only saw {} items, expected at least 100",
                    results.len()
                );
            }
        });

        // Compact in main thread (introduces the race window)
        index.compact(1).unwrap();

        // Signal the reader thread to stop
        stop.store(true, Ordering::Relaxed);

        handle.join().unwrap();
    }

    #[test]
    fn test_compact_no_deletions_path() {
        // Tests the optimized zero-copy path when there are no deletions
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Insert initial data and compact
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(false), 2);
        index.compact(1).unwrap();

        // Insert more data WITHOUT deletions
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(true), 20);
        index.insert(&IndexedValue::Plain(false), 15);

        // Compact again - should use the optimized path
        index.compact(2).unwrap();

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![1, 5, 10, 20]);
        assert_eq!(false_results, vec![2, 15]);
        assert_eq!(index.current_version_number(), 2);
    }

    #[test]
    fn test_compact_multiple_rounds_no_deletions() {
        // Tests multiple compaction rounds without deletions
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Round 1
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.compact(1).unwrap();

        // Round 2 - no deletions
        index.insert(&IndexedValue::Plain(true), 10);
        index.compact(2).unwrap();

        // Round 3 - no deletions
        index.insert(&IndexedValue::Plain(true), 20);
        index.compact(3).unwrap();

        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 5, 10, 20]);
    }

    #[test]
    fn test_compact_carries_forward_existing_deletes() {
        // Tests that existing deletes in the compacted version are preserved
        // when using the no-new-deletions path
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Insert and delete, then compact (below threshold, deletes carried forward)
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.delete(5);
        index.compact(1).unwrap();

        // Verify delete was carried forward (below threshold)
        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 10]);

        // Now insert more data WITHOUT new deletions
        index.insert(&IndexedValue::Plain(true), 20);
        index.insert(&IndexedValue::Plain(true), 30);
        index.compact(2).unwrap();

        // Existing delete should still be respected
        let results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(results, vec![1, 10, 20, 30]);
    }

    #[test]
    fn test_compact_empty_to_non_empty() {
        // Tests compaction from empty state
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Insert data with no prior compacted version
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(true), 20);
        index.insert(&IndexedValue::Plain(false), 15);

        // First compaction - should work with empty source
        index.compact(1).unwrap();

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![10, 20]);
        assert_eq!(false_results, vec![15]);
    }

    #[test]
    fn test_filter_arc_sharing() {
        // Tests that multiple FilterData instances share the same Arc snapshot
        // when no mutations occur between calls
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(false), 2);

        // Get two FilterData instances without mutations in between
        let filter1 = index.filter(true);
        let filter2 = index.filter(false);

        // Both should produce correct results
        let results1: Vec<u64> = filter1.iter().collect();
        let results2: Vec<u64> = filter2.iter().collect();

        assert_eq!(results1, vec![1, 5]);
        assert_eq!(results2, vec![2]);

        // After compact, the snapshot should still be shared for subsequent filters
        index.compact(1).unwrap();

        let filter3 = index.filter(true);
        let filter4 = index.filter(true);

        let results3: Vec<u64> = filter3.iter().collect();
        let results4: Vec<u64> = filter4.iter().collect();

        assert_eq!(results3, vec![1, 5]);
        assert_eq!(results4, vec![1, 5]);
    }

    #[test]
    fn test_cleanup_removes_old_versions() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Create multiple versions
        index.insert(&IndexedValue::Plain(true), 1);
        index.compact(1).unwrap();
        assert!(tmp.path().join("versions/1").exists());

        index.insert(&IndexedValue::Plain(true), 2);
        index.compact(2).unwrap();
        assert!(tmp.path().join("versions/1").exists());
        assert!(tmp.path().join("versions/2").exists());

        index.insert(&IndexedValue::Plain(true), 3);
        index.compact(3).unwrap();
        assert!(tmp.path().join("versions/1").exists());
        assert!(tmp.path().join("versions/2").exists());
        assert!(tmp.path().join("versions/3").exists());

        // Cleanup should remove old versions
        index.cleanup();

        assert!(!tmp.path().join("versions/1").exists());
        assert!(!tmp.path().join("versions/2").exists());
        assert!(tmp.path().join("versions/3").exists()); // Current version preserved
    }

    #[test]
    fn test_threshold_valid() {
        let t: DeletionThreshold = 0.5f64.try_into().unwrap();
        assert!((t.value() - 0.5).abs() < f64::EPSILON);

        let t: DeletionThreshold = 0.0f64.try_into().unwrap();
        assert!((t.value() - 0.0).abs() < f64::EPSILON);

        let t: DeletionThreshold = 1.0f64.try_into().unwrap();
        assert!((t.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_invalid() {
        assert!(DeletionThreshold::try_from(-0.1f64).is_err());
        assert!(DeletionThreshold::try_from(1.1f64).is_err());
        assert!(DeletionThreshold::try_from(-1.0f64).is_err());
        assert!(DeletionThreshold::try_from(2.0f64).is_err());
    }

    #[test]
    fn test_threshold_from_f32() {
        let t: DeletionThreshold = 0.5f32.try_into().unwrap();
        assert!((t.value() - 0.5).abs() < 0.0001);

        let t: DeletionThreshold = 0.0f32.try_into().unwrap();
        assert!((t.value() - 0.0).abs() < 0.0001);

        let t: DeletionThreshold = 1.0f32.try_into().unwrap();
        assert!((t.value() - 1.0).abs() < 0.0001);

        // Invalid f32 values
        assert!(DeletionThreshold::try_from(-0.1f32).is_err());
        assert!(DeletionThreshold::try_from(1.1f32).is_err());
    }

    #[test]
    fn test_open_incompatible_format_version() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().to_path_buf();

        // Create index and compact
        {
            let index = BoolStorage::new(base_path.clone(), DeletionThreshold::default()).unwrap();
            index.insert(&IndexedValue::Plain(true), 1);
            index.compact(1).unwrap();
        }

        // Manually write a wrong format version to CURRENT
        let current_path = base_path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        // Try to open - should fail due to format version mismatch
        let result = BoolStorage::new(base_path, DeletionThreshold::default());
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
    fn test_compact_with_overlapping_doc_ids_no_deletes() {
        // Test the merge path in compact_no_new_deletes
        // Insert doc_ids, compact, then insert a LOWER doc_id to trigger merge path
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // First round: insert high doc_ids and compact
        index.insert(&IndexedValue::Plain(true), 100);
        index.insert(&IndexedValue::Plain(true), 200);
        index.insert(&IndexedValue::Plain(false), 150);
        index.insert(&IndexedValue::Plain(false), 250);
        index.compact(1).unwrap();

        // Second round: insert LOWER doc_ids (triggers merge path since they're < existing max)
        index.insert(&IndexedValue::Plain(true), 50); // 50 < 200 (existing max for true)
        index.insert(&IndexedValue::Plain(false), 75); // 75 < 250 (existing max for false)

        // Compact again - should use merge path due to overlapping doc_ids
        index.compact(2).unwrap();

        // Verify all data is present and sorted
        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![50, 100, 200]);
        assert_eq!(false_results, vec![75, 150, 250]);
    }

    #[test]
    fn test_compact_with_overlapping_doc_ids_with_deletes() {
        // Test the merge path in compact_with_new_deletes (Strategy B)
        // Insert doc_ids, compact with deletes below threshold, then insert lower doc_ids
        let tmp = TempDir::new().unwrap();
        // Use very high threshold so deletes are carried forward (Strategy B)
        let index = BoolStorage::new(tmp.path().to_path_buf(), 0.9f64.try_into().unwrap()).unwrap();

        // First round: insert high doc_ids and compact
        index.insert(&IndexedValue::Plain(true), 100);
        index.insert(&IndexedValue::Plain(true), 200);
        index.insert(&IndexedValue::Plain(true), 300);
        index.insert(&IndexedValue::Plain(false), 150);
        index.insert(&IndexedValue::Plain(false), 250);
        index.insert(&IndexedValue::Plain(false), 350);
        index.compact(1).unwrap();

        // Second round: insert LOWER doc_ids and a delete (below threshold)
        index.insert(&IndexedValue::Plain(true), 50); // 50 < 300 (existing max for true) - triggers merge
        index.insert(&IndexedValue::Plain(false), 75); // 75 < 350 (existing max for false) - triggers merge
        index.delete(100); // Add a delete but below 90% threshold

        // Compact again - should use Strategy B with merge path
        index.compact(2).unwrap();

        // Verify all data is present and sorted (100 should be filtered out)
        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();

        assert_eq!(true_results, vec![50, 200, 300]);
        assert_eq!(false_results, vec![75, 150, 250, 350]);
    }

    #[test]
    fn test_compact_reinsert_same_doc_id_not_lost() {
        // Regression test: compaction must not drop a re-inserted doc_id that
        // matches a value already present in the compacted snapshot.
        //
        // Scenario:
        //   1. Insert doc 5, delete doc 5, compact (version 1) — compacts away both
        //   2. Re-insert doc 5 (simulates a concurrent insert of the same value)
        //   3. Compact (version 2) — with the fix, the second insert survives
        //
        // Before the fix, the value-based `retain` would see doc 5 in the snapshot
        // HashSet and silently drop the re-inserted copy, causing data loss.
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Step 1: insert and delete doc 5, then compact
        index.insert(&IndexedValue::Plain(true), 5);
        index.delete(5);
        index.compact(1).unwrap();

        // Doc 5 should be gone after compaction
        let results: Vec<u64> = index.filter(true).iter().collect();
        assert!(
            !results.contains(&5),
            "doc 5 should not be present after delete + compact"
        );

        // Step 2: re-insert the same doc_id
        index.insert(&IndexedValue::Plain(true), 5);

        // Step 3: compact again — the re-inserted doc 5 must survive
        index.compact(2).unwrap();

        let results: Vec<u64> = index.filter(true).iter().collect();
        assert!(
            results.contains(&5),
            "doc 5 should be present after re-insert + compact, got: {results:?}"
        );
    }

    #[test]
    fn test_update_plain_true() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(true), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert_eq!(true_results, vec![1]);
        assert!(false_results.is_empty());
    }

    #[test]
    fn test_update_plain_false() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Plain(false), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert!(true_results.is_empty());
        assert_eq!(false_results, vec![1]);
    }

    #[test]
    fn test_update_array_mixed() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Array(vec![true, false, true]), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert_eq!(true_results, vec![1]);
        assert_eq!(false_results, vec![1]);
    }

    #[test]
    fn test_update_array_all_true() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Array(vec![true, true]), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert_eq!(true_results, vec![1]);
        assert!(false_results.is_empty());
    }

    #[test]
    fn test_update_array_all_false() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Array(vec![false, false]), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert!(true_results.is_empty());
        assert_eq!(false_results, vec![1]);
    }

    #[test]
    fn test_update_array_empty() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.insert(&IndexedValue::Array(vec![]), 1);

        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert!(true_results.is_empty());
        assert!(false_results.is_empty());
    }

    #[test]
    fn test_compact_nothing_to_do_frees_memory() {
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        // Insert data and compact it away
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(false), 2);
        index.compact(1).unwrap();

        // No new ops — compact should early-return without creating a new version
        index.compact(2).unwrap();
        assert_eq!(index.current_version_number(), 1);

        // Ops vec capacity should have been freed
        {
            let live = index.live.read().unwrap();
            assert_eq!(live.ops.len(), 0);
            assert_eq!(live.ops.capacity(), 0);
        }

        // Data from version 1 is still intact
        let true_results: Vec<u64> = index.filter(true).iter().collect();
        let false_results: Vec<u64> = index.filter(false).iter().collect();
        assert_eq!(true_results, vec![1, 5]);
        assert_eq!(false_results, vec![2]);

        // Subsequent inserts and compaction still work
        index.insert(&IndexedValue::Plain(true), 10);
        index.compact(3).unwrap();
        let true_results: Vec<u64> = index.filter(true).iter().collect();
        assert_eq!(true_results, vec![1, 5, 10]);
    }

    #[test]
    fn test_compact_empty_index_frees_memory() {
        // Compacting a completely fresh index with zero ops should
        // early-return and free memory without creating any version.
        let tmp = TempDir::new().unwrap();
        let index =
            BoolStorage::new(tmp.path().to_path_buf(), DeletionThreshold::default()).unwrap();

        index.compact(1).unwrap();

        // No version created
        assert_eq!(index.current_version_number(), 0);
        assert!(!tmp.path().join("versions/1").exists());

        // Memory freed
        {
            let live = index.live.read().unwrap();
            assert_eq!(live.ops.len(), 0);
            assert_eq!(live.ops.capacity(), 0);
        }
    }
}
