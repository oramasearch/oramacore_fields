//! Memory-mapped compacted index for efficient range queries.
//!
//! This module provides the persistent storage layer used by `NumberStorage`
//! to store data on disk with memory-mapped access.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use super::error::Error;
use super::platform;
use super::IndexableNumber;

/// Compaction metadata stored in meta.bin.
///
/// Tracks the ratio of changes since the last full header rebuild,
/// used to decide when a full rebuild is needed to normalize header spacing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactionMeta {
    /// Number of inserts since last full header rebuild.
    pub changes_since_rebuild: u64,
    /// Total entry count when header was last fully rebuilt.
    pub total_at_rebuild: u64,
}

/// Statistics about a compacted version.
#[derive(Debug, Clone)]
pub struct CompactedVersionStats {
    /// Size of header.idx in bytes.
    pub header_size_bytes: u64,
    /// Size of deleted.bin in bytes.
    pub deleted_size_bytes: u64,
    /// Number of data files.
    pub data_file_count: usize,
    /// Total size of all data files in bytes.
    pub data_total_bytes: u64,
}

/// Default index stride: add header entry every ~1000 cumulative doc_ids.
pub const DEFAULT_INDEX_STRIDE: u32 = 1000;

/// Default bucket target size: 1GB.
pub const DEFAULT_BUCKET_TARGET_BYTES: usize = 1024 * 1024 * 1024;

/// Behavior when an exact key match is not found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyTolerance {
    /// Return the nearest key greater than the target.
    /// Use for "greater than" queries.
    NextHigher,
    /// Return the nearest key less than the target.
    /// Use for "less than" queries.
    NextLower,
}

/// A memory-mapped compacted index using header + data file format.
pub struct CompactedVersion<T: IndexableNumber> {
    /// Version offset (directory name).
    pub offset: u64,
    /// Memory-mapped header file.
    header: Option<HeaderFile<T>>,
    /// Memory-mapped data files.
    data: Vec<DataFile<T>>,
    /// Memory-mapped deleted file.
    deleted: Option<Mmap>,
    /// Pre-computed deleted set for O(1) lookup.
    deleted_set: Arc<HashSet<u64>>,
    /// Compaction metadata for incremental compaction.
    pub meta: CompactionMeta,
    /// Phantom marker for type parameter.
    _marker: PhantomData<T>,
}

impl<T: IndexableNumber> CompactedVersion<T> {
    /// Create an empty compacted index (for new indexes).
    pub fn empty() -> Self {
        Self {
            offset: 0,
            header: None,
            data: Vec::new(),
            deleted: None,
            deleted_set: Arc::new(HashSet::new()),
            meta: CompactionMeta::default(),
            _marker: PhantomData,
        }
    }

    /// Load a compacted index from disk.
    pub fn load(base_path: &Path, offset: u64) -> Result<Self, Error> {
        let version_dir = base_path.join("versions").join(offset.to_string());

        let header_path = version_dir.join("header.idx");
        let deleted_path = version_dir.join("deleted.bin");

        // Load header file
        let header = if header_path.exists() {
            let file = File::open(&header_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            platform::advise_random(&mmap);
            Some(HeaderFile::from_mmap(mmap))
        } else {
            None
        };

        // Load data files (discover by scanning directory)
        let mut data = Vec::new();
        for i in 0u32.. {
            let data_path = version_dir.join(format!("data_{i:04}.dat"));
            if !data_path.exists() {
                break;
            }
            let file = File::open(&data_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            platform::advise_sequential(&mmap);
            data.push(DataFile::from_mmap(mmap));
        }

        // Load deleted file
        let deleted = if deleted_path.exists() {
            let file = File::open(&deleted_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            platform::advise_sequential(&mmap);
            Some(mmap)
        } else {
            None
        };

        // Build deleted set once at load time
        let deleted_set = Arc::new(match &deleted {
            Some(mmap) => {
                let count = mmap.len() / 8;
                (0..count)
                    .map(|i| {
                        let start = i * 8;
                        let bytes: [u8; 8] = mmap[start..start + 8].try_into().unwrap();
                        u64::from_ne_bytes(bytes)
                    })
                    .collect()
            }
            None => HashSet::new(),
        });

        // Load compaction metadata
        let meta = read_meta_file(&version_dir);

        Ok(Self {
            offset,
            header,
            data,
            deleted,
            deleted_set,
            meta,
            _marker: PhantomData,
        })
    }

    /// Filter for exact key match, returning all matching doc_ids.
    pub fn filter_eq(&self, value: T) -> Vec<u64> {
        let Some(header) = &self.header else {
            return vec![];
        };

        let Some(entry) = header.find_key(value, KeyTolerance::NextLower) else {
            return vec![];
        };

        let bucket_index = entry.bucket_index as usize;
        if bucket_index >= self.data.len() {
            return vec![];
        }

        let bucket = &self.data[bucket_index];
        bucket.get_doc_ids_for_key(value, entry.bucket_offset as usize)
    }

    /// Get the total number of (key, doc_id) pairs.
    ///
    /// Note: This iterates all entries to compute the count.
    pub fn entry_count(&self) -> usize {
        self.iter().count()
    }

    /// Get the number of deleted doc_ids.
    pub fn deleted_count(&self) -> usize {
        match &self.deleted {
            Some(mmap) => mmap.len() / size_of::<u64>(),
            None => 0,
        }
    }

    /// Get statistics about the compacted version files.
    pub fn stats(&self) -> CompactedVersionStats {
        let header_size_bytes = self
            .header
            .as_ref()
            .map(|h| h.mmap.len() as u64)
            .unwrap_or(0);
        let deleted_size_bytes = self.deleted.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let data_file_count = self.data.len();
        let data_total_bytes = self.data.iter().map(|d| d.len() as u64).sum();

        CompactedVersionStats {
            header_size_bytes,
            deleted_size_bytes,
            data_file_count,
            data_total_bytes,
        }
    }

    /// Get the number of header entries.
    pub fn header_entry_count(&self) -> usize {
        match &self.header {
            Some(h) => h.mmap.len() / size_of::<HeaderEntry<T>>(),
            None => 0,
        }
    }

    /// Validate the integrity of the compacted version.
    ///
    /// Checks:
    /// - header.idx size is divisible by entry size (24 bytes)
    /// - deleted.bin size is divisible by 8 bytes
    ///
    /// Returns a list of (check_name, passed, details) tuples.
    pub fn validate(&self) -> Vec<(String, bool, Option<String>)> {
        let mut results = Vec::new();
        let header_entry_size = size_of::<HeaderEntry<T>>();

        // Check header alignment
        if let Some(header) = &self.header {
            let size = header.mmap.len();
            if size.is_multiple_of(header_entry_size) {
                let entries = size / header_entry_size;
                results.push((
                    "header.idx".to_string(),
                    true,
                    Some(format!("{size} bytes ({entries} entries)")),
                ));
            } else {
                results.push((
                    "header.idx".to_string(),
                    false,
                    Some(format!(
                        "{size} bytes (not divisible by {header_entry_size})"
                    )),
                ));
            }
        } else {
            results.push((
                "header.idx".to_string(),
                true,
                Some("0 bytes (empty)".to_string()),
            ));
        }

        // Check deleted alignment
        if let Some(deleted) = &self.deleted {
            let size = deleted.len();
            if size.is_multiple_of(8) {
                let count = size / 8;
                results.push((
                    "deleted.bin".to_string(),
                    true,
                    Some(format!("{size} bytes ({count} entries)")),
                ));
            } else {
                results.push((
                    "deleted.bin".to_string(),
                    false,
                    Some(format!("{size} bytes (not divisible by 8)")),
                ));
            }
        } else {
            results.push((
                "deleted.bin".to_string(),
                true,
                Some("0 bytes (empty)".to_string()),
            ));
        }

        // Check data files exist (0 files is valid for empty index)
        let data_count = self.data.len();
        if data_count == 0 {
            results.push((
                "data files".to_string(),
                true,
                Some("No data files (empty index)".to_string()),
            ));
        } else {
            let total_bytes: u64 = self.data.iter().map(|d| d.len() as u64).sum();
            results.push((
                "data files".to_string(),
                true,
                Some(format!("{data_count} file(s), total: {total_bytes} bytes")),
            ));
        }

        results
    }

    /// Get the deleted doc_ids as a Vec.
    pub fn deleted_slice(&self) -> Vec<u64> {
        match &self.deleted {
            Some(mmap) => {
                let count = mmap.len() / 8;
                (0..count)
                    .map(|i| {
                        let start = i * 8;
                        let bytes: [u8; 8] = mmap[start..start + 8].try_into().unwrap();
                        u64::from_ne_bytes(bytes)
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    /// Get the deleted doc_ids as an Arc<HashSet> for O(1) lookups.
    pub fn deleted_set(&self) -> Arc<HashSet<u64>> {
        Arc::clone(&self.deleted_set)
    }

    /// Iterate all entries as (key, doc_id) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (T, u64)> + '_ {
        self.iter_range(None, None)
    }

    /// Iterate entries in a value range [min, max].
    ///
    /// Both bounds are inclusive. Use None for unbounded.
    pub fn iter_range(
        &self,
        min: Option<T>,
        max: Option<T>,
    ) -> impl Iterator<Item = (T, u64)> + '_ {
        AscendingRangeIterator::new(self, min, max)
    }

    /// Iterate all entries in descending order by value.
    pub fn iter_descending(&self) -> impl Iterator<Item = (T, u64)> + '_ {
        ReverseRangeIterator::new(self)
    }

    /// Get all header entries as a slice (for reuse during incremental compaction).
    pub fn header_entries(&self) -> Vec<HeaderEntry<T>> {
        match &self.header {
            Some(h) => {
                let entries: &[HeaderEntry<T>] = unsafe { entries_from_mmap_unchecked(&h.mmap) };
                entries.to_vec()
            }
            None => Vec::new(),
        }
    }

    /// Get the number of data files (buckets).
    pub fn data_file_count(&self) -> usize {
        self.data.len()
    }

    /// Given a key range [min_key, max_key] from live inserts,
    /// return (first_affected_bucket, last_affected_bucket) inclusive.
    /// Returns None if no header/data or if the range covers ALL buckets.
    pub fn find_affected_bucket_range(&self, min_key: T, max_key: T) -> Option<(usize, usize)> {
        let header = self.header.as_ref()?;

        if self.data.is_empty() {
            return None;
        }

        // Find first affected bucket: where min_key would land
        let first = match header.find_key(min_key, KeyTolerance::NextLower) {
            Some(entry) => entry.bucket_index as usize,
            None => 0, // min_key is before all header entries, affects first bucket
        };

        // Find last affected bucket: where max_key would land
        let last = match header.find_key(max_key, KeyTolerance::NextHigher) {
            Some(entry) => entry.bucket_index as usize,
            None => self.data.len() - 1, // max_key is beyond all header entries, affects last bucket
        };

        // Clamp to valid range
        let first = first.min(self.data.len() - 1);
        let last = last.min(self.data.len() - 1);

        // If ALL buckets are affected, return None (no savings)
        if first == 0 && last == self.data.len() - 1 {
            return None;
        }

        Some((first, last))
    }

    /// Iterate all entries from bucket `first` through bucket `last` (inclusive).
    pub fn iter_bucket_range(
        &self,
        first: usize,
        last: usize,
    ) -> impl Iterator<Item = (T, u64)> + '_ {
        BucketRangeIterator::new(self, first, last)
    }
}

/// Memory-mapped data file containing entries.
pub struct DataFile<T: IndexableNumber> {
    mmap: Mmap,
    _marker: PhantomData<T>,
}

impl<T: IndexableNumber> DataFile<T> {
    /// Create a DataFile from a memory-mapped region.
    pub fn from_mmap(mmap: Mmap) -> Self {
        DataFile {
            mmap,
            _marker: PhantomData,
        }
    }

    /// Get doc_ids for a specific key, starting search from the given offset.
    pub fn get_doc_ids_for_key(&self, key: T, offset: usize) -> Vec<u64> {
        let mmap_len = self.mmap.len();
        let header_size = size_of::<DataEntryHeader<T>>();

        // Bounds check before accessing the header
        if offset + header_size > mmap_len {
            return vec![];
        }

        let data_at_offset = &self.mmap[offset..(offset + header_size)];
        let data_at_offset: &DataEntryHeader<T> =
            unsafe { &*(data_at_offset.as_ptr() as *const DataEntryHeader<T>) };

        match T::compare(data_at_offset.key, key) {
            Ordering::Equal => {
                self.extract_doc_ids(offset, data_at_offset.doc_id_count as usize, mmap_len)
            }
            Ordering::Less => {
                // data_at_offset.key < key, need to search FORWARD
                self.search_forward(key, offset, data_at_offset, mmap_len)
            }
            Ordering::Greater => {
                // data_at_offset.key > key, need to search BACKWARD
                self.search_backward(key, offset, data_at_offset, mmap_len)
            }
        }
    }

    /// Read a DataEntryHeader at the given offset.
    fn read_entry_header(&self, offset: usize) -> Option<&DataEntryHeader<T>> {
        let header_size = size_of::<DataEntryHeader<T>>();
        if offset + header_size > self.mmap.len() {
            return None;
        }

        let data = &self.mmap[offset..(offset + header_size)];
        Some(unsafe { &*(data.as_ptr() as *const DataEntryHeader<T>) })
    }

    fn extract_doc_ids(&self, offset: usize, count: usize, mmap_len: usize) -> Vec<u64> {
        self.extract_doc_ids_slice(offset, count, mmap_len).to_vec()
    }

    fn extract_doc_ids_slice(&self, offset: usize, count: usize, mmap_len: usize) -> &[u64] {
        let header_size = size_of::<DataEntryHeader<T>>();
        let doc_ids_start = offset + header_size;
        let Some(doc_ids_size) = count.checked_mul(size_of::<u64>()) else {
            return &[];
        };
        let Some(doc_ids_end) = doc_ids_start.checked_add(doc_ids_size) else {
            return &[];
        };

        // Bounds validation before unsafe access
        if doc_ids_end > mmap_len {
            return &[];
        }

        unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr().add(doc_ids_start) as *const u64, count)
        }
    }

    fn search_forward(
        &self,
        key: T,
        initial_offset: usize,
        initial_entry: &DataEntryHeader<T>,
        mmap_len: usize,
    ) -> Vec<u64> {
        let header_size = size_of::<DataEntryHeader<T>>();
        let mut current_offset = initial_offset;
        let mut next_offset = initial_entry.next_entry_offset as usize;

        // Loop while there's a valid next entry
        while next_offset > current_offset && next_offset + header_size <= mmap_len {
            let next_data = &self.mmap[next_offset..(next_offset + header_size)];
            let next_entry: &DataEntryHeader<T> =
                unsafe { &*(next_data.as_ptr() as *const DataEntryHeader<T>) };

            match T::compare(next_entry.key, key) {
                Ordering::Equal => {
                    return self.extract_doc_ids(
                        next_offset,
                        next_entry.doc_id_count as usize,
                        mmap_len,
                    );
                }
                Ordering::Less => {
                    // Keep searching forward
                    if next_entry.next_entry_offset as usize == 0
                        || next_entry.next_entry_offset as usize <= next_offset
                    {
                        break; // No more entries or invalid next pointer
                    }
                    current_offset = next_offset;
                    next_offset = next_entry.next_entry_offset as usize;
                }
                Ordering::Greater => {
                    // Overshot - key not found
                    break;
                }
            }
        }

        vec![]
    }

    fn search_backward(
        &self,
        key: T,
        initial_offset: usize,
        initial_entry: &DataEntryHeader<T>,
        mmap_len: usize,
    ) -> Vec<u64> {
        let header_size = size_of::<DataEntryHeader<T>>();
        let mut current_offset = initial_offset;
        let mut prev_offset = initial_entry.prev_entry_offset as usize;

        // Loop while there's a valid previous entry
        while prev_offset < current_offset && prev_offset + header_size <= mmap_len {
            let prev_data = &self.mmap[prev_offset..(prev_offset + header_size)];
            let prev_entry: &DataEntryHeader<T> =
                unsafe { &*(prev_data.as_ptr() as *const DataEntryHeader<T>) };

            match T::compare(prev_entry.key, key) {
                Ordering::Equal => {
                    return self.extract_doc_ids(
                        prev_offset,
                        prev_entry.doc_id_count as usize,
                        mmap_len,
                    );
                }
                Ordering::Greater => {
                    // Keep searching backward
                    if prev_entry.prev_entry_offset as usize >= prev_offset {
                        break; // No more entries or invalid prev pointer
                    }
                    current_offset = prev_offset;
                    prev_offset = prev_entry.prev_entry_offset as usize;
                }
                Ordering::Less => {
                    // Overshot - key not found
                    break;
                }
            }
        }

        vec![]
    }

    /// Get the length of the underlying mmap.
    fn len(&self) -> usize {
        self.mmap.len()
    }
}

/// Memory-mapped header file for binary search.
pub struct HeaderFile<T: IndexableNumber> {
    mmap: Mmap,
    _marker: PhantomData<T>,
}

impl<T: IndexableNumber> HeaderFile<T> {
    /// Create a HeaderFile from a memory-mapped region.
    pub fn from_mmap(mmap: Mmap) -> Self {
        HeaderFile {
            mmap,
            _marker: PhantomData,
        }
    }

    /// Find a key using binary search with tolerance.
    pub fn find_key(&self, key: T, tolerance: KeyTolerance) -> Option<HeaderEntry<T>> {
        let entries: &[HeaderEntry<T>] = unsafe { entries_from_mmap_unchecked(&self.mmap) };

        if entries.is_empty() {
            return None;
        }

        let res = entries.binary_search_by(|header_entry| T::compare(header_entry.key, key));

        match res {
            Ok(idx) => Some(entries[idx]),
            Err(idx) => match tolerance {
                KeyTolerance::NextHigher => {
                    if idx < entries.len() {
                        Some(entries[idx])
                    } else {
                        None
                    }
                }
                KeyTolerance::NextLower => {
                    if idx > 0 {
                        Some(entries[idx - 1])
                    } else {
                        None
                    }
                }
            },
        }
    }
}

unsafe fn entries_from_mmap_unchecked<T>(mmap: &Mmap) -> &[T] {
    transmute_slice(mmap)
}

unsafe fn transmute_slice<T>(slice: &[u8]) -> &[T] {
    assert!(slice.len().is_multiple_of(size_of::<T>()));
    assert!(
        (slice.as_ptr() as usize).is_multiple_of(align_of::<T>()),
        "Slice is not properly aligned for type T"
    );
    let ptr = slice.as_ptr() as *const T;
    let len = slice.len() / size_of::<T>();
    std::slice::from_raw_parts(ptr, len)
}

/// Header entry pointing to a position in a data file.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeaderEntry<T: IndexableNumber> {
    /// The indexed key.
    pub key: T,
    /// Which data file (bucket) this entry points to.
    pub bucket_index: u64,
    /// Byte offset in the data file.
    pub bucket_offset: u64,
}

/// Data entry header containing key and navigation pointers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DataEntryHeader<T: IndexableNumber> {
    /// The key for this entry.
    pub key: T,
    /// Byte offset to next entry (0 if last).
    pub next_entry_offset: u64,
    /// Byte offset to previous entry (0 if first).
    pub prev_entry_offset: u64,
    /// Number of doc_ids following this header.
    pub doc_id_count: u64,
}

/// Iterator over entries in ascending order within a range.
struct AscendingRangeIterator<'a, T: IndexableNumber> {
    index: &'a CompactedVersion<T>,
    max: Option<T>,
    current_bucket_id: usize,
    current_offset: usize,
    finished: bool,
    // Current entry state
    current_key: Option<T>,
    current_doc_ids: &'a [u64],
    current_doc_idx: usize,
}

impl<'a, T: IndexableNumber> AscendingRangeIterator<'a, T> {
    fn new(index: &'a CompactedVersion<T>, min: Option<T>, max: Option<T>) -> Self {
        // Find starting position using header
        let (bucket_id, offset) = if let Some(min_key) = min {
            if let Some(header) = &index.header {
                if let Some(entry) = header.find_key(min_key, KeyTolerance::NextLower) {
                    (entry.bucket_index as usize, entry.bucket_offset as usize)
                } else {
                    // No lower entry found, start from beginning
                    (0, 0)
                }
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };

        let mut iter = Self {
            index,
            max,
            current_bucket_id: bucket_id,
            current_offset: offset,
            finished: false,
            current_key: None,
            current_doc_ids: &[],
            current_doc_idx: 0,
        };

        // If we have a min key, seek forward to find the exact position
        if let Some(min_key) = min {
            iter.seek_to_key(min_key);
        } else {
            // Load first entry
            iter.load_next_entry();
        }

        iter
    }

    /// Seek forward to find the first entry with key >= target.
    fn seek_to_key(&mut self, target: T) {
        loop {
            if !self.load_next_entry() {
                self.finished = true;
                return;
            }

            if let Some(key) = self.current_key {
                match T::compare(key, target) {
                    Ordering::Less => {
                        // Skip this entry, continue to next
                        self.current_doc_idx = self.current_doc_ids.len();
                    }
                    Ordering::Equal | Ordering::Greater => {
                        // Found starting position
                        return;
                    }
                }
            }
        }
    }

    /// Load the next entry from the current data file.
    /// Returns false if no more entries.
    fn load_next_entry(&mut self) -> bool {
        loop {
            // Check if we have a valid bucket
            if self.current_bucket_id >= self.index.data.len() {
                return false;
            }

            let data_file = &self.index.data[self.current_bucket_id];
            let header_size = size_of::<DataEntryHeader<T>>();

            // Check if we've reached the end of this data file
            if self.current_offset + header_size > data_file.len() {
                // Move to next data file
                self.current_bucket_id += 1;
                self.current_offset = 0;
                continue;
            }

            // Read entry header
            let Some(entry_header) = data_file.read_entry_header(self.current_offset) else {
                return false;
            };

            let key = entry_header.key;
            let count = entry_header.doc_id_count as usize;

            // Extract doc_ids
            let doc_ids_start = self.current_offset + header_size;
            let Some(doc_ids_size) = count.checked_mul(size_of::<u64>()) else {
                return false;
            };
            let Some(doc_ids_end) = doc_ids_start.checked_add(doc_ids_size) else {
                return false;
            };

            if doc_ids_end > data_file.len() {
                return false;
            }

            let doc_ids =
                data_file.extract_doc_ids_slice(self.current_offset, count, data_file.len());

            // Update state
            self.current_key = Some(key);
            self.current_doc_ids = doc_ids;
            self.current_doc_idx = 0;

            // Move offset to next entry
            self.current_offset = doc_ids_end;

            return true;
        }
    }
}

impl<T: IndexableNumber> Iterator for AscendingRangeIterator<'_, T> {
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            // Check if we have doc_ids left in current entry
            if self.current_doc_idx < self.current_doc_ids.len() {
                let key = self.current_key?;
                let doc_id = self.current_doc_ids[self.current_doc_idx];
                self.current_doc_idx += 1;

                // Check max bound
                if let Some(max_key) = self.max {
                    if T::compare(key, max_key) == Ordering::Greater {
                        self.finished = true;
                        return None;
                    }
                }

                return Some((key, doc_id));
            }

            // Load next entry
            if !self.load_next_entry() {
                self.finished = true;
                return None;
            }

            // Check max bound for new entry
            if let Some(key) = self.current_key {
                if let Some(max_key) = self.max {
                    if T::compare(key, max_key) == Ordering::Greater {
                        self.finished = true;
                        return None;
                    }
                }
            }
        }
    }
}

/// Iterator over entries in descending order.
struct ReverseRangeIterator<'a, T: IndexableNumber> {
    index: &'a CompactedVersion<T>,
    current_bucket_id: isize,
    current_offset: usize,
    finished: bool,
    // Current entry state (doc_ids iterated in reverse)
    current_key: Option<T>,
    current_doc_ids: &'a [u64],
    current_doc_idx: usize,
}

impl<'a, T: IndexableNumber> ReverseRangeIterator<'a, T> {
    fn new(index: &'a CompactedVersion<T>) -> Self {
        if index.data.is_empty() {
            return Self {
                index,
                current_bucket_id: -1,
                current_offset: 0,
                finished: true,
                current_key: None,
                current_doc_ids: &[],
                current_doc_idx: 0,
            };
        }

        // Start from the last bucket
        let last_bucket_id = index.data.len() - 1;
        let last_offset = Self::find_last_entry_offset(&index.data[last_bucket_id]);

        let mut iter = Self {
            index,
            current_bucket_id: last_bucket_id as isize,
            current_offset: last_offset,
            finished: false,
            current_key: None,
            current_doc_ids: &[],
            current_doc_idx: 0,
        };

        // Load the last entry
        iter.load_current_entry();
        iter
    }

    /// Find the offset of the last entry in a data file by scanning forward.
    fn find_last_entry_offset(data_file: &DataFile<T>) -> usize {
        let header_size = size_of::<DataEntryHeader<T>>();
        let mut offset = 0;
        let mut last_offset = 0;

        while offset + header_size <= data_file.len() {
            if let Some(header) = data_file.read_entry_header(offset) {
                last_offset = offset;
                let Some(doc_ids_size) =
                    (header.doc_id_count as usize).checked_mul(size_of::<u64>())
                else {
                    break;
                };
                let Some(next_offset) = offset
                    .checked_add(header_size)
                    .and_then(|v| v.checked_add(doc_ids_size))
                else {
                    break;
                };
                if next_offset >= data_file.len() {
                    break;
                }
                offset = next_offset;
            } else {
                break;
            }
        }

        last_offset
    }

    /// Load the entry at current position.
    fn load_current_entry(&mut self) {
        if self.current_bucket_id < 0 || self.current_bucket_id as usize >= self.index.data.len() {
            self.finished = true;
            return;
        }

        let data_file = &self.index.data[self.current_bucket_id as usize];
        let header_size = size_of::<DataEntryHeader<T>>();

        if self.current_offset + header_size > data_file.len() {
            self.finished = true;
            return;
        }

        if let Some(header) = data_file.read_entry_header(self.current_offset) {
            let doc_ids = data_file.extract_doc_ids_slice(
                self.current_offset,
                header.doc_id_count as usize,
                data_file.len(),
            );

            self.current_key = Some(header.key);
            self.current_doc_ids = doc_ids;
            // Start from the END of doc_ids for descending order
            self.current_doc_idx = self.current_doc_ids.len();
        } else {
            self.finished = true;
        }
    }

    /// Move to the previous entry.
    fn load_prev_entry(&mut self) -> bool {
        if self.current_bucket_id < 0 {
            return false;
        }

        let data_file = &self.index.data[self.current_bucket_id as usize];

        if let Some(header) = data_file.read_entry_header(self.current_offset) {
            let prev_offset = header.prev_entry_offset as usize;

            if prev_offset < self.current_offset {
                // Move to previous entry in same bucket
                self.current_offset = prev_offset;
                self.load_current_entry();
                return !self.finished;
            } else if self.current_bucket_id > 0 {
                // Move to previous bucket (find its last entry)
                self.current_bucket_id -= 1;
                let prev_data_file = &self.index.data[self.current_bucket_id as usize];
                self.current_offset = Self::find_last_entry_offset(prev_data_file);
                self.load_current_entry();
                return !self.finished;
            }
        }

        false
    }
}

impl<T: IndexableNumber> Iterator for ReverseRangeIterator<'_, T> {
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            // Yield doc_ids in reverse order within entry
            if self.current_doc_idx > 0 {
                self.current_doc_idx -= 1;
                let key = self.current_key?;
                let doc_id = self.current_doc_ids[self.current_doc_idx];
                return Some((key, doc_id));
            }

            // Move to previous entry
            if !self.load_prev_entry() {
                self.finished = true;
                return None;
            }
        }
    }
}

/// Iterator over entries in a contiguous range of buckets.
struct BucketRangeIterator<'a, T: IndexableNumber> {
    index: &'a CompactedVersion<T>,
    last_bucket_id: usize,
    current_bucket_id: usize,
    current_offset: usize,
    finished: bool,
    current_key: Option<T>,
    current_doc_ids: &'a [u64],
    current_doc_idx: usize,
}

impl<'a, T: IndexableNumber> BucketRangeIterator<'a, T> {
    fn new(index: &'a CompactedVersion<T>, first: usize, last: usize) -> Self {
        let finished = first >= index.data.len() || last >= index.data.len();
        let mut iter = Self {
            index,
            last_bucket_id: last,
            current_bucket_id: first,
            current_offset: 0,
            finished,
            current_key: None,
            current_doc_ids: &[],
            current_doc_idx: 0,
        };

        if !iter.finished {
            iter.load_next_entry();
        }

        iter
    }

    fn load_next_entry(&mut self) -> bool {
        loop {
            if self.current_bucket_id > self.last_bucket_id {
                return false;
            }

            if self.current_bucket_id >= self.index.data.len() {
                return false;
            }

            let data_file = &self.index.data[self.current_bucket_id];
            let header_size = size_of::<DataEntryHeader<T>>();

            if self.current_offset + header_size > data_file.len() {
                self.current_bucket_id += 1;
                self.current_offset = 0;
                continue;
            }

            let Some(entry_header) = data_file.read_entry_header(self.current_offset) else {
                return false;
            };

            let key = entry_header.key;
            let count = entry_header.doc_id_count as usize;

            let doc_ids_start = self.current_offset + header_size;
            let Some(doc_ids_size) = count.checked_mul(size_of::<u64>()) else {
                return false;
            };
            let Some(doc_ids_end) = doc_ids_start.checked_add(doc_ids_size) else {
                return false;
            };

            if doc_ids_end > data_file.len() {
                return false;
            }

            let doc_ids =
                data_file.extract_doc_ids_slice(self.current_offset, count, data_file.len());

            self.current_key = Some(key);
            self.current_doc_ids = doc_ids;
            self.current_doc_idx = 0;
            self.current_offset = doc_ids_end;

            return true;
        }
    }
}

impl<T: IndexableNumber> Iterator for BucketRangeIterator<'_, T> {
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            if self.current_doc_idx < self.current_doc_ids.len() {
                let key = self.current_key?;
                let doc_id = self.current_doc_ids[self.current_doc_idx];
                self.current_doc_idx += 1;
                return Some((key, doc_id));
            }

            if !self.load_next_entry() {
                self.finished = true;
                return None;
            }
        }
    }
}

// ============================================================================
// Write functionality
// ============================================================================

/// Write a new compacted version with configurable index stride and bucket size.
///
/// Returns the total number of (key, doc_id) entries written.
pub fn write_version_with_config<T: IndexableNumber>(
    version_dir: &Path,
    entries: impl Iterator<Item = (T, u64)>,
    deleted: &[u64],
    index_stride: u32,
    bucket_target_bytes: usize,
) -> Result<u64, Error> {
    // Use Peekable to check for empty without consuming
    let mut peekable = entries.peekable();

    if peekable.peek().is_none() {
        write_empty_version(version_dir, deleted)?;
        return Ok(0);
    }

    // Write data files (with inline grouping) and collect header entries
    let (header_entries, total_entry_count) =
        write_data_files(version_dir, peekable, index_stride, bucket_target_bytes)?;

    // Write header file
    write_header_file::<T>(version_dir, &header_entries)?;

    // Write deleted.bin
    write_deleted_file(version_dir, deleted)?;

    Ok(total_entry_count)
}

/// Write data files from a raw (key, doc_id) iterator and return header entries + total entry count.
///
/// Groups consecutive equal keys inline, avoiding an O(N) pre-grouping allocation.
fn write_data_files<T: IndexableNumber>(
    version_dir: &Path,
    entries: impl Iterator<Item = (T, u64)>,
    index_stride: u32,
    bucket_target_bytes: usize,
) -> Result<(Vec<HeaderEntry<T>>, u64), Error> {
    let mut header_entries = Vec::new();
    let mut current_bucket_id: u64 = 0;
    let mut current_bucket_path = version_dir.join(format!("data_{current_bucket_id:04}.dat"));
    let mut current_bucket_data: Vec<u8> = Vec::new();

    let mut cumulative_doc_count: u64 = 0;
    let header_size = size_of::<DataEntryHeader<T>>();

    // Track previous entry offset for linking
    let mut prev_entry_offset: u64 = 0;
    let mut prev_entry_patch_offset: Option<usize> = None;

    let mut total_entry_count: u64 = 0;

    // Inline grouping state
    let mut current_key: Option<T> = None;
    let mut current_doc_ids: Vec<u64> = Vec::new();

    // Closure to flush a grouped (key, doc_ids) into the bucket buffer.
    // Handles header entry creation and bucket splitting.
    let flush_group = |key: T,
                       doc_ids: &[u64],
                       header_entries: &mut Vec<HeaderEntry<T>>,
                       current_bucket_data: &mut Vec<u8>,
                       cumulative_doc_count: &mut u64,
                       prev_entry_offset: &mut u64,
                       prev_entry_patch_offset: &mut Option<usize>,
                       current_bucket_id: &mut u64,
                       current_bucket_path: &mut PathBuf,
                       total_entry_count: &mut u64|
     -> Result<(), Error> {
        let doc_count = doc_ids.len() as u64;
        *cumulative_doc_count += doc_count;
        *total_entry_count += doc_count;

        let entry_start = current_bucket_data.len();
        let doc_ids_bytes = doc_count as usize * size_of::<u64>();
        let entry_size = header_size + doc_ids_bytes;

        // Check if we should add a header entry (sparse index)
        if *cumulative_doc_count >= index_stride as u64 || header_entries.is_empty() {
            header_entries.push(HeaderEntry {
                key,
                bucket_index: *current_bucket_id,
                bucket_offset: entry_start as u64,
            });
            *cumulative_doc_count = 0;
        }

        // Check if bucket is full (target size exceeded)
        if current_bucket_data.len() + entry_size > bucket_target_bytes
            && !current_bucket_data.is_empty()
        {
            write_bucket_file(current_bucket_path, current_bucket_data)?;

            *current_bucket_id += 1;
            *current_bucket_path = version_dir.join(format!("data_{:04}.dat", *current_bucket_id));
            current_bucket_data.clear();

            *prev_entry_offset = 0;
            *prev_entry_patch_offset = None;

            if let Some(last_entry) = header_entries.last_mut() {
                if T::compare(last_entry.key, key) == Ordering::Equal {
                    last_entry.bucket_index = *current_bucket_id;
                    last_entry.bucket_offset = 0;
                }
            }
        }

        let entry_offset = current_bucket_data.len();

        if let Some(patch_offset) = *prev_entry_patch_offset {
            let next_entry_bytes = (entry_offset as u64).to_ne_bytes();
            current_bucket_data[patch_offset..patch_offset + 8].copy_from_slice(&next_entry_bytes);
        }

        // Write DataEntryHeader
        current_bucket_data.extend_from_slice(&key.to_bytes());
        let next_patch = current_bucket_data.len();
        current_bucket_data.extend_from_slice(&0u64.to_ne_bytes());
        current_bucket_data.extend_from_slice(&(*prev_entry_offset).to_ne_bytes());
        current_bucket_data.extend_from_slice(&doc_count.to_ne_bytes());

        for &doc_id in doc_ids {
            current_bucket_data.extend_from_slice(&doc_id.to_ne_bytes());
        }

        *prev_entry_offset = entry_offset as u64;
        *prev_entry_patch_offset = Some(next_patch);

        Ok(())
    };

    for (key, doc_id) in entries {
        match current_key {
            Some(cur) if T::compare(cur, key) == Ordering::Equal => {
                current_doc_ids.push(doc_id);
            }
            _ => {
                if let Some(cur) = current_key {
                    flush_group(
                        cur,
                        &current_doc_ids,
                        &mut header_entries,
                        &mut current_bucket_data,
                        &mut cumulative_doc_count,
                        &mut prev_entry_offset,
                        &mut prev_entry_patch_offset,
                        &mut current_bucket_id,
                        &mut current_bucket_path,
                        &mut total_entry_count,
                    )?;
                    current_doc_ids.clear();
                }
                current_key = Some(key);
                current_doc_ids.push(doc_id);
            }
        }
    }

    // Flush the last group
    if let Some(cur) = current_key {
        flush_group(
            cur,
            &current_doc_ids,
            &mut header_entries,
            &mut current_bucket_data,
            &mut cumulative_doc_count,
            &mut prev_entry_offset,
            &mut prev_entry_patch_offset,
            &mut current_bucket_id,
            &mut current_bucket_path,
            &mut total_entry_count,
        )?;
    }

    // Write last bucket
    if !current_bucket_data.is_empty() {
        write_bucket_file(&current_bucket_path, &current_bucket_data)?;
    }

    Ok((header_entries, total_entry_count))
}

/// Write a bucket file to disk.
fn write_bucket_file(path: &Path, data: &[u8]) -> Result<(), Error> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    file.write_all(data)?;
    file.sync_all()?;

    Ok(())
}

/// Write the header file (array of HeaderEntry<T>).
pub fn write_header_file<T: IndexableNumber>(
    version_dir: &Path,
    header_entries: &[HeaderEntry<T>],
) -> Result<(), Error> {
    let header_path = version_dir.join("header.idx");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&header_path)?;

    // Write header entries directly (no file header, just raw entries)
    for entry in header_entries {
        file.write_all(&entry.key.to_bytes())?;
        file.write_all(&entry.bucket_index.to_ne_bytes())?;
        file.write_all(&entry.bucket_offset.to_ne_bytes())?;
    }

    file.sync_all()?;
    Ok(())
}

/// Write the deleted file.
pub fn write_deleted_file(version_dir: &Path, deleted: &[u64]) -> Result<(), Error> {
    let deleted_path = version_dir.join("deleted.bin");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&deleted_path)?;

    for &doc_id in deleted {
        file.write_all(&doc_id.to_ne_bytes())?;
    }

    file.sync_all()?;
    Ok(())
}

/// Write an empty version (no entries).
fn write_empty_version(version_dir: &Path, deleted: &[u64]) -> Result<(), Error> {
    // Write empty header file
    let header_path = version_dir.join("header.idx");
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&header_path)?
        .sync_all()?;

    // Write deleted file
    write_deleted_file(version_dir, deleted)?;

    Ok(())
}

/// Copy a data file from old version dir to new version dir.
pub fn copy_bucket_file(
    src_dir: &Path,
    src_idx: usize,
    dst_dir: &Path,
    dst_idx: usize,
) -> Result<(), Error> {
    let src_path = src_dir.join(format!("data_{src_idx:04}.dat"));
    let dst_path = dst_dir.join(format!("data_{dst_idx:04}.dat"));
    std::fs::copy(&src_path, &dst_path)?;
    Ok(())
}

/// Copy header.idx from src to dst directory.
pub fn copy_header_file(src_dir: &Path, dst_dir: &Path) -> Result<(), Error> {
    let src_path = src_dir.join("header.idx");
    let dst_path = dst_dir.join("header.idx");
    std::fs::copy(&src_path, &dst_path)?;
    Ok(())
}

/// Write entries to a single data file at the given bucket index.
/// Does NOT split into multiple files. Returns new header entries for this bucket.
pub fn write_single_bucket<T: IndexableNumber>(
    version_dir: &Path,
    entries: impl Iterator<Item = (T, u64)>,
    index_stride: u32,
    bucket_id: u64,
) -> Result<Vec<HeaderEntry<T>>, Error> {
    let mut header_entries = Vec::new();
    let mut current_bucket_data: Vec<u8> = Vec::new();

    let mut cumulative_doc_count: u64 = 0;
    let mut prev_entry_offset: u64 = 0;
    let mut prev_entry_patch_offset: Option<usize> = None;

    let mut current_key: Option<T> = None;
    let mut current_doc_ids: Vec<u64> = Vec::new();

    let flush_group = |key: T,
                       doc_ids: &[u64],
                       header_entries: &mut Vec<HeaderEntry<T>>,
                       current_bucket_data: &mut Vec<u8>,
                       cumulative_doc_count: &mut u64,
                       prev_entry_offset: &mut u64,
                       prev_entry_patch_offset: &mut Option<usize>| {
        let doc_count = doc_ids.len() as u64;
        *cumulative_doc_count += doc_count;

        let entry_start = current_bucket_data.len();

        if *cumulative_doc_count >= index_stride as u64 || header_entries.is_empty() {
            header_entries.push(HeaderEntry {
                key,
                bucket_index: bucket_id,
                bucket_offset: entry_start as u64,
            });
            *cumulative_doc_count = 0;
        }

        let entry_offset = current_bucket_data.len();

        // Patch previous entry's next_entry_offset
        if let Some(patch_offset) = *prev_entry_patch_offset {
            let next_entry_bytes = (entry_offset as u64).to_ne_bytes();
            current_bucket_data[patch_offset..patch_offset + 8].copy_from_slice(&next_entry_bytes);
        }

        // Write DataEntryHeader
        current_bucket_data.extend_from_slice(&key.to_bytes());
        let next_entry_patch_offset_val = current_bucket_data.len();
        current_bucket_data.extend_from_slice(&0u64.to_ne_bytes()); // next_entry_offset placeholder
        current_bucket_data.extend_from_slice(&(*prev_entry_offset).to_ne_bytes()); // prev_entry_offset
        current_bucket_data.extend_from_slice(&doc_count.to_ne_bytes()); // doc_id_count

        // Write doc_ids
        for &doc_id in doc_ids {
            current_bucket_data.extend_from_slice(&doc_id.to_ne_bytes());
        }

        *prev_entry_offset = entry_offset as u64;
        *prev_entry_patch_offset = Some(next_entry_patch_offset_val);
    };

    for (key, doc_id) in entries {
        match current_key {
            Some(cur) if T::compare(cur, key) == std::cmp::Ordering::Equal => {
                current_doc_ids.push(doc_id);
            }
            _ => {
                if let Some(cur) = current_key {
                    flush_group(
                        cur,
                        &current_doc_ids,
                        &mut header_entries,
                        &mut current_bucket_data,
                        &mut cumulative_doc_count,
                        &mut prev_entry_offset,
                        &mut prev_entry_patch_offset,
                    );
                    current_doc_ids.clear();
                }
                current_key = Some(key);
                current_doc_ids.push(doc_id);
            }
        }
    }

    // Flush the last group
    if let Some(cur) = current_key {
        flush_group(
            cur,
            &current_doc_ids,
            &mut header_entries,
            &mut current_bucket_data,
            &mut cumulative_doc_count,
            &mut prev_entry_offset,
            &mut prev_entry_patch_offset,
        );
    }

    if !current_bucket_data.is_empty() {
        let bucket_path = version_dir.join(format!("data_{bucket_id:04}.dat"));
        write_bucket_file(&bucket_path, &current_bucket_data)?;
    }

    Ok(header_entries)
}

/// Write compaction metadata to meta.bin.
pub fn write_meta_file(version_dir: &Path, meta: &CompactionMeta) -> Result<(), Error> {
    let meta_path = version_dir.join("meta.bin");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&meta_path)?;

    file.write_all(&meta.changes_since_rebuild.to_ne_bytes())?;
    file.write_all(&meta.total_at_rebuild.to_ne_bytes())?;
    file.sync_all()?;

    Ok(())
}

/// Read compaction metadata from meta.bin. Returns default if missing.
fn read_meta_file(version_dir: &Path) -> CompactionMeta {
    let meta_path = version_dir.join("meta.bin");
    if !meta_path.exists() {
        return CompactionMeta::default();
    }

    let Ok(data) = std::fs::read(&meta_path) else {
        return CompactionMeta::default();
    };

    if data.len() < 16 {
        return CompactionMeta::default();
    }

    let changes_since_rebuild = u64::from_ne_bytes(data[0..8].try_into().unwrap());
    let total_at_rebuild = u64::from_ne_bytes(data[8..16].try_into().unwrap());

    CompactionMeta {
        changes_since_rebuild,
        total_at_rebuild,
    }
}

/// Get the path to the version directory.
pub fn version_dir(base_path: &Path, offset: u64) -> PathBuf {
    base_path.join("versions").join(offset.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write as IoWrite;
    use tempfile::{tempdir, TempDir};

    // Helper to create a memory-mapped file from bytes
    fn create_mmap_from_bytes(bytes: &[u8]) -> Mmap {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.bin");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();

        let file = File::open(&file_path).unwrap();
        unsafe { Mmap::map(&file).unwrap() }
    }

    // Helper to serialize a DataEntryHeader to bytes
    fn serialize_data_entry<T: IndexableNumber>(entry: &DataEntryHeader<T>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of::<DataEntryHeader<T>>());
        bytes.extend_from_slice(&entry.key.to_bytes());
        bytes.extend_from_slice(&entry.next_entry_offset.to_ne_bytes());
        bytes.extend_from_slice(&entry.prev_entry_offset.to_ne_bytes());
        bytes.extend_from_slice(&entry.doc_id_count.to_ne_bytes());
        bytes
    }

    // Helper to serialize a HeaderEntry to bytes
    fn serialize_header_entry<T: IndexableNumber>(entry: &HeaderEntry<T>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of::<HeaderEntry<T>>());
        bytes.extend_from_slice(&entry.key.to_bytes());
        bytes.extend_from_slice(&entry.bucket_index.to_ne_bytes());
        bytes.extend_from_slice(&entry.bucket_offset.to_ne_bytes());
        bytes
    }

    // ==================== DataFile Tests ====================

    #[test]
    fn test_get_doc_ids_exact_match() {
        // Create a single data entry with key=100 and 2 doc IDs
        let entry = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 2,
        };
        let mut bytes = serialize_data_entry(&entry);
        bytes.extend_from_slice(&1u64.to_ne_bytes()); // doc_id 1
        bytes.extend_from_slice(&2u64.to_ne_bytes()); // doc_id 2

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        let result = data_file.get_doc_ids_for_key(100, 0);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_get_doc_ids_not_found() {
        // Create a single data entry with key=100
        let entry = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry);
        bytes.extend_from_slice(&1u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Search for non-existent key
        let result = data_file.get_doc_ids_for_key(200, 0);
        assert_eq!(result, Vec::<u64>::new());
    }

    #[test]
    fn test_get_doc_ids_empty_result() {
        // Create a data entry with count=0
        let entry = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 0,
        };
        let bytes = serialize_data_entry(&entry);

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        let result = data_file.get_doc_ids_for_key(100, 0);
        assert_eq!(result, Vec::<u64>::new());
    }

    #[test]
    fn test_get_doc_ids_multiple_docs() {
        // Create a data entry with 5 doc IDs
        let entry = DataEntryHeader::<u64> {
            key: 42,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 5,
        };
        let mut bytes = serialize_data_entry(&entry);
        for i in 10..15 {
            bytes.extend_from_slice(&(i as u64).to_ne_bytes());
        }

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        let result = data_file.get_doc_ids_for_key(42, 0);
        assert_eq!(result, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_get_doc_ids_forward_search() {
        // Create two entries: key=50, then key=100
        // Entry 0 at offset 0
        let header_size = size_of::<DataEntryHeader<u64>>();
        let entry0_size = header_size + 8; // 1 doc_id

        let entry0 = DataEntryHeader::<u64> {
            key: 50,
            next_entry_offset: entry0_size as u64, // points to entry1
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry0);
        bytes.extend_from_slice(&100u64.to_ne_bytes()); // doc_id for entry0

        // Entry 1 at offset entry0_size
        let entry1 = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0, // points back to entry0
            doc_id_count: 1,
        };
        bytes.extend(serialize_data_entry(&entry1));
        bytes.extend_from_slice(&200u64.to_ne_bytes()); // doc_id for entry1

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Start at entry0 (key=50), search for key=100 (should go forward)
        let result = data_file.get_doc_ids_for_key(100, 0);
        assert_eq!(result, vec![200]);
    }

    #[test]
    fn test_get_doc_ids_backward_search() {
        // Create two entries: key=50, then key=100
        let header_size = size_of::<DataEntryHeader<u64>>();
        let entry0_size = header_size + 8; // 1 doc_id

        // Entry 0 at offset 0
        let entry0 = DataEntryHeader::<u64> {
            key: 50,
            next_entry_offset: entry0_size as u64,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry0);
        bytes.extend_from_slice(&100u64.to_ne_bytes()); // doc_id for entry0

        // Entry 1 at offset entry0_size
        let entry1 = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0, // points back to entry0
            doc_id_count: 1,
        };
        bytes.extend(serialize_data_entry(&entry1));
        bytes.extend_from_slice(&200u64.to_ne_bytes()); // doc_id for entry1

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Start at entry1 (key=100), search for key=50 (should go backward)
        let result = data_file.get_doc_ids_for_key(50, entry0_size);
        assert_eq!(result, vec![100]);
    }

    #[test]
    fn test_get_doc_ids_out_of_bounds() {
        let entry = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry);
        bytes.extend_from_slice(&1u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Invalid offset beyond mmap
        let result = data_file.get_doc_ids_for_key(100, 1000);
        assert_eq!(result, Vec::<u64>::new());
    }

    // ==================== HeaderFile Tests ====================

    #[test]
    fn test_find_key_exact() {
        let entries = vec![
            HeaderEntry::<u64> {
                key: 10,
                bucket_index: 0,
                bucket_offset: 0,
            },
            HeaderEntry::<u64> {
                key: 20,
                bucket_index: 0,
                bucket_offset: 100,
            },
            HeaderEntry::<u64> {
                key: 30,
                bucket_index: 1,
                bucket_offset: 0,
            },
        ];

        let mut bytes = Vec::new();
        for entry in &entries {
            bytes.extend(serialize_header_entry(entry));
        }

        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        // Exact match
        let result = header_file.find_key(20, KeyTolerance::NextLower);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 20);
        assert_eq!(result.unwrap().bucket_offset, 100);
    }

    #[test]
    fn test_find_key_next_lower() {
        let entries = vec![
            HeaderEntry::<u64> {
                key: 10,
                bucket_index: 0,
                bucket_offset: 0,
            },
            HeaderEntry::<u64> {
                key: 20,
                bucket_index: 0,
                bucket_offset: 100,
            },
            HeaderEntry::<u64> {
                key: 30,
                bucket_index: 1,
                bucket_offset: 0,
            },
        ];

        let mut bytes = Vec::new();
        for entry in &entries {
            bytes.extend(serialize_header_entry(entry));
        }

        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        // Search for 25 with NextLower tolerance - should return 20
        let result = header_file.find_key(25, KeyTolerance::NextLower);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 20);
    }

    #[test]
    fn test_find_key_next_higher() {
        let entries = vec![
            HeaderEntry::<u64> {
                key: 10,
                bucket_index: 0,
                bucket_offset: 0,
            },
            HeaderEntry::<u64> {
                key: 20,
                bucket_index: 0,
                bucket_offset: 100,
            },
            HeaderEntry::<u64> {
                key: 30,
                bucket_index: 1,
                bucket_offset: 0,
            },
        ];

        let mut bytes = Vec::new();
        for entry in &entries {
            bytes.extend(serialize_header_entry(entry));
        }

        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        // Search for 25 with NextHigher tolerance - should return 30
        let result = header_file.find_key(25, KeyTolerance::NextHigher);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 30);
    }

    #[test]
    fn test_find_key_not_found() {
        let entries = vec![
            HeaderEntry::<u64> {
                key: 10,
                bucket_index: 0,
                bucket_offset: 0,
            },
            HeaderEntry::<u64> {
                key: 20,
                bucket_index: 0,
                bucket_offset: 100,
            },
        ];

        let mut bytes = Vec::new();
        for entry in &entries {
            bytes.extend(serialize_header_entry(entry));
        }

        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        // Search for 5 with NextLower - no lower key exists
        let result = header_file.find_key(5, KeyTolerance::NextLower);
        assert!(result.is_none());

        // Search for 25 with NextHigher - no higher key exists
        let result = header_file.find_key(25, KeyTolerance::NextHigher);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_key_empty_header() {
        let bytes: Vec<u8> = Vec::new();
        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        let result = header_file.find_key(10, KeyTolerance::NextLower);
        assert!(result.is_none());

        let result = header_file.find_key(10, KeyTolerance::NextHigher);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_key_single_entry() {
        let entry = HeaderEntry::<u64> {
            key: 50,
            bucket_index: 0,
            bucket_offset: 0,
        };

        let bytes = serialize_header_entry(&entry);
        let mmap = create_mmap_from_bytes(&bytes);
        let header_file = HeaderFile::<u64>::from_mmap(mmap);

        // Exact match
        let result = header_file.find_key(50, KeyTolerance::NextLower);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 50);

        // NextLower for higher key
        let result = header_file.find_key(100, KeyTolerance::NextLower);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 50);

        // NextHigher for lower key
        let result = header_file.find_key(10, KeyTolerance::NextHigher);
        assert!(result.is_some());
        assert_eq!(result.unwrap().key, 50);
    }

    // ==================== f64 Tests ====================

    #[test]
    fn test_filter_eq_f64() {
        let entry = DataEntryHeader::<f64> {
            key: std::f64::consts::PI,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 2,
        };
        let mut bytes = serialize_data_entry(&entry);
        bytes.extend_from_slice(&10u64.to_ne_bytes());
        bytes.extend_from_slice(&20u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<f64>::from_mmap(mmap);

        let result = data_file.get_doc_ids_for_key(std::f64::consts::PI, 0);
        assert_eq!(result, vec![10, 20]);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_get_doc_ids_boundary_keys() {
        // Test with u64::MAX
        let entry = DataEntryHeader::<u64> {
            key: u64::MAX,
            next_entry_offset: 0,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry);
        bytes.extend_from_slice(&999u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        let result = data_file.get_doc_ids_for_key(u64::MAX, 0);
        assert_eq!(result, vec![999]);
    }

    #[test]
    fn test_forward_search_terminates_on_overshoot() {
        // Create entries: key=50, key=100, key=150
        // Search for key=75 starting at 50 - should terminate when hitting 100
        let header_size = size_of::<DataEntryHeader<u64>>();
        let entry0_size = header_size + 8;

        let entry0 = DataEntryHeader::<u64> {
            key: 50,
            next_entry_offset: entry0_size as u64,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry0);
        bytes.extend_from_slice(&100u64.to_ne_bytes());

        let entry1 = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: (entry0_size * 2) as u64,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        bytes.extend(serialize_data_entry(&entry1));
        bytes.extend_from_slice(&200u64.to_ne_bytes());

        let entry2 = DataEntryHeader::<u64> {
            key: 150,
            next_entry_offset: 0,
            prev_entry_offset: entry0_size as u64,
            doc_id_count: 1,
        };
        bytes.extend(serialize_data_entry(&entry2));
        bytes.extend_from_slice(&300u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Search for 75 starting at entry0 - should not find it
        let result = data_file.get_doc_ids_for_key(75, 0);
        assert_eq!(result, Vec::<u64>::new());
    }

    #[test]
    fn test_backward_search_terminates_on_overshoot() {
        // Create entries: key=50, key=100
        // Search for key=75 starting at 100 - should terminate when hitting 50
        let header_size = size_of::<DataEntryHeader<u64>>();
        let entry0_size = header_size + 8;

        let entry0 = DataEntryHeader::<u64> {
            key: 50,
            next_entry_offset: entry0_size as u64,
            prev_entry_offset: 0,
            doc_id_count: 1,
        };
        let mut bytes = serialize_data_entry(&entry0);
        bytes.extend_from_slice(&100u64.to_ne_bytes());

        let entry1 = DataEntryHeader::<u64> {
            key: 100,
            next_entry_offset: 0,
            prev_entry_offset: 0, // points to entry0
            doc_id_count: 1,
        };
        bytes.extend(serialize_data_entry(&entry1));
        bytes.extend_from_slice(&200u64.to_ne_bytes());

        let mmap = create_mmap_from_bytes(&bytes);
        let data_file = DataFile::<u64>::from_mmap(mmap);

        // Search for 75 starting at entry1 - should not find it
        let result = data_file.get_doc_ids_for_key(75, entry0_size);
        assert_eq!(result, Vec::<u64>::new());
    }

    // ==================== CompactedVersion Tests ====================

    #[test]
    fn test_empty_index() {
        let index: CompactedVersion<u64> = CompactedVersion::empty();
        assert_eq!(index.entry_count(), 0);
        assert_eq!(index.deleted_count(), 0);
        assert_eq!(index.iter().count(), 0);
        assert_eq!(index.filter_eq(100), Vec::<u64>::new());
    }

    #[test]
    fn test_write_and_load() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Write entries
        let entries = vec![(10u64, 1), (20, 2), (30, 3)];
        let deleted = vec![100u64, 200];
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &deleted,
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        // Load and verify
        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        assert_eq!(index.entry_count(), 3);
        assert_eq!(index.deleted_count(), 2);

        let loaded_entries: Vec<_> = index.iter().collect();
        assert_eq!(loaded_entries, entries);

        let loaded_deleted = index.deleted_slice();
        assert_eq!(loaded_deleted, deleted);
    }

    #[test]
    fn test_iter_range() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Write entries
        let entries = vec![(10u64, 1), (20, 2), (30, 3), (40, 4), (50, 5)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Range query [20, 40]
        let range: Vec<_> = index.iter_range(Some(20), Some(40)).collect();
        assert_eq!(range, vec![(20, 2), (30, 3), (40, 4)]);

        // Range query [30, None]
        let range: Vec<_> = index.iter_range(Some(30), None).collect();
        assert_eq!(range, vec![(30, 3), (40, 4), (50, 5)]);

        // Range query [None, 30]
        let range: Vec<_> = index.iter_range(None, Some(30)).collect();
        assert_eq!(range, vec![(10, 1), (20, 2), (30, 3)]);

        // Range query [None, None] (all)
        let range: Vec<_> = index.iter_range(None, None).collect();
        assert_eq!(range.len(), 5);
    }

    #[test]
    fn test_f64_entries() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries = vec![(-1.0f64, 1), (0.0, 2), (1.5, 3), (std::f64::consts::PI, 4)];
        write_version_with_config::<f64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<f64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let loaded: Vec<_> = index.iter().collect();
        assert_eq!(loaded.len(), 4);

        // Check values (using approximate comparison for floats)
        for (i, (v, d)) in loaded.iter().enumerate() {
            assert!((v - entries[i].0).abs() < f64::EPSILON);
            assert_eq!(*d, entries[i].1);
        }
    }

    #[test]
    fn test_filter_eq_with_index() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Write entries with multiple docs per key
        let entries = vec![(10u64, 1), (10, 2), (10, 3), (20, 4), (20, 5), (30, 6)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Filter for key=10
        let result = index.filter_eq(10);
        assert_eq!(result, vec![1, 2, 3]);

        // Filter for key=20
        let result = index.filter_eq(20);
        assert_eq!(result, vec![4, 5]);

        // Filter for non-existent key
        let result = index.filter_eq(100);
        assert_eq!(result, Vec::<u64>::new());
    }

    #[test]
    fn test_grouped_entries() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Multiple doc_ids for same key
        let entries = vec![(10u64, 1), (10, 2), (10, 3), (20, 4), (20, 5), (30, 6)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        let loaded: Vec<_> = index.iter().collect();
        assert_eq!(loaded, entries);

        // Query just key=10
        let range: Vec<_> = index.iter_range(Some(10), Some(10)).collect();
        assert_eq!(range, vec![(10, 1), (10, 2), (10, 3)]);
    }

    #[test]
    fn test_binary_search_with_many_entries() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Write many entries for binary search testing
        let entries: Vec<_> = (0..1000u64).map(|i| (i * 10, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Query for value 5000 should find entries starting at index 500
        let range: Vec<_> = index.iter_range(Some(5000), Some(5020)).collect();
        assert_eq!(range, vec![(5000, 500), (5010, 501), (5020, 502)]);
    }

    #[test]
    fn test_empty_write() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Write empty entries with some deleted
        let entries: Vec<(u64, u64)> = vec![];
        let deleted = vec![1u64, 2, 3];
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &deleted,
            DEFAULT_INDEX_STRIDE,
            DEFAULT_BUCKET_TARGET_BYTES,
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        assert_eq!(index.entry_count(), 0);
        assert_eq!(index.deleted_count(), 3);
        assert_eq!(index.deleted_slice(), vec![1, 2, 3]);
    }

    // ==================== Bucket Split Tests ====================

    // Helper to count bucket files in a version directory
    fn count_bucket_files(version_dir: &Path) -> usize {
        let mut count = 0;
        for i in 0u32.. {
            let data_path = version_dir.join(format!("data_{i:04}.dat"));
            if data_path.exists() {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    // ==================== Category 1: Split Threshold Tests ====================

    #[test]
    fn test_bucket_split_at_exact_threshold() {
        // Entry size = 32 bytes (header) + 8 bytes per doc_id
        // Single entry with 1 doc_id = 40 bytes
        // Use bucket_target_bytes = 50 to trigger split after first entry
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Insert 3 entries with 1 doc_id each (40 bytes each)
        let entries = vec![(10u64, 1), (20, 2), (30, 3)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            1,  // index_stride = 1 to create header entry for each key
            50, // bucket_target_bytes = 50 (less than 2 entries)
        )
        .unwrap();

        // Should have 3 bucket files (one entry per bucket)
        let bucket_count = count_bucket_files(&version_dir);
        assert_eq!(bucket_count, 3);

        // Verify data integrity
        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, vec![(10, 1), (20, 2), (30, 3)]);
    }

    #[test]
    fn test_bucket_split_respects_non_empty_check() {
        // Even if a single entry exceeds bucket_target_bytes, it should be written
        // The check is: current_bucket_data.len() + entry_size > bucket_target_bytes
        //              AND !current_bucket_data.is_empty()
        // So the first entry is always written regardless of size
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Entry with 10 doc_ids = 32 + 80 = 112 bytes
        // Use bucket_target_bytes = 50 (smaller than single entry)
        let entries = vec![
            (10u64, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
        ];
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            50, // Much smaller than the single grouped entry
        )
        .unwrap();

        // Should have exactly 1 bucket (first entry is always written)
        let bucket_count = count_bucket_files(&version_dir);
        assert_eq!(bucket_count, 1);

        // Verify data integrity
        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, entries);
    }

    // ==================== Category 2: Data Integrity ====================

    #[test]
    fn test_data_integrity_across_bucket_boundary() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Create enough entries to span multiple buckets
        let entries: Vec<_> = (1..=20u64).map(|i| (i * 10, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            100, // Small bucket size
        )
        .unwrap();

        // Verify multiple buckets were created
        let bucket_count = count_bucket_files(&version_dir);
        assert!(
            bucket_count > 1,
            "Expected multiple buckets, got {bucket_count}"
        );

        // Verify ALL data is readable
        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, entries);

        // Verify each entry individually via filter_eq
        for (key, doc_id) in &entries {
            let result = index.filter_eq(*key);
            assert_eq!(result, vec![*doc_id], "Failed for key {key}");
        }
    }

    #[test]
    fn test_grouped_entries_across_buckets() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Multiple doc_ids for the same key should stay grouped
        let entries = vec![
            (10u64, 1),
            (10, 2),
            (10, 3),
            (20, 4),
            (20, 5),
            (30, 6),
            (30, 7),
            (30, 8),
        ];
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            100, // Small bucket to force splits
        )
        .unwrap();

        // Verify grouped entries are intact
        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        assert_eq!(index.filter_eq(10), vec![1, 2, 3]);
        assert_eq!(index.filter_eq(20), vec![4, 5]);
        assert_eq!(index.filter_eq(30), vec![6, 7, 8]);

        // Verify total count
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, entries);
    }

    // ==================== Category 3: Range Queries ====================

    #[test]
    fn test_range_query_spanning_multiple_buckets() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries: Vec<_> = (1..=10u64).map(|i| (i * 10, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            1,
            80, // Force splits approximately every 2 entries
        )
        .unwrap();

        // Verify multiple buckets
        let bucket_count = count_bucket_files(&version_dir);
        assert!(bucket_count > 1);

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Range query spanning multiple buckets
        let range: Vec<_> = index.iter_range(Some(30), Some(70)).collect();
        assert_eq!(range, vec![(30, 3), (40, 4), (50, 5), (60, 6), (70, 7)]);
    }

    #[test]
    fn test_range_query_starting_mid_bucket() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries: Vec<_> = (1..=10u64).map(|i| (i * 10, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            1,
            200, // Larger bucket to have multiple entries per bucket
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Query starting at a key that's not at bucket boundary
        let range: Vec<_> = index.iter_range(Some(45), Some(75)).collect();
        assert_eq!(range, vec![(50, 5), (60, 6), (70, 7)]);

        // Query for exact key in middle
        let range: Vec<_> = index.iter_range(Some(50), Some(50)).collect();
        assert_eq!(range, vec![(50, 5)]);
    }

    // ==================== Category 4: Header Index ====================

    #[test]
    fn test_header_index_points_to_correct_bucket() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // Create entries that will span multiple buckets
        let entries: Vec<_> = (1..=5u64).map(|i| (i * 100, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            1,  // Header entry for each key
            50, // One entry per bucket
        )
        .unwrap();

        // Should have 5 buckets
        let bucket_count = count_bucket_files(&version_dir);
        assert_eq!(bucket_count, 5);

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Each key should be findable via filter_eq (uses header lookup)
        assert_eq!(index.filter_eq(100), vec![1]);
        assert_eq!(index.filter_eq(200), vec![2]);
        assert_eq!(index.filter_eq(300), vec![3]);
        assert_eq!(index.filter_eq(400), vec![4]);
        assert_eq!(index.filter_eq(500), vec![5]);
    }

    #[test]
    fn test_header_entry_updated_on_bucket_boundary() {
        // When an entry crosses bucket boundary, the header should point to new bucket
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries = vec![(10u64, 1), (20, 2), (30, 3)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.into_iter(),
            &[],
            1,  // Header for each entry
            50, // Force each entry to new bucket
        )
        .unwrap();

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Verify header correctly points to each entry
        // If header wasn't updated, filter_eq would fail
        assert_eq!(index.filter_eq(10), vec![1]);
        assert_eq!(index.filter_eq(20), vec![2]);
        assert_eq!(index.filter_eq(30), vec![3]);
    }

    // ==================== Category 5: Edge Cases ====================

    #[test]
    fn test_single_entry_per_bucket() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries: Vec<_> = (1..=5u64).map(|i| (i * 10, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            40, // Exactly one entry per bucket
        )
        .unwrap();

        assert_eq!(count_bucket_files(&version_dir), 5);

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, entries);
    }

    #[test]
    fn test_large_entry_with_many_doc_ids() {
        // A single entry with many doc_ids exceeding bucket size
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        // 100 doc_ids = 32 + 800 = 832 bytes for one grouped entry
        let entries: Vec<_> = (1..=100u64).map(|i| (42u64, i)).collect();
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            100, // Much smaller than entry size
        )
        .unwrap();

        // Should still work - entry written even though it exceeds bucket size
        let bucket_count = count_bucket_files(&version_dir);
        assert_eq!(bucket_count, 1); // Single grouped entry

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let result = index.filter_eq(42);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_doubly_linked_list_bucket_local_only() {
        // prev/next pointers should reset at bucket boundary
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries = vec![(10u64, 1), (20, 2), (30, 3), (40, 4)];
        write_version_with_config::<u64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            50, // One entry per bucket
        )
        .unwrap();

        // 4 buckets
        assert_eq!(count_bucket_files(&version_dir), 4);

        let index: CompactedVersion<u64> = CompactedVersion::load(temp.path(), 1).unwrap();

        // Verify each entry is independently accessible
        // (prev/next pointers within bucket don't cross boundaries)
        for (key, doc_id) in &entries {
            assert_eq!(index.filter_eq(*key), vec![*doc_id]);
        }

        // Also verify full iteration works
        let all: Vec<_> = index.iter().collect();
        assert_eq!(all, entries);
    }

    #[test]
    fn test_empty_bucket_never_created() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries = vec![(10u64, 1), (20, 2)];
        write_version_with_config::<u64>(&version_dir, entries.into_iter(), &[], 1, 50).unwrap();

        // Verify no empty bucket files
        for i in 0u32..10 {
            let data_path = version_dir.join(format!("data_{i:04}.dat"));
            if data_path.exists() {
                let metadata = std::fs::metadata(&data_path).unwrap();
                assert!(metadata.len() > 0, "Bucket {i} should not be empty");
            }
        }
    }

    #[test]
    fn test_bucket_split_with_f64_keys() {
        let temp = TempDir::new().unwrap();
        let version_dir = temp.path().join("versions").join("1");
        std::fs::create_dir_all(&version_dir).unwrap();

        let entries = vec![(-100.5f64, 1), (-50.0, 2), (0.0, 3), (50.5, 4), (100.0, 5)];
        write_version_with_config::<f64>(
            &version_dir,
            entries.clone().into_iter(),
            &[],
            1,
            50, // Force splits
        )
        .unwrap();

        let bucket_count = count_bucket_files(&version_dir);
        assert!(bucket_count > 1);

        let index: CompactedVersion<f64> = CompactedVersion::load(temp.path(), 1).unwrap();
        let all: Vec<_> = index.iter().collect();

        // Verify f64 ordering is preserved
        assert_eq!(all.len(), entries.len());
        for (i, (v, d)) in all.iter().enumerate() {
            assert!((v - entries[i].0).abs() < f64::EPSILON);
            assert_eq!(*d, entries[i].1);
        }

        // Verify filter_eq works
        assert_eq!(index.filter_eq(0.0), vec![3]);
        assert_eq!(index.filter_eq(50.5), vec![4]);
    }
}
