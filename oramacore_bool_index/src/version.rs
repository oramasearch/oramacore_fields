//! Compacted version with memory-mapped file handles.
//!
//! This module provides read-only access to compacted index versions stored on disk.
//! Each version is a snapshot of the index at a particular point in time, containing:
//! - `true.bin`: sorted doc_ids where the boolean value is true
//! - `false.bin`: sorted doc_ids where the boolean value is false
//! - `deleted.bin`: doc_ids that should be excluded from both sets
//!
//! Files are memory-mapped for efficient random access without loading into heap.
//! The kernel manages paging data in/out as needed.

use crate::io::version_dir;
use crate::platform::advise_sequential;
use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A compacted version containing memory-mapped postings files.
///
/// This struct holds memory-mapped views of the three postings files that make up
/// a compacted version. The `Mmap` handles keep the files mapped for the lifetime
/// of this struct.
///
/// # Thread Safety
///
/// `CompactedVersion` is `Send` and `Sync` because `Mmap` is read-only after creation.
/// Multiple threads can safely read from the same mapped regions concurrently.
/// The index uses `ArcSwap<CompactedVersion>` to allow lock-free version swaps
/// during compaction while readers continue using the old version.
pub struct CompactedVersion {
    /// The version number of this compacted version.
    pub version_number: u64,
    /// Memory-mapped file containing sorted doc_ids with value=true.
    pub true_postings: Option<Mmap>,
    /// Memory-mapped file containing sorted doc_ids with value=false.
    pub false_postings: Option<Mmap>,
    /// Memory-mapped file containing doc_ids to exclude from results.
    pub deletes: Option<Mmap>,
}

impl CompactedVersion {
    /// Create an empty version with no data.
    pub fn empty() -> Self {
        Self {
            version_number: 0,
            true_postings: None,
            false_postings: None,
            deletes: None,
        }
    }

    /// Load a compacted version from disk.
    ///
    /// Memory-maps the postings files in the version directory. Missing files
    /// are treated as empty (returns `None` for that field).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Files exist but cannot be opened (permission denied)
    /// - Memory mapping fails (out of address space, file too large)
    ///
    /// # Complexity
    ///
    /// O(1) - memory mapping is lazy, actual I/O happens on access.
    pub fn load(base_path: &Path, version_number: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_number);

        let true_postings = Self::load_mmap(&dir.join("true.bin"))?;
        let false_postings = Self::load_mmap(&dir.join("false.bin"))?;
        let deletes = Self::load_mmap(&dir.join("deleted.bin"))?;

        Ok(Self {
            version_number,
            true_postings,
            false_postings,
            deletes,
        })
    }

    /// Load a memory-mapped file, returning None if the file doesn't exist or is empty.
    ///
    /// # Safety
    ///
    /// The `unsafe` block in `Mmap::map` is safe because:
    /// - The file is opened read-only, so no concurrent modifications from this process
    /// - The index design ensures files are never modified after creation (immutable versions)
    /// - The `Mmap` lifetime is tied to `CompactedVersion`, ensuring the mapping stays valid
    fn load_mmap(path: &Path) -> Result<Option<Mmap>> {
        if !path.exists() {
            return Ok(None);
        }

        let file =
            File::open(path).with_context(|| format!("Failed to open file for mmap: {path:?}"))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get file metadata: {path:?}"))?;

        if metadata.len() == 0 {
            return Ok(None);
        }

        let mmap =
            unsafe { Mmap::map(&file).with_context(|| format!("Failed to mmap file: {path:?}"))? };

        advise_sequential(&mmap);

        Ok(Some(mmap))
    }

    /// Get the true postings as a slice of u64.
    ///
    /// Returns an empty slice if the true.bin file was missing or empty.
    /// The returned slice is valid for the lifetime of this `CompactedVersion`.
    pub fn true_postings_slice(&self) -> &[u64] {
        Self::mmap_as_u64_slice(self.true_postings.as_ref())
    }

    /// Get the false postings as a slice of u64.
    ///
    /// Returns an empty slice if the false.bin file was missing or empty.
    /// The returned slice is valid for the lifetime of this `CompactedVersion`.
    pub fn false_postings_slice(&self) -> &[u64] {
        Self::mmap_as_u64_slice(self.false_postings.as_ref())
    }

    /// Get postings for a given boolean value.
    pub fn postings_slice(&self, value: bool) -> &[u64] {
        if value {
            self.true_postings_slice()
        } else {
            self.false_postings_slice()
        }
    }

    /// Get the deletes entries as a slice of u64 (doc_ids to delete from both sets).
    pub fn deletes_slice(&self) -> &[u64] {
        Self::mmap_as_u64_slice(self.deletes.as_ref())
    }

    /// Convert an mmap to a u64 slice.
    ///
    /// # Safety
    ///
    /// The `unsafe` block is safe because:
    ///
    /// 1. **Alignment**: `Mmap` returns page-aligned memory (OS pages are always >= 8 bytes
    ///    aligned), so casting to `*const u64` is valid. Even if the file size is not a
    ///    multiple of 8, we only read `len / 8` elements.
    ///
    /// 2. **Initialization**: The postings files are written as sequences of native-endian
    ///    u64 values by `write_postings()` (using `to_ne_bytes()`). The native-endian format
    ///    matches the pointer cast to `*const u64`, so no byte-swapping is needed.
    ///    The bytes are always valid u64 representations.
    ///
    /// 3. **Lifetime**: The returned slice borrows from `&self`, and `self` owns the `Mmap`.
    ///    The mapping (and thus the underlying memory) remains valid for the slice's lifetime.
    ///
    /// 4. **Aliasing**: We only create shared (`&`) references. The file is opened read-only
    ///    and is never modified after creation (immutable version design).
    ///
    /// 5. **Size**: `len / 8` ensures we don't read past the end. Any trailing bytes (< 8)
    ///    from a corrupted file are ignored (defensive, shouldn't happen with valid data).
    fn mmap_as_u64_slice(mmap: Option<&Mmap>) -> &[u64] {
        match mmap {
            Some(m) => {
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                // SAFETY: See function-level safety documentation above.
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{ensure_version_dir, write_postings};
    use tempfile::TempDir;

    #[test]
    fn test_empty_version() {
        let version = CompactedVersion::empty();
        assert_eq!(version.version_number, 0);
        assert!(version.true_postings_slice().is_empty());
        assert!(version.false_postings_slice().is_empty());
        assert!(version.deletes_slice().is_empty());
    }

    #[test]
    fn test_load_version() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Create version directory and files
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        let true_postings = vec![1u64, 5, 10];
        let false_postings = vec![2u64, 6, 11];
        let deleted = vec![100u64, 200];

        write_postings(&version_path.join("true.bin"), &true_postings).unwrap();
        write_postings(&version_path.join("false.bin"), &false_postings).unwrap();
        write_postings(&version_path.join("deleted.bin"), &deleted).unwrap();

        // Load and verify
        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert_eq!(version.true_postings_slice(), &true_postings);
        assert_eq!(version.false_postings_slice(), &false_postings);
        assert_eq!(version.deletes_slice(), &deleted);
    }

    #[test]
    fn test_load_version_missing_file() {
        // Test loading a version where one of the binary files doesn't exist
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Create version directory but only write some files (not all)
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        let true_postings = vec![1u64, 5, 10];
        write_postings(&version_path.join("true.bin"), &true_postings).unwrap();
        // Intentionally skip false.bin and deleted.bin

        // Load should succeed, missing files return None/empty slices
        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert_eq!(version.true_postings_slice(), &true_postings);
        assert!(version.false_postings_slice().is_empty());
        assert!(version.deletes_slice().is_empty());
    }
}
