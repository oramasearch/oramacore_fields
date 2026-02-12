//! File I/O for reading and writing postings files and managing versions.

use anyhow::{anyhow, Context, Result};

/// Current format version for the index structure.
/// Increment this when making incompatible changes to the file format.
pub const FORMAT_VERSION: u32 = 1;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Copy a postings file and optionally append new sorted values.
///
/// All values in `new_values` must be strictly greater than `existing_max`.
///
/// # Errors
///
/// Returns an error on I/O failure (read, write, or sync).
pub fn copy_and_append_postings(
    src: &Path,
    dst: &Path,
    existing_max: Option<u64>,
    new_values: &[u64],
) -> Result<()> {
    // Debug assert invariant: new values must be > existing max
    // The caller ensures safety via can_append_safely() with merge fallback
    #[cfg(debug_assertions)]
    if let Some(max) = existing_max {
        if let Some(&first_new) = new_values.first() {
            debug_assert!(
                first_new > max,
                "Invariant violated: new insert {first_new} <= existing max {max}"
            );
        }
    }

    if src.exists() && src.metadata()?.len() > 0 {
        fs::copy(src, dst).with_context(|| format!("Failed to copy {src:?} to {dst:?}"))?;

        // If we won't append (which would sync), sync the copy explicitly
        if new_values.is_empty() {
            File::open(dst)
                .with_context(|| format!("Failed to open {dst:?} for sync"))?
                .sync_all()
                .with_context(|| format!("Failed to sync {dst:?}"))?;
        }
    }

    if !new_values.is_empty() {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dst)
            .with_context(|| format!("Failed to open {dst:?} for append"))?;
        let mut writer = BufWriter::new(file);
        for &v in new_values {
            writer
                .write_all(&v.to_ne_bytes())
                .with_context(|| format!("Failed to append to {dst:?}"))?;
        }
        writer
            .into_inner()
            .map_err(|e| e.into_error())
            .with_context(|| format!("Failed to flush buffer for {dst:?}"))?
            .sync_all()
            .with_context(|| format!("Failed to sync {dst:?}"))?;
    } else if !dst.exists() {
        File::create(dst)
            .with_context(|| format!("Failed to create empty file {dst:?}"))?
            .sync_all()
            .with_context(|| format!("Failed to sync {dst:?}"))?;
    }

    Ok(())
}

/// Write u64 values from an iterator to a file in native-endian format.
///
/// Creates or truncates the file. Returns an error on I/O failure.
pub fn write_postings_from_iter<I>(path: &Path, iter: I) -> Result<()>
where
    I: Iterator<Item = u64>,
{
    let file =
        File::create(path).with_context(|| format!("Failed to create postings file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for v in iter {
        writer
            .write_all(&v.to_ne_bytes())
            .with_context(|| format!("Failed to write postings to: {path:?}"))?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync postings file: {path:?}"))?;

    Ok(())
}

/// Write a slice of u64 values to a file in native-endian format.
///
/// Creates or truncates the file. Returns an error on I/O failure.
pub fn write_postings(path: &Path, postings: &[u64]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create postings file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for &v in postings {
        writer
            .write_all(&v.to_ne_bytes())
            .with_context(|| format!("Failed to write postings to: {path:?}"))?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync postings file: {path:?}"))?;

    Ok(())
}

/// Read the CURRENT file to get the format version and version number.
///
/// Returns `Ok(None)` if the file doesn't exist (new index).
/// Returns an error if the file exists but is malformed or unreadable.
pub fn read_current(base_path: &Path) -> Result<Option<(u32, u64)>> {
    let current_path = base_path.join("CURRENT");

    if !current_path.exists() {
        return Ok(None);
    }

    let contents =
        std::fs::read_to_string(&current_path).with_context(|| "Failed to read CURRENT file")?;

    let mut lines = contents.lines();

    let format_version: u32 = lines
        .next()
        .ok_or_else(|| anyhow!("CURRENT file is empty"))?
        .trim()
        .parse()
        .with_context(|| {
            format!("Failed to parse format version from CURRENT file: {contents:?}")
        })?;

    let version_number: u64 = lines
        .next()
        .ok_or_else(|| anyhow!("CURRENT file missing version number line"))?
        .trim()
        .parse()
        .with_context(|| {
            format!("Failed to parse version number from CURRENT file: {contents:?}")
        })?;

    Ok(Some((format_version, version_number)))
}

/// Atomically write the CURRENT file with format version and version number.
///
/// Uses write-to-temp-then-rename for atomicity. On error, the CURRENT
/// file is unchanged.
pub fn write_current_atomic(base_path: &Path, version_number: u64) -> Result<()> {
    let current_path = base_path.join("CURRENT");
    let tmp_path = base_path.join("CURRENT.tmp");

    let mut file = File::create(&tmp_path)
        .with_context(|| format!("Failed to create CURRENT.tmp: {tmp_path:?}"))?;

    writeln!(file, "{FORMAT_VERSION}")
        .with_context(|| "Failed to write format version to CURRENT.tmp")?;
    write!(file, "{version_number}")
        .with_context(|| "Failed to write version number to CURRENT.tmp")?;

    file.sync_all()
        .with_context(|| "Failed to sync CURRENT.tmp")?;

    // Atomically rename to CURRENT
    fs::rename(&tmp_path, &current_path)
        .with_context(|| "Failed to atomically rename CURRENT.tmp to CURRENT")?;

    // Sync parent directory to ensure rename is durable (best-effort).
    // We don't propagate errors here because the rename already succeeded —
    // reporting failure would mislead the caller into thinking CURRENT wasn't updated.
    if let Ok(canonical) = base_path.canonicalize() {
        if let Ok(dir) = OpenOptions::new().read(true).open(&canonical) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

/// Get the path to a version directory.
///
/// Version directories are stored as `{base_path}/versions/{version_number}/`.
/// Each version directory contains: `true.bin`, `false.bin`, `deleted.bin`.
pub fn version_dir(base_path: &Path, version_number: u64) -> std::path::PathBuf {
    base_path.join("versions").join(version_number.to_string())
}

/// Ensure the versions directory and specific version subdirectory exist.
///
/// Creates `{base_path}/versions/{version_number}/` and all parent directories as needed.
///
/// # Errors
///
/// Returns an error if directory creation fails (permission denied, disk full).
pub fn ensure_version_dir(base_path: &Path, version_number: u64) -> Result<std::path::PathBuf> {
    let dir = version_dir(base_path, version_number);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create version directory: {dir:?}"))?;
    Ok(dir)
}

/// Sync a directory to ensure file operations are durable.
pub fn sync_dir(path: &Path) -> Result<()> {
    let dir = OpenOptions::new()
        .read(true)
        .open(path)
        .with_context(|| format!("Failed to open directory for sync: {path:?}"))?;

    dir.sync_all()
        .with_context(|| format!("Failed to sync directory: {path:?}"))?;

    Ok(())
}

/// List all version numbers found in the versions directory.
pub fn list_version_dirs(base_path: &Path) -> Result<impl Iterator<Item = u64> + '_> {
    let versions_dir = base_path.join("versions");

    let entries = fs::read_dir(&versions_dir)
        .with_context(|| format!("Failed to read versions directory: {versions_dir:?}"))?;

    let iter = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let file_type = entry.file_type().ok()?;
            if !file_type.is_dir() {
                return None;
            }
            Some(entry.file_name())
        })
        .filter_map(|name| name.to_str()?.parse::<u64>().ok());

    Ok(iter)
}

/// Remove a version directory and all its contents.
pub fn remove_version_dir(base_path: &Path, version_number: u64) -> Result<()> {
    let dir = version_dir(base_path, version_number);
    fs::remove_dir_all(&dir).with_context(|| format!("Failed to remove version directory: {dir:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_and_read_current() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Initially no CURRENT file
        assert_eq!(read_current(base_path).unwrap(), None);

        // Write and read back
        write_current_atomic(base_path, 42).unwrap();
        assert_eq!(read_current(base_path).unwrap(), Some((FORMAT_VERSION, 42)));

        // Update
        write_current_atomic(base_path, 100).unwrap();
        assert_eq!(
            read_current(base_path).unwrap(),
            Some((FORMAT_VERSION, 100))
        );
    }

    #[test]
    fn test_write_postings() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.bin");

        let postings = vec![1u64, 10, 100, 1000];
        write_postings(&path, &postings).unwrap();

        // Read back and verify
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), postings.len() * 8);

        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        assert_eq!(read_back, postings);
    }

    #[test]
    fn test_version_dir() {
        let base = Path::new("/base");
        assert_eq!(version_dir(base, 123), Path::new("/base/versions/123"));
    }

    #[test]
    fn test_copy_and_append_postings_empty_src() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");

        // Source doesn't exist, append new values
        let new_values = vec![10u64, 20, 30];
        copy_and_append_postings(&src, &dst, None, &new_values).unwrap();

        let bytes = std::fs::read(&dst).unwrap();
        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(read_back, new_values);
    }

    #[test]
    fn test_copy_and_append_postings_with_existing() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");

        // Create source file
        let existing = vec![1u64, 5, 10];
        write_postings(&src, &existing).unwrap();

        // Copy and append
        let new_values = vec![20u64, 30];
        copy_and_append_postings(&src, &dst, Some(10), &new_values).unwrap();

        let bytes = std::fs::read(&dst).unwrap();
        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(read_back, vec![1, 5, 10, 20, 30]);
    }

    #[test]
    fn test_copy_and_append_postings_copy_only() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");

        // Create source file
        let existing = vec![1u64, 5, 10];
        write_postings(&src, &existing).unwrap();

        // Copy without appending
        copy_and_append_postings(&src, &dst, Some(10), &[]).unwrap();

        let bytes = std::fs::read(&dst).unwrap();
        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(read_back, existing);
    }

    #[test]
    fn test_copy_and_append_postings_empty_creates_file() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");

        // Both empty - should still create dst
        copy_and_append_postings(&src, &dst, None, &[]).unwrap();
        assert!(dst.exists());
        assert_eq!(std::fs::read(&dst).unwrap().len(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Invariant violated")]
    fn test_copy_and_append_postings_invariant_violation() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");

        // Violate invariant: new value <= existing max
        // Note: This only panics in debug builds due to debug_assert!
        let new_values = vec![5u64]; // 5 <= 10
        copy_and_append_postings(&src, &dst, Some(10), &new_values).unwrap();
    }
}
