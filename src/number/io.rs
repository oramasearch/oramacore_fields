//! Atomic I/O operations for NumberStorage.

use super::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Current file format version.
pub const FORMAT_VERSION: u32 = 1;

/// Ensure the version directory exists and return its path.
pub fn ensure_version_dir(base_path: &Path, offset: u64) -> Result<PathBuf, Error> {
    let versions_dir = base_path.join("versions");
    fs::create_dir_all(&versions_dir)?;

    let version_dir = versions_dir.join(offset.to_string());
    fs::create_dir_all(&version_dir)?;

    Ok(version_dir)
}

/// Write the CURRENT file atomically using temp file + rename.
pub fn write_current_atomic(base_path: &Path, offset: u64) -> Result<(), Error> {
    let current_path = base_path.join("CURRENT");
    let tmp_path = base_path.join("CURRENT.tmp");

    // Write to temp file
    let mut file = File::create(&tmp_path)?;
    writeln!(file, "{FORMAT_VERSION}")?;
    write!(file, "{offset}")?;
    file.sync_all()?;

    // Atomic rename
    fs::rename(&tmp_path, &current_path)?;

    // Sync parent directory for durability
    sync_dir(base_path)?;

    Ok(())
}

/// Read the CURRENT file and return the format version and version offset.
///
/// Returns None if the file doesn't exist.
pub fn read_current(base_path: &Path) -> Result<Option<(u32, u64)>, Error> {
    let current_path = base_path.join("CURRENT");

    if !current_path.exists() {
        return Ok(None);
    }

    let file = File::open(&current_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read format version
    let version_line = lines.next().ok_or(Error::CorruptedEntry)??;
    let format_version: u32 = version_line
        .trim()
        .parse()
        .map_err(|_| Error::CorruptedEntry)?;

    // Read offset
    let offset_line = lines.next().ok_or(Error::CorruptedEntry)??;
    let offset: u64 = offset_line
        .trim()
        .parse()
        .map_err(|_| Error::CorruptedEntry)?;

    Ok(Some((format_version, offset)))
}

/// Sync a directory to ensure durability.
pub fn sync_dir(path: &Path) -> Result<(), Error> {
    // On Unix, we can sync the directory
    #[cfg(unix)]
    {
        let dir = OpenOptions::new().read(true).open(path)?;
        dir.sync_all()?;
    }

    // On non-Unix, this is a no-op
    #[cfg(not(unix))]
    {
        let _ = path;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_ensure_version_dir() {
        let temp = TempDir::new().unwrap();
        let base_path = temp.path();

        let version_dir = ensure_version_dir(base_path, 42).unwrap();
        assert!(version_dir.exists());
        assert_eq!(version_dir, base_path.join("versions").join("42"));
    }

    #[test]
    fn test_write_and_read_current() {
        let temp = TempDir::new().unwrap();
        let base_path = temp.path();

        // Initially no CURRENT file
        assert_eq!(read_current(base_path).unwrap(), None);

        // Write CURRENT
        write_current_atomic(base_path, 123).unwrap();

        // Read back
        assert_eq!(
            read_current(base_path).unwrap(),
            Some((FORMAT_VERSION, 123))
        );

        // Update CURRENT
        write_current_atomic(base_path, 456).unwrap();
        assert_eq!(
            read_current(base_path).unwrap(),
            Some((FORMAT_VERSION, 456))
        );
    }
}
