use anyhow::{anyhow, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const FORMAT_VERSION: u32 = 1;

/// Write a slice of u64 values to a file.
pub fn write_postings(path: &Path, postings: &[u64]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create postings file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for &v in postings {
        writer
            .write_all(&v.to_le_bytes())
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

/// Write u64 values from an iterator to a file.
pub fn write_postings_from_iter<I>(path: &Path, iter: I) -> Result<()>
where
    I: Iterator<Item = u64>,
{
    let file =
        File::create(path).with_context(|| format!("Failed to create postings file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for v in iter {
        writer
            .write_all(&v.to_le_bytes())
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

/// Read the format version and version number from the CURRENT file.
pub fn read_current(base_path: &Path) -> Result<Option<(u32, u64)>> {
    let current_path = base_path.join("CURRENT");

    if !current_path.exists() {
        return Ok(None);
    }

    let contents =
        fs::read_to_string(&current_path).with_context(|| "Failed to read CURRENT file")?;

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

/// Write the CURRENT file atomically with the format version and version number.
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

    fs::rename(&tmp_path, &current_path)
        .with_context(|| "Failed to atomically rename CURRENT.tmp to CURRENT")?;

    if let Ok(canonical) = base_path.canonicalize() {
        if let Some(parent) = canonical.parent() {
            if let Ok(dir) = OpenOptions::new().read(true).open(parent) {
                let _ = dir.sync_all();
            }
        }
    }

    Ok(())
}

pub fn version_dir(base_path: &Path, version_number: u64) -> std::path::PathBuf {
    base_path.join("versions").join(version_number.to_string())
}

pub fn ensure_version_dir(base_path: &Path, version_number: u64) -> Result<std::path::PathBuf> {
    let dir = version_dir(base_path, version_number);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create version directory: {dir:?}"))?;
    Ok(dir)
}

pub fn sync_dir(path: &Path) -> Result<()> {
    let dir = OpenOptions::new()
        .read(true)
        .open(path)
        .with_context(|| format!("Failed to open directory for sync: {path:?}"))?;

    dir.sync_all()
        .with_context(|| format!("Failed to sync directory: {path:?}"))?;

    Ok(())
}

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

        assert_eq!(read_current(base_path).unwrap(), None);

        write_current_atomic(base_path, 42).unwrap();
        assert_eq!(read_current(base_path).unwrap(), Some((FORMAT_VERSION, 42)));

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

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), postings.len() * 8);

        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        assert_eq!(read_back, postings);
    }

    #[test]
    fn test_version_dir() {
        let base = Path::new("/base");
        assert_eq!(version_dir(base, 123), Path::new("/base/versions/123"));
    }
}
