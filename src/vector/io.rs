use super::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 1;

pub fn read_current(base_path: &Path) -> Result<Option<(u32, u64)>, Error> {
    let current_path = base_path.join("CURRENT");
    if !current_path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&current_path)
        .map_err(|e| Error::CorruptedFile(format!("failed to read CURRENT: {e}")))?;
    let mut lines = contents.lines();
    let format_version: u32 = lines
        .next()
        .ok_or_else(|| Error::CorruptedFile("CURRENT file is empty".into()))?
        .trim()
        .parse()
        .map_err(|e| Error::CorruptedFile(format!("failed to parse format version: {e}")))?;
    let version_number: u64 = lines
        .next()
        .ok_or_else(|| Error::CorruptedFile("CURRENT file missing version number".into()))?
        .trim()
        .parse()
        .map_err(|e| Error::CorruptedFile(format!("failed to parse version number: {e}")))?;
    Ok(Some((format_version, version_number)))
}

pub fn write_current_atomic(base_path: &Path, version_number: u64) -> Result<(), Error> {
    let current_path = base_path.join("CURRENT");
    let tmp_path = base_path.join("CURRENT.tmp");

    let mut file = File::create(&tmp_path)?;
    writeln!(file, "{FORMAT_VERSION}")?;
    write!(file, "{version_number}")?;
    file.sync_all()?;

    fs::rename(&tmp_path, &current_path)?;

    if let Ok(canonical) = base_path.canonicalize() {
        if let Ok(dir) = OpenOptions::new().read(true).open(&canonical) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

pub fn version_dir(base_path: &Path, version_number: u64) -> PathBuf {
    base_path.join("versions").join(version_number.to_string())
}

pub fn ensure_version_dir(base_path: &Path, version_number: u64) -> Result<PathBuf, Error> {
    let dir = version_dir(base_path, version_number);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn sync_dir(path: &Path) -> Result<(), Error> {
    let dir = OpenOptions::new().read(true).open(path)?;
    dir.sync_all()?;
    Ok(())
}

pub fn list_version_dirs(base_path: &Path) -> Result<Vec<u64>, Error> {
    let versions_dir = base_path.join("versions");
    if !versions_dir.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(&versions_dir)?;
    let mut versions: Vec<u64> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let file_type = entry.file_type().ok()?;
            if !file_type.is_dir() {
                return None;
            }
            entry.file_name().to_str()?.parse::<u64>().ok()
        })
        .collect();
    versions.sort_unstable();
    Ok(versions)
}

pub fn remove_version_dir(base_path: &Path, version_number: u64) -> Result<(), Error> {
    let dir = version_dir(base_path, version_number);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Write sorted doc_ids to a binary file (native-endian u64).
pub fn write_postings(path: &Path, doc_ids: &[u64]) -> Result<(), Error> {
    let mut file = File::create(path)?;
    for &id in doc_ids {
        file.write_all(&id.to_ne_bytes())?;
    }
    file.sync_all()?;
    Ok(())
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
    }

    #[test]
    fn test_version_dir() {
        let base = Path::new("/base");
        assert_eq!(version_dir(base, 123), Path::new("/base/versions/123"));
    }

    #[test]
    fn test_ensure_version_dir() {
        let tmp = TempDir::new().unwrap();
        let dir = ensure_version_dir(tmp.path(), 1).unwrap();
        assert!(dir.exists());
        assert!(dir.is_dir());
    }

    #[test]
    fn test_list_version_dirs() {
        let tmp = TempDir::new().unwrap();
        ensure_version_dir(tmp.path(), 3).unwrap();
        ensure_version_dir(tmp.path(), 1).unwrap();
        ensure_version_dir(tmp.path(), 2).unwrap();
        let versions = list_version_dirs(tmp.path()).unwrap();
        assert_eq!(versions, vec![1, 2, 3]);
    }
}
