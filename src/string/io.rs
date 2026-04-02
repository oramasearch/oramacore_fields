use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 2;
/// Previous format version for v1→v2 migration detection.
pub const FORMAT_VERSION_V1: u32 = 1;

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
    // On Unix, we can sync the directory
    #[cfg(unix)]
    {
        let dir = OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("Failed to open directory for sync: {path:?}"))?;

        dir.sync_all()
            .with_context(|| format!("Failed to sync directory: {path:?}"))?;
    }

    // On non-Unix, this is a no-op
    #[cfg(not(unix))]
    {
        let _ = path;
    }

    Ok(())
}

pub fn list_version_dirs(base_path: &Path) -> Result<Box<dyn Iterator<Item = u64>>> {
    let versions_dir = base_path.join("versions");

    // Return empty iterator if versions directory doesn't exist yet
    // (no compaction has been performed)
    if !versions_dir.exists() {
        return Ok(Box::new(std::iter::empty()));
    }

    let entries = fs::read_dir(&versions_dir)
        .with_context(|| format!("Failed to read versions directory: {versions_dir:?}"))?;

    let versions = entries.filter_map(|entry| entry.ok()).filter_map(|entry| {
        let file_type = entry.file_type().ok()?;
        if !file_type.is_dir() {
            return None;
        }
        entry.file_name().to_str()?.parse::<u64>().ok()
    });

    Ok(Box::new(versions))
}

pub fn remove_version_dir(base_path: &Path, version_number: u64) -> Result<()> {
    let dir = version_dir(base_path, version_number);
    fs::remove_dir_all(&dir).with_context(|| format!("Failed to remove version directory: {dir:?}"))
}

/// Return the path for a segment data directory: `base_path/segments/seg_{id}`.
pub fn segment_data_dir(base_path: &Path, segment_id: u64) -> PathBuf {
    base_path.join("segments").join(format!("seg_{segment_id}"))
}

/// Create and return the segment data directory.
pub fn ensure_segment_dir(base_path: &Path, segment_id: u64) -> Result<PathBuf> {
    let dir = segment_data_dir(base_path, segment_id);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create segment directory: {dir:?}"))?;
    Ok(dir)
}

/// List all segment IDs that exist under `base_path/segments/`.
pub fn list_segment_dirs(base_path: &Path) -> Result<Vec<u64>> {
    let segments_dir = base_path.join("segments");
    if !segments_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = fs::read_dir(&segments_dir)
        .with_context(|| format!("Failed to read segments directory: {segments_dir:?}"))?;

    let mut ids: Vec<u64> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            name.strip_prefix("seg_")?.parse::<u64>().ok()
        })
        .collect();
    ids.sort();
    Ok(ids)
}

/// Remove a segment directory.
pub fn remove_segment_dir(base_path: &Path, segment_id: u64) -> Result<()> {
    let dir = segment_data_dir(base_path, segment_id);
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .with_context(|| format!("Failed to remove segment directory: {dir:?}"))?;
    }
    Ok(())
}

/// A single entry in the manifest describing one segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub segment_id: u64,
    pub num_postings: usize,
    pub num_deletes: usize,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub total_doc_length: u64,
    pub total_documents: u64,
}

/// Write manifest.json to the version directory.
/// Asserts that entries are ordered by `min_doc_id` ascending.
pub fn write_manifest(version_dir: &Path, entries: &[ManifestEntry]) -> Result<()> {
    // Validate ordering invariant
    for i in 1..entries.len() {
        assert!(
            entries[i].min_doc_id >= entries[i - 1].min_doc_id,
            "Manifest entries must be ordered by min_doc_id: entry {} has min_doc_id {} but entry {} has min_doc_id {}",
            i - 1, entries[i - 1].min_doc_id, i, entries[i].min_doc_id
        );
    }

    let path = version_dir.join("manifest.json");
    let json =
        serde_json::to_string_pretty(entries).with_context(|| "Failed to serialize manifest")?;

    let file =
        File::create(&path).with_context(|| format!("Failed to create manifest file: {path:?}"))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(json.as_bytes())?;
    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush manifest buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync manifest file: {path:?}"))?;

    Ok(())
}

/// Read manifest.json from the version directory. Returns error if ordering is violated.
pub fn read_manifest(version_dir: &Path) -> Result<Vec<ManifestEntry>> {
    let path = version_dir.join("manifest.json");
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read manifest file: {path:?}"))?;

    let entries: Vec<ManifestEntry> = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse manifest file: {path:?}"))?;

    // Validate ordering invariant
    for i in 1..entries.len() {
        if entries[i].min_doc_id < entries[i - 1].min_doc_id {
            return Err(anyhow!(
                "Manifest entries not ordered by min_doc_id: entry {} has {} but entry {} has {}",
                i - 1,
                entries[i - 1].min_doc_id,
                i,
                entries[i].min_doc_id
            ));
        }
    }

    Ok(entries)
}

/// Write the CURRENT file atomically with a specific format version and version number.
pub fn write_current_atomic_versioned(
    base_path: &Path,
    format_version: u32,
    version_number: u64,
) -> Result<()> {
    let current_path = base_path.join("CURRENT");
    let tmp_path = base_path.join("CURRENT.tmp");

    let mut file = File::create(&tmp_path)
        .with_context(|| format!("Failed to create CURRENT.tmp: {tmp_path:?}"))?;

    writeln!(file, "{format_version}")
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

/// Write sorted u64 doc_ids to a file.
pub fn write_deleted(path: &Path, doc_ids: &[u64]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create deleted file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for &v in doc_ids {
        writer
            .write_all(&v.to_ne_bytes())
            .with_context(|| format!("Failed to write to: {path:?}"))?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync file: {path:?}"))?;

    Ok(())
}

/// Write u64 values from an iterator to a file.
pub fn write_deleted_from_iter<I>(path: &Path, iter: I) -> Result<()>
where
    I: Iterator<Item = u64>,
{
    let file =
        File::create(path).with_context(|| format!("Failed to create deleted file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for v in iter {
        writer
            .write_all(&v.to_ne_bytes())
            .with_context(|| format!("Failed to write to: {path:?}"))?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync file: {path:?}"))?;

    Ok(())
}

/// Write doc_lengths as sorted (doc_id: u64, field_length: u32) entries (12 bytes each).
pub fn write_doc_lengths(path: &Path, entries: &[(u64, u32)]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create doc_lengths file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for &(doc_id, field_length) in entries {
        writer.write_all(&doc_id.to_ne_bytes())?;
        writer.write_all(&field_length.to_ne_bytes())?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush doc_lengths buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync doc_lengths file: {path:?}"))?;

    Ok(())
}

/// Write global info: (total_document_length: u64, total_documents: u64).
pub fn write_global_info(
    path: &Path,
    total_document_length: u64,
    total_documents: u64,
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create global_info file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    writer.write_all(&total_document_length.to_ne_bytes())?;
    writer.write_all(&total_documents.to_ne_bytes())?;

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush global_info buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync global_info file: {path:?}"))?;

    Ok(())
}

/// Read global info: (total_document_length, total_documents).
pub fn read_global_info(path: &Path) -> Result<(u64, u64)> {
    if !path.exists() {
        return Ok((0, 0));
    }

    let bytes = fs::read(path).with_context(|| format!("Failed to read global_info: {path:?}"))?;

    if bytes.len() != 16 {
        return Err(anyhow!(
            "global_info.bin has invalid size: {} bytes (expected 16)",
            bytes.len()
        ));
    }

    let total_document_length = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
    let total_documents = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());

    Ok((total_document_length, total_documents))
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
    fn test_version_dir() {
        let base = Path::new("/base");
        assert_eq!(version_dir(base, 123), Path::new("/base/versions/123"));
    }

    #[test]
    fn test_write_and_read_global_info() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("global_info.bin");

        write_global_info(&path, 12345, 100).unwrap();
        let (total_len, total_docs) = read_global_info(&path).unwrap();
        assert_eq!(total_len, 12345);
        assert_eq!(total_docs, 100);
    }

    #[test]
    fn test_read_global_info_missing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nonexistent.bin");
        let (total_len, total_docs) = read_global_info(&path).unwrap();
        assert_eq!(total_len, 0);
        assert_eq!(total_docs, 0);
    }

    #[test]
    fn test_write_doc_lengths() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("doc_lengths.dat");

        let entries = vec![(1u64, 10u32), (5, 20), (10, 30)];
        write_doc_lengths(&path, &entries).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * 12); // 3 entries * 12 bytes each
    }

    #[test]
    fn test_write_deleted() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("deleted.bin");

        let doc_ids = vec![1u64, 5, 10];
        write_deleted(&path, &doc_ids).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * 8);

        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(read_back, doc_ids);
    }
}
