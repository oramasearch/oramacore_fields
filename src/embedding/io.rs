use super::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 2;

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

/// Segment data directory: `base_path/segments/seg_{id}/`.
pub fn segment_data_dir(base_path: &Path, segment_id: u64) -> PathBuf {
    base_path.join("segments").join(format!("seg_{segment_id}"))
}

/// Create the segment data directory if it doesn't exist.
pub fn ensure_segment_dir(base_path: &Path, segment_id: u64) -> Result<PathBuf, Error> {
    let dir = segment_data_dir(base_path, segment_id);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Write manifest.json with segment metadata entries.
pub fn write_manifest(version_dir: &Path, entries: &[ManifestEntry]) -> Result<(), Error> {
    let segments: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            serde_json::json!({
                "segment_id": e.segment_id,
                "num_nodes": e.num_nodes,
                "num_deletes": e.num_deletes,
                "min_doc_id": e.min_doc_id,
                "max_doc_id": e.max_doc_id,
                "nodes_at_last_rebuild": e.nodes_at_last_rebuild,
                "insertions_since_rebuild": e.insertions_since_rebuild,
            })
        })
        .collect();
    let manifest = serde_json::json!({ "segments": segments });
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| Error::CorruptedFile(format!("failed to serialize manifest: {e}")))?;
    let path = version_dir.join("manifest.json");
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

/// Read manifest.json from a version directory.
pub fn read_manifest(version_dir: &Path) -> Result<Vec<ManifestEntry>, Error> {
    let path = version_dir.join("manifest.json");
    let contents = std::fs::read_to_string(&path)
        .map_err(|e| Error::CorruptedFile(format!("failed to read manifest.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| Error::CorruptedFile(format!("invalid manifest.json: {e}")))?;
    let segments = v["segments"]
        .as_array()
        .ok_or_else(|| Error::CorruptedFile("manifest.json missing segments array".into()))?;
    let mut entries = Vec::with_capacity(segments.len());
    for seg in segments {
        entries.push(ManifestEntry {
            segment_id: seg["segment_id"]
                .as_u64()
                .ok_or_else(|| Error::CorruptedFile("missing segment_id in manifest".into()))?,
            num_nodes: seg["num_nodes"]
                .as_u64()
                .ok_or_else(|| Error::CorruptedFile("missing num_nodes in manifest".into()))?
                as usize,
            num_deletes: seg["num_deletes"]
                .as_u64()
                .ok_or_else(|| Error::CorruptedFile("missing num_deletes in manifest".into()))?
                as usize,
            min_doc_id: seg["min_doc_id"]
                .as_u64()
                .ok_or_else(|| Error::CorruptedFile("missing min_doc_id in manifest".into()))?,
            max_doc_id: seg["max_doc_id"]
                .as_u64()
                .ok_or_else(|| Error::CorruptedFile("missing max_doc_id in manifest".into()))?,
            nodes_at_last_rebuild: seg["nodes_at_last_rebuild"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing nodes_at_last_rebuild in manifest".into())
            })? as usize,
            insertions_since_rebuild: seg["insertions_since_rebuild"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing insertions_since_rebuild in manifest".into())
            })? as usize,
        });
    }
    Ok(entries)
}

/// Write a per-segment delete file: `version_dir/seg_{id}.del`.
pub fn write_delete_file(
    version_dir: &Path,
    segment_id: u64,
    deletes: &[u64],
) -> Result<(), Error> {
    let path = version_dir.join(format!("seg_{segment_id}.del"));
    let mut file = File::create(&path)?;
    for &id in deletes {
        file.write_all(&id.to_ne_bytes())?;
    }
    file.sync_all()?;
    Ok(())
}

/// Load a per-segment delete file as mmap. Returns None if file is missing or empty.
pub fn load_delete_file(
    version_dir: &Path,
    segment_id: u64,
) -> Result<Option<memmap2::Mmap>, Error> {
    let path = version_dir.join(format!("seg_{segment_id}.del"));
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(&path)?;
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        return Ok(None);
    }
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(Some(mmap))
}

/// List segment ids from the `segments/` directory.
pub fn list_segment_dirs(base_path: &Path) -> Result<Vec<u64>, Error> {
    let segments_dir = base_path.join("segments");
    if !segments_dir.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(&segments_dir)?;
    let mut ids: Vec<u64> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            let id_str = name.strip_prefix("seg_")?;
            id_str.parse::<u64>().ok()
        })
        .collect();
    ids.sort_unstable();
    Ok(ids)
}

/// Per-segment metadata stored in manifest.json.
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    pub segment_id: u64,
    pub num_nodes: usize,
    pub num_deletes: usize,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub nodes_at_last_rebuild: usize,
    pub insertions_since_rebuild: usize,
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
