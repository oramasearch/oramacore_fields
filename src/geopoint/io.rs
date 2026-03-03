use anyhow::{anyhow, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 2;

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

    let version_id: u64 = lines
        .next()
        .ok_or_else(|| anyhow!("CURRENT file missing version_id line"))?
        .trim()
        .parse()
        .with_context(|| format!("Failed to parse version_id from CURRENT file: {contents:?}"))?;

    Ok(Some((format_version, version_id)))
}

pub fn write_current_atomic(base_path: &Path, version_id: u64) -> Result<()> {
    let current_path = base_path.join("CURRENT");
    let tmp_path = base_path.join("CURRENT.tmp");

    let mut file = File::create(&tmp_path)
        .with_context(|| format!("Failed to create CURRENT.tmp: {tmp_path:?}"))?;

    writeln!(file, "{FORMAT_VERSION}")
        .with_context(|| "Failed to write format version to CURRENT.tmp")?;
    write!(file, "{version_id}").with_context(|| "Failed to write version_id to CURRENT.tmp")?;

    file.sync_all()
        .with_context(|| "Failed to sync CURRENT.tmp")?;

    fs::rename(&tmp_path, &current_path)
        .with_context(|| "Failed to atomically rename CURRENT.tmp to CURRENT")?;

    if let Ok(dir) = OpenOptions::new()
        .read(true)
        .open(base_path.canonicalize()?)
    {
        let _ = dir.sync_all();
    }

    Ok(())
}

pub fn write_u64_slice(path: &Path, values: &[u64]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("Failed to create file: {path:?}"))?;

    let mut writer = BufWriter::new(file);
    for &v in values {
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

pub fn version_dir(base_path: &Path, version_id: u64) -> PathBuf {
    base_path.join("versions").join(version_id.to_string())
}

pub fn ensure_version_dir(base_path: &Path, version_id: u64) -> Result<PathBuf> {
    let dir = version_dir(base_path, version_id);
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

pub fn remove_version_dir(base_path: &Path, version_id: u64) -> Result<()> {
    let dir = version_dir(base_path, version_id);
    fs::remove_dir_all(&dir).with_context(|| format!("Failed to remove version directory: {dir:?}"))
}

// --- Segment directory helpers ---

pub fn segment_subdir(version_dir: &Path, index: usize) -> PathBuf {
    version_dir.join(format!("segment_{index}"))
}

pub fn ensure_segment_subdir(version_dir: &Path, index: usize) -> Result<PathBuf> {
    let dir = segment_subdir(version_dir, index);
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create segment directory: {dir:?}"))?;
    Ok(dir)
}

/// Copies segment data files from one directory to another.
pub fn copy_dir_contents(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    for filename in &["inner.idx", "leaves.dat"] {
        let src = src_dir.join(filename);
        let dst = dst_dir.join(filename);
        if src.exists() {
            fs::copy(&src, &dst).with_context(|| {
                format!("Failed to copy {filename} from {src_dir:?} to {dst_dir:?}")
            })?;
        }
    }
    Ok(())
}

// --- Manifest read/write ---

/// Writes the segment manifest file with per-segment point counts.
pub fn write_manifest(version_dir: &Path, point_counts: &[u64]) -> Result<()> {
    let path = version_dir.join("manifest.bin");
    let file =
        File::create(&path).with_context(|| format!("Failed to create manifest.bin: {path:?}"))?;
    let mut writer = BufWriter::new(file);

    let num_segments = point_counts.len() as u64;
    writer
        .write_all(&num_segments.to_ne_bytes())
        .with_context(|| "Failed to write num_segments to manifest.bin")?;

    for &count in point_counts {
        writer
            .write_all(&count.to_ne_bytes())
            .with_context(|| "Failed to write point_count to manifest.bin")?;
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush manifest.bin: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync manifest.bin: {path:?}"))?;

    Ok(())
}

/// Reads the segment manifest file and returns per-segment point counts.
pub fn read_manifest(version_dir: &Path) -> Result<Vec<u64>> {
    let path = version_dir.join("manifest.bin");
    let data = fs::read(&path).with_context(|| format!("Failed to read manifest.bin: {path:?}"))?;

    if data.len() < 8 {
        anyhow::bail!("manifest.bin too small: {} bytes", data.len());
    }

    let num_segments = u64::from_ne_bytes(data[..8].try_into().unwrap()) as usize;
    let expected_size = 8 + num_segments * 8;

    if data.len() != expected_size {
        anyhow::bail!(
            "manifest.bin size mismatch: expected {expected_size} bytes for {num_segments} segments, got {}",
            data.len()
        );
    }

    let mut point_counts = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let offset = 8 + i * 8;
        let count = u64::from_ne_bytes(data[offset..offset + 8].try_into().unwrap());
        point_counts.push(count);
    }

    Ok(point_counts)
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
    fn test_write_u64_slice() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.bin");

        let values = vec![1u64, 10, 100, 1000];
        write_u64_slice(&path, &values).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), values.len() * 8);

        let read_back: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        assert_eq!(read_back, values);
    }

    #[test]
    fn test_version_dir() {
        let base = Path::new("/base");
        assert_eq!(version_dir(base, 123), Path::new("/base/versions/123"));
    }

    #[test]
    fn test_segment_subdir() {
        let ver = Path::new("/base/versions/1");
        assert_eq!(
            segment_subdir(ver, 0),
            Path::new("/base/versions/1/segment_0")
        );
        assert_eq!(
            segment_subdir(ver, 3),
            Path::new("/base/versions/1/segment_3")
        );
    }

    #[test]
    fn test_manifest_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let counts = vec![100u64, 200, 50];
        write_manifest(dir, &counts).unwrap();

        let read_back = read_manifest(dir).unwrap();
        assert_eq!(read_back, counts);
    }

    #[test]
    fn test_manifest_empty() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let counts: Vec<u64> = vec![];
        write_manifest(dir, &counts).unwrap();

        let read_back = read_manifest(dir).unwrap();
        assert!(read_back.is_empty());
    }

    #[test]
    fn test_copy_dir_contents() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&dst).unwrap();

        fs::write(src.join("inner.idx"), b"inner data").unwrap();
        fs::write(src.join("leaves.dat"), b"leaf data").unwrap();

        copy_dir_contents(&src, &dst).unwrap();

        assert_eq!(fs::read(dst.join("inner.idx")).unwrap(), b"inner data");
        assert_eq!(fs::read(dst.join("leaves.dat")).unwrap(), b"leaf data");
    }
}
