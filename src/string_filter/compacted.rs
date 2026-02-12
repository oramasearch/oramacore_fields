use super::io::version_dir;
use super::merge::merge_sorted_u64_into;
use super::platform::advise_sequential;
use anyhow::{Context, Result};
use fst::Map;
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

/// An on-disk, read-only snapshot of the index at a specific version.
pub struct CompactedVersion {
    pub version_number: u64,
    /// Key-to-offset map for postings lookup.
    fst_map: Option<Map<Mmap>>,
    /// Postings data mapped from disk.
    postings_mmap: Option<Mmap>,
    /// Deleted doc_ids mapped from disk.
    deleted_mmap: Option<Mmap>,
}

impl CompactedVersion {
    pub fn empty() -> Self {
        Self {
            version_number: 0,
            fst_map: None,
            postings_mmap: None,
            deleted_mmap: None,
        }
    }

    pub fn load(base_path: &Path, version_number: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_number);

        let fst_path = dir.join("keys.fst");
        let postings_path = dir.join("postings.dat");
        let deleted_path = dir.join("deleted.bin");

        let fst_map = Self::load_fst(&fst_path)?;
        let postings_mmap = Self::load_mmap(&postings_path)?;
        let deleted_mmap = Self::load_mmap(&deleted_path)?;

        Ok(Self {
            version_number,
            fst_map,
            postings_mmap,
            deleted_mmap,
        })
    }

    fn load_fst(path: &Path) -> Result<Option<Map<Mmap>>> {
        if !path.exists() {
            return Ok(None);
        }

        let file =
            File::open(path).with_context(|| format!("Failed to open FST file: {path:?}"))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get FST metadata: {path:?}"))?;

        if metadata.len() == 0 {
            return Ok(None);
        }

        let mmap = unsafe {
            Mmap::map(&file).with_context(|| format!("Failed to mmap FST file: {path:?}"))?
        };

        let map = Map::new(mmap).map_err(|e| anyhow::anyhow!("Failed to parse FST file: {e}"))?;

        Ok(Some(map))
    }

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

    /// Return the sorted doc_ids for a key, or `None` if the key is not present.
    pub fn lookup(&self, key: &str) -> Option<&[u64]> {
        let fst_map = self.fst_map.as_ref()?;
        let postings_mmap = self.postings_mmap.as_ref()?;

        let offset = fst_map.get(key)? as usize;
        let data = postings_mmap.as_ref();

        if offset + 8 > data.len() {
            return None;
        }

        let count = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?) as usize;

        let start = offset + 8;
        let end = start + count * 8;

        if end > data.len() {
            return None;
        }

        let ptr = data[start..end].as_ptr() as *const u64;
        // SAFETY: Mmap returns page-aligned memory (>= 8 bytes aligned).
        // The postings file is written as sequences of LE u64. The offset+8 start
        // is also 8-byte aligned because offsets are always at 8-byte boundaries.
        // The returned slice borrows from &self and the mapping remains valid.
        Some(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Return the deleted doc_ids as a sorted slice.
    pub fn deletes_slice(&self) -> &[u64] {
        Self::mmap_as_u64_slice(self.deleted_mmap.as_ref())
    }

    fn mmap_as_u64_slice(mmap: Option<&Mmap>) -> &[u64] {
        match mmap {
            Some(m) => {
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            None => &[],
        }
    }

    /// Iterate all keys and their posting lists in lexicographic order.
    pub fn iter_all(&self) -> CompactedIterator<'_> {
        CompactedIterator {
            stream: self.fst_map.as_ref().map(|m| m.stream()),
            postings_mmap: self.postings_mmap.as_ref(),
        }
    }

    /// Return the number of unique keys.
    pub fn key_count(&self) -> usize {
        self.fst_map.as_ref().map_or(0, |m| m.len())
    }

    /// Get the total number of doc_id entries across all posting lists, O(1).
    ///
    /// Derived from the postings format: each key contributes 8 bytes (count) plus
    /// count×8 bytes (doc_ids), so `total_postings = postings_bytes / 8 - key_count`.
    pub fn total_postings(&self) -> usize {
        let postings_bytes = self.postings_mmap.as_ref().map_or(0, |m| m.len());
        let key_count = self.key_count();
        if postings_bytes == 0 {
            return 0;
        }
        postings_bytes / 8 - key_count
    }

    /// Build a new compacted version by streaming a sorted merge of compacted + live entries
    /// directly into FST builder + postings writer, avoiding O(K) heap allocations.
    ///
    /// - `compacted_iter`: iterator over the existing compacted version's entries (lexicographic order)
    /// - `live_iter`: iterator over the live snapshot's entries (lexicographic order)
    /// - `deleted_set`: if `Some`, filter out these doc_ids from every posting list (apply-deletes strategy)
    /// - `deletes_iter`: iterator of doc_ids to write to `deleted.bin` (carry-forward strategy passes merged deletes here)
    /// - `path`: the version directory to write into
    pub fn build_from_sorted_sources<'a, 'b, L, D>(
        compacted_iter: &mut CompactedIterator<'a>,
        live_iter: &mut std::iter::Peekable<L>,
        deleted_set: Option<&HashSet<u64>>,
        deletes_iter: D,
        path: &Path,
    ) -> Result<()>
    where
        L: Iterator<Item = (&'b str, &'b [u64])>,
        D: Iterator<Item = u64>,
    {
        let fst_path = path.join("keys.fst");
        let postings_path = path.join("postings.dat");
        let deleted_path = path.join("deleted.bin");

        let postings_file = File::create(&postings_path)
            .with_context(|| format!("Failed to create postings file: {postings_path:?}"))?;
        let mut postings_writer = BufWriter::new(postings_file);

        let fst_file = File::create(&fst_path)
            .with_context(|| format!("Failed to create FST file: {fst_path:?}"))?;
        let fst_writer = BufWriter::new(fst_file);

        let mut fst_builder = fst::MapBuilder::new(fst_writer)
            .map_err(|e| anyhow::anyhow!("Failed to create FST builder: {e}"))?;

        let mut current_offset: u64 = 0;

        // Reusable buffers
        let mut compacted_key_buf: Vec<u8> = Vec::new();
        let mut merged_ids_buf: Vec<u64> = Vec::new();
        let mut filtered_ids_buf: Vec<u64> = Vec::new();

        // Peek the first compacted entry
        let mut compacted_entry: Option<&'a [u64]> =
            compacted_iter.next_entry_into(&mut compacted_key_buf);

        // Action enum to split decide/execute phases (avoids borrow conflicts on compacted_key_buf)
        enum ActionToEmit {
            Compacted,
            Live,
            Merged,
        }

        loop {
            let live_peek = live_iter.peek();
            let action = match (&compacted_entry, live_peek) {
                (None, None) => break,
                (Some(_), None) => ActionToEmit::Compacted,
                (None, Some(_)) => ActionToEmit::Live,
                (Some(_), Some(&(live_key, _))) => {
                    match compacted_key_buf.as_slice().cmp(live_key.as_bytes()) {
                        std::cmp::Ordering::Less => ActionToEmit::Compacted,
                        std::cmp::Ordering::Greater => ActionToEmit::Live,
                        std::cmp::Ordering::Equal => ActionToEmit::Merged,
                    }
                }
            };

            match action {
                ActionToEmit::Compacted => {
                    let doc_ids = compacted_entry.unwrap();
                    write_entry(
                        &compacted_key_buf,
                        doc_ids,
                        deleted_set,
                        &mut filtered_ids_buf,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                    compacted_entry = compacted_iter.next_entry_into(&mut compacted_key_buf);
                }
                ActionToEmit::Live => {
                    let (live_key, live_ids) = live_iter.next().unwrap();
                    write_entry(
                        live_key.as_bytes(),
                        live_ids,
                        deleted_set,
                        &mut filtered_ids_buf,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                }
                ActionToEmit::Merged => {
                    let compacted_ids = compacted_entry.unwrap();
                    let (_, live_ids) = live_iter.next().unwrap();
                    merge_sorted_u64_into(compacted_ids, live_ids, &mut merged_ids_buf);
                    write_entry(
                        &compacted_key_buf,
                        &merged_ids_buf,
                        deleted_set,
                        &mut filtered_ids_buf,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                    compacted_entry = compacted_iter.next_entry_into(&mut compacted_key_buf);
                }
            }
        }

        fst_builder
            .finish()
            .map_err(|e| anyhow::anyhow!("Failed to finish FST: {e}"))?;

        let postings_inner = postings_writer
            .into_inner()
            .map_err(|e| e.into_error())
            .with_context(|| "Failed to flush postings buffer")?;
        postings_inner
            .sync_all()
            .with_context(|| "Failed to sync postings file")?;

        // Write deleted.bin from iterator
        super::io::write_postings_from_iter(&deleted_path, deletes_iter)?;

        Ok(())
    }
}

/// Write a single entry (key + doc_ids) to the FST builder and postings writer,
/// optionally filtering out deleted doc_ids.
fn write_entry(
    key: &[u8],
    doc_ids: &[u64],
    deleted_set: Option<&HashSet<u64>>,
    filtered_buf: &mut Vec<u64>,
    fst_builder: &mut fst::MapBuilder<BufWriter<File>>,
    postings_writer: &mut BufWriter<File>,
    current_offset: &mut u64,
) -> Result<()> {
    let ids_to_write: &[u64] = if let Some(set) = deleted_set {
        filtered_buf.clear();
        filtered_buf.extend(doc_ids.iter().filter(|id| !set.contains(id)));
        filtered_buf.as_slice()
    } else {
        doc_ids
    };

    if ids_to_write.is_empty() {
        return Ok(());
    }

    fst_builder
        .insert(key, *current_offset)
        .map_err(|e| anyhow::anyhow!("Failed to insert key into FST: {e}"))?;

    let count = ids_to_write.len() as u64;
    postings_writer
        .write_all(&count.to_le_bytes())
        .with_context(|| "Failed to write posting count")?;

    for &doc_id in ids_to_write {
        postings_writer
            .write_all(&doc_id.to_le_bytes())
            .with_context(|| "Failed to write doc_id")?;
    }

    *current_offset += 8 + (ids_to_write.len() as u64) * 8;

    Ok(())
}

/// Iterator over all keys and posting lists in a compacted version.
pub struct CompactedIterator<'a> {
    stream: Option<fst::map::Stream<'a>>,
    postings_mmap: Option<&'a Mmap>,
}

impl<'a> CompactedIterator<'a> {
    /// Get the next entry, writing the key bytes into the provided buffer to avoid allocation.
    /// Returns the doc_ids slice from the mmap.
    pub fn next_entry_into(&mut self, key_buf: &mut Vec<u8>) -> Option<&'a [u64]> {
        use fst::Streamer;
        let stream = self.stream.as_mut()?;
        let postings_mmap = self.postings_mmap?;

        let (key_bytes, offset) = stream.next()?;
        key_buf.clear();
        key_buf.extend_from_slice(key_bytes);
        let offset = offset as usize;
        let data = postings_mmap.as_ref();

        if offset + 8 > data.len() {
            return Some(&[]);
        }

        let count = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?) as usize;
        let start = offset + 8;
        let end = start + count * 8;

        if end > data.len() {
            return Some(&[]);
        }

        let ptr = data[start..end].as_ptr() as *const u64;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };

        Some(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::super::io::ensure_version_dir;
    use super::*;
    use tempfile::TempDir;

    fn build_from_entries(entries: &[(&str, &[u64])], deleted: &[u64], path: &Path) {
        let empty = CompactedVersion::empty();
        let mut compacted_iter = empty.iter_all();
        let mut live_iter = entries.iter().copied().peekable();
        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            None,
            deleted.iter().copied(),
            path,
        )
        .unwrap();
    }

    #[test]
    fn test_empty_version() {
        let version = CompactedVersion::empty();
        assert_eq!(version.version_number, 0);
        assert!(version.lookup("anything").is_none());
        assert!(version.deletes_slice().is_empty());
    }

    #[test]
    fn test_build_and_load() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(
            &[
                ("apple", &[1u64, 3, 5]),
                ("banana", &[2u64, 4]),
                ("cherry", &[6u64]),
            ],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert_eq!(version.lookup("apple"), Some([1u64, 3, 5].as_slice()));
        assert_eq!(version.lookup("banana"), Some([2u64, 4].as_slice()));
        assert_eq!(version.lookup("cherry"), Some([6u64].as_slice()));
        assert_eq!(version.lookup("missing"), None);
        assert!(version.deletes_slice().is_empty());
    }

    #[test]
    fn test_build_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(&[("hello", &[1u64, 2, 3])], &[10u64, 20, 30], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert_eq!(version.lookup("hello"), Some([1u64, 2, 3].as_slice()));
        assert_eq!(version.deletes_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_iter_all() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(
            &[
                ("alpha", &[1u64]),
                ("beta", &[2u64, 3]),
                ("gamma", &[4u64, 5, 6]),
            ],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        let mut iter = version.iter_all();
        let mut key_buf = Vec::new();
        let mut results = Vec::new();
        while let Some(doc_ids) = iter.next_entry_into(&mut key_buf) {
            let key = String::from_utf8(key_buf.clone()).unwrap();
            results.push((key, doc_ids.to_vec()));
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], ("alpha".to_string(), vec![1]));
        assert_eq!(results[1], ("beta".to_string(), vec![2, 3]));
        assert_eq!(results[2], ("gamma".to_string(), vec![4, 5, 6]));
    }

    #[test]
    fn test_empty_entries_skipped() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(
            &[
                ("apple", &[1u64]),
                ("banana", &[]), // empty, should be skipped
                ("cherry", &[2u64]),
            ],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert_eq!(version.lookup("apple"), Some([1u64].as_slice()));
        assert_eq!(version.lookup("banana"), None);
        assert_eq!(version.lookup("cherry"), Some([2u64].as_slice()));
    }

    #[test]
    fn test_load_missing_files() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let _version_path = ensure_version_dir(base_path, 1).unwrap();

        // Don't write any files - load should succeed with empty data
        let version = CompactedVersion::load(base_path, 1).unwrap();

        assert!(version.lookup("anything").is_none());
        assert!(version.deletes_slice().is_empty());
        assert_eq!(version.key_count(), 0);
    }

    #[test]
    fn test_total_postings() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(
            &[
                ("alpha", &[1u64]),
                ("beta", &[2u64, 3]),
                ("gamma", &[4u64, 5, 6]),
            ],
            &[],
            &version_path,
        );
        let version = CompactedVersion::load(base_path, 1).unwrap();

        // 1 + 2 + 3 = 6 total postings
        assert_eq!(version.total_postings(), 6);

        // Verify it matches the iteration-based count
        let mut iter_count = 0;
        let mut iter = version.iter_all();
        let mut key_buf = Vec::new();
        while let Some(doc_ids) = iter.next_entry_into(&mut key_buf) {
            iter_count += doc_ids.len();
        }
        assert_eq!(version.total_postings(), iter_count);
    }

    #[test]
    fn test_total_postings_empty() {
        let version = CompactedVersion::empty();
        assert_eq!(version.total_postings(), 0);
    }

    #[test]
    fn test_iter_all_matches_lookup() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(
            &[
                ("alpha", &[1u64]),
                ("beta", &[2u64, 3]),
                ("gamma", &[4u64, 5, 6]),
            ],
            &[],
            &version_path,
        );
        let version = CompactedVersion::load(base_path, 1).unwrap();

        let mut iter = version.iter_all();
        let mut key_buf = Vec::new();
        while let Some(doc_ids) = iter.next_entry_into(&mut key_buf) {
            let key = std::str::from_utf8(&key_buf).unwrap();
            assert_eq!(version.lookup(key), Some(doc_ids));
        }
    }

    // ──────────────────────────────────────────────────────────────
    //  build_from_sorted_sources with both compacted + live
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_merge_compacted_and_live() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Build initial compacted version
        let v1_path = ensure_version_dir(base_path, 1).unwrap();
        build_from_entries(&[("apple", &[1u64, 3]), ("cherry", &[5u64])], &[], &v1_path);
        let v1 = CompactedVersion::load(base_path, 1).unwrap();

        // Merge with live entries (banana is new, apple overlaps)
        let v2_path = ensure_version_dir(base_path, 2).unwrap();
        let live_entries: Vec<(&str, &[u64])> = vec![("apple", &[2u64, 4]), ("banana", &[6u64, 7])];
        let mut compacted_iter = v1.iter_all();
        let mut live_iter = live_entries.into_iter().peekable();
        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            None,
            std::iter::empty(),
            &v2_path,
        )
        .unwrap();

        let v2 = CompactedVersion::load(base_path, 2).unwrap();

        // apple: merged [1,2,3,4]
        assert_eq!(v2.lookup("apple"), Some([1u64, 2, 3, 4].as_slice()));
        // banana: only from live
        assert_eq!(v2.lookup("banana"), Some([6u64, 7].as_slice()));
        // cherry: only from compacted
        assert_eq!(v2.lookup("cherry"), Some([5u64].as_slice()));
        assert_eq!(v2.key_count(), 3);
        assert_eq!(v2.total_postings(), 7);
    }

    // ──────────────────────────────────────────────────────────────
    //  build_from_sorted_sources with apply-deletes
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_with_apply_deletes_filtering() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Build initial
        let v1_path = ensure_version_dir(base_path, 1).unwrap();
        build_from_entries(&[("x", &[1u64, 2, 3, 4, 5])], &[], &v1_path);
        let v1 = CompactedVersion::load(base_path, 1).unwrap();

        // Rebuild with deletes applied
        let v2_path = ensure_version_dir(base_path, 2).unwrap();
        let deleted: HashSet<u64> = [2, 4].into_iter().collect();
        let mut compacted_iter = v1.iter_all();
        let empty: Vec<(&str, &[u64])> = vec![];
        let mut live_iter = empty.into_iter().peekable();
        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            Some(&deleted),
            std::iter::empty(),
            &v2_path,
        )
        .unwrap();

        let v2 = CompactedVersion::load(base_path, 2).unwrap();
        assert_eq!(v2.lookup("x"), Some([1u64, 3, 5].as_slice()));
        assert!(v2.deletes_slice().is_empty());
    }

    // ──────────────────────────────────────────────────────────────
    //  build_from_sorted_sources: apply-deletes removes entire key
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_apply_deletes_removes_entire_key() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        let v1_path = ensure_version_dir(base_path, 1).unwrap();
        build_from_entries(&[("a", &[1u64, 2]), ("b", &[3u64])], &[], &v1_path);
        let v1 = CompactedVersion::load(base_path, 1).unwrap();

        // Delete all doc_ids for key "b"
        let v2_path = ensure_version_dir(base_path, 2).unwrap();
        let deleted: HashSet<u64> = [3].into_iter().collect();
        let mut compacted_iter = v1.iter_all();
        let empty: Vec<(&str, &[u64])> = vec![];
        let mut live_iter = empty.into_iter().peekable();
        CompactedVersion::build_from_sorted_sources(
            &mut compacted_iter,
            &mut live_iter,
            Some(&deleted),
            std::iter::empty(),
            &v2_path,
        )
        .unwrap();

        let v2 = CompactedVersion::load(base_path, 2).unwrap();
        assert_eq!(v2.lookup("a"), Some([1u64, 2].as_slice()));
        assert_eq!(v2.lookup("b"), None); // entire key removed
        assert_eq!(v2.key_count(), 1);
    }

    // ──────────────────────────────────────────────────────────────
    //  Single key single doc_id
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_single_entry() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(&[("only", &[42u64])], &[], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.lookup("only"), Some([42u64].as_slice()));
        assert_eq!(version.key_count(), 1);
        assert_eq!(version.total_postings(), 1);
    }

    // ──────────────────────────────────────────────────────────────
    //  Unicode keys in compacted version
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_unicode_keys() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        // Note: entries must be in lexicographic byte order
        let mut entries: Vec<(&str, &[u64])> =
            vec![("日本語", &[1u64]), ("中文", &[2u64]), ("emoji🎉", &[3u64])];
        entries.sort_by_key(|e| e.0);

        build_from_entries(&entries, &[], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.lookup("日本語"), Some([1u64].as_slice()));
        assert_eq!(version.lookup("中文"), Some([2u64].as_slice()));
        assert_eq!(version.lookup("emoji🎉"), Some([3u64].as_slice()));
    }

    // ──────────────────────────────────────────────────────────────
    //  Build with carry-forward deletes
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_with_carry_forward_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(&[("hello", &[1u64, 2, 3])], &[10u64, 20], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.lookup("hello"), Some([1u64, 2, 3].as_slice()));
        assert_eq!(version.deletes_slice(), &[10, 20]);
    }

    // ──────────────────────────────────────────────────────────────
    //  Empty build (no entries, no deletes)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_empty() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_from_entries(&[], &[], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.key_count(), 0);
        assert_eq!(version.total_postings(), 0);
        assert!(version.deletes_slice().is_empty());
        assert!(version.lookup("anything").is_none());
    }

    // ──────────────────────────────────────────────────────────────
    //  Many keys
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn test_build_many_keys() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        let owned: Vec<(String, Vec<u64>)> = (0..200u64)
            .map(|i| (format!("key_{i:04}"), vec![i, i + 1000]))
            .collect();
        let entries: Vec<(&str, &[u64])> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();

        build_from_entries(&entries, &[], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.key_count(), 200);
        assert_eq!(version.total_postings(), 400);

        assert_eq!(version.lookup("key_0000"), Some([0u64, 1000].as_slice()));
        assert_eq!(version.lookup("key_0199"), Some([199u64, 1199].as_slice()));
        assert_eq!(version.lookup("key_0200"), None);
    }
}
