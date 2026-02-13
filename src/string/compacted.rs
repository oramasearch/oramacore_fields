use super::io::{version_dir, write_deleted_from_iter, write_doc_lengths, write_global_info};
use super::live::PostingTuple;
use super::platform::advise_sequential;
use anyhow::{Context, Result};
use fst::automaton::{Levenshtein, Str};
use fst::{Automaton, IntoStreamer, Map, Streamer};
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// An on-disk, read-only snapshot of the string index at a specific version.
pub struct CompactedVersion {
    pub version_number: u64,
    /// Term-to-offset map for postings lookup.
    fst_map: Option<Map<Mmap>>,
    /// Postings data mapped from disk (variable-length per term).
    postings_mmap: Option<Mmap>,
    /// Doc lengths data mapped from disk: sorted (doc_id:u64, field_length:u32) entries.
    doc_lengths_mmap: Option<Mmap>,
    /// Deleted doc_ids mapped from disk.
    deleted_mmap: Option<Mmap>,
    /// Sum of all field lengths of documents in this compacted version.
    pub total_document_length: u64,
    /// Number of documents in this compacted version.
    pub total_documents: u64,
}

impl CompactedVersion {
    pub fn empty() -> Self {
        Self {
            version_number: 0,
            fst_map: None,
            postings_mmap: None,
            doc_lengths_mmap: None,
            deleted_mmap: None,
            total_document_length: 0,
            total_documents: 0,
        }
    }

    pub fn load(base_path: &Path, version_number: u64) -> Result<Self> {
        let dir = version_dir(base_path, version_number);

        let fst_path = dir.join("keys.fst");
        let postings_path = dir.join("postings.dat");
        let doc_lengths_path = dir.join("doc_lengths.dat");
        let deleted_path = dir.join("deleted.bin");
        let global_info_path = dir.join("global_info.bin");

        let fst_map = Self::load_fst(&fst_path)?;
        let postings_mmap = Self::load_mmap(&postings_path)?;
        let doc_lengths_mmap = Self::load_mmap(&doc_lengths_path)?;
        let deleted_mmap = Self::load_mmap(&deleted_path)?;

        let (total_document_length, total_documents) =
            super::io::read_global_info(&global_info_path)?;

        Ok(Self {
            version_number,
            fst_map,
            postings_mmap,
            doc_lengths_mmap,
            deleted_mmap,
            total_document_length,
            total_documents,
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

    /// Lookup postings for a term. Returns a PostingsReader that iterates over entries.
    pub fn lookup_postings(&self, term: &str) -> Option<PostingsReader<'_>> {
        let fst_map = self.fst_map.as_ref()?;
        let postings_mmap = self.postings_mmap.as_ref()?;

        let offset = fst_map.get(term)? as usize;
        let data = postings_mmap.as_ref();

        if offset + 8 > data.len() {
            return None;
        }

        let doc_count = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        // Skip 4-byte pad
        let entries_start = offset + 8;

        Some(PostingsReader {
            data,
            pos: entries_start,
            remaining: doc_count,
        })
    }

    /// Search for terms matching the given token and tolerance.
    ///
    /// Returns `(matched_term, is_exact, PostingsReader)` triples.
    /// - `Some(0)`: exact match only
    /// - `None`: prefix search via FST `Str::starts_with()` automaton
    /// - `Some(n)`: Levenshtein distance <= n via FST automaton
    pub fn search_terms(&self, token: &str, tolerance: Option<u8>) -> Vec<(String, bool, PostingsReader<'_>)> {
        let fst_map = match self.fst_map.as_ref() {
            Some(m) => m,
            None => return vec![],
        };
        let postings_mmap = match self.postings_mmap.as_ref() {
            Some(m) => m,
            None => return vec![],
        };

        match tolerance {
            Some(0) => {
                // Exact match
                if let Some(reader) = self.lookup_postings(token) {
                    vec![(token.to_string(), true, reader)]
                } else {
                    vec![]
                }
            }
            None => {
                // Prefix search
                let automaton = Str::new(token).starts_with();
                self.search_with_automaton(fst_map, postings_mmap, automaton, token)
            }
            Some(n) => {
                // Levenshtein
                match Levenshtein::new(token, n as u32) {
                    Ok(automaton) => {
                        self.search_with_automaton(fst_map, postings_mmap, automaton, token)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Levenshtein automaton construction failed for '{}' with distance {}: {}",
                            token, n, e
                        );
                        vec![]
                    }
                }
            }
        }
    }

    /// Generic helper: stream FST matches using the given automaton and build PostingsReaders.
    fn search_with_automaton<'a, A: fst::Automaton>(
        &'a self,
        fst_map: &'a Map<Mmap>,
        postings_mmap: &'a Mmap,
        automaton: A,
        token: &str,
    ) -> Vec<(String, bool, PostingsReader<'a>)> {
        let mut stream = fst_map.search(automaton).into_stream();
        let data = postings_mmap.as_ref();
        let mut results = Vec::new();

        while let Some((key_bytes, offset)) = stream.next() {
            let offset = offset as usize;
            if offset + 8 > data.len() {
                continue;
            }

            let doc_count = match data[offset..offset + 4].try_into() {
                Ok(bytes) => u32::from_le_bytes(bytes) as usize,
                Err(_) => continue,
            };
            let entries_start = offset + 8;

            let key = match std::str::from_utf8(key_bytes) {
                Ok(k) => k.to_string(),
                Err(_) => continue,
            };
            let is_exact = key == token;

            results.push((
                key,
                is_exact,
                PostingsReader {
                    data,
                    pos: entries_start,
                    remaining: doc_count,
                },
            ));
        }

        results
    }

    /// Look up the field length for a doc_id using binary search on doc_lengths_mmap.
    pub fn field_length(&self, doc_id: u64) -> Option<u32> {
        let mmap = self.doc_lengths_mmap.as_ref()?;
        let data = mmap.as_ref();
        let entry_size = 12; // u64 + u32
        let count = data.len() / entry_size;

        if count == 0 {
            return None;
        }

        // Binary search
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let offset = mid * entry_size;
            let mid_doc_id =
                u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            match mid_doc_id.cmp(&doc_id) {
                std::cmp::Ordering::Equal => {
                    let field_len =
                        u32::from_le_bytes(data[offset + 8..offset + 12].try_into().ok()?);
                    return Some(field_len);
                }
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }

        None
    }

    /// Return the deleted doc_ids as a sorted slice.
    pub fn deletes_slice(&self) -> &[u64] {
        match self.deleted_mmap.as_ref() {
            Some(m) => {
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            None => &[],
        }
    }

    /// Return the number of unique terms.
    pub fn term_count(&self) -> usize {
        self.fst_map.as_ref().map_or(0, |m| m.len())
    }

    /// Iterate all terms and their posting data in lexicographic order (for compaction).
    pub fn iter_terms(&self) -> CompactedTermIterator<'_> {
        CompactedTermIterator {
            stream: self.fst_map.as_ref().map(|m| m.stream()),
            postings_mmap: self.postings_mmap.as_ref(),
        }
    }

    /// Iterate all doc_lengths in doc_id order (for compaction).
    pub fn iter_doc_lengths(&self) -> DocLengthIterator<'_> {
        let data = self.doc_lengths_mmap.as_ref().map(|m| m.as_ref());
        DocLengthIterator {
            data: data.unwrap_or(&[]),
            pos: 0,
        }
    }

    /// Count total postings across all terms by scanning the FST offsets.
    pub fn total_postings(&self) -> usize {
        let mut count = 0;
        let mut iter = self.iter_terms();
        let mut key_buf = Vec::new();
        while let Some(reader) = iter.next_term_into(&mut key_buf) {
            count += reader.remaining;
        }
        count
    }

    /// Build a new compacted version by merging compacted + live data.
    ///
    /// - `compacted_terms`: term iterator from previous compacted version
    /// - `live_terms`: sorted (term, postings) from live snapshot
    /// - `compacted_doc_lengths`: doc_length iterator from previous compacted version
    /// - `live_doc_lengths`: sorted (doc_id, field_length) from live snapshot
    /// - `deleted_set`: if `Some`, doc_ids to exclude from output postings and doc_lengths
    /// - `deletes_to_write`: doc_ids to write to deleted.bin
    /// - `path`: version directory to write into
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_sorted_sources<'a>(
        compacted_terms: &mut CompactedTermIterator<'a>,
        live_terms: &mut [(&str, &[PostingTuple])],
        compacted_doc_lengths: &mut DocLengthIterator<'_>,
        live_doc_lengths: &[(u64, u16)],
        deleted_set: Option<&HashSet<u64>>,
        deletes_to_write: &[u64],
        path: &Path,
    ) -> Result<()> {
        let fst_path = path.join("keys.fst");
        let postings_path = path.join("postings.dat");
        let doc_lengths_path = path.join("doc_lengths.dat");
        let deleted_path = path.join("deleted.bin");
        let global_info_path = path.join("global_info.bin");

        // ── Build postings + FST ──
        let postings_file = File::create(&postings_path)
            .with_context(|| format!("Failed to create postings file: {postings_path:?}"))?;
        let mut postings_writer = BufWriter::new(postings_file);

        let fst_file = File::create(&fst_path)
            .with_context(|| format!("Failed to create FST file: {fst_path:?}"))?;
        let fst_writer = BufWriter::new(fst_file);

        let mut fst_builder = fst::MapBuilder::new(fst_writer)
            .map_err(|e| anyhow::anyhow!("Failed to create FST builder: {e}"))?;

        let mut current_offset: u64 = 0;
        let mut compacted_key_buf: Vec<u8> = Vec::new();
        let mut live_idx = 0;

        let mut compacted_entry: Option<PostingsReader<'a>> =
            compacted_terms.next_term_into(&mut compacted_key_buf);

        enum Action {
            Compacted,
            Live,
            Merged,
        }

        loop {
            let live_peek = if live_idx < live_terms.len() {
                Some(live_terms[live_idx])
            } else {
                None
            };

            let action = match (&compacted_entry, live_peek) {
                (None, None) => break,
                (Some(_), None) => Action::Compacted,
                (None, Some(_)) => Action::Live,
                (Some(_), Some((live_key, _))) => {
                    match compacted_key_buf.as_slice().cmp(live_key.as_bytes()) {
                        std::cmp::Ordering::Less => Action::Compacted,
                        std::cmp::Ordering::Greater => Action::Live,
                        std::cmp::Ordering::Equal => Action::Merged,
                    }
                }
            };

            match action {
                Action::Compacted => {
                    let reader = compacted_entry.take().unwrap();
                    let entries: Vec<PostingEntry> = reader.collect();
                    write_postings_entry(
                        &compacted_key_buf,
                        &entries,
                        deleted_set,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                    compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
                }
                Action::Live => {
                    let (live_key, live_postings) = live_terms[live_idx];
                    live_idx += 1;
                    let entries: Vec<PostingEntry> = live_postings
                        .iter()
                        .map(|(doc_id, exact, stemmed)| PostingEntry {
                            doc_id: *doc_id,
                            exact_positions: exact.clone(),
                            stemmed_positions: stemmed.clone(),
                        })
                        .collect();
                    write_postings_entry(
                        live_key.as_bytes(),
                        &entries,
                        deleted_set,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                }
                Action::Merged => {
                    let reader = compacted_entry.take().unwrap();
                    let (_, live_postings) = live_terms[live_idx];
                    live_idx += 1;

                    let merged = merge_posting_entries(reader, live_postings);
                    write_postings_entry(
                        &compacted_key_buf,
                        &merged,
                        deleted_set,
                        &mut fst_builder,
                        &mut postings_writer,
                        &mut current_offset,
                    )?;
                    compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
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

        // ── Build doc_lengths ──
        let merged_doc_lengths =
            merge_doc_lengths(compacted_doc_lengths, live_doc_lengths, deleted_set);
        write_doc_lengths(
            &doc_lengths_path,
            &merged_doc_lengths,
        )?;

        // ── Write deleted.bin ──
        write_deleted_from_iter(&deleted_path, deletes_to_write.iter().copied())?;

        // ── Write global_info.bin ──
        // Compute new totals from merged doc_lengths
        let mut new_total_doc_length: u64 = 0;
        let mut new_total_documents: u64 = 0;
        for &(_, field_len) in &merged_doc_lengths {
            new_total_doc_length += field_len as u64;
            new_total_documents += 1;
        }

        // If we're carrying forward (no apply-deletes), we need to account for
        // documents that are in deleted.bin but not yet physically removed from
        // doc_lengths. But since we rebuild doc_lengths fresh with filtering,
        // the merged_doc_lengths already represents the true set.
        // However, if we didn't apply deletes, deleted docs may still be in the
        // merged doc_lengths (since deleted_set is None). In that case, the global
        // info should reflect ALL docs in doc_lengths (including pending deletes).
        write_global_info(&global_info_path, new_total_doc_length, new_total_documents)?;

        Ok(())
    }
}

/// A single posting entry read from the compacted postings file.
#[derive(Debug, Clone)]
pub struct PostingEntry {
    pub doc_id: u64,
    pub exact_positions: Vec<u32>,
    pub stemmed_positions: Vec<u32>,
}

/// Reader that iterates over posting entries for a single term from the mmap.
pub struct PostingsReader<'a> {
    data: &'a [u8],
    pos: usize,
    pub remaining: usize,
}

impl<'a> Iterator for PostingsReader<'a> {
    type Item = PostingEntry;

    fn next(&mut self) -> Option<PostingEntry> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        let data = self.data;
        let pos = self.pos;

        // doc_id: u64
        let doc_id = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        // exact_count: u32
        let exact_count =
            u32::from_le_bytes(data[pos + 8..pos + 12].try_into().ok()?) as usize;
        // stemmed_count: u32
        let stemmed_count =
            u32::from_le_bytes(data[pos + 12..pos + 16].try_into().ok()?) as usize;

        let mut cursor = pos + 16;

        let mut exact_positions = Vec::with_capacity(exact_count);
        for _ in 0..exact_count {
            let v = u32::from_le_bytes(data[cursor..cursor + 4].try_into().ok()?);
            exact_positions.push(v);
            cursor += 4;
        }

        let mut stemmed_positions = Vec::with_capacity(stemmed_count);
        for _ in 0..stemmed_count {
            let v = u32::from_le_bytes(data[cursor..cursor + 4].try_into().ok()?);
            stemmed_positions.push(v);
            cursor += 4;
        }

        self.pos = cursor;

        Some(PostingEntry {
            doc_id,
            exact_positions,
            stemmed_positions,
        })
    }
}

/// Iterator over terms in the compacted FST.
pub struct CompactedTermIterator<'a> {
    stream: Option<fst::map::Stream<'a>>,
    postings_mmap: Option<&'a Mmap>,
}

impl<'a> CompactedTermIterator<'a> {
    /// Advance to the next term, writing the key into `key_buf` and returning a PostingsReader.
    pub fn next_term_into(&mut self, key_buf: &mut Vec<u8>) -> Option<PostingsReader<'a>> {
        use fst::Streamer;
        let stream = self.stream.as_mut()?;
        let postings_mmap = self.postings_mmap?;

        let (key_bytes, offset) = stream.next()?;
        key_buf.clear();
        key_buf.extend_from_slice(key_bytes);
        let offset = offset as usize;
        let data = postings_mmap.as_ref();

        if offset + 8 > data.len() {
            return Some(PostingsReader {
                data,
                pos: offset,
                remaining: 0,
            });
        }

        let doc_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        let entries_start = offset + 8;

        Some(PostingsReader {
            data,
            pos: entries_start,
            remaining: doc_count,
        })
    }
}

/// Iterator over doc_lengths from the compacted mmap.
pub struct DocLengthIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for DocLengthIterator<'a> {
    type Item = (u64, u32);

    fn next(&mut self) -> Option<(u64, u32)> {
        if self.pos + 12 > self.data.len() {
            return None;
        }

        let doc_id = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().ok()?);
        let field_len =
            u32::from_le_bytes(self.data[self.pos + 8..self.pos + 12].try_into().ok()?);
        self.pos += 12;

        Some((doc_id, field_len))
    }
}

/// Write a single term's postings entry to disk, optionally filtering out deleted doc_ids.
fn write_postings_entry(
    key: &[u8],
    entries: &[PostingEntry],
    deleted_set: Option<&HashSet<u64>>,
    fst_builder: &mut fst::MapBuilder<BufWriter<File>>,
    postings_writer: &mut BufWriter<File>,
    current_offset: &mut u64,
) -> Result<()> {
    // Filter entries if needed
    let filtered: Vec<&PostingEntry>;
    let entries_to_write: &[&PostingEntry] = if let Some(set) = deleted_set {
        filtered = entries.iter().filter(|e| !set.contains(&e.doc_id)).collect();
        &filtered
    } else {
        // Can't avoid allocation here without unsafe, but it's only during compaction
        filtered = entries.iter().collect();
        &filtered
    };

    if entries_to_write.is_empty() {
        return Ok(());
    }

    fst_builder
        .insert(key, *current_offset)
        .map_err(|e| anyhow::anyhow!("Failed to insert key into FST: {e}"))?;

    let doc_count = entries_to_write.len() as u32;
    postings_writer.write_all(&doc_count.to_le_bytes())?;
    postings_writer.write_all(&0u32.to_le_bytes())?; // pad
    *current_offset += 8;

    for entry in entries_to_write {
        postings_writer.write_all(&entry.doc_id.to_le_bytes())?;
        let exact_count = entry.exact_positions.len() as u32;
        let stemmed_count = entry.stemmed_positions.len() as u32;
        postings_writer.write_all(&exact_count.to_le_bytes())?;
        postings_writer.write_all(&stemmed_count.to_le_bytes())?;
        for &pos in &entry.exact_positions {
            postings_writer.write_all(&pos.to_le_bytes())?;
        }
        for &pos in &entry.stemmed_positions {
            postings_writer.write_all(&pos.to_le_bytes())?;
        }
        // 8 (doc_id) + 4 (exact_count) + 4 (stemmed_count) + 4*exact + 4*stemmed
        *current_offset +=
            16 + (exact_count as u64) * 4 + (stemmed_count as u64) * 4;
    }

    Ok(())
}

/// Merge compacted posting entries with live posting entries. Both must be sorted by doc_id.
/// Live entries win on conflict (same doc_id).
fn merge_posting_entries(
    compacted: PostingsReader<'_>,
    live: &[PostingTuple],
) -> Vec<PostingEntry> {
    let compacted_entries: Vec<PostingEntry> = compacted.collect();
    let mut result = Vec::with_capacity(compacted_entries.len() + live.len());

    let mut ci = 0;
    let mut li = 0;

    while ci < compacted_entries.len() && li < live.len() {
        let c_id = compacted_entries[ci].doc_id;
        let l_id = live[li].0;

        match c_id.cmp(&l_id) {
            std::cmp::Ordering::Less => {
                result.push(compacted_entries[ci].clone());
                ci += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(PostingEntry {
                    doc_id: live[li].0,
                    exact_positions: live[li].1.clone(),
                    stemmed_positions: live[li].2.clone(),
                });
                li += 1;
            }
            std::cmp::Ordering::Equal => {
                // Live wins
                result.push(PostingEntry {
                    doc_id: live[li].0,
                    exact_positions: live[li].1.clone(),
                    stemmed_positions: live[li].2.clone(),
                });
                ci += 1;
                li += 1;
            }
        }
    }

    while ci < compacted_entries.len() {
        result.push(compacted_entries[ci].clone());
        ci += 1;
    }
    while li < live.len() {
        result.push(PostingEntry {
            doc_id: live[li].0,
            exact_positions: live[li].1.clone(),
            stemmed_positions: live[li].2.clone(),
        });
        li += 1;
    }

    result
}

/// Merge compacted doc_lengths with live doc_lengths, optionally filtering deletes.
/// Live entries win on conflict (same doc_id).
fn merge_doc_lengths(
    compacted: &mut DocLengthIterator<'_>,
    live: &[(u64, u16)],
    deleted_set: Option<&HashSet<u64>>,
) -> Vec<(u64, u32)> {
    let mut result = Vec::new();
    let mut li = 0;
    let mut compacted_next = compacted.next();

    loop {
        let live_peek = if li < live.len() {
            Some((live[li].0, live[li].1 as u32))
        } else {
            None
        };

        match (compacted_next, live_peek) {
            (None, None) => break,
            (Some((c_id, c_len)), None) => {
                if deleted_set.is_none_or(|s| !s.contains(&c_id)) {
                    result.push((c_id, c_len));
                }
                compacted_next = compacted.next();
            }
            (None, Some((l_id, l_len))) => {
                if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                    result.push((l_id, l_len));
                }
                li += 1;
            }
            (Some((c_id, c_len)), Some((l_id, l_len))) => {
                match c_id.cmp(&l_id) {
                    std::cmp::Ordering::Less => {
                        if deleted_set.is_none_or(|s| !s.contains(&c_id)) {
                            result.push((c_id, c_len));
                        }
                        compacted_next = compacted.next();
                    }
                    std::cmp::Ordering::Greater => {
                        if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                            result.push((l_id, l_len));
                        }
                        li += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        // Live wins
                        if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                            result.push((l_id, l_len));
                        }
                        compacted_next = compacted.next();
                        li += 1;
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::super::io::ensure_version_dir;
    use super::*;
    use tempfile::TempDir;

    fn build_simple(
        entries: &[(&str, Vec<(u64, Vec<u32>, Vec<u32>)>)],
        doc_lengths: &[(u64, u16)],
        deleted: &[u64],
        path: &Path,
    ) {
        let empty = CompactedVersion::empty();
        let mut compacted_terms = empty.iter_terms();
        let mut live: Vec<_> = entries
            .iter()
            .map(|(k, v)| (*k, v.as_slice()))
            .collect();
        let mut compacted_dl = empty.iter_doc_lengths();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &mut live,
            &mut compacted_dl,
            doc_lengths,
            None,
            deleted,
            path,
        )
        .unwrap();
    }

    #[test]
    fn test_empty_version() {
        let version = CompactedVersion::empty();
        assert_eq!(version.version_number, 0);
        assert!(version.lookup_postings("anything").is_none());
        assert!(version.deletes_slice().is_empty());
        assert_eq!(version.total_documents, 0);
    }

    #[test]
    fn test_build_and_load() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("hello", vec![(1, vec![0], vec![]), (2, vec![0, 3], vec![1])]),
                ("world", vec![(3, vec![1], vec![])]),
            ],
            &[(1, 5), (2, 10), (3, 3)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        // Check postings
        let mut reader = version.lookup_postings("hello").unwrap();
        let e1 = reader.next().unwrap();
        assert_eq!(e1.doc_id, 1);
        assert_eq!(e1.exact_positions, vec![0]);
        assert!(e1.stemmed_positions.is_empty());

        let e2 = reader.next().unwrap();
        assert_eq!(e2.doc_id, 2);
        assert_eq!(e2.exact_positions, vec![0, 3]);
        assert_eq!(e2.stemmed_positions, vec![1]);

        assert!(reader.next().is_none());

        // Check doc_lengths
        assert_eq!(version.field_length(1), Some(5));
        assert_eq!(version.field_length(2), Some(10));
        assert_eq!(version.field_length(3), Some(3));
        assert_eq!(version.field_length(999), None);

        // Check global info
        assert_eq!(version.total_documents, 3);
        assert_eq!(version.total_document_length, 18); // 5 + 10 + 3

        assert!(version.deletes_slice().is_empty());
    }

    #[test]
    fn test_build_with_deletes() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[("hello", vec![(1, vec![0], vec![])])],
            &[(1, 5)],
            &[10, 20],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.deletes_slice(), &[10, 20]);
    }

    #[test]
    fn test_field_length_binary_search() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[("a", vec![
                (1, vec![0], vec![]),
                (5, vec![0], vec![]),
                (10, vec![0], vec![]),
                (100, vec![0], vec![]),
            ])],
            &[(1, 3), (5, 7), (10, 15), (100, 20)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.field_length(1), Some(3));
        assert_eq!(version.field_length(5), Some(7));
        assert_eq!(version.field_length(10), Some(15));
        assert_eq!(version.field_length(100), Some(20));
        assert_eq!(version.field_length(2), None);
        assert_eq!(version.field_length(50), None);
    }

    #[test]
    fn test_iter_terms() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("alpha", vec![(1, vec![0], vec![])]),
                ("beta", vec![(2, vec![0], vec![])]),
                ("gamma", vec![(3, vec![0, 1], vec![2])]),
            ],
            &[(1, 1), (2, 1), (3, 3)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();
        let mut iter = version.iter_terms();
        let mut key_buf = Vec::new();
        let mut results = Vec::new();

        while let Some(reader) = iter.next_term_into(&mut key_buf) {
            let key = String::from_utf8(key_buf.clone()).unwrap();
            let entries: Vec<_> = reader.collect();
            results.push((key, entries.len()));
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], ("alpha".to_string(), 1));
        assert_eq!(results[1], ("beta".to_string(), 1));
        assert_eq!(results[2], ("gamma".to_string(), 1));
    }

    #[test]
    fn test_iter_doc_lengths() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[("a", vec![(1, vec![0], vec![]), (5, vec![0], vec![])])],
            &[(1, 3), (5, 7)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();
        let lengths: Vec<_> = version.iter_doc_lengths().collect();
        assert_eq!(lengths, vec![(1, 3), (5, 7)]);
    }

    #[test]
    fn test_total_postings() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("a", vec![(1, vec![0], vec![])]),
                ("b", vec![(2, vec![0], vec![]), (3, vec![0], vec![])]),
            ],
            &[(1, 1), (2, 1), (3, 1)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.total_postings(), 3);
    }

    #[test]
    fn test_empty_build() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(&[], &[], &[], &version_path);

        let version = CompactedVersion::load(base_path, 1).unwrap();
        assert_eq!(version.term_count(), 0);
        assert_eq!(version.total_postings(), 0);
        assert_eq!(version.total_documents, 0);
    }

    #[test]
    fn test_merge_compacted_and_live() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();

        // Build v1
        let v1_path = ensure_version_dir(base_path, 1).unwrap();
        build_simple(
            &[("hello", vec![(1, vec![0], vec![]), (3, vec![1], vec![])])],
            &[(1, 5), (3, 7)],
            &[],
            &v1_path,
        );
        let v1 = CompactedVersion::load(base_path, 1).unwrap();

        // Build v2 by merging v1 + live
        let v2_path = ensure_version_dir(base_path, 2).unwrap();
        let mut compacted_terms = v1.iter_terms();
        let live_postings = vec![(2u64, vec![0u32], vec![1u32])];
        let mut live_terms: Vec<_> =
            vec![("hello", live_postings.as_slice())];
        let mut compacted_dl = v1.iter_doc_lengths();
        let live_doc_lengths = &[(2u64, 4u16)];

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &mut live_terms,
            &mut compacted_dl,
            live_doc_lengths,
            None,
            &[],
            &v2_path,
        )
        .unwrap();

        let v2 = CompactedVersion::load(base_path, 2).unwrap();

        // hello should now have docs 1, 2, 3
        let entries: Vec<PostingEntry> = v2.lookup_postings("hello").unwrap().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].doc_id, 1);
        assert_eq!(entries[1].doc_id, 2);
        assert_eq!(entries[2].doc_id, 3);

        assert_eq!(v2.total_documents, 3);
        assert_eq!(v2.field_length(2), Some(4));
    }

    #[test]
    fn test_search_terms_exact() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("apple", vec![(1, vec![0], vec![])]),
                ("application", vec![(2, vec![0], vec![])]),
                ("banana", vec![(3, vec![0], vec![])]),
            ],
            &[(1, 3), (2, 3), (3, 3)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        let results = version.search_terms("apple", Some(0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
        assert!(results[0].1); // is_exact

        let results = version.search_terms("missing", Some(0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_terms_prefix() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("apple", vec![(1, vec![0], vec![])]),
                ("application", vec![(2, vec![0], vec![])]),
                ("banana", vec![(3, vec![0], vec![])]),
            ],
            &[(1, 3), (2, 3), (3, 3)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        // Prefix "app" should match "apple" and "application"
        let results = version.search_terms("app", None);
        assert_eq!(results.len(), 2);
        let terms: Vec<&str> = results.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(terms.contains(&"apple"));
        assert!(terms.contains(&"application"));
        // Neither is an exact match for "app"
        for (_, is_exact, _) in &results {
            assert!(!is_exact);
        }

        // Prefix "apple" should match exactly "apple"
        let results = version.search_terms("apple", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
        assert!(results[0].1); // is_exact

        // Prefix "xyz" should match nothing
        let results = version.search_terms("xyz", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_terms_levenshtein() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let version_path = ensure_version_dir(base_path, 1).unwrap();

        build_simple(
            &[
                ("apple", vec![(1, vec![0], vec![])]),
                ("apply", vec![(2, vec![0], vec![])]),
                ("banana", vec![(3, vec![0], vec![])]),
            ],
            &[(1, 3), (2, 3), (3, 3)],
            &[],
            &version_path,
        );

        let version = CompactedVersion::load(base_path, 1).unwrap();

        // Distance 1 from "apple" should match "apple" (exact) and "apply" (1 edit)
        let results = version.search_terms("apple", Some(1));
        assert_eq!(results.len(), 2);
        let terms: Vec<&str> = results.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(terms.contains(&"apple"));
        assert!(terms.contains(&"apply"));
        // "apple" is exact, "apply" is not
        for (term, is_exact, _) in &results {
            if term == "apple" {
                assert!(is_exact);
            } else {
                assert!(!is_exact);
            }
        }

        // Distance 0 via Levenshtein is exact match
        let results = version.search_terms("apple", Some(0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
        assert!(results[0].1);
    }
}
