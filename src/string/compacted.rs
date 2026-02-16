use super::io::{version_dir, write_deleted_from_iter, write_global_info};
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

        let doc_count = u32::from_ne_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        // Skip 4-byte pad
        let entries_start = offset + 8;

        Some(PostingsReader {
            data,
            pos: entries_start,
            remaining: doc_count,
        })
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
            let mid_doc_id = u64::from_ne_bytes(data[offset..offset + 8].try_into().ok()?);
            match mid_doc_id.cmp(&doc_id) {
                std::cmp::Ordering::Equal => {
                    let field_len =
                        u32::from_ne_bytes(data[offset + 8..offset + 12].try_into().ok()?);
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

    /// Callback-based term search: invokes `f(is_exact, reader)` for each matching term.
    ///
    /// - `Some(0)`: exact match only
    /// - `None`: prefix search via FST `Str::starts_with()` automaton
    /// - `Some(n)`: Levenshtein distance <= n via FST automaton
    pub fn for_each_term_match<F>(&self, token: &str, tolerance: Option<u8>, mut f: F) -> Result<()>
    where
        F: FnMut(bool, PostingsReader<'_>),
    {
        let fst_map = match self.fst_map.as_ref() {
            Some(m) => m,
            None => return Ok(()),
        };
        let postings_mmap = match self.postings_mmap.as_ref() {
            Some(m) => m,
            None => return Ok(()),
        };

        match tolerance {
            Some(0) => {
                if let Some(reader) = self.lookup_postings(token) {
                    f(true, reader);
                }
            }
            None => {
                let automaton = Str::new(token).starts_with();
                self.for_each_automaton_match(fst_map, postings_mmap, automaton, token, &mut f);
            }
            Some(n) => {
                let automaton = Levenshtein::new(token, n as u32).map_err(|e| {
                    anyhow::anyhow!(
                        "Levenshtein automaton construction failed for '{token}' with distance {n}: {e}"
                    )
                })?;
                self.for_each_automaton_match(fst_map, postings_mmap, automaton, token, &mut f);
            }
        }

        Ok(())
    }

    /// Generic helper: stream FST matches using the given automaton and invoke callback per match.
    fn for_each_automaton_match<'a, A: fst::Automaton, F>(
        &'a self,
        fst_map: &'a Map<Mmap>,
        postings_mmap: &'a Mmap,
        automaton: A,
        token: &str,
        f: &mut F,
    ) where
        F: FnMut(bool, PostingsReader<'a>),
    {
        let mut stream = fst_map.search(automaton).into_stream();
        let data = postings_mmap.as_ref();

        while let Some((key_bytes, offset)) = stream.next() {
            let offset = offset as usize;
            if offset + 8 > data.len() {
                continue;
            }

            let doc_count = match data[offset..offset + 4].try_into() {
                Ok(bytes) => u32::from_ne_bytes(bytes) as usize,
                Err(_) => continue,
            };
            let entries_start = offset + 8;

            let is_exact = key_bytes == token.as_bytes();

            f(
                is_exact,
                PostingsReader {
                    data,
                    pos: entries_start,
                    remaining: doc_count,
                },
            );
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
        live_terms: &[(&str, &[PostingTuple])],
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

        // Reusable byte buffer for serializing entries before writing (allocated once)
        let mut entries_buf: Vec<u8> = Vec::new();

        let mut compacted_entry: Option<PostingsReader<'a>> =
            compacted_terms.next_term_into(&mut compacted_key_buf);

        loop {
            let live_peek = if live_idx < live_terms.len() {
                Some(live_terms[live_idx])
            } else {
                None
            };

            match (compacted_entry, live_peek) {
                (None, None) => break,
                (Some(mut reader), None) => {
                    entries_buf.clear();
                    let mut count: u32 = 0;
                    while let Some(entry) = reader.next_ref() {
                        if deleted_set.is_none_or(|s| !s.contains(&entry.doc_id)) {
                            write_entry_to_buf(
                                &mut entries_buf,
                                entry.doc_id,
                                entry.exact_positions,
                                entry.stemmed_positions,
                            );
                            count += 1;
                        }
                    }
                    if count > 0 {
                        flush_term_buf(
                            &compacted_key_buf,
                            &entries_buf,
                            count,
                            &mut fst_builder,
                            &mut postings_writer,
                            &mut current_offset,
                        )?;
                    }
                    compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
                }
                (None, Some(_)) => {
                    let (live_key, live_postings) = live_terms[live_idx];
                    live_idx += 1;
                    entries_buf.clear();
                    let mut count: u32 = 0;
                    for &(doc_id, ref exact, ref stemmed) in live_postings {
                        if deleted_set.is_none_or(|s| !s.contains(&doc_id)) {
                            write_entry_to_buf(&mut entries_buf, doc_id, exact, stemmed);
                            count += 1;
                        }
                    }
                    if count > 0 {
                        flush_term_buf(
                            live_key.as_bytes(),
                            &entries_buf,
                            count,
                            &mut fst_builder,
                            &mut postings_writer,
                            &mut current_offset,
                        )?;
                    }
                    compacted_entry = None;
                }
                (Some(reader), Some((live_key, _))) => {
                    match compacted_key_buf.as_slice().cmp(live_key.as_bytes()) {
                        std::cmp::Ordering::Less => {
                            let mut reader = reader;
                            entries_buf.clear();
                            let mut count: u32 = 0;
                            while let Some(entry) = reader.next_ref() {
                                if deleted_set.is_none_or(|s| !s.contains(&entry.doc_id)) {
                                    write_entry_to_buf(
                                        &mut entries_buf,
                                        entry.doc_id,
                                        entry.exact_positions,
                                        entry.stemmed_positions,
                                    );
                                    count += 1;
                                }
                            }
                            if count > 0 {
                                flush_term_buf(
                                    &compacted_key_buf,
                                    &entries_buf,
                                    count,
                                    &mut fst_builder,
                                    &mut postings_writer,
                                    &mut current_offset,
                                )?;
                            }
                            compacted_entry =
                                compacted_terms.next_term_into(&mut compacted_key_buf);
                        }
                        std::cmp::Ordering::Greater => {
                            // Live comes first; put compacted reader back
                            compacted_entry = Some(reader);
                            let (live_key, live_postings) = live_terms[live_idx];
                            live_idx += 1;
                            entries_buf.clear();
                            let mut count: u32 = 0;
                            for &(doc_id, ref exact, ref stemmed) in live_postings {
                                if deleted_set.is_none_or(|s| !s.contains(&doc_id)) {
                                    write_entry_to_buf(&mut entries_buf, doc_id, exact, stemmed);
                                    count += 1;
                                }
                            }
                            if count > 0 {
                                flush_term_buf(
                                    live_key.as_bytes(),
                                    &entries_buf,
                                    count,
                                    &mut fst_builder,
                                    &mut postings_writer,
                                    &mut current_offset,
                                )?;
                            }
                        }
                        std::cmp::Ordering::Equal => {
                            let mut reader = reader;
                            let (_, live_postings) = live_terms[live_idx];
                            live_idx += 1;

                            // Streaming sorted merge: both sources sorted by doc_id
                            entries_buf.clear();
                            let mut count: u32 = 0;
                            let mut li = 0;
                            let mut compacted_next = reader.next_ref();

                            loop {
                                let live_peek_entry = if li < live_postings.len() {
                                    Some(&live_postings[li])
                                } else {
                                    None
                                };

                                match (&compacted_next, live_peek_entry) {
                                    (None, None) => break,
                                    (Some(c), None) => {
                                        if deleted_set.is_none_or(|s| !s.contains(&c.doc_id)) {
                                            write_entry_to_buf(
                                                &mut entries_buf,
                                                c.doc_id,
                                                c.exact_positions,
                                                c.stemmed_positions,
                                            );
                                            count += 1;
                                        }
                                        compacted_next = reader.next_ref();
                                    }
                                    (None, Some(l)) => {
                                        if deleted_set.is_none_or(|s| !s.contains(&l.0)) {
                                            write_entry_to_buf(&mut entries_buf, l.0, &l.1, &l.2);
                                            count += 1;
                                        }
                                        li += 1;
                                    }
                                    (Some(c), Some(l)) => match c.doc_id.cmp(&l.0) {
                                        std::cmp::Ordering::Less => {
                                            if deleted_set.is_none_or(|s| !s.contains(&c.doc_id)) {
                                                write_entry_to_buf(
                                                    &mut entries_buf,
                                                    c.doc_id,
                                                    c.exact_positions,
                                                    c.stemmed_positions,
                                                );
                                                count += 1;
                                            }
                                            compacted_next = reader.next_ref();
                                        }
                                        std::cmp::Ordering::Greater => {
                                            if deleted_set.is_none_or(|s| !s.contains(&l.0)) {
                                                write_entry_to_buf(
                                                    &mut entries_buf,
                                                    l.0,
                                                    &l.1,
                                                    &l.2,
                                                );
                                                count += 1;
                                            }
                                            li += 1;
                                        }
                                        std::cmp::Ordering::Equal => {
                                            // Live wins
                                            if deleted_set.is_none_or(|s| !s.contains(&l.0)) {
                                                write_entry_to_buf(
                                                    &mut entries_buf,
                                                    l.0,
                                                    &l.1,
                                                    &l.2,
                                                );
                                                count += 1;
                                            }
                                            compacted_next = reader.next_ref();
                                            li += 1;
                                        }
                                    },
                                }
                            }

                            if count > 0 {
                                flush_term_buf(
                                    &compacted_key_buf,
                                    &entries_buf,
                                    count,
                                    &mut fst_builder,
                                    &mut postings_writer,
                                    &mut current_offset,
                                )?;
                            }
                            compacted_entry =
                                compacted_terms.next_term_into(&mut compacted_key_buf);
                        }
                    }
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

        // ── Build doc_lengths + global_info (streaming) ──
        let (new_total_doc_length, new_total_documents) = merge_and_write_doc_lengths(
            &doc_lengths_path,
            compacted_doc_lengths,
            live_doc_lengths,
            deleted_set,
        )?;

        // ── Write deleted.bin ──
        write_deleted_from_iter(&deleted_path, deletes_to_write.iter().copied())?;

        // ── Write global_info.bin ──
        write_global_info(&global_info_path, new_total_doc_length, new_total_documents)?;

        Ok(())
    }
}

/// Zero-copy borrowing variant — slices point into mmap'd data.
pub struct PostingEntryRef<'a> {
    pub doc_id: u64,
    pub exact_positions: &'a [u32],
    pub stemmed_positions: &'a [u32],
}

/// Reader that iterates over posting entries for a single term from the mmap.
pub struct PostingsReader<'a> {
    data: &'a [u8],
    pos: usize,
    pub remaining: usize,
}

impl<'a> PostingsReader<'a> {
    /// Zero-copy next: returns borrowed slices into the mmap'd data.
    pub fn next_ref(&mut self) -> Option<PostingEntryRef<'a>> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        let data = self.data;
        let pos = self.pos;

        // doc_id: u64 (read byte-by-byte, no alignment needed)
        let doc_id = u64::from_ne_bytes(data[pos..pos + 8].try_into().ok()?);
        // exact_count: u32
        let exact_count = u32::from_ne_bytes(data[pos + 8..pos + 12].try_into().ok()?) as usize;
        // stemmed_count: u32
        let stemmed_count = u32::from_ne_bytes(data[pos + 12..pos + 16].try_into().ok()?) as usize;

        let cursor = pos + 16;

        // Cast position byte ranges to &[u32] via from_raw_parts.
        // Alignment is guaranteed: mmap base is page-aligned, offsets are multiples of 4
        // (header is 8 bytes, entry headers are 16 bytes, positions are 4 bytes each).
        let exact_ptr = data[cursor..].as_ptr() as *const u32;
        debug_assert!(
            exact_ptr as usize % std::mem::align_of::<u32>() == 0,
            "exact_positions pointer is not aligned"
        );
        let exact_positions = unsafe { std::slice::from_raw_parts(exact_ptr, exact_count) };

        let stemmed_offset = cursor + exact_count * 4;
        let stemmed_ptr = data[stemmed_offset..].as_ptr() as *const u32;
        debug_assert!(
            stemmed_ptr as usize % std::mem::align_of::<u32>() == 0,
            "stemmed_positions pointer is not aligned"
        );
        let stemmed_positions = unsafe { std::slice::from_raw_parts(stemmed_ptr, stemmed_count) };

        self.pos = stemmed_offset + stemmed_count * 4;

        Some(PostingEntryRef {
            doc_id,
            exact_positions,
            stemmed_positions,
        })
    }
}

impl<'a> Iterator for PostingsReader<'a> {
    type Item = PostingEntryRef<'a>;

    fn next(&mut self) -> Option<PostingEntryRef<'a>> {
        self.next_ref()
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

        let doc_count = u32::from_ne_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
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

        let doc_id = u64::from_ne_bytes(self.data[self.pos..self.pos + 8].try_into().ok()?);
        let field_len = u32::from_ne_bytes(self.data[self.pos + 8..self.pos + 12].try_into().ok()?);
        self.pos += 12;

        Some((doc_id, field_len))
    }
}

/// Serialize a single posting entry (doc_id + positions) into a byte buffer.
#[inline]
fn write_entry_to_buf(buf: &mut Vec<u8>, doc_id: u64, exact: &[u32], stemmed: &[u32]) {
    buf.extend_from_slice(&doc_id.to_ne_bytes());
    buf.extend_from_slice(&(exact.len() as u32).to_ne_bytes());
    buf.extend_from_slice(&(stemmed.len() as u32).to_ne_bytes());
    for &pos in exact {
        buf.extend_from_slice(&pos.to_ne_bytes());
    }
    for &pos in stemmed {
        buf.extend_from_slice(&pos.to_ne_bytes());
    }
}

/// Flush a completed term's entries buffer to disk and register it in the FST.
fn flush_term_buf(
    key: &[u8],
    entries_buf: &[u8],
    count: u32,
    fst_builder: &mut fst::MapBuilder<BufWriter<File>>,
    postings_writer: &mut BufWriter<File>,
    current_offset: &mut u64,
) -> Result<()> {
    fst_builder
        .insert(key, *current_offset)
        .map_err(|e| anyhow::anyhow!("Failed to insert key into FST: {e}"))?;

    postings_writer.write_all(&count.to_ne_bytes())?;
    postings_writer.write_all(&0u32.to_ne_bytes())?; // pad
    postings_writer.write_all(entries_buf)?;

    *current_offset += 8 + entries_buf.len() as u64;

    Ok(())
}

/// Merge compacted doc_lengths with live doc_lengths, write directly to disk,
/// and return (total_doc_length, total_documents). No intermediate Vec.
fn merge_and_write_doc_lengths(
    path: &Path,
    compacted: &mut DocLengthIterator<'_>,
    live: &[(u64, u16)],
    deleted_set: Option<&HashSet<u64>>,
) -> Result<(u64, u64)> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create doc_lengths file: {path:?}"))?;
    let mut writer = BufWriter::new(file);

    let mut total_doc_length: u64 = 0;
    let mut total_documents: u64 = 0;
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
                    writer.write_all(&c_id.to_ne_bytes())?;
                    writer.write_all(&c_len.to_ne_bytes())?;
                    total_doc_length += c_len as u64;
                    total_documents += 1;
                }
                compacted_next = compacted.next();
            }
            (None, Some((l_id, l_len))) => {
                if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                    writer.write_all(&l_id.to_ne_bytes())?;
                    writer.write_all(&l_len.to_ne_bytes())?;
                    total_doc_length += l_len as u64;
                    total_documents += 1;
                }
                li += 1;
            }
            (Some((c_id, c_len)), Some((l_id, l_len))) => match c_id.cmp(&l_id) {
                std::cmp::Ordering::Less => {
                    if deleted_set.is_none_or(|s| !s.contains(&c_id)) {
                        writer.write_all(&c_id.to_ne_bytes())?;
                        writer.write_all(&c_len.to_ne_bytes())?;
                        total_doc_length += c_len as u64;
                        total_documents += 1;
                    }
                    compacted_next = compacted.next();
                }
                std::cmp::Ordering::Greater => {
                    if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                        writer.write_all(&l_id.to_ne_bytes())?;
                        writer.write_all(&l_len.to_ne_bytes())?;
                        total_doc_length += l_len as u64;
                        total_documents += 1;
                    }
                    li += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Live wins
                    if deleted_set.is_none_or(|s| !s.contains(&l_id)) {
                        writer.write_all(&l_id.to_ne_bytes())?;
                        writer.write_all(&l_len.to_ne_bytes())?;
                        total_doc_length += l_len as u64;
                        total_documents += 1;
                    }
                    compacted_next = compacted.next();
                    li += 1;
                }
            },
        }
    }

    writer
        .into_inner()
        .map_err(|e| e.into_error())
        .with_context(|| format!("Failed to flush doc_lengths buffer for: {path:?}"))?
        .sync_all()
        .with_context(|| format!("Failed to sync doc_lengths file: {path:?}"))?;

    Ok((total_doc_length, total_documents))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::type_complexity)]

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
        let live: Vec<_> = entries.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let mut compacted_dl = empty.iter_doc_lengths();

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &live,
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
                (
                    "hello",
                    vec![(1, vec![0], vec![]), (2, vec![0, 3], vec![1])],
                ),
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
            &[(
                "a",
                vec![
                    (1, vec![0], vec![]),
                    (5, vec![0], vec![]),
                    (10, vec![0], vec![]),
                    (100, vec![0], vec![]),
                ],
            )],
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
            results.push((key, reader.count()));
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
        let live_terms: Vec<_> = vec![("hello", live_postings.as_slice())];
        let mut compacted_dl = v1.iter_doc_lengths();
        let live_doc_lengths = &[(2u64, 4u16)];

        CompactedVersion::build_from_sorted_sources(
            &mut compacted_terms,
            &live_terms,
            &mut compacted_dl,
            live_doc_lengths,
            None,
            &[],
            &v2_path,
        )
        .unwrap();

        let v2 = CompactedVersion::load(base_path, 2).unwrap();

        // hello should now have docs 1, 2, 3
        let entries: Vec<PostingEntryRef> = v2.lookup_postings("hello").unwrap().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].doc_id, 1);
        assert_eq!(entries[1].doc_id, 2);
        assert_eq!(entries[2].doc_id, 3);

        assert_eq!(v2.total_documents, 3);
        assert_eq!(v2.field_length(2), Some(4));
    }

    #[test]
    fn test_for_each_term_match_exact() {
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

        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("apple", Some(0), |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].0); // is_exact

        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("missing", Some(0), |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_for_each_term_match_prefix() {
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
        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("app", None, |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert_eq!(results.len(), 2);
        // Neither is an exact match for "app"
        for (is_exact, _) in &results {
            assert!(!is_exact);
        }

        // Prefix "apple" should match exactly "apple"
        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("apple", None, |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].0); // is_exact

        // Prefix "xyz" should match nothing
        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("xyz", None, |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_for_each_term_match_levenshtein() {
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
        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("apple", Some(1), |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert_eq!(results.len(), 2);
        let exact_count = results.iter().filter(|(is_exact, _)| *is_exact).count();
        assert_eq!(exact_count, 1);

        // Distance 0 via Levenshtein is exact match
        let mut results: Vec<(bool, usize)> = Vec::new();
        version
            .for_each_term_match("apple", Some(0), |is_exact, reader| {
                results.push((is_exact, reader.count()));
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].0);
    }
}
