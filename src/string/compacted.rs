use super::io::{read_manifest, segment_data_dir, version_dir, ManifestEntry};
use super::platform::advise_random;
use anyhow::{Context, Result};
use fst::automaton::{Levenshtein, Str};
use fst::{Automaton, IntoStreamer, Map, Streamer};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// ─── Segment ───────────────────────────────────────────────────────────────

/// An immutable on-disk segment: one (FST, postings, doc_lengths) triple.
pub struct Segment {
    pub segment_id: u64,
    fst_map: Option<Map<Mmap>>,
    postings_mmap: Option<Mmap>,
    doc_lengths_mmap: Option<Mmap>,
    pub num_postings: usize,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
}

impl Segment {
    /// Load a segment from `base_path/segments/seg_{id}/`.
    pub fn load(base_path: &Path, entry: &ManifestEntry) -> Result<Self> {
        let dir = segment_data_dir(base_path, entry.segment_id);

        let fst_path = dir.join("keys.fst");
        let postings_path = dir.join("postings.dat");
        let doc_lengths_path = dir.join("doc_lengths.dat");

        let fst_map = load_fst(&fst_path)?;
        let postings_mmap = load_mmap(&postings_path)?;
        let doc_lengths_mmap = load_mmap(&doc_lengths_path)?;

        Ok(Self {
            segment_id: entry.segment_id,
            fst_map,
            postings_mmap,
            doc_lengths_mmap,
            num_postings: entry.num_postings,
            min_doc_id: entry.min_doc_id,
            max_doc_id: entry.max_doc_id,
        })
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

    /// Look up the field length for a doc_id using galloping (exponential) search
    /// starting from `*cursor`.
    #[inline]
    pub fn field_length_galloping(&self, doc_id: u64, cursor: &mut usize) -> Option<u32> {
        let mmap = self.doc_lengths_mmap.as_ref()?;
        let data = mmap.as_ref();
        let entry_size = 12; // u64 + u32
        let count = data.len() / entry_size;

        if count == 0 {
            return None;
        }

        let mut lo = *cursor;
        if lo >= count {
            lo = 0;
        }

        // Quick check at cursor position
        let lo_byte = lo * entry_size;
        let lo_doc = u64::from_ne_bytes(data[lo_byte..lo_byte + 8].try_into().ok()?);
        if lo_doc == doc_id {
            let field_len = u32::from_ne_bytes(data[lo_byte + 8..lo_byte + 12].try_into().ok()?);
            *cursor = lo;
            return Some(field_len);
        }
        if lo_doc > doc_id {
            lo = 0;
        }

        // Galloping: jump forward by doubling steps until we overshoot
        let mut jump = 1usize;
        let mut hi = lo + jump;
        while hi < count {
            let byte_off = hi * entry_size;
            let hi_doc = u64::from_ne_bytes(data[byte_off..byte_off + 8].try_into().ok()?);
            if hi_doc == doc_id {
                let field_len =
                    u32::from_ne_bytes(data[byte_off + 8..byte_off + 12].try_into().ok()?);
                *cursor = hi;
                return Some(field_len);
            }
            if hi_doc > doc_id {
                break;
            }
            lo = hi;
            jump *= 2;
            hi = lo + jump;
        }
        if hi > count {
            hi = count;
        }

        // Binary search within the narrowed range [lo, hi)
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let byte_off = mid * entry_size;
            let mid_doc = u64::from_ne_bytes(data[byte_off..byte_off + 8].try_into().ok()?);
            match mid_doc.cmp(&doc_id) {
                std::cmp::Ordering::Equal => {
                    let field_len =
                        u32::from_ne_bytes(data[byte_off + 8..byte_off + 12].try_into().ok()?);
                    *cursor = mid;
                    return Some(field_len);
                }
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }

        *cursor = lo;
        None
    }

    /// Callback-based term search: invokes `f(is_exact, reader)` for each matching term.
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
                for_each_automaton_match(fst_map, postings_mmap, automaton, token, &mut f);
            }
            Some(n) => {
                let automaton = Levenshtein::new(token, n as u32).map_err(|e| {
                    anyhow::anyhow!(
                        "Levenshtein automaton construction failed for '{token}' with distance {n}: {e}"
                    )
                })?;
                for_each_automaton_match(fst_map, postings_mmap, automaton, token, &mut f);
            }
        }

        Ok(())
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
}

// ─── SegmentList ───────────────────────────────────────────────────────────

/// A versioned collection of immutable segments plus global delete list and stats.
pub struct SegmentList {
    pub segments: Vec<Segment>,
    pub version_number: u64,
    deleted_mmap: Option<Mmap>,
    pub total_document_length: u64,
    pub total_documents: u64,
}

impl SegmentList {
    pub fn empty() -> Self {
        Self {
            segments: Vec::new(),
            version_number: 0,
            deleted_mmap: None,
            total_document_length: 0,
            total_documents: 0,
        }
    }

    pub fn load(base_path: &Path, version_number: u64) -> Result<Self> {
        let ver_dir = version_dir(base_path, version_number);

        let manifest = read_manifest(&ver_dir)?;

        let mut segments = Vec::with_capacity(manifest.len());
        for entry in &manifest {
            segments.push(Segment::load(base_path, entry)?);
        }

        let deleted_path = ver_dir.join("deleted.bin");
        let deleted_mmap = load_mmap(&deleted_path)?;

        let global_info_path = ver_dir.join("global_info.bin");
        let (total_document_length, total_documents) =
            super::io::read_global_info(&global_info_path)?;

        Ok(Self {
            segments,
            version_number,
            deleted_mmap,
            total_document_length,
            total_documents,
        })
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

    /// Returns true if any segment has data.
    pub fn has_data(&self) -> bool {
        !self.segments.is_empty() || self.deleted_mmap.is_some()
    }
}

// ─── build_segment_data ────────────────────────────────────────────────────

/// Result of building a single segment.
pub struct SegmentBuildResult {
    pub num_postings: usize,
    pub total_doc_length: u64,
    pub total_documents: u64,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
}

/// Build a single segment's on-disk data (keys.fst, postings.dat, doc_lengths.dat)
/// by merging compacted + live sources, filtering deletions.
///
/// Returns per-segment metadata. Stats reflect only what's written (deleted docs excluded).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_segment_data<'a, P: std::ops::Deref<Target = [u32]>>(
    compacted_terms: &mut CompactedTermIterator<'a>,
    live_terms: &[(&str, &[(u64, P, P)])],
    compacted_doc_lengths: &mut DocLengthIterator<'_>,
    live_doc_lengths: &[(u64, u16)],
    deleted_set: Option<&[u64]>,
    segment_dir: &Path,
) -> Result<SegmentBuildResult> {
    let fst_path = segment_dir.join("keys.fst");
    let postings_path = segment_dir.join("postings.dat");
    let doc_lengths_path = segment_dir.join("doc_lengths.dat");

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
    let mut entries_buf: Vec<u8> = Vec::new();
    let mut num_postings: usize = 0;

    let mut compacted_entry: Option<PostingsReader<'a>> =
        compacted_terms.next_term_into(&mut compacted_key_buf);

    let mut delete_cursor = SortedDeleteCursor::new(deleted_set);

    loop {
        delete_cursor.reset();

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
                    if delete_cursor.should_keep(entry.doc_id) {
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
                    num_postings += count as usize;
                }
                compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
            }
            (None, Some(_)) => {
                let (live_key, live_postings) = live_terms[live_idx];
                live_idx += 1;
                entries_buf.clear();
                let mut count: u32 = 0;
                for &(doc_id, ref exact, ref stemmed) in live_postings {
                    if delete_cursor.should_keep(doc_id) {
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
                    num_postings += count as usize;
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
                            if delete_cursor.should_keep(entry.doc_id) {
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
                            num_postings += count as usize;
                        }
                        compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
                    }
                    std::cmp::Ordering::Greater => {
                        compacted_entry = Some(reader);
                        let (live_key, live_postings) = live_terms[live_idx];
                        live_idx += 1;
                        entries_buf.clear();
                        let mut count: u32 = 0;
                        for &(doc_id, ref exact, ref stemmed) in live_postings {
                            if delete_cursor.should_keep(doc_id) {
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
                            num_postings += count as usize;
                        }
                    }
                    std::cmp::Ordering::Equal => {
                        let mut reader = reader;
                        let (_, live_postings) = live_terms[live_idx];
                        live_idx += 1;

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
                                    if delete_cursor.should_keep(c.doc_id) {
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
                                    if delete_cursor.should_keep(l.0) {
                                        write_entry_to_buf(&mut entries_buf, l.0, &l.1, &l.2);
                                        count += 1;
                                    }
                                    li += 1;
                                }
                                (Some(c), Some(l)) => match c.doc_id.cmp(&l.0) {
                                    std::cmp::Ordering::Less => {
                                        if delete_cursor.should_keep(c.doc_id) {
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
                                        if delete_cursor.should_keep(l.0) {
                                            write_entry_to_buf(&mut entries_buf, l.0, &l.1, &l.2);
                                            count += 1;
                                        }
                                        li += 1;
                                    }
                                    std::cmp::Ordering::Equal => {
                                        // Live wins
                                        if delete_cursor.should_keep(l.0) {
                                            write_entry_to_buf(&mut entries_buf, l.0, &l.1, &l.2);
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
                            num_postings += count as usize;
                        }
                        compacted_entry = compacted_terms.next_term_into(&mut compacted_key_buf);
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

    // ── Build doc_lengths ──
    let (total_doc_length, total_documents, min_doc_id, max_doc_id) = merge_and_write_doc_lengths(
        &doc_lengths_path,
        compacted_doc_lengths,
        live_doc_lengths,
        deleted_set,
    )?;

    Ok(SegmentBuildResult {
        num_postings,
        total_doc_length,
        total_documents,
        min_doc_id,
        max_doc_id,
    })
}

// ─── Helper types ──────────────────────────────────────────────────────────

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

        let doc_id = u64::from_ne_bytes(data[pos..pos + 8].try_into().ok()?);
        let exact_count = u32::from_ne_bytes(data[pos + 8..pos + 12].try_into().ok()?) as usize;
        let stemmed_count = u32::from_ne_bytes(data[pos + 12..pos + 16].try_into().ok()?) as usize;

        let cursor = pos + 16;

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
    /// Create an empty iterator (no data).
    pub fn empty() -> Self {
        Self {
            stream: None,
            postings_mmap: None,
        }
    }

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

impl<'a> DocLengthIterator<'a> {
    /// Create an empty iterator.
    pub fn empty() -> Self {
        Self { data: &[], pos: 0 }
    }
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

// ─── Internal helpers ──────────────────────────────────────────────────────

fn load_fst(path: &Path) -> Result<Option<Map<Mmap>>> {
    if !path.exists() {
        return Ok(None);
    }

    let file = File::open(path).with_context(|| format!("Failed to open FST file: {path:?}"))?;

    let metadata = file
        .metadata()
        .with_context(|| format!("Failed to get FST metadata: {path:?}"))?;

    if metadata.len() == 0 {
        return Ok(None);
    }

    let mmap =
        unsafe { Mmap::map(&file).with_context(|| format!("Failed to mmap FST file: {path:?}"))? };

    advise_random(&mmap);

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

    advise_random(&mmap);

    Ok(Some(mmap))
}

/// Generic helper: stream FST matches using the given automaton and invoke callback per match.
fn for_each_automaton_match<'a, A: fst::Automaton, F>(
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

/// Serialize a single posting entry (doc_id + positions) into a byte buffer.
#[inline]
pub(super) fn write_entry_to_buf(buf: &mut Vec<u8>, doc_id: u64, exact: &[u32], stemmed: &[u32]) {
    let needed = 8 + 4 + 4 + (exact.len() + stemmed.len()) * 4;
    buf.reserve(needed);
    buf.extend_from_slice(&doc_id.to_ne_bytes());
    buf.extend_from_slice(&(exact.len() as u32).to_ne_bytes());
    buf.extend_from_slice(&(stemmed.len() as u32).to_ne_bytes());
    let exact_bytes =
        unsafe { std::slice::from_raw_parts(exact.as_ptr() as *const u8, exact.len() * 4) };
    buf.extend_from_slice(exact_bytes);
    let stemmed_bytes =
        unsafe { std::slice::from_raw_parts(stemmed.as_ptr() as *const u8, stemmed.len() * 4) };
    buf.extend_from_slice(stemmed_bytes);
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

/// Cursor-based membership checker for a sorted slice of `u64` values.
pub(super) struct SortedDeleteCursor<'a> {
    slice: Option<&'a [u64]>,
    pos: usize,
}

impl<'a> SortedDeleteCursor<'a> {
    pub(super) fn new(slice: Option<&'a [u64]>) -> Self {
        Self { slice, pos: 0 }
    }

    pub(super) fn reset(&mut self) {
        self.pos = 0;
    }

    /// Returns `true` if `doc_id` should be kept (i.e., is NOT in the delete set).
    pub(super) fn should_keep(&mut self, doc_id: u64) -> bool {
        let slice = match self.slice {
            None => return true,
            Some(s) => s,
        };
        while self.pos < slice.len() && slice[self.pos] < doc_id {
            self.pos += 1;
        }
        self.pos >= slice.len() || slice[self.pos] != doc_id
    }
}

/// Merge compacted doc_lengths with live doc_lengths, write directly to disk,
/// and return (total_doc_length, total_documents, min_doc_id, max_doc_id).
fn merge_and_write_doc_lengths(
    path: &Path,
    compacted: &mut DocLengthIterator<'_>,
    live: &[(u64, u16)],
    deleted_set: Option<&[u64]>,
) -> Result<(u64, u64, u64, u64)> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create doc_lengths file: {path:?}"))?;
    let mut writer = BufWriter::new(file);

    let mut total_doc_length: u64 = 0;
    let mut total_documents: u64 = 0;
    let mut min_doc_id: u64 = u64::MAX;
    let mut max_doc_id: u64 = 0;
    let mut li = 0;
    let mut compacted_next = compacted.next();

    let mut delete_cursor = SortedDeleteCursor::new(deleted_set);

    loop {
        let live_peek = if li < live.len() {
            Some((live[li].0, live[li].1 as u32))
        } else {
            None
        };

        match (compacted_next, live_peek) {
            (None, None) => break,
            (Some((c_id, c_len)), None) => {
                if delete_cursor.should_keep(c_id) {
                    writer.write_all(&c_id.to_ne_bytes())?;
                    writer.write_all(&c_len.to_ne_bytes())?;
                    total_doc_length += c_len as u64;
                    total_documents += 1;
                    if c_id < min_doc_id {
                        min_doc_id = c_id;
                    }
                    if c_id > max_doc_id {
                        max_doc_id = c_id;
                    }
                }
                compacted_next = compacted.next();
            }
            (None, Some((l_id, l_len))) => {
                if delete_cursor.should_keep(l_id) {
                    writer.write_all(&l_id.to_ne_bytes())?;
                    writer.write_all(&l_len.to_ne_bytes())?;
                    total_doc_length += l_len as u64;
                    total_documents += 1;
                    if l_id < min_doc_id {
                        min_doc_id = l_id;
                    }
                    if l_id > max_doc_id {
                        max_doc_id = l_id;
                    }
                }
                li += 1;
            }
            (Some((c_id, c_len)), Some((l_id, l_len))) => match c_id.cmp(&l_id) {
                std::cmp::Ordering::Less => {
                    if delete_cursor.should_keep(c_id) {
                        writer.write_all(&c_id.to_ne_bytes())?;
                        writer.write_all(&c_len.to_ne_bytes())?;
                        total_doc_length += c_len as u64;
                        total_documents += 1;
                        if c_id < min_doc_id {
                            min_doc_id = c_id;
                        }
                        if c_id > max_doc_id {
                            max_doc_id = c_id;
                        }
                    }
                    compacted_next = compacted.next();
                }
                std::cmp::Ordering::Greater => {
                    if delete_cursor.should_keep(l_id) {
                        writer.write_all(&l_id.to_ne_bytes())?;
                        writer.write_all(&l_len.to_ne_bytes())?;
                        total_doc_length += l_len as u64;
                        total_documents += 1;
                        if l_id < min_doc_id {
                            min_doc_id = l_id;
                        }
                        if l_id > max_doc_id {
                            max_doc_id = l_id;
                        }
                    }
                    li += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Live wins
                    if delete_cursor.should_keep(l_id) {
                        writer.write_all(&l_id.to_ne_bytes())?;
                        writer.write_all(&l_len.to_ne_bytes())?;
                        total_doc_length += l_len as u64;
                        total_documents += 1;
                        if l_id < min_doc_id {
                            min_doc_id = l_id;
                        }
                        if l_id > max_doc_id {
                            max_doc_id = l_id;
                        }
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

    // If no documents were written, normalize min/max
    if total_documents == 0 {
        min_doc_id = 0;
        max_doc_id = 0;
    }

    Ok((total_doc_length, total_documents, min_doc_id, max_doc_id))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::type_complexity)]

    use super::super::io::{
        ensure_segment_dir, ensure_version_dir, write_deleted, write_global_info, write_manifest,
        ManifestEntry,
    };
    use super::*;
    use tempfile::TempDir;

    fn build_simple_segment(
        entries: &[(&str, Vec<(u64, Vec<u32>, Vec<u32>)>)],
        doc_lengths: &[(u64, u16)],
        segment_dir: &Path,
    ) -> SegmentBuildResult {
        let mut compacted_terms = CompactedTermIterator::empty();
        let live: Vec<_> = entries.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let mut compacted_dl = DocLengthIterator::empty();

        build_segment_data(
            &mut compacted_terms,
            &live,
            &mut compacted_dl,
            doc_lengths,
            None,
            segment_dir,
        )
        .unwrap()
    }

    #[test]
    fn test_segment_build_and_load() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let seg_dir = ensure_segment_dir(base_path, 0).unwrap();

        let result = build_simple_segment(
            &[
                (
                    "hello",
                    vec![(1, vec![0], vec![]), (2, vec![0, 3], vec![1])],
                ),
                ("world", vec![(3, vec![1], vec![])]),
            ],
            &[(1, 5), (2, 10), (3, 3)],
            &seg_dir,
        );

        assert_eq!(result.num_postings, 3);
        assert_eq!(result.total_documents, 3);
        assert_eq!(result.total_doc_length, 18);
        assert_eq!(result.min_doc_id, 1);
        assert_eq!(result.max_doc_id, 3);

        // Create manifest and load via SegmentList
        let ver_dir = ensure_version_dir(base_path, 1).unwrap();
        let manifest = vec![ManifestEntry {
            segment_id: 0,
            num_postings: result.num_postings,
            num_deletes: 0,
            min_doc_id: result.min_doc_id,
            max_doc_id: result.max_doc_id,
            total_doc_length: result.total_doc_length,
            total_documents: result.total_documents,
        }];
        write_manifest(&ver_dir, &manifest).unwrap();
        write_deleted(&ver_dir.join("deleted.bin"), &[]).unwrap();
        write_global_info(
            &ver_dir.join("global_info.bin"),
            result.total_doc_length,
            result.total_documents,
        )
        .unwrap();

        let seg_list = SegmentList::load(base_path, 1).unwrap();
        assert_eq!(seg_list.segments.len(), 1);

        let seg = &seg_list.segments[0];
        let mut reader = seg.lookup_postings("hello").unwrap();
        let e1 = reader.next().unwrap();
        assert_eq!(e1.doc_id, 1);
        let e2 = reader.next().unwrap();
        assert_eq!(e2.doc_id, 2);
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_empty_segment() {
        let tmp = TempDir::new().unwrap();
        let seg_dir = ensure_segment_dir(tmp.path(), 0).unwrap();

        let result = build_simple_segment(&[], &[], &seg_dir);
        assert_eq!(result.num_postings, 0);
        assert_eq!(result.total_documents, 0);
    }

    #[test]
    fn test_segment_field_length_galloping() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let seg_dir = ensure_segment_dir(base_path, 0).unwrap();

        build_simple_segment(
            &[("a", vec![(1, vec![0], vec![]), (5, vec![0], vec![])])],
            &[(1, 3), (5, 7)],
            &seg_dir,
        );

        let ver_dir = ensure_version_dir(base_path, 1).unwrap();
        let manifest = vec![ManifestEntry {
            segment_id: 0,
            num_postings: 2,
            num_deletes: 0,
            min_doc_id: 1,
            max_doc_id: 5,
            total_doc_length: 10,
            total_documents: 2,
        }];
        write_manifest(&ver_dir, &manifest).unwrap();
        write_deleted(&ver_dir.join("deleted.bin"), &[]).unwrap();
        write_global_info(&ver_dir.join("global_info.bin"), 10, 2).unwrap();

        let seg_list = SegmentList::load(base_path, 1).unwrap();
        let seg = &seg_list.segments[0];

        let mut cursor = 0;
        assert_eq!(seg.field_length_galloping(1, &mut cursor), Some(3));
        assert_eq!(seg.field_length_galloping(5, &mut cursor), Some(7));
        assert_eq!(seg.field_length_galloping(999, &mut cursor), None);
    }

    #[test]
    fn test_segment_for_each_term_match() {
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path();
        let seg_dir = ensure_segment_dir(base_path, 0).unwrap();

        build_simple_segment(
            &[
                ("apple", vec![(1, vec![0], vec![])]),
                ("application", vec![(2, vec![0], vec![])]),
                ("banana", vec![(3, vec![0], vec![])]),
            ],
            &[(1, 3), (2, 3), (3, 3)],
            &seg_dir,
        );

        let ver_dir = ensure_version_dir(base_path, 1).unwrap();
        let manifest = vec![ManifestEntry {
            segment_id: 0,
            num_postings: 3,
            num_deletes: 0,
            min_doc_id: 1,
            max_doc_id: 3,
            total_doc_length: 9,
            total_documents: 3,
        }];
        write_manifest(&ver_dir, &manifest).unwrap();
        write_deleted(&ver_dir.join("deleted.bin"), &[]).unwrap();
        write_global_info(&ver_dir.join("global_info.bin"), 9, 3).unwrap();

        let seg_list = SegmentList::load(base_path, 1).unwrap();
        let seg = &seg_list.segments[0];

        // Prefix search
        let mut results: Vec<(bool, usize)> = Vec::new();
        seg.for_each_term_match("app", None, |is_exact, reader| {
            results.push((is_exact, reader.count()));
        })
        .unwrap();
        assert_eq!(results.len(), 2);
    }
}
