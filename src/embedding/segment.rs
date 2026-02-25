use super::config::DistanceMetric;
use super::distance::Distance;
use super::error::Error;
use super::hnsw::{GRAPH_HEADER_SIZE, SENTINEL};
use super::io::{load_delete_file, read_manifest, segment_data_dir, version_dir, ManifestEntry};
use super::platform::{advise_random, advise_sequential};
use super::quantization::QuantizationParams;
use super::DocumentFilter;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Scalar metadata stored as JSON in hnsw.meta.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompactedConfig {
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub num_nodes: usize,
    pub max_level: usize,
    pub entry_point: u32,
    pub node_block_size: usize,
}

/// One HNSW graph segment with its own mmaps and per-segment quantization.
pub struct Segment {
    pub segment_id: u64,
    pub config: CompactedConfig,
    pub raw_vectors: Mmap,
    pub quantized_vectors: Mmap,
    pub graph: Mmap,
    pub doc_ids: Mmap,
    pub levels: Mmap,
    pub deletes: Option<Mmap>,
    pub quantization_params: QuantizationParams,
    pub num_nodes: usize,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub nodes_at_last_rebuild: usize,
    pub insertions_since_rebuild: usize,
}

impl Segment {
    /// Load a segment from disk.
    /// Segment data files are in `segments/seg_{id}/`.
    /// Per-version delete file is in `versions/{version_number}/seg_{id}.del`.
    pub fn load(
        base_path: &Path,
        entry: &ManifestEntry,
        version_number: u64,
    ) -> Result<Self, Error> {
        let seg_dir = segment_data_dir(base_path, entry.segment_id);
        let ver_dir = version_dir(base_path, version_number);

        let meta_path = seg_dir.join("hnsw.meta");
        let contents = std::fs::read_to_string(&meta_path)
            .map_err(|e| Error::CorruptedFile(format!("failed to read hnsw.meta: {e}")))?;
        let config = parse_meta(&contents)?;

        let raw_vectors = load_mmap_with_advice(&seg_dir.join("vectors.raw"), advise_random)?
            .ok_or_else(|| Error::CorruptedFile("missing vectors.raw".into()))?;
        let quantized_vectors =
            load_mmap_with_advice(&seg_dir.join("vectors.quantized"), advise_random)?
                .ok_or_else(|| Error::CorruptedFile("missing vectors.quantized".into()))?;
        let graph = load_mmap_with_advice(&seg_dir.join("hnsw.graph"), advise_random)?
            .ok_or_else(|| Error::CorruptedFile("missing hnsw.graph".into()))?;
        let doc_ids = load_mmap(&seg_dir.join("doc_ids.bin"))?
            .ok_or_else(|| Error::CorruptedFile("missing doc_ids.bin".into()))?;
        let levels = load_mmap(&seg_dir.join("levels.bin"))?
            .ok_or_else(|| Error::CorruptedFile("missing levels.bin".into()))?;

        let quant_path = seg_dir.join("quantization.bin");
        let quantization_params = QuantizationParams::read_from_file(&quant_path)?;

        let deletes = load_delete_file(&ver_dir, entry.segment_id)?;

        Ok(Self {
            segment_id: entry.segment_id,
            config,
            raw_vectors,
            quantized_vectors,
            graph,
            doc_ids,
            levels,
            deletes,
            quantization_params,
            num_nodes: entry.num_nodes,
            min_doc_id: entry.min_doc_id,
            max_doc_id: entry.max_doc_id,
            nodes_at_last_rebuild: entry.nodes_at_last_rebuild,
            insertions_since_rebuild: entry.insertions_since_rebuild,
        })
    }

    /// Get the sorted deleted doc_ids for this segment.
    pub fn deletes_slice(&self) -> &[u64] {
        match self.deletes.as_ref() {
            Some(m) => {
                let ptr = m.as_ptr() as *const u64;
                let len = m.len() / 8;
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            None => &[],
        }
    }

    /// Get the doc_id for a given node index.
    fn doc_id_at(&self, node_idx: u32) -> u64 {
        let offset = node_idx as usize * 8;
        let bytes: [u8; 8] = self.doc_ids[offset..offset + 8].try_into().unwrap();
        u64::from_ne_bytes(bytes)
    }

    /// Get doc_ids as a slice.
    pub fn doc_ids_slice(&self) -> &[u64] {
        let ptr = self.doc_ids.as_ptr() as *const u64;
        let len = self.doc_ids.len() / 8;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get the level of a node.
    fn node_level(&self, node_idx: u32) -> u8 {
        self.levels[node_idx as usize]
    }

    /// Get levels as a byte slice.
    pub fn levels_slice(&self) -> &[u8] {
        &self.levels
    }

    /// Get raw vectors as a flat f32 slice.
    pub fn raw_vectors_slice(&self) -> &[f32] {
        let ptr = self.raw_vectors.as_ptr() as *const f32;
        let len = self.raw_vectors.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get quantized vectors as a flat i8 slice.
    pub fn quantized_vectors_slice(&self) -> &[i8] {
        let ptr = self.quantized_vectors.as_ptr() as *const i8;
        let len = self.quantized_vectors.len();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get the graph mmap bytes.
    pub fn graph_bytes(&self) -> &[u8] {
        &self.graph
    }

    /// Get raw f32 vector for a node.
    fn raw_vector(&self, node_idx: u32, dimensions: usize) -> &[f32] {
        let offset = node_idx as usize * dimensions * 4;
        let ptr = self.raw_vectors[offset..].as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, dimensions) }
    }

    /// Get quantized i8 vector for a node.
    fn quantized_vector(&self, node_idx: u32, dimensions: usize) -> &[i8] {
        let offset = node_idx as usize * dimensions;
        let ptr = self.quantized_vectors[offset..].as_ptr() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, dimensions) }
    }

    /// Get neighbors of a node at a given level from the graph mmap.
    fn get_neighbors(&self, node_idx: u32, level: usize) -> &[u8] {
        let base = GRAPH_HEADER_SIZE + node_idx as usize * self.config.node_block_size;

        let (offset, count) = if level == 0 {
            (base, self.config.m0)
        } else {
            (
                base + self.config.m0 * 4 + (level - 1) * self.config.m * 4,
                self.config.m,
            )
        };

        let end = offset + count * 4;
        &self.graph[offset..end]
    }

    /// Parse neighbor indices from raw bytes, filtering out SENTINEL values.
    fn parse_neighbors(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
        bytes.chunks_exact(4).filter_map(|chunk| {
            let idx = u32::from_ne_bytes(chunk.try_into().unwrap());
            if idx == SENTINEL {
                None
            } else {
                Some(idx)
            }
        })
    }

    /// Two-phase HNSW search reusing caller-provided buffers.
    /// Results are left in `ctx.scored` (sorted by distance, truncated to k).
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_context<D: Distance, F: DocumentFilter>(
        &self,
        query_raw: &[f32],
        query_quantized: &[i8],
        k: usize,
        ef: usize,
        segment_deletes: &[u64],
        live_deletes: &[u64],
        filter: Option<&F>,
        ctx: &mut super::search_context::SearchContext,
    ) {
        if self.config.num_nodes == 0 {
            ctx.scored.clear();
            return;
        }

        let dimensions = self.config.dimensions;
        let entry_point = self.config.entry_point;

        // Phase 1: Navigate upper layers greedily using quantized distance
        let mut current = entry_point;
        let max_node_level = self.node_level(entry_point) as usize;

        for level in (1..=max_node_level).rev() {
            loop {
                let mut changed = false;
                let neighbor_bytes = self.get_neighbors(current, level);
                let current_qvec = self.quantized_vector(current, dimensions);
                let mut best_dist = D::quantized_distance(query_quantized, current_qvec);

                for neighbor_idx in Self::parse_neighbors(neighbor_bytes) {
                    if self.node_level(neighbor_idx) < level as u8 {
                        continue;
                    }
                    let nq = self.quantized_vector(neighbor_idx, dimensions);
                    let d = D::quantized_distance(query_quantized, nq);
                    if d < best_dist {
                        current = neighbor_idx;
                        best_dist = d;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Phase 1b: Beam search at level 0 using quantized distance
        let ef_actual = ef.max(k);
        ctx.candidates_i32.clear();
        ctx.results_i32.clear();
        ctx.visited.clear();
        ctx.visited.grow(self.config.num_nodes);

        let entry_qvec = self.quantized_vector(current, dimensions);
        let entry_dist = D::quantized_distance(query_quantized, entry_qvec);
        ctx.candidates_i32.push(MinHeapItemI32 {
            index: current,
            distance: entry_dist,
        });
        ctx.results_i32.push(MaxHeapItemI32 {
            index: current,
            distance: entry_dist,
        });
        ctx.visited.visit(current as usize);

        while let Some(MinHeapItemI32 {
            index: c_idx,
            distance: c_dist,
        }) = ctx.candidates_i32.pop()
        {
            let worst = ctx
                .results_i32
                .peek()
                .map(|r| r.distance)
                .unwrap_or(i32::MAX);
            if c_dist > worst && ctx.results_i32.len() >= ef_actual {
                break;
            }

            let neighbor_bytes = self.get_neighbors(c_idx, 0);
            for neighbor_idx in Self::parse_neighbors(neighbor_bytes) {
                let n = neighbor_idx as usize;
                if !ctx.visited.visit(n) {
                    continue;
                }

                let nq = self.quantized_vector(neighbor_idx, dimensions);
                let d = D::quantized_distance(query_quantized, nq);

                let worst = ctx
                    .results_i32
                    .peek()
                    .map(|r| r.distance)
                    .unwrap_or(i32::MAX);
                if d < worst || ctx.results_i32.len() < ef_actual {
                    ctx.candidates_i32.push(MinHeapItemI32 {
                        index: neighbor_idx,
                        distance: d,
                    });
                    ctx.results_i32.push(MaxHeapItemI32 {
                        index: neighbor_idx,
                        distance: d,
                    });
                    if ctx.results_i32.len() > ef_actual {
                        ctx.results_i32.pop();
                    }
                }
            }
        }

        // Phase 2: Rescore with raw f32 distance, filter deleted
        ctx.scored.clear();
        while let Some(item) = ctx.results_i32.pop() {
            let doc_id = self.doc_id_at(item.index);
            if segment_deletes.binary_search(&doc_id).is_ok()
                || live_deletes.binary_search(&doc_id).is_ok()
            {
                continue;
            }
            if let Some(f) = filter {
                if !f.contains(doc_id) {
                    continue;
                }
            }
            let raw_vec = self.raw_vector(item.index, dimensions);
            let dist = D::distance(query_raw, raw_vec);
            ctx.scored.push((doc_id, dist));
        }

        ctx.scored
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ctx.scored.truncate(k);
    }

    /// Unchecked doc_id access (for internal use during compaction).
    pub fn doc_id_at_unchecked(&self, node_idx: u32) -> u64 {
        let offset = node_idx as usize * 8;
        let bytes: [u8; 8] = self.doc_ids[offset..offset + 8].try_into().unwrap();
        u64::from_ne_bytes(bytes)
    }

    /// Unchecked raw vector access (for internal use during compaction).
    pub fn raw_vector_unchecked(&self, node_idx: u32, dimensions: usize) -> &[f32] {
        let offset = node_idx as usize * dimensions * 4;
        let ptr = self.raw_vectors[offset..].as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, dimensions) }
    }
}

/// The atomically-swappable segment collection.
pub struct SegmentList {
    pub segments: Vec<Segment>,
    pub version_number: u64,
}

impl SegmentList {
    pub fn empty() -> Self {
        Self {
            segments: Vec::new(),
            version_number: 0,
        }
    }

    pub fn load(base_path: &Path, version_number: u64) -> Result<Self, Error> {
        let ver_dir = version_dir(base_path, version_number);
        let entries = read_manifest(&ver_dir)?;

        let mut segments = Vec::with_capacity(entries.len());
        for entry in &entries {
            segments.push(Segment::load(base_path, entry, version_number)?);
        }

        Ok(Self {
            segments,
            version_number,
        })
    }

    /// Total number of vectors across all segments.
    pub fn total_embeddings(&self) -> usize {
        self.segments.iter().map(|s| s.num_nodes).sum()
    }
}

// --- Helper functions ---

pub fn parse_meta(contents: &str) -> Result<CompactedConfig, Error> {
    let v: serde_json::Value = serde_json::from_str(contents)
        .map_err(|e| Error::CorruptedFile(format!("invalid hnsw.meta JSON: {e}")))?;

    let metric = match v["metric"].as_str() {
        Some("L2") => DistanceMetric::L2,
        Some("DotProduct") => DistanceMetric::DotProduct,
        Some("Cosine") => DistanceMetric::Cosine,
        other => {
            return Err(Error::CorruptedFile(format!(
                "unknown metric in hnsw.meta: {other:?}"
            )))
        }
    };

    let m = v["m"]
        .as_u64()
        .ok_or_else(|| Error::CorruptedFile("missing m in hnsw.meta".into()))? as usize;
    let m0 = v["m0"]
        .as_u64()
        .ok_or_else(|| Error::CorruptedFile("missing m0 in hnsw.meta".into()))?
        as usize;
    let max_level = v["max_level"]
        .as_u64()
        .ok_or_else(|| Error::CorruptedFile("missing max_level in hnsw.meta".into()))?
        as usize;

    // Backward compatibility: old segments without node_block_size use the old (buggy) formula
    let node_block_size = v["node_block_size"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| m0 * 4 + max_level * m * 4);

    Ok(CompactedConfig {
        dimensions: v["dimensions"]
            .as_u64()
            .ok_or_else(|| Error::CorruptedFile("missing dimensions in hnsw.meta".into()))?
            as usize,
        metric,
        m,
        m0,
        ef_construction: v["ef_construction"]
            .as_u64()
            .ok_or_else(|| Error::CorruptedFile("missing ef_construction in hnsw.meta".into()))?
            as usize,
        num_nodes: v["num_nodes"]
            .as_u64()
            .ok_or_else(|| Error::CorruptedFile("missing num_nodes in hnsw.meta".into()))?
            as usize,
        max_level,
        entry_point: v["entry_point"]
            .as_u64()
            .ok_or_else(|| Error::CorruptedFile("missing entry_point in hnsw.meta".into()))?
            as u32,
        node_block_size,
    })
}

pub fn load_mmap(path: &Path) -> Result<Option<Mmap>, Error> {
    load_mmap_with_advice(path, advise_sequential)
}

pub fn load_mmap_with_advice(path: &Path, advice_fn: fn(&Mmap)) -> Result<Option<Mmap>, Error> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        return Ok(None);
    }
    let mmap = unsafe { Mmap::map(&file)? };
    advice_fn(&mmap);
    Ok(Some(mmap))
}

// Min-heap for i32 distances
pub(crate) struct MinHeapItemI32 {
    pub(crate) index: u32,
    pub(crate) distance: i32,
}

impl PartialEq for MinHeapItemI32 {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for MinHeapItemI32 {}
impl PartialOrd for MinHeapItemI32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinHeapItemI32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.cmp(&self.distance) // reversed for min-heap
    }
}

// Max-heap for i32 distances
pub(crate) struct MaxHeapItemI32 {
    pub(crate) index: u32,
    pub(crate) distance: i32,
}

impl PartialEq for MaxHeapItemI32 {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for MaxHeapItemI32 {}
impl PartialOrd for MaxHeapItemI32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxHeapItemI32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}
