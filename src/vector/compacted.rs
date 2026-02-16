use super::config::DistanceMetric;
use super::distance::{DistanceFn, QuantizedDistanceFn};
use super::error::Error;
use super::hnsw::{GRAPH_HEADER_SIZE, SENTINEL};
use super::io::version_dir;
use super::platform::{advise_random, advise_sequential};
use super::quantization::QuantizationParams;
use memmap2::Mmap;
use std::collections::BinaryHeap;
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
}

pub struct CompactedVersion {
    pub version_number: u64,
    pub config: Option<CompactedConfig>,
    pub raw_vectors: Option<Mmap>,
    pub quantized_vectors: Option<Mmap>,
    pub graph: Option<Mmap>,
    pub doc_ids: Option<Mmap>,
    pub levels: Option<Mmap>,
    pub deletes: Option<Mmap>,
    pub quantization_params: Option<QuantizationParams>,
}

impl CompactedVersion {
    pub fn empty() -> Self {
        Self {
            version_number: 0,
            config: None,
            raw_vectors: None,
            quantized_vectors: None,
            graph: None,
            doc_ids: None,
            levels: None,
            deletes: None,
            quantization_params: None,
        }
    }

    pub fn load(base_path: &Path, version_number: u64) -> Result<Self, Error> {
        let dir = version_dir(base_path, version_number);

        // Load hnsw.meta (JSON)
        let meta_path = dir.join("hnsw.meta");
        let config = if meta_path.exists() {
            let contents = std::fs::read_to_string(&meta_path)
                .map_err(|e| Error::CorruptedFile(format!("failed to read hnsw.meta: {e}")))?;
            Some(Self::parse_meta(&contents)?)
        } else {
            None
        };

        let raw_vectors = Self::load_mmap(&dir.join("vectors.raw"))?;
        let quantized_vectors = Self::load_mmap(&dir.join("vectors.quantized"))?;
        let graph = Self::load_mmap_with_advice(&dir.join("hnsw.graph"), advise_random)?;
        let doc_ids = Self::load_mmap(&dir.join("doc_ids.bin"))?;
        let levels = Self::load_mmap(&dir.join("levels.bin"))?;
        let deletes = Self::load_mmap(&dir.join("deleted.bin"))?;

        let quant_path = dir.join("quantization.bin");
        let quantization_params = if quant_path.exists() {
            Some(QuantizationParams::read_from_file(&quant_path)?)
        } else {
            None
        };

        Ok(Self {
            version_number,
            config,
            raw_vectors,
            quantized_vectors,
            graph,
            doc_ids,
            levels,
            deletes,
            quantization_params,
        })
    }

    fn parse_meta(contents: &str) -> Result<CompactedConfig, Error> {
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

        Ok(CompactedConfig {
            dimensions: v["dimensions"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing dimensions in hnsw.meta".into())
            })? as usize,
            metric,
            m: v["m"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing m in hnsw.meta".into())
            })? as usize,
            m0: v["m0"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing m0 in hnsw.meta".into())
            })? as usize,
            ef_construction: v["ef_construction"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing ef_construction in hnsw.meta".into())
            })? as usize,
            num_nodes: v["num_nodes"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing num_nodes in hnsw.meta".into())
            })? as usize,
            max_level: v["max_level"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing max_level in hnsw.meta".into())
            })? as usize,
            entry_point: v["entry_point"].as_u64().ok_or_else(|| {
                Error::CorruptedFile("missing entry_point in hnsw.meta".into())
            })? as u32,
        })
    }

    fn load_mmap(path: &Path) -> Result<Option<Mmap>, Error> {
        Self::load_mmap_with_advice(path, advise_sequential)
    }

    fn load_mmap_with_advice(
        path: &Path,
        advice_fn: fn(&Mmap),
    ) -> Result<Option<Mmap>, Error> {
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

    /// Get the sorted deleted doc_ids.
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
        let mmap = self.doc_ids.as_ref().expect("doc_ids mmap missing");
        let offset = node_idx as usize * 8;
        let bytes: [u8; 8] = mmap[offset..offset + 8].try_into().unwrap();
        u64::from_ne_bytes(bytes)
    }

    /// Get the level of a node.
    fn node_level(&self, node_idx: u32) -> u8 {
        let mmap = self.levels.as_ref().expect("levels mmap missing");
        mmap[node_idx as usize]
    }

    /// Get neighbors of a node at a given level from the graph mmap.
    fn get_neighbors(&self, node_idx: u32, level: usize, config: &CompactedConfig) -> &[u8] {
        let graph = self.graph.as_ref().expect("graph mmap missing");
        let node_block_size = config.m0 * 4 + config.max_level * config.m * 4;
        let base = GRAPH_HEADER_SIZE + node_idx as usize * node_block_size;

        let (offset, count) = if level == 0 {
            (base, config.m0)
        } else {
            (
                base + config.m0 * 4 + (level - 1) * config.m * 4,
                config.m,
            )
        };

        let end = offset + count * 4;
        &graph[offset..end]
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

    /// Get raw f32 vector for a node.
    fn raw_vector(&self, node_idx: u32, dimensions: usize) -> &[f32] {
        let mmap = self.raw_vectors.as_ref().expect("raw_vectors mmap missing");
        let offset = node_idx as usize * dimensions * 4;
        let len = dimensions;
        let ptr = mmap[offset..].as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get quantized i8 vector for a node.
    fn quantized_vector(&self, node_idx: u32, dimensions: usize) -> &[i8] {
        let mmap = self
            .quantized_vectors
            .as_ref()
            .expect("quantized_vectors mmap missing");
        let offset = node_idx as usize * dimensions;
        let ptr = mmap[offset..].as_ptr() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, dimensions) }
    }

    /// Two-phase HNSW search on mmap'd data.
    /// Phase 1: Quantized distance for beam search.
    /// Phase 2: Rescore top candidates with raw f32 distance.
    pub fn search(
        &self,
        query_raw: &[f32],
        query_quantized: &[i8],
        k: usize,
        ef: usize,
        distance_fn: DistanceFn,
        quantized_distance_fn: QuantizedDistanceFn,
        deleted: &[u64],
    ) -> Vec<(u64, f32)> {
        let config = match &self.config {
            Some(c) => c,
            None => return Vec::new(),
        };

        if self.graph.is_none() || config.num_nodes == 0 {
            return Vec::new();
        }

        let dimensions = config.dimensions;
        let entry_point = config.entry_point;

        // Phase 1: Navigate upper layers greedily using quantized distance
        let mut current = entry_point;
        let max_node_level = self.node_level(entry_point) as usize;

        for level in (1..=max_node_level).rev() {
            loop {
                let mut changed = false;
                let neighbor_bytes = self.get_neighbors(current, level, config);
                let current_qvec = self.quantized_vector(current, dimensions);
                let current_dist = quantized_distance_fn(query_quantized, current_qvec);

                for neighbor_idx in Self::parse_neighbors(neighbor_bytes) {
                    if self.node_level(neighbor_idx) < level as u8 {
                        continue;
                    }
                    let nq = self.quantized_vector(neighbor_idx, dimensions);
                    let d = quantized_distance_fn(query_quantized, nq);
                    if d < current_dist {
                        current = neighbor_idx;
                        changed = true;
                        break;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Phase 1b: Beam search at level 0 using quantized distance
        let ef_actual = ef.max(k);
        let mut candidates: BinaryHeap<MinHeapItemI32> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxHeapItemI32> = BinaryHeap::new();
        let mut visited = vec![false; config.num_nodes];

        let entry_qvec = self.quantized_vector(current, dimensions);
        let entry_dist = quantized_distance_fn(query_quantized, entry_qvec);
        candidates.push(MinHeapItemI32 {
            index: current,
            distance: entry_dist,
        });
        results.push(MaxHeapItemI32 {
            index: current,
            distance: entry_dist,
        });
        visited[current as usize] = true;

        while let Some(MinHeapItemI32 {
            index: c_idx,
            distance: c_dist,
        }) = candidates.pop()
        {
            let worst = results.peek().map(|r| r.distance).unwrap_or(i32::MAX);
            if c_dist > worst && results.len() >= ef_actual {
                break;
            }

            let neighbor_bytes = self.get_neighbors(c_idx, 0, config);
            for neighbor_idx in Self::parse_neighbors(neighbor_bytes) {
                let n = neighbor_idx as usize;
                if visited[n] {
                    continue;
                }
                visited[n] = true;

                let nq = self.quantized_vector(neighbor_idx, dimensions);
                let d = quantized_distance_fn(query_quantized, nq);

                let worst = results.peek().map(|r| r.distance).unwrap_or(i32::MAX);
                if d < worst || results.len() < ef_actual {
                    candidates.push(MinHeapItemI32 {
                        index: neighbor_idx,
                        distance: d,
                    });
                    results.push(MaxHeapItemI32 {
                        index: neighbor_idx,
                        distance: d,
                    });
                    if results.len() > ef_actual {
                        results.pop();
                    }
                }
            }
        }

        // Phase 2: Rescore with raw f32 distance, filter deleted
        let mut scored: Vec<(u64, f32)> = results
            .into_iter()
            .filter_map(|item| {
                let doc_id = self.doc_id_at(item.index);
                if deleted.binary_search(&doc_id).is_ok() {
                    return None;
                }
                let raw_vec = self.raw_vector(item.index, dimensions);
                let dist = distance_fn(query_raw, raw_vec);
                Some((doc_id, dist))
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// Min-heap for i32 distances
struct MinHeapItemI32 {
    index: u32,
    distance: i32,
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
struct MaxHeapItemI32 {
    index: u32,
    distance: i32,
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
