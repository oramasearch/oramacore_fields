use super::config::EmbeddingConfig;
use super::distance::Distance;
use super::error::Error;
use rand::RngExt;
use std::collections::BinaryHeap;
use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;

pub const GRAPH_HEADER_SIZE: usize = 16;
pub const SENTINEL: u32 = u32::MAX;

/// Compact visited-set using a bitset (1 bit per node instead of 1 byte).
/// For 1M nodes this uses ~125 KB instead of ~1 MB.
pub(crate) struct VisitedBitset {
    bits: Vec<u64>,
}

impl VisitedBitset {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            bits: vec![0u64; num_nodes.div_ceil(64)],
        }
    }

    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Grow to accommodate at least `num_nodes` nodes.
    pub fn grow(&mut self, num_nodes: usize) {
        let needed = num_nodes.div_ceil(64);
        if needed > self.bits.len() {
            self.bits.resize(needed, 0);
        }
    }

    /// Mark node `i` as visited. Returns `true` if it was NOT previously visited.
    #[inline]
    pub fn visit(&mut self, i: usize) -> bool {
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        if self.bits[word] & bit != 0 {
            return false; // already visited
        }
        self.bits[word] |= bit;
        true
    }
}

/// In-memory HNSW graph builder.
pub struct HnswBuilder<D: Distance> {
    config: EmbeddingConfig,
    nodes: Vec<HnswNode>,
    entry_point: usize,
    current_max_level: usize,
    _phantom: PhantomData<D>,
}

struct HnswNode {
    level: usize,
    neighbors: Vec<Vec<u32>>, // neighbors[layer] = vec of node indices
}

/// Heuristic neighbor selection (Algorithm 4 from HNSW paper).
/// Free function to avoid borrow issues.
fn select_neighbors_heuristic<D: Distance>(
    candidates: &[(u32, f32)],
    max_neighbors: usize,
    vectors: &[f32],
    dimensions: usize,
) -> Vec<u32> {
    if candidates.len() <= max_neighbors {
        return candidates.iter().map(|(idx, _)| *idx).collect();
    }

    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<u32> = Vec::with_capacity(max_neighbors);

    for &(c_idx, c_dist) in &sorted_candidates {
        if selected.len() >= max_neighbors {
            break;
        }

        // Check if this candidate is closer to the query than to any already selected neighbor
        let mut good = true;
        let c_vec = &vectors[c_idx as usize * dimensions..(c_idx as usize + 1) * dimensions];
        for &s_idx in &selected {
            let s_vec = &vectors[s_idx as usize * dimensions..(s_idx as usize + 1) * dimensions];
            let dist_to_selected = D::distance(c_vec, s_vec);
            if dist_to_selected < c_dist {
                good = false;
                break;
            }
        }

        if good {
            selected.push(c_idx);
        }
    }

    // If heuristic didn't fill up, add remaining by distance
    if selected.len() < max_neighbors {
        for &(c_idx, _) in &sorted_candidates {
            if selected.len() >= max_neighbors {
                break;
            }
            if !selected.contains(&c_idx) {
                selected.push(c_idx);
            }
        }
    }

    selected
}

impl<D: Distance> HnswBuilder<D> {
    pub fn new(config: &EmbeddingConfig) -> Self {
        Self {
            config: config.clone(),
            nodes: Vec::new(),
            entry_point: 0,
            current_max_level: 0,
            _phantom: PhantomData,
        }
    }

    /// Build an HNSW graph from a set of vectors (flat f32 buffer).
    /// doc_ids is parallel to vectors: doc_ids[i] is the doc_id for vector i.
    pub fn build(
        &mut self,
        vectors: &[f32],
        doc_ids: &[u64],
        dimensions: usize,
    ) -> Result<(), Error> {
        let num_vectors = vectors.len() / dimensions;
        if num_vectors == 0 {
            return Ok(());
        }
        if num_vectors > u32::MAX as usize {
            return Err(Error::TooManyNodes {
                count: num_vectors,
                max: u32::MAX as usize,
            });
        }

        let mut rng = rand::rng();
        let ml = 1.0 / (self.config.m as f64).ln();

        // Assign levels to all nodes
        let mut levels = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let r: f64 = rng.random();
            let level = ((-r.ln() * ml).floor() as usize).min(self.config.max_level - 1);
            levels.push(level);
        }

        // Initialize nodes
        self.nodes = levels
            .iter()
            .map(|&level| HnswNode {
                level,
                neighbors: (0..=level).map(|_| Vec::new()).collect(),
            })
            .collect();

        // Insert first node as entry point
        self.entry_point = 0;
        self.current_max_level = levels[0];

        // Ignore doc_ids here; they're written separately
        let _ = doc_ids;

        // Insert remaining nodes with a reusable visited bitset
        let mut visited = VisitedBitset::new(num_vectors);
        for idx in 1..num_vectors {
            self.insert_node(idx, &levels, vectors, dimensions, &mut visited);
        }

        Ok(())
    }

    fn insert_node(
        &mut self,
        idx: usize,
        levels: &[usize],
        vectors: &[f32],
        dimensions: usize,
        visited: &mut VisitedBitset,
    ) {
        let node_level = levels[idx];
        let node_vec = &vectors[idx * dimensions..(idx + 1) * dimensions];

        let mut ep = self.entry_point;

        // Navigate from top to node_level + 1 (greedy descent)
        if self.current_max_level > node_level {
            for level in (node_level + 1..=self.current_max_level).rev() {
                ep = self.greedy_closest(ep, node_vec, level, vectors, dimensions);
            }
        }

        // For each level from min(node_level, current_max_level) down to 0:
        // search for ef_construction nearest, connect
        let start_level = node_level.min(self.current_max_level);
        let ef_construction = self.config.ef_construction;
        let m = self.config.m;
        let m0 = self.config.m0;

        for level in (0..=start_level).rev() {
            let max_neighbors = if level == 0 { m0 } else { m };

            visited.clear();
            let candidates = self.search_layer(
                ep,
                node_vec,
                ef_construction,
                level,
                vectors,
                dimensions,
                visited,
            );

            // Select neighbors using heuristic
            let selected =
                select_neighbors_heuristic::<D>(&candidates, max_neighbors, vectors, dimensions);

            // Set neighbors for new node
            self.nodes[idx].neighbors[level] = selected.clone();

            // Add bidirectional connections
            for &neighbor_idx in &selected {
                let ni = neighbor_idx as usize;
                if level >= self.nodes[ni].neighbors.len() {
                    continue;
                }
                self.nodes[ni].neighbors[level].push(idx as u32);
                // Prune if over capacity
                if self.nodes[ni].neighbors[level].len() > max_neighbors {
                    let neighbor_vec = &vectors[ni * dimensions..(ni + 1) * dimensions];
                    let neighbor_candidates: Vec<(u32, f32)> = self.nodes[ni].neighbors[level]
                        .iter()
                        .map(|&n| {
                            let nv =
                                &vectors[n as usize * dimensions..(n as usize + 1) * dimensions];
                            let d = D::distance(neighbor_vec, nv);
                            (n, d)
                        })
                        .collect();
                    let pruned = select_neighbors_heuristic::<D>(
                        &neighbor_candidates,
                        max_neighbors,
                        vectors,
                        dimensions,
                    );
                    self.nodes[ni].neighbors[level] = pruned;
                }
            }

            // Update entry point for next level's search
            if !candidates.is_empty() {
                ep = candidates[0].0 as usize;
            }
        }

        // Update global entry point if new node has higher level
        if node_level > self.current_max_level {
            self.entry_point = idx;
            self.current_max_level = node_level;
        }
    }

    /// Greedy search for the single closest node at a given level.
    fn greedy_closest(
        &self,
        start: usize,
        query: &[f32],
        level: usize,
        vectors: &[f32],
        dimensions: usize,
    ) -> usize {
        let mut current = start;
        let start_vec = &vectors[current * dimensions..(current + 1) * dimensions];
        let mut current_dist = D::distance(query, start_vec);

        loop {
            let mut changed = false;
            let neighbors = &self.nodes[current].neighbors;
            if level >= neighbors.len() {
                break;
            }
            for &neighbor_idx in &neighbors[level] {
                if neighbor_idx == SENTINEL {
                    continue;
                }
                let nv = &vectors
                    [neighbor_idx as usize * dimensions..(neighbor_idx as usize + 1) * dimensions];
                let d = D::distance(query, nv);
                if d < current_dist {
                    current = neighbor_idx as usize;
                    current_dist = d;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search a single layer for ef nearest neighbors.
    /// Returns vec of (node_index, distance) sorted by distance ascending.
    /// The caller must provide a pre-cleared `VisitedBitset` large enough for all nodes.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        ef: usize,
        level: usize,
        vectors: &[f32],
        dimensions: usize,
        visited: &mut VisitedBitset,
    ) -> Vec<(u32, f32)> {
        let entry_vec = &vectors[entry * dimensions..(entry + 1) * dimensions];
        let entry_dist = D::distance(query, entry_vec);

        let mut candidates: BinaryHeap<MinHeapItem> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxHeapItem> = BinaryHeap::new();

        candidates.push(MinHeapItem {
            index: entry as u32,
            distance: entry_dist,
        });
        results.push(MaxHeapItem {
            index: entry as u32,
            distance: entry_dist,
        });
        visited.visit(entry);

        while let Some(MinHeapItem {
            index: c_idx,
            distance: c_dist,
        }) = candidates.pop()
        {
            let worst_result_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if c_dist > worst_result_dist && results.len() >= ef {
                break;
            }

            let neighbors = &self.nodes[c_idx as usize].neighbors;
            if level >= neighbors.len() {
                continue;
            }

            for &neighbor_idx in &neighbors[level] {
                if neighbor_idx == SENTINEL {
                    continue;
                }
                let n = neighbor_idx as usize;
                if !visited.visit(n) {
                    continue;
                }

                let nv = &vectors[n * dimensions..(n + 1) * dimensions];
                let d = D::distance(query, nv);

                let worst_result_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                if d < worst_result_dist || results.len() < ef {
                    candidates.push(MinHeapItem {
                        index: neighbor_idx,
                        distance: d,
                    });
                    results.push(MaxHeapItem {
                        index: neighbor_idx,
                        distance: d,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<(u32, f32)> = results
            .into_iter()
            .map(|item| (item.index, item.distance))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result_vec
    }

    /// Write the HNSW graph to disk.
    pub fn write_graph(&self, path: &Path) -> Result<(), Error> {
        let num_nodes = self.nodes.len() as u32;
        let m = self.config.m as u16;
        let m0 = self.config.m0 as u16;
        let max_level = self.config.max_level as u16;
        let entry_point = self.entry_point as u32;

        let node_block_size = self.config.node_block_size();
        let total_size = GRAPH_HEADER_SIZE + self.nodes.len() * node_block_size;

        let mut buffer = vec![0xFFu8; total_size]; // Fill with SENTINEL bytes

        // Write header
        buffer[0..4].copy_from_slice(&num_nodes.to_ne_bytes());
        buffer[4..6].copy_from_slice(&m.to_ne_bytes());
        buffer[6..8].copy_from_slice(&m0.to_ne_bytes());
        buffer[8..10].copy_from_slice(&max_level.to_ne_bytes());
        buffer[10..14].copy_from_slice(&entry_point.to_ne_bytes());
        buffer[14..16].copy_from_slice(&0u16.to_ne_bytes()); // padding

        // Write per-node adjacency lists
        for (i, node) in self.nodes.iter().enumerate() {
            let base = GRAPH_HEADER_SIZE + i * node_block_size;

            for (level, neighbors) in node.neighbors.iter().enumerate() {
                let offset = if level == 0 {
                    base
                } else {
                    base + self.config.m0 * 4 + (level - 1) * self.config.m * 4
                };

                for (j, &neighbor_idx) in neighbors.iter().enumerate() {
                    let max_for_level = if level == 0 {
                        self.config.m0
                    } else {
                        self.config.m
                    };
                    if j >= max_for_level {
                        break;
                    }
                    let pos = offset + j * 4;
                    buffer[pos..pos + 4].copy_from_slice(&neighbor_idx.to_ne_bytes());
                }
            }
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&buffer)?;
        file.sync_all()?;

        Ok(())
    }

    /// Write per-node levels to disk (one u8 per node).
    pub fn write_levels(&self, path: &Path) -> Result<(), Error> {
        let levels: Vec<u8> = self.nodes.iter().map(|n| n.level as u8).collect();
        let mut file = std::fs::File::create(path)?;
        file.write_all(&levels)?;
        file.sync_all()?;
        Ok(())
    }

    /// Load an existing HNSW graph from its binary format for incremental updates.
    pub fn load_from_graph(
        graph_bytes: &[u8],
        levels_bytes: &[u8],
        config: &EmbeddingConfig,
        node_block_size: usize,
    ) -> Result<Self, Error> {
        if graph_bytes.len() < GRAPH_HEADER_SIZE {
            return Err(Error::CorruptedFile("graph too small for header".into()));
        }

        let num_nodes = u32::from_ne_bytes(graph_bytes[0..4].try_into().unwrap()) as usize;
        let entry_point = u32::from_ne_bytes(graph_bytes[10..14].try_into().unwrap()) as usize;

        if num_nodes == 0 {
            return Ok(Self {
                config: config.clone(),
                nodes: Vec::new(),
                entry_point: 0,
                current_max_level: 0,
                _phantom: PhantomData,
            });
        }

        let m0 = config.m0;
        let m = config.m;

        let mut nodes = Vec::with_capacity(num_nodes);
        let mut current_max_level: usize = 0;

        for (i, &level_byte) in levels_bytes.iter().enumerate().take(num_nodes) {
            let level = level_byte as usize;
            if level > current_max_level {
                current_max_level = level;
            }

            let base = GRAPH_HEADER_SIZE + i * node_block_size;
            let mut neighbors: Vec<Vec<u32>> = Vec::with_capacity(level + 1);

            for lv in 0..=level {
                let (offset, count) = if lv == 0 {
                    (base, m0)
                } else {
                    (base + m0 * 4 + (lv - 1) * m * 4, m)
                };

                let mut layer_neighbors = Vec::new();
                for j in 0..count {
                    let pos = offset + j * 4;
                    let idx = u32::from_ne_bytes(graph_bytes[pos..pos + 4].try_into().unwrap());
                    if idx != SENTINEL {
                        layer_neighbors.push(idx);
                    }
                }
                neighbors.push(layer_neighbors);
            }

            nodes.push(HnswNode { level, neighbors });
        }

        Ok(Self {
            config: config.clone(),
            nodes,
            entry_point,
            current_max_level,
            _phantom: PhantomData,
        })
    }

    /// Insert new nodes into an existing graph (incremental update).
    /// `start_index` is the first index of the new nodes in the combined vectors buffer.
    /// `new_levels` are the assigned levels for each new node.
    /// `all_vectors` is the flat f32 buffer of ALL vectors (old + new).
    pub fn insert_nodes(
        &mut self,
        start_index: usize,
        new_levels: &[usize],
        all_vectors: &[f32],
        dimensions: usize,
    ) -> Result<(), Error> {
        // Extend self.nodes with placeholder entries for new nodes
        for &level in new_levels {
            self.nodes.push(HnswNode {
                level,
                neighbors: (0..=level).map(|_| Vec::new()).collect(),
            });
        }

        // Build a combined levels array for insert_node
        let all_levels: Vec<usize> = self.nodes.iter().map(|n| n.level).collect();

        // Insert each new node with a reusable visited bitset
        let mut visited = VisitedBitset::new(self.nodes.len());
        for i in 0..new_levels.len() {
            let idx = start_index + i;
            visited.grow(self.nodes.len());
            self.insert_node(idx, &all_levels, all_vectors, dimensions, &mut visited);
        }

        Ok(())
    }

    pub fn entry_point(&self) -> u32 {
        self.entry_point as u32
    }

    #[cfg(test)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// Min-heap item (smallest distance at top)
#[derive(Clone)]
struct MinHeapItem {
    index: u32,
    distance: f32,
}

impl PartialEq for MinHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for MinHeapItem {}
impl PartialOrd for MinHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// Max-heap item (largest distance at top)
#[derive(Clone)]
struct MaxHeapItem {
    index: u32,
    distance: f32,
}

impl PartialEq for MaxHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for MaxHeapItem {}
impl PartialOrd for MaxHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::distance::L2;
    use tempfile::TempDir;

    #[test]
    fn test_build_small_graph() {
        let config = EmbeddingConfig::new(2, crate::embedding::config::DistanceMetric::L2).unwrap();
        let mut builder = HnswBuilder::<L2>::new(&config);

        let vectors = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            0.0, 1.0, // node 2
            1.0, 1.0, // node 3
        ];
        let doc_ids = vec![10, 20, 30, 40];

        builder.build(&vectors, &doc_ids, 2).unwrap();
        assert_eq!(builder.num_nodes(), 4);
    }

    #[test]
    fn test_write_graph() {
        let config = EmbeddingConfig::new(2, crate::embedding::config::DistanceMetric::L2).unwrap();
        let mut builder = HnswBuilder::<L2>::new(&config);

        let vectors = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let doc_ids = vec![10, 20, 30];

        builder.build(&vectors, &doc_ids, 2).unwrap();

        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("hnsw.graph");
        builder.write_graph(&graph_path).unwrap();
        assert!(graph_path.exists());

        let levels_path = tmp.path().join("levels.bin");
        builder.write_levels(&levels_path).unwrap();
        assert!(levels_path.exists());
    }

    #[test]
    fn test_empty_build() {
        let config = EmbeddingConfig::new(2, crate::embedding::config::DistanceMetric::L2).unwrap();
        let mut builder = HnswBuilder::<L2>::new(&config);
        builder.build(&[], &[], 2).unwrap();
        assert_eq!(builder.num_nodes(), 0);
    }
}
