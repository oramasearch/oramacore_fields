use super::hnsw::VisitedBitset;
use super::live::HeapItem;
use super::segment::{MaxHeapItemI32, MinHeapItemI32};
use std::collections::{BinaryHeap, HashSet};

/// Reusable buffers for HNSW beam search inside a single segment.
pub(crate) struct SegmentSearchBuffers {
    pub(crate) candidates_i32: BinaryHeap<MinHeapItemI32>,
    pub(crate) results_i32: BinaryHeap<MaxHeapItemI32>,
    pub(crate) visited: VisitedBitset,
    pub(crate) scored: Vec<(u64, f32)>,
}

impl SegmentSearchBuffers {
    fn new() -> Self {
        Self {
            candidates_i32: BinaryHeap::new(),
            results_i32: BinaryHeap::new(),
            visited: VisitedBitset::new(0),
            scored: Vec::new(),
        }
    }
}

/// Reusable buffers for brute-force live layer search.
pub(crate) struct LiveSearchBuffers {
    pub(crate) live_heap: BinaryHeap<HeapItem>,
    pub(crate) live_results: Vec<(u64, f32)>,
}

impl LiveSearchBuffers {
    fn new() -> Self {
        Self {
            live_heap: BinaryHeap::new(),
            live_results: Vec::new(),
        }
    }
}

/// Buffers used by `search_inner_ctx` (everything except `normalized`).
pub(crate) struct SearchBuffers {
    pub(crate) all_results: Vec<(u64, f32)>,
    pub(crate) query_quantized: Vec<i8>,
    pub(crate) segment: SegmentSearchBuffers,
    pub(crate) live: LiveSearchBuffers,
    pub(crate) seen: HashSet<u64>,
}

impl SearchBuffers {
    fn new() -> Self {
        Self {
            all_results: Vec::new(),
            query_quantized: Vec::new(),
            segment: SegmentSearchBuffers::new(),
            live: LiveSearchBuffers::new(),
            seen: HashSet::new(),
        }
    }
}

/// Reusable search buffers to avoid per-query heap allocations.
///
/// Create once and pass into repeated `search_with_context()` /
/// `search_with_filter_and_context()` calls. All internal buffers are
/// cleared (preserving capacity) between uses, so after the first query
/// no further allocations occur on the search path.
pub struct SearchContext {
    pub(crate) normalized: Vec<f32>,
    pub(crate) inner: SearchBuffers,
}

impl SearchContext {
    pub fn new() -> Self {
        Self {
            normalized: Vec::new(),
            inner: SearchBuffers::new(),
        }
    }
}

impl Default for SearchContext {
    fn default() -> Self {
        Self::new()
    }
}
