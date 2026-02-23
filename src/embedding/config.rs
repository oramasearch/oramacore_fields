use super::error::Error;

#[cfg(feature = "cli")]
use serde::Serialize;

/// Deletion ratio threshold (0.0–1.0).
/// When a segment's `num_deletes / num_nodes` exceeds this, the segment is fully rebuilt.
#[derive(Debug, Clone, Copy)]
pub struct DeletionThreshold(f64);

impl DeletionThreshold {
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for DeletionThreshold {
    fn default() -> Self {
        DeletionThreshold(0.1)
    }
}

impl TryFrom<f64> for DeletionThreshold {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if (0.0..=1.0).contains(&value) {
            Ok(DeletionThreshold(value))
        } else {
            Err("threshold must be between 0.0 and 1.0")
        }
    }
}

impl TryFrom<f32> for DeletionThreshold {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        DeletionThreshold::try_from(value as f64)
    }
}

/// Configuration for multi-segment compaction behaviour.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Maximum number of nodes per segment.
    /// When the last segment + live insertions would exceed this, a new segment is created.
    /// Must be <= u32::MAX. Default: 1_000_000.
    pub max_nodes_per_segment: u32,

    /// Deletion ratio threshold (0.0–1.0).
    /// When a segment's `num_deletes / num_nodes` exceeds this, it is fully rebuilt.
    /// Default: 0.1.
    pub deletion_threshold: DeletionThreshold,

    /// Insertion ratio threshold (0.0–1.0).
    /// When `insertions_since_rebuild / nodes_at_last_rebuild` exceeds this,
    /// the last segment is fully rebuilt instead of incrementally updated.
    /// Default: 0.3.
    pub insertion_rebuild_threshold: f64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_nodes_per_segment: 1_000_000,
            deletion_threshold: DeletionThreshold::default(),
            insertion_rebuild_threshold: 0.3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub enum DistanceMetric {
    L2,
    DotProduct,
    Cosine,
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::L2 => write!(f, "L2"),
            DistanceMetric::DotProduct => write!(f, "DotProduct"),
            DistanceMetric::Cosine => write!(f, "Cosine"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
}

impl EmbeddingConfig {
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Result<Self, Error> {
        if dimensions == 0 || dimensions > 4096 {
            return Err(Error::InvalidDimensions(dimensions));
        }
        Ok(Self {
            dimensions,
            metric,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            max_level: 16,
        })
    }

    pub fn node_block_size(&self) -> usize {
        self.m0 * 4 + self.max_level * self.m * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = EmbeddingConfig::new(384, DistanceMetric::Cosine).unwrap();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(EmbeddingConfig::new(0, DistanceMetric::L2).is_err());
        assert!(EmbeddingConfig::new(4097, DistanceMetric::L2).is_err());
    }

    #[test]
    fn test_boundary_dimensions() {
        assert!(EmbeddingConfig::new(1, DistanceMetric::L2).is_ok());
        assert!(EmbeddingConfig::new(4096, DistanceMetric::L2).is_ok());
    }

    #[test]
    fn test_node_block_size() {
        let config = EmbeddingConfig::new(128, DistanceMetric::L2).unwrap();
        // m0=32, max_level=16, m=16
        // 32*4 + 16*16*4 = 128 + 1024 = 1152
        assert_eq!(config.node_block_size(), 1152);
    }
}
