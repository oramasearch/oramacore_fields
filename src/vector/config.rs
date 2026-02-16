use super::error::Error;

#[cfg(feature = "cli")]
use serde::Serialize;

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
pub struct VectorConfig {
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
}

impl VectorConfig {
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
        let config = VectorConfig::new(384, DistanceMetric::Cosine).unwrap();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(VectorConfig::new(0, DistanceMetric::L2).is_err());
        assert!(VectorConfig::new(4097, DistanceMetric::L2).is_err());
    }

    #[test]
    fn test_boundary_dimensions() {
        assert!(VectorConfig::new(1, DistanceMetric::L2).is_ok());
        assert!(VectorConfig::new(4096, DistanceMetric::L2).is_ok());
    }

    #[test]
    fn test_node_block_size() {
        let config = VectorConfig::new(128, DistanceMetric::L2).unwrap();
        // m0=32, max_level=16, m=16
        // 32*4 + 16*16*4 = 128 + 1024 = 1152
        assert_eq!(config.node_block_size(), 1152);
    }
}
