pub struct Embedding {
    pub(crate) vector: Vec<f32>,
}

pub enum IndexedValue {
    Single(Embedding),
    Array(Vec<Embedding>),
}

pub struct VectorIndexer {
    dimensions: usize,
}

impl VectorIndexer {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Index a JSON value, expecting an array of numbers.
    pub fn index_vec(&self, vector: &[f32]) -> Option<IndexedValue> {
        if vector.len() != self.dimensions {
            return None;
        }
        for &v in vector {
            if !v.is_finite() {
                return None;
            }
        }

        Some(IndexedValue::Single(Embedding { vector: vector.to_vec() }))
    }

    /// Index a JSON value, expecting an array of numbers.
    pub fn index_vec_vec(&self, vectors: &[Vec<f32>]) -> Option<IndexedValue> {
        for vector in vectors {
            if vector.len() != self.dimensions {
                return None;
            }
            for &v in vector {
                if !v.is_finite() {
                    return None;
                }
            }
        }

        Some(IndexedValue::Array(
            vectors.iter()
                .map(|v| Embedding { vector: v.clone() })
                .collect()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_vec_valid() {
        let indexer = VectorIndexer::new(3);
        let indexed = indexer.index_vec(&[1.0, 2.0, 3.0]);
        assert!(indexed.is_some());
        match indexed.unwrap() {
            IndexedValue::Single(e) => assert_eq!(e.vector, vec![1.0, 2.0, 3.0]),
            IndexedValue::Array(_) => panic!("expected Single"),
        }
    }

    #[test]
    fn test_index_vec_empty() {
        let indexer = VectorIndexer::new(3);
        assert!(indexer.index_vec(&[]).is_none());
    }

    #[test]
    fn test_index_vec_non_finite() {
        let indexer = VectorIndexer::new(2);
        assert!(indexer.index_vec(&[1.0, f32::NAN]).is_none());
        assert!(indexer.index_vec(&[f32::INFINITY, 1.0]).is_none());
    }

    #[test]
    fn test_index_vec_vec_valid() {
        let indexer = VectorIndexer::new(2);
        let indexed = indexer.index_vec_vec(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(indexed.is_some());
        match indexed.unwrap() {
            IndexedValue::Array(embeddings) => {
                assert_eq!(embeddings.len(), 2);
                assert_eq!(embeddings[0].vector, vec![1.0, 2.0]);
                assert_eq!(embeddings[1].vector, vec![3.0, 4.0]);
            }
            IndexedValue::Single(_) => panic!("expected Array"),
        }
    }

    #[test]
    fn test_index_vec_vec_empty_inner() {
        let indexer = VectorIndexer::new(2);
        assert!(indexer.index_vec_vec(&[vec![1.0, 2.0], vec![]]).is_none());
    }
}
