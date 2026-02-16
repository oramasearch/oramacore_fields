use serde_json::Value;

pub enum IndexedValue {
    Single(Vec<f32>),
}

pub struct VectorIndexer {
    dimensions: usize,
}

impl VectorIndexer {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Index a JSON value, expecting an array of numbers.
    pub fn index_json(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Array(arr) => {
                if arr.len() != self.dimensions {
                    return None;
                }
                let floats: Option<Vec<f32>> = arr
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                floats.map(IndexedValue::Single)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_json_valid() {
        let indexer = VectorIndexer::new(3);
        let value = json!([1.0, 2.0, 3.0]);
        let indexed = indexer.index_json(&value);
        assert!(indexed.is_some());
        match indexed.unwrap() {
            IndexedValue::Single(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
        }
    }

    #[test]
    fn test_index_json_wrong_dimensions() {
        let indexer = VectorIndexer::new(3);
        let value = json!([1.0, 2.0]);
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_index_json_not_array() {
        let indexer = VectorIndexer::new(3);
        let value = json!("hello");
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_index_json_non_numeric() {
        let indexer = VectorIndexer::new(2);
        let value = json!([1.0, "hello"]);
        assert!(indexer.index_json(&value).is_none());
    }
}
