use serde_json::Value;

pub struct BoolIndexer {
    is_array: bool,
}

impl BoolIndexer {
    pub fn new(is_array: bool) -> Self {
        BoolIndexer { is_array }
    }

    /// Index a JSON object based on the specified key, returning an IndexedValue.
    pub fn index_json(&self, value: &Value) -> Option<IndexedValue> {
        if self.is_array {
            self.index_json_array(value)
        } else {
            self.index_json_plain(value)
        }
    }

    fn index_json_plain(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Bool(b) => Some(IndexedValue::Plain(*b)),
            _ => None,
        }
    }

    fn index_json_array(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Array(arr) => {
                let bools: Vec<_> = arr.iter().filter_map(|v| v.as_bool()).collect();
                Some(IndexedValue::Array(bools))
            }
            _ => None,
        }
    }
}

pub enum IndexedValue {
    Plain(bool),
    Array(Vec<bool>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_json_plain() {
        let indexer = BoolIndexer::new(false);
        let value = json!(true);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(true)));

        let value = json!(false);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(false)));
    }

    #[test]
    fn test_index_json_array() {
        let indexer = BoolIndexer::new(true);
        let value = json!([true, false, true]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec![true, false, true]));

        let value = json!([false, false]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec![false, false]));
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = BoolIndexer::new(false);
        let value = json!(123);
        let indexed = indexer.index_json(&value);
        assert!(indexed.is_none());
    }
}
