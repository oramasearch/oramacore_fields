use serde_json::Value;

pub struct StringIndexer {
    is_array: bool,
}

impl StringIndexer {
    pub fn new(is_array: bool) -> Self {
        StringIndexer { is_array }
    }

    pub fn index_json(&self, value: &Value) -> Option<IndexedValue> {
        if self.is_array {
            self.index_json_array(value)
        } else {
            self.index_json_plain(value)
        }
    }

    fn index_json_plain(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::String(s) => Some(IndexedValue::Plain(s.clone())),
            _ => None,
        }
    }

    fn index_json_array(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Array(arr) => {
                let strings: Vec<_> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                Some(IndexedValue::Array(strings))
            }
            _ => None,
        }
    }
}

pub enum IndexedValue {
    Plain(String),
    Array(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_json_plain() {
        let indexer = StringIndexer::new(false);
        let value = json!("hello");
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(s) if s == "hello"));
    }

    #[test]
    fn test_index_json_array() {
        let indexer = StringIndexer::new(true);
        let value = json!(["apple", "banana", "cherry"]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(
            matches!(indexed, IndexedValue::Array(arr) if arr == vec!["apple", "banana", "cherry"])
        );
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = StringIndexer::new(false);
        let value = json!(123);
        let indexed = indexer.index_json(&value);
        assert!(indexed.is_none());
    }

    #[test]
    fn test_index_json_array_mixed() {
        let indexer = StringIndexer::new(true);
        let value = json!(["hello", 123, "world", true]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec!["hello", "world"]));
    }
}
