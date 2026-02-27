use serde_json::Value;

pub struct StringIndexer<F: Fn(&str) -> bool> {
    is_array: bool,
    filter: F,
}

impl<F: Fn(&str) -> bool> StringIndexer<F> {
    pub fn new(is_array: bool, filter: F) -> Self {
        StringIndexer { is_array, filter }
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
            Value::String(s) if (self.filter)(s) => Some(IndexedValue::Plain(s.clone())),
            _ => None,
        }
    }

    fn index_json_array(&self, value: &Value) -> Option<IndexedValue> {
        match value {
            Value::Array(arr) => {
                let strings: Vec<_> = arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .filter(|s| (self.filter)(s))
                    .map(|s| s.to_string())
                    .collect();
                Some(IndexedValue::Array(strings))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        let indexer = StringIndexer::new(false, |_| true);
        let value = json!("hello");
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(s) if s == "hello"));
    }

    #[test]
    fn test_index_json_array() {
        let indexer = StringIndexer::new(true, |_| true);
        let value = json!(["apple", "banana", "cherry"]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(
            matches!(indexed, IndexedValue::Array(arr) if arr == vec!["apple", "banana", "cherry"])
        );
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = StringIndexer::new(false, |_| true);
        let value = json!(123);
        let indexed = indexer.index_json(&value);
        assert!(indexed.is_none());
    }

    #[test]
    fn test_index_json_array_mixed() {
        let indexer = StringIndexer::new(true, |_| true);
        let value = json!(["hello", 123, "world", true]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec!["hello", "world"]));
    }

    #[test]
    fn test_filter_rejects_plain() {
        let indexer = StringIndexer::new(false, |s| s != "secret");
        assert!(indexer.index_json(&json!("hello")).is_some());
        assert!(indexer.index_json(&json!("secret")).is_none());
    }

    #[test]
    fn test_filter_rejects_array_elements() {
        let indexer = StringIndexer::new(true, |s| !s.is_empty());
        let value = json!(["apple", "", "cherry", ""]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec!["apple", "cherry"]));
    }
}
