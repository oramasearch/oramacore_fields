use super::key::IndexableNumber;
use serde_json::Value;

/// Extracts numeric values from JSON for indexing in a [`NumberStorage`](super::NumberStorage).
///
/// Supports two modes:
/// - **Plain** (`is_array = false`): Extracts a single number from a JSON value.
/// - **Array** (`is_array = true`): Extracts all numbers from a JSON array.
///
/// # Example
///
/// ```
/// use oramacore_fields::number::NumberIndexer;
/// use serde_json::json;
///
/// // Plain scalar field
/// let indexer = NumberIndexer::<u64>::new(false);
/// let value = indexer.index_json(&json!(42)); // Some(IndexedValue::Plain(42))
///
/// // Array field
/// let indexer = NumberIndexer::<u64>::new(true);
/// let value = indexer.index_json(&json!([10, 20, 30])); // Some(IndexedValue::Array([10, 20, 30]))
/// ```
pub struct NumberIndexer<T: IndexableNumber> {
    is_array: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: IndexableNumber> NumberIndexer<T> {
    /// Create a new indexer.
    ///
    /// - `is_array = false`: expects a JSON number, returns `IndexedValue::Plain`.
    /// - `is_array = true`: expects a JSON array of numbers, returns `IndexedValue::Array`.
    pub fn new(is_array: bool) -> Self {
        NumberIndexer {
            is_array,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Extract numeric values from a JSON value.
    ///
    /// Returns `None` if the value doesn't match the expected shape
    /// (e.g., a string when expecting a number, or a scalar when expecting an array).
    pub fn index_json(&self, value: &Value) -> Option<IndexedValue<T>> {
        if self.is_array {
            self.index_json_array(value)
        } else {
            self.index_json_plain(value)
        }
    }

    fn index_json_plain(&self, value: &Value) -> Option<IndexedValue<T>> {
        T::from_json_number(value).map(IndexedValue::Plain)
    }

    fn index_json_array(&self, value: &Value) -> Option<IndexedValue<T>> {
        match value {
            Value::Array(arr) => {
                let numbers: Vec<_> = arr.iter().filter_map(T::from_json_number).collect();
                Some(IndexedValue::Array(numbers))
            }
            _ => None,
        }
    }
}

/// A value extracted from JSON by [`NumberIndexer`], ready for indexing.
///
/// - `Plain(T)`: A single value. Passed to [`NumberStorage::insert`](super::NumberStorage::insert),
///   it creates one index entry.
/// - `Array(Vec<T>)`: Multiple values. Creates one entry per element, so the document
///   is found when querying for any of them.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexedValue<T: IndexableNumber> {
    Plain(T),
    Array(Vec<T>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_json_plain_u64() {
        let indexer = NumberIndexer::<u64>::new(false);
        let value = json!(42);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(42)));

        let value = json!(0);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Plain(0)));
    }

    #[test]
    fn test_index_json_plain_f64() {
        let indexer = NumberIndexer::<f64>::new(false);
        let value = json!(std::f64::consts::PI);
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Plain(v) => assert!((v - std::f64::consts::PI).abs() < f64::EPSILON),
            _ => panic!("Expected Plain"),
        }
    }

    #[test]
    fn test_index_json_array_u64() {
        let indexer = NumberIndexer::<u64>::new(true);
        let value = json!([10, 20, 30]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec![10, 20, 30]));

        let value = json!([5]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec![5]));
    }

    #[test]
    fn test_index_json_array_f64() {
        let indexer = NumberIndexer::<f64>::new(true);
        let value = json!([1.5, -0.5, std::f64::consts::PI]);
        let indexed = indexer.index_json(&value).unwrap();
        match indexed {
            IndexedValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!((arr[0] - 1.5).abs() < f64::EPSILON);
                assert!((arr[1] - (-0.5)).abs() < f64::EPSILON);
                assert!((arr[2] - std::f64::consts::PI).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = NumberIndexer::<u64>::new(false);
        let value = json!("not a number");
        assert!(indexer.index_json(&value).is_none());

        let value = json!(true);
        assert!(indexer.index_json(&value).is_none());
    }

    #[test]
    fn test_index_json_array_filters_non_numbers() {
        let indexer = NumberIndexer::<u64>::new(true);
        let value = json!([10, "skip", 20, null, 30]);
        let indexed = indexer.index_json(&value).unwrap();
        assert!(matches!(indexed, IndexedValue::Array(arr) if arr == vec![10, 20, 30]));
    }

    #[test]
    fn test_index_json_array_not_array() {
        let indexer = NumberIndexer::<u64>::new(true);
        let value = json!(42);
        assert!(indexer.index_json(&value).is_none());
    }
}
