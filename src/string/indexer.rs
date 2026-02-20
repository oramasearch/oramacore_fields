use std::collections::HashMap;

/// Per-term positional data for a document.
#[derive(Debug, Clone)]
pub struct TermData {
    /// Token positions where the exact (unstemmed) form appears.
    pub(crate) exact_positions: Vec<u32>,
    /// Token positions where the stemmed form appears.
    pub(crate) stemmed_positions: Vec<u32>,
}

impl TermData {
    pub fn new(exact_positions: Vec<u32>, stemmed_positions: Vec<u32>) -> Self {
        Self {
            exact_positions,
            stemmed_positions,
        }
    }
}

/// The value to insert into the string index for a single document.
#[derive(Debug, Clone)]
pub struct IndexedValue {
    /// Number of tokens in the document field.
    pub(crate) field_length: u16,
    /// Per-term positional data. Keys are the normalized term strings.
    pub(crate) terms: HashMap<String, TermData>,
}

impl IndexedValue {
    pub fn new(field_length: u16, terms: HashMap<String, TermData>) -> Self {
        Self {
            field_length,
            terms,
        }
    }
}

/// Tokenization strategy for converting raw text into tokens.
///
/// Implementations control how text is split into tokens and optionally stemmed.
pub trait Tokenizer {
    /// Tokenize and optionally stem the input string.
    /// Returns `(token, Option<stemmed>)` pairs in order.
    /// If stemmed is `None`, only exact positions are recorded for that token.
    fn tokenize_and_stem(&self, input: &str) -> Vec<(String, Option<String>)>;
}

/// Converts raw strings or JSON values into [`IndexedValue`] using a pluggable [`Tokenizer`].
pub struct StringIndexer<T: Tokenizer> {
    is_array: bool,
    tokenizer: T,
}

impl<T: Tokenizer> StringIndexer<T> {
    pub fn new(is_array: bool, tokenizer: T) -> Self {
        Self {
            is_array,
            tokenizer,
        }
    }

    /// Extract string(s) from a JSON value and build an [`IndexedValue`].
    ///
    /// - If `is_array` is true, expects a JSON array of strings. All elements'
    ///   tokens are concatenated into one continuous position stream.
    /// - Otherwise, expects a JSON string.
    ///
    /// Returns `None` if the JSON value doesn't match the expected shape.
    pub fn index_json(&self, value: &serde_json::Value) -> Option<IndexedValue> {
        if self.is_array {
            let arr = value.as_array()?;
            let mut all_tokens: Vec<(String, Option<String>)> = Vec::new();
            for item in arr {
                let s = item.as_str()?;
                let tokens = self.tokenizer.tokenize_and_stem(s);
                all_tokens.extend(tokens);
            }
            Some(Self::build_indexed_value(&all_tokens, 0))
        } else {
            let s = value.as_str()?;
            Some(self.index_str(s))
        }
    }

    /// Tokenize a single string and build an [`IndexedValue`] with term positions.
    fn index_str(&self, value: &str) -> IndexedValue {
        let tokens = self.tokenizer.tokenize_and_stem(value);
        Self::build_indexed_value(&tokens, 0)
    }

    fn build_indexed_value(
        tokens: &[(String, Option<String>)],
        position_offset: u32,
    ) -> IndexedValue {
        let mut terms: HashMap<String, TermData> = HashMap::new();
        let token_count = tokens.len();

        for (i, (original, stemmed)) in tokens.iter().enumerate() {
            let position = position_offset + i as u32;

            terms
                .entry(original.clone())
                .or_insert_with(|| TermData {
                    exact_positions: Vec::new(),
                    stemmed_positions: Vec::new(),
                })
                .exact_positions
                .push(position);

            if let Some(stem) = stemmed {
                if stem != original {
                    terms
                        .entry(stem.clone())
                        .or_insert_with(|| TermData {
                            exact_positions: Vec::new(),
                            stemmed_positions: Vec::new(),
                        })
                        .stemmed_positions
                        .push(position);
                } else {
                    // Stemmed form is the same as original — record as stemmed too
                    terms
                        .get_mut(original)
                        .unwrap()
                        .stemmed_positions
                        .push(position);
                }
            }
        }

        let field_length = std::cmp::min(token_count, (u16::MAX - 1) as usize) as u16;

        IndexedValue {
            field_length,
            terms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_value_construction() {
        let mut terms = HashMap::new();
        terms.insert(
            "hello".to_string(),
            TermData {
                exact_positions: vec![0],
                stemmed_positions: vec![],
            },
        );
        terms.insert(
            "world".to_string(),
            TermData {
                exact_positions: vec![1],
                stemmed_positions: vec![],
            },
        );
        let value = IndexedValue {
            field_length: 2,
            terms,
        };
        assert_eq!(value.field_length, 2);
        assert_eq!(value.terms.len(), 2);
    }

    #[test]
    fn test_term_data_with_positions() {
        let td = TermData {
            exact_positions: vec![0, 5, 10],
            stemmed_positions: vec![2, 7],
        };
        assert_eq!(td.exact_positions.len(), 3);
        assert_eq!(td.stemmed_positions.len(), 2);
    }

    /// A simple whitespace tokenizer with no stemming, for tests.
    struct NoStemTokenizer;
    impl Tokenizer for NoStemTokenizer {
        fn tokenize_and_stem(&self, input: &str) -> Vec<(String, Option<String>)> {
            input
                .split_whitespace()
                .map(|t| (t.to_lowercase(), None))
                .collect()
        }
    }

    /// A tokenizer that lowercases and "stems" by trimming trailing 's'.
    struct FakeStemTokenizer;
    impl Tokenizer for FakeStemTokenizer {
        fn tokenize_and_stem(&self, input: &str) -> Vec<(String, Option<String>)> {
            input
                .split_whitespace()
                .map(|t| {
                    let lower = t.to_lowercase();
                    let stemmed = lower.trim_end_matches('s').to_string();
                    let stemmed = if stemmed == lower {
                        None
                    } else {
                        Some(stemmed)
                    };
                    (lower, stemmed)
                })
                .collect()
        }
    }

    #[test]
    fn test_index_str_single_token() {
        let indexer = StringIndexer::new(false, NoStemTokenizer);
        let iv = indexer.index_str("Hello");
        assert_eq!(iv.field_length, 1);
        assert_eq!(iv.terms.len(), 1);
        let td = iv.terms.get("hello").unwrap();
        assert_eq!(td.exact_positions, vec![0]);
        assert!(td.stemmed_positions.is_empty());
    }

    #[test]
    fn test_index_str_with_stemming() {
        let indexer = StringIndexer::new(false, FakeStemTokenizer);
        let iv = indexer.index_str("cats");
        assert_eq!(iv.field_length, 1);
        // "cats" exact at 0, "cat" stemmed at 0
        let cats = iv.terms.get("cats").unwrap();
        assert_eq!(cats.exact_positions, vec![0]);
        assert!(cats.stemmed_positions.is_empty());
        let cat = iv.terms.get("cat").unwrap();
        assert!(cat.exact_positions.is_empty());
        assert_eq!(cat.stemmed_positions, vec![0]);
    }

    #[test]
    fn test_index_str_multiple_tokens() {
        let indexer = StringIndexer::new(false, NoStemTokenizer);
        let iv = indexer.index_str("the quick brown fox");
        assert_eq!(iv.field_length, 4);
        assert_eq!(iv.terms.len(), 4);
        assert_eq!(iv.terms["the"].exact_positions, vec![0]);
        assert_eq!(iv.terms["quick"].exact_positions, vec![1]);
        assert_eq!(iv.terms["brown"].exact_positions, vec![2]);
        assert_eq!(iv.terms["fox"].exact_positions, vec![3]);
    }

    #[test]
    fn test_index_json_plain() {
        let indexer = StringIndexer::new(false, NoStemTokenizer);
        let json = serde_json::json!("hello world");
        let iv = indexer.index_json(&json).unwrap();
        assert_eq!(iv.field_length, 2);
        assert_eq!(iv.terms["hello"].exact_positions, vec![0]);
        assert_eq!(iv.terms["world"].exact_positions, vec![1]);
    }

    #[test]
    fn test_index_json_array() {
        let indexer = StringIndexer::new(true, NoStemTokenizer);
        let json = serde_json::json!(["hello world", "foo bar"]);
        let iv = indexer.index_json(&json).unwrap();
        assert_eq!(iv.field_length, 4);
        assert_eq!(iv.terms["hello"].exact_positions, vec![0]);
        assert_eq!(iv.terms["world"].exact_positions, vec![1]);
        assert_eq!(iv.terms["foo"].exact_positions, vec![2]);
        assert_eq!(iv.terms["bar"].exact_positions, vec![3]);
    }

    #[test]
    fn test_index_json_invalid() {
        let indexer = StringIndexer::new(false, NoStemTokenizer);
        assert!(indexer.index_json(&serde_json::json!(42)).is_none());
        assert!(indexer.index_json(&serde_json::json!(true)).is_none());
        assert!(indexer.index_json(&serde_json::json!(null)).is_none());
        assert!(indexer.index_json(&serde_json::json!([1, 2])).is_none());
    }

    #[test]
    fn test_index_str_empty() {
        let indexer = StringIndexer::new(false, NoStemTokenizer);
        let iv = indexer.index_str("");
        assert_eq!(iv.field_length, 0);
        assert!(iv.terms.is_empty());
    }
}
