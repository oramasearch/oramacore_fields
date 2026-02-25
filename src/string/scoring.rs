/// Compute the BM25F normalized term frequency.
///
/// `tf` is the raw term occurrence count in the document field.
/// `field_length` is the number of tokens in the document field.
/// `avg_field_length` is the average field length across all documents.
/// `b` controls the degree of length normalization (0.0 = none, 1.0 = full).
#[inline]
pub fn bm25f_normalized_tf(tf: f32, field_length: f32, avg_field_length: f32, b: f32) -> f32 {
    if avg_field_length == 0.0 {
        return 0.0;
    }
    tf / (1.0 - b + b * (field_length / avg_field_length))
}

/// Compute the BM25F normalized term frequency using a precomputed inverse average field length.
///
/// Equivalent to `bm25f_normalized_tf` but eliminates one division per call by accepting
/// `inv_avg_fl = 1.0 / avg_field_length` (precomputed once per search).
#[inline]
pub fn bm25f_normalized_tf_fast(tf: f32, field_length: f32, inv_avg_fl: f32, b: f32) -> f32 {
    if inv_avg_fl == 0.0 {
        return 0.0;
    }
    tf / (1.0 - b + b * field_length * inv_avg_fl)
}

/// Compute the Lucene-style IDF (always non-negative).
///
/// `total_documents` is the total number of documents in the corpus.
/// `corpus_df` is the number of documents containing the term.
#[inline]
pub fn calculate_idf(total_documents: u64, corpus_df: u64) -> f32 {
    if corpus_df == 0 {
        return 0.0;
    }
    let n = total_documents as f64;
    let df = corpus_df as f64;
    ((1.0 + (n - df + 0.5) / (df + 0.5)).ln()) as f32
}

/// Compute the final BM25F score for a single term.
///
/// `aggregated_score` is the accumulated normalized TF across fields (for single-field, it's just the normalized TF).
/// `k` is the term frequency saturation parameter.
/// `idf` is the inverse document frequency for this term.
#[inline]
pub fn bm25f_score(aggregated_score: f32, k: f32, idf: f32) -> f32 {
    idf * (k + 1.0) * aggregated_score / (k + aggregated_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_tf_basic() {
        // tf=2, field_length=10, avg=10, b=0.75
        // expected: 2 / (1 - 0.75 + 0.75 * (10/10)) = 2 / 1.0 = 2.0
        let result = bm25f_normalized_tf(2.0, 10.0, 10.0, 0.75);
        assert!((result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_tf_short_document() {
        // Short doc (field_length < avg) should boost tf
        // tf=1, field_length=5, avg=10, b=0.75
        // expected: 1 / (1 - 0.75 + 0.75 * 0.5) = 1 / 0.625 = 1.6
        let result = bm25f_normalized_tf(1.0, 5.0, 10.0, 0.75);
        assert!((result - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_tf_long_document() {
        // Long doc (field_length > avg) should penalize tf
        // tf=1, field_length=20, avg=10, b=0.75
        // expected: 1 / (1 - 0.75 + 0.75 * 2.0) = 1 / 1.75 ≈ 0.5714
        let result = bm25f_normalized_tf(1.0, 20.0, 10.0, 0.75);
        assert!((result - 1.0 / 1.75).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_tf_no_normalization() {
        // b=0 means no length normalization
        // tf=3, field_length=100, avg=10, b=0
        // expected: 3 / (1 - 0 + 0 * 10) = 3 / 1 = 3.0
        let result = bm25f_normalized_tf(3.0, 100.0, 10.0, 0.0);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_tf_zero_avg() {
        let result = bm25f_normalized_tf(1.0, 10.0, 0.0, 0.75);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_idf_basic() {
        // 100 docs, 10 contain the term
        // idf = ln(1 + (100 - 10 + 0.5) / (10 + 0.5)) = ln(1 + 90.5/10.5)
        let result = calculate_idf(100, 10);
        let expected = (1.0 + 90.5 / 10.5_f64).ln() as f32;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_idf_all_docs_match() {
        // When all docs match, IDF should still be positive (Lucene-style)
        let result = calculate_idf(100, 100);
        assert!(result > 0.0);
    }

    #[test]
    fn test_idf_single_doc() {
        let result = calculate_idf(1, 1);
        assert!(result > 0.0);
    }

    #[test]
    fn test_idf_rare_term() {
        // Rare term: only 1 doc out of 1000
        let result = calculate_idf(1000, 1);
        // Should be high
        assert!(result > 5.0);
    }

    #[test]
    fn test_idf_zero_df() {
        let result = calculate_idf(100, 0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bm25f_score_basic() {
        // aggregated_score=2.0, k=1.2, idf=2.0
        // score = 2.0 * (1.2 + 1.0) * 2.0 / (1.2 + 2.0) = 2.0 * 2.2 * 2.0 / 3.2 = 2.75
        let result = bm25f_score(2.0, 1.2, 2.0);
        let expected = 2.0 * 2.2 * 2.0 / 3.2;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_bm25f_score_saturation() {
        // As aggregated_score increases, the score approaches idf * (k + 1)
        let score_low = bm25f_score(1.0, 1.2, 2.0);
        let score_high = bm25f_score(100.0, 1.2, 2.0);
        let max = 2.0 * (1.2 + 1.0);

        assert!(score_high > score_low);
        assert!((score_high - max).abs() < 0.1);
    }

    #[test]
    fn test_bm25f_score_zero_aggregated() {
        let result = bm25f_score(0.0, 1.2, 2.0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_tf_fast_matches_original() {
        let cases = [
            (2.0, 10.0, 10.0, 0.75),
            (1.0, 5.0, 10.0, 0.75),
            (1.0, 20.0, 10.0, 0.75),
            (3.0, 100.0, 10.0, 0.0),
            (5.0, 8.0, 12.0, 0.5),
            (1.0, 1.0, 1.0, 1.0),
        ];
        for (tf, fl, avg_fl, b) in cases {
            let original = bm25f_normalized_tf(tf, fl, avg_fl, b);
            let inv_avg_fl = if avg_fl > 0.0 { 1.0 / avg_fl } else { 0.0 };
            let fast = bm25f_normalized_tf_fast(tf, fl, inv_avg_fl, b);
            assert!(
                (original - fast).abs() < 1e-6,
                "Mismatch for tf={tf}, fl={fl}, avg_fl={avg_fl}, b={b}: original={original}, fast={fast}"
            );
        }
    }

    #[test]
    fn test_normalized_tf_fast_zero_inv_avg() {
        let result = bm25f_normalized_tf_fast(1.0, 10.0, 0.0, 0.75);
        assert!((result - 0.0).abs() < 1e-6);
    }
}
