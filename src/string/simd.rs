//! SIMD-optimized batch BM25F scoring functions.
//!
//! Provides platform-specific implementations:
//! - aarch64: NEON (compile-time, always available)
//! - x86_64: AVX2+FMA (runtime-detected)
//! - Fallback: scalar loops

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn x86_has_avx2_fma() -> bool {
    static DETECTED: OnceLock<bool> = OnceLock::new();
    *DETECTED.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

// ---------------------------------------------------------------------------
// Dispatch functions
// ---------------------------------------------------------------------------

/// Batch-compute BM25F normalized TF for `len` elements.
///
/// `out[i] = tfs[i] / (1.0 - b + b * field_lengths[i] * inv_avg_fl)`
#[inline]
#[allow(unreachable_code)]
pub fn batch_normalized_tf(
    tfs: &[f32],
    field_lengths: &[f32],
    inv_avg_fl: f32,
    b: f32,
    out: &mut [f32],
    len: usize,
) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return neon::batch_normalized_tf(tfs, field_lengths, inv_avg_fl, b, out, len);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if x86_has_avx2_fma() {
        return unsafe { avx2::batch_normalized_tf(tfs, field_lengths, inv_avg_fl, b, out, len) };
    }

    scalar::batch_normalized_tf(tfs, field_lengths, inv_avg_fl, b, out, len)
}

/// Batch-compute BM25F final score for `len` elements.
///
/// `out[i] = idf * (k + 1.0) * ntfs[i] / (k + ntfs[i])`
#[inline]
#[allow(unreachable_code)]
pub fn batch_bm25f_score(ntfs: &[f32], k: f32, idf: f32, out: &mut [f32], len: usize) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return neon::batch_bm25f_score(ntfs, k, idf, out, len);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if x86_has_avx2_fma() {
        return unsafe { avx2::batch_bm25f_score(ntfs, k, idf, out, len) };
    }

    scalar::batch_bm25f_score(ntfs, k, idf, out, len)
}

// ---------------------------------------------------------------------------
// Scalar — always available
// ---------------------------------------------------------------------------

pub mod scalar {
    /// Batch-compute normalized TF (scalar fallback).
    #[inline]
    pub fn batch_normalized_tf(
        tfs: &[f32],
        field_lengths: &[f32],
        inv_avg_fl: f32,
        b: f32,
        out: &mut [f32],
        len: usize,
    ) {
        let one_minus_b = 1.0 - b;
        for i in 0..len {
            let denom = one_minus_b + b * field_lengths[i] * inv_avg_fl;
            out[i] = tfs[i] / denom;
        }
    }

    /// Batch-compute BM25F final score (scalar fallback).
    #[inline]
    pub fn batch_bm25f_score(ntfs: &[f32], k: f32, idf: f32, out: &mut [f32], len: usize) {
        let idf_k1 = idf * (k + 1.0);
        for i in 0..len {
            out[i] = idf_k1 * ntfs[i] / (k + ntfs[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON — 4-wide f32, always available
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod neon {
    use std::arch::aarch64::*;

    #[inline]
    pub fn batch_normalized_tf(
        tfs: &[f32],
        field_lengths: &[f32],
        inv_avg_fl: f32,
        b: f32,
        out: &mut [f32],
        len: usize,
    ) {
        let one_minus_b = 1.0 - b;

        // Safety: NEON is always available on aarch64.
        // Pointer arithmetic is bounded by `len`.
        unsafe {
            let v_one_minus_b = vdupq_n_f32(one_minus_b);
            let v_b = vdupq_n_f32(b);
            let v_inv_avg = vdupq_n_f32(inv_avg_fl);

            let tfs_ptr = tfs.as_ptr();
            let fl_ptr = field_lengths.as_ptr();
            let out_ptr = out.as_mut_ptr();

            let chunks = len / 4;
            for i in 0..chunks {
                let off = i * 4;
                let v_tf = vld1q_f32(tfs_ptr.add(off));
                let v_fl = vld1q_f32(fl_ptr.add(off));
                // denom = one_minus_b + b * fl * inv_avg_fl
                let v_denom = vfmaq_f32(v_one_minus_b, v_b, vmulq_f32(v_fl, v_inv_avg));
                let v_result = vdivq_f32(v_tf, v_denom);
                vst1q_f32(out_ptr.add(off), v_result);
            }

            // Scalar tail (0-3 elements)
            for i in (chunks * 4)..len {
                let denom = one_minus_b + b * *fl_ptr.add(i) * inv_avg_fl;
                *out_ptr.add(i) = *tfs_ptr.add(i) / denom;
            }
        }
    }

    #[inline]
    pub fn batch_bm25f_score(ntfs: &[f32], k: f32, idf: f32, out: &mut [f32], len: usize) {
        let idf_k1 = idf * (k + 1.0);

        // Safety: NEON is always available on aarch64.
        unsafe {
            let v_idf_k1 = vdupq_n_f32(idf_k1);
            let v_k = vdupq_n_f32(k);

            let ntfs_ptr = ntfs.as_ptr();
            let out_ptr = out.as_mut_ptr();

            let chunks = len / 4;
            for i in 0..chunks {
                let off = i * 4;
                let v_ntf = vld1q_f32(ntfs_ptr.add(off));
                let v_denom = vaddq_f32(v_k, v_ntf);
                let v_result = vdivq_f32(vmulq_f32(v_idf_k1, v_ntf), v_denom);
                vst1q_f32(out_ptr.add(off), v_result);
            }

            // Scalar tail
            for i in (chunks * 4)..len {
                let ntf = *ntfs_ptr.add(i);
                *out_ptr.add(i) = idf_k1 * ntf / (k + ntf);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2+FMA — 8-wide f32, runtime-detected
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod avx2 {
    #[allow(unused_imports)]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn batch_normalized_tf(
        tfs: &[f32],
        field_lengths: &[f32],
        inv_avg_fl: f32,
        b: f32,
        out: &mut [f32],
        len: usize,
    ) {
        let one_minus_b = 1.0 - b;
        let v_one_minus_b = _mm256_set1_ps(one_minus_b);
        let v_b = _mm256_set1_ps(b);
        let v_inv_avg = _mm256_set1_ps(inv_avg_fl);

        let tfs_ptr = tfs.as_ptr();
        let fl_ptr = field_lengths.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let chunks = len / 8;
        for i in 0..chunks {
            let off = i * 8;
            let v_tf = _mm256_loadu_ps(tfs_ptr.add(off));
            let v_fl = _mm256_loadu_ps(fl_ptr.add(off));
            // denom = one_minus_b + b * fl * inv_avg_fl
            let v_fl_scaled = _mm256_mul_ps(v_fl, v_inv_avg);
            let v_denom = _mm256_fmadd_ps(v_b, v_fl_scaled, v_one_minus_b);
            let v_result = _mm256_div_ps(v_tf, v_denom);
            _mm256_storeu_ps(out_ptr.add(off), v_result);
        }

        // Scalar tail (0-7 elements)
        for i in (chunks * 8)..len {
            let denom = one_minus_b + b * *fl_ptr.add(i) * inv_avg_fl;
            *out_ptr.add(i) = *tfs_ptr.add(i) / denom;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn batch_bm25f_score(ntfs: &[f32], k: f32, idf: f32, out: &mut [f32], len: usize) {
        let idf_k1 = idf * (k + 1.0);
        let v_idf_k1 = _mm256_set1_ps(idf_k1);
        let v_k = _mm256_set1_ps(k);

        let ntfs_ptr = ntfs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let chunks = len / 8;
        for i in 0..chunks {
            let off = i * 8;
            let v_ntf = _mm256_loadu_ps(ntfs_ptr.add(off));
            let v_denom = _mm256_add_ps(v_k, v_ntf);
            let v_numer = _mm256_mul_ps(v_idf_k1, v_ntf);
            let v_result = _mm256_div_ps(v_numer, v_denom);
            _mm256_storeu_ps(out_ptr.add(off), v_result);
        }

        // Scalar tail
        for i in (chunks * 8)..len {
            let ntf = *ntfs_ptr.add(i);
            *out_ptr.add(i) = idf_k1 * ntf / (k + ntf);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_batch_normalized_tf() {
        let tfs = [1.0, 2.0, 3.0, 5.0, 0.5];
        let fls = [10.0, 5.0, 20.0, 8.0, 15.0];
        let inv_avg_fl = 1.0 / 12.0;
        let b = 0.75;
        let mut out = [0.0f32; 5];

        scalar::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out, 5);

        for i in 0..5 {
            let expected = tfs[i] / (1.0 - b + b * fls[i] * inv_avg_fl);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "Mismatch at {i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_scalar_batch_bm25f_score() {
        let ntfs = [0.5, 1.0, 2.0, 5.0, 10.0];
        let k = 1.2;
        let idf = 2.5;
        let mut out = [0.0f32; 5];

        scalar::batch_bm25f_score(&ntfs, k, idf, &mut out, 5);

        let idf_k1 = idf * (k + 1.0);
        for i in 0..5 {
            let expected = idf_k1 * ntfs[i] / (k + ntfs[i]);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "Mismatch at {i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_neon_batch_normalized_tf() {
        // Test with sizes that exercise both SIMD and scalar tail
        for len in [1, 3, 4, 5, 7, 8, 9, 16] {
            let tfs: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.5).collect();
            let fls: Vec<f32> = (0..len).map(|i| (i as f32 + 3.0) * 2.0).collect();
            let inv_avg_fl = 1.0 / 10.0;
            let b = 0.75;

            let mut out_neon = vec![0.0f32; len];
            let mut out_scalar = vec![0.0f32; len];

            neon::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_neon, len);
            scalar::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_scalar, len);

            for i in 0..len {
                assert!(
                    (out_neon[i] - out_scalar[i]).abs() < 1e-5,
                    "NEON mismatch at {i} (len={len}): neon={}, scalar={}",
                    out_neon[i],
                    out_scalar[i]
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_neon_batch_bm25f_score() {
        for len in [1, 3, 4, 5, 7, 8, 9, 16] {
            let ntfs: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let k = 1.2;
            let idf = 2.5;

            let mut out_neon = vec![0.0f32; len];
            let mut out_scalar = vec![0.0f32; len];

            neon::batch_bm25f_score(&ntfs, k, idf, &mut out_neon, len);
            scalar::batch_bm25f_score(&ntfs, k, idf, &mut out_scalar, len);

            for i in 0..len {
                assert!(
                    (out_neon[i] - out_scalar[i]).abs() < 1e-5,
                    "NEON mismatch at {i} (len={len}): neon={}, scalar={}",
                    out_neon[i],
                    out_scalar[i]
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_batch_normalized_tf() {
        if !x86_has_avx2_fma() {
            return;
        }

        for len in [1, 3, 7, 8, 9, 15, 16, 17] {
            let tfs: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.5).collect();
            let fls: Vec<f32> = (0..len).map(|i| (i as f32 + 3.0) * 2.0).collect();
            let inv_avg_fl = 1.0 / 10.0;
            let b = 0.75;

            let mut out_avx2 = vec![0.0f32; len];
            let mut out_scalar = vec![0.0f32; len];

            unsafe {
                avx2::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_avx2, len);
            }
            scalar::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_scalar, len);

            for i in 0..len {
                assert!(
                    (out_avx2[i] - out_scalar[i]).abs() < 1e-5,
                    "AVX2 mismatch at {i} (len={len}): avx2={}, scalar={}",
                    out_avx2[i],
                    out_scalar[i]
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_batch_bm25f_score() {
        if !x86_has_avx2_fma() {
            return;
        }

        for len in [1, 3, 7, 8, 9, 15, 16, 17] {
            let ntfs: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let k = 1.2;
            let idf = 2.5;

            let mut out_avx2 = vec![0.0f32; len];
            let mut out_scalar = vec![0.0f32; len];

            unsafe {
                avx2::batch_bm25f_score(&ntfs, k, idf, &mut out_avx2, len);
            }
            scalar::batch_bm25f_score(&ntfs, k, idf, &mut out_scalar, len);

            for i in 0..len {
                assert!(
                    (out_avx2[i] - out_scalar[i]).abs() < 1e-5,
                    "AVX2 mismatch at {i} (len={len}): avx2={}, scalar={}",
                    out_avx2[i],
                    out_scalar[i]
                );
            }
        }
    }

    #[test]
    fn test_dispatch_batch_normalized_tf() {
        let tfs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let fls = [10.0, 5.0, 20.0, 8.0, 15.0, 12.0, 3.0, 25.0, 7.0];
        let inv_avg_fl = 1.0 / 12.0;
        let b = 0.75;
        let mut out_dispatch = [0.0f32; 9];
        let mut out_scalar = [0.0f32; 9];

        batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_dispatch, 9);
        scalar::batch_normalized_tf(&tfs, &fls, inv_avg_fl, b, &mut out_scalar, 9);

        for i in 0..9 {
            assert!(
                (out_dispatch[i] - out_scalar[i]).abs() < 1e-5,
                "Dispatch mismatch at {i}: dispatch={}, scalar={}",
                out_dispatch[i],
                out_scalar[i]
            );
        }
    }

    #[test]
    fn test_dispatch_batch_bm25f_score() {
        let ntfs = [0.5, 1.0, 2.0, 5.0, 10.0, 0.1, 3.0, 7.0, 0.8];
        let k = 1.2;
        let idf = 2.5;
        let mut out_dispatch = [0.0f32; 9];
        let mut out_scalar = [0.0f32; 9];

        batch_bm25f_score(&ntfs, k, idf, &mut out_dispatch, 9);
        scalar::batch_bm25f_score(&ntfs, k, idf, &mut out_scalar, 9);

        for i in 0..9 {
            assert!(
                (out_dispatch[i] - out_scalar[i]).abs() < 1e-5,
                "Dispatch mismatch at {i}: dispatch={}, scalar={}",
                out_dispatch[i],
                out_scalar[i]
            );
        }
    }
}
