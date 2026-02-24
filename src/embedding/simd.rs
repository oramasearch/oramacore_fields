//! SIMD-optimized distance function implementations.
//!
//! Provides platform-specific implementations:
//! - aarch64: NEON (compile-time, always available)
//! - x86_64: AVX2+FMA (runtime-detected)
//! - Fallback: scalar loops

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::sync::OnceLock;

/// Returns true if the CPU supports both AVX2 and FMA instructions.
/// Result is cached after the first call (one atomic load on subsequent calls).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn x86_has_avx2_fma() -> bool {
    static DETECTED: OnceLock<bool> = OnceLock::new();
    *DETECTED.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

// ---------------------------------------------------------------------------
// f32 distance functions
// ---------------------------------------------------------------------------

pub mod f32_ops {
    pub mod scalar {
        #[inline]
        pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let mut sum = 0.0f32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    let d = ca[i] - cb[i];
                    sum += d * d;
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                let d = x - y;
                sum += d * d;
            }
            sum
        }

        #[inline]
        pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let mut sum = 0.0f32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    sum += ca[i] * cb[i];
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                sum += x * y;
            }
            -sum
        }

        #[inline]
        pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    dot += ca[i] * cb[i];
                    norm_a += ca[i] * ca[i];
                    norm_b += cb[i] * cb[i];
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                dot += x * y;
                norm_a += x * x;
                norm_b += y * y;
            }
            let denom = (norm_a * norm_b).sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }

        #[inline]
        pub fn cosine_distance_prenorm(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let mut dot = 0.0f32;
            let mut norm_b = 0.0f32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    dot += ca[i] * cb[i];
                    norm_b += cb[i] * cb[i];
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                dot += x * y;
                norm_b += y * y;
            }
            let denom = norm_b.sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }

    // -----------------------------------------------------------------------
    // aarch64 NEON — 4-wide f32, always available
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    pub mod neon {
        use std::arch::aarch64::*;

        #[inline]
        pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            // Safety: NEON is always available on aarch64.
            // Pointer arithmetic is bounded by slice length.
            unsafe {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let d0 = vsubq_f32(vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
                    acc0 = vfmaq_f32(acc0, d0, d0);
                    let d1 =
                        vsubq_f32(vld1q_f32(a_ptr.add(off + 4)), vld1q_f32(b_ptr.add(off + 4)));
                    acc1 = vfmaq_f32(acc1, d1, d1);
                    let d2 =
                        vsubq_f32(vld1q_f32(a_ptr.add(off + 8)), vld1q_f32(b_ptr.add(off + 8)));
                    acc2 = vfmaq_f32(acc2, d2, d2);
                    let d3 = vsubq_f32(
                        vld1q_f32(a_ptr.add(off + 12)),
                        vld1q_f32(b_ptr.add(off + 12)),
                    );
                    acc3 = vfmaq_f32(acc3, d3, d3);
                }

                // Remaining 4-element chunks
                let mut off = main_iters * 16;
                let rem4 = (len - off) / 4;
                for _ in 0..rem4 {
                    let d = vsubq_f32(vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
                    acc0 = vfmaq_f32(acc0, d, d);
                    off += 4;
                }

                acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                let mut sum = vaddvq_f32(acc0);

                // Scalar tail (0-3 elements)
                for i in off..len {
                    let d = *a_ptr.add(i) - *b_ptr.add(i);
                    sum += d * d;
                }
                sum
            }
        }

        #[inline]
        pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    acc0 = vfmaq_f32(acc0, vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
                    acc1 = vfmaq_f32(
                        acc1,
                        vld1q_f32(a_ptr.add(off + 4)),
                        vld1q_f32(b_ptr.add(off + 4)),
                    );
                    acc2 = vfmaq_f32(
                        acc2,
                        vld1q_f32(a_ptr.add(off + 8)),
                        vld1q_f32(b_ptr.add(off + 8)),
                    );
                    acc3 = vfmaq_f32(
                        acc3,
                        vld1q_f32(a_ptr.add(off + 12)),
                        vld1q_f32(b_ptr.add(off + 12)),
                    );
                }

                let mut off = main_iters * 16;
                let rem4 = (len - off) / 4;
                for _ in 0..rem4 {
                    acc0 = vfmaq_f32(acc0, vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
                    off += 4;
                }

                acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                let mut sum = vaddvq_f32(acc0);

                for i in off..len {
                    sum += *a_ptr.add(i) * *b_ptr.add(i);
                }
                -sum
            }
        }

        #[inline]
        pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let zero = vdupq_n_f32(0.0);
                let (mut d0, mut d1, mut d2, mut d3) = (zero, zero, zero, zero);
                let (mut na0, mut na1, mut na2, mut na3) = (zero, zero, zero, zero);
                let (mut nb0, mut nb1, mut nb2, mut nb3) = (zero, zero, zero, zero);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let a0 = vld1q_f32(a_ptr.add(off));
                    let b0 = vld1q_f32(b_ptr.add(off));
                    d0 = vfmaq_f32(d0, a0, b0);
                    na0 = vfmaq_f32(na0, a0, a0);
                    nb0 = vfmaq_f32(nb0, b0, b0);

                    let a1 = vld1q_f32(a_ptr.add(off + 4));
                    let b1 = vld1q_f32(b_ptr.add(off + 4));
                    d1 = vfmaq_f32(d1, a1, b1);
                    na1 = vfmaq_f32(na1, a1, a1);
                    nb1 = vfmaq_f32(nb1, b1, b1);

                    let a2 = vld1q_f32(a_ptr.add(off + 8));
                    let b2 = vld1q_f32(b_ptr.add(off + 8));
                    d2 = vfmaq_f32(d2, a2, b2);
                    na2 = vfmaq_f32(na2, a2, a2);
                    nb2 = vfmaq_f32(nb2, b2, b2);

                    let a3 = vld1q_f32(a_ptr.add(off + 12));
                    let b3 = vld1q_f32(b_ptr.add(off + 12));
                    d3 = vfmaq_f32(d3, a3, b3);
                    na3 = vfmaq_f32(na3, a3, a3);
                    nb3 = vfmaq_f32(nb3, b3, b3);
                }

                let mut off = main_iters * 16;
                let rem4 = (len - off) / 4;
                for _ in 0..rem4 {
                    let av = vld1q_f32(a_ptr.add(off));
                    let bv = vld1q_f32(b_ptr.add(off));
                    d0 = vfmaq_f32(d0, av, bv);
                    na0 = vfmaq_f32(na0, av, av);
                    nb0 = vfmaq_f32(nb0, bv, bv);
                    off += 4;
                }

                d0 = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
                na0 = vaddq_f32(vaddq_f32(na0, na1), vaddq_f32(na2, na3));
                nb0 = vaddq_f32(vaddq_f32(nb0, nb1), vaddq_f32(nb2, nb3));
                let mut dot = vaddvq_f32(d0);
                let mut norm_a = vaddvq_f32(na0);
                let mut norm_b = vaddvq_f32(nb0);

                for i in off..len {
                    let x = *a_ptr.add(i);
                    let y = *b_ptr.add(i);
                    dot += x * y;
                    norm_a += x * x;
                    norm_b += y * y;
                }

                let denom = (norm_a * norm_b).sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            }
        }

        #[inline]
        pub fn cosine_distance_prenorm(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let zero = vdupq_n_f32(0.0);
                let (mut d0, mut d1, mut d2, mut d3) = (zero, zero, zero, zero);
                let (mut nb0, mut nb1, mut nb2, mut nb3) = (zero, zero, zero, zero);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let a0 = vld1q_f32(a_ptr.add(off));
                    let b0 = vld1q_f32(b_ptr.add(off));
                    d0 = vfmaq_f32(d0, a0, b0);
                    nb0 = vfmaq_f32(nb0, b0, b0);

                    let a1 = vld1q_f32(a_ptr.add(off + 4));
                    let b1 = vld1q_f32(b_ptr.add(off + 4));
                    d1 = vfmaq_f32(d1, a1, b1);
                    nb1 = vfmaq_f32(nb1, b1, b1);

                    let a2 = vld1q_f32(a_ptr.add(off + 8));
                    let b2 = vld1q_f32(b_ptr.add(off + 8));
                    d2 = vfmaq_f32(d2, a2, b2);
                    nb2 = vfmaq_f32(nb2, b2, b2);

                    let a3 = vld1q_f32(a_ptr.add(off + 12));
                    let b3 = vld1q_f32(b_ptr.add(off + 12));
                    d3 = vfmaq_f32(d3, a3, b3);
                    nb3 = vfmaq_f32(nb3, b3, b3);
                }

                let mut off = main_iters * 16;
                let rem4 = (len - off) / 4;
                for _ in 0..rem4 {
                    let av = vld1q_f32(a_ptr.add(off));
                    let bv = vld1q_f32(b_ptr.add(off));
                    d0 = vfmaq_f32(d0, av, bv);
                    nb0 = vfmaq_f32(nb0, bv, bv);
                    off += 4;
                }

                d0 = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
                nb0 = vaddq_f32(vaddq_f32(nb0, nb1), vaddq_f32(nb2, nb3));
                let mut dot = vaddvq_f32(d0);
                let mut norm_b = vaddvq_f32(nb0);

                for i in off..len {
                    let x = *a_ptr.add(i);
                    let y = *b_ptr.add(i);
                    dot += x * y;
                    norm_b += y * y;
                }

                let denom = norm_b.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // x86_64 AVX2+FMA — 8-wide f32
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub mod avx2 {
        #[allow(unused_imports)]
        use std::arch::x86_64::*;

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn hsum_f32x8(v: __m256) -> f32 {
            let v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let v64 = _mm_hadd_ps(v128, v128);
            let v32 = _mm_hadd_ps(v64, v64);
            _mm_cvtss_f32(v32)
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;
                let d0 = _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(off)),
                    _mm256_loadu_ps(b_ptr.add(off)),
                );
                acc0 = _mm256_fmadd_ps(d0, d0, acc0);
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 8)),
                    _mm256_loadu_ps(b_ptr.add(off + 8)),
                );
                acc1 = _mm256_fmadd_ps(d1, d1, acc1);
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 16)),
                    _mm256_loadu_ps(b_ptr.add(off + 16)),
                );
                acc2 = _mm256_fmadd_ps(d2, d2, acc2);
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 24)),
                    _mm256_loadu_ps(b_ptr.add(off + 24)),
                );
                acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            }

            let mut off = main_iters * 32;
            let rem8 = (len - off) / 8;
            for _ in 0..rem8 {
                let d = _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(off)),
                    _mm256_loadu_ps(b_ptr.add(off)),
                );
                acc0 = _mm256_fmadd_ps(d, d, acc0);
                off += 8;
            }

            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut sum = hsum_f32x8(acc0);

            for i in off..len {
                let d = *a_ptr.add(i) - *b_ptr.add(i);
                sum += d * d;
            }
            sum
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(off)),
                    _mm256_loadu_ps(b_ptr.add(off)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 8)),
                    _mm256_loadu_ps(b_ptr.add(off + 8)),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 16)),
                    _mm256_loadu_ps(b_ptr.add(off + 16)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(off + 24)),
                    _mm256_loadu_ps(b_ptr.add(off + 24)),
                    acc3,
                );
            }

            let mut off = main_iters * 32;
            let rem8 = (len - off) / 8;
            for _ in 0..rem8 {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_ptr.add(off)),
                    _mm256_loadu_ps(b_ptr.add(off)),
                    acc0,
                );
                off += 8;
            }

            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut sum = hsum_f32x8(acc0);

            for i in off..len {
                sum += *a_ptr.add(i) * *b_ptr.add(i);
            }
            -sum
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let z = _mm256_setzero_ps();
            let (mut d0, mut d1, mut d2, mut d3) = (z, z, z, z);
            let (mut na0, mut na1, mut na2, mut na3) = (z, z, z, z);
            let (mut nb0, mut nb1, mut nb2, mut nb3) = (z, z, z, z);

            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;
                let a0 = _mm256_loadu_ps(a_ptr.add(off));
                let b0 = _mm256_loadu_ps(b_ptr.add(off));
                d0 = _mm256_fmadd_ps(a0, b0, d0);
                na0 = _mm256_fmadd_ps(a0, a0, na0);
                nb0 = _mm256_fmadd_ps(b0, b0, nb0);

                let a1 = _mm256_loadu_ps(a_ptr.add(off + 8));
                let b1 = _mm256_loadu_ps(b_ptr.add(off + 8));
                d1 = _mm256_fmadd_ps(a1, b1, d1);
                na1 = _mm256_fmadd_ps(a1, a1, na1);
                nb1 = _mm256_fmadd_ps(b1, b1, nb1);

                let a2 = _mm256_loadu_ps(a_ptr.add(off + 16));
                let b2 = _mm256_loadu_ps(b_ptr.add(off + 16));
                d2 = _mm256_fmadd_ps(a2, b2, d2);
                na2 = _mm256_fmadd_ps(a2, a2, na2);
                nb2 = _mm256_fmadd_ps(b2, b2, nb2);

                let a3 = _mm256_loadu_ps(a_ptr.add(off + 24));
                let b3 = _mm256_loadu_ps(b_ptr.add(off + 24));
                d3 = _mm256_fmadd_ps(a3, b3, d3);
                na3 = _mm256_fmadd_ps(a3, a3, na3);
                nb3 = _mm256_fmadd_ps(b3, b3, nb3);
            }

            let mut off = main_iters * 32;
            let rem8 = (len - off) / 8;
            for _ in 0..rem8 {
                let av = _mm256_loadu_ps(a_ptr.add(off));
                let bv = _mm256_loadu_ps(b_ptr.add(off));
                d0 = _mm256_fmadd_ps(av, bv, d0);
                na0 = _mm256_fmadd_ps(av, av, na0);
                nb0 = _mm256_fmadd_ps(bv, bv, nb0);
                off += 8;
            }

            d0 = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
            na0 = _mm256_add_ps(_mm256_add_ps(na0, na1), _mm256_add_ps(na2, na3));
            nb0 = _mm256_add_ps(_mm256_add_ps(nb0, nb1), _mm256_add_ps(nb2, nb3));
            let mut dot = hsum_f32x8(d0);
            let mut norm_a = hsum_f32x8(na0);
            let mut norm_b = hsum_f32x8(nb0);

            for i in off..len {
                let x = *a_ptr.add(i);
                let y = *b_ptr.add(i);
                dot += x * y;
                norm_a += x * x;
                norm_b += y * y;
            }

            let denom = (norm_a * norm_b).sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn cosine_distance_prenorm(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let z = _mm256_setzero_ps();
            let (mut d0, mut d1, mut d2, mut d3) = (z, z, z, z);
            let (mut nb0, mut nb1, mut nb2, mut nb3) = (z, z, z, z);

            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;
                let a0 = _mm256_loadu_ps(a_ptr.add(off));
                let b0 = _mm256_loadu_ps(b_ptr.add(off));
                d0 = _mm256_fmadd_ps(a0, b0, d0);
                nb0 = _mm256_fmadd_ps(b0, b0, nb0);

                let a1 = _mm256_loadu_ps(a_ptr.add(off + 8));
                let b1 = _mm256_loadu_ps(b_ptr.add(off + 8));
                d1 = _mm256_fmadd_ps(a1, b1, d1);
                nb1 = _mm256_fmadd_ps(b1, b1, nb1);

                let a2 = _mm256_loadu_ps(a_ptr.add(off + 16));
                let b2 = _mm256_loadu_ps(b_ptr.add(off + 16));
                d2 = _mm256_fmadd_ps(a2, b2, d2);
                nb2 = _mm256_fmadd_ps(b2, b2, nb2);

                let a3 = _mm256_loadu_ps(a_ptr.add(off + 24));
                let b3 = _mm256_loadu_ps(b_ptr.add(off + 24));
                d3 = _mm256_fmadd_ps(a3, b3, d3);
                nb3 = _mm256_fmadd_ps(b3, b3, nb3);
            }

            let mut off = main_iters * 32;
            let rem8 = (len - off) / 8;
            for _ in 0..rem8 {
                let av = _mm256_loadu_ps(a_ptr.add(off));
                let bv = _mm256_loadu_ps(b_ptr.add(off));
                d0 = _mm256_fmadd_ps(av, bv, d0);
                nb0 = _mm256_fmadd_ps(bv, bv, nb0);
                off += 8;
            }

            d0 = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
            nb0 = _mm256_add_ps(_mm256_add_ps(nb0, nb1), _mm256_add_ps(nb2, nb3));
            let mut dot = hsum_f32x8(d0);
            let mut norm_b = hsum_f32x8(nb0);

            for i in off..len {
                let x = *a_ptr.add(i);
                let y = *b_ptr.add(i);
                dot += x * y;
                norm_b += y * y;
            }

            let denom = norm_b.sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }
}

// ---------------------------------------------------------------------------
// i8 distance functions
// ---------------------------------------------------------------------------

pub mod i8_ops {
    pub mod scalar {
        #[inline]
        pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let mut sum = 0i32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    let d = ca[i] as i32 - cb[i] as i32;
                    sum += d * d;
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                let d = x as i32 - y as i32;
                sum += d * d;
            }
            sum
        }

        #[inline]
        pub fn dot_product_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let mut sum = 0i32;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    sum += ca[i] as i32 * cb[i] as i32;
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                sum += x as i32 * y as i32;
            }
            -sum
        }

        #[inline]
        pub fn cosine_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let mut dot: i64 = 0;
            let mut norm_a: i64 = 0;
            let mut norm_b: i64 = 0;
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let rem_a = chunks_a.remainder();
            let rem_b = chunks_b.remainder();
            for (ca, cb) in chunks_a.zip(chunks_b) {
                for i in 0..8 {
                    let xi = ca[i] as i64;
                    let yi = cb[i] as i64;
                    dot += xi * yi;
                    norm_a += xi * xi;
                    norm_b += yi * yi;
                }
            }
            for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
                let xi = x as i64;
                let yi = y as i64;
                dot += xi * yi;
                norm_a += xi * xi;
                norm_b += yi * yi;
            }
            let denom_sq = norm_a * norm_b;
            if denom_sq == 0 {
                return 1_000_000;
            }
            let denom = (denom_sq as f64).sqrt();
            let cos_sim = dot as f64 / denom;
            ((1.0 - cos_sim) * 1_000_000.0) as i32
        }
    }

    // -----------------------------------------------------------------------
    // aarch64 NEON — 16-wide i8 with widening multiply-accumulate
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    pub mod neon {
        use std::arch::aarch64::*;

        #[inline]
        pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let va = vld1q_s8(a_ptr.add(off));
                    let vb = vld1q_s8(b_ptr.add(off));

                    // Widen to i16 and subtract
                    let a_lo = vmovl_s8(vget_low_s8(va));
                    let b_lo = vmovl_s8(vget_low_s8(vb));
                    let diff_lo = vsubq_s16(a_lo, b_lo);

                    let a_hi = vmovl_s8(vget_high_s8(va));
                    let b_hi = vmovl_s8(vget_high_s8(vb));
                    let diff_hi = vsubq_s16(a_hi, b_hi);

                    // Widening multiply-accumulate: d² into i32
                    acc0 = vmlal_s16(acc0, vget_low_s16(diff_lo), vget_low_s16(diff_lo));
                    acc1 = vmlal_s16(acc1, vget_high_s16(diff_lo), vget_high_s16(diff_lo));
                    acc0 = vmlal_s16(acc0, vget_low_s16(diff_hi), vget_low_s16(diff_hi));
                    acc1 = vmlal_s16(acc1, vget_high_s16(diff_hi), vget_high_s16(diff_hi));
                }

                let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));

                let off = main_iters * 16;
                for i in off..len {
                    let d = *a_ptr.add(i) as i32 - *b_ptr.add(i) as i32;
                    sum += d * d;
                }
                sum
            }
        }

        #[inline]
        pub fn dot_product_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let va = vld1q_s8(a_ptr.add(off));
                    let vb = vld1q_s8(b_ptr.add(off));

                    let a_lo = vmovl_s8(vget_low_s8(va));
                    let b_lo = vmovl_s8(vget_low_s8(vb));
                    let a_hi = vmovl_s8(vget_high_s8(va));
                    let b_hi = vmovl_s8(vget_high_s8(vb));

                    // Widening multiply-accumulate: a*b into i32
                    acc0 = vmlal_s16(acc0, vget_low_s16(a_lo), vget_low_s16(b_lo));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a_lo), vget_high_s16(b_lo));
                    acc0 = vmlal_s16(acc0, vget_low_s16(a_hi), vget_low_s16(b_hi));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a_hi), vget_high_s16(b_hi));
                }

                let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));

                let off = main_iters * 16;
                for i in off..len {
                    sum += *a_ptr.add(i) as i32 * *b_ptr.add(i) as i32;
                }
                -sum
            }
        }

        #[inline]
        pub fn cosine_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let z = vdupq_n_s32(0);
                let (mut dot0, mut dot1) = (z, z);
                let (mut na0, mut na1) = (z, z);
                let (mut nb0, mut nb1) = (z, z);

                let main_iters = len / 16;
                for i in 0..main_iters {
                    let off = i * 16;
                    let va = vld1q_s8(a_ptr.add(off));
                    let vb = vld1q_s8(b_ptr.add(off));

                    let a_lo = vmovl_s8(vget_low_s8(va));
                    let b_lo = vmovl_s8(vget_low_s8(vb));
                    let a_hi = vmovl_s8(vget_high_s8(va));
                    let b_hi = vmovl_s8(vget_high_s8(vb));

                    let (al, ah) = (vget_low_s16(a_lo), vget_high_s16(a_lo));
                    let (bl, bh) = (vget_low_s16(b_lo), vget_high_s16(b_lo));
                    let (ahl, ahh) = (vget_low_s16(a_hi), vget_high_s16(a_hi));
                    let (bhl, bhh) = (vget_low_s16(b_hi), vget_high_s16(b_hi));

                    dot0 = vmlal_s16(dot0, al, bl);
                    dot1 = vmlal_s16(dot1, ah, bh);
                    dot0 = vmlal_s16(dot0, ahl, bhl);
                    dot1 = vmlal_s16(dot1, ahh, bhh);

                    na0 = vmlal_s16(na0, al, al);
                    na1 = vmlal_s16(na1, ah, ah);
                    na0 = vmlal_s16(na0, ahl, ahl);
                    na1 = vmlal_s16(na1, ahh, ahh);

                    nb0 = vmlal_s16(nb0, bl, bl);
                    nb1 = vmlal_s16(nb1, bh, bh);
                    nb0 = vmlal_s16(nb0, bhl, bhl);
                    nb1 = vmlal_s16(nb1, bhh, bhh);
                }

                let mut dot = vaddvq_s32(vaddq_s32(dot0, dot1)) as i64;
                let mut norm_a = vaddvq_s32(vaddq_s32(na0, na1)) as i64;
                let mut norm_b = vaddvq_s32(vaddq_s32(nb0, nb1)) as i64;

                let off = main_iters * 16;
                for i in off..len {
                    let xi = *a_ptr.add(i) as i64;
                    let yi = *b_ptr.add(i) as i64;
                    dot += xi * yi;
                    norm_a += xi * xi;
                    norm_b += yi * yi;
                }

                let denom_sq = norm_a * norm_b;
                if denom_sq == 0 {
                    return 1_000_000;
                }
                let denom = (denom_sq as f64).sqrt();
                let cos_sim = dot as f64 / denom;
                ((1.0 - cos_sim) * 1_000_000.0) as i32
            }
        }
    }

    // -----------------------------------------------------------------------
    // x86_64 AVX2 — 16-wide i8 via cvtepi8_epi16 + madd_epi16
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub mod avx2 {
        #[allow(unused_imports)]
        use std::arch::x86_64::*;

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn hsum_i32x8(v: __m256i) -> i32 {
            let v128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            let v64 = _mm_hadd_epi32(v128, v128);
            let v32 = _mm_hadd_epi32(v64, v64);
            _mm_cvtsi128_si32(v32)
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub unsafe fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();

            // Process 32 i8 elements per iteration (2 × 16)
            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;

                let va0 = _mm_loadu_si128(a_ptr.add(off) as *const __m128i);
                let vb0 = _mm_loadu_si128(b_ptr.add(off) as *const __m128i);
                let a16_0 = _mm256_cvtepi8_epi16(va0);
                let b16_0 = _mm256_cvtepi8_epi16(vb0);
                let diff0 = _mm256_sub_epi16(a16_0, b16_0);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(diff0, diff0));

                let va1 = _mm_loadu_si128(a_ptr.add(off + 16) as *const __m128i);
                let vb1 = _mm_loadu_si128(b_ptr.add(off + 16) as *const __m128i);
                let a16_1 = _mm256_cvtepi8_epi16(va1);
                let b16_1 = _mm256_cvtepi8_epi16(vb1);
                let diff1 = _mm256_sub_epi16(a16_1, b16_1);
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(diff1, diff1));
            }

            // Handle remaining 16-element chunk
            let mut off = main_iters * 32;
            if off + 16 <= len {
                let va = _mm_loadu_si128(a_ptr.add(off) as *const __m128i);
                let vb = _mm_loadu_si128(b_ptr.add(off) as *const __m128i);
                let a16 = _mm256_cvtepi8_epi16(va);
                let b16 = _mm256_cvtepi8_epi16(vb);
                let diff = _mm256_sub_epi16(a16, b16);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(diff, diff));
                off += 16;
            }

            let mut sum = hsum_i32x8(_mm256_add_epi32(acc0, acc1));

            for i in off..len {
                let d = *a_ptr.add(i) as i32 - *b_ptr.add(i) as i32;
                sum += d * d;
            }
            sum
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub unsafe fn dot_product_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();

            let main_iters = len / 32;
            for i in 0..main_iters {
                let off = i * 32;

                let va0 = _mm_loadu_si128(a_ptr.add(off) as *const __m128i);
                let vb0 = _mm_loadu_si128(b_ptr.add(off) as *const __m128i);
                let a16_0 = _mm256_cvtepi8_epi16(va0);
                let b16_0 = _mm256_cvtepi8_epi16(vb0);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a16_0, b16_0));

                let va1 = _mm_loadu_si128(a_ptr.add(off + 16) as *const __m128i);
                let vb1 = _mm_loadu_si128(b_ptr.add(off + 16) as *const __m128i);
                let a16_1 = _mm256_cvtepi8_epi16(va1);
                let b16_1 = _mm256_cvtepi8_epi16(vb1);
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a16_1, b16_1));
            }

            let mut off = main_iters * 32;
            if off + 16 <= len {
                let va = _mm_loadu_si128(a_ptr.add(off) as *const __m128i);
                let vb = _mm_loadu_si128(b_ptr.add(off) as *const __m128i);
                let a16 = _mm256_cvtepi8_epi16(va);
                let b16 = _mm256_cvtepi8_epi16(vb);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a16, b16));
                off += 16;
            }

            let mut sum = hsum_i32x8(_mm256_add_epi32(acc0, acc1));

            for i in off..len {
                sum += *a_ptr.add(i) as i32 * *b_ptr.add(i) as i32;
            }
            -sum
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub unsafe fn cosine_distance_i8(a: &[i8], b: &[i8]) -> i32 {
            debug_assert_eq!(a.len(), b.len());
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let z = _mm256_setzero_si256();
            let mut dot_acc = z;
            let mut norma_acc = z;
            let mut normb_acc = z;

            let main_iters = len / 16;
            for i in 0..main_iters {
                let off = i * 16;
                let va = _mm_loadu_si128(a_ptr.add(off) as *const __m128i);
                let vb = _mm_loadu_si128(b_ptr.add(off) as *const __m128i);
                let a16 = _mm256_cvtepi8_epi16(va);
                let b16 = _mm256_cvtepi8_epi16(vb);

                dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(a16, b16));
                norma_acc = _mm256_add_epi32(norma_acc, _mm256_madd_epi16(a16, a16));
                normb_acc = _mm256_add_epi32(normb_acc, _mm256_madd_epi16(b16, b16));
            }

            let mut dot = hsum_i32x8(dot_acc) as i64;
            let mut norm_a = hsum_i32x8(norma_acc) as i64;
            let mut norm_b = hsum_i32x8(normb_acc) as i64;

            let off = main_iters * 16;
            for i in off..len {
                let xi = *a_ptr.add(i) as i64;
                let yi = *b_ptr.add(i) as i64;
                dot += xi * yi;
                norm_a += xi * xi;
                norm_b += yi * yi;
            }

            let denom_sq = norm_a * norm_b;
            if denom_sq == 0 {
                return 1_000_000;
            }
            let denom = (denom_sq as f64).sqrt();
            let cos_sim = dot as f64 / denom;
            ((1.0 - cos_sim) * 1_000_000.0) as i32
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: cross-validate SIMD output against scalar for many dimension sizes
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIMS: &[usize] = &[
        0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 128, 384, 385, 512, 768, 1024,
    ];

    fn make_f32_vecs(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * 0.1) - (dim as f32 * 0.05))
            .collect();
        let b: Vec<f32> = (0..dim)
            .map(|i| ((i * 7 + 3) as f32 * 0.1) - (dim as f32 * 0.05))
            .collect();
        (a, b)
    }

    fn make_i8_vecs(dim: usize) -> (Vec<i8>, Vec<i8>) {
        let a: Vec<i8> = (0..dim)
            .map(|i| ((i * 37 + 13) % 255) as i32 - 127)
            .map(|v| v as i8)
            .collect();
        let b: Vec<i8> = (0..dim)
            .map(|i| ((i * 73 + 29) % 255) as i32 - 127)
            .map(|v| v as i8)
            .collect();
        (a, b)
    }

    fn f32_tol(scalar: f32) -> f32 {
        1e-4 * scalar.abs().max(1.0)
    }

    // --- f32 cross-validation ---

    #[test]
    fn test_l2_f32_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_f32_vecs(dim);
            let scalar = f32_ops::scalar::l2_distance(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = f32_ops::neon::l2_distance(&a, &b);
                assert!(
                    (neon - scalar).abs() < f32_tol(scalar),
                    "NEON l2 mismatch at dim={dim}: neon={neon}, scalar={scalar}"
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { f32_ops::avx2::l2_distance(&a, &b) };
                assert!(
                    (avx - scalar).abs() < f32_tol(scalar),
                    "AVX2 l2 mismatch at dim={dim}: avx={avx}, scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn test_dot_product_f32_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_f32_vecs(dim);
            let scalar = f32_ops::scalar::dot_product_distance(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = f32_ops::neon::dot_product_distance(&a, &b);
                assert!(
                    (neon - scalar).abs() < f32_tol(scalar),
                    "NEON dot mismatch at dim={dim}: neon={neon}, scalar={scalar}"
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { f32_ops::avx2::dot_product_distance(&a, &b) };
                assert!(
                    (avx - scalar).abs() < f32_tol(scalar),
                    "AVX2 dot mismatch at dim={dim}: avx={avx}, scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn test_cosine_f32_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_f32_vecs(dim);
            let scalar = f32_ops::scalar::cosine_distance(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = f32_ops::neon::cosine_distance(&a, &b);
                assert!(
                    (neon - scalar).abs() < f32_tol(scalar),
                    "NEON cosine mismatch at dim={dim}: neon={neon}, scalar={scalar}"
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { f32_ops::avx2::cosine_distance(&a, &b) };
                assert!(
                    (avx - scalar).abs() < f32_tol(scalar),
                    "AVX2 cosine mismatch at dim={dim}: avx={avx}, scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn test_cosine_prenorm_f32_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_f32_vecs(dim);
            let scalar = f32_ops::scalar::cosine_distance_prenorm(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = f32_ops::neon::cosine_distance_prenorm(&a, &b);
                assert!(
                    (neon - scalar).abs() < f32_tol(scalar),
                    "NEON cosine_prenorm mismatch at dim={dim}: neon={neon}, scalar={scalar}"
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { f32_ops::avx2::cosine_distance_prenorm(&a, &b) };
                assert!(
                    (avx - scalar).abs() < f32_tol(scalar),
                    "AVX2 cosine_prenorm mismatch at dim={dim}: avx={avx}, scalar={scalar}"
                );
            }
        }
    }

    // --- i8 cross-validation ---

    #[test]
    fn test_l2_i8_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_i8_vecs(dim);
            let scalar = i8_ops::scalar::l2_distance_i8(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = i8_ops::neon::l2_distance_i8(&a, &b);
                assert_eq!(neon, scalar, "NEON l2_i8 mismatch at dim={dim}");
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { i8_ops::avx2::l2_distance_i8(&a, &b) };
                assert_eq!(avx, scalar, "AVX2 l2_i8 mismatch at dim={dim}");
            }
        }
    }

    #[test]
    fn test_dot_product_i8_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_i8_vecs(dim);
            let scalar = i8_ops::scalar::dot_product_distance_i8(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = i8_ops::neon::dot_product_distance_i8(&a, &b);
                assert_eq!(neon, scalar, "NEON dot_i8 mismatch at dim={dim}");
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { i8_ops::avx2::dot_product_distance_i8(&a, &b) };
                assert_eq!(avx, scalar, "AVX2 dot_i8 mismatch at dim={dim}");
            }
        }
    }

    #[test]
    fn test_cosine_i8_simd_vs_scalar() {
        for &dim in TEST_DIMS {
            let (a, b) = make_i8_vecs(dim);
            let scalar = i8_ops::scalar::cosine_distance_i8(&a, &b);

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let neon = i8_ops::neon::cosine_distance_i8(&a, &b);
                assert!(
                    (neon - scalar).abs() <= 1,
                    "NEON cosine_i8 mismatch at dim={dim}: neon={neon}, scalar={scalar}"
                );
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if x86_has_avx2_fma() {
                let avx = unsafe { i8_ops::avx2::cosine_distance_i8(&a, &b) };
                assert!(
                    (avx - scalar).abs() <= 1,
                    "AVX2 cosine_i8 mismatch at dim={dim}: avx={avx}, scalar={scalar}"
                );
            }
        }
    }
}
