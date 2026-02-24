use super::simd;

#[inline]
#[allow(unreachable_code)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::f32_ops::neon::l2_distance(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::f32_ops::avx2::l2_distance(a, b) };
    }

    simd::f32_ops::scalar::l2_distance(a, b)
}

#[inline]
#[allow(unreachable_code)]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::f32_ops::neon::dot_product_distance(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::f32_ops::avx2::dot_product_distance(a, b) };
    }

    simd::f32_ops::scalar::dot_product_distance(a, b)
}

#[inline]
#[allow(unreachable_code)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::f32_ops::neon::cosine_distance(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::f32_ops::avx2::cosine_distance(a, b) };
    }

    simd::f32_ops::scalar::cosine_distance(a, b)
}

/// Cosine distance where `a` is already a unit vector (||a|| = 1).
/// Skips computing norm_a, saving `dim` multiply-accumulate operations per call.
#[inline]
#[allow(unreachable_code)]
pub fn cosine_distance_prenorm(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::f32_ops::neon::cosine_distance_prenorm(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::f32_ops::avx2::cosine_distance_prenorm(a, b) };
    }

    simd::f32_ops::scalar::cosine_distance_prenorm(a, b)
}

#[inline]
#[allow(unreachable_code)]
pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::i8_ops::neon::l2_distance_i8(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::i8_ops::avx2::l2_distance_i8(a, b) };
    }

    simd::i8_ops::scalar::l2_distance_i8(a, b)
}

#[inline]
#[allow(unreachable_code)]
pub fn dot_product_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::i8_ops::neon::dot_product_distance_i8(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::i8_ops::avx2::dot_product_distance_i8(a, b) };
    }

    simd::i8_ops::scalar::dot_product_distance_i8(a, b)
}

#[inline]
#[allow(unreachable_code)]
pub fn cosine_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd::i8_ops::neon::cosine_distance_i8(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    if simd::x86_has_avx2_fma() {
        return unsafe { simd::i8_ops::avx2::cosine_distance_i8(a, b) };
    }

    simd::i8_ops::scalar::cosine_distance_i8(a, b)
}

pub trait Distance {
    fn distance(a: &[f32], b: &[f32]) -> f32;
    fn quantized_distance(a: &[i8], b: &[i8]) -> i32;
}

pub struct L2;
impl Distance for L2 {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        l2_distance(a, b)
    }
    #[inline(always)]
    fn quantized_distance(a: &[i8], b: &[i8]) -> i32 {
        l2_distance_i8(a, b)
    }
}

pub struct DotProduct;
impl Distance for DotProduct {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        dot_product_distance(a, b)
    }
    #[inline(always)]
    fn quantized_distance(a: &[i8], b: &[i8]) -> i32 {
        dot_product_distance_i8(a, b)
    }
}

pub struct Cosine;
impl Distance for Cosine {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        cosine_distance(a, b)
    }
    #[inline(always)]
    fn quantized_distance(a: &[i8], b: &[i8]) -> i32 {
        cosine_distance_i8(a, b)
    }
}

/// Cosine distance with pre-normalized query vector.
/// Uses `cosine_distance_prenorm` for f32 (skips query norm computation).
/// Quantized path still uses full `cosine_distance_i8` since quantization
/// changes norms and the quantized search is approximate anyway.
pub struct CosineNorm;
impl Distance for CosineNorm {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        cosine_distance_prenorm(a, b)
    }
    #[inline(always)]
    fn quantized_distance(a: &[i8], b: &[i8]) -> i32 {
        cosine_distance_i8(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!((l2_distance(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!((l2_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, distance = -32
        assert!((dot_product_distance(&a, &b) - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_i8() {
        let a: [i8; 3] = [1, 2, 3];
        let b: [i8; 3] = [4, 5, 6];
        assert_eq!(l2_distance_i8(&a, &b), 27);
    }

    #[test]
    fn test_dot_product_distance_i8() {
        let a: [i8; 3] = [1, 2, 3];
        let b: [i8; 3] = [4, 5, 6];
        assert_eq!(dot_product_distance_i8(&a, &b), -32);
    }

    #[test]
    fn test_distance_trait() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((L2::distance(&a, &b) - 27.0).abs() < 1e-6);
        assert!((DotProduct::distance(&a, &b) - (-32.0)).abs() < 1e-6);
        assert!(Cosine::distance(&a, &a).abs() < 1e-6);

        let ai: [i8; 3] = [1, 2, 3];
        let bi: [i8; 3] = [4, 5, 6];
        assert_eq!(L2::quantized_distance(&ai, &bi), 27);
        assert_eq!(DotProduct::quantized_distance(&ai, &bi), -32);
    }
}
