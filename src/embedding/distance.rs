use super::config::DistanceMetric;

pub type DistanceFn = fn(&[f32], &[f32]) -> f32;
pub type QuantizedDistanceFn = fn(&[i8], &[i8]) -> i32;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
}

pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    -a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>()
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
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

pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i32 - y as i32;
            d * d
        })
        .sum::<i32>()
}

pub fn dot_product_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    -a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum::<i32>()
}

pub fn cosine_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot: i64 = 0;
    let mut norm_a: i64 = 0;
    let mut norm_b: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
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

pub fn resolve_distance_fn(metric: DistanceMetric) -> DistanceFn {
    match metric {
        DistanceMetric::L2 => l2_distance,
        DistanceMetric::DotProduct => dot_product_distance,
        DistanceMetric::Cosine => cosine_distance,
    }
}

pub fn resolve_quantized_distance_fn(metric: DistanceMetric) -> QuantizedDistanceFn {
    match metric {
        DistanceMetric::L2 => l2_distance_i8,
        DistanceMetric::DotProduct => dot_product_distance_i8,
        DistanceMetric::Cosine => cosine_distance_i8,
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
    fn test_resolve_functions() {
        let _ = resolve_distance_fn(DistanceMetric::L2);
        let _ = resolve_distance_fn(DistanceMetric::DotProduct);
        let _ = resolve_distance_fn(DistanceMetric::Cosine);
        let _ = resolve_quantized_distance_fn(DistanceMetric::L2);
        let _ = resolve_quantized_distance_fn(DistanceMetric::DotProduct);
        let _ = resolve_quantized_distance_fn(DistanceMetric::Cosine);
    }
}
