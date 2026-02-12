/// Controls when deletions are physically removed during compaction (value between 0.0 and 1.0).
///
/// For example, a threshold of 0.1 triggers removal when deletions exceed 10% of total entries.
#[derive(Debug, Clone, Copy)]
pub struct Threshold(f64);

impl Threshold {
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Threshold(0.1)
    }
}

impl TryFrom<f64> for Threshold {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if (0.0..=1.0).contains(&value) {
            Ok(Threshold(value))
        } else {
            Err("threshold must be between 0.0 and 1.0")
        }
    }
}

impl TryFrom<f32> for Threshold {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Threshold::try_from(value as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_valid() {
        let t: Threshold = 0.5f64.try_into().unwrap();
        assert!((t.value() - 0.5).abs() < f64::EPSILON);

        let t: Threshold = 0.0f64.try_into().unwrap();
        assert!((t.value() - 0.0).abs() < f64::EPSILON);

        let t: Threshold = 1.0f64.try_into().unwrap();
        assert!((t.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_invalid() {
        assert!(Threshold::try_from(-0.1f64).is_err());
        assert!(Threshold::try_from(1.1f64).is_err());
        assert!(Threshold::try_from(-1.0f64).is_err());
        assert!(Threshold::try_from(2.0f64).is_err());
    }

    #[test]
    fn test_threshold_from_f32() {
        let t: Threshold = 0.5f32.try_into().unwrap();
        assert!((t.value() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_threshold_default() {
        let t = Threshold::default();
        assert!((t.value() - 0.1).abs() < f64::EPSILON);
    }
}
