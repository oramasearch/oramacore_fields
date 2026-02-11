//! Configuration types for NumberStorage.

use crate::error::Error;

/// Delete ratio threshold for compaction strategy.
///
/// When the ratio of deleted entries to total entries exceeds this threshold,
/// compaction will apply deletions immediately. Otherwise, deletions are
/// carried forward to the next compaction.
///
/// Valid range: 0.0 to 1.0 (exclusive of both endpoints).
/// Default: 0.1 (10%)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    value: f64,
}

impl Threshold {
    /// Create a new Threshold from a f64 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not in the range (0.0, 1.0).
    pub fn try_new(value: f64) -> Result<Self, Error> {
        if value.is_nan() || value <= 0.0 || value >= 1.0 {
            return Err(Error::NaNNotAllowed);
        }
        Ok(Self { value })
    }

    /// Get the threshold value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self { value: 0.1 }
    }
}

impl TryFrom<f64> for Threshold {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_threshold() {
        let t = Threshold::default();
        assert!((t.value() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_valid_threshold() {
        let t = Threshold::try_new(0.5).unwrap();
        assert!((t.value() - 0.5).abs() < f64::EPSILON);

        let t = Threshold::try_new(0.01).unwrap();
        assert!((t.value() - 0.01).abs() < f64::EPSILON);

        let t = Threshold::try_new(0.99).unwrap();
        assert!((t.value() - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_threshold() {
        assert!(Threshold::try_new(0.0).is_err());
        assert!(Threshold::try_new(1.0).is_err());
        assert!(Threshold::try_new(-0.1).is_err());
        assert!(Threshold::try_new(1.5).is_err());
        assert!(Threshold::try_new(f64::NAN).is_err());
    }

    #[test]
    fn test_try_from() {
        let t: Threshold = 0.2.try_into().unwrap();
        assert!((t.value() - 0.2).abs() < f64::EPSILON);
    }
}
