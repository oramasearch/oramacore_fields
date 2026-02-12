//! IndexableNumber trait for generic numeric types.

use super::error::Error;
use serde_json::Value;
use std::cmp::Ordering;

/// A trait for numeric types that can be indexed.
///
/// This trait provides the interface for encoding/decoding values
/// to/from bytes, comparing values, and validating inputs.
pub trait IndexableNumber: Copy + Clone + Send + Sync + PartialOrd + 'static {
    /// Convert the value to 8 bytes (native-endian).
    fn to_bytes(self) -> [u8; 8];

    /// Convert 8 bytes (native-endian) back to a value.
    fn from_bytes(bytes: [u8; 8]) -> Self;

    /// Return the type name for metadata and error messages.
    fn type_name() -> &'static str;

    /// Compare two values with total ordering.
    fn compare(a: Self, b: Self) -> Ordering;

    /// Validate the value before insertion.
    ///
    /// Returns an error if the value is invalid (e.g., NaN for f64).
    fn validate(self) -> Result<(), Error>;

    /// Extract a value from a JSON number.
    ///
    /// Returns `None` if the value cannot be represented as this type.
    fn from_json_number(value: &Value) -> Option<Self>;
}

impl IndexableNumber for u64 {
    #[inline]
    fn to_bytes(self) -> [u8; 8] {
        self.to_ne_bytes()
    }

    #[inline]
    fn from_bytes(bytes: [u8; 8]) -> Self {
        u64::from_ne_bytes(bytes)
    }

    fn type_name() -> &'static str {
        "u64"
    }

    #[inline]
    fn compare(a: Self, b: Self) -> Ordering {
        a.cmp(&b)
    }

    #[inline]
    fn validate(self) -> Result<(), Error> {
        // All u64 values are valid
        Ok(())
    }

    #[inline]
    fn from_json_number(value: &Value) -> Option<Self> {
        value.as_u64()
    }
}

impl IndexableNumber for f64 {
    #[inline]
    fn to_bytes(self) -> [u8; 8] {
        self.to_bits().to_ne_bytes()
    }

    #[inline]
    fn from_bytes(bytes: [u8; 8]) -> Self {
        f64::from_bits(u64::from_ne_bytes(bytes))
    }

    fn type_name() -> &'static str {
        "f64"
    }

    #[inline]
    fn compare(a: Self, b: Self) -> Ordering {
        // Use total_cmp for consistent ordering (handles NaN, -0.0, etc.)
        a.total_cmp(&b)
    }

    #[inline]
    fn validate(self) -> Result<(), Error> {
        if self.is_nan() {
            return Err(Error::NaNNotAllowed);
        }
        Ok(())
    }

    #[inline]
    fn from_json_number(value: &Value) -> Option<Self> {
        value.as_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_roundtrip() {
        let values = [0u64, 1, 42, u64::MAX, u64::MAX / 2];
        for v in values {
            let bytes = v.to_bytes();
            let decoded = u64::from_bytes(bytes);
            assert_eq!(v, decoded);
        }
    }

    #[test]
    fn test_u64_ordering() {
        assert_eq!(u64::compare(0, 1), Ordering::Less);
        assert_eq!(u64::compare(1, 1), Ordering::Equal);
        assert_eq!(u64::compare(2, 1), Ordering::Greater);
    }

    #[test]
    fn test_u64_validate() {
        assert!(0u64.validate().is_ok());
        assert!(u64::MAX.validate().is_ok());
    }

    #[test]
    fn test_f64_roundtrip() {
        let values = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            std::f64::consts::PI,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for v in values {
            let bytes = v.to_bytes();
            let decoded = f64::from_bytes(bytes);
            // Use to_bits for comparison to handle -0.0 vs 0.0
            assert_eq!(v.to_bits(), decoded.to_bits());
        }
    }

    #[test]
    fn test_f64_ordering() {
        // Normal ordering
        assert_eq!(f64::compare(0.0, 1.0), Ordering::Less);
        assert_eq!(f64::compare(1.0, 1.0), Ordering::Equal);
        assert_eq!(f64::compare(2.0, 1.0), Ordering::Greater);

        // Negative numbers
        assert_eq!(f64::compare(-1.0, 0.0), Ordering::Less);
        assert_eq!(f64::compare(-2.0, -1.0), Ordering::Less);

        // Special values
        assert_eq!(
            f64::compare(f64::NEG_INFINITY, f64::INFINITY),
            Ordering::Less
        );
        assert_eq!(f64::compare(f64::NEG_INFINITY, -1000.0), Ordering::Less);
        assert_eq!(f64::compare(1000.0, f64::INFINITY), Ordering::Less);

        // total_cmp treats -0.0 < +0.0
        assert_eq!(f64::compare(-0.0, 0.0), Ordering::Less);
    }

    #[test]
    fn test_f64_validate() {
        assert!(0.0f64.validate().is_ok());
        assert!((-0.0f64).validate().is_ok());
        assert!(1.0f64.validate().is_ok());
        assert!(f64::INFINITY.validate().is_ok());
        assert!(f64::NEG_INFINITY.validate().is_ok());

        // NaN is not allowed
        assert!(f64::NAN.validate().is_err());
    }

    #[test]
    fn test_type_names() {
        assert_eq!(u64::type_name(), "u64");
        assert_eq!(f64::type_name(), "f64");
    }
}
