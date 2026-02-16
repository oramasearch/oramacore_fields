//! Error types for the NumberStorage crate.

use std::fmt;
use std::io;

/// Errors that can occur during NumberStorage operations.
#[derive(Debug)]
pub enum Error {
    /// NaN values are not allowed in f64 indexes.
    NaNNotAllowed,

    /// Type mismatch when opening an index.
    TypeMismatch {
        expected: &'static str,
        found: String,
    },

    /// Invalid magic bytes in file header.
    InvalidMagic {
        expected: &'static str,
        found: String,
    },

    /// Unsupported file format version.
    UnsupportedVersion { version: u32 },

    /// Corrupted entry data.
    CorruptedEntry,

    /// Cannot compact to same version number as current active version.
    VersionConflict { version: u64 },

    /// I/O error.
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NaNNotAllowed => write!(f, "NaN values are not allowed"),
            Error::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {expected}, found {found}")
            }
            Error::InvalidMagic { expected, found } => {
                write!(f, "invalid magic: expected {expected}, found {found}")
            }
            Error::UnsupportedVersion { version } => {
                write!(f, "unsupported format version: {version}")
            }
            Error::VersionConflict { version } => write!(
                f,
                "cannot compact to version {version}: same as current active version"
            ),
            Error::CorruptedEntry => write!(f, "corrupted entry data"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::NaNNotAllowed
            | Error::TypeMismatch { .. }
            | Error::InvalidMagic { .. }
            | Error::UnsupportedVersion { .. }
            | Error::CorruptedEntry
            | Error::VersionConflict { .. } => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::NaNNotAllowed;
        assert_eq!(err.to_string(), "NaN values are not allowed");

        let err = Error::TypeMismatch {
            expected: "u64",
            found: "f64".to_string(),
        };
        assert_eq!(err.to_string(), "type mismatch: expected u64, found f64");

        let err = Error::InvalidMagic {
            expected: "HDRv0001",
            found: "BADMAGIC".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "invalid magic: expected HDRv0001, found BADMAGIC"
        );

        let err = Error::UnsupportedVersion { version: 99 };
        assert_eq!(err.to_string(), "unsupported format version: 99");

        let err = Error::CorruptedEntry;
        assert_eq!(err.to_string(), "corrupted entry data");
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
