use std::fmt;

#[derive(Debug)]
pub enum Error {
    DimensionMismatch { expected: usize, got: usize },
    EmptyVector,
    InvalidDimensions(usize),
    NonFiniteValue,
    TooManyNodes { count: usize, max: usize },
    Io(std::io::Error),
    CorruptedFile(String),
    FormatVersionMismatch { expected: u32, found: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Error::EmptyVector => write!(f, "empty vector"),
            Error::InvalidDimensions(d) => write!(f, "invalid dimensions: {d}"),
            Error::NonFiniteValue => write!(f, "vector contains non-finite value (NaN or Inf)"),
            Error::TooManyNodes { count, max } => {
                write!(f, "too many nodes: {count} exceeds maximum {max}")
            }
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::CorruptedFile(msg) => write!(f, "corrupted file: {msg}"),
            Error::FormatVersionMismatch { expected, found } => {
                write!(
                    f,
                    "format version mismatch: expected {expected}, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
