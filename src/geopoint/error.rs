use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidLatitude(f64),
    InvalidLongitude(f64),
    TooFewVertices(usize),
    UnsupportedVersion(u32),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidLatitude(v) => write!(f, "invalid latitude {v}: must be in -90.0..=90.0"),
            Error::InvalidLongitude(v) => {
                write!(f, "invalid longitude {v}: must be in -180.0..=180.0")
            }
            Error::TooFewVertices(n) => {
                write!(f, "too few vertices ({n}): polygon requires at least 3")
            }
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version: {v}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
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
