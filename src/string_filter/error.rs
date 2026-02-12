use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    InvalidThreshold,
    CorruptedIndex(String),
    InvalidVersion(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::InvalidThreshold => write!(f, "threshold must be between 0.0 and 1.0"),
            Error::CorruptedIndex(msg) => write!(f, "corrupted index: {msg}"),
            Error::InvalidVersion(msg) => write!(f, "invalid version: {msg}"),
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
