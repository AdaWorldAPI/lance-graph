//! Error type for tileset parsing.

use std::fmt;

/// Errors produced while reading a 3D Tiles tileset document.
#[derive(Debug)]
pub enum Error {
    /// The input was not valid JSON, or did not match the tileset schema.
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Json(e) => write!(f, "invalid tileset JSON: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Json(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

/// Convenience alias for fallible tileset operations.
pub type Result<T> = std::result::Result<T, Error>;
