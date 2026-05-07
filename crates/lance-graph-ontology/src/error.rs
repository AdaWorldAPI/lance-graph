//! Error type for the ontology crate.
//!
//! Public error enum with `thiserror` derives. The variants name the failure
//! site (TTL, Lance, namespace, bridge) so that consumers can pattern-match
//! and recover where appropriate. No serde, per the workspace "no JSON
//! serialization in types" rule.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("TTL parse error in {path}: {message}")]
    TtlParse { path: PathBuf, message: String },

    #[error("namespace `{0}` is not registered")]
    UnknownNamespace(String),

    #[error("bridge `{bridge_id}`: public name `{public_name}` is not in scope (namespace lock)")]
    OutOfScope {
        bridge_id: String,
        public_name: String,
    },

    #[error("OGIT URI `{0}` does not match the expected `ogit.<Namespace>:<Entity>` shape")]
    InvalidOgitUri(String),

    #[error("ontology registry has no entry for `{0}`")]
    NotFound(String),

    #[error("toml decode error in semantic types: {0}")]
    TomlDecode(String),

    #[error("checksum mismatch for `{0}` — TTL fragment changed but registry says it is idempotent")]
    ChecksumMismatch(String),

    #[error("hydration produced 0 mappings from {0:?} — refusing to commit an empty registry")]
    EmptyHydration(PathBuf),

    #[cfg(feature = "lance-cache")]
    #[error("lance dataset error: {0}")]
    Lance(String),

    #[cfg(feature = "lance-cache")]
    #[error("arrow record-batch error: {0}")]
    Arrow(String),

    #[error("internal: {0}")]
    Other(String),
}

impl Error {
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
