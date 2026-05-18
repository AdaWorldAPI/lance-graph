//! Crate-local error type for `lance-graph-tikv-provider` (Glue #2).
//!
//! Per plan §5: this crate bridges TiKV ranges to Arrow `TableProvider`.
//! Errors can originate from TiKV client calls, Arrow schema validation,
//! or binary decoding of KV bytes into columnar rows.
//!
//! Sprint 1 TODO: wire the `Tikv` variant to `tikv_client::Error` via
//! `From<tikv_client::Error>` once the real client dep is enabled.

/// Crate-local error variants for `lance-graph-tikv-provider`.
///
/// Per plan §5: this crate performs three classes of operations that can
/// fail — TiKV I/O, Arrow schema/buffer operations, and binary decoding
/// of KV bytes into columnar rows.  Each has its own variant so callers
/// can pattern-match on error class without pulling in upstream error types.
///
/// Sprint 1 TODO: add `From<tikv_client::Error>` and
/// `From<arrow_schema::ArrowError>` conversions.
#[derive(Debug)]
pub enum Error {
    /// A TiKV client operation failed (connection, transaction, scan, etc.).
    ///
    /// Wraps a stringified error so this crate has no direct dep on
    /// `tikv_client::Error` until Sprint 1 wires the real client.
    Tikv(String),

    /// An Arrow schema or buffer operation failed.
    ///
    /// Covers `ArrowError`, schema mismatch, and projection out-of-bounds.
    Arrow(String),

    /// Binary decoding of a TiKV value into Arrow columns failed.
    ///
    /// Occurs when the raw bytes stored in TiKV don't match the expected
    /// encoding described by the `NodeShape` or `EdgeShape` schema.
    Decode(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Tikv(msg) => write!(f, "tikv error: {msg}"),
            Error::Arrow(msg) => write!(f, "arrow error: {msg}"),
            Error::Decode(msg) => write!(f, "decode error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convert `Error` into DataFusion's `DataFusionError` so `scan()` can
/// use `?` inside `async fn scan(...) -> DfResult<...>`.
///
/// Sprint 1 TODO: use `DataFusionError::External` once the DataFusion dep
/// is finalised; for now wraps as `Execution`.
impl From<Error> for datafusion::error::DataFusionError {
    fn from(e: Error) -> Self {
        datafusion::error::DataFusionError::Execution(e.to_string())
    }
}
