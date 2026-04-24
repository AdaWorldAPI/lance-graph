//! Error type for the archetype transcode crate.
//!
//! Per ADR-0001 Decision 1, this crate defines its own error surface rather
//! than mirroring the Python `VangelisTech/archetype` exceptions. The
//! variants below are scoped to the scaffold (DU-2.1..2.6) — Lance I/O
//! wiring for `World::fork` / `World::at_tick` is deliberately parked
//! behind DU-2.8.

use thiserror::Error;

/// Top-level error type for archetype transcode operations.
///
/// All fallible methods in this crate return `Result<T, ArchetypeError>`.
/// The `Unimplemented` variant is used for stubs that will be wired in
/// follow-up deliverables; see `World::fork` / `World::at_tick` for the
/// canonical example.
#[derive(Debug, Error)]
pub enum ArchetypeError {
    /// A stub method that has not yet been wired. The `method` field names
    /// the specific method (for example, `"World::fork"`). Once the
    /// corresponding deliverable (DU-2.7 / DU-2.8) lands, the variant
    /// stays but the call site no longer returns it.
    #[error("archetype method `{method}` is not yet implemented (scaffold stub)")]
    Unimplemented {
        /// Fully-qualified method name, for example `"World::fork"`.
        method: &'static str,
    },

    /// A `Processor::process` invocation received a `RecordBatch` whose
    /// schema does not match what the processor declared via `matches`.
    /// The `expected` / `actual` fields are human-readable descriptions;
    /// no Arrow schema equality is defined at the scaffold stage.
    #[error("schema mismatch: expected {expected}, got {actual}")]
    SchemaMismatch {
        /// Human-readable description of the expected schema.
        expected: String,
        /// Human-readable description of the actual schema that arrived.
        actual: String,
    },

    /// Placeholder for Lance dataset I/O errors. Once DU-2.8 wires
    /// `lance::checkout(branch)` into `World::fork`, the inner type will
    /// be upgraded from `String` to `lance::Error`. Today it carries a
    /// bare message — no `lance` dependency on this PR per the plan's
    /// Non-goals section.
    #[error("lance I/O error: {0}")]
    LanceIo(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimplemented_carries_method_name() {
        let err = ArchetypeError::Unimplemented { method: "World::fork" };
        let msg = format!("{err}");
        assert!(msg.contains("World::fork"));
        assert!(msg.contains("not yet implemented"));
    }

    #[test]
    fn schema_mismatch_formats() {
        let err = ArchetypeError::SchemaMismatch {
            expected: "Schema{a: Int32}".to_string(),
            actual: "Schema{a: Utf8}".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("expected"));
        assert!(msg.contains("Int32"));
        assert!(msg.contains("Utf8"));
    }
}
