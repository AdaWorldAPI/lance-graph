//! The `Processor` trait — a function from `RecordBatch` to `RecordBatch`
//! gated by a schema predicate.
//!
//! Per ADR-0001 Decision 1, Processors transcode Python's ECS-system
//! concept into Rust. A processor is NOT a method on a world; it is a
//! free-standing operator that declares (a) which schemas it matches
//! and (b) how it transforms a matching batch. The world drives the
//! dispatch in a later deliverable (DU-2.7+); today only the trait
//! lives here.

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;

use crate::error::ArchetypeError;

/// A transcode operator on Arrow `RecordBatch`es.
///
/// Implementors declare a schema matcher and a batch transformer. The
/// scaffold trait uses associated functions (no `&self`) so that the
/// World-level dispatcher can hold a `&'static` table of processors
/// without needing heap allocation. A later deliverable may introduce
/// a `dyn Processor` object-safe variant alongside.
pub trait Processor {
    /// Return `true` iff this processor can operate on a batch with the
    /// given schema. Called once per batch before `process`; mismatches
    /// surface via the enclosing loop, not via `ArchetypeError`.
    fn matches(schema: &Schema) -> bool;

    /// Transform the input batch. Implementations MUST return a schema
    /// that is either identical to the input or a schema-migration
    /// documented in the crate README. On schema violations, return
    /// `ArchetypeError::SchemaMismatch`.
    fn process(batch: RecordBatch) -> Result<RecordBatch, ArchetypeError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Identity processor — matches any single-column Int64 schema and
    /// returns the input unchanged. Exists purely to prove the trait
    /// is constructable.
    struct IdentityProcessor;

    impl Processor for IdentityProcessor {
        fn matches(schema: &Schema) -> bool {
            schema.fields().len() == 1
                && schema.field(0).data_type() == &DataType::Int64
        }

        fn process(batch: RecordBatch) -> Result<RecordBatch, ArchetypeError> {
            Ok(batch)
        }
    }

    #[test]
    fn trait_object_is_constructable() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        assert!(IdentityProcessor::matches(&schema));

        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![arr]).unwrap();

        let out = IdentityProcessor::process(batch).unwrap();
        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.num_columns(), 1);
    }

    #[test]
    fn matches_rejects_wrong_schema() {
        let schema = Schema::new(vec![Field::new("x", DataType::Utf8, false)]);
        assert!(!IdentityProcessor::matches(&schema));
    }
}
