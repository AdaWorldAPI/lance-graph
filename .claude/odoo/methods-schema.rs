//! SoA-shaped DTO extraction schema — mirrors /tmp/work/methods.parquet.
//!
//! Auto-emitted by extract_soa.py. The columns below are the 1-to-1
//! Arrow schema; the codegen reads them via arrow-rs RecordBatch with
//! zero-copy column access.
//!
//! Each column is a contiguous Arrow buffer; cross-column sweeps run
//! through arrow::compute kernels (SIMD-accelerated). Reading
//! "all `family` values" = one ChunkedArray scan; filtering "where
//! transitivity = 'I' AND tekamolo = 'KA'" = two parallel column scans
//! + one boolean AND. The shape downstream IS the shape upstream.

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub fn methods_schema() -> Schema {
    Schema::new(vec![
        Field::new("function_name",     DataType::Utf8,                              false),
        Field::new("name_root",         DataType::Utf8,                              false),
        Field::new("family",            DataType::Utf8,                              false),  // dict-encoded in parquet
        Field::new("file",              DataType::Utf8,                              false),
        Field::new("line_start",        DataType::UInt32,                            false),
        Field::new("line_end",          DataType::UInt32,                            false),
        Field::new("body_lines",        DataType::UInt32,                            false),
        Field::new("signature",         DataType::Utf8,                              false),
        Field::new("match_id",          DataType::Utf8,                              false),
        Field::new("primary_decorator", DataType::Utf8,                              false),
        Field::new("decorators_all",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("transitivity",      DataType::Utf8,                              false),  // T | I
        Field::new("tekamolo",          DataType::Utf8,                              false),  // TE|KA|MO|LO|QU|(none)
        Field::new("mengenmass",        DataType::Utf8,                              false),  // money|percent|rate|count|date|none
        Field::new("regulatory_anchor",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("axis",              DataType::Utf8,                              false),  // Deterministic|Heuristic|Hybrid
        Field::new("reads_fields",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("writes_fields",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("invokes",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("raises",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("traverses",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("reads_env",         DataType::Boolean,                           false),
        Field::new("atom_count",        DataType::UInt32,                            false),
        Field::new("cost_estimate_ns",  DataType::UInt32,                            false),
    ])
}
