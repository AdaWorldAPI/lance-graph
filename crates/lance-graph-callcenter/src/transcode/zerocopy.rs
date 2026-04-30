//! Outer-ontology row data → Arrow `RecordBatch` (the wire shape).
//!
//! Inputs are owned scalar / vector columns; outputs are Arrow arrays
//! that DataFusion + Lance + downstream consumers all understand.
//!
//! ## Why `OwnedColumn`?
//!
//! The cheap-zerocopy lane in Arrow 57 is `Vec<T>` → `Buffer`: for
//! `T: ArrowNativeType` it is an `O(1)` reinterpretation — Vec's
//! allocation becomes the Buffer's allocation, no per-element copy.
//! Borrowed `&[T]` references can't take that lane without either
//! moving ownership or wrapping in a custom-owner `Buffer::from_bytes`,
//! both of which require the producer to expose owned slices.
//!
//! Today's producer surface (`BindSpace` in `cognitive-shader-driver`)
//! does not yet hand out ownership of its column allocations. So this
//! round we accept owned columns and document the path explicitly. The
//! BindSpace zerocopy view ships when the producer side adds an
//! accessor (tracked in callcenter's wiring plan).
//!
//! ## Domain-agnostic
//!
//! The mapper takes any `Ontology` + `entity_type` name and projects
//! them to an Arrow schema. No medcare-specific or smb-specific code
//! lives here — that's all in `ontology_dto::medcare_ontology()` /
//! `ontology_dto::smb_ontology()`.

#[cfg(any(feature = "persist", feature = "query-lite"))]
use std::sync::Arc;

#[cfg(any(feature = "persist", feature = "query-lite"))]
use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, StringArray,
    UInt32Array, UInt64Array,
};
#[cfg(any(feature = "persist", feature = "query-lite"))]
use arrow::buffer::Buffer;
#[cfg(any(feature = "persist", feature = "query-lite"))]
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
#[cfg(any(feature = "persist", feature = "query-lite"))]
use arrow::record_batch::RecordBatch;

use lance_graph_contract::ontology::{entity_type_id, EntityTypeId, Ontology};
use lance_graph_contract::property::{Marking, PropertyKind, Schema, SemanticType};

/// Lightweight enum mirroring the Arrow types this module emits.
/// Arrow-feature-gated translation lives in [`arrow_data_type`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrowTypeCode {
    Utf8,
    UInt32,
    UInt64,
    Float32,
    /// Fixed-size list of f32 with the given size (e.g. 16 384 for VSA carriers).
    FixedSizeListF32(usize),
    /// Fixed-size binary of the given byte width (e.g. 64 for `Fingerprint`).
    FixedSizeBinary(usize),
    /// Arrow `Null` — used for fields whose payload doesn't cross the boundary.
    Null,
}

/// One column in the outer-ontology view of an entity type.
#[derive(Clone, Debug)]
pub struct OuterColumn {
    pub name: &'static str,
    pub kind: PropertyKind,
    pub semantic_type: SemanticType,
    pub marking: Marking,
    pub arrow_type_code: ArrowTypeCode,
}

/// Outer-ontology projection of one entity type's columns. Derived from
/// an [`Ontology`] schema by [`OuterSchema::from_ontology`].
#[derive(Clone, Debug)]
pub struct OuterSchema {
    /// Locale-stable schema key (`"Patient"`, `"Customer"`, …).
    pub entity_type: &'static str,
    /// Numeric id assigned by the parent ontology.
    pub entity_type_id: EntityTypeId,
    /// Body columns derived from the `Schema`.
    pub columns: Vec<OuterColumn>,
}

impl OuterSchema {
    /// Derive an outer schema for `entity_type` from `ontology`. Returns
    /// `None` if the entity type is not declared.
    pub fn from_ontology(ontology: &Ontology, entity_type: &str) -> Option<Self> {
        let schema = ontology.schema(entity_type)?;
        let id = entity_type_id(ontology, entity_type);
        Some(Self {
            entity_type: schema.name,
            entity_type_id: id,
            columns: schema_columns(schema),
        })
    }
}

fn schema_columns(schema: &Schema) -> Vec<OuterColumn> {
    schema
        .properties
        .iter()
        .map(|p| OuterColumn {
            name: p.predicate,
            kind: p.kind,
            semantic_type: p.semantic_type.clone(),
            marking: p.marking,
            arrow_type_code: arrow_type_for_semantic(&p.semantic_type),
        })
        .collect()
}

/// Round-1 collapses every semantic type to `Utf8`. Round 2 plumbs richer
/// Arrow types (`Date32`, `Decimal128`, etc.) per consumer demand.
fn arrow_type_for_semantic(_st: &SemanticType) -> ArrowTypeCode {
    ArrowTypeCode::Utf8
}

// ── Arrow wiring (feature `persist`) ─────────────────────────────────────────

/// Map an [`OuterSchema`] to an Arrow [`SchemaRef`].
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn arrow_schema(soa: &OuterSchema) -> SchemaRef {
    let mut fields: Vec<Field> = Vec::with_capacity(soa.columns.len() + 2);
    fields.push(Field::new("id", DataType::UInt64, false));
    fields.push(Field::new("entity_type", DataType::Utf8, false));
    for col in &soa.columns {
        fields.push(Field::new(
            col.name,
            arrow_data_type(col.arrow_type_code),
            !matches!(col.kind, PropertyKind::Required),
        ));
    }
    Arc::new(ArrowSchema::new(fields))
}

/// Translate an [`ArrowTypeCode`] to a concrete Arrow [`DataType`].
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn arrow_data_type(code: ArrowTypeCode) -> DataType {
    match code {
        ArrowTypeCode::Utf8 => DataType::Utf8,
        ArrowTypeCode::UInt32 => DataType::UInt32,
        ArrowTypeCode::UInt64 => DataType::UInt64,
        ArrowTypeCode::Float32 => DataType::Float32,
        ArrowTypeCode::FixedSizeListF32(n) => DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            n as i32,
        ),
        ArrowTypeCode::FixedSizeBinary(n) => DataType::FixedSizeBinary(n as i32),
        ArrowTypeCode::Null => DataType::Null,
    }
}

/// Owned column input — moves ownership into the Arrow Buffer.
#[cfg(any(feature = "persist", feature = "query-lite"))]
#[derive(Debug)]
pub enum OwnedColumn {
    UInt64(Vec<u64>),
    UInt32(Vec<u32>),
    Float32(Vec<f32>),
    Utf8(Vec<String>),
    /// VSA carriers — flat row-major `f32`. `inner_size` is the per-row
    /// dimensionality (e.g. 16384 for Vsa16kF32). Total length must be
    /// `nrows * inner_size`.
    FixedSizeListF32 {
        flat: Vec<f32>,
        inner_size: usize,
    },
    /// Fingerprints — flat row-major bytes, `inner_size` bytes per row.
    FixedSizeBinary {
        flat: Vec<u8>,
        inner_size: usize,
    },
}

#[cfg(any(feature = "persist", feature = "query-lite"))]
impl OwnedColumn {
    /// Length in rows. Returns the floor on misaligned shapes; the
    /// downstream `into_array()` returns a typed `ShapeMismatch` error,
    /// so the caller never panics.
    pub fn rows(&self) -> usize {
        match self {
            OwnedColumn::UInt64(v) => v.len(),
            OwnedColumn::UInt32(v) => v.len(),
            OwnedColumn::Float32(v) => v.len(),
            OwnedColumn::Utf8(v) => v.len(),
            OwnedColumn::FixedSizeListF32 { flat, inner_size } => {
                if *inner_size == 0 {
                    0
                } else {
                    flat.len() / inner_size
                }
            }
            OwnedColumn::FixedSizeBinary { flat, inner_size } => {
                if *inner_size == 0 {
                    0
                } else {
                    flat.len() / inner_size
                }
            }
        }
    }

    fn into_array(self) -> Result<ArrayRef, TranscodeError> {
        match self {
            OwnedColumn::UInt64(v) => Ok(Arc::new(UInt64Array::from(v)) as ArrayRef),
            OwnedColumn::UInt32(v) => Ok(Arc::new(UInt32Array::from(v)) as ArrayRef),
            OwnedColumn::Float32(v) => Ok(Arc::new(Float32Array::from(v)) as ArrayRef),
            OwnedColumn::Utf8(v) => Ok(Arc::new(StringArray::from(v)) as ArrayRef),
            OwnedColumn::FixedSizeListF32 { flat, inner_size } => {
                if inner_size == 0 || flat.len() % inner_size != 0 {
                    return Err(TranscodeError::ShapeMismatch);
                }
                let nrows = flat.len() / inner_size;
                let values = Float32Array::from(flat);
                let field = Arc::new(Field::new("item", DataType::Float32, false));
                let arr =
                    FixedSizeListArray::try_new(field, inner_size as i32, Arc::new(values), None)
                        .map_err(|_| TranscodeError::ShapeMismatch)?;
                debug_assert_eq!(arr.len(), nrows);
                Ok(Arc::new(arr) as ArrayRef)
            }
            OwnedColumn::FixedSizeBinary { flat, inner_size } => {
                if inner_size == 0 || flat.len() % inner_size != 0 {
                    return Err(TranscodeError::ShapeMismatch);
                }
                let buf = Buffer::from_vec(flat);
                let arr = FixedSizeBinaryArray::try_new(inner_size as i32, buf, None)
                    .map_err(|_| TranscodeError::ShapeMismatch)?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }
}

/// Build a `RecordBatch` from named owned columns. `id` and
/// `entity_type` are filled from the schema; all other columns must
/// match `soa.columns` by name. Undeclared columns are rejected at the
/// boundary — silent widening would defeat the ontology's purpose.
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn from_columns(
    soa: &OuterSchema,
    ids: Vec<u64>,
    body_columns: Vec<(&str, OwnedColumn)>,
) -> Result<RecordBatch, TranscodeError> {
    let nrows = ids.len();
    let arrow_schema = arrow_schema(soa);

    let mut by_name: Vec<(String, Option<OwnedColumn>)> = body_columns
        .into_iter()
        .map(|(n, c)| (n.to_string(), Some(c)))
        .collect();

    let entity_type_strs: Vec<&'static str> = (0..nrows).map(|_| soa.entity_type).collect();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    arrays.push(Arc::new(UInt64Array::from(ids)) as ArrayRef);
    arrays.push(Arc::new(StringArray::from(entity_type_strs)) as ArrayRef);

    for soa_col in &soa.columns {
        let slot = by_name
            .iter_mut()
            .find(|(n, c)| n == soa_col.name && c.is_some());
        let owned = match slot {
            Some((_, slot_opt)) => slot_opt
                .take()
                .ok_or_else(|| TranscodeError::MissingColumn(soa_col.name.to_string()))?,
            None => return Err(TranscodeError::MissingColumn(soa_col.name.to_string())),
        };
        if owned.rows() != nrows {
            return Err(TranscodeError::RowCountMismatch);
        }
        arrays.push(owned.into_array()?);
    }

    if let Some((extra, _)) = by_name.iter().find(|(_, c)| c.is_some()) {
        return Err(TranscodeError::UndeclaredColumn(extra.clone()));
    }

    RecordBatch::try_new(arrow_schema, arrays).map_err(TranscodeError::Arrow)
}

/// Errors from the transcode layer.
#[derive(Debug)]
pub enum TranscodeError {
    MissingColumn(String),
    UndeclaredColumn(String),
    RowCountMismatch,
    ShapeMismatch,
    #[cfg(any(feature = "persist", feature = "query-lite"))]
    Arrow(arrow::error::ArrowError),
}

impl core::fmt::Display for TranscodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TranscodeError::MissingColumn(c) => write!(f, "missing column: {c}"),
            TranscodeError::UndeclaredColumn(c) => write!(
                f,
                "column {c} not declared in ontology schema (refusing to widen the boundary)"
            ),
            TranscodeError::RowCountMismatch => write!(f, "row count mismatch across columns"),
            TranscodeError::ShapeMismatch => write!(f, "fixed-size column shape mismatch"),
            #[cfg(any(feature = "persist", feature = "query-lite"))]
            TranscodeError::Arrow(e) => write!(f, "arrow error: {e}"),
        }
    }
}

impl std::error::Error for TranscodeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::Schema;

    fn build_test_ontology() -> Ontology {
        Ontology::builder("Test")
            .schema(
                Schema::builder("Patient")
                    .required("patient_id")
                    .required("name")
                    .required("geburtsdatum")
                    .optional("krankenkasse")
                    .build(),
            )
            .build()
    }

    #[test]
    fn outer_schema_derives_required_and_optional_from_ontology() {
        let ontology = build_test_ontology();
        let s = OuterSchema::from_ontology(&ontology, "Patient").expect("declared");
        assert_eq!(s.entity_type, "Patient");
        assert_eq!(s.entity_type_id, 1);
        assert_eq!(s.columns.len(), 4);
        assert!(s
            .columns
            .iter()
            .any(|c| c.name == "patient_id" && c.kind == PropertyKind::Required));
        assert!(s
            .columns
            .iter()
            .any(|c| c.name == "krankenkasse" && c.kind == PropertyKind::Optional));
    }

    #[test]
    fn outer_schema_returns_none_for_unknown_entity_type() {
        let ontology = build_test_ontology();
        assert!(OuterSchema::from_ontology(&ontology, "Unknown").is_none());
    }

    #[cfg(any(feature = "persist", feature = "query-lite"))]
    #[test]
    fn arrow_schema_includes_id_and_entity_type_first() {
        let ont = Ontology::builder("T")
            .schema(Schema::builder("Patient").required("name").build())
            .build();
        let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
        let s = arrow_schema(&soa);
        assert_eq!(s.field(0).name(), "id");
        assert_eq!(s.field(1).name(), "entity_type");
        assert_eq!(s.field(2).name(), "name");
    }

    #[cfg(any(feature = "persist", feature = "query-lite"))]
    #[test]
    fn from_columns_builds_record_batch_in_declared_order() {
        let ont = Ontology::builder("T")
            .schema(
                Schema::builder("Patient")
                    .required("name")
                    .required("age")
                    .build(),
            )
            .build();
        let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
        let batch = from_columns(
            &soa,
            vec![10u64, 11u64, 12u64],
            vec![
                (
                    "name",
                    OwnedColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
                ),
                (
                    "age",
                    OwnedColumn::Utf8(vec!["1".into(), "2".into(), "3".into()]),
                ),
            ],
        )
        .unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4);
    }

    #[cfg(any(feature = "persist", feature = "query-lite"))]
    #[test]
    fn from_columns_rejects_undeclared_column() {
        let ont = Ontology::builder("T")
            .schema(Schema::builder("Patient").required("name").build())
            .build();
        let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
        let err = from_columns(
            &soa,
            vec![1u64],
            vec![
                ("name", OwnedColumn::Utf8(vec!["a".into()])),
                ("uninvited", OwnedColumn::Utf8(vec!["x".into()])),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, TranscodeError::UndeclaredColumn(_)));
    }
}
