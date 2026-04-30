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
    /// 32-bit days since Unix epoch — Arrow `Date32`.
    Date32,
    /// Fixed-size list of f32 with the given size (e.g. 16 384 for VSA carriers).
    FixedSizeListF32(usize),
    /// Fixed-size binary of the given byte width (e.g. 64 for `Fingerprint`).
    FixedSizeBinary(usize),
    /// Arrow `Null` — used for fields whose payload doesn't cross the boundary.
    Null,
}

/// One column in the outer-ontology view of an entity type.
///
/// Carries the [`CodecRoute`] copied from the upstream `PropertySpec` so
/// that read-side dispatch (in [`super::cam_pq_decode`]) can consult the
/// declarative route directly, without re-running the model-weight
/// classifier in `lance_graph_contract::cam::route_tensor`.
#[derive(Clone, Debug)]
pub struct OuterColumn {
    pub name: &'static str,
    pub kind: PropertyKind,
    pub semantic_type: SemanticType,
    pub marking: Marking,
    pub arrow_type_code: ArrowTypeCode,
    /// Codec route declared by the upstream `PropertySpec`. Round-1 of
    /// the transcode crate inferred this from the column name; round-2
    /// honours the contract's own field — every PropertySpec already
    /// carries a `codec_route` and hand-rolled inference would only
    /// ever drift from it.
    pub codec_route: lance_graph_contract::cam::CodecRoute,
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
            codec_route: p.codec_route,
        })
        .collect()
}

/// Map a `SemanticType` to the Arrow type the external surface should
/// emit. Picked so that DataFusion predicate pushdown is meaningful:
///
/// - **`Currency` → `Float32`**: numeric comparison (`amount > 1000.0`)
///   becomes a fast scan filter instead of a string compare.
/// - **`Date(_)` → `Date32`**: temporal comparison (`birth >= 1980-01-01`)
///   pushes down through Arrow's date kernels.
/// - **`CustomerId` / `InvoiceNumber` → `UInt64`**: per-tenant numeric
///   identifiers; faster equality checks than string equality.
/// - **Everything else → `Utf8`**: opaque text, lexical compare only.
///
/// Currency carries an ISO 4217 code (`Currency("EUR")`); the code is
/// metadata at the schema layer, not per-row data, so we don't widen the
/// Arrow type to a struct. Consumers that need the code read it from the
/// `OuterColumn.semantic_type` field.
fn arrow_type_for_semantic(st: &SemanticType) -> ArrowTypeCode {
    match st {
        SemanticType::Currency(_) => ArrowTypeCode::Float32,
        SemanticType::Date(_) => ArrowTypeCode::Date32,
        SemanticType::CustomerId | SemanticType::InvoiceNumber => ArrowTypeCode::UInt64,
        // Geo / File / Image / Address / Iban / Email / Phone / Url /
        // TaxId / PlainText all collapse to opaque text. Round 3 may
        // pivot specific ones (Geo → struct{lat: f32, lon: f32}) when
        // a consumer asks.
        _ => ArrowTypeCode::Utf8,
    }
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
        ArrowTypeCode::Date32 => DataType::Date32,
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
    /// Date32 — days since Unix epoch (1970-01-01). Negative values are
    /// pre-epoch. Backed by Arrow's native `Date32Array`.
    Date32(Vec<i32>),
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
            OwnedColumn::Date32(v) => v.len(),
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
            OwnedColumn::Date32(v) => Ok(Arc::new(arrow::array::Date32Array::from(v)) as ArrayRef),
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

/// Build a `RecordBatch` for a **partial** row write — for PATCH-style
/// upserts where only a subset of fields is being changed.
///
/// Rules (stricter than [`from_columns`] in one place, looser in another):
///
/// 1. **Required columns** declared in the ontology must still be
///    present. Required-by-construction means the entity row can't exist
///    without them; allowing partial writes that omit required fields
///    would invite silent rows missing key data.
/// 2. **Optional / Free columns** may be omitted. Omitted columns appear
///    as Arrow null arrays in the output batch — DataFusion's standard
///    `IS NULL` filter sees them.
/// 3. Undeclared columns are still rejected (same as the strict path).
/// 4. The mode is honest about itself — round-1 emits all-null arrays
///    for missing optionals, which costs `O(nrows)` bytes per skipped
///    column. A full Arrow null-bitmap path is the round-2 lift; this
///    keeps the API surface stable while lifting the contract.
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn from_columns_partial(
    soa: &OuterSchema,
    ids: Vec<u64>,
    body_columns: Vec<(&str, OwnedColumn)>,
) -> Result<RecordBatch, TranscodeError> {
    use arrow::array::new_null_array;

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

    for (idx, soa_col) in soa.columns.iter().enumerate() {
        let slot = by_name
            .iter_mut()
            .find(|(n, c)| n == soa_col.name && c.is_some());
        match slot {
            Some((_, slot_opt)) => {
                let owned = slot_opt.take().expect("just-checked-some");
                if owned.rows() != nrows {
                    return Err(TranscodeError::RowCountMismatch);
                }
                arrays.push(owned.into_array()?);
            }
            None => {
                if matches!(soa_col.kind, PropertyKind::Required) {
                    return Err(TranscodeError::MissingColumn(soa_col.name.to_string()));
                }
                // Optional / Free — fill with nulls. Field index in
                // arrow_schema is `idx + 2` (id + entity_type prefix).
                let field = arrow_schema.field(idx + 2);
                arrays.push(new_null_array(field.data_type(), nrows));
            }
        }
    }

    if let Some((extra, _)) = by_name.iter().find(|(_, c)| c.is_some()) {
        return Err(TranscodeError::UndeclaredColumn(extra.clone()));
    }

    RecordBatch::try_new(arrow_schema, arrays).map_err(TranscodeError::Arrow)
}

/// Build a `RecordBatch` from a stream of [`ExpandedTriple`]s — the
/// **reverse direction** that the original transcode round flagged as
/// deferred (Phase 5 in #309's ROADMAP, reframed here as Phase-2-B).
///
/// `ExpandedTriple` is what `Ontology::expand_entity()` returns — one
/// triple per (entity_id, predicate). A row in the outer-DTO view is
/// the gather of all triples sharing one `subject_label`. This helper
/// performs that gather: groups by subject, projects each group's
/// predicate→value pairs into the schema's column slots, and emits a
/// single `RecordBatch` covering all subjects in the input slice.
///
/// ## Domain-agnostic
///
/// Works for any `(Ontology, entity_type)` pair. Subject extraction
/// uses the canonical `entity:{type}:{id}` label format that
/// `expand_entity()` produces. Callers that mint subject labels by
/// other means must canonicalise first.
///
/// ## What this honestly does NOT do today
///
/// - It does NOT consult an `SpoStore`. The Phase-2 plan doc described
///   that as `walk SpoStore::scan(lookup)`, but `SpoStore` is
///   fingerprint-Hamming-indexed and one-way (the FNV-1a fingerprint
///   doesn't round-trip back to `entity_id`). A real SpoStore reader
///   needs a side-table mapping subject fingerprint → entity_id, which
///   is a separate primitive.
/// - It does NOT push values through the per-column codec. Every value
///   crosses as the `object_label` string from the triple. Round 3
///   adds typed-value reconstruction.
///
/// ## Errors
///
/// - `UnknownEntityType` if any triple's `entity_type_id` doesn't
///   match the schema's id.
/// - `Arrow` for `RecordBatch::try_new` failures (shape misalignment).
///
/// Triples whose `predicate` isn't declared in the schema are silently
/// dropped. This matches the BBB "outer view shows only declared
/// fields" rule — undeclared properties stay inside the substrate.
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn triples_to_batch(
    soa: &OuterSchema,
    triples: &[lance_graph_contract::ontology::ExpandedTriple],
) -> Result<RecordBatch, TranscodeError> {
    use std::collections::BTreeMap;

    // Group triples by subject_label, preserving insertion order via the
    // BTreeMap (lexical subject sort). Entity_id parsed from
    // "entity:{type}:{id}" once per group.
    let mut grouped: BTreeMap<String, (u64, Vec<&lance_graph_contract::ontology::ExpandedTriple>)> =
        BTreeMap::new();

    for t in triples {
        if t.entity_type_id != soa.entity_type_id {
            return Err(TranscodeError::EntityTypeMismatch {
                expected: soa.entity_type_id,
                got: t.entity_type_id,
            });
        }
        let entity_id = parse_entity_id_from_label(&t.subject_label, soa.entity_type)
            .ok_or_else(|| TranscodeError::BadSubjectLabel(t.subject_label.clone()))?;
        grouped
            .entry(t.subject_label.clone())
            .or_insert_with(|| (entity_id, Vec::new()))
            .1
            .push(t);
    }

    let nrows = grouped.len();
    let arrow_schema = arrow_schema(soa);

    // Build columns. id + entity_type are always present; body columns
    // are projected from the gathered triples.
    let mut ids: Vec<u64> = Vec::with_capacity(nrows);
    let mut entity_type_strs: Vec<&'static str> = Vec::with_capacity(nrows);
    // For each declared body column, accumulate one Option<String> per
    // row in iteration order; missing predicates become null.
    let ncols = soa.columns.len();
    let mut body: Vec<Vec<Option<String>>> =
        (0..ncols).map(|_| Vec::with_capacity(nrows)).collect();

    for (_, (entity_id, group_triples)) in &grouped {
        ids.push(*entity_id);
        entity_type_strs.push(soa.entity_type);

        // For each declared column, find the matching triple (if any).
        for (col_idx, soa_col) in soa.columns.iter().enumerate() {
            let value = group_triples
                .iter()
                .find(|t| t.predicate == soa_col.name)
                .map(|t| t.object_label.clone());
            body[col_idx].push(value);
        }
    }

    // Materialise into Arrow arrays. id + entity_type first; body
    // columns as nullable string arrays (round-1 — every value crosses
    // as Utf8; round-3 plumbs typed reconstruction).
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    arrays.push(Arc::new(UInt64Array::from(ids)) as ArrayRef);
    arrays.push(Arc::new(StringArray::from(entity_type_strs)) as ArrayRef);

    for (idx, col_values) in body.into_iter().enumerate() {
        let field = arrow_schema.field(idx + 2);
        // Required columns disallow null per the schema. If we collected
        // a None for a required column the row is incomplete; surface
        // that as an error rather than silently dropping the field.
        let required = !field.is_nullable();
        if required && col_values.iter().any(|v| v.is_none()) {
            return Err(TranscodeError::MissingColumn(soa.columns[idx].name.into()));
        }
        // Round-1: every column emits as nullable Utf8 regardless of
        // the schema's declared Arrow type. The schema's typed
        // semantic_type → ArrowTypeCode mapping (Float32, Date32, etc.)
        // applies on the from_columns / from_columns_partial path,
        // which has typed input. For triples_to_batch the input is
        // string-shaped (object_label), so round-1 keeps it Utf8.
        // Cast to declared type happens at the consumer; round 3 adds
        // an in-place per-column casting layer here.
        let _ = field; // Silence unused-warning when the cast layer lands.
        let arr = StringArray::from(col_values);
        arrays.push(Arc::new(arr) as ArrayRef);
    }

    // The schema we computed assumes typed columns (Float32, Date32,
    // etc.) per arrow_schema(). Since round-1 emits Utf8 for every body
    // column, we re-derive a "round-1 lenient" schema here that swaps
    // every body field to Utf8 (nullable per the original `kind`).
    //
    // This avoids a `RecordBatch::try_new` mismatch error and is
    // documented honestly: round 3 swaps to typed values + the lenient
    // schema goes away.
    let lenient_schema = round1_lenient_schema(soa);
    RecordBatch::try_new(lenient_schema, arrays).map_err(TranscodeError::Arrow)
}

/// Round-1 helper: produce a `SchemaRef` whose body columns are all
/// nullable `Utf8`, matching what [`triples_to_batch`] emits.
#[cfg(any(feature = "persist", feature = "query-lite"))]
pub fn round1_lenient_schema(soa: &OuterSchema) -> SchemaRef {
    let mut fields: Vec<Field> = Vec::with_capacity(soa.columns.len() + 2);
    fields.push(Field::new("id", DataType::UInt64, false));
    fields.push(Field::new("entity_type", DataType::Utf8, false));
    for col in &soa.columns {
        fields.push(Field::new(col.name, DataType::Utf8, true));
    }
    Arc::new(ArrowSchema::new(fields))
}

/// Parse "entity:{type}:{id}" into the trailing `id`. Used by
/// [`triples_to_batch`] to recover entity_id from the canonical
/// subject label that `Ontology::expand_entity()` mints.
#[cfg(any(feature = "persist", feature = "query-lite"))]
fn parse_entity_id_from_label(subject_label: &str, expected_type: &str) -> Option<u64> {
    let prefix = format!("entity:{expected_type}:");
    subject_label
        .strip_prefix(&prefix)
        .and_then(|rest| rest.parse::<u64>().ok())
}

/// Errors from the transcode layer.
#[derive(Debug)]
pub enum TranscodeError {
    MissingColumn(String),
    UndeclaredColumn(String),
    RowCountMismatch,
    ShapeMismatch,
    /// A triple's `entity_type_id` did not match the schema's id —
    /// `triples_to_batch` rejects mixed-type input rather than silently
    /// projecting the wrong rows.
    EntityTypeMismatch {
        expected: lance_graph_contract::ontology::EntityTypeId,
        got: lance_graph_contract::ontology::EntityTypeId,
    },
    /// `subject_label` didn't follow the canonical `entity:{type}:{id}`
    /// shape that `Ontology::expand_entity()` produces.
    BadSubjectLabel(String),
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
            TranscodeError::EntityTypeMismatch { expected, got } => write!(
                f,
                "triple's entity_type_id ({got}) does not match the schema's ({expected})"
            ),
            TranscodeError::BadSubjectLabel(s) => write!(
                f,
                "subject label `{s}` is not in the canonical `entity:{{type}}:{{id}}` form"
            ),
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

    // ── triples_to_batch (Phase-2-B reverse path) ────────────────────────────
    #[cfg(any(feature = "persist", feature = "query-lite"))]
    mod triples_round_trip {
        use super::*;
        use lance_graph_contract::ontology::SchemaExpander;

        fn build_ontology() -> Ontology {
            Ontology::builder("Test")
                .schema(
                    Schema::builder("Patient")
                        .required("patient_id")
                        .required("name")
                        .optional("krankenkasse")
                        .build(),
                )
                .build()
        }

        #[test]
        fn triples_to_batch_produces_one_row_per_subject() {
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            // Build triples for two patients via expand_entity (the
            // canonical mint).
            let mut triples = ont.expand_entity(
                "Patient",
                42,
                &[
                    ("patient_id", b"42-AA"),
                    ("name", b"Anna Mueller"),
                    ("krankenkasse", b"AOK"),
                ],
            );
            triples.extend(ont.expand_entity(
                "Patient",
                17,
                &[("patient_id", b"17-BB"), ("name", b"Boris Stolz")],
            ));
            let batch = triples_to_batch(&soa, &triples).unwrap();
            assert_eq!(batch.num_rows(), 2);
            assert_eq!(batch.num_columns(), 5);
            assert_eq!(batch.schema().field(0).name(), "id");
            assert_eq!(batch.schema().field(1).name(), "entity_type");
        }

        #[test]
        fn triples_to_batch_rejects_mixed_entity_types() {
            let ont = Ontology::builder("Test")
                .schema(Schema::builder("Patient").required("name").build())
                .schema(Schema::builder("Diagnosis").required("code").build())
                .build();
            let patient_soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            // Triples for Diagnosis fed to a Patient soa — must error.
            let triples = ont.expand_entity("Diagnosis", 1, &[("code", b"M51")]);
            let err = triples_to_batch(&patient_soa, &triples).unwrap_err();
            assert!(matches!(err, TranscodeError::EntityTypeMismatch { .. }));
        }

        #[test]
        fn triples_to_batch_returns_empty_batch_for_empty_input() {
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            let batch = triples_to_batch(&soa, &[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 5); // id + entity_type + 3 body
        }

        #[test]
        fn triples_to_batch_drops_undeclared_predicates_silently() {
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            // expand_entity with an undeclared "weird" predicate. The
            // ontology's expand_entity emits triples even for
            // undeclared predicates (with PropertyKind::Free defaults);
            // triples_to_batch should drop them since they're not in
            // the schema's column list.
            let triples = ont.expand_entity(
                "Patient",
                1,
                &[
                    ("patient_id", b"1-XX"),
                    ("name", b"X"),
                    ("weird", b"shouldn't surface"),
                ],
            );
            let batch = triples_to_batch(&soa, &triples).unwrap();
            assert_eq!(batch.num_rows(), 1);
            assert_eq!(batch.num_columns(), 5);
            assert!(batch.schema().fields().iter().all(|f| f.name() != "weird"));
        }

        #[test]
        fn triples_to_batch_rejects_missing_required_column() {
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            // Only one of two required columns supplied for entity 1.
            let triples = ont.expand_entity("Patient", 1, &[("patient_id", b"1-XX")]);
            let err = triples_to_batch(&soa, &triples).unwrap_err();
            assert!(matches!(err, TranscodeError::MissingColumn(_)));
        }

        #[test]
        fn triples_to_batch_subject_label_round_trip() {
            // Verifies the subject_label format that expand_entity
            // mints and parse_entity_id_from_label expects agree —
            // i.e. "entity:Patient:42" → 42.
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            let triples =
                ont.expand_entity("Patient", 999_999, &[("patient_id", b"X"), ("name", b"Y")]);
            let batch = triples_to_batch(&soa, &triples).unwrap();
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(id_col.value(0), 999_999);
        }

        #[test]
        fn triples_to_batch_preserves_lex_subject_order() {
            // BTreeMap groups by lexical subject_label, so two patients
            // 17 and 42 should appear in order "entity:Patient:17"
            // before "entity:Patient:42" (lexical, not numeric).
            let ont = build_ontology();
            let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
            let mut triples =
                ont.expand_entity("Patient", 42, &[("patient_id", b"42"), ("name", b"A")]);
            triples.extend(ont.expand_entity(
                "Patient",
                17,
                &[("patient_id", b"17"), ("name", b"B")],
            ));
            let batch = triples_to_batch(&soa, &triples).unwrap();
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            // Lexical order: "17" < "42" → row 0 is 17.
            assert_eq!(id_col.value(0), 17);
            assert_eq!(id_col.value(1), 42);
        }
    }
}
