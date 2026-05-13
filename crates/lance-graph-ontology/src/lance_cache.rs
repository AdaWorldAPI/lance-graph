//! Lance dataset persistence for the `ontology_dictionary` table.
//!
//! Feature-gated behind `lance-cache` so the crate compiles without
//! `protoc` (Lance's `lance-encoding` build-script requires `protoc` via
//! `prost-build`). When enabled, this module owns the Arrow schema for
//! the dictionary table and translates between `MappingRow` and
//! `arrow::record_batch::RecordBatch`.
//!
//! ## Tables
//!
//! - `ontology_dictionary` — append-only rows, never UPDATE / DELETE.
//!   Soft-deletes go through the `active: Boolean` column.
//! - `ontology_meta` — single row updated, holds `ttl_root_checksum` for
//!   idempotent re-hydration.
//!
//! Every method here is async because Lance's native I/O is async.

use crate::error::{Error, Result};
use crate::namespace::{NamespaceId, OgitUri, SchemaKind, SchemaPtr};
use crate::proposal::MappingRow;
use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    UInt32Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_graph_contract::property::{Marking, SemanticType};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DICTIONARY_NAME: &str = "ontology_dictionary";
const META_NAME: &str = "ontology_meta";

pub struct LanceWriter {
    base: PathBuf,
}

impl LanceWriter {
    pub async fn open_or_create(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            base: path.to_path_buf(),
        })
    }

    pub fn dictionary_path(&self) -> PathBuf {
        self.base.join(DICTIONARY_NAME)
    }

    pub fn meta_path(&self) -> PathBuf {
        self.base.join(META_NAME)
    }

    pub async fn flush(&self, rows: &[MappingRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let batch = rows_to_record_batch(rows)?;
        let schema = batch.schema();
        let path = self.dictionary_path();
        let path_str = path.to_string_lossy().to_string();
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let stream = futures::stream::iter(vec![Ok(batch)]);
        let reader =
            arrow::record_batch::RecordBatchIterator::new(stream.into_inner_unwrap_iter(), schema);
        Dataset::write(reader, &path_str, Some(write_params))
            .await
            .map_err(|e| Error::Lance(format!("write {}: {e}", path_str)))?;
        Ok(())
    }

    pub async fn replay(&self) -> Result<Vec<MappingRow>> {
        let path = self.dictionary_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let path_str = path.to_string_lossy().to_string();
        let dataset = Dataset::open(&path_str)
            .await
            .map_err(|e| Error::Lance(format!("open {}: {e}", path_str)))?;
        let scanner = dataset
            .scan()
            .try_into_stream()
            .await
            .map_err(|e| Error::Lance(format!("scan: {e}")))?;
        use futures::StreamExt;
        let mut rows = Vec::new();
        let mut stream = scanner;
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(|e| Error::Lance(format!("batch: {e}")))?;
            rows.append(&mut record_batch_to_rows(&batch)?);
        }
        Ok(rows)
    }

    pub async fn last_root_checksum(&self) -> Result<Option<String>> {
        let path = self.meta_path();
        if !path.exists() {
            return Ok(None);
        }
        let path_str = path.to_string_lossy().to_string();
        let dataset = Dataset::open(&path_str)
            .await
            .map_err(|e| Error::Lance(format!("open meta: {e}")))?;
        let mut stream = dataset
            .scan()
            .try_into_stream()
            .await
            .map_err(|e| Error::Lance(format!("scan meta: {e}")))?;
        use futures::StreamExt;
        if let Some(batch) = stream.next().await {
            let batch = batch.map_err(|e| Error::Lance(format!("meta batch: {e}")))?;
            let col = batch
                .column_by_name("ttl_root_checksum")
                .ok_or_else(|| Error::Lance("missing ttl_root_checksum".to_string()))?;
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Lance("ttl_root_checksum not String".to_string()))?;
            if arr.len() > 0 {
                return Ok(Some(arr.value(0).to_string()));
            }
        }
        Ok(None)
    }

    pub async fn set_last_root_checksum(&self, checksum: &str) -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("ttl_root_checksum", DataType::Utf8, false),
            Field::new(
                "last_hydrated_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("crate_version", DataType::Utf8, false),
        ]));
        let now = chrono_micros();
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![checksum])),
            Arc::new(TimestampMicrosecondArray::from(vec![now])),
            Arc::new(StringArray::from(vec![env!("CARGO_PKG_VERSION")])),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| Error::Arrow(format!("meta batch: {e}")))?;
        let path = self.meta_path();
        let path_str = path.to_string_lossy().to_string();
        // Meta is a single-row table — overwrite.
        let stream = futures::stream::iter(vec![Ok(batch)]);
        let reader =
            arrow::record_batch::RecordBatchIterator::new(stream.into_inner_unwrap_iter(), schema);
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        Dataset::write(reader, &path_str, Some(write_params))
            .await
            .map_err(|e| Error::Lance(format!("write meta: {e}")))?;
        Ok(())
    }
}

fn dictionary_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("bridge_id", DataType::Utf8, false),
        Field::new("public_name", DataType::Utf8, false),
        Field::new("ogit_uri", DataType::Utf8, false),
        Field::new("namespace_id", DataType::UInt8, false),
        Field::new("schema_ptr", DataType::UInt32, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("semantic_type", DataType::Utf8, false),
        Field::new("marking", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("created_by", DataType::Utf8, false),
        Field::new("source_uri", DataType::Utf8, false),
        Field::new("active", DataType::Boolean, false),
        Field::new("checksum", DataType::Utf8, false),
    ]))
}

fn rows_to_record_batch(rows: &[MappingRow]) -> Result<RecordBatch> {
    let bridge_id: Vec<&str> = rows.iter().map(|r| r.bridge_id.as_str()).collect();
    let public_name: Vec<&str> = rows.iter().map(|r| r.public_name.as_str()).collect();
    let ogit_uri: Vec<&str> = rows.iter().map(|r| r.ogit_uri.as_str()).collect();
    let namespace_id: Vec<u8> = rows.iter().map(|r| r.namespace_id.raw()).collect();
    let schema_ptr: Vec<u32> = rows.iter().map(|r| r.schema_ptr.raw()).collect();
    let kind: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
    let semantic_type: Vec<String> = rows
        .iter()
        .map(|r| semantic_type_label(&r.semantic_type))
        .collect();
    let marking: Vec<&str> = rows.iter().map(|r| marking_label(r.marking)).collect();
    let confidence: Vec<f32> = rows.iter().map(|r| r.confidence).collect();
    let created_at: Vec<i64> = rows.iter().map(|r| r.created_at_us).collect();
    let created_by: Vec<&str> = rows.iter().map(|r| r.created_by.as_str()).collect();
    let source_uri: Vec<&str> = rows.iter().map(|r| r.source_uri.as_str()).collect();
    let active: Vec<bool> = rows.iter().map(|r| r.active).collect();
    let checksum: Vec<&str> = rows.iter().map(|r| r.checksum.as_str()).collect();

    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(bridge_id)),
        Arc::new(StringArray::from(public_name)),
        Arc::new(StringArray::from(ogit_uri)),
        Arc::new(UInt8Array::from(namespace_id)),
        Arc::new(UInt32Array::from(schema_ptr)),
        Arc::new(StringArray::from(kind)),
        Arc::new(StringArray::from(semantic_type)),
        Arc::new(StringArray::from(marking)),
        Arc::new(Float32Array::from(confidence)),
        Arc::new(TimestampMicrosecondArray::from(created_at)),
        Arc::new(StringArray::from(created_by)),
        Arc::new(StringArray::from(source_uri)),
        Arc::new(BooleanArray::from(active)),
        Arc::new(StringArray::from(checksum)),
    ];
    RecordBatch::try_new(dictionary_schema(), cols).map_err(|e| Error::Arrow(format!("{e}")))
}

fn record_batch_to_rows(batch: &RecordBatch) -> Result<Vec<MappingRow>> {
    let bridge_id = string_col(batch, "bridge_id")?;
    let public_name = string_col(batch, "public_name")?;
    let ogit_uri = string_col(batch, "ogit_uri")?;
    let namespace_id = u8_col(batch, "namespace_id")?;
    let schema_ptr = u32_col(batch, "schema_ptr")?;
    let kind = string_col(batch, "kind")?;
    let semantic_type = string_col(batch, "semantic_type")?;
    let marking = string_col(batch, "marking")?;
    let confidence = f32_col(batch, "confidence")?;
    let created_at = ts_col(batch, "created_at")?;
    let created_by = string_col(batch, "created_by")?;
    let source_uri = string_col(batch, "source_uri")?;
    let active = bool_col(batch, "active")?;
    let checksum = string_col(batch, "checksum")?;

    let mut rows = Vec::with_capacity(bridge_id.len());
    for i in 0..bridge_id.len() {
        // D-CASCADE-V1-7: codec-cascade columns not yet persisted; replay
        // defaults them. Producer pipeline writer is the follow-up.
        rows.push(MappingRow {
            bridge_id: bridge_id.value(i).to_string(),
            public_name: public_name.value(i).to_string(),
            ogit_uri: OgitUri::from_string_unchecked(ogit_uri.value(i)),
            namespace_id: NamespaceId(namespace_id.value(i)),
            schema_ptr: SchemaPtr::from_raw(schema_ptr.value(i)),
            kind: SchemaKind::parse(kind.value(i)).unwrap_or(SchemaKind::Entity),
            semantic_type: parse_semantic_type_label(semantic_type.value(i)),
            marking: parse_marking_label(marking.value(i)),
            confidence: confidence.value(i),
            created_at_us: created_at.value(i),
            created_by: created_by.value(i).to_string(),
            source_uri: source_uri.value(i).to_string(),
            active: active.value(i),
            checksum: checksum.value(i).to_string(),
            identity_codec: Default::default(),
            qualia_meta: Default::default(),
            thinking_style: None,
            attribute_sources: Vec::new(),
            subject_type: String::new(),
            object_type: String::new(),
            entity_type_ref: String::new(),
        });
    }
    Ok(rows)
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-Utf8 column `{name}`")))
}
fn u8_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt8Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-U8 column `{name}`")))
}
fn u32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-U32 column `{name}`")))
}
fn f32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-F32 column `{name}`")))
}
fn ts_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a TimestampMicrosecondArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<TimestampMicrosecondArray>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-Timestamp column `{name}`")))
}
fn bool_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| Error::Arrow(format!("missing or non-Bool column `{name}`")))
}

fn marking_label(m: Marking) -> &'static str {
    match m {
        Marking::Public => "Public",
        Marking::Internal => "Internal",
        Marking::Pii => "Pii",
        Marking::Financial => "Financial",
        Marking::Restricted => "Restricted",
    }
}
fn parse_marking_label(s: &str) -> Marking {
    match s {
        "Public" => Marking::Public,
        "Pii" => Marking::Pii,
        "Financial" => Marking::Financial,
        "Restricted" => Marking::Restricted,
        _ => Marking::Internal,
    }
}

fn semantic_type_label(t: &SemanticType) -> String {
    match t {
        SemanticType::PlainText => "PlainText".to_string(),
        SemanticType::Iban => "Iban".to_string(),
        SemanticType::Email => "Email".to_string(),
        SemanticType::Phone => "Phone".to_string(),
        SemanticType::Address => "Address".to_string(),
        SemanticType::Url => "Url".to_string(),
        SemanticType::TaxId => "TaxId".to_string(),
        SemanticType::CustomerId => "CustomerId".to_string(),
        SemanticType::InvoiceNumber => "InvoiceNumber".to_string(),
        SemanticType::Image => "Image".to_string(),
        SemanticType::Currency(code) => format!("Currency({code})"),
        SemanticType::File(mime) => format!("File({mime})"),
        SemanticType::Date(p) => format!("Date({p:?})"),
        SemanticType::Geo(g) => format!("Geo({g:?})"),
    }
}
fn parse_semantic_type_label(s: &str) -> SemanticType {
    match s {
        "PlainText" => SemanticType::PlainText,
        "Iban" => SemanticType::Iban,
        "Email" => SemanticType::Email,
        "Phone" => SemanticType::Phone,
        "Address" => SemanticType::Address,
        "Url" => SemanticType::Url,
        "TaxId" => SemanticType::TaxId,
        "CustomerId" => SemanticType::CustomerId,
        "InvoiceNumber" => SemanticType::InvoiceNumber,
        "Image" => SemanticType::Image,
        // Parametric variants need a static-string parameter we cannot
        // recover from the dictionary; fall through to PlainText.
        _ => SemanticType::PlainText,
    }
}

fn chrono_micros() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}
