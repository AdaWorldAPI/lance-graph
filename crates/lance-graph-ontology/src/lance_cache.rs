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
use crate::proposal::{AttributeProvenance, IdentityCodec, MappingRow, QualiaMeta};
use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, FixedSizeBinaryBuilder, FixedSizeListArray,
    FixedSizeListBuilder, Float32Array, Float32Builder, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_graph_contract::property::{Marking, SemanticType};
use lance_graph_contract::thinking::ThinkingStyle;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DICTIONARY_NAME: &str = "ontology_dictionary";
const META_NAME: &str = "ontology_meta";

// Why this exists (read before proposing a migration path):
//
// `ontology_dictionary` is a CACHE of hydrated TTL, keyed in the meta table
// by `ttl_root_checksum`. The TTL files on disk are the source of truth;
// this Lance dataset is a fast-path projection so hydration doesn't re-parse
// on every boot. BindSpace (FingerprintColumns / QualiaColumn / MetaColumn /
// EdgeColumn) is the live runtime SoA and is unrelated — it never lands here.
//
// Because we're cache, not source-of-truth, schema evolution does NOT need
// a per-version migration ladder. On version mismatch we invalidate (delete
// the cache directory) and let hydration re-derive from TTL. That eliminates
// a class of "silent default-fill smuggles synthesized zeros into the
// codebook" bugs at the cost of one cold rebuild on the first boot after a
// version bump. Cold rebuild is acceptable; codebook contamination is not.
//
// "Unknown" version (newer than this binary expects, e.g. a feature branch
// wrote v3 columns we don't know about) is also invalidated — forward-incompat
// datasets get a clean rebuild rather than corrupting the running binary's
// view of the codebook.
//
// **Rule for the next editor:** if you change `dictionary_schema()` in any
// way (add / remove / rename / retype a column), bump `SCHEMA_VERSION` in
// the same commit. The `schema_version_pinned` unit test fails loudly
// otherwise — that's the compile-adjacent guard. The runtime guard is
// `LanceWriter::open_or_create`, which checks the on-disk version against
// this constant and invalidates on any mismatch.
pub const SCHEMA_VERSION: u32 = 2;

pub struct LanceWriter {
    base: PathBuf,
}

impl LanceWriter {
    pub async fn open_or_create(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let writer = Self {
            base: path.to_path_buf(),
        };
        writer.invalidate_if_stale_schema().await?;
        Ok(writer)
    }

    // Read the persisted `schema_version` from the meta table. Returns:
    //   Ok(Some(n)) — meta exists and the column was readable
    //   Ok(None)    — meta dir is absent (fresh install) OR the column is
    //                 missing / unreadable (pre-versioning v1 deployment,
    //                 or a corrupted meta file — both treated as "stale,
    //                 invalidate" by the caller)
    async fn read_schema_version(&self) -> Result<Option<u32>> {
        let path = self.meta_path();
        if !path.exists() {
            return Ok(None);
        }
        let path_str = path.to_string_lossy().to_string();
        let dataset = match Dataset::open(&path_str).await {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        let mut stream = match dataset.scan().try_into_stream().await {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        use futures::StreamExt;
        if let Some(batch) = stream.next().await {
            let Ok(batch) = batch else { return Ok(None) };
            let Some(col) = batch.column_by_name("schema_version") else {
                return Ok(None);
            };
            let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() else {
                return Ok(None);
            };
            if arr.len() > 0 {
                return Ok(Some(arr.value(0)));
            }
        }
        Ok(None)
    }

    // On version mismatch, drop the cache so the next hydration rebuilds
    // from TTL. See the module-level reasoning comment above `SCHEMA_VERSION`
    // for why we invalidate instead of migrating.
    async fn invalidate_if_stale_schema(&self) -> Result<()> {
        let on_disk = self.read_schema_version().await?;
        if on_disk == Some(SCHEMA_VERSION) {
            return Ok(());
        }
        for sub in [self.dictionary_path(), self.meta_path()] {
            if sub.exists() {
                std::fs::remove_dir_all(&sub).map_err(|source| Error::Io {
                    path: sub.clone(),
                    source,
                })?;
            }
        }
        Ok(())
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
        let reader =
            arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
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
        // `schema_version` is the cache-coherence handshake — read on open
        // by `invalidate_if_stale_schema` to decide whether the on-disk
        // dictionary is still meaningful to this binary.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("ttl_root_checksum", DataType::Utf8, false),
            Field::new(
                "last_hydrated_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("crate_version", DataType::Utf8, false),
            Field::new("schema_version", DataType::UInt32, false),
        ]));
        let now = chrono_micros();
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![checksum])),
            Arc::new(TimestampMicrosecondArray::from(vec![now])),
            Arc::new(StringArray::from(vec![env!("CARGO_PKG_VERSION")])),
            Arc::new(UInt32Array::from(vec![SCHEMA_VERSION])),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| Error::Arrow(format!("meta batch: {e}")))?;
        let path = self.meta_path();
        let path_str = path.to_string_lossy().to_string();
        // Meta is a single-row table — overwrite.
        let reader =
            arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
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
        // ── legacy columns (schema v1) ──────────────────────────────────────
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
        // ── D-CASCADE-V1-7 columns (schema v2) ─────────────────────────────
        // IdentityCodec — CAM-PQ hot-path bundle
        Field::new("cam_pq_code", DataType::FixedSizeBinary(6), false),
        Field::new("base17_head", DataType::FixedSizeBinary(8), false),
        Field::new("palette_key", DataType::UInt32, false),
        Field::new("scent", DataType::UInt8, false),
        // QualiaMeta — Pillar-0 dispatch bundle.
        // Item nullability mirrors what `FixedSizeListBuilder<Float32Builder>`
        // produces by default (nullable items). We never actually write nulls,
        // but the schema has to agree with the builder for `RecordBatch::try_new`
        // to accept the column. The outer list field stays non-null.
        Field::new(
            "qualia",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                18,
            ),
            false,
        ),
        Field::new("codec_meta", DataType::UInt32, false),
        Field::new("codec_edge", DataType::UInt64, false),
        // ThinkingStyle (nullable: None → empty string on disk)
        Field::new("thinking_style", DataType::Utf8, true),
        // AttributeProvenance list encoded as `predicate\x1fsource_uri` pairs
        // joined by `\x1e` (ASCII Record Separator / Unit Separator). Empty
        // string means no sources. Kept as plain Utf8 to avoid nested-list
        // Lance encoding overhead for what is typically a short list.
        Field::new("attribute_sources_enc", DataType::Utf8, false),
        // Edge-/attribute-only type-ref strings
        Field::new("subject_type", DataType::Utf8, false),
        Field::new("object_type", DataType::Utf8, false),
        Field::new("entity_type_ref", DataType::Utf8, false),
    ]))
}

fn rows_to_record_batch(rows: &[MappingRow]) -> Result<RecordBatch> {
    // ── legacy columns ──────────────────────────────────────────────────────
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

    // ── D-CASCADE-V1-7: IdentityCodec ──────────────────────────────────────
    let mut cam_pq_code_builder = FixedSizeBinaryBuilder::new(6);
    let mut base17_head_builder = FixedSizeBinaryBuilder::new(8);
    let palette_key: Vec<u32> = rows
        .iter()
        .map(|r| r.identity_codec.palette_key)
        .collect();
    let scent: Vec<u8> = rows.iter().map(|r| r.identity_codec.scent).collect();
    for r in rows {
        cam_pq_code_builder
            .append_value(r.identity_codec.cam_pq_code)
            .map_err(|e| Error::Arrow(format!("cam_pq_code: {e}")))?;
        base17_head_builder
            .append_value(r.identity_codec.base17_head)
            .map_err(|e| Error::Arrow(format!("base17_head: {e}")))?;
    }

    // ── D-CASCADE-V1-7: QualiaMeta ──────────────────────────────────────────
    // qualia: FixedSizeList<Float32, 18>
    let mut qualia_builder = FixedSizeListBuilder::new(Float32Builder::new(), 18);
    for r in rows {
        for &v in &r.qualia_meta.qualia {
            qualia_builder.values().append_value(v);
        }
        qualia_builder.append(true);
    }
    let codec_meta: Vec<u32> = rows.iter().map(|r| r.qualia_meta.meta).collect();
    let codec_edge: Vec<u64> = rows.iter().map(|r| r.qualia_meta.edge).collect();

    // ── D-CASCADE-V1-7: ThinkingStyle, AttributeProvenance, type-refs ───────
    let thinking_style: Vec<Option<&str>> = rows
        .iter()
        .map(|r| r.thinking_style.as_ref().map(thinking_style_label))
        .collect();
    let attribute_sources_enc: Vec<String> = rows
        .iter()
        .map(|r| encode_attribute_sources(&r.attribute_sources))
        .collect();
    let subject_type: Vec<&str> = rows.iter().map(|r| r.subject_type.as_str()).collect();
    let object_type: Vec<&str> = rows.iter().map(|r| r.object_type.as_str()).collect();
    let entity_type_ref: Vec<&str> = rows.iter().map(|r| r.entity_type_ref.as_str()).collect();

    let qualia_arr = qualia_builder.finish();

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
        // v2 cascade columns
        Arc::new(cam_pq_code_builder.finish()),
        Arc::new(base17_head_builder.finish()),
        Arc::new(UInt32Array::from(palette_key)),
        Arc::new(UInt8Array::from(scent)),
        Arc::new(qualia_arr),
        Arc::new(UInt32Array::from(codec_meta)),
        Arc::new(UInt64Array::from(codec_edge)),
        Arc::new(StringArray::from(thinking_style)),
        Arc::new(StringArray::from(attribute_sources_enc)),
        Arc::new(StringArray::from(subject_type)),
        Arc::new(StringArray::from(object_type)),
        Arc::new(StringArray::from(entity_type_ref)),
    ];
    RecordBatch::try_new(dictionary_schema(), cols).map_err(|e| Error::Arrow(format!("{e}")))
}

fn record_batch_to_rows(batch: &RecordBatch) -> Result<Vec<MappingRow>> {
    // ── legacy columns (always present) ─────────────────────────────────────
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

    // ── D-CASCADE-V1-7 columns (optional for backward compat) ───────────────
    // Older cache files written with schema v1 will be missing these columns.
    // Backward-compat policy: lossy-allow — missing columns default to the
    // same values that `MappingRow::default()` / the old reader supplied.
    let cam_pq_code_arr = fsb_col_opt(batch, "cam_pq_code");
    let base17_head_arr = fsb_col_opt(batch, "base17_head");
    let palette_key_arr = u32_col_opt(batch, "palette_key");
    let scent_arr = u8_col_opt(batch, "scent");
    let qualia_arr = fsl_f32_col_opt(batch, "qualia");
    let codec_meta_arr = u32_col_opt(batch, "codec_meta");
    let codec_edge_arr = u64_col_opt(batch, "codec_edge");
    let thinking_style_arr = string_col_opt(batch, "thinking_style");
    let attr_src_enc_arr = string_col_opt(batch, "attribute_sources_enc");
    let subject_type_arr = string_col_opt(batch, "subject_type");
    let object_type_arr = string_col_opt(batch, "object_type");
    let entity_type_ref_arr = string_col_opt(batch, "entity_type_ref");

    let mut rows = Vec::with_capacity(bridge_id.len());
    for i in 0..bridge_id.len() {
        let identity_codec = IdentityCodec {
            cam_pq_code: cam_pq_code_arr
                .and_then(|a| a.value(i).try_into().ok())
                .unwrap_or([0u8; 6]),
            base17_head: base17_head_arr
                .and_then(|a| a.value(i).try_into().ok())
                .unwrap_or([0u8; 8]),
            palette_key: palette_key_arr.map(|a| a.value(i)).unwrap_or(0),
            scent: scent_arr.map(|a| a.value(i)).unwrap_or(0),
        };
        let qualia_meta = QualiaMeta {
            qualia: qualia_arr
                .map(|a| {
                    let list = a.value(i);
                    let f32s = list
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("qualia inner type is Float32");
                    let mut arr = [0f32; 18];
                    for (slot, &v) in arr.iter_mut().zip(f32s.values()) {
                        *slot = v;
                    }
                    arr
                })
                .unwrap_or([0f32; 18]),
            meta: codec_meta_arr.map(|a| a.value(i)).unwrap_or(0),
            edge: codec_edge_arr.map(|a| a.value(i)).unwrap_or(0),
        };
        let thinking_style = thinking_style_arr
            .and_then(|a| {
                if a.is_null(i) || a.value(i).is_empty() {
                    None
                } else {
                    parse_thinking_style_label(a.value(i))
                }
            });
        let attribute_sources = attr_src_enc_arr
            .map(|a| decode_attribute_sources(a.value(i)))
            .unwrap_or_default();
        let subject_type = subject_type_arr
            .map(|a| a.value(i).to_string())
            .unwrap_or_default();
        let object_type = object_type_arr
            .map(|a| a.value(i).to_string())
            .unwrap_or_default();
        let entity_type_ref = entity_type_ref_arr
            .map(|a| a.value(i).to_string())
            .unwrap_or_default();

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
            identity_codec,
            qualia_meta,
            thinking_style,
            attribute_sources,
            subject_type,
            object_type,
            entity_type_ref,
        });
    }
    Ok(rows)
}

// ── required column accessors (error on missing) ───────────────────────────

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

// ── optional column accessors (None on missing — backward compat) ───────────

fn string_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
}
fn u8_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt8Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
}
fn u32_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
}
fn u64_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt64Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
}
fn fsb_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a FixedSizeBinaryArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
}
fn fsl_f32_col_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a FixedSizeListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
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

// ── ThinkingStyle label round-trip ──────────────────────────────────────────

fn thinking_style_label(ts: &ThinkingStyle) -> &'static str {
    match ts {
        ThinkingStyle::Logical => "Logical",
        ThinkingStyle::Analytical => "Analytical",
        ThinkingStyle::Critical => "Critical",
        ThinkingStyle::Systematic => "Systematic",
        ThinkingStyle::Methodical => "Methodical",
        ThinkingStyle::Precise => "Precise",
        ThinkingStyle::Creative => "Creative",
        ThinkingStyle::Imaginative => "Imaginative",
        ThinkingStyle::Innovative => "Innovative",
        ThinkingStyle::Artistic => "Artistic",
        ThinkingStyle::Poetic => "Poetic",
        ThinkingStyle::Playful => "Playful",
        ThinkingStyle::Empathetic => "Empathetic",
        ThinkingStyle::Compassionate => "Compassionate",
        ThinkingStyle::Supportive => "Supportive",
        ThinkingStyle::Nurturing => "Nurturing",
        ThinkingStyle::Gentle => "Gentle",
        ThinkingStyle::Warm => "Warm",
        ThinkingStyle::Direct => "Direct",
        ThinkingStyle::Concise => "Concise",
        ThinkingStyle::Efficient => "Efficient",
        ThinkingStyle::Pragmatic => "Pragmatic",
        ThinkingStyle::Blunt => "Blunt",
        ThinkingStyle::Frank => "Frank",
        ThinkingStyle::Curious => "Curious",
        ThinkingStyle::Exploratory => "Exploratory",
        ThinkingStyle::Questioning => "Questioning",
        ThinkingStyle::Investigative => "Investigative",
        ThinkingStyle::Speculative => "Speculative",
        ThinkingStyle::Philosophical => "Philosophical",
        ThinkingStyle::Reflective => "Reflective",
        ThinkingStyle::Contemplative => "Contemplative",
        ThinkingStyle::Metacognitive => "Metacognitive",
        ThinkingStyle::Wise => "Wise",
        ThinkingStyle::Transcendent => "Transcendent",
        ThinkingStyle::Sovereign => "Sovereign",
    }
}

fn parse_thinking_style_label(s: &str) -> Option<ThinkingStyle> {
    match s {
        "Logical" => Some(ThinkingStyle::Logical),
        "Analytical" => Some(ThinkingStyle::Analytical),
        "Critical" => Some(ThinkingStyle::Critical),
        "Systematic" => Some(ThinkingStyle::Systematic),
        "Methodical" => Some(ThinkingStyle::Methodical),
        "Precise" => Some(ThinkingStyle::Precise),
        "Creative" => Some(ThinkingStyle::Creative),
        "Imaginative" => Some(ThinkingStyle::Imaginative),
        "Innovative" => Some(ThinkingStyle::Innovative),
        "Artistic" => Some(ThinkingStyle::Artistic),
        "Poetic" => Some(ThinkingStyle::Poetic),
        "Playful" => Some(ThinkingStyle::Playful),
        "Empathetic" => Some(ThinkingStyle::Empathetic),
        "Compassionate" => Some(ThinkingStyle::Compassionate),
        "Supportive" => Some(ThinkingStyle::Supportive),
        "Nurturing" => Some(ThinkingStyle::Nurturing),
        "Gentle" => Some(ThinkingStyle::Gentle),
        "Warm" => Some(ThinkingStyle::Warm),
        "Direct" => Some(ThinkingStyle::Direct),
        "Concise" => Some(ThinkingStyle::Concise),
        "Efficient" => Some(ThinkingStyle::Efficient),
        "Pragmatic" => Some(ThinkingStyle::Pragmatic),
        "Blunt" => Some(ThinkingStyle::Blunt),
        "Frank" => Some(ThinkingStyle::Frank),
        "Curious" => Some(ThinkingStyle::Curious),
        "Exploratory" => Some(ThinkingStyle::Exploratory),
        "Questioning" => Some(ThinkingStyle::Questioning),
        "Investigative" => Some(ThinkingStyle::Investigative),
        "Speculative" => Some(ThinkingStyle::Speculative),
        "Philosophical" => Some(ThinkingStyle::Philosophical),
        "Reflective" => Some(ThinkingStyle::Reflective),
        "Contemplative" => Some(ThinkingStyle::Contemplative),
        "Metacognitive" => Some(ThinkingStyle::Metacognitive),
        "Wise" => Some(ThinkingStyle::Wise),
        "Transcendent" => Some(ThinkingStyle::Transcendent),
        "Sovereign" => Some(ThinkingStyle::Sovereign),
        _ => None,
    }
}

// ── AttributeProvenance encode/decode ───────────────────────────────────────
// Wire format: pairs of `predicate_iri\x1fsource_uri` joined by `\x1e`.
// ASCII Unit Separator (0x1F) splits each pair; ASCII Record Separator (0x1E)
// splits pairs from each other. Empty string → no sources.

const PAIR_SEP: char = '\x1e';
const FIELD_SEP: char = '\x1f';

fn encode_attribute_sources(sources: &[AttributeProvenance]) -> String {
    if sources.is_empty() {
        return String::new();
    }
    sources
        .iter()
        .map(|ap| format!("{}{FIELD_SEP}{}", ap.predicate_iri, ap.source_uri))
        .collect::<Vec<_>>()
        .join(&PAIR_SEP.to_string())
}

fn decode_attribute_sources(encoded: &str) -> Vec<AttributeProvenance> {
    if encoded.is_empty() {
        return Vec::new();
    }
    encoded
        .split(PAIR_SEP)
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, FIELD_SEP);
            let predicate_iri = parts.next()?.to_string();
            let source_uri = parts.next()?.to_string();
            Some(AttributeProvenance {
                predicate_iri,
                source_uri,
            })
        })
        .collect()
}

// ── Round-trip test ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::{NamespaceId, OgitUri, SchemaKind, SchemaPtr};
    use crate::proposal::{AttributeProvenance, IdentityCodec, MappingRow, QualiaMeta};
    use lance_graph_contract::property::{Marking, SemanticType};
    use lance_graph_contract::thinking::ThinkingStyle;

    /// Build a `MappingRow` with non-default values for every D-CASCADE-V1-7
    /// field, write it to an in-memory `RecordBatch`, read it back, and assert
    /// field-by-field equality for all 10+ new columns.
    #[test]
    fn cascade_cols_round_trip_record_batch() {
        let row = MappingRow {
            bridge_id: "woa".to_string(),
            public_name: "Customer".to_string(),
            ogit_uri: OgitUri::from_string_unchecked("ogit.WorkOrder:Customer"),
            namespace_id: NamespaceId(3),
            schema_ptr: SchemaPtr::from_raw(42),
            kind: SchemaKind::Entity,
            semantic_type: SemanticType::PlainText,
            marking: Marking::Internal,
            confidence: 0.95,
            created_at_us: 1_700_000_000_000_000,
            created_by: "ogit_hydrator_v1".to_string(),
            source_uri: "https://example.com/woa.ttl".to_string(),
            active: true,
            checksum: "abc123".to_string(),
            // D-CASCADE-V1-7 fields — all non-default
            identity_codec: IdentityCodec {
                cam_pq_code: [0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x02],
                base17_head: [0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05, 0x06],
                palette_key: 12345,
                scent: 7,
            },
            qualia_meta: QualiaMeta {
                qualia: [
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8,
                ],
                meta: 0xDEAD_BEEF,
                edge: 0x0102_0304_0506_0708,
            },
            thinking_style: Some(ThinkingStyle::Investigative),
            attribute_sources: vec![
                AttributeProvenance {
                    predicate_iri: "ogit.WorkOrder:fahrtKm".to_string(),
                    source_uri: "AdaWorldAPI/WoA/models.py:Customer.fahrt_km".to_string(),
                },
                AttributeProvenance {
                    predicate_iri: "ogit.WorkOrder:status".to_string(),
                    source_uri: "AdaWorldAPI/WoA/models.py:Customer.status".to_string(),
                },
            ],
            subject_type: "Employee".to_string(),
            object_type: "WorkOrder".to_string(),
            entity_type_ref: "Customer".to_string(),
        };

        let batch = rows_to_record_batch(std::slice::from_ref(&row))
            .expect("rows_to_record_batch must not fail");
        let mut back = record_batch_to_rows(&batch).expect("record_batch_to_rows must not fail");
        assert_eq!(back.len(), 1, "expected 1 row back");
        let r = back.remove(0);

        // Legacy fields
        assert_eq!(r.bridge_id, row.bridge_id);
        assert_eq!(r.checksum, row.checksum);
        assert_eq!(r.confidence, row.confidence);

        // IdentityCodec
        assert_eq!(
            r.identity_codec.cam_pq_code,
            row.identity_codec.cam_pq_code,
            "cam_pq_code mismatch"
        );
        assert_eq!(
            r.identity_codec.base17_head,
            row.identity_codec.base17_head,
            "base17_head mismatch"
        );
        assert_eq!(
            r.identity_codec.palette_key,
            row.identity_codec.palette_key,
            "palette_key mismatch"
        );
        assert_eq!(
            r.identity_codec.scent,
            row.identity_codec.scent,
            "scent mismatch"
        );

        // QualiaMeta
        assert_eq!(
            r.qualia_meta.qualia,
            row.qualia_meta.qualia,
            "qualia mismatch"
        );
        assert_eq!(r.qualia_meta.meta, row.qualia_meta.meta, "codec_meta mismatch");
        assert_eq!(r.qualia_meta.edge, row.qualia_meta.edge, "codec_edge mismatch");

        // ThinkingStyle
        assert_eq!(
            r.thinking_style,
            row.thinking_style,
            "thinking_style mismatch"
        );

        // AttributeProvenance
        assert_eq!(
            r.attribute_sources,
            row.attribute_sources,
            "attribute_sources mismatch"
        );

        // Type-ref strings
        assert_eq!(r.subject_type, row.subject_type, "subject_type mismatch");
        assert_eq!(r.object_type, row.object_type, "object_type mismatch");
        assert_eq!(
            r.entity_type_ref,
            row.entity_type_ref,
            "entity_type_ref mismatch"
        );
    }

    /// Verify that `thinking_style = None` round-trips correctly (null column).
    #[test]
    fn cascade_cols_thinking_style_none_round_trip() {
        let mut row = MappingRow {
            bridge_id: "ogit".to_string(),
            public_name: "IPAddress".to_string(),
            ogit_uri: OgitUri::from_string_unchecked("ogit.Network:IPAddress"),
            namespace_id: NamespaceId(1),
            schema_ptr: SchemaPtr::from_raw(1),
            kind: SchemaKind::Entity,
            semantic_type: SemanticType::PlainText,
            marking: Marking::Public,
            confidence: 1.0,
            created_at_us: 0,
            created_by: "test".to_string(),
            source_uri: String::new(),
            active: true,
            checksum: "x".to_string(),
            identity_codec: IdentityCodec::default(),
            qualia_meta: QualiaMeta::default(),
            thinking_style: None,
            attribute_sources: Vec::new(),
            subject_type: String::new(),
            object_type: String::new(),
            entity_type_ref: String::new(),
        };
        // Suppress unused-mut warning — field needed by struct initialiser pattern.
        let _ = &mut row;

        let batch = rows_to_record_batch(std::slice::from_ref(&row))
            .expect("rows_to_record_batch must not fail");
        let mut back = record_batch_to_rows(&batch).expect("record_batch_to_rows must not fail");
        let r = back.remove(0);
        assert_eq!(r.thinking_style, None, "None thinking_style must survive round-trip");
        assert!(r.attribute_sources.is_empty(), "empty attribute_sources must survive round-trip");
    }

    // Pins the schema field-set against `SCHEMA_VERSION`. If you change
    // `dictionary_schema()` without bumping `SCHEMA_VERSION`, this test
    // fails — that's the compile-adjacent guard for the cache-coherence
    // contract. To fix: bump `SCHEMA_VERSION` in lance_cache.rs and update
    // the `expected` list below with the new field set (printed on failure).
    #[test]
    fn schema_version_pinned() {
        let schema = dictionary_schema();
        let actual: Vec<(String, String, bool)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), format!("{:?}", f.data_type()), f.is_nullable()))
            .collect();
        // Pinned to SCHEMA_VERSION = 2.
        let expected: Vec<(&str, &str, bool)> = vec![
            ("bridge_id", "Utf8", false),
            ("public_name", "Utf8", false),
            ("ogit_uri", "Utf8", false),
            ("namespace_id", "UInt8", false),
            ("schema_ptr", "UInt32", false),
            ("kind", "Utf8", false),
            ("semantic_type", "Utf8", false),
            ("marking", "Utf8", false),
            ("confidence", "Float32", false),
            ("created_at", "Timestamp(Microsecond, None)", false),
            ("created_by", "Utf8", false),
            ("source_uri", "Utf8", false),
            ("active", "Boolean", false),
            ("checksum", "Utf8", false),
            ("cam_pq_code", "FixedSizeBinary(6)", false),
            ("base17_head", "FixedSizeBinary(8)", false),
            ("palette_key", "UInt32", false),
            ("scent", "UInt8", false),
            // qualia data_type debug format depends on arrow internals; the
            // round-trip tests catch any drift in item nullability, so here
            // we only assert the column name and outer nullability.
            ("qualia", "__skip__", false),
            ("codec_meta", "UInt32", false),
            ("codec_edge", "UInt64", false),
            ("thinking_style", "Utf8", true),
            ("attribute_sources_enc", "Utf8", false),
            ("subject_type", "Utf8", false),
            ("object_type", "Utf8", false),
            ("entity_type_ref", "Utf8", false),
        ];
        assert_eq!(
            actual.len(),
            expected.len(),
            "column count drifted from SCHEMA_VERSION = {SCHEMA_VERSION}; bump the constant and update this pin. actual = {actual:#?}",
        );
        for (i, ((a_name, a_type, a_null), (e_name, e_type, e_null))) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(a_name.as_str(), *e_name, "column {i} name drifted");
            assert_eq!(
                *a_null, *e_null,
                "column {i} ({e_name}) outer-nullability drifted from SCHEMA_VERSION = {SCHEMA_VERSION}",
            );
            if *e_type != "__skip__" {
                assert_eq!(
                    a_type.as_str(),
                    *e_type,
                    "column {i} ({e_name}) type drifted from SCHEMA_VERSION = {SCHEMA_VERSION}; bump the constant and update this pin",
                );
            }
        }
    }

    // Runtime guard test: a meta table written by a binary that did NOT
    // know about `schema_version` (the v1 pre-versioning shape) must cause
    // `open_or_create` to wipe the cache directory so hydration rebuilds
    // from TTL. Same path covers "future v3 wrote columns we don't know".
    #[tokio::test]
    async fn stale_meta_invalidates_cache_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "lance_cache_invalidate_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let writer = LanceWriter::open_or_create(&tmp).await.unwrap();

        // Plant a fake v1-shaped meta (no schema_version column) and a
        // dictionary dir; opening again must remove both.
        let v1_meta_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("ttl_root_checksum", DataType::Utf8, false),
            Field::new(
                "last_hydrated_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("crate_version", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            v1_meta_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["pretend_v1_checksum"])),
                Arc::new(TimestampMicrosecondArray::from(vec![0i64])),
                Arc::new(StringArray::from(vec!["0.0.0"])),
            ],
        )
        .unwrap();
        let reader = arrow::record_batch::RecordBatchIterator::new(
            vec![Ok(batch)].into_iter(),
            v1_meta_schema,
        );
        Dataset::write(
            reader,
            writer.meta_path().to_string_lossy().as_ref(),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        std::fs::create_dir_all(writer.dictionary_path()).unwrap();
        std::fs::write(writer.dictionary_path().join("sentinel"), b"x").unwrap();

        // Re-open: the stale meta (no schema_version) must trigger
        // invalidation of both dictionary and meta directories.
        let _writer2 = LanceWriter::open_or_create(&tmp).await.unwrap();
        assert!(
            !writer.dictionary_path().exists(),
            "stale schema must wipe dictionary_path"
        );
        assert!(
            !writer.meta_path().exists(),
            "stale schema must wipe meta_path"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
