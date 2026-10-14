//! Shared harness for the WAL shootout (plan `.claude/plans/wal-shootout-v1.md`):
//! the same logical cycle written by two arms.
//!
//! - **L** — `LanceCycleWriter::commit_cycle`, one Lance append per cycle.
//! - **M** — a lancedb 0.40 table with an `LsmWriteSpec`, written by the upsert
//!   form of `merge_insert`, which lancedb dispatches to the MemWAL `ShardWriter`.
//!
//! M stores the rows `LanceCycleWriter` stores (one frame row, one landing row
//! per cast, one image row per distinct `row` with its 512-byte payload) plus a
//! `pk` column, because MemWAL dedups by an unenforced primary key.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{
    builder::{FixedSizeBinaryBuilder, UInt32Builder, UInt8Builder},
    Array, Int64Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance_graph::graph::cycle_sink::{
    cycle_store_schema, LanceCycleWriter, EPISODIC_WITNESS_BYTES,
};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{CycleFrame, CycleId, DetachedCycleBatch, SweepSlot};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::LsmWriteSpec;

/// The lancedb table name used by every M directory.
pub const M_TABLE: &str = "cycles";

/// Rows one cycle of `n` casts (distinct rows) occupies on either arm:
/// one frame row, `n` landing rows, `n` image rows.
pub fn rows_per_cycle(n: usize) -> usize {
    1 + 2 * n
}

/// `n` artifact casts for `cycle`, one per distinct row, each with a
/// deterministic 512-byte payload. `stream_position` is monotonic across
/// cycles, as `SweepSlot` requires.
pub fn casts(cycle: u64, n: usize) -> Vec<SweepSlot> {
    (0..n)
        .map(|i| {
            let mut payload = vec![0u8; EPISODIC_WITNESS_BYTES];
            payload[..8].copy_from_slice(&cycle.to_le_bytes());
            payload[8..16].copy_from_slice(&(i as u64).to_le_bytes());
            for (j, b) in payload[16..].iter_mut().enumerate() {
                *b = (cycle as usize ^ i ^ j) as u8;
            }
            SweepSlot {
                cycle: CycleId(cycle),
                stream_position: (cycle << 32) | i as u64,
                owner: 1,
                row: i as u64,
                paired_move: None,
                payload,
            }
        })
        .collect()
}

/// The frozen cycle both arms write.
pub fn frozen(cycle: u64, base_version: u64, n: usize) -> DetachedCycleBatch {
    DetachedCycleBatch::freeze(
        CycleFrame::new(CycleId(cycle), DatasetVersion(base_version)),
        casts(cycle, n),
    )
}

// ─────────────────────────────── arm L ───────────────────────────────

/// Commit one cycle of `n` casts on top of the writer's current head.
pub async fn l_commit(w: &mut LanceCycleWriter, cycle: u64, n: usize) {
    use lance_graph_planner::persist_sink::WalSink;
    let batch = frozen(cycle, w.head().0, n);
    w.commit_cycle(batch).await.expect("L commit_cycle");
}

/// Rows per cycle in the L store at its latest version (`None` if the store
/// does not exist yet).
pub async fn l_rows_by_cycle(path: &str) -> Option<BTreeMap<u64, usize>> {
    let ds = match lance::dataset::Dataset::open(path).await {
        Ok(ds) => ds,
        Err(lance::Error::DatasetNotFound { .. }) => return None,
        Err(e) => panic!("open {path}: {e}"),
    };
    let mut scan = ds.scan();
    scan.project(&["cycle"]).expect("project");
    let mut stream = scan.try_into_stream().await.expect("scan");
    let mut out = BTreeMap::new();
    while let Some(b) = stream.try_next().await.expect("scan batch") {
        count_cycles(&b, &mut out);
    }
    Some(out)
}

// ─────────────────────────────── arm M ───────────────────────────────

/// The M schema: the cycle-store schema plus `pk`.
pub fn m_schema() -> SchemaRef {
    let mut fields: Vec<Field> = cycle_store_schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new("pk", DataType::Int64, false));
    Arc::new(Schema::new(fields))
}

/// Create the M table in `dir` with its primary key and an unsharded write
/// spec. `flush_ms` is the WAL flush interval the shard writers will use.
pub async fn m_create(dir: &str, flush_ms: u64) -> lancedb::Table {
    let db = lancedb::connect(dir).execute().await.expect("connect");
    let t = db
        .create_empty_table(M_TABLE, m_schema())
        .execute()
        .await
        .expect("create_empty_table");
    t.set_unenforced_primary_key(["pk"])
        .await
        .expect("primary key");
    let defaults: HashMap<String, String> = [
        ("durable_write".to_string(), "true".to_string()),
        (
            "max_wal_flush_interval_ms".to_string(),
            flush_ms.to_string(),
        ),
    ]
    .into_iter()
    .collect();
    t.set_lsm_write_spec(
        LsmWriteSpec::unsharded()
            .with_maintained_indexes(Vec::new())
            .with_writer_config_defaults(defaults),
    )
    .await
    .expect("set_lsm_write_spec");
    t
}

/// Open the existing M table in `dir`.
pub async fn m_open(dir: &str) -> lancedb::Table {
    let db = lancedb::connect(dir).execute().await.expect("connect");
    db.open_table(M_TABLE).execute().await.expect("open_table")
}

/// The M rows of one frozen cycle, in the layout `LanceCycleWriter` writes.
pub fn m_batch(batch: &DetachedCycleBatch) -> RecordBatch {
    let cycle = batch.frame.cycle.0;
    let base = batch.frame.base_version.0;
    let n = 1 + batch.landings.len() + batch.image.len();
    let mut kind = UInt8Builder::with_capacity(n);
    let mut stream = Vec::with_capacity(n);
    let mut owner = UInt32Builder::with_capacity(n);
    let mut row = Vec::with_capacity(n);
    let mut payload = FixedSizeBinaryBuilder::with_capacity(n, EPISODIC_WITNESS_BYTES as i32);
    // frame
    kind.append_value(0);
    stream.push(0);
    owner.append_value(0);
    row.push(0);
    payload.append_null();
    // landings (metadata only)
    for s in &batch.landings {
        kind.append_value(1);
        stream.push(s.stream_position);
        owner.append_value(s.owner);
        row.push(s.row);
        payload.append_null();
    }
    // image rows (final payload)
    for (r, p) in &batch.image {
        kind.append_value(2);
        stream.push(0);
        owner.append_value(0);
        row.push(*r);
        payload.append_value(p).expect("512-byte payload");
    }
    let nulls_u32 = || Arc::new(UInt32Array::from(vec![None::<u32>; n])) as Arc<dyn Array>;
    let nulls_u8 = || Arc::new(UInt8Array::from(vec![None::<u8>; n])) as Arc<dyn Array>;
    // lancedb accepts Int64 (not UInt64) as a primary key.
    let pk: Vec<i64> = (0..n as u64).map(|i| ((cycle << 24) | i) as i64).collect();
    RecordBatch::try_new(
        m_schema(),
        vec![
            Arc::new(kind.finish()),
            Arc::new(UInt64Array::from(vec![cycle; n])),
            Arc::new(UInt64Array::from(vec![base; n])),
            Arc::new(UInt64Array::from(vec![batch.batch_hash; n])),
            Arc::new(UInt64Array::from(stream)),
            Arc::new(owner.finish()),
            Arc::new(UInt64Array::from(row)),
            nulls_u32(),
            nulls_u8(),
            nulls_u8(),
            nulls_u32(),
            nulls_u8(),
            Arc::new(payload.finish()),
            Arc::new(Int64Array::from(pk)),
        ],
    )
    .expect("M batch")
}

/// Write one frozen cycle through the upsert `merge_insert` (MemWAL path).
pub async fn m_put(t: &lancedb::Table, batch: &DetachedCycleBatch) {
    let rb = m_batch(batch);
    let schema = rb.schema();
    let mut mi = t.merge_insert(&["pk"]);
    mi.when_matched_update_all(None)
        .when_not_matched_insert_all();
    mi.execute(Box::new(RecordBatchIterator::new(vec![Ok(rb)], schema)))
        .await
        .expect("M merge_insert");
}

/// Rows per cycle as an M reader sees them. `lsm = true` routes through the
/// LSM scanner (memtables of writers open in THIS process + flushed
/// generations); `false` reads the base table only.
pub async fn m_rows_by_cycle(t: &lancedb::Table, lsm: bool) -> BTreeMap<u64, usize> {
    let mut stream = t
        .query()
        .select(Select::columns(&["cycle"]))
        .use_lsm(lsm)
        .execute()
        .await
        .expect("M query");
    let mut out = BTreeMap::new();
    while let Some(b) = stream.try_next().await.expect("M batch") {
        count_cycles(&b, &mut out);
    }
    out
}

fn count_cycles(b: &RecordBatch, out: &mut BTreeMap<u64, usize>) {
    let c: &UInt64Array = b
        .column_by_name("cycle")
        .and_then(|c| c.as_any().downcast_ref())
        .expect("cycle column");
    for i in 0..c.len() {
        *out.entry(c.value(i)).or_default() += 1;
    }
}

/// Bytes and files under `dir`, recursively.
pub fn disk_usage(dir: &std::path::Path) -> (u64, u64) {
    let mut bytes = 0;
    let mut files = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let (b, f) = disk_usage(&p);
                bytes += b;
                files += f;
            } else if let Ok(m) = e.metadata() {
                bytes += m.len();
                files += 1;
            }
        }
    }
    (bytes, files)
}
