// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The CONCRETE cognitive-cycle Lance sink — the storage-proven implementation of
//! `lance_graph_planner::persist_sink::WalSink` over the official Lance
//! transaction/write path (`lance = 9.0.0`).
//!
//! `persist_sink` (the planner seam) deliberately builds NO concrete sink: its
//! `FakeWalSink` proves the algebra (fence, ordering, coalescing, recovery) in
//! process memory only — "compile+test green ≠ storage proven" (the Ladybug
//! lesson). This module closes that gap: every trait operation here is true
//! against a REOPENED dataset on real storage.
//!
//! # The §I.6 invariant (the ruled contract this sink makes physical)
//!
//! ```text
//! 64k thoughts read sealed Vn → write-side temporal deinterlace →
//! one detached cycle batch → ONE official Lance commit →
//! exactly one real DatasetVersion Vn+1 → no open or partial visibility
//! ```
//!
//! - **One durable append per cycle.** `commit_cycle` performs a single
//!   `Dataset::write` / `Dataset::append` (the official Lance insert path —
//!   `InsertBuilder` under the hood, the same transaction machinery every Lance
//!   writer uses). No bespoke ledger, no acknowledgement protocol, no parallel
//!   replay system: Lance's own manifest/version chain IS the WAL.
//! - **The epistemic fence, both halves.** Pre-commit: the dataset's current
//!   version must equal the cycle's sealed predecessor `base` (`Vn`), else the
//!   commit is refused with nothing written. Post-commit: the published version
//!   must be exactly `base + 1`. Lance's optimistic concurrency auto-resolves
//!   append-append conflicts (a foreign interleaved writer would yield
//!   `base + 2`), so under the one-writer-per-mailbox doctrine a post-check
//!   mismatch is a LOUD timeline anomaly, never silently accepted.
//! - **Order is a write-side property.** Landings arrive already deinterlaced
//!   (`DetachedCycleBatch::freeze` ran the loom); they are stored in that
//!   canonical order and scanned back with Lance's in-order scan. This sink
//!   never sorts on read.
//! - **All-or-nothing visibility.** An unsealed / fenced / failed cycle leaves
//!   no rows and no version: after restart + reopen it is simply absent.
//!   Recovery is a read of the sealed store (`scan_sealed` + the caller's
//!   watermark in `recover_and_apply`) — idempotent without any sidecar state.
//!
//! # Domain 0x09 — the patient SoA witness store (why the schema is rich)
//!
//! The patient SoA at classid domain `0x09` is the ONLY place patient reasoning
//! is ever written to Lance. Everything else the reasoner touches — the
//! interlocked ontologies at domain `0x03`, crosswalks, RO edges — is IMMUTABLE
//! for the duration of a representation window: a cycle takes that immutability
//! for granted (its `base_version` names the sealed ontology-bearing predecessor
//! it read), and therefore never needs to restate ontology content. What it MUST
//! state — maximally richly — is the WITNESSING:
//!
//! - **`payload`** carries the canonical witness node bytes (the 512-byte
//!   `key(16) | edges(16) | value(480)` node ABI): the EpisodicWitness row —
//!   visited ontology addresses, executed crosswalk mappings, the exact RO /
//!   ontology edge identifiers walked, supporting / contradicting / missing
//!   observations, NARS truth + confidence, differential branches. Domain-0x09
//!   keys, edges pointing INTO the immutable 0x03 address space.
//! - **The landing columns** carry the dynamic-reasoning-update record: which
//!   mailbox reasoned (`owner`), where in the canonical thought stream
//!   (`stream_position`), which SoA row the update lands on (`row`), and the
//!   Rubicon lifecycle step the thought cast (`move_*` — the sealed reflection
//!   of the thinking, applied post-SEAL only).
//! - **The frame row** (one per cycle, `kind = 0`) seals the cycle ↔ version
//!   mapping INSIDE the same atomic commit, so the coarse timeline
//!   (`versions()`) survives restart with zero sidecar files — the sealed
//!   versioning is literally a reflection of the thinking that produced it.
//!
//! Downstream (the Gotham display, differential views, any consumer) reads the
//! sealed version — never a live recomputation: the witness is examined in
//! place, at the version its cycle published.

use std::sync::Arc;

use arrow_array::{
    builder::{BinaryBuilder, UInt32Builder, UInt8Builder},
    Array, BinaryArray, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    CycleId, DetachedCycleBatch, LandedSlot, SweepSlot, WalSink, WriteFailed,
};

/// Row kind discriminant: the per-cycle frame row (cycle ↔ version mapping,
/// sealed inside the same atomic commit as its landings).
const KIND_FRAME: u8 = 0;
/// Row kind discriminant: one landing (a thought's persistence record).
const KIND_LANDING: u8 = 1;

/// The Arrow schema of the cycle store — one dataset, two row kinds.
///
/// | column                        | type            | frame row | landing row |
/// |-------------------------------|-----------------|-----------|-------------|
/// | `kind`                        | `UInt8`         | 0         | 1           |
/// | `cycle`                       | `UInt64`        | cycle id  | cycle id    |
/// | `base_version`                | `UInt64`        | `Vn`      | `Vn`        |
/// | `stream_position`             | `UInt64`        | 0         | canonical order key |
/// | `owner`                       | `UInt32`        | 0         | mailbox     |
/// | `row`                         | `UInt64`        | 0         | SoA row     |
/// | `move_mailbox`                | `UInt32?`       | null      | paired move (or null) |
/// | `move_from` / `move_to`       | `UInt8?`        | null      | Rubicon edge |
/// | `move_witness_chain_position` | `UInt32?`       | null      | witness pointer (R4) |
/// | `move_exec`                   | `UInt8?`        | null      | exec target |
/// | `payload`                     | `Binary`        | empty     | witness node bytes |
///
/// The sealed version of every row's cycle is `base_version + 1` — an identity
/// the commit path VERIFIES against the real published Lance version (it is
/// never assumed), so reads may derive it without a sidecar mapping.
pub fn cycle_store_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("kind", DataType::UInt8, false),
        Field::new("cycle", DataType::UInt64, false),
        Field::new("base_version", DataType::UInt64, false),
        Field::new("stream_position", DataType::UInt64, false),
        Field::new("owner", DataType::UInt32, false),
        Field::new("row", DataType::UInt64, false),
        Field::new("move_mailbox", DataType::UInt32, true),
        Field::new("move_from", DataType::UInt8, true),
        Field::new("move_to", DataType::UInt8, true),
        Field::new("move_witness_chain_position", DataType::UInt32, true),
        Field::new("move_exec", DataType::UInt8, true),
        Field::new("payload", DataType::Binary, false),
    ]))
}

/// The concrete Lance-backed cycle sink.
///
/// Cheap to clone / recreate: it holds only the dataset path and opens the
/// dataset per operation (the restart-survival guarantee is thereby exercised on
/// EVERY call, not just in tests). Point it at the domain-0x09 patient witness
/// store (e.g. `<base>/witness_cycles.lance`) — one sink instance per store.
#[derive(Debug, Clone)]
pub struct LanceCycleSink {
    dataset_path: String,
}

impl LanceCycleSink {
    /// A sink over the Lance dataset at `path` (local path or object-store URI —
    /// anything `Dataset::open` accepts). The dataset is created on the first
    /// committed cycle; a missing dataset is simply "nothing sealed yet".
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            dataset_path: path.into(),
        }
    }

    /// The dataset path this sink commits to.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }

    /// Open the store if it exists; `None` = nothing sealed yet (a state, not an
    /// error — distinguishing it from a real I/O failure is the caller-visible
    /// difference between an empty timeline and a broken one).
    async fn open_if_exists(&self) -> Result<Option<Dataset>, WriteFailed> {
        match Dataset::open(&self.dataset_path).await {
            Ok(ds) => Ok(Some(ds)),
            Err(lance::Error::DatasetNotFound { .. }) => Ok(None),
            Err(e) => Err(WriteFailed(format!("open {}: {e}", self.dataset_path))),
        }
    }

    /// Build the single atomic RecordBatch for a cycle: the frame row first,
    /// then the landings in their ALREADY-canonical order (the loom ran in
    /// `DetachedCycleBatch::freeze`; storage order = stream order by contract).
    fn build_batch(batch: &DetachedCycleBatch) -> Result<RecordBatch, WriteFailed> {
        let n = batch.landings.len() + 1;
        let mut kind = Vec::with_capacity(n);
        let mut cycle = Vec::with_capacity(n);
        let mut base_version = Vec::with_capacity(n);
        let mut stream_position = Vec::with_capacity(n);
        let mut owner = Vec::with_capacity(n);
        let mut row = Vec::with_capacity(n);
        let mut move_mailbox = UInt32Builder::with_capacity(n);
        let mut move_from = UInt8Builder::with_capacity(n);
        let mut move_to = UInt8Builder::with_capacity(n);
        let mut move_wcp = UInt32Builder::with_capacity(n);
        let mut move_exec = UInt8Builder::with_capacity(n);
        let mut payload = BinaryBuilder::new();

        // Frame row — the cycle ↔ version mapping, sealed atomically with its
        // landings (a zero-landing cycle still advances the timeline).
        kind.push(KIND_FRAME);
        cycle.push(batch.frame.cycle.0);
        base_version.push(batch.frame.base_version.0);
        stream_position.push(0);
        owner.push(0);
        row.push(0);
        move_mailbox.append_null();
        move_from.append_null();
        move_to.append_null();
        move_wcp.append_null();
        move_exec.append_null();
        payload.append_value([]);

        for s in &batch.landings {
            kind.push(KIND_LANDING);
            cycle.push(s.cycle.0);
            base_version.push(batch.frame.base_version.0);
            stream_position.push(s.stream_position);
            owner.push(s.owner);
            row.push(s.row);
            match &s.paired_move {
                Some(m) => {
                    move_mailbox.append_value(m.mailbox);
                    move_from.append_value(m.from as u8);
                    move_to.append_value(m.to as u8);
                    move_wcp.append_value(m.witness_chain_position);
                    move_exec.append_value(m.exec as u8);
                }
                None => {
                    move_mailbox.append_null();
                    move_from.append_null();
                    move_to.append_null();
                    move_wcp.append_null();
                    move_exec.append_null();
                }
            }
            payload.append_value(&s.payload);
        }

        RecordBatch::try_new(
            cycle_store_schema(),
            vec![
                Arc::new(UInt8Array::from(kind)),
                Arc::new(UInt64Array::from(cycle)),
                Arc::new(UInt64Array::from(base_version)),
                Arc::new(UInt64Array::from(stream_position)),
                Arc::new(UInt32Array::from(owner)),
                Arc::new(UInt64Array::from(row)),
                Arc::new(move_mailbox.finish()),
                Arc::new(move_from.finish()),
                Arc::new(move_to.finish()),
                Arc::new(move_wcp.finish()),
                Arc::new(move_exec.finish()),
                Arc::new(payload.finish()),
            ],
        )
        .map_err(|e| WriteFailed(format!("build cycle batch: {e}")))
    }

    /// Read the whole store at its LATEST version, in stored (insertion) order —
    /// Lance's in-order scan; this sink never sorts on read.
    async fn read_all_rows(&self, ds: &Dataset) -> Result<Vec<StoredRow>, WriteFailed> {
        let mut scan = ds.scan();
        scan.scan_in_order(true);
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan {}: {e}", self.dataset_path)))?
            .try_collect()
            .await
            .map_err(|e| WriteFailed(format!("collect {}: {e}", self.dataset_path)))?;

        let mut rows = Vec::new();
        for b in &batches {
            let col_u8 = |name: &str| -> Result<&UInt8Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let col_u32 = |name: &str| -> Result<&UInt32Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let col_u64 = |name: &str| -> Result<&UInt64Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let kind = col_u8("kind")?;
            let cycle = col_u64("cycle")?;
            let base_version = col_u64("base_version")?;
            let stream_position = col_u64("stream_position")?;
            let owner = col_u32("owner")?;
            let row = col_u64("row")?;
            let move_mailbox = col_u32("move_mailbox")?;
            let move_from = col_u8("move_from")?;
            let move_to = col_u8("move_to")?;
            let move_wcp = col_u32("move_witness_chain_position")?;
            let move_exec = col_u8("move_exec")?;
            let payload: &BinaryArray = b
                .column_by_name("payload")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column payload".into()))?;

            for i in 0..b.num_rows() {
                let paired_move = if move_mailbox.is_valid(i) {
                    Some(KanbanMove {
                        mailbox: move_mailbox.value(i) as MailboxId,
                        from: KanbanColumn::from_u8(move_from.value(i)),
                        to: KanbanColumn::from_u8(move_to.value(i)),
                        witness_chain_position: move_wcp.value(i),
                        exec: ExecTarget::from_u8(move_exec.value(i)),
                    })
                } else {
                    None
                };
                rows.push(StoredRow {
                    kind: kind.value(i),
                    cycle: CycleId(cycle.value(i)),
                    base_version: DatasetVersion(base_version.value(i)),
                    slot: SweepSlot {
                        cycle: CycleId(cycle.value(i)),
                        stream_position: stream_position.value(i),
                        owner: owner.value(i),
                        row: row.value(i),
                        paired_move,
                        payload: payload.value(i).to_vec(),
                    },
                });
            }
        }
        Ok(rows)
    }
}

/// One decoded store row (frame or landing) — internal read shape.
struct StoredRow {
    kind: u8,
    cycle: CycleId,
    base_version: DatasetVersion,
    slot: SweepSlot,
}

impl StoredRow {
    /// The version this row's cycle sealed into — `base + 1`, the identity the
    /// commit path verified against the real published Lance version.
    fn sealed_version(&self) -> DatasetVersion {
        DatasetVersion(self.base_version.0 + 1)
    }
}

impl WalSink for LanceCycleSink {
    /// THE single amortized durable append for a whole cycle, over the official
    /// Lance insert path — one commit, one new `DatasetVersion`, all-or-nothing.
    ///
    /// The epistemic fence, both halves:
    /// 1. **Pre-commit:** the store's current version must equal `base` (`Vn`).
    ///    An empty store has head `DatasetVersion(0)`, so the first cycle must
    ///    declare base 0 (it read no sealed predecessor). A stale base is
    ///    refused with NOTHING written.
    /// 2. **Post-commit:** the published version must be exactly `base + 1`.
    ///    Lance auto-resolves append-append conflicts, so a foreign interleaved
    ///    writer surfaces here as a loud timeline anomaly instead of silently
    ///    shifting the cycle ↔ version identity that reads derive.
    async fn commit_cycle(
        &self,
        base: DatasetVersion,
        batch: DetachedCycleBatch,
    ) -> Result<DatasetVersion, WriteFailed> {
        if batch.frame.base_version != base {
            return Err(WriteFailed(format!(
                "frame base {:?} != commit base {base:?}",
                batch.frame.base_version
            )));
        }
        let record_batch = Self::build_batch(&batch)?;
        let schema = cycle_store_schema();
        let published = match self.open_if_exists().await? {
            None => {
                // Empty store: sealed head is DatasetVersion(0) by convention.
                if base.0 != 0 {
                    return Err(WriteFailed(format!(
                        "stale base {base:?}: sealed head is DatasetVersion(0) (empty store)"
                    )));
                }
                let reader = RecordBatchIterator::new(vec![Ok(record_batch)], schema);
                let params = WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                };
                let ds = Dataset::write(reader, &self.dataset_path, Some(params))
                    .await
                    .map_err(|e| WriteFailed(format!("create commit: {e}")))?;
                ds.version().version
            }
            Some(mut ds) => {
                let head = ds.version().version;
                if head != base.0 {
                    return Err(WriteFailed(format!(
                        "stale base {base:?}: sealed head is DatasetVersion({head})"
                    )));
                }
                let reader = RecordBatchIterator::new(vec![Ok(record_batch)], schema);
                ds.append(reader, None)
                    .await
                    .map_err(|e| WriteFailed(format!("append commit: {e}")))?;
                ds.version().version
            }
        };
        if published != base.0 + 1 {
            // The data IS committed at `published` — but the one-writer §I.6
            // timeline is broken (a foreign writer interleaved). Surface loudly;
            // never let a shifted identity pass as a sealed cycle.
            return Err(WriteFailed(format!(
                "timeline anomaly: committed at DatasetVersion({published}), expected {}",
                base.0 + 1
            )));
        }
        Ok(DatasetVersion(published))
    }

    /// Committed landings only, in the STORED canonical order, from the
    /// REOPENED dataset — never an in-memory echo. `from_version` filters to
    /// cycles sealed strictly after it.
    async fn scan_sealed(
        &self,
        from_version: Option<DatasetVersion>,
    ) -> Result<Vec<LandedSlot>, WriteFailed> {
        let Some(ds) = self.open_if_exists().await? else {
            return Ok(Vec::new());
        };
        let rows = self.read_all_rows(&ds).await?;
        Ok(rows
            .into_iter()
            .filter(|r| r.kind == KIND_LANDING)
            .filter(|r| from_version.is_none_or(|f| r.sealed_version() > f))
            .map(|r| LandedSlot {
                version: r.sealed_version(),
                slot: r.slot,
            })
            .collect())
    }

    /// The cheap coarse timeline — the per-cycle frame rows, each sealed in the
    /// same atomic commit as its landings, read back from the reopened store.
    async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed> {
        let Some(ds) = self.open_if_exists().await? else {
            return Ok(Vec::new());
        };
        let rows = self.read_all_rows(&ds).await?;
        Ok(rows
            .into_iter()
            .filter(|r| r.kind == KIND_FRAME)
            .map(|r| (r.cycle, r.sealed_version()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests — every guarantee proven against a REOPENED dataset (fresh sink
// instance, fresh `Dataset::open`), never an in-memory echo.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_planner::persist_sink::{persist_cycle, CycleFrame};

    fn mv(owner: MailboxId) -> KanbanMove {
        KanbanMove {
            mailbox: owner,
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
            witness_chain_position: 7,
            exec: ExecTarget::Elixir,
        }
    }

    fn slot(cycle: u64, stream_position: u64, owner: MailboxId, row: u64) -> SweepSlot {
        SweepSlot {
            cycle: CycleId(cycle),
            stream_position,
            owner,
            row,
            paired_move: Some(mv(owner)),
            payload: vec![stream_position as u8; 4],
        }
    }

    /// One cycle → ONE official Lance commit → exactly one real DatasetVersion
    /// `base + 1`; a fresh sink over the same path (restart) reads the sealed
    /// landings and the cycle ↔ version mapping back from storage.
    #[tokio::test]
    async fn seal_survives_restart_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("witness_cycles.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());

        let frame = CycleFrame::new(CycleId(1), DatasetVersion(0));
        let casts = vec![slot(1, 20, 5, 100), slot(1, 10, 5, 101)];
        let v = persist_cycle(&sink, frame, casts).await.unwrap();
        assert_eq!(v, DatasetVersion(1));

        // The REAL Lance version chain agrees — not a private counter.
        let ds = Dataset::open(path.to_str().unwrap()).await.unwrap();
        assert_eq!(ds.version().version, 1);

        // Restart: a brand-new sink instance, nothing shared but the path.
        let reopened = LanceCycleSink::new(path.to_str().unwrap());
        let sealed = reopened.scan_sealed(None).await.unwrap();
        assert_eq!(sealed.len(), 2);
        // Stored canonical order (deinterlaced at freeze: 10 before 20) — the
        // scan preserves it, it does not repair it.
        assert_eq!(sealed[0].slot.stream_position, 10);
        assert_eq!(sealed[1].slot.stream_position, 20);
        assert_eq!(sealed[0].version, DatasetVersion(1));
        assert_eq!(sealed[0].slot.paired_move, Some(mv(5)));
        assert_eq!(sealed[0].slot.payload, vec![10u8; 4]);

        let versions = reopened.versions().await.unwrap();
        assert_eq!(versions, vec![(CycleId(1), DatasetVersion(1))]);
    }

    /// A stale `base` is fenced with NOTHING written: the store's version chain,
    /// landings, and timeline are untouched — proven on reopen.
    #[tokio::test]
    async fn stale_base_is_fenced_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("witness_cycles.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());

        // Empty store: a caller claiming a sealed predecessor V3 is refused.
        let err = sink
            .commit_cycle(
                DatasetVersion(3),
                DetachedCycleBatch::freeze(CycleFrame::new(CycleId(1), DatasetVersion(3)), vec![]),
            )
            .await
            .unwrap_err();
        assert!(err.0.contains("stale base"), "{err}");
        assert!(Dataset::open(path.to_str().unwrap()).await.is_err());

        // Seal cycle 1 at base 0 → V1; then a sibling still reading base 0 is
        // fenced, and the store is byte-for-byte the sealed head it was.
        persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(1, 1, 2, 40)],
        )
        .await
        .unwrap();
        let err = sink
            .commit_cycle(
                DatasetVersion(0),
                DetachedCycleBatch::freeze(
                    CycleFrame::new(CycleId(2), DatasetVersion(0)),
                    vec![slot(2, 2, 2, 41)],
                ),
            )
            .await
            .unwrap_err();
        assert!(err.0.contains("stale base"), "{err}");

        let reopened = LanceCycleSink::new(path.to_str().unwrap());
        let ds = Dataset::open(path.to_str().unwrap()).await.unwrap();
        assert_eq!(ds.version().version, 1, "fenced commit must not publish");
        assert_eq!(reopened.scan_sealed(None).await.unwrap().len(), 1);
        assert_eq!(reopened.versions().await.unwrap().len(), 1);
    }

    /// Sequential cycles chain the sealed horizon: V1 → V2 → V3; `scan_sealed`
    /// filters strictly-after; `versions` is the full coarse timeline.
    #[tokio::test]
    async fn sequential_cycles_chain_and_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("witness_cycles.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());

        for (cycle, base) in [(1u64, 0u64), (2, 1), (3, 2)] {
            let v = persist_cycle(
                &sink,
                CycleFrame::new(CycleId(cycle), DatasetVersion(base)),
                vec![slot(cycle, cycle * 10, 9, cycle)],
            )
            .await
            .unwrap();
            assert_eq!(v, DatasetVersion(base + 1));
        }

        let reopened = LanceCycleSink::new(path.to_str().unwrap());
        assert_eq!(
            reopened.versions().await.unwrap(),
            vec![
                (CycleId(1), DatasetVersion(1)),
                (CycleId(2), DatasetVersion(2)),
                (CycleId(3), DatasetVersion(3)),
            ]
        );
        // Strictly after V1: cycles 2 and 3 only.
        let after_v1 = reopened.scan_sealed(Some(DatasetVersion(1))).await.unwrap();
        assert_eq!(after_v1.len(), 2);
        assert!(after_v1.iter().all(|l| l.version > DatasetVersion(1)));
    }

    /// A zero-landing cycle still advances the sealed timeline (its frame row
    /// commits atomically) while contributing nothing to `scan_sealed`.
    #[tokio::test]
    async fn empty_cycle_advances_timeline_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("witness_cycles.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());

        let v = persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![],
        )
        .await
        .unwrap();
        assert_eq!(v, DatasetVersion(1));

        let reopened = LanceCycleSink::new(path.to_str().unwrap());
        assert!(reopened.scan_sealed(None).await.unwrap().is_empty());
        assert_eq!(
            reopened.versions().await.unwrap(),
            vec![(CycleId(1), DatasetVersion(1))]
        );
    }

    /// An empty store is a state, not an error: nothing sealed, empty timeline.
    #[tokio::test]
    async fn empty_store_reads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("never_created.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());
        assert!(sink.scan_sealed(None).await.unwrap().is_empty());
        assert!(sink.versions().await.unwrap().is_empty());
    }

    /// A no-move landing round-trips as `None` (nullable move columns), and a
    /// large-ish payload survives byte-exact.
    #[tokio::test]
    async fn move_nullability_and_payload_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("witness_cycles.lance");
        let sink = LanceCycleSink::new(path.to_str().unwrap());

        let witness_node = (0..=255u8).cycle().take(512).collect::<Vec<u8>>();
        let mut s = slot(1, 3, 11, 900);
        s.paired_move = None;
        s.payload = witness_node.clone();
        persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![s],
        )
        .await
        .unwrap();

        let reopened = LanceCycleSink::new(path.to_str().unwrap());
        let sealed = reopened.scan_sealed(None).await.unwrap();
        assert_eq!(sealed.len(), 1);
        assert_eq!(sealed[0].slot.paired_move, None);
        assert_eq!(sealed[0].slot.payload, witness_node);
        assert_eq!(sealed[0].slot.owner, 11);
        assert_eq!(sealed[0].slot.row, 900);
    }
}
