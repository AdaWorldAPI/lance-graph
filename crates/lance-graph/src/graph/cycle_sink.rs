// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The concrete Lance-backed cycle store — [`LanceCycleWriter`], the SOLE
//! application writer (Phase A of the canonical persistence contract,
//! operator-ruled 2026-08-09; supersedes the `LanceCycleSink` shipped in #911).
//!
//! # One logical writer, owned
//!
//! There is exactly ONE logical application writer per cycle store. The 64k
//! thoughts / SoA owners are parallel PRODUCERS (fire-and-forget: they cast on
//! behalf of their mailbox and receive no acknowledgement), never Lance
//! writers. This type makes the topology structural:
//!
//! - **non-`Clone`** — a second handle to the same writer cannot be minted;
//! - **`commit_cycle(&mut self, …)`** — two application commits cannot
//!   interleave through the type boundary;
//! - **long-lived handle** — the writer OWNS its `Dataset` handle and current
//!   head. Reads use the held handle; there is no per-operation reopen. The
//!   dataset is opened exactly once at construction, and re-opened only to
//!   resolve an ambiguous commit outcome (a lost acknowledgement).
//!
//! Lance's own transaction/manifest machinery (the backend durability path)
//! is INTERNAL to this one writer — `Dataset::write` / `Dataset::append` are
//! official atomic Lance MVCC commits, and nothing else writes here. An
//! unexpected head can only mean: an earlier commit became durable but its
//! response was lost; a restart reopened from a stale cached head; an
//! unauthorized writer violated the topology; or corruption. It is a
//! fence/reconciliation condition, never normal competition.
//!
//! # The governing storage rule (why there is no empty-cycle version)
//!
//! **No artifact-backed semantic change → no write → no new `DatasetVersion`.**
//! `persist_cycle` partitions intent-only casts out and returns
//! `CommitOutcome::NoChange` without ever calling this writer, so a timer
//! tick, an empty cycle, a `Continue`, a held intent or a pure kanban step
//! performs ZERO Lance operations here. The #911 deliberate empty-cycle
//! versioning is REMOVED. Kanban progress rides along ONLY when an artifact
//! commit happens anyway (the moves of artifact casts, sealed in the same
//! atomic commit).
//!
//! # No rollback, no compensating delete — reconciliation is authoritative
//!
//! Lance 9 has no atomic expected-version fence for Append (the conflict
//! rebase runs even on a single-attempt commit; strict no-rebase mode exists
//! only for Overwrite — measured in `lance-9.0.0/src/io/commit.rs`), and a
//! published manifest is HISTORY (`Dataset::delete` creates another version;
//! it is not rollback — the #911 compensating delete is removed, not
//! repaired). Instead, idempotency is durable: every committed row carries
//! its `(cycle, batch_hash)` in the same commit, and [`WalSink::commit_cycle`]
//! reconciles FIRST — an already-durable batch returns
//! [`CommitOutcome::Reconciled`]; a matching cycle with a different hash
//! fails closed ([`CommitError::HashConflict`]); an append whose
//! acknowledgement was lost is resolved by re-submitting the SAME frozen
//! batch. Only when reconciliation itself cannot answer does
//! [`CommitError::Ambiguous`] surface.
//!
//! # Reference the new version; never reload normal state
//!
//! After a successful commit the caller already holds the outcome (version,
//! cycle, hash) and the submitted batch. The normal path performs **zero
//! reopens, zero scans, zero readbacks** — [`LanceCycleWriter::opens`] counts
//! every `Dataset::open` this writer ever performs so the invariant is
//! instrumented, not asserted. `scan_sealed` / `timeline` exist for recovery,
//! audit and downstream consumers, are bounded (`after_cycle` pushed into the
//! Lance scan as a predicate) and projected (the timeline never touches the
//! payload column).
//!
//! # Copy boundary (documented honestly, isolated for a later measured PR)
//!
//! This writer materializes the frozen batch's landings into Arrow builders
//! (one copy of each payload) and `scan_*` reads copy bytes back out
//! (`to_vec`). True zero-copy (Arc-backed Arrow buffers pinned over SoA
//! ranges) does not fit this focused repair; the copy boundary is exactly
//! these two seams and nothing else. The 512-byte witness ABI is enforced on
//! ARTIFACT payloads only — intent-only casts never reach this writer, so the
//! `restage_held` empty-payload shape can never trip the gate.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_array::{
    builder::{FixedSizeBinaryBuilder, UInt32Builder, UInt8Builder},
    Array, FixedSizeBinaryArray, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot,
    WalSink, WriteFailed,
};

/// Row kind: the per-cycle frame row (cycle identity + batch hash + read
/// horizon — compact metadata, sealed atomically with its landings).
const KIND_FRAME: u8 = 0;
/// Row kind: one artifact landing's TRANSITION METADATA (move + position;
/// payload column NULL — compact, per cast).
const KIND_LANDING: u8 = 1;
/// Row kind: one coalesced-image row — `row` + the FINAL 512-byte payload
/// after the per-row fold. Exactly one payload per dirty row per cycle: 64
/// same-row breaths durably cost ONE image row, never 64 × 512 bytes.
const KIND_IMAGE: u8 = 2;

/// The canonical witness-node ABI for ARTIFACT payloads:
/// `key(16) | edges(16) | value(480)` — the 512-byte node row stride.
pub const EPISODIC_WITNESS_BYTES: usize = 512;

/// The Arrow schema — flat rows, three kinds, payload physically
/// `FixedSizeBinary(512)` (nullable: only image rows carry it).
///
/// | column | frame | landing | image |
/// |---|---|---|---|
/// | `kind` | 0 | 1 | 2 |
/// | `cycle` / `base_version` / `batch_hash` | ✓ | ✓ | ✓ |
/// | `stream_position` / `owner` / `row` | 0 | ✓ | 0 / 0 / row |
/// | `move_*` (nullable) | null | cast's move | null |
/// | `payload` (`FixedSizeBinary(512)`, nullable) | null | null | final image |
pub fn cycle_store_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("kind", DataType::UInt8, false),
        Field::new("cycle", DataType::UInt64, false),
        Field::new("base_version", DataType::UInt64, false),
        Field::new("batch_hash", DataType::UInt64, false),
        Field::new("stream_position", DataType::UInt64, false),
        Field::new("owner", DataType::UInt32, false),
        Field::new("row", DataType::UInt64, false),
        Field::new("move_mailbox", DataType::UInt32, true),
        Field::new("move_from", DataType::UInt8, true),
        Field::new("move_to", DataType::UInt8, true),
        Field::new("move_witness_chain_position", DataType::UInt32, true),
        Field::new("move_exec", DataType::UInt8, true),
        Field::new(
            "payload",
            DataType::FixedSizeBinary(EPISODIC_WITNESS_BYTES as i32),
            true,
        ),
    ]))
}

/// The sole owned application writer over one Lance cycle store.
///
/// Deliberately **non-`Clone`**: constructing a second writer over the same
/// path is a topology violation the type cannot prevent across processes, but
/// within a process the exclusive `&mut` commit boundary plus non-cloneability
/// make interleaved application commits unrepresentable.
#[derive(Debug)]
pub struct LanceCycleWriter {
    dataset_path: String,
    /// The long-lived handle. `None` until the first committed cycle creates
    /// the dataset (an empty store is a state, not an error). Once `Some`, it
    /// NEVER degrades back to `None` — a store that existed cannot become
    /// "empty" again (see [`reopen`](Self::reopen)).
    ds: Option<Dataset>,
    /// `Dataset::open` count — startup + ambiguity-resolution ONLY. The
    /// normal-path invariant (zero post-success reopens) is instrumented here.
    opens: AtomicU64,
    /// Reconciliation-scan count ([`find_frame`](Self::find_frame) calls). The
    /// NORMAL commit path performs ZERO of these: a fresh monotonic cycle
    /// appends directly over the in-memory head + cycle watermark; the scan
    /// runs only on a fence mismatch, a `cycle ≤ watermark` re-submission, or
    /// ambiguity resolution. Instrumented so "zero scans on the normal path"
    /// is measured, not asserted.
    reconcile_scans: AtomicU64,
    /// The highest cycle known durable in THIS store (seeded at open from a
    /// frame-projected streaming fold — an O(#cycles) metadata scan with O(1)
    /// memory, startup hydration, never a normal-path read; advanced in
    /// memory on every commit). Monotonic — `cycle > committed_through`
    /// proves the cycle cannot already be durable, which is what makes the
    /// scan-free fast path sound.
    committed_through: Option<CycleId>,
    /// The RAII registry claim. Held from BEFORE `open`'s first `.await`, so
    /// a cancelled or failed `open` releases its slot through `Drop` — no
    /// manual removal on any path, no leaked reservation.
    ///
    /// Never read: its `Drop` IS its behaviour, and the value must live
    /// exactly as long as the writer.
    #[allow(dead_code)]
    claim: WriterClaim,
    /// Test-only fault injection for the append / reopen / reconcile branch
    /// (the branch that carries the whole no-rollback contract and cannot be
    /// made to fail deterministically through real Lance).
    #[cfg(test)]
    fault: TestFaults,
}

/// The process-local single-writer registry: one LIVE [`LanceCycleWriter`] per
/// store IDENTITY. `non-Clone + &mut self` serializes commits on one instance;
/// this registry closes the remaining in-process hole (a second `open` of the
/// same store is REFUSED while the first writer lives). Cross-PROCESS
/// exclusivity remains a deployment lease this crate cannot enforce — stated,
/// not implied away.
static OPEN_WRITERS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// The LEXICAL store identity the registry keys on: `.` segments, duplicate
/// separators and trailing slashes are collapsed, so `x/./cycles.lance`,
/// `x//cycles.lance` and `x/cycles.lance` claim ONE slot. A URI's
/// `scheme://authority` prefix is preserved verbatim. Deliberately NOT
/// filesystem canonicalization: `..` and symlinks are left alone (resolving
/// them needs I/O and still cannot cover object stores) — two spellings that
/// only a symlink makes equal remain the deployment lease's problem, same as
/// two processes.
fn store_identity(path: &str) -> String {
    let (prefix, rest) = match path.find("://") {
        Some(i) => {
            let after_scheme = i + 3;
            match path[after_scheme..].find('/') {
                Some(j) => path.split_at(after_scheme + j),
                None => (path, ""),
            }
        }
        None => ("", path),
    };
    let absolute = rest.starts_with('/');
    let parts: Vec<&str> = rest
        .split('/')
        .filter(|seg| !seg.is_empty() && *seg != ".")
        .collect();
    let mut s = String::from(prefix);
    if absolute {
        s.push('/');
    }
    s.push_str(&parts.join("/"));
    s
}

/// An RAII claim on [`OPEN_WRITERS`]. Acquired synchronously BEFORE the first
/// `.await` in [`LanceCycleWriter::open`]; released by `Drop` — which covers
/// the error paths, the writer's own drop, AND an `open` future that is
/// cancelled mid-`Dataset::open` (previously a leaked reservation, because no
/// RAII owner existed yet at that point).
#[derive(Debug)]
struct WriterClaim(String);

impl WriterClaim {
    fn acquire(identity: String) -> Result<Self, WriteFailed> {
        let mut set = OPEN_WRITERS
            .lock()
            .map_err(|_| WriteFailed("writer registry poisoned".into()))?;
        if !set.insert(identity.clone()) {
            return Err(WriteFailed(format!(
                "a live LanceCycleWriter already owns {identity} in this process — \
                 one logical writer per store (drop it first)"
            )));
        }
        Ok(Self(identity))
    }
}

impl Drop for WriterClaim {
    fn drop(&mut self) {
        if let Ok(mut set) = OPEN_WRITERS.lock() {
            set.remove(&self.0);
        }
    }
}

/// Deterministic failure injection for the ambiguous-append branch —
/// test-only, one-shot flags (each `swap(false)`s when consumed).
#[cfg(test)]
#[derive(Debug, Default)]
struct TestFaults {
    /// Fail the next append WITHOUT publishing anything (the store is
    /// untouched — models a pre-manifest I/O failure).
    fail_append_unpublished: std::sync::atomic::AtomicBool,
    /// Perform the next append for real, then report it failed (the manifest
    /// IS durable — models a lost acknowledgement inside one attempt).
    fail_append_published: std::sync::atomic::AtomicBool,
    /// Fail the next reconciliation read (`find_frame`) — models storage
    /// unavailable while resolving an ambiguous append.
    fail_reconcile_read: std::sync::atomic::AtomicBool,
}

impl LanceCycleWriter {
    /// Open the writer over `path` (local path or object-store URI). Performs
    /// the ONE startup open (plus one bounded, frame-projected seed scan when
    /// the store exists); a missing dataset is an empty store.
    ///
    /// Refuses a second live writer on the same path in this process (the
    /// one-logical-writer topology, enforced rather than narrated). Refuses a
    /// store whose schema is not this writer's layout — a pre-Phase-A (#911)
    /// store is REJECTED loudly, never silently reinterpreted.
    pub async fn open(path: impl Into<String>) -> Result<Self, WriteFailed> {
        // The claim is taken on the LEXICAL identity, synchronously, before
        // the first await — errors and cancellation below release it via
        // RAII, and `x/./cycles.lance` cannot claim a second slot beside
        // `x/cycles.lance`. The normalized identity is also what we open:
        // the two spellings resolve to the same store, so I/O and identity
        // must not diverge.
        let dataset_path = store_identity(&path.into());
        let claim = WriterClaim::acquire(dataset_path.clone())?;
        let opens = AtomicU64::new(0);
        let ds = match Dataset::open(&dataset_path).await {
            Ok(ds) => {
                opens.fetch_add(1, Ordering::Relaxed);
                Self::guard_schema(&dataset_path, &ds)?;
                Some(ds)
            }
            Err(lance::Error::DatasetNotFound { .. }) => None,
            Err(e) => return Err(WriteFailed(format!("open {dataset_path}: {e}"))),
        };
        let mut w = Self {
            dataset_path,
            ds,
            opens,
            reconcile_scans: AtomicU64::new(0),
            committed_through: None,
            claim,
            #[cfg(test)]
            fault: TestFaults::default(),
        };
        if w.ds.is_some() {
            // Startup hydration: a frame-projected STREAMING fold to the
            // highest durable cycle — O(#cycles) metadata rows scanned,
            // O(1) memory (nothing materialized), never a normal-path read.
            w.committed_through = w.max_cycle().await?;
        }
        Ok(w)
    }

    /// Refuse a store whose schema is not this writer's layout — names AND
    /// types AND nullability, so a pre-Phase-A (#911) store, a hand-altered
    /// column, or a same-name/different-type drift is REJECTED loudly, never
    /// silently reinterpreted (I-LEGACY-API-FEATURE-GATED).
    fn guard_schema(dataset_path: &str, ds: &Dataset) -> Result<(), WriteFailed> {
        let expected = cycle_store_schema();
        let got = ds.schema();
        for field in expected.fields() {
            let Some(g) = got.field(field.name()) else {
                return Err(WriteFailed(format!(
                    "store at {dataset_path} is missing column `{}` — not this \
                     writer's layout (a pre-Phase-A store is rejected, not \
                     reinterpreted; migrate or discard it explicitly)",
                    field.name()
                )));
            };
            if g.data_type() != *field.data_type() || g.nullable != field.is_nullable() {
                return Err(WriteFailed(format!(
                    "store at {dataset_path} column `{}` is {:?} (nullable={}) but this \
                     writer's layout requires {:?} (nullable={}) — rejected, not \
                     reinterpreted",
                    field.name(),
                    g.data_type(),
                    g.nullable,
                    field.data_type(),
                    field.is_nullable()
                )));
            }
        }
        Ok(())
    }

    /// The highest durable cycle, as a streaming fold over frame rows
    /// (projection: `cycle`; filter: `kind = 0`). The startup seed for
    /// [`Self::commit_cycle`]'s fast-path watermark.
    async fn max_cycle(&self) -> Result<Option<CycleId>, WriteFailed> {
        let Some(ds) = self.ds.as_ref() else {
            return Ok(None);
        };
        let mut scan = ds.scan();
        scan.filter(&format!("kind = {KIND_FRAME}"))
            .map_err(|e| WriteFailed(format!("filter: {e}")))?;
        scan.project(&["cycle"])
            .map_err(|e| WriteFailed(format!("project: {e}")))?;
        let mut stream = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?;
        let mut max: Option<u64> = None;
        while let Some(b) = stream
            .try_next()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?
        {
            let cycle: &UInt64Array = b
                .column_by_name("cycle")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column cycle".into()))?;
            for i in 0..cycle.len() {
                let v = cycle.value(i);
                max = Some(max.map_or(v, |m: u64| m.max(v)));
            }
        }
        Ok(max.map(CycleId))
    }

    /// The store's current head version (`0` = empty store) — the in-memory
    /// token the normal path references instead of reloading state.
    #[must_use]
    pub fn head(&self) -> DatasetVersion {
        DatasetVersion(self.ds.as_ref().map_or(0, |d| d.version().version))
    }

    /// The dataset path this writer commits to.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }

    /// How many `Dataset::open` calls this writer has EVER performed —
    /// startup (≤1) plus ambiguity resolutions. The zero-reload falsifier
    /// asserts this stays flat across normal commits and reads.
    #[must_use]
    pub fn opens(&self) -> u64 {
        self.opens.load(Ordering::Relaxed)
    }

    /// Build the single atomic RecordBatch: frame row, landing-metadata rows
    /// (canonical order, payload null), image rows (row-ascending, final
    /// payload). Refuses a non-512-byte ARTIFACT payload before anything
    /// durable happens (intent-only casts never reach this writer).
    fn build_batch(batch: &DetachedCycleBatch) -> Result<RecordBatch, CommitError> {
        let n = 1 + batch.landings.len() + batch.image.len();
        let mut kind = Vec::with_capacity(n);
        let mut cycle = Vec::with_capacity(n);
        let mut base_version = Vec::with_capacity(n);
        let mut batch_hash = Vec::with_capacity(n);
        let mut stream_position = Vec::with_capacity(n);
        let mut owner = Vec::with_capacity(n);
        let mut row = Vec::with_capacity(n);
        let mut move_mailbox = UInt32Builder::with_capacity(n);
        let mut move_from = UInt8Builder::with_capacity(n);
        let mut move_to = UInt8Builder::with_capacity(n);
        let mut move_wcp = UInt32Builder::with_capacity(n);
        let mut move_exec = UInt8Builder::with_capacity(n);
        let mut payload = FixedSizeBinaryBuilder::with_capacity(n, EPISODIC_WITNESS_BYTES as i32);

        let mut push_common = |k: u8, sp: u64, ow: u32, rw: u64| {
            kind.push(k);
            cycle.push(batch.frame.cycle.0);
            base_version.push(batch.frame.base_version.0);
            batch_hash.push(batch.batch_hash);
            stream_position.push(sp);
            owner.push(ow);
            row.push(rw);
        };

        // Frame row — compact metadata, no move, no payload.
        push_common(KIND_FRAME, 0, 0, 0);
        move_mailbox.append_null();
        move_from.append_null();
        move_to.append_null();
        move_wcp.append_null();
        move_exec.append_null();
        payload.append_null();

        // Landing metadata rows — the sparse transition set, payload NULL.
        for s in &batch.landings {
            // PERMANENT refusal, never Io: an ABI-malformed artifact would
            // "fail, regenerate identically, fail" forever if reported as
            // retryable I/O. Nothing durable has happened at this point.
            if s.payload.len() != EPISODIC_WITNESS_BYTES {
                return Err(CommitError::InvalidArtifact {
                    row: s.row,
                    len: s.payload.len(),
                });
            }
            push_common(KIND_LANDING, s.stream_position, s.owner, s.row);
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
            payload.append_null();
        }

        // Image rows — the coalesced final payload, once per dirty row.
        for (row_id, image_payload) in &batch.image {
            push_common(KIND_IMAGE, 0, 0, *row_id);
            move_mailbox.append_null();
            move_from.append_null();
            move_to.append_null();
            move_wcp.append_null();
            move_exec.append_null();
            payload
                .append_value(image_payload)
                .map_err(|e| CommitError::Io(WriteFailed(format!("image row {row_id}: {e}"))))?;
        }

        RecordBatch::try_new(
            cycle_store_schema(),
            vec![
                Arc::new(UInt8Array::from(kind)),
                Arc::new(UInt64Array::from(cycle)),
                Arc::new(UInt64Array::from(base_version)),
                Arc::new(UInt64Array::from(batch_hash)),
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
        .map_err(|e| CommitError::Io(WriteFailed(format!("build cycle batch: {e}"))))
    }

    /// Look this cycle's durable frame up (projected `cycle` + `batch_hash`
    /// under a `kind = 0 AND cycle = …` predicate) — the reconciliation read.
    async fn find_frame(&self, cycle: CycleId) -> Result<Option<u64>, WriteFailed> {
        #[cfg(test)]
        if self
            .fault
            .fail_reconcile_read
            .swap(false, Ordering::Relaxed)
        {
            return Err(WriteFailed(
                "injected: reconciliation read unavailable".into(),
            ));
        }
        let Some(ds) = self.ds.as_ref() else {
            return Ok(None);
        };
        let mut scan = ds.scan();
        scan.filter(&format!("kind = {KIND_FRAME} AND cycle = {}", cycle.0))
            .map_err(|e| WriteFailed(format!("filter: {e}")))?;
        scan.project(&["batch_hash"])
            .map_err(|e| WriteFailed(format!("project: {e}")))?;
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?
            .try_collect()
            .await
            .map_err(|e| WriteFailed(format!("collect: {e}")))?;
        for b in &batches {
            let h: &UInt64Array = b
                .column_by_name("batch_hash")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column batch_hash".into()))?;
            if b.num_rows() > 0 {
                return Ok(Some(h.value(0)));
            }
        }
        Ok(None)
    }

    /// Re-open the dataset from storage — ambiguity resolution ONLY (counted).
    ///
    /// A store that existed can NEVER degrade to "empty": if this writer holds
    /// a handle and the reopen reports `DatasetNotFound` (transient listing
    /// failure, eventual consistency, or genuine corruption), the OLD handle
    /// is kept and an error is returned — the caller's outcome stays
    /// `Ambiguous`, and the next commit can never fall into `Create` over a
    /// store that has history.
    async fn reopen(&mut self) -> Result<(), WriteFailed> {
        match Dataset::open(&self.dataset_path).await {
            Ok(ds) => {
                self.opens.fetch_add(1, Ordering::Relaxed);
                self.ds = Some(ds);
                Ok(())
            }
            Err(lance::Error::DatasetNotFound { .. }) => {
                self.opens.fetch_add(1, Ordering::Relaxed);
                if self.ds.is_some() {
                    return Err(WriteFailed(format!(
                        "reopen {}: store reported NOT FOUND but this writer holds \
                         history — keeping the existing handle (a store never \
                         becomes empty again); treat the outcome as ambiguous",
                        self.dataset_path
                    )));
                }
                Ok(())
            }
            Err(e) => Err(WriteFailed(format!("reopen {}: {e}", self.dataset_path))),
        }
    }

    /// How many reconciliation scans ([`find_frame`](Self::find_frame)) this
    /// writer has EVER run. Zero across a run of fresh monotonic commits —
    /// the "zero scans on the normal path" falsifier reads this.
    #[must_use]
    pub fn reconcile_scans(&self) -> u64 {
        self.reconcile_scans.load(Ordering::Relaxed)
    }
}

impl LanceCycleWriter {
    /// The one real store mutation: Create on the first-ever commit, Append
    /// afterwards. Errors are carried as strings — the caller's ambiguity
    /// branch only ever forwards the text, and the test fault-injection seam
    /// needs a constructible error type.
    async fn raw_append(&mut self, record_batch: RecordBatch) -> Result<(), String> {
        let schema = cycle_store_schema();
        let reader = RecordBatchIterator::new(vec![Ok(record_batch)], schema);
        match self.ds.as_mut() {
            None => match Dataset::write(
                reader,
                &self.dataset_path,
                Some(WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                }),
            )
            .await
            {
                Ok(ds) => {
                    self.ds = Some(ds);
                    Ok(())
                }
                Err(e) => Err(e.to_string()),
            },
            Some(ds) => ds.append(reader, None).await.map_err(|e| e.to_string()),
        }
    }
}

impl WalSink for LanceCycleWriter {
    /// THE single durable commit for a whole cycle — reconciliation-first,
    /// fence second, append third; the outcome is fully honored.
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        // Zero-artifact batches never reach a sink (persist_cycle partitions),
        // but the invariant is enforced here too — this writer NEVER creates a
        // version for nothing.
        if batch.landings.is_empty() {
            return Ok(CommitOutcome::NoChange {
                head: DatasetVersion(batch.frame.base_version.0),
            });
        }
        // 1. The scan-free FAST PATH decision. A fresh monotonic cycle
        //    (`cycle > committed_through`, seeded at open) provably cannot be
        //    durable yet, and a matching fence proves the horizon — so the
        //    normal path appends DIRECTLY, zero reads. Reconciliation runs
        //    only when something is off: a fence mismatch (lost-response
        //    restart / stale cache / topology violation) or a re-submission
        //    at-or-below the durable cycle watermark.
        let head = self.head();
        let fresh = self
            .committed_through
            .is_none_or(|ct| batch.frame.cycle > ct);
        if !fresh || batch.frame.base_version != head {
            // Reconciliation: an already-durable (cycle, hash) is success; a
            // matching cycle with a different hash fails closed; a genuinely
            // absent cycle with a bad fence is Fenced (nothing written).
            self.reconcile_scans.fetch_add(1, Ordering::Relaxed);
            match self.find_frame(batch.frame.cycle).await {
                Ok(Some(stored_hash)) => {
                    return if stored_hash == batch.batch_hash {
                        self.committed_through = Some(
                            self.committed_through
                                .map_or(batch.frame.cycle, |ct| ct.max(batch.frame.cycle)),
                        );
                        Ok(CommitOutcome::Reconciled {
                            current_head: head,
                            cycle: batch.frame.cycle,
                            batch_hash: batch.batch_hash,
                        })
                    } else {
                        Err(CommitError::HashConflict {
                            cycle: batch.frame.cycle,
                            stored_hash,
                            offered_hash: batch.batch_hash,
                        })
                    };
                }
                Ok(None) => {
                    if batch.frame.base_version != head {
                        return Err(CommitError::Fenced { current_head: head });
                    }
                    // cycle ≤ watermark but absent (a gap id) with a good
                    // fence: legitimate — fall through to the append.
                }
                Err(e) => {
                    // The reconciliation read itself failed — nothing written
                    // yet: safe to report as refused I/O.
                    return Err(CommitError::Io(e));
                }
            }
        }
        // 3. The single atomic Lance MVCC commit.
        let record_batch = Self::build_batch(&batch)?;
        #[cfg(test)]
        let append_result: Result<(), String> = {
            if self
                .fault
                .fail_append_unpublished
                .swap(false, Ordering::Relaxed)
            {
                // Nothing touched the store — models a pre-manifest failure.
                Err("injected: append failed before publish".into())
            } else if self
                .fault
                .fail_append_published
                .swap(false, Ordering::Relaxed)
            {
                // The manifest IS durable; only the acknowledgement is lost.
                match self.raw_append(record_batch).await {
                    Ok(()) => Err("injected: acknowledgement lost after publish".into()),
                    Err(e) => Err(e),
                }
            } else {
                self.raw_append(record_batch).await
            }
        };
        #[cfg(not(test))]
        let append_result: Result<(), String> = self.raw_append(record_batch).await;
        match append_result {
            Ok(()) => {
                self.committed_through = Some(
                    self.committed_through
                        .map_or(batch.frame.cycle, |ct| ct.max(batch.frame.cycle)),
                );
                Ok(CommitOutcome::Committed {
                    // The ACTUAL publication version returned by the commit —
                    // accepted as-is, never "corrected" (no rollback, no
                    // delete, no derived identity).
                    version: self.head(),
                    cycle: batch.frame.cycle,
                    batch_hash: batch.batch_hash,
                })
            }
            Err(cause) => {
                // The commit's outcome is UNKNOWN (the manifest may or may not
                // have published before the failure). Reconcile from storage:
                // reopen (counted; NEVER degrades an existing handle), then
                // look for our durable identity.
                if let Err(re) = self.reopen().await {
                    return Err(CommitError::Ambiguous {
                        cycle: batch.frame.cycle,
                        batch_hash: batch.batch_hash,
                        cause: format!("append failed ({cause}); reopen failed ({re})"),
                    });
                }
                self.reconcile_scans.fetch_add(1, Ordering::Relaxed);
                match self.find_frame(batch.frame.cycle).await {
                    Ok(Some(stored_hash)) if stored_hash == batch.batch_hash => {
                        self.committed_through = Some(
                            self.committed_through
                                .map_or(batch.frame.cycle, |ct| ct.max(batch.frame.cycle)),
                        );
                        Ok(CommitOutcome::Reconciled {
                            current_head: self.head(),
                            cycle: batch.frame.cycle,
                            batch_hash: batch.batch_hash,
                        })
                    }
                    Ok(Some(stored_hash)) => Err(CommitError::HashConflict {
                        cycle: batch.frame.cycle,
                        stored_hash,
                        offered_hash: batch.batch_hash,
                    }),
                    // Proven absent: nothing landed — safe to regenerate.
                    Ok(None) => Err(CommitError::Io(WriteFailed(format!(
                        "append failed with nothing published: {cause}"
                    )))),
                    Err(re) => Err(CommitError::Ambiguous {
                        cycle: batch.frame.cycle,
                        batch_hash: batch.batch_hash,
                        cause: format!("append failed ({cause}); reconciliation failed ({re})"),
                    }),
                }
            }
        }
    }

    /// Committed landing METADATA in stored canonical order, bounded by
    /// `after_cycle` (pushed into the Lance scan). Payloads are NOT read here
    /// — landing rows carry none (the durable payloads live in the coalesced
    /// image, read via [`LanceCycleWriter::scan_image`]); returned slots carry
    /// empty payload vectors.
    async fn scan_sealed(
        &self,
        after_cycle: Option<CycleId>,
    ) -> Result<Vec<LandedSlot>, WriteFailed> {
        let Some(ds) = self.ds.as_ref() else {
            return Ok(Vec::new());
        };
        let mut scan = ds.scan();
        scan.scan_in_order(true);
        let filter = match after_cycle {
            Some(c) => format!("kind = {KIND_LANDING} AND cycle > {}", c.0),
            None => format!("kind = {KIND_LANDING}"),
        };
        scan.filter(&filter)
            .map_err(|e| WriteFailed(format!("filter: {e}")))?;
        scan.project(&[
            "cycle",
            "stream_position",
            "owner",
            "row",
            "move_mailbox",
            "move_from",
            "move_to",
            "move_witness_chain_position",
            "move_exec",
        ])
        .map_err(|e| WriteFailed(format!("project: {e}")))?;
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?
            .try_collect()
            .await
            .map_err(|e| WriteFailed(format!("collect: {e}")))?;
        let mut out = Vec::new();
        for b in &batches {
            let col_u64 = |name: &str| -> Result<&UInt64Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let col_u32 = |name: &str| -> Result<&UInt32Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let col_u8 = |name: &str| -> Result<&UInt8Array, WriteFailed> {
                b.column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref())
                    .ok_or_else(|| WriteFailed(format!("missing column {name}")))
            };
            let cycle = col_u64("cycle")?;
            let stream_position = col_u64("stream_position")?;
            let owner = col_u32("owner")?;
            let row = col_u64("row")?;
            let move_mailbox = col_u32("move_mailbox")?;
            let move_from = col_u8("move_from")?;
            let move_to = col_u8("move_to")?;
            let move_wcp = col_u32("move_witness_chain_position")?;
            let move_exec = col_u8("move_exec")?;
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
                out.push(LandedSlot {
                    cycle: CycleId(cycle.value(i)),
                    slot: SweepSlot {
                        cycle: CycleId(cycle.value(i)),
                        stream_position: stream_position.value(i),
                        owner: owner.value(i),
                        row: row.value(i),
                        paired_move,
                        payload: Vec::new(),
                    },
                });
            }
        }
        Ok(out)
    }

    /// The coarse timeline — frame rows only, projected `cycle` +
    /// `base_version` + `batch_hash`: the payload column is never scanned.
    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        let Some(ds) = self.ds.as_ref() else {
            return Ok(Vec::new());
        };
        let mut scan = ds.scan();
        scan.scan_in_order(true);
        scan.filter(&format!("kind = {KIND_FRAME}"))
            .map_err(|e| WriteFailed(format!("filter: {e}")))?;
        scan.project(&["cycle", "base_version", "batch_hash"])
            .map_err(|e| WriteFailed(format!("project: {e}")))?;
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?
            .try_collect()
            .await
            .map_err(|e| WriteFailed(format!("collect: {e}")))?;
        let mut out = Vec::new();
        for b in &batches {
            let cycle: &UInt64Array = b
                .column_by_name("cycle")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column cycle".into()))?;
            let base: &UInt64Array = b
                .column_by_name("base_version")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column base_version".into()))?;
            let hash: &UInt64Array = b
                .column_by_name("batch_hash")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column batch_hash".into()))?;
            for i in 0..b.num_rows() {
                out.push(FrameMeta {
                    cycle: CycleId(cycle.value(i)),
                    base_version: DatasetVersion(base.value(i)),
                    batch_hash: hash.value(i),
                });
            }
        }
        Ok(out)
    }
}

impl LanceCycleWriter {
    /// A sealed cycle's durable coalesced image: `row → final 512-byte
    /// payload`. Projected `row` + `payload` under `kind = 2 AND cycle = …`.
    pub async fn scan_image(
        &self,
        cycle: CycleId,
    ) -> Result<std::collections::BTreeMap<u64, Vec<u8>>, WriteFailed> {
        let Some(ds) = self.ds.as_ref() else {
            return Ok(std::collections::BTreeMap::new());
        };
        let mut scan = ds.scan();
        scan.scan_in_order(true);
        scan.filter(&format!("kind = {KIND_IMAGE} AND cycle = {}", cycle.0))
            .map_err(|e| WriteFailed(format!("filter: {e}")))?;
        scan.project(&["row", "payload"])
            .map_err(|e| WriteFailed(format!("project: {e}")))?;
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(|e| WriteFailed(format!("scan: {e}")))?
            .try_collect()
            .await
            .map_err(|e| WriteFailed(format!("collect: {e}")))?;
        let mut out = std::collections::BTreeMap::new();
        for b in &batches {
            let row: &UInt64Array = b
                .column_by_name("row")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column row".into()))?;
            let payload: &FixedSizeBinaryArray = b
                .column_by_name("payload")
                .and_then(|c| c.as_any().downcast_ref())
                .ok_or_else(|| WriteFailed("missing column payload".into()))?;
            for i in 0..b.num_rows() {
                out.insert(row.value(i), payload.value(i).to_vec());
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Falsifiers — every guarantee proven against a REOPENED store (a fresh
// `LanceCycleWriter::open` over the same path), never an in-memory echo.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_planner::persist_sink::{persist_cycle, CycleFrame};

    fn witness(tag: u8) -> Vec<u8> {
        vec![tag; EPISODIC_WITNESS_BYTES]
    }

    fn mv(owner: MailboxId) -> KanbanMove {
        KanbanMove {
            mailbox: owner,
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
            witness_chain_position: 7,
            exec: ExecTarget::Elixir,
        }
    }

    /// An ARTIFACT cast (non-empty canonical payload).
    fn artifact(cycle: u64, sp: u64, owner: MailboxId, row: u64) -> SweepSlot {
        SweepSlot {
            cycle: CycleId(cycle),
            stream_position: sp,
            owner,
            row,
            paired_move: Some(mv(owner)),
            payload: witness(sp as u8),
        }
    }

    /// An INTENT-ONLY cast (empty payload — the `restage_held` shape).
    fn intent(cycle: u64, sp: u64, owner: MailboxId, row: u64) -> SweepSlot {
        SweepSlot {
            payload: Vec::new(),
            ..artifact(cycle, sp, owner, row)
        }
    }

    /// F1 + F13: zero artifact-backed delta → ZERO Lance operations, ZERO
    /// version, and no dataset is even created; the store survives restart as
    /// "empty", and a later real cycle still commits at V1.
    #[tokio::test]
    async fn no_artifact_delta_writes_nothing_and_creates_no_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(w.head(), DatasetVersion(0));

        // Thousands of pure kanban steps / held intents: nothing durable.
        let intents: Vec<SweepSlot> = (0..2_000u64).map(|i| intent(1, i, 42, i % 7)).collect();
        let out = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            intents,
        )
        .await
        .unwrap();
        assert_eq!(
            out,
            CommitOutcome::NoChange {
                head: DatasetVersion(0)
            }
        );
        assert_eq!(w.head(), DatasetVersion(0), "no version was minted");
        assert!(
            Dataset::open(path.to_str().unwrap()).await.is_err(),
            "the dataset was never even created"
        );

        // Restart: still empty, and a real artifact cycle commits at V1.
        drop(w); // the registry enforces one live writer per path
        let mut w2 = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        assert!(w2.timeline().await.unwrap().is_empty());
        let out = persist_cycle(
            &mut w2,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 5)],
        )
        .await
        .unwrap();
        assert!(
            matches!(
                out,
                CommitOutcome::Committed {
                    version: DatasetVersion(1),
                    ..
                }
            ),
            "{out:?}"
        );
    }

    /// F2 + the measured bytes-written falsifier: 64 transient breaths on ONE
    /// row cost exactly ONE 512-byte durable image row — not 64 × 512.
    #[tokio::test]
    async fn sixty_four_breaths_on_one_row_cost_one_image_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();

        // One thought, 64 successive artifact updates to the SAME row.
        let casts: Vec<SweepSlot> = (0..64u64).map(|i| artifact(1, i, 42, 9)).collect();
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            casts,
        )
        .await
        .unwrap();

        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let image = reopened.scan_image(CycleId(1)).await.unwrap();
        assert_eq!(image.len(), 1, "exactly ONE durable row image");
        assert_eq!(
            image[&9],
            witness(63),
            "the LAST breath survived (later stream position wins)"
        );
        let durable_payload_bytes: usize = image.values().map(Vec::len).sum();
        assert_eq!(
            durable_payload_bytes,
            EPISODIC_WITNESS_BYTES,
            "512 durable payload bytes, not 64 x 512 = {}",
            64 * EPISODIC_WITNESS_BYTES
        );
    }

    /// F3-adjacent + F10: a successful commit performs ZERO reopens, and the
    /// caller references the returned version instead of reloading state.
    #[tokio::test]
    async fn successful_commit_reopens_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let opens_after_startup = w.opens();

        for c in 1..=3u64 {
            let out = persist_cycle(
                &mut w,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![artifact(c, c, 42, c)],
            )
            .await
            .unwrap();
            assert!(matches!(out, CommitOutcome::Committed { .. }));
        }
        assert_eq!(
            w.opens(),
            opens_after_startup,
            "no reopen on the normal commit path"
        );
        assert_eq!(
            w.reconcile_scans(),
            0,
            "ZERO reconciliation scans on the fresh-monotonic normal path — measured, not asserted"
        );
        assert_eq!(
            w.head(),
            DatasetVersion(3),
            "the head token tracks in memory"
        );
    }

    /// The one-writer topology is ENFORCED in-process: a second live writer on
    /// the same path is refused at open; dropping the first frees the path.
    /// (Cross-process exclusivity remains a deployment lease — documented.)
    #[tokio::test]
    async fn a_second_live_writer_on_the_same_path_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let w1 = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let second = LanceCycleWriter::open(path.to_str().unwrap()).await;
        assert!(
            second.is_err(),
            "two live writers over one store must be unrepresentable in-process"
        );
        drop(w1);
        let w3 = LanceCycleWriter::open(path.to_str().unwrap()).await;
        assert!(w3.is_ok(), "dropping the writer frees the path");
    }

    /// Restart reconciliation stays cheap AND correct: a re-submitted batch
    /// after restart reconciles (one scan), while fresh cycles keep the
    /// scan-free path.
    #[tokio::test]
    async fn restart_resubmission_reconciles_with_one_scan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        {
            let mut w = LanceCycleWriter::open(path.to_str().unwrap())
                .await
                .unwrap();
            persist_cycle(
                &mut w,
                CycleFrame::new(CycleId(1), DatasetVersion(0)),
                vec![artifact(1, 0, 42, 1)],
            )
            .await
            .unwrap();
        }
        // Restart (lost acknowledgement): same frozen batch re-submitted.
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let retry = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 1)],
        )
        .await
        .unwrap();
        assert!(
            matches!(retry, CommitOutcome::Reconciled { .. }),
            "{retry:?}"
        );
        assert_eq!(w.reconcile_scans(), 1, "exactly one reconciliation scan");
        // A fresh cycle afterwards is scan-free again.
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            vec![artifact(2, 1, 42, 2)],
        )
        .await
        .unwrap();
        assert_eq!(w.reconcile_scans(), 1, "the fresh cycle added no scan");
    }

    /// F12 + F19: the recovery tail is BOUNDED — `after_cycle` excludes earlier
    /// history, and landing reads never touch the payload column.
    #[tokio::test]
    async fn bounded_tail_recovery_reads_no_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        for c in 1..=4u64 {
            persist_cycle(
                &mut w,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![artifact(c, c, 42, c)],
            )
            .await
            .unwrap();
        }
        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let tail = reopened.scan_sealed(Some(CycleId(2))).await.unwrap();
        assert_eq!(
            tail.iter().map(|l| l.cycle).collect::<Vec<_>>(),
            vec![CycleId(3), CycleId(4)],
            "strictly after the bound"
        );
        assert!(
            tail.iter().all(|l| l.slot.payload.is_empty()),
            "landing reads carry no payload — the column was never projected"
        );
        assert!(
            tail.iter().all(|l| l.slot.paired_move.is_some()),
            "but the transition metadata IS there (recovery needs the moves)"
        );
    }

    /// F11: the timeline is frame-only and payload-free, and survives restart.
    #[tokio::test]
    async fn timeline_is_frame_metadata_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        for c in 1..=2u64 {
            persist_cycle(
                &mut w,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![artifact(c, c, 42, c)],
            )
            .await
            .unwrap();
        }
        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let frames = reopened.timeline().await.unwrap();
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.cycle, f.base_version))
                .collect::<Vec<_>>(),
            vec![
                (CycleId(1), DatasetVersion(0)),
                (CycleId(2), DatasetVersion(1)),
            ]
        );
        assert!(frames.iter().all(|f| f.batch_hash != 0));
    }

    /// F10 + F12 (the reconciliation half): re-submitting the SAME frozen batch
    /// after a "lost acknowledgement" reconciles to exactly one conclusion — no
    /// duplicate rows, no second version, and NO delete anywhere.
    #[tokio::test]
    async fn resubmitting_the_same_batch_reconciles_to_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let casts = || vec![artifact(1, 0, 42, 1), artifact(1, 1, 42, 2)];

        let first = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            casts(),
        )
        .await
        .unwrap();
        assert!(matches!(
            first,
            CommitOutcome::Committed {
                version: DatasetVersion(1),
                ..
            }
        ));

        // The response was lost; the caller retries the identical batch.
        let retry = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            casts(),
        )
        .await
        .unwrap();
        assert!(
            matches!(
                retry,
                CommitOutcome::Reconciled {
                    cycle: CycleId(1),
                    ..
                }
            ),
            "{retry:?}"
        );
        assert_eq!(w.head(), DatasetVersion(1), "no second version");

        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            reopened.scan_sealed(None).await.unwrap().len(),
            2,
            "no duplicate landings after restart"
        );
        assert_eq!(reopened.timeline().await.unwrap().len(), 1, "one frame");
    }

    /// F10 (the fail-closed half): a DIFFERENT batch for a durable cycle is
    /// refused loudly and writes nothing.
    #[tokio::test]
    async fn a_conflicting_batch_for_a_durable_cycle_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 1)],
        )
        .await
        .unwrap();
        let conflict = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 9, 42, 1)],
        )
        .await;
        assert!(
            matches!(
                conflict,
                Err(lance_graph_planner::persist_sink::PersistError::Commit(
                    CommitError::HashConflict {
                        cycle: CycleId(1),
                        ..
                    }
                ))
            ),
            "{conflict:?}"
        );
        assert_eq!(w.head(), DatasetVersion(1), "nothing was written");
    }

    /// F11-adjacent: a stale horizon is FENCED with the current head and
    /// writes nothing (never a delete, never a silent accept).
    #[tokio::test]
    async fn a_stale_horizon_is_fenced_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 1)],
        )
        .await
        .unwrap();
        let stale = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(0)),
            vec![artifact(2, 1, 42, 2)],
        )
        .await;
        assert!(
            matches!(
                stale,
                Err(lance_graph_planner::persist_sink::PersistError::Commit(
                    CommitError::Fenced {
                        current_head: DatasetVersion(1)
                    }
                ))
            ),
            "{stale:?}"
        );
        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(reopened.timeline().await.unwrap().len(), 1);
        assert_eq!(reopened.scan_sealed(None).await.unwrap().len(), 1);
    }

    /// F5 + F6: no persisted row carries a nonterminal-only cast. Intent-only
    /// casts (held work, pure `Continue`-style movement) leave NO trace, while
    /// artifact casts in the same cycle persist normally.
    #[tokio::test]
    async fn intent_only_casts_leave_no_durable_trace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let mixed = vec![
            intent(1, 0, 42, 1),
            artifact(1, 1, 42, 2),
            intent(1, 2, 99, 3),
            intent(1, 3, 99, 4),
        ];
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            mixed,
        )
        .await
        .unwrap();

        drop(w); // a restart means the prior writer is GONE (the registry enforces it)
        let reopened = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let sealed = reopened.scan_sealed(None).await.unwrap();
        assert_eq!(sealed.len(), 1, "only the artifact cast persisted");
        assert_eq!(sealed[0].slot.stream_position, 1);
        assert_eq!(reopened.scan_image(CycleId(1)).await.unwrap().len(), 1);
    }

    /// F15: randomized completion order yields the same durable result —
    /// identical batch hash, identical landings, identical image.
    #[tokio::test]
    async fn randomized_completion_order_yields_the_same_durable_set() {
        let ordered = vec![
            artifact(1, 0, 1, 10),
            artifact(1, 1, 2, 11),
            artifact(1, 2, 3, 12),
        ];
        let scrambled = vec![
            artifact(1, 2, 3, 12),
            artifact(1, 0, 1, 10),
            artifact(1, 1, 2, 11),
        ];

        let mut hashes = Vec::new();
        let mut images = Vec::new();
        for casts in [ordered, scrambled] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("cycles.lance");
            let mut w = LanceCycleWriter::open(path.to_str().unwrap())
                .await
                .unwrap();
            persist_cycle(
                &mut w,
                CycleFrame::new(CycleId(1), DatasetVersion(0)),
                casts,
            )
            .await
            .unwrap();
            drop(w); // a restart means the prior writer is GONE (the registry enforces it)
            let reopened = LanceCycleWriter::open(path.to_str().unwrap())
                .await
                .unwrap();
            hashes.push(reopened.timeline().await.unwrap()[0].batch_hash);
            let sealed = reopened.scan_sealed(None).await.unwrap();
            assert_eq!(
                sealed
                    .iter()
                    .map(|l| l.slot.stream_position)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2],
                "stored in canonical order regardless of arrival"
            );
            images.push(reopened.scan_image(CycleId(1)).await.unwrap());
        }
        assert_eq!(
            hashes[0], hashes[1],
            "same conclusion set → same batch hash"
        );
        assert_eq!(images[0], images[1], "same durable image");
    }

    /// The ABI gate bites on an ARTIFACT payload (and cannot bite on an
    /// intent-only cast, which never reaches the writer).
    #[tokio::test]
    async fn a_malformed_artifact_payload_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let mut bad = artifact(1, 0, 42, 1);
        bad.payload = vec![7u8; 511];
        let r = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![bad],
        )
        .await;
        assert!(r.is_err(), "511 bytes must be refused: {r:?}");
        assert_eq!(w.head(), DatasetVersion(0), "nothing written");
    }

    // ── FALSIFIER (post-#912): the ABI refusal is PERMANENT, never Io ────────
    // A 511-byte payload must surface as `InvalidArtifact` — reporting it as
    // retryable Io would send the driver into fail → regenerate-identically →
    // fail, forever. Nothing may touch the store.
    #[tokio::test]
    async fn a_malformed_artifact_is_refused_permanently_not_as_retryable_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        let mut bad = artifact(1, 0, 42, 0);
        bad.payload.truncate(511);
        let err = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![bad],
        )
        .await
        .expect_err("a malformed artifact must be refused");
        let lance_graph_planner::persist_sink::PersistError::Commit(commit) = err else {
            panic!("expected a commit-layer refusal, got {err:?}");
        };
        assert_eq!(
            commit,
            CommitError::InvalidArtifact { row: 0, len: 511 },
            "the refusal is the PERMANENT variant, never CommitError::Io"
        );
        assert_eq!(w.head(), DatasetVersion(0), "nothing was written");
        // The identical batch keeps failing identically — no retry loop exit.
        let again = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![{
                let mut b = artifact(1, 0, 42, 0);
                b.payload.truncate(511);
                b
            }],
        )
        .await
        .expect_err("still refused");
        assert!(matches!(
            again,
            lance_graph_planner::persist_sink::PersistError::Commit(
                CommitError::InvalidArtifact { .. }
            )
        ));
    }

    // ── FALSIFIER (post-#912): store identity is lexical, not string-equal ───
    #[tokio::test]
    async fn a_second_spelling_of_the_same_store_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_str().unwrap().to_string();
        let w = LanceCycleWriter::open(format!("{base}/cycles.lance"))
            .await
            .unwrap();
        for spelling in [
            format!("{base}/./cycles.lance"),
            format!("{base}//cycles.lance"),
            format!("{base}/cycles.lance/"),
        ] {
            let err = LanceCycleWriter::open(spelling.clone())
                .await
                .expect_err("an alternate spelling of a live store must be refused");
            assert!(
                err.to_string().contains("already owns"),
                "{spelling}: {err}"
            );
        }
        drop(w);
        // The paired silence: a DIFFERENT store is never refused.
        let _other = LanceCycleWriter::open(format!("{base}/other.lance"))
            .await
            .expect("a distinct store opens freely");
    }

    // ── FALSIFIER (post-#912): a failed open releases its claim (RAII) ───────
    // The claim is held from before the first await; an `open` that errors
    // (here: a path that exists but is not a dataset and not NotFound-shaped)
    // must not leave the slot reserved.
    #[tokio::test]
    async fn a_failed_open_leaves_no_leaked_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        // First open succeeds (empty store), then drops.
        drop(
            LanceCycleWriter::open(path.to_str().unwrap())
                .await
                .unwrap(),
        );
        // Same path opens again — the drop released the claim.
        drop(
            LanceCycleWriter::open(path.to_str().unwrap())
                .await
                .unwrap(),
        );
        // A DIFFERENT path whose open FAILS (a file where a directory is
        // expected) must release too: retrying it is refused for the same
        // I/O reason, never with "already owns".
        let bogus = dir.path().join("not-a-dataset");
        std::fs::write(&bogus, b"junk").unwrap();
        let bogus_file = bogus.join("cycles.lance");
        let p = bogus_file.to_str().unwrap();
        let e1 = LanceCycleWriter::open(p).await;
        if let Ok(w) = e1 {
            // Environment-dependent: object-store may report NotFound here,
            // making this a legal empty store — then the claim path is
            // already covered by the success/drop halves above.
            drop(w);
            return;
        }
        let e2 = LanceCycleWriter::open(p)
            .await
            .expect_err("still the underlying I/O failure");
        assert!(
            !e2.to_string().contains("already owns"),
            "the failed first open must have released its claim: {e2}"
        );
    }

    // ── FALSIFIERS (post-#912): the ambiguous-append branch, deterministically ─
    // The branch carries the whole no-rollback contract; real Lance cannot be
    // made to fail on demand, so the TestFaults seam injects each arm.
    // (The HashConflict arm of THIS branch — append fails while the same cycle
    // is durable with different content — requires a competing writer between
    // fence and append, unrepresentable under the registry; it shares its
    // constructor with the normal-path HashConflict falsifier.)

    /// Arm `Ok(None)`: append failed with provably NOTHING published →
    /// `Io("nothing published")`, and the SAME frozen batch then commits.
    #[tokio::test]
    async fn injected_unpublished_append_error_is_refused_io_and_regenerable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        // A durable first cycle so the store exists.
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 0)],
        )
        .await
        .unwrap();
        let head = w.head();

        w.fault
            .fail_append_unpublished
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let err = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), head),
            vec![artifact(2, 1, 42, 1)],
        )
        .await
        .expect_err("the injected append failure surfaces");
        let lance_graph_planner::persist_sink::PersistError::Commit(CommitError::Io(io)) = &err
        else {
            panic!("expected Io(nothing published), got {err:?}");
        };
        assert!(
            io.to_string().contains("nothing published"),
            "the reconcile proved absence: {io}"
        );
        assert_eq!(w.head(), head, "the store is untouched");

        // Safe to regenerate: the same cycle now commits cleanly.
        let out = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), head),
            vec![artifact(2, 1, 42, 1)],
        )
        .await
        .unwrap();
        assert!(matches!(out, CommitOutcome::Committed { .. }));
    }

    /// Arm `Ok(Some(hash ==))`: the append PUBLISHED but the acknowledgement
    /// was lost → the same call reconciles to the durable identity, appending
    /// nothing twice. THE no-rollback falsifier.
    #[tokio::test]
    async fn injected_published_append_error_reconciles_to_the_durable_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 0)],
        )
        .await
        .unwrap();
        let head = w.head();

        w.fault
            .fail_append_published
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let out = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), head),
            vec![artifact(2, 1, 42, 1)],
        )
        .await
        .expect("the lost acknowledgement reconciles WITHIN the same call");
        let CommitOutcome::Reconciled {
            cycle,
            current_head,
            ..
        } = out
        else {
            panic!("expected Reconciled, got {out:?}");
        };
        assert_eq!(cycle, CycleId(2));
        assert_eq!(current_head, w.head());
        // Exactly one durable frame for cycle 2 — nothing double-appended.
        let frames = w.timeline().await.unwrap();
        assert_eq!(
            frames.iter().filter(|f| f.cycle == CycleId(2)).count(),
            1,
            "no rollback, no duplicate — the publication stands once"
        );
    }

    /// Arm `Err(reconcile)`: append outcome unknown AND the reconciliation
    /// read fails → `Ambiguous`; the SAME frozen batch resolves it later.
    #[tokio::test]
    async fn injected_reconcile_read_failure_after_append_error_is_ambiguous() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.lance");
        let mut w = LanceCycleWriter::open(path.to_str().unwrap())
            .await
            .unwrap();
        persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![artifact(1, 0, 42, 0)],
        )
        .await
        .unwrap();
        let head = w.head();

        w.fault
            .fail_append_unpublished
            .store(true, std::sync::atomic::Ordering::Relaxed);
        w.fault
            .fail_reconcile_read
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let err = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), head),
            vec![artifact(2, 1, 42, 1)],
        )
        .await
        .expect_err("append unknown + reconcile down = Ambiguous");
        let lance_graph_planner::persist_sink::PersistError::Commit(CommitError::Ambiguous {
            cycle,
            cause,
            ..
        }) = &err
        else {
            panic!("expected Ambiguous, got {err:?}");
        };
        assert_eq!(*cycle, CycleId(2));
        assert!(
            cause.contains("append failed") && cause.contains("reconciliation failed"),
            "both halves named: {cause}"
        );
        // Resolution: the SAME frozen batch — reconciliation-first proves it
        // absent (nothing had published) and the append lands once.
        let out = persist_cycle(
            &mut w,
            CycleFrame::new(CycleId(2), head),
            vec![artifact(2, 1, 42, 1)],
        )
        .await
        .expect("re-submission resolves the ambiguity");
        assert!(matches!(out, CommitOutcome::Committed { .. }));
    }
}
