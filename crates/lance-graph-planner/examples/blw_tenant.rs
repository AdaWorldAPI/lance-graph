//! `blw_tenant` — **D-BLW-1**: the Bible arm on the REAL post-#879 substrate.
//!
//! **ONE tenant. Verses are ROWS inside it.** An owner is a *tenant*, not a
//! shard (`E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`, plan §12.1a′): one
//! `MailboxSoA` = one mailbox = one kanban board. This harness constructs
//! **exactly one** `MailboxSoA` and never a second one — the corpus is seated as
//! rows of that single owner, and "N thoughts at once" is data-parallelism over
//! **rows inside the owner's slice**.
//!
//! ## Why this file exists
//!
//! The two prior BLW harnesses (`blw_lens_twin.rs`, `blw_texture.rs`) returned
//! **0** for the substrate grep
//! (`batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope`)
//! and were therefore free-standing loops that could not be evidence for any
//! substrate claim, however green (plan §12.7, defect 2). This harness runs on
//! the shipped surfaces:
//!
//! | what | surface consumed |
//! |---|---|
//! | the tenant | `cognitive_shader_driver::mailbox_soa::MailboxSoA<N>` (`mailbox_soa.rs:58`) |
//! | the only row mutator | `MailboxSoA::write_row` — cycle-aware (`mailbox_soa.rs:417`) |
//! | the read lens | `MailboxSoaView::{identity_plane_at, energy, n_rows}` (`soa_view.rs:67`) |
//! | the pre-write cast | `owner_adapter::emit_bootstrap_intent` → `BatchWriter::cast` (`owner_adapter.rs:92`, `batch_writer.rs:104`) |
//! | the seal | `persist_sink::persist_cycle` — N casts → ONE WAL append → ONE version (`persist_sink.rs:335`) |
//! | the post-seal apply | `persist_sink::recover_and_apply` → `MailboxSoaOwner::try_advance_phase` (`persist_sink.rs:396`, `:430`) |
//! | the lifecycle guard | `MailboxSoaOwner::try_advance_phase` direct (`soa_view.rs:311`) |
//!
//! ## The §4 trap — the applied step is THE PAIRED MOVE, never the scheduler's
//!
//! `owner_adapter.rs`'s module doc: *"never manufacture a generic
//! `next_phases().first()` transition merely because some version appeared."*
//! `NextPhaseScheduler` is right there and is **not** the applier. Cycle 3 makes
//! that falsifiable rather than merely stated: from `Evaluation` the scheduler's
//! forward arc is `Commit`, and the harness casts `Evaluation → Plan`. If the
//! applier ever fabricated the scheduler's default, the assertion in
//! [`PROBE-TRAP`](fn.main.html) fails. That is the whole point of the cycle.
//!
//! ## What this harness does NOT claim
//!
//! - **No durability.** `MemWal` is in-process `Mutex`/`Vec`, mirroring
//!   `persist_sink`'s own `FakeWalSink`. Per that module's own header:
//!   *"`compile+test green ≠ storage proven`"*. **No concrete Lance sink exists
//!   anywhere in the tree** and this file does not add one. The version numbers
//!   here are sequence numbers, not Lance versions.
//! - **No `deinterlace` / `DeinterlaceRow` read.** There is still no production
//!   `DeinterlaceRow` implementor and no production caller of `deinterlace`
//!   (`batch_writer.rs` module doc). The durability *observation* seam is
//!   **not** wired here — see the report's "seam I stopped at".
//! - **No stance instrument.** The row body is a deterministic bloom-containment
//!   read over the content identity plane, deliberately NOT a Hegel/Nietzsche/
//!   Kant/Wittgenstein projection: §12.3c retired κ and §12.7 recorded the
//!   texture rewrite as a KILL. **D-BLW-1 is the substrate deliverable**; the
//!   instrument is D-BLW-2's problem and is not smuggled in here.
//!
//! ## The memory figure, and what struct it is a figure OF
//!
//! The canonical row is **512 B** (`NODE_ROW_STRIDE`, const-asserted
//! `size_of::<NodeRow>() == 512`), so 2,048 canonical rows = **1 MiB**. The
//! byte image this harness snapshots is a figure of **`MailboxSoA<2048>`**, not
//! of `NodeRow`: its content/topic/angle identity planes alone are
//! `3 × 2048 × 256 × 8 B` = **12 MiB** (6,144 B/row, 12× the canon). That
//! divergence is the open question `ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` and
//! is **not** resolved here — it is named so no figure below is silently read as
//! a figure of the canonical node row.
//!
//! ## Run
//!
//! ```text
//! cargo run -p lance-graph-planner --example blw_tenant            # 2,000 verses (default)
//! cargo run -p lance-graph-planner --example blw_tenant -- 512     # bounded
//! BLW_KJV_TSV=/path/to/kjv_verses.tsv cargo run ... --example blw_tenant
//! ```
//!
//! The corpus is **bounded on purpose**: the full 31,102-verse run of the prior
//! harness exceeded a 10-minute budget (§12.7). Default 2,000; the cap is the
//! tenant capacity `N_CAP`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
use lance_graph_contract::soa_view::{IdentityPlane, MailboxSoaOwner, MailboxSoaView};
use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
use lance_graph_planner::persist_sink::{
    persist_cycle, recover_and_apply, CommitError, CommitOutcome, CycleFrame, CycleId,
    DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot, WalSink, WriteFailed,
};
use lance_graph_planner::traits::StrategyOutcome;

// ── the ONE tenant ──────────────────────────────────────────────────────────

/// Row capacity of the single tenant. `MailboxSoA<N>` is const-generic, so the
/// capacity is a type-level constant; the *logical* row count is declared with
/// `set_populated` and read back through `MailboxSoaView::n_rows`.
const N_CAP: usize = 2048;

/// Default bounded corpus size (§12.7: the unbounded run blew a 10-min budget).
const DEFAULT_VERSES: usize = 2_000;

/// The tenant's mailbox id. Deliberately **non-zero**: `mailbox 0` is the
/// bootstrap sentinel (`owner_adapter::BOOTSTRAP_OWNER`), so a zero id here
/// would make the rebind assertion vacuous.
const TENANT_ID: MailboxId = 7;

/// The tenant's 6-bit witness slot (`w_slot < 64` is asserted by the ctor).
const TENANT_W_SLOT: u8 = 7;

/// Firing threshold handed to the ctor. Unused by this harness's read path
/// (`consume_firing` is not exercised — the harness never delivers batons).
const TENANT_THRESHOLD: f32 = 1.0;

/// The single tenant type. **One** of these is constructed, ever.
type Tenant = MailboxSoA<N_CAP>;

// ── the byte image (the falsifier's instrument) ─────────────────────────────

/// Bytes of the per-row fixed columns in the image: energy(4) + plasticity(1) +
/// last_active_cycle(4) + last_write_cycle(4) + edge(8) + qualia(8) + meta(4) +
/// entity_type(2) + temporal(8) + expert(2) + sigma(1).
const FIXED_COLS: usize = 4 + 1 + 4 + 4 + 8 + 8 + 4 + 2 + 8 + 2 + 1;

/// Bytes of the three autopoiesis-triangle style lanes (12 atoms each).
const STYLE_LANES: usize = 12 * 3;

/// Bytes of one identity plane (`WORDS_PER_FP` u64).
const PLANE_BYTES: usize = WORDS_PER_FP * 8;

/// Total bytes one row contributes to the image.
const ROW_IMG: usize = FIXED_COLS + STYLE_LANES + 3 * PLANE_BYTES;

/// Bytes of the tenant-level scalar head: mailbox_id(4) + w_slot(1) +
/// current_cycle(4) + phase(1) + populated(8) + stale_write_count(8) +
/// threshold(4).
const SCALAR_IMG: usize = 4 + 1 + 4 + 1 + 8 + 8 + 4;

/// Total image length — asserted at runtime so a column silently dropped from
/// [`snapshot`] cannot pass as "byte-identical" (the exact defect the previous
/// arm shipped: a 6-column snapshot calling itself a full comparison).
const IMAGE_LEN: usize = SCALAR_IMG + N_CAP * ROW_IMG;

/// A **complete** little-endian byte image of the tenant's backing store:
/// every tenant scalar, then every per-row column of **every capacity row**
/// (`0..N_CAP`, not `0..populated` — a mutation to a padding row must be
/// visible too), including all three identity planes and all three style lanes.
///
/// This is the falsifier's instrument. It reads only `&self`.
fn snapshot(o: &Tenant) -> Vec<u8> {
    let mut b = Vec::with_capacity(IMAGE_LEN);

    // ── tenant scalars ──
    b.extend_from_slice(&o.mailbox_id.to_le_bytes());
    b.push(o.w_slot);
    b.extend_from_slice(&o.current_cycle.to_le_bytes());
    b.push(o.phase() as u8);
    b.extend_from_slice(&(o.populated() as u64).to_le_bytes());
    b.extend_from_slice(&o.stale_write_count().to_le_bytes());
    b.extend_from_slice(&o.threshold.to_bits().to_le_bytes());

    // Hoisted zero-copy column borrows (all are capacity-length `N_CAP`).
    let energy = o.energy();
    let edges = o.edges_raw();
    let meta = o.meta_raw();
    let etype = o.entity_type();

    // Driven off `energy`'s iterator (length `N_CAP`) rather than a bare index
    // range, so the loop covers every capacity row by construction.
    for (row, e) in energy.iter().enumerate() {
        b.extend_from_slice(&e.to_bits().to_le_bytes());
        b.push(o.plasticity_counter[row]);
        b.extend_from_slice(&o.last_active_cycle[row].to_le_bytes());
        b.extend_from_slice(&o.last_write_cycle[row].to_le_bytes());
        b.extend_from_slice(&edges[row].to_le_bytes());
        b.extend_from_slice(&o.qualia[row].0.to_le_bytes());
        b.extend_from_slice(&meta[row].to_le_bytes());
        b.extend_from_slice(&etype[row].to_le_bytes());
        b.extend_from_slice(&o.temporal[row].to_le_bytes());
        b.extend_from_slice(&o.expert[row].to_le_bytes());
        b.push(o.sigma[row]);
        b.extend_from_slice(&o.frozen_style[row]);
        b.extend_from_slice(&o.learned_style[row]);
        b.extend_from_slice(&o.explore_style[row]);
        for w in o.content_row(row) {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for w in o.topic_row(row) {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for w in o.angle_row(row) {
            b.extend_from_slice(&w.to_le_bytes());
        }
    }
    b
}

/// Byte offset of the first difference, or `None` when the images are identical.
/// Never `assert_eq!`s the images themselves — they are ~12 MiB and a failure
/// message must be an offset, not a memory dump.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Human-readable location of an image offset — so a can-fire result names the
/// column it detected, not just "they differ".
fn locate(off: usize) -> String {
    if off < SCALAR_IMG {
        return format!("tenant scalar head (+{off})");
    }
    let rel = off - SCALAR_IMG;
    let row = rel / ROW_IMG;
    let f = rel % ROW_IMG;
    let region = if f < FIXED_COLS {
        "fixed columns"
    } else if f < FIXED_COLS + STYLE_LANES {
        "style lanes"
    } else if f < FIXED_COLS + STYLE_LANES + PLANE_BYTES {
        "CONTENT plane"
    } else if f < FIXED_COLS + STYLE_LANES + 2 * PLANE_BYTES {
        "TOPIC plane"
    } else {
        "ANGLE plane"
    };
    format!("row {row}, {region} (+{f} in row)")
}

/// The image slice belonging to one row.
fn row_image(img: &[u8], row: usize) -> &[u8] {
    let lo = SCALAR_IMG + row * ROW_IMG;
    &img[lo..lo + ROW_IMG]
}

// ── the row body: a deterministic bloom read (NOT a stance instrument) ──────

/// Bits set per token in a 16,384-bit identity plane.
const BLOOM_K: usize = 4;

/// FNV-1a over `bytes`, salted with `seed`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ seed.wrapping_mul(0x100_0000_01b3);
    for &c in bytes {
        h ^= u64::from(c);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Set this token's `BLOOM_K` bits in a `WORDS_PER_FP`-word plane.
fn bloom_add(plane: &mut [u64], token: &str, salt: u64) {
    for k in 0..BLOOM_K {
        let h = fnv1a(
            token.as_bytes(),
            salt ^ (k as u64).wrapping_mul(0x9E37_79B9),
        );
        let bit = (h % (WORDS_PER_FP as u64 * 64)) as usize;
        plane[bit / 64] |= 1u64 << (bit % 64);
    }
}

/// Lowercased alphanumeric tokens of length ≥ 2.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_ascii_lowercase)
}

/// Build a plane from a verse's tokens.
fn encode_plane(text: &str, salt: u64) -> Vec<u64> {
    let mut plane = vec![0u64; WORDS_PER_FP];
    for t in tokens(text) {
        bloom_add(&mut plane, &t, salt);
    }
    plane
}

/// Build a probe plane from a single term.
fn probe_plane(term: &str, salt: u64) -> Vec<u64> {
    let mut plane = vec![0u64; WORDS_PER_FP];
    bloom_add(&mut plane, term, salt);
    plane
}

/// The aggregate a read sweep produces — an owned `Copy` microcopy, never a
/// borrow into the SoA (`data-flow.md` category 2).
#[derive(Debug, Clone, Copy)]
struct Sweep {
    /// Rows visited (bounded by `n_rows()`, the declared logical size).
    scanned: usize,
    /// Rows whose content plane contains every probe bit.
    fired: usize,
    /// Mean per-row overlap fraction against the probe, in `[0,1]`.
    mean_similarity: f32,
    /// Mean per-row energy — reported, not used as a gate (the harness delivers
    /// no `CausalEdge64` batons, so `apply_edges` never runs).
    mean_energy: f32,
}

/// Evaluate **every** row through the read-only view.
///
/// Reads are **borrowed row slices** into the tenant's backing store
/// (`identity_plane_at`, `energy`); the reasoning is on owned `Copy`
/// microcopies; nothing is written. The `&V` receiver is the structural
/// guarantee: **no `&mut self` during computation** (`data-flow.md`).
fn sweep_rows<V: MailboxSoaView>(view: &V, probe: &[u64]) -> (Sweep, Vec<u32>) {
    let n_rows = view.n_rows();
    let energy = view.energy();
    let probe_bits: u32 = probe.iter().map(|w| w.count_ones()).sum();

    let mut fired = Vec::new();
    let mut sim_acc = 0.0f64;
    let mut energy_acc = 0.0f64;
    let mut scanned = 0usize;

    for (row, row_energy) in energy.iter().enumerate().take(n_rows) {
        // BORROWED slice into the backing store — zero-copy (data-flow §1).
        let Some(plane) = view.identity_plane_at(row, IdentityPlane::Content) else {
            continue;
        };
        // From here on: owned Copy microcopies only (data-flow §2).
        let overlap: u32 = probe
            .iter()
            .zip(plane)
            .map(|(p, w)| (p & w).count_ones())
            .sum();
        scanned += 1;
        energy_acc += f64::from(*row_energy);
        if probe_bits > 0 {
            sim_acc += f64::from(overlap) / f64::from(probe_bits);
            if overlap == probe_bits {
                fired.push(row as u32);
            }
        }
    }

    let denom = if scanned == 0 { 1.0 } else { scanned as f64 };
    (
        Sweep {
            scanned,
            fired: fired.len(),
            mean_similarity: (sim_acc / denom) as f32,
            mean_energy: (energy_acc / denom) as f32,
        },
        fired,
    )
}

// ── the write descriptor `P` (DTO purity) ───────────────────────────────────

/// The `BatchWriter` payload — a **descriptor** (dirty row range + cycle), never
/// owned delta bytes (`batch_writer.rs` Addendum-6: the sink reads the LIVE
/// store at flush).
///
/// **It carries NO owner / mailbox / tenant field.** Ownership rides the *cast
/// pairing* (`BatchWriter::on_behalf_of`), never the DTO — the write-on-behalf
/// iron rule. The harness asserts this pairing rather than restating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowSpanDescriptor {
    /// First dirty row (inclusive).
    row_lo: u32,
    /// Last dirty row (exclusive).
    row_hi: u32,
    /// The owner cycle the span belongs to.
    cycle: u32,
}

impl RowSpanDescriptor {
    /// The descriptor's own LE bytes — what a landing carries in place of a
    /// `NodeRowPacket` slice (`SweepSlot::payload`: *"a descriptor … bytes here"*).
    fn to_le_bytes(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.row_lo.to_le_bytes());
        out[4..8].copy_from_slice(&self.row_hi.to_le_bytes());
        out[8..12].copy_from_slice(&self.cycle.to_le_bytes());
        out
    }
}

// ── the WAL seam (in-process; NOT durability) ───────────────────────────────

/// One sealed cycle, as this in-process sink holds it.
struct SealedCycle {
    /// The cycle identity.
    cycle: CycleId,
    /// The sealed predecessor this cycle's thoughts read (`Vn`).
    base_version: DatasetVersion,
    /// The batch's durable idempotency hash.
    batch_hash: u64,
    /// Landings AS COMMITTED — already write-side ordered before the append.
    landings: Vec<SweepSlot>,
    /// Distinct rows in the coalesced final image.
    image_rows: usize,
}

/// An in-process `WalSink`, mirroring `persist_sink`'s own `FakeWalSink`
/// (including its optimistic-concurrency fence on the frame's `base_version`
/// and its reconciliation-first `(cycle, batch_hash)` lookup).
///
/// **This proves the CONTRACT (one append per cycle, stored order, sealed read
/// horizon), NOT durability.** There is no WAL, no restart, no manifest, and no
/// Lance table anywhere in this file.
struct MemWal {
    sealed: Mutex<Vec<SealedCycle>>,
    /// The store's physical head version (0 = empty store).
    head: AtomicU64,
    /// Physical appends — must be exactly ONE per committed cycle.
    wal_writes: AtomicU64,
}

impl MemWal {
    fn new() -> Self {
        Self {
            sealed: Mutex::new(Vec::new()),
            head: AtomicU64::new(0),
            wal_writes: AtomicU64::new(0),
        }
    }
    fn wal_writes(&self) -> u64 {
        self.wal_writes.load(Ordering::SeqCst)
    }
    fn head(&self) -> DatasetVersion {
        DatasetVersion(self.head.load(Ordering::SeqCst))
    }
    fn image_rows_of(&self, cycle: CycleId) -> Option<usize> {
        self.sealed
            .lock()
            .expect("MemWal poisoned")
            .iter()
            .find(|s| s.cycle == cycle)
            .map(|s| s.image_rows)
    }
}

impl WalSink for MemWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        let mut sealed = self.sealed.lock().expect("MemWal poisoned");
        // Reconciliation-first: an already-durable (cycle, hash) is success, a
        // matching cycle with a different hash fails closed.
        if let Some(rec) = sealed.iter().find(|s| s.cycle == batch.frame.cycle) {
            return if rec.batch_hash == batch.batch_hash {
                Ok(CommitOutcome::Reconciled {
                    current_head: DatasetVersion(self.head.load(Ordering::SeqCst)),
                    cycle: batch.frame.cycle,
                    batch_hash: batch.batch_hash,
                })
            } else {
                Err(CommitError::HashConflict {
                    cycle: batch.frame.cycle,
                    stored_hash: rec.batch_hash,
                    offered_hash: batch.batch_hash,
                })
            };
        }
        // Epistemic horizon: a commit must target the current sealed head.
        let head = DatasetVersion(self.head.load(Ordering::SeqCst));
        if batch.frame.base_version != head {
            return Err(CommitError::Fenced { current_head: head });
        }
        // THE single amortized append for the whole cycle.
        self.wal_writes.fetch_add(1, Ordering::SeqCst);
        let version = DatasetVersion(self.head.fetch_add(1, Ordering::SeqCst) + 1);
        let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
        sealed.push(SealedCycle {
            cycle,
            base_version: batch.frame.base_version,
            batch_hash,
            image_rows: batch.image.len(),
            landings: batch.landings,
        });
        Ok(CommitOutcome::Committed {
            version,
            cycle,
            batch_hash,
        })
    }

    async fn scan_sealed(
        &self,
        after_cycle: Option<CycleId>,
    ) -> Result<Vec<LandedSlot>, WriteFailed> {
        // Returned in STORED order — never re-sorted (order was fixed at seal).
        Ok(self
            .sealed
            .lock()
            .expect("MemWal poisoned")
            .iter()
            .filter(|s| after_cycle.is_none_or(|c| s.cycle > c))
            .flat_map(|s| {
                s.landings.iter().map(|slot| LandedSlot {
                    cycle: s.cycle,
                    // `scan_sealed` is payload-free by contract; payloads
                    // live in the image read.
                    slot: SweepSlot {
                        payload: Vec::new(),
                        ..slot.clone()
                    },
                })
            })
            .collect())
    }

    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        Ok(self
            .sealed
            .lock()
            .expect("MemWal poisoned")
            .iter()
            .map(|s| FrameMeta {
                cycle: s.cycle,
                base_version: s.base_version,
                batch_hash: s.batch_hash,
            })
            .collect())
    }
}

// ── corpus ──────────────────────────────────────────────────────────────────

/// Read `index\ttext` rows, bounded to `limit`.
fn load_verses(path: &str, limit: usize) -> std::io::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(_, t)| t.to_string()))
        .take(limit)
        .collect())
}

/// Seat the verses as ROWS of the one tenant.
///
/// This is the **builder** phase, not a compute path — `data-flow.md` allows
/// `&mut` for a builder. Row columns go through `write_row`, the SoA's ONE
/// cycle-aware mutator; `energy` is initialised through its public column
/// because `WriteCell` carries no energy field (production energy arrives via
/// `apply_edges` from `CausalEdge64` batons, which this harness has no source
/// for — so the sweep reports energy rather than gating on it).
///
/// The **angle** plane is deliberately left all-zero: it is the silent region
/// the can-fire twin later mutates, which is the sharpest possible test that
/// [`snapshot`] genuinely reads the identity planes (the previous arm's
/// snapshot did not).
fn seed_tenant(owner: &mut Tenant, verses: &[String]) -> usize {
    let cycle = owner.cycle();
    let mut seated = 0usize;
    for (row, text) in verses.iter().enumerate().take(N_CAP) {
        let content = encode_plane(text, 0);
        let topic = encode_plane(text, 0xA5A5_A5A5);
        let cell = WriteCell {
            content: Some(content.as_slice()),
            topic: Some(topic.as_slice()),
            entity_type: Some((row % 251) as u16),
            temporal: Some(row as u64),
            meta: Some(MetaWord((text.len() as u32) & 0x00FF_FFFF)),
            ..WriteCell::default()
        };
        if owner.write_row(row, cycle, &cell) == WriteOutcome::Accepted {
            // Builder-phase direct column init (see fn doc).
            owner.energy[row] = (text.len() as f32) * 0.01;
            seated += 1;
        }
    }
    owner.set_populated(seated);
    seated
}

// ── the cycle ───────────────────────────────────────────────────────────────

/// The bootstrap lifecycle intent a strategy SURFACES: `mailbox 0`,
/// `witness_chain_position 0` — the zero-fallback sentinel `rebind_bootstrap`
/// recognises. `from`/`to`/`exec` are preserved bit-for-bit by the rebind.
fn bootstrap_intent(from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
    KanbanMove {
        mailbox: 0,
        from,
        to,
        witness_chain_position: 0,
        exec: ExecTarget::Elixir,
    }
}

/// Apply the gated write-back to the fired rows ONLY, through `write_row`.
/// Returns how many landings the gate accepted.
fn stamp_fired_rows(owner: &mut Tenant, fired: &[u32], stamp: u64) -> usize {
    let cycle = owner.cycle();
    let mut accepted = 0usize;
    for &row in fired {
        let cell = WriteCell {
            temporal: Some(stamp),
            ..WriteCell::default()
        };
        if owner.write_row(row as usize, cycle, &cell) == WriteOutcome::Accepted {
            accepted += 1;
        }
    }
    accepted
}

/// Everything one sealed cycle needs from its caller.
struct CycleSpec {
    /// Cycle identity.
    id: CycleId,
    /// The lifecycle arc this cycle's thought INTENDS.
    from: KanbanColumn,
    /// …and where it intends to go.
    to: KanbanColumn,
}

// ── main ────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("BLW_KJV_TSV").unwrap_or_else(|_| "/tmp/kjv_verses.tsv".to_string());
    let limit = std::env::args()
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(DEFAULT_VERSES)
        .min(N_CAP);

    println!("== D-BLW-1 — ONE tenant, verses as ROWS, on the post-#879 substrate ==");
    println!("corpus       : {path} (bounded to {limit})");

    let verses = load_verses(&path, limit)?;
    if verses.is_empty() {
        return Err(format!("no verses parsed from {path} (expected `index\\ttext` rows)").into());
    }

    // ── THE ONE TENANT. Constructed once; never a second owner in this file. ──
    let mut owner: Tenant = MailboxSoA::new(TENANT_ID, TENANT_W_SLOT, TENANT_THRESHOLD);
    let seated = seed_tenant(&mut owner, &verses);
    // Move the cycle stamp off 0 so the bootstrap-cycle rebind (0 → live) is
    // non-vacuous when it is asserted below.
    owner.tick();

    println!(
        "tenant       : mailbox {} w_slot {} — 1 owner, {} rows seated of {} capacity",
        owner.mailbox_id(),
        owner.w_slot,
        owner.n_rows(),
        N_CAP
    );
    println!(
        "image        : {} B of MailboxSoA<{}> (3 identity planes = {} B/row; canon NodeRow = 512 B/row \
         → 2048 canon rows would be {} B). ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON, unresolved.",
        IMAGE_LEN,
        N_CAP,
        3 * PLANE_BYTES,
        N_CAP * 512
    );
    assert!(seated > 0, "no rows seated — nothing to evaluate");
    assert_eq!(
        owner.n_rows(),
        seated,
        "declared logical size == seated rows"
    );

    // ── image anti-vacuity: the instrument must actually cover the store ──
    let base_img = snapshot(&owner);
    assert_eq!(
        base_img.len(),
        IMAGE_LEN,
        "the byte image must cover every column of every capacity row"
    );
    let nonzero = base_img.iter().filter(|b| **b != 0).count();
    // Scale-free coverage gate: each seated row contributes ~250 non-zero bytes
    // (two bloom planes + the fixed columns), so `> seated * 8` holds at ANY
    // corpus size while still failing loudly for an image that is effectively
    // all-zero — where "byte-identical" would be trivially true. A percentage
    // gate would have been tuned to one corpus size.
    assert!(
        nonzero > seated * 8,
        "image is near-uniformly zero ({nonzero} non-zero of {} over {seated} rows) — \
         'byte-identical' would be trivially true",
        base_img.len()
    );
    println!(
        "instrument   : {} B image, {nonzero} non-zero bytes ({:.2}%)",
        base_img.len(),
        100.0 * nonzero as f64 / base_img.len() as f64
    );

    // ── PROBE-GUARD: an illegal Rubicon edge is refused AND mutates nothing ──
    // `try_advance_phase` is the checked mutator; Planning → Evaluation is an
    // illegal skip. Can-fire (the guard barks) + can-stay-silent (no mutation).
    let before_guard = snapshot(&owner);
    let refused = owner.try_advance_phase(KanbanColumn::Evaluation);
    let after_guard = snapshot(&owner);
    assert!(
        refused.is_err(),
        "Planning → Evaluation must be refused by the Rubicon DAG"
    );
    assert_eq!(
        first_diff(&before_guard, &after_guard),
        None,
        "a refused transition must not mutate the tenant"
    );
    println!(
        "PROBE-GUARD  : illegal edge refused ({}), tenant byte-identical — OK",
        refused.unwrap_err()
    );

    // ── PROBE-LENS: the row body discriminates (can-fire AND can-stay-silent) ──
    let probe = probe_plane("god", 0);
    let (hit_sweep, _) = sweep_rows(&owner, &probe);
    let silent_probe = probe_plane("zzqxjvw", 0);
    let (silent_sweep, silent_rows) = sweep_rows(&owner, &silent_probe);
    // BOTH halves, unconditionally: something fires and something does not. This
    // is what makes the "untouched remainder is byte-identical" check below
    // non-vacuous — an empty remainder would prove nothing.
    assert!(
        hit_sweep.fired > 0 && hit_sweep.fired < hit_sweep.scanned,
        "the lens must fire on some rows and NOT on others: {hit_sweep:?}"
    );
    assert!(
        silent_rows.is_empty(),
        "an absent term must fire nothing: {silent_sweep:?}"
    );
    // The stronger SPARSENESS claim is gated on corpus size, and the gate is
    // stated rather than tuned: a short prefix of Genesis 1 is ~90% "God", so a
    // 10-verse run legitimately fires on almost every row. Asserting sparseness
    // there would be an assertion about Genesis 1, not about the filter.
    if hit_sweep.scanned >= 512 {
        assert!(
            hit_sweep.fired * 2 < hit_sweep.scanned,
            "over a corpus this size the dirty set must be a SPARSE minority: {hit_sweep:?}"
        );
    }
    println!(
        "PROBE-LENS   : probe 'god' fires {}/{} ({:.1}%), absent term fires {} — discriminating",
        hit_sweep.fired,
        hit_sweep.scanned,
        100.0 * hit_sweep.fired as f64 / hit_sweep.scanned as f64,
        silent_sweep.fired
    );

    // ── the sealed cycles ────────────────────────────────────────────────────
    let mut sink = MemWal::new();
    let mut writer: BatchWriter<RowSpanDescriptor> = BatchWriter::new();
    let mut watermark: Option<u64> = None;
    let mut stream_position: u64 = 0;

    let plan = [
        CycleSpec {
            id: CycleId(1),
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
        },
        CycleSpec {
            id: CycleId(2),
            from: KanbanColumn::CognitiveWork,
            to: KanbanColumn::Evaluation,
        },
        // The §4 TRAP cycle: the scheduler's forward arc from Evaluation is
        // `Commit`; the thought casts `Plan`. The applier must honour the CAST.
        CycleSpec {
            id: CycleId(3),
            from: KanbanColumn::Evaluation,
            to: KanbanColumn::Plan,
        },
    ];

    // `explicit_counter_loop` is a FALSE POSITIVE here and its suggested
    // rewrite would be wrong. `stream_position` is not a per-iteration loop
    // counter: it advances once per FIRED ROW inside the landing `map` below
    // (hundreds per cycle) and once more for the tenant-level landing, so it is
    // a monotonic position in the witness stream across the whole run.
    // `(0_u64..).zip(plan.iter())` would reset it to the cycle index and
    // silently change what the stream positions mean.
    #[expect(
        clippy::explicit_counter_loop,
        reason = "stream_position is a witness-stream position advanced per landing, not a loop counter"
    )]
    for spec in &plan {
        assert_eq!(owner.phase(), spec.from, "cycle {:?} precondition", spec.id);
        let owner_cycle = owner.cycle();

        // ── ① THE CENTRAL FALSIFIER — evaluating ALL rows mutates NOTHING ──
        let pre_eval = snapshot(&owner);
        let (sweep, fired) = sweep_rows(&owner, &probe);
        let post_eval = snapshot(&owner);
        if let Some(off) = first_diff(&pre_eval, &post_eval) {
            panic!(
                "FALSIFIER FAILED: evaluating {} rows mutated the tenant at byte {off} — {}",
                sweep.scanned,
                locate(off)
            );
        }
        assert!(
            sweep.scanned == owner.n_rows() && sweep.scanned > 0,
            "the sweep must visit every logical row (non-vacuous silence): {sweep:?}"
        );

        // ── ② gated write-back — ONLY the sparse dirty set advances ──
        let accepted = stamp_fired_rows(&mut owner, &fired, u64::from(owner_cycle) << 32);
        assert_eq!(accepted, fired.len(), "every gated write must be Accepted");
        let post_write = snapshot(&owner);

        // Anti-vacuity (§12.3 D-BLW-1 falsifier): the untouched remainder is
        // byte-identical, and the touched set genuinely changed. BOTH halves.
        let mut changed = 0usize;
        let mut leaked = Vec::new();
        for row in 0..N_CAP {
            let differs = row_image(&pre_eval, row) != row_image(&post_write, row);
            let is_dirty = fired.binary_search(&(row as u32)).is_ok();
            if differs {
                changed += 1;
            }
            if differs != is_dirty {
                leaked.push(row);
            }
        }
        assert!(
            leaked.is_empty(),
            "only the sealed sparse set may advance; divergent rows: {:?}",
            &leaked[..leaked.len().min(8)]
        );
        assert_eq!(changed, fired.len(), "changed rows == dirty rows");
        assert!(
            changed > 0,
            "a write-back that changes nothing proves nothing"
        );

        // ── ③ the pre-write cast — write-on-behalf, ahead of the write ──
        let outcome = StrategyOutcome {
            reliability: sweep.mean_similarity,
            intended_move: Some(bootstrap_intent(spec.from, spec.to)),
        };
        let span = RowSpanDescriptor {
            row_lo: fired.first().copied().unwrap_or(0),
            row_hi: fired.last().copied().unwrap_or(0) + 1,
            cycle: owner_cycle,
        };
        // The owner id is READ FROM THE OWNER at the call site, so the cast
        // cannot name a mailbox other than the SoA it writes.
        let cast =
            emit_bootstrap_intent(&outcome, owner.mailbox_id(), owner_cycle, &mut writer, span)
                .expect("a bootstrap sentinel must rebind and cast");

        // Ownership rides the CAST PAIRING, never the DTO. Every landing below
        // takes its owner from HERE, not from `RowSpanDescriptor` (which has no
        // owner field at all).
        let cast_owner = writer
            .on_behalf_of(cast)
            .expect("the cast is recorded on the writer");
        assert_eq!(
            cast_owner,
            owner.mailbox_id(),
            "write-on-behalf of the live owner"
        );
        let cast_move = {
            let moves = writer
                .intent_moves(cast)
                .expect("the cast recorded its intent");
            assert_eq!(moves.len(), 1, "one lifecycle intent per cast");
            moves[0]
        };
        // Anti-vacuity on the rebind: the sentinel fields ACTUALLY moved.
        assert_ne!(cast_move.mailbox, 0, "owner sentinel was rebound");
        assert_eq!(cast_move.mailbox, owner.mailbox_id());
        assert_ne!(
            cast_move.witness_chain_position, 0,
            "cycle sentinel rebound"
        );
        assert_eq!(cast_move.witness_chain_position, owner_cycle);
        assert_eq!((cast_move.from, cast_move.to), (spec.from, spec.to));

        // ── ④ the seal — N landings → ONE append → ONE version ──
        let mut slots: Vec<SweepSlot> = fired
            .iter()
            .map(|&row| {
                stream_position += 1;
                SweepSlot {
                    cycle: spec.id,
                    stream_position,
                    owner: cast_owner,
                    row: u64::from(row),
                    // A ROW landing carries NO lifecycle move: the kanban board
                    // belongs to the TENANT, not to a row. One tenant = one
                    // board, so exactly ONE landing per cycle carries the step.
                    paired_move: None,
                    payload: RowSpanDescriptor {
                        row_lo: row,
                        row_hi: row + 1,
                        cycle: owner_cycle,
                    }
                    .to_le_bytes()
                    .to_vec(),
                }
            })
            .collect();
        stream_position += 1;
        slots.push(SweepSlot {
            cycle: spec.id,
            stream_position,
            owner: cast_owner,
            row: 0,
            paired_move: Some(cast_move),
            payload: span.to_le_bytes().to_vec(),
        });
        let n_landings = slots.len();

        let appends_before = sink.wal_writes();
        let base = sink.head();
        let outcome = persist_cycle(&mut sink, CycleFrame::new(spec.id, base), slots).await?;
        let CommitOutcome::Committed { version, .. } = outcome else {
            panic!(
                "cycle {:?}: expected a fresh Committed outcome (every cycle id in this \
                 harness's plan is used exactly once), got {outcome:?}",
                spec.id
            );
        };
        assert_eq!(
            sink.wal_writes() - appends_before,
            1,
            "{n_landings} landings → exactly ONE WAL append"
        );

        // ── ⑤ THE TRAP — what the scheduler WOULD have proposed, recorded now ──
        let scheduler_would = NextPhaseScheduler
            .on_version(&owner, version, ExecTarget::Native)
            .map(|m| m.to);

        // ── ⑥ the post-seal apply — THE PAIRED MOVE, via try_advance_phase ──
        // `after_cycle` bounds the read to cycles strictly after the PREVIOUS
        // cycle in this plan (cycle ids are the 1-indexed plan position), which
        // is exactly this cycle's own newly-sealed landings — the cycle-keyed
        // equivalent of the old `scan_sealed(Some(base))` version-keyed bound.
        let after_cycle = (spec.id.0 > 1).then(|| CycleId(spec.id.0 - 1));
        let sealed = sink.scan_sealed(after_cycle).await?;
        let pre_apply = snapshot(&owner);
        let recovered = recover_and_apply(&mut owner, &sealed, watermark).map_err(|(_, e)| e)?;
        watermark = recovered.watermark;
        let post_apply = snapshot(&owner);

        assert_eq!(
            recovered.applied.len(),
            1,
            "one tenant, one board → exactly one lifecycle step per cycle"
        );
        let applied = recovered.applied[0];
        assert_eq!(
            (applied.from, applied.to),
            (cast_move.from, cast_move.to),
            "the applied step must be THE PAIRED MOVE that was cast"
        );
        assert_eq!(
            owner.phase(),
            spec.to,
            "the tenant advanced to the cast target"
        );

        // The lifecycle step is a TENANT scalar — it must touch no row bytes.
        let apply_diff = first_diff(&pre_apply, &post_apply)
            .expect("applying a phase step must change the image");
        assert!(
            apply_diff < SCALAR_IMG,
            "the kanban step must move only tenant scalars, not row bytes — moved {}",
            locate(apply_diff)
        );

        println!(
            "cycle {:>2}     : {} rows swept (mean sim {:.4}, mean energy {:.4}), {} dirty, \
             {} landings → 1 append → {:?}; image_rows {:?}; applied {:?}→{:?} (cast), \
             scheduler would have said {:?}",
            spec.id.0,
            sweep.scanned,
            sweep.mean_similarity,
            sweep.mean_energy,
            fired.len(),
            n_landings,
            version,
            sink.image_rows_of(spec.id),
            applied.from,
            applied.to,
            scheduler_would
        );

        // ── the §4 trap, made falsifiable on the one cycle where they differ ──
        if spec.id == CycleId(3) {
            assert_eq!(
                scheduler_would,
                Some(KanbanColumn::Commit),
                "NextPhaseScheduler's forward arc from Evaluation is Commit"
            );
            assert_eq!(applied.to, KanbanColumn::Plan);
            assert_ne!(
                Some(applied.to),
                scheduler_would,
                "PROBE-TRAP: the applier honoured the CAST move, not the scheduler's default"
            );
            println!(
                "PROBE-TRAP   : scheduler proposed {:?}, cast said {:?}, applied {:?} — \
                 the paired move won",
                scheduler_would.expect("checked above"),
                cast_move.to,
                applied.to
            );
        }

        owner.tick();
    }

    // ── THE CAN-FIRE TWIN — the comparator must DETECT deliberate mutation ──
    // A guard that cannot bark is the defect one level up. Two probes, chosen
    // so a snapshot that skipped a region cannot pass:
    //   (a) a small fixed column (`meta`);
    //   (b) the ANGLE identity plane, which is all-zero for the entire run —
    //       a snapshot that never reads it would report "identical".
    let cycle_now = owner.cycle();

    let pre_mut = snapshot(&owner);
    let victim = seated / 2;
    let bump = WriteCell {
        meta: Some(MetaWord(0x00AB_CDEF)),
        ..WriteCell::default()
    };
    assert_eq!(
        owner.write_row(victim, cycle_now, &bump),
        WriteOutcome::Accepted
    );
    let post_mut = snapshot(&owner);
    let d_fixed = first_diff(&pre_mut, &post_mut)
        .expect("CAN-FIRE FAILED: a deliberate fixed-column mutation went undetected");
    let where_fixed = locate(d_fixed);
    // The first byte to move is the `last_write_cycle` stamp the gate itself
    // writes, ahead of `meta` in the row image — both are fixed columns.
    assert!(
        where_fixed.contains("fixed columns"),
        "the detected difference must be in the fixed columns, got {where_fixed}"
    );
    println!(
        "PROBE-MUT-a  : gated one-column write to row {victim} detected at byte {d_fixed} — {where_fixed}"
    );

    let pre_mut2 = snapshot(&owner);
    let mut poisoned = vec![0u64; WORDS_PER_FP];
    poisoned[WORDS_PER_FP - 1] = 1; // one bit, in the last word of the plane
    let bump2 = WriteCell {
        angle: Some(poisoned.as_slice()),
        ..WriteCell::default()
    };
    assert_eq!(
        owner.write_row(victim, cycle_now, &bump2),
        WriteOutcome::Accepted
    );
    let post_mut2 = snapshot(&owner);
    let d_angle = first_diff(&pre_mut2, &post_mut2)
        .expect("CAN-FIRE FAILED: a one-bit ANGLE-plane mutation went undetected");
    let where_angle = locate(d_angle);
    assert!(
        where_angle.contains("ANGLE"),
        "the detected difference must be in the ANGLE plane, got {where_angle}"
    );
    println!(
        "PROBE-MUT-b  : 1-bit ANGLE-plane mutation detected at byte {d_angle} — {where_angle}"
    );

    // ── summary ──────────────────────────────────────────────────────────────
    let frames = sink.timeline().await?;
    println!("--");
    println!(
        "tenants      : 1 (never N) — mailbox {}, final phase {:?}, cycle {}",
        owner.mailbox_id(),
        owner.phase(),
        owner.cycle()
    );
    println!(
        "seal         : {} cycles → {} versions, {} WAL appends, watermark {:?}",
        plan.len(),
        frames.len(),
        sink.wal_writes(),
        watermark
    );
    println!(
        "casts        : {} recorded on the BatchWriter",
        writer.casts().len()
    );
    println!(
        "NOT PROVEN   : durability (MemWal is in-process), `deinterlace`/`DeinterlaceRow` \
         (no production implementor exists), and any stance/semantic claim."
    );
    Ok(())
}
