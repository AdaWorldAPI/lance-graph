//! `blw_fusion` — **D-BLW-3**: the Horizontverschmelzung fusion falsifier.
//!
//! Spec: `.claude/board/exec-runs/dblw3-design-opus.md` (design, Opus,
//! design-only, no code) + `.claude/board/exec-runs/dblw3-api-inventory-sonnet.md`
//! (exact signatures — wins on any design/inventory conflict). Re-scoped per
//! the design's own §0/B1: the kanban-plan D3 wording — **two ontology
//! projections of one cohort, over the tenant's own row data** — not the
//! four-stance framing (dead 3× over, see design §0).
//!
//! **This is the FIRST `DeinterlaceRow` implementor and the FIRST
//! `deinterlace` caller anywhere in the tree** (design B6). It reuses
//! `blw_tenant.rs`'s ONE-TENANT / `MailboxSoA` / seal-sequence pattern
//! (`persist_cycle` / `recover_and_apply`, `E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`)
//! — pieces are COPIED here, not imported from an example, with provenance
//! comments at each site — and adds the P1/P2 prerequisites the design's
//! §1.4 proves the shipped harness lacks: **incremental seating** (the
//! ranking pool must grow) and a **horizon-relative rank criterion** (so a
//! verdict can actually move between horizons).
//!
//! ## The measurement, in one line
//!
//! Two rank-based binary projections (SIGHT-basin `A`, BODY-basin `B`) over
//! the tenant's content identity plane, read TWICE at ONE pin `V_pin = V4`
//! under `Strict` (a-priori, rung 0) vs `Aware` (hindsight, rung 5) — the
//! ONLY thing that varies is what the reader is *permitted* to know, never
//! the pin, the row set, or the code path (design §5.1). An inert
//! containment control `Z` proves the plumbing contributes zero movement
//! (design §2.4, gate G2).
//!
//! ## Eight adversarial corrections applied on top of the design (C1-C8)
//!
//! Each is cited at its site below with a `// C<n>:` comment.
//!
//! - **C1** — extensional identity asserted THREE ways (G1c): `at(V4,5)`
//!   (Aware), `at(V4,9)` (Retro), and `at(V8,0)` (Strict) all return the
//!   IDENTICAL row sequence, because `knowable_from` is constant at `0` and
//!   `V8` is the max horizon present — "one function, three names."
//! - **C2** — every emitted row's `knowable_from()` is asserted CONSTANT
//!   and `<= V_pin` (G3), not merely documented as such.
//! - **C3** — no substrate-exercise claim: `deinterlace` reduces to
//!   `filter(v <= ref) + stable sort` on this corpus; `Unknowable` /
//!   `DependsClosure` / HLC are inert. Printed in the §6 not-claimed block.
//! - **C4** — fold-order bug fix: `sort_by_key` is STABLE and three
//!   projections share the `(v, v)` sort key, so a subject-only fold across
//!   ALL projections would return an emission-order artifact. `fold_last_by_subject`
//!   filters to ONE projection FIRST.
//! - **C5** — signed churn (gained/lost) reported BESIDE Hamming, never
//!   averaged into `Δκ`.
//! - **C6** — the pre-registered null is printed BEFORE any result.
//! - **C7** — the drop test runs over ALL EIGHT horizons, not just `V_pin`.
//! - **C8** — every κ ships the FULL `BinaryAssociation` table; degeneracy
//!   is stamped BEFORE pairing (§3.2a ordering).
//!
//! ## A ninth correction, delivered mid-build by the coordinator
//!
//! **G6's per-subject row count is NOT uniform across the fixed prefix.**
//! The design's literal G6 text ("every subject in the fixed prefix ... 8
//! Aware rows ... 4 Strict rows") is wrong for slices 2-4: the fixed prefix
//! (1000 verses) spans FOUR seating slices (250 verses each), not one, and
//! §1.5's own emission rule ("a verse seated in slice `s` has verdict rows
//! at horizons `Vs..V8`") makes the count a function of `s`. Corrected here
//! as `Aware = 9 - s`, `Strict = 5 - s` rows, verified arithmetically in the
//! doc comment on [`seating_slice`] before being coded as an assertion.
//!
//! ## What this harness does NOT claim
//!
//! See the `§6 not-claimed` block printed at the end of `main`, plus the
//! honest blockers B1-B9 in the design note. In short: no validity claim
//! (only reliability), no p-value, no stance claim, no cross-language claim,
//! no zero-copy claim, no durability claim, no HLC claim, no substrate-
//! exercise claim beyond the rank criterion (C3).
//!
//! ## Run
//!
//! ```text
//! cargo run -p lance-graph-planner --example blw_fusion
//! BLW_KJV_TSV=/path/to/kjv_verses.tsv cargo run -p lance-graph-planner --example blw_fusion
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
use jc::stats::{binary_association, BinaryAssociation};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::{IdentityPlane, MailboxSoaView};
use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
use lance_graph_planner::persist_sink::{
    persist_cycle, recover_and_apply, CommitError, CommitOutcome, CycleFrame, CycleId,
    DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot, WalSink, WriteFailed,
};
use lance_graph_planner::temporal::{
    deinterlace, DeinterlaceRow, LanceVersion, NoDeps, QueryReference,
};
use lance_graph_planner::traits::StrategyOutcome;

// ── PRE-REGISTERED run shape (design §3.6) — fixed BEFORE any number exists,
// NON-ADJUSTABLE after a κ is computed. Changing any of these after seeing a
// result would be fitting, not measuring. ─────────────────────────────────

/// Row capacity of the single tenant (matches `blw_tenant.rs:107`'s `N_CAP`).
const N_CAP: usize = 2048;

/// Total corpus verses. PRE-REGISTERED §3.6 (`DEFAULT_VERSES` equivalent).
const M_VERSES: usize = 2_000;

/// Sealed cycles. PRE-REGISTERED §3.6 — two full loops of the Rubicon DAG
/// (`Planning -> CognitiveWork -> Evaluation -> Plan -> Planning -> ...`,
/// `kanban.rs:101-109`, `Plan => &[Planning]` at `:106` — VERIFIED before
/// this file was written).
const S_CYCLES: usize = 8;

/// Verses seated per cycle. PRE-REGISTERED §3.6.
const SLICE: usize = M_VERSES / S_CYCLES; // 250

/// The fixed prefix — complete after cycle `V_PIN_CYCLE`. PRE-REGISTERED §3.6.
const K_FIXED_PREFIX: usize = 1_000;

/// The 1-indexed cycle whose SEALED VERSION is `V_pin`. PRE-REGISTERED §3.6.
const V_PIN_CYCLE: usize = 4;

/// a-priori rung (`EpistemicMode::Strict`, `temporal.rs:91-97`). §5.1.
const RUNG_STRICT: u8 = 0;
/// hindsight rung (`EpistemicMode::Aware`). §5.1.
const RUNG_AWARE: u8 = 5;
/// G1c-only: the `Retro` rung, used solely for the extensional-identity check.
const RUNG_RETRO: u8 = 9;

/// The rank criterion's quantile (§2.3). Documentation constant — the actual
/// operationalization is `pool_size / 4` (integer floor), because `250 * q`
/// is not always an integer (e.g. `750 * 0.25 = 187.5`); floor via integer
/// division is the deterministic, documented choice. PRE-REGISTERED, NON-ADJUSTABLE.
const Q_QUANTILE: f64 = 0.25;

/// §3.1 fusion band — reused VERBATIM from §12.3a (the discrimination twin's
/// can-discriminate ceiling / can-agree floor). PRE-REGISTERED, NON-ADJUSTABLE.
const KAPPA_REDUNDANCY_FLOOR: f64 = 0.80;
/// PRE-REGISTERED, NON-ADJUSTABLE (§3.1).
const KAPPA_NO_SHARED_HORIZON_CEILING: f64 = 0.20;

/// §3.3 movement threshold — VERBATIM from §12.3b (one-fifth of the band span).
/// PRE-REGISTERED, NON-ADJUSTABLE.
const MOVEMENT_THRESHOLD: f64 = 0.10;
/// §3.4 drop threshold — VERBATIM from §12.3b (two-decimal κ reporting floor).
/// PRE-REGISTERED, NON-ADJUSTABLE.
const DROP_THRESHOLD: f64 = 0.01;

/// §3.5 degeneracy guard — VERBATIM from §12.3a. PRE-REGISTERED, NON-ADJUSTABLE.
const DEGENERACY_LOW: f64 = 0.01;
/// PRE-REGISTERED, NON-ADJUSTABLE (§3.5).
const DEGENERACY_HIGH: f64 = 0.99;
/// §3.5 unstable guard — VERBATIM from §12.3a. PRE-REGISTERED, NON-ADJUSTABLE.
const UNSTABLE_EXPECTED_AGREEMENT: f64 = 0.95;
/// §3.5 COLLAPSED guard fraction (`n01 + n10 < 0.05*N`). PRE-REGISTERED, NON-ADJUSTABLE.
const COLLAPSED_FRACTION: f64 = 0.05;

/// The tenant's mailbox id. Distinct from `blw_tenant.rs`'s `TENANT_ID = 7`
/// (separate binary, separate process — no actual collision risk, kept
/// distinct for clarity only).
const TENANT_ID: MailboxId = 13;
/// The tenant's witness slot (`w_slot < 64` asserted by the ctor).
const TENANT_W_SLOT: u8 = 13;
/// Unused by this harness's read path (mirrors `blw_tenant.rs:120-122`).
const TENANT_THRESHOLD: f32 = 1.0;

/// The single tenant type. **One** of these is constructed, ever
/// (`E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`).
type Tenant = MailboxSoA<N_CAP>;

// ── §2.2 the two projections (seed sets, provenance-only per design §2.2) ──

/// Projection A — the SIGHT basin.
const SEED_A: &[&str] = &[
    "eyes",
    "opened",
    "knew",
    "know",
    "saw",
    "see",
    "wise",
    "understanding",
];
/// Projection B — the BODY basin.
const SEED_B: &[&str] = &[
    "naked",
    "nakedness",
    "ashamed",
    "shame",
    "clothed",
    "covered",
    "garment",
    "skin",
];

/// A|B|Z — see design §2.2 (A, B) and §2.4 (Z, the inertness control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proj {
    A,
    B,
    Z,
}

// ── §1.1 the verdict row shape ──────────────────────────────────────────────

/// One (verse, horizon, projection) verdict. Emitted as the sealed series is
/// produced; never reconstructed from a version. Design §1.1.
#[derive(Clone, Debug)]
struct VerdictRow {
    /// The tenant row's stable verse reference, e.g. "kjv:00417".
    subject: String,
    /// The SEALED VERSION this verdict was COMPUTED FROM (design §1.2 — the
    /// storage-frame clock, `lance_version`; no competing reading).
    horizon: u64,
    /// A | B | Z.
    projection: Proj,
    /// The binary verdict itself.
    verdict: bool,
}

impl DeinterlaceRow for VerdictRow {
    fn subject(&self) -> &str {
        &self.subject
    }
    fn lance_version(&self) -> LanceVersion {
        self.horizon
    }
    /// CONSTANT `0` — a class-registration clock (design §1.2), NOT a
    /// per-row warrant time. This corpus has ONE verdict-row class, hence
    /// one registration event, hence one value for every row. C2 (below,
    /// `assert_knowable_from_is_constant`) verifies this at runtime rather
    /// than trusting the doc comment.
    fn knowable_from(&self) -> LanceVersion {
        0
    }
    // hlc_tick() DEFAULTED to `None` — design §1.3 (no HLC writer in this arm).
}

// ── row-body bloom machinery — COPIED from `blw_tenant.rs` with provenance,
// per the task brief ("consume the pattern; do not import from an example —
// copy the needed pieces, citing provenance"). Unchanged except `probe_plane`
// is generalized to `bloom_of_terms` (multi-term OR-bundle) for the seed sets. ─

/// Bits set per token in a 16,384-bit identity plane. Provenance: `blw_tenant.rs:249`.
const BLOOM_K: usize = 4;

/// FNV-1a over `bytes`, salted with `seed`. Provenance: `blw_tenant.rs:252-259`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ seed.wrapping_mul(0x100_0000_01b3);
    for &c in bytes {
        h ^= u64::from(c);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Set this token's `BLOOM_K` bits in a `WORDS_PER_FP`-word plane.
/// Provenance: `blw_tenant.rs:262-271`.
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

/// Lowercased alphanumeric tokens of length >= 2. Provenance: `blw_tenant.rs:274-278`.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_ascii_lowercase)
}

/// Build a plane from a verse's tokens. Provenance: `blw_tenant.rs:281-287`.
/// Deliberately drops the `topic` plane (§2.1: "a second hash of the same
/// data" — never read here, so never written).
fn encode_plane(text: &str, salt: u64) -> Vec<u64> {
    let mut plane = vec![0u64; WORDS_PER_FP];
    for t in tokens(text) {
        bloom_add(&mut plane, &t, salt);
    }
    plane
}

/// Build a bloom from several already-tokenized terms (an OR-bundle of the
/// SAME construction `encode_plane` uses, salt 0 — the salt the content
/// plane itself uses, so overlap/containment against it is meaningful).
/// Generalizes `blw_tenant.rs::probe_plane` (single term) to a seed SET.
fn bloom_of_terms(terms: &[&str]) -> Vec<u64> {
    let mut plane = vec![0u64; WORDS_PER_FP];
    for t in terms {
        bloom_add(&mut plane, t, 0);
    }
    plane
}

/// `score(i, j) = popcount(content_plane[i] & bloom(S_j))` (design §2.3.1).
/// The plane is a BORROWED row slice (`data-flow.md` §1); the score is an
/// OWNED `Copy` `u32` microcopy (§2). No `&mut self` during computation.
/// Provenance (pattern): `blw_tenant.rs::sweep_rows`, `blw_tenant.rs:317-346`.
fn score_row(owner: &Tenant, row: usize, seed: &[u64]) -> u32 {
    let plane = owner
        .identity_plane_at(row, IdentityPlane::Content)
        .expect("row must have a content plane (seeded before scoring)");
    seed.iter()
        .zip(plane)
        .map(|(s, w)| (s & w).count_ones())
        .sum()
}

/// `Z(i)`: does row `i`'s content plane CONTAIN every bit of `probe`? The
/// SAME containment read D-BLW-1 shipped (`blw_tenant.rs:333-345`,
/// `overlap == probe_bits`) — design §2.4.
fn contains_all(owner: &Tenant, row: usize, probe: &[u64]) -> bool {
    let plane = owner
        .identity_plane_at(row, IdentityPlane::Content)
        .expect("row must have a content plane (seeded before scoring)");
    let probe_bits: u32 = probe.iter().map(|w| w.count_ones()).sum();
    let overlap: u32 = probe
        .iter()
        .zip(plane)
        .map(|(p, w)| (p & w).count_ones())
        .sum();
    probe_bits > 0 && overlap == probe_bits
}

/// §2.3 the rank criterion: `verdict(i, j, V) = true` iff `score(i, j)` is
/// in the strict top `q` of the POOL as of `V` (every row seated by `V`),
/// ties broken by ASCENDING row index. `n_pos = pool_size / 4` is the floor
/// operationalization of `Q_QUANTILE` (see its doc comment). Returns
/// `verdict[row]` for `row` in `0..pool_size`, in ROW-INDEX order (not
/// sorted-score order) so it is directly usable for emission.
fn rank_verdicts(owner: &Tenant, pool_size: usize, seed: &[u64]) -> Vec<bool> {
    let mut scored: Vec<(u32, usize)> = (0..pool_size)
        .map(|row| (score_row(owner, row, seed), row))
        .collect();
    // Descending score, ties broken by ASCENDING row index (§2.3).
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // PRE-REGISTERED q = Q_QUANTILE, floor operationalization (see the const's
    // doc comment). f64 mult of a small usize by 0.25 is exact; the `as usize`
    // cast floors, matching the previous `pool_size / 4` integer division
    // bit-for-bit for every pool size this harness produces.
    let n_pos = (pool_size as f64 * Q_QUANTILE) as usize;
    let mut verdict = vec![false; pool_size];
    for &(_, row) in scored.iter().take(n_pos) {
        verdict[row] = true;
    }
    verdict
}

// ── the write descriptor `P` (DTO purity) — COPIED from `blw_tenant.rs:360-389` ──

/// The `BatchWriter` payload — a DESCRIPTOR, never owned delta bytes. Carries
/// NO owner/mailbox/tenant field — ownership rides the cast pairing, never
/// the DTO (the write-on-behalf iron rule). Provenance: `blw_tenant.rs:369-389`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowSpanDescriptor {
    row_lo: u32,
    row_hi: u32,
    cycle: u32,
}

impl RowSpanDescriptor {
    fn to_le_bytes(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.row_lo.to_le_bytes());
        out[4..8].copy_from_slice(&self.row_hi.to_le_bytes());
        out[8..12].copy_from_slice(&self.cycle.to_le_bytes());
        out
    }
}

// ── the WAL seam (in-process; NOT durability) — COPIED from `blw_tenant.rs:391-501` ──

/// One sealed cycle, as this in-process sink holds it. Provenance: `blw_tenant.rs:394-403`
/// (dropped `image_rows` — blw_tenant read it only through `image_rows_of`, a debug-print
/// helper this harness has no caller for; keeping a write-only field would be dead code).
struct SealedCycle {
    cycle: CycleId,
    base_version: DatasetVersion,
    batch_hash: u64,
    landings: Vec<SweepSlot>,
}

/// An in-process `WalSink`, mirroring `persist_sink`'s own `FakeWalSink` AND
/// `blw_tenant.rs`'s `MemWal` (`blw_tenant.rs:405-501`). Proves the CONTRACT
/// (one append per cycle, stored order, sealed read horizon), NOT durability.
struct MemWal {
    sealed: Mutex<Vec<SealedCycle>>,
    /// The store's physical head version (0 = empty store).
    head: AtomicU64,
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
                    version: DatasetVersion(self.head.load(Ordering::SeqCst)),
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
        let head = DatasetVersion(self.head.load(Ordering::SeqCst));
        if batch.frame.base_version != head {
            return Err(CommitError::Fenced { current_head: head });
        }
        self.wal_writes.fetch_add(1, Ordering::SeqCst);
        let version = DatasetVersion(self.head.fetch_add(1, Ordering::SeqCst) + 1);
        let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
        sealed.push(SealedCycle {
            cycle,
            base_version: batch.frame.base_version,
            batch_hash,
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
        Ok(self
            .sealed
            .lock()
            .expect("MemWal poisoned")
            .iter()
            .filter(|s| after_cycle.is_none_or(|c| s.cycle > c))
            .flat_map(|s| {
                s.landings.iter().map(|slot| LandedSlot {
                    cycle: s.cycle,
                    slot: slot.clone(),
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

// ── corpus + seeding — adapted from `blw_tenant.rs:503-550` for P1 (incremental) ──

/// Read `index\ttext` rows, bounded to `limit`. Provenance: `blw_tenant.rs:505-513`.
fn load_verses(path: &str, limit: usize) -> std::io::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(_, t)| t.to_string()))
        .take(limit)
        .collect())
}

/// Seat one INCREMENTAL slice of verses as rows `[row_lo, row_lo + verses.len())`.
///
/// **P1 (design §1.4), justification per the ⊘ correction: the ranking POOL
/// must grow — this is what makes a verdict horizon-dependent at all.** The
/// caller invokes this STRICTLY between the post-apply of cycle `c-1` and
/// the pre-eval of cycle `c` (see the loop in `main`), so `blw_tenant.rs`'s
/// "untouched remainder is byte-identical" property is preserved even though
/// this harness does not re-run that falsifier itself.
///
/// Adapted from `blw_tenant.rs::seed_tenant` (`blw_tenant.rs:528-550`):
/// drops the `topic` plane (§2.1, never read here) and the `angle` write
/// (no can-fire-twin instrument in this harness); keeps `content` (the ONLY
/// real per-verse signal, §2.1), the builder-phase direct `energy` init, and
/// `write_row`'s cycle-aware gating.
fn seed_slice(owner: &mut Tenant, row_lo: usize, verses: &[String]) -> usize {
    let cycle = owner.cycle();
    let mut seated = 0usize;
    for (i, text) in verses.iter().enumerate() {
        let row = row_lo + i;
        let content = encode_plane(text, 0);
        let cell = WriteCell {
            content: Some(content.as_slice()),
            entity_type: Some((row % 251) as u16),
            temporal: Some(row as u64),
            meta: Some(MetaWord((text.len() as u32) & 0x00FF_FFFF)),
            ..WriteCell::default()
        };
        if owner.write_row(row, cycle, &cell) == WriteOutcome::Accepted {
            owner.energy[row] = (text.len() as f32) * 0.01;
            seated += 1;
        }
    }
    assert_eq!(
        seated,
        verses.len(),
        "P1: every write in a slice must be Accepted — else row<->subject addressing desyncs"
    );
    seated
}

/// The bootstrap lifecycle intent a strategy SURFACES: `mailbox 0`,
/// `witness_chain_position 0` — the zero-fallback sentinel `rebind_bootstrap`
/// recognises. Provenance: `blw_tenant.rs:554-565`.
fn bootstrap_intent(from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
    KanbanMove {
        mailbox: 0,
        from,
        to,
        witness_chain_position: 0,
        exec: ExecTarget::Elixir,
    }
}

/// Everything one sealed cycle needs from its caller. Provenance: `blw_tenant.rs:584-592`.
struct CycleSpec {
    id: CycleId,
    from: KanbanColumn,
    to: KanbanColumn,
}

// ── analysis helpers ─────────────────────────────────────────────────────

/// Parse the numeric row index out of a `"kjv:NNNNN"` subject.
fn subject_index(subject: &str) -> usize {
    subject
        .trim_start_matches("kjv:")
        .parse()
        .expect("subject must be kjv:NNNNN")
}

/// The 1-indexed seating slice (== the cycle number) row `row` was FIRST
/// written in. `row 0..249 -> slice 1`, `250..499 -> slice 2`, etc.
///
/// **Arithmetic verified before coding, per the coordinator's correction:**
/// design §1.5 states a verse seated in slice `s` has verdict rows at
/// horizons `Vs..V8` and none before. That span has `8 - s + 1 = 9 - s`
/// elements. Under Aware@V4 (which — per C1/G1c — admits EVERY row
/// regardless of horizon, since `knowable_from = 0` is always `<= ref`) a
/// subject's ENTIRE `Vs..V8` span is admitted: `9 - s` rows. Under
/// Strict@V4 (admits only `horizon <= V4`), the admitted sub-span is
/// `Vs..V4`, which for `s <= 4` has `4 - s + 1 = 5 - s` elements. `s=1`:
/// 8/4. `s=2`: 7/3. `s=3`: 6/2. `s=4`: 5/1. Used by gate G6.
fn seating_slice(row: usize) -> usize {
    row / SLICE + 1
}

/// C4 fix: filter to ONE projection FIRST (three projections share the
/// `(v, v)` sort key and `sort_by_key` is STABLE — a subject-only fold
/// across mixed projections would return an emission-order artifact, not a
/// horizon-correct verdict). Then fold by subject taking the LAST row
/// (`deinterlace` already sorted ascending by horizon, `temporal.rs:369-374`,
/// so "last" == the verdict from the LATEST admitted horizon).
fn fold_last_by_subject(rows: &[VerdictRow], proj: Proj) -> Vec<(String, bool)> {
    let filtered: Vec<&VerdictRow> = rows.iter().filter(|r| r.projection == proj).collect();
    // C4 anti-vacuity: after filtering to ONE projection, (subject, horizon)
    // must be unique — a duplicate here means the emission loop is broken.
    let mut seen: HashSet<(&str, u64)> = HashSet::new();
    for r in &filtered {
        assert!(
            seen.insert((r.subject.as_str(), r.horizon)),
            "C4: duplicate (subject, horizon) row survived the projection filter for {proj:?}: {}@V{}",
            r.subject,
            r.horizon
        );
    }
    let mut last: BTreeMap<String, bool> = BTreeMap::new();
    for r in &filtered {
        // `filtered` preserves deinterlace's ascending-horizon order (a
        // stable filter over a stably-sorted input); each insert overwrites,
        // so the final value is the LATEST horizon's verdict.
        last.insert(r.subject.clone(), r.verdict);
    }
    last.into_iter().collect()
}

/// Restrict a folded vector to subjects `< prefix` (the §12.3b anti-sample-
/// growth control, applied here by gate G5).
fn restrict_to_prefix(folded: &[(String, bool)], prefix: usize) -> Vec<(String, bool)> {
    folded
        .iter()
        .filter(|(s, _)| subject_index(s) < prefix)
        .cloned()
        .collect()
}

fn bools_only(folded: &[(String, bool)]) -> Vec<bool> {
    folded.iter().map(|(_, b)| *b).collect()
}

/// Total Hamming distance between two aligned bool vectors.
fn hamming(a: &[bool], b: &[bool]) -> usize {
    assert_eq!(
        a.len(),
        b.len(),
        "hamming: vectors must be aligned same-length"
    );
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// C5: signed churn, gained (false->true) and lost (true->false) reported
/// SEPARATELY — a κ can be unchanged while verdicts churn in cancelling
/// directions, so `gained + lost == hamming` must never be collapsed back
/// into one signed-away number.
fn churn(before: &[bool], after: &[bool]) -> (usize, usize) {
    assert_eq!(before.len(), after.len());
    let mut gained = 0usize;
    let mut lost = 0usize;
    for (b, a) in before.iter().zip(after) {
        match (*b, *a) {
            (false, true) => gained += 1,
            (true, false) => lost += 1,
            _ => {}
        }
    }
    (gained, lost)
}

fn positive_rate(bools: &[bool]) -> f64 {
    if bools.is_empty() {
        return 0.0;
    }
    bools.iter().filter(|b| **b).count() as f64 / bools.len() as f64
}

/// §3.5 degeneracy guard — VERBATIM interval from §12.3a.
fn is_degenerate(rate: f64) -> bool {
    !(DEGENERACY_LOW..=DEGENERACY_HIGH).contains(&rate)
}

/// C8: print the FULL `BinaryAssociation` table, never a bare κ. κ as κ,
/// never "ICC"; φ as φ, never "Pearson" (§6 item 3 / C2 of the design).
fn print_association_table(label: &str, assoc: &BinaryAssociation) {
    let kappa_str = assoc.kappa.map_or_else(
        || format!("undefined(p_e={:.4})", assoc.expected_agreement),
        |k| format!("{k:.4}"),
    );
    let phi_str = assoc
        .phi
        .map_or_else(|| "undefined(constant)".to_string(), |p| format!("{p:.4}"));
    println!(
        "  {label}: n00={} n01={} n10={} n11={} | rate_a={:.4} rate_b={:.4} | p_o={:.4} p_e={:.4} | \
         kappa={kappa_str} | phi={phi_str}",
        assoc.n00,
        assoc.n01,
        assoc.n10,
        assoc.n11,
        assoc.positive_rate_a,
        assoc.positive_rate_b,
        assoc.observed_agreement,
        assoc.expected_agreement,
    );
}

/// §3.1 band classification.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Band {
    Redundancy,
    /// Intermediate chance-corrected agreement — NOT a fusion verdict
    /// (renamed from `Fusion` 2026-08-05: a middle kappa is merely
    /// intermediate agreement under the observed marginals; a fusion
    /// VERDICT additionally requires §12.4 D3b's held-out criterion).
    Intermediate,
    NoSharedHorizon,
    Undefined,
}
fn classify_band(kappa: Option<f64>) -> Band {
    match kappa {
        None => Band::Undefined,
        Some(k) if k > KAPPA_REDUNDANCY_FLOOR => Band::Redundancy,
        Some(k) if k < KAPPA_NO_SHARED_HORIZON_CEILING => Band::NoSharedHorizon,
        Some(_) => Band::Intermediate,
    }
}

// ── main ────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("BLW_KJV_TSV").unwrap_or_else(|_| "/tmp/kjv_verses.tsv".to_string());
    println!("== D-BLW-3 — Horizontverschmelzung: two rank projections, Strict vs Aware, one real sealed series ==");
    println!(
        "corpus       : {path} (PRE-REGISTERED to exactly {M_VERSES} verses, {S_CYCLES} cycles x \
         {SLICE}/cycle, q={Q_QUANTILE}, V_pin at cycle {V_PIN_CYCLE}, K={K_FIXED_PREFIX})"
    );

    let verses = load_verses(&path, M_VERSES)?;
    assert_eq!(
        verses.len(),
        M_VERSES,
        "PRE-REGISTERED §3.6: this harness requires exactly {M_VERSES} verses (got {}); the run \
         shape (S={S_CYCLES}, SLICE={SLICE}, V_PIN at cycle {V_PIN_CYCLE}, K={K_FIXED_PREFIX}) is \
         fixed BEFORE any kappa exists and is not renegotiated by a short corpus",
        verses.len()
    );

    let seed_a = bloom_of_terms(SEED_A);
    let seed_b = bloom_of_terms(SEED_B);
    let god_probe = bloom_of_terms(&["god"]);

    // ── THE ONE TENANT ───────────────────────────────────────────────────
    let mut owner: Tenant = MailboxSoA::new(TENANT_ID, TENANT_W_SLOT, TENANT_THRESHOLD);

    // P1 — slice 1 seeded BEFORE the loop (there is no "post-apply of cycle 0").
    let mut seated_total = seed_slice(&mut owner, 0, &verses[0..SLICE]);
    owner.set_populated(seated_total);
    owner.tick(); // cycle 0 -> 1, mirrors blw_tenant.rs:618.

    let mut sink = MemWal::new();
    let mut writer: BatchWriter<RowSpanDescriptor> = BatchWriter::new();
    let mut watermark: Option<u64> = None;

    // Two full Rubicon-DAG loops (verified legal against kanban.rs above).
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
        CycleSpec {
            id: CycleId(3),
            from: KanbanColumn::Evaluation,
            to: KanbanColumn::Plan,
        },
        CycleSpec {
            id: CycleId(4),
            from: KanbanColumn::Plan,
            to: KanbanColumn::Planning,
        },
        CycleSpec {
            id: CycleId(5),
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
        },
        CycleSpec {
            id: CycleId(6),
            from: KanbanColumn::CognitiveWork,
            to: KanbanColumn::Evaluation,
        },
        CycleSpec {
            id: CycleId(7),
            from: KanbanColumn::Evaluation,
            to: KanbanColumn::Plan,
        },
        CycleSpec {
            id: CycleId(8),
            from: KanbanColumn::Plan,
            to: KanbanColumn::Planning,
        },
    ];
    assert_eq!(
        plan.len(),
        S_CYCLES,
        "the DAG plan must have exactly S_CYCLES entries"
    );

    let mut all_rows: Vec<VerdictRow> =
        Vec::with_capacity(3 * SLICE * S_CYCLES * (S_CYCLES + 1) / 2);
    let mut sealed_versions: BTreeMap<usize, LanceVersion> = BTreeMap::new();
    let mut v_pin: Option<LanceVersion> = None;
    let mut v8: Option<LanceVersion> = None;

    for (idx, spec) in plan.iter().enumerate() {
        let c = idx + 1; // 1-indexed cycle number.
        assert_eq!(owner.phase(), spec.from, "cycle {c} precondition");
        let owner_cycle = owner.cycle();

        // ── P1: seed slice c strictly between post-apply(c-1) and pre-eval(c) ──
        // Slice 1 was already seated above (no "cycle 0" post-apply exists).
        if c > 1 {
            let n = seed_slice(
                &mut owner,
                (c - 1) * SLICE,
                &verses[(c - 1) * SLICE..c * SLICE],
            );
            seated_total += n;
            owner.set_populated(seated_total);
        }
        assert_eq!(
            seated_total,
            c * SLICE,
            "P1: pool must grow by exactly one slice per cycle"
        );
        assert_eq!(
            owner.n_rows(),
            seated_total,
            "declared logical size == seated rows"
        );

        // ── §1.5 emission: score+verdict the FULL POOL seated by this horizon ──
        let verdict_a = rank_verdicts(&owner, seated_total, &seed_a);
        let verdict_b = rank_verdicts(&owner, seated_total, &seed_b);
        let verdict_z: Vec<bool> = (0..seated_total)
            .map(|row| contains_all(&owner, row, &god_probe))
            .collect();

        // ── the seal (mirrors blw_tenant.rs steps 3/4/6; one landing per
        // cycle since there is no probe-sweep dirty set here) ──
        let outcome = StrategyOutcome {
            reliability: 1.0,
            intended_move: Some(bootstrap_intent(spec.from, spec.to)),
        };
        let span = RowSpanDescriptor {
            row_lo: ((c - 1) * SLICE) as u32,
            row_hi: (c * SLICE) as u32,
            cycle: owner_cycle,
        };
        let cast =
            emit_bootstrap_intent(&outcome, owner.mailbox_id(), owner_cycle, &mut writer, span)
                .expect("a bootstrap sentinel must rebind and cast");
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
        assert_eq!((cast_move.from, cast_move.to), (spec.from, spec.to));

        let slots = vec![SweepSlot {
            cycle: spec.id,
            stream_position: c as u64,
            owner: cast_owner,
            row: 0,
            paired_move: Some(cast_move),
            payload: span.to_le_bytes().to_vec(),
        }];
        let appends_before = sink.wal_writes();
        let base = sink.head();
        let commit_outcome =
            persist_cycle(&mut sink, CycleFrame::new(spec.id, base), slots).await?;
        let CommitOutcome::Committed { version, .. } = commit_outcome else {
            panic!(
                "cycle {:?}: expected a fresh Committed outcome (every cycle id in this \
                 harness's plan is used exactly once), got {commit_outcome:?}",
                spec.id
            );
        };
        assert_eq!(
            sink.wal_writes() - appends_before,
            1,
            "one landing -> exactly ONE WAL append"
        );

        // `after_cycle` bounds the read to cycles strictly after the PREVIOUS
        // cycle in `plan` (cycle ids are the 1-indexed plan position), which is
        // exactly this cycle's own newly-sealed landing — the cycle-keyed
        // equivalent of the old `scan_sealed(Some(base))` version-keyed bound.
        let after_cycle = (spec.id.0 > 1).then(|| CycleId(spec.id.0 - 1));
        let sealed = sink.scan_sealed(after_cycle).await?;
        let recovered = recover_and_apply(&mut owner, &sealed, watermark).map_err(|(_, e)| e)?;
        watermark = recovered.watermark;
        assert_eq!(
            recovered.applied.len(),
            1,
            "one tenant, one board -> exactly one lifecycle step per cycle"
        );
        let applied = recovered.applied[0];
        assert_eq!((applied.from, applied.to), (cast_move.from, cast_move.to));
        assert_eq!(
            owner.phase(),
            spec.to,
            "the tenant advanced to the cast target"
        );

        let vc: LanceVersion = version.0;
        sealed_versions.insert(c, vc);
        if c == V_PIN_CYCLE {
            v_pin = Some(vc);
        }
        if c == S_CYCLES {
            v8 = Some(vc);
        }

        // §1.5 EMISSION RULE: one VerdictRow per seated verse x per projection.
        // Zipped (not range-indexed) to avoid indexing three parallel Vecs by
        // a manual loop counter.
        for (row, ((&va, &vb), &vz)) in verdict_a
            .iter()
            .zip(verdict_b.iter())
            .zip(verdict_z.iter())
            .enumerate()
        {
            let subject = format!("kjv:{row:05}");
            all_rows.push(VerdictRow {
                subject: subject.clone(),
                horizon: vc,
                projection: Proj::A,
                verdict: va,
            });
            all_rows.push(VerdictRow {
                subject: subject.clone(),
                horizon: vc,
                projection: Proj::B,
                verdict: vb,
            });
            all_rows.push(VerdictRow {
                subject,
                horizon: vc,
                projection: Proj::Z,
                verdict: vz,
            });
        }

        println!(
            "cycle {c:>2}    : sealed V{vc} — pool {seated_total} rows, {:?}->{:?}",
            applied.from, applied.to
        );
        owner.tick();
    }

    let v_pin = v_pin.expect("V_PIN_CYCLE must have fired");
    let v8 = v8.expect("S_CYCLES must have fired");
    println!("--");
    println!(
        "row count    : {} (expected {})",
        all_rows.len(),
        3 * SLICE * S_CYCLES * (S_CYCLES + 1) / 2
    );
    assert_eq!(
        all_rows.len(),
        3 * SLICE * S_CYCLES * (S_CYCLES + 1) / 2,
        "§1.5: Sum_{{c=1..S}} (SLICE*c) * 3 projections must match exactly"
    );

    // C6: the pre-registered null, printed BEFORE any result below.
    println!();
    println!(
        "PRE-REGISTERED NULL (C6): if the only movement channel were frontier truncation the \
         trajectory would decay as 7/h; the rank criterion's pool growth is the designed channel; \
         a flat or decaying trajectory triggers the plan's regrade and is a reportable KILL, not a \
         failure of the harness."
    );

    // ── G3 (disclosure, not a can-fire gate) + C2 precondition assert ──
    {
        let kf0 = all_rows[0].knowable_from();
        for r in &all_rows {
            assert_eq!(
                r.knowable_from(),
                kf0,
                "C2/G3: every emitted row must return the SAME knowable_from"
            );
            assert!(
                r.knowable_from() <= v_pin,
                "C2/G3: every emitted row's knowable_from must be <= V_pin"
            );
        }
        println!(
            "G3 (disclosure): knowable_from is CONSTANT at {kf0} across all {} rows and <= V_pin={v_pin} — \
             the Unknowable branch is never exercised in this harness (see C3 / §6 below)",
            all_rows.len()
        );
    }

    // ── G1 — the mode axis: `QueryReference::at(v, rung)` selects the mode
    // by choosing the rung and by nothing else (`temporal.rs:167-175`). ──
    let qref_strict_pin = QueryReference::at(v_pin, RUNG_STRICT);
    let qref_aware_pin = QueryReference::at(v_pin, RUNG_AWARE);
    let strict_v4 = deinterlace(&all_rows, &qref_strict_pin, &NoDeps);
    let aware_v4 = deinterlace(&all_rows, &qref_aware_pin, &NoDeps);

    // G1a can-fire: Aware admits strictly more rows at the SAME pin.
    assert!(
        aware_v4.len() > strict_v4.len(),
        "G1a can-fire: Aware must admit strictly more rows than Strict at V{v_pin} \
         (strict={}, aware={})",
        strict_v4.len(),
        aware_v4.len()
    );
    println!("G1a can-fire: Strict@V{v_pin} admits {} rows, Aware@V{v_pin} admits {} rows — Aware admits strictly more", strict_v4.len(), aware_v4.len());

    // G1b can-stay-silent: at V8 nothing is Anachronistic, so Strict and
    // Aware must admit the SAME set — proving G1a's difference came from
    // future-horizon rows, not mode-dependent plumbing.
    let qref_strict_v8 = QueryReference::at(v8, RUNG_STRICT);
    let qref_aware_v8 = QueryReference::at(v8, RUNG_AWARE);
    let strict_v8_rows = deinterlace(&all_rows, &qref_strict_v8, &NoDeps);
    let aware_v8_rows = deinterlace(&all_rows, &qref_aware_v8, &NoDeps);
    assert_eq!(
        strict_v8_rows.len(),
        aware_v8_rows.len(),
        "G1b can-stay-silent: at V8 (max horizon present) Strict and Aware must admit the SAME COUNT"
    );
    let fold_strict_v8_a = fold_last_by_subject(&strict_v8_rows, Proj::A);
    let fold_aware_v8_a = fold_last_by_subject(&aware_v8_rows, Proj::A);
    assert_eq!(
        fold_strict_v8_a, fold_aware_v8_a,
        "G1b can-stay-silent: folded A must be byte-identical at V8"
    );
    println!(
        "G1b can-stay-silent: at V8 Strict and Aware fold to the SAME {} subjects — OK",
        fold_strict_v8_a.len()
    );

    // G1c — C1: extensional identity, asserted THREE ways. With
    // knowable_from constant <= ref, Aware@V4 == Retro@V4 == Strict@V8 over
    // ONE row set (every row is admitted by all three reads — see
    // `seating_slice`'s doc comment for the arithmetic).
    let qref_retro_pin = QueryReference::at(v_pin, RUNG_RETRO);
    let retro_v4 = deinterlace(&all_rows, &qref_retro_pin, &NoDeps);
    fn seq_key(rows: &[VerdictRow]) -> Vec<(String, u64, Proj)> {
        rows.iter()
            .map(|r| (r.subject.clone(), r.horizon, r.projection))
            .collect()
    }
    assert_eq!(
        aware_v4.len(),
        retro_v4.len(),
        "G1c: Aware@V4 vs Retro@V4 must be the same length"
    );
    assert_eq!(
        aware_v4.len(),
        strict_v8_rows.len(),
        "G1c: Aware@V4 vs Strict@V8 must be the same length"
    );
    assert_eq!(
        seq_key(&aware_v4),
        seq_key(&retro_v4),
        "G1c: Aware@V4 vs Retro@V4 must be the IDENTICAL row sequence"
    );
    assert_eq!(
        seq_key(&aware_v4),
        seq_key(&strict_v8_rows),
        "G1c: Aware@V4 vs Strict@V8 must be the IDENTICAL row sequence"
    );
    println!(
        "G1c (C1): at(V{v_pin},5)=Aware, at(V{v_pin},9)=Retro, and at(V{v8},0)=Strict all return the \
         IDENTICAL {}-row sequence — one function, three names — the mode framing is doctrinal, not \
         mechanical.",
        aware_v4.len()
    );

    // ── G5 — the fixed verse set is actually fixed (§12.3b anti-sample-growth) ──
    let fold_a_strict_pin =
        restrict_to_prefix(&fold_last_by_subject(&strict_v4, Proj::A), K_FIXED_PREFIX);
    let fold_a_aware_pin =
        restrict_to_prefix(&fold_last_by_subject(&aware_v4, Proj::A), K_FIXED_PREFIX);
    let fold_b_strict_pin =
        restrict_to_prefix(&fold_last_by_subject(&strict_v4, Proj::B), K_FIXED_PREFIX);
    let fold_b_aware_pin =
        restrict_to_prefix(&fold_last_by_subject(&aware_v4, Proj::B), K_FIXED_PREFIX);
    let fold_z_strict_pin =
        restrict_to_prefix(&fold_last_by_subject(&strict_v4, Proj::Z), K_FIXED_PREFIX);
    let fold_z_aware_pin =
        restrict_to_prefix(&fold_last_by_subject(&aware_v4, Proj::Z), K_FIXED_PREFIX);
    assert_eq!(
        fold_a_strict_pin.len(),
        K_FIXED_PREFIX,
        "G5: Strict-restricted fold must be exactly K rows"
    );
    assert_eq!(
        fold_a_aware_pin.len(),
        K_FIXED_PREFIX,
        "G5: Aware-restricted fold must be exactly K rows"
    );
    let subjects_strict: Vec<&String> = fold_a_strict_pin.iter().map(|(s, _)| s).collect();
    let subjects_aware: Vec<&String> = fold_a_aware_pin.iter().map(|(s, _)| s).collect();
    assert_eq!(
        subjects_strict, subjects_aware,
        "G5: restricted folds must share the IDENTICAL subject sequence"
    );
    println!("G5: both restricted folds are exactly {K_FIXED_PREFIX} subjects, identical sequence — anti-sample-growth confound removed");

    let a_strict = bools_only(&fold_a_strict_pin);
    let a_aware = bools_only(&fold_a_aware_pin);
    let b_strict = bools_only(&fold_b_strict_pin);
    let b_aware = bools_only(&fold_b_aware_pin);
    let z_strict = bools_only(&fold_z_strict_pin);
    let z_aware = bools_only(&fold_z_aware_pin);

    // G1a can-fire, continued: the folded A-verdict vector must differ.
    let (gain_a_pin, lost_a_pin) = churn(&a_strict, &a_aware);
    let hamming_a_pin = gain_a_pin + lost_a_pin;
    assert!(
        hamming_a_pin > 0,
        "G1a can-fire: folded A must differ on >= 1 verse between Strict and Aware at V_pin"
    );

    // ── G2 — the inert control ──
    assert_eq!(
        z_strict, z_aware,
        "G2 can-stay-silent: Z (horizon-independent control) moved between reads — plumbing leak, \
         VOIDS the (A,B) movement result"
    );
    let assoc_zz = binary_association(&z_strict, &z_aware);
    if let Some(zz) = assoc_zz {
        print_association_table("G2 kappa(Z,Z)", &zz);
    }
    println!("G2: Z control byte-identical across reads ({} rows), A differs on {hamming_a_pin} — plumbing contributes zero movement", z_strict.len());

    // ── G6 — the fold neither drops nor duplicates (CORRECTED per the
    // coordinator's mid-build message: per-subject counts vary by the
    // subject's OWN seating slice, not uniformly across the prefix). ──
    let mut aware_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for r in aware_v4
        .iter()
        .filter(|r| r.projection == Proj::A && subject_index(&r.subject) < K_FIXED_PREFIX)
    {
        *aware_counts.entry(subject_index(&r.subject)).or_insert(0) += 1;
    }
    let mut strict_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for r in strict_v4
        .iter()
        .filter(|r| r.projection == Proj::A && subject_index(&r.subject) < K_FIXED_PREFIX)
    {
        *strict_counts.entry(subject_index(&r.subject)).or_insert(0) += 1;
    }
    for row in 0..K_FIXED_PREFIX {
        let s = seating_slice(row);
        let expect_aware = 9 - s;
        let expect_strict = 5 - s;
        assert_eq!(
            aware_counts.get(&row).copied().unwrap_or(0),
            expect_aware,
            "G6: subject kjv:{row:05} (seated slice {s}) must have exactly {expect_aware} Aware rows"
        );
        assert_eq!(
            strict_counts.get(&row).copied().unwrap_or(0),
            expect_strict,
            "G6: subject kjv:{row:05} (seated slice {s}) must have exactly {expect_strict} Strict rows"
        );
    }
    assert_eq!(
        aware_counts.len(),
        K_FIXED_PREFIX,
        "G6: Aware fold must cover exactly K subjects"
    );
    assert_eq!(
        strict_counts.len(),
        K_FIXED_PREFIX,
        "G6: Strict fold must cover exactly K subjects"
    );
    println!("G6: per-subject row counts verified (Aware=9-s, Strict=5-s per seating slice s=1..4), both folds cover exactly {K_FIXED_PREFIX} subjects");

    // ── G7 — the ordering the fold depends on is real, not an input-order accident ──
    let mut rows_desc: Vec<VerdictRow> = all_rows.clone();
    rows_desc.reverse(); // deliberately non-ascending horizon order.
    let aware_v4_desc = deinterlace(&rows_desc, &qref_aware_pin, &NoDeps);
    let fold_a_aware_desc = restrict_to_prefix(
        &fold_last_by_subject(&aware_v4_desc, Proj::A),
        K_FIXED_PREFIX,
    );
    assert_eq!(
        fold_a_aware_desc, fold_a_aware_pin,
        "G7: descending-input fold must equal the ascending-input fold"
    );
    println!("G7: feeding rows in descending horizon order yields the IDENTICAL fold — the sort, not input order, does the work");

    // ── G4 — the band guards themselves can fire and can stay silent ──
    //
    // DEGENERATE can-fire — CORRECTED after the first real run. The design's
    // fixture was "ANY overlap with 'god'", premised on blw_tenant's "~90 %"
    // note; measured HERE it is 0.1285 — comfortably non-degenerate, so the
    // can-fire half could never fire. An empirical fixture rots with the
    // corpus; these two cannot: the same scoring pipeline under criteria that
    // are degenerate BY CONSTRUCTION (`score >= 0` fires on every verse;
    // `score > u32::MAX` on none). Real corpus, real pipeline, both tails.
    let art_all: Vec<bool> = (0..M_VERSES)
        .map(|row| {
            // `score >= 0` written as a tautology the compiler cannot fold
            // away for us: every u32 clears zero.
            let s = score_row(&owner, row, &god_probe);
            s.wrapping_add(1) != 0 || s == u32::MAX
        })
        .collect();
    let art_none: Vec<bool> = (0..M_VERSES)
        .map(|row| u64::from(score_row(&owner, row, &god_probe)) > u64::from(u32::MAX))
        .collect();
    let (rate_all, rate_none) = (positive_rate(&art_all), positive_rate(&art_none));
    assert!(
        is_degenerate(rate_all) && is_degenerate(rate_none),
        "G4 DEGENERATE can-fire: constant-by-construction projections must stamp on BOTH tails, \
         got all={rate_all:.4} none={rate_none:.4}"
    );
    println!(
        "G4 DEGENERATE can-fire: constant projections stamped on both tails (all={rate_all:.4}, none={rate_none:.4})"
    );
    // ...and the retired empirical fixture is kept as an OBSERVATION on real
    // data: 'god any-overlap' measured inside the band is a genuine
    // can-stay-silent instance, printed rather than asserted degenerate.
    let god_any: Vec<bool> = (0..M_VERSES)
        .map(|row| score_row(&owner, row, &god_probe) > 0)
        .collect();
    let god_rate = positive_rate(&god_any);
    assert!(
        !is_degenerate(god_rate),
        "G4 observation drifted: 'god any-overlap' now degenerate at {god_rate:.4} — re-examine the corpus/tokenizer before trusting the run"
    );
    println!(
        "G4 DEGENERATE can-stay-silent (real-data arm): 'god any-overlap' rate={god_rate:.4} — inside the band, NOT stamped"
    );

    // DEGENERATE can-stay-silent: A and B at q=0.25 must NOT be stamped.
    // Hard by construction for Strict (pool == prefix, rate == exactly
    // n_pos/pool_size); soft-but-checked for Aware (a real, data-dependent
    // marginal on the fixed prefix — design §2.3's informative quantity).
    let rate_a_strict = positive_rate(&a_strict);
    let rate_b_strict = positive_rate(&b_strict);
    let rate_a_aware = positive_rate(&a_aware);
    let rate_b_aware = positive_rate(&b_aware);
    assert!(
        !is_degenerate(rate_a_strict),
        "G4 DEGENERATE can-stay-silent: A_strict outside bounds: {rate_a_strict}"
    );
    assert!(
        !is_degenerate(rate_b_strict),
        "G4 DEGENERATE can-stay-silent: B_strict outside bounds: {rate_b_strict}"
    );
    let a_aware_degenerate = is_degenerate(rate_a_aware);
    let b_aware_degenerate = is_degenerate(rate_b_aware);
    println!(
        "G4 DEGENERATE can-stay-silent: rate_a_strict={rate_a_strict:.4} rate_b_strict={rate_b_strict:.4} \
         rate_a_aware={rate_a_aware:.4}{} rate_b_aware={rate_b_aware:.4}{}",
        if a_aware_degenerate { " STAMPED DEGENERATE" } else { "" },
        if b_aware_degenerate { " STAMPED DEGENERATE" } else { "" }
    );

    // COLLAPSED can-fire: (A, A) — identical by construction.
    let assoc_aa =
        binary_association(&a_strict, &a_strict).expect("A vs A must be structurally valid");
    assert_eq!(
        assoc_aa.n01 + assoc_aa.n10,
        0,
        "G4 COLLAPSED can-fire: (A,A) must show ZERO discordant cells"
    );
    println!("G4 COLLAPSED can-fire: (A,A) n01+n10=0 — the guard reads counts, not labels — OK");

    // COLLAPSED can-stay-silent for the REAL (A,B) pair — a genuine,
    // data-dependent, PRE-ACCEPTED possible outcome (design §3.5/§4): if it
    // fires, no fusion claim may be made, but the harness does NOT panic —
    // reported below at the point the fusion claim is (or is not) made.
    let assoc_strict = binary_association(&a_strict, &b_strict)
        .unwrap_or_else(|| panic!("C8/§3.2a KILL: binary_association returned None on a CLAIMED pair (Strict@V_pin) — structural"));
    let assoc_aware = binary_association(&a_aware, &b_aware)
        .unwrap_or_else(|| panic!("C8/§3.2a KILL: binary_association returned None on a CLAIMED pair (Aware@V_pin) — structural"));
    let collapsed_floor = (COLLAPSED_FRACTION * K_FIXED_PREFIX as f64) as u64;
    let strict_collapsed = assoc_strict.n01 + assoc_strict.n10 < collapsed_floor;
    let aware_collapsed = assoc_aware.n01 + assoc_aware.n10 < collapsed_floor;
    let strict_unstable = assoc_strict.expected_agreement > UNSTABLE_EXPECTED_AGREEMENT;
    let aware_unstable = assoc_aware.expected_agreement > UNSTABLE_EXPECTED_AGREEMENT;
    println!(
        "G4 COLLAPSED can-stay-silent (real A,B pair): strict {}discordant={}, aware {}discordant={} \
         (floor {collapsed_floor}) — {}",
        if strict_collapsed { "STAMPED COLLAPSED, " } else { "" },
        assoc_strict.n01 + assoc_strict.n10,
        if aware_collapsed { "STAMPED COLLAPSED, " } else { "" },
        assoc_aware.n01 + assoc_aware.n10,
        if strict_collapsed || aware_collapsed {
            "a real, pre-accepted possible outcome — see the fusion verdict below"
        } else {
            "neither stamped — OK"
        }
    );
    if strict_unstable || aware_unstable {
        println!(
            "§3.5 UNSTABLE guard: strict expected_agreement={:.4}{} aware expected_agreement={:.4}{}",
            assoc_strict.expected_agreement,
            if strict_unstable { " STAMPED UNSTABLE" } else { "" },
            assoc_aware.expected_agreement,
            if aware_unstable { " STAMPED UNSTABLE" } else { "" }
        );
    }

    // ── kappa(A,B) tables at V_pin, both reads (C8: FULL table always) ──
    println!("-- kappa(A,B) @ V_pin=V{v_pin} --");
    print_association_table("Strict (a-priori) ", &assoc_strict);
    print_association_table("Aware  (hindsight)", &assoc_aware);

    let band_strict = classify_band(assoc_strict.kappa);
    let band_aware = classify_band(assoc_aware.kappa);
    println!(
        "band @ V_pin: Strict={band_strict:?} (kappa={:?}) Aware={band_aware:?} (kappa={:?})",
        assoc_strict.kappa, assoc_aware.kappa
    );

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum BandOutcome {
        InIn,
        OutOutSameSide,
        BandExit,
        UndefinedKappa,
    }
    let band_outcome =
        if matches!(band_strict, Band::Undefined) || matches!(band_aware, Band::Undefined) {
            BandOutcome::UndefinedKappa
        } else if band_strict == Band::Intermediate && band_aware == Band::Intermediate {
            BandOutcome::InIn
        } else if band_strict == band_aware {
            BandOutcome::OutOutSameSide
        } else {
            BandOutcome::BandExit
        };
    println!("§3.2 band outcome: {band_outcome:?}");

    // ── §3.3 movement (VERBATIM from §12.3b) at V_pin ──
    let delta_kappa_pin: Option<f64> = match (assoc_strict.kappa, assoc_aware.kappa) {
        (Some(ka), Some(kh)) => Some(kh - ka),
        _ => None,
    };
    println!(
        "§3.3 movement @ V_pin: delta_kappa={delta_kappa_pin:?} (threshold {MOVEMENT_THRESHOLD})"
    );
    let movement_fires = delta_kappa_pin.is_some_and(|d| d.abs() >= MOVEMENT_THRESHOLD);
    if movement_fires {
        let candidate_permitted = band_outcome == BandOutcome::InIn
            && !strict_collapsed
            && !aware_collapsed
            && !strict_unstable
            && !aware_unstable;
        if candidate_permitted {
            println!("§3.3: MOVEMENT FIRES at V_pin and band is IN/IN — COMPLEMENTARITY CANDIDATE (a fusion VERDICT additionally requires §12.4 D3b's held-out criterion, which remains BLOCKED)");
        } else {
            println!("§3.3: MOVEMENT FIRES at V_pin but the band/guard state does not permit a fusion claim (§3.2/§3.5)");
        }
    }

    // ── §5.3 the explicit "identical -- drop the distinction" test, C5 Hamming companion ──
    let (gain_b_pin, lost_b_pin) = churn(&b_strict, &b_aware);
    let hamming_b_pin = gain_b_pin + lost_b_pin;
    println!(
        "§5.3 churn (C5, never averaged into delta_kappa) @ V_pin: A gained={gain_a_pin} lost={lost_a_pin} \
         hamming={hamming_a_pin}; B gained={gain_b_pin} lost={lost_b_pin} hamming={hamming_b_pin}"
    );
    let outcome_53 = match delta_kappa_pin {
        Some(d) if d.abs() < DROP_THRESHOLD && hamming_a_pin == 0 && hamming_b_pin == 0 => {
            "DROP — reads genuinely identical"
        }
        Some(d) if d.abs() < DROP_THRESHOLD => {
            "CHURN-WITHOUT-REALIGNMENT — reads differ but agreement does not — NO FUSION CLAIM"
        }
        Some(d) if d.abs() >= MOVEMENT_THRESHOLD => "MOVEMENT FIRES (reported above)",
        Some(_) => "MIDDLE GROUND — 0.01 <= |delta_kappa| < 0.10, reported, no verdict claimed",
        None => "UNDEFINED — at least one kappa is undefined at V_pin",
    };
    println!("§5.3 named outcome @ V_pin: {outcome_53}");
    if outcome_53.starts_with("DROP") {
        println!("A-PRIORI/HINDSIGHT DISTINCTION DOES NO WORK — DROPPED");
    }

    // ── C7: drop test over ALL EIGHT horizons ──
    println!();
    println!(
        "== C7: drop test over ALL EIGHT horizons (Strict@Vk vs Aware@Vk, k=1..{S_CYCLES}) =="
    );
    let mut max_abs_delta: f64 = 0.0;
    let mut any_undefined = false;
    let mut hamming_at_v8: Option<(usize, usize)> = None;
    // External-review catch (2026-08-05): kappa can hide CANCELLING churn, and
    // Hamming at V8 is zero BY CONSTRUCTION (the identical case) — so a DROP
    // keyed on V8 alone could fire while verdicts swapped in opposite
    // directions at V1..V7. Track the maxima across ALL horizons instead.
    let mut max_ham_a: usize = 0;
    let mut max_ham_b: usize = 0;
    for c in 1..=plan.len() {
        let prefix_k = c * SLICE;
        let vk = *sealed_versions
            .get(&c)
            .expect("every cycle must have sealed a version");
        let qref_s = QueryReference::at(vk, RUNG_STRICT);
        let qref_a = QueryReference::at(vk, RUNG_AWARE);
        let s_rows = deinterlace(&all_rows, &qref_s, &NoDeps);
        let a_rows = deinterlace(&all_rows, &qref_a, &NoDeps);
        let fa_s = restrict_to_prefix(&fold_last_by_subject(&s_rows, Proj::A), prefix_k);
        let fa_a = restrict_to_prefix(&fold_last_by_subject(&a_rows, Proj::A), prefix_k);
        let fb_s = restrict_to_prefix(&fold_last_by_subject(&s_rows, Proj::B), prefix_k);
        let fb_a = restrict_to_prefix(&fold_last_by_subject(&a_rows, Proj::B), prefix_k);
        assert_eq!(
            fa_s.len(),
            prefix_k,
            "C7 k={c}: Strict-restricted fold must be exactly the k-th fixed prefix"
        );
        assert_eq!(
            fa_a.len(),
            prefix_k,
            "C7 k={c}: Aware-restricted fold must be exactly the k-th fixed prefix"
        );

        let a_s_bools = bools_only(&fa_s);
        let a_a_bools = bools_only(&fa_a);
        let b_s_bools = bools_only(&fb_s);
        let b_a_bools = bools_only(&fb_a);

        let assoc_s = binary_association(&a_s_bools, &b_s_bools);
        let assoc_a = binary_association(&a_a_bools, &b_a_bools);
        let (ks, ka) = match (&assoc_s, &assoc_a) {
            (Some(s), Some(a)) => (s.kappa, a.kappa),
            _ => {
                any_undefined = true;
                (None, None)
            }
        };
        let dk = match (ks, ka) {
            (Some(s), Some(a)) => Some(a - s),
            _ => {
                any_undefined = true;
                None
            }
        };
        if let Some(d) = dk {
            max_abs_delta = max_abs_delta.max(d.abs());
        }

        let ham_a = hamming(&a_s_bools, &a_a_bools);
        let ham_b = hamming(&b_s_bools, &b_a_bools);
        max_ham_a = max_ham_a.max(ham_a);
        max_ham_b = max_ham_b.max(ham_b);
        if c == plan.len() {
            hamming_at_v8 = Some((ham_a, ham_b));
        }

        println!("  k={c} V{vk} pool={prefix_k}: kappa_strict={ks:?} kappa_aware={ka:?} delta={dk:?} hamming_A={ham_a} hamming_B={ham_b}");
    }

    // C7 sanity: k = S_CYCLES is "the identical case" — Strict@V8 == Aware@V8
    // by construction (G1b/G1c-style reasoning: V8 is the max horizon
    // present, so nothing classifies Anachronistic under either read).
    let (ham_a_v8, ham_b_v8) = hamming_at_v8.expect("k=S_CYCLES must have run");
    assert_eq!(
        ham_a_v8, 0,
        "C7 sanity: at k=S_CYCLES (the identical case) Hamming_A must be 0"
    );
    assert_eq!(
        ham_b_v8, 0,
        "C7 sanity: at k=S_CYCLES (the identical case) Hamming_B must be 0"
    );

    let drop_fires =
        !any_undefined && max_abs_delta < DROP_THRESHOLD && max_ham_a == 0 && max_ham_b == 0;
    println!(
        "C7 DROP verdict: max|delta_kappa| over {S_CYCLES} horizons = {max_abs_delta:.4} (threshold {DROP_THRESHOLD}); \
         max hamming over ALL horizons=(A:{max_ham_a}, B:{max_ham_b}); identical-case (k={S_CYCLES}) hamming=(A:{ham_a_v8}, B:{ham_b_v8}) -> {}",
        if drop_fires {
            "DROP FIRES trajectory-wide"
        } else {
            "DROP DOES NOT FIRE — the distinction moves somewhere in the trajectory"
        }
    );

    // ── §6 not-claimed block (+ C3 addition) ──
    println!();
    println!("== §6 -- what is NOT claimed ==");
    println!("1. C3 ceiling: kappa/phi measure OVERLAP (reliability), never validity. NOT claimed: the later horizon reads verses better/more truly/more completely -- only DIFFERENTLY.");
    println!("2. C4 (design numbering): NO p-value is reported. jc::stats p-values are classical independent-sample; verses within one book are domain-correlated (I-NOISE-FLOOR-JIRAK). No significance claim.");
    println!("3. C2 (design numbering): kappa/phi always ship the FULL BinaryAssociation table, never bare. Spearman omitted (redundant on binary data). KR-20 NOT reported (2 items; pooling Z in would be a category error).");
    println!("4. No stance claim -- Hegel/Nietzsche/Kant/Wittgenstein are not inputs here.");
    println!("5. No §12.6 anchor claim (not A1/A2/A3) -- the seed vocabulary is A1-derived for PROVENANCE ONLY.");
    println!("6. No cross-language claim -- one lane (KJV) only.");
    println!(
        "7. No zero-copy claim -- deinterlace .cloned()s the admitted rows (temporal.rs:363)."
    );
    println!("8. No durability claim -- MemWal is in-process Mutex/Vec; 'versions' are sequence numbers, not Lance versions.");
    println!("9. No HLC/multi-writer claim, and no knowable_from/schema-clock claim -- that field is constant here (verified above, G3), the Unknowable branch is never reached.");
    println!("10. No parallelism/scale claim (D-BLW-4's axis).");
    println!("11. No fusion claim outside the IN/IN band, and none at all before D3b.");
    println!(
        "C3 (this task's correction): under this corpus deinterlace reduces to filter(v <= ref) + \
         stable sort; Unknowable/DependsClosure/HLC axes are inert, not exercised; the permitted \
         claim is FIRST DeinterlaceRow implementor and FIRST deinterlace caller; the finding lives \
         in the rank criterion, not temporal.rs."
    );

    Ok(())
}
