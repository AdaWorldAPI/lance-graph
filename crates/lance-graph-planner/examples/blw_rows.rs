//! `blw_rows` — **D-BLW-4**: N ROW-LEVEL thought bodies inside **ONE** tenant.
//!
//! # The axis, stated first because it has been wrong twice
//!
//! **Owner-count is NOT a scale knob.** An owner is a **TENANT** — one mailbox,
//! one kanban board, one `KanbanActor` that is its sole mutator (`CLAUDE.md`
//! §V3 rulings; `E-CE64-MB-4`; the SoA is *moved* into the actor and that move
//! is the compile-time no-aliasing proof). Two prior attempts got this wrong in
//! the same direction: the first tiled the Bible across 64 owners and
//! **fabricated 63 tenants** (plan §12.1a′); the second kept owner-count as the
//! axis and merely made the owners *lightweight*, which is worse — it preserved
//! the wrong unit and made the wrong thing cheap (§12.3a′).
//!
//! **The axis here is ROWS INSIDE ONE OWNER.** "N thoughts at once" is
//! data-parallelism over the verse rows of a single tenant's slice — exactly
//! what §12.1's own diagram always said (*"apply stance L to **the owner's
//! slice**"*). This file constructs **exactly one** `MailboxSoA` and never a
//! second one, at any row count, for any reason.
//!
//! # Which half is parallelised — and which is not
//!
//! | half | surface | parallel? |
//! |---|---|---|
//! | **read / evaluate** | `&V: MailboxSoaView` — `identity_plane_at`, `energy` | **YES** — `std::thread::scope`, contiguous row chunks, shared `&Tenant` |
//! | **write back** | `MailboxSoA::write_row` (`&mut self`, the ONE cycle-aware mutator) | **NO — single-mutator by construction** |
//!
//! The read side is where parallelism is free: reads take `&V`, so *no `&mut
//! self` during computation* (`.claude/rules/data-flow.md`) is **structural**,
//! not a convention — many rows can be evaluated concurrently from borrowed
//! slices. The write side is `&mut self` on the owner, so it is sequential by
//! type, and this harness makes no attempt to parallelise it. **Any speedup
//! reported below is a speedup of the READ half only.**
//!
//! The soundness precondition is a *compile* obligation, not an argument in a
//! comment: [`sweep_concurrent`] carries `V: Sync` (which is what makes the
//! shared `&V` `Send` for `std::thread::scope`), and it is instantiated at
//! `Tenant`. If `MailboxSoA<N>` ever grows an interior-mutability field, this
//! example stops compiling rather than silently changing meaning.
//!
//! # Pre-registration (fixed HERE, before the measurement was written)
//!
//! Every threshold below is a constant in this file, declared ahead of the
//! measurement code and **non-adjustable after the fact** — a miss is a miss.
//! §12.3a′ carries the W2 protocol over unchanged (median of ≥5 runs after one
//! discarded warm-up; a can-fire *and* a can-stay-silent half) and re-pins only
//! the unit, from owners to rows.
//!
//! | gate | constant | what it decides |
//! |---|---|---|
//! | **G-A** *precondition* | [`BODY_FLOOR_US`], [`MIN_SEQ_WALL_MS`], [`MIN_THREADS_TO_EVALUATE`] | whether this design can test the claim **at all** on this machine/profile |
//! | **G-B** *row axis* | [`THROUGHPUT_FLATNESS`] | whether sequential rows/s is flat enough across row counts for "rows/second" to be a meaningful unit |
//! | **G-C** *concurrency* | [`SPEEDUP_GATE`] over [`RUNS`] + [`WARMUPS`] | the actual claim: concurrent ≥ 2× sequential at the largest row count |
//!
//! **G-A is the honest half.** If the per-row body is cheaper than
//! [`BODY_FLOOR_US`], or the whole sequential sweep is shorter than
//! [`MIN_SEQ_WALL_MS`], or the machine offers fewer than
//! [`MIN_THREADS_TO_EVALUATE`] threads, then a speedup number would be measuring
//! thread-spawn overhead and scheduler noise. In that case **G-C is NOT
//! EVALUATED** and the run prints `INCONCLUSIVE`. A null result reported as null
//! is the correct output; a number produced anyway would not be.
//!
//! **Kill condition (§12.3a′, restated):** if row-level concurrency does not
//! beat sequential under this protocol, the arm's claim (a) regrades to
//! *"N-scale **sequential** row evaluation within one tenant"* — still true,
//! different claim.
//!
//! # What is asserted vs what is reported
//!
//! - **Correctness falsifiers `panic!`.** The iron rule (evaluation mutates
//!   nothing), determinism under concurrency, and the comparator's own
//!   can-bark/can-stay-silent twins are `assert!`s. A failure there is a defect.
//! - **Measurement gates print.** G-A/G-B/G-C emit `PASS` / `KILL` /
//!   `INCONCLUSIVE` lines and do **not** panic. A measurement that misses a
//!   pre-registered threshold is evidence, not a bug — and the process rule is
//!   that it regrades the claim rather than failing the build.
//!
//! # What this harness does NOT claim
//!
//! - **No durability, no seal, no applied lifecycle step.** D-BLW-1
//!   (`examples/blw_tenant.rs`) owns the cast → seal → `recover_and_apply` loop
//!   and proved it; re-running it here would add wall time and no new evidence.
//!   This file casts the write intent (ahead of the write, as designed) and
//!   **never advances a phase** — *no successful write ⇒ no applied step*
//!   (`owner_adapter.rs` module doc). The cast is deliberately left dangling.
//! - **No stance instrument.** The row body is a deterministic bit-mixing read
//!   over the borrowed content identity plane. It is a *load*, chosen so its
//!   cost is tunable and its output is row-dependent — **not** a
//!   Hegel/Nietzsche/Kant/Wittgenstein projection (§12.3c retired κ, §12.7
//!   recorded the texture rewrite as a KILL). Nothing here is a semantic claim.
//! - **No memory-bandwidth claim.** A row's content plane is 2 KiB and the body
//!   re-reads it `reps` times, so the body is **compute-bound by construction**.
//!   This measures compute parallelism over rows, not streaming bandwidth.
//!
//! # The memory figure, and what struct it is a figure OF
//!
//! The canonical row is **512 B** (`NODE_ROW_STRIDE`, const-asserted
//! `size_of::<NodeRow>() == 512`). The byte image this harness snapshots is a
//! figure of **`MailboxSoA<2048>`**, whose content/topic/angle identity planes
//! alone are `3 × 2048 × 256 × 8 B` = **12 MiB** (6,144 B/row, **12× the
//! canon**). That divergence is the open question
//! `ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` and is **not** resolved here — it is
//! named so no figure printed below is silently read as a canonical-node-row
//! figure, and so the two are never averaged.
//!
//! # Duplication, declared
//!
//! `snapshot` / `first_diff` / `locate` / `row_image` / the bloom encoder are
//! ported from `examples/blw_tenant.rs` (D-BLW-1). Cargo examples cannot import
//! one another, so the alternative was to move them into the library — which
//! would put a test instrument on the production surface for no consumer. The
//! copy is deliberate; if `MailboxSoA` grows a column, **both** files' `snapshot`
//! must grow with it or their "byte-identical" claims silently narrow.
//!
//! # Run
//!
//! ```text
//! cargo run -p lance-graph-planner --example blw_rows           # 2,000 rows (default)
//! cargo run -p lance-graph-planner --example blw_rows -- 512    # bounded
//! BLW_BODY_REPS=96 cargo run ... --example blw_rows             # a LABELLED re-run
//! BLW_KJV_TSV=/path/to/kjv_verses.tsv cargo run ... --example blw_rows
//! ```
//!
//! The pre-registered gates apply to the **default** configuration. Any run with
//! `BLW_BODY_REPS` set is a labelled re-run and its verdict is reported as such.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::time::Instant;

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::soa_view::{IdentityPlane, MailboxSoaView};
use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
use lance_graph_planner::traits::StrategyOutcome;

// ── the ONE tenant ──────────────────────────────────────────────────────────

/// Row capacity of the single tenant (type-level const; the *logical* size is
/// declared with `set_populated` and read back through `n_rows`).
const N_CAP: usize = 2048;

/// Default bounded corpus size. §12.7: an unbounded 31,102-verse run blew a
/// 10-minute budget in a sibling harness, so the corpus is bounded on purpose.
const DEFAULT_ROWS: usize = 2_000;

/// The tenant's mailbox id. Deliberately **non-zero**: `mailbox 0` is the
/// bootstrap sentinel (`owner_adapter::BOOTSTRAP_OWNER`), so a zero id would
/// make the rebind assertion below vacuous.
const TENANT_ID: MailboxId = 7;

/// The tenant's 6-bit witness slot (`w_slot < 64` is asserted by the ctor).
const TENANT_W_SLOT: u8 = 7;

/// Firing threshold handed to the ctor. Unused by this harness's read path
/// (`consume_firing` is never exercised — no `CausalEdge64` batons are delivered).
const TENANT_THRESHOLD: f32 = 1.0;

/// The single tenant type. **One** of these is constructed, ever.
type Tenant = MailboxSoA<N_CAP>;

// ── PRE-REGISTERED THRESHOLDS (fixed before the measurement was written) ────

/// Timed runs per (row count, mode). Median is reported. Inherited from W2 via
/// §12.3a′ ("median of ≥5 runs after one discarded warm-up") unchanged.
const RUNS: usize = 5;

/// Discarded warm-up runs per (row count, mode). Inherited from W2.
const WARMUPS: usize = 1;

/// **G-C.** Median concurrent-vs-sequential speedup required at the largest row
/// count. Inherited from W2's `≥2×` verbatim — §12.3a′ re-pins the *unit*
/// (owners → rows), not the threshold, so weakening it here would be fitting.
const SPEEDUP_GATE: f64 = 2.0;

/// **G-A.** Below this many hardware threads, a 2× gate is unreachable for
/// reasons that have nothing to do with the substrate, so G-C is not evaluated.
const MIN_THREADS_TO_EVALUATE: usize = 4;

/// **G-A.** Per-row body cost floor, in microseconds. W2's protocol specified
/// "≥100 µs bodies"; §12.3a′ carries the protocol over and re-pins the unit.
/// A body below this floor makes a speedup number a measurement of thread-spawn
/// overhead.
const BODY_FLOOR_US: f64 = 100.0;

/// **G-A.** Minimum sequential wall time (ms) at the largest row count. Below
/// this the sweep is too short to distinguish anything from scheduler noise.
const MIN_SEQ_WALL_MS: f64 = 50.0;

/// **G-B.** Sequential rows/second must agree within ±this fraction of the
/// median across all row counts for "rows/second" to be a meaningful unit. If
/// throughput is not flat, the row axis is not clean and **no scaling claim is
/// made** — that is a reportable outcome, not a failure.
const THROUGHPUT_FLATNESS: f64 = 0.25;

/// Anti-vacuity on the *instrument*: at least this fraction of the per-row
/// verdicts must be distinct values. A near-constant verdict vector cannot
/// detect a reordering or a lost update, which would make every equality check
/// below vacuous — the defect one level up.
const VERDICT_DISTINCT_FRACTION: f64 = 0.5;

/// Row counts swept. Clamped to the seated corpus; counts above it are dropped.
const ROW_COUNTS: [usize; 3] = [256, 1024, 2_000];

/// Default per-row body repetitions. Pinned so the default configuration's body
/// cost lands near [`BODY_FLOOR_US`] in an unoptimised build; the *measured*
/// cost is reported and G-A judges it. Override with `BLW_BODY_REPS` for a
/// labelled re-run.
const DEFAULT_BODY_REPS: usize = 48;

/// Upper bound on threads used, so a many-core host does not turn the thread
/// sweep into the dominant cost.
const MAX_THREADS: usize = 8;

/// The row-count floor above which the dirty-set sparseness assertion is
/// enforced. Stated rather than tuned: on a short corpus a data-dependent filter
/// can legitimately select almost everything, and asserting sparseness there
/// would be an assertion about the corpus prefix, not about the filter.
const SPARSENESS_FLOOR_ROWS: usize = 512;

// ── the byte image (the falsifier's instrument) ─────────────────────────────

/// Bytes of the per-row fixed columns: energy(4) + plasticity(1) +
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
/// [`snapshot`] cannot pass as "byte-identical".
const IMAGE_LEN: usize = SCALAR_IMG + N_CAP * ROW_IMG;

/// A **complete** little-endian byte image of the tenant's backing store: every
/// tenant scalar, then every per-row column of **every capacity row**
/// (`0..N_CAP`, not `0..populated` — a mutation to a padding row must be visible
/// too), including all three identity planes and all three style lanes.
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

// ── the row content encoding (a deterministic bloom, NOT a stance) ──────────

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

/// Set this token's [`BLOOM_K`] bits in a `WORDS_PER_FP`-word plane.
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

// ── the row-level thought body ──────────────────────────────────────────────

/// Evaluate ONE row against `probe`, `reps` times, and fold the results into a
/// verdict word.
///
/// **Reads only.** The plane is a **borrowed row slice** into the tenant's
/// backing store (`identity_plane_at` — zero-copy, `data-flow.md` §1); every
/// intermediate is an owned `Copy` microcopy (§2); nothing is written. The `&V`
/// receiver is the structural guarantee that this cannot mutate: *no `&mut self`
/// during computation*.
///
/// **Why the output is a mixing word and not a boolean.** The determinism and
/// equality checks below can only detect a reordering or a lost update if
/// adjacent rows produce *different* values. A boolean verdict would make a
/// swapped pair invisible. The rotation by `r` also makes each rep read the row
/// differently, so `reps` cannot be optimised down to one pass.
///
/// Returns `0` for a row the view declines (`row >= populated`), which the
/// callers never request.
fn row_body<V: MailboxSoaView>(view: &V, row: usize, probe: &[u64], reps: usize) -> u64 {
    let Some(plane) = view.identity_plane_at(row, IdentityPlane::Content) else {
        return 0;
    };
    let mut acc = u64::from(view.energy()[row].to_bits());
    for r in 0..reps {
        let salt = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut overlap: u32 = 0;
        for (p, w) in probe.iter().zip(plane) {
            overlap += (p & w.rotate_left((r % 64) as u32)).count_ones();
        }
        acc = acc.rotate_left(7) ^ u64::from(overlap).wrapping_add(salt);
    }
    acc
}

/// Sequential evaluation of `rows` row bodies. The baseline.
fn sweep_sequential<V: MailboxSoaView>(
    view: &V,
    probe: &[u64],
    rows: usize,
    reps: usize,
    out: &mut [u64],
) {
    assert_eq!(out.len(), rows, "output vector must be one slot per row");
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = row_body(view, row, probe, reps);
    }
}

/// Concurrent evaluation of `rows` row bodies **inside the one tenant**.
///
/// `threads` contiguous row chunks; every worker holds the SAME shared `&V` and
/// writes only into its own disjoint `&mut [u64]` slice (`chunks_mut`). There is
/// no lock, no atomic, and no `unsafe`: the read side needs none because it is
/// `&V`, and the write side is disjoint by slicing.
///
/// **The `V: Sync` bound is the compile-time proof.** `std::thread::scope`
/// requires the captured `&V` to be `Send`, which holds exactly when `V: Sync`.
/// Instantiating this at [`Tenant`] therefore makes "the tenant is safe to share
/// across the row workers" a fact the compiler checks, not a claim in a comment.
///
/// **This does not fabricate a tenant.** `view` is the one owner; the unit being
/// divided is its rows.
fn sweep_concurrent<V: MailboxSoaView + Sync>(
    view: &V,
    probe: &[u64],
    rows: usize,
    reps: usize,
    threads: usize,
    out: &mut [u64],
) {
    assert_eq!(out.len(), rows, "output vector must be one slot per row");
    if rows == 0 {
        return;
    }
    let chunk = rows.div_ceil(threads.max(1));
    // The chunk borrows are taken BEFORE entering the scope, deliberately: a
    // reborrow created inside the `scope` closure lives only for that closure
    // body, which does not satisfy the `'scope` bound `spawn` requires. Taken
    // here they belong to the enclosing environment and do satisfy it. (This is
    // a lifetime requirement, not a performance choice.)
    let chunks: Vec<&mut [u64]> = out.chunks_mut(chunk).collect();
    std::thread::scope(|s| {
        for (c, slot) in chunks.into_iter().enumerate() {
            let lo = c * chunk;
            s.spawn(move || {
                for (i, cell) in slot.iter_mut().enumerate() {
                    *cell = row_body(view, lo + i, probe, reps);
                }
            });
        }
    });
}

// ── comparators (they must be able to bark) ─────────────────────────────────

/// Index of the first element-by-element mismatch, or `None` when identical.
/// Length mismatch reports the shorter length as the offending index.
fn first_mismatch(a: &[u64], b: &[u64]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Number of distinct verdict values — the anti-vacuity measure for the
/// instrument itself. `O(n log n)` on a copy; the vectors are small.
fn distinct_count(v: &[u64]) -> usize {
    let mut c = v.to_vec();
    c.sort_unstable();
    c.dedup();
    c.len()
}

/// Median of a non-empty sample. Panics on NaN rather than silently ordering it.
fn median(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty(), "median of an empty sample");
    v.sort_by(|a, b| a.partial_cmp(b).expect("timing sample contained NaN"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ── the write descriptor `P` (DTO purity) ───────────────────────────────────

/// The `BatchWriter` payload — a **descriptor** (dirty row range + cycle), never
/// owned delta bytes (`batch_writer.rs` Addendum-6: the sink reads the LIVE
/// store at flush).
///
/// **It carries NO owner / mailbox / tenant field.** Ownership rides the *cast
/// pairing* (`BatchWriter::on_behalf_of`), never the DTO — the write-on-behalf
/// iron rule. The harness asserts that pairing rather than restating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowSpanDescriptor {
    /// First dirty row (inclusive).
    row_lo: u32,
    /// Last dirty row (exclusive).
    row_hi: u32,
    /// The owner cycle the span belongs to.
    cycle: u32,
}

// ── corpus + seeding ────────────────────────────────────────────────────────

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
/// The **builder** phase, not a compute path — `data-flow.md` allows `&mut` for
/// a builder. Row columns go through `write_row`, the SoA's ONE cycle-aware
/// mutator; `energy` is initialised through its public column because
/// `WriteCell` carries no energy field (production energy arrives via
/// `apply_edges` from `CausalEdge64` batons, which this harness has no source
/// for).
///
/// The **angle** plane is deliberately left all-zero: it is the silent region
/// the can-fire twin later mutates, which is the sharpest available test that
/// [`snapshot`] genuinely reads the identity planes.
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

// ── one measured point ──────────────────────────────────────────────────────

/// Median wall seconds of `RUNS` timed sweeps after `WARMUPS` discarded ones,
/// for one (row count, threads) cell.
///
/// The sequential result of each timed run is compared element-by-element
/// against the concurrent result of the same run, so **determinism under
/// concurrency is checked `RUNS + WARMUPS` times, not once**.
struct Timing {
    /// Median sequential wall, seconds.
    seq_s: f64,
    /// Median concurrent wall, seconds.
    conc_s: f64,
}

impl Timing {
    /// Sequential rows per second.
    fn seq_rows_per_s(&self, rows: usize) -> f64 {
        rows as f64 / self.seq_s
    }
    /// Concurrent rows per second.
    fn conc_rows_per_s(&self, rows: usize) -> f64 {
        rows as f64 / self.conc_s
    }
    /// Median-of-medians speedup. Reported, never adjusted.
    fn speedup(&self) -> f64 {
        self.seq_s / self.conc_s
    }
}

/// Time one (row count, threads) cell under both modes, asserting equality of
/// the two result vectors on **every** timed run.
fn measure(owner: &Tenant, probe: &[u64], rows: usize, reps: usize, threads: usize) -> Timing {
    let mut seq = vec![0u64; rows];
    let mut conc = vec![0u64; rows];
    let mut seq_t = Vec::with_capacity(RUNS);
    let mut conc_t = Vec::with_capacity(RUNS);

    for run in 0..(WARMUPS + RUNS) {
        let t0 = Instant::now();
        sweep_sequential(owner, probe, rows, reps, &mut seq);
        let seq_el = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        sweep_concurrent(owner, probe, rows, reps, threads, &mut conc);
        let conc_el = t1.elapsed().as_secs_f64();

        // Determinism under concurrency, checked on EVERY run: the concurrent
        // result must equal the sequential result element-by-element. A chunk
        // boundary off by one, a lost write, or a reordered store shows up here
        // as a named index — never as an aggregate that happens to match.
        if let Some(i) = first_mismatch(&seq, &conc) {
            panic!(
                "FALSIFIER FAILED: concurrent result diverged from sequential at row {i} \
                 (rows={rows}, threads={threads}, run={run}): seq={:#018x} conc={:#018x}",
                seq[i], conc[i]
            );
        }

        if run >= WARMUPS {
            seq_t.push(seq_el);
            conc_t.push(conc_el);
        }
    }

    Timing {
        seq_s: median(seq_t),
        conc_s: median(conc_t),
    }
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("BLW_KJV_TSV").unwrap_or_else(|_| "/tmp/kjv_verses.tsv".to_string());
    let limit = std::env::args()
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ROWS)
        .min(N_CAP);
    let reps_override = std::env::var("BLW_BODY_REPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|r| *r > 0);
    let reps = reps_override.unwrap_or(DEFAULT_BODY_REPS);
    let labelled_rerun = reps_override.is_some();

    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_THREADS);

    println!("== D-BLW-4 — N ROW-LEVEL thought bodies within ONE owner ==");
    println!("axis         : ROWS inside one tenant. NEVER owners. (§12.1a′ / §12.3a′)");
    println!("corpus       : {path} (bounded to {limit})");
    println!(
        "config       : body_reps {reps}{}, threads {threads} (cap {MAX_THREADS}), \
         runs {RUNS} + {WARMUPS} discarded warm-up",
        if labelled_rerun {
            " [LABELLED RE-RUN — gates were pre-registered for the default]"
        } else {
            " (default)"
        }
    );

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

    assert!(seated > 0, "no rows seated — nothing to evaluate");
    assert_eq!(
        owner.n_rows(),
        seated,
        "declared logical size == seated rows"
    );
    println!(
        "tenant       : mailbox {} w_slot {} — 1 owner, {} rows seated of {} capacity",
        owner.mailbox_id(),
        owner.w_slot,
        owner.n_rows(),
        N_CAP
    );
    println!(
        "image        : {IMAGE_LEN} B of MailboxSoA<{N_CAP}> ({} B/row of identity planes; \
         canon NodeRow = 512 B/row → {} B for {N_CAP} canon rows). \
         ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON unresolved — DIFFERENT structs.",
        3 * PLANE_BYTES,
        N_CAP * 512
    );

    let probe = probe_plane("god", 0);
    let probe_bits: u32 = probe.iter().map(|w| w.count_ones()).sum();
    assert!(
        probe_bits > 0,
        "an all-zero probe makes every body constant"
    );

    // ── PROBE-VERDICT — can this instrument detect anything at all? ──────────
    // Everything downstream is an equality check over the verdict vector. If the
    // verdicts were near-constant, every one of those checks would pass on a
    // broken sweep. So the discriminating power of the instrument is measured
    // BEFORE it is used as evidence.
    let mut baseline = vec![0u64; seated];
    sweep_sequential(&owner, &probe, seated, reps, &mut baseline);
    let distinct = distinct_count(&baseline);
    assert!(
        (distinct as f64) >= VERDICT_DISTINCT_FRACTION * seated as f64,
        "PROBE-VERDICT: only {distinct} distinct verdicts over {seated} rows — the instrument \
         cannot detect a reordering, so every equality check below would be vacuous"
    );
    println!(
        "PROBE-VERDICT: {distinct}/{seated} distinct row verdicts ({:.1}%) — the equality checks \
         below are non-vacuous",
        100.0 * distinct as f64 / seated as f64
    );

    // ── PROBE-DETECT — the comparator must bark, AND must stay silent ────────
    // Both halves on NON-TRIVIAL input (the real 2,000-row verdict vector, not
    // an empty one): a comparator that fires on everything carries exactly as
    // much information as one that never fires.
    let untouched = baseline.clone();
    assert_eq!(
        first_mismatch(&baseline, &untouched),
        None,
        "PROBE-DETECT (silent half): an identical non-trivial vector must compare equal"
    );

    let mut lost_update = baseline.clone();
    let victim_idx = seated / 3;
    // Non-vacuity of the probe itself: zeroing a slot only simulates an unwritten
    // slot if the real verdict there was NOT already zero.
    assert_ne!(
        baseline[victim_idx], 0,
        "the lost-update probe needs a non-zero verdict at row {victim_idx} to be observable"
    );
    lost_update[victim_idx] = 0; // a worker that never wrote its slot
    assert_eq!(
        first_mismatch(&baseline, &lost_update),
        Some(victim_idx),
        "PROBE-DETECT (lost update): a single unwritten slot must be located exactly"
    );

    assert!(
        seated >= 2 && baseline[0] != baseline[1],
        "PROBE-DETECT (reordering) needs ≥2 rows with DIFFERENT verdicts — a rotation is only \
         observable when adjacent verdicts differ"
    );
    let mut reordered = baseline.clone();
    reordered.rotate_left(1); // a chunk boundary off by one
    let rot_at = first_mismatch(&baseline, &reordered)
        .expect("PROBE-DETECT (reordering): a rotated result vector went undetected");
    println!(
        "PROBE-DETECT : silent on an identical vector; lost update located at {victim_idx}; \
         reordering located at {rot_at} — the comparator discriminates"
    );

    // ── PROBE-IRON — evaluating N rows CONCURRENTLY mutates NOTHING ──────────
    // The central falsifier, run over the concurrent path specifically (the one
    // that could race). Full byte image before and after; a difference anywhere
    // in 12 MiB fails and names its column.
    let pre_eval = snapshot(&owner);
    // The instrument must actually cover the store, or "byte-identical" is a
    // statement about what the snapshot forgot to read.
    assert_eq!(
        pre_eval.len(),
        IMAGE_LEN,
        "the byte image must cover every column of every capacity row"
    );
    let nonzero = pre_eval.iter().filter(|b| **b != 0).count();
    // Scale-free coverage gate: each seated row contributes ~250 non-zero bytes
    // (two bloom planes + the fixed columns), so `> seated * 8` holds at ANY
    // corpus size while still failing loudly for an effectively all-zero image,
    // where "byte-identical" would be trivially true.
    assert!(
        nonzero > seated * 8,
        "image is near-uniformly zero ({nonzero} non-zero of {} over {seated} rows) — \
         'byte-identical' would be trivially true",
        pre_eval.len()
    );
    let mut conc = vec![0u64; seated];
    sweep_concurrent(&owner, &probe, seated, reps, threads, &mut conc);
    let post_eval = snapshot(&owner);
    if let Some(off) = first_diff(&pre_eval, &post_eval) {
        panic!(
            "FALSIFIER FAILED: concurrently evaluating {seated} rows mutated the tenant at byte \
             {off} — {}",
            locate(off)
        );
    }
    assert_eq!(
        first_mismatch(&baseline, &conc),
        None,
        "the concurrent sweep must equal the sequential sweep element-by-element"
    );
    // Run the concurrent sweep a second time: the same inputs must produce the
    // identical vector. (Distinct from the equality above — that compares two
    // ALGORITHMS; this compares two RUNS of the same one.)
    let mut conc2 = vec![0u64; seated];
    sweep_concurrent(&owner, &probe, seated, reps, threads, &mut conc2);
    assert_eq!(
        first_mismatch(&conc, &conc2),
        None,
        "two runs of the concurrent sweep must be bit-identical"
    );
    println!(
        "PROBE-IRON   : {seated} rows evaluated concurrently on {threads} threads — tenant \
         byte-identical over {} B ({nonzero} non-zero, {:.2}%); concurrent == sequential; \
         two concurrent runs identical",
        pre_eval.len(),
        100.0 * nonzero as f64 / pre_eval.len() as f64
    );

    // ── the measurement grid: rows/second per ROW COUNT ─────────────────────
    let mut counts: Vec<usize> = Vec::new();
    for &c in &ROW_COUNTS {
        if c <= seated {
            counts.push(c);
        }
    }
    println!("--");
    println!("rows/second by ROW COUNT (one tenant; median of {RUNS} runs):");
    let mut cells: Vec<(usize, Timing)> = Vec::new();
    for &rows in &counts {
        let cell = measure(&owner, &probe, rows, reps, threads);
        println!(
            "  rows {rows:>5} : seq {:>10.0} rows/s ({:>8.2} ms)   conc {:>10.0} rows/s \
             ({:>8.2} ms)   speedup {:.2}x",
            cell.seq_rows_per_s(rows),
            cell.seq_s * 1e3,
            cell.conc_rows_per_s(rows),
            cell.conc_s * 1e3,
            cell.speedup()
        );
        cells.push((rows, cell));
    }

    // ── the thread sweep at the largest row count ───────────────────────────
    let largest = counts.last().copied().unwrap_or(seated);
    let mut tcs: Vec<usize> = vec![1, 2, threads];
    tcs.retain(|&t| t <= threads);
    tcs.sort_unstable();
    tcs.dedup();
    println!("--");
    println!("rows/second by THREAD COUNT at {largest} rows (T=1 = threading-overhead control):");
    for &t in &tcs {
        let cell = measure(&owner, &probe, largest, reps, t);
        println!(
            "  threads {t:>2} : conc {:>10.0} rows/s ({:>8.2} ms)   vs sequential {:>10.0} rows/s \
             → {:.2}x",
            cell.conc_rows_per_s(largest),
            cell.conc_s * 1e3,
            cell.seq_rows_per_s(largest),
            cell.speedup()
        );
    }

    // ── the iron rule, extended over the ENTIRE measured workload ───────────
    // PROBE-IRON above covered one sweep. This covers every sweep the grid and
    // the thread sweep just ran (tens of thousands of row bodies, on 1..T
    // threads) against the SAME pre-measurement image — so "evaluation mutates
    // nothing" is a statement about the workload that was timed, not about a
    // separate demonstration sweep.
    let post_grid = snapshot(&owner);
    if let Some(off) = first_diff(&pre_eval, &post_grid) {
        panic!(
            "FALSIFIER FAILED: the measurement grid mutated the tenant at byte {off} — {}",
            locate(off)
        );
    }
    println!("PROBE-IRON+  : the whole timed workload left the tenant byte-identical");

    // ── the pre-registered verdicts ─────────────────────────────────────────
    println!("--");
    let headline = cells.last();

    // G-A — can this design test the claim on this machine/profile at all?
    let (body_us, seq_wall_ms) = match headline {
        Some((rows, c)) => (c.seq_s * 1e6 / *rows as f64, c.seq_s * 1e3),
        None => (0.0, 0.0),
    };
    let body_ok = body_us >= BODY_FLOOR_US;
    let wall_ok = seq_wall_ms >= MIN_SEQ_WALL_MS;
    let threads_ok = threads >= MIN_THREADS_TO_EVALUATE;
    let ga = body_ok && wall_ok && threads_ok && headline.is_some();
    println!(
        "G-A precond  : {} — per-row body {body_us:.1} µs (floor {BODY_FLOOR_US:.0}) · \
         sequential wall {seq_wall_ms:.1} ms (floor {MIN_SEQ_WALL_MS:.0}) · \
         {threads} threads (floor {MIN_THREADS_TO_EVALUATE})",
        if ga { "MET" } else { "NOT MET" }
    );
    if !ga {
        let mut why: Vec<&str> = Vec::new();
        if !body_ok {
            why.push("body below floor (a speedup here would measure thread-spawn overhead)");
        }
        if !wall_ok {
            why.push("sweep too short to distinguish from scheduler noise");
        }
        if !threads_ok {
            why.push("too few hardware threads for a 2x gate to be reachable");
        }
        if headline.is_none() {
            why.push("no row count fit the corpus");
        }
        println!("               reasons: {}", why.join("; "));
    }

    // G-B — is the ROW axis clean? (sequential rows/s flat across row counts)
    if cells.len() >= 2 {
        let tputs: Vec<f64> = cells.iter().map(|(r, c)| c.seq_rows_per_s(*r)).collect();
        let med = median(tputs.clone());
        let worst = tputs
            .iter()
            .map(|t| ((t - med) / med).abs())
            .fold(0.0f64, f64::max);
        let flat = worst <= THROUGHPUT_FLATNESS;
        println!(
            "G-B row axis : {} — sequential rows/s deviates at most {:.1}% from the median \
             (allowed {:.0}%)",
            if flat { "PASS" } else { "NOT FLAT" },
            100.0 * worst,
            100.0 * THROUGHPUT_FLATNESS
        );
        if !flat {
            println!(
                "               ⇒ rows/second is NOT a clean unit on this run; no scaling claim \
                 is made on the row axis."
            );
        }
    } else {
        println!(
            "G-B row axis : INCONCLUSIVE — {} row count(s) measured, need ≥2 to say anything \
             about scaling",
            cells.len()
        );
    }

    // G-C — the claim itself.
    match (ga, headline) {
        (true, Some((rows, c))) => {
            let s = c.speedup();
            if s >= SPEEDUP_GATE {
                println!(
                    "G-C claim    : PASS — {s:.2}x ≥ {SPEEDUP_GATE:.1}x at {rows} rows on \
                     {threads} threads (READ half only)"
                );
            } else {
                println!(
                    "G-C claim    : KILL — {s:.2}x < {SPEEDUP_GATE:.1}x at {rows} rows on \
                     {threads} threads. Per §12.3a′ the claim REGRADES to \"{rows}-row \
                     SEQUENTIAL evaluation within one tenant\" — still true, different claim."
                );
            }
        }
        _ => println!(
            "G-C claim    : NOT EVALUATED (INCONCLUSIVE) — G-A's preconditions were not met, so \
             any speedup number here would not be about the substrate. Re-run with a larger \
             BLW_BODY_REPS, or on a host with ≥{MIN_THREADS_TO_EVALUATE} threads."
        ),
    }

    // ── the gated write-back — the half that is NOT parallel ────────────────
    // Read side done. This is the single-mutator half: `write_row` takes
    // `&mut self`, so it is sequential by type. The dirty set is data-dependent
    // (a property of the verdicts), not a fixed slice.
    println!("--");
    let dirty: Vec<u32> = conc
        .iter()
        .enumerate()
        .filter(|(_, v)| **v % 64 == 0)
        .map(|(row, _)| row as u32)
        .collect();

    if seated >= SPARSENESS_FLOOR_ROWS {
        // BOTH halves, and only above the stated floor: on a short corpus a
        // data-dependent filter can legitimately select nothing or everything,
        // and asserting sparseness there would be an assertion about the corpus.
        assert!(
            !dirty.is_empty(),
            "the dirty-set filter selected nothing over {seated} rows — degenerate verdicts"
        );
        assert!(
            dirty.len() * 2 < seated,
            "the dirty set must be a SPARSE minority: {} of {seated}",
            dirty.len()
        );
    }

    if dirty.is_empty() {
        println!("write-back   : SKIPPED — the filter selected no rows at this corpus size");
    } else {
        let pre_write = snapshot(&owner);
        let cycle = owner.cycle();
        let stamp = u64::from(cycle) << 32;
        let mut accepted = 0usize;
        for &row in &dirty {
            let cell = WriteCell {
                temporal: Some(stamp),
                ..WriteCell::default()
            };
            if owner.write_row(row as usize, cycle, &cell) == WriteOutcome::Accepted {
                accepted += 1;
            }
        }
        assert_eq!(accepted, dirty.len(), "every gated write must be Accepted");
        let post_write = snapshot(&owner);

        // Anti-vacuity, BOTH halves: the untouched remainder is byte-identical
        // AND the touched set genuinely changed.
        let mut changed = 0usize;
        let mut leaked: Vec<usize> = Vec::new();
        for row in 0..N_CAP {
            let differs = row_image(&pre_write, row) != row_image(&post_write, row);
            let is_dirty = dirty.binary_search(&(row as u32)).is_ok();
            if differs {
                changed += 1;
            }
            if differs != is_dirty {
                leaked.push(row);
            }
        }
        assert!(
            leaked.is_empty(),
            "only the gated dirty set may advance; divergent rows: {:?}",
            &leaked[..leaked.len().min(8)]
        );
        assert_eq!(changed, dirty.len(), "changed rows == dirty rows");
        assert!(
            changed > 0,
            "a write-back that changes nothing proves nothing"
        );
        println!(
            "write-back   : {} of {seated} rows gated through write_row (SEQUENTIAL — \
             `&mut self`, single mutator); untouched remainder byte-identical",
            dirty.len()
        );

        // ── the pre-write cast — write-on-behalf, ahead of the write ────────
        // NO seal, NO applied step: `no successful write ⇒ no applied step`
        // (owner_adapter module doc). D-BLW-1 owns that loop; duplicating it
        // here would add wall time and no new evidence.
        let mut writer: BatchWriter<RowSpanDescriptor> = BatchWriter::new();
        let span = RowSpanDescriptor {
            row_lo: dirty[0],
            row_hi: dirty[dirty.len() - 1] + 1,
            cycle,
        };
        let outcome = StrategyOutcome {
            reliability: (dirty.len() as f32) / (seated as f32),
            intended_move: Some(bootstrap_intent(
                KanbanColumn::Planning,
                KanbanColumn::CognitiveWork,
            )),
        };
        // The owner id is READ FROM THE OWNER at the call site, so the cast
        // cannot name a mailbox other than the SoA it describes.
        let cast = emit_bootstrap_intent(&outcome, owner.mailbox_id(), cycle, &mut writer, span)
            .expect("a bootstrap sentinel must rebind and cast");
        let cast_owner = writer
            .on_behalf_of(cast)
            .expect("the cast is recorded on the writer");
        assert_eq!(
            cast_owner,
            owner.mailbox_id(),
            "write-on-behalf of the live owner"
        );
        let moves = writer
            .intent_moves(cast)
            .expect("the cast recorded its intent");
        assert_eq!(moves.len(), 1, "one lifecycle intent per cast");
        // Anti-vacuity on the rebind: the sentinel fields ACTUALLY moved.
        assert_ne!(moves[0].mailbox, 0, "owner sentinel was rebound");
        assert_eq!(moves[0].mailbox, owner.mailbox_id());
        assert_ne!(moves[0].witness_chain_position, 0, "cycle sentinel rebound");
        assert_eq!(moves[0].witness_chain_position, cycle);

        // The descriptor round-trips through the writer UNCHANGED and still
        // carries no owner field — DTO purity, checked rather than asserted in
        // prose. `drain_pending_payloads` is the shipped eager-drain handoff.
        let drained: Vec<(_, RowSpanDescriptor)> = writer.drain_pending_payloads().collect();
        assert_eq!(drained.len(), 1, "one staged payload per cast");
        assert_eq!(drained[0].0, cast, "payload stays paired with its cast id");
        assert_eq!(
            drained[0].1, span,
            "the descriptor is not rewritten in transit"
        );
        // Every descriptor field is read here — the struct carries an owner
        // NOWHERE, which is the point: ownership rode the cast pairing above.
        println!(
            "cast         : rows [{}..{}) cycle {} cast on behalf of mailbox {cast_owner}; \
             phase left at {:?} (no seal ⇒ no applied step, by design)",
            span.row_lo,
            span.row_hi,
            span.cycle,
            owner.phase()
        );
    }

    // ── THE CAN-FIRE TWIN — the byte comparator must DETECT deliberate change ─
    // A guard that cannot bark is the defect one level up. Two probes, chosen so
    // a snapshot that skipped a region cannot pass:
    //   (a) a small fixed column (`meta`);
    //   (b) the ANGLE identity plane, all-zero for the entire run — a snapshot
    //       that never reads it would report "identical".
    let cycle_now = owner.cycle();
    let victim = seated / 2;

    let pre_mut = snapshot(&owner);
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
    assert!(
        where_fixed.contains("fixed columns"),
        "the detected difference must be in the fixed columns, got {where_fixed}"
    );
    println!("PROBE-MUT-a  : gated one-column write to row {victim} detected — {where_fixed}");

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
    println!("PROBE-MUT-b  : 1-bit ANGLE-plane mutation detected — {where_angle}");

    // ── summary ─────────────────────────────────────────────────────────────
    println!("--");
    println!(
        "tenants      : 1 (never N) — mailbox {}, phase {:?}, cycle {}",
        owner.mailbox_id(),
        owner.phase(),
        owner.cycle()
    );
    println!(
        "parallelised : the READ half only (`&V: MailboxSoaView`, borrowed row slices). \
         The WRITE half is `write_row` on `&mut self` — single-mutator by construction, \
         NOT parallel, and no speedup is claimed for it."
    );
    println!(
        "NOT PROVEN   : durability (nothing is sealed here); any lifecycle advance (no seal ⇒ \
         no applied step); `deinterlace`/`DeinterlaceRow` (no production implementor exists); \
         any stance or semantic claim; and any statement about the 512 B canonical NodeRow — \
         every byte figure above is a figure of MailboxSoA<{N_CAP}>."
    );
    Ok(())
}
