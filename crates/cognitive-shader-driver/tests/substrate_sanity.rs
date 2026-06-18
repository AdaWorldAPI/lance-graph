//! Substrate sanity harness — "is the substrate NaN-free and non-tautological?"
//!
//! Two failure classes this guards against:
//!
//! 1. **NaN/Inf** — any f32 surface (qualia f32 projection, energy accumulator)
//!    that produces a non-finite value silently poisons every downstream
//!    cosine / distance / free-energy read.
//! 2. **Tautology** — a substrate operation that is *trivially true*: a write
//!    gate that always accepts, a qualia projection that collapses every input
//!    to one vector, a write that ignores its field-presence. A green test that
//!    only ever exercises the trivial path certifies nothing.
//!
//! These run on the DEFAULT build (no feature needed): `MailboxSoA` is an
//! unconditional `pub mod`, and the cycle-aware `write_row` gate lives on it.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::qualia::QualiaI4_16D;

fn content_plane(seed: u64) -> Vec<u64> {
    let mut c = vec![0u64; WORDS_PER_FP];
    c[0] = seed;
    c[WORDS_PER_FP - 1] = seed.wrapping_mul(0x9E37_79B9);
    c
}

// ─────────────────────────────── NaN / Inf ────────────────────────────────

/// The qualia f32 projection is finite for every i4 value across every dim —
/// including the signed extremes. A NaN here poisons all cosine reads.
#[test]
fn qualia_f32_projection_is_finite_over_full_i4_range() {
    for dim in 0..16usize {
        for v in -8i8..=7 {
            let q = QualiaI4_16D::ZERO.with(dim, v);
            let f = q.to_f32_17d();
            for (i, x) in f.iter().enumerate() {
                assert!(
                    x.is_finite(),
                    "qualia dim={dim} val={v} produced non-finite f32 at out[{i}] = {x}"
                );
            }
        }
    }
    // All dims at the extreme simultaneously.
    let mut q = QualiaI4_16D::ZERO;
    for dim in 0..16usize {
        q = q.with(dim, if dim % 2 == 0 { 7 } else { -8 });
    }
    assert!(
        q.to_f32_17d().iter().all(|x| x.is_finite()),
        "all-extreme qualia produced a non-finite f32"
    );
}

/// The energy accumulator stays finite through write/consume; `consume_firing`
/// resets a fired row to a finite 0.0 (not NaN).
#[test]
fn energy_accumulator_stays_finite_through_consume() {
    let mut mb: MailboxSoA<8> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(8);
    mb.energy[0] = 3.5;
    mb.energy[1] = -2.0;
    assert!(mb.energy.iter().all(|e| e.is_finite()));

    assert!(mb.consume_firing(0), "row 0 above threshold should fire");
    assert!(
        mb.energy.iter().all(|e| e.is_finite()),
        "consume must leave all energies finite"
    );
    assert_eq!(mb.energy_at(0), 0.0, "fired row resets to a finite 0.0");
}

// ─────────────────────────────── Tautology ────────────────────────────────

/// THE core anti-tautology: the cycle gate must DISCRIMINATE — accept the
/// current cycle, reject stale (older) and future (newer). A gate that always
/// returns `Accepted` is a tautology that re-opens the stale-overwrite hole.
#[test]
fn write_gate_discriminates_current_stale_future() {
    let mut mb: MailboxSoA<8> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(8);

    let m_a = MetaWord::new(1, 1, 100, 100, 0);
    let cell_a = WriteCell {
        meta: Some(m_a),
        ..Default::default()
    };

    // current_cycle == 0: accepted, stamped.
    assert_eq!(mb.write_row(0, 0, &cell_a), WriteOutcome::Accepted);
    assert_eq!(mb.last_write_cycle_at(0), 0);
    assert_eq!(mb.meta_at(0), m_a, "accepted write applied the cell");

    // Advance to cycle 1; a write for cycle 0 is now STALE — must NOT apply.
    mb.tick();
    let m_stale = MetaWord::new(2, 2, 200, 200, 0);
    let cell_stale = WriteCell {
        meta: Some(m_stale),
        ..Default::default()
    };
    assert_eq!(mb.write_row(0, 0, &cell_stale), WriteOutcome::Stale);
    assert_eq!(mb.stale_write_count(), 1, "stale write counted");
    assert_eq!(
        mb.meta_at(0),
        m_a,
        "STALE write must NOT overwrite the row (no cycle-blind clobber)"
    );

    // A write for cycle 5 (current is 1) is FUTURE — must NOT apply.
    let cell_future = WriteCell {
        meta: Some(MetaWord::new(3, 3, 50, 50, 0)),
        ..Default::default()
    };
    assert_eq!(mb.write_row(0, 5, &cell_future), WriteOutcome::Future);
    assert_eq!(mb.meta_at(0), m_a, "FUTURE write must NOT apply");
    assert_eq!(mb.stale_write_count(), 1, "future is not counted as stale");
}

/// The gate is wrap-aware: after `current_cycle` wraps past `u32::MAX`, a write
/// from the pre-wrap epoch is still classified STALE (not mis-read as Future).
#[test]
fn write_gate_is_wrap_aware() {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(4);
    // Force the clock near the wrap boundary.
    mb.current_cycle = 1; // pretend we just wrapped 0xFFFF_FFFF -> 0 -> 1
    let cell = WriteCell {
        meta: Some(MetaWord::new(1, 1, 10, 10, 0)),
        ..Default::default()
    };
    // A straggler stamped 0xFFFF_FFFF (one epoch behind) must be STALE, not Future.
    assert_eq!(
        mb.write_row(0, u32::MAX, &cell),
        WriteOutcome::Stale,
        "pre-wrap straggler must be stale, not future"
    );
}

/// Field-presence is honoured: a `WriteCell` that sets only `meta` must leave
/// `qualia` untouched. A write that clobbers every column regardless of
/// presence is a tautology.
#[test]
fn write_cell_field_presence_is_not_a_tautology() {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(4);
    let q0 = mb.qualia_at(0);
    let cell = WriteCell {
        meta: Some(MetaWord::new(5, 5, 1, 1, 0)),
        ..Default::default() // qualia/content/edge = None
    };
    assert_eq!(mb.write_row(0, 0, &cell), WriteOutcome::Accepted);
    assert_eq!(
        mb.qualia_at(0),
        q0,
        "a meta-only WriteCell must NOT touch qualia (field-presence honoured)"
    );
}

/// Writes carry DISTINCT data: two rows written with different content read back
/// different. A substrate that collapses distinct writes to one value is a
/// degenerate (tautological) store.
#[test]
fn distinct_writes_read_back_distinct() {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(4);
    let c0 = content_plane(0x1111);
    let c1 = content_plane(0x2222);
    let w0 = WriteCell {
        content: Some(&c0),
        ..Default::default()
    };
    let w1 = WriteCell {
        content: Some(&c1),
        ..Default::default()
    };
    assert_eq!(mb.write_row(0, 0, &w0), WriteOutcome::Accepted);
    assert_eq!(mb.write_row(1, 0, &w1), WriteOutcome::Accepted);
    assert_ne!(
        mb.content_row(0),
        mb.content_row(1),
        "distinct content writes must NOT collapse to one value"
    );
    assert_eq!(mb.content_row(0), &c0[..]);
}

/// The qualia projection discriminates: opposite i4 inputs map to different
/// f32 vectors (the projection is not a constant tautology).
#[test]
fn qualia_projection_discriminates() {
    let pos = QualiaI4_16D::ZERO.with(0, 5).to_f32_17d();
    let neg = QualiaI4_16D::ZERO.with(0, -5).to_f32_17d();
    assert_ne!(pos, neg, "opposite qualia must not project to the same f32");
}

/// An edge with zero mantissa contributes zero energy (sanity: the accumulator
/// is not fabricating energy from nothing — would be a tautological "always
/// firing" substrate).
#[test]
fn zero_edge_contributes_no_energy() {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(0, 0, 1.0);
    mb.set_populated(4);
    let accepted = mb.apply_edges(&[(0u16, CausalEdge64::ZERO)]);
    assert_eq!(accepted, 1, "w_slot-0 zero edge is accepted (same corpus)");
    assert_eq!(
        mb.energy_at(0),
        0.0,
        "a zero-mantissa edge must add zero energy (no fabricated firing)"
    );
}
