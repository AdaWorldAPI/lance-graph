//! PROBE-TARSKI-SIGNED-WITNESS-1 — is `Belief::rung` an impoverished
//! projection of a signed derivational field?
//!
//! **The claim under test (operator, 2026-08-23).** The scalar Tarski rung
//! (`Belief::rung: u32`) answers only "how far did the successful proof
//! climb?" A `24×i4` signed witness field over the SAME 16-byte dock
//! (`classid(4) + 12-byte register`) could answer "where did the proof climb,
//! where did it fail, and where does a counter-proof bite?" — with
//! `+n` = constructive derivation depth, `-n` = falsifying derivation depth,
//! `0` = absent/unresolved, per locus.
//!
//! **This is a probe-local ClassView READING, not a new tenant and not a
//! mint.** The physical carrier is the shipped `G24N4` nibble law
//! ([`CausalWitnessFacet`], `causal_witness.rs`) over the shipped 16-byte
//! dock ([`FacetCascade`]). The shipped A9 reading is operator-locked
//! "loci, not magnitudes" — THIS reading (sign = support vs falsification,
//! magnitude = derivational depth) is a DIFFERENT ClassView over the same
//! bytes, exactly as the "one register, three readings" doctrine (PR #729)
//! provides. The probe-local classid below is a placeholder, NOT an OGAR
//! mint; nothing here canonizes.
//!
//! **⚠ MEMORY-ABI ESCAPE, acknowledged in place.** `BeliefArena` remains the
//! acknowledged non-canonical AoS owner (#1004,
//! `E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1`); this probe uses it as
//! the PARITY ORACLE only. The witness docks are a fixed stack array of
//! 16-byte Copy rows — no heap row population, no hash, no materialized
//! mask, no new backing store beyond the dock rows themselves. This probe
//! informs the `BELIEF-ABI-RESTORATION-1` charter's step-2 ruling on `rung`;
//! it does NOT execute the charter's step 3.
//!
//! # Fixtures (all gates must be able to fail)
//!
//! - **A** — ordinary positive derivation chain, no falsifier. The G24N4
//!   positive support projection must reproduce `Belief::rung` exactly.
//! - **B** — same chain plus a freq-0 falsifying counter-chain reaching the
//!   apex statement S. The shipped merge law (`admit_derived` CHOICE,
//!   `belief.rs:247`: a pure-derived candidate replaces only on STRICT
//!   expectation gain) silently DROPS the counter-derivation — so the
//!   arena's scalar state at S is bit-identical to fixture A. The witness
//!   field alone retains the falsifier, without disturbing the support lane.
//! - **B'** — a DEEPER falsifying counter-chain. The arena's scalar state at
//!   S is provably identical between B and B' (zero bits of falsifier depth
//!   survive); the witness distinguishes them.
//! - **C** — falsifier removed. The witness register returns byte-identical
//!   to fixture A: the negative lane vanishes, the positive lane is exact.
//!
//! Physical conservation gates: same 16-byte dock, same LE geometry, G24N4
//! reading only, borrowed zero-copy register views, i4 ceiling stated
//! honestly (+7/−8 — derivations deeper than rung 7 need escalation, not a
//! wider nibble).

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, WITNESS_REGISTER_BYTES};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::facet::FacetCascade;
use lance_graph_planner::nars::belief::{BeliefArena, Belief, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;

/// Probe-local ClassView id for the Tarski signed-witness READING.
/// Deliberately in a nonsense range: this is NOT an OGAR mint.
const PROBE_TARSKI_CLASSID: u32 = 0xFFFF_0009;

/// Dock capacity — fixed stack array, no heap row population.
const MAX_DOCKS: usize = 64;

/// One 16-byte dock row: `classid(4, LE) | 24×i4 register(12)`.
type Dock = [u8; 16];

const _: () = assert!(core::mem::size_of::<Dock>() == 16, "16-byte dock");
const _: () = assert!(
    core::mem::size_of::<CausalWitnessFacet>() == WITNESS_REGISTER_BYTES,
    "G24N4 register is the dock payload width"
);
const _: () = assert!(
    core::mem::size_of::<FacetCascade>() == 16,
    "FacetCascade shares the dock width"
);

const EMPTY_DOCK: Dock = [0u8; 16];

/// Borrowed zero-copy G24N4 view of a dock's 12-byte payload
/// (`CausalWitnessFacet::from_register_ref` — a pointer reborrow, no copy).
fn witness_of(dock: &Dock) -> &CausalWitnessFacet {
    let reg: &[u8; WITNESS_REGISTER_BYTES] = dock[4..16].try_into().expect("12-byte payload");
    CausalWitnessFacet::from_register_ref(reg)
}

/// Write-back path per the borrow-strategy law: owned Copy microcopy of the
/// 12-byte register, nibble edit, gated write-back. Never `&mut` a view.
fn set_locus(dock: &mut Dock, locus: Locus, offset: i8) {
    let mut w = *witness_of(dock);
    w.set(locus as usize, offset);
    dock[4..16].copy_from_slice(&w.to_register());
}

fn mint_dock() -> Dock {
    let mut d = EMPTY_DOCK;
    d[0..4].copy_from_slice(&PROBE_TARSKI_CLASSID.to_le_bytes());
    d
}

/// The legacy projection: `support_ceiling = max(positive loci)`.
/// For the positive-only corpus this must equal `Belief::rung` exactly
/// (observed ground = rung 0 = no bound positive locus).
fn support_ceiling(w: &CausalWitnessFacet) -> u32 {
    (0..24).map(|s| w.get(s).max(0) as u32).max().unwrap_or(0)
}

/// `falsifier_ceiling = max(abs(negative loci))`.
fn falsifier_ceiling(w: &CausalWitnessFacet) -> u32 {
    (0..24).map(|s| (-w.get(s)).max(0) as u32).max().unwrap_or(0)
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

/// Scan the closed arena for falsifying derivations the scalar path DROPPED:
/// composable premise pairs whose candidate conclusion targets an existing
/// belief of opposing polarity (stored expectation > 0.5, candidate ≤ 0.5).
/// Returns, per target statement, the DEEPEST such counter-derivation
/// (`max(r1, r2) + 1` — the same depth law `close_transitive` uses).
///
/// This is the "falsifier listener": it re-derives with the arena's own
/// public truth functions what `admit_derived`'s CHOICE discards silently.
fn scan_falsifiers(arena: &BeliefArena) -> Vec<(CStmt, u32)> {
    let entries = arena.entries();
    let mut out: Vec<(CStmt, u32)> = Vec::new();
    for ei in entries {
        if !ei.stmt.cop.transits() {
            continue;
        }
        for ej in entries {
            if ej.stmt.cop != ei.stmt.cop || ej.stmt.s != ei.stmt.p {
                continue;
            }
            let stmt = CStmt {
                s: ei.stmt.s,
                cop: ei.stmt.cop,
                p: ej.stmt.p,
            };
            let Some(stored) = arena.get(stmt) else {
                continue;
            };
            let cand = ei.truth.deduction(&ej.truth);
            if stored.truth.expectation() > 0.5 + 1e-6 && cand.expectation() <= 0.5 + 1e-6 {
                let depth = ei.rung.max(ej.rung) + 1;
                match out.iter_mut().find(|(s, _)| *s == stmt) {
                    Some((_, d)) => *d = (*d).max(depth),
                    None => out.push((stmt, depth)),
                }
            }
        }
    }
    out
}

/// Mint the witness docks for a closed arena: positive lane
/// (`Locus::SupportedBy`) = derivation rung; negative lane
/// (`Locus::Contradiction`) = deepest dropped falsifying derivation.
/// Panics (probe-honesty) if any depth exceeds the i4 ceiling.
fn mint_witness_docks(arena: &BeliefArena) -> ([Dock; MAX_DOCKS], usize) {
    let mut docks = [EMPTY_DOCK; MAX_DOCKS];
    let n = arena.entries().len();
    assert!(n <= MAX_DOCKS, "probe capacity");
    for (i, b) in arena.entries().iter().enumerate() {
        docks[i] = mint_dock();
        if b.rung > 0 {
            assert!(b.rung <= 7, "i4 ceiling: rung {} > +7 needs escalation", b.rung);
            set_locus(&mut docks[i], Locus::SupportedBy, b.rung as i8);
        }
    }
    for (stmt, depth) in scan_falsifiers(arena) {
        assert!(depth <= 8, "i4 ceiling: falsifier depth {depth} > 8");
        let idx = arena
            .entries()
            .iter()
            .position(|b| b.stmt == stmt)
            .expect("falsifier target exists");
        set_locus(&mut docks[idx], Locus::Contradiction, -(depth as i8));
    }
    (docks, n)
}

/// Positive fixture: the 4-link chain `1→2→3→4→5`, f=1.0, c=0.9, disjoint
/// stamps, closed to fixed point.
fn build_positive_arena() -> BeliefArena {
    let mut a = BeliefArena::new();
    for (k, (s, p)) in [(1u16, 2u16), (2, 3), (3, 4), (4, 5)].iter().enumerate() {
        a.observe(
            inh(*s, *p),
            TruthValue::new(1.0, 0.9),
            Stamp::source(k as u32 + 1),
        );
    }
    a.close_transitive(16);
    assert!(a.reached_fixed_point, "fixture chain must close");
    a
}

/// Add a falsifying counter-chain `1→…→5` whose LAST link has frequency 0,
/// then re-close. `hops` = number of links (3 → counter-depth 2; 5 → 3).
fn add_falsifier_chain(a: &mut BeliefArena, hops: usize) {
    // concepts 6.. are counter-branch intermediates
    let mut nodes: Vec<u16> = vec![1];
    for i in 0..hops - 1 {
        nodes.push(6 + i as u16);
    }
    nodes.push(5);
    for (k, w) in nodes.windows(2).enumerate() {
        let f = if k == hops - 1 { 0.0 } else { 1.0 };
        a.observe(
            inh(w[0], w[1]),
            TruthValue::new(f, 0.9),
            Stamp::source(10 + k as u32),
        );
    }
    a.close_transitive(16);
    assert!(a.reached_fixed_point, "falsifier closure must fix");
}

/// Bit-exact scalar-state snapshot of one belief (f32s by bit pattern).
///
/// Premises are EXCLUDED from the snapshot — a measured necessity, not a
/// convenience, and itself charter evidence (BELIEF-ABI-RESTORATION-1 F4:
/// ephemeral indexes leaking into cognitive state). Two IDENTICALLY-built
/// arenas diverge in `premises` at TWO layers, both from `close_transitive`
/// admitting its per-pass `derived: HashMap` in iteration order:
/// 1. **index values** — the same derivation stores different numeric arena
///    indices (observed live: `(1,5)` premises `[6,4]` vs `[5,6]`);
/// 2. **route ties** — equal-expectation derivations (`(1,4)` via
///    `(1,3)∘(3,4)` vs `(1,2)∘(2,4)`) are tie-broken by whichever the
///    HashMap yields first, so even statement-RESOLVED premises differ.
///
/// The epistemically stable scalar state is (stmt, truth, stamp, rung,
/// contradiction); the premise route is nondeterministic decoration.
fn scalar_state(b: &Belief) -> (CStmt, u32, u32, u64, u32, u32) {
    (
        b.stmt,
        b.truth.frequency.to_bits(),
        b.truth.confidence.to_bits(),
        b.stamp.0,
        b.rung,
        b.contradiction.to_bits(),
    )
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ================= Fixture A =================
    let arena_a = build_positive_arena();
    let (docks_a, n_a) = mint_witness_docks(&arena_a);

    // A1 — hand-authored rung expectations (balanced composition: rung is
    // ~log2 of span, NOT span length — the #1002 arc's law, restated here).
    let expected: [((u16, u16), u32); 10] = [
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 4), 0),
        ((4, 5), 0),
        ((1, 3), 1),
        ((2, 4), 1),
        ((3, 5), 1),
        ((1, 4), 2),
        ((2, 5), 2),
        ((1, 5), 2),
    ];
    let mut a1_ok = arena_a.entries().len() == expected.len();
    for ((s, p), r) in expected {
        a1_ok &= arena_a.get(inh(s, p)).map(|b| b.rung) == Some(r);
    }
    gate(
        "A1 authored rung fixture",
        a1_ok,
        format!("{} statements, rungs 0/1/2 as authored", expected.len()),
    );

    // A2 — legacy projection parity + negative lane silent in fixture A.
    let mut a2_ok = true;
    for (i, b) in arena_a.entries().iter().enumerate() {
        let w = witness_of(&docks_a[i]);
        a2_ok &= support_ceiling(w) == b.rung;
        a2_ok &= falsifier_ceiling(w) == 0;
    }
    gate(
        "A2 support projection == Belief::rung, falsifier lane silent",
        a2_ok,
        format!("{n_a}/{n_a} docks: support_ceiling parity, zero negatives"),
    );

    // A3 — dock conservation: LE geometry, classid recovery via the shipped
    // FacetCascade reading, register identity through the borrowed view.
    let mut a3_ok = true;
    for dock in docks_a.iter().take(n_a) {
        let fc = FacetCascade::from_bytes(dock);
        a3_ok &= fc.facet_classid == PROBE_TARSKI_CLASSID;
        // Full LE round-trip through the shipped classid reading: the dock's
        // bytes survive the FacetCascade decode/encode unchanged. (NOT
        // `tier_bytes()` — that is the coarse→fine hi:lo LADDER view; the
        // wire order per tier is `[lo, hi]`, from_bytes' own doc.)
        a3_ok &= fc.to_bytes() == *dock;
        a3_ok &= witness_of(dock).to_register() == dock[4..16];
    }
    gate(
        "A3 dock conservation (classid LE + register identity, both readings)",
        a3_ok,
        format!("{n_a} docks read identically as FacetCascade and G24N4 view"),
    );

    let s_apex = inh(1, 5);
    let apex_idx = arena_a
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex");
    let apex_a_state = scalar_state(&arena_a.entries()[apex_idx]);
    let apex_a_dock = docks_a[apex_idx];

    // ================= Fixture B (falsifier, depth 2) =================
    let mut arena_b = build_positive_arena();
    add_falsifier_chain(&mut arena_b, 3);
    let (docks_b, _n_b) = mint_witness_docks(&arena_b);

    // B1 — the falsifier is INVISIBLE to the scalar state: every fixture-A
    // belief (apex included) is bit-identical after the counter-chain. This
    // is the shipped CHOICE law doing the discarding (belief.rs:247).
    let mut b1_ok = true;
    for b in arena_a.entries() {
        let after = arena_b.get(b.stmt).expect("fixture-A statement survives");
        b1_ok &= scalar_state(after) == scalar_state(b);
    }
    gate(
        "B1 falsifier invisible in scalar state (CHOICE drops it)",
        b1_ok,
        format!("{n_a}/{n_a} fixture-A beliefs bit-identical post-falsifier"),
    );

    // B2 — the witness RETAINS what the scalar dropped, without touching the
    // support lane: apex dock differs from fixture A in EXACTLY the
    // Contradiction nibble; +2 support survives beside −2 falsifier.
    let apex_b_idx = arena_b
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex in B");
    let apex_b_dock = docks_b[apex_b_idx];
    let wb = witness_of(&apex_b_dock);
    let changed_slots: Vec<usize> = (0..24)
        .filter(|&s| wb.get(s) != witness_of(&apex_a_dock).get(s))
        .collect();
    gate(
        "B2 falsifier lands beside support (neither overwrites the other)",
        wb.at(Locus::SupportedBy) == 2
            && wb.at(Locus::Contradiction) == -2
            && changed_slots == vec![Locus::Contradiction as usize],
        format!(
            "apex: SupportedBy {:+}, Contradiction {:+}, changed slots {:?}",
            wb.at(Locus::SupportedBy),
            wb.at(Locus::Contradiction),
            changed_slots
        ),
    );

    // B3 — can-stay-silent: the falsifier channel fires ONLY at the apex.
    let hotspots: Vec<CStmt> = arena_b
        .entries()
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let w = witness_of(&docks_b[*i]);
            support_ceiling(w) > 0 && falsifier_ceiling(w) > 0
        })
        .map(|(_, b)| b.stmt)
        .collect();
    gate(
        "B3 dialectical hotspot detection is discriminating",
        hotspots == vec![s_apex],
        format!("exactly one hotspot (the apex), {} beliefs scanned", arena_b.entries().len()),
    );

    // ================= Fixture B' (falsifier, depth 3) =================
    let mut arena_b2 = build_positive_arena();
    add_falsifier_chain(&mut arena_b2, 5);
    let (docks_b2, _) = mint_witness_docks(&arena_b2);
    let apex_b2_idx = arena_b2
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex in B'");
    let wb2 = witness_of(&docks_b2[apex_b2_idx]);

    // B4 — the depth discrimination the scalar CANNOT make: apex scalar
    // state is identical between depth-2 and depth-3 falsifier worlds
    // (zero bits of falsifier depth survive in the arena), while the
    // witness field distinguishes them.
    let b_state = scalar_state(&arena_b.entries()[apex_b_idx]);
    let b2_state = scalar_state(&arena_b2.entries()[apex_b2_idx]);
    gate(
        "B4 scalar state identical across falsifier depths; witness differs",
        b_state == b2_state
            && b_state == apex_a_state
            && wb.at(Locus::Contradiction) == -2
            && wb2.at(Locus::Contradiction) == -3,
        format!(
            "apex scalar bit-equal in A/B/B'; witness −2 vs {:+}",
            wb2.at(Locus::Contradiction)
        ),
    );

    // ================= Fixture C (falsifier removed) =================
    let arena_c = build_positive_arena();
    let (docks_c, n_c) = mint_witness_docks(&arena_c);
    // Per-STATEMENT comparison, not per-index: admission order is HashMap-
    // nondeterministic (the F4 finding above), so entry i of two identical
    // builds need not be the same statement.
    let mut c1_ok = n_c == n_a;
    for (i, b) in arena_a.entries().iter().enumerate() {
        let ci = arena_c
            .entries()
            .iter()
            .position(|cb| cb.stmt == b.stmt)
            .expect("fixture-C statement set matches A");
        c1_ok &= docks_c[ci] == docks_a[i];
    }
    gate(
        "C1 falsifier removal restores byte-identical docks",
        c1_ok,
        format!("{n_a} docks byte-equal to fixture A (negative lane vanished)"),
    );

    // ================= Physical honesty gates =================
    // P1 — the i4 ceiling is +7/−8: depth beyond it CLAMPS, it does not
    // widen. Deeper derivations need escalation (ClassView re-election /
    // bucket rollover), never a wider nibble in this reading.
    let mut probe = mint_dock();
    set_locus(&mut probe, Locus::SupportedBy, 9);
    let clamped_hi = witness_of(&probe).at(Locus::SupportedBy);
    set_locus(&mut probe, Locus::Contradiction, -9);
    let clamped_lo = witness_of(&probe).at(Locus::Contradiction);
    gate(
        "P1 i4 depth ceiling stated honestly (+7/−8, clamp not widen)",
        clamped_hi == 7 && clamped_lo == -8,
        format!("+9 → {clamped_hi:+}, −9 → {clamped_lo:+}"),
    );

    // P2 — masked election over the signed field is fail-closed and
    // discriminating: electing ONLY the support lane reads +2 at the apex
    // and refuses the bound-but-unelected falsifier lane.
    let support_mask = WideFieldMask::from_positions(&[Locus::SupportedBy as u8]);
    gate(
        "P2 locus election reads support, refuses unelected falsifier",
        wb.elected(&support_mask, Locus::SupportedBy) == Some(2)
            && wb.elected(&support_mask, Locus::Contradiction).is_none()
            && wb.elected(&WideFieldMask::EMPTY, Locus::SupportedBy).is_none(),
        "elected(SupportedBy)=+2; Contradiction unelected → None; EMPTY fail-closed".to_string(),
    );

    println!("PROBE-TARSKI-SIGNED-WITNESS-1: ALL {pass} GATES GREEN");
    println!(
        "verdict: Belief::rung is the positive-lane projection of a signed \
         G24N4 derivational field; the falsifier lane carries what the \
         shipped CHOICE law provably discards"
    );
}
