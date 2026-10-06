//! **Round 6: one SPOFC → do → learn cycle as a compile.**
//!
//! Question: does one semantic cycle lower onto pieces that already exist,
//! namely binding, reading, mask, projection, fold, scalar comparison and
//! revision? Revision stays the explicit semantic terminal. This file adds no
//! primitive, no write path and no learn op.
//!
//! ```text
//! SPOFC   per candidate (s, o): |X| = COUNT(S_s ∧ W), |X∧Y| = COUNT(S_s ∧ O_o ∧ W)
//!         → arm_to_truth_u8 → {s, p, o, f, c} → scalar compare → one claim bit
//! do      the observation itself: the owner writes the NEW rows into the
//!         resident planes before the cycle. The evaluator only reads them.
//! learn   EncounterEvidence → GadamerRevision::revise → verdict → delta.resulting
//! ```
//!
//! Every population read is a Quack `Filter` lowered with `lower_fused` to one
//! mask-risc program. Every number below is checked against a bit-at-a-time
//! reference that never touches mask-risc. The reference also builds the same
//! `EncounterEvidence` and calls the same `GadamerRevision`.
//!
//! The second half asks whether revision itself compiles: its kind decision
//! is a scalar table over seven `Any` folds of at most two masks each, and its
//! resulting horizon is three `Keep` outputs plus one mask passed through.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeSet;

use lance_graph_arm_discovery::{arm_to_truth_u8, CandidateRule, TruthU8, NARS_PERSONALITY_K};
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionDelta, RevisionKind,
    RevisionPolicy,
};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, Lowering, MaskOp, Operand, Out, Planes, Program, Scratch,
    Terminal, Value, TILE_WORDS,
};
use lance_graph_quack::{lower_fused, Agg, Filter, Mask, Query};

// ── thread-local heap meter (the `diff_plateau.rs` pattern) ──────────────

struct Counting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn bytes() -> usize {
    BYTES.with(Cell::get)
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + layout.size()));
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

// ── the resident observation population ─────────────────────────────────

const TILE_ROWS: usize = TILE_WORDS * 64;
const SUBJECTS: usize = 4;
const OBJECTS: usize = 4;

/// Mask plane indices. `S_s` and `O_o` are one-hot over observed rows.
const fn s_plane(s: usize) -> u16 {
    s as u16
}
const fn o_plane(o: usize) -> u16 {
    (SUBJECTS + o) as u16
}
/// Every observed row.
const VALID: u16 = (SUBJECTS + OBJECTS) as u16;
/// Rows observed before this cycle (the prior version's validity).
const OLD: u16 = VALID + 1;
/// Rows the "do" step observed in this cycle. `OLD ∪ NEW = VALID`, disjoint.
const NEW: u16 = VALID + 2;
const PLANES: usize = SUBJECTS + OBJECTS + 3;

/// Claim threshold: frequency ≥ 0.6 and confidence ≥ 0.9 (u8 scale).
const F_MIN: u8 = 153;
const C_MIN: u8 = 230;
/// A new window contradicts a prior claim when its frequency falls below this
/// over at least `MIN_ANTE` new antecedent rows.
const F_CONTRA: u8 = 102;
const MIN_ANTE: u32 = 8;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn set(words: &mut [u64], i: usize) {
    words[i / 64] |= 1 << (i % 64);
}

fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// The world after the "do": old rows say `o = s` (80 %); in the NEW window
/// subject 2 says `o = 3`. So claim (2,2) holds overall, is contradicted by
/// the new window, and claim (2,3) appears.
fn world(n: usize, seed: u64) -> Vec<Vec<u64>> {
    let mut s_ = seed;
    let mut p = vec![vec![0u64; words_for(n)]; PLANES];
    for i in 0..n {
        let r = lcg(&mut s_);
        if r.is_multiple_of(10) {
            continue; // unobserved
        }
        let new = r % 10 >= 7;
        let s = (lcg(&mut s_) % SUBJECTS as u64) as usize;
        let agrees = lcg(&mut s_) % 10 < 8;
        let o = match (new, s, agrees) {
            (true, 2, true) => 3,
            (_, _, true) => s,
            _ => (lcg(&mut s_) % OBJECTS as u64) as usize,
        };
        set(&mut p[s_plane(s) as usize], i);
        set(&mut p[o_plane(o) as usize], i);
        set(&mut p[VALID as usize], i);
        set(&mut p[if new { NEW } else { OLD } as usize], i);
    }
    p
}

// ── one compiled read ────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Meter {
    programs: usize,
    fused: usize,
    /// No ops at all: a terminal straight over one resident plane.
    direct: usize,
    heap: usize,
    written_words: usize,
}

/// Lower `filter` with `lower_fused`, execute it over the resident planes,
/// and meter it. Lowering itself builds a `Program` (a compile step); only
/// the execution is metered.
fn read(planes: &[Vec<u64>], n: usize, filter: Filter, agg: Agg, m: &mut Meter) -> Value {
    let program = lower_fused(&Query { filter, agg }).expect("lowers");
    let masks: Vec<&[u64]> = planes.iter().map(Vec::as_slice).collect();
    let p = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let slots = if program.requires_scratch() {
        program.scratch_slots as usize
    } else {
        0
    };
    let tile = lance_graph_mask_risc::tile_words_for(n);
    let mut buf = vec![0u64; lance_graph_mask_risc::scratch_words_for(tile, slots).expect("fits")];
    let before = bytes();
    let value = {
        let mut scratch = Scratch::over_for_program(&mut buf, &program, n).expect("scratch");
        execute_into(&program, &p, &Foreign::NONE, &mut scratch, Out::None).expect("runs")
    };
    m.heap += bytes() - before;
    m.programs += 1;
    m.fused += usize::from(matches!(
        program.lowering(),
        Lowering::Ternlog(_) | Lowering::Tern2(_) | Lowering::Range(_)
    ));
    m.direct +=
        usize::from(!program.requires_scratch() && matches!(program.lowering(), Lowering::Tiled));
    m.written_words += buf.iter().filter(|&&w| w != 0).count();
    value
}

fn count(planes: &[Vec<u64>], n: usize, f: Filter, m: &mut Meter) -> u32 {
    match read(planes, n, f, Agg::Count, m) {
        Value::Count(c) => c as u32,
        other => panic!("not a count: {other:?}"),
    }
}

fn pl(i: u16) -> Filter {
    Filter::plane(Mask(i))
}

// ── SPOFC: per-candidate evidence → truth → claim bit ────────────────────

/// The integer evidence for one candidate over one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Evidence {
    ante: u32,
    co: u32,
}

fn truth(e: Evidence, window: u32) -> TruthU8 {
    arm_to_truth_u8(
        &CandidateRule {
            antecedent: vec![],
            consequent: vec![],
            cooccur: e.co,
            antecedent_count: e.ante,
            window,
        },
        NARS_PERSONALITY_K,
    )
}

fn holds(t: TruthU8) -> bool {
    t.frequency >= F_MIN && t.confidence >= C_MIN
}

fn contradicts(e: Evidence, t: TruthU8) -> bool {
    e.ante >= MIN_ANTE && t.frequency < F_CONTRA
}

/// How a cycle gets its counts. The compiled source reads through Quack and
/// mask-risc; the reference counts rows one at a time.
trait Source {
    fn evidence(&mut self, s: usize, o: usize, window: u16) -> Evidence;
    fn window(&mut self, window: u16) -> u32;
}

struct Compiled<'a> {
    planes: &'a [Vec<u64>],
    n: usize,
    meter: Meter,
}

impl Source for Compiled<'_> {
    fn evidence(&mut self, s: usize, o: usize, w: u16) -> Evidence {
        let (p, n, m) = (self.planes, self.n, &mut self.meter);
        Evidence {
            ante: count(p, n, Filter::and([pl(s_plane(s)), pl(w)]), m),
            co: count(
                p,
                n,
                Filter::and([pl(s_plane(s)), pl(o_plane(o)), pl(w)]),
                m,
            ),
        }
    }
    fn window(&mut self, w: u16) -> u32 {
        let (p, n, m) = (self.planes, self.n, &mut self.meter);
        count(p, n, pl(w), m)
    }
}

struct Reference<'a> {
    planes: &'a [Vec<u64>],
    n: usize,
}

impl Reference<'_> {
    fn rows(&self, ps: &[u16]) -> u32 {
        (0..self.n)
            .filter(|&i| ps.iter().all(|&p| bit(&self.planes[p as usize], i)))
            .count() as u32
    }
}

impl Source for Reference<'_> {
    fn evidence(&mut self, s: usize, o: usize, w: u16) -> Evidence {
        Evidence {
            ante: self.rows(&[s_plane(s), w]),
            co: self.rows(&[s_plane(s), o_plane(o), w]),
        }
    }
    fn window(&mut self, w: u16) -> u32 {
        self.rows(&[w])
    }
}

/// The claims that hold over window `w`, as one claim bit per candidate.
fn claims(src: &mut impl Source, w: u16) -> u64 {
    let window = src.window(w);
    let mut bits = 0u64;
    for s in 0..SUBJECTS {
        for o in 0..OBJECTS {
            if holds(truth(src.evidence(s, o, w), window)) {
                bits |= 1 << (s * OBJECTS + o);
            }
        }
    }
    bits
}

type Horizon = InterpretiveHorizon<(), u64>;

/// The prior horizon: what the OLD window alone supports.
fn prior(src: &mut impl Source) -> Horizon {
    let held = claims(src, OLD);
    InterpretiveHorizon {
        id: HorizonId(0),
        awareness: (),
        question: QuestionId(6),
        language: LanguageId(0),
        grammar: GrammarId(0),
        codebook: CodebookId(0),
        lens: LensId(0),
        projected_claims: held,
        // OLD rows established these claims: they are the prior's roots.
        independent_roots: held,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

/// The encounter the NEW window produces against `prior`.
fn encounter(src: &mut impl Source, prior: &Horizon) -> EncounterEvidence<u64> {
    let window = src.window(NEW);
    let (mut proposed, mut contradictions) = (0u64, 0u64);
    for s in 0..SUBJECTS {
        for o in 0..OBJECTS {
            let k = s * OBJECTS + o;
            let e = src.evidence(s, o, NEW);
            let t = truth(e, window);
            if holds(t) {
                proposed |= 1 << k;
            }
            if prior.projected_claims >> k & 1 == 1 && contradicts(e, t) {
                contradictions |= 1 << k;
            }
        }
    }
    // NEW rows are disjoint from OLD rows, so every claim the NEW window
    // establishes is rooted in it independently; nothing is inherited.
    // Revision itself decides which of these roots are NEW against ancestry.
    EncounterEvidence {
        proposed_claims: proposed,
        independent_roots: proposed,
        inherited_roots: 0,
        resistance: contradictions,
        contradictions,
        affected_parts: proposed | prior.projected_claims,
    }
}

fn ancestry(h: &Horizon) -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: h.independent_roots,
        ancestry_derived_roots: h.inherited_roots,
        ancestor_claims: h.projected_claims,
        closes_cycle: false,
    }
}

/// One full cycle: SPOFC over OLD → prior; the "do" already wrote NEW;
/// SPOFC over NEW → encounter; learn = revise.
fn cycle(src: &mut impl Source) -> (Horizon, EncounterEvidence<u64>, RevisionDelta<(), u64>) {
    let p = prior(src);
    let e = encounter(src, &p);
    let d = GadamerRevision.revise(&p, &e, &ancestry(&p));
    (p, e, d)
}

const fn claim(s: usize, o: usize) -> u64 {
    1 << (s * OBJECTS + o)
}

// ── tests: the cycle ─────────────────────────────────────────────────────

/// FAILS IF: any compiled read disagrees with the row reference, so the
/// compiled cycle produces a different prior, encounter or revision.
#[test]
fn the_compiled_cycle_equals_the_row_reference() {
    for (n, seed) in [(3 * TILE_ROWS + 77, 1), (TILE_ROWS + 1, 2), (4_001, 3)] {
        let planes = world(n, seed);
        let mut compiled = Compiled {
            planes: &planes,
            n,
            meter: Meter::default(),
        };
        let mut reference = Reference { planes: &planes, n };
        let (cp, ce, cd) = cycle(&mut compiled);
        let (rp, re, rd) = cycle(&mut reference);
        assert_eq!(cp, rp, "n={n}: prior");
        assert_eq!(ce, re, "n={n}: encounter");
        assert_eq!(cd, rd, "n={n}: revision");
    }
}

/// FAILS IF: the fixture stops exercising the cycle: (2,2) must hold
/// before, be contradicted by the new window, and (2,3) must be introduced
/// with a new independent root, so revision reaches `HorizonFusion`.
#[test]
fn the_cycle_reaches_fusion_with_a_preserved_contradiction() {
    let n = 3 * TILE_ROWS + 77;
    let planes = world(n, 1);
    let mut src = Compiled {
        planes: &planes,
        n,
        meter: Meter::default(),
    };
    let (p, e, d) = cycle(&mut src);
    assert_ne!(p.projected_claims & claim(2, 2), 0, "prior holds (2,2)");
    assert_eq!(p.projected_claims & claim(2, 3), 0, "prior lacks (2,3)");
    assert_eq!(
        e.contradictions,
        claim(2, 2),
        "the new window contradicts (2,2) only"
    );
    assert_ne!(e.proposed_claims & claim(2, 3), 0, "(2,3) is proposed");
    assert_ne!(e.independent_roots & claim(2, 3), 0, "(2,3) has a new root");
    assert_eq!(d.kind, RevisionKind::HorizonFusion);
    assert_eq!(d.evidential_effect, EvidentialEffect::IncreaseEligible);
    assert_eq!(
        d.resulting.unresolved_tension,
        claim(2, 2),
        "the contradiction is carried forward, not resolved"
    );
}

/// FAILS IF: an SPOFC read writes a population-sized derived word, falls off
/// the fused lowering, or allocates during execution.
#[test]
fn every_spofc_read_is_one_fused_fold() {
    let n = 3 * TILE_ROWS + 77;
    let planes = world(n, 1);
    let mut src = Compiled {
        planes: &planes,
        n,
        meter: Meter::default(),
    };
    let _ = cycle(&mut src);
    let m = &src.meter;
    // prior and encounter: one window size + 16 candidates × (|X|, |X∧Y|).
    assert_eq!(m.programs, 2 * (1 + 2 * 16));
    assert_eq!(
        m.direct, 2,
        "the three window sizes read one plane directly"
    );
    assert_eq!(
        m.fused + m.direct,
        m.programs,
        "every other read lowered to one fold"
    );
    assert_eq!(m.written_words, 0, "no derived words written");
    assert_eq!(m.heap, 0, "no execution allocation");
}

/// FAILS IF: revision is not causally necessary. Replaying the same
/// encounter from the revised horizon must not mint evidence a second time;
/// replaying it from the prior (revision bypassed) does.
#[test]
fn replay_from_the_revised_horizon_does_not_mint_twice() {
    let n = 3 * TILE_ROWS + 77;
    let planes = world(n, 1);
    let mut src = Compiled {
        planes: &planes,
        n,
        meter: Meter::default(),
    };
    let (p, e, d) = cycle(&mut src);

    let again = GadamerRevision.revise(&d.resulting, &e, &ancestry(&d.resulting));
    assert_eq!(again.new_independent_roots, 0, "no new root on replay");
    assert_ne!(again.evidential_effect, EvidentialEffect::IncreaseEligible);
    assert_eq!(again.kind, RevisionKind::ContradictionPreserved);

    let bypass = GadamerRevision.revise(&p, &e, &ancestry(&p));
    assert_eq!(
        bypass.evidential_effect,
        EvidentialEffect::IncreaseEligible,
        "bypassing revision lets the same observation count again"
    );
}

// ── tests: does revision itself compile? ─────────────────────────────────

/// Masks of one revision call, as resident 64-row planes.
const PRIOR_PROJ: u16 = 0;
const PROPOSED: u16 = 1;
const ENC_ROOTS: u16 = 2;
const RESISTANCE: u16 = 3;
const CONTRA: u16 = 4;
const ANC_ROOTS: u16 = 5;
const ANC_CLAIMS: u16 = 6;
const PRIOR_ROOTS: u16 = 7;
const PRIOR_INH: u16 = 8;
const ENC_INH: u16 = 9;
const PRIOR_TENSION: u16 = 10;

fn masks_of(p: &Horizon, e: &EncounterEvidence<u64>, a: &BasisView<u64>) -> [[u64; 1]; 11] {
    [
        [p.projected_claims],
        [e.proposed_claims],
        [e.independent_roots],
        [e.resistance],
        [e.contradictions],
        [a.ancestry_independent_roots],
        [a.ancestor_claims],
        [p.independent_roots],
        [p.inherited_roots],
        [e.inherited_roots],
        [p.unresolved_tension],
    ]
}

fn run_word(masks: &[[u64; 1]; 11], program: &Program, out: Option<&mut [u64]>) -> Value {
    let ms: Vec<&[u64]> = masks.iter().map(|m| m.as_slice()).collect();
    let p = Planes {
        n_rows: 64,
        masks: &ms,
        lanes: &[],
    };
    let slots = program.scratch_slots as usize;
    let tile = lance_graph_mask_risc::tile_words_for(64);
    let mut buf = vec![0u64; lance_graph_mask_risc::scratch_words_for(tile, slots).expect("fits")];
    let mut scratch = Scratch::over_for_program(&mut buf, program, 64).expect("scratch");
    let out = match out {
        Some(o) => Out::Mask(o),
        None => Out::None,
    };
    execute_into(program, &p, &Foreign::NONE, &mut scratch, out).expect("runs")
}

fn pp(i: u16) -> Operand {
    Operand::Plane(i)
}

/// `Any` of one binary op over two planes, or of a plane itself.
fn any_of(
    masks: &[[u64; 1]; 11],
    op: Option<fn(Operand, Operand, u16) -> MaskOp>,
    a: u16,
    b: u16,
) -> bool {
    let program = match op {
        Some(f) => Program::new(
            vec![f(pp(a), pp(b), 0)],
            Terminal::Any {
                mask: Operand::Scratch(0),
            },
        ),
        None => Program::new(vec![], Terminal::Any { mask: pp(a) }),
    };
    match run_word(masks, &program, None) {
        Value::Bool(x) => x,
        other => panic!("not a bool: {other:?}"),
    }
}

fn and_not(a: Operand, b: Operand, dst: u16) -> MaskOp {
    MaskOp::AndNot { a, b, dst }
}
fn xor(a: Operand, b: Operand, dst: u16) -> MaskOp {
    MaskOp::Xor { a, b, dst }
}

/// The revision kind, derived from seven `Any` folds and one scalar. This is
/// the LAW as a decision table; every mask question is a mask-risc program.
fn compiled_kind(masks: &[[u64; 1]; 11], closes_cycle: bool) -> RevisionKind {
    let has_new_root = any_of(masks, Some(and_not), ENC_ROOTS, ANC_ROOTS);
    let has_resistance = any_of(masks, None, RESISTANCE, 0);
    let has_contradiction = any_of(masks, None, CONTRA, 0);
    let same_projection = !any_of(masks, Some(xor), PRIOR_PROJ, PROPOSED);
    let recycles = !any_of(masks, Some(and_not), PROPOSED, ANC_CLAIMS);
    // introduced ∪ preserved = proposed, so their union's non-emptiness is
    // one fold over the proposed plane.
    let proposed_any = any_of(masks, None, PROPOSED, 0);
    let withdrawn_any = any_of(masks, Some(and_not), PRIOR_PROJ, PROPOSED);
    let revised_any = !same_projection;

    if closes_cycle && !has_new_root && !has_resistance {
        RevisionKind::ClosedCycle
    } else if !has_new_root && !has_resistance && (same_projection || recycles) {
        RevisionKind::Echo
    } else if has_contradiction && proposed_any && has_new_root {
        RevisionKind::HorizonFusion
    } else if has_contradiction {
        RevisionKind::ContradictionPreserved
    } else if has_resistance && withdrawn_any {
        RevisionKind::AssumptionExposed
    } else if has_new_root && same_projection {
        RevisionKind::IndependentConfirmation
    } else if has_new_root {
        RevisionKind::HorizonExpansion
    } else if has_resistance || revised_any {
        RevisionKind::Reinterpretation
    } else {
        RevisionKind::Suspended
    }
}

/// `Keep` of `lhs | rhs`, where `rhs` may be `a & !b`.
fn keep(masks: &[[u64; 1]; 11], ops: Vec<MaskOp>, last: u16) -> (u64, Lowering) {
    let program = Program::new(
        ops,
        Terminal::Keep {
            mask: Operand::Scratch(last),
        },
    );
    let mut out = [0u64; 1];
    run_word(masks, &program, Some(&mut out));
    (out[0], program.lowering())
}

/// Every horizon the 2-bit universe can form: seven 2-bit masks plus
/// `closes_cycle`. Prior roots / inheritance / tension are fixed per case
/// from the same bits so the resulting-horizon check has non-trivial inputs.
fn exhaustive() -> impl Iterator<Item = (Horizon, EncounterEvidence<u64>, BasisView<u64>)> {
    (0u32..1 << 15).map(|x| {
        let f = |i: u32| u64::from(x >> (2 * i) & 3);
        let p = InterpretiveHorizon {
            id: HorizonId(0),
            awareness: (),
            question: QuestionId(6),
            language: LanguageId(0),
            grammar: GrammarId(0),
            codebook: CodebookId(0),
            lens: LensId(0),
            projected_claims: f(0),
            // Disjoint from the ancestry roots (bits 0..2), so `roots \ ancestry`
            // is not absorbed by the prior's own roots.
            independent_roots: f(6) << 4,
            inherited_roots: f(1) << 3,
            unresolved_tension: f(4) << 5,
            revision_index: 0,
        };
        let e = EncounterEvidence {
            proposed_claims: f(1),
            independent_roots: f(2),
            inherited_roots: f(6) << 2,
            resistance: f(3),
            contradictions: f(4),
            affected_parts: f(0) | f(1),
        };
        let a = BasisView {
            ancestry_independent_roots: f(5),
            ancestry_derived_roots: 0,
            ancestor_claims: f(6),
            closes_cycle: x >> 14 & 1 == 1,
        };
        (p, e, a)
    })
}

/// FAILS IF: the decision table over fused folds disagrees with
/// `GadamerRevision::revise` on any case of the 2-bit universe, or the
/// universe stops reaching every reachable kind.
#[test]
fn the_revision_decision_compiles_to_seven_folds() {
    let mut reached = BTreeSet::new();
    for (p, e, a) in exhaustive() {
        let d = GadamerRevision.revise(&p, &e, &a);
        let masks = masks_of(&p, &e, &a);
        assert_eq!(
            compiled_kind(&masks, a.closes_cycle),
            d.kind,
            "{p:?} {e:?} {a:?}"
        );
        reached.insert(format!("{:?}", d.kind));
    }
    let expected: BTreeSet<String> = [
        "IndependentConfirmation",
        "Reinterpretation",
        "HorizonExpansion",
        "HorizonFusion",
        "AssumptionExposed",
        "ContradictionPreserved",
        "Echo",
        "ClosedCycle",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(
        reached, expected,
        "every kind but Suspended is reached; Suspended is unreachable (see the round report)"
    );
}

/// FAILS IF: the resulting horizon is not three fused `Keep` outputs over
/// resident planes plus the proposed claims passed through unchanged.
#[test]
fn the_resulting_horizon_is_three_keeps_and_one_passthrough() {
    for (p, e, a) in exhaustive().step_by(7) {
        let d = GadamerRevision.revise(&p, &e, &a);
        let masks = masks_of(&p, &e, &a);

        let (roots, l1) = keep(
            &masks,
            vec![
                and_not(pp(ENC_ROOTS), pp(ANC_ROOTS), 0),
                MaskOp::Or {
                    a: pp(PRIOR_ROOTS),
                    b: Operand::Scratch(0),
                    dst: 1,
                },
            ],
            1,
        );
        let (inherited, l2) = keep(
            &masks,
            vec![MaskOp::Or {
                a: pp(PRIOR_INH),
                b: pp(ENC_INH),
                dst: 0,
            }],
            0,
        );
        let (tension, l3) = keep(
            &masks,
            vec![MaskOp::Or {
                a: pp(PRIOR_TENSION),
                b: pp(CONTRA),
                dst: 0,
            }],
            0,
        );
        assert_eq!(roots, d.resulting.independent_roots);
        assert_eq!(inherited, d.resulting.inherited_roots);
        assert_eq!(tension, d.resulting.unresolved_tension);
        assert_eq!(
            d.resulting.projected_claims, e.proposed_claims,
            "passthrough"
        );
        for l in [l1, l2, l3] {
            assert!(matches!(l, Lowering::TernlogKeep(_)), "{l:?}");
        }
    }
}
