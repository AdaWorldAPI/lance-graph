//! D-CTX-7: `GadamerRevision` as the evidence writer of EpistemicState5.
//!
//! `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` (D-CE64-TIME-0): no production
//! code writes `CausalEdge64` bits 59..63 from evidence, so evidence never
//! reaches what the next cycle may do. D-CTX-6 showed the shape of an
//! evidence gate: an observation of the resident palette is a new independent
//! root and earns `IncreaseEligible`; the rendered surface has no root and
//! earns nothing. This probe puts that gate in front of the register:
//!
//! ```text
//! cycle k:  start-of-cycle copy of the CE64 row (MailboxSoaView)
//!           encounters -> GadamerRevision -> declared transition -> code
//!           write_row(k)                      (bits 59..63 only)
//! cycle k+1: read through MailboxSoaView -> #1370 eligibility
//! ```
//!
//! # The tracked relation
//!
//! One CE64 row tracks "pixel `P` is a material boundary" on a 16 × 16
//! palette tile. A source is one Moore direction `d` of `P`: it reads the
//! palette at `P` and at its neighbour in direction `d`, nothing else. A
//! source either sees a different material (`Differs`, supports the claim) or
//! the same (`Same`, says nothing on its own). Eight `Same` readings in one
//! cycle are a complete check and refute the claim.
//!
//! The revision horizon of a cycle starts from the register (claim projected
//! iff code `ASSOCIATED`) with no roots, and is dropped at the end of the
//! cycle, like D-CE64-TIME-0's `SupportLedger`.
//!
//! # The declared transition (generation V1; thresholds are policy pins)
//!
//! | from | to | when, in this cycle |
//! |---|---|---|
//! | 0 `Direct × Open` | 4 `Direct × Associated` | ≥ 2 `Differs` encounters the revision admitted (`IncreaseEligible`), each a new root |
//! | 4 | 0 | the revision returned `ContradictionPreserved` on the claim (a complete `Same` check) |
//!
//! # One field, one code
//!
//! Bits 59..63 are written and read as ONE joint EpistemicState5 code
//! (`spare() << 2 | truth_raw()`, the #1370 reading), never as the truth half
//! (59..60) or the band half (61..63) alone: `with_code` sets both through the
//! shipped writers in one step, and every code it can produce is declared in
//! `EPI_LAW`. The two codes used agree with D-GSO-7a's band reading as well:
//! `3 >> 2 = 0` (Open) and `7 >> 2 = 1` (Associated).
//!
//! Promotion needs earned evidence. Demotion reads a preserved contradiction:
//! it lowers a certification and mints no evidence. Nothing else moves the
//! code; a rendered witness (D-CTX-4) presented as an inherited
//! interpretation never does.
//!
//! # What it does NOT decide
//!
//! - That this transition belongs in production. It is one declared,
//!   versioned candidate for the missing writer.
//! - Cross-cycle corroboration: the horizon dies with its cycle, so two
//!   single sources in two cycles never combine (pinned below). Carrying the
//!   source set needs state the register does not have.
//! - F/C: the transition writes the code only; `learn` is not called.
//!
//! Run: `cargo run -p cognitive-shader-driver --example revision_epistemic_writer_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example revision_epistemic_writer_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;

#[path = "support/fisher_relation.rs"]
mod fisher_relation;

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;

#[path = "support/ewa.rs"]
mod ewa;

#[path = "support/boundary.rs"]
mod boundary;

use affordance_law::{measure, raw5, LawGen};
use bgz_tensor::fisher_z::FisherZTable;
use boundary::strongest_boundary_where;
use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use ewa::{render_isotropic, Field};
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ, MOORE};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionKind, RevisionPolicy,
};
use lance_graph_contract::soa_view::MailboxSoaView;
use virtual_surfel::{neighbor, Tile, PIXELS};

/// The row of the mailbox that holds the register.
const ROW: usize = 0;
/// EpistemicState5 codes of the #1370 law used here.
/// `Direct × Open` (was code 3 `Direct × Observed` under the #1370
/// probe-local codebook; observation is evidence, not a coordinate).
const CODE_OPEN: u8 = 0;
/// `Direct × Associated` (was code 7).
const CODE_ASSOCIATED: u8 = 4;
/// The claim "P is a material boundary": bit 0 of the claim mask.
const CLAIM: u64 = 1;

/// The declared transition, versioned. `V1` promotes on 2 earned sources;
/// `V2` (a stricter candidate) on 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransitionGen {
    V1,
    #[cfg_attr(not(test), allow(dead_code))]
    V2,
}

impl TransitionGen {
    fn sources_to_promote(self) -> u32 {
        match self {
            TransitionGen::V1 => 2,
            TransitionGen::V2 => 3,
        }
    }
}

type Horizon = InterpretiveHorizon<(), u64>;

// ── the world ──────────────────────────────────────────────────────────────

/// What one source reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reading {
    Differs,
    Same,
}

/// One encounter presented to the revision in a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encounter {
    /// Source = Moore direction `d` of `P`, read from the resident tile.
    Observe(u8),
    /// A complete check: all eight directions read in this cycle.
    CompleteCheck,
    /// The D-CTX-4 witness over the rendered field, at the given pixel.
    Render(Morton8x8),
}

/// Reads the resident tile only: the material at `p` and at its neighbour in
/// direction `d`. Off-tile neighbours read as `Same`.
fn observe(tile: &Tile, p: Morton8x8, d: u8) -> Reading {
    let (dx, dy) = MOORE[usize::from(d)];
    match neighbor(p, dx, dy) {
        Some(n) if tile[n.code() as usize] != tile[p.code() as usize] => Reading::Differs,
        _ => Reading::Same,
    }
}

/// The cycle's horizon: claim projected iff the register is certified.
fn horizon_from(register: CausalEdge64) -> Horizon {
    InterpretiveHorizon {
        id: HorizonId(0),
        awareness: (),
        question: QuestionId(7),
        language: LanguageId(0),
        grammar: GrammarId(0),
        codebook: CodebookId(0),
        lens: LensId(0),
        projected_claims: if raw5(register) == CODE_ASSOCIATED {
            CLAIM
        } else {
            0
        },
        independent_roots: 0,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
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

/// The evidence an encounter presents, given the horizon so far, and the
/// mask of its roots that support the claim (a `Same` reading never does).
fn evidence(tile: &Tile, p: Morton8x8, h: &Horizon, e: Encounter) -> (EncounterEvidence<u64>, u64) {
    match e {
        Encounter::Observe(d) => {
            let root = 1u64 << d;
            let differs = observe(tile, p, d) == Reading::Differs;
            let proposed_claims = if differs {
                h.projected_claims | CLAIM
            } else {
                h.projected_claims
            };
            (
                EncounterEvidence {
                    proposed_claims,
                    independent_roots: root,
                    inherited_roots: 0,
                    resistance: 0,
                    contradictions: 0,
                    affected_parts: if differs { CLAIM } else { 0 },
                },
                if differs { root } else { 0 },
            )
        }
        Encounter::CompleteCheck => {
            let differing = (0..8u8)
                .filter(|&d| observe(tile, p, d) == Reading::Differs)
                .fold(0u64, |m, d| m | 1 << d);
            let all_same = differing == 0;
            let ev = if all_same {
                EncounterEvidence {
                    proposed_claims: h.projected_claims & !CLAIM,
                    independent_roots: 0xFF,
                    inherited_roots: 0,
                    resistance: CLAIM,
                    contradictions: CLAIM,
                    affected_parts: CLAIM,
                }
            } else {
                EncounterEvidence {
                    proposed_claims: h.projected_claims | CLAIM,
                    independent_roots: 0xFF,
                    inherited_roots: 0,
                    resistance: 0,
                    contradictions: 0,
                    affected_parts: CLAIM,
                }
            };
            (ev, differing)
        }
        // The rendered surface proposes the claim and would count every root
        // it carried; it carries none, so only the revision gate keeps it out
        // of `earned_support`.
        Encounter::Render(_) => (
            EncounterEvidence {
                proposed_claims: h.projected_claims | CLAIM,
                independent_roots: 0,
                inherited_roots: CLAIM,
                resistance: 0,
                contradictions: 0,
                affected_parts: CLAIM,
            },
            u64::MAX,
        ),
    }
}

/// One cycle's revision: present every encounter, adopt `delta.resulting`
/// only on `IncreaseEligible`, and settle the code by the declared
/// transition. Takes the start-of-cycle register; returns the next one.
fn settle(
    gen: TransitionGen,
    tile: &Tile,
    p: Morton8x8,
    register: CausalEdge64,
    encounters: &[Encounter],
) -> CausalEdge64 {
    let mut h = horizon_from(register);
    let mut earned_support = 0u32;
    let mut contradicted = false;
    for &e in encounters {
        let (ev, supporting) = evidence(tile, p, &h, e);
        let delta = GadamerRevision.revise(&h, &ev, &ancestry(&h));
        if delta.evidential_effect == EvidentialEffect::IncreaseEligible {
            earned_support += (delta.new_independent_roots & supporting).count_ones();
            h = delta.resulting;
        } else if delta.kind == RevisionKind::ContradictionPreserved
            && delta.contradictions & CLAIM != 0
        {
            contradicted = true;
        }
    }
    let code = match raw5(register) {
        CODE_OPEN if earned_support >= gen.sources_to_promote() => CODE_ASSOCIATED,
        CODE_ASSOCIATED if contradicted => CODE_OPEN,
        other => other,
    };
    with_code(register, code)
    // `h` is dropped here: the horizon does not survive the cycle.
}

// ── the register ───────────────────────────────────────────────────────────

/// Write a declared V1 EpistemicState5 code (bits 59..63, jointly). The
/// transition only ever lands on declared codes, so `write_code` never fires
/// its expect.
fn with_code(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    affordance_law::write_code(edge, code)
}

/// The register before any evidence: every field set, code `Direct × Open`.
fn initial() -> CausalEdge64 {
    with_code(
        CausalEdge64::pack_v2(
            1,
            2,
            3,
            177,
            91,
            CausalMask::SO,
            0,
            PlasticityState::ALL_HOT,
        ),
        CODE_OPEN,
    )
}

fn eligible(edge: CausalEdge64) -> u64 {
    measure(LawGen::V1, edge, EdgeProvenance::V2Stamped).expect("declared code")
}

fn read(view: &dyn MailboxSoaView) -> CausalEdge64 {
    assert!(ROW < view.n_rows(), "the register row is declared");
    CausalEdge64(view.edges_raw()[ROW])
}

fn mailbox(seed: CausalEdge64) -> MailboxSoA<4> {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(1, 0, 0.5);
    mb.set_populated(ROW + 1);
    write(&mut mb, seed);
    mb.tick();
    mb
}

fn write(mb: &mut MailboxSoA<4>, edge: CausalEdge64) {
    let cell = WriteCell {
        edge: Some(edge),
        ..WriteCell::default()
    };
    assert_eq!(mb.write_row(ROW, mb.cycle(), &cell), WriteOutcome::Accepted);
}

/// One cycle's encounters on one tile.
struct Cycle<'a> {
    tile: &'a Tile,
    encounters: &'a [Encounter],
}

/// Runs cycles through the mailbox row. Returns the register and the
/// eligibility each cycle read at its start, plus the final read.
fn run(gen: TransitionGen, p: Morton8x8, cycles: &[Cycle<'_>]) -> (Vec<CausalEdge64>, Vec<u64>) {
    let mut mb = mailbox(initial());
    let (mut regs, mut elig) = (Vec::new(), Vec::new());
    for c in cycles {
        let at_start = read(&mb);
        regs.push(at_start);
        elig.push(eligible(at_start));
        write(&mut mb, settle(gen, c.tile, p, at_start, c.encounters));
        mb.tick();
    }
    let last = read(&mb);
    regs.push(last);
    elig.push(eligible(last));
    (regs, elig)
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// Material `left` for `x < 8`, `right` for the rest.
fn split(left: u8, right: u8) -> Tile {
    core::array::from_fn(|i| {
        if Morton8x8::from_code(i as u16).x() < 8 {
            left
        } else {
            right
        }
    })
}

/// The tracked pixel: on the boundary of `split`, three east directions differ.
fn tracked() -> Morton8x8 {
    Morton8x8::from_xy(7, 7)
}

/// Direction ordinals into `MOORE` (NW N NE W E SW S SE).
const NE: u8 = 2;
const E: u8 = 4;
#[cfg(test)]
const SE: u8 = 7;
#[cfg(test)]
const N: u8 = 1;

/// Every inner pixel's rendered witness, as encounters.
fn render_encounters(tile: &Tile, law: &PairwiseFisherZ<'_>) -> Vec<Encounter> {
    let mut f: Field = [0.0; PIXELS];
    render_isotropic(tile, &[255; PIXELS], law, &mut f);
    (0..PIXELS as u16)
        .filter_map(|c| {
            let p = Morton8x8::from_code(c);
            strongest_boundary_where(&f, |q| q == p).map(|w| Encounter::Render(w.at))
        })
        .collect()
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (a, b) = law.pair_with_code(-100..=-80);
    let (tile, flat) = (split(a, b), [a; PIXELS]);
    let renders = render_encounters(&tile, &law);
    let p = tracked();
    let cycles = [
        Cycle {
            tile: &tile,
            encounters: &renders,
        },
        Cycle {
            tile: &tile,
            encounters: &[Encounter::Observe(E)],
        },
        Cycle {
            tile: &tile,
            encounters: &[Encounter::Observe(E), Encounter::Observe(NE)],
        },
        Cycle {
            tile: &flat,
            encounters: &[Encounter::CompleteCheck],
        },
    ];
    let labels = [
        "render witnesses only",
        "one Differs source",
        "two Differs sources",
        "tile rewritten, complete Same check",
    ];
    let (regs, elig) = run(TransitionGen::V1, p, &cycles);
    println!("D-CTX-7: GadamerRevision as the evidence writer of EpistemicState5");
    println!(
        "  tracked pixel ({}, {}), {} rendered witnesses in cycle 0",
        p.x(),
        p.y(),
        renders.len()
    );
    for k in 0..regs.len() {
        let what = labels.get(k).copied().unwrap_or("(final read)");
        println!(
            "  cycle {k}: code {:>2}  eligible {:#04x}   encounters: {what}",
            raw5(regs[k]),
            elig[k]
        );
    }
    let mb = mailbox(initial());
    let at = read(&mb);
    let (_, n, bytes) =
        allocations_during(|| settle(TransitionGen::V1, &tile, p, at, cycles[2].encounters));
    println!("  one settle    : {n} allocations, {bytes} B");
}

#[cfg(test)]
mod tests {
    use super::affordance_law::{OBSERVE_FOLD, STRATIFY};
    use super::*;
    use causal_edge::layout::EPISTEMIC_MASK;

    const OBS: u64 = 1 << OBSERVE_FOLD;
    const STRAT: u64 = 1 << STRATIFY;

    fn fixture() -> (FisherZTable, u8, u8) {
        let table = FisherZTable::build(&representatives(1), 256);
        let (a, b) = PairwiseFisherZ::borrow(&table).pair_with_code(-100..=-80);
        (table, a, b)
    }

    /// FAILS IF: earned evidence does not reach the next cycle, or does so in
    /// the cycle that earned it. The code each cycle reads at its start:
    /// render only, one source, two sources (promotes, read by cycle 3), a
    /// complete Same check after the tile changed (demotes, read by cycle 4).
    #[test]
    fn earned_evidence_is_read_by_the_next_cycle() {
        let (table, a, b) = fixture();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, flat) = (split(a, b), [a; PIXELS]);
        let renders = render_encounters(&tile, &law);
        let (regs, elig) = run(
            TransitionGen::V1,
            tracked(),
            &[
                Cycle {
                    tile: &tile,
                    encounters: &renders,
                },
                Cycle {
                    tile: &tile,
                    encounters: &[Encounter::Observe(E)],
                },
                Cycle {
                    tile: &tile,
                    encounters: &[Encounter::Observe(E), Encounter::Observe(NE)],
                },
                Cycle {
                    tile: &flat,
                    encounters: &[Encounter::CompleteCheck],
                },
            ],
        );
        let codes: Vec<u8> = regs.iter().map(|r| raw5(*r)).collect();
        assert_eq!(codes, [0, 0, 0, 4, 0]);
        assert_eq!(elig, [OBS, OBS, OBS, OBS | STRAT, OBS]);
    }

    /// FAILS IF: the rendered surface moves the code. Every inner pixel's
    /// witness, repeated ten times, is presented while the code is `Direct × Open`
    /// and again while it is ASSOCIATED: neither promotes nor demotes.
    #[test]
    fn rendered_witnesses_never_move_the_code() {
        let (table, a, b) = fixture();
        let law = PairwiseFisherZ::borrow(&table);
        let tile = split(a, b);
        let once = render_encounters(&tile, &law);
        assert_eq!(once.len(), 36);
        let many: Vec<Encounter> = once.iter().cycle().take(360).copied().collect();
        for start in [CODE_OPEN, CODE_ASSOCIATED] {
            let reg = with_code(initial(), start);
            let next = settle(TransitionGen::V1, &tile, tracked(), reg, &many);
            assert_eq!(raw5(next), start);
        }
    }

    /// FAILS IF: support is counted per encounter instead of per earned root.
    /// A complete check of a boundary pixel is one encounter with eight new
    /// roots (three of them `Differs`): it promotes on its own.
    #[test]
    fn a_complete_check_of_a_boundary_pixel_promotes() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let next = settle(
            TransitionGen::V1,
            &tile,
            tracked(),
            initial(),
            &[Encounter::CompleteCheck],
        );
        assert_eq!(raw5(next), CODE_ASSOCIATED);
    }

    /// FAILS IF: a complete check counts its `Same` readings as support. A
    /// pixel with exactly one differing neighbour admits eight new roots and
    /// one supporting root: no promotion under V1. With two differing
    /// neighbours it promotes under V1 and not under V2.
    #[test]
    fn a_complete_check_counts_only_differing_roots() {
        let (_, a, b) = fixture();
        let p = tracked();
        let island = |cells: &[(u8, u8)]| -> Tile {
            let mut t = [a; PIXELS];
            for &(x, y) in cells {
                t[Morton8x8::from_xy(x, y).code() as usize] = b;
            }
            t
        };
        let check = [Encounter::CompleteCheck];
        let one = island(&[(8, 8)]);
        assert_eq!(
            (0..8)
                .filter(|&d| observe(&one, p, d) == Reading::Differs)
                .count(),
            1
        );
        assert_eq!(
            raw5(settle(TransitionGen::V1, &one, p, initial(), &check)),
            CODE_OPEN
        );
        let two = island(&[(8, 8), (8, 7)]);
        assert_eq!(
            (0..8)
                .filter(|&d| observe(&two, p, d) == Reading::Differs)
                .count(),
            2
        );
        assert_eq!(
            raw5(settle(TransitionGen::V1, &two, p, initial(), &check)),
            CODE_ASSOCIATED
        );
        assert_eq!(
            raw5(settle(TransitionGen::V2, &two, p, initial(), &check)),
            CODE_OPEN
        );
    }

    /// FAILS IF: one source counts more than once. The same direction fifty
    /// times in one cycle is one root and never promotes; two different
    /// directions do.
    #[test]
    fn one_source_repeated_never_certifies() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let same = [Encounter::Observe(E); 50];
        let next = settle(TransitionGen::V1, &tile, tracked(), initial(), &same);
        assert_eq!(raw5(next), CODE_OPEN);
        let two = [Encounter::Observe(E), Encounter::Observe(SE)];
        let next = settle(TransitionGen::V1, &tile, tracked(), initial(), &two);
        assert_eq!(raw5(next), CODE_ASSOCIATED);
    }

    /// FAILS IF: a source that sees the same material counts as support. N of
    /// (7, 7) on the split tile is the same material: N + E is one supporting
    /// source and does not promote.
    #[test]
    fn a_same_reading_is_not_support() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        assert_eq!(observe(&tile, tracked(), N), Reading::Same);
        let enc = [Encounter::Observe(N), Encounter::Observe(E)];
        let next = settle(TransitionGen::V1, &tile, tracked(), initial(), &enc);
        assert_eq!(raw5(next), CODE_OPEN);
    }

    /// Pins the open half of the issue: the horizon dies with its cycle, so
    /// one source in cycle k and another in cycle k+1 never combine.
    #[test]
    fn sources_split_across_cycles_do_not_combine() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let (regs, _) = run(
            TransitionGen::V1,
            tracked(),
            &[
                Cycle {
                    tile: &tile,
                    encounters: &[Encounter::Observe(E)],
                },
                Cycle {
                    tile: &tile,
                    encounters: &[Encounter::Observe(NE)],
                },
            ],
        );
        assert!(regs.iter().all(|r| raw5(*r) == CODE_OPEN));
    }

    /// FAILS IF: the transition is not a declared, versioned law. The same
    /// two sources promote under V1 and not under V2.
    #[test]
    fn the_transition_generation_decides() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let two = [Encounter::Observe(E), Encounter::Observe(NE)];
        let v1 = settle(TransitionGen::V1, &tile, tracked(), initial(), &two);
        let v2 = settle(TransitionGen::V2, &tile, tracked(), initial(), &two);
        assert_eq!((raw5(v1), raw5(v2)), (CODE_ASSOCIATED, CODE_OPEN));
        let three = [
            Encounter::Observe(E),
            Encounter::Observe(NE),
            Encounter::Observe(SE),
        ];
        assert_eq!(
            raw5(settle(
                TransitionGen::V2,
                &tile,
                tracked(),
                initial(),
                &three
            )),
            CODE_ASSOCIATED
        );
    }

    /// FAILS IF: the writer touches anything but bits 59..63, or a partial
    /// check demotes. Promotion and demotion change only the code field; a
    /// complete check on a tile that still has the boundary keeps it.
    #[test]
    fn only_the_code_field_moves() {
        let (_, a, b) = fixture();
        let (tile, flat) = (split(a, b), [a; PIXELS]);
        let up = settle(
            TransitionGen::V1,
            &tile,
            tracked(),
            initial(),
            &[Encounter::Observe(E), Encounter::Observe(NE)],
        );
        assert_eq!((up.0 ^ initial().0) & !EPISTEMIC_MASK, 0);
        assert_ne!(up, initial());
        let kept = settle(
            TransitionGen::V1,
            &tile,
            tracked(),
            up,
            &[Encounter::CompleteCheck],
        );
        assert_eq!(kept, up);
        let down = settle(
            TransitionGen::V1,
            &flat,
            tracked(),
            up,
            &[Encounter::CompleteCheck],
        );
        assert_eq!(down, initial());
    }

    /// FAILS IF: cycle k's own eligibility reads its write. The mailbox does
    /// not double-buffer (a plain read after the write sees it), so cycle k
    /// measures from its start-of-cycle copy; the next cycle reads the write.
    #[test]
    fn cycle_k_measures_its_start_copy() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let mut mb = mailbox(initial());
        let at_start = read(&mb);
        let next = settle(
            TransitionGen::V1,
            &tile,
            tracked(),
            at_start,
            &[Encounter::Observe(E), Encounter::Observe(NE)],
        );
        write(&mut mb, next);
        assert_eq!(raw5(read(&mb)), CODE_ASSOCIATED, "no double buffer");
        assert_eq!(eligible(at_start), OBS);
        mb.tick();
        assert_eq!(eligible(read(&mb)), OBS | STRAT);
        // Restart from the persisted word continues identically.
        let restored = CausalEdge64::from_le_bytes(read(&mb).to_le_bytes());
        assert_eq!(restored, next);
    }

    /// FAILS IF: the writer can leave bits 59..63 on an undeclared joint code.
    /// Every start code the law declares, under every encounter set used
    /// here, settles to a declared code, and the two halves are always
    /// written together.
    #[test]
    fn every_write_is_a_declared_joint_code() {
        let (_, a, b) = fixture();
        let (tile, flat) = (split(a, b), [a; PIXELS]);
        let sets: [&[Encounter]; 4] = [
            &[],
            &[Encounter::Observe(E), Encounter::Observe(NE)],
            &[Encounter::CompleteCheck],
            &[Encounter::Render(tracked())],
        ];
        let meaningful = lance_graph_contract::epistemic_state5::facts_population(0);
        assert_eq!(meaningful.count_ones(), 24);
        for start in (0..32u8).filter(|c| meaningful >> c & 1 == 1) {
            for t in [&tile, &flat] {
                for enc in sets {
                    let next = settle(
                        TransitionGen::V1,
                        t,
                        tracked(),
                        with_code(initial(), start),
                        enc,
                    );
                    assert!(
                        meaningful >> raw5(next) & 1 == 1,
                        "{start} -> {}",
                        raw5(next)
                    );
                    assert_eq!(next.spare(), raw5(next) >> 2);
                    assert_eq!(next.truth_raw(), raw5(next) & 0b11);
                }
            }
        }
    }

    /// FAILS IF: one cycle's settle allocates.
    #[test]
    fn a_settle_allocates_nothing() {
        let (_, a, b) = fixture();
        let tile = split(a, b);
        let enc = [
            Encounter::Observe(E),
            Encounter::Observe(NE),
            Encounter::CompleteCheck,
        ];
        let (r, n, bytes) =
            allocations_during(|| settle(TransitionGen::V1, &tile, tracked(), initial(), &enc));
        assert_eq!(raw5(r), CODE_ASSOCIATED);
        assert_eq!((n, bytes), (0, 0));
    }
}
