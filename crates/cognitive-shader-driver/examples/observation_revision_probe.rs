//! D-CTX-6: render → measurement → observation → `GadamerRevision` → replay.
//!
//! Claim under test: the rendered surface is interpretation, not evidence. A
//! measurement over it may choose WHERE to look; only an observation of the
//! resident palette, admitted by the revision contract, may change what is
//! believed. This closes the loop to #1344 on the D-CTX surface.
//!
//! # The belief state is the revision horizon
//!
//! `InterpretiveHorizon<(), [u64; 4]>`, one bit per pixel (lane = Morton code):
//!
//! ```text
//! independent_roots bit p   pixel p has been observed
//! projected_claims  bit p   pixel p was observed to be a material boundary
//! ```
//!
//! A pixel is a material boundary when a Moore neighbour holds a different
//! palette byte. That is a fact about the resident tile; nothing rendered
//! enters it.
//!
//! # Two kinds of encounter
//!
//! - **from the render:** the D-CTX-4 witness proposes "boundary at p". It has
//!   no independent root (the field is derived from the tile already resident)
//!   and is presented as an inherited interpretation. `GadamerRevision` returns
//!   `NoIncrease`, and the probe adopts `delta.resulting` only on
//!   `IncreaseEligible`, so belief does not move.
//! - **from an observation:** `observe(tile, p)` reads the palette at p and its
//!   Moore neighbours. It is a new independent root; the revision returns
//!   `IncreaseEligible` and the result is adopted.
//!
//! # The loop
//!
//! Render once (transient, 2 KiB on the stack), then repeat: measure the
//! steepest unobserved inner pixel, observe it, revise, adopt. The horizon
//! enters the measurement only as an address predicate ("not yet observed").
//! The loop ends when every inner pixel is observed.
//!
//! The counterfactual docket (`RevisionVerdict::is_acceptable`) is not walked:
//! this updates the probe's working horizon, not actual-world state.
//!
//! Run: `cargo run -p cognitive-shader-driver --example observation_revision_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example observation_revision_probe`

use bgz_tensor::fisher_z::FisherZTable;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidenceMask, EvidentialEffect, GadamerRevision,
    GrammarId, HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionDelta,
    RevisionPolicy,
};

#[path = "support/fisher_relation.rs"]
mod fisher_relation;
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ, MOORE};

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;
use virtual_surfel::{neighbor, Tile, PIXELS};

#[path = "support/ewa.rs"]
mod ewa;
use ewa::{render_isotropic, Field};

#[path = "support/boundary.rs"]
mod boundary;
use boundary::{strongest_boundary_where, BoundaryWitness, INNER};

type Mask = [u64; 4];
type Horizon = InterpretiveHorizon<(), Mask>;
type Delta = RevisionDelta<(), Mask>;

// ── masks ──────────────────────────────────────────────────────────────────

fn bit(p: Morton8x8) -> Mask {
    let c = p.code() as usize;
    let mut m = [0u64; 4];
    m[c >> 6] = 1 << (c & 63);
    m
}

fn has(m: &Mask, p: Morton8x8) -> bool {
    m.intersects(&bit(p))
}

fn count(m: &Mask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

fn inner_mask() -> Mask {
    let mut m = Mask::empty();
    for y in INNER {
        for x in INNER {
            m = m.union(&bit(Morton8x8::from_xy(x, y)));
        }
    }
    m
}

// ── the world ──────────────────────────────────────────────────────────────

/// What looking at the resident tile at one pixel returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Observation {
    at: Morton8x8,
    boundary: bool,
}

/// Reads the resident palette only: the material at `at` and its Moore
/// neighbours. Takes no field, no law, no witness.
fn observe(tile: &Tile, at: Morton8x8) -> Observation {
    let here = tile[at.code() as usize];
    let boundary = MOORE
        .iter()
        .any(|&(dx, dy)| neighbor(at, dx, dy).is_some_and(|n| tile[n.code() as usize] != here));
    Observation { at, boundary }
}

fn render(tile: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
    let mut f = [0.0; PIXELS];
    render_isotropic(tile, &[255; PIXELS], law, &mut f);
    f
}

fn prior() -> Horizon {
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: (),
        question: QuestionId(6),
        language: LanguageId(0),
        grammar: GrammarId(0),
        codebook: CodebookId(0),
        lens: LensId(0),
        projected_claims: Mask::empty(),
        independent_roots: Mask::empty(),
        inherited_roots: Mask::empty(),
        unresolved_tension: Mask::empty(),
        revision_index: 0,
    }
}

fn ancestry(h: &Horizon) -> BasisView<Mask> {
    BasisView {
        ancestry_independent_roots: h.independent_roots,
        ancestry_derived_roots: h.inherited_roots,
        ancestor_claims: h.projected_claims,
        closes_cycle: false,
    }
}

// ── encounters ─────────────────────────────────────────────────────────────

/// The rendered witness as an encounter: a proposed claim with no root of its
/// own, carried as an inherited interpretation.
fn encounter_from_render(h: &Horizon, w: &BoundaryWitness) -> EncounterEvidence<Mask> {
    let b = bit(w.at);
    EncounterEvidence {
        proposed_claims: h.projected_claims.union(&b),
        independent_roots: Mask::empty(),
        inherited_roots: b,
        resistance: Mask::empty(),
        contradictions: Mask::empty(),
        affected_parts: b,
    }
}

/// An observation as an encounter: one new independent root, and the claim
/// the tile supports.
fn encounter_from_observation(h: &Horizon, o: &Observation) -> EncounterEvidence<Mask> {
    let b = bit(o.at);
    let proposed_claims = if o.boundary {
        h.projected_claims.union(&b)
    } else {
        h.projected_claims.difference(&b)
    };
    EncounterEvidence {
        proposed_claims,
        independent_roots: b,
        inherited_roots: Mask::empty(),
        resistance: Mask::empty(),
        contradictions: Mask::empty(),
        affected_parts: b,
    }
}

/// The only write: revise, and adopt `delta.resulting` only when the revision
/// says the encounter earned it.
fn admit(h: &Horizon, e: &EncounterEvidence<Mask>) -> (Horizon, Delta) {
    let delta = GadamerRevision.revise(h, e, &ancestry(h));
    let next = if delta.evidential_effect == EvidentialEffect::IncreaseEligible {
        delta.resulting.clone()
    } else {
        h.clone()
    };
    (next, delta)
}

// ── the loop ───────────────────────────────────────────────────────────────

/// One step: measure the steepest unobserved inner pixel, observe it, admit.
/// `None` when every inner pixel has been observed.
fn step(tile: &Tile, field: &Field, h: &Horizon) -> Option<(Horizon, Observation, Delta)> {
    let w = strongest_boundary_where(field, |p| !has(&h.independent_roots, p))?;
    let o = observe(tile, w.at);
    let (next, delta) = admit(h, &encounter_from_observation(h, &o));
    Some((next, o, delta))
}

/// Runs the loop to completion. Returns the final horizon, the number of
/// steps, and the step at which the last true inner boundary was observed.
fn episode(tile: &Tile, law: &PairwiseFisherZ<'_>) -> (Horizon, usize, usize) {
    let field = render(tile, law);
    let truth = boundary_oracle(tile).intersection(&inner_mask());
    let (mut h, mut steps, mut covered_at) = (prior(), 0, 0);
    while let Some((next, _, _)) = step(tile, &field, &h) {
        h = next;
        steps += 1;
        if covered_at == 0 && truth.is_subset_of(&h.projected_claims) {
            covered_at = steps;
        }
    }
    (h, steps, covered_at)
}

/// The visit a row-major scan of the inner region needs to observe every true
/// boundary (the comparison for the render-guided order).
fn row_major_coverage(tile: &Tile) -> usize {
    let truth = boundary_oracle(tile).intersection(&inner_mask());
    let mut seen = Mask::empty();
    let mut steps = 0;
    for y in INNER {
        for x in INNER {
            steps += 1;
            let p = Morton8x8::from_xy(x, y);
            if observe(tile, p).boundary {
                seen = seen.union(&bit(p));
            }
            if truth.is_subset_of(&seen) {
                return steps;
            }
        }
    }
    steps
}

// ── oracle and fixtures ────────────────────────────────────────────────────

/// Boundary pixels from x/y arithmetic, without `neighbor` or `MOORE`.
fn boundary_oracle(tile: &Tile) -> Mask {
    let v = |x: i32, y: i32| tile[Morton8x8::from_xy(x as u8, y as u8).code() as usize];
    let mut m = Mask::empty();
    for y in 0..16i32 {
        for x in 0..16i32 {
            let mut differs = false;
            for ny in (y - 1).max(0)..=(y + 1).min(15) {
                for nx in (x - 1).max(0)..=(x + 1).min(15) {
                    differs |= v(nx, ny) != v(x, y);
                }
            }
            if differs {
                m = m.union(&bit(Morton8x8::from_xy(x as u8, y as u8)));
            }
        }
    }
    m
}

/// Material `left` for `x < at`, `right` for the rest.
fn split_at(left: u8, right: u8, at: u8) -> Tile {
    core::array::from_fn(|i| {
        if Morton8x8::from_code(i as u16).x() < at {
            left
        } else {
            right
        }
    })
}

/// Four quadrants meeting at (8, 8).
fn quadrants(m: [u8; 4]) -> Tile {
    core::array::from_fn(|i| {
        let p = Morton8x8::from_code(i as u16);
        m[usize::from(p.x() >= 8) + 2 * usize::from(p.y() >= 8)]
    })
}

fn fixtures(law: &PairwiseFisherZ<'_>) -> Vec<(&'static str, Tile)> {
    let (a, b) = law.pair_with_code(-100..=-80);
    let (c, d) = law.pair_with_code(80..=100);
    vec![
        ("split at x = 8", split_at(a, b, 8)),
        ("split at x = 2 (outside inner)", split_at(a, b, 2)),
        ("quadrants", quadrants([a, b, c, d])),
        ("uniform", [a; PIXELS]),
        ("repeated", virtual_surfel::repeated_tile()),
    ]
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (a, b) = law.pair_with_code(-100..=-80);

    println!("D-CTX-6 render -> measurement -> observation -> GadamerRevision -> replay");
    println!("  law generation : {:#018x}", law.generation);

    // The rendered witness alone.
    let tile = split_at(a, b, 8);
    let field = render(&tile, &law);
    let w = strongest_boundary_where(&field, |_| true).expect("inner region");
    let (after, delta) = admit(&prior(), &encounter_from_render(&prior(), &w));
    println!(
        "  render witness at ({}, {}) |g| {:.3} -> {:?} / {:?}; belief moved: {}",
        w.at.x(),
        w.at.y(),
        w.magnitude,
        delta.kind,
        delta.evidential_effect,
        after != prior()
    );

    // An observation at the same pixel.
    let o = observe(&tile, w.at);
    let (after, delta) = admit(&prior(), &encounter_from_observation(&prior(), &o));
    println!(
        "  observation   at ({}, {}) boundary {} -> {:?} / {:?}; belief moved: {}",
        o.at.x(),
        o.at.y(),
        o.boundary,
        delta.kind,
        delta.evidential_effect,
        after != prior()
    );

    // The render pointing where nothing is.
    let leak = split_at(a, b, 2);
    let lw = strongest_boundary_where(&render(&leak, &law), |_| true).expect("inner region");
    let lo = observe(&leak, lw.at);
    println!(
        "  boundary at x = 2: render points at ({}, {}) |g| {:.3}; observation says boundary {}",
        lw.at.x(),
        lw.at.y(),
        lw.magnitude,
        lo.boundary
    );

    println!("  full loop, steps until every true inner boundary is observed:");
    for (name, tile) in fixtures(&law) {
        let truth = count(&boundary_oracle(&tile).intersection(&inner_mask()));
        let (h, steps, covered) = episode(&tile, &law);
        println!(
            "    {name:<32} true {truth:>2}  render-guided {covered:>2}  row-major {:>2}  steps {steps}  claims {}",
            row_major_coverage(&tile),
            count(&h.projected_claims)
        );
    }

    let h = prior();
    let (_, n, bytes) = allocations_during(|| step(&tile, &field, &h));
    println!("  one loop step  : {n} allocations, {bytes} B");
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::revision::RevisionKind;

    /// Inner pixels: the population the measurement can address (6 × 6).
    const INNER_PIXELS: usize = 36;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    /// FAILS IF: a rendered witness, presented to the revision, changes the
    /// belief state. Every inner pixel's witness is presented in turn to a
    /// horizon that already holds observations; none is adopted.
    #[test]
    fn rendered_field_alone_cannot_change_belief() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        for (name, tile) in fixtures(&law) {
            let field = render(&tile, &law);
            // A horizon with some genuine observations in it, so the test is
            // not only about the empty prior.
            let mut h = prior();
            for _ in 0..5 {
                h = step(&tile, &field, &h).expect("inner pixels remain").0;
            }
            let before = h.clone();
            for code in 0..PIXELS as u16 {
                let p = Morton8x8::from_code(code);
                let Some(w) = strongest_boundary_where(&field, |q| q == p) else {
                    continue;
                };
                let (next, delta) = admit(&h, &encounter_from_render(&h, &w));
                assert_ne!(
                    delta.evidential_effect,
                    EvidentialEffect::IncreaseEligible,
                    "{name}: render at {code} earned evidence"
                );
                h = next;
            }
            assert_eq!(h, before, "{name}");
        }
    }

    /// FAILS IF: an observation is not admitted, or the claim it writes is not
    /// the tile's. Every inner pixel is observed directly; the adopted claims
    /// equal the x/y oracle.
    #[test]
    fn an_observation_is_admitted_and_writes_the_tiles_fact() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        for (name, tile) in fixtures(&law) {
            let mut h = prior();
            for y in INNER {
                for x in INNER {
                    let o = observe(&tile, Morton8x8::from_xy(x, y));
                    let (next, delta) = admit(&h, &encounter_from_observation(&h, &o));
                    assert_eq!(
                        delta.evidential_effect,
                        EvidentialEffect::IncreaseEligible,
                        "{name}"
                    );
                    let expected = if o.boundary {
                        RevisionKind::HorizonExpansion
                    } else {
                        RevisionKind::IndependentConfirmation
                    };
                    assert_eq!(delta.kind, expected, "{name}");
                    h = next;
                }
            }
            let inner = inner_mask();
            assert_eq!(h.independent_roots, inner, "{name}");
            assert_eq!(
                h.projected_claims,
                boundary_oracle(&tile).intersection(&inner),
                "{name}"
            );
        }
    }

    /// FAILS IF: the render's suggestion can stand in for the fact. With the
    /// material change at x = 2, outside the inner region, the footprint still
    /// leaks a gradient into it: the render points at an inner pixel. The tile
    /// has no boundary there, and the belief records an observed negative.
    #[test]
    fn the_render_can_point_where_nothing_is_and_the_observation_wins() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (a, b) = law.pair_with_code(-100..=-80);
        let tile = split_at(a, b, 2);
        let w = strongest_boundary_where(&render(&tile, &law), |_| true).unwrap();
        assert!(w.magnitude > 1.0, "the render must suggest something");
        let o = observe(&tile, w.at);
        assert!(!o.boundary, "fixture: no material change at the witness");
        assert!(!has(&boundary_oracle(&tile), w.at));

        let (h, delta) = admit(&prior(), &encounter_from_observation(&prior(), &o));
        assert_eq!(delta.kind, RevisionKind::IndependentConfirmation);
        assert!(has(&h.independent_roots, w.at));
        assert!(!has(&h.projected_claims, w.at));
        // And the full loop over this tile believes in no inner boundary.
        let (h, _, _) = episode(&tile, &law);
        assert_eq!(h.projected_claims, Mask::empty());
        assert_eq!(h.independent_roots, inner_mask());
    }

    /// FAILS IF: what is believed depends on the rendering law rather than on
    /// the tile. Two different Fisher-Z laws render different fields and may
    /// visit in a different order; the final belief must be identical.
    #[test]
    fn belief_depends_on_the_tile_not_on_the_law() {
        let (t1, t2) = (table(), FisherZTable::build(&representatives(2), 256));
        let (l1, l2) = (PairwiseFisherZ::borrow(&t1), PairwiseFisherZ::borrow(&t2));
        assert_ne!(l1.generation, l2.generation);
        let mut fields_differ = false;
        for (name, tile) in fixtures(&l1) {
            fields_differ |= render(&tile, &l1) != render(&tile, &l2);
            let (h1, s1, _) = episode(&tile, &l1);
            let (h2, s2, _) = episode(&tile, &l2);
            assert_eq!(h1.projected_claims, h2.projected_claims, "{name}");
            assert_eq!(h1.independent_roots, h2.independent_roots, "{name}");
            assert_eq!((s1, s2), (INNER_PIXELS, INNER_PIXELS), "{name}");
        }
        assert!(fields_differ, "the two laws must render differently");
    }

    /// FAILS IF: the loop is not replayable or counts an observation twice.
    /// Running it again gives the same horizon; presenting every observation
    /// again to the final horizon is an echo and moves nothing.
    #[test]
    fn replay_is_identical_and_repetition_is_an_echo() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        for (name, tile) in fixtures(&law) {
            let (h, steps, _) = episode(&tile, &law);
            assert_eq!(episode(&tile, &law).0, h, "{name}");
            assert_eq!(usize::from(h.revision_index), steps, "{name}");
            for y in INNER {
                for x in INNER {
                    let o = observe(&tile, Morton8x8::from_xy(x, y));
                    let (next, delta) = admit(&h, &encounter_from_observation(&h, &o));
                    assert_eq!(delta.kind, RevisionKind::Echo, "{name}");
                    assert_eq!(delta.evidential_effect, EvidentialEffect::NoIncrease);
                    assert_eq!(next, h, "{name}");
                }
            }
        }
    }

    /// FAILS IF: the render's visit order changes from what was measured.
    /// The render orders the visits; it does not decide what is found (that
    /// is `belief_depends_on_the_tile_not_on_the_law`). Whether the order is
    /// useful is NOT established: steering by the steepest unobserved pixel
    /// covers the split tile's 12 inner boundary pixels at step 24 against 34
    /// for a row-major scan, but the quadrants tile at 35 against 34. The
    /// steepest rendered pixel on the split tile is (6, 7), one pixel off the
    /// material boundary (x = 7, 8): the rendered maximum is displaced.
    #[test]
    fn the_render_orders_the_visits_and_the_order_is_pinned() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (a, b) = law.pair_with_code(-100..=-80);
        let split = split_at(a, b, 8);
        assert_eq!(
            count(&boundary_oracle(&split).intersection(&inner_mask())),
            12
        );
        let w = strongest_boundary_where(&render(&split, &law), |_| true).unwrap();
        assert_eq!((w.at.x(), w.at.y()), (6, 7));
        assert!(!observe(&split, w.at).boundary);
        let measured: Vec<(usize, usize)> = fixtures(&law)
            .iter()
            .map(|(_, t)| (episode(t, &law).2, row_major_coverage(t)))
            .collect();
        assert_eq!(measured, [(24, 34), (1, 1), (35, 34), (1, 1), (35, 36)]);
    }

    /// FAILS IF: a loop step allocates: the witness, the observation, the
    /// encounter, the delta and both horizons are fixed-size values.
    #[test]
    fn a_loop_step_allocates_nothing() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (a, b) = law.pair_with_code(-100..=-80);
        let tile = split_at(a, b, 8);
        let field = render(&tile, &law);
        let h = prior();
        let (r, n, bytes) = allocations_during(|| step(&tile, &field, &h));
        assert!(r.is_some());
        assert_eq!((n, bytes), (0, 0));
    }
}
