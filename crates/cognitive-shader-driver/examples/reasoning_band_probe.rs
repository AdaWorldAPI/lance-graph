//! D-GSO-7 (P7): reasoning-band earning and downgrade.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §7 (Pearl 2³ projection), §8 (bits 61..63 are a permission band) and §18 P7.
//!
//! Claim under test: the reasoning band on a `CausalEdge64` rises only when a
//! proof obligation was actually executed and passed, falls under
//! contradicting independent evidence or detected confounding, and every
//! change replays from the same events.
//!
//! # Reused, unchanged
//!
//! - **The band:** `ReasoningBand` in bits 61..63, written only with
//!   `with_reasoning_band` and read only through
//!   `band_reading::BandDeclarations::project_band` (declared `Present`,
//!   asserted provenance). An unreadable register gives no transition.
//! - **The projection a test ran under:** Pearl's `CausalMask`: `SO`
//!   association, `PO` intervention, `SPO` counterfactual, `SP` confounder
//!   check.
//! - **Independent evidence:** `ontology_warrant::Quorum` counts (silence is
//!   abstention).
//! - **Confounding:** `CausalMask::simpsons_paradox_risk` on the `SO` and `PO`
//!   direction triads.
//!
//! # The ladder this probe enforces (a policy pin, not canon)
//!
//! | target band | obligation | must already hold |
//! |---|---|---|
//! | `Association` | an `SO` observation a majority of speaking sources corroborate | — |
//! | `Causal` | an intervention trial under `PO`, executed here and passed | `Association` |
//! | `Counterfactual` | a removal attack under `SPO`, executed here and passed | `Causal` |
//!
//! # A pass is computed, never supplied
//!
//! An event carries trial *data*, not a verdict. [`run`] executes the trial and
//! derives the outcome, so no caller can hand in a `Passed`:
//!
//! - **Intervention (`PO`):** treated and control counts from a run under
//!   do(P). Passed when both arms were measured and the treated rate is
//!   higher; failed otherwise; not run when an arm is empty.
//! - **Counterfactual (`SPO`):** a removal attack on premise masks. The
//!   conclusion is the rule's premise set. Passed when it derives with the
//!   candidate and stops deriving without it; failed when it does not derive
//!   at all or still derives without the candidate (dispensable); not run when
//!   the candidate is empty or not among the premises.
//!
//! Replay executes the trials again from the same data.
//!
//! One rung per obligation, no skipping: an observation never certifies more
//! than association, and a passed intervention on an edge that never held
//! association does not raise it.
//!
//! Downgrades:
//!
//! - a failed test drops the band to the rung below the one it guarded;
//! - contradicting independent evidence caps the band at `Association` when
//!   the contradiction is a minority, and drops it to `Surface` when the
//!   contradicting sources outnumber the corroborating ones;
//! - detected confounding caps the band at `Association`.
//!
//! `Relation`, `Perspective`, `Meta` and `Transcendent` have no obligation here
//! and are never reached. The ordinals are not the ladder; the proof chain is.
//!
//! # What this probe does not decide
//!
//! - The thresholds (majority, minority cap) are policy pins.
//! - The trial data are evidence the caller supplies; the probe does not
//!   collect them. Whether that evidence is genuine is the evidence layer's
//!   job; what this probe guarantees is that the band follows from executing
//!   the trial on it, not from a stated outcome.
//! - The trial rules (rate comparison, set-based removal) are minimal stand-ins
//!   for real intervention statistics and derivation.
//! - No writes to stored rows. The band lives on a local `CausalEdge64`.
//!
//! Run: `cargo run -p cognitive-shader-driver --example reasoning_band_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example reasoning_band_probe`
//!
//! D-EPI-MIG-0 (2026-10-07): PROBE-ONLY legacy. Under the canonical
//! Cartesian reading (`EpistemicState5 = Topology2 × Certification3`) the
//! topology half this probe writes is canonical as-is, but the 61..63 half is
//! written through the historical `ReasoningBand` names (reasoning levels:
//! `Surface … Transcendent`), which have no declared mapping onto the
//! certification coordinate. Mapping `Causal` to `Supports` or `Causes` would
//! be semantic invention, so the probe is not migrated; it keeps the
//! deprecated band writer as a record of that reading. No production code may
//! follow it.
//!
//! D-EPI-LEGACY-DEPROJECT-0 (2026-10-07): superseded by P7a
//! (`relational_certification_probe`), which owns the certification rungs
//! with different obligations (P7a `Causes` needs ≥ 2 intervention-backed
//! sources; this probe's `Causal` needs one). The `Counterfactual` rung —
//! survived a removal attack — is counterfactual evidence and belongs to
//! `lance_graph_planner::chain_counterfactual`; P7a states that removal is
//! not the causal path. Kept as shipped (#1360).
#![allow(deprecated)]

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::ReasoningBand;
use causal_edge::pearl::CausalMask;
use lance_graph_contract::band_reading::{
    BandDeclarations, BandPresence, BandReadError, BandReading, EdgeProvenance,
};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::ontology_warrant::Quorum;
use lance_graph_contract::rail_geometry::RailAxis;

/// The class whose edges carry a band in this probe.
const CLASS: ClassId = 0x0901;
/// The rail the band is declared on.
const RAIL: RailAxis = RailAxis::Taxonomy;

/// One class, band declared present.
fn declarations() -> BandDeclarations {
    let mut d = BandDeclarations::new();
    d.declare(
        CLASS,
        RAIL,
        BandReading {
            band: BandPresence::Present,
            ..BandReading::ZERO_FALLBACK
        },
    );
    d
}

/// How a trial ended. Computed by [`run`], never supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestOutcome {
    Passed,
    Failed,
    NotRun,
}

/// Counts from an intervention run under do(P): hits and size of each arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InterventionTrial {
    treated_hits: u16,
    treated_n: u16,
    control_hits: u16,
    control_n: u16,
}

/// A removal attack: does the conclusion (the `rule`'s premise set) still
/// derive from `premises` once `candidate` is removed?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RemovalTrial {
    premises: u64,
    rule: u64,
    candidate: u64,
}

/// The data of one proof trial. Its projection is fixed by its kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trial {
    Intervention(InterventionTrial),
    Counterfactual(RemovalTrial),
    /// A test was named under a projection but nothing was executed.
    NotRun(CausalMask),
}

/// Execute a trial and derive its projection and outcome from its data.
fn run(trial: Trial) -> (CausalMask, TestOutcome) {
    match trial {
        Trial::Intervention(t) => {
            let outcome = if t.treated_n == 0 || t.control_n == 0 {
                TestOutcome::NotRun
            } else if u32::from(t.treated_hits) * u32::from(t.control_n)
                > u32::from(t.control_hits) * u32::from(t.treated_n)
            {
                TestOutcome::Passed
            } else {
                TestOutcome::Failed
            };
            (CausalMask::PO, outcome)
        }
        Trial::Counterfactual(t) => {
            let derives = |p: u64| p & t.rule == t.rule;
            let outcome = if t.candidate == 0 || t.premises & t.candidate != t.candidate {
                TestOutcome::NotRun
            } else if derives(t.premises) && !derives(t.premises & !t.candidate) {
                TestOutcome::Passed
            } else {
                TestOutcome::Failed
            };
            (CausalMask::SPO, outcome)
        }
        Trial::NotRun(mask) => (mask, TestOutcome::NotRun),
    }
}

/// One thing that happened to the claim the edge carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// Observational evidence folded under a projection.
    Observed { mask: CausalMask, quorum: Quorum },
    /// A proof trial, executed when the event is applied.
    Tested(Trial),
    /// Independent evidence bearing on the claim: corroborating, silent and
    /// conflicting sources.
    Independent { quorum: Quorum },
    /// A confounder check: the `SO` and `PO` direction triads.
    ConfounderCheck { so_direction: u8, po_direction: u8 },
}

/// Does a majority of speaking sources corroborate? Silence is not counted.
const fn majority_corroborates(q: Quorum) -> bool {
    q.corroborating > q.conflicting
}

/// The band after `event`, from `band`. Pure; the ladder above in code.
fn next_band(band: ReasoningBand, event: Event) -> ReasoningBand {
    use ReasoningBand::{Association, Causal, Counterfactual, Surface};
    let cap = |b: ReasoningBand, at: ReasoningBand| {
        if b.to_bits_3() > at.to_bits_3() {
            at
        } else {
            b
        }
    };
    match event {
        Event::Observed { mask, quorum } => {
            if mask == CausalMask::SO && band == Surface && majority_corroborates(quorum) {
                Association
            } else {
                band
            }
        }
        Event::Tested(trial) => match run(trial) {
            (CausalMask::PO, TestOutcome::Passed) if band == Association => Causal,
            (CausalMask::SPO, TestOutcome::Passed) if band == Causal => Counterfactual,
            (CausalMask::PO, TestOutcome::Failed) => cap(band, Association),
            (CausalMask::SPO, TestOutcome::Failed) => cap(band, Causal),
            _ => band,
        },
        Event::Independent { quorum } => {
            if quorum.conflicting > quorum.corroborating {
                Surface
            } else if quorum.conflicting > 0 {
                cap(band, Association)
            } else {
                band
            }
        }
        Event::ConfounderCheck {
            so_direction,
            po_direction,
        } => {
            if CausalMask::simpsons_paradox_risk(so_direction, po_direction) {
                cap(band, Association)
            } else {
                band
            }
        }
    }
}

/// Read the band through the contract, apply `event`, write it back.
fn apply(
    decl: &BandDeclarations,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
    event: Event,
) -> Result<CausalEdge64, BandReadError> {
    let raw = decl.project_band(CLASS, RAIL, edge.reasoning_band().to_bits_3(), provenance)?;
    let band = ReasoningBand::from_bits_3(raw);
    Ok(edge.with_reasoning_band(next_band(band, event)))
}

/// Fold a whole event sequence over an edge. Fails on the first unreadable
/// read.
fn replay(
    decl: &BandDeclarations,
    start: CausalEdge64,
    provenance: EdgeProvenance,
    events: &[Event],
) -> Result<CausalEdge64, BandReadError> {
    events
        .iter()
        .try_fold(start, |e, &ev| apply(decl, e, provenance, ev))
}

/// An intervention run whose treated arm beats control (30/40 vs 12/40).
const INTERVENTION_PASS: Trial = Trial::Intervention(InterventionTrial {
    treated_hits: 30,
    treated_n: 40,
    control_hits: 12,
    control_n: 40,
});
/// A removal attack where the candidate is a required premise.
const REMOVAL_PASS: Trial = Trial::Counterfactual(RemovalTrial {
    premises: 0b0111,
    rule: 0b0011,
    candidate: 0b0010,
});
/// A removal attack where the candidate is not needed: the conclusion still
/// derives without it.
const REMOVAL_FAIL: Trial = Trial::Counterfactual(RemovalTrial {
    premises: 0b0111,
    rule: 0b0011,
    candidate: 0b0100,
});

/// The full earning path: observe, intervene, counterfactual.
#[cfg(test)]
fn earning_path() -> [Event; 3] {
    [
        Event::Observed {
            mask: CausalMask::SO,
            quorum: Quorum::new(3, 1, 1),
        },
        Event::Tested(INTERVENTION_PASS),
        Event::Tested(REMOVAL_PASS),
    ]
}

fn main() {
    let decl = declarations();
    let scenario = [
        Event::Observed {
            mask: CausalMask::SO,
            quorum: Quorum::new(3, 1, 1),
        },
        Event::Tested(Trial::NotRun(CausalMask::PO)),
        Event::Tested(INTERVENTION_PASS),
        Event::ConfounderCheck {
            so_direction: 0b100,
            po_direction: 0b000,
        },
        Event::Tested(INTERVENTION_PASS),
        Event::Tested(REMOVAL_PASS),
        Event::Tested(REMOVAL_FAIL),
        Event::Independent {
            quorum: Quorum::new(1, 0, 3),
        },
    ];
    let mut edge = CausalEdge64::ZERO;
    println!("start: {:?}", edge.reasoning_band());
    for ev in scenario {
        edge = apply(&decl, edge, EdgeProvenance::V2Stamped, ev).expect("declared");
        println!("{ev:?} -> {:?}", edge.reasoning_band());
    }
    let again = replay(
        &decl,
        CausalEdge64::ZERO,
        EdgeProvenance::V2Stamped,
        &scenario,
    )
    .expect("declared");
    assert_eq!(again.0, edge.0);
    println!("replayed: same edge bits");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ReasoningBand::*;

    const ALL_BANDS: [ReasoningBand; 8] = [
        Surface,
        Association,
        Relation,
        Causal,
        Counterfactual,
        Perspective,
        Meta,
        Transcendent,
    ];
    const ALL_MASKS: [CausalMask; 8] = [
        CausalMask::None,
        CausalMask::O,
        CausalMask::P,
        CausalMask::PO,
        CausalMask::S,
        CausalMask::SO,
        CausalMask::SP,
        CausalMask::SPO,
    ];

    fn obs(mask: CausalMask, q: Quorum) -> Event {
        Event::Observed { mask, quorum: q }
    }
    /// A trial whose execution gives `(mask, outcome)`. Only `PO` and `SPO`
    /// trials can pass or fail; any other projection can only be named, so it
    /// maps to `NotRun`.
    fn test(mask: CausalMask, outcome: TestOutcome) -> Event {
        let trial = match (mask, outcome) {
            (CausalMask::PO, TestOutcome::Passed) => INTERVENTION_PASS,
            (CausalMask::PO, TestOutcome::Failed) => Trial::Intervention(InterventionTrial {
                treated_hits: 10,
                treated_n: 40,
                control_hits: 12,
                control_n: 40,
            }),
            (CausalMask::SPO, TestOutcome::Passed) => REMOVAL_PASS,
            (CausalMask::SPO, TestOutcome::Failed) => REMOVAL_FAIL,
            (m, _) => Trial::NotRun(m),
        };
        Event::Tested(trial)
    }

    /// Every event the tests sweep: all masks × a few quorums for
    /// observations, all masks × outcomes for tests, a few independent
    /// quorums and confounder checks.
    fn all_events() -> Vec<Event> {
        let quorums = [
            Quorum::new(0, 0, 0),
            Quorum::new(3, 1, 1),
            Quorum::new(1, 0, 1),
            Quorum::new(1, 0, 3),
            Quorum::new(0, 5, 0),
        ];
        let mut v = Vec::new();
        for m in ALL_MASKS {
            for q in quorums {
                v.push(obs(m, q));
            }
            for o in [
                TestOutcome::Passed,
                TestOutcome::Failed,
                TestOutcome::NotRun,
            ] {
                v.push(test(m, o));
            }
        }
        for q in quorums {
            v.push(Event::Independent { quorum: q });
        }
        for (so, po) in [(0b100, 0b000), (0b100, 0b100), (0, 0)] {
            v.push(Event::ConfounderCheck {
                so_direction: so,
                po_direction: po,
            });
        }
        v
    }

    /// FAILS IF: any amount or kind of observational evidence lifts the band
    /// past `Association`, or an observation under a non-`SO` projection
    /// lifts it at all.
    #[test]
    fn association_cannot_skip_to_causal() {
        let decl = declarations();
        let flood: Vec<Event> = ALL_MASKS
            .iter()
            .flat_map(|&m| std::iter::repeat_n(obs(m, Quorum::new(50, 0, 0)), 20))
            .collect();
        let e = replay(&decl, CausalEdge64::ZERO, EdgeProvenance::V2Stamped, &flood).unwrap();
        assert_eq!(e.reasoning_band(), Association);

        for m in ALL_MASKS.into_iter().filter(|&m| m != CausalMask::SO) {
            assert_eq!(
                next_band(Surface, obs(m, Quorum::new(9, 0, 0))),
                Surface,
                "{m:?}"
            );
        }
        // A split or silent quorum certifies nothing either.
        assert_eq!(
            next_band(Surface, obs(CausalMask::SO, Quorum::new(1, 0, 1))),
            Surface
        );
        assert_eq!(
            next_band(Surface, obs(CausalMask::SO, Quorum::new(0, 9, 0))),
            Surface
        );
    }

    /// FAILS IF: a test that was not run, or failed, raises the band; a passed
    /// intervention fails to raise `Association` to `Causal`; or a proof is
    /// accepted for a rung whose predecessor is not held.
    #[test]
    fn only_an_executed_passing_test_raises_and_only_one_rung() {
        assert_eq!(
            next_band(Association, test(CausalMask::PO, TestOutcome::Passed)),
            Causal
        );
        assert_eq!(
            next_band(Causal, test(CausalMask::SPO, TestOutcome::Passed)),
            Counterfactual
        );

        for o in [TestOutcome::NotRun, TestOutcome::Failed] {
            assert_eq!(
                next_band(Association, test(CausalMask::PO, o)),
                Association,
                "{o:?}"
            );
            assert_eq!(next_band(Causal, test(CausalMask::SPO, o)), Causal, "{o:?}");
        }
        // No skipping: an intervention on an edge that never held association,
        // and a counterfactual on an edge that never held causal permission.
        assert_eq!(
            next_band(Surface, test(CausalMask::PO, TestOutcome::Passed)),
            Surface
        );
        assert_eq!(
            next_band(Association, test(CausalMask::SPO, TestOutcome::Passed)),
            Association
        );
        // A passed test under the wrong projection certifies nothing.
        for m in [
            CausalMask::SO,
            CausalMask::SP,
            CausalMask::S,
            CausalMask::None,
        ] {
            assert_eq!(
                next_band(Association, test(m, TestOutcome::Passed)),
                Association,
                "{m:?}"
            );
        }
    }

    /// FAILS IF: a trial passes when its data do not support it, or the band
    /// follows anything but the executed trial. There is no `Passed` input:
    /// the outcome is derived from the data every time the event is applied.
    #[test]
    fn a_pass_is_derived_from_the_trial_data() {
        let iv = |th, tn, ch, cn| {
            Trial::Intervention(InterventionTrial {
                treated_hits: th,
                treated_n: tn,
                control_hits: ch,
                control_n: cn,
            })
        };
        // Intervention: higher treated rate passes; equal or lower fails;
        // an unmeasured arm means the trial was not run.
        assert_eq!(
            run(iv(30, 40, 12, 40)),
            (CausalMask::PO, TestOutcome::Passed)
        );
        assert_eq!(
            run(iv(12, 40, 12, 40)),
            (CausalMask::PO, TestOutcome::Failed)
        );
        assert_eq!(run(iv(3, 4, 30, 40)), (CausalMask::PO, TestOutcome::Failed));
        assert_eq!(run(iv(3, 4, 0, 0)), (CausalMask::PO, TestOutcome::NotRun));
        // Rates, not counts: 3/4 beats 60/100 although 3 < 60.
        assert_eq!(
            run(iv(3, 4, 60, 100)),
            (CausalMask::PO, TestOutcome::Passed)
        );

        // Removal: a required premise passes; a dispensable one fails; a
        // conclusion that never derived fails; a candidate outside the
        // premises gives nothing to attack.
        let rm = |premises, rule, candidate| {
            Trial::Counterfactual(RemovalTrial {
                premises,
                rule,
                candidate,
            })
        };
        assert_eq!(
            run(rm(0b0111, 0b0011, 0b0010)),
            (CausalMask::SPO, TestOutcome::Passed)
        );
        assert_eq!(
            run(rm(0b0111, 0b0011, 0b0100)),
            (CausalMask::SPO, TestOutcome::Failed)
        );
        assert_eq!(
            run(rm(0b0001, 0b0011, 0b0001)),
            (CausalMask::SPO, TestOutcome::Failed)
        );
        assert_eq!(
            run(rm(0b0011, 0b0011, 0b1000)),
            (CausalMask::SPO, TestOutcome::NotRun)
        );
        assert_eq!(
            run(rm(0b0011, 0b0011, 0)),
            (CausalMask::SPO, TestOutcome::NotRun)
        );

        // The band follows the data: the same event shape with unsupporting
        // data does not raise, and replay re-executes rather than remembering.
        let decl = declarations();
        let assoc = [earning_path()[0]];
        let start = replay(&decl, CausalEdge64::ZERO, EdgeProvenance::V2Stamped, &assoc).unwrap();
        let good = apply(
            &decl,
            start,
            EdgeProvenance::V2Stamped,
            Event::Tested(iv(30, 40, 12, 40)),
        );
        let bad = apply(
            &decl,
            start,
            EdgeProvenance::V2Stamped,
            Event::Tested(iv(12, 40, 30, 40)),
        );
        assert_eq!(good.unwrap().reasoning_band(), Causal);
        assert_eq!(bad.unwrap().reasoning_band(), Association);
    }

    /// FAILS IF: any transition from any band raises it by more than one rung
    /// of the ladder, or raises it without the matching obligation. Sweeps
    /// all 8 bands × every event kind.
    #[test]
    fn no_rise_without_its_obligation_anywhere() {
        let mut rises = 0;
        for b in ALL_BANDS {
            for ev in all_events() {
                let n = next_band(b, ev);
                if n.to_bits_3() <= b.to_bits_3() {
                    continue;
                }
                rises += 1;
                let legal = match (b, n, ev) {
                    (
                        Surface,
                        Association,
                        Event::Observed {
                            mask: CausalMask::SO,
                            ..
                        },
                    ) => true,
                    (Association, Causal, Event::Tested(t)) => {
                        run(t) == (CausalMask::PO, TestOutcome::Passed)
                    }
                    (Causal, Counterfactual, Event::Tested(t)) => {
                        run(t) == (CausalMask::SPO, TestOutcome::Passed)
                    }
                    _ => false,
                };
                assert!(legal, "{b:?} --{ev:?}--> {n:?}");
            }
        }
        assert_eq!(rises, 3, "exactly the three ladder steps can rise");
    }

    /// FAILS IF: contradicting independent evidence or confounding leaves an
    /// earned band in place, or silence counts as contradiction.
    #[test]
    fn contradiction_and_confounding_lower_the_band() {
        // Majority contradiction: suspended to Surface.
        assert_eq!(
            next_band(
                Counterfactual,
                Event::Independent {
                    quorum: Quorum::new(1, 0, 3)
                }
            ),
            Surface
        );
        // Minority contradiction: causal permission suspended, association kept.
        assert_eq!(
            next_band(
                Counterfactual,
                Event::Independent {
                    quorum: Quorum::new(3, 0, 1)
                }
            ),
            Association
        );
        assert_eq!(
            next_band(
                Causal,
                Event::Independent {
                    quorum: Quorum::new(3, 0, 1)
                }
            ),
            Association
        );
        // Silence and pure corroboration lower nothing.
        assert_eq!(
            next_band(
                Causal,
                Event::Independent {
                    quorum: Quorum::new(0, 9, 0)
                }
            ),
            Causal
        );
        assert_eq!(
            next_band(
                Causal,
                Event::Independent {
                    quorum: Quorum::new(4, 2, 0)
                }
            ),
            Causal
        );
        // Confounding (SO and PO disagree in direction) caps at Association.
        let confounded = Event::ConfounderCheck {
            so_direction: 0b100,
            po_direction: 0b000,
        };
        let clean = Event::ConfounderCheck {
            so_direction: 0b100,
            po_direction: 0b100,
        };
        assert_eq!(next_band(Counterfactual, confounded), Association);
        assert_eq!(next_band(Counterfactual, clean), Counterfactual);
        // A failed test drops to the rung below the one it guarded.
        assert_eq!(
            next_band(Counterfactual, test(CausalMask::SPO, TestOutcome::Failed)),
            Causal
        );
        assert_eq!(
            next_band(Counterfactual, test(CausalMask::PO, TestOutcome::Failed)),
            Association
        );
        // A downgrade never raises: Surface stays Surface.
        for ev in all_events() {
            if let Event::Independent { .. } | Event::ConfounderCheck { .. } = ev {
                assert_eq!(next_band(Surface, ev), Surface, "{ev:?}");
            }
        }
    }

    /// FAILS IF: the same events from the same edge end in different bits, the
    /// band on the edge is not the one the ladder computes, or a band change
    /// disturbs the rest of the edge.
    #[test]
    fn band_changes_replay_from_the_events() {
        let decl = declarations();
        let mut events = earning_path().to_vec();
        events.push(Event::Independent {
            quorum: Quorum::new(3, 0, 1),
        });
        events.push(test(CausalMask::PO, TestOutcome::Passed));
        let start = CausalEdge64::ZERO.with_w_slot(17);

        let a = replay(&decl, start, EdgeProvenance::V2Stamped, &events).unwrap();
        let b = replay(&decl, start, EdgeProvenance::V2Stamped, &events).unwrap();
        assert_eq!(a.0, b.0);
        // Earned to Counterfactual, cut to Association by a minority
        // contradiction, re-earned to Causal by a fresh passing intervention.
        assert_eq!(a.reasoning_band(), Causal);
        // Only bits 61..63 moved.
        assert_eq!(a.with_reasoning_band(Surface).0, start.0);

        // The trace is the fold of `next_band`, step for step.
        let mut band = Surface;
        let mut edge = start;
        for &ev in &events {
            band = next_band(band, ev);
            edge = apply(&decl, edge, EdgeProvenance::V2Stamped, ev).unwrap();
            assert_eq!(edge.reasoning_band(), band);
        }
    }

    /// FAILS IF: a band is read from bits the contract refuses: unknown or v1
    /// provenance, an undeclared class, or a class that declares no band.
    #[test]
    fn an_unreadable_band_gives_no_transition() {
        let decl = declarations();
        let ev = earning_path()[0];
        for p in [EdgeProvenance::Unknown, EdgeProvenance::V1Legacy] {
            assert_eq!(
                apply(&decl, CausalEdge64::ZERO, p, ev),
                Err(BandReadError::UnknownProvenance),
                "{p:?}"
            );
        }
        assert_eq!(
            apply(
                &BandDeclarations::new(),
                CausalEdge64::ZERO,
                EdgeProvenance::V2Stamped,
                ev
            ),
            Err(BandReadError::UndeclaredClass(CLASS))
        );
        let mut absent = BandDeclarations::new();
        absent.declare(CLASS, RAIL, BandReading::ZERO_FALLBACK);
        assert_eq!(
            apply(&absent, CausalEdge64::ZERO, EdgeProvenance::V2Stamped, ev),
            Err(BandReadError::BandAbsent)
        );
        assert!(apply(&decl, CausalEdge64::ZERO, EdgeProvenance::V3Register, ev).is_ok());
    }
}
