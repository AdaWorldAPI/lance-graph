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
//! | `Causal` | an intervention test under `PO`, run and passed | `Association` |
//! | `Counterfactual` | a counterfactual test under `SPO`, run and passed | `Causal` |
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
//! - A "test" is supplied as an outcome; running the intervention or
//!   counterfactual itself is the caller's job. What the probe enforces is
//!   that a `NotRun` outcome can never raise the band.
//! - No writes to stored rows. The band lives on a local `CausalEdge64`.
//!
//! Run: `cargo run -p cognitive-shader-driver --example reasoning_band_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example reasoning_band_probe`

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

/// How a test ended. `NotRun` is the honest default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestOutcome {
    Passed,
    Failed,
    NotRun,
}

/// One thing that happened to the claim the edge carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// Observational evidence folded under a projection.
    Observed { mask: CausalMask, quorum: Quorum },
    /// A proof test under a projection, with its outcome.
    Tested {
        mask: CausalMask,
        outcome: TestOutcome,
    },
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
        Event::Tested { mask, outcome } => match (mask, outcome) {
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

/// The full earning path: observe, intervene, counterfactual.
#[cfg(test)]
fn earning_path() -> [Event; 3] {
    [
        Event::Observed {
            mask: CausalMask::SO,
            quorum: Quorum::new(3, 1, 1),
        },
        Event::Tested {
            mask: CausalMask::PO,
            outcome: TestOutcome::Passed,
        },
        Event::Tested {
            mask: CausalMask::SPO,
            outcome: TestOutcome::Passed,
        },
    ]
}

fn main() {
    let decl = declarations();
    let scenario = [
        Event::Observed {
            mask: CausalMask::SO,
            quorum: Quorum::new(3, 1, 1),
        },
        Event::Tested {
            mask: CausalMask::PO,
            outcome: TestOutcome::NotRun,
        },
        Event::Tested {
            mask: CausalMask::PO,
            outcome: TestOutcome::Passed,
        },
        Event::ConfounderCheck {
            so_direction: 0b100,
            po_direction: 0b000,
        },
        Event::Tested {
            mask: CausalMask::PO,
            outcome: TestOutcome::Passed,
        },
        Event::Tested {
            mask: CausalMask::SPO,
            outcome: TestOutcome::Failed,
        },
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
    fn test(mask: CausalMask, outcome: TestOutcome) -> Event {
        Event::Tested { mask, outcome }
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
                let legal = matches!(
                    (b, n, ev),
                    (
                        Surface,
                        Association,
                        Event::Observed {
                            mask: CausalMask::SO,
                            ..
                        }
                    ) | (
                        Association,
                        Causal,
                        Event::Tested {
                            mask: CausalMask::PO,
                            outcome: TestOutcome::Passed
                        }
                    ) | (
                        Causal,
                        Counterfactual,
                        Event::Tested {
                            mask: CausalMask::SPO,
                            outcome: TestOutcome::Passed
                        }
                    )
                );
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
