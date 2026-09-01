//! **D-DCR-3 (W3) — counterfactual replay.** `dismech-causal-replay-v1` §3 W3.
//!
//! The SAME W1 replay with one edge cut — Pearl rung 3. **Never a second
//! replay path:** [`crate::dismech_replay::replay_chain`] runs both the
//! factual and the counterfactual arm, so a divergence can only come from the
//! cut and never from two implementations drifting apart.
//!
//! # The question this answers: was that edge LOAD-BEARING?
//!
//! Cut one step, replay, and compare — but compare *what*. Not the trace: the
//! trace differs whenever anything is removed, because `NarsTables::revise`
//! accumulates evidence (`w/(w+1)`), so even deleting a duplicate step moves
//! the truth. Byte-difference is therefore the wrong instrument here; it
//! answers "did anything change" when the question is "did the ANSWER change".
//!
//! So the comparison is a **thresholded verdict** — which is the operator's
//! kind-2 Mengenlehre (2026-09-01): elimination is a READING of a continuous
//! quantity at a measured threshold, never the primitive. A load-bearing edge
//! moves the chain's truth ACROSS the threshold; a redundant one moves it and
//! stays on the same side.
//!
//! # The counterfactual is PRESERVED, not resolved
//!
//! `contract::counterfactual`'s invariant is that a counterfactual *"stays in
//! a separate lane — it is NEVER written as observed SPO truth"*, mechanically
//! enforced by the `InferenceType::Counterfactual` (`-6`) mantissa. That is
//! the same instinct as the ruling above: the road not taken is retained as
//! structure, not collapsed away. [`counterfactual_replay`] therefore tags the
//! cut arm's terminal edge before returning it — the cut result can never be
//! mistaken for something that was observed.
//!
//! # Precision about the answer, not just its polarity
//!
//! A verdict that collapsed to `bool` would throw away what kind of relation
//! was cut. [`EdgeRole`] carries the cut edge's own `CausalTopology` (bits
//! 59-60) and `ReasoningBand` (bits 61-63) alongside the verdict, so a
//! consumer can tell "the edge that EXPLAINS this chain is load-bearing" from
//! "an edge that merely RELATES TO it is". Same bits, read through their own
//! lenses; nothing is re-derived.

use causal_edge::edge::InferenceType;
use causal_edge::tables::NarsTables;
use causal_edge::CausalEdge64;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::counterfactual::EpisodicEdge;

use crate::dismech_replay::{replay_chain, ChainStep, ComposeTables, ReplayError, ReplayTraceRow};

/// The bridge `contract::counterfactual` documents as BLOCKED on workspace
/// structure — `impl EpisodicEdge for CausalEdge64`.
///
/// It lives HERE because this is the first crate that depends on **both**
/// `lance-graph-contract` (zero-dep, so it cannot see `CausalEdge64`) and
/// `causal-edge` (which must not depend on the contract). Neither producer can
/// host it; the consumer that already has both can. That is the same seam
/// `recipe_vocab` occupies for `ogar-loco` in the armed tier.
///
/// A newtype rather than a blanket impl on `CausalEdge64`: the write is a
/// deliberate act at one seam, and a bare `impl` would let any caller reach
/// the counterfactual nibble by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterfactualEdge(pub CausalEdge64);

impl EpisodicEdge for CounterfactualEdge {
    fn set_inference_mantissa(&mut self, m: i8) {
        self.0 = self.0.with_inference_mantissa(m);
    }
    fn inference_mantissa(&self) -> i8 {
        self.0.inference_mantissa()
    }
}

/// Whether a chain's replayed truth clears the consistency bar.
///
/// Deliberately a two-state reading of a continuous quantity, per the kind-2
/// ruling — the threshold is where elimination is allowed to happen, and it is
/// named rather than implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The chain's replayed truth clears the bar.
    Consistent,
    /// It does not.
    Inconsistent,
}

/// The FREQUENCY a replayed chain must reach to read as [`Verdict::Consistent`].
///
/// A POLICY PIN for the bar's VALUE (128 is the midpoint of `u8`, not a
/// derived number) — but the choice of **frequency over confidence is
/// measured, and it matters**.
///
/// `NarsTables::build(1)` drives revision's confidence to a fixed point:
/// measured across a weak 3-chain, a strong 4-chain, and a mixed one, the
/// terminal confidence was **170 in every case**, while frequency separated
/// them cleanly (78 / 247 / 93). A confidence-based verdict would therefore
/// have been a **vacuous threshold** — every chain on the same side of every
/// bar, a gate that cannot fire and cannot stay silent, discriminating
/// nothing while looking rigorous.
///
/// It is also the semantically right axis: "is this chain consistent?" asks
/// how strongly the composed relation holds, not how much evidence has piled
/// up behind it. Confidence saturating is the substrate working as designed;
/// reading a verdict off it was the error.
///
/// [`the_bar_is_not_inert`] pins that the knob does something in BOTH
/// directions — the inertness test a threshold parameter owes.
pub const DEFAULT_FREQUENCY_BAR: u8 = 128;

/// Read a replayed chain's terminal truth as a verdict at `bar`.
///
/// Reads FREQUENCY — see [`DEFAULT_FREQUENCY_BAR`] for why confidence is the
/// wrong axis here, measured rather than assumed.
#[must_use]
pub fn verdict_at(trace: &[ReplayTraceRow], bar: u8) -> Verdict {
    match trace.last() {
        Some(row) if row.edge.frequency_u8() >= bar => Verdict::Consistent,
        // An empty replay asserts nothing, so it cannot be consistent. Fails
        // closed, matching `dismech_evidence`'s rule that an absence is never
        // silently promoted into an assertion.
        _ => Verdict::Inconsistent,
    }
}

/// What cutting one edge revealed about it.
///
/// Carries the cut edge's own relation flavour beside the verdict, so a
/// consumer can distinguish an edge that EXPLAINS from one that merely RELATES
/// TO — the precision the operator's ruling requires of an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRole {
    /// Index of the step that was cut.
    pub step: usize,
    /// The predicate ordinal the cut step travelled under.
    pub predicate: u8,
    /// Verdict with the edge present.
    pub factual: Verdict,
    /// Verdict with the edge cut.
    pub counterfactual: Verdict,
    /// The cut edge's causal topology (bits 59-60, `CausalTopology` lens).
    pub topology: causal_edge::layout::CausalTopology,
    /// The cut edge's reasoning band (bits 61-63).
    pub band: causal_edge::layout::ReasoningBand,
}

impl EdgeRole {
    /// The edge is load-bearing: removing it CHANGED the answer.
    #[must_use]
    pub fn is_load_bearing(&self) -> bool {
        self.factual != self.counterfactual
    }
}

/// The chain with step `index` removed. Returns `None` if the index is out of
/// range — a cut that names no step is refused, never silently a no-op.
#[must_use]
pub fn cut_step(chain: &[ChainStep], index: usize) -> Option<Vec<ChainStep>> {
    if index >= chain.len() {
        return None;
    }
    let mut out = Vec::with_capacity(chain.len() - 1);
    out.extend_from_slice(&chain[..index]);
    out.extend_from_slice(&chain[index + 1..]);
    Some(out)
}

/// Everything a cut needs besides the chain itself, bundled so a caller
/// cannot hand the factual arm one owner and the counterfactual arm another.
///
/// Same reasoning as [`ComposeTables`], which bundles the three palette tables
/// for the same reason: the two arms must be replayed under IDENTICAL
/// conditions, or a divergence stops being attributable to the cut — which is
/// the only thing this module measures.
#[derive(Clone, Copy)]
pub struct CutContext<'a> {
    /// The NARS revision tables.
    pub tables: &'a NarsTables,
    /// The three palette-composition tables.
    pub compose: ComposeTables<'a>,
    /// The owning mailbox for both traces.
    pub owner: MailboxId,
    /// Durable base for the FACTUAL arm; the cut arm reserves the range
    /// immediately after it.
    pub base_seq: u64,
    /// The frequency bar the verdict is read at (see [`DEFAULT_FREQUENCY_BAR`]).
    pub bar: u8,
}

/// Both arms of one counterfactual, with the cut arm tagged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counterfactual {
    /// What the cut revealed about the edge.
    pub role: EdgeRole,
    /// The factual trace — the chain as recorded.
    pub factual: Vec<ReplayTraceRow>,
    /// The counterfactual trace. Its terminal edge carries the
    /// `InferenceType::Counterfactual` (`-6`) mantissa, so it can never be
    /// mistaken for observed truth.
    pub counterfactual: Vec<ReplayTraceRow>,
}

/// Replay `chain` twice — as recorded, and with step `index` cut — and report
/// what the cut edge was doing.
///
/// Both arms go through [`replay_chain`]; there is no second replay path.
///
/// # Errors
///
/// [`ReplayError`] from either arm. `None` when `index` names no step.
pub fn counterfactual_replay(
    chain: &[ChainStep],
    index: usize,
    seed: CausalEdge64,
    cx: CutContext<'_>,
) -> Option<Result<Counterfactual, ReplayError>> {
    let CutContext {
        tables,
        compose,
        owner,
        base_seq,
        bar,
    } = cx;
    let cut = cut_step(chain, index)?;
    let (predicate, cut_edge) = chain[index];

    Some((|| {
        let factual = replay_chain(chain, seed, tables, compose, owner, base_seq)?;
        // The counterfactual arm reserves its OWN durable range, immediately
        // after the factual one — the two traces coexist and must never share
        // a coordinate (`LocalCausalRow::cast_seq` uniqueness).
        let cf_base = crate::dismech_replay::next_base_seq(base_seq, chain.len()).ok_or(
            ReplayError::SequenceExhausted {
                base_seq,
                steps: chain.len(),
            },
        )?;
        let mut counterfactual = replay_chain(&cut, seed, tables, compose, owner, cf_base)?;

        // Tag the road not taken, through the contract's own deposit path.
        if let Some(last) = counterfactual.last_mut() {
            let mut tagged = CounterfactualEdge(last.edge);
            tagged.set_inference_mantissa(InferenceType::Counterfactual.to_mantissa());
            last.edge = tagged.0;
        }

        Ok(Counterfactual {
            role: EdgeRole {
                step: index,
                predicate,
                factual: verdict_at(&factual, bar),
                counterfactual: verdict_at(&counterfactual, bar),
                topology: cut_edge.topology(),
                band: cut_edge.reasoning_band(),
            },
            factual,
            counterfactual,
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::{CausalMask, PlasticityState};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 33) as usize % n
        }
    }

    fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        core::array::from_fn(|_| {
            let mut t = Box::new([0u8; 256 * 256]);
            for v in t.iter_mut() {
                *v = rng.below(256) as u8;
            }
            t
        })
    }

    /// An edge with an explicit truth, so a fixture can place a chain
    /// deliberately on one side of the bar rather than hoping.
    fn edge_with(freq: u8, conf: u8) -> CausalEdge64 {
        CausalEdge64::pack(
            7,
            9,
            11,
            freq,
            conf,
            CausalMask::PO,
            0b101,
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0,
        )
    }

    fn fixture() -> (NarsTables, [Box<[u8; 256 * 256]>; 3]) {
        (NarsTables::build(1), compose_tables())
    }

    /// `cut_step` is the fixture's own instrument, so it is pinned before
    /// anything is concluded from a cut.
    #[test]
    fn cutting_removes_exactly_one_step_and_refuses_an_index_that_names_none() {
        let c: Vec<ChainStep> = (0..5)
            .map(|i| (0x90 + i as u8, edge_with(200, 180)))
            .collect();

        let cut = cut_step(&c, 2).expect("index 2 is in range");
        assert_eq!(cut.len(), 4);
        assert_eq!(
            cut.iter().map(|s| s.0).collect::<Vec<_>>(),
            vec![0x90, 0x91, 0x93, 0x94],
            "exactly the named step must be gone, order otherwise preserved",
        );
        // Refused, never a silent no-op — a cut that named nothing would
        // otherwise report "not load-bearing" for every out-of-range index.
        assert!(cut_step(&c, 5).is_none());
        assert!(cut_step(&c, usize::MAX).is_none());
        assert!(cut_step(&[], 0).is_none());
    }

    /// GATE — can-fire. Cutting a LOAD-BEARING edge flips the verdict.
    ///
    /// The fixture's SHAPE is the coverage: a chain of weak steps whose truth
    /// only clears the bar because ONE strong step carries it. Remove that
    /// step and the answer changes.
    #[test]
    fn cutting_a_load_bearing_edge_flips_the_verdict() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        // MEASURED straddle, not a guessed one: this pair puts the factual
        // arm at frequency 133 and the cut arm at 120, either side of the 128
        // bar. The probe that found it swept weak/strong pairs and chain
        // lengths; most combinations do NOT straddle, which is why the
        // fixture is pinned to numbers rather than to an intuition.
        let chain: Vec<ChainStep> = vec![(0x91, edge_with(250, 250)), (0x92, edge_with(40, 30))];
        let seed = edge_with(200, 200);

        let cf = counterfactual_replay(
            &chain,
            0,
            seed,
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR,
            },
        )
        .expect("index 0 is in range")
        .expect("reservation fits");

        // Anti-vacuity: the factual arm must actually be Consistent, or
        // "the verdict flipped" would be measuring a chain that never held.
        assert_eq!(cf.role.factual, Verdict::Consistent);
        assert_eq!(cf.role.counterfactual, Verdict::Inconsistent);
        assert!(cf.role.is_load_bearing());
        assert_eq!(cf.role.step, 0);
        assert_eq!(
            cf.role.predicate, 0x91,
            "the verdict must name the cut edge"
        );
        // The measured numbers, pinned: a later change to the kernel that
        // moved these would silently turn this gate vacuous.
        assert_eq!(cf.factual.last().unwrap().edge.frequency_u8(), 133);
        assert_eq!(cf.counterfactual.last().unwrap().edge.frequency_u8(), 120);
    }

    /// GATE — can-stay-SILENT. Cutting a REDUNDANT edge must NOT flip the
    /// verdict, on a fixture where the cut demonstrably still MOVES the truth.
    ///
    /// That second clause is the whole test. `NarsTables::revise` accumulates,
    /// so removing any step changes the numbers; if the fixture were built so
    /// the cut changed nothing at all, "the verdict held" would be true of a
    /// no-op and would prove nothing about redundancy.
    #[test]
    fn cutting_a_redundant_edge_moves_the_truth_but_not_the_verdict() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        // Four strong steps: any one of them is redundant to the verdict,
        // because the remaining three still clear the bar comfortably.
        let chain: Vec<ChainStep> = (0..4)
            .map(|i| (0x90 + i as u8, edge_with(250, 250)))
            .collect();
        let seed = edge_with(200, 200);

        let cf = counterfactual_replay(
            &chain,
            2,
            seed,
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR,
            },
        )
        .expect("index 2 is in range")
        .expect("reservation fits");

        assert_eq!(cf.role.factual, Verdict::Consistent);
        assert_eq!(cf.role.counterfactual, Verdict::Consistent);
        assert!(!cf.role.is_load_bearing());

        // ...and the cut DID move the substrate. Without this the silence
        // above would be indistinguishable from a cut that never happened.
        assert_ne!(
            cf.counterfactual.len(),
            cf.factual.len(),
            "the cut arm must be one step shorter",
        );
        let f_conf = cf.factual.last().unwrap().edge.confidence_u8();
        let c_conf = cf.counterfactual.last().unwrap().edge.confidence_u8();
        let f_freq = cf.factual.last().unwrap().edge.frequency_u8();
        let c_freq = cf.counterfactual.last().unwrap().edge.frequency_u8();
        assert!(
            f_conf != c_conf || f_freq != c_freq,
            "the cut must move the truth ({f_freq}/{f_conf} vs {c_freq}/{c_conf}) \
             or this gate is measuring a no-op",
        );
    }

    /// The counterfactual arm is TAGGED, and the factual arm is not — the
    /// road not taken can never be read as observed truth.
    #[test]
    fn the_cut_arm_is_tagged_counterfactual_and_the_factual_arm_is_not() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        let chain: Vec<ChainStep> = (0..4)
            .map(|i| (0x90 + i as u8, edge_with(220, 210)))
            .collect();

        let cf = counterfactual_replay(
            &chain,
            0,
            edge_with(200, 200),
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR,
            },
        )
        .expect("in range")
        .expect("fits");

        let want = InferenceType::Counterfactual.to_mantissa();
        assert_eq!(want, -6, "the contract's road-not-taken nibble");
        assert_eq!(
            cf.counterfactual.last().unwrap().edge.inference_mantissa(),
            want,
        );
        assert_ne!(
            cf.factual.last().unwrap().edge.inference_mantissa(),
            want,
            "the FACTUAL arm must never carry the counterfactual tag",
        );
    }

    /// The two arms coexist, so their durable coordinates must not collide —
    /// the same `cast_seq` uniqueness rule W1 states, now with two traces
    /// alive at once.
    #[test]
    fn the_two_arms_never_share_a_durable_coordinate() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        let chain: Vec<ChainStep> = (0..6)
            .map(|i| (0x90 + i as u8, edge_with(220, 210)))
            .collect();

        let cf = counterfactual_replay(
            &chain,
            3,
            edge_with(200, 200),
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 1_000,
                bar: DEFAULT_FREQUENCY_BAR,
            },
        )
        .expect("in range")
        .expect("fits");

        let f: Vec<u64> = cf.factual.iter().map(ReplayTraceRow::cast_seq).collect();
        let c: Vec<u64> = cf
            .counterfactual
            .iter()
            .map(ReplayTraceRow::cast_seq)
            .collect();
        assert_eq!(f.len(), 6);
        assert_eq!(c.len(), 5);
        assert!(
            f.iter().all(|x| !c.contains(x)),
            "factual {f:?} and counterfactual {c:?} share a coordinate",
        );
        // Contiguous: the cut arm starts exactly where the factual one ended.
        assert_eq!(*f.last().unwrap() + 1, c[0]);
    }

    /// The verdict carries the cut edge's own relation flavour, so "the edge
    /// that EXPLAINS this is load-bearing" stays distinguishable from "an edge
    /// that merely RELATES TO it is". Read through the lenses, never
    /// re-derived.
    #[test]
    fn the_role_reports_the_cut_edges_own_topology_and_band_not_a_bare_boolean() {
        use causal_edge::layout::{CausalTopology, ReasoningBand};
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };

        // Two chains differing ONLY in the cut edge's lens bits.
        let plain = edge_with(250, 250);
        let flavoured = plain
            .with_topology(CausalTopology::IndirectKnownIntermediates)
            .with_reasoning_band(ReasoningBand::Causal);

        for (edge, want_topo, want_band) in [
            (plain, plain.topology(), plain.reasoning_band()),
            (
                flavoured,
                CausalTopology::IndirectKnownIntermediates,
                ReasoningBand::Causal,
            ),
        ] {
            let chain: Vec<ChainStep> = vec![
                (0x90, edge_with(250, 250)),
                (0x9D, edge),
                (0x92, edge_with(250, 250)),
            ];
            let cf = counterfactual_replay(
                &chain,
                1,
                edge_with(200, 200),
                CutContext {
                    tables: &tables,
                    compose: tabs,
                    owner: 3,
                    base_seq: 100,
                    bar: DEFAULT_FREQUENCY_BAR,
                },
            )
            .expect("in range")
            .expect("fits");
            assert_eq!(cf.role.topology, want_topo);
            assert_eq!(cf.role.band, want_band);
            assert_eq!(cf.role.predicate, 0x9D, "perturbs");
        }

        // Anti-vacuity: the two flavours must actually DIFFER, or the loop
        // above compares a value with itself twice.
        assert_ne!(plain.topology(), CausalTopology::IndirectKnownIntermediates);
        assert_ne!(plain.reasoning_band(), ReasoningBand::Causal);
    }

    /// An out-of-range cut is refused at the top level too, not just in
    /// `cut_step` — so a caller cannot receive a "not load-bearing" verdict
    /// for an edge that was never cut.
    #[test]
    fn an_out_of_range_cut_yields_no_verdict_at_all() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        let chain: Vec<ChainStep> = (0..3)
            .map(|i| (0x90 + i as u8, edge_with(220, 210)))
            .collect();
        assert!(counterfactual_replay(
            &chain,
            3,
            edge_with(200, 200),
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR
            }
        )
        .is_none());
        // ...while an in-range one does yield a verdict, so `None` means
        // "out of range" and not "this never works".
        assert!(counterfactual_replay(
            &chain,
            2,
            edge_with(200, 200),
            CutContext {
                tables: &tables,
                compose: tabs,
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR
            }
        )
        .is_some());
    }
    /// The bar is a real knob, not decoration: raising it must silence a
    /// verdict that fires, and lowering it must admit one that does not. The
    /// inertness test a threshold parameter owes — and the test that would
    /// have caught the confidence axis, where every chain sat at 170 and no
    /// bar could separate anything.
    #[test]
    fn the_bar_is_not_inert() {
        let (tables, t) = fixture();
        let tabs = ComposeTables {
            s: &t[0],
            p: &t[1],
            o: &t[2],
        };
        let chain: Vec<ChainStep> = vec![(0x91, edge_with(250, 250)), (0x92, edge_with(40, 30))];
        let trace = replay_chain(&chain, edge_with(200, 200), &tables, tabs, 3, 100)
            .expect("reservation fits");
        let f = trace.last().unwrap().edge.frequency_u8();
        assert_eq!(f, 133, "the fixture must sit where the probe measured it");

        assert_eq!(verdict_at(&trace, f), Verdict::Consistent, "at the bar");
        assert_eq!(
            verdict_at(&trace, f + 1),
            Verdict::Inconsistent,
            "raising silences"
        );
        assert_eq!(
            verdict_at(&trace, f - 1),
            Verdict::Consistent,
            "lowering admits"
        );

        // ...and the axis matters: the SAME trace's confidence is saturated,
        // so a confidence bar cannot discriminate. Pinned so nobody switches
        // the verdict back onto it.
        let c = trace.last().unwrap().edge.confidence_u8();
        assert_eq!(
            c, 170,
            "confidence saturates — measured across every fixture here"
        );

        // An empty replay asserts nothing and fails closed at any bar.
        assert_eq!(verdict_at(&[], 0), Verdict::Inconsistent);
        assert_eq!(verdict_at(&[], 255), Verdict::Inconsistent);
    }
}
