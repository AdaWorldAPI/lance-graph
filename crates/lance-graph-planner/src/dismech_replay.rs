//! **D-DCR-1 (W1) — the replay core.** `dismech-causal-replay-v1` §3 W1.
//!
//! Replays a RECORDED causal chain against a seed, one packed step at a time,
//! and emits a trace that `temporal.rs` can deinterlace. Replay, never
//! generation: the chain is an input, and the same inputs must produce a
//! byte-identical trace forever — that is the plan's keystone and this
//! module's first gate.
//!
//! # What this composes (and what it deliberately does not add)
//!
//! Every moving part is already shipped; W1 is the wiring that proves they
//! compose (`F-RLR-2`: proposing a new carrier before `ogar_loco` is shown
//! insufficient is an automatic STOP):
//!
//! | part | provenance |
//! |---|---|
//! | the step kernel | [`NarsTables::revise`] + [`CausalEdge64::forward`] — the kernel W0 measured at ~35 ns |
//! | the step *address* | a `u8` predicate ordinal — the dismech palette's `FnIndex` (`0x90..=0xA2`) |
//! | the trace | [`crate::temporal::LocalCausalRow`], deinterlaced by the shipped helpers |
//!
//! **The palette binds at the membrane, never here** (plan §2 constraint 6).
//! `ogar-loco`/`ogar-dismech` are NOT dependencies of this crate and must not
//! become ones: `lance-graph-planner` is in-workspace, while the OGAR surface
//! lives in the workspace-EXCLUDED armed tier (`lance-graph-ogar`). A step
//! therefore carries the ordinal as a plain `u8` on the hot path, and the
//! claim *"these ordinals ARE the dismech palette"* is pinned by a
//! conformance test in that armed tier — where the real
//! `ogar_dismech::CAUSAL_PREDICATES` is reachable — rather than asserted here.
//!
//! # The `cast_seq` precondition is load-bearing, not paperwork
//!
//! [`LocalCausalRow::cast_seq`] requires a key *unique and monotonic per owner
//! ACROSS process restarts*, and says in its own doc that a per-process
//! counter is **not** valid. [`replay_chain`] therefore takes a caller-supplied
//! `base_seq` (a durable-log position in production, e.g.
//! `persist_sink::LandedWitness`'s `DurableCoordinate::log_order`) and never
//! invents one. A replay that minted its own counter would produce traces that
//! collide across restarts — the exact failure the trait warns about.

use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use causal_edge::CausalEdge64;
use lance_graph_contract::collapse_gate::MailboxId;

use crate::temporal::LocalCausalRow;

/// The three 256×256 palette-composition tables [`CausalEdge64::forward`]
/// needs, borrowed together so a caller cannot pass two of one chain's tables
/// and one of another's. Borrowed, never owned: these are large and shared.
#[derive(Clone, Copy)]
pub struct ComposeTables<'a> {
    /// Subject-plane composition.
    pub s: &'a [u8; 256 * 256],
    /// Predicate-plane composition.
    pub p: &'a [u8; 256 * 256],
    /// Object-plane composition.
    pub o: &'a [u8; 256 * 256],
}

/// One row of a replay trace — the witness of a single step.
///
/// Implements [`LocalCausalRow`] so the shipped deinterlace helpers read it;
/// it holds no state the step did not produce, and nothing is recomputed from
/// it at read time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayTraceRow {
    owner: MailboxId,
    cast_seq: u64,
    /// Index of the step within its chain — the address a perturbation is
    /// reported AT, so a diff names a position rather than a row.
    pub step: u32,
    /// The packed edge this step produced.
    pub edge: CausalEdge64,
}

impl ReplayTraceRow {
    /// The owning mailbox.
    #[must_use]
    pub fn owner(&self) -> MailboxId {
        self.owner
    }
    /// The durable ordering key this row was emitted under.
    #[must_use]
    pub fn cast_seq(&self) -> u64 {
        self.cast_seq
    }
}

impl LocalCausalRow for ReplayTraceRow {
    fn owner(&self) -> MailboxId {
        self.owner
    }
    fn cast_seq(&self) -> u64 {
        self.cast_seq
    }
}

/// One recorded step: the predicate ordinal it travels under, and the packed
/// weight edge the chain recorded for it.
///
/// A pair, not a carrier: this is the loco `(function : value)` shape the
/// substrate already reads every 12-byte payload as — `function` is the
/// palette ordinal, `value` is the weight.
pub type ChainStep = (u8, CausalEdge64);

/// The single step, isolated so the replay and any future accelerator agree on
/// what a step IS. Composes the two halves W0 measured, in that order:
/// the table lookup (evidence fusion) then the packed forward (palette
/// composition + truth propagation).
#[inline]
#[must_use]
pub fn replay_step(
    running: CausalEdge64,
    weight: CausalEdge64,
    tables: &NarsTables,
    compose: ComposeTables<'_>,
) -> CausalEdge64 {
    // 1. fuse this step's evidence into the running truth (the lookup half)
    let revised = tables.revise(
        running.frequency_u8(),
        running.confidence_u8(),
        weight.frequency_u8(),
        weight.confidence_u8(),
    );
    // 2. propagate palettes + causality (the packed half)
    let mut out = running.forward(weight, compose.s, compose.p, compose.o);
    // 3. the revised truth is the step's truth — written back into the packed
    //    edge, not carried beside it (there is no second truth register).
    out.set_frequency_u8(unpack_f(revised));
    out.set_confidence_u8(unpack_c(revised));
    out
}

/// Replay a recorded chain against `seed`, emitting one trace row per step.
///
/// `base_seq` is the caller's DURABLE ordering position (see the module doc):
/// row *i* is stamped `base_seq + i`, so a chain replayed from the same log
/// position is byte-identical, and two chains from different positions never
/// collide. Nothing here mints a counter.
///
/// The predicate ordinal rides each step but does not alter the arithmetic in
/// W1 — it is the ADDRESS the step travels under, carried into the trace's
/// step index so a consumer can join a diff back to the palette at the
/// membrane. W3's counterfactual arm is what makes it selective.
#[must_use]
pub fn replay_chain(
    chain: &[ChainStep],
    seed: CausalEdge64,
    tables: &NarsTables,
    compose: ComposeTables<'_>,
    owner: MailboxId,
    base_seq: u64,
) -> Vec<ReplayTraceRow> {
    let mut running = seed;
    let mut trace = Vec::with_capacity(chain.len());
    for (i, &(_predicate, weight)) in chain.iter().enumerate() {
        running = replay_step(running, weight, tables, compose);
        trace.push(ReplayTraceRow {
            owner,
            cast_seq: base_seq + i as u64,
            step: u32::try_from(i).unwrap_or(u32::MAX),
            edge: running,
        });
    }
    trace
}

/// The first step index at which two traces differ, or `None` when they are
/// byte-identical. This is the determinism gate's instrument AND the
/// perturbation gate's: a perturbation must be reported at the step it was
/// applied to, never earlier.
#[must_use]
pub fn first_divergence(a: &[ReplayTraceRow], b: &[ReplayTraceRow]) -> Option<u32> {
    a.iter()
        .zip(b.iter())
        .find(|(x, y)| x.edge != y.edge)
        .map(|(x, _)| x.step)
        .or_else(|| {
            if a.len() == b.len() {
                None
            } else {
                Some(a.len().min(b.len()) as u32)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::edge::InferenceType;
    use causal_edge::{CausalMask, PlasticityState};

    /// Deterministic fixture PRNG — a replay test that seeded from the clock
    /// could not test replay.
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

    fn compose_tables() -> Box<[[u8; 256 * 256]; 3]> {
        let mut t = Box::new([[0u8; 256 * 256]; 3]);
        for (k, tab) in t.iter_mut().enumerate() {
            for i in 0..256usize {
                for j in 0..256usize {
                    tab[i * 256 + j] = ((i * 7 + j * 13 + k * 31) % 256) as u8;
                }
            }
        }
        t
    }

    fn edge(rng: &mut Lcg) -> CausalEdge64 {
        CausalEdge64::pack(
            rng.below(256) as u8,
            rng.below(256) as u8,
            rng.below(256) as u8,
            (100 + rng.below(150)) as u8,
            (100 + rng.below(120)) as u8,
            CausalMask::PO,
            0b101,
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0,
        )
    }

    /// A chain whose predicate ordinals sit in the dismech palette's band.
    /// The BAND is asserted here; that these exact ordinals are the palette's
    /// is the armed tier's conformance test, not this crate's claim.
    fn chain(rng: &mut Lcg, len: usize) -> Vec<ChainStep> {
        (0..len)
            .map(|_| ((0x90 + rng.below(0x13)) as u8, edge(rng)))
            .collect()
    }

    /// GATE (a) — determinism. Same inputs, byte-identical trace. This is the
    /// keystone: if it can fail, nothing above it means anything.
    #[test]
    fn the_same_chain_replays_byte_identically() {
        let t = NarsTables::build(1);
        let c = compose_tables();
        let tabs = ComposeTables {
            s: &c[0],
            p: &c[1],
            o: &c[2],
        };
        let mut rng = Lcg(0x0BAD_5EED_1234_5678);
        let ch = chain(&mut rng, 12);
        let seed = edge(&mut rng);

        let a = replay_chain(&ch, seed, &t, tabs, 7, 1_000);
        let b = replay_chain(&ch, seed, &t, tabs, 7, 1_000);
        assert_eq!(a, b, "replay is not deterministic");
        assert_eq!(first_divergence(&a, &b), None);
        // Anti-vacuity, and it must cover BOTH halves of the kernel — a
        // determinism gate is trivially satisfied by a kernel that does
        // nothing. Measured while writing this: stubbing `forward` to a no-op
        // still moved the edge, because the truth write-back alone changes it,
        // so "the edge differs from the seed" was too weak a check. Assert the
        // palette moved (forward ran) AND the truth moved (revise ran).
        assert_eq!(a.len(), 12);
        assert_ne!(a[11].edge, seed, "the chain must transform the seed");
        let moved_palette = a[11].edge.s_idx() != seed.s_idx()
            || a[11].edge.p_idx() != seed.p_idx()
            || a[11].edge.o_idx() != seed.o_idx();
        assert!(moved_palette, "the forward (palette) half did not run");
        assert!(
            a[11].edge.frequency_u8() != seed.frequency_u8()
                || a[11].edge.confidence_u8() != seed.confidence_u8(),
            "the revise (truth) half did not run"
        );
    }

    /// GATE (b) — can-fire, AT THE RIGHT INDEX. Perturbing step k must change
    /// the trace at k, never before it. "Differs somewhere" is not the claim.
    #[test]
    fn a_perturbed_step_diverges_at_exactly_that_step() {
        let t = NarsTables::build(1);
        let c = compose_tables();
        let tabs = ComposeTables {
            s: &c[0],
            p: &c[1],
            o: &c[2],
        };
        let mut rng = Lcg(0x0BAD_5EED_1234_5678);
        let ch = chain(&mut rng, 12);
        let seed = edge(&mut rng);
        let base = replay_chain(&ch, seed, &t, tabs, 7, 1_000);

        for k in [0usize, 5, 11] {
            let mut perturbed = ch.clone();
            // flip the weight's subject palette index — a real edge change
            let mut w = perturbed[k].1;
            w.set_s_idx(w.s_idx().wrapping_add(1));
            perturbed[k].1 = w;

            let other = replay_chain(&perturbed, seed, &t, tabs, 7, 1_000);
            assert_eq!(
                first_divergence(&base, &other),
                Some(k as u32),
                "perturbing step {k} must first diverge AT step {k}"
            );
            // …and everything before k is untouched.
            assert_eq!(base[..k], other[..k]);
        }
    }

    /// GATE (c) — silence. A perturbation to a chain that is NOT replayed
    /// changes nothing. Without this, gate (b) would pass for a replay that
    /// simply hashes its whole input universe.
    #[test]
    fn perturbing_a_sibling_chain_changes_nothing() {
        let t = NarsTables::build(1);
        let c = compose_tables();
        let tabs = ComposeTables {
            s: &c[0],
            p: &c[1],
            o: &c[2],
        };
        let mut rng = Lcg(0x0BAD_5EED_1234_5678);
        let ch = chain(&mut rng, 12);
        let mut sibling = chain(&mut rng, 12);
        let seed = edge(&mut rng);

        let base = replay_chain(&ch, seed, &t, tabs, 7, 1_000);
        let mut w = sibling[3].1;
        w.set_s_idx(w.s_idx().wrapping_add(17));
        sibling[3].1 = w;
        let after = replay_chain(&ch, seed, &t, tabs, 7, 1_000);

        assert_eq!(base, after, "a sibling chain must not reach this replay");
        // anti-vacuity: the sibling really is a different chain, and replaying
        // IT really does differ — otherwise "unchanged" is trivially true.
        let sib_trace = replay_chain(&sibling, seed, &t, tabs, 7, 1_000);
        assert_ne!(base, sib_trace);
    }

    /// The trace is not decorative: it survives the SHIPPED deinterlace, and
    /// two owners' interleaved rows split back into their own trajectories in
    /// order. If this fails, `temporal.rs` integration is a doc claim.
    #[test]
    fn the_trace_deinterlaces_into_per_owner_trajectories() {
        let t = NarsTables::build(1);
        let c = compose_tables();
        let tabs = ComposeTables {
            s: &c[0],
            p: &c[1],
            o: &c[2],
        };
        let mut rng = Lcg(0x0BAD_5EED_1234_5678);
        let (ch_a, ch_b) = (chain(&mut rng, 4), chain(&mut rng, 4));
        let seed = edge(&mut rng);

        let a = replay_chain(&ch_a, seed, &t, tabs, 1, 10);
        let b = replay_chain(&ch_b, seed, &t, tabs, 2, 500);
        // interleave them the way a shared durable log would
        let mut global = Vec::new();
        for i in 0..4 {
            global.push(b[i]);
            global.push(a[i]);
        }

        let only_a = crate::temporal::local_trajectory_of(&global, 1);
        assert_eq!(only_a, a, "owner 1's local trajectory is its own replay");
        let only_b = crate::temporal::local_trajectory_of(&global, 2);
        assert_eq!(only_b, b);
        // cast_seq is monotonic per owner — the precondition the trait names.
        assert!(only_a.windows(2).all(|w| w[0].cast_seq() < w[1].cast_seq()));
    }
}
