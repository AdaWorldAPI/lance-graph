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
use core::fmt;

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
    /// The DisMech predicate ordinal this step travelled under.
    ///
    /// **Carried as witness, not as an operand.** W1's arithmetic never reads
    /// it — but a trace that dropped it would not be a witness of the recorded
    /// program: two chains with identical weights and different relations
    /// (`causes` vs `protects_against`) would replay to byte-identical traces,
    /// and no consumer could reconstruct or validate what was actually
    /// recorded. Resolve it through [`chain_step_predicate`] (or, at the
    /// membrane, the real palette).
    ///
    /// Added after review on #1120: the module previously claimed the ordinal
    /// was "carried into the trace's step index" — it was not carried at all,
    /// and the step index is `i`. That was a doc claim no behaviour backed.
    pub predicate: u8,
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
///
/// The ordinal's DOMAIN is checkable without leaving the workspace:
/// [`chain_step_predicate`] resolves it through the contract's zero-dep
/// palette mirror, and the armed tier fuses that mirror against the real
/// `ogar_dismech::RELATIONS`.
pub type ChainStep = (u8, CausalEdge64);

/// Resolve a step's predicate ordinal to `(ordinal, name, curie)`, or `None`
/// when the byte names no minted DisMech predicate.
///
/// **Why this exists rather than a bare `u8`.** The replay arithmetic does not
/// read the ordinal in W1 — it is the ADDRESS the step travels under, not an
/// operand — so nothing in the hot path would ever notice a corrupt one. That
/// is exactly why the domain needs a name: a chain carrying `0xA3` is not a
/// causal chain at all (`0xA3` is the palette's SEARCH band), and without this
/// a replay would trace it to a byte-identical, entirely meaningless result.
///
/// The mirror is the contract's; the fuse that proves it IS the palette lives
/// in `lance_graph_ogar::parity::assert_dismech_palette_parity`, because
/// `ogar-dismech` is reachable only from that workspace-EXCLUDED armed tier.
/// Mirror here, authority there — the same split `ogar_codebook` already uses.
#[must_use]
pub fn chain_step_predicate(step: ChainStep) -> Option<&'static (u8, &'static str, &'static str)> {
    lance_graph_contract::dismech_evidence::dismech_predicate(step.0)
}

/// A replay could not be performed as asked.
///
/// Deliberately small: replay has exactly one way to fail, and it is a
/// property of the ADDRESS SPACE the caller offered, never of the recorded
/// chain (see [`validate_chain`] for why a chain's content is not judged here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayError {
    /// `[base_seq, base_seq + chain.len())` does not fit in `u64`.
    ///
    /// Not a theoretical guard: the alternative is wrapping, and wrapped
    /// coordinates alias the START of the durable log — duplicate `cast_seq`
    /// values that violate [`LocalCausalRow`]'s uniqueness requirement and
    /// silently degrade `local_trajectory_of` to scan order. Refused loudly
    /// rather than emitted quietly. (Raised by review on #1120: the previous
    /// `base_seq + i as u64` panicked in debug builds and wrapped in release.)
    SequenceExhausted {
        /// The base the caller offered.
        base_seq: u64,
        /// How many steps the chain needed.
        steps: usize,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceExhausted { base_seq, steps } => write!(
                f,
                "durable sequence exhausted: base {base_seq} cannot reserve {steps} steps"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

/// A step whose predicate ordinal names no minted DisMech predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmintedOrdinal {
    /// Index of the offending step within the chain.
    pub step: usize,
    /// The ordinal that resolved to nothing.
    pub ordinal: u8,
}

/// Check that every step of a chain travels under a minted DisMech predicate.
///
/// # Why this is an ADMISSION check and NOT part of [`replay_chain`]
///
/// Review on #1120 proposed validating inside the replay loop and returning an
/// error before emitting a trace row. The defect it names is real — a chain
/// carrying `0xA3` (the palette's SEARCH band) is not a causal chain, and
/// replaying it produces a byte-identical, meaningless trace. But the remedy
/// belongs one step earlier, for a reason that goes to the plan's keystone:
///
/// **Replay must not refuse history.** A recorded chain is a fact about what
/// was evaluated; the engine's job is to reproduce it, not to judge it. If the
/// palette ever drops or renumbers an ordinal, a replay that validated would
/// start returning `Err` for chains that were perfectly valid when recorded —
/// and "yesterday's evaluation replays today byte-for-byte" is precisely the
/// property the whole wave exists to hold.
///
/// So the judgement happens once, when a chain is FIRST accepted, and replay
/// stays total over everything already admitted.
///
/// # "Admission" means first acceptance — NOT loading a recording
///
/// The previous wording listed *"loading a recording"* as an admission site,
/// which **contradicted the paragraph above it** — reloading an older durable
/// recording and validating it against today's palette rejects exactly the
/// history the split exists to keep replayable. Codex caught this on #1122;
/// the argument was right and the instruction beneath it was wrong.
///
/// The rule, stated so the two cannot drift apart again:
///
/// - **A chain arriving from outside** (a producer, a boundary, a new
///   recording being made) is validated against the CURRENT palette, here.
/// - **A chain being re-read from the durable log** is already admitted. It is
///   not re-validated — the fact that it was recorded IS its admission, under
///   whatever palette was current then. Replay it.
/// - A caller that genuinely needs to check an old recording must check it
///   against the palette version it was admitted under, which this function
///   cannot do: it has one palette, today's. That is a versioned-palette
///   capability nothing in this wave has, and inventing one here would be
///   worse than declining.
///
/// Fails closed and reports WHICH step, so a rejection is actionable rather
/// than a boolean.
pub fn validate_chain(chain: &[ChainStep]) -> Result<(), UnmintedOrdinal> {
    for (step, &entry) in chain.iter().enumerate() {
        if chain_step_predicate(entry).is_none() {
            return Err(UnmintedOrdinal {
                step,
                ordinal: entry.0,
            });
        }
    }
    Ok(())
}

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
/// # `base_seq` RESERVES a half-open range, it is not a chain counter
///
/// This call stamps `[base_seq, base_seq + chain.len())` — one durable
/// coordinate per STEP, not per chain. A caller that advances by 1 per chain
/// therefore overlaps: bases 10 and 11 over 4-step chains produce
/// `[10,11,12,13]` and `[11,12,13,14]`, and the shared coordinates violate
/// [`LocalCausalRow::cast_seq`]'s uniqueness requirement — after which
/// `local_trajectory_of` preserves SCAN order rather than durable causal
/// order, so a crash replay varies with how rows happened to interleave.
///
/// Advance with [`next_base_seq`], which is the reservation stated as code.
/// The overlap cannot be detected inside one call (it is a property ACROSS
/// calls), so this is a precondition, not a check — said plainly rather than
/// implied. `overlapping_bases_collide_which_is_why_next_base_seq_exists`
/// pins both directions. (Raised in review on #1120; the prior doc said two
/// chains "never collide", which was true only for a caller already
/// following the rule it did not state.)
///
/// The predicate ordinal rides each step into [`ReplayTraceRow::predicate`].
/// It does not alter W1's arithmetic — it is the ADDRESS the step travels
/// under — but it IS part of the witness, so a consumer can join a diff back
/// to the palette at the membrane. W3's counterfactual arm makes it selective.
///
/// # Errors
///
/// [`ReplayError::SequenceExhausted`] when `[base_seq, base_seq + chain.len())`
/// does not fit in `u64`. The reservation is checked ONCE, up front, so the
/// loop cannot emit a partial trace and then discover it has no coordinate
/// left — a half-written trace is worse than a refusal.
pub fn replay_chain(
    chain: &[ChainStep],
    seed: CausalEdge64,
    tables: &NarsTables,
    compose: ComposeTables<'_>,
    owner: MailboxId,
    base_seq: u64,
) -> Result<Vec<ReplayTraceRow>, ReplayError> {
    // Check the whole reservation before writing anything.
    let steps = chain.len();
    if steps > 0 {
        u64::try_from(steps - 1)
            .ok()
            .and_then(|last| base_seq.checked_add(last))
            .ok_or(ReplayError::SequenceExhausted { base_seq, steps })?;
    }

    let mut running = seed;
    let mut trace = Vec::with_capacity(steps);
    for (i, &(predicate, weight)) in chain.iter().enumerate() {
        running = replay_step(running, weight, tables, compose);
        trace.push(ReplayTraceRow {
            owner,
            cast_seq: base_seq + i as u64,
            step: u32::try_from(i).unwrap_or(u32::MAX),
            predicate,
            edge: running,
        });
    }
    Ok(trace)
}

/// The next FREE durable coordinate after replaying a `chain_len`-step chain
/// from `base_seq` — the half-open reservation `[base_seq, base_seq + len)`
/// stated as code, so a caller replaying several chains in sequence cannot
/// reach for a naive `+ 1`.
///
/// Returns `None` when the range is exhausted, i.e. when there IS no free
/// coordinate after this chain.
///
/// # Why this is an `Option` and not a saturating `u64`
///
/// It was a saturating `u64`, and that was a real bug — found independently by
/// both reviewers on #1122, in code written to fix their own earlier finding
/// about this same field.
///
/// Saturation collapses "exhausted" into a *usable-looking* answer. Replaying
/// 4 steps from `u64::MAX - 3` mints through `u64::MAX`; the saturating helper
/// then returned `u64::MAX` — a coordinate it had just handed out — and a
/// caller following the documented sequential pattern would replay a one-step
/// chain there (legal, since the reservation only needs `steps - 1` to be
/// addable) and emit a **duplicate `cast_seq`**. That is precisely the
/// duplicate the old doc comment claimed saturation prevented.
///
/// Worse than the bug: the test asserted `next_base_seq(u64::MAX, 4) ==
/// u64::MAX` and called it "the saturating guard", pinning the defect as
/// intended behaviour. A guard that cannot say "no" is not a guard — the same
/// lesson as `E-A-DETERMINISM-GATE-IS-TRIVIALLY-SATISFIED-BY-A-KERNEL-THAT-
/// DOES-NOTHING-1`, met again one PR later while fixing a review comment.
///
/// Exhaustion is now representable, so a caller must handle it rather than
/// receive a coordinate that is already spoken for.
#[must_use]
pub const fn next_base_seq(base_seq: u64, chain_len: usize) -> Option<u64> {
    // `base + len` is the first coordinate PAST the half-open reservation, so
    // it is exactly the next free base — and it overflows precisely when the
    // reservation ran to the end of the range (4 steps from `u64::MAX - 3`
    // mint through `u64::MAX`, leaving nothing). `checked_add` therefore IS
    // the boundary; no separate exhaustion test is needed. A zero-length
    // chain reserves nothing and returns its own base.
    base_seq.checked_add(chain_len as u64)
}

/// The first step whose replayed CONTENT differs, or `None` when the two
/// traces carry the same content throughout.
///
/// # What "content" means here, and why it is not the whole row
///
/// Content is `(predicate, edge)` — the relation the step travelled under and
/// the edge it produced. **Addressing (`owner`, `cast_seq`) is deliberately
/// NOT compared**, and the narrowing is the point rather than a shortcut: the
/// same chain replayed by the same owner from two different durable positions
/// is the SAME causal witness at two addresses, and an instrument that called
/// those "divergent" would report a difference at step 0 for every comparison
/// the plan actually needs. `step` is excluded for the same reason — it is
/// positional, and position is what the return value NAMES.
///
/// This is the determinism gate's instrument AND the perturbation gate's: a
/// perturbation must be reported at the step it was applied to, never earlier.
///
/// (Contract narrowed explicitly after review on #1120, which correctly
/// observed that the previous wording — "where the traces differ" — promised
/// a whole-row comparison the body did not perform. Fixed by naming the
/// contract, not by widening the comparison: see
/// `addressing_is_not_content_but_the_predicate_is`.)
#[must_use]
pub fn first_divergence(a: &[ReplayTraceRow], b: &[ReplayTraceRow]) -> Option<u32> {
    a.iter()
        .zip(b.iter())
        .find(|(x, y)| (x.predicate, x.edge) != (y.predicate, y.edge))
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

        let a = replay_chain(&ch, seed, &t, tabs, 7, 1_000).expect("reservation fits");
        let b = replay_chain(&ch, seed, &t, tabs, 7, 1_000).expect("reservation fits");
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
        let base = replay_chain(&ch, seed, &t, tabs, 7, 1_000).expect("reservation fits");

        for k in [0usize, 5, 11] {
            let mut perturbed = ch.clone();
            // flip the weight's subject palette index — a real edge change
            let mut w = perturbed[k].1;
            w.set_s_idx(w.s_idx().wrapping_add(1));
            perturbed[k].1 = w;

            let other =
                replay_chain(&perturbed, seed, &t, tabs, 7, 1_000).expect("reservation fits");
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

        let base = replay_chain(&ch, seed, &t, tabs, 7, 1_000).expect("reservation fits");
        let mut w = sibling[3].1;
        w.set_s_idx(w.s_idx().wrapping_add(17));
        sibling[3].1 = w;
        let after = replay_chain(&ch, seed, &t, tabs, 7, 1_000).expect("reservation fits");

        assert_eq!(base, after, "a sibling chain must not reach this replay");
        // anti-vacuity: the sibling really is a different chain, and replaying
        // IT really does differ — otherwise "unchanged" is trivially true.
        let sib_trace = replay_chain(&sibling, seed, &t, tabs, 7, 1_000).expect("reservation fits");
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

        let a = replay_chain(&ch_a, seed, &t, tabs, 1, 10).expect("reservation fits");
        let b = replay_chain(&ch_b, seed, &t, tabs, 2, 500).expect("reservation fits");
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
    #[test]
    fn a_chain_steps_ordinal_names_a_minted_predicate_and_a_search_op_does_not() {
        // The core cannot reach `ogar_dismech` (workspace boundary), so what
        // it CAN pin is that a step's ordinal resolves through the contract
        // mirror to the predicate the plan names — and that the byte one past
        // the band does not. The mirror-IS-the-palette half is fused in
        // `lance_graph_ogar::parity::assert_dismech_palette_parity`; neither
        // half alone is the claim.
        let mut rng = Lcg(0x51D3_7C4A_9B2E_6F08);
        let w = edge(&mut rng);
        assert_eq!(chain_step_predicate((0x90, w)).map(|p| p.1), Some("causes"),);
        assert_eq!(
            chain_step_predicate((0xA2, w)).map(|p| p.1),
            Some("variant_of"),
        );
        // Can-stay-silent, on a byte that is a REAL slot elsewhere in the
        // palette rather than an arbitrary one: 0xA3 is `CANDIDATES`, the
        // first search op. An empty-input silence case would prove nothing.
        assert!(chain_step_predicate((0xA3, w)).is_none());
        assert!(chain_step_predicate((0x8F, w)).is_none());
    }
    #[test]
    fn two_chains_differing_only_in_predicate_do_not_replay_identically() {
        // The P1 finding from #1120, pinned. Identical weights, one relation
        // swapped: `causes` (0x90) vs `protects_against` (0x95) — two
        // predicates whose real-world meanings are near opposites, so a trace
        // that cannot tell them apart is not a witness of anything.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let mut rng = Lcg(0x0C0D_E4A1_7B39_5E62);
        let seed = edge(&mut rng);
        let steps: Vec<CausalEdge64> = (0..6).map(|_| edge(&mut rng)).collect();

        let a: Vec<ChainStep> = steps.iter().map(|&w| (0x90u8, w)).collect();
        let mut b = a.clone();
        b[3].0 = 0x95;

        // Anti-vacuity: the two chains must be identical in EVERY other
        // respect, or this test would pass for the wrong reason.
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.1, y.1,
                "weights must match — only the predicate may differ"
            );
        }
        assert_ne!(a[3].0, b[3].0);

        let ta = replay_chain(&a, seed, &tables, tabs, 7, 100).expect("reservation fits");
        let tb = replay_chain(&b, seed, &tables, tabs, 7, 100).expect("reservation fits");

        // The packed arithmetic is genuinely identical — W1 does not read the
        // ordinal — so the ONLY thing that can distinguish these traces is the
        // predicate being carried as witness.
        for (x, y) in ta.iter().zip(tb.iter()) {
            assert_eq!(
                x.edge, y.edge,
                "W1 arithmetic must not depend on the ordinal"
            );
        }
        assert_eq!(
            first_divergence(&ta, &tb),
            Some(3),
            "the swapped relation must be visible in the witness, at its own step",
        );
    }

    #[test]
    fn addressing_is_not_content_but_the_predicate_is() {
        // The P2 `first_divergence` finding, pinned two-sided.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let mut rng = Lcg(0x51CE_2D07_A4B8_1193);
        let seed = edge(&mut rng);
        let c = chain(&mut rng, 5);

        // (a) SILENT on pure addressing: same chain, same owner, different
        //     durable base — the same causal witness at another address.
        let at_100 = replay_chain(&c, seed, &tables, tabs, 1, 100).expect("reservation fits");
        let at_900 = replay_chain(&c, seed, &tables, tabs, 1, 900).expect("reservation fits");
        assert_ne!(
            at_100[0].cast_seq, at_900[0].cast_seq,
            "the addresses must actually differ, or this proves nothing",
        );
        assert_eq!(first_divergence(&at_100, &at_900), None);

        // (b) ...and a different OWNER is addressing too.
        let other = replay_chain(&c, seed, &tables, tabs, 2, 100).expect("reservation fits");
        assert_ne!(at_100[0].owner(), other[0].owner());
        assert_eq!(first_divergence(&at_100, &other), None);

        // (c) FIRES on content: one predicate swapped, addressing untouched.
        let mut c2 = c.clone();
        c2[2].0 = if c2[2].0 == 0x90 { 0x91 } else { 0x90 };
        let swapped = replay_chain(&c2, seed, &tables, tabs, 1, 100).expect("reservation fits");
        assert_eq!(first_divergence(&at_100, &swapped), Some(2));
    }

    #[test]
    fn overlapping_bases_collide_which_is_why_next_base_seq_exists() {
        // The P2 durable-coordinate finding, pinned as the precondition it is.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let mut rng = Lcg(0xD0A7_31FC_5E28_44B9);
        let seed = edge(&mut rng);
        let c = chain(&mut rng, 4);
        let owner = 3;

        let coords =
            |t: &[ReplayTraceRow]| -> Vec<u64> { t.iter().map(|r| r.cast_seq()).collect() };

        // WRONG: advance by one per chain. The reservation is per STEP, so the
        // ranges overlap on 3 of 4 coordinates.
        let naive_a = replay_chain(&c, seed, &tables, tabs, owner, 10).expect("reservation fits");
        let naive_b = replay_chain(&c, seed, &tables, tabs, owner, 11).expect("reservation fits");
        let (ca, cb) = (coords(&naive_a), coords(&naive_b));
        let shared = ca.iter().filter(|x| cb.contains(x)).count();
        assert_eq!(
            shared, 3,
            "a naive +1 advance must demonstrably collide, or the precondition is decoration",
        );

        // RIGHT: advance by the reservation.
        let ok_a = replay_chain(&c, seed, &tables, tabs, owner, 10).expect("reservation fits");
        let next = next_base_seq(10, c.len()).expect("10 + 4 is addressable");
        assert_eq!(next, 14);
        let ok_b = replay_chain(&c, seed, &tables, tabs, owner, next).expect("reservation fits");
        let (oa, ob) = (coords(&ok_a), coords(&ok_b));
        assert!(
            oa.iter().all(|x| !ob.contains(x)),
            "correctly advanced ranges must be disjoint",
        );
        // ...and contiguous, so the log has no unexplained holes.
        assert_eq!(*oa.last().unwrap() + 1, ob[0]);

        // Exhaustion must be REPRESENTABLE, not collapsed into a usable base.
        //
        // This assertion previously read `next_base_seq(u64::MAX, 4) ==
        // u64::MAX` and called it "the saturating guard" — pinning a defect as
        // intended behaviour. Both reviewers on #1122 found it independently:
        // saturation hands back a coordinate the reservation ALREADY minted,
        // so a caller following the documented sequential pattern emits a
        // duplicate `cast_seq` — exactly what the guard claimed to prevent.
        assert_eq!(next_base_seq(u64::MAX, 4), None);

        // The boundary that makes it concrete, and the reason `None` is not
        // merely defensive: replay 4 steps from the last base that fits, and
        // the range is genuinely used up.
        let top = u64::MAX - 3;
        let last =
            replay_chain(&c, seed, &tables, tabs, owner, top).expect("the exact fit is allowed");
        assert_eq!(last.last().unwrap().cast_seq(), u64::MAX);
        assert_eq!(
            next_base_seq(top, c.len()),
            None,
            "there is no free coordinate after a reservation that ends at u64::MAX",
        );
        // ...and the old saturating answer would have been u64::MAX, which
        // `replay_chain` still ACCEPTS for a one-step chain — so the duplicate
        // was reachable, not theoretical.
        let dup = replay_chain(&c[..1], seed, &tables, tabs, owner, u64::MAX)
            .expect("a one-step chain at the top is legal");
        assert_eq!(
            dup[0].cast_seq(),
            last.last().unwrap().cast_seq(),
            "this is the duplicate the old saturating helper would have handed out",
        );
    }
    #[test]
    fn a_chain_carrying_a_search_op_is_refused_at_admission_and_still_replays() {
        // CodeRabbit #1120 asked that `replay_chain` itself reject an ordinal
        // outside the predicate band. Both halves are pinned here, because the
        // SPLIT is the decision, not either half alone.
        let mut rng = Lcg(0x5EA2_C401_9D3B_77E6);
        let mut c = chain(&mut rng, 5);
        c[2].0 = 0xA3; // CANDIDATES — a real slot, in the SEARCH band

        // (a) admission REFUSES it, and names the step so the rejection is
        //     actionable rather than a boolean.
        assert_eq!(
            validate_chain(&c),
            Err(UnmintedOrdinal {
                step: 2,
                ordinal: 0xA3
            }),
        );
        // Anti-vacuity: the same chain without that step must PASS, or the
        // check could be rejecting everything.
        let mut clean = c.clone();
        clean[2].0 = 0x90;
        assert_eq!(validate_chain(&clean), Ok(()));

        // (b) replay stays TOTAL over it. Replay must not refuse history: a
        //     recorded chain is a fact, and a replay that judged content would
        //     start returning Err for chains that were valid when recorded —
        //     against the keystone this whole wave exists to hold.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let seed = edge(&mut rng);
        let t = replay_chain(&c, seed, &tables, tabs, 1, 10).expect("reservation fits");
        assert_eq!(t.len(), 5);
        assert_eq!(
            t[2].predicate, 0xA3,
            "the witness records what was replayed"
        );
    }

    #[test]
    fn a_reservation_that_cannot_fit_is_refused_before_a_partial_trace_exists() {
        // CodeRabbit #1120: `base_seq + i` panicked in debug and wrapped in
        // release. Wrapped coordinates alias the start of the durable log.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let mut rng = Lcg(0x77C1_0E5B_2A94_3D18);
        let c = chain(&mut rng, 4);
        let seed = edge(&mut rng);

        assert_eq!(
            replay_chain(&c, seed, &tables, tabs, 1, u64::MAX).unwrap_err(),
            ReplayError::SequenceExhausted {
                base_seq: u64::MAX,
                steps: 4
            },
        );

        // Can-stay-silent, and NOT on a trivially small base: the last base
        // that exactly fits a 4-step chain must still succeed, so the check
        // rejects only what genuinely overflows rather than anything large.
        let exact = u64::MAX - 3;
        let t =
            replay_chain(&c, seed, &tables, tabs, 1, exact).expect("the exact fit must be allowed");
        assert_eq!(t.last().unwrap().cast_seq(), u64::MAX);

        // A one-step chain at the very top fits: the reservation is half-open,
        // so `steps - 1` is what has to be addable, not `steps`.
        let one = &c[..1];
        assert!(replay_chain(one, seed, &tables, tabs, 1, u64::MAX).is_ok());

        // An empty chain reserves nothing and cannot exhaust anything.
        assert!(replay_chain(&[], seed, &tables, tabs, 1, u64::MAX)
            .expect("empty reserves nothing")
            .is_empty());
    }
}
