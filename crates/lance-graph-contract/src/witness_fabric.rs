// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `witness_fabric` — make the A9 witness tenant REAL: compute the social loci
//! (quorum / contradiction) from a WINDOW of real peer rows instead of
//! hand-setting them, follow loci chains with a hop budget, and recognize a
//! persisted contradiction as an opinion. Thinking as agreement topology
//! (`E-THINKING-EXPANSION-QUEUE-1` Tier 3).
//!
//! These are FUNCTIONS over a slice of `(position, CausalWitnessFacet)` rows —
//! never a materialized `W×W` fabric struct (AGI-as-SoA: methods over the
//! existing carrier, not a new stored layer).
//!
//! # Loci converge on the SAME EVENT, not the same offset
//!
//! A locus offset is relative to its OWN row's stream position. Two rows agree
//! about a dimension iff they point at the **same absolute event**: row A at
//! `pos_a` with offset `o_a` and row B at `pos_b` with `o_b` converge iff
//! `pos_a + o_a == pos_b + o_b` (plan §2.9: "converge on the SAME context
//! events"). [`CausalWitnessFacet::agrees_at`](crate::causal_witness) compares
//! bare offsets (co-located rows); the fabric compares absolute targets.

use crate::causal_witness::{CausalWitnessFacet, Locus};

/// The named loci that carry CONTENT agreement — everything except the two
/// social loci (Quorum / Contradiction), which the fabric is COMPUTING and so
/// must not read as input (no self-reference).
const CONTENT_LOCI: [Locus; 14] = [
    Locus::Temporal,
    Locus::Kausal,
    Locus::Modal,
    Locus::Lokal,
    Locus::SMeaning,
    Locus::PMeaning,
    Locus::OMeaning,
    Locus::Antecedent,
    Locus::BasinAnchor,
    Locus::SupportedBy,
    Locus::Supports,
    Locus::RunbookEvidence,
    Locus::QualiaReference,
    Locus::MeaningLevel,
];

/// Count content loci on which two rows converge on the **same absolute event**.
#[must_use]
pub fn absolute_agreement(
    pos_a: usize,
    a: CausalWitnessFacet,
    pos_b: usize,
    b: CausalWitnessFacet,
) -> usize {
    CONTENT_LOCI
        .iter()
        .filter(|&&l| {
            a.is_bound(l)
                && b.is_bound(l)
                && (pos_a as isize + a.at(l) as isize) == (pos_b as isize + b.at(l) as isize)
        })
        .count()
}

/// The elected social peers for a focal row: the offsets to bind into its
/// Quorum / Contradiction loci, computed from the window fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PeerElection {
    /// Signed offset (`∈[−8,+7]`) to the agreeing peer, or `0` if none found.
    pub quorum_offset: i8,
    /// Signed offset to the preserved dissenting peer, or `0` if none.
    pub contradiction_offset: i8,
    /// Content-loci agreement count with the elected quorum peer.
    pub quorum_agreement: usize,
}

/// **E-QUORUM-FABRIC-1** — elect the focal row's quorum + contradiction peers
/// from the window. Quorum = the peer with the MOST content-loci convergence.
/// Contradiction = a peer that converges on ≥1 content locus but points its
/// **Kausal** (cause) edge at a DIFFERENT event — shares context, disagrees on
/// the cause (a preserved dissent, not mere unrelatedness). Offsets are the peer
/// position relative to the focal, clamped to the `±8` window (peers outside it
/// are not electable social loci).
///
/// `window` is `(stream_position, register)`; `focal_idx` indexes into it.
#[must_use]
pub fn elect_peers(focal_idx: usize, window: &[(usize, CausalWitnessFacet)]) -> PeerElection {
    let Some(&(focal_pos, focal)) = window.get(focal_idx) else {
        return PeerElection::default();
    };
    let mut best_q: Option<(usize, i8)> = None; // (agreement, offset)
    let mut best_c: Option<(usize, i8)> = None; // (agreement, offset)
    for (i, &(pos, peer)) in window.iter().enumerate() {
        if i == focal_idx {
            continue;
        }
        let delta = pos as isize - focal_pos as isize;
        if !(-8..=7).contains(&delta) {
            continue; // outside the ±8 window — not an electable social locus
        }
        let offset = delta as i8;
        let agree = absolute_agreement(focal_pos, focal, pos, peer);
        if agree == 0 {
            continue;
        }
        // Quorum candidate: maximize agreement.
        if best_q.is_none_or(|(a, _)| agree > a) {
            best_q = Some((agree, offset));
        }
        // Contradiction candidate: agrees on context BUT the Kausal cause points
        // elsewhere (both bound, different absolute target).
        let kausal_conflict = focal.is_bound(Locus::Kausal)
            && peer.is_bound(Locus::Kausal)
            && (focal_pos as isize + focal.cause() as isize)
                != (pos as isize + peer.cause() as isize);
        if kausal_conflict && best_c.is_none_or(|(a, _)| agree > a) {
            best_c = Some((agree, offset));
        }
    }
    PeerElection {
        quorum_offset: best_q.map(|(_, o)| o).unwrap_or(0),
        contradiction_offset: best_c.map(|(_, o)| o).unwrap_or(0),
        quorum_agreement: best_q.map(|(a, _)| a).unwrap_or(0),
    }
}

/// The result of following a locus chain across the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainResolution {
    /// Final signed offset from the focal to the resolved event, if resolved
    /// within budget and window; `None` if unresolved.
    pub final_offset: Option<i8>,
    /// Hops taken (0 = the focal's own locus resolved directly).
    pub hops: u8,
    /// The chain left the `±8` window — the signal a `temporal.rs` version-range
    /// read (`QueryReference::at`) is required. The contract emits the signal;
    /// the consumer does the read (no widening of the i4 nibble, no new witness
    /// variant). **Genuinely non-local: more hop budget will NOT help.**
    pub out_of_horizon: bool,
    /// The hop budget ran out mid-chain. **This is NOT non-locality** — it says
    /// "ask again with more budget", and the multipass loop's next iteration
    /// supplies exactly that.
    ///
    /// These two were ONE `escalated: bool` until `E-MULTIPASS-WAS-SINGLE-PASS-1`:
    /// conflating them made the wave abort at budget 1 (which truncates every
    /// ≥2-hop chain) before reaching the budget that resolves it, so the
    /// "multipass" wave was single-pass for every non-trivial chain. Splitting
    /// the carrier makes that conflation **unrepresentable** rather than merely
    /// fixed at each call site.
    pub budget_exhausted: bool,
}

impl ChainResolution {
    /// Either escalation cause — the v1 `escalated` reading, preserved for
    /// callers that genuinely do not care WHY the chain did not settle.
    ///
    /// Prefer the specific field: a caller that loops over increasing budgets
    /// must distinguish them, and using this in that position is what caused
    /// `E-MULTIPASS-WAS-SINGLE-PASS-1`.
    #[inline]
    #[must_use]
    pub const fn escalated(&self) -> bool {
        self.out_of_horizon || self.budget_exhausted
    }
}

/// **E-LOCI-CHAIN-ESCALATE-1** — follow the focal row's `locus` chain: the
/// offset points at a peer; if that peer's SAME locus is also bound, hop again
/// (the register following its own nibbles, `E-L9-REAL-TEXT-1`), up to
/// `max_hops`. Escalate (rather than widen) when the chain leaves the `±8`
/// window or exhausts the budget.
#[must_use]
pub fn resolve_chain(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    max_hops: u8,
) -> ChainResolution {
    let Some(&(focal_pos, focal)) = window.get(focal_idx) else {
        return ChainResolution {
            final_offset: None,
            hops: 0,
            out_of_horizon: false,
            budget_exhausted: false,
        };
    };
    // Map stream position → window index for O(1) hops.
    let idx_of = |pos: isize| window.iter().position(|&(p, _)| p as isize == pos);

    let mut cur_idx = focal_idx;
    let mut cur = focal;
    let mut hops = 0u8;
    loop {
        let off = cur.at(locus);
        if off == 0 {
            // chain broke before reaching a terminal — escalate if we hopped at all
            return ChainResolution {
                final_offset: None,
                hops,
                // Chain broke mid-walk (locus unbound). Not a horizon exit;
                // more budget cannot revive a broken chain either, but the
                // caller should treat it as "did not settle", so report it on
                // the budget side rather than inventing a third cause.
                out_of_horizon: false,
                budget_exhausted: hops > 0,
            };
        }
        let cur_pos = window[cur_idx].0 as isize;
        let target_pos = cur_pos + off as isize;
        let total = target_pos - focal_pos as isize;
        // Left the window the focal can address in one i4 → escalate.
        if !(-8..=7).contains(&total) {
            return ChainResolution {
                final_offset: None,
                hops,
                out_of_horizon: true,
                budget_exhausted: false,
            };
        }
        let Some(next_idx) = idx_of(target_pos) else {
            // target event not present in this window → resolved to the offset,
            // but its filler is out of window: escalate for the version read.
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                out_of_horizon: true,
                budget_exhausted: false,
            };
        };
        let next = window[next_idx].1;
        // If the target does NOT re-bind this locus, the chain terminates here.
        if !next.is_bound(locus) {
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                out_of_horizon: false,
                budget_exhausted: false,
            };
        }
        hops += 1;
        if hops >= max_hops {
            // budget exhausted mid-chain → escalate.
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                out_of_horizon: false,
                budget_exhausted: true,
            };
        }
        cur_idx = next_idx;
        cur = next;
    }
}

/// The grounding verdict of the **multipass Markov standing wave** over a locus.
///
/// The `±8` nibble is only the **reference horizon** (the cheap local window), NOT
/// a bound on awareness (operator ruling `E-HORIZON-NOT-BOUND-1`). A chain that
/// leaves it is not coincidental — its causality lives over a longer time span, so
/// the recipe [`Escalate`](WaveGrounding::Escalate)s to a `temporal.rs`
/// version-range read (search causality over time). There is no `Coincidental`
/// wave verdict: the wave cannot decide "genuinely spurious" from the local window
/// alone — only the escalated read can, and that is the consumer's call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveGrounding {
    /// Bound AND the standing wave SETTLES within the `±8` reference horizon —
    /// increasing the hop budget stops changing the resolved target (per-rung
    /// persistence, D-CSW-1): the locus is **causally grounded locally** (cheap).
    Causal,
    /// Bound, and the chain PERSISTS but leaves the `±8` reference horizon (or
    /// exhausts the hop budget mid-chain) — the causality lives over a longer time
    /// span. The signal to **escalate** — NOT a failure and NOT coincidental (a
    /// distant cause is still a cause). The `±8` window is dimensioned for GRAMMAR
    /// (local positional dependencies); causality past it is no longer positional,
    /// so escalation is a REPRESENTATION SWITCH to an ABSOLUTE address, not a wider
    /// offset (`E-GRAMMAR-LOCAL-CAUSAL-ABSOLUTE-1`): `part_of:is_a` (the AriGraph
    /// episodic basin) OR a centroid (CAM-PQ / palette256) — the two jobs of ONE
    /// implicit 256×256 tile, which holds an entire work's `≤64k` SPO (the whole
    /// Bible ≈ 31k, ~half a tile — `E-WHOLE-WORK-IS-ONE-TILE-1`), so the target is
    /// BOUNDED, not an unbounded search. The consumer does that absolute read
    /// (D-CSW-2); the contract only emits the signal.
    Escalate,
    /// The locus is not bound at all (unbound at the source).
    Unbound,
}

/// **The multipass Markov standing wave** over one locus — the LITERAL Markov-chain
/// resolution (not a coarse scalar proxy). Runs [`resolve_chain`] at increasing hop
/// budgets `1..=passes` (the standing wave's passes) and asks whether the resolved
/// target **persists**: a locus [`Causal`](WaveGrounding::Causal) within the `±8`
/// reference horizon settles (two successive budgets agree, chain terminated); one
/// whose chain leaves that horizon [`Escalate`](WaveGrounding::Escalate)s to a
/// `temporal.rs` read (the horizon is a reference, not a bound —
/// `E-HORIZON-NOT-BOUND-1`). This is D-CSW-1's "per-rung persistence separates
/// causal from coincidental" applied to grounding — the honest second resolution
/// beside the single-pass structural binding
/// ([`is_bound`](crate::causal_witness::CausalWitnessFacet::is_bound)).
#[must_use]
pub fn standing_wave_grounded(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    passes: u8,
) -> WaveGrounding {
    let Some(&(_, focal)) = window.get(focal_idx) else {
        return WaveGrounding::Unbound;
    };
    if !focal.is_bound(locus) {
        return WaveGrounding::Unbound;
    }
    let mut last: Option<i8> = None;
    let max_budget = passes.max(1);
    for budget in 1..=max_budget {
        let r = resolve_chain(focal_idx, window, locus, budget);
        // **Budget exhaustion is NOT the same as leaving the horizon.**
        // `ChainResolution::escalated` conflates them, and returning on the
        // first `escalated` defeated the whole multipass: a 2-hop chain is
        // truncated at budget 1, so the loop escalated before ever trying
        // budget 2 — the very budget that resolves it. Only a chain that STILL
        // escalates at the final budget is genuinely non-local
        // (`E-MULTIPASS-WAS-SINGLE-PASS-1`). Below the final budget, an
        // escalation means "needs more hops" — which is exactly what the next
        // iteration supplies.
        // Now expressible directly: a horizon exit is terminal at ANY budget
        // (more hops cannot bring it back inside), while budget exhaustion is
        // exactly what the next iteration fixes.
        if r.out_of_horizon {
            return WaveGrounding::Escalate;
        }
        if r.budget_exhausted {
            if budget == max_budget {
                return WaveGrounding::Escalate;
            }
            last = None; // nothing trustworthy resolved at this budget
            continue;
        }
        match r.final_offset {
            // settled: this budget resolved to the same target the previous did
            // (adding budget stopped moving it) and it did not escalate.
            Some(off) => {
                if last == Some(off) {
                    return WaveGrounding::Causal; // the wave stood still → causal
                }
                last = Some(off);
            }
            // resolved to nothing inside the horizon without escalating — the
            // chain has no local target; treat as needing the wider search.
            None => return WaveGrounding::Escalate,
        }
    }
    // A single-hop terminal chain resolves identically at every budget: `last`
    // is Some but the "two agree" short-circuit needs ≥2 passes to fire, so a
    // resolved-non-escalated chain is causal (grounded in the reference horizon).
    if last.is_some() {
        WaveGrounding::Causal
    } else {
        WaveGrounding::Escalate
    }
}

/// **The stratified standing wave** — [`standing_wave_grounded`] plus the pass
/// at which the wave actually SETTLED, so the caller can normalize its thinking
/// depth to what the resolution cost (`E-STANDING-WAVE-IS-UNSTRATIFIED-SUDOKU-1`).
///
/// Returns `(grounding, settle_pass)` where `settle_pass` is the hop budget that
/// produced the verdict, counted from 1 — the same counting the internal budget
/// loop uses. Feed it straight to
/// [`RungLevel::for_pass`](crate::cognitive_shader::RungLevel::for_pass) to get
/// the rung this resolution is normalized to, and thence to
/// [`RungLevel::admissible_recipes`](crate::cognitive_shader::RungLevel::admissible_recipes)
/// for the tactics that may legally fire on it:
///
/// ```
/// # use lance_graph_contract::witness_fabric::{standing_wave_stratified, WaveGrounding};
/// # use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
/// # use lance_graph_contract::cognitive_shader::RungLevel;
/// # let window: Vec<(usize, CausalWitnessFacet)> = vec![];
/// let (grounding, settle_pass) =
///     standing_wave_stratified(0, &window, Locus::Antecedent, 8);
/// let rung = RungLevel::for_pass(settle_pass);
/// let admissible = rung.admissible_recipes().count();
/// # let _ = (grounding, admissible);
/// ```
///
/// **Why a cheap settle is a SHALLOW rung, not a lucky one.** A locus that
/// grounds on pass 1 needed one hop; permitting a Counterfactual tactic to fire
/// on it would spend rung-6 machinery on a rung-0 fact. Conversely a chain that
/// only settles at pass 7 has earned the deeper tactics. This is the Sudoku
/// discipline — exhaust the cheap constraints, escalate only the residual —
/// expressed as a rung rather than a heuristic.
///
/// **The cheapest observable ground costs TWO passes, not one** (pinned by
/// `cheap_settle_normalizes_to_a_shallow_rung`). [`Causal`](WaveGrounding::Causal)
/// is declared only when two SUCCESSIVE budgets resolve the same target, so even
/// a single-hop terminal chain reports `settle_pass == 2`. That is a feature of
/// the mapping, not an off-by-one: it leaves pass 0 → [`Surface`](
/// crate::cognitive_shader::RungLevel::Surface) cleanly reserved for "nothing was
/// resolved", and makes [`Shallow`](crate::cognitive_shader::RungLevel::Shallow)
/// the shallowest rung any real grounding can earn. Both admit only the
/// [`Gate`](crate::recipes::Bucket::Gate) bucket, so the discipline is unaffected.
///
/// [`Escalate`](WaveGrounding::Escalate) reports the pass at which the chain left
/// the `±8` horizon; that is itself a rung elevation (positional → absolute
/// address, `E-GRAMMAR-LOCAL-CAUSAL-ABSOLUTE-1`), so the caller may elevate
/// rather than treat it as a dead end. [`Unbound`](WaveGrounding::Unbound)
/// reports pass 0 — nothing was resolved, so no rung was earned.
#[must_use]
pub fn standing_wave_stratified(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    passes: u8,
) -> (WaveGrounding, u8) {
    let Some(&(_, focal)) = window.get(focal_idx) else {
        return (WaveGrounding::Unbound, 0);
    };
    if !focal.is_bound(locus) {
        return (WaveGrounding::Unbound, 0);
    }
    let mut last: Option<i8> = None;
    let max_budget = passes.max(1);
    for budget in 1..=max_budget {
        let r = resolve_chain(focal_idx, window, locus, budget);
        // Same budget-exhaustion-vs-horizon distinction as
        // `standing_wave_grounded`; the two MUST stay in verdict parity
        // (pinned by `stratified_never_disagrees_with_grounded`).
        if r.out_of_horizon {
            return (WaveGrounding::Escalate, budget);
        }
        if r.budget_exhausted {
            if budget == max_budget {
                return (WaveGrounding::Escalate, budget);
            }
            last = None;
            continue;
        }
        match r.final_offset {
            Some(off) => {
                if last == Some(off) {
                    return (WaveGrounding::Causal, budget);
                }
                last = Some(off);
            }
            None => return (WaveGrounding::Escalate, budget),
        }
    }
    // Same terminal-chain case as `standing_wave_grounded`: a single-hop chain
    // resolves identically at every budget, so it is causal at the FIRST pass —
    // the cheapest possible grounding, and the one that most deserves a shallow
    // rung.
    if last.is_some() {
        (WaveGrounding::Causal, 1)
    } else {
        (WaveGrounding::Escalate, passes.max(1))
    }
}

/// WHY an escalation fired — the distinction [`WaveGrounding::Escalate`]
/// erases, surfaced (external-review adjudication: the reviewer's "hard
/// max-pass truncation" claim landed on this exact spot — the CARRIER
/// distinguishes the reasons since the `E-MULTIPASS-WAS-SINGLE-PASS-1` fix,
/// but the wave verdict collapsed both back into one variant).
///
/// The two reasons demand OPPOSITE responses, which is why conflating them is
/// not cosmetic:
/// * [`OutOfHorizon`](Self::OutOfHorizon) — the chain genuinely left the ±8
///   window. More budget cannot help; the correct move is the representation
///   switch (basin edge / `temporal.rs` version read).
/// * [`BudgetExhausted`](Self::BudgetExhausted) — the chain was still walking
///   when the FINAL budget ran out. Slow convergence, not non-locality; the
///   correct move is to retry with a larger `passes`, which is cheap, before
///   paying for the wide read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EscalateReason {
    OutOfHorizon,
    BudgetExhausted,
}

/// [`standing_wave_stratified`] plus the escalation REASON — additive
/// diagnosis, never a second opinion: the `(grounding, settle_pass)` pair is
/// parity-pinned bit-identical to the undiagnosed functions
/// (`diagnosed_never_disagrees_with_stratified`).
///
/// `reason` is `Some` iff `grounding == Escalate`. A `None` reason on an
/// `Escalate` (or vice versa) is unrepresentable by construction of this
/// function — tested, not promised.
#[must_use]
pub fn standing_wave_diagnosed(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    passes: u8,
) -> (WaveGrounding, u8, Option<EscalateReason>) {
    let Some(&(_, focal)) = window.get(focal_idx) else {
        return (WaveGrounding::Unbound, 0, None);
    };
    if !focal.is_bound(locus) {
        return (WaveGrounding::Unbound, 0, None);
    }
    let mut last: Option<i8> = None;
    let max_budget = passes.max(1);
    for budget in 1..=max_budget {
        let r = resolve_chain(focal_idx, window, locus, budget);
        if r.out_of_horizon {
            return (
                WaveGrounding::Escalate,
                budget,
                Some(EscalateReason::OutOfHorizon),
            );
        }
        if r.budget_exhausted {
            if budget == max_budget {
                return (
                    WaveGrounding::Escalate,
                    budget,
                    Some(EscalateReason::BudgetExhausted),
                );
            }
            last = None;
            continue;
        }
        match r.final_offset {
            Some(off) => {
                if last == Some(off) {
                    return (WaveGrounding::Causal, budget, None);
                }
                last = Some(off);
            }
            // Resolved to nothing inside the horizon without escalating: no
            // local target exists, and no amount of budget manufactures one —
            // the wider read is genuinely required, so this is horizon-class,
            // not budget-class.
            None => {
                return (
                    WaveGrounding::Escalate,
                    budget,
                    Some(EscalateReason::OutOfHorizon),
                )
            }
        }
    }
    if last.is_some() {
        (WaveGrounding::Causal, 1, None)
    } else {
        (
            WaveGrounding::Escalate,
            max_budget,
            Some(EscalateReason::BudgetExhausted),
        )
    }
}

/// **The passive quorum mantissa** — what discriminates once admissibility no
/// longer can.
///
/// At [`RungLevel::Transcendent`](crate::cognitive_shader::RungLevel::Transcendent)
/// every one of the 34 tactics is admissible, so the rung gate has no
/// discriminating power left. Continuing to pick a *dominant* tactic there is
/// eigenvalue-following with nothing to justify it. The discrimination must
/// switch from ACTIVE selection to **PASSIVE observation of agreement**: how
/// much of the window converges on the same absolute events, as a magnitude.
///
/// Returns a `0..=15` mantissa (i4 range, so it lands in the existing qualia
/// nibble carving without widening anything): `0` = no peer agrees, `15` =
/// saturated agreement.
///
/// # Hindsight-blindness is prevented BY THE SIGNATURE, not by discipline
///
/// This function cannot see any resolution, verdict, settle pass, or outcome —
/// it takes only the window. That is deliberate and load-bearing: a quorum that
/// could observe the answer would weight the peers that "turned out right",
/// which is textbook hindsight bias — every past state reads as having pointed
/// at the settled result. Because the outcome is *unavailable at the type
/// level*, no future edit can quietly introduce that weighting without changing
/// the signature and failing review.
///
/// The same no-self-reference rule as [`elect_peers`] applies: only
/// [`CONTENT_LOCI`] contribute — the social loci (Quorum, Contradiction) are
/// what this COMPUTES and so must not be read as input.
#[must_use]
pub fn quorum_mantissa(focal_idx: usize, window: &[(usize, CausalWitnessFacet)]) -> u8 {
    let Some(&(focal_pos, focal)) = window.get(focal_idx) else {
        return 0;
    };
    let peers = window.len().saturating_sub(1);
    if peers == 0 {
        return 0;
    }
    // Total agreements observed, and the ceiling they are measured against.
    let mut agreed = 0usize;
    for (i, &(pos, peer)) in window.iter().enumerate() {
        if i == focal_idx {
            continue;
        }
        agreed += absolute_agreement(focal_pos, focal, pos, peer);
    }
    let ceiling = peers * CONTENT_LOCI.len();
    if ceiling == 0 {
        return 0;
    }
    // Scale into 0..=15 with saturating rounding-down (never claims more
    // agreement than observed).
    ((agreed * 15) / ceiling).min(15) as u8
}

/// The multi-hop causal **trajectory** of one locus — the shape the tail is
/// graded by.
///
/// Rows that disagree (low [`quorum_mantissa`]) are not noise to discard; they
/// are the tail worth riding. Grading them by *how* their causality resolved —
/// hop count, whether it left the horizon, where it terminated — turns the tail
/// into a clusterable feature space instead of a reject pile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TrajectorySignature {
    /// Hops taken to resolve (0 = resolved directly at the focal).
    pub hops: u8,
    /// The chain left the `±8` horizon or exhausted its budget.
    pub escalated: bool,
    /// Terminal signed offset, or `None` if it resolved to nothing locally.
    pub terminal_offset: Option<i8>,
}

impl TrajectorySignature {
    /// Two trajectories belong to the same **meta-basin** when they share hop
    /// depth and escalation status — the causal SHAPE — regardless of which
    /// absolute event they terminated on. Same shape, same basin; the terminal
    /// offset then distinguishes MINI basins inside it.
    #[must_use]
    pub const fn same_meta_basin(self, other: Self) -> bool {
        self.hops == other.hops && self.escalated == other.escalated
    }
}

/// Compute the [`TrajectorySignature`] of one locus for one row.
#[must_use]
pub fn trajectory_of(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    max_hops: u8,
) -> TrajectorySignature {
    let r = resolve_chain(focal_idx, window, locus, max_hops);
    TrajectorySignature {
        hops: r.hops,
        escalated: r.escalated(),
        terminal_offset: r.final_offset,
    }
}

/// **E-CONTRADICTION-OPINION-1** — a stance/opinion is a row whose Contradiction
/// locus stays BOUND across successive revisions (committed-contradiction
/// persistence as first-class epistemic state). `revisions` is the same row's
/// register at successive versions (oldest→newest).
///
/// Returns how many revisions kept the contradiction bound; the row IS an
/// opinion iff that persistence covers the whole (non-empty) history.
#[must_use]
pub fn opinion_strength(revisions: &[CausalWitnessFacet]) -> usize {
    revisions
        .iter()
        .filter(|r| r.is_bound(Locus::Contradiction))
        .count()
}

/// A row is an OPINION iff its contradiction survived every revision (a preserved
/// dissent, not transient noise). Empty history is not an opinion.
#[must_use]
pub fn is_opinion(revisions: &[CausalWitnessFacet]) -> bool {
    !revisions.is_empty() && opinion_strength(revisions) == revisions.len()
}

// ── The VERSION axis — belief-state traversal ─────────────────────────────
//
// [`TrajectorySignature`] grades how a locus resolved across the ±8 SPATIAL
// window. This is its twin on the TEMPORAL axis: how a belief resolved across
// successive revisions (Lance versions, oldest→newest). Same three questions —
// how many steps, did it destabilize, where did it end — asked of time instead
// of position.
//
// The pairing is not decorative. `E-MARKOV-TEMPORAL-STREAM-1` routes a chain
// that leaves the ±8 horizon to a `temporal.rs` version-range read, so
// [`WaveGrounding::Escalate`] is literally the door from the spatial axis to
// this one. Grading both with the same shape means an escalation does not fall
// off a cliff into an ungraded medium.
//
// `opinion_strength` / `is_opinion` already consume revision series — this
// extends that shape rather than opening a parallel one.

/// How a belief moved across successive revisions — the version-axis twin of
/// [`TrajectorySignature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RevisionTrajectory {
    /// Revisions observed (the series length actually read).
    pub steps: u8,
    /// Times the locus changed target between consecutive revisions — the
    /// temporal analogue of [`TrajectorySignature::escalated`]: instability,
    /// not failure. A belief that flips is not a broken belief; it is a belief
    /// under pressure, and that is a signal worth keeping.
    pub flips: u8,
    /// Revisions since the last flip (equals `steps` when it never flipped).
    /// High = calcified, low = live.
    pub stable_for: u8,
    /// Where it currently stands; `None` = unbound at the latest revision.
    pub terminal: Option<i8>,
}

impl RevisionTrajectory {
    /// **Churn as a `0..=15` mantissa** — flips per step, in the same i4 range
    /// as [`quorum_mantissa`] so it lands in the existing qualia nibble carving
    /// without widening anything.
    ///
    /// This is a qualia dimension nothing else hydrates: a basin whose beliefs
    /// keep flipping is *epistemically different* from one that calcified, even
    /// when both currently hold the same values. Only the version axis can see
    /// that difference — a snapshot cannot.
    #[must_use]
    pub const fn churn_mantissa(&self) -> u8 {
        if self.steps <= 1 {
            return 0; // a single observation cannot have churned
        }
        // flips ∈ 0..=steps-1 → scale that range onto 0..=15.
        let denom = (self.steps - 1) as u16;
        let scaled = (self.flips as u16 * 15) / denom;
        if scaled > 15 {
            15
        } else {
            scaled as u8
        }
    }

    /// Two beliefs share a **temporal meta-basin** when they churned alike —
    /// the same grouping-by-shape rule [`TrajectorySignature::same_meta_basin`]
    /// applies spatially. Terminal value is deliberately NOT compared: it is
    /// what distinguishes mini-basins inside.
    #[must_use]
    pub const fn same_churn_basin(self, other: Self) -> bool {
        self.churn_mantissa() == other.churn_mantissa()
    }
}

/// Grade a belief's traversal across a revision series (oldest→newest).
///
/// # Read-as-of is enforced by the parameter, not by discipline
///
/// `upto` bounds how much of the history is visible: a trajectory computed
/// "as of" revision *v* passes `upto = v + 1` and **cannot** observe anything
/// that arrived later. This is the same structural guarantee as
/// [`quorum_mantissa`]'s outcome-blind signature, applied to time: retrospective
/// calibration ("was the confidence at *v* justified by what was knowable at
/// *v*?") becomes an honest measurement instead of the textbook hindsight trap,
/// because the future is not reachable from the call.
///
/// Passing `upto >= revisions.len()` reads the whole history — which is correct
/// for "what is the current state", and wrong for any as-of question. The
/// distinction is the caller's to make, deliberately: it is visible at every
/// call site rather than buried in a flag.
#[must_use]
pub fn revision_trajectory(
    revisions: &[CausalWitnessFacet],
    locus: Locus,
    upto: usize,
) -> RevisionTrajectory {
    let visible = &revisions[..upto.min(revisions.len())];
    if visible.is_empty() {
        return RevisionTrajectory::default();
    }
    let mut flips = 0u8;
    let mut stable_for = 0u8;
    let mut prev: Option<i8> = None;
    for r in visible {
        let cur = if r.is_bound(locus) {
            Some(r.at(locus))
        } else {
            None
        };
        match (prev, cur) {
            // First observation establishes the baseline; it is not a flip.
            (None, _) => stable_for = 1,
            (Some(p), Some(c)) if p == c => stable_for = stable_for.saturating_add(1),
            // Any change of target — including bind→unbind — is a flip.
            _ => {
                flips = flips.saturating_add(1);
                stable_for = 1;
            }
        }
        if cur.is_some() {
            prev = cur;
        }
    }
    let last = visible.last().expect("non-empty");
    RevisionTrajectory {
        steps: visible.len().min(u8::MAX as usize) as u8,
        flips,
        stable_for,
        terminal: if last.is_bound(locus) {
            Some(last.at(locus))
        } else {
            None
        },
    }
}

// ── The TEMPORAL PERIPHERY — superseded belief states ─────────────────────
//
// **The present is the dominant mode of the time axis.** A reader that always
// takes HEAD treats every overwritten belief state as a reject pile — which is
// the exact eigenvalue-following failure `E-PERIPHERAL-DISSENT-GUARDS-THE-
// STRATIFICATION-1` names, applied to time instead of to the rung ladder. The
// anti-eigenvalue three-part test (**enumerable / spread-sampled /
// escalation-capable**) was applied there and swept across the spatial prunes;
// this is its application to the version axis.
//
// The same shape as the spatial side: FUNCTIONS over a slice, never a
// materialized history struct. [`belief_runs`] enumerates, [`superseded_spread_sample`]
// samples with a stride (the temporal form of `RungLevel::peripheral_sample`),
// [`suggest_reopening`] escalates — and, like every other periphery channel in
// this module, it SUGGESTS and never decides.
//
// **Divergence from [`revision_trajectory`], stated rather than hidden.** The
// run model here treats "unbound" as a state of its own, so `a → unbound → a`
// is three runs; `revision_trajectory` carries the last BOUND value forward and
// counts that same series as one flip. Neither is wrong — churn asks "how often
// did the belief move", runs ask "which distinct states did the history hold".
// They are deliberately not unified: doing so would change
// `revision_trajectory`'s shipped semantics.

/// One contiguous stretch of the history during which a belief held ONE state.
///
/// A run that was replaced carries `superseded_at`; the newest run does not.
/// The superseded runs ARE the temporal periphery — the states a HEAD read
/// discards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BeliefRun {
    /// Index (into the visible history) of the first revision holding it.
    pub first_revision: u8,
    /// Consecutive revisions that held it. Never 0 — a run exists because a
    /// revision held it.
    pub held_for: u8,
    /// The state held; `None` = the locus was unbound across this run.
    /// **Absent is not a value**: an unbound run is a real run, but it is never
    /// a candidate for re-opening (there is no belief to re-open).
    pub value: Option<i8>,
    /// Revision index at which a DIFFERENT state took over, or `None` if this
    /// run is still current. This is the "when did the flip happen" that
    /// [`RevisionTrajectory::flips`] can only count.
    pub superseded_at: Option<u8>,
}

impl BeliefRun {
    /// This run was replaced by a later one — it is part of the periphery.
    #[inline]
    #[must_use]
    pub const fn is_superseded(&self) -> bool {
        self.superseded_at.is_some()
    }

    /// Revision indices this run covers, as a half-open `[start, end)` pair.
    /// Used to prove the runs TILE the history rather than merely being derived
    /// from it.
    #[inline]
    #[must_use]
    pub const fn span(&self) -> (u8, u8) {
        (self.first_revision, self.first_revision + self.held_for)
    }
}

/// **Enumerate** the belief states a history held, oldest→newest.
///
/// This is part 1 of the three-part periphery test: *a prune nobody can
/// enumerate is a blind spot; a prune you can enumerate is a budget.*
/// [`RevisionTrajectory::flips`] counts the flips; this NAMES them — which
/// revisions held which state, and at which revision each was superseded.
///
/// # Read-as-of is enforced by the parameter, not by discipline
///
/// `upto` bounds visibility exactly as in [`revision_trajectory`]: a run set
/// computed as of revision *v* passes `upto = v + 1` and cannot observe
/// anything later. Retrospective judgement of a past state must not be able to
/// see what came after it, or it is hindsight wearing an audit's clothes.
///
/// Visibility is additionally capped at 255 revisions so the indices stay in
/// `u8` (the same clamping [`RevisionTrajectory::steps`] already does).
#[must_use]
pub fn belief_runs(revisions: &[CausalWitnessFacet], locus: Locus, upto: usize) -> Vec<BeliefRun> {
    let end = upto.min(revisions.len()).min(u8::MAX as usize);
    let mut runs: Vec<BeliefRun> = Vec::new();
    for (i, r) in revisions[..end].iter().enumerate() {
        let value = if r.is_bound(locus) {
            Some(r.at(locus))
        } else {
            None
        };
        match runs.last_mut() {
            Some(last) if last.value == value => last.held_for = last.held_for.saturating_add(1),
            _ => runs.push(BeliefRun {
                first_revision: i as u8,
                held_for: 1,
                value,
                superseded_at: None,
            }),
        }
    }
    for i in 1..runs.len() {
        let at = runs[i].first_revision;
        runs[i - 1].superseded_at = Some(at);
    }
    runs
}

/// The temporal periphery itself: every run that a later revision replaced.
///
/// Empty for an empty history AND for a history that never changed — in both
/// cases nothing has been superseded. **Absent ≠ zero**: "no superseded states"
/// because there is no history is not the same claim as "no superseded states
/// because the belief is stable", and the caller can tell them apart by asking
/// [`belief_runs`] for the whole picture.
#[must_use]
pub fn superseded_runs(
    revisions: &[CausalWitnessFacet],
    locus: Locus,
    upto: usize,
) -> Vec<BeliefRun> {
    let mut runs = belief_runs(revisions, locus, upto);
    runs.pop(); // the newest run is the present, not the periphery
    runs
}

/// **Sample the periphery with a SPREAD** — up to `k` superseded runs, strided
/// across the WHOLE history rather than taken from its recent edge.
///
/// Part 2 of the three-part test, and the part most easily got wrong. Taking
/// the `k` most-recent superseded states is the temporal form of sampling the
/// cheap edge: it re-creates the present-dominance blindness one level down,
/// because the states nearest HEAD are the ones a HEAD reader already half-sees.
/// The stride reaches the OLD end too.
///
/// Deterministic by construction (no RNG) — mirroring
/// [`RungLevel::peripheral_sample`](crate::cognitive_shader::RungLevel), whose
/// reasoning applies verbatim: a reproducible sample is the only auditable one.
#[must_use]
pub fn superseded_spread_sample(
    revisions: &[CausalWitnessFacet],
    locus: Locus,
    upto: usize,
    k: usize,
) -> Vec<BeliefRun> {
    let periphery = superseded_runs(revisions, locus, upto);
    let n = periphery.len();
    let take = k.min(n);
    if take == 0 {
        return Vec::new();
    }
    let stride = (n / take).max(1);
    (0..take)
        .filter_map(|i| periphery.get(i * stride).copied())
        .collect()
}

/// Why a superseded state is worth re-opening. Never a verdict — see
/// [`ReopeningSuggestion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReopeningReason {
    /// The value was ABANDONED and later came back — it appears in ≥2 distinct
    /// runs. A belief the history keeps returning to is not settled noise; the
    /// flip away from it is the part that needs justifying.
    Reverted,
    /// A LONG-stable value was replaced by a strictly SHORTER-lived one. The
    /// system traded accumulated stability for something it then did not keep,
    /// which is the structural silhouette of a regression.
    LongStableThenBrief,
}

/// The evidence a suggestion carries, so a consumer can weigh it instead of
/// being asserted at. Same discipline as
/// `OutlierSuggestion::basin_size` on the spatial side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ReopeningEvidence {
    /// Total revisions the suggested value was held across the visible history
    /// (summed over all its runs).
    pub total_held: u8,
    /// Revisions the CURRENT state has been held. A suggestion whose value
    /// out-held the incumbent reads differently from one that did not.
    pub current_held: u8,
    /// Distinct runs in which the suggested value appeared.
    pub occurrences: u8,
}

/// A superseded state proposed for re-examination, with its evidence attached.
///
/// **Nothing here prunes, decides, or scores.** It is the temporal sibling of
/// [`WaveGrounding::Escalate`] and `peripheral_dissent`: a signal that the
/// consumer may act on, never an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReopeningSuggestion {
    /// The superseded run being surfaced.
    pub run: BeliefRun,
    /// The structural pattern that surfaced it.
    pub reason: ReopeningReason,
    /// What the pattern was measured against.
    pub evidence: ReopeningEvidence,
}

/// **Escalation-capability** — surface superseded states that structurally
/// deserve re-opening, oldest→newest, deterministically.
///
/// # What this CANNOT measure, stated plainly
///
/// The honest signal would be "a past state predicted better than the present
/// one". **This crate cannot compute that.** Scoring prediction requires ground
/// truth — an outcome each belief state can be checked against — and a zero-dep
/// contract crate has no oracle, no outcome column, and (by the read-as-of rule
/// above) deliberately no access to the future of the revision it speaks for.
/// Any function here claiming to rank past states by predictive accuracy would
/// be inventing the thing it measures.
///
/// What IS measurable without an oracle is the SHAPE of the history, and two
/// shapes are honest proxies for "worth re-opening":
///
/// * [`Reverted`](ReopeningReason::Reverted) — the history came back to a value
///   it had abandoned. Recurrence after abandonment is evidence the abandonment
///   was premature, *without needing to know which value is right*.
/// * [`LongStableThenBrief`](ReopeningReason::LongStableThenBrief) — a value
///   held for ≥2 revisions was replaced by a strictly shorter-lived one. The
///   trade is visible in the structure alone.
///
/// **These are proxies, not the signal.** A genuine predicted-better test needs
/// an outcome the consumer holds; the right shape for it is a consumer-side
/// function taking `(belief_runs(.., upto), outcome_at(upto))`, which keeps the
/// oracle outside the contract and the read-as-of bound intact.
///
/// Unbound runs are never suggested: there is no belief to re-open, and
/// treating absence as a candidate would be the absent-as-zero error.
#[must_use]
pub fn suggest_reopening(
    revisions: &[CausalWitnessFacet],
    locus: Locus,
    upto: usize,
) -> Vec<ReopeningSuggestion> {
    let runs = belief_runs(revisions, locus, upto);
    if runs.len() < 2 {
        return Vec::new(); // nothing was superseded — no periphery to escalate
    }
    let current_held = runs[runs.len() - 1].held_for;
    let mut out: Vec<ReopeningSuggestion> = Vec::new();
    for i in 0..runs.len() - 1 {
        let run = runs[i];
        let Some(value) = run.value else {
            continue; // absent is not a belief
        };
        let same: Vec<&BeliefRun> = runs.iter().filter(|r| r.value == Some(value)).collect();
        let occurrences = same.len().min(u8::MAX as usize) as u8;
        let total_held = same
            .iter()
            .fold(0u8, |acc, r| acc.saturating_add(r.held_for));
        let evidence = ReopeningEvidence {
            total_held,
            current_held,
            occurrences,
        };
        // Recurrence is the stronger signal; report it once, on the EARLIEST
        // run holding the value, so a value returning three times does not
        // emit three near-identical suggestions.
        let first_of_value = same
            .first()
            .is_some_and(|r| r.first_revision == run.first_revision);
        if occurrences >= 2 && first_of_value {
            out.push(ReopeningSuggestion {
                run,
                reason: ReopeningReason::Reverted,
                evidence,
            });
            continue;
        }
        if occurrences < 2 && run.held_for >= 2 && runs[i + 1].held_for < run.held_for {
            out.push(ReopeningSuggestion {
                run,
                reason: ReopeningReason::LongStableThenBrief,
                evidence,
            });
        }
    }
    out
}

// ── The Epistemic Foresight Test (minimal honest form) ────────────────────
//
// External-review convergence: the reviewer's "decisive experiment" for
// functional awareness — *did the substrate, at version v, correctly identify
// which of its own beliefs were most likely to require later correction?* —
// is exactly the read-as-of calibration probe this module's `upto` bounds
// were built for. This is the MINIMAL honest form: the only risk signal used
// is churn-as-of-v, so the claim under test is narrow and falsifiable
// ("past instability predicts future revision"), not a composite
// awareness score. Richer risk profiles (method competence, coverage,
// independence) are CONSUMER-side — they need data this zero-dep crate
// does not hold.
//
// Hindsight discipline: the risk is computed through `upto = v`, so it
// physically cannot see the post-v segment it is scored against.

/// One belief's foresight sample: what the substrate could say about itself
/// AS OF `v`, paired with what actually happened AFTER `v`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForesightSample {
    /// Churn mantissa computed from revisions `..v` only (`0..=15`).
    pub risk_asof_v: u8,
    /// Flips observed in the post-`v` segment.
    pub flips_after: u8,
    /// Post-`v` revisions observed (the denominator; 0 = not scorable).
    pub steps_after: u8,
}

/// Sample one belief series at split point `v`.
///
/// Returns `None` when either side of the split is empty — a belief with no
/// pre-`v` history has no self-knowledge to score, and one with no post-`v`
/// history has no outcome to score against. Absent is not zero on either
/// axis.
#[must_use]
pub fn foresight_sample(
    revisions: &[CausalWitnessFacet],
    locus: Locus,
    v: usize,
) -> Option<ForesightSample> {
    if v == 0 || v >= revisions.len() {
        return None;
    }
    let before = revision_trajectory(revisions, locus, v);
    if before.steps == 0 {
        return None;
    }
    // The post segment must OVERLAP the boundary by one revision so a flip
    // across the split itself is counted: compare from the last pre-v state.
    let after = revision_trajectory(&revisions[v - 1..], locus, usize::MAX);
    Some(ForesightSample {
        risk_asof_v: before.churn_mantissa(),
        flips_after: after.flips,
        steps_after: after.steps.saturating_sub(1),
    })
}

/// Flip-rate per risk band over many sampled beliefs — the calibration curve
/// of the substrate's self-doubt.
///
/// Bands: low = `0..=5`, mid = `6..=10`, high = `11..=15`. Each entry is
/// `(scorable_beliefs, total_flips, total_post_steps)`; rates are the
/// caller's division so that empty bands stay visibly empty instead of
/// becoming a confident 0.0.
///
/// The PASS condition (scored by the consumer, stated here so it cannot
/// drift): high-risk beliefs flip more often per post-step than low-risk
/// ones. The test does not ask the substrate to know the future — it asks
/// whether it understands the weakness of its own present.
#[must_use]
pub fn foresight_calibration(samples: &[ForesightSample]) -> [(u32, u32, u32); 3] {
    let mut bins = [(0u32, 0u32, 0u32); 3];
    for s in samples {
        if s.steps_after == 0 {
            continue; // not scorable — never a confident zero
        }
        let b = match s.risk_asof_v {
            0..=5 => 0,
            6..=10 => 1,
            _ => 2,
        };
        bins[b].0 += 1;
        bins[b].1 += u32::from(s.flips_after);
        bins[b].2 += u32::from(s.steps_after);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    #[test]
    fn absolute_agreement_is_same_event_not_same_offset() {
        // A at pos 5 points Kausal −3 → event 2. B at pos 7 points Kausal −5 → event 2.
        // Different OFFSETS but the SAME absolute event → they agree.
        let a = w(&[(Locus::Kausal, -3)]);
        let b = w(&[(Locus::Kausal, -5)]);
        assert_eq!(absolute_agreement(5, a, 7, b), 1);
        // Same offset but different positions → different events → no agreement.
        let c = w(&[(Locus::Kausal, -3)]);
        assert_eq!(absolute_agreement(5, a, 7, c), 0);
    }

    #[test]
    fn elect_peers_picks_the_max_agreement_quorum() {
        // focal at pos 5; two content loci to SMeaning@7, Kausal@2.
        let focal = w(&[(Locus::SMeaning, 2), (Locus::Kausal, -3)]);
        let strong = w(&[(Locus::SMeaning, 1), (Locus::Kausal, -4)]); // pos 6 → S@7, K@2: both agree
        let weak = w(&[(Locus::SMeaning, 4)]); // pos 4 → S@8: disagree
        let window = [(5usize, focal), (6, strong), (4, weak)];
        let e = elect_peers(0, &window);
        assert_eq!(e.quorum_offset, 1, "the agreeing peer is at +1");
        assert_eq!(e.quorum_agreement, 2, "both content loci converge");
    }

    #[test]
    fn elect_peers_finds_kausal_dissenter_as_contradiction() {
        // A peer that agrees on SMeaning but points Kausal elsewhere = contradiction.
        let focal = w(&[(Locus::SMeaning, 2), (Locus::Kausal, -3)]); // pos 5 → S@7, K@2
        let dissenter = w(&[(Locus::SMeaning, 1), (Locus::Kausal, 0)]); // pos 6 → S@7 agree; K unbound? no
        let dissenter = dissenter.with(Locus::Kausal, 2); // pos 6 → K@8 ≠ focal K@2
        let window = [(5usize, focal), (6, dissenter)];
        let e = elect_peers(0, &window);
        assert_eq!(
            e.contradiction_offset, 1,
            "the Kausal-dissenting peer at +1"
        );
    }

    #[test]
    fn peers_outside_the_window_are_not_electable() {
        let focal = w(&[(Locus::SMeaning, 2)]);
        let far = w(&[(Locus::SMeaning, 2)]); // agrees, but 20 away
        let window = [(5usize, focal), (25, far)];
        assert_eq!(elect_peers(0, &window).quorum_offset, 0);
    }

    #[test]
    fn resolve_chain_hops_then_escalates_on_budget() {
        // pos 3 → Kausal +2 → pos 5 (rebinds) → +2 → pos 7 (rebinds) → +2 → pos 9
        let a = w(&[(Locus::Kausal, 2)]);
        let b = w(&[(Locus::Kausal, 2)]);
        let c = w(&[(Locus::Kausal, 2)]);
        let d = w(&[]); // terminal, no Kausal
        let window = [(3usize, a), (5, b), (7, c), (9, d)];
        // budget 1 hop: after 1 hop, escalate.
        let r1 = resolve_chain(0, &window, Locus::Kausal, 1);
        assert!(r1.escalated() && r1.hops == 1);
        // budget 5: chain resolves to pos 9 (total offset +6) within window.
        let r5 = resolve_chain(0, &window, Locus::Kausal, 5);
        assert_eq!(r5.final_offset, Some(6));
        assert!(!r5.escalated());
    }

    #[test]
    fn resolve_chain_escalates_when_leaving_the_window() {
        // pos 0 → Kausal +7 → pos 7 (rebinds) → +7 would be pos 14 = total +14 > 7 → escalate.
        let a = w(&[(Locus::Kausal, 7)]);
        let b = w(&[(Locus::Kausal, 7)]);
        let window = [(0usize, a), (7, b)];
        let r = resolve_chain(0, &window, Locus::Kausal, 8);
        assert!(
            r.escalated(),
            "chain leaves the ±8 window → escalate to version-range read"
        );
    }

    #[test]
    fn opinion_is_a_persisted_contradiction() {
        let bound = w(&[(Locus::Contradiction, -4)]);
        let clear = w(&[(Locus::SMeaning, 1)]);
        assert!(is_opinion(&[bound, bound, bound]));
        assert!(
            !is_opinion(&[bound, clear, bound]),
            "a dropped contradiction is not a stance"
        );
        assert!(!is_opinion(&[]), "empty history is not an opinion");
        assert_eq!(opinion_strength(&[bound, clear, bound]), 2);
    }

    // ── stratified standing wave (E-STANDING-WAVE-IS-UNSTRATIFIED-SUDOKU-1) ──

    /// The load-bearing non-breaking guarantee: adding the settle-pass must not
    /// change ANY verdict. `standing_wave_stratified` is `standing_wave_grounded`
    /// plus instrumentation, never a second opinion.
    #[test]
    fn stratified_never_disagrees_with_grounded() {
        let windows: Vec<Vec<(usize, CausalWitnessFacet)>> = vec![
            // unbound focal
            vec![(0, CausalWitnessFacet::ZERO), (1, CausalWitnessFacet::ZERO)],
            // single-hop terminal chain
            vec![
                (0, w(&[(Locus::Antecedent, 1)])),
                (1, CausalWitnessFacet::ZERO),
            ],
            // two-hop chain that terminates
            vec![
                (0, w(&[(Locus::Antecedent, 1)])),
                (1, w(&[(Locus::Antecedent, 1)])),
                (2, CausalWitnessFacet::ZERO),
            ],
            // chain pointing outside the window → leaves the horizon
            vec![(0, w(&[(Locus::Antecedent, 7)]))],
            // backwards chain
            vec![
                (0, CausalWitnessFacet::ZERO),
                (1, w(&[(Locus::Kausal, -1)])),
            ],
            // empty window (focal_idx out of range)
            vec![],
        ];
        for (i, win) in windows.iter().enumerate() {
            for locus in [Locus::Antecedent, Locus::Kausal, Locus::Temporal] {
                for passes in [1u8, 2, 3, 8] {
                    let flat = standing_wave_grounded(0, win, locus, passes);
                    let (strat, pass) = standing_wave_stratified(0, win, locus, passes);
                    assert_eq!(
                        flat, strat,
                        "window {i} locus {locus:?} passes {passes}: verdict diverged"
                    );
                    // A verdict that resolved something must name a real pass;
                    // Unbound earned no rung and must report 0.
                    match strat {
                        WaveGrounding::Unbound => assert_eq!(pass, 0),
                        _ => assert!(
                            (1..=passes.max(1)).contains(&pass),
                            "window {i}: settle pass {pass} outside 1..={passes}"
                        ),
                    }
                }
            }
        }
    }

    /// The Sudoku payoff: a chain that grounds on the FIRST pass is normalized
    /// to the shallowest rung, so only the cheap gate tactics may fire on it.
    #[test]
    fn cheap_settle_normalizes_to_a_shallow_rung() {
        use crate::cognitive_shader::RungLevel;
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
        ];
        let (grounding, pass) = standing_wave_stratified(0, &win, Locus::Antecedent, 8);
        assert_eq!(grounding, WaveGrounding::Causal);
        // NOT 1: the wave declares Causal only when two SUCCESSIVE budgets
        // resolve the same target, so observing agreement costs a second pass.
        // Pass 2 is therefore the cheapest ground the wave can report, which
        // leaves pass 0 / `Surface` cleanly reserved for "nothing resolved".
        assert_eq!(pass, 2, "cheapest observable ground costs two budgets");
        let rung = RungLevel::for_pass(pass);
        assert_eq!(rung, RungLevel::Shallow);
        // Shallow still admits only the cheap gate bucket — the stratification
        // holds at the cheapest real grounding, which is the case that matters.
        for r in rung.admissible_recipes() {
            assert_eq!(r.bucket, crate::recipes::Bucket::Gate);
        }
        // Rung-0 admits strictly fewer tactics than the full catalogue — the
        // stratification is doing real work, not passing everything through.
        let admissible = rung.admissible_recipes().count();
        assert!(
            admissible < crate::recipes::RECIPES.len(),
            "rung 0 admitted the whole catalogue ({admissible}) — stratification inert"
        );
    }

    #[test]
    fn quorum_mantissa_is_passive_bounded_and_ordered() {
        // No peers → no quorum (not "perfect agreement with nobody").
        assert_eq!(quorum_mantissa(0, &[(0, w(&[(Locus::Kausal, -1)]))]), 0);
        assert_eq!(quorum_mantissa(0, &[]), 0);

        // Two rows pointing Kausal at the SAME absolute event agree; pointing at
        // different events does not. Agreement must order the mantissa.
        let agree = vec![
            (5, w(&[(Locus::Kausal, -3)])), // → event 2
            (7, w(&[(Locus::Kausal, -5)])), // → event 2
        ];
        let disagree = vec![
            (5, w(&[(Locus::Kausal, -3)])), // → event 2
            (7, w(&[(Locus::Kausal, -1)])), // → event 6
        ];
        let q_agree = quorum_mantissa(0, &agree);
        let q_disagree = quorum_mantissa(0, &disagree);
        assert!(
            q_agree > q_disagree,
            "agreement {q_agree} must exceed disagreement {q_disagree}"
        );
        // i4 range — lands in the existing nibble carving, never widened.
        for win in [&agree, &disagree] {
            assert!(quorum_mantissa(0, win) <= 15);
        }
    }

    /// The hindsight guard is STRUCTURAL: the mantissa is a pure function of the
    /// window, so re-running it after any amount of downstream resolution
    /// returns the identical value. Nothing about the outcome can leak in.
    #[test]
    fn quorum_mantissa_cannot_see_the_outcome() {
        let win = vec![
            (5, w(&[(Locus::Kausal, -3), (Locus::Antecedent, 1)])),
            (7, w(&[(Locus::Kausal, -5)])),
            (6, w(&[(Locus::SMeaning, 2)])),
        ];
        let before = quorum_mantissa(0, &win);
        // Resolve things — the "hindsight" a biased implementation would use.
        let _ = standing_wave_stratified(0, &win, Locus::Kausal, 8);
        let _ = elect_peers(0, &win);
        let _ = trajectory_of(0, &win, Locus::Antecedent, 8);
        assert_eq!(
            before,
            quorum_mantissa(0, &win),
            "mantissa moved after resolution — hindsight leaked in"
        );
    }

    #[test]
    fn trajectories_group_by_causal_shape_not_by_target() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
        ];
        let a = trajectory_of(0, &win, Locus::Antecedent, 8);
        let c = trajectory_of(2, &win, Locus::Antecedent, 8);
        // Same causal SHAPE (same hop depth, same escalation) → same meta-basin,
        // even though they terminate on different absolute events.
        assert!(a.same_meta_basin(c));
        // An escalating chain is a DIFFERENT shape, never merged in.
        let far = vec![(0, w(&[(Locus::Antecedent, 7)]))];
        let e = trajectory_of(0, &far, Locus::Antecedent, 8);
        assert!(e.escalated);
        assert!(
            !a.same_meta_basin(e),
            "escalated chain merged into a settled basin"
        );
    }

    // ── version axis (belief-state traversal) ────────────────────────────

    #[test]
    fn revision_trajectory_counts_flips_not_steps() {
        let a = w(&[(Locus::Kausal, -3)]);
        let b = w(&[(Locus::Kausal, -5)]);
        // Calcified: same target every revision → no flips, fully stable.
        let calm = [a, a, a, a];
        let t = revision_trajectory(&calm, Locus::Kausal, usize::MAX);
        assert_eq!((t.steps, t.flips), (4, 0));
        assert_eq!(t.stable_for, 4);
        assert_eq!(t.terminal, Some(-3));
        assert_eq!(t.churn_mantissa(), 0);

        // Churning: alternates every revision → maximal churn.
        let churn = [a, b, a, b];
        let c = revision_trajectory(&churn, Locus::Kausal, usize::MAX);
        assert_eq!((c.steps, c.flips), (4, 3));
        assert_eq!(c.stable_for, 1, "just flipped");
        assert_eq!(c.churn_mantissa(), 15, "3 flips over 3 opportunities");

        // Two beliefs holding the SAME current value can differ entirely in
        // churn — the thing a snapshot cannot see.
        assert_eq!(t.terminal, Some(-3));
        assert_eq!(
            revision_trajectory(&[a, b, a], Locus::Kausal, 3).terminal,
            Some(-3)
        );
        assert!(!t.same_churn_basin(c));
    }

    /// The read-as-of guarantee: a trajectory computed as of revision v is
    /// unaffected by anything appended after v. Hindsight cannot leak in
    /// because the future is not reachable from the call.
    #[test]
    fn read_as_of_cannot_see_later_revisions() {
        let a = w(&[(Locus::Kausal, -3)]);
        let b = w(&[(Locus::Kausal, -5)]);
        let history_then = [a, a, a];
        let history_now = [a, a, a, b, b, a];
        let as_of_3_then = revision_trajectory(&history_then, Locus::Kausal, 3);
        let as_of_3_now = revision_trajectory(&history_now, Locus::Kausal, 3);
        assert_eq!(
            as_of_3_then, as_of_3_now,
            "later revisions leaked into an as-of read"
        );
        // And the full read DOES differ — proving the bound is doing work
        // rather than the two reads being trivially equal.
        let full = revision_trajectory(&history_now, Locus::Kausal, usize::MAX);
        assert_ne!(full, as_of_3_now);
        assert!(full.flips > as_of_3_now.flips);
    }

    #[test]
    fn bind_to_unbind_is_a_flip_and_empty_history_is_not_a_belief() {
        let bound = w(&[(Locus::Kausal, -3)]);
        let clear = CausalWitnessFacet::ZERO;
        // Losing a binding is a change of belief, not a non-event.
        let t = revision_trajectory(&[bound, clear], Locus::Kausal, usize::MAX);
        assert_eq!(t.flips, 1);
        assert_eq!(t.terminal, None, "unbound at the latest revision");
        // Absent ≠ zero: no history yields the default, not a confident zero.
        let empty = revision_trajectory(&[], Locus::Kausal, usize::MAX);
        assert_eq!(empty, RevisionTrajectory::default());
        assert_eq!(empty.steps, 0);
        assert_eq!(empty.churn_mantissa(), 0);
        // A single observation cannot have churned (no denominator).
        assert_eq!(
            revision_trajectory(&[bound], Locus::Kausal, usize::MAX).churn_mantissa(),
            0
        );
    }

    /// Churn is a per-locus reading of ONE register series — a belief may be
    /// calcified on one dimension while churning on another.
    #[test]
    fn churn_is_per_locus_not_per_row() {
        let r1 = w(&[(Locus::Kausal, -3), (Locus::Temporal, 1)]);
        let r2 = w(&[(Locus::Kausal, -3), (Locus::Temporal, 4)]);
        let r3 = w(&[(Locus::Kausal, -3), (Locus::Temporal, 2)]);
        let hist = [r1, r2, r3];
        let kausal = revision_trajectory(&hist, Locus::Kausal, usize::MAX);
        let temporal = revision_trajectory(&hist, Locus::Temporal, usize::MAX);
        assert_eq!(kausal.flips, 0, "cause never moved");
        assert_eq!(temporal.flips, 2, "time reference moved twice");
        assert!(temporal.churn_mantissa() > kausal.churn_mantissa());
    }

    /// **Regression for `E-MULTIPASS-WAS-SINGLE-PASS-1`.** A genuine 2-hop
    /// chain that terminates INSIDE the ±8 horizon must ground, not escalate.
    /// Before the fix the loop returned `Escalate` at budget 1 (where the chain
    /// is truncated) and never tried budget 2 — the budget that resolves it —
    /// so the "multipass" wave was effectively single-pass for every chain
    /// longer than one hop.
    #[test]
    fn multi_hop_chains_actually_get_their_extra_passes() {
        // 0 →(+1) 1 →(+1) 2(terminal). Two hops, wholly inside the window.
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
        ];
        // The underlying chain genuinely resolves once given the budget.
        assert!(resolve_chain(0, &win, Locus::Antecedent, 1).budget_exhausted);
        let r2 = resolve_chain(0, &win, Locus::Antecedent, 2);
        assert!(!r2.escalated());
        assert_eq!(r2.final_offset, Some(2));

        // So the wave must GROUND it, not escalate it.
        let (g, pass) = standing_wave_stratified(0, &win, Locus::Antecedent, 8);
        assert_eq!(
            g,
            WaveGrounding::Causal,
            "2-hop chain inside the horizon escalated — multipass defeated again"
        );
        assert!(pass >= 2, "a 2-hop ground cannot be earned at pass {pass}");
        assert_eq!(standing_wave_grounded(0, &win, Locus::Antecedent, 8), g);

        // A chain that TRULY leaves the horizon still escalates — the fix must
        // not turn genuine non-locality into false grounding.
        let far = vec![(0, w(&[(Locus::Antecedent, 7)]))];
        assert_eq!(
            standing_wave_stratified(0, &far, Locus::Antecedent, 8).0,
            WaveGrounding::Escalate
        );

        // And a budget too small to ever resolve still escalates.
        assert_eq!(
            standing_wave_stratified(0, &win, Locus::Antecedent, 1).0,
            WaveGrounding::Escalate,
            "passes=1 cannot resolve a 2-hop chain and must say so"
        );
    }

    /// The split carrier makes the `E-MULTIPASS-WAS-SINGLE-PASS-1` conflation
    /// UNREPRESENTABLE: the two escalation causes are now distinct fields, and
    /// a chain can be budget-exhausted at one budget and clean at the next
    /// while never being out-of-horizon at either.
    #[test]
    fn escalation_causes_are_distinguishable_not_one_bool() {
        let two_hop = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
        ];
        let r1 = resolve_chain(0, &two_hop, Locus::Antecedent, 1);
        assert!(r1.budget_exhausted, "budget 1 truncates a 2-hop chain");
        assert!(
            !r1.out_of_horizon,
            "…but the chain never left the ±8 window"
        );
        assert!(r1.escalated(), "the v1 reading still sees an escalation");

        let r2 = resolve_chain(0, &two_hop, Locus::Antecedent, 2);
        assert!(
            !r2.escalated(),
            "more budget resolves it — as it always could"
        );

        // A genuine horizon exit is the OTHER cause, and no budget fixes it.
        let far = vec![(0, w(&[(Locus::Antecedent, 7)]))];
        for b in [1u8, 2, 8, 64] {
            let r = resolve_chain(0, &far, Locus::Antecedent, b);
            assert!(
                r.out_of_horizon,
                "budget {b}: horizon exit is budget-independent"
            );
            assert!(!r.budget_exhausted);
        }
    }

    // ── temporal periphery (superseded belief states) ────────────────────

    /// Part 1 — ENUMERABLE, and provably complete: the runs must TILE the
    /// visible history. This is not implied by the code — an off-by-one in the
    /// run-closing logic, or a dropped final revision, leaves a hole or an
    /// overlap that this reconstruction catches.
    #[test]
    fn belief_runs_partition_the_visible_history() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let hist = [v(1), v(1), v(2), CausalWitnessFacet::ZERO, v(2), v(2), v(3)];
        let runs = belief_runs(&hist, Locus::Kausal, usize::MAX);
        // Reconstruct the covered index set from the spans alone.
        let mut covered = vec![0u8; hist.len()];
        for r in &runs {
            let (start, end) = r.span();
            for slot in covered.iter_mut().take(end as usize).skip(start as usize) {
                *slot += 1;
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "runs do not tile the history exactly: {covered:?}"
        );

        // Superseded ∪ current == all, and they are disjoint.
        let superseded = superseded_runs(&hist, Locus::Kausal, usize::MAX);
        assert_eq!(superseded.len() + 1, runs.len());
        assert!(superseded.iter().all(BeliefRun::is_superseded));
        assert!(!runs.last().unwrap().is_superseded());
        assert!(
            !superseded.iter().any(|r| r == runs.last().unwrap()),
            "the current run leaked into the periphery"
        );

        // The flips are NAMED, not merely counted — `RevisionTrajectory` can
        // only report how many there were.
        assert_eq!(superseded[0].superseded_at, Some(2));
        assert_eq!(superseded[0].value, Some(1));
        assert_eq!(superseded[0].held_for, 2);
        let counted = revision_trajectory(&hist, Locus::Kausal, usize::MAX).flips;
        assert!(
            counted > 0 && !superseded.is_empty(),
            "a history with flips must have a non-empty periphery"
        );

        // Absent ≠ zero, in both directions.
        assert!(belief_runs(&[], Locus::Kausal, usize::MAX).is_empty());
        assert!(superseded_runs(&[v(1); 4], Locus::Kausal, usize::MAX).is_empty());
        assert_eq!(belief_runs(&[v(1); 4], Locus::Kausal, usize::MAX).len(), 1);
    }

    /// Part 2 — the sample must SPREAD. Asserting only "it returned k items"
    /// would pass for the recent-edge sampler that re-creates the blindness,
    /// so the test asserts it REACHES the oldest superseded state and is not
    /// the recent-k slice.
    #[test]
    fn superseded_spread_sample_reaches_an_early_revision() {
        // 10 distinct values (all inside the i4 offset range) → 10 runs → 9
        // superseded.
        let hist: Vec<CausalWitnessFacet> = [1i8, 2, 3, 4, 5, 6, 7, -1, -2, -3]
            .iter()
            .map(|&o| w(&[(Locus::Kausal, o)]))
            .collect();
        let periphery = superseded_runs(&hist, Locus::Kausal, usize::MAX);
        assert_eq!(periphery.len(), 9);

        let sample = superseded_spread_sample(&hist, Locus::Kausal, usize::MAX, 3);
        assert_eq!(sample.len(), 3);
        assert_eq!(
            sample[0].first_revision, 0,
            "sample never reached the OLDEST superseded state"
        );
        assert!(
            sample
                .iter()
                .any(|r| r.first_revision as usize >= periphery.len() / 2),
            "sample never reached the recent half either — a spread covers both"
        );
        // Explicitly NOT the cheap (recent) edge.
        let recent_k: Vec<BeliefRun> = periphery[periphery.len() - 3..].to_vec();
        assert_ne!(sample, recent_k, "spread sample degenerated to recent-k");

        // Bounds: asking for more than exists yields what exists, not padding.
        assert_eq!(
            superseded_spread_sample(&hist, Locus::Kausal, usize::MAX, 100).len(),
            9
        );
        assert!(superseded_spread_sample(&hist, Locus::Kausal, usize::MAX, 0).is_empty());
        assert!(superseded_spread_sample(&[], Locus::Kausal, usize::MAX, 5).is_empty());
    }

    /// Part 3 — the escalation channel must be able to FIRE, and must be able
    /// to stay silent. A detector that always fires is as useless as one that
    /// never does.
    #[test]
    fn reopening_fires_on_reversion_and_stays_silent_on_stability() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);

        // FIRES — the history abandoned `1`, then came back to it.
        let reverted = [v(1), v(1), v(1), v(2), v(1)];
        let s = suggest_reopening(&reverted, Locus::Kausal, usize::MAX);
        assert_eq!(s.len(), 1, "expected exactly one suggestion, got {s:?}");
        assert_eq!(s[0].reason, ReopeningReason::Reverted);
        assert_eq!(s[0].run.first_revision, 0, "reported on the EARLIEST run");
        assert_eq!(s[0].evidence.occurrences, 2);
        assert_eq!(s[0].evidence.total_held, 4);
        assert_eq!(s[0].evidence.current_held, 1);

        // FIRES — a 3-revision stable belief traded for a 1-revision one.
        let regressed = [v(5), v(5), v(5), v(6)];
        let s2 = suggest_reopening(&regressed, Locus::Kausal, usize::MAX);
        assert_eq!(s2.len(), 1);
        assert_eq!(s2[0].reason, ReopeningReason::LongStableThenBrief);
        assert_eq!(s2[0].evidence.total_held, 3);

        // SILENT — monotonically stable: nothing was ever superseded.
        assert!(suggest_reopening(&[v(1); 6], Locus::Kausal, usize::MAX).is_empty());
        // SILENT — a single revision, and an empty history.
        assert!(suggest_reopening(&[v(1)], Locus::Kausal, usize::MAX).is_empty());
        assert!(suggest_reopening(&[], Locus::Kausal, usize::MAX).is_empty());
        // SILENT — stability INCREASING (brief → long-stable) is the healthy
        // direction and must not be flagged. This is the case a naive
        // "any flip is suspicious" detector would fire on.
        assert!(
            suggest_reopening(&[v(5), v(6), v(6), v(6)], Locus::Kausal, usize::MAX).is_empty(),
            "flagged a history that got MORE stable"
        );
        // SILENT — an unbound run is not a belief to re-open.
        let unbound_first = [CausalWitnessFacet::ZERO, CausalWitnessFacet::ZERO, v(7)];
        assert!(suggest_reopening(&unbound_first, Locus::Kausal, usize::MAX).is_empty());
    }

    /// Read-as-of on the periphery: a judgement made as of revision v cannot
    /// see the reversion that happens at v+1. Enforced by the signature.
    #[test]
    fn temporal_periphery_cannot_see_later_revisions() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let full = [v(1), v(1), v(1), v(2), v(1)];
        // As of revision 4 the reversion has not happened yet.
        let as_of_4 = suggest_reopening(&full, Locus::Kausal, 4);
        assert_eq!(
            as_of_4.len(),
            1,
            "as-of-4 sees the long-stable trade, not the reversion"
        );
        assert_eq!(as_of_4[0].reason, ReopeningReason::LongStableThenBrief);
        // With the whole history it becomes a reversion — the bound is doing
        // real work, not returning a trivially equal answer.
        let full_view = suggest_reopening(&full, Locus::Kausal, usize::MAX);
        assert_eq!(full_view[0].reason, ReopeningReason::Reverted);
        assert_ne!(as_of_4, full_view);
        // A truncated view is identical to the same-length prefix stored alone.
        let prefix = [v(1), v(1), v(1), v(2)];
        assert_eq!(
            belief_runs(&full, Locus::Kausal, 4),
            belief_runs(&prefix, Locus::Kausal, usize::MAX),
            "later revisions leaked into an as-of run enumeration"
        );
    }

    /// Determinism: same history ⇒ same sample and same suggestions. A
    /// periphery that varied run to run would not be auditable.
    #[test]
    fn temporal_periphery_is_deterministic() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let hist = [v(1), v(2), v(1), v(3), v(3), v(2), v(4), v(1)];
        for upto in [3usize, 5, usize::MAX] {
            let a = superseded_spread_sample(&hist, Locus::Kausal, upto, 3);
            let b = superseded_spread_sample(&hist, Locus::Kausal, upto, 3);
            assert_eq!(a, b);
            let sa = suggest_reopening(&hist, Locus::Kausal, upto);
            let sb = suggest_reopening(&hist, Locus::Kausal, upto);
            assert_eq!(sa, sb);
            // Ordering is oldest→newest, so a consumer diffing two runs sees
            // stable positions.
            assert!(sa
                .windows(2)
                .all(|p| p[0].run.first_revision < p[1].run.first_revision));
        }
        // And the channel is not inert on this window.
        assert!(!suggest_reopening(&hist, Locus::Kausal, usize::MAX).is_empty());
    }

    // ── the Epistemic Foresight Test ──────────────────────────────────────

    /// The risk signal is computed through `upto = v` and CANNOT see the
    /// post-`v` segment. Falsifier: a series that is calm before the split and
    /// wild after it must report LOW risk — an implementation that leaked
    /// hindsight (churn over the whole series) would report 7 here, not 0.
    #[test]
    fn foresight_risk_is_hindsight_blind() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        // Calm-then-wild: risk must not know about the storm.
        let calm_then_wild = [v(2), v(2), v(2), v(2), v(3), v(2), v(3)];
        let s = foresight_sample(&calm_then_wild, Locus::Kausal, 4).unwrap();
        assert_eq!(s.risk_asof_v, 0, "hindsight leaked into the risk signal");
        assert_eq!(s.flips_after, 3);
        assert_eq!(s.steps_after, 3);

        // Wild-then-calm: the sample must honestly record the miscalibrated
        // case (high self-doubt, no later correction) — a function that only
        // ever emitted hypothesis-confirming samples would be worthless as a
        // calibration input.
        let wild_then_calm = [v(2), v(3), v(2), v(3), v(3), v(3), v(3)];
        let s2 = foresight_sample(&wild_then_calm, Locus::Kausal, 4).unwrap();
        assert_eq!(s2.risk_asof_v, 15);
        assert_eq!(s2.flips_after, 0, "the calm future was miscounted");
        assert_eq!(s2.steps_after, 3);
    }

    /// A flip exactly ACROSS the split is counted — the post segment overlaps
    /// the boundary by one revision. Without the overlap this history would
    /// report `flips_after == 0` and the test fails.
    #[test]
    fn foresight_counts_the_boundary_flip() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let hist = [v(2), v(2), v(2), v(5)];
        let s = foresight_sample(&hist, Locus::Kausal, 3).unwrap();
        assert_eq!(s.flips_after, 1, "the cross-boundary flip was dropped");
        assert_eq!(s.steps_after, 1);
        // And the overlap revision itself is not double-counted as a step.
        let no_flip = [v(2), v(2), v(2), v(2)];
        let s2 = foresight_sample(&no_flip, Locus::Kausal, 3).unwrap();
        assert_eq!(s2.flips_after, 0);
        assert_eq!(s2.steps_after, 1);
    }

    /// Absent is not zero on either side of the split: no pre-`v` history has
    /// no self-knowledge to score, no post-`v` history has no outcome.
    #[test]
    fn foresight_refuses_empty_sides() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let hist = [v(1), v(2), v(3)];
        assert_eq!(foresight_sample(&hist, Locus::Kausal, 0), None);
        assert_eq!(foresight_sample(&hist, Locus::Kausal, 3), None);
        assert_eq!(foresight_sample(&hist, Locus::Kausal, 99), None);
        assert_eq!(foresight_sample(&[], Locus::Kausal, 1), None);
        assert_eq!(foresight_sample(&[v(1)], Locus::Kausal, 1), None);
        // Interior splits of the same series ARE scorable.
        assert!(foresight_sample(&hist, Locus::Kausal, 1).is_some());
        assert!(foresight_sample(&hist, Locus::Kausal, 2).is_some());
    }

    /// The calibration bins route by band, keep raw counts, and SKIP
    /// unscorable samples. Falsifier: the `steps_after == 0` sample sits in
    /// the high band — an implementation that counted it would report
    /// `(2, 2, 2)` there instead of `(1, 2, 2)`.
    #[test]
    fn foresight_calibration_bins_exactly_and_skips_unscorable() {
        let s = |risk: u8, flips: u8, steps: u8| ForesightSample {
            risk_asof_v: risk,
            flips_after: flips,
            steps_after: steps,
        };
        let samples = [
            s(2, 1, 4),
            s(5, 0, 2),
            s(8, 3, 3),
            s(15, 2, 2),
            s(12, 0, 0), // not scorable — must NOT become a confident zero
        ];
        let bins = foresight_calibration(&samples);
        assert_eq!(bins[0], (2, 1, 6));
        assert_eq!(bins[1], (1, 3, 3));
        assert_eq!(bins[2], (1, 2, 2));
        // Empty input stays visibly empty — all-zero counts, not a rate.
        assert_eq!(foresight_calibration(&[]), [(0, 0, 0); 3]);
    }

    /// End-to-end: on histories where past instability genuinely predicts
    /// future revision, the calibration curve must SLOPE — high-risk beliefs
    /// flip more per post-step than low-risk ones. This is the PASS condition
    /// of the foresight test actually firing on constructed data, not merely
    /// the plumbing type-checking.
    #[test]
    fn foresight_calibration_slopes_when_churn_predicts_revision() {
        let v = |o: i8| w(&[(Locus::Kausal, o)]);
        let split = 4usize;
        // Three calcified beliefs that stay put, three churning ones that
        // keep churning.
        let histories: Vec<Vec<CausalWitnessFacet>> = vec![
            vec![v(1), v(1), v(1), v(1), v(1), v(1), v(1)],
            vec![v(4), v(4), v(4), v(4), v(4), v(4), v(4)],
            vec![v(-2), v(-2), v(-2), v(-2), v(-2), v(-2), v(-2)],
            vec![v(1), v(2), v(1), v(2), v(1), v(2), v(1)],
            vec![v(3), v(5), v(3), v(5), v(3), v(5), v(3)],
            vec![v(-1), v(2), v(-1), v(2), v(-1), v(2), v(-1)],
        ];
        let samples: Vec<ForesightSample> = histories
            .iter()
            .filter_map(|h| foresight_sample(h, Locus::Kausal, split))
            .collect();
        assert_eq!(samples.len(), 6, "every history must be scorable");
        let bins = foresight_calibration(&samples);
        let (low, high) = (bins[0], bins[2]);
        assert!(low.0 > 0 && high.0 > 0, "both extreme bands must be hit");
        // Flip-rate comparison via cross-multiplication (no float division).
        assert!(
            high.1 * low.2 > low.1 * high.2,
            "high-risk beliefs did not flip more per step: low={low:?} high={high:?}"
        );
        // The low band on THIS data is exactly zero flips — knowable input.
        assert_eq!(low.1, 0);
    }

    /// The diagnosed wave is instrumentation over the stratified one — the
    /// `(grounding, pass)` pair may never differ, and the reason is present
    /// exactly on escalations.
    #[test]
    fn diagnosed_never_disagrees_with_stratified_and_reasons_are_exact() {
        let windows: Vec<Vec<(usize, CausalWitnessFacet)>> = vec![
            vec![(0, CausalWitnessFacet::ZERO)],
            vec![
                (0, w(&[(Locus::Antecedent, 1)])),
                (1, CausalWitnessFacet::ZERO),
            ],
            vec![
                (0, w(&[(Locus::Antecedent, 1)])),
                (1, w(&[(Locus::Antecedent, 1)])),
                (2, CausalWitnessFacet::ZERO),
            ],
            vec![(0, w(&[(Locus::Antecedent, 7)]))],
            vec![],
        ];
        for (i, win) in windows.iter().enumerate() {
            for locus in [Locus::Antecedent, Locus::Kausal] {
                for passes in [1u8, 2, 8] {
                    let (g, p) = standing_wave_stratified(0, win, locus, passes);
                    let (dg, dp, reason) = standing_wave_diagnosed(0, win, locus, passes);
                    assert_eq!((g, p), (dg, dp), "window {i} passes {passes}: diverged");
                    assert_eq!(
                        reason.is_some(),
                        dg == WaveGrounding::Escalate,
                        "window {i}: reason must exist iff Escalate"
                    );
                }
            }
        }
    }

    /// The two escalation reasons demand opposite responses — prove the
    /// diagnosis separates them on real chains.
    #[test]
    fn escalation_reasons_separate_retry_from_representation_switch() {
        // A 2-hop chain with passes=1: slow convergence, NOT non-locality.
        let two_hop = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
        ];
        let (g, _, r) = standing_wave_diagnosed(0, &two_hop, Locus::Antecedent, 1);
        assert_eq!(g, WaveGrounding::Escalate);
        assert_eq!(
            r,
            Some(EscalateReason::BudgetExhausted),
            "budget starvation must say RETRY, not 'do the wide read'"
        );
        // And the retry it recommends actually works:
        let (g2, _, r2) = standing_wave_diagnosed(0, &two_hop, Locus::Antecedent, 8);
        assert_eq!(g2, WaveGrounding::Causal);
        assert_eq!(r2, None);

        // A chain that leaves the window: no budget will ever help.
        let far = vec![(0, w(&[(Locus::Antecedent, 7)]))];
        for passes in [1u8, 8, 200] {
            let (g, _, r) = standing_wave_diagnosed(0, &far, Locus::Antecedent, passes);
            assert_eq!(g, WaveGrounding::Escalate);
            assert_eq!(
                r,
                Some(EscalateReason::OutOfHorizon),
                "horizon exit must not be mistaken for budget starvation at passes={passes}"
            );
        }
    }

    /// An unbound locus earns no rung at all — pass 0, distinct from "grounded
    /// cheaply at pass 1". Absent is not the same as shallow.
    #[test]
    fn unbound_earns_no_pass() {
        let win = vec![(0, CausalWitnessFacet::ZERO)];
        let (g, pass) = standing_wave_stratified(0, &win, Locus::Antecedent, 8);
        assert_eq!(g, WaveGrounding::Unbound);
        assert_eq!(pass, 0);
    }
}
