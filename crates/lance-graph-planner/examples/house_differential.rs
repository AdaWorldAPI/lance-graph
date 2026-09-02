//! **PROBE-HOUSE-DIFFERENTIAL-1 (D-HOUSE-1).**
//! `.claude/plans/house-differential-style-v1.md` §4.
//!
//! ```text
//! cargo run -p lance-graph-planner --example house_differential --release
//! ```
//!
//! # The question
//!
//! Does the House cycle — the board stratum (RCR → admit → ASC on the leader
//! → council) run IN PARALLEL with its peripheral strata (an abstraction
//! stratum below it and a synthesis stratum beside it), scheduled in
//! dependency order — recover a planted cause more often than RCR alone,
//! above a size-preserving shuffle null, on synthetic arenas whose
//! distractors share predicates with the planted cause?
//!
//! # Rungs are strata, not a position
//!
//! The first cut of this probe read `RungLevel` as a single active position
//! and sampled the periphery from `RungLevel::Counterfactual`'s
//! `peripheral_recipes()` — which is EMPTY (no recipe has `min_rung` above
//! Counterfactual), so condition (c) came back structurally unrunnable. That
//! was the scalar-era reading. The ruled model (`persona-vs-rung-ladder.md`,
//! 2026-08-30 scope correction) is that rungs are STRATA scheduled in
//! parallel in dependency order: the board stratum at Counterfactual (RCR #4,
//! ASC #7 — `Tier::ExtremelyHard`, floor raised to 6) runs alongside the
//! strata whose floors are lower — CAS #8 (Gate → Surface, 0) and CR #11
//! (Control → Analogical, 3). "Periphery of the board" therefore means
//! **the other strata**, not "recipes excluded at the board's own rung".
//! `peripheral_sample_where` is still called and its size printed, as a
//! measured value only — it is not what fires.
//!
//! The three strata and their dependency order:
//!
//! ```text
//! S0  CAS-down  {G→O_far, C*→G} ⊢ C*→O_far   (rung 0; needs only the arena)
//! S6  board     RCR → admit top-3 → ASC(leader) → council   (needs S0's admits)
//! S3  CR        synthesize the runner-up's second-opinion report (needs S6's ranking)
//! ```
//!
//! # Pre-registered PASS/KILL rule (plan §4)
//!
//! **PASS iff** (a) `A2 p@1 − A0 p@1 ≥ 0.05` AND `A2 p@1 > AN p95`,
//! (b) the planted cause is eliminated in 0 arenas, (c) the periphery FIRES
//! (changes the board's final ranking) on ≥ 10 % and ≤ 90 % of arenas
//! (can-fire AND can-stay-silent). **KILL** otherwise.
//!
//! # AN — the null, and what "p95" is the p95 OF
//!
//! Per permutation `perm_idx ∈ 0..25`, every arena's `(cause, feature)` rule
//! list has its FEATURE half Fisher-Yates shuffled (cause sequence and the
//! per-slot truth stay put — size-preserving), AND the far parent `G` is
//! re-owned by a uniformly drawn cause instead of `C*` (otherwise the far
//! fact would leak the planted identity straight through the shuffle). The
//! identical A2 procedure runs on each; the aggregate p@1 over the 200 arenas
//! is ONE null value; the 25 values give the null distribution. The
//! pre-registered comparator is the **p95 of those 25 aggregate values**
//! (nearest rank). The earlier cut took the p95 of 200 per-arena hit rates —
//! a per-arena tail statistic, not the distribution of the aggregate being
//! compared — that number is still printed, labelled secondary.
//!
//! # Fixture-design note (documented rather than silently added)
//!
//! Giving every cause the LITERAL same `TruthValue::new(0.9, 0.6)` on every
//! rule edge makes every RCR-abduced `case→cause` candidate for a shared
//! feature numerically identical; ties break ascending by cause id and
//! `C_STAR = 1` is the smallest, so A0's p@1 would be exactly 1.0 by
//! construction. Each rule edge therefore carries a small deterministic
//! jitter (`±0.04` frequency, `±0.06` confidence) drawn from the same
//! per-arena `SplitMix64` stream. The far fact (present on even-indexed
//! arenas only) is deliberately STRONG (`0.98/0.9` on both `C*→G` and
//! `G→O_far`, `0.95/0.9` on `case→O_far`) so that S0's deduction, once
//! admitted, gives RCR a candidate that beats the jittered band — the
//! stratum has to be able to matter for (c) to be a real measurement.
//!
//! # Two variants, both pre-registered, both reported (2026-09-02 run)
//!
//! **Variant 1** (S3 unconditional) — (a) PASS: A2 p@1 0.820 vs A0 0.320
//! (Δ 0.500) and vs AN p95 0.395; (b) PASS: 0 eliminations; (c) **FAIL**:
//! the periphery changed the ranking VECTOR on 182/200 arenas (0.910),
//! because S3 synthesizes a report on every arena whose runner-up is a
//! distractor. **KILL** under the pre-registered rule — and the defect is
//! S3's design, not the cycle's: `A2-S3` (S3 alone) has p@1 identical to
//! `A1c` (0.610), so S3 reorders the tail (177/200) and never the leader.
//! An always-on stratum that never moves p@1 is the "fires on everything"
//! failure mode (CLAUDE.md § falsifiability).
//!
//! **Variant 2** (S3 depends on S6's council: synthesize only on `split`) —
//! pre-registered after variant 1 was written, before it was re-run. (a)
//! PASS (same numbers — the null is identical because the gate never
//! opened under the null either), (b) PASS, (c) PASS at 0.500 (100/200,
//! exactly the far-fact arenas). The pre-registered rule returns **PASS**;
//! the RECORDED result is split in two, because the council split on
//! **0/200** arenas, so S3 never ran: the **base path `S0 + S6` PASSES**,
//! and the **council-gated S3 arm is INCONCLUSIVE** — not exercised, not
//! passed. An S3 PASS needs a fixture on which the council split is
//! reachable. The council does not split here
//! because this probe derives `humility = 1 − margin(top1, top2)`, which
//! sits at ≈ 1 on every board (margins are small), and
//! `InnerCouncil::from_signals` gives Catalyst weight `1 − |humility − 0.5|·2`
//! — zero at that extreme. That is a property of this probe's signal
//! derivation, not a defect in the council; a probe that wants S3 to be
//! reachable needs a humility signal that is not pinned at 1.
//!
//! What the run establishes: the House cycle's gain over RCR alone comes in
//! two steps — ASC on the leader (0.320 → 0.610) and the abstraction stratum
//! S0 (0.610 → 0.820) — and S0 is a real, silence-capable peripheral stratum
//! (fires on exactly the 100 arenas that carry the far fact). What it does
//! NOT establish: any contribution from CR as a runner-up synthesis stratum.
//!
//! **Robustness arms run alongside (not part of the verdict).** With the far
//! fact weakened to the rule-edge band (`0.9/0.6` everywhere) S0's deduced
//! link no longer out-scores the direct abductions and S0 goes inert
//! (fire 0/200, A2 = A1c): the stratum contributes when the abstraction
//! chain carries evidence stronger than the direct band, not otherwise.
//! With the far parent left owned by `C*` under the null (the leak this
//! file's `permuted_far_owner` removes) AN p95 rose 0.475 → 0.735 (measured
//! on the pre-review cut) and A2 still cleared it, by 0.085.
//!
//! **Disable-runs (verified red).** G1 red when S0 is forced on under
//! `STRATA_OFF`; G4 red when it filters `CasUp` instead of `CasDown`; G5 red
//! when the predicate drops its contradiction half; jitter zeroed ⇒ A0 p@1 =
//! 1.000 (the documented tie-break defect).
//!
//! # Two review findings folded in before merge (Codex on #1141)
//!
//! - **S0 was focused on the label.** The first pushed cut called
//!   `cas_abstract(arena, C_STAR, …)` — the hidden answer — in both arms. Now
//!   the focus set is every observable `is_a` subject except the case
//!   ([`s0_focus_subjects`]), the same procedure in the real and null arms.
//!   The real arm is unchanged (only `C*` has a parent with rows of its own,
//!   so the sweep finds the same single deduction); the NULL moved: with the
//!   re-owned far parent now also abstracted, AN p95 fell 0.475 → 0.425
//!   (mean 0.428 → 0.370). The label focus had been starving the null, i.e.
//!   making it lenient, not inflating the real arm — but the procedure was
//!   not applicable without the answer, which is disqualifying on its own.
//! - **The null shuffle admitted duplicate pairs** (CodeRabbit). A plain
//!   feature permutation can give one cause the same feature twice;
//!   `instantiate` then revises that belief with pooled disjoint evidence
//!   the real fixture never has. Distinctness is now enforced by a repaired
//!   shuffle ([`permuted_rule_edges`]; 11 784 repair swaps and 1 base
//!   re-draw over the 5 000 null shuffles — so the defect was frequent, not
//!   theoretical). The null moved again: p95 0.425 → **0.395**, mean 0.370 →
//!   0.343; the per-arena secondary p95 0.640 → 0.800.
//! - **Elimination was read off the ranking.** Condition (b) now reads the
//!   arena belief through [`eliminated_in_arena`] (below floor AND a
//!   contradiction recorded by a disjoint revision), with G5 proving the
//!   predicate two-sided. What (b) still cannot do on THIS fixture: fire on
//!   `C*` — no counter-evidence for the true cause is ever prepared, so 0 is
//!   by construction; G5 is the evidence the predicate would count a real
//!   elimination if one occurred, not evidence that the cycle avoids one.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};

use lance_graph_contract::cognitive_shader::RungLevel;
use lance_graph_contract::counterfactual::deposit_counterfactual;
use lance_graph_contract::escalation::InnerCouncil;
use lance_graph_contract::recipes::recipe;

use lance_graph_planner::dismech_counterfactual::CounterfactualEdge;
use lance_graph_planner::nars::tactics::{ASC_ID, CAS_ID, CR_ID, RCR_ID, TR_ID};
use lance_graph_planner::nars::{
    asc_challenge, cas_abstract, cr_synthesize, rcr_abduce, AscOutcome, BeliefArena, CStmt,
    Candidate, Copula, Frontier, Stamp, Tactic, Throttle, TruthValue,
};

// ─────────────────────────────────────────────────────────────────────────
// Fixture constants (pre-registered, plan §4)
// ─────────────────────────────────────────────────────────────────────────

const N_ARENAS: usize = 200;
const N_PERMS: u32 = 25;
const BASE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Term-id space (u16). Disjoint ranges so no accidental collisions.
const CASE: u16 = 1000;
const C_STAR: u16 = 1;
const DISTRACTOR_IDS: [u16; 5] = [2, 3, 4, 5, 6];
const CAUSE_IDS: [u16; 6] = [1, 2, 3, 4, 5, 6];
const SHARED_FEATURE_START: u16 = 100;
const SHARED_FEATURE_LEN: u16 = 100; // 100..=199
const PRIVATE_FEATURE_START: u16 = 200;
const PRIVATE_FEATURE_LEN: u16 = 100; // 200..=299
/// The far parent `G` — an `is_a` parent of the far fact's owner, reachable
/// only through CAS-down (S0). Never a shared feature, never a cause.
const FAR_PARENT: u16 = 950;
/// The far observation `O_far` — a property of `G` that `case` also shows.
const FAR_FEATURE: u16 = 850;

const HUB_INDEGREE: usize = 32; // > 6 causes + case sharing a feature
const ELIMINATION_FLOOR: f32 = 0.05;
/// Reported only: the size of `RungLevel::Counterfactual`'s sampled
/// periphery under the scalar reading (expected 0 — see the module doc).
const PERIPHERY_K: usize = 4;

/// Base id for the (never-observed) per-distractor counter-evidence stamps —
/// far past the ≤ ~45 real observation stamps any one arena ever uses, so
/// disjointness is guaranteed by construction rather than checked at runtime
/// (the `debug_assert!` in [`instantiate`] is the checkable half of that
/// claim).
const COUNTER_STAMP_BASE: u32 = 50;

/// Which peripheral strata run alongside the board. Both off ⇒ A1c exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Strata {
    /// S0 — CAS-down over `C*`'s `is_a` parents, admitted before the board.
    cas: bool,
    /// S3 — CR synthesis of the runner-up's second-opinion report, after the
    /// board has ranked.
    cr: bool,
    /// S3 depends on S6's COUNCIL as well as its ranking: synthesize the
    /// runner-up's report only when the council verdict was `split`. The
    /// second pre-registered variant (see the module doc).
    cr_on_split: bool,
}

const STRATA_OFF: Strata = Strata {
    cas: false,
    cr: false,
    cr_on_split: false,
};
/// Variant 1 (pre-registered first): S3 unconditional.
const STRATA_ON: Strata = Strata {
    cas: true,
    cr: true,
    cr_on_split: false,
};
/// Variant 2 (pre-registered second, before variant 1 was re-run): S3 gated
/// on the council split.
const STRATA_ON_SPLIT: Strata = Strata {
    cas: true,
    cr: true,
    cr_on_split: true,
};

// ─────────────────────────────────────────────────────────────────────────
// Deterministic PRNG — the fixture's ONLY source of randomness.
// ─────────────────────────────────────────────────────────────────────────

/// SplitMix64. Seeded `0x9E3779B97F4A7C15 ^ i` per arena; permutation
/// streams reuse it seeded `arena_seed ^ (0x9E3779B97F4A7C15 * perm_idx)`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A value in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() >> 1) as usize % n
    }
}

/// Partial Fisher-Yates: `k` distinct elements from `items`, in ascending
/// order (canonical, for reproducible printing/debugging — the SET chosen is
/// still RNG-driven, only its presentation is sorted).
fn choose_k(rng: &mut SplitMix64, items: &[u16], k: usize) -> Vec<u16> {
    let mut pool = items.to_vec();
    let n = pool.len();
    let k = k.min(n);
    for i in 0..k {
        let j = i + rng.below(n - i);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool.sort_unstable();
    pool
}

/// `TruthValue::new(base_f, base_c)` with small deterministic jitter — see
/// the module doc's fixture-design note for why this exists.
fn jittered_truth(
    rng: &mut SplitMix64,
    base_f: f32,
    base_c: f32,
    jit_f: f32,
    jit_c: f32,
) -> TruthValue {
    let df = (rng.next_f32() * 2.0 - 1.0) * jit_f;
    let dc = (rng.next_f32() * 2.0 - 1.0) * jit_c;
    TruthValue::new(base_f + df, base_c + dc)
}

// ─────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────

/// One arena's pre-registered structure. Instantiated fresh into a
/// [`BeliefArena`] per arm (each arm gets its OWN clone — plan §4 "Arms").
struct ArenaFixture {
    seed: u64,
    /// C*'s planted features — also exactly what `case` observes.
    case_features: Vec<u16>,
    case_truths: Vec<TruthValue>,
    /// `(cause, feature)` rule edges, C*'s first then each distractor's, in
    /// creation order. AN's permutation reshuffles the FEATURE half of this
    /// list only (the cause sequence and the truth-per-slot both stay put —
    /// "keep every cause's rule COUNT identical, permute WHICH feature each
    /// rule points to").
    rule_edges: Vec<(u16, u16)>,
    rule_truths: Vec<TruthValue>,
    /// Per-distractor counter-evidence, index-aligned with
    /// [`DISTRACTOR_IDS`]. Held aside — never observed at fixture-build
    /// time. Only [`asc_challenge`] (S6) and [`cr_synthesize`] (S3) consume
    /// it, for two DIFFERENT distractors (leader vs runner-up).
    counter: [(TruthValue, Stamp); 5],
    /// Even-indexed arenas carry the far fact: `owner→G`, `G→O_far`,
    /// `case→O_far`. The owner is `C*` in the real arms and a drawn cause
    /// under the null (see [`permuted_far_owner`]).
    far_fact: bool,
    far_owner_parent_truth: TruthValue,
    far_parent_feature_truth: TruthValue,
    far_case_truth: TruthValue,
}

/// Build arena `idx`'s fixture from its own SplitMix64 stream.
fn build_fixture(idx: usize) -> ArenaFixture {
    let seed = BASE_SEED ^ (idx as u64);
    let mut rng = SplitMix64::new(seed);

    let feat_count = 4 + rng.below(3); // 4, 5, or 6
    let shared_pool: Vec<u16> =
        (SHARED_FEATURE_START..SHARED_FEATURE_START + SHARED_FEATURE_LEN).collect();
    let c_star_features = choose_k(&mut rng, &shared_pool, feat_count);

    let mut rule_edges: Vec<(u16, u16)> = Vec::new();
    let mut rule_truths: Vec<TruthValue> = Vec::new();
    for &f in &c_star_features {
        rule_edges.push((C_STAR, f));
        rule_truths.push(jittered_truth(&mut rng, 0.9, 0.6, 0.04, 0.06));
    }

    let private_pool: Vec<u16> =
        (PRIVATE_FEATURE_START..PRIVATE_FEATURE_START + PRIVATE_FEATURE_LEN).collect();
    let mut counter: [(TruthValue, Stamp); 5] = [(TruthValue::default(), Stamp::default()); 5];

    for (di, &d_id) in DISTRACTOR_IDS.iter().enumerate() {
        let share_count = (1 + rng.below(3)).min(c_star_features.len());
        let shared = choose_k(&mut rng, &c_star_features, share_count);
        let private_count = 2 + rng.below(2);
        let private = choose_k(&mut rng, &private_pool, private_count);
        for &f in shared.iter().chain(private.iter()) {
            rule_edges.push((d_id, f));
            rule_truths.push(jittered_truth(&mut rng, 0.9, 0.6, 0.04, 0.06));
        }
        // Frequency ≤ 0.3, per the plan; fixed at 0.25 (its own literal example).
        counter[di] = (
            TruthValue::new(0.25, 0.6),
            Stamp::source(COUNTER_STAMP_BASE + di as u32),
        );
    }

    let case_truths: Vec<TruthValue> = c_star_features
        .iter()
        .map(|_| TruthValue::new(0.9, 0.6))
        .collect();

    let far_fact = idx.is_multiple_of(2);

    ArenaFixture {
        seed,
        case_features: c_star_features,
        case_truths,
        rule_edges,
        rule_truths,
        counter,
        far_fact,
        far_owner_parent_truth: TruthValue::new(0.98, 0.9),
        far_parent_feature_truth: TruthValue::new(0.98, 0.9),
        far_case_truth: TruthValue::new(0.95, 0.9),
    }
}

/// The permutation stream for `(arena, perm_idx)`: `arena_seed ^ (BASE_SEED * perm_idx)`.
fn perm_rng(fixture: &ArenaFixture, perm_idx: u32) -> SplitMix64 {
    SplitMix64::new(fixture.seed ^ BASE_SEED.wrapping_mul(perm_idx as u64))
}

/// Upper bound on repair swaps within one draw, and on full re-draws of the
/// base permutation when a repair dead-ends. Exceeding either is a fixture
/// defect and panics loudly rather than admitting a duplicate.
const MAX_SHUFFLE_REPAIRS: u32 = 10_000;
const MAX_SHUFFLE_RESTARTS: u32 = 1_000;

/// One null shuffle's bookkeeping: repair swaps applied and base
/// permutations discarded because a repair dead-ended.
#[derive(Clone, Copy, Debug, Default)]
struct ShuffleStats {
    repairs: u32,
    restarts: u32,
}

/// A size-preserving permutation of `fixture.rule_edges`' FEATURE half only
/// (plan §4 "AN"): the cause sequence and per-slot truth stay fixed, the
/// features are Fisher-Yates shuffled among slots.
///
/// **Distinctness is enforced.** A raw shuffle can hand the same feature to
/// two slots of one cause; `instantiate` would then `observe` the identical
/// `CStmt` twice with disjoint stamps and REVISE it — pooling evidence the
/// real fixture never pools, so the null would no longer be size-preserving
/// (CodeRabbit on #1141). Shared features occur up to six times in the pool,
/// so a duplicate-free plain permutation is too rare to reach by rejection
/// (measured: none in 1000 draws on arena 0). Instead the shuffle is
/// REPAIRED: for each duplicate slot, a partner slot of a DIFFERENT cause is
/// drawn from the same stream and the two features are swapped if neither
/// side gains a duplicate. A repair can dead-end (every legal partner is
/// exhausted — measured on arena 6, perm 6); then the base permutation is
/// re-drawn and repaired afresh. The feature multiset and every cause's
/// slot count are unchanged by construction; both counters are returned so
/// the run can report them.
fn permuted_rule_edges(fixture: &ArenaFixture, perm_idx: u32) -> (Vec<(u16, u16)>, ShuffleStats) {
    let mut rng = perm_rng(fixture, perm_idx);
    let causes: Vec<u16> = fixture.rule_edges.iter().map(|&(c, _)| c).collect();
    let base: Vec<u16> = fixture.rule_edges.iter().map(|&(_, f)| f).collect();
    let n = base.len();
    let has = |features: &[u16], cause: u16, f: u16, except: usize| -> bool {
        (0..n).any(|k| k != except && causes[k] == cause && features[k] == f)
    };
    let mut stats = ShuffleStats::default();
    'restart: loop {
        let mut features = base.clone();
        for i in 0..n {
            let j = i + rng.below(n - i);
            features.swap(i, j);
        }
        let mut repairs_this_draw = 0u32;
        loop {
            let Some(i) = (0..n).find(|&i| has(&features, causes[i], features[i], i)) else {
                return (causes.into_iter().zip(features).collect(), stats);
            };
            let mut order: Vec<usize> = (0..n).filter(|&j| causes[j] != causes[i]).collect();
            for k in 0..order.len() {
                let r = k + rng.below(order.len() - k);
                order.swap(k, r);
            }
            let partner = order.into_iter().find(|&j| {
                !has(&features, causes[i], features[j], i)
                    && !has(&features, causes[j], features[i], j)
            });
            let Some(j) = partner else {
                stats.restarts += 1;
                assert!(
                    stats.restarts <= MAX_SHUFFLE_RESTARTS,
                    "arena seed {:#x} perm {perm_idx}: no repairable shuffle in {MAX_SHUFFLE_RESTARTS} draws",
                    fixture.seed
                );
                continue 'restart;
            };
            features.swap(i, j);
            stats.repairs += 1;
            repairs_this_draw += 1;
            assert!(
                repairs_this_draw <= MAX_SHUFFLE_REPAIRS,
                "arena seed {:#x} perm {perm_idx}: repair did not converge in {MAX_SHUFFLE_REPAIRS} swaps",
                fixture.seed
            );
        }
    }
}

/// Under the null the far parent is owned by a uniformly drawn cause. Drawn
/// from its OWN stream (`perm_rng` xor a constant) so the choice does not
/// depend on how many repair swaps [`permuted_rule_edges`] needed.
fn permuted_far_owner(fixture: &ArenaFixture, perm_idx: u32) -> u16 {
    let mut rng = perm_rng(fixture, perm_idx);
    rng.0 ^= 0x5851_F42D_4C95_7F2D;
    CAUSE_IDS[rng.below(CAUSE_IDS.len())]
}

/// Build a fresh [`BeliefArena`] from `fixture`, using the given
/// `(cause, feature)` rule edges (either `fixture.rule_edges` unpermuted, or
/// a [`permuted_rule_edges`] output — same length/index order either way, so
/// `fixture.rule_truths[idx]` stays correctly aligned) and the given far-fact
/// owner (`C_STAR` in the real arms).
///
/// Stamp ids are assigned by a fresh 0-based counter every call, so every
/// observation in the arena is bit-disjoint from every other (S4) — as long
/// as the total observation count stays under [`COUNTER_STAMP_BASE`] (50),
/// which the `debug_assert!` below checks rather than assumes.
fn instantiate(fixture: &ArenaFixture, rule_edges: &[(u16, u16)], far_owner: u16) -> BeliefArena {
    let mut arena = BeliefArena::new();
    let mut next_stamp: u32 = 0;

    for (idx, &(cause, feature)) in rule_edges.iter().enumerate() {
        arena.observe(
            CStmt {
                s: cause,
                cop: Copula::Inh,
                p: feature,
            },
            fixture.rule_truths[idx],
            Stamp::source(next_stamp),
        );
        next_stamp += 1;
    }
    for (idx, &f) in fixture.case_features.iter().enumerate() {
        arena.observe(
            CStmt {
                s: CASE,
                cop: Copula::Inh,
                p: f,
            },
            fixture.case_truths[idx],
            Stamp::source(next_stamp),
        );
        next_stamp += 1;
    }
    if fixture.far_fact {
        arena.observe(
            CStmt {
                s: far_owner,
                cop: Copula::Inh,
                p: FAR_PARENT,
            },
            fixture.far_owner_parent_truth,
            Stamp::source(next_stamp),
        );
        next_stamp += 1;
        arena.observe(
            CStmt {
                s: FAR_PARENT,
                cop: Copula::Inh,
                p: FAR_FEATURE,
            },
            fixture.far_parent_feature_truth,
            Stamp::source(next_stamp),
        );
        next_stamp += 1;
        arena.observe(
            CStmt {
                s: CASE,
                cop: Copula::Inh,
                p: FAR_FEATURE,
            },
            fixture.far_case_truth,
            Stamp::source(next_stamp),
        );
        next_stamp += 1;
    }
    debug_assert!(
        next_stamp < COUNTER_STAMP_BASE,
        "instantiate used {next_stamp} stamps — must stay below COUNTER_STAMP_BASE \
         ({COUNTER_STAMP_BASE}) or counter-evidence stamps are no longer guaranteed disjoint"
    );
    arena
}

/// The held-aside counter-evidence for a DISTRACTOR; `None` for `C*`.
fn counter_for(fixture: &ArenaFixture, cause: u16) -> Option<(TruthValue, Stamp)> {
    DISTRACTOR_IDS
        .iter()
        .position(|&d| d == cause)
        .map(|i| fixture.counter[i])
}

// ─────────────────────────────────────────────────────────────────────────
// Ranking — "the board = the candidates with stmt.s == case, ranked by
// truth.expectation()" (plan §2 step 1), ties broken ascending cause id.
// ─────────────────────────────────────────────────────────────────────────

/// The board read off the ARENA: `case→cause` beliefs by expectation desc, cause id asc.
fn rank_arena(arena: &BeliefArena, case: u16) -> Vec<(u16, f32)> {
    let mut v: Vec<(u16, f32)> = arena
        .entries()
        .iter()
        .filter(|b| b.stmt.s == case && b.stmt.cop == Copula::Inh && CAUSE_IDS.contains(&b.stmt.p))
        .map(|b| (b.stmt.p, b.truth.expectation()))
        .collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    v
}

/// One `Candidate` per cause (the max-expectation one, if `rcr_abduce`
/// proposed several via different shared middle terms), sorted the same way
/// as [`rank_arena`]. Deterministic: `BTreeMap` keeps insertion order out of
/// it entirely.
fn best_candidates(frontier: &Frontier, case: u16) -> Vec<Candidate> {
    let mut map: BTreeMap<u16, Candidate> = BTreeMap::new();
    for c in &frontier.candidates {
        if c.stmt.s == case && c.stmt.cop == Copula::Inh && CAUSE_IDS.contains(&c.stmt.p) {
            map.entry(c.stmt.p)
                .and_modify(|cur| {
                    if c.truth.expectation() > cur.truth.expectation() {
                        *cur = *c;
                    }
                })
                .or_insert(*c);
        }
    }
    let mut v: Vec<Candidate> = map.into_values().collect();
    v.sort_by(|a, b| {
        b.truth
            .expectation()
            .partial_cmp(&a.truth.expectation())
            .unwrap_or(Ordering::Equal)
            .then(a.stmt.p.cmp(&b.stmt.p))
    });
    v
}

/// `C*` is ranked first.
fn hits_at_1(ranked: &[(u16, f32)]) -> bool {
    ranked.first().map(|&(c, _)| c == C_STAR).unwrap_or(false)
}
/// `C*` is within the top three.
fn hits_at_3(ranked: &[(u16, f32)]) -> bool {
    ranked.iter().take(3).any(|&(c, _)| c == C_STAR)
}

// ─────────────────────────────────────────────────────────────────────────
// Arms
// ─────────────────────────────────────────────────────────────────────────

struct A0Result {
    ranked: Vec<(u16, f32)>,
    board_candidate_count: usize,
}

/// **A0** — RCR alone.
fn run_a0(fixture: &ArenaFixture, throttle: &Throttle) -> A0Result {
    let arena = instantiate(fixture, &fixture.rule_edges, C_STAR);
    let frontier = rcr_abduce(&arena, throttle);
    let ranked = best_candidates(&frontier, CASE)
        .into_iter()
        .map(|c| (c.stmt.p, c.truth.expectation()))
        .collect();
    A0Result {
        ranked,
        board_candidate_count: frontier.candidates.len(),
    }
}

struct BoardResult {
    ranked: Vec<(u16, f32)>,
    asc_outcome: Option<AscOutcome>,
    board_candidate_count: usize,
}

/// S6 steps 1 (board), 1b (admit top-3), 3 (challenge the leader) — shared
/// by A1, A1c (via [`run_a1c`]), A2, and every AN permutation.
///
/// The leader is challenged ONLY when it is a distractor (never `C_STAR` —
/// the fixture prepares counter-evidence for distractors only; there is
/// deliberately nothing to challenge the true cause with). Re-ranks by
/// reading the ARENA afterward, not the frontier — the arena is what the
/// challenge mutated.
fn run_board_admit_challenge(
    arena: &mut BeliefArena,
    fixture: &ArenaFixture,
    throttle: &Throttle,
) -> BoardResult {
    let frontier = rcr_abduce(arena, throttle);
    let board_candidate_count = frontier.candidates.len();
    let ranked_candidates = best_candidates(&frontier, CASE);

    for c in ranked_candidates.iter().take(3) {
        arena.admit_derived(c.stmt, c.truth, &c.premises, c.rung);
    }

    let leader = ranked_candidates.first().map(|c| c.stmt.p);
    let mut asc_outcome = None;
    if let Some(leader_cause) = leader {
        if leader_cause != C_STAR {
            if let Some((counter_truth, counter_stamp)) = counter_for(fixture, leader_cause) {
                let target = CStmt {
                    s: CASE,
                    cop: Copula::Inh,
                    p: leader_cause,
                };
                asc_outcome = Some(asc_challenge(arena, target, counter_truth, counter_stamp));
            }
        }
    }

    let ranked = rank_arena(arena, CASE);
    BoardResult {
        ranked,
        asc_outcome,
        board_candidate_count,
    }
}

/// **A1c** — the whole board stratum S6: A1 (steps 1, 1b, 3) + council
/// (step 4). No peripheral strata.
///
/// Signals: `trust` = top-1 expectation, `humility` = `1 - margin(top1,
/// top2)`, `flow`/`load` from the board's raw candidate count. On a split
/// verdict the RUNNER-UP is deposited as a −6 counterfactual via
/// [`CounterfactualEdge`] (the shipped `impl EpisodicEdge`, not a new one) —
/// never the majority pole, and never written as observed SPO truth.
///
/// Asserts the council never reorders the ranking (plan §4b):
/// `deposit_counterfactual` only ever touches the episodic edge it is
/// handed, never `arena`. Returns the ranking and whether the council split
/// (the only council output a downstream stratum may depend on).
fn run_a1c(
    arena: &mut BeliefArena,
    fixture: &ArenaFixture,
    throttle: &Throttle,
) -> (Vec<(u16, f32)>, bool) {
    let board = run_board_admit_challenge(arena, fixture, throttle);
    let ranking_before = board.ranked.clone();

    let top1 = ranking_before.first().map(|&(_, e)| e).unwrap_or(0.5);
    let top2 = ranking_before.get(1).map(|&(_, e)| e).unwrap_or(top1);
    let margin = (top1 - top2).max(0.0);
    let trust = top1.clamp(0.0, 1.0);
    let humility = (1.0 - margin).clamp(0.0, 1.0);
    let cc = board.board_candidate_count as f32;
    let flow = (1.0 / (1.0 + cc)).clamp(0.0, 1.0);
    let load = (cc / 64.0).clamp(0.0, 1.0);
    let verdict = InnerCouncil::from_signals(trust, humility, flow, load);

    if verdict.split {
        if let Some(&(runner_up, runner_exp)) = ranking_before.get(1) {
            // Truncation to u8 is only for the demonstration edge's s/p
            // indices — CausalEdge64::pack needs u8, our term space is u16.
            let case_u8 = (CASE % 256) as u8;
            let runner_u8 = (runner_up % 256) as u8;
            let freq_u8 = (runner_exp.clamp(0.0, 1.0) * 255.0).round() as u8;
            let conf_u8 = (verdict.confidence.clamp(0.0, 1.0) * 255.0).round() as u8;
            let raw = CausalEdge64::pack(
                case_u8,
                runner_u8,
                0,
                freq_u8,
                conf_u8,
                CausalMask::PO,
                0b101,
                InferenceType::Abduction,
                PlasticityState::S_HOT,
                0,
            );
            let mut wrapped = CounterfactualEdge(raw);
            deposit_counterfactual(&verdict, &mut wrapped);
        }
    }

    let ranking_after = rank_arena(arena, CASE);
    assert_eq!(
        ranking_before, ranking_after,
        "council must never reorder the ranking — deposit_counterfactual may only touch the \
         episodic edge, never `arena`"
    );

    (ranking_after, verdict.split)
}

struct A2Outcome {
    ranked: Vec<(u16, f32)>,
    /// S0 admitted ≥ 1 CAS-down candidate.
    s0_admitted: usize,
    /// S3 synthesized the runner-up's report.
    s3_ran: bool,
    /// S6's council verdict split.
    split: bool,
    /// S3 changed the ORDER of the board (not merely an expectation).
    s3_reordered: bool,
}

/// The abstraction stratum's focus set: every subject that carries an `is_a`
/// row in the arena, except the case itself. Derived from what is OBSERVABLE
/// in the arena — never from the planted label (Codex P1 on #1141: focusing
/// `cas_abstract` on `C_STAR` seeded S0 with the answer in the real arm and
/// starved it under the null). The same procedure runs in both arms.
fn s0_focus_subjects(arena: &BeliefArena) -> Vec<u16> {
    let mut v: Vec<u16> = arena
        .entries()
        .iter()
        .filter(|b| b.stmt.cop == Copula::Inh && b.stmt.s != CASE)
        .map(|b| b.stmt.s)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The plan's elimination predicate (§2 step 6), read off the ARENA belief,
/// never off a `(cause, expectation)` pair: `case→cause` is present, its
/// expectation is below [`ELIMINATION_FLOOR`], AND a contradiction depth is
/// recorded on it — which the arena only ever records through a revision
/// with DISJOINT stamps (`BeliefArena::observe` → `Revised`), i.e. an
/// independently-sourced challenge. A low belief with no recorded
/// contradiction is "ranked low", not eliminated (Codex P2 on #1141).
fn eliminated_in_arena(arena: &BeliefArena, cause: u16) -> bool {
    arena
        .get(CStmt {
            s: CASE,
            cop: Copula::Inh,
            p: cause,
        })
        .map(|b| b.truth.expectation() < ELIMINATION_FLOOR && b.contradiction > 0.0)
        .unwrap_or(false)
}

/// **A2** — the parallel strata, in dependency order:
///
/// 1. **S0 (CAS #8, rung floor 0)** — `cas_abstract` over EVERY observable
///    `is_a` subject ([`s0_focus_subjects`]); every `CasDown` candidate
///    (`{G→P, S→G} ⊢ S→P`) is admitted.
///    Only the DOWN half is admitted: the UP half (`{C*→P, C*→G} ⊢ G→P`)
///    mints `feature→feature` inductions that are noise for a `case→cause`
///    board. On arenas without the far fact every parent `G` is a shared
///    feature with no `is_a` rows of its own, so there is nothing to deduce
///    and S0 admits nothing — the stratum's silence half.
/// 2. **S6 (board)** — [`run_a1c`] on the arena S0 left behind.
/// 3. **S3 (CR #11, rung floor 3)** — the runner-up's second-opinion report
///    (its held-aside counter-evidence) is synthesized via
///    [`cr_synthesize`], then the board is re-read. Skipped when the
///    runner-up is `C*` (no report exists for the true cause).
///
/// With both strata off this IS [`run_a1c`] (guard G1).
fn run_a2(
    arena: &mut BeliefArena,
    fixture: &ArenaFixture,
    throttle: &Throttle,
    strata: Strata,
) -> A2Outcome {
    let mut s0_admitted = 0usize;
    if strata.cas {
        for focus in s0_focus_subjects(arena) {
            let fr = cas_abstract(arena, focus, throttle);
            for c in fr.candidates.iter().filter(|c| c.tactic == Tactic::CasDown) {
                arena.admit_derived(c.stmt, c.truth, &c.premises, c.rung);
                s0_admitted += 1;
            }
        }
    }

    let (mut ranked, split) = run_a1c(arena, fixture, throttle);
    let board_order: Vec<u16> = ranked.iter().map(|&(c, _)| c).collect();

    let mut s3_ran = false;
    if strata.cr && (!strata.cr_on_split || split) {
        if let Some(&(runner_up, _)) = ranked.get(1) {
            if runner_up != C_STAR {
                if let Some((report_truth, report_stamp)) = counter_for(fixture, runner_up) {
                    let stmt = CStmt {
                        s: CASE,
                        cop: Copula::Inh,
                        p: runner_up,
                    };
                    let _ = cr_synthesize(arena, stmt, report_truth, report_stamp);
                    s3_ran = true;
                    ranked = rank_arena(arena, CASE);
                }
            }
        }
    }

    let s3_reordered = s3_ran
        && ranked
            .iter()
            .map(|&(c, _)| c)
            .ne(board_order.iter().copied());
    A2Outcome {
        ranked,
        s0_admitted,
        s3_ran,
        split,
        s3_reordered,
    }
}

/// AN — ONE permutation's aggregate p@1 over all arenas, running the
/// identical A2 procedure (the given strata) on the shuffled fixture with the far
/// parent re-owned. Also returns the per-arena hit vector for the secondary
/// (per-arena) statistic and the number of duplicate-pair repair swaps.
fn run_an_for_perm(
    fixtures: &[ArenaFixture],
    throttle: &Throttle,
    perm_idx: u32,
    strata: Strata,
) -> (f64, Vec<bool>, ShuffleStats) {
    let mut hits: Vec<bool> = Vec::with_capacity(fixtures.len());
    let mut shuffle = ShuffleStats::default();
    for fixture in fixtures {
        let (edges, st) = permuted_rule_edges(fixture, perm_idx);
        shuffle.repairs += st.repairs;
        shuffle.restarts += st.restarts;
        let owner = permuted_far_owner(fixture, perm_idx);
        let mut arena = instantiate(fixture, &edges, owner);
        let out = run_a2(&mut arena, fixture, throttle, strata);
        hits.push(hits_at_1(&out.ranked));
    }
    let n = hits.len().max(1) as f64;
    let p = hits.iter().filter(|&&h| h).count() as f64 / n;
    (p, hits, shuffle)
}

/// The cause ORDER of a ranking, expectations dropped.
fn order_of(ranked: &[(u16, f32)]) -> Vec<u16> {
    ranked.iter().map(|&(c, _)| c).collect()
}

struct NullSummary {
    aggregates: Vec<f64>,
    mean: f64,
    /// The pre-registered comparator: p95 of the per-permutation aggregates.
    p95: f64,
    max: f64,
    /// Secondary: p95 of the per-arena hit rates (the earlier cut's number).
    per_arena_p95: f64,
    /// Repair swaps / base re-draws used to keep every `(cause, feature)` pair distinct.
    shuffle: ShuffleStats,
}

/// The full null: `N_PERMS` permutations of the given strata over every arena.
fn run_null(fixtures: &[ArenaFixture], throttle: &Throttle, strata: Strata) -> NullSummary {
    let mut aggregates: Vec<f64> = Vec::with_capacity(N_PERMS as usize);
    let mut per_arena_hits: Vec<u32> = vec![0; fixtures.len()];
    let mut shuffle = ShuffleStats::default();
    for perm_idx in 0..N_PERMS {
        let (p, hits, st) = run_an_for_perm(fixtures, throttle, perm_idx, strata);
        shuffle.repairs += st.repairs;
        shuffle.restarts += st.restarts;
        aggregates.push(p);
        for (i, h) in hits.iter().enumerate() {
            if *h {
                per_arena_hits[i] += 1;
            }
        }
    }
    let mean = aggregates.iter().sum::<f64>() / N_PERMS as f64;
    let mut sorted_agg = aggregates.clone();
    sorted_agg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p95 = percentile95(&sorted_agg);
    let max = sorted_agg.last().copied().unwrap_or(0.0);
    let mut per_arena_rates: Vec<f64> = per_arena_hits
        .iter()
        .map(|&h| f64::from(h) / f64::from(N_PERMS))
        .collect();
    per_arena_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let per_arena_p95 = percentile95(&per_arena_rates);
    NullSummary {
        aggregates,
        mean,
        p95,
        max,
        per_arena_p95,
        shuffle,
    }
}

/// 95th percentile, nearest-rank method. `sorted` must already be ascending.
fn percentile95(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let rank = ((0.95 * n as f64).ceil() as usize).max(1);
    sorted[(rank - 1).min(n - 1)]
}

// ─────────────────────────────────────────────────────────────────────────
// Guards — each is a real falsifier, disable-verified by the orchestrator;
// this file only computes the PASS/FAIL boolean honestly.
// ─────────────────────────────────────────────────────────────────────────

/// **G1** — with both strata OFF, A2's ranking is element-for-element
/// identical to A1c's on a far-fact arena (arena 0).
fn guard_g1(throttle: &Throttle) -> bool {
    let fixture = build_fixture(0);
    let mut arena_a1c = instantiate(&fixture, &fixture.rule_edges, C_STAR);
    let (a1c_ranked, _) = run_a1c(&mut arena_a1c, &fixture, throttle);

    let mut arena_a2 = instantiate(&fixture, &fixture.rule_edges, C_STAR);
    let a2 = run_a2(&mut arena_a2, &fixture, throttle, STRATA_OFF);

    a2.s0_admitted == 0 && !a2.s3_ran && a2.ranked == a1c_ranked
}

/// **G2** — on arena 0, `rcr_abduce` produces at least one candidate with
/// `stmt.s == case && stmt.p == C_STAR` (the RCR direction the whole probe
/// depends on — see plan §2 step 1).
fn guard_g2(throttle: &Throttle) -> bool {
    let fixture = build_fixture(0);
    let arena = instantiate(&fixture, &fixture.rule_edges, C_STAR);
    let frontier = rcr_abduce(&arena, throttle);
    frontier
        .candidates
        .iter()
        .any(|c| c.stmt.s == CASE && c.stmt.p == C_STAR && c.stmt.cop == Copula::Inh)
}

/// **G3** — calling `asc_challenge` on the top candidate WITHOUT
/// `admit_derived` first returns `AscOutcome::NoTarget`; after
/// `admit_derived` it does not.
fn guard_g3(throttle: &Throttle) -> bool {
    let fixture = build_fixture(0);
    let mut arena = instantiate(&fixture, &fixture.rule_edges, C_STAR);
    let frontier = rcr_abduce(&arena, throttle);
    let top = best_candidates(&frontier, CASE);
    let Some(leader) = top.first() else {
        return false;
    };
    let target = leader.stmt;
    let (counter_truth, counter_stamp) = counter_for(&fixture, leader.stmt.p)
        .unwrap_or((TruthValue::new(0.1, 0.6), Stamp::source(60)));

    let before = asc_challenge(&mut arena, target, counter_truth, counter_stamp);
    let before_no_target = matches!(before, AscOutcome::NoTarget);

    arena.admit_derived(leader.stmt, leader.truth, &leader.premises, leader.rung);
    // A fresh disjoint stamp — `before` may already have consumed
    // `counter_stamp` had it not been NoTarget; using a new one keeps this
    // half of the guard independent of that.
    let after = asc_challenge(&mut arena, target, counter_truth, Stamp::source(61));
    let after_no_target = matches!(after, AscOutcome::NoTarget);

    before_no_target && !after_no_target
}

/// **G4** — the far fact is reachable through CAS-down ONLY, under the SAME
/// label-blind focus procedure A2 uses: on an even-indexed (far-fact) arena
/// the whole focus sweep yields ≥ 1 `CasDown` candidate and every one is
/// exactly `C*→O_far` (no other subject has a parent with rows of its own);
/// on an odd-indexed arena it yields 0 `CasDown` candidates (the UP half may
/// fire on both — that is why the guard filters on the tactic, not on
/// emptiness).
fn guard_g4(throttle: &Throttle) -> bool {
    let even = build_fixture(0);
    let odd = build_fixture(1);
    assert!(
        even.far_fact && !odd.far_fact,
        "fixture parity invariant broken"
    );

    let arena_even = instantiate(&even, &even.rule_edges, C_STAR);
    let arena_odd = instantiate(&odd, &odd.rule_edges, C_STAR);

    let down_of = |arena: &BeliefArena| -> Vec<CStmt> {
        s0_focus_subjects(arena)
            .into_iter()
            .flat_map(|focus| {
                cas_abstract(arena, focus, throttle)
                    .candidates
                    .into_iter()
                    .filter(|c| c.tactic == Tactic::CasDown)
                    .map(|c| c.stmt)
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let down_even = down_of(&arena_even);
    let down_odd = down_of(&arena_odd).len();

    let expected = CStmt {
        s: C_STAR,
        cop: Copula::Inh,
        p: FAR_FEATURE,
    };
    !down_even.is_empty() && down_even.iter().all(|&s| s == expected) && down_odd == 0
}

/// **G5** — the elimination predicate is two-sided: it FIRES on a belief
/// driven below the floor by repeated DISJOINT challenges (contradiction
/// recorded), and STAYS SILENT on a belief that is merely observed low with
/// no challenge on record (same expectation band, no contradiction).
fn guard_g5() -> bool {
    let d = DISTRACTOR_IDS[0];
    let stmt = CStmt {
        s: CASE,
        cop: Copula::Inh,
        p: d,
    };

    let mut challenged = BeliefArena::new();
    challenged.observe(stmt, TruthValue::new(0.9, 0.6), Stamp::source(0));
    for k in 1..=3u32 {
        challenged.observe(stmt, TruthValue::new(0.0, 0.95), Stamp::source(k));
    }
    let fires = eliminated_in_arena(&challenged, d);
    let challenged_low = challenged
        .get(stmt)
        .map(|b| b.truth.expectation() < ELIMINATION_FLOOR)
        .unwrap_or(false);

    let mut unchallenged = BeliefArena::new();
    unchallenged.observe(stmt, TruthValue::new(0.0, 0.95), Stamp::source(0));
    let silent = !eliminated_in_arena(&unchallenged, d);
    let unchallenged_low = unchallenged
        .get(stmt)
        .map(|b| b.truth.expectation() < ELIMINATION_FLOOR)
        .unwrap_or(false);

    // Anti-vacuity: BOTH beliefs must sit below the floor, so the predicate
    // is discriminating on the contradiction half alone.
    challenged_low && unchallenged_low && fires && silent
}

/// `PASS` / `FAIL` for a guard boolean.
fn pf(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────

struct ArmMetrics {
    p_at_1: f64,
    p_at_3: f64,
    /// Arenas on which [`eliminated_in_arena`] held for `C*` after the arm
    /// ran — read from the arena belief, not from the ranking.
    elimination_fp: usize,
}

/// One arm's per-arena record: the final ranking and whether `C*` met the
/// elimination predicate in that arm's arena.
type ArmRun = (Vec<(u16, f32)>, bool);

/// p@1 / p@3 / elimination count over one arm's runs.
fn arm_metrics(runs: &[ArmRun]) -> ArmMetrics {
    let n = runs.len().max(1);
    let mut hits1 = 0usize;
    let mut hits3 = 0usize;
    let mut elim = 0usize;
    for (r, eliminated) in runs {
        if hits_at_1(r) {
            hits1 += 1;
        }
        if hits_at_3(r) {
            hits3 += 1;
        }
        if *eliminated {
            elim += 1;
        }
    }
    ArmMetrics {
        p_at_1: hits1 as f64 / n as f64,
        p_at_3: hits3 as f64 / n as f64,
        elimination_fp: elim,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────

/// Runs guards, every arm, both nulls, the structural table and both verdicts.
fn main() {
    let throttle = Throttle::new(0.0, 4096, HUB_INDEGREE);

    println!("PROBE-HOUSE-DIFFERENTIAL-1 (D-HOUSE-1)");
    println!("fixture: N={N_ARENAS} arenas, N_PERMS={N_PERMS}, base_seed=0x{BASE_SEED:016X}");
    println!(
        "throttle: c_min={:.2} budget={} hub_indegree={}",
        throttle.c_min, throttle.budget, throttle.hub_indegree
    );
    println!(
        "terms: case={CASE} c_star={C_STAR} distractors={DISTRACTOR_IDS:?} far_parent={FAR_PARENT} far_feature={FAR_FEATURE}"
    );
    println!();

    // ── GUARDS ─────────────────────────────────────────────────────────
    println!("== GUARDS ==");
    let g1 = guard_g1(&throttle);
    println!("G1 strata_off_implies_a2_equals_a1c       : {}", pf(g1));
    let g2 = guard_g2(&throttle);
    println!("G2 fixture_direction_yields_case_to_cause : {}", pf(g2));
    let g3 = guard_g3(&throttle);
    println!("G3 admission_is_load_bearing              : {}", pf(g3));
    let g4 = guard_g4(&throttle);
    println!("G4 far_fact_is_cas_down_only              : {}", pf(g4));
    let g5 = guard_g5();
    println!("G5 elimination_predicate_two_sided        : {}", pf(g5));
    if !(g1 && g2 && g3 && g4 && g5) {
        println!(
            "NOTE: one or more anti-vacuity guards failed — no metric can be trusted until the \
             failing guard is fixed; exiting non-zero so a red guard is never read as a verdict."
        );
        std::process::exit(1);
    }
    println!();

    // ── run every arm across the N_ARENAS arenas ───────────────────────
    let fixtures: Vec<ArenaFixture> = (0..N_ARENAS).map(build_fixture).collect();

    let mut a0_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a1_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a1c_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a2s0_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a2s3_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a2_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut a2v_r: Vec<ArmRun> = Vec::with_capacity(N_ARENAS);
    let mut candidate_counts: Vec<usize> = Vec::with_capacity(N_ARENAS);
    let mut periphery_fires = 0usize;
    let mut periphery_reorders = 0usize;
    let mut s0_admits_arenas = 0usize;
    let mut s3_ran_arenas = 0usize;
    let mut s3_reordered_arenas = 0usize;
    let mut council_splits = 0usize;
    let mut v_periphery_fires = 0usize;
    let mut v_periphery_reorders = 0usize;
    let mut v_s3_ran_arenas = 0usize;
    let mut far_fact_count = 0usize;
    let mut asc_revised = 0usize;
    let mut asc_blocked = 0usize;
    let mut asc_no_target = 0usize;
    let mut asc_skipped = 0usize;

    for fixture in &fixtures {
        if fixture.far_fact {
            far_fact_count += 1;
        }

        let a0 = run_a0(fixture, &throttle);
        candidate_counts.push(a0.board_candidate_count);
        // A0 never admits or challenges anything: no arena belief `case→C*`
        // exists, so the predicate is false by construction.
        a0_r.push((a0.ranked, false));

        let mut arena1 = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let a1 = run_board_admit_challenge(&mut arena1, fixture, &throttle);
        match a1.asc_outcome {
            None => asc_skipped += 1,
            Some(AscOutcome::Revised { .. }) => asc_revised += 1,
            Some(AscOutcome::BlockedSelfReference) => asc_blocked += 1,
            Some(AscOutcome::NoTarget) => asc_no_target += 1,
        }
        a1_r.push((a1.ranked, eliminated_in_arena(&arena1, C_STAR)));

        let mut arena1c = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let (a1c_ranked, _) = run_a1c(&mut arena1c, fixture, &throttle);
        let a1c_elim = eliminated_in_arena(&arena1c, C_STAR);

        let mut arena_s0 = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let a2s0 = run_a2(
            &mut arena_s0,
            fixture,
            &throttle,
            Strata {
                cas: true,
                cr: false,
                cr_on_split: false,
            },
        );
        a2s0_r.push((a2s0.ranked, eliminated_in_arena(&arena_s0, C_STAR)));

        let mut arena_s3 = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let a2s3 = run_a2(
            &mut arena_s3,
            fixture,
            &throttle,
            Strata {
                cas: false,
                cr: true,
                cr_on_split: false,
            },
        );
        a2s3_r.push((a2s3.ranked, eliminated_in_arena(&arena_s3, C_STAR)));

        let mut arena2 = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let a2 = run_a2(&mut arena2, fixture, &throttle, STRATA_ON);
        if a2.s0_admitted > 0 {
            s0_admits_arenas += 1;
        }
        if a2.s3_ran {
            s3_ran_arenas += 1;
        }
        if a2.s3_reordered {
            s3_reordered_arenas += 1;
        }
        if a2.split {
            council_splits += 1;
        }
        // "Fires" (pre-registered) = the peripheral strata CHANGED the
        // board's final ranking vector `(cause, expectation)`. "Reorders"
        // (diagnostic) = changed the ORDER of causes.
        if a2.ranked != a1c_ranked {
            periphery_fires += 1;
        }
        if order_of(&a2.ranked) != order_of(&a1c_ranked) {
            periphery_reorders += 1;
        }
        a2_r.push((a2.ranked, eliminated_in_arena(&arena2, C_STAR)));

        let mut arena2v = instantiate(fixture, &fixture.rule_edges, C_STAR);
        let a2v = run_a2(&mut arena2v, fixture, &throttle, STRATA_ON_SPLIT);
        if a2v.s3_ran {
            v_s3_ran_arenas += 1;
        }
        if a2v.ranked != a1c_ranked {
            v_periphery_fires += 1;
        }
        if order_of(&a2v.ranked) != order_of(&a1c_ranked) {
            v_periphery_reorders += 1;
        }
        a2v_r.push((a2v.ranked, eliminated_in_arena(&arena2v, C_STAR)));
        a1c_r.push((a1c_ranked, a1c_elim));
    }

    // ── AN — the size-preserving null, identical A2 procedure, per variant
    let an = run_null(&fixtures, &throttle, STRATA_ON);
    let anv = run_null(&fixtures, &throttle, STRATA_ON_SPLIT);

    // ── METRICS ────────────────────────────────────────────────────────
    println!(
        "== METRICS (N={N_ARENAS} arenas; far-fact present on {far_fact_count}/{N_ARENAS}) =="
    );
    let m_a0 = arm_metrics(&a0_r);
    let m_a1 = arm_metrics(&a1_r);
    let m_a1c = arm_metrics(&a1c_r);
    let m_a2s0 = arm_metrics(&a2s0_r);
    let m_a2s3 = arm_metrics(&a2s3_r);
    let m_a2 = arm_metrics(&a2_r);
    let m_a2v = arm_metrics(&a2v_r);
    let mean_cc = candidate_counts.iter().sum::<usize>() as f64 / N_ARENAS as f64;
    let periphery_rate = periphery_fires as f64 / N_ARENAS as f64;
    let reorder_rate = periphery_reorders as f64 / N_ARENAS as f64;
    let v_periphery_rate = v_periphery_fires as f64 / N_ARENAS as f64;
    let v_reorder_rate = v_periphery_reorders as f64 / N_ARENAS as f64;
    let scalar_periphery = RungLevel::Counterfactual
        .peripheral_sample_where(PERIPHERY_K, |_| true)
        .count();

    println!("{:<8} {:>8} {:>8} {:>10}", "arm", "p@1", "p@3", "elim_fp");
    for (name, m) in [
        ("A0", &m_a0),
        ("A1", &m_a1),
        ("A1c", &m_a1c),
        ("A2-S0", &m_a2s0),
        ("A2-S3", &m_a2s3),
        ("A2", &m_a2),
        ("A2'", &m_a2v),
    ] {
        println!(
            "{:<8} {:>8.3} {:>8.3} {:>10}",
            name, m.p_at_1, m.p_at_3, m.elimination_fp
        );
    }
    println!("mean board candidate_count (board stage, same across arms) = {mean_cc:.2}");
    println!(
        "A2 strata: S0 admitted on {s0_admits_arenas}/{N_ARENAS} arenas, S3 ran on {s3_ran_arenas}/{N_ARENAS}"
    );
    println!(
        "A2 (variant 1, S3 unconditional): fire rate (ranking vector != A1c) = {periphery_rate:.3} \
         ({periphery_fires}/{N_ARENAS}); ORDER-change rate = {reorder_rate:.3} ({periphery_reorders}/{N_ARENAS}); \
         S3 reordered on {s3_reordered_arenas}/{N_ARENAS}"
    );
    println!(
        "A2' (variant 2, S3 on council split): council split on {council_splits}/{N_ARENAS}; S3 ran on \
         {v_s3_ran_arenas}/{N_ARENAS}; fire rate = {v_periphery_rate:.3} ({v_periphery_fires}/{N_ARENAS}); \
         ORDER-change rate = {v_reorder_rate:.3} ({v_periphery_reorders}/{N_ARENAS})"
    );
    println!(
        "scalar-reading periphery sample at RungLevel::Counterfactual (reported only) = {scalar_periphery}"
    );
    println!(
        "A1 ASC outcomes: Revised={asc_revised} BlockedSelfReference={asc_blocked} \
         NoTarget={asc_no_target} Skipped(leader==C*/no-leader)={asc_skipped}"
    );
    println!();

    println!(
        "== AN — size-preserving null ({N_PERMS} perms x {N_ARENAS} arenas = {} runs, A2 procedure) ==",
        u64::from(N_PERMS) * N_ARENAS as u64
    );
    for (label, n) in [("AN  (variant 1)", &an), ("AN' (variant 2)", &anv)] {
        println!(
            "{label} aggregate p@1 per permutation: {:.3?}",
            n.aggregates
        );
        println!(
            "{label} p@1 mean                                   = {:.3}",
            n.mean
        );
        println!(
            "{label} p@1 p95  (of the {N_PERMS} per-permutation aggregates) = {:.3}   <- comparator",
            n.p95
        );
        println!(
            "{label} p@1 max  (of the {N_PERMS} per-permutation aggregates) = {:.3}",
            n.max
        );
        println!(
            "{label} p95 of the {N_ARENAS} per-arena hit rates (secondary)   = {:.3}",
            n.per_arena_p95
        );
        println!(
            "{label} distinct-pair enforcement: {} repair swaps, {} base re-draws (over {} shuffles)",
            n.shuffle.repairs,
            n.shuffle.restarts,
            u64::from(N_PERMS) * N_ARENAS as u64
        );
    }
    println!();

    // ── STRUCTURAL TABLE ───────────────────────────────────────────────
    println!("== STRUCTURAL TABLE ==");
    println!(
        "{:<4} {:<5} {:<15} {:<10} min_rung",
        "id", "code", "tier", "bucket"
    );
    for id in [RCR_ID, TR_ID, ASC_ID, CAS_ID, CR_ID] {
        let r = recipe(id).expect("recipe ids 4/6/7/8/11 must exist in the 34-recipe catalogue");
        println!(
            "{:<4} {:<5} {:<15?} {:<10?} {:?}",
            r.id,
            r.code,
            r.tier,
            r.bucket,
            r.min_rung()
        );
    }
    println!();
    println!("{:<5} {:>12} {:>12}", "rung", "admissible", "peripheral");
    for p in 0u8..=9 {
        let rung = RungLevel::from_u8(p);
        let adm = rung.admissible_recipes().count();
        let per = rung.peripheral_recipes().count();
        println!("{p:<5} {adm:>12} {per:>12}   ({rung:?})");
    }
    println!();

    // ── VERDICT — once per pre-registered variant ──────────────────────
    let v1 = verdict(
        "variant 1 (S3 unconditional)",
        &m_a0,
        &m_a2,
        &an,
        periphery_rate,
    );
    let v2 = verdict(
        "variant 2 (S3 on council split)",
        &m_a0,
        &m_a2v,
        &anv,
        v_periphery_rate,
    );
    println!("== SUMMARY ==");
    println!(
        "variant 1 (S3 unconditional)      : {}",
        if v1 { "PASS" } else { "KILL" }
    );
    println!(
        "variant 2 (S3 on council split)   : {}",
        if v2 { "PASS" } else { "KILL" }
    );
}

/// The pre-registered PASS/KILL rule, applied to one variant. Prints the
/// three conditions and returns PASS.
fn verdict(
    label: &str,
    m_a0: &ArmMetrics,
    m_a2: &ArmMetrics,
    an: &NullSummary,
    fire_rate: f64,
) -> bool {
    println!("== VERDICT — {label} ==");

    let a_margin = m_a2.p_at_1 - m_a0.p_at_1;
    let a_margin_ok = a_margin >= 0.05;
    let a_beats_an = m_a2.p_at_1 > an.p95;
    let a_pass = a_margin_ok && a_beats_an;
    println!(
        "(a) A2 p@1 − A0 p@1 = {a_margin:.3} (≥0.05? {a_margin_ok}) AND A2 p@1 ({:.3}) > AN p95 \
         ({:.3})? {a_beats_an}  =>  {}",
        m_a2.p_at_1,
        an.p95,
        pf(a_pass)
    );

    let b_pass = m_a2.elimination_fp == 0;
    println!(
        "(b) A2 planted-cause elimination false positives = {}  =>  {}",
        m_a2.elimination_fp,
        pf(b_pass)
    );

    let c_pass = (0.10..=0.90).contains(&fire_rate);
    println!(
        "(c) periphery fire-rate window: rate={fire_rate:.3}, needs [0.10, 0.90]  =>  {}",
        pf(c_pass)
    );

    let mut failed: Vec<&str> = Vec::new();
    if !a_pass {
        failed.push("(a) recovery margin vs A0 and vs the AN null");
    }
    if !b_pass {
        failed.push("(b) elimination false-positive");
    }
    if !c_pass {
        failed.push("(c) periphery fire-rate window");
    }
    if failed.is_empty() {
        println!("VERDICT: PASS");
    } else {
        println!("VERDICT: KILL — failed: {}", failed.join("; "));
    }
    println!();
    failed.is_empty()
}

#[cfg(test)]
mod tests {
    //! The five guards as `cargo test --example house_differential` tests, so a
    //! regression in `rcr_abduce`, `cas_abstract`, `asc_challenge` or the
    //! elimination predicate fails the suite instead of only printing FAIL.
    use super::*;

    fn throttle() -> Throttle {
        Throttle::new(0.0, 4096, HUB_INDEGREE)
    }

    #[test]
    fn g1_strata_off_equals_a1c() {
        assert!(guard_g1(&throttle()));
    }

    #[test]
    fn g2_fixture_direction_yields_case_to_cause() {
        assert!(guard_g2(&throttle()));
    }

    #[test]
    fn g3_admission_is_load_bearing() {
        assert!(guard_g3(&throttle()));
    }

    #[test]
    fn g4_far_fact_is_cas_down_only() {
        assert!(guard_g4(&throttle()));
    }

    #[test]
    fn g5_elimination_predicate_two_sided() {
        assert!(guard_g5());
    }

    #[test]
    fn null_shuffle_never_yields_a_duplicate_pair() {
        for idx in 0..8 {
            let fixture = build_fixture(idx);
            for perm_idx in 0..N_PERMS {
                let (edges, _) = permuted_rule_edges(&fixture, perm_idx);
                let mut seen = edges.clone();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), edges.len(), "arena {idx} perm {perm_idx}");
                assert_eq!(edges.len(), fixture.rule_edges.len(), "size preserved");
            }
        }
    }
}
