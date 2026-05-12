# PR-J-1: INT4-32D Thinking Atoms

**Tier-3 specialization spec — Pattern J canonical (post-PR #359 letter assignment).**
**Tech-debt anchor:** TD-INT4-32D-ATOMS-6.
**Sprint-3 owner:** W7 (this spec) -> engineer pickup.

---

## Goal

Land a 16-byte cognitive-style fingerprint — `ThinkingAtom32x4` = 32 dims
x 4 bits = 128 bits = 16 bytes — and a K-NN proximity search over the
12-entry `p64-bridge::STYLES` codebook. The fingerprint is the fast-path
fallback the orchestrator uses when OGIT does **not** yet have a best-
practice thinking-style pattern for a brand-new domain.

A new domain shows up, the cognitive shader assembles a `ThinkingAtom32x4`
from situation features, and `knn_thinking_styles` returns top-K nearest
known styles by INT4 cosine. The dispatcher then either adopts the
nearest verbatim or blends the top-K into a transient style while OGIT
catches up.

This is the cold-start safety net for Pattern G (Best-Practice Thinking
Inheritance, deferred this sprint). PR-J-1 makes Pattern G implementable
later: without an INT4-32D codebook, Pattern G has no proximity fallback.

---

## Files to touch (Rust)

| File | Change |
|---|---|
| `crates/lance-graph-contract/src/thinking_atom.rs` | **NEW** -- `ThinkingAtom32x4`, nibble accessors, `cosine_int4`, `DIM_NAMES: [&'static str; 32]` |
| `crates/thinking-engine/src/atom_proximity.rs` | **NEW** -- `knn_thinking_styles(query, codebook, k)` |
| `crates/p64-bridge/src/lib.rs` | Extend `STYLES` with hand-coded `int4_32d_fingerprint` per style (12 x 16 = 192 bytes static) |

No new crate dependencies. Pure Rust, `no_std`-friendly arithmetic in the
contract type.

---

## API sketch (~120 LOC)

```rust
// crates/lance-graph-contract/src/thinking_atom.rs

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThinkingAtom32x4 {
    /// 32 nibbles packed two-per-byte. dims[i] low nibble = dim 2i,
    /// high nibble = dim 2i+1. Each nibble in [0..15].
    pub dims: [u8; 16],
}

impl ThinkingAtom32x4 {
    pub const ZERO: Self = Self { dims: [0u8; 16] };
    pub const NUM_DIMS: usize = 32;
    /// Maximum cosine_int4 score: 32 x 0xF = 480.
    pub const COSINE_MAX: u32 = 480;

    pub fn nibble(&self, idx: usize) -> u8 {
        debug_assert!(idx < Self::NUM_DIMS);
        let byte = self.dims[idx / 2];
        if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 }
    }

    pub fn set_nibble(&mut self, idx: usize, val: u8) {
        debug_assert!(idx < Self::NUM_DIMS && val <= 0x0F);
        let byte = &mut self.dims[idx / 2];
        if idx & 1 == 0 {
            *byte = (*byte & 0xF0) | (val & 0x0F);
        } else {
            *byte = (*byte & 0x0F) | (val << 4);
        }
    }

    /// Per-nibble min-sum over 32 dims. Returns [0, 480]; higher == closer.
    /// SWAR-style: 16-byte loop, two nibbles/iter, compiler auto-vectorises.
    pub fn cosine_int4(&self, other: &Self) -> u32 {
        let mut acc: u32 = 0;
        for i in 0..16 {
            let (a, b) = (self.dims[i], other.dims[i]);
            let a_lo = (a & 0x0F) as u32;
            let b_lo = (b & 0x0F) as u32;
            let a_hi = (a >> 4) as u32;
            let b_hi = (b >> 4) as u32;
            acc += a_lo.min(b_lo) + a_hi.min(b_hi);
        }
        acc
    }

    /// Per-nibble inversion (n -> 15 - n). For anti-style queries.
    pub fn invert_nibbles(&self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..16 {
            let a = self.dims[i];
            let lo = 0x0F - (a & 0x0F);
            let hi = 0x0F - (a >> 4);
            out.dims[i] = lo | (hi << 4);
        }
        out
    }

    /// Hand-curated v1 dimension catalogue. Order is contract-stable;
    /// appending dims is a breaking change to the codebook layout.
    pub const DIM_NAMES: [&'static str; 32] = [
        "reasoning_polarity",       //  0: deductive <-> inductive <-> abductive
        "context_window",           //  1: point <-> historical <-> cumulative
        "uncertainty_handling",     //  2: Bayesian <-> NARS <-> frequentist
        "graph_traversal_shape",    //  3: BFS <-> DFS <-> random-walk <-> spectral
        "temporal_horizon",         //  4: now <-> near <-> middle <-> far
        "abstraction_level",        //  5: literal <-> schematic <-> categorical
        "evidence_appetite",        //  6: parsimonious <-> exhaustive
        "contradiction_tolerance",  //  7: collapse-fast <-> hold-open
        "modality_preference",      //  8: linguistic <-> spatial <-> numeric <-> mixed
        "compositionality_depth",   //  9: flat <-> shallow <-> recursive
        "memory_recency_weight",    // 10: now-biased <-> uniform <-> distant-biased
        "exploration_vs_exploit",   // 11: greedy <-> epsilon <-> Thompson
        "metacognition_depth",      // 12: zeroth <-> first-order <-> second-order
        "speed_vs_thoroughness",    // 13: snap <-> deliberate
        "analogy_reach",            // 14: same-domain <-> adjacent <-> distant
        "causal_reasoning_mode",    // 15: forward <-> backward <-> counterfactual
        "decomposition_strategy",   // 16: top-down <-> bottom-up <-> middle-out
        "ambiguity_resolution",     // 17: defer <-> commit-fast <-> hold-multiple
        "salience_focus",           // 18: foreground <-> background <-> peripheral
        "narrative_coherence",      // 19: episodic <-> thematic <-> systemic
        "feedback_sensitivity",     // 20: open-loop <-> closed-loop <-> adaptive
        "domain_specialization",    // 21: generalist <-> specialist
        "constraint_handling",      // 22: relaxation <-> propagation <-> backtracking
        "abstraction_origin",       // 23: data-driven <-> theory-driven
        "reuse_vs_synthesis",       // 24: pattern-match <-> synthesize <-> blend
        "verification_appetite",    // 25: trust <-> spot-check <-> formally-verify
        "social_affect_weight",     // 26: ignore <-> consider <-> centralize
        "risk_posture",             // 27: averse <-> neutral <-> seeking
        "communication_register",   // 28: terse <-> conversational <-> exhaustive
        "energy_budget",            // 29: low <-> medium <-> high
        "novelty_appetite",         // 30: in-distribution <-> explore-novel
        "self_revision_rate",       // 31: stable <-> revisable <-> volatile
    ];
}

// crates/thinking-engine/src/atom_proximity.rs

use lance_graph_contract::thinking_atom::ThinkingAtom32x4;
use lance_graph_contract::thinking::ThinkingStyle;

pub fn knn_thinking_styles(
    query: &ThinkingAtom32x4,
    codebook: &[(ThinkingStyle, ThinkingAtom32x4)],
    k: usize,
) -> Vec<(ThinkingStyle, u32)> {
    let mut scored: Vec<_> = codebook
        .iter()
        .map(|(style, atom)| (style.clone(), query.cosine_int4(atom)))
        .collect();
    scored.sort_by_key(|(_, score)| core::cmp::Reverse(*score));
    scored.truncate(k);
    scored
}
```

For p64-bridge: each `STYLES` entry gains a `pub const`
`int4_32d_fingerprint: ThinkingAtom32x4` literal hand-coded by mapping
the style's known cognitive profile onto the 32 named dims.

---

## Design notes (engineer-facing detail)

**Why nibble-min-sum, not popcount-AND.** Popcount-AND treats each nibble
as 4 unrelated bits; a nibble of `0x5` (0101) reads as "2 bits set" and
loses the *value-5-on-a-0..15-axis* semantics. Min-sum `min(a, b)` per
nibble preserves the ordinal axis: query `0xF` against codebook `0xF`
scores 15, against `0x7` scores 7, against `0x0` scores 0. Sum over 32
dims gives `[0, 480]` with monotone "closer == higher" semantics. See
open-question 2 for the worked comparison.

**Why 32 dims hand-curated.** 32 dims fits exactly in 16 bytes at INT4
— the smallest power-of-two width that gives meaningful cognitive
resolution. PCA-derived dims (v2) would need a `(situation, chosen-style)`
corpus we do not yet have; hand-curating from existing thinking-engine
literature (NARS, ReAct, Six Hats) bootstraps in one engineer-day. v2
PCA replacement is gated on ~10K dispatch decisions logged from deployed
v1. Dim **order** is contract-stable; appending is a breaking change.

**Why cache the per-style fingerprint as `const`.** 12 x 16 = 192 bytes
total. Lazy computation adds startup-order risk to the existing
`once_cell` machinery in p64-bridge with no measurable benefit. Cache as
`pub const` and expose via the existing `STYLES` array.

**Relationship to PR-B-1 ContextBundle.** `ContextBundle` (W3) reserves
a `thinking_styles` slot. Once PR-B-1 lands, that slot is populated by
calling `knn_thinking_styles(query_atom, &p64_bridge::STYLES, k=3)` and
storing the top-3 styles + scores. PR-J-1 does NOT modify ContextBundle
directly — W3 owns that surface.

---

## Test plan

| Test | Coverage |
|---|---|
| `tests/atom_nibble_round_trip.rs` | For all 32 nibble indices and all `v in 0..=15`: `set_nibble(i, v); assert_eq!(nibble(i), v)`. Proves the byte-packing is correct on both even and odd indices. |
| `tests/atom_cosine_self_max.rs` | `cosine_int4(x, x)` equals the sum of `x`'s own nibbles for arbitrary `x`; equals `COSINE_MAX (480)` for the maximal atom. |
| `tests/atom_cosine_zero_min.rs` | `cosine_int4(all_max, all_min) == 0`; `cosine_int4(x, x.invert_nibbles())` is bounded by the asymmetric-inversion lemma (= sum of `min(n, 15-n)` per nibble). |
| `tests/knn_thinking_styles.rs` | Build a query atom that copies one of the 12 STYLES fingerprints; assert `knn_thinking_styles(&q, &STYLES, k=3)` returns that style first with score `COSINE_MAX`. Also asserts sort-stability: ties broken in codebook order. |

**Suggested fixture:** add a 4-style mini-codebook at
`crates/thinking-engine/tests/fixtures/atom_codebook_mini.rs` so the
`knn_thinking_styles` test does not need the full 12-entry STYLES table
loaded just to validate ranking semantics.

---

## Dependencies

- **PR-B-1 (W3 spec)** is the consumer integration point (the
  `ContextBundle.thinking_styles` slot is where K-NN results land).
  PR-J-1 itself does NOT depend on PR-B-1 to compile or test — the K-NN
  function takes a generic codebook slice. **PR-J-1 can land in
  parallel with PR-B-1.**
- No external crate dependencies. Pure Rust, no `unsafe`, no SIMD
  intrinsics in v1 (compiler auto-vectorisation of the 16-byte min-sum
  loop is sufficient for the 12-entry codebook scan).

---

## Acceptance criteria

- [ ] `ThinkingAtom32x4` defined in `lance-graph-contract` with 32
      named dimensions in `DIM_NAMES`.
- [ ] `nibble` / `set_nibble` accessors with `debug_assert!` bounds.
- [ ] `cosine_int4` implemented as nibble-min-sum (NOT popcount-AND), in
      a 16-byte loop the compiler can auto-vectorise.
- [ ] `invert_nibbles` helper for anti-style queries.
- [ ] `knn_thinking_styles` returns top-K sorted descending by score.
- [ ] `p64-bridge::STYLES` extended: 12 hand-coded `int4_32d_fingerprint`
      literals. Total static footprint ~192 bytes.
- [ ] 4 new tests landed (round-trip, self-max, zero-min, knn).
- [ ] No new external crate deps; contract crate stays `no_std`-clean.

---

## Effort

**Small.** ~120 LOC of Rust + 4 small tests + 12 hand-coded fingerprints.
Estimate **~1 engineer-day** end-to-end.

Rough breakdown:

- `ThinkingAtom32x4` + nibble accessors + DIM_NAMES — 60 LOC
- `cosine_int4` + `invert_nibbles` — 25 LOC
- `knn_thinking_styles` — 15 LOC
- 12 hand-coded `int4_32d_fingerprint` literals — ~30 LOC (most time
  is on cognitive curation, not typing)
- 4 new tests — 40 LOC

---

## Open questions for the engineer

1. **Hand-curate vs PCA-derive the 32 dims.** Recommend **hand-curate
   v1** (the 32 names listed). Reasoning: PCA over what corpus? We do
   not yet have a `(situation, chosen-style)` telemetry stream large
   enough to learn dimensions from. v1 ships now; flag a follow-up
   `TD-INT4-32D-PCA-v2` once we have ~10K dispatch decisions logged
   from the deployed K-NN. **Do not block PR-J-1 on PCA.**
2. **`cosine_int4` metric: popcount-AND vs nibble-min-sum.** Recommend
   **nibble-min-sum**. INT4 nibbles encode an ordinal axis (0..15);
   popcount-AND throws away ordinality. Worked test: `min-sum(0xF, 0xF)
   - min-sum(0xF, 0x0) = 15`; `popcount(0xF & 0xF) - popcount(0xF & 0x0)
   = 4`. Min-sum's signal is 4x richer at INT4 resolution, stays in
   `u32` arithmetic with no overflow risk (max 32 x 15 = 480 << 2^32).
3. **Per-style fingerprint storage: cached `const` vs lazy.** Recommend
   **cached const**. 12 x 16 = 192 bytes static is negligible; lazy
   adds startup-order entanglement with no benefit. Re-visit only if
   the codebook grows past ~10K entries (then heap + Lance-backed
   table — different PR).
4. **What does an "anti-style" query mean?** `invert_nibbles` is
   exposed but PR-J-1 does not consume it. Document as a slot for the
   dispatcher to use when negative evidence is present (situation
   excludes "deductive" -> query the inverted atom). Consumer wiring
   lands in PR-G-* once Pattern G is spec'd.
5. **Tie-breaking when two codebook styles score identically.** Rust's
   `sort_by_key` is stable; ties resolve in codebook order. Document
   this guarantee on `knn_thinking_styles` — downstream consumers MUST
   NOT assume any other rule.

---

## Cross-references

- `.claude/board/TECH_DEBT.md` — TD-INT4-32D-ATOMS-6
- `.claude/specs/pr-b-1-context-bundle.md` (W3 sister; the
  `thinking_styles` slot is the consumer of `knn_thinking_styles`)
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` (W2 sister; sibling Tier-1
  spec, not a dependency)
- `.claude/specs/sprint-3-execution-plan.md` (W1 master execution plan)
- `.claude/knowledge/tier-0-pattern-recognition.md` — Pattern J section
- `crates/p64-bridge/src/lib.rs` — `STYLES` (the 12-entry codebook this
  spec extends with `int4_32d_fingerprint`)
- Pattern G (Best-Practice Thinking Inheritance) — deferred this
  sprint; PR-J-1 is the proximity-search foundation Pattern G consumes
