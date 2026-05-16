# Iron Rules Doctrine — Meta-Pattern, Per-Rule Analysis, and Promotion Template

> **READ BY:** Every agent touching architectural decisions, before any
> iron-rule change. Tier-0 mandatory for any sprint that proposes adding,
> retiring, or amending a `CLAUDE.md §Substrate-level iron rules` entry.
> Also read by meta-Opus reviewers when grading FINDINGs that look like
> iron-rule candidates.
>
> **Status:** DOCTRINE (consolidates the four canonical iron rules and
> the promotion ceremony). Created sprint-13 preflight (PP-2) per
> Wave G CSI-18 observation that the four current iron rules share a
> meta-pattern; future iron rules should fit the template intentionally,
> not by retrofit.
>
> **Predecessors:**
> - `CLAUDE.md §Substrate-level iron rules` — the four canonical entries
> - `.claude/board/EPIPHANIES.md` — the FINDINGs that preceded each rule
>   (E-SUBSTRATE-1, E-ORIG-7 → Jirak, the 2026-04-21 VSA register-loss
>   refinement, E-META-10)
> - `.claude/board/sprint-log-11/meta-review-opus.md` §CSI-18 — the
>   Wave G Opus review entry that named the meta-pattern explicitly
> - `.claude/plans/cognitive-substrate-convergence-v2.md §13` — the
>   sprint-13 risk matrix carrying §13.8 v1-API-under-v2 forward
>
> **Not a contract.** This is a doctrinal companion to CLAUDE.md, not a
> replacement. The four iron rules in CLAUDE.md remain the canonical
> source-of-truth for the rule text; this doc names the SHAPE they
> share and the CEREMONY that admits a fifth.

---

## 1. The Iron Rule Meta-Pattern

All four iron rules ratified through sprint-12 share one shape:

> **Surface:** a substrate-level guarantee, or an anti-pattern bounded by
> a substrate-level guarantee.
>
> **Backing:** a citation — peer-reviewed paper, observed-bug pattern with
> N ≥ 3 instances, or a doctrinal architectural choice that was made
> explicitly (not inherited).
>
> **Consequences:** 3-5 enforceable rules, each in `Do NOT X` or
> `MUST do Y` form, that an implementer can check at code-review time
> without re-deriving the rationale.
>
> **Cross-refs:** the EPIPHANIES FINDING that preceded the iron rule,
> plus the downstream code sites where the rule binds.

Distilled: an iron rule is a **substrate-level invariant + N enforceable
guards + 1 named test pattern that systematically catches violations**.
The "iron" in iron rule is the test pattern — what makes the rule
self-enforcing across many implementers and many sessions, instead of a
style guideline that depends on reviewer attention.

The four current iron rules are each pinned to a different axis:

| Axis | Iron rule | What gets bounded |
|---|---|---|
| Substrate operator | `I-SUBSTRATE-MARKOV` | which binding operator may be used in transition kernels |
| Statistical model | `I-NOISE-FLOOR-JIRAK` | which Berry-Esseen variant grounds σ-threshold claims |
| Data semantics | `I-VSA-IDENTITIES` | what kind of bits VSA may operate on (identity vs content) |
| API version | `I-LEGACY-API-FEATURE-GATED` | how v1 accessors coexist with v2 layout flags |

Each axis is a different *kind* of substrate guarantee: an algebraic one
(operator), a probabilistic one (model), a semantic one (data class), a
versioning one (API surface). A fifth axis — e.g. memory ordering, or
ABI-stability under codebook rebase — would require its own iron rule
when an N ≥ 3 violation pattern is observed.

---

## 2. Per-Rule Analysis

### 2.1 I-SUBSTRATE-MARKOV

| Field | Value |
|---|---|
| **Axis** | Substrate operator (which binding algebra is admissible) |
| **Originating FINDING** | `EPIPHANIES.md` 2026-04-20 E-SUBSTRATE-1 — VSA-bundling guarantees Chapman-Kolmogorov by construction |
| **Year ratified** | 2026 (added 2026-04-20 per `[FORMAL-SCAFFOLD]` reclassification) |
| **Backing citation** | Johnson-Lindenstrauss + concentration-of-measure in d=10000; saturating bundle is associative and commutative in expectation; deviations from associativity decay at rate `~e^(-d)`. See `[FORMAL-SCAFFOLD]` five-pillar entry in EPIPHANIES. |
| **Hard constraint** | Do NOT replace bundle with XOR or non-commutative binding for state-transition paths without reviewing `[FORMAL-SCAFFOLD]`. `MergeMode::Xor` is a legitimate single-writer delta merge mode (per I1) but is NOT a Markov-respecting transition kernel. |
| **Soft constraints** | (1) D7's implicit Markov reliance is *grounded*, not silent — Chapman-Kolmogorov consistency is an implementation sanity check, not a falsification gate. (2) Any substrate-level change that weakens associativity (operator swap, dimension reduction below 10000, removal of concentration-of-measure assumption) MUST consult `[FORMAL-SCAFFOLD]` and document the trade-off. |
| **Cross-cutting test pattern** | **Chapman-Kolmogorov consistency test.** For any new transition kernel `K`, verify `K(2τ) ≈ K(τ)²` in expectation across a sampled state distribution. If the test fails, either (a) the operator is non-bundling (likely an I-rule violation) or (b) there is an implementation bug. The test can only fail from bugs, never from theoretical violations, because the bundle is CK *by construction* in d=10000. |
| **Binding code sites** | `contract::collapse_gate::MergeMode::{Bundle, Xor}`, `crystal/fingerprint.rs vsa_bundle`, `crystal/fingerprint.rs vsa_bind`, anything that constructs a `Vsa16kF32` state representation in a transition path |
| **Sibling iron rules invoked** | `I-VSA-IDENTITIES` (VSA bundles identities, not content — combines with this rule to forbid bundle-of-CAM-PQ-codes); `I-NOISE-FLOOR-JIRAK` (CAM-PQ contamination is the weak-dependence source that breaks classical Berry-Esseen on top of this Markov substrate) |

### 2.2 I-NOISE-FLOOR-JIRAK

| Field | Value |
|---|---|
| **Axis** | Statistical model (which Berry-Esseen rate governs significance claims) |
| **Originating FINDING** | `EPIPHANIES.md` E-ORIG-7 — Jirak Berry-Esseen under weak dependence IS the Phase-5 noise-floor lemma (folded into `[FORMAL-SCAFFOLD]` four-pillar entry, then promoted to iron rule when the noise-floor question recurred across multiple sprints) |
| **Year ratified** | 2026 (added 2026-04-20 alongside I-SUBSTRATE-MARKOV) |
| **Backing citation** | Jirak 2016, "Berry-Esseen theorems under weak dependence," arxiv 1606.01617, Annals of Probability 44(3) pp. 2024–2063. Rate: `n^(p/2-1)` for `p ∈ (2,3]`, `n^(-1/2)` in L^q for `p ≥ 4`. |
| **Hard constraint** | Classical IID Berry-Esseen is WRONG for this system. Bits in the workspace's 16384-bit fingerprints are weakly dependent by construction (correlated embedding projections, overlapping role-key slices, codebook-shared quantization, XOR-bundle weak dependence). |
| **Soft constraints** | (1) ICC, Spearman ρ, and similar significance metrics MUST cite Jirak's rate, not classical Berry-Esseen, when claiming "observed value is N σ above noise floor." (2) σ-threshold calibration (UNBUNDLE_HARDNESS_THRESHOLD, ABDUCTION_THRESHOLD, Σ-tier Rubicon) SHOULD cite Jirak-derived bounds; hand-tuned values are acceptable but MUST say so. (3) The three revival candidates in `[FORMAL-SCAFFOLD]` *Coupled revival track* (VAMPE + Jirak pair) replace hand-tuned σ thresholds with bound-derived ones when they activate. |
| **Cross-cutting test pattern** | **Berry-Esseen-rate citation in σ-threshold derivations.** Any constant or formula that participates in a "N σ above noise floor" claim is grep-able for either (a) a Jirak rate citation `p ∈ (2,3]` or `p ≥ 4`, or (b) an explicit `// hand-tuned, see TD-SIGMA-TIER-THRESHOLDS-1` annotation. The PR-review pattern is: grep for `σ`, `sigma`, `threshold`, `noise_floor` in changed files; any new occurrence without citation is a P1 finding. |
| **Binding code sites** | `crates/sigma-tier-router/src/lib.rs` Rubicon threshold, `cognitive-shader-driver` UNBUNDLE_HARDNESS_THRESHOLD, ABDUCTION_THRESHOLD, any noise-floor claim in CHANGELOG / PR description, any test that asserts "N σ" |
| **Sibling iron rules invoked** | `I-SUBSTRATE-MARKOV` (the Markov substrate is the *source* of weak dependence — Jirak is the right tool because the substrate guarantees the assumption); `I-VSA-IDENTITIES` (CAM-PQ contamination of bundles is one specific weak-dependence mechanism this rule covers) |

### 2.3 I-VSA-IDENTITIES

| Field | Value |
|---|---|
| **Axis** | Data semantics (what class of bits VSA may operate on) |
| **Originating FINDING** | `EPIPHANIES.md` 2026-04-21 — VSA operates on identities, not content — the refined iron rule (which superseded earlier same-day "VSA must be FP32 multiply/add on identities, not XOR on bitpacked content" CORRECTION-OF entry and the D5 Frankenstein revert) |
| **Year ratified** | 2026 (added 2026-04-21, one day after I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK) |
| **Backing citation** | Observed-bug pattern: D5 Frankenstein shipped `Vsa10k = [u64; 157]` bitpacked + slice-masked XOR `RoleKey::bind/unbind` + `vsa_xor` / `vsa_similarity` (Hamming-based reinvention) — three composite errors in one PR, plus the 5-role lossless test passed for the WRONG reason. N ≥ 3 violation count established within a single PR. Reverted via commit `0ae9f90`. |
| **Hard constraint** | VSA operates on IDENTITY fingerprints that POINT TO content. Never on content's bitpacked/quantized register itself. Superposing CAM-PQ codes, quantized indices, or sign-binarized fingerprints destroys the mapping from bits back to codebook entries (the "register-loss problem"). |
| **Soft constraints** | (1) **CAM-PQ vs VSA:** Never superpose CAM-PQ codes directly. CAM-PQ is for *search* (compressed NN); VSA is for *bundling* (lossless role superposition). Switching between them requires decompression, not mixing. (2) **Lazy VSA check:** If `Vsa16kF32` is reached for as a fancy lookup when a `HashMap` would do, stop — that is register laziness, not VSA usage. (3) **Archetype / persona / thinking-style unification:** All Layer-2 role catalogues, each entry one identity fingerprint; content (slots, rules, prompts) lives in YAML; resonance (cosine vs codebook) dispatches to content. (4) **The four tests before reaching for VSA:** register laziness, bundle size N ≤ √d / 4, role orthogonality, cleanup codebook — any "no" short-circuits. |
| **Cross-cutting test pattern** | **Register-loss test.** For any code that calls `vsa_bundle` or `vsa_bind`: assert that the inputs are *identity* fingerprints (constructed from a known role-key catalogue) and not *content* fingerprints (CAM-PQ codes, sign-binarized embeddings, quantized indices). The test takes the form: `assert!(is_identity_fp(x))` at every bundle call site, or equivalently a type-system gate where `Vsa16kF32` is only constructable through a known catalogue API. CI grep: `vsa_bundle.*cam_pq`, `vsa_bind.*quantized` patterns are bug signals. |
| **Binding code sites** | `crystal/fingerprint.rs` (`Vsa16kF32`, `Vsa16kBF16`, `Vsa16kF16`, `Vsa16kI8`, `Binary16K`; `vsa_bind`, `vsa_bundle`, `vsa_cosine`), `grammar/role_keys.rs` and any future per-domain `*/role_keys.rs`, AriGraph entity_index, EpisodicMemory store, the four CAM-PQ codebooks |
| **Sibling iron rules invoked** | `I-SUBSTRATE-MARKOV` (the bundle algebra that Markov stands on is the same algebra this rule restricts to identities); `I-NOISE-FLOOR-JIRAK` (CAM-PQ-in-bundle is one weak-dependence mechanism — separating CAM-PQ from VSA reduces the dependence structure to one Jirak can bound cleanly) |

### 2.4 I-LEGACY-API-FEATURE-GATED

| Field | Value |
|---|---|
| **Axis** | API version (how v1 accessors coexist with v2 layout flags) |
| **Originating FINDING** | `EPIPHANIES.md` E-META-10 — v1-API-under-v2-feature alias pattern: systematic layout-bit boundary testing required (sprint-11 meta-review; pending prepend to EPIPHANIES per W-F11 in `sprint-log-11/meta-review.md`) |
| **Year ratified** | 2026 (sprint-12 onwards; promotion text drafted in `sprint-log-11/meta-review-opus.md` CSI-18) |
| **Backing citation** | Observed-bug pattern: PR #383 had 4 instances of v1-accessor-corrupts-v2-reclaim-zone in a single PR (`pack()` temporal write, `inference_type()` raw discriminant return, `set_temporal()`, `forward()`). Each required a separate fix commit. N ≥ 3 violation count established. Codex review caught all 4. |
| **Hard constraint** | Any v1 API path that writes to bits reclaimed by a v2 feature flag MUST be either feature-gated to no-op (`#[cfg(not(feature = "<v2-feature>"))]`) or routed through the canonical v2 accessor. No silent shims. |
| **Soft constraints** | (1) Field-isolation matrix tests are mandatory at the layout-bit boundary — one test per accessor pair, checking no bit-bleed between adjacent fields. The `v2_layout_tests.rs` 16-test matrix is the reference artifact. (2) Sprint-implementors adding any new v1-compat path must run the same matrix, not just round-trip tests. (3) Generalization: applies to any codebase with versioned bit layouts under feature flags. Detection: grep all writes to the feature-gated zone by non-v2 code paths before each PR that touches the layout. (4) Backward-compat shims for layout-breaking changes are NOT "just rename the accessor" — they are an audit + gate + test triple. |
| **Cross-cutting test pattern** | **Field-isolation matrix tests.** For every accessor of a struct whose layout is feature-gated: assert that writing to field A does not change bits in field B, for every (A, B) pair where A and B share the underlying integer. The matrix grows as ~`k(k-1)/2` for `k` fields; for `CausalEdge64` with 6 fields, this is 15-16 tests. The matrix is the test pattern — not "the round-trip passes," but "no bit-bleed across the matrix." |
| **Binding code sites** | `crates/causal-edge/src/edge.rs` (`pack()`, `inference_type()`, `set_temporal()`, `forward()`), `crates/causal-edge/tests/v2_layout_tests.rs` (the reference field-isolation matrix), any future feature-gated layout change to a packed struct in the workspace |
| **Sibling iron rules invoked** | None directly substrate-level; this rule is the *process* iron rule that protects the substrate ones. It is the only rule in the canonical four whose backing is purely observed-bug rather than theoretical, and the only one whose test pattern is purely structural (matrix coverage) rather than mathematical (CK consistency, Berry-Esseen citation, identity-vs-content). |

---

## 3. Iron Rule Template — Promotion Checklist

A FINDING in EPIPHANIES.md is a candidate for promotion to iron rule when
it satisfies ALL of the following. Use this checklist verbatim in the
meta-review entry that recommends promotion.

```
[ ] Surface statement: one sentence, bold, in the form of either
    a substrate-level guarantee ("X holds by construction") OR
    an anti-pattern ("Do NOT do X in context Y").

[ ] Backing citation: at least one of
    [ ] peer-reviewed paper with arxiv ID + journal + page range
    [ ] observed-bug pattern with N ≥ 3 instances cited by PR/commit
    [ ] doctrinal architectural choice with explicit ratification trail
        (cite the LATEST_STATE entry + ratification commit)

[ ] 3-5 enforceable consequences, each in `Do NOT X` or `MUST do Y` form.
    Each consequence is checkable at code-review time without
    re-deriving the rationale.

[ ] At least 1 named test pattern that systematically catches violations:
    [ ] the pattern is grep-able OR
    [ ] expressible as a CI assertion OR
    [ ] structured as a coverage matrix (like field-isolation tests)

[ ] Cross-reference to the predecessor FINDING in EPIPHANIES with date
    and entry title.

[ ] Sibling-iron-rule analysis: which existing iron rules does this one
    invoke, override, or compose with? (Even "none directly" must be
    stated explicitly.)

[ ] Axis identification: which axis does this rule pin?
    (substrate operator / statistical model / data semantics / API
    version / NEW — if NEW, explain why the existing four axes don't
    cover it.)
```

### Promotion ceremony

1. **Discovery.** A FINDING accumulates in EPIPHANIES.md. It is referenced
   by N ≥ 3 PR / sprint-log entries OR cites a load-bearing paper.
2. **Meta-Opus recommendation.** A meta-Opus reviewer names the FINDING
   as an iron-rule candidate in a `sprint-log-N/meta-review-opus.md`
   entry (the CSI-18 / E-META-10 promotion text in
   `sprint-log-11/meta-review-opus.md §CSI-18` is the canonical
   example).
3. **Iron-rule PR.** A governance-only PR adds the section to
   `CLAUDE.md §Substrate-level iron rules`. The PR description quotes
   the promotion checklist with each box ticked.
4. **Doctrine update.** This file (`iron-rules-doctrine.md`) appends a
   §2.N entry filling the per-rule analysis table.
5. **Test-pattern wiring.** The named test pattern lands as a CI gate
   OR a documented grep-able pattern in a sprint plan / TECH_DEBT entry.

The ceremony is intentionally heavy. Iron rules are forever — the
APPEND-ONLY discipline applies to retracting one (a retraction adds a
SUPERSEDED entry, not a deletion). Promote slowly; demote never.

---

## 4. Sprint-12 Lesson: Codex P1 Review as Canonical Pre-Merge Gate

Per E-META-10 and the resulting I-LEGACY-API-FEATURE-GATED rule, codex
review is now the enforcement mechanism for layout-bit boundary tests.
This is itself a process iron rule, of a different kind: a *workflow*
invariant rather than a substrate one.

**The process iron rule:** every PR that touches a feature-gated bit
layout MUST receive a codex review before merge, and the codex review
MUST include explicit field-isolation matrix verification.

**Why codex, not human review:** PR #383 had 4 v1-under-v2 instances in
one PR. Human review caught some; codex caught all 4 systematically.
The pattern is mechanical (grep all write sites to the feature-gated bit
zone for non-v2 code paths) and codex executes the pattern uniformly,
without depending on reviewer attention or sprint context.

**Why this is a process iron rule, not a style rule:**

- The pattern recurs across PRs (E-META-10's "every codex review catches
  the same anti-pattern" finding from `sprint-log-11/meta-review-opus.md`
  §"Honest Reflection")
- The cost of a single P0 escape is high (corrupting reclaimed bits in a
  packed struct is silent until a downstream consumer hits the corrupt
  value; bisection is then expensive)
- The cost of the gate is bounded (codex review per PR is already in the
  workflow; the marginal cost is including the field-isolation matrix
  output in the review)

**Codex P1 finding levels (informational, not part of the rule):**

| Level | Meaning | Action |
|---|---|---|
| P0 | Corrupts shipped behavior | Block merge until fixed |
| P1 | Violates iron rule / fails layout matrix | Block merge until fixed or explicit waiver in PR description |
| P2 | Style or convention violation | Fix preferred; merge OK if documented |
| P3 | Suggestion | Informational |

The "codex P1 = canonical pre-merge gate" framing means: an iron-rule
violation is a P1 codex finding, and a P1 codex finding blocks merge
unless the PR description explicitly waives it with rationale. Iron
rules without a corresponding codex P1 gate are weaker than iron rules
with one — sprint-13+ should fund codex-gate wiring for any new iron
rule at the time of promotion, not as a follow-up.

**Cross-ref:** `sprint-log-11/meta-review-opus.md` CSI-18 (the
recommendation); `sprint-log-11/meta-review.md` §E-META-10 NEW entry
(the FINDING); `cognitive-substrate-convergence-v2.md §13.8` (the
risk-matrix carry-forward).

---

## 5. What's NOT an Iron Rule

Distinguishing iron rules from softer patterns prevents inflation. The
following patterns are real and useful, but do not qualify for iron-rule
status:

### 5.1 Style rules (convention, not invariant)

| Pattern | Why it's a style rule |
|---|---|
| "Every deprecated method needs a `#[deprecated]` migration pointer" | High-value but no observed-bug N ≥ 3 *substrate-corruption* pattern. Missing pointers cause migration friction, not substrate violations. Belongs in CONTRIBUTING / lint config. |
| "All public APIs documented with examples" | Documentation quality; not invariant. |
| "Worker prompts must include git-reconcile step" (E-META-10's third honest-reflection point) | Workflow discipline; the cost of a single violation is bounded (plan drift, not silent corruption). Belongs in worker template, not iron rules. |
| "Cargo workspace members / exclude consistency" (TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1 pattern) | One-off configuration bug; not a recurring substrate-corruption pattern. |

### 5.2 Conventions (project-specific, contextual)

| Pattern | Why it's a convention |
|---|---|
| "Use `Vsa16kF32` for switchboard, `Vsa16kBF16` for AMX hot path" | Format-selection convention. The *iron* rule is I-VSA-IDENTITIES (operate on identities); choice of carrier precision is per-workload. |
| "Worker outputs include their lib.rs hunk OR aggregation worker spawns last" | Sprint-coordination convention. Variable across sprint topology. |
| "Plan documents at `.claude/plans/*-v<N>.md` with APPEND-ONLY versioning" | Workspace convention. |

### 5.3 Knowledge (descriptive, not prescriptive)

| Pattern | Why it's knowledge |
|---|---|
| The three-zone hot-path model (Zone-1 thinking-engine MatVec + AriGraph entity_index) | Architectural description; informs decisions but does not constrain them. Lives in `LATEST_STATE.md` and knowledge docs. |
| The 14-paper landscape and `[FORMAL-SCAFFOLD]` five pillars | Reference scaffolding; cite when load-bearing, otherwise dormant. |

### 5.4 The line between style and iron

A pattern crosses from style to iron when it satisfies BOTH:

1. **N ≥ 3 violation count** in observed bug data (PR fixes, codex
   findings, sprint-log entries) — this distinguishes "could happen" from
   "does happen repeatedly." `I-LEGACY-API-FEATURE-GATED` crossed this
   line with 4 instances in PR #383 alone.
2. **Substrate-level consequence** — the violation corrupts substrate
   guarantees (Markov, Berry-Esseen, identity-vs-content, layout-bit
   isolation). Style violations cause friction; iron-rule violations
   cause silent corruption.

The "every v1 API path under v2 feature" pattern WAS promoted
(sprint-12). The related "every deprecated method needs a migration
pointer" pattern is softer — missing pointers cause user friction, not
substrate corruption. Promotion to iron status requires the second
criterion, not just the first.

### 5.5 Candidates currently under N ≥ 3 watch

Sprint-13 onwards should track FINDING accumulation against the
promotion checklist. Current watch list (informational; promotion not
yet recommended):

| Candidate | Current N | Notes |
|---|---|---|
| ABI-stability under codebook rebase | N = 1 (one D5 Frankenstein-adjacent observation) | Watch for second + third instances. |
| Memory-ordering for the contract bus | N = 0 confirmed | No observed bug pattern; convention coverage may be sufficient. |
| Workspace member/exclude consistency | N = 2 (cognitive-shader-driver + one prior) | Approaching threshold; track in TECH_DEBT until N ≥ 3 + substrate-level consequence. |
| Mandatory board-hygiene rule (E-META-9) | N = 2 (PR #381 + the 2026-04-20 PR #223/#224/#225 gap) | Process discipline; promotion path may be "workflow iron rule" rather than substrate. |

---

## 6. Cross-References

### Canonical iron rules

- `CLAUDE.md §I-SUBSTRATE-MARKOV` — substrate operator iron rule
- `CLAUDE.md §I-NOISE-FLOOR-JIRAK` — statistical model iron rule
- `CLAUDE.md §I-VSA-IDENTITIES` — data semantics iron rule
- `CLAUDE.md §I-LEGACY-API-FEATURE-GATED` — API version iron rule
  (pending sprint-13 ratification; promotion text in CSI-18)

### Originating EPIPHANIES findings

- `EPIPHANIES.md` 2026-04-20 E-SUBSTRATE-1 — VSA-bundling guarantees
  Chapman-Kolmogorov by construction
- `EPIPHANIES.md` E-ORIG-7 — Jirak Berry-Esseen under weak dependence
  IS the Phase-5 noise-floor lemma (folded into `[FORMAL-SCAFFOLD]`
  four-pillar entry)
- `EPIPHANIES.md` 2026-04-21 — VSA operates on identities, not content
  — the refined iron rule (+ 2026-04-21 CORRECTION-OF entry on D5
  Frankenstein)
- `EPIPHANIES.md` E-META-10 — v1-API-under-v2-feature alias pattern
  (pending prepend per W-F11; promotion text in
  `sprint-log-11/meta-review-opus.md` §CSI-18)

### Meta-pattern observation and risk carry-forward

- `.claude/board/sprint-log-11/meta-review-opus.md` §CSI-18 — the
  Wave G Opus review entry that named the meta-pattern across the four
  iron rules and recommended I-LEGACY-API-FEATURE-GATED promotion
- `.claude/plans/cognitive-substrate-convergence-v2.md §13.8 —
  v1-API-under-v2 alias anti-pattern (E-META-10)` — the sprint-13 risk
  matrix entry carrying the pattern forward
- `.claude/board/sprint-log-11/meta-review.md` §E-META-10 NEW — the
  Sonnet draft entry that defined the FINDING and the doctrinal claim

### Supporting doctrine

- `.claude/knowledge/vsa-switchboard-architecture.md` — the full
  three-layer VSA architecture (switchboard carrier / domain role
  catalogues / content stores), the decision matrix referenced by
  I-VSA-IDENTITIES
- `.claude/knowledge/i4-substrate-decisions.md` — sprint-11
  implementation outcomes for the i4 substrate (L-1..L-20 locked
  decisions); referenced for the substrate context that
  I-LEGACY-API-FEATURE-GATED protects
- `.claude/knowledge/lab-vs-canonical-surface.md` — the AGI-is-the-
  struct-of-arrays doctrine (I1-I11 invariants) that the four iron
  rules collectively bound
- `FormatBestPractices.md` (referenced by I-VSA-IDENTITIES) —
  Jirak-grounded per-workload decision matrix

### Codex P1 / pre-merge gate

- `.claude/board/sprint-log-11/meta-review-opus.md` §"Honest Reflection"
  point 2 — "every codex review catches the same v1-API-under-v2
  anti-pattern"
- PR #383 commits `42b3215` + `b44ce87` — the four-instance
  observed-bug record that established N ≥ 3 for E-META-10
- `crates/causal-edge/tests/v2_layout_tests.rs` — the reference
  field-isolation matrix artifact (16 tests)

---

## 7. Doctrine Versioning

This doc is APPEND-ONLY at the section level. To add a fifth iron rule:

1. Add a new §2.N entry with the per-rule analysis table filled in.
2. Update §1's axis table.
3. Append a §6 cross-reference.
4. Do NOT delete existing entries; iron-rule retractions append a
   `SUPERSEDED` annotation to the relevant §2.N table row, with a
   pointer to the retraction PR.

The promotion checklist in §3 is also APPEND-ONLY — refinements add
new checkboxes; existing ones are not removed.

**Version:** v1 (sprint-13 preflight PP-2, 2026-05-16)
**Successor:** None planned. v2 would be authored only if the
meta-pattern itself shifts (e.g., a fifth axis is identified that
breaks the current four-axis framing).
