# Sprint-12 Wave G Meta-Review — OPUS Honest Cross-Cutting Review (Opus 4.7, W-Meta-Opus, 2026-05-16)

> **Scope:** Independent Opus review of sprint-12 Wave G fleet (6 Sonnet workers, 4 commits on top of main `b526485`). Sibling to my own sprint-11 / Wave F file `meta-review-opus.md` in the same directory; this file extends the CSI numbering (CSI-1..13 prior → CSI-14..18 here).
>
> **Authority:** W-Meta-Opus (main-thread spawn, Opus 4.7 per Model Policy). I verified each Wave G output against the working tree at HEAD, not against worker self-reports.
>
> **Predecessor:** `.claude/board/sprint-log-11/meta-review-opus.md` (Wave F). The Wave F sprint-11 grade was **B** (revised down from W-F10's B+ by three blocker-class registration gaps CSI-7/8/9).

---

## 1. Executive Summary

### Wave G grade: **A−**
### Sprint-12 grade-so-far: **B+** (Wave F integration debt + Wave G follow-through = net positive trajectory)

**Headline:** Wave G is the discipline correction Wave F earned. The six workers shipped the QualiaColumn cutover (D-CSV-5b), the CAM-PQ-indexed WitnessCorpus (D-CSV-6b HashMap surface), the i4 batch evaluation API (D-CSV-13 scaffolding), the Jirak-derived Σ-tier thresholds (D-CSV-15 partial — math derivation done; principled VAMPE pairing still sprint-13+), the I-LEGACY-API-FEATURE-GATED iron-rule promotion, and the one-line cognitive-shader-driver workspace fix. **All six workers stayed in lane.** No new CSI-7/8/9-class blockers materialized. The workspace fix (W-G6) and the Jirak math correction (W-G4) both materially exceed the floor a Sonnet worker is expected to hit.

**Why A− not A:**

- W-G2 ships `CamPqWitnessIndex` (the name pre-commits to CAM-PQ semantics) but the backing store is a `HashMap<u64, Vec<usize>>` placeholder. The naming/implementation mismatch is acknowledged in the doc comment but the name will read as load-bearing once consumers attach. (CSI-15 below.)
- W-G3's `mul_assess_vec` convenience wrapper allocates inside the loop; this is consistent with its `_vec` name but worth flagging because the surrounding 5 `_batch` functions are explicitly SIMD-shaped. The `_vec` wrapper has only ONE length assert (qualia/mantissas), not two — but it allocates the output, so the second assert isn't applicable. Correct as designed; the cross-cutting finding is naming hygiene (CSI-16).
- The eight new W-G4 Jirak tests are good, but the test that compares `jirak_p(4.0) > jirak_p(3.0)` uses variance of inter-tier deltas as a convexity proxy. That is a reasonable surrogate but not a Jirak-mathematical statement; the test would pass for any α>1 spacing. Not wrong, just under-claims the spec's "p ≥ 4 asymptotically iid" assertion.

**Why not A:** Wave G is not just clean delivery; it actively repaired three Wave F gaps:
1. W-G6 fixes `TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1` (moves cognitive-shader-driver from `exclude` to `members` — the same issue I called out as CSI-7's structural sibling).
2. W-G5 promotes E-META-10 to iron rule `I-LEGACY-API-FEATURE-GATED` per my Wave F §4 recommendation — the recommendation was explicit and was acted on.
3. W-G1 completes the D-CSV-5b cutover end-to-end (bindspace.rs, engine_bridge.rs, driver.rs all touched in a single coordinated commit) — this is exactly the "main-thread aggregation pass" pattern Wave F botched.

---

## 2. Per-Worker Grade Table

| Worker | Grade | Key finding |
|---|---|---|
| **W-G1** | **A−** | QualiaColumn → QualiaI4Column cutover lands cleanly across `bindspace.rs` (BindSpace.qualia field + Builder + 5 new D-CSV-5b tests), `engine_bridge.rs` (dispatch_busdto + write_qualia_observed + read_qualia_decomposed all convert at boundary via `.to_f32_17d()` / `from_f32_17d()`), `driver.rs` (qualia reads converted at the call site, alpha_composite hit_qualia_f32 pre-materialized for closure lifetime). `QualiaColumn` is correctly marked `#[deprecated(since = "0.2.0")]` with migration pointer. 18 tests in bindspace.rs (was 13 pre-cutover) — net +5 i4 tests, no test loss. CSI-14 check below: ALL non-test/non-comment QualiaColumn references in the crate are in deprecation context (the from_f32 bulk converter + the meta-test verifying the deprecation attribute is present). No leftover live f32 qualia writes. |
| **W-G2** | **B+** | WitnessCorpus CAM-PQ-indexed HashMap surface is functionally correct (15 tests; iter / query / cam_pq_search / evict_stale_before / Arc-CoW all green). The doc comment correctly cites the upstream blocker (ndarray::hpc::cam_pq operates on 256D+ float vectors / GraphHV, not u64 SPO → Vec<usize>). Module is registered in `arigraph/mod.rs` (`pub mod witness_corpus; pub use witness_corpus::{CamPqWitnessIndex, WitnessCorpus, WitnessEntry, WitnessId};` — discipline that the Wave F equivalents failed at). **Downgrade reason:** the type name `CamPqWitnessIndex` (CSI-15) pre-commits to CAM-PQ semantics that the HashMap doesn't have. A real CAM-PQ index would rank by distance from a query SPO vector; this returns insertion order. The doc says so honestly, but the name will mislead future consumers. |
| **W-G3** | **A−** | Five batch functions (`dk_position_batch`, `trust_texture_batch`, `flow_state_batch`, `gate_decision_batch`, `mul_assess_batch`) + one `mul_assess_vec` convenience wrapper. All five `_batch` functions take parallel `&[A]`, `&[B]`, `&mut [C]` slices with TWO length asserts each (qualia/mantissas + input/output). The single-input `trust_texture_batch` correctly has only ONE assert (no mantissas). The `_vec` wrapper allocates and has only one assert as expected. Zero allocations in the 5 hot-path batch functions. 6 batch tests + 1 length-mismatch panic test + 1 empty-input test. The contract is exactly what AVX-512 lane intrinsics target. |
| **W-G4** | **A** | Worker corrected my spec error. The original (Wave F → Wave G) brief said "p ≥ 4 collapses linear"; that was inverted. The correct Jirak 2016 statement is: rate is `n^(p/2-1)` for `p ∈ (2,3]`, and `n^(-1/2)` in L^q for `p ≥ 4`. The worker derived `Σ_k = k^(p/2) / 10^(p/2)`, normalized so Σ10 = 1.0 exactly. **Spot-check verified:** for p=3, Σ1 = 1/10^1.5 ≈ 0.031623 (matches `test_jirak_default_endpoints`), Σ5 ≈ 0.353553, Σ10 = 1.0 (anchored). The Jirak 2016 citation (arxiv 1606.01617) is present in both the module-level doc comment (line 16) AND the `jirak_p` method doc comment (line 100). `Default::default()` returns the Jirak-derived bands; the sprint-11 hand-tuned linear values are preserved as `SigmaTierBands::hand_tuned()` for backwards comparison. 12 pre-existing tests + 8 new Jirak tests = 20 total. All 12 pre-existing tests use `default_bands()` (now marked `#[deprecated]`) under `#[allow(deprecated)]` — disciplined backwards compatibility. |
| **W-G5** | **A** | (1) CLAUDE.md now has FOUR iron rules: I-SUBSTRATE-MARKOV, I-NOISE-FLOOR-JIRAK, I-VSA-IDENTITIES (all sprint-11) + the new I-LEGACY-API-FEATURE-GATED (this PR). The new rule is well-scoped (it specifies the 5 codex P1 catches by number, mandates field-isolation matrix tests at layout-bit boundaries, and points to E-META-10 + CSI-2 + the i4-substrate-decisions knowledge doc). (2) EPIPHANIES.md E-META-10 has the "PROMOTED to iron rule" header in the **Status (2026-05-16):** line — append-only discipline preserved. (3) TECH_DEBT.md now has TD-LEGACY-API-FEATURE-GATED-RESOLVED-1 at the top, marking the resolution chain. The main-thread consolidation (worker initially wrote to repo-root TECH_DEBT.md; main thread moved it to `.claude/board/TECH_DEBT.md`) is exactly the kind of aggregation Wave F was missing. |
| **W-G6** | **A−** | One-line resolution to TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1: moved `crates/cognitive-shader-driver` from `exclude` to `members` in `Cargo.toml` with a comment citing the TD ticket. This unblocks `cargo build -p cognitive-shader-driver` from workspace root — the same friction class as CSI-7 (sigma-tier-router). Grade is A− not A only because the symmetric fix for sigma-tier-router (CSI-7 from Wave F) is not in this PR — it's queued for a separate aggregation commit. The crate moved IS in the workspace `members` list now. |

**Test count rollup:** Total #[test] markers across the six modified files: **99**. Breakdown: bindspace.rs 18, sigma-tier-router 20, mul.rs 23 (8 batch + 15 i4_eval scalar), witness_corpus.rs 15, engine_bridge.rs 10, driver.rs 13. Per-worker test additions versus pre-Wave-G baseline: W-G1 added 5 (D-CSV-5b cutover), W-G2 added 7 (CamPqWitnessIndex + index-aware query/evict tests), W-G3 added 8 (batch parity + panic + empty), W-G4 added 8 (Jirak math). Wave G test delta: **+28 unit tests** toward the sprint-10 1550 Miri target.

---

## 3. Cross-Cutting Findings (CSI-14..18)

These extend the CSI-1..13 numbering from sprint-11 / Wave F. Each is verified against the working tree at HEAD.

### CSI-14 (CONFIRMED OK) — W-G1 QualiaColumn deprecation covers all live call sites in cognitive-shader-driver

**Verification:** grep'd `crates/cognitive-shader-driver/src/` for `QualiaColumn`. Five matches, all in `bindspace.rs`:

- Line 141: `pub struct QualiaColumn(pub Box<[f32]>)` — preceded by `#[deprecated(since = "0.2.0", note = "use QualiaI4Column directly; this f32 column was retired in D-CSV-5b cutover")]`. ✓
- Line 144: `impl QualiaColumn { ... }` — under `#[allow(deprecated)]`. ✓
- Line 198: `pub fn from_f32(qualia_f32: &QualiaColumn) -> Self` (on QualiaI4Column) — migration helper, takes the deprecated type as input. ✓
- Line 697: in test `test_qualia_column_deprecation_warning_present`, inside `#[allow(deprecated)]` block. ✓
- Line 700: same test, asserting the deprecated type still allocates during the deprecation cycle. ✓

No fresh f32 qualia writes. `engine_bridge.rs` and `driver.rs` reads convert at the boundary via `q_i4.to_f32_17d()`. `lib.rs` re-exports `QualiaColumn` under `#[allow(deprecated)]` with a comment "deprecated — use QualiaI4Column".

**External crates still referencing QualiaColumn:** `lance_graph_ontology::lance_cache.rs:41` (doc comment only) and `lance_graph_callcenter::transcode::mod.rs:12` (doc comment only). No live code paths.

**Status:** no fix needed. W-G1 stayed in lane and the cutover is complete in this crate. This is a positive finding — the kind that exposes well-scoped work.

### CSI-15 (P2) — W-G2's `CamPqWitnessIndex` name pre-commits to CAM-PQ semantics the HashMap doesn't have

**Files:** `crates/lance-graph/src/graph/arigraph/witness_corpus.rs:101-167` (the type) + `mod.rs:18` (the re-export).

The type is named `CamPqWitnessIndex` but the backing store is `HashMap<u64, Vec<usize>>` — a hash bucket from packed SPO to entry positions. A real CAM-PQ index would (a) operate on multi-dimensional vectors not u64 keys, (b) return distance-ranked top-k, and (c) consult the `ndarray::hpc::cam_pq::CamPqCodec`. The current implementation does none of these. The doc comment (lines 90-100) is honest about this: "this PR ships a HashMap-backed index as the canonical surface" and "TECH_DEBT: upgrade `by_spo` HashMap to ndarray::hpc::cam_pq once upstream adds SPO witness-tuple support."

**Why this matters:** the type is `pub` and re-exported from `arigraph::mod.rs`. Once consumers depend on `CamPqWitnessIndex` as a type name, renaming becomes ABI-breaking. The current `cam_pq_search(spo, k)` method is documented as returning "first k entries in chain order; no distance ranking is applied" — sprint-13+ when ndarray's codec lands, the method semantics will change (distance ranking) but the name will already imply distance ranking, so the "before/after" delta will be silent at the call site.

**Recommendation:** rename the HashMap-backed type to `WitnessIndexHashMap` (or `WitnessIndexByExactSpo`) and reserve `CamPqWitnessIndex` for the sprint-13+ ndarray-codec backed type. ~30 LOC churn (rename + doc updates + tests). **Sprint-12 P2 housekeeping; the rename is cheaper now than after consumers attach.** This is the kind of premature naming-claim that creates ABI archaeology debt.

### CSI-16 (CONFIRMED OK) — W-G3 batch API length-assertion discipline is complete

**Verification:** all FIVE `_batch` functions assert BOTH `qualia.len() == mantissas.len()` AND `qualia.len() == out.len()`:

- `dk_position_batch` lines 644-646 — two asserts ✓
- `trust_texture_batch` line 654 — one assert (no mantissas; trust_texture is qualia-only by design) ✓
- `flow_state_batch` lines 662-663 — two asserts ✓
- `gate_decision_batch` lines 671-672 — two asserts ✓
- `mul_assess_batch` lines 680-681 — two asserts ✓

The `mul_assess_vec` convenience wrapper at line 689 has only ONE assert (qualia/mantissas) because it ALLOCATES the output (no `&mut [T]` to size-check). The `_vec` naming makes this explicit; no allocations leak into the hot path because the 5 SIMD-shaped functions all take pre-allocated `&mut [T]`.

**Status:** no fix needed. The original spec language ("all 5 batch functions; verify the panic for length mismatch") is met. There is also a positive `test_batch_panic_on_length_mismatch` test asserting the panic fires. The empty-input test (`test_batch_empty_input_returns_empty_output`) covers the n=0 edge case for all six surfaces.

### CSI-17 (LOW) — Wave F → Wave G Jirak spec error did not propagate beyond the worker brief

**Verification:** grep'd the workspace for `Jirak` co-occurring with `p.*4` or `p ?[≥>=]+ ?4`. Results (live code/docs only):

- `CLAUDE.md:307`: "`n^(p/2-1)` for `p ∈ (2,3]`, `n^(-1/2)` in L^q for `p ≥ 4`" — **CORRECT** statement of Jirak's two regimes. ✓
- `FormatBestPractices.md:90`: same correct statement. ✓
- `crates/jc/src/jirak.rs:12-13`: "Jirak's theorem gives the correct rate: n^(p/2-1) for p ∈ (2,3] ... For p ≥ 4 the [classical rate holds]" — **CORRECT**. ✓
- `crates/sigma-tier-router/src/lib.rs:69`: "`n^(-1/2)` in L^q for `p ≥ 4`  (the 'asymptotically iid' regime)" — **CORRECT** (this is W-G4's own derivation). ✓

The original "p ≥ 4 collapses linear" framing in my Wave G brief was inverted; the workspace itself (CLAUDE.md, FormatBestPractices.md, jc/jirak.rs) had the correct statement the whole time. W-G4 corrected the brief by reading the workspace's own iron rule, which is exactly the right discipline.

**Status:** no fix needed. This CSI confirms the worker corrected the spec by consulting CLAUDE.md — the consult-don't-guess pattern worked. **Positive finding.** If the brief had been followed verbatim, the bands would be wrong; the worker's diligence is grade-A behavior.

### CSI-18 (MED) — Four iron rules now span four axes; doctrine consolidation deferred to sprint-13

**Observation:** CLAUDE.md as of HEAD has FOUR iron rules:

1. **I-SUBSTRATE-MARKOV** — VSA bundling guarantees Chapman-Kolmogorov in d=10000 by construction. (2026-04-20)
2. **I-NOISE-FLOOR-JIRAK** — Bits are weakly dependent; use Jirak 2016 rates not classical Berry-Esseen. (2026-04-20)
3. **I-VSA-IDENTITIES** — VSA operates on identity fingerprints that POINT TO content; never on bitpacked content. (2026-04-21)
4. **I-LEGACY-API-FEATURE-GATED** — v1 API paths under v2-layout features must route through canonical mapping or feature-gate to no-op with migration pointer. (2026-05-16, W-G5)

Each rule pattern says, in compressed form: **"Do X consistently across the codebase; document deviations explicitly."** The Markov rule says "don't replace bundle with XOR without consulting [FORMAL-SCAFFOLD]". The Jirak rule says "cite weakly-dependent Berry-Esseen, not classical, on every threshold". The VSA-Identities rule says "use the register before reaching for VSA; bundle identities, not content". The Legacy-API rule says "the same function name MUST NOT silently produce different semantics under different feature flags."

**Emerging meta-rule:** every iron rule formalizes a discipline against silent drift across some axis (substrate operator / statistical model / data semantics / API version). The pattern is the same; only the axis differs.

**Recommendation for sprint-13:** consolidate doctrinal commentary in a single `.claude/knowledge/iron-rules-doctrine.md` knowledge doc cross-referencing all four rules + the meta-pattern. ~250 LOC of synthesis. The current arrangement (four scattered iron-rule sections in CLAUDE.md + per-rule cross-refs to EPIPHANIES/TECH_DEBT) works but does not surface the meta-pattern that a new iron rule should fit the "no silent drift across axis X" template. Sprint-13 doctrinal worker would benefit from this anchor.

---

## 4. Sprint-12 Grade-So-Far + Sprint-13 Spawn Decision

### What's shipped in sprint-12 (Wave F + Wave G combined)

From the cognitive-substrate-convergence-v2 plan §11:

| Phase | D-id | Status post-Wave-G | Notes |
|---|---|---|---|
| A | D-CSV-1/2/3/4 | Shipped (PR #383/#384) | sprint-11 |
| B | D-CSV-5a | Shipped (PR #385) | sprint-11 Wave F |
| B | **D-CSV-5b** | **Shipped (Wave G W-G1)** | QualiaColumn cutover complete in cognitive-shader-driver |
| B | D-CSV-6a + D-CSV-7 | Shipped (PR #386) | sprint-11 Wave F |
| B | **D-CSV-6b** | **Shipped (Wave G W-G2, HashMap surface)** | Full CAM-PQ codec sprint-13+ (CSI-15) |
| C | D-CSV-8 + D-CSV-9 | Shipped (PR #387) | sprint-11 Wave F |
| C | D-CSV-10 | Shipped (Wave F W-F1 + Wave G W-G4) | Jirak-derived bands replace hand-tuned baseline |
| C | **D-CSV-13 (SIMD vectorization)** | **Scaffolded (Wave G W-G3)** | Batch API contract; AVX-512/NEON intrinsics queued sprint-13 |
| D | D-CSV-11 | In PR (sprint-11 Wave F W-F4/5/6, needs ndarray CSI-9 fix) | cross-repo blocker remains |
| D | D-CSV-12 | In PR (sprint-11 Wave F W-F7) | on-Think methods D-CSV-14 sprint-13+ |
| E | D-CSV-15 (Jirak Σ10 threshold) | Partially shipped (Wave G W-G4 math; VAMPE coupled-revival sprint-13+) | TD-SIGMA-TIER-THRESHOLDS-1 resolution path opened |

**What's left for sprint-12:** the AVX-512/NEON intrinsics for D-CSV-13 (Wave G shipped the batch API contract; the intrinsic backing is sprint-12 follow-on) + the ndarray cross-repo aggregation PR for CSI-9 (still a hard blocker on D-CSV-11 productization).

**What carries to sprint-13:** D-CSV-14 (on-Think method migration), D-CSV-15 full VAMPE coupled-revival, the `CamPqWitnessIndex` rename (CSI-15), the iron-rules doctrine consolidation (CSI-18), and the real ndarray::hpc::cam_pq witness-tuple wiring (sprint-13+ dependency on upstream ndarray work).

### Pre-spawn checklist for sprint-13

Recommend spawning sprint-13 AFTER the following pre-fleet hygiene:

1. **CSI-15 rename** (~30 LOC): rename `CamPqWitnessIndex` → `WitnessIndexHashMap`; reserve the CAM-PQ name for sprint-13+'s ndarray-codec backed type. **Cheaper now than after consumers attach.**
2. **CSI-7 follow-through** (~3 LOC, sister to W-G6): add `sigma-tier-router` to parent workspace `members` similarly to how W-G6 added cognitive-shader-driver. The sigma-tier-router Cargo.toml may still declare a standalone `[workspace]`; verify and remove if so. Wave F left this open.
3. **CSI-9 cross-repo PR** (~4 LOC in ndarray): register `qualia` + `splat_field` in `/home/user/ndarray/src/hpc/stream/mod.rs`. Blocker on D-CSV-11 productization; coordination with AdaWorldAPI/ndarray upstream required.
4. **CSI-18 doctrine doc** (~250 LOC): consolidate the four iron rules into `.claude/knowledge/iron-rules-doctrine.md` to anchor sprint-13 doctrinal workers.

Items 1+2+4 are local; item 3 is the only true cross-repo blocker. Standing user ratifications from Wave F + Wave G (OQ-CSV-6 Jirak math now done, D-CSV-5b cutover complete, E-META-10 promoted) remain valid.

---

## 5. Final Reflection

Wave G is what Wave F should have been. Six workers, six in-lane deliveries, three Wave F debt items actively repaired (workspace conflict, E-META-10 promotion, D-CSV-5b cutover follow-through), no new blocker-class regressions. The two soft findings (CSI-15 naming pre-commitment and CSI-16 doctrine consolidation deferred) are sprint-13 housekeeping, not Wave G failures. The Jirak math correction (W-G4) is the standout — the worker noticed the brief was wrong, consulted CLAUDE.md's own iron rule, derived the correct math, and shipped it with eight new tests. That is exactly the consult-don't-guess discipline this workspace's CCA2A pattern is meant to produce. Sprint-13 should pick up CSI-15 / CSI-7-symmetric / CSI-18 / CSI-9 as the four pre-spawn hygiene items, and the doctrinal anchor (CSI-18) should be the first knowledge-doc deliverable so sprint-13's workers have a single place to land "this is how the four iron rules compose." The structural improvement Wave G earned is that the fleet template now demonstrably **can** ship clean integration commits when the worker prompts include the registration discipline — Wave G is the proof. Wave F was not.

---

## 6. Cross-references

- **Wave F Opus meta-review (predecessor):** `.claude/board/sprint-log-11/meta-review-opus.md` — sprint-11 grade B, CSI-1..13
- **Sprint-10 meta-review (format precedent):** `.claude/board/sprint-log-10/meta-review.md`
- **Convergence plan v2:** `.claude/plans/cognitive-substrate-convergence-v2.md` (D-CSV-* deliverable table)
- **i4-substrate-decisions knowledge:** `.claude/knowledge/i4-substrate-decisions.md` (W-F11)
- **Iron rules:** `CLAUDE.md` §Substrate-level iron rules — I-SUBSTRATE-MARKOV + I-NOISE-FLOOR-JIRAK + I-VSA-IDENTITIES + I-LEGACY-API-FEATURE-GATED (new this wave)
- **Wave G worker outputs (HEAD at 67c2ca8):**
  - W-G1: `crates/cognitive-shader-driver/src/bindspace.rs` + `engine_bridge.rs` + `driver.rs` (D-CSV-5b cutover)
  - W-G2: `crates/lance-graph/src/graph/arigraph/witness_corpus.rs` + `mod.rs` (CAM-PQ HashMap surface)
  - W-G3: `crates/lance-graph-contract/src/mul.rs` (batch API, 5 + 1 functions)
  - W-G4: `crates/sigma-tier-router/src/lib.rs` (Jirak bands as Default::default())
  - W-G5: `CLAUDE.md` (I-LEGACY-API-FEATURE-GATED) + `.claude/board/EPIPHANIES.md` (E-META-10 promotion header) + `.claude/board/TECH_DEBT.md` (TD-RESOLVED-1 entry)
  - W-G6: `Cargo.toml` (cognitive-shader-driver: exclude → members)

---

*End of sprint-12 Wave G Opus meta-review. W-Meta-Opus (Opus 4.7), main-thread, 2026-05-16. Authored after independent verification pass on the 4 Wave G commits (7d7b537, 03ce219, 291878f, 67c2ca8) against working tree at HEAD. Grades are independent of worker self-reports.*
