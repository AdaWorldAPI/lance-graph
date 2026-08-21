## stage-3 — RE-SCOPED 2026-08-20: S3.0 closed as NOT-NEEDED; the ladder's empty column is HYDRATION, not ADDRESS

PR #973 (S3.0 address, first attempt) closed unmerged. Its retraction PR then
attempted a corrected `CausalLiteral` and was **stopped before merge by the
operator** — four unearned claims, recorded in
`E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1`. No new
address type is minted: `IdentityQuad` (4 × u24 in one V3 facet, ratified
2026-08-17) already carries exact four-component identity, and no falsifier
showed existing addressing insufficient.

| D-id | Deliverable | Status |
|---|---|---|
| S3.0 | exact literal address | **CLOSED — NOT NEEDED** (use `IdentityQuad` / `ClassAddr` / V3 rail) |
| S3.0b | qualified causal regime | Queued |
| S3.1 | hydrate `CausalMeta` + `EpistemicMeta` over EXISTING addresses | **Next** |
| S3.1b | EntropyWork · BasinSet · Attention; `RowFocusMask × WideFieldMask` | Queued |
| S3.2 | V3 / CE64 leg from the hydrated node | Queued |
| S3.3 | `ResolvedPredicate`; unknown fails CLOSED; explicit composition | Queued |
| S3.4 | DisMech ORACLE experiment — hide intermediates, hydrate neighbourhood, NARS recovers, compare vs truth | **the gate** |
| S3.5 | NARS evidence mass: raw/source/HEEL/effective, deterministic W+/W− | Queued |
| S3.6 | JC measurement per predicate × cohort × horizon × instrument | Queued |
| S3.7 | `ReasoningEpisode`; measure the 17 confidence-mute kernels | Queued |
| S3.8 | potholes, first_possible vs first_derived, strict historical replay | Queued |
| S3.9 | Meta / Rubicon → OGAR-loco; Pearl qualification earned by receipts | Queued |

**Open, and NOT an S3 deliverable:** `ClassId = u16` near-exhausted for
relations (MedCare-rs commitment #10) — an OGAR/lance-graph classid-mint
capacity question for the operator.

---

### D-CV3-* — DisMech × Causality-V3 (plan `dismech-causality-v3-v1.md`, 2026-08-21)

| D-id | Scope | Repo | Status | Falsifier |
|---|---|---|---|---|
| D-CV3-0 | Pin corpus (`/workspace/dismech` @`557e15436`) + emit 3 frozen TSVs from a typed parse; no new types | MedCare-rs | **Queued** | fresh container reproduces 2,449 / 4,076 / 361 exactly |
| D-CV3-1 | Splits A (random edge) + B (disease-held-out, 534 groups) as committed artifacts | MedCare-rs | **Queued** — gates on D-CV3-0 | group-disjointness; held-out share 15-25% |
| D-CV3-2 | Level-0 scorer: Recall@K + MRR + abstention, structural only | MedCare-rs | **Queued** — gates on D-CV3-1 | non-trivial on BOTH arms; can-fire + can-stay-silent pair |
| D-CV3-3 | `HoleV3` as `ValueTenant = 16`; `awareness_state` orthogonal to `unknown_kind`; NOT in CE64 (0 free bits) | lance-graph | **Queued** — gates on D-CV3-2 green | field-isolation matrix; `ENVELOPE_LAYOUT_VERSION` unchanged; two-axis independence round-trip |
| D-CV3-4 | Producer: `dismech_evidence` -> hole rows (its first caller) | lance-graph | **Queued** — gates on D-CV3-3 | populated rows 0 -> 4,076 + 361, else a 5th EXISTS-UNCALLED carrier |
| D-CV3-5 | `Communities` x `EpisodicBasins` cross-validation (the only available unknown-unknown detector) | lance-graph | **Queued** | fires on a synthetic bridge AND stays silent on a coherent graph |
| D-CV3-6 | Call `reciprocal_rank_fusion` in `OsintRetriever::retrieve` (cheapest real integration) | lance-graph | **Queued** — gated on G0 | fused ranking differs from BFS-only on >=1 real query |


### D-ACR-* — Alpha-channel rung overlay (plan `alpha-channel-rung-overlay-v1.md`, 2026-08-21)

Fills `hhtl-thinking-tables-le-contract-v1.md` §2.3's empty **Rung ladder**
row. Mints no type. Six of nine brainstorm pieces already had homes; these are
the rest.

| D-id | Scope | Status | Falsifier |
|---|---|---|---|
| D-ACR-0 | Audit `attention_mask.rs`/`attention_mask_actor.rs`: residue carrier, or a name collision? Report only | **Shipped** 2026-08-21 — `.claude/ATTENTION_MASK_AUDIT_2026_08_21.md`. Verdict **EXISTS-UNCALLED** + a name collision: the shipped type is a *rename register file* (`causaledge64-mailbox-rename-soa-v1.md` §4), not a residue carrier | recorded EXISTS-UNCALLED (0 callers workspace-wide + 3 sibling repos); piece E regraded — not a basis for D-ACR-1 |
| D-ACR-1 | `RowFocusMask` — the one missing primitive (S3.1b names it; no crate contains it) | **Next** — D-ACR-0 cleared it; starts clean (no reuse of `AttentionMaskSoA`). Must state its basis first: `WideFieldMask` is `u8`-capped at 256 positions (loud refusal above), `FieldMask` silently drops `>= 64` — a row population is neither | can-fire AND can-stay-silent on non-trivial input |
| D-ACR-3 | The one-way invariant as a test: no ontology-owned write traces to a patient-tagged read through ANY call path (corrected from write-authorization-only after CodeRabbit found a session-derived value can flow to the ontology owner via a shared parameter/return, then be written as the owner's own act) | Queued — gates on D-ACR-1 | a write whose call graph includes a session-tagged read is the bug, even if the write itself is authored by the ontology owner |
| D-ACR-2 | Mint the Rung-ladder rail | Queued — gates on operator mint decision (HTT §8 Q3) | `rail_carving` gains its first non-default consumer |
| D-ACR-4 | Second-order row at the same address, separate table | Queued | a rung-2 read reconstructs where rung-1 looked, on a fixture with an independent answer |
| D-ACR-5 | 64k lowering | **BLOCKED** — dialectic V4's own gate (V0–V3 green at small scale) | — |

**Not claimed:** that the residue improves recall or finds needles. Graded a
pruner, never a proof.

## preparation-arc plan wave — 2026-08-19 (operator: "integration plans for all open arcs")

Five plans, each PROPOSED (no code — the reset charter's audit-first order
holds; ARC-B is the ratification vehicle for the operator's "perfect shape").
Deferred-for-missing-integration ideas are a MANDATORY section in each plan.

| D-id family | Plan | Repo | Status |
|---|---|---|---|
| D-HTT-1..11 | `hhtl-thinking-tables-le-contract-v1.md` — ARC B LE addressing contract + thinking-table rows; ARC C seam gated | lance-graph | **Proposed** (awaits operator ratification) |
| D-CFR-1..6 | `counterfactual-rung3-closure-v1.md` — contract InferenceType widening, stopgap retirement, L1-mask probe, Pearl-2³ disambiguation | lance-graph | **Proposed** |
| D-MAR-1..2 | `mask-algebra-revision-read-v1.md` — FieldMask/WideFieldMask `difference`+`is_subset_of`; RevisionKind classification read | lance-graph | **Proposed** |
| D-EWU-1..9 | `ew64-witness-unification-v1.md` — ARC D/E: tenant-14 CausalWitness vs EpisodicWitness64 demarcation (READING L recommended, not banked), phase gates, OD-1..5 | lance-graph | **Proposed** |
| D-DCG-* | `causal-graph-soa-integration-v1.md` — dismech CausalGraph → SoA landing (relations-as-rows vs overflow) | dismech-rs | **Proposed** (authoring) |

## erasure-seals-compaction-research-v1 — RP-SEAL (operator charter; PASS 1 CONSOLIDATED + §0 STORNO; T0.1 F-PHYS-ORDER FIXED + T0.2 F-QREF-STRICT PINNED + T0.3 F-AWARENESS-LAG LANDED (no schema field — coordinates already durable) 2026-08-19; X-C2-1 LANDED (harness + controls green; C2 truth table confirmed at hash tier; wip dead-code finding); E2 RE-VERIFIED (FNV+b+1+fsync confirmed; scatter CONTESTED -> multi-machine sweep; PMU absent -> cache-miss metric struck) — TIER 0 COMPLETE; SEAL STORNO (accumulated-on-hot-path, 12 reqs) recorded + finalization map delivered — implementation awaits map ratification → docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md; Tier 0 probes next)

Plan: `.claude/plans/erasure-seals-compaction-research-v1.md` (the operator's
15-researcher program; supersedes lotus deliverables 3-9's sequencing).
Independent pass = background workflow `wf_ca974718-1b4`: 5 domains
(A Lance storage / B spatial layout / C erasure coding / D temporal /
E SoA-cache) x 3 roles (builder/adversary/scout); adversaries on the
strongest tier, builders+scouts on the grindwork tier; first pass blind to
docs/lotus/** and all boards (independence rule). Consolidation (evidence
matrix + cross-domain graph) and the tiered attack pass run main-thread
AFTER all 15 reports land. Reports: session scratchpad `rp-seal-v1/`,
committed at consolidation as Appendix H.

## lotus-seal-fractal-commit-frontier — research arc (PRE-REGISTERED 2026-08-18)

Charter: operator, 2026-08-18 (research + architecture; "first audit the
current code, build falsifiers, and try to kill the idea"). Docs live in
`docs/lotus/`. Verdict gate (A-E) before ANY implementation phase; no
metaphor becomes a type name without council approval.

| D-id | Deliverable | Status |
|---|---|---|
| D-LOTUS-1 | LOTUS-FRONTIER-AUDIT.md — Phase 0 archaeology, 4 lenses, VERIFIED/INFERENCE/HYPOTHESIS/BLOCKER graded; §6 answers the permeability/amortization question | **Shipped** — PR #961, 0cc171f |
| D-LOTUS-2 | F-ORD-REAL-FALSIFIER.md + the pre-registered tests (GREEN defect pin + `#[ignore]`d RED falsifier, `cycle_driver.rs`) | **Shipped** — PR #961, 0cc171f; RED landed BEFORE any fix, per charter |
| D-LOTUS-3 | LOTUS-SEAL-DESIGN.md (16⁴ tree, petal seal, H(parent)=H(children) — NEW design per audit §2.3) | Queued (post-verdict) |
| D-LOTUS-4 | COMMA-MODULATION-EXPERIMENT.md (petal-index-tier comma; locality-vs-resonance trade; "if it doesn't survive measurement, delete it") | Queued |
| D-LOTUS-5 | TEMPORAL-FRONTIER-SEAM.md (rung-qualified frontier visibility; hindsight-invariance falsifier; audit §6.4) | Queued |
| D-LOTUS-6 | PREPARED-ARTIFACT-PUBLICATION.md | **UNBLOCKED 2026-08-18** (operator pin ruling sanctioned the upstream-git source consult; exact v9.0.0 tag + current upstream on disk) — now routed through RP-SEAL Domain A rather than written solo, per the research charter |
| D-LOTUS-7 | BENCHMARK-PLAN.md (seal/append split; cycle-major vs tile-major k∈{1,4,16}; F-AMORT version-range scaling; F-MEM RAM high-water freeze-vs-petal; viewport partial hydration) | Queued |
| D-LOTUS-8 | PRE-REGISTERED-FALSIFIERS.md (the F-* catalogue) | Queued |
| D-LOTUS-9 | Final verdict A-E | Queued — gates all implementation |

## weather-soa-bake-v1 — Zarr → NodeRow, the missing bake (PRE-REGISTERED 2026-08-13)

Plan: `.claude/plans/weather-soa-bake-v1.md`. The arc's Python-over-Zarr phase
ran because the Zarr→`NodeRow` path was never specified —
`weather-substrate-poc-v2.md` names `crates/weather-poc` and describes **no bake
step**. Waves W0–W5; every bar pre-registered and committed BEFORE its run, each
with a control that can lose and a stay-silent twin.

| D-id | Deliverable | Wave | Status | Feeds |
|---|---|---|---|---|
| D-WXS-0 | classid mint for the weather-cell + statics classes (OGAR-side; `0x0F = Geo` exists, appid/classview open) | 0 | **Blocked (operator/OGAR)** | the bake must REFUSE to write under `0x0000_0000` |
| D-WXS-1 | field manifest v1 — (facet, pair, byte) → (variable, level, unit, floor id), a committed data artifact, ClassView-side | 0 | **SHIPPED 2026-08-13** — `data/field_manifest_v1.tsv` (22 rows = F0 5 pairs + F1/F2 3 pairs each, reserved slots emit NO row) + `manifest.rs`, 13/13; collision guard **disable-verified** (removed → only `colliding_entries_are_rejected` fails, BOTH stay-silent twins stay green). Bar B0's end-to-end half (mutating an entry changes written bytes) DEFERRED — the bake does not exist yet | slot purity §2; bar B0 |
| D-WXS-1a | variable census as a committed re-runnable probe (17 surface + 91 upper-air + 14 static = 122 fields; 92,044 six-hourly steps) | 0 | **SHIPPED 2026-08-13** — `era5_variable_census.py` + `.json`; `--selftest` PASS on all 10 constants, orchestrator-rerun independently; guard disable-verified (one constant broken → exit 1, correct message) | ends the chat-only-figure defect for the census |
| D-WXS-2 | key codec `(lat,lon) ↔ NodeGuid` — HEEL 16° tile / HIP within-tile / TWIG dormant; ragged tiles; lon-wrap range-SET | 1 | **SHIPPED 2026-08-13** — `key.rs`, 5/5 green; exhaustive 1,038,240-cell round-trip + collision-free; both bar-B1 halves **disable-verified** by the orchestrator (zeroing the HIP lat byte kills 3 tests incl. collision + ragged; removing the seam split kills the wrap twin while the non-wrap twin stays green) | a 16° box becomes a HEEL-prefix scan; bar B1 |
| D-WXS-2a | **NEW — row-major vs Morton, pre-registered comparison.** The shipped key assigns one WHOLE byte per axis; OGAR's cascade doctrine specifies the axis bytes **nibble-interleaved** (Morton). §1.2 deviated from the canon it cites and did not say so — now recorded as plan §1.3a. The prefix-scan claim holds under both; what differs is neighbour locality (`lat ± 1` is 1440 cells away under row-major) and how many ranges a non-tile-aligned box needs | 1 | **CLOSED by KILL 2026-08-13** — half A run, `examples/layout_probe.rs` + `.json`. **primary FAIL** (MORTON won ONE metric, not both: neighbour locality 16 vs 32 = 2x better, range count 212.50 vs 140.00 = ~1.5x worse). **control PASS** (CONTROL-BAD 3100 ranges / 15862 neighbour — the metrics do measure locality). **stay-silent twin PASS** (tile-aligned = exactly 1 range under BOTH, `[1,1,1,1]`). Pre-registered consequence applied: **no code change, no migration**. §1.3b's word "harmless" corrected in §1.3c — MORTON wins on exactly the metric half B would care about, so the honest reading is *no unambiguous win in key space* | gates any downstream assumption of Morton locality — measured against the ζ stencil (D-WXS-9), metric stated before the run |
| D-WXS-3 | shared canonical floor calibration (global 0.4–99.6 pct, frozen per epoch, stamped in dataset metadata) | 1 | **SHIPPED 2026-08-13** — `floor.rs`; bar B2 **disable-verified** (widening the "narrow" control floor kills only the control, twin stays green); version-stamp mismatch detected, ±½-bucket round-trip asserted. **AMENDED same day** (`E-A-TOTAL-FUNCTION-THAT-CANNOT-REFUSE-IS-A-CORRUPTION-PATH-1`): `saturation_of` folded non-finite input into the metric — `quantize` sends `NaN`/`-inf`→0 and `+inf`→255, all **rim** buckets, so an all-`NaN` population scored **1.0** ("fully saturated") where the truth is "no data at all". Now returns `SaturationScore {fraction, finite, non_finite}` — reported, never folded and never silently dropped. `calibrate` checked **CLEAN** (already filters `is_finite`) | bar B2 — this is the bar's own INSTRUMENT, so the defect would have corrupted a measurement, not a value |
| D-WXS-3b | **NEW — the L4 lane (pack/unpack ONE 16-byte facet).** The plan gave the lane a worker in §6.2 but **no D-id in §4's ladder** — it jumped D-WXS-3 → D-WXS-4. Added here as the pack/unpack half the bake will call | 1 | **SHIPPED 2026-08-13** — `lane.rs`, 33/33 crate-wide; 4 disables verified by the orchestrator (lo/hi swap → the swap test; hard-coded slot → 3 tests incl. manifest-load-bearing; version guard bypassed → the version test; unmapped slots emitting values → the reserved-slot test). The lane names no ERA5 variable in its own source — the caller's closure owns that **AMENDED same day (codex P1, PR #948):** `pack_facet` accepted non-finite readings; `quantize` maps them to valid-looking buckets, so a missing ARCO-ERA5 chunk (all-`NaN` — **valid store semantics**, and five W1 variables 404 at the arc's own fixture timestep) would have been stored as plausible low-bucket measurements. Now `LaneError::NonFiniteValue`, covering `±inf` too since they land on the rim. Disable-verified | precursor to bar B3; §2.6 slot purity as code |
| D-WXS-4 | the bake: one timestep → 1,038,240 NodeRows → ONE Lance version | 1 | Queued — **blocked behind D-WXS-0** (must refuse to write without a minted classid) | bar B3; the missing path |
| D-WXS-5 | statics bake — separate classid, separate dataset, exactly ONE version | 1 | Queued | bar B4; avoids ~1.3 PB of rewritten constants |
| D-WXS-6 | version-range read (`QueryReference::at(v,rung)` + `deinterlace`) + version-count scaling measurement | 2 | Queued | bar B5; KILL if growth is superlinear at 92,044 versions |
| D-WXS-7 | **D-WXA-5 re-homed and RE-SPECIFIED** — ρ(code_dist, field_dist) via `jc::reliability::spearman` over whole-grid pairs. (a) ρ ≥ 0.9996; (b) shuffled-codebook control < 0.98; (c) 16/64/256-level ladder must be MONOTONE before any verdict | 3 | **RUN 2026-08-13 — 12/12 PASS, all 3 real seasons.** `crates/weather-poc/examples/fidelity_probe.rs` (jc as dev-dep) against real live-fetched ARCO-ERA5 grid data (200k pairs/comparison). K×K ρ256 = 0.999909/0.999895/0.999684 — **does not replicate** the earlier smaller-scale near-miss (0.999556); shuffle ρ collapses to 0.02–0.024; ladder strictly monotone all 3 | poc-v2's ρ≥0.98 concern did not fire at grid scale — the earlier near-miss was fixture-scale, not grid-scale |
| D-WXS-8 | cross-variable comparability at grid scale (≥4 variables, ≥2 units, ≥3 seasons); per-variable-floor control must LOSE | 3 | **RUN 2026-08-13 — mixed, reported in full.** ⊘ **Figures corrected same day** (first written as control 16/16, primary 10/16 — both denominators wrong; there are **19** cross-unit pairs, counted from the JSON not from stdout by eye). Control (per-var must lose) **19/19 PASS** — KILL does not fire, ρ_pervar 0.245–0.939 vs ρ_shared 0.9987–0.9999 on identical pairs. Primary (ρ_shared≥0.9996) **9/19 PASS / 10 FAIL** (winter 2/9, spring 4/5, summer 3/5) — close misses (0.9987–0.9996) but **failing on a majority** of cross-unit pairs, concentrated in winter (the only season with `mean_sea_level_pressure`). Stay-silent twin: `\|diff\|≤0.0001` PASS at spring/summer, FAIL at winter (0.000174); **zero-empty-buckets FAILS all 3 seasons** (38/39/45 of 256) — the literal "zero" from the small-fixture claim does not hold at grid scale | P2 (0.9997 vs 0.857–0.875) **CONFIRMED at grid scale, 3 seasons, 19 cross-unit pairs** — the directional claim holds; the exact thresholds do not hold universally |
| D-WXS-9 | ζ = ∂v/∂x − ∂u/∂y as a substrate read (neighbour-key stencil) + the differencing-amplifies-quantisation falsifier, reported per ζ-magnitude decile | 4 | Queued | bar B8; KILL ⇒ ζ becomes its own baked lane |
| D-WXS-10 | **D-CZ-8 re-homed** — ζ-percentile regime bands over the WHOLE grid + coverage-matched donors + range-normalised `L`; shuffled-ζ control; same-band stay-silent twin | 4 | Queued | the grid buys CONTROL of the §7.9 confound, NOT its dissolution; KILL if `L̄`~range ρ≈1.0 survives matching |
| D-WXS-11 | MEASURE a full-grid bake wall time; state the ~8.1 s serial extrapolation as the prior and confirm/correct it in the same artifact | 5 | Queued | replaces an extrapolation from ONE cycle (514 ms / 65,536 owners); D-KIA-A2 still unbuilt |
| D-WXS-12 | jc ↔ ndarray reliability agreement (poc-v2 D-WXB-4, carried verbatim; jc is the authority, degenerate input reported not folded) | 5 | Queued | every Phase-A/C number is computed with jc |

**Capacity, corrected in the plan (§0.4).** "32 facets/cell" overstates the
usable budget: 2 slots are key+edges and `VALUE_TENANTS` carves the slab
contiguously — the committed assertion
(`canonical_node.rs` `value_tenants_contiguous_within_slab`) pins the current
Full carve at **188 of 480 B**, so the free budget is **292 B = 18 facets =
216 payload bytes**, not 384. All 122 fields still fit at 1 B/field — **one
cell is one node** — but at 2 B/field they do **not** fit as 4+12 facets.

> **Board-hygiene note (2026-08-13):** `weather-substrate-poc-v2.md`'s
> `D-WXA-*` / `D-WXB-*` / `D-WXC-*` ladder has **no rows on this board**
> (`grep -c WXA` = 0, verified). None of those deliverables was ever built.
> The rows above do not supersede them; the poc-v2 ladder should be landed as
> its own block, marked NEVER BUILT, so the gap is visible rather than inferred.

> **C5 note:** the global grid does **not** unblock `GEO-GOLDEN-HI`. The golden
> index floor needs N ≥ F(17)² = 2,550,409; the grid has 1,038,240. Short by
> ~2.5×, not by three orders of magnitude — still not constructible.

## SUBSTRATE_FORMULA_MATRIX — the arc's rated inventory (2026-08-12)

Document: `probes/weather-p1/SUBSTRATE_FORMULA_MATRIX.md`. Not a plan and not
a deliverable ladder — the **consolidated rating** of every formula,
encoding, sampling geometry, physical model and statistical instrument the
weather arc (#920–#946) actually put under a pre-registered bar.

| D-id | Deliverable | Status | Feeds |
|---|---|---|---|
| D-MTX-1 | Re-extract every tested primitive from the COMMITTED artifacts (7 parallel readers + 1; 131 primitives, 2.35 M subagent tokens) | **DONE** | the matrix's provenance rule — built from artifacts, never from session memory |
| D-MTX-2 | Two-scale rating (fitness A/B/C/D/V × evidence `[G]`/`[H]`/`[S]`), 56 rows across physics / encodings / geometries / instruments | **DONE** | reading comfort zones off instead of arguing them |
| D-MTX-3 | Known-effect vs discovered-explanation pairing (14 pairs, K1–K14) | **DONE** | separates prior art from what this arc added — incl. 4 cases where measurement went AGAINST the prior |
| D-MTX-4 | Figure verification against source JSONs (28 headline figures + circular/CT set) | **DONE — 0 mismatches**, 1 rounding fixed (Rayleigh p 0.689 → 0.688) | the audit-terminates-at-an-artifact rule, applied to this document itself |
| D-MTX-5 | Refresh after the cross-swap matrix runs (C2–C6) | **DONE 2026-08-13** — §0 UPDATE + 5 new encoding rows (E15–E19) + 3 geometry rows (G16–G18) + lessons 10–13 + §5 gap closures | the off-diagonal cells are measured; §5's cross-swap and CAL-FISHERZ gaps are closed, the geometry gap is re-stated as structurally unreachable |
| D-MTX-6 | §7 NEXT STEPS — three tiers, with N1 (the classid mint) named as the single blocker and an explicit "explicitly NOT next" list | **DONE 2026-08-13** | the product-lead read: the codec is measured, the substrate is built and gated, one operator decision separates it from a running bake |

**Why a "C" tier exists.** The arc's founding result is that most substrate
formulas are neither good nor bad — they have homes. Fisher-z alone carries
**three** measured verdicts (tail read 8.3× win, level read 4.7× loss,
CI-frame not-a-win). A single ranking would have destroyed the finding.

**What the matrix is honest about:** the majority of its rows are negative.
Two entries are **V (VOID)** rather than D — the apparatus could not
distinguish anything — and §5 lists 13 gaps including *every* off-diagonal
cross-swap cell, all ten EV probes, and five preflight rows that are
permanently unreproducible because their coordinates were never recorded.

## substrate-comfort-zones-v1 — the comfort-zone map (PRE-REGISTERED 2026-08-12)

Plan: `.claude/plans/substrate-comfort-zones-v1.md`. A **cross-swap**
(donor × target) transfer matrix per geometry arm → where does each
substrate formula feel at home, and how badly does it travel. §1 preflight
already run and it corrected two regime definitions before any bar existed.

| D-id | Deliverable | Status | Feeds |
|---|---|---|---|
| D-CZ-0 | §1 regime preflight (`\|∇p\|` ladder, elevation-confound screen, speed-is-not-the-discriminator finding) | **DONE — and now REPRODUCED + partly corrected (§6.1/6.2).** It had NO committed script or JSON when marked DONE. 4 rows reproduce (1.004/1.022/0.994/0.931); the 5 EXCLUDED land candidates are **unreproducible** (centres never recorded). Definition identified: Pa per grid cell **without cos(lat)** — R3 ~40 % low; metric-corrected ladder 10.3/15.5/61.2/100.9, order survives, range 9.3× → **9.8×**. Original text:**  — ladder R1 Amazon 10.2 → R2 ocean 14.9 → R3 W Siberia 43.8 → R4 storm 95.6 (9.3× range); 4 land candidates excluded on elev σ > 150 m | the regime axis all other rows score on |
| D-CZ-1 | C0 controls (shuffled codebook + degenerate geometry), losability-smoke-tested BEFORE the full run | **DONE — PASS** (`substrate_comfort_d_cz_0_1.py/.json`). Both controls lose to both real arms on BOTH metrics in ALL FOUR regimes; mechanism visible (`GEO-DEGENERATE` saturates **72–97 %** per regime — corrected 2026-08-12 from a stale 92–97 %, see plan §6.3; R4 is the low end at 0.7224). C1b `separation` = **6.28** vs the ≥ 3 bar. **AND the run amended C4**: `ρ` is SATURATED on the diagonal (real-arm spread 3e-6…4.7e-5) so C4 could not have fired — `L` keeps `ρ` off-diagonal, C4 moves to RMSE in Pa (§6.4) | gates every cell — a control that can't lose voids its cell |
| D-CZ-2 | C1 regime-ladder stability across ≥3 timesteps | **DONE — PASS.** `\|∇p\|` order R1<R2<R3<R4 holds at all 3 tested timesteps (`substrate_comfort_d_cz_2_7.py/.json`, §7.1) | anti-cherry-pick on the whole regime axis |
| D-CZ-2b | **C1b constancy is measured** — `separation ≥ 3` | **DONE — PASS.** separation = **5.87–8.24** at all 3 timesteps | earns the phrase "held constant" |
| D-CZ-2c | **C1c the suitability ASSUMPTION** | **DONE — PASS, distinguishable=True.** Decay/Gini/tail-ratio R4-vs-R1 ratios 0.88/0.71/**0.385** — all deviate ≥ 20 % from 1. **Licenses the cross-swap interpretation.** A construction bug in C0 was found and disable-verified fixed en route (§7.0) — `GEO-DEGENERATE` was built from a shuffled array slice, not a spatial patch | was a null-and-void gate; passed, so downstream stands |
| D-CZ-3 | **C2 degenerate-row verification, both halves** | **DONE — PASS both halves.** Dynamic `L ≡ 0` exactly for every donor; `CAL-ABS` off-diagonal `L` ranges **0.011 (R1) → 0.947 (R4)**, demonstrably non-degenerate | the can-it-DIFFER gate; satisfied |
| D-CZ-4 | C3 **transfer loss** vs turbulence | **DONE — FAILS. ⚠ But the "reversal" reading is WITHDRAWN (§7.9).** `L̄` = 0.011 → 0.309 → 0.671 → 0.690 is **rank-correlated ρ = 1.000 with each regime's own value range**, and `L` tracks `saturation` at Pearson **+0.917** — it largely restates *wide boxes are hard to cover*, arithmetic not meteorology. What stands: the hypothesis is not supported | hypothesis unsupported; the causal reading is confounded |
| D-CZ-5 | **C4 the crossover** | **DONE — FAILS, no sign flip** (that part is solid). ⚠ The "+10.78 Pa growing margin" is **range-inflated** — RMSE in Pa is not comparable across regimes differing 18× in range; normalised as a ratio it reads 3.96 / 1.85 / 1.14 / **2.35**, NOT monotone, R1 extreme (§7.9). Weak form passes trivially as pre-registered | hypothesis **not supported**; margin trend withdrawn |
| D-CZ-6 | C5 geometry floor | **NOT RUN — structural blocker.** Box holds 4225 cells; the golden index floor needs N ≥ 2 550 409. `GEO-GOLDEN-HI` cannot be constructed at box scale on this grid (§7.5) | reported, not faked with an interpolated lattice |
| D-CZ-8 | **NEW — vorticity regime + range-matched transfer metric.** Every bar so far runs on a SCALAR pressure field; no wind, no rotation enters any metric, so "turbulence" was never operationalised dynamically (§7.9). Needs ζ = ∂v/∂x − ∂u/∂y (or Okubo–Weiss) as the discriminator AND a range-normalised `L` so coverage cannot masquerade as the finding | pre-condition for re-asking C3/C4 as a turbulence question |
| D-CZ-7 | C6 the transfer matrix (the deliverable) | **DONE.** Full donor×target table, all 5 arms × 4 regimes, every cell raw in `substrate_comfort_d_cz_2_7.json` → `C6_matrix` (§7.6) | comfort read off the diagonal, travel-cost off the off-diagonal |

> **2026-08-12 — rows re-cut, not restated.** The operator ruled that the
> plan's §2 was built as a horse race where a cross-swap diagnostic belongs
> (premise: the model *captures* the phenomenon but is *not calibrated*, so
> miscalibration is the condition of measurement, not an arm). §2 was
> rebuilt around the 4×4 donor×target transfer matrix and the bars were
> renumbered; two new bars (C1b, C1c) exist because "held constant" and
> "the manufactured spread is on the right axis" are now measured rather
> than assumed. **Every row above was still Queued when this happened — no
> measured result was reinterpreted.** Old→new: C2→C4 (crossover, now
> against the diagonal on `ρ` rather than against `ABS-OWN` on RMSE),
> C3→C3 (now transfer loss, not an RMSE ratio), C4→C5, C5→C6.

## golden-vs-tempered-stride-v1 — head-vs-gut queue — RUN 2026-08-12

Plan: `.claude/plans/golden-vs-tempered-stride-v1.md`. Standalone, zero fetch,
< 5 s wall time (turned out lighter than the ~5 min pre-registered estimate —
pure stdlib arithmetic, no numpy/scipy needed after all). All four bars ran
against `probes/weather-p1/golden_vs_tempered_probe.py`; results in the
matching `.json`. **Two real methodology defects caught by the run itself**
(T1's m* definition, T3's float round-trip false negative) — both fixed in
the committed script, both explained inline in the plan; neither weakens the
qualitative synthesis, both tightened specific numbers.

| D-id | Deliverable | Status | Result |
|---|---|---|---|
| D-GVT-T1 | Crossover sweep across 10 q, useful-range metric | **RUN (twice-corrected)** | VERIFIED-permanent m* at **1.9–2.7× q** (codex P1 on #935: the earlier 1.0–1.4× figure was a FIRST crossing, not permanent — golden's non-monotonic sequence dips back above the ceiling; now suffix-verified at 76–83 reported checkpoints per row). Also codex P2: useful-range floor `q//2`→`⌈q/2⌉`. |
| D-GVT-T2 | Asymptotic golden-advantage pass/fail bar | **RUN — PASS** | golden ahead 68.2–106.4× at m=200q, all 10 q |
| D-GVT-T3 | Closure-occupancy guarantee (tempered) vs variable (golden) | **RUN — PASS** | tempered 140/140 exact (integer-verified); golden 124–127/140 across 5 phases |
| D-GVT-T4 | Naive-rounding collapse hazard rate | **RUN — PASS** | 114/292 = 39.0 % of q∈[8,300) collapse under naive rounding |

## weather-w-probes-v1 — W-probe queue (PRE-REGISTERED 2026-08-12)

Plan: `.claude/plans/weather-w-probes-v1.md` (worker briefs; §0 preamble is
verbatim-paste for every Sonnet worker, incl. the stranded-rescue checkpoint
protocol). Product-lead frame: `probes/weather-p1/COMET_TAIL_REPORT.md` §10.
Wave 1 = parallel, no operator gate beyond go-ahead; gated rows named.

| D-id | Deliverable | Wave | Status | Feeds |
|---|---|---|---|---|
| D-W5 | Spiral-ADI anisotropy (v2, full-band control, V-matched iterations, bump 3.35σ from mask) | 1 | **B2 FAIL / B3 VOID CONFIRMED / B4 INCOMPLETE** — B2: real diffusion resolved, aniso 1.5251 vs 1.25 bar, clean baseline 1.0046, operator contributes ~0.52. B3: family A 99.68 % + family B 99.56 % (both link families, QUALIFYING population n=4.78M out of the headline lattice N=7.65M, not the 62k sub-sample) land on a pure Fibonacci offset — dominated by the two discovered strides 2584=F(18)/4181=F(19) respectively. B4: downgraded to a DESCRIPTIVE reading over n=8–17 only — n=19 was dropped without pre-authorization, bar not satisfied, stays open | domino.rs gather-design claim REFUTED at this test point (v1's "unblocked" was the same inert-operator artifact that also drove B2's false PASS); `E-ON-A-GOLDEN-LATTICE-LOCALITY-IS-FIBONACCI-MEMBERSHIP-1` strengthened, third dated update; B4's n=19 remains an open follow-up |
| D-W2sA | Golden-vs-grid pairing on real cos-lat metric (zero-ties G1, CV G2) | 1 | **RUN — G1 VOID, G2/G4 FAIL (control degenerate: two identical translated grids are symmetry-uniform, CV ~1e-12 — cannot lose any evenness comparison; diagnosed via smoke test, run as-specified, `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`)** | honest falsifier for evenness DEFERRED (offset/rotation-varied or spacing-mismatched control); §10.5 properties 1–3 untouched |
| D-W6 | Two-component deconvolution (geo + bow, global lstsq, 38 eqs / 2 params; B3 = stranded stratification via v_rel) | 1 | **RUN COMPLETE — B0 VOID** (single-geo R²=−0.104, worse than the mean; both anti-vacuity controls score ≥ the ceiling — the model has nothing to identify); B1/B2 correctly report VOID per rule; B3 stranded stratum EMPTY (n=0 — a structural consequence of CT-F14's displacement≥250km filter implying `\|v_storm\|≥11.57 m/s`, not a physics finding) | vector-sum model DISCONFIRMED as specified on this sample; CT-F17 gate now moot for this model form — a revised model needs its own W6-shaped test first; `E-THE-DISPLACEMENT-FILTER-ATE-THE-STRANDED-STRATUM-1` |
| D-W2sB | α-window sweep β∈[0.85,1.15] | gated (W2s-a) | Queued | corridor α discriminator |
| D-W7 | Corridor two-regime α field probe | gated (W6) | Queued | §10.3 physics |
| D-CT-F17 | FRESH-sample verdict (1959–1979, N=70 candidates, V-test p<0.05 ∧ R̄≥0.35; independent adversarial spec audit MANDATORY before bars commit) | gated (W6 + audit) | Queued | the directional claim's verdict path |

## 2026-08-11 — EV queue re-graded to v2 specs (post-audit; rows below unchanged in STATUS)

All eleven EV rows in the block below still read **Queued** — correctly, none has
run. But their **specifications are now v2**: the 13-agent pre-registration audit
(plan §8) found **0 of 11 v1 specs sound** and every one was rewritten. Cite the
plan's §3 v2 text, never the v1 shape summarized in the row descriptions.
**EV-9 is Wave 0 and unblocked** (no data needed).

## weather-substrate-evaluation-v1 — EV queue (PRE-REGISTERED 2026-08-11)

Plan: `.claude/plans/weather-substrate-evaluation-v1.md` (known-vs-test ledger;
verify/attack-audited before ACTIVE). Waves: 0 = no data · 1 = one fixture
re-fetch · 2 = scale.

| D-id | Deliverable | Wave | Status | Feeds |
|---|---|---|---|---|
| EV-1 | Advection-as-Morton-shift falsifier (wind (dx,dy) tile shift vs persistence, calm-tile silence half) | 1 | Queued | §12.16 [S] regrade |
| EV-2 | Wind-lane encode at FIELD level (nearest-n vs u8-palette-circular vs u16-linear; wrap-corruption can-fire + 90°-sector silence) | 1 | Queued | D-3 |
| EV-3 | Floor-sensitivity sweep [0.1..2.0] K → per-variable flip-points | 1 | Queued | D-1 |
| EV-4 | Saturation-window sweep → sat% vs interior-CI tradeoff curve | 1 | Queued | D-2 |
| EV-5 | U-shaped variables (total_cloud_cover, sea_ice_cover) — the shape rule's OTHER half, two-sided by construction | 1 | Queued | E-TRANSFORM-SHAPE promotion/refutation |
| EV-6 | P1/P2 harness re-expressed on the SHIPPED RollingFloor frame; equivalence gate 0.848/0.820/95.65 + 0.9997/0.999556 to 1e-4 rel | 1 | Queued | D-6 |
| EV-7a | 16k×16k 3DGS top-k scale run (1 048 576 tiles; heel-reject ∉ {0%,100%}) | 2 | Queued | operator-named capability |
| EV-7b | Comma anti-moiré falsifier at tile scale (regular stride MUST alias; comma must not) | 2 | Queued | D-QUANTGATE evidence |
| EV-8 | Jirak effective-n for the P2 correlations (spatial autocorrelation → n_eff) | 1 | Queued | P5 |
| EV-9 | Commit the K-12/K-13 orphan measurements as disable-verified helix tests | 0 | Queued | honesty-split closure |
| EV-10 | Second timestep + season for P1 ([H]→promotion or timestep-conditional regrade) | 1 | Queued | K-9 robustness |

## oracle-funnel-probe v1 — PROBE-ORACLE-FUNNEL staged (PRE-REGISTERED 2026-08-05)

Plan: `.claude/plans/oracle-funnel-probe-v1.md`. Consumes OGAR #241/#244
(`validate` W-1, `FnSpec.name` W-2, `FunnelTally` W-4); wishlist delivery
recorded in OGAR handover `2026-08-05-1430-…-to-ogar-loco.md`.

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| PROBE-ORACLE-FUNNEL-S0 | 3-arm deterministic funnel probe (R floor / G mid / W ceiling), E1-E4 pre-registered, legend-cost anchor | OGAR (`ogar-loco` example) + plan here | MEASURED — E1-E4 all met (0.5%/2.5%/100%, spread 99.5 pts) | plan §5; OGAR PR #245; `E-THE-LEGEND-IS-NOT-THE-GRAMMAR-1` |
| PROBE-ORACLE-FUNNEL-S1 | LLM arm via rig CompletionModel (D-RLG-1); legend token cost under prompt caching; validity feedback only | lance-graph | GATED (operator word + API) | plan §4 |
| PROBE-GADAMER-BAG | NARS-34 vocabulary survivors scored by Gadamer-projection falsifiers | both | GATED (W-3 mint: operator byte assignments + ResultBehavior) | plan §4 |

## 2026-08-04 — Arm BLW retractions (prepended; the D-BLW rows below are restored to their original text, Status field only updated)

Four errors on one axis, operator-ruled. Recorded here rather than by rewriting
the rows in place (append-only governance; the earlier in-place rewrite was
caught on review and reverted).

- **D-BLW-1 — shape void.** An owner is a **TENANT**, not a shard
  (`CLAUDE.md`: "one mailbox = one kanban board as tenant"; one `MailboxSoA` is
  moved into one `KanbanActor` as sole mutator). Tiling the Bible across 64
  owners fabricated 63 tenants. The shipped test was **deleted** — it was green
  on that fabricated shape. Correct shape: ONE tenant, verses as ROWS, thoughts
  row-level over the owner's slice. `E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`.
- **D-BLW-4 — axis void.** Owner-count is not a scale knob, and the follow-up
  "measure it with 4,096 *lightweight* owners" kept the wrong unit and merely
  made it cheap. Harness deleted. Correct axis: N row-level thought bodies
  within one owner, A2/W2 protocol carried verbatim. Plan §12.3a′.
- **The memory case was measured off the wrong struct.** Canon is
  `NODE_ROW_STRIDE = 512` (const-asserted), so the whole 64k bake is **32 MiB**,
  not 384 MiB — the 6,144 B/row figure is `MailboxSoA`'s hot planes, 12× the
  canonical row. No tiling, no `#[ignore]`, no CI split was ever needed.
  Open question logged as `ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON`.
- **D-BLW-2 / D-BLW-3 designs stand** (§12.3a, §12.3b) and are unaffected by the
  above; their builds are not done.

## kanban-64k-inverted-awareness v1 — parallel thinking + inverted-awareness witness (PLANNED 2026-08-02)

Plan: `.claude/plans/kanban-64k-inverted-awareness-v1.md` (operator anchors a/b; R1-R15 review basis).

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| PROBE-IGNITION | The write path DRIVEN: arm by MetaWord write -> discover by board scan -> cast write-on-behalf -> seal -> apply. 64 real MailboxSoA owners, real KJV corpus, 6 cycles, 7 cohorts, 11 gates both halves | lance-graph | **GREEN 2026-08-05** — 2/2 tests; c1 = 24 casts (20 Flow + 4 Block), c5/c6 rest with zero casts and no seal; G9/G10 pin the two OPEN #879 caveats | `tests/probe_ignition.rs`; AGENT_LOG 2026-08-05 |
| MEASURE-64K-AXES | Operator-specified five-axis benchmark: B0 dummy baseline, B1a/B1b ownership+representation split, W0/W1 WAL segment curve (one fdatasync/one version per cycle), T0-T2 temporal phases over 1,048,576 rows, L1a/L1b chunked-layout control, EXP-KIA-A2-64K exploratory concurrency (non-claiming, A2 untouched). One release binary `measure_wal_curve`; four answers, axes never blended | lance-graph | Stage A0 MEASURED (3 of 4 answers; WAL knee NOT REPRODUCIBLE and unclaimed). v3's M-arm and O-arm both **MEASURED 2026-08-05, both NEGATIVE** (pre-registered two-sided, so both are findings): M-arm — digests MATCHED (`68128e3662df105c`), reorder 9.4 ms, downstream −25.8 ms ⇒ **Δtotal +35.2 ms, Morton LOSES**; ordered-chunk fast path 350.9 ms was slower than the generic 339.7 ms. O-arm — **DIVERGED** (`64565f362db2e4a5` ≠ `3e71c2aa7be8e325`) ⇒ the seal's ordering is load-bearing FOR THIS O-B CONSTRUCTION; follow-on question queued as PROBE-SEAL-VS-TEMPORAL. Open measurement defect: `ISS-MARM-T1-4X-A0-GAP` (M-arm T1 320-340 ms vs A0's 78-86 ms — blocks that one cross-run comparison only). A-arm (allocator-vs-architecture decomposition) deferred. v2 rolling-epoch-closure model supersedes v1's execution model (v1 = Stage A0 baseline + instrumentation, lane in build); then A1/A2 rolling+Morton lane; crypto REMOVED from the seal benchmark per operator sanity-check (seal path verified crypto-free in source; encryption = separate later layer, AEADs dep no longer blocking); EXP-KIA-A2-ROLLING-CLOSURE recorded (A2 frozen) | plans measure-64k-axes-v1 + v2 |
| D-IGN-B | Ignition starts the REAL lenses: arming z ∈ {0 unarmed, 1-4 = the four stances, 5 = Fusion (Strict/Aware gap read)} — six ordinals in MetaWord's 6-bit field, no 36-style bridge (Q1 sidestepped); thought bodies = the shipped nars stance machinery via cycle_driver's pluggable seam (D-BLW-1 precedent) + blw_fusion's two-projection read. Can-fire: different lenses over byte-identical rows ⇒ non-identical readouts; silent twin: same lens ⇒ bit-identical; unarmed ⇒ none | lance-graph | **GREEN 2026-08-05** — 1/1 test, L0-L7 + z5-BLOCKED; L1 Kant≠Wittgenstein over byte-identical rows, same-lens bit-identical; Hegel/Nietzsche NON-empty on the text path; z=5 reserved with the printed blocker | plan cycle-driver 12.11; `tests/d_ign_b_lenses.rs` |
| D-BLW-5 | Observer-effect loop: a jc statistic about the cohort fed back into awareness; four pre-registered arms (true/false±/placebo) + the §12.8 bloom criterion as the frozen null instrument. KILL: placebo movement invalidates; T-silence is a reportable null. Payload refined §12.9a: distribution shape × Prozentrang (never the raw scalar); single-measurement law + remeasure guard; doctrine doc `observer-effect-tfpn-doctrine.md` | lance-graph | **PAUSED by operator 2026-08-05** — the Opus design lane was stopped mid-run (controlling signal; not relaunched). Banked and committed: the Sonnet API inventory (`exec-runs/d-blw-5-api-inventory-sonnet.md` — BeliefArena admits hand-built statements; jc+run_cycle live in disjoint crates, supervisor+jc dev-dep pre-ruled acceptable; ndarray unreachable supervisor-side). Gate to resume: operator direction | plan cycle-driver 12.9 |
| PROBE-ARC-TORQUE | Torque of an arc = 2× Heron triangle area from 3 HHTL O(1) distances (magnitude metric-only; chirality via helix_orient Fisher-2z frame codes); Fisher 2z = logit((1+r)/2) as the additive equal-information embedding, hydratable via tanh. Stage B: translator stray/mindset vs source (floor = intra-language variance; Romans 5:12 in-quo/eph-hō as known-answer falsifier). Stage C: author-bias fields on the redactional layer + attribution of non-canonical books (G1-G5 in-canon ground-truth gates first). KILL per stage: F1 radial/tangential non-separation; clamp-rate ceiling; G1-G5 failures | lance-graph | Queued — proposed §12.10, behind PROBE-IGNITION + D-BLW-5 | plan cycle-driver 12.10 |
| D-ACK-CLEANUP | Delete the ack/pump/tick theater entirely + add the visibility surface (operator-directed, context-hot). `kanban_actor.rs` → message-free module: `PhaseCensus` (`&self` fleet census; absorbing DAG-derived; empty ≠ at-rest) + pure `mul_target`/`parse_kanban_step`; `KanbanMsg`/`KanbanActor`/5 RPC drivers/`run_to_absorbing`/`KanbanRouteError` DELETED. Lane E migrated to direct `&mut` owner (supervisor+ractor out of its feature); W2b probe rewritten direct + census-over-real-SoA. Zombie verdict: half yes (lib.rs re-exports + lane E kept it alive; `ack_and_propose` already absent — the ack half lived only in docs). OGAR boundary verified: zero consumers; its ActionHandler ack surface is legitimate membrane protocol. Kanbanstep (`VersionScheduler::on_version`) NOT theater — stays canonical; naming question flagged only | lance-graph | **SHIPPED 2026-08-05** — all gates green (supervisor 9 lib + w2b 3/3 + cycle-driver 4/4; onebrc lane-e 20/20; clippy --no-deps clean; fmt clean) | `E-ACK-THEATER-DELETED-1`; TD-MESSAGE-RESIDUE resolution; `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` |
| D-HWV-1 / EXP-HOT-WINDOW | The hot version window (operator-directed): publication clock decoupled from persistence clock — every sealed cycle publishes to RAM immediately (`published_head`), durability is a batched background **sync barrier** (`durable_head`, barrier-flush fork: K unsynced Lance commits + ONE fdatasync, so 1 cycle = 1 real DatasetVersion survives everywhere). Vertical batching (time, not owners); Nagle-shaped flush policy (bytes/16-dirty/200ms/pressure/shutdown/durable-only-reader). Five panel-bought invariants H-1..H-5 (checkpoint fencing, torn-tail cleanup, no-veto-after-publish, zero-copy conditions, rung-decided visibility — H-5's original "ack rebase" clause RETRACTED per `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`: no pump/ack/scheduler; seal → publish → immediately queryable → durability trails; the window is a resident horizon of immutable versions, not a message queue). Version-multiplexing fork REJECTED (temporal.rs has no cycle-within-version coordinate ⇒ intra-version hindsight). P1-P5 pre-registered with named KILLs | lance-graph | **Design banked 2026-08-05, panel-hardened (1 sweep + 1 adversarial refuter; fork inverted by evidence), NOT built** — build lane gated on operator word; P1/P2 comparisons additionally gated on ISS-MARM-T1-4X-A0-GAP + TD-LANCE9 remeasure | plan measure-64k-axes-v4; `E-HOT-WINDOW-DECOUPLES-THE-CLOCKS-1`; v2 cross-note; seal-vs-temporal caveats |
| PROBE-SEAL-VS-TEMPORAL (3 probes) | The question the O-arm divergence opened: **what does the seal compute that `temporal.rs` does not encode?** Answered from shipped source — cross-owner TOTAL order (temporal's `cast_seq` forbids cross-owner comparison ⇒ partial order) · arrival as an ordering input, durably recorded nowhere else · the per-row coalescing FOLD (`temporal.rs` has no row concept) · cohort + read horizon `CycleFrame{cycle, base_version}`. Three pre-registered probes: **SEAL-TIE-DENSITY** (do cross-owner `stream_position` ties occur? ties ⇒ order partly derives from non-durable arrival), **FOLD-COLLISION-RATE** (do two owners write one row? zero ⇒ the fold is structural-but-unexercised), **ARRIVAL-ASCENDING-CONTROL** (the can-stay-silent twin — digests must MATCH when arrival is owner-ascending; divergence there ⇒ the four-item account is incomplete) | lance-graph | Queued — doctrine + probe queue landed, none run. Standing position: temporal.rs = authoritative TEMPORAL model, seal = authoritative ORDERING model, the gap = an explicit research question | `.claude/knowledge/seal-vs-temporal-ordering-information.md`; `E-SEAL-AND-TEMPORAL-ARE-DIFFERENT-OBJECTS-1`; plan measure-64k-axes-v3 |
| D-KIA-0 | jc capability map + dichotomous-statistics decision note (phi/KR-20/kappa naming; Spearman dropped at view 2) | lance-graph | Queued | plan W0 |
| D-KIA-A1 | ⊘ RESCOPED 2026-08-04 (E-ACTOR-IS-NOT-THE-PHASE-PATH-1): #879 is the complete phase-progression path; KanbanActor has no assigned architectural responsibility (legacy compatibility code). SHIPPED: held-owner reschedule/wake. OPEN: run_cycle drained-writer retry guard; missing-owner counter in cognitive_pass | lance-graph | Queued | plan W1 |
| D-KIA-C1b | jc additive-only extension: kappa + McDonald's omega + r-family effect size (R/R-squared, eta-squared = explained variance) + t-test (t/df/p) + a named phi wrapper. Cohen's d explicitly OUT — calculated separately if ever wanted. HARD CONSTRAINT: additive only — pearson/spearman/cronbach_alpha/icc keep their arithmetic, signature and semantics; any diff changing an existing jc statistic is an automatic reject. ONE sanctioned edit: widening reliability.rs private helpers (mean/all_finite/average_ranks/pop_var) to pub(crate) for reuse, visibility only, no body change. C1 audit found phi = pearson-on-binaries (already present in substance) and KR-20 = alpha-on-dichotomous (naming only); kappa absent = the real gap. SHIPPED as crates/jc/src/stats.rs: cohen_kappa, omega_total, phi, multiple_r/multiple_r_squared, eta_squared, t_test_one_sample/paired/welch/student, anova_one_way; 31 new tests (107 lib + 11 doctests green), clippy-clean. Existing-file diff is visibility-only (mean/all_finite -> pub(crate); average_ranks/pop_var NOT widened, unused). Unblocks D3a (overlap MEASUREMENT) — NOT a fusion claim: kappa is chance-corrected agreement under the observed marginals and says nothing about incremental value, so fusion still needs D3b's external criterion per the plan's own C3. Corrective slice (external review): omega sign-erasure + R-squared scale-dependence fixed; BinaryAssociation/kr20 added | lance-graph | Shipped (#887) + corrective slice | plan W0/C1b |
| D-KIA-A2 | parallelism falsifier (protocol pre-registered: median-of-5, >=2x at >=4k owners, +/-10% stay-silent; kill = regrade claim (a)) | lance-graph | Queued | plan W2 |
| D-KIA-B1 | catalog binary-range criterion contract type + generalized catalog-mirror drift guard | lance-graph | Queued | plan W3 |
| D-KIA-C5 | cohort-statistic witness type under the ELEVATED carve-out + held-out anti-circularity gate | lance-graph | Queued | plan W4 |
| D-KIA-D1 | observer/observed as two Locus categories over one arena (cheapest-first) | lance-graph | Queued | plan W5 |
| D-KIA-C2 | Name the dichotomous statistics correctly (Pearson->phi, alpha->KR-20, kappa NOT a renamed ICC, Spearman dropped on binaries). AUDIT RESULT 2026-08-04: the jc reliability battery has exactly 4 consumers (style_table_agreement, rung_divergence_reliability, partof_isa_vs_palette256, l9_loci_real_text) and NONE is dichotomous — style columns, rung levels 1-10, palette/taxonomy distances, i4 loci offsets are all continuous/ordinal, so Pearson/alpha/ICC are correctly named at every existing call site and there is ZERO rename work today. The discipline binds PROSPECTIVELY at the first binary-criteria witness (D3). Surfaced instead: TD-STATS-DEGENERACY-CONTRACT-DIVERGENCE | lance-graph | Audited (no rename work; binds at D3) | plan W0/C2 |
| D-BLW-1 | One 64k KJV SoA + the four-stance lens body wired into cycle_driver's 5.4 pluggable thought seam; Outcome round-trips via emit_bootstrap_intent. Reuses P4a/P4b/P4c falsifiers at KJV scale | lance-graph | Retracted (shape void) — rebuild queued | plan cycle-driver 12 |
| D-BLW-2 | The four stances (Hegel/Nietzsche/Kant/Wittgenstein) as READS over the sealed version, not four bakes. Discrimination twin: pairwise binary_association must show lenses can differ AND can agree; report counts + both marginals, never bare kappa | lance-graph | MEASURED KILL 2026-08-04 (plan §12.7) — instrument writes 3 of 24 loci, 1 shared, so agreement_count is capped at 1 before any verse is read; rebuild queued | plan cycle-driver 12 |
| D-BLW-3 | Horizontverschmelzung as a measured trajectory across the sealed series, under a-priori (single-version filter) and hindsight (version-range cascade) reads. KILL: flat kappa regrades the claim to four independent stance reads — not Gadamer | lance-graph | **SHIPPED + MEASURED 2026-08-04** (`examples/blw_fusion.rs`, re-scoped per design B1 to two rank projections over the tenant): band IN/IN (κ 0.49/0.46); Δκ at V_pin −0.031 = middle ground, no fusion verdict; the 8-horizon table shows the a-priori/hindsight gap CLOSING monotonically (Δκ −0.485→0, Hamming A 152→0) — DROP does not fire; first `DeinterlaceRow` implementor + `deinterlace` caller | plan cycle-driver 12 + §12.8 result |
| D-BLW-4 | 64k concurrent thought bodies at KJV scale. Inherits W2's pre-registered thresholds (median of 5+ runs, 2x at 4096+ owners, 100us bodies). KILL: regrades to 64k-scale SEQUENTIAL sparse cycles | lance-graph | Retracted (axis void) — rescope queued | plan cycle-driver 12 |
| D-KIA-D3a | DESCRIPTIVE binary overlap: contingency counts + BOTH marginals + observed/expected agreement + kappa + phi, via jc::stats::binary_association. Claim ceiling is overlap / disagreement / marginal asymmetry / redundancy-or-complementarity CANDIDATE. No fusion or validity claim | lance-graph | Queued (unblocked) | plan W6 |
| D-KIA-D3b | HELD-OUT fusion falsifier — BLOCKED until an external criterion and a criterion-appropriate scoring rule are chosen. Continuous criterion: pre-registered delta-R-squared = R2(A+B) - max(R2(A),R2(B)). Binary criterion: a proper held-out score, NOT R-squared forced onto it. Reliability is not validity (plan C3) | lance-graph | Blocked (needs external criterion) | plan W6 |

## PROBE-BABEL-STANCES — two Rosetta stones + four-channel phase split (IN PR — slice 2, 2026-07-28)

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| PROBE-BABEL-STANCES | Slice 2 rebuild after codex P1 retraction of slice-1's headline (synset-vs-lemma coordinate confusion + en-kjv self-resultant pollution). Two Rosetta stones: the CORPUS stone (three-axis synset×POS×frequency-band convergence grid) + per-lane LANGUAGE stones (private `BeliefArena`, learns within-language `Sim` beliefs while traversing the corpus stone). Four-channel phase attribution (Morphologie/Syntax/Semantik/Pragmatik) replaces pooled divergence: German KNOW diverges on all three live channels, Latin KNOW silent on all three, Czech DIE morphology-only (CHECK row, report-only). Translationese finding: the pragmatic channel is coherent antiphase across every **verified** lane (en-kjv, de-luther1545, la-vulgate, el-lxx) — inherited calque, not independent convergence — with the Czech and Aramaic rows reading the same way but **reported, not claimed** (CHECK rows never gate CI). Tracking census inverts the slice-1 headline: only German lexically tracks the grid's two KNOW coordinates. Valency: English marks the distinction by subcategorization frame, not lexeme. Passion (quale × magnitude × phase) peaks at (German, KNOW) = 0.3130. Two self-caught defects fixed in-flight (MAD=0 degenerate-population branch; quale/confidence conflation on CHECK rows). Honest limitation: the `prefix\|stem` morph carving is typologically blind to agglutinative suffixing (Finnish); spine slot 6 left RESERVED rather than reconstructed. | lance-graph | **IN PR** — `crates/lance-graph-planner/examples/probe_babel_stances.rs` green in CI (`.github/workflows/rust-test.yml` runs both probe examples explicitly, closing the slice-1 P2 finding) | `E-TWO-ROSETTA-STONES-AND-THE-FOUR-CHANNEL-SPLIT-1`; slice-1 entry `E-THE-GRID-COLLAPSES-WHAT-A-LANGUAGE-SPLITS-1` Status amended to RETRACTED; `crates/lance-graph-planner/examples/probe_babel_stances.rs`; task #52 |

## PROBE-EYES-OPENED — the Adam awareness event printed from the KJV bake (SHIPPED 2026-07-28)

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| B6-ASPECT-PANEL | Four philosopher stances (Hegel/Nietzsche/Kant/Wittgenstein) as pure reads over the unchanged arena — crystal/Doppelspalt operational; Kant ablation flips the corpus crown (a priori constitutive); Wittgenstein crowns naked (3 games); invariance core stance-independent | lance-graph | **SHIPPED** — fixture asserted, corpus report | `E-FOUR-LIGHTS-ONE-CRYSTAL-1`; task #50 |


| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| PROBE-EYES-OPENED | Gen 3:7 awareness as printed epistemic structure: B1 reversal blades (blind contradiction ranking → {eat, die} @0.850 + 2 discovered: god→good, they→respect), B2 reflexive rung lift (10 lifts on real Gen 1-4, exactly ONE self-referential = 3:7 knew→naked), B3 causal chain (Impl(naked→afraid) from the text's "because"), B4 Hermeneutik (pass-2 stamp-overlap → CHOICE only → fixed point: NARS supplies the hermeneutic circle's termination proof). Contract: `clause_cues::{is_negation, is_perception_verb}`. | lance-graph | **SHIPPED + MEASURED** — fixture asserts green in CI; real-corpus blind run recorded | `E-EYES-OPENED-PRINTS-BLIND-1`; `examples/probe_eyes_opened.rs`; task #48 |

## rosetta-codebook-convergence-v1 — Bible Rosetta SoA + qualia agreement (ACTIVE)

Plan: `.claude/plans/rosetta-codebook-convergence-v1.md` (operator convergence arc, 3 same-day corrections baked). Absorbs W4/W15/W17/W18.

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| D-RCC-1 | lanes-to-singleton probe (calibrator) | lance-graph | **v1 RUN** (`build_rosetta_probe.py`; 4 PD lanes kjv/luther1545/elberfelder1905/bkr; census 31,103 union / 31,097 common; swallow+grape receipts incl. LIVE Ps-84 versification-offset + two-lane rescue; en→de split census 48.9% of 3,071 mid-freq words; `tongue→Zunge/Sprache` real sense split) | `E-RCC-1-FOUR-LANES-ONE-KEY-1`; report in local `out/` |
| D-RCC-2 | Rosetta SoA shape (contract) | lance-graph | Queued | plan §2 |
| D-RCC-3 | corpus-derived word alignment | lance-graph | **In flight** (`build_alignment.py`; PMI v0 proven inside D-RCC-1; also the successor to the failed monolingual closed-class detector — labels TRANSFER through alignment) | plan §2 |
| D-RCC-4 | qualia-agreement vector (POS-routed) | lance-graph | Queued — un-gates W11 | plan §2 |
| D-RCC-5 | CLAM/WordNet probe + CHAODA lane-anomaly read | lance-graph | Queued (taxonomic arm runnable now) | plan §2 |
| D-RCC-6 | cross-lane constraint propagation to fixpoint | lance-graph | Queued | plan §2 |
| D-RCC-7 | Czech lane | lance-graph | **Data landed** (bkr lane fetched + in census) | plan §2 |
| D-RCC-9 | **Greek SOURCE lane (new)** | lance-graph | **Shipped** — Tischendorf 8th ed., stated `Public Domain`; TR + WH are both CC BY-NC-SA (the age-implies-PD assumption is false). 27 books / 7,895 verses; 62 KJV verses `TextAbsent`. Unblocks *source outranks translation*. | `E-PD-GREEK-LANE-ACQUIRED-TISCHENDORF-1`; `fetch_greek_lane.py` |
| D-RCC-8 | Rosetta package Release | lance-graph | Licence blocker §4.4 **DISCHARGED** (PD Greek acquired); remaining: per-treebank re-verify, Kralická edition provenance, versification map source | plan §4 |

## dialectic-engine-v1 — the reasoning cathedral (ACTIVE)

Plan: `.claude/plans/dialectic-engine-v1.md` (six operator pillars + S1-S12 synthesis). V0-V1 SHIPPED; V2-V5 queued.

| D-id | Deliverable | Repo | Status | Evidence |
|---|---|---|---|---|
| D-DIA-V0 | Belief arena falsifying slice (triple-keyed + in-place stamped revision + copula-gated transitivity) | lance-graph | Shipped (all 4 registered gates green first run) | `deepnsm-v2/src/belief.rs`; 90 tests + clippy clean; `E-DIALECTIC-ENGINE-SYNTHESIS-1` |
| D-DIA-V1 | The five tactics (RCR/TR/ASC/CAS/CR) over the Belief arena + throttles + ReasoningGap | lance-graph | Shipped (PR #816; lance-graph-planner `nars/{belief,tactics}` over `TruthValue`; 17 nars + 233 planner tests) | plan §3, §4 |
| D-DIA-V2-A | insight/mush S10 detector + size-preserving null falsifier | lance-graph | Shipped (PR #819 merged; `nars/insight.rs`; null falsifier caught + fixed 2 formula confounds — `E-S10-COHERENCE-CLOSURE-DENSITY-1`) | plan §1 S10, §4 |
| D-DIA-V2-B | The loop: bias→recipe tactic-LUT, byte-lane council, texture window | lance-graph | **bias→tactic LUT SHIPPED** (`E-DIA-V2-B-BIAS-TACTIC-LUT-1`; `nars/tactic_select.rs` `tactic_for_bias`/`TacticChoice` reusing `contract::sensorium::GraphBias`; confusion-matrix falsifier `examples/tactic_select_confusion.rs` — G1 5/5 diagonal, G2 3/3 structural discrimination, G3 beats constant policy; honest split: RCR/TR/CAS structural, ASC/CR normative; 4 unit + 28 nars tests, clippy clean). Byte-lane i8 council + texture window still queued. | plan §1 S8, §4 |
| D-DIA-V3 | Dissolution → field rung-elevation (the cathedral floors) + Staunen↔Wisdom flow accounting | lance-graph | **V3-A + V3-B + V3-C SHIPPED (the S11 loop is CLOSED).** V3-A detector (`E-DIA-V3-A-DISSOLUTION-DETECTOR-1`; `nars/dissolution.rs`, S10 mirror on `insight::Snapshot`; null PASS). V3-B response (`E-DIA-V3-B-FIELD-ELEVATION-1`; `nars/elevation.rs` `elevate_field` — mass-induction mints abstract parents over shared-predicate clusters, HHTL grows upward; falsifier proves the PAYOFF = one parent-fact propagates to all k children via closure, and the honest guard = nothing minted on structureless noise; 3 tests). V3-C `regulate_cycle` (`nars/regulate.rs`) composes detect_dissolution→should_elevate→elevate_field into one active-inference cycle (`E-DIA-V3-C-REGULATION-LOOP-1`; elevation TRIGGERED by measurement not chosen; 3 tests — dissolving→elevates, crystallizing→no-op, loop bounded across cycles). **S9 epiphany attractors SHIPPED** (`E-DIA-S9-EPIPHANY-ATTRACTOR-RATE-1`; `nars/epiphany.rs` `rank_epiphany_attractors` — rank basins by density not count, E-DOOMSCROLL 3rd confirmation; 3 tests, rate-vs-count divergence falsifier). V3 fully shipped. | plan §0 pillars 3-5, §1 S11, §4 |
| D-DIA-V4 | Foveated HHTL-trie field-search (addressing-first ladder; field search = total-function floor). Rung 1 = `PROBE-CODEBOOK-44` (16-way hierarchical codebook), rung 2 = foveated morton-comma descent, rung 3 = `PremultipliedOver` blasgraph `mxv` floor. Kuzu factorized-processing alignment. | lance-graph + bgz17 | Rung 1 MECHANISM-GREEN + REAL-DATA ρ RUN (`E-PROBE-CODEBOOK-44-MECHANISM-1`: `bgz17::build_hierarchical`, prefix==ancestry purity 1.0 vs flat 0.16; real-data ρ on jina-v3 = hierarchy fidelity-neutral (structure-is-free confirmed), but anchor-close blocked by the Base17 17-dim fold ceiling ρ=0.26 — `TD-BASE17-FOLD-CEILING-SINGLE-WORD`, NOT the codebook; M1 not fully closed). **Rungs 2-3 + M26 SHIPPED** (`E-DIA-V4-FIELD-SEARCH-LOOP-1`, 4-agent Sonnet fleet, Opus-gated): rung 2 `foveated_descend` (8× prune + full recall at fovea_k=2), rung 3 `premultiplied_over` commutative composite (bgz17 palette, not blasgraph), M26 `Belief⟷SpoFacet` lossless byte round-trip. #4 CV-sweep refined `TD-BASE17-FOLD-CEILING`. Architecture `E-FOVEATED-HHTL-TRIE-FIELD-SEARCH-1` | plan §4, S1 |
| D-DIA-V5 | Reach-out felt integration (dull shadow vs insight) + qualia ablation falsifier | lance-graph | **V5-A SHIPPED** (`E-DIA-V5-A-FELT-INTEGRATION-1`; `nars/reach_out.rs` `reach_out_integrate`/`FeltOutcome` — the §3.6 middle-term-click criterion as a structural test: fetched bridge quarantined at prior 0.1, NewInsight iff ≥1 derivation composed vs DullShadow; 4 tests incl. size-matched + quarantine-cap). Remaining V5: the S12 qualia-ablation FIELD-scale falsifier. | plan §3.6, S12 |

## scientific-kg-substrate-v1 — crawl → OCR → terms → reason → MUL (scoping)

Plan: `.claude/plans/scientific-kg-substrate-v1.md`. PROPOSED scoping; outward-facing crawl (D-SCI-3) BLOCKED on §4 decisions. D-SCI-1 buildable on a further "Go".

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-SCI-INSIGHT | Main-insight surface — "the paper speaks and thinks for itself" (no-LLM, emergent, auditable) | lance-graph | **SHIPPED — two surfaces.** (1) `basin_resonance::rank_basins` (`E-SCI-INSIGHT-BASIN-RESONANCE-CLICK-1`, the operator reframe) — the honest single measure: `resonance = staunen × wisdom` per basin, coherence-vs-evidence kinds from one measure; 5 tests incl. E-BASIN-WIDTH null + E-DOOMSCROLL rate-not-count. (2) `insights::extract_main_insights` (`E-SCI-INSIGHT-PAPER-SPEAKS-FOR-ITSELF-1`, the typed catalog) — CoreTheme(S9) + Conclusion(ladder) + Bridge; #832's two Codex-P2s (empty-premise ladder, unbounded bridge strength) corrected in follow-up. Real-paper leg = D-SCI-1. Validation = the Kant/Schopenhauer/Hegel/Precht connective-tissue oracle. | plan; the insight-surfacing half |
| D-SCI-1 | Term/entity extraction — the gate (inverse of the colorblind finding); feeds D-SCI-INSIGHT. Corpus (operator): public-domain Gutenberg text OR ephemeral single-arXiv-paper via tesseract/spider (never committed). | lance-graph | **In progress — RELATION-EXTRACTION + ARCHETYPE-CONSUMER + FSM-FEEDER slices shipped.** (1) `examples/insight_relation_read.rs` (`E-SCI-1-RELATION-EXTRACTION-FINDS-THE-CENTRE-1`, #841): sparse typed `Inh` skeleton instead of dense ±window, closing the #836/#837 articulation loop. (2) `grammar/verb_lexicon.rs` + `examples/insight_archetype_read.rs` (`E-SCI-1-VERB-TABLE-ARCHETYPE-CONSUMER-AND-FSM-FEEDER-1`): extraction now READS the 144-cell `verb_table` archetype (verb→family→TEKAMOLO slot; causal→Kausal, grounding→Lokal, discriminative falsifier asserts) — the operator's "consume verb_table, not hand-roll" requirement. (3) `deepnsm-v2/fsm.rs` `Pos::Rel`: the movement feeder-tag (single-level relative clause preserves the matrix subject), feeding the ±8 antecedent pointer. (4) `examples/insight_spo_tekamolo_read.rs` (`E-SCI-1-SPO-TEKAMOLO-QUALIA-EXTRACTION-1`): the FULL-RECORD extractor — one clause → S·P·O (predicate typed via `read_verb`, consuming slice 2's archetype) + Temporal·Kausal·Modal·Lokal packed into a real `TekamoloFacet` (#839, byte-for-byte round-trip) + the canonical 17D qualia vector into value tenant #1 via `from_f32_17d` + the Familienaufstellung gestalt read over the constellation's mean felt vector; incl. the `verb_lexicon` `-ied→-y` core-gap fix (carried/carries). Honest next rung (still Queued): term/entity NP extraction + real verb-argument structure + corpus-tune the starter priors + cue lexicons. (5) `examples/insight_coca_read.rs` + `examples/data/coca/` (`E-SCI-1-COCA-GROUNDED-EXTRACTION-1`): the corpus-grounded extractor — PoS/lemma/transitivity/NP-compound/Lokal all from REAL COCA data (master lexicon + ngrams.info samples), verb family still from the #842 archetype; incl. two Codex #843 P2 fixes (copula-as-content, -ly overmatch). | plan; 5 slices shipped |
| D-SCI-2 | OCR ingest via the tesseract-rs doc.v1 seam | lance-graph + tesseract-rs | Queued | plan |
| D-SCI-3 | The crawl (spider-rs) — OUTWARD-FACING | lance-graph | Blocked | plan §4 (scope + robots + fork coords) |
| D-SCI-4a | `curiosity_mul` + qualia texture gestalt — the MUL exploration-gateway wire | lance-graph-contract | Shipped | `exploration.rs` + 12 tests (G-CM-1..5 + wonder-invariance); adversarially verified; `E-MUL-EXPLORATION-GATEWAY-1` |
| D-SCI-4b | Held-out frontier-ordering probe (curiosity_mul beats MUL-blind) + adjacent thinking | lance-graph | Queued | plan; corpus probe, later |

## self-reasoning-substrate-v1 — the derivation DAG as the pointer fabric one level up

Plan: `.claude/plans/self-reasoning-substrate-v1.md`. D-SRS-1/2/4 SHIPPED (D-SRS-4 CONFIRMED — the graph faithfully recovers its own provenance + confidence trajectory); D-SRS-3 SHIPPED as a falsifier that FIRED (conjecture not confirmed — width self-report is a member-count artifact; deterministic sep −0.002 formal KILL after the HashMap-order fix). Plan COMPLETE.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-SRS-1 | Derivation-pointer fabric over the 31,327-triple Bible KG | lance-graph | Shipped | `src/reason.rs` + 7 tests + `bible_wave` leg; soundness gate green (100% resolvable, acyclic); `E-SELF-REASONING-FABRIC-1` |
| D-SRS-2 | Shape detector + ancestry radix-trie relocation (reshaped) | lance-graph | Shipped | `src/{shape,ancestry}.rs` + 63 tests; v1 taxonomy self-falsified, v2 measured router green (trie==closure exact, 4.0×); SPOG G-lane; `E-SHAPE-DETECTOR-MEASURED-1` |
| D-SRS-3 | Basin self-codes + uncertainty self-report | lance-graph | Shipped (falsifier fired — conjecture NOT confirmed) | `src/basin.rs` + 72 tests + `bible_wave` leg; raw split-half ρ=0.583 refuted by label-shuffle null (member-count artifact); constant-n sep 0.051 + Bessel real ρ=0.002 ⇒ no semantic signal; `E-BASIN-WIDTH-IS-N-ARTIFACT-1` |
| D-SRS-3b | Evidence-composite uncertainty (operator-corrected: MUL×rung×NARS×freq) + kanbanstep drive | lance-graph | Shipped (kanban drive real; 3 cross-basin gates KILLED — composite=size) | `src/evidence.rs` + 86 tests; `EvidenceBasin::{gate,advance}` → `contract::kanban` (6 Flow/160 Hold/1 Block). G-SRS3b-1 (novelty) sep 0.007; G-SRS3b-2 (open-Q yield) ρ +0.326 but sep −0.013; G-SRS3b-3 (partial ρ|size) −0.077 → composite carries NO cross-basin signal beyond size; validated home = per-basin kanban drive. `E-DOOMSCROLL-VS-RUNG-LADDER-QUERY-1` |
| D-SRS-4 | The self-reference falsifier (provenance + confidence-delta) | lance-graph | Shipped (CONFIRMED — positive) | `src/introspect.rs` + 77 tests + `bible_wave` leg; G-SRS4-1 all 50k derived triples re-compose from premises; G-SRS4-2 windowed NARS confidence read == independent recount (0.500→0.991); `E-SELF-REFERENCE-LOOP-CLOSED-1` |

## literature-probe-ladder-v1 — literature as falsifier: 8 genres → 8 LC artifacts → previously-impossible milestones

Plan: `.claude/plans/literature-probe-ladder-v1.md`. PROPOSED, doc-only — captured pre-compaction (PR #803), no code yet.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-LIT-1 | Milton inversion via canonical cross-instances | lance-graph | Queued | plan |
| D-LIT-2 | Christie red-herring vs clue-chain, differential Jirak gate | lance-graph | Queued | plan |
| D-LIT-3 | Synoptic elect_peers source recovery | lance-graph | Queued | plan |
| D-LIT-4 | Derivation fabric over Bible triples — **shares its gate with D-SRS-1** | lance-graph | Queued | plan |

## w3-template-mask-v1 — W3: LC template mask + count-derived pair table (no finetuning)

Plan: `.claude/plans/w3-template-mask-v1.md`. PROPOSED, gated on D-W3M-1 — captured pre-compaction (PR #803), no code yet.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-W3M-1 | Count-vs-oracle probe — THE gate | lance-graph | Queued | plan |
| D-W3M-2 | 8 KB LC relation mask + cheap-check-first | lance-graph | Queued | plan |
| D-W3M-3 | StepMask integration | lance-graph | Queued | plan |

## causal-rung-standing-wave-v1 — p64→v3 cognition layer: amortized 2³ ladder + standing-wave awareness

Plan: `.claude/plans/causal-rung-standing-wave-v1.md`. Consumes M20 A1 (shipped) + A2/A3/A5/A6 (queued, auditor+mint gated) + selection #776 + temporal.rs. Probe-first: the probe is the next deliverable, not code; everything CONJECTURE until D-CSW-* report.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-WITNESS-FABRIC-1 | Tier-3 make witness tenants real (E-WITNESS-FABRIC-1): elect_peers (quorum/contradiction from window fabric, absolute-event agreement) + resolve_chain (hop budget + temporal.rs escalation) + is_opinion (persisted contradiction) | lance-graph | **In PR** (branch) — `witness_fabric` module, +7 tests → 983 green, clippy clean; algebra FINDING, Aesop-corpus falsifier registered CONJECTURE | `witness_fabric.rs`; E-WITNESS-FABRIC-1 |
| D-DISORDER-GATE-1 | Tier-1 mode router (E-DISORDER-GATE-1): defect-backed NaN groundedness pre-gate + Cynefin mechanical-core classify→elect (saccade/sweep/field-gather/stabilize) + MUL MountStupid veto | lance-graph | **In PR** (branch) — `dispatch_mode` module + `disorder_gate_probe`, 982 tests green, clippy clean, 5 probe gates green | `dispatch_mode.rs`; probe `disorder_gate_probe`; E-DISORDER-GATE-1 |
| D-REC-WIRE-1 | Recipe claim-audit (34 kernels measure on a scalar proxy, not the real organ) → wire the 3 real tenants (A9 CausalWitnessFacet 24 loci + SPO + qualia) into a rung-ordered, NaN-gated causal ladder keyed by NARS inference type | lance-graph | **Shipped** — PR #780 MERGED `8a00988` (operator-gated) — 3 contract modules + 3 probes, 970 tests green, clippy clean, all example gates green | `causal_witness.rs`/`recipe_substrate.rs`/`recipe_dispatch.rs`; probes `recipe_claim_audit`/`loci_recipe_relevance`/`recipe_ladder_over_substrate`; E-RECIPE-SUBSTRATE-WIRING-1 / E-LADDER-UNSHADOWS-SELECTOR-1 |
| D-REC-LOCI-1 | Door C — recipe dispatch gated on the real 24-loci causal-witness organ (closes #780 Axis B on the dispatch path); rung ORDER organ-derived (`loci_rung` = deepest required dimension); Maslow climb CARRIES lower-rung awareness up (anti-rediscovery) + higher thinking PRUNES lower-related | lance-graph | **In PR** (branch, this session) — `recipe_loci` module (11 tests) + `recipe_loci_walk` probe (4 measured gates: selector 7/34, organ 34/34 grounded, carry monotone, prune fires + apex survives), clippy clean | `recipe_loci.rs`; `examples/recipe_loci_walk.rs`; E-RECIPE-LOCI-ORGAN-GATE-1 |
| D-GUARD-1 | The recipe grounding gate is the MULTIPASS MARKOV STANDING WAVE, not a coarse scalar prefilter (operator ruling): `dispatch_guard` composes single-pass BINDING ∧ `witness_fabric::standing_wave_grounded`; scalar `nan_disqualifier` DROPPED (tautological subset). The ±8 is only the REFERENCE HORIZON — a chain that leaves it is `Escalate` (search causality over time / the absolute AriGraph SPO+Leiden basin), NOT coincidental (Romeo & Juliet: a distant cause is still a cause) | lance-graph | **In PR #785** (draft, this session) — `dispatch_guard` module (4 tests) + `standing_wave_grounded`/`WaveGrounding{Causal,Escalate,Unbound}` + `dispatch_guard_redundancy` (4 gates: single-pass blind 34/34, wave flips 34/34 Fires→Escalate) + jc `rung_divergence_reliability` (α 0.504 DISTINCT FACETS); contract green, clippy clean | `dispatch_guard.rs`; `witness_fabric.rs`; `examples/dispatch_guard_redundancy.rs`; `jc/examples/rung_divergence_reliability.rs`; E-MARKOV-STANDING-WAVE-GATE-1 / E-SUDOKU-TISSUE-WEAVE-1 / E-HORIZON-NOT-BOUND-1 |
| D-CSW-0 | Plan doc + O1 decision (canonical ladder masks + per-class facet election) | lance-graph | In PR #777; **O1 DECIDED (operator 2026-07-21: canonical masks)** | plan §2 |
| D-CSW-1 | Standing-wave probe: per-rung persistence over an ordered stream separates causal from coincidental vs single-cycle + p64 3×u8 baseline; escalation cascade prunes at zero separation cost | lance-graph | **v5 SPLIT VERDICT** — CORE standing-wave claim GREEN (auc_wave .997 vs single .878, vs p64-**wave** .500 [M3], vs **reverse .000** [M2 orientation control, +.997]; cascade .997); the SEPARATE §0.5 escalation-ECONOMICS gate 3a **KILLS** (pruned .333 < .40 registered — reported not retuned; the M1 witness correction dropped it from v4's .458) → DEFERRED. v1 fixture-ceiling + v2 mean−std KILLs recorded on the way; **leg 2 (real temporal.rs/Lance versions, wild corpora) NOT RUN** — genuine gap: no labeled real-causal-pair corpus + no real persisted Lance version data. _(CORRECTION E-DCSW1-LEG2-BLOCK-CORRECTION-1: the earlier "needs protoc" reason was WRONG — `lance-graph-planner` has no protoc dep and builds here in 19.78s; protoc is absent but blocks only the full workspace, not the planner. `temporal.rs` IS compilable here; a narrower synthetic-version-stream probe is feasible, flagged for operator.)_ _(v3 historical: .972/.875/.375/.458 — pre M1/M2/M3 fixes.)_ | probe `deepnsm/examples/probe_dcsw1_standing_wave.rs` (v5); E-DCSW1-V5-SPLIT-VERDICT / E-CAUSAL-TISSUE-ALREADY-SHIPS-1 / E-DCSW1-LEG1-GREEN; plan §6.2/§6.5 |
| D-CSW-2 | Basin→causal-edge candidate probe: co-occupancy + rung survival vs basin-only / rung-only ablations | lance-graph | **Contract-level mechanism PASS** — synthetic AND-gate fixture (real `PairPalette`+witness-fabric primitives, deterministic): joint precision@25 = 1.000 vs basin-only 0.520 / rung-only 0.520 (margin +0.480 each, registered pass ≥0.15). Promotes the JOINT-SIGNAL MECHANISM from CONJECTURE to scoped FINDING — NOT the real-corpus D-CSW-2 claim itself, which stays open pending real basins from real data. | probe `lance-graph-contract/examples/probe_dcsw2_basin_rung.rs`; E-DCSW2-CONTRACT-MECHANISM-GREEN-1; plan §6.3 |
| D-CSW-3 | jc reliability: full-width amortized ladder vs CE64 64-bit cram (extends M20 D-AW-5) | lance-graph | Queued (needs A2/A6 lanes + real data) | plan §6 |

## soa-32-tenant-awareness-redundancy-v1 — M20: honest full-width awareness → jc-measured collapse

Plan: `.claude/plans/soa-32-tenant-awareness-redundancy-v1.md`. Advances ENTROPY-MILESTONES M20. Builds the full-width awareness SoA (13→32 tenants) BESIDE CausalEdge64/EW64 (kept for reference), then jc measures the true awareness width. Rides the D-TRI / BoardAggregates batched OGAR mint. NO bytes land before the envelope-auditor verdict.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-AW-1 | Assembly plan + `v3-envelope-auditor` layout-delta gate (RESERVE-DON'T-RECLAIM, fit, version-stability, slot-purity) | lance-graph | In progress (plan drafted; auditor running) | plan §2/§4 |
| D-AW-2 | A1 `SpoFacet` — 3 SPO + 3 episodicwitness palette256² (`6×(8:8)` L4); user's base design | lance-graph | Reading primitive SHIPPED (`awareness_facet::SpoFacet`, 6 tests + doctest, reuses #729 rails); byte carve + OGAR value_schema mint (Place 2) still pending | plan §2 A1, §0.5 |
| D-AW-3 | A2–A7 awareness facets (PearlRung/NarsTruth/FreeEnergy/StreamCycle/DirInfer/WitnessLens), derived from CE64/EW64 fields; batched-mint gated | lance-graph | Queued (auditor + mint gated) | plan §2 A2–A7 |
| D-AW-4 | Redundant sibling lanes (2nd representation per construct — Fisher-z i8 / raw-COCA-12bit) to reach ~32; count jc-derived, not pre-committed | lance-graph | Queued | plan §2 |
| D-AW-L9 | L9 `TekamoloWindowBinding` schema (A9 = 24 edge loci of Markov context agreement; rungs occupy zero slots) + real-text probe: validity ante 0.727 / kausal 0.750 (gates green), reliability battery well-posed (per-dim ICC .07–.68, α .448 = distinct facets), v1 noun-only KILL 0.455 → v2 loci-chaining (the register following its own nibbles) | lance-graph | Schema in plan §2.9 (awaiting operator L9 §3-catalogue ratification); probe SHIPPED green | probe `jc/examples/l9_loci_real_text.rs`; E-L9-REAL-TEXT-1 |
| D-AW-5 | jc collapse gate — **EXTENDS D-TRI-2** to the awareness lanes: Cronbach α per construct + pairwise ICC/Spearman → measured awareness width (M20 mechanical gate) | lance-graph | Queued (needs lanes + real data) | plan §3 |

## graphrag-doc-retrieval-soa-integration-v1 — retrieval over AriGraph (expand-in-place, no new crate)

Plan: `.claude/plans/graphrag-doc-retrieval-soa-integration-v1.md` (v1.2). Pure/reversible capabilities land ahead of G0; the D-GR-2 retrieval WIRING is gated on the G0 real-corpus verdict.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-GR-1 | `DocGraphQuery` zero-dep contract trait + `ScoredId` (rung→walk dispatch) + D-GR-2 design | lance-graph | Shipped (#716) — `doc_graph.rs`, 9 tests | plan §5 |
| D-GR-3a | `TripletGraph::communities()` multi-level Louvain, deterministic | lance-graph | Shipped (#714) | plan §3b |
| D-GR-3b | PPR (`personalized_pagerank`) + Leiden `refine_connected` + BM25 (`Bm25Index`) — pure capabilities | lance-graph | Shipped (#716) — 13 tests | plan §3b, §5 |
| G0 | P-GRAPH-LOADBEARING harness (vector-only vs vector+PPR+community) | lance-graph | Harness shipped (#716); real-corpus verdict OPEN | plan §5, §6 |
| D-GR-2 | Fuse CAM-PQ+SPO-G+PPR+community into `retrieval.rs` under the #708 RungElevator | lance-graph | Design done (in `doc_graph.rs` module-doc); impl GATED on G0 | plan §5 |
| D-GR-2a | RRF fusion primitive (`reciprocal_rank_fusion`, Cormack 2009) — the fusion algebra D-GR-2 needs; pure, ahead of G0 | lance-graph | Shipped (#724) — `arigraph/rrf.rs`, 9 tests + doctest | plan §5 |
| D-GR-2b | Chained `episodic_search` (AriGraph Eq. 1) — semantic-seeded episodic recall; pure, ahead of G0 | lance-graph | Shipped (#725) — `arigraph/episodic.rs`, 6 tests | plan §5 |
| D-GR-2c | Thesis partition (`theses()` — PersonalAI per-proposition, no-LLM structural heuristic); pure, ahead of G0 | lance-graph | Shipped (#727) — `arigraph/episodic.rs`, 5 tests | plan §5 |
| D-GR-2d | Evidence-chain path structure (`associated_paths` + `render_chain`, StepChain Πsᵤ); pure, ahead of G0 | lance-graph | Shipped — `arigraph/triplet_graph.rs`, 6 tests | plan §5 |
| D-GR-4 | Community summaries (no-LLM DeepNSM; Rig-oracle tail) | lance-graph | Deferred (W3-coupled) | plan §5 |
| D-GR-5 | `ogar-doc` reconstruct/related-docs → `DocGraphQuery` seam | lance-graph + OGAR | Deferred (mint-gated, doc-W4 council) | plan §5 |
| D-GR-6 | Witness-KV separation (DocumentID handle → consumer KV) | lance-graph | Deferred (doc-W4 council) | plan §4a, §5 |
| P-COMMUNITY-BASIN-AGREE (S1) | Empirical probe: Leiden community vs `is_a`-basin agreement, φ via `jc::pearson` (consumes jc science, doesn't extend it) | lance-graph | Harness SHIPPED — φ=1.0 aligned / 0.55 bridged (`robot` = the bridge); real-corpus verdict OPEN, **gates the D-TRI-1 community-id mint** | plan §6, #719 |

## triangle-tenants-gestalt-separation-v1 — triangle tenants, surface separation, chess quarantine

Plan: `.claude/plans/triangle-tenants-gestalt-separation-v1.md`. Design shipped; ALL layout work mint-gated (rides the same batched mint as W2a BoardAggregates + Tasks-SoA task classid + chess classids).

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-TRI-1 | Triangle tenant spec (Frozen/Learned/Explore, 12 slots x palette256) through envelope-auditor T1-T6, behind the batched mint | lance-graph | **VALUE-TENANT HALF MERGED (#717, main 74d16f92)** — 3 lanes `ValueTenant::{FrozenStyle=10,LearnedStyle=11,ExploreStyle=12}` U8×12 at row_offset 152/164/176; `Full`-only; zero ENVELOPE_LAYOUT_VERSION bump; `NodeRow::{style_lane,set_style_lane,triangle_for}`; auditor LAYOUT-CLEAN; Codex P2 fixed. **classid half = SPEC READY, CORRECTED 2026-07-18** (`dtri1-classid-mint-spec-v1.md`; `E-COGNITIVE-ATOMS-ALREADY-FROZEN`): the **only real concept mint is chess `0x06`**; the proposed "Cognition `0x03` domain" was a rediscovery error (cognitive task types are already frozen atoms in `holograph::dntree` — INFERS=74/DEDUCES=82/INDUCES=83/ABDUCES=84/COUNTERFACTUAL=0x84/syllogize()), byte `0x03` stays reserved. BoardAggregates@188 = value tenant (no classid). Persona = per-consumer opt-in mint (chess-yes / business-no), not this batch. **Gated on the doc-W4 batched-mint council** (never solo); community-id does NOT mint (S1 retracted by #722). One open knob: BoardAggregates width. | plan §1/§2/§2a/§5/§6, `dtri1-classid-mint-spec-v1.md` |
| D-TRI-2 | 12-family vs 12-step reading agreement: jc battery (ICC, Pearson/Spearman, Cronbach alpha) over real shader cycles; collapse only on measured identity | lance-graph | Queued | plan §4, §6 |
| D-TRI-3 | Nail->hammer dispatch probe: object resonance -> atom retrieval vs inverted baseline; no inverted read path exists structurally | lance-graph | Queued | plan §3, §6 |
| D-TRI-4 | Chess<->thinking transfer measurement (validity + reliability) — the quarantine-lift gate | lance-graph + stockfish-rs | Queued | plan §5, §6 |
| D-TRI-5 | Emulation != resonance: counterfactual-goalstate emulation vs resonance-only baseline on opponent move prediction (operator correction §2a; builds on D-SF-OPPONENT-1/3) | lance-graph + stockfish-rs | Queued | plan §2a, §6 |
| D-TRI-6 | Pyramid settlement: settle-rung distribution over real shader cycles (base-heavy expected); homeostatic descent verified; elevator threshold jc-calibrated | lance-graph | In PR (P3) — ascent loop WIRED (driver rung→predicate-plane widen; identity-at-base, superset-monotone); settlement probe green; real-cycle distribution + jc threshold calibration still open | plan §3a, §6 |

## epiphany-integration-2026-07-04-v3 — membranes, parity, unified ruff phases

Plan: `.claude/plans/epiphany-integration-2026-07-04-v3.md`. Full 13-agent review pipeline complete; execution queued behind D1→A1. Cross-repo (ruff/OGAR/lance-graph); consumer-side baton homes per BH-1/2/6.

| D-id | Title | Repo | Status | Evidence |
|---|---|---|---|---|
| D-EPI-D1 | Broadcast entry: lane claims ruff work + plan announce | lance-graph | Queued | plan §2 Group D |
| D-EPI-A1 | NEW ruff branch off origin/main@HEAD (never re-point shared name — BH-4) | ruff | Queued | plan §2 A1 |
| D-EPI-D2 | Minimal ruff board file w/ plan pointer + A2b/A7 gate markers | ruff | Queued | plan §2 D2 (the consumer-side baton home) |
| D-EPI-A2a | Predicate registry freeze (derived count 62; prose cites test) | ruff | Queued | S5: assert exists at triple.rs |
| D-EPI-A2b | Opacity invariant into IR record (4-crate cascade) | ruff | Blocked (B1 council verdict) | plan §4 edge |
| D-EPI-A3 | C# golden fixture (Python #40 + Ruby already emit inherits_from) | ruff | Queued | S5-D3 |
| D-EPI-A4 | Reassembler generalization (FEATURE: per-predicate inverse logic) | ruff | Queued (own PR) | cascade-impact rescope |
| D-EPI-A5 | Cross-language convergence gate (tests E-CONVERGENCE-GATE-FIRST-1 [CONJ]) | ruff | Queued | greenfield (S5) |
| D-EPI-A6 | Mint→ndjson seam + registry-version stamp | ruff | Queued | iron-rule I-LEGACY consequence |
| D-EPI-A7 | Falsifier-fence CI + genericize surviving medcare:* fixtures | ruff | Blocked (Q-A7) | BH-5 |
| D-EPI-B1 | Council filing: 2 parents + 14 rows | lance-graph | Queued | plan §1 |
| D-EPI-B2 | v3 census forward-ref convention + W6-AriGraph pointer | lance-graph (V3-owned) | Queued (broadcast-first, 7-day fallback) | BH-3 |
| D-EPI-B3 | tenants.md registry columns (10 shipped + BoardAggregates PENDING-GATED) | lance-graph (V3-owned) | Queued (broadcast-first, 7-day fallback) | S1-D1/S2-D1 |
| D-EPI-B4 | E-V3-RIG-ARM amendment (mounts-on; shell/organs) | lance-graph | Queued | S2-D2 trim |
| D-EPI-C1 | OGAR falsifier-fence non-negotiable bullet | OGAR | Queued | S3 insertion point |
| D-EPI-C2 | OGAR Türsteher-carry bullet (§1.6, cites capstone) | OGAR | Queued | S3 insertion point |

## deepnsm-v3-convergence-v1 — DeepNSM is the encoder that fills reserved tenants

Plan: `.claude/plans/deepnsm-v3-convergence-v1.md` (`E-V3-DEEPNSM-IS-THE-ENCODER-NOT-A-MIGRATION-1`). Static convergence PROVEN by #624 P0–P5; the memory layer is the genuinely-unbuilt seam. Extends `v3-convergence-wiring-v1` (wire-don't-invent).

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-DNV-1 | Gridlake carrier: `GridBatch::as_gridlake_columns` → `ndarray::simd::MultiLaneColumn` (i32 min/max, i64 sum, u64 count); the carrier the COCA `Cell` also rides | onebrc-probe (+ndarray) | Shipped (#641, error-type follow-up #642) | lane-j pulls ndarray; LE roundtrip + unaligned reject + typed GridlakeCarrierError |
| D-DNV-2 | deepnsm `SpoTriple` → `CausalEdge64` S/P/O+freq/conf → `SpoHead`; run `nars_engine.all_projections()` (2³) end-to-end from a real COCA FSM parse | deepnsm + planner + causal-edge (osint probe) | In PR | `p6_real_coca_2cube.rs`: 2 tests green — real-parse S/P/O round-trips the edge carrier (extends P2), and the 2³ ladder holds on a real-derived head (extends P3b); palette is the documented codebook stand-in |
| D-DNV-3 | arm-discovery as the 2nd proposer leg into one SpoStore (shares palette256 oracle) | arm-discovery + deepnsm | Blocked (ARM-JIRAK-FLOOR) | D-ARM-7 Jirak noise floor is the hard prereq |
| D-DNV-4 | Episodic-witness tenant + `basin=family` wake (`witness_tombstone` calcify chain) | contract + arigraph | Blocked (own wave + probe) | no episodic-witness ValueTenant; calcify chain is `todo!()`; basin=family doc-only |

## v3-substrate-integration-v1 — the .claude/v3/ consolidation (W0–W6)

Plan: `.claude/v3/INTEGRATION-PLAN.md` (stub: `.claude/plans/v3-substrate-integration-v1.md`). Adopts (does not re-mint) D-MBX-A6, D-PERT-1, D-CC-*, D-VCW-3/5/7, D-CCF-4.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-V3-W0a | `.claude/v3/` tree (README, plan, COMPONENT-MAP, ENTROPY-MILESTONES, MODULE-TABLE, soa_layout/*) | docs | Shipped (this PR) | complete: 7/7 mappers synthesized; MODULE-TABLE = 304/304 files (21/21 census chunks); soa_layout 5/5 docs |
| D-V3-W0b | V3 awareness layer (knowledge docs, v3-* agent cards, /v3 skill, /v3-audit command, CLAUDE.md+BOOT.md entrypoints) | docs | Shipped (this PR) | 4 knowledge docs, 4 cards, skill+command registered |
| D-V3-W1a | SoaEnvelope::mailbox_owner() ownership stamp | lance-graph-contract | Shipped | this branch; 775 contract tests green |
| D-V3-W1b | Ahead-firing batch writer (cast pairing + AHEAD KanbanMove at cast) | planner-adjacent | Shipped (#631, 2026-07-02; row flipped 2026-07-10 hygiene) | W1 STARTED 2026-07-02; WAL-shaped per preflight addendum (M24: cast = intent record) |
| D-V3-W1c | Delegation cache (cast id vs envelope stamp) | batch writer | Shipped (#631, 2026-07-02, collapsed into W1b per M24; row flipped 2026-07-10 hygiene) | W1 STARTED 2026-07-02; collapses into W1b writer (M24) |
| D-V3-W1d | MailboxId minting path (non-zero owners, uniqueness debug_assert) | contract | In progress | W1 STARTED 2026-07-02 |
| D-V3-W1e | Probes: ahead-update ordering + delegation miss | contract/planner | Shipped (#631, 4/4 green; verified live un-ignored 2026-07-10) | W1 STARTED 2026-07-02; probe lands FIRST (probe-first gate) + kill-after-cast replay test (M24) |
| D-V3-W2a | Per-mailbox kanban board as TENANT | contract | Queued (GATED: Addendum-12a — BoardAggregates 10th ValueTenant @152 + T1-T6 + board classid via next BATCHED mint, never solo; deliberately deferred 2026-07-10) | field-isolation matrix mandatory |
| D-V3-W2b | Supervisor wiring: moves via MailboxSoaOwner::advance_phase | lance-graph-supervisor | Shipped (kanban_actor.rs + tests/w2b_real_owner_probe.rs; re-verified 3/3 green 2026-07-10 — row was stale) | plan W2 |
| D-V3-W2c | symbiont SurrealDB-on-kv-lance arm | symbiont | RE-SCOPED (E-ORCHESTRATION-ORGANS-1, 2026-07-10): storage + SurrealQL read-glove + ExecTarget lowering ONLY — never orchestration; kanban-updates-as-KV-transactions dropped | POC = kanban_loop.rs (read glove); resolves the W2c/D-PG-6 dual-row contradiction |
| D-V3-W2d | 550 ms budget hooks via planner elevation/ | lance-graph-planner | In PR (2026-07-10, branch `claude/review-claude-board-files-nhqgx1`) | `elevation::cycle::CycleBudget` (M12 allocator): reads the Libet anchor, advisory `admits` (reprioritize-never-gate), measured consts (66µs/card lane-E, ~0.5µs/step), +5 tests; load-balancer consumption = W2 residue |
| D-V3-W3a | StepMask in contract (sibling of FieldMask) | lance-graph-contract | In PR (2026-07-10, branch `claude/review-claude-board-files-nhqgx1`) | `contract::step_mask::StepMask`, +5 tests (866 lib green), selection-never-control-flow doc'd |
| D-V3-W3b | ElixirTemplate → graph-flow GraphBuilder adapter (ownership inheritance) | rs-graph-llm seam | Queued | plan W3 |
| D-V3-W3c | Rig oracle node + equivalence-gated compile-down | cognitive-compiler + rig | Queued | D-VCW-7 lineage |
| D-V3-W3d | Template catalogue keyed internally (classid keying deferred to P4) | template-runtime | Queued | plan W3 |
| D-V3-W4a | BusDto cast-pairing call sites | cognitive-shader-driver | In PR (2026-07-10, branch `claude/review-claude-board-files-nhqgx1`) | `MailboxSoA::cast_on_behalf` (owner from the CARRIER — mispair unrepresentable) + `BatchWriter::on_behalf_of` audit getter; 3 tests incl. literal BusDto arm; fixed pre-existing standalone `with-planner` E0432 (planner_bridge gated onto its wire transport) |
| D-V3-W4b | L4 learning-loop end-to-end probe (residue → owner-stamped lane → next-cycle template read) | cross-crate | Queued | plan W4 |
| D-V3-W5a | q2 CI re-bakes + body.soa re-release + drop FMA_V3_CLASSID_LEGACY | q2 | Queued | handover continuation §1 |
| D-V3-W5b | cpic contract pull with mereology (kinds → cascade positions) | q2 + contract | Queued | handover F3 |
| D-V3-W5c | Consumer write-on-behalf adoption (bakes annotated bootstrap; new online writes via batch writer) | fleet | Queued | write-on-behalf.md |
| D-V3-W5e | ladybug-rs + smb-office-rs contract pulls | siblings | Queued | never bridges |
| D-V3-W6a | Adoption/corpus scanner (ONE two-metric range-count tool) | lance-graph | In PR (counting logic shipped 9c55646 2026-07-02 — row was stale; runnable examples/adoption_scan.rs added 2026-07-10; Lance-dataset sweep = residue) | E-V3-MARKER-IS-A-MONITOR; note: 0x1000 PERMANENT per E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1 — scanner counts forms, monitor never retires |
| D-V3-W6b | Legacy alias retirement (corpus-proof-gated) | contract + consumers | Blocked (corpus proof) | plan W6 |
| D-V3-W6c | Custom half opens: render + template catalogue dispatch | contract | Blocked (P4 operator checkpoint) | completes F2 styles-as-lenses |

## temporal-markov-and-style-classes-v1 — the ratified 2026-07-10 cognition arc

Plan: `.claude/plans/temporal-markov-and-style-classes-v1.md`. Rulings: E-MARKOV-TEMPORAL-STREAM-1 / E-THINKING-STYLES-ARE-CLASSES-1 / E-ORCHESTRATION-ORGANS-1 / E-ACK-IS-THE-KANBAN-TRIGGER-1.

| D-id | Deliverable | Owner | Status | Notes |
|---|---|---|---|---|
| D-MTS-1 | Markov-as-stream parity probe (temporal version-range vs VSA ±5 braid, DeepNSM corpus) | lance-graph | Queued | gates ALL VSA-path removal; truth-architect reviews |
| D-MTS-2 | L4 palette256² shader fidelity certification (vs 0.96–0.998 anchors; representation engineered first) | cognitive-shader-driver | Queued | certification-officer battery |
| D-MTS-3 | Hierarchical-4⁴ vs flat-256 codebook fidelity (OGAR F11-adjacent) | ndarray/bgz17 | Queued | 2bit×2bit cascade prefix rigor |
| D-MTS-4 | M4 cutover target sharpened: MailboxSoA + temporal stream + palette tenants | driver | Queued | rides M4 parity gate |
| D-MTS-5 | Pythagorean-comma vertical-quorum probe (comma-offset vs aligned pyramid; coprime-walk quantization per D-QUANTGATE; Jirak significance) | shader/ndarray | **Measured GREEN 2026-07-10** | `perturbation-sim/examples/comma_quorum.rs`, all pre-registered gates PASS: comma N_eff 11.00/12 vs strict 1.00 / unit 2.49 / rational 3.92; replay bit-identical any order; fresh level-12 +0.83 witnesses at max\|ρ\|=0.156; 82 KB touched vs ~69 GB dense. Bonus measured boundary: N_eff(comma) = min(L, spectral participation) — run #1 FAIL 3.24 kept in the chronicle. See E-COMMA-QUORUM-MEASURED-1 |
| D-TSC-1 | M9 ThinkingStyle dedup (5+ copies → contract taxonomy) | workspace | **Shipped 2026-07-10** (first 5+3 council run: spec v1→v2→v3 ratified; `contract::style_family::StyleFamily` + `default_runbook()`/`family()`; FIVE divergent tables replaced — E-FIVE-STYLE-TABLES-1; G1 grep = 1 enum + 3 deprecated aliases; 1549 tests green across 4 crates) | UNBLOCKS D-TSC-2..4 + StepMask catalogue (M9 gate); behavior changes documented + G7-pinned |
| D-TSC-1b | D-TSC-1 dedup MEASURED: `jc::reliability` agreement probe over the 3 shipped 12-family param tables (`UNIFIED_STYLES` / thinking-engine `StyleParams` / planner `FieldModulation`) — the D-TRI-2 mint-free cousin | jc example + workspace | **In PR** | `crates/jc/examples/style_table_agreement.rs`; A≡B perfect (M9 confirmed); planner 7-explicit IDENTITY; only the 5 planner `default_modulation` fallbacks drift (Mode A ICC 0.71 AMBIGUOUS) → TD to fill them from canonical; retires numeric half of O5. E-D-TRI-2-MINT-BLOCKED-1 |
| D-TTV-1 | Thinking-related tenants → V3 value-tenant substrate (keep old CausalEdge64 as perturbation baseline) | driver/contract | Queued | E-THINKING-TENANTS-V3-1; envelope-auditor gated; batched mints only |
| D-MTS-6 | Smaller-CausalEdge64 × comma awareness curve vs old-CE64 baseline (find the knee: how many stored bits the comma reconstruction replaces before awareness degrades) | shader/perturbation-sim | **Measured GREEN 2026-07-10** | `perturbation-sim/examples/comma_awareness.rs`, all pre-registered gates PASS: **k\*=1** (2 explicit truth bits/edge vs baseline 16) matches all three awareness proxies; aligned control needs k\*=4; the comma lattice buys ≈3.4 effective bits ≈ log₂(12); replay bit-identical; run-#1 G1 mis-registration diagnosed (max flip margin 1.7e-5 = boundary noise) — see E-COMMA-AWARENESS-MEASURED-1. **D-MTS-6b** (driver-integrated fixture) gates any real CE64 shrink |
| D-TSC-2 | Batched cognition-domain mint in OGAR (+ classify_form reconciliation if 0xFFFF) | OGAR | Queued (blocked by D-TSC-1) | never solo; COUNT_FUSE |
| D-TSC-3 | Style masks + rung set + KausalSpec as class-record properties | contract + OGAR | Queued (blocked by D-TSC-1/2) | dispatch stays MetaWord bits |
| D-TSC-4 | W6c coexistence re-ruling (catalogue shares custom half with PERMANENT 0x1000) | operator | ESCALATED | ruling needed, not assumed |
| D-ORG-1 | BatchWriter::ack_and_propose self-pumping loop + probes | planner | Shipped (2026-07-10, 2 tests green) | E-ACK-IS-THE-KANBAN-TRIGGER-1 |
| D-ORG-2 | W2c re-scope to storage/read-glove | board | Shipped (2026-07-10) | row above updated |

## classid-canon-custom-flip-v1 — the TRIGGERED §2.3 atomic flip

Plan: `.claude/plans/classid-canon-custom-flip-v1.md`. Operator trigger 2026-07-02.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-CCF-0 | compose_classid/split_classid/CLASSID_CANON_HIGH + route all sites (zero behavior) | lance-graph-contract | Shipped (fd9bf6b) | plan §3/§4 P0 |
| D-CCF-1 | Flip + mint new-form classids (0x0701_1000 / 0x0A01_1000 / 0x0E01_1000) coexisting | lance-graph-contract | In PR (#628) | gated on P0 probes |
| D-CCF-2 | OGAR#95 hi-u16 app-prefix reconciliation | contract + OGAR | In PR (OGAR #147; prefix = custom half) | plan §2 row / §4 P2 |
| D-CCF-3 | q2 re-mints (osint-bake + cpic via contract pull; dissolves ISS-Q2-CPIC-MIRROR) | q2 (gate WAIVED) | In PR (q2 #71; .soa re-bakes deferred to CI/dev; cpic interim 0x0E01_000N, full contract pull tracked) | plan §4 P3 |
| D-CCF-4 | 0x1000 marker retirement | all | RESCINDED (operator 2026-07-03, E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1: v2/v3 coexist permanently by schema — retirement off the table) | plan §4 P4 (superseded) |
| D-PERT-1 | Rename dto.rs ResonanceDto → PerturbationDto (split, not dedup; deprecated alias; awareness_dto keeps Resonance) | thinking-engine + engine_bridge | Shipped (#630, 2026-07-02; verified in-code 2026-07-10 — row was stale) | E-TWO-RESONANCES-SPLIT |

## v3-convergence-wiring-v1 — wire, don't invent (the seam list)

Plan: `.claude/plans/v3-convergence-wiring-v1.md`. Sonnet-grindwork/Fable-decisions split.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-VCW-1a | RungLevel arithmetic + RungElevator (sustained-BLOCK policy over certified mask algebra; converged with escalation::rung_delta) | lance-graph-contract | **Shipped** | 755 lib tests green incl. 6 new; clippy clean |
| D-VCW-1b | Driver wiring: elevator through cycle loop, ctx.rung proxy retired, wire/grpc from_u8 dedup | cognitive-shader-driver | **Shipped** | driver 100/100 green (2 new tests: sustained-BLOCK elevation across dispatches + rung load-bearing in tactic selection); driver-persistent RwLock elevator, base-change reset |
| D-VCW-2 | P6 wave-convergence probe (wave dist == certified palette read) | lance-graph core (arigraph) | **Shipped** | markov_soa 6/6 green (2 new P6 tests) |
| D-VCW-3 | P7 render probe (bitmask → askama; fields == masked tenants) | q2 (**gate WAIVED 2026-07-02**) | Queued (unblocked) | spec ready (plan §3) |
| D-VCW-4 | One-row registry + read-mode parity fuse | contract (+OGAR Phase B) | Queued | plan §4; Phase B operator-gated |
| D-VCW-5 | cascade3 nibble-ancestry falsifier | q2 (**gate WAIVED 2026-07-02**) | Queued (unblocked) | ISS-Q2-CASCADE3-NIBBLE-ANCESTRY |
| D-VCW-6 | Rule 7: negative-existence claims need exhaustive-search declaration | knowledge doc | **Shipped** | autoattended-multiagent-pattern.md §5 Rule 7 |
| D-VCW-7 | rig/rs-graph-llm FailureTicket loop | rs-graph-llm (sibling) | Deferred | plan §6; probe-first when opened |

## cognitive-compilation-v1 — Elixir-template stack (LLM teaches, Lance-Graph runs)

Plan: `.claude/plans/cognitive-compilation-v1.md`. The new idea is the
Elixir-shaped template; the rest of the loop reuses existing organs.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-CC-RUNTIME-1 | elixir-template: representation + parser + `source_ranking_v1` slice | elixir-template | **Scaffolded** | 6 tests green (parse, version split, custom atom, round-trip, 7-step slice + guardrail); clippy clean |
| D-CC-RUNTIME-2 | template-runtime: deterministic OGAR-action dispatch (reflex executor) | template-runtime | **Scaffolded** | 4 tests green (threaded dispatch, unknown-action, empty, unimplemented-bubbles); action bodies deferred |
| D-CC-EQUIV-1 | template-equivalence: replay grading | template-equivalence | **Scaffolded** | 4 tests green (Exact, RankOrder-within-tolerance, new-claim-fail, confidence-drift-fail); Semantic deferred |
| D-CC-COMPILER-1 | cognitive-compiler: trace→template synthesis surface | cognitive-compiler | **Scaffolded** | 3 tests green (NotImplemented contract, non-Execution reject, unsourced-claim reject); synthesis = first probe |
| D-CC-RIG-1 | rig-surrealdb pointed at AdaWorldAPI kv-lance fork | rig (sibling) | Queued | additive Cargo wiring |
| D-CC-RUBICON-1 | graph-flow Task for templates (isolated, cherry-pickable) | rs-graph-llm (sibling) | Queued | local copy + branch push as recovery paths |
| D-CC-OGAR-1 | OGAR canonical classes for the loop | OGAR ogar-ontology/ogar-from-elixir | **Exists** | reused, not rebuilt |
| D-CC-INDEX/REVIEW/PROMOTE/LEDGER/FENCE | basin match / reviewers / PR automation / provenance / ownership fence | planner / agents / surreal kv-lance / ractor | **Exists or DEFERRED** | not this PR |

---

## symbiont-golden-image-harness — the living all-in-one substrate binary + the first runtime edges

The golden image (`crates/symbiont`, workspace-`exclude`d): the full Ada stack in ONE binary, then real cross-crate edges onto the canonical SoA. Plan: `crates/symbiont/INTEGRATION_PLAN.md` (PR #555, merged `37cc21b2`).

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D0 | Golden image compiles+links (lockstep lance-7) | symbiont | **Shipped** | git-deps build `CARGO_EXIT=0`, unified `lance 7.0.0 / lancedb 0.30.0 / df 53.1 / arrow 58`, binary runs |
| D1 | Grid→NodeRow bridge — each bus = 1 SoA board, f64 → `Energy` tenant | symbiont/bridge.rs | **Shipped** | 2 probes green; 64 buses→64 NodeRows, perturbation in the Energy(f32) tenant, all finite |
| E2 | Parallel SoA sweep at scale (16k boards = 8 MiB, zero-copy) | symbiont/bridge.rs | **Shipped** | `run_scale_demo(16384)` → 8 MiB, all 16384 Energy tenants finite |
| D3-AMX | Domino POC — 16-board AMX 16×16 BF16 Morton-tile cascade + NaN-projection | symbiont/domino.rs | **Shipped** | 3/3 tests green; 256 boards × 16 AMX-16×16 batches × 3-stage BF16 Morton-tile Domino cascade, NaN-clean via the projection surface. Polyfill-only (`ndarray::simd::bf16_tile_gemm_16x16` re-export `05bfea7a` jirak; `f32_to_bf16_batch_rne`; only `morton4` consumer-side). **Ran AVX-512 fallback** — AMX genuinely OFF on this guest (functional probe `/tmp/amxcheck`: XCR0 tile bits 17/18 = 0, `arch_prctl(158)` XTILEDATA = **-95 -EOPNOTSUPP** kernel refuses; CPUID also masked). NOT merely CPUID-masked → cannot be enabled here; a forced byte-encoded TDPBF16PS would fault. AMX dispatch correct + arch_prctl-158 gotcha-safe; fires `[AMX TDPBF16PS]` on an AMX-granted guest. |
| D2 | Kanban loop — pure-SoA slice (version-tick → `NextPhaseScheduler` → `try_advance_phase`) | symbiont/kanban_loop.rs | **Shipped (slice)** | 2/2 tests green; `SymbiontBoard` impls `MailboxSoaView`+`MailboxSoaOwner` over the `Vec<NodeRow>`, a `u32` tick stands in for the Lance subscription; drove `Planning→CognitiveWork[BF16 Domino sweep]→Evaluation→Commit`, Libet anchor on the Σ-crossing, halted absorbing in 3 cycles, NaN-clean. Reuses the COMPLETE contract kanban surface (`KanbanColumn`/`KanbanMove`/`NextPhaseScheduler`/`MailboxSoa{View,Owner}`) — zero new types. **ractor = ownership guarantee** (no messages, no tokio; E-CE64-MB-4 / #477 "nothing transmitted between mailboxes") — already embodied by `SymbiontBoard`'s single `&mut` owner, NOT a deferred message actor. **Trigger is SYNCHRONOUS — the writer fires it:** `VersionScheduler::on_version(&view, DatasetVersion(u64), exec)` is a sync pure function; a batch writer knows the version it committed and fires the kanban update inline (`on_version`→`try_advance_phase`, no async). `surreal_container/tests/scheduler_seam.rs` drives the whole Rubicon arc with plain `#[test]` feeding `DatasetVersion(i)`; `cognitive-shader-driver` `MailboxSoA` test 11 runs the same in-RAM loop (`mailbox_soa.rs:700` "no surreal / ractor message bus needed"). This loop's `u32` tick IS that pattern. Async is ONLY the Lance write I/O + the SUBSCRIPTION variant `LanceVersionScheduler::drive_once` (async because it READS a version it didn't write; shipped, 5 tokio tests). Only `surreal_container::read_via_kv_lance` is a stub. |
| E1 | Spain-grid acceptance gate (real fixture, NaN-free, clippy+machete clean) | symbiont | Queued | the north star — first N *real* nodes on the SoA in parallel |
| BT | Battle-test plan (probes A1–E3, gated behind singleton-BindSpace→SoA) | workspace | **Shipped (doc)** | `crates/symbiont/BATTLE_TEST_PLAN.md`; A1 partial-green + D1 green; A2–E3 specced |

---

## entropy-ladder-spo-rung-v1 — Staunen↔Wisdom entropy coordinate unifies SPO rungs + NARS reliability (R1 shipped; R2–R6 roadmap)

Plan path: `.claude/plans/entropy-ladder-spo-rung-v1.md`. Foundation: `ndarray::hpc::{reliability, edge_codec, entropy_ladder}`. Selector: `lance-graph-contract::EdgeCodecFlavor`.

| D-id | Title | Crate(s) / repo | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|
| D-EL-1 | Entropy-ladder foundation (reliability + edge_codec + entropy_ladder + EdgeCodecFlavor + bgz17 fix) | ndarray + lance-graph-contract + bgz17 | LOW | **In PR** | `d3b608f`,`83be7c3`,`920671d`,`6d48ced`; ρ=−0.78; ICC 0.97–0.99 |
| D-EL-2 | `entropy_class` → CausalEdge64 spare bits [63:61] | `causal-edge` | MED | **Queued** | version-gated + field-isolation (I-LEGACY-API-FEATURE-GATED) |
| D-EL-3 | CAM-PQ AMX centroid assignment (GEMM + 2×2/4×4 grid) | `ndarray` | MED | **Queued** | bit-exact + GMAC/s probe |
| D-EL-4 | HHTL+helix basin attraction | `lance-graph` + `helix` | MED | **Probe queued** | +15% recall vs HHTL-alone gate |
| D-EL-5 | Markov SPO rung-ladder → episodic context | `deepnsm` / `lance-graph` | MED | **Probe queued** | prune-without-recall-loss gate |
| D-EL-6 | Energy axis / particle↔wave | `lance-graph` MailboxSoA | MED | **Blocked** | gated on Mailbox-SoA map |
| D-EL-COCA | Superposition 2/3 pruning (cluster-identity layer) | `deepnsm` | HIGH | **Design** | I-VSA-IDENTITIES design-gate |

---

## singleton-to-snapshot-nudge-v1 — every shared-mutable singleton → per-owner MailboxSoA + Arc-swap snapshot (7 deliverables; codebooks left as-is)

Plan path: `.claude/plans/singleton-to-snapshot-nudge-v1.md`. Companions: `bindspace-singleton-to-mailbox-soa-v1` (BindSpace dissolution), `cycle-coherent-soa-snapshot-v1` (snapshot mechanism). Debt: TD-UNBUNDLE-FROM-1.

| D-id | Title | Crate(s) / repo | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|
| D-SNGL-1 | Workspace-wide singleton census (codebook vs shared-mutable) | docs/architecture | LOW | **Queued** | audit only; gates classification |
| D-SNGL-2 | Classification gate — "mutated-after-init?" decision procedure | docs/architecture | LOW | **Queued** | gates on D-SNGL-1 |
| D-SNGL-3 | `AttentionMatrix.gestalt` correctness (raw-sum+count or rebuild) | `lance-graph-planner::cache::kv_bundle` | MED | **In progress** | `unbundle_from` deprecated this session (branch `claude/stoic-turing-M0Eiq`); full fix pending |
| D-SNGL-4 | `ndarray/crates/burn` ATTENTION_CACHE / LINEAR_CACHE audit | `ndarray` | LOW | **Queued** | classify JIT-cache vs runtime-belief |
| D-SNGL-5 | `SnapshotProvider` adoption checklist per nudged crate | workspace | LOW | **Queued** | gates on D-SOA-SNAP-1/2 |
| D-SNGL-6 | No-cross-cycle-lag falsification per nudged crate | workspace | MED | **Queued** | reuses D-SOA-SNAP-5 shape |
| D-SNGL-7 | Board hygiene + E-SINGLETON-IS-CODEBOOK-OR-SOA | `.claude/board` | LOW | **In progress** | this entry + INTEGRATION_PLANS prepend |

---

## cesium-osm-substrate-v1 — OpenStreetMap as 6th Cesium ingest source class (7 deliverables; substrate-reuse with splat-native)

Plan path: `.claude/plans/cesium-osm-substrate-v1.md`. Parent: `3DGS-ArcGIS-Cesium-ingestion-plan.md` (structural). Sibling: `splat-native-ultrasound-v1.md` (Gaussian3D carrier reuse). OGAR coordination 2026-06-05 locked Q1/Q2/Q3 rulings. OGAR-side docs PR (DOMAIN-INSTANCES §2.6 + RDF-OWL-ALIGNMENT §10 Phase 2c) queued behind this PR.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Sprint | Status | PR / Evidence |
|---|---|---|---|---|---|---|---|
| D-OSM-1 | `crates/cesium/src/osm_pbf.rs` stub (mirrors `arcgis_pbf.rs` shape; OsmNode/OsmWay/OsmRelation/OsmPbfBlock + OSM-XYZ → TMS Y-flip helper; no osmpbf dep yet) | `ndarray` | 400 | LOW | P1 sprint 1 | **Queued** | foundation; gates nothing upstream |
| D-OSM-2 | osmpbf v0.4 consumer + Arrow RecordBatch emitter → Lance datasets `osm_nodes` / `osm_ways` / `osm_relations` (tags as Q1 v1 fallback `List<Struct<key,value>>`; qk_tms_path per Q2) | `lance-graph` | 600 | MED | P1 sprint 1-2 | **Queued** | gates on D-OSM-1 |
| D-OSM-3 | OSM tag → SPO triple lift (`(Way#123, ogar:hasTag, "building=yes")`); **OGAR-crossing contract** that `ogar-from-osm-pbf` Phase 2c consumes | `lance-graph-ontology` | 200 | LOW | P2 sprint 3 | **Queued** | gates on D-OSM-2 + OGAR readiness signal |
| D-OSM-4 | `ndarray::simd::dem::batched_sample_height` W1c primitive (bilinear interp; all three backends AVX-512/NEON/scalar) | `ndarray` | 300 | MED | P2 sprint 3 | **Queued** | foundation; sibling to D-SPLAT-2 |
| D-OSM-5 | Geospatial splat-fit: OSM footprint × DEM → extruded `Gaussian3D` batch (consumes D-SPLAT-1 carrier + D-SPLAT-3 SoA verbatim — substrate-reuse payoff) | new `crates/splat-fit-geo` OR `splat-fit` `geo` feature | 800 | MED-HIGH | P3 sprint 4-5 | **Queued** | gates on D-OSM-1 + D-OSM-2 + D-OSM-4 + D-SPLAT-1 + D-SPLAT-3 |
| D-OSM-6 | `cesium-3dtiles-writer` crate — b3dm/cmpt/tileset.json emitter (**the genuine Rust gap; first-of-its-kind**); MVP scope, gltf-crate-based | `ndarray` (new `crates/cesium-3dtiles-writer` or `writer` feature on existing `cesium` crate) | 500 | HIGH | P3 sprint 4-5 | **Queued** | gates on D-OSM-5 + D-SPLAT-3 |
| D-OSM-7 | Nominatim sidecar HTTP adapter (UX-edge optional; geocoding/reverse-geocoding via reqwest); response → D-OSM-2 primary path | `lance-graph` or new `crates/nominatim-client` | 150 | LOW | P4 sprint 6+ (optional; ship on UX-edge demand only) | **Queued** | independent path |

---

## splat-native-ultrasound-v1 — CPU-only Gaussian-splat ultrasound SaMD (14 deliverables across ndarray/lance-graph/MedCare-rs/OGAR + new standalone crates)

Plan path: `.claude/plans/splat-native-ultrasound-v1.md`. Companions: ndarray `.claude/plans/splat-native-ultrasound-simd-substrate-v1.md`; OGAR `docs/SPLAT-NATIVE-CUSTOMER.md`; MedCare-rs `.claude/handovers/2026-06-05-splat-native-medcare-hipaa-wire.md`. Customer of OGAR PR #30 §6 FMA bones-rendering litmus + ADR-022 SaMD audit-controls evidence base.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Sprint | Status | PR / Evidence |
|---|---|---|---|---|---|---|---|
| D-SPLAT-1 | `Gaussian3D` carrier (`mu`/`sigma_packed`/`amplitude`/`opacity`/`sh[16]`/`frame_idx`/`class_id`; 80 B/row) | `lance-graph-contract::splat` | 120 | LOW | P1 sprint 1-2 | **Queued** | gates on `MailboxSoAHeader` (D-MBX-10) or own feature flag |
| D-SPLAT-2 | `ndarray::simd::splat` batch ops — `batched_cholesky_3x3` / `batched_mahalanobis` / `batched_opacity_blend` / `batched_sh_eval_l3` / `batched_se3_transform`; all three backends (AVX-512/NEON/scalar) | `ndarray::src/simd_splat.rs` | 600 | MED | P1 sprint 1-2 | **Queued** | foundation; none |
| D-SPLAT-3 | `SplatBatch<N>` SoA carrier (per-column slices for SIMD sweep; inherits MailboxSoAHeader versioning) | `lance-graph-contract::splat` | 150 | LOW | P1 sprint 1-2 | **Queued** | gates on D-SPLAT-1 |
| D-SPLAT-4 | SH-aware palette extension in `crates/bgz17` (256×256×2B compose table; SH-basis-id per centroid) | `bgz17::sh_palette` | 250 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-1 |
| D-SPLAT-5 | Splat-to-splat registration math — Σ-sandwich Mahalanobis ICP + SE(3) Levenberg-Marquardt | `lance-graph::splat::registration` | 400 | HIGH | P4 sprint 6-7 | **Queued** | gates on D-SPLAT-2 + D-SPLAT-3 |
| D-SPLAT-6 | `crates/splat-fit` engine — RF/IQ → beamformed → local-maxima → PSF estimate → SH projection → emit Gaussian3D batch | `crates/splat-fit` (new standalone, 0-dep, ndarray-hpc feature) | 1500 | HIGH | P2 sprint 3 | **Queued** | gates on D-SPLAT-1 + D-SPLAT-2 + OQ-SPLAT-3 |
| D-SPLAT-7 | Splat actors — `SplatFitActor`/`PoseAccumulatorActor`/`RegistrationActor`, each owns one `MailboxSoA<Gaussian3D>`; consumes bardioc #17 Rubicon kanban verbatim | `crates/splat-actors` (or `ractor_actors`) | 500 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-3 + D-SPLAT-6 + bardioc #17 (shipped) |
| D-SPLAT-8 | FMA atlas hydrator — TTL → `fma_class.lance` + `fma_relation.lance` + `fma_atlas_splat.lance` (~150M Gaussians full body) | `lance-graph-ontology` + `crates/fma-hydrator` | 800 | HIGH | P4 sprint 7-8 | **Queued** | gates on OGAR PR #30 Phase 8 + D-SPLAT-3 + ndarray PR #189 (shipped) |
| D-SPLAT-9 | `fma_blueprint::style_recipe` D-Atom catalogue (AnatomicalRegion, OrganSystem, Innervation, Vasculature, Joint, Muscle, Bone, OrganParenchyma, Tract); mirrors PR #433 Odoo pattern | `lance-graph-ontology::fma_blueprint` | 400 | LOW | P4 sprint 7-8 | **Queued** | gates on D-SPLAT-8 |
| D-SPLAT-10 | `memory.ultrasound_frame.lance` + `memory.ultrasound_splat.lance` datasets via `soa_mapping.rs`; new `SensitivityReason::UltrasoundRawPHI`/`UltrasoundAnonymized` variants in `column_mask_bridge` | MedCare-rs `crates/medcare-analytics` | 250 | MED | P5 sprint 9-10 | **Queued** | gates on D-SPLAT-3 + MedCare PR #162 (shipped) |
| D-SPLAT-11 | `commit_event` audit chain for splat ingest via `LanceMembrane::commit_event` (callcenter PR #467, sole-writer membrane); `KnowableFromStore::register("ogit-medcare/ultrasound_ingest", Some(ddl_hint))` | MedCare-rs `crates/medcare-analytics` | 100 | LOW | P5 sprint 9-10 | **Queued** | gates on D-SPLAT-10 + PR #467 (shipped) + OGAR #25/#31 (shipped) |
| D-SPLAT-12 | AR splat renderer — HoloLens OpenXR (clinical AR target) + Cesium ion + Three.js (browser fallback) + headless PNG (regression); CPU does math, GPU only paints | `crates/splat-render` (new) | 1200 | HIGH | P6 sprint 11-13 | **Queued** | gates on D-SPLAT-2 + D-SPLAT-3 + D-SPLAT-5 |
| D-SPLAT-13 | IMU/POSE 4D accumulator — VIO against splat features at IMU rate (~200 Hz); splat-corrected pose at frame rate (~30 Hz); Planning-column readiness at t = −550ms | `splat-actors::PoseAccumulatorActor` | 200 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-7 |
| D-SPLAT-14 | SaMD documentation track — research-tool → clinical-study → Class IIa (IEC 62366 / IEC 80001 / ISO 14971 / MDR Annex VIII Rule 11). ADR-022 firewall IS the audit-controls evidence base | `q2`/`quarto` or `docs/` | 600 | LOW | P7 sprint 14+ (parallel through P4-P6) | **Queued** | gates on none architecturally; v1/v2/v3 phased |

---

# Status Board — Cross-Deliverable View

> Deliverable-level status across all active integration plans.
> **Status** and **PR / Evidence** columns are the only mutable
> fields — title, plan-version, and scope are immutable.
>
> For plan-level status see `INTEGRATION_PLANS.md`.
> For per-PR decision history see `PR_ARC_INVENTORY.md`.
> For current contract inventory see `LATEST_STATE.md`.

---

## D-HELIX-1 — `crates/helix` golden-spiral Place/Residue codec (zero-dep + optional ndarray-hpc)

**Status:** Shipped (branch `claude/gallant-rubin-Y9pQd`; **61 unit + 6 doctests green** on the default zero-dep build AND under `--features ndarray-hpc`; clippy -D warnings + fmt clean). New standalone crate (empty `[workspace]`, root `exclude`) realising the user's `KNOWLEDGE.md`: `HemispherePoint` (√u equal-area placement) → `CurveRuler` (stride-4-over-17) → `Similarity` (Fisher-Z/arctanh) → `RollingFloor` (256-palette; occupancy-drift + version stamp) → `ResidueEdge` (3-byte endpoint pair) + `DistanceLut` (metric-safe 256×256 L1; `distance_adaptive` vs non-metric `distance_heuristic`) + `prove()` (2-D discrepancy companion to `jc::weyl`). Optional `ndarray-hpc` = batch Fisher-Z via `simd_ln_f32`. ~80% clean-room overlap with CERTIFIED primitives (E-HELIX-OVERLAP / TD-HELIX-OVERLAP-1); consolidation path in `KNOWLEDGE.md`. Process: autoattended — 5 research agents + 4 parallel Sonnet leaf workers + central consolidation. Next (owed): fidelity-vs-ground-truth probe (naive-u8 floor gate ≥0.9980 Pearson, CONJECTURE). **Update (post-#460):** ndarray is now a MANDATORY non-optional **git** dep (codex P2 + directive "ndarray is mandatory for lance-graph"); `simd.rs` always uses `ndarray::simd`; `ndarray-hpc` feature removed. 63 unit + 6 doctests green; clippy/fmt clean. See E-HELIX-NDARRAY-MANDATORY.

## D-A3 — I4x32/I4x64 signed-i4 CAM codec (carrier `pack`/`unpack` + the 256-bit wide carrier)

**Status:** Shipped (branch `claude/jolly-cori-clnf9`; contract lib **562 green**, offline). `I4x32::pack`/`unpack` (two's-complement signed-i4 nibble; even→low/odd→high; saturate `[−8,7]`; sign-agnostic) + new `I4x64` (256-bit / 64 signed dims) + private `sext4`. The carrier is a deterministic 32×/64× **CAM address** + sparse-intensity "smell" — NOT a similarity vector (no vector search, no float; the `{instance,reference}` dual REJECTED, "64" = 64 poles). 33 atoms → dims 0..32. Resolved the 3 stale BLOCKED notes. Plan `.claude/plans/a3-carrier-v1.md` (5-research + 3-brutal sandwich). Next: A4 (CAM-address resolver + `is_signed` + `AtomLane`/`LaneMask` newtypes).

## D-EW64-2 — EpisodicEdges64 MRU promote (Hebbian hot-tier "stronger immediate edges")

| D-id | deliverable | status | PR / evidence |
|---|---|---|---|
| D-EW64-2 | `EpisodicEdges64::{promote, strongest}` — MRU slot-order = strength; fire→slot 0, evict coldest (`E-EW64-STRENGTH-IS-CE64-PLASTICITY`) | In PR (claude/jolly-cori-clnf9) | contract lib 533 green (+5); default clippy clean |

## Status Legend

| Status | Meaning |
|---|---|
| **Shipped** | Merged to main. PR column cites the merge commit. |
| **In PR** | PR open, under review. Not yet merged. |
| **In progress** | Active branch, code in flight, not yet PR. |
| **Queued** | Next up; spec is clear; work not started. |
| **Backlog** | Future; still in scope but not yet queued for a phase. |
| **Deferred** | Explicitly parked. Rationale recorded. Will be revisited. |
| **Abandoned** | Removed from scope. Rationale recorded. Will not be revisited. |

Rules:
- New rows APPEND (at the bottom of the relevant section).
- Status field is the ONLY field that gets edited in place.
- When a deliverable ships, record the PR number — never delete the
  row.
- When a deliverable is superseded by a different design, keep the
  row with Status = Abandoned and cite the replacement.

---

## normalized-entity-holy-grail-v1 — typed unified normalization + Op chain

Stage 1 contract surface scaffold. Typed consumer pipeline grammar that
unifies OGIT/OWL/DOLCE/Odoo inheritance + cognitive shader + JIT +
MailboxSoA into one surface. Plan path:
`.claude/plans/normalized-entity-holy-grail-v1.md`.

### Stage 1 deliverables (D-NEH-1a..g)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| **D-NEH-1a** | `cognition::{NormalizedEntity, stages, Op, OpKind, MailboxRow, Output}` typed surface | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1b** | `transaction::{Interactive, Bulk, Periodisch, Context, OgitCtx/OwlCtx/DolceCtx/FibuCtx}` context shapes | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1c** | 5-verb advancement methods on `NormalizedEntity<S>` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1d** | `CascadeKind` + `TraversalMode` + `CascadeWalker` trait | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1e** | Compile-fail tests + 7 positive typestate tests | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1f** | Crate doc + example chain + `docs/COGNITION_HOLY_GRAIL.md` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1g** | Board hygiene (AGENT_LOG + STATUS_BOARD) | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |

### Stage 2..7 deliverables (future plans)

| D-id | Title | Status |
|---|---|---|
| D-NEH-2a..z | ~50 Op kernel bodies + shader dispatch wiring | **Backlog** |
| D-NEH-3a..c | Consumer DSL macros (medcare/woa/smb) | **Backlog** |
| D-NEH-4a..b | Stream + GenServer integration | **Backlog** |
| D-NEH-5 | Jahresabrechnung kernel + fiscal-close JIT | **Backlog** |
| D-NEH-6 | palantir-foundry parity audit | **Backlog** |
| D-NEH-7 | elixir-OTP parity audit | **Backlog** |

---

## codec-sweep-via-lab-infra-v1 — JIT-first codec sweep

Active integration plan. 7 Phase 0 deliverables (D0.1–D0.7) + Phases
1–5 queued. One upfront Wire-surface rebuild; every candidate
afterwards is a JIT kernel, not a rebuild. Plan path:
`.claude/plans/codec-sweep-via-lab-infra-v1.md`.

### CI Gate — JC Substrate Proof

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| CI-JC | `.github/workflows/jc-proof.yml` — runs prove_it on every PR touching `crates/jc/` or `cam.rs` | **In PR** | 5-min timeout, exits 0 = substrate sound |

### Phase 0 — API hardening (partial in PR #225; remainder queued)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0.1 | Extend `WireCalibrate` + `WireTensorView` (64-byte-aligned decode, object-oriented methods) | **Shipped** | #227 — 55/55 tests passing |
| D0.2 | `WireTokenAgreement` endpoint stub — I11 cert gate (Phase 0 surface, Phase 2 harness) | **In PR** | branch — `WireTokenAgreement` + `WireTokenAgreementResult` + `WireBaseline` DTOs + 3 round-trip tests. Stub handler returns `stub:true` / `backend:"stub"` until D2.1–D2.3 wire real decode-and-compare. |
| D0.3 | `WireSweep` streaming endpoint + Lance append stub | **In PR** | branch — `WireSweepGrid` + `cardinality()` + `enumerate()` → `Vec<WireCodecParams>` + `WireMeasure` enum + `WireSweepRequest` / `WireSweepResult` / `WireSweepResponse` DTOs + 5 tests. Streaming handler + Lance writer defer to Phase 3 D3.1. |
| D0.4 | Surface freeze (commit + rebuild) | **Ready** | D0.1–D0.7 all Shipped / In PR; freeze fires on merge of this PR. |
| D0.5 | `auto_detect.rs` — `ModelFingerprint` from `config.json` | **In PR** | branch — `auto_detect::{detect, ModelFingerprint, DetectError}` + HF config.json parser + per-architecture lane/distance heuristics (llama/qwen3/bert/modernbert/xlm-roberta/generic) + 8 tests. CODING_PRACTICES gap 1 remediated. |
| D0.6 | `CodecParamsBuilder` fluent API | **Shipped** | #225 — `contract::cam` +290 LOC of codec-params types, 14 tests (CODING_PRACTICES gap 3) |
| D0.7 | Precision-ladder validation (OPQ↔BF16x32, Hadamard pow2, overfit guard) | **Shipped** | #225 — `CodecParamsError` at `.build()` BEFORE JIT compile |

### Phase 1 — JIT codec kernels

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D1.1 | `CodecKernelCache` — structural cache layer (generic over handle) | **In PR** | branch — `CodecKernelCache<H>` + `StubKernel` + `get_or_compile` / `try_get_or_compile` with RwLock concurrent-safe double-check + compile/hit/ratio counters + 9 tests. Scaffold ships NOW; D1.1b Cranelift IR emission follows. |
| D1.1b | Adapter: `CodecKernelEngine` wrapping `ndarray::hpc::jitson_cranelift::JitEngine` with two-phase BUILD/RUN lifecycle (Arc-freeze). CodecParams → CodecScanParams adapter + codec-specific IR emission in jitson_cranelift/scan_jit analog | **Queued** | target ~250 LOC; `JitEngine` already ships (`/home/user/ndarray/src/hpc/jitson_cranelift/engine.rs`); the work is the CodecParams adapter + codec-specific JITSON template |
| D1.2 | Rotation primitives: Identity / Hadamard / OPQ as `RotationKernel` impls | **In PR** | branch — `RotationKernel` trait (Send+Sync+Debug, object-safe) + `IdentityRotation` (no-op) + `HadamardRotation` (real Sylvester butterfly, O(N log N) in-place, norm²-scaling verified) + `OpqRotationStub` (matrix-blob-id placeholder for D1.1b) + `build(&Rotation, dim)` factory + `RotationError` typed errors + 15 tests. Hadamard stays at Tier-3 F32x16 (add/sub, not matmul → no AMX benefit per Rule C). |
| D1.3 | Residual PQ via decode-kernel composition | **In PR** | branch — `DecodeKernel` trait (Send+Sync+Debug, object-safe, encode/decode/signature/bytes_per_row/dim/backend) + `StubDecodeKernel` (byte-exact round-trip for testing) + `ResidualComposer` (base + residual with subtract/add; nests recursively for depth >1) + `DecodeError` typed errors + 9 tests. Scope clarified: hydration/calibration path, NOT cascade inference (cascade uses `p64_bridge::CognitiveShader` per `cognitive-shader-architecture.md` line 582). |

### Phase 2 — Token-agreement harness (I11 cert gate) — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2.1 | Token-agreement harness scaffold (reference model stub + top-k comparator + stub result) | **In PR** | branch — `ReferenceModel::{load, stub}` + `TokenAgreementError` + `TopKAgreement::{compare, top1_rate, top5_rate, meets_cert_gate, aggregate}` + `TokenAgreementHarness::{measure_stub, measure_full}` + 13 tests. Real safetensors load + decode loop defer to D2.2. |
| D2.2 | Decode-and-compare loop (top-k, per-layer MSE) | **Queued** | target ~220 LOC |
| D2.3 | Handler wiring for `/v1/shader/token-agreement` | **In PR** | branch — `token_agreement_handler` routes `WireTokenAgreement` → TryFrom(CodecParams) at ingress (precision-ladder + overfit guard fire here) → `ReferenceModel::load` or stub fallback on nonexistent paths → `TokenAgreementHarness::measure_stub()` → `WireTokenAgreementResult { stub:true }`. Route added: `POST /v1/shader/token-agreement`. Phase 0 Wire + Phase 2 harness now round-trip end-to-end. |

### Phase 3 — Sweep driver + Lance logger — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D3.1 | Server-side sweep handler + Lance fragment append | **In PR** | branch — `sweep_handler` batch mode: enumerates `WireSweepGrid::enumerate()`, validates each via TryFrom(CodecParams) at ingress, returns `WireSweepResponse { results: [WireSweepResult { kernel_hash, stub:true }], cardinality, elapsed_ms }`. SSE streaming + real calibrate/token-agreement per point deferred to D3.1b. Route: `POST /v1/shader/sweep`. |
| D3.2 | Client-side driver + config files | **In PR** | branch — 3 starter YAML configs (`configs/codec/{00_pr220_baseline, 10_wider_codebook, 12_hadamard_pre_rotation}.yaml`), `scripts/codec_sweep.sh` curl wrapper, `configs/codec/README.md`, YAML-shape spec-drift guard test. 118/118 tests pass. |

### Phase 4 — Frontier analysis — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D4.1 | DataFusion SQL over `sweep_results` Lance | **Queued** | target ~80 LOC |
| D4.2 | Pareto frontier notebook | **Queued** | target ~120 LOC |

### Phase 5 — Graduation — Fires per-candidate

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D5  | Graduation to canonical `OrchestrationBridge` (per winner) | **Queued** | target ~120 LOC per graduation; gate: ICC ≥ 0.99 held-out + token-agreement top1 ≥ 0.99 |

---

## elegant-herding-rocket-v1 — Phase-structured

Active integration plan, 12 deliverables D0 + D2–D11 (D1 dropped
early — CausalityFlow extension deferred). Plan path:
`.claude/plans/elegant-herding-rocket-v1.md`.

### Phase 1 — Shipped (PR #210, merged 2026-04-19)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0  | grammar-landscape.md + linguistic-epiphanies + fractal-codec knowledge docs | **Shipped** | #210 — 3 docs, 1151 LOC |
| D4  | ContextChain reasoning ops (coherence / replay / disambiguate / WeightingKernel) | **Shipped** | #210 — 396 LOC, 8 tests |
| D6  | Role-key catalogue with contiguous `[start:stop]` slice addressing | **Shipped** | #210 — 404 LOC, 7 tests |

### Phase 2 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2  | DeepNSM emits `FailureTicket` on low coverage (wiring step 4) | **Queued** | — |
| D3  | Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` | **Queued** | — |
| D5  | Markov ±5 bundler + Trajectory + content_fp (wiring steps 1-3) | **Shipped** | PR #243 — `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). 63 deepnsm tests pass. |
| D7  | Thinking styles + free-energy + RoleKey-as-operator | **Shipped** | PR #243 — `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery_margin (295 LOC added, 14 tests). 175 contract tests pass. |

### Phase 3 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D8  | Story-context bridge: AriGraph commit + global_context + contradiction (wiring steps 5-6) | **Queued** | — |
| D10 | Forward-validation harness (Animal Farm: chapter-10 > chapter-1 accuracy = AGI test) | **Queued** | — |

### Phase 4 — Backlog

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D9  | ONNX story-arc export + ArcPressure / ArcDerivative awareness hook | **Backlog** | — |
| D11 | Bundle-perturb emergence interface (transformer-free generative stack) | **Backlog** | — |

### Dropped / Deferred from the plan itself

| D-id | Title | Status | Notes |
|---|---|---|---|
| D1  | CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source) | **Deferred** | User decision; follow-up PR after Phase 2 |

---

## Infrastructure / governance (not in elegant-herding-rocket)

Workspace-level bootstrap work. Tracked here rather than PR_ARC
because it's process, not architecture.

| Item | Status | PR / Evidence |
|---|---|---|
| CLAUDE.md §Session Start — three mandatory reads | **Shipped** | #211 |
| CLAUDE.md §A2A Orchestration — two layers (runtime + session) | **Shipped** | #211 |
| CLAUDE.md §Model Policy — grindwork vs accumulation + never Haiku | **Shipped** | #211 |
| CLAUDE.md §GitHub Access Policy — zipball-for-reads | **Shipped** | #211 |
| `.claude/BOOT.md` session entry + prior-art links | **Shipped** | #211 |
| `.claude/agents/BOOT.md` orchestration spec (renamed from README) | **Shipped** | #211 |
| `.claude/agents/README.md` function inventory | **Shipped** | #211 |
| `.claude/board/LATEST_STATE.md` current-state snapshot | **Shipped** | #211 |
| `.claude/board/PR_ARC_INVENTORY.md` append-only decision arc | **Shipped** | #211 |
| `.claude/board/INTEGRATION_PLANS.md` versioned plan index | **Shipped** | #211 |
| `.claude/board/STATUS_BOARD.md` this file | **Shipped** | #211 |
| `.claude/settings.json` team-shared governance (ask/deny + hooks) | **Shipped** | #211 |
| `.claude/hooks/session-start.sh` + `post-compact.sh` | **Shipped** | #211 |
| `.claude/skills/cca2a/` pattern-explanation skill | **Shipped** | #211 |
| `.claude/plans/elegant-herding-rocket-v1.md` plan in workspace | **Shipped** | #211 |

## Infrastructure — queued

| Item | Status | Notes |
|---|---|---|
| `.claude/rules/` with `paths:` frontmatter | **Backlog** | Audit rec 2; replace / complement `READ BY:` headers with path-scoped loading |
| Skill `context: fork` + `agent:` field | **Backlog** | Audit rec 4; read-only isolation for search-only skill variants |
| Auto memory (`~/.claude/projects/<proj>/memory/`) | **Backlog** | Audit rec; unstructured addition to curated LATEST_STATE |

---

## Cross-cutting research threads (orthogonal to grammar work)

Separate research thread — not entangled with grammar/crystal/A2A.
Tracked here so it doesn't get lost.

| Item | Status | Notes |
|---|---|---|
| Named-Entity pre-pass (NER) — biggest OSINT blocker | **Deferred** | Dedicated PR after Phase 2 |
| FP_WORDS = 160 migration (currently 157) | **Deferred** | Needs coordinated ndarray change |
| Crystal4K 41:1 persistence compression | **Deferred** | ladybug-rs owns it; would port later |
| 200–500 YAML TEKAMOLO templates per language | **Deferred** | Training pipeline; future |
| Cross-linguistic active parsers (EN+FI+RU+TR) | **Deferred** | Role keys exist; parsers later |
| Fractal-descriptor leaf codec (MFDFA on Hadamard) | **Research** | `.claude/knowledge/fractal-codec-argmax-regime.md`. 30-min probe first. |
| UK Biobank cardiac MRI benchmark | **Research** | Downstream of fractal-codec probe |
| Chess vertical (ruci + lichess-bot integration) | **Deferred** | Capstone Tier 0, parallel stream |
| Wikidata ingest (1.2 B triples → 14.4 GB) | **Deferred** | `.claude/knowledge/wikidata-spo-nars-at-scale.md` |
| OSINT pipeline (spider + reader-lm + DeepNSM) | **Deferred** | `.claude/knowledge/osint-pipeline-openclaw.md` |
| Python/TypeScript grammar-stack convergence | **Deferred** | `.claude/knowledge/grammar-landscape.md` §7 |

---

## Prior-art audit (61 + 41 = 102 existing docs)

Before this session, the workspace accumulated 61 `.claude/*.md`
top-level docs + 41 `.claude/prompts/*.md` files across prior
sessions. They are indexed in `.claude/BOOT.md §Existing content`
and `CLAUDE.md §Prior art`, but their individual **status** (still
active / superseded / archival) has not been audited.

Status rows per bucket, not per file (102 rows would drown the
board — use filesystem + INTEGRATION_PLANS + PR_ARC for per-file
history):

| Bucket | Count | Status | Notes |
|---|---|---|---|
| `.claude/*.md` top-level calibration reports / handovers / audits / snapshots | 61 | **Indexed** | Pointed at from BOOT.md + CLAUDE.md. Per-file active/superseded status: **Backlog** (needs one-pass audit). |
| `.claude/prompts/*.md` scoped session / probe / handover prompts | 41 | **Indexed** | Pointed at from BOOT.md via `SCOPED_PROMPTS.md` index. Per-file status: **Backlog**. |
| `.claude/knowledge/*.md` structured knowledge | 12 | **Active** | Current; each has `READ BY:` header; used by Knowledge Activation triggers. |
| `.claude/agents/*.md` specialist + meta-agent cards | 24 | **Active** | Current; used by spawning + Knowledge Activation. |
| `.claude/hooks/*.sh` | 2 | **Active** | Wired via settings.json. |
| `.claude/skills/cca2a/*.md` | 3 | **Active** | Current. |
| `.claude/plans/*.md` integration plans | 1 (v1) | **Active** | Elegant herding rocket v1, Phase 1 shipped. |

**Backlog item — prior-art audit.** One-pass sweep across the
61+41 files. Per file: label as active / superseded / archival
with a one-line note. Deliverable = an `ARCHIVE_INDEX.md` that
splits the 102 into current vs historical, plus rename/move of
superseded files into an `archive/` subdirectory. Estimate ~200
LOC of meta work, ~2 hours of reading. **Not urgent**; useful
before the next major planning session.

---

## ADR 0001 — Archetype transcode + Lance/DataFusion stack + Persona 16^32

Three-decision architectural lock, accepted 2026-04-24. First ADR in the
workspace. Path: `.claude/adr/0001-archetype-transcode-stack.md`.

| Decision | Status | Mutability |
|---|---|---|
| **D1 — Archetype is TRANSCODED, not bridged** | **Accepted** | Immutable (unlocking requires new ADR) |
| **D2 — Stack lock** (Lance + DataFusion + Supabase-shape scheduler + Arrow temporal; Polars rejected; Ballista deferred to 1s-P99) | **Accepted** | Ballista threshold mutable; rest immutable |
| **D3 — Persona 16^32 is THE identity space** (56-bit PersonaSignature; atom vector BBB-banned) | **Accepted** | Immutable; shared-DTO unification OPEN for future ADRs |

**Follow-up items tracked** (per ADR implications):

| Item | Priority | Location |
|---|---|---|
| DU-2 clarification (rename "bridge" → "transcode") | P2 | `unified-integration-v1.md` DU-2 |
| First `lance-graph-archetype` skeleton crate | P1 (when deliverable lands) | — |
| Grok gRPC A2A expert adapter | P2 | `TECH_DEBT.md` 2026-04-24 |
| Enrichment-shape follow-up ADR | P2 | `TECH_DEBT.md` 2026-04-24 |
| Ballista threshold tuning (post-benchmark amend) | P3 | `TECH_DEBT.md` 2026-04-24 |

Merged via PR #249 (2026-04-24).

---

## callcenter-membrane-v1 — Supabase-shape over Lance + DataFusion

External callcenter membrane crate. BBB enforced by Arrow type system at
compile time. Plan: `.claude/plans/callcenter-membrane-v1.md`. **Validated
by ADR 0001 Decision 2** (DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`
pattern IS the Supabase-shape transcode approach).

### DM-0 / DM-1 — Shipped in this session

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-0 | `ExternalMembrane` trait + `CommitFilter` in `lance-graph-contract/src/external_membrane.rs` | **Shipped** | session 2026-04-22 — `pub mod external_membrane` added to contract lib.rs |
| DM-1 | `lance-graph-callcenter` crate skeleton: `Cargo.toml` (feature gates) + `src/lib.rs` (stub + UNKNOWN markers) | **Shipped** | session 2026-04-22 — added to workspace members |

### DM-2 through DM-9 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-2 | `LanceMembrane: ExternalMembrane` impl with `project()` + compile-time BBB leak test | **In progress** | Phase A shipped `9a8d6a0` — `LanceMembrane` struct + `project()` + `ingest()` + `subscribe()` stub. Phase B: full Lance append + version counter pending DM-4. |
| DM-3 | `CommitFilter` → DataFusion `Expr` translator (`[query]` feature) | **Queued** | — |
| DM-4 | `LanceVersionWatcher` — tails Lance version counter, emits Phoenix `postgres_changes` (`[realtime]`) | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-4a/b/c: `version_watcher.rs` (117 LOC, 4 tests), `lib.rs` `pub mod version_watcher`, `LanceMembrane::watcher` field + `project()` calls `bump()`, `subscribe()` returns `watch::Receiver<CognitiveEventRow>`. |
| DM-5 | `PhoenixServer` — minimal WS server, Phoenix channel subset (`[realtime]`) | **Queued** | Resolve UNKNOWN-2 (which consumers need Phoenix wire?) first |
| DM-6 | `DrainTask` — `steering_intent` Lance read → `UnifiedStep` → `OrchestrationBridge::route()` | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-6a/b scaffold: `drain.rs` (89 LOC, 2 tests), `lib.rs` `pub mod drain`, `Poll::Pending` until follow-up PR wires real drain loop. |
| DM-7 | `JwtMiddleware` + `ActorContext` → `LogicalPlan` RLS rewriter (`[auth]`) | **Queued** | Resolve UNKNOWN-3 (pgwire?) + UNKNOWN-4 (actor_id type) first |
| DM-8 | `PostgRestHandler` — query-string → DataFusion SQL → Lance scan → Arrow response (`[serve]`) | **Queued** | Confirm PostgREST compat needed (§ 8 stop point 4) before building |
| DM-9 | End-to-end test: shader fires → `LanceMembrane::project()` → Lance append → Phoenix subscriber receives event | **Queued** | Depends on DM-2 through DM-6 |

---

## grammar-foundry-followup-v1 — Wire stubs to existing tissue

Plan: `.claude/plans/grammar-foundry-followup-v1.md`. Session 2026-04-29.
Six explicit stubs in PRs #275-#283 + 1 keystone (LF-12 Pipeline DAG). 13 PRs total in 3 waves.

### Wave 1 — no deps (parallel)

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-S1 | LF-12 Pipeline DAG: `UnifiedStep.depends_on` + topological executor | **Queued** | Keystone. Unblocks F4, G2, G6 |
| PR-F1 | PolicyRewriter UDF wrap: `RedactionMode` executors (closes `policy.rs:122`) | **Queued** | Unblocks F2, F5 |
| PR-F3 | Audit log Lance-backed writer (closes `lib.rs:100`) | **Queued** | |
| PR-F6 | `dn_path.rs` real scent via CAM-PQ (closes `dn_path.rs:53`) | **Queued** | Risk: bgz-tensor dep |
| PR-G1 | Triangle bridge real Causality footprint (closes `triangle_bridge.rs:90,221`) | **Queued** | |
| PR-G3 | ContextChain real `Binary16K` fingerprint (closes `context_chain.rs:345`) | **Queued** | |
| PR-G4 | verb_table seed 10/12 families (closes empty `default_table()` rows) | **Queued** | |
| PR-G5 | AriGraph episodic unbundle/rebundle (per `integration-plan-grammar-crystal-arigraph.md`) | **Queued** | |

### Wave 2 — depends on Wave 1

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-F2 | RowEncryption + DifferentialPrivacy executors (closes `policy.rs:147,181`) | **Queued** | After F1; needs key-mgmt ADR |
| PR-F4 | PostgREST → DataFusion dispatch (closes `EchoHandler` stub) | **Queued** | After S1 |
| PR-F5 | `audit_from_plan()` helper (closes `orchestration.rs:202` `unimplemented!`) | **Queued** | After F1 |
| PR-G2 | Disambiguator wiring at parser boundary + FailureTicket emission | **Queued** | After S1 |

### Wave 3 — depends on Waves 1+2

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-G6 | Animal Farm harness real run (D10 from PR #243) | **Queued** | After G1+G2+G3; text licensing needed |

---

## unified-integration-v1 — PersonaHub × ONNX × Archetype × MM-CoT × RoleDB

Plan: `.claude/plans/unified-integration-v1.md`. Session 2026-04-23.

| D-id | Title | Status | Notes |
|---|---|---|---|
| DU-0 | PersonaHub 56-bit compression: `(atom_bitset: u32, palette_weight: u8, template_id: u16)` offline extraction from 370M HF parquet rows | **Queued** | Runs offline; no code deps. Output: `personas.bin` + `sigs_dedup.bin` + `templates/*.yaml` |
| DU-1 | ONNX persona classifier @ L4/L5 — 288-class `(ExternalRole × ThinkingStyle)` product prediction; `style_oracle: Option<&OnnxPersonaClassifier>` in Think struct | **Queued** | Needs ~10K labeled cycles from Lance internal_cold (DM-2 must ship first); replaces Chronos proposal |
| DU-2 | Archetype ECS bridge crate `lance-graph-archetype-bridge` — `ArchetypeWorld → Blackboard`, `ArchetypeTick → UnifiedStep`, `project() → DataFrame component` adapters | **Queued** | Needs DM-2 (ExternalMembrane impl) before adapter can be built |
| DU-3 | RoleDB DataFusion VSA UDFs: `unbind`, `bundle`, `hamming_dist`, `braid_at`, `top_k` — registers in DataFusion session | **Queued** | Fingerprint column type decision needed first (FixedSizeBinary vs FixedSizeList); see open question in plan § 5 |
| DU-4 | MM-CoT stage split: add `rationale_phase: bool` to `CognitiveEventRow`; surface `FacultyDescriptor.is_asymmetric()` in projected RecordBatch | **Shipped** (Phase A: 2026-04-23 `a05979e`; Phase B: 2026-04-24) | Phase A: field exists. Phase B: `set_faculty_context()` on `LanceMembrane` wires `rationale_phase` from `AtomicBool`; orchestration layer calls it with `FacultyDescriptor::is_asymmetric()` + stage. Column is live, not ghost. |
| DU-5 | Board hygiene: DU-0 through DU-4 registered; INTEGRATION_PLANS.md + LATEST_STATE.md updated | **Shipped** (2026-04-23, commit `a05979e`) | Plan corrections + precision-tier §18 + father-grandfather concept committed in follow-up. |

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge

Active plan, 7 deliverables (D-SPLAT-1..7) staged across 6 PRs of the
`gaussian-splat-cam-plane-workaround.md` doc-sequence. PR 1+2 in flight
on branch `claude/splat-osint-ingestion`.
Plan path: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-SPLAT-1 | `crates/lance-graph-contract/src/splat.rs` — `SplatChannel`, `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`, `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`, `ReasoningWitness64` + 10 unit tests | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-2 | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Σ-push-forward demo for OSINT 5-hop chain, side-by-side vs naive convolution | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-3 | `witness_to_splat()` deterministic conversion (PR 2 of doc-sequence) | **In PR** | branch `claude/phase-3b-witness-to-splat` |
| D-SPLAT-4 | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (PR 3 of doc-sequence) | **Queued** | — |
| D-SPLAT-5 | `PlanarSplatBundle4096` with local/short/medium/long bands (PR 4 of doc-sequence) | **Queued** | — |
| D-SPLAT-6 | Semantic-CAM-distance integration — survivor tile selection vs splatted pressure planes (PR 5 of doc-sequence) | **Queued** | — |
| D-SPLAT-7 | Replay fallback — exact 4096-cycle ThoughtCycleSoA replay slice when certificate insufficient (PR 6 of doc-sequence) | **Queued** | — |

Cross-ref: SPLAT-1 row in `ARCHITECTURE_ENTROPY_LEDGER.md` (Aspirational → Wired stage 1, entropy 4 → 2).

---


## causaledge64-mailbox-rename-soa-v1 — sprint-10 spec corpus + sprint-11 impl queue

Active integration plan. Specs shipped via PR #372 (merged 2026-05-14, governance-only).
Plan path: `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`.

### Sprint-10 — spec sprint (12 CCA2A workers + Opus meta) — Shipped

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1 | par-tile crate apex + Mailbox<T> + 3 backings + AttentionMask SoA + BindSpaceView | **Spec shipped** | #372 — `pr-ce64-mb-1-par-tile-crate.md` (W1) |
| D-CE64-MB-2 | CausalEdge64 v2 layout proposal + OQ-LAYOUT-1 BLOCKER finding | **Spec shipped** | #372 — `pr-ce64-mb-2-causaledge64-v2.md` (W2) |
| D-CE64-MB-2-regress | PAL8 / NARS regression tests (accessor-based, post-OQ-LAYOUT-1) | **Spec shipped** | #372 — `pr-ce64-mb-2-pal8-nars-regression.md` (W3) |
| D-CE64-MB-3 | BindSpace E/F/G/H column extension | **Spec shipped** | #372 — `pr-ce64-mb-3-bindspace-efgh.md` (W4) |
| D-CE64-MB-4 | AriGraph SPO-G + ghost edges + SpoWitnessChain + SCHEMA_VERSION 2→3 | **Spec shipped** | #372 — `pr-ce64-mb-4-arigraph-spo-g.md` (W5) |
| D-CE64-MB-5 | MailboxSoA<N> + AttentionMaskActor (single tick per cycle) | **Spec shipped** | #372 — `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) |
| D-CE64-MB-6 | SigmaTierRouter + banding + INT4-32D cold-start + Hebbian plasticity + KernelHandle cache + Σ9-10 escalation | **Spec shipped** | #372 — `pr-ce64-mb-6-sigma-tier-router.md` (W7) |
| D-CE64-MB-7 | bevy 0.14 cull plugin proof-PR | **Spec shipped** | #372 — `pr-ce64-mb-7-bevy-cull-plugin.md` (W9) |
| D-NDARRAY-MIRI-COMPLETE | Miri coverage ~760 → ~1550 | **Spec shipped** | #372 — `pr-ndarray-miri-complete.md` (W8) |
| D-SPRINT-10-DEPGRAPH | 8 PRs × 6 waves + parallel-landability + cross-spec consistency checks | **Spec shipped** | #372 — `sprint-10-pr-dep-graph.md` (W10) |
| D-SPRINT-10-TESTPLAN | Unified test plan + Miri growth target + proptest Miri runtime | **Spec shipped** | #372 — `sprint-10-test-plan.md` (W11) |
| D-SPRINT-10-EXECPLAN | Sprint-11 fleet definition + post-merge governance + worker prompt template | **Spec shipped** | #372 — `sprint-10-execution-plan.md` (W12) |
| D-SPRINT-10-META | Opus meta-review (CSI-1..6 + E-META-1..5 + sprint-11 gate decision) | **Shipped** | #372 — `.claude/board/sprint-log-10/meta-review.md` |

### Sprint-11 — implementation wave — Queued (blocked)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1-impl | par-tile crate impl (W1 → code) | **Queued** | blocked on OQ-5 user ratification (rayon vendor) |
| D-CE64-MB-2-impl | CausalEdge64 v2 layout impl (W2 → code) | **Queued** | blocked on CSI-1 user ratification (which Option A/B/C/D/E for bit reclaim) |
| D-CE64-MB-2-regress-impl | PAL8 / NARS regression test impl (W3 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-3-impl | BindSpace E/F/G/H impl (W4 → code) | **Queued** | blocked on D-CE64-MB-1-impl |
| D-CE64-MB-4-impl | AriGraph SPO-G + ghosts impl (W5 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-5-impl | MailboxSoA + AttentionMaskActor impl (W6 → code) | **Queued** | blocked on OQ-3 user ratification (plasticity granularity) + CSI-2 spec patch (g_slot_at_drop field) |
| D-CE64-MB-6-impl | SigmaTierRouter impl (W7 → code) | **Queued** | blocked on OQ-1 user ratification (Σ4-Σ5 banding) + CSI-3 spec patch (PR-J1 Wave 0.5 prerequisite) |
| D-CE64-MB-7-impl | bevy cull plugin impl (W9 → code) | **Queued** | blocked on D-CE64-MB-1-impl + CSI-4 spec patch (BindSpaceView::empty_static() ctor in W1) |
| D-NDARRAY-MIRI-COMPLETE-impl | Miri coverage impl (W8 → code) | **Queued** | independent; can spawn first |
| D-PR-J1-INT4-32D-ATOMS | INT4-32D codebook for SigmaTierRouter cold-start | **Queued** | new Wave 0.5 prerequisite; not in original W10 dep graph |
| D-CSI-2 | W6 CompartmentReport `g_slot_at_drop: u8` field patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-3 | W10 dep graph PR-J1 Wave 0.5 row patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-4 | W1 spec `BindSpaceView::empty_static()` + `from_arc()` constructors | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-5 | W1 spec move `SigmaTier` to `lance-graph-contract::orchestration` | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-6 | W11 test-count drift reconciliation | **Queued** | small spec edit; pre-sprint-11 |

### User-ratification gates (block sprint-11 spawn)

| Gate | Wave blocked | Resolution path |
|---|---|---|
| **CSI-1** — CausalEdge64 bit-reclaim Option (A/B/C/D/E) | Wave 2 (D-CE64-MB-2-impl) | User picks; meta-review recommends Option C-conservative (drop temporal + G-slot, allocate W-slot + lens) |
| **OQ-1** — Σ4-Σ5 banding (Tokio reflex vs InMem cycle-speed) | Wave 5 (D-CE64-MB-6-impl) | Default Tokio is safe-to-ship; ratification only PROMOTES |
| **OQ-3** — Plasticity update granularity (bit-counter per emission + NARS revise at AriGraph commit) | Wave 4 (D-CE64-MB-5-impl) | Tentative resolution recorded; user formal-acknowledge |
| **OQ-5** — Rayon vendor decision (std::thread::scope first vs vendored-rayon) | Wave 1 (D-CE64-MB-1-impl) | Tentative defer; user formal-acknowledge |

### Reunification track (sprint-12+)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-REUNIFY-1 | Acknowledge dual `CausalEdge64` types in TYPE_DUPLICATION_MAP + LATEST_STATE + EPIPHANIES | **Shipped** | this commit (post-merge #372 board-hygiene tail) |
| D-REUNIFY-2 | 8-channel → SPO transcoder spec at thinking-engine L3 commit boundary | **Backlog** | per Option R-3; sprint-12+ |
| D-REUNIFY-3 | `Think` carrier struct prototype unifying thinking-engine cascade + cognitive-shader-driver SoA | **Backlog** | per `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` sprint-12 |
| D-REUNIFY-4 | Splat op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) as methods on `Think` | **Backlog** | sprint-13+ |
| D-REUNIFY-5 | rayon work-stealing par_* method variants | **Backlog** | sprint-14+ |
| D-REUNIFY-6 | OWL DOLCE / OntologyFilter wiring into `emit_causal_edges_filtered` | **Backlog** | sprint-15+ |

---

## cognitive-substrate-convergence-v1 — i4 mantissa + gapless baton + active inference

Active integration plan. Authored 2026-05-15 (cross-session A2A discussion).
Plan path: `.claude/plans/cognitive-substrate-convergence-v1.md`.
Consolidates sprint-10 architectural decisions before context dilution.

### Phase A — Substrate primitives (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-1 | `causal-edge` crate v2 layout (signed mantissa, W-slot, lens, drop temporal) | **Shipped** | PR #383 merge `03bd175`; OQ-CSV-2 ratified to 6 bits (default) |
| D-CSV-2 | `QualiaI4_16D` type in `lance-graph-contract::qualia` + f32↔i4 migration helpers | **Shipped** | PR #384 merge `0751a8b`; OQ-CSV-1 ratified to Option α (canonical convergence-observable vocab; drop dim 16 "integration") |
| D-CSV-3 | InferenceType signed-mantissa expansion (absorbs PR-LL-1 Intervention/Counterfactual into canonical edge enum) | **Shipped** | PR #383 merge `03bd175`, paired with D-CSV-1 in same crate |
| D-CSV-4 | `CollapseGateEmission` wire format spec + impl per plan §8 | **Shipped** | PR #383 merge `03bd175`, contract crate (Vec instead of SmallVec to preserve zero-dep — TD-COLLAPSE-GATE-SMALLVEC-1) |

### Phase B — Storage & dispatch path (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-5a | QualiaColumn migration phase 5a — sibling `QualiaI4Column` add + double-write (no read-side change) | **Shipped** | PR #385 merge `6f58418`; OQ-CSV-4 ratified to sibling-cutover (default); 5b cutover follows in separate PR |
| D-CSV-5b | QualiaColumn migration phase 5b — flip readers to i4, drop f32 column, drop f32 push arg | **In PR (#390 W-G1)** | sprint-12 Wave G fleet; depends on D-CSV-5a (merged) + downstream reader audit |
| D-CSV-6a | `WitnessCorpus` partial (W-slot anchor + chain invariant; sorted by emission cycle, drop-oldest truncation) | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-7) |
| D-CSV-6b | `WitnessCorpus` full (CAM-PQ-indexed, unbounded, salience decay) | **In PR (#390 W-G2)** | sprint-12 Wave G fleet; depends on D-CSV-6a (merged) |
| D-CSV-7 | MailboxSoA integration: W-slot referencing + per-row plasticity accumulator + apply_edges | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-6a) |

### Phase C — Reasoning path (sprint-12)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-8 | MUL evaluation in integer SIMD: DK/TrustTexture/FlowState/GateDecision consume i4 qualia + signed mantissa | **Shipped** | PR #387 merge `e042c70` (scalar i4 path; AVX-512/NEON deferred → D-CSV-13/13b sprint-13 per TD-D-CSV-8-SIMD-1) |
| D-CSV-9 | 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary | **Shipped** | PR #387 merge `e042c70` (paired with D-CSV-8) |
| D-CSV-10 | Σ-tier Rubicon-resonance dispatch in `SigmaTierRouter`: ΔF + resonance threshold → Σ10 commit | **In PR (#388 W-F1)** | sprint-12 Wave F; sigma-tier-router crate present in workspace post-Wave G #390 cargo metadata (hand-tuned threshold per OQ-CSV-6; Jirak-derived → D-CSV-15 sprint-13+) |

### Phase D — Streaming infrastructure (sprint-12 productization)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-11a | Vertical streaming structs in ndarray: `QualiaStream` / `QualiaI4Row` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11b | Vertical streaming structs in ndarray: `InferenceStream` / `InferenceRow` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11c | Vertical streaming structs in ndarray: `SplatFieldStream` (+ `par_*` rayon variants deferred to sprint-14+ behind `parallel` feature) | **Shipped** | ndarray PR #147 merge `d867b1c`; `par_*` rayon variants deferred (Queued sprint-14+) |
| D-CSV-12 | Splat shader op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) — scalar standalone ops | **Shipped** | PR #388 merge `77f2d26` (W-F7 scalar; on-Think method migration → D-CSV-14 sprint-13) |

### Phase E — Sprint-12/13 new entries (NEW in v2 + sprint-13 preflight)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-13 | Batch i4 scalar MUL (paired with D-CSV-8 SIMD-readiness) | **Shipped** | PR #388 merge `77f2d26` (W-G3 batch i4 scalar) |
| D-CSV-13b | SIMD vectorization of D-CSV-8 i4 MUL evaluation (AVX-512 + NEON intrinsics) | **In PR (sprint-13/W-I1 salvage)** | branch `claude/sprint-13-w-i1-salvage`; AVX-512F+BW dispatch via `simd_caps()`; bench on Skylake-AVX512 host = 8.7× dk / 7.4× trust / 5.2× flow / 10.2× gate_disc / 3.1× mul_assess at batch 1024 — all SHIP gates met; 5 SIMD-vs-scalar parity tests over 10 sizes green |
| D-CSV-14 | On-Think method migration for D-CSV-12 splat ops (struct-method surface per L-20) | **Queued (PP-4 spec drafting)** | sprint-13; depends on D-CSV-11 streaming substrate (shipped via ndarray #147) |
| D-CSV-15 | Σ10 Jirak-derived threshold (TD-SIGMA-TIER-THRESHOLDS-1 resolution) | **In PR (#390 W-G4 Jirak threshold)** | sprint-12 Wave G partial; full VAMPE coupled-revival deferred sprint-13+ |
| D-CSV-16 | NEW sprint-13 entry | **Queued (PP-5 spec drafting)** | sprint-13 preflight |
| D-CSV-17 | NEW sprint-13 entry | **Queued (PP-3 spec drafting)** | sprint-13 preflight |

### Open-question gates (block specific D-CSV-* spawns)

| Gate | Blocks | Recommendation |
|---|---|---|
| **OQ-CSV-1** Qualia 16D per-dim assignment | D-CSV-2, D-CSV-5 | Ratify proposed §7.2 layout with `qualia-engineer` agent cross-check against `thinking-engine/src/qualia.rs` |
| **OQ-CSV-2** W-slot width 6 vs 8 bits | D-CSV-1 | Default 6 (= 64 active corpora); promote to 8 if multi-tenant SaaS demands |
| **OQ-CSV-4** QualiaColumn migration phasing | D-CSV-5 | Default sibling-column-then-cutover (lower risk; 1 extra PR worth it) |
| **OQ-CSV-6** Σ10 Rubicon threshold derivation | D-CSV-10 (sprint-12) | Hand-tuned acceptable for sprint-11/12 with TECH_DEBT note per `I-NOISE-FLOOR-JIRAK`; principled Jirak derivation deferred to VAMPE coupled-revival sprint-13+ |

### Cross-spec patches (one bundled prep PR pre-sprint-11) — **SHIPPED via PR #381 (merged 2026-05-16, commit `a7c0545`)**

| Spec | Patch | LOC | Status |
|---|---|---|---|
| `pr-ce64-mb-2-causaledge64-v2.md` (W2) | §3 bit layout → plan §6; OQ-LAYOUT-1 resolved; signed-mantissa rationale; G-slot API stripped from test plan + risk matrix (codex P1) | ~160 actual | **Shipped** |
| `pr-ce64-mb-2-pal8-nars-regression.md` (W3) | Tests parameterized on v2 layout; mantissa roundtrip + lens 4-state; v1-temporal=0 safe-migration fix + version-gate test (codex P1) | ~370 actual | **Shipped** |
| `pr-ce64-mb-3-bindspace-efgh.md` (W4) | QualiaColumn migration step (D-CSV-5) cross-ref | ~40 actual | **Shipped** |
| `pr-ce64-mb-4-arigraph-spo-g.md` (W5) | `SpoWitnessChain<32>` → `WitnessCorpus`; `W5-INV-CHAIN-ORDER` invariant; W-slot semantics | ~316 actual | **Shipped** |
| `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) | `g_slot_at_drop` field (CSI-2); spatial-temporal accumulator semantics | ~50 actual | **Shipped** |
| `pr-ce64-mb-6-sigma-tier-router.md` (W7) | Σ10 Rubicon-resonance threshold; integer-SIMD MUL path | ~120 actual | **Shipped** |
| `sprint-10-pr-dep-graph.md` (W10) | PR-J1-INT4-32D-ATOMS + CAM-PQ wiring elevated to Wave 3 hard dep | ~50 actual | **Shipped** |
| `sprint-10-test-plan.md` (W11) | Refresh test counts for v2; i4-roundtrip + signed-mantissa-product tests | ~87 actual | **Shipped** |

**Total spec-patch LOC:** ~1,200 actual across 5 commits (`9bd66d9`, `f730528`, `5253c79`, `e4d15a3`, `33509ab`) merged 2026-05-16 in PR #381. Original ~870 estimate undershot W3 (codex P1 fix added ~280 LOC) and W5 (full WitnessCorpus section added ~16 LOC over estimate). All 8 workers complete. Sprint-11 spawn now unblocked on the spec-patch dimension; remaining gates: OQ-CSV-1, OQ-CSV-2, OQ-CSV-4 user ratifications.

---

## rung-persona-orchestration-v1 — time-bound persona orchestration (checklist → meta-recipe → hot/cold/feedback anneal)

Active proposal. Authored 2026-05-26. Plan path:
`.claude/plans/rung-persona-orchestration-v1.md`. Sibling/time-bound
composition layer over `rung-mul-grounding-v1`. Grounds ladybug's
hot/cold/feedback loop onto our contract types + SoA floor
(restore-on-SoA, not port). Epiphany: `E-RIGID-RULES-OPEN-DOORS`.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-PERSONA-1 | escalation+epiphany loop = the checklist (`felt_parse` collapse-hint + `InnerCouncil`/`HdrResonance` split + `EpiphanyDetector`; green-flip = Epiphany/Wisdom ghost) — NOT a bespoke verifier | contract + planner | 160 | LOW | **In progress** | branch `claude/splat3d-cpu-simd-renderer-MAOO0` |
| D-PERSONA-2 | meta-recipe manifest (declarative child-spec, recipe-as-data, macro-evaluable) | contract | 150 | MED | **Queued** | — |
| D-PERSONA-3 | hot/cold/feedback wiring — anneal + `CrystalCodebook`→wisdom-marker cold path + Preload hydrate | planner + Lance | 240 | MED | **Queued** | — |
| D-PERSONA-4 | macro-eval harness (scenario→trace→discover→diagnose; suspect-bridge = blasgraph betweenness; 5 rubrics from D-RUNG-MUL) | planner + Lance | 280 | HIGH | **Queued** | — |
| D-PERSONA-5 | ractor outer-swarm runtime under `OrchestrationBridge` (batons as messages, async only at boundary) | planner | 200 | MED | **Queued** | — |
| D-PERSONA-6 | `odoo_scanner` + `OdooBridge` — harvest Odoo `l10n_de` → Finance-ns `MappingProposal`s; bind existing `TaxEngine`; GoBD by construction | ontology + contract + planner | 280 | MED | **Queued** | — |

---

## unified-soa-convergence-v1 — ONE LE SoA end-to-end across 9 consumers + version gate + Lance 6.0.1 stack + 4-phase Rubicon kanban

> **Plan P0 status:** SHIPPED in PR #434 (merged 2026-05-29). Deliverable rows below remain Queued; they ship in follow-up PRs per phase sequencing.

Plan path: `.claude/plans/unified-soa-convergence-v1.md`. Handover `.claude/handovers/2026-05-29-1825-soa-convergence-author-to-impl.md`. Review addendum `.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md`. Epiphany `E-SOA-IS-THE-ONLY` (+ §11.3/4/6 refinements).

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-A1 | migrated thoughtspace columns landed on `MailboxSoA<N>` (`edges`/`qualia`/`meta`/`entity_type`) | cognitive-shader-driver | 60 | LOW | **Shipped** | between #418 and #433 (verified `mailbox_soa.rs` 2026-05-29) |
| D-MBX-A2 | close BindSpace expressivity gaps in `MailboxSoA<N>` (`content_ref`, S/P/O role slices, temporal/expert fold per OQ-2) | cognitive-shader-driver + contract | 140 | MED | **Shipped (carrier)** | columns landed post-2026-06-13 via W1 `22f5120a` (temporal/expert/sigma) + W1b `707360dc` (dense content/topic/angle planes) + W1c + W4a shim, with accessors + parity + field-isolation tests in `mailbox_soa.rs`. OQ-1 resolved (dense planes hot, per reconciliation-doc supersession of the ≤6B-ref framing). S/P/O role slices = **NON-GAP** (roles are VSA-unbind vs `contract::grammar::role_keys`, not a per-row column — `I-VSA-IDENTITIES`). Residual: OQ-2 fold-vs-standalone (landed standalone; deferrable). See `E-DMBXA2-SHIPPED-RECONCILE`. |
| D-MBX-A3 | `witness_arc: [u32; W]` per-row column (the belief-state arc handle into AriGraph episodic Markov chain) | cognitive-shader-driver | 100 | MED | **Queued** | gates on D-MBX-A2 + OQ-11.2 |
| D-MBX-A4 | Staunen × Wisdom counterfactual plasticity spreader (Hebbian, hot-path-only, Planning-gated) | cognitive-shader-driver | 80 | LOW | **Queued — design** | gates on D-MBX-A3 + OQ-11.1 + `phase` field |
| D-MBX-A5 | SPO-W witness pointer dual-residency (SoA / kanban / mailbox index); SoA decides commit modality (chain pointer vs cold fact) | cognitive-shader-driver + AriGraph SPO-G | 150 | HIGH | **Queued** | gates on D-MBX-A3 + D-MBX-4 |
| D-MBX-A6 | `lance-graph-planner` DTO surface overhaul: DTOs as SoA-row-lenses; planner output = `KanbanMove`s; 5-phase feature-gated cutover (OQ-11.7) | lance-graph-planner + contract | 600 | HIGH | **Queued** | gates on D-MBX-10 + D-MBX-8 + OQ-11.7 |
| D-MBX-7 | `lance-graph` containers ≡ `MailboxSoA` layout ≡ `ndarray::simd_soa.rs`-aligned (1.4–4.2× SIMD payoff; hard prereq for SurrealDB transparent view) | lance-graph + ndarray | 300 | HIGH | **Queued** | gates on D-MBX-A2 + D-MBX-10 + D-MBX-11 + PR-NDARRAY-MIRI-COMPLETE |
| D-MBX-8 | Σ10 commit stamps **t = −550 ms** wall-clock (Libet anchor) in `SigmaTierRouter`; downstream ractor START fires | sigma-tier-router + shader-driver | 120 | MED | **Queued** | gates on D-MBX-A4 + D-MBX-A6 Phase 1 |
| D-MBX-9 | Rubicon kanban view in `surrealkv`-on-lance (4 columns: Planning · Cognitive work · Evaluation · Commit·Plan·Prune); ractor lifecycle hooks = kanban moves | surreal_container + ractor | 250 | HIGH | **Queued** | gates on D-MBX-7 + D-MBX-8 + surreal_container BLOCKED(B/C/D) resolved (OQ-11.6) + D-PERSONA-5 |
| D-MBX-9-IN | contract slice of D-MBX-9 IN-direction (`E-SUBSTRATE-IS-THE-SCHEDULER`): `scheduler::{DatasetVersion, VersionScheduler, NextPhaseScheduler}` — Lance `versions()` tick → next legal `KanbanMove`, zero-dep, read-only-over-view (propose-not-dispose) | lance-graph-contract | 130 | LOW | **Shipped (contract)** | 509 lib tests (+6); clippy pedantic-clean; CI-gated twin `D-MBX-9-IN-impl` (LanceVersionScheduler over `VersionedGraph::versions()`) named not written |
| D-H2H-1 | head2head superposition winner-select (item 4, Go infight-vs-Raumgewinn): `head2head::{Head2Head, WinnerCriterion, CompetitionOutcome}` — `select(&Blackboard)` arg-extremum over existing bids, select-not-duplicate | lance-graph-contract | 130 | LOW | **Shipped (contract)** | 516 lib tests (+7); clippy pedantic+nursery clean; parallel-mailbox executor = CI-gated consumer side |
| D-EW64-1 | `episodic_edges::{EpisodicEdges64, EdgeRef}` — AriGraph episodic edges (4x[4b family|12b local]); intra=inherited (~98.6%), cross=4-bit nibble->OGIT-class palette (~1.4%) | lance-graph-contract | 120 | LOW | **Shipped (contract)** | 527 lib tests; clippy clean; SoA columns = D-EW64-2 (CI-gated) |
| D-VIEW-1 | `view_angle::ViewAngle` — 4-bit view-schema selector; presence-bitmask-as-attention (inherited) | lance-graph-contract | 40 | LOW | **Shipped (contract)** | 527 lib tests; clippy clean |
| D-MBX-10 | SoA version byte at layout root (`MailboxSoAHeader`); refuse v(N>M) bytes on v(M) reader; field-isolation matrix tests on every column op (`I-LEGACY-API-FEATURE-GATED` discipline) | lance-graph-contract | 100 | HIGH | **Queued** | foundation — should land early in P2; gates on OQ-11.5 |
| D-MBX-11 | Lance bump (5 Cargo.toml) — **OBE: main jumped `=6.0.0 → =7.0.0`, not `=6.0.1`** | workspace Cargo.toml | 10 | LOW | **Abandoned (superseded by #445, 2026-06-14)** | done by PR #445 (lance/lance-linalg `=7.0.0`, lancedb `=0.30.0`, object_store 0.13.2); `=6.0.1` never existed on the lancedb path. Residual: TD-SURREALDB-KVLANCE-LANCE7 (fork still pins 6) |
| D-MBX-12 | 8-PR workspace-wide consumer alignment: 12.1 AriGraph · 12.2 Vsa16k audit · 12.4 lance-graph · 12.5 planner · 12.6 shader-driver · 12.7 callcenter · 12.8 ontology audit · 12.9 thinking-styles | per-crate | 800 | per-PR | **Queued (multi-PR)** | sequencing per OQ-11.8: 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8 |
| D-MBX-A6-P1 | contract slice of D-MBX-A6: `kanban::{KanbanColumn, KanbanMove}` + `soa_view::{MailboxSoaView, MailboxSoaOwner}` + `StepDomain::Kanban` — the planner⟷ractor⟷surreal seam, zero-dep, no parallel DTO family | lance-graph-contract | 340 | HIGH | **Shipped** | #437 (merged 9161bd7); + `class_id` N1 hook ride-along |
| D-MBX-A6-P2 | Rubicon lifecycle enforcement + exec-target tag: `KanbanColumn::{next_phases, can_transition_to, is_absorbing}` (the lifecycle DAG) + `MailboxSoaOwner::try_advance_phase` (checked, `RubiconTransitionError`) + `ExecTarget{Native,Jit,SurrealQl,Elixir}` on `KanbanMove` | lance-graph-contract | 120 | LOW | **In PR** | builds on P1; 489 lib tests (+4); downstream cargo-check clean; gates the ractor owner-impl + planner emit (P3) |
| D-MBX-A6-P3a | StyleStrategy: thinking-style -> cluster -> mechanism -> recipe_kernels Tactic selection (planning substrate; carries tau JIT addr) | lance-graph-planner | 130 | LOW | **In PR** | #439; first cut of A6-P3 consumer wiring; planner now consumes contract recipes/styles; deferred: i4-32D decode, Outcome->Candidate, tau->JIT, membrane commit |
| D-MBX-A6-P3b | output overhaul: `StrategyOutcome{reliability, intended_move: Option<KanbanMove>}` carrier on `PlanInput.outcome`; StyleStrategy retires the dead-store `_reliability`, SURFACES reliability + a bootstrap intended move (Planning→CognitiveWork, owner 0, warden-BOOTSTRAP-OK) — plan still pure | lance-graph-planner | 130 | LOW | **In progress** | additive Option field (6 in-crate literals); UNBLOCKED (no mint, not OQ-11.7); deferred: compose thread-out + contract-promote + owner-consume; E-STRATEGY-OUTCOME-CARRIER-1 |
| D-MBX-A6-P3c | owner-consume: `lance_graph_planner::owner_adapter` = the `Outcome → KanbanMove` bootstrap-rebind + ahead-cast adapter. `rebind_bootstrap` (mailbox 0/cycle 0 sentinel → live owner; refuses an already-owned move = no ownership theft) + `emit_bootstrap_intent` → `BatchWriter::cast(on_behalf = owner)`. Fire-and-forget (no ack/ledger/WAL/arbitration/callback); the move is the pre-write "parcel address", the lifecycle STEP stays post-write. Completes P3b's deferred `owner-consume`. | lance-graph-planner | 90 | LOW | **In PR** | 5 falsifiable probes (rebind 0→live anti-vacuity + no-theft + on-behalf cast + non-vacuous no-op silence); lance-free, builds without protoc. Persistence sink (drain→Lance 7 `mem_wal::WalAppender::append`) verified-but-gated (protoc missing + disk); knowledge doc `.claude/v3/knowledge/d-mbx-a6-owner-consume-and-persistence.md`; `E-KANBANMOVE-IS-THE-PARCEL-ADDRESS-STEP-IS-THE-DELIVERY-SCAN-1` |
| D-MBX-A6-P3d | persistence sink ORDERING CORE + durable-witness reshape + temporal layer-1 (the POST-write half). `lance_graph_planner::persist_sink`: two clock domains (async `persist_cast` no-owner-borrow → `DurableReceipt`; sync `apply_durable_step` no-await → `try_advance_phase`). Crash-durability: `DurableWitness{owner,cast_id,cycle,paired_move}` CO-LOCATED with the SoA payload in one generation via `DurableWrite::append(&witness,&payload)`; `scan_witnesses(from)` bounded replay seam returning `LandedWitness{coordinate,witness}`; `recover_and_apply(owner,landed,applied_through)` replays the pending tail in **durable-log order** (`DurableCoordinate::log_order`, NOT the resettable `cast_id`) with a durable **watermark** for cyclic-safe idempotence, returning `Recovered{applied,watermark}`; `StalePhase` = corruption above the watermark (sync path: safe-to-drop stale). `temporal::{LocalCausalRow, local_trajectories, local_trajectory_of}` = layer-1 CAUSAL deinterlacing (global interleaved log → per-owner local chain), composing with the existing layer-2 epistemic projection. Durability proof = `DurableCoordinate` (opaque `seq`, API-honest), never `LanceVersion`. | lance-graph-planner | 175 | LOW | **Merged** | #878 (merged; reshaped in place to the cycle/WAL model = P3e — `E-THE-DURABLE-UNIT-IS-THE-CYCLE-NOT-THE-CAST-ONE-WAL-WRITE-PER-SWEEP-1`; + the §2 sparse-delta storage ruling `E-COMPLETE-CYCLE-IS-PHYSICALLY-SPARSE-NOT-A-FULL-REWRITE-1`, RATIFIED/UNIMPLEMENTED); ordering/recovery CONTRACT probed (348 planner lib tests, clippy+fmt clean) — crash-durability NOT storage-proven (in-process fake, no real MemWAL/restart); review-hardened ×2 (Bugbot: cast_id-resets→durable-position order; Codex/CodeRabbit Critical: cyclic idempotence via durable watermark + negative control; cross-owner reject; concurrent-drain retryable receipt; bounded scan; contract-probe honesty); builds NO concrete `LanceShardSink`; generation-vs-per-cast seam (finding 5) surfaced for operator decision; `E-THE-PAIRED-MOVE-MUST-BE-DURABLE-CO-LOCATED-NOT-IN-MEMORY-ONLY-1` |
| D-MBX-A6-P4 | **Cycle loop-closure driver** (PLANNED): the seam that makes the merged `persist_sink` cycle/WAL bootstrap load-bearing at 64k. Closes `owners think over Vn → produce material updates emit sparse intents → planner collects/coalesces/seals (one WAL, Vn+1) + exposes the sealed paired-transition set → supervisor applies ONLY the sealed sparse transitions (represented owners advance one legal step; unrepresented owners byte-identical) → CognitiveWork runs the thought → owner_adapter casts next intent → loop`. **A DatasetVersion is global knowledge, NOT permission to advance every mailbox (sparse-cycle ruling).** Applied INLINE by the writer (no dataset re-read; NOT 64k async `drive_once`). Interim rule: ≤1 durable phase transition per owner per sealed cycle. Mints NO new types. Sub-deliverables P4a (drain+seal) / P4b (apply sealed sparse set) / P4c (CognitiveWork+cast round-trip) / P4d (wait-free emit) / P4e (recovery) / P4f (sparse routing scale 16k/64k, W2a-gated), probe-first. | lance-graph-supervisor | 400 | HIGH | **P4a–P4f Shipped (slice)** | `lance-graph-supervisor::cycle_driver` behind feature `cycle-driver` (optional one-way `lance-graph-planner` dep, verified acyclic; default build stays light). **P4a** `collect_casts`+`seal_cycle` (drain `BatchWriter` → `persist_cycle` → one WAL write / one `DatasetVersion` + sparse `SealedTransition` set). **P4b** `apply_sealed_transitions` (iterate ONLY the sealed sparse set; one legal `try_advance_phase`/represented owner; unrepresented byte-identical; interim ≤1/owner/cycle → `deferred`; unknown → `missing`; reads NO dataset). **P4c** `run_cognitive_work` (owners entering CognitiveWork run a pluggable thought seam → `owner_adapter::emit_bootstrap_intent` casts the next intent write-on-behalf → next cycle) **+ MUL-gate plug** `shade_owner`/`run_cognitive_work_gated[_over]` (the real MUL *gate* — `contract::mul::i4_eval::gate_decision_i4(qualia,mantissa)` → `KanbanColumn::advance_on_gate`, Flow→forward / Block→Prune-where-legal / Hold→rescheduled via `held_owners` — packaged as a bootstrap-sentinel `StrategyOutcome`; `mul_target` composed for the driver, mints no decision logic; **NOT the cognitive-shader-driver/MailboxSoA dispatch** — qualia/mantissa via a caller extractor bridging the deferred `MailboxSoaView::qualia()` seam, MailboxSoa contract UNCHANGED). **Review round (grain-of-salt, 2026-08-02):** retry-safe seal (`SealFailure` carries the byte-identical frozen cycle); restart-stable `stream_position` (= caller's durable `position_base` + CastId, `next_position_base` cursor — CastId alone was the P3d "cast_id is provenance only" trap); normal apply advances the per-owner recovery **watermark** with the phase (one rule with recovery — no replay/StalePhase after a crash); ≤1-move/owner enforced **pre-seal** (`HeldIntent`+`restage_held` — sealed set == applied set, recovery agrees; no seal-then-discard, no silent `moves.first()` truncation); mid-apply error returns the applied prefix `Err((partial,cause))`; Hold = reschedule not strand (`held_owners` + `_over` re-poll); `recover_fleet` partitions history once (O(history), not O(fleet×history)). **P4d** wait-free-at-the-cast-boundary (sequential pass; concurrent per-owner execution = the actor leg, honestly scoped) (an incomplete owner never blocks a completed one — no barrier). **P4e** `recover_fleet` (composes `persist_sink::recover_and_apply` per owner; per-owner watermark idempotence; partial-progress kept on error). **P4f** `CountingFleet` scale probe. `run_cycle` convenience. **19 lib tests** incl. the **64k/17 falsifier** (exactly 17 advance, 65 519 byte-identical, one WAL write, zero reads), failed-seal→byte-identical-retry, restart-stable positions across writer reconstruction, normal-apply-advances-watermark (crash → recovery replays NOTHING), held-move-lands-next-cycle + recovery-agrees, multi-move no-truncation, mid-apply prefix preservation, the **gate falsifier** (Flow casts + advances; Hold rescheduled and WOKEN on a later re-poll — the gate discriminates: three outputs for three inputs, absorbing-column no-successor), P4d unfinished-A-never-blocks-B (both represented), P4e idempotence + load-bearing-watermark negative control, P4f **O(dirty)-not-O(fleet)**; clippy+fmt clean. Mints NO domain types (reuses `SweepSlot`/`CycleFrame`/`KanbanMove`/`DatasetVersion`/`PersistError`/`StrategyOutcome`/`recover_and_apply`/`emit_bootstrap_intent`/`MailboxSoaOwner`). **Honesty ledger:** control-loop contract PROVEN (falsifiers over fakes) · actor-owned production wiring NOT proven (`MailboxFleet` HashMap = probe/registry fleet; bridging into `KanbanActor`-owned state open) · cognitive-shader-driver/SoA thought NOT proven (the gate is real, its inputs are extractor-fed) · durability FAKE (contract-probe `WalSink`, until `LanceShardSink`). Plan `.claude/plans/cycle-loop-closure-driver-v1.md`; `E-KANBANMOVE-IS-THE-PARCEL-ADDRESS-STEP-IS-THE-DELIVERY-SCAN-1` + `E-COMPLETE-CYCLE-IS-PHYSICALLY-SPARSE-NOT-A-FULL-REWRITE-1` + `E-D-MBX-SPINE-IS-STRAIGHT-TRACK-VERSION-IS-NOT-A-FLEET-STEP-SIGNAL-1` |
| D-MBX-A6-P3-M1 | `Tactic::requires() -> ThoughtMask` + `ThoughtField`/`ThoughtMask` (checklist-as-data keystone): 34 tactics declare their ThoughtCtx field-reads; `covered_by` = reliability-coverage gate | lance-graph-contract | 120 | LOW | **In PR** | #439; the panel-recalibrated keystone (extraction not construction); makes P1/P7/P11 derived; teeth-test asserts masks varied not stub |
| D-CLS-FM | `class_view`: FieldMask(u64 presence) + ClassView meta-DTO resolver trait + ClassProjection (the class flies ABOVE the SoA; labels resolved late from OGIT cache, zero in the bytes) — extends ObjectView, reuses class_id | lance-graph-contract | 270 | LOW | **Shipped** | #441 D-CLS contract foundation; OD-gates ratified; presence!=semantics (C2); N3 stable positions; 3 teeth-tests |
| D-CLS-RES | `class_resolver`: `RegistryClassView` impls `ClassView` over the live OntologyRegistry — the ontology-side 'parser' (class_id -> shape, DOLCE resolved LATE via classify_odoo from the cache URI, memoized over the O(n) registry scan) | lance-graph-ontology | 200 | LOW | **Shipped** | #441 D-CLS; makes the contract trait live; field-set supplied (D-CLS audit deferred); 4 teeth-tests |
| D-CLS-SIG | `class_signature`: deterministic structural-signature audit of curated OdooEntity consts (FNV-1a over kind+field-hist+method-hist+state-machine) -> shape-family group-by + `object_view()` derives the real ObjectView bit-basis (fills the D-CLS-RES placeholder) | lance-graph-ontology | 230 | LOW | **Shipped** | #441 D-CLS; the HONEST D-CLS-3 (group-by-on-structural-hash, NOT aerial-cluster vaporware, classes.md:43); 4 teeth-tests over real l1 data |
| D-CLS-AUDIT | `class_signature` corpus audit: `curated_entities()` (all 15 l-lanes, 64 consts) + `corpus_summary()` + falsifiable test that the real curated corpus collapses entities->fewer shape-families (classes.md:42 CONFIRMED on real data, not asserted) | lance-graph-ontology | 90 | LOW | **Shipped** | #441 D-CLS Wave-2 input; +clippy fix (unused FieldMask import in class_resolver) |
| D-CLS-RENDER | `ClassView::render_rows` + `RenderRow{label,predicate}` — the off-bits-skipped render surface (C2 presence-only; template-agnostic, askama engine deferred to its own crate-Wave) | lance-graph-contract | 50 | LOW | **Shipped** | #441 D-CLS; the render LOGIC (classes.md:49), not the engine; +doc-lint fix |
| D-WIKI-HHTL-1 | `contract::hhtl::NiblePath`: the 16ⁿ Abstammung bucket router (subClassOf nibble path, bit-shift O(1), `root`/`child`/`basin`/`parent`/`is_ancestor_of`) + `FieldMask::inherit` (mask-inherits-as-delta). DOLCE-agnostic (`basin: u8` = dolce_id 0..3, resolved through the cache — NO enum); multi-parent = facet bit in the same mask, NOT a 2nd path. The downstream router #438 names. | lance-graph-contract | 155 | LOW | **In PR** | Wikidata-HHTL slice 1 = the 16ⁿ router (hub-side); reuses #441 FieldMask; convergent with D-ARM-14 (firewall preserved). 4 teeth-tests; 501 contract lib green. See FINDING D-CLS↔D-ARM-14 (EPIPHANIES). |
| D-WIKI-HHTL-2 | `wikidata_hhtl`: the N4 second-domain falsifier — `WikidataClass` (curated real QIDs: human/person/city/film/tv-series/event) routed through the SAME class-meta-DTO: `nibble_path()` (basin=cache dolce_id, NO enum), `presence_mask()`=FieldMask, `signature()`=StructuralSignature over the canonical property-set, `dcls_triple()`=(ClassId,StructuralSignature,FieldMask), + `WikidataClassView` impls the #441 `ClassView`. | lance-graph-ontology | 290 | LOW | **In PR** | Wikidata-HHTL slice 2; classes.md N4 CONFIRMED on data: corpus collapses to fewer shape-families (film≡tv-series twin), triple shape domain-independent, ClassView resolves Wikidata unchanged, subclass inherits path+mask-as-delta. Reuses contract::hash::fnv1a. 5 teeth-tests; 245 ontology lib green. Deferred: the 115M streaming load (separate plan). |

---

## bindspace-singleton-to-mailbox-soa-v1 — dissolve `Arc<BindSpace>` into per-mailbox `MailboxSoA<N>`

Plan path: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`. Epiphany `E-MAILBOX-IS-BINDSPACE`. Migration of the shared singleton address space into mailbox-owned ephemeral thoughtspace (LE-contract SoA columns); drops the 64 KB `Vsa16kF32` `cycle` plane.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-1 | add migrated columns (`edges`/`qualia`/`meta`/`entity_type`) to `MailboxSoA<N>` behind `mailbox-thoughtspace` feature | cognitive-shader-driver | 120 | MED | **Queued** | gated on D-CE64-MB-1-impl + PR-NDARRAY-MIRI-COMPLETE |
| W3+W4a | atomic read/write shim (`backing::{BackingStore,BackingStoreWrite}`) — `driver.run()` keeps ONE body, `mailbox-thoughtspace` (default-OFF) flips substrate singleton→`MailboxSoA`; 6 reads re-pointed; W2 differential proves bit-identity; firewall lint + field-isolation + footprint gates | cognitive-shader-driver | 600 | MED | **In PR** | branch `claude/bindspace-mailbox-soa-w3-w4a`; default 97+2+2 tests, feature 98+2+2+4; clippy/fmt clean; `unbind_busdto` C5 downgrade feature-gated (cycle plane never migrated). Plan `bindspace-mailbox-soa-w3-w4a-impl-v1.md` |
| D-MBX-2 | move `engine_bridge` per-row read/write surface onto mailbox rows; `cycle` plane becomes a transient local | cognitive-shader-driver | 180 | MED | **Queued** | blocked on D-MBX-1 + OQ-1 (content-ref shape) |
| D-MBX-3 | `ShaderDriver` holds a sea-star of mailboxes; kill the `BindSpace::zeros(4096)` singleton in `serve.rs` | cognitive-shader-driver | 160 | HIGH | **Queued** | blocked on D-MBX-2 + OQ-2 (temporal/expert fold) |
| D-MBX-4 | death → SPO-G quad + Lance tombstone-witness (link-integrity back-pointer) | cognitive-shader-driver + Lance | 200 | HIGH | **Queued** | blocked on D-MBX-3 + Zone-2 persistence |
| D-MBX-5 | delete `BindSpace` singleton + `Vsa16kF32` `cycle` plane; remove feature gate | cognitive-shader-driver | 80 | MED | **Queued** | blocked on D-MBX-4 + OQ-4 (CLAUDE.md "The Click" doctrinal update) |
| D-MBX-6 | `ThoughtStruct` = transparent hot/cold view over SurrealDB container table(s) (same SoA both tiers; ~64k–256k hot ceiling, ~6 KB/thought) | cognitive-shader-driver + surreal_container | 220 | HIGH | **Queued** | blocked on D-MBX-3 + surreal_container unblock (BLOCKED A/B/C/D) or callcenter Zone-2 |
| TD-RESONANCEDTO-DUP-1 | dedup the two `ResonanceDto` (thinking-engine) | thinking-engine | 60 | LOW | **Deferred** | user 2026-05-27 — fold into D-MBX-2 |

---

## odoo-savant-reasoners-v2 — reshape: `Reasoner` trait → typed composition over `CausalEdge64` + `Tactic` + `callcenter/role_keys`

Reshape of v1 (shipped PR #420). v1's `Reasoner` trait surface fails CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove" litmus tests; v2 routes the canonical path through the agnostic substrate that already exists (CausalEdge64 + Tactic + 33-TSV atoms + role-key catalogues). v1 stays under `legacy-reasoner` feature with `#[deprecated]` until woa-rs migrates. Plan path: `.claude/plans/odoo-savant-reasoners-v2.md`. Driver epiphany: `E-SAVANT-COMPOSITION-1`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-SAV-5a | `SavantPattern` + `TacticInvocation` + `EdgeEmissionSpec` + `AtomTouchMask` primitives (Group D, zero-dep, in contract) | lance-graph-contract | 200 | HIGH | **Queued** | additive — ships with this plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section + EPIPHANIES entry (board hygiene) |
| D-ODOO-SAV-5b | `callcenter/role_keys.rs` with 25 disjoint Vsa16kF32 slices + lookup-by-enum + slice-allocation manifest (Group E) | lance-graph-callcenter | 250 | HIGH | **Queued** | parallel with 5a — independent; coordinate disjoint slice range vs `grammar/role_keys.rs` |
| D-ODOO-SAV-5c | 25 `SavantPattern` consts drawn from `.claude/odoo/savants/<N>.md` slot 1/4 + `.claude/odoo/L*.md` business semantics (Group F) | lance-graph-callcenter | 600 | MED | **Queued** | blocked on 5a + 5b; likely one D-id per savant in a Wave if translation is large; 14 NEEDS-INPUT savants ship pattern + emission spec only |
| D-ODOO-SAV-5d | `#[deprecated]` + `legacy-reasoner` feature gate + migration pointers on v1 `Reasoner` trait + 4 `*Reasoner` impls + `SavantConclusion` + `SavantSuggestion` + `build_conclusion` (Group G) | lance-graph-contract + lance-graph-callcenter | 120 | HIGH | **Queued** | blocked on 5c (so the migration pointer names a real target); removal in a follow-up after woa-rs migrates |
| D-ODOO-SAV-5e | End-to-end test: FiscalPositionResolver `SavantPattern` over a synthetic ontology fixture → expected `CausalEdge64` row (SPO + NARS truth + v2 signed mantissa); the proof the reshape works | lance-graph-callcenter tests | 150 | MED | **Queued** | ships with 5c completion as the round-trip proof; uses `CausalEdge64::pack_v2` per `I-LEGACY-API-FEATURE-GATED` |

---

## odoo-business-logic-blueprint-v1 — typed Odoo entity DTOs as the substrate for OGIT → OWL → DOLCE → FIBU/FIBO normalization + JITson / recipe codegen

PREREQUISITE for `odoo-savant-reasoners-v2` Group F (per `E-SAVANT-COMPOSITION-1`). Establishes the typed `OdooEntity` + sub-types layer that the inheritance chain operates on — replaces today's ad-hoc string-keyed maps against `model_name`. Both passes (L-docs first as curated filter, Odoo source extraction second as exhaustive backing). All 15 lanes (L1–L15). Plan path: `.claude/plans/odoo-business-logic-blueprint-v1.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-BP-1a | `OdooEntity` + sub-types (`OdooField`/`OdooMethod`/`OdooDecorator`/`OdooStateMachine`/`OdooConstraint`/`OdooProvenance`) — zero-dep, const-only, no serde | lance-graph-ontology | 300 | HIGH | **Queued** | ships with plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section (board hygiene); additive — zero churn to existing call sites |
| D-ODOO-BP-1b | L-doc projection: one `OdooEntity` const per entity, 15 lanes, per-lane module `odoo_blueprint::l{1..15}`, provenance=Curated with line-range citations | lance-graph-ontology | 2500 | HIGH | **Queued** | blocked on 1a; ships in Waves (L1-L5, L6-L10, L11-L15), one subagent per lane (Sonnet, mechanical prose→const projection); ~5 entities/lane average × 15 lanes ≈ 75-200 consts |
| D-ODOO-BP-1c | Wire OGIT classifier to take `&OdooEntity` (replaces string-keyed `resolve_odoo`); uses field/method semantics for richer dispatch; covers 0x63/0x90 from PR #414 | lance-graph-ontology + lance-graph-callcenter::family_table | 250 | HIGH | **Queued** | blocked on 1b; parallel with 1d/1e |
| D-ODOO-BP-1d | Wire OWL hydrator to take `&OdooEntity`: relational fields → edges, computed fields → SHACL-equivalent constraints, decorators → axioms | lance-graph-ontology | 350 | MED | **Queued** | blocked on 1b; parallel with 1c/1e |
| D-ODOO-BP-1e | Wire DOLCE classifier + FIBU/FIBO alignment to take `&OdooEntity`; closes D-ODOO-SAV-2's `None`-class alignment for stock.* / analytic.distribution.model / account.account.tag over typed input | lance-graph-ontology | 200 | HIGH | **Queued** | blocked on 1b; parallel with 1c/1d |
| D-ODOO-BP-1f | Odoo source extraction tool: tree-sitter Python AST → candidate `OdooEntity` consts with Confidence=Extracted; validates + extends 1b's curated set | tools/odoo-blueprint-extractor/ | 800 | MED | **Queued** | blocked on 1b/c/d/e; conflicts (curated vs extracted) flag for ratification, default to curated |
| D-ODOO-BP-1g | Wire JITson → recipes: `jit::JitCompiler` compiles `Tactic` kernels parameterized by `(&OdooEntity, AtomTouchMask)`; produces DTO-ish NARS that lands in shader-driver | lance-graph-contract::jit + thinking-engine | 400 | MED | **Queued** | blocked on 1c/d/e; proof-of-concept on FiscalPositionResolver, the rest follow in `odoo-savant-reasoners-v2` Group F |
| D-ODOO-STYLE-1 | `style_recipe.rs` — Phase 1 D-Atom interpretation step: typed Odoo SoA → `OdooStyleRecipe` cognitive fingerprints (12 DAtom basis, 7-rule cascade, FNV-1a recipe_id, never stored back as triples) | lance-graph-ontology::odoo_blueprint | 746 | HIGH | **Shipped** | commit `feb8be54` (PR #433 merged); 13/13 tests; DAtom::ALL discriminant-order pinned; OdooStyleRecipe != contract::recipe::StyleRecipe (documented) |
| D-ODOO-OP-1 | `op_emitter.rs` — Phase 2 bucket-dispatch codegen: `bucket_corpus` groups OdooStyleRecipe by OdooMethodKind; `emit_op_dispatch` emits compilable Rust (RECIPE_* consts + per-kind Op structs + static Op slices); deterministic, recipe_id dedup collapses identical DAtom profiles | lance-graph-ontology::odoo_blueprint | 400 | HIGH | **Shipped** | commit `63f3e2ca`; 12/12 tests; zero-dep emitted output; 230/230 existing tests green |

---

## streaming-arm-nars-discovery-v1 — upstream proposer leg into the SPO substrate (20K-200K rows/window pair-stats + optional Aerial+ → NARS-truth → SpoStore hypothesis test → council ratification → op_emitter codegen)

The missing UPSTREAM discovery leg. Today's proposers (curated L-docs + AST-extracted Odoo source) are bounded by the literal artifact; this plan adds runtime-tabular-data ARM discovery, gated through the epiphany-brainstorm-council before reaching the deterministic codegen path. Plan: `.claude/plans/streaming-arm-nars-discovery-v1.md`. Handover: `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ARM-1 | `ProvenanceTier::{Curated,Extracted,ArmDiscovered,Ratified,Conjecture}` enum + ordering | lance-graph-contract | 50 | HIGH | **Queued** | blocks all other D-ARM-*; additive |
| D-ARM-2 | `Proposer` trait + `CandidateRule` carrier + `WindowMetadata` | lance-graph-contract | 100 | HIGH | **Queued** | blocks D-ARM-3, D-ARM-9. D-ARM-13 shipped **local mirrors** (`rule::{CandidateRule, Proposer, Item}`) ahead of this — see **TD-ARM-CARRIER-FORK**: re-export via `pub use` when this lands (firewall allows path-dep on zero-dep contract). Field set diverges — local carries bare `n: u32`, this plans `WindowMetadata`; reconcile (recommend `n: u32`) so the shape matches. |
| D-ARM-3 | Pair-stats proposer (default trunk, deterministic, k² pair counters per window) | lance-graph-arm-discovery::proposer::pair_stats | 400 | HIGH | **Queued** | depends on D-ARM-1/2/7; blocks D-ARM-12 |
| D-ARM-4 | ARM-truth → NARS-truth translator + Odoo `FeedProjector` impl | lance-graph-arm-discovery::translator | 200 | HIGH | **Partially shipped (branch)** | The translator substance landed early inside D-ARM-13: `translator::{arm_to_nars, NarsTruth, CandidateTriple, FeedProjector}` (verbatim paper §2/§3.3 mapping, 35/35 tests). REMAINING: the real Odoo `FeedProjector` (currently a `DebugProjector` stub emitting `implies`) + contract homing on D-ARM-1/2. Depends on D-ARM-1/2. |
| D-ARM-5 | Hypothesis test: SpoStore round-trip, NARS revision, contradiction commit per The Click | lance-graph-arm-discovery::hypothesis | 350 | MED | **Queued** | depends on D-ARM-4; verifies `spo::truth::Contradiction` primitive exists |
| D-ARM-6 | `RatificationQueue` ring buffer + corrections-to-#434 spec PR (`discovery_arc D=8`, `discovery_origin u8`) | lance-graph-arm-discovery::queue + #434 spec follow-up | 200 + spec | MED | **Queued** | depends on PR #434 D-MBX-A3 landing |
| D-ARM-7 | Jirak-2016 weak-dependence significance thresholds (mandatory Stage A floor) | lance-graph-arm-discovery::jirak | 150 | HIGH | **Queued — HARD PREREQUISITE** | blocks D-ARM-3; cites I-NOISE-FLOOR-JIRAK. **ISSUE ARM-JIRAK-FLOOR (2026-05-30, 3-savant review):** D-ARM-13 ships the Aerial proposer with NO Jirak floor (classical `min_support`/`min_confidence` only). MUST land before D-ARM-5 wires the proposer to a live `SpoStore`, else the substrate calcifies on thin-but-frequent noise (plan §11.1). **ENGINE EXISTS:** `jc::jirak` (Jirak-Cartan Pillar 5) is the weak-dependence Berry-Esseen rate (`n^(p/2-1)`); this deliverable is the *gate function* (rule → significant?) that derives its threshold from it — NOT a from-scratch Jirak impl. See E-ARM-JC-RESOLVES-BOTH-SEAMS + `splat-codebook-aerial-wikidata-compression.md`. |
| D-ARM-8 | `Feed` + `FeedProjector` + window-size config + Odoo `account.move` projector example | lance-graph-arm-discovery::feed | 250 | MED | **Queued** | depends on D-ARM-2 |
| D-ARM-9 | Aerial+ IPC client (feature-gated `arm-aerial`, NDJSON over Unix socket) | lance-graph-arm-discovery::proposer::aerial_ipc | 200 | MED | **Superseded by D-ARM-13** | The native in-process Aerial+ transcode (D-ARM-13, branch `claude/jolly-cori-clnf9`) replaces the need for the Python IPC client. The determinism-boundary rationale the IPC was designed for (keep the nondeterministic autoencoder out of the Rust path) is now met in-process via seed (`aerial::Rng`) + `aerial` feature gate + workspace `exclude`. Keep this row ONLY if a Python-only Aerial variant is later required; otherwise close as Abandoned-by-replacement. |
| D-ARM-10 | `op_emitter::bucket_corpus` ratification filter (`confidence ≥ Ratified`) + 2 tests | lance-graph-ontology::op_emitter | 30 | HIGH | **Queued** | depends on D-ARM-1 |
| D-ARM-11 | `style_recipe.rs` rule 8 — ArmDiscovered backing adds `DAtom::Compute` weight 2 (provisional) | lance-graph-ontology::style_recipe | 80 | MED | **Queued** | depends on D-ARM-1 |
| D-ARM-12 | End-to-end pipeline test + bench (synthetic Odoo feed → all 5 stages → council micro-batch) | lance-graph-arm-discovery::tests + benches | 400 | MED | **Queued** | depends on Waves 1-6; informs OQ-ARM-2 + OQ-ARM-7 |
| D-ARM-13 | **Aerial+ Rust transcode — deterministic codebook-probe backend** (float-free). The paper's `f32` denoising autoencoder is REPLACED by an integer `CodebookDistance` oracle (palette256 distance, ρ=0.9973 vs cosine): the reconstruction probe is a codebook top-k, not a softmax over float weights. Integer evidence counts + ppm gates + `TruthU8` (= CausalEdge64 wire). `AerialProposer` impl of `Proposer`. Count loop is a row-bitset SoA (`RowMasks`) → AND+popcount, routed through `ndarray::simd::U64x8` under the `ndarray-simd` feature. | lance-graph-arm-discovery::aerial | ~1.1K | HIGH | **Shipped (branch)** | branch `claude/jolly-cori-clnf9`; standalone zero-dep crate (excluded); **33/33** tests + clippy `-D warnings` clean on BOTH default (scalar) and `--features ndarray-simd`; **zero f32 in the discovery path** (audit), float only at the `TruthValue`/`Triple` serialization edge. Bitwise-deterministic ⇒ joins the trunk; the nondeterminism firewall + D-ARM-9 IPC rationale are moot. SIMD target-cpu caveat: real AVX-512/AMX kernels need `-C target-cpu=native`/`x86-64-v4`. v1 (autoencoder) superseded per the user's no-float directive. |
| D-ARM-14 | **Splat-codebook oracle + Wikidata skeleton discovery** — wire the certified jc splat codebook into aerial as the `CodebookDistance` oracle, discover OWL/DOLCE+ SPO HHTL classes + basins, drive the `wikidata-hhtl-load.md` deterministic compression (skeleton + basins + CAM-dedup + thin rows). | lance-graph-arm-discovery::aerial + crates/jc + wikidata loader | ~? | MED | **In progress** | **Phase 1 (branch `claude/jolly-cori-clnf9-darm14`):** the two aerial-side seams — `aerial::TopKDistance` (the sparse splat-top-k `CodebookDistance` the 10000² BLASGraph splat actually emits; top-k per node, not a dense table) + `aerial::ontology::{DolceCategory, OntologyProjector}` (DOLCE 4-facet skeleton → `rdfs:subClassOf`/`rdf:type` SPO). End-to-end test: splat top-k → aerial discovers `occupation→DOLCE-class` → projects the skeleton triple. 41/41 + clippy clean (default + `ndarray-simd`), zero-dep. Float still OFFLINE in jc only (`ewa_sandwich`+`sigma_codebook_probe` ρ=0.9973+`pflug` Lε); aerial online path integer. **Phase 2 (branch `claude/jolly-cori-clnf9-darm14-p2`):** the proposer→hub landing. (a) `OntologyProjector::dolce_id()` — emits the stable `dolce_id` u8 (= basin nibble) the hub routes by, NOT a hardcoded IRI (the OD-DOLCE alignment #442 deferred to this lane; basin ordering already matches `dolce_id::{ENDURANT=0..}`). (b) gated worked example `tests/wikidata_landing.rs` (`--features landing`, opt-in `dev-dep lance-graph-contract` à la jc): splat top-k → aerial recovers all 6 DOLCE basins → lands each on the REAL merged `contract::hhtl::NiblePath` (16ⁿ router, #442) + `class_view::FieldMask` (+`inherit`) + `hash::fnv1a_str` signature. CONFIRMED on data: corpus collapses 6→5 families (film≡tv-series twin), human⊂person inherits path + mask-as-delta, basin preserved. 42/42 default (zero-dep) + landing test green, clippy clean both. Rebased onto post-#442 main; the inline-nibble stand-in swapped for the real `NiblePath`. **Remaining:** real jc/blasgraph splat producing the lists; the ndjson→`WikidataClass` loader; gated on D-ARM-7 (`jc::jirak`). Map: `splat-codebook-aerial-wikidata-compression.md`; E-ARM-JC-RESOLVES-BOTH-SEAMS. |
| D-ARM-SYN-1 | Add `Implies`/`CoOccursWith` to `ruff_spo_triplet::Predicate` closed vocabulary (+ `Provenance` tier) so ARM rules load through the same `parse_triples` ndjson path as the static extractor | ruff/ruff_spo_triplet | 40 | MED | **Queued** | council-gated (deliberate ontology change); blocks SYN-2; see `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` §1 |
| D-ARM-SYN-2 | `CandidateRule → ruff_spo_triplet::ModelGraph` adapter so the Aerial runtime-data leg joins the `ruff_python_dto_check` static-AST leg in one graph before `expand()` | lance-graph-arm-discovery + ruff_spo_triplet | 120 | MED | **Queued** | depends on SYN-1; synergy doc §2 |
| D-ARM-SYN-3 | Calibrate `ProvenanceTier::ArmDiscovered` `(f,c)` below the `op_emitter` ratification gate + below static `Inferred (0.85,0.75)` so un-ratified ARM truth is council-visible but codegen-filtered | lance-graph-contract + lance-graph-ontology::op_emitter | 30 | MED | **Queued** | depends on D-ARM-1 + SYN-1; synergy doc §3/§4 |
| **D-CHESS-BRINGUP-1** | **Chess-into-OWL falsification slice** — encode 3-5 opening positions as OWL/ttl (meaning in CONTENT, no chess-special SoA field), run `lance-graph-arm-discovery::AerialProposer` (the now-shipped #436 Rust transcode) over it, see whether GM-flavored *legal* candidates fall out, AND whether chess needs columns Odoo's SoA didn't have. The cognitive-risc-core/classes spec's emphatic **N4 freeze-time non-negotiable** — falsifies/confirms "one SoA serves all" cheaply on a board, before the WAL freezes. Council R2+R4 verdict 2026-05-30: this is FIRST, not a peer option. | lance-graph-arm-discovery (read) + new crate `chess-bringup-test` | ~300 | HIGH | **Queued** | NEW (council recalibration 2026-05-30). NOT in scope for branch `claude/activate-lance-graph-att-k2pHI` (per R1 — needs its own branch + freeze-decision authority). Unblocked by #436's Aerial+ shipping in Rust (user-flagged 2026-05-30 "aerial+ has been transcoded and is now a lance-graph-* crate"). Cross-ref `cognitive-risc-core.md` §"The bring-up test"; `cognitive-risc-classes.md:66` N4; `post-438-integration-options-v1.md` §1 Option G. |

---

## odoo-classes-bitmask-render-v1 — bounded-weekend ClassId + FieldMask + per-class askama templates (Aerial+ discovers ~10-15 shape-families from 66 OdooEntities; presence-bitmask render path)

The bounded-weekend fix `cognitive-risc-classes.md:56-57` prescribes (discriminator + parent-pointer + parent-walking; full machinery deferred). 4-way `DolceCategory` consolidation + `ClassId(u16)` hook + per-class `FieldPositionTable` + `FieldMask(u64)` + per-class askama templates. **All deliverables `Blocked-on-OD` until spec owner ratifies OD-DOLCE-CANONICAL, OD-CLASSID-WIDTH, OD-CLASSID-VS-ENTITYKIND, OD-TEMPLATE-ENGINE.** Plan: `.claude/plans/odoo-classes-bitmask-render-v1.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| **D-CLS-1** | Canonical `DolceCategory` in `lance-graph-contract` + re-exports from 3 sites + `From<DolceCategory> for DolceMarker` | lance-graph-contract + 3 modified | 80 | HIGH | **Blocked-on-OD** | Wave 1A, Sonnet. Additive per C6. Arm-discovery uses local newtype + TryFrom (zero-dep stance preserved) |
| **D-CLS-2** | Structural-signature audit of 66 OdooEntities → `.claude/knowledge/odoo-66-structural-signatures.psv` | lance-graph-ontology (read only) | 230 | HIGH | **Blocked-on-OD** | Wave 1B, Sonnet. Read-only emit; BLAKE3-128 truncated u64 per-entity hash |
| **D-CLS-3** | Aerial+ structural-hash → ~10-15 candidate shape-families + ratified `CANONICAL_CLASS_TABLE` | lance-graph-arm-discovery (example) | 350 | MED | **Blocked-on-OD** | Wave 2A, **Opus**. SPEC-OWNER GATE after output: names + ratifies clusters |
| **D-CLS-4** | New `lance-graph-ontology-render` crate skeleton + askama dep + `exclude=` workspace entry | new crate | 70 | HIGH | **Blocked-on-OD** | Wave 1C, Sonnet. Standalone like bgz17/deepnsm |
| **D-CLS-5** | `ClassId(u16)` newtype + `UNCLASSIFIED` const in `lance-graph-contract::cognition::entity` | lance-graph-contract | 40 | HIGH | **Blocked-on-OD** | Wave 3A, Sonnet. The N1 hook |
| **D-CLS-6** | `class_id: ClassId` field on `OdooEntity` + back-fill 66 consts via ratified CANONICAL_CLASS_TABLE | lance-graph-ontology (mod + 15 lanes + new CLASS_TABLE.rs) | 260 | HIGH | **Blocked-on-OD** | Wave 3B, Sonnet. Mechanical edit across 15 lane files |
| **D-CLS-7** | `FieldMask(u64)` + `FieldPositionTable` (N3 append-only positions) + per-class width audit | lance-graph-contract (new field_mask.rs) + lance-graph-ontology (CLASS_TABLE extend + class_audit.rs) | 250 | HIGH | **Blocked-on-OD** | Waves 3C + 3D + 3E split (5 Sonnet agents in Wave 3 total) |
| **D-CLS-8** | Per-class askama templates (~10-15 .txt.j2) + `render(entity, mask) -> String` + per-class smoke tests | lance-graph-ontology-render (lib + templates + tests) | 510 | MED | **Blocked-on-OD** | Wave 4 (3 Sonnet agents — templates, dispatch, tests) |
| **D-CLS-9** | Integration test 66 entities × class templates + C2 mutant-mask test + mask-density audit report | lance-graph-ontology-render (tests + audit + snapshots) | 2,310 | HIGH | **Blocked-on-OD** | Wave 5A, **Opus**. Bulk LOC is 66 generated snapshots + per-class density report |

---

## wikidata-lazy-spine-hydration-v1 — the NiblePath-keyed tiered hydration manager + addressing (the "agnostic lazy world-spine" runtime)

The one missing runtime piece behind the converged delta-card / world-spine vision (`delta-card-addressing-integration-map.md`, `agnostic-lazy-world-spine.md`). Plan: `.claude/plans/wikidata-lazy-spine-hydration-v1.md` (9 D-ids, authored by the W1 wave worker). All gated on D-ARM-7 (Jirak floor) before any hydrated rule writes a live store; firewall (aerial = zero-dep proposer, hub owns contract/ontology) preserved.

| D-id | Deliverable | Crate(s) | LOC | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-LWS-1 | Sparse radix range-delegation register (path-compressed trie over the frozen ontology; occupied branch points only; reuses `NiblePath` as the address — never re-encodes identity) | lance-graph-contract / -ontology | ~? | MED | **Queued** | partition-as-address; 27-bit floor with ~0-bit row |
| D-LWS-2 | Delta-card value model (`reconstruct = deck ⊗ delta`; per-entity surprise as a `FieldMask` delta over the inherited archetype; modal member = empty card) | lance-graph-contract | ~? | MED | **Queued** | built on `FieldMask::inherit` |
| D-LWS-3 | RISC compose-cache + per-predicate composability flag (store generators, compose ≤7-hop closure via `ComposeTable`/`mxm`; dissolves the hub problem) | lance-graph + bgz-tensor | ~? | MED | **Queued** | generators=continuant/cold, composed=occurrent/evictable |
| D-LWS-4 | I/P/B frame model over Lance versioning (I=frozen radix+base, P=append, B=compose-cache, GOP=compaction) | lance-graph | ~? | MED | **Queued (spike)** | R2: repo wires dataset-level `VersionedGraph`, not fragment-level — fragment GOP is a NEW spike |
| D-LWS-5 | **The `NiblePath`-keyed tiered hydration manager** (THE missing piece): hot `MailboxSoaView` ↔ cold `VersionedGraph`, address-not-join, agnostic SoA, carries CE64+witness arc; write-refusal until D-ARM-7 | lance-graph | ~? | MED | **Queued** | centerpiece; D-ARM-7 write-refusal acceptance test |
| D-LWS-6 | Foveated prefetch cascade (`HhtlCache::route` Skip/Attend/Compose/Escalate decides periphery prefetch into the 256K envelope) | lance-graph + bgz-tensor | ~? | MED | **Queued** | the Google-Maps tile prefetch |
| D-LWS-7 | Eviction on the DOLCE continuant/occurrent 1-bit (`dolce_id==PERDURANT` ⇒ occurrent ⇒ evictable; 4-facet axis preserved, residence bit derived) | lance-graph | ~? | MED | **Queued** | the perm/temp residence policy |
| D-LWS-8 | Probe harness — runs the 3 falsifiers (Louvain-CLAM locality, delta-card residual, compose hit-rate) on real `data/ontologies/*.ttl` + fixtures; PRODUCES the gates | crates/jc + lance-graph | ~941 | HIGH | **Probe-1 SHIPPED** | `jc/examples/ontology_locality_probe.rs` RUN: **locality 98.6%, max fan-out 3 (≤16), Q=0.325 → PASS** on real ontologies (not yet Wikidata). Probes 2-3 queued. |
| D-LWS-9 | DEFERRED full Wikidata 115M load (skeleton+basins+CAM-dedup+thin rows) | wikidata loader | ~? | LOW | **Deferred** | gated on all 3 probes PASSED + D-ARM-7; CONJECTURE (no dump on disk) |

## Markov substrate clarification (markov_soa / EW64) — three-Markovs taxonomy

| D-id | Deliverable | Crate(s) | LOC | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-MKV-SOA | `arigraph::markov_soa` — the Markov *wave* (AriGraph cold-path chain promoted to hot-path SoA); vocabulary-agnostic `SpoRanks{u16}` + `SoaWavePrimer` + `WaveProjection::best_guess_match(injected dist)`; the "hybrid+ autocomplete" #2 proposer (dark-horse) | lance-graph::graph::arigraph | ~230 | MED | **Shipped (branch, unverified-offline)** | moved out of deepnsm (SoC fix); match = AriGraph's own cam_pq, language stays upstream; 4 tests written, core doesn't build in sandbox → verify on full checkout. Findings: three-Markovs, markov_soa-IS-AriGraph |
| D-EW64-NOTE | `MailboxSoaView` doc note: `EpisodicWitness64` = AriGraph in the mailbox SoA view (the particle; cold→hot); deferred accessor (qualia-pattern) | lance-graph-contract::soa_view | ~20 | HIGH | **Shipped (branch)** | verified (contract builds, 3/3 soa_view tests); EW64 not yet a code symbol — P2 of three-Markovs ordering |

---

## Update protocol

When a deliverable ships:
1. Edit this file's Status column in place for the row → **Shipped**.
2. Fill in PR / Evidence column with the merge commit or PR #.
3. Append a new section to `PR_ARC_INVENTORY.md` (Added / Locked /
   Deferred / Docs / Confidence).
4. Update `LATEST_STATE.md` (Recently Shipped PRs + Current Inventory
   if types change).

When a deliverable moves phase (e.g. Queued → In progress → In PR):
1. Edit Status column in place. Don't reorder rows.
2. If the move reflects scope correction, also update
   `INTEGRATION_PLANS.md` Status line for the parent plan.

When a new deliverable is added to a plan:
1. Append a new row at the bottom of the plan's section.
2. D-id is sequential in the plan (D12, D13, etc.).
3. Original scope becomes immutable once committed.

When a deliverable is abandoned:
1. Edit Status → **Abandoned**. Don't remove the row.
2. Cite the replacement in Notes.

## D-EW64-3 / D-EW64-4 (2026-06-01, autoattended)

| D-id | deliverable | status | evidence |
|---|---|---|---|
| D-EW64-3 | `EpisodicEdges64::{coldest, contains}` — MRU cold-tier read surface | In PR | contract lib 545 green; clippy clean |
| D-EW64-4 | `DemotionSink` trait + `promote_into` — hot→cold exit seam (impls gated OQ-11.6) | In PR | contract lib 545 green; clippy clean |

---

## identity-architecture-exists-vs-needs-v1 — structured NodeGuid + frugal north-star OGAR mint

Plan path: `.claude/plans/identity-architecture-exists-vs-needs-v1.md`. Epiphanies: E-IDENTITY-WHITEBOX-1, E-OGAR-NORTHSTAR-1. Rides in the open identity PR on `claude/nice-edison-g4rhhl`.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-IDENTITY-1 | `identity::NodeGuid` (UUIDv8) + `NiblePath::from_packed` — byte layout, version/variant gates, field-isolation matrix | `lance-graph-contract` | ~250 | LOW | **Shipped** | Phase A; +15 contract tests, clippy-D clean |
| D-IDENTITY-2 | Frugal north-star mint: dedup-by-URI global template id + `entity_type↔NiblePath` bijection pair table + round-trip tests (moves 1+2+3) | `lance-graph-ontology` | ~250 | LOW | **In PR** | dedup + `register_class_path`/`niblepath_of`/`entity_type_of`/`rows_with_entity_type`; +5 tests, 14 registry green |
| D-IDENTITY-3 | Gate legacy positional `contract/ontology.rs:85 entity_type_id` per I-LEGACY-API-FEATURE-GATED (move 4) | `lance-graph-contract` / -ontology | ~80 | MED | **Queued** | needs consumer audit first |
| D-IDENTITY-4 | Pair-table Lance persistence (re-register-on-hydration → persisted) | `lance-graph-ontology` | ~60 | LOW | **Queued** | TECH_DEBT TD-PAIRTABLE-1 |

---

## polyglot-container-query-membrane-v1 — three dialects, one HHTL address space, mailbox as cold path

Plan path: `.claude/plans/polyglot-container-query-membrane-v1.md`. Research grounded 2026-06-09; rides on `claude/nice-edison-g4rhhl`.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-PG-1 | `addr64` left-aligned HHTL codec + order-preservation property test (subtree ⇔ contiguous range) | `lance-graph-contract` | ~120 | LOW | **Queued** | first brick; everything stands on it |
| D-PG-2 | `SoaEnvelope` impl for `MailboxSoA<N>` (= identity N3, confirmed live) + LE parity test | `cognitive-shader-driver` | ~150 | LOW | **Queued** | gap re-verified 2026-06-09 (§2.4 of plan) |
| D-PG-3 | Read-only mailbox `Transactable` adapter (5 methods, phase-pinned) + hot==cold differential test | shader-driver + fork contract | ~250 | MED | **Queued** | gated on D-PG-1,2 |
| D-PG-4 | `SurrealqlParse` strategy → ArenaIR (SELECT point/range) + selector rule | `lance-graph-planner` | ~300 | MED | **Queued** | slot proven by sparql_parse |
| D-PG-5 | DDL ⇄ registry bridge (DEFINE walker → mint; reverse via C16b `ToSql`) | `lance-graph-ontology` | ~250 | MED | **Queued** | gated on fork C16c |
| D-PG-6 | (optional) `surreal_container` unblock → kanban view over LanceDB | `surreal_container` | ~200 | LOW | **Queued** | ruling-compliant; OQ-PG1 open |
| D-PG-7 | Deterministic foveated tree-builder (CLAM-style 16-way bootstrap + append-stable insertion → `register_class_path`) | `lance-graph-ontology` + ndarray CLAM | ~300 | MED | **Queued** | plan §8 addendum; gated on D-PG-1; determinism + append-stability property tests mandatory |
