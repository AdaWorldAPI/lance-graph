# 2026-10-17 — Co-research: Simpson's paradox as an epistemic signal

**Status:** OPEN. This is an exploration map from the co-research council
(`.claude/agents/coresearch-council.md`). It ratifies nothing; the operator picks
what to take forward.

## Question

Should Simpson's paradox (an association that reverses when the population is
stratified) be an epistemic signal for NARS revision? If so, in what form and
where? The question opened when D-CGW-1 (#1594) removed the per-edge Simpson
carrier (CE64 bits 43..45, `detect_simpsons_paradox`).

## Premise gate: the question was split

The premise check (verified against `certification.rs`) found four concepts
under one name:

| concept | home today |
|---|---|
| stratification reversal (Simpson proper) | `lance-graph-contract/src/certification.rs`: `strata: [M;4]`, rung 3 `Supports` = "in every declared stratum exposure does not lower the outcome". The Simpson fixture certifies `Related`, not `Supports`. |
| observational-vs-trial disagreement | planner `Operation::ConfounderCheck`. `CausalMask::simpsons_paradox_risk` is a misnomer: it detects confounding. |
| evidential overlap | revision gate Echo / ClosedCycle (G1, G2). Not Simpson. |
| reference class | no carrier |

## Load-bearing findings

1. **NARS revision is the amalgamation Simpson warns about.**
   - `isa::truth` revision computes `f = (f1·w1 + f2·w2)/(w1+w2)`. With `w = n` (c = n/(n+1)), revising stratum beliefs equals the union recount.
   - On the Simpson fixture: S0 exposed 6/8 vs unexposed 2/2, S1 exposed 0/2 vs unexposed 1/8. Both strata lower the outcome. Revised: exposed 0.6, unexposed 0.3, so the outcome is raised.
   - At u8 codes the result is 0.601 vs 0.299, the same direction (hand-computed; probe P-pool pins it).
   - Mantel-Haenszel weighting would not reverse; NAL revision does.
   - Sources: NAL draft Def 3.7 / §4.1 (SECTION-READ); Blyth via Vellaisamy 1403.6329 (READ-IN-FULL).
2. **Nothing in NARS stops it.**
   - NAL and ONA guard revision only by evidential overlap. Disjoint strata pass that check trivially.
   - ONA has no subgroup or reference-class check (ONA `Inference.c`, SECTION-READ, MIT).
3. **The data cannot pick a side.**
   - Pooled vs stratified is decided by the causal graph, not the counts (Pearl R-414 SECTION-READ; Lindley-Novick and Hernán SECONDHAND).
   - So the only admissible reaction is Suspend or refusal. Demoting is already rejected: production `pearl` never demotes an intervention-certified `Causes`.
4. **"Pool only when every stratum agrees" is not sufficient.**
   - The reversal also runs the other way: all strata raise while the marginal lowers, when exposure shares differ across strata.
   - Sure-thing is about the causal effect, not the marginal association. Admission also needs the balance condition (row 3).
5. **Everything is presence bits and popcount.**
   - `compare` decides direction by `popcount(Y∧a)·|b|` vs `popcount(Y∧b)·|a|`, with no division and no float.
   - Each stratum costs 4 popcounts. The folds are generic over `PopulationMask` (u64 .. `[u64;1024]`).

## Crosswalk and verdicts

| # | idea | relation | firewall | verdict |
|---|---|---|---|---|
| 1 | Stratum-aware pool guard: refuse/suspend revising beliefs of one question from different declared strata when a reversal is present | NEW | CONFLICT, fits via SEAM-C2M + SEAM-STRATUM-PROVENANCE | **PROBE** (P-pool, P-guard) |
| 2 | Admit pooling at `Supports` **and** balanced exposure (row 3) | PARTIAL | CONFLICT, fits via SEAM-C2M | **PROBE** (with row 3) |
| 3 | Mask-only collapsibility pre-check: `|E∧Sᵢ|·|¬E| == |¬E∧Sᵢ|·|E|` for all i, no outcome counts | NEW | CONFLICT, lives in `certification.rs` as a fold beside `compare` | **PROBE** (sweep the D-GSO-7a 2,500-model family; sufficient only, rarely fires on real data) |
| 4 | Explicit reversal report: every reversing stratum, not first-fail; the caller sets the pooled claim's bit in `contradictions` → ContradictionPreserved → Suspend | PARTIAL | CONFLICT (nearest PASS), via SEAM-C2M | **PROBE**; trap below |
| 5 | Reference class by specificity: narrowest declared stratum with adequate statistics | NEW | CONFLICT, via SEAM-ADMISSIBLE-MASK | **PARK** behind row 10 (collider counter-case required) |
| 6 | Specious-rule filter in arm-discovery (Hämäläinen & Webb 1709.03915): leverage given X and ¬X, integer δ prefilter, top-K only | NEW | PASS, conditional on integer form and declared X | **PROBE** |
| 7 | Candidate strata from a concept lattice (FCA) | NEW | TRAP without SEAM-PROPOSE-RESEAL | **PARK** (proposer only; re-seal before certifying) |
| 8 | Stratum as a coordinate range (layout sorted by stratum; popcount over slices, no AND with S) | PARTIAL | PASS (one sort order per layout; overlapping strata stay AND remainder) | **PROBE** (parity vs AND; `fold_ranges` is quack-private, verify first) |
| 9 | Rename `simpsons_paradox_risk` / `Measured::confounded` to say obs-vs-trial disagreement | ALREADY-HAVE (misnamed) | PASS | **ADOPT-NOW** (~12 sites, no production consumer) |
| 10 | Enforce strata declared before the seal (mirror "receipt after seal does not count") | PARTIAL | PASS | **ADOPT-NOW** as a test-first prerequisite for 1, 5, 6, 7 |
| 11 | Per-edge Simpson field | — | TRAP (A1) | SKIP |
| 12 | Demote certification on reversal | — | TRAP (prior ruling D-PEARL-PROD-0) | SKIP |
| 13 | Partial pooling (hierarchical, MCMC) | — | TRAP (float, A5) | SKIP |
| 14 | Dempster-Shafer combination | lesson already in gate tension | import TRAP (A5) | SKIP |

## The seams (firewall critic), all keeping the gate pure set algebra

- **SEAM-C2M (certification→mask, two-pass).**
  - Certification folds in the counting layer. Its verdict leaves only as a mask and enters `revise` as `contradictions` or claims.
  - The gate never calls certification, and the two modules do not import each other.
  - The adapter lives in the planner or in a free contract fn outside `GadamerRevision`.
- **SEAM-STRATUM-PROVENANCE.**
  - Stratum of origin travels in the existing claim/root masks, scoped by `question` (G7).
  - It is a fourth axis, separate from evidential independence (A6, PR #854's unbuilt `EvidentialBase<K>`).
  - The gate gains no stratum field.
- **SEAM-ADMISSIBLE-MASK.** "Adequate statistics" is pre-reduced by the counting layer to an admissible-class mask.
- **SEAM-PROPOSE-RESEAL.** Data-derived strata only seed a fresh sealed model.

**Index-space warning.** Certification masks index units (rows); gate masks index claims. Only the pooled claim's bit crosses, never the stratum masks themselves.

**Trap inside row 4.** In the gate, contradictions plus any new root give HorizonFusion (IncreaseEligible). More pooled evidence of the same kind must not count as that root. Only stratum-qualified evidence counts: an intervention, or a new declared stratification. Without that restriction G2 holds formally and is violated in meaning. Probe P-guard checks this.

## Probes (pre-registered; each has a silence twin)

- **P-pool (S).** Revise the Simpson fixture's two stratum beliefs with `isa::truth` (f32 and u8).
  - Kill row 1: revised exposed ≤ unexposed.
  - Expected: 0.6 vs 0.3, reversed, equal to the union recount.
  - Twins: the robust fixture (one stratum) must not reverse; two strata that both raise with balanced exposure must admit.
- **P-guard (M).** Gate `revise` with the bridge (rows 1 + 4).
  - Simpson → Suspend; same-stratum disjoint evidence → IncreaseEligible.
  - Kill: the decision needs a count in the gate, or fires on same-stratum operands, or a pooled-kind "new root" lifts the Suspend.
- **P-balance (S/M).** The row-3 pre-check over the 2,500-model family.
  - Kill: "collapsible" on any reversing model.
  - Twin: equal exposure share per stratum → collapsible and no reversal.
- **P-report (M).** Add a third reversing stratum.
  - Kill: the report is not exactly {S0, S1, S2}.
  - Twin: two agreeing non-empty strata → empty report, no Suspend.
- **P-collider (M, row 5).** Stratum S = E ∨ Y over 20 units with E independent of Y.
  - Kill: specificity picks the S-belief.
  - Twin: a declared confounder stratum must win.
- **P-specious (S, row 6).** RowMasks over the Simpson units, items E, Y, S0.
  - Kill: E→Y not flagged, or the δ prefilter drops a rule the full test flags.
  - Twin: a robust rule is not flagged.
- **P-range (M, row 8).** Sort the fixture by stratum.
  - Kill: range counts ≠ AND counts, or not ns-range on `[u64;1024]` at v4.
- **P-seal (S/M, row 10).** Union a mask into `strata[1]` after the seal.
  - Kill: certification changes.
  - Twin: pre-seal strata still certify Simpson as `Related`.
- **P-dredge (L, row 7).** Lattice strata on noise must not find reversals above a shuffled-label null.

## Not searched

- Multi-level drill-down beyond one partition: Fabris-Freitas only at ABSTRACT-ONLY. `[M;4]` is a single-level scope limit.
- Pei Wang's own writing on reference class in NARS: not found.
- Commercial OLAP reversal warnings: none found.
- Read grades:
  - Kyburg was read only through Loui.
  - Hernán, Touretzky/Horty and Greenland are SECONDHAND or ABSTRACT-ONLY; they need a read before citing in a ratified doc.
  - VLDB 2511.00748 (20–48% redundant paradoxes): ABSTRACT-ONLY.

## OPEN

- Which ideas to take forward: operator decision.
- D-GSO-7a's "strata declared before data" and D-PEARL-PROD-0's "which contradiction lowers a certification" remain open; rows 10 and 4 address them.
- Stratum-of-origin vs the unbuilt `EvidentialBase<K>` (#854): kept as separate axes, design not started.
