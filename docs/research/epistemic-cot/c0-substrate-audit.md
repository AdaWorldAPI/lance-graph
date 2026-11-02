# D-EPI-COT C0: substrate audit, and what C1/C2 add

**Date:** 2026-10-28. **Base:** `main` at `694b243bd` (after #1839).
**Status:** C0 audit complete. C1/C2 have a first implementation,
`crates/lance-graph-planner/src/nars/support.rs`. C3 to C6 are not started.

**Question.** Which source conditions are load-bearing for a conclusion, and
how does that knowledge change after a controlled ablation over several
competing chains of thought?

**Method.**
- Read in full: the planner's NARS truth, inference and belief arena; the
  contract's certification model, `epistemic_state5`, `nars.rs`; the
  CFA-2, EAW-2 and EAW-3 research notes; the 34-recipe plan.
- Mapped by two read-only agents, then spot-checked against the source
  (recipes 4, 11 and 31, and the inference match arms): `recipe_kernels.rs`
  (2,382 lines), `recipes.rs`, `recipe_dispatch.rs`, `recipe_substrate.rs`.

## 1. What exists, classified

| Piece | Where | Status | What it does / lacks |
|---|---|---|---|
| NARS truth laws | `planner/nars/truth.rs` | **Working** | `revise`, `deduction`, `induction`, `abduction`, `analogy`. Evidence horizon fixed at k = 1 (`w = c/(1−c)`, `c = w/(w+1)`), not a declared parameter. |
| `NarsInference::Synthesis` | `planner/nars/inference.rs` | **Not a NAL law** | `apply` returns the arithmetic mean of f and of c, which is not a NAL truth function. Its callers were not audited. |
| Pearl rungs 2 and 3 | `planner/nars/inference.rs` | **Mapped to abduction** | `Intervention` and `Counterfactual` both run the abduction truth function. No intervention semantics. |
| Typed statements | `planner/nars/belief.rs` `CStmt`, `Copula {Inh, Sim, Impl, Rel}` | **Working** | `S cop P`. `Rel` never transits. |
| Evidential stamps | `belief.rs` `Stamp(u64)` | **Working, lossy** | Disjoint stamps revise; overlapping ones go to CHOICE; an empty stamp never pools. Sources fold modulo 64, which can only create false overlap. |
| Multiple derivations of one statement | `BeliefArena::admit_derived` | **Missing** | One entry per statement. A second derivation replaces the first only if its expectation is higher; the other is dropped. |
| Source roots of a derived statement | `admit_derived` | **Missing** | Derived entries get an empty stamp; their roots are reachable only by walking `premises`, which hold one pair. |
| Retraction | arena | **Missing** | Nothing removes or downgrades a derived belief when a premise loses support. |
| Term-logic tactics | `planner/nars/tactics.rs` | **Working for 5 ids** | RCR (4) abduction, TR (6) analogy, ASC (7) challenge, CAS (8) induction/deduction, CR (11) revision/CHOICE. Each returns `Candidate`s with premises; pinned to the recipe taxonomy by `tactic_matches_recipe_taxonomy`. |
| 34 recipe kernels | `contract/recipe_kernels.rs` | **Scalar only** | See section 2. |
| Certification | `contract/certification.rs` | **Working, statistically thin** | See section 3. |
| `EpistemicState5` | `contract/epistemic_state5.rs` | **Working** | `Topology2 × Certification3`, 24 meaningful codes; reads gated by class, rail, generation and provenance. |
| Recipe eligibility from Epi5 | `cognitive-shader-driver/examples/shared/affordance_law.rs` | **Lab only** | `state[raw5] & pearl[mask]`. No eligibility field in `recipes.rs` or `recipe_dispatch.rs`. |
| Thought-Git | `crates/thought-git-probe` (excluded) | **Lab** | Slot pool with generations (63 live slots), receipts, oracle, squash, replay. No derivation ledger. |
| Per-class lane layout, sparse absence | `contract/lane_codec.rs`, #1839 | **Working** | Absent = presence 0 and value 0. |

## 2. The 34 recipes

Every contract kernel has the signature `fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome`.
`ThoughtCtx` holds `sd`, `free_energy`, `dissonance`, `temperature`,
`confidence`, `rung`, `candidates: Vec<f32>` and `beliefs: Vec<(u32, f32, f32)>`.
The `beliefs` tuples are `(topic, f, c)`, with no subject, copula, predicate or
stamp.

Across all 34 kernels:
- no kernel calls a NARS truth function;
- no kernel builds or returns a statement;
- no kernel reads a source, a stamp, an Epi5 state or a chain.

`KernelMaturity::Operational` (31 of 34) means only that `apply` changes
`ThoughtCtx` or returns a non-zero `delta_conf`. Are (19), Zcf (24) and Hkf
(34) are `Demonstration`.

Per recipe, the questions the mission asks have one answer for all 34
kernels:

- consumed statements: none;
- Epi5 / evidence prerequisites: none;
- inference law executed: none (the declared `inference(id)` is a label);
- multiple chains: no;
- source roots kept: no;
- new statement produced: no;
- survives ablation: not applicable;
- can retract: no.

What each kernel actually does (mapped, checked on 4, 11, 31):

| id | code | declared inference | effect |
|---|---|---|---|
| 1 | RTE | Induction | reads `free_energy`; writes `rung`, `free_energy` |
| 2 | HTD | Induction | rewrites `candidates` around their mean |
| 3 | SMAD | Revision | reads `candidates`; no write |
| 4 | RCR | Abduction | `candidates.reverse()` |
| 5 | TCP | Deduction | prunes `candidates` by `sd` |
| 6 | TR | Induction | perturbs `candidates` by `temperature` |
| 7 | ASC | Revision | reads `confidence`; no write |
| 8 | CAS | Deduction | quantizes `candidates` by `rung` |
| 9 | IRS | Induction | rewrites `candidates` by `temperature` |
| 10 | MCP | Revision | reads `confidence`, `free_energy`; no write |
| 11 | CR | Revision | detects `(topic, f)` pairs with `|Δf| > 0.5`; returns −0.2 |
| 12 | TCA | Deduction | rotates `candidates` |
| 13 | CDT | Induction | rewrites `candidates` by `temperature` |
| 14 | MCT | Deduction | rewrites `candidates` around the mean |
| 15 | LSI | Induction | writes `sd` from the `candidates` spread |
| 16 | PSO | Deduction | sorts `candidates` |
| 17 | CDI | Revision | pushes a belief tuple; writes `dissonance` |
| 18 | CWS | Revision | pushes the best candidate as a belief tuple |
| 19 | ARE | Abduction | constants only (Demonstration) |
| 20 | TCF | Revision | filters `candidates` |
| 21 | SSR | Revision | reads `confidence`, `free_energy`; no write |
| 22 | ETD | Induction | regroups `candidates` |
| 23 | AMP | Induction | writes `rung` from `free_energy` |
| 24 | ZCF | Deduction | constants only (Demonstration) |
| 25 | HPM | Deduction | rewrites `candidates` |
| 26 | CUR | Deduction | `retain` on `candidates` |
| 27 | MPC | Revision | rewrites `candidates` toward the mean |
| 28 | SSAM | Abduction | reads `sd`; no write |
| 29 | IDR | Deduction | rewrites `candidates` around the max |
| 30 | SPP | Revision | reads `candidates`; no write |
| 31 | ICR | Counterfactual | `free_energy` × `confidence` pole sensitivity; returns a `delta_conf` |
| 32 | SDD | Deduction | `candidates` mean; returns a `delta_conf` |
| 33 | DTMF | Revision | writes `temperature` from the gate state |
| 34 | HKF | Deduction | constants only (Demonstration) |

`run_with` writes `ctx.confidence += delta_conf` for every kernel that runs.

**Consequence.** Under F9, all 34 contract kernels fail the
epistemic-productivity test by construction: their output type cannot carry
a statement. The five planner tactics (4, 6, 7, 8, 11) are the only
implementations that compose statements by their terms with the NARS laws.
They are the starting point for the first vertical slice, not the contract
kernels.

## 3. Certification: what the folds guarantee

`CertificationModel` folds unit masks. Associated, Related, Supports and
CausalCandidate compare outcome rates by cross-multiplication. `Causes`
requires executed randomized arms with at least two `InterventionBacked`
sources, and `observational_certification` cannot return it.

| Requirement | Present? | Note |
|---|---|---|
| Exact joint and conditional populations | yes | intersection, difference, count |
| Positive vs known-negative vs UNKNOWN outcome | **no** | `outcome` is one mask; a unit with no recorded outcome counts as a negative |
| Selection and missingness bias | **no** | no model |
| Effect size | **no** | strict `>` only; any difference certifies |
| Sample size, uncertainty | **no** | a 1-vs-0 split certifies; `MIN_SOURCES = 2` counts attestors, not units |
| Robustness over strata | partly | four declared strata; `clean` mask can only refute |
| Simpson reversal | yes | stops at `Related`; pinned by a fixture |
| Multiple comparisons | **no** | `associated` takes any of up to five populations, with no correction |
| Temporality | partly | `ordered` mask at `CausalCandidate` |
| Source independence | partly | distinct `EvidenceSourceId`s; correlated attestors are not modelled |
| Intervention identification | yes for RCT | executed arms only; no other identification design |

None of these are fixed in this change; C3 owns them.

## 4. What C1/C2 add: `nars::support`

`DerivationLedger` sits beside `BeliefArena` and does not change it. It
appends observations (statement, truth, source) and admitted derivations
(chain, law, premises, conclusion).

**Admission.** `derive` refuses a step whose terms do not fit its law's
figure (`Law::conclusion`), using the figures the planner tactics use:

- deduction: `{M→P, S→M}` in either order;
- induction: `{S→G, S→P} ⊢ G→P`;
- abduction: `{P→M, S→M} ⊢ S→P`;
- analogy: `{S cop P, S↔S′}`.

`admit` records a tactic's `Candidate` directly. The tactics stay the
authority that proposes steps; the ledger keeps what was admitted.

Under any `Ablation` (sources removed, derivations removed) it computes:

- **proofs**: one per surviving observation, one per derivation whose
  premises have evidence. A derivation reads its premises' *pooled* evidence,
  as NARS derives from revised beliefs. A statement may not support itself on
  one path;
- **pooled truth**: the root-disjoint set of proofs with the most total
  evidence weight `Σ c/(1−c)`, revised together. Proofs sharing a root are
  never pooled; roots are exact source ids, not the arena's modulo-64 stamp;
- **chains** that still derive the statement on their own;
- **`load(stmt, x, y)`**, about existence: `FirstCritical`, `SecondCritical`,
  `Synergistic` (each single ablation removes it), `Redundant` (only the joint
  one does), `NotLoadBearing`, `Absent`;
- **`sensitivity(stmt, ablation)`**, about strength: Δfrequency, Δconfidence,
  chains lost, roots lost. A conclusion can survive while its evidence
  collapses; `load` says `NotLoadBearing` and `sensitivity` shows the drop.

**Bounds.** Each refuses rather than return a partial answer:
`Refusal::Steps` (`MAX_STEPS`, work), `Refusal::Depth` (`MAX_DEPTH = 64`,
recursion depth), and `Refusal::Pool` (`MAX_POOL = 20` undominated proofs per
statement for the exact selection). The evidence horizon stays k = 1, as
`TruthValue` fixes it.

Falsifiers covered, each with a disable that turns it red:

| Falsifier | Test | Disable that fails it |
|---|---|---|
| F1 / L7 shared root not double-counted | `f1_shared_root_is_not_double_counted`, `sixty_three_replays_of_one_root_count_once` | revise every proof; drop the dominance filter |
| F2 lost proof ≠ lost conclusion | `f2_a_lost_proof_is_not_a_lost_conclusion` | ignore source ablation; drop the joint check |
| F3 / L5 joint ablation exposes redundancy | `f3_joint_ablation_exposes_overdetermination` | same two |
| L6 survival with collapsed evidence | `l6_a_surviving_conclusion_can_still_lose_evidence` | (exact values asserted) |
| L8 invalid figure not admitted | `l8_an_invalid_figure_is_not_admitted` | skip the figure check |
| figures = tactics | `figures_match_the_planner_tactics`, `tactic_candidates_are_admitted` | — |
| C: most evidence, not strongest proof | `selection_prefers_more_evidence_over_the_single_strongest_proof` | greedy pick by confidence |
| F9 scalar change is not productive | `f9_only_a_grounded_derivation_is_productive` | count ungrounded conclusions |
| F10 contradiction revises; original replays | `f10_a_new_contradiction_revises_and_the_original_replays` | ignore source ablation |
| no self-support | `a_statement_does_not_support_itself` | remove the path guard (stack overflow) |
| bounds | `evaluation_refuses_past_the_step_bound`, `…_depth_bound`, `selection_refuses_past_the_pool_bound` | remove each bound |

**Exact values.**
- F10: the premise 2→3 is revised to `(0.9,0.9) ⊕ (0.05,0.9)` with
  contradiction 0.85. The conclusion is
  `deduction((0.9,0.9), revised) ⊕ deduction((0.9,0.9),(0.9,0.9))`. Chain B's
  independent support keeps the conclusion's frequency at 0.72 instead of
  letting it fall to chain A's 0.43.
- L6: removing source 1 leaves the conclusion derivable through the weaker
  chain B. Its frequency falls by more than 0.3 and its confidence by more
  than 0.5, while `load` reports `NotLoadBearing`.

**Two earlier versions were wrong.**
- The first pooled only at the conclusion. On F10 the counter-evidence sat in
  a proof that shared source 1 with the others and was skipped, so the truth
  did not move. Pooling at each premise fixed it.
- The second pooled greedily by single-proof confidence. That can keep one
  strong proof over roots {1, 2} and drop two independent proofs over {1}
  and {2} that together carry more evidence. The exact selection fixed it.

## 5. Open, with the reason each is open

- **The Epi5 bridge (C3):** relating ledger results, `CertificationModel` and
  `EpistemicState5` (L1–L4: association without causation; `Direct × Causes`;
  `IndirectKnown × Causes`; `IndirectUnknown × Causes`). This goes in a
  follow-up PR, not this one. Load-bearing is not a new bit: the ledger's
  receipts explain why an existing Epi5 reading is warranted; nothing here
  writes one.
- **F4 (Simpson under stratification):** handled by the existing
  certification fixture; not re-tested here.
- **F5 (abductive intermediate stays IndirectUnknown):** C3/C6.
- **F6 (trigger vs mechanism)** and **outcome dependence:** need a
  method/effect model (HIRO ON/WHEN/DO); C4.
- **F7 (model-scoped dependency vs certified cause):** C3/C4.
- **F8 (eligibility differs by topology/certification at one address):** the
  eligibility law is lab-only (`affordance_law.rs`); C3.
- **Cost:** evaluation is not memoized (the path guard makes evidence
  path-dependent); bounded by the three refusals above.
- **Evidence horizon k:** fixed at 1 in `TruthValue` (and `HORIZON = 1` in
  `lance-graph-nal`). Declaring it is a separate change.
