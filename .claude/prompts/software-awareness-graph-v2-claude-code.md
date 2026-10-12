# Claude Code work prompt — D-SAG-2: falsify, observe, predict, revise

**Base:** lance-graph main at merged #1509 (42fa2a347cda22eaaa327bba514f2029cc9daa82). Read the plan `../plans/software-awareness-graph-v2-ruff-ona-closed-loop.md`, the evidence contract `../knowledge/software-awareness-provenance-parity.md`, and the primary-source shelf `../../.arxiv/2026-10_software_awareness_ona_ruff_research_excerpts.md` before editing code.

## Mission

Make the existing SoftwareGraph's predictions **independently falsifiable**, then determine experimentally whether feedback from Ruff or a runtime oracle can improve a subsequent prediction or decision. Preserve one Quack → mask-risc execution substrate; R2IL remains an alternate lowering frontend; OGAR remains sole authority for external effects. Treat #1496–#1501 as design/research and #1503–#1509 as partial implemented foundations. No second graph engine or ontology.

## Read-first files

- lance-graph: crates/lance-graph-glove-parity/src/{awareness,method,decl,gremlin}.rs; tests/{awareness,compose,declared,gremlin}.rs; examples/awareness_loop.rs.
- lance-graph: crates/lance-graph-quack/src/{bind,lib}.rs; crates/lance-graph-r2il/src/procs.rs; crates/lance-graph-contract/src/{class_view,hotplug}.rs; crates/lance-graph-dir-sim/src/{actors,view}.rs.
- Ruff: crates/ruff_spo_triplet/src/{ir,expand,reassemble,triple}.rs; crates/ruff_{python,ruby,cpp,csharp}_spo; crates/ruff_r2il; .claude/OGAR-POLYGLOT-AST-INTEGRATION.md.
- Existing plans/primary excerpts in PRs #1496, #1497, #1498, #1499, #1500, #1501, #1508 and #1509. Search actual code before asserting a function exists or a gate shipped.

## Order of execution

1. **Inventory on exact revisions.** Obtain repository HEAD SHA and each dependency revision; mark contract shipped/tested, research-only, or unknown. The initial plan uses #1509's merge; if main advanced, reconcile rather than silently substitute the branch.
2. **Try to refute three audit findings first.** Write small tests in existing glove-parity test modules:
   - link() shared-edge ctx/provenance dedup versus partial reproject;
   - project_compute() replacement across changed class DAG and cycle status;
   - caller-issued Certificate with forged test name accepted as Verified.
   For each: record failing reproduction or evidence that the claim was false. Do not implement fixes until the falsifier is real. Ensure that a historical receipt and a current projection are different concepts.
3. **Define the minimal receipt contract.** Exact source revision, snapshot, evidence source key, derivation, observed relation, method key, schema seal, reading, universe, oracle result, environment fingerprint. Distinguish verified test from user annotation and compiler verdict; a note must never promote itself.
4. **Ruff side, in read-only adapter/probe only.** Pin Ruff SHA. Select minimum supported fixtures:
   - Odoo model and parent/extension view (`inherit_id`);
   - Ruby model with same unordered body-facts but different ordered MethodOre;
   - C file-scope `static` versus C++ static method, overloaded C++ methods;
   - explicit unsupported round-trip (core-7) and unprojected free-function calls.
   Build an exact crosswalk from Ruff identifiers/provenance to SAG identifiers or emit typed residuals. Do not infer universal cross-language equality.
5. **Impact forecasting.** Freeze A and produce a machine-readable PredictionReceipt **before** altering a fixture; edit to B in a separate commit; reharvest blind. Compare predicted and observed deltas with TP/FP/FN/Unknown (only on supported complete universe) and source-span evidence.
6. **One feedback mechanism, not a reasoning empire.** Choose the smallest revisable hypothesis over candidate impact/unknown, update it only from independently stamped outcomes. Repeat a paired case with evidence feedback disabled, same budget and candidate order. If the chosen subsequent action doesn't change or doesn't improve the predeclared target, report that; do not claim learning.
7. **Only after above gates:** compare scheduler variants (FIFO, confidence, Shannon, EIG, Moore/Staunen), evidence overlap/fission, and Rubicon commitment/rollback. Separate scheduling from inference law and permission checks. Keep effect execution unimplemented unless operator authorizes a future PR.
8. **H1 optimization stays a separate experiment.** Existing 1.59–1.78× scalar cost gap is not improvement. Benchmark a real fused implementation only if separately approved after correctness and NULL/UNKNOWN controls; publish native/v3 tiers.

## Concrete acceptance tests

- P01: shared field claim has two independent ownership witnesses; reprojecting one doesn't erase the other's claim.
- P02: compute dependency/cycle disappears in current projection at B, while history (if kept) is explicitly versioned.
- P03: forged run metadata cannot create independent Verified proof.
- P04–P09: class/seal/reading identity, dependent stamps, pinned snapshot, Ruby event order, C semantic distinction, restricted Ruff round-trip.
- P10–P13: Odoo view impact, explicit unknown, typed NoRelationCarrier, full 256-position DAG.
- P14–P16: preregistered forecast, failed-feedback ablation, irrelevant entropy distractors.
- P17–P20: Undetermined role cannot authorize, H1 mixed-workload/NULL counterexample, Hyperon four-branch cardinality, static ≠ runtime behavior.
- Every guard fixed must have disable-red-run evidence. A passing assertion against handwritten expected fixtures alone is not an independent oracle.

## Mandatory outputs from the coding session

Provide: exact files changed; design decision ledger; list of falsifiers with red/green and counterexamples; fixture and oracle identifiers; CI commands and *actual* outcomes; measured cost per variant; unresolved typed gaps; a source/revision table; and either a narrowly scoped implementation PR (only the passed wave) or a clear blocker report.

Track predictions and observed outcomes separately, with timestamps or pinned revisions. Permit the correct conclusion to be "no improvement", "Unknown", or "this layer cannot express that relation". Keep the register/ISA bytes unchanged; any new semantic reading or method authority requires a separately reviewed contract.

## PR hygiene

- No editing the generated board index by hand; if adding a board entry run .claude/tools/entries_index.py --write and --check, retaining the curated finding column.
- Run plan_dids.py, append_only_gate.py and relevant repository CI gates as applicable; do not claim they passed unless executed.
- Avoid status-board inflation: distinguish shipped, measured locally, untested, proposed and blocked.
- Do not auto-merge, grant permissions, modify policy, deploy or overwrite source code as a side effect of an awareness conclusion.

## Required deferred-probe supplement (read before implementation)

Read `../knowledge/sag-v2-observability-causality-and-evaluation-addendum.md` and `../../.arxiv/2026-10_software_awareness_ona_observability_metamorphic_holdout_excerpts.md`; they add P21–P40 to the previous P01–P20. In particular:

- Refutation by missing observation is legal **only** when Ruff/runtime collector positively certifies predicate support, scope completeness, matching frozen revision, successful scan and closed observation interval. Otherwise return `NotObservable`, `Incomplete`, `Pending` or `Stale`. ONA's published *Assumption of Failure* is a research mechanism, not a license for closed-world Ruff assumptions.
- Distinguish `NoOp` causal control, read-only `Probe` (it consumes resources), and inert `ProposedMethod` before any OGAR-gated action. Try A→A no-op, semantic-preserving A→B, real change A→B, and A→B→A' revert.
- Obtain preregistered predicted deltas before editing. Compare full versus incremental reharvest; keep one evidence stamp for multiple pipeline copies of the same source fact; test independent-source twins.
- Use development / validation / protected holdout **grouped by repository and edit family**, not randomly selected lines, and count oracle reuse. On frozen holdout test no-feedback, correct-feedback, shuffled-feedback, and no-op arms with identical budget; record null or negative results honestly.
- Bound self-observation recursion: a Quack query sees snapshot k, its telemetry writes k+1; no query in the same epoch may certify itself.
- Preserve separate source-revision/event-time/ingestion-time/decision-time and treat late receipts or partial coverage explicitly. Never turn attention score, CARE heuristic or popcount of unknown cells into truth/authorization or alter CE64's byte/ISA contract.

Run the original P01–P20 gates before claiming D-SAG-2 success. The deferred P21–P40 tests are additional hypotheses, not claims of implementation.

