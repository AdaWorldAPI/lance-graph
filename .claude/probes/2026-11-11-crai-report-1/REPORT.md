# D-CRAI-REPORT-1 and D-CODEGRAPH-RECON-2 (Gate A): Codegraph evidence as review reports

**Status:** probe, advisory only. Board: EXO-229 (intake), EXO-230 (navigation). report-rs #33 /
EXO-185 (RECON-2 Gate B) moved to another session; this one covers #2000's follow-up, coverage-aware
impact, the report generator, and the ndarray #422 admission case.

Nothing here posts PR comments, applies fixes, uploads anything or blocks CI.

## 1. What landed

| piece | where | what it does |
|---|---|---|
| coverage-aware impact | `crates/lance-graph-codegraph/src/impact_report.rs` | `Codegraph::impact_report(symbol, &ExtractionScope)` → `ImpactAnswer`: the unchanged `Codegraph::impact` mask plus typed `CoverageGap`s |
| CLI | `cg_query … impact-report SYMBOL [--deps a,b] [--lib-only] [--no-freshness]` | the same, for agents |
| report generator | `crai_report.py` | review input → `review.md` + `review.ndjson` |
| #422 evidence builder | `build_q8_evidence.py` | git + Cargo + Charon store → review input |
| examples | `examples/lance-graph-2000/`, `examples/ndarray-422/` | inputs and rendered reports |

### Coverage-aware impact (RECON-2 Gate A)

`Codegraph::impact` keeps its position-mask semantics (Decision 1). A caller in an unloaded crate has
no position, so it is reported as missing coverage and never as an entry in `Graded::unknown`. Facts the
rows cannot carry are declared by the caller in `ExtractionScope`:
- which targets were extracted;
- which crates depend on the symbol's crate;
- where the source is, for freshness.

Anything not declared becomes a gap, never an assumption.

| case | test (`tests/impact_coverage.rs`, on real `rev1` Charon facts) | disable run |
|---|---|---|
| no callers in a fully observed scope → no gap | `no_callers_inside_a_fully_observed_scope_has_no_gap` | — (silence twin) |
| public symbol, consumers not in store / not declared | `a_public_symbol_with_unobserved_consumers_reports_missing_coverage` | red (both gaps) |
| library-only extraction → test callers unobserved | `library_only_extraction_reports_unobserved_test_callers` | red |
| callers only through dispatch | `dispatch_only_callers_report_unresolved_dispatch` | red |
| changed source / freshness unchecked | `a_changed_source_file_reports_stale_coverage` | red (both) |
| symbol absent from the store | `a_symbol_absent_from_the_store_is_not_an_empty_impact` | red |
| consumer crate added: gap → CONFIRMED | `adding_the_consumer_crate_upgrades_missing_coverage_to_confirmed` | red |

`ImpactReport::no_known_gap()` is documented as "no known gap within the declared scope", never as
completeness. That matches `diag::Coverage::no_known_gap`.

One test premise was wrong and was corrected, not the code: `util::double` is not dispatch-free. It is
called by `Square::area`, which `area_of` reaches through dispatch, so `area_of` is a legitimate possible
impact of `double`. The silence twin uses `util::triple`.

### The report generator (Gates 1 and 3)

Records keep CodeRabbit's public `--agent` fields (`fileName`, `comment`, `severity`, `suggestions`,
`codegenInstructions`; crai-rs CR-SRC-01) and add crai-rs's five separate axes.

**The generator computes, and never accepts from its input:**
- the evidence class: the highest grade among witnesses current at their own repository's head;
- the finding status (`stale` when a witness is old);
- whether the review is clean.

`TEST_OBSERVED` needs a falsifier recorded as red under a disable run and then green.

| Gate 3 control (and crai-rs id) | test | disable run |
|---|---|---|
| missing consumer crates → UNKNOWN, never "unaffected" | `test_missing_consumer_crates_yield_unknown_not_unaffected` | red |
| stale packet → no current-head finding (RS-F04) | `test_stale_packet_cannot_produce_a_current_head_finding` | red |
| evidence current only at its own repo's head | `test_cross_repo_evidence_is_current_only_at_its_own_head` | red |
| PARTIAL / SKIPPED / NOT_RUN / FAILED with 0 findings never clean (RS-F03) | `test_partial_or_skipped_with_zero_findings_is_never_clean` | red |
| named test without an executed falsifier stays NOT_RUN (RS-F05) | `test_a_named_test_without_an_executed_falsifier_stays_not_run` | red |
| compiler evidence vs reviewer hypothesis keep their grades; confidence never from severity (RS-F01/F02) | `test_compiler_evidence_and_reviewer_hypotheses_keep_separate_grades` | red |
| irrelevant benchmark does not become a defect | `test_an_irrelevant_benchmark_does_not_become_a_defect` | red |
| repeated input → byte-identical reports | `test_repeated_input_regenerates_byte_identical_reports` | red |
| POSSIBLE never a confirmed caller (RS-F07) | `test_possible_dispatch_is_never_a_confirmed_caller` | red |
| repetition is not a second witness (RS-F08) | `test_repetition_is_not_a_second_witness` | red |
| agent text is quoted, never structure (RS-F09) | `test_agent_text_is_quoted_never_structure` | red |
| withdrawn keeps its record (RS-F10) | `test_withdrawn_keeps_its_record` | red |

On real data:
- **#2000:** the packet regenerated at the merged head gives F1 `open`/COMPILER_DERIVED and F2
  `open`/SOURCE_STRUCTURE. The packet committed from an older head gives both `stale`/ASSERTED, with no
  coverage claim.
- **Reproducibility:** both examples regenerate byte-identical from their inputs.

Placement: the generator lives beside the packet in lance-graph, not in crai-rs, although crai-rs owns
the review semantics (its §7 owner table). crai-rs declares itself CR-H0 "documentation and
preregistered falsification only … not a … checker, runtime, PR reviewer". Moving executable code there
changes that phase, so it is the owner's decision.

## 2. ndarray #422 (Gate 2): does the quotient optimisation reach the Foundry consumer?

**Generated finding, from tools only:** no.
- **The change:** #422's Rust changes are confined to the `u64x64` example target
  (`[[example]] name = "u64x64"`). No library Rust file changed, and no crate can depend on an example.
- **G1 (`max_quantities`, `network_capacity`):** reaches a single division, the scalar ceiling
  `(q + per_staff - 1) / per_staff` in `cook_ops`. It is the threshold operand of a mask predicate and
  names no lane.
- **G6 (`execute_can_cook`, `PreparedMethod::{prepare, run}`):** reaches no division at all.
- **The stock quotient `stock / d`:** appears only in the oracle (`oracle.rs:62`) and the materialised
  null model (`arm_a.rs:75`), reached as POSSIBLE through closure bodies.
- **A narrower point the PR's own spec gets wrong:** "No quotient is evaluated on the executable path"
  is too strong, because a scalar ceiling division builds the staff threshold. It should say "no
  per-row stock quotient".

How the evidence was made trustworthy:
- **Charon first translated the wrong target.** `foundry-parity-lab` has a library and a `bench`
  binary, and `charon cargo` picked the binary. The library came through as opaque dependency items:
  47 `BODY_NOT_STRUCTURED` residuals and zero calls. Read naively, that store says "the consumer calls
  nothing, so nothing is reached", which is the false negative this work exists to prevent. The
  residuals flagged it. Re-run with `-- --lib`: 1,636 facts, 928 resolved calls, 0 residuals.
- **Divisions come from Charon's assertions.** Each integer division carries a `DivisionByZero` and an
  `Overflow:Div` effect at its exact span, so "which divisions does this path reach" is a compiler
  fact, not a text search.
- **The threshold check is mechanical.** The builder reads the `t:` operands inside `cook_ops`'s
  definition span (SOURCE_STRUCTURE) and checks that each executable division sits in one that names
  no lane.

Grading against the held-out manual review: §5.

## 3. Corrections to RECON-1 (#2000)

- **r2il was not clean.** `lance_graph_r2il`'s CRATE row is `PARTIAL`: Charon reported an error on
  `pub type Select<'s> = dyn Fn(…)`. The RECON-1 report and the EXO-229 comment said the four crates ran
  clean with one residual. The residual count was right, and the crate status was not. `impact_report`
  now reports it as `PartialCrate`.
- **The reproduction record is not on main.** `c7151a1` was pushed after #2000 merged at `90fc5a7`, so
  it is carried on this branch.

## 4. Gap matrix: what Codegraph still cannot prove

| gap | effect on an answer | measured where | what would close it |
|---|---|---|---|
| crates not in the store | external callers have no position | dt64: 261/370 public symbols; `admit`: 13 consumers missing | one Charon artifact per consumer (EXO-229) |
| test targets, `#[cfg(test)]` modules, examples | test callers unobserved | `LaneBinding::check`: 48 name-matched test sites, none resolved | Charon/RA on test targets |
| calls inside closures passed to std adapters | parent→closure edge missing | foundry-parity-lab: 56 closure bodies linked only by name | link by Charon's naming as POSSIBLE (done in the #422 builder, not in Codegraph) |
| macro arguments (RA) | call rows missing; region marked | `lane_socket.rs`: 7/7 sites | Charon (LLBC sees through macros) |
| wrong Cargo target extracted | an apparently empty crate | foundry-parity-lab picked `bench` | name the target; treat `BODY_NOT_STRUCTURED` as refusal |
| Charon errors inside a crate | `PARTIAL` CRATE, zero residuals | r2il (`dyn Fn` type alias) | read CRATE status, not only residuals |
| declared scope is trusted | a wrong dependents list removes a real gap | my own wrong-column `--deps` gave "no consumer gap" | dependents from Cargo metadata, not a caller-built list |
| store location-dependence | identical content, different digests | contract build-script outputs, LLBC path | relative `OUT_DIR` paths in ruff_charon |
| semantics of a body | no BODY_SEMANTICS beyond R2IL bindings | Rust functions are `NOT_A_METHOD` | out of scope |
| production data | no recipe distribution | #422: PRODUCTION_DISTRIBUTION = UNKNOWN | not a code fact |
| performance | no timing evidence | #422 benchmarks: ASSERTED | executed benchmark with disable arms |

## 5. Grading (Gate 4)

Filled from the independent grader below.

## 6. Recommendation: advisory GitHub CI

Not built in this slice. If you want it:

1. A `workflow_dispatch`-only job, plus an opt-in label. It runs Charon on the crates a PR touches,
   their direct dependents, and test targets. It renders `review.md` / `review.ndjson` as an artifact.
2. **No comment, no status check, no blocking.** The run status is printed in the summary, and
   PARTIAL is the honest default.
3. **Refuse rather than report empty:** a crate with `BODY_NOT_STRUCTURED` residuals, a non-COMPLETE
   CRATE row, or a stale `files.tsv` makes the job say `PARTIAL` with the reason, never `clean`.
4. **Cost** (measured here): Charon 2–29 s per crate (contract 21 s, dt64 29 s, foundry lib 2 s once
   built); ruff_charon import ≤ 7 s; impact/report < 1 s. RA over the whole workspace takes 108 s and
   should run only when macro-region coverage is wanted.
5. Promote to a PR comment only after a held-out precision measurement over several real PRs. One
   graded case is an indication, not a calibration.
