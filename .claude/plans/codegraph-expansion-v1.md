# D-CODEGRAPH-EXPAND-1: what the Codegraph still needs, and who supplies it (plan v1)

> **2026-11-11 · PLAN.** Coordination: Linear project **Codegraph · Collaboration Board** (`P-EXO-18`),
> channel EXO-232. GitHub is canonical; Linear records owners, evidence links and open decisions.
> Follows #1999, #2000 and #2001. Every gap below was measured there; every statement about
> egglog, DialEgg and Kythe is a hypothesis until its slice has read the source.

## 1. Question

After #2001 the Codegraph answers callers, callees and impact over real Charon stores, and says what
it could not see. Which gaps remain, which can be closed by hand-written edges, and where do R2IL,
egglog (mlir-egglog, eggcc, DialEgg) and Kythe fit?

## 2. Answer in one table

| gap | measured | kind of fix | supplier | slice |
|---|---|---|---|---|
| closure bodies have no edge from their parent | 96 in foundry-parity-lab; caused a missed division | derived edge, graded POSSIBLE | Codegraph crate | CG-1 |
| consumer crates not in the store | 13 of 14 consumers of `admit` | extract dependents | extraction driver | CG-2 |
| test targets and examples not extracted | 48 test sites for `LaneBinding::check`, none resolved | extract those targets | extraction driver | CG-2 |
| wrong Cargo target extracted | `bench` binary instead of the library | name the target; residuals are a refusal | extraction driver | CG-2 |
| dependents list built by hand | a wrong-column list removed a real gap | take it from `cargo metadata` | extraction driver | CG-2 |
| store does not record its revision | witnesses stamped with the wrong revision | provenance in the store | ruff_charon | CG-3 |
| absolute paths in the store | same input, different digest at another target dir | paths relative to `OUT_DIR` | ruff_charon | CG-3 |
| Charon error inside a crate | r2il is PARTIAL on a `dyn Fn` type alias | minimal case, upstream or typed gap | Charon | CG-3 |
| effects are extracted but not queryable | 213 effect rows in the foundry store; the Codegraph reads none | reader in the Codegraph | Codegraph crate | CG-8 |
| target kind, cfg and feature are not rows | "example-only" was read from `Cargo.toml` by hand; 32 inactive regions skipped | hand-rolled rows | `cargo metadata`; cfg source undecided | CG-8 |
| no def-use, no loop multiplicity | "the divisor is `per_staff`" was read by eye | new row kind from LLBC | ruff_charon | CG-5 |
| edge names are local | no shared vocabulary with other tools | mapping to Kythe edge kinds | document | CG-4 |
| what a body means | divisions classified by a source-span regex | rules over a lowered body | R2IL + egglog, unproven | CG-5 |
| review evidence is partial | mask-risc read as source; corrected review not re-graded | extract, test, re-grade | review report | CG-6 |
| one repository only | no held-out manual review yet | second repository | report-rs #33 | CG-7 |

**The split that decides everything else:** extraction gaps (CG-1 to CG-3) are closed by extractors
and drivers. An e-graph proves that two expressions are equivalent; it cannot supply a call edge that
was never emitted. So egglog is a candidate for body semantics (CG-5) only.

## 3. Slices

| slice | Linear | owner repo | depends on | gate |
|---|---|---|---|---|
| CG-1 closure-ownership edge | EXO-233 | lance-graph | #2001; a decision on mask vs separate relation | `can_cook_guard` impact includes `methods` as POSSIBLE; closure-free twin gains nothing; disable run red |
| CG-2 extraction driver | EXO-234 | lance-graph, uses ruff_charon | #2001 | `admit` reports no `ConsumersNotInStore`; residuals or a non-COMPLETE crate give PARTIAL, never empty success |
| CG-3 store provenance | EXO-235 | ruff, checked in lance-graph | a format decision | two extractions at different target dirs are byte-identical; a store at another revision is refused |
| CG-4 Kythe edge vocabulary | EXO-236 | lance-graph, document | CG-1 | every row cites the Kythe schema and a Codegraph source line |
| CG-5 body semantics, co-research | EXO-237 | research | operator go-ahead | one rule classifies the two ceiling divisions and the oracle quotient of #422; disable run red |
| CG-6 review report follow-ups | EXO-238 | lance-graph; crai-rs owns semantics | #2001 | mask-risc store exists; builder has negative controls; two operator decisions recorded |
| CG-7 report-rs #33 transfer | EXO-239 (record: EXO-185) | report-rs session | #2001; CG-2 if needed | strict grading against the held-out manual review |
| CG-8 effects, target kind, cfg as rows | EXO-241 | lance-graph | #2001; shares `cargo metadata` with CG-2 | "which divisions does `max_quantities` reach" is answered inside the Codegraph crate; a missing effects file is a gap |

Handover prompt for the CG-7 session: EXO-240.

**Order.** CG-1, CG-2 and CG-8 first; they are small, measured and independent. CG-7 can start now from the
#2001 branch. CG-3 waits for a format decision. CG-4 is a document and can run beside anything. CG-5
needs a go-ahead and is research before it is a build.

## 3a. The CG-5 probe (proposed by the report-rs session, checked 2026-11-11)

Recover the ndarray #422 answer with no manual reading:

1. target-kind and cfg rows (CG-8);
2. def-use rows from LLBC through ruff_charon (a new row kind, so it needs the CG-3 format decision);
3. a small egglog rule pack: reachable from the entry point, contains a division, divisor traces to
   `per_staff`.

**Pass:** it reports `arm_b.rs:245` and `method.rs:58`, each with a proof.
**Fail:** removing the division in a scratch copy leaves the answer unchanged.

Checked against source before adoption:
- **Confirmed.** `mlir-egglog` here is a Python toy compiler for NumPy expressions; DialEgg is not
  checked out; `egglog` and `eggcc` are. `crates/egglog-r2il-probe` exists. `KYTHE.md` records 24
  agreeing symbols and no `ref/call` edges. The divisor is `per_staff: 4 + (m as i32 % 5)`
  (`fixture.rs:105`).
- **Corrected.** The store has ten predicate codes, not six. Division sites needed no token scan:
  ruff_charon already emits them as `ASSERT DivisionByZero` effects, and what is missing is a reader
  (CG-8). Charon is built on this machine but is not on `PATH`.
- **Agreed.** No MLIR in the path: Rust does not lower to it, so it would be a third representation.
  What transfers from DialEgg is the pattern, one egglog term per IR operation, applied to LLBC.
  egglog derives and proves; it is a cold oracle arm, never a per-query path, because the Codegraph
  already computes closure over masks.

## 4. Mapping to plans and issues that already exist

This plan adds no second Codegraph programme. It supplies measured inputs to these:

| existing | what it already owns | relation to this plan |
|---|---|---|
| D-CODEGRAPH-0 (`d-codegraph-0-resident-codegraph-v1.md`, EXO-229) | Charon → ruff → resident rows, queries | CG-1 and CG-2 are its next increments |
| D-CODEGRAPH-0-BIN (`d-codegraph-0-binary-lanes-v1.md`) | binary lanes, a provenance header, per-row `(artifact, FunDeclId)` | CG-3 is the text-store half of the same provenance question; decide them together |
| D-CODEGRAPH-0-D | a local stdio MCP over the queries | not in this plan; `cg_query` is its CLI precursor |
| D-CGMM-0 (`d-cgmm-*.md`, EXO-155) | Kythe-shaped immutable evidence, lazy demand, FieldViewDef, DT64 binding | CG-4 feeds D-CGMM-1 (source facts); CG-1 feeds D-CGMM-8 (evidence and call graph) |
| EXO-85, EXO-84 | MeTTa / Datalog assimilation, native polyfill | CG-5 must not duplicate these; it asks one narrower question |
| EXO-73, EXO-101 | recursion driver, datafrog parity | unaffected; the Codegraph closure is not a new recursion driver |
| D-CMR-8 (`d-cmr-concept-chain-normalization-v1.md`) | equality-saturation probe, `egg` / `egglog` as references only | CG-5 reuses its stance: reference and probe, not a runtime |
| `ascent-byods-dr-fold-polyfill-v1.md`, entry 2026-10-21 | earlier evaluation of egglog, datafrog and Ascent | CG-5 starts from that evaluation |
| D-FOUNDRY-PARITY-1 (EXO-227) | G1 to G6 | CG-6 carries the open #422 verification |
| EXO-230 | navigation and the review evidence packet | CG-6 |
| EXO-185 | report-rs #33 | CG-7 |

## 5. Constraints

- No new graph database, no second Rust parser, no blocking CI reviewer.
- These change only by a decision issue in `P-EXO-11`: the Charon and ruff_charon artifact formats,
  the lance-graph-contract / ClassView API, DT64 admission, Quack / mask-RISC execution, XLOG
  semantics, crai-rs review semantics.
- A name-derived edge is POSSIBLE, never CONFIRMED.
- Nothing from CG-5 enters a per-query path. Cold costs are reported in microseconds and are not gates.
- Every guard has a test that fails when the guard is disabled.

## 6. Open decisions

1. May a name-derived closure edge enter `impact`'s masks, or is it a separate relation (CG-1)?
2. May the ruff_charon store format gain a revision and relative paths, and is that decided together
   with D-CODEGRAPH-0-BIN's provenance header (CG-3)?
3. Go-ahead for the CG-5 co-research pass.
4. Where the review report generator lives, and whether the advisory CI job is built (CG-6).
