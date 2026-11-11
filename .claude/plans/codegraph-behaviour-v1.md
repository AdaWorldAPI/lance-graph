# D-CODEGRAPH-BEHAVIOR-1: from changed code to registered behaviour (plan v1)

> **2026-11-11 · PLAN.** Ruling: D-CODEGRAPH-BEHAVIOR-RULING-1 (operator, 2026-11-11).
> Coordination: Linear project **Codegraph · Collaboration Board** (`P-EXO-18`), channel EXO-232.
> GitHub is canonical. This plan adds no slice to that board and no owner: it maps proposed stages
> onto existing owners and defines one read-only experiment, CG-BEH-1.
> It extends `codegraph-expansion-v1.md` (PR #2001, slices CG-1 to CG-8) and does not replace it.

## 1. Question

A function changes. Which **registered behaviours** may change with it, on what evidence, and
what is not known?

Today the Codegraph answers "which functions call it". DT64 knows the lenses, their words and the
methods that use them. Nothing joins the two.

## 2. Ownership (ruled; no competing owner)

| concern | owner | issue |
|---|---|---|
| Codegraph → behaviour-lens declarations → `MethodBinding` | D-CGMM-0 | EXO-155 |
| `ClassView` contract, including value-schema presets | D-QR-CAP | EXO-121 |
| effects reader (`effects.tsv`), target kind, cfg rows | CG-8 | EXO-241 |
| extraction targets and dependents | CG-2 | EXO-234 |
| store provenance format | CG-3 | EXO-235 |
| body semantics through egglog (bounded, read-only, go-ahead given) | CG-5 | EXO-237 |
| review report follow-ups; crai-rs owns review semantics | CG-6 | EXO-238 |
| report-rs #33 transfer | CG-7 | EXO-239, EXO-185 |
| Rails / Odoo / SAP business parity | D-XGP | EXO-95 |

## 3. Crosswalk of the nine proposed stages

The stages N-1 to N-9 were proposed on 2026-11-11 and are **not** filed as issues.
Each is a stage of an existing owner.

| stage | what | lands under | note |
|---|---|---|---|
| N-1 | SPO triples as graded store rows | EXO-155, stage CGMM-1 (source facts) | name-matched rows are `SOURCE_STRUCTURE + POSSIBLE` |
| N-2 | Python import rows from `ruff_graph` | EXO-155, CGMM-1 | Odoo only; file-level |
| N-3 | corpus conventions and residual rows (ore / config / slag) | EXO-155, CGMM-1 and CGMM-4 | §6 below |
| N-4 | more frontends (plain C, PHP, Go, TypeScript) | parked | after N-1 is proven on one corpus |
| N-5 | code → algebra word | **CG-BEH-1**, under EXO-155 (CGMM-1 / CGMM-8) | §5 below |
| N-6 | method ON × WHEN × DO rows from registered bindings | **CG-BEH-1**, under EXO-155 (CGMM-6 reads) | DO only in v0; see §5.4 |
| N-7 | impact through behaviour | **CG-BEH-1**; consumer: CG-6 review report | |
| N-8 | harvested methods into candidate ON / WHEN / DO | EXO-155, CGMM-3 to CGMM-5; EXO-95 | §7 below; precedent EXO-210 |
| N-9 | value-schema preset per harvested class | EXO-121 | a representation choice, never a behaviour claim |

A new Linear issue for CG-BEH-1 is proposed only if EXO-155's stages cannot carry its gate
(§5.6) as an independently reviewed deliverable.

## 4. Wave 0: one coverage-aware impact API (PR #2001 and PR #2002)

Both PRs are open against the same base and both add `Codegraph::impact_report` in
`crates/lance-graph-codegraph/src/impact_report.rs`, with different types. Neither may be merged
over the other. This section is a proposal for the two owning sessions; nothing here was pushed
to either branch.

### 4.1 What each has that the other lacks

| | PR #2001 (`26bf53e`) | PR #2002 (`e436b0c`) |
|---|---|---|
| input | `ExtractionScope<'a> { targets: Targets, dependents: Dependents<'a>, sources: Option<(&Path, &Path)> }` | `Scope { store, source_root, targets: Vec<(String, Targets)>, consumers_closed: bool, excluded: Vec<(String, Vec<String>)> }` |
| targets | one value for the whole store: `LibraryOnly` / `LibraryAndTests` | per crate: `{ lib, tests, examples, benches }` |
| consumers | `Dependents::NotDeclared` / `Declared(&[&str])`; a declared crate missing from the store is named (`ConsumersNotInStore`) | `consumers_closed: bool`; when closed, every stored crate's coverage is checked |
| result | `ImpactAnswer::Found(ImpactReport)` / `SymbolNotInStore { symbol, observed }` | `ImpactReport { graded, symbol: SymbolState, gaps }`, so a missing symbol still carries its crate's gaps |
| only here | trait-dispatch gap (`possible`) separate from dynamic calls; `MissingSource` separate from `StaleSource`; `InvalidFileList`; a crate without a CRATE row; a symbol whose rows a refresh removed; `observed` crate list with run status | `SourceFilesUnread`; `RegionsExcluded`; `TargetsUndeclared`; a private symbol behind a public caller; a missing symbol's target and harvest gaps; `owner_stored` |
| tests | 12 | 14 |
| consumers of the output | `crai_report.py`, `build_q8_evidence.py`, `cg_query impact-report` | the report-rs #33 probe, `cg_query report` |

**One name, two meanings.** `UnresolvedDispatch` means trait dispatch the compiler did not resolve
(`possible`) in #2001, and dynamic calls (`UNRESOLVED_CALL`) in #2002. #2001 calls the second
`DynamicCalls`. A merged enum must not keep the shared name for either.

### 4.2 Proposed single shape

- **Result:** #2002's struct (`graded`, `symbol`, `gaps`), plus #2001's `observed: Vec<(String, RunStatus)>`.
  A missing symbol then carries gaps, and the caller still sees which crates were read.
- **Targets:** #2002's per-crate struct. `LibraryOnly` is `Targets::LIB_ONLY`.
- **Consumers:** `Consumers::NotDeclared | Declared(Vec<String>)`. Declared and all present in the
  store is #2002's "closed" case and runs its per-crate check. A declared crate that is absent is
  #2001's `ConsumersNotInStore`, by name.
- **Excluded regions and unread files:** from #2002, unchanged.
- **Gaps:** the union, with the clash renamed: `TraitDispatchUnresolved { possible }` and
  `DynamicCalls { unknown }`. `StaleSource` and `MissingSource` stay separate. The harvest gap keeps
  #2002's one variant with its three causes, or #2001's three variants; pick one and map the other's tests.
- **Claims:** `no_known_gap()` (no gap) and `closed_in_scope()` (found and no gap) both stay, both
  documented as local to the declared scope.
- **`cg_query`:** one subcommand, `impact-report`, with the union of flags
  (`--deps`, `--targets CRATE=…`, `--exclude CRATE=REGION`, `--no-freshness`).

### 4.3 Gate for Wave 0

- Every one of the 26 existing tests passes against the one API, or is merged with its twin and the
  merge is listed. The seven RECON-2 cases exist in both suites.
- Each guard still has a disable run that goes red, including #2002's two directions for unread files.
- `crai_report.py` and the report-rs #33 probe both run on the unified output and regenerate their
  committed examples byte-identical, or the diff is explained line by line.
- Procedure: a third branch from `main`. Neither PR is rebased onto the other first.

## 5. CG-BEH-1: symbol → algebra word → ordered method uses → classid

Read-only. No new word, no new classid, no new binding, no R2IL preparation, no execution.

### 5.1 What exists (read 2026-11-11 at `317c4ffc0`)

- **The lens.** `POLYFILL_LENS = 0xFFFF_F000` (`lance-graph-r2il/src/polyfill.rs`). Its words are
  `POLYFILL: [FrozenWord; 17]`; word `p` is the entry at index `p`. Each entry binds an R2IL opcode,
  an arity and the aliases a view may use (R2IL, SQL and Quack names). The opcode, arity and R2IL
  name are checked against OGAR's vocabulary by test.
- **The dispatch.** One function, `FoldDialect::call` (`lance-graph-r2il/src/lib.rs`), matches on
  the opcode. `binding::EXECUTABLE` lists the same 17 opcodes in the same order.
- **A method's uses.** `MethodBinding::effects` is an ordered list; each step names one
  `AlgebraRef { lens, word }`, a target field and ordered arguments. `BindStore::admit` returns the
  stored binding; `words_used` and `users_of` index the same facts.
- **What does not exist.** No production crate registers a binding with DO effects. Foundry's
  `can_cook_method` has `effects: Vec::new()`: it is ON + WHEN only. The bindings with ordered
  effects are in the r2il and dt64 integration tests. The Rails scope bindings of EXO-210 live in
  another repository.

### 5.2 Two relations, three kinds of evidence

| relation | read from | evidence |
|---|---|---|
| `symbol → realizes → opcode` | the match arms of `FoldDialect::call` | source structure. Not a compiler call fact, and not a behavioural test |
| `opcode → word` | the `POLYFILL` table, by position | source structure, cross-checked against `EXECUTABLE` |
| `method → uses[ordinal] → word` | `BindStore::admit` in a test run | DT64 registration |

The report prints the line where source evidence ends and registration evidence begins.

**Why `realizes` is graded this way.** A match arm shows that the function contains the code that
runs for that opcode. It does not show that the code computes what the word declares. That needs a
behavioural witness per word. The r2il integration tests compare whole method answers with a
scalar oracle; a mapping from those tests to single words is not built here. A function whose name
matches a word is never an edge.

**Granularity.** One symbol dispatches every opcode, so at symbol level a change to
`FoldDialect::call` is a candidate change to all 17 words. Only the arm's line span narrows it.
A compiler call graph cannot narrow it at all: Charon sees one function.

### 5.3 The fixture

- `crates/lance-graph-r2il/tests/integration/polyfill.rs`: two tests that read the ordered uses of
  two existing test bindings through `method_ref` and `admit`, cross-check `words_used` and
  `users_of`, and export the rows.
- `.claude/probes/2026-11-11-cg-beh-1/beh.py`: `words`, `dispatch` and `impact` over the source
  tree and the exported rows; `--self-test` and `disable_runs.py` for the negative controls.

### 5.4 What v0 leaves out, on purpose

- **ON and WHEN rows.** The binding carries `on` and the guard witness; v0 reads DO only, because
  the algebra join needs only DO. ON and WHEN stay separate relations when they are added.
- **The native path.** `binding_native::prepare_native` lowers steps to mask-risc programs. That
  second realization is not mapped; the report lists it as a residual.
- **mask-risc and `ndarray::simd`.** The words' implementations below the dialect are not followed.
- **Compiler facts.** No Charon store of `lance-graph-r2il` is available in this environment, and
  #2001 records that crate's extraction as PARTIAL. The arm reading is source structure.

### 5.5 Capacity

`AlgebraRef.word` is a `u8`: a lens holds at most 256 words. The probe refuses a table larger than
that. This is independent of how many positions the Codegraph can address.

### 5.6 Gate

- For one existing binding, the query "what may change if this arm of `FoldDialect::call` changes"
  returns its using methods and their classid, each row with its evidence.
- **Missing dispatch:** with one opcode's dispatch mapping removed, the words that depend on it are
  UNKNOWN, and every other mapping's rows are unchanged.
- **Stale witness:** a dispatch witness taken on another revision of the source gives no current answer.
- A name match produces no edge. A word with no user is "no registered user in this store", never
  "unaffected". A symbol that dispatches nothing has "no behaviour edge", never "no impact".
- Each guard has a disable run that goes red.

## 6. Ore, config, slag (the Ruff track)

Three kinds of rows, never merged:

| kind | holds | may contain |
|---|---|---|
| ore | declarations, operations and ordered events observed in source | only what a harvester emitted, with its grade |
| config | corpus conventions and interpretation rules | explicit rows; each derivation names the rows it used |
| slag | residuals and unmet admission obligations | the obligation and the config row or witness that is missing |

Removing one config row must invalidate exactly the derivations that name it. No source edge is
invented to fill a gap.

## 7. ON, WHEN and DO stay separate for legacy code

| source construct | candidate target | admitted only when |
|---|---|---|
| field-change trigger (`@api.depends`) | DT64 ON | the field basis is attested by the source |
| lifecycle callback (`before_save`, `after_commit`) | an event occurrence | not representable as an ON mask by convention |
| predicate or guard | DT64 WHEN | it lowers to a supported, equivalent `SetExpr` |
| ordered pure operation | DT64 DO | every step is an admitted word with admitted arguments |
| database writes, transactions, callbacks | effect residual or host boundary | not admitted to R2IL in general |

EXO-210 is the precedent: Rails scopes were admitted for a narrow, differentially tested subset;
NULL-able comparisons were refused with a counterexample; every callback stayed outside.

A value-schema preset (`ValueSchema::Bootstrap` and the others) is a representation choice. It
establishes nothing about inheritance, callback order or permissions.

## 8. Dependency order

```
0  reconcile PR #2001 + PR #2002 (§4)            two owning sessions
1  CG-2 extraction driver, CG-8 effects reader,  EXO-234, EXO-241, EXO-235
   CG-3 provenance decision
2  code → algebra word                           CG-BEH-1 (this plan), then on compiler facts after 1
3  algebra → method ON × WHEN × DO               EXO-155
4  behaviour impact, consumed by the report      EXO-155, CG-6
5  Ruff ore / config / slag, candidate           EXO-155, EXO-95      (runs beside 2 to 4)
   ON / WHEN / DO
6  cold admission and the crai-rs report         EXO-155, EXO-238     (where 4 and 5 meet)
E  egglog probe, read-only                       EXO-237              (after the effects reader)
```

CG-BEH-1 v0 does not wait for step 0 or step 1: it touches neither `impact_report` nor the store.
Its second version, on compiler facts, does.

## 9. Constraints

- No new F### word, classid, DT64 binding or execution path. No production admission.
- No second graph, parser or review reporter. No blocking CI check.
- A name-derived edge is POSSIBLE at most, and in §5 it is no edge at all.
- egglog stays a cold, read-only probe (EXO-237). DialEgg and MLIR are comparative reading only.
- The diagrams in `.mermaid/03`, `/05` and `/06` are compared against, never used as an oracle.
