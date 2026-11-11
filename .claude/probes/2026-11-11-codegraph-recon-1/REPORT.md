# D-CODEGRAPH-RECON-1: Mermaid ↔ Codegraph × Kythe × R2IL, reconstruction and review oracle

**Status:** probe, draft. Board: EXO-229 (compiler intake), EXO-230 (navigation). Nothing here changes
lance-graph-contract, ClassView, DT64 admission, Quack/mask-RISC, XLOG, Charon/ruff_charon formats,
crai-rs semantics or Thought-Git. No new database, store, authority or executor.

Revision: lance-graph `9f4563b` (origin/main) for every crate read; this branch adds only the read-only
`cg_query` example, this probe directory, and the regenerated `.codegraph/coverage.tsv` row it causes.

## 1. Corpus

`corpus.tsv`: 444 relationships from `.mermaid/03-dr-dt64-binding.md` (223), `05-methods.md` (102) and
`06-authorization.md` (119). Each row has subject, relation, object, type, page status, the cited source,
the verified current `path:line`, how it could be recovered, and the prover.

| recoverable | rows |
|---|---|
| MECHANICAL-SYNTAX | 240 |
| MECHANICAL-CALLGRAPH | 119 |
| INTERPRETATION | 61 |
| NOT-RECOVERABLE | 24 |

Verification against source: 435 confirmed at a current line, 7 contradicted (all partial), 2 moved, 0 not
found. Two claims were re-read by hand before use (R266: `register_derived` also inserts; R417/R422:
`walk_rails_authorized` takes a caller-supplied `allowed` closure, so the drawn `ScopedDecision →
walk_rails_authorized` edge has no call site). The pages still mark #1996 as open, though it is merged.
Ten rows carry that stale label.

## 2. Compiler intake (EXO-229 Gate A, then B)

Charon `c69e231` (nightly-2026-09-17; cargo-miri absent, so std bodies are opaque) → `ruff_charon import`
→ `lance-graph-codegraph`. Library targets only, default features, x86_64-unknown-linux-gnu.

| crate | charon s | import s | rows | residual |
|---|---|---|---|---|
| lance-graph-dt64 | 29 | 1.2 | 2,636 | 0 |
| lance-graph-contract | 21 | 7 | 31,017 | 1 (`REFERENCE_NOT_NAMED rung_schedule::requires_of`) |
| lance-graph-quack | 14 | 1 | 1,799 | 0 |
| lance-graph-r2il | 7 | 1 | 2,322 | 0 |

- Four-crate store: 9,138 positions. `open` takes 159 ms. `impact` + `callers` over every position, in
  one process, take 1,224 ms with **0 refusals**.
- **Correction to the brief.** `ClosureWidth(64)` and the 256-id tile belong to the single-tile
  `Cube::closure`. Codegraph on main builds `Cube::new_tiled` and answers with `closure_wide`
  (`lance-graph-xlog/src/wide.rs`, `MAX_POSITIONS = 2^20 − 256`). A query on this store does not meet
  those two limits. The refusal path is not exercised by these stores; the crate's own
  `Inject::Closure` / `Inject::Stage` tests cover it.
- **Defect: false unaffected.** `impact` of a `PUBLIC` symbol returns an empty set when its callers are in
  crates the store does not hold. On dt64 alone, 261 of 370 public symbols do this, including
  `BindStore::admit` and `method_ref`, which have about 140 call sites in other crates. The store holds
  the `PUBLIC` row and its observed files, so it has what it needs to say UNKNOWN. crai-rs ranks a false
  unaffected above UNKNOWN in severity. `examples/cg_query.rs` now prints an
  `UNKNOWN external-callers-unobserved` line for every public symbol and lists the observed files.
  Disable run: removing the check drops the line (1 → 0). The library API is unchanged; whether
  `Graded.unknown` should carry this is an EXO-230 decision.
- A cross-crate edge is recovered: `lance_graph_r2il::binding::prepare → BindStore::admit` (CONFIRMED).
- Grep false positive caught: `lance-graph-quack/src/bind.rs` has five `.admit(` calls. They are quack's
  own test-only typed code space, and quack does not depend on dt64.

### rust-analyzer (sag-ra-probe, `GRAPH=1`, whole workspace)

108 s, 1,214 files, 84,391 call rows, 547 MIR errors (480 `NotSupported`). Coverage limits, all measured:

- **Calls inside macro arguments have no call row.** All 7 `LaneBinding::check` sites in
  `dt64/tests/lane_socket.rs` are inside `assert_eq!`. The probe does emit a `macro` region for each, so a
  consumer can mark those regions UNKNOWN.
- **`#[cfg(test)]` unit modules are not indexed at all.** For example, `report/src/lane_source/tests.rs`
  gets no `file` row, so an absent row cannot tell unobserved from absent.
- 7,247 call rows have callee kind `Unknown` (unresolved). 6,280 are constructions (`EnumMember`,
  `Struct`), not calls.
- Crates in the workspace `exclude` list (most probes and labs) are not loaded.

## 3. Reconstruction (`recon.py`)

- `machine-recovered.mmd`: built from store rows only. `seeds.txt` names 15 nodes; it names no edges, and
  `recover` never opens the corpus. Result: 70 confirmed edges, 4 `possible` (dispatch), 0 stale, and 14
  `callers outside …: UNKNOWN` nodes for public seeds. Byte-identical on a second run.
- `reconciled.mmd`: the same plus labelled human rows. 25 GAP nodes (`:::gap`, never an edge), 6 LAB,
  10 proposed/open-PR, and 7 contradicted rows, shown as notes, not edges.
- The original `.mermaid/` files are unchanged.

### Recall of the corpus' call edges (`score.tsv`)

A hit means a compiler call row sits at the row's cited `path:line` (or on one of the next two lines).
Rows citing a file the tool never saw are UNOBSERVED, not misses.

| arm | hit | miss | unobserved | recall over observed |
|---|---|---|---|---|
| Charon store | 61 | 26 | 32 | 61/87 |
| rust-analyzer | 91 | 8 | 20 | 91/99 |

The headline gap is mostly about what each tool calls a call:

- 19 of Charon's 26 misses are variant or struct construction (`Err(Refusal::X {..})`,
  `Ok(MethodRef {..})`). These are the diagram's refuses/returns edges. RA records them as call rows;
  Charon lowers them to aggregate construction. Of RA's 91 hits, 23 are construction only, so it scores
  68 on real call sites.
- 5 rows cite a definition or doc line, not a call (R184, R206, R305, R415, R422), and miss under both
  tools. R422 is the contradicted `ScopedDecision → walk_rails_authorized` edge, which both tools
  correctly fail to find.
- On true call sites Charon misses 3: two chained `.routed_at(..)?` calls (the span starts on the
  receiver line) and one `borrow` inside a closure.

## 4. Negative tests (`recon.py --self-test`, 11 tests)

| negative | test | disable run |
|---|---|---|
| stale facts cannot claim current code | 01 | red |
| renamed / same-name symbols keep their own identity | 02 | red |
| dynamic call never absent, never confirmed | 03 | red |
| macro-hidden call stays UNKNOWN | 04 | red |
| missing artifact is not empty | 05 | red |
| guard-only field read is not omitted | 06 | red |
| public symbol reports external callers UNKNOWN | 06b | red |
| RBAC gap never rendered as an edge | 07 | red |
| a source reference is not a call | 08 | red after a fix: the first version only checked `confirmed`, and stayed green under the disable |
| lab / proposed / merged stay distinct | 09 | red |
| reconstruction cannot pass by copying | 10 | red |

Each disable asserts that its patch applied. The same check also ran on real data: a one-line edit inside
`LaneBinding::check` turned the packet `stale: true` (`packet-stale.json`), and the restored tree gives
`FRESH`.

## 5. Review evidence packet (`packet-LaneBinding-check.json`)

This is a read-only prototype of the crai-rs `ReviewEvidencePacket`. It uses the CR-H0 §3 OUT fields,
the CR-H1 identity fields and the five CR-H0-REVIEW-SEMANTICS axes. It carries evidence, never a verdict.
`severity` is null, `permission_state` is `NOT_DERIVED`, and every test is `NOT_RUN`. For a change to
`LaneBinding::check`:

- **Confirmed:** `lane` (lane.rs:360) and `borrow_all` (lane.rs:386).
- **Unknown:** external callers (the symbol is `PUBLIC`), plus test and example targets.
- **Contracts:** `LaneSource::lane_of` (dispatch).
- **Tests:** 48 name-matched candidate sites, all `SOURCE_STRUCTURE / POSSIBLE`. 23 are inside macro
  regions and 25 are in files RA does not index. None is confirmed by a compiler row, and none counts as
  TEST_OBSERVED, because nothing ran.
- **Gap:** `cargo_lock_sha` is `NOT_TRACKED`, because the repository tracks no Cargo.lock. CR-H1's
  identity assumes one.

## 6. Retrieval measurement (4 paths × 2 runs × 10 questions, same revision)

Arms:
- **A:** source only.
- **B:** `.codegraph/` first, freshness-checked, then source.
- **C:** compiler facts only: `cg_query` over the 4-crate store plus the RA graph. It could read source
  only at a line a tool had cited.
- **D:** hybrid.

Each arm had its own inputs and an explicit list of what it must not read. One grader graded all 80
answers against a held-out, source-checked key, with run labels randomised.

| run | correct | partial | wrong | tokens | tool calls | seconds |
|---|---|---|---|---|---|---|
| A1 | 8 | 2 | 0 | 545,985 | 45 | 272 |
| A2 | 7 | 3 | 0 | 519,347 | 31 | 198 |
| B1 | 6 | 4 | 0 | 515,749 | 33 | 215 |
| B2 | 7 | 3 | 0 | 536,113 | 35 | 230 |
| C1 | 7 | 3 | 0 | 541,365 | 54 | 385 |
| C2 | 8 | 2 | 0 | 537,291 | 43 | 340 |
| D1 | 7 | 3 | 0 | 534,740 | 48 | 320 |
| D2 | 9 | 1 | 0 | 546,203 | 49 | 344 |

Every run carries ~425k of shared base context, so compare differences, not totals.

All 80 answers were coverage-honest: none presented an unobserved region as empty, and none stated a
false item as fact. That is the main change from MEM-1, where the index arm under-reported Q3.

| question | what separated the arms |
|---|---|
| Q3 test callers | C is PARTIAL in both runs. It sees the 2 in-crate callers and correctly marks the test sites UNKNOWN (macro regions, cfg(test), test targets), but cannot name them. A, B and D grep them. |
| Q8 blast radius | Only C and D give compiler-resolved counts (WideFieldMask 259 refs in 4 crates; MethodBinding 46 in 3). A and B give text counts (49 files / 23 crates) that mix code, tests and doc comments. Only D2 gives both, with the doc-only split. |
| Q4 reachability | Text arms list fixture functions named `method_ref` as call sites (D1), or misfile a test file as non-test (B1). C has no such errors but sees no test callers. |
| Q1 impls | B is PARTIAL in both runs: miscounting or misfiling impls while working from text. |

Reading:
- Compiler facts bought precision on reference and call questions and cost coverage of tests.
- The hybrid arm got the best single run (9/10).
- No arm was more accurate overall with n = 2; the spread inside an arm is as large as between arms.
- Facts-backed arms were 25–90% slower in wall time, with more tool calls. Their token use was no
  higher.

### Confounds, stated

- `.codegraph/coverage.tsv` was stale while B and D ran. This branch's new `cg_query` example had changed
  the codegraph crate's row. Every B and D run reported the failed `--check` and verified against source.
  The guard fired on real drift, and it is regenerated on this branch.
- C2 says a recursive Cargo.toml grep printed four lines under `.claude/probes/`. It says it did not use
  them. The run is kept and marked.
- The grader found two errors in the key. Both were verified in source and corrected:
  - Q4: the four r2il test callers call `method_ref`, not `admit`.
  - Q6: `foundry-parity-lab/src/method.rs:218-266` lowers WHEN to mask-RISC as a LAB path outside R2IL.

  No cell was graded around them. Ten further true facts the key lacked are listed in the grader notes.

## 7. What R2IL contributes here

- R2IL reads registered DT64 `MethodBinding` effects. `prepare` (`r2il/src/binding.rs:321`) admits the
  binding, refuses WHEN (`:336`) and three-valued truth (`:339`), and binds each effect to an opcode.
- For an ordinary Rust function, such as `LaneBinding::check`, the packet records `r2il.method =
  NOT_A_METHOD, body_semantics = UNKNOWN`. Negative test 08b covers this; its disable run is red.
- **Not done:** readable per-method step reports (`Prepared::steps()`). They need a store with registered
  bindings, and the only such fixtures live in r2il's integration tests, which cannot be imported. It
  would be a new r2il example, which is new surface beyond this probe.

## 8. Open

- **crai-rs seam:** the packet is a prototype. Field-level reads (`field_readers`) and test sites are
  name matches (POSSIBLE). A resolved receiver needs RA occurrence rows joined to spans, which is the
  next step.
- **The mission brief is cut off** at §8 "Each finding should include: Source revision and diff". Each
  packet finding carries the identity block and `changed_source.diff`. Anything specified after that
  sentence is not implemented.
- **Whether `Codegraph::impact` should return UNKNOWN for public symbols** (today it is example-side
  only) is an EXO-230 decision.
- **Charon test targets:** artifacts for test targets and consumer crates would close Q3 for the
  compiler arm (EXO-229).

## Reproduce

Artifacts are not committed (build outputs; ~60 MB). From a checkout with Charon `c69e231` and
`ruff_charon` built:

```sh
# per crate, from crates/<crate>/
CARGO_TARGET_DIR=$T CARGO_PROFILE_DEV_DEBUG=0 charon cargo --no-dedup-serialized-ast --dest-file $T/<crate>.llbc
# from crates/
ruff_charon import $STORE $T/<crate>.llbc <crate_snake>
# rust-analyzer facts, from the repo root (≈110 s)
GRAPH=1 crates/sag-ra-probe/target/release/sag-ra-probe "$PWD" > ra_graph.tsv
# queries
cargo run --release --manifest-path crates/lance-graph-codegraph/Cargo.toml --example cg_query -- $STORE . sweep
python3 .claude/probes/2026-11-11-codegraph-recon-1/recon.py score --store $STORE --src . --corpus corpus.tsv --ra ra_graph.tsv
```
