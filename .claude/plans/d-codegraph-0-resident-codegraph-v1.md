# D-CODEGRAPH-0 — a resident Codegraph: Charon → ruff → lance-graph + crai-rs

**Date 2026-11-02 · Status PROPOSED; slice A in progress · Board: [STATUS_BOARD](../board/STATUS_BOARD.md) D-CODEGRAPH-0 · Entry: [gap map](../board/entries/2026-11-02-d-codegraph-0-gap-map.md)**

**Goal.** Import Rust source once into a persistent, queryable, incrementally
maintained Codegraph, so coding agents query established facts instead of
reconstructing them by text search. Correctness gates come before any agent
interface; MCP is last.

**Representation (operator, 2026-10-31, restated 2026-11-02).** Kythe
semantic facts + R2IL executable semantics + DT64/DR binding. Quack,
mask-risc and relalg are polyfill underneath R2IL, not the representation.
Route: **Charon → ruff → lance-graph + crai-rs.**

| Layer | Owner | Carries |
|---|---|---|
| Rust frontend | Charon (`AdaWorldAPI/charon`, `charon-lib` stable with `--no-default-features`) | LLBC: items, spans, bodies, resolved calls, trait-ref kind per call |
| Intake | ruff (`ruff_charon`, new) | LLBC → Kythe facts (nodes, anchors, edges) with a witness status per edge |
| Executable semantics | `ogar-r2il` (vocabulary) + `lance-graph-r2il` (`FoldDialect`, executes) | method bodies as R2IL words; never pre-converted to SPO (OGAR ruling 2026-08-26) |
| Binding | `lance-graph-dt64` (`MethodBinding`, `BindStore`, DR `MapTable`) | method identity, admission, global → local maps |
| Query / execution polyfill | Quack → mask-risc, `lance-graph-xlog` cube | populations, closures, counts |
| Review semantics | crai-rs (CR-H0 review semantics, CR-H1 falsifiers) | finding record, evidence grade, lifecycle |
| Independent oracle | rust-analyzer (`sag-ra-probe GRAPH=1`) + rustc (`cargo check`) | a second frontend to compare against; never the ingestion path |

## 1. Gap map (verified 2026-11-02 against `main`)

**Exists:**
- `sag-ra-probe` (#1876, #1878, #1883, #1894, lab, excluded): rust-analyzer
  facts; Charon run out of process with call classification in Python
  (`tools/llbc_calls.py`); span join of the two frontends; identity that
  survives a witnessed move; Kythe entries from rust-analyzer facts with
  `/xrefs` parity 24/24.
- `lance-graph-r2il` (#1895, #1897, #1899, #1902–#1904): `FoldDialect`
  executes the R2IL fold and arithmetic band over mask-risc; polyfill lens
  `0xFFFF::0xF000`.
- `ogar-r2il`: the R2IL arity table (82 ops, `0x90..=0xE1`) plus the fold band.
- `tract-harvest-probe` (#1876): the codegraph row layout (4+12 key per
  node / value / use site, one edge row per use, attributes in value columns).
- `dt64-admission-probe` (#1891): consumes DEFINES / CALLS / effect evidence
  rows to decide whether a method may execute.
- `lance-graph-xlog` cube (#1907, #1911–#1913): relations, joins, closure.
- crai-rs: review semantics and falsifier matrices (docs only; nothing run).

**Missing, in pipeline order:**

| # | Gap | Evidence | Slice |
|---|---|---|---|
| G1 | No Charon intake in ruff (no branch mentions Charon or LLBC) | ruff `origin/main` + every remote branch, 2026-11-02 | A |
| G2 | No Kythe producer in Rust; the only one is `sag-ra-probe/tools/kythe_entries.py`, fed by rust-analyzer | `git grep` over lance-graph, OGAR, ruff | A |
| G3 | No Kythe `ref/call` edges anywhere: rust-analyzer gives references, and a reference is not a call | `sag-ra-probe/KYTHE.md` "Not produced" | A |
| G4 | No Rust-body → R2IL lowering; R2IL arms are machine code (`ruff_r2il`, r2sleigh) and fold/arithmetic (`lance-graph-r2il`) | crate descriptions + sources | C (design check first) |
| G5 | No resident store or incremental update for code facts | no crate holds them | B |
| G6 | Closure in the xlog cube refuses more than 64 positions per tile; frontier masks may not feed another program (`graph_hop` quarantined, ISS-GATHER-HOP-1616) | `cube.rs` `closure`, `graph_hop.rs` header | C |
| G7 | Method body anchor is open: `ogar-loco::Program` is not the body (operator correction 2026-10-31) | `2026-10-31-coresearch-mandatory-codegraph.md` | C |
| G8 | The ruff → lance-graph sink is unmerged (`RefinedTruthSink`, ruff branches `claude/r2il-pr2-*`) | `git ls-remote` ruff | B |
| G9 | `sag-ra-probe` did not build from a fresh resolution (`unicode-ident` 1.0.24 vs `unicode-properties` 0.1.5) | build log 2026-11-02; fixed by pinning `unicode-properties = "=0.1.4"` | done |

## 2. Slices

**A — intake for one real crate.** `ruff_charon`: read LLBC with
`charon_lib::deserialize_llbc`, emit Kythe entries (anchors with byte
offsets, `defines/binding`, `ref`, `ref/call`, `childof`, `extends`) and a
witness status per call: `RESOLVED` (`Fun`, `TraitImpl`), `POSSIBLE`
(`Clause`, `Dyn`), `UNKNOWN` (function pointer). Checked against the
rust-analyzer oracle on `sag-ra-probe/fixtures/codegraph` and against rustc.
Then one real crate (`lance-graph-mask-risc`, measured by Charon before:
459/459 bodies).

**B — resident and incremental.** Rows reach lance-graph through the ruff
sink seam (G8) and are keyed per the `tract-harvest-probe` layout. Per-file
content digests; an update invalidates the changed files plus the files that
reference symbols defined in them; parity test: clean import equals
incremental import as sets.

**C — queries, diagnostics, bodies.** Callers, callees, references, impact
through the xlog cube and Quack; tiled closure for more than 64 positions
(G6); structural diagnostics with crai-rs's finding record. Body lowering to
R2IL only after its own design check (G4, G7).

**D — local MCP.** A thin adapter over C's query functions, only after the
B and C gates pass.

**E — token economics.** Same questions, same model, same revision, with and
without the Codegraph; no savings claimed before this is measured.

## 3. Oracle fixture

`crates/sag-ra-probe/fixtures/codegraph/{rev1,rev2}` (three crates):
aliases (`use … as t3`), a public re-export (`pub use … as Sq`), cross-module
and cross-crate calls, trait dispatch through a bound and through `dyn`, a
`#[cfg(any())]` item, a reference used only inside `assert_eq!`, unresolved
names (`cg_broken`), a module cycle (`shapes` ↔ `util`), and in rev2 a deleted
function and a new one.

rustc on rev1 (`cargo check -p cg_lib -p cg_app`): `unused_imports` at
`cg_lib/src/shapes.rs:2` (`triple`), `dead_code` at `cg_lib/src/util.rs:15`
(`private_unused`). The alias `t3`, used only inside `assert_eq!`, is not
reported unused.

The rust-analyzer oracle is `sag-ra-probe GRAPH=1 NO_MIR=1`, which emits
`file`, `node`, `occ` (with `use` / `pubuse` roles), `call`, `impl`,
`import`, `macro` and `diag` rows. Measured on rev1: 25 nodes, 104
occurrences, 10 calls, 3 impls, 9 imports, 1 macro marker, 3 diagnostics,
6.6 s wall time. One known gap of the oracle itself: rust-analyzer's call
hierarchy misses the call `t3(1)` inside `assert_eq!`, while its index does
record the reference.

## 4. Gates

- Every resolved edge traces to a frontend fact; nothing is inferred from
  names.
- `POSSIBLE` dispatch is never counted as a confirmed caller.
- An import is reported unused only where the file has no macro call with
  arguments and the import is not a trait, a glob or a re-export; otherwise
  the finding is `POSSIBLE` at most.
- Clean-import and incremental-import results are equal.
- Each guard has a disable run that turns its test red.

## 5. Open

- Where the method body revision is anchored (G7).
- Whether R2IL can carry Rust places, drops and panics, or needs a separate
  reading (G4).
- Merge order of the ruff sink branches (G8).
