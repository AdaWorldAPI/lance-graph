# D-CODEGRAPH-0: gap map from Charon to a resident Codegraph

**Date 2026-11-02 · D-id D-CODEGRAPH-0 · Status MEASURED (gap map), slice A started · Plan [d-codegraph-0-resident-codegraph-v1](../../plans/d-codegraph-0-resident-codegraph-v1.md)**

The representation is Kythe semantic facts + R2IL executable semantics + DT64/DR binding (operator, 2026-10-31); the route is Charon → ruff → lance-graph + crai-rs. Checked against `main` of lance-graph (`527a0c7`), OGAR (`5b82ae0`), ruff (`4c3acd4`) and the last 50 PRs of lance-graph and OGAR, the last 16 of ruff.

Findings:
- **No Charon intake exists in ruff**, on `main` or any remote branch. Charon has only been run out of process, with Python readers in `sag-ra-probe/tools/`.
- **No Kythe producer exists in Rust.** The only one is `sag-ra-probe/tools/kythe_entries.py`, fed by rust-analyzer facts, and it emits no `ref/call` edges.
- **No Rust-body → R2IL lowering exists.** R2IL is lifted from machine code (`ruff_r2il`) and executed for the fold and arithmetic band (`lance-graph-r2il`).
- `charon-lib` from `AdaWorldAPI/charon` builds on stable with `--no-default-features` and exposes `deserialize_llbc`, so the intake can link it. Producing LLBC needs the nightly Charon driver.
- crai-rs holds review semantics and falsifier matrices only; its fixtures were never run.
- `sag-ra-probe` did not build from a fresh resolution: `ra_ap_hir_def` pins `unicode-ident =1.0.24` (Unicode 17) while `unicode-properties` 0.1.5 is Unicode 18, and `ra-ap-rustc_lexer` asserts they match. Fixed by pinning `unicode-properties = "=0.1.4"`.

Added in this commit:
- `sag-ra-probe GRAPH=1` (with `NO_MIR=1` and `FILES=`): typed rust-analyzer rows for use as the independent oracle (`file`, `node`, `occ` with `use` / `pubuse` roles, `call`, `impl`, `import`, `macro`, `diag`).
- The oracle fixture `fixtures/codegraph/{rev1,rev2}`. rustc on rev1 reports `unused_imports` at `cg_lib/src/shapes.rs:2` and `dead_code` at `cg_lib/src/util.rs:15`. The probe on rev1: 25 nodes, 104 occurrences, 10 calls, 3 impls, 9 imports, 1 macro marker, 3 diagnostics.
- rust-analyzer's call hierarchy misses `t3(1)` inside `assert_eq!`; its index records the reference. Any unused-import finding in a file with a macro call is therefore `POSSIBLE`, never proven.

OPEN: the method body anchor (G7), whether R2IL can carry Rust places, drops and panics (G4), merge order of ruff's `claude/r2il-pr2-*` sink branches (G8).

## Python pipeline on the fixture (2026-11-02, same day)

The existing `sag-ra-probe` tools, run unchanged on `fixtures/codegraph/rev1` before any new Rust:
- Unit tests: `tools/test_tools.py` 15/15, `tools/test_kythe_entries.py` 9/9.
- **Kythe bridge:** `XREFS=1` → `kythe_entries.py` (457 entries, 0 skipped) → Kythe v0.0.76 `/xrefs`: 22/22 symbols equal rust-analyzer's occurrence sets; 19/19 hand-read pins pass (`fixtures/codegraph/rev1_xrefs.pin.tsv`, positions read with `cat -n`). Disable run: one pin moved by one column fails, and only that pin.
- **Charon** (AdaWorldAPI fork `c69e231`, `nightly-2026-09-17`, built with `--no-default-features --features rustc` because the default `minirust` feature needs a git submodule): `cg_lib` 27 s, 218 KB LLBC, `has_errors False`.
- `llbc_calls.py` → `span_join.py` → `project_rows.py` on `cg_lib`:
  - **join:** 11/11 bodies (10 EXACT, 1 ENCLOSED), 0 unjoined; the two `Shape::area` impls are told apart by position;
  - **impls:** 5/5 paired, 0 one-sided;
  - **rows:** 22 DEFINES, 4 HAS_IMPLS (rust-analyzer); 3 CALLS WITNESSED, 2 MAY_DISPATCH POSSIBLE (`trait:Clause` for the bound, `dynamic` for `dyn`), 4 MAY_PANIC (Charon).
- `cg_app`: `run` → `area_of`, `area_dyn`, `double` and the `Square` impl's `area` as static calls, all cross-crate.

What the two frontends disagree on, and who is right:
1. **The call inside `assert_eq!`.** Charon witnesses `checked → triple` (through the alias `t3`). rust-analyzer's call hierarchy misses it; its index records only the reference. Charon is right: the call is compiled.
2. **Compiled-out code.** rust-analyzer records a reference to `double` at `cg_lib/src/lib.rs` 261..267, inside `#[cfg(any())]`, and Kythe serves it as a live reference. Charon has no such item. rustc is the authority: the reference does not exist in the build. Any reader of rust-analyzer references must drop those inside its own `inactive_code` diagnostic ranges (here 210..272).

OPEN from this run: a cross-crate callee is named by Charon as `cg_lib::shapes::Impl::area`, ambiguous by name; the join only covers the local crate, so cross-crate callees still need a span join against the defining crate.

## The earlier sessions' runs, reproduced here (2026-11-02)

Before building on D-SAG-6-1/6-2 (#1876, #1878, #1883, #1894), their recorded runs were repeated in this container with the same tools:

| Run | Recorded | Here |
|---|---|---|
| `ident_chain.py` on `sag_ident` A→B, A→B declared, A→C, A→D | AMBIGUOUS; DECLARED (`sid-6` continues); MOVED via resolve; REBOUND | identical |
| Kythe on `sag_fixture` | 138 rows, 591 entries, 4 `Attribute` skipped, 24/24 symbols, 3 pins fail (all inside `assert_eq!` in `tests/law.rs`), exit 1 | identical |
| Charon + `span_join.py` on `lance-graph-mask-risc` | 459 bodies: EXACT 140, ENCLOSED 186, DERIVED 123, UNJOINED 10; impls paired 175, ra-only 0, charon-only 205 | EXACT **141**, ENCLOSED 186, DERIVED **122**, UNJOINED 10 (1 `const _`, 9 drop glue); impls identical |

The one difference: the derived `PartialEq::eq` of `Operand` (`ir.rs:6`) and of `Tern2Fold` (`ir.rs:1533`) now join EXACT, because rust-analyzer reports a symbol for the derive-generated method whose name sits on the `#[derive(...)]` line and EXACT is tried before DERIVED. The joined item is the same; only the class changed. `mask-risc`'s source has not changed since #1883, so the cause is a tool or frontend version, not determined from the record. Measured here: whole-workspace rust-analyzer dump 9.2 s load + 134 s index (1,491 files, 1,144,242 occurrences); Charon on `mask-risc` 33 s build, 441 MB LLBC (`--no-dedup-serialized-ast`); `span_join.py` 63 s.

Not rerun: the contract crate (7,189 bodies). At 441 MB of LLBC for 459 bodies, the Python join loading the file whole is the limit, which is one concrete reason the intake belongs in Rust over `charon_lib`'s typed reader.

## Slice A: `ruff_charon` (ruff `53c6711a84`, branch `claude/clone-repositories-71a5sw`)

A ruff crate (excluded from the ruff workspace, like `ruff_r2il`) reading LLBC through `charon-lib` from the AdaWorldAPI fork on stable 1.99. Three outputs:
- `calls`: **byte-identical to `llbc_calls.py`** on `cg_lib` (12 lines), `cg_app` (2) and `lance-graph-mask-risc` (460 lines, 441 MB LLBC, 5.6 s release);
- `rows`: `DEFINES` / `CALLS_RESOLVED` / `MAY_DISPATCH` / `UNRESOLVED_CALL`, grade `COMPILER`, byte spans, residuals, and a receipt `harvested == represented + residual`. `mask-risc`: 16,149 = 16,149 + 0 (459 DEFINES, 15,684 CALLS_RESOLVED, 2 MAY_DISPATCH, 4 UNRESOLVED_CALL);
- `kythe`: `defines`, `ref/call`, `childof` and file nodes. Through Kythe v0.0.76 `/xrefs`, `tools/kythe_calls_check.py` with hand-read pins (`fixtures/codegraph/rev1_calls.pin.tsv`): 3/3, including `t3(1)` inside `assert_eq!` and the bound dispatch `s.area()` pointing at the trait. Without file nodes Kythe returns no references and all 3 fail.

Closed by this slice: the cross-crate callee ambiguity. Charon's formatter names impls (`cg_lib::shapes::impl_Shape_for_Square::area`), so `cg_app::run`'s callee is the `Square` impl, not a path shared by two items.

Tests (6 integration, 3 unit) on the fixture's committed LLBC, with `llbc_calls.py`'s output as the reference. Disable runs, each red in exactly its own test: bound dispatch counted as resolved; no file nodes; callee id dropped from the report; impl names by path only.

Not yet: rust-analyzer as an independent check of the rows (the comparison is still the Python span join), resident storage (slice B), body → R2IL (G4).

## Correction (2026-11-02): G8 withdrawn

The ruff sink is not unmerged. Ruff PRs #101 (`RefinedTruthSink` + offline backend), #102 (v2 facts/residuals artifacts and their reader) and #103 (round-trip reconstruction oracle) merged on 2026-08-18. The #121 replay carried their content onto today's `main` (`crates/ruff_r2il/src/sink.rs:72` `pub trait RefinedTruthSink`; `read_facts`, `read_residuals`), where it has since changed. The three `claude/r2il-pr2-*` branches only look unmerged because they share no history with the rebuilt `main`. Slice B builds on `main`'s `sink.rs`, not on those branches.

## Slice B: the offline store (ruff #130, merged `95be2e480b`)

`ruff_charon::store::Store` persists per crate: relation rows, residuals, and the FNV-1a digest of every source file Charon read. The artifacts follow `ruff_r2il::sink`'s discipline (`#version` + `#schema`, readers keyed by column names). `RefinedTruthSink` is not reused: it is typed to R2IL `FlatFact`s, and these rows are call relations.

The unit of update is the crate: `import` replaces all of a crate's artifacts, `stale(root)` names crates with a changed or vanished file, `with_dependents` adds their dependents, and `dangling()` reports call rows whose callee no stored crate defines.

On the fixture, rev1 → rev2 (rev2 now also deletes `area_dyn` from `cg_lib` and drops its call from `cg_app`):
- **Incremental == clean.** Refreshing both crates gives the same rows as a clean rev2 import; through the CLI the store directories are byte-identical (`diff -r`).
- **Deletions.** `triple` and `area_dyn` leave no rows.
- **Skipped dependent.** Refreshing only `cg_lib` leaves exactly one dangling edge, `cg_app::run → cg_lib::area_dyn`.
- **Staleness.** `stale` is empty on unchanged sources, names `cg_lib` after a one-byte edit, and names both crates on the rev2 tree.

Five disable runs each turn their own tests red.

G5 (resident, incremental store) is closed for the offline form. Still open: residency in lance-graph's SoA (a lance-graph-side reader of these artifacts), then the queries of slice C.

ruff #131 (open) corrects four Codex findings on #129/#130: a body Charon left unstructured is still harvested, a call through a known trait impl resolves to the impl's method or the trait default, temporary files are unique per call, and a temporary name left by an interrupted import is skipped instead of aborting the import.

## Slice C: queries and diagnostics (ruff #132 merged `ed8fd4194f`; `crates/lance-graph-codegraph`)

Rows v2 in ruff adds the facts the queries need: `PUBLIC`, `IMPLEMENTS` (impl method → `Trait::method`), `SYNTHETIC`, `REFERENCES` (function pointers), a `CRATE … COMPLETE|PARTIAL` row, and `MAY_DISPATCH` naming the method. Charon marks trait-impl methods public, so cg_lib rev1 has 9 `PUBLIC` rows for its 7 `pub fn`s.

`lance-graph-codegraph` reads the artifacts by column name (no dependency on ruff_charon) and builds one xlog cube over 15 positions (rev1). Answers are masks graded confirmed / possible / unknown; `MAY_DISPATCH` is never confirmed, and a dyn or pointer call is unknown. On the fixture:
- `impact(double)` rev1: confirmed {Square::area, run}, possible {area_of}, unknown {area_dyn}; rev2: confirmed 4 including the 2-hop `checked`.
- `callees(run)` rev2: 3 callees, 4 sites.
- Dead function: rev1 `private_unused`, fnref `unused_private` (the pointer-only `helper` stays silent); rustc's `dead_code` agrees on both.
- Dangling call: mixed store (cg_lib rev2 + cg_app rev1) exactly `run → area_dyn`, status stale; rustc reports E0425 at `cg_app/src/lib.rs:7` on the same tree.
- Unused import: NOT_RUN, reason NO_IMPORT_FACTS (LLBC has no `use` items).
- Release timing, ns per call: callers 5103, callees 3643, references 184, impact 387; open 166 µs and first closure 5 µs (cold, reported, not gated).
- `cube().stats().rows_materialized == 0` after the whole suite.

Twelve disable runs each turn their own test red. The one that first stayed green (dropping the impl-method exclusion from the dead-function check) is now covered by a synthetic store: Charon marks every impl method of a public trait public, so the fixtures never reach that case.

Open: tiled closure for more than 64 positions and multi-tile cubes for more than 256 (mask-risc has 459 functions, so every cube query on it is refused today, as a typed UNKNOWN); import facts for unused imports; body → R2IL (G4, G7).

## Wide closure, merge on read, DR versioned index (branch `claude/clone-repositories-71a5sw`)

Operator rulings: wider call chain, presence bits for thin provisioning, sparse write merged on read; multi-word presence; refold only when called; versioned, replayable immutability is the DR index (one RouteVersion counter, old versions readable by number).

- dt64: `MapTable::locate` (global → (tile, local)), `LocalMap::extends`, `Axis::for_survivors`. The tile index is append-only: an extension is a length at a version, a replacement a new generation, a removal an event. An old frame reads its own layout; `map_at(version)` reads any version; a foreign frame or a future version is refused; lookups allocate nothing.
- xlog: `closure_wide` over any number of positions, one row at a time, keeping only nonzero 256-position blocks with a multi-word presence mask; `Cube::new_tiled`; a per-relation delta log (multiset: ADD +1, RETIRE −1); every read sees base + log; `refresh` recomputes only rows that reached a retired edge's source; `refold` only when called; `rebase` accepts a frame only if every tile extends the old one.
- codegraph: stores of any size; `Pos` is u32; `impact` reads whole closure rows (it copied only the first word, wrong past 64 positions); `refresh_crate` re-reads one crate and logs the edge changes (positions never renumber; growth past the reservation is NeedsReopen); rows are interned id lanes; an unchanged crate is skipped by digest.
- Bugs found by measurement: `callers` panicked on a multi-tile store (cube mask width vs symbol width); refresh lost a duplicate call site because the fold was a set (a call graph is a bag).
- Measured on mask-risc's store (695 symbols, 3 tiles), release: impact 286 ns, callers 29 µs, references 45 µs; closures 257 / 337 µs cold; RCONF keeps 614 of 2,304 blocks, RPOSS 951. Refresh of an unchanged crate 1.5–1.7 ms: a failed gate, caused by reading and hashing the 2.5 MB text rows file; the fix is D-CODEGRAPH-0-BIN (binary lanes with a header digest).
- Every new guard was disable-verified red; the re-pinned tests keep their old assertion as a comment.

Open: binary lanes (plan `d-codegraph-0-binary-lanes-v1.md`); counting through Quack instead of xlog's `fold_counts`; predictive prefetch, measured before claimed; cross-tile ℕ-compose (D-DYNBAG §7.4).
