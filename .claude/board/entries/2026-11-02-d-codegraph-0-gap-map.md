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
