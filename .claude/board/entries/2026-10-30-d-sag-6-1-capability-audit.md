# D-SAG-6-1-AUDIT: capability audit — rust-analyzer as a stable Rust indexer, Charon, Kythe, ruff

**Date 2026-10-30 · D-id D-SAG-6-1 · Status MEASURED (partial audit) · Plan [D-SAG-6](../../plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md), ledger [D-SAG-6-1](../../plans/d-sag-6-1-audit-reference-ledger-v1.md)**

Read-only. Lab probe `crates/sag-ra-probe` (own workspace, excluded, nothing depends on it). No production crate, ABI or source rewrite changed.

The audit asked:
1. Can rust-analyzer's published library crates index this workspace on the pinned stable toolchain, how fast, and how complete is its MIR?
2. Does Charon run on stable?
3. Does Kythe still have a Rust indexer, and do its serving tools run here?
4. Does the ruff fork have a Rust frontend?

## 1. rust-analyzer (`ra_ap_*` 0.0.358) — MEASURED

Run on lance-graph `c691f8674`, rustc 1.99.0 stable, 16 GB host. The probe loads the workspace (`load_workspace_at`), walks every local crate's bodies, lowers each to MIR, then builds rust-analyzer's `StaticIndex`, the index its SCIP exporter uses.

| Quantity | Value |
|---|---|
| builds on stable 1.99 | yes; dev build of the ra_ap tree 53 s |
| load workspace | 47.4 s cold (first `cargo metadata`), 1.2 s warm |
| local crates | 404: workspace members plus path-dependency checkouts (ndarray, OGAR via `[patch]`) |
| bodies with code | 32,570 |
| **MIR lowered** | **32,023 (98.3%)** in 62 s |
| MIR errors | 547: `NotSupported` 480 (samples: `builtin#asm`, `builtin#offset_of`), `TypeError` 18, `UnresolvedName` 17 (e.g. `OGIT::DOLCE_V1`), `UnresolvedMethod` 12, `HasErrors` 11, `MissingFunctionDefinition` 7, `UnresolvedField` 1, `InaccessibleLocal` 1 |
| **symbol index** | **23 s**: 1,472 files, 146,050 symbols (145,246 with a definition span), 1,129,630 occurrences / reference rows |
| peak RSS | 5.96 GB, sampled every 2 s from outside |

Reproduce from `crates/sag-ra-probe`: `cargo run --release -- /home/user/lance-graph`. The two timed runs gave identical counts.

Conditions and limits:
- **Sysroot is required.** It needs `CargoConfig.sysroot = RustLibSource::Discover` and the `rust-src` component. Without them, MIR coverage was 7,943 / 25,092 (32%) and the index held 124,834 symbols, because `push`, `collect` and `is_empty` stayed unresolved. That earlier run is the disable arm and shows the sysroot is load-bearing.
- **Proc macros are not expanded.** Expanding them needs the build-script run (`load_out_dirs_from_check`), which is a full `cargo check` of the target. It was not run (disk). Derive-generated items are therefore missing from both counts.
- `NotSupported` is mostly inline assembly. Those bodies stay at the symbol level only.
- Peak memory near 6 GB is the operational cost of a whole-workspace index. A per-crate or incremental build was not measured.

## 2. Charon — VERIFIED (source)

`AeneasVerif/charon` main, `charon/rust-toolchain`: `channel = "nightly-2026-09-17"`, components `rustc-dev`, `llvm-tools`, `rust-src`, `miri`. It requires nightly and rustc internals, so it cannot run under the stable-only rule. At most it could be an offline cross-check, never a dependency.

## 3. Kythe — VERIFIED (source, release)

- **No Rust indexer.** Kythe `master` `69141f02` (2026-09-18) has no Rust indexer in tree. It has only `kythe/go/extractors/cmd/rust/rust_project_to_kzip.go` (2025), an extractor that packages a `rust-project.json` into a kzip for an indexer elsewhere.
- **Release v0.0.76** (tarball sha256 `eea430f7…296356`, 354 MB) ships indexers for Go, proto, C++, Java/JVM and textproto, plus `extractors/rustproject_extractor`. It has no Rust indexer and no web UI.
- **Serving tools run here.** `entrystream`, `write_tables` and `http_server` all run on this host.
- **End-to-end check passed.** Hand-written entries (one file, one function, a `defines/binding` anchor and a `ref/call` anchor) went through `entrystream --read_format=json --sort` → `write_tables` → `http_server`. `/xrefs` returned 1 definition and 1 reference with byte, line and column spans.

So Kythe works as the human view only if our producer writes Kythe entries. Its own toolchain cannot index Rust.

## 4. ruff fork — VERIFIED (tree)

ruff `main` `f2cb523` has frontends `cpp`, `csharp`, `metta`, `python`, `ruby` and `sqlalchemy` (`*_spo`), plus `ruff_r2il`. It has no Rust frontend.

## Consequence for the D-SAG-6 design

- The stable route for the Rust reading is `ra_ap`. It gives the symbol graph (definitions, references and spans) and MIR for 98% of bodies, in about 1.5 minutes and 6 GB for this workspace.
- Charon is out under the stable rule.
- Kythe and SCIP are export targets for the human view, generated from the same rows.

Open:
- proc-macro coverage, which needs a build-script run;
- incremental or per-crate indexing cost;
- the shape of the export to Kythe entries / SCIP;
- whether a Rust frontend lives in ruff (feature-gated) or in a lance-graph lab crate (dependency direction: ruff never imports lance-graph).
