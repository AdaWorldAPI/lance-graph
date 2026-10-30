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

## 5. Charon on one crate, out-of-process — MEASURED

Charon `61123872` (2026-10-30, `charon_version` 0.1.304) was built from source in the scratchpad. It needs `nightly-2026-09-17` with `rustc-dev`, `llvm-tools` and `rust-src` (≈1.5 GB, installed with `--profile minimal`). The `charon` and `charon-driver` binaries are built with `--no-default-features --features rustc`, which drops the optional minirust export and its git submodule. Build time 1 min 45 s.

Run on `crates/lance-graph-mask-risc` in place, with target and output outside the repository. `git status` was clean afterwards.

```
RUSTUP_TOOLCHAIN=nightly-2026-09-17 CARGO_TARGET_DIR=<scratch> charon cargo --dest <scratch>/out
```

| Quantity | Value |
|---|---|
| wall time | 13.5 s (including compiling the ndarray dependency) |
| peak `charon-driver` RSS | 976 MB |
| errors | 0; `has_errors: false` |
| output | one LLBC JSON file, 72.5 MB |
| local function declarations | 459, every one with a `Structured` body (none opaque or failed) |
| external function declarations | 416: 410 `Opaque` (Charon's default for dependencies), 5 `Intrinsic`, 1 `Structured` |
| other declarations | 201 types, 781 globals, 50 traits, 539 trait impls, 99 files |

Each function carries:
- its path;
- a span plus the verbatim `source_text`;
- doc comments;
- the generic signature;
- a structured body with locals.

The 459 is Charon's own count: it lists closures, trait-impl methods and generated items separately. A text count finds 197 `fn` lines in `src/`, and the two numbers are not comparable one-to-one.

**Stable reader: does not hold today.** Charon's manifest says `--no-default-features` compiles `charon-lib` on a stable toolchain. At this commit, `cargo check --no-default-features --lib` on 1.99.0 fails: `src/utils.rs` uses the unstable library feature `drop_guard` (3× E0658). A stable consumer (ruff on 1.99) cannot link `charon-lib` now. It would have to read the LLBC JSON with its own types, or wait until `drop_guard` stabilises or Charon drops it.

Not measured:
- MIR-level coverage on crates with heavy unsafe, inline asm or proc macros;
- the cost of translating dependencies (`--extract-opaque-bodies`);
- the miri full-MIR sysroot. `cargo-miri` was not installed, so Charon fell back to rustc's default sysroot, with a warning.

## Consequence for the D-SAG-6 design

- The stable route for the Rust reading is `ra_ap`. It gives the symbol graph (definitions, references and spans) and MIR for 98% of bodies, in about 1.5 minutes and 6 GB for this workspace.
- Charon cannot be a dependency under the stable rule. Out-of-process it works: it translated every local body of `lance-graph-mask-risc` in 13.5 s. Its output must be read with our own types until `charon-lib` builds on stable.
- Kythe and SCIP are export targets for the human view, generated from the same rows.

Open:
- proc-macro coverage, which needs a build-script run;
- incremental or per-crate indexing cost;
- the shape of the export to Kythe entries / SCIP;
- whether a Rust frontend lives in ruff (feature-gated) or in a lance-graph lab crate (dependency direction: ruff never imports lance-graph).
