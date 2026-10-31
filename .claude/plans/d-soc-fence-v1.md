# D-SOC-FENCE — one place per concern, checked in CI

**Status:** PROPOSED, plan only. No tool, workflow, mapping file or code change lands with this plan.

**Date:** 2026-10-31. **Trigger:** PR #1884's `xlog-relcomp-probe/src/native.rs` counts by calling `ndarray::simd` group and scatter functions directly. That bypasses `lance-graph-quack` (the query surface) and `lance-graph-mask-risc` (the one fold evaluator). D-DT64-6 A.5 addenda 8 and 12 had already ruled that folds name mask-risc's vocabulary and belong to Quack. Nothing mechanical caught it.

**Rule.** Each concern has exactly one owning place. Code elsewhere reaches that concern only through its owner. The record of owners and exceptions is one mapping file. A CI check reads that file and reports every use outside it as a separation-of-concerns (SoC) violation. Known violations are listed in a baseline that only ever shrinks.

## 1. Blast radius (why plan-only)

Counted on 2026-10-31 at `84f80eb` with `git grep`. **These are candidate counts, not findings**: a search finds candidates and only reading them decides.

| What was searched | Candidate files, by crate |
|---|---|
| `ndarray::simd` masking, group, scatter, gather or popcount functions imported outside mask-risc | lance-graph-planner 5, dir-rmap-probe 5, math-polyfill-probe 4, drf-focus-probe 4, crepe-plan-probe 4, cognitive-shader-driver 4, soa-view-probe 3, lance-graph-onnx 3, tract-harvest-probe 2, lance-graph-benches 2, d-diamond-1-probe 2, r2il-mask-abi-probe 1, lance-graph-quack 1, lance-graph-contract 1, jc 1 |
| `ndarray::hpc::` imported | thinking-engine 32, bgz-tensor 13, lance-graph-planner 10, lance-graph 4, and 7 other crates with fewer than 4 |
| `core::arch` / `std::arch` | 8 crates. Some hits are feature detection (`is_x86_feature_detected!`), not intrinsics |
| CE64 bits 59..63 written by shift outside `causal-edge` | cognitive-shader-driver examples (2 files) |
| `mask_risc::execute*` called from Quack | 3 sites, all after a `#[cfg(test)]` at line 2510 (test-only, to confirm) |

Two false-positive classes are already visible: test-only code, and feature detection. A check that fails on the first run would block unrelated PRs across about 25 crates. So the check starts in report-only mode (Wave 1), and blocks only new violations once the baseline is committed (Wave 2).

## 2. The mapping file: `.claude/soc/one-place.toml`

The mapping file is the only place where an owner is written down. Docs, cards and CLAUDE.md files link to it rather than restating it.

```toml
[[concern]]
id        = "fold-evaluator"
statement = "Masked folds (keyed count/sum/min/max, scatter, gather, popcount terminals) are evaluated by mask-risc"
owner     = { crate = "lance-graph-mask-risc", paths = ["src/exec.rs"] }
basis     = ["D-DT64-6 A.5 addendum 8", "lance-graph-mask-risc Cargo description"]
symbols   = ["ndarray::simd::masked_group_*", "ndarray::simd::mask_scatter_*", "ndarray::simd::mask_gather_*"]
reach_via = ["lance_graph_quack::*", "lance_graph_mask_risc::{Program, execute, execute_into}"]

[[concern.sanctioned]]
crate = "lance-graph-onnx"
scope = "elementwise aliases only (mask_and/or/not, *_to_mask)"
basis = "D-ONNX-NATIVE"
```

Every concern carries five things:
- `statement`: the concern in one sentence;
- `owner`: crate and paths;
- `basis`: the ruling or contract it comes from;
- `symbols`: what only the owner may name;
- `reach_via`: what everyone else uses instead.

A sanctioned exception carries a named crate, a scope and a basis. It is never a blanket exemption.

Each use the check finds is classified as one of:

| class | meaning |
|---|---|
| `OWNER` | inside the owner's paths |
| `SANCTIONED` | a named exception with its scope and basis |
| `TEST` | in `#[cfg(test)]`, `tests/` or `benches/`. Allowed by default; a concern can opt out |
| `TRACKED` | a known violation listed in the baseline, with an owner and a D-id |
| `VIOLATION` | anything else; fails CI from Wave 2 |

### Concerns to confirm in Wave 0

Each row below is a proposal. Wave 0 confirms or corrects it by reading the cited basis and the owner's code. No row enters the file on this plan's say-so.

| id | owner (proposed) | everyone else goes through | basis |
|---|---|---|---|
| `simd-facade` | `ndarray::simd` (the AdaWorldAPI fork) | `ndarray::simd::*`, never `ndarray::hpc::*`, `core::arch` intrinsics or a second SIMD crate | lance-graph-java and tesseract-rs `simd-savant`; mask-risc Cargo description ("T1 facade"). In lance-graph, which `hpc` modules are re-exported by `simd` is not yet established. Wave 0 decides whether `hpc` imports are violations or sanctioned |
| `fold-evaluator` | `lance-graph-mask-risc` | Quack, or a mask-risc `Program` | addendum 8; Quack crate docs |
| `query-surface` | `lance-graph-quack` builds Programs and never evaluates them | — | Quack crate docs ("may build a Program and must never evaluate one") |
| `selection-description` | `lance-graph-relalg` | — | addendum 8 ("relalg describes") |
| `dr-addressing` | `lance_graph_dt64::dr` (`FrameStore`, `Guards`, `Focus`, `Scope`, `LocalMap`, `ClassAxis`) | `dt64::dr` | addendum 12. The `dr-math-binding-probe` copies are `TRACKED` duplicates |
| `bind-store` | `lance-graph-dt64` (`BindStore`, `MethodBinding`, the route log) | `BindStore` | D-DT64-6 D4/D5 |
| `method-identity` | **OPEN** (`MethodRef` vs `MethodSignature` vs `MethodKey`, EXO-50) | — | listed with no owner; not enforced until decided |
| `classid-split` | `ogar_codebook::{classid_concept, classid_canon, …}`, `dt64::view_class` | those functions; no bit math on a composed classid | `/v3-audit` item 1 |
| `epi5-field` | `CausalEdge64::{epistemic_raw5, with_epistemic_raw5}`, `Epi5Declarations::project_state5` | those; no shifts or masks on bits 59..63 elsewhere | `epistemic_state5.rs` module docs |
| `presence` | the class's `FieldMask` / `WideFieldMask` | presence bits; never a sentinel value inside a field | `class_view.rs` C2 |

`/v3-audit` keeps its own checks. Its item 1 moves into the mapping file, so the rule is written in one place, and the command reads it from there.

## 3. The check: `soc-fence`

There are two levels, and neither relies on a regex for its verdict.

1. **Dependency fence (exact).** `cargo metadata` runs over the root workspace and over every standalone probe workspace (each `crates/*/Cargo.toml` with its own `[workspace]`). It lists which crates depend directly on each owner's underlying dependency (for example `ndarray`). A crate may depend on `ndarray` directly only if it is an owner or sanctioned for at least one concern that needs it.
2. **Symbol fence (parsed).** A small Rust tool built on `syn` parses every `.rs` file.
   - It resolves `use` trees (groups, renames, globs, `self`) and fully qualified paths in expressions and macros, and records `#[cfg(test)]` scope.
   - It reports `(concern, crate, file, line, symbol, class)`.
   - It is `syn` rather than `grep` because the measured failures here are grep failures: G11 (lance-graph-java) missed fully qualified imports, and CLAUDE.md records `impl lance_graph_contract::hotplug::CapabilityAuthority for …` evading `impl.*CapabilityAuthority`.
   - Precedent for a Rust tool under `.claude/tools/`: `preflight_drift.rs`.

**Baseline** (`.claude/soc/known-violations.toml`). One entry per `TRACKED` use: `(concern, crate, file, symbol, owner, d_id, reason)`. It is a ratchet in both directions. A new violation fails CI, and so does a baseline entry that no longer matches anything. Fixing a violation therefore means deleting its line in the same PR, and the list cannot outlive the code (as with `known_integration_call_breaches.json` in ghidra-mcp).

**Workflow** (`.github/workflows/soc-fence.yml`). Runs on changes to `crates/**`, `.claude/soc/**` and the tool itself. In Wave 1 it reports and always passes. From Wave 2 it fails on `VIOLATION` and on a stale baseline entry.

**Its own falsifiers** (in the tool's tests). A fixture crate holds one violation per concern, written in three spellings: a plain `use`, a grouped `use`, and a fully qualified path in an expression. Each must be reported (the can-fire half). A clean fixture holds owner, sanctioned and test-only uses, and must report nothing (the can-stay-silent half). Removing any concern from the mapping file must turn its fixture test red. Feature detection (`is_x86_feature_detected!`) is in the clean fixture.

## 4. Waves

| Wave | D-id | Work | Gate | Tier |
|---|---|---|---|---|
| 0 | D-SOC-FENCE-0 | Census. Read every candidate in §1 and classify it OWNER / SANCTIONED / TEST / VIOLATION. Confirm or correct each concern in §2 against its basis. Write the mapping file draft and the baseline draft | every baseline entry cites the file and line it read; each concern's basis is quoted, not paraphrased | Haiku `harvest-scout` / `quote-harvester` for candidates and quotes; Sonnet classifies; Opus rules on owner disputes |
| 1 | D-SOC-FENCE-1 | The `syn` tool, its fixture falsifiers, the workflow in report-only mode | fixture can-fire and can-stay-silent both green; a disable run per concern red | Sonnet from a written spec; Opus reviews |
| 2 | D-SOC-FENCE-2 | Commit the mapping file and baseline; the workflow blocks new violations and stale entries | run on main: zero `VIOLATION`, baseline matches exactly | Opus |
| 3 | D-SOC-FENCE-3 | Burn down the baseline, one PR per crate, each deleting its own baseline lines. Each move routes the crate through its concern's `reach_via` (for example XLOG's counting through Quack → mask-risc) | each PR: the crate's tests green; its baseline lines gone | owning sessions |
| 4 | D-SOC-FENCE-4 | Fold `/v3-audit` item 1 and the lance-graph-java G11 allowlist into the mapping file, so each fence has one written source | `/v3-audit` and the G11 test both read the file; their old inline lists deleted | Opus |

Nothing in Waves 0–2 changes production code. Wave 3 is the only wave that touches crates, and it splits by owner.

## 5. Decisions needed before Wave 0

1. **Which concerns are in.** The ten rows of §2 are proposals; `method-identity` stays unenforced until EXO-50 decides.
2. **Whether `ndarray::hpc::*` imports are violations** in lance-graph (thinking-engine 32, bgz-tensor 13, planner 10). The lance-graph-java and tesseract-rs rule says never; whether lance-graph follows it has not been ruled.
3. **Whether lab probes are in scope.** Proposed: yes, as `TRACKED` entries, so a probe's shortcut is visible rather than exempt.
4. **Tool language.** Proposed: Rust with `syn`, to resolve paths. A Python regex tool would repeat the measured G11 failure.

## 6. The first known violation

PR #1884's `crates/xlog-relcomp-probe/src/native.rs` (concern `fold-evaluator`). It is to be fixed in that PR, by routing the counts through Quack → mask-risc and keeping only `checked_binary_i64` direct, rather than entered in the baseline.
