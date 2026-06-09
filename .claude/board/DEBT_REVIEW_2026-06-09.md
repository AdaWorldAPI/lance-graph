# Debt Review — unsafe / clippy / unused (2026-06-09)

> **Scope:** `adaworldapi/lance-graph` (this repo) + cross-ref to
> `adaworldapi/ndarray`.
> **Branch:** `claude/quirky-volta-m2r6ak`. **Toolchain:** Rust 1.95.0 (pinned).
> **Method:** `cargo clippy --workspace --all-targets`, `cargo-machete` for
> unused deps, grep censuses for unsafe/SAFETY ratios + `#[allow]`. Numbers are
> measured. Formal TD rows added to `TECH_DEBT.md` this session:
> `TD-UNUSED-DEPS-MACHETE-2026-06`, `TD-CLIPPY-ONTOLOGY-12`.

## Cross-repo headline

| | **lance-graph** | **ndarray** |
|---|---|---|
| Workspace clippy | 🟢 GREEN (exit 0, 53 warnings / ~38 unique) | 🔴 RED (one fixed this session; one pre-existing by-design guard remains) |
| Real issue | **Unused deps** across member crates | Green gate is **suppression-masked** across new code |
| Unsafe | Well-quarantined (heavy unsafe in *excluded* `holograph`) | ~300-block undocumented gap in `hpc/`+`simd_*` |

Full ndarray detail: `ndarray/.claude/board/DEBT_REVIEW_2026-06-09.md`.

---

## 0. Environment finding (blocked the review at first)
`cargo clippy --workspace` failed with *"Could not find `protoc`"* — the
protobuf-compiler is **absent** from this environment, and a transitive build dep
(prost/tonic under the `lance` stack + the lab `grpc` feature) needs it.
Installed `protoc 3.21.12` to complete the review. **The CI image must provide
`protobuf-compiler`** or the workspace cannot be linted/built from clean.

## 1. Clippy — healthy

Workspace `--all-targets` clippy is **GREEN (exit 0)** with **53 warnings
(~38 unique)**. Breakdown by crate:

| crate | warnings | note |
|---|---|---|
| `lance-graph-ontology` | 12 (lib) | the one member worth a sweep → `TD-CLIPPY-ONTOLOGY-12` |
| `cognitive-shader-driver` | 7 | confirms `TD-CLIPPY-SHADER-DRIVER` (small, real) |
| `causal-edge` | 7 | *excluded* crate |
| `lance-graph-callcenter` | 5 | |
| `lance-graph-planner` | 4 | |
| `lance-graph` (core) | 2 | test target only |
| `p64-bridge` | 1 | *excluded* crate |

**~17 of the 53 are intentional v2-layout deprecation migrations** —
`CausalEdge64::inference_type()` / `set_temporal()` deprecation warnings, the
documented `I-LEGACY-API-FEATURE-GATED` churn. **Leave them**; they retire when
the v2 layout migration completes. The remainder is minor: a few
`needless_range_loop`, doc-indent, one `too_many_arguments (8/7)`, one
`erasing_op`, dead const/fn.

Already-logged excluded-crate clippy debt: `TD-BGZ-TENSOR-5-FAILURES-330`
(5 size-assert) and `TD-FMT-STANDALONE-CRATES-4400` (~4400 fmt hunks).
**`TD-DEEPNSM-CLIPPY-195` was RESOLVED in PR #479** (this review rebased onto
post-#479 main `4d26776`): the deepnsm clippy sweep landed and its CI step was
promoted advisory → gating, so deepnsm is no longer outstanding clippy debt.

## 2. Unsafe — well-quarantined

- **Workspace members:** 35 `unsafe {` + 21 `unsafe fn` + 5 `unsafe impl`, with
  33 `// SAFETY:` comments — roughly 1:1, good hygiene.
- **Heavy unsafe is in *excluded* crates** (not CI-gated): `holograph/src/ffi.rs`
  (61), `hamming.rs` (9), `bitpack.rs` (3); `thinking-engine` (a few). A
  deliberate quarantine — but it means that FFI surface is unaudited by the main
  pipeline. Spot-check candidate: `lance-graph-contract/src/mul.rs` (23
  unsafe-token hits — the one member hotspot).

## 3. Unused

### 3a. Unused dependencies (`cargo-machete`) — the main actionable finding
**Member crates** carrying unused deps (→ `TD-UNUSED-DEPS-MACHETE-2026-06`):

| crate | unused deps | verification |
|---|---|---|
| `lance-graph` (core) | `bgz17`, `bgz-tensor`, `lancedb`, `datafusion-expr` | first 3 = **0 source references** (verified) |
| `lance-graph-planner` | `bgz17`, `p64`, `p64-bridge`, `serde`, `serde_yml` | |
| `surreal_container` | `futures`, `lance`, `lancedb`, `snafu`, `tokio` | |
| `lance-graph-callcenter` | `axum`, `tokio-tungstenite`, `tower-http` | |
| `lance-graph-ontology` | `arrow-array`, `once_cell` | |
| `lance-graph-catalog` | `snafu` | |
| `lance-graph-archetype` | `lance-graph-contract` | |
| `lance-graph-supervisor` | `lance-graph-callcenter`, `lance-graph-contract` | |

⚠️ **Verified false positive — do NOT remove:** `cognitive-shader-driver` →
`prost`. It is `optional = true` behind the lab-only `grpc` feature
(`Cargo.toml:61,82`); machete can't see feature-gated use without
`--with-metadata`. **Triage rule:** every machete hit needs a per-entry check —
optional/feature-gated deps and `-src` linker crates are false positives.

(Excluded crates also flagged: `osint` 8, `holograph` 8, `thinking-engine`
`hf-hub`, `deepnsm` `ndarray`, `learning`/`cognitive` `contract` — lower
priority, not CI-gated.)

### 3b. Dead code / unused imports
`#[allow]` census across crates: 81 `unused_imports`, 60 `dead_code`, 31
`deprecated` (the v2 migration), 18 `too_many_arguments`. Mostly transitional;
no curated triage doc exists yet (ndarray has `UNUSED_INVENTORY_1.95.md`; a
lance-graph equivalent would be the analogous follow-up if 3b is prioritized).

## Prioritized actions (lance-graph)
1. **Drop the verified-unused core deps** (`bgz17`, `bgz-tensor`, `lancedb`)
   after a feature-gate check — slims the spine crate's dependency tree.
   (`TD-UNUSED-DEPS-MACHETE-2026-06`.)
2. Sweep `lance-graph-ontology`'s 12 clippy warnings (`TD-CLIPPY-ONTOLOGY-12`).
3. Ensure CI provides `protobuf-compiler` (§0) — latent breakage from clean.
4. (Optional) Spot-audit `lance-graph-contract/src/mul.rs` unsafe.

**Do NOT touch:** the v2 deprecation warnings (intentional), `prost`
(feature-gated false positive), excluded-crate debt already logged in
`TECH_DEBT.md`.
