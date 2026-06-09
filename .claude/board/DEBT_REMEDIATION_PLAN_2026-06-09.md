# Debt Remediation Plan — 2026-06-09 (lance-graph)

> **Companion to** `DEBT_REVIEW_2026-06-09.md` (the findings) and the TD rows
> `TD-UNUSED-DEPS-MACHETE-2026-06`, `TD-CLIPPY-ONTOLOGY-12`,
> `TD-ENV-PROTOC-MISSING` in `TECH_DEBT.md`.
> **Branch:** `claude/quirky-volta-m2r6ak` (rebased onto post-#479 main `4d26776`).
> **Cross-repo twin:** `ndarray/.claude/board/DEBT_REMEDIATION_PLAN_2026-06-09.md`.

## Hard execution constraints (non-negotiable — set by the user 2026-06-09)

1. **No autofix.** No `cargo clippy --fix` (it mangled `reader_state.rs` in
   #479). Work via **tightly-scoped Sonnet agents that reason and write**.
2. **No deletion of unused code or dependencies — ever — without explicit
   per-item confirmation.** All "unused" findings are a **propose-and-confirm
   queue**, never an autonomous action list. **P0 fork policy makes this sharper:**
   removing `lancedb`/`lance`/`ndarray`-family deps must NOT break AdaWorldAPI-fork
   wiring — verify load-bearing-ness before a dep is even a candidate.
3. **Every wave ends at a named review gate**, per
   `.claude/rules/agent-cargo-hygiene.md` (fleet without worktrees, edit-only,
   Opus orchestrator compiles + lints once).

## State: lance-graph is the HEALTHY repo

Unlike ndarray (suppression-masked, one RED), lance-graph's workspace clippy is
**GREEN** (`cargo clippy --workspace --all-targets` exit 0, 53 warnings / ~38
unique), and member unsafe hygiene is sound (35 `unsafe {` / 33 `// SAFETY:`;
the heavy FFI unsafe is quarantined in the *excluded* `holograph`). So there is
**no P0 clippy fire here** — the work is hygiene + keeping the quarantine honest.

Of the 53 warnings, **~17 are intentional** `I-LEGACY-API-FEATURE-GATED` v2
deprecation migrations (`CausalEdge64::inference_type()` / `set_temporal()`) —
**LEAVE them**; they retire with the v2 layout migration.

## P0 — Core (only one, and it's trivial)

| # | Item | Why P0 | Effort | Owner / gate |
|---|---|---|---|---|
| **C3** | Add `protobuf-compiler` to CI image + dev bootstrap (`TD-ENV-PROTOC-MISSING`). | From clean, `cargo clippy --workspace`/`build` fails *"Could not find `protoc`"* (prost/tonic under the `lance` stack + lab `grpc` feature). Repro/CI fragility. | XS (CI YAML + docs) | CI |

## Low-hanging fruit (additive/rewrite, scoped — no deletion)

- **`lance-graph-ontology`: 12 lib clippy warnings** (`TD-CLIPPY-ONTOLOGY-12`) —
  the one member cluster in an otherwise-GREEN workspace. Mostly
  `needless_range_loop` + doc-indent; one scoped agent, rewrites not deletions.
- **(Optional) `cognitive-shader-driver` 7 warnings** (`TD-CLIPPY-SHADER-DRIVER`,
  already logged) — small, fold into the same sweep if convenient.

## Leave alone / gate

- **v2 deprecation warnings (~17)** — intentional migration. Leave.
- **`TD-DEEPNSM-CLIPPY-195`** — **already RESOLVED in #479** (deepnsm clippy swept,
  CI promoted advisory→gating). No action.
- **`TD-BGZ-TENSOR-5-FAILURES-330`, `TD-FMT-STANDALONE-CRATES-4400`** — already
  logged, excluded-crate scope, separate efforts.
- **Unused deps** → **propose-and-confirm queue** (bottom). Nothing removed
  without sign-off.

## Structural fix (prevents recurrence)

1. **Keep the excluded-crate quarantine honest.** The heavy unsafe + the
   bulk of clippy debt lives in *excluded* crates (`holograph`,
   `thinking-engine`, `deepnsm`, `bgz-tensor`). That's a deliberate boundary —
   but when a crate graduates `exclude → members` (the Phase-3 plan does this for
   `bgz17`), it MUST pass the member clippy gate first. Add that as a graduation
   checklist item.
2. **Unused-dep discipline under P0 fork policy.** Track in `TECH_DEBT.md`; remove
   only on per-item sign-off; never let a removal silently drop fork wiring.
3. **`protoc` is a documented prereq** (C3) — no more clean-env breakage.

## Wave sequencing (lance-graph slice)

| Wave | Scope | Agent(s) | Gate | Acceptance |
|---|---|---|---|---|
| **W0** | C3 (CI YAML + build-prereq docs) | CI/`integration-lead` | Opus `--workspace` lint w/ protoc | clean-env build/clippy succeeds |
| **W1** | `lance-graph-ontology` 12-warning sweep (one scope) | Sonnet, `product-engineer`-style review | Opus `cargo clippy -p lance-graph-ontology` GREEN at `-D warnings` | rewrites only, tests green |
| **W2** | Present the propose-and-confirm queue → act ONLY on confirmed items | (review) | **user sign-off per item** | nothing deleted without a tick |

### Scoped-agent mission template (every wave)
> Scope: ONE crate/file. Model: Sonnet. **Edit-only; no worktree; no
> `cargo build`/`check`; NO `clippy --fix`; NO deletion of any dep or item.**
> Read `DEBT_REVIEW_2026-06-09.md` + this plan + `AGENT_LOG.md` first; prepend an
> `AGENT_LOG.md` entry on completion. Opus compiles + lints centrally.

## Propose-and-confirm queue (lance-graph) — NOTHING removed without your sign-off

**Member-crate unused deps (`cargo-machete`, verified on post-#479 tree):**

| crate | candidate deps | note |
|---|---|---|
| `lance-graph` (core) | `bgz17`, `bgz-tensor`, `lancedb`, `datafusion-expr` | first 3 = **0 src refs**; `lancedb` removal is **P0-fork-sensitive** — verify first |
| `lance-graph-planner` | `bgz17`, `p64`, `p64-bridge`, `serde`, `serde_yml` | |
| `surreal_container` | `futures`, `lance`, `lancedb`, `snafu`, `tokio` | `lance`/`lancedb` fork-sensitive |
| `lance-graph-callcenter` | `axum`, `tokio-tungstenite`, `tower-http` | |
| `lance-graph-ontology` | `arrow-array`, `once_cell` | |
| `lance-graph-catalog` | `snafu` | |
| `lance-graph-archetype` | `lance-graph-contract` | |
| `lance-graph-supervisor` | `lance-graph-callcenter`, `lance-graph-contract` | |

⚠️ **Verified FALSE POSITIVE — keep:** `cognitive-shader-driver → prost`
(`optional = true`, lab-only `grpc` feature). Triage rule: every machete hit
needs a per-entry check; optional/feature-gated + `-src` deps are false positives.

(Excluded crates also flagged — `osint` 8, `holograph` 8, `thinking-engine`
`hf-hub`, `deepnsm` `ndarray`, `learning`/`cognitive` `contract` — lower
priority, not CI-gated.)

Cross-ref: `DEBT_REVIEW_2026-06-09.md`, `TECH_DEBT.md` (TD-UNUSED-DEPS-MACHETE-2026-06).
