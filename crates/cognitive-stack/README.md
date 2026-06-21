# cognitive-stack — Cognitive Compilation golden image

> The **new stack** (Elixir-shaped templates) linked on top of the **old stack**
> (every AdaWorldAPI fork) in one binary. The cognitive-compilation sibling of
> `crates/symbiont`.

## Purpose

Prove — by compiling and linking — that the *compiled-cognition reflex path*
stands up as a single artifact across the whole Ada fork set, **without an LLM in
it**. That last part is the point of the architecture:

> Use LLMs to **discover** cognition, traces to **prove** it, templates to
> **compile** it, and Lance-Graph to **run** it without asking the LLM again.

So this binary contains the runtime/reflex half only:

- **NEW stack** — the Elixir-template crates that were the missing piece:
  - `elixir-template` — the `pipeline do step :x end` representation + parser +
    the `source_ranking_v1` first slice; steps bind to OGAR actions.
  - `template-runtime` — the deterministic reflex executor (OGAR-action dispatch).
  - `template-equivalence` — replay grading that gates promotion (§18).
  - `cognitive-compiler` — trace → template synthesis (used at learning time).
- **OLD stack** — the AdaWorldAPI forks that already existed:
  - `lance-graph` (+ `lance-graph-contract`, `lance-graph-ogar`) — the spine:
    query / codec / SoA contract / OGAR Active-Record bridge.
  - `ndarray` — the foundation: SIMD / HPC / Fingerprint / CAM-PQ.
  - `ractor` — control-plane ownership fence.
  - `surrealdb` (kv-lance only) — provenance / timeline view.
  - `OGAR` (`ogar-vocab` / `ogar-ontology` / `ogar-adapter-surrealql`) — the
    semantic type system.

**rig (the LLM teacher/critic) is deliberately NOT linked here.** It is used only
at learning/escalation time (a separate crate, `rig` repo, wired to the same
surrealdb-kv-lance fork). The reflex binary having no LLM dependency is the
verifiable expression of "no LLM in the hot path".

## Usage

```bash
# Local build (needs the ndarray fork checked out as a sibling of lance-graph,
# and protobuf-compiler installed for lance-encoding's build script):
cargo build --manifest-path crates/cognitive-stack/Cargo.toml

# Run — prints the linked stack + the first-slice reflex template:
cargo run --manifest-path crates/cognitive-stack/Cargo.toml

# Container build (from the lance-graph repo root) — clones the ndarray fork and
# installs protoc itself:
docker build -f crates/cognitive-stack/Dockerfile -t cognitive-stack .
docker run --rm cognitive-stack
```

Expected output (shape):

```
cognitive-stack — linked golden image
  NEW: elixir-template · template-runtime · template-equivalence · cognitive-compiler
  OLD: lance-graph + ndarray + ractor + surrealdb(kv-lance) + OGAR  [AdaWorldAPI forks]

reflex template `source_ranking` v1 → 7 OGAR actions (runs deterministically, no LLM in hot path):
    · ogar.action.ExtractSources
    · ogar.action.NormalizeClaims
    · ...
runtime ready: 0 OGAR actions registered; equivalence gate rank_tolerance=1, no_new_claims=true
```

## Build requirements

| Need | Why |
|---|---|
| Rust **1.95** | fork MSRV (ndarray fork requires 1.95) |
| `protobuf-compiler` (`protoc`) | `lance-encoding`'s build script |
| `cmake`, `clang`, `pkg-config`, `libssl-dev` | aws-lc-sys / zstd / lz4 in the closure |
| ndarray fork as a sibling of lance-graph | lance-graph path-deps `../../../ndarray` |

## Relationship to `crates/symbiont`

`symbiont` proves the **old stack** links (it predates the template work).
`cognitive-stack` adds the **new stack** on top with the identical, proven fork
wiring (same git pins, same `[patch]`). If `symbiont` builds, the only delta here
is four zero-dep crates and a `main` that exercises them.

## Status

Surface + manifest + Dockerfile landed. The full multi-fork compile is validated
via the Dockerfile (Railway/CI), mirroring `symbiont`'s validation model. See
`INTEGRATION.md` for the integration plan and `../../.claude/plans/cognitive-compilation-v1.md`
for the full design.
