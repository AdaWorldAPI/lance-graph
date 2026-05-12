# Codec Sweep YAML Configs

Request bodies for the lab REST surface (`/v1/shader/sweep`,
`/v1/shader/calibrate`, `/v1/shader/token-agreement`). Consumed by
`scripts/codec_sweep.sh` — see that script for the curl invocation.

**Rule D enforcement:** adding a new codec candidate is authoring a new
YAML file in this directory. Zero Rust changes. Zero rebuilds. The
running `shader-lab` binary JIT-compiles each unique `CodecParams`
signature; overlapping signatures across YAMLs hit the kernel cache.

## Inventory

| File | Target | What it tests |
|------|--------|---------------|
| `00_pr220_baseline.yaml` | q_proj | PR #220 baseline regression (6×256, identity rotation) |
| `10_wider_codebook.yaml` | gate_proj | PR #220 fix (a) — centroids ∈ {256, 512, 1024} |
| `12_hadamard_pre_rotation.yaml` | o_proj | PR #220 fix (c) — Hadamard × centroids |

Additional configs from plan Appendix A (residual PQ, OPQ, composite,
CartanCascade 4-tier) land as separate YAML files as Phase 1b Cranelift
wiring makes the real decode paths runnable.

## Expected results under Phase 0/2 stubs

Every candidate returns `stub: true` + `backend: "stub"` in the response
until D2.2 (real decode-and-compare) lands. Clients that trust these
rates as real measurements hit the machine-checkable `stub` wall — that's
the anti-#219 defense at the type level (see `EPIPHANIES.md` 2026-04-20
"D0.2 stub flag is anti-#219 defense at the type level").

## DoS ceiling

The sweep handler rejects grids with cardinality > 10,000 before
enumeration. Multiply axis lengths to budget — e.g., `3 × 3 × 3 × 3 = 81`
is fine; `100 × 100 = 10,000` is the exact ceiling.
