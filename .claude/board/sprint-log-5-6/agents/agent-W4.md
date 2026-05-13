# agent-W4 — sprint-log-5-6 scratchpad

> **Worker:** W4 (S5-W11) | **Role:** CI matrix + green-gate spec
> **Spec output:** `.claude/specs/sprint-5-ci-matrix.md`
> **Started:** 2026-05-13

## Read-order completed

1. `.github/workflows/` — 6 files: build.yml, jc-proof.yml, release.yml, rust-publish.yml, rust-test.yml, style.yml
2. `.claude/board/LATEST_STATE.md` — PR #364 details, ndarray#142 VBMI gate
3. commit a3c753f — ndarray hpc-extras opt-in for blake3 (noted in #364 CI fix row)
4. `.claude/specs/sprint-6-conformance-test.md` (W12 sibling) — A1-A10 assertions, `--features consumer-conformance` blocking step
5. ndarray#142 P0 SIGILL on non-VBMI AVX-512 — VBMI gate for `permute_bytes`; Skylake-X / Cascade Lake / Ice Lake-SP unsafe

## Key findings

- Existing matrix: single-OS (ubuntu-24.04), single-toolchain (stable), no macOS/Windows
- RUSTFLAGS: `-C target-cpu=x86-64-v3` — AVX-512 NOT enabled (safe re: SIGILL)
- clippy lance-graph advisory; clippy contract gating
- sprint-6 W10 conformance gate needs `--features consumer-conformance`
- PR-D3a/b, PR-D4 are sprint-5 follow-ons; PR-E1/E2/E3/F1/G1/G2 are sprint-6 cascade

## Status: DONE
