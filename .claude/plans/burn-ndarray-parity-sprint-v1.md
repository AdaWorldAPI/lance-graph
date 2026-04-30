# Burn ↔ ndarray Parity Sprint v1

> **Status:** Active
> **Author:** main thread (Opus 4.7), session 2026-04-30
> **Source:** Other-session-shared parity list (16 items, P0-P2)
> **Target repo:** AdaWorldAPI/ndarray
> **Base:** `origin/master` `888e5982` (post PR #115 merge)

---

## Sprint topology

**12 work agents + 2 meta-coordinators.** Each work agent owns one item (or a tightly-coupled group). Each works on its own feature branch off ndarray master.

### Wave 1 — Tier 1 (parallel, independent)

| Agent | Item(s) | Branch | Effort | Model |
|---|---|---|---|---|
| **A1** | (15) Dep gating — blake3/p64/fractal optional | `claude/burn-A1-dep-gating` | ~30 LOC | Sonnet |
| **A2** | (1)+(2)+(3) F16/BF16 type + SIMD + slice ops | `claude/burn-A2-half-simd` | ~750 LOC | **Opus** |
| **A3** | (4)+(5) I8/I16 SIMD + slice ops | `claude/burn-A3-int-simd` | ~450 LOC | **Opus** |
| **A4** | (6) AMX matmul public ndarray-shaped API | `claude/burn-A4-amx-matmul` | ~300 LOC | **Opus** |
| **A5** | (11) Q4 quant helpers (i4 packing variant) | `claude/burn-A5-q4-quant` | ~80 LOC | Sonnet |
| **A6** | (10) MKL public API (sgemm/dgemm/sgemm_bf16) | `claude/burn-A6-mkl-public` | ~100 LOC | Sonnet |
| **A7** | (9) F32x16/F64x8 NEON verification spike | `claude/burn-A7-neon-verify` | ~100 LOC | Sonnet |

### Wave 2 — Independent items (parallel with Wave 1)

| Agent | Item(s) | Branch | Effort | Model |
|---|---|---|---|---|
| **A10** | (13) Reduction ops dispatcher | `claude/burn-A10-reductions` | ~300 LOC | Sonnet |
| **A12** | (16) no_std polyfill (LazyLock) | `claude/burn-A12-nostd-polyfill` | ~150 LOC | Sonnet |

### Wave 3 — Depends on Wave 1 (queued)

| Agent | Item(s) | Depends on | Effort | Model |
|---|---|---|---|---|
| **A8** | (7)+(8) AVX2/NEON int fallbacks | A3 (I8/I16 types) | ~700 LOC | Opus |
| **A9** | (12) VNNI INT8 GEMM | A3 (I8 types) | ~250 LOC | Opus |
| **A11** | (14) WASM SIMD parity | independent | ~800 LOC | Sonnet |

### Meta-coordinators (continuous, parallel)

| Agent | Role | Cadence | Model |
|---|---|---|---|
| **M1** | integration-lead — branch tracking, file conflict detection, CI status | poll every 60s, 15 polls or 25 min | Opus |
| **M2** | brutal-reviewer — invariant compliance, on-the-fly review of each push | poll every 90s, review every new commit | Opus |

## Branch + PR conventions

- Branch from `origin/master` `888e5982`
- Branch name: `claude/burn-A<N>-<short-name>`
- PR title: `feat(<area>): <item> for burn parity (sprint A<N>)`
- PR body links back to this plan + the parity list
- Each PR is independently mergeable

## CI scope per agent

ndarray CI is broken pre-existing (4348 fmt diffs, 7 main-lib clippy errors, 26 nostd errors). Agents must:
- Make their own changes pass `cargo build`, `cargo test --lib` for affected crates
- NOT fix unrelated pre-existing CI issues (scope creep)
- Document in PR body what CI checks they pass vs what's pre-existing red

## Conflict-risk matrix

| | A2 | A3 | A5 | A8 | A9 |
|---|---|---|---|---|---|
| A2 (F16/BF16) | — | simd_avx512.rs, simd_avx2.rs, simd_neon.rs | hpc/quantized.rs | — | — |
| A3 (I8/I16) | simd files | — | — | simd files | hpc/quantized.rs |

Mitigation: each agent on its own branch; conflicts resolved at PR merge order. M1 tracks conflicts.

## Acceptance per agent

Each agent's PR must demonstrate:
1. ✅ `cargo build` succeeds on the agent's branch
2. ✅ All new tests pass (each agent adds its own tests for the new functionality)
3. ✅ `cargo fmt` clean on touched files
4. ✅ No regressions in pre-existing tests of the agent's affected crate(s)

## Sprint stop condition

When all 12 work agents have either:
- Pushed a passing PR (success), OR
- Reported a blocker the main thread cannot resolve in 1 turn (defer)

M1 + M2 file final consolidated reports.
