# 2026-09-29 — D-PFP-1 Perturbationsfeld probe: INVALID (data-driven), both lenses

**Status:** MEASURED · DONE (verdict INVALID) — `crates/perturbationsfeld-probe`, plan `.claude/plans/perturbationsfeld-probe-v1.md` §9

**Pre-registration:** `crates/perturbationsfeld-probe/PREREG.md`, committed in `3e1d0b2210243812efa2aa156613975d41968c07` (pushed before the run), sha256 `84753434a979124226a52f837ca721f99f8096d134dfdd66cca5a41569b06438`. No constant changed. The run is deterministic (a second invocation reproduced the output byte-for-byte).

## Verdict

`VERDICT: INVALID` — PRIMARY (Jina v5) and REPLICATION (BGE-M3) both INVALID on the same two checks: **degeneracy ceiling (32 of 32 stimuli excluded)** and **θ inertness (E)**.

## Classification: data, not harness (harness-bug clause does not apply)

`examples/diag.rs` (pre-registered constants only) shows, for the PRIMARY lens (floor = 191, the p75): after `think(10)` under RESET the energy is spread over **all 256 rows** (sum 1, max ≈ 0.00961), so **no row clears θ = 0.01** and the E set is empty for every stimulus — each stimulus is degenerate by the pre-registered rule. θ inertness fails as a consequence (the E set cannot shrink at 2θ when it is already empty). The probe behaved as specified.

## Measured beside the verdict (observations, not an outcome)

- **Every stimulus converges to the same top set.** N0 (cross-stimulus top-set overlap of `P_n`) = **1.0000** on both lenses; the positive control (the least-similar centroid pair as single-id stimuli) overlaps **8 / 8** on both lenses. Four PRIMARY stimuli with disjoint-looking inputs end at top ids 184, 234, 246 with energies (≈ 0.0096 / 0.0087 / 0.0085) equal across stimuli to within ~1e-6. Both INPUT-INSENSITIVE conditions are met by the numbers, but the pre-registered order evaluates INVALID first, so INPUT-INSENSITIVE is **not** the verdict.
- **Scope of that observation:** the two tracked 256² tables, the p75 floor, `think(10)` (loops `cycle()`), RESET. Other engines (`sparsify`, `think_with_temperature`, signed / BF16 engines) were not run.
- **Production-adjacent consequence (same scope):** with max energy ≈ 0.0096 < `SCAN_WORTHY_ENERGY` = 0.01, the `dispatch_from_top_k` active filter (`engine_bridge.rs:130-133`) would be empty and fall back to the `[0, 64)` window for every stimulus. Not tested on the production call path (which has no production caller of `dispatch_from_top_k`).

## Raw output (run 1)

```
D-PFP-1 Perturbationsfeld probe — PREREG commit 3e1d0b22
── jina-v5 (PRIMARY) ── N = 256
  stimuli valid (base) 0 / 32, comparison-valid Q_valid 0
  positive control pair (0, 1) overlap 8 / 8
  empirical null N0 (cross-stimulus top-set overlap) 1.0000  [uniform reference 8/256 = 0.0312]
  retention (Σ ids kept of 8): R_E 0  R_C 0  R_S 0  R_Em 0   (per stimulus: E 0.000  C 0.000)
  ΔR = R_E − R_C over comparison-valid stimuli: 0  (margin ±0)
  relabel sanity: D(E,S) 0 over 0 eligible (needs ≥ 0)
  mean |ids_C| 0.00  mean |ids_E| 0.00  top_k padding rate 0.333  mean L1(E,C) -0.0000
  OUTCOME: INVALID — theta inertness (E); degeneracy ceiling (32 of 32 excluded)
── bge-m3 (REPLICATION) ── N = 256
  stimuli valid (base) 0 / 32, comparison-valid Q_valid 0
  positive control pair (5, 123) overlap 8 / 8
  empirical null N0 (cross-stimulus top-set overlap) 1.0000  [uniform reference 8/256 = 0.0312]
  retention (Σ ids kept of 8): R_E 0  R_C 0  R_S 0  R_Em 0   (per stimulus: E 0.000  C 0.000)
  ΔR = R_E − R_C over comparison-valid stimuli: 0  (margin ±0)
  relabel sanity: D(E,S) 0 over 0 eligible (needs ≥ 0)
  mean |ids_C| 0.00  mean |ids_E| 0.00  top_k padding rate 0.333  mean L1(E,C) -0.0000
  OUTCOME: INVALID — theta inertness (E); degeneracy ceiling (32 of 32 excluded)
VERDICT: INVALID
```

## Diagnostic (PRIMARY, first 4 stimuli)

```
size 256 floor 191
stim 0: ids [136, 151, 87, 79, 90, 171, 83, 220] pre 1 after1cycle 1.0000004 | think: sum 0.9999999 nz 256 max 0.009614392 >θ 0 cycles 6 top [(184, 0.009614392), (234, 0.008747272), (246, 0.008512111)]
stim 1: ids [59, 100, 249, 37, 87, 146, 169, 234] pre 1 after1cycle 1.0000001 | think: sum 1.0000002 nz 256 max 0.009614759 >θ 0 cycles 6 top [(184, 0.009614759), (234, 0.008749689), (246, 0.008511895)]
stim 2: ids [118, 151, 100, 14, 6, 234, 82, 57] pre 1 after1cycle 0.99999976 | think: sum 0.99999994 nz 256 max 0.009614053 >θ 0 cycles 6 top [(184, 0.009614053), (234, 0.008750533), (246, 0.008511884)]
stim 3: ids [128, 54, 122, 184, 177, 144, 224, 151] pre 1 after1cycle 0.9999999 | think: sum 1.0000002 nz 256 max 0.00961387 >θ 0 cycles 6 top [(184, 0.00961387), (234, 0.008745162), (246, 0.008511538)]
row 10: max 255 diag 255 count>floor 50
```

## OPEN

- Whether the thinking-engine is input-sensitive at all under any configuration is untested; this run only shows it is not under `think(10)`, p75 floor, RESET, on these two tables. A re-registration (D-PFP-1b) with a θ the field can cross would be moot unless the single attractor is first broken — deciding HOW (sparsify / temperature / fewer cycles) is a design choice, not a probe fix, and is not made here.
- The Perturbationsfeld loop question (mask/program → field → fold → perturbation → cross-cycle consequence) stays OPEN: the fold→perturbation→next-cycle leg carries no information through this engine configuration, but that is a statement about the engine configuration, not about mask-risc or the loop topology.
