# 2026-10-21 — D-CMR-2b phase 1: a scratch-free `Pred` fold does not pay

**Status:** MEASURED (lab probe, no production change). **Basis:** D-CMR-2's open item (`entries/2026-10-21-d-cmr-2-lowering.md`); operator authorization 2026-10-21; conditions of `gpt-ada-01/007`; claim EXO-50 `session_013z2e1YXkm6o6HNrDn5BPWA/023`.

- **Dependencies:** ndarray sibling at `origin/master` `37a3b59` (workspace `[patch]`); `ndarray::simd` has every compare→mask primitive (`*_i32_to_mask[_under]`, `*_u32_to_mask[_under]`), `mask_ternlog_popcount`, `popcount_batch_u64`; no fused compare-and-count primitive. mask-risc lib 47/47 against it. Host AVX-512 (Xeon @ 2.10 GHz, 4 cores), `target-cpu=native`; the probe prints the tier.
- **What scratch is today:** caller-owned, `slots × TILE_WORDS` (256 words = 2 KiB per slot), allocated once per `Scratch::for_program`, never population-sized. So the question was per-op tile write-back / re-read and the separate `And` + count passes, not heap memory.
- **Probe:** `crates/lance-graph-mask-risc/examples/pred_scratch_probe.rs`. Shapes `S1` Count(a), `S2` Count(a∧b), `S2u` b gated `under` a, `S3` Count(a∧b∧plane) over `i32` lanes; 16k / 1M / 4M rows; ~1 / 50 / 99 % selectivity. Arms: **A** `execute_extent` with reused scratch; **A+alloc** with `Scratch::for_program` per call (what `fold::set_lower` does); **B** compares into fresh stack chunks + one `mask_ternlog_popcount`; **B′** the same fold over reused chunks. Every configuration asserted against a bit-serial oracle and arm A first. Timing interleaved per rep with rotating order, median of 101 per-rep ratios, plus a null arm (A timed twice).
- **Result (runs 2–4; run 1 used non-interleaved timing and is discarded as noise — S1's B/A moved 0.78–1.13 with selectivity, which S1's work does not depend on):**

  | | 16k rows (1 tile) | 1M | 4M |
  |---|---|---|---|
  | null A′/A | 0.985–0.999 | 0.963–1.047 | 0.981–1.011 |
  | A+alloc / A | 1.009–1.098 | 0.969–1.018 | 0.990–1.017 |
  | B′/A, S3 (2 `And` passes saved) | 0.946–0.982 | 0.957–0.996 | 0.984–0.998 |
  | B′/A, S2 (1 saved) | 0.958–0.996 | 0.957–1.021 | 0.980–0.998 |
  | B′/A, S1 / S2u (none saved) | 0.968–1.051 | 0.988–1.033 | 0.993–1.033 |
  | B/A (fresh stack chunks) | 0.973–1.094 | 0.960–1.045 | 0.982–1.043 |

- **Reading:** the best case is ~5 % at one tile on S3; from 1M rows the effect is 0–4 %, inside the null's spread. Where no pass is saved the fused count is slower (it reads three operands where `popcount` reads one). Fresh stack chunks cost 2–5 % at one tile, so "scratch-free" is never better than the reused caller scratch the executor already takes. The only measurable cost of today's path is the per-call `Scratch::for_program` at one tile (~1–10 %), which vanishes from 1M rows.
- **Decision input:** phase 2 (a `Lowering` arm for `Pred` in `ir.rs` / `exec.rs`) is not supported by this measurement. Not done.

OPEN: none of D-CMR-2b's own. Not measured: `u32` / strided lanes, more than two comparisons, `Keep` terminals, other tiers (AVX2, NEON).
