# 2026-10-21 — D-CMR-2b phase 1: a scratch-free `Pred` fold does not pay on whole extents; on `set_lower`'s short runs, per-run `execute_extent` is 2.5–4× slower

**Status:** MEASURED (lab probe, no production change). **Basis:** D-CMR-2's open item (`entries/2026-10-21-d-cmr-2-lowering.md`); operator authorization 2026-10-21; conditions of `gpt-ada-01/007`; claim EXO-50 `session_013z2e1YXkm6o6HNrDn5BPWA/023`.

- **Dependencies:** ndarray sibling at `origin/master` `37a3b59` (workspace `[patch]`); `ndarray::simd` has every compare→mask primitive (`*_i32_to_mask[_under]`, `*_u32_to_mask[_under]`), `mask_ternlog_popcount`, `popcount_batch_u64`; no fused compare-and-count primitive. mask-risc lib 47/47 against it. Host AVX-512 (Xeon @ 2.10 GHz, 4 cores), `target-cpu=native`; the probe prints the tier.
- **What scratch is today:** caller-owned, `slots × TILE_WORDS` (256 words = 2 KiB per slot), allocated once per `Scratch::for_program`, never population-sized. So the question was per-op tile write-back / re-read and the separate `And` + count passes, not heap memory.
- **Probe:** `crates/lance-graph-benches/examples/pred_scratch_probe.rs`, run with `cargo run --release -p lance-graph-benches --example pred_scratch_probe`. Shapes `S1` Count(a), `S2` Count(a∧b), `S2u` b gated `under` a, `S3` Count(a∧b∧plane) over `i32` lanes; ~1 / 50 / 99 % selectivity. Two workloads: **whole** (one extent `0..n`, n = 16k / 1M / 4M) and **runs** (`set_lower`'s shape: over n = 1M, K non-overlapping runs of 200, 3,000 or 40,000 rows at odd starts, about a quarter of the rows). Arms: **A** `execute_extent` per extent with reused scratch; **A+alloc** with `Scratch::for_program` once per call (what `SetPlan::count` does); **Ac** A with the program compiled once (`Program::compile` + `execute_compiled`); **B** compares into fresh chunks + one `mask_ternlog_popcount`, plane shifted into a window when a run starts mid-word; **B′** the same over reused chunks. Every configuration asserted against a bit-serial oracle before timing. Timing interleaved per rep with rotating order, median of 101 per-rep ratios, plus a null arm (A timed twice).
- **Result (3 runs; ranges over the 3 selectivities × 3 runs):**

  | workload | null A′/A | A+alloc/A | Ac/A | B′/A |
  |---|---|---|---|---|
  | whole 16k (1 tile) | 0.989–0.999 | 1.026–1.090 | 0.979–1.005 | S1 1.40–1.46, S2/S2u/S3 1.06–1.26 |
  | whole 1M | 0.957–1.012 | 0.971–1.027 | 0.971–1.033 | S1 1.04–1.20, S2/S2u/S3 0.95–1.04 |
  | whole 4M | 0.984–1.014 | 0.989–1.023 | 0.987–1.011 | 0.970–1.028 |
  | 1,310 runs × 200 | 0.952–1.033 | 0.981–1.023 | 0.821–0.948 | **0.236–0.524** (S3 0.24–0.27, S2 0.30–0.33) |
  | 87 runs × 3,000 | 0.971–0.997 | 0.989–1.023 | 0.905–1.027 | S2/S3 0.67–0.80, S2u 0.86–0.93, S1 0.74–1.06 |
  | 6 runs × 40,000 | 0.971–0.999 | 0.985–1.029 | 0.993–1.029 | 1.00–1.45 |

- **Reading, whole extents and long runs:** no win for the fold; it is equal or slower. The earlier version of this probe (same fold, whole extents only, before the move to `lance-graph-benches`) measured B′/A 0.95–1.05 at 16k; this version measures 1.06–1.46 with the same A. The cause is not found. 64-byte alignment of the chunks was tried and does not change it. Either way the fold does not beat A there.
- **Reading, short runs:** at 200 rows per run, per-run `execute_extent` costs about 170–520 ns a run (S1 → S3) and the fold is 2.5–4× faster. Compiling once (`Ac`) cuts A by only 5–18 %, so re-deriving the lowering per call is a small part of the gap. From the code (not attributed by measurement): `extent_tiles` executes an unaligned extent as a head-edge word, a body and a tail-edge word, each running every op and the terminal, and `execute_compiled` validates on every call. Per-call allocation (`A+alloc`) costs nothing measurable.
- **Decision input:** a `Pred` `Lowering` arm for whole extents (phase 2 as first framed) is not supported. The measured gap is per-run cost on short runs, which is what `set_lower` produces at optics/etch. Not decided which remedy (fewer passes per edge tile in `execute_extent`, a run-batched entry point, or a short-run fold); not built.

OPEN: (1) where the per-run cost of `execute_extent` goes on a 200-row run — not attributed between edge-tile passes, validation and op dispatch; (2) why this probe's fold is slower at full tiles than the earlier version. Not measured: `u32` / strided lanes, more than two comparisons, `Keep` terminals, other tiers (AVX2, NEON), real `set_lower` run distributions.
