# High-value float/BF16 lanes — where the extra mile pays, and what calcifies (2026-07-27)

> **The qualifying test (from settled canon).** A float lane is justified iff
> ALL four hold:
> 1. the computation is **genuine entropy work** — it creates information no
>    integer read can (training, statistics, transforms), not a reorganization;
> 2. it runs **once / at a boundary**, amortized over every subsequent
>    `[a,b]` read (never per-query, never per-reasoning-step);
> 3. the **stored result is low-entropy** — palette codes, i8/u8 tables,
>    byte-sized calibration constants — never the float itself as state;
> 4. it passes a **certification gate** before any reading is trusted
>    (harness lanes: u8-CDF ρ≥0.9990 · i8 r≥0.9980 · **bf16-RNE ≥0.9999**;
>    significance per I-NOISE-FLOOR-JIRAK, never classical Berry-Esseen).
>
> BF16 specifically earns its place where the SOURCE data is already BF16
> (model weights) or where AMX `TDPBF16PS` (`amx_matmul.rs:336` —
> `C += A(bf16)×B(bf16) → f32`, byte-encoded tiles) turns the one-time build
> from hours into minutes. ndarray surface verified: `simd::{BF16x16, BF16x8,
> f32_to_bf16_batch_rne, bf16_to_f32_batch}`, `hpc::bf16_tile_gemm_16x16`,
> `hpc::quantized::{f32_to_bf16_rounded, …}`.

## The lanes, ranked by value

### L1 — Codebook training (the highest-leverage float in the system)
- **Float work:** k-means / γ-fold centroid construction over real vectors.
- **Why float:** centroid means and distance minimization are continuous
  optimization; no integer read produces a codebook.
- **Measured this session:** swapping a hand-rolled trainer for `simd::kmeans`
  moved ρ vs exact **0.8494 → 0.9725** — codebook quality is worth ~0.13 ρ,
  more than any other single knob touched all day.
- **BF16 extra mile:** training corpora ARE model weights stored BF16
  (bgz7 shards; Jina v5 safetensors is BF16 native). AMX bf16 tiles batch the
  train-time distance GEMMs; `f32_to_bf16_batch_rne` is the certified-lossless
  ingest cast (RNE lane ≥0.9999).
- **Calcifies to:** 6×256 codebook + pair tables. Build once, read forever.

### L2 — Pair-table / FisherZ baking (the certified cosine-replacement build)
- **Float work:** all pairwise cosines between centroid representatives →
  `atanh` (Fisher-z) → per-family affine.
- **Calcifies to:** `FisherZTable` — **k×k i8 (64 KB) + `FamilyGamma` 8 LE
  bytes**; `CosineGamma` 4 B/codebook. VALIDATED 2026-05-26 (10 000×10 000
  splat, θ≈1.45–1.6 ≈ cos 0.90–0.92). *"26 groups × 64 KB = 1.6 MB for the
  entire 1.7B model"* — a 1.7B-parameter model's similarity structure in 1.6 MB.
- **This is the archetype of the whole pattern:** unbounded float entropy work
  in, kilobytes of exact-at-lookup integer table out.

### L3 — Model-weight ingestion & lens baking (BF16-native by necessity)
- **Float work:** streaming GGUF/safetensors weights through the 5-lane
  encoder (thinking-engine: u8-CDF, i8, γ+φ ×2, **bf16-RNE**); CDF estimation,
  γ+φ calibration.
- **Why BF16 is not optional here:** the SOURCE dtype is BF16; the RNE lane is
  the certified-transport reference (≥0.9999 Pearson+Spearman+Cronbach) the
  other four lanes certify against.
- **Calcifies to:** baked u8/i8 lenses (the 5-lane reranker lens is **1.1 MB**),
  savant palettes (core 14.7 KB k=64; psychology/linguistics 206 KB k=256 —
  extracted from 9B∩27B / v1\v2 weight DIFFS, a float operation by nature).

### L4 — Calibration & certification statistics
- **Float work:** Pearson/Spearman/ICC/Cronbach, rolling-floor bucket
  estimation (`cascade::calibrate` — `threshold = μ + 3σ`, the σ3 = 0.9973
  band), `CosineGamma` center/spread measurement, drift scores in
  multinomial-SD units.
- **Calcifies to:** bucket bounds, γ offsets (4–8 B), and **thresholds as
  single bytes** — the VALIDATED entry's θ aperture lands as `theta_accept_q8`
  (u8) on the splat. A whole calibration campaign stores bytes.
- **Note:** this lane is also the GATE for every other lane — it cannot itself
  be retired to integer without losing the ability to certify.

### L5 — γ-fold holographic containers (BF16 as the stored residual format)
- **Float work:** `euler_gamma_fold` — γ·i rotations, √(n+γ) radii, folding N
  similar rows into one container.
- **Calcifies to:** `StackedN { data: Vec<u16> }` — **the container IS bf16**
  (`stacked_n.rs:55,64`): centroid + folded residuals at 2 B/sample, N members
  in ~2× one member's space. Recovery SNR ≈ 0.96 Pearson at documented params.
- **This is the one lane where BF16 is the low-entropy OUTPUT format itself**,
  not just the compute dtype — sanctioned because the container is a sealed
  build artifact read back through the unfold, never a mutable reasoning value.

### L6 — Ingestion-boundary encode (the front door)
- **Float work:** external embeddings (1024-dim BF16 from bge-m3/Jina) →
  per-subspace argmin against the codebook — float dot/L2, once per arriving
  row.
- **BF16 extra mile:** batch encode is a BF16 GEMM against the codebook —
  exactly `TDPBF16PS`'s shape; AMX turns bulk ingest from a cast-then-f32 pass
  into native-dtype tiles.
- **Calcifies to:** 6-byte CAM codes / palette256 pairs per row. After this
  boundary the row never sees a float again.

### L7 — The ratified per-edge uncertainty Σ (already settled; listed for completeness)
- The EWA sandwich `Σ' = M·Σ·Mᵀ` — tiny 2×2/3×3 float PSD metadata per edge,
  co-certified sibling of the integer SELECT lane (Pillar 6/7, May board).
- **Not a candidate for change in either direction:** it stays float (it is
  certified PSD algebra, not bulk arithmetic), and it never grows into a bulk
  lane. Listed so nobody "optimizes" it into u8 or cites it to justify bulk float.

## Anti-lanes (float that does NOT qualify — measured, not asserted)
- **Per-query ADC tables** — 6 144 B + 13–19 µs *per query*; 853–1 241 ms of
  pure table-building at a 64 k cohort vs the 550 ms SLA. Fails test 2
  (per-query, not amortized). The `[a,b]` static table replaces it.
- **Reasoning-path folds** (truth revision, coherence, entropy over beliefs) —
  the legacy f32 carrier surface; retirement scoped in
  `F32-RETIREMENT-SCOPE.md`, gated on the P0 tenant path.
- **Exact float scans** — 190–276 ns/cand vs 4–9 ns/cand table reads;
  55× against the doctrine at zero per-query state.

## The pattern, one line
**Float is the furnace, palette256 is the ingot.** BF16 is justified exactly
where the ore is already BF16 (model weights) or where AMX tiles make the
furnace an order of magnitude cheaper — and every product leaves the furnace
as codes, tables, and single-byte calibration constants that the standing wave
reads as `[a,b]` forever after.
