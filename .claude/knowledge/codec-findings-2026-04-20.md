# Codec Findings — What Works, Under What Constraints (2026-04-20)

> **READ BY:** agents working on bgz-tensor codec candidates, bgz17
> palette, CAM-PQ, attention compression, quantization research. Before
> proposing new codecs, read this document so you don't re-derive
> measurements that already exist.
>
> **Source:** this session's measured results on Qwen3-8B L0 k_proj /
> gate_proj / q_proj, 128 rows each, through `codec_rnd_bench.rs`.
> Commits: fc386bb (fractal leaf) → 1d51d21 (zipper family complete).

---

## Does the zipper family fix the argmax blind spot?

**No.** The argmax blind spot (Invariant I2 — L2-clustering codecs fail
on near-orthogonal high-dim rows) is ALREADY fixed by:

- `Had-Q5×D-R` (shared codebook) — ICC 0.989
- I8-Hadamard (per-row) — ICC ~0.9

The zipper family maxes out at ICC 0.20 (Zipper-Full, 64 B). It does
NOT compete for argmax correctness. It fixes **different** blind
spots: no-codebook-calibration need, progressive decode, bundling
with negative cancellation, Fibonacci-weighted superposition capacity.

If your problem is "I need argmax ICC > 0.9 at minimal bytes" → use
Had-Q5×D-R / I8-Hadamard. Do not expect the zipper to displace them.

If your problem is "I need a codec that works on an unknown new
population without retraining, AND supports bundling, AND truncates
progressively" → the zipper is the only candidate in the current
sweep that satisfies all three simultaneously, at the cost of ICC ≈ 0.2
vs 0.9.

---

## TL;DR

1. **For argmax compression with freedom to ship a shared codebook**:
   `Had-Q5×D-R` already in the 67-codec sweep delivers ICC ≈ 0.99 at
   ~0 per-row bytes. Nothing else comes close. **Use it.**
2. **For argmax compression with per-row-only storage** (no shared
   codebook allowed): I8-Hadamard at 9 B/row is the existing leader
   (ICC ≈ 0.9). The zipper family tops out at ICC ≈ 0.2 on q_proj;
   it does NOT beat I8-Hadamard on pure ICC at any byte budget tested.
3. **Fractal leaf row-level statistics are empirically DEAD.** Both
   magnitude envelope (D, w, σ, H) and phase flip-density are
   **sign-flip invariant** — WHT linearity makes cos(x, −x) = 1.0 in
   descriptor space but −1.0 in ground truth → ICC collapses to
   −0.999. Do not pursue without breaking the invariance.
4. **Zipper codecs have a distinct Pareto axis** (bundling +
   progressive-matryoshka + anti-moiré-by-construction without
   codebook), not pure ICC. Use only when those properties matter.

---

## Invariants established by measurement

**I1 — Sign-flip invariance kills argmax ICC.** Any descriptor whose
value is unchanged under row negation (variance, flip count, L2
magnitude, absolute-Hadamard statistics) will produce ICC ≈ −1 on
argmax populations. Ground truth separates (x, −x) at cos = −1; invariant
descriptors merge them at cos = +1. Perfect ranking inversion.

**I2 — Per-row normalization destroys inter-row magnitude info.** If
the codec divides by a per-row max-abs or L2 norm before quantization,
all rows land in the same [-1, 1] range → inter-row magnitude
differences vanish → only shape is preserved. Observed empirically:
Zipper-I8-φ(64B) with per-row μ-law normalization scores 0.153 vs
Zipper-7^7×7(18B) with GLOBAL scale at 0.144 — same order, 3.5× more
bytes for no additional ICC.

**I3 — Maximally-irrational strides beat harmonic strides for argmax.**
Quintenzirkel stride (log₂(3/2), ≈ 0.585) loses to φ-stride (1/φ ≈ 0.618)
across every tested byte budget on q_proj. Conclusion: for
argmax-regime codec sampling, choose stride based on
maximal-irrationality, not harmonic proximity. Quintenzirkel may still
win on other tasks (progressive-perceptual decode, music) — not this one.

**I4 — Aperiodic φ-stride sampling beats linear dyadic sampling when
the signal has butterfly structure** (proven by Zipper-Phase at 8 B
beating Base17 at 34 B). The X-Trans / family-zipper principle is
real when the probe avoids the transform's own frequencies.

**I5 — Sign bits carry less information per byte than i8 values, BUT
sign bits avoid the quantization-noise + normalization pitfalls that
plague i8 codecs.** Sign-only Zipper-Full (64 B) scored ICC 0.204;
Zipper-I8-φ(64B) only 0.153. Lesson: if you can't use a global
population-calibrated quantization scale, sign-only outperforms
naive per-row i8.

---

## Measured codec hierarchy (q_proj, Qwen3-8B L0)

| Codec | Bytes | ICC_3_1 | Domain of applicability |
|---|---|---|---|
| Passthrough | row × 4 | 1.000 | Index regime, exact recovery required |
| **Had-Q5×D-R** (shared codebook) | 0/row | **0.989** | **Argmax regime, codebook deployable** |
| I8-Hadamard (est) | 9/row | ~0.9 | Argmax regime, per-row-only |
| Zipper-Full (sign+mag) | 64 | 0.204 | Argmax, need no-codebook + bundling |
| Zipper-Full I8-φ | 64 | 0.153 | — (dominated by sign-full at same bytes) |
| Zipper-7^7×7 | 18 | 0.144 | Argmax, compact + progressive decode |
| Zipper-Phase (sign only) | 8 | 0.097 | Argmax, absolute minimum bytes |
| Zipper-5^5×5 | 10 | 0.066 | — (dominated by 7^7×7 at similar size) |
| Base17 | 34 | 0.024 | — (dominated by Zipper-Phase at 1/4 bytes) |
| Zipper-5^5 | 2 | 0.021 | Minimum-byte coarse signature |
| Zipper-7^7 | 3 | 0.028 | Minimum-byte, finer than 5^5 |
| Fractal-Desc (magnitude) | 7 | **−0.996** | DEAD — sign-flip invariance |
| Fractal-Phase (flip density) | 5 | **−0.997** | DEAD — sign-flip invariance |
| Fractal+Base17 | 41 | −0.488 | DEAD — fractal contaminates Base17 |

**Newly-discovered Pareto point: Zipper-7^7×7 at 18 B/row, ICC 0.144.**
Fills the gap between Base17 (34 B, 0.024) and Zipper-Full (64 B, 0.20).
First bipolar-signed codec to reach >0.1 ICC without a shared codebook.

---

## Decision tree — which codec for which constraint

```
Can you ship a shared codebook (per-role / per-layer)?
├── YES → Had-Q5×D-R (ICC 0.989 / 0 B-per-row). Done.
└── NO — per-row-only storage required
    │
    ├── Do you need progressive / matryoshka decode?
    │   (read 3 B for coarse, 18 B for fine, 64 B for full)
    │   ├── YES → Zipper-7^7 truncation hierarchy
    │   │        (3 B → 18 B → 64 B continuum)
    │   └── NO
    │       │
    │       ├── Do you need VSA-style bundling
    │       │   with negative cancellation?
    │       │   ├── YES → Zipper-5^5 or Zipper-7^7 (bipolar)
    │       │   └── NO → I8-Hadamard at 9 B (existing leader)
    │       │
    │       └── Is per-row identity required?
    │           (index-regime tensor — embedding, lm_head)
    │           ├── YES → Passthrough or SpiralEncoding
    │           │        (no compression survives Invariant I1)
    │           └── NO → I8-Hadamard at 9 B
    │
    └── Exotic: need anti-moiré without codebook calibration?
        (novel population, no prior Hadamard profile)
        └── Zipper-Full (64 B). Anti-moiré by construction via
            φ-stride; no training needed.
```

---

## What NOT to do (measured dead ends)

1. **Do not compute row-level fractal statistics** (MFDFA, flip density,
   Hurst, spectrum width) on Hadamard-rotated coefficients and use them
   as a codec. Sign-flip invariant → ICC → −1.
2. **Do not use per-row max-abs normalization before quantization**
   in an argmax codec. Inter-row magnitude info destroyed.
3. **Do not use Quintenzirkel stride** for argmax sampling — measured
   worse than φ-stride at every tested size.
4. **Do not blend a high-ICC codec (e.g., Base17) with a sign-flip-
   invariant descriptor.** The invariant component drags the combined
   score toward −ICC(invariant) ≈ −1.
5. **Do not expect fractal shape parameters to recover per-row identity.**
   Fractal descriptors produce "statistical twins" — same shape, different
   coefficient assignments. Usable for argmax-rank if sign is preserved
   separately (e.g., Zipper-Full), useless for index-regime.

---

## Unmeasured probes (probe queue)

These are still open questions — no measurement has ruled them in or out:

1. **MRI-style differential phase** — N Hadamard rotations with
   different perturbations, sample phase at each, aggregate inter-view
   deltas. Sidesteps sign-flip invariance by measuring differences, not
   absolutes. Predicted ICC ≥ 0.3 at 32 B based on audio/MRI precedent.
2. **Fibonacci-weighted bundling** — Zeckendorf-decomposition-decoded
   bundle with 256-signal capacity in i8 (vs standard ~15). Measured
   at retrieval bench (not the current pair-cosine bench). Predicted
   log-rank recovery ≈ 8 reliable signals at F(13) = 233.
3. **Audiophile multi-band precision** — population-calibrated
   non-uniform bit allocation: 8 bits for top-16 |coef| positions,
   3 bits for middle-48, sign-only for bottom. Total ~20 B. Predicted
   ICC ≥ 0.4 (matches Opus/CELT perceptual coding patterns).
4. **JL multi-view phase cleaning** — N random JL projections, phase
   at each, bit-vote aggregate. Predicted √N SNR improvement on phase
   stream.
5. **Gamma-calibrated global scale** — instead of population-median
   `global_scale` for 5^5/7^7, calibrate via ICC-optimization
   (grid-search on held-out rows). Expected to sharpen the discrimination
   thresholds. +0.02-0.05 ICC.

Each is a ~50-100 LOC candidate + one bench run (~20 min).

---

## How to add a new codec candidate (recipe)

```rust
// crates/thinking-engine/examples/codec_rnd_bench.rs
#[cfg(feature = "lab")]
struct MyCodec { /* state */ }

#[cfg(feature = "lab")]
impl CodecCandidate for MyCodec {
    fn name(&self) -> &str { "My-Codec(NB)" }
    fn bytes_per_row(&self) -> usize { N }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        // 1. Encode each row to your descriptor
        // 2. Compute pairwise similarity in descriptor space
        // 3. Return n*(n-1)/2 scores in upper-triangle order
    }
}

// Register in main() under the lab-gated block:
codecs.push(Box::new(MyCodec { /* init */ }));
```

Run: `cargo run --release --features lab --manifest-path crates/thinking-engine/Cargo.toml --example codec_rnd_bench -- /path/to/shard.safetensors`. Wall time ~20 min on Qwen3-8B shard 1.

ICC_3_1 is the key metric. Top-5 recall shows whether argmax neighbors
are preserved. Pearson r vs Spearman ρ divergence reveals non-linear
bias (calibratable) vs random (fundamental).

---

## Files touched by this research (2026-04-19 / 20)

Lab-gated code:
- `crates/bgz-tensor/src/fractal_descriptor.rs` — MFDFA magnitude (DEAD)
- `crates/bgz-tensor/src/zipper.rs` — sign + I8 + 5^5 + 7^7 variants
- `crates/bgz-tensor/examples/fractal_probe.rs` — HF streaming probe
- `crates/thinking-engine/examples/codec_rnd_bench.rs` — lab candidates

Ledger:
- `.claude/board/EPIPHANIES.md` — 5 dated findings covering the full arc
- `.claude/board/IDEAS.md` — zipper architecture, fractal round-trip
- `.claude/skills/cca2a/procedure-bookkeeping.md` — three-pass recipe

Cross-reference these before proposing new codec probes.
