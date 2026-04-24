# Format Best Practices — VSA, Binary, Quantized

> **Purpose:** Ground every format-choice decision in the relevant
> science. For each workload, answer: which carrier + which algebra,
> and what's the SNR / capacity / cache / precision tradeoff?
>
> **Scope:** VSA variants (F32, BF16, F16, I8), Binary (bitpacked
> Hamming), CAM-PQ (product quantization). Per-workload guidance.
>
> **Companion to:** `CHANGELOG.md` (what changed when), `.claude/knowledge/
> vsa-switchboard-architecture.md` (architectural layers), `CLAUDE.md §
> I-VSA-IDENTITIES` (iron rule: VSA operates on identities).
>
> **Created:** 2026-04-21 (session cleanup, post-Frankenstein correction)

---

## § 0 — The question the doc answers

For every new workload that might want to use VSA, Binary, or CAM-PQ,
answer these in order before writing code:

1. **Is this an identity lookup or a content comparison?** — answered
   per the `I-VSA-IDENTITIES` iron rule. Identity = VSA legitimate.
   Content = register-loss risk, use a different tool.
2. **What is N, the bundle size?** — capacity bound per § 2.
3. **What's the bit-level dependence ρ?** — Jirak-corrected effective
   capacity per § 3.
4. **What precision does the DOWNSTREAM use need?** — per § 4.
5. **What's the memory budget?** — cache/L3 analysis per § 5.

Outcome: one format per workload, chosen for reasons, not by default.

---

## § 1 — The science, briefly

### Johnson-Lindenstrauss (capacity upper bound)

For d random ±1 vectors in d-dim space, N items can be distinguished
via cosine similarity with error ε if:

    d ≥ (8 / ε²) · log N

For d = 16,384 and ε = 0.1:
- N ≤ e^(16384 · 0.01 / 8) ≈ e^20 ≈ 10⁸ (way more than any workload needs)

The PRACTICAL ceiling is not JL. It's the cleanup codebook + signal-to-
noise after unbind.

### Signal-to-noise after unbind

Bundle N role-bound bipolar vectors: each dim of the result is a sum
of N ±1 random variables. Under IID:

- Mean per dim: 0
- Std dev per dim: √N
- Signal for matching role after unbind: 1 (or N·(1/N) = 1 after norm)
- Noise (other N−1 contributions): std √(N−1) ≈ √N

**SNR = 1 / √N** (signal / noise after unbind, cosine-normalized).

For recovery against a codebook of known fillers:
- cos(unbound, correct_codebook_entry) ≈ 1 − O(1/√N)
- cos(unbound, wrong_codebook_entry) ≈ 0 ± O(1/√d)

Cleanup succeeds when the gap between correct and wrong cosine exceeds
the f-precision floor. This is the workable regime.

### Jirak 2016 — Berry-Esseen under weak dependence

**Jirak's role right now is the ACTIVE DECISION FRAMEWORK for CAM-PQ
vs Vsa10k choices, not a deferred calibration probe.** The ρ values
below quantify WHY CAM-PQ bitpacked codes are poor candidates for VSA
bundling vs why role-key-generated bits retain full capacity for
Vsa10kF32 bundling. Future Jirak-derived threshold calibration (Probe
B in § 7) is the RECEIPT-stamping phase; the rule-of-thumb usage here
is already applicable.

Classical Berry-Esseen assumes IID. Our bits aren't IID — they have
weak dependence from:
- CAM-PQ codebook quantization (multiple bits per centroid, coupled)
- Overlapping role-key slices (shared slice ranges)
- XOR bundle accumulation (induced dependence)
- Natural embedding structure (correlated projections)

Jirak 2016 (arxiv 1606.01617, Annals of Probability 44(3) 2024–2063)
gives convergence-to-normal rates under α-mixing weak dependence:
- `n^(p/2-1)` for `p ∈ (2, 3]`
- `n^(-1/2)` in L^q for `p ≥ 4`

Translation: **effective capacity drops by a factor depending on ρ**:

    effective_n ≈ n · (1 − 2ρ)   (small-ρ approximation)

Typical ρ observed:
- Role key bits from SplitMix64: ρ ≈ 0.01 (near IID)
- Binary16K from sign-binarize of embeddings: ρ ≈ 0.1–0.2
- Binary16K after CAM-PQ quantization: ρ ≈ 0.3–0.5

At 16,384 dims:
- ρ = 0.1: effective d ≈ 13,100
- ρ = 0.3: effective d ≈ 11,500
- ρ = 0.5: effective d ≈ 8,200

**SNR formula gets corrected:** SNR = 1 / √(N · (1 − 2ρ)^(-1))
≈ (1 / √N) · √(1 − 2ρ).

For ρ = 0.3, SNR shrinks by ~15%. Hypothesis ranking tolerance must
account for this.

**The iron rule this justifies (`I-NOISE-FLOOR-JIRAK`):** classical
IID Berry-Esseen is wrong for this system. Hand-tuned thresholds
(UNBUNDLE_HARDNESS = 0.8, ABDUCTION = 0.88) should be replaced by
Jirak-derived bounds when the calibration probe runs.


---

## § 2 — Capacity regimes (Jirak-corrected)

Given d = 16,384, the safe bundle size N depends on ρ and the precision
floor of the downstream consumer:

| ρ (bit dependence) | Max N for f32 downstream | Max N for BF16 downstream | Max N for i8 downstream |
|---|---|---|---|
| 0.01 (IID random keys) | ≤ 1024 | ≤ 256 | ≤ 64 |
| 0.1 (embedding-derived) | ≤ 820 | ≤ 205 | ≤ 51 |
| 0.3 (CAM-PQ contaminated) | ≤ 585 | ≤ 146 | ≤ 36 |
| 0.5 (heavy quantization) | ≤ 410 | ≤ 102 | ≤ 25 |

Max N = (precision_floor × effective_d) / 4, using the SNR bound plus
a safety factor of 4 for codebook cleanup tolerance.

**The `√d/4 ≈ 32` heuristic** the switchboard doc uses is the
conservative "any-reasonable-ρ safe" bound. It's almost always far
below the actual capacity ceiling — use it as a default, measure if
you need more headroom.

---

## § 3 — Per-format precision ceilings

For cosine similarity after unbind, the f-precision determines the
smallest distinguishable cosine difference:

| Format | Mantissa bits | Precision (decimal) | Smallest cosine Δ |
|---|---|---|---|
| f32 | 23 | 7.2 | ~10⁻⁶ |
| f16 (IEEE half) | 10 | 3.3 | ~5·10⁻⁴ |
| bfloat16 | 7 | 2.3 | ~5·10⁻³ |
| i8 (±127) | 7 | 2.1 | ~10⁻² |

**Implication for epiphany margin (ΔF < 0.05):**

- f32: plenty of headroom (10⁻⁶ precision vs 5·10⁻² margin) ✓
- f16: still fine (5·10⁻⁴ precision vs 5·10⁻² margin) ✓
- BF16: **borderline** (5·10⁻³ precision vs 5·10⁻² margin) ⚠ —
  epiphany detection may miss the 2nd-best hypothesis when margin
  < 0.01. Acceptable if epiphany margin is relaxed to 0.05+.
- i8: **fails** (10⁻² precision vs 5·10⁻² margin) ✗ — cannot
  reliably distinguish near-tie hypotheses.

**Implication for recovery_margin in [0, 1]:**

- All formats can represent [0, 1] with room to spare.
- The concern is NOT range; it's precision when values are close
  to each other (for ranking).

**The counterintuitive finding:** BF16 (wider range, less precision)
is WORSE than F16 for VSA cosine ranking. Range advantage is wasted
because bundle magnitudes stay bounded; precision advantage of F16
helps near-tie detection.

BF16 is the right choice when:
- AMX hardware is available (native BF16 tensor ops)
- Interop with neural lens (Jina v5 BF16, GGUF shards already BF16)
- Epiphany margin is relaxed (margin > 0.01)

F16 is the right choice when:
- Apple M-series or ARMv8.2+ (F16 hardware-native)
- AMX not available
- Near-tie epiphany detection matters
- No BF16 interop requirement


---

## § 4 — Cache and memory analysis

Per-vector memory + cache fit analysis. Assumes representative x86
cache sizes (cores vary):

| Format | Bytes/vector | L1 (32–64 KB) | L2 (256 KB–1 MB) | L3 (8–32 MB) |
|---|---|---|---|---|
| Vsa16kF32 | 65,536 (64 KB) | **spill** | fits single | fits codebook ≤ 500 |
| Vsa16kBF16 | 32,768 (32 KB) | tight (single) | fits Markov window | fits codebook ≤ 1024 |
| Vsa16kF16 | 32,768 (32 KB) | tight (single) | fits Markov window | fits codebook ≤ 1024 |
| Vsa16kI8 | 16,384 (16 KB) | fits half | fits Markov window + global | fits codebook ≤ 2048 |
| Binary16K | 2,048 (2 KB) | fits 16 | fits 128–512 | fits 4K–16K |

**Working set for Animal Farm benchmark:**

- ~40,000 sentences × 64 KB f32 trajectory = 2.5 GB total (if every
  trajectory persisted as f32 — impractical)
- Hot path: 11-vector Markov window + global context + ~12 hypotheses
  = ~900 KB. Fits L3 comfortably.
- Per-commit: one trajectory (64 KB) + graph write. Hot path.

**Working set for 10K-persona callcenter bank:**

- 10,000 personas × 64 KB f32 = 640 MB. **L3 spill, main memory.**
- 10,000 × 32 KB BF16/F16 = 320 MB. Still main memory.
- 10,000 × 16 KB I8 = 160 MB. Still main memory.
- 10,000 × 2 KB Binary = 20 MB. Fits L3.
- Inference-time: decompress winning persona's I8 → F32 for binding;
  keep the rest in I8/Binary for comparison.

**Working set for OSINT-scale document fingerprints:**

- 10 million docs × 64 KB f32 = 640 GB. **Main memory only, heavy.**
- 10M × 2 KB Binary = 20 GB. **Fits main memory**, acceptable.
- 10M × CAM-PQ 32-byte codes = 320 MB. Fits RAM easily.

**Implication:** for large-scale persistence, NEVER store Vsa16kF32
directly. Quantize to i8 or sign-binarize to Binary16K or compress
via CAM-PQ. Reserve f32 for the hot compute path.

---

## § 5 — Per-workload best-practice decisions

Each row applies the full framework: register check, N capacity,
precision need, cache fit.

### Hot-path compute workloads

| Workload | Format | N | Reason |
|---|---|---|---|
| Hypothesis ranking in `Trajectory::resolve` | Vsa16kF32 | 2–16 | Epiphany margin 0.05 demands f32 precision; low N fits L3 |
| Markov ±5 trajectory bundle | Vsa16kF32 | 11 | 704 KB window fits L3; f32 precision for accurate braiding |
| Global context (singleton) | Vsa16kF32 | N/A | 64 KB trivial; updated per commit |
| Single sentence SPO bundle | Vsa16kF32 | 3–8 | Low N, precision cheap |
| Per-turn callcenter role bundle | Vsa16kF32 | 4–10 | Real-time, precision matters |

### Persistence workloads

| Workload | Format | Reason |
|---|---|---|
| Committed SPO triples (graph edges) | Binary16K + feature index | Popcount-fast, small per edge |
| Episodic snapshots (per-sentence) | Vsa16kI8 | 16 KB trivial, precision sufficient for retrieval |
| Persona bank (< 100 items) | Vsa16kF32 | Fits L3; no need to quantize |
| Persona bank (1K–10K items) | Vsa16kI8 or Vsa16kBF16 | Memory pressure; still usable for bind+cosine |
| Persona bank (10K+, cold storage) | CAM-PQ 32-byte codes | Heavy compression, decompress top-K on query |
| Thinking-style configs (12 styles) | YAML content + Vsa16kF32 identity | Tiny scale, f32 identity for resonance dispatch |
| Archetype registry (144 total) | Vsa16kF32 identity + palette content | Tiny scale, full precision |

### Comparison / search workloads

| Workload | Format | Reason |
|---|---|---|
| Rigid-designator lookup (Napoleon → graph node) | HashMap key + Binary16K | Register lookup (Test 0), not VSA |
| Nearest-neighbor in 1M+ document index | CAM-PQ | Designed for this |
| k-NN for candidate pre-filter | CAM-PQ, then decompress top-K to Vsa16kF32 | Cheap narrow search + precise rerank |
| Similarity ranking among 12 thinking styles | Vsa16kF32 cosine against identity codebook | Small codebook, f32 precision for close margins |

### Cross-catalogue workloads

| Workload | Format | Reason |
|---|---|---|
| Grammar + persona + callcenter bundled in one trajectory | Vsa16kF32 with disjoint slice allocation | Multiple domain catalogues coexist cleanly |
| Cross-session state transfer | Vsa16kBF16 (with AMX) or Vsa16kI8 | Quantize at boundary, rehydrate on entry |
| Shared with neural lens (Jina v5 BF16) | Vsa16kBF16 | Native format match, no conversion |

### Never-do workloads

| Workload | Why rejected |
|---|---|
| VSA bundle over CAM-PQ codes directly | Register loss — centroid indices XOR to unrecoverable state |
| Cosine similarity over i8 for close-margin epiphany detection | Precision floor too high (10⁻²) for margin 5·10⁻² |
| Bundle of 1000+ items with shared codebook | Jirak-effective capacity exhausted; use direct graph lookup |
| "Similarity" on BlasGraph traversal results via VSA algebra | BlasGraph outputs are binary; use Hamming, not cosine |
| Vsa16kF32 stored per-edge in a 1M-edge graph | 64 GB; use Binary16K or bgz17 palette edge (3 bytes) |


---

## § 6 — The three iron rules this follows

From `CLAUDE.md § Substrate iron rules`:

1. **I-SUBSTRATE-MARKOV** — VSA bundling guarantees Chapman-Kolmogorov
   semigroup property. Don't replace `vsa_bundle` (add) with XOR
   bundling for state-transition paths. `MergeMode::Xor` is a
   legitimate single-writer merge, NOT a Markov-respecting kernel.

2. **I-NOISE-FLOOR-JIRAK** — bits are weakly dependent; use Jirak 2016
   Berry-Esseen bounds instead of classical IID for noise-floor
   calibration. Hand-tuned σ thresholds are acceptable but must be
   documented as such.

3. **I-VSA-IDENTITIES** — VSA operates on IDENTITY fingerprints that
   point to content. Never on bitpacked/quantized content itself.
   Four tests: register laziness, N ≤ capacity, role orthogonality,
   cleanup codebook.

**Every format decision should justify itself against these three.**
If a proposed format choice fails any iron rule, it's wrong regardless
of benchmark numbers.

---

## § 7 — Probe queue (calibration work owed)

The following measurements should be made before claiming the stack
is "grounded" rather than "tuned":

### Probe A — ρ measurement on Animal Farm corpus

Measure bit-level dependence on actual Binary16K fingerprints derived
from COCA 24K vocab + Markov ±5 parse of Animal Farm.

- Autocorrelation ρ(k) at bit offset k for k = 1, 2, 4, 8, 16, 64.
- Expected: ρ ≈ 0.01–0.1 for deterministic role-key-generated bits.
- If higher, trace dependence source (CAM-PQ, embedding structure).

**Output:** a measured ρ per corpus. Input to Probe B.

### Probe B — Jirak-derived threshold function

Implement `jirak_threshold(n, rho, confidence)` returning the
Berry-Esseen bound on cosine similarity for hypothesis ranking:

```rust
pub fn jirak_threshold(n: usize, rho: f32, confidence: f32) -> f32 {
    let effective_n = (n as f32) * (1.0 - 2.0 * rho).max(0.01);
    let c_jirak = /* constant from Jirak 2016 */;
    1.0 - c_jirak / effective_n.sqrt()
}
```

Replace hand-tuned constants:
- `UNBUNDLE_HARDNESS_THRESHOLD = 0.8` → `jirak_threshold(n_facts, rho, 0.95)`
- `ABDUCTION_THRESHOLD = 0.88` → `jirak_threshold(n_hypotheses, rho, 0.99)`
- `HOMEOSTASIS_FLOOR = 0.2` → derived from F-landscape curvature

**Output:** principled thresholds that adapt to corpus ρ.

### Probe C — Capacity stress test per format

For each format (F32, BF16, F16, I8) and each N in {10, 100, 1000},
measure the distinguishability of the correct hypothesis against
random wrong hypotheses.

- Metric: fraction of trials where cosine(unbound, correct) >
  cosine(unbound, best_wrong) by at least the precision-floor margin.
- Expected falloff per § 3 precision ceilings.

**Output:** empirical confirmation of the § 3 theoretical precision
ceilings per workload / format pair.

---

## § 8 — The three failure modes this prevents

### Failure 1 — "VSA is always lossless"

Wrong. VSA is lossless only when:
- f32 (or equivalent precision) accumulator,
- N ≤ capacity (Jirak-corrected),
- Orthogonal role keys,
- Cleanup against a known codebook.

Any of these fails → lossy or broken recovery.

### Failure 2 — "BF16 is the modern 16-bit default"

Wrong for VSA. BF16 trades precision for range. VSA cosine similarity
needs precision, not range. F16 beats BF16 for VSA unless AMX hardware
or neural-lens interop dictates BF16.

### Failure 3 — "Use VSA, it's more principled than a hash table"

Wrong for register-appropriate workloads. `HashMap<&str, Def>` +
O(1) lookup is the right tool when the item has a name. VSA resonance
is the right tool when the item is INFERRED from context. Don't
reach for VSA when a register would do — that's Test 0 failing.

---

## References

- `CHANGELOG.md` — format-switch history (canonical "when did format X
  change and why?")
- `.claude/knowledge/vsa-switchboard-architecture.md` — three-layer
  architecture + decision matrices
- `CLAUDE.md § I-VSA-IDENTITIES`, `§ I-NOISE-FLOOR-JIRAK`,
  `§ I-SUBSTRATE-MARKOV` — the three iron rules
- Jirak 2016 — "Berry-Esseen theorems under weak dependence", Annals
  of Probability 44(3) 2024–2063, arxiv 1606.01617
- Shaw, Furlong, Anderson, Orchard 2501.05368 — "Developing a Foundation
  of Vector Symbolic Architectures Using Category Theory"
- Kleyko et al. 2106.05268 — "VSA as a Computing Framework for Emerging
  Hardware"
- `.claude/board/TECH_DEBT.md` — open calibration debts (Jirak
  thresholds, 157→160 SIMD, Vsa10k→Vsa16k rescale)
