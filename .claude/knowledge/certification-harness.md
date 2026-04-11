# Certification Harness — technical reference

**Agent**: `.claude/agents/certification-officer.md`
**Canonical primitives**: `ndarray::hpc::*`, `bgz_tensor::{quality, gamma_phi}`, `thinking_engine::cronbach`
**Reuses**: `crates/thinking-engine/examples/seven_lane_encoder.rs` + `ndarray::hpc::safetensors::stream_index_safetensors_bf16`

> **Purpose**: prove that a derived format (lab BF16, Base17, γ+φ i8,
> palette, bgz-hhtl-d) preserves the semantic properties of its source
> file to a quantified target, and produce a reproducible JSON report.

---

## The three metrics (all reported to 4 decimal places)

### Pearson r

Linear correlation between reference cosines and derived cosines.
Measures whether the derivation is a linear function of the reference.

Computed via `bgz_tensor::quality::pearson(x: &[f64], y: &[f64]) -> f64`
(at `crates/bgz-tensor/src/quality.rs:13`). Returns a value in
`[-1, 1]`; the certification target is a lower bound.

### Spearman ρ

Rank correlation. Measures whether the derivation preserves the sort
order of pairs by similarity. **The critical metric for retrieval
quality** — a format can drift by a constant scale factor and still
pass Pearson, but Spearman catches any rank inversion.

Computed via `bgz_tensor::quality::spearman(x: &[f64], y: &[f64]) -> f64`
(at `crates/bgz-tensor/src/quality.rs:47`). Internally converts both
inputs to ranks via `fn ranks(values: &[f64]) -> Vec<f64>` (same file,
line 59) with average-for-ties resolution.

### Cronbach α

Internal consistency across lenses. Measures whether the derivation
behaves as the same psychometric instrument as the reference. In a
two-lens setup (`ref_col`, `der_col`), α quantifies how much the two
lenses agree item-by-item (each pair is an item).

Computed via `thinking_engine::cronbach::cronbach_alpha(items: &[&[f32]]) -> f32`
(at `crates/thinking-engine/src/cronbach.rs:27`). The harness converts
f64 cosine columns to f32 at the boundary (this is lossless because
cosine values are in `[-1, 1]` which has ~8 bits of relevant range,
well within f32 precision).

---

## Target thresholds (per-lane, per-derivation class)

| Lane | Format       | Required metric          | Threshold | Entry tax reason |
|------|--------------|--------------------------|-----------|------------------|
| 1    | `u8` CDF     | Spearman ρ               | ≥ 0.9990  | 256-level quantization |
| 2    | `i8` direct  | Pearson r                | ≥ 0.9980  | 127× linear, 1/127 ULP |
| 3    | `u8` γ+φ     | Spearman ρ               | ≥ 0.9990  | γ+φ + 256-level |
| 4    | `i8` γ+φ     | Pearson r                | ≥ 0.9980  | γ+φ + signed 127× |
| 5    | `f32` delta  | ‖delta‖ norm (reported)  | no threshold | non-zero only for gate tensors |
| 6    | `bf16` RNE   | Pearson, Spearman, Cronbach α | **≥ 0.9999** | 3-bit mantissa drop RNE |
| 7    | `u8` drift   | mean / max drift (reported) | no threshold | quality signal |

**Lane 6 is the "atomic clock" lane** — it must hit 0.9999 or better
on all three metrics simultaneously. It is the only lane certified
against the full three-metric battery; the other lanes have the
single most-appropriate metric as the primary pass gate.

**Lane 5 and Lane 7 are informational**: the report records their
statistics but they do not participate in the pass/fail decision.

**Verdict aggregation**: the overall result is `PASS` iff lanes 1, 2,
3, 4, 6 all hit their thresholds on all three sample sets (CLAM
centroids, random pairs, corpus pairs). Any single miss is `FAIL`
with diagnostic output.

---

## Source access modes

### Mode A: Local mmap (source fits on disk)

Used for Jina v5 (1.19 GB), BGE-M3 (1.5 GB), ModernBERT, and any
other model below ~12 GB given the current ~13 GB disk budget.

```rust
let file = File::open(source_path)?;
let mmap = unsafe { MmapOptions::new().map(&file)? };  // zero-copy
// Per-pair row reads seek into mmap at
//   tensor_data_offset + token_idx * hidden_dim * dtype_bytes
// No Vec<f32> buffer is allocated. The upcast runs inline in a
// stack window when each row is needed.
```

Peak RSS: ~50 MB (metric accumulators + small stack windows).
Peak disk: whatever is already on disk.

### Mode B: HTTP streaming (source does not fit)

Used for Qwen 3.5 9B (~18 GB), 27B (~55 GB), and up to 397B at
~800 GB. This is the ONLY mode for the large variants. Uses the
canonical pattern already battle-tested by
`ndarray::hpc::safetensors::stream_index_safetensors_bf16`:

```rust
use ndarray::hpc::http_reader::HttpRangeReader;

let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, shard);
let size = head_content_length(&url)?;
let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);
// 256 MB read-ahead chunks. 128 MB is acceptable on memory-
// constrained systems but should be a power of 2 and ≥ 64 MB.
```

The certification harness does a **two-pass** run:

1. **Extraction pass**: stream the source sequentially in chunks,
   check which of the sampled token rows have byte offsets inside
   the current chunk, extract those rows on the fly into a small
   local file `{source_slug}_extracted_rows.bin` (~4 MB total for
   1000 pairs × 2 rows × 1024 × 2 bytes).
2. **Certification pass**: mmap the ~4 MB extraction file and run
   the standard Mode-A pipeline on it. Pair indices are remapped
   to local indices `[0, 2*n_pairs)`.

Extraction file layout (little-endian, no alignment padding):

```
[ 32 bytes ]  source SHA-256
[  8 bytes ]  sampler seed (u64)
[  4 bytes ]  n_pairs (u32)
[  4 bytes ]  hidden_dim (u32)
[  2 bytes ]  dtype_code  (0 = F32, 1 = BF16, 2 = F16)
[  2 bytes ]  reserved
[ n_pairs × 8 bytes ]  token indices (u32 pair_a, u32 pair_b)
[ n_pairs × 2 × hidden_dim × dtype_bytes ]  row data
```

A re-run can skip the extraction pass if the extraction file exists
AND its first 48 bytes match (sha256 + seed + n_pairs + hidden_dim +
dtype_code). Any mismatch invalidates and triggers a fresh extraction.

Peak disk during extraction: ~4 MB output + 256 MB HTTP window
(transient).
Peak RSS during extraction: ~300 MB (HTTP buffer + metadata).
Peak RSS during certification: ~50 MB (same as Mode A).

---

## Reuse of `seven_lane_encoder.rs`

The 7-lane encoder at `crates/thinking-engine/examples/seven_lane_encoder.rs`
is the canonical lane-derivation pipeline. The certification harness
**calls** it — does not replace it. Responsibilities split:

- **Encoder** produces the 256² CLAM-centroid cosine matrix and the 7
  lane output files. Writes `jina-v5-7lane/cosine_matrix_256x256.f32`,
  `distance_table_256x256.{u8,i8,bf16,gamma_phi.u8,gamma_phi.i8}`,
  `silu_deltas_256x256.f32`, `spiral_drift_256x256.u8`, and
  `encoding_metadata.json`.
- **Certification officer** reads those outputs AND re-runs the
  lane-derivation logic on an independent random-pair sample,
  computes the three metrics per lane × per sample set, and writes
  the JSON report.

### Pending fix in `seven_lane_encoder.rs`

Lane 6 (`lane6_bf16`) currently uses the workspace's legacy
`f32_to_bf16` truncation path. For the lab-BF16 atomic-clock goal,
swap the two call sites around line 288 and line 291 to use
`ndarray::simd::f32_to_bf16_scalar_rne` (round-to-nearest-even,
byte-exact vs hardware `_mm512_cvtneps_pbh`). The swap is surgical:

```rust
// BEFORE
lane6_bf16[i * n_cent + i] = f32_to_bf16(1.0);
// ...
let bf = f32_to_bf16(cos);

// AFTER
lane6_bf16[i * n_cent + i] = ndarray::simd::f32_to_bf16_scalar_rne(1.0);
// ...
let bf = ndarray::simd::f32_to_bf16_scalar_rne(cos);
```

This is a required prerequisite to lab-BF16 certification — without
it, Lane 6 drifts ~1 ULP from hardware on ~50 % of cosines and
cannot hit the 0.9999 target.

For pure-batch encoding of many values at once, use
`ndarray::simd::f32_to_bf16_batch_rne(input: &[f32], output: &mut [u16])`
instead of the scalar routine for ~16× throughput.

---

## NaN scan protocol

Every pipeline stage runs a NaN scan. A single NaN invalidates the
certification until the NaN is explained (or the source is known to
contain NaN as structural state, which is unusual for trained models).

Scan points:

1. **Raw source bytes** (optional, skipped if dtype is integer):
   scan the tensor data region for F32/BF16/F16 bit patterns that
   decode to NaN. For BF16, any value `b` with `b & 0x7F80 == 0x7F80
   && b & 0x007F != 0` is NaN. For F16, `b & 0x7C00 == 0x7C00 && b &
   0x03FF != 0`. For F32, `b & 0x7F800000 == 0x7F800000 && b &
   0x007FFFFF != 0`.
2. **After upcast to F32**: use `f32::is_nan()` on every element of
   the stack window. A NaN here means the upcast is broken (should
   never happen post the Apr 11 2026 NaN quiet-bit fix in
   `ndarray::hpc::gguf::f16_to_f32`).
3. **After cosine computation**: check the reference and derived
   cosines for NaN. NaN here means either (a) a zero-norm vector
   crept into the sample or (b) the derivation introduced a
   numerical instability.
4. **After metric computation**: Pearson / Spearman / Cronbach
   should never return NaN given non-NaN inputs. If they do, the
   accumulator has a divide-by-zero (pathological input).

If any scan fires, halt, write a NaN diagnostic to the JSON report
in a `nan_events` array, and exit with code 2 (distinct from a
normal FAIL which is code 1).

```json
"nan_events": [
  {
    "stage": "cosine_reference",
    "pair_index": 742,
    "token_a": 58193,
    "token_b": 140552,
    "vector_a_norm": 0.0,
    "vector_b_norm": 1.0
  }
]
```

---

## Output JSON format

File path: `.claude/knowledge/certification/{source_slug}_{derivation_slug}.json`

```json
{
  "source": {
    "path": "crates/thinking-engine/data/jina-v5-onnx/model.safetensors",
    "sha256": "030f08ea0e2be8a58eb549a9daa90e9ed3b5db3f562df2625eee26b3ef1c5baf",
    "tensor": "embed_tokens.weight",
    "dtype": "BF16",
    "shape": [151936, 1024],
    "bytes": 311164928
  },
  "derivation": {
    "class": "7_lane_encoder",
    "revision_commit": "c489d31",
    "lane_6_method": "f32_to_bf16_scalar_rne",
    "n_centroids_clam": 256
  },
  "sample_sets": {
    "clam_centroids": { "n_pairs": 32640, "seed": null },
    "random": { "n_pairs": 1000, "seed": "0x9E3779B97F4A7C15", "range": [0, 151669] },
    "corpus": { "n_pairs": 21, "source": "jina_v5_ground_truth.rs tier-1..4" }
  },
  "nan_scan": { "passed": true, "events": [] },
  "metrics": {
    "lane_1_u8_cdf": {
      "clam_centroids": { "pearson": 0.9992, "spearman": 0.9999, "cronbach_alpha": 0.9995 },
      "random":         { "pearson": 0.9988, "spearman": 0.9995, "cronbach_alpha": 0.9991 },
      "corpus":         { "pearson": 0.9983, "spearman": 0.9997, "cronbach_alpha": 0.9987 }
    },
    "lane_2_i8_direct":    { "...": "..." },
    "lane_3_u8_gamma_phi": { "...": "..." },
    "lane_4_i8_gamma_phi": { "...": "..." },
    "lane_5_silu_delta":   { "l2_norm": 0.0, "note": "zero for embed_tokens" },
    "lane_6_bf16_rne": {
      "clam_centroids": { "pearson": 0.99997, "spearman": 0.99995, "cronbach_alpha": 0.99996 },
      "random":         { "pearson": 0.99995, "spearman": 0.99994, "cronbach_alpha": 0.99995 },
      "corpus":         { "pearson": 0.99999, "spearman": 1.00000, "cronbach_alpha": 0.99999 }
    },
    "lane_7_spiral_drift": { "mean": 0.0955, "max": 0.4336, "p99": 0.2812 }
  },
  "verdict": {
    "overall": "PASS",
    "lane_1": "PASS",
    "lane_2": "PASS",
    "lane_3": "PASS",
    "lane_4": "PASS",
    "lane_6": "PASS",
    "notes": []
  },
  "runtime": {
    "total_seconds": 8.3,
    "peak_rss_mb": 52.1,
    "stage_breakdown": {
      "inventory": 0.1,
      "clam_256": 4.2,
      "lanes_1_4": 1.1,
      "lane_6_rne": 0.3,
      "lane_7_spiral": 0.8,
      "metrics": 0.5,
      "report": 0.1,
      "nan_scan": 1.2
    }
  },
  "provenance": {
    "branch": "claude/risc-thought-engine-TCZw7",
    "commit": "86e586e",
    "workspace_primer_version_rules": ["7", "8", "21", "22", "23"],
    "agent": "certification-officer.md"
  }
}
```

---

## How to certify a new model

1. Download the model safetensors (or confirm it is on disk).
2. Wake `certification-officer` with the source path.
3. Agent reads `CLAUDE.md § Certification Process`, this doc, and
   the source header.
4. Agent runs the existing `seven_lane_encoder.rs` to produce the
   7-lane tables (with the Lane 6 RNE swap applied in-place if not
   already done).
5. Agent runs the additional SplitMix64 + corpus pair samples
   through the lane-derivation logic (reused from the encoder).
6. Agent computes metrics and writes the JSON report.
7. Agent returns a one-line summary + the JSON path.

---

## Retest policy

Any certification result where the source or derivation touched the
pre-Apr-11-2026 `ndarray::hpc::gguf::f16_to_f32` primitive is a
retest candidate per `workspace-primer.md` Rule 22. Retest by:

1. Regenerating the 7-lane encoder output under the current
   (post-NaN-fix) primitives.
2. Running a fresh certification against the new lane outputs.
3. Comparing the new metrics against the previous recorded metrics
   byte-for-byte.
4. If the delta is zero or within f32 rounding, the old result is
   exonerated. If the delta is material, the old result is
   invalidated and the new numbers replace it.

---

## One sentence that should survive any refactor

**Reuse the 7-lane encoder for lane derivation, add the certification
layer via SplitMix64 independent validation + 4-decimal Pearson /
Spearman / Cronbach α at every lane, NaN scan at every stage, and
fail loud; the atomic clock is Lane 6 at ≥ 0.9999.**
