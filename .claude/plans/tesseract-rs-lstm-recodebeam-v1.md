# tesseract-rs — LSTM Forward + recodebeam Decoder v1

> **Type:** plan (sub-plan). Deliverables D-OCR-20/21/22.
> **Status:** PLANTED 2026-06-15 — design only.
> **Front:** post-#496. Forward pass targets `ndarray` (SIMD/BLAS/CLAM provider). Oracle = `tesseract-rs` FFI fork.
> **Canon anchors:** master §4; `ndarray` SIMD kernels; bit-reproducibility doctrine (DeepNSM "bit-reproducible", envelope version-stamp).
> **Skip-by-rule:** no legacy matcher; no layout. Input is a line crop, output is text + per-step posteriors.

---

## 0. Intent

Run the hydrated LSTM (D-OCR-11) forward on `ndarray` to produce per-timestep
class posteriors, then decode them with a faithful `recodebeam` (dictionary-aware,
CTC-style) to text — **byte-identical to Tesseract** on fixed line crops. This is
the tier that makes "1:1 Tesseract" provable; everything else is plumbing.

## 1. Forward pass (`tesseract-rs/src/lstm/`, on ndarray) — D-OCR-20

Faithful transcode of `lstm/` numerics. Each maps to ndarray ops:

| Tesseract unit | ndarray realization | Exactness note |
|---|---|---|
| `WeightMatrix::MatrixDotVector` (int8) | int8 GEMV via ndarray SIMD kernel | **accumulation order + rounding must match** (D-OCR-22) |
| `FullyConnected` | matmul + bias + activation LUT | activation table must be the same fixed-point LUT |
| `LSTM` cell (gates i/f/o/g, peephole) | elementwise on ndarray slices | sigmoid/tanh LUTs identical to C++ |
| `Convolve` / `Maxpool` | im2col + GEMM / window-max | stride/pad identical |
| `Softmax` / `LogSoftmax` | row softmax | only at the output; feeds the beam |

Float path is straightforward. **The int8 path is where silent drift lives** — it
is the whole point of D-OCR-22.

## 2. recodebeam decoder (`tesseract-rs/src/recodebeam.rs`) — D-OCR-21

Hand-port (NOT codegen): tie-breaking, normalization, and dawg interaction are
under-documented and behaviorally subtle.

- Beam over the **recoder** codes (not raw unichars): the `RecodeBeamSearch`
  maintains dawg-constrained and unconstrained beams; final path picks per
  Tesseract's certainty/rating rule.
- DAWG dictionary (`dict/{dawg,trie,permdawg}`) — **codegen-amenable** node-array
  walks; the *interaction* with the beam is hand-ported.
- Output: best text + per-token rating/certainty → becomes per-token confidence at
  the emit stage (master §1).

## 3. int8-SIMD numeric exactness conformance — D-OCR-22

The conformance contract that earns "1:1":

1. Pin the int8 GEMV accumulation order to Tesseract's (block/tile order matters).
2. Match the fixed-point rounding mode of `IntSimdMatrix` (AVX2/512/NEON variants
   reduce in a defined order — replicate it, do not "improve" it).
3. Identical activation LUTs (sigmoid/tanh/softmax) — copy the tables, not the
   formulae.
4. Conformance harness: feed N line crops, compare per-timestep argmax AND the
   full posterior (within 0 ULP for int8) against an FFI dump from the oracle.

## 4. Ground-truth oracle

The `AdaWorldAPI/tesseract-rs` FFI fork (thin bindings, `src/{lib,page_seg_mode}.rs`)
is built **only** as the oracle: it runs real `libtesseract` to dump (a) per-matrix
weights, (b) per-timestep posteriors, (c) final decoded text for the same crops.
The Rust port is diffed against these. The oracle is a dev/test dependency, never a
runtime path, and the lone place the Leptonica C fork is compiled.

## 5. Deliverables

- **D-OCR-20:** forward pass on ndarray reproduces C++ per-timestep posteriors
  (float path 1:1; int8 path within the D-OCR-22 contract) on a 1k-crop set.
- **D-OCR-21:** `recodebeam` + DAWG reproduces C++ decoded text byte-identical on
  the same set.
- **D-OCR-22:** int8 conformance harness green on ≥ 10k crops across 2+ languages.

## 6. Open decisions

- **OD-20a:** target one SIMD width first (AVX2) for exactness, then NEON/AVX-512;
  or define the scalar reference as canonical and treat SIMD as "must equal scalar"?
- **OD-21a:** support Tesseract's `lstm_choice_mode` (top-k per timestep) now — it
  feeds the OCR `ValueTenant` top-k candidates (ocr-soa-integration OD-1) — or later?
