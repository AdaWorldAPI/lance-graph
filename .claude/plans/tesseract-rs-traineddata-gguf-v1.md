# tesseract-rs — traineddata → GGUF → embedanything Host v1

> **Type:** plan (sub-plan, v2). Deliverables D-OCR-10/16. Replaces v1 `traineddata-ndarray`.
> **Status:** PLANTED 2026-06-15. The LSTM is HOSTED, not transcoded.
> **Host chain:** `.traineddata` → GGUF → `embedanything` DTO (candle) → `ndarray` AMX; `bgz_tensor` weight store.
> **Canon:** `.grok/NDARRAY_BGZ_EMBEDANYTHING_INTEGRATION.md`; ndarray::hpc GGUF loader.

---

## 0. Intent
Make Tesseract's recognizer "just another GGUF model behind embedanything." Parse
the `.traineddata`, extract the recognizer net + weights, **export GGUF**, and run
it on the existing inference runbook. No bespoke LSTM kernels; reuse candle's GGUF
loader + ndarray AMX + bgz_tensor storage.

## 1. Loader (`tesseract-rs/src/traineddata/`) — D-OCR-10
Parse the modern (LSTM) `.traineddata` components: `lstm` (VGSL net + weights),
`lstm-unicharset`, `lstm-recoder` (UnicharCompress), `unicharset`, `*.dawg`.
Skip legacy (`inttemp`/`normproto`/`shapetable`/adaptive). Container/unicharset/
recoder/dawg parse = **AST-DLL codegen** (D-OCR-40); VGSL net-spec = hand (tiny grammar).

## 2. GGUF export — D-OCR-10
Walk the VGSL graph → emit a GGUF model file:
- map Tesseract layers (Conv, LSTM/BiLSTM, FullyConnected, Output/Softmax) to GGUF tensors + arch metadata;
- preserve int8 quantization + per-row scales exactly (GGUF Q8 / per-tensor scale) so the hosted run can match C++ within the int8 contract;
- `bgz_tensor` stores the exported tensors (compressed, Lance-native, random-access) per its weight-store role.

## 3. Hosted run — D-OCR-16
`embedanything::infer_sequence` (the D-OCR-15 extension) loads the GGUF via candle,
runs CNN+BiLSTM on the ndarray AMX path, returns `[T, n_classes]` posteriors.
Acceptance: per-timestep posteriors match a libtesseract dump within the int8
exactness contract (float path 1:1) on a 1k-crop set. recodebeam (D-OCR-21) consumes these.

## 4. Open
- **OD-10a:** GGUF int8 (Q8_0) vs keep Tesseract's exact int8 layout in a custom GGUF kv — whichever reproduces C++ posteriors to the bit.
- **OD-10b:** candle's BiLSTM/CTC coverage — confirm candle expresses Tesseract's BiLSTM + softmax exactly, else add the missing op in the candle fork.
