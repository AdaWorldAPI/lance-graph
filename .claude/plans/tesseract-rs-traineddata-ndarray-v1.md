# tesseract-rs — traineddata → ndarray Model Loader v1

> **Type:** plan (sub-plan of `tesseract-rs-transcode-master-v1`). Deliverables D-OCR-10/11.
> **Status:** PLANTED 2026-06-15 — design only.
> **Front:** post-#496. Reuses `ndarray::hpc` weight-loading pattern (`src/hpc/{gguf.rs,gguf_indexer.rs,safetensors.rs,models/safetensors.rs}`).
> **Canon anchors:** master plan §4; `ndarray` hpc loaders; `lance-graph-contract` LE column contract.
> **Skip-by-rule:** legacy classifier `.traineddata` components (templates, adaptive) are NOT loaded.

---

## 0. Intent

Make a modern Tesseract `.traineddata` file loadable in pure Rust as a hydrated
weight set + symbol tables, using the *same* discipline as the existing GGUF /
safetensors loaders in `ndarray::hpc`. A `.traineddata` is, for our purposes, just
another model container: parse the directory of components, pull the LSTM weights,
and hydrate them into `ndarray` arrays addressable by the forward pass (D-OCR-20).

## 1. `.traineddata` is a TAR-like component bundle

Modern (LSTM) `.traineddata` holds, among ~15 components, the ones we need:

| Component | What it is | We need it for |
|---|---|---|
| `lstm` | the recognizer network (VGSL spec + weights) | D-OCR-11 weight hydration |
| `lstm-unicharset` | LSTM-specific unicharset | symbol ↔ class index |
| `lstm-recoder` | `UnicharCompress` recoder (codepoint → recode codes) | recodebeam (D-OCR-21) |
| `unicharset` | full unicharset (props, scripts, ranges) | char props / number-grammar |
| `*.lstm-*-dawg` | dictionary DAWGs (word/number/punc/system) | dict correction (D-OCR-21) |
| `version`, `config` | metadata | provenance, default params |

Components we **ignore** (legacy): `inttemp`, `pffmtable`, `normproto`,
`shapetable`, `*.params-model` (adaptive). The loader skips them by name.

## 2. Loader layout (`tesseract-rs/src/traineddata/`)

```
traineddata/
  container.rs     // TessdataManager: offset table parse → component byte slices  (CODEGEN: D-OCR-40)
  unicharset.rs    // Unicharset parse: id↔utf8, char props, script, ranges          (CODEGEN)
  recoder.rs       // UnicharCompress: codepoint↔recode-code maps                    (CODEGEN)
  vgsl.rs          // VGSL network-spec parser → layer graph                          (hand-port: small, gnarly grammar)
  weights.rs       // weight blobs → ndarray hydration (calls ndarray::hpc pattern)   (hand)
  dawg.rs          // DAWG/Trie node arrays (squished + unsquished)                   (CODEGEN)
  mod.rs           // TrainedData { net, unicharset, recoder, dicts }
```

## 3. Weight hydration — sibling of the GGUF path (D-OCR-11)

The forward pass (D-OCR-20) consumes `ndarray` views, exactly as the GGUF/Qwen
work does. `weights.rs` mirrors `ndarray::hpc::gguf`:

- Add a `ModelSource::TrainedData` alongside the existing GGUF/safetensors sources
  in `ndarray::hpc` so the LSTM weights flow through the same hydration/indexer.
- Each Tesseract `WeightMatrix` (int8 quantized + float scale, or float) becomes
  an `ndarray` 2-D array plus a per-row scale vector — preserve the quantization
  exactly (see D-OCR-22; bit-exactness depends on it).
- No transpose/normalization on load: store as Tesseract stores, defer layout to
  the forward kernels so the numeric path is auditable against the C++.

**Iron rule (inherited from the SoA envelope work):** the loader knows the LE byte
contract for every weight blob; it never guesses. The component byte slices carry
their own descriptor; hydration is a typed carve, not a reinterpretation.

## 4. Codegen vs hand-port assignment

| Module | Route | Why |
|---|---|---|
| `container.rs`, `unicharset.rs`, `recoder.rs`, `dawg.rs` | **AST-DLL codegen (D-OCR-4x)** | mechanical struct/table walks; faithful 1:1 from C++ AST |
| `vgsl.rs` | hand-port | tiny but a bespoke mini-grammar; codegen overkill |
| `weights.rs` | hand | bridges into `ndarray::hpc`; integration glue, not transcode |

## 5. Deliverables

- **D-OCR-10:** `TrainedData` loads a real `eng.traineddata` (4/5) → unicharset,
  recoder, DAWGs, and the VGSL net spec, with legacy components skipped. Test:
  round-trip the unicharset id↔utf8 against a C++ `combine_tessdata -u` dump.
- **D-OCR-11:** LSTM weights hydrated into `ndarray` via a new
  `ndarray::hpc::ModelSource::TrainedData`; shapes + quantization scales match a
  C++ dump of the same network. Test: per-matrix shape + int8 scale equality.

## 6. Open decisions

- **OD-10a:** support both float and int8 `.traineddata`, or int8-only for v1?
  (int8 is the common `*_best`/`*_fast` shipping form and the harder exactness case.)
- **OD-10b:** vendor a pinned `eng.traineddata` as a test fixture, or fetch in CI?
