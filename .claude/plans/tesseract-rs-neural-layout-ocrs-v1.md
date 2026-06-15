# tesseract-rs — Neural Layout + Recognition via ocrs/rten v1

> **Type:** plan (sub-plan). Deliverables D-OCR-30/31.
> **Status:** PLANTED 2026-06-15 — design only. Ships FIRST (POC default recognizer).
> **Front:** post-#496. Uses `AdaWorldAPI/{ocrs,rten,tract,ort}` forks as-is.
> **Canon anchors:** master §2 (engine trait), §3 (skip layout). Pure-Rust posture (no C/C++ deps except opt-in `ort`).
> **Skip-by-rule:** this plan is *why* Tesseract `textord` is never transcoded — neural layout replaces it.

---

## 0. Intent

Provide the default, modern recognition path with **zero C/C++ dependencies** by
wiring the already-forked neural OCR stack behind the `Recognizer` trait. This is
both the POC recognizer (ships before the transcode tier) and the permanent
replacement for Tesseract's layout heuristics.

## 1. The forked stack (confirmed surfaces)

- **`ocrs`** (`ocrs/ocrs/src/`): `detection.rs` (text detection), `layout_analysis.rs`
  + `layout_analysis/` (reading order / columns), `recognition.rs` (line recognizer),
  `model.rs`, `preprocess.rs`, `text_items.rs`. Full detect→layout→recognize.
- **`rten`** (`rten-*` crates): `rten-convert` (ONNX→`.rten`), `rten-onnx`,
  `rten-model-file` (`.rten`), `rten-text` (CTC/text decode), `rten-simd` /
  `rten-gemm` / `rten-vecmath` (kernels), `rten-imageproc`. ocrs runs on rten.
- **`rten-ndarray-demo`**: reference for image-rs + ndarray + rten integration
  (`mobilenet.rten` present) — the wiring template.
- **`tract`** (pure-Rust general ONNX/TF) — arbitrary model escape hatch.
- **`ort`** (ONNX Runtime FFI) — GPU / exotic-op escape, **feature-gated, off by default**.

## 2. Backend wiring — D-OCR-30

`tesseract-rs/src/engine/ocrs_backend.rs` implements `Recognizer`:

```
preprocess (image/imageproc) ─► ocrs::detection ─► ocrs::layout_analysis (reading order)
   ─► per-line crops ─► ocrs::recognition (rten / rten-text CTC) ─► tokens + confidence + bbox
```

- Models: convert the ocrs detection + recognition ONNX to `.rten` via `rten-convert`
  once; vendor the `.rten` blobs (or fetch in CI). Confirm the converter + current
  model assets are present in the fork before relying on them (D-OCR-30 acceptance).
- Output normalized to the **same token struct** the transcode tier emits, so the
  emit stage (ocr-soa-integration) is engine-agnostic.

## 3. Layout: why neural, not ported — D-OCR-30 (rationale)

Tesseract's `textord/`+`ccstruct/` layout is intrusive linked-list + cyclic-mutable
blob-graph heuristics (`BLOBNBOX`, `ColPartition`, tab-stop finder). A syntax-directed
transcode (ruff/AST codegen rewrites *syntax*, not *ownership*) turns it into
`Rc<RefCell<>>` sludge or `unsafe`. `ocrs` already does detection+layout+reading-order
as a neural model — strictly better and pure-Rust. **Decision: layout is never
transcoded; it is `ocrs`.** Tesseract's recognizer (the LSTM) is the only thing worth
the bit-exact transcode (D-OCR-2x), and even then only as a compat/accuracy fallback.

## 4. tract + ort escape hatches — D-OCR-31

- `tract_backend.rs`: load an arbitrary ONNX OCR model (custom-trained, PaddleOCR
  export). Pure Rust, CPU. The "I have a specific recognizer" path.
- `ort_backend.rs`: GPU (CUDA/CoreML/TensorRT) or an op `tract` rejects. Feature
  `ort-gpu`, off by default; documented as the sole C++ dependency.

## 5. Deliverables

- **D-OCR-30:** `Ocrs` backend end-to-end on a scanned PDF page → tokens+bbox+conf;
  models present/converted; image→`.rten` path runs with no C deps.
- **D-OCR-31:** `Tract` backend loads + runs one custom ONNX recognizer; `Ort`
  backend compiles only under `--features ort-gpu`.

## 6. Open decisions

- **OD-30a:** front-end = hand-rolled `pdfium-render`+`image`+`imageproc`, or adopt
  `AdaWorldAPI/ferrules` (layout-aware Rust document parser) as the PDF front-end?
- **OD-30b:** keep a third "OCR-free" branch (`colpali`/`Qwen3-VL-Embedding`,
  document-image → retrieval directly) as a separate plan, or note-and-defer?
