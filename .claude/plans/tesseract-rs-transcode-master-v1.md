# Tesseract → tesseract-rs — 1:1 Transcode Master Plan v2

> **Type:** plan family root. SUPERSEDES v1 (which wrongly skipped layout).
> **Status:** PLANTED 2026-06-15 v2 — design locked. 1:1 behavioral transcode of ALL
>   of Tesseract; the LSTM forward is the ONLY swapped component.
> **Front:** post-#496. Hosts: `embedanything` DTO (GGUF→candle→ndarray-AMX, per
>   `.grok/NDARRAY_BGZ_EMBEDANYTHING_INTEGRATION.md`); `bgz_tensor` weight store.
> **Canon:** OGAR/CLAUDE.md GUID P0; lance-graph/CLAUDE.md SoA node; canonical_node.rs.

---

## 0. The whole decision in one line

Transcode **every** Tesseract module 1:1 for behavioral parity (validated against
the `tesseract-rs` FFI oracle), EXCEPT the LSTM recognizer forward pass, which is
**hosted** on the existing runbook: recognizer weights → GGUF → `embedanything`
DTO (candle backend) → `ndarray` AMX path → per-timestep posteriors. Everything
else — container, unicharset, recoder, **textord/ccstruct layout**, recodebeam,
DAWG dict, the minimal Leptonica ops Tesseract calls — is faithfully ported.

## 1. What is 1:1 transcoded (≈200k LOC, mechanically)

| Tesseract area | Route | 1:1 fidelity rule |
|---|---|---|
| `ccutil/tessdatamanager`, `unicharset`, `unicharcompress` (recoder) | AST-DLL codegen | byte-faithful tables |
| `dict/{dawg,trie,permdawg}` | AST-DLL codegen | node-array walks 1:1 |
| **`textord/` + `ccstruct/` (layout, tab-stops, ColPartition, reading order)** | AST-DLL codegen → **faithful raw-pointer/unsafe Rust** | ELIST/CLIST + cyclic blob graphs transcribed as raw-pointer intrusive lists; behavior-identical; safe-refactor is a LATER behavior-preserving pass, never a 1:1 deviation |
| `recodebeam` (beam + DAWG interaction) | hand-port | tie-break/normalization 1:1 |
| Leptonica ops Tesseract actually calls (Otsu/Sauvola, deskew, despeckle, CC-label, scale) | hand-port onto `image`/`imageproc` | numeric parity per-op |
| `lstm/{network,lstm,fullyconnected,convolve,weightmatrix}` | **NOT transcoded — HOSTED** | see §2 |

**The unsafe-is-fine ruling:** a true 1:1 image of intrusive-pointer C++ is
raw-pointer Rust. We accept `unsafe` as the faithful transcription; correctness is
proven by the oracle diff, not by safe-Rust aesthetics. Refactor to arena/index
graphs is a separate, oracle-gated step AFTER 1:1 is green. This is what makes the
200k LOC mechanical (codegen + faithful transcription) instead of an ownership redesign.

## 2. The ONE swap — LSTM hosted, not ported

```
.traineddata LSTM weights ──► GGUF export ──► embedanything DTO (candle) ──► ndarray AMX
   ──► per-timestep posteriors ──► (transcoded) recodebeam + DAWG ──► text + confidence
```

- The `.traineddata` loader (D-OCR-10) extracts the recognizer net + weights and
  **exports GGUF**, not raw ndarray hydration — so it enters the existing runbook
  unchanged. `bgz_tensor` stores/streams the weight tensors (its §2.3 use-case).
- `embedanything::infer()` runs the CNN+BiLSTM. recodebeam (transcoded) decodes the
  posteriors. Only the matmul/gate compute is delegated; the decode stays 1:1.
- Reuses every existing optimization (candle GGUF loader, ndarray AMX, bgz storage)
  — zero new inference stack.

## 3. The one DTO extension OCR imposes

`embedanything::infer` today shapes outputs as `EmbeddingVector`/`HypothesisScore`.
The recognizer needs **per-timestep posteriors** (`[T, n_classes]`) for CTC/recodebeam.
→ **D-OCR-15:** add a sequence-output variant to the DTO (`infer_sequence → [T,C]`),
the only change the OCR use forces on the shared interface. Narrow, additive.

## 4. Deliverable index (D-OCR-NN)

| ID | Deliverable | Depends |
|---|---|---|
| D-OCR-10 | `.traineddata` parse → unicharset/recoder/DAWG + recognizer-net → **GGUF export** | D-OCR-40 |
| D-OCR-15 | `embedanything` sequence-output (`infer_sequence → [T,C]`) | — |
| D-OCR-16 | recognizer GGUF runs via embedanything(candle)→ndarray; posteriors match C++ | D-OCR-10,15 |
| D-OCR-21 | recodebeam + DAWG transcode (decode over hosted posteriors) | D-OCR-16 |
| D-OCR-30 | **textord/ccstruct layout 1:1** (raw-pointer faithful) | D-OCR-40 |
| D-OCR-31 | minimal Leptonica ops on image/imageproc (numeric parity) | — |
| D-OCR-40 | AST-DLL clang→IR→Rust codegen harness (ruff emission) | — |
| D-OCR-42 | oracle diff-gate (every module vs libtesseract FFI) | D-OCR-21,30 |
| D-OCR-50 | OCR token → canonical NodeRow (OGAR class, HHTL, ValueSchema, edges) | canon |
| D-OCR-52 | DeepNSM + char-confusion + CAM/PQ token repair | D-OCR-50 |
| D-OCR-53 | bit-reproducibility harness (crop→text→NodeRow golden diff) | D-OCR-21,30,**50,51** |

Critical path: **40 → {10,30} → 16 → 21 → 42 → 53**. D-OCR-15 parallel (tiny). The
layout transcode (30) and the recognizer host (16) are independent until decode.

## 5. Success criteria

1. `tesseract-rs` reproduces libtesseract output **byte-identical** on ≥10k line crops AND full-page layouts (oracle gate) — layout included, since it's 1:1.
2. LSTM runs ONLY via embedanything(candle)→ndarray; no transcoded LSTM kernels.
3. Zero crates.io (forks + ndarray/lance family); `ort` sole opt-in C++ (off by default).
4. Recognized tokens land as canonical NodeRows with no bespoke geometry.

## 6. Sub-plans (re-cast for v2)
- `tesseract-rs-traineddata-gguf-v1` (D-OCR-10/16) — loader → GGUF → embedanything host.
- `tesseract-rs-layout-transcode-v1` (D-OCR-30/31) — textord/ccstruct 1:1 + Leptonica ops.
- `tesseract-rs-recodebeam-transcode-v1` (D-OCR-21) — decoder over hosted posteriors.
- `tesseract-rs-ast-dll-codegen-v1` (D-OCR-40/42) — the codegen harness (covers layout).
- `ocr-canonical-soa-integration-v1` (D-OCR-50/52/53) — OCR token = NodeRow + repair.
