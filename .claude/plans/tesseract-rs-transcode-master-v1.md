# Tesseract → tesseract-rs Transcode — Master Plan v1

> **Type:** plan family root (forward marker / co-architecture). Plants the
>   sub-plans; owns the deliverable index, the dependency DAG, and the
>   skip-list rationale.
> **Status:** PLANTED 2026-06-15 — design only, no code. Layout/contracts proposed
>   against the post-#496 front.
> **Front:** post-#496. `canonical_node.rs` carries `NodeGuid` / `EdgeBlock` /
>   `EdgeCodecFlavor` / `NodeRow` / `ValueTenant` / `ValueSchema` / `NodeRowPacket`;
>   `class_view.rs` carries `ClassView` (`edge_codec_flavor`, `value_schema`) +
>   `FieldMask`. These are the integration surface, not a thing to re-derive.
> **Canon anchors (all sub-plans must match, never restate):**
>   - Operator GUID canon — `OGAR/CLAUDE.md` P0 (`classid·HEEL·HIP·TWIG·family·identity`, RFC-waived).
>   - Doc-lock — `lance-graph/CLAUDE.md` (commit `4ea6ac9`): SoA node, zero-fallback ladder, reserve-don't-reclaim.
>   - Code form — `crates/lance-graph-contract/src/canonical_node.rs` (#489+#490+#492+#494+#496).
>   - Three-tier model — `docs/architecture/soa-three-tier-model.md` (zero-copy, no emission).
>   - Supersession map — `.claude/plans/soa-migration-diff-resolution-2026-06-13.md`.
> **Skip-by-rule:** anything behind the migration front is residue, not authority.
>   This plan does NOT conform to deprecated `BindSpace` row geometry.

---

## 0. Intent (one paragraph)

Stand up `tesseract-rs` as a **pure-Rust, bit-reproducible OCR substrate** that (a)
runs *existing* modern Tesseract `.traineddata` LSTM models with byte-identical
output, (b) replaces Tesseract's accreted C++ layout heuristics with the neural
`ocrs`/`rten` path already forked into the account, (c) uses a clang-AST → Rust
**codegen harness** (built on the `ruff` AST/codegen crates) for the mechanical
leaf modules instead of hand-porting C++, and (d) emits every recognized token as
a **canonical SoA `NodeRow`** so OCR output lands directly in the lance-graph
cognitive substrate (OGAR class, HHTL address, `ValueSchema` over `ValueTenant`s,
`EdgeCodecFlavor` adjacency, DeepNSM/CAM-PQ correction) with no boundary tax.

The whole effort doubles as the **bit-reproducibility regression harness** the SoA
migration needs: Tesseract C++ (via the `tesseract-rs` FFI fork as oracle) vs the
Rust port, diffed to the byte on fixed line crops.

## 1. The spine (target data flow)

```
PDF / image
  └─► [front-end]      pdfium-render + image + imageproc  ── OR ──  ferrules (layout-aware)
        └─► [segment]  ocrs detection + layout_analysis   (NEURAL — replaces textord)   ┐
        └─► [recognize] EITHER  tesseract-rs traineddata→LSTM-on-ndarray→recodebeam      │  engine
                        OR      ocrs recognition (rten / rten-text CTC)                   │  trait
                        OR      tract (arbitrary ONNX recognizer)                         ┘
              └─► tokens + per-token confidence + bbox + top-k candidates
                    └─► [repair] DeepNSM (vocabulary·codebook·parser·encoder·similarity)
                          + CAM/PQ nearest-valid-token (helix / TurbovecResidue / CAKES)
                          └─► [emit] canonical NodeRow  (classid=OCR class, HHTL address,
                                ValueSchema OCR preset, EdgeCodecFlavor adjacency)
                                └─► NodeRowPacket → SoaEnvelope → Lance (kv-lance)
```

## 2. Three engine paths behind ONE trait (the central decision)

`tesseract-rs` exposes a single `Recognizer` trait with three backends; the engine
is a runtime/feature choice, never a fork of the pipeline:

| Backend | Source | Use when | C/C++ deps |
|---|---|---|---|
| `TranscodeLstm` | this plan family (traineddata→ndarray→recodebeam) | need an *existing* `.traineddata` language model, bit-identical to Tesseract | **none** |
| `Ocrs` | `AdaWorldAPI/ocrs` + `rten` | default modern path; detection+layout+recognition, pure Rust | **none** |
| `Tract` | `AdaWorldAPI/tract` | arbitrary/custom ONNX recognizer (PaddleOCR export, custom-trained) | **none** |
| `Ort` (escape) | `AdaWorldAPI/ort` | GPU needed, or an op `tract` rejects | ONNX Runtime (C++) |

**Default = `Ocrs`.** `TranscodeLstm` is the accuracy/compat fallback AND the
oracle's reference. `Ort` is opt-in only (it's the lone C++ dependency).

## 3. What we SKIP and why (do not transcode)

| Tesseract component | Disposition | Reason |
|---|---|---|
| Legacy pattern matcher (`classify/`, integer matcher, adaptive classifier) | **DROP** | Deprecated in Tesseract 4/5; LSTM supersedes it. |
| Leptonica (C, ~250k LOC) | **REPLACE, not port** | Reimplement only the ~dozen ops Tesseract calls (Otsu/Sauvola, deskew, despeckle, CC-label, scale) on `image`/`imageproc`. `AdaWorldAPI/leptonica` fork kept only as oracle build dep. |
| `textord/` + `ccstruct/` layout analysis (tab-stops, columns, `ColPartition`, baselines, reading order) | **REPLACE with `ocrs`** | Tens of thousands of LOC of intrusive linked-list / cyclic-mutable-graph heuristics. `ruff`-style syntax codegen cannot redesign ownership; `ocrs` neural layout is better anyway. See `tesseract-rs-neural-layout-ocrs-v1`. |
| CMake / `BOOL_VAR`/`INT_VAR` global param registry / renderer plumbing | **DROP** | Cargo + a config struct + the `ClassView` registry replace it. |
| C# (NuGet `Tesseract`, app wrappers) | **N/A** | No C# in OCR core; C# only in app wrappers — out of scope. |

## 4. What we transcode faithfully (bit-exact tier)

Routed through the AST-DLL codegen harness where mechanical, hand-ported where
numeric exactness is subtle. See sub-plans for module-by-module assignment.

- `.traineddata` container (`ccutil/tessdatamanager`), `unicharset`, recoder
  (`lstm/unicharcompress`) → **D-OCR-1x** (`...-traineddata-ndarray-v1`).
- LSTM forward (`lstm/{network,lstm,fullyconnected,convolve,weightmatrix}`) +
  `recodebeam` decoder + DAWG dict (`dict/{dawg,trie,permdawg}`) → **D-OCR-2x**
  (`...-lstm-recodebeam-v1`).

## 5. Deliverable index (D-OCR-NN) and DAG

| ID | Deliverable | Sub-plan | Depends on |
|---|---|---|---|
| D-OCR-00 | This master + skip-list + engine trait | (this) | — |
| D-OCR-10 | `.traineddata`/unicharset/recoder reader | traineddata-ndarray | D-OCR-40 |
| D-OCR-11 | LSTM weight hydration onto `ndarray::hpc` | traineddata-ndarray | D-OCR-10 |
| D-OCR-20 | LSTM forward pass on ndarray | lstm-recodebeam | D-OCR-11 |
| D-OCR-21 | `recodebeam` decoder + DAWG dict | lstm-recodebeam | D-OCR-20 |
| D-OCR-22 | int8-SIMD numeric-exactness conformance | lstm-recodebeam | D-OCR-20 |
| D-OCR-30 | `ocrs`/`rten` neural detect+layout+recognize backend | neural-layout-ocrs | — |
| D-OCR-31 | `tract` arbitrary-ONNX backend + `ort` escape | neural-layout-ocrs | D-OCR-30 |
| D-OCR-40 | clang C++ AST → IR ("AST DLL") | ast-dll-codegen | — |
| D-OCR-41 | IR → Rust emission via `ruff` codegen crates | ast-dll-codegen | D-OCR-40 |
| D-OCR-42 | diff-gate: emitted Rust vs FFI oracle | ast-dll-codegen | D-OCR-41 |
| D-OCR-50 | OCR class + HHTL address scheme | ocr-canonical-soa-integration | canon |
| D-OCR-51 | OCR `ValueSchema` preset over existing tenants | ocr-canonical-soa-integration | D-OCR-50 |
| D-OCR-52 | DeepNSM + CAM/PQ token repair wiring | ocr-canonical-soa-integration | D-OCR-51 |
| D-OCR-53 | bit-reproducibility harness (oracle diff) | ocr-canonical-soa-integration | D-OCR-21, D-OCR-30 |

DAG critical path: **D-OCR-40 → 10 → 11 → 20 → 21 → 53** (the transcode oracle),
with **D-OCR-30** (ocrs default) parallel and independent — ships first as the POC
recognizer while the transcode tier matures behind it.

## 6. Success criteria

1. `Ocrs` backend produces tokens → canonical `NodeRow`s persisted via `NodeRowPacket` (POC gate).
2. `TranscodeLstm` reproduces Tesseract C++ output **byte-identical** on ≥ 10k fixed line crops (oracle gate).
3. Zero new crates.io deps (forks/path/`ndarray`/`lance`-family only); `Ort` is the sole opt-in C++ dep, feature-gated off by default.
4. No layout-heuristic C++ transcoded (skip-list honored).
5. OCR nodes carry no bespoke row geometry — they ride `ValueSchema`/`ValueTenant`/`EdgeCodecFlavor` unchanged.

## 7. Open co-architecture decisions (resolve with operator)

- **OD-1:** Does an OCR token get a *dedicated* `ValueTenant` (bbox + per-char
  confidence + top-k recodebeam candidates), or ride `Meta`+`Fingerprint`+
  `TurbovecResidue`? Adding a tenant is canon-significant (tenants never move/reuse).
  POC rides existing; dedicated tenant deferred. (See ocr-soa-integration §OD-1.)
- **OD-2:** Is `ocrs`/`rten` the permanent default with `TranscodeLstm` as compat
  fallback, or is the transcode tier the eventual primary? Affects how much codegen
  effort D-OCR-4x warrants.
- **OD-3:** Does the AST-DLL frontend use libclang directly, or a clang→JSON dump
  consumed by a Rust IR? (See ast-dll-codegen §OD-3.)
