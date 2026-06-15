# tesseract-rs — Layout (textord/ccstruct) 1:1 Transcode v1

> **Type:** plan (sub-plan, v2 — the part v1 wrongly skipped). Deliverables D-OCR-30/31.
> **Status:** PLANTED 2026-06-15. FAITHFUL 1:1, raw-pointer where C++ is intrusive.
> **Canon:** oracle-gated behavioral parity; safe-refactor is a later, separate, oracle-preserving pass.

---

## 0. Intent
Reproduce Tesseract's page layout (the ~tens-of-thousands LOC the v1 plan wrongly
proposed replacing with ocrs) byte-for-byte. This is the bulk of the "free 200k LOC":
mechanical, codegen-amenable in structure, made tractable by accepting raw-pointer Rust.

## 1. Faithful-transcription ruling
Tesseract layout is intrusive `ELIST`/`CLIST` doubly-linked lists + cyclic mutable
blob graphs (`BLOBNBOX`, `TO_BLOCK`, `ColPartition`, tab-stop finder, baseline fit,
reading order). The 1:1 image is **raw-pointer Rust** (`*mut`, intrusive nodes,
manual lifetimes) — behavior-identical, NOT redesigned. Ownership redesign (arena/
slotmap/index graphs) is a LATER pass, gated to preserve oracle output. Do NOT
redesign during transcode; that breaks 1:1 and is the trap.

## 2. Modules (D-OCR-30)
`ccstruct/{blobbox,coutline,polyblk,...}`, `textord/{tabfind,colfind,colpartition,
tablefind,baselinefit,textlineprojection,wordseg,...}`. AST-DLL codegen emits the
struct/method skeletons + intrusive-list ops as raw-pointer Rust; the gnarly
control flow is reviewed against the C++ AST, diff-gated per function.

## 3. Leptonica ops (D-OCR-31)
Only the ~dozen ops Tesseract calls: Otsu/Sauvola binarize, projection/Radon
deskew, despeckle, connected-component label, scale. Hand-port onto `image`/
`imageproc` with per-op numeric parity vs the Leptonica fork (built as oracle).
The rest of Leptonica is NOT ported.

## 4. Acceptance (D-OCR-30/31)
Full-page layout output (block/line/word boxes + reading order) byte-identical to
libtesseract on a fixed page set; per-op image results bit-equal on fixtures.

## 5. Open
- **OD-30a:** one intrusive-list helper crate (`tess_elist`) shared across modules, or inline per-module? (Shared reduces unsafe surface.)
