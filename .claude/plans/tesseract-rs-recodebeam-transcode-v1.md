# tesseract-rs — recodebeam Decoder 1:1 Transcode v1

> **Type:** plan (sub-plan, v2). Deliverable D-OCR-21. Replaces v1 lstm-recodebeam (LSTM now hosted).
> **Status:** PLANTED 2026-06-15. Decoder transcoded 1:1 over HOSTED posteriors.
> **Canon:** consumes `[T,C]` from D-OCR-16 (embedanything host); oracle-gated.

---

## 0. Intent
Transcode only the DECODER. The LSTM forward is hosted (D-OCR-16); recodebeam takes
its `[T, n_classes]` posteriors and produces text exactly as Tesseract does.

## 1. recodebeam (`tesseract-rs/src/recodebeam.rs`) — hand-port, D-OCR-21
Beam over recoder codes; dawg-constrained + unconstrained beams; certainty/rating
tie-break 1:1. DAWG node-array walks = codegen (D-OCR-40); the beam↔dawg interaction
is hand-ported (behavioral subtlety). Output: text + per-token rating/certainty →
becomes per-token confidence at emit (integration plan).

## 2. int8 exactness boundary
Numeric exactness now lives at the HOST boundary (D-OCR-16 posteriors), not in
transcoded kernels. recodebeam consumes posteriors; if the host reproduces C++
posteriors to the int8 contract, the decoder's 1:1 transcription yields 1:1 text.

## 3. Acceptance
recodebeam text byte-identical to libtesseract on ≥10k crops given oracle posteriors
(isolates decoder correctness from host numeric parity).

## 4. Open
- **OD-21a:** support `lstm_choice_mode` top-k now (feeds OCR evidence tenant) or later.
