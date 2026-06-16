# OCR Transcode — Gating Probes v1

> **Type:** plan (probe queue for the tesseract-rs OCR transcode family).
> **Status:** PLANTED 2026-06-16 — from the 5-specialist framing of #497 (cascade /
>   family-codec / palette / dto-soa / truth-architect) against the post-#498 substrate.
> **Why:** the #497 plan family makes several load-bearing claims that are **asserted,
>   not measured**. Per the workspace insight-update cycle (CLAUDE.md: Claim → Probe →
>   Run → FINDING/correct), these probes gate the expensive transcode work. Run the
>   cheap ones (< 3 h, existing crates) BEFORE funding the ~200k-LOC layout transcode.
> **Cross-ref:** `tesseract-rs-transcode-master-v1.md`, `ocr-canonical-soa-integration-v1.md`,
>   `soa-centroid-attention-field-synthesis-v1.md`.

---

## The four primary gating probes

### OCR-RT — residue → codebook-rank round-trip (settles the "reversible without a hash" claim)

- **Claim under test:** "an OCR token is reversible through residue + codebook, no
  hash/string column" (the migration's headline rationale).
- **Current evidence (FINDING):** there is **no `residue → rank` inverse** in code.
  `deepnsm/vocabulary.rs` maps `rank → &str` via a stored table; every
  `nearest_words(rank,k)` / `word_neighbors(word,k)` entry point takes a *known*
  rank/word as input. Helix `encode` is lossy quantization; `from_bytes` recovers the
  6 bytes, never the source `n`. So the round-trip does not exist today.
- **Probe:** given a word → its codebook rank → encode to (helix `Signed360` 6 B ⊕
  turbovec PQ residue), attempt to recover the **rank** from the residue bytes ALONE
  (no stored-rank lookup). Needs deepnsm `Codebook` + helix `Signed360` wired in one
  crate (they are not today — that wiring is itself part of the gate).
- **Pass:** **100 %** of the 4096-word vocab round-trips residue→rank→word exactly —
  a reversibility gate must be exact; a single miss fails it (a lossy map is NOT
  "reversible"). Any tolerance belongs in a separate *quality* probe, never this gate.
- **Fail:** any miss, OR recovery requires the original rank as input ⇒ "reversible
  without a hash" is FALSE; the corrected plans already say text = identity →
  content-store lookup, codebook = repair signal (this probe confirms or lifts that).
- **Cost:** ~80 LOC once deepnsm+helix are co-located; the wiring is the real work.

### OCR-DET — repair-stage determinism (settles the bit-reproducible golden diff)

- **Claim under test:** D-OCR-53 "crop → NodeRow byte golden diff is bit-reproducible."
- **Current evidence:** DeepNSM repair/similarity is **f32** (`encoder.rs` similarity
  → f32; `pipeline.rs` weighted blend). VSA bind/bundle is integer, but repair is on
  the critical path before the NodeRow is written.
- **Probe:** run the DeepNSM repair path (`nearest_words` + similarity) twice / across
  a scalar-vs-SIMD toggle on 1k garbage fixtures; compare repaired-token bytes.
- **Pass:** byte-identical both runs ⇒ repair may stay inside the frozen-mode boundary.
- **Fail:** any divergence ⇒ the f32 repair stage MUST be carved out of (or pinned in)
  the frozen-mode bit-repro boundary, and the helix `floor_version` MUST be pinned in
  the golden bytes (else the rolling floor rolls and the diff spuriously fails).
- **Cost:** ~60 LOC, < 1 h, pure deepnsm (compilable today as a `deepnsm` example).

### OCR-POST — GGUF posterior parity (gates the entire LSTM host swap)

- **Claim under test:** "int8-exact LSTM posteriors" across `.traineddata` → GGUF →
  candle → ndarray-AMX, so the transcoded recodebeam yields 1:1 text.
- **Probe:** dump libtesseract per-timestep `[T,C]` posteriors for ONE crop; run the
  same recognizer GGUF via embedanything(candle)→ndarray; compare.
- **Pass:** max per-timestep |Δposterior| within the int8 quantization step on the crop.
- **Fail:** candle cannot express Tesseract's BiLSTM/CTC (OD-10b), OR Δ exceeds the
  int8 step ⇒ the "1:1 text" chain is unfounded; the swap needs a candle-fork op
  before any decode work proceeds.
- **Cost:** needs candle wiring + a GGUF model + libtesseract oracle — NOT runnable in
  this checkout. A 1-crop spike, not a 1k-crop acceptance, until it passes once.

### OCR-SCHEMA — ValueSchema fit (settles the §0 anti-invention question)

- **Claim under test:** "OCR needs a dedicated `ValueSchema::Ocr`."
- **Current evidence (FINDING):** shipped `ocr.rs` rides the POC-`Full` default and
  writes per-tenant; `Compressed` already = {Fingerprint, HelixResidue, TurbovecResidue,
  EntityType}. A 5th enum variant collides with the #496 §0 guardrail.
- **Probe (pure design / test, ~30 min):** assert against `canonical_node.rs` that the
  OCR tenant set (minus the deferred-to-content-store fields) is ⊆ an existing preset's
  `field_mask()`; if it fits `Full`/`Compressed`, no new variant is needed.
- **Pass:** a tenant-by-tenant table fitting an existing preset ⇒ ride it (no enum change).
- **Fail:** a genuine gap ⇒ escalate to operator to **mint an OCR class** (ClassView
  selecting existing tenants), NOT extend the `ValueSchema` enum.
- **Cost:** compilable today as a `lance-graph-contract` test (`ocr_schema_fit`).

---

## Secondary probes (cascade performance — convert asserted numbers to facts)

These back the perf claims ("95 % pairs skipped", "8K at Super-8 cost", early-exit):

- **P-OCR-EARLYEXIT:** CAM-PQ stroke-cascade skip-rate on a real OCR-token codebook
  (`cam_pq.rs` `cascade_query` with calibrated `heel_threshold`/`branch_threshold`).
- **P-OCR-CAKES-RECALL:** `clam.rs` `knn_dfs_sieve` recall vs brute force on the token codebook.
- **P-HELIX-OCR-FIDELITY:** reuse the already-owed #459 ≥ 0.9980 Pearson floor gate for
  the residue round-trip fidelity (still NOT RUN — see #459 deferred).

---

## DAG honesty

The master critical path ends `… → {50,51} → 53` (golden diff). D-OCR-53 silently
assumes OCR-RT + OCR-DET + OCR-POST all hold; none is measured. **OCR-DET and
OCR-SCHEMA cost < 2 h with existing crates** — run them first; record results here
(CONJECTURE → FINDING). If **OCR-RT** fails, the "use the new architecture's
reversibility" rationale collapses regardless of transcode fidelity, so it is the
single highest-leverage probe before the ~200k-LOC layout transcode is funded.
