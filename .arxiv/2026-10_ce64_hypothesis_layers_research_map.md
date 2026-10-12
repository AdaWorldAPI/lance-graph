# CE64 hypothesis layers — research map

**Date:** 2026-10-12
**Status:** Index and open questions. No new paper was read for this note.
Every "read" row points at a distillation already in this folder; every
"unsearched" row is a search to run, not a finding.
**Serves:** `.claude/plans/ce64-hypothesis-layers-v1.md` (D-HL-0..9),
`.claude/knowledge/ce64-reasoning-substrate-plateau.md`.

## Punchline

> **The design asks four research questions, and this folder already answers parts of three: accumulate wide and round once (BF16), fuse independent evidence cumulatively and dependent evidence by averaging (subjective logic), and treat confidence as evidence stability (Wang). The fourth — persisted hypothesis layers as sparse deltas merged on read — has storage precedent in log-structured and copy-on-write systems, but no paper here covers it yet.**

## Q1 — Where does precision get lost, and how often?

- **Read:** `1905.12322_narrow_storage_wide_accumulate_round_once.md`. Narrow
  storage, wide accumulation, one rounding at the store.
- **Measured here since:** D-RPF-P5 E2 (in-place unit revision stalls at
  c = 244 after ~20 observations).
- **Test:** D-HL-2 (merge-on-read vs in-place).
- **Unsearched:** stochastic rounding is already parked by the 2026-10-08
  coresearch (determinism). Kahan/compensated summation for integer evidence
  is not needed if the fold is exact integer; confirm before searching.

## Q2 — When is evidence independent?

- **Read:** `0712.1182_fusion_needs_to_know_whether_evidence_is_independent.md`
  (cumulative vs averaging fusion); `2001_wang_confidence_is_evidence_stability_not_a_95pct_interval.md`;
  `2009_wang_two_numbers_preserve_direction_and_stability_of_evidence.md`;
  `1303.1517_revision_is_not_bayesian_conditioning.md`.
- **Measured here since:** D-RPF-P5 E3 (duplicates inflate certainty 2–3.5×).
- **Test:** D-HL-3 (the layer id as the stamp).
- **Unsearched:** NARS evidential-base overlap in OpenNARS/ONA source
  (how stamps are compared and truncated); provenance semirings for
  derivation lineage.

## Q3 — How to know the binary encoding is right?

- **Read:** `2604.18873_executable_narsese_needs_runtime_validation_not_label_guessing.md`
  (1000 cases, 200 held out, ONA protocol). A candidate corpus for D-HL-4
  beyond the NAL-1/2 fixtures, subject to its licence and format.
  `2026-08_nal1_to_nal9_is_a_capability_ladder_not_nine_cycles.md`
  (each level independently falsifiable).
- **Method:** ICC / Spearman / Cronbach via `jc::reliability`.
- **Unsearched:** agreement statistics for ordinal codes with ties at the
  saturation ends (weighted kappa vs ICC on u8 codes).

## Q4 — Persisted hypothesis layers: sparse delta, merge on read

- **Read in this workspace (not a paper):**
  `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` (measured in
  one consumer); `contract/src/alpha.rs` (ephemeral overlay, allocate ≠ claim).
- **Unsearched:** log-structured merge and merge-on-read delta stores;
  copy-on-write snapshot layering; Lance's own deletion vectors and row-id
  masks as a delta mechanism; MVCC visibility as the analogue of "which
  layers are visible to this read". Search these before designing
  D-HL-7's record, so the record reuses a known shape.

## Q5 — What is a validation ladder for?

- **Read:** `2018_self_in_nars_requires_internal_experience_and_mental_operations.md`;
  `2503.13467_metacognition_requires_replayable_self_models_not_just_logs.md`
  (retrieve trace → diagnose → correct → rerun → compare against a no-meta
  control).
- **Measured here since:** D-CE64-STAUNEN-0 (naive surprise is captured by
  noise; habituation and learning progress fix it); D-CE64-CYCLE-0 (what the
  register must carry).
- **Test:** D-HL-8 must include the no-meta control the review prescribes.

## Q6 — Correlation vs causation in bits 59..63

- **Read:** `1608.07879_actual_causality_needs_contingencies_not_just_deletion.md`;
  `1301.3898_counterfactual_causation_can_be_bounded_not_guessed.md`.
- **In code:** D-GSO-7a relational certification; D-PEARL-IO-0.
- **Unsearched:** none required for this plan.
