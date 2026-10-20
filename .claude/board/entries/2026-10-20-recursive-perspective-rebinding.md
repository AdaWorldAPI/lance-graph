# Recursive perspective rebinding: distillation (2026-10-20)

**Status:** PROPOSED, lab only (D-RPR-0). Plan: `.claude/plans/recursive-perspective-rebinding-v1.md`. Documentation only.

## Findings (read on origin/main)

- `WORLD × ANGLE × THOUGHT` needs no new nouns. WORLD is a sealed version with thought 0 as givens (D-TGA:6–9). ANGLE is `Focus` + `CsrPred`/tile in the `rmap.rs` probe, with `MapPalette` proposed (D-RMAP-1A). THOUGHT is `ThoughtDelta` over one shared tile (32 B per thought, per-eval flat across 1/8/64; D-RMAP-1A §8.5).
- `TableA[i] : TableB[i]` is the measured H14c/H14d expression `Thought(t)[T_a[i] : T_b[i]]`; the MeTTa council v3 header names those plans as the correct path and marks the preimage-join design a gather violation.
- `rebind` (k slots, one forward seek each) is measured: it wins for small k, loses at k = 256, and on the dense relation from k ≥ 16 (D-RMAP-1A §8.4). Recursive questioning is cheap only if real chains keep k small; that distribution is unmeasured.
- The probe already emits a missing-dependency signal: `ScopedReach.escaped` plus the `IncompleteFocus` refusal. Hypothesis H1: escapes seed the next angle.
- "Teleport" in this repo includes D-TELEPORT-0/1/2, source-restricted folds through Lance 13's row-address prefilter, which is the prefetch mechanism here.

## Open

A1 (pre-fold binding admission), H1, the k distribution, n-ary co-addressing and repeated variables, selective retraction provenance (D-TGA-3), a refusal for recursive dependency cycles, `ThoughtCommitId` never-reused rule.
