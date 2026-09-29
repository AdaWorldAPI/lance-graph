# 2026-09-29 — The shipped Cam96 is twelve independent axes, not six pairs

**Status:** VERIFIED-IN-CODE · OPEN — spec `.claude/plans/deepnsm-v2-cam96-pairwise-v1.md` (D-C96P-1..8)

## What the code does
- `Cam96Space` requires 12 axis codebooks (`space.rs:195-196`).
- `encode` quantizes each axis's own 8-d chunk against its own centroids (`:230-251`).
- `distance` sums 12 per-axis squared-L2 terms (`:257-272`).
- `rails()` pairs bytes for display only (`:287-289`); no distance reads it.
- The producer split "each 16-d subspace into a 256:256 pair", i.e. 12 × 8-d PQ (`probes/fidelity_48_vs_96.py:7`, `probes/train_codebook.py:57, 77`).
- Consequence: a rail's two bytes are two unrelated halves of a subspace. That is twelve indices, a finer point. It is not the `6×(u8:u8)` six relations that `E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1` and the facet plan (`deepnsm-morton-comma-facet-v1.md` §0) specify.

## Measured context (already on record)
At 96 bits, held-out KJV: RQ point 0.786, 12-axis 0.774, 48-bit point 0.617 (`probes/README.md` §4). The 12-axis code does not beat an equal-budget point.

## Also found
- `SemanticSpace` scores words at their frequency-rank `(basin, identity)` address (`space.rs:74-80`, `vocab.rs:14-21`). That is the facet plan's `PF=Payload` defection, and `SemanticSpace` is still exported.
- `recipe_substrate::PairPalette` is the same two-independent-axes shape (`recipe_substrate.rs:125-147`) → `ISS-PAIRPALETTE-IS-TWO-AXES-NOT-A-PAIR`.
- `episodic_basin::BasinRow.self_code` persists 12 Cam96 bytes with no marker of which code shape wrote them (`episodic_basin.rs:78-97`).

## Open
- The pair-code artifact needs the 96-d embeddings. They are not in the `v0.1.0-cam96-data` release.
- Where the frequency/PoS header lives, given the classid canon, is undecided.
