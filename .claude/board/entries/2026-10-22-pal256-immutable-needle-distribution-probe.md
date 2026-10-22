# Palette256 is a frozen needle; Palette256:Palette256 is a distribution

Status: **lab-only test contract**. No codebook rebake, CE64 layout change, new engine, or production certification.

## Existing ownership

- `.claude/knowledge/reference-frame-vs-motion.md`: corpus/calibration is versioned and immutable within a reference snapshot; attention, belief updates, and learning are overlays.
- `.claude/board/entries/2026-08-26-e-palette256-is-a-needle-the-colon-is-the-distribution-1.md`: a single palette index answers *which*; pairwise calibrated LUT answers *how related*.
- `bgz-tensor::fisher_z::FisherZTable`: already implements frozen `256 x 256 x i8 + FamilyGamma`. Do **not** add another distance table.

## Falsifiers and next boundary

1. Replaying 64 lookup rounds may never mutate the LUT or its calibration bytes.
2. A `u8` identity cannot be ranked as semantic similarity by sorting the IDs.
3. The pairwise diagonal is not a substitute for address equality; if a caller needs self-identity, answer from the addressing layer.
4. Rebuilding calibrated references requires a **new version and digest**. Future work: attest codebook identity + family-gamma bytes at BIND time and reject stale codebooks.
5. The documented Spearman 0.9998 is *pair-ranking agreement on sampled pairs*, not a probability of correctness and not a multi-hop theorem.

Run: `cargo test --manifest-path crates/bgz-tensor/Cargo.toml --test palette256_reference_contract`.
