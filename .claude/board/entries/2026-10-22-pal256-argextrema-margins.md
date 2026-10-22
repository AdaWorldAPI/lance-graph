# Typed argmax/argmin on pairwise LUT values

Status: **test-only**. Already available: `bgz-tensor::fisher_z::FamilyGamma`, `FisherZTable`, existing `ndarray::hpc::statistics` argmax and argmin and `ndarray::bitwise` top-k.

The new hypothesis is an **uncertainty-aware admission**: if lower bound of the best candidate does not exceed every competitor's upper bound, return `Undecided`; only skip dequantization when the palette order and error calibration are genuinely witnessed.

Rule guards: same reference version and family gamma, no signed-byte / unsigned-index confusion (XOR 0x80 preserves signed order), no cross-family rank comparison, no guarantee when encoding ties or intervals overlap.

Falsifier: two candidates 0.500 +/- 0.02 vs 0.505 +/- 0.02 must not produce a certified winner. Do not conflate cosine score with NARS frequency.

Run: `cargo test --manifest-path crates/bgz-tensor/Cargo.toml --test palette256_argextrema_probe`.
