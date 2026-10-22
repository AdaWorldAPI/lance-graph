# Palette256 loud-signal / MP3 analogy: measurable salience, not semantic proof

Status: **test-only / no bake change**. Immutable codebook remains reference; salience is a cold candidate selection policy, not a learned mutation of indices or LUT values.

Experiment: top-k magnitude keeps almost all squared energy on a heavy-head fixture. **Negative control:** low-energy rare coordinate is the *only* class discriminator. Thus preserved spectral energy != preserved reasoning truth.

Acceptance for any future corpus bake: declare signal family, norm, salience threshold, protected rare-event strata, held-out top-k recall, calibrated error/abstention and versioned codebook digest. Do not claim psychoacoustic masking is directly transferable to cognition.

Commands: `cargo test --manifest-path crates/bgz-tensor/Cargo.toml --test palette256_salience_probe`.
