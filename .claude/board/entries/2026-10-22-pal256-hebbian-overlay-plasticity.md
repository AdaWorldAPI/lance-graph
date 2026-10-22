# Hebbian / anti-Hebbian plasticity as a separate overlay, not codebook retraining

Status: **lab-only**. The Google-search formula `Δz = η pre post (1 − tanh(|z|))` is a local update proposal. It is neither canonical NARS revision nor a Bayesian posterior.

Existing owner: `.claude/knowledge/reference-frame-vs-motion.md`: immutable calibrated reference versus dynamic motion/alpha/thought. Keep the Fisher-Z pairwise LUT unchanged and version the update overlay independently.

Probe claims ONLY:
- bounded finite latent and bounded tanh readout on tested activations;
- independent source gating prevents immediate evidence echo;
- negative coactivation can decrease the plastic overlay;
- codebook address never changes.

Refusals/next experiments: dependent sources sharing ancestry, adversarial wrong associations, negative saturation recovery, catastrophic reinforcement, forgetting/decay, convergence, calibration vs a held-out event oracle. The toy 64-bit witness cannot replace full provenance handling.

Run: `cargo test --manifest-path crates/causal-edge/Cargo.toml --test palette256_hebbian_overlay_probe`.
