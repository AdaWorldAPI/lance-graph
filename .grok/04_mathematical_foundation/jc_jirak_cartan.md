# JC (Jirak-Cartan) — Mathematical Verification Layer

**Location**: `crates/jc/`

**CI Integration**: `.github/workflows/jc-proof.yml` (runs on every push/PR touching `crates/jc/**`)

## Purpose

The `jc` crate provides **executable mathematical proofs** that the core substrate (binary-Hamming causal fields + VSA bundling + codebook quantization) has the required statistical and geometric properties.

It is not just testing — it is **continuous formal verification** of architectural invariants.

## Current Pillars (as of 2026-05)

| Pillar | Module                    | Status     | Description |
|--------|---------------------------|------------|-------------|
| E-SUBSTRATE-1 | `substrate.rs`           | Active     | Bundle associativity & Markov structure |
| Cartan-Kuranishi | `cartan.rs`           | Deferred   | Role keys = Cartan characters |
| φ-Weyl     | `weyl.rs`                 | Active     | Optimal collocation without aliasing |
| Preconditioner | `precond.rs`           | Active     | Fast prolongation convergence |
| Jirak Berry-Esseen | `jirak.rs`         | Active     | Weak dependence noise floor (critical for bundling) |
| **Pearl 2³** | `pearl.rs`               | **Active** | Mask classification accuracy (three-plane vs bundled) |
| Hadamard Concentration | `koestenberger.rs` | Active | Concentration on Hadamard spaces |
| Hilbert CLT | `dueker_zoubouloglou.rs` | Active | AR(1) in high dimension |
| EWA-Sandwich | `ewa_sandwich.rs`        | Active     | Multi-hop Σ push-forward |
| Pflug Lipschitz | `pflug.rs`            | Active     | Nested distance on DN-trees |
| Hambly-Lyons | `hambly_lyons.rs`        | Active     | Signature uniqueness on tree-quotients |

## Why This Matters

- Proves that weak dependence from codebook sharing + role-key overlap is **handled correctly** (Jirak).
- Explicitly validates **Pearl 2³** behavior in the bundled regime.
- Gives geometric and statistical guarantees needed for safe superposition and large projections.

## Connection to Architecture

This crate is the **mathematical spine** that allows us to confidently build higher-level structures (multi-mask superposition, L4 4096 projections, tensor operations) on top of the causal register.

---

*Living document. Update when new pillars are activated.*