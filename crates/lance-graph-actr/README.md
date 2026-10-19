# lance-graph-actr

Experimental ACT-R-inspired **production arbitration kernel**, not a complete ACT-R cognitive architecture.

The cognitive condition matcher remains in relalg/R2IL/Quack/Mask-RISC. This crate receives already-evaluated candidate eligibility and selects a witnessed **intent**; it cannot execute external effects.

## Boundary

- Activation = memory accessibility.
- Utility = estimated procedural value.
- NAL evidence = epistemic support, deliberately outside this arbitration score.
- Motivation = future optional priority signal, not implicit authorization.
- Snapshot identity and witness version are explicit; currently local experimental types, to be replaced by canonical contract ABI.
- The 64-thought frontier is a caller policy, not an ACT-R buffer count.

## Run

`cargo test --manifest-path crates/lance-graph-actr/Cargo.toml`

Standalone `[workspace]` avoids changing the main workspace's dependency graph.

## Next

1. Differential match eligibility against relalg/Mask-RISC.
2. Add typed goal/retrieval buffers, cycle timing, utility updates and outcome traces.
3. Wire canonical address and witness types without parallel ABI definitions.
4. Use CerebroCortex ACT-R activation fixtures as numerical reference.
5. Validate conflict resolution, stale witnesses and visibility **before** effect dispatch.

No claim of full ACT-R fidelity, performance improvement, or runtime integration.
