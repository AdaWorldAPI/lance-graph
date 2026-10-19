# lance-graph-act-r

Experimental ACT-R cognitive **production arbitration kernel**, not a complete ACT-R cognitive architecture.

ACT-R is *Adaptive Control of Thought, Rational*. This crate is not an actor, mailbox or `ractor` component.

The cognitive condition matcher remains in relalg/R2IL/Quack/Mask-RISC. This crate receives already-evaluated candidate eligibility and selects a witnessed **intent**; it cannot execute external effects.

## Boundary

- Activation = memory accessibility.
- Utility = estimated procedural value.
- NAL evidence = epistemic support, deliberately outside this arbitration score.
- Motivation = future optional priority signal, not implicit authorization.
- Every `Candidate` carries `evaluated_at`, the snapshot its eligibility was computed against; `arbitrate` refuses the whole batch (`Refusal::StaleCandidate`) if any candidate was evaluated at another snapshot.
- Snapshot identity and witness version are explicit; currently local experimental types, to be replaced by canonical contract ABI.
- The 64-thought frontier is a caller policy, not an ACT-R buffer count.

## Run

`cargo test --manifest-path crates/lance-graph-act-r/Cargo.toml`

Standalone `[workspace]` avoids changing the main workspace's dependency graph; the crate is listed in the root `exclude` and has its own CI job (`.github/workflows/act-r.yml`).

```rust
use lance_graph_act_r::{Candidate, Decision, Snapshot};
```

## Next

1. Differential match eligibility against relalg/Mask-RISC.
2. Add typed goal/retrieval buffers, cycle timing, utility updates and outcome traces.
3. Wire canonical address and witness types without parallel ABI definitions.
4. Use CerebroCortex ACT-R activation fixtures as numerical reference.
5. Validate conflict resolution, stale witnesses and visibility **before** effect dispatch.

No claim of full ACT-R fidelity, performance improvement, or runtime integration.
