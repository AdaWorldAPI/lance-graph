# Architecture notes

This directory holds cross-substrate *architecture* documents. They are navigation and falsification maps, **not** alternate sources of truth for the byte-level ISA, a new engine specification, or authorization for implementation.

## Start here

- [Cartesian Epistemic Self-Model](cartesian-epistemic-self-model.md) — How deterministic thought-first addressing, two orientations of a 64-thought presence matrix, declared CE64 epistemic readings, alpha-channel saccades, demand-scoped relational closure, and provenance could support an independently testable software self-model. Includes current implementation status, measured bounds, known conflicts, scaling risks, and proposed falsification gates.
- [Neurofunctional Rubikon / Moore Synaptic Plasticity](neurofunctional-rubicon-moore-plasticity.md) — Maps ada-rs GoalGenerator/WillEngine, merged qualia-profile #1571, experimental Hyperon/OpenPsi, Heckhausen + Axon Rubikon, Moore tenant/NARS16×8 local learning, and thought-Git into one falsifiable three-timescale control loop. Explicit CE64/W migrations, source-root independence and independent lesion/learning tests.


## Source precedence

When an architecture statement conflicts with current code or a measured source, prefer the canonical code + its current approved decision record. In particular:

1. [`../knowledge/epistemic-frontier-capstone.md`](../knowledge/epistemic-frontier-capstone.md) — field and layer map with status / open points.
2. [`../plans/ce64-hypothesis-layers-v1.md`](../plans/ce64-hypothesis-layers-v1.md) — thought-lane and dual occupancy experiments.
3. [`../plans/d-holo-h14c-thought-first-v1.md`](../plans/d-holo-h14c-thought-first-v1.md) — actual point-read, bitmap, memory and Lance I/O measurements.
4. [`../../crates/lance-graph-contract/src/epistemic_state5.rs`](../../crates/lance-graph-contract/src/epistemic_state5.rs) and [`../../crates/causal-edge/src/layout.rs`](../../crates/causal-edge/src/layout.rs) — declared epistemic reading and canonical physical CE64 field map.
5. [`../knowledge/software-awareness-provenance-parity.md`](../knowledge/software-awareness-provenance-parity.md) — self-model evidence/oracle boundary and withdrawal of the first SAG runtime build.

**Writing rule:** Every significant claim must distinguish `CURRENT-CONTRACT`, `MEASURED`, `DECISION`, `PROPOSED`, and `OPEN`. A beautiful diagram is not evidence of shipped behavior. Do not create new opcodes, widen CE64, or conceal migration conflicts in a documentation PR.

