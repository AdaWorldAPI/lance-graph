# CausalEdge64 — The Atomic Causal Register

**Location**: `crates/causal-edge/src/edge.rs`

## Core Idea

`CausalEdge64` is a single `u64` that serves as the **universal causal + epistemic register** for the entire system.

It is the atomic unit that flows through the L1–4 resonance loop.

## Bit Layout (High-Level)

- S/P/O palette indices (8-bit each)
- NARS frequency + confidence (8-bit each)
- `CausalMask` (Pearl 2³ — 3 bits)
- Direction triad
- `InferenceType` (NARS)
- `PlasticityState` (3 bits)
- Temporal index

## Key Capabilities

| Method       | Purpose                                      | Speed     |
|--------------|----------------------------------------------|-----------|
| `pack()`     | Construct from components                    | O(1)      |
| `forward()`  | NARS-style inference (depends on InferenceType) | Very fast |
| `learn()`    | Evidence-driven revision + plasticity update | Fast      |
| `matches_causal()` | Query predicate for Pearl level         | O(1)      |

## Related Modules

- `pearl.rs` — `CausalMask` enum (Association / Intervention / Counterfactual)
- `plasticity.rs` — Per-plane hot/cold/frozen control

## Open Research Questions

- How to best represent 8-mask superposition over `CausalEdge64` in L1/L2?
- Can `forward()` be lifted into a tensor operation while keeping register speed?

---

*This file is a living note. Expand as needed.*