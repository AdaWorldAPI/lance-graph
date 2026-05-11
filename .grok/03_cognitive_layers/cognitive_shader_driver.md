# cognitive-shader-driver Crate

**Location**: `crates/cognitive-shader-driver/`

**Role**: Likely the **SoA (Structure of Arrays) wiring layer** and thought-cycle bus — the "cognitive shader driver" that orchestrates data movement between resonance, collapse, L4, and CausalEdge64.

## Key Modules (High Signal)

| File                        | Purpose                                      | Entropy | Notes |
|-----------------------------|----------------------------------------------|---------|-------|
| `driver.rs`                 | Main driver logic                            | High    | Core orchestration |
| `wire.rs`                   | Wiring / data movement                       | High    | SoA bus implementation? |
| `bindspace.rs`              | Binding space (VSA-style?)                   | Medium  | Likely related to XOR binding in diagrams |
| `rotation_kernel.rs`        | Rotation / transformation kernels            | Medium  | Geometric operations on representations |
| `sigma_rosetta.rs`          | Sigma / codebook translation                 | Medium  | Codebook bridging |
| `auto_style.rs` / `auto_detect.rs` | Automatic style / mode detection      | Medium  | Possibly feeds MetaOrchestrator |
| `engine_bridge.rs`          | Bridge to thinking engine / resonance        | Medium  | Connection point to holograph |

## Current Understanding

This crate appears to implement the **practical runtime layer** that makes the L1–4 loop and SoA wiring concrete. It contains codec, kernel, wire, and bridge components — suggesting it is the "plumbing" between the mathematical primitives (`CausalEdge64`, resonance) and higher cognition.

## Connection to Diagrams

User explicitly stated:  
> "The cognitive-shader-driver is The SoA"

This crate is therefore the implementation of the central bus shown in the Resonance-Based Cognitive System diagrams.

## Open Questions

- How does it consume/produce `CausalEdge64`?
- Relationship to `MetaOrchestrator` style selection?
- Does it implement the "Promotion Membrane" logic?

---

*High-signal area. Needs deeper targeted exploration.*

**See canonical NARS + Thinking inventory & migration plan**: `.grok/03_cognitive_layers/NARS_THINKING_IMPLEMENTATIONS_INVENTORY_MIGRATION.md` — this is now the single source for how NARS ops, thinking styles, and MUL fit into the shader-driver SoA surface, BindSpace zero-copy, and NarsOp cam integration.