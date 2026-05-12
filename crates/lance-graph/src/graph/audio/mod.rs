//! Audio graph: AudioFrame ↔ HHTL cascade ↔ lance-graph nodes.
//!
//! Wires the ndarray audio primitives into the lance-graph execution model:
//!
//! ```text
//! PCM → AudioFrame(48B) ─┐
//!                         ├→ AudioNode (graph node, lance-versioned)
//! PhaseDescriptor(4B) ───┘       │
//!                                ├→ SpiralAddress (highheelbgz encoding)
//!                                ├→ RouteAction (HHTL cascade level)
//!                                └→ ComposeTable (multi-hop reasoning)
//! ```
//!
//! Each AudioNode = one 20ms frame stored as a lance-graph vertex.
//! Edges = temporal adjacency + spectral similarity.
//! Search = HHTL cascade over AudioNodes (content-based audio retrieval).

pub mod hhtl_bridge;
pub mod node;
