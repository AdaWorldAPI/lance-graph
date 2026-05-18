//! # cognitive-shader-actor
//!
//! **Glue #4** from the lance-graph ↔ surrealdb ↔ sea-orm ↔ ndarray integration plan (§6).
//!
//! Wraps cognitive shaders (today: in-process invocations in `cognitive-shader-driver`,
//! `thinking-engine`, `causal-edge`, `deepnsm`, `holograph`) as first-class
//! [`ractor::Actor`] instances sitting inside a supervision tree.
//!
//! ## Why this crate exists (plan §6)
//!
//! Today the cognitive crates run in-process. For an AGI-style architecture with
//! let-it-crash recovery and back-pressure they need to live in ractor's supervision
//! tree. This crate is the adapter layer:
//!
//! - **Additive only** — the cognitive crates themselves are unchanged. Existing
//!   in-process call sites keep working.
//! - Consumers that want supervised behaviour depend on this crate; consumers that
//!   want direct invocation do not.
//!
//! ## Supervisor topology (plan §6)
//!
//! ```text
//! ShaderSupervisor (one-for-one, max 5 restarts in 60 s)
//! ├── ThinkingEngineActor
//! ├── CausalEdgeActor
//! ├── DeepNSMActor
//! ├── HolographActor
//! └── CognitiveShaderDriverActor  (orchestrates the others)
//! ```
//!
//! A misbehaving shader is restarted by its supervisor without affecting peers.
//!
//! ## Sprint status
//!
//! This crate is scaffolded in Sprint 1 (fan-out, worker LG-2). Real ractor wiring
//! lands in Sprint 2 (plan §7). All method bodies below use `unimplemented!("LG-2 stub — Sprint 2")`.
//!
//! ## Module layout
//!
//! | Module | Contents |
//! |---|---|
//! | [`messages`] | `ShaderMessage<P>` — the mailbox vocabulary |
//! | [`actor`] | `CognitiveShaderActor<S>` — the `ractor::Actor` adapter |
//! | [`supervisor`] | `ShaderSupervisor` — restart policy + child spawning |

/// Message vocabulary for the shader actor mailbox.
///
/// See plan §6 — `ShaderMessage` variants match the plan's API sketch exactly.
pub mod messages;

/// The `ractor::Actor` adapter for any [`SupervisableShader`] implementor.
///
/// See plan §6 — `CognitiveShaderActor<S>`.
///
/// [`SupervisableShader`]: lance_graph_contract::actor::SupervisableShader
pub mod actor;

/// Supervisor that manages shader actor children with restart back-off.
///
/// See plan §6 supervisor topology diagram.
pub mod supervisor;
