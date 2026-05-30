//! # lance-graph-contract — The Single Source of Truth
//!
//! Zero-dependency trait crate that defines the contract between:
//! - **lance-graph-planner** (implements these traits)
//! - **ladybug-rs** (calls Planner + CamPq + OrchestrationBridge)
//! - **crewai-rust** (calls ThinkingStyleContract + MulContract)
//! - **n8n-rs** (calls JitContract + OrchestrationBridge)
//!
//! # Why This Exists
//!
//! Before this crate, each consumer duplicated thinking style enums,
//! field modulation structs, and query plan types. Now:
//!
//! ```text
//! crewai-rust ──┐
//!               │
//! n8n-rs ───────┤── depend on ──► lance-graph-contract (traits only)
//!               │
//! ladybug-rs ───┘
//!
//! lance-graph-planner ──► implements lance-graph-contract traits
//! ```
//!
//! # Module Layout
//!
//! - [`thinking`] — 36 thinking styles, 6 clusters, τ addresses, 23D vectors
//! - [`mul`] — MUL assessment (Dunning-Kruger, trust, flow, compass)
//! - [`plan`] — Query planning traits (PlanStrategy, PlanResult)
//! - [`cam`] — CAM-PQ distance contract (6-byte fingerprint ops)
//! - [`jit`] — JIT compilation contract (jitson template → kernel)
//! - [`orchestration`] — Bridge trait for single-binary routing
//! - [`nars`] — NARS inference types shared across all consumers
//! - [`collapse_gate`] — Per-row write airgap (`GateDecision`, `MergeMode`)
//! - [`cycle_accumulator`] — Per-cadence flush gate; absorbs the L1↔L3
//!   speed ratio. Distinct from `collapse_gate` per topology I-4.

pub mod cognition;
pub mod transaction;

pub mod a2a_blackboard;
pub mod atoms;
pub mod auth;
pub mod callcenter;
pub mod cam;
pub mod codegen_spine;
pub mod cognitive_shader;
pub mod collapse_gate;
pub mod container;
pub mod crystal;
pub mod cycle_accumulator;
pub mod distance;
pub mod escalation;
pub mod exploration;
pub mod external_membrane;
pub mod faculty;
pub mod grammar;
pub mod graph_render;
pub mod hash;
pub mod high_heel;
pub mod jit;
pub mod kanban;
pub mod literal_graph;
pub mod mail;
pub mod manifest;
pub mod mul;
pub mod nars;
pub mod ocr;
pub mod ontology;
pub mod orchestration;
pub mod orchestration_mode;
pub mod persona;
pub mod plan;
pub mod property;
pub mod proprioception;
pub mod qualia;
pub use qualia::{
    axis_index, axis_label, qualia_to_state, QualiaI4_16D, QualiaVector, AXIS_LABELS, MIDPOINT,
    QUALIA_DIMS, QUALIA_I4_DIMS, QUALIA_I4_LABELS, ZERO,
};
pub mod reasoning;
pub mod recipe_kernels;
pub mod recipes;
pub mod repository;
pub mod savants;
pub mod scenario;
pub mod sensorium;
pub mod sigma_propagation;
pub mod sla;
pub mod soa_view;
pub mod splat;
pub mod tax;
pub mod thinking;
pub mod vsa;
pub mod witness_table;
pub mod world_map;
pub mod world_model;

// Re-exports for the most commonly used collapse_gate types.
pub use collapse_gate::{CollapseGateEmission, GateDecision, MailboxId, MergeMode};
pub use kanban::{ExecTarget, KanbanColumn, KanbanMove, RubiconTransitionError};
pub use soa_view::{MailboxSoaOwner, MailboxSoaView};
