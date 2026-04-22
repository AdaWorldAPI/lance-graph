//! External membrane boundary — the Blood-Brain Barrier (BBB) contract.
//!
//! `ExternalMembrane` is the typed boundary between the canonical cognitive
//! substrate and the external callcenter surface. It lives in the zero-dep
//! contract crate so consumers can declare they implement it without pulling
//! in Arrow, Lance, axum, or any other heavy dependency.
//!
//! **BBB Invariant (enforced at compile time by the Arrow type system):**
//!
//! `Self::Commit` MUST NOT contain Vsa10k, RoleKey, SemiringChoice, NarsTruth,
//! HammingMin, or any other VSA or semiring type. Those types do not implement
//! Arrow's `Array` trait, so they physically cannot appear in a `RecordBatch`
//! column. The compiler rejects the violation — no runtime check needed.
//!
//! Only Arrow-scalar-compatible primitives cross the barrier:
//! `u8`, `u16`, `u32`, `u64`, `f32`, `bool`, bytes, strings, timestamps.
//!
//! **READ BY:** every session touching the callcenter crate, realtime
//! subscriptions, external API surface, or any BBB-crossing work.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md`

use crate::cognitive_shader::{ShaderBus, MetaWord};
use crate::orchestration::UnifiedStep;

// ═══════════════════════════════════════════════════════════════════════════
// EXTERNAL ROLE TAXONOMY
// ═══════════════════════════════════════════════════════════════════════════

/// Who sent or received an external event.
///
/// Used as the role-bind key when XOR-braiding external events into the
/// Markov trajectory. Same ±5 braiding mechanism as the grammar parser —
/// `RoleKey::bind(payload, role as u16)` at the callcenter impl site.
///
/// Inbound roles (consumer → substrate): 0–5.
/// Outbound roles (substrate → consumer): 6–7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExternalRole {
    // ── Inbound (consumer identity) ──
    User        = 0,
    Consumer    = 1,
    N8n         = 2,
    OpenClaw    = 3,
    CrewaiUser  = 4,
    CrewaiAgent = 5,
    // ── Outbound (substrate identity as seen externally) ──
    Rag         = 6,  // cognitive shader acting as knowledge retriever
    Agent       = 7,  // specific cognitive agent that handled this step
}

/// Whether an external crossing is a seed, passive context, or outbound commit.
///
/// - `Seed`: triggers a blackboard reasoning cycle (DrainTask routes to OrchestrationBridge).
/// - `Context`: passive — role-bound and XOR'd into the trajectory bundle without
///   starting a new cycle. The Markov ±5 window absorbs it into the next natural cycle.
/// - `Commit`: projected scalar row leaving the substrate toward an external subscriber.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalEventKind {
    Seed,
    Context,
    Commit,
}

// ═══════════════════════════════════════════════════════════════════════════
// COMMIT FILTER
// ═══════════════════════════════════════════════════════════════════════════

/// Scalar-only predicate for filtering projected commits.
///
/// All fields are Arrow-scalar-compatible (`u64`, `u8`, `bool`).
/// No VSA types. No semiring types.
#[derive(Clone, Debug, Default)]
pub struct CommitFilter {
    /// Filter to a specific actor (plain u64 hash for v1; see UNKNOWN-4).
    pub actor_id: Option<u64>,
    /// Upper bound on free_energy (packed u8 from `MetaWord::free_e()`).
    pub max_free_energy: Option<u8>,
    /// Filter to a specific thinking style ordinal.
    pub style_ordinal: Option<u8>,
    /// When `Some(true)`, only committed cycles (not failure cycles).
    pub is_commit: Option<bool>,
}

/// The typed boundary between the canonical cognitive substrate and
/// the external callcenter surface.
///
/// Implementations live in `lance-graph-callcenter` (the `LanceMembrane`
/// struct). This trait is here, in the zero-dep contract crate, so that
/// consumers can depend on the boundary shape without pulling in Arrow,
/// Lance, DataFusion, axum, or any other heavy crate.
///
/// # Associated Types
///
/// - `Commit` — Arrow scalar row in the `LanceMembrane` impl.
///   Must NOT contain any VSA or semiring type (see BBB invariant above).
/// - `Intent` — External intent shape entering through the callcenter.
///   Translated into a `UnifiedStep` for `OrchestrationBridge::route()`.
/// - `Subscription` — Handle returned by `subscribe()`; wired to a
///   `tokio::sync::watch` on the Lance version counter in the impl.
pub trait ExternalMembrane: Send + Sync {
    /// Projected scalar representation of one committed `ShaderBus` cycle.
    type Commit: Send;

    /// External intent shape — the inbound side of the BBB.
    type Intent: Send;

    /// Subscription handle returned by `subscribe()`.
    type Subscription: Send;

    /// Project a committed cognitive cycle to a scalar commit record.
    ///
    /// Strips all VSA fields. Produces Arrow scalars only.
    /// Called on every `CollapseGate` fire with `EmitMode::Persist`.
    fn project(&self, bus: &ShaderBus, meta: MetaWord) -> Self::Commit;

    /// Translate external intent to canonical dispatch.
    ///
    /// The external consumer's intent enters here and exits as a
    /// `UnifiedStep` ready for `OrchestrationBridge::route()`.
    fn ingest(&self, intent: Self::Intent) -> UnifiedStep;

    /// Subscribe to projected commits matching the filter.
    ///
    /// The `LanceMembrane` implementation wires this to a
    /// `tokio::sync::watch` on the Lance version counter.
    fn subscribe(&self, filter: CommitFilter) -> Self::Subscription;
}
