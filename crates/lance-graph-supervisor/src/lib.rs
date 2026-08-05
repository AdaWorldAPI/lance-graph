//! `lance-graph-supervisor` — ractor-supervised actor tree for callcenter fan-out.
//!
//! Implements `CallcenterSupervisor` (PR-G2, TD-RACTOR-SUPERVISOR-5):
//! one ractor actor per active G slot, one-for-one supervision, exponential
//! backoff (100ms→30s), bounded mailboxes (default 1024).
//!
//! # CC-2 compliance (byte-layout stability)
//!
//! `UnifiedAuditEvent` and `AuthOp` are **NOT modified**.
//! Actor lifecycle events use the **separate** `LifecycleAuditEvent` type
//! (own 18-byte canonical_bytes layout, own sink trait).
//! This preserves the 26-byte `UnifiedAuditEvent::canonical_bytes` layout
//! that `W2`'s verify CLI and `W12 A1` byte-layout assertions depend on.
//!
//! # CC-3 compliance (SuperDomain::System)
//!
//! `SuperDomain::System` (variant = 8) is added to `lance-graph-callcenter`'s
//! `super_domain.rs`. It is exempt from the §13.4 hard-lock partner matrix
//! (documented in the variant's doc comment). Lifecycle audit events routed
//! to `SuperDomain::System` do not pollute domain-partitioned chains.
//!
//! # Feature gates
//!
//! - `supervisor` — enables ractor + static_assertions; required for all actor code.
//! - `supervisor-lifecycle-audit` — enables `LifecycleAuditEvent` emission in
//!   the supervisor's spawn/respawn paths. Off by default (no audit noise in dev).
//!
//! # Build verification
//!
//! ```bash
//! cargo check -p lance-graph-supervisor
//! cargo test  -p lance-graph-supervisor
//! # regression — ensure unified_audit didn't change
//! cargo test  -p lance-graph-callcenter unified_audit::canonical_bytes
//! ```

// ─── Always-present modules (no feature gate needed) ─────────────────────────

pub mod consumer_msg;
pub mod error;
pub mod lifecycle_audit;

// ─── cycle-driver feature — the P4 loop-closure driver (persist_sink caller) ──

/// P4a/P4b: the cycle loop-closure driver. Drains the fleet's staged planner
/// casts into one sealed cycle (`persist_cycle` → one WAL write, one
/// `DatasetVersion`) and applies ONLY the sealed **sparse** transition set to the
/// fleet — represented owners advance one legal step, unrepresented owners stay
/// byte-identical. Behind the `cycle-driver` feature (pulls `lance-graph-planner`);
/// it does NOT need the ractor `supervisor` feature — it applies through
/// `MailboxSoaOwner::try_advance_phase` directly, no message bus.
#[cfg(feature = "cycle-driver")]
pub mod cycle_driver;

pub use consumer_msg::{
    CalibrateRequest, CalibrateResponse, ConsumerEnvelope, ConsumerReply, CrystalResponse,
    DispatchRequest, HealthStatus, IngestAck, IngestRequest, ProbeRequest, ProbeResponse,
    Qualia17DResponse, QualiaRequest, StyleList, StylesRequest, TensorsRequest, TensorsResponse,
};
pub use error::SupervisorErr;
pub use lifecycle_audit::{
    LifecycleAuditEvent, LifecycleAuditSink, LifecycleEventType, NoopLifecycleSink,
};

// ─── supervisor feature — ractor actor tree ───────────────────────────────────

#[cfg(feature = "supervisor")]
pub mod supervisor;

#[cfg(feature = "supervisor")]
pub mod actors;

/// Message-free kanban visibility + pure helpers. The actor/ack/tick surface
/// that used to live here was DELETED 2026-08-05
/// (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`): phase progression is the
/// #879 sealed-cycle path (`cycle_driver`); observation is a `&self` census,
/// never an RPC. See the module tombstone.
#[cfg(feature = "supervisor")]
pub mod kanban_actor;

#[cfg(feature = "supervisor")]
pub use kanban_actor::{mul_target, parse_kanban_step, PhaseCensus};

#[cfg(feature = "supervisor")]
pub use supervisor::{
    CallcenterSupervisor, ChildSummary, ConsumerSlot, ModuleEntry, StubConsumerActor,
    SupervisorHealthSummary, SupervisorMsg, SupervisorState, DEFAULT_MAILBOX_CAPACITY,
};

#[cfg(feature = "supervisor")]
pub use actors::MedcareConsumerActor;
