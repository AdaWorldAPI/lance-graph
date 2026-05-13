//! lance-graph-callcenter — External callcenter membrane.
//!
//! Implements the `ExternalMembrane` trait from `lance-graph-contract`
//! with Lance + DataFusion as the storage and query layer.
//!
//! # Architecture (four layers, § 2 of the design plan)
//!
//! ```text
//! A — Canonical internal substrate (untouched, in lance-graph-contract)
//!     Vsa10k · BindSpace SoA · CognitiveShader · CollapseGate · AriGraph
//!
//! B — ExternalMembrane trait (in lance-graph-contract, zero-dep)
//!     project() · ingest() · subscribe()
//!
//! C — Dual ledger (Lance datasets, [persist] feature)
//!     cognitive_event · steering_intent · memory · actor / session
//!
//! D — This crate (LanceMembrane + server, feature-gated)
//!     LanceMembrane · CommitFilter→Expr · PhoenixServer · DrainTask
//!     JwtMiddleware · PostgRestHandler
//! ```
//!
//! # Feature Gates
//!
//! - `default = []`   — contract re-exports only, zero external deps
//! - `[persist]`      — Arrow RecordBatch + Lance dataset ops
//! - `[query]`        — DataFusion Expr translator (CommitFilter → Expr)
//! - `[realtime]`     — tokio watch, WebSocket, Phoenix channel shapes
//! - `[serve]`        — axum WS server (implies realtime + query)
//! - `[auth]`         — JWT verify + actor context
//! - `[full]`         — all of the above
//!
//! # Open Unknowns (resolve before DM-2+)
//!
//! - UNKNOWN-1: Does `cognitive-shader-driver`'s `ShaderSink` trait overlap
//!   with `ExternalMembrane`? Inspect `crates/cognitive-shader-driver/src/`
//!   before wiring DM-2.
//! - UNKNOWN-2: Which consumers (n8n-rs / crewai-rust / openclaw) need
//!   Phoenix wire protocol vs direct Rust API?
//! - UNKNOWN-3: Does n8n-rs need a pgwire connection?
//! - UNKNOWN-4: Right `actor_id` type — u64 hash or proper identity type?
//! - UNKNOWN-5: Lance dataset path / `LANCE_URI` env var convention.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md`

pub use lance_graph_contract::external_membrane::{CommitFilter, ExternalMembrane};
pub use lance_graph_contract::ontology::{Label, Locale};

// ── External ontology DTO surface (the "Foundry outside" layer) ──────────────
pub mod ontology_dto;
pub use ontology_dto::{medcare_ontology, smb_ontology, OntologyDto};

// ── Phase A: BBB spine (DM-2) ────────────────────────────────────────────────
// UNKNOWN-1 resolved: ShaderSink is internal BindSpace ingestion; no overlap
// with ExternalMembrane. See lance_membrane.rs module doc for details.
pub mod dn_path;
pub mod external_intent;
pub mod lance_membrane;

pub use dn_path::DnPath;
pub use external_intent::{CognitiveEventRow, ExternalIntent};
pub use lance_membrane::LanceMembrane;

// DU-3 — RoleDB DataFusion VSA UDFs ([query] feature)
#[cfg(feature = "query")]
pub mod vsa_udfs;

#[cfg(feature = "query")]
pub use vsa_udfs::register_vsa_udfs;

// DM-3 — CommitFilter → DataFusion Expr translator ([query] feature)
#[cfg(feature = "query")]
pub mod filter_expr;

// DM-4 — LanceVersionWatcher: tail version counter → Phoenix events ([realtime])
#[cfg(feature = "realtime")]
pub mod version_watcher;

// DM-5 — PhoenixServer: minimal WS server, Phoenix channel subset ([realtime])
// #[cfg(feature = "realtime")]
// pub mod phoenix;

// DM-6 — DrainTask: steering_intent → UnifiedStep → OrchestrationBridge
pub mod drain;

// DM-7 — JwtMiddleware + ActorContext → LogicalPlan RLS rewriter ([auth])
//         UNKNOWN-3 resolved: DataFusion LogicalPlan layer (NOT pgwire).
//         UNKNOWN-4 resolved: actor_id: String (JWT sub claim flows through unchanged).
// DM-7 JWT extraction: ActorContext from JWT token (auth-jwt — no datafusion dep)
#[cfg(any(feature = "auth-jwt", feature = "auth", feature = "full"))]
pub mod auth;

// DM-7 RLS rewriter: DataFusion OptimizerRule injecting tenant/actor predicates.
// Gated on query-lite (activated by both auth-rls and auth-rls-lite).
#[cfg(any(
    feature = "auth-rls-lite",
    feature = "auth-rls",
    feature = "auth",
    feature = "full"
))]
pub mod rls;

// DM-8 — PostgRestHandler: query-string → DataFusion SQL → Lance → Arrow ([serve])
//         Confirm PostgREST compat is needed before building (§ 8 stop point 4).
//         A5: dependency-free stub gated behind `postgrest` feature.
#[cfg(feature = "postgrest")]
pub mod postgrest;

// LF-90 — append-only audit log for every RLS-rewritten query.
//         A3: in-memory ring buffer skeleton; Lance-backed writer arrives later.
#[cfg(feature = "audit-log")]
pub mod audit;

// PR #278 outlook E1 — generalized PolicyRewriter trait (column masking,
// row encryption, differential privacy stubs) sharing the OptimizerRule slot
// with the existing RLS rewriter. Gated on auth-rls-lite (where the
// DataFusion types live).
#[cfg(any(
    feature = "auth-rls-lite",
    feature = "auth-rls",
    feature = "auth",
    feature = "full"
))]
pub mod policy;

// Outer ↔ inner ontology transcode — reusable Foundry primitives.
// Domain-agnostic mapper between the wire-shape DTO surface (already in
// `ontology_dto`) and the inner SoA / SPO substrate. Also hosts the
// **single deliberate transition bandaid**: `parallelbetrieb` for the
// MySQL ↔ DataFusion ↔ SPO reconciler. See `transcode/mod.rs`.
pub mod transcode;
