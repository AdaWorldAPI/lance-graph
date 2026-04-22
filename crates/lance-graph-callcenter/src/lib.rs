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

// DM-2 — LanceMembrane: ExternalMembrane impl + compile-time BBB leak test
//         Resolve UNKNOWN-1 before uncommenting.
// mod lance_membrane;
// pub use lance_membrane::LanceMembrane;

// DM-3 — CommitFilter → DataFusion Expr translator ([query] feature)
// #[cfg(feature = "query")]
// pub mod filter;

// DM-4 — LanceVersionWatcher: tail version counter → Phoenix events ([realtime])
// #[cfg(feature = "realtime")]
// pub mod version_watcher;

// DM-5 — PhoenixServer: minimal WS server, Phoenix channel subset ([realtime])
// #[cfg(feature = "realtime")]
// pub mod phoenix;

// DM-6 — DrainTask: steering_intent → UnifiedStep → OrchestrationBridge
// pub mod drain;

// DM-7 — JwtMiddleware + ActorContext → LogicalPlan RLS rewriter ([auth])
//         Resolve UNKNOWN-3 (pgwire?) and UNKNOWN-4 (actor_id type) first.
// #[cfg(feature = "auth")]
// pub mod auth;

// DM-8 — PostgRestHandler: query-string → DataFusion SQL → Lance → Arrow ([serve])
//         Confirm PostgREST compat is needed before building (§ 8 stop point 4).
// #[cfg(feature = "serve")]
// pub mod postgrest;
