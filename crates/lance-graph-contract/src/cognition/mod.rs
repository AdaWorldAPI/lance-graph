//! Typed consumer pipeline grammar for normalized OGIT/OWL/DOLCE/Odoo
//! entities, per `.claude/plans/normalized-entity-holy-grail-v1.md`.
//!
//! ## The carrier
//!
//! [`NormalizedEntity<Stage>`] holds the 4-way inheritance chain as a
//! typed lens into a `MailboxSoA` row. `Stage` is phantom-typed;
//! advancement happens via the five-verb algebra and is
//! compile-time-enforced.
//!
//! ## The algebra (E-OP-FIVE-VERBS-1)
//!
//! - `resolve_ogit`   — `Raw` → `WithOgit`
//! - `hydrate_owl`    — `WithOgit` → `WithOwl`
//! - `classify_dolce` — `WithOwl` → `WithDolce`
//! - `align_fibu`     — `WithDolce` → `Normalized`
//! - `op` / `chk_data` / `review` / `abduct` / `report` / `output` — the
//!   `think` op-chain over the normalized carrier
//!
//! ## The Op trait (E-OP-THREE-CALLSITES-1)
//!
//! [`Op<I,O>`](op::Op) has three call sites — `apply` (cold),
//! `apply_stream` (warm, deferred to Stage 2), `apply_soa` (hot,
//! JIT-compiled, deferred to Stage 2). One trait, three speeds, one
//! set of const data.
//!
//! ## Cross-references
//!
//! - Plan: `.claude/plans/normalized-entity-holy-grail-v1.md`
//! - Epiphanies: E-NORMALIZED-ENTITY-1, E-OP-FIVE-VERBS-1,
//!   E-OP-THREE-CALLSITES-1, E-CONSUMER-CANNOT-INTERPRET-1
//! - Mailbox SoA: PR #427 (thoughtspace columns)
//! - Codebook: `super::callcenter::ogit_uris`
//!
//! ## Example consumer chain (woa-rs invoice flow)
//!
//! ```no_run
//! use lance_graph_contract::cognition::*;
//! use lance_graph_contract::cognition::entity::{OdooEntityRef, MailboxRow};
//! use lance_graph_contract::transaction::Interactive;
//!
//! # let ctx = Interactive::new();
//! # let invoice = NormalizedEntity::<Raw>::raw(
//! #     OdooEntityRef("account.move"),
//! #     MailboxRow { mailbox_ref: 0, row_idx: 0 },
//! # );
//! # /*
//! // Concrete Op types (from Stage 2 kernel implementations):
//! // KontenerkennungSkr04, SkrAccountInRange, FiscalPositionResolver,
//! // VatLiability, GoBdLockCheck, UStvaKennzahlAggregator
//!
//! let result = invoice
//!     .resolve_ogit(&ctx)       // Raw → WithOgit
//!     .hydrate_owl(&ctx)        // WithOgit → WithOwl
//!     .classify_dolce(&ctx)     // WithOwl → WithDolce
//!     .align_fibu(&ctx)         // WithDolce → Normalized
//!     .op(KontenerkennungSkr04)                       // Normalized → Normalized
//!     .chk_data(SkrAccountInRange::new(8400..=8499))  // → Checked
//!     .review(FiscalPositionResolver)                 // → Reviewed
//!     .abduct(VatLiability)                           // → Abducted
//!     .op(GoBdLockCheck)                              // → Abducted
//!     .report(UStvaKennzahlAggregator)                // → Reported
//!     .output();                                      // → Output, cascade fires
//! # */
//! ```
//!
//! Same chain shape inside `Bulk` (warm path) and `Periodisch` (hot
//! JIT path) — the context picks the call site per
//! E-OP-THREE-CALLSITES-1 + E-TRANSACTION-CONTEXT-1.

pub mod advance;
pub mod cascade;
pub mod entity;
pub mod op;
pub mod stages;

pub use entity::NormalizedEntity;
pub use op::{Op, OpError, OpKind, Output};
pub use stages::{
    Abducted, Checked, Normalized, Raw, Reported, Reviewed, Stage, WithDolce, WithOgit, WithOwl,
};
pub use cascade::{CascadeKind, CascadeWalker, TraversalMode};
