//! OGAR-driven tenant port bridges.
//!
//! This is the OGAR side of the OGIT/OGAR separation (operator,
//! 2026-06-20). These bridges couple to `ogar_vocab::ports` /
//! `ogar_vocab::class_ids` (the OGAR codebook + `PortSpec` class schema),
//! so they live here in `lance-graph-ogar`, NOT in `lance-graph-ontology`
//! (which is OGIT and must not depend on `ogar-vocab`). The OGIT-side
//! legacy bridges (`SpearBridge` / `SharePointBridge` / `OgitBridge` +
//! the legacy `WoaBridge` pass-through) stay in
//! `lance_graph_ontology::bridges`.
//!
//! # Two layers
//!
//! - The **generic harness** [`unified::UnifiedBridge<P: PortSpec>`] is the
//!   one-and-only `lance_graph_ontology::NamespaceBridge` impl for
//!   OGAR-driven ports. It inherits everything that varies between ports
//!   (`NAMESPACE` / `BRIDGE_ID` / public-name → class_id aliases) from
//!   `ogar_vocab::ports::PortSpec`. Adding a port is `impl PortSpec for
//!   FooPort {…}` in OGAR — no bridge boilerplate here.
//! - The **per-port aliases** ([`OpenProjectBridge`], [`RedmineBridge`],
//!   [`MedcareBridge`], [`WoaBridge`], [`SmbBridge`], [`OdooBridge`]) are
//!   thin `type` aliases over the harness.
//!
//! # OGAR-driven ports (`UnifiedBridge<P>` aliases)
//!
//! - [`OpenProjectBridge`]: `UnifiedBridge<ogar_vocab::ports::OpenProjectPort>`
//!   — locks to the `OpenProject` namespace. `WorkPackage` / `TimeEntry`
//!   / `Project` etc. resolve to OGAR canonical class_ids via the
//!   port's alias table.
//! - [`RedmineBridge`]: `UnifiedBridge<ogar_vocab::ports::RedminePort>` —
//!   locks to the `Redmine` namespace. `Issue` / `TimeEntry` / `Project`
//!   etc. resolve to the SAME OGAR canonical class_ids as the
//!   OpenProject equivalents.
//! - [`MedcareBridge`]: `UnifiedBridge<ogar_vocab::ports::HealthcarePort>`
//!   — locks to the `Healthcare` namespace. `Patient` / `Diagnosis` /
//!   `LabValue` / `Medication` / `Treatment` / `Visit` / `VitalSign`
//!   resolve to the `0x09XX` Health codebook (Northstar T9).
//! - [`WoaBridge`]: `UnifiedBridge<ogar_vocab::ports::WoaPort>` — locks
//!   to the `WorkOrder` namespace. `Customer` / `Vorgang` / `Position` /
//!   `Stundenzettel` / `TimesheetActivity` / `TimeEntry` etc. resolve to
//!   the canonical commerce `0x02XX` block + `BILLABLE_WORK_ENTRY`
//!   (0x0103). The planner-side `TimeEntry` (OpenProject/Redmine) and
//!   the WoA `Stundenzettel` collapse to the same id — the operator's
//!   *"planner times align with billable hours"* statement realised as
//!   data.
//! - [`SmbBridge`]: `UnifiedBridge<ogar_vocab::ports::SmbPort>` — locks
//!   to the `SMB` namespace. `Kunde` / `Auftrag` / `Rechnung` /
//!   `Stundenzettel` etc. resolve to the SAME commerce block + same
//!   `BILLABLE_WORK_ENTRY`, giving smb-office-rs cross-fork convergence
//!   with WoA + OpenProject + Odoo.
//! - [`OdooBridge`]: `UnifiedBridge<ogar_vocab::ports::OdooPort>` —
//!   locks to the `Odoo` namespace. Odoo's Python class names
//!   (`res.partner` / `account.move` / `account.move.line` /
//!   `hr.attendance` / …) resolve to the same canonical block.
//!   Closes the planner→ERP→billing convergence chain at every hop.

pub mod unified;

mod medcare_bridge;
mod odoo_bridge;
mod openproject_bridge;
mod redmine_bridge;
mod smb_bridge;
mod woa_bridge;

pub use medcare_bridge::{HealthcarePort, MedcareBridge};
pub use odoo_bridge::{OdooBridge, OdooPort};
pub use openproject_bridge::{OpenProjectBridge, OpenProjectPort};
pub use redmine_bridge::{RedmineBridge, RedminePort};
pub use smb_bridge::{SmbBridge, SmbPort};
pub use unified::UnifiedBridge;
pub use woa_bridge::{WoaBridge, WoaPort};

// Compatibility shims for the pre-migration constants. `bridges`
// previously re-exported `OPENPROJECT_CODEBOOK` / `REDMINE_CODEBOOK`
// directly; both now live in `ogar_vocab::ports::*_ALIASES` (the
// canonical layer is the single source of truth). The re-exports here
// are `#[deprecated]` in the per-port modules and forward to the OGAR
// constants — existing consumers keep compiling (codex P2 on PR #570).
#[allow(deprecated)]
pub use openproject_bridge::OPENPROJECT_CODEBOOK;
#[allow(deprecated)]
pub use redmine_bridge::REDMINE_CODEBOOK;
