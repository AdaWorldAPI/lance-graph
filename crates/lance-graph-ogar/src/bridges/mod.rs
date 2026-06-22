//! OGAR-driven tenant port bridges.
//!
//! This is the OGAR side of the OGIT/OGAR separation (operator,
//! 2026-06-20). These bridges couple to `ogar_vocab::ports` /
//! `ogar_vocab::class_ids` (the OGAR codebook + `PortSpec` class schema),
//! so they live here in `lance-graph-ogar`, NOT in `lance-graph-ontology`
//! (which is OGIT and must not depend on `ogar-vocab`). The OGIT-side
//! legacy bridges (`WoaBridge` / `SpearBridge` / `SharePointBridge` /
//! `OgitBridge`) stay in `lance_graph_ontology::bridges`.
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
//!   [`MedcareBridge`]) are thin `type` aliases over the harness.
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
//!   OpenProject equivalents, so cross-fork convergence is the default
//!   not the exception.
//! - [`MedcareBridge`]: `UnifiedBridge<ogar_vocab::ports::HealthcarePort>`
//!   — locks to the `Healthcare` namespace. `Patient` / `Diagnosis` /
//!   `LabValue` / `Medication` / `Treatment` / `Visit` / `VitalSign`
//!   resolve to the `0x09XX` Health codebook (Northstar T9).
//! - [`WoaBridge`]: `UnifiedBridge<ogar_vocab::ports::WoaPort>` — locks
//!   to the `WorkOrder` namespace. WoA's German/English commerce names
//!   (`Vorgang` / `Kunde` / `Rechnung` / …) resolve to the `0x02XX`
//!   commerce block, and `Stundenzettel` / `TimeEntry` resolve to the
//!   SAME `BILLABLE_WORK_ENTRY` (`0x0103`) as the planner ports — the
//!   cross-fork convergence pin (OGAR #93).
//! - [`SmbBridge`]: `UnifiedBridge<ogar_vocab::ports::SmbPort>` — locks
//!   to the `SMB` namespace. Sister of [`WoaBridge`]: SMB's `Kunde` /
//!   `Auftrag` / `Stundenzettel` resolve to the SAME canonical class_ids
//!   as the WoA equivalents (OGAR #93).
//! - [`OdooBridge`]: `UnifiedBridge<ogar_vocab::ports::OdooPort>` — locks
//!   to the `Odoo` namespace. Odoo model names (`account.move` /
//!   `res.partner` / …) resolve to the `0x02XX` commerce block, and the
//!   cross-arm `account.analytic.line` resolves to `BILLABLE_WORK_ENTRY`
//!   (`0x0103`) — the commerce-arm convergence pin (OGAR #94).

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
// The OGAR-port bridges added for #93/#94 expose the same `*_ALIASES`
// constants, so they get the matching `*_CODEBOOK` shim for symmetry.
#[allow(deprecated)]
pub use odoo_bridge::ODOO_CODEBOOK;
#[allow(deprecated)]
pub use smb_bridge::SMB_CODEBOOK;
#[allow(deprecated)]
pub use woa_bridge::WOA_CODEBOOK;
