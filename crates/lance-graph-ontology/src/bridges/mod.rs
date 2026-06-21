//! Default tenant bridge implementations.
//!
//! # Two layers
//!
//! - The **generic harness** [`unified::UnifiedBridge<P: PortSpec>`] is
//!   the one-and-only NamespaceBridge impl for OGAR-driven ports. It
//!   inherits everything that varies between ports
//!   (`NAMESPACE` / `BRIDGE_ID` / public-name → class_id aliases) from
//!   [`ogar_vocab::ports::PortSpec`]. Adding a port is `impl PortSpec
//!   for FooPort {…}` in OGAR — no bridge boilerplate here.
//! - The **legacy per-tenant bridges** ([`WoaBridge`], [`SpearBridge`],
//!   [`SharePointBridge`], [`OgitBridge`]) keep their bespoke struct
//!   shape for now. They predate OGAR's codebook and don't yet have a
//!   `PortSpec` impl in `ogar-vocab::ports`. When the WorkOrder /
//!   EmailCorrespondance / SharePoint namespaces get promoted into the
//!   codebook, these collapse the same way OpenProject, Redmine, and
//!   MedCare already did.
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
//!   resolve to the `0x09XX` Health codebook (Northstar T9). Single-
//!   tenant today; a future FMA / SNOMED curator converges on the same
//!   ids.
//!
//! # Per-tenant bridges (legacy struct shape)
//!
//! - [`OgitBridge`]: pass-through bridge for tools that already speak raw
//!   OGIT URIs. `bridge_id = "ogit"`. Locks to whatever namespace its
//!   constructor is called with.
//! - [`WoaBridge`]: locks to the `WorkOrder` namespace.
//! - [`SpearBridge`]: locks to the `EmailCorrespondance` namespace.
//! - [`SharePointBridge`]: locks to the `SharePoint` namespace.
//!
//! The `smb-bridge` and `callcenter-bridge` are NOT created in this
//! session: smb stays on its native ontology fallback, callcenter has
//! its own auth + per-customer scoping concerns that need a separate
//! design pass.

pub mod unified;

mod medcare_bridge;
mod ogit_bridge;
mod openproject_bridge;
mod redmine_bridge;
mod sharepoint_bridge;
mod spear_bridge;
mod woa_bridge;

pub use medcare_bridge::{HealthcarePort, MedcareBridge};
pub use ogit_bridge::OgitBridge;
pub use openproject_bridge::{OpenProjectBridge, OpenProjectPort};
pub use redmine_bridge::{RedmineBridge, RedminePort};
pub use sharepoint_bridge::SharePointBridge;
pub use spear_bridge::SpearBridge;
pub use unified::UnifiedBridge;
pub use woa_bridge::WoaBridge;

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
