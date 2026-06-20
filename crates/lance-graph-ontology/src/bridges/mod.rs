//! Default tenant bridge implementations.
//!
//! Seven bridges ship today:
//!
//! - [`OgitBridge`]: pass-through bridge for tools that already speak raw
//!   OGIT URIs. `bridge_id = "ogit"`. Locks to whatever namespace its
//!   constructor is called with (typically the namespace of the caller).
//! - [`WoaBridge`]: locks to the `WorkOrder` namespace. Public names like
//!   `Customer`, `WorkOrder`, `Position` are translated via the registry
//!   to the corresponding `ogit.WorkOrder:*` URIs.
//! - [`MedcareBridge`]: locks to the `Healthcare` namespace.
//! - [`SpearBridge`]: locks to the `EmailCorrespondance` namespace. Used
//!   by the spear columnar mail server with stalwart (IMAP/JMAP) and
//!   SharePoint as upstream mail-orchestration producers — both feed the
//!   same `ogit.EmailCorrespondance:*` URIs through this bridge.
//! - [`SharePointBridge`]: locks to the `SharePoint` namespace. Used by
//!   the Sharepoint→smb-office-rs content orchestrator (UploadIntent /
//!   DriveScope / ComplianceTagging) — distinct from EmailCorrespondance:
//!   one covers documents / drives / sites, the other covers mail.
//! - [`OpenProjectBridge`]: locks to the `OpenProject` namespace.
//!   Public names like `WorkPackage` / `TimeEntry` / `Project` resolve
//!   to `ogit.OpenProject:*` URIs. Northstar plan §3 C4 — supplies the
//!   port (`openproject-nexgen-rs` + `op-canon`) with the scoped
//!   registry view every consumer that touches OpenProject data on the
//!   unified bridge goes through.
//! - [`RedmineBridge`]: locks to the `Redmine` namespace. Public names
//!   like `Issue` / `TimeEntry` / `Project` resolve to
//!   `ogit.Redmine:*` URIs. Northstar plan §3 C5 — sibling of
//!   `OpenProjectBridge` over the same 32 promoted concepts; both
//!   bridges' `entity_type_id()` returns the SAME OGAR codebook id
//!   for the canonical concept (`Issue` and `WorkPackage` both →
//!   `0x0102 project_work_item`), so cross-fork convergence is the
//!   default not the exception.
//!
//! The `smb-bridge` and `callcenter-bridge` are NOT created in this
//! session: smb stays on its native ontology fallback, callcenter has its
//! own auth + per-customer scoping concerns that need a separate design pass.

mod medcare_bridge;
mod ogit_bridge;
mod openproject_bridge;
mod redmine_bridge;
mod sharepoint_bridge;
mod spear_bridge;
mod woa_bridge;

pub use medcare_bridge::MedcareBridge;
pub use ogit_bridge::OgitBridge;
pub use openproject_bridge::OpenProjectBridge;
pub use redmine_bridge::RedmineBridge;
pub use sharepoint_bridge::SharePointBridge;
pub use spear_bridge::SpearBridge;
pub use woa_bridge::WoaBridge;
