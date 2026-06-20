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
//!   `OpenProjectBridge` over the same 32 promoted concepts. Both
//!   bridges synthesize EntityRefs whose `entity_type_id()` is the
//!   shared OGAR codebook id for the canonical concept (so
//!   `WorkPackage` and `Issue` both → `0x0102 project_work_item`),
//!   delivering the cross-fork convergence pin.
//!
//! The shared codebook constants both project-management bridges
//! reference live in [`codebook`] — single source of truth so the
//! OpenProject port and the Redmine port can't drift on the same
//! canonical concept's class_id.
//!
//! The `smb-bridge` and `callcenter-bridge` are NOT created in this
//! session: smb stays on its native ontology fallback, callcenter has its
//! own auth + per-customer scoping concerns that need a separate design pass.

pub mod codebook;
mod medcare_bridge;
mod ogit_bridge;
mod openproject_bridge;
mod redmine_bridge;
mod sharepoint_bridge;
mod spear_bridge;
mod woa_bridge;

pub use medcare_bridge::MedcareBridge;
pub use ogit_bridge::OgitBridge;
pub use openproject_bridge::{OpenProjectBridge, OPENPROJECT_CODEBOOK};
pub use redmine_bridge::{RedmineBridge, REDMINE_CODEBOOK};
pub use sharepoint_bridge::SharePointBridge;
pub use spear_bridge::SpearBridge;
pub use woa_bridge::WoaBridge;
