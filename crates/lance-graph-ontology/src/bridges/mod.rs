//! Default tenant bridge implementations.
//!
//! Five bridges ship today:
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
//!
//! The `smb-bridge` and `callcenter-bridge` are NOT created in this
//! session: smb stays on its native ontology fallback, callcenter has its
//! own auth + per-customer scoping concerns that need a separate design pass.

mod medcare_bridge;
mod ogit_bridge;
mod sharepoint_bridge;
mod spear_bridge;
mod woa_bridge;

pub use medcare_bridge::MedcareBridge;
pub use ogit_bridge::OgitBridge;
pub use sharepoint_bridge::SharePointBridge;
pub use spear_bridge::SpearBridge;
pub use woa_bridge::WoaBridge;
