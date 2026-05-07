//! Default tenant bridge implementations.
//!
//! Three bridges ship in this session:
//!
//! - [`OgitBridge`]: pass-through bridge for tools that already speak raw
//!   OGIT URIs. `bridge_id = "ogit"`. Locks to whatever namespace its
//!   constructor is called with (typically the namespace of the caller).
//! - [`WoaBridge`]: locks to the `WorkOrder` namespace. Public names like
//!   `Customer`, `WorkOrder`, `Position` are translated via the registry
//!   to the corresponding `ogit.WorkOrder:*` URIs.
//! - [`MedcareBridge`]: locks to the `Healthcare` namespace.
//!
//! The `smb-bridge` and `callcenter-bridge` are NOT created in this
//! session: smb stays on its native ontology fallback, callcenter has its
//! own auth + per-customer scoping concerns that need a separate design pass.

mod medcare_bridge;
mod ogit_bridge;
mod woa_bridge;

pub use medcare_bridge::MedcareBridge;
pub use ogit_bridge::OgitBridge;
pub use woa_bridge::WoaBridge;
