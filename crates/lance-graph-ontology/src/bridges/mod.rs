//! Default tenant bridge implementations (OGIT side).
//!
//! These are the **legacy per-tenant bridges** with a bespoke struct
//! shape, locked to one OGIT namespace each and routing resolution
//! through the shared registry. They predate OGAR's codebook and have
//! no `PortSpec` impl in `ogar-vocab::ports`.
//!
//! The OGAR-driven port bridges (the generic `UnifiedBridge<P: PortSpec>`
//! harness and its `OpenProjectBridge` / `RedmineBridge` / `MedcareBridge`
//! aliases) live in the `lance-graph-ogar` crate (`lance_graph_ogar::bridges`)
//! — they couple to `ogar_vocab::ports` / `ogar_vocab::class_ids`, which is
//! OGAR, not OGIT. This crate (OGIT) must not depend on `ogar-vocab`.
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

mod ogit_bridge;
mod sharepoint_bridge;
mod spear_bridge;
mod woa_bridge;

pub use ogit_bridge::OgitBridge;
pub use sharepoint_bridge::SharePointBridge;
pub use spear_bridge::SpearBridge;
pub use woa_bridge::WoaBridge;
