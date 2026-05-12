//! SharePoint (content orchestration) tenant bridge — locks to the
//! `SharePoint` namespace (OGIT/NTO/SharePoint/).
//!
//! SharePoint's role in the cascade: the Sharepoint repo is a C# .NET 8
//! content-orchestration platform (UploadIntent / DriveScope / IUploadScheduler)
//! whose Rust transcode target is `smb-office-rs` (per SharePoint's
//! `RUST_TRANSCODE_PLAN.md`). The Rust side normalises every SharePoint
//! addressing pair (`SiteId`, `DriveId`) and every ComplianceTagging
//! attribute through `SharePointBridge` so it lands on the same
//! `ogit.SharePoint:*` URIs in the registry — regardless of whether the
//! producer is the C# spine via FFI or a future native Rust transport.
//!
//! Mail traffic flowing through SharePoint mailboxes still routes through
//! `SpearBridge` / `EmailCorrespondance` — the two namespaces are
//! complementary, not overlapping: SharePoint covers documents / drives /
//! sites, EmailCorrespondance covers mail headers and bodies.
//!
//! The OGIT `SharePoint` namespace is sparsely populated today (only
//! `SiteId` / `DriveId` attribute TTLs ship in the initial wave); this
//! bridge will keep working as additional entities (`Site`, `Drive`,
//! `DriveItem`, `UploadIntent`, …) are emitted by future TTL waves —
//! the scope-lock catches any cross-namespace leak before it reaches the
//! orchestrator's storage schema.

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

pub const NAMESPACE: &str = "SharePoint";

pub struct SharePointBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl SharePointBridge {
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for SharePointBridge {
    fn bridge_id(&self) -> &'static str {
        "sharepoint"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for SharePointBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}
