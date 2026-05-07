//! Spear (columnar mail server) tenant bridge — locks to the
//! `EmailCorrespondance` namespace (OGIT/NTO/EmailCorrespondance/).
//!
//! Spear's role in the cascade: it is the columnar Lance-backed mail store
//! consumed by the Stalwart IMAP/JMAP front-end and the SharePoint
//! mail-orchestration adapter. Both producers normalise their mail traffic
//! through this bridge so that an inbound message from Stalwart and the
//! same message reflected from a SharePoint mailbox land on the *same*
//! `ogit.EmailCorrespondance:*` URIs in the registry.
//!
//! The OGIT `EmailCorrespondance` namespace is sparsely populated today
//! (only `CarbonCopy` / `BlindCarbonCopy` attributes); this bridge will
//! keep working as additional entities (`Mail`, `Mailbox`, `Thread`,
//! `Attachment`, …) are emitted by future TTL waves — the scope-lock
//! catches any cross-namespace leak before it reaches Spear's columnar
//! schema.

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

pub const NAMESPACE: &str = "EmailCorrespondance";

pub struct SpearBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl SpearBridge {
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for SpearBridge {
    fn bridge_id(&self) -> &'static str {
        "spear"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for SpearBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}
