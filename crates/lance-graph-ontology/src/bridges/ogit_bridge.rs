//! OGIT pass-through bridge.
//!
//! Tools that already speak raw OGIT URIs do not need a public-name
//! dictionary — they hand the URI directly. The `OgitBridge` provides a
//! consistent `NamespaceBridge` surface for those callers, locking to a
//! single namespace at construction time.
//!
//! A common pattern is to spin one `OgitBridge` per OGIT namespace that
//! the consumer cares about, e.g. one for `Network`, another for `Auth`.

// This module IS the deprecated `OgitBridge`'s implementation; its own
// impl blocks reference the deprecated struct intentionally. Consumers in
// other crates still get the warning. See docs/CONSUMER-BRIDGE-DEPRECATION.md.
#![allow(deprecated)]

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

/// **Deprecated:** pull the classid via the OGAR PortSpec for the
/// relevant namespace (`ogar_vocab::ports::*Port::class_id(name)`).
/// See `docs/CONSUMER-BRIDGE-DEPRECATION.md` + AdaWorldAPI/OGAR#95.
#[deprecated(
    note = "pull the classid via the OGAR PortSpec for the namespace (e.g. `WoaPort::class_id(name)`) — see AdaWorldAPI/OGAR#95 + docs/CONSUMER-BRIDGE-DEPRECATION.md"
)]
pub struct OgitBridge {
    registry: Arc<OntologyRegistry>,
    namespace_name: String,
    g_lock: NamespaceId,
}

impl OgitBridge {
    /// Construct an OGIT bridge locked to the given namespace.
    pub fn for_namespace(registry: Arc<OntologyRegistry>, namespace: &str) -> Result<Self> {
        let g_lock = registry
            .namespace_id(namespace)
            .ok_or_else(|| Error::UnknownNamespace(namespace.to_string()))?;
        Ok(Self {
            registry,
            namespace_name: namespace.to_string(),
            g_lock,
        })
    }

    pub fn namespace_name(&self) -> &str {
        &self.namespace_name
    }
}

impl NamespaceBridge for OgitBridge {
    fn bridge_id(&self) -> &'static str {
        "ogit"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for OgitBridge {
    /// Default constructor: locks to the `Network` namespace, which is the
    /// most heavily-populated OGIT namespace and a reasonable smoke-test
    /// default. Most consumers should call `OgitBridge::for_namespace`
    /// directly with their own namespace name.
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::for_namespace(registry, "Network")
    }
}
