//! WoA (Work Order Application) tenant bridge — locks to the `WorkOrder`
//! namespace. Phase 6 of this session emits the corresponding TTL into
//! `AdaWorldAPI/OGIT/NTO/WorkOrder/`.

use crate::bridge::{BridgeError, BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

pub const NAMESPACE: &str = "WorkOrder";

/// **Deprecated:** pull the classid via the OGAR PortSpec instead —
/// `ogar_vocab::ports::WoaPort::class_id(name)`. The codebook-aware
/// OGAR-driven bridge `lance_graph_ogar::bridges::WoaBridge` is also
/// deprecated; both share the same replacement path. See
/// `docs/CONSUMER-BRIDGE-DEPRECATION.md` + AdaWorldAPI/OGAR#95.
#[deprecated(
    note = "pull the classid via `WoaPort::class_id(name)` — see AdaWorldAPI/OGAR#95 + docs/CONSUMER-BRIDGE-DEPRECATION.md"
)]
pub struct WoaBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl WoaBridge {
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for WoaBridge {
    fn bridge_id(&self) -> &'static str {
        "woa"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for WoaBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

// Compile-only check that BridgeError is reachable from this crate.
#[allow(dead_code)]
fn _compile_check(b: &WoaBridge) -> std::result::Result<(), BridgeError> {
    let _ = b.entity("WorkOrder")?;
    Ok(())
}
