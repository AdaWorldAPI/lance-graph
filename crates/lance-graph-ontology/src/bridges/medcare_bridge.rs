//! MedCare (healthcare) tenant bridge — locks to the `Healthcare`
//! namespace. The Healthcare namespace itself is reserved and will be
//! populated by a future session (the FMA / SNOMED / RadLex import is the
//! remit of `lance-graph-rdf` in `lance-graph-rdf-fma-snomed-v1`).

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

pub const NAMESPACE: &str = "Healthcare";

pub struct MedcareBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl MedcareBridge {
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for MedcareBridge {
    fn bridge_id(&self) -> &'static str {
        "medcare"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for MedcareBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}
