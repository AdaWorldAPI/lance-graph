//! Odoo (ERP) tenant bridge — locks to the `Odoo` namespace.
//!
//! Odoo lands in OGIT at G=50, inheriting from FIBO-FND (financial
//! foundations) per `modules/odoo/manifest.yaml`. Per `hydrators/odoo.rs`
//! (Seam decision 1 / Option B), odoo does NOT get its own CAM codebook
//! family — every odoo class with an alignment axiom is
//! `owl:equivalentClass`-routed into an existing FIBO/SKR slot. This bridge
//! is the public-name → OGIT URI translator for the odoo extraction
//! surface; the heavy lifting (foundry-family resolution, Layer-2
//! alignment pivot lookup) lives in
//! `lance_graph_callcenter::odoo_alignment` and is consumed via the
//! single-source rule documented there.

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

pub const NAMESPACE: &str = "Odoo";

pub struct OdooBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl OdooBridge {
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for OdooBridge {
    fn bridge_id(&self) -> &'static str {
        "odoo"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for OdooBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}
