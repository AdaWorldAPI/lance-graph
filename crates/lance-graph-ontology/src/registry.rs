//! `OntologyRegistry` — the in-memory dictionary + (optional) Lance cache.
//!
//! The registry is the single canonical surface for the new crate. It does
//! three things:
//!
//! 1. Hydrates from a TTL root via [`OntologyRegistry::hydrate_once`] (or
//!    its sync sibling [`OntologyRegistry::hydrate_once_sync`]). Idempotent
//!    via SHA256 of the TTL root.
//! 2. Resolves `(bridge_id, public_name)` → `SchemaPtr` and OGIT URI →
//!    `SchemaPtr` for the hot path.
//! 3. Persists rows to a Lance dataset (under the `lance-cache` feature).
//!    Without that feature the registry is in-memory only — sufficient for
//!    tests and for consumers that re-hydrate from TTL on every start.
//!
//! Carrier-method doctrine throughout: methods on the registry, not free
//! functions on registry state.

use crate::error::{Error, Result};
use crate::namespace::{NamespaceId, SchemaPtr};
use crate::proposal::{
    HydrationFailure, HydrationReport, IdentityCodec, MappingHandle, MappingProposal,
    MappingProposalKind, MappingRow, ProvenanceBundle, QualiaMeta,
};
use crate::semantic_types::SemanticTypeMap;
use crate::ttl_parse::{parse_ttl_directory, ttl_root_checksum};
use lance_graph_contract::property::{Marking, SemanticType};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// The single ontology registry.
pub struct OntologyRegistry {
    inner: RwLock<RegistryState>,
    sem_map: SemanticTypeMap,
    #[cfg_attr(not(feature = "lance-cache"), allow(dead_code))]
    lance_path: Option<PathBuf>,
}

#[derive(Default)]
struct RegistryState {
    rows: Vec<MappingRow>,
    by_bridge_name: HashMap<(String, String), u32>,
    by_uri: HashMap<String, u32>,
    by_namespace: HashMap<String, NamespaceId>,
    namespace_order: Vec<String>,
    last_root_checksum: Option<String>,
}

impl OntologyRegistry {
    /// In-memory registry. Lance persistence is disabled.
    pub fn new_in_memory() -> Self {
        Self {
            inner: RwLock::new(RegistryState::default()),
            sem_map: SemanticTypeMap::defaults().clone(),
            lance_path: None,
        }
    }

    /// In-memory registry with a custom semantic-type map.
    pub fn with_semantic_types(sem: SemanticTypeMap) -> Self {
        Self {
            inner: RwLock::new(RegistryState::default()),
            sem_map: sem,
            lance_path: None,
        }
    }

    /// Lance-backed registry. Opens the dataset at `lance_path` (creating
    /// it if missing) and replays the dictionary into memory. Async because
    /// Lance's I/O surface is async.
    #[cfg(feature = "lance-cache")]
    pub async fn open(lance_path: &Path) -> Result<Self> {
        use crate::lance_cache::LanceWriter;
        let writer = LanceWriter::open_or_create(lance_path).await?;
        let mut state = RegistryState::default();
        for row in writer.replay().await? {
            state.absorb_row(row);
        }
        state.last_root_checksum = writer.last_root_checksum().await?;
        Ok(Self {
            inner: RwLock::new(state),
            sem_map: SemanticTypeMap::defaults().clone(),
            lance_path: Some(lance_path.to_path_buf()),
        })
    }

    /// Sync hydration. Reads TTL files under `ttl_root`, parses each, and
    /// appends every produced `MappingProposal` to the in-memory dictionary.
    /// Idempotent: skips parsing if the TTL root checksum matches the last
    /// hydration. Lance persistence is NOT touched here — see
    /// [`OntologyRegistry::hydrate_once`] for the persisted variant.
    pub fn hydrate_once_sync(
        &self,
        ttl_root: &Path,
        namespaces: &[&str],
    ) -> Result<HydrationReport> {
        let root_checksum = ttl_root_checksum(ttl_root)?;
        if let Some(prev) = &self.inner.read().unwrap().last_root_checksum {
            if prev == &root_checksum {
                return Ok(HydrationReport {
                    from_cache: true,
                    ..Default::default()
                });
            }
        }

        let (proposals, failures) =
            parse_ttl_directory(ttl_root, "ogit", &self.sem_map, namespaces)?;
        if proposals.is_empty() && failures.is_empty() {
            return Err(Error::EmptyHydration(ttl_root.to_path_buf()));
        }

        let mut report = HydrationReport {
            failures: failures.clone(),
            failed: failures.len() as u32,
            ..Default::default()
        };
        let mut state = self.inner.write().unwrap();
        let mut seen_namespaces: std::collections::BTreeSet<String> = Default::default();
        for proposal in proposals {
            seen_namespaces.insert(proposal.namespace.clone());
            match state.append(proposal, &self.sem_map) {
                AppendOutcome::Inserted(_) => report.registered += 1,
                AppendOutcome::Idempotent => report.skipped_idempotent += 1,
                AppendOutcome::Failed(reason) => {
                    report.failed += 1;
                    report.failures.push(HydrationFailure {
                        source: ttl_root.display().to_string(),
                        reason,
                    });
                }
            }
        }
        state.last_root_checksum = Some(root_checksum);
        report.namespaces_seen = seen_namespaces.into_iter().collect();
        Ok(report)
    }

    /// Async hydration. Same as `hydrate_once_sync` but additionally
    /// persists newly-registered rows to the Lance dataset (when the
    /// crate is built with `lance-cache`). Without `lance-cache` it is
    /// equivalent to `hydrate_once_sync` and is provided for API parity.
    #[cfg(feature = "lance-cache")]
    pub async fn hydrate_once(
        &self,
        ttl_root: &Path,
        namespaces: &[&str],
    ) -> Result<HydrationReport> {
        let report = self.hydrate_once_sync(ttl_root, namespaces)?;
        if report.from_cache {
            return Ok(report);
        }
        if let Some(lance_path) = &self.lance_path {
            use crate::lance_cache::LanceWriter;
            let writer = LanceWriter::open_or_create(lance_path).await?;
            let rows: Vec<MappingRow> = self.inner.read().unwrap().rows.clone();
            writer.flush(&rows).await?;
            if let Some(cs) = &self.inner.read().unwrap().last_root_checksum {
                writer.set_last_root_checksum(cs).await?;
            }
        }
        Ok(report)
    }

    /// Append a single proposal directly. Used by schema scanners and
    /// customer admin forms.
    pub fn append_mapping(&self, proposal: MappingProposal) -> Result<MappingHandle> {
        let mut state = self.inner.write().unwrap();
        match state.append(proposal, &self.sem_map) {
            AppendOutcome::Inserted(handle) => Ok(handle),
            AppendOutcome::Idempotent => {
                // Resolve back the existing handle.
                let bridge_id = state.rows.last().map(|r| r.bridge_id.clone()).unwrap_or_default();
                let public_name = state.rows.last().map(|r| r.public_name.clone()).unwrap_or_default();
                let key = (bridge_id, public_name);
                let idx = *state
                    .by_bridge_name
                    .get(&key)
                    .ok_or_else(|| Error::other("idempotent append produced no row index"))?;
                let row = &state.rows[idx as usize];
                Ok(MappingHandle {
                    schema_ptr: row.schema_ptr,
                    row_index: idx,
                })
            }
            AppendOutcome::Failed(reason) => Err(Error::other(reason)),
        }
    }

    /// Resolve `(bridge_id, public_name)` → `SchemaPtr`.
    pub fn resolve(&self, bridge_id: &str, public_name: &str) -> Option<SchemaPtr> {
        let state = self.inner.read().unwrap();
        let key = (bridge_id.to_string(), public_name.to_string());
        state
            .by_bridge_name
            .get(&key)
            .map(|idx| state.rows[*idx as usize].schema_ptr)
    }

    /// Resolve raw OGIT URI → `SchemaPtr`.
    pub fn resolve_uri(&self, ogit_uri: &str) -> Option<SchemaPtr> {
        let state = self.inner.read().unwrap();
        state
            .by_uri
            .get(ogit_uri)
            .map(|idx| state.rows[*idx as usize].schema_ptr)
    }

    /// Get the full row for a given OGIT URI.
    pub fn row_for_uri(&self, ogit_uri: &str) -> Option<MappingRow> {
        let state = self.inner.read().unwrap();
        state
            .by_uri
            .get(ogit_uri)
            .map(|idx| state.rows[*idx as usize].clone())
    }

    /// Look up a namespace's `NamespaceId` (G).
    pub fn namespace_id(&self, name: &str) -> Option<NamespaceId> {
        self.inner.read().unwrap().by_namespace.get(name).copied()
    }

    /// Names of all known namespaces, in registration order (1-indexed in
    /// the returned slice; `NamespaceId(0)` is reserved for unknown).
    pub fn namespace_names(&self) -> Vec<String> {
        self.inner.read().unwrap().namespace_order.clone()
    }

    /// Enumerate all rows under a given namespace.
    pub fn enumerate(&self, namespace: &str) -> Vec<MappingRow> {
        let state = self.inner.read().unwrap();
        let id = match state.by_namespace.get(namespace) {
            Some(id) => *id,
            None => return Vec::new(),
        };
        state
            .rows
            .iter()
            .filter(|r| r.namespace_id == id)
            .cloned()
            .collect()
    }

    /// Count the rows in the dictionary.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().rows.is_empty()
    }

    /// Thread a [`ProvenanceBundle`] onto its row (FIX-3, consumes
    /// [`crate::ttl_parse::parse_with_provenance`] — no re-walk).
    pub fn attach_provenance(&self, bundle: &ProvenanceBundle) -> bool {
        let mut s = self.inner.write().unwrap();
        s.by_uri.get(&bundle.entity_uri).copied().map(|idx| {
            s.rows[idx as usize].attribute_sources = bundle.attribute_sources.clone();
        }).is_some()
    }

    /// Attach a `ThinkingStyle` (D-PARITY-V2-12) to the row at `ogit_uri`.
    pub fn attach_thinking_style(
        &self,
        ogit_uri: &str,
        style: lance_graph_contract::thinking::ThinkingStyle,
    ) -> bool {
        let mut s = self.inner.write().unwrap();
        s.by_uri.get(ogit_uri).copied().map(|idx| {
            s.rows[idx as usize].thinking_style = Some(style);
        }).is_some()
    }

    /// Resolve a `BindSpace.entity_type` index to its row (D-CASCADE-V1-7).
    pub fn enumerate_first_with_entity_type_id(&self, entity_type_id: u16) -> Option<MappingRow> {
        let s = self.inner.read().unwrap();
        s.rows.iter().find(|r| r.schema_ptr.entity_type_id() == entity_type_id).cloned()
    }

    /// Export the registry to an OGIT-shaped TTL fragment for the named
    /// namespace. Used by the Lance ↔ OGIT round-trip and for fork PRs
    /// that promote schema-scanner suggestions back into the canonical
    /// vocabulary.
    pub fn export_ttl(&self, namespace: &str, out: &Path) -> Result<()> {
        let rows = self.enumerate(namespace);
        let mut buf = String::new();
        buf.push_str("@prefix ogit: <http://www.purl.org/ogit/> .\n");
        buf.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        buf.push_str("@prefix dcterms: <http://purl.org/dc/terms/> .\n");
        buf.push_str(&format!(
            "@prefix ogit.{ns}: <http://www.purl.org/ogit/{ns}/> .\n\n",
            ns = namespace
        ));
        for row in rows {
            let name = row.ogit_uri.name().unwrap_or("Unknown");
            let kind = row.kind.as_str();
            buf.push_str(&format!("# kind: {kind}; bridge: {}\n", row.bridge_id));
            buf.push_str(&format!("ogit.{}:{}\n", namespace, name));
            buf.push_str("\ta rdfs:Class ;\n");
            buf.push_str("\trdfs:subClassOf ogit:Entity ;\n");
            buf.push_str(&format!("\trdfs:label \"{}\" ;\n", name));
            buf.push_str(&format!(
                "\tdcterms:source \"{}\" .\n\n",
                row.source_uri.replace('"', "'")
            ));
        }
        std::fs::write(out, buf).map_err(|source| Error::Io {
            path: out.to_path_buf(),
            source,
        })?;
        Ok(())
    }
}

enum AppendOutcome {
    Inserted(MappingHandle),
    Idempotent,
    Failed(String),
}

impl RegistryState {
    fn append(
        &mut self,
        proposal: MappingProposal,
        sem: &SemanticTypeMap,
    ) -> AppendOutcome {
        let key = (proposal.bridge_id.clone(), proposal.public_name.clone());
        if let Some(existing) = self.by_bridge_name.get(&key) {
            let row = &self.rows[*existing as usize];
            if row.checksum == proposal.checksum {
                return AppendOutcome::Idempotent;
            }
        }
        // Allocate or look up the namespace id.
        let namespace_id = if let Some(id) = self.by_namespace.get(&proposal.namespace) {
            *id
        } else {
            let next = (self.namespace_order.len() + 1) as u8;
            if next == 0 {
                return AppendOutcome::Failed("namespace overflow".to_string());
            }
            let id = NamespaceId(next);
            self.by_namespace.insert(proposal.namespace.clone(), id);
            self.namespace_order.push(proposal.namespace.clone());
            id
        };

        let kind = proposal.schema_kind();
        let entity_type_id = (self.rows.len() + 1) as u16;
        let schema_ptr = SchemaPtr::new(namespace_id, entity_type_id, kind);

        let semantic_type = match &proposal.kind {
            MappingProposalKind::Attribute { semantic_type, .. } => semantic_type.clone(),
            _ => sem.lookup(proposal.ogit_uri.as_str()),
        };
        // D-CASCADE-V1-7: derive subject/object/entity-type strings
        // (META-NUDGE-1); codec/qualia/thinking attach via `attach_*`.
        let entity_name = proposal.ogit_uri.name().unwrap_or(&proposal.public_name).to_string();
        let (subject_type, object_type, entity_type_ref) = match &proposal.kind {
            MappingProposalKind::Edge { link } => {
                (link.subject_type.to_string(), link.object_type.to_string(), String::new())
            }
            MappingProposalKind::Attribute { .. } => (String::new(), String::new(), entity_name),
            MappingProposalKind::Entity { schema } => {
                (String::new(), String::new(), schema.name.to_string())
            }
        };
        let row = MappingRow {
            bridge_id: proposal.bridge_id.clone(),
            public_name: proposal.public_name.clone(),
            ogit_uri: proposal.ogit_uri.clone(),
            namespace_id,
            schema_ptr,
            kind,
            semantic_type,
            marking: proposal.marking,
            confidence: proposal.confidence,
            created_at_us: now_micros(),
            created_by: proposal.created_by.clone(),
            source_uri: proposal.source_uri.clone(),
            active: true,
            checksum: proposal.checksum.clone(),
            identity_codec: IdentityCodec::default(),
            qualia_meta: QualiaMeta::default(),
            thinking_style: None,
            attribute_sources: Vec::new(),
            subject_type,
            object_type,
            entity_type_ref,
        };
        let idx = self.rows.len() as u32;
        self.rows.push(row);
        self.by_bridge_name.insert(key, idx);
        self.by_uri
            .insert(proposal.ogit_uri.as_str().to_string(), idx);
        AppendOutcome::Inserted(MappingHandle {
            schema_ptr,
            row_index: idx,
        })
    }

    // Used by `lance_cache::LanceWriter::replay()` when reconstituting the
    // in-memory state from a Lance dataset on `OntologyRegistry::open`.
    // The reader only compiles under the `lance-cache` feature; suppress
    // the dead-code lint when the feature is off.
    #[cfg_attr(not(feature = "lance-cache"), allow(dead_code))]
    fn absorb_row(&mut self, row: MappingRow) {
        let key = (row.bridge_id.clone(), row.public_name.clone());
        if !self.by_namespace.contains_key(row.ogit_uri.namespace().unwrap_or("")) {
            let ns = row.ogit_uri.namespace().unwrap_or("").to_string();
            if !ns.is_empty() {
                self.by_namespace.insert(ns.clone(), row.namespace_id);
                if !self.namespace_order.contains(&ns) {
                    self.namespace_order.push(ns);
                }
            }
        }
        let idx = self.rows.len() as u32;
        self.by_bridge_name.insert(key, idx);
        self.by_uri.insert(row.ogit_uri.as_str().to_string(), idx);
        self.rows.push(row);
    }
}

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// Suppress unused-warning for `Marking` / `SemanticType` re-exports we
// surface via MappingRow but don't otherwise reference here.
#[allow(dead_code)]
const _MARKER: (Marking, SemanticType) = (Marking::Internal, SemanticType::PlainText);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::{OgitUri, SchemaKind};
    use crate::proposal::MappingProposalKind;
    use lance_graph_contract::property::Schema;

    fn proposal(uri: &str) -> MappingProposal {
        let parsed = OgitUri::parse(uri).unwrap();
        let ns = parsed.namespace().unwrap().to_string();
        let name = parsed.name().unwrap().to_string();
        MappingProposal {
            public_name: uri.to_string(),
            bridge_id: "ogit".to_string(),
            ogit_uri: parsed,
            namespace: ns,
            kind: MappingProposalKind::Entity {
                schema: Schema::builder(Box::leak(name.into_boxed_str()))
                    .required("id")
                    .build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: format!("test://{uri}"),
            checksum: format!("checksum-{uri}"),
            created_by: "test".to_string(),
        }
    }

    #[test]
    fn append_and_resolve() {
        let reg = OntologyRegistry::new_in_memory();
        let h = reg.append_mapping(proposal("ogit.Network:IPAddress")).unwrap();
        assert_eq!(reg.len(), 1);
        let resolved = reg.resolve("ogit", "ogit.Network:IPAddress").unwrap();
        assert_eq!(resolved, h.schema_ptr);
        assert_eq!(resolved.kind(), SchemaKind::Entity);
        assert!(resolved.namespace_id().is_known());
    }

    #[test]
    fn idempotent_double_append() {
        let reg = OntologyRegistry::new_in_memory();
        reg.append_mapping(proposal("ogit.Network:IPAddress")).unwrap();
        let h = reg.append_mapping(proposal("ogit.Network:IPAddress")).unwrap();
        // Same checksum → idempotent: reuses the existing row.
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.resolve("ogit", "ogit.Network:IPAddress").unwrap(), h.schema_ptr);
    }

    #[test]
    fn enumerate_groups_by_namespace() {
        let reg = OntologyRegistry::new_in_memory();
        reg.append_mapping(proposal("ogit.Network:IPAddress")).unwrap();
        reg.append_mapping(proposal("ogit.Network:MACAddress")).unwrap();
        reg.append_mapping(proposal("ogit.Auth:Account")).unwrap();
        assert_eq!(reg.enumerate("Network").len(), 2);
        assert_eq!(reg.enumerate("Auth").len(), 1);
        assert_eq!(reg.enumerate("Missing").len(), 0);
    }

    #[test]
    fn namespace_ids_are_dense_and_unique() {
        let reg = OntologyRegistry::new_in_memory();
        reg.append_mapping(proposal("ogit.A:X")).unwrap();
        reg.append_mapping(proposal("ogit.B:Y")).unwrap();
        assert_ne!(reg.namespace_id("A"), reg.namespace_id("B"));
        assert_eq!(reg.namespace_id("A").unwrap().raw(), 1);
        assert_eq!(reg.namespace_id("B").unwrap().raw(), 2);
    }
}
