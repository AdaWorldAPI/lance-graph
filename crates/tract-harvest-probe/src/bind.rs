//! The ONNX registry read as a consumer of the DR64 route / DT64 bind store
//! split (D-DT64-6 §1: "DR64 is direct routing … DT64 is the DTO bind
//! store"; EXO-50 drafts -08 / -09: `address --route--> classid --binding-->
//! method`).
//!
//! [`Registry::lookup`] does both halves in one call: it picks a
//! registration from the node's signature and hands back its body. Here they
//! are separated without changing either:
//!
//! - [`route`] is the binder. It runs the signature match once per node and
//!   records only `node position → classid`, sparse; an unregistered node
//!   has no route. This is the table DR would own.
//! - [`resolve`] is the bind-store side. It takes a routed classid and finds
//!   its registration by classid alone, never by signature. This works
//!   because every registration carries its own classid and no classid is
//!   registered twice (`classids_are_unique` pins that).
//!
//! Nothing here is the DT64 crate or the DR route table; both are other
//! owners' (EXO-50 -05 decision). This module only shows that the ONNX
//! instance already factors into them, so adapting it is a move, not a
//! redesign.
use crate::registry::{Fact, Registration, Registry};
use crate::source::SourceInventory;
use lance_graph_contract::class_view::ClassId;
use std::collections::{BTreeMap, HashMap};

/// `node position → classid`, sparse. `version` digests the registry
/// version and every entry, so a re-route is a different version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routes {
    pub version: u64,
    pub map: BTreeMap<usize, ClassId>,
}

/// Which link of `position → classid → registration` failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unresolved {
    /// No route at this position: UNKNOWN, never "no behaviour".
    Route { node: usize },
    /// Routed to a classid the bind store does not hold.
    Class { node: usize, classid: ClassId },
}

/// Bind every node by signature once, keeping only its classid. The binder
/// reads per-value facts, not a tile: a node position is the source node
/// index, so a graph larger than one 64-node tile routes too. Initializers
/// without a fact get their protobuf dtype and dims, as in `fold`.
pub fn route(src: &SourceInventory, facts: &HashMap<String, Fact>, registry: &Registry) -> Routes {
    let facts = crate::fold::with_initializer_facts(src, facts);
    let mut map = BTreeMap::new();
    for (n, node) in src.nodes.iter().enumerate() {
        let args: Vec<Option<&Fact>> = node.inputs.iter().map(|i| facts.get(i)).collect();
        if let Some(r) = registry.lookup(node, &args) {
            map.insert(n, r.classid);
        }
    }
    let mut parts: Vec<Vec<u8>> = vec![registry.version.to_le_bytes().to_vec()];
    parts.extend(map.iter().map(|(n, c)| {
        let mut b = (*n as u64).to_le_bytes().to_vec();
        b.extend((*c as u64).to_le_bytes());
        b
    }));
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Routes {
        version: crate::plan::digest64(&refs),
        map,
    }
}

impl Registry {
    /// The registration holding `classid`; the bind-store lookup.
    pub fn by_classid(&self, classid: ClassId) -> Option<&Registration> {
        self.regs.iter().find(|r| r.classid == classid)
    }
}

/// `position → classid → registration`, naming the link that failed.
pub fn resolve<'r>(
    routes: &Routes,
    registry: &'r Registry,
    node: usize,
) -> Result<&'r Registration, Unresolved> {
    let &classid = routes.map.get(&node).ok_or(Unresolved::Route { node })?;
    registry
        .by_classid(classid)
        .ok_or(Unresolved::Class { node, classid })
}
