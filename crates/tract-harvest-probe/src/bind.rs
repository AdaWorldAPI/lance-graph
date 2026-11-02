//! The ONNX registry read as a consumer of the DR64 route / DT64 bind store
//! split (D-DT64-6 §1: "DR64 is direct routing … DT64 is the DTO bind
//! store").
//!
//! A classid is a codebook lens, not a codebook item: the ONNX operator
//! codebook is one lens ([`crate::registry::ONNX_LENS`]), and a node binds
//! to an item in it, [`MethodKey`] `{ lens, ordinal }`.
//!
//! [`Registry::lookup`] picks an item from the node's signature and hands
//! back its body in one call. Here the two halves are separated without
//! changing either:
//!
//! - [`route`] is the binder. It runs the signature match once per node and
//!   records only `node position → (lens, ordinal)`, sparse; an unregistered
//!   node has no route. This is the table DR would own.
//! - [`resolve`] is the bind-store side. It reads the routed key through the
//!   lens and takes the item at that ordinal, never by signature.
//!
//! Nothing here is the DT64 crate or the DR route table; both are other
//! owners'. This module shows the ONNX instance already factors into them.
use crate::registry::{Fact, MethodKey, Registration, Registry};
use crate::source::SourceInventory;
use lance_graph_contract::class_view::ClassId;
use std::collections::{BTreeMap, HashMap};

/// `node position → (lens, ordinal)`, sparse. `version` digests the
/// registry version and every entry, so a re-route is a different version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routes {
    pub version: u64,
    pub map: BTreeMap<usize, MethodKey>,
}

/// Which link of `position → lens → item` failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unresolved {
    /// No route at this position: UNKNOWN, never "no behaviour".
    Route { node: usize },
    /// Routed through a lens this store does not read.
    Lens { node: usize, lens: ClassId },
    /// The lens is right but holds no item at this ordinal.
    Item { node: usize, key: MethodKey },
}

/// Bind every node by signature once, keeping only its codebook key. The
/// binder reads per-value facts: a node position is the global source node
/// index, the same one the DR64 tiles map to local bits.
/// Initializers without a fact get their protobuf dtype and dims, as in
/// `fold`.
pub fn route(src: &SourceInventory, facts: &HashMap<String, Fact>, registry: &Registry) -> Routes {
    let facts = crate::fold::with_initializer_facts(src, facts);
    let mut map = BTreeMap::new();
    for (n, node) in src.nodes.iter().enumerate() {
        let args: Vec<Option<&Fact>> = node.inputs.iter().map(|i| facts.get(i)).collect();
        if let Some(r) = registry.lookup(node, &args) {
            map.insert(n, registry.key_of(r));
        }
    }
    let mut parts: Vec<Vec<u8>> = vec![registry.revision().to_le_bytes().to_vec()];
    parts.extend(map.iter().map(|(n, k)| {
        let mut b = (*n as u64).to_le_bytes().to_vec();
        b.extend(k.lens.to_le_bytes());
        b.extend(k.ordinal.to_le_bytes());
        b
    }));
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Routes {
        version: crate::plan::digest64(&refs),
        map,
    }
}

/// `position → lens → item`, naming the link that failed.
pub fn resolve<'r>(
    routes: &Routes,
    registry: &'r Registry,
    node: usize,
) -> Result<&'r Registration, Unresolved> {
    let &key = routes.map.get(&node).ok_or(Unresolved::Route { node })?;
    if key.lens != registry.lens {
        return Err(Unresolved::Lens {
            node,
            lens: key.lens,
        });
    }
    registry.get(key).ok_or(Unresolved::Item { node, key })
}
