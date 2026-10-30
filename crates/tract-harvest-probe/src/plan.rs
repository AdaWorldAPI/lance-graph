//! Resolve once, then route: every Identity position of the DR#3 graph is
//! bound to the registration of its op signature, its WHEN is evaluated over
//! the cold facts, and the result is a schedule of `(node, classid, DO)` read
//! off the `dep` relation. Execution routes each node to its DO and nothing
//! else; it never consults Tract.
//!
//! The plan is sealed in a [`Frame`] whose revisions are (source digest,
//! registry version, coordinate assignment). A plan whose frame is not the
//! store's current seal refuses to run (stale generation, D-DT64-6 D8).

use std::collections::HashMap;

use dr_math_binding_probe::{Frame, FrameRevisions, FrameStore};
use lance_graph_contract::class_view::ClassId;
use sha2::{Digest, Sha256};

use crate::dr3::OnnxGraph;
use crate::registry::{
    render_signature, signature_dtype, NodeView, Numeric, Refusal, Registry, K3,
};
use crate::source::SourceInventory;
use crate::value::Value;

/// One resolved Identity position.
#[derive(Debug, Clone)]
pub struct Bound {
    pub node: usize,
    pub name: String,
    pub classid: ClassId,
    pub method: &'static str,
    pub numeric: Numeric,
    reg: usize,
}

pub struct Plan<'r> {
    pub graph: OnnxGraph,
    /// In schedule order.
    pub steps: Vec<Bound>,
    registry: &'r Registry,
    nodes: Vec<crate::source::SourceNode>,
    frame: Frame,
}

fn digest64(parts: &[&[u8]]) -> u64 {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    u64::from_le_bytes(h.finalize()[..8].try_into().expect("8 bytes"))
}

/// The revisions a plan of this source under this registry is sealed at.
pub fn revisions(src: &SourceInventory, registry: &Registry, graph: &OnnxGraph) -> FrameRevisions {
    let coords: Vec<&[u8]> = graph.props.iter().map(|p| p.as_bytes()).collect();
    FrameRevisions {
        source: digest64(&[src.model_sha256.as_bytes()]),
        relation_map: registry.version,
        coordinates: digest64(&coords),
    }
}

/// Bind every node of `graph` against `registry`, in schedule order.
pub fn resolve<'r>(
    src: &SourceInventory,
    graph: OnnxGraph,
    registry: &'r Registry,
    store: &FrameStore,
) -> Result<Plan<'r>, Refusal> {
    let opset = src.default_opset().unwrap_or(0);
    let order = graph.schedule()?;
    let mut steps = Vec::with_capacity(order.len());
    for &n in &order {
        let node = &src.nodes[n];
        let args: Vec<_> = graph.args[n]
            .iter()
            .map(|p| p.map(|p| &graph.facts[p]))
            .collect();
        let reg = registry
            .lookup(node, &args)
            .ok_or_else(|| Refusal::Unregistered {
                node: node.name.clone(),
                signature: render_signature(node, signature_dtype(node, &args, Some(0))),
            })?;
        if node.default_domain() && !reg.opsets.contains(&opset) {
            return Err(Refusal::Opset {
                node: node.name.clone(),
                opset,
            });
        }
        let consts = graph.args[n]
            .iter()
            .map(|p| p.and_then(|p| graph.consts[p].as_ref()))
            .collect();
        let truth = (reg.when)(&NodeView { node, args, consts });
        if truth != K3::True {
            return Err(Refusal::When {
                node: node.name.clone(),
                classid: reg.classid,
                truth,
                guard: reg.guard,
            });
        }
        let reg_ix = registry
            .regs
            .iter()
            .position(|r| std::ptr::eq(r, reg))
            .expect("from the table");
        steps.push(Bound {
            node: n,
            name: node.name.clone(),
            classid: reg.classid,
            method: reg.method,
            numeric: reg.numeric,
            reg: reg_ix,
        });
    }
    if store.revisions() != revisions(src, registry, &graph) {
        return Err(Refusal::Frame(
            dr_math_binding_probe::Refusal::SourceChanged,
        ));
    }
    Ok(Plan {
        graph,
        steps,
        registry,
        nodes: src.nodes.clone(),
        frame: store.seal(),
    })
}

impl Plan<'_> {
    /// Run the plan: bind the caller's inputs by name, route each step to its
    /// DO, return every value by name. Refuses first if the frame is stale.
    pub fn run(
        &self,
        store: &FrameStore,
        inputs: &[(&str, Value)],
    ) -> Result<HashMap<String, Value>, Refusal> {
        store.check(&self.frame).map_err(Refusal::Frame)?;
        let g = &self.graph;
        let mut vals: Vec<Option<Value>> = g.consts.clone();
        for (name, v) in inputs {
            let p = g
                .props
                .iter()
                .position(|p| p == name)
                .ok_or_else(|| Refusal::Runtime {
                    node: (*name).into(),
                    what: "no such graph input".into(),
                })?;
            vals[p] = Some(v.clone());
        }
        for s in &self.steps {
            let node = &self.nodes[s.node];
            let args: Vec<&Value> = g.args[s.node]
                .iter()
                .flatten()
                .map(|&p| {
                    vals[p].as_ref().ok_or_else(|| Refusal::Runtime {
                        node: node.name.clone(),
                        what: format!("value {} unbound", g.props[p]),
                    })
                })
                .collect::<Result<_, _>>()?;
            let out = (self.registry.regs[s.reg].do_)(&args, node)?;
            for (&p, v) in g.outs[s.node].iter().zip(out) {
                vals[p] = Some(v);
            }
        }
        Ok(g.props
            .iter()
            .cloned()
            .zip(vals)
            .filter_map(|(n, v)| v.map(|v| (n, v)))
            .collect())
    }

    /// `(classid, method)` per scheduled node, for receipts.
    pub fn census(&self) -> Vec<(ClassId, &'static str)> {
        self.steps.iter().map(|s| (s.classid, s.method)).collect()
    }
}

/// Registration coverage of a graph too large for one tile: per op type, how
/// many nodes have a registered signature with WHEN TRUE, a registered
/// signature with WHEN not TRUE, or no registration. Facts and constants are
/// looked up by value name, as [`OnnxGraph::build`] does.
pub fn coverage(
    src: &SourceInventory,
    facts: &HashMap<String, crate::registry::Fact>,
    registry: &Registry,
) -> std::collections::BTreeMap<String, [usize; 3]> {
    let mut consts: HashMap<&str, &Value> = src
        .initializers
        .iter()
        .filter_map(|i| i.value.as_ref().map(|v| (i.name.as_str(), v)))
        .collect();
    for n in src
        .nodes
        .iter()
        .filter(|n| n.op_type == "Constant" && n.default_domain())
    {
        if let (Some(crate::source::Attr::Tensor(Some(v))), Some(o)) =
            (n.attr("value"), n.outputs.first())
        {
            consts.insert(o.as_str(), v);
        }
    }
    let mut out = std::collections::BTreeMap::new();
    for node in &src.nodes {
        let args: Vec<_> = node.inputs.iter().map(|i| facts.get(i)).collect();
        let k = match registry.lookup(node, &args) {
            None => 2,
            Some(reg) => {
                let consts = node
                    .inputs
                    .iter()
                    .map(|i| consts.get(i.as_str()).copied())
                    .collect();
                usize::from((reg.when)(&NodeView { node, args, consts }) != K3::True)
            }
        };
        let key = if node.default_domain() {
            node.op_type.clone()
        } else {
            format!("{}:{}", node.domain, node.op_type)
        };
        out.entry(key).or_insert([0; 3])[k] += 1;
    }
    out
}
