//! Cold shape-subgraph folding (D-TRACT-SHAPE-1).
//!
//! An ONNX node is folded, i.e. executed once while the plan is built and its
//! outputs kept as constants, only when its result is **invariant under the
//! sealed source and the resolved input facts**:
//!
//! - every input is a constant value of the source (initializer, `Constant`,
//!   or an earlier fold), or
//! - it is `Shape` and its input's shape is fully known in the facts (no
//!   symbolic, no undetermined dimension). The plan refuses at run time any
//!   input whose dtype, rank or known dimensions differ from the facts
//!   (`plan::check_against_fact`), so a dimension folded here is the one the
//!   run sees.
//!
//! Everything else stays dynamic. A fold runs the node's registered DO (the
//! same immutable method the run would call), after its WHEN guard is TRUE.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::registry::{Dim, Fact, NodeView, Registry, K3};
use crate::source::SourceInventory;
use crate::value::Value;

/// What a value depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dep {
    /// Source constants only.
    Const,
    /// Constants and fully known input shapes.
    Shape,
    /// An input shape with a symbolic dimension.
    Symbolic,
    /// An input value, or a shape the facts do not determine.
    Data,
}

fn shape_dep(f: Option<&Fact>) -> Dep {
    match f.and_then(|f| f.shape.as_ref()) {
        Some(s) if s.iter().all(|d| matches!(d, Dim::Known(_))) => Dep::Shape,
        Some(_) => Dep::Symbolic,
        None => Dep::Data,
    }
}

/// The dependency class of every value of the graph (by value name), and of
/// every node (the strongest class among its inputs; a `Shape` node takes
/// the class of its input's *shape*).
pub fn classify(
    src: &SourceInventory,
    facts: &HashMap<String, Fact>,
) -> (HashMap<String, Dep>, Vec<Dep>) {
    let mut dep: HashMap<String, Dep> = HashMap::new();
    for i in &src.initializers {
        dep.insert(i.name.clone(), Dep::Const);
    }
    for i in &src.inputs {
        dep.insert(i.clone(), Dep::Data);
    }
    let mut node_dep = vec![None; src.nodes.len()];
    // Fixed point: the source order need not be topological.
    loop {
        let mut changed = false;
        for (n, node) in src.nodes.iter().enumerate() {
            if node_dep[n].is_some() {
                continue;
            }
            let ins: Vec<&String> = node.inputs.iter().filter(|s| !s.is_empty()).collect();
            let d = if node.op_type == "Shape" && node.default_domain() {
                match ins.first() {
                    Some(i) => Some(shape_dep(facts.get(i.as_str()))),
                    None => Some(Dep::Data),
                }
            } else {
                ins.iter()
                    .map(|i| dep.get(i.as_str()).copied())
                    .try_fold(Dep::Const, |a, d| d.map(|d| a.max(d)))
            };
            if let Some(d) = d {
                node_dep[n] = Some(d);
                for o in node.outputs.iter().filter(|o| !o.is_empty()) {
                    dep.insert(o.clone(), d);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // A node whose inputs never resolve (a cycle, or a dangling name) is Data.
    (
        dep,
        node_dep
            .into_iter()
            .map(|d| d.unwrap_or(Dep::Data))
            .collect(),
    )
}

/// Per op type, how many nodes fall in each class `[Const, Shape, Symbolic, Data]`.
pub fn census(
    src: &SourceInventory,
    node_dep: &[Dep],
    ops: &[&str],
) -> BTreeMap<String, [usize; 4]> {
    let mut out = BTreeMap::new();
    for (node, d) in src.nodes.iter().zip(node_dep) {
        if ops.contains(&node.op_type.as_str()) {
            out.entry(node.op_type.clone()).or_insert([0; 4])[*d as usize] += 1;
        }
    }
    out
}

/// The result of folding a graph.
#[derive(Debug, Default)]
pub struct Folded {
    /// Folded node indices.
    pub nodes: HashSet<usize>,
    /// Values produced by folded nodes.
    pub values: HashMap<String, Value>,
    /// Foldable by dependency (Const or Shape) but not folded, with the reason.
    pub blocked: Vec<(usize, String)>,
    /// Nodes whose DO refused on their constant inputs (an int64 overflow, a
    /// zero divisor): the refusal holds for every run, so resolution reports
    /// it instead of scheduling the node.
    pub refused: Vec<(usize, crate::registry::Refusal)>,
}

impl Folded {
    /// Folded values still read by a node that is not folded (or that are
    /// graph outputs): the constants the plan must retain.
    pub fn retained<'a>(&'a self, src: &'a SourceInventory) -> Vec<(&'a str, &'a Value)> {
        let live: HashSet<&str> = src
            .nodes
            .iter()
            .enumerate()
            .filter(|(n, _)| !self.nodes.contains(n))
            .flat_map(|(_, node)| node.inputs.iter().map(String::as_str))
            .chain(src.outputs.iter().map(String::as_str))
            .collect();
        let mut v: Vec<_> = self
            .values
            .iter()
            .filter(|(k, _)| live.contains(k.as_str()))
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }
}

/// The value `Shape` produces for a known shape, honouring `start`/`end`
/// (opset 15) with ONNX clamping. `None` if a dimension is not known.
pub fn shape_value(shape: &[Dim], node: &crate::source::SourceNode) -> Option<Value> {
    let dims: Vec<i64> = shape
        .iter()
        .map(|d| {
            if let Dim::Known(v) = d {
                Some(*v as i64)
            } else {
                None
            }
        })
        .collect::<Option<_>>()?;
    let out = crate::registry::shape_slice(&dims, node);
    Some(Value::I64(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[out.len()]), out).expect("1-D"),
    ))
}

/// `facts` plus the source's own declaration for every initializer that has
/// no admitted fact: its dtype and dims are stated by the protobuf and need
/// no frontend authority. The same rule `OnnxGraph::build` applies, so a
/// census over raw facts sees what a plan sees.
pub fn with_initializer_facts(
    src: &SourceInventory,
    facts: &HashMap<String, Fact>,
) -> HashMap<String, Fact> {
    let mut out = facts.clone();
    for i in &src.initializers {
        let e = out.entry(i.name.clone()).or_default();
        if e.dtype.is_none() {
            *e = Fact {
                dtype: Some(crate::value::DType::from_onnx(i.data_type)),
                shape: Some(i.dims.iter().map(|&d| Dim::Known(d as usize)).collect()),
            };
        }
    }
    out
}

/// Fold every node whose result is invariant under the source and the
/// facts. Uses the registry's own WHEN and DO; never a separate evaluator.
pub fn fold(src: &SourceInventory, facts: &HashMap<String, Fact>, registry: &Registry) -> Folded {
    let facts = &with_initializer_facts(src, facts);
    let mut known: HashMap<String, Value> = src
        .initializers
        .iter()
        .filter_map(|i| i.value.clone().map(|v| (i.name.clone(), v)))
        .collect();
    let mut out = Folded::default();
    let (_, node_dep) = classify(src, facts);
    loop {
        let mut changed = false;
        for (n, node) in src.nodes.iter().enumerate() {
            // Skips work only: a Symbolic or Data node never has every
            // input known, so it could not fold anyway (measured: removing
            // this test changes no result).
            if out.nodes.contains(&n)
                || node_dep[n] > Dep::Shape
                || out.refused.iter().any(|(r, _)| *r == n)
            {
                continue;
            }
            let args: Vec<Option<&Fact>> = node.inputs.iter().map(|i| facts.get(i)).collect();
            let Some(reg) = registry.lookup(node, &args) else {
                continue;
            };
            let consts: Vec<Option<&Value>> = node.inputs.iter().map(|i| known.get(i)).collect();
            if (reg.when)(&NodeView {
                node,
                args: args.clone(),
                consts: consts.clone(),
            }) != K3::True
            {
                continue;
            }
            let vals = if node.op_type == "Shape" && node.default_domain() {
                match node
                    .inputs
                    .first()
                    .and_then(|i| facts.get(i))
                    .and_then(|f| f.shape.as_deref())
                {
                    Some(s) => shape_value(s, node).map(|v| vec![v]),
                    None => None,
                }
            } else {
                let present: Vec<&Value> = node
                    .inputs
                    .iter()
                    .filter(|i| !i.is_empty())
                    .map(|i| known.get(i))
                    .collect::<Option<_>>()
                    .unwrap_or_default();
                if present.len() != node.inputs.iter().filter(|i| !i.is_empty()).count() {
                    None
                } else {
                    match (registry.method(reg).do_)(&present, node) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            out.refused.push((n, e));
                            None
                        }
                    }
                }
            };
            if let Some(vals) = vals {
                for (o, v) in node.outputs.iter().zip(vals) {
                    known.insert(o.clone(), v.clone());
                    out.values.insert(o.clone(), v);
                }
                out.nodes.insert(n);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (n, node) in src.nodes.iter().enumerate() {
        if out.nodes.contains(&n) || node_dep[n] > Dep::Shape || node.op_type == "Constant" {
            continue;
        }
        let args: Vec<Option<&Fact>> = node.inputs.iter().map(|i| facts.get(i)).collect();
        let why = match registry.lookup(node, &args) {
            None => format!("unregistered {}", node.op_type),
            Some(reg) => {
                let consts: Vec<Option<&Value>> =
                    node.inputs.iter().map(|i| known.get(i)).collect();
                let missing = node
                    .inputs
                    .iter()
                    .zip(&consts)
                    .any(|(i, c)| !i.is_empty() && c.is_none());
                if missing && !(node.op_type == "Shape" && node.default_domain()) {
                    format!("{}: an input is not folded upstream", node.op_type)
                } else {
                    let t = (reg.when)(&NodeView { node, args, consts });
                    format!("{}: WHEN {t:?}", node.op_type)
                }
            }
        };
        out.blocked.push((n, why));
    }
    out
}
