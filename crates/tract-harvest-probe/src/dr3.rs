//! An ONNX graph as a DR#3 graph (user decision 2026-10-30): every node is an
//! **Identity** position, every tensor value (graph input, initializer, node
//! output) a **Property** position, and the dataflow is two relations over one
//! 64 × 64 tile:
//!
//! - `on[node]`: the Property positions the node reads (its ON);
//! - `do_[node]`: the Property positions the node writes (its DO).
//!
//! ON and DO share the Property domain, which is what makes chaining a
//! composition: `dep = do_ ∘ read_by` (`read_by` = the transpose of `on`) is
//! Identity → Identity, "node i writes a value node j reads". The schedule is
//! read off `dep`; no second dependency structure exists.
//!
//! The relations carry *which* positions; the **ordered** arguments (ONNX
//! operands are positional, `Sub(a, b) ≠ Sub(b, a)`) are kept beside them,
//! because a set loses order (D-DT64-6: a registration must reference ordered
//! behaviour, not a DO set).
//!
//! Facts (dtype, shape) come from Tract, used only as a cold typing frontend;
//! names, attributes and constant values come from the source protobuf.

use std::collections::HashMap;

use dr_math_binding_probe::relcomp::{compose, Rows, Work};
use tract_onnx::prelude::*;

use crate::registry::{Dim, Fact, Refusal};
use crate::source::{Attr, SourceInventory};
use crate::value::{DType, Value};

/// Positions per axis in one DR64 tile.
pub const TILE: usize = 64;

/// The graph on its two axes.
pub struct OnnxGraph {
    /// Identity axis: node index in source order.
    pub nodes: usize,
    /// Property axis: value names, position = index.
    pub props: Vec<String>,
    pub on: Rows,
    pub do_: Rows,
    /// Ordered arguments per node (`None`: an omitted optional input).
    pub args: Vec<Vec<Option<usize>>>,
    /// Ordered outputs per node.
    pub outs: Vec<Vec<usize>>,
    /// Facts per Property position.
    pub facts: Vec<Fact>,
    /// Values known before execution: initializers and `Constant` outputs.
    pub consts: Vec<Option<Value>>,
    /// Graph inputs (positions bound by the caller) and outputs.
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

fn transpose(r: &Rows) -> Rows {
    let mut t = [0u64; 64];
    for (i, &row) in r.iter().enumerate() {
        let mut bits = row;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            t[j] |= 1 << i;
            bits &= bits - 1;
        }
    }
    t
}

impl OnnxGraph {
    /// Lay the source graph onto the tile. `facts` is keyed by value name.
    pub fn build(src: &SourceInventory, facts: &HashMap<String, Fact>) -> Result<Self, Refusal> {
        if src.nodes.len() > TILE {
            return Err(Refusal::TileOverflow {
                axis: "Identity",
                count: src.nodes.len(),
            });
        }
        let mut props: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut intern = |name: &str, props: &mut Vec<String>| -> usize {
            *index.entry(name.to_string()).or_insert_with(|| {
                props.push(name.to_string());
                props.len() - 1
            })
        };
        for n in &src.inputs {
            intern(n, &mut props);
        }
        for i in &src.initializers {
            intern(&i.name, &mut props);
        }
        let mut args = Vec::with_capacity(src.nodes.len());
        let mut outs = Vec::with_capacity(src.nodes.len());
        for node in &src.nodes {
            args.push(
                node.inputs
                    .iter()
                    .map(|s| (!s.is_empty()).then(|| intern(s, &mut props)))
                    .collect::<Vec<_>>(),
            );
            outs.push(
                node.outputs
                    .iter()
                    .map(|s| intern(s, &mut props))
                    .collect::<Vec<_>>(),
            );
        }
        if props.len() > TILE {
            return Err(Refusal::TileOverflow {
                axis: "Property",
                count: props.len(),
            });
        }

        let (mut on, mut do_) = ([0u64; 64], [0u64; 64]);
        for (n, (a, o)) in args.iter().zip(&outs).enumerate() {
            for &p in a.iter().flatten() {
                on[n] |= 1 << p;
            }
            for &p in o {
                do_[n] |= 1 << p;
            }
        }

        let mut consts: Vec<Option<Value>> = vec![None; props.len()];
        let mut fact_of: Vec<Fact> = vec![Fact::default(); props.len()];
        for (p, name) in props.iter().enumerate() {
            if let Some(f) = facts.get(name) {
                fact_of[p] = f.clone();
            }
        }
        for i in &src.initializers {
            let p = index_of(&props, &i.name);
            consts[p] = i.value.clone();
            if fact_of[p].dtype.is_none() {
                fact_of[p] = Fact {
                    dtype: Some(DType::from_onnx(i.data_type)),
                    shape: Some(i.dims.iter().map(|&d| Dim::Known(d as usize)).collect()),
                };
            }
        }
        for (node, o) in src.nodes.iter().zip(&outs) {
            if node.op_type == "Constant" && node.default_domain() {
                if let (Some(Attr::Tensor(Some(v))), Some(&p)) = (node.attr("value"), o.first()) {
                    consts[p] = Some(v.clone());
                }
            }
        }
        let inputs = src.inputs.iter().map(|n| index_of(&props, n)).collect();
        let outputs = src
            .outputs
            .iter()
            .filter_map(|n| props.iter().position(|p| p == n))
            .collect();
        Ok(Self {
            nodes: src.nodes.len(),
            props,
            on,
            do_,
            args,
            outs,
            facts: fact_of,
            consts,
            inputs,
            outputs,
        })
    }

    /// `dep[i]` has bit `j` iff node `i` writes a value node `j` reads.
    pub fn dep(&self) -> (Rows, Work) {
        let mut w = Work::default();
        let d = compose(&self.do_, &transpose(&self.on), &mut w);
        (d, w)
    }

    /// The same relation computed directly from the source edges, for the
    /// check that the composition is the dataflow and nothing else.
    pub fn dep_from_edges(&self) -> Rows {
        let mut d = [0u64; 64];
        for (i, o) in self.outs.iter().enumerate() {
            for (j, a) in self.args.iter().enumerate() {
                if a.iter().flatten().any(|p| o.contains(p)) {
                    d[i] |= 1 << j;
                }
            }
        }
        d
    }

    /// A topological order of the Identity axis from `dep` (Kahn), stable in
    /// source order. `Err(Cycle)` if `dep` is not acyclic.
    pub fn schedule(&self) -> Result<Vec<usize>, Refusal> {
        let (dep, _) = self.dep();
        let mut indeg = vec![0usize; self.nodes];
        for row in dep.iter().take(self.nodes) {
            let mut bits = *row;
            while bits != 0 {
                indeg[bits.trailing_zeros() as usize] += 1;
                bits &= bits - 1;
            }
        }
        let mut done = 0u64;
        let mut order = Vec::with_capacity(self.nodes);
        while order.len() < self.nodes {
            let next = (0..self.nodes)
                .find(|&n| done & (1 << n) == 0 && indeg[n] == 0)
                .ok_or(Refusal::Cycle)?;
            done |= 1 << next;
            order.push(next);
            let mut bits = dep[next];
            while bits != 0 {
                indeg[bits.trailing_zeros() as usize] -= 1;
                bits &= bits - 1;
            }
        }
        Ok(order)
    }
}

fn index_of(props: &[String], name: &str) -> usize {
    props.iter().position(|p| p == name).expect("interned")
}

fn dtype_of(dt: DatumType) -> DType {
    match dt {
        DatumType::F32 => DType::F32,
        DatumType::F64 => DType::F64,
        DatumType::F16 => DType::F16,
        DatumType::I32 => DType::I32,
        DatumType::I64 => DType::I64,
        DatumType::Bool => DType::Bool,
        _ => DType::Other(-1),
    }
}

/// Tract as the cold typing frontend: import and type the graph (no
/// declutter, so every source value keeps its label) and read one [`Fact`]
/// per labelled outlet. `pins` fixes input shapes Tract cannot resolve alone.
pub fn tract_facts(
    bytes: &[u8],
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<HashMap<String, Fact>, Refusal> {
    let onnx = tract_onnx::onnx();
    let proto = onnx.proto_model_for_read(&mut &*bytes).map_err(fe)?;
    facts_of(onnx.model_for_proto_model(&proto).map_err(fe)?, pins)
}

/// [`tract_facts`] for a model with external data next to it: the weights
/// stay lazy (named, never read).
pub fn tract_facts_path(
    path: &std::path::Path,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<HashMap<String, Fact>, Refusal> {
    let onnx = tract_onnx::onnx().with_lazy_external_data(true);
    facts_of(onnx.model_for_path(path).map_err(fe)?, pins)
}

fn fe(e: TractError) -> Refusal {
    Refusal::Frontend(format!("{e:#}"))
}

fn facts_of(
    mut m: InferenceModel,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<HashMap<String, Fact>, Refusal> {
    for (i, dt, shape) in pins {
        m.set_input_fact(*i, InferenceFact::dt_shape(*dt, shape.clone()))
            .map_err(fe)?;
    }
    let m = m.into_typed().map_err(fe)?;
    let mut out = HashMap::new();
    for node in m.nodes() {
        for slot in 0..node.outputs.len() {
            let o = OutletId::new(node.id, slot);
            let label = m
                .outlet_label(o)
                .map(str::to_string)
                .unwrap_or_else(|| node.name.clone());
            let f = m.outlet_fact(o).map_err(fe)?;
            let shape = f
                .shape
                .iter()
                .map(|d| {
                    d.as_i64()
                        .map(|v| Dim::Known(v as usize))
                        .unwrap_or_else(|| Dim::Symbol(d.to_string()))
                })
                .collect();
            out.insert(
                label,
                Fact {
                    dtype: Some(dtype_of(f.datum_type)),
                    shape: Some(shape),
                },
            );
        }
    }
    Ok(out)
}
