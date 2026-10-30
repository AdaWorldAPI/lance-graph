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
use tract_onnx::tract_hir::infer::Factoid;

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
    m: InferenceModel,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<HashMap<String, Fact>, Refusal> {
    Ok(fact_sources_of(m, pins)?.merged())
}

/// Facts from both Tract stages, kept apart so they can be compared.
///
/// - `inference`: the inference model after `analyse`. It keeps every
///   source value's label, but a fact may be partial; a shape is taken only
///   when its rank and every dimension are determined (an undetermined
///   dimension is not a symbol: two of them need not be equal).
/// - `typed`: the typed model, complete per outlet but labelling only the
///   values that survive typing (2,546 of 7,620 on the Jina graph).
#[derive(Debug, Default)]
pub struct FactSources {
    pub inference: HashMap<String, Fact>,
    pub typed: HashMap<String, Fact>,
    /// Why the typed stage failed, if it did. The inference facts stand on
    /// their own (Tract refuses to type a graph with an undetermined input
    /// dimension, or more than one symbol through `Flatten`).
    pub typed_error: Option<String>,
}

impl FactSources {
    /// Inference facts first; the typed fact fills a missing dtype or shape.
    pub fn merged(&self) -> HashMap<String, Fact> {
        let mut out = self.inference.clone();
        for (k, t) in &self.typed {
            let e = out.entry(k.clone()).or_default();
            if e.dtype.is_none() {
                e.dtype = t.dtype;
            }
            if e.shape.is_none() {
                e.shape = t.shape.clone();
            }
        }
        out
    }

    /// Values both stages describe, where they disagree on dtype or shape.
    pub fn disagreements(&self) -> Vec<(String, Fact, Fact)> {
        let mut out: Vec<_> = self
            .inference
            .iter()
            .filter_map(|(k, i)| {
                let t = self.typed.get(k)?;
                let dt = matches!((i.dtype, t.dtype), (Some(a), Some(b)) if a != b);
                let sh = matches!((&i.shape, &t.shape), (Some(a), Some(b)) if a != b);
                (dt || sh).then(|| (k.clone(), i.clone(), t.clone()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

/// Both fact sources for an in-memory model.
pub fn tract_fact_sources(
    bytes: &[u8],
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    let onnx = tract_onnx::onnx();
    let proto = onnx.proto_model_for_read(&mut &*bytes).map_err(fe)?;
    fact_sources_of(onnx.model_for_proto_model(&proto).map_err(fe)?, pins)
}

/// Both fact sources for a model with lazy external data.
pub fn tract_fact_sources_path(
    path: &std::path::Path,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    let onnx = tract_onnx::onnx().with_lazy_external_data(true);
    fact_sources_of(onnx.model_for_path(path).map_err(fe)?, pins)
}

fn label<F, O>(m: &Graph<F, O>, o: OutletId) -> String
where
    F: tract_onnx::prelude::Fact + Clone + 'static,
    O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    m.outlet_label(o)
        .map(str::to_string)
        .unwrap_or_else(|| m.node(o.node).name.clone())
}

fn dim_of(d: &TDim) -> Dim {
    d.as_i64()
        .map(|v| Dim::Known(v as usize))
        .unwrap_or_else(|| Dim::Symbol(d.to_string()))
}

fn fact_sources_of(
    mut m: InferenceModel,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    for (i, dt, shape) in pins {
        m.set_input_fact(*i, InferenceFact::dt_shape(*dt, shape.clone()))
            .map_err(fe)?;
    }
    let mut out = FactSources::default();

    // Inference stage: analyse without stopping at the first node it cannot
    // type, then read whatever each labelled outlet determined.
    let mut inf = m.clone();
    inf.analyse(true).map_err(fe)?;
    for node in inf.nodes() {
        for slot in 0..node.outputs.len() {
            let o = OutletId::new(node.id, slot);
            let f = inf.outlet_fact(o).map_err(fe)?;
            let shape = if f.shape.is_open() {
                None
            } else {
                f.shape
                    .dims()
                    .map(|d| d.concretize().map(|d| dim_of(&d)))
                    .collect()
            };
            let fact = Fact {
                dtype: f.datum_type.concretize().map(dtype_of),
                shape,
            };
            if fact.dtype.is_some() || fact.shape.is_some() {
                out.inference.insert(label(&inf, o), fact);
            }
        }
    }

    // Typed stage.
    let m = match m.into_typed() {
        Ok(m) => m,
        Err(e) => {
            out.typed_error = Some(format!("{e:#}"));
            return Ok(out);
        }
    };
    for node in m.nodes() {
        for slot in 0..node.outputs.len() {
            let o = OutletId::new(node.id, slot);
            let f = m.outlet_fact(o).map_err(fe)?;
            out.typed.insert(
                label(&m, o),
                Fact {
                    dtype: Some(dtype_of(f.datum_type)),
                    shape: Some(f.shape.iter().map(dim_of).collect()),
                },
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(dt: DType, shape: &[usize]) -> Fact {
        Fact {
            dtype: Some(dt),
            shape: Some(shape.iter().map(|&d| Dim::Known(d)).collect()),
        }
    }

    /// `disagreements` reports a dtype or shape conflict and stays silent on
    /// agreement or on a value only one stage describes; `merged` prefers the
    /// inference fact and fills only what it lacks from the typed one.
    #[test]
    fn fact_sources_disagree_and_merge() {
        let mut fs = FactSources::default();
        fs.inference
            .insert("same".into(), fact(DType::F32, &[2, 3]));
        fs.typed.insert("same".into(), fact(DType::F32, &[2, 3]));
        fs.inference
            .insert("shape".into(), fact(DType::F32, &[2, 3]));
        fs.typed.insert("shape".into(), fact(DType::F32, &[3, 2]));
        fs.inference.insert("dtype".into(), fact(DType::F32, &[1]));
        fs.typed.insert("dtype".into(), fact(DType::I32, &[1]));
        fs.inference
            .insert("only_inf".into(), fact(DType::I64, &[4]));
        fs.typed
            .insert("only_typed".into(), fact(DType::Bool, &[5]));
        fs.inference.insert(
            "partial".into(),
            Fact {
                dtype: Some(DType::F32),
                shape: None,
            },
        );
        fs.typed.insert("partial".into(), fact(DType::F32, &[7]));
        let d: Vec<String> = fs.disagreements().into_iter().map(|(k, ..)| k).collect();
        assert_eq!(d, ["dtype", "shape"]);
        let m = fs.merged();
        assert_eq!(m["shape"], fact(DType::F32, &[2, 3]), "inference wins");
        assert_eq!(m["only_typed"], fact(DType::Bool, &[5]));
        assert_eq!(m["only_inf"], fact(DType::I64, &[4]));
        assert_eq!(
            m["partial"],
            fact(DType::F32, &[7]),
            "typed fills the missing shape"
        );
    }
}
