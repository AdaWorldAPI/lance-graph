//! An ONNX graph as a DR#3 graph (user decision 2026-10-30): every node is an
//! **Identity** position, every tensor value (graph input, initializer, node
//! output) a **Property** position, and the dataflow is two relations:
//!
//! DR64 is global to local, not a size limit. A global position `p` lives in
//! tile `p / 64` at local bit `p % 64`; a relation is the set of its non-empty
//! 64 × 64 tiles ([`Tiled`]), and composition composes tile by tile with the
//! one DR64 kernel. The only ceiling is the address width: 65,536 positions
//! per axis.
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

/// Positions per axis a graph may address (the u16 address width).
pub const MAX_POSITIONS: usize = 1 << 16;

/// A Boolean relation over global positions, held as its non-empty 64 × 64
/// DR64 tiles keyed `(row tile, column tile)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tiled {
    pub tiles: std::collections::BTreeMap<(u16, u16), Rows>,
}

impl Tiled {
    pub fn set(&mut self, i: usize, j: usize) {
        let t = self
            .tiles
            .entry(((i / TILE) as u16, (j / TILE) as u16))
            .or_insert([0; 64]);
        t[i % TILE] |= 1 << (j % TILE);
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        self.tiles
            .get(&((i / TILE) as u16, (j / TILE) as u16))
            .is_some_and(|t| t[i % TILE] >> (j % TILE) & 1 == 1)
    }

    /// Every `(i, j)` in the relation, global positions.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.tiles.iter().flat_map(|(&(ti, tj), t)| {
            t.iter().enumerate().flat_map(move |(r, &row)| {
                let mut bits = row;
                std::iter::from_fn(move || {
                    (bits != 0).then(|| {
                        let c = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        (ti as usize * TILE + r, tj as usize * TILE + c)
                    })
                })
            })
        })
    }

    pub fn transpose(&self) -> Tiled {
        Tiled {
            tiles: self
                .tiles
                .iter()
                .map(|(&(ti, tj), t)| ((tj, ti), transpose(t)))
                .collect(),
        }
    }

    /// `out = self ∘ other`: tile `(ti, tj)` is the OR over `k` of the DR64
    /// composition of tile `(ti, k)` with tile `(k, tj)`.
    pub fn compose(&self, other: &Tiled, work: &mut Work) -> Tiled {
        let mut out = Tiled::default();
        for (&(ti, k), a) in &self.tiles {
            for (&(_, tj), b) in other.tiles.range((k, 0)..=(k, u16::MAX)) {
                let c = compose(a, b, work);
                if c.iter().any(|&r| r != 0) {
                    let o = out.tiles.entry((ti, tj)).or_insert([0; 64]);
                    for (x, y) in o.iter_mut().zip(c) {
                        *x |= y;
                    }
                }
            }
        }
        out
    }
}

/// The graph on its two axes.
pub struct OnnxGraph {
    /// Identity axis: node index in source order.
    pub nodes: usize,
    /// Property axis: value names, position = index.
    pub props: Vec<String>,
    pub on: Tiled,
    pub do_: Tiled,
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
    /// Lay the source graph onto DR64 tiles. `facts` is keyed by value name.
    pub fn build(src: &SourceInventory, facts: &HashMap<String, Fact>) -> Result<Self, Refusal> {
        if src.nodes.len() > MAX_POSITIONS {
            return Err(Refusal::AddressOverflow {
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
        if props.len() > MAX_POSITIONS {
            return Err(Refusal::AddressOverflow {
                axis: "Property",
                count: props.len(),
            });
        }

        let (mut on, mut do_) = (Tiled::default(), Tiled::default());
        for (n, (a, o)) in args.iter().zip(&outs).enumerate() {
            for &p in a.iter().flatten() {
                on.set(n, p);
            }
            for &p in o {
                do_.set(n, p);
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
    pub fn dep(&self) -> (Tiled, Work) {
        let mut w = Work::default();
        let d = self.do_.compose(&self.on.transpose(), &mut w);
        (d, w)
    }

    /// The same relation computed directly from the source edges, for the
    /// check that the composition is the dataflow and nothing else.
    pub fn dep_from_edges(&self) -> Tiled {
        let mut producer: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, o) in self.outs.iter().enumerate() {
            for &p in o {
                producer.entry(p).or_default().push(i);
            }
        }
        let mut d = Tiled::default();
        for (j, a) in self.args.iter().enumerate() {
            for p in a.iter().flatten() {
                for &i in producer.get(p).into_iter().flatten() {
                    d.set(i, j);
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
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); self.nodes];
        for (i, j) in dep.pairs() {
            indeg[j] += 1;
            succ[i].push(j);
        }
        let mut ready: std::collections::BTreeSet<usize> =
            (0..self.nodes).filter(|&n| indeg[n] == 0).collect();
        let mut order = Vec::with_capacity(self.nodes);
        while let Some(next) = ready.pop_first() {
            order.push(next);
            for &j in &succ[next] {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    ready.insert(j);
                }
            }
        }
        if order.len() < self.nodes {
            return Err(Refusal::Cycle);
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
        // Tract's ONNX `Shape` (onnx/src/ops/array/shape.rs) and `Size` emit
        // `TDim`, and ops over them keep it; the ONNX type of those values
        // is int64 (the spec's output type of `Shape`/`Size`).
        DatumType::TDim => DType::I64,
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
    Ok(tract_fact_sources(bytes, pins)?.merged())
}

/// [`tract_facts`] for a model with external data next to it: the weights
/// stay lazy (named, never read).
pub fn tract_facts_path(
    path: &std::path::Path,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<HashMap<String, Fact>, Refusal> {
    Ok(tract_fact_sources_path(path, pins)?.merged())
}

fn fe(e: TractError) -> Refusal {
    Refusal::Frontend(format!("{e:#}"))
}

/// A Tract outlet: node name and output slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Outlet {
    pub node: String,
    pub slot: usize,
}

/// A Tract fact not admitted for a source value, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejected {
    pub stage: &'static str,
    pub value: String,
    pub reason: String,
}

/// Facts from both Tract stages, kept apart so they can be compared.
///
/// - `inference`: the inference model after `analyse`. It keeps every
///   source value's label, but a fact may be partial; a shape is taken only
///   when its rank and every dimension are determined (an undetermined
///   dimension is not a symbol: two of them need not be equal).
/// - `typed`: the typed model, complete per outlet but labelling only the
///   values that survive typing (2,546 of 7,620 on the Jina graph).
///
/// A Tract fact is admitted for a source value only through [`admit`]: a
/// matching label alone is not authority. Rejections are kept in `rejected`.
#[derive(Debug, Default)]
pub struct FactSources {
    pub inference: HashMap<String, Fact>,
    pub typed: HashMap<String, Fact>,
    /// Why the typed stage failed, if it did. The inference facts stand on
    /// their own (Tract refuses to type a graph with an undetermined input
    /// dimension, or more than one symbol through `Flatten`).
    pub typed_error: Option<String>,
    pub rejected: Vec<Rejected>,
    /// The `model_sha256` of the source these facts were read from.
    pub source_sha256: String,
}

impl FactSources {
    /// Inference facts first; the typed fact fills a missing dtype or shape.
    /// Both maps hold only admitted facts, so this never merges two tensors
    /// that merely share a label.
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

    /// [`merged`](Self::merged), but only for the source the facts were read
    /// from: facts of another model (a stale binding) refuse.
    pub fn merged_for(&self, src: &SourceInventory) -> Result<HashMap<String, Fact>, Refusal> {
        if self.source_sha256 != src.model_sha256 {
            return Err(Refusal::Frontend(format!(
                "facts read from model {} applied to model {}",
                self.source_sha256, src.model_sha256
            )));
        }
        Ok(self.merged())
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

/// The Tract outlet each source value must come from, by Tract's own naming
/// rule (`onnx/src/model.rs`): graph inputs and initializers are nodes named
/// after the value; a node takes its source name, else its first output's
/// name. `None` when that identity is not unique (two source nodes with one
/// name) or not reproducible (an unnamed node whose first output is empty,
/// which Tract names by a running counter).
pub fn source_producers(src: &SourceInventory) -> HashMap<String, Option<Outlet>> {
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let node_name = |n: &crate::source::SourceNode| {
        if !n.name.is_empty() {
            Some(n.name.clone())
        } else {
            n.outputs.first().filter(|o| !o.is_empty()).cloned()
        }
    };
    for n in &src.nodes {
        if let Some(name) = node_name(n) {
            *by_name.entry(name).or_default() += 1;
        }
    }
    for v in src
        .inputs
        .iter()
        .chain(src.initializers.iter().map(|i| &i.name))
    {
        *by_name.entry(v.clone()).or_default() += 1;
    }
    let mut out: HashMap<String, Option<Outlet>> = HashMap::new();
    let mut put = |v: &str, o: Option<Outlet>| {
        let e = out.entry(v.to_string()).or_insert(o.clone());
        if *e != o {
            *e = None; // the value has two producers in the source
        }
    };
    for v in src
        .inputs
        .iter()
        .chain(src.initializers.iter().map(|i| &i.name))
    {
        put(
            v,
            (by_name[v] == 1).then(|| Outlet {
                node: v.clone(),
                slot: 0,
            }),
        );
    }
    for n in &src.nodes {
        let name = node_name(n);
        let unique = name.as_ref().is_some_and(|m| by_name[m] == 1);
        for (position, o) in n.outputs.iter().enumerate() {
            if o.is_empty() {
                continue; // an omitted optional output names no value
            }
            let slot = tract_slot(&n.outputs, position);
            put(
                o,
                name.clone()
                    .filter(|_| unique)
                    .map(|node| Outlet { node, slot }),
            );
        }
    }
    out
}

/// The Tract outlet slot of ONNX output `position`: Tract numbers a node's
/// outlets after dropping empty output names (`onnx/src/model.rs:150`,
/// `output.iter().filter(|s| !s.is_empty()).enumerate()`), so the slot is the
/// number of named outputs before `position`. The ONNX position itself is
/// not the slot whenever an omitted optional output precedes it.
pub fn tract_slot(outputs: &[String], position: usize) -> usize {
    outputs[..position].iter().filter(|o| !o.is_empty()).count()
}

/// Admit one stage's facts: a fact for source value `v` is kept only if
/// exactly one outlet of the stage carries `v`'s label and that outlet is
/// `v`'s producer (see [`source_producers`]). Labels that name no source
/// value are Tract-internal and ignored.
pub fn admit(
    stage: &'static str,
    raw: Vec<(String, Outlet, Fact)>,
    producers: &HashMap<String, Option<Outlet>>,
) -> (HashMap<String, Fact>, Vec<Rejected>) {
    let mut by_label: HashMap<String, Vec<(Outlet, Fact)>> = HashMap::new();
    for (l, o, f) in raw {
        by_label.entry(l).or_default().push((o, f));
    }
    let (mut ok, mut rej) = (HashMap::new(), Vec::new());
    let mut labels: Vec<_> = by_label.into_iter().collect();
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    for (value, mut outs) in labels {
        let Some(want) = producers.get(&value) else {
            continue;
        };
        let reason = match (outs.len(), want) {
            (n, _) if n > 1 => Some(format!("label carried by {n} outlets")),
            (_, None) => Some("source producer not unique".to_string()),
            (_, Some(w)) if outs[0].0 != *w => Some(format!(
                "Tract outlet {:?} is not the source producer {:?}",
                outs[0].0, w
            )),
            _ => None,
        };
        match reason {
            Some(reason) => rej.push(Rejected {
                stage,
                value,
                reason,
            }),
            None => {
                ok.insert(value, outs.pop().expect("one").1);
            }
        }
    }
    (ok, rej)
}

/// Both fact sources for an in-memory model.
pub fn tract_fact_sources(
    bytes: &[u8],
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    let src = SourceInventory::from_bytes(bytes).map_err(|e| Refusal::Frontend(e.to_string()))?;
    let onnx = tract_onnx::onnx();
    let proto = onnx.proto_model_for_read(&mut &*bytes).map_err(fe)?;
    fact_sources_of(&src, onnx.model_for_proto_model(&proto).map_err(fe)?, pins)
}

/// Both fact sources for a model with lazy external data.
pub fn tract_fact_sources_path(
    path: &std::path::Path,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    let bytes = std::fs::read(path).map_err(|e| Refusal::Frontend(e.to_string()))?;
    let src = SourceInventory::from_bytes(&bytes).map_err(|e| Refusal::Frontend(e.to_string()))?;
    let onnx = tract_onnx::onnx().with_lazy_external_data(true);
    fact_sources_of(&src, onnx.model_for_path(path).map_err(fe)?, pins)
}

fn outlet_of<F, O>(m: &Graph<F, O>, o: OutletId) -> (Option<String>, Outlet)
where
    F: tract_onnx::prelude::Fact + Clone + 'static,
    O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    (
        m.outlet_label(o).map(str::to_string),
        Outlet {
            node: m.node(o.node).name.clone(),
            slot: o.slot,
        },
    )
}

fn dim_of(d: &TDim) -> Dim {
    d.as_i64()
        .map(|v| Dim::Known(v as usize))
        .unwrap_or_else(|| Dim::Symbol(d.to_string()))
}

/// Tract panics on some invalid graphs: a constant `Gather` index outside the
/// gathered dimension indexes out of bounds in `tract-core`
/// (`ops/array/gather.rs`) during `analyse`. A panic of the cold frontend is a
/// frontend refusal, never a crash of resolution.
fn no_panic<T>(stage: &str, f: impl FnOnce() -> T) -> Result<T, Refusal> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| {
        let why = p
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| p.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        Refusal::Frontend(format!("Tract panicked during {stage}: {why}"))
    })
}

fn fact_sources_of(
    src: &SourceInventory,
    mut m: InferenceModel,
    pins: &[(usize, DatumType, Vec<usize>)],
) -> Result<FactSources, Refusal> {
    for (i, dt, shape) in pins {
        m.set_input_fact(*i, InferenceFact::dt_shape(*dt, shape.clone()))
            .map_err(fe)?;
    }
    let producers = source_producers(src);
    let mut out = FactSources {
        source_sha256: src.model_sha256.clone(),
        ..FactSources::default()
    };

    // Inference stage: analyse without stopping at the first node it cannot
    // type, then read whatever each labelled outlet determined. An unlabelled
    // outlet is identified by its node's name (Tract labels a single-output
    // node's outlet by the node only).
    let mut inf = m.clone();
    no_panic("analyse", || inf.analyse(true))?.map_err(fe)?;
    let mut raw = Vec::new();
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
                let (label, outlet) = outlet_of(&inf, o);
                raw.push((label.unwrap_or_else(|| outlet.node.clone()), outlet, fact));
            }
        }
    }
    let (ok, rej) = admit("inference", raw, &producers);
    out.inference = ok;
    out.rejected.extend(rej);

    // Typed stage.
    let m = match no_panic("typing", || m.into_typed()) {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            out.typed_error = Some(format!("{e:#}"));
            return Ok(out);
        }
        Err(Refusal::Frontend(e)) => {
            out.typed_error = Some(e);
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    let mut raw = Vec::new();
    for node in m.nodes() {
        for slot in 0..node.outputs.len() {
            let o = OutletId::new(node.id, slot);
            let f = m.outlet_fact(o).map_err(fe)?;
            let (label, outlet) = outlet_of(&m, o);
            raw.push((
                label.unwrap_or_else(|| outlet.node.clone()),
                outlet,
                Fact {
                    dtype: Some(dtype_of(f.datum_type)),
                    shape: Some(f.shape.iter().map(dim_of).collect()),
                },
            ));
        }
    }
    let (ok, rej) = admit("typed", raw, &producers);
    out.typed = ok;
    out.rejected.extend(rej);
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

    fn outlet(node: &str, slot: usize) -> Outlet {
        Outlet {
            node: node.into(),
            slot,
        }
    }

    /// A Tract fact is admitted for a source value only from the value's
    /// producer outlet, and only if one outlet carries the label: a matching
    /// label alone is not authority. Labels that name no source value are
    /// ignored.
    #[test]
    fn a_label_alone_does_not_admit_a_fact() {
        let producers: HashMap<String, Option<Outlet>> = [
            ("a", Some(outlet("n_a", 0))),
            ("b", Some(outlet("n_b", 1))),
            ("c", Some(outlet("n_c", 0))),
            ("d", None),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let x = fact(DType::F32, &[2]);
        let raw = vec![
            ("a".to_string(), outlet("n_a", 0), x.clone()),
            ("b".to_string(), outlet("n_b", 0), x.clone()),
            ("c".to_string(), outlet("n_c", 0), x.clone()),
            ("c".to_string(), outlet("n_c.1", 0), fact(DType::I64, &[3])),
            ("d".to_string(), outlet("n_d", 0), x.clone()),
            ("tract_internal".to_string(), outlet("x", 0), x.clone()),
        ];
        let (ok, rej) = admit("t", raw, &producers);
        assert_eq!(ok.keys().collect::<Vec<_>>(), [&"a".to_string()]);
        let why: Vec<(&str, &str)> = rej
            .iter()
            .map(|r| (r.value.as_str(), r.reason.as_str()))
            .collect();
        assert_eq!(why.len(), 3, "{why:?}");
        assert!(why[0].0 == "b" && why[0].1.contains("is not the source producer"));
        assert_eq!(why[1], ("c", "label carried by 2 outlets"));
        assert_eq!(why[2], ("d", "source producer not unique"));
    }

    /// DR64 is global to local: a 200-node chain (four Identity tiles, five
    /// Property tiles) builds, `dep` composed tile by tile equals the source
    /// edges including a cross-tile edge, and the schedule is the chain order.
    #[test]
    fn graphs_beyond_one_tile_lay_over_tiles() {
        use crate::source::SourceNode;
        const N: usize = 200;
        let nodes: Vec<SourceNode> = (0..N)
            .map(|i| SourceNode {
                name: format!("n{i}"),
                op_type: if i == N - 1 { "Add" } else { "Abs" }.into(),
                domain: String::new(),
                inputs: if i == 0 {
                    vec!["x".into()]
                } else if i == N - 1 {
                    vec![format!("v{}", i - 1), "v0".into()]
                } else {
                    vec![format!("v{}", i - 1)]
                },
                outputs: vec![format!("v{i}")],
                attributes: vec![],
                attrs: vec![],
            })
            .collect();
        let src = SourceInventory {
            nodes,
            inputs: vec!["x".into()],
            model_sha256: String::new(),
            inlined: vec![],
            ir_version: 8,
            opsets: vec![],
            initializers: vec![],
            outputs: vec![format!("v{}", N - 1)],
        };
        let g = OnnxGraph::build(&src, &HashMap::new()).unwrap();
        assert_eq!((g.nodes, g.props.len()), (N, N + 1));
        let (dep, work) = g.dep();
        assert_eq!(dep, g.dep_from_edges());
        assert!(dep.tiles.len() > 1 && work.row_ors > 0);
        assert_eq!(dep.pairs().count(), N);
        assert!(dep.contains(0, N - 1), "cross-tile edge v0 -> last");
        assert!(dep.contains(63, 64), "edge across a tile boundary");
        assert_eq!(g.schedule().unwrap(), (0..N).collect::<Vec<_>>());
    }

    /// Tract's naming: a named node, else its first output; a duplicated
    /// name or an unnamed node without a first output has no producer.
    #[test]
    fn source_producers_follow_tract_naming() {
        use crate::source::SourceNode;
        let n = |name: &str, outs: &[&str]| SourceNode {
            name: name.into(),
            op_type: "Abs".into(),
            domain: String::new(),
            inputs: vec!["x".into()],
            outputs: outs.iter().map(|s| s.to_string()).collect(),
            attributes: vec![],
            attrs: vec![],
        };
        let src = SourceInventory {
            nodes: vec![
                n("p", &["v1", "", "v2"]),
                n("", &["v3"]),
                n("dup", &["v4"]),
                n("dup", &["v5"]),
                n("", &["", "v6"]),
            ],
            inputs: vec!["x".into()],
            model_sha256: String::new(),
            inlined: vec![],
            ir_version: 8,
            opsets: vec![],
            initializers: vec![],
            outputs: vec![],
        };
        let p = source_producers(&src);
        assert_eq!(p["x"], Some(outlet("x", 0)));
        assert_eq!(p["v1"], Some(outlet("p", 0)));
        assert_eq!(p["v2"], Some(outlet("p", 1)), "empty outputs take no slot");
        assert_eq!(p["v3"], Some(outlet("v3", 0)));
        assert_eq!((p["v4"].clone(), p["v5"].clone()), (None, None));
        assert_eq!(p["v6"], None);
    }

    fn inv(
        nodes: Vec<crate::source::SourceNode>,
        inputs: &[&str],
        inits: &[&str],
    ) -> SourceInventory {
        SourceInventory {
            nodes,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            model_sha256: String::new(),
            inlined: vec![],
            ir_version: 8,
            opsets: vec![],
            initializers: inits
                .iter()
                .map(|n| crate::source::SourceInitializer {
                    name: n.to_string(),
                    data_type: 1,
                    dims: vec![],
                    placement: crate::source::Placement::Inline {
                        sha256: String::new(),
                    },
                    value: None,
                })
                .collect(),
            outputs: vec![],
        }
    }

    fn node(name: &str, outs: &[&str]) -> crate::source::SourceNode {
        crate::source::SourceNode {
            name: name.into(),
            op_type: "Split".into(),
            domain: String::new(),
            inputs: vec!["x".into()],
            outputs: outs.iter().map(|s| s.to_string()).collect(),
            attributes: vec![],
            attrs: vec![],
        }
    }

    /// The ONNX position of an output is enumerated first and then mapped to
    /// Tract's slot, which counts only the named outputs before it.
    #[test]
    fn tract_slot_counts_named_outputs_before_the_position() {
        let o = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(tract_slot(&o(&["", "Y"]), 1), 0);
        assert_eq!(tract_slot(&o(&["A", "", "B", "C"]), 2), 1);
        assert_eq!(tract_slot(&o(&["A", "", "B", "C"]), 3), 2);
        assert_eq!(tract_slot(&o(&["A", "B"]), 1), 1);
    }

    /// `["", "Y"]`: Y is slot 0. Several outputs with a gap: slots 0, 1, 2.
    #[test]
    fn producers_with_omitted_and_multiple_outputs() {
        let p = source_producers(&inv(
            vec![node("a", &["", "Y"]), node("b", &["P", "", "Q", "R"])],
            &["x"],
            &[],
        ));
        assert_eq!(p["Y"], Some(outlet("a", 0)));
        assert_eq!(
            (p["P"].clone(), p["Q"].clone(), p["R"].clone()),
            (
                Some(outlet("b", 0)),
                Some(outlet("b", 1)),
                Some(outlet("b", 2))
            )
        );
        assert!(!p.contains_key(""), "an omitted output names no value");
    }

    /// Two producers for one value, a value that is also a graph input or an
    /// initializer, two nodes with one name, and an unnamed node without a
    /// first output: no producer, so no fact is admitted for those values.
    #[test]
    fn producer_collisions_have_no_producer() {
        let p = source_producers(&inv(
            vec![
                node("a", &["v"]),
                node("b", &["v"]),
                node("c", &["x"]),
                node("d", &["w"]),
                node("dup", &["u1"]),
                node("dup", &["u2"]),
                node("", &["", "z"]),
                node("e", &["k"]),
            ],
            &["x"],
            &["k"],
        ));
        for v in ["v", "x", "u1", "u2", "z", "k"] {
            assert_eq!(p[v], None, "{v}");
        }
        assert_eq!(
            p["w"],
            Some(outlet("d", 0)),
            "a unique producer still counts"
        );
    }

    /// A fact bound to a producer of an older source: the same label on an
    /// outlet the current source no longer has is rejected.
    #[test]
    fn a_fact_from_a_renamed_producer_is_rejected() {
        let now = source_producers(&inv(vec![node("n_new", &["v"])], &["x"], &[]));
        let raw = vec![("v".to_string(), outlet("n_old", 0), fact(DType::F32, &[2]))];
        let (ok, rej) = admit("t", raw, &now);
        assert!(ok.is_empty());
        assert!(
            rej[0].reason.contains("is not the source producer"),
            "{}",
            rej[0].reason
        );
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
