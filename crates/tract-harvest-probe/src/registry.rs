//! The DT64 ON × WHEN × DO registration of ONNX operators (D-DT64-6 §1:
//! "Behaviour is registered as a DTO: classid → polyfill over mask-risc /
//! ndarray"; user decision 2026-10-30: one classid per op signature).
//!
//! An ONNX node is a registered method once its **op signature**
//! `(domain, op_type, dtype, discrete attributes)` matches a
//! [`Registration`]. The registration carries:
//!
//! - **ON**: the node's input positions (the DR#3 `on` relation, see `dr3`);
//! - **WHEN**: a typed three-valued guard over the node's facts (shapes,
//!   dtypes, constant operand values). UNKNOWN never counts as TRUE;
//! - **DO**: the polyfill, an existing `lance-graph-onnx` / `ndarray`
//!   function, writing the node's output positions.
//!
//! The classid's behaviour is immutable: a new behaviour is a new classid, never
//! an edit (D-DT64-6 D5). An unregistered signature is UNKNOWN (D-SAG-6 F06),
//! never "no behaviour": resolution refuses it.
//!
//! Classids here are **lab ordinals** (0x0F01…), not OGAR-minted concepts. A
//! production registration would mint the concept in OGAR and activate it
//! through `lance-graph-contract::hotplug`.

use std::ops::RangeInclusive;

use lance_graph_contract::class_view::ClassId;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::source::{Attr, SourceNode};
use crate::value::{DType, Value};

/// Three-valued truth of a WHEN guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum K3 {
    True,
    False,
    Unknown,
}

/// One dimension as Tract's typing knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Symbol(String),
}

/// What the cold frontend knows about one Property position.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Fact {
    pub dtype: Option<DType>,
    pub shape: Option<Vec<Dim>>,
}

/// A node as its guard sees it: the source node, the facts of its arguments
/// (in source order) and the values of its constant arguments.
pub struct NodeView<'a> {
    pub node: &'a SourceNode,
    pub args: Vec<Option<&'a Fact>>,
    pub consts: Vec<Option<&'a Value>>,
}

/// How exactly a DO reproduces the ONNX semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Numeric {
    /// Bit-identical to any IEEE-754 / exact-integer implementation.
    Exact,
    /// Depends on evaluation order; bounded by measurement, not proof.
    Bounded(&'static str),
}

/// Why a graph does not resolve or a plan does not run.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The cold frontend could not import or type the graph.
    Frontend(String),
    /// More than 64 positions on one axis: the graph needs several DR64 tiles.
    TileOverflow { axis: &'static str, count: usize },
    /// No registration for this op signature: UNKNOWN, never "no behaviour".
    Unregistered { node: String, signature: String },
    /// A registered signature outside the opsets its registration covers.
    Opset { node: String, opset: i64 },
    /// The WHEN guard is FALSE or UNKNOWN for this node.
    When {
        node: String,
        classid: ClassId,
        truth: K3,
        guard: &'static str,
    },
    /// The dataflow relation has a cycle (ONNX graphs are acyclic).
    Cycle,
    /// A native shape or range refusal from `lance-graph-onnx`.
    Native(lance_graph_onnx::Refusal),
    /// The plan's frame is not the store's current seal (stale generation).
    Frame(dr_math_binding_probe::Refusal),
    /// A DO received values its guard did not admit (a run-time fact).
    Runtime { node: String, what: String },
}

impl From<lance_graph_onnx::Refusal> for Refusal {
    fn from(r: lance_graph_onnx::Refusal) -> Self {
        Refusal::Native(r)
    }
}

/// A DO polyfill: argument values in source order, the source node for its
/// attributes; returns the output values in source order.
pub type Polyfill = fn(&[&Value], &SourceNode) -> Result<Vec<Value>, Refusal>;

/// A WHEN guard: truth and the guard's name.
pub type Guard = fn(&NodeView) -> K3;

/// A discrete integer attribute the signature admits.
///
/// - `value: Some(v)`: the attribute must equal `v` (or be absent with
///   `default == Some(v)`); a different value is a different signature.
/// - `value: None`: any integer, read by the DO as a parameter (an axis).
///   Absent is admitted only if `default` is `Some` (the ONNX default the DO
///   applies). The method's behaviour does not depend on which value: the
///   same immutable function reads it.
#[derive(Debug, Clone, Copy)]
pub struct AttrRule {
    pub name: &'static str,
    pub value: Option<i64>,
    pub default: Option<i64>,
}

/// One registered method.
pub struct Registration {
    pub classid: ClassId,
    pub method: &'static str,
    pub domain: &'static str,
    pub op_type: &'static str,
    /// Which argument's dtype keys the signature (`None`: the output's, for
    /// `Constant`).
    pub dtype_arg: Option<usize>,
    pub dtype: DType,
    /// Every discrete attribute the node may carry, with its required value.
    /// A node attribute not listed here does not match (an unknown attribute
    /// is not silently ignored).
    pub attrs: &'static [AttrRule],
    pub opsets: RangeInclusive<i64>,
    pub when: Guard,
    pub guard: &'static str,
    pub do_: Polyfill,
    pub numeric: Numeric,
}

/// The registration table, versioned. Its version is the frame's
/// `relation_map` revision: a different table is a different frame.
pub struct Registry {
    pub version: u64,
    pub regs: Vec<Registration>,
}

/// The dtype a node's signature is keyed by.
pub fn signature_dtype(
    node: &SourceNode,
    args: &[Option<&Fact>],
    dtype_arg: Option<usize>,
) -> Option<DType> {
    match dtype_arg {
        Some(i) => args.get(i).copied().flatten().and_then(|f| f.dtype),
        None => match node.attr("value") {
            Some(Attr::Tensor(Some(v))) => Some(v.dtype()),
            _ => None,
        },
    }
}

/// `domain:op_type(dtype){attrs}`, for reporting a signature.
pub fn render_signature(node: &SourceNode, dtype: Option<DType>) -> String {
    let domain = if node.domain.is_empty() {
        "ai.onnx"
    } else {
        &node.domain
    };
    let attrs: Vec<String> = node
        .attrs
        .iter()
        .map(|(n, a)| match a {
            Attr::Int(v) => format!("{n}={v}"),
            Attr::Ints(v) => format!("{n}={v:?}"),
            Attr::Str(v) => format!("{n}={v:?}"),
            Attr::Float(v) => format!("{n}={v:e}"),
            Attr::Floats(v) => format!("{n}=floats[{}]", v.len()),
            Attr::Tensor(Some(v)) => format!("{n}=tensor<{:?}>", v.dtype()),
            Attr::Tensor(None) => format!("{n}=tensor<?>"),
            Attr::Other(t) => format!("{n}=type{t}"),
        })
        .collect();
    format!(
        "{domain}:{}({dtype:?}){{{}}}",
        node.op_type,
        attrs.join(",")
    )
}

impl Registry {
    /// The registration whose signature this node has, if any.
    pub fn lookup(&self, node: &SourceNode, args: &[Option<&Fact>]) -> Option<&Registration> {
        let domain = if node.domain.is_empty() {
            "ai.onnx"
        } else {
            node.domain.as_str()
        };
        self.regs.iter().find(|r| {
            r.domain == domain
                && r.op_type == node.op_type
                && signature_dtype(node, args, r.dtype_arg) == Some(r.dtype)
                && attrs_match(node, r.attrs)
        })
    }
}

fn attrs_match(node: &SourceNode, rules: &[AttrRule]) -> bool {
    // Every node attribute is either the tensor `value` of a Constant or a
    // listed discrete attribute with the required value.
    let listed = node.attrs.iter().all(|(n, a)| {
        (n == "value" && matches!(a, Attr::Tensor(_)))
            || rules.iter().any(|r| {
                r.name == n && matches!(a, Attr::Int(v) if r.value.is_none_or(|w| *v == w))
            })
    });
    let defaults = rules.iter().all(|r| {
        node.attr(r.name).is_some()
            || match r.value {
                Some(w) => r.default == Some(w),
                None => r.default.is_some(),
            }
    });
    listed && defaults
}

// ── guards ──────────────────────────────────────────────────────────────────

fn known(shape: &Option<Vec<Dim>>) -> Option<Vec<usize>> {
    shape
        .as_ref()?
        .iter()
        .map(|d| {
            if let Dim::Known(v) = d {
                Some(*v)
            } else {
                None
            }
        })
        .collect()
}

/// Multidirectional broadcasting over Tract's facts. A symbolic dim agrees
/// with 1 or with the same symbol; anything else symbolic is UNKNOWN.
fn broadcast_k3(a: &[Dim], b: &[Dim]) -> K3 {
    let mut unknown = false;
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match (x, y) {
            (Dim::Known(p), Dim::Known(q)) if p == q || *p == 1 || *q == 1 => {}
            (Dim::Known(_), Dim::Known(_)) => return K3::False,
            (Dim::Symbol(s), Dim::Symbol(t)) if s == t => {}
            (Dim::Known(1), _) | (_, Dim::Known(1)) => {}
            _ => unknown = true,
        }
    }
    if unknown {
        K3::Unknown
    } else {
        K3::True
    }
}

fn guard_true(_: &NodeView) -> K3 {
    K3::True
}

fn guard_broadcast(v: &NodeView) -> K3 {
    match (
        v.args.first().copied().flatten(),
        v.args.get(1).copied().flatten(),
    ) {
        (Some(a), Some(b)) => match (&a.shape, &b.shape) {
            (Some(x), Some(y)) => broadcast_k3(x, y),
            _ => K3::Unknown,
        },
        _ => K3::Unknown,
    }
}

fn guard_cast_identity(v: &NodeView) -> K3 {
    // The signature already fixes `to = FLOAT` and the input dtype F32.
    match v.args.first().copied().flatten().and_then(|f| f.dtype) {
        Some(DType::F32) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

/// `Pow(x, e)` admitted as `Mul(x, x)` only when every exponent element is
/// exactly 2.0 AND broadcasting `e` against `x` leaves `x`'s shape unchanged
/// (the polyfill ignores `e`, so it cannot grow the output).
fn guard_pow2(v: &NodeView) -> K3 {
    let Some(e) = v.consts.get(1).copied().flatten() else {
        return K3::Unknown;
    };
    match e.f32s() {
        Some(x) if !x.is_empty() && x.iter().all(|x| x.to_bits() == 2.0f32.to_bits()) => {}
        _ => return K3::False,
    }
    let Some(base) = v
        .args
        .first()
        .copied()
        .flatten()
        .and_then(|f| f.shape.as_ref())
    else {
        return K3::Unknown;
    };
    exponent_keeps_shape(base, &e.shape())
}

/// Does broadcasting a tensor of `exp` shape against `base` leave `base`'s
/// shape unchanged?
fn exponent_keeps_shape(base: &[Dim], exp: &[usize]) -> K3 {
    if exp.len() > base.len() {
        return if exp.iter().take(exp.len() - base.len()).all(|&d| d == 1) {
            exponent_keeps_shape(base, &exp[exp.len() - base.len()..])
        } else {
            K3::False
        };
    }
    let mut unknown = false;
    for (b, &e) in base.iter().rev().zip(exp.iter().rev()) {
        match b {
            _ if e == 1 => {}
            Dim::Known(k) if *k == e => {}
            Dim::Known(_) => return K3::False,
            Dim::Symbol(_) => unknown = true,
        }
    }
    if unknown {
        K3::Unknown
    } else {
        K3::True
    }
}

fn guard_single_axis(v: &NodeView) -> K3 {
    match v.consts.get(1).copied().flatten().and_then(Value::i64s) {
        Some(a) if a.len() == 1 => K3::True,
        Some(_) => K3::False,
        None if v.node.inputs.len() < 2 => K3::False, // absent axes: all axes, not admitted here
        None => K3::Unknown,
    }
}

fn guard_rank1_sum(v: &NodeView) -> K3 {
    let rank1 = match v
        .args
        .first()
        .copied()
        .flatten()
        .and_then(|f| f.shape.as_ref())
    {
        Some(s) if s.len() == 1 => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    };
    if rank1 != K3::True {
        return rank1;
    }
    match v.node.inputs.get(1).filter(|s| !s.is_empty()) {
        None => K3::True,
        Some(_) => match v.consts.get(1).copied().flatten().and_then(Value::i64s) {
            Some(a) if a == [0] || a == [-1] => K3::True,
            Some(_) => K3::False,
            None => K3::Unknown,
        },
    }
}

fn guard_scalar_rhs(v: &NodeView) -> K3 {
    match v
        .args
        .get(1)
        .copied()
        .flatten()
        .and_then(|f| known(&f.shape))
    {
        Some(s) if s.iter().product::<usize>() == 1 => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

fn guard_where(v: &NodeView) -> K3 {
    let shape = |i: usize| {
        v.args
            .get(i)
            .copied()
            .flatten()
            .and_then(|f| f.shape.clone())
    };
    match (shape(0), shape(1), shape(2)) {
        (Some(c), Some(x), Some(y)) => {
            let same_cx = if c == x { K3::True } else { K3::Unknown };
            let y_ok = match known(&Some(y.clone())) {
                Some(s) if s.iter().product::<usize>() == 1 => K3::True,
                _ if y == x => K3::True,
                Some(_) => K3::False,
                None => K3::Unknown,
            };
            match (same_cx, y_ok) {
                (_, K3::False) => K3::False,
                (K3::True, K3::True) => K3::True,
                _ => K3::Unknown,
            }
        }
        _ => K3::Unknown,
    }
}

fn int_attr(n: &SourceNode, name: &str, default: i64) -> i64 {
    match n.attr(name) {
        Some(Attr::Int(v)) => *v,
        _ => default,
    }
}

/// `axis` normalised against `rank`, ONNX range `[-rank, rank)`.
fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    (-r..r).contains(&axis).then(|| axis.rem_euclid(r) as usize)
}

fn arg_shape<'a>(v: &'a NodeView, i: usize) -> Option<&'a Vec<Dim>> {
    v.args
        .get(i)
        .copied()
        .flatten()
        .and_then(|f| f.shape.as_ref())
}

/// `Gather` over shape data: the indices must be a constant (their validity
/// is otherwise a run-time fact) inside `[-d, d)` of the gathered dimension.
fn guard_gather(v: &NodeView) -> K3 {
    let Some(data) = arg_shape(v, 0) else {
        return K3::Unknown;
    };
    let Some(axis) = norm_axis(int_attr(v.node, "axis", 0), data.len()) else {
        return K3::False;
    };
    match v.args.get(1).copied().flatten().and_then(|f| f.dtype) {
        Some(DType::I64 | DType::I32) => {}
        Some(_) => return K3::False,
        None => return K3::Unknown,
    }
    let Some(idx) = v.consts.get(1).copied().flatten().and_then(Value::i64s) else {
        return K3::Unknown;
    };
    match data[axis] {
        Dim::Known(d) => {
            let d = d as i64;
            if idx.iter().all(|i| (-d..d).contains(i)) {
                K3::True
            } else {
                K3::False
            }
        }
        Dim::Symbol(_) => K3::Unknown,
    }
}

/// `Unsqueeze` (opset 13: `axes` is an input): constant, unique axes within
/// the output rank.
fn guard_unsqueeze(v: &NodeView) -> K3 {
    if v.node.inputs.get(1).is_none_or(|s| s.is_empty()) {
        return K3::False;
    }
    let Some(axes) = v.consts.get(1).copied().flatten().and_then(Value::i64s) else {
        return K3::Unknown;
    };
    let Some(data) = arg_shape(v, 0) else {
        return K3::Unknown;
    };
    match unsqueeze_shape(data.len(), &axes) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

/// The positions of the inserted axes in the output, sorted, or `None` if
/// an axis is out of range or repeated.
fn unsqueeze_shape(rank: usize, axes: &[i64]) -> Option<Vec<usize>> {
    let out = rank + axes.len();
    let mut a: Vec<usize> = axes
        .iter()
        .map(|&x| norm_axis(x, out))
        .collect::<Option<_>>()?;
    a.sort_unstable();
    a.windows(2).all(|w| w[0] != w[1]).then_some(a)
}

/// `Concat`: every input of one rank and one dtype, equal in every dimension
/// but `axis`.
fn guard_concat(v: &NodeView) -> K3 {
    let facts: Vec<&Fact> = match v.args.iter().copied().collect::<Option<Vec<_>>>() {
        Some(f) if !f.is_empty() => f,
        _ => return K3::Unknown,
    };
    let dt = facts[0].dtype;
    if facts.iter().any(|f| f.dtype.is_none()) {
        return K3::Unknown;
    }
    if facts.iter().any(|f| f.dtype != dt) {
        return K3::False;
    }
    let Some(shapes) = facts
        .iter()
        .map(|f| f.shape.as_ref())
        .collect::<Option<Vec<_>>>()
    else {
        return K3::Unknown;
    };
    let rank = shapes[0].len();
    if shapes.iter().any(|s| s.len() != rank) {
        return K3::False;
    }
    let Some(axis) = norm_axis(int_attr(v.node, "axis", i64::MIN), rank) else {
        return K3::False;
    };
    let mut unknown = false;
    for d in (0..rank).filter(|&d| d != axis) {
        for s in &shapes[1..] {
            match (&shapes[0][d], &s[d]) {
                (Dim::Known(a), Dim::Known(b)) if a != b => return K3::False,
                (Dim::Known(_), Dim::Known(_)) => {}
                (Dim::Symbol(a), Dim::Symbol(b)) if a == b => {}
                _ => unknown = true,
            }
        }
    }
    if unknown {
        K3::Unknown
    } else {
        K3::True
    }
}

/// The output shape of `Reshape(data, target)` with `allowzero = 0`: `0`
/// copies the input dimension, one `-1` is inferred. `None` if the target
/// is invalid for this input.
fn reshape_target(input: &[usize], target: &[i64]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(target.len());
    let mut infer = None;
    for (i, &t) in target.iter().enumerate() {
        match t {
            0 => out.push(*input.get(i)?),
            -1 if infer.is_none() => {
                infer = Some(i);
                out.push(1);
            }
            t if t > 0 => out.push(t as usize),
            _ => return None,
        }
    }
    let total: usize = input.iter().product();
    let known: usize = out.iter().product();
    match infer {
        Some(i) if known != 0 && total.is_multiple_of(known) => out[i] = total / known,
        Some(_) => return None,
        None if known != total => return None,
        None => {}
    }
    Some(out)
}

/// `Reshape` with a constant target over a fully known input shape.
fn guard_reshape(v: &NodeView) -> K3 {
    let Some(target) = v.consts.get(1).copied().flatten().and_then(Value::i64s) else {
        return K3::Unknown;
    };
    let Some(input) = v
        .args
        .first()
        .copied()
        .flatten()
        .and_then(|f| known(&f.shape))
    else {
        return K3::Unknown;
    };
    match reshape_target(&input, &target) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

// ── DO polyfills: existing native functions only ───────────────────────────

fn f32_arg<'a>(
    args: &[&'a Value],
    i: usize,
    node: &SourceNode,
) -> Result<&'a ArrayD<f32>, Refusal> {
    match args.get(i) {
        Some(Value::F32(a)) => Ok(a),
        other => Err(Refusal::Runtime {
            node: node.name.clone(),
            what: format!("arg {i}: expected f32, got {:?}", other.map(|v| v.dtype())),
        }),
    }
}

fn i32_arg<'a>(
    args: &[&'a Value],
    i: usize,
    node: &SourceNode,
) -> Result<&'a ArrayD<i32>, Refusal> {
    match args.get(i) {
        Some(Value::I32(a)) => Ok(a),
        other => Err(Refusal::Runtime {
            node: node.name.clone(),
            what: format!("arg {i}: expected i32, got {:?}", other.map(|v| v.dtype())),
        }),
    }
}

fn do_constant(_: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    match n.attr("value") {
        Some(Attr::Tensor(Some(v))) => Ok(vec![v.clone()]),
        _ => Err(Refusal::Runtime {
            node: n.name.clone(),
            what: "Constant without a decodable value".into(),
        }),
    }
}

fn do_identity(a: &[&Value], _: &SourceNode) -> Result<Vec<Value>, Refusal> {
    Ok(vec![a[0].clone()])
}

macro_rules! binary_do {
    ($name:ident, $f:path) => {
        fn $name(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
            Ok(vec![Value::F32($f(f32_arg(a, 0, n)?, f32_arg(a, 1, n)?)?)])
        }
    };
}
binary_do!(do_add, lance_graph_onnx::add);
binary_do!(do_sub, lance_graph_onnx::sub);
binary_do!(do_mul, lance_graph_onnx::mul);
binary_do!(do_div, lance_graph_onnx::div);

fn do_abs(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let mut out = ArrayD::<f32>::zeros(x.raw_dim());
    lance_graph_onnx::abs_f32(x.view(), out.view_mut());
    Ok(vec![Value::F32(out)])
}

fn do_sqrt(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let mut out = ArrayD::<f32>::zeros(x.raw_dim());
    lance_graph_onnx::sqrt_f32(x.view(), out.view_mut());
    Ok(vec![Value::F32(out)])
}

/// `Pow(x, 2)` admitted only with the exponent exactly 2.0 (its WHEN), so it
/// is one correctly rounded multiply, `Mul(x, x)`.
fn do_pow2(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    Ok(vec![Value::F32(lance_graph_onnx::mul(x, x)?)])
}

fn axis_of(a: &[&Value], n: &SourceNode, rank: usize) -> Result<usize, Refusal> {
    let ax = a
        .get(1)
        .and_then(|v| v.i64s())
        .and_then(|v| v.first().copied());
    let ax = ax.ok_or_else(|| Refusal::Runtime {
        node: n.name.clone(),
        what: "no axis".into(),
    })?;
    let r = rank as i64;
    if (-r..r).contains(&ax) {
        Ok(ax.rem_euclid(r) as usize)
    } else {
        Err(Refusal::Runtime {
            node: n.name.clone(),
            what: format!("axis {ax} out of rank {rank}"),
        })
    }
}

/// `ReduceMean` (keepdims 1) over one axis: `ndarray`'s `mean_axis`. The
/// float sum reassociates, so the method is Bounded.
fn do_reduce_mean(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let ax = Axis(axis_of(a, n, x.ndim())?);
    let m = x.mean_axis(ax).ok_or_else(|| Refusal::Runtime {
        node: n.name.clone(),
        what: "empty axis".into(),
    })?;
    Ok(vec![Value::F32(m.insert_axis(ax))])
}

fn reduce_sum_i32(a: &[&Value], n: &SourceNode, keepdims: i64) -> Result<Vec<Value>, Refusal> {
    let x = i32_arg(a, 0, n)?;
    let axes = a.get(1).and_then(|v| v.i64s());
    let r = lance_graph_onnx::ReduceSum::new(axes.as_deref(), keepdims, 0)?;
    let xs: Vec<i32> = x.iter().copied().collect();
    let s = r.sum_i32(&xs)?;
    let shape: &[usize] = if s.keepdims { &[1] } else { &[] };
    Ok(vec![Value::I32(
        ArrayD::from_shape_vec(IxDyn(shape), vec![s.value]).expect("one element"),
    )])
}

fn do_reduce_sum_i32_k0(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    reduce_sum_i32(a, n, 0)
}

fn do_reduce_sum_i32_k1(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    reduce_sum_i32(a, n, 1)
}

/// `Greater(X, t)`, scalar `t`: `ndarray::simd::gt_i32_to_mask` into a packed
/// `tensor(bool)`.
fn do_greater_i32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = i32_arg(a, 0, n)?;
    let t = i32_arg(a, 1, n)?
        .iter()
        .copied()
        .next()
        .ok_or_else(|| Refusal::Runtime {
            node: n.name.clone(),
            what: "empty threshold".into(),
        })?;
    let xs: Vec<i32> = x.iter().copied().collect();
    let mut words = vec![0u64; lance_graph_onnx::mask_words(xs.len())];
    lance_graph_onnx::greater_i32(&xs, t, &mut words);
    Ok(vec![Value::Bool {
        shape: x.shape().to_vec(),
        words,
    }])
}

/// `Where(C, X, Y)`, int32, `Y` of `X`'s shape: `blend_i32`; `Y` scalar:
/// `blend_i32_scalar` (ndarray #399), which reads the scalar directly.
fn do_where_i32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let Some(Value::Bool { shape, words }) = a.first() else {
        return Err(Refusal::Runtime {
            node: n.name.clone(),
            what: "condition is not a bool tensor".into(),
        });
    };
    let x = i32_arg(a, 1, n)?;
    let y = i32_arg(a, 2, n)?;
    if shape.as_slice() != x.shape() {
        return Err(Refusal::Runtime {
            node: n.name.clone(),
            what: "condition and X shapes differ".into(),
        });
    }
    let xs: Vec<i32> = x.iter().copied().collect();
    let mut out = vec![0i32; xs.len()];
    if let (1, Some(&yv)) = (y.len(), y.iter().next()) {
        // Scalar Y: the mask picks X, the rest is the scalar; no Y vector.
        ndarray::simd::blend_i32_scalar(words, &xs, yv, &mut out);
    } else {
        let ys: Vec<i32> = y.iter().copied().collect();
        if ys.len() != xs.len() {
            return Err(Refusal::Runtime {
                node: n.name.clone(),
                what: "Y does not broadcast to X".into(),
            });
        }
        lance_graph_onnx::where_i32(words, &xs, &ys, &mut out);
    }
    Ok(vec![Value::I32(
        ArrayD::from_shape_vec(IxDyn(x.shape()), out).expect("same length"),
    )])
}

fn i64_arg<'a>(
    args: &[&'a Value],
    i: usize,
    node: &SourceNode,
) -> Result<&'a ArrayD<i64>, Refusal> {
    match args.get(i) {
        Some(Value::I64(a)) => Ok(a),
        other => Err(Refusal::Runtime {
            node: node.name.clone(),
            what: format!("arg {i}: expected i64, got {:?}", other.map(|v| v.dtype())),
        }),
    }
}

fn runtime(n: &SourceNode, what: impl Into<String>) -> Refusal {
    Refusal::Runtime {
        node: n.name.clone(),
        what: what.into(),
    }
}

/// `Shape`'s `start`/`end` (opset 15) applied to `dims`, with ONNX clamping.
pub fn shape_slice(dims: &[i64], node: &SourceNode) -> Vec<i64> {
    let r = dims.len() as i64;
    let clamp = |v: i64| (if v < 0 { v + r } else { v }).clamp(0, r) as usize;
    let (s, e) = (
        clamp(int_attr(node, "start", 0)),
        clamp(int_attr(node, "end", r)),
    );
    if s < e {
        dims[s..e].to_vec()
    } else {
        vec![]
    }
}

fn i64_vector(v: Vec<i64>) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).expect("1-D"))
}

/// `Shape`: the value's shape (no element is read).
fn do_shape(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let dims: Vec<i64> = x.shape().iter().map(|&d| d as i64).collect();
    Ok(vec![i64_vector(shape_slice(&dims, n))])
}

/// The same elements in a new shape: an `ndarray` reshape of the standard
/// layout, or the packed mask with its logical shape replaced.
fn reshape_value(v: &Value, shape: &[usize], n: &SourceNode) -> Result<Value, Refusal> {
    fn re<T: Clone>(a: &ArrayD<T>, s: &[usize]) -> Option<ArrayD<T>> {
        a.as_standard_layout()
            .into_owned()
            .into_shape_with_order(IxDyn(s))
            .ok()
    }
    let bad = || runtime(n, format!("cannot reshape {:?} to {shape:?}", v.shape()));
    Ok(match v {
        Value::F32(a) => Value::F32(re(a, shape).ok_or_else(bad)?),
        Value::I32(a) => Value::I32(re(a, shape).ok_or_else(bad)?),
        Value::I64(a) => Value::I64(re(a, shape).ok_or_else(bad)?),
        Value::Bool { shape: s, words } => {
            if s.iter().product::<usize>() != shape.iter().product::<usize>() {
                return Err(bad());
            }
            Value::Bool {
                shape: shape.to_vec(),
                words: words.clone(),
            }
        }
    })
}

/// `Gather` over `int64` data: `ndarray`'s `select` along `axis`, then the
/// index tensor's shape in place of that axis. An index outside `[-d, d)`
/// refuses.
fn do_gather_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = i64_arg(a, 0, n)?;
    let idx_v = a.get(1).ok_or_else(|| runtime(n, "no indices"))?;
    let idx = idx_v
        .i64s()
        .ok_or_else(|| runtime(n, "indices are not integers"))?;
    let axis = norm_axis(int_attr(n, "axis", 0), x.ndim())
        .ok_or_else(|| runtime(n, "axis out of range"))?;
    let d = x.shape()[axis] as i64;
    let ix: Vec<usize> = idx
        .iter()
        .map(|&i| {
            (-d..d)
                .contains(&i)
                .then(|| i.rem_euclid(d) as usize)
                .ok_or_else(|| runtime(n, format!("index {i} outside [-{d}, {d})")))
        })
        .collect::<Result<_, _>>()?;
    let sel = x.select(Axis(axis), &ix);
    let mut shape = x.shape()[..axis].to_vec();
    shape.extend(idx_v.shape());
    shape.extend(&x.shape()[axis + 1..]);
    Ok(vec![reshape_value(&Value::I64(sel), &shape, n)?])
}

fn do_unsqueeze(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let axes = a
        .get(1)
        .and_then(|v| v.i64s())
        .ok_or_else(|| runtime(n, "no axes"))?;
    let at = unsqueeze_shape(x.shape().len(), &axes)
        .ok_or_else(|| runtime(n, format!("invalid axes {axes:?}")))?;
    let mut shape = x.shape();
    for p in at {
        shape.insert(p, 1);
    }
    Ok(vec![reshape_value(x, &shape, n)?])
}

fn do_reshape(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let target = a
        .get(1)
        .and_then(|v| v.i64s())
        .ok_or_else(|| runtime(n, "no target shape"))?;
    let shape = reshape_target(&x.shape(), &target)
        .ok_or_else(|| runtime(n, format!("target {target:?} invalid for {:?}", x.shape())))?;
    Ok(vec![reshape_value(x, &shape, n)?])
}

/// `Concat` along `axis`: `ndarray::concatenate` over the input views.
fn do_concat(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    fn cat<T: Clone>(xs: Vec<&ArrayD<T>>, ax: usize, n: &SourceNode) -> Result<ArrayD<T>, Refusal> {
        let views: Vec<_> = xs.iter().map(|x| x.view()).collect();
        ndarray::concatenate(Axis(ax), &views).map_err(|e| runtime(n, e.to_string()))
    }
    let rank = a
        .first()
        .ok_or_else(|| runtime(n, "no input"))?
        .shape()
        .len();
    let ax = norm_axis(int_attr(n, "axis", i64::MIN), rank)
        .ok_or_else(|| runtime(n, "axis out of range"))?;
    let out = match a[0] {
        Value::I64(_) => Value::I64(cat(
            (0..a.len())
                .map(|i| i64_arg(a, i, n))
                .collect::<Result<_, _>>()?,
            ax,
            n,
        )?),
        Value::F32(_) => Value::F32(cat(
            (0..a.len())
                .map(|i| f32_arg(a, i, n))
                .collect::<Result<_, _>>()?,
            ax,
            n,
        )?),
        other => return Err(runtime(n, format!("concat of {:?}", other.dtype()))),
    };
    Ok(vec![out])
}

/// `Erf`: `ndarray::hpc::vml::vserf`, at most 2 ULP over `[-5, 5]`.
fn do_erf(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let mut out = ArrayD::<f32>::zeros(x.raw_dim());
    ndarray::hpc::vml::vserf(x.view(), out.view_mut());
    Ok(vec![Value::F32(out)])
}

// ── the table ───────────────────────────────────────────────────────────────

const OPSETS: RangeInclusive<i64> = 13..=18;
const NO_ATTRS: &[AttrRule] = &[];
const CAST_TO_F32: &[AttrRule] = &[AttrRule {
    name: "to",
    value: Some(1),
    default: None,
}];
const CAST_TO_I64: &[AttrRule] = &[AttrRule {
    name: "to",
    value: Some(7),
    default: None,
}];
const KEEPDIMS_1: &[AttrRule] = &[AttrRule {
    name: "keepdims",
    value: Some(1),
    default: Some(1),
}];
const KEEPDIMS_0: &[AttrRule] = &[AttrRule {
    name: "keepdims",
    value: Some(0),
    default: None,
}];
const GATHER_AXIS: &[AttrRule] = &[AttrRule {
    name: "axis",
    value: None,
    default: Some(0),
}];
const CONCAT_AXIS: &[AttrRule] = &[AttrRule {
    name: "axis",
    value: None,
    default: None,
}];
/// `end` defaults to the rank; `i64::MAX` clamps to it.
const SHAPE_RANGE: &[AttrRule] = &[
    AttrRule {
        name: "start",
        value: None,
        default: Some(0),
    },
    AttrRule {
        name: "end",
        value: None,
        default: Some(i64::MAX),
    },
];
const ALLOWZERO_0: &[AttrRule] = &[AttrRule {
    name: "allowzero",
    value: Some(0),
    default: Some(0),
}];

macro_rules! reg {
    ($cid:expr, $m:expr, $op:expr, $arg:expr, $dt:expr, $attrs:expr, $when:path, $do:path, $num:expr) => {
        Registration {
            classid: $cid,
            method: $m,
            domain: "ai.onnx",
            op_type: $op,
            dtype_arg: $arg,
            dtype: $dt,
            attrs: $attrs,
            opsets: OPSETS,
            when: $when,
            guard: stringify!($when),
            do_: $do,
            numeric: $num,
        }
    };
}

/// The lab registry: every op the fixtures use, nothing else.
pub fn registry() -> Registry {
    use DType::*;
    use Numeric::*;
    Registry {
        version: 2,
        regs: vec![
            reg!(
                0x0F01,
                "Constant: the attribute tensor",
                "Constant",
                None,
                F32,
                NO_ATTRS,
                guard_true,
                do_constant,
                Exact
            ),
            reg!(
                0x0F02,
                "Constant: the attribute tensor",
                "Constant",
                None,
                I64,
                NO_ATTRS,
                guard_true,
                do_constant,
                Exact
            ),
            reg!(
                0x0F03,
                "Constant: the attribute tensor",
                "Constant",
                None,
                I32,
                NO_ATTRS,
                guard_true,
                do_constant,
                Exact
            ),
            reg!(
                0x0F10,
                "Cast f32 -> f32: identity",
                "Cast",
                Some(0),
                F32,
                CAST_TO_F32,
                guard_cast_identity,
                do_identity,
                Exact
            ),
            // The signature fixes input dtype I64 and `to = INT64`.
            reg!(
                0x0F11,
                "Cast i64 -> i64: identity",
                "Cast",
                Some(0),
                I64,
                CAST_TO_I64,
                guard_true,
                do_identity,
                Exact
            ),
            reg!(
                0x0F20,
                "lance_graph_onnx::add (broadcast checked)",
                "Add",
                Some(0),
                F32,
                NO_ATTRS,
                guard_broadcast,
                do_add,
                Exact
            ),
            reg!(
                0x0F21,
                "lance_graph_onnx::sub (broadcast checked)",
                "Sub",
                Some(0),
                F32,
                NO_ATTRS,
                guard_broadcast,
                do_sub,
                Exact
            ),
            reg!(
                0x0F22,
                "lance_graph_onnx::mul (broadcast checked)",
                "Mul",
                Some(0),
                F32,
                NO_ATTRS,
                guard_broadcast,
                do_mul,
                Exact
            ),
            reg!(
                0x0F23,
                "lance_graph_onnx::div (broadcast checked)",
                "Div",
                Some(0),
                F32,
                NO_ATTRS,
                guard_broadcast,
                do_div,
                Exact
            ),
            reg!(
                0x0F30,
                "lance_graph_onnx::abs_f32 = vml::vsabs",
                "Abs",
                Some(0),
                F32,
                NO_ATTRS,
                guard_true,
                do_abs,
                Exact
            ),
            reg!(
                0x0F31,
                "lance_graph_onnx::sqrt_f32 = vml::vssqrt",
                "Sqrt",
                Some(0),
                F32,
                NO_ATTRS,
                guard_true,
                do_sqrt,
                Exact
            ),
            reg!(
                0x0F32,
                "Pow(x, 2.0) = lance_graph_onnx::mul(x, x)",
                "Pow",
                Some(0),
                F32,
                NO_ATTRS,
                guard_pow2,
                do_pow2,
                Exact
            ),
            reg!(
                0x0F40,
                "ndarray mean_axis, keepdims 1",
                "ReduceMean",
                Some(0),
                F32,
                KEEPDIMS_1,
                guard_single_axis,
                do_reduce_mean,
                Bounded("float sum over the reduced axis; order is the implementation's")
            ),
            reg!(
                0x0F41,
                "lance_graph_onnx::ReduceSum::sum_i32, keepdims 0",
                "ReduceSum",
                Some(0),
                I32,
                KEEPDIMS_0,
                guard_rank1_sum,
                do_reduce_sum_i32_k0,
                Exact
            ),
            reg!(
                0x0F42,
                "lance_graph_onnx::ReduceSum::sum_i32, keepdims 1",
                "ReduceSum",
                Some(0),
                I32,
                KEEPDIMS_1,
                guard_rank1_sum,
                do_reduce_sum_i32_k1,
                Exact
            ),
            reg!(
                0x0F50,
                "lance_graph_onnx::greater_i32 = simd::gt_i32_to_mask",
                "Greater",
                Some(0),
                I32,
                NO_ATTRS,
                guard_scalar_rhs,
                do_greater_i32,
                Exact
            ),
            reg!(
                0x0F51,
                "simd::blend_i32 (Y tensor) / simd::blend_i32_scalar (Y scalar)",
                "Where",
                Some(1),
                I32,
                NO_ATTRS,
                guard_where,
                do_where_i32,
                Exact
            ),
            reg!(
                0x0F33,
                "ndarray::hpc::vml::vserf",
                "Erf",
                Some(0),
                F32,
                NO_ATTRS,
                guard_true,
                do_erf,
                Bounded("at most 2 ULP over [-5, 5] (ndarray #399)")
            ),
            reg!(
                0x0F60,
                "the input's shape, start/end clamped",
                "Shape",
                Some(0),
                F32,
                SHAPE_RANGE,
                guard_true,
                do_shape,
                Exact
            ),
            reg!(
                0x0F61,
                "the input's shape, start/end clamped",
                "Shape",
                Some(0),
                I64,
                SHAPE_RANGE,
                guard_true,
                do_shape,
                Exact
            ),
            reg!(
                0x0F62,
                "the input's shape, start/end clamped",
                "Shape",
                Some(0),
                I32,
                SHAPE_RANGE,
                guard_true,
                do_shape,
                Exact
            ),
            reg!(
                0x0F63,
                "the input's shape, start/end clamped",
                "Shape",
                Some(0),
                Bool,
                SHAPE_RANGE,
                guard_true,
                do_shape,
                Exact
            ),
            reg!(
                0x0F64,
                "ndarray select along axis, int64 data",
                "Gather",
                Some(0),
                I64,
                GATHER_AXIS,
                guard_gather,
                do_gather_i64,
                Exact
            ),
            reg!(
                0x0F65,
                "ndarray reshape: insert unit axes",
                "Unsqueeze",
                Some(0),
                I64,
                NO_ATTRS,
                guard_unsqueeze,
                do_unsqueeze,
                Exact
            ),
            reg!(
                0x0F66,
                "ndarray reshape: insert unit axes",
                "Unsqueeze",
                Some(0),
                F32,
                NO_ATTRS,
                guard_unsqueeze,
                do_unsqueeze,
                Exact
            ),
            reg!(
                0x0F67,
                "ndarray::concatenate along axis",
                "Concat",
                Some(0),
                I64,
                CONCAT_AXIS,
                guard_concat,
                do_concat,
                Exact
            ),
            reg!(
                0x0F68,
                "ndarray::concatenate along axis",
                "Concat",
                Some(0),
                F32,
                CONCAT_AXIS,
                guard_concat,
                do_concat,
                Exact
            ),
            reg!(
                0x0F69,
                "ndarray reshape, allowzero 0",
                "Reshape",
                Some(0),
                I64,
                ALLOWZERO_0,
                guard_reshape,
                do_reshape,
                Exact
            ),
            reg!(
                0x0F6A,
                "ndarray reshape, allowzero 0",
                "Reshape",
                Some(0),
                F32,
                ALLOWZERO_0,
                guard_reshape,
                do_reshape,
                Exact
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow_view<'a>(node: &'a SourceNode, base: &'a Fact, exp: &'a Value) -> NodeView<'a> {
        NodeView {
            node,
            args: vec![Some(base), None],
            consts: vec![None, Some(exp)],
        }
    }

    fn twos(shape: &[usize]) -> Value {
        Value::F32(ArrayD::from_elem(IxDyn(shape), 2.0))
    }

    fn node(op: &str, inputs: usize, attrs: Vec<(String, Attr)>) -> SourceNode {
        SourceNode {
            name: format!("n_{op}"),
            op_type: op.into(),
            domain: String::new(),
            inputs: (0..inputs).map(|i| format!("in{i}")).collect(),
            outputs: vec!["out".into()],
            attributes: attrs.iter().map(|(n, _)| n.clone()).collect(),
            attrs,
        }
    }

    fn f(shape: Vec<Dim>) -> Fact {
        Fact {
            dtype: Some(DType::F32),
            shape: Some(shape),
        }
    }

    /// Concat compares every non-axis dimension. Two different symbols are
    /// not proven equal (UNKNOWN, never TRUE); the same symbol is; known
    /// dimensions that differ are FALSE.
    #[test]
    fn concat_never_aliases_two_symbols() {
        use Dim::*;
        let n = node("Concat", 2, vec![("axis".into(), Attr::Int(1))]);
        let check = |a: Vec<Dim>, b: Vec<Dim>| {
            let (fa, fb) = (f(a), f(b));
            guard_concat(&NodeView {
                node: &n,
                args: vec![Some(&fa), Some(&fb)],
                consts: vec![None, None],
            })
        };
        let s = |x: &str| Symbol(x.into());
        assert_eq!(
            check(vec![s("N"), Known(2)], vec![s("M"), Known(3)]),
            K3::Unknown
        );
        assert_eq!(
            check(vec![s("N"), Known(2)], vec![s("N"), Known(3)]),
            K3::True
        );
        assert_eq!(
            check(vec![Known(4), Known(2)], vec![Known(5), Known(3)]),
            K3::False
        );
        assert_eq!(
            check(vec![Known(4), Known(2)], vec![Known(4)]),
            K3::False,
            "rank"
        );
    }

    /// Gather over a symbolic dimension cannot prove a constant index in
    /// range (UNKNOWN); a known dimension decides it.
    #[test]
    fn gather_index_range_is_three_valued() {
        use Dim::*;
        let n = node("Gather", 2, vec![("axis".into(), Attr::Int(0))]);
        let idx = Value::I64(ArrayD::from_elem(IxDyn(&[]), 2));
        let ifact = Fact {
            dtype: Some(DType::I64),
            shape: Some(vec![]),
        };
        let check = |d: Dim| {
            let data = Fact {
                dtype: Some(DType::I64),
                shape: Some(vec![d]),
            };
            guard_gather(&NodeView {
                node: &n,
                args: vec![Some(&data), Some(&ifact)],
                consts: vec![None, Some(&idx)],
            })
        };
        assert_eq!(check(Known(3)), K3::True);
        assert_eq!(check(Known(2)), K3::False);
        assert_eq!(check(Symbol("N".into())), K3::Unknown);
        let data = Fact {
            dtype: Some(DType::I64),
            shape: Some(vec![Known(3)]),
        };
        let no_const = guard_gather(&NodeView {
            node: &n,
            args: vec![Some(&data), Some(&ifact)],
            consts: vec![None, None],
        });
        assert_eq!(
            no_const,
            K3::Unknown,
            "a run-time index is not proven in range"
        );
    }

    /// `0` copies, one `-1` infers, the element count must match.
    #[test]
    fn reshape_target_rules() {
        assert_eq!(reshape_target(&[2, 3], &[3, -1]), Some(vec![3, 2]));
        assert_eq!(reshape_target(&[2, 3], &[0, -1]), Some(vec![2, 3]));
        assert_eq!(
            reshape_target(&[2, 3], &[4, -1]),
            None,
            "6 is not a multiple of 4"
        );
        assert_eq!(reshape_target(&[2, 3], &[-1, -1]), None);
        assert_eq!(reshape_target(&[2, 3], &[5]), None);
        assert_eq!(
            reshape_target(&[2, 3], &[0, 0, 0]),
            None,
            "0 past the input rank"
        );
        assert_eq!(reshape_target(&[2, 3], &[-2, 3]), None);
    }

    /// An attribute rule with `value: None` admits any integer, absent only
    /// with a default; a fixed rule keeps a different value a different
    /// signature.
    #[test]
    fn attr_rules_any_and_fixed() {
        let any = |a: Vec<(String, Attr)>| attrs_match(&node("Gather", 2, a), GATHER_AXIS);
        assert!(any(vec![("axis".into(), Attr::Int(-1))]));
        assert!(any(vec![]), "absent axis takes the default 0");
        assert!(
            !any(vec![("batch_dims".into(), Attr::Int(0))]),
            "an unlisted attribute does not match"
        );
        let cat = |a: Vec<(String, Attr)>| attrs_match(&node("Concat", 2, a), CONCAT_AXIS);
        assert!(cat(vec![("axis".into(), Attr::Int(3))]));
        assert!(!cat(vec![]), "Concat's axis has no default");
        let k = |a: Vec<(String, Attr)>| attrs_match(&node("ReduceMean", 2, a), KEEPDIMS_1);
        assert!(k(vec![]) && k(vec![("keepdims".into(), Attr::Int(1))]));
        assert!(!k(vec![("keepdims".into(), Attr::Int(0))]));
    }

    #[test]
    fn pow2_requires_the_exponent_not_to_grow_the_base() {
        let node = SourceNode {
            name: "p".into(),
            op_type: "Pow".into(),
            domain: String::new(),
            inputs: vec!["x".into(), "e".into()],
            outputs: vec!["y".into()],
            attributes: vec![],
            attrs: vec![],
        };
        let base = |s: &[usize]| Fact {
            dtype: Some(DType::F32),
            shape: Some(s.iter().map(|&d| Dim::Known(d)).collect()),
        };
        // scalar and broadcast-neutral exponents
        for (b, e) in [
            (&[1, 10, 1024][..], &[][..]),
            (&[1, 10, 1024], &[1]),
            (&[2, 3], &[1, 3]),
            (&[2, 3], &[1, 1, 1]),
        ] {
            assert_eq!(
                guard_pow2(&pow_view(&node, &base(b), &twos(e))),
                K3::True,
                "{b:?} ^ {e:?}"
            );
        }
        // the exponent would grow the output: the polyfill must not run
        for (b, e) in [(&[1][..], &[2][..]), (&[2, 1], &[2, 3]), (&[3], &[2, 3])] {
            assert_eq!(
                guard_pow2(&pow_view(&node, &base(b), &twos(e))),
                K3::False,
                "{b:?} ^ {e:?}"
            );
        }
        // a symbolic base dim against a non-1 exponent dim is UNKNOWN
        let sym = Fact {
            dtype: Some(DType::F32),
            shape: Some(vec![Dim::Symbol("S".into())]),
        };
        assert_eq!(guard_pow2(&pow_view(&node, &sym, &twos(&[4]))), K3::Unknown);
        // a non-2 exponent is FALSE whatever its shape
        assert_eq!(
            guard_pow2(&pow_view(
                &node,
                &base(&[3]),
                &Value::F32(ArrayD::from_elem(IxDyn(&[]), 3.0))
            )),
            K3::False
        );
    }
}
