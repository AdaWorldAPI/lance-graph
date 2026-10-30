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

/// A required discrete attribute value, with the ONNX default when absent.
#[derive(Debug, Clone, Copy)]
pub struct AttrRule {
    pub name: &'static str,
    pub value: i64,
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
            || rules
                .iter()
                .any(|r| r.name == n && matches!(a, Attr::Int(v) if *v == r.value))
    });
    let defaults = rules
        .iter()
        .all(|r| node.attr(r.name).is_some() || r.default == Some(r.value));
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

fn guard_pow2(v: &NodeView) -> K3 {
    match v.consts.get(1).copied().flatten().and_then(Value::f32s) {
        Some(e) if !e.is_empty() && e.iter().all(|x| x.to_bits() == 2.0f32.to_bits()) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
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

/// `Where(C, X, Y)`, int32, `Y` scalar or of `X`'s shape: `blend_i32`.
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
    let ys: Vec<i32> = if y.len() == 1 {
        vec![y.iter().copied().next().unwrap_or(0); xs.len()]
    } else {
        y.iter().copied().collect()
    };
    if ys.len() != xs.len() {
        return Err(Refusal::Runtime {
            node: n.name.clone(),
            what: "Y does not broadcast to X".into(),
        });
    }
    let mut out = vec![0i32; xs.len()];
    lance_graph_onnx::where_i32(words, &xs, &ys, &mut out);
    Ok(vec![Value::I32(
        ArrayD::from_shape_vec(IxDyn(x.shape()), out).expect("same length"),
    )])
}

// ── the table ───────────────────────────────────────────────────────────────

const OPSETS: RangeInclusive<i64> = 13..=18;
const NO_ATTRS: &[AttrRule] = &[];
const CAST_TO_F32: &[AttrRule] = &[AttrRule {
    name: "to",
    value: 1,
    default: None,
}];
const KEEPDIMS_1: &[AttrRule] = &[AttrRule {
    name: "keepdims",
    value: 1,
    default: Some(1),
}];
const KEEPDIMS_0: &[AttrRule] = &[AttrRule {
    name: "keepdims",
    value: 0,
    default: None,
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

/// The lab registry: every op the two fixtures use, nothing else.
pub fn registry() -> Registry {
    use DType::*;
    use Numeric::*;
    Registry {
        version: 1,
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
                "lance_graph_onnx::where_i32 = simd::blend_i32",
                "Where",
                Some(1),
                I32,
                NO_ATTRS,
                guard_where,
                do_where_i32,
                Exact
            ),
        ],
    }
}
