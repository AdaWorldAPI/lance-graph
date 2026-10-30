//! The DT64 ON × WHEN × DO registration of ONNX operators (D-DT64-6 §1:
//! "Behaviour is registered as a DTO: classid → polyfill over mask-risc /
//! ndarray").
//!
//! **A classid is a codebook lens, not a codebook item.** The ONNX operator
//! codebook is read through one lens, [`ONNX_LENS`]; each op signature is an
//! item in it, addressed as [`MethodKey`] `{ lens, ordinal }`. The ordinal is
//! the item's position in [`Registry::regs`], derived and never allocated
//! separately; reordering is a new [`Registry::version`]. (An earlier version
//! of this crate gave every op signature its own classid. That spent one
//! identity per codebook item, and the module doc attributed it to a user
//! decision that was never made: it was this session's assumption.)
//!
//! **The registry is a field view over lens-agnostic algebra** (the Redmine
//! ERB field-view pattern: agnostic field formats, per-tracker labelled
//! custom fields, a per-tracker set of enabled fields):
//!
//! - [`Registry::methods`] are the algebra addresses: one DO body each,
//!   addressed by position, carrying no ONNX label and no lens. Several items
//!   may share one (the three `Constant` dtypes read the same body).
//! - [`Registry::regs`] are the lens's fields: each labels one algebra
//!   address with an op signature and the WHEN that admits it. A field's
//!   position is its ordinal.
//! - [`Registry::exposed`] is the lens's [`WideFieldMask`]: which field
//!   positions it shows. A field outside the mask is absent — it neither
//!   matches nor resolves — and is never a different behaviour. The codebook
//!   is not capped at one mask: field `o` is position `o % 256` of block
//!   `o / 256` (global to local), up to 65,536 fields.
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
//! An item's behaviour is immutable: a new behaviour is a new item (a new
//! ordinal under a new registry version), never an edit (D-DT64-6 D5). An
//! unregistered signature is UNKNOWN (D-SAG-6 F06), never "no behaviour":
//! resolution refuses it.
//!
//! [`ONNX_LENS`] is a **lab** classid, not an OGAR-minted concept. A
//! production lens would be minted in OGAR and activated through
//! `lance-graph-contract::hotplug`, with the items as its capabilities.

use std::ops::RangeInclusive;

use lance_graph_contract::class_view::{ClassId, WideFieldMask};
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
    /// More positions on one axis than the u16 address width (65,536).
    AddressOverflow { axis: &'static str, count: usize },
    /// No registration for this op signature: UNKNOWN, never "no behaviour".
    Unregistered { node: String, signature: String },
    /// A registered signature outside the opsets its registration covers.
    Opset { node: String, opset: i64 },
    /// The WHEN guard is FALSE or UNKNOWN for this node.
    When {
        node: String,
        key: MethodKey,
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
/// - `value: None`: any integer or integer list, read by the DO as a
///   parameter (an axis, a permutation). Absent is admitted only if `default` is `Some` (the ONNX default the DO
///   applies). The method's behaviour does not depend on which value: the
///   same immutable function reads it.
#[derive(Debug, Clone, Copy)]
pub struct AttrRule {
    pub name: &'static str,
    pub value: Option<i64>,
    pub default: Option<i64>,
}

/// Field positions per exposure block: one `WideFieldMask`.
pub const MASK_BLOCK: usize = 256;

/// The lab classid of the ONNX operator codebook: one lens over every item.
pub const ONNX_LENS: ClassId = 0x0F00;

/// One codebook item: the lens it is read through and its position there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MethodKey {
    pub lens: ClassId,
    pub ordinal: u16,
}

/// One lens-agnostic algebra address: a DO body and how exactly it
/// reproduces ONNX. It carries no label; the fields of a lens label it.
#[derive(Debug, Clone, Copy)]
pub struct Method {
    pub name: &'static str,
    pub do_: Polyfill,
    pub numeric: Numeric,
}

/// One field of the lens: an op-signature label and its WHEN over the
/// algebra address [`Registration::method`].
#[derive(Clone)]
pub struct Registration {
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
    /// The algebra address this field labels: a position in
    /// [`Registry::methods`].
    pub method: u16,
}

/// The registration table, versioned. Its version is the frame's
/// `relation_map` revision: a different table is a different frame.
pub struct Registry {
    /// The codebook lens every item is read through.
    pub lens: ClassId,
    pub version: u64,
    /// The lens-agnostic algebra addresses, by position.
    pub methods: Vec<Method>,
    /// The lens's fields; a field's ordinal is its index here.
    pub regs: Vec<Registration>,
    /// Which field positions the lens exposes, in blocks of
    /// [`MASK_BLOCK`] (a `WideFieldMask` position is a `u8`).
    pub exposed: Vec<WideFieldMask>,
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
    /// Build a lens from `(algebra address, field)` drafts: equal addresses
    /// (by name) are one position, and every field is exposed.
    ///
    /// # Panics
    /// If two drafts name one address with different bodies or numerics, or
    /// the table outgrows its address widths.
    pub fn build(lens: ClassId, version: u64, drafts: Vec<(Method, Registration)>) -> Self {
        let mut methods: Vec<Method> = Vec::new();
        let mut regs = Vec::with_capacity(drafts.len());
        for (m, mut r) in drafts {
            let ix = match methods.iter().position(|x| x.name == m.name) {
                Some(ix) => {
                    let x = &methods[ix];
                    assert!(
                        std::ptr::fn_addr_eq(x.do_, m.do_) && x.numeric == m.numeric,
                        "address {:?} has two bodies",
                        m.name
                    );
                    ix
                }
                None => {
                    methods.push(m);
                    methods.len() - 1
                }
            };
            r.method = u16::try_from(ix).expect("at most 65,536 addresses");
            regs.push(r);
        }
        assert!(regs.len() <= 1 << 16, "a field ordinal is a u16");
        let exposed = (0..regs.len().div_ceil(MASK_BLOCK))
            .map(|b| WideFieldMask::full_for((regs.len() - b * MASK_BLOCK).min(MASK_BLOCK)))
            .collect();
        Registry {
            lens,
            version,
            methods,
            regs,
            exposed,
        }
    }

    /// The algebra address `reg` labels.
    pub fn method(&self, reg: &Registration) -> &Method {
        &self.methods[usize::from(reg.method)]
    }

    /// Does the lens expose field `ordinal`?
    pub fn exposes(&self, ordinal: usize) -> bool {
        self.exposed
            .get(ordinal / MASK_BLOCK)
            .is_some_and(|m| m.has((ordinal % MASK_BLOCK) as u8))
    }

    /// Show or hide field `ordinal`. Hiding is a new [`revision`](Self::revision).
    pub fn expose(&mut self, ordinal: usize, on: bool) {
        let (b, local) = (ordinal / MASK_BLOCK, (ordinal % MASK_BLOCK) as u8);
        if self.exposed.len() <= b {
            self.exposed.resize(b + 1, WideFieldMask::EMPTY);
        }
        let kept: Vec<u8> = (0..=u8::MAX)
            .filter(|&p| p != local && self.exposed[b].has(p))
            .chain(on.then_some(local))
            .collect();
        self.exposed[b] = WideFieldMask::from_positions(&kept);
    }

    /// The revision a frame and a route are keyed by: the table version and
    /// the exposed field positions. Hiding a field is a new revision.
    pub fn revision(&self) -> u64 {
        let shown: Vec<u8> = (0..self.regs.len())
            .filter(|&i| self.exposes(i))
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect();
        crate::plan::digest64(&[&self.version.to_le_bytes(), &shown])
    }

    /// The key of `reg`, which must be an item of this table.
    pub fn key_of(&self, reg: &Registration) -> MethodKey {
        let ix = self
            .regs
            .iter()
            .position(|r| std::ptr::eq(r, reg))
            .expect("an item of this table");
        MethodKey {
            lens: self.lens,
            ordinal: u16::try_from(ix).expect("at most 65,536 items"),
        }
    }

    /// The item at `key`, if it is read through this table's lens.
    pub fn get(&self, key: MethodKey) -> Option<&Registration> {
        if key.lens != self.lens || !self.exposes(usize::from(key.ordinal)) {
            return None;
        }
        self.regs.get(usize::from(key.ordinal))
    }

    /// The exposed field whose signature this node has, if any.
    pub fn lookup(&self, node: &SourceNode, args: &[Option<&Fact>]) -> Option<&Registration> {
        let domain = if node.domain.is_empty() {
            "ai.onnx"
        } else {
            node.domain.as_str()
        };
        self.regs.iter().enumerate().find_map(|(i, r)| {
            (self.exposes(i)
                && r.domain == domain
                && r.op_type == node.op_type
                && signature_dtype(node, args, r.dtype_arg) == Some(r.dtype)
                && attrs_match(node, r.attrs))
            .then_some(r)
        })
    }
}

pub(crate) fn attrs_match(node: &SourceNode, rules: &[AttrRule]) -> bool {
    // Every node attribute is either the tensor `value` of a Constant or a
    // listed discrete attribute with the required value.
    let listed = node.attrs.iter().all(|(n, a)| {
        (n == "value" && matches!(a, Attr::Tensor(_)))
            || rules.iter().any(|r| {
                r.name == n
                    && (matches!(a, Attr::Int(v) if r.value.is_none_or(|w| *v == w))
                        || (r.value.is_none() && matches!(a, Attr::Ints(_))))
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

pub(crate) fn guard_true(_: &NodeView) -> K3 {
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
pub(crate) fn guard_gather(v: &NodeView) -> K3 {
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
pub(crate) fn guard_reshape(v: &NodeView) -> K3 {
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

/// A binary int64 comparison as admitted here: the right side has the left
/// side's shape (pairwise) or one element (scalar). Other broadcasts are not
/// admitted.
fn guard_pair_or_scalar(v: &NodeView) -> K3 {
    let (Some(a), Some(b)) = (arg_shape(v, 0), arg_shape(v, 1)) else {
        return K3::Unknown;
    };
    if a == b && a.iter().all(|d| matches!(d, Dim::Known(_))) {
        return K3::True;
    }
    match known(&Some(b.clone())) {
        Some(s) if s.iter().product::<usize>() == 1 && b.len() <= a.len() => K3::True,
        Some(_) if known(&Some(a.clone())).is_some() => K3::False,
        _ => K3::Unknown,
    }
}

/// Binary int64 arithmetic as admitted here: equal known shapes, or one
/// side with one element and rank at most the other's (it broadcasts). Other
/// broadcasts are not admitted.
fn guard_binary_i64(v: &NodeView) -> K3 {
    let (Some(a), Some(b)) = (arg_shape(v, 0), arg_shape(v, 1)) else {
        return K3::Unknown;
    };
    let (Some(ka), Some(kb)) = (known(&Some(a.clone())), known(&Some(b.clone()))) else {
        return K3::Unknown;
    };
    let one = |s: &[usize]| s.iter().product::<usize>() == 1;
    if ka == kb || (one(&kb) && kb.len() <= ka.len()) || (one(&ka) && ka.len() <= kb.len()) {
        K3::True
    } else {
        K3::False
    }
}

/// `Range`: three one-element inputs; a constant zero delta is FALSE.
fn guard_range(v: &NodeView) -> K3 {
    let mut t = K3::True;
    for i in 0..3 {
        match arg_shape(v, i).map(|s| known(&Some(s.clone()))) {
            Some(Some(s)) if s.iter().product::<usize>() == 1 && s.len() <= 1 => {}
            Some(Some(_)) => return K3::False,
            _ => t = K3::Unknown,
        }
    }
    match v.consts.get(2).copied().flatten().and_then(Value::i64s) {
        Some(d) if d.first() == Some(&0) => K3::False,
        _ => t,
    }
}

/// int64 compares: a same-shape or one-element right side (the scalar
/// kernel), or any multidirectional broadcast of two known shapes (both
/// sides broadcast to the common shape, then the pair kernel). An
/// incompatible pair is FALSE; an unknown dimension is UNKNOWN.
fn guard_compare_i64(v: &NodeView) -> K3 {
    match guard_pair_or_scalar(v) {
        K3::True => K3::True,
        K3::Unknown => K3::Unknown,
        K3::False => {
            let (Some(a), Some(b)) = (
                known(&arg_shape(v, 0).cloned()),
                known(&arg_shape(v, 1).cloned()),
            ) else {
                return K3::Unknown;
            };
            if broadcast_shapes(&a, &b).is_some() {
                K3::True
            } else {
                K3::False
            }
        }
    }
}

/// `MatMul` f32 as admitted here, for left `[.., m, k]` and right
/// `[.., k, n]` with matching `k`:
/// - a rank-2 right operand (a weight): the left's leading dimensions are
///   flattened into `m`;
/// - equal leading dimensions on both sides (flattened into one batch axis),
///   or right leading dimensions all 1 (shared by every left batch entry).
///
/// Other ranks and batch broadcasts are not admitted.
fn guard_matmul(v: &NodeView) -> K3 {
    let (Some(a), Some(b)) = (
        known(&arg_shape(v, 0).cloned()),
        known(&arg_shape(v, 1).cloned()),
    ) else {
        return K3::Unknown;
    };
    if matmul_plan(&a, &b).is_some() {
        K3::True
    } else {
        K3::False
    }
}

/// How an admitted `MatMul` runs: `(batch, m, k, n, right_batch)` over
/// flattened views, and the output shape.
type MatmulPlan = ((usize, usize, usize, usize, usize), Vec<usize>);

fn matmul_plan(a: &[usize], b: &[usize]) -> Option<MatmulPlan> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let (m, k) = (a[a.len() - 2], a[a.len() - 1]);
    let (kb, n) = (b[b.len() - 2], b[b.len() - 1]);
    if k != kb {
        return None;
    }
    let (la, lb) = (&a[..a.len() - 2], &b[..b.len() - 2]);
    let mut out = la.to_vec();
    out.extend([m, n]);
    let batch: usize = la.iter().product();
    if lb.is_empty() {
        // A weight: every left row against one [k, n].
        return Some(((1, batch * m, k, n, 1), out));
    }
    if la == lb {
        return Some(((batch, m, k, n, batch), out));
    }
    if la.len() == lb.len() && lb.iter().all(|&d| d == 1) {
        return Some(((batch, m, k, n, 1), out));
    }
    None
}

/// `Transpose`: `perm` (default: reversed axes) is a permutation of the
/// input's rank.
fn guard_transpose(v: &NodeView) -> K3 {
    let Some(r) = arg_shape(v, 0).map(Vec::len) else {
        return K3::Unknown;
    };
    match transpose_perm(v.node, r) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

fn transpose_perm(n: &SourceNode, rank: usize) -> Option<Vec<usize>> {
    let perm: Vec<usize> = match n.attr("perm") {
        None => (0..rank).rev().collect(),
        Some(Attr::Ints(p)) => p
            .iter()
            .map(|&x| usize::try_from(x).ok())
            .collect::<Option<_>>()?,
        Some(_) => return None,
    };
    let mut seen = vec![false; rank];
    if perm.len() != rank {
        return None;
    }
    for &p in &perm {
        if p >= rank || std::mem::replace(&mut seen[p], true) {
            return None;
        }
    }
    Some(perm)
}

/// Element-wise bool `And`: two known shapes that broadcast.
fn guard_and(v: &NodeView) -> K3 {
    match (
        known(&arg_shape(v, 0).cloned()),
        known(&arg_shape(v, 1).cloned()),
    ) {
        (Some(a), Some(b)) if broadcast_shapes(&a, &b).is_some() => K3::True,
        (Some(_), Some(_)) => K3::False,
        _ => K3::Unknown,
    }
}

/// A packed `tensor(bool)` broadcast to `to` and repacked: the same bits a
/// broadcast view reads.
fn broadcast_bool(shape: &[usize], words: &[u64], to: &[usize]) -> Option<Vec<u64>> {
    let len: usize = shape.iter().product();
    let bits = ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..len)
            .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
            .collect(),
    )
    .ok()?;
    let b = bits.broadcast(IxDyn(to))?;
    let mut out = vec![0u64; lance_graph_onnx::mask_words(b.len())];
    for (i, &v) in b.iter().enumerate() {
        out[i / 64] |= u64::from(v) << (i % 64);
    }
    Some(out)
}

/// `Cast` to the input's own dtype (bool to bool): the identity.
fn guard_cast_bool_identity(v: &NodeView) -> K3 {
    match v.args.first().copied().flatten().and_then(|f| f.dtype) {
        Some(DType::Bool) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

/// `ConstantOfShape`: a constant, non-negative shape and a one-element value.
fn guard_constant_of_shape(v: &NodeView) -> K3 {
    match v.node.attr("value") {
        None => {}
        Some(Attr::Tensor(Some(t))) if t.shape().iter().product::<usize>() == 1 => {}
        Some(_) => return K3::False,
    }
    match arg_shape(v, 0).map(Vec::len) {
        Some(1) => {}
        Some(_) => return K3::False,
        None => return K3::Unknown,
    }
    match v.consts.first().copied().flatten().and_then(Value::i64s) {
        Some(d) if d.iter().all(|&x| x >= 0) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

/// The multidirectional broadcast of two shapes, or `None`.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let at = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    (0..r)
        .map(|i| match (at(a, i), at(b, i)) {
            (x, y) if x == y || y == 1 => Some(x),
            (1, y) => Some(y),
            _ => None,
        })
        .collect()
}

/// `Expand`: a constant target over a fully known input shape that
/// broadcasts against it.
pub(crate) fn guard_expand(v: &NodeView) -> K3 {
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
    if target.iter().any(|&t| t < 0) {
        return K3::False;
    }
    let t: Vec<usize> = target.iter().map(|&t| t as usize).collect();
    match broadcast_shapes(&input, &t) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

// ── DO polyfills: existing native functions only ───────────────────────────

pub(crate) fn f32_arg<'a>(
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
pub(crate) fn do_gather_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    // Bool data: the same index walk over the unpacked elements.
    if let Some(b) = a.first().and_then(|v| unpack_bools(v)) {
        let as_i64 = Value::I64(b.mapv(i64::from));
        let mut args = a.to_vec();
        args[0] = &as_i64;
        let out = do_gather_i64(&args, n)?;
        return Ok(match &out[0] {
            Value::I64(v) => vec![pack_bools(&v.mapv(|x| x != 0))],
            other => vec![other.clone()],
        });
    }
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

pub(crate) fn do_reshape(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
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

/// An int64 comparison into a packed `tensor(bool)`: the `_pair` kernel when
/// the right side has the left side's length, the scalar kernel for a
/// one-element right side.
fn compare_i64(
    a: &[&Value],
    n: &SourceNode,
    pair: fn(&[i64], &[i64], &mut [u64]),
    scalar: fn(&[i64], i64, &mut [u64]),
) -> Result<Vec<Value>, Refusal> {
    let x = i64_arg(a, 0, n)?;
    let y = i64_arg(a, 1, n)?;
    let xs: Vec<i64> = x.iter().copied().collect();
    let mut words = vec![0u64; lance_graph_onnx::mask_words(xs.len())];
    if y.shape() == x.shape() {
        let ys: Vec<i64> = y.iter().copied().collect();
        pair(&xs, &ys, &mut words);
    } else if let (1, Some(&t)) = (y.len(), y.iter().next()) {
        if y.ndim() > x.ndim() {
            return compare_broadcast(x, y, n, pair);
        }
        scalar(&xs, t, &mut words);
    } else {
        return compare_broadcast(x, y, n, pair);
    }
    Ok(vec![Value::Bool {
        shape: x.shape().to_vec(),
        words,
    }])
}

/// Both sides broadcast to their common shape (ndarray `broadcast`), then
/// the pair kernel.
fn compare_broadcast(
    x: &ArrayD<i64>,
    y: &ArrayD<i64>,
    n: &SourceNode,
    pair: fn(&[i64], &[i64], &mut [u64]),
) -> Result<Vec<Value>, Refusal> {
    let shape = broadcast_shapes(x.shape(), y.shape()).ok_or_else(|| {
        runtime(
            n,
            format!("{:?} and {:?} do not broadcast", x.shape(), y.shape()),
        )
    })?;
    let bx: Vec<i64> = x
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| runtime(n, "broadcast failed"))?
        .iter()
        .copied()
        .collect();
    let by: Vec<i64> = y
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| runtime(n, "broadcast failed"))?
        .iter()
        .copied()
        .collect();
    let mut words = vec![0u64; lance_graph_onnx::mask_words(bx.len())];
    pair(&bx, &by, &mut words);
    Ok(vec![Value::Bool { shape, words }])
}

/// The most elements `Range` may produce: a policy bound, so a large or
/// wrong-signed range refuses instead of allocating.
pub const RANGE_CAPACITY: usize = 1 << 16;

fn arith_refusal(n: &SourceNode, e: ndarray::hpc::checked_i64::I64ArithError) -> Refusal {
    runtime(n, format!("checked int64: {e}"))
}

fn binary_i64(
    a: &[&Value],
    n: &SourceNode,
    op: ndarray::hpc::checked_i64::I64Op,
) -> Result<Vec<Value>, Refusal> {
    let x = i64_arg(a, 0, n)?;
    let y = i64_arg(a, 1, n)?;
    let shape = if x.len() == 1 && y.ndim() >= x.ndim() {
        y.shape().to_vec()
    } else {
        x.shape().to_vec()
    };
    let xs: Vec<i64> = x.iter().copied().collect();
    let ys: Vec<i64> = y.iter().copied().collect();
    let mut out = vec![0i64; shape.iter().product()];
    ndarray::hpc::checked_i64::checked_binary_i64(op, &xs, &ys, &mut out)
        .map_err(|e| arith_refusal(n, e))?;
    Ok(vec![Value::I64(
        ArrayD::from_shape_vec(IxDyn(&shape), out).expect("broadcast length"),
    )])
}

fn do_add_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    binary_i64(a, n, ndarray::hpc::checked_i64::I64Op::Add)
}

fn do_sub_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    binary_i64(a, n, ndarray::hpc::checked_i64::I64Op::Sub)
}

fn do_mul_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    binary_i64(a, n, ndarray::hpc::checked_i64::I64Op::Mul)
}

fn do_div_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    binary_i64(a, n, ndarray::hpc::checked_i64::I64Op::Div)
}

fn do_range_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let mut s = [0i64; 3];
    for (i, v) in s.iter_mut().enumerate() {
        let x = i64_arg(a, i, n)?;
        *v = match (x.len(), x.iter().next()) {
            (1, Some(&v)) => v,
            _ => return Err(runtime(n, format!("arg {i}: not a scalar"))),
        };
    }
    let mut out = vec![0i64; RANGE_CAPACITY];
    let len = ndarray::hpc::checked_i64::range_i64(s[0], s[1], s[2], &mut out)
        .map_err(|e| arith_refusal(n, e))?;
    out.truncate(len);
    Ok(vec![i64_vector(out)])
}

fn do_cast_i64_f32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    // `as` rounds to nearest, ties to even, as C++ `static_cast` under the
    // default rounding mode.
    Ok(vec![Value::F32(i64_arg(a, 0, n)?.mapv(|v| v as f32))])
}

fn do_matmul_f32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let y = f32_arg(a, 1, n)?;
    let bad = |e: String| runtime(n, e);
    let ((b, m, k, nn, rb), out_shape) = matmul_plan(x.shape(), y.shape())
        .ok_or_else(|| bad(format!("{:?} x {:?} is not admitted", x.shape(), y.shape())))?;
    // Standard-layout copies only when an input is not already contiguous;
    // the reshapes below are then views.
    let (xs, ys) = (x.as_standard_layout(), y.as_standard_layout());
    let xv = xs
        .view()
        .into_shape_with_order((b, m, k))
        .map_err(|e| bad(e.to_string()))?;
    let yv = ys
        .view()
        .into_shape_with_order((rb, k, nn))
        .map_err(|e| bad(e.to_string()))?;
    let mut out = ndarray::Array3::<f32>::zeros((b, m, nn));
    ndarray::simd::matmul_f32_batched(xv, yv, out.view_mut()).map_err(|e| bad(format!("{e:?}")))?;
    Ok(vec![Value::F32(
        out.into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| bad(e.to_string()))?,
    )])
}

fn do_transpose(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let perm = transpose_perm(n, x.shape().len()).ok_or_else(|| runtime(n, "bad perm"))?;
    Ok(vec![match x {
        Value::F32(v) => Value::F32(
            v.view()
                .permuted_axes(IxDyn(&perm))
                .as_standard_layout()
                .into_owned(),
        ),
        Value::I64(v) => Value::I64(
            v.view()
                .permuted_axes(IxDyn(&perm))
                .as_standard_layout()
                .into_owned(),
        ),
        other => return Err(runtime(n, format!("Transpose of {:?}", other.dtype()))),
    }])
}

fn do_cos(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let mut out = ArrayD::<f32>::zeros(x.raw_dim());
    ndarray::hpc::vml::vscos(x.view(), out.view_mut());
    Ok(vec![Value::F32(out)])
}

fn do_sin(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?;
    let mut out = ArrayD::<f32>::zeros(x.raw_dim());
    ndarray::hpc::vml::vssin(x.view(), out.view_mut());
    Ok(vec![Value::F32(out)])
}

fn do_and_bool(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let (
        Some(Value::Bool {
            shape: sa,
            words: wa,
        }),
        Some(Value::Bool {
            shape: sb,
            words: wb,
        }),
    ) = (a.first(), a.get(1))
    else {
        return Err(runtime(n, "And needs two bool tensors"));
    };
    let shape = broadcast_shapes(sa, sb)
        .ok_or_else(|| runtime(n, format!("{sa:?} and {sb:?} do not broadcast")))?;
    let bc = |s: &Vec<usize>, w: &Vec<u64>| {
        if *s == shape {
            Some(w.clone())
        } else {
            broadcast_bool(s, w, &shape)
        }
    };
    let (Some(xa), Some(xb)) = (bc(sa, wa), bc(sb, wb)) else {
        return Err(runtime(n, "broadcast failed"));
    };
    let mut words = vec![0u64; xa.len()];
    ndarray::simd::mask_and(&xa, &xb, &mut words);
    Ok(vec![Value::Bool { shape, words }])
}

fn do_equal_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    compare_i64(
        a,
        n,
        ndarray::simd::eq_i64_pair_to_mask,
        ndarray::simd::eq_i64_to_mask,
    )
}

fn do_less_or_equal_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    compare_i64(
        a,
        n,
        ndarray::simd::le_i64_pair_to_mask,
        ndarray::simd::le_i64_to_mask,
    )
}

/// `Where(C, X, Y)` over int64: `blend_i64`, or `blend_i64_scalar` for a
/// scalar `Y`.
fn do_where_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let Some(Value::Bool { shape, words }) = a.first() else {
        return Err(runtime(n, "condition is not a bool tensor"));
    };
    let x = i64_arg(a, 1, n)?;
    let y = i64_arg(a, 2, n)?;
    if shape.as_slice() != x.shape() {
        return Err(runtime(n, "condition and X shapes differ"));
    }
    let xs: Vec<i64> = x.iter().copied().collect();
    let mut out = vec![0i64; xs.len()];
    if let (1, Some(&yv)) = (y.len(), y.iter().next()) {
        ndarray::simd::blend_i64_scalar(words, &xs, yv, &mut out);
    } else if y.shape() == x.shape() {
        let ys: Vec<i64> = y.iter().copied().collect();
        ndarray::simd::blend_i64(words, &xs, &ys, &mut out);
    } else {
        return Err(runtime(n, "Y does not broadcast to X"));
    }
    Ok(vec![Value::I64(
        ArrayD::from_shape_vec(IxDyn(x.shape()), out).expect("same length"),
    )])
}

/// `ConstantOfShape`: `ArrayD::from_elem` of the attribute's one element
/// (ONNX default: float 0); a bool value fills the packed mask.
fn do_constant_of_shape(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let dims = a
        .first()
        .and_then(|v| v.i64s())
        .ok_or_else(|| runtime(n, "no shape"))?;
    let shape: Vec<usize> = dims
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| runtime(n, format!("negative dimension {d}"))))
        .collect::<Result<_, _>>()?;
    let len: usize = shape.iter().product();
    let sh = IxDyn(&shape);
    let out = match n.attr("value") {
        None => Value::F32(ArrayD::from_elem(sh, 0.0)),
        Some(Attr::Tensor(Some(t))) => match t {
            Value::F32(v) => Value::F32(ArrayD::from_elem(
                sh,
                v.iter().copied().next().unwrap_or(0.0),
            )),
            Value::I32(v) => {
                Value::I32(ArrayD::from_elem(sh, v.iter().copied().next().unwrap_or(0)))
            }
            Value::I64(v) => {
                Value::I64(ArrayD::from_elem(sh, v.iter().copied().next().unwrap_or(0)))
            }
            Value::Bool { words, .. } => {
                let mut w = vec![0u64; lance_graph_onnx::mask_words(len)];
                if words.first().is_some_and(|w| w & 1 == 1) {
                    w.fill(u64::MAX);
                    if !len.is_multiple_of(64) {
                        if let Some(last) = w.last_mut() {
                            *last = (1u64 << (len % 64)) - 1;
                        }
                    }
                }
                Value::Bool { shape, words: w }
            }
        },
        Some(_) => return Err(runtime(n, "value is not a decodable tensor")),
    };
    Ok(vec![out])
}

/// `Expand`: `ndarray`'s `broadcast` to the multidirectional broadcast of the
/// input shape and the target.
pub(crate) fn do_expand(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let target: Vec<usize> = a
        .get(1)
        .and_then(|v| v.i64s())
        .ok_or_else(|| runtime(n, "no target shape"))?
        .iter()
        .map(|&t| usize::try_from(t).map_err(|_| runtime(n, format!("negative dimension {t}"))))
        .collect::<Result<_, _>>()?;
    let shape = broadcast_shapes(&x.shape(), &target).ok_or_else(|| {
        runtime(
            n,
            format!("{:?} does not broadcast to {target:?}", x.shape()),
        )
    })?;
    fn bc<T: Clone>(a: &ArrayD<T>, s: &[usize]) -> Option<ArrayD<T>> {
        a.broadcast(IxDyn(s)).map(|v| v.to_owned())
    }
    let bad = || runtime(n, "broadcast failed");
    Ok(vec![match x {
        Value::F32(v) => Value::F32(bc(v, &shape).ok_or_else(bad)?),
        Value::I64(v) => Value::I64(bc(v, &shape).ok_or_else(bad)?),
        b @ Value::Bool { .. } => {
            pack_bools(&bc(&unpack_bools(b).ok_or_else(bad)?, &shape).ok_or_else(bad)?)
        }
        other => return Err(runtime(n, format!("Expand of {:?}", other.dtype()))),
    }])
}

// ── the table ───────────────────────────────────────────────────────────────

// ── hand-written fields for Jina's remaining op types (plan §0.10) ────────

/// A packed bool tensor as an `ArrayD<bool>` (for the view ops, which move
/// whole elements).
fn unpack_bools(v: &Value) -> Option<ArrayD<bool>> {
    let Value::Bool { shape, words } = v else {
        return None;
    };
    let n: usize = shape.iter().product();
    let bits = (0..n)
        .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
        .collect();
    ArrayD::from_shape_vec(IxDyn(shape), bits).ok()
}

/// An `ArrayD<bool>` packed, valid bits only.
fn pack_bools(a: &ArrayD<bool>) -> Value {
    let mut words = vec![0u64; lance_graph_onnx::mask_words(a.len())];
    for (i, &b) in a.iter().enumerate() {
        if b {
            words[i / 64] |= 1 << (i % 64);
        }
    }
    Value::Bool {
        shape: a.shape().to_vec(),
        words,
    }
}

fn const_i64s(v: &NodeView, i: usize) -> Option<Vec<i64>> {
    v.consts.get(i).copied().flatten().and_then(Value::i64s)
}

/// `Slice` (opset ≥ 10): per listed axis, `[start, end)` with step, ONNX
/// clamping (negative indices count from the end, then clamp to `[0, d]`).
/// Positive steps only.
fn slice_spec(
    shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Option<Vec<Option<(usize, usize, usize)>>> {
    let r = shape.len();
    let axes: Vec<usize> = match axes {
        Some(a) => a.iter().map(|&x| norm_axis(x, r)).collect::<Option<_>>()?,
        None => (0..starts.len()).collect(),
    };
    if starts.len() != axes.len() || ends.len() != axes.len() {
        return None;
    }
    let mut spec = vec![None; r];
    for (k, &ax) in axes.iter().enumerate() {
        let step = steps.map_or(Some(1), |s| s.get(k).copied())?;
        if step <= 0 || spec[ax].is_some() {
            return None;
        }
        let d = shape[ax] as i64;
        let clamp = |v: i64| (if v < 0 { v + d } else { v }).clamp(0, d) as usize;
        spec[ax] = Some((clamp(starts[k]), clamp(ends[k]), step as usize));
    }
    Some(spec)
}

fn guard_slice(v: &NodeView) -> K3 {
    let Some(shape) = arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) else {
        return K3::Unknown;
    };
    let (Some(st), Some(en)) = (const_i64s(v, 1), const_i64s(v, 2)) else {
        return K3::Unknown;
    };
    let present = |i| v.node.inputs.get(i).is_some_and(|s: &String| !s.is_empty());
    let axes = if present(3) { const_i64s(v, 3) } else { None };
    let steps = if present(4) { const_i64s(v, 4) } else { None };
    if (present(3) && axes.is_none()) || (present(4) && steps.is_none()) {
        return K3::Unknown;
    }
    match slice_spec(&shape, &st, &en, axes.as_deref(), steps.as_deref()) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

fn do_slice(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let int = |i: usize| a.get(i).and_then(|v| v.i64s());
    let st = int(1).ok_or_else(|| runtime(n, "no starts"))?;
    let en = int(2).ok_or_else(|| runtime(n, "no ends"))?;
    let (axes, steps) = (int(3), int(4));
    let spec = slice_spec(&x.shape(), &st, &en, axes.as_deref(), steps.as_deref())
        .ok_or_else(|| runtime(n, "slice parameters out of range"))?;
    fn cut<T: Clone>(a: &ArrayD<T>, spec: &[Option<(usize, usize, usize)>]) -> ArrayD<T> {
        a.slice_each_axis(|ax| match spec[ax.axis.index()] {
            Some((s, e, step)) => {
                ndarray::Slice::new(s as isize, Some(e.max(s) as isize), step as isize)
            }
            None => ndarray::Slice::from(..),
        })
        .to_owned()
    }
    Ok(vec![match x {
        Value::F32(v) => Value::F32(cut(v, &spec)),
        Value::I64(v) => Value::I64(cut(v, &spec)),
        b @ Value::Bool { .. } => pack_bools(&cut(
            &unpack_bools(b).ok_or_else(|| runtime(n, "bad bool"))?,
            &spec,
        )),
        other => return Err(runtime(n, format!("Slice of {:?}", other.dtype()))),
    }])
}

/// `Where(C, X, Y)` with f32 values: `X` and `Y` each either one element or
/// `C`'s shape, so the output has `C`'s shape.
fn guard_where_f32(v: &NodeView) -> K3 {
    let Some(c) = arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) else {
        return K3::Unknown;
    };
    let side = |i| match arg_shape(v, i).and_then(|s| known(&Some(s.clone()))) {
        Some(s) if s.iter().product::<usize>() == 1 || s == c => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    };
    match (side(1), side(2)) {
        (K3::False, _) | (_, K3::False) => K3::False,
        (K3::True, K3::True) => K3::True,
        _ => K3::Unknown,
    }
}

/// `blend_f32` / `blend_f32_scalar` (a set bit selects `X`). A scalar `X`
/// with a tensor `Y` blends `Y` under the complemented condition.
fn do_where_f32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let Some(Value::Bool { shape, words }) = a.first() else {
        return Err(runtime(n, "condition is not a bool tensor"));
    };
    let len: usize = shape.iter().product();
    let x = f32_arg(a, 1, n)?;
    let y = f32_arg(a, 2, n)?;
    let flat = |t: &ArrayD<f32>| t.iter().copied().collect::<Vec<f32>>();
    let mut out = vec![0f32; len];
    match (x.len() == 1, y.len() == 1) {
        (false, false) => ndarray::simd::blend_f32(words, &flat(x), &flat(y), &mut out),
        (false, true) => ndarray::simd::blend_f32_scalar(
            words,
            &flat(x),
            y.iter().copied().next().unwrap_or(0.0),
            &mut out,
        ),
        (true, false) => {
            let mut inv: Vec<u64> = words.iter().map(|w| !w).collect();
            if !len.is_multiple_of(64) {
                if let Some(l) = inv.last_mut() {
                    *l &= (1u64 << (len % 64)) - 1;
                }
            }
            ndarray::simd::blend_f32_scalar(
                &inv,
                &flat(y),
                x.iter().copied().next().unwrap_or(0.0),
                &mut out,
            )
        }
        (true, true) => {
            let xs = vec![x.iter().copied().next().unwrap_or(0.0); len];
            ndarray::simd::blend_f32_scalar(
                words,
                &xs,
                y.iter().copied().next().unwrap_or(0.0),
                &mut out,
            )
        }
    }
    let out = ArrayD::from_shape_vec(IxDyn(shape), out).map_err(|e| runtime(n, e.to_string()))?;
    Ok(vec![Value::F32(out)])
}

const SOFTMAX_LAST: &[AttrRule] = &[AttrRule {
    name: "axis",
    value: Some(-1),
    default: Some(-1),
}];

fn guard_known_shape(v: &NodeView) -> K3 {
    match arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) {
        Some(s) if !s.is_empty() => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

/// `Softmax` over the last axis (opset ≥ 13): `softmax_axis_f32` on the
/// `[rows, last]` view.
fn do_softmax_last(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = f32_arg(a, 0, n)?.as_standard_layout().into_owned();
    let shape = x.shape().to_vec();
    let last = *shape.last().ok_or_else(|| runtime(n, "rank 0"))?;
    let rows = x.len() / last.max(1);
    let x2 = x
        .into_shape_with_order((rows, last))
        .map_err(|e| runtime(n, e.to_string()))?;
    let mut out = ndarray::Array2::<f32>::zeros((rows, last));
    ndarray::hpc::activations::softmax_axis_f32(x2.view(), out.view_mut(), Axis(1));
    let out = out
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|e| runtime(n, e.to_string()))?;
    Ok(vec![Value::F32(out)])
}

/// `Gather` axis 0 over a 2-D f32 table by i64 indices of any shape (run-time
/// indices allowed): `gather_rows_f32`, which refuses an index outside
/// `[-rows, rows)`.
fn guard_gather_rows(v: &NodeView) -> K3 {
    let Some(data) = arg_shape(v, 0) else {
        return K3::Unknown;
    };
    if data.len() != 2 || norm_axis(int_attr(v.node, "axis", 0), 2) != Some(0) {
        return K3::False;
    }
    match v.args.get(1).copied().flatten().and_then(|f| f.dtype) {
        Some(DType::I64) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

fn do_gather_rows_f32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let table = f32_arg(a, 0, n)?
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| runtime(n, e.to_string()))?;
    let idx_v = a.get(1).ok_or_else(|| runtime(n, "no indices"))?;
    let ids = idx_v
        .i64s()
        .ok_or_else(|| runtime(n, "indices are not i64"))?;
    let cols = table.ncols();
    let mut out = ndarray::Array2::<f32>::zeros((ids.len(), cols));
    ndarray::hpc::gather::gather_rows_f32(table, &ids, out.view_mut())
        .map_err(|e| runtime(n, format!("{e:?}")))?;
    let mut shape = idx_v.shape();
    shape.push(cols);
    let out = out
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|e| runtime(n, e.to_string()))?;
    Ok(vec![Value::F32(out)])
}

/// `Squeeze` (opset ≥ 13, axes as input 1): the listed axes, each of size 1;
/// absent axes, every size-1 axis.
fn squeeze_shape(shape: &[usize], axes: Option<&[i64]>) -> Option<Vec<usize>> {
    let r = shape.len();
    let drop: Vec<usize> = match axes {
        Some(a) => a.iter().map(|&x| norm_axis(x, r)).collect::<Option<_>>()?,
        None => (0..r).filter(|&i| shape[i] == 1).collect(),
    };
    if drop.iter().any(|&i| shape[i] != 1) {
        return None;
    }
    Some(
        (0..r)
            .filter(|i| !drop.contains(i))
            .map(|i| shape[i])
            .collect(),
    )
}

fn guard_squeeze(v: &NodeView) -> K3 {
    let Some(shape) = arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) else {
        return K3::Unknown;
    };
    let present = v.node.inputs.get(1).is_some_and(|s| !s.is_empty());
    let axes = const_i64s(v, 1);
    if present && axes.is_none() {
        return K3::Unknown;
    }
    match squeeze_shape(&shape, axes.as_deref()) {
        Some(_) => K3::True,
        None => K3::False,
    }
}

fn do_squeeze(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let axes = a.get(1).and_then(|v| v.i64s());
    let shape = squeeze_shape(&x.shape(), axes.as_deref())
        .ok_or_else(|| runtime(n, "squeezed axis is not of size 1"))?;
    Ok(vec![reshape_value(x, &shape, n)?])
}

const FLATTEN_AXIS: &[AttrRule] = &[AttrRule {
    name: "axis",
    value: None,
    default: Some(1),
}];

fn flatten_shape(shape: &[usize], axis: i64) -> Option<Vec<usize>> {
    let r = shape.len() as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if !(0..=r).contains(&a) {
        return None;
    }
    let a = a as usize;
    Some(vec![
        shape[..a].iter().product(),
        shape[a..].iter().product(),
    ])
}

fn guard_flatten(v: &NodeView) -> K3 {
    match arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) {
        Some(s) => match flatten_shape(&s, int_attr(v.node, "axis", 1)) {
            Some(_) => K3::True,
            None => K3::False,
        },
        None => K3::Unknown,
    }
}

fn do_flatten(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = a.first().ok_or_else(|| runtime(n, "no input"))?;
    let shape = flatten_shape(&x.shape(), int_attr(n, "axis", 1))
        .ok_or_else(|| runtime(n, "axis out of range"))?;
    Ok(vec![reshape_value(x, &shape, n)?])
}

/// `ReduceSum` over int64, axes as input 1 (absent: all), each lane folded
/// with `checked_i64(Add)`: an overflow refuses rather than wraps.
fn guard_reduce_sum_i64(v: &NodeView) -> K3 {
    let Some(shape) = arg_shape(v, 0).and_then(|s| known(&Some(s.clone()))) else {
        return K3::Unknown;
    };
    let present = v.node.inputs.get(1).is_some_and(|s| !s.is_empty());
    match (present, const_i64s(v, 1)) {
        (false, _) => K3::True,
        (true, None) => K3::Unknown,
        (true, Some(ax)) if ax.iter().all(|&x| norm_axis(x, shape.len()).is_some()) => K3::True,
        _ => K3::False,
    }
}

fn do_reduce_sum_i64(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    use ndarray::hpc::checked_i64::{checked_i64, I64Op};
    let mut x = i64_arg(a, 0, n)?.clone();
    let r = x.ndim();
    let mut axes: Vec<usize> = match a.get(1).and_then(|v| v.i64s()) {
        Some(ax) => ax
            .iter()
            .map(|&v| norm_axis(v, r).ok_or_else(|| runtime(n, "axis out of range")))
            .collect::<Result<_, _>>()?,
        None => (0..r).collect(),
    };
    axes.sort_unstable();
    axes.dedup();
    let keep = int_attr(n, "keepdims", 1) != 0;
    for &ax in axes.iter().rev() {
        let mut lanes = Vec::new();
        for lane in x.lanes(Axis(ax)) {
            let mut acc = 0i64;
            for &v in lane {
                acc = checked_i64(I64Op::Add, acc, v).map_err(|e| runtime(n, format!("{e:?}")))?;
            }
            lanes.push(acc);
        }
        let mut shape = x.shape().to_vec();
        shape.remove(ax);
        x = ArrayD::from_shape_vec(IxDyn(&shape), lanes).map_err(|e| runtime(n, e.to_string()))?;
        if keep {
            x.insert_axis_inplace(Axis(ax));
        }
    }
    Ok(vec![Value::I64(x)])
}

/// `Cast` int64 → bool: nonzero is true.
fn do_cast_i64_bool(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    let x = i64_arg(a, 0, n)?;
    Ok(vec![pack_bools(&x.mapv(|v| v != 0))])
}

/// `GatherElements` over f32 with i64 indices of the data's rank: the output
/// has the indices' shape, and element `ix` reads the data at `ix` with the
/// axis coordinate replaced by `indices[ix]`.
fn guard_gather_elements(v: &NodeView) -> K3 {
    let (Some(d), Some(i)) = (arg_shape(v, 0), arg_shape(v, 1)) else {
        return K3::Unknown;
    };
    if d.len() != i.len() || norm_axis(int_attr(v.node, "axis", 0), d.len()).is_none() {
        return K3::False;
    }
    match v.args.get(1).copied().flatten().and_then(|f| f.dtype) {
        Some(DType::I64) => K3::True,
        Some(_) => K3::False,
        None => K3::Unknown,
    }
}

fn do_gather_elements_f32(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {
    use ndarray::Dimension as _;
    let data = f32_arg(a, 0, n)?;
    let idx = i64_arg(a, 1, n)?;
    let axis = norm_axis(int_attr(n, "axis", 0), data.ndim())
        .ok_or_else(|| runtime(n, "axis out of range"))?;
    if idx.ndim() != data.ndim() {
        return Err(runtime(n, "indices rank differs from data rank"));
    }
    let d = data.shape()[axis] as i64;
    for (ix, &k) in idx.indexed_iter() {
        if !(-d..d).contains(&k) {
            return Err(runtime(n, format!("index {k} outside [-{d}, {d})")));
        }
        for (ax, (&i, &dim)) in ix.slice().iter().zip(data.shape()).enumerate() {
            if ax != axis && i >= dim {
                return Err(runtime(n, "indices exceed the data on a non-gather axis"));
            }
        }
    }
    let out = ArrayD::from_shape_fn(idx.raw_dim(), |ix| {
        let mut j = ix.clone();
        j[axis] = idx[&ix].rem_euclid(d) as usize;
        data[&j]
    });
    Ok(vec![Value::F32(out)])
}

pub(crate) const OPSETS: RangeInclusive<i64> = 13..=18;
pub(crate) const NO_ATTRS: &[AttrRule] = &[];
const CAST_TO_F32: &[AttrRule] = &[AttrRule {
    name: "to",
    value: Some(1),
    default: None,
}];
const CAST_TO_BOOL: &[AttrRule] = &[AttrRule {
    name: "to",
    value: Some(9),
    default: None,
}];
/// `perm` is any list; its guard checks it is a permutation. The default
/// `-1` only marks the attribute optional (absent means reversed axes); no
/// body reads it.
const TRANSPOSE_PERM: &[AttrRule] = &[AttrRule {
    name: "perm",
    value: None,
    default: Some(-1),
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
    ($m:expr, $op:expr, $arg:expr, $dt:expr, $attrs:expr, $when:path, $do:path, $num:expr) => {
        (
            Method {
                name: $m,
                do_: $do,
                numeric: $num,
            },
            Registration {
                domain: "ai.onnx",
                op_type: $op,
                dtype_arg: $arg,
                dtype: $dt,
                attrs: $attrs,
                opsets: OPSETS,
                when: $when,
                guard: stringify!($when),
                method: 0,
            },
        )
    };
}

/// The lab registry: every op the fixtures use, nothing else.
pub fn registry() -> Registry {
    use DType::*;
    use Numeric::*;
    let mut drafts = vec![
            reg!(
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
                "simd::eq_i64_pair_to_mask / eq_i64_to_mask (ndarray broadcast first when shapes differ)",
                "Equal",
                Some(0),
                I64,
                NO_ATTRS,
                guard_compare_i64,
                do_equal_i64,
                Exact
            ),
            reg!(
                "simd::le_i64_pair_to_mask / le_i64_to_mask (ndarray broadcast first when shapes differ)",
                "LessOrEqual",
                Some(0),
                I64,
                NO_ATTRS,
                guard_compare_i64,
                do_less_or_equal_i64,
                Exact
            ),
            reg!(
                "simd::blend_i64 / blend_i64_scalar",
                "Where",
                Some(1),
                I64,
                NO_ATTRS,
                guard_where,
                do_where_i64,
                Exact
            ),
            reg!(
                "ndarray ArrayD::from_elem of the value attribute",
                "ConstantOfShape",
                Some(0),
                I64,
                NO_ATTRS,
                guard_constant_of_shape,
                do_constant_of_shape,
                Exact
            ),
            reg!(
                "ndarray broadcast, multidirectional",
                "Expand",
                Some(0),
                I64,
                NO_ATTRS,
                guard_expand,
                do_expand,
                Exact
            ),
            reg!(
                "ndarray broadcast, multidirectional",
                "Expand",
                Some(0),
                F32,
                NO_ATTRS,
                guard_expand,
                do_expand,
                Exact
            ),
            reg!(
                "hpc::checked_i64::checked_binary_i64(Add)",
                "Add",
                Some(0),
                I64,
                NO_ATTRS,
                guard_binary_i64,
                do_add_i64,
                Exact
            ),
            reg!(
                "hpc::checked_i64::checked_binary_i64(Sub)",
                "Sub",
                Some(0),
                I64,
                NO_ATTRS,
                guard_binary_i64,
                do_sub_i64,
                Exact
            ),
            reg!(
                "hpc::checked_i64::checked_binary_i64(Mul)",
                "Mul",
                Some(0),
                I64,
                NO_ATTRS,
                guard_binary_i64,
                do_mul_i64,
                Exact
            ),
            reg!(
                "hpc::checked_i64::checked_binary_i64(Div), truncating",
                "Div",
                Some(0),
                I64,
                NO_ATTRS,
                guard_binary_i64,
                do_div_i64,
                Exact
            ),
            reg!(
                "hpc::checked_i64::range_i64, capacity RANGE_CAPACITY",
                "Range",
                Some(0),
                I64,
                NO_ATTRS,
                guard_range,
                do_range_i64,
                Exact
            ),
            reg!(
                "Cast i64 -> f32: as f32 (round to nearest even)",
                "Cast",
                Some(0),
                I64,
                CAST_TO_F32,
                guard_true,
                do_cast_i64_f32,
                Exact
            ),
            reg!(
                "Cast bool -> bool: identity",
                "Cast",
                Some(0),
                Bool,
                CAST_TO_BOOL,
                guard_cast_bool_identity,
                do_identity,
                Exact
            ),
            reg!(
                "simd::matmul_f32_batched (f32 sgemm, never AMX)",
                "MatMul",
                Some(0),
                F32,
                NO_ATTRS,
                guard_matmul,
                do_matmul_f32,
                Bounded("sgemm accumulation order; exact when K = 1")
            ),
            reg!(
                "ndarray permuted_axes, standard layout",
                "Transpose",
                Some(0),
                F32,
                TRANSPOSE_PERM,
                guard_transpose,
                do_transpose,
                Exact
            ),
            reg!(
                "ndarray permuted_axes, standard layout",
                "Transpose",
                Some(0),
                I64,
                TRANSPOSE_PERM,
                guard_transpose,
                do_transpose,
                Exact
            ),
            reg!(
                "ndarray::hpc::vml::vscos",
                "Cos",
                Some(0),
                F32,
                NO_ATTRS,
                guard_true,
                do_cos,
                Bounded("libm f32 cos per lane; ULP against ORT measured in tests")
            ),
            reg!(
                "ndarray::hpc::vml::vssin",
                "Sin",
                Some(0),
                F32,
                NO_ATTRS,
                guard_true,
                do_sin,
                Bounded("libm f32 sin per lane; ULP against ORT measured in tests")
            ),
            reg!(
                "simd::mask_and on packed tensor(bool) (ndarray broadcast first when shapes differ)",
                "And",
                Some(0),
                Bool,
                NO_ATTRS,
                guard_and,
                do_and_bool,
                Exact
            ),
            reg!(
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
                "ndarray reshape, allowzero 0",
                "Reshape",
                Some(0),
                F32,
                ALLOWZERO_0,
                guard_reshape,
                do_reshape,
                Exact
            ),
        reg!("ndarray slice_each_axis, positive steps", "Slice", Some(0), F32, NO_ATTRS, guard_slice, do_slice, Exact),
        reg!("ndarray slice_each_axis, positive steps", "Slice", Some(0), Bool, NO_ATTRS, guard_slice, do_slice, Exact),
        reg!("simd::blend_f32 / blend_f32_scalar", "Where", Some(1), F32, NO_ATTRS, guard_where_f32, do_where_f32, Exact),
        reg!("hpc::activations::softmax_axis_f32 on [rows, last]", "Softmax", Some(0), F32, SOFTMAX_LAST, guard_known_shape, do_softmax_last,
            Bounded("exp and a sum per row; ULP against ORT measured in tests")),
        reg!("hpc::gather::gather_rows_f32", "Gather", Some(0), F32, GATHER_AXIS, guard_gather_rows, do_gather_rows_f32, Exact),
        reg!("ndarray reshape: drop size-1 axes", "Squeeze", Some(0), F32, NO_ATTRS, guard_squeeze, do_squeeze, Exact),
        reg!("ndarray reshape to [prod(..axis), prod(axis..)]", "Flatten", Some(0), Bool, FLATTEN_AXIS, guard_flatten, do_flatten, Exact),
        reg!("checked_i64(Add) folded per lane, keepdims 0", "ReduceSum", Some(0), I64, KEEPDIMS_0, guard_reduce_sum_i64, do_reduce_sum_i64, Exact),
        reg!("checked_i64(Add) folded per lane, keepdims 1", "ReduceSum", Some(0), I64, KEEPDIMS_1, guard_reduce_sum_i64, do_reduce_sum_i64, Exact),
        reg!("Cast i64 -> bool: nonzero", "Cast", Some(0), I64, CAST_TO_BOOL, guard_true, do_cast_i64_bool, Exact),
        reg!("index walk: data at indices along axis", "GatherElements", Some(0), F32, GATHER_AXIS, guard_gather_elements, do_gather_elements_f32, Exact),
    ];
    drafts.extend(crate::generated::drafts());
    Registry::build(ONNX_LENS, 8, drafts)
}

/// The hand-written fields alone, without `generated`. The generator triages
/// against this table, so regenerating is a fixed point: its own output never
/// hides the gaps it was generated from.
pub fn hand_written() -> Registry {
    let mut r = registry();
    let n = r.regs.len() - crate::generated::drafts().len();
    r.regs.truncate(n);
    r.exposed = Registry::build(r.lens, r.version, Vec::new()).exposed;
    for i in 0..n {
        r.expose(i, true);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codebook is not capped at one 256-position mask: 600 fields span
    /// three exposure blocks, every one resolves, and hiding a field in a
    /// later block hides that field only and moves the revision.
    #[test]
    fn codebook_spans_mask_blocks() {
        let base = registry();
        let drafts: Vec<(Method, Registration)> = (0..600)
            .map(|i| {
                let r = &base.regs[i % base.regs.len()];
                (*base.method(r), r.clone())
            })
            .collect();
        let mut big = Registry::build(ONNX_LENS, 1, drafts);
        assert_eq!(big.exposed.len(), 3);
        assert_eq!(big.methods.len(), base.methods.len());
        let key = |o: u16| MethodKey {
            lens: ONNX_LENS,
            ordinal: o,
        };
        assert!((0..600u16).all(|o| big.get(key(o)).is_some()));
        assert!(big.get(key(600)).is_none());
        let before = big.revision();
        big.expose(599, false);
        assert!(big.get(key(599)).is_none());
        assert!(big.get(key(599 - 256)).is_some() && big.get(key(598)).is_some());
        assert_ne!(big.revision(), before);
        big.expose(599, true);
        assert_eq!(big.revision(), before);
    }

    #[test]
    fn rotary_guards_fire_and_stay_silent() {
        use Dim::*;
        let k = |d: &[usize]| f(d.iter().map(|&x| Known(x)).collect());
        let two = |op: &str, g: Guard, a: Fact, b: Fact| {
            let n = node(op, 2, vec![]);
            g(&NodeView {
                node: &n,
                args: vec![Some(&a), Some(&b)],
                consts: vec![None, None],
            })
        };
        // MatMul: a rank-2 weight, equal leading dims, or a shared right batch.
        assert_eq!(
            two("MatMul", guard_matmul, k(&[2, 3]), k(&[3, 4])),
            K3::True
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[1, 10, 8]), k(&[8, 4])),
            K3::True
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[1, 2, 5, 8]), k(&[1, 2, 8, 5])),
            K3::True
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[5, 2, 3]), k(&[1, 3, 4])),
            K3::True
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[2, 3]), k(&[2, 4])),
            K3::False
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[1, 2, 3]), k(&[5, 3, 4])),
            K3::False
        );
        assert_eq!(
            two("MatMul", guard_matmul, k(&[2, 3]), k(&[1, 3, 4])),
            K3::False
        );
        assert_eq!(two("MatMul", guard_matmul, k(&[3]), k(&[3, 4])), K3::False);
        // Compares: pair, scalar and any multidirectional broadcast.
        assert_eq!(
            two("LessOrEqual", guard_compare_i64, k(&[1, 10]), k(&[10, 1])),
            K3::True
        );
        assert_eq!(
            two("LessOrEqual", guard_compare_i64, k(&[2, 3]), k(&[4, 3])),
            K3::False
        );
        assert_eq!(
            two(
                "LessOrEqual",
                guard_compare_i64,
                k(&[2]),
                f(vec![Symbol("N".into())])
            ),
            K3::Unknown
        );
        // And: any two known shapes that broadcast.
        assert_eq!(two("And", guard_and, k(&[3, 3]), k(&[3, 3])), K3::True);
        assert_eq!(two("And", guard_and, k(&[]), k(&[1, 1, 3, 3])), K3::True);
        assert_eq!(two("And", guard_and, k(&[3, 3]), k(&[2, 3])), K3::False);
        // Transpose: perm must permute the rank; absent means reversed.
        let x = k(&[1, 2, 3]);
        let t = |attrs: Vec<(String, Attr)>| {
            let n = node("Transpose", 1, attrs);
            guard_transpose(&NodeView {
                node: &n,
                args: vec![Some(&x)],
                consts: vec![None],
            })
        };
        assert_eq!(
            t(vec![("perm".into(), Attr::Ints(vec![0, 2, 1]))]),
            K3::True
        );
        assert_eq!(t(vec![]), K3::True);
        assert_eq!(
            t(vec![("perm".into(), Attr::Ints(vec![0, 1, 1]))]),
            K3::False
        );
        assert_eq!(t(vec![("perm".into(), Attr::Ints(vec![0, 1]))]), K3::False);
        assert_eq!(
            t(vec![("perm".into(), Attr::Ints(vec![0, 1, 3]))]),
            K3::False
        );
    }

    #[test]
    fn rotary_dos() {
        let fv =
            |s: &[usize], v: Vec<f32>| Value::F32(ArrayD::from_shape_vec(IxDyn(s), v).unwrap());
        let iv =
            |s: &[usize], v: Vec<i64>| Value::I64(ArrayD::from_shape_vec(IxDyn(s), v).unwrap());
        let n = |op: &str| node(op, 2, vec![]);
        // MatMul with K = 3 against the naive product.
        let a = fv(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fv(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        match &do_matmul_f32(&[&a, &b], &n("MatMul")).unwrap()[0] {
            Value::F32(o) => assert_eq!(
                o.iter().copied().collect::<Vec<_>>(),
                [58.0, 64.0, 139.0, 154.0]
            ),
            other => panic!("{other:?}"),
        }
        // Batched with a shared right operand.
        let a3 = fv(&[2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b3 = fv(&[1, 2, 1], vec![10.0, 1.0]);
        match &do_matmul_f32(&[&a3, &b3], &n("MatMul")).unwrap()[0] {
            Value::F32(o) => {
                assert_eq!(o.shape(), [2, 1, 1]);
                assert_eq!(o.iter().copied().collect::<Vec<_>>(), [12.0, 34.0]);
            }
            other => panic!("{other:?}"),
        }
        // Transpose [0, 2, 1] moves elements, not just the shape.
        let x = iv(&[1, 2, 3], vec![0, 1, 2, 3, 4, 5]);
        let tn = node(
            "Transpose",
            1,
            vec![("perm".into(), Attr::Ints(vec![0, 2, 1]))],
        );
        match &do_transpose(&[&x], &tn).unwrap()[0] {
            Value::I64(o) => {
                assert_eq!(o.shape(), [1, 3, 2]);
                assert_eq!(o.iter().copied().collect::<Vec<_>>(), [0, 3, 1, 4, 2, 5]);
            }
            other => panic!("{other:?}"),
        }
        // Broadcast compare [1, 3] <= [3, 1]: lower triangle, tail bits clear.
        let k = iv(&[1, 3], vec![0, 1, 2]);
        let q = iv(&[3, 1], vec![0, 1, 2]);
        match &do_less_or_equal_i64(&[&k, &q], &n("LessOrEqual")).unwrap()[0] {
            Value::Bool { shape, words } => {
                assert_eq!(shape, &[3, 3]);
                // Row i holds j <= i: bits 0; 3, 4; 6, 7, 8.
                assert_eq!(words, &[0b1_1101_1001]);
            }
            other => panic!("{other:?}"),
        }
        // A rank-3 left against a rank-2 weight; two batches, so the weight
        // must be applied to every flattened row, not only the first batch.
        let a3w = fv(&[2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        match &do_matmul_f32(&[&a3w, &b], &n("MatMul")).unwrap()[0] {
            Value::F32(o) => {
                assert_eq!(o.shape(), [2, 1, 2]);
                assert_eq!(
                    o.iter().copied().collect::<Vec<_>>(),
                    [58.0, 64.0, 139.0, 154.0]
                );
            }
            other => panic!("{other:?}"),
        }
        // And broadcasts a row [1, 3] against a [2, 3] block.
        let row = Value::Bool {
            shape: vec![1, 3],
            words: vec![0b101],
        };
        let blk = Value::Bool {
            shape: vec![2, 3],
            words: vec![0b110_111],
        };
        match &do_and_bool(&[&row, &blk], &n("And")).unwrap()[0] {
            Value::Bool { shape, words } => {
                assert_eq!(shape, &[2, 3]);
                assert_eq!(words, &[0b100_101]);
            }
            other => panic!("{other:?}"),
        }
        // i64 -> f32 rounds to nearest even.
        match &do_cast_i64_f32(&[&iv(&[2], vec![16_777_217, -3])], &n("Cast")).unwrap()[0] {
            Value::F32(o) => {
                assert_eq!(o.iter().copied().collect::<Vec<_>>(), [16_777_216.0, -3.0])
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn checked_i64_guards_fire_and_stay_silent() {
        let fi = |d: &[usize]| Fact {
            dtype: Some(DType::I64),
            shape: Some(d.iter().map(|&x| Dim::Known(x)).collect()),
        };
        let sc = |x: i64| Value::I64(ArrayD::from_shape_vec(IxDyn(&[]), vec![x]).unwrap());
        let r = node("Range", 3, vec![]);
        let (s0, d0, d1) = (fi(&[]), sc(0), sc(1));
        let view = |d: &Value| -> K3 {
            guard_range(&NodeView {
                node: &r,
                args: vec![Some(&s0), Some(&s0), Some(&s0)],
                consts: vec![None, None, Some(d)],
            })
        };
        assert_eq!(view(&d0), K3::False, "a constant zero step");
        assert_eq!(view(&d1), K3::True);
        let b = node("Add", 2, vec![]);
        let g = |x: &[usize], y: &[usize]| {
            let (fx, fy) = (fi(x), fi(y));
            guard_binary_i64(&NodeView {
                node: &b,
                args: vec![Some(&fx), Some(&fy)],
                consts: vec![None, None],
            })
        };
        assert_eq!(g(&[2], &[2]), K3::True);
        assert_eq!(g(&[2], &[]), K3::True);
        assert_eq!(g(&[1], &[2, 3]), K3::True, "a one-element side broadcasts");
        assert_eq!(g(&[2], &[2, 1]), K3::False);
        assert_eq!(g(&[2], &[3]), K3::False);
    }

    #[test]
    fn checked_i64_dos_refuse() {
        let n = |op: &str| node(op, 2, vec![]);
        let v = |x: Vec<i64>| Value::I64(ArrayD::from_shape_vec(IxDyn(&[x.len()]), x).unwrap());
        let s = |x: i64| Value::I64(ArrayD::from_shape_vec(IxDyn(&[]), vec![x]).unwrap());
        let what = |r: Result<Vec<Value>, Refusal>| match r {
            Err(Refusal::Runtime { what, .. }) => what,
            other => panic!("{other:?}"),
        };
        let (a, z, big) = (v(vec![6, 3]), v(vec![1, 0]), v(vec![i64::MAX, 1]));
        assert!(what(do_div_i64(&[&a, &z], &n("Div"))).contains("by zero"));
        assert!(what(do_mul_i64(&[&big, &v(vec![2])], &n("Mul"))).contains("overflows"));
        assert!(what(do_add_i64(&[&big, &v(vec![1])], &n("Add"))).contains("overflows"));
        assert!(what(do_range_i64(&[&s(0), &s(3), &s(0)], &n("Range"))).contains("delta 0"));
        let cap = RANGE_CAPACITY as i64;
        assert!(what(do_range_i64(&[&s(0), &s(cap + 1), &s(1)], &n("Range"))).contains("elements"));
        // Div truncates toward zero; a one-element left side broadcasts.
        match &do_div_i64(&[&v(vec![-7]), &v(vec![2, -2])], &n("Div")).unwrap()[0] {
            Value::I64(o) => assert_eq!(o.iter().copied().collect::<Vec<_>>(), [-3, 3]),
            other => panic!("{other:?}"),
        }
    }

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

    /// `Shape`'s `start`/`end` (opset 15): negative counts from the end,
    /// both clamp to `[0, rank]`, an empty range is an empty shape.
    #[test]
    fn shape_start_end_clamp() {
        let at = |a: Vec<(String, Attr)>| shape_slice(&[2, 3, 5, 7], &node("Shape", 1, a));
        assert_eq!(at(vec![]), [2, 3, 5, 7]);
        assert_eq!(
            at(vec![
                ("start".into(), Attr::Int(1)),
                ("end".into(), Attr::Int(-1))
            ]),
            [3, 5]
        );
        assert_eq!(at(vec![("start".into(), Attr::Int(-10))]), [2, 3, 5, 7]);
        assert_eq!(at(vec![("end".into(), Attr::Int(10))]), [2, 3, 5, 7]);
        assert_eq!(
            at(vec![
                ("start".into(), Attr::Int(3)),
                ("end".into(), Attr::Int(1))
            ]),
            [] as [i64; 0]
        );
    }

    fn i64v(shape: &[usize], v: Vec<i64>) -> Value {
        Value::I64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap())
    }

    fn bits_of(v: &Value) -> (Vec<usize>, Vec<u64>) {
        match v {
            Value::Bool { shape, words } => (shape.clone(), words.clone()),
            other => panic!("{other:?}"),
        }
    }

    /// 70 elements cross a mask word: the pair and scalar compares set the
    /// right bits, and no bit past element 69 is set (tail bits clear).
    #[test]
    fn int64_compares_pack_tail_bits() {
        let n = node("Equal", 2, vec![]);
        let xs: Vec<i64> = (0..70).map(|i| i % 3).collect();
        let x = i64v(&[70], xs.clone());
        let pair = i64v(
            &[70],
            (0..70).map(|i| if i == 69 { 2 } else { 0 }).collect(),
        );
        let (shape, w) = bits_of(&do_equal_i64(&[&x, &pair], &n).unwrap()[0]);
        assert_eq!(shape, [70]);
        assert_eq!(w.len(), 2);
        for i in 0..70 {
            let want = xs[i] == if i == 69 { 2 } else { 0 };
            assert_eq!((w[i / 64] >> (i % 64)) & 1 == 1, want, "pair {i}");
        }
        assert_eq!(w[1] >> 6, 0, "tail bits past element 69");
        let one = i64v(&[], vec![1]);
        let (_, w) = bits_of(&do_less_or_equal_i64(&[&x, &one], &n).unwrap()[0]);
        for i in 0..70 {
            assert_eq!((w[i / 64] >> (i % 64)) & 1 == 1, xs[i] <= 1, "scalar {i}");
        }
        assert_eq!(w[1] >> 6, 0);
        let bad = i64v(&[2], vec![1, 2]);
        assert!(
            do_equal_i64(&[&x, &bad], &n).is_err(),
            "a non-admitted broadcast refuses at run time too"
        );
    }

    /// int64 `Where` over a 70-element mask, scalar and tensor `Y`.
    #[test]
    fn int64_where_scalar_and_tensor() {
        let n = node("Where", 3, vec![]);
        let mut words = vec![0u64; 2];
        for i in (0..70).filter(|i| i % 5 == 0) {
            words[i / 64] |= 1 << (i % 64);
        }
        let c = Value::Bool {
            shape: vec![70],
            words,
        };
        let x = i64v(&[70], (0..70).collect());
        let ys = i64v(&[70], (0..70).map(|i| -i).collect());
        let s = i64v(&[], vec![i64::MIN]);
        for (y, f) in [
            (&s, (|_: i64| i64::MIN) as fn(i64) -> i64),
            (&ys, |i: i64| -i),
        ] {
            let out = do_where_i64(&[&c, &x, y], &n).unwrap();
            let Value::I64(o) = &out[0] else { panic!() };
            for (i, &v) in o.iter().enumerate() {
                assert_eq!(v, if i % 5 == 0 { i as i64 } else { f(i as i64) }, "{i}");
            }
        }
    }

    /// ONNX `ConstantOfShape`: the value attribute's one element (default
    /// float 0), an empty shape gives a scalar, a zero dimension an empty
    /// tensor, a negative dimension or a multi-element value is FALSE, a
    /// non-constant shape UNKNOWN; a bool value fills the mask without tail
    /// bits.
    #[test]
    fn constant_of_shape_semantics() {
        let with = |v: Option<Value>| {
            node(
                "ConstantOfShape",
                1,
                v.map(|v| vec![("value".to_string(), Attr::Tensor(Some(v)))])
                    .unwrap_or_default(),
            )
        };
        let run = |n: &SourceNode, d: Vec<i64>| {
            do_constant_of_shape(&[&i64v(&[d.len()], d)], n)
                .unwrap()
                .remove(0)
        };
        assert_eq!(
            run(&with(None), vec![2]),
            Value::F32(ArrayD::zeros(IxDyn(&[2])))
        );
        assert_eq!(
            run(&with(Some(i64v(&[1], vec![5]))), vec![]),
            i64v(&[], vec![5]),
            "empty shape: a scalar"
        );
        assert_eq!(
            run(&with(Some(i64v(&[1], vec![5]))), vec![2, 0]).shape(),
            [2, 0]
        );
        let tru = Value::Bool {
            shape: vec![1],
            words: vec![1],
        };
        let (shape, w) = bits_of(&run(&with(Some(tru)), vec![70]));
        assert_eq!((shape, w), (vec![70], vec![u64::MAX, (1 << 6) - 1]));
        let guard = |n: &SourceNode, d: Option<Vec<i64>>| {
            let c = d.map(|d| i64v(&[d.len()], d));
            let fact = Fact {
                dtype: Some(DType::I64),
                shape: Some(vec![Dim::Known(1)]),
            };
            guard_constant_of_shape(&NodeView {
                node: n,
                args: vec![Some(&fact)],
                consts: vec![c.as_ref()],
            })
        };
        assert_eq!(guard(&with(None), Some(vec![2, 3])), K3::True);
        assert_eq!(guard(&with(None), Some(vec![-1])), K3::False);
        assert_eq!(
            guard(&with(Some(i64v(&[2], vec![1, 2]))), Some(vec![2])),
            K3::False
        );
        assert_eq!(guard(&with(None), None), K3::Unknown);
        let rank2 = Fact {
            dtype: Some(DType::I64),
            shape: Some(vec![Dim::Known(1), Dim::Known(2)]),
        };
        let c = i64v(&[1, 2], vec![2, 3]);
        let n = with(None);
        assert_eq!(
            guard_constant_of_shape(&NodeView {
                node: &n,
                args: vec![Some(&rank2)],
                consts: vec![Some(&c)]
            }),
            K3::False,
            "the shape input is 1-D"
        );
    }

    /// ONNX `Expand`: multidirectional broadcast of the input shape and the
    /// target, so a target dimension 1 keeps the input's and a shorter
    /// target is padded on the left.
    #[test]
    fn expand_semantics() {
        let n = node("Expand", 2, vec![]);
        let x = i64v(&[3, 1], vec![1, 2, 3]);
        let out = do_expand(&[&x, &i64v(&[2], vec![1, 4])], &n)
            .unwrap()
            .remove(0);
        assert_eq!(out.shape(), [3, 4], "target 1 keeps the input dimension 3");
        assert_eq!(out.i64s().unwrap()[..4], [1, 1, 1, 1]);
        let out = do_expand(&[&x, &i64v(&[3], vec![2, 1, 1])], &n)
            .unwrap()
            .remove(0);
        assert_eq!(out.shape(), [2, 3, 1]);
        assert!(
            do_expand(&[&x, &i64v(&[2], vec![2, 1])], &n).is_err(),
            "3 against 2"
        );
        assert!(do_expand(&[&x, &i64v(&[1], vec![-1])], &n).is_err());
    }

    /// int64 compares admit a same-shape or a one-element right side only.
    #[test]
    fn pair_or_scalar_admission() {
        use Dim::*;
        let n = node("Equal", 2, vec![]);
        let check = |a: Vec<Dim>, b: Vec<Dim>| {
            let (fa, fb) = (f(a), f(b));
            guard_pair_or_scalar(&NodeView {
                node: &n,
                args: vec![Some(&fa), Some(&fb)],
                consts: vec![None, None],
            })
        };
        assert_eq!(check(vec![Known(3)], vec![Known(3)]), K3::True);
        assert_eq!(check(vec![Known(3)], vec![Known(1)]), K3::True);
        assert_eq!(check(vec![Known(3)], vec![]), K3::True);
        assert_eq!(
            check(vec![Known(3), Known(1)], vec![Known(1), Known(3)]),
            K3::False,
            "the scalar/pair kernels alone do not admit a general broadcast"
        );
        assert_eq!(check(vec![Symbol("N".into())], vec![Known(3)]), K3::Unknown);
    }

    #[test]
    fn broadcast_shapes_rules() {
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 2, 3]), Some(vec![2, 2, 3]));
        assert_eq!(broadcast_shapes(&[3], &[1]), Some(vec![3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[4, 3]), None);
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
