//! Admission: match one typed subgraph against a pre-written native method and
//! prepare its recipe, or refuse with a typed reason.
//!
//! There is no general lowering here. Each method is a fixed pattern over
//! Tract's typed graph (before declutter, where every source node still has a
//! named counterpart) and produces a concrete recipe struct whose `run` calls
//! existing native functions directly. Nothing reads the graph at run time.
//!
//! Tract supplies structure, shapes and constant values. The source inventory
//! supplies what Tract does not keep: the operator domain, the opset, and the
//! identity and digest of every initializer the recipe consumed.

use std::sync::Arc;

use ndarray::{Array1, ArrayD};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::{
    cast::Cast, change_axes::AxisOp, konst::Const, logic::Iff, nn::Reduce, nn::Reducer,
};

use crate::source::{Placement, SourceInventory};

/// Opsets the methods were checked against (the Jina export is opset 18).
pub const ADMITTED_OPSETS: std::ops::RangeInclusive<i64> = 13..=18;

/// Why a subgraph is not admitted. Every variant names where it failed.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The cold frontend could not import or type the graph.
    Frontend(String),
    /// A source node outside the default ONNX domain. Tract looks operators up
    /// by `op_type` alone, so it would import it as the default-domain op.
    ForeignDomain { node: String, domain: String },
    /// The default-domain opset is missing or outside [`ADMITTED_OPSETS`].
    Opset(Option<i64>),
    /// A source operator the method does not contain.
    UnsupportedOp { node: String, op_type: String },
    /// The typed graph does not have the method's shape at this node.
    Pattern {
        node: String,
        expected: &'static str,
        found: String,
    },
    /// A rewrite the method relies on does not hold for this graph.
    RewriteAssumption {
        node: String,
        law: &'static str,
        found: String,
    },
    /// A dtype the method does not admit.
    Dtype { node: String, found: String },
    /// A native shape or range refusal from `lance-graph-onnx`.
    Native(lance_graph_onnx::Refusal),
    /// The recipe was prepared from different initializer bytes or a different
    /// model than the one presented.
    StaleGeneration {
        initializer: String,
        prepared: String,
        presented: String,
    },
}

impl From<lance_graph_onnx::Refusal> for Refusal {
    fn from(r: lance_graph_onnx::Refusal) -> Self {
        Refusal::Native(r)
    }
}

/// How exactly a recipe reproduces the source semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Numeric {
    /// Bit-identical to any IEEE-754 / exact-integer implementation.
    Exact,
    /// Contains an operation whose result depends on evaluation order; the
    /// bound is measured against the oracle, not proven.
    Bounded { op: String, reason: &'static str },
}

/// What admission attests about a prepared recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    pub method: &'static str,
    pub model_sha256: String,
    pub opset: i64,
    pub frontend: &'static str,
    /// The matched source nodes, by source name, in graph order.
    pub source_nodes: Vec<String>,
    /// Consumed initializers: (source name, digest of the source placement).
    pub initializers: Vec<(String, String)>,
    pub numeric: Numeric,
}

impl Receipt {
    /// Refuse if `src` is not the generation this recipe was prepared from:
    /// any consumed initializer whose placement or bytes changed.
    pub fn check_generation(&self, src: &SourceInventory) -> Result<(), Refusal> {
        for (name, prepared) in &self.initializers {
            let presented = src
                .initializer(name)
                .map(placement_digest)
                .unwrap_or_default();
            if &presented != prepared {
                return Err(Refusal::StaleGeneration {
                    initializer: name.clone(),
                    prepared: prepared.clone(),
                    presented,
                });
            }
        }
        Ok(())
    }
}

pub const FRONTEND: &str =
    "tract-onnx (AdaWorldAPI/tract main), InferenceModel::into_typed, no declutter";

fn placement_digest(i: &crate::source::SourceInitializer) -> String {
    match &i.placement {
        Placement::Inline { sha256 } => format!("inline:{sha256}"),
        Placement::External {
            location,
            offset,
            length,
        } => format!("external:{location}@{offset}+{length}"),
    }
}

// ── typed-graph helpers ─────────────────────────────────────────────────────

fn node_at(m: &TypedModel, o: OutletId) -> &TypedNode {
    &m.nodes()[o.node]
}

/// The constant feeding `o`, looking through `AddAxis` (Tract's rank fix-ups).
fn konst(m: &TypedModel, mut o: OutletId) -> Option<Arc<Tensor>> {
    loop {
        let n = node_at(m, o);
        if let Some(c) = n.op_as::<Const>() {
            return Some(c.val().clone());
        }
        match n.op_as::<AxisOp>() {
            Some(AxisOp::Add(_)) => o = n.inputs[0],
            _ => return None,
        }
    }
}

fn expect<'m>(m: &'m TypedModel, o: OutletId, op: &'static str) -> Result<&'m TypedNode, Refusal> {
    let n = node_at(m, o);
    if n.op.name() == op {
        Ok(n)
    } else {
        Err(Refusal::Pattern {
            node: n.name.clone(),
            expected: op,
            found: n.op.name().to_string(),
        })
    }
}

fn scalar_f32(
    m: &TypedModel,
    n: &TypedNode,
    o: OutletId,
    law: &'static str,
) -> Result<f32, Refusal> {
    let t = konst(m, o).ok_or_else(|| Refusal::RewriteAssumption {
        node: n.name.clone(),
        law,
        found: "not a constant".into(),
    })?;
    let v = t.to_plain_array_view::<f32>().map_err(|e| Refusal::Dtype {
        node: n.name.clone(),
        found: e.to_string(),
    })?;
    let first = *v.iter().next().ok_or_else(|| Refusal::RewriteAssumption {
        node: n.name.clone(),
        law,
        found: "empty".into(),
    })?;
    if v.iter().all(|x| x.to_bits() == first.to_bits()) {
        Ok(first)
    } else {
        Err(Refusal::RewriteAssumption {
            node: n.name.clone(),
            law,
            found: format!("{v:?}"),
        })
    }
}

fn source_checks(src: &SourceInventory, allowed: &[&str]) -> Result<i64, Refusal> {
    let opset = src
        .default_opset()
        .filter(|v| ADMITTED_OPSETS.contains(v))
        .ok_or(Refusal::Opset(src.default_opset()))?;
    for n in &src.nodes {
        if !n.default_domain() {
            return Err(Refusal::ForeignDomain {
                node: n.name.clone(),
                domain: n.domain.clone(),
            });
        }
        if !allowed.contains(&n.op_type.as_str()) {
            return Err(Refusal::UnsupportedOp {
                node: n.name.clone(),
                op_type: n.op_type.clone(),
            });
        }
    }
    Ok(opset)
}

/// Import and type a model from bytes with Tract, cold. Declutter is not run.
pub fn typed(bytes: &[u8]) -> Result<TypedModel, Refusal> {
    let onnx = tract_onnx::onnx();
    let proto = onnx
        .proto_model_for_read(&mut &*bytes)
        .map_err(|e| Refusal::Frontend(format!("{e:#}")))?;
    onnx.model_for_proto_model(&proto)
        .and_then(|m| m.into_typed())
        .map_err(|e| Refusal::Frontend(format!("{e:#}")))
}

fn names(m: &TypedModel, ids: &[usize]) -> Vec<String> {
    ids.iter().map(|&i| m.nodes()[i].name.clone()).collect()
}

fn initializers(src: &SourceInventory, names: &[&str]) -> Vec<(String, String)> {
    names
        .iter()
        .filter_map(|n| src.initializer(n))
        .map(|i| (i.name.clone(), placement_digest(i)))
        .collect()
}

// ── method 1: the tiny semantic fixture ─────────────────────────────────────

/// `S = ReduceSum(Where(Greater(X, t), X, 0), axes=[0], keepdims=0)` and
/// `Q = Div(Sqrt(Abs(Mul(Add(F, B), F))), B)`, both exact.
///
/// Laws relied on, each checked at admission:
/// - `ReduceSum(Where(c, X, 0)) == masked_sum(X, c)`: the else-value must be
///   the constant `0` and the then-value the same `X` the comparison reads.
/// - `B` is one weight read twice; both reads must be the same constant.
/// - Broadcasting of `B` against `F` is checked by the native adapter.
#[derive(Debug, Clone)]
pub struct TinyRecipe {
    pub t: i32,
    pub b: Array1<f32>,
    reduce: lance_graph_onnx::ReduceSum,
    pub receipt: Receipt,
}

pub fn admit_tiny(src: &SourceInventory, m: &TypedModel) -> Result<TinyRecipe, Refusal> {
    let opset = source_checks(
        src,
        &[
            "Greater",
            "Where",
            "ReduceSum",
            "Add",
            "Mul",
            "Abs",
            "Sqrt",
            "Div",
        ],
    )?;
    let [s, q] = m
        .output_outlets()
        .map_err(|e| Refusal::Frontend(e.to_string()))?[..]
    else {
        return Err(Refusal::Pattern {
            node: "<outputs>".into(),
            expected: "two outputs",
            found: "other".into(),
        });
    };

    // S branch.
    let rm = expect(m, s, "RmAxis")?;
    let red = node_at(m, rm.inputs[0]);
    let op = red.op_as::<Reduce>().ok_or_else(|| Refusal::Pattern {
        node: red.name.clone(),
        expected: "Reduce<Sum>",
        found: red.op.name().to_string(),
    })?;
    if !matches!(op.reducer, Reducer::Sum) || op.axes.as_slice() != [0] {
        return Err(Refusal::Pattern {
            node: red.name.clone(),
            expected: "Reduce<Sum> axes [0]",
            found: red.op.name().to_string(),
        });
    }
    let iff = node_at(m, red.inputs[0]);
    if iff.op_as::<Iff>().is_none() {
        return Err(Refusal::Pattern {
            node: iff.name.clone(),
            expected: "Iff",
            found: iff.op.name().to_string(),
        });
    }
    let gt = expect(m, iff.inputs[0], "GT")?;
    let x = gt.inputs[0];
    let zero_law = "ReduceSum(Where(c, X, Z)) == masked_sum(X, c) needs Z == 0 and the same X";
    if iff.inputs[1] != x {
        return Err(Refusal::RewriteAssumption {
            node: iff.name.clone(),
            law: zero_law,
            found: "then-value is not the compared X".into(),
        });
    }
    let z = konst(m, iff.inputs[2]).ok_or_else(|| Refusal::RewriteAssumption {
        node: iff.name.clone(),
        law: zero_law,
        found: "else-value not constant".into(),
    })?;
    let z = z.to_plain_array_view::<i32>().map_err(|e| Refusal::Dtype {
        node: iff.name.clone(),
        found: e.to_string(),
    })?;
    if z.iter().any(|&v| v != 0) {
        return Err(Refusal::RewriteAssumption {
            node: iff.name.clone(),
            law: zero_law,
            found: format!("Z = {z:?}"),
        });
    }
    let t = konst(m, gt.inputs[1])
        .and_then(|t| {
            t.to_plain_array_view::<i32>()
                .ok()
                .and_then(|v| (v.len() == 1).then(|| v.iter().copied().next().unwrap_or(0)))
        })
        .ok_or_else(|| Refusal::Pattern {
            node: gt.name.clone(),
            expected: "Greater(X, scalar int32 const)",
            found: "other".into(),
        })?;

    // Q branch.
    let div = expect(m, q, "Div")?;
    let sqrt = expect(m, div.inputs[0], "Sqrt")?;
    let abs = expect(m, sqrt.inputs[0], "Abs")?;
    let mul = expect(m, abs.inputs[0], "Mul")?;
    let add = expect(m, mul.inputs[0], "Add")?;
    let f = add.inputs[0];
    if mul.inputs[1] != f {
        return Err(Refusal::Pattern {
            node: mul.name.clone(),
            expected: "Mul(Add(F, B), F)",
            found: "other second operand".into(),
        });
    }
    let b_add = konst(m, add.inputs[1]);
    let b_div = konst(m, div.inputs[1]);
    let (Some(b_add), Some(b_div)) = (b_add, b_div) else {
        return Err(Refusal::Pattern {
            node: add.name.clone(),
            expected: "B constant",
            found: "not constant".into(),
        });
    };
    let flat = |t: &Arc<Tensor>| -> Result<Vec<u32>, Refusal> {
        Ok(t.to_plain_array_view::<f32>()
            .map_err(|e| Refusal::Dtype {
                node: add.name.clone(),
                found: e.to_string(),
            })?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    if flat(&b_add)? != flat(&b_div)? {
        return Err(Refusal::RewriteAssumption {
            node: div.name.clone(),
            law: "B read twice is one weight",
            found: "different constants".into(),
        });
    }
    let b: Array1<f32> =
        Array1::from_iter(b_add.to_plain_array_view::<f32>().unwrap().iter().copied());
    // F's typed shape against B's source shape: broadcasting decided once, here.
    let f_shape: Vec<usize> = m
        .outlet_fact(f)
        .map_err(|e| Refusal::Frontend(e.to_string()))?
        .shape
        .as_concrete()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    lance_graph_onnx::broadcastable(&f_shape, b.shape())?;

    let ids = [
        gt.id, iff.id, red.id, add.id, mul.id, abs.id, sqrt.id, div.id,
    ];
    Ok(TinyRecipe {
        t,
        b,
        reduce: lance_graph_onnx::ReduceSum::new(Some(&[0]), 0, 0)?,
        receipt: Receipt {
            method: "tiny.v0: greater_i32 + masked_sum_i32 (sum_where_greater_i32); add, mul, vsabs, vssqrt, div",
            model_sha256: src.model_sha256.clone(),
            opset,
            frontend: FRONTEND,
            source_nodes: names(m, &ids),
            initializers: initializers(src, &["T", "Z", "axes", "B"]),
            numeric: Numeric::Exact,
        },
    })
}

impl TinyRecipe {
    /// Run the admitted method. Only `lance-graph-onnx` / `ndarray` are called.
    pub fn run(
        &self,
        x: &[i32],
        f: &ndarray::ArrayRef<f32, ndarray::Ix2>,
        scratch: &mut [u64],
    ) -> Result<(lance_graph_onnx::Reduced, ndarray::Array2<f32>), Refusal> {
        use lance_graph_onnx as onnx;
        let s = self.reduce.sum_where_greater_i32(x, self.t, scratch)?;
        let g = onnx::add(f, &self.b)?;
        let h = onnx::mul(&g, f)?;
        let mut a = ndarray::Array2::<f32>::zeros(h.raw_dim());
        onnx::abs_f32(h.view(), a.view_mut());
        let mut y = ndarray::Array2::<f32>::zeros(a.raw_dim());
        onnx::sqrt_f32(a.view(), y.view_mut());
        let q = onnx::div(&y, &self.b)?;
        Ok((s, q))
    }
}

// ── method 2: Qwen3 RMSNorm as exported by the Jina v5 ONNX ─────────────────

/// `Mul_1(w, Cast_1(Mul(x, Div(1, Sqrt(Add(ReduceMean(Pow(x, 2), [-1]), eps))))))`
/// with `x = Cast(input)`, both casts f32 → f32.
///
/// Laws checked at admission: both `Cast`s are identities (`to == F32` on an
/// f32 input); `Pow`'s exponent is exactly `2.0` (so `Pow(x, 2) = x * x`, one
/// correctly rounded multiply); `Div`'s numerator is exactly `1.0`; the mean
/// runs over the last axis only with `keepdims`. The mean itself reassociates
/// a float sum, so the method is [`Numeric::Bounded`], never `Exact`.
#[derive(Debug, Clone)]
pub struct RmsNormRecipe {
    pub eps: f32,
    pub weight: Array1<f32>,
    pub receipt: Receipt,
}

pub fn admit_rmsnorm(src: &SourceInventory, m: &TypedModel) -> Result<RmsNormRecipe, Refusal> {
    let opset = source_checks(
        src,
        &[
            "Cast",
            "Constant",
            "Pow",
            "ReduceMean",
            "Add",
            "Sqrt",
            "Div",
            "Mul",
        ],
    )?;
    let out = m
        .output_outlets()
        .map_err(|e| Refusal::Frontend(e.to_string()))?[0];

    let mul1 = expect(m, out, "Mul")?;
    let weight = konst(m, mul1.inputs[0]).ok_or_else(|| Refusal::Pattern {
        node: mul1.name.clone(),
        expected: "Mul(weight const, ·)",
        found: "not constant".into(),
    })?;
    let cast1 = expect(m, mul1.inputs[1], "Cast")?;
    let mul = expect(m, cast1.inputs[0], "Mul")?;
    let cast = expect(m, mul.inputs[0], "Cast")?;
    let div = expect(m, mul.inputs[1], "Div")?;
    let sqrt = expect(m, div.inputs[1], "Sqrt")?;
    let add = expect(m, sqrt.inputs[0], "Add")?;
    let red = node_at(m, add.inputs[0]);
    let pow = expect(m, red.inputs[0], "Pow")?;

    for c in [cast, cast1] {
        let to = c.op_as::<Cast>().map(|c| c.to);
        let from = m.outlet_fact(c.inputs[0]).map(|f| f.datum_type).ok();
        if to != Some(f32::datum_type()) || from != Some(f32::datum_type()) {
            return Err(Refusal::Dtype {
                node: c.name.clone(),
                found: format!("Cast {from:?} -> {to:?}"),
            });
        }
    }
    if pow.inputs[0] != cast.id.into() {
        return Err(Refusal::Pattern {
            node: pow.name.clone(),
            expected: "Pow(x, 2) of the cast input",
            found: "other base".into(),
        });
    }
    let e = scalar_f32(
        m,
        pow,
        pow.inputs[1],
        "Pow(x, 2) == x * x needs the exponent exactly 2.0",
    )?;
    if e.to_bits() != 2.0f32.to_bits() {
        return Err(Refusal::RewriteAssumption {
            node: pow.name.clone(),
            law: "Pow(x, 2) == x * x needs the exponent exactly 2.0",
            found: format!("{e:e}"),
        });
    }
    let one = scalar_f32(m, div, div.inputs[0], "Div(1, s)")?;
    if one.to_bits() != 1.0f32.to_bits() {
        return Err(Refusal::RewriteAssumption {
            node: div.name.clone(),
            law: "Div(1, s)",
            found: format!("{one:e}"),
        });
    }
    let rank = m.outlet_fact(pow.inputs[0]).map(|f| f.rank()).unwrap_or(0);
    match red.op_as::<Reduce>() {
        Some(r) if matches!(r.reducer, Reducer::Mean) && r.axes.as_slice() == [rank - 1] => {}
        _ => {
            return Err(Refusal::Pattern {
                node: red.name.clone(),
                expected: "Reduce<Mean> over the last axis",
                found: red.op.name().to_string(),
            })
        }
    }
    let eps = scalar_f32(m, add, add.inputs[1], "Add(mean, eps)")?;
    let weight: Array1<f32> = Array1::from_iter(
        weight
            .to_plain_array_view::<f32>()
            .map_err(|e| Refusal::Dtype {
                node: mul1.name.clone(),
                found: e.to_string(),
            })?
            .iter()
            .copied(),
    );
    let hidden = m
        .outlet_fact(cast.id.into())
        .ok()
        .and_then(|f| f.shape.as_concrete().map(|s| s[rank - 1]));
    if hidden != Some(weight.len()) {
        return Err(Refusal::Pattern {
            node: mul1.name.clone(),
            expected: "weight length == hidden",
            found: format!("{hidden:?} vs {}", weight.len()),
        });
    }

    let ids = [
        cast.id, pow.id, red.id, add.id, sqrt.id, div.id, mul.id, cast1.id, mul1.id,
    ];
    let mut source_nodes = names(m, &ids);
    // Tract renames the Mean reduce to `<source>.mean`; map it back.
    for n in &mut source_nodes {
        if let Some(s) = n.strip_suffix(".mean") {
            *n = s.to_string();
        }
    }
    Ok(RmsNormRecipe {
        eps,
        weight,
        receipt: Receipt {
            method: "rmsnorm.f32.v0: mul(x,x); mean_axis (BOUNDED); add eps; vssqrt; div(1,·); mul; mul(weight)",
            model_sha256: src.model_sha256.clone(),
            opset,
            frontend: FRONTEND,
            source_nodes,
            initializers: initializers(src, &["layers.0.input_layernorm.weight"]),
            numeric: Numeric::Bounded { op: "ReduceMean".into(), reason: "float sum over the hidden axis; evaluation order is the implementation's" },
        },
    })
}

impl RmsNormRecipe {
    /// Run over `x` of shape `[..., hidden]`. Returns `(output, mean)`; the
    /// mean is returned so the one bounded step can be compared on its own.
    pub fn run(&self, x: &ArrayD<f32>) -> Result<(ArrayD<f32>, ArrayD<f32>), Refusal> {
        use lance_graph_onnx as onnx;
        let last = ndarray::Axis(x.ndim() - 1);
        let sq = onnx::mul(x, x)?;
        let mean = sq.mean_axis(last).expect("hidden > 0").insert_axis(last);
        let eps = ndarray::arr0(self.eps).into_dyn();
        let shifted = onnx::add(&mean, &eps)?;
        let mut root = ArrayD::<f32>::zeros(shifted.raw_dim());
        onnx::sqrt_f32(shifted.view(), root.view_mut());
        let one = ndarray::arr0(1.0f32).into_dyn();
        let inv = onnx::div(&one, &root)?;
        let normed = onnx::mul(x, &inv)?;
        let out = onnx::mul(&self.weight.view().into_dyn(), &normed)?;
        Ok((out, mean))
    }
}
