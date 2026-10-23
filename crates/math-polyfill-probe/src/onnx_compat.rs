//! ONNX semantic conformance for one composition, WITHOUT an ONNX graph.
//!
//! [`reduce_sum_where_greater`] answers what the ONNX subgraph
//! `ReduceSum(Where(Greater(X, T), X, 0))` answers, for the admitted subset
//! below, by calling the native [`SumWhere`](crate::native::SumWhere). There is
//! no `ModelProto`, no node, no tensor object and no runtime: the operator
//! attributes arrive as plain Rust values and are checked against the ONNX
//! specification (opset 18+ `ReduceSum`, where `axes` is an input).
//!
//! Admitted subset (everything else is a typed [`Refusal`]):
//! - `X`: `int32`, rank 1. `T`: `int32`, a scalar (shape `[]` or `[1]`;
//!   multidirectional broadcasting of a scalar).
//! - `axes`: absent, or exactly `[0]` / `[-1]`. With absent axes and
//!   `noop_with_empty_axes = 0`, ONNX reduces every axis, which for rank 1 is
//!   the same reduction.
//! - `keepdims`: `0` (output shape `[]`) or `1` (output shape `[1]`).
//! - Output dtype is `int32`, as ONNX `ReduceSum` keeps the input type. The
//!   specification does not define int32 overflow; this wrapper refuses an
//!   exact sum outside `i32` instead of wrapping.

use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::{FoldDialect, FoldError};

use crate::native::{Pred, SumWhere};

/// Why the wrapper does not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// `X` is not rank 1.
    Rank(usize),
    /// `T` is not a scalar.
    ThresholdNotScalar(Vec<usize>),
    /// `axes` other than absent, `[0]` or `[-1]`.
    Axes(Vec<i64>),
    /// `keepdims` other than 0 or 1.
    KeepDims(i64),
    /// `noop_with_empty_axes = 1` with absent axes returns `X` unchanged, a
    /// tensor result outside this wrapper's scalar output.
    NoopWithEmptyAxes,
    /// The exact sum does not fit `int32`.
    OutOfDtypeRange(i64),
    /// The native polyfill refused.
    Native(FoldError),
}

/// The output, shaped as ONNX shapes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Output {
    /// The value.
    pub value: i32,
    /// `true`: shape `[1]` (`keepdims = 1`); `false`: shape `[]`.
    pub keepdims: bool,
}

/// `ReduceSum(Where(Greater(X, T), X, 0))` for the admitted subset.
pub fn reduce_sum_where_greater(
    x: &[i32],
    x_shape: &[usize],
    t: i32,
    t_shape: &[usize],
    axes: Option<&[i64]>,
    keepdims: i64,
    noop_with_empty_axes: i64,
) -> Result<Output, Refusal> {
    if x_shape.len() != 1 || x_shape[0] != x.len() {
        return Err(Refusal::Rank(x_shape.len()));
    }
    if !(t_shape.is_empty() || t_shape == [1]) {
        return Err(Refusal::ThresholdNotScalar(t_shape.to_vec()));
    }
    match axes {
        None if noop_with_empty_axes == 1 => return Err(Refusal::NoopWithEmptyAxes),
        None => {}
        Some(a) if a == [0] || a == [-1] => {}
        Some(a) => return Err(Refusal::Axes(a.to_vec())),
    }
    let keepdims = match keepdims {
        0 => false,
        1 => true,
        k => return Err(Refusal::KeepDims(k)),
    };
    let lanes = [LaneRef::I32(x)];
    let planes = Planes {
        n_rows: x.len(),
        masks: &[],
        lanes: &lanes,
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let op = SumWhere {
        lane: 0,
        present: None,
        pred: Pred::Gt(t),
    };
    let s = op.eval(&mut d, &mut Vec::new()).map_err(Refusal::Native)?;
    let value = i32::try_from(s).map_err(|_| Refusal::OutOfDtypeRange(s))?;
    Ok(Output { value, keepdims })
}
