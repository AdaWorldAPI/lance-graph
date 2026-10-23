//! `lance-graph-onnx` — ONNX mathematical compatibility over existing native
//! functions (D-ONNX-NATIVE-0, `.claude/plans/d-onnx-native-v1.md`).
//!
//! ONNX is a vocabulary over our existing mathematics, not another execution
//! system. The normal path is
//!
//! ```text
//! ONNX-compatible Rust API → existing native function
//! ```
//!
//! and nothing more. This crate has no graph, AST, interpreter, virtual
//! machine, runtime dispatcher, `LazyLock`/`OnceLock`, ClassId routing or
//! per-call backend choice. It depends on `ndarray` only, and no native crate
//! depends on it.
//!
//! Each operator is one of:
//!
//! - **ALIAS**: the native function has the ONNX contract. It is re-exported
//!   under the ONNX name; the symbol is the native one.
//! - **INLINE**: a small adapter does only the semantic work the native
//!   function does not (a shape check before broadcasting, an `int32` range
//!   check), then calls it.
//! - Anything else has no function here. The crosswalk in the plan lists each
//!   such operator as NATIVE (a native composition still to be built) or REFUSE.
//!
//! # Conventions
//!
//! - **Tensors** are `ndarray` arrays or slices. Rank-1 `int32` operators take
//!   `&[i32]`, as the native kernels do.
//! - **`tensor(bool)`** is a packed mask: bit `i % 64` of word `i / 64` is
//!   element `i`, and bits past the last element are zero. That is the
//!   representation of every `ndarray::simd` mask kernel. Use [`mask_words`]
//!   to size one.
//! - **Exact versus approximate**: every operator here is exact (IEEE-754 for
//!   floats, exact integers). Polynomial `Exp`/`Log`/`Tanh`/`Sigmoid` and
//!   palette operations are approximate and are not exported.
//!
//! # Example
//!
//! ```
//! use lance_graph_onnx as onnx;
//!
//! // Greater(X, 0) → tensor(bool), then ReduceSum(Where(·, X, 0)).
//! let x = [3, -1, 4, -1, 5];
//! let mut cond = vec![0u64; onnx::mask_words(x.len())];
//! onnx::greater_i32(&x, 0, &mut cond);
//! assert_eq!(cond[0], 0b10101);
//!
//! let rs = onnx::ReduceSum::new(None, 0, 0).unwrap();
//! let mut scratch = vec![0u64; onnx::mask_words(x.len())];
//! assert_eq!(rs.sum_where_greater_i32(&x, 0, &mut scratch), Ok(onnx::Reduced { value: 12, keepdims: false }));
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{Array, ArrayBase, ArrayRef, Data, DimMax, Dimension};

// ── ALIAS: identical contracts, re-exported under the ONNX name ────────────

/// `Abs` (`float`). Alias of `ndarray::hpc::vml::vsabs`.
pub use ndarray::hpc::vml::vsabs as abs_f32;
/// `Sqrt` (`float`). Alias of `ndarray::hpc::vml::vssqrt`.
pub use ndarray::hpc::vml::vssqrt as sqrt_f32;

/// `Equal(X, T)`, scalar `T`. Alias of `ndarray::simd::eq_i32_to_mask`.
pub use ndarray::simd::eq_i32_to_mask as equal_i32;
/// `GreaterOrEqual(X, T)`, scalar `T`. Alias of `ndarray::simd::ge_i32_to_mask`.
pub use ndarray::simd::ge_i32_to_mask as greater_or_equal_i32;
/// `Greater(X, T)`, `int32` `X`, scalar `T`. Alias of `ndarray::simd::gt_i32_to_mask`.
pub use ndarray::simd::gt_i32_to_mask as greater_i32;
/// `LessOrEqual(X, T)`, scalar `T`. Alias of `ndarray::simd::le_i32_to_mask`.
pub use ndarray::simd::le_i32_to_mask as less_or_equal_i32;
/// `Less(X, T)`, scalar `T`. Alias of `ndarray::simd::lt_i32_to_mask`.
pub use ndarray::simd::lt_i32_to_mask as less_i32;

/// `Where(C, X, Y)`, `int32`, equal shapes. Alias of `ndarray::simd::blend_i32`
/// (a set bit takes `X`).
pub use ndarray::simd::blend_i32 as where_i32;
/// `And` on `tensor(bool)`. Alias of `ndarray::simd::mask_and`.
pub use ndarray::simd::mask_and as and;
/// `Not` on `tensor(bool)` of `n` elements. Alias of `ndarray::simd::mask_not`.
pub use ndarray::simd::mask_not as not;
/// `Or` on `tensor(bool)`. Alias of `ndarray::simd::mask_or`.
pub use ndarray::simd::mask_or as or;
/// `Xor` on `tensor(bool)`. Alias of `ndarray::simd::mask_xor`.
pub use ndarray::simd::mask_xor as xor;

/// Words of a packed `tensor(bool)` of `n` elements.
pub const fn mask_words(n: usize) -> usize {
    n.div_ceil(64)
}

// ── refusals ───────────────────────────────────────────────────────────────

/// Why an operator does not answer. Unsupported dtypes are refused by type
/// (there is no function for them), so only runtime facts appear here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The shapes are not multidirectionally broadcastable.
    Broadcast { lhs: Vec<usize>, rhs: Vec<usize> },
    /// `axes` other than absent, `[0]` or `[-1]` for a rank-1 input.
    Axes(Vec<i64>),
    /// `keepdims` other than 0 or 1.
    KeepDims(i64),
    /// `noop_with_empty_axes = 1` with absent axes returns the input
    /// unchanged, a tensor result outside the scalar reductions here.
    NoopWithEmptyAxes,
    /// The exact result does not fit `int32`. ONNX does not define integer
    /// overflow; this crate refuses instead of wrapping.
    OutOfDtypeRange(i64),
}

// ── INLINE: element-wise float arithmetic with broadcasting ────────────────

/// The float types whose arithmetic is exact IEEE-754 in both ONNX and
/// `ndarray`. Integer arithmetic is not admitted: ONNX leaves overflow
/// undefined and `ndarray`'s `+` wraps in release and panics in debug.
pub trait Ieee:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + sealed::Sealed
{
}
impl Ieee for f32 {}
impl Ieee for f64 {}
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// ONNX multidirectional (numpy) broadcasting: shapes align from the right,
/// and each aligned pair is equal or contains a 1. `ndarray`'s arithmetic
/// broadcasts the same way but panics on a mismatch, so this check turns the
/// panic into a [`Refusal`]. It reads only the two shapes.
pub fn broadcastable(lhs: &[usize], rhs: &[usize]) -> Result<(), Refusal> {
    let ok = lhs
        .iter()
        .rev()
        .zip(rhs.iter().rev())
        .all(|(&a, &b)| a == b || a == 1 || b == 1);
    if ok {
        Ok(())
    } else {
        Err(Refusal::Broadcast {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        })
    }
}

macro_rules! binary {
    ($(#[$doc:meta])* $name:ident, $op:tt) => {
        $(#[$doc])*
        #[inline]
        pub fn $name<T, D, E>(
            a: &ArrayRef<T, D>,
            b: &ArrayRef<T, E>,
        ) -> Result<Array<T, <D as DimMax<E>>::Output>, Refusal>
        where
            T: Ieee,
            D: Dimension + DimMax<E>,
            E: Dimension,
        {
            broadcastable(a.shape(), b.shape())?;
            Ok(a $op b)
        }
    };
}

binary!(
    /// `Add`. INLINE: [`broadcastable`], then `ndarray`'s `&a + &b`.
    add, +
);
binary!(
    /// `Sub`. INLINE: [`broadcastable`], then `ndarray`'s `&a - &b`.
    sub, -
);
binary!(
    /// `Mul`. INLINE: [`broadcastable`], then `ndarray`'s `&a * &b`.
    mul, *
);
binary!(
    /// `Div` (float). INLINE: [`broadcastable`], then `ndarray`'s `&a / &b`.
    /// Division by zero follows IEEE-754 (`±inf`, `NaN`) in both.
    div, /
);

/// `Neg`. INLINE: `ndarray`'s element map with `-v`, the same body as its
/// `Neg for &ArrayRef` impl.
///
/// Not an alias of `ndarray::hpc::vml::vsneg`: its 16-lane body computes
/// `0.0 - v`, so `Neg(+0.0)` gives `+0.0` instead of `-0.0` whenever the input
/// has at least 16 elements (its scalar tail computes `-v` and is correct).
#[inline]
pub fn neg<T: Ieee, S: Data<Elem = T>, D: Dimension>(a: &ArrayBase<S, D>) -> Array<T, D> {
    a.map(|&v| -v)
}

// ── INLINE: ReduceSum, attributes validated once ───────────────────────────

/// `ReduceSum` over a rank-1 `int32` input, with its attributes validated
/// once. Every reduction axis of a rank-1 input is axis 0, so absent `axes`
/// (with `noop_with_empty_axes = 0`), `[0]` and `[-1]` are the same reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceSum {
    keepdims: bool,
}

/// A rank-1 reduction's result. `keepdims` decides the ONNX output shape:
/// `[1]` when true, `[]` when false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduced {
    pub value: i32,
    pub keepdims: bool,
}

impl ReduceSum {
    /// Validate `axes`, `keepdims` and `noop_with_empty_axes` (opset 18+,
    /// where `axes` is an input).
    pub fn new(
        axes: Option<&[i64]>,
        keepdims: i64,
        noop_with_empty_axes: i64,
    ) -> Result<Self, Refusal> {
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
        Ok(Self { keepdims })
    }

    fn fit(self, s: i64) -> Result<Reduced, Refusal> {
        let value = i32::try_from(s).map_err(|_| Refusal::OutOfDtypeRange(s))?;
        Ok(Reduced {
            value,
            keepdims: self.keepdims,
        })
    }

    /// `ReduceSum(X)`. INLINE: an exact `i64` sum and the `int32` range check.
    /// `ndarray` has no unmasked `i32 → i64` sum, so this is a plain iterator
    /// reduction, which LLVM vectorizes (plan §4).
    pub fn sum_i32(self, x: &[i32]) -> Result<Reduced, Refusal> {
        self.fit(x.iter().map(|&v| i64::from(v)).sum())
    }

    /// `ReduceSum(Where(Greater(X, T), X, 0))`, the composition #1752 probes.
    /// INLINE: `ndarray::simd::gt_i32_to_mask` into the caller's `scratch`
    /// (at least [`mask_words`]`(x.len())` words), `masked_sum_i32`, then the
    /// `int32` range check. No allocation.
    pub fn sum_where_greater_i32(
        self,
        x: &[i32],
        t: i32,
        scratch: &mut [u64],
    ) -> Result<Reduced, Refusal> {
        ndarray::simd::gt_i32_to_mask(x, t, scratch);
        self.fit(ndarray::simd::masked_sum_i32(x, scratch))
    }
}
