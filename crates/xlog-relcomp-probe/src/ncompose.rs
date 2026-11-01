//! ℕ-compose: `p(X, Z) :- r(X, Y), s(Y, Z)` with both ends kept, as an
//! integer matrix product over DR tiles (D-DYNBAG-1, plan
//! `d-dynbag-holographic-bags-v1.md` §7).
//!
//! R and S are sets of tuples (§9.2). On local tiles they are 0/1 matrices
//! A (|X| × |Y|) and B (|Y| × |Z|). Then `(A · B)[x][z]` is the number of `y`
//! with `r(x, y) ∧ s(y, z)`: the ℕ answer, one per witness. Its projection
//! `> 0` is the Boolean composition (`relcomp::compose`).
//!
//! The product is `ndarray::simd::matmul_i8_to_i32` (AMX `TDPBUSD`, AVX-512
//! VNNI, or the scalar reference, bit-identical). One hop over 0/1 inputs is
//! exact: a cell is at most the Y tile width. A chain of hops (counts feeding
//! the next product) is not admitted here (§7.3).
//!
//! Lab only. The axes are DR's `LocalMap`s, one per X, Y, Z, chosen by the
//! caller; an id a map does not hold is refused. Bounds on X or Z are refused
//! by [`admit_pair`]; they are a later step.

use lance_graph_dt64::dr::{LocalMap, MapRefusal};
use ndarray::Array2;
use ndarray::simd::matmul_i8_to_i32;
use xlog_core::RelId;
use xlog_ir::{JoinType, ProjectExpr, RirNode};

use crate::{LowerRefusal, node_name, scan_with_bound};

/// `p(X, Z) :- r(X, Y), s(Y, Z)`, admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairQuery {
    /// `r(X, Y)`.
    pub left: RelId,
    /// `s(Y, Z)`.
    pub right: RelId,
}

/// Why the product did not answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComposeRefusal {
    /// An id outside the axis's DR map.
    Domain(MapRefusal),
    /// The product itself refused (shape).
    Matmul(String),
}

/// Admit XLOG's compiled body for `p(X, Z) :- r(X, Y), s(Y, Z)`: an inner
/// join on Y keeping X and Z, in either join order. Bounds, other
/// projections and other joins are refused.
pub fn admit_pair(body: &RirNode) -> Result<PairQuery, LowerRefusal> {
    // XLOG's optimizer promotes the chain to `ChainJoin` (in either join
    // order, chosen from statistics) and keeps the plain plan as `fallback`.
    // Admit it only when its own relations name the same query as its
    // fallback; a disagreement is refused, never resolved by picking one.
    if let RirNode::ChainJoin {
        left,
        right,
        left_key,
        right_key,
        fallback,
        ..
    } = body
    {
        let (r_side, s_side) = match (*left_key, *right_key) {
            (1, 0) => (left, right),
            (0, 1) => (right, left),
            _ => {
                return Err(LowerRefusal::NeedsInverse {
                    left_keys: vec![*left_key],
                    right_keys: vec![*right_key],
                });
            }
        };
        let own = PairQuery {
            left: scan_with_bound(r_side, 0)?.0,
            right: scan_with_bound(s_side, 1)?.0,
        };
        return match admit_pair(fallback) {
            Ok(fb) if fb == own => Ok(own),
            Ok(_) => Err(LowerRefusal::ChainJoinDisagrees),
            Err(e) => Err(e),
        };
    }
    let RirNode::Project { input, columns } = body else {
        return Err(LowerRefusal::UnsupportedNode(node_name(body)));
    };
    let RirNode::Join {
        left,
        right,
        left_keys,
        right_keys,
        join_type,
    } = input.as_ref()
    else {
        return Err(LowerRefusal::UnsupportedNode(node_name(input)));
    };
    if *join_type != JoinType::Inner {
        return Err(LowerRefusal::UnsupportedJoin(*join_type));
    }
    // Join output columns are left ++ right:
    //   r(X,Y) ⋈ s(Y,Z): keys [1]/[0], columns [X,Y,Y,Z]: X at 0, Z at 3;
    //   s(Y,Z) ⋈ r(X,Y): keys [0]/[1], columns [Y,Z,X,Y]: X at 2, Z at 1.
    let (r_side, s_side, xz) = match (left_keys.as_slice(), right_keys.as_slice()) {
        ([1], [0]) => (left, right, (0, 3)),
        ([0], [1]) => (right, left, (2, 1)),
        _ => {
            return Err(LowerRefusal::NeedsInverse {
                left_keys: left_keys.clone(),
                right_keys: right_keys.clone(),
            });
        }
    };
    match columns.as_slice() {
        [ProjectExpr::Column(x), ProjectExpr::Column(z)] if (*x, *z) == xz => {}
        _ => return Err(LowerRefusal::UnsupportedProjection),
    }
    let (l, lb) = scan_with_bound(r_side, 0)?;
    let (r, rb) = scan_with_bound(s_side, 1)?;
    if lb.is_some() || rb.is_some() {
        return Err(LowerRefusal::UnsupportedFilter);
    }
    Ok(PairQuery { left: l, right: r })
}

/// A relation's 0/1 matrix over two DR axes. A repeated tuple sets the same
/// cell: base relations are sets.
fn matrix(
    tuples: &[(u32, u32)],
    rows: &LocalMap,
    cols: &LocalMap,
) -> Result<Array2<i8>, ComposeRefusal> {
    let mut m = Array2::<i8>::zeros((rows.width().positions(), cols.width().positions()));
    for &(a, b) in tuples {
        let i = rows.local(u64::from(a)).map_err(ComposeRefusal::Domain)?;
        let j = cols.local(u64::from(b)).map_err(ComposeRefusal::Domain)?;
        m[(usize::from(i), usize::from(j))] = 1;
    }
    Ok(m)
}

/// Witnesses per (x, z), indexed by local position on the X and Z tiles.
pub fn pair_counts(
    _q: &PairQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    xs: &LocalMap,
    ys: &LocalMap,
    zs: &LocalMap,
) -> Result<Array2<i32>, ComposeRefusal> {
    let a = matrix(left, xs, ys)?;
    let b = matrix(right, ys, zs)?;
    let mut out = Array2::<i32>::zeros((a.nrows(), b.ncols()));
    matmul_i8_to_i32(a.view(), b.view(), out.view_mut())
        .map_err(|e| ComposeRefusal::Matmul(format!("{e:?}")))?;
    Ok(out)
}
