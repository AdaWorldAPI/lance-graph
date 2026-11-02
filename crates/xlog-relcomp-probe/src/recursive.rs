//! Recursive reachability: XLOG's compiled recursive SCC, evaluated on the
//! DR64 register with the existing compose kernel.
//!
//! XLOG compiles
//!
//! ```text
//! path(X, Y) :- edge(X, Y).
//! path(X, Y) :- path(X, Z), edge(Z, Y).
//! ```
//!
//! into one recursive SCC with two rule bodies: `Scan edge`, and a
//! `ChainJoin` of `Scan path` with `Scan edge` on `path.1 = edge.0`,
//! projecting columns 0 and 3, whose `fallback` is the same join written
//! out. [`admit_closure`] accepts exactly that shape and refuses everything
//! else. [`closure`] evaluates it semi-naively (each round composes only the
//! newly derived pairs with `edge`), which is the evaluation order XLOG's own
//! runtime uses for a recursive SCC.
//!
//! Derivations are never stored. [`support`] and [`affected`] read them back
//! from the closure as masks when asked: the base facts an answer depends on,
//! and the answers a base fact feeds.

use std::collections::HashMap;

use dr_math_binding_probe::relcomp::{Chain, Rows, Work, compose};
use dr_math_binding_probe::{FrameStore, Guards, MethodSignature};
use xlog_core::RelId;
use xlog_ir::{ExecutionPlan, JoinType, ProjectExpr, RirNode};

use crate::{EvalRefusal, LowerRefusal, SLOTS, bind};

/// The admitted recursive SCC: `head` is the closure of `edge`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureQuery {
    pub head: String,
    pub edge: RelId,
}

/// Admit XLOG's compiled plan when it is exactly one linear transitive
/// closure over one base relation, or say why not.
pub fn admit_closure(
    plan: &ExecutionPlan,
    rel_ids: &HashMap<String, RelId>,
) -> Result<ClosureQuery, LowerRefusal> {
    let recursive: Vec<_> = plan.sccs.iter().filter(|s| s.is_recursive).collect();
    let [scc] = recursive.as_slice() else {
        return Err(LowerRefusal::UnsupportedNode("recursive SCC count"));
    };
    let [head] = scc.predicates.as_slice() else {
        return Err(LowerRefusal::UnsupportedNode("mutual recursion"));
    };
    let head_id = *rel_ids
        .get(head)
        .ok_or(LowerRefusal::UnsupportedNode("head id"))?;
    let rules: Vec<&RirNode> = plan
        .rules_by_scc
        .iter()
        .flatten()
        .filter(|r| &r.head == head)
        .map(|r| &r.body)
        .collect();
    let mut base = None;
    let mut step = None;
    for body in rules {
        match body {
            RirNode::Scan { rel } if *rel != head_id && base.is_none() => base = Some(*rel),
            _ if step.is_none() => step = Some(step_edge(body, head_id)?),
            _ => return Err(LowerRefusal::UnsupportedNode("extra rule in the SCC")),
        }
    }
    match (base, step) {
        (Some(b), Some(s)) if b == s => Ok(ClosureQuery {
            head: head.clone(),
            edge: b,
        }),
        (Some(_), Some(_)) => Err(LowerRefusal::UnsupportedNode("step over another relation")),
        _ => Err(LowerRefusal::UnsupportedNode("missing base or step rule")),
    }
}

/// The relation a step `head(X, Y) :- head(X, Z), edge(Z, Y)` extends by.
fn step_edge(body: &RirNode, head: RelId) -> Result<RelId, LowerRefusal> {
    let joined =
        |left: &RirNode, right: &RirNode, lk: usize, rk: usize, cols: &[ProjectExpr]| match (
            left, right,
        ) {
            (RirNode::Scan { rel: l }, RirNode::Scan { rel: r }) if *l == head && *r != head => {
                if (lk, rk) != (1, 0) {
                    return Err(LowerRefusal::NeedsInverse {
                        left_keys: vec![lk],
                        right_keys: vec![rk],
                    });
                }
                if !matches!(cols, [ProjectExpr::Column(0), ProjectExpr::Column(3)]) {
                    return Err(LowerRefusal::UnsupportedProjection);
                }
                Ok(*r)
            }
            _ => Err(LowerRefusal::UnsupportedNode(
                "non-linear or reordered step",
            )),
        };
    let plain = |node: &RirNode| match node {
        RirNode::Project { input, columns } => match input.as_ref() {
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                if *join_type != JoinType::Inner {
                    return Err(LowerRefusal::UnsupportedJoin(*join_type));
                }
                let ([lk], [rk]) = (left_keys.as_slice(), right_keys.as_slice()) else {
                    return Err(LowerRefusal::NeedsInverse {
                        left_keys: left_keys.clone(),
                        right_keys: right_keys.clone(),
                    });
                };
                joined(left, right, *lk, *rk, columns)
            }
            _ => Err(LowerRefusal::UnsupportedNode("projected non-join")),
        },
        _ => Err(LowerRefusal::UnsupportedNode("step shape")),
    };
    match body {
        RirNode::ChainJoin {
            left,
            right,
            left_key,
            right_key,
            output_columns,
            fallback,
        } => {
            let fast = joined(left, right, *left_key, *right_key, output_columns)?;
            if plain(fallback)? != fast {
                return Err(LowerRefusal::ChainJoinDisagrees);
            }
            Ok(fast)
        }
        other => plain(other),
    }
}

/// The closure and the work done to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Closure {
    pub rows: Rows,
    /// Rounds until no new pair appeared (the last round derives nothing).
    pub rounds: usize,
    /// Row ORs done by `compose`.
    pub work: Work,
}

/// Bind `edge` under the sealed frame, type `edge ∘ edge`, then evaluate the
/// closure semi-naively. `valid` masks the assigned slots of the one axis.
pub fn closure(
    store: &FrameStore,
    guards: &Guards,
    now: (dr_math_binding_probe::Focus, dr_math_binding_probe::Scope),
    sig: MethodSignature,
    edge: &Rows,
    valid: u64,
) -> Result<Closure, EvalRefusal> {
    let e = bind(store, guards, now, sig, edge)?;
    // The existing typing: the row axis of `edge` is its column axis.
    Chain::new(store, vec![e.clone(), e.clone()]).map_err(EvalRefusal::Chain)?;
    let mut path = [0u64; SLOTS];
    for (p, w) in path.iter_mut().zip(e.rows) {
        *p = w & valid;
    }
    let mut delta = path;
    let mut work = Work::default();
    let mut rounds = 0;
    loop {
        rounds += 1;
        let next = compose(&delta, e.rows, &mut work);
        let mut any = false;
        for i in 0..SLOTS {
            delta[i] = next[i] & valid & !path[i];
            path[i] |= delta[i];
            any |= delta[i] != 0;
        }
        if !any {
            return Ok(Closure {
                rows: path,
                rounds,
                work,
            });
        }
    }
}

/// `R* = I ∪ R` over the assigned slots.
fn reflexive(path: &Rows, valid: u64) -> Rows {
    let mut out = *path;
    for (i, w) in out.iter_mut().enumerate() {
        if valid >> i & 1 == 1 {
            *w |= 1 << i;
        }
    }
    out
}

/// The `edge` facts a derivation of `path(x, y)` can use: `(u, v)` with
/// `u` reachable from `x` (or `u = x`) and `y` reachable from `v` (or
/// `v = y`). Empty when `path(x, y)` does not hold. Computed from the closure
/// on demand; no derivation is stored.
pub fn support(edge: &Rows, path: &Rows, valid: u64, x: usize, y: usize) -> Rows {
    let mut out = [0u64; SLOTS];
    if path[x] >> y & 1 == 0 {
        return out;
    }
    let star = reflexive(path, valid);
    // Columns v whose closure reaches y: column y of R*, read row by row.
    let to_y: u64 = (0..SLOTS)
        .filter(|&v| star[v] >> y & 1 == 1)
        .fold(0, |m, v| m | 1 << v);
    for u in 0..SLOTS {
        if star[x] >> u & 1 == 1 {
            out[u] = edge[u] & to_y;
        }
    }
    out
}

/// The `path` answers a base fact `edge(u, v)` feeds: `(a, b)` with `u`
/// reachable from `a` (or `a = u`) and `b` reachable from `v` (or `b = v`).
pub fn affected(path: &Rows, valid: u64, u: usize, v: usize) -> Rows {
    let star = reflexive(path, valid);
    let mut out = [0u64; SLOTS];
    for a in 0..SLOTS {
        if star[a] >> u & 1 == 1 {
            out[a] = star[v];
        }
    }
    out
}

/// The derivation semi-naive evaluation performed for `path(x, y)`, as the
/// `edge` facts `[(x, z1), (z1, z2), ..., (zk, y)]` in slot coordinates.
///
/// Replays the rounds [`closure`] runs (round 1 is `edge & valid`; round
/// r+1 is `compose(delta, edge) & valid & !path`) and keeps each round's
/// delta. `path(x, y)` enters the delta in exactly one round r; for r = 1 the
/// derivation is the base fact, otherwise it extends a pair `(x, z)` from
/// round r-1 by `edge(z, y)`, taking the smallest such `z`. The chain length
/// is therefore the shortest-path length. `None` when `path(x, y)` does not
/// hold. Allocates; it is a cold replay and does not touch [`closure`].
pub fn trace(edge: &Rows, valid: u64, x: usize, y: usize) -> Option<Vec<(usize, usize)>> {
    if x >= SLOTS || y >= SLOTS {
        return None;
    }
    let mut path = [0u64; SLOTS];
    for (p, w) in path.iter_mut().zip(edge) {
        *p = w & valid;
    }
    let mut deltas: Vec<Rows> = vec![path];
    let mut work = Work::default();
    loop {
        let last = deltas[deltas.len() - 1];
        let next = compose(&last, edge, &mut work);
        let mut delta = [0u64; SLOTS];
        let mut any = false;
        for i in 0..SLOTS {
            delta[i] = next[i] & valid & !path[i];
            path[i] |= delta[i];
            any |= delta[i] != 0;
        }
        if !any {
            break;
        }
        deltas.push(delta);
    }
    let mut round = deltas.iter().position(|d| d[x] >> y & 1 == 1)?;
    let mut chain = Vec::with_capacity(round + 1);
    let mut end = y;
    while round > 0 {
        let prev = &deltas[round - 1];
        let z = (0..SLOTS).find(|&z| prev[x] >> z & 1 == 1 && edge[z] >> end & 1 == 1)?;
        chain.push((z, end));
        end = z;
        round -= 1;
    }
    chain.push((x, end));
    chain.reverse();
    Some(chain)
}
