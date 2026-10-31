//! D-XLOG-DR64-0: lower one restricted, XLOG-compiled rule body onto the
//! existing DR64 Boolean relation register, and answer it by forward/backward
//! mask propagation over one aligned 64-slot domain.
//!
//! What is borrowed and from where:
//!
//! - **Frontend** (XLOG, CPU-only crates `xlog-core` / `xlog-ir` /
//!   `xlog-logic`): parsing, lowering to RIR, and the magic-set decision. The
//!   XLOG executor (`xlog-runtime`) is CUDA-coupled and is not used.
//! - **Register, guards, kernel** (`dr-math-binding-probe`): `bind_dr64`,
//!   `Chain` (column domain = next row domain, one current frame, one focus,
//!   one scope, two-valued Distinct only) and `relcomp::compose`.
//!
//! What is new here is only the admission of a RIR shape and three
//! bit-moves: read one row (forward), one `compose` against a one-column
//! selector and a pack of its column (backward), and one AND. No opcode, no
//! runtime, no transposed edge store, no per-row Java-or-Rust engine beyond
//! the 64-row register.
//!
//! The admitted shape is exactly
//!
//! ```text
//! Project([Column(1)],
//!   Join(Inner, left_keys [1], right_keys [0],
//!     Filter(Scan R, col0 == x)?,
//!     Filter(Scan S, col1 == z)?))
//! ```
//!
//! together with the same join with its sides swapped (XLOG's optimizer
//! orders joins from statistics) and the optimizer's `ChainJoin` promotion,
//! which is admitted only when it agrees with its own `fallback` plan.
//!
//! i.e. `q(Y) :- r(x?, Y), s(Y, z?)`: one-step, α-acyclic, set semantics.
//! For that body the intersection of the forward and backward candidate
//! masks IS the answer (it is the semijoin reduction of a two-atom chain).
//! Anything else is refused, never approximated.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

pub mod native;
pub mod recursive;

use dr_math_binding_probe::relcomp::{ComposeRefusal, Rows, Work, compose};
use dr_math_binding_probe::{
    Axis, BoundDr64, Focus, FrameStore, Guards, MethodSignature, Operand, Refusal, Scope, bind_dr64,
};
use xlog_core::RelId;
use xlog_ir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
use xlog_logic::ast::{BodyLiteral, Program, Term};

/// The slot count of one DR64 axis.
pub const SLOTS: usize = 64;

/// Why a program, rule or fact set does not lower onto the register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerRefusal {
    /// A node kind outside the admitted shape (aggregate, recursion,
    /// union, difference, multiway, tensor, …). Carries the node name.
    UnsupportedNode(&'static str),
    /// Only inner joins on `left.1 = right.0` are admitted. Semi, anti
    /// (negation) and outer joins are refused.
    UnsupportedJoin(JoinType),
    /// The join is not `left.col1 = right.col0`; answering it would need an
    /// admitted inverse of one operand, which this probe does not have.
    NeedsInverse {
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
    },
    /// A filter that is not a single `column == u32 constant` on the free
    /// end of its relation.
    UnsupportedFilter,
    /// The projection is not the shared middle variable alone.
    UnsupportedProjection,
    /// The program carries probabilistic facts or directives: a Boolean
    /// support mask cannot carry weights, so it is not lowered at all.
    Probabilistic,
    /// The program contains a negated, epistemic or arithmetic body literal.
    NonPositiveBody,
    /// A fact argument is not a non-negative integer that fits `u32`
    /// (a symbol is a different identity domain, never a slot).
    FactTerm { predicate: String },
    /// A scanned predicate is also the head of a rule, so its facts are not
    /// its whole relation. Reading only the facts would under-answer.
    DerivedRelation { predicate: String },
    /// A relation read by the rule is not binary.
    Arity { predicate: String, arity: usize },
    /// The relation id has no name in the compiler's table.
    UnknownRel(RelId),
    /// A `ChainJoin` whose own fields and whose `fallback` plan describe
    /// different queries.
    ChainJoinDisagrees,
    /// A `GroupBy` other than one `Count` keyed by the answer column.
    UnsupportedAggregate,
}

/// The admitted rule body, in RIR terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainQuery {
    /// Left relation: `r(X, Y)`.
    pub left: RelId,
    /// `X == x` when bound.
    pub left_bound: Option<u32>,
    /// Right relation: `s(Y, Z)`.
    pub right: RelId,
    /// `Z == z` when bound.
    pub right_bound: Option<u32>,
}

/// Admit one XLOG-compiled rule body, or say why not.
pub fn admit(body: &RirNode) -> Result<ChainQuery, LowerRefusal> {
    // XLOG's optimizer promotes a two-atom chain to `ChainJoin` and keeps the
    // plain plan as `fallback`. Admit it only when its own fields describe
    // the same query as its fallback; a disagreement is refused, never
    // resolved by picking one.
    if let RirNode::ChainJoin {
        left,
        right,
        left_key,
        right_key,
        output_columns,
        fallback,
    } = body
    {
        if (*left_key, *right_key) != (1, 0) {
            return Err(LowerRefusal::NeedsInverse {
                left_keys: vec![*left_key],
                right_keys: vec![*right_key],
            });
        }
        if !matches!(
            output_columns.as_slice(),
            [ProjectExpr::Column(1)] | [ProjectExpr::Column(2)]
        ) {
            return Err(LowerRefusal::UnsupportedProjection);
        }
        let (l, lb) = scan_with_bound(left, 0)?;
        let (r, rb) = scan_with_bound(right, 1)?;
        let own = ChainQuery {
            left: l,
            left_bound: lb,
            right: r,
            right_bound: rb,
        };
        return match admit(fallback) {
            Ok(fb) if fb == own => Ok(own),
            _ => Err(LowerRefusal::ChainJoinDisagrees),
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
    // Join output columns are left ++ right. XLOG's cost-based optimizer
    // picks the join order from statistics, so both orders of the same chain
    // occur for the same rule:
    //   r(X,Y) ⋈ s(Y,Z): keys [1]/[0], columns [X,Y,Y,Z], Y at 1 or 2;
    //   s(Y,Z) ⋈ r(X,Y): keys [0]/[1], columns [Y,Z,X,Y], Y at 0 or 3.
    let column = match columns.as_slice() {
        [ProjectExpr::Column(c)] => *c,
        _ => return Err(LowerRefusal::UnsupportedProjection),
    };
    let (r_side, s_side, y_cols) = match (left_keys.as_slice(), right_keys.as_slice()) {
        ([1], [0]) => (left, right, [1, 2]),
        ([0], [1]) => (right, left, [0, 3]),
        _ => {
            return Err(LowerRefusal::NeedsInverse {
                left_keys: left_keys.clone(),
                right_keys: right_keys.clone(),
            });
        }
    };
    if !y_cols.contains(&column) {
        return Err(LowerRefusal::UnsupportedProjection);
    }
    let (left, left_bound) = scan_with_bound(r_side, 0)?;
    let (right, right_bound) = scan_with_bound(s_side, 1)?;
    Ok(ChainQuery {
        left,
        left_bound,
        right,
        right_bound,
    })
}

/// `Scan`, or `Filter(Scan, Column(free) == U32)` where `free` is the end of
/// the relation that does not take part in the join.
fn scan_with_bound(node: &RirNode, free: usize) -> Result<(RelId, Option<u32>), LowerRefusal> {
    match node {
        RirNode::Scan { rel } => Ok((*rel, None)),
        RirNode::Filter { input, predicate } => {
            let RirNode::Scan { rel } = input.as_ref() else {
                return Err(LowerRefusal::UnsupportedNode(node_name(input)));
            };
            match predicate {
                Expr::Compare {
                    left,
                    op: CompareOp::Eq,
                    right,
                } => match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(c), Expr::Const(ConstValue::U32(v))) if *c == free => {
                        Ok((*rel, Some(*v)))
                    }
                    _ => Err(LowerRefusal::UnsupportedFilter),
                },
                _ => Err(LowerRefusal::UnsupportedFilter),
            }
        }
        other => Err(LowerRefusal::UnsupportedNode(node_name(other))),
    }
}

fn node_name(n: &RirNode) -> &'static str {
    match n {
        RirNode::Unit => "Unit",
        RirNode::Scan { .. } => "Scan",
        RirNode::Filter { .. } => "Filter",
        RirNode::Project { .. } => "Project",
        RirNode::Join { .. } => "Join",
        RirNode::ChainJoin { .. } => "ChainJoin",
        RirNode::GroupBy { .. } => "GroupBy",
        RirNode::Union { .. } => "Union",
        RirNode::Distinct { .. } => "Distinct",
        RirNode::Diff { .. } => "Diff",
        RirNode::Fixpoint { .. } => "Fixpoint",
        RirNode::MultiWayJoin { .. } => "MultiWayJoin",
        _ => "other",
    }
}

/// Program-level checks the RIR alone cannot show (a probabilistic fact
/// lowers to the same deterministic RIR shape).
pub fn admit_program(program: &Program) -> Result<(), LowerRefusal> {
    if program.is_probabilistic_profile() {
        return Err(LowerRefusal::Probabilistic);
    }
    for rule in &program.rules {
        if rule
            .body
            .iter()
            .any(|l| !matches!(l, BodyLiteral::Positive(_)))
        {
            return Err(LowerRefusal::NonPositiveBody);
        }
    }
    Ok(())
}

/// The ground binary `u32` tuples of one predicate, from the program's facts.
/// Refused when a rule also defines the predicate (see
/// [`LowerRefusal::DerivedRelation`]).
pub fn facts(program: &Program, predicate: &str) -> Result<Vec<(u32, u32)>, LowerRefusal> {
    // A rule with this head derives tuples no fact lists. The register would
    // hold only part of the relation, so refuse instead of under-answering.
    // (Conservative: a same-name rule of another arity is refused too.)
    if program
        .rules
        .iter()
        .any(|r| !r.is_fact() && r.head.predicate == predicate)
    {
        return Err(LowerRefusal::DerivedRelation {
            predicate: predicate.to_string(),
        });
    }
    let mut out = Vec::new();
    for rule in program.rules.iter().filter(|r| r.is_fact()) {
        if rule.head.predicate != predicate {
            continue;
        }
        if rule.head.terms.len() != 2 {
            return Err(LowerRefusal::Arity {
                predicate: predicate.to_string(),
                arity: rule.head.terms.len(),
            });
        }
        let id = |t: &Term| match t {
            Term::Integer(i) => u32::try_from(*i).ok(),
            _ => None,
        };
        match (id(&rule.head.terms[0]), id(&rule.head.terms[1])) {
            // A base relation is a set: XLOG's fact loader skips a row it
            // has already seen, so a repeated fact is not a second witness.
            (Some(a), Some(b)) => {
                if !out.contains(&(a, b)) {
                    out.push((a, b));
                }
            }
            _ => {
                return Err(LowerRefusal::FactTerm {
                    predicate: predicate.to_string(),
                });
            }
        }
    }
    Ok(out)
}

/// Why source ids do not fit one axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainRefusal {
    /// More distinct ids than slots.
    Overflow { needed: usize },
    /// An id that this axis never assigned a slot.
    Unmapped(u32),
}

/// Source-domain ordinal map of ONE axis: a slot is meaningful only under
/// the axis it was assigned in. The same numeric id on two axes is two
/// unrelated slots, by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddressMap {
    pub axis: Axis,
    slot_of: BTreeMap<u32, u8>,
    id_of: Vec<u32>,
}

impl AddressMap {
    /// Assign slots to `ids` in ascending id order.
    pub fn new(axis: Axis, ids: impl IntoIterator<Item = u32>) -> Result<Self, DomainRefusal> {
        let mut sorted: Vec<u32> = ids.into_iter().collect();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() > SLOTS {
            return Err(DomainRefusal::Overflow {
                needed: sorted.len(),
            });
        }
        let slot_of = sorted
            .iter()
            .enumerate()
            .map(|(s, &id)| (id, s as u8))
            .collect();
        Ok(Self {
            axis,
            slot_of,
            id_of: sorted,
        })
    }

    pub fn slot(&self, id: u32) -> Result<usize, DomainRefusal> {
        self.slot_of
            .get(&id)
            .map(|&s| s as usize)
            .ok_or(DomainRefusal::Unmapped(id))
    }

    pub fn len(&self) -> usize {
        self.id_of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_of.is_empty()
    }

    /// The bits of the assigned slots; every bit at or past `len()` is zero.
    pub fn valid(&self) -> u64 {
        match self.len() {
            SLOTS => u64::MAX,
            n => (1u64 << n) - 1,
        }
    }

    /// Source ids of the set bits of `mask`, ascending.
    pub fn ids(&self, mask: u64) -> Vec<u32> {
        (0..self.len())
            .filter(|&s| mask >> s & 1 == 1)
            .map(|s| self.id_of[s])
            .collect()
    }
}

/// Fill one 64x64 register from `(row id, column id)` tuples.
pub fn register(
    tuples: &[(u32, u32)],
    rows: &AddressMap,
    cols: &AddressMap,
) -> Result<Rows, DomainRefusal> {
    let mut r = [0u64; SLOTS];
    for &(a, b) in tuples {
        r[rows.slot(a)?] |= 1u64 << cols.slot(b)?;
    }
    Ok(r)
}

/// Why a bound chain query was not answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalRefusal {
    Bind(Refusal),
    Chain(ComposeRefusal),
    Domain(DomainRefusal),
    /// The left operand's column axis is not the right operand's row axis.
    AxisMismatch,
}

/// Masks over the middle axis, and the work the kernel did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Propagation {
    /// Forward from the bound left end (`u64::MAX`-free: masked to the axis).
    pub forward: u64,
    /// Backward from the bound right end.
    pub backward: u64,
    /// `forward & backward`.
    pub answer: u64,
    /// Row ORs done by `compose` (backward only; forward is one row read).
    pub work: Work,
    /// 64-bit words read from the two registers.
    pub words_read: u64,
}

/// Bind both registers under one frame, focus and scope, type the chain,
/// then propagate. `mid` is the axis both registers share.
#[allow(clippy::too_many_arguments)]
pub fn propagate(
    store: &FrameStore,
    guards: &Guards,
    now: (Focus, Scope),
    (left_sig, left_rows): (MethodSignature, &Rows),
    (right_sig, right_rows): (MethodSignature, &Rows),
    left_slot: Option<usize>,
    right_slot: Option<usize>,
    mid: &AddressMap,
) -> Result<Propagation, EvalRefusal> {
    let l = bind(store, guards, now, left_sig, left_rows)?;
    let r = bind(store, guards, now, right_sig, right_rows)?;
    // `mid` must be the axis the answer is read in. Whether the right
    // operand's rows are that same axis is the existing chain typing's job.
    if left_sig.column != mid.axis {
        return Err(EvalRefusal::AxisMismatch);
    }
    // The existing typing: column domain of l == row domain of r, one
    // current frame, one focus, one scope, Boolean Distinct.
    dr_math_binding_probe::relcomp::Chain::new(store, vec![l.clone(), r.clone()])
        .map_err(EvalRefusal::Chain)?;

    let valid = mid.valid();
    let mut words_read = 0u64;
    let forward = match left_slot {
        Some(x) => {
            words_read += 1;
            l.rows[x] & valid
        }
        // Unbound left end: every Y that has some left partner.
        None => {
            words_read += SLOTS as u64;
            l.rows.iter().fold(0, |a, w| a | w) & valid
        }
    };
    let mut work = Work::default();
    let backward = match right_slot {
        Some(z) => {
            // Preimage of {z} under the right relation, with the existing
            // kernel and no transposed copy: compose with the selector
            // C[z] = 1 (a 64 x 1 relation into one output slot), then read
            // column 0 of the result.
            let mut sel = [0u64; SLOTS];
            sel[z] = 1;
            let col = compose(r.rows, &sel, &mut work);
            words_read += SLOTS as u64;
            pack_column0(&col) & valid
        }
        None => {
            words_read += SLOTS as u64;
            r.rows
                .iter()
                .enumerate()
                .fold(0, |m, (i, w)| m | (u64::from(*w != 0) << i))
                & valid
        }
    };
    Ok(Propagation {
        forward,
        backward,
        answer: forward & backward,
        work,
        words_read,
    })
}

fn bind<'a>(
    store: &FrameStore,
    guards: &Guards,
    now: (Focus, Scope),
    sig: MethodSignature,
    rows: &'a Rows,
) -> Result<BoundDr64<'a>, EvalRefusal> {
    bind_dr64(
        store,
        sig,
        sig,
        guards,
        now,
        Operand {
            rows,
            validity: None,
        },
    )
    .map_err(EvalRefusal::Bind)
}

/// Bit `i` of the result is bit 0 of row `i`.
fn pack_column0(rows: &Rows) -> u64 {
    rows.iter()
        .enumerate()
        .fold(0, |m, (i, w)| m | ((w & 1) << i))
}
