//! D-DRMC-5: associativity of Boolean relation composition, the first law the
//! certified-compiler boundary can certify (charter, "Experimental success
//! criterion").
//!
//! `(R ∘ S) ∘ T = R ∘ (S ∘ T)` over OR/AND is established algebra. What this
//! probe tests is whether the pieces around it preserve it:
//!
//! 1. **Typing**: a [`Chain`] of bound DR64 operands composes only where each
//!    operand's column domain is the next one's row domain, under one frame
//!    issue, one focus and one scope, and only for two-valued Distinct
//!    relations (the Boolean semiring).
//! 2. **Law admission**: [`admit_composition_assoc`] admits associativity only
//!    for the `BooleanCompose` shape over two-valued Distinct relations. An
//!    [`AdmittedLaw`] cannot be built any other way.
//! 3. **Proof**: egglog, given only the admitted rule, proves that two
//!    association orders of the same chain are equal, and checks its own
//!    proof. An equation the admitted rule does not imply (commutation) is
//!    refused.
//! 4. **Lowering**: an order lowers to a list of compose steps;
//!    [`lower`]/[`raise`] round-trip, and the steps keep the leaves in chain
//!    order. This is a structural check of the lowering, not a proof of the
//!    compose kernel; the kernel is tested against an independent pair
//!    enumeration in `tests/relcomp.rs`.
//! 5. **Certificate**: [`certify_reassociation`] issues a
//!    [`RecipeCertificate`] only with all of the above.
//!
//! Cost: composing `A` with `B` ORs one row of `B` per set bit of `A`, so the
//! [`Work`] of an order is the popcount of every left operand it composes.
//! Orders of one chain compute the same relation at different cost; picking
//! one is the specialization-selection question, still open (see README).
//!
//! The compose kernel is plain `u64` word arithmetic, no intrinsics.
use egglog::{CommandOutput, EGraph};

use crate::recipe::{CompiledReference, LawReference, LawShape, Primitive, RecipeCertificate};
use crate::{BoundDr64, FrameStore, Population, Truth};

/// One 64 × 64 Boolean relation register (512 B).
pub type Rows = [u64; 64];

/// An association order over chain positions `0..n`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelExpr {
    Leaf(usize),
    Comp(Box<RelExpr>, Box<RelExpr>),
}

impl RelExpr {
    pub fn comp(a: RelExpr, b: RelExpr) -> Self {
        RelExpr::Comp(Box::new(a), Box::new(b))
    }

    /// Leaves left to right.
    pub fn leaves(&self) -> Vec<usize> {
        match self {
            RelExpr::Leaf(i) => vec![*i],
            RelExpr::Comp(a, b) => {
                let mut v = a.leaves();
                v.extend(b.leaves());
                v
            }
        }
    }

    /// The `Rel` term egglog sees: leaf `i` is `(Base \"r{i}\")`.
    pub fn egglog_term(&self) -> String {
        match self {
            RelExpr::Leaf(i) => format!("(Base \"r{i}\")"),
            RelExpr::Comp(a, b) => format!("(Comp {} {})", a.egglog_term(), b.egglog_term()),
        }
    }
}

/// Every association order of a chain of `n` operands (Catalan(n-1) of them).
pub fn all_orders(n: usize) -> Vec<RelExpr> {
    fn range(lo: usize, hi: usize) -> Vec<RelExpr> {
        if hi - lo == 1 {
            return vec![RelExpr::Leaf(lo)];
        }
        let mut out = Vec::new();
        for split in lo + 1..hi {
            for a in range(lo, split) {
                for b in range(split, hi) {
                    out.push(RelExpr::comp(a.clone(), b));
                }
            }
        }
        out
    }
    if n == 0 { Vec::new() } else { range(0, n) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeRefusal {
    EmptyChain,
    /// Only two-valued Distinct relations compose over OR/AND here.
    NotBooleanDistinct {
        at: usize,
    },
    /// Operand `at`'s column domain is not operand `at + 1`'s row domain.
    DomainMismatch {
        at: usize,
    },
    DifferentFrame {
        at: usize,
    },
    StaleFrame,
    DifferentFocus {
        at: usize,
    },
    DifferentScope {
        at: usize,
    },
    /// The order does not cover this chain's positions `0..n` in order.
    NotThisChain,
}

/// Operands that compose in sequence: domains line up, one frame issue that
/// is still current, one focus, one scope.
#[derive(Debug)]
pub struct Chain<'a> {
    operands: Vec<BoundDr64<'a>>,
}

impl<'a> Chain<'a> {
    pub fn new(store: &FrameStore, operands: Vec<BoundDr64<'a>>) -> Result<Self, ComposeRefusal> {
        let first = operands.first().ok_or(ComposeRefusal::EmptyChain)?;
        for (at, op) in operands.iter().enumerate() {
            if op.method.truth != Truth::TwoValued || op.method.population != Population::Distinct {
                return Err(ComposeRefusal::NotBooleanDistinct { at });
            }
            if !op.frame.same_issue(&first.frame) {
                return Err(ComposeRefusal::DifferentFrame { at });
            }
            if op.focus != first.focus {
                return Err(ComposeRefusal::DifferentFocus { at });
            }
            if op.scope != first.scope {
                return Err(ComposeRefusal::DifferentScope { at });
            }
        }
        for (at, pair) in operands.windows(2).enumerate() {
            if pair[0].method.column != pair[1].method.row {
                return Err(ComposeRefusal::DomainMismatch { at });
            }
        }
        store
            .check(&first.frame)
            .map_err(|_| ComposeRefusal::StaleFrame)?;
        Ok(Self { operands })
    }

    pub fn len(&self) -> usize {
        self.operands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    /// Evaluate one association order with the compose kernel.
    pub fn eval(&self, order: &RelExpr) -> Result<(Rows, Work), ComposeRefusal> {
        if order.leaves() != (0..self.len()).collect::<Vec<_>>() {
            return Err(ComposeRefusal::NotThisChain);
        }
        let mut work = Work::default();
        let rows = self.eval_rec(order, &mut work);
        Ok((rows, work))
    }

    fn eval_rec(&self, order: &RelExpr, work: &mut Work) -> Rows {
        match order {
            RelExpr::Leaf(i) => *self.operands[*i].rows,
            RelExpr::Comp(a, b) => {
                let (a, b) = (self.eval_rec(a, work), self.eval_rec(b, work));
                compose(&a, &b, work)
            }
        }
    }
}

/// Row ORs performed: one per set bit of each left operand composed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Work {
    pub row_ors: u64,
}

/// `out[i] = OR over j in a[i] of b[j]`: Boolean composition over OR/AND.
pub fn compose(a: &Rows, b: &Rows, work: &mut Work) -> Rows {
    let mut out = [0u64; 64];
    for (o, &row) in out.iter_mut().zip(a.iter()) {
        let mut bits = row;
        while bits != 0 {
            *o |= b[bits.trailing_zeros() as usize];
            bits &= bits - 1;
            work.row_ors += 1;
        }
    }
    out
}

// ── law admission ──────────────────────────────────────────────────────────

/// Associativity, admitted for one law shape. No public constructor: only
/// [`admit_composition_assoc`] makes one.
///
/// ```compile_fail
/// use dr_math_binding_probe::relcomp::AdmittedLaw;
/// let forged = AdmittedLaw { shape: todo!() };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedLaw {
    shape: LawShape,
}

impl AdmittedLaw {
    pub fn shape(&self) -> LawShape {
        self.shape
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LawRefusal {
    /// Associativity is admitted for relation composition only.
    NotComposition,
    /// Only the Boolean semiring (two-valued, Distinct) is admitted here.
    /// Bag (counting) and three-valued composition are other semirings and
    /// are not admitted by this probe.
    NotBooleanSemiring,
}

pub fn admit_composition_assoc(shape: LawShape) -> Result<AdmittedLaw, LawRefusal> {
    if shape.primitive != Primitive::BooleanCompose {
        return Err(LawRefusal::NotComposition);
    }
    if shape.truth != Truth::TwoValued || shape.population != Population::Distinct {
        return Err(LawRefusal::NotBooleanSemiring);
    }
    Ok(AdmittedLaw { shape })
}

// ── proof (egglog) ─────────────────────────────────────────────────────────

const HEADER: &str = "(datatype Rel (Base String) (Comp Rel Rel))\n(ruleset assoc)\n\
(rewrite (Comp (Comp a b) c) (Comp a (Comp b c)) :ruleset assoc)\n\
(rewrite (Comp a (Comp b c)) (Comp (Comp a b) c) :ruleset assoc)\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofRefusal {
    /// The two orders do not share leaves; no rewrite law could relate them.
    DifferentLeaves,
    /// egglog found no proof from the admitted rule (or failed); its message.
    NotProved(String),
}

/// Ask egglog to prove `from = to` using only the admitted associativity rule
/// (both directions). Returns egglog's proof text; egglog checks every proof
/// it returns.
pub fn prove_reassociation(
    law: &AdmittedLaw,
    from: &RelExpr,
    to: &RelExpr,
) -> Result<String, ProofRefusal> {
    if from.leaves() != to.leaves() {
        return Err(ProofRefusal::DifferentLeaves);
    }
    // Every order of an n-chain is reachable in at most n - 2 rotations.
    let fuel = from.leaves().len().max(2);
    prove_equation(law, &from.egglog_term(), &to.egglog_term(), fuel)
}

/// Ask egglog to prove an arbitrary equation between two `Rel` terms from the
/// admitted rule, running it `fuel` times. Proves only; issues nothing.
pub fn prove_equation(
    _law: &AdmittedLaw,
    lhs: &str,
    rhs: &str,
    fuel: usize,
) -> Result<String, ProofRefusal> {
    let program = format!(
        "{HEADER}(let lhs {lhs})\n(let rhs {rhs})\n(run assoc {fuel})\n(prove (= lhs rhs))\n"
    );
    EGraph::new_with_proofs()
        .parse_and_run_program(None, &program)
        .map_err(|e| ProofRefusal::NotProved(e.to_string()))?
        .into_iter()
        .find_map(|out| match out {
            CommandOutput::ProveExists {
                proof_store,
                proof_id,
            } => Some(proof_store.proof_to_string(proof_id)),
            _ => None,
        })
        .ok_or_else(|| ProofRefusal::NotProved("no proof returned".to_owned()))
}

// ── lowering ───────────────────────────────────────────────────────────────

/// One compose step over a value stack: leaves are pushed in chain order,
/// `Compose` pops right then left and pushes their composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Push(usize),
    Compose,
}

pub fn lower(order: &RelExpr) -> Vec<Step> {
    match order {
        RelExpr::Leaf(i) => vec![Step::Push(*i)],
        RelExpr::Comp(a, b) => {
            let mut v = lower(a);
            v.extend(lower(b));
            v.push(Step::Compose);
            v
        }
    }
}

/// The order a step list computes, or `None` if it is malformed.
pub fn raise(steps: &[Step]) -> Option<RelExpr> {
    let mut stack = Vec::new();
    for s in steps {
        match *s {
            Step::Push(i) => stack.push(RelExpr::Leaf(i)),
            Step::Compose => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(RelExpr::comp(a, b));
            }
        }
    }
    if stack.len() == 1 { stack.pop() } else { None }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringRefusal {
    /// The steps do not raise back to the order they were lowered from.
    NotRoundTrip,
}

/// Structural lowering check for one order.
pub fn validate_lowering(order: &RelExpr, steps: &[Step]) -> Result<(), LoweringRefusal> {
    if raise(steps).as_ref() != Some(order) {
        return Err(LoweringRefusal::NotRoundTrip);
    }
    Ok(())
}

// ── certificate ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertifyRefusal {
    /// The law reference names a different shape than the admitted law.
    LawNotAdmitted,
    Proof(ProofRefusal),
    Lowering(LoweringRefusal),
    MissingCodeIdentity,
}

/// Certify the recipe "evaluate `from` as `to`": the law is admitted for the
/// reference's shape, egglog proves `from = to` from the admitted rule, and
/// `to` lowers and raises back to itself.
pub fn certify_reassociation(
    admitted: &AdmittedLaw,
    law: LawReference,
    from: &RelExpr,
    to: &RelExpr,
    compiled: CompiledReference,
) -> Result<RecipeCertificate, CertifyRefusal> {
    if law.shape != admitted.shape || compiled.law != law {
        return Err(CertifyRefusal::LawNotAdmitted);
    }
    let proof = prove_reassociation(admitted, from, to).map_err(CertifyRefusal::Proof)?;
    validate_lowering(to, &lower(to)).map_err(CertifyRefusal::Lowering)?;
    if compiled.compiler_revision == 0 || compiled.compiled_digest == [0u8; 32] {
        return Err(CertifyRefusal::MissingCodeIdentity);
    }
    Ok(RecipeCertificate::issue(law, compiled, proof))
}
