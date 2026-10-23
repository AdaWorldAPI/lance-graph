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
//!    association orders of the same chain are equal, and checks the proof
//!    with its own checker (not an independent check). An equation the admitted rule does not imply (commutation) is
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
use std::cell::Cell;

use egglog::proof::{Justification, ProofId, ProofStore};
use egglog::{CommandOutput, EGraph};
use sha2::{Digest, Sha256};

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
///
/// The chain borrows the store it was checked against, so the frame stays
/// current until the chain is dropped: advancing or dropping the store while
/// a chain is alive does not compile.
///
/// ```compile_fail
/// use dr_math_binding_probe::{FrameRevisions, FrameStore, relcomp::Chain};
/// let mut st = FrameStore::new(FrameRevisions { source: 1, relation_map: 1, coordinates: 1 });
/// let chain = Chain::new(&st, Vec::new());
/// st.advance(FrameRevisions { source: 2, relation_map: 1, coordinates: 1 });
/// drop(chain);
/// ```
#[derive(Debug)]
pub struct Chain<'a> {
    _store: &'a FrameStore,
    operands: Vec<BoundDr64<'a>>,
}

impl<'a> Chain<'a> {
    pub fn new(
        store: &'a FrameStore,
        operands: Vec<BoundDr64<'a>>,
    ) -> Result<Self, ComposeRefusal> {
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
        Ok(Self {
            _store: store,
            operands,
        })
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

thread_local! {
    static EGGLOG_CALLS: Cell<u64> = const { Cell::new(0) };
}

/// How many times this thread has run egglog through [`prove_equation`], the
/// only egglog call site in this crate. Lets a test assert that a cached
/// recipe executes without the prover. Thread-local, so parallel tests do not
/// see each other's calls.
pub fn egglog_calls() -> u64 {
    EGGLOG_CALLS.with(Cell::get)
}

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

/// egglog's proof of one equation, kept as egglog's own object: the
/// hash-consed proof arena and the id of the root proof in it. Never reduced
/// to a string or a digest; [`EggProof::render`] is for display only.
///
/// egglog checks this proof before returning it (its crate-private
/// `check_proof`, before and after simplification, panicking on failure).
/// That is egglog's own check, not an independent one.
#[derive(Clone)]
pub struct EggProof {
    store: ProofStore,
    root: ProofId,
}

impl EggProof {
    /// The proof arena. [`ProofStore::get`] reads any node in it.
    pub fn store(&self) -> &ProofStore {
        &self.store
    }

    /// The root proof, an id into [`EggProof::store`].
    pub fn root(&self) -> ProofId {
        self.root
    }

    /// egglog's pretty-printed rendering. For display; not the proof.
    pub fn render(&self) -> String {
        self.store.proof_to_string(self.root)
    }

    /// Distinct proof nodes reachable from the root; a shared sub-proof
    /// counts once.
    pub fn node_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut todo = vec![self.root];
        while let Some(id) = todo.pop() {
            if !seen.insert(id) {
                continue;
            }
            match self.store.get(id).justification() {
                Justification::Fiat | Justification::Eval => {}
                Justification::Rule { premise_proofs, .. } => todo.extend(premise_proofs),
                Justification::MergeFn {
                    old_proof,
                    new_proof,
                    ..
                } => todo.extend([*old_proof, *new_proof]),
                Justification::Trans(a, b) => todo.extend([*a, *b]),
                Justification::Sym(a) => todo.push(*a),
                Justification::Congr {
                    proof, child_proof, ..
                } => todo.extend([*proof, *child_proof]),
                Justification::ContainerNormalize { proof } => todo.push(*proof),
            }
        }
        seen.len()
    }
}

impl std::fmt::Debug for EggProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EggProof")
            .field("root", &format_args!("{}", self.root))
            .field("nodes", &self.node_count())
            .finish()
    }
}

/// Ask egglog to prove `from = to` using only the admitted associativity rule
/// (both directions). Returns egglog's proof object (see [`EggProof`]).
pub fn prove_reassociation(
    law: &AdmittedLaw,
    from: &RelExpr,
    to: &RelExpr,
) -> Result<EggProof, ProofRefusal> {
    if from.leaves() != to.leaves() {
        return Err(ProofRefusal::DifferentLeaves);
    }
    // Every order of an n-chain is reachable in at most n - 2 rotations.
    let fuel = from.leaves().len().max(2);
    prove_equation(law, &from.egglog_term(), &to.egglog_term(), fuel)
}

/// Ask egglog to prove an arbitrary equation between two `Rel` terms, running
/// it `fuel` times. Proves only; issues nothing.
///
/// The `_law` argument gates the call (only an [`AdmittedLaw`] can be passed)
/// but does not choose the rules: the rewrites are fixed in `HEADER`, the two
/// directions of associativity, the only law this probe admits.
pub fn prove_equation(
    _law: &AdmittedLaw,
    lhs: &str,
    rhs: &str,
    fuel: usize,
) -> Result<EggProof, ProofRefusal> {
    EGGLOG_CALLS.with(|n| n.set(n.get() + 1));
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
            } => Some(EggProof {
                store: proof_store,
                root: proof_id,
            }),
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

/// SHA-256 over a canonical encoding of a step list: a domain tag, then per
/// step `0x00` + the leaf index as `u64` little-endian, or `0x01` for
/// `Compose`. This is the compiled identity a reassociation certificate is
/// bound to; the artifact is the step list, evaluated by the compose kernel.
pub fn lowering_digest(steps: &[Step]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"dr-math-binding-probe/lowering/v1");
    for s in steps {
        match *s {
            Step::Push(i) => {
                h.update([0u8]);
                h.update((i as u64).to_le_bytes());
            }
            Step::Compose => h.update([1u8]),
        }
    }
    h.finalize().into()
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
    /// The compiled digest is not the digest of `to`'s checked lowering.
    CompiledMismatch,
}

/// Certify the recipe "evaluate `from` as `to`": the law is admitted for the
/// reference's shape, egglog proves `from = to` from the admitted rule, `to`
/// lowers and raises back to itself, and `compiled.compiled_digest` is the
/// [`lowering_digest`] of that checked lowering. A caller cannot attach a
/// digest of anything else.
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
    let steps = lower(to);
    validate_lowering(to, &steps).map_err(CertifyRefusal::Lowering)?;
    if compiled.compiler_revision == 0 || compiled.compiled_digest == [0u8; 32] {
        return Err(CertifyRefusal::MissingCodeIdentity);
    }
    if compiled.compiled_digest != lowering_digest(&steps) {
        return Err(CertifyRefusal::CompiledMismatch);
    }
    let source_digest = lowering_digest(&lower(from));
    Ok(RecipeCertificate::issue(
        law,
        compiled,
        source_digest,
        steps,
        proof,
    ))
}

// ── reuse on a bound chain (X8) ────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReuseRefusal {
    /// The runtime ABI is not the one the certificate was compiled for.
    WrongTarget,
    /// The requested order is not the query the certificate answers.
    NotCertifiedQuery,
    /// The requested order does not cover this chain's positions in order.
    NotThisChain,
    /// The certified program does not run on this chain (malformed steps or a
    /// leaf outside the chain).
    ProgramDoesNotFit,
}

/// Answer `requested` on `chain` with a certified recipe, without proving
/// anything: check the target, check that `requested` is the query the
/// certificate answers, then run the certified step list with the compose
/// kernel. The chain carries its own frame, focus, scope and semiring guards
/// (checked by [`Chain::new`]); the certificate carries none of them, so the
/// same certificate serves any chain that passes them.
///
/// On any refusal nothing is computed: the caller decides on a baseline.
pub fn execute_certified(
    cert: &RecipeCertificate,
    chain: &Chain<'_>,
    requested: &RelExpr,
    runtime_abi: u32,
) -> Result<(Rows, Work), ReuseRefusal> {
    if cert.compiled().target_abi != runtime_abi {
        return Err(ReuseRefusal::WrongTarget);
    }
    if lowering_digest(&lower(requested)) != *cert.source_digest() {
        return Err(ReuseRefusal::NotCertifiedQuery);
    }
    if requested.leaves() != (0..chain.len()).collect::<Vec<_>>() {
        return Err(ReuseRefusal::NotThisChain);
    }
    run_steps(chain, cert.program()).ok_or(ReuseRefusal::ProgramDoesNotFit)
}

fn run_steps(chain: &Chain<'_>, steps: &[Step]) -> Option<(Rows, Work)> {
    let mut work = Work::default();
    let mut stack: Vec<Rows> = Vec::new();
    for s in steps {
        match *s {
            Step::Push(i) => stack.push(*chain.operands.get(i)?.rows),
            Step::Compose => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(compose(&a, &b, &mut work));
            }
        }
    }
    if stack.len() == 1 {
        stack.pop().map(|r| (r, work))
    } else {
        None
    }
}
