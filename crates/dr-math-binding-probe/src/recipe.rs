//! D-DRMC-3: mathematical LAW identity != compiled TARGET recipe != current DATA binding.
//!
//! Metadata-only guarded admission, NEVER a compiler, runtime scheduler,
//! binary loader, effect executor or data source attestor. The math law key
//! deliberately excludes class/method and source versions so a proof about a
//! mathematical shape can potentially be reused across ClassViews.
//!
//! Authority is split, and none of it sits here:
//! - egglog derives equivalences from admitted laws and checks its own proofs
//!   (egglog's checker, not an independent one);
//! - law admission owns the assumptions a law holds under (a finite-population
//!   report never admits a law);
//! - a certified-compiler boundary, after both of those and a lowering check,
//!   issues a [`RecipeCertificate`];
//! - Stage B's `ReplayCapsule` is replay and audit evidence for its own E1
//!   substitution language, not a proof authority for anything else.
//!
//! This single-operand layer has no certifier for any law shape, so it checks
//! everything it can and then refuses with `NoCertifiedCompiler`. One law is
//! certified elsewhere: associativity of Boolean-Distinct composition over a
//! chain of DR64 operands, by [`crate::relcomp::certify_reassociation`].
use crate::{AxisKind, BoundDr64, Population, Truth};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Selection,
    BooleanCompose,
    ArgmaxMasked,
    CountViolations,
}

/// Semantics on which a *single mathematical transformation* is conditional.
/// Not a ClassId/MethodId, not a data snapshot or a cache address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LawShape {
    pub primitive: Primitive,
    pub row_kind: AxisKind,
    pub column_kind: AxisKind,
    pub truth: Truth,
    pub population: Population,
}

impl LawShape {
    pub fn for_method(method: crate::MethodSignature, primitive: Primitive) -> Self {
        Self {
            primitive,
            row_kind: method.row.kind,
            column_kind: method.column.kind,
            truth: method.truth,
            population: method.population,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LawReference {
    pub id: u64,
    pub revision: u32,
    pub shape: LawShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    R2ilMaskRisc,
    ResidentDr64Simd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompiledReference {
    pub law: LawReference,
    pub compiled_digest: [u8; 32],
    pub compiler_revision: u32,
    pub target_abi: u32,
    pub backend: Backend,
}

/// The only thing that makes a recipe reusable: issued by the certified-
/// compiler boundary once egglog's proof check, law admission and the lowering
/// check have all passed. No public constructor and no caller-set status, so a
/// caller can neither declare a law checked nor attach a digest of its own.
///
/// ```compile_fail
/// use dr_math_binding_probe::recipe::RecipeCertificate;
/// let forged = RecipeCertificate { law: todo!(), compiled: todo!(), source_digest: [0; 32], program: vec![], proof: todo!() };
/// ```
///
/// Not `PartialEq`: egglog's proof arena has no equality, and two proofs of
/// one equation need not be the same proof.
#[derive(Clone, Debug)]
pub struct RecipeCertificate {
    law: LawReference,
    compiled: CompiledReference,
    /// Digest of the query the proof starts from (`lowering_digest` of the
    /// `from` order). The certificate answers this query and no other; the
    /// compiled digest names the program that answers it.
    source_digest: [u8; 32],
    /// The checked step list `compiled.compiled_digest` names, kept so a
    /// reuse runs the program without searching for it or re-lowering.
    program: Vec<crate::relcomp::Step>,
    /// egglog's proof object, not a rendering or a digest of it.
    proof: crate::relcomp::EggProof,
}

impl RecipeCertificate {
    /// Only the in-crate certifiers call this, after their own checks
    /// (today: [`crate::relcomp::certify_reassociation`]).
    pub(crate) fn issue(
        law: LawReference,
        compiled: CompiledReference,
        source_digest: [u8; 32],
        program: Vec<crate::relcomp::Step>,
        proof: crate::relcomp::EggProof,
    ) -> Self {
        Self {
            law,
            compiled,
            source_digest,
            program,
            proof,
        }
    }

    /// The checked step list this certificate's compiled digest names.
    pub fn program(&self) -> &[crate::relcomp::Step] {
        &self.program
    }

    /// Digest of the query this certificate answers.
    pub fn source_digest(&self) -> &[u8; 32] {
        &self.source_digest
    }

    /// The proof the certifier was given, as egglog's own object.
    pub fn proof(&self) -> &crate::relcomp::EggProof {
        &self.proof
    }

    pub fn law(&self) -> &LawReference {
        &self.law
    }

    pub fn compiled(&self) -> &CompiledReference {
        &self.compiled
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    PureMathematics,
    Proposal,
    EffectfulAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeRefusal {
    UnsupportedMathematicalShape,
    LawChanged,
    MissingCodeIdentity,
    WrongTarget,
    EffectRequiresSeparateAuthorization,
    /// Every other check passed, but no certified-compiler boundary exists
    /// for this law shape, so no certificate can be issued.
    NoCertifiedCompiler,
}

/// This is a *selection receipt* for a certified compiled artifact, not
/// executable machine code or a substitute for source/version validation.
#[derive(Clone, Debug)]
pub struct RecipeOffer<'a> {
    pub operand: BoundDr64<'a>,
    pub certificate: RecipeCertificate,
}

/// Cold-stage metadata admission for a mathematically typed, already-bound
/// register: effect capability, law shape, law identity, code identity and
/// target first, then certification. This single-operand path has no
/// certifier for any shape and refuses with `NoCertifiedCompiler`; the one
/// certifier that exists, for reassociating a composition chain, is
/// [`crate::relcomp::certify_reassociation`].
pub fn admit_recipe<'a>(
    operand: BoundDr64<'a>,
    primitive: Primitive,
    law: LawReference,
    compiled: CompiledReference,
    runtime_abi: u32,
    capability: Capability,
) -> Result<RecipeOffer<'a>, RecipeRefusal> {
    if capability != Capability::PureMathematics {
        return Err(RecipeRefusal::EffectRequiresSeparateAuthorization);
    }
    if law.shape != LawShape::for_method(operand.method, primitive) {
        return Err(RecipeRefusal::UnsupportedMathematicalShape);
    }
    if law != compiled.law {
        return Err(RecipeRefusal::LawChanged);
    }
    if compiled.compiler_revision == 0 || compiled.compiled_digest == [0u8; 32] {
        return Err(RecipeRefusal::MissingCodeIdentity);
    }
    if compiled.target_abi != runtime_abi {
        return Err(RecipeRefusal::WrongTarget);
    }
    let certificate = certify(law, compiled)?;
    Ok(RecipeOffer {
        operand,
        certificate,
    })
}

/// The single-operand certifier: none exists for any law shape.
fn certify(
    _law: LawReference,
    _compiled: CompiledReference,
) -> Result<RecipeCertificate, RecipeRefusal> {
    Err(RecipeRefusal::NoCertifiedCompiler)
}
