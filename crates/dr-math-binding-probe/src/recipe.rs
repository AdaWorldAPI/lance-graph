//! D-DRMC-3: mathematical LAW identity != compiled TARGET recipe != current DATA binding.
//!
//! Metadata-only guarded admission, NEVER a compiler, proof checker, runtime
//! scheduler, binary loader, effect executor or data source attestor. The math
//! law key deliberately excludes class/method and source versions so a proof
//! about a mathematical shape can potentially be reused across ClassViews.
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

/// An assertion *from an external checking boundary*. Supplying `Checked` is
/// not proof! The production owner must run an independent checker against
/// the referenced statement, domain axioms and replay capsule, and attest it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofDisposition {
    Candidate,
    CheckedByExternalAuthority,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofReference {
    pub law: LawReference,
    pub checker_revision: u32,
    pub proof_digest: [u8; 32],
    pub status: ProofDisposition,
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
    UncheckedLaw,
    MissingProofIdentity,
    MissingCodeIdentity,
    WrongTarget,
    EffectRequiresSeparateAuthorization,
}

/// This is a *selection receipt* for an already compiled artifact, not
/// executable machine code or a substitute for source/version validation.
#[derive(Clone, Copy, Debug)]
pub struct RecipeOffer<'a> {
    pub operand: BoundDr64<'a>,
    pub artifact: CompiledReference,
    pub proof: ProofReference,
}

/// Cold-stage metadata selection for a mathematically typed, already-bound
/// register. It does not verify proof hashes or attest their provenance; it
/// refuses a missing/unchecked proof *even if* a finite-population report
/// happened to find zero counterexamples.
pub fn offer_recipe<'a>(
    operand: BoundDr64<'a>,
    primitive: Primitive,
    proof: ProofReference,
    compiled: CompiledReference,
    runtime_abi: u32,
    capability: Capability,
) -> Result<RecipeOffer<'a>, RecipeRefusal> {
    if capability != Capability::PureMathematics {
        return Err(RecipeRefusal::EffectRequiresSeparateAuthorization);
    }
    let observed_shape = LawShape::for_method(operand.method, primitive);
    if proof.law.shape != observed_shape {
        return Err(RecipeRefusal::UnsupportedMathematicalShape);
    }
    if proof.law != compiled.law {
        return Err(RecipeRefusal::LawChanged);
    }
    if proof.status != ProofDisposition::CheckedByExternalAuthority {
        return Err(RecipeRefusal::UncheckedLaw);
    }
    if proof.checker_revision == 0 || proof.proof_digest == [0u8; 32] {
        return Err(RecipeRefusal::MissingProofIdentity);
    }
    if compiled.compiler_revision == 0 || compiled.compiled_digest == [0u8; 32] {
        return Err(RecipeRefusal::MissingCodeIdentity);
    }
    if compiled.target_abi != runtime_abi {
        return Err(RecipeRefusal::WrongTarget);
    }
    Ok(RecipeOffer { operand, artifact: compiled, proof })
}
