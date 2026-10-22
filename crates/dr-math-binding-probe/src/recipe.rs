//! D-DRMC-3: mathematical LAW identity != compiled TARGET recipe != current DATA binding.
//!
//! Metadata-only guarded admission, NEVER a compiler, runtime scheduler,
//! binary loader, effect executor or data source attestor. The math law key
//! deliberately excludes class/method and source versions so a proof about a
//! mathematical shape can potentially be reused across ClassViews.
//!
//! Proof and target identity are Stage B's (`egglog-r2il-probe`): a
//! [`ProofReference`] exists only as a [`ReplayCapsule`] that egglog's own
//! checker re-proved, and the target is Stage B's [`TargetAbi`]. Nothing here
//! lets a caller declare a law checked. Stage B's checker proves exact-integer
//! substitution statements; no statement language or checker exists yet for a
//! Boolean-DR64 law, so the only statements this layer can carry are Stage B's.
use egglog_r2il_probe::artifact::ReplayCapsule;
use egglog_r2il_probe::catalogue::{SemanticKey, TargetAbi};

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

/// A law: its id and revision, the DR64 shape it applies to, and the statement
/// a proof must prove.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LawReference {
    pub id: u64,
    pub revision: u32,
    pub shape: LawShape,
    pub statement: SemanticKey,
}

/// A law together with a proof of its statement that was re-checked when this
/// value was made. Fields are private and the only constructor runs
/// [`ReplayCapsule::recheck`], so a caller can neither assert "checked" nor
/// attach a digest of its own.
///
/// ```compile_fail
/// use dr_math_binding_probe::recipe::ProofReference;
/// // No public fields: a proof cannot be written down, only re-checked.
/// let forged = ProofReference { law: todo!(), capsule: todo!() };
/// ```
#[derive(Clone, Debug)]
pub struct ProofReference {
    law: LawReference,
    capsule: ReplayCapsule,
}

impl ProofReference {
    /// Re-check `capsule` with a fresh egglog proof session, and require that
    /// it proves `law.statement`.
    pub fn rechecked(law: LawReference, capsule: ReplayCapsule) -> Result<Self, RecipeRefusal> {
        if SemanticKey::from(capsule.problem()) != law.statement {
            return Err(RecipeRefusal::ProofOfOtherStatement);
        }
        capsule.recheck().map_err(|_| RecipeRefusal::UncheckedLaw)?;
        Ok(Self { law, capsule })
    }

    pub fn law(&self) -> &LawReference {
        &self.law
    }

    /// egglog's checked proof rendering.
    pub fn proof(&self) -> &str {
        self.capsule.proof()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    R2ilMaskRisc,
    ResidentDr64Simd,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledReference {
    pub law: LawReference,
    pub compiled_digest: [u8; 32],
    pub compiler_revision: u32,
    pub target: TargetAbi,
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
    /// The proof did not survive a fresh re-check.
    UncheckedLaw,
    /// The proof re-checks, but for a different statement than the law's.
    ProofOfOtherStatement,
    MissingCodeIdentity,
    WrongTarget,
    EffectRequiresSeparateAuthorization,
}

/// This is a *selection receipt* for an already compiled artifact, not
/// executable machine code or a substitute for source/version validation.
#[derive(Clone, Debug)]
pub struct RecipeOffer<'a> {
    pub operand: BoundDr64<'a>,
    pub artifact: CompiledReference,
    pub proof: ProofReference,
}

/// Cold-stage metadata selection for a mathematically typed, already-bound
/// register. The proof was re-checked when the [`ProofReference`] was made; a
/// finite-population report cannot make one.
pub fn offer_recipe<'a>(
    operand: BoundDr64<'a>,
    primitive: Primitive,
    proof: ProofReference,
    compiled: CompiledReference,
    runtime: &TargetAbi,
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
    if compiled.compiler_revision == 0 || compiled.compiled_digest == [0u8; 32] {
        return Err(RecipeRefusal::MissingCodeIdentity);
    }
    if &compiled.target != runtime {
        return Err(RecipeRefusal::WrongTarget);
    }
    Ok(RecipeOffer {
        operand,
        artifact: compiled,
        proof,
    })
}
