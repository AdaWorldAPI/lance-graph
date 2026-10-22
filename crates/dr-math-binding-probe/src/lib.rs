//! D-DRMC-1: borrowed typed DR64 operand admission, NOT an evaluator.
//!
//! This is a stand-alone probe of the ClassView -> Quack-style BIND contract.
//! It does not implement a ClassView registry, own RelationshipMap data, decide
//! mathematical equivalence, attest a source, or run any per-row operation.
//! Canonical code will remain in the existing owners; this lab tests the shape.
//!
//! The bitplane is precisely 64 rows of 64 participation bits (512 B). It
//! carries neither global ids nor SQL NULL, bag multiplicity or provenance.
#![forbid(unsafe_code)]

use lance_graph_contract::class_view::ClassId;

/// What one numeric coordinate means according to the resolved ClassView.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisKind {
    Identity,
    Property,
    Community,
    Capability,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    pub class: ClassId,
    pub kind: AxisKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Truth {
    TwoValued,
    ThreeValued,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Population {
    Distinct,
    Bag,
}

/// Per-method semantic law; *not* a single semantics choice for a whole class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodSignature {
    pub owner_class: ClassId,
    pub method_ordinal: u16,
    pub row: Axis,
    pub column: Axis,
    pub truth: Truth,
    pub population: Population,
}

impl MethodSignature {
    pub const fn presence(
        owner_class: ClassId,
        method_ordinal: u16,
        row: Axis,
        column: Axis,
    ) -> Self {
        Self {
            owner_class,
            method_ordinal,
            row,
            column,
            truth: Truth::TwoValued,
            population: Population::Distinct,
        }
    }
}

/// Caller-provided *values*, not independently attested source identities.
/// Source owners must attest these fields before production use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingWitness {
    pub source_revision: u64,
    pub relation_map_revision: u64,
    pub coordinate_revision: u64,
    pub focus_revision: u64,
    pub authorization_scope: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MethodChanged,
    AxisChanged,
    SemanticsChanged,
    ThreeValuedNeedsValidity,
    BagNeedsMultiplicity,
    SourceChanged,
    MapChanged,
    CoordinatesChanged,
    FocusChanged,
    ScopeChanged,
}

/// Borrowed fixed-width register, with resolved type and current witness.
/// Never owns the payload or reinterprets bits into foreign keys.
#[derive(Clone, Copy, Debug)]
pub struct BoundDr64<'a> {
    pub method: MethodSignature,
    pub witness: BindingWitness,
    pub rows: &'a [u64; 64],
}

/// Validate *the declaration and equality of supplied witnesses*, not their
/// authenticity. A production binder must additionally check the sealed
/// source/index identity, ClassView registry, tenant ACL, and lifetime.
pub fn bind_dr64<'a>(
    declared: MethodSignature,
    requested: MethodSignature,
    bound_at: BindingWitness,
    observed_now: BindingWitness,
    rows: &'a [u64; 64],
) -> Result<BoundDr64<'a>, Refusal> {
    if (declared.owner_class, declared.method_ordinal)
        != (requested.owner_class, requested.method_ordinal)
    {
        return Err(Refusal::MethodChanged);
    }
    if (declared.row, declared.column) != (requested.row, requested.column) {
        return Err(Refusal::AxisChanged);
    }
    if (declared.truth, declared.population) != (requested.truth, requested.population) {
        return Err(Refusal::SemanticsChanged);
    }
    if declared.truth != Truth::TwoValued {
        return Err(Refusal::ThreeValuedNeedsValidity);
    }
    if declared.population != Population::Distinct {
        return Err(Refusal::BagNeedsMultiplicity);
    }
    if bound_at.source_revision != observed_now.source_revision {
        return Err(Refusal::SourceChanged);
    }
    if bound_at.relation_map_revision != observed_now.relation_map_revision {
        return Err(Refusal::MapChanged);
    }
    if bound_at.coordinate_revision != observed_now.coordinate_revision {
        return Err(Refusal::CoordinatesChanged);
    }
    if bound_at.focus_revision != observed_now.focus_revision {
        return Err(Refusal::FocusChanged);
    }
    if bound_at.authorization_scope != observed_now.authorization_scope {
        return Err(Refusal::ScopeChanged);
    }
    Ok(BoundDr64 {
        method: declared,
        witness: observed_now,
        rows,
    })
}

/// Independent 64x64 finite-population reference audit; NEVER a production evaluator.
pub mod population;

/// Reusable mathematical law / proof / compiled artifact identity separation (lab only).
pub mod recipe;
