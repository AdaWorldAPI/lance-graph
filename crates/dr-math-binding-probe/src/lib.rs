//! D-DRMC-1: borrowed typed DR64 operand admission, NOT an evaluator.
//!
//! This is a stand-alone probe of the ClassView -> Quack-style BIND contract.
//! It does not implement a ClassView registry, own RelationshipMap data, decide
//! mathematical equivalence, attest a source, or run any per-row operation.
//! Canonical code will remain in the existing owners; this lab tests the shape.
//!
//! The bitplane is precisely 64 rows of 64 participation bits (512 B). It
//! carries neither global ids, bag multiplicity nor provenance. A three-valued
//! method carries its validity bits WITH the relation bits (see [`Operand`]).
#![forbid(unsafe_code)]

use std::sync::{Arc, Weak};

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

/// The revisions that identify the reference: which source, which relation
/// map, which coordinate assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRevisions {
    pub source: u64,
    pub relation_map: u64,
    pub coordinates: u64,
}

/// Lab stand-in for the owner of a frame. Only it issues [`Frame`]s, and a
/// frame compares by identity of that issue, never by value: two stores that
/// sit at the same numbers do not alias, and re-sealing equal numbers is a
/// new frame. The mechanism mirrors `dir-rmap-probe`'s `Witness` (a
/// `Weak<Snapshot>` compared by `ptr_eq`); that type is tied to dir-sim's
/// `VersionStore`, so it is not reused here.
#[derive(Debug)]
pub struct FrameStore {
    identity: Arc<()>,
    current: Arc<FrameRevisions>,
}

impl FrameStore {
    pub fn new(revisions: FrameRevisions) -> Self {
        Self {
            identity: Arc::new(()),
            current: Arc::new(revisions),
        }
    }

    /// Seal the current frame.
    pub fn seal(&self) -> Frame {
        Frame {
            store: Arc::downgrade(&self.identity),
            seal: Arc::downgrade(&self.current),
            at: *self.current,
        }
    }

    /// Move to new revisions; every frame sealed before is stale from now on.
    pub fn advance(&mut self, revisions: FrameRevisions) {
        self.current = Arc::new(revisions);
    }

    pub fn revisions(&self) -> FrameRevisions {
        *self.current
    }

    /// `Ok` iff `frame` is this store's current seal. A frame of another store
    /// is `ForeignFrame` even at equal numbers; a stale frame of this store
    /// names the first revision that moved, or `FrameResealed` if none did.
    pub fn check(&self, frame: &Frame) -> Result<(), Refusal> {
        if !Weak::ptr_eq(&frame.store, &Arc::downgrade(&self.identity)) {
            return Err(Refusal::ForeignFrame);
        }
        if Weak::ptr_eq(&frame.seal, &Arc::downgrade(&self.current)) {
            return Ok(());
        }
        let (then, now) = (frame.at, *self.current);
        Err(if then.source != now.source {
            Refusal::SourceChanged
        } else if then.relation_map != now.relation_map {
            Refusal::MapChanged
        } else if then.coordinates != now.coordinates {
            Refusal::CoordinatesChanged
        } else {
            Refusal::FrameResealed
        })
    }
}

/// A store-issued seal over the reference revisions. No public constructor: a
/// caller cannot build a frame from numbers, only receive one from
/// [`FrameStore::seal`].
#[derive(Clone, Debug)]
pub struct Frame {
    store: Weak<()>,
    seal: Weak<FrameRevisions>,
    at: FrameRevisions,
}

impl Frame {
    /// The revisions this frame was sealed at (for reporting, never identity).
    pub fn revisions(&self) -> FrameRevisions {
        self.at
    }

    /// Same issue of the same store.
    pub fn same_issue(&self, other: &Frame) -> bool {
        Weak::ptr_eq(&self.store, &other.store) && Weak::ptr_eq(&self.seal, &other.seal)
    }
}

/// Which 64x64 window of the frame is read. Motion, not reference: a new
/// focus over the same frame is not a new frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Focus {
    pub revision: u64,
}

/// What the reader is authorized to see. Policy, checked as its own guard at
/// borrow time; never part of frame identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scope {
    pub id: u64,
}

/// What a binding was made under.
#[derive(Clone, Debug)]
pub struct Guards {
    pub frame: Frame,
    pub focus: Focus,
    pub scope: Scope,
}

/// The borrowed relation bits and, for a three-valued method, which of them
/// are known. Validity is a carrier OF R (one bit per R bit, like an Arrow
/// validity bitmap or Quack's `BoundField.validity`), never a separately typed
/// plane with a method of its own.
#[derive(Clone, Copy, Debug)]
pub struct Operand<'a> {
    pub rows: &'a [u64; 64],
    pub validity: Option<&'a [u64; 64]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MethodChanged,
    AxisChanged,
    SemanticsChanged,
    ThreeValuedNeedsValidity,
    TwoValuedHasNoValidity,
    BagNeedsMultiplicity,
    ForeignFrame,
    SourceChanged,
    MapChanged,
    CoordinatesChanged,
    FrameResealed,
    FocusChanged,
    ScopeChanged,
}

/// Borrowed fixed-width register with its resolved type and guards. Never
/// owns the payload or reinterprets bits into foreign keys.
#[derive(Clone, Debug)]
pub struct BoundDr64<'a> {
    pub method: MethodSignature,
    pub frame: Frame,
    pub focus: Focus,
    pub scope: Scope,
    pub rows: &'a [u64; 64],
    pub validity: Option<&'a [u64; 64]>,
}

/// Validate the declaration, then each guard on its own: the frame against
/// the store that issued it, then the focus, then the scope. Checks seal
/// identity and equality of focus and scope, not their authenticity; a
/// production binder also needs the ClassView registry, the tenant ACL and
/// the source owner.
pub fn bind_dr64<'a>(
    store: &FrameStore,
    declared: MethodSignature,
    requested: MethodSignature,
    bound_at: &Guards,
    now: (Focus, Scope),
    operand: Operand<'a>,
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
    match (declared.truth, operand.validity) {
        (Truth::ThreeValued, None) => return Err(Refusal::ThreeValuedNeedsValidity),
        (Truth::TwoValued, Some(_)) => return Err(Refusal::TwoValuedHasNoValidity),
        _ => {}
    }
    if declared.population != Population::Distinct {
        return Err(Refusal::BagNeedsMultiplicity);
    }
    store.check(&bound_at.frame)?;
    if bound_at.focus != now.0 {
        return Err(Refusal::FocusChanged);
    }
    if bound_at.scope != now.1 {
        return Err(Refusal::ScopeChanged);
    }
    Ok(BoundDr64 {
        method: declared,
        frame: bound_at.frame.clone(),
        focus: now.0,
        scope: now.1,
        rows: operand.rows,
        validity: operand.validity,
    })
}

/// Independent 64x64 finite-population reference audit; NEVER a production evaluator.
pub mod population;

/// Reusable mathematical law / proof / compiled artifact identity separation (lab only).
pub mod recipe;

/// HHTL16 Cartesian direct routing: address map, rotation and fold-based
/// projections over a bound DR64 register (D-DRCA-1, lab only).
pub mod hhtl16;

/// Associativity of Boolean relation composition: typing, law admission,
/// egglog proof, lowering check and the first certificate.
pub mod relcomp;

/// D-DRCA-2: a global WHERE selection (65,536 addresses, three-valued)
/// projected into one table's 64-slot frame and read over DR#3, with every
/// reuse re-admitted against its guards.
pub mod where_dr3;

/// D-DRCA-3: DR #4 method routing over one admitted register. DT-owned
/// method key, guard witness and effect order are carried opaque.
pub mod dr4;
