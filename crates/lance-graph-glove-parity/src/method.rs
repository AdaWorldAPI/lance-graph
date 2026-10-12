//! Methods of a concept, bound once against one glove, run through Quack or
//! R2IL (D-CMH-1, first slice; `.claude/plans/class-method-schema-hotplug-v1.md`).
//!
//! A [`MethodSpec`] belongs to the CONCEPT: it names its fields in the
//! concept's canonical basis (the authority's `ClassView`), never in a
//! glove's native names. [`MethodTable::prepare`] binds every spec once,
//! cold, against one glove's claims and `Binder`, and records what it
//! refused. After that a call carries numbers only:
//!
//! ```text
//!   label ──resolve (cold)──► MethodKey
//!   MethodKey + borrowed planes ──call──► Quack Query ─► lower ─► mask-risc
//!                                     └─► R2IL procedure ─► FoldDialect ─► mask-risc
//! ```
//!
//! [`MethodTable::call`] is to methods what `sql()` is to Quack: one entry
//! point that turns a name into numbers once and leaves execution to the
//! algebra. A label is never an address: [`MethodTable::resolve`] turns it
//! into a [`MethodKey`], and only the key reaches a call.
//!
//! Two masks, two algebras:
//!
//! - [`PreparedMethod::required`] is a [`WideFieldMask`] over the canonical
//!   basis: WHICH FIELDS the method reads. A method prepares only when
//!   `required ⊆ available`, where `available` is the basis positions this
//!   glove has a proven (`Exact` or `Converted`) claim for.
//! - The row mask the method runs on lives in mask-risc and is never stored
//!   here.
//!
//! An action body prepares, so the table can name it, and is refused by
//! every pure call ([`CallError::NotPure`]).
//!
//! Scope of this slice: one pure body kind ([`MethodBody::CountWhereEq`]) on a
//! non-nullable `Code` field. Everything else is refused by name.
use lance_graph_contract::WideFieldMask;
use lance_graph_mask_risc::{execute_into, ExecError, Foreign, Out, Planes, Scratch, Value};
use lance_graph_quack::bind::{Binder, FieldKind};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, LowerError, Mask, Query};
use lance_graph_r2il::{procs, FoldDialect, FoldError, Val};

use crate::basis::{FieldMap, Grade};

/// A method's stable identity. Issued by the method's owner; never derived
/// from its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MethodKey(pub [u8; 16]);

/// What a method may do (OGAR #357 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectClass {
    /// Reads borrowed planes, returns a value.
    Pure,
    /// Reads a pinned dataset version.
    ReadVersion,
    /// Builds an inert proposal.
    Propose,
    /// Changes the world. Only OGAR's action gate runs it.
    ExternalMutate,
}

/// A method's body, in the concept's canonical field names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodBody {
    /// How many live rows have `field = literal`.
    CountWhereEq {
        field: &'static str,
        literal: &'static str,
    },
    /// An OGAR action. Named here; never executed here.
    Action { action: &'static str },
}

/// A method as its concept declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodSpec {
    pub key: MethodKey,
    /// Canonical concept id (`ogar_vocab::class_ids`).
    pub concept: u16,
    pub effect: EffectClass,
    pub body: MethodBody,
}

impl MethodSpec {
    /// The canonical fields the body reads.
    #[must_use]
    pub fn reads(&self) -> Vec<&'static str> {
        match self.body {
            MethodBody::CountWhereEq { field, .. } => vec![field],
            MethodBody::Action { .. } => Vec::new(),
        }
    }
}

/// Why a method did not prepare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareError {
    /// The spec belongs to another concept than the table's.
    WrongConcept { spec: u16, table: u16 },
    /// The body names a field the canonical basis does not have.
    UnknownField(&'static str),
    /// The basis is wider than a field position can address.
    BasisTooWide(usize),
    /// A required field has no proven claim in this glove
    /// (`required ⊄ available`).
    UnmappedRequired(&'static str),
    /// The glove's binder has no table of that name.
    UnknownTable(&'static str),
    /// The glove's binder does not serve the claimed field.
    Unbound(&'static str),
    /// The field is nullable. Three-valued `=` is not in this slice.
    Nullable(&'static str),
    /// The body needs a different field kind.
    KindMismatch(&'static str),
    /// The literal is not in the field's domain.
    UnknownValue {
        field: &'static str,
        literal: &'static str,
    },
    /// The declared effect does not match the body.
    EffectMismatch,
}

/// Why a call did not run.
#[derive(Debug)]
pub enum CallError {
    /// No method with that key.
    UnknownMethod,
    /// The method was refused at preparation.
    Refused(PrepareError),
    /// The method is an action; a pure call never runs it.
    NotPure,
    Lower(LowerError),
    Exec(ExecError),
    Fold(FoldError),
}

/// Which frontend lowers the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frontend {
    Quack,
    R2il,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    CountEq { live: Mask, col: Col, code: u32 },
    Action,
}

/// A method bound against one glove: numbers only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedMethod {
    pub key: MethodKey,
    pub effect: EffectClass,
    /// The canonical fields the method reads, as basis positions.
    pub required: WideFieldMask,
    bound: Bound,
}

impl PreparedMethod {
    /// The bound Quack query of a pure body; `None` for an action.
    #[must_use]
    pub fn query(&self) -> Option<Query> {
        match self.bound {
            Bound::CountEq { live, col, code } => Some(Query {
                filter: Filter::and([Filter::plane(live), Filter::cmp(col, Cmp::EqU32(code))]),
                agg: Agg::Count,
            }),
            Bound::Action => None,
        }
    }

    /// Run the method over `planes` through `frontend`.
    ///
    /// # Errors
    ///
    /// [`CallError::NotPure`] for an action, otherwise the frontend's own
    /// refusal.
    pub fn call(&self, planes: &Planes<'_>, frontend: Frontend) -> Result<i64, CallError> {
        let Bound::CountEq { live, col, code } = self.bound else {
            return Err(CallError::NotPure);
        };
        match frontend {
            Frontend::Quack => {
                let program = lower(&self.query().expect("pure")).map_err(CallError::Lower)?;
                let mut scratch =
                    Scratch::for_program(&program, planes.n_rows).map_err(CallError::Exec)?;
                match execute_into(&program, planes, &Foreign::NONE, &mut scratch, Out::None)
                    .map_err(CallError::Exec)?
                {
                    Value::Count(n) => Ok(i64::try_from(n).expect("a row count fits i64")),
                    other => panic!("a Count terminal returns Count, got {other:?}"),
                }
            }
            Frontend::R2il => {
                let mut d = FoldDialect::new(planes, &Foreign::NONE);
                let mut stack = Vec::new();
                procs::count_eq(&mut d, &mut stack, live.0, col.0, code)
                    .map_err(CallError::Fold)?;
                match stack.as_slice() {
                    [Val::Scalar(n)] => Ok(*n),
                    other => panic!("count_eq leaves one scalar, got {other:?}"),
                }
            }
        }
    }
}

/// One concept's methods, bound against one glove.
#[derive(Debug)]
pub struct MethodTable {
    labels: Vec<(&'static str, MethodKey)>,
    methods: Vec<(MethodKey, Result<PreparedMethod, PrepareError>)>,
}

impl MethodTable {
    /// Bind every method of `concept` once.
    ///
    /// - `basis`: the concept's canonical fields in `ClassView` order;
    /// - `claims`: this glove's native → canonical claims;
    /// - `binder`, `table`: this glove's resolution membrane;
    /// - `specs`: `(label, spec)`; the label is display text only.
    ///
    /// A refused method stays in the table with its reason.
    #[must_use]
    pub fn prepare(
        concept: u16,
        basis: &[String],
        claims: &[FieldMap],
        binder: &dyn Binder,
        table: &'static str,
        specs: &[(&'static str, MethodSpec)],
    ) -> Self {
        let available = available(basis, claims);
        Self {
            labels: specs.iter().map(|(l, s)| (*l, s.key)).collect(),
            methods: specs
                .iter()
                .map(|(_, s)| {
                    (
                        s.key,
                        prepare_one(concept, basis, claims, available.as_ref(), binder, table, s),
                    )
                })
                .collect(),
        }
    }

    /// The key a label names. Cold: done once, never per call.
    #[must_use]
    pub fn resolve(&self, label: &str) -> Option<MethodKey> {
        self.labels
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, k)| *k)
    }

    /// The prepared method, or why it was refused.
    #[must_use]
    pub fn get(&self, key: MethodKey) -> Option<&Result<PreparedMethod, PrepareError>> {
        self.methods.iter().find(|(k, _)| *k == key).map(|(_, m)| m)
    }

    /// Call a method by key.
    ///
    /// # Errors
    ///
    /// [`CallError::UnknownMethod`], [`CallError::Refused`], or the
    /// method's own call error.
    pub fn call(
        &self,
        key: MethodKey,
        planes: &Planes<'_>,
        frontend: Frontend,
    ) -> Result<i64, CallError> {
        match self.get(key).ok_or(CallError::UnknownMethod)? {
            Ok(m) => m.call(planes, frontend),
            Err(e) => Err(CallError::Refused(e.clone())),
        }
    }
}

/// Basis positions with a proven claim (`Exact` or `Converted`).
fn available(basis: &[String], claims: &[FieldMap]) -> Result<WideFieldMask, PrepareError> {
    let mut mask = WideFieldMask::EMPTY;
    for (i, name) in basis.iter().enumerate() {
        let proven = claims
            .iter()
            .any(|c| c.canonical == name && matches!(c.grade, Grade::Exact | Grade::Converted));
        if proven {
            mask = mask.with(u8::try_from(i).map_err(|_| PrepareError::BasisTooWide(basis.len()))?);
        }
    }
    Ok(mask)
}

fn prepare_one(
    concept: u16,
    basis: &[String],
    claims: &[FieldMap],
    available: Result<&WideFieldMask, &PrepareError>,
    binder: &dyn Binder,
    table: &'static str,
    spec: &MethodSpec,
) -> Result<PreparedMethod, PrepareError> {
    if spec.concept != concept {
        return Err(PrepareError::WrongConcept {
            spec: spec.concept,
            table: concept,
        });
    }
    let available = available.map_err(Clone::clone)?;
    let mut required = WideFieldMask::EMPTY;
    for field in spec.reads() {
        let pos = basis
            .iter()
            .position(|b| b == field)
            .ok_or(PrepareError::UnknownField(field))?;
        required =
            required.with(u8::try_from(pos).map_err(|_| PrepareError::BasisTooWide(basis.len()))?);
    }
    if !required.is_subset_of(available) {
        let missing = spec
            .reads()
            .into_iter()
            .find(|f| {
                let pos = basis.iter().position(|b| b == f).expect("checked above");
                !available.has(u8::try_from(pos).expect("checked above"))
            })
            .expect("a field outside available exists");
        return Err(PrepareError::UnmappedRequired(missing));
    }
    let bound = match (spec.body, spec.effect) {
        (MethodBody::CountWhereEq { field, literal }, EffectClass::Pure) => {
            let t = binder
                .table(table)
                .ok_or(PrepareError::UnknownTable(table))?;
            let claim = claims
                .iter()
                .find(|c| c.canonical == field)
                .expect("available implies a claim");
            // A converted claim is served under its canonical name (the
            // derived lens the conversion is proven on); an exact one under
            // the glove's own name.
            let name = match claim.grade {
                Grade::Converted => claim.canonical,
                _ => claim.native,
            };
            let f = binder.field(t, name).ok_or(PrepareError::Unbound(field))?;
            if f.validity.is_some() {
                return Err(PrepareError::Nullable(field));
            }
            if f.kind != FieldKind::Code {
                return Err(PrepareError::KindMismatch(field));
            }
            let code = binder
                .code(t, f.col, literal)
                .ok_or(PrepareError::UnknownValue { field, literal })?;
            Bound::CountEq {
                live: binder.live(t),
                col: f.col,
                code,
            }
        }
        (MethodBody::Action { .. }, EffectClass::Propose | EffectClass::ExternalMutate) => {
            Bound::Action
        }
        _ => return Err(PrepareError::EffectMismatch),
    };
    Ok(PreparedMethod {
        key: spec.key,
        effect: spec.effect,
        required,
        bound,
    })
}
