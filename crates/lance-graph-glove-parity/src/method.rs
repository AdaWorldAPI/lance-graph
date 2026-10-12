//! Methods of a concept, bound once against one glove, compiled into Quack
//! algebra (D-CMH-1; `.claude/plans/class-method-schema-hotplug-v1.md`).
//!
//! A method is DATA, not a Rust function. A [`MethodSpec`] belongs to the
//! concept and declares its body as a query in the concept's canonical field
//! names (the authority's `ClassView`): a conjunction of conditions, a list of
//! folds, and a [`Law`] that turns the folds' values into the result. Adding a
//! method adds a spec; it adds no Rust.
//!
//! [`MethodTable::prepare`] binds every spec once, cold, against one glove's
//! claims and `Binder`, and records what it refused. After that a call carries
//! numbers only, and one generic lowering serves every method:
//!
//! ```text
//!   label ──resolve (cold)──► MethodKey
//!   MethodKey + borrowed planes ──call──► Quack Query per fold ─► lower ─► mask-risc
//!                                     └─► R2IL calls per fold ─► FoldDialect ─► mask-risc
//!                                     └─► Law ─► result
//! ```
//!
//! The chain this serves: a caller writes `sql(query)`; a class-qualified
//! method inside it (`Class::method(payload)`) expands into Quack algebra; Quack
//! compiles it. R2IL is the second frontend onto the same positions and checks
//! parity. A label is never an address: [`MethodTable::resolve`] turns it into
//! a [`MethodKey`], and only the key reaches a call.
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
//! Scope: conditions are `=`/`!=` on non-nullable `Code` fields; folds are
//! `COUNT` and `SUM` over a non-nullable `I32` field; the only law is
//! [`Law::Values`]. Everything else is refused by name.
use lance_graph_contract::WideFieldMask;
use lance_graph_mask_risc::{execute_into, ExecError, Foreign, Out, Planes, Scratch, Value};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
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

/// One condition of a body's `WHERE`, on a canonical field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// `field = literal`.
    Eq {
        field: &'static str,
        literal: &'static str,
    },
    /// `field != literal`.
    Ne {
        field: &'static str,
        literal: &'static str,
    },
}

impl Cond {
    fn field(self) -> &'static str {
        match self {
            Cond::Eq { field, .. } | Cond::Ne { field, .. } => field,
        }
    }
}

/// One aggregate of a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// `COUNT(*)` over the selected rows.
    Count,
    /// `SUM(field)` over the selected rows.
    Sum(&'static str),
}

/// How the folds' values become the method's result. A fixed set the
/// compiler knows; a method names one, it never supplies one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// The fold values, in order.
    Values,
}

/// A method's body, in the concept's canonical field names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodBody {
    /// `SELECT folds FROM concept WHERE filter`, finished by `law`.
    Query {
        filter: &'static [Cond],
        folds: &'static [Fold],
        law: Law,
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
    /// The canonical fields the body reads, each once, in order of use.
    #[must_use]
    pub fn reads(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if let MethodBody::Query { filter, folds, .. } = self.body {
            let fields = filter
                .iter()
                .map(|c| c.field())
                .chain(folds.iter().filter_map(|f| match f {
                    Fold::Count => None,
                    Fold::Sum(field) => Some(*field),
                }));
            for f in fields {
                if !out.contains(&f) {
                    out.push(f);
                }
            }
        }
        out
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
    /// The field is nullable. Three-valued comparison is not in scope.
    Nullable(&'static str),
    /// The body uses the field as a different kind.
    KindMismatch(&'static str),
    /// The literal is not in the field's domain.
    UnknownValue {
        field: &'static str,
        literal: &'static str,
    },
    /// A query body with no fold computes nothing.
    NoFold,
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
enum BoundFold {
    Count,
    Sum(Col),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Bound {
    Query {
        live: Mask,
        /// `(lane, code, negate)`.
        conds: Vec<(Col, u32, bool)>,
        folds: Vec<BoundFold>,
        law: Law,
    },
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
    /// The bound Quack queries of a query body, one per fold (a mask-risc
    /// program has one terminal); empty for an action.
    #[must_use]
    pub fn queries(&self) -> Vec<Query> {
        let Bound::Query {
            live, conds, folds, ..
        } = &self.bound
        else {
            return Vec::new();
        };
        let filter = || {
            Filter::and(
                std::iter::once(Filter::plane(*live)).chain(conds.iter().map(
                    |&(col, code, negate)| {
                        Filter::cmp(
                            col,
                            if negate {
                                Cmp::NeU32(code)
                            } else {
                                Cmp::EqU32(code)
                            },
                        )
                    },
                )),
            )
        };
        folds
            .iter()
            .map(|f| Query {
                filter: filter(),
                agg: match f {
                    BoundFold::Count => Agg::Count,
                    BoundFold::Sum(col) => Agg::SumI32(*col),
                },
            })
            .collect()
    }

    /// Run the method over `planes` through `frontend`.
    ///
    /// # Errors
    ///
    /// [`CallError::NotPure`] for an action, otherwise the frontend's own
    /// refusal.
    pub fn call(&self, planes: &Planes<'_>, frontend: Frontend) -> Result<Vec<i64>, CallError> {
        let Bound::Query {
            live,
            conds,
            folds,
            law,
        } = &self.bound
        else {
            return Err(CallError::NotPure);
        };
        let values = match frontend {
            Frontend::Quack => self
                .queries()
                .iter()
                .map(|q| run_quack(q, planes))
                .collect::<Result<Vec<_>, _>>()?,
            Frontend::R2il => {
                let conds: Vec<(u16, u32, bool)> =
                    conds.iter().map(|&(c, v, n)| (c.0, v, n)).collect();
                folds
                    .iter()
                    .map(|f| run_r2il(*live, &conds, *f, planes))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        match law {
            Law::Values => Ok(values),
        }
    }
}

fn run_quack(q: &Query, planes: &Planes<'_>) -> Result<i64, CallError> {
    let program = lower(q).map_err(CallError::Lower)?;
    let mut scratch = Scratch::for_program(&program, planes.n_rows).map_err(CallError::Exec)?;
    match execute_into(&program, planes, &Foreign::NONE, &mut scratch, Out::None)
        .map_err(CallError::Exec)?
    {
        Value::Count(n) => Ok(i64::try_from(n).expect("a row count fits i64")),
        Value::SumI64(s) => Ok(s),
        other => panic!("a Count or SumI32 terminal, got {other:?}"),
    }
}

/// The same selection is emitted once per fold: a fold ends the epoch of
/// every slot built before it.
fn run_r2il(
    live: Mask,
    conds: &[(u16, u32, bool)],
    fold: BoundFold,
    planes: &Planes<'_>,
) -> Result<i64, CallError> {
    let mut d = FoldDialect::new(planes, &Foreign::NONE);
    let mut stack = Vec::new();
    procs::select_u32(&mut d, &mut stack, live.0, conds).map_err(CallError::Fold)?;
    match fold {
        BoundFold::Count => procs::count(&mut d, &mut stack),
        BoundFold::Sum(col) => procs::sum(&mut d, &mut stack, col.0),
    }
    .map_err(CallError::Fold)?;
    match stack.as_slice() {
        [Val::Scalar(n)] => Ok(*n),
        other => panic!("a fold leaves one scalar, got {other:?}"),
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
        let glove = Glove {
            basis,
            claims,
            binder,
            table,
        };
        Self {
            labels: specs.iter().map(|(l, s)| (*l, s.key)).collect(),
            methods: specs
                .iter()
                .map(|(_, s)| (s.key, glove.prepare(concept, available.as_ref(), s)))
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
    ) -> Result<Vec<i64>, CallError> {
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

/// One glove's resolution inputs.
struct Glove<'a> {
    basis: &'a [String],
    claims: &'a [FieldMap],
    binder: &'a dyn Binder,
    table: &'static str,
}

impl Glove<'_> {
    fn prepare(
        &self,
        concept: u16,
        available: Result<&WideFieldMask, &PrepareError>,
        spec: &MethodSpec,
    ) -> Result<PreparedMethod, PrepareError> {
        if spec.concept != concept {
            return Err(PrepareError::WrongConcept {
                spec: spec.concept,
                table: concept,
            });
        }
        let available = available.map_err(Clone::clone)?;
        let reads = spec.reads();
        let mut required = WideFieldMask::EMPTY;
        let mut missing = None;
        for field in &reads {
            let pos = self.position(field)?;
            required = required.with(pos);
            if missing.is_none() && !available.has(pos) {
                missing = Some(*field);
            }
        }
        if !required.is_subset_of(available) {
            return Err(PrepareError::UnmappedRequired(
                missing.expect("a field outside available exists"),
            ));
        }
        let bound = match (spec.body, spec.effect) {
            (MethodBody::Query { filter, folds, law }, EffectClass::Pure) => {
                if folds.is_empty() {
                    return Err(PrepareError::NoFold);
                }
                let t = self
                    .binder
                    .table(self.table)
                    .ok_or(PrepareError::UnknownTable(self.table))?;
                let conds = filter
                    .iter()
                    .map(|c| self.cond(t, *c))
                    .collect::<Result<Vec<_>, _>>()?;
                let folds = folds
                    .iter()
                    .map(|f| match f {
                        Fold::Count => Ok(BoundFold::Count),
                        Fold::Sum(field) => {
                            let b = self.field(t, field)?;
                            if b.kind != FieldKind::I32 {
                                return Err(PrepareError::KindMismatch(field));
                            }
                            Ok(BoundFold::Sum(b.col))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Bound::Query {
                    live: self.binder.live(t),
                    conds,
                    folds,
                    law,
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

    fn position(&self, field: &'static str) -> Result<u8, PrepareError> {
        let pos = self
            .basis
            .iter()
            .position(|b| b == field)
            .ok_or(PrepareError::UnknownField(field))?;
        u8::try_from(pos).map_err(|_| PrepareError::BasisTooWide(self.basis.len()))
    }

    /// A required field, through its proven claim. A converted claim is
    /// served under its canonical name (the derived lens the conversion is
    /// proven on); an exact one under the glove's own name.
    fn field(&self, t: TableId, field: &'static str) -> Result<BoundField, PrepareError> {
        let claim = self
            .claims
            .iter()
            .find(|c| c.canonical == field && matches!(c.grade, Grade::Exact | Grade::Converted))
            .expect("available implies a proven claim");
        let name = match claim.grade {
            Grade::Converted => claim.canonical,
            _ => claim.native,
        };
        let b = self
            .binder
            .field(t, name)
            .ok_or(PrepareError::Unbound(field))?;
        if b.validity.is_some() {
            return Err(PrepareError::Nullable(field));
        }
        Ok(b)
    }

    fn cond(&self, t: TableId, c: Cond) -> Result<(Col, u32, bool), PrepareError> {
        let (field, literal, negate) = match c {
            Cond::Eq { field, literal } => (field, literal, false),
            Cond::Ne { field, literal } => (field, literal, true),
        };
        let b = self.field(t, field)?;
        if b.kind != FieldKind::Code {
            return Err(PrepareError::KindMismatch(field));
        }
        let code = self
            .binder
            .code(t, b.col, literal)
            .ok_or(PrepareError::UnknownValue { field, literal })?;
        Ok((b.col, code, negate))
    }
}
