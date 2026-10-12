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
//! A spec is bound only to the glove it was written for: its
//! [`SchemaSeal`] (the canonical basis, in order) and its [`ReadingId`] must
//! equal the glove's, and a call must name the [`Universe`] (source
//! generation and population) the method was prepared against.
//!
//! Scope: conditions are `=`/`!=` on non-nullable `Code` fields; folds are
//! `COUNT`, `SUM` and grouped `SUM` over non-nullable `I32` fields; laws are
//! [`Law::Values`] and [`Law::NarsRevision`]. Everything else is refused by
//! name.
//!
//! A [`Law`] is the one place a domain's semantics are written in Rust: it is
//! a certified finisher, checked against an independent oracle, and every
//! method of every concept that names it reuses it. A method never supplies
//! its own.
use lance_graph_contract::WideFieldMask;
use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, LowerError, Mask, Query};
use lance_graph_r2il::{procs, FoldDialect, FoldError, Val};

use crate::basis::{FieldMap, Grade};

/// A method's stable identity. Issued by the method's owner; never derived
/// from its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MethodKey(pub [u8; 16]);

/// The canonical basis a spec was written against: its field names, in
/// `ClassView` order. Reordering, renaming, adding or removing a field
/// changes the seal, so a spec written for one basis never binds another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaSeal(pub u64);

impl SchemaSeal {
    /// FNV-1a over the names, each terminated by `0xFF` (not a UTF-8 byte),
    /// so `["ab","c"]` and `["a","bc"]` differ.
    #[must_use]
    pub fn of(basis: &[String]) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for name in basis {
            for b in name.bytes().chain(std::iter::once(0xFF)) {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self(h)
    }
}

/// Which interpretation of the glove's bytes a spec assumes. Issued with the
/// reading (the ClassView's); a different reading of the same lanes is a
/// different id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadingId(pub u16);

/// The population a method was prepared against: source generation and row
/// universe, issued by whoever owns the planes. Rows from another universe
/// are not rows of this method's table, whatever their shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Universe(pub u64);

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
    /// `SUM(val) GROUP BY key` over the selected rows, for the key domain
    /// `0..groups`. Rows whose key is at or past `groups` are dropped by the
    /// terminal, not refused.
    GroupSum {
        key: &'static str,
        val: &'static str,
        groups: u32,
    },
}

/// How the folds' values become the method's result. A fixed set the
/// compiler knows; a method names one, it never supplies one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// The fold values, in order.
    Values,
    /// NAL revision of independent evidence, per statement.
    ///
    /// The body's folds must be exactly `[GroupSum { key, val: w },
    /// GroupSum { key, val: wf }]` over the same key and domain, where each
    /// row carries one piece of evidence as a fixed-point evidence weight
    /// `w = scale · c / (1 - c)` and `wf = round(w · f)`. Revision of
    /// independent evidence adds weights, so for each statement
    /// `f = Σwf / Σw` and `c = (Σw/scale) / (Σw/scale + horizon)`, computed
    /// once from the sums (never by chaining pairwise revisions).
    ///
    /// Adding weights is only revision when the evidence is independent: a
    /// repeated stamp would count twice. The body's filter must therefore
    /// contain `independent = "true"`, the payload's own mark of the first
    /// occurrence of each `(statement, stamp)`. The algebra cannot compute
    /// that mark yet (see the capability ledger); a body without it is
    /// refused.
    NarsRevision {
        /// The field marking the first occurrence of each evidential stamp.
        independent: &'static str,
        /// Fixed-point scale of `w`.
        scale: u32,
        /// The evidential horizon `k`.
        horizon: u32,
    },
}

/// A NAL truth value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Truth {
    pub f: f64,
    pub c: f64,
}

/// One fold's value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldValue {
    Scalar(i64),
    /// One value per group, `0..groups`.
    Groups(Vec<i64>),
}

/// A method's typed result.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// [`Law::Values`]: the fold values, in order.
    Values(Vec<FoldValue>),
    /// [`Law::NarsRevision`]: one truth per statement; `None` where no
    /// independent evidence was selected.
    Truths(Vec<Option<Truth>>),
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
    /// The basis the body's field names refer to.
    pub schema: SchemaSeal,
    /// The reading of the glove's bytes the body assumes.
    pub reading: ReadingId,
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
                .chain(folds.iter().flat_map(|f| match *f {
                    Fold::Count => vec![],
                    Fold::Sum(field) => vec![field],
                    Fold::GroupSum { key, val, .. } => vec![key, val],
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
    /// The spec was written for another canonical basis.
    SchemaMismatch { spec: SchemaSeal, glove: SchemaSeal },
    /// The spec assumes another reading of the glove's bytes.
    ReadingMismatch { spec: ReadingId, glove: ReadingId },
    /// The body's folds do not have the shape its law requires.
    LawShape,
    /// The law adds evidence, and the body does not select independent
    /// evidence (`Law::NarsRevision::independent`).
    EvidenceNotIndependent(&'static str),
    /// Quack could not lower the bound body.
    Lower(LowerError),
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
    /// The planes belong to another population than the method was
    /// prepared against.
    UniverseMismatch {
        method: Universe,
        planes: Universe,
    },
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
    GroupSum { key: Col, val: Col, groups: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Bound {
    Query {
        live: Mask,
        /// `(lane, code, negate)`.
        conds: Vec<(Col, u32, bool)>,
        folds: Vec<BoundFold>,
        law: Law,
        /// The Quack programs, lowered once at preparation, one per fold.
        programs: Vec<Program>,
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
    /// The population the method was bound against.
    pub universe: Universe,
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
        bound_queries(*live, conds, folds)
    }

    /// The lowered Quack programs a call executes, one per fold. Lowered once,
    /// when the method was prepared; a call never lowers.
    #[must_use]
    pub fn programs(&self) -> &[Program] {
        match &self.bound {
            Bound::Query { programs, .. } => programs,
            Bound::Action => &[],
        }
    }
}

fn bound_queries(live: Mask, conds: &[(Col, u32, bool)], folds: &[BoundFold]) -> Vec<Query> {
    let filter = || {
        Filter::and(std::iter::once(Filter::plane(live)).chain(conds.iter().map(
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
        )))
    };
    folds
        .iter()
        .map(|f| Query {
            filter: filter(),
            agg: match *f {
                BoundFold::Count => Agg::Count,
                BoundFold::Sum(col) => Agg::SumI32(col),
                BoundFold::GroupSum { key, val, .. } => Agg::GroupSumI32 { key, val },
            },
        })
        .collect()
}

impl PreparedMethod {
    /// Run the method over `planes` through `frontend`.
    ///
    /// # Errors
    ///
    /// [`CallError::NotPure`] for an action, [`CallError::UniverseMismatch`]
    /// for planes of another population, otherwise the frontend's own
    /// refusal.
    pub fn call(
        &self,
        planes: &Planes<'_>,
        universe: Universe,
        frontend: Frontend,
    ) -> Result<Answer, CallError> {
        let Bound::Query {
            live,
            conds,
            folds,
            law,
            programs,
        } = &self.bound
        else {
            return Err(CallError::NotPure);
        };
        if universe != self.universe {
            return Err(CallError::UniverseMismatch {
                method: self.universe,
                planes: universe,
            });
        }
        let values = match frontend {
            Frontend::Quack => programs
                .iter()
                .zip(folds)
                .map(|(p, f)| run_quack(p, *f, planes))
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
        Ok(match *law {
            Law::Values => Answer::Values(values),
            Law::NarsRevision { scale, horizon, .. } => {
                let [FoldValue::Groups(w), FoldValue::Groups(wf)] = values.as_slice() else {
                    unreachable!("preparation checked the law's shape")
                };
                Answer::Truths(nars_revision(w, wf, scale, horizon))
            }
        })
    }
}

/// [`Law::NarsRevision`]'s finisher: per group, from the evidence sums.
fn nars_revision(w: &[i64], wf: &[i64], scale: u32, horizon: u32) -> Vec<Option<Truth>> {
    w.iter()
        .zip(wf)
        .map(|(&w, &wf)| {
            (w > 0).then(|| {
                let weight = w as f64 / f64::from(scale);
                Truth {
                    f: wf as f64 / w as f64,
                    c: weight / (weight + f64::from(horizon)),
                }
            })
        })
        .collect()
}

fn run_quack(
    program: &Program,
    fold: BoundFold,
    planes: &Planes<'_>,
) -> Result<FoldValue, CallError> {
    let mut scratch = Scratch::for_program(program, planes.n_rows).map_err(CallError::Exec)?;
    if let BoundFold::GroupSum { groups, .. } = fold {
        let mut sink = vec![0i64; groups as usize];
        execute_into(
            program,
            planes,
            &Foreign::NONE,
            &mut scratch,
            Out::I64(&mut sink),
        )
        .map_err(CallError::Exec)?;
        return Ok(FoldValue::Groups(sink));
    }
    match execute_into(program, planes, &Foreign::NONE, &mut scratch, Out::None)
        .map_err(CallError::Exec)?
    {
        Value::Count(n) => Ok(FoldValue::Scalar(
            i64::try_from(n).expect("a row count fits i64"),
        )),
        Value::SumI64(s) => Ok(FoldValue::Scalar(s)),
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
) -> Result<FoldValue, CallError> {
    let mut stack = Vec::new();
    if let BoundFold::GroupSum { key, val, groups } = fold {
        let mut sink = vec![0i64; groups as usize];
        let mut d = FoldDialect::with_groups(planes, &Foreign::NONE, &mut sink);
        procs::select_u32(&mut d, &mut stack, live.0, conds).map_err(CallError::Fold)?;
        procs::group_sum(&mut d, &mut stack, key.0, val.0).map_err(CallError::Fold)?;
        assert!(stack.is_empty(), "GROUP_SUM pushes nothing, got {stack:?}");
        drop(d);
        return Ok(FoldValue::Groups(sink));
    }
    let mut d = FoldDialect::new(planes, &Foreign::NONE);
    procs::select_u32(&mut d, &mut stack, live.0, conds).map_err(CallError::Fold)?;
    match fold {
        BoundFold::Count => procs::count(&mut d, &mut stack),
        BoundFold::Sum(col) => procs::sum(&mut d, &mut stack, col.0),
        BoundFold::GroupSum { .. } => unreachable!("handled above"),
    }
    .map_err(CallError::Fold)?;
    match stack.as_slice() {
        [Val::Scalar(n)] => Ok(FoldValue::Scalar(*n)),
        other => panic!("a fold leaves one scalar, got {other:?}"),
    }
}

/// One concept's methods, bound against one glove.
#[derive(Debug)]
pub struct MethodTable {
    concept: u16,
    table: &'static str,
    labels: Vec<(&'static str, MethodKey)>,
    methods: Vec<(MethodKey, Result<PreparedMethod, PrepareError>)>,
}

impl MethodTable {
    /// Bind every method of the glove's concept once. `specs` are
    /// `(label, spec)`; the label is display text only.
    ///
    /// A refused method stays in the table with its reason.
    #[must_use]
    pub fn prepare(glove: &Glove<'_>, specs: &[(&'static str, MethodSpec)]) -> Self {
        let available = available(glove.basis, glove.claims);
        let seal = SchemaSeal::of(glove.basis);
        Self {
            concept: glove.concept,
            table: glove.table,
            labels: specs.iter().map(|(l, s)| (*l, s.key)).collect(),
            methods: specs
                .iter()
                .map(|(_, s)| (s.key, glove.prepare(seal, available.as_ref(), s)))
                .collect(),
        }
    }

    /// The concept whose methods this table holds.
    #[must_use]
    pub fn concept(&self) -> u16 {
        self.concept
    }

    /// The glove table its methods were bound against.
    #[must_use]
    pub fn table(&self) -> &'static str {
        self.table
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
        universe: Universe,
        frontend: Frontend,
    ) -> Result<Answer, CallError> {
        match self.get(key).ok_or(CallError::UnknownMethod)? {
            Ok(m) => m.call(planes, universe, frontend),
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

/// One glove of one concept: everything a spec is bound against.
pub struct Glove<'a> {
    /// Canonical concept id (`ogar_vocab::class_ids`).
    pub concept: u16,
    /// The concept's canonical fields in `ClassView` order.
    pub basis: &'a [String],
    /// This glove's native → canonical claims.
    pub claims: &'a [FieldMap],
    /// This glove's resolution membrane.
    pub binder: &'a dyn Binder,
    /// The glove's table for the concept.
    pub table: &'static str,
    /// The reading of the glove's bytes it serves.
    pub reading: ReadingId,
    /// The population its planes hold.
    pub universe: Universe,
}

impl Glove<'_> {
    fn prepare(
        &self,
        seal: SchemaSeal,
        available: Result<&WideFieldMask, &PrepareError>,
        spec: &MethodSpec,
    ) -> Result<PreparedMethod, PrepareError> {
        if spec.concept != self.concept {
            return Err(PrepareError::WrongConcept {
                spec: spec.concept,
                table: self.concept,
            });
        }
        // The basis IS the field mask's universe: one wider than a position
        // can address is refused whole, even if no claimed or read field
        // sits past the edge.
        if self.basis.len() > usize::from(u8::MAX) + 1 {
            return Err(PrepareError::BasisTooWide(self.basis.len()));
        }
        if spec.schema != seal {
            return Err(PrepareError::SchemaMismatch {
                spec: spec.schema,
                glove: seal,
            });
        }
        if spec.reading != self.reading {
            return Err(PrepareError::ReadingMismatch {
                spec: spec.reading,
                glove: self.reading,
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
                check_law(law, filter, folds)?;
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
                    .map(|f| match *f {
                        Fold::Count => Ok(BoundFold::Count),
                        Fold::Sum(field) => {
                            Ok(BoundFold::Sum(self.lane(t, field, FieldKind::I32)?))
                        }
                        Fold::GroupSum { key, val, groups } => Ok(BoundFold::GroupSum {
                            key: self.lane(t, key, FieldKind::Code)?,
                            val: self.lane(t, val, FieldKind::I32)?,
                            groups,
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let live = self.binder.live(t);
                let programs = bound_queries(live, &conds, &folds)
                    .iter()
                    .map(lower)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(PrepareError::Lower)?;
                Bound::Query {
                    live,
                    conds,
                    folds,
                    law,
                    programs,
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
            universe: self.universe,
            bound,
        })
    }

    /// A required field's lane, which must be of `kind`.
    fn lane(&self, t: TableId, field: &'static str, kind: FieldKind) -> Result<Col, PrepareError> {
        let b = self.field(t, field)?;
        if b.kind != kind {
            return Err(PrepareError::KindMismatch(field));
        }
        Ok(b.col)
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

/// The body shape a law needs, checked before anything binds.
fn check_law(law: Law, filter: &[Cond], folds: &[Fold]) -> Result<(), PrepareError> {
    match law {
        Law::Values => Ok(()),
        Law::NarsRevision { independent, .. } => {
            let [Fold::GroupSum {
                key: k1,
                val: w,
                groups: g1,
            }, Fold::GroupSum {
                key: k2,
                val: wf,
                groups: g2,
            }] = folds
            else {
                return Err(PrepareError::LawShape);
            };
            if k1 != k2 || g1 != g2 || w == wf {
                return Err(PrepareError::LawShape);
            }
            let selects_independent = filter.iter().any(|c| {
                *c == Cond::Eq {
                    field: independent,
                    literal: "true",
                }
            });
            if !selects_independent {
                return Err(PrepareError::EvidenceNotIndependent(independent));
            }
            Ok(())
        }
    }
}
