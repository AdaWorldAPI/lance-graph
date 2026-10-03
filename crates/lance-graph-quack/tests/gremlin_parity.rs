//! Frontend-parity witness: a Gremlin-shaped traversal lowered onto Quack's
//! [`Query`], the same target the SQL lowering uses.
//!
//! # What this file is for
//!
//! It tests one hypothesis. SQL (DuckDB) and Gremlin were designed
//! independently. If both lower onto the same small population algebra, then
//! a Gremlin traversal and its SQL equivalent should produce the same
//! [`Query`] value. That is a check BELOW the frontend, not only a check on the
//! final number.
//!
//! The adapter here is test code. It adds no operator to Quack and no op to
//! mask-risc. Every traversal it accepts lowers to a `Query { filter, agg }`
//! that Quack already had. Every traversal it cannot lower is refused, and the
//! refusal names the missing capability.
//!
//! # The invariant that makes it work
//!
//! Gremlin counts traversers with bulk (bag semantics). The adapter keeps one
//! table as the ANCHOR: the table whose rows each stand for exactly one
//! traverser of bulk 1. Two moves keep that true:
//!
//! - a hop along a FUNCTIONAL relation (each anchor row names at most one
//!   target) does not change the anchor. The target's fields are read through
//!   the foreign key ([`Filter::EqU32Via`], [`GroupAddr::Via`]). This is the
//!   functional-indexed consumption class: resident reads composed, nothing
//!   materialised;
//! - a hop that fans out (the reverse of a foreign key, or an edge table)
//!   MOVES the anchor to the table whose rows are the paths: the child table,
//!   or the edge table. The old anchor's predicates are carried across through
//!   the foreign key. One hop of fan-out is therefore still one population.
//!
//! A second fan-out, or a functional read two foreign keys deep, cannot be
//! expressed this way. Those are refused, and they are the two real gaps (see
//! [`Refusal::NonFunctionalChain`] and [`Refusal::ComposedFunctionalHop`]).
//!
//! # The graph view of the DuckDB fixture
//!
//! `duckdb/fixture.rs` is an ERP fixture. Read as a property graph:
//!
//! - `line -billedTo-> partner` is functional (`line.partner_id`);
//! - `line -partOf-> doc` is functional (`line.doc_id`);
//! - `doc -tradesWith-> partner` is many-to-many, with `line` as its edge
//!   table (`src = doc_id`, `dst = partner_id`);
//! - `doc -ownedBy-> company` is functional (`doc.company`). The `company`
//!   table (3 rows, one `region` field) is built here. It exists only so that a
//!   two-deep functional chain can be written down.
//!
//! # The oracles
//!
//! Where DuckDB already answered the SQL equivalent (`cases.tsv`), the
//! traversal is checked against that answer. Every traversal is also checked
//! against [`oracle`]: a row-at-a-time Gremlin interpreter, with bulk, that
//! never touches Quack or mask-risc.

#[path = "duckdb/fixture.rs"]
#[allow(dead_code)]
mod fixture;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, words_for, ExecError, Foreign, ForeignPlane as FPlane, LaneRef, Out, Planes,
    Scratch, Value,
};
use lance_graph_quack::{
    lower, Agg, Cmp, Col, Filter, ForeignLane, ForeignPlane, GroupAddr, GroupAgg, Mask, Query,
};

use fixture::col::{AMOUNT, DOC_ID_U32, PARTNER_ID, STATUS};
use fixture::{DOC_ROWS, LINE_ROWS, PARTNER_ROWS};

// =====================================================================
// The traversal vocabulary — typed, no strings, no text parser.
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Table {
    Line,
    Doc,
    Partner,
    Company,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Field {
    Status,
    Amount,
    DocType,
    Country,
    Region,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rel {
    BilledTo,
    PartOf,
    OwnedBy,
    TradesWith,
}

/// A predicate in `has(field, p)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum P {
    Eq(i64),
    Gt(i64),
}

/// The steps. Only the ones a test needs; adding one needs a test.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Step {
    V(Table),
    Has(Field, P),
    Out(Rel),
    In(Rel),
    /// `where(<sub-traversal>)`: keep the traverser if the sub-traversal
    /// reaches anything.
    Where(Vec<Step>),
    Dedup,
    Values(Field),
    Count,
    Sum,
    GroupCountBy(Field),
    /// Return the current elements (the end of a traversal with no reducer).
    Emit,
    Limit(u32),
    Paths,
    /// `repeat(out(rel)).times(k)`.
    RepeatTimes(Rel, u32),
    /// `property(..)`, `addE(..)`, `aggregate(..)`, `sideEffect(..)`.
    SideEffect,
}

use Step::*;

// =====================================================================
// The schema: where each field lives and how each relation is stored.
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    U32,
    I32,
}

fn home(f: Field) -> Table {
    match f {
        Field::Status | Field::Amount => Table::Line,
        Field::DocType => Table::Doc,
        Field::Country => Table::Partner,
        Field::Region => Table::Company,
    }
}

fn kind(f: Field) -> Kind {
    match f {
        Field::Amount => Kind::I32,
        _ => Kind::U32,
    }
}

/// How many values a `u32` field takes: the group universe of a groupCount.
fn domain(f: Field) -> u32 {
    match f {
        Field::Status => 3,
        Field::Country => 8,
        Field::DocType => 4,
        Field::Region => 2,
        Field::Amount => 0,
    }
}

fn rows(t: Table) -> usize {
    match t {
        Table::Line => LINE_ROWS,
        Table::Doc => DOC_ROWS,
        Table::Partner => PARTNER_ROWS,
        Table::Company => COMPANY_ROWS,
    }
}

/// The column of `f` when its table is the anchor ([`Planes::lanes`] index).
fn local_col(f: Field) -> Col {
    match f {
        Field::Status => STATUS,
        Field::Amount => AMOUNT,
        Field::DocType | Field::Country | Field::Region => Col(0),
    }
}

/// The foreign lane of `f` when its table is reached through an fk
/// ([`Foreign::lanes`] index; one numbering shared by every anchor).
fn foreign_lane(f: Field) -> Option<ForeignLane> {
    match f {
        Field::Country => Some(ForeignLane(0)),
        Field::DocType => Some(ForeignLane(1)),
        Field::Region => Some(ForeignLane(2)),
        _ => None,
    }
}

/// Each table's validity plane, as a foreign plane ([`Foreign::planes`]).
fn foreign_alpha(t: Table) -> ForeignPlane {
    match t {
        Table::Partner => ForeignPlane(0),
        Table::Doc => ForeignPlane(1),
        Table::Line => ForeignPlane(2),
        Table::Company => ForeignPlane(3),
    }
}

/// Each table's validity plane when it is the anchor: plane 0 of its own
/// `Planes`. Quack's rule: the table IS its validity plane.
const ALPHA: Mask = Mask(0);

/// The fk column on `doc` when `doc` is the anchor.
const DOC_COMPANY: Col = Col(1);

/// How a relation is stored. DECLARED here, never guessed by the lowering.
#[derive(Debug, Clone, Copy)]
enum Carrier {
    /// Each `on` row holds the row index of one `to` row.
    Fk { on: Table, col: Col, to: Table },
    /// Each `edges` row holds a `src` row index and a `dst` row index.
    Edge {
        edges: Table,
        src: Col,
        dst: Col,
        src_t: Table,
        dst_t: Table,
    },
}

fn carrier(r: Rel) -> Carrier {
    match r {
        Rel::BilledTo => Carrier::Fk {
            on: Table::Line,
            col: PARTNER_ID,
            to: Table::Partner,
        },
        Rel::PartOf => Carrier::Fk {
            on: Table::Line,
            col: DOC_ID_U32,
            to: Table::Doc,
        },
        Rel::OwnedBy => Carrier::Fk {
            on: Table::Doc,
            col: DOC_COMPANY,
            to: Table::Company,
        },
        Rel::TradesWith => Carrier::Edge {
            edges: Table::Line,
            src: DOC_ID_U32,
            dst: PARTNER_ID,
            src_t: Table::Doc,
            dst_t: Table::Partner,
        },
    }
}

// =====================================================================
// The adapter.
// =====================================================================

/// Why a traversal was not lowered. Each variant names what is missing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Refusal {
    /// A field read through two foreign keys (`out(partOf).out(ownedBy)
    /// .has(..)`). Functional-indexed: it needs a composed read
    /// `v[fk2[fk1[i]]]`. The IR reads one fk deep. Gap: the tile-local gather
    /// chain (#1308 option (a)).
    ComposedFunctionalHop,
    /// A fan-out followed by another hop (M:N then M:N, or a functional hop
    /// then a fan-out). Non-functional-indexed: it needs a barrier and a
    /// workspace inside one computation. Gap: no typed phase handoff exists
    /// (#1308 options (b)/(d), #1310 "b-lite").
    NonFunctionalChain,
    /// `has(field, gt(..))` on a table reached through an fk. The IR has
    /// `EqU32Via` only. Gap: ordered comparison through an fk.
    ViaPredicateNotEquality(Field),
    /// `has` on a `u32` field with an ordered comparison. The IR orders
    /// `i32` lanes only (`fixture.rs` records the same gap for `doc_id`).
    OrderedU32(Field),
    /// `values(f).sum()` where `f` is reached through an fk. Gap: a sum of a
    /// foreign value lane (`Σ v[fk[i]]`).
    ValueThroughHop(Field),
    /// A field that is not on the element the traverser is at.
    FieldNotOnCursor(Field),
    /// A relation whose stored direction does not start at the cursor, and no
    /// reverse lane is declared.
    NoCarrierFromCursor(Rel),
    /// Returning elements that may repeat. A mask holds a set; a bag of
    /// vertices has no mask form. `dedup()` first.
    BagOfElementsNotAMask,
    /// A fold over a de-duplicated population other than `count()` or
    /// `emit`. Needs a per-group distinct fold, which does not exist.
    FoldOverSet,
    /// A step after `dedup()` that moves or filters the set population.
    ConsumesSetPopulation,
    /// `limit`, `range`, `order`: a mask has no order or position.
    Positional,
    /// `path()`: an answer per path, not per element.
    PathMultiplicity,
    /// `repeat()`: loop state; refused in all forms (see the parity notes).
    Repeat,
    /// A step with an effect outside the population.
    SideEffect,
    /// The traversal does not start with `V()` or ends without a terminal.
    Malformed,
}

/// Where the traverser is, relative to the anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cursor {
    /// On the anchor row itself.
    Anchor,
    /// One functional read away: the anchor's `fk` column names a `table` row.
    Via { fk: Col, table: Table },
}

/// One predicate, kept symbolic until the anchor is final.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Atom {
    /// The table's validity, read on the anchor or through `fk`.
    Alpha { table: Table, via: Option<Col> },
    /// `field p`, read on the anchor or through `fk`.
    Has {
        field: Field,
        p: P,
        via: Option<Col>,
    },
}

#[derive(Debug, Clone)]
struct State {
    anchor: Table,
    cursor: Cursor,
    atoms: Vec<Atom>,
    /// Set by `dedup()` on a `Via` cursor: the population is the set of
    /// distinct cursor elements, no longer one per anchor row.
    set: bool,
    value: Option<Field>,
}

/// The result of lowering: the anchor table and the Quack query over it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Lowered {
    anchor: Table,
    query: Query,
}

fn lower_traversal(steps: &[Step]) -> Result<Lowered, Refusal> {
    let (first, rest) = steps.split_first().ok_or(Refusal::Malformed)?;
    let V(t) = *first else {
        return Err(Refusal::Malformed);
    };
    let mut st = State {
        anchor: t,
        cursor: Cursor::Anchor,
        atoms: vec![Atom::Alpha {
            table: t,
            via: None,
        }],
        set: false,
        value: None,
    };
    let (last, middle) = rest.split_last().ok_or(Refusal::Malformed)?;
    for s in middle {
        apply(&mut st, s)?;
    }
    let agg = terminal(&st, last)?;
    let filter = build_filter(&st)?;
    Ok(Lowered {
        anchor: st.anchor,
        query: Query { filter, agg },
    })
}

fn apply(st: &mut State, s: &Step) -> Result<(), Refusal> {
    if st.set && !matches!(s, Has(..)) {
        return Err(Refusal::ConsumesSetPopulation);
    }
    match *s {
        Has(f, p) => has(st, f, p),
        Out(r) => hop(st, r, true),
        In(r) => hop(st, r, false),
        Where(ref sub) => where_(st, sub),
        Dedup => {
            match st.cursor {
                // Anchor rows are distinct elements already: one traverser each.
                Cursor::Anchor => {}
                Cursor::Via { .. } => st.set = true,
            }
            Ok(())
        }
        Values(f) => {
            if home(f) != st.anchor || st.cursor != Cursor::Anchor {
                return Err(match st.cursor {
                    Cursor::Via { table, .. } if home(f) == table => Refusal::ValueThroughHop(f),
                    _ => Refusal::FieldNotOnCursor(f),
                });
            }
            st.value = Some(f);
            Ok(())
        }
        Limit(_) => Err(Refusal::Positional),
        Paths => Err(Refusal::PathMultiplicity),
        RepeatTimes(..) => Err(Refusal::Repeat),
        SideEffect => Err(Refusal::SideEffect),
        V(_) | Count | Sum | GroupCountBy(_) | Emit => Err(Refusal::Malformed),
    }
}

fn has(st: &mut State, f: Field, p: P) -> Result<(), Refusal> {
    let via = match st.cursor {
        Cursor::Anchor if home(f) == st.anchor => None,
        Cursor::Via { fk, table } if home(f) == table => Some(fk),
        _ => return Err(Refusal::FieldNotOnCursor(f)),
    };
    st.atoms.push(Atom::Has { field: f, p, via });
    Ok(())
}

/// Carry every anchor atom across the fk `col` of the new anchor. An atom
/// already read through an fk would become two fks deep.
fn carry(atoms: &[Atom], col: Col) -> Result<Vec<Atom>, Refusal> {
    atoms
        .iter()
        .map(|a| match *a {
            Atom::Alpha { table, via: None } => Ok(Atom::Alpha {
                table,
                via: Some(col),
            }),
            Atom::Has {
                field,
                p,
                via: None,
            } => Ok(Atom::Has {
                field,
                p,
                via: Some(col),
            }),
            _ => Err(Refusal::ComposedFunctionalHop),
        })
        .collect()
}

fn hop(st: &mut State, r: Rel, out: bool) -> Result<(), Refusal> {
    match (carrier(r), out, st.cursor) {
        // Functional, in the stored direction: the anchor stays.
        (Carrier::Fk { on, col, to }, true, Cursor::Anchor) if on == st.anchor => {
            st.cursor = Cursor::Via { fk: col, table: to };
            st.atoms.push(Atom::Alpha {
                table: to,
                via: Some(col),
            });
            Ok(())
        }
        // Functional, a second time: two fks deep.
        (Carrier::Fk { on, .. }, true, Cursor::Via { table, .. }) if on == table => {
            Err(Refusal::ComposedFunctionalHop)
        }
        // Reverse of an fk: fan-out. The child table becomes the anchor.
        (Carrier::Fk { on, col, to }, false, Cursor::Anchor) if to == st.anchor => {
            let mut atoms = vec![Atom::Alpha {
                table: on,
                via: None,
            }];
            atoms.extend(carry(&st.atoms, col)?);
            st.anchor = on;
            st.atoms = atoms;
            Ok(())
        }
        // An edge table: fan-out. The edge table becomes the anchor.
        (
            Carrier::Edge {
                edges,
                src,
                dst,
                src_t,
                dst_t,
            },
            _,
            Cursor::Anchor,
        ) => {
            let (from_col, from_t, to_col, to_t) = if out {
                (src, src_t, dst, dst_t)
            } else {
                (dst, dst_t, src, src_t)
            };
            if from_t != st.anchor {
                return Err(Refusal::NoCarrierFromCursor(r));
            }
            let mut atoms = vec![Atom::Alpha {
                table: edges,
                via: None,
            }];
            atoms.extend(carry(&st.atoms, from_col)?);
            atoms.push(Atom::Alpha {
                table: to_t,
                via: Some(to_col),
            });
            st.anchor = edges;
            st.cursor = Cursor::Via {
                fk: to_col,
                table: to_t,
            };
            st.atoms = atoms;
            Ok(())
        }
        // Any fan-out (or a reverse read) from a cursor that already left the
        // anchor needs a second population.
        (_, _, Cursor::Via { .. }) => Err(Refusal::NonFunctionalChain),
        _ => Err(Refusal::NoCarrierFromCursor(r)),
    }
}

/// `where(sub)`. Two shapes lower:
/// - `sub` stays on the anchor (functional hops and `has` only): its atoms
///   join the anchor's, and the cursor does not move;
/// - `sub` starts by fanning out (`in(rel)` over an fk) and then only filters:
///   the traversal becomes "the distinct parents of the surviving children",
///   i.e. the child table as anchor with a set over the fk.
fn where_(st: &mut State, sub: &[Step]) -> Result<(), Refusal> {
    if st.cursor != Cursor::Anchor {
        return Err(Refusal::NonFunctionalChain);
    }
    let mut inner = st.clone();
    for s in sub {
        apply(&mut inner, s)?;
    }
    if inner.anchor == st.anchor {
        st.atoms = inner.atoms;
        return Ok(());
    }
    match sub.first() {
        Some(In(r)) => match carrier(*r) {
            Carrier::Fk { col, to, .. } if inner.cursor == Cursor::Anchor => {
                st.anchor = inner.anchor;
                st.atoms = inner.atoms;
                st.cursor = Cursor::Via { fk: col, table: to };
                st.set = true;
                Ok(())
            }
            _ => Err(Refusal::NonFunctionalChain),
        },
        _ => Err(Refusal::NonFunctionalChain),
    }
}

fn terminal(st: &State, s: &Step) -> Result<Agg, Refusal> {
    match (s, st.cursor, st.set) {
        (Count, _, false) => Ok(Agg::Count),
        (Count, Cursor::Via { fk, .. }, true) => Ok(Agg::CountDistinctOrderedU32 { key: fk }),
        (Emit, Cursor::Anchor, false) => Ok(Agg::Rows),
        (Emit, Cursor::Via { .. }, false) => Err(Refusal::BagOfElementsNotAMask),
        (Emit, Cursor::Via { fk, table }, true) => Ok(Agg::ScatterOrU32 {
            fk,
            out_rows: rows(table) as u32,
        }),
        (Sum, _, false) => match st.value {
            Some(f) if kind(f) == Kind::I32 => Ok(Agg::SumI32(local_col(f))),
            _ => Err(Refusal::Malformed),
        },
        (GroupCountBy(f), cursor, false) => {
            let f = *f;
            let key = match cursor {
                Cursor::Anchor if home(f) == st.anchor => GroupAddr::Local(local_col(f)),
                Cursor::Via { fk, table } if home(f) == table => GroupAddr::Via {
                    fk,
                    key: foreign_lane(f).ok_or(Refusal::FieldNotOnCursor(f))?,
                },
                _ => return Err(Refusal::FieldNotOnCursor(f)),
            };
            Ok(Agg::GroupReduce {
                key,
                agg: GroupAgg::Count,
            })
        }
        (Sum | GroupCountBy(_), _, true) => Err(Refusal::FoldOverSet),
        (Limit(_), ..) => Err(Refusal::Positional),
        (Paths, ..) => Err(Refusal::PathMultiplicity),
        (RepeatTimes(..), ..) => Err(Refusal::Repeat),
        (SideEffect, ..) => Err(Refusal::SideEffect),
        _ => Err(Refusal::Malformed),
    }
}

fn build_filter(st: &State) -> Result<Filter, Refusal> {
    let parts = st
        .atoms
        .iter()
        .map(|a| match *a {
            Atom::Alpha { via: None, .. } => Ok(Filter::Plane(ALPHA)),
            Atom::Alpha {
                table,
                via: Some(fk),
            } => Ok(Filter::semijoin(fk, foreign_alpha(table))),
            Atom::Has {
                field,
                p,
                via: None,
            } => {
                let c = local_col(field);
                match (kind(field), p) {
                    (Kind::U32, P::Eq(v)) => Ok(Filter::cmp(c, Cmp::EqU32(v as u32))),
                    (Kind::I32, P::Eq(v)) => Ok(Filter::cmp(c, Cmp::EqI32(v as i32))),
                    (Kind::I32, P::Gt(v)) => Ok(Filter::cmp(c, Cmp::GtI32(v as i32))),
                    (Kind::U32, P::Gt(_)) => Err(Refusal::OrderedU32(field)),
                }
            }
            Atom::Has {
                field,
                p,
                via: Some(fk),
            } => match (kind(field), p, foreign_lane(field)) {
                (Kind::U32, P::Eq(v), Some(lane)) => Ok(Filter::eq_u32_via(fk, lane, v as u32)),
                _ => Err(Refusal::ViaPredicateNotEquality(field)),
            },
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Filter::and(parts))
}

// =====================================================================
// Execution harness: the fixture as SoA, per anchor.
// =====================================================================

const COMPANY_ROWS: usize = 3;
const COMPANY_REGION: [u32; COMPANY_ROWS] = [0, 1, 1];

fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        if let Some(last) = w.last_mut() {
            *last = (1u64 << (n % 64)) - 1;
        }
    }
    w
}

struct World {
    fx: fixture::Fixture,
    alpha: BTreeMap<Table, Vec<u64>>,
}

impl World {
    fn new() -> Self {
        let fx = fixture::generate();
        let alpha = [Table::Line, Table::Doc, Table::Partner, Table::Company]
            .into_iter()
            .map(|t| (t, ones(rows(t))))
            .collect();
        World { fx, alpha }
    }

    /// Run a lowered query. `out` is the sink the terminal needs.
    fn run(&self, l: &Lowered, out: Out<'_>) -> Result<Value, ExecError> {
        let fx = &self.fx;
        let lanes: Vec<LaneRef<'_>> = match l.anchor {
            // The fixture's column order (`fixture::col`).
            Table::Line => vec![
                LaneRef::I32(&fx.line.amount),
                LaneRef::I32(&fx.line.qty),
                LaneRef::I32(&fx.line.doc_id),
                LaneRef::U32(&fx.line.status),
                LaneRef::U32(&fx.line.cost_center),
                LaneRef::U32(&fx.line.gl_account),
                LaneRef::U32(&fx.line.partner_id),
                LaneRef::U32(&fx.line.doc_id_u32),
                LaneRef::I32(&fx.line.discount),
            ],
            Table::Doc => vec![
                LaneRef::U32(&fx.doc.doc_type),
                LaneRef::U32(&fx.doc.company),
            ],
            Table::Partner => vec![
                LaneRef::U32(&fx.partner.country),
                LaneRef::U32(&fx.partner.pgroup),
            ],
            Table::Company => vec![LaneRef::U32(&COMPANY_REGION)],
        };
        let masks: [&[u64]; 1] = [&self.alpha[&l.anchor]];
        let planes = Planes {
            n_rows: rows(l.anchor),
            masks: &masks,
            lanes: &lanes,
        };
        let fplanes = [Table::Partner, Table::Doc, Table::Line, Table::Company].map(|t| FPlane {
            words: &self.alpha[&t],
            rows: rows(t),
        });
        let flanes = [
            LaneRef::U32(&fx.partner.country),
            LaneRef::U32(&fx.doc.doc_type),
            LaneRef::U32(&COMPANY_REGION),
        ];
        let foreign = Foreign {
            planes: &fplanes,
            lanes: &flanes,
        };
        let program = lower(&l.query).expect("an adapter query always lowers");
        let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
        execute_into(&program, &planes, &foreign, &mut scratch, out)
    }

    fn count(&self, l: &Lowered) -> u64 {
        match self.run(l, Out::None).expect("runs") {
            Value::Count(c) => c as u64,
            v => panic!("expected a count, got {v:?}"),
        }
    }

    fn groups(&self, l: &Lowered, k: usize) -> Vec<i64> {
        let mut out = vec![0i64; k];
        assert_eq!(
            self.run(l, Out::I64(&mut out)).expect("runs"),
            Value::GroupReduced
        );
        out
    }
}

// =====================================================================
// The oracle: Gremlin traverser semantics, row at a time, with bulk.
// It never touches Quack or mask-risc.
// =====================================================================

mod oracle {
    use super::*;

    pub type Elem = (Table, usize);

    #[derive(Debug, Clone, PartialEq)]
    pub enum Ans {
        Count(u64),
        Sum(i64),
        /// Gremlin's groupCount map: only keys some traverser reached.
        Groups(BTreeMap<i64, u64>),
        Elems(Vec<Elem>),
    }

    fn field(w: &World, (t, r): Elem, f: Field) -> i64 {
        let fx = &w.fx;
        assert_eq!(home(f), t, "oracle: {f:?} read on {t:?}");
        match f {
            Field::Status => fx.line.status[r] as i64,
            Field::Amount => fx.line.amount[r] as i64,
            Field::DocType => fx.doc.doc_type[r] as i64,
            Field::Country => fx.partner.country[r] as i64,
            Field::Region => COMPANY_REGION[r] as i64,
        }
    }

    /// `(from table, to table, pairs)` of a relation, in stored direction.
    fn pairs(w: &World, r: Rel) -> (Table, Table, Vec<(usize, usize)>) {
        let fx = &w.fx;
        match r {
            Rel::BilledTo => (
                Table::Line,
                Table::Partner,
                fx.line
                    .partner_id
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p as usize))
                    .collect(),
            ),
            Rel::PartOf => (
                Table::Line,
                Table::Doc,
                fx.line
                    .doc_id_u32
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (i, d as usize))
                    .collect(),
            ),
            Rel::OwnedBy => (
                Table::Doc,
                Table::Company,
                fx.doc
                    .company
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| (i, c as usize))
                    .collect(),
            ),
            Rel::TradesWith => (
                Table::Doc,
                Table::Partner,
                (0..LINE_ROWS)
                    .map(|i| {
                        (
                            fx.line.doc_id_u32[i] as usize,
                            fx.line.partner_id[i] as usize,
                        )
                    })
                    .collect(),
            ),
        }
    }

    fn step(w: &World, trav: Vec<(Elem, u64)>, s: &Step) -> Vec<(Elem, u64)> {
        match *s {
            Has(f, p) => trav
                .into_iter()
                .filter(|&(e, _)| {
                    let v = field(w, e, f);
                    match p {
                        P::Eq(x) => v == x,
                        P::Gt(x) => v > x,
                    }
                })
                .collect(),
            Out(r) | In(r) => {
                let (a, b, prs) = pairs(w, r);
                let (from, to) = if matches!(s, Out(_)) { (a, b) } else { (b, a) };
                let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (x, y) in prs {
                    let (x, y) = if matches!(s, Out(_)) { (x, y) } else { (y, x) };
                    if y < rows(to) {
                        adj.entry(x).or_default().push(y);
                    }
                }
                let mut next = Vec::new();
                for ((t, r0), bulk) in trav {
                    assert_eq!(t, from);
                    for &y in adj.get(&r0).map(Vec::as_slice).unwrap_or(&[]) {
                        next.push(((to, y), bulk));
                    }
                }
                next
            }
            Where(ref sub) => trav
                .into_iter()
                .filter(|&t| {
                    let mut inner = vec![(t.0, 1)];
                    for s in sub {
                        inner = step(w, inner, s);
                    }
                    !inner.is_empty()
                })
                .collect(),
            Dedup => {
                let mut seen: BTreeMap<Elem, u64> = BTreeMap::new();
                for (e, _) in trav {
                    seen.insert(e, 1);
                }
                seen.into_iter().collect()
            }
            RepeatTimes(r, k) => {
                let mut cur = trav;
                for _ in 0..k {
                    cur = step(w, cur, &Out(r));
                }
                cur
            }
            ref other => panic!("oracle: no step semantics for {other:?}"),
        }
    }

    pub fn eval(w: &World, steps: &[Step]) -> Ans {
        let V(t) = steps[0] else {
            panic!("starts with V")
        };
        let mut trav: Vec<(Elem, u64)> = (0..rows(t)).map(|r| ((t, r), 1)).collect();
        let mut value = None;
        let (last, middle) = steps[1..].split_last().expect("has a terminal");
        for s in middle {
            match s {
                Values(f) => value = Some(*f),
                s => trav = step(w, trav, s),
            }
        }
        match last {
            Count => Ans::Count(trav.iter().map(|&(_, b)| b).sum()),
            Sum => {
                let f = value.expect("values() before sum()");
                Ans::Sum(trav.iter().map(|&(e, b)| field(w, e, f) * b as i64).sum())
            }
            GroupCountBy(f) => {
                let mut m = BTreeMap::new();
                for &(e, b) in &trav {
                    *m.entry(field(w, e, *f)).or_insert(0) += b;
                }
                Ans::Groups(m)
            }
            Emit => Ans::Elems(trav.iter().map(|&(e, _)| e).collect()),
            other => panic!("oracle: no terminal {other:?}"),
        }
    }

    pub fn count(w: &World, steps: &[Step]) -> u64 {
        match eval(w, steps) {
            Ans::Count(c) => c,
            a => panic!("expected a count, got {a:?}"),
        }
    }
}

// =====================================================================
// DuckDB's committed answers (never hand-edited; see duckdb/README.txt).
// =====================================================================

fn duckdb(id: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/duckdb/cases.tsv");
    let text = std::fs::read_to_string(path).expect("cases.tsv");
    text.lines()
        .find_map(|l| {
            let mut p = l.splitn(3, '\t');
            (p.next() == Some(id)).then(|| p.nth(1).unwrap_or("").to_string())
        })
        .unwrap_or_else(|| panic!("no DuckDB case {id}"))
}

/// The atoms of a lowered filter, order-insensitive. Two traversals that
/// reach the same population by different routes produce the same multiset.
fn atom_set(q: &Query) -> Vec<String> {
    let Filter::And(parts) = &q.filter else {
        panic!("adapter filters are a flat AND")
    };
    let mut v: Vec<String> = parts.iter().map(|p| format!("{p:?}")).collect();
    v.sort();
    v
}

fn groups_encoded(sink: &[i64]) -> String {
    sink.iter()
        .enumerate()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join(";")
}

// =====================================================================
// Parity: the same query from SQL and from Gremlin is the same Query.
// =====================================================================

/// `SELECT SUM(l.amount) FROM line l JOIN partner p ON p.rid=l.partner_id
/// WHERE l.status=1 AND p.country=3` (DuckDB: `join_sum_country`) against
/// `g.V().hasLabel('line').has('status',1)
///   .where(out('billedTo').has('country',3)).values('amount').sum()`.
///
/// FAILS IF: the functional hop changes the anchor, the far-side predicate is
/// not read through the fk, or execution disagrees with DuckDB or the oracle.
#[test]
fn functional_where_equals_sql_join_sum() {
    let w = World::new();
    let g = [
        V(Table::Line),
        Has(Field::Status, P::Eq(1)),
        Where(vec![Out(Rel::BilledTo), Has(Field::Country, P::Eq(3))]),
        Values(Field::Amount),
        Sum,
    ];
    // The SQL reading: FROM line (its plane), WHERE l.status = 1, the inner
    // join to partner (partner must exist), p.country = 3 read through the fk.
    let sql = Query {
        filter: Filter::and([
            Filter::Plane(ALPHA),
            Filter::cmp(STATUS, Cmp::EqU32(1)),
            Filter::semijoin(PARTNER_ID, foreign_alpha(Table::Partner)),
            Filter::eq_u32_via(PARTNER_ID, ForeignLane(0), 3),
        ]),
        agg: Agg::SumI32(AMOUNT),
    };
    let l = lower_traversal(&g).expect("lowers");
    assert_eq!(l.anchor, Table::Line);
    assert_eq!(l.query, sql, "Gremlin and SQL must meet at the same Query");
    assert_eq!(lower(&l.query), lower(&sql));

    let got = match w.run(&l, Out::None).expect("runs") {
        Value::SumI64(s) => s,
        v => panic!("{v:?}"),
    };
    assert_eq!(got.to_string(), duckdb("join_sum_country"));
    assert_eq!(oracle::eval(&w, &g), oracle::Ans::Sum(got));
}

/// `g.V().hasLabel('partner').has('country',3).in('billedTo').has('status',1)
///   .values('amount').sum()` walks the relation the OTHER way: it fans out
/// from partners to lines. It must land on the same anchor and the same atom
/// set as the forward `where(out(..))` traversal, and on DuckDB's answer.
///
/// FAILS IF: the reverse hop does not move the anchor to `line`, or the
/// partner predicates are not carried through the fk.
#[test]
fn reverse_fanout_reanchors_onto_the_same_population() {
    let w = World::new();
    let reverse = [
        V(Table::Partner),
        Has(Field::Country, P::Eq(3)),
        In(Rel::BilledTo),
        Has(Field::Status, P::Eq(1)),
        Values(Field::Amount),
        Sum,
    ];
    let forward = [
        V(Table::Line),
        Has(Field::Status, P::Eq(1)),
        Where(vec![Out(Rel::BilledTo), Has(Field::Country, P::Eq(3))]),
        Values(Field::Amount),
        Sum,
    ];
    let r = lower_traversal(&reverse).expect("lowers");
    let f = lower_traversal(&forward).expect("lowers");
    assert_eq!(
        r.anchor,
        Table::Line,
        "fan-out moves the anchor to the child table"
    );
    assert_eq!(atom_set(&r.query), atom_set(&f.query));
    assert_eq!(r.query.agg, f.query.agg);
    let got = match w.run(&r, Out::None).expect("runs") {
        Value::SumI64(s) => s,
        v => panic!("{v:?}"),
    };
    assert_eq!(got.to_string(), duckdb("join_sum_country"));
    assert_eq!(oracle::eval(&w, &reverse), oracle::Ans::Sum(got));
}

/// `SELECT p.country, COUNT(*) FROM line l JOIN partner p ... WHERE
/// l.status=1 GROUP BY p.country` (DuckDB: `join_group_count_country`)
/// against `g.V().hasLabel('line').has('status',1).out('billedTo')
///   .groupCount().by('country')`.
///
/// Gremlin counts traversers with bulk, so after a functional hop the count
/// per partner country is the count of lines: SQL's `COUNT(*)` exactly. The
/// two differ only in presentation — Gremlin's map omits zero keys, SQL's
/// LEFT JOIN keeps them — which is the frontend's to render.
#[test]
fn functional_hop_group_count_equals_sql_group_by_via() {
    let w = World::new();
    let g = [
        V(Table::Line),
        Has(Field::Status, P::Eq(1)),
        Out(Rel::BilledTo),
        GroupCountBy(Field::Country),
    ];
    let sql = Query {
        filter: Filter::and([
            Filter::Plane(ALPHA),
            Filter::cmp(STATUS, Cmp::EqU32(1)),
            Filter::semijoin(PARTNER_ID, foreign_alpha(Table::Partner)),
        ]),
        agg: Agg::GroupReduce {
            key: GroupAddr::Via {
                fk: PARTNER_ID,
                key: ForeignLane(0),
            },
            agg: GroupAgg::Count,
        },
    };
    let l = lower_traversal(&g).expect("lowers");
    assert_eq!(l.query, sql);
    let sink = w.groups(&l, domain(Field::Country) as usize);
    assert_eq!(groups_encoded(&sink), duckdb("join_group_count_country"));
    let oracle::Ans::Groups(m) = oracle::eval(&w, &g) else {
        panic!()
    };
    for (k, &v) in sink.iter().enumerate() {
        assert_eq!(
            m.get(&(k as i64)).copied().unwrap_or(0),
            v as u64,
            "group {k}"
        );
    }
}

/// `SELECT COUNT(*) FROM doc d WHERE EXISTS(SELECT 1 FROM line l WHERE
/// l.doc_id=d.rid AND l.status=1)` (DuckDB: 511) against
/// `g.V().hasLabel('doc').where(in('partOf').has('status',1)).count()`.
///
/// The existential over children lowers to the child table as anchor and a
/// DISTINCT over the fk — the same terminal Quack's own lowering of this SQL
/// uses (`duckdb_differential.rs::join_count_docs_with_posted`). On the
/// generated fixture `doc_id` is not in key order, so execution REFUSES
/// (`LaneNotOrdered`) exactly as it does for the SQL path: same logical
/// answer, same physical refusal, never a seen-set.
#[test]
fn exists_over_children_is_distinct_over_the_child_space() {
    let w = World::new();
    let g = [
        V(Table::Doc),
        Where(vec![In(Rel::PartOf), Has(Field::Status, P::Eq(1))]),
        Count,
    ];
    let l = lower_traversal(&g).expect("lowers");
    assert_eq!(l.anchor, Table::Line);
    assert_eq!(
        l.query,
        Query {
            filter: Filter::and([
                Filter::Plane(ALPHA),
                Filter::semijoin(DOC_ID_U32, foreign_alpha(Table::Doc)),
                Filter::cmp(STATUS, Cmp::EqU32(1)),
            ]),
            agg: Agg::CountDistinctOrderedU32 { key: DOC_ID_U32 },
        }
    );
    assert_eq!(
        w.run(&l, Out::None),
        Err(ExecError::LaneNotOrdered { lane: DOC_ID_U32.0 })
    );
    assert_eq!(
        oracle::count(&w, &g).to_string(),
        duckdb("join_count_docs_with_posted")
    );
}

// =====================================================================
// Many-to-many: one hop is one population (the edge table).
// =====================================================================

const DOC_T1_COUNTRY3: [Step; 5] = [
    V(Table::Doc),
    Has(Field::DocType, P::Eq(1)),
    Out(Rel::TradesWith),
    Has(Field::Country, P::Eq(3)),
    Count,
];

/// `g.V().hasLabel('doc').has('doc_type',1).out('tradesWith')
///   .has('country',3).count()` over an M:N relation stored as an edge table.
/// SQL: `SELECT COUNT(*) FROM doc d JOIN line l ON l.doc_id=d.rid JOIN
/// partner p ON p.rid=l.partner_id WHERE d.doc_type=1 AND p.country=3`.
///
/// One M:N hop lowers to ONE program over the edge population: both
/// endpoints are functional reads from an edge row. Bulk = one per edge row.
#[test]
fn many_to_many_one_hop_is_one_program_over_the_edge_table() {
    let w = World::new();
    let l = lower_traversal(&DOC_T1_COUNTRY3).expect("lowers");
    assert_eq!(l.anchor, Table::Line);
    let sql = Query {
        filter: Filter::and([
            Filter::Plane(ALPHA),
            Filter::semijoin(DOC_ID_U32, foreign_alpha(Table::Doc)),
            Filter::eq_u32_via(DOC_ID_U32, ForeignLane(1), 1),
            Filter::semijoin(PARTNER_ID, foreign_alpha(Table::Partner)),
            Filter::eq_u32_via(PARTNER_ID, ForeignLane(0), 3),
        ]),
        agg: Agg::Count,
    };
    assert_eq!(l.query, sql);
    let got = w.count(&l);
    assert!(got > 0, "anti-vacuity: the fixture must reach something");
    assert_eq!(got, oracle::count(&w, &DOC_T1_COUNTRY3));
}

/// The same hop folded by a target field: `groupCount().by('country')`.
#[test]
fn many_to_many_group_count_by_target_field() {
    let w = World::new();
    let g = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        GroupCountBy(Field::Country),
    ];
    let l = lower_traversal(&g).expect("lowers");
    let sink = w.groups(&l, 8);
    let oracle::Ans::Groups(m) = oracle::eval(&w, &g) else {
        panic!()
    };
    let nonzero = sink.iter().filter(|&&v| v > 0).count();
    assert!(nonzero > 1, "anti-vacuity: more than one group is reached");
    for (k, &v) in sink.iter().enumerate() {
        assert_eq!(
            m.get(&(k as i64)).copied().unwrap_or(0),
            v as u64,
            "group {k}"
        );
    }
}

/// Bag versus set after a fan-out. `.count()` counts paths (edge rows);
/// `.dedup().count()` counts distinct partners. The two differ on this
/// fixture, and the adapter keeps them apart:
/// - the bag count is `Count` over the edge table;
/// - the set count is `CountDistinctOrderedU32` over `partner_id`, which the
///   generated (unordered) layout physically refuses — it never falls back to
///   the bag;
/// - the set itself, as the DEMANDED result (`dedup()` then emit), is a
///   `ScatterOrU32` mask over partners whose popcount is the oracle's set.
#[test]
fn bag_and_set_stay_apart_after_a_fanout() {
    let w = World::new();
    let bag = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        Count,
    ];
    let set = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        Dedup,
        Count,
    ];
    let emit = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        Dedup,
        Emit,
    ];
    let bag_truth = oracle::count(&w, &bag);
    let set_truth = oracle::count(&w, &set);
    assert!(
        bag_truth > set_truth,
        "anti-vacuity: {bag_truth} paths vs {set_truth} partners"
    );

    let lb = lower_traversal(&bag).expect("lowers");
    assert_eq!(w.count(&lb), bag_truth);

    let ls = lower_traversal(&set).expect("lowers");
    assert_eq!(
        ls.query.agg,
        Agg::CountDistinctOrderedU32 { key: PARTNER_ID }
    );
    assert_eq!(
        w.run(&ls, Out::None),
        Err(ExecError::LaneNotOrdered { lane: PARTNER_ID.0 })
    );

    let le = lower_traversal(&emit).expect("lowers");
    let mut mask = vec![0u64; words_for(PARTNER_ROWS)];
    assert_eq!(
        w.run(&le, Out::Mask(&mut mask)).expect("runs"),
        Value::Scattered
    );
    let popcount: u64 = mask.iter().map(|w| u64::from(w.count_ones())).sum();
    assert_eq!(popcount, set_truth);

    // Without dedup, returning partners is a bag: no mask holds it.
    let bag_emit = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        Emit,
    ];
    assert_eq!(
        lower_traversal(&bag_emit),
        Err(Refusal::BagOfElementsNotAMask)
    );
}

// =====================================================================
// The two real gaps, each with its can-fire / stay-silent pair.
// =====================================================================

/// A functional read two fks deep: `g.V().hasLabel('line').out('partOf')
///   .out('ownedBy').has('region',1).count()`. The question is well defined
/// (the oracle answers it); the IR reads one fk deep, so it is refused as a
/// composed functional hop. One fk deep, the same shape lowers and agrees.
#[test]
fn functional_two_hop_is_refused_one_hop_is_not() {
    let w = World::new();
    let two = [
        V(Table::Line),
        Out(Rel::PartOf),
        Out(Rel::OwnedBy),
        Has(Field::Region, P::Eq(1)),
        Count,
    ];
    assert_eq!(lower_traversal(&two), Err(Refusal::ComposedFunctionalHop));
    let truth = oracle::count(&w, &two);
    assert!(
        truth > 0 && truth < LINE_ROWS as u64,
        "anti-vacuity: {truth}"
    );

    let one = [
        V(Table::Line),
        Out(Rel::PartOf),
        Has(Field::DocType, P::Eq(1)),
        Count,
    ];
    let l = lower_traversal(&one).expect("one fk deep lowers");
    assert_eq!(w.count(&l), oracle::count(&w, &one));

    // A fan-out after a functional read (documents sharing a company) is a
    // second population, not a deeper read.
    let doc_side = [
        V(Table::Doc),
        Out(Rel::OwnedBy),
        Has(Field::Region, P::Eq(1)),
        In(Rel::OwnedBy),
        Count,
    ];
    assert_eq!(lower_traversal(&doc_side), Err(Refusal::NonFunctionalChain));
}

/// Two fan-outs: `g.V().hasLabel('doc').has('doc_type',1).out('tradesWith')
///   .in('tradesWith').count()` — documents that share a partner with a
/// type-1 document, counted per path. It needs the first hop's result as a
/// workspace across a barrier. Refused; the one-hop prefix is not.
#[test]
fn many_to_many_two_hop_is_refused_one_hop_is_not() {
    let w = World::new();
    let two = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        In(Rel::TradesWith),
        Count,
    ];
    assert_eq!(lower_traversal(&two), Err(Refusal::NonFunctionalChain));
    let paths = oracle::count(&w, &two);
    let mut distinct = two.to_vec();
    distinct.insert(4, Dedup);
    let docs = oracle::count(&w, &distinct);
    assert!(
        paths > docs && docs > 0,
        "anti-vacuity: {paths} paths, {docs} docs"
    );

    let one = [
        V(Table::Doc),
        Has(Field::DocType, P::Eq(1)),
        Out(Rel::TradesWith),
        Count,
    ];
    assert!(lower_traversal(&one).is_ok());

    // A functional hop followed by a fan-out from the far side is the same
    // class: lines that share a partner.
    let share = [V(Table::Line), Out(Rel::BilledTo), In(Rel::BilledTo), Count];
    assert_eq!(lower_traversal(&share), Err(Refusal::NonFunctionalChain));
}

/// An ordered comparison on a table reached through an fk. Equality lowers
/// (`EqU32Via`); `gt` has no via form.
#[test]
fn far_side_ordered_predicate_is_refused_equality_is_not() {
    let gt = [
        V(Table::Line),
        Out(Rel::BilledTo),
        Has(Field::Country, P::Gt(3)),
        Count,
    ];
    assert_eq!(
        lower_traversal(&gt),
        Err(Refusal::ViaPredicateNotEquality(Field::Country))
    );
    let eq = [
        V(Table::Line),
        Out(Rel::BilledTo),
        Has(Field::Country, P::Eq(3)),
        Count,
    ];
    assert!(lower_traversal(&eq).is_ok());
    // On the anchor, an ordered compare over an i32 lane lowers.
    let local = [V(Table::Line), Has(Field::Amount, P::Gt(1000)), Count];
    let w = World::new();
    let l = lower_traversal(&local).expect("lowers");
    assert_eq!(w.count(&l), oracle::count(&w, &local));
}

/// A sum of a value reached through an fk: `out('billedTo').values('country')
///   .sum()`. Gap: no fold of a foreign value lane.
#[test]
fn value_through_hop_is_refused() {
    let s = [
        V(Table::Line),
        Out(Rel::BilledTo),
        Values(Field::Country),
        Sum,
    ];
    assert_eq!(
        lower_traversal(&s),
        Err(Refusal::ValueThroughHop(Field::Country))
    );
}

/// Things that are not population algebra at all.
#[test]
fn frontend_semantics_are_refused_by_name() {
    let base = || vec![V(Table::Line), Has(Field::Status, P::Eq(1))];
    let with = |s: Step, t: Step| {
        let mut v = base();
        v.push(s);
        v.push(t);
        v
    };
    assert_eq!(
        lower_traversal(&with(Limit(10), Emit)),
        Err(Refusal::Positional)
    );
    assert_eq!(
        lower_traversal(&with(Paths, Count)),
        Err(Refusal::PathMultiplicity)
    );
    assert_eq!(
        lower_traversal(&with(SideEffect, Count)),
        Err(Refusal::SideEffect)
    );
    assert_eq!(
        lower_traversal(&with(RepeatTimes(Rel::BilledTo, 2), Count)),
        Err(Refusal::Repeat)
    );
    // After dedup, the population is a set: no further hop, no per-group fold.
    let after_set = [
        V(Table::Doc),
        Out(Rel::TradesWith),
        Dedup,
        In(Rel::TradesWith),
        Count,
    ];
    assert_eq!(
        lower_traversal(&after_set),
        Err(Refusal::ConsumesSetPopulation)
    );
    let set_fold = [
        V(Table::Doc),
        Out(Rel::TradesWith),
        Dedup,
        GroupCountBy(Field::Country),
    ];
    assert_eq!(lower_traversal(&set_fold), Err(Refusal::FoldOverSet));
    // Silence: the plain prefix lowers.
    assert!(lower_traversal(&[V(Table::Line), Has(Field::Status, P::Eq(1)), Count]).is_ok());
}

/// The adapter tax: building and lowering the traversal against executing
/// it. Printed, not asserted (timing is machine-dependent).
#[test]
fn adapter_tax_is_printed() {
    let w = World::new();
    const N: u32 = 2_000;
    let t0 = Instant::now();
    let mut l = None;
    for _ in 0..N {
        l = Some(lower_traversal(std::hint::black_box(&DOC_T1_COUNTRY3)).expect("lowers"));
    }
    let adapter = t0.elapsed() / N;
    let l = l.expect("lowered");
    let t1 = Instant::now();
    for _ in 0..N {
        std::hint::black_box(lower(&l.query).expect("lowers"));
    }
    let quack = t1.elapsed() / N;
    let t2 = Instant::now();
    let mut c = 0;
    for _ in 0..N / 10 {
        c = w.count(&l);
    }
    let exec = t2.elapsed() / (N / 10);
    eprintln!(
        "METRIC gremlin_parity adapter_ns={} quack_lower_ns={} exec_ns={} rows={} count={c} \
         materialized_rows=0",
        adapter.as_nanos(),
        quack.as_nanos(),
        exec.as_nanos(),
        LINE_ROWS
    );
}
