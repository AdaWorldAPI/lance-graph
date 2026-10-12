//! Gremlin steps as typed building blocks of a class method (D-CMH-G1).
//!
//! A traversal is a slice of [`Step`]s. [`compose`] turns it into a
//! [`MethodBody`] in the concept's canonical field names: conditions, folds,
//! or a predicate. After that it is an ordinary method: [`MethodTable`]
//! binds it, inlines the predicates it names, checks claims and laws, and
//! Quack lowers it. No step is interpreted at execution time, and nothing
//! here evaluates a row.
//!
//! [`MethodTable`]: crate::method::MethodTable
//!
//! # The anchor contract
//!
//! The traversal starts at `V(concept)`: the concept's table, where each live
//! row is one vertex and one traverser of bulk 1 (the anchor rule of
//! `crates/lance-graph-quack/tests/gremlin_parity.rs`, #1311). Every step this
//! slice accepts keeps that true:
//!
//! | step | on a vertex | on `values(f)` |
//! |---|---|---|
//! | `has(f, eq/neq)`, `where(method())` | a condition | refused (`FilterAfterValues`) |
//! | `dedup()` | identity: anchor rows are distinct vertices | refused (`DedupOfValues`: a set of values is another multiset) |
//! | `values(f)` | cursor moves to the value | refused (`ValuesOfValues`) |
//! | `count()` | `COUNT(*)` | refused (`CountOfValues`: `COUNT(f)` is not `COUNT(*)` under nulls) |
//! | `sum()` | refused (`SumOfVertices`) | `SUM(f)` |
//! | `out` / `in` / `both` | refused (`NoRelationCarrier`): the glove declares no relation carrier | — |
//!
//! A traversal that ends on vertices is a predicate ([`MethodBody::Select`]);
//! one that ends on a fold is a query; one that ends on `values(f)` returns a
//! stream this layer has no answer type for (`NoTerminal`). Effectful steps
//! (`addV`, `property`, `drop`, …) have no [`Step`] at all: a traversal cannot
//! express them, so it cannot carry authority.
use crate::method::{Cond, Fold, Law, MethodBody, MethodRef};

/// A predicate in `has(field, pred)`, over a coded field's literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pred<'a> {
    /// `P.eq(literal)`.
    Eq(&'a str),
    /// `P.neq(literal)`.
    Neq(&'a str),
}

/// One traversal step. Names are canonical: a concept id, the concept's
/// canonical field names, method references, relation labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step<'a> {
    /// `V()` over a concept's population.
    V(u16),
    /// `has(field, pred)`.
    Has { field: &'a str, pred: Pred<'a> },
    /// `where(method())`: a predicate method of the same concept.
    Where(MethodRef<'a>),
    /// `values(field)`.
    Values(&'a str),
    /// `dedup()`.
    Dedup,
    /// `count()`.
    Count,
    /// `sum()`.
    Sum,
    /// `out(relation)`.
    Out(&'a str),
    /// `in(relation)`.
    In(&'a str),
    /// `both(relation)`.
    Both(&'a str),
}

/// Why a traversal does not compose. Each names the missing capability or
/// the contract it would break.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GremlinRefusal {
    /// The traversal does not start with exactly one `V(concept)`.
    NoStart,
    /// `V` names another concept than the method's.
    AnchorMismatch { step: u16, concept: u16 },
    /// A hop over a relation the glove declares no carrier for.
    NoRelationCarrier(String),
    /// `dedup()` after `values(f)`: distinct values, a different multiset.
    DedupOfValues,
    /// `count()` after `values(f)`: `COUNT(f)`, no such fold yet.
    CountOfValues,
    /// `sum()` of vertices.
    SumOfVertices,
    /// A filter after `values(f)`.
    FilterAfterValues,
    /// `values(f)` after `values(g)`.
    ValuesOfValues,
    /// A step after the terminal fold.
    AfterTerminal,
    /// The traversal ends on a stream of values.
    NoTerminal,
}

/// A composed traversal: the body's conditions and folds, owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composed<'a> {
    filter: Vec<Cond<'a>>,
    folds: Vec<Fold<'a>>,
}

impl<'a> Composed<'a> {
    /// The body: a predicate when the traversal ends on vertices, otherwise
    /// a query finished by [`Law::Values`].
    #[must_use]
    pub fn body(&self) -> MethodBody<'_> {
        if self.folds.is_empty() {
            MethodBody::Select {
                filter: &self.filter,
            }
        } else {
            MethodBody::Query {
                filter: &self.filter,
                folds: &self.folds,
                law: Law::Values,
            }
        }
    }

    /// The conditions, in step order.
    #[must_use]
    pub fn filter(&self) -> &[Cond<'a>] {
        &self.filter
    }

    /// The folds; empty for a predicate.
    #[must_use]
    pub fn folds(&self) -> &[Fold<'a>] {
        &self.folds
    }
}

enum Cursor<'a> {
    Vertex,
    Value(&'a str),
    Done,
}

/// Compose `steps` into the body of a method of `concept`.
///
/// # Errors
///
/// [`GremlinRefusal`] naming the first step that breaks the anchor contract.
pub fn compose<'a>(concept: u16, steps: &[Step<'a>]) -> Result<Composed<'a>, GremlinRefusal> {
    let (first, rest) = steps.split_first().ok_or(GremlinRefusal::NoStart)?;
    match *first {
        Step::V(c) if c == concept => {}
        Step::V(c) => return Err(GremlinRefusal::AnchorMismatch { step: c, concept }),
        _ => return Err(GremlinRefusal::NoStart),
    }
    let mut filter = Vec::new();
    let mut folds = Vec::new();
    let mut cursor = Cursor::Vertex;
    for &step in rest {
        if matches!(cursor, Cursor::Done) {
            return Err(GremlinRefusal::AfterTerminal);
        }
        cursor = match (step, cursor) {
            (Step::V(_), _) => return Err(GremlinRefusal::NoStart),
            (Step::Out(r) | Step::In(r) | Step::Both(r), _) => {
                return Err(GremlinRefusal::NoRelationCarrier(r.to_owned()))
            }
            (Step::Has { field, pred }, Cursor::Vertex) => {
                filter.push(match pred {
                    Pred::Eq(literal) => Cond::Eq { field, literal },
                    Pred::Neq(literal) => Cond::Ne { field, literal },
                });
                Cursor::Vertex
            }
            (Step::Where(r), Cursor::Vertex) => {
                filter.push(Cond::Method(r));
                Cursor::Vertex
            }
            (Step::Has { .. } | Step::Where(_), _) => {
                return Err(GremlinRefusal::FilterAfterValues)
            }
            (Step::Dedup, Cursor::Vertex) => Cursor::Vertex,
            (Step::Dedup, _) => return Err(GremlinRefusal::DedupOfValues),
            (Step::Values(f), Cursor::Vertex) => Cursor::Value(f),
            (Step::Values(_), _) => return Err(GremlinRefusal::ValuesOfValues),
            (Step::Count, Cursor::Vertex) => {
                folds.push(Fold::Count);
                Cursor::Done
            }
            (Step::Count, _) => return Err(GremlinRefusal::CountOfValues),
            (Step::Sum, Cursor::Value(f)) => {
                folds.push(Fold::Sum(f));
                Cursor::Done
            }
            (Step::Sum, _) => return Err(GremlinRefusal::SumOfVertices),
        };
    }
    if matches!(cursor, Cursor::Value(_)) {
        return Err(GremlinRefusal::NoTerminal);
    }
    Ok(Composed { filter, folds })
}
