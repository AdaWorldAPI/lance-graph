//! The shape `crepe_plan!` builds. TEMPORARY: a cold compiler representation
//! private to this probe, not a global rule IR. Who owns the rule IR is D-RA-2,
//! an open maintainer decision; this type takes no position on it.
//!
//! Names are source labels here. [`crate::resolve`] maps them to ids through the
//! label table once; nothing after resolution holds a string.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub relations: Vec<PlanRelation>,
    pub strata: Vec<Vec<&'static str>>,
    pub rules: Vec<PlanRule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRelation {
    pub name: &'static str,
    pub arity: usize,
    pub kind: PlanRelKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanRelKind {
    Input,
    Output,
    Intermediate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRule {
    pub goal: PlanAtom,
    pub body: Vec<PlanClause>,
    pub stratum: usize,
    pub recursive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanAtom {
    pub relation: &'static str,
    pub negated: bool,
    pub fields: Vec<PlanField>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanClause {
    Atom(PlanAtom),
    Unsupported(&'static str),
}

/// One field, as crepe's binding analysis classified it from left to right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanField {
    Ignored,
    Ref(&'static str),
    NewVar(&'static str),
    BoundVar(&'static str),
    Const(u64),
    Expr(&'static str),
}
