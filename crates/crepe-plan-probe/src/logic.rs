//! The resolved rules: relation and variable ids, no strings. Built once from a
//! [`Plan`] through the label table, then rewritten by [`crate::optimize`].
//!
//! The admitted subset is flat, positive, non-recursive conjunctions of atoms
//! over `@input` relations, with integer constants. Everything else is refused
//! by name at resolution.

use crate::plan::{Plan, PlanClause, PlanField, PlanRelKind};

/// A relation id: its position in the label table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelId(pub u16);

/// A variable id, numbered by first occurrence within one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    Var(VarId),
    Const(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atom {
    pub rel: RelId,
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Atom>,
}

/// The label table: relation names to ids, built from the plan's relations.
/// The probe's stand-in for the label codebook; nothing after resolution keeps
/// a name.
#[derive(Debug, Clone)]
pub struct Labels {
    names: Vec<&'static str>,
    arity: Vec<usize>,
    input: Vec<bool>,
}

impl Labels {
    pub fn of(plan: &Plan) -> Labels {
        Labels {
            names: plan.relations.iter().map(|r| r.name).collect(),
            arity: plan.relations.iter().map(|r| r.arity).collect(),
            input: plan
                .relations
                .iter()
                .map(|r| r.kind == PlanRelKind::Input)
                .collect(),
        }
    }

    pub fn id(&self, name: &str) -> Option<RelId> {
        self.names
            .iter()
            .position(|n| *n == name)
            .map(|i| RelId(i as u16))
    }

    /// The label, for diagnostics at the test boundary only.
    pub fn name(&self, rel: RelId) -> &'static str {
        self.names[rel.0 as usize]
    }

    pub fn arity(&self, rel: RelId) -> usize {
        self.arity[rel.0 as usize]
    }

    pub fn is_input(&self, rel: RelId) -> bool {
        self.input[rel.0 as usize]
    }
}

/// Why a rule is outside the admitted subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A condition, `let` or `for` clause.
    NonAtomClause(&'static str),
    /// A negated atom: needs an explicit universe; not in this spike.
    Negation,
    /// The rule reads its own stratum (recursion): needs the D-MTP-3 driver.
    Recursive,
    /// A body atom over a derived relation: needs rule unfolding, not in this
    /// spike.
    DerivedBodyRelation(&'static str),
    /// A `ref` binding or a non-literal expression in a field.
    UnsupportedField(&'static str),
    /// A head variable the body does not bind.
    UnboundHeadVar(&'static str),
    /// A relation name the label table does not know.
    UnknownRelation(&'static str),
}

/// Resolves every rule of the plan: each one is a rule in the subset or a
/// named refusal.
pub fn resolve(plan: &Plan, labels: &Labels) -> Vec<Result<Rule, Refusal>> {
    plan.rules
        .iter()
        .map(|rule| {
            if rule.recursive {
                return Err(Refusal::Recursive);
            }
            let mut vars: Vec<&'static str> = Vec::new();
            let var = |name: &'static str, vars: &mut Vec<&'static str>| {
                let i = vars.iter().position(|v| *v == name).unwrap_or_else(|| {
                    vars.push(name);
                    vars.len() - 1
                });
                Term::Var(VarId(i as u16))
            };
            let mut body = Vec::new();
            for clause in &rule.body {
                let atom = match clause {
                    PlanClause::Unsupported(what) => return Err(Refusal::NonAtomClause(what)),
                    PlanClause::Atom(a) => a,
                };
                if atom.negated {
                    return Err(Refusal::Negation);
                }
                let rel = labels
                    .id(atom.relation)
                    .ok_or(Refusal::UnknownRelation(atom.relation))?;
                if !labels.is_input(rel) {
                    return Err(Refusal::DerivedBodyRelation(atom.relation));
                }
                let mut terms = Vec::new();
                for f in &atom.fields {
                    terms.push(match *f {
                        PlanField::NewVar(n) | PlanField::BoundVar(n) => var(n, &mut vars),
                        // `_` binds nothing: a variable no other field names.
                        PlanField::Ignored => {
                            vars.push("_");
                            Term::Var(VarId((vars.len() - 1) as u16))
                        }
                        PlanField::Const(c) => Term::Const(c),
                        PlanField::Ref(n) | PlanField::Expr(n) => {
                            return Err(Refusal::UnsupportedField(n));
                        }
                    });
                }
                body.push(Atom { rel, terms });
            }
            let head_rel = labels
                .id(rule.goal.relation)
                .ok_or(Refusal::UnknownRelation(rule.goal.relation))?;
            let mut head_terms = Vec::new();
            for f in &rule.goal.fields {
                head_terms.push(match *f {
                    PlanField::BoundVar(n) => {
                        let i = vars
                            .iter()
                            .position(|v| *v == n)
                            .ok_or(Refusal::UnboundHeadVar(n))?;
                        Term::Var(VarId(i as u16))
                    }
                    PlanField::Const(c) => Term::Const(c),
                    PlanField::Ignored => return Err(Refusal::UnsupportedField("_")),
                    PlanField::NewVar(n) | PlanField::Ref(n) | PlanField::Expr(n) => {
                        return Err(Refusal::UnsupportedField(n));
                    }
                });
            }
            Ok(Rule {
                head: Atom {
                    rel: head_rel,
                    terms: head_terms,
                },
                body,
            })
        })
        .collect()
}

impl Rule {
    /// The same rule with variables renumbered by first occurrence (body
    /// first, then head). Two rules that differ only in variable names are
    /// equal after this.
    pub fn canonical(&self) -> Rule {
        let mut map: Vec<(VarId, VarId)> = Vec::new();
        let mut rn = |t: &Term| match *t {
            Term::Const(c) => Term::Const(c),
            Term::Var(v) => {
                let n = match map.iter().find(|(a, _)| *a == v) {
                    Some(&(_, b)) => b,
                    None => {
                        let b = VarId(map.len() as u16);
                        map.push((v, b));
                        b
                    }
                };
                Term::Var(n)
            }
        };
        let body: Vec<Atom> = self
            .body
            .iter()
            .map(|a| Atom {
                rel: a.rel,
                terms: a.terms.iter().map(&mut rn).collect(),
            })
            .collect();
        let head = Atom {
            rel: self.head.rel,
            terms: self.head.terms.iter().map(&mut rn).collect(),
        };
        Rule { head, body }
    }
}
