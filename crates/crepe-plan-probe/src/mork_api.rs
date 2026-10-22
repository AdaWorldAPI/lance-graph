//! The logical interface: MORK's `query_multi (pattern, effect)` shape
//! (`kernel/src/space.rs`; D-MPS-9, §7c of the Schnellverschluss plan). An
//! optimized rule becomes one conjunctive pattern and an output template. No
//! MORK code is copied or called here; only the shape of the verb is matched.
//! MORK itself stays the oracle behind `mork_pathmap_probe::boundary`.
//!
//! The admitted subset is D-MPS-9's bounded first scope: flat patterns of
//! fixed arity whose variables are shared between atoms. Nested patterns, a
//! variable repeated inside one atom and cyclic bodies are refused by name.

use lance_graph_relalg::join_body::is_alpha_acyclic;
use mork_pathmap_probe::boundary::{OracleAtom, OracleRule, OracleTerm};

use crate::logic::{RelId, Rule, Term, VarId};

/// One S-expression element.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// The relation symbol at the head of an atom.
    Sym(RelId),
    /// `$v`.
    Var(VarId),
    /// An integer constant.
    Int(u64),
    /// A sub-expression. Never produced by [`query_multi`]; it exists so a
    /// caller's nested pattern can be refused instead of flattened.
    Nested(Vec<Expr>),
}

/// `(rel a b …)`.
pub type Pattern = Vec<Expr>;

/// `query_multi`: every binding of the conjunction `patterns`, written through
/// `template`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryMulti {
    pub patterns: Vec<Pattern>,
    pub template: Pattern,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorkRefusal {
    /// A pattern contains a sub-expression.
    NestedPattern { pattern: usize },
    /// One atom names the same variable twice (`(R $x $x)`).
    RepeatedVarInAtom { pattern: usize, var: VarId },
    /// The body hypergraph is cyclic (needs `EqU32ViaLane` / `RunsMeet`,
    /// ratified but not authorized).
    Cyclic,
    /// A pattern does not start with a relation symbol.
    NoRelation { pattern: usize },
    /// An integer constant does not fit the oracle's `u32` symbol space.
    ConstOutOfRange { value: u64 },
}

/// The pattern of one rule.
pub fn query_multi(rule: &Rule) -> QueryMulti {
    let term = |t: &Term| match *t {
        Term::Var(v) => Expr::Var(v),
        Term::Const(c) => Expr::Int(c),
    };
    let atom = |rel: RelId, terms: &[Term]| -> Pattern {
        std::iter::once(Expr::Sym(rel))
            .chain(terms.iter().map(term))
            .collect()
    };
    QueryMulti {
        patterns: rule.body.iter().map(|a| atom(a.rel, &a.terms)).collect(),
        template: atom(rule.head.rel, &rule.head.terms),
    }
}

/// Refuses a pattern outside D-MPS-9's first scope.
pub fn admit(q: &QueryMulti) -> Result<(), MorkRefusal> {
    let mut edges = Vec::new();
    for (i, p) in q.patterns.iter().enumerate() {
        if !matches!(p.first(), Some(Expr::Sym(_))) {
            return Err(MorkRefusal::NoRelation { pattern: i });
        }
        let mut vars: Vec<u16> = Vec::new();
        for e in &p[1..] {
            match e {
                Expr::Nested(_) | Expr::Sym(_) => {
                    return Err(MorkRefusal::NestedPattern { pattern: i });
                }
                Expr::Var(v) => {
                    if vars.contains(&v.0) {
                        return Err(MorkRefusal::RepeatedVarInAtom {
                            pattern: i,
                            var: *v,
                        });
                    }
                    vars.push(v.0);
                }
                Expr::Int(_) => {}
            }
        }
        edges.push(vars);
    }
    if !is_alpha_acyclic(&edges) {
        return Err(MorkRefusal::Cyclic);
    }
    Ok(())
}

/// The rule as the MORK oracle's input (`mork_pathmap_probe::boundary`), after
/// the same admission as [`admit`]: a refused pattern returns its refusal
/// unchanged. Relation and variable ids carry over as numbers; a constant must
/// fit a `u32`, the oracle's symbol space. Nothing here names MORK.
pub fn to_oracle(rule: &Rule) -> Result<OracleRule, MorkRefusal> {
    admit(&query_multi(rule))?;
    let term = |t: &Term| -> Result<OracleTerm, MorkRefusal> {
        Ok(match *t {
            Term::Var(VarId(v)) => OracleTerm::Var(v),
            Term::Const(c) => OracleTerm::Const(
                u32::try_from(c).map_err(|_| MorkRefusal::ConstOutOfRange { value: c })?,
            ),
        })
    };
    let terms =
        |ts: &[Term]| -> Result<Vec<OracleTerm>, MorkRefusal> { ts.iter().map(term).collect() };
    let body = rule
        .body
        .iter()
        .map(|a| {
            Ok(OracleAtom {
                rel: a.rel.0,
                terms: terms(&a.terms)?,
            })
        })
        .collect::<Result<Vec<_>, MorkRefusal>>()?;
    Ok(OracleRule {
        head: terms(&rule.head.terms)?,
        body,
    })
}

/// The pattern as MORK text, `(Parent 17 $1)`, for reports and test output
/// only. Nothing reads it back.
pub fn render(q: &QueryMulti, name: impl Fn(RelId) -> &'static str) -> String {
    fn expr(e: &Expr, name: &dyn Fn(RelId) -> &'static str) -> String {
        match e {
            Expr::Sym(r) => name(*r).to_string(),
            Expr::Var(v) => format!("${}", v.0),
            Expr::Int(c) => c.to_string(),
            Expr::Nested(es) => list(es, name),
        }
    }
    fn list(es: &[Expr], name: &dyn Fn(RelId) -> &'static str) -> String {
        let parts: Vec<String> = es.iter().map(|e| expr(e, name)).collect();
        format!("({})", parts.join(" "))
    }
    let pats: Vec<String> = q.patterns.iter().map(|p| list(p, &name)).collect();
    format!("(, {}) -> {}", pats.join(" "), list(&q.template, &name))
}
