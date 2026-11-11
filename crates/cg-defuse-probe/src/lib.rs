//! CG-5 probe (Linear EXO-237): classify division and remainder sites from
//! def-use rows.
//!
//! Three arms answer the same question for one site, "is this a scalar
//! threshold or a per-element quotient?":
//!
//! - [`derive`]: the def-use rows closed by egglog in proofs mode. A verdict
//!   other than [`Class::Unknown`] carries the proof egglog checked.
//! - [`fixpoint`]: the same two closures as plain Rust loops. It shares the
//!   rows and nothing else, so it checks the egglog encoding.
//! - [`source_class`]: a scan of the source text alone. It shares neither the
//!   rows nor the rules.
//!
//! [`nesting_class`] is the cheap comparator: loop nesting of the statement.
//!
//! The rows are written by `.claude/probes/2026-11-11-cg5-defuse/extract_defuse.py`
//! from Charon LLBC. They are flow-insensitive: a local with two different
//! defining statements has two definitions.
//!
//! Meaning of the classes, over the rows of one function:
//!
//! - an *element source* is the result of an iterator `next` call inside a
//!   loop, or a value read through an index projection;
//! - a site is [`Class::PerElement`] when an operand depends on an element
//!   source through any chain of definitions;
//! - a local is *scalar* when it is an argument, or it is not an element
//!   source, has exactly one definition, and every local that definition
//!   reads is scalar. A site is
//!   [`Class::ScalarThreshold`] when both operands are scalar or constants;
//! - anything else is [`Class::Unknown`].

use std::collections::{BTreeMap, BTreeSet};

use egglog::{CommandOutput, EGraph};

/// One operand of a site.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operand {
    /// A local of the function.
    Local(u32),
    /// A literal.
    Const,
    /// A global or static place. Not classified.
    Global,
}

/// One definition of a local: `(kind, line, locals read)`.
///
/// Charon can copy a statement into several exit paths of a loop. The copies
/// have the same kind, line and sources, so they are one definition here.
pub type Def = (String, u32, Vec<u32>);

/// One division or remainder statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    /// Function holding the statement.
    pub func: String,
    /// Statement number inside the function body.
    pub stmt: u32,
    /// `Div` or `Rem`.
    pub op: String,
    /// Source file, relative to the crate.
    pub file: String,
    /// One-based source line.
    pub line: u32,
    /// Number of LLBC `Loop` nodes around the statement.
    pub loop_depth: u32,
    /// Dividend.
    pub lhs: Operand,
    /// Divisor.
    pub rhs: Operand,
}

/// The def-use rows of one LLBC file.
#[derive(Debug, Default, Clone)]
pub struct Facts {
    /// Arguments, per function.
    pub args: BTreeSet<(String, u32)>,
    /// Definitions, per `(function, local)`.
    pub defs: BTreeMap<(String, u32), BTreeSet<Def>>,
    /// Element sources: `next` results and index reads.
    pub elems: BTreeSet<(String, u32)>,
    /// Division and remainder statements, in file order.
    pub sites: Vec<Site>,
}

/// Why a rows file was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum FactsRefusal {
    /// The first line is not `#version 1`.
    Version,
    /// A row has an unknown kind or too few fields; the one-based line.
    Row(usize),
    /// A field that must be a number is not; the one-based line.
    Number(usize),
}

fn num(s: &str, line: usize) -> Result<u32, FactsRefusal> {
    s.parse().map_err(|_| FactsRefusal::Number(line))
}

fn operand(s: &str, line: usize) -> Result<Operand, FactsRefusal> {
    Ok(match s {
        "const" => Operand::Const,
        "global" => Operand::Global,
        n => Operand::Local(num(n, line)?),
    })
}

/// Reads a rows file.
pub fn parse(text: &str) -> Result<Facts, FactsRefusal> {
    let mut lines = text.lines().enumerate();
    match lines.next() {
        Some((_, "#version 1")) => {}
        _ => return Err(FactsRefusal::Version),
    }
    let mut facts = Facts::default();
    for (i, row) in lines {
        let n = i + 1;
        if row.starts_with('#') || row.is_empty() {
            continue;
        }
        let f: Vec<&str> = row.split('\t').collect();
        let need = |k: usize| {
            if f.len() < k {
                Err(FactsRefusal::Row(n))
            } else {
                Ok(())
            }
        };
        need(3)?;
        let func = f[1].to_string();
        match f[0] {
            "ARG" => {
                facts.args.insert((func, num(f[2], n)?));
            }
            "DEF" => {
                need(8)?;
                let mut srcs = Vec::new();
                if f[7] != "-" {
                    for s in f[7].split(',') {
                        srcs.push(num(s, n)?);
                    }
                }
                facts
                    .defs
                    .entry((func, num(f[2], n)?))
                    .or_default()
                    .insert((f[4].to_string(), num(f[5], n)?, srcs));
            }
            "ELEM" | "INDEX" => {
                facts.elems.insert((func, num(f[2], n)?));
            }
            "SITE" => {
                need(9)?;
                facts.sites.push(Site {
                    func,
                    stmt: num(f[2], n)?,
                    op: f[3].to_string(),
                    file: f[4].to_string(),
                    line: num(f[5], n)?,
                    loop_depth: num(f[6], n)?,
                    lhs: operand(f[7], n)?,
                    rhs: operand(f[8], n)?,
                });
            }
            _ => return Err(FactsRefusal::Row(n)),
        }
    }
    Ok(facts)
}

/// The answer for one site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Both operands are fixed before any element is read.
    ScalarThreshold,
    /// An operand depends on an element source.
    PerElement,
    /// Neither could be shown.
    Unknown,
}

/// The plain-Rust arm: both closures as loops to a fixpoint.
///
/// Dependence on an element is tested first, so this arm needs no rule that
/// keeps element sources out of the scalar set; the egglog arm does, because
/// there a site in both classes merges them ([`DeriveRefusal::Conflict`]).
pub fn fixpoint(facts: &Facts) -> Vec<Class> {
    let mut elem: BTreeSet<(String, u32)> = facts.elems.clone();
    let mut scalar: BTreeSet<(String, u32)> = facts.args.clone();
    loop {
        let mut changed = false;
        for (key, defs) in &facts.defs {
            let func = &key.0;
            if !elem.contains(key)
                && defs
                    .iter()
                    .any(|d| d.2.iter().any(|s| elem.contains(&(func.clone(), *s))))
            {
                elem.insert(key.clone());
                changed = true;
            }
            if !scalar.contains(key)
                && defs.len() == 1
                && defs
                    .iter()
                    .all(|d| d.2.iter().all(|s| scalar.contains(&(func.clone(), *s))))
            {
                scalar.insert(key.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let is_elem = |f: &str, o: &Operand| match o {
        Operand::Local(l) => elem.contains(&(f.to_string(), *l)),
        _ => false,
    };
    let is_scalar = |f: &str, o: &Operand| match o {
        Operand::Local(l) => scalar.contains(&(f.to_string(), *l)),
        Operand::Const => true,
        Operand::Global => false,
    };
    facts
        .sites
        .iter()
        .map(|s| {
            if is_elem(&s.func, &s.lhs) || is_elem(&s.func, &s.rhs) {
                Class::PerElement
            } else if is_scalar(&s.func, &s.lhs) && is_scalar(&s.func, &s.rhs) {
                Class::ScalarThreshold
            } else {
                Class::Unknown
            }
        })
        .collect()
}

/// The cheap comparator: a statement inside an LLBC loop is per-element.
pub fn nesting_class(site: &Site) -> Class {
    if site.loop_depth > 0 {
        Class::PerElement
    } else {
        Class::ScalarThreshold
    }
}

/// The egglog vocabulary and rules. `T` is the one truth value; a property
/// holds when its term is in `T`'s class.
pub const RULES: &str = r#"
(datatype V (L String i64) (K))
(datatype Srcs (Nil) (Cons V Srcs))
(datatype Fact
  (T)
  (Elem V)
  (Scalar V)
  (AllScalar Srcs)
  (Flow V V)
  (SoleDef V Srcs))
(datatype Class (ClassOf String i64) (PerElement) (ScalarThreshold))
(datatype SiteFact (Site String i64 V V))
(ruleset defuse)
(union (Scalar (K)) (T))
(union (AllScalar (Nil)) (T))
(rule ((= f (Flow a b)) (= (Elem a) (T)))
      ((union (Elem b) (T))) :ruleset defuse)
(rule ((= l (Cons v r)) (= (Scalar v) (T)) (= (AllScalar r) (T)))
      ((union (AllScalar l) (T))) :ruleset defuse)
(rule ((= d (SoleDef x s)) (= (AllScalar s) (T)))
      ((union (Scalar x) (T))) :ruleset defuse)
(rule ((= s (Site f n a b)) (= (Elem a) (T)))
      ((union (ClassOf f n) (PerElement))) :ruleset defuse)
(rule ((= s (Site f n a b)) (= (Elem b) (T)))
      ((union (ClassOf f n) (PerElement))) :ruleset defuse)
(rule ((= s (Site f n a b)) (= (Scalar a) (T)) (= (Scalar b) (T)))
      ((union (ClassOf f n) (ScalarThreshold))) :ruleset defuse)
"#;

fn local(func: &str, l: u32) -> String {
    format!("(L {func:?} {l})")
}

fn operand_term(func: &str, o: &Operand) -> Option<String> {
    match o {
        Operand::Local(l) => Some(local(func, *l)),
        Operand::Const => Some("(K)".to_string()),
        Operand::Global => None,
    }
}

/// The rows as egglog facts. A site with a global operand gets a fresh local
/// that nothing defines, so no rule can classify it.
pub fn encode(facts: &Facts) -> String {
    let mut s = String::from(RULES);
    for (f, l) in &facts.args {
        s.push_str(&format!("(union (Scalar {}) (T))\n", local(f, *l)));
    }
    for (f, l) in &facts.elems {
        s.push_str(&format!("(union (Elem {}) (T))\n", local(f, *l)));
    }
    for ((f, l), defs) in &facts.defs {
        for d in defs {
            for src in &d.2 {
                s.push_str(&format!("(Flow {} {})\n", local(f, *src), local(f, *l)));
            }
        }
        if defs.len() == 1
            && !facts.args.contains(&(f.clone(), *l))
            && !facts.elems.contains(&(f.clone(), *l))
            && let Some(d) = defs.iter().next()
        {
            let mut list = String::from("(Nil)");
            for src in d.2.iter().rev() {
                list = format!("(Cons {} {list})", local(f, *src));
            }
            s.push_str(&format!("(SoleDef {} {list})\n", local(f, *l)));
        }
    }
    for site in &facts.sites {
        let free = |k: u32| format!("(L {:?} {})", format!("{}#global", site.func), k);
        let a = operand_term(&site.func, &site.lhs).unwrap_or_else(|| free(site.stmt * 2));
        let b = operand_term(&site.func, &site.rhs).unwrap_or_else(|| free(site.stmt * 2 + 1));
        s.push_str(&format!("(Site {:?} {} {a} {b})\n", site.func, site.stmt));
    }
    s
}

/// One site with its derived class.
#[derive(Debug, Clone)]
pub struct Verdict {
    /// The site.
    pub site: Site,
    /// The derived class.
    pub class: Class,
    /// The proof egglog checked; `None` for [`Class::Unknown`].
    pub proof: Option<String>,
}

/// The result of one cold derivation.
#[derive(Debug, Clone)]
pub struct Derivation {
    /// One verdict per site, in row order.
    pub verdicts: Vec<Verdict>,
    /// `(run defuse 1)` steps until nothing changed.
    pub iterations: usize,
}

/// Why a derivation produced no verdicts.
#[derive(Debug, PartialEq, Eq)]
pub enum DeriveRefusal {
    /// egglog rejected the program.
    Egglog(String),
    /// The rules were still changing the e-graph after `fuel` steps.
    Fuel(usize),
    /// `PerElement` and `ScalarThreshold` were merged: a site got both.
    Conflict,
}

fn step_changed(outs: &[CommandOutput]) -> bool {
    for out in outs {
        if let CommandOutput::RunSchedule(report) = out {
            return report
                .iterations
                .first()
                .map_or(report.updated, |it| it.changed());
        }
    }
    true
}

fn prove(egraph: &mut EGraph, goal: &str) -> Option<String> {
    let outs = egraph
        .parse_and_run_program(None, &format!("(prove {goal})"))
        .ok()?;
    outs.iter().find_map(|out| match out {
        CommandOutput::ProveExists {
            proof_store,
            proof_id,
        } => Some(proof_store.proof_to_string(*proof_id)),
        _ => None,
    })
}

/// The egglog arm: loads the rows, steps the rules to a fixpoint, then asks
/// for a proof of each class of each site.
pub fn derive(facts: &Facts, fuel: usize) -> Result<Derivation, DeriveRefusal> {
    let mut egraph = EGraph::new_with_proofs();
    egraph
        .parse_and_run_program(None, &encode(facts))
        .map_err(|e| DeriveRefusal::Egglog(e.to_string()))?;
    let mut iterations = 0;
    loop {
        if iterations == fuel {
            return Err(DeriveRefusal::Fuel(fuel));
        }
        iterations += 1;
        let outs = egraph
            .parse_and_run_program(None, "(run defuse 1)")
            .map_err(|e| DeriveRefusal::Egglog(e.to_string()))?;
        if !step_changed(&outs) {
            break;
        }
    }
    if prove(&mut egraph, "(= (PerElement) (ScalarThreshold))").is_some() {
        return Err(DeriveRefusal::Conflict);
    }
    let mut verdicts = Vec::with_capacity(facts.sites.len());
    for site in &facts.sites {
        let goal = |class: &str| format!("(= (ClassOf {:?} {}) ({class}))", site.func, site.stmt);
        let (class, proof) = if let Some(p) = prove(&mut egraph, &goal("PerElement")) {
            (Class::PerElement, Some(p))
        } else if let Some(p) = prove(&mut egraph, &goal("ScalarThreshold")) {
            (Class::ScalarThreshold, Some(p))
        } else {
            (Class::Unknown, None)
        };
        verdicts.push(Verdict {
            site: site.clone(),
            class,
            proof,
        });
    }
    Ok(Derivation {
        verdicts,
        iterations,
    })
}

fn idents(text: &str) -> Vec<&str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| {
            w.chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        })
        .collect()
}

/// The source-only arm. `line` is one-based and must hold a `/` or `%`.
///
/// Finds the `for PATTERN in` headers whose braces are still open at `line`
/// and collects the names their patterns bind. The site is per-element when
/// the operator's line uses one of those names, or reads through `[`; it is a
/// scalar threshold otherwise. `None` when the line has no `/` or `%`
/// outside a comment.
pub fn source_class(source: &str, line: usize) -> Option<Class> {
    let lines: Vec<&str> = source.lines().collect();
    let code = |l: &str| l.split("//").next().unwrap_or("").to_string();
    let target = code(lines.get(line.checked_sub(1)?)?);
    if !target.contains('/') && !target.contains('%') {
        return None;
    }
    // (brace depth at which the loop body opened, names bound by its pattern)
    let mut open: Vec<(i32, Vec<String>)> = Vec::new();
    let mut depth = 0i32;
    for raw in &lines[..line - 1] {
        let l = code(raw);
        let mut header = l.trim_start().strip_prefix("for ").and_then(|rest| {
            rest.split_once(" in ").map(|(pat, _)| {
                idents(pat)
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
        });
        for c in l.chars() {
            match c {
                '{' => {
                    depth += 1;
                    if let Some(names) = header.take() {
                        open.push((depth, names));
                    }
                }
                '}' => {
                    if open.last().is_some_and(|(d, _)| *d == depth) {
                        open.pop();
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
    }
    let bound: BTreeSet<&str> = open
        .iter()
        .flat_map(|(_, names)| names.iter().map(String::as_str))
        .collect();
    let used = idents(&target);
    if target.contains('[') || used.iter().any(|w| bound.contains(w)) {
        Some(Class::PerElement)
    } else {
        Some(Class::ScalarThreshold)
    }
}
