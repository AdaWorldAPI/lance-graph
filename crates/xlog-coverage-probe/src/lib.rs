//! XLOG coverage reconciliation.
//!
//! XLOG's census (`xlog-cube/examples/coverage.rs`) counts which compiled
//! rules the cube admits, over EMPTY base relations, so "admitted" says
//! nothing about correctness. This probe runs the same compiled plan on the
//! cube over NON-EMPTY base relations and compares the cube's output with an
//! independent naive Datalog evaluator over the parsed AST ([`naive`]), with
//! XLOG's own reference evaluator
//! (`xlog_logic::hypergraph::evaluate_scc_fixpoint`) as a cross-check. Neither
//! reads the compiled plan the cube runs. Each compiled rule gets a [`Rung`]
//! on an evidence ladder.
//!
//! A lab probe and an oracle arm: cold, correctness only, never a timing gate.

use std::collections::{BTreeMap, BTreeSet};

pub mod datafrog_arm;
pub mod naive;

use xlog_core::{RelId, ScalarType};
use xlog_cube::{CubeExecutor, Report};
use xlog_ir::ExecutionPlan;
use xlog_logic::hypergraph::{
    evaluate_scc_fixpoint, AppearanceOrder, FixpointConfig, RefRelation, RefRelationStore, RefValue,
};
use xlog_logic::{parse_program, Atom, BodyLiteral, Compiler, Program, Rule, Term};

/// The seed whose base relations are the complete graph over values 0..6
/// (36 rows, self-loops included) instead of random rows. Sparse random rows
/// almost never close a triangle, cycle or clique; this seed always does.
pub const DENSE_SEED: u64 = 0xD15E_0000_0000_0008;

/// The seed whose base relations are the complete graph over values 0..4
/// (16 rows). A wide rule over the dense seed can exceed the naive oracle's
/// bound (a 7-variable path is 6^7 rows); over this one it is 4^7. A seed on
/// which the oracle hits its bound is skipped for that head, not judged.
pub const DENSE_SMALL_SEED: u64 = 0xD15E_0000_0000_0004;

/// Six random seeds and the two dense ones. Each random seed draws one set
/// of random base rows.
pub const DEFAULT_SEEDS: [u64; 8] = [
    DENSE_SEED,
    DENSE_SMALL_SEED,
    0x2545_F491_4F6C_DD1D,
    0x9E37_79B9_7F4A_7C15,
    0xD1B5_4A32_D192_ED03,
    0x5EED_D1A6,
    0x1234_5678_9ABC_DEF1,
    0xC2B2_AE3D_27D4_EB4F,
];

/// The reference evaluator enumerates every assignment of the rule's
/// variables over a value domain. A rule whose bound `values ^ variables`
/// exceeds this is reported oracle-unavailable instead of being run, so a
/// census over large programs terminates. A policy pin, not a measurement.
const MAX_ASSIGNMENTS: f64 = 5_000_000.0;

/// The evidence ladder, lowest to highest. Ord follows declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rung {
    /// The literal did not compile. Program-level.
    Observed,
    /// Compiled, but `execute_plan` returned `Err` for the whole plan.
    Compiled,
    /// The cube refused this rule's SCC.
    Refused,
    /// The cube reported the rule as ran over EMPTY base relations (the
    /// admission census) and refused it over non-empty ones. In a recursive
    /// SCC an empty seed delta ends the loop before any recursive body is
    /// lowered, so admission over empty bases cannot see these.
    RefusedOnData,
    /// The cube ran it; no oracle answer was available.
    LoweredUnverified,
    /// Oracle answer differs from the cube's on at least one seed.
    OracleDisagrees,
    /// Oracle answer equals the cube's on every seed.
    OracleAgrees,
}

impl Rung {
    /// The rung's name as printed by the census.
    pub fn label(self) -> &'static str {
        match self {
            Rung::Observed => "observed",
            Rung::Compiled => "compiled",
            Rung::Refused => "refused",
            Rung::RefusedOnData => "refused-on-data",
            Rung::LoweredUnverified => "lowered-unverified",
            Rung::OracleDisagrees => "oracle-disagrees",
            Rung::OracleAgrees => "oracle-agrees",
        }
    }
}

/// One compiled rule (same unit as coverage.rs: one entry of
/// `report.ran` or `report.refused`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleRecord {
    /// The rule's head predicate.
    pub head: String,
    /// Where the rule sits on the ladder.
    pub rung: Rung,
    /// Refusal text, oracle-unavailable reason, or disagreement detail. Empty for OracleAgrees.
    pub reason: String,
    /// The rule's SCC is recursive (xlog-cube evaluates recursive SCCs with a
    /// host loop over owned rows, not on cube masks). From `Scc::is_recursive`.
    pub host_loop: bool,
    /// Which oracle produced the verdict: `naive` (this crate's independent
    /// evaluator), `xlog-ref` (XLOG's reference evaluator, used only when the
    /// naive one is unavailable), or empty when neither answered.
    pub oracle: &'static str,
    /// XLOG's reference evaluator on the same seeds, as a cross-check:
    /// `agree`, `disagree`, or `unavailable`. Never decides the rung while
    /// the naive oracle has an answer.
    pub xlog_ref: &'static str,
    /// Seeds on which the deciding oracle's answer for this head was
    /// non-empty. Anti-vacuity: an agreement over only empty answers is
    /// reported, not hidden.
    pub nonempty_seeds: usize,
    /// The Datafrog evaluator ([`datafrog_arm`]) on the same seeds, as a
    /// third cross-check: `agree`, `disagree`, `not-applicable`, or empty when
    /// no Datafrog run happened for the rule. Never decides the rung.
    pub datafrog: &'static str,
    /// The first disagreement or not-applicable reason of the Datafrog run.
    /// Empty otherwise.
    pub datafrog_reason: String,
}

/// The result for one program literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramRecord {
    /// One record per compiled rule.
    pub rules: Vec<RuleRecord>,
    /// The program-level rung when no rule exists (Observed / Compiled).
    pub program_rung: Option<Rung>,
    /// Why the program-level rung was assigned. Empty when `program_rung` is `None`.
    pub program_reason: String,
}

/// Result of comparing two row sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compare {
    /// Both sides hold the same set of rows.
    Agree,
    /// The sides differ; both lists are sorted.
    Disagree {
        /// Rows the cube produced and the oracle did not.
        cube_only: Vec<Vec<i128>>,
        /// Rows the oracle produced and the cube did not.
        oracle_only: Vec<Vec<i128>>,
    },
}

/// Every `r#"..."#` literal of `text`, exactly as coverage.rs splits them.
pub fn literals(text: &str) -> Vec<&str> {
    text.split("r#\"")
        .skip(1)
        .filter_map(|t| t.split("\"#").next())
        .collect()
}

/// Set comparison: both sides are deduplicated first.
pub fn compare(cube: &[Vec<i128>], oracle: &[Vec<i128>]) -> Compare {
    let c: BTreeSet<Vec<i128>> = cube.iter().cloned().collect();
    let o: BTreeSet<Vec<i128>> = oracle.iter().cloned().collect();
    let cube_only: Vec<Vec<i128>> = c.difference(&o).cloned().collect();
    let oracle_only: Vec<Vec<i128>> = o.difference(&c).cloned().collect();
    if cube_only.is_empty() && oracle_only.is_empty() {
        Compare::Agree
    } else {
        Compare::Disagree {
            cube_only,
            oracle_only,
        }
    }
}

/// What the oracle stage concluded for one head.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Agree,
    Disagree(String),
    Unavailable(String),
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Verdict::Agree => "agree",
            Verdict::Disagree(_) => "disagree",
            Verdict::Unavailable(_) => "unavailable",
        }
    }

    /// Disagreement outranks unavailability, which outranks agreement: a head
    /// is `OracleAgrees` only if every seed agreed.
    fn rank(&self) -> u8 {
        match self {
            Verdict::Agree => 0,
            Verdict::Unavailable(_) => 1,
            Verdict::Disagree(_) => 2,
        }
    }
}

type OracleRows = Result<Vec<Vec<i128>>, String>;

/// Per head, across seeds: how many seeds gave each oracle a non-empty
/// answer, and the first seeded run on which the cube refused the rule.
#[derive(Debug, Clone, Default)]
struct Nonempty {
    naive: usize,
    xref: usize,
    refused_on_data: Option<String>,
}

/// Per head: the naive, `xlog-ref` and Datafrog verdicts, and the counts.
type Verdicts = (Verdict, Verdict, Verdict, Nonempty);

/// A base predicate and the rows it gets on each seed.
#[derive(Debug, Clone)]
enum BaseRows {
    /// Binary `u32`: the cube can hold it. The program's facts when it states
    /// any (used on every seed), otherwise `None` (random rows per seed).
    Binary(Option<Vec<(u32, u32)>>),
    /// Any other arity or type: the cube cannot hold it, so it never gets
    /// rows and refuses every rule that reads it. The naive oracle gets the
    /// stated integer facts (none stated is the empty relation), or the
    /// reason it cannot. Only the rules that READ it are affected.
    Other(Result<Vec<Vec<i128>>, String>),
}

type BaseSpec = (String, BaseRows);

/// What the oracles evaluate for one program, built once from the AST.
struct OracleProgram {
    /// The rules per head: the AST's proper rules, plus one generated rule
    /// per query (`__xlog_query_{i}`, built as the compiler builds it).
    by_head: BTreeMap<String, Vec<Rule>>,
    /// Ground facts the program states for a derived predicate: they seed
    /// that predicate in the naive oracle and are handed to the cube, as
    /// any host would. `Err` when they are not integer rows.
    derived_facts: BTreeMap<String, Result<Vec<Vec<i128>>, String>>,
}

/// The base rows handed to the cube, per relation name.
type Supplied = Vec<(String, Vec<(u32, u32)>)>;

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

/// Reconcile one program: compile, run on the cube, and compare each ran
/// head with the reference evaluator on every seed.
pub fn reconcile(src: &str, seeds: &[u64]) -> ProgramRecord {
    let mut c = Compiler::new();
    let plan = match c.compile(src) {
        Ok(p) => p,
        Err(e) => {
            return ProgramRecord {
                rules: Vec::new(),
                program_rung: Some(Rung::Observed),
                program_reason: first_line(&e.to_string()),
            }
        }
    };
    let rel_ids: Vec<(String, RelId)> = c
        .rel_ids()
        .iter()
        .map(|(name, id)| (name.clone(), *id))
        .collect();

    // Admission run, as coverage.rs: base relations of arity 2 are empty.
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
        let derived = plan
            .sccs
            .iter()
            .zip(&plan.rules_by_scc)
            .any(|(s, r)| !r.is_empty() && s.predicates.contains(name));
        if !derived && plan.rel_arities.get(id) == Some(&2) {
            ex.put_relation(name, &[]);
        }
    }
    let report = match ex.execute_plan(&plan) {
        Ok(r) => r,
        Err(e) => {
            return ProgramRecord {
                rules: Vec::new(),
                program_rung: Some(Rung::Compiled),
                program_reason: format!("{e:?}"),
            }
        }
    };

    let mut host_loop_of: BTreeMap<String, bool> = BTreeMap::new();
    for (i, scc_rules) in plan.rules_by_scc.iter().enumerate() {
        let recursive = plan.sccs.get(i).is_some_and(|s| s.is_recursive);
        for r in scc_rules {
            *host_loop_of.entry(r.head.clone()).or_insert(false) |= recursive;
        }
    }
    let host_loop = |head: &str| host_loop_of.get(head).copied().unwrap_or(false);

    let mut rules: Vec<RuleRecord> = Vec::new();
    for head in &report.ran {
        rules.push(RuleRecord {
            head: head.clone(),
            rung: Rung::LoweredUnverified,
            reason: String::new(),
            host_loop: host_loop(head),
            oracle: "",
            xlog_ref: "",
            nonempty_seeds: 0,
            datafrog: "",
            datafrog_reason: String::new(),
        });
    }
    for (head, why) in &report.refused {
        rules.push(RuleRecord {
            head: head.clone(),
            rung: Rung::Refused,
            reason: why.clone(),
            host_loop: host_loop(head),
            oracle: "",
            xlog_ref: "",
            nonempty_seeds: 0,
            datafrog: "",
            datafrog_reason: String::new(),
        });
    }

    let ran: BTreeSet<String> = report.ran.iter().cloned().collect();
    if !ran.is_empty() {
        let verdicts = oracle_stage(src, &plan, &rel_ids, seeds, &ran);
        for r in rules
            .iter_mut()
            .filter(|r| r.rung == Rung::LoweredUnverified)
        {
            let (naive, xref, dfrog, nonempty) =
                verdicts.get(&r.head).cloned().unwrap_or_else(|| {
                    let none = Verdict::Unavailable("no oracle verdict".to_string());
                    (none.clone(), none.clone(), none, Nonempty::default())
                });
            if let Some(why) = nonempty.refused_on_data {
                r.rung = Rung::RefusedOnData;
                r.reason = why;
                continue;
            }
            r.xlog_ref = xref.label();
            let (label, why) = datafrog_label(&dfrog);
            r.datafrog = label;
            r.datafrog_reason = why;
            let (chosen, name) = match (&naive, &xref) {
                (Verdict::Unavailable(_), Verdict::Agree | Verdict::Disagree(_)) => {
                    (xref.clone(), "xlog-ref")
                }
                (Verdict::Unavailable(_), _) => (naive.clone(), ""),
                _ => (naive.clone(), "naive"),
            };
            r.oracle = name;
            r.nonempty_seeds = if name == "xlog-ref" {
                nonempty.xref
            } else {
                nonempty.naive
            };
            match chosen {
                // Empty answers agreeing is not evidence (D-XLOG-COV-0 stop
                // condition): such a rule stays unverified, with the reason.
                Verdict::Agree if r.nonempty_seeds == 0 => {
                    r.reason = format!("agreed only on empty answers ({name} oracle)");
                }
                Verdict::Agree => r.rung = Rung::OracleAgrees,
                Verdict::Disagree(detail) => {
                    r.rung = Rung::OracleDisagrees;
                    r.reason = detail;
                }
                Verdict::Unavailable(why) => r.reason = why,
            }
        }
    }

    ProgramRecord {
        rules,
        program_rung: None,
        program_reason: String::new(),
    }
}

/// Run every seed: cube on the seed's rows, reference evaluator on the same
/// rows, and merge the per-seed verdicts per ran head.
fn oracle_stage(
    src: &str,
    plan: &ExecutionPlan,
    rel_ids: &[(String, RelId)],
    seeds: &[u64],
    ran: &BTreeSet<String>,
) -> BTreeMap<String, Verdicts> {
    let all = |why: String| -> BTreeMap<String, Verdicts> {
        ran.iter()
            .map(|h| {
                let v = Verdict::Unavailable(why.clone());
                (h.clone(), (v.clone(), v.clone(), v, Nonempty::default()))
            })
            .collect()
    };
    if seeds.is_empty() {
        return all("no seeds".to_string());
    }
    let program = match parse_program(src) {
        Ok(p) => p,
        Err(e) => return all(format!("ast parse failed: {}", first_line(&e.to_string()))),
    };
    let base = base_relations(&program);
    let op = oracle_program(&program);

    let pool = value_pool(&program);
    let mut merged: BTreeMap<String, Verdicts> = BTreeMap::new();
    let mut counts: BTreeMap<String, Nonempty> = BTreeMap::new();
    // Per head: naive verdicts counted, and the last seed skipped because
    // the naive oracle hit its bound there.
    let mut judged: BTreeMap<String, usize> = BTreeMap::new();
    let mut skipped: BTreeMap<String, String> = BTreeMap::new();
    let mut note = |head: &str, naive: Option<Verdict>, xref: Verdict, dfrog: Verdict| {
        let slot = merged.entry(head.to_string()).or_insert((
            Verdict::Agree,
            Verdict::Agree,
            Verdict::Agree,
            Nonempty::default(),
        ));
        if let Some(naive) = naive {
            if slot.0.rank() < naive.rank() {
                slot.0 = naive;
            }
        }
        if slot.1.rank() < xref.rank() {
            slot.1 = xref;
        }
        if slot.2.rank() < dfrog.rank() {
            slot.2 = dfrog;
        }
    };
    for &seed in seeds {
        let seeded = seed_relations(&base, &op, seed, &pool);
        let (ex, seed_report) = match run_cube(plan, rel_ids, &seeded.supplied) {
            Ok(x) => x,
            Err(e) => {
                for head in ran {
                    let v = Verdict::Unavailable(format!("seed {seed}: cube plan failed: {e}"));
                    note(head, Some(v.clone()), v.clone(), v);
                }
                continue;
            }
        };
        let dfrog_rows = datafrog_for_seed(&op, &base, plan, seeded.naive.clone());
        let naive_rows = naive_for_seed(&op, &base, plan, seeded.naive);
        let oracle = oracle_for_seed(&op, &base, plan, seeded.store);
        for head in ran {
            if !seed_report.ran.contains(head) {
                let why = seed_report
                    .refused
                    .iter()
                    .find(|(h, _)| h == head)
                    .map_or("not run", |(_, w)| w.as_str());
                let why = format!("seed {seed}: cube refused it over non-empty bases: {why}");
                counts
                    .entry(head.clone())
                    .or_default()
                    .refused_on_data
                    .get_or_insert(why.clone());
                let v = Verdict::Unavailable(why);
                note(head, Some(v.clone()), v.clone(), v);
                continue;
            }
            let c = counts.entry(head.clone()).or_default();
            if matches!(naive_rows.get(head), Some(Ok(r)) if !r.is_empty()) {
                c.naive += 1;
            }
            if matches!(oracle.get(head), Some(Ok(r)) if !r.is_empty()) {
                c.xref += 1;
            }
            let cube = ex.relation(head).unwrap_or_default();
            let judge = |rows: Option<&OracleRows>| match rows {
                None => Verdict::Unavailable(format!("no AST rule for head {head}")),
                Some(Err(why)) => Verdict::Unavailable(why.clone()),
                Some(Ok(rows)) => verdict_of(seed, &cube, rows),
            };
            let naive = match naive_rows.get(head) {
                Some(Err(why)) if is_bound(why) => {
                    skipped.insert(head.clone(), format!("seed {seed}: {why}"));
                    None
                }
                rows => {
                    *judged.entry(head.clone()).or_default() += 1;
                    Some(judge(rows))
                }
            };
            note(
                head,
                naive,
                judge(oracle.get(head)),
                judge(dfrog_rows.get(head)),
            );
        }
    }
    // A head the naive oracle judged on no seed at all has no naive verdict.
    for (head, why) in skipped {
        if judged.get(&head).copied().unwrap_or(0) == 0 {
            if let Some(slot) = merged.get_mut(&head) {
                slot.0 = Verdict::Unavailable(why);
            }
        }
    }
    for (head, c) in counts {
        if let Some(slot) = merged.get_mut(&head) {
            slot.3 = c;
        }
    }
    merged
}

/// The naive oracle stopped at one of its size bounds: that seed says
/// nothing about the rule, so it is skipped rather than judged.
fn is_bound(why: &str) -> bool {
    why.starts_with("naive oracle: more than")
}

/// One seed, one head: the cube's rows against the oracle's.
fn verdict_of(seed: u64, cube: &[Vec<u32>], oracle: &[Vec<i128>]) -> Verdict {
    let cube: Vec<Vec<i128>> = cube
        .iter()
        .map(|row| row.iter().map(|&v| i128::from(v)).collect())
        .collect();
    match compare(&cube, oracle) {
        Compare::Agree => Verdict::Agree,
        Compare::Disagree {
            cube_only,
            oracle_only,
        } => Verdict::Disagree(format!(
            "seed={seed} cube_only={:?} oracle_only={:?}",
            first_three(&cube_only),
            first_three(&oracle_only)
        )),
    }
}

fn first_three(rows: &[Vec<i128>]) -> &[Vec<i128>] {
    &rows[..rows.len().min(3)]
}

/// Base predicates of a program: every predicate some rule or query body
/// reads that is not the head of a rule with a non-empty body. A binary one
/// whose every use and stated fact has arity 2, with ground facts that fit
/// `u32`, is [`BaseRows::Binary`]; any other is [`BaseRows::Other`] and
/// affects only the rules that read it.
fn base_relations(program: &Program) -> Vec<BaseSpec> {
    let derived: BTreeSet<&str> = program
        .proper_rules()
        .map(|r| r.head.predicate.as_str())
        .collect();
    let mut arities: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    let bodies = program
        .rules
        .iter()
        .flat_map(|r| r.body.iter().filter_map(BodyLiteral::atom))
        .chain(program.queries.iter().map(|q| &q.atom));
    for atom in bodies {
        if !derived.contains(atom.predicate.as_str()) {
            arities
                .entry(atom.predicate.as_str())
                .or_default()
                .insert(atom.terms.len());
        }
    }
    let facts = facts_by_predicate(program);
    let mut out: Vec<BaseSpec> = Vec::new();
    for (name, used) in &arities {
        let stated = facts.get(name);
        let binary_rows = || -> Option<Option<Vec<(u32, u32)>>> {
            if !used.iter().all(|&n| n == 2) {
                return None;
            }
            let Some(atoms) = stated else {
                return Some(None);
            };
            let mut rows = Vec::with_capacity(atoms.len());
            for atom in atoms {
                match atom.terms.as_slice() {
                    [Term::Integer(x), Term::Integer(y)] => {
                        rows.push((u32::try_from(*x).ok()?, u32::try_from(*y).ok()?));
                    }
                    _ => return None,
                }
            }
            Some(Some(rows))
        };
        let rows = match binary_rows() {
            Some(rows) => BaseRows::Binary(rows),
            None => BaseRows::Other(match stated {
                None => Ok(Vec::new()),
                Some(atoms) => integer_rows(atoms).ok_or_else(|| {
                    format!(
                        "base relation {name} is not binary u32 and its facts are not integer rows"
                    )
                }),
            }),
        };
        out.push(((*name).to_string(), rows));
    }
    out
}

fn facts_by_predicate(program: &Program) -> BTreeMap<&str, Vec<&Atom>> {
    let mut facts: BTreeMap<&str, Vec<&Atom>> = BTreeMap::new();
    for fact in program.facts() {
        facts
            .entry(fact.head.predicate.as_str())
            .or_default()
            .push(&fact.head);
    }
    facts
}

/// Ground integer rows of `atoms`, or `None` if any term is not an integer.
fn integer_rows(atoms: &[&Atom]) -> Option<Vec<Vec<i128>>> {
    atoms
        .iter()
        .map(|a| {
            a.terms
                .iter()
                .map(|t| match t {
                    Term::Integer(n) => Some(i128::from(*n)),
                    _ => None,
                })
                .collect()
        })
        .collect()
}

/// The generated rule of query `index`, as XLOG's compiler builds it
/// (`xlog-logic/src/compile.rs`): head `__xlog_query_{index}` over the query
/// atom's distinct variables in order of first appearance, or the single
/// constant `1` when it has none; body the query atom.
fn query_rule(index: usize, atom: &Atom) -> Rule {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut terms: Vec<Term> = Vec::new();
    for t in &atom.terms {
        for name in t.variables() {
            if seen.insert(name) {
                terms.push(Term::Variable(name.to_string()));
            }
        }
    }
    if terms.is_empty() {
        terms.push(Term::Integer(1));
    }
    Rule {
        head: Atom {
            predicate: format!("__xlog_query_{index}"),
            terms,
        },
        body: vec![BodyLiteral::Positive(atom.clone())],
    }
}

fn oracle_program(program: &Program) -> OracleProgram {
    let mut by_head: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    for r in program.proper_rules() {
        by_head
            .entry(r.head.predicate.clone())
            .or_default()
            .push(r.clone());
    }
    for (i, q) in program.queries.iter().enumerate() {
        let r = query_rule(i, &q.atom);
        by_head.entry(r.head.predicate.clone()).or_default().push(r);
    }
    let derived_facts = facts_by_predicate(program)
        .into_iter()
        .filter(|(p, _)| by_head.contains_key(*p))
        .map(|(p, atoms)| {
            let rows = integer_rows(&atoms)
                .ok_or_else(|| format!("facts of derived predicate {p} are not integer rows"));
            (p.to_string(), rows)
        })
        .collect();
    OracleProgram {
        by_head,
        derived_facts,
    }
}

/// The first predicate a rule reads (positively or under negation) that is
/// unavailable to an oracle, with its reason.
fn blocked_by<'a>(
    rules: impl IntoIterator<Item = &'a Rule>,
    unavailable: &BTreeMap<String, String>,
) -> Option<String> {
    rules
        .into_iter()
        .flat_map(|r| r.body.iter().filter_map(BodyLiteral::atom))
        .find_map(|a| unavailable.get(&a.predicate).cloned())
}

/// Values 0..16 plus every integer constant the program mentions that fits
/// `u32`, sorted and deduplicated.
fn value_pool(program: &Program) -> Vec<u32> {
    let mut pool: BTreeSet<u32> = (0..16).collect();
    let mut add = |t: &Term| {
        if let Term::Integer(n) = t {
            if let Ok(v) = u32::try_from(*n) {
                pool.insert(v);
            }
        }
    };
    for rule in &program.rules {
        rule.head.terms.iter().for_each(&mut add);
        for lit in &rule.body {
            match lit {
                BodyLiteral::Comparison(c) => {
                    add(&c.left);
                    add(&c.right);
                }
                other => {
                    if let Some(a) = other.atom() {
                        a.terms.iter().for_each(&mut add);
                    }
                }
            }
        }
    }
    pool.into_iter().collect()
}

fn fnv1a64(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The rows of one base predicate for one seed: the program's own facts when
/// it states any, otherwise 1..=12 random rows over `pool` (values 0..16 plus
/// every integer constant of the program in `u32` range, so a constant
/// selection can hit) from a xorshift64 seeded with `seed ^ fnv1a64(name)`.
/// Sorted, deduplicated.
fn rows_for(
    name: &str,
    facts: &Option<Vec<(u32, u32)>>,
    seed: u64,
    pool: &[u32],
) -> Vec<(u32, u32)> {
    let mut rows: Vec<(u32, u32)> = match facts {
        Some(f) => f.clone(),
        None if seed == DENSE_SEED => (0..6).flat_map(|a| (0..6).map(move |b| (a, b))).collect(),
        None if seed == DENSE_SMALL_SEED => {
            (0..4).flat_map(|a| (0..4).map(move |b| (a, b))).collect()
        }
        None => {
            let mut s = seed ^ fnv1a64(name);
            if s == 0 {
                s = 0x9E37_79B9_7F4A_7C15;
            }
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let len = 1 + (next() % 12) as usize;
            let mut rows = Vec::with_capacity(len);
            for _ in 0..len {
                let a = pool[(next() % pool.len() as u64) as usize];
                let b = pool[(next() % pool.len() as u64) as usize];
                rows.push((a, b));
            }
            rows
        }
    };
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// One seed's rows. `store` is the reference evaluator's (binary bases
/// only, schema `[U32, U32]`); `naive` holds every base relation the naive
/// oracle can read, binary or not, plus the stated facts of derived
/// predicates; `supplied` is what the cube gets: the binary bases and the
/// binary `u32` facts of derived predicates, as a host would put them.
struct Seeded {
    store: RefRelationStore,
    naive: BTreeMap<String, naive::Relation>,
    supplied: Supplied,
}

fn seed_relations(base: &[BaseSpec], op: &OracleProgram, seed: u64, pool: &[u32]) -> Seeded {
    let mut store = RefRelationStore::new();
    let mut naive: BTreeMap<String, naive::Relation> = BTreeMap::new();
    let mut supplied = Vec::with_capacity(base.len());
    for (name, rows) in base {
        match rows {
            BaseRows::Binary(facts) => {
                let rows = rows_for(name, facts, seed, pool);
                store.insert(
                    name.clone(),
                    RefRelation {
                        schema: vec![ScalarType::U32, ScalarType::U32],
                        rows: rows
                            .iter()
                            .map(|&(a, b)| vec![RefValue::U32(a), RefValue::U32(b)])
                            .collect(),
                    },
                );
                naive.insert(
                    name.clone(),
                    rows.iter()
                        .map(|&(a, b)| vec![i128::from(a), i128::from(b)])
                        .collect(),
                );
                supplied.push((name.clone(), rows));
            }
            BaseRows::Other(Ok(rows)) => {
                naive.insert(name.clone(), rows.iter().cloned().collect());
            }
            BaseRows::Other(Err(_)) => {}
        }
    }
    for (name, rows) in &op.derived_facts {
        let Ok(rows) = rows else { continue };
        naive.insert(name.clone(), rows.iter().cloned().collect());
        let pairs: Option<Vec<(u32, u32)>> = rows
            .iter()
            .map(|r| match r.as_slice() {
                [a, b] => Some((u32::try_from(*a).ok()?, u32::try_from(*b).ok()?)),
                _ => None,
            })
            .collect();
        if let Some(pairs) = pairs {
            supplied.push((name.clone(), pairs));
        }
    }
    Seeded {
        store,
        naive,
        supplied,
    }
}

/// A fresh cube executor over the given base rows.
fn run_cube(
    plan: &ExecutionPlan,
    rel_ids: &[(String, RelId)],
    supplied: &[(String, Vec<(u32, u32)>)],
) -> Result<(CubeExecutor, Report), String> {
    let mut ex = CubeExecutor::new();
    for (name, id) in rel_ids {
        ex.register_relation(*id, name);
    }
    for (name, rows) in supplied {
        ex.put_relation(name, rows);
    }
    let report = ex.execute_plan(plan).map_err(|e| format!("{e:?}"))?;
    Ok((ex, report))
}

/// The reference evaluator over the program's AST: walk the plan's SCCs in
/// stratum order, evaluate those whose predicates head oracle rules, and
/// merge each result into the store before the next SCC. A component is
/// unavailable when it reads an unavailable predicate (a non-binary base, a
/// derived predicate with facts, or a component that failed earlier); only
/// the components that depend on it are affected.
fn oracle_for_seed(
    op: &OracleProgram,
    base: &[BaseSpec],
    plan: &ExecutionPlan,
    mut store: RefRelationStore,
) -> BTreeMap<String, OracleRows> {
    let mut unavailable: BTreeMap<String, String> = base
        .iter()
        .filter(|(_, r)| matches!(r, BaseRows::Other(_)))
        .map(|(n, _)| {
            (
                n.clone(),
                format!("reads base relation {n}, which is not binary u32"),
            )
        })
        .collect();
    let mut out: BTreeMap<String, OracleRows> = BTreeMap::new();
    for i in scc_order(plan) {
        let Some(scc) = plan.sccs.get(i) else {
            continue;
        };
        let heads: Vec<&str> = scc
            .predicates
            .iter()
            .map(String::as_str)
            .filter(|p| op.by_head.contains_key(*p))
            .collect();
        if heads.is_empty() {
            continue;
        }
        let rules: BTreeMap<String, Vec<Rule>> = heads
            .iter()
            .map(|h| ((*h).to_string(), op.by_head[*h].clone()))
            .collect();
        let why = if let Some(p) = heads.iter().find(|p| op.derived_facts.contains_key(**p)) {
            Some(format!("derived predicate {p} also has ground facts"))
        } else {
            blocked_by(rules.values().flatten(), &unavailable)
        };
        let why = why.or_else(|| {
            let nvars = rules
                .values()
                .flatten()
                .map(variable_count)
                .max()
                .unwrap_or(0);
            let constants: usize = rules.values().flatten().map(integer_constants).sum();
            let values = domain_size(&store) + constants;
            let bound = (values.max(1) as f64).powi(i32::try_from(nvars).unwrap_or(i32::MAX));
            (bound > MAX_ASSIGNMENTS).then(|| {
                format!("exhaustive reference evaluator bound: {values} values ^ {nvars} variables")
            })
        });
        let result = match why {
            Some(why) => Err(why),
            None => evaluate_scc_fixpoint(
                &rules,
                &store,
                &AppearanceOrder,
                &FixpointConfig {
                    max_iterations: 256,
                },
            )
            .map_err(|e| format!("{e:?}")),
        };
        match result {
            Ok(derived) => {
                for (head, mut rel) in derived {
                    narrow_to_u32(&mut rel);
                    out.insert(head.clone(), rows_to_i128(&rel));
                    store.insert(head, rel);
                }
            }
            Err(why) => {
                mark_failed(&mut out, &heads, &why);
                for h in &heads {
                    unavailable.insert((*h).to_string(), why.clone());
                }
            }
        }
    }
    out
}

/// The stratum order of the plan's SCC indices: every SCC named by a
/// stratum, in stratum order, then any SCC no stratum names.
fn scc_order(plan: &ExecutionPlan) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::new();
    for stratum in &plan.strata {
        for &i in &stratum.sccs {
            let i = i as usize;
            if !order.contains(&i) {
                order.push(i);
            }
        }
    }
    for i in 0..plan.sccs.len() {
        if !order.contains(&i) {
            order.push(i);
        }
    }
    order
}

/// The independent naive evaluator over the same seed rows, component by
/// component in the plan's stratum order. Only the ORDER comes from the
/// plan; every row is derived from the AST. A derived predicate starts from
/// the facts the program states for it. A component that reads an
/// unavailable predicate is unavailable; the others are judged.
fn naive_for_seed(
    op: &OracleProgram,
    base: &[BaseSpec],
    plan: &ExecutionPlan,
    mut store: BTreeMap<String, naive::Relation>,
) -> BTreeMap<String, OracleRows> {
    let mut unavailable: BTreeMap<String, String> = base
        .iter()
        .filter_map(|(n, r)| match r {
            BaseRows::Other(Err(why)) => Some((n.clone(), why.clone())),
            _ => None,
        })
        .chain(
            op.derived_facts
                .iter()
                .filter_map(|(n, r)| r.as_ref().err().map(|why| (n.clone(), why.clone()))),
        )
        .collect();
    let mut out: BTreeMap<String, OracleRows> = BTreeMap::new();
    for i in scc_order(plan) {
        let Some(scc) = plan.sccs.get(i) else {
            continue;
        };
        let heads: Vec<&str> = scc
            .predicates
            .iter()
            .map(String::as_str)
            .filter(|p| op.by_head.contains_key(*p))
            .collect();
        if heads.is_empty() {
            continue;
        }
        let rules: Vec<&Rule> = heads.iter().flat_map(|h| op.by_head[*h].iter()).collect();
        let blocked = heads
            .iter()
            .find_map(|h| unavailable.get(*h).cloned())
            .or_else(|| blocked_by(rules.iter().copied(), &unavailable));
        let result = match blocked {
            Some(why) => Err(why),
            None => naive::evaluate_component(&rules, &mut store),
        };
        match result {
            Ok(()) => {
                for h in &heads {
                    let rows = store.get(*h).cloned().unwrap_or_default();
                    out.insert((*h).to_string(), Ok(rows.into_iter().collect()));
                }
            }
            Err(why) => {
                mark_failed(&mut out, &heads, &why);
                for h in &heads {
                    unavailable.insert((*h).to_string(), why.clone());
                }
            }
        }
    }
    out
}

/// The Datafrog evaluator over the same seed rows, component by component in
/// the plan's stratum order: a structural copy of [`naive_for_seed`] that calls
/// [`datafrog_arm::evaluate_component`]. A component that reads an unavailable
/// predicate is not applicable, with a reason in the arm's own wording.
fn datafrog_for_seed(
    op: &OracleProgram,
    base: &[BaseSpec],
    plan: &ExecutionPlan,
    mut store: BTreeMap<String, naive::Relation>,
) -> BTreeMap<String, OracleRows> {
    let mut unavailable: BTreeMap<String, String> = base
        .iter()
        .filter_map(|(n, r)| match r {
            BaseRows::Other(Err(why)) => Some((n.clone(), why.clone())),
            _ => None,
        })
        .chain(
            op.derived_facts
                .iter()
                .filter_map(|(n, r)| r.as_ref().err().map(|why| (n.clone(), why.clone()))),
        )
        .collect();
    let mut out: BTreeMap<String, OracleRows> = BTreeMap::new();
    for i in scc_order(plan) {
        let Some(scc) = plan.sccs.get(i) else {
            continue;
        };
        let heads: Vec<&str> = scc
            .predicates
            .iter()
            .map(String::as_str)
            .filter(|p| op.by_head.contains_key(*p))
            .collect();
        if heads.is_empty() {
            continue;
        }
        let rules: Vec<&Rule> = heads.iter().flat_map(|h| op.by_head[*h].iter()).collect();
        let result = match datafrog_blocked(&heads, &rules, &unavailable) {
            Some(why) => Err(why),
            None => datafrog_arm::evaluate_component(&rules, &mut store),
        };
        match result {
            Ok(()) => {
                for h in &heads {
                    let rows = store.get(*h).cloned().unwrap_or_default();
                    out.insert((*h).to_string(), Ok(rows.into_iter().collect()));
                }
            }
            Err(why) => {
                mark_failed(&mut out, &heads, &why);
                for h in &heads {
                    unavailable.insert((*h).to_string(), why.clone());
                }
            }
        }
    }
    out
}

/// The first predicate a component's heads or rule bodies read that is
/// unavailable, worded as a Datafrog not-applicable reason. An upstream reason
/// that already is one is reused.
fn datafrog_blocked(
    heads: &[&str],
    rules: &[&Rule],
    unavailable: &BTreeMap<String, String>,
) -> Option<String> {
    let from_heads = heads.iter().find_map(|h| {
        unavailable
            .get(*h)
            .map(|why| ((*h).to_string(), why.clone()))
    });
    let hit = from_heads.or_else(|| {
        rules
            .iter()
            .flat_map(|r| r.body.iter().filter_map(BodyLiteral::atom))
            .find_map(|a| {
                unavailable
                    .get(&a.predicate)
                    .map(|why| (a.predicate.clone(), why.clone()))
            })
    });
    hit.map(|(pred, why)| {
        if why.starts_with(datafrog_arm::NOT_APPLICABLE) {
            why
        } else {
            format!(
                "{}reads {pred}, which is unavailable: {why}",
                datafrog_arm::NOT_APPLICABLE
            )
        }
    })
}

/// The census label and reason of one Datafrog verdict.
fn datafrog_label(v: &Verdict) -> (&'static str, String) {
    match v {
        Verdict::Agree => ("agree", String::new()),
        Verdict::Disagree(detail) => ("disagree", detail.clone()),
        Verdict::Unavailable(why) => ("not-applicable", why.clone()),
    }
}

fn mark_failed(out: &mut BTreeMap<String, OracleRows>, heads: &[&str], why: &str) {
    for h in heads {
        out.insert((*h).to_string(), Err(why.to_string()));
    }
}

fn variable_count(rule: &Rule) -> usize {
    rule.body_variables()
        .into_iter()
        .collect::<BTreeSet<&str>>()
        .len()
}

fn integer_constants(rule: &Rule) -> usize {
    let is_int = |t: &Term| usize::from(matches!(t, Term::Integer(_)));
    let mut n: usize = rule.head.terms.iter().map(is_int).sum();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Comparison(c) => n += is_int(&c.left) + is_int(&c.right),
            other => {
                if let Some(a) = other.atom() {
                    n += a.terms.iter().map(is_int).sum::<usize>();
                }
            }
        }
    }
    n
}

fn domain_size(store: &RefRelationStore) -> usize {
    let mut seen: BTreeSet<&RefValue> = BTreeSet::new();
    for rel in store.values() {
        for row in &rel.rows {
            for v in row {
                seen.insert(v);
            }
        }
    }
    seen.len()
}

fn int_of(v: &RefValue) -> Option<i128> {
    match v {
        RefValue::U32(n) => Some(i128::from(*n)),
        RefValue::U64(n) => Some(i128::from(*n)),
        RefValue::I32(n) => Some(i128::from(*n)),
        RefValue::I64(n) => Some(i128::from(*n)),
        RefValue::Bool(_) | RefValue::Symbol(_) => None,
    }
}

/// The cube holds `u32` only. The reference evaluator emits a head integer
/// constant as `I64`, and equality across variants is false in its matcher
/// (`U32(1) != I64(1)`), so a derived column whose every value is an integer
/// in `u32` range is narrowed to `U32` before a later SCC reads it.
fn narrow_to_u32(rel: &mut RefRelation) {
    let max = i128::from(u32::MAX);
    for col in 0..rel.schema.len() {
        let fits = rel.rows.iter().all(|row| {
            row.get(col)
                .and_then(int_of)
                .is_some_and(|v| (0..=max).contains(&v))
        });
        if !fits {
            continue;
        }
        for row in rel.rows.iter_mut() {
            if let Some(v) = row.get(col).and_then(int_of) {
                row[col] = RefValue::U32(v as u32);
            }
        }
        rel.schema[col] = ScalarType::U32;
    }
    rel.rows.sort();
    rel.rows.dedup();
}

fn rows_to_i128(rel: &RefRelation) -> OracleRows {
    let mut out: Vec<Vec<i128>> = Vec::with_capacity(rel.rows.len());
    for row in &rel.rows {
        let mut cells: Vec<i128> = Vec::with_capacity(row.len());
        for v in row {
            match int_of(v) {
                Some(n) => cells.push(n),
                None => return Err("non-integer cell".to_string()),
            }
        }
        out.push(cells);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const POOL: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    #[test]
    fn a_disagreement_names_the_first_three_rows_of_each_side() {
        let oracle: Vec<Vec<i128>> = (0..5).map(|i| vec![i, 9]).collect();
        let detail = match verdict_of(7, &[vec![1, 2]], &oracle) {
            Verdict::Disagree(d) => d,
            other => panic!("expected Disagree, got {other:?}"),
        };
        assert!(detail.starts_with("seed=7 "), "{detail}");
        assert!(detail.contains("cube_only=[[1, 2]]"), "{detail}");
        assert!(detail.contains("[2, 9]"), "{detail}");
        assert!(
            !detail.contains("[3, 9]"),
            "only three rows are named: {detail}"
        );
        assert_eq!(verdict_of(7, &[vec![1, 2]], &[vec![1, 2]]), Verdict::Agree);
    }

    #[test]
    fn a_verdict_of_disagreement_outranks_the_rest() {
        let agree = Verdict::Agree;
        let none = Verdict::Unavailable("x".into());
        let bad = Verdict::Disagree("y".into());
        assert!(agree.rank() < none.rank() && none.rank() < bad.rank());
    }

    #[test]
    fn narrowing_turns_i64_head_constants_into_u32_and_leaves_the_rest() {
        let mut rel = RefRelation {
            schema: vec![ScalarType::U32, ScalarType::I64],
            rows: vec![
                vec![RefValue::U32(1), RefValue::I64(1)],
                vec![RefValue::U32(2), RefValue::I64(-1)],
            ],
        };
        narrow_to_u32(&mut rel);
        assert_eq!(rel.schema, vec![ScalarType::U32, ScalarType::I64]);
        assert_eq!(
            rel.rows[0][1],
            RefValue::I64(1),
            "a negative keeps the column wide"
        );
        let mut rel = RefRelation {
            schema: vec![ScalarType::U32, ScalarType::I64],
            rows: vec![vec![RefValue::U32(1), RefValue::I64(1)]],
        };
        narrow_to_u32(&mut rel);
        assert_eq!(rel.schema, vec![ScalarType::U32, ScalarType::U32]);
        assert_eq!(rel.rows[0][1], RefValue::U32(1));
    }

    #[test]
    fn generated_rows_are_small_deterministic_and_seed_dependent() {
        let mut distinct = BTreeSet::new();
        for seed in DEFAULT_SEEDS
            .into_iter()
            .filter(|&s| s != DENSE_SEED && s != DENSE_SMALL_SEED)
        {
            let rows = rows_for("r", &None, seed, &POOL);
            assert!(!rows.is_empty() && rows.len() <= 12, "{rows:?}");
            assert!(rows.iter().all(|&(a, b)| a < 16 && b < 16), "{rows:?}");
            assert_eq!(rows, rows_for("r", &None, seed, &POOL));
            distinct.insert(rows);
        }
        assert!(distinct.len() > 1, "every seed drew the same rows");
        assert_eq!(rows_for("r", &None, DENSE_SEED, &POOL).len(), 36);
        assert_eq!(rows_for("r", &None, DENSE_SMALL_SEED, &POOL).len(), 16);
        assert_ne!(
            rows_for("r", &None, 5, &POOL),
            rows_for("s", &None, 5, &POOL)
        );
        let stated = Some(vec![(3, 4), (1, 2), (3, 4)]);
        assert_eq!(rows_for("r", &stated, 5, &POOL), vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn base_relations_classify_what_is_not_binary_u32() {
        let ternary = parse_program("q(X,Z) :- t(X,Y,Z), r(Z,W).").unwrap();
        let base = base_relations(&ternary);
        assert!(
            matches!(&base[..], [(r, BaseRows::Binary(None)), (t, BaseRows::Other(Ok(rows)))]
                if r == "r" && t == "t" && rows.is_empty()),
            "{base:?}"
        );
        let words = parse_program("q(X,Z) :- r(X,Y), s(Y,Z). r(\"a\", 1).").unwrap();
        let base = base_relations(&words);
        assert!(
            matches!(&base[0], (r, BaseRows::Other(Err(why))) if r == "r" && why.starts_with("base relation r ")),
            "{base:?}"
        );
        let ok = parse_program("q(X,Z) :- r(X,Y), s(Y,Z). r(1, 2). r(2, 3).").unwrap();
        let base = base_relations(&ok);
        assert!(
            matches!(&base[..], [(r, BaseRows::Binary(Some(rows))), (s, BaseRows::Binary(None))]
                if r == "r" && s == "s" && rows == &vec![(1, 2), (2, 3)]),
            "{base:?}"
        );
        let n_ary = parse_program("q(X) :- t(X,1,2). t(5, 1, 2).").unwrap();
        let base = base_relations(&n_ary);
        assert!(
            matches!(&base[..], [(t, BaseRows::Other(Ok(rows)))] if t == "t" && rows == &vec![vec![5, 1, 2]]),
            "{base:?}"
        );
    }

    #[test]
    fn a_query_rule_is_built_as_the_compiler_builds_it() {
        let p = parse_program("p(1, 2). ?- p(X, X). ?- p(1, 2).").unwrap();
        let r0 = query_rule(0, &p.queries[0].atom);
        assert_eq!(r0.head.predicate, "__xlog_query_0");
        assert_eq!(r0.head.terms, vec![Term::Variable("X".to_string())]);
        let r1 = query_rule(1, &p.queries[1].atom);
        assert_eq!(r1.head.terms, vec![Term::Integer(1)]);
        assert_eq!(r1.body.len(), 1);
    }

    /// Anti-vacuity: the probe agreeing with the oracle means little if the
    /// generated base rows never make the join produce anything.
    #[test]
    fn generated_rows_make_the_chain_join_produce_rows() {
        let mut c = Compiler::new();
        let plan = c.compile("q(X,Z) :- r(X,Y), s(Y,Z).").expect("compile");
        let rel_ids: Vec<(String, RelId)> =
            c.rel_ids().iter().map(|(n, i)| (n.clone(), *i)).collect();
        let base = vec![
            ("r".to_string(), BaseRows::Binary(None)),
            ("s".to_string(), BaseRows::Binary(None)),
        ];
        let op = oracle_program(&parse_program("q(X,Z) :- r(X,Y), s(Y,Z).").unwrap());
        let nonempty = DEFAULT_SEEDS
            .iter()
            .filter(|&&seed| {
                let seeded = seed_relations(&base, &op, seed, &POOL);
                let (ex, report) = run_cube(&plan, &rel_ids, &seeded.supplied).expect("plan");
                assert!(report.refused.is_empty(), "{report:?}");
                !ex.relation("q").unwrap_or_default().is_empty()
            })
            .count();
        assert!(nonempty > 0, "no seed produced a row of q");
    }
}
