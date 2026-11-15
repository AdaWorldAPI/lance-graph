//! XLOG coverage reconciliation.
//!
//! XLOG's census (`xlog-cube/examples/coverage.rs`) counts which compiled
//! rules the cube admits, over EMPTY base relations, so "admitted" says
//! nothing about correctness. This probe runs the same compiled plan on the
//! cube over NON-EMPTY base relations and compares the cube's output with
//! XLOG's own CPU reference fixpoint evaluator
//! (`xlog_logic::hypergraph::evaluate_scc_fixpoint`), which works from the
//! parsed AST and not from the compiled plan the cube runs. Each compiled
//! rule gets a [`Rung`] on an evidence ladder.
//!
//! A lab probe and an oracle arm: cold, correctness only, never a timing gate.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{RelId, ScalarType};
use xlog_cube::{CubeExecutor, Report};
use xlog_ir::ExecutionPlan;
use xlog_logic::hypergraph::{
    evaluate_scc_fixpoint, AppearanceOrder, FixpointConfig, RefRelation, RefRelationStore,
    RefValue,
};
use xlog_logic::{parse_program, Atom, BodyLiteral, Compiler, Program, Rule, Term};

/// Six distinct non-zero seeds. Each seed draws one set of random base rows.
pub const DEFAULT_SEEDS: [u64; 6] = [
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

/// A base predicate and, when the program states ground facts for it, those
/// facts (used instead of random rows).
type BaseSpec = (String, Option<Vec<(u32, u32)>>);

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
        });
    }
    for (head, why) in &report.refused {
        rules.push(RuleRecord {
            head: head.clone(),
            rung: Rung::Refused,
            reason: why.clone(),
            host_loop: host_loop(head),
        });
    }

    let ran: BTreeSet<String> = report.ran.iter().cloned().collect();
    if !ran.is_empty() {
        let verdicts = oracle_stage(src, &plan, &rel_ids, seeds, &ran);
        for r in rules.iter_mut().filter(|r| r.rung == Rung::LoweredUnverified) {
            match verdicts.get(&r.head) {
                Some(Verdict::Agree) => r.rung = Rung::OracleAgrees,
                Some(Verdict::Disagree(detail)) => {
                    r.rung = Rung::OracleDisagrees;
                    r.reason = detail.clone();
                }
                Some(Verdict::Unavailable(why)) => r.reason = why.clone(),
                None => r.reason = "no oracle verdict".to_string(),
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
) -> BTreeMap<String, Verdict> {
    let all = |why: String| -> BTreeMap<String, Verdict> {
        ran.iter()
            .map(|h| (h.clone(), Verdict::Unavailable(why.clone())))
            .collect()
    };
    if seeds.is_empty() {
        return all("no seeds".to_string());
    }
    let program = match parse_program(src) {
        Ok(p) => p,
        Err(e) => return all(format!("ast parse failed: {}", first_line(&e.to_string()))),
    };
    let base = match base_relations(&program) {
        Ok(b) => b,
        Err(why) => return all(why),
    };

    let mut merged: BTreeMap<String, Verdict> = BTreeMap::new();
    let mut note = |head: &str, v: Verdict| {
        let replace = merged.get(head).is_none_or(|old| old.rank() < v.rank());
        if replace {
            merged.insert(head.to_string(), v);
        }
    };
    for &seed in seeds {
        let (store, supplied) = seed_relations(&base, seed);
        let (ex, seed_report) = match run_cube(plan, rel_ids, &supplied) {
            Ok(x) => x,
            Err(e) => {
                for head in ran {
                    note(
                        head,
                        Verdict::Unavailable(format!("seed {seed}: cube plan failed: {e}")),
                    );
                }
                continue;
            }
        };
        let oracle = oracle_for_seed(&program, plan, store);
        for head in ran {
            if !seed_report.ran.contains(head) {
                let why = seed_report
                    .refused
                    .iter()
                    .find(|(h, _)| h == head)
                    .map_or("not run", |(_, w)| w.as_str());
                note(
                    head,
                    Verdict::Unavailable(format!("seed {seed}: cube did not run it: {why}")),
                );
                continue;
            }
            let v = match oracle.get(head) {
                None => Verdict::Unavailable(format!("no AST rule for head {head}")),
                Some(Err(why)) => Verdict::Unavailable(why.clone()),
                Some(Ok(rows)) => {
                    let cube = ex.relation(head).unwrap_or_default();
                    verdict_of(seed, &cube, rows)
                }
            };
            note(head, v);
        }
    }
    merged
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

/// Base predicates of a program: every predicate some rule body reads that
/// is not the head of a rule with a non-empty body. All of them must be
/// binary (every use arity 2) with, where the program states facts for them,
/// ground non-negative integer facts that fit `u32`. Otherwise the whole
/// program has no oracle.
fn base_relations(program: &Program) -> Result<Vec<BaseSpec>, String> {
    let derived: BTreeSet<&str> = program
        .proper_rules()
        .map(|r| r.head.predicate.as_str())
        .collect();
    let mut arities: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for rule in &program.rules {
        for lit in &rule.body {
            if let Some(atom) = lit.atom() {
                if !derived.contains(atom.predicate.as_str()) {
                    arities
                        .entry(atom.predicate.as_str())
                        .or_default()
                        .insert(atom.terms.len());
                }
            }
        }
    }
    let mut facts: BTreeMap<&str, Vec<&Atom>> = BTreeMap::new();
    for fact in program.facts() {
        facts
            .entry(fact.head.predicate.as_str())
            .or_default()
            .push(&fact.head);
    }
    let mut out: Vec<BaseSpec> = Vec::new();
    for (name, used) in &arities {
        let not_binary = || format!("base relation {name} is not binary u32");
        let stated = facts.get(name);
        let fact_arities_ok = stated.is_none_or(|atoms| atoms.iter().all(|a| a.terms.len() == 2));
        if !used.iter().all(|&n| n == 2) || !fact_arities_ok {
            return Err(not_binary());
        }
        let rows = match stated {
            None => None,
            Some(atoms) => {
                let mut rows: Vec<(u32, u32)> = Vec::new();
                for atom in atoms {
                    match atom.terms.as_slice() {
                        [Term::Integer(x), Term::Integer(y)] => {
                            match (u32::try_from(*x), u32::try_from(*y)) {
                                (Ok(x), Ok(y)) => rows.push((x, y)),
                                _ => return Err(not_binary()),
                            }
                        }
                        _ => return Err(not_binary()),
                    }
                }
                Some(rows)
            }
        };
        out.push(((*name).to_string(), rows));
    }
    Ok(out)
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
/// it states any, otherwise 1..=12 random rows over values 0..16 from a
/// xorshift64 seeded with `seed ^ fnv1a64(name)`. Sorted, deduplicated.
fn rows_for(name: &str, facts: &Option<Vec<(u32, u32)>>, seed: u64) -> Vec<(u32, u32)> {
    let mut rows: Vec<(u32, u32)> = match facts {
        Some(f) => f.clone(),
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
                let a = (next() % 16) as u32;
                let b = (next() % 16) as u32;
                rows.push((a, b));
            }
            rows
        }
    };
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The seed's base rows, as the oracle's store (schema `[U32, U32]`) and as
/// the rows handed to the cube.
fn seed_relations(
    base: &[BaseSpec],
    seed: u64,
) -> (RefRelationStore, Vec<(String, Vec<(u32, u32)>)>) {
    let mut store = RefRelationStore::new();
    let mut supplied = Vec::with_capacity(base.len());
    for (name, facts) in base {
        let rows = rows_for(name, facts, seed);
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
        supplied.push((name.clone(), rows));
    }
    (store, supplied)
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
/// stratum order, evaluate those whose predicates head AST rules with
/// non-empty bodies, and merge each result into the store before the next SCC.
/// A failure (or a derived predicate that also has facts) makes that SCC's
/// heads and every later SCC's heads oracle-unavailable.
fn oracle_for_seed(
    program: &Program,
    plan: &ExecutionPlan,
    mut store: RefRelationStore,
) -> BTreeMap<String, OracleRows> {
    let mut by_head: BTreeMap<&str, Vec<&Rule>> = BTreeMap::new();
    for r in program.proper_rules() {
        by_head.entry(r.head.predicate.as_str()).or_default().push(r);
    }
    let fact_heads: BTreeSet<&str> = program
        .facts()
        .map(|f| f.head.predicate.as_str())
        .collect();

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

    let mut out: BTreeMap<String, OracleRows> = BTreeMap::new();
    let mut poisoned: Option<String> = None;
    for i in order {
        let Some(scc) = plan.sccs.get(i) else {
            continue;
        };
        let heads: Vec<&str> = scc
            .predicates
            .iter()
            .map(String::as_str)
            .filter(|p| by_head.contains_key(p))
            .collect();
        if heads.is_empty() {
            continue;
        }
        if let Some(why) = &poisoned {
            mark_failed(&mut out, &heads, why);
            continue;
        }
        if let Some(p) = heads.iter().find(|p| fact_heads.contains(*p)) {
            let why = format!("derived predicate {p} also has ground facts");
            mark_failed(&mut out, &heads, &why);
            poisoned = Some(why);
            continue;
        }
        let rules: BTreeMap<String, Vec<Rule>> = heads
            .iter()
            .map(|h| {
                (
                    (*h).to_string(),
                    by_head[h].iter().map(|r| (**r).clone()).collect(),
                )
            })
            .collect();
        let nvars = rules
            .values()
            .flatten()
            .map(variable_count)
            .max()
            .unwrap_or(0);
        let constants: usize = rules.values().flatten().map(integer_constants).sum();
        let values = domain_size(&store) + constants;
        let bound = (values.max(1) as f64).powi(i32::try_from(nvars).unwrap_or(i32::MAX));
        if bound > MAX_ASSIGNMENTS {
            let why =
                format!("exhaustive reference evaluator bound: {values} values ^ {nvars} variables");
            mark_failed(&mut out, &heads, &why);
            poisoned = Some(why);
            continue;
        }
        match evaluate_scc_fixpoint(
            &rules,
            &store,
            &AppearanceOrder,
            &FixpointConfig { max_iterations: 256 },
        ) {
            Ok(derived) => {
                for (head, mut rel) in derived {
                    narrow_to_u32(&mut rel);
                    let rows = rows_to_i128(&rel);
                    out.insert(head.clone(), rows);
                    store.insert(head, rel);
                }
            }
            Err(e) => {
                let why = format!("{e:?}");
                mark_failed(&mut out, &heads, &why);
                poisoned = Some(why);
            }
        }
    }
    out
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
        assert!(!detail.contains("[3, 9]"), "only three rows are named: {detail}");
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
        assert_eq!(rel.rows[0][1], RefValue::I64(1), "a negative keeps the column wide");
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
        for seed in DEFAULT_SEEDS {
            let rows = rows_for("r", &None, seed);
            assert!(!rows.is_empty() && rows.len() <= 12, "{rows:?}");
            assert!(rows.iter().all(|&(a, b)| a < 16 && b < 16), "{rows:?}");
            assert_eq!(rows, rows_for("r", &None, seed));
            distinct.insert(rows);
        }
        assert!(distinct.len() > 1, "every seed drew the same rows");
        assert_ne!(rows_for("r", &None, 5), rows_for("s", &None, 5));
        let stated = Some(vec![(3, 4), (1, 2), (3, 4)]);
        assert_eq!(rows_for("r", &stated, 5), vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn base_relations_refuse_what_is_not_binary_u32() {
        let ternary = parse_program("q(X,Z) :- t(X,Y,Z), r(Z,W).").unwrap();
        let why = base_relations(&ternary).unwrap_err();
        assert!(why.starts_with("base relation t "), "{why}");
        let words = parse_program("q(X,Z) :- r(X,Y), s(Y,Z). r(\"a\", 1).").unwrap();
        let why = base_relations(&words).unwrap_err();
        assert!(why.starts_with("base relation r "), "{why}");
        let ok = parse_program("q(X,Z) :- r(X,Y), s(Y,Z). r(1, 2). r(2, 3).").unwrap();
        let base = base_relations(&ok).unwrap();
        assert_eq!(
            base,
            vec![
                ("r".to_string(), Some(vec![(1, 2), (2, 3)])),
                ("s".to_string(), None)
            ]
        );
    }

    /// Anti-vacuity: the probe agreeing with the oracle means little if the
    /// generated base rows never make the join produce anything.
    #[test]
    fn generated_rows_make_the_chain_join_produce_rows() {
        let mut c = Compiler::new();
        let plan = c.compile("q(X,Z) :- r(X,Y), s(Y,Z).").expect("compile");
        let rel_ids: Vec<(String, RelId)> = c
            .rel_ids()
            .iter()
            .map(|(n, i)| (n.clone(), *i))
            .collect();
        let base = vec![("r".to_string(), None), ("s".to_string(), None)];
        let nonempty = DEFAULT_SEEDS
            .iter()
            .filter(|&&seed| {
                let (_, supplied) = seed_relations(&base, seed);
                let (ex, report) = run_cube(&plan, &rel_ids, &supplied).expect("plan");
                assert!(report.refused.is_empty(), "{report:?}");
                !ex.relation("q").unwrap_or_default().is_empty()
            })
            .count();
        assert!(nonempty > 0, "no seed produced a row of q");
    }
}
