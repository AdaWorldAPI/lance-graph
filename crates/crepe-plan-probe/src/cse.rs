//! Cross-rule CSE (D-RU-3 / U3): every rule's star over ONE shared table,
//! and the parts several rules share evaluated once.
//!
//! STATUS: PROBE (results only), like the rest of this crate.
//!
//! [`crate::pipeline::evaluate`] prepares one table per rule and runs one
//! fold per rule, so a core that 256 rules share is laid out and ANDed 256
//! times. Here:
//!
//! 1. the distinct sides ([`SideKey`]) of all stars are laid out once, over
//!    one domain: the union of their centers. A rule's joint mask is the AND
//!    of its sides' presence planes; a center outside some side's domain is
//!    absent from that side's plane, so the union domain changes no rule's
//!    mask, only its width;
//! 2. each rule becomes `And(Sym(side), …)` over those planes, and
//!    [`share`] lifts every part two or more rules hold into a let
//!    (`lance_graph_contract::set_program`);
//! 3. [`run_program`] evaluates each let once into a plane and every rule
//!    over the sides plus those planes; the tuples are read per rule with
//!    [`tuples_over_mask`], exactly as the per-rule path reads them.
//!
//! With `sharing` off the same table is used and no let is formed, so the
//! two effects (one table, shared folds) are measured apart.
//!
//! The shared path ignores [`Options::semijoin`]: a semijoin reduces one
//! star's sides to the centers its restricted sides have, which differs per
//! rule and would give each rule its own domain again.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use lance_graph_contract::set_algebra::SetExpr;
use lance_graph_contract::set_program::{SetProgram, share};
use lance_graph_mask_risc::{Planes, words_for};
use lance_graph_relalg::{Answer, Measure, ProgramCost, ProgramError, run_program};
use mork_pathmap_probe::Semantics;
use mork_pathmap_probe::join::{JoinRefusal, Side};

use crate::logic::{Rule, VarId};
use crate::native::{PreparedStar, SideKey, SidePlan, StarPlan, Store, prepare, tuples_over_mask};
use crate::optimize::{Demand, Options, ResultKind};
use crate::pipeline::{Outcome, PipelineRefusal, Planned, plan};

/// Every distinct side of a set of stars, laid out once over one domain.
pub struct SharedStars {
    /// The table; its side `k` holds `keys[k]`.
    pub prepared: PreparedStar,
    /// For star `j`, the table side of each of its plan sides.
    pub rule_sides: Vec<Vec<usize>>,
}

/// Lays out the distinct sides of `stars` over the union of their centers.
pub fn prepare_shared(stars: &[StarPlan], store: &Store<'_>) -> SharedStars {
    let mut keys: Vec<SideKey> = Vec::new();
    let rule_sides = stars
        .iter()
        .map(|s| {
            s.sides
                .iter()
                .map(|side| match keys.iter().position(|k| *k == side.key) {
                    Some(i) => i,
                    None => {
                        keys.push(side.key);
                        keys.len() - 1
                    }
                })
                .collect()
        })
        .collect();
    // A star whose sides are the distinct keys: `prepare` lays them out over
    // one domain with the same per-side restriction as the per-rule path.
    let all = StarPlan {
        center: VarId(0),
        sides: keys
            .iter()
            .map(|&key| SidePlan { key, emit: None })
            .collect(),
        head: Vec::new(),
    };
    SharedStars {
        prepared: prepare(&all, store),
        rule_sides,
    }
}

/// Each star as the conjunction of its table sides, as set expressions over
/// `Sym(side)`.
pub fn rule_sets(shared: &SharedStars) -> Vec<SetExpr> {
    shared
        .rule_sides
        .iter()
        .map(|sides| SetExpr::And(sides.iter().map(|&k| SetExpr::Sym(k as u32)).collect()))
        .collect()
}

/// The version a program over this table is built for: the generations of
/// its sides, in order. A program for other generations is refused by
/// [`run_program`].
pub fn table_version(generations: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for g in generations {
        for b in g.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h ^ generations.len() as u64
}

/// Why the shared path refused.
#[derive(Debug, Clone, PartialEq)]
pub enum CseRefusal {
    Pipeline(PipelineRefusal),
    /// A side is stale against the generations the caller passed.
    Table(JoinRefusal),
    /// The caller passed another number of generations than the table has
    /// sides.
    Generations {
        sides: usize,
        given: usize,
    },
    Program(ProgramError),
}

/// What one shared evaluation cost, each part on its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct SharedReport {
    /// Sides laid out (distinct across all rules).
    pub sides: usize,
    /// Slots of the shared domain.
    pub domain: usize,
    /// Rows laid out over all sides.
    pub rows_prepared: usize,
    /// Lets formed.
    pub lets: usize,
    /// Folds, ops, planes read and materialized.
    pub cost: ProgramCost,
    pub plan: Duration,
    pub prepare: Duration,
    pub execute: Duration,
}

/// The program the shared path runs: the rules over the table's sides, with
/// the shared parts as lets when `sharing` is on.
pub fn program(shared: &SharedStars, version: u64, sharing: bool) -> SetProgram {
    let resident = shared.prepared.table.sides.len() as u32;
    let queries = rule_sets(shared);
    if sharing {
        share(version, resident, &queries)
    } else {
        SetProgram {
            version,
            resident,
            lets: Vec::new(),
            queries,
        }
    }
}

/// [`crate::pipeline::evaluate`] over one shared table. Same rewrites, same
/// answer; `sharing` decides whether shared parts are evaluated once.
pub fn evaluate_shared(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &Store<'_>,
    sharing: bool,
) -> Result<(Outcome, SharedReport), CseRefusal> {
    let t = Instant::now();
    let planned = plan(rules, demand, kind, opts).map_err(CseRefusal::Pipeline)?;
    let plan_cost = t.elapsed();
    let t = Instant::now();
    let shared = prepare_shared(&planned.stars, store);
    let prepare_cost = t.elapsed();
    let generations: Vec<u64> = shared
        .prepared
        .keys
        .iter()
        .map(|k| store.view(k.rel).1)
        .collect();
    let (outcome, mut report) = execute_shared(&planned, &shared, kind, &generations, sharing)?;
    report.plan = plan_cost;
    report.prepare = prepare_cost;
    Ok((outcome, report))
}

/// The run half over a prepared shared table. `generations[k]` is the
/// current generation of table side `k`.
pub fn execute_shared(
    planned: &Planned,
    shared: &SharedStars,
    kind: ResultKind,
    generations: &[u64],
    sharing: bool,
) -> Result<(Outcome, SharedReport), CseRefusal> {
    let t = Instant::now();
    let table = &shared.prepared.table;
    if generations.len() != table.sides.len() {
        return Err(CseRefusal::Generations {
            sides: table.sides.len(),
            given: generations.len(),
        });
    }
    table.check(generations).map_err(CseRefusal::Table)?;
    let version = table_version(generations);
    let prog = program(shared, version, sharing);

    let n = table.domain.len();
    let masks: Vec<&[u64]> = table.sides.iter().map(Side::present).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let filtered = planned.optimized.post_filter.iter().any(Option::is_some);
    let mask_only = kind == ResultKind::Exists && !filtered;
    let rules = prog.queries.len();
    let measure = if mask_only {
        Measure::Count
    } else {
        Measure::Keep
    };
    let mut kept: Vec<Vec<u64>> = if mask_only {
        Vec::new()
    } else {
        vec![vec![0u64; words_for(n)]; rules]
    };
    let mut sinks: Vec<&mut [u64]> = kept.iter_mut().map(Vec::as_mut_slice).collect();
    let run = run_program(&prog, version, &planes, &vec![measure; rules], &mut sinks)
        .map_err(CseRefusal::Program)?;

    let outcome = if mask_only {
        Outcome::Exists(run.answers.iter().any(|a| *a != Answer::Count(0)))
    } else {
        let mut bag: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
        for (j, star) in planned.stars.iter().enumerate() {
            let sides: Vec<&Side> = shared.rule_sides[j]
                .iter()
                .map(|&k| &table.sides[k])
                .collect();
            for (row, m) in tuples_over_mask(&table.domain, &sides, star, &kept[j], true) {
                *bag.entry(row).or_insert(0) += m;
            }
        }
        bag.retain(|row, _| {
            row.iter()
                .zip(&planned.optimized.post_filter)
                .all(|(v, f)| f.is_none_or(|f| f == *v))
        });
        match kind {
            ResultKind::Exists => Outcome::Exists(!bag.is_empty()),
            ResultKind::Count(Semantics::Set) => Outcome::Count(bag.len() as u64),
            ResultKind::Count(Semantics::Bag) => Outcome::Count(bag.values().sum()),
            ResultKind::Bindings(Semantics::Set) => {
                Outcome::Bindings(bag.into_keys().map(|r| (r, 1)).collect())
            }
            ResultKind::Bindings(Semantics::Bag) => Outcome::Bindings(bag.into_iter().collect()),
        }
    };
    let report = SharedReport {
        sides: table.sides.len(),
        domain: n,
        rows_prepared: shared.prepared.rows_prepared,
        lets: prog.lets.len(),
        cost: run.cost,
        execute: t.elapsed(),
        ..SharedReport::default()
    };
    Ok((outcome, report))
}
