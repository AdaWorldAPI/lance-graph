//! The stages wired together, with every intermediate product kept for the
//! report: rules, rewrites, MORK patterns, star plans, mask-risc programs.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use lance_graph_mask_risc::Program;
use mork_pathmap_probe::Semantics;

use crate::logic::{RelId, Rule};
use crate::mork_api::{MorkRefusal, QueryMulti, admit, query_multi};
use crate::native::{
    NativeRefusal, PreparedStar, StarPlan, Store, bindings_run, joint, prepare, star,
};
use crate::optimize::{Demand, Optimized, Options, ResultKind, optimize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineRefusal {
    Mork { rule: usize, why: MorkRefusal },
    Native { rule: usize, why: NativeRefusal },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Exists(bool),
    Count(u64),
    /// Sorted tuples with multiplicities (1 under set semantics).
    Bindings(Vec<(Vec<u64>, u64)>),
}

/// Costs, each measured on its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct Costs {
    pub optimize: Duration,
    pub map_and_admit: Duration,
    pub prepare: Duration,
    pub execute: Duration,
}

/// Everything one evaluation produced.
#[derive(Debug, Clone)]
pub struct Report {
    pub optimized: Optimized,
    pub patterns: Vec<QueryMulti>,
    pub stars: Vec<StarPlan>,
    pub programs: Vec<Program>,
    /// Rows laid out over all prepared tables.
    pub rows_prepared: usize,
    /// mask-risc programs run (0 for a rule relalg answered symbolically).
    pub programs_run: usize,
    pub costs: Costs,
}

/// Rewrites, maps, prepares and runs `rules` for `demand`.
pub fn evaluate(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &Store<'_>,
) -> Result<(Outcome, Report), PipelineRefusal> {
    let t = Instant::now();
    let optimized = optimize(rules, demand, kind, opts);
    let optimize_cost = t.elapsed();

    let t = Instant::now();
    let mut patterns = Vec::new();
    let mut stars = Vec::new();
    for (i, rule) in &optimized.rules {
        let q = query_multi(rule);
        admit(&q).map_err(|why| PipelineRefusal::Mork { rule: *i, why })?;
        stars.push(star(&q).map_err(|why| PipelineRefusal::Native { rule: *i, why })?);
        patterns.push(q);
    }
    let map_cost = t.elapsed();

    let t = Instant::now();
    let prepared: Vec<PreparedStar> = stars.iter().map(|s| prepare(s, store)).collect();
    let prepare_cost = t.elapsed();

    let t = Instant::now();
    let generations = |s: &StarPlan| -> Vec<u64> {
        s.sides
            .iter()
            .map(|side| store.view(side.key.rel).1)
            .collect()
    };
    let native = |rule: usize, why| PipelineRefusal::Native { rule, why };
    let mut programs = Vec::new();
    let mut programs_run = 0;
    let filtered = optimized.post_filter.iter().any(Option::is_some);
    let outcome = if kind == ResultKind::Exists && !filtered {
        // Mask-only terminal: no payload is read.
        let mut any = false;
        for ((i, _), (s, p)) in optimized.rules.iter().zip(stars.iter().zip(&prepared)) {
            let (run, prog) = joint(
                p,
                s,
                &generations(s),
                lance_graph_relalg::Measure::Count,
                None,
            )
            .map_err(|w| native(*i, w))?;
            programs_run += run.programs;
            programs.extend(prog);
            any |= run.answer != lance_graph_relalg::Answer::Count(0);
        }
        Outcome::Exists(any)
    } else {
        let mut bag: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
        for ((i, _), (s, p)) in optimized.rules.iter().zip(stars.iter().zip(&prepared)) {
            // One program per rule: the Keep mask that the bindings are read
            // from; no separate count pass.
            let (rows, run, prog) =
                bindings_run(p, s, &generations(s), true).map_err(|w| native(*i, w))?;
            programs_run += run.programs;
            programs.extend(prog);
            for (row, m) in rows {
                *bag.entry(row).or_insert(0) += m;
            }
        }
        bag.retain(|row, _| {
            row.iter()
                .zip(&optimized.post_filter)
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
    let execute_cost = t.elapsed();

    let report = Report {
        rows_prepared: prepared.iter().map(|p| p.rows_prepared).sum(),
        programs_run,
        optimized,
        patterns,
        stars,
        programs,
        costs: Costs {
            optimize: optimize_cost,
            map_and_admit: map_cost,
            prepare: prepare_cost,
            execute: execute_cost,
        },
    };
    Ok((outcome, report))
}

/// The goal's relation id, for building a [`Demand`].
pub fn demand(goal: RelId, bound: Vec<Option<u64>>) -> Demand {
    Demand { goal, bound }
}
