//! The stages wired together, with every intermediate product kept for the
//! report: rules, rewrites, MORK patterns, star plans, mask-risc programs.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use lance_graph_mask_risc::Program;
use mork_pathmap_probe::Semantics;

use crate::addressed::{AddressedRelation, Lookup, ReadPolicy, prepare_addressed_with};
use crate::lance_store::{LanceStore, LanceStoreError, SideIo, prepare_lance};
use crate::logic::{RelId, Rule};
use crate::mork_api::{MorkRefusal, QueryMulti, admit, query_multi};
use crate::native::{
    FoldKind, NativeRefusal, PreparedStar, StarPlan, Store, TerminalPath, bindings_run,
    count_over_mask, fold_kind, joint, prepare, prepare_semijoin, star,
};
use crate::optimize::{Demand, Optimized, Options, ResultKind, optimize};
use lance_graph_mask_risc::words_for;
use mork_pathmap_probe::join::Side;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineRefusal {
    Mork {
        rule: usize,
        why: MorkRefusal,
    },
    Native {
        rule: usize,
        why: NativeRefusal,
    },
    /// A Lance read or write failed (stage 3 only).
    Lance(String),
    /// The addressed arm refused a side (Arm B only): stale lane, unsorted
    /// table, a demand on the second column, or no attested lane.
    Addressed(String),
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

/// The output of [`plan`]: what is evaluated, before any row is read.
#[derive(Debug, Clone)]
pub struct Planned {
    pub optimized: Optimized,
    pub patterns: Vec<QueryMulti>,
    pub stars: Vec<StarPlan>,
    pub optimize_cost: Duration,
    pub map_cost: Duration,
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
    /// Which terminal answered.
    pub terminal: TerminalPath,
    pub costs: Costs,
}

/// The cold half: rewrites and maps `rules` for `demand`; reads no row.
pub fn plan(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
) -> Result<Planned, PipelineRefusal> {
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

    Ok(Planned {
        optimized,
        patterns,
        stars,
        optimize_cost,
        map_cost,
    })
}

/// Rewrites, maps, prepares and runs `rules` for `demand`.
pub fn evaluate(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &Store<'_>,
) -> Result<(Outcome, Report), PipelineRefusal> {
    let planned = plan(rules, demand, kind, opts)?;
    let t = Instant::now();
    let prepared: Vec<PreparedStar> = planned
        .stars
        .iter()
        .map(|s| {
            if opts.semijoin {
                prepare_semijoin(s, store)
            } else {
                prepare(s, store)
            }
        })
        .collect();
    let prepare_cost = t.elapsed();

    let generations: Vec<Vec<u64>> = planned
        .stars
        .iter()
        .map(|s| {
            s.sides
                .iter()
                .map(|side| store.view(side.key.rel).1)
                .collect()
        })
        .collect();
    execute(planned, prepared, prepare_cost, kind, &generations)
}

/// The run half: the joint mask per prepared star, then the terminal.
/// `generations[j]` is the current generation of each side of star `j`.
/// A count [`fold_kind`] admits is answered by [`count_over_mask`] without
/// building a row; everything else enumerates.
pub fn execute(
    planned: Planned,
    prepared: Vec<PreparedStar>,
    prepare_cost: Duration,
    kind: ResultKind,
    generations: &[Vec<u64>],
) -> Result<(Outcome, Report), PipelineRefusal> {
    execute_with(planned, prepared, prepare_cost, kind, generations, true)
}

/// [`execute`]; `fold = false` forces the enumerating terminal (the oracle the
/// fold is tested against).
pub fn execute_with(
    planned: Planned,
    prepared: Vec<PreparedStar>,
    prepare_cost: Duration,
    kind: ResultKind,
    generations: &[Vec<u64>],
    fold: bool,
) -> Result<(Outcome, Report), PipelineRefusal> {
    let Planned {
        optimized,
        patterns,
        stars,
        optimize_cost,
        map_cost,
    } = planned;
    let t = Instant::now();
    let native = |rule: usize, why| PipelineRefusal::Native { rule, why };
    let mut programs = Vec::new();
    let mut programs_run = 0;
    let filtered = optimized.post_filter.iter().any(Option::is_some);
    let folding = if fold {
        fold_kind(&stars, kind, filtered)
    } else {
        None
    };
    let mut terminal = TerminalPath::Enumerated;
    let outcome = if kind == ResultKind::Exists && !filtered {
        terminal = TerminalPath::Mask;
        // Mask-only terminal: no payload is read.
        let mut any = false;
        for (j, ((i, _), (s, p))) in optimized
            .rules
            .iter()
            .zip(stars.iter().zip(&prepared))
            .enumerate()
        {
            let (run, prog) = joint(
                p,
                s,
                &generations[j],
                lance_graph_relalg::Measure::Count,
                None,
            )
            .map_err(|w| native(*i, w))?;
            programs_run += run.programs;
            programs.extend(prog);
            any |= run.answer != lance_graph_relalg::Answer::Count(0);
        }
        Outcome::Exists(any)
    } else if let Some(fk) = folding {
        terminal = TerminalPath::Folded;
        let mut total = 0u64;
        for (j, ((i, _), (s, p))) in optimized
            .rules
            .iter()
            .zip(stars.iter().zip(&prepared))
            .enumerate()
        {
            let mut mask = vec![0u64; words_for(p.table.domain.len())];
            let (run, prog) = joint(
                p,
                s,
                &generations[j],
                lance_graph_relalg::Measure::Keep,
                Some(&mut mask),
            )
            .map_err(|w| native(*i, w))?;
            programs_run += run.programs;
            programs.extend(prog);
            let sides: Vec<&Side> = p.table.sides.iter().collect();
            let c = count_over_mask(
                &p.table.domain,
                &sides,
                s,
                &mask,
                &optimized.post_filter,
                fk,
            )
            .map_err(|w| native(*i, w))?;
            total = total
                .checked_add(c)
                .ok_or_else(|| native(*i, NativeRefusal::CountOverflow))?;
            if fk == FoldKind::Exists && total > 0 {
                break;
            }
        }
        match fk {
            FoldKind::Exists => Outcome::Exists(total > 0),
            FoldKind::Bag | FoldKind::Set => Outcome::Count(total),
        }
    } else {
        let mut bag: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
        for (j, ((i, _), (s, p))) in optimized
            .rules
            .iter()
            .zip(stars.iter().zip(&prepared))
            .enumerate()
        {
            // One program per rule: the Keep mask that the bindings are read
            // from; no separate count pass.
            let (rows, run, prog) =
                bindings_run(p, s, &generations[j], true).map_err(|w| native(*i, w))?;
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
        terminal,
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

/// [`evaluate`] over Lance tables: each side is read under its address mask
/// (stage 3, `lance_store::prepare_lance`), and a side's generation is its
/// table's version at execution. Also returns what each side read.
pub async fn evaluate_lance(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &LanceStore,
) -> Result<(Outcome, Report, Vec<Vec<SideIo>>), PipelineRefusal> {
    let planned = plan(rules, demand, kind, opts)?;
    let lance = |e: LanceStoreError| PipelineRefusal::Lance(format!("{e:?}"));
    let t = Instant::now();
    let mut prepared = Vec::with_capacity(planned.stars.len());
    let mut io = Vec::with_capacity(planned.stars.len());
    for s in &planned.stars {
        let (p, side_io) = prepare_lance(s, store, opts.semijoin)
            .await
            .map_err(lance)?;
        prepared.push(p);
        io.push(side_io);
    }
    let prepare_cost = t.elapsed();
    let mut generations = Vec::with_capacity(planned.stars.len());
    for s in &planned.stars {
        generations.push(store.generations(s).await.map_err(lance)?);
    }
    let (outcome, report) = execute(planned, prepared, prepare_cost, kind, &generations)?;
    Ok((outcome, report, io))
}

/// [`evaluate`] through the addressed arm (Arm B): each side is read only at
/// the lane ranges of its demanded keys, located on the attested
/// `SealedFacetLane`s in `lanes`. Also returns what each side read and the
/// lookups each side made.
pub async fn evaluate_addressed(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &LanceStore,
    lanes: &[AddressedRelation],
) -> Result<(Outcome, Report, Vec<Vec<SideIo>>, Vec<Vec<Lookup>>), PipelineRefusal> {
    evaluate_addressed_with(rules, demand, kind, opts, store, lanes, ReadPolicy::EXACT).await
}

/// [`evaluate_addressed`] with an explicit [`ReadPolicy`] for every side.
pub async fn evaluate_addressed_with(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &LanceStore,
    lanes: &[AddressedRelation],
    policy: ReadPolicy,
) -> Result<(Outcome, Report, Vec<Vec<SideIo>>, Vec<Vec<Lookup>>), PipelineRefusal> {
    let planned = plan(rules, demand, kind, opts)?;
    let lance = |e: LanceStoreError| PipelineRefusal::Lance(format!("{e:?}"));
    let t = Instant::now();
    let mut prepared = Vec::with_capacity(planned.stars.len());
    let mut io = Vec::with_capacity(planned.stars.len());
    let mut lookups = Vec::with_capacity(planned.stars.len());
    for s in &planned.stars {
        let (p, side_io, l) = prepare_addressed_with(s, store, lanes, policy)
            .await
            .map_err(|e| PipelineRefusal::Addressed(format!("{e:?}")))?;
        prepared.push(p);
        io.push(side_io);
        lookups.push(l);
    }
    let prepare_cost = t.elapsed();
    let mut generations = Vec::with_capacity(planned.stars.len());
    for s in &planned.stars {
        generations.push(store.generations(s).await.map_err(lance)?);
    }
    let (outcome, report) = execute(planned, prepared, prepare_cost, kind, &generations)?;
    Ok((outcome, report, io, lookups))
}

/// The goal's relation id, for building a [`Demand`].
pub fn demand(goal: RelId, bound: Vec<Option<u64>>) -> Demand {
    Demand { goal, bound }
}
