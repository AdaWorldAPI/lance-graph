//! The stages wired together, with every intermediate product kept for the
//! report: rules, rewrites, MORK patterns, star plans, mask-risc programs.

use std::time::{Duration, Instant};

use lance_graph_mask_risc::Program;
use mork_pathmap_probe::Semantics;

use crate::addressed::{AddressedRelation, Lookup, ReadPolicy, prepare_addressed_with};
use crate::lance_store::{LanceStore, LanceStoreError, SideIo, prepare_lance};
use crate::logic::{RelId, Rule};
use crate::mork_api::{MorkRefusal, QueryMulti, admit, query_multi};
use crate::native::{
    DerivationView, DistinctGroup, NativeRefusal, PreparedStar, RuleCoordinates, RuleFilter,
    SetRefusal, StarPlan, StarView, Store, add_distinct, any_derivation, bag_count, canonical,
    for_each_derivation, joint, kept, matching, narrow_center, prepare, prepare_semijoin, star,
};
use crate::optimize::{Demand, Optimized, Options, ResultKind, optimize};
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
    /// A set answer no fold gives (see [`SetRefusal`]).
    Set {
        rule: usize,
        why: SetRefusal,
    },
}

/// The answers themselves, never as owned rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answers {
    /// Set semantics: per head template, the free column's distinct values
    /// as a mask over the value space. Canonical (see
    /// [`crate::native::canonical`]).
    Distinct(Vec<DistinctGroup>),
    /// Bag semantics: per rule, the kept slots. The rows stay in the
    /// prepared tables ([`Report::tables`]); read them with
    /// [`crate::native::for_each_slot`].
    Derivations(Vec<RuleCoordinates>),
}

impl Answers {
    /// Calls `f` with each rule's view over its table. The view holds
    /// references to the lanes, never values.
    fn rules(&self, tables: &[PreparedStar], mut f: impl FnMut(&StarView<'_>, &RuleCoordinates)) {
        let Answers::Derivations(rules) = self else {
            return;
        };
        for rc in rules {
            let table = &tables[rc.table].table;
            let sides: Vec<&Side> = rc.sides.iter().map(|&k| &table.sides[k]).collect();
            let view = StarView {
                domain: &table.domain,
                sides: &sides,
                star: &rc.star,
            };
            f(&view, rc);
        }
    }

    /// Rows answered: derivations under bag semantics, distinct rows under
    /// set semantics. A fold over the masks.
    pub fn total(&self, tables: &[PreparedStar]) -> u64 {
        match self {
            Answers::Distinct(groups) => groups.iter().map(DistinctGroup::len).sum(),
            Answers::Derivations(_) => {
                let mut n = 0;
                self.rules(tables, |v, rc| n += bag_count(v, &rc.kept, &rc.filter));
                n
            }
        }
    }

    /// Rows reading as `pattern` (`None`: any value), counted like
    /// [`Self::total`]. A fold; no row is built.
    pub fn matching(&self, tables: &[PreparedStar], pattern: impl Fn(usize) -> Option<u64>) -> u64 {
        match self {
            Answers::Distinct(groups) => groups
                .iter()
                .map(|g| {
                    let mut free = None;
                    for (c, t) in g.template.iter().enumerate() {
                        match (*t, pattern(c)) {
                            (Some(k), Some(v)) if k != v => return 0,
                            (None, p) => free = Some(p),
                            _ => {}
                        }
                    }
                    match free {
                        // No free column: the group answers one row or none.
                        None => g.len(),
                        Some(None) => g.len(),
                        Some(Some(v)) => {
                            let bit = usize::try_from(v).unwrap_or(usize::MAX);
                            u64::from(
                                g.values
                                    .get(bit / 64)
                                    .is_some_and(|w| w >> (bit % 64) & 1 == 1),
                            )
                        }
                    }
                })
                .sum(),
            Answers::Derivations(_) => {
                let mut n = 0;
                self.rules(tables, |v, rc| n += matching(v, rc, &pattern));
                n
            }
        }
    }

    /// Every answered row, read in place through its template, with its
    /// weight; `get(c)` reads column `c`. Rows are not deduplicated across
    /// slots or rules (that would need a seen-set).
    pub fn for_each(&self, tables: &[PreparedStar], mut f: impl FnMut(&dyn Fn(usize) -> u64, u64)) {
        match self {
            Answers::Distinct(groups) => {
                for g in groups {
                    for (w, &word) in g.values.iter().enumerate() {
                        let mut word = word;
                        while word != 0 {
                            let v = (w * 64 + word.trailing_zeros() as usize) as u64;
                            f(&|c| g.template[c].unwrap_or(v), 1);
                            word &= word - 1;
                        }
                    }
                }
            }
            Answers::Derivations(_) => self.rules(tables, |v, rc| {
                for_each_derivation(v, rc, |d: &DerivationView<'_>, w| f(&|c| d.get(c), w));
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Exists(bool),
    Count(u64),
    Bindings(Answers),
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
    pub costs: Costs,
    /// The prepared tables, in star order: what [`Answers::Derivations`]
    /// addresses.
    pub tables: Vec<PreparedStar>,
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
    store: &Store,
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
                .map(|side| store.view(side.key.rel).generation)
                .collect()
        })
        .collect();
    execute(planned, prepared, prepare_cost, kind, &generations)
}

/// The run half: the joint mask per prepared star, then the terminal.
/// `generations[j]` is the current generation of each side of star `j`.
pub fn execute(
    planned: Planned,
    prepared: Vec<PreparedStar>,
    prepare_cost: Duration,
    kind: ResultKind,
    generations: &[Vec<u64>],
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
    let outcome = if kind == ResultKind::Exists && !filtered {
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
    } else {
        let mut any = false;
        let mut derivations = 0u64;
        let mut groups: Vec<DistinctGroup> = Vec::new();
        let mut coordinates: Vec<RuleCoordinates> = Vec::new();
        for (j, ((i, _), (s, p))) in optimized
            .rules
            .iter()
            .zip(stars.iter().zip(&prepared))
            .enumerate()
        {
            // One program per rule: the Keep mask every terminal folds over.
            let (mut mask, run, prog) = kept(p, s, &generations[j]).map_err(|w| native(*i, w))?;
            programs_run += run.programs;
            programs.extend(prog);
            let Some(filter) = RuleFilter::of(s, &optimized.post_filter) else {
                continue;
            };
            narrow_center(&mut mask, &p.table.domain, &filter);
            let sides: Vec<_> = p.table.sides.iter().collect();
            let view = StarView {
                domain: &p.table.domain,
                sides: &sides,
                star: s,
            };
            match kind {
                ResultKind::Exists => any |= any_derivation(&view, &mask, &filter),
                ResultKind::Count(Semantics::Bag) => {
                    derivations += bag_count(&view, &mask, &filter)
                }
                ResultKind::Count(Semantics::Set) | ResultKind::Bindings(Semantics::Set) => {
                    add_distinct(&view, &mask, &filter, &mut groups)
                        .map_err(|why| PipelineRefusal::Set { rule: *i, why })?
                }
                ResultKind::Bindings(Semantics::Bag) => coordinates.push(RuleCoordinates {
                    rule: *i,
                    table: j,
                    sides: (0..s.sides.len()).collect(),
                    star: s.clone(),
                    filter,
                    kept: mask,
                }),
            }
        }
        canonical(&mut groups);
        match kind {
            ResultKind::Exists => Outcome::Exists(any),
            ResultKind::Count(Semantics::Bag) => Outcome::Count(derivations),
            ResultKind::Count(Semantics::Set) => {
                Outcome::Count(groups.iter().map(DistinctGroup::len).sum())
            }
            ResultKind::Bindings(Semantics::Set) => Outcome::Bindings(Answers::Distinct(groups)),
            ResultKind::Bindings(Semantics::Bag) => {
                Outcome::Bindings(Answers::Derivations(coordinates))
            }
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
        tables: prepared,
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
