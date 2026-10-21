//! Cross-rule CSE (D-RU-3 / U3): rules that share a side read one table, and
//! the parts several rules share are evaluated once.
//!
//! STATUS: PROBE (results only), like the rest of this crate.
//!
//! [`crate::pipeline::evaluate`] prepares one table per rule and runs one
//! fold per rule, so a core that 256 rules share is laid out and ANDed 256
//! times. Here:
//!
//! 1. the stars are grouped ([`Layout`]). With [`Layout::Components`], stars
//!    that share a side ([`SideKey`]), directly or through other stars, form
//!    one group; a star that shares nothing is a group of its own;
//! 2. each group's distinct sides are laid out once, over one domain: the
//!    union of their centers. A rule's joint mask is the AND of its sides'
//!    presence planes, and a center a side lacks is absent from that side's
//!    plane, so the wider domain changes no rule's mask, only its width;
//! 3. each rule becomes `And(Sym(side), …)` over its group's planes, and
//!    [`share`] lifts every part two or more of the group's rules hold into
//!    a let (`lance_graph_contract::set_program`);
//! 4. [`run_program`] evaluates each let once into a plane and every rule
//!    over the sides plus those planes. The tuples are read per rule with
//!    [`tuples_over_mask`], exactly as the per-rule path reads them.
//!
//! Width is the price. A side is laid out densely over its table's domain,
//! so a table pays per slot for every side it holds. [`Layout::OneTable`]
//! puts every side over the union of all centers. Measured in
//! `examples/rule_cse.rs`, that is slower than the per-rule path when the
//! rules share nothing. [`Layout::Components`] pays it only within a group
//! that shares something.
//!
//! With `sharing` off the same tables are used and no let is formed, so the
//! two effects (fewer tables, shared folds) are measured apart.
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

/// How the stars are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// One table for all stars, over the union of all their centers.
    OneTable,
    /// One table per group of stars that share a side, directly or through
    /// other stars; a star that shares nothing keeps a table of its own, as
    /// in the per-rule path.
    Components,
}

/// Stars laid out as tables, each holding the distinct sides of its stars
/// over one domain.
pub struct SharedStars {
    pub groups: Vec<Group>,
}

/// One table and the stars that read it.
pub struct Group {
    /// The table; its side `k` holds `prepared.keys[k]`.
    pub prepared: PreparedStar,
    /// The stars of this group, as indices into the planned stars.
    pub stars: Vec<usize>,
    /// For `stars[i]`, the table side of each of its plan sides.
    pub rule_sides: Vec<Vec<usize>>,
}

/// Star indices grouped by `layout`, each group in star order.
fn partition(stars: &[StarPlan], layout: Layout) -> Vec<Vec<usize>> {
    if layout == Layout::OneTable || stars.is_empty() {
        return vec![(0..stars.len()).collect()];
    }
    // Union-find over stars, joined through each side key's first holder.
    let mut parent: Vec<usize> = (0..stars.len()).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut holder: BTreeMap<(u16, u8, Option<u64>, bool), usize> = BTreeMap::new();
    for (j, s) in stars.iter().enumerate() {
        for side in &s.sides {
            let k = side.key;
            let key = (k.rel.0, k.key_col, k.restrict, k.unary);
            match holder.get(&key) {
                Some(&h) => {
                    let (a, b) = (root(&mut parent, h), root(&mut parent, j));
                    parent[a.max(b)] = a.min(b);
                }
                None => {
                    holder.insert(key, j);
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for j in 0..stars.len() {
        let r = root(&mut parent, j);
        groups.entry(r).or_default().push(j);
    }
    groups.into_values().collect()
}

/// Lays out the distinct sides of each group's stars over the union of that
/// group's centers.
pub fn prepare_shared(stars: &[StarPlan], store: &Store<'_>, layout: Layout) -> SharedStars {
    let groups = partition(stars, layout)
        .into_iter()
        .map(|members| {
            let mut keys: Vec<SideKey> = Vec::new();
            let rule_sides = members
                .iter()
                .map(|&j| {
                    stars[j]
                        .sides
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
            // A star whose sides are the distinct keys: `prepare` lays them
            // out over one domain with the same per-side restriction as the
            // per-rule path.
            let all = StarPlan {
                center: VarId(0),
                sides: keys
                    .iter()
                    .map(|&key| SidePlan { key, emit: None })
                    .collect(),
                head: Vec::new(),
            };
            Group {
                prepared: prepare(&all, store),
                stars: members,
                rule_sides,
            }
        })
        .collect();
    SharedStars { groups }
}

/// Each star of the group as the conjunction of its table sides, as set
/// expressions over `Sym(side)`.
pub fn rule_sets(group: &Group) -> Vec<SetExpr> {
    group
        .rule_sides
        .iter()
        .map(|sides| SetExpr::And(sides.iter().map(|&k| SetExpr::Sym(k as u32)).collect()))
        .collect()
}

/// The current generation of every side of every group, in table order.
pub fn generations(shared: &SharedStars, store: &Store<'_>) -> Vec<Vec<u64>> {
    shared
        .groups
        .iter()
        .map(|g| {
            g.prepared
                .keys
                .iter()
                .map(|k| store.view(k.rel).1)
                .collect()
        })
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
    /// The caller passed another number of generation vectors than there
    /// are groups, or of generations than a group's table has sides.
    Generations {
        sides: usize,
        given: usize,
    },
    Program(ProgramError),
    /// The tables were prepared for other stars than the plan holds: a star
    /// in no group or in two, or a side the table holds under another key.
    NoCoordinateWitness,
}

/// What one shared evaluation cost, each part on its own, summed over the
/// groups.
#[derive(Debug, Clone, Copy, Default)]
pub struct SharedReport {
    /// Tables laid out.
    pub groups: usize,
    /// Sides laid out (distinct within each group).
    pub sides: usize,
    /// Slots over all tables' domains: the width every fold of a group runs
    /// at, summed.
    pub domain: usize,
    /// Rows laid out over all sides.
    pub rows_prepared: usize,
    /// Lets formed.
    pub lets: usize,
    /// Folds, ops, planes read and materialized.
    pub cost: ProgramCost,
    pub plan: Duration,
    pub prepare: Duration,
    /// Building the programs: finding the shared parts (compile time).
    pub share: Duration,
    /// The folds: lets and rules.
    pub fold: Duration,
    /// Reading the tuples over the kept masks.
    pub tuples: Duration,
    /// All of the run half: checks, share, folds, tuples.
    pub execute: Duration,
}

/// The program a group runs: its rules over the table's sides, with the
/// shared parts as lets when `sharing` is on.
pub fn program(group: &Group, version: u64, sharing: bool) -> SetProgram {
    let resident = group.prepared.table.sides.len() as u32;
    let queries = rule_sets(group);
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

/// [`crate::pipeline::evaluate`] over shared tables laid out by `layout`.
/// Same rewrites, same answer; `sharing` decides whether the parts a
/// group's rules share are evaluated once.
pub fn evaluate_shared(
    rules: &[Rule],
    demand: &Demand,
    kind: ResultKind,
    opts: Options,
    store: &Store<'_>,
    layout: Layout,
    sharing: bool,
) -> Result<(Outcome, SharedReport), CseRefusal> {
    let t = Instant::now();
    let planned = plan(rules, demand, kind, opts).map_err(CseRefusal::Pipeline)?;
    let plan_cost = t.elapsed();
    let t = Instant::now();
    let shared = prepare_shared(&planned.stars, store, layout);
    let prepare_cost = t.elapsed();
    let gens = generations(&shared, store);
    let (outcome, mut report) = execute_shared(&planned, &shared, kind, &gens, sharing)?;
    report.plan = plan_cost;
    report.prepare = prepare_cost;
    Ok((outcome, report))
}

/// The coordinate witness: every planned star is in exactly one group, and
/// each of its sides maps to a table side holding exactly that side's key.
/// Tables prepared for another plan are refused before any plane is read.
pub fn witness(planned: &Planned, shared: &SharedStars) -> Result<(), CseRefusal> {
    let mut seen = vec![false; planned.stars.len()];
    for g in &shared.groups {
        if g.stars.len() != g.rule_sides.len() {
            return Err(CseRefusal::NoCoordinateWitness);
        }
        for (&j, sides) in g.stars.iter().zip(&g.rule_sides) {
            let Some(star) = planned.stars.get(j) else {
                return Err(CseRefusal::NoCoordinateWitness);
            };
            if seen[j] || star.sides.len() != sides.len() {
                return Err(CseRefusal::NoCoordinateWitness);
            }
            seen[j] = true;
            for (side, &k) in star.sides.iter().zip(sides) {
                if g.prepared.keys.get(k) != Some(&side.key) {
                    return Err(CseRefusal::NoCoordinateWitness);
                }
            }
        }
    }
    if seen.iter().all(|&b| b) {
        Ok(())
    } else {
        Err(CseRefusal::NoCoordinateWitness)
    }
}

/// The run half over prepared shared tables. `generations[g][k]` is the
/// current generation of side `k` of group `g`.
pub fn execute_shared(
    planned: &Planned,
    shared: &SharedStars,
    kind: ResultKind,
    generations: &[Vec<u64>],
    sharing: bool,
) -> Result<(Outcome, SharedReport), CseRefusal> {
    let t = Instant::now();
    witness(planned, shared)?;
    if generations.len() != shared.groups.len() {
        return Err(CseRefusal::Generations {
            sides: shared.groups.len(),
            given: generations.len(),
        });
    }
    let filtered = planned.optimized.post_filter.iter().any(Option::is_some);
    let mask_only = kind == ResultKind::Exists && !filtered;
    let measure = if mask_only {
        Measure::Count
    } else {
        Measure::Keep
    };
    let mut report = SharedReport {
        groups: shared.groups.len(),
        ..SharedReport::default()
    };
    let mut any = false;
    let mut bag: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
    for (group, gens) in shared.groups.iter().zip(generations) {
        let table = &group.prepared.table;
        if gens.len() != table.sides.len() {
            return Err(CseRefusal::Generations {
                sides: table.sides.len(),
                given: gens.len(),
            });
        }
        table.check(gens).map_err(CseRefusal::Table)?;
        let version = table_version(gens);
        let ts = Instant::now();
        let prog = program(group, version, sharing);
        report.share += ts.elapsed();

        let n = table.domain.len();
        let masks: Vec<&[u64]> = table.sides.iter().map(Side::present).collect();
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let rules = prog.queries.len();
        let mut kept: Vec<Vec<u64>> = if mask_only {
            Vec::new()
        } else {
            vec![vec![0u64; words_for(n)]; rules]
        };
        let mut sinks: Vec<&mut [u64]> = kept.iter_mut().map(Vec::as_mut_slice).collect();
        let tf = Instant::now();
        let run = run_program(&prog, version, &planes, &vec![measure; rules], &mut sinks)
            .map_err(CseRefusal::Program)?;
        report.fold += tf.elapsed();

        let tt = Instant::now();
        if mask_only {
            any |= run.answers.iter().any(|a| *a != Answer::Count(0));
        } else {
            for (i, &j) in group.stars.iter().enumerate() {
                let sides: Vec<&Side> = group.rule_sides[i]
                    .iter()
                    .map(|&k| &table.sides[k])
                    .collect();
                let star = &planned.stars[j];
                for (row, m) in tuples_over_mask(&table.domain, &sides, star, &kept[i], true) {
                    *bag.entry(row).or_insert(0) += m;
                }
            }
        }
        report.tuples += tt.elapsed();
        report.sides += table.sides.len();
        report.domain += n;
        report.rows_prepared += group.prepared.rows_prepared;
        report.lets += prog.lets.len();
        let c = run.cost;
        report.cost.programs += c.programs;
        report.cost.passes += c.passes;
        report.cost.lane_reads += c.lane_reads;
        report.cost.plane_reads += c.plane_reads;
        report.cost.planes_materialized += c.planes_materialized;
        report.cost.plane_bytes += c.plane_bytes;
        report.cost.max_scratch_slots = report.cost.max_scratch_slots.max(c.max_scratch_slots);
    }
    let tt = Instant::now();
    let outcome = if mask_only {
        Outcome::Exists(any)
    } else {
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
    report.tuples += tt.elapsed();
    report.execute = t.elapsed();
    Ok((outcome, report))
}
