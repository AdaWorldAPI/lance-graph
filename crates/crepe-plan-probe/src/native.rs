//! Native lowering of an admitted `query_multi` star onto the D-MPS-8
//! versioned relationship table, through relalg into one mask-risc program.
//!
//! STATUS: PROBE. The table (`mork_pathmap_probe::join`) is measured but not
//! admitted (D-RMAP-1's binding membrane waits on EXO-46). Nothing here is a
//! production path.
//!
//! # The only shape lowered
//!
//! A star on one shared variable `c`: every pattern names `c`, and every other
//! variable occurs in exactly one pattern. Each pattern becomes one *side*, laid
//! out by the slot of `c` over the shared domain. A constant on the side's other
//! column restricts the rows the side is prepared from (cold, at preparation,
//! once per prepared table). Then
//!
//! ```text
//! joint = present_0 ∧ present_1 ∧ … ∧ present_k        (relalg: And of Plane(i))
//! ```
//!
//! is the set of `c` values every side has. That mask is all relalg computes.
//! Existence and distinct-`c` counts read only it; tuple emission and bag
//! counts are a separate terminal that reads each side at the slot the mask
//! names, never at a position taken from another side's values.
//!
//! Anything else (a chain, a pattern of arity > 2, a side without `c`) is
//! refused by name.

use lance_graph_mask_risc::{MaskOp, Planes, Pred, Program, Terminal, words_for};
use lance_graph_relalg::{Answer, Measure, Query, Rel, RelError, Run, run_with_program};
use mork_pathmap_probe::RelationView;
use mork_pathmap_probe::join::{JoinDomain, JoinRefusal, JoinTable, Side, for_each_set};

use crate::logic::{RelId, VarId};
use crate::mork_api::{Expr, QueryMulti};
use crate::optimize::ResultKind;
use mork_pathmap_probe::Semantics;

/// Which rows of which relation a side holds, and how they are keyed. Two
/// sides are interchangeable only when their keys are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SideKey {
    pub rel: RelId,
    /// The column holding the shared variable (`0` for a unary relation).
    pub key_col: u8,
    /// `Some(k)`: only rows whose other column is `k`.
    pub restrict: Option<u64>,
    pub unary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidePlan {
    pub key: SideKey,
    /// The variable this side's other column binds. Every side whose other
    /// column is a variable emits, whether or not the template uses it; the
    /// head ([`HeadTerm::Side`]) decides which emitted values reach a row.
    pub emit: Option<VarId>,
}

/// Where an output column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadTerm {
    Const(u64),
    /// The shared variable.
    Center,
    /// The value column of side `i`.
    Side(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarPlan {
    pub center: VarId,
    pub sides: Vec<SidePlan>,
    pub head: Vec<HeadTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeRefusal {
    /// No variable occurs in every pattern.
    NoSharedVariable,
    /// A variable other than the center occurs in two patterns (a chain or a
    /// second join variable).
    NotAStar {
        var: VarId,
    },
    /// The D-MPS-8 sides are binary or unary.
    Arity {
        pattern: usize,
        arity: usize,
    },
    /// A template variable no pattern binds.
    UnboundTemplateVar(VarId),
    /// The prepared table was built for other sides than this plan reads.
    NoCoordinateWitness,
    /// A side was laid out over another domain than the table's.
    DomainMismatch {
        side: usize,
    },
    /// The caller passed a generation vector of another length than the
    /// table has sides; `JoinTable::check` would skip the missing ones.
    Generations {
        sides: usize,
        given: usize,
    },
    /// From the table: a stale generation, or rows outside the domain.
    Table(JoinRefusal),
    Relalg(RelError),
    /// A folded count does not fit in `u64`. Returned only when the exact
    /// answer is not representable.
    CountOverflow,
}

/// Reads the star out of an admitted pattern.
pub fn star(q: &QueryMulti) -> Result<StarPlan, NativeRefusal> {
    let args: Vec<&[Expr]> = q.patterns.iter().map(|p| &p[1..]).collect();
    let rel = |i: usize| match q.patterns[i][0] {
        Expr::Sym(r) => r,
        _ => unreachable!("admitted patterns start with a symbol"),
    };
    let vars_of = |a: &[Expr]| -> Vec<VarId> {
        a.iter()
            .filter_map(|e| match e {
                Expr::Var(v) => Some(*v),
                _ => None,
            })
            .collect()
    };
    let first = vars_of(args.first().copied().unwrap_or(&[]));
    let center = first
        .into_iter()
        .find(|v| args.iter().all(|a| vars_of(a).contains(v)))
        .ok_or(NativeRefusal::NoSharedVariable)?;
    let mut sides = Vec::new();
    for (i, a) in args.iter().enumerate() {
        let key_col = a
            .iter()
            .position(|e| *e == Expr::Var(center))
            .expect("center") as u8;
        let (restrict, emit, unary) = match a.len() {
            1 => (None, None, true),
            2 => match &a[1 - key_col as usize] {
                Expr::Int(k) => (Some(*k), None, false),
                Expr::Var(v) => (None, Some(*v), false),
                _ => unreachable!("admitted patterns are flat"),
            },
            n => {
                return Err(NativeRefusal::Arity {
                    pattern: i,
                    arity: n,
                });
            }
        };
        sides.push(SidePlan {
            key: SideKey {
                rel: rel(i),
                key_col,
                restrict,
                unary,
            },
            emit,
        });
    }
    for (i, s) in sides.iter().enumerate() {
        if let Some(v) = s.emit
            && sides
                .iter()
                .enumerate()
                .any(|(j, t)| j != i && t.emit == Some(v))
        {
            return Err(NativeRefusal::NotAStar { var: v });
        }
    }
    let head = q.template[1..]
        .iter()
        .map(|e| match *e {
            Expr::Int(c) => Ok(HeadTerm::Const(c)),
            Expr::Var(v) if v == center => Ok(HeadTerm::Center),
            Expr::Var(v) => sides
                .iter()
                .position(|s| s.emit == Some(v))
                .map(HeadTerm::Side)
                .ok_or(NativeRefusal::UnboundTemplateVar(v)),
            _ => unreachable!("templates are flat"),
        })
        .collect::<Result<_, _>>()?;
    Ok(StarPlan {
        center,
        sides,
        head,
    })
}

/// The relations a plan reads, by id, with their generations.
pub struct Store<'a> {
    /// `(relation, rows, generation)`. A unary relation's rows are `[v, v]`.
    pub relations: Vec<(RelId, &'a [[u32; 2]], u64)>,
}

impl<'a> Store<'a> {
    pub fn view(&self, rel: RelId) -> (&'a [[u32; 2]], u64) {
        let (_, rows, g) = self
            .relations
            .iter()
            .find(|(r, _, _)| *r == rel)
            .expect("relation in store");
        (rows, *g)
    }
}

/// A relationship table prepared for one plan's sides, with the witness that
/// says which sides those are and which domain they share.
pub struct PreparedStar {
    pub table: JoinTable,
    pub keys: Vec<SideKey>,
    /// Fingerprint of the domain every side is laid out over.
    pub domain_fp: u64,
    /// Rows laid out, over all sides (after restriction).
    pub rows_prepared: usize,
}

/// Lays the plan's sides out over one shared domain. A side's restriction is
/// applied to the rows it is prepared from.
pub fn prepare(plan: &StarPlan, store: &Store<'_>) -> PreparedStar {
    prepare_with(plan, store, false)
}

/// [`prepare`] with semijoin reduction: when at least one side is restricted
/// by a constant, every side keeps only the rows whose center the restricted
/// sides all have. A center outside that set cannot reach the joint mask, so
/// the answer and its multiplicities are unchanged; the domain shrinks to the
/// reduced centers. Without a restricted side nothing changes.
pub fn prepare_semijoin(plan: &StarPlan, store: &Store<'_>) -> PreparedStar {
    prepare_with(plan, store, true)
}

fn prepare_with(plan: &StarPlan, store: &Store<'_>, semijoin: bool) -> PreparedStar {
    let mut owned: Vec<(Vec<[u32; 2]>, u64)> = plan
        .sides
        .iter()
        .map(|s| {
            let (rows, g) = store.view(s.key.rel);
            let other = 1 - s.key.key_col as usize;
            let kept = rows
                .iter()
                .filter(|r| s.key.unary || s.key.restrict.is_none_or(|k| u64::from(r[other]) == k))
                .copied()
                .collect();
            (kept, g)
        })
        .collect();
    if semijoin {
        // Centers every restricted side has: sorted, deduplicated, intersected.
        let mut centers: Option<Vec<u32>> = None;
        for (s, (rows, _)) in plan.sides.iter().zip(&owned) {
            if s.key.unary || s.key.restrict.is_none() {
                continue;
            }
            let mut c: Vec<u32> = rows.iter().map(|r| r[s.key.key_col as usize]).collect();
            c.sort_unstable();
            c.dedup();
            centers = Some(match centers {
                None => c,
                Some(prev) => prev
                    .into_iter()
                    .filter(|v| c.binary_search(v).is_ok())
                    .collect(),
            });
        }
        if let Some(centers) = centers {
            for (s, (rows, _)) in plan.sides.iter().zip(owned.iter_mut()) {
                let col = s.key.key_col as usize;
                rows.retain(|r| centers.binary_search(&r[col]).is_ok());
            }
        }
    }
    prepare_from_rows(plan, owned)
}

/// Lays out rows already selected per side (`(rows, generation)`, in plan
/// side order) over one shared domain.
pub fn prepare_from_rows(plan: &StarPlan, owned: Vec<(Vec<[u32; 2]>, u64)>) -> PreparedStar {
    let inputs: Vec<(RelationView<'_>, usize)> = owned
        .iter()
        .zip(&plan.sides)
        .map(|((rows, g), s)| {
            (
                RelationView {
                    rows,
                    generation: *g,
                },
                s.key.key_col as usize,
            )
        })
        .collect();
    let table = JoinTable::prepare_star(&inputs);
    PreparedStar {
        domain_fp: fingerprint(&table),
        keys: plan.sides.iter().map(|s| s.key).collect(),
        rows_prepared: owned.iter().map(|(r, _)| r.len()).sum(),
        table,
    }
}

/// FNV-1a over the domain's values: equal domains, equal fingerprints.
pub fn fingerprint(table: &JoinTable) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for slot in 0..table.domain.len() {
        for b in table.domain.id(slot).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h ^ table.domain.len() as u64
}

impl PreparedStar {
    /// Replaces side `i` with one prepared elsewhere. Refused unless that side
    /// was laid out over this table's domain: equal slot numbers over two
    /// different domains name different values.
    pub fn graft(
        &mut self,
        i: usize,
        side: Side,
        key: SideKey,
        side_domain_fp: u64,
    ) -> Result<(), NativeRefusal> {
        if side_domain_fp != self.domain_fp {
            return Err(NativeRefusal::DomainMismatch { side: i });
        }
        self.table.sides[i] = side;
        self.keys[i] = key;
        Ok(())
    }

    /// The coordinate witness: this table holds exactly the plan's sides.
    pub fn witness(&self, plan: &StarPlan) -> Result<(), NativeRefusal> {
        let wanted: Vec<SideKey> = plan.sides.iter().map(|s| s.key).collect();
        if wanted != self.keys {
            return Err(NativeRefusal::NoCoordinateWitness);
        }
        Ok(())
    }
}

/// What the native path answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeAnswer {
    Exists(bool),
    Count(u64),
    /// Output tuples with their multiplicities (all 1 under set semantics).
    Bindings(Vec<(Vec<u64>, u64)>),
}

/// The joint mask and its relalg run, after the witness and generation checks.
pub fn joint(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    measure: Measure,
    keep: Option<&mut [u64]>,
) -> Result<(Run, Option<Program>), NativeRefusal> {
    prep.witness(plan)?;
    if current.len() != prep.table.sides.len() {
        return Err(NativeRefusal::Generations {
            sides: prep.table.sides.len(),
            given: current.len(),
        });
    }
    prep.table.check(current).map_err(NativeRefusal::Table)?;
    let masks: Vec<&[u64]> = prep.table.sides.iter().map(|s| s.present()).collect();
    let planes = Planes {
        n_rows: prep.table.domain.len(),
        masks: &masks,
        lanes: &[],
    };
    let rel = Rel::And((0..masks.len() as u16).map(Rel::Plane).collect());
    run_with_program(&Query { rel, measure }, &planes, keep).map_err(NativeRefusal::Relalg)
}

/// Number of distinct center values every side has: relalg's popcount of the
/// joint mask, no payload read.
pub fn distinct_centers(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
) -> Result<u64, NativeRefusal> {
    match joint(prep, plan, current, Measure::Count, None)?.0.answer {
        Answer::Count(n) => Ok(n),
        other => unreachable!("Count measure answered {other:?}"),
    }
}

/// Tuples over the joint mask. Reads each side at the slot the mask names.
/// A side whose value the template does not use still multiplies the count of
/// every tuple under bag semantics: each of its rows at that slot is one more
/// derivation.
pub fn bindings(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    bag: bool,
) -> Result<Vec<(Vec<u64>, u64)>, NativeRefusal> {
    bindings_run(prep, plan, current, bag).map(|(rows, _, _)| rows)
}

/// Rows with multiplicities, the relalg run, and the program behind the mask.
pub type BindingsRun = (Vec<(Vec<u64>, u64)>, Run, Option<Program>);

/// [`bindings`], plus the relalg run and the program that produced the mask.
pub fn bindings_run(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    bag: bool,
) -> Result<BindingsRun, NativeRefusal> {
    let n = prep.table.domain.len();
    let mut mask = vec![0u64; words_for(n)];
    let (run, program) = joint(prep, plan, current, Measure::Keep, Some(&mut mask))?;
    let sides: Vec<&Side> = prep.table.sides.iter().collect();
    let rows = tuples_over_mask(&prep.table.domain, &sides, plan, &mask, bag);
    Ok((rows, run, program))
}

/// The output tuples of `plan` at the slots `mask` names, with
/// multiplicities. `sides[i]` is the laid-out side for `plan.sides[i]`, over
/// `domain`. Each side is read at the slot the mask names, never at a
/// position taken from another side's values. Without `bag`, every
/// multiplicity is 1.
pub fn tuples_over_mask(
    domain: &JoinDomain,
    sides: &[&Side],
    plan: &StarPlan,
    mask: &[u64],
    bag: bool,
) -> Vec<(Vec<u64>, u64)> {
    let n = domain.len();
    let mut out: Vec<(Vec<u64>, u64)> = Vec::new();
    for_each_set(mask, n, |slot| {
        let center = u64::from(domain.id(slot));
        let mut tuples: Vec<(Vec<Option<u64>>, u64)> = vec![(vec![None; plan.sides.len()], 1)];
        for (i, s) in plan.sides.iter().enumerate() {
            let vals = sides[i].at(slot);
            if s.emit.is_some() {
                tuples = tuples
                    .into_iter()
                    .flat_map(|(t, m)| {
                        vals.iter().map(move |&v| {
                            let mut t = t.clone();
                            t[i] = Some(u64::from(v));
                            (t, m)
                        })
                    })
                    .collect();
            } else {
                let k = vals.len() as u64;
                for t in &mut tuples {
                    t.1 *= k;
                }
            }
        }
        for (t, m) in tuples {
            let row: Vec<u64> = plan
                .head
                .iter()
                .map(|h| match *h {
                    HeadTerm::Const(c) => c,
                    HeadTerm::Center => center,
                    HeadTerm::Side(i) => t[i].expect("emitting side"),
                })
                .collect();
            out.push((row, m));
        }
    });
    out.sort();
    let mut merged: Vec<(Vec<u64>, u64)> = Vec::new();
    for (row, m) in out {
        match merged.last_mut() {
            Some((r, c)) if *r == row => *c += m,
            _ => merged.push((row, m)),
        }
    }
    if !bag {
        for e in &mut merged {
            e.1 = 1;
        }
    }
    merged
}

/// True when the program contains an op that reads at a position taken from
/// another lane's values, or writes by one (the 2026-10-16 ruling): a
/// `Gather`, an `EqU32Via` predicate, a scatter terminal or `GroupSumViaI32`.
pub fn has_foreign_hop(p: &Program) -> bool {
    let op_hops = p.ops.iter().any(|op| match op {
        MaskOp::Gather { .. } => true,
        MaskOp::Pred { pred, .. } => matches!(pred, Pred::EqU32Via { .. }),
        _ => false,
    });
    let terminal_hops = matches!(
        p.terminal,
        Terminal::ScatterOrU32 { .. }
            | Terminal::ScatterCountU32 { .. }
            | Terminal::GroupSumViaI32 { .. }
    );
    op_hops || terminal_hops
}

/// A count the native terminal answers without enumerating a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldKind {
    /// `Σ_slot Π_side factor`, factors counting duplicates.
    Bag,
    /// `Σ_slot Π_(head-referenced side) |distinct factor|`; only for one rule
    /// whose head holds the center.
    Set,
    /// Whether any slot has every factor ≥ 1. No product is formed.
    Exists,
}

/// Which terminal answered a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalPath {
    /// The joint mask alone (unfiltered `Exists`).
    Mask,
    /// [`count_over_mask`]: no row was built.
    Folded,
    /// Every binding built and merged.
    #[default]
    Enumerated,
}

/// Whether `kind` over `stars` can be folded. Decided from the plan alone,
/// before any row is read (fold law 1); the data can only refuse a fold by
/// overflow, never redirect it to enumeration.
///
/// - `Count(Bag)`, any number of rules: per-row filtering commutes with a bag
///   merge, which only adds multiplicities, so per-rule folds sum.
/// - `Count(Set)`, one rule whose head holds [`HeadTerm::Center`]: see
///   [`count_over_mask`] for why that factorization is exact. Several rules
///   need a set union across rules; without the center, rows of two slots may
///   coincide (fold law 7's count-distinct across factors). Both enumerate.
/// - `Exists` with a post-filter (without one, the mask alone answers).
pub fn fold_kind(stars: &[StarPlan], kind: ResultKind, filtered: bool) -> Option<FoldKind> {
    match kind {
        ResultKind::Count(Semantics::Bag) => Some(FoldKind::Bag),
        ResultKind::Count(Semantics::Set)
            if stars.len() == 1 && stars[0].head.contains(&HeadTerm::Center) =>
        {
            Some(FoldKind::Set)
        }
        ResultKind::Exists if filtered => Some(FoldKind::Exists),
        _ => None,
    }
}

/// One slot's contribution: 0 if any factor is 0 (checked first, so a large
/// product never overflows ahead of a zero), else the checked product.
pub fn fold_slot(factors: &[u64]) -> Result<u64, NativeRefusal> {
    if factors.contains(&0) {
        return Ok(0);
    }
    factors
        .iter()
        .try_fold(1u64, |acc, &k| acc.checked_mul(k))
        .ok_or(NativeRefusal::CountOverflow)
}

/// The count [`tuples_over_mask`] followed by the post-filter and the
/// terminal would give, without building a row. Positions come only from the
/// joint mask; the values at a slot are data, compared and counted, never
/// used as a position.
///
/// A side's factor at slot `y` is the number of values in `at(y)` equal to
/// every post-filter constant on that side's head columns (all of them when
/// it has none). A `Const` head column whose filter differs answers 0; a
/// `Center` column whose filter differs from the slot's value drops the slot.
///
/// **Set** (\[REASONING\], pinned by the randomized fold-vs-enumeration test;
/// it narrows fold law 7, whose own grade is \[literature + reasoning\]).
/// Assumptions: (1) the center column's value is `domain.id(y)`; (2) a slot
/// is in the joint mask only when every side is non-empty at it; (3)
/// post-filters touch head columns only. `JoinDomain` is injective, so a row
/// holding the center determines its slot, and rows of different slots
/// differ. Within a slot, a row is fixed by the values of the head-referenced
/// sides, so the distinct rows are the product of their distinct filtered
/// value sets; a side the head does not reference contributes 1 by (2).
/// Hence `Count(Set) = Σ_slot Π |distinct|` exactly. [`fold_kind`] admits
/// `Set` only under these conditions.
pub fn count_over_mask(
    domain: &JoinDomain,
    sides: &[&Side],
    plan: &StarPlan,
    mask: &[u64],
    post_filter: &[Option<u64>],
    kind: FoldKind,
) -> Result<u64, NativeRefusal> {
    debug_assert!(kind != FoldKind::Set || plan.head.contains(&HeadTerm::Center));
    let filter = |k: usize| post_filter.get(k).copied().flatten();
    let mut center_filters = Vec::new();
    let mut side_filters: Vec<Vec<u64>> = vec![Vec::new(); sides.len()];
    let mut referenced = vec![false; sides.len()];
    for (k, h) in plan.head.iter().enumerate() {
        match (*h, filter(k)) {
            (HeadTerm::Const(c), Some(f)) if c != f => return Ok(0),
            (HeadTerm::Const(_), _) => {}
            (HeadTerm::Center, f) => center_filters.extend(f),
            (HeadTerm::Side(i), f) => {
                referenced[i] = true;
                side_filters[i].extend(f);
            }
        }
    }
    let mut scratch: Vec<u32> = Vec::new();
    let mut factors: Vec<u64> = Vec::with_capacity(sides.len());
    let mut total = 0u64;
    let mut found = false;
    let mut refused = None;
    for_each_set(mask, domain.len(), |slot| {
        if refused.is_some() || found {
            return;
        }
        let center = u64::from(domain.id(slot));
        if center_filters.iter().any(|&f| f != center) {
            return;
        }
        factors.clear();
        for (i, side) in sides.iter().enumerate() {
            let vals = side.at(slot);
            let fs = &side_filters[i];
            let keep = |v: &u32| fs.iter().all(|&f| u64::from(*v) == f);
            let k = match kind {
                FoldKind::Set if !referenced[i] => 1,
                FoldKind::Set => {
                    scratch.clear();
                    scratch.extend(vals.iter().copied().filter(keep));
                    scratch.sort_unstable();
                    scratch.dedup();
                    scratch.len()
                }
                _ if fs.is_empty() => vals.len(),
                _ => vals.iter().filter(|v| keep(v)).count(),
            };
            factors.push(k as u64);
        }
        if kind == FoldKind::Exists {
            found = !factors.contains(&0);
            return;
        }
        match fold_slot(&factors)
            .and_then(|c| total.checked_add(c).ok_or(NativeRefusal::CountOverflow))
        {
            Ok(t) => total = t,
            Err(e) => refused = Some(e),
        }
    });
    if let Some(e) = refused {
        return Err(e);
    }
    Ok(if kind == FoldKind::Exists {
        u64::from(found)
    } else {
        total
    })
}

#[cfg(test)]
mod fold_tests {
    use super::*;
    use std::collections::BTreeMap;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// One star: its plan, its rows per side (`[center, other]`), its table.
    struct Star {
        plan: StarPlan,
        table: JoinTable,
    }

    fn key(i: usize) -> SideKey {
        SideKey {
            rel: RelId(i as u16),
            key_col: 0,
            restrict: None,
            unary: false,
        }
    }

    /// `emits[i]` says whether side `i` emits; `head` and `rows` as given.
    fn star(rows: Vec<Vec<[u32; 2]>>, emits: &[bool], head: Vec<HeadTerm>) -> Star {
        let inputs: Vec<(RelationView<'_>, usize)> = rows
            .iter()
            .map(|r| {
                (
                    RelationView {
                        rows: r,
                        generation: 1,
                    },
                    0,
                )
            })
            .collect();
        let table = JoinTable::prepare_star(&inputs);
        let sides = emits
            .iter()
            .enumerate()
            .map(|(i, &e)| SidePlan {
                key: key(i),
                emit: e.then_some(VarId(i as u16 + 1)),
            })
            .collect();
        Star {
            plan: StarPlan {
                center: VarId(0),
                sides,
                head,
            },
            table,
        }
    }

    fn mask(t: &JoinTable) -> Vec<u64> {
        let n = t.domain.len();
        let mut m = vec![u64::MAX; words_for(n)];
        for s in &t.sides {
            for (w, p) in m.iter_mut().zip(s.present()) {
                *w &= p;
            }
        }
        m
    }

    fn fold(s: &Star, filter: &[Option<u64>], kind: FoldKind) -> Result<u64, NativeRefusal> {
        let sides: Vec<&Side> = s.table.sides.iter().collect();
        count_over_mask(
            &s.table.domain,
            &sides,
            &s.plan,
            &mask(&s.table),
            filter,
            kind,
        )
    }

    /// The enumerating terminal, exactly as `pipeline::execute` runs it.
    fn enumerate(stars: &[Star], filter: &[Option<u64>]) -> BTreeMap<Vec<u64>, u64> {
        let mut bag: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
        for s in stars {
            let sides: Vec<&Side> = s.table.sides.iter().collect();
            for (row, m) in
                tuples_over_mask(&s.table.domain, &sides, &s.plan, &mask(&s.table), true)
            {
                *bag.entry(row).or_insert(0) += m;
            }
        }
        bag.retain(|row, _| {
            row.iter()
                .zip(filter)
                .all(|(v, f)| f.is_none_or(|f| f == *v))
        });
        bag
    }

    fn random_star(r: &mut Rng, arity: usize, center_in_head: bool) -> Star {
        let nsides = 1 + r.below(3) as usize;
        let emits: Vec<bool> = (0..nsides).map(|_| r.below(3) != 0).collect();
        let rows: Vec<Vec<[u32; 2]>> = (0..nsides)
            .map(|_| {
                (0..1 + r.below(24))
                    .map(|_| [r.below(6) as u32, r.below(4) as u32])
                    .collect()
            })
            .collect();
        let emitting: Vec<usize> = (0..nsides).filter(|&i| emits[i]).collect();
        let mut head: Vec<HeadTerm> = (0..arity)
            .map(|_| match r.below(3) {
                0 => HeadTerm::Const(r.below(3)),
                1 if !emitting.is_empty() => {
                    HeadTerm::Side(emitting[r.below(emitting.len() as u64) as usize])
                }
                _ => HeadTerm::Center,
            })
            .collect();
        head.retain(|h| center_in_head || *h != HeadTerm::Center);
        while head.len() < arity {
            head.push(HeadTerm::Const(1));
        }
        if center_in_head && !head.contains(&HeadTerm::Center) {
            head[0] = HeadTerm::Center;
        }
        star(rows, &emits, head)
    }

    fn random_filter(r: &mut Rng, arity: usize) -> Vec<Option<u64>> {
        (0..arity)
            .map(|_| (r.below(3) == 0).then(|| r.below(4)))
            .collect()
    }

    /// G5: the fold equals the enumerating terminal on random stars with
    /// duplicate values, constants, filters on every column kind, emitting
    /// sides the head does not use, and 1–3 rules.
    #[test]
    fn folded_counts_equal_the_enumerating_terminal() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut bag_nonzero, mut set_nonzero, mut exists_true, mut exists_false) = (0, 0, 0, 0);
        for _ in 0..600 {
            let arity = 1 + r.below(3) as usize;
            let rules = 1 + r.below(3) as usize;
            let filter = random_filter(&mut r, arity);
            let stars: Vec<Star> = (0..rules)
                .map(|_| {
                    let center = r.below(2) == 0;
                    random_star(&mut r, arity, center)
                })
                .collect();
            let bag = enumerate(&stars, &filter);
            let folded_bag: u64 = stars
                .iter()
                .map(|s| fold(s, &filter, FoldKind::Bag).unwrap())
                .sum();
            assert_eq!(folded_bag, bag.values().sum::<u64>(), "Bag");
            bag_nonzero += usize::from(folded_bag > 0);
            let exists = stars
                .iter()
                .any(|s| fold(s, &filter, FoldKind::Exists).unwrap() > 0);
            assert_eq!(exists, !bag.is_empty(), "Exists");
            if exists {
                exists_true += 1;
            } else {
                exists_false += 1;
            }
            let one = std::slice::from_ref(&stars[0]);
            if stars[0].plan.head.contains(&HeadTerm::Center) {
                let set = enumerate(one, &filter).len() as u64;
                assert_eq!(fold(&stars[0], &filter, FoldKind::Set).unwrap(), set, "Set");
                set_nonzero += usize::from(set > 0);
            }
        }
        // The random space reaches non-trivial answers on every path.
        assert!(
            bag_nonzero > 100 && set_nonzero > 50,
            "{bag_nonzero} {set_nonzero}"
        );
        assert!(
            exists_true > 50 && exists_false > 50,
            "{exists_true} {exists_false}"
        );
    }

    /// G8(i): an emitting side the head does not use, with two distinct
    /// values at the slot, contributes 1 under Set.
    #[test]
    fn an_unreferenced_emitting_side_does_not_multiply_a_set_count() {
        let s = star(
            vec![vec![[1, 7], [1, 8]], vec![[1, 2], [1, 3]]],
            &[true, true],
            vec![HeadTerm::Center, HeadTerm::Side(1)],
        );
        let set = enumerate(std::slice::from_ref(&s), &[None, None]).len() as u64;
        assert_eq!(set, 2);
        assert_eq!(fold(&s, &[None, None], FoldKind::Set).unwrap(), set);
    }

    /// G8(ii): duplicates at a slot count under Bag, once under Set.
    #[test]
    fn duplicates_count_for_bag_and_once_for_set() {
        let s = star(
            vec![vec![[1, 5], [1, 5], [1, 6]], vec![[1, 9], [1, 9]]],
            &[true, true],
            vec![HeadTerm::Center, HeadTerm::Side(0), HeadTerm::Side(1)],
        );
        let none = [None, None, None];
        assert_eq!(fold(&s, &none, FoldKind::Bag).unwrap(), 6);
        assert_eq!(fold(&s, &none, FoldKind::Set).unwrap(), 2);
        let set = enumerate(std::slice::from_ref(&s), &none).len() as u64;
        assert_eq!(set, 2);
    }

    /// Why `fold_kind` refuses Set without the center: rows of two slots
    /// coincide, so the per-slot product overcounts (fold law 7). The fold
    /// would answer 2; the set holds one row.
    #[test]
    fn without_the_center_two_slots_can_give_one_row() {
        let s = star(
            vec![vec![[1, 7], [2, 7]], vec![[1, 0], [2, 0]]],
            &[true, false],
            vec![HeadTerm::Side(0)],
        );
        let set = enumerate(std::slice::from_ref(&s), &[None]).len() as u64;
        assert_eq!(set, 1);
        let per_slot = count_over_mask_unchecked_set(&s);
        assert_eq!(per_slot, 2, "the hazard is real on this fixture");
        assert_eq!(
            fold_kind(
                std::slice::from_ref(&s.plan),
                ResultKind::Count(Semantics::Set),
                false
            ),
            None
        );
    }

    /// The Set product with the center precondition skipped (release builds
    /// skip the `debug_assert`); used only to show the hazard above.
    fn count_over_mask_unchecked_set(s: &Star) -> u64 {
        let sides: Vec<&Side> = s.table.sides.iter().collect();
        let mut total = 0;
        for_each_set(&mask(&s.table), s.table.domain.len(), |slot| {
            let mut v: Vec<u32> = sides[0].at(slot).to_vec();
            v.sort_unstable();
            v.dedup();
            total += v.len() as u64;
        });
        total
    }

    /// A head that repeats a side with two different filters answers 0.
    #[test]
    fn a_side_filtered_to_two_values_at_once_answers_zero() {
        let s = star(
            vec![vec![[1, 5], [1, 6]]],
            &[true],
            vec![HeadTerm::Side(0), HeadTerm::Side(0)],
        );
        let filter = [Some(5), Some(6)];
        assert_eq!(fold(&s, &filter, FoldKind::Bag).unwrap(), 0);
        assert!(enumerate(std::slice::from_ref(&s), &filter).is_empty());
        assert_eq!(fold(&s, &[Some(5), Some(5)], FoldKind::Bag).unwrap(), 1);
    }

    /// G9(v): a filter that empties one side at one slot drops that slot
    /// only; Exists still sees the other slot.
    #[test]
    fn a_zero_factor_drops_its_slot_only() {
        let s = star(
            vec![vec![[1, 4], [2, 5]], vec![[1, 8], [2, 8]]],
            &[true, true],
            vec![HeadTerm::Side(0)],
        );
        let filter = [Some(5)];
        assert_eq!(fold(&s, &filter, FoldKind::Bag).unwrap(), 1);
        assert_eq!(fold(&s, &filter, FoldKind::Exists).unwrap(), 1);
        assert_eq!(fold(&s, &[Some(9)], FoldKind::Exists).unwrap(), 0);
    }

    /// G10: the slot product refuses only when the exact answer exceeds u64,
    /// and a zero factor wins over factors whose product would overflow.
    #[test]
    fn overflow_refuses_only_unrepresentable_counts() {
        let big = 1u64 << 40;
        assert_eq!(fold_slot(&[big, big]), Err(NativeRefusal::CountOverflow));
        assert_eq!(fold_slot(&[big, big, 0]), Ok(0));
        assert_eq!(fold_slot(&[0, big, big]), Ok(0));
        assert_eq!(fold_slot(&[1 << 31, 1 << 32]), Ok(1 << 63));
        assert_eq!(fold_slot(&[]), Ok(1));
    }

    /// G7 at the plan level: what folds is decided before any row is read.
    #[test]
    fn fold_kind_admits_only_the_proven_shapes() {
        let with = |h: Vec<HeadTerm>| StarPlan {
            center: VarId(0),
            sides: vec![],
            head: h,
        };
        let c = with(vec![HeadTerm::Center, HeadTerm::Side(0)]);
        let nc = with(vec![HeadTerm::Side(0)]);
        let set = ResultKind::Count(Semantics::Set);
        let bag = ResultKind::Count(Semantics::Bag);
        assert_eq!(
            fold_kind(std::slice::from_ref(&c), set, false),
            Some(FoldKind::Set)
        );
        assert_eq!(fold_kind(std::slice::from_ref(&nc), set, false), None);
        assert_eq!(fold_kind(&[c.clone(), c.clone()], set, false), None);
        assert_eq!(
            fold_kind(&[nc.clone(), c.clone()], bag, false),
            Some(FoldKind::Bag)
        );
        assert_eq!(
            fold_kind(std::slice::from_ref(&nc), ResultKind::Exists, true),
            Some(FoldKind::Exists)
        );
        assert_eq!(
            fold_kind(std::slice::from_ref(&nc), ResultKind::Exists, false),
            None
        );
        assert_eq!(
            fold_kind(&[c], ResultKind::Bindings(Semantics::Bag), false),
            None
        );
    }
}
