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
//! Every terminal folds over it in slot order and builds no row:
//!
//! - existence and the bag count (`Σ_slot Π_i |side_i at slot|`, law 7) read
//!   the sides' lengths in place;
//! - a set answer with one free head column scatters that column's values
//!   into a mask over the value space; more free columns are refused;
//! - bag bindings are handed out as coordinates (the kept slots) plus the
//!   sides' rows borrowed in place; the caller applies the head template.
//!
//! A side is read at the slot the mask names, never at a position taken from
//! another side's values. No terminal copies values into owned rows or sorts
//! them: a tuple violates zero copy, folding and the substrate's natural
//! (slot) order (lance-graph `CLAUDE.md`, P0).
//!
//! Anything else (a chain, a pattern of arity > 2, a side without `c`) is
//! refused by name.

use lance_graph_mask_risc::{MaskOp, Planes, Pred, Program, Terminal, words_for};
use lance_graph_relalg::{Answer, Measure, Query, Rel, RelError, Run, run_with_program};
use mork_pathmap_probe::RelationView;
use mork_pathmap_probe::join::{
    JoinDomain, JoinRefusal, JoinTable, Run as SideRun, SelectedSide, Side, for_each_set,
    for_each_set_ahead, rows,
};
use ndarray::simd::eq_u32_to_mask;

use crate::logic::{RelId, VarId};
use crate::mork_api::{Expr, QueryMulti};

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
    /// The variable this side's other column binds, if the template uses it.
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

/// The relations a plan reads, by id. Each is two lanes and a generation
/// ([`RelationView`]); a unary relation's lanes are equal.
pub struct Store {
    pub relations: Vec<(RelId, RelationView)>,
}

impl Store {
    /// Builds each relation's lanes once from authored fixture rows. For
    /// tests and examples; a stored relation arrives as lanes.
    pub fn from_rows<R: AsRef<[[u32; 2]]>>(
        relations: impl IntoIterator<Item = (RelId, R, u64)>,
    ) -> Store {
        Store {
            relations: relations
                .into_iter()
                .map(|(r, rows, g)| (r, RelationView::from_rows(rows.as_ref(), g)))
                .collect(),
        }
    }

    pub fn view(&self, rel: RelId) -> &RelationView {
        &self
            .relations
            .iter()
            .find(|(r, _)| *r == rel)
            .expect("relation in store")
            .1
    }
}

/// A relationship table prepared for one plan's sides, with the witness that
/// says which sides those are and which domain they share.
#[derive(Debug, Clone)]
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
pub fn prepare(plan: &StarPlan, store: &Store) -> PreparedStar {
    prepare_views(plan, side_views(plan, store), false)
}

/// [`prepare`] with semijoin reduction: when at least one side is restricted
/// by a constant, every side keeps only the rows whose center the restricted
/// sides all have. A center outside that set cannot reach the joint mask, so
/// the answer and its multiplicities are unchanged; the domain shrinks to the
/// reduced centers. Without a restricted side nothing changes.
pub fn prepare_semijoin(plan: &StarPlan, store: &Store) -> PreparedStar {
    prepare_views(plan, side_views(plan, store), true)
}

/// Each plan side's relation, as lanes shared with the store (a reference
/// count each, no copy).
fn side_views(plan: &StarPlan, store: &Store) -> Vec<RelationView> {
    plan.sides
        .iter()
        .map(|s| store.view(s.key.rel).clone())
        .collect()
}

/// Lays the sides out over one shared domain, in plan side order, from their
/// relations' lanes. A side's restriction, and with `semijoin` the centers
/// every restricted side has, select row positions: the index map leaves the
/// other rows out, and no row is copied.
pub fn prepare_views(plan: &StarPlan, views: Vec<RelationView>, semijoin: bool) -> PreparedStar {
    let restricted = |i: usize| {
        let k = plan.sides[i].key;
        !k.unary && k.restrict.is_some()
    };
    // A restricted side's rows: one compare over the lane holding the
    // constant, into a mask.
    let restriction: Vec<Option<Vec<u64>>> = (0..views.len())
        .map(|i| {
            let k = plan.sides[i].key;
            let c = k.restrict.filter(|_| restricted(i))?;
            let lane = &views[i].cols[1 - k.key_col as usize];
            let mut m = vec![0u64; words_for(lane.len())];
            if let Ok(c) = u32::try_from(c) {
                eq_u32_to_mask(lane, c, &mut m);
            }
            Some(m)
        })
        .collect();
    // Centers every restricted side has: one domain per restricted side over
    // its kept rows, intersected.
    let centers: Option<JoinDomain> = if semijoin {
        let mut acc: Option<JoinDomain> = None;
        for (i, m) in restriction.iter().enumerate() {
            let Some(m) = m else { continue };
            let key = &views[i].cols[plan.sides[i].key.key_col as usize];
            let c = JoinDomain::build(rows(key.len(), Some(m)).map(|row| key[row]));
            acc = Some(match acc {
                None => c,
                Some(prev) => {
                    let both: Vec<u32> = (0..prev.len())
                        .map(|s| prev.id(s))
                        .filter(|&v| c.contains(v))
                        .collect();
                    JoinDomain::build(both.iter().copied())
                }
            });
        }
        acc
    } else {
        None
    };
    // Each side's keep mask: its restriction, and with centers the rows whose
    // center every restricted side has. `None`: every row.
    let keeps: Vec<Option<Vec<u64>>> = (0..views.len())
        .map(|i| {
            let Some(c) = &centers else {
                return restriction[i].clone();
            };
            let key = &views[i].cols[plan.sides[i].key.key_col as usize];
            let mut m = vec![0u64; words_for(key.len())];
            for row in rows(key.len(), restriction[i].as_deref()) {
                if c.contains(key[row]) {
                    m[row >> 6] |= 1 << (row & 63);
                }
            }
            Some(m)
        })
        .collect();
    let inputs: Vec<SelectedSide<'_>> = views
        .iter()
        .zip(&plan.sides)
        .zip(&keeps)
        .map(|((v, s), keep)| (v, s.key.key_col as usize, keep.as_deref()))
        .collect();
    let table = JoinTable::prepare_star_where(&inputs);
    PreparedStar {
        domain_fp: fingerprint(&table),
        keys: plan.sides.iter().map(|s| s.key).collect(),
        rows_prepared: table.sides.iter().map(Side::rows).sum(),
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

/// The joint mask kept for the terminal, the relalg run, and the program
/// that produced it. The mask is the answer's coordinates: slots of the
/// shared domain, in natural order.
pub fn kept(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
) -> Result<(Vec<u64>, Run, Option<Program>), NativeRefusal> {
    let mut mask = vec![0u64; words_for(prep.table.domain.len())];
    let (run, program) = joint(prep, plan, current, Measure::Keep, Some(&mut mask))?;
    Ok((mask, run, program))
}

/// One star read through its laid-out sides: the domain the kept mask
/// addresses and, per plan side, the side holding it. Borrowed; nothing is
/// copied out of the table.
#[derive(Clone, Copy)]
pub struct StarView<'a> {
    pub domain: &'a JoinDomain,
    pub sides: &'a [&'a Side],
    pub star: &'a StarPlan,
}

/// What the post filter asks of one rule, resolved against its head.
/// `None` from [`RuleFilter::of`]: a head constant contradicts the filter,
/// so the rule contributes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleFilter {
    /// The center must equal this value.
    pub center: Option<u64>,
    /// Side `i`'s value must equal `side[i]`.
    pub side: Vec<Option<u64>>,
}

impl RuleFilter {
    pub fn of(star: &StarPlan, post_filter: &[Option<u64>]) -> Option<RuleFilter> {
        let mut f = RuleFilter {
            center: None,
            side: vec![None; star.sides.len()],
        };
        for (h, want) in star.head.iter().zip(post_filter) {
            let Some(k) = *want else { continue };
            match *h {
                HeadTerm::Const(c) if c != k => return None,
                HeadTerm::Const(_) => {}
                HeadTerm::Center => match f.center {
                    Some(prev) if prev != k => return None,
                    _ => f.center = Some(k),
                },
                HeadTerm::Side(i) => match f.side[i] {
                    Some(prev) if prev != k => return None,
                    _ => f.side[i] = Some(k),
                },
            }
        }
        Some(f)
    }
}

/// Narrows `kept` to the filtered center: one bit, or none when the value
/// is outside the domain. A word-wise pass; no value is read.
pub fn narrow_center(kept: &mut [u64], domain: &JoinDomain, filter: &RuleFilter) {
    let Some(k) = filter.center else { return };
    let slot = u32::try_from(k).ok().and_then(|v| domain.slot_of(v));
    for (w, word) in kept.iter_mut().enumerate() {
        *word &= match slot {
            Some(s) if s >> 6 == w => 1 << (s & 63),
            _ => 0,
        };
    }
}

/// Derivations side `i` contributes at `slot`: its row count there, or the
/// rows equal to its filter value. Read in place.
fn side_count(view: &StarView<'_>, filter: &RuleFilter, slot: usize, i: usize) -> u64 {
    let vals = view.sides[i].at(slot);
    match filter.side[i] {
        None => vals.len() as u64,
        Some(k) => vals.iter().filter(|&v| u64::from(v) == k).count() as u64,
    }
}

/// Derivations at `slot`: the product of every side's count (law 7). Zero
/// when a filtered side has no matching row.
fn slot_derivations(view: &StarView<'_>, filter: &RuleFilter, slot: usize) -> u64 {
    (0..view.sides.len())
        .map(|i| side_count(view, filter, slot, i))
        .product()
}

/// Whether any kept slot yields a derivation. Stops at the first.
pub fn any_derivation(view: &StarView<'_>, kept: &[u64], filter: &RuleFilter) -> bool {
    if filter.side.iter().all(Option::is_none) {
        return kept.iter().any(|&w| w != 0);
    }
    let n = view.domain.len();
    for (wi, &w) in kept.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            let slot = wi * 64 + w.trailing_zeros() as usize;
            if slot < n && slot_derivations(view, filter, slot) > 0 {
                return true;
            }
            w &= w - 1;
        }
    }
    false
}

/// The bag count over the kept slots, `Σ_slot Π_i count_i(slot)`, without
/// building a row. Reads side lengths (and, for a filtered side, its
/// values) in place, in slot order.
pub fn bag_count(view: &StarView<'_>, kept: &[u64], filter: &RuleFilter) -> u64 {
    let mut total = 0u64;
    for_each_set(kept, view.domain.len(), |slot| {
        total += slot_derivations(view, filter, slot);
    });
    total
}

/// Distinct answers whose head has at most one free column: every other
/// position is a constant (a head constant, or a column the post filter
/// fixes). The free column's distinct values are a mask over the value
/// space (`ScatterOrU32`'s shape: the scattered mask is the demanded
/// result). `values` is trimmed of trailing zero words; a template with no
/// free column answers one row when `values == [1]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DistinctGroup {
    /// Per head position: `Some(c)` a constant, `None` the free column.
    pub template: Vec<Option<u64>>,
    pub values: Vec<u64>,
}

impl DistinctGroup {
    /// Distinct rows in this group.
    pub fn len(&self) -> u64 {
        self.values.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.values.iter().all(|&w| w == 0)
    }
}

/// Why a set answer is not answered by a fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetRefusal {
    /// More than one free head column: distinct rows over a projection need
    /// a seen-set over value pairs, which no terminal provides (GAP-A2).
    MultiColumn { free: usize },
    /// Two rules' templates can share a row without being the same
    /// template, so their union is not a union of value masks.
    OverlappingTemplates,
    /// A value too wide for a value-space mask (the free column must be a
    /// u32).
    ValueOutOfRange,
}

fn set_bit(values: &mut Vec<u64>, v: u64) {
    let w = (v >> 6) as usize;
    if values.len() <= w {
        values.resize(w + 1, 0);
    }
    values[w] |= 1 << (v & 63);
}

fn disjoint(a: &[Option<u64>], b: &[Option<u64>]) -> bool {
    a.iter()
        .zip(b)
        .any(|(x, y)| matches!((x, y), (Some(p), Some(q)) if p != q))
}

/// Adds one rule's distinct answers to `groups`. Fails closed: a head with
/// two or more free columns, or a template that overlaps another group's
/// without equalling it, is refused rather than deduplicated by sorting.
pub fn add_distinct(
    view: &StarView<'_>,
    kept: &[u64],
    filter: &RuleFilter,
    groups: &mut Vec<DistinctGroup>,
) -> Result<(), SetRefusal> {
    let template: Vec<Option<u64>> = view
        .star
        .head
        .iter()
        .map(|h| match *h {
            HeadTerm::Const(c) => Some(c),
            HeadTerm::Center => filter.center,
            HeadTerm::Side(i) => filter.side[i],
        })
        .collect();
    let free: Vec<usize> = (0..template.len())
        .filter(|&p| template[p].is_none())
        .collect();
    if free.len() > 1 {
        return Err(SetRefusal::MultiColumn { free: free.len() });
    }
    let g = match groups.iter().position(|g| g.template == template) {
        Some(g) => g,
        None => {
            if groups.iter().any(|g| !disjoint(&g.template, &template)) {
                return Err(SetRefusal::OverlappingTemplates);
            }
            groups.push(DistinctGroup {
                template,
                values: Vec::new(),
            });
            groups.len() - 1
        }
    };
    let n = view.domain.len();
    let values = &mut groups[g].values;
    match free.first().map(|&p| view.star.head[p]) {
        None => {
            if any_derivation(view, kept, filter) {
                set_bit(values, 0);
            }
        }
        Some(HeadTerm::Center) => for_each_set(kept, n, |slot| {
            if slot_derivations(view, filter, slot) > 0 {
                set_bit(values, u64::from(view.domain.id(slot)));
            }
        }),
        Some(HeadTerm::Side(i)) => for_each_set(kept, n, |slot| {
            let others = (0..view.sides.len())
                .filter(|&j| j != i)
                .all(|j| side_count(view, filter, slot, j) > 0);
            if others {
                for v in view.sides[i].at(slot).iter() {
                    set_bit(values, u64::from(v));
                }
            }
        }),
        Some(HeadTerm::Const(_)) => unreachable!("a constant is never free"),
    }
    Ok(())
}

/// Puts `groups` in canonical form: trailing zero words trimmed, empty
/// groups dropped, sorted by template. Equal answers compare equal however
/// the rules were planned.
pub fn canonical(groups: &mut Vec<DistinctGroup>) {
    for g in groups.iter_mut() {
        while g.values.last() == Some(&0) {
            g.values.pop();
        }
    }
    groups.retain(|g| !g.values.is_empty());
    groups.sort();
}

/// One rule's derivations as coordinates: the rule, its star, its filter
/// and the kept slots. The values stay in the table; read them with
/// [`for_each_slot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleCoordinates {
    pub rule: usize,
    /// The table the kept slots address (the pipeline's star index, or the
    /// shared path's group index).
    pub table: usize,
    /// For each plan side, its side index in that table.
    pub sides: Vec<usize>,
    pub star: StarPlan,
    pub filter: RuleFilter,
    pub kept: Vec<u64>,
}

/// One kept slot, borrowed: the center and every side's rows there. The
/// derivations at the slot are the cartesian product of the emitting sides'
/// filtered rows, each counted once per row of every non-emitting side; the
/// caller applies the head template.
pub struct SlotView<'a> {
    pub slot: usize,
    pub center: u32,
    pub sides: &'a [SideRun<'a>],
    pub filter: &'a RuleFilter,
    pub derivations: u64,
}

/// Calls `f` for every kept slot with at least one derivation, in slot
/// order, with the sides' rows borrowed in place.
pub fn for_each_slot(view: &StarView<'_>, rc: &RuleCoordinates, mut f: impl FnMut(&SlotView<'_>)) {
    let mut sides: Vec<SideRun<'_>> = vec![SideRun::default(); view.sides.len()];
    let ahead = |next: usize| view.sides.iter().for_each(|s| s.at(next).prefetch());
    for_each_set_ahead(&rc.kept, view.domain.len(), ahead, |slot| {
        let derivations = slot_derivations(view, &rc.filter, slot);
        if derivations == 0 {
            return;
        }
        for (i, s) in view.sides.iter().enumerate() {
            sides[i] = s.at(slot);
        }
        f(&SlotView {
            slot,
            center: view.domain.id(slot),
            sides: &sides,
            filter: &rc.filter,
            derivations,
        });
    });
}

/// Derivations at the kept slots whose head reads as `pattern` (`None` at a
/// column: any value). A fold, never an enumeration: per slot it is law 7
/// with each side's count narrowed to the pattern's value. When the pattern
/// pins the center, its global id is inverted to the local slot
/// (`slot_of`, the one pointer table) and only that slot is read. With an
/// all-`None` pattern this is [`bag_count`].
pub fn matching(
    view: &StarView<'_>,
    rc: &RuleCoordinates,
    pattern: impl Fn(usize) -> Option<u64>,
) -> u64 {
    let k = view.sides.len();
    let mut center: Option<u64> = rc.filter.center;
    let mut want: Vec<Option<u64>> = vec![None; k];
    for (c, h) in view.star.head.iter().enumerate() {
        let Some(v) = pattern(c) else { continue };
        let slot = match *h {
            HeadTerm::Const(x) if x != v => return 0,
            HeadTerm::Const(_) => continue,
            HeadTerm::Center => &mut center,
            HeadTerm::Side(i) => &mut want[i],
        };
        match *slot {
            Some(w) if w != v => return 0,
            _ => *slot = Some(v),
        }
    }
    for (i, w) in want.iter_mut().enumerate() {
        match (*w, rc.filter.side[i]) {
            (Some(a), Some(b)) if a != b => return 0,
            (None, b) => *w = b,
            _ => {}
        }
    }
    let at_slot = |slot: usize| -> u64 {
        (0..k)
            .map(|i| {
                let vals = view.sides[i].at(slot);
                match want[i] {
                    None => vals.len() as u64,
                    Some(t) => vals.iter().filter(|&v| u64::from(v) == t).count() as u64,
                }
            })
            .product()
    };
    if let Some(v) = center {
        let Some(slot) = u32::try_from(v).ok().and_then(|y| view.domain.slot_of(y)) else {
            return 0;
        };
        let kept = rc
            .kept
            .get(slot / 64)
            .is_some_and(|w| w >> (slot % 64) & 1 == 1);
        return if kept { at_slot(slot) } else { 0 };
    }
    let mut total = 0u64;
    for_each_set(&rc.kept, view.domain.len(), |slot| total += at_slot(slot));
    total
}

/// One derivation, read in place: the head template over the lanes at the
/// side positions `at` (the `"{0}{1}" -f $laneA[$i],$laneB[$i]` shape).
/// Nothing is copied; [`DerivationView::get`] reads one column.
pub struct DerivationView<'a> {
    pub center: u32,
    head: &'a [HeadTerm],
    sides: &'a [SideRun<'a>],
    at: &'a [usize],
}

impl DerivationView<'_> {
    pub fn arity(&self) -> usize {
        self.head.len()
    }

    pub fn get(&self, c: usize) -> u64 {
        match self.head[c] {
            HeadTerm::Const(k) => k,
            HeadTerm::Center => u64::from(self.center),
            HeadTerm::Side(i) => u64::from(self.sides[i].get(self.at[i])),
        }
    }
}

/// Every derivation at the kept slots, in slot order, as a template read over
/// the side positions; `weight` counts the non-emitting sides' rows. Walks
/// positions, never values: the only state is one position per side.
pub fn for_each_derivation(
    view: &StarView<'_>,
    rc: &RuleCoordinates,
    mut f: impl FnMut(&DerivationView<'_>, u64),
) {
    let k = view.sides.len();
    let emitting: Vec<usize> = (0..k)
        .filter(|&i| view.star.sides[i].emit.is_some())
        .collect();
    let mut at = vec![0usize; k];
    let mut sides: Vec<SideRun<'_>> = vec![SideRun::default(); k];
    let keeps = |i: usize, v: u32| rc.filter.side[i].is_none_or(|t| t == u64::from(v));
    let ahead = |next: usize| view.sides.iter().for_each(|s| s.at(next).prefetch());
    for_each_set_ahead(&rc.kept, view.domain.len(), ahead, |slot| {
        let center = view.domain.id(slot);
        if rc.filter.center.is_some_and(|t| t != u64::from(center)) {
            return;
        }
        for (i, s) in view.sides.iter().enumerate() {
            sides[i] = s.at(slot);
        }
        let weight: u64 = (0..k)
            .filter(|i| !emitting.contains(i))
            .map(|i| sides[i].iter().filter(|&v| keeps(i, v)).count() as u64)
            .product();
        if weight == 0 || emitting.iter().any(|&i| sides[i].is_empty()) {
            return;
        }
        for &i in &emitting {
            at[i] = 0;
        }
        loop {
            if emitting.iter().all(|&i| keeps(i, sides[i].get(at[i]))) {
                f(
                    &DerivationView {
                        center,
                        head: &view.star.head,
                        sides: &sides,
                        at: &at,
                    },
                    weight,
                );
            }
            // Odometer over the emitting sides' positions.
            let mut d = 0;
            while d < emitting.len() {
                let i = emitting[d];
                at[i] += 1;
                if at[i] < sides[i].len() {
                    break;
                }
                at[i] = 0;
                d += 1;
            }
            if d == emitting.len() {
                break;
            }
        }
    });
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
