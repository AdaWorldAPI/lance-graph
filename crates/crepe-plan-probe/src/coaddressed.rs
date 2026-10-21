//! The co-addressed join, as a named operator (D-RU-4 / U4).
//!
//! STATUS: PROBE, like [`crate::native`]: the table underneath
//! (`mork_pathmap_probe::join`) is measured but not admitted.
//!
//! # What it is
//!
//! A conjunction of per-table [`Rel`]s over one [`JoinDomain`]:
//!
//! ```text
//! CoJoin(table, [rel_0, …, rel_k])
//!   rows_i  = rel_i over side i's own rows            (relalg fold, per side)
//!   count_i = |rows_i| per slot                       (GroupReduce Count, keyed
//!                                                      by side i's slot lane)
//!   joint   = ∧_i (count_i > 0)                       (relalg And of planes)
//! ```
//!
//! [`crate::native`] applies a side's restriction cold: the rows are filtered
//! before the side is laid out, so every new constant needs a new prepared
//! table. Here the restriction is a `Rel` evaluated at query time over one
//! table prepared without it, and it can be any `Rel` relalg lowers (equality,
//! inequality, `Or`, `Not`), not only one constant.
//!
//! # Why it is not a gather
//!
//! Each side's rows are laid out by slot, so row `j` of side `i` sits in the
//! run `offsets[s]..offsets[s + 1]` of its slot `s`. The slot lane is built
//! from those offsets, the side's own layout, and is non-decreasing. The
//! projection `rows_i → count_i` adds each selected row to the run it already
//! sits in. The joint is an `And` of planes over the one domain. Tuple
//! enumeration reads each side at the slot the joint names and tests the row
//! bits in that slot's run. No operand is read at a position taken from
//! another operand's values. [`CoEval::programs`] keeps every mask-risc
//! program the operator ran, so a caller can check that with
//! [`crate::native::has_foreign_hop`].
//!
//! # What is counted
//!
//! [`CoCost`]: programs and ops run, side rows read (only for sides whose
//! `Rel` is not `All`), and domain slots scanned. A side with `Rel::All`
//! costs nothing beyond the star path: its presence plane and offsets are
//! read as prepared.

use lance_graph_mask_risc::ExecError;
use lance_graph_mask_risc::{
    Foreign, GroupFold, GroupKey, LaneRef, Operand, Out, Planes, Program, Scratch, Terminal,
    execute_into, words_for,
};
use lance_graph_relalg::{Measure, Query, Rel, RelError, Run, normalize, run_with_program};
use mork_pathmap_probe::RelationView;
use mork_pathmap_probe::join::{JoinDomain, JoinRefusal, JoinTable, Side, for_each_set};

use crate::native::{HeadTerm, SideKey, StarPlan, Store, fingerprint};

/// The lane a side's `Rel` names: the side's other column.
pub const VALUE_LANE: u16 = 0;

/// One side: the laid-out rows and their slot lane.
#[derive(Clone, Debug)]
pub struct CoSide {
    pub side: Side,
    /// `slot[j]`: the slot row `j` sits in. Built from `side.offsets()`, so
    /// non-decreasing.
    slot: Vec<u32>,
}

impl CoSide {
    pub fn new(side: Side) -> CoSide {
        let off = side.offsets();
        let mut slot = Vec::with_capacity(side.positions().len());
        for s in 0..off.len() - 1 {
            for _ in off[s]..off[s + 1] {
                slot.push(s as u32);
            }
        }
        CoSide { side, slot }
    }

    pub fn n_rows(&self) -> usize {
        self.side.positions().len()
    }

    /// Rows of this side at slot `s`, as a row range.
    fn run(&self, s: usize) -> std::ops::Range<usize> {
        let off = self.side.offsets();
        off[s] as usize..off[s + 1] as usize
    }
}

/// One domain and the sides laid out over it, without restrictions.
///
/// The fields are private: [`CoTable::prepare`] is the only way to build one,
/// so every side is laid out over this table's domain. Equal slot numbers over
/// two different domains name different centers, and nothing in [`eval`]
/// could tell them apart.
pub struct CoTable {
    domain: JoinDomain,
    sides: Vec<CoSide>,
    /// The sides' keys, with `restrict: None`: a restriction is a query-time
    /// `Rel` here, not part of the layout.
    keys: Vec<SideKey>,
    domain_fp: u64,
}

/// `key` without its restriction.
pub fn layout_key(key: SideKey) -> SideKey {
    SideKey {
        restrict: None,
        ..key
    }
}

impl CoTable {
    /// Lays out every side of `plan` from all its rows, ignoring restrictions.
    pub fn prepare(plan: &StarPlan, store: &Store) -> CoTable {
        let inputs: Vec<(&RelationView, usize)> = plan
            .sides
            .iter()
            .map(|s| (store.view(s.key.rel), s.key.key_col as usize))
            .collect();
        let table = JoinTable::prepare_star(&inputs);
        CoTable {
            domain_fp: fingerprint(&table),
            keys: plan.sides.iter().map(|s| layout_key(s.key)).collect(),
            sides: table.sides.into_iter().map(CoSide::new).collect(),
            domain: table.domain,
        }
    }

    pub fn domain(&self) -> &JoinDomain {
        &self.domain
    }

    pub fn sides(&self) -> &[CoSide] {
        &self.sides
    }

    pub fn keys(&self) -> &[SideKey] {
        &self.keys
    }

    /// Fingerprint of the domain every side is laid out over
    /// ([`crate::native::fingerprint`]).
    pub fn domain_fp(&self) -> u64 {
        self.domain_fp
    }

    pub fn rows(&self) -> usize {
        self.sides.iter().map(CoSide::n_rows).sum()
    }
}

/// The per-side `Rel`s that express `plan`'s restrictions: `value == k`, or
/// `Empty` for a constant no `u32` value can equal; `All` when unrestricted.
pub fn plan_rels(plan: &StarPlan) -> Vec<Rel> {
    plan.sides
        .iter()
        .map(|s| match s.key.restrict {
            _ if s.key.unary => Rel::All,
            None => Rel::All,
            Some(k) => match u32::try_from(k) {
                Ok(k) => Rel::eq_u32(VALUE_LANE, k),
                Err(_) => Rel::Empty,
            },
        })
        .collect()
}

/// What the operator cost, counted from what it ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoCost {
    /// mask-risc programs executed.
    pub programs: usize,
    /// mask-risc ops executed.
    pub passes: usize,
    /// Sides whose `Rel` was not answered from the description alone.
    pub sides_filtered: usize,
    /// Rows of those sides read (each read twice: the value lane by the fold,
    /// the slot lane by the projection).
    pub side_rows_read: usize,
    /// Domain slots scanned to turn counts into planes.
    pub slots_scanned: usize,
}

/// Why the operator did not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoRefusal {
    /// The table holds other sides than the plan reads.
    NoCoordinateWitness,
    /// One `Rel` per side is needed.
    Rels {
        sides: usize,
        rels: usize,
    },
    /// One generation per side is needed.
    Generations {
        sides: usize,
        given: usize,
    },
    /// A side is stale.
    Table(JoinRefusal),
    Relalg(RelError),
    Exec(ExecError),
}

/// Rows of one side the `Rel` selected.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selected {
    /// Every row (`Rel` normalized to `All`): read the side as prepared.
    All,
    /// The rows in a mask over the side's rows, and the count per slot.
    Some { rows: Vec<u64>, counts: Vec<i64> },
}

/// The operator's result: the joint slots, each side's selection, and cost.
pub struct CoEval {
    pub joint: Vec<u64>,
    selected: Vec<Selected>,
    pub cost: CoCost,
    /// Every mask-risc program the operator ran.
    pub programs: Vec<Program>,
}

/// Runs `rels[i]` over side `i` of `table` and intersects the sides on the
/// shared domain. Refused unless the table holds exactly `plan`'s sides at
/// the generations in `current`.
pub fn eval(
    table: &CoTable,
    plan: &StarPlan,
    rels: &[Rel],
    current: &[u64],
) -> Result<CoEval, CoRefusal> {
    let wanted: Vec<SideKey> = plan.sides.iter().map(|s| layout_key(s.key)).collect();
    if wanted != table.keys {
        return Err(CoRefusal::NoCoordinateWitness);
    }
    let k = table.sides.len();
    if rels.len() != k {
        return Err(CoRefusal::Rels {
            sides: k,
            rels: rels.len(),
        });
    }
    if current.len() != k {
        return Err(CoRefusal::Generations {
            sides: k,
            given: current.len(),
        });
    }
    for (i, (s, &now)) in table.sides.iter().zip(current).enumerate() {
        if s.side.generation != now {
            return Err(CoRefusal::Table(JoinRefusal::Stale {
                side: i,
                bound: s.side.generation,
                current: now,
            }));
        }
    }
    let n = table.domain.len();
    let words = words_for(n);
    let mut cost = CoCost::default();
    let mut programs = Vec::new();
    let mut selected = Vec::with_capacity(k);
    let mut planes: Vec<Vec<u64>> = Vec::with_capacity(k);
    for (s, rel) in table.sides.iter().zip(rels) {
        let (sel, plane) = select(s, rel, n, &mut cost, &mut programs)?;
        selected.push(sel);
        planes.push(plane);
    }
    let masks: Vec<&[u64]> = table
        .sides
        .iter()
        .zip(&planes)
        .zip(&selected)
        .map(|((s, p), sel)| match sel {
            Selected::All => s.side.present(),
            Selected::Some { .. } => p.as_slice(),
        })
        .collect();
    let view = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let rel = Rel::And((0..k as u16).map(Rel::Plane).collect());
    let mut joint = vec![0u64; words];
    let (run, program) = run_with_program(
        &Query {
            rel,
            measure: Measure::Keep,
        },
        &view,
        Some(&mut joint),
    )
    .map_err(CoRefusal::Relalg)?;
    add_run(&mut cost, run, program, &mut programs);
    Ok(CoEval {
        joint,
        selected,
        cost,
        programs,
    })
}

fn add_run(cost: &mut CoCost, run: Run, program: Option<Program>, programs: &mut Vec<Program>) {
    cost.programs += run.programs;
    cost.passes += run.passes;
    if let Some(p) = program {
        programs.push(p);
    }
}

/// One side's selection and, when filtered, its slot plane.
fn select(
    s: &CoSide,
    rel: &Rel,
    n: usize,
    cost: &mut CoCost,
    programs: &mut Vec<Program>,
) -> Result<(Selected, Vec<u64>), CoRefusal> {
    let rel = normalize(rel);
    if rel == Rel::All {
        return Ok((Selected::All, Vec::new()));
    }
    let rows_n = s.n_rows();
    let row_words = words_for(rows_n);
    let mut rows = vec![0u64; row_words];
    let mut counts = vec![0i64; n];
    let mut plane = vec![0u64; words_for(n)];
    if rel != Rel::Empty && rows_n > 0 {
        // The restriction runs over the relation's lane in place, in row
        // order; side row `j` then reads its bit at `positions()[j]`.
        let lane = s.side.lane();
        let mut lane_rows = vec![0u64; words_for(lane.len())];
        let lanes = [LaneRef::U32(lane)];
        let view = Planes {
            n_rows: lane.len(),
            masks: &[],
            lanes: &lanes,
        };
        let (run, program) = run_with_program(
            &Query {
                rel,
                measure: Measure::Keep,
            },
            &view,
            Some(&mut lane_rows),
        )
        .map_err(CoRefusal::Relalg)?;
        add_run(cost, run, program, programs);
        for (j, &p) in s.side.positions().iter().enumerate() {
            let p = p as usize;
            rows[j >> 6] |= (lane_rows[p >> 6] >> (p & 63) & 1) << (j & 63);
        }
        project(s, &rows, &mut counts, cost, programs)?;
        cost.sides_filtered += 1;
        cost.side_rows_read += rows_n;
        cost.slots_scanned += n;
        for (slot, &c) in counts.iter().enumerate() {
            if c > 0 {
                plane[slot >> 6] |= 1 << (slot & 63);
            }
        }
    }
    Ok((Selected::Some { rows, counts }, plane))
}

/// `counts[slot] = |{ selected rows in slot's run }|`, one mask-risc
/// `GroupReduce Count` keyed by the side's own slot lane.
fn project(
    s: &CoSide,
    rows: &[u64],
    counts: &mut [i64],
    cost: &mut CoCost,
    programs: &mut Vec<Program>,
) -> Result<(), CoRefusal> {
    let program = Program::new(
        Vec::new(),
        Terminal::GroupReduce {
            mask: Operand::Plane(0),
            key: GroupKey::Lane(0),
            fold: GroupFold::Count,
        },
    );
    let masks = [rows];
    let lanes = [LaneRef::U32(&s.slot)];
    let view = Planes {
        n_rows: s.n_rows(),
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::new(
        words_for(s.n_rows()),
        (program.scratch_slots as usize).max(1),
    );
    execute_into(
        &program,
        &view,
        &Foreign::NONE,
        &mut scratch,
        Out::I64(counts),
    )
    .map_err(CoRefusal::Exec)?;
    cost.programs += 1;
    cost.passes += program.ops.len() + 1;
    programs.push(program);
    Ok(())
}

impl CoEval {
    /// Selected rows of side `i` at slot `s`.
    fn mult(&self, table: &CoTable, i: usize, s: usize) -> u64 {
        match &self.selected[i] {
            Selected::All => table.sides[i].run(s).len() as u64,
            Selected::Some { counts, .. } => counts[s] as u64,
        }
    }

    /// Selected values of side `i` at slot `s`, in row order.
    fn values_at<'a>(
        &'a self,
        table: &'a CoTable,
        i: usize,
        s: usize,
    ) -> impl Iterator<Item = u32> + 'a {
        let side = &table.sides[i];
        let rows = match &self.selected[i] {
            Selected::All => None,
            Selected::Some { rows, .. } => Some(rows.as_slice()),
        };
        side.run(s).filter_map(move |j| {
            let keep = rows.is_none_or(|m| m[j >> 6] >> (j & 63) & 1 == 1);
            keep.then(|| side.side.lane()[side.side.positions()[j] as usize])
        })
    }

    /// Distinct center values every side has a selected row at.
    pub fn distinct_centers(&self) -> u64 {
        self.joint.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// Bag answers of the star, `Σ_slot Π_i mult_i(slot)`: a co-addressed dot
    /// product of the per-side counts, without enumerating a row.
    pub fn bag_count(&self, table: &CoTable) -> u64 {
        let mut total = 0u64;
        for_each_set(&self.joint, table.domain.len(), |s| {
            total += (0..table.sides.len())
                .map(|i| self.mult(table, i, s))
                .product::<u64>();
        });
        total
    }

    /// Derivations whose head reads as `pattern` (`None` at a column: any
    /// value). A fold over the joint, never an enumeration: per slot the
    /// product of each side's selected rows, narrowed to the pattern's value
    /// where it pins that side. A pinned center is inverted to its one slot.
    /// With an all-`None` pattern this is [`CoEval::bag_count`].
    pub fn matching(
        &self,
        table: &CoTable,
        plan: &StarPlan,
        pattern: impl Fn(usize) -> Option<u64>,
    ) -> u64 {
        let k = table.sides.len();
        let mut center: Option<u64> = None;
        let mut want: Vec<Option<u64>> = vec![None; k];
        for (c, h) in plan.head.iter().enumerate() {
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
        let at = |s: usize| -> u64 {
            (0..k)
                .map(|i| match want[i] {
                    None => self.mult(table, i, s),
                    Some(t) => self
                        .values_at(table, i, s)
                        .filter(|&v| u64::from(v) == t)
                        .count() as u64,
                })
                .product()
        };
        if let Some(v) = center {
            let Some(s) = u32::try_from(v).ok().and_then(|y| table.domain.slot_of(y)) else {
                return 0;
            };
            // A slot outside the joint has a side with no selected row, so
            // its product is already 0.
            return at(s);
        }
        let mut total = 0u64;
        for_each_set(&self.joint, table.domain.len(), |s| total += at(s));
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{RelId, VarId};
    use crate::native::{self, SidePlan, has_foreign_hop};

    const R: RelId = RelId(0);
    const S: RelId = RelId(1);
    const T: RelId = RelId(2);

    /// `Out(x, c, z) <- R(x, c), S(c, z), T(c, k)`: three binary sides on `c`.
    fn plan(r: Option<u64>, s: Option<u64>, t: Option<u64>) -> StarPlan {
        let side = |rel, key_col, restrict, emit| SidePlan {
            key: SideKey {
                rel,
                key_col,
                restrict,
                unary: false,
            },
            emit,
        };
        StarPlan {
            center: VarId(1),
            sides: vec![
                side(R, 1, r, r.is_none().then_some(VarId(0))),
                side(S, 0, s, s.is_none().then_some(VarId(2))),
                side(T, 0, t, None),
            ],
            head: [
                r.is_none().then_some(HeadTerm::Side(0)),
                Some(HeadTerm::Center),
                s.is_none().then_some(HeadTerm::Side(1)),
            ]
            .into_iter()
            .flatten()
            .collect(),
        }
    }

    /// Deterministic rows: `n` rows, both columns below `m`.
    fn rows(seed: u64, n: usize, m: u32) -> Vec<[u32; 2]> {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % u64::from(m)) as u32
        };
        (0..n).map(|_| [next(), next()]).collect()
    }

    struct Fixture {
        r: Vec<[u32; 2]>,
        s: Vec<[u32; 2]>,
        t: Vec<[u32; 2]>,
    }

    impl Fixture {
        fn new() -> Self {
            Fixture {
                r: rows(0x9E37_79B9, 600, 40),
                s: rows(0x7F4A_7C15, 500, 40),
                t: rows(0x2545_F491, 400, 40),
            }
        }
        fn store(&self) -> Store {
            Store::from_rows([(R, &self.r, 1), (S, &self.s, 2), (T, &self.t, 3)])
        }
    }

    /// Nested loops over the rows: the derivations whose head reads as
    /// `pattern`, counted. Pinned columns prune each loop as soon as the
    /// values they read are known. Builds no row.
    fn brute(
        f: &Fixture,
        p: &StarPlan,
        keep: [&dyn Fn(u32) -> bool; 3],
        pattern: &[Option<u64>],
    ) -> u64 {
        // Whether the head columns readable from `known` agree with `pattern`.
        let agrees = |known: &dyn Fn(HeadTerm) -> Option<u64>| {
            p.head
                .iter()
                .zip(pattern)
                .all(|(&h, &want)| want.is_none_or(|w| known(h).is_none_or(|v| v == w)))
        };
        let mut n = 0u64;
        for r in &f.r {
            let c = r[1];
            let kr = |h: HeadTerm| match h {
                HeadTerm::Const(k) => Some(k),
                HeadTerm::Center => Some(u64::from(c)),
                HeadTerm::Side(0) => Some(u64::from(r[0])),
                HeadTerm::Side(_) => None,
            };
            if !keep[0](r[0]) || !agrees(&kr) {
                continue;
            }
            for s in &f.s {
                let ks = |h: HeadTerm| match h {
                    HeadTerm::Side(1) => Some(u64::from(s[1])),
                    h => kr(h),
                };
                if s[0] != c || !keep[1](s[1]) || !agrees(&ks) {
                    continue;
                }
                for t in &f.t {
                    let kt = |h: HeadTerm| match h {
                        HeadTerm::Side(2) => Some(u64::from(t[1])),
                        h => ks(h),
                    };
                    if t[0] == c && keep[2](t[1]) && agrees(&kt) {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// The star path's view of a prepared star with no post filter.
    fn star_coordinates(star: &native::PreparedStar, p: &StarPlan) -> native::RuleCoordinates {
        let (kept, _, _) = native::kept(star, p, &GENS).expect("star");
        native::RuleCoordinates {
            rule: 0,
            table: 0,
            sides: (0..p.sides.len()).collect(),
            star: p.clone(),
            filter: native::RuleFilter::of(p, &vec![None; p.head.len()]).expect("no filter"),
            kept,
        }
    }

    const GENS: [u64; 3] = [1, 2, 3];

    /// The operator over one unrestricted table equals the star path prepared
    /// with each restriction, for every restriction combination tried, in
    /// bag count, per-derivation multiplicities and distinct centers.
    #[test]
    fn parity_with_the_star_path() {
        let f = Fixture::new();
        let store = f.store();
        let table = CoTable::prepare(&plan(None, None, None), &store);
        let mut nonempty = 0;
        for (r, s, t) in [
            (None, None, None),
            (Some(3), None, None),
            (None, Some(7), Some(11)),
            (Some(5), Some(9), Some(13)),
            (Some(u64::from(u32::MAX) + 1), None, None),
        ] {
            let p = plan(r, s, t);
            let ev = eval(&table, &p, &plan_rels(&p), &GENS).expect("eval");
            let star = native::prepare(&p, &store);
            let rc = star_coordinates(&star, &p);
            let sides: Vec<&Side> = star.table.sides.iter().collect();
            let view = native::StarView {
                domain: &star.table.domain,
                sides: &sides,
                star: &p,
            };
            let total = native::matching(&view, &rc, |_| None);
            assert_eq!(ev.bag_count(&table), total, "{r:?} {s:?} {t:?}");
            assert_eq!(ev.matching(&table, &p, |_| None), total);
            // Every derivation the star path reads, pinned in full, has the
            // same multiplicity here; with equal totals the bags are equal.
            native::for_each_derivation(&view, &rc, |d, _| {
                let pin = |c: usize| Some(d.get(c));
                assert_eq!(
                    ev.matching(&table, &p, pin),
                    native::matching(&view, &rc, pin),
                    "{r:?} {s:?} {t:?}"
                );
            });
            assert_eq!(
                ev.distinct_centers(),
                native::distinct_centers(&star, &p, &GENS).expect("star")
            );
            nonempty += usize::from(total > 0);
        }
        // Not vacuous: most cases have answers, so equality compares rows.
        assert!(nonempty >= 4, "only {nonempty} cases had answers");
    }

    /// A restriction the star path cannot express (`Or`, `Not`) matches the
    /// nested-loop reference.
    #[test]
    fn disjunctive_restrictions_match_the_reference() {
        let f = Fixture::new();
        let store = f.store();
        let p = plan(None, None, None);
        let table = CoTable::prepare(&p, &store);
        let rels = vec![
            Rel::Or(vec![Rel::eq_u32(VALUE_LANE, 1), Rel::eq_u32(VALUE_LANE, 2)]),
            Rel::Not(Box::new(Rel::eq_u32(VALUE_LANE, 4))),
            Rel::ne_u32(VALUE_LANE, 0),
        ];
        let ev = eval(&table, &p, &rels, &GENS).expect("eval");
        let keep: [&dyn Fn(u32) -> bool; 3] = [&|x| x == 1 || x == 2, &|z| z != 4, &|k| k != 0];
        let total = brute(&f, &p, keep, &[None; 3]);
        assert!(total > 0, "fixture must have answers");
        assert_eq!(ev.bag_count(&table), total);
        // Per center, and per pinned (x, c): each count equals the oracle's.
        let mut pinned = 0;
        for c in 0..40u64 {
            assert_eq!(
                ev.matching(&table, &p, |i| (i == 1).then_some(c)),
                brute(&f, &p, keep, &[None, Some(c), None]),
                "c = {c}"
            );
            for x in [1u64, 2, 3] {
                let want = brute(&f, &p, keep, &[Some(x), Some(c), None]);
                let pat = [Some(x), Some(c), None];
                assert_eq!(ev.matching(&table, &p, |i| pat[i]), want, "x={x} c={c}");
                pinned += usize::from(want > 0);
            }
        }
        // x = 3 is excluded by the restriction; 1 and 2 must match somewhere.
        assert!(pinned > 0, "pinned patterns must find answers");
        assert_eq!(ev.cost.sides_filtered, 3);
        assert_eq!(ev.cost.side_rows_read, table.rows());
    }

    /// No program the operator runs reads at a position taken from another
    /// operand's values, and with a filtered side it does run programs.
    #[test]
    fn no_gather() {
        let f = Fixture::new();
        let store = f.store();
        let p = plan(Some(3), Some(7), None);
        let table = CoTable::prepare(&plan(None, None, None), &store);
        let ev = eval(&table, &p, &plan_rels(&p), &GENS).expect("eval");
        // Two filtered sides: a fold and a projection each, plus the joint.
        assert_eq!(ev.programs.len(), 5, "{:?}", ev.programs);
        for prog in &ev.programs {
            assert!(!has_foreign_hop(prog), "foreign hop in {prog:?}");
        }
    }

    /// Unrestricted sides cost nothing past the star path: one program, the
    /// joint, and no side row read.
    #[test]
    fn all_sides_read_as_prepared() {
        let f = Fixture::new();
        let store = f.store();
        let p = plan(None, None, None);
        let table = CoTable::prepare(&p, &store);
        let ev = eval(&table, &p, &plan_rels(&p), &GENS).expect("eval");
        assert_eq!(ev.cost.side_rows_read, 0);
        assert_eq!(ev.cost.sides_filtered, 0);
        assert_eq!(ev.programs.len(), 1);
    }

    #[test]
    fn refusals() {
        let f = Fixture::new();
        let store = f.store();
        let p = plan(Some(3), None, None);
        let table = CoTable::prepare(&p, &store);
        let rels = plan_rels(&p);
        assert!(matches!(
            eval(&table, &p, &rels, &[1, 2, 4]),
            Err(CoRefusal::Table(JoinRefusal::Stale { side: 2, .. }))
        ));
        assert_eq!(
            eval(&table, &p, &rels, &[1, 2]).err(),
            Some(CoRefusal::Generations { sides: 3, given: 2 })
        );
        assert_eq!(
            eval(&table, &p, &rels[..2], &GENS).err(),
            Some(CoRefusal::Rels { sides: 3, rels: 2 })
        );
        let mut other = p.clone();
        other.sides[2].key.rel = R;
        assert_eq!(
            eval(&table, &other, &rels, &GENS).err(),
            Some(CoRefusal::NoCoordinateWitness)
        );
        // A restriction is not part of the layout: the same table answers it.
        assert!(eval(&table, &plan(None, Some(1), None), &rels, &GENS).is_ok());
    }
}
