//! D-CMR-2: lowering of the contract's set algebra onto mask-risc.
//!
//! A [`SetExpr`] (`lance_graph_contract::set_algebra`) says WHAT rows are
//! selected. This module decides HOW mask-risc reads them, in the normal form
//! `Runs ∩ residual`:
//!
//! - **Runs** come from comparisons on an attested sorted lane (a
//!   [`SortedBasis`]): each comparison is one counted seek, never a scan. That
//!   is the same-table, attested-key seek the 2026-10-20 ADMISSION ruling
//!   (option B) admits in preparation.
//! - **The residual** is everything else. It is compiled once into a
//!   mask-risc [`Program`] and run over each run as an execution extent.
//!
//! Three measurement levels, cheapest first (plan
//! `d-cmr-concept-chain-normalization-v1.md` §3.4); [`SetPlan`] picks the
//! first that applies:
//!
//! | level | when | what runs |
//! |---|---|---|
//! | [`Level::Geometry`] | no residual | `Count = Σ(hi − lo)`; no plane, lane or extent is read |
//! | [`Level::Optics`] | residual, a measure is wanted | the residual program over each run as an extent (`execute_extent`, `Count` terminal) |
//! | [`Level::Etch`] | the set itself is wanted | the residual program with a `Keep` terminal over each run; bits outside the runs are zero |
//!
//! **Known limit, stated rather than hidden:** mask-risc's fused paths decline
//! a `Pred` (`Program::fused_ternlog`, `ir.rs`), so a residual comparison runs
//! tiled and writes per-tile scratch (at most `TILE_WORDS` words, never a
//! population-sized mask). A scratch-free predicate fold is a mask-risc change
//! (D-CMR-2b), not part of this lowering.
//!
//! Costs are reported apart, as the ADMISSION ruling requires: the attestation
//! scan ([`SortedBasis::attest_reads`]), the preparation seeks
//! ([`SetPlan::seeks`]) and the query's extents ([`Measured`]).
//!
//! Not here: joins, gathers, positional addressing across tables, preimage
//! (D-CMR-5), two-dimensional coordinates (D-CMR-4).

use lance_graph_contract::set_algebra::{normalize, Cmp, Res, Runs, SetExpr};
use lance_graph_mask_risc::{
    execute_extent, words_for, ExecError, Foreign, MaskOp, Operand, Out, Planes, Pred, Program,
    Scratch, Terminal, Value,
};

/// Why a set could not be lowered or measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The basis lane is not non-decreasing in row order; `row` is the first
    /// row whose key is smaller than its predecessor's.
    BasisNotSorted { row: usize },
    /// The basis has a different number of rows than the plan.
    BasisRows { basis: usize, plan: usize },
    /// The planes span a different number of rows than the plan.
    PlaneRows { planes: usize, plan: usize },
    /// The `Keep` destination is not `words_for(n_rows)` long.
    OutLen { got: usize, want: usize },
    /// A `Sym` id does not fit a mask-risc plane index.
    SymOutOfRange(u32),
    /// The residual program needs more scratch slots than mask-risc addresses.
    TooManySlots,
    /// mask-risc refused the residual program.
    Exec(ExecError),
}

impl From<ExecError> for LowerError {
    fn from(e: ExecError) -> Self {
        LowerError::Exec(e)
    }
}

/// An attested sorted `i32` lane: keys are non-decreasing in row order, so a
/// comparison on this lane selects one contiguous row range (or, for `!=`,
/// at most two), found by binary search.
///
/// Attestation is one sequential scan of the keys, counted in
/// [`SortedBasis::attest_reads`]; it is preparation cost, reported apart from
/// seeks and query reads. The keys must be the same values the planes carry
/// for `lane` (`Planes::lanes[lane]`), because a basis comparison that stays
/// in a residual (inside a mixed `Or`) is evaluated on the planes.
#[derive(Debug, Clone, Copy)]
pub struct SortedBasis<'a> {
    lane: u16,
    keys: &'a [i32],
}

impl<'a> SortedBasis<'a> {
    /// Attest `keys` as the sorted basis for `lane`, or refuse with the first
    /// row out of order.
    pub fn attest(lane: u16, keys: &'a [i32]) -> Result<Self, LowerError> {
        if let Some(i) = keys.windows(2).position(|w| w[1] < w[0]) {
            return Err(LowerError::BasisNotSorted { row: i + 1 });
        }
        Ok(SortedBasis { lane, keys })
    }

    /// The lane this basis describes.
    pub fn lane(&self) -> u16 {
        self.lane
    }

    /// Keys read by the attestation scan (one per row).
    pub fn attest_reads(&self) -> u64 {
        self.keys.len() as u64
    }

    /// The rows where `key op t`, as runs. Every boundary is one binary
    /// search (one seek, counted in `seeks`).
    fn rows(&self, op: Cmp, t: i32, seeks: &mut u64) -> Runs {
        let n = self.keys.len() as u64;
        let mut seek = |strict: bool| {
            *seeks += 1;
            if strict {
                self.keys.partition_point(|&k| k < t) as u64
            } else {
                self.keys.partition_point(|&k| k <= t) as u64
            }
        };
        match op {
            Cmp::Lt => Runs::from_intervals([(0, seek(true))]),
            Cmp::Ge => Runs::from_intervals([(seek(true), n)]),
            Cmp::Le => Runs::from_intervals([(0, seek(false))]),
            Cmp::Gt => Runs::from_intervals([(seek(false), n)]),
            Cmp::Eq => {
                let lo = seek(true);
                Runs::from_intervals([(lo, seek(false))])
            }
            Cmp::Ne => {
                let lo = seek(true);
                let hi = seek(false);
                Runs::from_intervals([(0, lo), (hi, n)])
            }
        }
    }
}

/// Which measurement level a plan uses for a measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Answered from the runs alone; nothing is read.
    Geometry,
    /// The residual program runs over each run, producing a measure.
    Optics,
    /// The residual program runs over each run, writing the set.
    Etch,
}

/// A measure plus what producing it cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measured<T> {
    pub value: T,
    pub level: Level,
    /// Extents the residual program was executed over (0 at geometry).
    pub extents: u64,
    /// Rows those extents span in total (0 at geometry).
    pub extent_rows: u64,
}

/// A [`SetExpr`] split into `Runs ∩ residual` over a population of `n_rows`,
/// ready to measure against [`Planes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetPlan {
    n_rows: usize,
    runs: Runs,
    /// `None` when the runs alone are the set (no residual).
    residual: Option<SetExpr>,
    seeks: u64,
}

impl SetPlan {
    /// Normalize `e` (the contract's [`normalize`]) and split it. Basis
    /// comparisons become runs; the rest becomes the residual. Without a
    /// basis the runs are the whole population.
    pub fn new(
        e: &SetExpr,
        n_rows: usize,
        basis: Option<&SortedBasis<'_>>,
    ) -> Result<SetPlan, LowerError> {
        if let Some(b) = basis {
            if b.keys.len() != n_rows {
                return Err(LowerError::BasisRows {
                    basis: b.keys.len(),
                    plan: n_rows,
                });
            }
        }
        let n = normalize(e);
        let mut seeks = 0;
        let full = Runs::full(n_rows as u64);
        let (runs, residual) = match exact_runs(&n, basis, n_rows, &mut seeks) {
            Some(r) => (r, None),
            None => match &n {
                SetExpr::And(cs) => {
                    let mut runs = full;
                    let mut rest = Vec::new();
                    for c in cs {
                        match exact_runs(c, basis, n_rows, &mut seeks) {
                            Some(r) => runs = runs.intersect(&r),
                            None => rest.push(c.clone()),
                        }
                    }
                    let residual = match rest.len() {
                        0 => None,
                        1 => rest.pop(),
                        _ => Some(SetExpr::And(rest)),
                    };
                    (runs, residual)
                }
                other => (full, Some(other.clone())),
            },
        };
        Ok(SetPlan {
            n_rows,
            runs,
            residual,
            seeks,
        })
    }

    /// `e` with no split at all: one run over the whole population and the
    /// normalized expression as the residual. The disable arm of the
    /// geometry level; it must give the same answers.
    pub fn unsplit(e: &SetExpr, n_rows: usize) -> SetPlan {
        let n = normalize(e);
        let (runs, residual) = match n {
            SetExpr::Empty => (Runs::empty(), None),
            SetExpr::All => (Runs::full(n_rows as u64), None),
            other => (Runs::full(n_rows as u64), Some(other)),
        };
        SetPlan {
            n_rows,
            runs,
            residual,
            seeks: 0,
        }
    }

    /// The runs the residual is evaluated over.
    pub fn runs(&self) -> &Runs {
        &self.runs
    }

    /// The residual, if any.
    pub fn residual(&self) -> Option<&SetExpr> {
        self.residual.as_ref()
    }

    /// Basis seeks spent building the runs (preparation cost).
    pub fn seeks(&self) -> u64 {
        self.seeks
    }

    /// The level a measure (not `Keep`) is answered at.
    pub fn measure_level(&self) -> Level {
        if self.residual.is_none() || self.runs.is_empty() {
            Level::Geometry
        } else {
            Level::Optics
        }
    }

    /// The number of selected rows.
    pub fn count(&self, planes: &Planes<'_>) -> Result<Measured<u64>, LowerError> {
        self.check_planes(planes)?;
        let residual = match (&self.residual, self.runs.is_empty()) {
            (Some(r), false) => r,
            _ => {
                return Ok(Measured {
                    value: self.runs.count(),
                    level: Level::Geometry,
                    extents: 0,
                    extent_rows: 0,
                })
            }
        };
        let program = compile(residual, Terminal::Count { mask: Operand::Plane(0) })?;
        let mut scratch = Scratch::for_program(&program, planes.n_rows)?;
        let mut total = 0u64;
        for &(lo, hi) in self.runs.intervals() {
            let v = execute_extent(
                &program,
                planes,
                &Foreign::NONE,
                &mut scratch,
                Out::None,
                lo as usize..hi as usize,
            )?;
            if let Value::Count(c) = v {
                total += c as u64;
            }
        }
        Ok(Measured {
            value: total,
            level: Level::Optics,
            extents: self.runs.intervals().len() as u64,
            extent_rows: self.runs.count(),
        })
    }

    /// Write the selected set into `out` (`words_for(n_rows)` words). Every
    /// bit outside the selection is cleared first.
    pub fn keep(&self, planes: &Planes<'_>, out: &mut [u64]) -> Result<Measured<()>, LowerError> {
        self.check_planes(planes)?;
        let want = words_for(self.n_rows);
        if out.len() != want {
            return Err(LowerError::OutLen {
                got: out.len(),
                want,
            });
        }
        out.fill(0);
        let residual = match (&self.residual, self.runs.is_empty()) {
            (Some(r), false) => r,
            _ => {
                for &(lo, hi) in self.runs.intervals() {
                    set_bits(out, lo as usize, hi as usize);
                }
                return Ok(Measured {
                    value: (),
                    level: Level::Geometry,
                    extents: 0,
                    extent_rows: 0,
                });
            }
        };
        let program = compile(residual, Terminal::Keep { mask: Operand::Plane(0) })?;
        let mut scratch = Scratch::for_program(&program, planes.n_rows)?;
        for &(lo, hi) in self.runs.intervals() {
            execute_extent(
                &program,
                planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(out),
                lo as usize..hi as usize,
            )?;
        }
        Ok(Measured {
            value: (),
            level: Level::Etch,
            extents: self.runs.intervals().len() as u64,
            extent_rows: self.runs.count(),
        })
    }

    fn check_planes(&self, planes: &Planes<'_>) -> Result<(), LowerError> {
        if planes.n_rows != self.n_rows {
            return Err(LowerError::PlaneRows {
                planes: planes.n_rows,
                plan: self.n_rows,
            });
        }
        Ok(())
    }
}

/// The runs `e` selects, when `e` is built only from comparisons on the
/// basis lane (and `All` / `Empty`); `None` otherwise.
fn exact_runs(
    e: &SetExpr,
    basis: Option<&SortedBasis<'_>>,
    n_rows: usize,
    seeks: &mut u64,
) -> Option<Runs> {
    match e {
        SetExpr::All => Some(Runs::full(n_rows as u64)),
        SetExpr::Empty => Some(Runs::empty()),
        SetExpr::Res(Res::I32 { lane, op, t }) => {
            let b = basis.filter(|b| b.lane == *lane)?;
            Some(b.rows(*op, *t, seeks))
        }
        SetExpr::And(cs) | SetExpr::Or(cs) => {
            // Decide purity first, so a mixed junction spends no seeks.
            if !cs.iter().all(|c| is_basis_only(c, basis)) {
                return None;
            }
            let mut parts = cs
                .iter()
                .map(|c| exact_runs(c, basis, n_rows, seeks).expect("checked basis-only"));
            let first = parts.next()?;
            Some(match e {
                SetExpr::And(_) => parts.fold(first, |a, b| a.intersect(&b)),
                _ => parts.fold(first, |a, b| a.union(&b)),
            })
        }
        _ => None,
    }
}

fn is_basis_only(e: &SetExpr, basis: Option<&SortedBasis<'_>>) -> bool {
    match e {
        SetExpr::All | SetExpr::Empty => true,
        SetExpr::Res(Res::I32 { lane, .. }) => basis.is_some_and(|b| b.lane == *lane),
        SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().all(|c| is_basis_only(c, basis)),
        _ => false,
    }
}

fn to_pred(r: Res) -> Pred {
    match r {
        Res::I32 { lane, op, t } => match op {
            Cmp::Lt => Pred::LtI32 { lane, t },
            Cmp::Le => Pred::LeI32 { lane, t },
            Cmp::Gt => Pred::GtI32 { lane, t },
            Cmp::Ge => Pred::GeI32 { lane, t },
            Cmp::Eq => Pred::EqI32 { lane, v: t },
            Cmp::Ne => Pred::NeI32 { lane, v: t },
        },
        Res::U32 { lane, eq: true, v } => Pred::EqU32 { lane, v },
        Res::U32 { lane, eq: false, v } => Pred::NeU32 { lane, v },
    }
}

/// Compile a residual into one program whose terminal reads its result.
fn compile(residual: &SetExpr, terminal: Terminal) -> Result<Program, LowerError> {
    let mut emit = Emit { ops: Vec::new(), next: 0 };
    let result = emit.expr(residual)?;
    let terminal = match terminal {
        Terminal::Count { .. } => Terminal::Count { mask: result },
        Terminal::Keep { .. } => Terminal::Keep { mask: result },
        other => other,
    };
    Ok(Program::new(emit.ops, terminal))
}

struct Emit {
    ops: Vec<MaskOp>,
    next: u16,
}

impl Emit {
    fn slot(&mut self) -> Result<u16, LowerError> {
        let s = self.next;
        self.next = self.next.checked_add(1).ok_or(LowerError::TooManySlots)?;
        Ok(s)
    }

    fn expr(&mut self, e: &SetExpr) -> Result<Operand, LowerError> {
        Ok(match e {
            SetExpr::Sym(i) => {
                Operand::Plane(u16::try_from(*i).map_err(|_| LowerError::SymOutOfRange(*i))?)
            }
            SetExpr::Res(r) => {
                let dst = self.slot()?;
                self.ops.push(MaskOp::Pred {
                    pred: to_pred(*r),
                    under: None,
                    dst,
                });
                Operand::Scratch(dst)
            }
            SetExpr::Not(x) => {
                let a = self.expr(x)?;
                let dst = self.slot()?;
                self.ops.push(MaskOp::Not { a, dst });
                Operand::Scratch(dst)
            }
            SetExpr::And(cs) | SetExpr::Or(cs) => {
                let and = matches!(e, SetExpr::And(_));
                let mut it = cs.iter();
                let mut acc = match it.next() {
                    Some(c) => self.expr(c)?,
                    None => return self.constant(and),
                };
                for c in it {
                    let b = self.expr(c)?;
                    let dst = self.slot()?;
                    self.ops.push(if and {
                        MaskOp::And { a: acc, b, dst }
                    } else {
                        MaskOp::Or { a: acc, b, dst }
                    });
                    acc = Operand::Scratch(dst);
                }
                acc
            }
            SetExpr::All => self.constant(true)?,
            SetExpr::Empty => self.constant(false)?,
        })
    }

    /// A constant mask, built as a contradiction or tautology over row 0's
    /// range: `Range{0, 0}` is empty; its negation is everything.
    fn constant(&mut self, all: bool) -> Result<Operand, LowerError> {
        let dst = self.slot()?;
        self.ops.push(MaskOp::Pred {
            pred: Pred::Range { lo: 0, hi: 0 },
            under: None,
            dst,
        });
        if !all {
            return Ok(Operand::Scratch(dst));
        }
        let neg = self.slot()?;
        self.ops.push(MaskOp::Not {
            a: Operand::Scratch(dst),
            dst: neg,
        });
        Ok(Operand::Scratch(neg))
    }
}

fn set_bits(out: &mut [u64], lo: usize, hi: usize) {
    for r in lo..hi {
        out[r / 64] |= 1u64 << (r % 64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_mask_risc::LaneRef;

    const N: usize = 1000; // not a multiple of 64

    struct Fx {
        basis: Vec<i32>,
        other: Vec<i32>,
        codes: Vec<u32>,
        p0: Vec<u64>,
        p1: Vec<u64>,
    }

    fn fx(seed: u64) -> Fx {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut basis: Vec<i32> = (0..N).map(|_| (next() % 50) as i32 - 10).collect();
        basis.sort_unstable();
        let other = (0..N).map(|_| (next() % 21) as i32 - 10).collect();
        let codes = (0..N).map(|_| (next() % 5) as u32).collect();
        let mut plane = |p: u64| {
            let mut w = vec![0u64; words_for(N)];
            for r in 0..N {
                if next() % p == 0 {
                    w[r / 64] |= 1 << (r % 64);
                }
            }
            w
        };
        let p0 = plane(3);
        let p1 = plane(2);
        Fx {
            basis,
            other,
            codes,
            p0,
            p1,
        }
    }

    // lane 0 = basis (sorted i32), lane 1 = other i32, lane 2 = u32 codes;
    // masks 0, 1.
    fn with_planes<T>(f: &Fx, k: impl FnOnce(&Planes<'_>) -> T) -> T {
        let lanes = [
            LaneRef::I32(&f.basis),
            LaneRef::I32(&f.other),
            LaneRef::U32(&f.codes),
        ];
        let masks: [&[u64]; 2] = [&f.p0, &f.p1];
        k(&Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        })
    }

    /// The row oracle: evaluates the expression row by row, by definition.
    fn holds(f: &Fx, e: &SetExpr, r: usize) -> bool {
        match e {
            SetExpr::All => true,
            SetExpr::Empty => false,
            SetExpr::Sym(i) => {
                let p = if *i == 0 { &f.p0 } else { &f.p1 };
                p[r / 64] >> (r % 64) & 1 == 1
            }
            SetExpr::Res(Res::I32 { lane, op, t }) => {
                let x = if *lane == 0 { f.basis[r] } else { f.other[r] };
                match op {
                    Cmp::Lt => x < *t,
                    Cmp::Le => x <= *t,
                    Cmp::Gt => x > *t,
                    Cmp::Ge => x >= *t,
                    Cmp::Eq => x == *t,
                    Cmp::Ne => x != *t,
                }
            }
            SetExpr::Res(Res::U32 { eq, v, .. }) => (f.codes[r] == *v) == *eq,
            SetExpr::Not(c) => !holds(f, c, r),
            SetExpr::And(cs) => cs.iter().all(|c| holds(f, c, r)),
            SetExpr::Or(cs) => cs.iter().any(|c| holds(f, c, r)),
        }
    }

    fn oracle(f: &Fx, e: &SetExpr) -> (u64, Vec<u64>) {
        let mut w = vec![0u64; words_for(N)];
        let mut c = 0;
        for r in 0..N {
            if holds(f, e, r) {
                c += 1;
                w[r / 64] |= 1 << (r % 64);
            }
        }
        (c, w)
    }

    struct Gen(u64);
    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn leaf(&mut self) -> SetExpr {
            const OPS: [Cmp; 6] = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne];
            match self.next() % 7 {
                0 | 1 | 2 => SetExpr::i32(
                    0,
                    OPS[(self.next() % 6) as usize],
                    (self.next() % 60) as i32 - 15,
                ),
                3 => SetExpr::i32(
                    1,
                    OPS[(self.next() % 6) as usize],
                    (self.next() % 25) as i32 - 12,
                ),
                4 => SetExpr::Res(Res::U32 {
                    lane: 2,
                    eq: self.next().is_multiple_of(2),
                    v: (self.next() % 5) as u32,
                }),
                _ => SetExpr::Sym((self.next() % 2) as u32),
            }
        }
        fn expr(&mut self, depth: u32) -> SetExpr {
            if depth == 0 || self.next().is_multiple_of(3) {
                return self.leaf();
            }
            match self.next() % 4 {
                0 => SetExpr::negated(self.expr(depth - 1)),
                1 | 2 => SetExpr::And((0..1 + self.next() % 3).map(|_| self.expr(depth - 1)).collect()),
                _ => SetExpr::Or((0..1 + self.next() % 3).map(|_| self.expr(depth - 1)).collect()),
            }
        }
    }

    /// FAILS IF: any level gives a different count or set than the row
    /// oracle, or the unsplit plan (geometry disabled) disagrees. Random
    /// nested expressions over a sorted basis lane, an unsorted lane, a code
    /// lane and two planes, N not a word multiple.
    #[test]
    fn every_level_equals_the_row_oracle() {
        let mut g = Gen(0xD2C2);
        let (mut geometry, mut optics, mut split_rows_saved) = (0, 0, 0u64);
        for k in 0..400 {
            let f = fx(1 + k % 5);
            let e = g.expr(4);
            let (want_c, want_w) = oracle(&f, &e);
            let basis = SortedBasis::attest(0, &f.basis).unwrap();
            let plan = SetPlan::new(&e, N, Some(&basis)).unwrap();
            let flat = SetPlan::unsplit(&e, N);
            with_planes(&f, |p| {
                let c = plan.count(p).unwrap();
                assert_eq!(c.value, want_c, "{e:?}");
                assert_eq!(flat.count(p).unwrap().value, want_c, "unsplit {e:?}");
                let mut w = vec![!0u64; words_for(N)];
                plan.keep(p, &mut w).unwrap();
                assert_eq!(w, want_w, "keep {e:?}");
                let mut w2 = vec![!0u64; words_for(N)];
                flat.keep(p, &mut w2).unwrap();
                assert_eq!(w2, want_w, "unsplit keep {e:?}");
                match c.level {
                    Level::Geometry => geometry += 1,
                    _ => optics += 1,
                }
                let fc = flat.count(p).unwrap();
                split_rows_saved += fc.extent_rows.saturating_sub(c.extent_rows);
            });
        }
        // Anti-vacuity: both levels are exercised, and the runs really
        // narrow the extents somewhere.
        assert!(geometry > 20 && optics > 20, "geometry {geometry}, optics {optics}");
        assert!(split_rows_saved > 0);
    }

    /// FAILS IF: a set with no residual reads anything. The planes carry no
    /// lanes and no masks at all, so any program run would be refused; the
    /// geometry level must answer without one.
    #[test]
    fn geometry_reads_no_plane_lane_or_extent() {
        let f = fx(7);
        let basis = SortedBasis::attest(0, &f.basis).unwrap();
        let e = SetExpr::Or(vec![
            SetExpr::And(vec![SetExpr::i32(0, Cmp::Ge, 0), SetExpr::i32(0, Cmp::Lt, 10)]),
            SetExpr::i32(0, Cmp::Gt, 30),
        ]);
        let plan = SetPlan::new(&e, N, Some(&basis)).unwrap();
        assert!(plan.residual().is_none());
        assert!(plan.runs().intervals().len() >= 2, "a multi-run set");
        let empty = Planes {
            n_rows: N,
            masks: &[],
            lanes: &[],
        };
        let c = plan.count(&empty).unwrap();
        assert_eq!(c.level, Level::Geometry);
        assert_eq!((c.extents, c.extent_rows), (0, 0));
        assert_eq!(c.value, oracle(&f, &e).0);
        let mut w = vec![0u64; words_for(N)];
        assert_eq!(plan.keep(&empty, &mut w).unwrap().level, Level::Geometry);
        assert_eq!(w, oracle(&f, &e).1);
        // Preparation: one seek per boundary, never a scan.
        assert!(plan.seeks() <= 6, "{} seeks", plan.seeks());
        assert_eq!(basis.attest_reads(), N as u64);
    }

    /// Experiment C (plan §7). FAILS IF: a sorted-basis range plus an
    /// unsorted predicate does not become runs plus a residual, or the
    /// residual runs over more rows than the range.
    #[test]
    fn a_range_plus_an_unsorted_predicate_splits_into_runs_and_residual() {
        let f = fx(3);
        let basis = SortedBasis::attest(0, &f.basis).unwrap();
        let e = SetExpr::And(vec![
            SetExpr::i32(0, Cmp::Ge, 5),
            SetExpr::i32(0, Cmp::Le, 20),
            SetExpr::i32(1, Cmp::Gt, 0),
        ]);
        let plan = SetPlan::new(&e, N, Some(&basis)).unwrap();
        assert_eq!(plan.residual(), Some(&SetExpr::i32(1, Cmp::Gt, 0)));
        assert_eq!(plan.runs().intervals().len(), 1);
        with_planes(&f, |p| {
            let c = plan.count(p).unwrap();
            assert_eq!(c.value, oracle(&f, &e).0);
            assert_eq!(c.level, Level::Optics);
            assert_eq!(c.extent_rows, plan.runs().count());
            assert!(c.extent_rows < N as u64 / 2, "{} rows", c.extent_rows);
        });
    }

    /// Silence twin. FAILS IF: comparisons on a lane that is not the basis
    /// are turned into runs. With the basis on lane 0, a lane-1 range stays
    /// residual and the runs are the whole population.
    #[test]
    fn a_non_basis_range_stays_residual() {
        let f = fx(4);
        let basis = SortedBasis::attest(0, &f.basis).unwrap();
        let e = SetExpr::And(vec![SetExpr::i32(1, Cmp::Ge, 0), SetExpr::i32(1, Cmp::Lt, 5)]);
        let plan = SetPlan::new(&e, N, Some(&basis)).unwrap();
        assert_eq!(plan.runs(), &Runs::full(N as u64));
        assert!(plan.residual().is_some());
        assert_eq!(plan.seeks(), 0);
    }

    /// FAILS IF: an unsorted lane is accepted as a basis.
    #[test]
    fn an_unsorted_lane_is_refused_as_basis() {
        let f = fx(5);
        assert!(f.other.windows(2).any(|w| w[1] < w[0]));
        assert!(matches!(
            SortedBasis::attest(1, &f.other),
            Err(LowerError::BasisNotSorted { .. })
        ));
    }

    /// FAILS IF: a mismatched population is measured instead of refused.
    #[test]
    fn row_count_mismatches_are_refused() {
        let f = fx(6);
        let basis = SortedBasis::attest(0, &f.basis).unwrap();
        assert!(matches!(
            SetPlan::new(&SetExpr::All, N - 1, Some(&basis)),
            Err(LowerError::BasisRows { .. })
        ));
        let plan = SetPlan::new(&SetExpr::Sym(0), N, Some(&basis)).unwrap();
        let wrong = Planes {
            n_rows: N + 1,
            masks: &[],
            lanes: &[],
        };
        assert!(matches!(plan.count(&wrong), Err(LowerError::PlaneRows { .. })));
        with_planes(&f, |p| {
            let mut short = vec![0u64; 3];
            assert!(matches!(plan.keep(p, &mut short), Err(LowerError::OutLen { .. })));
        });
    }
}
