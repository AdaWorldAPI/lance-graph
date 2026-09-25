//! Shared synthetic fixtures + the per-row ORACLE.
//!
//! The oracle is a scalar per-row loop — licensed ONLY here, as a test oracle
//! (the `#[cfg(test)]` role raw intrinsics play under `ndarray::simd`). It
//! re-derives every cell independently of the crate, from the raw vectors.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use lance_graph_report::*;

/// Deterministic xorshift.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Raw fixture: ordinal fields (with domains) and signed value fields.
pub struct Fixture {
    pub n: usize,
    pub ords: Vec<(FieldId, Vec<u32>, u32)>,
    pub vals: Vec<(FieldId, Vec<i32>)>,
    pub masks: Vec<(MaskId, Vec<u64>)>,
}

impl Fixture {
    pub fn batch(&self) -> AbiBatch {
        let mut b = AbiBatch::new(SourceId(1), 1, self.n);
        for (f, v, d) in &self.ords {
            b = b
                .with_column(Column::coordinate(*f, v.clone().into(), *d))
                .unwrap();
        }
        for (f, v) in &self.vals {
            b = b
                .with_column(Column::value(*f, LaneData::I32(v.clone().into())))
                .unwrap();
        }
        for (m, w) in &self.masks {
            b = b.with_mask(*m, Arc::from(w.clone())).unwrap();
        }
        b
    }

    fn ord(&self, f: FieldId) -> Option<&Vec<u32>> {
        self.ords
            .iter()
            .find(|(x, _, _)| *x == f)
            .map(|(_, v, _)| v)
    }
    fn domain(&self, f: FieldId) -> u32 {
        self.ords.iter().find(|(x, _, _)| *x == f).unwrap().2
    }
    fn val(&self, f: FieldId) -> Option<&Vec<i32>> {
        self.vals.iter().find(|(x, _)| *x == f).map(|(_, v)| v)
    }

    /// Per-row selection oracle.
    pub fn selected(&self, s: &Selection, i: usize) -> bool {
        match s {
            Selection::All => true,
            Selection::Range(r) => (r.lo as usize) <= i && i < r.hi as usize,
            Selection::Mask(m) => {
                let w = &self.masks.iter().find(|(x, _)| x == m).unwrap().1;
                w[i / 64] >> (i % 64) & 1 == 1
            }
            Selection::Predicate(PredicatePlan::Cmp { field, op, value }) => {
                self.cmp(*field, *op, *value, i)
            }
            Selection::Predicate(PredicatePlan::In { field, values }) => {
                values.iter().any(|v| self.cmp(*field, CmpOp::Eq, *v, i))
            }
            Selection::And(a, b) => self.selected(a, i) && self.selected(b, i),
            Selection::Or(a, b) => self.selected(a, i) || self.selected(b, i),
            Selection::AndNot(a, b) => self.selected(a, i) && !self.selected(b, i),
        }
    }

    fn cmp(&self, f: FieldId, op: CmpOp, v: Scalar, i: usize) -> bool {
        let (x, y): (i64, i64) = match v {
            Scalar::Int(k) => (i64::from(self.val(f).unwrap()[i]), k),
            Scalar::Ordinal(k) => (i64::from(self.ord(f).unwrap()[i]), i64::from(k)),
        };
        match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
        }
    }

    /// Canonical coordinate of row `i` under `coords` (sorted), or None if
    /// outside some domain.
    pub fn coord(&self, coords: &[CoordSpec], i: usize) -> Option<Vec<u32>> {
        coords
            .iter()
            .map(|c| match c {
                CoordSpec::Field(f) => {
                    let o = self.ord(*f).unwrap()[i];
                    (o < self.domain(*f)).then_some(o)
                }
                CoordSpec::Bucket {
                    field,
                    origin,
                    width,
                    count,
                } => {
                    let v = i64::from(self.val(*field).unwrap()[i]);
                    let b = (v - origin).div_euclid(*width);
                    (0..i64::from(*count)).contains(&b).then_some(b as u32)
                }
                CoordSpec::MaskSet { .. } => {
                    panic!("a mask set places a row in several cells; use `cells`")
                }
            })
            .collect()
    }

    fn in_mask(&self, m: MaskId, i: usize) -> bool {
        let w = &self.masks.iter().find(|(x, _)| *x == m).unwrap().1;
        w[i / 64] >> (i % 64) & 1 == 1
    }

    /// Every canonical coordinate row `i` lands on: one per lane dimension,
    /// and one per member of each mask set holding the row (so zero or
    /// several). The cartesian product of the per-dimension choices.
    pub fn cells(&self, coords: &[CoordSpec], i: usize) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new()];
        for c in coords {
            let members: Vec<u32> = match c {
                CoordSpec::MaskSet { count, .. } => (0..*count)
                    .filter(|&m| self.in_mask(c.member_mask(m).unwrap(), i))
                    .collect(),
                _ => self.coord(std::slice::from_ref(c), i).unwrap_or_default(),
            };
            out = out
                .into_iter()
                .flat_map(|prefix| {
                    members.iter().map(move |&m| {
                        let mut p = prefix.clone();
                        p.push(m);
                        p
                    })
                })
                .collect();
        }
        out
    }

    /// Oracle cells: canonical coordinate → (count, sum, min, max) of `m`.
    pub fn oracle(
        &self,
        plan: &ReportPlan,
        m: Option<FieldId>,
    ) -> HashMap<Vec<u32>, (i64, i64, i64, i64)> {
        let coords = plan.physical_key().coords;
        let mut out: HashMap<Vec<u32>, (i64, i64, i64, i64)> = HashMap::new();
        for i in 0..self.n {
            if !self.selected(&plan.selection, i) {
                continue;
            }
            let v = m.map_or(0, |f| i64::from(self.val(f).unwrap()[i]));
            for c in self.cells(&coords, i) {
                let e = out.entry(c).or_insert((0, 0, i64::MAX, i64::MIN));
                e.0 += 1;
                e.1 += v;
                e.2 = e.2.min(v);
                e.3 = e.3.max(v);
            }
        }
        out
    }
}

/// A synthetic population: `dims` ordinal fields F0.. with the given
/// domains, value field F100 in [-50, 50), second value field F101 in [0, 1000).
pub fn synthetic(n: usize, domains: &[u32], seed: u64) -> Fixture {
    let mut r = Rng(seed | 1);
    let ords = domains
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            (
                FieldId(i as u32),
                (0..n).map(|_| r.below(u64::from(d)) as u32).collect(),
                d,
            )
        })
        .collect();
    let vals = vec![
        (
            FieldId(100),
            (0..n).map(|_| r.below(100) as i32 - 50).collect(),
        ),
        (FieldId(101), (0..n).map(|_| r.below(1000) as i32).collect()),
    ];
    Fixture {
        n,
        ords,
        vals,
        masks: vec![],
    }
}

pub fn src() -> SourceRef {
    SourceRef {
        id: SourceId(1),
        generation: 1,
    }
}

/// Compare every stored/presentable cell of `res` against the oracle for
/// COUNT / SUM / MIN / MAX of `m` (plan must carry those four measures in
/// that order).
pub fn assert_matches_oracle(fx: &Fixture, plan: &ReportPlan, res: &ReportResult, m: FieldId) {
    let or = fx.oracle(plan, Some(m));
    let dims = res.space().dims();
    let ms = res.measures().to_vec();
    // Every oracle cell must read back exactly.
    for (c, (cnt, sum, mn, mx)) in &or {
        let fixed: Vec<Option<u32>> = c.iter().map(|&x| Some(x)).collect();
        assert_eq!(
            res.space().merged(&ms[0], &fixed),
            CellValue::Int(*cnt),
            "count at {c:?}"
        );
        assert_eq!(
            res.space().merged(&ms[1], &fixed),
            CellValue::Int(*sum),
            "sum at {c:?}"
        );
        assert_eq!(
            res.space().merged(&ms[2], &fixed),
            CellValue::Int(*mn),
            "min at {c:?}"
        );
        assert_eq!(
            res.space().merged(&ms[3], &fixed),
            CellValue::Int(*mx),
            "max at {c:?}"
        );
    }
    // And no cell the oracle lacks may carry rows.
    if !res.space().is_sparse() && !dims.is_empty() {
        let total: i64 = or.values().map(|v| v.0).sum();
        assert_eq!(res.grand_total(&ms[0]), CellValue::Int(total));
    } else {
        assert_eq!(
            res.space().stored_cells(),
            or.len().max(usize::from(dims.is_empty()))
        );
    }
}

pub fn four(m: FieldId) -> [Measure; 4] {
    [
        Measure::count(),
        Measure::of(MeasureKind::Sum, m),
        Measure::of(MeasureKind::Min, m),
        Measure::of(MeasureKind::Max, m),
    ]
}

pub fn with_four(mut p: ReportPlan, m: FieldId) -> ReportPlan {
    for x in four(m) {
        p = p.measure(x);
    }
    p
}
