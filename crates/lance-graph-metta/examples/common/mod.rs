//! Shared by the D-MTP probes: the fact layout, the gather path (D-MTP-1),
//! the range path (D-MTP-1b) and the datafrog oracle. Probe code, not a
//! shipped API.
#![allow(dead_code)]

use std::time::Instant;

use datafrog::Relation;
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, ForeignPlane, LaneRef, Out, Planes, Value};
use lance_graph_quack::{lower, Agg, Col, Filter, ForeignPlane as QForeign, Mask, Query};

const DST: Col = Col(0);
const VALID: Mask = Mask(0);
const VALID_FOREIGN: QForeign = QForeign(0);

/// A half-open row interval `[lo, hi)`.
pub type Run = (u32, u32);

/// A functional binary relation laid out over its subjects' address space:
/// row `a` is the fact whose subject has address `a`, lane `dst` holds the
/// object's address, plane `valid` marks the addresses that have a fact.
pub struct Relation2 {
    pub n: usize,
    pub dst: Vec<u32>,
    pub valid: Vec<u64>,
    /// `valid`'s runs, recorded during ingest when the facts arrive in
    /// address order (`None` otherwise).
    pub declared_runs: Option<Vec<Run>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum IngestError {
    /// A subject already has a fact: not a functional relation.
    SecondFact { subject: u32 },
    /// An address outside the declared address space.
    OutOfSpace { addr: u32 },
}

impl Relation2 {
    /// Ingest facts `(subject, object)` over addresses `0..n`. A row without a
    /// fact gets the previous row's `dst` (it lies outside `valid`, so the
    /// value is never read as a fact; it only keeps a monotone lane monotone).
    pub fn ingest(n: usize, facts: &[(u32, u32)]) -> Result<Self, IngestError> {
        let mut dst = vec![u32::MAX; n];
        let mut valid = vec![0u64; words_for(n)];
        let mut runs: Vec<Run> = Vec::new();
        let mut in_order = true;
        let mut last: Option<u32> = None;
        for &(s, o) in facts {
            for a in [s, o] {
                if a as usize >= n {
                    return Err(IngestError::OutOfSpace { addr: a });
                }
            }
            let (w, b) = (s as usize / 64, s as usize % 64);
            if (valid[w] >> b) & 1 == 1 {
                return Err(IngestError::SecondFact { subject: s });
            }
            valid[w] |= 1 << b;
            dst[s as usize] = o;
            match (in_order, last) {
                (true, Some(p)) if s <= p => in_order = false,
                (true, _) => match runs.last_mut() {
                    Some(r) if r.1 == s => r.1 = s + 1,
                    _ => runs.push((s, s + 1)),
                },
                _ => {}
            }
            last = Some(s);
        }
        let mut prev = 0u32;
        for d in dst.iter_mut() {
            if *d == u32::MAX {
                *d = prev;
            } else {
                prev = *d;
            }
        }
        Ok(Relation2 {
            n,
            dst,
            valid,
            declared_runs: in_order.then_some(runs),
        })
    }

    /// The order witness for the range path: `dst` non-decreasing over rows.
    pub fn dst_monotone(&self) -> bool {
        self.dst.windows(2).all(|w| w[0] <= w[1])
    }

    // ───────────── gather path (D-MTP-1): Plane(valid) AND Semijoin ─────────────

    fn query(&self) -> Filter {
        Filter::and([Filter::plane(VALID), Filter::semijoin(DST, VALID_FOREIGN)])
    }

    fn run(&self, agg: Agg, kept: &mut [u64]) -> Value {
        let keep = matches!(agg, Agg::Rows);
        let program = lower(&Query {
            filter: self.query(),
            agg,
        })
        .expect("lowers");
        let lanes = [LaneRef::U32(&self.dst)];
        let masks = [self.valid.as_slice()];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        let fplanes = [ForeignPlane {
            words: &self.valid,
            rows: self.n,
        }];
        let foreign = Foreign {
            planes: &fplanes,
            lanes: &[],
        };
        let mut scratch = Scratch::for_program(&program, self.n).expect("carves");
        let out = if keep { Out::Mask(kept) } else { Out::None };
        execute_into(&program, &planes, &foreign, &mut scratch, out).expect("runs")
    }

    /// `size-atom` of the collapsed result, by the gather program.
    pub fn count(&self) -> usize {
        match self.run(Agg::Count, &mut []) {
            Value::Count(c) => c,
            other => panic!("not a count: {other:?}"),
        }
    }

    /// The bindings `($x, $z)`, emitted only for rows the gather program kept.
    pub fn pairs(&self, kept: &mut [u64], out: &mut Vec<(u32, u32)>) {
        match self.run(Agg::Rows, kept) {
            Value::Mask(_) => {}
            other => panic!("not a kept mask: {other:?}"),
        }
        out.clear();
        for (w, &word) in kept.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let x = (w * 64 + bits.trailing_zeros() as usize) as u32;
                let y = self.dst[x as usize];
                out.push((x, self.dst[y as usize]));
                bits &= bits - 1;
            }
        }
    }

    // ───────────── range path (D-MTP-1b): the hop as interval arithmetic ─────────────

    /// `valid`'s runs, found by a word scan of the plane.
    pub fn scanned_runs(&self) -> Vec<Run> {
        let mut runs: Vec<Run> = Vec::new();
        for (w, &word) in self.valid.iter().enumerate() {
            let mut bits = word;
            let base = (w * 64) as u32;
            while bits != 0 {
                let start = bits.trailing_zeros();
                let len = (!(bits >> start)).trailing_zeros().min(64 - start);
                let (lo, hi) = (base + start, base + start + len);
                match runs.last_mut() {
                    Some(r) if r.1 == lo => r.1 = hi,
                    _ => runs.push((lo, hi)),
                }
                bits = if start + len >= 64 { 0 } else { bits & (u64::MAX << (start + len)) };
            }
        }
        runs
    }

    /// The rows whose target lies in one of `runs`: one interval per run,
    /// found by binary search on the monotone `dst` lane.
    pub fn preimage(&self, runs: &[Run]) -> Vec<Run> {
        runs.iter()
            .filter_map(|&(a, b)| {
                let lo = self.dst.partition_point(|&d| d < a) as u32;
                let hi = self.dst.partition_point(|&d| d < b) as u32;
                (lo < hi).then_some((lo, hi))
            })
            .collect()
    }

    /// The kept rows of `Plane(valid) AND Semijoin(dst, valid)` as intervals.
    /// `None` when the order witness fails: the caller uses the gather.
    pub fn kept_ranges(&self, valid_runs: &[Run]) -> Option<Vec<Run>> {
        if !self.dst_monotone() {
            return None;
        }
        Some(intersect(valid_runs, &self.preimage(valid_runs)))
    }

    /// The bindings `($x, $z)` by walking the kept intervals.
    pub fn range_pairs(&self, kept: &[Run], out: &mut Vec<(u32, u32)>) {
        out.clear();
        for &(lo, hi) in kept {
            for x in lo..hi {
                let y = self.dst[x as usize];
                out.push((x, self.dst[y as usize]));
            }
        }
    }
}

/// Intersection of two sorted, disjoint interval lists.
pub fn intersect(a: &[Run], b: &[Run]) -> Vec<Run> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Total length of an interval list.
pub fn total(runs: &[Run]) -> usize {
    runs.iter().map(|&(lo, hi)| (hi - lo) as usize).sum()
}

/// The datafrog oracle: `(x, z)` with `parent(x, y)` and `parent(y, z)`.
pub fn oracle(facts: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let parent: Relation<(u32, u32)> = facts.iter().copied().collect();
    let by_dst: Relation<(u32, u32)> = facts.iter().map(|&(x, y)| (y, x)).collect();
    let pairs = Relation::from_join(&by_dst, &parent, |_y, &x, &z| (x, z));
    pairs.elements.clone()
}

/// Best wall time of `reps` runs, and the last result.
pub fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (out.expect("reps > 0"), best)
}
