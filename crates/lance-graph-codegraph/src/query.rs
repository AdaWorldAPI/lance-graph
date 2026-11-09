//! Graded codegraph queries. Every answer is a mask over symbol positions (or
//! over artifact rows); no names and no tuples are built.
//!
//! Grading: **confirmed** means a `CALLS_RESOLVED` / `REFERENCES` row backs
//! it. **possible** means only `MAY_DISPATCH` (plus `IMPLEMENTS`) backs it; a
//! `MAY_DISPATCH` row is never confirmed. **unknown** means a dynamic call
//! (`UNRESOLVED_CALL`, the `?dynamic` position) or a function-pointer
//! reference could reach it; it is never reported as "unaffected".

use lance_graph_xlog::cube::{CubeRefusal, RelId};
use lance_graph_xlog::wide::WideRows;
use ndarray::simd::{mask_and, mask_andnot_assign, mask_or_assign};

use crate::artifact::{P_CALLS, P_IMPLEMENTS, P_MAY, P_REFERENCES};
use crate::graph::{Codegraph, Pos, CALLS, IMPL, MAY, NODE, RCONF, RPOSS};

/// An answer in three grades, each a position mask of `words()` words.
/// `confirmed` is backed by resolved rows only; `possible` by `MAY_DISPATCH`
/// (never confirmed); `unknown` is reachable through a dynamic call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graded {
    /// Backed by `CALLS_RESOLVED` / `REFERENCES` rows.
    pub confirmed: Vec<u64>,
    /// Backed only by `MAY_DISPATCH` through an `IMPLEMENTS` edge.
    pub possible: Vec<u64>,
    /// Reachable by a dynamic call or function pointer; not decidable.
    pub unknown: Vec<u64>,
}

/// A graded caller/callee answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calls {
    /// The three grades.
    pub graded: Graded,
    /// Confirmed call sites (`CALLS_RESOLVED` rows) behind `graded.confirmed`.
    pub sites: u64,
}

/// Rows that mention a symbol, as a row mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sites {
    /// Row mask, `rows().div_ceil(64)` words.
    pub rows: Vec<u64>,
    /// Rows that are confirmed (`CALLS_RESOLVED` / `REFERENCES`).
    pub confirmed: u64,
    /// Rows that are only possible (`MAY_DISPATCH` through an impl).
    pub possible: u64,
}

/// Why a query was refused.
#[derive(Debug)]
pub enum QueryRefusal {
    /// Any other cube refusal.
    Cube(CubeRefusal),
}

fn refuse(e: CubeRefusal) -> QueryRefusal {
    QueryRefusal::Cube(e)
}

fn has(mask: &[u64], p: Pos) -> bool {
    mask.get(p as usize / 64)
        .is_some_and(|w| w >> (p % 64) & 1 == 1)
}

fn intersects(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).any(|(x, y)| x & y != 0)
}

fn bits(mask: &[u64]) -> impl Iterator<Item = Pos> + '_ {
    (0..mask.len() * 64)
        .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
        .filter_map(|p| Pos::try_from(p).ok())
}

impl Codegraph {
    /// Partners of `v` in `rel`, as a position mask and the derivation count.
    /// `into`: positions `x` with `(x, v)`; otherwise positions `y` with
    /// `(v, y)`. The join with `NODE` only restricts to interned positions.
    /// An out-of-range `v` has no partners: an empty mask and zero.
    fn partners(&mut self, rel: RelId, v: Pos, into: bool) -> Result<(Vec<u64>, u64), CubeRefusal> {
        if self.symbol(v).is_none() {
            return Ok((vec![0; self.words()], 0));
        }
        let (bound_col, join_col) = if into { (1, 0) } else { (0, 1) };
        let words = self.words();
        let (cube, table) = self.cube_and_table_mut();
        let scan = cube.scan(rel)?;
        let bound = cube.bind(table, scan, bound_col, v)?;
        let node = cube.scan(NODE)?;
        let join = cube.join(bound, join_col, node, 0)?;
        let mut mask = cube.support(table, join)?;
        // A multi-tile cube answers over every reserved position (256 per
        // tile); the graph's masks cover only the interned symbols. Positions
        // past the last symbol are never set (coordinate == position), so the
        // tail words carry nothing and are dropped.
        mask.resize(words, 0);
        let counts = cube.bag_count(table, join)?;
        let sites = counts.iter().fold(0u64, |a, &c| {
            a.saturating_add(u64::try_from(c).unwrap_or(0))
        });
        Ok((mask, sites))
    }

    /// OR into `acc` the partners of every set bit of `from`.
    fn or_partners(
        &mut self,
        acc: &mut [u64],
        from: &[u64],
        rel: RelId,
        into: bool,
    ) -> Result<(), CubeRefusal> {
        for t in bits(from) {
            let (m, _) = self.partners(rel, t, into)?;
            mask_or_assign(acc, &m);
        }
        Ok(())
    }

    /// Who calls `g`. Confirmed: callers in `CALLS` (`CALLS_RESOLVED` rows),
    /// with `sites` their row count. Possible: callers that `MAY_DISPATCH` to
    /// a `Trait::method` that `g` implements (never confirmed). Unknown: the
    /// dynamic callers, when `g` is an impl or is referenced (a dyn call or
    /// fn-pointer call may reach it). An out-of-range `g` yields empty masks.
    pub fn callers(&mut self, g: Pos) -> Result<Calls, QueryRefusal> {
        let w = self.words();
        let (confirmed, sites) = self.partners(CALLS, g, true).map_err(refuse)?;

        let (traits, _) = self.partners(IMPL, g, false).map_err(refuse)?;
        let mut possible = vec![0u64; w];
        self.or_partners(&mut possible, &traits, MAY, true)
            .map_err(refuse)?;
        mask_andnot_assign(&mut possible, &confirmed);

        let mut unknown = vec![0u64; w];
        let reachable = has(self.impl_subjects(), g) || has(self.referenced(), g);
        if reachable {
            unknown.copy_from_slice(self.dynamic_callers());
            let mut seen = confirmed.clone();
            mask_or_assign(&mut seen, &possible);
            mask_andnot_assign(&mut unknown, &seen);
        }
        Ok(Calls {
            graded: Graded {
                confirmed,
                possible,
                unknown,
            },
            sites,
        })
    }

    /// What `f` calls. Confirmed: callees in `CALLS`, `sites` their row
    /// count. Possible: impls of every `Trait::method` that `f` may dispatch
    /// to (never confirmed). Unknown: the `?dynamic` position, when `f` has an
    /// unresolved call. An out-of-range `f` yields empty masks.
    pub fn callees(&mut self, f: Pos) -> Result<Calls, QueryRefusal> {
        let w = self.words();
        let (confirmed, sites) = self.partners(CALLS, f, false).map_err(refuse)?;

        let (traits, _) = self.partners(MAY, f, false).map_err(refuse)?;
        let mut possible = vec![0u64; w];
        self.or_partners(&mut possible, &traits, IMPL, true)
            .map_err(refuse)?;
        mask_andnot_assign(&mut possible, &confirmed);

        let mut unknown = vec![0u64; w];
        if let Some(d) = self.pos("?dynamic") {
            if self.symbol(f).is_some() && has(self.dynamic_callers(), f) {
                unknown[d as usize / 64] |= 1 << (d % 64);
            }
        }
        Ok(Calls {
            graded: Graded {
                confirmed,
                possible,
                unknown,
            },
            sites,
        })
    }

    /// Artifact rows that refer to `g`. Confirmed: `CALLS_RESOLVED` and
    /// `REFERENCES` rows whose object is `g`. Possible: `MAY_DISPATCH` rows
    /// whose `Trait::method` target (the part before " via ") is implemented
    /// by `g`; never confirmed. This is a scan over the row lanes producing a
    /// row mask, not a tuple decode. An out-of-range `g` yields no rows.
    pub fn references(&self, g: Pos) -> Sites {
        let rows = self.rows();
        let mut mask = vec![0u64; rows.div_ceil(64)];
        let (mut confirmed, mut possible) = (0u64, 0u64);
        let Some(gid) = self.id_of(g) else {
            return Sites {
                rows: mask,
                confirmed,
                possible,
            };
        };
        // One fold over the integer lanes, no string compare: the
        // `Trait::method` node ids `g` implements come from IMPLEMENTS rows by
        // id, and each MAY_DISPATCH row carries its precomputed target id.
        let l = self.lanes();
        let mut impls: Vec<u32> = (0..rows)
            .filter(|&i| l.pred[i] == P_IMPLEMENTS && l.subject[i] == gid)
            .map(|i| l.object[i])
            .collect();
        impls.sort_unstable();
        impls.dedup();
        for i in 0..rows {
            let hit = match l.pred[i] {
                P_CALLS | P_REFERENCES if l.object[i] == gid => {
                    confirmed += 1;
                    true
                }
                P_MAY if impls.binary_search(&l.may_target[i]).is_ok() => {
                    possible += 1;
                    true
                }
                _ => false,
            };
            if hit {
                mask[i / 64] |= 1 << (i % 64);
            }
        }
        Sites {
            rows: mask,
            confirmed,
            possible,
        }
    }

    /// Functions affected if `g` changes, from the reverse closures.
    /// Confirmed: functions in `RCONF⁺` of `g`. Possible: in `RPOSS⁺` but not
    /// confirmed (dispatch edges). Unknown: when `g`, or anything it reaches
    /// in `RPOSS⁺`, is an impl or referenced and a `?dynamic` position exists,
    /// the dynamic callers (`RPOSS⁺` of `?dynamic`) not already graded. All
    /// masks are restricted to functions. Closures of any width are read; an
    /// out-of-range `g` yields empty masks.
    pub fn impact(&self, g: Pos) -> Result<Graded, QueryRefusal> {
        let w = self.words();
        let cc = &self.closure(RCONF).map_err(refuse)?.0;
        let cp = &self.closure(RPOSS).map_err(refuse)?.0;
        // `row_into` zero-extends or truncates to the buffer, so a `words()`
        // long buffer holds exactly the interned positions' bits.
        let row = |c: &WideRows, p: Pos| -> Vec<u64> {
            let mut m = vec![0u64; w];
            c.row_into(p, &mut m);
            m
        };
        if self.symbol(g).is_none() {
            let z = vec![0u64; w];
            return Ok(Graded {
                confirmed: z.clone(),
                possible: z.clone(),
                unknown: z,
            });
        }

        let mut confirmed = vec![0u64; w];
        mask_and(&row(cc, g), self.fns(), &mut confirmed);

        let reach_poss = row(cp, g);
        let mut possible = vec![0u64; w];
        mask_and(&reach_poss, self.fns(), &mut possible);
        mask_andnot_assign(&mut possible, &confirmed);

        let mut unknown = vec![0u64; w];
        if let Some(d) = self.pos("?dynamic") {
            let mut reach = reach_poss;
            reach[g as usize / 64] |= 1 << (g % 64);
            let mut dyn_reachable = self.impl_subjects().to_vec();
            mask_or_assign(&mut dyn_reachable, self.referenced());
            if intersects(&reach, &dyn_reachable) {
                mask_and(&row(cp, d), self.fns(), &mut unknown);
                let mut seen = confirmed.clone();
                mask_or_assign(&mut seen, &possible);
                mask_andnot_assign(&mut unknown, &seen);
            }
        }
        Ok(Graded {
            confirmed,
            possible,
            unknown,
        })
    }
}
