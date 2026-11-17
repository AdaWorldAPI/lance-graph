//! Conjunctive bodies over the cube's binary relations (D-XLOG-W7 items 1
//! and 3): `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`, the four-cycle,
//! `clique5`, and acyclic chains, with one terminal.
//!
//! A body is a list of [`Atom`]s over variables `0..k`. The terminal binds
//! the variables in that order. A variable's candidates are the AND of
//! - every atom's column projection it appears in, and
//! - for each atom that joins it to an earlier variable, that atom's row at
//!   the earlier variable's bound position (forward for `rel(earlier, v)`,
//!   reverse for `rel(v, earlier)`).
//!
//! Every atom is therefore checked exactly when its later variable is bound,
//! which is what lets a cyclic body close: `e3(X, Z)` is applied to `Z`'s
//! candidates under the `X` already bound, so the walk does not lose the
//! start node the way a forward frontier does. The last variable is answered
//! by a popcount; no tuple is built.
//!
//! [`Cube::body_answer`] returns the number of witnesses (distinct
//! bindings of all `k` variables, Boolean per base relation as cube
//! relations are) and one support mask per variable: the positions that
//! occur in at least one witness, the single-column projection of the head.
//! [`Cube::materialize_witnesses`] is the one named place this module builds
//! rows, for a caller that downloads the answer (the edge).
//!
//! Scope: one tile (the row masks are `n × words` per atom and direction,
//! built per call, as [`Cube::reach`] builds its edge rows), at most
//! [`MAX_BODY_VARS`] variables, no repeated variable inside one atom.

use lance_graph_dt64::dr::MapTable;
use ndarray::simd::{mask_and, mask_or_assign, mask_ternlog_popcount, ternlog};

use super::{Cube, CubeRefusal, RelId};

/// The most variables one body may bind.
pub const MAX_BODY_VARS: usize = 8;

/// `rel(vars[0], vars[1])`: column 0 binds `vars[0]`, column 1 binds
/// `vars[1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atom {
    pub rel: RelId,
    pub vars: [u8; 2],
}

/// Why a body did not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BodyRefusal {
    /// The cube refused (frame, unknown relation, a tiled cube).
    Cube(CubeRefusal),
    /// Zero variables, or more than [`MAX_BODY_VARS`].
    Arity(usize),
    /// An atom names a variable outside `0..k`.
    VarOutOfRange { atom: usize, var: u8 },
    /// An atom names the same variable twice.
    SameVariable { atom: usize },
    /// A variable no atom names: it would range over the whole tile.
    Unused(u8),
    /// The witness count does not fit a `u64`.
    CountOverflow,
}

impl From<CubeRefusal> for BodyRefusal {
    fn from(r: CubeRefusal) -> Self {
        Self::Cube(r)
    }
}

/// What [`Cube::body_answer`] returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyAnswer {
    /// Witnesses: distinct bindings of all variables.
    pub count: u64,
    /// One mask per variable over the tile's local positions.
    pub support: Vec<Vec<u64>>,
}

/// One constraint on a variable from an atom whose other end is earlier.
#[derive(Clone, Copy)]
struct Check {
    /// Index into the row arena: `(atom * 2 + dir) * n * words`.
    base: usize,
    /// The earlier variable whose bound position selects the row.
    earlier: usize,
}

/// The per-call row masks and checks of one body.
struct Plan {
    n: usize,
    words: usize,
    k: usize,
    rows: Vec<u64>,
    proj: Vec<u64>,
    checks: Vec<Vec<Check>>,
}

impl Plan {
    fn new(cube: &Cube, atoms: &[Atom], k: usize) -> Result<Self, BodyRefusal> {
        if k == 0 || k > MAX_BODY_VARS {
            return Err(BodyRefusal::Arity(k));
        }
        let mut used = [false; MAX_BODY_VARS];
        for (i, a) in atoms.iter().enumerate() {
            for &v in &a.vars {
                if usize::from(v) >= k {
                    return Err(BodyRefusal::VarOutOfRange { atom: i, var: v });
                }
                used[usize::from(v)] = true;
            }
            if a.vars[0] == a.vars[1] {
                return Err(BodyRefusal::SameVariable { atom: i });
            }
        }
        if let Some(v) = (0..k).find(|&v| !used[v]) {
            return Err(BodyRefusal::Unused(v as u8));
        }
        let n = cube.n;
        let words = n.div_ceil(64);
        debug_assert!(words <= 4, "one tile is at most 256 positions");
        let mut rows = vec![0u64; atoms.len() * 2 * n * words];
        let mut proj = vec![0u64; k * words];
        for v in 0..k {
            let p = &mut proj[v * words..(v + 1) * words];
            for (i, w) in p.iter_mut().enumerate() {
                let bits = n.saturating_sub(i * 64).min(64);
                *w = if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
            }
        }
        let mut col = vec![0u64; words];
        let mut checks: Vec<Vec<Check>> = vec![Vec::new(); k];
        for (i, a) in atoms.iter().enumerate() {
            let rel = cube.lanes(a.rel)?;
            let (c0, c1) = (&rel.cols[0], &rel.cols[1]);
            let fwd = i * 2 * n * words;
            let bwd = fwd + n * words;
            for (&x, &y) in c0.iter().zip(c1) {
                let (x, y) = (x as usize, y as usize);
                rows[fwd + x * words + y / 64] |= 1 << (y % 64);
                rows[bwd + y * words + x / 64] |= 1 << (x % 64);
            }
            for (side, cv) in [(0usize, c0), (1, c1)] {
                col.fill(0);
                for &x in cv {
                    col[x as usize / 64] |= 1 << (x % 64);
                }
                let v = usize::from(a.vars[side]);
                let p = &mut proj[v * words..(v + 1) * words];
                for (dst, &c) in p.iter_mut().zip(&col) {
                    *dst &= c;
                }
            }
            let (u, w) = (usize::from(a.vars[0]), usize::from(a.vars[1]));
            if u < w {
                // rel(u, w): w's candidates are u's forward row.
                checks[w].push(Check {
                    base: fwd,
                    earlier: u,
                });
            } else {
                // rel(u, w) with w earlier: u's candidates are w's reverse row.
                checks[u].push(Check {
                    base: bwd,
                    earlier: w,
                });
            }
        }
        Ok(Self {
            n,
            words,
            k,
            rows,
            proj,
            checks,
        })
    }

    /// `cand[v] = proj[v] ∧ rows of every check at the bound positions`.
    fn candidates(&self, v: usize, bound: &[usize], cand: &mut [u64]) {
        let w = self.words;
        cand.copy_from_slice(&self.proj[v * w..(v + 1) * w]);
        for c in &self.checks[v] {
            let at = c.base + bound[c.earlier] * w;
            let row = &self.rows[at..at + w];
            let mut tmp = [0u64; 4];
            mask_and(cand, row, &mut tmp[..w]);
            cand.copy_from_slice(&tmp[..w]);
        }
    }
}

/// The walk's state: one bound position per variable, allocated once per
/// call. Candidates live in a 4-word stack buffer per level (a tile is at
/// most 256 positions).
struct Walk<'p> {
    plan: &'p Plan,
    bound: Vec<usize>,
}

impl<'p> Walk<'p> {
    fn new(plan: &'p Plan) -> Self {
        Self {
            plan,
            bound: vec![0; plan.k],
        }
    }

    /// Witnesses under the current binding of `0..v`; marks support.
    fn count(&mut self, v: usize, support: &mut [u64]) -> Result<u64, BodyRefusal> {
        let (w, k) = (self.plan.words, self.plan.k);
        let mut cand = [0u64; 4];
        self.plan.candidates(v, &self.bound, &mut cand[..w]);
        if v + 1 == k {
            let c = &cand[..w];
            let n = mask_ternlog_popcount::<{ ternlog::AND2 }>(c, c, c);
            if n > 0 {
                mask_or_assign(&mut support[v * w..(v + 1) * w], c);
            }
            return Ok(n);
        }
        let mut total = 0u64;
        for (wi, &word) in cand[..w].iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let x = wi * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.bound[v] = x;
                let sub = self.count(v + 1, support)?;
                if sub > 0 {
                    support[v * w + x / 64] |= 1 << (x % 64);
                    total = total.checked_add(sub).ok_or(BodyRefusal::CountOverflow)?;
                }
            }
        }
        Ok(total)
    }

    /// Push every witness under the current binding of `0..v`, as local
    /// positions, `k` per witness.
    fn emit(&mut self, v: usize, out: &mut Vec<usize>) {
        let (w, k) = (self.plan.words, self.plan.k);
        let mut cand = [0u64; 4];
        self.plan.candidates(v, &self.bound, &mut cand[..w]);
        for (wi, &word) in cand[..w].iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let x = wi * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.bound[v] = x;
                if v + 1 == k {
                    out.extend_from_slice(&self.bound);
                } else {
                    self.emit(v + 1, out);
                }
            }
        }
    }
}

impl Cube {
    /// The witness count and per-variable support of a conjunctive body,
    /// with no tuple built. See the module docs for the order of checks.
    ///
    /// # Errors
    ///
    /// A [`BodyRefusal`]; a tiled cube is [`CubeRefusal::Tiled`].
    pub fn body_answer(
        &self,
        table: &MapTable,
        atoms: &[Atom],
        k: usize,
    ) -> Result<BodyAnswer, BodyRefusal> {
        if self.tiled {
            return Err(CubeRefusal::Tiled.into());
        }
        self.check_frame(table)?;
        let plan = Plan::new(self, atoms, k)?;
        let mut support = vec![0u64; k * plan.words];
        let count = if plan.n == 0 {
            0
        } else {
            Walk::new(&plan).count(0, &mut support)?
        };
        let support = support
            .chunks(plan.words.max(1))
            .map(<[u64]>::to_vec)
            .collect();
        Ok(BodyAnswer { count, support })
    }

    /// The witnesses of a conjunctive body as global ids, `k` per witness,
    /// in local-position order: the one place this module builds rows, for
    /// a caller that downloads the answer.
    ///
    /// # Errors
    ///
    /// As [`Cube::body_answer`].
    pub fn materialize_witnesses(
        &self,
        table: &MapTable,
        atoms: &[Atom],
        k: usize,
    ) -> Result<Vec<u64>, BodyRefusal> {
        if self.tiled {
            return Err(CubeRefusal::Tiled.into());
        }
        let map = self.map(table)?;
        let plan = Plan::new(self, atoms, k)?;
        let mut locals = Vec::new();
        if plan.n > 0 {
            Walk::new(&plan).emit(0, &mut locals);
        }
        let out: Vec<u64> = locals
            .iter()
            .map(|&x| map.global(x as u8).expect("local position from intake"))
            .collect();
        let mut s = self.stats.get();
        s.rows_materialized += (out.len() / k) as u64;
        self.stats.set(s);
        Ok(out)
    }
}
