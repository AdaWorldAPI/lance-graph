//! Conjunctive bodies over the cube's binary relations (D-XLOG-W7 items 1
//! and 3): `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`, the four-cycle,
//! `clique5`, and acyclic chains, with one terminal.
//!
//! A body is a list of [`Atom`]s over variables `0..k`. The terminal binds
//! the variables in that order. A variable's candidates are the AND of
//! - for each atom that joins it to an earlier variable, that atom's row at
//!   the earlier variable's bound position (forward for `rel(earlier, v)`,
//!   reverse for `rel(v, earlier)`), and
//! - every atom's column projection it appears in. This one only prunes:
//!   every atom is also checked by its later variable, so the answer does not
//!   depend on it (a disable run removing it stays green); it keeps the walk
//!   from visiting positions with no row in that column.
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
use ndarray::simd::{mask_and, mask_or_assign, mask_ternlog_popcount, ternlog, U64x4};

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
    /// Per variable, every atom touching it: `(other variable, row base,
    /// selectivity)`. The row at `other`'s bound position holds this
    /// variable's candidates; the selectivity is that block's mean nonzero
    /// row popcount over `n`.
    adj: Vec<Vec<(usize, usize, f64)>>,
    /// Per variable, the variables it shares an atom with.
    nbr: [u8; MAX_BODY_VARS],
    /// The count's step for each `(unbound, bound)` state, indexed
    /// `rem | bound << k`, chosen once by [`Plan::optimize`].
    steps: Vec<Step>,
    /// The estimated cost of each state; negative means not yet planned.
    cost: Vec<f64>,
}

/// Words per row mask: a tile holds at most 256 positions.
const ROW_WORDS: usize = 4;

/// What the factorized count does in one `(unbound, bound)` state.
#[derive(Clone, Copy)]
enum Step {
    /// The unbound variables fall into independent components.
    Split([u8; MAX_BODY_VARS], usize),
    /// Bind this variable next.
    Bind(usize),
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
        debug_assert!(n <= ROW_WORDS * 64, "one tile is at most 256 positions");
        // A fixed stride: one row is one `U64x4`; the words past `n` stay 0.
        let words = ROW_WORDS;
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
        let mut adj: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); k];
        let mut nbr = [0u8; MAX_BODY_VARS];
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
            let sel = |base: usize| {
                let (mut bits, mut live) = (0u64, 0u64);
                for row in rows[base..base + n * words].chunks_exact(words) {
                    let c: u32 = row.iter().map(|x| x.count_ones()).sum();
                    bits += u64::from(c);
                    live += u64::from(c > 0);
                }
                if live == 0 {
                    0.0
                } else {
                    bits as f64 / live as f64 / n as f64
                }
            };
            adj[w].push((u, fwd, sel(fwd)));
            adj[u].push((w, bwd, sel(bwd)));
            nbr[u] |= 1 << w;
            nbr[w] |= 1 << u;
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
        let states = 1usize << (2 * k);
        let mut plan = Self {
            n,
            words,
            k,
            rows,
            proj,
            checks,
            adj,
            nbr,
            steps: vec![Step::Bind(0); states],
            cost: vec![-1.0; states],
        };
        let all = if k == 8 { u8::MAX } else { (1u8 << k) - 1 };
        plan.optimize(all, 0);
        Ok(plan)
    }

    fn state(&self, rem: u8, bound: u8) -> usize {
        usize::from(rem) | usize::from(bound) << self.k
    }

    /// Plan the state `(rem, bound)` and every state it reaches: independent
    /// components split (their costs add, their counts multiply); otherwise
    /// bind the variable minimising `candidates × (1 + cost of the rest)`,
    /// with candidates estimated as `|proj| × Π selectivity` over the atoms
    /// to bound variables. The last variable costs one popcount.
    fn optimize(&mut self, rem: u8, bound: u8) -> f64 {
        let at = self.state(rem, bound);
        if self.cost[at] >= 0.0 {
            return self.cost[at];
        }
        let (comps, n) = self.components(rem);
        let (step, cost) = if rem.count_ones() == 1 {
            (Step::Bind(rem.trailing_zeros() as usize), 1.0)
        } else if n > 1 {
            let c = comps[..n].iter().map(|&c| self.optimize(c, bound)).sum();
            (Step::Split(comps, n), c)
        } else {
            let w = self.words;
            let mut best = (usize::MAX, f64::INFINITY);
            for v in (0..self.k).filter(|&v| rem >> v & 1 == 1) {
                let proj: u32 = self.proj[v * w..(v + 1) * w]
                    .iter()
                    .map(|x| x.count_ones())
                    .sum();
                let mut cand = f64::from(proj);
                for &(other, _, sel) in &self.adj[v] {
                    if bound >> other & 1 == 1 {
                        cand *= sel;
                    }
                }
                let c = cand * (1.0 + self.optimize(rem & !(1 << v), bound | 1 << v));
                if c < best.1 {
                    best = (v, c);
                }
            }
            (Step::Bind(best.0), best.1)
        };
        self.steps[at] = step;
        self.cost[at] = cost;
        cost
    }

    /// The connected components of `rem` over atoms inside it.
    fn components(&self, rem: u8) -> ([u8; MAX_BODY_VARS], usize) {
        let mut out = [0u8; MAX_BODY_VARS];
        let (mut left, mut n) = (rem, 0);
        while left != 0 {
            let mut comp = left.isolate_lowest_one();
            loop {
                let mut grow = comp;
                let mut bits = comp;
                while bits != 0 {
                    let v = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    grow |= self.nbr[v] & rem;
                }
                if grow == comp {
                    break;
                }
                comp = grow;
            }
            out[n] = comp;
            n += 1;
            left &= !comp;
        }
        (out, n)
    }

    /// `v`'s candidates under the bound variables `bound` at `vals`.
    fn candidates_bound(&self, v: usize, bound: u8, vals: &[usize], cand: &mut [u64]) {
        let w = self.words;
        cand.copy_from_slice(&self.proj[v * w..(v + 1) * w]);
        for &(other, base, _) in &self.adj[v] {
            if bound >> other & 1 == 1 {
                let at = base + vals[other] * w;
                let mut tmp = [0u64; 4];
                mask_and(cand, &self.rows[at..at + w], &mut tmp[..w]);
                cand.copy_from_slice(&tmp[..w]);
            }
        }
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

    /// Witnesses over the unbound variables `rem`, given `bound`; marks
    /// support. Independent components multiply; a component's marks are
    /// kept only when every component has a witness.
    fn count(
        &mut self,
        rem: u8,
        bound: u8,
        support: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<u64, BodyRefusal> {
        if rem == 0 {
            return Ok(1);
        }
        let plan = self.plan;
        let (w, k) = (plan.words, plan.k);
        let step = plan.steps[plan.state(rem, bound)];
        match step {
            Step::Split(comps, n) => {
                // One support block per component from the arena; the rest of
                // the arena goes to the components' own splits.
                let (parts, rest) = scratch.split_at_mut(MAX_BODY_VARS * k * w);
                let mut total = 1u64;
                for (comp, part) in comps[..n].iter().zip(parts.chunks_exact_mut(k * w)) {
                    part.fill(0);
                    let c = self.count(*comp, bound, part, rest)?;
                    if c == 0 {
                        return Ok(0);
                    }
                    total = total.checked_mul(c).ok_or(BodyRefusal::CountOverflow)?;
                }
                for part in parts.chunks_exact(k * w).take(n) {
                    mask_or_assign(support, part);
                }
                Ok(total)
            }
            Step::Bind(v) => {
                let mut cand = [0u64; 4];
                plan.candidates_bound(v, bound, &self.bound, &mut cand[..w]);
                let bit = 1u8 << v;
                if rem == bit {
                    let c = &cand[..w];
                    let n = mask_ternlog_popcount::<{ ternlog::AND2 }>(c, c, c);
                    if n > 0 {
                        mask_or_assign(&mut support[v * w..(v + 1) * w], c);
                    }
                    return Ok(n);
                }
                let rest = rem & !bit;
                if rest.count_ones() == 1 {
                    return self.bind_leaves(v, &cand[..w], &[rest], bound, support);
                }
                if let Step::Split(comps, n) = plan.steps[plan.state(rest, bound | bit)] {
                    if comps[..n].iter().all(|c| c.count_ones() == 1) {
                        return self.bind_leaves(v, &cand[..w], &comps[..n], bound, support);
                    }
                }
                let mut total = 0u64;
                for (wi, &word) in cand[..w].iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let x = wi * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        self.bound[v] = x;
                        let sub = self.count(rem & !bit, bound | bit, support, scratch)?;
                        if sub > 0 {
                            support[v * w + x / 64] |= 1 << (x % 64);
                            total = total.checked_add(sub).ok_or(BodyRefusal::CountOverflow)?;
                        }
                    }
                }
                Ok(total)
            }
        }
    }

    /// `Σ_{x ∈ cand(v)} Π_i |cand_i(x)|` for leaves `i` that become
    /// independent once `v` is bound. Each leaf's part fixed by the outer
    /// binding is computed once; per `x` it costs one AND and one popcount
    /// against its rows at `v`.
    fn bind_leaves(
        &mut self,
        v: usize,
        cand: &[u64],
        comps: &[u8],
        bound: u8,
        support: &mut [u64],
    ) -> Result<u64, BodyRefusal> {
        let plan = self.plan;
        let w = plan.words;
        let m = comps.len();
        // Per leaf: the fixed part, and the row bases it reads at `v`.
        let mut fixed = [[0u64; 4]; MAX_BODY_VARS];
        let mut at_v: [[usize; MAX_BODY_VARS]; MAX_BODY_VARS] = [[0; MAX_BODY_VARS]; MAX_BODY_VARS];
        let mut n_at_v = [0usize; MAX_BODY_VARS];
        let mut var = [0usize; MAX_BODY_VARS];
        for i in 0..m {
            let u = comps[i].trailing_zeros() as usize;
            var[i] = u;
            plan.candidates_bound(u, bound, &self.bound, &mut fixed[i][..w]);
            for &(other, base, _) in &plan.adj[u] {
                if other == v {
                    at_v[i][n_at_v[i]] = base;
                    n_at_v[i] += 1;
                }
            }
        }
        let fixed: [U64x4; MAX_BODY_VARS] = std::array::from_fn(|i| U64x4::from_array(fixed[i]));
        let mut sup = [U64x4::splat(0); MAX_BODY_VARS];
        let mut tmp = [U64x4::splat(0); MAX_BODY_VARS];
        let mut total = 0u64;
        for (wi, &word) in cand.iter().enumerate() {
            let mut bits = word;
            'x: while bits != 0 {
                let x = wi * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let mut prod = 1u64;
                for i in 0..m {
                    let mut t = fixed[i];
                    for &base in &at_v[i][..n_at_v[i]] {
                        let at = base + x * w;
                        t &= U64x4::from_slice(&plan.rows[at..at + w]);
                    }
                    let c = t.popcnt().reduce_sum();
                    if c == 0 {
                        continue 'x;
                    }
                    tmp[i] = t;
                    prod = prod.checked_mul(c).ok_or(BodyRefusal::CountOverflow)?;
                }
                support[v * w + x / 64] |= 1 << (x % 64);
                for i in 0..m {
                    sup[i] |= tmp[i];
                }
                total = total.checked_add(prod).ok_or(BodyRefusal::CountOverflow)?;
            }
        }
        for i in 0..m {
            let u = var[i];
            mask_or_assign(&mut support[u * w..(u + 1) * w], &sup[i].to_array());
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
            let all = if k == 8 { u8::MAX } else { (1u8 << k) - 1 };
            // A split's blocks never nest more than `k` deep: a component
            // is connected, so a bind separates any two splits.
            let mut scratch = vec![0u64; k * MAX_BODY_VARS * k * plan.words];
            Walk::new(&plan).count(all, 0, &mut support, &mut scratch)?
        };
        let out = plan.n.div_ceil(64);
        let support = support
            .chunks(plan.words)
            .map(|c| c[..out].to_vec())
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
