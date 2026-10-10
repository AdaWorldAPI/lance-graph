//! **D-SCF-CARE-PAIR-0 — can one certified local relation make a later
//! derivation unnecessary?** A test-only probe over the unchanged crossword
//! core (`shared/crossword_core.rs`, #1387/#1398). No new primitive: every
//! population pass is `ndarray::simd::{mask_ternlog_popcount,
//! mask_andnot_assign}` or the core's own `State::narrow`.
//!
//! # What the existing core does, and does not
//!
//! `propagate_token` is singleton-triggered: only a slot whose mask reaches
//! popcount 1 hands letters to its crossings. Letter support of a slot that
//! still holds many candidates is never propagated. That gap is what every
//! arm below measures against.
//!
//! # Arms (same puzzles, same givens, same popcount slot policy, same op budget)
//!
//! | arm | root | per node |
//! |---|---|---|
//! | `Baseline` | fixed point (#1387) | fixed point |
//! | `Support` | + letter support at every crossing (zero-population certificate) | the same, recomputed from resident masks |
//! | `Single` | + single-crossing counterfactuals: assume one letter on one cell, settle, restore | fixed point |
//! | `Pair` | `Single` + two-crossing counterfactuals; contradictions kept as PAIRWISE nogoods | fixed point + nogoods |
//! | `PairZ` | `Pair`, and zero-population pairs stored as nogoods too | as `Pair` |
//!
//! A pairwise nogood is never split into two unary exclusions. A unary
//! exclusion is derived from pairs only when every live partner letter failed.
//!
//! # Boundary relations (existential elimination)
//!
//! `pair_counts` is `n(a,b) = |D ∧ P(L,i,a) ∧ P(L,j,b)|` — the exact relation
//! of one slot projected on two of its cells, with multiplicities. `compose`
//! eliminates the shared cell: `N(x,y) = Σ_z n1(x,z) n2(z,y)`. A chain of
//! slots whose other cells are uncrossed (the only setting where a word has
//! internal variables; an NYT grid has none) folds to one 31×31 table, keyed
//! by a canonical `ChainKey` (language, dictionary fingerprint, links up to
//! reversal, no-repeat vacuity, unconstrained context).
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example crossword_crossing_care_probe`
//! Tests (CI): `cargo test -p cognitive-shader-driver --example crossword_crossing_care_probe`

use std::time::Instant;

use deepnsm_v2::vocab::WordId;
use ndarray::simd::{
    mask_andnot_assign, mask_ternlog_popcount, masked_group_count_u32_pair, ternlog,
};

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::Rng;

#[path = "shared/crossword_core.rs"]
mod crossword_core;
use crossword_core::*;

#[path = "shared/lab.rs"]
mod lab;

const SYMS: usize = MooreSymbol8::COUNT;

/// `|a ∧ b|` without writing anything.
fn pc2(a: &[u64], b: &[u64]) -> u64 {
    mask_ternlog_popcount::<{ ternlog::AND2 }>(a, b, b)
}

/// `|a ∧ b ∧ c|` without writing anything.
fn pc3(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    mask_ternlog_popcount::<{ ternlog::AND3 }>(a, b, c)
}

fn sym(x: usize) -> MooreSymbol8 {
    MooreSymbol8(x as u8)
}

fn letters(m: u32) -> impl Iterator<Item = usize> {
    (1..SYMS).filter(move |x| m >> x & 1 == 1)
}

// ─────────────────────────────── relations ───────────────────────────────

/// `n[a * SYMS + b]`: words with symbol `a` at the first position and `b` at
/// the second. A relation, not two domains.
#[derive(Clone, PartialEq, Eq, Debug)]
struct PairCounts {
    n: Vec<u64>,
}

impl PairCounts {
    fn zero() -> Self {
        PairCounts {
            n: vec![0; SYMS * SYMS],
        }
    }
    fn get(&self, a: usize, b: usize) -> u64 {
        self.n[a * SYMS + b]
    }
    fn support(&self) -> usize {
        self.n.iter().filter(|&&v| v > 0).count()
    }
    /// The letters each side can carry, read independently.
    fn marginals(&self) -> (u32, u32) {
        let (mut x, mut y) = (0u32, 0u32);
        for a in 0..SYMS {
            for b in 0..SYMS {
                if self.get(a, b) > 0 {
                    x |= 1 << a;
                    y |= 1 << b;
                }
            }
        }
        (x, y)
    }
    /// Pairs the product of the marginals admits that the relation does not.
    fn spurious_in_product(&self) -> usize {
        let (x, y) = self.marginals();
        (x.count_ones() * y.count_ones()) as usize - self.support()
    }
    fn transpose(&self) -> Self {
        let mut t = Self::zero();
        for a in 0..SYMS {
            for b in 0..SYMS {
                t.n[b * SYMS + a] = self.get(a, b);
            }
        }
        t
    }
}

/// The mask arm: fused AND popcounts over resident populations, nothing
/// written. Returns the relation and the number of population passes.
fn pair_counts_mask(hot: &Hot, len: usize, d: &[u64], i: usize, j: usize) -> (PairCounts, u64) {
    let mut out = PairCounts::zero();
    let mut ops = 0;
    for a in 1..SYMS {
        let pa = hot.pop(len, i, sym(a));
        ops += 1;
        if pc2(d, pa) == 0 {
            continue;
        }
        for b in 1..SYMS {
            ops += 1;
            out.n[a * SYMS + b] = pc3(d, pa, hot.pop(len, j, sym(b)));
        }
    }
    (out, ops)
}

/// The oracle: walk the candidate ids and read their spellings.
fn pair_counts_scan(hot: &Hot, d: &[u64], i: usize, j: usize) -> PairCounts {
    let mut out = PairCounts::zero();
    for w in bits(d) {
        out.n[hot.letter(w, i).0 as usize * SYMS + hot.letter(w, j).0 as usize] += 1;
    }
    out
}

/// Resident per-position letter columns: `lanes[i][w]` = the symbol of word
/// `w` at offset `i` (0 past its length). Built once per language, like the
/// populations.
fn letter_lanes(hot: &Hot) -> Vec<Vec<u32>> {
    let n = hot.len.len();
    (0..MAX_LEN)
        .map(|i| (0..n).map(|w| hot.spell[w * STRIDE + i].0 as u32).collect())
        .collect()
}

/// The shipped one-pass arm: mask-risc's `GroupReduce { Pair, Count }`
/// kernel, `ndarray::simd::masked_group_count_u32_pair`, keyed
/// `letter_i * SYMS + letter_j`. No composite lane is written.
fn pair_counts_group(lanes: &[Vec<u32>], d: &[u64], i: usize, j: usize) -> PairCounts {
    let mut out = vec![0i64; SYMS * SYMS];
    masked_group_count_u32_pair(d, &lanes[i], &lanes[j], SYMS as u32, &mut out);
    PairCounts {
        n: out.into_iter().map(|v| v as u64).collect(),
    }
}

/// Existential elimination of the shared cell: `Σ_z r1(x,z) r2(z,y)`.
fn compose(r1: &PairCounts, r2: &PairCounts) -> PairCounts {
    let mut out = PairCounts::zero();
    for x in 0..SYMS {
        for z in 0..SYMS {
            let a = r1.get(x, z);
            if a == 0 {
                continue;
            }
            for y in 0..SYMS {
                out.n[x * SYMS + y] += a * r2.get(z, y);
            }
        }
    }
    out
}

/// The adversary: a relation rebuilt from its marginals, weight 1 per pair.
#[cfg(test)]
fn from_marginals(r: &PairCounts) -> PairCounts {
    let (x, y) = r.marginals();
    let mut out = PairCounts::zero();
    for a in letters(x) {
        for b in letters(y) {
            out.n[a * SYMS + b] = 1;
        }
    }
    out
}

// ─────────────────────────────── cells ───────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Crossing {
    cell: u16,
    a: (usize, usize),
    b: (usize, usize),
}

fn crossings(puz: &Puzzle) -> Vec<Crossing> {
    puz.occupant
        .iter()
        .enumerate()
        .filter(|(_, o)| o[0] != NONE && o[1] != NONE)
        .map(|(c, o)| Crossing {
            cell: c as u16,
            a: untile(o[0]),
            b: untile(o[1]),
        })
        .collect()
}

fn cell_letter(hot: &Hot, puz: &Puzzle, st: &State, cell: u16) -> Option<usize> {
    puz.occupant[cell as usize]
        .iter()
        .filter(|&&t| t != NONE)
        .map(|&t| untile(t))
        .find(|&(s, _)| st.placed[s] != UNSET)
        .map(|(s, o)| hot.letter(st.placed[s], o).0 as usize)
}

/// Letters slot `s` can carry at offset `i`.
fn support(hot: &Hot, puz: &Puzzle, st: &State, (s, i): (usize, usize), ops: &mut u64) -> u32 {
    if st.placed[s] != UNSET {
        return 1 << hot.letter(st.placed[s], i).0;
    }
    let l = puz.len[s] as usize;
    let mut m = 0;
    for a in 1..SYMS {
        *ops += 1;
        if pc2(st.mask(s), hot.pop(l, i, sym(a))) > 0 {
            m |= 1 << a;
        }
    }
    m
}

/// Remove letter `a` from `(s, i)`. `Ok(true)` when the mask changed.
fn exclude(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    (s, i): (usize, usize),
    a: usize,
    queue: &mut Vec<u8>,
    ops: &mut u64,
) -> Result<bool, Stop> {
    if st.placed[s] != UNSET {
        return if hot.letter(st.placed[s], i).0 as usize == a {
            Err(Stop::Contradiction(s as u8))
        } else {
            Ok(false)
        };
    }
    let p = hot.pop(puz.len[s] as usize, i, sym(a));
    *ops += 1;
    if pc2(st.mask(s), p) == 0 {
        return Ok(false);
    }
    let b = st.blocks;
    mask_andnot_assign(&mut st.cand[s * b..(s + 1) * b], p);
    *ops += 2;
    match st.count(s) {
        0 => Err(Stop::Contradiction(s as u8)),
        1 => {
            st.placed[s] = first_bit(st.mask(s));
            queue.push(s as u8);
            Ok(true)
        }
        _ => Ok(true),
    }
}

/// Assume letter `a` on `cell`: both occupants narrow to it.
fn assume(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    cell: u16,
    a: usize,
    queue: &mut Vec<u8>,
) -> Result<(), Stop> {
    for t in puz.occupant[cell as usize] {
        if t == NONE {
            continue;
        }
        let (s, o) = untile(t);
        if st.placed[s] != UNSET {
            if hot.letter(st.placed[s], o).0 as usize != a {
                return Err(Stop::Contradiction(s as u8));
            }
            continue;
        }
        st.narrow(s, hot.pop(puz.len[s] as usize, o, sym(a)), queue)?;
    }
    Ok(())
}

// ─────────────────────────────── nogoods ───────────────────────────────

/// Pairwise nogoods `¬(cell X = a ∧ cell Y = b)`, valid in the root context
/// (the givens) they were derived in.
#[derive(Clone, Default, Debug)]
struct Nogoods {
    list: Vec<(u16, u8, u16, u8)>,
}

fn nogood_pass(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    ng: &Nogoods,
    queue: &mut Vec<u8>,
    ops: &mut u64,
) -> Result<bool, Stop> {
    let mut changed = false;
    for &(cx, a, cy, b) in &ng.list {
        let (lx, ly) = (cell_letter(hot, puz, st, cx), cell_letter(hot, puz, st, cy));
        let (other, ban) = match (lx == Some(a as usize), ly == Some(b as usize)) {
            (true, true) => return Err(Stop::Contradiction(0)),
            (true, false) if ly.is_none() => (cy, b),
            (false, true) if lx.is_none() => (cx, a),
            _ => continue,
        };
        for t in puz.occupant[other as usize] {
            if t != NONE {
                changed |= exclude(hot, puz, st, untile(t), ban as usize, queue, ops)?;
            }
        }
    }
    Ok(changed)
}

/// Letter support at every crossing whose two slots are both open.
fn support_pass(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    cr: &[Crossing],
    queue: &mut Vec<u8>,
    ops: &mut u64,
) -> Result<bool, Stop> {
    let mut changed = false;
    for c in cr {
        if st.placed[c.a.0] != UNSET || st.placed[c.b.0] != UNSET {
            continue;
        }
        let sa = support(hot, puz, st, c.a, ops);
        let sb = support(hot, puz, st, c.b, ops);
        for x in letters(sa & !sb) {
            changed |= exclude(hot, puz, st, c.a, x, queue, ops)?;
        }
        for x in letters(sb & !sa) {
            changed |= exclude(hot, puz, st, c.b, x, queue, ops)?;
        }
    }
    Ok(changed)
}

/// The fixed point, plus whatever the arm adds per node.
fn settle(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    queue: &mut Vec<u8>,
    cr: Option<&[Crossing]>,
    ng: Option<&Nogoods>,
    ops: &mut u64,
) -> Result<(), Stop> {
    loop {
        let a0 = st.ands;
        propagate_token(hot, puz, st, queue)?;
        *ops += (st.ands - a0) as u64;
        let mut changed = false;
        if let Some(cr) = cr {
            changed |= support_pass(hot, puz, st, cr, queue, ops)?;
        }
        if let Some(ng) = ng {
            changed |= nogood_pass(hot, puz, st, ng, queue, ops)?;
        }
        if queue.is_empty() && !changed {
            return Ok(());
        }
    }
}

/// Letter support cached per `(slot, offset)`, keyed by the slot's popcount
/// when it was computed. Masks only shrink, so an unchanged count means an
/// unchanged mask and a still-valid cache.
#[derive(Clone)]
struct SupCache {
    count: Vec<u64>,
    sup: Vec<u32>,
}

impl SupCache {
    fn new(slots: usize) -> Self {
        SupCache {
            count: vec![u64::MAX; slots],
            sup: vec![0; slots * STRIDE],
        }
    }
}

/// [`support_pass`], re-checking only crossings that touch a slot whose mask
/// changed since its support was cached.
fn support_pass_inc(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    cr: &[Crossing],
    cache: &mut SupCache,
    queue: &mut Vec<u8>,
    ops: &mut u64,
) -> Result<bool, Stop> {
    let mut fresh = vec![false; puz.slots()];
    for (s, f) in fresh.iter_mut().enumerate() {
        if st.placed[s] != UNSET {
            continue;
        }
        *ops += 1;
        let c = st.count(s);
        if c == cache.count[s] {
            continue;
        }
        cache.count[s] = c;
        *f = true;
        for o in 0..puz.len[s] as usize {
            if puz.cross[s * STRIDE + o] != NONE {
                cache.sup[s * STRIDE + o] = support(hot, puz, st, (s, o), ops);
            }
        }
    }
    let mut changed = false;
    for c in cr {
        if st.placed[c.a.0] != UNSET || st.placed[c.b.0] != UNSET {
            continue;
        }
        if !fresh[c.a.0] && !fresh[c.b.0] {
            continue;
        }
        let sa = cache.sup[c.a.0 * STRIDE + c.a.1];
        let sb = cache.sup[c.b.0 * STRIDE + c.b.1];
        for x in letters(sa & !sb) {
            changed |= exclude(hot, puz, st, c.a, x, queue, ops)?;
        }
        for x in letters(sb & !sa) {
            changed |= exclude(hot, puz, st, c.b, x, queue, ops)?;
        }
    }
    Ok(changed)
}

/// The fixed point plus incremental letter support.
fn settle_inc(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    queue: &mut Vec<u8>,
    cr: &[Crossing],
    cache: &mut SupCache,
    ops: &mut u64,
) -> Result<(), Stop> {
    loop {
        let a0 = st.ands;
        propagate_token(hot, puz, st, queue)?;
        *ops += (st.ands - a0) as u64;
        let changed = support_pass_inc(hot, puz, st, cr, cache, queue, ops)?;
        if queue.is_empty() && !changed {
            return Ok(());
        }
    }
}

// ─────────────────────────────── root counterfactuals ───────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Baseline,
    Support,
    SupportInc,
    Single,
    Pair,
    PairZ,
}

const ARMS: [Arm; 6] = [
    Arm::Baseline,
    Arm::Support,
    Arm::SupportInc,
    Arm::Single,
    Arm::Pair,
    Arm::PairZ,
];

/// Disable switches. Each one breaks a soundness rule on purpose.
#[derive(Clone, Copy, Debug, Default)]
struct Broken {
    /// A pairwise nogood becomes two unary exclusions.
    split_pairs: bool,
    /// A unary exclusion from the FIRST failing partner, not all of them.
    incomplete_unary: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Pre {
    probes: u64,
    cf_contradictions: u64,
    unary_zero_pop: u64,
    unary_probe: u64,
    pair_zero_pop: u64,
    pair_probe: u64,
    unary_from_pairs: u64,
    clone_bytes: u64,
    /// Σ popcount over all slots, before and after the root work.
    cands_before: u64,
    cands_after: u64,
    /// Candidates the refuted assumptions would have removed (observed, not committed).
    cf_pressure: u64,
}

fn total(st: &State, slots: usize) -> u64 {
    (0..slots).map(|s| st.count(s)).sum()
}

/// One counterfactual world: assume, settle, report whether it contradicts.
/// The original is untouched; the world is a copy that is dropped.
fn refuted(
    hot: &Hot,
    puz: &Puzzle,
    st: &State,
    asg: &[(u16, usize)],
    pre: &mut Pre,
    ops: &mut u64,
) -> bool {
    pre.probes += 1;
    pre.clone_bytes += (st.cand.len() * 8 + st.placed.len() * 2) as u64;
    let mut w = st.clone();
    let a0 = w.ands;
    let mut q = Vec::new();
    let assumed = asg
        .iter()
        .try_for_each(|&(c, a)| assume(hot, puz, &mut w, c, a, &mut q));
    // The assumption's own narrows; `settle` charges its work itself.
    *ops += (w.ands - a0) as u64;
    let r = assumed.and_then(|_| settle(hot, puz, &mut w, &mut q, None, None, ops));
    if r.is_err() {
        pre.cf_contradictions += 1;
        true
    } else {
        pre.cf_pressure += total(st, puz.slots()) - total(&w, puz.slots());
        false
    }
}

fn exclude_cell(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    cell: u16,
    a: usize,
    q: &mut Vec<u8>,
    ops: &mut u64,
) -> Result<(), Stop> {
    for t in puz.occupant[cell as usize] {
        if t != NONE {
            exclude(hot, puz, st, untile(t), a, q, ops)?;
        }
    }
    Ok(())
}

fn single_root(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    cr: &[Crossing],
    pre: &mut Pre,
    ops: &mut u64,
) -> Result<(), Stop> {
    loop {
        let mut changed = false;
        for c in cr {
            if cell_letter(hot, puz, st, c.cell).is_some() {
                continue;
            }
            let mut q = Vec::new();
            let sa = support(hot, puz, st, c.a, ops);
            let sb = support(hot, puz, st, c.b, ops);
            // Zero population on one side certifies the letter is impossible.
            for x in letters(sa ^ sb) {
                pre.unary_zero_pop += 1;
                exclude_cell(hot, puz, st, c.cell, x, &mut q, ops)?;
                changed = true;
            }
            for x in letters(sa & sb) {
                if refuted(hot, puz, st, &[(c.cell, x)], pre, ops) {
                    pre.unary_probe += 1;
                    exclude_cell(hot, puz, st, c.cell, x, &mut q, ops)?;
                    changed = true;
                }
            }
            settle(hot, puz, st, &mut q, None, None, ops)?;
        }
        if !changed {
            return Ok(());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair_root(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    store_zero: bool,
    broken: Broken,
    ng: &mut Nogoods,
    pre: &mut Pre,
    ops: &mut u64,
) -> Result<(), Stop> {
    for s in 0..puz.slots() {
        if st.placed[s] != UNSET {
            continue;
        }
        let l = puz.len[s] as usize;
        let open: Vec<(usize, u16)> = (0..l)
            .filter(|&i| puz.cross[s * STRIDE + i] != NONE)
            .map(|i| (i, puz.cell[s * STRIDE + i]))
            .filter(|&(_, c)| cell_letter(hot, puz, st, c).is_none())
            .collect();
        for p in 0..open.len() {
            for r in p + 1..open.len() {
                if st.placed[s] != UNSET {
                    break;
                }
                let ((i, cx), (j, cy)) = (open[p], open[r]);
                let (lx, ly) = (live(hot, puz, st, cx, ops), live(hot, puz, st, cy, ops));
                let mut bad = [0u32; SYMS];
                let mut q = Vec::new();
                for a in letters(lx) {
                    for b in letters(ly) {
                        *ops += 1;
                        let n = pc3(st.mask(s), hot.pop(l, i, sym(a)), hot.pop(l, j, sym(b)));
                        let fail = if n == 0 {
                            pre.pair_zero_pop += 1;
                            if store_zero {
                                ng.list.push((cx, a as u8, cy, b as u8));
                            }
                            true
                        } else if refuted(hot, puz, st, &[(cx, a), (cy, b)], pre, ops) {
                            pre.pair_probe += 1;
                            ng.list.push((cx, a as u8, cy, b as u8));
                            if broken.split_pairs {
                                exclude_cell(hot, puz, st, cx, a, &mut q, ops)?;
                                exclude_cell(hot, puz, st, cy, b, &mut q, ops)?;
                            }
                            true
                        } else {
                            false
                        };
                        if fail {
                            bad[a] |= 1 << b;
                        }
                    }
                }
                // A unary exclusion only when the search over partners was complete.
                for a in letters(lx) {
                    let all_failed = bad[a] == ly;
                    let any_failed = bad[a] != 0;
                    if all_failed || (broken.incomplete_unary && any_failed) {
                        pre.unary_from_pairs += 1;
                        exclude_cell(hot, puz, st, cx, a, &mut q, ops)?;
                    }
                }
                for b in letters(ly) {
                    if letters(lx).all(|a| bad[a] >> b & 1 == 1) {
                        pre.unary_from_pairs += 1;
                        exclude_cell(hot, puz, st, cy, b, &mut q, ops)?;
                    }
                }
                settle(hot, puz, st, &mut q, None, Some(ng), ops)?;
            }
        }
    }
    Ok(())
}

/// Letters both occupants of `cell` can carry.
fn live(hot: &Hot, puz: &Puzzle, st: &State, cell: u16, ops: &mut u64) -> u32 {
    puz.occupant[cell as usize]
        .iter()
        .filter(|&&t| t != NONE)
        .fold(u32::MAX, |m, &t| m & support(hot, puz, st, untile(t), ops))
}

// ─────────────────────────────── search ───────────────────────────────

#[derive(Clone, Default)]
struct Run {
    pre: Pre,
    root: Option<State>,
    ng: Nogoods,
    fills: u64,
    nodes: u64,
    ops: u64,
    root_ops: u64,
    exhausted: bool,
    ns: f64,
}

impl Run {
    fn solved(&self) -> bool {
        !self.exhausted
    }
}

struct Search<'a> {
    hot: &'a Hot,
    puz: &'a Puzzle,
    cr: Option<&'a [Crossing]>,
    /// Incremental support: the crossings, with a cache carried per state.
    inc: Option<&'a [Crossing]>,
    ng: Option<&'a Nogoods>,
    budget: u64,
    cap: u64,
    run: Run,
}

impl Search<'_> {
    fn go(&mut self, st: State, cache: Option<SupCache>) -> bool {
        let open: Vec<usize> = (0..self.puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .collect();
        self.run.ops += open.len() as u64;
        let Some(s) = open.into_iter().min_by_key(|&s| (st.count(s), s)) else {
            if no_repeats(&st.placed) {
                self.run.fills += 1;
            }
            return self.run.fills < self.cap;
        };
        for w in bits(st.mask(s)) {
            if st.placed.contains(&w) {
                continue;
            }
            if self.run.ops >= self.budget {
                self.run.exhausted = true;
                return false;
            }
            self.run.nodes += 1;
            let mut next = st.clone();
            let mut next_cache = cache.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            let mut ops = 0;
            let settled = match (self.inc, next_cache.as_mut()) {
                (Some(cr), Some(c)) => {
                    settle_inc(self.hot, self.puz, &mut next, &mut q, cr, c, &mut ops)
                }
                _ => settle(
                    self.hot, self.puz, &mut next, &mut q, self.cr, self.ng, &mut ops,
                ),
            };
            let ok = settled.is_ok() && no_repeats(&next.placed);
            self.run.ops += ops;
            if ok && !self.go(next, next_cache) {
                return false;
            }
        }
        true
    }
}

fn run(
    hot: &Hot,
    puz: &Puzzle,
    givens: &[(u8, WordId)],
    arm: Arm,
    broken: Broken,
    budget: u64,
    cap: u64,
) -> Run {
    let t = Instant::now();
    let cr = crossings(puz);
    let mut out = Run::default();
    let mut ops = 0;
    let Ok((mut st, mut q)) = start(hot, puz, givens) else {
        return out;
    };
    let use_support = arm == Arm::Support;
    let mut cache = (arm == Arm::SupportInc).then(|| SupCache::new(puz.slots()));
    let root = settle(hot, puz, &mut st, &mut q, None, None, &mut ops).and_then(|_| {
        out.pre.cands_before = total(&st, puz.slots());
        match arm {
            Arm::Baseline => Ok(()),
            Arm::Support => settle(hot, puz, &mut st, &mut q, Some(&cr), None, &mut ops),
            Arm::SupportInc => settle_inc(
                hot,
                puz,
                &mut st,
                &mut q,
                &cr,
                cache.as_mut().expect("the incremental arm carries a cache"),
                &mut ops,
            ),
            Arm::Single => single_root(hot, puz, &mut st, &cr, &mut out.pre, &mut ops),
            Arm::Pair | Arm::PairZ => {
                single_root(hot, puz, &mut st, &cr, &mut out.pre, &mut ops)?;
                pair_root(
                    hot,
                    puz,
                    &mut st,
                    arm == Arm::PairZ,
                    broken,
                    &mut out.ng,
                    &mut out.pre,
                    &mut ops,
                )
            }
        }
    });
    out.root_ops = ops;
    out.ops = ops;
    if root.is_err() {
        out.ns = t.elapsed().as_nanos() as f64;
        return out;
    }
    out.pre.cands_after = total(&st, puz.slots());
    out.root = Some(st.clone());
    let ng = std::mem::take(&mut out.ng);
    let mut s = Search {
        hot,
        puz,
        cr: use_support.then_some(cr.as_slice()),
        inc: cache.is_some().then_some(cr.as_slice()),
        ng: (!ng.list.is_empty()).then_some(&ng),
        budget,
        cap,
        run: out,
    };
    s.go(st, cache);
    let mut out = s.run;
    out.ng = ng;
    out.ns = t.elapsed().as_nanos() as f64;
    out
}

/// Independent soundness oracle: the planted solution must survive every
/// exclusion and violate no nogood.
fn sound(hot: &Hot, puz: &Puzzle, r: &Run, solution: &[WordId]) -> bool {
    let Some(root) = &r.root else {
        return false;
    };
    let kept = (0..puz.slots()).all(|s| {
        let w = solution[s];
        root.mask(s)[w as usize / 64] >> (w % 64) & 1 == 1
    });
    let letter_at = |cell: u16| {
        let (s, o) = untile(puz.occupant[cell as usize][0]);
        hot.letter(solution[s], o).0
    };
    kept && r
        .ng
        .list
        .iter()
        .all(|&(cx, a, cy, b)| !(letter_at(cx) == a && letter_at(cy) == b))
}

// ─────────────────────────────── chains ───────────────────────────────

/// One internal slot of a chain: its length, the offset it is entered at and
/// the offset it leaves by.
type Link = (u8, u8, u8);

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChainKey {
    lang: Lang,
    dict: u64,
    links: Vec<Link>,
    /// Every internal length is unique in the puzzle, so no-repeat cannot bind.
    no_repeat_vacuous: bool,
}

/// A chain read off one puzzle, between two end slots.
#[derive(Clone, Debug)]
struct Chain {
    /// The end slots' crossing offsets: `x` at the start slot, `y` at the end.
    x_off: usize,
    y_off: usize,
    links: Vec<Link>,
    no_repeat_vacuous: bool,
}

/// Walk from `from` through slots with exactly two crossings to `to`. `None`
/// when any internal slot has a third crossing (a constrained context).
fn chain(puz: &Puzzle, from: usize, to: usize) -> Option<Chain> {
    let crosses = |s: usize| -> Vec<(usize, usize, usize)> {
        (0..puz.len[s] as usize)
            .filter_map(|o| {
                let c = puz.cross[s * STRIDE + o];
                (c != NONE).then(|| {
                    let (t, j) = untile(c);
                    (o, t, j)
                })
            })
            .collect()
    };
    let start = crosses(from);
    if start.len() != 1 {
        return None;
    }
    let (x_off, mut cur, mut entered) = start[0];
    let mut prev = from;
    let mut links = Vec::new();
    while cur != to {
        let cs = crosses(cur);
        if cs.len() != 2 {
            return None;
        }
        let &(out, next, next_in) = cs.iter().find(|&&(_, t, _)| t != prev)?;
        links.push((puz.len[cur], entered as u8, out as u8));
        prev = cur;
        cur = next;
        entered = next_in;
        if links.len() > puz.slots() {
            return None;
        }
    }
    let internal: Vec<u8> = links.iter().map(|l| l.0).collect();
    let no_repeat_vacuous = internal
        .iter()
        .all(|&l| puz.len.iter().filter(|&&m| m == l).count() == 1);
    Some(Chain {
        x_off,
        y_off: entered,
        links,
        no_repeat_vacuous,
    })
}

/// The canonical key: links up to reversal. `true` when the reversed reading
/// was taken, so the stored relation is transposed for this instance.
fn canonical(c: &Chain, lang: Lang, dict: u64) -> (ChainKey, bool) {
    let rev: Vec<Link> = c.links.iter().rev().map(|&(l, i, o)| (l, o, i)).collect();
    let reversed = rev < c.links;
    (
        ChainKey {
            lang,
            dict,
            links: if reversed { rev } else { c.links.clone() },
            no_repeat_vacuous: c.no_repeat_vacuous,
        },
        reversed,
    )
}

/// FNV-1a over the populations a chain reads.
fn dict_fingerprint(hot: &Hot, lens: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |w: u64| {
        for byte in w.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut ls = lens.to_vec();
    ls.sort_unstable();
    ls.dedup();
    for l in ls {
        eat(u64::from(l));
        hot.all[l as usize].iter().for_each(|&w| eat(w));
        hot.at[l as usize].iter().for_each(|&w| eat(w));
    }
    h
}

/// Fold a canonical chain in an unconstrained context.
fn fold(hot: &Hot, links: &[Link]) -> PairCounts {
    let mut acc: Option<PairCounts> = None;
    for &(l, i, o) in links {
        let (r, _) = pair_counts_mask(
            hot,
            l as usize,
            &hot.all[l as usize],
            i as usize,
            o as usize,
        );
        acc = Some(match acc {
            None => r,
            Some(a) => compose(&a, &r),
        });
    }
    acc.expect("a chain has at least one link")
}

/// A stored boundary relation and the key it is valid under.
struct Boundary {
    key: ChainKey,
    rel: PairCounts,
}

/// Validate and read: `None` unless this instance's canonical key is the stored one.
fn reuse(
    b: &Boundary,
    hot: &Hot,
    puz: &Puzzle,
    from: usize,
    to: usize,
) -> Option<(Chain, PairCounts)> {
    let c = chain(puz, from, to)?;
    let lens: Vec<u8> = c.links.iter().map(|l| l.0).collect();
    let (key, reversed) = canonical(&c, puz.lang, dict_fingerprint(hot, &lens));
    (key == b.key).then(|| {
        (
            c,
            if reversed {
                b.rel.transpose()
            } else {
                b.rel.clone()
            },
        )
    })
}

fn grid(rows: &[&str]) -> Grid {
    Grid {
        side: rows.len(),
        rows: rows
            .iter()
            .map(|r| {
                r.bytes()
                    .enumerate()
                    .fold(0u32, |m, (c, b)| m | (u32::from(b == b'.') << c))
            })
            .collect(),
    }
}

/// Training: A(3) - B1(4: 0→3) - B2(5: 0→4) - C(6).
const T: &[&str] = &[
    "...######",
    "##.######",
    "##.######",
    "##.....##",
    "######.##",
    "######.##",
    "######.##",
    "######.##",
    "######.##",
];
/// One link: A(3) - B(4: 0→3) - C(6).
#[cfg(test)]
const T1: &[&str] = &[
    "...######",
    "##.######",
    "##.######",
    "##......#",
    "#########",
    "#########",
    "#########",
    "#########",
    "#########",
];
/// Held out: the same links, a 7-letter A, another place on the board.
const H1: &[&str] = &[
    ".......####",
    "######.####",
    "######.####",
    "######.....",
    "##########.",
    "##########.",
    "##########.",
    "##########.",
    "##########.",
    "###########",
    "###########",
];
/// Held out, walked from the other end: C'(6) - B2(5: 4→0) - B1(4: 3→0) - A'(3).
const H2: &[&str] = &["##....", "##.##.", "##.##.", "#####.", "......", "######"];
/// T with a 7-letter slot crossing B2's middle: the context is constrained,
/// and no length collides with an internal slot, so only context refuses it.
const H3: &[&str] = &[
    "...#######",
    "##.#######",
    "##.#######",
    "##.....###",
    "####.#.###",
    "####.#.###",
    "####.#.###",
    "####.#.###",
    "####.#.###",
    "####.#####",
];
/// Two internal slots of length 4: no-repeat binds.
const H4: &[&str] = &[
    "...######",
    "##.######",
    "##.######",
    "##....###",
    "#####.###",
    "#####.###",
    "#####.###",
    "#####.###",
    "#####.###",
];

/// T's links exactly, but A has length 5 like B2: only no-repeat refuses it.
const H5: &[&str] = &[
    ".....####",
    "####.####",
    "####.####",
    "####.....",
    "########.",
    "########.",
    "########.",
    "########.",
    "########.",
];

/// T's links between two 6-letter ends: a query may bind both ends to one word.
#[cfg(test)]
const H6: &[&str] = &[
    "......####",
    "#####.####",
    "#####.####",
    "#####.....",
    "#########.",
    "#########.",
    "#########.",
    "#########.",
    "#########.",
    "##########",
];

/// The end slots of a chain fixture: the 3- and 6-letter slots when both
/// exist (H3's extra slot also has one crossing), else the two one-crossing slots.
fn ends(puz: &Puzzle) -> (usize, usize) {
    if puz.len.iter().filter(|&&l| l == 3).count() == 1
        && puz.len.iter().filter(|&&l| l == 6).count() == 1
        && puz.len.iter().filter(|&&l| l == 7).count() == 1
    {
        return (slot_of_len(puz, 3), slot_of_len(puz, 6));
    }
    let one: Vec<usize> = (0..puz.slots())
        .filter(|&s| {
            (0..puz.len[s] as usize)
                .filter(|&o| puz.cross[s * STRIDE + o] != NONE)
                .count()
                == 1
        })
        .collect();
    (one[0], one[1])
}

fn slot_of_len(puz: &Puzzle, l: u8) -> usize {
    puz.len.iter().position(|&m| m == l).expect("fixture slot")
}

/// Words of one length, by id.
fn words_of(hot: &Hot, len: usize) -> Vec<WordId> {
    bits(&hot.all[len])
}

/// The independent oracle for one chain query: the core's own search.
fn oracle_count(hot: &Hot, puz: &Puzzle, from: usize, a: WordId, to: usize, c: WordId) -> usize {
    let mut b = usize::MAX;
    count_fills(
        hot,
        puz,
        &[(from as u8, a), (to as u8, c)],
        usize::MAX,
        &mut b,
    )
    .expect("unbounded")
}

/// The two end words are fixed by the query, not folded: no-repeat between
/// them is checked here (equal ids only happen when both ends share a length).
fn lookup(rel: &PairCounts, hot: &Hot, ch: &Chain, a: WordId, c: WordId) -> u64 {
    if a == c {
        return 0;
    }
    rel.get(
        hot.letter(a, ch.x_off).0 as usize,
        hot.letter(c, ch.y_off).0 as usize,
    )
}

fn train(hot: &Hot) -> (Boundary, f64) {
    let t = Instant::now();
    let puz = compile(&grid(T), Lang::En);
    let (f, to) = ends(&puz);
    let c = chain(&puz, f, to).expect("training chain");
    let lens: Vec<u8> = c.links.iter().map(|l| l.0).collect();
    let (key, reversed) = canonical(&c, Lang::En, dict_fingerprint(hot, &lens));
    assert!(!reversed, "the training chain is already canonical");
    let rel = fold(hot, &key.links);
    (Boundary { key, rel }, t.elapsed().as_nanos() as f64)
}

// ─────────────────────────────── report ───────────────────────────────

struct Created2 {
    c: Created,
}

fn make(hot: &Hot, side: usize, n: usize, seed: u64) -> Vec<Created2> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        if let Ok(c) = create(hot, side, &mut rng, 100_000) {
            out.push(Created2 { c });
        }
    }
    out
}

fn half(g: &[(u8, WordId)]) -> Vec<(u8, WordId)> {
    g[..g.len() / 2].to_vec()
}

fn arms_report(hot: &Hot, set: &[Created2], open: bool, budget: u64) {
    let cap = if open { 50 } else { 2 };
    println!(
        "  {:<8} {:>6} {:>9} {:>11} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8}",
        "arm",
        "solved",
        "nodes",
        "ops",
        "root ops",
        "probes",
        "u0",
        "uP",
        "pairs",
        "uPair",
        "clone MB",
        "ms"
    );
    let mut answers: Vec<Vec<Option<u64>>> = Vec::new();
    for arm in ARMS {
        let (mut solved, mut nodes, mut ops, mut rops, mut ns) = (0, 0, 0, 0, 0.0);
        let mut pre = Pre::default();
        let mut ans = Vec::new();
        let mut all_sound = true;
        for x in set {
            let g = if open {
                half(&x.c.givens)
            } else {
                x.c.givens.clone()
            };
            let r = run(hot, &x.c.puz, &g, arm, Broken::default(), budget, cap);
            all_sound &= sound(hot, &x.c.puz, &r, &x.c.solution);
            solved += u64::from(r.solved());
            nodes += r.nodes;
            ops += r.ops;
            rops += r.root_ops;
            ns += r.ns;
            ans.push(r.solved().then_some(r.fills));
            let p = r.pre;
            pre.probes += p.probes;
            pre.unary_zero_pop += p.unary_zero_pop;
            pre.unary_probe += p.unary_probe;
            pre.pair_probe += p.pair_probe;
            pre.unary_from_pairs += p.unary_from_pairs;
            pre.clone_bytes += p.clone_bytes;
        }
        assert!(
            all_sound,
            "{arm:?}: an exclusion removed the planted solution"
        );
        println!(
            "  {:<8} {:>3}/{:<2} {:>9} {:>11} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9.1} {:>8.1}",
            format!("{arm:?}"),
            solved,
            set.len(),
            nodes,
            ops,
            rops,
            pre.probes,
            pre.unary_zero_pop,
            pre.unary_probe,
            pre.pair_probe,
            pre.unary_from_pairs,
            pre.clone_bytes as f64 / 1e6,
            ns / 1e6
        );
        answers.push(ans);
    }
    for (k, a) in answers.iter().enumerate() {
        for (i, v) in a.iter().enumerate() {
            if let (Some(x), Some(y)) = (v, answers[0][i]) {
                assert_eq!(*x, y, "{:?} answer differs on puzzle {i}", ARMS[k]);
            }
        }
    }
}

/// L2: the open workload with no fill cap. A capped enumeration stops at a
/// different place depending on the search order; the uncapped tree does not.
fn uncapped_report(hot: &Hot, set: &[Created2]) {
    println!(" open, no fill cap (L2):");
    for arm in [
        Arm::Baseline,
        Arm::Support,
        Arm::SupportInc,
        Arm::Single,
        Arm::Pair,
    ] {
        let (mut nodes, mut fills, mut done, mut ops, mut ns) = (0, 0, 0, 0, 0.0);
        for x in set {
            let r = run(
                hot,
                &x.c.puz,
                &half(&x.c.givens),
                arm,
                Broken::default(),
                u64::MAX,
                u64::MAX,
            );
            assert!(sound(hot, &x.c.puz, &r, &x.c.solution), "{arm:?}");
            if r.solved() {
                done += 1;
                nodes += r.nodes;
                fills += r.fills;
                ops += r.ops;
                ns += r.ns;
            }
        }
        println!(
            "  {:<10} finished {done}/{}  nodes {nodes:>10}  fills {fills}  ops {ops:>11}  ms {:>9.1}",
            format!("{arm:?}"),
            set.len(),
            ns / 1e6
        );
    }
}

fn chain_report(hot: &Hot) {
    let t = Instant::now();
    let lanes = &letter_lanes(hot);
    println!(
        "  letter lanes: {} positions x {} ids, built in {:.1} us",
        lanes.len(),
        lanes[0].len(),
        t.elapsed().as_nanos() as f64 / 1e3
    );
    for (l, i, j) in [(4usize, 0usize, 3usize), (5, 0, 4)] {
        let t = Instant::now();
        let (m, passes) = pair_counts_mask(hot, l, &hot.all[l], i, j);
        let mask_ns = t.elapsed().as_nanos() as f64;
        let t = Instant::now();
        let scan = pair_counts_scan(hot, &hot.all[l], i, j);
        let scan_ns = t.elapsed().as_nanos() as f64;
        assert_eq!(m, scan);
        let t = Instant::now();
        let group = pair_counts_group(lanes, &hot.all[l], i, j);
        let group_ns = t.elapsed().as_nanos() as f64;
        assert_eq!(group, scan);
        println!(
            "  P_all({l}) on ({i},{j}): {} pairs of {} in the marginal product ({} spurious); mask {passes} passes {:.1} us, id scan {:.1} us, group-count {:.1} us",
            m.support(),
            m.support() + m.spurious_in_product(),
            m.spurious_in_product(),
            mask_ns / 1e3,
            scan_ns / 1e3,
            group_ns / 1e3
        );
    }
    let (b, construct_ns) = train(hot);
    println!(
        "  training key: links {:?}, no-repeat vacuous {}, support {} of 961 pairs, construct {:.1} us",
        b.key.links,
        b.key.no_repeat_vacuous,
        b.rel.support(),
        construct_ns / 1e3
    );
    for (name, rows) in [("H1", H1), ("H2", H2), ("H3", H3), ("H4", H4), ("H5", H5)] {
        let puz = compile(&grid(rows), Lang::En);
        let (f, to) = ends(&puz);
        let ok = reuse(&b, hot, &puz, f, to).is_some();
        println!("  {name}: reusable = {ok}");
    }
    let mut rng = Rng(7);
    for (name, rows) in [("H1", H1), ("H2", H2)] {
        let puz = compile(&grid(rows), Lang::En);
        let (f, to) = ends(&puz);
        let t = Instant::now();
        let (ch, rel) = reuse(&b, hot, &puz, f, to).expect("valid");
        let validate_ns = t.elapsed().as_nanos() as f64;
        // The same validation with the dictionary generation already known.
        let t = Instant::now();
        let c = chain(&puz, f, to).expect("valid");
        let (key, _) = canonical(&c, puz.lang, b.key.dict);
        assert_eq!(key, b.key);
        let key_only_ns = t.elapsed().as_nanos() as f64;
        let wa = words_of(hot, puz.len[f] as usize);
        let wc = words_of(hot, puz.len[to] as usize);
        let qs: Vec<(WordId, WordId)> = (0..1000)
            .map(|_| {
                (
                    wa[rng.below(wa.len() as u64) as usize],
                    wc[rng.below(wc.len() as u64) as usize],
                )
            })
            .collect();
        let t = Instant::now();
        let indep: Vec<usize> = qs
            .iter()
            .map(|&(a, c)| oracle_count(hot, &puz, f, a, to, c))
            .collect();
        let indep_ns = t.elapsed().as_nanos() as f64 / qs.len() as f64;
        let t = Instant::now();
        let folded: Vec<u64> = qs
            .iter()
            .map(|&(a, c)| lookup(&rel, hot, &ch, a, c))
            .collect();
        let lookup_ns = t.elapsed().as_nanos() as f64 / qs.len() as f64;
        for (x, y) in indep.iter().zip(&folded) {
            assert_eq!(*x as u64, *y, "{name}: folded count differs from search");
        }
        let nonzero = indep.iter().filter(|&&n| n > 0).count();
        println!(
            "  {name}: 1000 queries identical ({nonzero} with a completion); search {:.1} us/query, lookup {:.3} us, validate {:.1} us (key only {:.1} us)",
            indep_ns / 1e3,
            lookup_ns / 1e3,
            validate_ns / 1e3,
            key_only_ns / 1e3
        );
        for n in [1.0, 10.0, 100.0, 1000.0] {
            let sfcr = n * indep_ns / (construct_ns + validate_ns + n * lookup_ns);
            println!("      SFCR({n:>4}) = {sfcr:.2}");
        }
        println!(
            "      break-even N = {:.0} (fingerprint each instance)",
            ((construct_ns + validate_ns) / (indep_ns - lookup_ns)).ceil()
        );
    }
}

// ── D-SELF-CALIBRATING-LAB-0: the crossword arm of the hypothesis lab ──────

/// Pre-registered before the run that reports it. Directions come from
/// D-SCF-CARE-PAIR-0 (support: far fewer nodes, more ops; single and pair
/// counterfactuals do not pay). Endpoints are natural logs, so a minimum
/// effect of 0.1 means about 10 %. Fresh puzzles (seed 0x1AB5EED, not the
/// 0xC0FFEE sets the original measured), side 5, open workload.
const LAB_PUZZLES: usize = 16;
const LAB_PREREG: [lab::Prereg; 4] = [
    lab::Prereg {
        id: "C1",
        hypothesis: "letter support cuts search nodes",
        baseline: "Baseline",
        treatment: "Support",
        endpoint: "ln nodes",
        higher_is_better: false,
        min_effect: 0.1,
        blocks: LAB_PUZZLES,
    },
    lab::Prereg {
        id: "C2",
        hypothesis: "letter support cuts total ops",
        baseline: "Baseline",
        treatment: "Support",
        endpoint: "ln ops",
        higher_is_better: false,
        min_effect: 0.1,
        blocks: LAB_PUZZLES,
    },
    lab::Prereg {
        id: "C3",
        hypothesis: "single-crossing counterfactuals cut total ops",
        baseline: "Baseline",
        treatment: "Single",
        endpoint: "ln ops",
        higher_is_better: false,
        min_effect: 0.1,
        blocks: LAB_PUZZLES,
    },
    lab::Prereg {
        id: "C4",
        hypothesis: "pair nogoods cut total ops beyond single",
        baseline: "Single",
        treatment: "Pair",
        endpoint: "ln ops",
        higher_is_better: false,
        min_effect: 0.1,
        blocks: LAB_PUZZLES,
    },
];

fn lab_stage(hot: &Hot) {
    let t0 = Instant::now();
    println!("D-SELF-CALIBRATING-LAB-0 — crossword arm (side 5, open: half givens, cap 50)");
    let set = make(hot, 5, LAB_PUZZLES, 0x1AB5EED);
    let arms = [Arm::Baseline, Arm::Support, Arm::Single, Arm::Pair];
    let runs: Vec<Vec<Run>> = arms
        .iter()
        .map(|&arm| {
            set.iter()
                .map(|x| {
                    run(
                        hot,
                        &x.c.puz,
                        &half(&x.c.givens),
                        arm,
                        Broken::default(),
                        40_000_000,
                        50,
                    )
                })
                .collect()
        })
        .collect();
    // Deterministic half: every arm is sound and reaches the same answer.
    let sound_all = arms.iter().enumerate().all(|(k, _)| {
        set.iter()
            .zip(&runs[k])
            .all(|(x, r)| sound(hot, &x.c.puz, r, &x.c.solution))
    });
    let same = (1..arms.len()).all(|k| {
        runs[k]
            .iter()
            .zip(&runs[0])
            .all(|(a, b)| a.solved() != b.solved() || !a.solved() || a.fills == b.fills)
    });
    let solved: Vec<usize> = runs
        .iter()
        .map(|v| v.iter().filter(|r| r.solved()).count())
        .collect();
    println!(
        "deterministic: every arm sound {sound_all}; same fill count where both solved {same}; solved {solved:?} of {LAB_PUZZLES}"
    );
    let ln = |v: &[Run], f: fn(&Run) -> u64| {
        v.iter()
            .map(|r| (f(r).max(1) as f64).ln())
            .collect::<Vec<_>>()
    };
    let nodes = |r: &Run| r.nodes;
    let ops = |r: &Run| r.ops;
    let results = [
        lab::paired(&ln(&runs[0], nodes), &ln(&runs[1], nodes), false),
        lab::paired(&ln(&runs[0], ops), &ln(&runs[1], ops), false),
        lab::paired(&ln(&runs[0], ops), &ln(&runs[2], ops), false),
        lab::paired(&ln(&runs[2], ops), &ln(&runs[3], ops), false),
    ];
    let rejected = lab::holm(&results.map(|r| r.p), 0.05);
    println!("\nFamily (Holm, alpha 0.05; effects in ln units, + = treatment better):");
    for i in 0..LAB_PREREG.len() {
        println!(
            "  {}",
            lab::report(&LAB_PREREG[i], &results[i], rejected[i])
        );
    }
    println!("\nPareto (geometric mean over puzzles; empirical):");
    for (k, arm) in arms.iter().enumerate() {
        let g = |v: Vec<f64>| (v.iter().sum::<f64>() / v.len() as f64).exp();
        println!(
            "  {:<9} nodes {:>10.0}  ops {:>12.0}",
            format!("{arm:?}"),
            g(ln(&runs[k], nodes)),
            g(ln(&runs[k], ops))
        );
    }
    println!("\nlab wall time {:.1}s", t0.elapsed().as_secs_f64());
}

fn main() {
    let (hot, _cold) = build_lexicon(Lang::En, &english_ranked());
    if std::env::args().nth(1).as_deref() == Some("lab") {
        lab_stage(&hot);
        return;
    }
    println!("D-SCF-CARE-PAIR-0 (English, {} ids)", hot.len.len());
    if std::env::args().nth(1).as_deref() == Some("l2") {
        uncapped_report(&hot, &make(&hot, 5, 10, 0xC0FFEE + 5));
        return;
    }
    for side in [5usize, 7] {
        let set = make(&hot, side, 10, 0xC0FFEE + side as u64);
        let unchecked: usize = set
            .iter()
            .map(|x| {
                x.c.puz
                    .occupant
                    .iter()
                    .filter(|o| o[0] != NONE && o[1] == NONE)
                    .count()
            })
            .sum();
        println!(
            "\nside {side}: {} created puzzles; unchecked cells {unchecked}",
            set.len()
        );
        for (open, label) in [
            (false, "prove (all givens, cap 2)"),
            (true, "open (half givens, cap 50)"),
        ] {
            println!(" {label}");
            arms_report(&hot, &set, open, 40_000_000);
        }
        if side == 5 {
            uncapped_report(&hot, &set);
        }
    }
    println!("\nchain boundary relations:");
    chain_report(&hot);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn hot() -> &'static Hot {
        static H: OnceLock<Hot> = OnceLock::new();
        H.get_or_init(|| build_lexicon(Lang::En, &english_ranked()).0)
    }

    fn set() -> &'static [Created2] {
        static S: OnceLock<Vec<Created2>> = OnceLock::new();
        S.get_or_init(|| make(hot(), 5, 4, 0x5EED))
    }

    #[test]
    fn mask_relation_equals_spelling_scan() {
        let h = hot();
        for (l, i, j) in [(4, 0, 3), (5, 1, 3), (6, 0, 5)] {
            let (m, _) = pair_counts_mask(h, l, &h.all[l], i, j);
            assert_eq!(m, pair_counts_scan(h, &h.all[l], i, j));
            // a narrowed context, too
            let mut d = h.all[l].clone();
            ndarray::simd::mask_and_assign(&mut d, h.pop(l, 2, MooreSymbol8::of('e').unwrap()));
            assert_eq!(
                pair_counts_mask(h, l, &d, i, j).0,
                pair_counts_scan(h, &d, i, j)
            );
            assert!(m.support() > 0);
        }
    }

    /// The `{(a,0),(b,1)}` adversary: rebuilding a relation from its two
    /// domains invents pairs. On real slots too.
    #[test]
    fn product_of_marginals_invents_pairs() {
        let mut r = PairCounts::zero();
        r.n[SYMS + 2] = 1;
        r.n[3 * SYMS + 4] = 1;
        assert_eq!(r.support(), 2);
        assert_eq!(from_marginals(&r).support(), 4);
        assert_eq!(r.spurious_in_product(), 2);
        let h = hot();
        let (b1, _) = pair_counts_mask(h, 4, &h.all[4], 0, 3);
        assert!(b1.spurious_in_product() > 0);
    }

    #[test]
    fn composition_matches_the_core_search_and_marginals_do_not() {
        let h = hot();
        let puz = compile(&grid(T), Lang::En);
        let (f, to) = ends(&puz);
        let ch = chain(&puz, f, to).unwrap();
        assert_eq!(ch.links, vec![(4, 0, 3), (5, 0, 4)]);
        let rel = fold(h, &ch.links);
        let naive = compose(
            &from_marginals(&pair_counts_mask(h, 4, &h.all[4], 0, 3).0),
            &from_marginals(&pair_counts_mask(h, 5, &h.all[5], 0, 4).0),
        );
        let mut rng = Rng(3);
        let (wa, wc) = (words_of(h, 3), words_of(h, 6));
        for _ in 0..60 {
            let a = wa[rng.below(wa.len() as u64) as usize];
            let c = wc[rng.below(wc.len() as u64) as usize];
            assert_eq!(
                lookup(&rel, h, &ch, a, c),
                oracle_count(h, &puz, f, a, to, c) as u64
            );
        }
        let disagree = (0..SYMS * SYMS)
            .filter(|&k| (rel.n[k] > 0) != (naive.n[k] > 0))
            .count();
        // Two links: the composed support stays strictly inside the product.
        assert_eq!(rel.support(), 557);
        assert!(disagree > 0 && (0..SYMS * SYMS).all(|k| rel.n[k] == 0 || naive.n[k] > 0));
        // One link: the marginal product invents pairs the search refutes.
        let one = compile(&grid(T1), Lang::En);
        let (f, to) = ends(&one);
        let ch = chain(&one, f, to).unwrap();
        assert_eq!(ch.links, vec![(4, 0, 3)]);
        let exact = fold(h, &ch.links);
        let naive = from_marginals(&exact);
        let mut spurious_refuted = 0;
        for a in words_of(h, 3) {
            let x = h.letter(a, ch.x_off).0 as usize;
            for c in words_of(h, 6).into_iter().step_by(97) {
                let y = h.letter(c, ch.y_off).0 as usize;
                if naive.get(x, y) > 0 && exact.get(x, y) == 0 {
                    assert_eq!(oracle_count(h, &one, f, a, to, c), 0);
                    spurious_refuted += 1;
                }
            }
        }
        assert!(
            spurious_refuted > 0,
            "the marginal adversary must fail somewhere"
        );
    }

    #[test]
    fn canonical_reuse_accepts_compatible_and_refuses_the_rest() {
        let h = hot();
        let (b, _) = train(h);
        let open = |rows: &[&str]| {
            let puz = compile(&grid(rows), Lang::En);
            let (f, to) = ends(&puz);
            (puz, f, to)
        };
        for rows in [H1, H2] {
            let (puz, f, to) = open(rows);
            let (ch, rel) = reuse(&b, h, &puz, f, to).expect("compatible");
            let mut rng = Rng(11);
            let (wa, wc) = (
                words_of(h, puz.len[f] as usize),
                words_of(h, puz.len[to] as usize),
            );
            for _ in 0..40 {
                let a = wa[rng.below(wa.len() as u64) as usize];
                let c = wc[rng.below(wc.len() as u64) as usize];
                assert_eq!(
                    lookup(&rel, h, &ch, a, c),
                    oracle_count(h, &puz, f, a, to, c) as u64
                );
            }
        }
        // H2 is the reversed reading: its key matches only through transposition.
        let (puz, f, to) = open(H2);
        assert!(canonical(&chain(&puz, f, to).unwrap(), Lang::En, 0).1);
        // H3: a third crossing on an internal slot; H4: no-repeat binds.
        for rows in [H3, H4, H5] {
            let (puz, f, to) = open(rows);
            assert!(reuse(&b, h, &puz, f, to).is_none());
        }
    }

    /// Forced reuse where validation refuses must be wrong somewhere: the
    /// key's fields are load-bearing.
    #[test]
    fn forced_reuse_past_a_refused_key_is_wrong() {
        let h = hot();
        let (b, _) = train(h);
        // H3: same ends and offsets as T, extra slot on B2.
        let puz = compile(&grid(H3), Lang::En);
        let (f, to) = (slot_of_len(&puz, 3), slot_of_len(&puz, 6));
        let ch = Chain {
            x_off: 2,
            y_off: 0,
            links: b.key.links.clone(),
            no_repeat_vacuous: true,
        };
        let mut rng = Rng(5);
        let (wa, wc) = (words_of(h, 3), words_of(h, 6));
        let mut wrong = 0;
        for _ in 0..30 {
            let a = wa[rng.below(wa.len() as u64) as usize];
            let c = wc[rng.below(wc.len() as u64) as usize];
            wrong += u64::from(
                lookup(&b.rel, h, &ch, a, c) != oracle_count(h, &puz, f, a, to, c) as u64,
            );
        }
        assert!(wrong > 0, "H3 ignored context");
        // H4: compose(R4, R4) counts b1 == b2; the core's search does not.
        let puz = compile(&grid(H4), Lang::En);
        let (f, to) = ends(&puz);
        let ch = chain(&puz, f, to).unwrap();
        assert!(!ch.no_repeat_vacuous);
        let rel = fold(h, &ch.links);
        let t = MooreSymbol8::of('t').unwrap();
        let a = words_of(h, 3)
            .into_iter()
            .find(|&w| h.letter(w, 2) == t)
            .unwrap();
        let c = words_of(h, 6)
            .into_iter()
            .find(|&w| h.letter(w, 0) == t)
            .unwrap();
        assert!(lookup(&rel, h, &ch, a, c) > oracle_count(h, &puz, f, a, to, c) as u64);
        // H5: T's own links, so only the no-repeat field separates it. The
        // stored relation counts B2 == A; the search does not.
        let puz = compile(&grid(H5), Lang::En);
        let (f, to) = ends(&puz);
        let ch = chain(&puz, f, to).unwrap();
        assert_eq!(ch.links, b.key.links);
        assert!(!ch.no_repeat_vacuous);
        let w4 = words_of(h, 4);
        let a = words_of(h, 5)
            .into_iter()
            .find(|&a| {
                w4.iter().any(|&b1| {
                    h.letter(b1, 0) == h.letter(a, 4) && h.letter(b1, 3) == h.letter(a, 0)
                })
            })
            .unwrap();
        let c = words_of(h, 6)
            .into_iter()
            .find(|&c| h.letter(c, 0) == h.letter(a, 4))
            .unwrap();
        assert!(lookup(&b.rel, h, &ch, a, c) > oracle_count(h, &puz, f, a, to, c) as u64);
    }

    /// Ends of equal length: one word on both ends has a folded count but no
    /// fill, so the lookup must refuse it.
    #[test]
    fn equal_end_words_are_not_a_completion() {
        let h = hot();
        let (b, _) = train(h);
        let puz = compile(&grid(H6), Lang::En);
        let (f, to) = ends(&puz);
        assert_eq!((puz.len[f], puz.len[to]), (6, 6));
        let (ch, rel) = reuse(&b, h, &puz, f, to).expect("compatible");
        let w = words_of(h, 6)
            .into_iter()
            .find(|&w| {
                rel.get(
                    h.letter(w, ch.x_off).0 as usize,
                    h.letter(w, ch.y_off).0 as usize,
                ) > 0
            })
            .expect("a word whose own letters are supported");
        assert_eq!(oracle_count(h, &puz, f, w, to, w), 0);
        assert_eq!(lookup(&rel, h, &ch, w, w), 0);
    }

    #[test]
    fn every_arm_is_sound_and_agrees() {
        let h = hot();
        for x in set() {
            let g = half(&x.c.givens);
            let base = run(
                h,
                &x.c.puz,
                &g,
                Arm::Baseline,
                Broken::default(),
                3_000_000,
                50,
            );
            for arm in ARMS {
                let r = run(h, &x.c.puz, &g, arm, Broken::default(), 3_000_000, 50);
                assert!(sound(h, &x.c.puz, &r, &x.c.solution), "{arm:?}");
                if r.solved() && base.solved() {
                    assert_eq!(r.fills, base.fills, "{arm:?}");
                }
                assert!(r.pre.cands_after <= r.pre.cands_before);
            }
        }
    }

    /// L1: incremental support reaches the same fixed point at every node, so
    /// the search tree is the full-support tree, node for node.
    #[test]
    fn incremental_support_is_the_full_support_search() {
        let h = hot();
        let mut fired = 0;
        for x in set() {
            for g in [x.c.givens.clone(), half(&x.c.givens)] {
                let full = run(
                    h,
                    &x.c.puz,
                    &g,
                    Arm::Support,
                    Broken::default(),
                    3_000_000,
                    50,
                );
                let inc = run(
                    h,
                    &x.c.puz,
                    &g,
                    Arm::SupportInc,
                    Broken::default(),
                    3_000_000,
                    50,
                );
                let base = run(
                    h,
                    &x.c.puz,
                    &g,
                    Arm::Baseline,
                    Broken::default(),
                    3_000_000,
                    50,
                );
                assert!(full.solved() && inc.solved());
                assert_eq!((inc.nodes, inc.fills), (full.nodes, full.fills));
                assert_eq!(
                    inc.root.as_ref().map(|r| r.cand.clone()),
                    full.root.as_ref().map(|r| r.cand.clone())
                );
                assert!(sound(h, &x.c.puz, &inc, &x.c.solution));
                fired += u64::from(inc.nodes < base.nodes);
            }
        }
        assert!(
            fired > 0,
            "support must prune somewhere, or the equality is vacuous"
        );
    }

    /// The shipped group-count kernel is the same relation as the oracle.
    #[test]
    fn group_count_relation_equals_spelling_scan() {
        let h = hot();
        let lanes = letter_lanes(h);
        for (l, i, j) in [(4, 0, 3), (5, 1, 3), (6, 0, 5)] {
            let mut d = h.all[l].clone();
            assert_eq!(
                pair_counts_group(&lanes, &d, i, j),
                pair_counts_scan(h, &d, i, j)
            );
            ndarray::simd::mask_and_assign(&mut d, h.pop(l, 2, MooreSymbol8::of('e').unwrap()));
            assert_eq!(
                pair_counts_group(&lanes, &d, i, j),
                pair_counts_scan(h, &d, i, j)
            );
        }
    }

    /// Counterfactual worlds are copies: probing leaves the state it read untouched.
    #[test]
    fn a_refuted_world_restores_exactly() {
        let h = hot();
        let x = &set()[0];
        let (mut st, mut q) = start(h, &x.c.puz, &half(&x.c.givens)).unwrap();
        let mut ops = 0;
        settle(h, &x.c.puz, &mut st, &mut q, None, None, &mut ops).unwrap();
        let before = (st.cand.clone(), st.placed.clone());
        let mut pre = Pre::default();
        let mut refutations = 0;
        for c in crossings(&x.c.puz) {
            if cell_letter(h, &x.c.puz, &st, c.cell).is_none() {
                for a in letters(live(h, &x.c.puz, &st, c.cell, &mut ops)) {
                    refutations += u64::from(refuted(
                        h,
                        &x.c.puz,
                        &st,
                        &[(c.cell, a)],
                        &mut pre,
                        &mut ops,
                    ));
                }
            }
        }
        assert_eq!((st.cand.clone(), st.placed.clone()), before);
        assert!(pre.probes > 0);
        let _ = refutations;
    }

    /// Single-crossing counterfactuals can fire (remove a letter the fixed
    /// point kept) and can stay silent (the planted letter is never removed).
    #[test]
    fn single_counterfactuals_fire_and_stay_sound() {
        let h = hot();
        let fired: u64 = set()
            .iter()
            .map(|x| {
                let r = run(
                    h,
                    &x.c.puz,
                    &half(&x.c.givens),
                    Arm::Single,
                    Broken::default(),
                    1,
                    1,
                );
                assert!(sound(h, &x.c.puz, &r, &x.c.solution));
                r.pre.unary_probe + r.pre.unary_zero_pop
            })
            .sum();
        assert!(fired > 0);
    }

    /// The two disables: a pairwise nogood split into unary exclusions, and a
    /// unary exclusion from one failing partner. Each must remove a planted
    /// letter somewhere, while the honest arm never does.
    #[test]
    fn splitting_a_pair_or_skipping_completeness_is_unsound() {
        let h = hot();
        for broken in [
            Broken {
                split_pairs: true,
                incomplete_unary: false,
            },
            Broken {
                split_pairs: false,
                incomplete_unary: true,
            },
        ] {
            let mut unsound = 0;
            let mut pairs = 0;
            for x in set() {
                let g = half(&x.c.givens);
                let honest = run(h, &x.c.puz, &g, Arm::Pair, Broken::default(), 1, 1);
                assert!(sound(h, &x.c.puz, &honest, &x.c.solution));
                pairs += honest.pre.pair_probe + honest.pre.pair_zero_pop;
                let r = run(h, &x.c.puz, &g, Arm::Pair, broken, 1, 1);
                unsound += u64::from(!sound(h, &x.c.puz, &r, &x.c.solution));
            }
            assert!(pairs > 0, "no pairwise refutation to split");
            assert!(
                unsound > 0,
                "{broken:?} stayed sound: the falsifier cannot fire"
            );
        }
    }
}
