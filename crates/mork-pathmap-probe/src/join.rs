//! A versioned relationship table that puts both inputs of
//! `Q(x, y, z) :- R(x, y), S(y, z)` at one coordinate, so the join runs as a
//! fold.
//!
//! STATUS: PROBE. This is a candidate for the "co-addressed producer" the
//! Schnellverschluss plan (§7) names as the precondition of a native arm. It is
//! not admitted: D-RMAP-1 says a pre-fold global-to-local binding is a candidate
//! membrane that needs its own review, and nothing here is wired into Quack,
//! R2IL or the production mask-risc path.
//!
//! # Layout
//!
//! - [`JoinDomain`]: the sorted distinct values of the shared variable `y`
//!   across both datasets. A value's position is its **slot**. This is the one
//!   global-to-local translation, done once in [`JoinTable::prepare`] by binary
//!   search and timed there.
//! - [`Side`]: one input relation laid out by `y` slot, as an index map only:
//!   a presence plane (bit `y` set when the relation has a row with that `y`),
//!   `off[y]..off[y + 1]`, and in that range the POSITIONS of those rows in
//!   the relation's own lanes. The values are never copied: a slot's rows are
//!   read in place as `lane[pos[k]]` (a [`Run`]), the template
//!   `"{0}{1}" -f $laneA[$i],$laneB[$i]` with the positions as `$i`. Each side
//!   is prepared from its own dataset and carries that dataset's generation,
//!   so R and S can come from different sources and be resealed
//!   independently.
//! - Answers are folds: [`JoinExec::bag_count`] reads only the offsets,
//!   [`JoinExec::matching`] counts the derivations of a pattern, and
//!   [`JoinExec::for_each_answer`] hands out rows read in place. While one slot
//!   is read the next slot's runs are prefetched (`ndarray::simd`), because
//!   the lanes are other tables' memory than the plane that chose the slot.
//!
//! # Evaluation, and why it is not a gather
//!
//! Leapfrog triejoin with `y` first intersects R's `y` level with S's `y` level,
//! then enumerates `x` and `z` independently (they share no variable). Here the
//! intersection is one mask-risc `And` over the two presence planes, which sit
//! over the same slot space. Enumeration then reads `off[y]` and the positions
//! of each side at the surviving slot `y`, and reads the lane at them: every
//! read is at the coordinate the mask names or at a position the index map
//! holds, never at a position taken from another lane's values. The `x` and `z` values are emitted, never used
//! as addresses. No operand is read at a position taken from another operand's
//! values (2026-10-16 ruling).
//!
//! Multiplicities are kept, so this arm answers bag counts; both upstream
//! indexes are sets.

use lance_graph_mask_risc::{
    Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal, execute_into, words_for,
};

use crate::{Lane, RelationView};

/// Why the table did not answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinRefusal {
    /// A side was prepared from another generation than the caller holds now.
    Stale {
        side: usize,
        bound: u64,
        current: u64,
    },
    /// A re-prepared side has a `y` the domain does not contain. The domain is
    /// the shared coordinate space; changing it re-addresses every side, so the
    /// whole table must be prepared again.
    OutsideDomain { side: usize, y: u32 },
    /// The filter plane is not over this table's slot space.
    FilterShape { words: usize, expected: usize },
}

/// The shared coordinate space: sorted distinct `y` values.
///
/// When the values span a range no wider than [`DENSE_SPAN`] times their
/// count, the space is a presence mask over that range plus a running count
/// per mask word: a value's slot is the word's count plus the popcount of the
/// bits below it. No sort, no search, and slots stay in value order. A
/// sparser set falls back to a sorted list and binary search.
#[derive(Clone, Debug)]
pub struct JoinDomain {
    ids: Vec<u32>,
    dense: Option<Dense>,
}

/// A value range at most this many times wider than the values in it is laid
/// out as a mask.
pub const DENSE_SPAN: u64 = 8;

#[derive(Clone, Debug)]
struct Dense {
    base: u32,
    span: u64,
    words: Vec<u64>,
    rank: Vec<u32>,
}

impl Dense {
    fn slot_of(&self, y: u32) -> Option<usize> {
        let d = u64::from(y.wrapping_sub(self.base));
        if y < self.base || d >= self.span {
            return None;
        }
        let (w, b) = ((d >> 6) as usize, d & 63);
        let word = self.words[w];
        if word >> b & 1 == 0 {
            return None;
        }
        Some(self.rank[w] as usize + (word & ((1u64 << b) - 1)).count_ones() as usize)
    }
}

impl JoinDomain {
    /// The union of every side's `y` values. The iterator is walked twice:
    /// once for the range, once for the mask.
    pub fn build<I>(ys: I) -> Self
    where
        I: IntoIterator<Item = u32>,
        I::IntoIter: Clone,
    {
        let it = ys.into_iter();
        let (mut lo, mut hi, mut n) = (u32::MAX, 0u32, 0u64);
        for y in it.clone() {
            lo = lo.min(y);
            hi = hi.max(y);
            n += 1;
        }
        if n == 0 {
            return JoinDomain {
                ids: Vec::new(),
                dense: None,
            };
        }
        let span = u64::from(hi - lo) + 1;
        if span > DENSE_SPAN * n.max(64) {
            let mut ids: Vec<u32> = it.collect();
            ids.sort_unstable();
            ids.dedup();
            return JoinDomain { ids, dense: None };
        }
        let mut words = vec![0u64; span.div_ceil(64) as usize];
        for y in it {
            let d = u64::from(y - lo);
            words[(d >> 6) as usize] |= 1 << (d & 63);
        }
        let mut rank = Vec::with_capacity(words.len());
        let mut ids = Vec::new();
        for (w, &word) in words.iter().enumerate() {
            rank.push(ids.len() as u32);
            let mut m = word;
            while m != 0 {
                ids.push(lo + (w as u32) * 64 + m.trailing_zeros());
                m &= m - 1;
            }
        }
        JoinDomain {
            ids,
            dense: Some(Dense {
                base: lo,
                span,
                words,
                rank,
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Global `y` to slot. Preparation only.
    pub fn slot_of(&self, y: u32) -> Option<usize> {
        match &self.dense {
            Some(d) => d.slot_of(y),
            None => self.ids.binary_search(&y).ok(),
        }
    }

    /// Whether the space is laid out as a mask over its value range.
    pub fn is_dense(&self) -> bool {
        self.dense.is_some()
    }

    /// Whether `y` is in the space.
    pub fn contains(&self, y: u32) -> bool {
        self.slot_of(y).is_some()
    }

    /// Slot to global `y`, for emitting answers.
    pub fn id(&self, slot: usize) -> u32 {
        self.ids[slot]
    }

    pub fn bytes(&self) -> usize {
        self.ids.len() * 4 + self.dense.as_ref().map_or(0, |d| d.words.len() * 12)
    }
}

/// One relation laid out by `y` slot, as an index map only: a presence plane,
/// `off[y]..off[y + 1]`, and in that range `pos`, the positions of the rows
/// with that `y` in the relation's own lanes. The values stay in the lane;
/// nothing is copied out of it. This map is the global-to-local inversion,
/// the one thing preparation materializes.
#[derive(Clone, Debug)]
pub struct Side {
    pub generation: u64,
    present: Vec<u64>,
    off: Vec<u32>,
    pos: Vec<u32>,
    /// The non-key lane, shared with the relation (a reference count, not a
    /// copy). Read at `pos`.
    lane: Lane,
}

/// The rows of one side at one slot, read in place: `lane[pos[k]]`. A
/// borrowed view; it holds positions and the lane, never values.
#[derive(Clone, Copy, Debug, Default)]
pub struct Run<'a> {
    lane: &'a [u32],
    pos: &'a [u32],
}

impl<'a> Run<'a> {
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// The `k`-th row's value, read from the lane at its position.
    pub fn get(&self, k: usize) -> u32 {
        self.lane[self.pos[k] as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + 'a {
        let lane = self.lane;
        self.pos.iter().map(move |&p| lane[p as usize])
    }

    /// Rows whose value equals `v`.
    pub fn count_eq(&self, v: u32) -> usize {
        self.iter().filter(|&x| x == v).count()
    }

    /// Asks for the run's first lane cache line ahead of its read: the lane
    /// sits in another table's memory than the plane that chose the slot.
    pub fn prefetch(&self) {
        if let Some(&p) = self.pos.first() {
            ndarray::simd::prefetch_read_t0(self.lane[p as usize..].as_ptr().cast());
        }
    }
}

impl Side {
    /// Lays the relation out by the slot of lane `key`; the other lane holds
    /// the value. A counting sort over slots, so positions within one slot
    /// are in row order.
    pub fn prepare(
        domain: &JoinDomain,
        view: &RelationView,
        key: usize,
        side: usize,
    ) -> Result<Side, JoinRefusal> {
        Self::prepare_where(domain, view, key, side, None)
    }

    /// [`Self::prepare`] over the rows the `keep` mask (one bit per row
    /// position; `None`: every row) admits. A restriction is a selection of
    /// positions, never a copy of rows: the index map walks the kept bits and
    /// leaves the other positions out.
    pub fn prepare_where(
        domain: &JoinDomain,
        view: &RelationView,
        key: usize,
        side: usize,
        keep: Option<&[u64]>,
    ) -> Result<Side, JoinRefusal> {
        let n = domain.len();
        let keys = &view.cols[key];
        let mut off = vec![0u32; n + 1];
        // Each kept row's slot, once (scratch), in kept-row order.
        let mut slots = Vec::new();
        for row in rows(keys.len(), keep) {
            let y = keys[row];
            let s = domain
                .slot_of(y)
                .ok_or(JoinRefusal::OutsideDomain { side, y })?;
            slots.push(s as u32);
            off[s + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut pos = vec![0u32; slots.len()];
        for (row, &s) in rows(keys.len(), keep).zip(&slots) {
            let at = &mut fill[s as usize];
            pos[*at as usize] = row as u32;
            *at += 1;
        }
        let mut present = vec![0u64; words_for(n)];
        for s in 0..n {
            if off[s + 1] > off[s] {
                present[s >> 6] |= 1 << (s & 63);
            }
        }
        Ok(Side {
            generation: view.generation,
            present,
            off,
            pos,
            lane: view.cols[1 - key].clone(),
        })
    }

    /// The rows at slot `y`, read in place.
    pub fn at(&self, y: usize) -> Run<'_> {
        Run {
            lane: &self.lane,
            pos: &self.pos[self.off[y] as usize..self.off[y + 1] as usize],
        }
    }

    /// Rows the index map holds, over all slots.
    pub fn rows(&self) -> usize {
        self.pos.len()
    }

    /// Rows at slot `y`, from the offsets alone (no lane read).
    pub fn count(&self, y: usize) -> usize {
        (self.off[y + 1] - self.off[y]) as usize
    }

    pub fn present(&self) -> &[u64] {
        &self.present
    }

    /// `off`: the start of each slot's run, `domain.len() + 1` entries.
    /// Row `k` of the side sits in slot `s` for `off[s] <= k < off[s + 1]`.
    pub fn offsets(&self) -> &[u32] {
        &self.off
    }

    /// Row positions into the relation's lanes, in slot order.
    pub fn positions(&self) -> &[u32] {
        &self.pos
    }

    /// The non-key lane, shared with the relation. The side's `k`-th value
    /// is `lane()[positions()[k]]`, read in place.
    pub fn lane(&self) -> &[u32] {
        &self.lane
    }

    /// Bytes the index map adds; the lane is shared with the relation and
    /// not counted.
    pub fn bytes(&self) -> usize {
        (self.present.len() * 8) + (self.off.len() * 4) + (self.pos.len() * 4)
    }
}

/// The versioned relationship table: one domain, one side per relation, each
/// side tied to its own dataset generation.
#[derive(Clone, Debug)]
pub struct JoinTable {
    pub domain: JoinDomain,
    pub sides: Vec<Side>,
}

/// One input of [`JoinTable::prepare_star_where`]: the relation, the lane
/// holding the shared variable, and which row positions to keep: a mask, one
/// bit per row (`None`: every row).
pub type SelectedSide<'a> = (&'a RelationView, usize, Option<&'a [u64]>);

/// The row positions `keep` admits, in row order: every row, or the set bits.
#[derive(Clone, Debug)]
pub enum Rows<'a> {
    All(std::ops::Range<usize>),
    Set {
        words: &'a [u64],
        n: usize,
        wi: usize,
        w: u64,
    },
}

/// The kept row positions of a relation with `n` rows.
pub fn rows(n: usize, keep: Option<&[u64]>) -> Rows<'_> {
    match keep {
        None => Rows::All(0..n),
        Some(words) => Rows::Set {
            words,
            n,
            wi: 0,
            w: words.first().copied().unwrap_or(0),
        },
    }
}

impl Iterator for Rows<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Rows::All(r) => r.next(),
            Rows::Set { words, n, wi, w } => loop {
                if *w != 0 {
                    let row = *wi * 64 + w.trailing_zeros() as usize;
                    *w &= *w - 1;
                    if row < *n {
                        return Some(row);
                    }
                    continue;
                }
                *wi += 1;
                *w = *words.get(*wi)?;
            },
        }
    }
}

/// Where the shared variable sits in each input row: `R(x, y)` keys column 1,
/// `S(y, z)` keys column 0.
pub const R_KEY: usize = 1;
pub const S_KEY: usize = 0;

impl JoinTable {
    /// Prepares the table for `R(x, y), S(y, z)`.
    pub fn prepare(r: &RelationView, s: &RelationView) -> JoinTable {
        Self::prepare_star(&[(r, R_KEY), (s, S_KEY)])
    }

    /// Prepares a star of relations that all share `y`: every relation is laid
    /// out over the same domain. `key` names the column holding `y`.
    pub fn prepare_star(inputs: &[(&RelationView, usize)]) -> JoinTable {
        let with: Vec<SelectedSide<'_>> = inputs.iter().map(|&(v, k)| (v, k, None)).collect();
        Self::prepare_star_where(&with)
    }

    /// [`Self::prepare_star`] over the rows each side's `keep` admits (by row
    /// position). The domain is the union of the kept rows' keys only.
    pub fn prepare_star_where(inputs: &[SelectedSide<'_>]) -> JoinTable {
        let domain = JoinDomain::build(inputs.iter().flat_map(|&(v, key, keep)| {
            let lane = &v.cols[key];
            rows(lane.len(), keep).map(move |row| lane[row])
        }));
        let sides = inputs
            .iter()
            .enumerate()
            .map(|(i, &(v, key, keep))| {
                Side::prepare_where(&domain, v, key, i, keep).expect("domain is the union")
            })
            .collect();
        JoinTable { domain, sides }
    }

    /// Re-prepares one side from a resealed dataset, keeping the domain and the
    /// other sides. Refuses if the new rows leave the domain.
    pub fn replace_side(
        &mut self,
        side: usize,
        view: &RelationView,
        key: usize,
    ) -> Result<(), JoinRefusal> {
        self.sides[side] = Side::prepare(&self.domain, view, key, side)?;
        Ok(())
    }

    /// Refuses unless every side's generation is the one the caller holds.
    pub fn check(&self, current: &[u64]) -> Result<(), JoinRefusal> {
        for (i, (side, &now)) in self.sides.iter().zip(current).enumerate() {
            if side.generation != now {
                return Err(JoinRefusal::Stale {
                    side: i,
                    bound: side.generation,
                    current: now,
                });
            }
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.domain.bytes() + self.sides.iter().map(Side::bytes).sum::<usize>()
    }

    pub fn words(&self) -> usize {
        words_for(self.domain.len())
    }
}

/// Runs the presence-plane intersection through mask-risc. Owns its output
/// plane and scratch, so repeated queries allocate nothing in the fold.
pub struct JoinExec {
    out: Vec<u64>,
    scratch: Scratch<'static>,
}

impl JoinExec {
    pub fn new(table: &JoinTable) -> Self {
        let words = table.words();
        JoinExec {
            out: vec![0; words],
            scratch: Scratch::new(words, 1),
        }
    }

    /// `joint = present_0 ∧ … ∧ present_k [∧ filter]`: the slots where every
    /// relation has a row (and the thought admits the slot).
    pub fn joint(
        &mut self,
        table: &JoinTable,
        filter: Option<&[u64]>,
    ) -> Result<&[u64], JoinRefusal> {
        let words = table.words();
        if let Some(f) = filter
            && f.len() != words
        {
            return Err(JoinRefusal::FilterShape {
                words: f.len(),
                expected: words,
            });
        }
        assert!(table.sides.len() >= 2, "a join needs two relations");
        let mut masks: Vec<&[u64]> = table.sides.iter().map(|s| s.present()).collect();
        if let Some(f) = filter {
            masks.push(f);
        }
        let mut ops = Vec::with_capacity(masks.len());
        let mut acc = Operand::Plane(0);
        for i in 1..masks.len() {
            ops.push(MaskOp::And {
                a: acc,
                b: Operand::Plane(i as u16),
                dst: 0,
            });
            acc = Operand::Scratch(0);
        }
        let prog = Program::new(
            ops,
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        let pl = Planes {
            n_rows: table.domain.len(),
            masks: &masks,
            lanes: &[],
        };
        execute_into(
            &prog,
            &pl,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(&mut self.out),
        )
        .expect("join program is valid");
        Ok(&self.out)
    }

    /// Derivations reading as `[x, y, z]` (`None`: any value): a fold. A
    /// pinned `y` is inverted to its slot and only that slot is read;
    /// otherwise every joint slot. Per slot it is `count_R(x) · count_S(z)`,
    /// with a free column counted from the offsets alone.
    pub fn matching(
        &mut self,
        table: &JoinTable,
        current: (u64, u64),
        filter: Option<&[u64]>,
        pattern: [Option<u32>; 3],
    ) -> Result<u64, JoinRefusal> {
        table.check(&[current.0, current.1])?;
        let n = table.domain.len();
        let joint = self.joint(table, filter)?;
        let (r, s) = (&table.sides[0], &table.sides[1]);
        let at = |slot: usize| -> u64 {
            let cr = pattern[0].map_or(r.count(slot), |x| r.at(slot).count_eq(x));
            let cs = pattern[2].map_or(s.count(slot), |z| s.at(slot).count_eq(z));
            cr as u64 * cs as u64
        };
        if let Some(y) = pattern[1] {
            let Some(slot) = table.domain.slot_of(y) else {
                return Ok(0);
            };
            let kept = joint[slot >> 6] >> (slot & 63) & 1 == 1;
            return Ok(if kept { at(slot) } else { 0 });
        }
        let mut total = 0u64;
        for_each_set_ahead(
            joint,
            n,
            |next| {
                r.at(next).prefetch();
                s.at(next).prefetch();
            },
            |slot| total += at(slot),
        );
        Ok(total)
    }

    /// Every bag derivation, in slot order, as a row read in place: `f` gets
    /// a [`JoinRow`] whose columns are lane reads at the row positions. The
    /// next slot's runs are prefetched while this one is read.
    pub fn for_each_answer(
        &mut self,
        table: &JoinTable,
        current: (u64, u64),
        mut f: impl FnMut(&JoinRow<'_>),
    ) -> Result<(), JoinRefusal> {
        table.check(&[current.0, current.1])?;
        let n = table.domain.len();
        let joint = self.joint(table, None)?;
        let (r, s) = (&table.sides[0], &table.sides[1]);
        for_each_set_ahead(
            joint,
            n,
            |next| {
                r.at(next).prefetch();
                s.at(next).prefetch();
            },
            |slot| {
                let (rr, sr) = (r.at(slot), s.at(slot));
                let y = table.domain.id(slot);
                for i in 0..rr.len() {
                    for j in 0..sr.len() {
                        f(&JoinRow {
                            y,
                            r: rr,
                            s: sr,
                            i,
                            j,
                        });
                    }
                }
            },
        );
        Ok(())
    }

    /// The number of bag answers of the star, `Σ_y Π_i |side_i at y|`, from
    /// the offsets alone: no lane is read.
    pub fn bag_count(
        &mut self,
        table: &JoinTable,
        current: &[u64],
        filter: Option<&[u64]>,
    ) -> Result<u64, JoinRefusal> {
        table.check(current)?;
        let n = table.domain.len();
        let joint = self.joint(table, filter)?;
        let mut total = 0u64;
        for_each_set(joint, n, |slot| {
            total += table
                .sides
                .iter()
                .map(|s| s.count(slot) as u64)
                .product::<u64>();
        });
        Ok(total)
    }
}

/// One derivation of `R(x, y), S(y, z)`, read in place: `x` is R's run at
/// row `i`, `z` is S's run at row `j`. The template `"{x}{y}{z}"` over the
/// lanes; nothing is copied.
#[derive(Clone, Copy, Debug)]
pub struct JoinRow<'a> {
    y: u32,
    r: Run<'a>,
    s: Run<'a>,
    i: usize,
    j: usize,
}

impl JoinRow<'_> {
    pub fn x(&self) -> u32 {
        self.r.get(self.i)
    }

    pub fn y(&self) -> u32 {
        self.y
    }

    pub fn z(&self) -> u32 {
        self.s.get(self.j)
    }
}

/// Calls `ahead(next)` for the next set bit before `f(slot)` for the current
/// one: the next slot's reads are requested while this one is processed.
pub fn for_each_set_ahead(
    words: &[u64],
    n: usize,
    mut ahead: impl FnMut(usize),
    mut f: impl FnMut(usize),
) {
    let mut pending: Option<usize> = None;
    for_each_set(words, n, |slot| {
        ahead(slot);
        if let Some(prev) = pending.replace(slot) {
            f(prev);
        }
    });
    if let Some(last) = pending {
        f(last);
    }
}

/// Calls `f` for every set bit below `n`, in slot order.
pub fn for_each_set(words: &[u64], n: usize, mut f: impl FnMut(usize)) {
    for (wi, &w) in words.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            let slot = wi * 64 + b;
            if slot < n {
                f(slot);
            }
            w &= w - 1;
        }
    }
}

/// A filter plane over the table's slots from a predicate on the global `y`.
/// Built per thought, from the domain, never from the sides.
pub fn filter_plane(table: &JoinTable, mut keep: impl FnMut(u32) -> bool) -> Vec<u64> {
    let n = table.domain.len();
    let mut p = vec![0u64; words_for(n)];
    for slot in 0..n {
        if keep(table.domain.id(slot)) {
            p[slot >> 6] |= 1 << (slot & 63);
        }
    }
    p
}
