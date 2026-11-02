//! Wide transitive closure: more than 64 positions, block presence, and a
//! sparse delta log that is merged on read.
//!
//! A *position* is a wide coordinate `tile_ord * 256 + local`. One closure row
//! is split into 256-position *blocks* (one dt64 tile, four `u64` words). A
//! block is allocated only when some bit in it is set, so a row that reaches
//! two tiles of a hundred stores two blocks, and the presence words say which.
//!
//! Everything here is a free function over parallel `src` / `dst` lanes so the
//! kernel can be tested without a `Cube`.
//!
//! Invariants worth knowing before editing:
//!
//! * Positions never renumber. A delta (and the old closure it is applied to)
//!   names positions by value, so renumbering one position would make every
//!   older row and every logged edge point at the wrong symbol. New symbols
//!   are appended at the end (`n_new >= old.n()`); a vanished symbol keeps its
//!   position and simply loses its edges.
//! * The closure is a derived read. It is never persisted as a delta; it is
//!   re-derived for the rows a change can reach.
//! * Deletions cannot be merged into a closure row: a removed edge may or may
//!   not have been the only path to a target. So every row that could reach a
//!   retired source is recomputed, and every other row is kept bit for bit.
//!   Insertions are monotone, so they are merged exactly by seeding.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::simd::{mask_andnot, mask_any, mask_or_assign};

/// Positions in one block: one dt64 tile.
pub const BLOCK_POSITIONS: usize = 256;

/// `u64` words in one block.
pub const BLOCK_WORDS: usize = 4;

/// Largest supported position count. Row offsets are `u32` block slots, so
/// `n * ceil(n / 256)` must fit in 32 bits; this bound guarantees it.
pub const MAX_POSITIONS: u32 = (1 << 20) - 256;

/// Closure (or edge) rows with block presence: a block is allocated only if
/// some word of it is nonzero. An absent block reads as zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WideRows {
    n: u32,
    blocks: u32,
    /// `n * pw` words, `pw = ceil(blocks / 64)`; bit `b` of a row's presence
    /// words says block `b` is allocated for that row.
    presence: Vec<u64>,
    /// Per row: the slot (in blocks) where its packed blocks start in `words`.
    offset: Vec<u32>,
    /// The present blocks only, four words each, row after row, block order.
    words: Vec<u64>,
}

/// Presence words per row for `blocks` blocks.
fn pw_of(blocks: u32) -> usize {
    (blocks as usize).div_ceil(64)
}

/// Calls `f` with the index of every set bit, lowest first.
fn for_each_bit(words: &[u64], mut f: impl FnMut(usize)) {
    for (wi, &w) in words.iter().enumerate() {
        let mut bits = w;
        while bits != 0 {
            f(wi * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

impl WideRows {
    /// An empty builder for `n` rows of `blocks` blocks.
    fn builder(n: u32, blocks: u32) -> Self {
        Self {
            n,
            blocks,
            presence: Vec::with_capacity(n as usize * pw_of(blocks)),
            offset: Vec::with_capacity(n as usize),
            words: Vec::new(),
        }
    }

    /// Appends the next row from a dense scratch of `blocks * 4` words,
    /// allocating exactly the blocks that have a set bit.
    fn push_dense_row(&mut self, dense: &[u64]) {
        let pw = pw_of(self.blocks);
        let start = self.presence.len();
        self.presence.resize(start + pw, 0);
        // Capacity is checked by `check_dims`, so the slot count fits `u32`.
        self.offset
            .push(u32::try_from(self.words.len() / BLOCK_WORDS).unwrap_or(u32::MAX));
        for b in 0..self.blocks as usize {
            let Some(chunk) = dense.get(b * BLOCK_WORDS..(b + 1) * BLOCK_WORDS) else {
                break;
            };
            if chunk.iter().any(|&w| w != 0) {
                if let Some(p) = self.presence.get_mut(start + b / 64) {
                    *p |= 1 << (b % 64);
                }
                self.words.extend_from_slice(chunk);
            }
        }
    }

    /// Positions (rows and columns).
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Blocks per row: `ceil(n / 256)`.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Blocks actually stored, over all rows.
    pub fn blocks_allocated(&self) -> u64 {
        (self.words.len() / BLOCK_WORDS) as u64
    }

    /// Whether position `y` is in row `x`. Tests the presence bit, then ranks
    /// it with a popcount to find the stored block; there is no search.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        if x >= self.n || y >= self.n {
            return false;
        }
        let pw = pw_of(self.blocks);
        let b = y as usize / BLOCK_POSITIONS;
        let (j, bit) = (b / 64, b % 64);
        let Some(row) = self.presence.get(x as usize * pw..(x as usize + 1) * pw) else {
            return false;
        };
        let Some(&word) = row.get(j) else {
            return false;
        };
        if word >> bit & 1 == 0 {
            return false;
        }
        let below: u32 = row.iter().take(j).map(|w| w.count_ones()).sum::<u32>()
            + (word & ((1u64 << bit) - 1)).count_ones();
        let Some(&start) = self.offset.get(x as usize) else {
            return false;
        };
        let slot = start as usize + below as usize;
        let within = (y as usize % BLOCK_POSITIONS) / 64;
        self.words
            .get(slot * BLOCK_WORDS + within)
            .is_some_and(|w| w >> (y % 64) & 1 == 1)
    }

    /// Writes row `x` as dense words into `out` (`ceil(n / 64)` words are
    /// meaningful; a longer `out` is zero-extended, a shorter one truncated).
    /// Absent blocks and rows `x >= n` read as zero.
    pub fn row_into(&self, x: u32, out: &mut [u64]) {
        out.fill(0);
        if x >= self.n {
            return;
        }
        let pw = pw_of(self.blocks);
        let Some(row) = self.presence.get(x as usize * pw..(x as usize + 1) * pw) else {
            return;
        };
        let Some(&start) = self.offset.get(x as usize) else {
            return;
        };
        let mut slot = start as usize;
        for_each_bit(row, |b| {
            if let Some(src) = self.words.get(slot * BLOCK_WORDS..(slot + 1) * BLOCK_WORDS) {
                for (k, &w) in src.iter().enumerate() {
                    if let Some(d) = out.get_mut(b * BLOCK_WORDS + k) {
                        *d = w;
                    }
                }
            }
            slot += 1;
        });
    }

    /// ORs row `y` into `dst` (dense, at least `blocks * 4` words), touching
    /// only the blocks row `y` has; absent blocks cost nothing.
    fn or_row_into(&self, y: u32, dst: &mut [u64]) {
        if y >= self.n {
            return;
        }
        let pw = pw_of(self.blocks);
        let Some(row) = self.presence.get(y as usize * pw..(y as usize + 1) * pw) else {
            return;
        };
        let Some(&start) = self.offset.get(y as usize) else {
            return;
        };
        let mut slot = start as usize;
        for_each_bit(row, |b| {
            if let (Some(s), Some(d)) = (
                self.words.get(slot * BLOCK_WORDS..(slot + 1) * BLOCK_WORDS),
                dst.get_mut(b * BLOCK_WORDS..(b + 1) * BLOCK_WORDS),
            ) {
                mask_or_assign(d, s);
            }
            slot += 1;
        });
    }
}

/// What one wide closure or refresh did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WideStats {
    /// Most composition rounds any one row needed, including the last, which
    /// found nothing new.
    pub rounds: u32,
    /// Rows of `edge` (or of a reused closure) or'ed into a frontier: one per
    /// pair that entered a delta. For a full build this equals `|edge⁺|`.
    pub rows_composed: u64,
    /// Blocks stored in the result.
    pub blocks_allocated: u64,
    /// Blocks a dense layout would store: `n * blocks`.
    pub blocks_reserved: u64,
    /// Rows recomputed because a retired edge could have changed them. Zero
    /// for a refresh that only added edges.
    pub rows_recomputed: u64,
}

/// Why a wide closure or refresh refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WideRefusal {
    /// A row's delta did not empty within one round per position, which a
    /// correct semi-naive closure cannot do.
    FixpointBound,
    /// A position at or beyond the allowed range (`n`), or `n` itself beyond
    /// [`MAX_POSITIONS`] (then `n` carries that bound).
    PositionOutOfRange {
        /// The offending position.
        pos: u32,
        /// The exclusive bound it violated.
        n: u32,
    },
    /// The `src` and `dst` lanes differ in length.
    LengthMismatch,
}

/// Checks `n` against [`MAX_POSITIONS`]; returns the blocks per row.
fn check_dims(n: u32) -> Result<u32, WideRefusal> {
    if n > MAX_POSITIONS {
        return Err(WideRefusal::PositionOutOfRange {
            pos: n,
            n: MAX_POSITIONS,
        });
    }
    Ok(n.div_ceil(BLOCK_POSITIONS as u32))
}

/// Checks the lanes have equal length and every position is below `n`.
fn validate(n: u32, src: &[u32], dst: &[u32]) -> Result<(), WideRefusal> {
    if src.len() != dst.len() {
        return Err(WideRefusal::LengthMismatch);
    }
    for &p in src.iter().chain(dst) {
        if p >= n {
            return Err(WideRefusal::PositionOutOfRange { pos: p, n });
        }
    }
    Ok(())
}

/// Per-row working words, each `blocks * 4` long and reused across rows.
struct Scratch {
    full: Vec<u64>,
    delta: Vec<u64>,
    next: Vec<u64>,
}

impl Scratch {
    fn new(blocks: u32) -> Self {
        let w = blocks as usize * BLOCK_WORDS;
        Self {
            full: vec![0; w],
            delta: vec![0; w],
            next: vec![0; w],
        }
    }
}

/// Rows of an older closure that stay valid: position `y` is reusable when it
/// existed then and no retired edge can reach it, so its old row is final.
struct Reuse<'a> {
    old: &'a WideRows,
    affected: &'a [bool],
}

impl Reuse<'_> {
    fn closed(&self, y: u32) -> bool {
        y < self.old.n() && !self.affected.get(y as usize).copied().unwrap_or(true)
    }
}

/// Semi-naive fixpoint on one row, from `s.full` / `s.delta` as seeded:
/// `delta' = (delta ∘ edge) andnot full; full |= delta'`.
///
/// With `reuse`, a delta bit `y` whose old row is still valid is not
/// composed through `edge`: its whole old row is or'ed into `full` instead.
/// That is exact because everything an unaffected `y` reaches is itself
/// unaffected (an affected member would make `y` affected), so none of those
/// positions needs further expansion.
fn run_fixpoint(
    edge: &WideRows,
    n: u32,
    s: &mut Scratch,
    reuse: Option<&Reuse<'_>>,
    stats: &mut WideStats,
) -> Result<(), WideRefusal> {
    let Scratch { full, delta, next } = s;
    let mut rounds = 0u32;
    while mask_any(delta) {
        // A path visits each position once, so a delta is empty by round n+1.
        if rounds > n {
            return Err(WideRefusal::FixpointBound);
        }
        rounds += 1;
        next.fill(0);
        let mut composed = 0u64;
        for_each_bit(delta, |y| {
            composed += 1;
            let y = u32::try_from(y).unwrap_or(u32::MAX);
            match reuse {
                Some(r) if r.closed(y) => r.old.or_row_into(y, full),
                _ => edge.or_row_into(y, next),
            }
        });
        stats.rows_composed += composed;
        mask_andnot(next, full, delta);
        mask_or_assign(full, delta);
    }
    stats.rounds = stats.rounds.max(rounds);
    Ok(())
}

/// Packs `(src, dst)` lanes into edge rows. Duplicate edges set the same bit,
/// so they cannot change the result.
fn build_edges(n: u32, blocks: u32, src: &[u32], dst: &[u32]) -> Result<WideRows, WideRefusal> {
    validate(n, src, dst)?;
    let nu = n as usize;
    // Counting sort of destinations by source.
    let mut start = vec![0usize; nu + 1];
    for &s in src {
        if let Some(c) = start.get_mut(s as usize + 1) {
            *c += 1;
        }
    }
    for i in 0..nu {
        let prev = start.get(i).copied().unwrap_or(0);
        if let Some(c) = start.get_mut(i + 1) {
            *c += prev;
        }
    }
    let mut cursor = start.clone();
    let mut sorted = vec![0u32; src.len()];
    for (&s, &d) in src.iter().zip(dst) {
        if let Some(c) = cursor.get_mut(s as usize) {
            if let Some(slot) = sorted.get_mut(*c) {
                *slot = d;
            }
            *c += 1;
        }
    }
    let mut out = WideRows::builder(n, blocks);
    let mut dense = vec![0u64; blocks as usize * BLOCK_WORDS];
    for x in 0..nu {
        let (lo, hi) = (
            start.get(x).copied().unwrap_or(0),
            start.get(x + 1).copied().unwrap_or(0),
        );
        dense.fill(0);
        for &d in sorted.get(lo..hi).unwrap_or(&[]) {
            if let Some(w) = dense.get_mut(d as usize / 64) {
                *w |= 1 << (d % 64);
            }
        }
        out.push_dense_row(&dense);
    }
    Ok(out)
}

/// Fills the stats fields that describe the result.
fn finish(mut stats: WideStats, rows: &WideRows) -> WideStats {
    stats.blocks_allocated = rows.blocks_allocated();
    stats.blocks_reserved = u64::from(rows.n()) * u64::from(rows.blocks());
    stats
}

/// `edge⁺` over `n` positions from parallel `src` / `dst` lanes (wide
/// coordinates), semi-naively, one row at a time.
///
/// Rows are independent, so only one row of working words is live at once;
/// a block no row reaches is never allocated. Duplicate edges do not change
/// the result. `rows_composed` equals `|edge⁺|`.
pub fn closure_wide(
    n: u32,
    src: &[u32],
    dst: &[u32],
) -> Result<(WideRows, WideStats), WideRefusal> {
    let blocks = check_dims(n)?;
    let edge = build_edges(n, blocks, src, dst)?;
    let mut stats = WideStats::default();
    let mut s = Scratch::new(blocks);
    let mut out = WideRows::builder(n, blocks);
    for x in 0..n {
        edge.row_into(x, &mut s.full);
        s.delta.copy_from_slice(&s.full);
        run_fixpoint(&edge, n, &mut s, None, &mut stats)?;
        out.push_dense_row(&s.full);
    }
    let stats = finish(stats, &out);
    Ok((out, stats))
}

/// Append-only delta lanes. Op `1` is ADD, `0` is RETIRE; each append gets a
/// strictly larger version than the one before. Nothing is updated in place.
#[derive(Clone, Debug, Default)]
pub struct DeltaLog {
    src: Vec<u32>,
    dst: Vec<u32>,
    version: Vec<u64>,
    op: Vec<u8>,
}

impl DeltaLog {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    fn append(&mut self, src: u32, dst: u32, op: u8) {
        let v = self.version.last().map_or(1, |&v| v + 1);
        self.src.push(src);
        self.dst.push(dst);
        self.version.push(v);
        self.op.push(op);
    }

    /// Appends an ADD of edge `src -> dst`.
    pub fn add(&mut self, src: u32, dst: u32) {
        self.append(src, dst, 1);
    }

    /// Appends a RETIRE of edge `src -> dst`.
    pub fn retire(&mut self, src: u32, dst: u32) {
        self.append(src, dst, 0);
    }

    /// Number of logged operations.
    pub fn len(&self) -> usize {
        self.src.len()
    }

    /// Version of the newest operation, or `0` for an empty log.
    pub fn last_version(&self) -> u64 {
        self.version.last().copied().unwrap_or(0)
    }

    /// Whether nothing has been logged.
    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }
}

fn key(s: u32, d: u32) -> u64 {
    u64::from(s) << 32 | u64::from(d)
}

fn lanes(keys: impl Iterator<Item = u64>) -> (Vec<u32>, Vec<u32>) {
    keys.map(|k| ((k >> 32) as u32, (k & 0xffff_ffff) as u32))
        .unzip()
}

/// The edge multiset after applying the log in append (= version) order:
/// occurrences per `(src, dst)` in the base (duplicates kept), ADD = +1,
/// RETIRE = -1 floored at 0 (retiring an absent edge is a no-op). Entries
/// with count 0 are removed.
fn fold_counts(base_src: &[u32], base_dst: &[u32], log: &DeltaLog) -> BTreeMap<u64, u32> {
    let mut counts: BTreeMap<u64, u32> = BTreeMap::new();
    for (&s, &d) in base_src.iter().zip(base_dst) {
        *counts.entry(key(s, d)).or_insert(0) += 1;
    }
    for ((&s, &d), &op) in log.src.iter().zip(&log.dst).zip(&log.op) {
        let k = key(s, d);
        if op == 1 {
            *counts.entry(k).or_insert(0) += 1;
        } else if let Some(c) = counts.get_mut(&k) {
            *c -= 1;
            if *c == 0 {
                counts.remove(&k);
            }
        }
    }
    counts
}

/// The set of edges present (count > 0) in the folded multiset.
fn fold_keys(base_src: &[u32], base_dst: &[u32], log: &DeltaLog) -> BTreeSet<u64> {
    fold_counts(base_src, base_dst, log).into_keys().collect()
}

/// Folds `base` and the log into the effective edge multiset, as `src` /
/// `dst` lanes: ascending `(src, dst)`, each edge repeated once per
/// occurrence. A call graph is a bag, so duplicates are preserved (two call
/// sites stay two). The log is applied in append order: ADD is +1, RETIRE is
/// -1 floored at 0 (retiring an absent edge is a no-op). This is the explicit
/// re-fold: the caller decides when the resident base is replaced by its
/// result and the log emptied; nothing triggers it automatically. Lanes of
/// unequal length are zipped to the shorter.
pub fn refold(base_src: &[u32], base_dst: &[u32], log: &DeltaLog) -> (Vec<u32>, Vec<u32>) {
    let counts = fold_counts(base_src, base_dst, log);
    lanes(
        counts
            .into_iter()
            .flat_map(|(k, c)| std::iter::repeat_n(k, c as usize)),
    )
}

/// Copies `rows` into a builder of `n` rows (`n >= rows.n()`), padding with
/// empty rows; blocks per row are recomputed for `n`.
fn widen(rows: &WideRows, n: u32, blocks: u32) -> WideRows {
    let mut out = WideRows::builder(n, blocks);
    let mut dense = vec![0u64; blocks as usize * BLOCK_WORDS];
    for x in 0..n {
        rows.row_into(x, &mut dense);
        out.push_dense_row(&dense);
    }
    out
}

/// A mask of `positions` over `blocks * 4` words.
fn position_mask(blocks: u32, positions: impl Iterator<Item = u32>) -> Vec<u64> {
    let mut m = vec![0u64; blocks as usize * BLOCK_WORDS];
    for p in positions {
        if let Some(w) = m.get_mut(p as usize / 64) {
            *w |= 1 << (p % 64);
        }
    }
    m
}

/// Brings `old` (the closure of the base edges) up to date with `log` without
/// a full rebuild, for `n_new >= old.n()` positions.
///
/// The effective edge set is `base` folded with the log. Two phases, both
/// exact:
///
/// 1. Retired edges (in `base`, not effective). The affected rows are
///    `A = { x : x == a or a ∈ R⁺(x) }` for every retired source `a`; they
///    are recomputed against the edges that survive, reusing `R⁺(y)` for any
///    `y ∉ A`. Every other row is kept bit for bit. `rows_recomputed` is
///    `|A|`.
/// 2. Added edges (effective, not in `base`). Monotone, so each row that can
///    reach an added source is seeded with that source's new targets and run
///    through the same fixpoint; nothing is recomputed and `rows_recomputed`
///    stays zero.
///
/// An edge added and then retired within the log cancels and changes nothing.
/// Base edges must lie below `old.n()`; log edges below `n_new`.
pub fn refresh(
    old: &WideRows,
    base_src: &[u32],
    base_dst: &[u32],
    log: &DeltaLog,
    n_new: u32,
) -> Result<(WideRows, WideStats), WideRefusal> {
    if n_new < old.n() {
        // Positions never renumber, so a closure cannot shrink.
        return Err(WideRefusal::PositionOutOfRange {
            pos: old.n(),
            n: n_new,
        });
    }
    let blocks = check_dims(n_new)?;
    validate(old.n(), base_src, base_dst)?;
    validate(n_new, &log.src, &log.dst)?;

    let base: BTreeSet<u64> = base_src
        .iter()
        .zip(base_dst)
        .map(|(&s, &d)| key(s, d))
        .collect();
    // Set-level view of the multiset fold: an edge is present iff its count
    // is > 0. A count dropping 2 -> 1 retires nothing; only a drop to 0
    // retires, only a rise from 0 adds.
    let eff = fold_keys(base_src, base_dst, log);
    let retired: Vec<u64> = base.difference(&eff).copied().collect();
    let added: Vec<u64> = eff.difference(&base).copied().collect();

    let mut stats = WideStats::default();
    let mut s = Scratch::new(blocks);

    // Phase 1: retired edges.
    let phase1;
    let r1: &WideRows = if retired.is_empty() {
        if n_new == old.n() {
            old
        } else {
            phase1 = widen(old, n_new, blocks);
            &phase1
        }
    } else {
        let (rs, _) = lanes(retired.iter().copied());
        let retired_mask = position_mask(blocks, rs.iter().copied());
        let mut affected = vec![false; n_new as usize];
        for x in 0..n_new {
            let hit = if retired_mask
                .get(x as usize / 64)
                .is_some_and(|w| w >> (x % 64) & 1 == 1)
            {
                true
            } else {
                old.row_into(x, &mut s.next);
                s.next.iter().zip(&retired_mask).any(|(&r, &m)| r & m != 0)
            };
            if let Some(a) = affected.get_mut(x as usize) {
                *a = hit;
            }
        }
        let (e1s, e1d) = lanes(base.intersection(&eff).copied());
        let e1 = build_edges(n_new, blocks, &e1s, &e1d)?;
        let reuse = Reuse {
            old,
            affected: &affected,
        };
        let mut out = WideRows::builder(n_new, blocks);
        for x in 0..n_new {
            if affected.get(x as usize).copied().unwrap_or(false) {
                e1.row_into(x, &mut s.full);
                s.delta.copy_from_slice(&s.full);
                run_fixpoint(&e1, n_new, &mut s, Some(&reuse), &mut stats)?;
                stats.rows_recomputed += 1;
            } else {
                old.row_into(x, &mut s.full);
            }
            out.push_dense_row(&s.full);
        }
        phase1 = out;
        &phase1
    };

    // Phase 2: added edges.
    if added.is_empty() {
        let result = if std::ptr::eq(r1, old) {
            old.clone()
        } else {
            r1.clone()
        };
        let stats = finish(stats, &result);
        return Ok((result, stats));
    }
    let (as_, ad) = lanes(added.iter().copied());
    let added_edges = build_edges(n_new, blocks, &as_, &ad)?;
    let (es, ed) = lanes(eff.iter().copied());
    let eff_edges = build_edges(n_new, blocks, &es, &ed)?;
    let added_mask = position_mask(blocks, as_.iter().copied());
    let mut out = WideRows::builder(n_new, blocks);
    for x in 0..n_new {
        r1.row_into(x, &mut s.full);
        // Sources a path from x can start an added edge at: x and R⁺(x).
        for (t, (&f, &m)) in s.next.iter_mut().zip(s.full.iter().zip(&added_mask)) {
            *t = f & m;
        }
        if added_mask
            .get(x as usize / 64)
            .is_some_and(|w| w >> (x % 64) & 1 == 1)
        {
            if let Some(t) = s.next.get_mut(x as usize / 64) {
                *t |= 1 << (x % 64);
            }
        }
        if mask_any(&s.next) {
            let Scratch { full, delta, next } = &mut s;
            delta.fill(0);
            for_each_bit(next, |src| {
                added_edges.or_row_into(u32::try_from(src).unwrap_or(u32::MAX), delta);
            });
            mask_andnot(delta, full, next);
            delta.copy_from_slice(next);
            mask_or_assign(full, delta);
            run_fixpoint(&eff_edges, n_new, &mut s, None, &mut stats)?;
        }
        out.push_dense_row(&s.full);
    }
    let stats = finish(stats, &out);
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The edge log is a multiset: duplicates in the base survive, retire
    /// removes one occurrence, over-retire floors at zero, add-then-retire
    /// cancels.
    #[test]
    fn refold_is_a_multiset() {
        let (bs, bd) = (vec![1, 1], vec![2, 2]);
        let mut log = DeltaLog::new();
        assert_eq!(refold(&bs, &bd, &log), (vec![1, 1], vec![2, 2]));
        log.retire(1, 2);
        assert_eq!(refold(&bs, &bd, &log), (vec![1], vec![2]));
        log.retire(1, 2);
        assert_eq!(refold(&bs, &bd, &log), (Vec::new(), Vec::new()));
        log.retire(1, 2); // absent: no-op, floored at 0
        assert_eq!(refold(&bs, &bd, &log), (Vec::new(), Vec::new()));
        let mut log = DeltaLog::new();
        log.add(5, 6);
        log.retire(5, 6);
        assert_eq!(refold(&bs, &bd, &log), (vec![1, 1], vec![2, 2]));
    }

    /// Naive reachability by DFS from every node, written independently of
    /// the kernel.
    fn naive(n: usize, edges: &[(u32, u32)]) -> Vec<Vec<bool>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a as usize].push(b as usize);
        }
        let mut out = vec![vec![false; n]; n];
        for x in 0..n {
            let mut stack: Vec<usize> = adj[x].clone();
            while let Some(y) = stack.pop() {
                if !out[x][y] {
                    out[x][y] = true;
                    stack.extend(adj[y].iter().copied());
                }
            }
        }
        out
    }

    fn build(n: u32, edges: &[(u32, u32)]) -> (WideRows, WideStats) {
        let s: Vec<u32> = edges.iter().map(|e| e.0).collect();
        let d: Vec<u32> = edges.iter().map(|e| e.1).collect();
        closure_wide(n, &s, &d).expect("closure")
    }

    fn assert_matches(rows: &WideRows, n: usize, edges: &[(u32, u32)]) -> u64 {
        let reference = naive(n, edges);
        let mut pairs = 0u64;
        assert_eq!(reference.len(), n);
        for (x, row) in reference.iter().enumerate() {
            for (y, &want) in row.iter().enumerate() {
                assert_eq!(rows.contains(x as u32, y as u32), want, "({x},{y})");
                pairs += u64::from(want);
            }
        }
        pairs
    }

    #[test]
    fn chain_of_five() {
        let edges: Vec<(u32, u32)> = (0..4).map(|i| (i, i + 1)).collect();
        let (rows, stats) = build(5, &edges);
        let pairs = assert_matches(&rows, 5, &edges);
        assert_eq!(pairs, 10);
        assert_eq!(stats.rows_composed, pairs);
        assert_eq!(stats.blocks_reserved, 5);
    }

    #[test]
    fn cycle_reaches_itself() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1)];
        let (rows, stats) = build(4, &edges);
        let pairs = assert_matches(&rows, 4, &edges);
        assert_eq!(pairs, 16);
        assert_eq!(stats.rows_composed, 16);
        assert!(rows.contains(2, 2));
    }

    #[test]
    fn chain_crossing_a_block_boundary() {
        let edges: Vec<(u32, u32)> = (0..299).map(|i| (i, i + 1)).collect();
        let (rows, stats) = build(300, &edges);
        let pairs = assert_matches(&rows, 300, &edges);
        assert_eq!(pairs, 300 * 299 / 2);
        assert_eq!(stats.rows_composed, pairs);
        // Row 0 reaches 1..=299: both blocks. Row 290 reaches only block 1.
        let mut dense = vec![0u64; 5];
        rows.row_into(290, &mut dense);
        assert_eq!(dense.iter().map(|w| w.count_ones()).sum::<u32>(), 9);
        assert!(stats.blocks_allocated < stats.blocks_reserved);
    }
}
