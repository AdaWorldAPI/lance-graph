//! Identity table and thought sessions: one trie per thought (S16, S8) vs one
//! identity trie per class with thought overlays hanging off the identity row
//! (M16, M8), plus session materialization.
//!
//! Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
//! ("Identity table and thought sessions").
//!
//! - Identity: a dense table of 64k rows x 32 lanes of 16 bytes (lane 0 key,
//!   1 edge, 2..31 value), the row being the NodeGuid's rank in its class trie.
//! - A thought writes value lanes only (lanes 2..31, indexed 0..29 here).
//! - All arms return the same merged delta (30 value lanes plus the mask of
//!   lanes written); a higher thought index wins a shared lane.
//! - S16 / S8: one trie per thought (16 levels with the classid, 8 without).
//! - M16 / M8: one identity trie per class; per row a 64-bit written-by
//!   thought mask, then entries (thought, 30-bit lane mask, packed cells)
//!   sorted by thought. One descent per read.
//! - Sessions: a session (one thought, or all 64 merged) materializes its view
//!   of a prefix range into a temporary buffer of 512-byte rows.
//!
//! No crate types; builds with plain `rustc -O`.
//! Run: `cargo run --release --example identity_session_probe` (tier printed).
//! Tests: `cargo test --example identity_session_probe`.

use std::hint::black_box;
use std::time::Instant;

const LANES: usize = 30;
const ROW_LANES: usize = 32;
const CLASSID: u64 = 0x0C0C_0000;
type Cell = [u8; 16];
type Lanes = [Cell; LANES];
type Row = [Cell; ROW_LANES];

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn cell(a: u64) -> Cell {
    let b = mix(a);
    let mut c = [0u8; 16];
    c[..8].copy_from_slice(&a.to_le_bytes());
    c[8..].copy_from_slice(&b.to_le_bytes());
    c
}

/// The value thought `t` wrote into value lane `lane` of `key`.
fn val(t: usize, key: u64, lane: usize) -> Cell {
    cell(mix(key ^ ((t as u64) << 48) ^ ((lane as u64) << 40)))
}

struct BitIter(u64);
impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(t)
    }
}

// ---------- trie ----------

/// Path of `key` in a trie of `depth` levels: 16 keeps the classid, 8 drops
/// it (one trie per class).
fn path(key: u64, depth: usize) -> u64 {
    if depth == 16 {
        key
    } else {
        key << 32
    }
}

fn nib(p: u64, level: usize) -> u32 {
    ((p >> (60 - 4 * level)) & 0xF) as u32
}

struct Trie {
    depth: usize,
    masks: Vec<Vec<u16>>,
    first: Vec<Vec<u32>>,
    leaves: usize,
}

impl Trie {
    /// `keys` sorted and unique; the leaf index is the key's rank.
    fn build(keys: &[u64], depth: usize) -> Self {
        let paths: Vec<u64> = keys.iter().map(|&k| path(k, depth)).collect();
        let mut t = Trie {
            depth,
            masks: vec![Vec::new(); depth],
            first: vec![Vec::new(); depth],
            leaves: keys.len(),
        };
        if keys.is_empty() {
            return t;
        }
        let mut groups = vec![(0usize, paths.len())];
        for level in 0..depth {
            let mut next = Vec::with_capacity(groups.len() * 2);
            for &(s, e) in &groups {
                t.first[level].push(next.len() as u32);
                let mut mask = 0u16;
                let mut i = s;
                while i < e {
                    let n = nib(paths[i], level);
                    let mut j = i + 1;
                    while j < e && nib(paths[j], level) == n {
                        j += 1;
                    }
                    mask |= 1 << n;
                    next.push((i, j));
                    i = j;
                }
                t.masks[level].push(mask);
            }
            groups = next;
        }
        t
    }
    fn is_empty(&self) -> bool {
        self.leaves == 0
    }
    fn bytes(&self) -> usize {
        self.masks.iter().map(|l| l.len() * 2).sum::<usize>()
            + self.first.iter().map(|l| l.len() * 4).sum::<usize>()
    }
    #[inline]
    fn mask(&self, level: usize, node: u32) -> u16 {
        self.masks[level][node as usize]
    }
    #[inline]
    fn child(&self, level: usize, node: u32, n: u32) -> Option<u32> {
        let m = self.mask(level, node);
        if (m >> n) & 1 == 0 {
            None
        } else {
            Some(self.first[level][node as usize] + (m & ((1u16 << n) - 1)).count_ones())
        }
    }
    /// Rank of `key`, counting descent steps.
    fn lookup(&self, key: u64, steps: &mut u64) -> Option<u32> {
        if self.is_empty() {
            return None;
        }
        let p = path(key, self.depth);
        let mut node = 0;
        for level in 0..self.depth {
            *steps += 1;
            node = self.child(level, node, nib(p, level))?;
        }
        Some(node)
    }
    /// Rank range [lo, hi) of the keys under the first `plen` nibbles of the
    /// path of `prefix`: descend, then follow the lowest and highest child.
    fn range(&self, prefix: u64, plen: usize) -> (u32, u32) {
        if self.is_empty() {
            return (0, 0);
        }
        let p = path(prefix, self.depth);
        let mut node = 0;
        for level in 0..plen {
            match self.child(level, node, nib(p, level)) {
                Some(c) => node = c,
                None => return (0, 0),
            }
        }
        let (mut lo, mut hi) = (node, node);
        for level in plen..self.depth {
            let ml = self.mask(level, lo);
            lo = self.child(level, lo, ml.trailing_zeros()).unwrap();
            let mh = self.mask(level, hi);
            hi = self.child(level, hi, 15 - mh.leading_zeros()).unwrap();
        }
        (lo, hi + 1)
    }
}

// ---------- data ----------

type Coverage = Vec<Vec<(u64, u32)>>;

fn universe(n: usize) -> Vec<u64> {
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut i = 0u64;
    while seen.len() < n {
        seen.insert((CLASSID << 32) | (mix(i) & 0xFFFF_FFFF));
        i += 1;
    }
    let mut u: Vec<u64> = seen.into_iter().collect();
    u.sort_unstable();
    u
}

fn coverage(univ: &[u64], thoughts: usize, d: f64, w: usize) -> Coverage {
    let cut = (d * u64::MAX as f64) as u64;
    (0..thoughts)
        .map(|t| {
            univ.iter()
                .filter(|&&k| d >= 1.0 || mix(k ^ ((t as u64) << 56) ^ 0xC0FE) < cut)
                .map(|&k| {
                    let mut m = 0u32;
                    let mut s = mix(k ^ ((t as u64) << 52) ^ 0xBEEF);
                    while (m.count_ones() as usize) < w {
                        m |= 1 << (s % LANES as u64);
                        s = mix(s);
                    }
                    (k, m)
                })
                .collect()
        })
        .collect()
}

/// The identity table: 64k rows x 32 lanes, row = rank in the class trie.
struct Identity {
    trie: Trie,
    rows: Vec<Row>,
}

fn identity(univ: &[u64], depth: usize) -> Identity {
    let rows = univ
        .iter()
        .map(|&k| {
            let mut r = [[0u8; 16]; ROW_LANES];
            r[0][..8].copy_from_slice(&k.to_le_bytes());
            r[1] = cell(mix(k ^ 0xED6E));
            for (lane, c) in r[2..].iter_mut().enumerate() {
                *c = cell(mix(k ^ 0xBA5E ^ ((lane as u64) << 40)));
            }
            r
        })
        .collect();
    Identity {
        trie: Trie::build(univ, depth),
        rows,
    }
}

// ---------- arm S: one trie per thought ----------

struct ThoughtTable {
    trie: Trie,
    lanes: Vec<u32>,
    starts: Vec<u32>,
    payload: Vec<Cell>,
}

fn build_s(cov: &Coverage, depth: usize) -> Vec<ThoughtTable> {
    cov.iter()
        .enumerate()
        .map(|(t, c)| {
            let keys: Vec<u64> = c.iter().map(|e| e.0).collect();
            let mut starts = Vec::with_capacity(c.len());
            let mut payload = Vec::new();
            for &(k, m) in c {
                starts.push(payload.len() as u32);
                payload.extend(BitIter(m as u64).map(|lane| val(t, k, lane)));
            }
            ThoughtTable {
                trie: Trie::build(&keys, depth),
                lanes: c.iter().map(|e| e.1).collect(),
                starts,
                payload,
            }
        })
        .collect()
}

fn bytes_s(s: &[ThoughtTable]) -> usize {
    s.iter()
        .map(|t| t.trie.bytes() + t.lanes.len() * 8 + t.payload.len() * 16)
        .sum()
}

impl ThoughtTable {
    #[inline]
    fn emit(&self, leaf: u32, sel: u32, out: &mut Lanes) -> u32 {
        let have = self.lanes[leaf as usize];
        let start = self.starts[leaf as usize] as usize;
        let m = have & sel;
        for lane in BitIter(m as u64) {
            out[lane] = self.payload[start + (have & ((1u32 << lane) - 1)).count_ones() as usize];
        }
        m
    }
}

fn select_s(s: &[ThoughtTable], thoughts: u64) -> Vec<&ThoughtTable> {
    s.iter()
        .enumerate()
        .filter(|(t, tt)| thoughts >> t & 1 == 1 && !tt.trie.is_empty())
        .map(|(_, tt)| tt)
        .collect()
}

fn point_s(
    sel: &[&ThoughtTable],
    lanes: u32,
    key: u64,
    scratch: &mut Vec<(u32, u32)>,
    steps: &mut u64,
) -> (Lanes, u32) {
    let mut out = [[0u8; 16]; LANES];
    let mut m = 0;
    let Some(first) = sel.first() else {
        return (out, 0);
    };
    let depth = first.trie.depth;
    let p = path(key, depth);
    scratch.clear();
    scratch.extend((0..sel.len() as u32).map(|i| (i, 0)));
    for level in 0..depth {
        let n = nib(p, level);
        let mut k = 0;
        for j in 0..scratch.len() {
            let (i, node) = scratch[j];
            *steps += 1;
            if let Some(c) = sel[i as usize].trie.child(level, node, n) {
                scratch[k] = (i, c);
                k += 1;
            }
        }
        scratch.truncate(k);
        if k == 0 {
            break;
        }
    }
    for &(i, leaf) in scratch.iter() {
        m |= sel[i as usize].emit(leaf, lanes, &mut out);
    }
    (out, m)
}

#[allow(clippy::too_many_arguments)]
fn dfs_s<F: FnMut(u64, &Lanes, u32)>(
    sel: &[&ThoughtTable],
    lanes: u32,
    depth: usize,
    level: usize,
    p: u64,
    alive: &[(u32, u32)],
    bufs: &mut [Vec<(u32, u32)>],
    f: &mut F,
) {
    if level == depth {
        let mut out = [[0u8; 16]; LANES];
        let mut m = 0;
        for &(i, leaf) in alive {
            m |= sel[i as usize].emit(leaf, lanes, &mut out);
        }
        if m != 0 {
            f(p, &out, m);
        }
        return;
    }
    let mut union = 0u16;
    for &(i, node) in alive {
        union |= sel[i as usize].trie.mask(level, node);
    }
    let (cur, rest) = bufs.split_first_mut().unwrap();
    for n in BitIter(union as u64) {
        cur.clear();
        for &(i, node) in alive {
            if let Some(c) = sel[i as usize].trie.child(level, node, n as u32) {
                cur.push((i, c));
            }
        }
        let child = p | ((n as u64) << (60 - 4 * level));
        dfs_s(sel, lanes, depth, level + 1, child, cur, rest, f);
    }
}

/// Range over the keys under `prefix` (classid + top HEEL nibble); `f` gets
/// the trie path of each emitted key.
fn range_s<F: FnMut(u64, &Lanes, u32)>(
    sel: &[&ThoughtTable],
    lanes: u32,
    prefix: u64,
    bufs: &mut [Vec<(u32, u32)>],
    f: &mut F,
) {
    let Some(first) = sel.first() else { return };
    let depth = first.trie.depth;
    let plen = depth - 7;
    let p = path(prefix, depth);
    let (head, rest) = bufs.split_first_mut().unwrap();
    head.clear();
    head.extend((0..sel.len() as u32).map(|i| (i, 0)));
    for level in 0..plen {
        let n = nib(p, level);
        head.retain_mut(|e| match sel[e.0 as usize].trie.child(level, e.1, n) {
            Some(c) => {
                e.1 = c;
                true
            }
            None => false,
        });
    }
    if !head.is_empty() {
        let base = p & !((1u64 << (64 - 4 * plen)) - 1);
        dfs_s(sel, lanes, depth, plen, base, head, rest, f);
    }
}

// ---------- arm M: identity trie + overlays per row ----------

struct Overlays {
    written: Vec<u64>,
    row_start: Vec<u32>,
    thought: Vec<u8>,
    lanes: Vec<u32>,
    start: Vec<u32>,
    payload: Vec<Cell>,
}

fn build_m(univ: &[u64], cov: &Coverage) -> Overlays {
    let mut per_row: Vec<Vec<(u8, u32)>> = vec![Vec::new(); univ.len()];
    for (t, c) in cov.iter().enumerate() {
        for &(k, m) in c {
            let r = univ.binary_search(&k).unwrap();
            per_row[r].push((t as u8, m));
        }
    }
    let mut o = Overlays {
        written: Vec::with_capacity(univ.len()),
        row_start: Vec::with_capacity(univ.len() + 1),
        thought: Vec::new(),
        lanes: Vec::new(),
        start: Vec::new(),
        payload: Vec::new(),
    };
    for (r, es) in per_row.iter().enumerate() {
        o.row_start.push(o.thought.len() as u32);
        let mut w = 0u64;
        for &(t, m) in es {
            w |= 1 << t;
            o.thought.push(t);
            o.lanes.push(m);
            o.start.push(o.payload.len() as u32);
            o.payload
                .extend(BitIter(m as u64).map(|lane| val(t as usize, univ[r], lane)));
        }
        o.written.push(w);
    }
    o.row_start.push(o.thought.len() as u32);
    o
}

fn bytes_m(o: &Overlays) -> usize {
    o.written.len() * 8 + o.row_start.len() * 4 + o.thought.len() * 9 + o.payload.len() * 16
}

impl Overlays {
    /// Merged delta of `row` for the selected thoughts and lanes.
    #[inline]
    fn merge(&self, row: usize, thoughts: u64, lanes: u32, out: &mut Lanes) -> u32 {
        if self.written[row] & thoughts == 0 {
            return 0;
        }
        let mut m = 0;
        for e in self.row_start[row] as usize..self.row_start[row + 1] as usize {
            if thoughts >> self.thought[e] & 1 == 0 {
                continue;
            }
            let have = self.lanes[e];
            let s = self.start[e] as usize;
            let mm = have & lanes;
            for lane in BitIter(mm as u64) {
                out[lane] = self.payload[s + (have & ((1u32 << lane) - 1)).count_ones() as usize];
            }
            m |= mm;
        }
        m
    }
}

fn point_m(
    id: &Identity,
    o: &Overlays,
    thoughts: u64,
    lanes: u32,
    key: u64,
    steps: &mut u64,
) -> (Lanes, u32) {
    let mut out = [[0u8; 16]; LANES];
    let m = match id.trie.lookup(key, steps) {
        Some(r) => o.merge(r as usize, thoughts, lanes, &mut out),
        None => 0,
    };
    (out, m)
}

fn range_m<F: FnMut(u64, &Lanes, u32)>(
    id: &Identity,
    o: &Overlays,
    thoughts: u64,
    lanes: u32,
    prefix: u64,
    f: &mut F,
) {
    let (lo, hi) = id.trie.range(prefix, id.trie.depth - 7);
    for r in lo as usize..hi as usize {
        if o.written[r] & thoughts == 0 {
            continue;
        }
        let mut out = [[0u8; 16]; LANES];
        let m = o.merge(r, thoughts, lanes, &mut out);
        if m != 0 {
            f(
                path(
                    u64::from_le_bytes(id.rows[r][0][..8].try_into().unwrap()),
                    id.trie.depth,
                ),
                &out,
                m,
            );
        }
    }
}

/// Session materialization: the session's view of a prefix range as
/// 512-byte rows (identity row, then the session's overlays).
fn materialize(id: &Identity, o: &Overlays, thoughts: u64, prefix: u64, buf: &mut Vec<Row>) {
    let (lo, hi) = id.trie.range(prefix, id.trie.depth - 7);
    materialize_rows(id, o, thoughts, lo as usize, hi as usize, buf);
}

fn materialize_rows(
    id: &Identity,
    o: &Overlays,
    thoughts: u64,
    lo: usize,
    hi: usize,
    buf: &mut Vec<Row>,
) {
    buf.clear();
    for r in lo..hi {
        let mut row = id.rows[r];
        if o.written[r] & thoughts != 0 {
            let mut out = [[0u8; 16]; LANES];
            let m = o.merge(r, thoughts, (1 << LANES) - 1, &mut out);
            for lane in BitIter(m as u64) {
                row[2 + lane] = out[lane];
            }
        }
        buf.push(row);
    }
}

// ---------- arm D: dense tile, the Morton code is the row ----------

/// One classid, all 65,536 HEEL codes (a full 256 x 256 Morton tile), HIP 0.
fn dense_universe() -> Vec<u64> {
    (0..65_536u64)
        .map(|c| (CLASSID << 32) | (c << 16))
        .collect()
}

/// Row of `key` in the dense tile: its HEEL code, after the classid check.
#[inline]
fn row_d(key: u64) -> Option<usize> {
    if key >> 32 != CLASSID || key & 0xFFFF != 0 {
        None
    } else {
        Some(((key >> 16) & 0xFFFF) as usize)
    }
}

/// Rows under a prefix (top HEEL nibble p): a Z-order prefix is a
/// contiguous code range.
fn rows_d(prefix: u64) -> (usize, usize) {
    let p = ((prefix >> 28) & 0xF) as usize;
    (p << 12, (p + 1) << 12)
}

fn point_d(o: &Overlays, thoughts: u64, lanes: u32, key: u64) -> (Lanes, u32) {
    let mut out = [[0u8; 16]; LANES];
    let m = match row_d(key) {
        Some(r) => o.merge(r, thoughts, lanes, &mut out),
        None => 0,
    };
    (out, m)
}

fn range_d<F: FnMut(u64, &Lanes, u32)>(
    univ: &[u64],
    o: &Overlays,
    thoughts: u64,
    lanes: u32,
    prefix: u64,
    f: &mut F,
) {
    let (lo, hi) = rows_d(prefix);
    for (r, &key) in univ.iter().enumerate().take(hi).skip(lo) {
        if o.written[r] & thoughts == 0 {
            continue;
        }
        let mut out = [[0u8; 16]; LANES];
        let m = o.merge(r, thoughts, lanes, &mut out);
        if m != 0 {
            f(path(key, 8), &out, m);
        }
    }
}

fn prefix_of(p: u64) -> u64 {
    (CLASSID << 32) | (p << 28)
}

// ---------- timing ----------

fn ns_per<F: FnMut()>(elements: u64, mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / elements.max(1) as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

struct Res {
    q2: f64,
    step: f64,
    q3: f64,
    emitted: u64,
}

fn run_s(sel: &[&ThoughtTable], lanes: u32, probes: &[u64]) -> Res {
    let mut scratch = Vec::with_capacity(sel.len());
    let mut steps = 0u64;
    for &k in probes {
        black_box(point_s(sel, lanes, k, &mut scratch, &mut steps));
    }
    let mut d = 0;
    let q2 = ns_per(probes.len() as u64, || {
        for &k in probes {
            black_box(point_s(sel, lanes, k, &mut scratch, &mut d));
        }
    });
    let mut bufs: Vec<Vec<(u32, u32)>> = (0..=16).map(|_| Vec::with_capacity(sel.len())).collect();
    let mut emitted = 0;
    for p in 0..16 {
        range_s(sel, lanes, prefix_of(p), &mut bufs, &mut |_, _, _| {
            emitted += 1
        });
    }
    let q3 = ns_per(emitted, || {
        let mut acc = 0u64;
        for p in 0..16 {
            range_s(sel, lanes, prefix_of(p), &mut bufs, &mut |k, o, m| {
                acc ^= k ^ m as u64 ^ o[0][0] as u64
            });
        }
        black_box(acc);
    });
    Res {
        q2,
        step: q2 * probes.len() as f64 / steps.max(1) as f64,
        q3,
        emitted,
    }
}

fn run_m(id: &Identity, o: &Overlays, thoughts: u64, lanes: u32, probes: &[u64]) -> Res {
    let mut steps = 0u64;
    for &k in probes {
        black_box(point_m(id, o, thoughts, lanes, k, &mut steps));
    }
    let mut d = 0;
    let q2 = ns_per(probes.len() as u64, || {
        for &k in probes {
            black_box(point_m(id, o, thoughts, lanes, k, &mut d));
        }
    });
    let mut emitted = 0;
    for p in 0..16 {
        range_m(id, o, thoughts, lanes, prefix_of(p), &mut |_, _, _| {
            emitted += 1
        });
    }
    let q3 = ns_per(emitted, || {
        let mut acc = 0u64;
        for p in 0..16 {
            range_m(id, o, thoughts, lanes, prefix_of(p), &mut |k, o, m| {
                acc ^= k ^ m as u64 ^ o[0][0] as u64
            });
        }
        black_box(acc);
    });
    Res {
        q2,
        step: q2 * probes.len() as f64 / steps.max(1) as f64,
        q3,
        emitted,
    }
}

fn main() {
    if std::env::args().any(|a| a == "dense") {
        main_dense();
        return;
    }
    const N: usize = 65_536;
    println!("tier: {}", tier());
    let univ = universe(N);
    let id16 = identity(&univ, 16);
    let id8 = identity(&univ, 8);
    println!(
        "identity: {} rows x {} lanes = {} bytes (shared by every arm); trie 16 levels {} B, 8 levels {} B",
        N,
        ROW_LANES,
        N * ROW_LANES * 16,
        id16.trie.bytes(),
        id8.trie.bytes()
    );
    let mut r = 12345u64;
    let probes: Vec<u64> = (0..N)
        .map(|_| {
            r = mix(r);
            univ[(r % N as u64) as usize]
        })
        .collect();
    println!(
        "\n{:>3} {:>5} {:>3} {:>4} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} {:>7} | {:>7}",
        "T", "d", "w", "sel", "Q1 S16", "Q1 S8", "Q1 M", "Q2 S16", "Q2 S8", "Q2 M16", "Q2 M8",
        "st S16", "st S8", "st M16", "st M8", "Q3 S16", "Q3 S8", "Q3 M16", "Q3 M8", "emitted"
    );
    let mut ref_overlays = None;
    for t in [1usize, 8, 64] {
        for (dl, d) in [("1%", 0.01), ("10%", 0.10), ("100%", 1.0)] {
            for w in [1usize, 4, 30] {
                let cells = (t as f64 * N as f64 * d * w as f64) as u64;
                if cells > 32_000_000 {
                    println!("{t:>3} {dl:>5} {w:>3}   skipped ({cells} lane-cells > 32 M)");
                    continue;
                }
                let cov = coverage(&univ, t, d, w);
                let s16 = build_s(&cov, 16);
                let s8 = build_s(&cov, 8);
                let o = build_m(&univ, &cov);
                let thoughts = if t == 64 { u64::MAX } else { (1u64 << t) - 1 };
                for (sl, lanes) in [("1", 1u32), ("all", (1u32 << LANES) - 1)] {
                    let a = run_s(&select_s(&s16, thoughts), lanes, &probes);
                    let b = run_s(&select_s(&s8, thoughts), lanes, &probes);
                    let c = run_m(&id16, &o, thoughts, lanes, &probes);
                    let e = run_m(&id8, &o, thoughts, lanes, &probes);
                    assert_eq!(a.emitted, b.emitted);
                    assert_eq!(a.emitted, c.emitted);
                    assert_eq!(a.emitted, e.emitted);
                    println!(
                        "{t:>3} {dl:>5} {w:>3} {sl:>4} | {:>10} {:>10} {:>10} | {:>8.1} {:>8.1} {:>8.1} {:>8.1} | {:>6.2} {:>6.2} {:>6.2} {:>6.2} | {:>7.1} {:>7.1} {:>7.1} {:>7.1} | {:>7}",
                        bytes_s(&s16), bytes_s(&s8), bytes_m(&o),
                        a.q2, b.q2, c.q2, e.q2, a.step, b.step, c.step, e.step,
                        a.q3, b.q3, c.q3, e.q3, a.emitted
                    );
                }
                if t == 64 && dl == "10%" && w == 4 {
                    ref_overlays = Some(o);
                }
            }
        }
    }

    // Sessions, at the reference cell, M8.
    let o = ref_overlays.expect("reference cell ran");
    let mut buf = Vec::with_capacity(N / 8);
    let mut rows = 0u64;
    for p in 0..16 {
        materialize(&id8, &o, 1, prefix_of(p), &mut buf);
        rows += buf.len() as u64;
    }
    let one = ns_per(rows * 64, || {
        for s in 0..64 {
            for p in 0..16 {
                materialize(&id8, &o, 1u64 << s, prefix_of(p), &mut buf);
                black_box(&buf);
            }
        }
    });
    let all = ns_per(rows, || {
        for p in 0..16 {
            materialize(&id8, &o, u64::MAX, prefix_of(p), &mut buf);
            black_box(&buf);
        }
    });
    println!(
        "\nsessions (reference cell, M8): one-thought session {:.2} ns/row, all-64 session {:.2} ns/row; \
         64 sessions over the whole class = {} bytes if all materialized at once",
        one,
        all,
        64 * N * ROW_LANES * 16
    );
    println!("\nQ1 bytes (identity table excluded); Q2 ns per point read; st = ns per descent step; Q3 ns per emitted NodeGuid.");
}

/// The dense-tile run: M8 (identity trie) vs D (Morton code is the row).
fn main_dense() {
    const N: usize = 65_536;
    println!(
        "tier: {}  (dense tile: one classid, all 65,536 HEEL codes)",
        tier()
    );
    let univ = dense_universe();
    let id8 = identity(&univ, 8);
    let mut r = 12345u64;
    let probes: Vec<u64> = (0..N)
        .map(|_| {
            r = mix(r);
            univ[(r % N as u64) as usize]
        })
        .collect();
    println!(
        "\n{:>3} {:>5} {:>3} {:>4} | {:>8} {:>8} | {:>7} {:>7} | {:>7}",
        "T", "d", "w", "sel", "Q2 M8", "Q2 D", "Q3 M8", "Q3 D", "emitted"
    );
    let mut ref_overlays = None;
    for t in [1usize, 8, 64] {
        for (dl, d) in [("1%", 0.01), ("10%", 0.10), ("100%", 1.0)] {
            for w in [1usize, 4, 30] {
                let cells = (t as f64 * N as f64 * d * w as f64) as u64;
                if cells > 32_000_000 {
                    println!("{t:>3} {dl:>5} {w:>3}   skipped ({cells} lane-cells > 32 M)");
                    continue;
                }
                let cov = coverage(&univ, t, d, w);
                let o = build_m(&univ, &cov);
                let thoughts = if t == 64 { u64::MAX } else { (1u64 << t) - 1 };
                for (sl, lanes) in [("1", 1u32), ("all", (1u32 << LANES) - 1)] {
                    let m8 = run_m(&id8, &o, thoughts, lanes, &probes);
                    let q2d = ns_per(probes.len() as u64, || {
                        for &k in &probes {
                            black_box(point_d(&o, thoughts, lanes, k));
                        }
                    });
                    let mut emitted = 0u64;
                    for p in 0..16 {
                        range_d(&univ, &o, thoughts, lanes, prefix_of(p), &mut |_, _, _| {
                            emitted += 1
                        });
                    }
                    assert_eq!(emitted, m8.emitted);
                    let q3d = ns_per(emitted, || {
                        let mut acc = 0u64;
                        for p in 0..16 {
                            range_d(&univ, &o, thoughts, lanes, prefix_of(p), &mut |k, o, m| {
                                acc ^= k ^ m as u64 ^ o[0][0] as u64
                            });
                        }
                        black_box(acc);
                    });
                    println!(
                        "{t:>3} {dl:>5} {w:>3} {sl:>4} | {:>8.1} {:>8.1} | {:>7.1} {:>7.1} | {:>7}",
                        m8.q2, q2d, m8.q3, q3d, emitted
                    );
                }
                if t == 64 && dl == "10%" && w == 4 {
                    ref_overlays = Some(o);
                }
            }
        }
    }
    let o = ref_overlays.expect("reference cell ran");
    let mut buf = Vec::with_capacity(N / 8);
    let rows = N as u64;
    let m8 = ns_per(rows, || {
        for p in 0..16 {
            materialize(&id8, &o, u64::MAX, prefix_of(p), &mut buf);
            black_box(&buf);
        }
    });
    let d1 = ns_per(rows * 64, || {
        for s in 0..64 {
            for p in 0..16 {
                let (lo, hi) = rows_d(prefix_of(p));
                materialize_rows(&id8, &o, 1u64 << s, lo, hi, &mut buf);
                black_box(&buf);
            }
        }
    });
    let dall = ns_per(rows, || {
        for p in 0..16 {
            let (lo, hi) = rows_d(prefix_of(p));
            materialize_rows(&id8, &o, u64::MAX, lo, hi, &mut buf);
            black_box(&buf);
        }
    });
    println!(
        "\nsessions (reference cell): M8 all-64 {m8:.2} ns/row; D one-thought {d1:.2} ns/row; D all-64 {dall:.2} ns/row"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn oracle(cov: &Coverage, thoughts: u64, lanes: u32) -> BTreeMap<u64, (Lanes, u32)> {
        let mut out: BTreeMap<u64, (Lanes, u32)> = BTreeMap::new();
        for (t, c) in cov.iter().enumerate() {
            if thoughts >> t & 1 == 0 {
                continue;
            }
            for &(k, m) in c {
                let m = m & lanes;
                if m == 0 {
                    continue;
                }
                let e = out.entry(k).or_insert(([[0u8; 16]; LANES], 0));
                for lane in BitIter(m as u64) {
                    e.0[lane] = val(t, k, lane);
                }
                e.1 |= m;
            }
        }
        out
    }

    fn same(got: (Lanes, u32), want: Option<&(Lanes, u32)>) {
        match want {
            Some(&(wo, wm)) => {
                assert_eq!(got.1, wm);
                for lane in BitIter(wm as u64) {
                    assert_eq!(got.0[lane], wo[lane], "lane {lane}");
                }
            }
            None => assert_eq!(got.1, 0),
        }
    }

    fn fixture() -> (Vec<u64>, Coverage) {
        let univ = universe(3000);
        let cov = coverage(&univ, 7, 0.3, 4);
        (univ, cov)
    }

    #[test]
    fn every_arm_matches_the_oracle() {
        let (univ, cov) = fixture();
        let o = build_m(&univ, &cov);
        for depth in [16, 8] {
            let s = build_s(&cov, depth);
            let id = identity(&univ, depth);
            for thoughts in [0b111_1111u64, 0b101_0010] {
                for lanes in [1u32, 0b1011_0000_0001, (1 << LANES) - 1] {
                    let want = oracle(&cov, thoughts, lanes);
                    let sel = select_s(&s, thoughts);
                    let mut scratch = Vec::new();
                    let mut st = 0;
                    for &k in &univ {
                        same(point_s(&sel, lanes, k, &mut scratch, &mut st), want.get(&k));
                        same(point_m(&id, &o, thoughts, lanes, k, &mut st), want.get(&k));
                    }
                    let mut gs = BTreeMap::new();
                    let mut gm = BTreeMap::new();
                    let mut bufs: Vec<Vec<(u32, u32)>> = (0..=16).map(|_| Vec::new()).collect();
                    for p in 0..16 {
                        range_s(&sel, lanes, prefix_of(p), &mut bufs, &mut |k, o, m| {
                            assert!(gs.insert(k, (*o, m)).is_none());
                        });
                        range_m(&id, &o, thoughts, lanes, prefix_of(p), &mut |k, o, m| {
                            assert!(gm.insert(k, (*o, m)).is_none());
                        });
                    }
                    assert_eq!(gs.len(), want.len());
                    assert_eq!(gm.len(), want.len());
                    for (k, w) in &want {
                        let pk = path(*k, depth);
                        same(gs[&pk], Some(w));
                        same(gm[&pk], Some(w));
                    }
                }
            }
        }
    }

    #[test]
    fn dense_tile_direct_rows_match_the_oracle_and_the_trie() {
        let univ = dense_universe();
        let cov = coverage(&univ, 5, 0.05, 3);
        let o = build_m(&univ, &cov);
        let id = identity(&univ, 8);
        for thoughts in [0b1_1111u64, 0b1_0010] {
            for lanes in [1u32, (1 << LANES) - 1] {
                let want = oracle(&cov, thoughts, lanes);
                let mut st = 0;
                for &k in univ.iter().step_by(7) {
                    let d = point_d(&o, thoughts, lanes, k);
                    same(d, want.get(&k));
                    let m = point_m(&id, &o, thoughts, lanes, k, &mut st);
                    assert_eq!(d.1, m.1);
                }
                let mut got = BTreeMap::new();
                for p in 0..16 {
                    let (lo, hi) = rows_d(prefix_of(p));
                    assert_eq!((lo as u32, hi as u32), id.trie.range(prefix_of(p), 1));
                    range_d(&univ, &o, thoughts, lanes, prefix_of(p), &mut |k, v, m| {
                        assert!(got.insert(k, (*v, m)).is_none());
                    });
                }
                assert_eq!(got.len(), want.len());
                for (k, w) in &want {
                    same(got[&path(*k, 8)], Some(w));
                }
            }
        }
        assert_eq!(row_d((CLASSID + 1) << 32), None, "other class");
        assert_eq!(row_d((CLASSID << 32) | 1), None, "HIP set");
    }

    #[test]
    fn classid_out_of_the_trie_halves_the_descent() {
        let (univ, _) = fixture();
        let (a, b) = (identity(&univ, 16), identity(&univ, 8));
        let (mut s16, mut s8) = (0, 0);
        for &k in &univ {
            assert_eq!(a.trie.lookup(k, &mut s16), b.trie.lookup(k, &mut s8));
        }
        assert_eq!(s16, 16 * univ.len() as u64);
        assert_eq!(s8, 8 * univ.len() as u64);
    }

    #[test]
    fn identity_rank_is_the_row() {
        let (univ, _) = fixture();
        let id = identity(&univ, 8);
        let mut st = 0;
        for (i, &k) in univ.iter().enumerate() {
            let r = id.trie.lookup(k, &mut st).unwrap() as usize;
            assert_eq!(r, i);
            assert_eq!(&id.rows[r][0][..8], &k.to_le_bytes());
        }
        let absent = (CLASSID << 32) | 0xFFFF_FFFF;
        assert!(!univ.contains(&absent));
        assert_eq!(id.trie.lookup(absent, &mut st), None);
    }

    #[test]
    fn prefix_range_is_exactly_the_keys_under_it() {
        let (univ, _) = fixture();
        let id = identity(&univ, 8);
        let mut total = 0;
        for p in 0..16 {
            let (lo, hi) = id.trie.range(prefix_of(p), 1);
            let want: Vec<usize> = (0..univ.len())
                .filter(|&i| (univ[i] >> 28) & 0xF == p)
                .collect();
            assert_eq!((lo as usize..hi as usize).collect::<Vec<_>>(), want);
            total += want.len();
        }
        assert_eq!(total, univ.len());
    }

    #[test]
    fn a_session_sees_identity_plus_only_its_own_overlays() {
        let (univ, cov) = fixture();
        let o = build_m(&univ, &cov);
        let id = identity(&univ, 8);
        let mut buf = Vec::new();
        for s in [0usize, 3] {
            let want = oracle(&cov, 1 << s, (1 << LANES) - 1);
            let mut changed = 0;
            for p in 0..16 {
                materialize(&id, &o, 1 << s, prefix_of(p), &mut buf);
                let (lo, _) = id.trie.range(prefix_of(p), 1);
                for (i, row) in buf.iter().enumerate() {
                    let r = lo as usize + i;
                    let k = univ[r];
                    assert_eq!(row[0], id.rows[r][0], "key lane never changes");
                    assert_eq!(row[1], id.rows[r][1], "edge lane untouched");
                    for lane in 0..LANES {
                        let exp = match want.get(&k) {
                            Some((wo, wm)) if wm >> lane & 1 == 1 => wo[lane],
                            _ => id.rows[r][2 + lane],
                        };
                        assert_eq!(row[2 + lane], exp);
                    }
                    changed += (*row != id.rows[r]) as usize;
                }
            }
            assert_eq!(changed, want.len(), "exactly the rows this session wrote");
            assert!(changed > 0 && changed < univ.len());
        }
    }
}
