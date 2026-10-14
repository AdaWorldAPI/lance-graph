//! Lane-tables addressed by the NodeGuid trie: per-lane tries (P) vs one
//! shared trie per thought with a lane mask per leaf (S).
//!
//! Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
//! ("Lane-tables addressed by the NodeGuid trie").
//!
//! A read is a merge on read over a selection (prefix x thought mask x lane
//! mask), keyed by the NodeGuid: every selected trie is descended in
//! lockstep, a trie drops out where its child bit is clear, and the selected
//! lanes of the surviving tries are overlaid in thought order (a higher
//! thought index wins a shared lane). The result is the merged delta (30
//! value lanes plus the mask of lanes written); applying it to the base row
//! is the same in both arms and is not timed.
//!
//! Trie (both arms): level-ordered, fan-out 16, one 16-bit child mask per
//! node, child index = first child + popcount(mask below the nibble). Path =
//! classid (8 nibbles) + HEEL (4) + HIP (4) = 16 levels; the leaf index is the
//! key's rank in the trie.
//!
//! Uses no crate types; builds with plain `rustc -O`.
//! Run: `cargo run --release --example lane_trie_probe` (tier is printed).
//! Tests: `cargo test --example lane_trie_probe`.

use std::hint::black_box;
use std::time::Instant;

const LEVELS: usize = 16;
const LANES: usize = 30;
const CLASSID: u64 = 0x0C0C_0000;
type Cell = [u8; 16];
type Lanes = [Cell; LANES];

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

fn nib(key: u64, level: usize) -> u32 {
    ((key >> (60 - 4 * level)) & 0xF) as u32
}

/// The value thought `t` wrote into `lane` of `key`.
fn val(t: usize, key: u64, lane: usize) -> Cell {
    let a = mix(key ^ ((t as u64) << 48) ^ ((lane as u64) << 40));
    let b = mix(a);
    let mut c = [0u8; 16];
    c[..8].copy_from_slice(&a.to_le_bytes());
    c[8..].copy_from_slice(&b.to_le_bytes());
    c
}

struct BitIter(u32);
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

#[derive(Default)]
struct Trie {
    masks: Vec<Vec<u16>>,
    first: Vec<Vec<u32>>,
    leaves: usize,
}

impl Trie {
    /// `keys` sorted and unique.
    fn build(keys: &[u64]) -> Self {
        let mut t = Trie {
            masks: vec![Vec::new(); LEVELS],
            first: vec![Vec::new(); LEVELS],
            leaves: keys.len(),
        };
        if keys.is_empty() {
            return t;
        }
        let mut groups = vec![(0usize, keys.len())];
        for level in 0..LEVELS {
            let mut next = Vec::with_capacity(groups.len() * 2);
            for &(s, e) in &groups {
                t.first[level].push(next.len() as u32);
                let mut mask = 0u16;
                let mut i = s;
                while i < e {
                    let n = nib(keys[i], level);
                    let mut j = i + 1;
                    while j < e && nib(keys[j], level) == n {
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
        debug_assert_eq!(groups.len(), keys.len());
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
}

// ---------- data ----------

/// Per thought, the keys it covers (sorted) with the lanes it wrote there.
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

/// Thought t covers each key with probability `d` and writes `w` distinct
/// lanes per covered key, drawn per key.
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

// ---------- arms ----------

/// One lane-table of arm P.
struct LaneTable {
    lane: usize,
    trie: Trie,
    payload: Vec<Cell>,
}

/// One thought-table of arm S.
struct ThoughtTable {
    trie: Trie,
    lanes: Vec<u32>,
    starts: Vec<u32>,
    payload: Vec<Cell>,
}

fn build_p(cov: &Coverage) -> Vec<Vec<LaneTable>> {
    cov.iter()
        .enumerate()
        .map(|(t, c)| {
            (0..LANES)
                .map(|lane| {
                    let keys: Vec<u64> = c
                        .iter()
                        .filter(|e| e.1 & (1 << lane) != 0)
                        .map(|e| e.0)
                        .collect();
                    LaneTable {
                        lane,
                        trie: Trie::build(&keys),
                        payload: keys.iter().map(|&k| val(t, k, lane)).collect(),
                    }
                })
                .collect()
        })
        .collect()
}

fn build_s(cov: &Coverage) -> Vec<ThoughtTable> {
    cov.iter()
        .enumerate()
        .map(|(t, c)| {
            let keys: Vec<u64> = c.iter().map(|e| e.0).collect();
            let mut starts = Vec::with_capacity(c.len());
            let mut payload = Vec::new();
            for &(k, m) in c {
                starts.push(payload.len() as u32);
                payload.extend(BitIter(m).map(|lane| val(t, k, lane)));
            }
            ThoughtTable {
                trie: Trie::build(&keys),
                lanes: c.iter().map(|e| e.1).collect(),
                starts,
                payload,
            }
        })
        .collect()
}

fn bytes_p(p: &[Vec<LaneTable>]) -> usize {
    p.iter()
        .flatten()
        .map(|l| l.trie.bytes() + l.payload.len() * 16)
        .sum()
}

fn bytes_s(s: &[ThoughtTable]) -> usize {
    s.iter()
        .map(|t| t.trie.bytes() + t.lanes.len() * 8 + t.payload.len() * 16)
        .sum()
}

/// What both arms share: a trie and a way to overlay a leaf.
trait Entry {
    fn trie(&self) -> &Trie;
    fn emit(&self, leaf: u32, sel_lanes: u32, out: &mut Lanes) -> u32;
}

impl Entry for &LaneTable {
    fn trie(&self) -> &Trie {
        &self.trie
    }
    #[inline]
    fn emit(&self, leaf: u32, _sel: u32, out: &mut Lanes) -> u32 {
        out[self.lane] = self.payload[leaf as usize];
        1 << self.lane
    }
}

impl Entry for &ThoughtTable {
    fn trie(&self) -> &Trie {
        &self.trie
    }
    #[inline]
    fn emit(&self, leaf: u32, sel: u32, out: &mut Lanes) -> u32 {
        let have = self.lanes[leaf as usize];
        let start = self.starts[leaf as usize] as usize;
        let m = have & sel;
        for lane in BitIter(m) {
            out[lane] = self.payload[start + (have & ((1u32 << lane) - 1)).count_ones() as usize];
        }
        m
    }
}

/// Arm P selection: the non-empty lane-tables of the selected thoughts and
/// lanes, in (thought, lane) order.
fn select_p(p: &[Vec<LaneTable>], thoughts: u64, lanes: u32) -> Vec<&LaneTable> {
    p.iter()
        .enumerate()
        .filter(|(t, _)| thoughts >> t & 1 == 1)
        .flat_map(|(_, ls)| ls.iter())
        .filter(|l| lanes >> l.lane & 1 == 1 && !l.trie.is_empty())
        .collect()
}

fn select_s(s: &[ThoughtTable], thoughts: u64) -> Vec<&ThoughtTable> {
    s.iter()
        .enumerate()
        .filter(|(t, tt)| thoughts >> t & 1 == 1 && !tt.trie.is_empty())
        .map(|(_, tt)| tt)
        .collect()
}

// ---------- reads ----------

/// Lockstep point read. `scratch` holds (entry, node) of the alive tries.
fn point<E: Entry>(
    sel: &[E],
    sel_lanes: u32,
    key: u64,
    scratch: &mut Vec<(u32, u32)>,
    steps: &mut u64,
) -> (Lanes, u32) {
    scratch.clear();
    scratch.extend((0..sel.len() as u32).map(|i| (i, 0)));
    for level in 0..LEVELS {
        let n = nib(key, level);
        let mut k = 0;
        for j in 0..scratch.len() {
            let (i, node) = scratch[j];
            *steps += 1;
            if let Some(c) = sel[i as usize].trie().child(level, node, n) {
                scratch[k] = (i, c);
                k += 1;
            }
        }
        scratch.truncate(k);
        if k == 0 {
            break;
        }
    }
    let mut out = [[0u8; 16]; LANES];
    let mut m = 0;
    for &(i, leaf) in scratch.iter() {
        m |= sel[i as usize].emit(leaf, sel_lanes, &mut out);
    }
    (out, m)
}

/// Lockstep range read over every key under the first `plen` nibbles of
/// `prefix`; `f` gets each covered key with its merged lanes.
fn range<E: Entry, F: FnMut(u64, &Lanes, u32)>(
    sel: &[E],
    sel_lanes: u32,
    prefix: u64,
    plen: usize,
    bufs: &mut [Vec<(u32, u32)>],
    f: &mut F,
) {
    let (head, rest) = bufs.split_first_mut().unwrap();
    head.clear();
    head.extend((0..sel.len() as u32).map(|i| (i, 0)));
    for level in 0..plen {
        let n = nib(prefix, level);
        head.retain_mut(|e| match sel[e.0 as usize].trie().child(level, e.1, n) {
            Some(c) => {
                e.1 = c;
                true
            }
            None => false,
        });
    }
    if !head.is_empty() {
        dfs(sel, sel_lanes, plen, prefix, head, rest, f);
    }
}

fn dfs<E: Entry, F: FnMut(u64, &Lanes, u32)>(
    sel: &[E],
    sel_lanes: u32,
    level: usize,
    key: u64,
    alive: &[(u32, u32)],
    bufs: &mut [Vec<(u32, u32)>],
    f: &mut F,
) {
    if level == LEVELS {
        let mut out = [[0u8; 16]; LANES];
        let mut m = 0;
        for &(i, leaf) in alive {
            m |= sel[i as usize].emit(leaf, sel_lanes, &mut out);
        }
        if m != 0 {
            f(key, &out, m);
        }
        return;
    }
    let mut union = 0u16;
    for &(i, node) in alive {
        union |= sel[i as usize].trie().mask(level, node);
    }
    let (cur, rest) = bufs.split_first_mut().unwrap();
    for n in BitIter(union as u32) {
        cur.clear();
        for &(i, node) in alive {
            if let Some(c) = sel[i as usize].trie().child(level, node, n as u32) {
                cur.push((i, c));
            }
        }
        let child_key = key | ((n as u64) << (60 - 4 * level));
        dfs(sel, sel_lanes, level + 1, child_key, cur, rest, f);
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

struct Row {
    q1: usize,
    q2: f64,
    q2_step: f64,
    q3: f64,
    emitted: u64,
}

fn measure<E: Entry>(sel: &[E], sel_lanes: u32, probes: &[u64], q1: usize) -> Row {
    let mut scratch = Vec::with_capacity(sel.len());
    let mut steps = 0u64;
    for &k in probes {
        black_box(point(sel, sel_lanes, k, &mut scratch, &mut steps));
    }
    let mut dummy = 0u64;
    let q2 = ns_per(probes.len() as u64, || {
        for &k in probes {
            black_box(point(sel, sel_lanes, k, &mut scratch, &mut dummy));
        }
    });
    let mut bufs: Vec<Vec<(u32, u32)>> = (0..=LEVELS)
        .map(|_| Vec::with_capacity(sel.len()))
        .collect();
    let mut emitted = 0u64;
    for p in 0..16 {
        range(
            sel,
            sel_lanes,
            prefix_of(p),
            9,
            &mut bufs,
            &mut |_, _, _| emitted += 1,
        );
    }
    let q3 = ns_per(emitted, || {
        let mut acc = 0u64;
        for p in 0..16 {
            range(
                sel,
                sel_lanes,
                prefix_of(p),
                9,
                &mut bufs,
                &mut |k, o, m| acc ^= k ^ m as u64 ^ o[0][0] as u64,
            );
        }
        black_box(acc);
    });
    Row {
        q1,
        q2,
        q2_step: q2 * probes.len() as f64 / steps.max(1) as f64,
        q3,
        emitted,
    }
}

fn main() {
    const N: usize = 65_536;
    println!("tier: {}", tier());
    let univ = universe(N);
    let mut r = 12345u64;
    let probes: Vec<u64> = (0..N)
        .map(|_| {
            r = mix(r);
            univ[(r % N as u64) as usize]
        })
        .collect();
    println!(
        "\n{:>3} {:>5} {:>3} {:>5} | {:>11} {:>11} | {:>9} {:>9} | {:>7} {:>7} | {:>7} {:>7} | {:>8}",
        "T", "d", "w", "sel", "Q1 P B", "Q1 S B", "Q2 P ns", "Q2 S ns", "step P", "step S",
        "Q3 P", "Q3 S", "emitted"
    );
    for t in [1usize, 8, 64] {
        for (dl, d) in [("1%", 0.01), ("10%", 0.10), ("100%", 1.0)] {
            for w in [1usize, 4, 30] {
                let cells = (t as f64 * N as f64 * d * w as f64) as u64;
                if cells > 32_000_000 {
                    println!("{t:>3} {dl:>5} {w:>3}   skipped ({cells} lane-cells > 32 M)");
                    continue;
                }
                let cov = coverage(&univ, t, d, w);
                let p = build_p(&cov);
                let s = build_s(&cov);
                let (bp, bs) = (bytes_p(&p), bytes_s(&s));
                let thoughts = if t == 64 { u64::MAX } else { (1u64 << t) - 1 };
                for (sl, lanes) in [("1", 1u32), ("all", (1u32 << LANES) - 1)] {
                    let rp = measure(&select_p(&p, thoughts, lanes), lanes, &probes, bp);
                    let rs = measure(&select_s(&s, thoughts), lanes, &probes, bs);
                    assert_eq!(rp.emitted, rs.emitted);
                    println!(
                        "{t:>3} {dl:>5} {w:>3} {sl:>5} | {:>11} {:>11} | {:>9.1} {:>9.1} | {:>7.2} {:>7.2} | {:>7.1} {:>7.1} | {:>8}",
                        rp.q1, rs.q1, rp.q2, rs.q2, rp.q2_step, rs.q2_step, rp.q3, rs.q3, rp.emitted
                    );
                }
            }
        }
    }
    println!("\nQ1 bytes; Q2 ns per point read; step = ns per trie-level step; Q3 ns per emitted NodeGuid.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Brute force: merged lanes per key, thoughts in index order.
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
                for lane in BitIter(m) {
                    e.0[lane] = val(t, k, lane);
                }
                e.1 |= m;
            }
        }
        out
    }

    fn check<E: Entry>(sel: &[E], cov: &Coverage, univ: &[u64], thoughts: u64, lanes: u32) {
        let want = oracle(cov, thoughts, lanes);
        let mut scratch = Vec::new();
        let mut steps = 0;
        for &k in univ {
            let (o, m) = point(sel, lanes, k, &mut scratch, &mut steps);
            match want.get(&k) {
                Some(&(wo, wm)) => {
                    assert_eq!(m, wm, "mask at {k:x}");
                    for lane in BitIter(wm) {
                        assert_eq!(o[lane], wo[lane], "lane {lane} at {k:x}");
                    }
                }
                None => assert_eq!(m, 0, "{k:x} not covered"),
            }
        }
        let mut got = BTreeMap::new();
        let mut bufs: Vec<Vec<(u32, u32)>> = (0..=LEVELS).map(|_| Vec::new()).collect();
        for p in 0..16 {
            range(sel, lanes, prefix_of(p), 9, &mut bufs, &mut |k, o, m| {
                assert!(got.insert(k, (*o, m)).is_none(), "duplicate {k:x}");
            });
        }
        assert_eq!(got.len(), want.len(), "range covers every covered key once");
        for (k, (wo, wm)) in &want {
            let (o, m) = got[k];
            assert_eq!(m, *wm);
            for lane in BitIter(m) {
                assert_eq!(o[lane], wo[lane]);
            }
        }
    }

    #[test]
    fn both_arms_match_the_oracle() {
        let univ = universe(3000);
        let cov = coverage(&univ, 6, 0.3, 4);
        let p = build_p(&cov);
        let s = build_s(&cov);
        for thoughts in [0b11_1111u64, 0b10_1001] {
            for lanes in [1u32, 0b1011_0000_0001, (1 << LANES) - 1] {
                check(&select_p(&p, thoughts, lanes), &cov, &univ, thoughts, lanes);
                check(&select_s(&s, thoughts), &cov, &univ, thoughts, lanes);
            }
        }
    }

    #[test]
    fn a_higher_thought_wins_a_shared_lane() {
        let univ = universe(200);
        let cov = coverage(&univ, 2, 1.0, LANES);
        let s = build_s(&cov);
        let p = build_p(&cov);
        let mut scratch = Vec::new();
        let mut steps = 0;
        let k = univ[17];
        let (os, _) = point(&select_s(&s, 0b11), 1, k, &mut scratch, &mut steps);
        let (op, _) = point(&select_p(&p, 0b11, 1), 1, k, &mut scratch, &mut steps);
        assert_eq!(os[0], val(1, k, 0));
        assert_eq!(op[0], val(1, k, 0));
        assert_ne!(val(0, k, 0), val(1, k, 0));
    }

    #[test]
    fn an_uncovered_key_drops_every_trie() {
        let univ = universe(500);
        let cov = coverage(&univ, 4, 0.5, 2);
        let s = build_s(&cov);
        let mut scratch = Vec::new();
        let mut steps = 0;
        let absent = (CLASSID << 32) | 0xFFFF_FFFF;
        assert!(!univ.contains(&absent));
        let (_, m) = point(
            &select_s(&s, 0b1111),
            (1 << LANES) - 1,
            absent,
            &mut scratch,
            &mut steps,
        );
        assert_eq!(m, 0);
        assert!(scratch.is_empty());
    }

    #[test]
    fn trie_ranks_keys_in_order() {
        let univ = universe(1000);
        let t = Trie::build(&univ);
        let mut scratch = Vec::new();
        let mut steps = 0;
        let lt = LaneTable {
            lane: 0,
            trie: t,
            payload: univ.iter().map(|&k| val(0, k, 0)).collect(),
        };
        let sel = [&lt];
        for (i, &k) in univ.iter().enumerate() {
            let (o, m) = point(&sel, 1, k, &mut scratch, &mut steps);
            assert_eq!(m, 1);
            assert_eq!(o[0], lt.payload[i]);
        }
    }

    #[test]
    fn empty_lane_tables_are_not_selected() {
        let univ = universe(400);
        let cov = coverage(&univ, 3, 0.2, 1);
        let p = build_p(&cov);
        let sel = select_p(&p, 0b111, (1 << LANES) - 1);
        assert!(sel.iter().all(|l| !l.trie.is_empty()));
        let total: usize = p.iter().flatten().filter(|l| !l.trie.is_empty()).count();
        assert_eq!(sel.len(), total);
        assert!(total < 3 * LANES, "some lane-tables stay empty at w = 1");
    }
}
