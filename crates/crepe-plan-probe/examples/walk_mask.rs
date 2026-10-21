//! crepe's `graph_walking` benchmark program, evaluated three ways: by the
//! code `crepe!` generates (hash sets and hash-map indexes, tuple at a time),
//! by `ascent!`, and over dense bit planes with `ndarray::simd` mask ops. Run
//! in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example walk_mask    # the benchmark
//! cargo test --example walk_mask             # the correctness fixtures
//! ```
//!
//! The program (copied from `crepe/benches/graph_walking.rs`):
//!
//! ```text
//! Node(x) <- Edge(x, _, _);   Node(x) <- Edge(_, x, _);
//! Walk(x, x, 0) <- Node(x);
//! Walk(x, z, l1 + l2) <- Edge(x, y, l1), Walk(y, z, l2), (l1 + l2 <= 50);
//! NoWalk(x, y) <- Node(x), Node(y), !Walk(x, y, _);
//! ```
//!
//! The mask form needs what the macro does not know: node ids dense in
//! `0..n`, and path lengths in `0..=MAX_PATH_LEN`. `Walk` is one `n x n` bit
//! plane per length; an edge of length `w` from `x` to `y` makes
//! `walk[l + w][x] |= delta[l][y]`, the semi-naive step is
//! `new &= !walk; walk |= new`, and the fixpoint is "every new plane is
//! zero". `NoWalk` is `node x node & !(OR over l of walk[l])`.
//!
//! # Correctness and speed are reported separately
//!
//! * **Oracle.** The bit planes are decoded into the canonical sets
//!   `Walk(x, z, len)` and `NoWalk(x, y)` and compared, tuple for tuple, with
//!   the sets ascent and crepe produce. Decoding happens outside every timed
//!   region. Above [`ORACLE_MAX_N`] neither oracle runs, and the table says
//!   `UNCHECKED` rather than comparing the mask result with itself.
//! * **Count-only time** is the fixpoint plus two popcounts; no tuple is
//!   built. **Tuple time** is decoding the finished planes into tuples, timed
//!   on its own. The crepe and ascent columns time their native runs, which
//!   build their tuples as they go.
//!
//! # What this is not
//!
//! The hop reads `delta[row(l2, y)]`, where `y` is a set bit of an adjacency
//! row: a row address taken from a value. That is a gather, which the
//! no-gather rule excludes from ADMISSION-B operators. The table counts these
//! value-addressed row reads. The result is a lab upper bound on what a dense
//! layout buys, to compare against a co-addressed tile path and against later
//! Datalog lowerings; it is not a candidate production operator.
//!
//! # Memory
//!
//! One length-indexed plane set is `(MAX_PATH_LEN + 1) * n * ceil(n / 64) * 8`
//! bytes, about `(L + 1) n^2 / 8`, and the fixpoint holds three (`walk`,
//! `delta`, `new`). At `L = 50` that is 1.6 MiB per structure at 512 nodes,
//! 6.4 MiB at 1,024, and 25.5 GiB at 65,536: a dense global matrix is not a
//! representation for an identity-scale graph.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use crepe::crepe;
use ndarray::simd::{mask_andnot_assign, mask_any, mask_or_assign, popcount_batch_u64};

pub(crate) const MAX_PATH_LEN: u32 = 50;
const REPS: usize = 9;
/// Largest `n` the benchmark runs the ascent and crepe oracles for.
const ORACLE_MAX_N: usize = 512;

crepe! {
    @input
    struct Edge(i32, i32, u32);

    @output
    struct Walk(i32, i32, u32);

    @output
    struct NoWalk(i32, i32);

    struct Node(i32);

    Node(x) <- Edge(x, _, _);
    Node(x) <- Edge(_, x, _);

    Walk(x, x, 0) <- Node(x);
    Walk(x, z, len1 + len2) <-
        Edge(x, y, len1),
        Walk(y, z, len2),
        (len1 + len2 <= MAX_PATH_LEN);

    NoWalk(x, y) <- Node(x), Node(y), !Walk(x, y, _);
}

ascent::ascent! {
    struct WalkAscent;
    relation edge(i32, i32, u32);
    relation node(i32);
    relation walk(i32, i32, u32);
    relation no_walk(i32, i32);

    node(x) <-- edge(x, _, _);
    node(x) <-- edge(_, x, _);
    walk(x, x, 0) <-- node(x);
    walk(x, z, l1 + l2) <-- edge(x, y, l1), walk(y, z, l2), if l1 + l2 <= MAX_PATH_LEN;
    no_walk(x, y) <-- node(x), node(y), !walk(x, y, _);
}

/// The benchmark's edges (every edge has length 5).
pub(crate) fn edges(n: i32) -> Vec<(i32, i32, u32)> {
    let mut e = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if (i + j) % 50 < 2 {
                e.push((i, j, 5));
            }
        }
    }
    e
}

/// The two output relations as canonical sets.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Tuples {
    pub(crate) walk: BTreeSet<(i32, i32, u32)>,
    pub(crate) no_walk: BTreeSet<(i32, i32)>,
}

impl Tuples {
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.walk.len(), self.no_walk.len())
    }
}

fn run_ascent(e: &[(i32, i32, u32)]) -> WalkAscent {
    let mut p = WalkAscent {
        edge: e.to_vec(),
        ..Default::default()
    };
    p.run();
    p
}

pub(crate) fn ascent_walk(e: &[(i32, i32, u32)]) -> (usize, usize) {
    let p = run_ascent(e);
    (p.walk.len(), p.no_walk.len())
}

pub(crate) fn ascent_tuples(e: &[(i32, i32, u32)]) -> Tuples {
    let p = run_ascent(e);
    Tuples {
        walk: p.walk.into_iter().collect(),
        no_walk: p.no_walk.into_iter().collect(),
    }
}

fn run_crepe(e: &[(i32, i32, u32)]) -> (Vec<Walk>, Vec<NoWalk>) {
    let mut rt = Crepe::new();
    rt.extend(e.iter().map(|&(a, b, w)| Edge(a, b, w)));
    let (walk, nowalk) = rt.run();
    (walk.into_iter().collect(), nowalk.into_iter().collect())
}

pub(crate) fn crepe_walk(e: &[(i32, i32, u32)]) -> (usize, usize) {
    let (walk, nowalk) = run_crepe(e);
    (walk.len(), nowalk.len())
}

pub(crate) fn crepe_tuples(e: &[(i32, i32, u32)]) -> Tuples {
    let (walk, nowalk) = run_crepe(e);
    Tuples {
        walk: walk.into_iter().map(|Walk(x, z, l)| (x, z, l)).collect(),
        no_walk: nowalk.into_iter().map(|NoWalk(x, y)| (x, y)).collect(),
    }
}

/// Words per row of an `n`-wide plane.
pub(crate) fn words(n: usize) -> usize {
    n.div_ceil(64)
}

pub(crate) fn set(row: &mut [u64], i: usize) {
    row[i / 64] |= 1 << (i % 64);
}

pub(crate) fn bit(row: &[u64], i: usize) -> bool {
    row[i / 64] >> (i % 64) & 1 == 1
}

/// The fixpoint of the program over bit planes.
pub(crate) struct Planes {
    pub(crate) n: usize,
    pub(crate) w: usize,
    pub(crate) lens: usize,
    pub(crate) node: Vec<u64>,
    /// `walk[l][x]`: the nodes `z` with `Walk(x, z, l)`.
    pub(crate) walk: Vec<u64>,
    /// Value-addressed row reads (`delta[row(l2, y)]` for a set bit `y`).
    pub(crate) row_reads: u64,
}

impl Planes {
    fn row(&self, l: usize, x: usize) -> usize {
        (l * self.n + x) * self.w
    }

    fn walk_row(&self, l: usize, x: usize) -> &[u64] {
        let r = self.row(l, x);
        &self.walk[r..r + self.w]
    }

    fn is_node(&self, x: usize) -> bool {
        bit(&self.node, x)
    }

    /// `(|Walk|, |NoWalk|)` from popcounts; no tuple is built.
    pub(crate) fn counts(&self) -> (usize, usize) {
        let walk = popcount_batch_u64(&self.walk) as usize;
        let mut reach = vec![0u64; self.w];
        let mut miss = vec![0u64; self.w];
        let mut nowalk = 0usize;
        for x in (0..self.n).filter(|&x| self.is_node(x)) {
            reach.fill(0);
            for l in 0..self.lens {
                mask_or_assign(&mut reach, self.walk_row(l, x));
            }
            miss.copy_from_slice(&self.node);
            mask_andnot_assign(&mut miss, &reach);
            nowalk += popcount_batch_u64(&miss) as usize;
        }
        (walk, nowalk)
    }

    /// Decode the planes into the canonical tuple sets.
    pub(crate) fn tuples(&self) -> Tuples {
        let mut walk = BTreeSet::new();
        let mut no_walk = BTreeSet::new();
        for x in (0..self.n).filter(|&x| self.is_node(x)) {
            let mut reach = vec![0u64; self.w];
            for l in 0..self.lens {
                let r = self.walk_row(l, x);
                mask_or_assign(&mut reach, r);
                for z in (0..self.n).filter(|&z| bit(r, z)) {
                    walk.insert((x as i32, z as i32, l as u32));
                }
            }
            for y in (0..self.n).filter(|&y| self.is_node(y) && !bit(&reach, y)) {
                no_walk.insert((x as i32, y as i32));
            }
        }
        Tuples { walk, no_walk }
    }

    /// Bytes of one length-indexed plane set; the fixpoint holds three.
    pub(crate) fn plane_bytes(&self) -> usize {
        self.walk.len() * 8
    }
}

/// The program over bit planes. Node ids must lie in `0..n`.
pub(crate) fn mask_walk(n: usize, e: &[(i32, i32, u32)]) -> Planes {
    let w = words(n);
    let lens = MAX_PATH_LEN as usize + 1;
    let row = |plane: usize, x: usize| (plane * n + x) * w;

    // Edges grouped by length: adjacency rows `x -> {y}`. An edge longer than
    // MAX_PATH_LEN starts no walk, but its ends are still nodes.
    let mut by_len: Vec<Option<Vec<u64>>> = vec![None; lens];
    let mut node = vec![0u64; w];
    for &(a, b, l) in e {
        assert!(
            (0..n as i32).contains(&a) && (0..n as i32).contains(&b),
            "edge ({a}, {b}) outside 0..{n}"
        );
        let (a, b, l) = (a as usize, b as usize, l as usize);
        set(&mut node, a);
        set(&mut node, b);
        if l < lens {
            let adj = by_len[l].get_or_insert_with(|| vec![0u64; n * w]);
            set(&mut adj[a * w..a * w + w], b);
        }
    }

    // walk[l][x] and the delta planes; Walk(x, x, 0) for every node.
    let mut walk = vec![0u64; lens * n * w];
    let mut delta = vec![0u64; lens * n * w];
    let mut live = vec![false; lens];
    for x in (0..n).filter(|&x| bit(&node, x)) {
        set(&mut walk[row(0, x)..row(0, x) + w], x);
        set(&mut delta[row(0, x)..row(0, x) + w], x);
    }
    live[0] = true;

    let mut row_reads = 0u64;
    let mut new = vec![0u64; lens * n * w];
    loop {
        new.fill(0);
        let mut new_live = vec![false; lens];
        for (l1, adj) in by_len.iter().enumerate() {
            let Some(adj) = adj else { continue };
            for l2 in (0..lens).filter(|&l2| live[l2]) {
                let l = l1 + l2;
                if l >= lens {
                    continue;
                }
                for x in 0..n {
                    let ar = &adj[x * w..x * w + w];
                    let dst = row(l, x);
                    for (wi, &word) in ar.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let y = wi * 64 + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            // The gather: a row address taken from a value.
                            let src = row(l2, y);
                            row_reads += 1;
                            // `new` and `delta` are different buffers.
                            mask_or_assign(&mut new[dst..dst + w], &delta[src..src + w]);
                        }
                    }
                }
                new_live[l] = true;
            }
        }
        let mut any = false;
        for (l, is_live) in new_live.iter_mut().enumerate() {
            if !*is_live {
                continue;
            }
            let span = row(l, 0)..row(l, 0) + n * w;
            mask_andnot_assign(&mut new[span.clone()], &walk[span.clone()]);
            if mask_any(&new[span.clone()]) {
                mask_or_assign(&mut walk[span.clone()], &new[span.clone()]);
                any = true;
            } else {
                *is_live = false;
            }
        }
        if !any {
            break;
        }
        std::mem::swap(&mut delta, &mut new);
        live = new_live;
    }

    Planes {
        n,
        w,
        lens,
        node,
        walk,
        row_reads,
    }
}

/// The first few tuples in one set and not the other, for a failure message.
fn difference<T: Ord + std::fmt::Debug>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> String {
    let only_a: Vec<_> = a.difference(b).take(5).collect();
    let only_b: Vec<_> = b.difference(a).take(5).collect();
    format!("only in mask {only_a:?}, only in oracle {only_b:?}")
}

/// Compare the mask tuples with one oracle's; `Err` names the difference.
pub(crate) fn agree(mask: &Tuples, oracle: &Tuples) -> Result<(), String> {
    if mask.walk != oracle.walk {
        return Err(format!("Walk: {}", difference(&mask.walk, &oracle.walk)));
    }
    if mask.no_walk != oracle.no_walk {
        return Err(format!(
            "NoWalk: {}",
            difference(&mask.no_walk, &oracle.no_walk)
        ));
    }
    Ok(())
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut t = Vec::with_capacity(REPS);
    let mut out = None;
    for _ in 0..REPS {
        let s = Instant::now();
        out = Some(black_box(f()));
        t.push(s.elapsed());
    }
    (median(t), out.expect("REPS > 0"))
}

fn ms(t: Option<Duration>) -> String {
    t.map_or("-".into(), |t| format!("{:.3}", t.as_secs_f64() * 1e3))
}

fn main() {
    println!(
        "{:>5} | {:>9} {:>9} | {:>9} {:>9} | {:>7} {:>7} | {:>9} {:>8} | {:>9} {:>10} | oracle",
        "n",
        "crepe ms",
        "ascent ms",
        "count ms",
        "tuple ms",
        "vs crp",
        "vs asc",
        "Walk",
        "NoWalk",
        "MiB x3",
        "row reads"
    );
    let mut failed = Vec::new();
    for n in [128usize, 256, 512, 1024] {
        let e = edges(n as i32);
        // Count-only: the fixpoint plus popcounts, no tuples.
        let (t_count, counts) = time(|| mask_walk(n, &e).counts());
        // Tuple materialization of finished planes, timed on its own.
        let planes = mask_walk(n, &e);
        let (t_tuples, mask) = time(|| planes.tuples());
        assert_eq!(
            counts,
            mask.counts(),
            "n={n}: popcounts disagree with the decoded tuples"
        );

        let (t_crepe, t_ascent, oracle) = if n <= ORACLE_MAX_N {
            let (tc, _) = time(|| crepe_walk(&e));
            let (ta, _) = time(|| ascent_walk(&e));
            let verdict = agree(&mask, &ascent_tuples(&e))
                .map_err(|d| format!("ascent {d}"))
                .and_then(|()| agree(&mask, &crepe_tuples(&e)).map_err(|d| format!("crepe {d}")));
            (Some(tc), Some(ta), Some(verdict))
        } else {
            (None, None, None)
        };
        let speed = |t: Option<Duration>| {
            t.map_or("-".into(), |t| {
                format!("{:.0}x", t.as_secs_f64() / t_count.as_secs_f64())
            })
        };
        let status = match &oracle {
            None => "UNCHECKED".to_string(),
            Some(Ok(())) => "sets equal (ascent, crepe)".to_string(),
            Some(Err(d)) => {
                failed.push(format!("n={n}: {d}"));
                "MISMATCH".to_string()
            }
        };
        println!(
            "{:>5} | {:>9} {:>9} | {:>9} {:>9} | {:>7} {:>7} | {:>9} {:>8} | {:>9.1} {:>10} | {status}",
            n,
            ms(t_crepe),
            ms(t_ascent),
            ms(Some(t_count)),
            ms(Some(t_tuples)),
            speed(t_crepe),
            speed(t_ascent),
            counts.0,
            counts.1,
            3.0 * planes.plane_bytes() as f64 / (1 << 20) as f64,
            planes.row_reads,
        );
    }
    assert!(failed.is_empty(), "oracle mismatch: {failed:#?}");
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The mask tuples equal both oracles' tuples, and the count-only path
    /// agrees with the decoded sets. Returns the mask tuples.
    fn check(n: usize, e: &[(i32, i32, u32)]) -> Tuples {
        let planes = mask_walk(n, e);
        let mask = planes.tuples();
        assert_eq!(planes.counts(), mask.counts(), "popcounts vs tuples");
        agree(&mask, &ascent_tuples(e)).unwrap_or_else(|d| panic!("n={n} ascent: {d}"));
        agree(&mask, &crepe_tuples(e)).unwrap_or_else(|d| panic!("n={n} crepe: {d}"));
        mask
    }

    /// splitmix64, so the fixtures are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    pub(crate) const LENGTHS: [u32; 5] = [1, 2, 3, 5, 7];

    /// `n` nodes, `per_node` random out-edges each with a length from
    /// `LENGTHS`, plus the ring `0 -> 1 -> 2 -> 0`.
    pub(crate) fn random_graph(n: usize, per_node: usize, seed: u64) -> Vec<(i32, i32, u32)> {
        let mut r = Rng(seed);
        let mut e = vec![(0, 1, 1), (1, 2, 2), (2, 0, 3)];
        for x in 0..n {
            for _ in 0..per_node {
                let y = r.below(n);
                e.push((x as i32, y as i32, LENGTHS[r.below(LENGTHS.len())]));
            }
        }
        e
    }

    /// Pairs reached at two or more distinct lengths.
    fn multi_length_pairs(t: &Tuples) -> usize {
        let mut per_pair = std::collections::BTreeMap::new();
        for &(x, z, _) in &t.walk {
            *per_pair.entry((x, z)).or_insert(0usize) += 1;
        }
        per_pair.values().filter(|&&c| c >= 2).count()
    }

    #[test]
    fn mixed_lengths_with_cycles() {
        let e = random_graph(65, 2, 0x5EED_0001);
        let t = check(65, &e);
        let lens: BTreeSet<u32> = t.walk.iter().map(|w| w.2).collect();
        // Anti-vacuity: the fixture reaches pairs at several lengths, spans
        // the length range, and leaves some pairs unreached.
        assert!(multi_length_pairs(&t) > 100, "too few multi-length pairs");
        assert!(lens.contains(&MAX_PATH_LEN), "no walk reaches MAX_PATH_LEN");
        assert!(
            lens.len() == MAX_PATH_LEN as usize + 1,
            "length gap: {lens:?}"
        );
        assert!(!t.no_walk.is_empty(), "NoWalk is empty");
    }

    #[test]
    fn alternative_path_lengths() {
        // 0 -1-> 1 directly, and 0 -2-> 2 -3-> 1; 1 -7-> 3; 0 -5-> 3.
        let e = [(0, 1, 1), (0, 2, 2), (2, 1, 3), (1, 3, 7), (0, 3, 5)];
        let t = check(4, &e);
        // Hand-derived, independent of both oracles.
        let expected: BTreeSet<(i32, i32, u32)> = [
            (0, 0, 0),
            (1, 1, 0),
            (2, 2, 0),
            (3, 3, 0),
            (0, 1, 1),
            (0, 1, 5),
            (0, 2, 2),
            (2, 1, 3),
            (0, 3, 5),
            (0, 3, 8),
            (0, 3, 12),
            (1, 3, 7),
            (2, 3, 10),
        ]
        .into();
        assert_eq!(t.walk, expected);
        let no_walk: BTreeSet<(i32, i32)> = [(1, 0), (1, 2), (2, 0), (3, 0), (3, 1), (3, 2)].into();
        assert_eq!(t.no_walk, no_walk);
    }

    #[test]
    fn a_cycle_stops_at_max_path_len() {
        // A ring of 7 edges of length 7: 0 returns to itself at 0 and 49;
        // the next lap (56) is over the bound.
        let e: Vec<_> = (0..7).map(|i| (i, (i + 1) % 7, 7)).collect();
        let t = check(7, &e);
        assert!(t.walk.contains(&(0, 0, 49)));
        assert!(!t.walk.contains(&(0, 0, 56)));
        assert!(t.walk.iter().all(|w| w.2 <= MAX_PATH_LEN));
        // Each node reaches each node at exactly one length below 50, or two
        // when 7k + 49 <= 50 (k = 0 only).
        assert_eq!(t.walk.len(), 7 * 7 + 7);
        assert!(t.no_walk.is_empty());
    }

    #[test]
    fn max_path_len_is_inclusive() {
        let e = [(0, 1, 50), (2, 3, 51), (4, 5, 25), (5, 6, 25), (6, 7, 1)];
        let t = check(8, &e);
        assert!(t.walk.contains(&(0, 1, 50)));
        assert!(!t.walk.iter().any(|w| (w.0, w.1) == (2, 3)));
        assert!(t.walk.contains(&(4, 6, 50)));
        assert!(!t.walk.iter().any(|w| (w.0, w.1) == (4, 7)));
        // An edge longer than the bound still makes its ends nodes.
        assert!(t.no_walk.contains(&(2, 3)));
        assert!(t.no_walk.contains(&(4, 7)));
    }

    #[test]
    fn word_boundaries() {
        for n in [63usize, 64, 65, 127, 128, 129] {
            let last = n as i32 - 1;
            // A ring through ids on both sides of each 64-bit boundary, plus
            // edges into and out of the last id.
            let ids: Vec<i32> = [0, 62, 63, 64, 65, 126, 127, 128]
                .into_iter()
                .filter(|&i| i < n as i32)
                .collect();
            let mut e = Vec::new();
            for (k, &a) in ids.iter().enumerate() {
                let b = ids[(k + 1) % ids.len()];
                e.push((a, b, LENGTHS[k % LENGTHS.len()]));
            }
            e.push((last, 0, 2));
            e.push((63.min(last), last, 3));
            e.extend(random_graph(n, 1, 0xB0B0 + n as u64));
            let t = check(n, &e);
            assert!(
                t.walk.iter().any(|w| w.0 == last && w.1 != last),
                "n={n}: the last id reaches nothing"
            );
        }
    }

    #[test]
    fn ids_that_are_not_nodes_take_no_part() {
        // Only even ids appear in edges; odd ids in 0..n are not nodes.
        let e: Vec<_> = (0..40)
            .map(|i| (2 * i, 2 * ((i * 7 + 3) % 40), LENGTHS[i as usize % 5]))
            .collect();
        let t = check(80, &e);
        assert!(t.walk.iter().all(|w| w.0 % 2 == 0 && w.1 % 2 == 0));
        assert!(t.no_walk.iter().all(|w| w.0 % 2 == 0 && w.1 % 2 == 0));
        assert!(!t.no_walk.is_empty());
    }

    #[test]
    fn zero_length_edges() {
        // A zero-length cycle and an edge leaving it.
        let e = [(0, 1, 0), (1, 0, 0), (1, 2, 3)];
        let t = check(3, &e);
        assert!(t.walk.contains(&(0, 1, 0)));
        assert!(t.walk.contains(&(0, 2, 3)));
    }

    #[test]
    fn the_benchmark_generator_at_small_n() {
        let t = check(64, &edges(64));
        assert!(!t.walk.is_empty() && !t.no_walk.is_empty());
    }
}
