//! crepe's `graph_walking` benchmark program, evaluated twice: once by the
//! code `crepe!` generates (hash sets and hash-map indexes, tuple at a time)
//! and once over dense bit planes with `ndarray::simd` mask ops. Both runs
//! must give the same tuple counts. Run in release from this crate's
//! directory:
//!
//! ```text
//! cargo run --release --example walk_mask
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

use std::hint::black_box;
use std::time::{Duration, Instant};

use crepe::crepe;
use ndarray::simd::{mask_andnot_assign, mask_any, mask_or_assign, popcount_batch_u64};

const MAX_PATH_LEN: u32 = 50;
const REPS: usize = 9;

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

fn edges(n: i32) -> Vec<(i32, i32, u32)> {
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

fn crepe_walk(e: &[(i32, i32, u32)]) -> (usize, usize) {
    let mut rt = Crepe::new();
    rt.extend(e.iter().map(|&(a, b, w)| Edge(a, b, w)));
    let (walk, nowalk) = rt.run();
    (walk.len(), nowalk.len())
}

/// Bits per row of an `n`-wide plane.
fn words(n: usize) -> usize {
    n.div_ceil(64)
}

fn set(row: &mut [u64], i: usize) {
    row[i / 64] |= 1 << (i % 64);
}

/// The same program over bit planes. Node ids must be dense in `0..n`.
fn mask_walk(n: usize, e: &[(i32, i32, u32)]) -> (usize, usize) {
    let w = words(n);
    let lens = MAX_PATH_LEN as usize + 1;
    let row = |plane: usize, x: usize| (plane * n + x) * w;

    // Edges grouped by length: adjacency rows `x -> {y}`.
    let mut by_len: Vec<Option<Vec<u64>>> = vec![None; lens];
    let mut node = vec![0u64; w];
    for &(a, b, l) in e {
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
    for x in 0..n {
        if node[x / 64] >> (x % 64) & 1 == 1 {
            set(&mut walk[row(0, x)..row(0, x) + w], x);
            set(&mut delta[row(0, x)..row(0, x) + w], x);
        }
    }
    live[0] = true;

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
                            let src = row(l2, y);
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

    let walk_count = popcount_batch_u64(&walk) as usize;
    // NoWalk: for each node x, nodes not reached at any length.
    let mut reach = vec![0u64; w];
    let mut nowalk = 0usize;
    for x in 0..n {
        if node[x / 64] >> (x % 64) & 1 == 0 {
            continue;
        }
        reach.fill(0);
        for l in 0..lens {
            mask_or_assign(&mut reach, &walk[row(l, x)..row(l, x) + w]);
        }
        let mut miss = node.clone();
        mask_andnot_assign(&mut miss, &reach);
        nowalk += popcount_batch_u64(&miss) as usize;
    }
    (walk_count, nowalk)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time(mut f: impl FnMut() -> (usize, usize)) -> (Duration, (usize, usize)) {
    let mut t = Vec::with_capacity(REPS);
    let mut out = (0, 0);
    for _ in 0..REPS {
        let s = Instant::now();
        out = black_box(f());
        t.push(s.elapsed());
    }
    (median(t), out)
}

fn main() {
    println!(
        "{:>5} | {:>10} {:>10} | {:>9} | {:>9} {:>9}",
        "n", "crepe ms", "mask ms", "speed-up", "Walk", "NoWalk"
    );
    for n in [128usize, 256, 512, 1024] {
        let e = edges(n as i32);
        let (tm, mask) = time(|| mask_walk(n, &e));
        let (tc, crep) = if n <= 512 {
            time(|| crepe_walk(&e))
        } else {
            (Duration::ZERO, mask)
        };
        assert_eq!(mask, crep, "n={n}: mask and crepe disagree");
        let speed = if tc.is_zero() {
            "-".to_string()
        } else {
            format!("{:.1}x", tc.as_secs_f64() / tm.as_secs_f64())
        };
        println!(
            "{:>5} | {:>10.3} {:>10.3} | {:>9} | {:>9} {:>9}",
            n,
            tc.as_secs_f64() * 1e3,
            tm.as_secs_f64() * 1e3,
            speed,
            mask.0,
            mask.1
        );
    }
}
