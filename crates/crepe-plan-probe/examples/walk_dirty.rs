//! D-RU-6 lab probe, arm A′: `walk_mask` with two frontier optimizations,
//! each switchable on its own, measured against arms A, B and C.
//!
//! STATUS: LAB ONLY. A′ is a gather path exactly like A (it reads a delta row
//! at a position taken from an adjacency bit), so it is not an ADMISSION-B
//! candidate. It exists to find out what causes arm C's win on long chains.
//!
//! ```text
//! cargo run --release --example walk_dirty    # arms A, A′ (three variants), B and C
//! cargo test --example walk_dirty             # correctness and the two switches
//! ```
//!
//! # The two switches
//!
//! * **`skip_clean`.** Each delta plane carries a bitmap of its non-zero rows.
//!   The hop masks each adjacency word with that bitmap before scanning it,
//!   so a set bit `y` whose delta row `y` did not change this iteration costs
//!   nothing. A already skips whole lengths with nothing new (`live`); this
//!   skips single rows.
//! * **`touched_only`.** A clears the whole `new` buffer each iteration and
//!   runs `andnot`, `any` and `or` over the full `n x w` span of every live
//!   length. With this switch only the rows the hop wrote are updated, and
//!   only the rows that held a delta are cleared again.
//!
//! With both switches off, A′ runs the same loop as A; a test asserts the
//! same tuple sets and the same number of value-addressed row reads.
//!
//! Counters: value-addressed row reads (the gathers), adjacency bits masked
//! out by `skip_clean`, and words swept by `fill`, `andnot`, `any`, `or` and
//! the row clears (the cost `touched_only` removes).
//!
//! Measured results: `.claude/board/entries/2026-10-21-d-ru-6-dirty-frontier.md`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use ndarray::simd::{mask_andnot_assign, mask_any, mask_or_assign};

// Arm C, which compiles arm B, which compiles arm A and the oracles.
#[allow(dead_code)]
#[path = "walk_sparse.rs"]
mod sparse;

use sparse::tile::baseline::{
    MAX_PATH_LEN, Planes, agree, ascent_tuples, bit, crepe_tuples, mask_walk, set, words,
};
use sparse::tile::{Edge, mib, ms, tile_eval, width};
use sparse::{DEFAULT_M, fixtures, sparse_eval};

const REPS: usize = 5;
const ORACLE_MAX_NODES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Opt {
    skip_clean: bool,
    touched_only: bool,
}

const VARIANTS: [(&str, Opt); 4] = [
    (
        "A′ off",
        Opt {
            skip_clean: false,
            touched_only: false,
        },
    ),
    (
        "A′ skip",
        Opt {
            skip_clean: true,
            touched_only: false,
        },
    ),
    (
        "A′ touched",
        Opt {
            skip_clean: false,
            touched_only: true,
        },
    ),
    (
        "A′ both",
        Opt {
            skip_clean: true,
            touched_only: true,
        },
    ),
];

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    iterations: usize,
    /// Value-addressed row reads (`delta[row(l2, y)]` at a scanned bit `y`).
    row_reads: u64,
    /// Adjacency bits whose delta row was clean, skipped by `skip_clean`.
    masked_out: u64,
    /// Words processed by whole-buffer and per-row sweeps.
    sweep_words: u64,
    /// Adjacency words scanned by the hop (every row `x`, both switches).
    adj_words: u64,
}

struct Dirty {
    planes: Planes,
    counters: Counters,
}

/// The program over bit planes, with the switches in `opt`.
fn dirty_walk(n: usize, e: &[Edge], opt: Opt) -> Dirty {
    let w = words(n);
    let lens = MAX_PATH_LEN as usize + 1;
    let row = |plane: usize, x: usize| (plane * n + x) * w;

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

    let mut walk = vec![0u64; lens * n * w];
    let mut delta = vec![0u64; lens * n * w];
    let mut new = vec![0u64; lens * n * w];
    let mut live = vec![false; lens];
    // Per length: the bitmap of non-zero delta rows, and the same rows as a
    // list (to clear them later without a sweep).
    let mut d_dirty = vec![0u64; lens * w];
    let mut d_rows: Vec<Vec<usize>> = vec![Vec::new(); lens];
    // Per length: rows of `new` the hop wrote, as a bitmap and a list.
    let mut n_mark = vec![0u64; lens * w];
    let mut n_rows: Vec<Vec<usize>> = vec![Vec::new(); lens];
    for x in (0..n).filter(|&x| bit(&node, x)) {
        set(&mut walk[row(0, x)..row(0, x) + w], x);
        set(&mut delta[row(0, x)..row(0, x) + w], x);
        set(&mut d_dirty[0..w], x);
        d_rows[0].push(x);
    }
    live[0] = true;

    let mut c = Counters::default();
    let bound = lens * n * n + 1;
    loop {
        c.iterations += 1;
        assert!(c.iterations <= bound, "fixpoint did not terminate");
        if !opt.touched_only {
            new.fill(0);
            c.sweep_words += new.len() as u64;
        }
        // What `touched_only` relies on instead of the fill: the rows the last
        // iteration left in `new` were cleared row by row. A stale bit would
        // not change the answer (it is already in `walk`, and `new &= !walk`
        // removes it), so only this check can see a missing clear.
        #[cfg(test)]
        assert!(new.iter().all(|&x| x == 0), "`new` is not clear");
        let mut new_live = vec![false; lens];
        for (l1, adj) in by_len.iter().enumerate() {
            let Some(adj) = adj else { continue };
            for l2 in (0..lens).filter(|&l2| live[l2]) {
                let l = l1 + l2;
                if l >= lens {
                    continue;
                }
                for x in 0..n {
                    let dst = row(l, x);
                    c.adj_words += w as u64;
                    for wi in 0..w {
                        let mut bits = adj[x * w + wi];
                        if opt.skip_clean {
                            let clean = bits & !d_dirty[l2 * w + wi];
                            c.masked_out += u64::from(clean.count_ones());
                            bits &= !clean;
                        }
                        while bits != 0 {
                            let y = wi * 64 + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            // The gather: a row address taken from a value.
                            let src = row(l2, y);
                            c.row_reads += 1;
                            mask_or_assign(&mut new[dst..dst + w], &delta[src..src + w]);
                            if opt.touched_only && !bit(&n_mark[l * w..l * w + w], x) {
                                set(&mut n_mark[l * w..l * w + w], x);
                                n_rows[l].push(x);
                            }
                        }
                    }
                }
                new_live[l] = true;
            }
        }

        // new &= !walk; walk |= new; and the next delta's dirty rows.
        let mut any = false;
        let mut next_dirty = vec![0u64; lens * w];
        let mut next_rows: Vec<Vec<usize>> = vec![Vec::new(); lens];
        for (l, is_live) in new_live.iter_mut().enumerate() {
            if !*is_live {
                continue;
            }
            let mut fresh = false;
            if opt.touched_only {
                for &x in &n_rows[l] {
                    let r = row(l, x);
                    mask_andnot_assign(&mut new[r..r + w], &walk[r..r + w]);
                    c.sweep_words += 2 * w as u64;
                    if mask_any(&new[r..r + w]) {
                        mask_or_assign(&mut walk[r..r + w], &new[r..r + w]);
                        c.sweep_words += w as u64;
                        set(&mut next_dirty[l * w..l * w + w], x);
                        next_rows[l].push(x);
                        fresh = true;
                    }
                }
            } else {
                let span = row(l, 0)..row(l, 0) + n * w;
                mask_andnot_assign(&mut new[span.clone()], &walk[span.clone()]);
                c.sweep_words += 2 * (n * w) as u64;
                if mask_any(&new[span.clone()]) {
                    mask_or_assign(&mut walk[span.clone()], &new[span.clone()]);
                    c.sweep_words += (n * w) as u64;
                    fresh = true;
                    if opt.skip_clean {
                        // Without a written-row list, the dirty rows need a scan.
                        for x in 0..n {
                            let r = row(l, x);
                            if mask_any(&new[r..r + w]) {
                                set(&mut next_dirty[l * w..l * w + w], x);
                            }
                        }
                        c.sweep_words += (n * w) as u64;
                    }
                }
            }
            if fresh {
                any = true;
            } else {
                *is_live = false;
            }
        }
        if opt.touched_only {
            for (l, rows) in n_rows.iter_mut().enumerate() {
                for &x in rows.iter() {
                    n_mark[l * w + x / 64] &= !(1 << (x % 64));
                }
                rows.clear();
            }
        }
        if !any {
            break;
        }
        std::mem::swap(&mut delta, &mut new);
        if opt.touched_only {
            // `new` now holds the old delta: clear only its non-zero rows.
            for (l, rows) in d_rows.iter().enumerate() {
                for &x in rows {
                    let r = row(l, x);
                    new[r..r + w].fill(0);
                    c.sweep_words += w as u64;
                }
            }
            // Written rows with nothing fresh are zero already (new &= !walk
            // left nothing), so the next delta's rows are exactly next_rows.
        }
        d_dirty = next_dirty;
        d_rows = next_rows;
        live = new_live;
    }

    Dirty {
        planes: Planes {
            n,
            w,
            lens,
            node,
            walk,
            row_reads: c.row_reads,
        },
        counters: c,
    }
}

// ---- benchmark -------------------------------------------------------------------

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn timed<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut t = Vec::with_capacity(REPS);
    let mut out = None;
    for _ in 0..REPS {
        let s = Instant::now();
        out = Some(black_box(f()));
        t.push(s.elapsed());
    }
    (median(t), out.expect("REPS > 0"))
}

fn status_kib(field: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .expect("Linux /proc")
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .expect("field present")
}

/// Child mode: one count-only run, resident high-water growth in KiB.
fn child(arm: &str, fixture: &str) {
    let (_, e) = fixtures()
        .into_iter()
        .find(|(n, _)| *n == fixture)
        .expect("known fixture");
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
    let before = status_kib("VmRSS:");
    let opt = VARIANTS
        .iter()
        .find(|(n, _)| *n == arm)
        .map(|(_, o)| *o)
        .expect("known variant");
    black_box(dirty_walk(width(&e), &e, opt).planes.counts());
    println!("{}", status_kib("VmHWM:").saturating_sub(before));
}

fn resident_kib(arm: &str, fixture: &str) -> usize {
    let out = std::process::Command::new(std::env::current_exe().expect("exe"))
        .args(["--resident", arm, fixture])
        .output()
        .expect("child runs");
    assert!(out.status.success(), "child failed: {out:?}");
    String::from_utf8(out.stdout)
        .expect("utf8")
        .trim()
        .parse()
        .expect("a number")
}

fn ratio(x: Duration, y: Duration) -> String {
    format!("{:.2}×", x.as_secs_f64() / y.as_secs_f64())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--resident") {
        child(&args[2], &args[3]);
        return;
    }
    println!("# count-only (ms, median of {REPS})\n");
    println!(
        "| fixture | nodes | A | A′ off | A′ skip | A′ touched | A′ both | B | C | C/A′both | A′both/A | oracle |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut work = Vec::new();
    let mut e2e = Vec::new();
    for (name, e) in fixtures() {
        let n = width(&e);
        let (t_a, a) = timed(|| mask_walk(n, &e));
        let a_counts = a.counts();
        let mut tv = Vec::new();
        let mut cv = Vec::new();
        for (_, opt) in VARIANTS {
            let (t, d) = timed(|| {
                let d = dirty_walk(n, &e, opt);
                black_box(d.planes.counts());
                d
            });
            assert_eq!(d.planes.counts(), a_counts, "{name} {opt:?}: counts");
            tv.push(t);
            cv.push(d);
        }
        // Count-only for A is its fixpoint plus the popcounts.
        let (t_a_count, _) = timed(|| mask_walk(n, &e).counts());
        let _ = t_a;
        let b = (0..REPS)
            .map(|_| tile_eval(&e, usize::MAX).expect("B answers"))
            .collect::<Vec<_>>();
        let c = (0..REPS)
            .map(|_| sparse_eval(&e, DEFAULT_M, usize::MAX).expect("C answers"))
            .collect::<Vec<_>>();
        let t_b = median(b.iter().map(|r| r.e2e_count()).collect());
        let t_c = median(c.iter().map(|r| r.e2e_count()).collect());
        let t_b_e2e = median(b.iter().map(|r| r.e2e()).collect());
        let t_c_e2e = median(c.iter().map(|r| r.e2e()).collect());

        // Tuples: every A′ variant against A, and the oracles where they run.
        let a_tuples = a.tuples();
        for (d, (vn, _)) in cv.iter().zip(VARIANTS) {
            agree(&d.planes.tuples(), &a_tuples).unwrap_or_else(|x| panic!("{name} {vn}: {x}"));
        }
        let c_last = c.into_iter().last().expect("REPS > 0");
        agree(&c_last.tuples, &a_tuples).unwrap_or_else(|x| panic!("{name} C: {x}"));
        let nodes = a_tuples
            .walk
            .iter()
            .filter(|w| w.0 == w.1 && w.2 == 0)
            .count();
        let oracle = if nodes <= ORACLE_MAX_NODES {
            agree(&a_tuples, &ascent_tuples(&e)).unwrap_or_else(|x| panic!("{name}: {x}"));
            agree(&a_tuples, &crepe_tuples(&e)).unwrap_or_else(|x| panic!("{name}: {x}"));
            "all = ascent = crepe"
        } else {
            "all equal; no Datalog oracle (UNCHECKED)"
        };
        println!(
            "| {name} | {nodes} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {oracle} |",
            ms(t_a_count),
            ms(tv[0]),
            ms(tv[1]),
            ms(tv[2]),
            ms(tv[3]),
            ms(t_b),
            ms(t_c),
            ratio(t_c, tv[3]),
            ratio(tv[3], t_a_count),
        );
        // End to end: the fixpoint and counts plus decoding the tuples.
        let both = &cv[3];
        let (t_dec, _) = timed(|| both.planes.tuples());
        let (t_a_dec, _) = timed(|| a.tuples());
        e2e.push((name, t_a_count + t_a_dec, tv[3] + t_dec, t_b_e2e, t_c_e2e));
        work.push((
            name,
            cv.iter().map(|d| d.counters).collect::<Vec<_>>(),
            a.row_reads,
        ));
    }
    println!("\n# end to end, tuples decoded (ms)\n");
    println!("| fixture | A | A′ both | B | C | C/A′both | C/A |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, a, d, b, c) in &e2e {
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} |",
            ms(*a),
            ms(*d),
            ms(*b),
            ms(*c),
            ratio(*c, *d),
            ratio(*c, *a)
        );
    }
    println!("\n# work counts per A′ variant (off / skip / touched / both)\n");
    println!(
        "| fixture | iterations | A row reads | row reads | bits masked out | words swept | adjacency words scanned |"
    );
    println!("|---|---:|---:|---|---|---|---:|");
    for (name, cs, a_reads) in &work {
        let col = |f: fn(&Counters) -> u64| {
            cs.iter()
                .map(|c| f(c).to_string())
                .collect::<Vec<_>>()
                .join(" / ")
        };
        println!(
            "| {name} | {} | {a_reads} | {} | {} | {} | {} |",
            cs[0].iterations,
            col(|c| c.row_reads),
            col(|c| c.masked_out),
            col(|c| c.sweep_words),
            cs[3].adj_words,
        );
    }
    println!("\n# resident memory, count-only (MiB, VmHWM growth in a fresh process)\n");
    println!("| fixture | A′ off | A′ both |");
    println!("|---|---:|---:|");
    for (name, ..) in &work {
        println!(
            "| {name} | {} | {} |",
            mib(resident_kib("A′ off", name) * 1024),
            mib(resident_kib("A′ both", name) * 1024)
        );
    }
}

// ---- tests -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::tile::baseline::tests::random_graph;
    use sparse::tile::chain;
    use sparse::{clustered, cyclic, dense};

    /// Every variant's tuples equal ascent's, crepe's and A's.
    fn check(e: &[Edge]) -> Vec<Counters> {
        let n = width(e);
        let a = mask_walk(n, e);
        let at = a.tuples();
        agree(&at, &ascent_tuples(e)).unwrap_or_else(|d| panic!("A ascent: {d}"));
        agree(&at, &crepe_tuples(e)).unwrap_or_else(|d| panic!("A crepe: {d}"));
        let mut out = Vec::new();
        for (name, opt) in VARIANTS {
            let d = dirty_walk(n, e, opt);
            assert_eq!(d.planes.counts(), a.counts(), "{name}: counts");
            agree(&d.planes.tuples(), &at).unwrap_or_else(|x| panic!("{name}: {x}"));
            out.push(d.counters);
        }
        // Off is A's loop: the same gathers, one by one.
        assert_eq!(out[0].row_reads, a.row_reads, "A′ off is not A's loop");
        out
    }

    #[test]
    fn mixed_lengths_with_cycles() {
        check(&random_graph(65, 2, 0x5EED_0001));
    }

    #[test]
    fn alternative_path_lengths() {
        check(&[(0, 1, 1), (0, 2, 2), (2, 1, 3), (1, 3, 7), (0, 3, 5)]);
    }

    #[test]
    fn zero_length_cycle_and_over_length_edges() {
        let mut e: Vec<Edge> = (0..200).map(|i| (i, (i + 1) % 200, 0)).collect();
        e.push((0, 100, 7));
        e.push((5, 6, 51));
        let c = check(&e);
        assert!(c[3].iterations >= 150);
    }

    #[test]
    fn deep_propagation_and_word_boundaries() {
        // 300 nodes: rows span five words, ids cross 64-bit boundaries.
        let c = check(&chain(300));
        assert!(c[3].iterations >= 200);
    }

    #[test]
    fn the_shared_fixtures() {
        check(&clustered(96, 2));
        check(&dense(48, 10, 3));
        check(&cyclic(80));
    }

    #[test]
    fn skip_clean_skips_and_touched_only_sweeps_less() {
        // On a chain the frontier is one row per length: the switches must
        // each do what they claim, and neither may change the gathers that
        // matter.
        let c = check(&chain(230));
        let [off, skip, touched, both] = [c[0], c[1], c[2], c[3]];
        assert_eq!(off.masked_out, 0);
        assert!(skip.masked_out > 0, "skip_clean masked nothing");
        assert_eq!(skip.row_reads + skip.masked_out, off.row_reads);
        assert_eq!(touched.row_reads, off.row_reads);
        assert!(
            touched.sweep_words * 10 < off.sweep_words,
            "touched_only swept {} of {} words",
            touched.sweep_words,
            off.sweep_words
        );
        assert_eq!(both.row_reads, skip.row_reads);
        assert!(both.sweep_words <= touched.sweep_words);
    }

    #[test]
    fn a_dense_frontier_gives_skip_clean_nothing_to_skip() {
        // Silence twin: when every delta row is dirty, nothing is masked.
        let e: Vec<Edge> = (0..40)
            .flat_map(|x| (0..40).map(move |y| (x, y, 1)))
            .collect();
        let c = check(&e);
        assert_eq!(c[1].masked_out, 0, "masked bits on an all-dirty frontier");
    }
}
