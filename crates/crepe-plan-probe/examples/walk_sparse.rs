//! D-RU-6 lab probe, arm C: the `graph_walking` program over the same RMAP
//! foci and tiles as arm B (`walk_tile.rs`, #1647), with execution sparsity.
//!
//! STATUS: LAB ONLY. D-RU-6 stays Queued/Lab; nothing here is admitted.
//!
//! ```text
//! cargo run --release --example walk_sparse    # arms A, B and C
//! cargo test --example walk_sparse             # correctness and refusals
//! ```
//!
//! # The three arms
//!
//! * **A**, `walk_mask`: dense bit planes; the hop reads a delta row at a
//!   position found by scanning an adjacency row (a gather). Unchanged.
//! * **B**, `walk_tile`: the same RMAP binding and tiles; the hop is a dense
//!   predicated tile product, a full-tile broadcast, a spread and one mask-risc
//!   `Ternlog` per contributing coordinate. Unchanged.
//! * **C**, this file: the same binding and projection as B; the product skips
//!   work instead of predicating it over the whole tile.
//!
//! # What C changes, against B
//!
//! 1. **Inactive tile products** are skipped: a product runs only for tile
//!    triples whose edge tile and delta tile both exist, and the delta holds
//!    only the bits that are new this iteration (semi-naive).
//! 2. **Inactive inner coordinates**: a row OR runs only where the delta bit
//!    `(y, z)` and the static row-occupancy bit of `A`'s row `y` are both set.
//!    An output tile is allocated only when the first OR into it happens.
//! 3. **Microtiles**: each delta tile carries an occupancy map of `m x m`
//!    blocks (`m` = 16, 32, 64 or 256 in the sweep), so an empty block of the
//!    inner coordinate is skipped with one bit test.
//! 4. **No broadcast or spread** is built, and there is no `Ternlog` per
//!    coordinate. mask-risc runs the per-tile update instead
//!    (`fresh = new & !walk`, `walk |= fresh`), two programs per changed tile.
//! 5. **Prepared metadata**: the row occupancy of each edge tile is built once,
//!    in preparation, and read by every iteration. The binding is never redone.
//!
//! # The address rule, audited at the Rust level
//!
//! Positions read during execution come from loop coordinates; an occupancy
//! bit decides whether work runs and is never used as an index:
//!
//! | Read | Index | From |
//! |---|---|---|
//! | edge tile `(I, K)`, delta tile `(K, J)`, output `(I, J)` | `i * pages + k`, … | the tile-triple loop |
//! | delta occupancy cell | `zb * nb + yb` | the block loops |
//! | delta row `z` | `z * ROW_WORDS + w` | the `z` loop inside the block |
//! | `A`'s row `y`, `A`'s row occupancy | `y = w * 64 + shift + b` | the word, shift and bit loops of the block |
//! | output row `z` | `z * ROW_WORDS + k` | the `z` loop |
//!
//! There is no `trailing_zeros` or bit scan in the execution path: the `b` loop
//! visits every coordinate of a non-empty segment and tests it. Note what this
//! does and does not mean. The set of row ORs C executes is exactly the set a
//! bit scan would find; C differs from arm A in how a position is obtained (a
//! loop coordinate, tested), not in which work is done. Whether that
//! distinction satisfies the intent of ADMISSION B is for the operator to
//! rule; this probe only shows that the rule as written holds.
//!
//! Measured results: `.claude/board/entries/2026-10-21-d-ru-6-sparse-compute.md`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lance_graph_mask_risc::{
    Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal, execute_into, words_for,
};
use mork_pathmap_probe::rmap::{Refusal, SLOTS, TILE_ROWS, Witness};

// Arm B (and, inside it, arm A and the oracles), compiled from its own file.
#[allow(dead_code)]
#[path = "walk_tile.rs"]
pub(crate) mod tile;

use tile::baseline::{Tuples, agree, ascent_tuples, crepe_tuples, edges, mask_walk};
use tile::{
    Bound, Edge, LENS, Projection, Refuse, Run, admit, attest, bind, chain, counts, mib, mixed, ms,
    project, tile_eval, tuples, width,
};

pub(crate) const REPS: usize = 5;
pub(crate) const ORACLE_MAX_NODES: usize = 512;
pub(crate) const ROW_WORDS: usize = SLOTS / 64;
/// The microtile side used by the main comparison, fixed before any run.
pub(crate) const DEFAULT_M: usize = 64;
pub(crate) const MICROTILES: [usize; 4] = [16, 32, 64, 256];

// ---- prepared metadata ---------------------------------------------------------

/// Bit `y` set when row `y` of a tile has any bit.
pub(crate) type RowOcc = [u64; ROW_WORDS];

pub(crate) fn row_occ(t: &[u64]) -> RowOcc {
    let mut o = [0u64; ROW_WORDS];
    for y in 0..SLOTS {
        if t[y * ROW_WORDS..(y + 1) * ROW_WORDS]
            .iter()
            .any(|&w| w != 0)
        {
            o[y >> 6] |= 1 << (y & 63);
        }
    }
    o
}

/// The edge tiles with their row occupancy, built once.
pub(crate) struct Prepared {
    pub(crate) by_len: Vec<(usize, Vec<Option<RowOcc>>)>,
}

pub(crate) fn prepare(p: &Projection) -> Prepared {
    Prepared {
        by_len: p
            .by_len
            .iter()
            .map(|(l, blocks)| {
                (
                    *l,
                    blocks.iter().map(|t| t.as_deref().map(row_occ)).collect(),
                )
            })
            .collect(),
    }
}

/// The `m`-bit segment `[yb * m, (yb + 1) * m)` of a 256-bit row, as
/// `(word, shift, len)` pieces of at most one word each.
pub(crate) fn segments(yb: usize, m: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let lo = yb * m;
    let hi = lo + m;
    (lo / 64..hi.div_ceil(64)).map(move |w| {
        let a = lo.max(w * 64);
        let b = hi.min((w + 1) * 64);
        (w, a - w * 64, b - a)
    })
}

pub(crate) fn low(len: usize) -> u64 {
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

/// Occupancy of the `m x m` blocks of a delta tile: cell `zb * nb + yb` set
/// when rows `zb * m ..` have a bit in columns `yb * m ..`.
pub(crate) fn block_occ(t: &[u64], m: usize) -> Vec<u64> {
    let nb = SLOTS / m;
    let mut occ = vec![0u64; (nb * nb).div_ceil(64)];
    for zb in 0..nb {
        for yb in 0..nb {
            let hit = (zb * m..(zb + 1) * m).any(|z| {
                segments(yb, m).any(|(w, sh, len)| (t[z * ROW_WORDS + w] >> sh) & low(len) != 0)
            });
            if hit {
                let c = zb * nb + yb;
                occ[c >> 6] |= 1 << (c & 63);
            }
        }
    }
    occ
}

// ---- execution -------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counters {
    /// Tile triples whose edge and delta tiles both exist.
    pub(crate) tile_products: u64,
    /// Occupied delta blocks visited.
    pub(crate) blocks: u64,
    /// Row ORs executed (one 256-bit OR each).
    pub(crate) row_ors: u64,
    /// mask-risc calls (the per-tile update).
    pub(crate) calls: u64,
}

/// The update programs and their buffers.
pub(crate) struct Exec {
    pub(crate) fresh_prog: Program,
    pub(crate) or_prog: Program,
    pub(crate) scratch: Scratch<'static>,
    pub(crate) buf: Box<[u64]>,
}

pub(crate) fn update_programs() -> (Program, Program) {
    let keep = Terminal::Keep {
        mask: Operand::Scratch(0),
    };
    (
        Program::new(
            vec![MaskOp::AndNot {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            }],
            keep,
        ),
        Program::new(
            vec![MaskOp::Or {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            }],
            keep,
        ),
    )
}

impl Exec {
    pub(crate) fn with_programs(fresh_prog: Program, or_prog: Program) -> Result<Exec, Refuse> {
        admit(&fresh_prog)?;
        admit(&or_prog)?;
        Ok(Exec {
            fresh_prog,
            or_prog,
            scratch: Scratch::new(words_for(TILE_ROWS), 1),
            buf: tile::tile(),
        })
    }

    pub(crate) fn new() -> Exec {
        let (f, o) = update_programs();
        Exec::with_programs(f, o).expect("the update programs have no gather")
    }

    /// `out = prog(a, b)` over one tile.
    pub(crate) fn run(
        &mut self,
        prog_is_fresh: bool,
        a: &[u64],
        b: &[u64],
        n: &mut Counters,
    ) -> Box<[u64]> {
        let masks: [&[u64]; 2] = [a, b];
        let pl = Planes {
            n_rows: TILE_ROWS,
            masks: &masks,
            lanes: &[],
        };
        let prog = if prog_is_fresh {
            &self.fresh_prog
        } else {
            &self.or_prog
        };
        execute_into(
            prog,
            &pl,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(&mut self.buf),
        )
        .expect("the update program is valid");
        n.calls += 1;
        std::mem::replace(&mut self.buf, tile::tile())
    }
}

/// `c |= a ∘ d` over occupied blocks only. `a` is `A(x, y)` (row `y`), `d` is
/// `D(y, z)` (row `z`), `c` is `C(x, z)` (row `z`). Every index is a loop
/// coordinate; see the module docs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product_into(
    a: &[u64],
    a_rows: &RowOcc,
    d: &[u64],
    d_occ: &[u64],
    m: usize,
    c: &mut Option<Box<[u64]>>,
    n: &mut Counters,
) {
    n.tile_products += 1;
    let nb = SLOTS / m;
    for zb in 0..nb {
        for yb in 0..nb {
            let cell = zb * nb + yb;
            if d_occ[cell >> 6] >> (cell & 63) & 1 == 0 {
                continue;
            }
            n.blocks += 1;
            for z in zb * m..(zb + 1) * m {
                for (w, sh, len) in segments(yb, m) {
                    let live = ((d[z * ROW_WORDS + w] & a_rows[w]) >> sh) & low(len);
                    if live == 0 {
                        continue;
                    }
                    for b in 0..len {
                        if live >> b & 1 == 0 {
                            continue;
                        }
                        let y = w * 64 + sh + b;
                        let out = c.get_or_insert_with(tile::tile);
                        for k in 0..ROW_WORDS {
                            out[z * ROW_WORDS + k] |= a[y * ROW_WORDS + k];
                        }
                        n.row_ors += 1;
                    }
                }
            }
        }
    }
}

pub(crate) struct Sparse {
    pub(crate) run: Run,
    pub(crate) counters: Counters,
}

pub(crate) fn recurse(
    b: &Bound,
    p: &Projection,
    prep: &Prepared,
    current: &Witness,
    m: usize,
    ex: &mut Exec,
) -> Result<Sparse, Refuse> {
    assert!(SLOTS.is_multiple_of(m), "microtile side divides 256");
    b.witness.check(current)?;
    p.witness.check(current)?;
    if !p.escaped.is_empty() {
        return Err(Refusal::IncompleteFocus.into());
    }
    let pages = b.pages();
    let tiles = pages * pages;
    let zero = || vec![vec![None; tiles]; LENS];
    let mut walk: Vec<tile::Blocks> = zero();
    let mut delta: Vec<tile::Blocks> = zero();
    for i in 0..pages {
        let mut t = tile::tile();
        for x in 0..b.page_len(i) {
            tile::set(
                &mut t,
                mork_pathmap_probe::rmap::tile_addr(x as u8, x as u8),
            );
        }
        walk[0][i * pages + i] = Some(t.clone());
        delta[0][i * pages + i] = Some(t);
    }
    let mut d_occ: Vec<Vec<Option<Vec<u64>>>> = vec![vec![None; tiles]; LENS];
    d_occ[0] = delta[0]
        .iter()
        .map(|t| t.as_deref().map(|t| block_occ(t, m)))
        .collect();
    let mut live = vec![false; LENS];
    live[0] = true;
    let mut n = Counters::default();
    let mut iterations = 0usize;
    let bound = LENS * b.nodes() * b.nodes() + 1;
    loop {
        iterations += 1;
        assert!(iterations <= bound, "fixpoint did not terminate");
        let mut new: Vec<tile::Blocks> = zero();
        for (l1, edge) in &prep.by_len {
            let edge_tiles = &p
                .by_len
                .iter()
                .find(|(l, _)| l == l1)
                .expect("same lengths")
                .1;
            for l2 in (0..LENS).filter(|&l2| live[l2]) {
                let l = l1 + l2;
                if l >= LENS {
                    continue;
                }
                for i in 0..pages {
                    for j in 0..pages {
                        for k in 0..pages {
                            let (Some(a), Some(a_rows), Some(dt), Some(occ)) = (
                                &edge_tiles[i * pages + k],
                                &edge[i * pages + k],
                                &delta[l2][k * pages + j],
                                &d_occ[l2][k * pages + j],
                            ) else {
                                continue;
                            };
                            product_into(a, a_rows, dt, occ, m, &mut new[l][i * pages + j], &mut n);
                        }
                    }
                }
            }
        }
        let mut next_live = vec![false; LENS];
        let mut next_occ: Vec<Vec<Option<Vec<u64>>>> = vec![vec![None; tiles]; LENS];
        for l in 0..LENS {
            for t in 0..tiles {
                let Some(nt) = new[l][t].take() else {
                    continue;
                };
                let w = walk[l][t].get_or_insert_with(tile::tile);
                let fresh = ex.run(true, &nt, w, &mut n);
                if fresh.iter().all(|&x| x == 0) {
                    continue;
                }
                *w = ex.run(false, w, &fresh, &mut n);
                next_occ[l][t] = Some(block_occ(&fresh, m));
                new[l][t] = Some(fresh);
                next_live[l] = true;
            }
        }
        if !next_live.iter().any(|&x| x) {
            break;
        }
        delta = new;
        d_occ = next_occ;
        live = next_live;
    }
    Ok(Sparse {
        run: Run {
            walk,
            iterations,
            counters: tile::Counters::default(),
        },
        counters: n,
    })
}

// ---- the whole arm C -----------------------------------------------------------

pub(crate) struct Report {
    pub(crate) attest: Duration,
    pub(crate) bind: Duration,
    pub(crate) project: Duration,
    pub(crate) prepare: Duration,
    pub(crate) recurse: Duration,
    pub(crate) count: Duration,
    pub(crate) decode: Duration,
    pub(crate) iterations: usize,
    pub(crate) counters: Counters,
    pub(crate) pages: usize,
    pub(crate) lookups: u64,
    pub(crate) decodes: u64,
    pub(crate) counts: (usize, usize),
    pub(crate) tuples: Tuples,
}

impl Report {
    pub(crate) fn e2e_count(&self) -> Duration {
        self.attest + self.bind + self.project + self.prepare + self.recurse + self.count
    }
    pub(crate) fn e2e(&self) -> Duration {
        self.attest + self.bind + self.project + self.prepare + self.recurse + self.decode
    }
}

pub(crate) fn sparse_eval(e: &[Edge], m: usize, max_pages: usize) -> Result<Report, Refuse> {
    let s = Instant::now();
    let facts = attest(e);
    let attest_t = s.elapsed();
    let s = Instant::now();
    let b = bind(e, facts, 1, None, max_pages)?;
    let bind_t = s.elapsed();
    let s = Instant::now();
    let p = project(&b, e);
    let project_t = s.elapsed();
    let s = Instant::now();
    let prep = prepare(&p);
    let prepare_t = s.elapsed();
    let mut ex = Exec::new();
    let s = Instant::now();
    let r = recurse(&b, &p, &prep, &b.witness, m, &mut ex)?;
    let recurse_t = s.elapsed();
    let s = Instant::now();
    let c = counts(&b, &r.run);
    let count_t = s.elapsed();
    let s = Instant::now();
    let (t, decodes) = tuples(&b, &r.run);
    let decode_t = s.elapsed();
    Ok(Report {
        attest: attest_t,
        bind: bind_t,
        project: project_t,
        prepare: prepare_t,
        recurse: recurse_t,
        count: count_t,
        decode: decode_t,
        iterations: r.run.iterations,
        counters: r.counters,
        pages: b.pages(),
        lookups: p.lookups,
        decodes,
        counts: c,
        tuples: t,
    })
}

// ---- fixtures ------------------------------------------------------------------

/// splitmix64.
pub(crate) fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `deg` random out-edges per node, lengths 1, 2, 3, 5 or 7.
pub(crate) fn sparse(n: i32, deg: usize, seed: u64) -> Vec<Edge> {
    let mut r = rng(seed);
    let lens = [1, 2, 3, 5, 7];
    let mut e = Vec::new();
    for x in 0..n {
        for _ in 0..deg {
            e.push((x, (r() % n as u64) as i32, lens[(r() % 5) as usize]));
        }
    }
    e
}

/// Communities of 32 nodes, 4 internal edges per node (lengths 1..=3), and
/// one cross edge per community (length 5).
pub(crate) fn clustered(n: i32, seed: u64) -> Vec<Edge> {
    let mut r = rng(seed);
    let mut e = Vec::new();
    for x in 0..n {
        let base = x / 32 * 32;
        for _ in 0..4 {
            let y = base + (r() % 32) as i32;
            e.push((x, y.min(n - 1), 1 + (r() % 3) as u32));
        }
    }
    for c in (0..n).step_by(32) {
        e.push((c, (c + 32 + (r() % 32) as i32) % n, 5));
    }
    e
}

/// Every ordered pair with probability `pct` %, length 1 or 2.
pub(crate) fn dense(n: i32, pct: u64, seed: u64) -> Vec<Edge> {
    let mut r = rng(seed);
    let mut e = Vec::new();
    for x in 0..n {
        for y in 0..n {
            if r() % 100 < pct {
                e.push((x, y, 1 + (r() % 2) as u32));
            }
        }
    }
    e
}

/// A zero-length ring, a length-1 ring and chords of length 7.
pub(crate) fn cyclic(n: i32) -> Vec<Edge> {
    let mut e: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n, 0)).collect();
    e.extend((0..n).map(|i| (i, (i + 2) % n, 1)));
    e.extend((0..n).step_by(17).map(|i| (i, (i + n / 2) % n, 7)));
    e
}

pub(crate) fn fixtures() -> Vec<(&'static str, Vec<Edge>)> {
    vec![
        ("sparse(512)", sparse(512, 1, 1)),
        ("clustered(512)", clustered(512, 2)),
        ("dense(128)", dense(128, 10, 3)),
        ("cyclic(300)", cyclic(300)),
        ("chain(230)", chain(230)),
        ("chain(700)", chain(700)),
        ("mixed(256)", mixed(256, 7)),
        ("edges(512)", edges(512)),
        ("edges(1024)", edges(1024)),
    ]
}

// ---- benchmark -------------------------------------------------------------------

pub(crate) fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

pub(crate) fn status_kib(field: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .expect("Linux /proc")
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .expect("field present")
}

/// Child mode: one arm on one fixture, resident high-water growth in KiB.
pub(crate) fn child(arm: &str, fixture: &str) {
    let (_, e) = fixtures()
        .into_iter()
        .find(|(n, _)| *n == fixture)
        .expect("known fixture");
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
    let before = status_kib("VmRSS:");
    match arm {
        "a-count" => {
            black_box(mask_walk(width(&e), &e).counts());
        }
        "a" => {
            let p = mask_walk(width(&e), &e);
            black_box((p.counts(), p.tuples()));
        }
        "b-count" => {
            let b = bind(&e, attest(&e), 1, None, usize::MAX).expect("binds");
            let p = project(&b, &e);
            let r = tile::recurse(&b, &p, &b.witness, &mut tile::Kernel::new()).expect("answers");
            black_box(counts(&b, &r));
        }
        "b" => {
            black_box(tile_eval(&e, usize::MAX).expect("answers"));
        }
        "c-count" => {
            let b = bind(&e, attest(&e), 1, None, usize::MAX).expect("binds");
            let p = project(&b, &e);
            let prep = prepare(&p);
            let r =
                recurse(&b, &p, &prep, &b.witness, DEFAULT_M, &mut Exec::new()).expect("answers");
            black_box(counts(&b, &r.run));
        }
        "c" => {
            black_box(sparse_eval(&e, DEFAULT_M, usize::MAX).expect("answers"));
        }
        _ => panic!("unknown arm {arm}"),
    }
    println!("{}", status_kib("VmHWM:").saturating_sub(before));
}

pub(crate) fn resident_kib(arm: &str, fixture: &str) -> usize {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--resident") {
        child(&args[2], &args[3]);
        return;
    }
    println!(
        "# A / B / C, count-only and end to end (ms, median of {REPS}; C uses {DEFAULT_M} x {DEFAULT_M} microtiles)\n"
    );
    println!(
        "| fixture | nodes | foci | A count | B count | C count | C/A | C/B | A e2e | B e2e | C e2e | C/A | oracle |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut rows = Vec::new();
    for (name, e) in fixtures() {
        let n = width(&e);
        let mut ac = Vec::new();
        let mut ad = Vec::new();
        let mut row_reads = 0;
        let mut a_tuples = None;
        for _ in 0..REPS {
            let s = Instant::now();
            let p = mask_walk(n, &e);
            black_box(p.counts());
            ac.push(s.elapsed());
            row_reads = p.row_reads;
            let s = Instant::now();
            a_tuples = Some(black_box(p.tuples()));
            ad.push(s.elapsed());
        }
        let a_tuples = a_tuples.expect("REPS > 0");
        let b_reps: Vec<_> = (0..REPS)
            .map(|_| tile_eval(&e, usize::MAX).expect("B answers"))
            .collect();
        let c_reps: Vec<_> = (0..REPS)
            .map(|_| sparse_eval(&e, DEFAULT_M, usize::MAX).expect("C answers"))
            .collect();
        let a_count = median(ac);
        let a_e2e = a_count + median(ad);
        let b_count = median(b_reps.iter().map(|r| r.e2e_count()).collect());
        let b_e2e = median(b_reps.iter().map(|r| r.e2e()).collect());
        let c_count = median(c_reps.iter().map(|r| r.e2e_count()).collect());
        let c_e2e = median(c_reps.iter().map(|r| r.e2e()).collect());
        let phase = |f: fn(&Report) -> Duration| median(c_reps.iter().map(f).collect());
        let phases = [
            phase(|r| r.attest),
            phase(|r| r.bind),
            phase(|r| r.project),
            phase(|r| r.prepare),
            phase(|r| r.recurse),
            phase(|r| r.count),
            phase(|r| r.decode),
        ];
        let b_last = b_reps.into_iter().last().expect("REPS > 0");
        let c_last = c_reps.into_iter().last().expect("REPS > 0");
        let nodes = c_last
            .tuples
            .walk
            .iter()
            .filter(|w| w.0 == w.1 && w.2 == 0)
            .count();
        assert_eq!(c_last.counts, c_last.tuples.counts(), "{name}: C counts");
        agree(&c_last.tuples, &b_last.tuples).unwrap_or_else(|d| panic!("{name}: C vs B {d}"));
        agree(&c_last.tuples, &a_tuples).unwrap_or_else(|d| panic!("{name}: C vs A {d}"));
        let oracle = if nodes <= ORACLE_MAX_NODES {
            agree(&c_last.tuples, &ascent_tuples(&e)).unwrap_or_else(|d| panic!("{name}: {d}"));
            agree(&c_last.tuples, &crepe_tuples(&e)).unwrap_or_else(|d| panic!("{name}: {d}"));
            "A = B = C = ascent = crepe"
        } else {
            "A = B = C; no Datalog oracle (UNCHECKED)"
        };
        let r = |x: Duration, y: Duration| format!("{:.2}×", x.as_secs_f64() / y.as_secs_f64());
        println!(
            "| {name} | {nodes} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {oracle} |",
            c_last.pages,
            ms(a_count),
            ms(b_count),
            ms(c_count),
            r(c_count, a_count),
            r(c_count, b_count),
            ms(a_e2e),
            ms(b_e2e),
            ms(c_e2e),
            r(c_e2e, a_e2e),
        );
        let sweep: Vec<Duration> = MICROTILES
            .iter()
            .map(|&m| {
                median(
                    (0..REPS)
                        .map(|_| sparse_eval(&e, m, usize::MAX).expect("C answers").recurse)
                        .collect(),
                )
            })
            .collect();
        rows.push((name, phases, c_last, b_last.counters, row_reads, sweep));
    }
    println!("\n# arm C by phase (ms) and work counts\n");
    println!(
        "| fixture | attest | bind | project | prepare | recurse | count | decode | iterations | C tile products | C blocks | C row ORs | C mask-risc calls | B tile products | B mask-risc calls |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, t, c, bc, _, _) in &rows {
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            ms(t[0]),
            ms(t[1]),
            ms(t[2]),
            ms(t[3]),
            ms(t[4]),
            ms(t[5]),
            ms(t[6]),
            c.iterations,
            c.counters.tile_products,
            c.counters.blocks,
            c.counters.row_ors,
            c.counters.calls,
            bc.tile_products,
            bc.calls,
        );
    }
    println!("\n# arm C recursion by microtile side (ms)\n");
    println!("| fixture | m=16 | m=32 | m=64 | m=256 |");
    println!("|---|---:|---:|---:|---:|");
    for (name, _, _, _, _, s) in &rows {
        println!(
            "| {name} | {} | {} | {} | {} |",
            ms(s[0]),
            ms(s[1]),
            ms(s[2]),
            ms(s[3])
        );
    }
    println!("\n# resident memory (MiB, VmHWM growth in a fresh process)\n");
    println!("| fixture | A count | B count | C count | A + decode | B + decode | C + decode |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, ..) in &rows {
        let k =
            ["a-count", "b-count", "c-count", "a", "b", "c"].map(|a| resident_kib(a, name) * 1024);
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} |",
            mib(k[0]),
            mib(k[1]),
            mib(k[2]),
            mib(k[3]),
            mib(k[4]),
            mib(k[5])
        );
    }
    println!("\n# value-derived reads by origin\n");
    println!(
        "| fixture | A exec: delta row at a scanned adjacency bit | B and C prep: codebook lookup per endpoint | B exec | C exec | decode: codebook decodes |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, _, c, _, row_reads, _) in &rows {
        println!(
            "| {name} | {row_reads} | {} | 0 | 0 | {} |",
            c.lookups, c.decodes
        );
    }
}

// ---- tests -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use tile::baseline::tests::random_graph;

    /// C's tuples, at every microtile side, equal ascent's, crepe's, A's and
    /// B's. Returns the report at the default side.
    fn check(e: &[Edge]) -> Report {
        let a = ascent_tuples(e);
        let c = crepe_tuples(e);
        let mask = mask_walk(width(e), e).tuples();
        let b = tile_eval(e, usize::MAX).expect("B answers");
        let mut out = None;
        for m in MICROTILES {
            let r = sparse_eval(e, m, usize::MAX).expect("C answers");
            assert_eq!(r.counts, r.tuples.counts(), "m={m}: counts vs tuples");
            agree(&r.tuples, &a).unwrap_or_else(|d| panic!("m={m} ascent: {d}"));
            agree(&r.tuples, &c).unwrap_or_else(|d| panic!("m={m} crepe: {d}"));
            agree(&r.tuples, &mask).unwrap_or_else(|d| panic!("m={m} A: {d}"));
            agree(&r.tuples, &b.tuples).unwrap_or_else(|d| panic!("m={m} B: {d}"));
            if m == DEFAULT_M {
                out = Some(r);
            }
        }
        out.expect("DEFAULT_M is swept")
    }

    #[test]
    fn mixed_lengths_with_cycles() {
        let r = check(&random_graph(65, 2, 0x5EED_0001));
        let lens: BTreeSet<u32> = r.tuples.walk.iter().map(|w| w.2).collect();
        assert_eq!(lens.len(), LENS);
    }

    #[test]
    fn alternative_path_lengths() {
        let r = check(&[(0, 1, 1), (0, 2, 2), (2, 1, 3), (1, 3, 7), (0, 3, 5)]);
        for w in [(0, 1, 1), (0, 1, 5), (0, 3, 8), (0, 3, 12)] {
            assert!(r.tuples.walk.contains(&w), "{w:?}");
        }
    }

    #[test]
    fn zero_length_cycle_and_over_length_edges() {
        let mut e: Vec<Edge> = (0..200).map(|i| (i, (i + 1) % 200, 0)).collect();
        e.push((0, 100, 7));
        e.push((5, 6, 51));
        let r = check(&e);
        assert!(r.iterations >= 150);
        assert!(r.tuples.walk.contains(&(150, 30, 49)));
        assert!(!r.tuples.walk.iter().any(|w| w.2 > 49 && w.2 != 50));
    }

    #[test]
    fn deep_propagation_terminates_across_foci() {
        // 300 nodes: two foci, and the chain crosses the boundary at 255/256.
        let r = check(&chain(300));
        assert!(r.iterations >= 200, "only {} iterations", r.iterations);
        assert_eq!(r.pages, 2);
    }

    #[test]
    fn each_delta_bit_is_expanded_once() {
        // Semi-naive: every Walk fact is new in exactly one iteration, so it
        // feeds at most one row OR per edge length and per source focus. A
        // delta that kept re-derived facts (naive evaluation) stays correct
        // but breaks this bound wherever facts have alternative derivations.
        for e in [
            cyclic(80),
            dense(48, 10, 3),
            random_graph(65, 2, 0x5EED_0001),
        ] {
            let r = check(&e);
            let lens = e.iter().map(|x| x.2).collect::<BTreeSet<_>>().len() as u64;
            let bound = r.tuples.walk.len() as u64 * lens * r.pages as u64;
            assert!(
                r.counters.row_ors <= bound,
                "{} row ORs > {bound}",
                r.counters.row_ors
            );
        }
    }

    #[test]
    fn cross_focus_completeness() {
        for n in [255usize, 256, 257, 300] {
            let mut e = random_graph(n, 1, 0xB0B0 + n as u64);
            e.push((255.min(n as i32 - 1), 0, 2));
            e.push((0, n as i32 - 1, 1));
            let r = check(&e);
            assert_eq!(r.pages, n.div_ceil(SLOTS));
        }
        let r = check(&random_graph(600, 1, 0x3F0C1));
        assert_eq!(r.pages, 3);
    }

    #[test]
    fn microtile_boundaries() {
        // Edges into and out of the first and last coordinate of every 16,
        // 32 and 64 block, inside one focus.
        let ids = [0, 15, 16, 31, 32, 63, 64, 127, 128, 191, 192, 255];
        let mut e = Vec::new();
        for (k, &a) in ids.iter().enumerate() {
            e.push((a, ids[(k + 1) % ids.len()], (k % 3) as u32));
            e.push((a, ids[(k + 5) % ids.len()], 4));
        }
        check(&e);
    }

    #[test]
    fn the_shared_fixtures() {
        check(&clustered(96, 2));
        check(&dense(48, 10, 3));
        check(&cyclic(80));
        check(&edges(64));
    }

    // ---- refusals ----

    #[test]
    fn a_stale_witness_is_refused() {
        let e = edges(64);
        let b = bind(&e, attest(&e), 1, None, usize::MAX).expect("binds");
        let p = project(&b, &e);
        let prep = prepare(&p);
        let mut ex = Exec::new();
        assert!(recurse(&b, &p, &prep, &b.witness, DEFAULT_M, &mut ex).is_ok());
        let w = Witness {
            focus: 2,
            ..b.witness
        };
        assert_eq!(
            recurse(&b, &p, &prep, &w, DEFAULT_M, &mut ex).err(),
            Some(Refuse::Rmap(Refusal::FocusChanged))
        );
        let w = Witness {
            facts: b.witness.facts ^ 1,
            ..b.witness
        };
        assert_eq!(
            recurse(&b, &p, &prep, &w, DEFAULT_M, &mut ex).err(),
            Some(Refuse::Rmap(Refusal::FactsChanged))
        );
    }

    #[test]
    fn an_incomplete_scope_and_too_many_foci_are_refused() {
        let e = random_graph(120, 2, 9);
        let scope: Vec<i32> = (0..100).chain([9_000]).collect();
        let b = bind(&e, attest(&e), 1, Some(&scope), usize::MAX).expect("binds");
        let p = project(&b, &e);
        assert!(!p.escaped.is_empty());
        assert_eq!(
            recurse(
                &b,
                &p,
                &prepare(&p),
                &b.witness,
                DEFAULT_M,
                &mut Exec::new()
            )
            .err(),
            Some(Refuse::Rmap(Refusal::IncompleteFocus))
        );
        assert_eq!(
            sparse_eval(&random_graph(300, 1, 3), DEFAULT_M, 1).err(),
            Some(Refuse::Rmap(Refusal::IncompleteFocus))
        );
    }

    #[test]
    fn a_scope_with_extra_ids_keeps_exact_parity() {
        let e: Vec<Edge> = vec![(0, 1, 1), (1, 2, 0), (2, 0, 3), (2, 3, 2)];
        let scope = [0, 1, 2, 3, 77, 900];
        let b = bind(&e, attest(&e), 1, Some(&scope), usize::MAX).expect("binds");
        assert_eq!(b.nodes(), 4);
        let p = project(&b, &e);
        let r = recurse(
            &b,
            &p,
            &prepare(&p),
            &b.witness,
            DEFAULT_M,
            &mut Exec::new(),
        )
        .expect("answers");
        let (t, _) = tuples(&b, &r.run);
        agree(&t, &ascent_tuples(&e)).unwrap_or_else(|d| panic!("{d}"));
    }

    #[test]
    fn a_wrong_path_length_is_detected() {
        let e = random_graph(40, 2, 11);
        let r = check(&e);
        let mut bad = Tuples {
            walk: r.tuples.walk.clone(),
            no_walk: r.tuples.no_walk.clone(),
        };
        let w = *bad
            .walk
            .iter()
            .find(|w| w.2 > 0 && w.2 < 50 && !bad.walk.contains(&(w.0, w.1, w.2 + 1)))
            .expect("a shiftable tuple");
        bad.walk.remove(&w);
        bad.walk.insert((w.0, w.1, w.2 + 1));
        assert_eq!(bad.counts(), r.tuples.counts());
        assert!(agree(&bad, &ascent_tuples(&e)).is_err());
    }

    #[test]
    fn a_gather_is_refused_before_execution() {
        let gather = Program::new(
            vec![MaskOp::Gather {
                lane: 0,
                foreign: 0,
                under: None,
                dst: 0,
            }],
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        let (f, o) = update_programs();
        assert_eq!(
            Exec::with_programs(gather.clone(), o).err(),
            Some(Refuse::UnauthorizedGather)
        );
        assert_eq!(
            Exec::with_programs(f, gather).err(),
            Some(Refuse::UnauthorizedGather)
        );
    }

    #[test]
    fn segments_cover_each_block_exactly() {
        for m in MICROTILES {
            for yb in 0..SLOTS / m {
                let mut bits = 0usize;
                for (w, sh, len) in segments(yb, m) {
                    assert!(sh + len <= 64);
                    assert_eq!(w * 64 + sh, yb * m + bits);
                    bits += len;
                }
                assert_eq!(bits, m);
            }
        }
    }
}
