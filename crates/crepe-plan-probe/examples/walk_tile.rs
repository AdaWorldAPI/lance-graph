//! D-RU-6 lab probe: crepe's `graph_walking` program over co-addressed RMAP
//! Boolean tiles, measured against `walk_mask` (the baseline, #1637, #1642).
//!
//! STATUS: LAB ONLY. Nothing here is a production path, and passing these
//! tests is not an admission. D-RU-6 stays Queued/Lab
//! (`.claude/plans/d-relalg-unified-v1.md` §8, §10, §14 U6).
//!
//! ```text
//! cargo run --release --example walk_tile    # the comparison
//! cargo test --example walk_tile             # correctness and refusals
//! ```
//!
//! # The program
//!
//! Same as `walk_mask` and crepe's benchmark: `Walk(x, z, len)` for
//! `len in 0..=MAX_PATH_LEN`, `NoWalk(x, y)` over nodes, set semantics, least
//! fixpoint. Several lengths per pair, cycles, zero-length edges, ids that are
//! not nodes and edges longer than the bound are all kept.
//!
//! # The two executors
//!
//! * **walk_mask** (the baseline, unchanged). One `n x n` bit plane per
//!   length. The hop iterates the set bits `y` of an adjacency row and reads
//!   `delta[row(l2, y)]`: a row address taken from a value (a gather).
//! * **walk_tile** (this file). Nodes are bound to 256-slot RMAP foci
//!   ([`Focus`]); every relation is a block matrix of 256 x 256 tiles over the
//!   `u8:u8` rail ([`tile_addr`]). The hop is a Boolean tile product
//!   `C = A ∘ B` evaluated in predicated form: for every coordinate `y`,
//!   `C |= broadcast(row y of A) & spread(bit y of every row of B)`, one
//!   mask-risc `Ternlog` program per `y`. Every address is a loop coordinate
//!   (`y`, a row `z`, a tile `(I, J)`); a value only decides what is written,
//!   never where something is read.
//!
//! # Phases, measured separately
//!
//! 1. attestation (a digest of the canonical edge list) and witness
//!    validation;
//! 2. identity resolution and focus binding (the frozen codebook and one
//!    [`Focus`] per 256 nodes);
//! 3. projection (edge tiles per length);
//! 4. recursion (the semi-naive fixpoint over tiles);
//! 5. tuple decoding, and the count-only alternative;
//! 6. end to end, peak heap, and value-derived reads by origin.
//!
//! # ADMISSION B: what this probe shows and what it does not
//!
//! The rule (operator, 2026-10-20, `entries/2026-10-20-six-decisions.md` §1):
//! preparation may do counted, version-witnessed same-table seeks over an
//! attested sorted key and build a projection for co-addressed evaluation;
//! frozen codebook decoding is representation translation if its mapping is
//! immutable within the attested version, its cost is accounted for, and it
//! does not become a relational lookup. Execution stays co-addressed.
//!
//! | Step | Reads | Classification here |
//! |---|---|---|
//! | attestation | sequential scan of the edge table | same-table, counted |
//! | codebook build | sequential scan of both endpoint columns, sort | same-table projection |
//! | slot resolution | one codebook lookup per edge endpoint, keyed by the edge's own value | frozen-codebook translation; probe witness, NOT the contract's attestation |
//! | edge tiles | writes at the resolved slots | projection of the same table |
//! | recursion | loop coordinates only; mask-risc programs admitted without `Gather` | co-addressed (demonstrated) |
//! | decoding | one codebook decode per output identity | frozen-codebook decoding |
//!
//! The execution side is gather-free, and a refusal guard enforces it. The
//! preparation side has the SHAPE ADMISSION B admits, but its witness is a
//! probe-local digest ([`Witness`] from `rmap`), not the attested sorted key
//! the ruling names (`OrderedLaneWitness`). So the binding path is NOT
//! demonstrated to satisfy ADMISSION B, and D-RU-6 does not move.
//!
//! Measured results and the disable runs: `.claude/board/entries/2026-10-21-d-ru-6-tile-walk-probe.md`.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use lance_graph_mask_risc::{
    Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal, execute_into, words_for,
};
use mork_pathmap_probe::rmap::{Focus, Refusal, SLOTS, TILE_ROWS, TILE_WORDS, Witness, tile_addr};

// The baseline, compiled from its own file so the comparison runs the merged
// code and its oracles, not a copy. Its `main` and the helpers this example
// does not call are dead code here.
#[allow(dead_code)]
#[path = "walk_mask.rs"]
mod baseline;

use baseline::{MAX_PATH_LEN, Tuples, agree, ascent_tuples, crepe_tuples, edges, mask_walk};

const LENS: usize = MAX_PATH_LEN as usize + 1;
const REPS: usize = 7;
const ORACLE_MAX_NODES: usize = 512;
/// Words per tile row (256 bits).
const ROW_WORDS: usize = SLOTS / 64;

type Edge = (i32, i32, u32);

// ---- refusals ----------------------------------------------------------------

/// Why the tile path refuses to answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refuse {
    /// A witness check failed, or the answer would depend on identities
    /// outside the bound foci.
    Rmap(Refusal),
    /// A program reads at a position taken from a value.
    UnauthorizedGather,
}

impl From<Refusal> for Refuse {
    fn from(r: Refusal) -> Self {
        Refuse::Rmap(r)
    }
}

/// Admit a program to the execution phase: no op may read at a position
/// taken from a lane's value. `MaskOp::Gather` is the only such op in
/// mask-risc's IR, and `Pred` reads lanes at the row's own position.
fn admit(p: &Program) -> Result<(), Refuse> {
    if p.ops.iter().any(|op| matches!(op, MaskOp::Gather { .. })) {
        return Err(Refuse::UnauthorizedGather);
    }
    Ok(())
}

// ---- phase 1: attestation ----------------------------------------------------

/// FNV-1a over a byte stream.
fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The facts digest: the edge set in canonical order. A probe-local stand-in
/// for an attested sorted key (see the module docs).
fn attest(e: &[Edge]) -> u64 {
    let mut s = e.to_vec();
    s.sort_unstable();
    s.dedup();
    fnv(s.iter().flat_map(|&(a, b, l)| {
        a.to_le_bytes()
            .into_iter()
            .chain(b.to_le_bytes())
            .chain(l.to_le_bytes())
    }))
}

// ---- phase 2: identity resolution and focus binding ---------------------------

/// The frozen codebook (sorted node ids) and one [`Focus`] per 256 of them.
struct Bound {
    witness: Witness,
    codebook: Vec<i32>,
    foci: Vec<Focus>,
}

impl Bound {
    fn pages(&self) -> usize {
        self.foci.len()
    }

    fn nodes(&self) -> usize {
        self.codebook.len()
    }

    fn page_len(&self, p: usize) -> usize {
        self.foci[p].len()
    }

    /// Global id of slot `s` of page `p`: a codebook decode.
    fn id(&self, p: usize, s: usize) -> i32 {
        self.foci[p].id(s as u8) as i32
    }

    /// `(page, slot)` of a node id: a codebook lookup keyed by a value. `None`
    /// when the id is not bound.
    fn resolve(&self, id: i32) -> Option<(usize, u8)> {
        let p = self
            .foci
            .partition_point(|f| f.id(0) as i32 <= id)
            .checked_sub(1)?;
        self.foci[p].slot_of(id as u32).map(|s| (p, s))
    }
}

/// Bind every endpoint of `e` (all nodes) or, with `scope`, only the given
/// ids. Refuses more than `max_pages` foci: a global answer is never
/// truncated to the foci that fit.
fn bind(
    e: &[Edge],
    facts: u64,
    generation: u64,
    scope: Option<&[i32]>,
    max_pages: usize,
) -> Result<Bound, Refuse> {
    let mut codebook: Vec<i32> = match scope {
        Some(s) => s.to_vec(),
        None => e.iter().flat_map(|&(a, b, _)| [a, b]).collect(),
    };
    codebook.sort_unstable();
    codebook.dedup();
    assert!(
        codebook.first().is_none_or(|&f| f >= 0),
        "ids must be non-negative"
    );
    if codebook.len().div_ceil(SLOTS) > max_pages {
        return Err(Refusal::IncompleteFocus.into());
    }
    let foci: Vec<Focus> = codebook
        .chunks(SLOTS)
        .map(|c| Focus::bind(&c.iter().map(|&i| i as u32).collect::<Vec<_>>(), generation))
        .collect();
    let seal = fnv(codebook.iter().flat_map(|i| i.to_le_bytes()));
    Ok(Bound {
        witness: Witness {
            facts,
            seal,
            focus: generation,
            scope: u64::from(scope.is_some()),
        },
        codebook,
        foci,
    })
}

// ---- phase 3: projection -----------------------------------------------------

/// A relation over the bound nodes: `pages x pages` tiles, row-major by page,
/// `None` for an all-zero tile. Tile `(I, J)` holds `R(x, z)` for `x` in page
/// `I`, `z` in page `J`, at `tile_addr(x, z)`: row `z`, bit `x`.
type Blocks = Vec<Option<Box<[u64]>>>;

fn tile() -> Box<[u64]> {
    vec![0u64; TILE_WORDS].into_boxed_slice()
}

fn set(t: &mut [u64], i: usize) {
    t[i >> 6] |= 1 << (i & 63);
}

fn get(t: &[u64], i: usize) -> bool {
    t[i >> 6] >> (i & 63) & 1 == 1
}

/// The edge relation, one block matrix per edge length.
struct Projection {
    witness: Witness,
    by_len: Vec<(usize, Blocks)>,
    /// Ids an edge reaches that no focus binds (only under a `scope`).
    escaped: BTreeSet<i32>,
    /// Edges whose endpoints sit in different foci (crossed by paging).
    cross_page_edges: u64,
    /// Codebook lookups keyed by an edge value.
    lookups: u64,
}

fn project(b: &Bound, e: &[Edge]) -> Projection {
    let pages = b.pages();
    let mut by_len: Vec<Option<Blocks>> = vec![None; LENS];
    let mut escaped = BTreeSet::new();
    let mut cross_page_edges = 0u64;
    let mut lookups = 0u64;
    for &(a, z, l) in e {
        lookups += 2;
        let (Some((i, x)), Some((j, y))) = (b.resolve(a), b.resolve(z)) else {
            for id in [a, z] {
                if b.resolve(id).is_none() {
                    escaped.insert(id);
                }
            }
            continue;
        };
        let l = l as usize;
        if l >= LENS {
            continue;
        }
        if i != j {
            cross_page_edges += 1;
        }
        let blocks = by_len[l].get_or_insert_with(|| vec![None; pages * pages]);
        set(
            blocks[i * pages + j].get_or_insert_with(tile),
            tile_addr(x, y),
        );
    }
    Projection {
        witness: b.witness,
        by_len: by_len
            .into_iter()
            .enumerate()
            .filter_map(|(l, b)| b.map(|b| (l, b)))
            .collect(),
        escaped,
        cross_page_edges,
        lookups,
    }
}

// ---- phase 4: recursion over tiles ---------------------------------------------

/// The tile product kernel: broadcast and spread buffers, the mask-risc
/// program and its scratch, reused across calls.
struct Kernel {
    bcast: Box<[u64]>,
    spread: Box<[u64]>,
    out: Box<[u64]>,
    prog: Program,
    scratch: Scratch<'static>,
}

/// `dst = c | (p & q)`: `Ternlog` with `(a, b, c) = (C, P, Q)`, table `0xF8`.
fn product_program() -> Program {
    Program::new(
        vec![MaskOp::Ternlog {
            imm: 0xF8,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    tile_products: u64,
    /// Coordinates `y` that contributed (both operands non-empty at `y`).
    steps: u64,
    /// mask-risc calls.
    calls: u64,
}

impl Kernel {
    fn with_program(prog: Program) -> Result<Kernel, Refuse> {
        admit(&prog)?;
        Ok(Kernel {
            bcast: tile(),
            spread: tile(),
            out: tile(),
            prog,
            scratch: Scratch::new(words_for(TILE_ROWS), 1),
        })
    }

    fn new() -> Kernel {
        Kernel::with_program(product_program()).expect("the product program has no gather")
    }

    /// `c |= a ∘ b`: `C(x, z) = ∃y A(x, y) ∧ B(y, z)`. Row `z` of `C` is the
    /// OR over `y` of row `y` of `A`, taken when bit `y` of row `z` of `B` is
    /// set. Every read is at a loop coordinate.
    fn product_into(&mut self, a: &[u64], b: &[u64], c: &mut Box<[u64]>, n: &mut Counters) {
        n.tile_products += 1;
        for y in 0..SLOTS {
            let ay = &a[y * ROW_WORDS..(y + 1) * ROW_WORDS];
            if ay.iter().all(|&w| w == 0) {
                continue;
            }
            let mut any = false;
            for z in 0..SLOTS {
                let on = get(&b[z * ROW_WORDS..(z + 1) * ROW_WORDS], y);
                any |= on;
                let w = if on { u64::MAX } else { 0 };
                self.spread[z * ROW_WORDS..(z + 1) * ROW_WORDS].fill(w);
            }
            if !any {
                continue;
            }
            for z in 0..SLOTS {
                self.bcast[z * ROW_WORDS..(z + 1) * ROW_WORDS].copy_from_slice(ay);
            }
            n.steps += 1;
            n.calls += 1;
            let masks: [&[u64]; 3] = [c, &self.bcast, &self.spread];
            let pl = Planes {
                n_rows: TILE_ROWS,
                masks: &masks,
                lanes: &[],
            };
            execute_into(
                &self.prog,
                &pl,
                &Foreign::NONE,
                &mut self.scratch,
                Out::Mask(&mut self.out),
            )
            .expect("the product program is valid");
            std::mem::swap(c, &mut self.out);
        }
    }
}

fn zero(pages: usize) -> Vec<Blocks> {
    vec![vec![None; pages * pages]; LENS]
}

/// The least fixpoint over tiles.
struct Run {
    walk: Vec<Blocks>,
    iterations: usize,
    counters: Counters,
}

fn recurse(b: &Bound, p: &Projection, current: &Witness, k: &mut Kernel) -> Result<Run, Refuse> {
    b.witness.check(current)?;
    p.witness.check(current)?;
    if !p.escaped.is_empty() {
        return Err(Refusal::IncompleteFocus.into());
    }
    let pages = b.pages();
    let mut walk = zero(pages);
    let mut delta = zero(pages);
    // Walk(x, x, 0) for every node.
    for i in 0..pages {
        let mut t = tile();
        for x in 0..b.page_len(i) {
            set(&mut t, tile_addr(x as u8, x as u8));
        }
        walk[0][i * pages + i] = Some(t.clone());
        delta[0][i * pages + i] = Some(t);
    }
    let mut live = vec![false; LENS];
    live[0] = true;
    let mut counters = Counters::default();
    let mut iterations = 0usize;
    // Every iteration that continues adds at least one bit.
    let bound = LENS * b.nodes() * b.nodes() + 1;
    loop {
        iterations += 1;
        assert!(iterations <= bound, "fixpoint did not terminate");
        let mut new = zero(pages);
        for (l1, edge) in &p.by_len {
            for l2 in (0..LENS).filter(|&l2| live[l2]) {
                let l = l1 + l2;
                if l >= LENS {
                    continue;
                }
                for i in 0..pages {
                    for j in 0..pages {
                        for m in 0..pages {
                            let (Some(a), Some(d)) =
                                (&edge[i * pages + m], &delta[l2][m * pages + j])
                            else {
                                continue;
                            };
                            let c = new[l][i * pages + j].get_or_insert_with(tile);
                            k.product_into(a, d, c, &mut counters);
                        }
                    }
                }
            }
        }
        let mut next_live = vec![false; LENS];
        for l in 0..LENS {
            for t in 0..pages * pages {
                let Some(n) = new[l][t].as_mut() else {
                    continue;
                };
                let w = walk[l][t].get_or_insert_with(tile);
                let mut any = false;
                for (nw, &ww) in n.iter_mut().zip(w.iter()) {
                    *nw &= !ww;
                    any |= *nw != 0;
                }
                if any {
                    for (ww, &nw) in w.iter_mut().zip(n.iter()) {
                        *ww |= nw;
                    }
                    next_live[l] = true;
                } else {
                    new[l][t] = None;
                }
            }
        }
        if !next_live.iter().any(|&x| x) {
            break;
        }
        delta = new;
        live = next_live;
    }
    Ok(Run {
        walk,
        iterations,
        counters,
    })
}

// ---- phase 5: decoding ---------------------------------------------------------

/// Per-pair reach over all lengths, tile by tile.
fn reach(r: &Run, t: usize) -> Box<[u64]> {
    let mut out = tile();
    for l in 0..LENS {
        if let Some(w) = &r.walk[l][t] {
            for (o, &x) in out.iter_mut().zip(w.iter()) {
                *o |= x;
            }
        }
    }
    out
}

fn popcount(t: &[u64]) -> usize {
    t.iter().map(|w| w.count_ones() as usize).sum()
}

/// `(|Walk|, |NoWalk|)` without building a tuple.
fn counts(b: &Bound, r: &Run) -> (usize, usize) {
    let pages = b.pages();
    let walk = r.walk.iter().flatten().flatten().map(|t| popcount(t)).sum();
    let reached: usize = (0..pages * pages).map(|t| popcount(&reach(r, t))).sum();
    (walk, b.nodes() * b.nodes() - reached)
}

/// The tuple sets, with every slot decoded through the codebook.
fn tuples(b: &Bound, r: &Run) -> (Tuples, u64) {
    let pages = b.pages();
    let mut walk = BTreeSet::new();
    let mut no_walk = BTreeSet::new();
    let mut decodes = 0u64;
    for i in 0..pages {
        for j in 0..pages {
            let t = i * pages + j;
            for (l, plane) in r.walk.iter().enumerate() {
                let Some(w) = &plane[t] else { continue };
                for z in 0..b.page_len(j) {
                    for x in 0..b.page_len(i) {
                        if get(w, tile_addr(x as u8, z as u8)) {
                            decodes += 2;
                            walk.insert((b.id(i, x), b.id(j, z), l as u32));
                        }
                    }
                }
            }
            let reached = reach(r, t);
            for z in 0..b.page_len(j) {
                for x in 0..b.page_len(i) {
                    if !get(&reached, tile_addr(x as u8, z as u8)) {
                        decodes += 2;
                        no_walk.insert((b.id(i, x), b.id(j, z)));
                    }
                }
            }
        }
    }
    (Tuples { walk, no_walk }, decodes)
}

/// Bytes held by the tiles of a run (walk planes only).
fn run_bytes(r: &Run) -> usize {
    r.walk.iter().flatten().flatten().count() * TILE_WORDS * 8
}

// ---- the whole tile path ---------------------------------------------------------

/// Every phase of one tile evaluation, timed.
struct TileReport {
    attest: Duration,
    validate: Duration,
    bind: Duration,
    project: Duration,
    recurse: Duration,
    count: Duration,
    decode: Duration,
    lookups: u64,
    decodes: u64,
    cross_page_edges: u64,
    pages: usize,
    iterations: usize,
    counters: Counters,
    tile_bytes: usize,
    counts: (usize, usize),
    tuples: Tuples,
}

impl TileReport {
    /// Every phase up to the counts: no tuple is built.
    fn e2e_count(&self) -> Duration {
        self.attest + self.validate + self.bind + self.project + self.recurse + self.count
    }

    /// Every phase up to the decoded tuples.
    fn e2e(&self) -> Duration {
        self.attest + self.validate + self.bind + self.project + self.recurse + self.decode
    }
}

fn tile_eval(e: &[Edge], max_pages: usize) -> Result<TileReport, Refuse> {
    {
        let s = Instant::now();
        let facts = attest(e);
        let attest_t = s.elapsed();
        let s = Instant::now();
        let b = bind(e, facts, 1, None, max_pages)?;
        let bind_t = s.elapsed();
        let s = Instant::now();
        let p = project(&b, e);
        let project_t = s.elapsed();
        let mut k = Kernel::new();
        let s = Instant::now();
        black_box(b.witness.check(&p.witness))?;
        let validate_t = s.elapsed();
        let s = Instant::now();
        let r = recurse(&b, &p, &b.witness, &mut k)?;
        let recurse_t = s.elapsed();
        let s = Instant::now();
        let c = counts(&b, &r);
        let count_t = s.elapsed();
        let s = Instant::now();
        let (t, decodes) = tuples(&b, &r);
        let decode_t = s.elapsed();
        Ok(TileReport {
            attest: attest_t,
            validate: validate_t,
            bind: bind_t,
            project: project_t,
            recurse: recurse_t,
            count: count_t,
            decode: decode_t,
            lookups: p.lookups,
            decodes,
            cross_page_edges: p.cross_page_edges,
            pages: b.pages(),
            iterations: r.iterations,
            counters: r.counters,
            tile_bytes: run_bytes(&r),
            counts: c,
            tuples: t,
        })
    }
}

// ---- benchmark -------------------------------------------------------------------

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(t: Duration) -> String {
    format!("{:.3}", t.as_secs_f64() * 1e3)
}

fn mib(b: usize) -> String {
    format!("{:.2}", b as f64 / (1 << 20) as f64)
}

/// A chain `0 -> 1 -> ... -> n-1`: length 0, except every 10th edge has
/// length 1, so a walk is at most `n / 10` long and the fixpoint needs about
/// `n` iterations.
fn chain(n: i32) -> Vec<Edge> {
    (0..n - 1)
        .map(|i| (i, i + 1, u32::from(i % 10 == 9)))
        .collect()
}

/// Every id in `0..n` with two out-edges of length 1, 2, 3, 5 or 7
/// (splitmix64, fixed seed), plus a ring.
fn mixed(n: i32, seed: u64) -> Vec<Edge> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let lens = [1, 2, 3, 5, 7];
    let mut e: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n, 3)).collect();
    for x in 0..n {
        for _ in 0..2 {
            let y = (next() % n as u64) as i32;
            e.push((x, y, lens[(next() % 5) as usize]));
        }
    }
    e
}

fn fixtures() -> Vec<(&'static str, Vec<Edge>)> {
    vec![
        ("edges(128)", edges(128)),
        ("edges(256)", edges(256)),
        ("edges(512)", edges(512)),
        ("edges(1024)", edges(1024)),
        ("chain(230)", chain(230)),
        ("mixed(256)", mixed(256, 7)),
    ]
}

/// walk_mask's three plane sets (`walk`, `delta`, `new`).
fn baseline_plane_bytes(e: &[Edge]) -> usize {
    3 * mask_walk(width(e), e).plane_bytes()
}

/// The largest node id plus one: walk_mask's plane width.
fn width(e: &[Edge]) -> usize {
    e.iter()
        .flat_map(|&(a, b, _)| [a, b])
        .max()
        .map_or(0, |m| m as usize + 1)
}

/// A `kB` field of `/proc/self/status`.
fn status_kib(field: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .expect("Linux /proc")
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .expect("field present")
}

/// Child mode: build the fixture, reset the resident high-water mark, run one
/// engine end to end (tuples decoded), print the growth of the high-water mark
/// over the resident set before the run, in KiB.
fn child(engine: &str, fixture: &str) {
    let (_, e) = fixtures()
        .into_iter()
        .find(|(n, _)| *n == fixture)
        .expect("known fixture");
    // Writing 5 to clear_refs resets VmHWM to the current RSS (Linux >= 4.0).
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
    let before = status_kib("VmRSS:");
    match engine {
        "mask" => {
            let p = mask_walk(width(&e), &e);
            black_box((p.counts(), p.tuples()));
        }
        "mask-count" => {
            black_box(mask_walk(width(&e), &e).counts());
        }
        "tile" => {
            black_box(tile_eval(&e, usize::MAX).expect("tile path answers"));
        }
        "tile-count" => {
            let b = bind(&e, attest(&e), 1, None, usize::MAX).expect("binds");
            let p = project(&b, &e);
            let r = recurse(&b, &p, &b.witness, &mut Kernel::new()).expect("answers");
            black_box(counts(&b, &r));
        }
        _ => panic!("unknown engine {engine}"),
    }
    println!("{}", status_kib("VmHWM:").saturating_sub(before));
}

/// Resident high-water growth of one engine on one fixture, in a fresh
/// process (jemalloc keeps freed pages, so in-process numbers would depend on
/// what ran before).
fn resident_kib(engine: &str, fixture: &str) -> usize {
    let out = std::process::Command::new(std::env::current_exe().expect("exe"))
        .args(["--resident", engine, fixture])
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
        "# correctness and end to end (ms, median of {REPS}; resident = VmHWM growth in a fresh process)\n"
    );
    println!(
        "| fixture | nodes | pages | mask count-only | tile count-only | tile / mask | mask + decode | tile + decode | tile / mask | Walk | NoWalk | oracle |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut memory = Vec::new();
    let mut detail = Vec::new();
    for (name, e) in fixtures() {
        let n = width(&e);
        // Baseline: count-only, then decoding of the same planes, timed apart.
        let mut tc = Vec::new();
        let mut td = Vec::new();
        let mut mask = None;
        let mut row_reads = 0;
        for _ in 0..REPS {
            let s = Instant::now();
            let p = mask_walk(n, &e);
            black_box(p.counts());
            tc.push(s.elapsed());
            row_reads = p.row_reads;
            let s = Instant::now();
            let t = black_box(p.tuples());
            td.push(s.elapsed());
            mask = Some(t);
        }
        let mask = mask.expect("REPS > 0");
        // Tile path, every phase.
        let reps: Vec<TileReport> = (0..REPS)
            .map(|_| tile_eval(&e, usize::MAX).expect("tile path answers"))
            .collect();
        let med = |f: fn(&TileReport) -> Duration| median(reps.iter().map(f).collect());
        let phases = [
            med(|r| r.attest),
            med(|r| r.validate),
            med(|r| r.bind),
            med(|r| r.project),
            med(|r| r.recurse),
            med(|r| r.count),
            med(|r| r.decode),
        ];
        let tile_e2e = med(|r| r.e2e());
        let tile_count = med(|r| r.e2e_count());
        let last = reps.into_iter().last().expect("REPS > 0");
        let nodes = last
            .tuples
            .walk
            .iter()
            .filter(|w| w.0 == w.1 && w.2 == 0)
            .count();
        let oracle = if nodes <= ORACLE_MAX_NODES {
            let a = ascent_tuples(&e);
            let c = crepe_tuples(&e);
            agree(&last.tuples, &a).unwrap_or_else(|d| panic!("{name}: tile vs ascent {d}"));
            agree(&last.tuples, &c).unwrap_or_else(|d| panic!("{name}: tile vs crepe {d}"));
            agree(&mask, &a).unwrap_or_else(|d| panic!("{name}: mask vs ascent {d}"));
            "tile = mask = ascent = crepe"
        } else {
            agree(&last.tuples, &mask).unwrap_or_else(|d| panic!("{name}: tile vs mask {d}"));
            "tile = mask; no Datalog oracle (UNCHECKED)"
        };
        assert_eq!(last.counts, last.tuples.counts(), "{name}: tile counts");
        let mask_count = median(tc);
        let mask_e2e = mask_count + median(td);
        println!(
            "| {name} | {nodes} | {} | {} | {} | {:.1}× | {} | {} | {:.1}× | {} | {} | {oracle} |",
            last.pages,
            ms(mask_count),
            ms(tile_count),
            tile_count.as_secs_f64() / mask_count.as_secs_f64(),
            ms(mask_e2e),
            ms(tile_e2e),
            tile_e2e.as_secs_f64() / mask_e2e.as_secs_f64(),
            last.counts.0,
            last.counts.1,
        );
        memory.push((
            name,
            ["mask-count", "tile-count", "mask", "tile"].map(|eng| resident_kib(eng, name)),
            baseline_plane_bytes(&e),
            last.tile_bytes,
        ));
        detail.push((name, phases, last, row_reads));
    }
    println!("\n# resident memory (MiB): VmHWM growth in a fresh process per cell\n");
    println!(
        "| fixture | mask count-only | tile count-only | mask + decode | tile + decode | mask planes (3 x walk) | tile walk tiles |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, kib, planes, tiles) in &memory {
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} |",
            mib(kib[0] * 1024),
            mib(kib[1] * 1024),
            mib(kib[2] * 1024),
            mib(kib[3] * 1024),
            mib(*planes),
            mib(*tiles),
        );
    }
    println!("\n# tile path by phase (ms, median of {REPS})\n");
    println!(
        "| fixture | attest | validate | bind | project | recurse | count-only | decode | iterations | tile products | contributing y | cross-page edges | walk tiles MiB |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, t, r, _) in &detail {
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            ms(t[0]),
            ms(t[1]),
            ms(t[2]),
            ms(t[3]),
            ms(t[4]),
            ms(t[5]),
            ms(t[6]),
            r.iterations,
            r.counters.tile_products,
            r.counters.steps,
            r.cross_page_edges,
            mib(r.tile_bytes),
        );
    }
    println!("\n# value-derived reads by origin\n");
    println!(
        "| fixture | walk_mask exec: delta row read at an adjacency bit | tile prep: codebook lookup per edge endpoint | tile exec | tile decode: codebook decode per output id |"
    );
    println!("|---|---:|---:|---:|---:|");
    for (name, _, r, row_reads) in &detail {
        println!(
            "| {name} | {row_reads} | {} | 0 | {} |",
            r.lookups, r.decodes
        );
    }
}

// ---- tests -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use baseline::tests::{LENGTHS, random_graph};

    /// The tile path's tuples equal ascent's, crepe's and walk_mask's.
    fn check(e: &[Edge]) -> TileReport {
        let r = tile_eval(e, usize::MAX).expect("tile path answers");
        assert_eq!(r.counts, r.tuples.counts(), "count-only vs tuples");
        agree(&r.tuples, &ascent_tuples(e)).unwrap_or_else(|d| panic!("ascent: {d}"));
        agree(&r.tuples, &crepe_tuples(e)).unwrap_or_else(|d| panic!("crepe: {d}"));
        let n = e
            .iter()
            .flat_map(|&(a, b, _)| [a, b])
            .max()
            .map_or(0, |m| m as usize + 1);
        agree(&r.tuples, &mask_walk(n, e).tuples()).unwrap_or_else(|d| panic!("walk_mask: {d}"));
        r
    }

    #[test]
    fn the_tile_product_equals_bitmat_compose() {
        use mork_pathmap_probe::tile_gemm::BitMat;
        let mut s = 0x7u64;
        let mut bit = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) % 17 == 0
        };
        let (mut a, mut b) = (BitMat::empty(), BitMat::empty());
        let (mut ta, mut tb) = (tile(), tile());
        for i in 0..SLOTS {
            for j in 0..SLOTS {
                if bit() {
                    a.set(i, j);
                    set(&mut ta, tile_addr(i as u8, j as u8));
                }
                if bit() {
                    b.set(i, j);
                    set(&mut tb, tile_addr(i as u8, j as u8));
                }
            }
        }
        let want = a.compose(&a.transpose(), &b);
        let mut c = tile();
        Kernel::new().product_into(&ta, &tb, &mut c, &mut Counters::default());
        for x in 0..SLOTS {
            for z in 0..SLOTS {
                assert_eq!(
                    get(&c, tile_addr(x as u8, z as u8)),
                    want.get(x, z),
                    "({x}, {z})"
                );
            }
        }
        assert!(want.edges() > 1000, "the product is not trivial");
    }

    #[test]
    fn mixed_lengths_with_cycles() {
        let e = random_graph(65, 2, 0x5EED_0001);
        let r = check(&e);
        let lens: BTreeSet<u32> = r.tuples.walk.iter().map(|w| w.2).collect();
        assert_eq!(lens.len(), LENS, "every length is reached");
        assert!(!r.tuples.no_walk.is_empty());
    }

    #[test]
    fn alternative_path_lengths() {
        let r = check(&[(0, 1, 1), (0, 2, 2), (2, 1, 3), (1, 3, 7), (0, 3, 5)]);
        for w in [(0, 1, 1), (0, 1, 5), (0, 3, 5), (0, 3, 8), (0, 3, 12)] {
            assert!(r.tuples.walk.contains(&w), "{w:?}");
        }
    }

    #[test]
    fn zero_length_edges_and_long_edges() {
        let r = check(&[(0, 1, 0), (1, 0, 0), (1, 2, 3), (2, 3, 50), (3, 4, 51)]);
        assert!(r.tuples.walk.contains(&(0, 2, 3)));
        assert!(r.tuples.walk.contains(&(2, 3, 50)));
        assert!(
            r.tuples.no_walk.contains(&(3, 4)),
            "a length-51 edge starts no walk"
        );
    }

    #[test]
    fn ids_that_are_not_nodes_take_no_part() {
        let e: Vec<Edge> = (0..40)
            .map(|i| (2 * i, 2 * ((i * 7 + 3) % 40), LENGTHS[i as usize % 5]))
            .collect();
        let r = check(&e);
        assert_eq!(r.pages, 1, "40 nodes bind into one focus");
    }

    #[test]
    fn a_chain_needs_at_least_200_iterations() {
        let r = check(&chain(230));
        assert!(r.iterations >= 200, "only {} iterations", r.iterations);
        assert_eq!(r.pages, 1);
    }

    #[test]
    fn a_zero_length_cycle_terminates() {
        // A ring of 200 zero-length edges and a chord of length 7: every pair
        // is reached at length 0, the chord adds lengths 7, 14, ..., 49.
        let mut e: Vec<Edge> = (0..200).map(|i| (i, (i + 1) % 200, 0)).collect();
        e.push((0, 100, 7));
        let r = check(&e);
        assert!(r.iterations >= 150, "only {} iterations", r.iterations);
        assert!(r.tuples.walk.contains(&(5, 4, 0)));
        assert!(r.tuples.walk.contains(&(150, 30, 49)));
        assert!(!r.tuples.walk.contains(&(150, 30, 56)));
    }

    #[test]
    fn paging_across_foci_keeps_the_global_answer() {
        for n in [255usize, 256, 257, 300] {
            let mut e = random_graph(n, 1, 0xB0B0 + n as u64);
            // Edges across every page boundary, both directions.
            e.push((255.min(n as i32 - 1), 0, 2));
            e.push((0, n as i32 - 1, 1));
            let r = check(&e);
            assert_eq!(r.pages, n.div_ceil(SLOTS), "n={n}");
            if n > SLOTS {
                assert!(r.cross_page_edges > 0, "n={n}: no edge crosses a page");
            }
        }
    }

    #[test]
    fn the_benchmark_generator_across_two_foci() {
        let r = check(&edges(260));
        assert_eq!(r.pages, 2);
        assert!(r.cross_page_edges > 0);
    }

    #[test]
    fn a_sparse_graph_across_three_foci() {
        // Sparse, so the debug-build oracles stay fast in CI.
        let r = check(&random_graph(600, 1, 0x3F0C1));
        assert_eq!(r.pages, 3);
        assert!(r.cross_page_edges > 0);
    }

    // ---- refusals ----

    #[test]
    fn a_stale_witness_is_refused() {
        let e = edges(64);
        let facts = attest(&e);
        let b = bind(&e, facts, 1, None, usize::MAX).expect("binds");
        let p = project(&b, &e);
        let mut k = Kernel::new();
        assert!(recurse(&b, &p, &b.witness, &mut k).is_ok());
        let mut w = b.witness;
        w.focus = 2;
        assert_eq!(
            recurse(&b, &p, &w, &mut k).err(),
            Some(Refuse::Rmap(Refusal::FocusChanged))
        );
        let mut changed = e.clone();
        changed.push((0, 1, 9));
        let w = Witness {
            facts: attest(&changed),
            ..b.witness
        };
        assert_eq!(
            recurse(&b, &p, &w, &mut k).err(),
            Some(Refuse::Rmap(Refusal::FactsChanged))
        );
        let w = Witness {
            seal: b.witness.seal ^ 1,
            ..b.witness
        };
        assert_eq!(
            recurse(&b, &p, &w, &mut k).err(),
            Some(Refuse::Rmap(Refusal::IndexResealed))
        );
    }

    #[test]
    fn too_many_nodes_for_the_foci_is_refused_not_truncated() {
        let e = random_graph(300, 1, 3);
        assert_eq!(
            tile_eval(&e, 1).err(),
            Some(Refuse::Rmap(Refusal::IncompleteFocus))
        );
        assert!(tile_eval(&e, 2).is_ok());
    }

    #[test]
    fn a_focus_that_misses_reached_nodes_is_refused() {
        let e = random_graph(120, 2, 9);
        let scope: Vec<i32> = (0..100).collect();
        let facts = attest(&e);
        let b = bind(&e, facts, 1, Some(&scope), usize::MAX).expect("binds");
        let p = project(&b, &e);
        assert!(!p.escaped.is_empty(), "the fixture leaves the scope");
        assert!(p.escaped.iter().all(|&i| i >= 100));
        assert_eq!(
            recurse(&b, &p, &b.witness, &mut Kernel::new()).err(),
            Some(Refuse::Rmap(Refusal::IncompleteFocus))
        );
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
            .find(|w| w.2 > 0 && w.2 < MAX_PATH_LEN && !bad.walk.contains(&(w.0, w.1, w.2 + 1)))
            .expect("a tuple whose next length is absent");
        bad.walk.remove(&w);
        bad.walk.insert((w.0, w.1, w.2 + 1));
        assert_eq!(bad.counts(), r.tuples.counts(), "counts cannot see it");
        let err = agree(&bad, &ascent_tuples(&e)).expect_err("the oracle sees it");
        assert!(err.starts_with("Walk"), "{err}");
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
        assert_eq!(
            Kernel::with_program(gather).err(),
            Some(Refuse::UnauthorizedGather),
            "a gathering program must not be admitted"
        );
        assert_eq!(admit(&product_program()), Ok(()));
    }
}
