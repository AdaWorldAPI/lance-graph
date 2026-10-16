//! D-HOLO H14c: thought-first positional reads, Hamming presence, memory.
//!
//! Pre-registered in `.claude/plans/d-holo-h14c-thought-first-v1.md` before
//! any run. The question is not how cheaply a layout rebuilds a row, but which
//! accesses remain once the thought, its operands and the Cartesian address
//! are known:
//!
//! ```text
//! Thought(t)[Table(a)[i] : Table(b)[i]]
//! ```
//!
//! The thought's program selects lane tables `a`, `b` (handles resolved once,
//! never from the row); `i` addresses each of them directly; a presence bit
//! says whether a value is there, never where. Three paths are measured apart:
//! A point projection, B presence populations (mask-risc programs, no value
//! loads), C an explicit terminal row.
//!
//! Arms over one fixture: C (positional table per touched (thought, lane),
//! lazily zeroed or eagerly committed), B (connected delta: row index, lane
//! mask, packed lanes, rank), S (sparse: sorted addresses, binary search; the
//! one explicit address translation, priced for memory only).
//!
//! Run (release, the tier is printed): `cargo run --release --example
//! holo_thought_first_probe`. Oracle tests: `cargo test --release --example
//! holo_thought_first_probe`. Raw rows are printed as `csv,` lines.
#![allow(dead_code)]

use std::hint::black_box;
use std::time::Instant;

use lance_graph_mask_risc::{
    execute, MaskOp, Operand as MOp, Planes, Program, Scratch, Terminal, Value,
};

const N: usize = 65_536;
const WORDS: usize = N / 64;
const LANE: usize = 16;
const LANES: usize = 32;
const ROW: usize = LANES * LANE;
const NONE: u32 = u32::MAX;

type Val = [u8; LANE];

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The value thought `t` writes into lane `l` at address `i`. Never all zero,
/// so an overlay of sixteen zero bytes only appears where a test puts it.
fn overlay_value(t: usize, l: usize, i: usize) -> Val {
    let a = mix(((t as u64) << 48) ^ ((l as u64) << 32) ^ i as u64);
    let b = mix(a ^ 0x5555);
    let mut v = [0u8; LANE];
    v[..8].copy_from_slice(&a.to_le_bytes());
    v[8..].copy_from_slice(&b.to_le_bytes());
    v[0] |= 1;
    v
}

/// The immutable base: one 512-byte row per address (the given).
fn base_rows() -> Vec<u8> {
    let mut b = vec![0u8; N * ROW];
    for (i, row) in b.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
        for l in 0..LANES {
            let v = mix(0xBA5E ^ ((l as u64) << 32) ^ i as u64).to_le_bytes();
            row[l * LANE..l * LANE + 8].copy_from_slice(&v);
            row[l * LANE + 8..(l + 1) * LANE].copy_from_slice(&v);
        }
    }
    b
}

fn base_lane(base: &[u8], i: usize, l: usize) -> Val {
    base[i * ROW + l * LANE..i * ROW + (l + 1) * LANE]
        .try_into()
        .unwrap()
}

#[inline]
fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

// ------------------------------------------------------------- counters

#[cfg(test)]
thread_local! {
    /// Lane tables a read consulted (one per requested lane).
    static CONSULTED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Presence bits tested.
    static PRESENCE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// 16-byte values loaded (overlay or base).
    static LOADS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Full rows built.
    static ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Explicit address translations (arm S only).
    static TRANSLATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

macro_rules! count {
    ($c:ident) => {
        #[cfg(test)]
        $c.with(|c| c.set(c.get() + 1));
    };
}

#[cfg(test)]
fn counters() -> [u64; 5] {
    [
        CONSULTED.with(|c| c.get()),
        PRESENCE.with(|c| c.get()),
        LOADS.with(|c| c.get()),
        ROWS.with(|c| c.get()),
        TRANSLATIONS.with(|c| c.get()),
    ]
}

#[cfg(test)]
fn reset() {
    for c in [&CONSULTED, &PRESENCE, &LOADS, &ROWS, &TRANSLATIONS] {
        c.with(|c| c.set(0));
    }
}

// ------------------------------------------------------------- fixture

/// Where dirty addresses fall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    /// Each address independently with probability `d`.
    Uniform,
    /// Runs of 64 consecutive addresses.
    Clustered,
    /// One contiguous range.
    Contiguous,
    /// Dense inside a few 256-address blocks (one Morton prefix cell each).
    Morton,
    /// Lanes of the touched set split the addresses between them.
    Disjoint,
    /// Every lane's set is the first lane's with 10 % of it changed.
    Overlapping,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::Uniform => "uniform",
            Pattern::Clustered => "clustered",
            Pattern::Contiguous => "contiguous",
            Pattern::Morton => "morton",
            Pattern::Disjoint => "disjoint",
            Pattern::Overlapping => "overlapping",
        }
    }
}

/// Dirty addresses of the `j`-th touched lane (of `u`) for thought `t`.
fn dirty_set(p: Pattern, d: f64, t: usize, j: usize, u: usize) -> Vec<u32> {
    let seed = mix(((t as u64) << 20) ^ (j as u64) ^ 0xD1E7);
    let thresh = (d * (u64::MAX as f64)) as u64;
    let take = |i: usize, s: u64| d >= 1.0 || mix(s ^ i as u64) < thresh;
    let mut v = Vec::new();
    match p {
        Pattern::Uniform => {
            for i in 0..N {
                if take(i, seed) {
                    v.push(i as u32);
                }
            }
        }
        Pattern::Clustered => {
            for b in 0..N / 64 {
                if take(b, seed) {
                    v.extend((b * 64..b * 64 + 64).map(|i| i as u32));
                }
            }
        }
        Pattern::Contiguous => {
            let n = (d * N as f64).round() as usize;
            let s = (seed as usize) % (N - n + 1);
            v.extend((s..s + n).map(|i| i as u32));
        }
        Pattern::Morton => {
            // blocks of 256 addresses, half of each chosen block dirty
            let blocks = ((d * N as f64) / 128.0).ceil() as usize;
            let mut chosen: Vec<usize> = (0..N / 256).collect();
            chosen.sort_by_key(|&b| mix(seed ^ b as u64));
            chosen.truncate(blocks.min(N / 256));
            chosen.sort();
            for b in chosen {
                for i in b * 256..b * 256 + 256 {
                    if mix(seed ^ 0xA ^ i as u64) & 1 == 0 {
                        v.push(i as u32);
                    }
                }
            }
        }
        Pattern::Disjoint => {
            let common = mix(((t as u64) << 20) ^ 0xD15);
            for i in 0..N {
                if take(i, common) && (mix(common ^ 0xF00 ^ i as u64) as usize) % u == j {
                    v.push(i as u32);
                }
            }
        }
        Pattern::Overlapping => {
            let first = mix(((t as u64) << 20) ^ 0xD1E7);
            for i in 0..N {
                let in_first = take(i, first);
                let flip = j > 0 && mix(seed ^ 0xF1 ^ i as u64) < u64::MAX / 10 && take(i, seed);
                if in_first ^ flip {
                    v.push(i as u32);
                }
            }
        }
    }
    v
}

/// Which lanes a thought touches: the first `u` of a fixed permutation of the
/// 32 lanes. A selection of `k` lanes takes the first `k` touched lanes.
fn touched_lanes(u: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..LANES).collect();
    all.sort_by_key(|&l| mix(0x7A11 ^ l as u64));
    all.truncate(u);
    all
}

/// One thought's writes: per touched lane its dirty addresses.
struct Writes {
    thought: usize,
    lanes: Vec<(usize, Vec<u32>)>,
}

fn writes(p: Pattern, d: f64, t: usize, u: usize) -> Writes {
    let lanes = touched_lanes(u);
    Writes {
        thought: t,
        lanes: lanes
            .iter()
            .enumerate()
            .map(|(j, &l)| (l, dirty_set(p, d, t, j, u)))
            .collect(),
    }
}

// ------------------------------------------------------------- arm C

/// How a positional table's pages are committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Commit {
    /// Zeroed allocation: the OS commits a page on its first write.
    Lazy,
    /// Every page written at seal.
    Eager,
}

/// One (thought, lane) table: the value at address `i` is `values[i]`.
struct LaneTable {
    present: Vec<u64>,
    values: Vec<Val>,
}

impl LaneTable {
    fn seal(t: usize, l: usize, dirty: &[u32], commit: Commit) -> Self {
        let mut values = vec![[0u8; LANE]; N];
        if commit == Commit::Eager {
            // write one byte per page so every page is committed
            for p in values.iter_mut().step_by(4096 / LANE) {
                p[0] = black_box(0);
            }
        }
        let mut present = vec![0u64; WORDS];
        for &i in dirty {
            let i = i as usize;
            present[i / 64] |= 1 << (i % 64);
            values[i] = overlay_value(t, l, i);
        }
        LaneTable { present, values }
    }
}

/// Every touched table of one thought, indexed by lane (32 handles).
struct ThoughtC {
    id: u16,
    tables: Vec<Option<LaneTable>>,
}

impl ThoughtC {
    fn seal(w: &Writes, commit: Commit) -> Self {
        let mut tables: Vec<Option<LaneTable>> = (0..LANES).map(|_| None).collect();
        for (l, dirty) in &w.lanes {
            tables[*l] = Some(LaneTable::seal(w.thought, *l, dirty, commit));
        }
        ThoughtC {
            id: w.thought as u16,
            tables,
        }
    }

    fn touched(&self) -> usize {
        self.tables.iter().filter(|t| t.is_some()).count()
    }
}

/// What to return for a lane the thought has no value for at `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fallback {
    /// The immutable base value.
    Base,
    /// Nothing: report `Absent`.
    Absent,
}

/// A thought program's operand selection: which thought, which lanes.
#[derive(Clone, Debug)]
struct Selection {
    thought: usize,
    lanes: Vec<usize>,
    fallback: Fallback,
}

/// One operand of a point projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Got {
    /// The thought wrote it; which thought is reported, so two thoughts'
    /// evidence at one address stays apart.
    Overlay { thought: u16, value: Val },
    /// Fallback to the immutable base.
    Base(Val),
    /// No overlay and no fallback.
    Absent,
    /// The aperture (authorization, time) excludes the address; nothing loaded.
    Denied,
}

/// A selection with its table handles resolved once: one handle per
/// requested lane, nothing else of the thought.
struct Resolved<'a> {
    thought: u16,
    slots: Vec<(usize, Option<&'a LaneTable>)>,
    fallback: Fallback,
    base: &'a [u8],
}

fn resolve<'a>(sel: &Selection, thoughts: &'a [ThoughtC], base: &'a [u8]) -> Resolved<'a> {
    let th = &thoughts[sel.thought];
    Resolved {
        thought: th.id,
        slots: sel
            .lanes
            .iter()
            .map(|&l| (l, th.tables[l].as_ref()))
            .collect(),
        fallback: sel.fallback,
        base,
    }
}

impl Resolved<'_> {
    #[inline]
    fn miss(&self, l: usize, i: usize) -> Got {
        match self.fallback {
            Fallback::Base => {
                count!(LOADS);
                Got::Base(base_lane(self.base, i, l))
            }
            Fallback::Absent => Got::Absent,
        }
    }

    /// `Thought(t)[Table(a)[i] : Table(b)[i] : …]`: one presence test and at
    /// most one load per requested lane, all at address `i`.
    #[inline]
    fn read(&self, i: usize, out: &mut [Got]) {
        for (o, &(l, table)) in out.iter_mut().zip(&self.slots) {
            count!(CONSULTED);
            *o = match table {
                Some(t) => {
                    count!(PRESENCE);
                    if bit(&t.present, i) {
                        count!(LOADS);
                        Got::Overlay {
                            thought: self.thought,
                            value: t.values[i],
                        }
                    } else {
                        self.miss(l, i)
                    }
                }
                None => self.miss(l, i),
            };
        }
    }

    /// The same read under an aperture (authorization and time composed into
    /// one mask): a denied address loads nothing.
    #[inline]
    fn read_under(&self, aperture: &[u64], i: usize, out: &mut [Got]) {
        if !bit(aperture, i) {
            out.iter_mut().for_each(|o| *o = Got::Denied);
            return;
        }
        self.read(i, out);
    }
}

/// The explicit terminal: one 512-byte row, base plus every overlay of the
/// thought. Only a named terminal calls this.
fn materialize_row(th: &ThoughtC, base: &[u8], i: usize) -> [u8; ROW] {
    count!(ROWS);
    let mut row: [u8; ROW] = base[i * ROW..(i + 1) * ROW].try_into().unwrap();
    for (l, t) in th.tables.iter().enumerate() {
        if let Some(t) = t {
            if bit(&t.present, i) {
                row[l * LANE..(l + 1) * LANE].copy_from_slice(&t.values[i]);
            }
        }
    }
    row
}

// ------------------------------------------------------------- arm B

/// Connected delta: per dirty row one record (lane mask, changed lanes packed
/// in lane order), found through a positional row index.
struct ThoughtB {
    id: u16,
    rec_of: Vec<u32>,
    masks: Vec<u32>,
    offs: Vec<u32>,
    vals: Vec<Val>,
}

impl ThoughtB {
    fn seal(w: &Writes) -> Self {
        let mut row_mask = vec![0u32; N];
        for (l, dirty) in &w.lanes {
            for &i in dirty {
                row_mask[i as usize] |= 1 << l;
            }
        }
        let mut b = ThoughtB {
            id: w.thought as u16,
            rec_of: vec![NONE; N],
            masks: Vec::new(),
            offs: Vec::new(),
            vals: Vec::new(),
        };
        for (i, &m) in row_mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            b.rec_of[i] = b.masks.len() as u32;
            b.masks.push(m);
            b.offs.push(b.vals.len() as u32);
            let mut mm = m;
            while mm != 0 {
                let l = mm.trailing_zeros() as usize;
                b.vals.push(overlay_value(w.thought, l, i));
                mm &= mm - 1;
            }
        }
        b
    }

    /// Lane `l` of row `i`: index lookup, mask test, popcount rank.
    #[inline]
    fn read(
        &self,
        sel_lanes: &[usize],
        fallback: Fallback,
        base: &[u8],
        i: usize,
        out: &mut [Got],
    ) {
        let r = self.rec_of[i];
        for (o, &l) in out.iter_mut().zip(sel_lanes) {
            *o = if r != NONE && self.masks[r as usize] >> l & 1 == 1 {
                let m = self.masks[r as usize];
                let rank = (m & ((1u32 << l) - 1)).count_ones();
                Got::Overlay {
                    thought: self.id,
                    value: self.vals[(self.offs[r as usize] + rank) as usize],
                }
            } else {
                match fallback {
                    Fallback::Base => Got::Base(base_lane(base, i, l)),
                    Fallback::Absent => Got::Absent,
                }
            };
        }
    }
}

// ------------------------------------------------------------- arm S

/// Sparse lane: dirty addresses sorted with their values. Lane `l` at `i` is
/// a binary search: an explicit address translation, priced, never direct.
struct SparseLane {
    coords: Vec<u32>,
    vals: Vec<Val>,
}

struct ThoughtS {
    id: u16,
    lanes: Vec<Option<SparseLane>>,
}

impl ThoughtS {
    fn seal(w: &Writes) -> Self {
        let mut lanes: Vec<Option<SparseLane>> = (0..LANES).map(|_| None).collect();
        for (l, dirty) in &w.lanes {
            lanes[*l] = Some(SparseLane {
                coords: dirty.clone(),
                vals: dirty
                    .iter()
                    .map(|&i| overlay_value(w.thought, *l, i as usize))
                    .collect(),
            });
        }
        ThoughtS {
            id: w.thought as u16,
            lanes,
        }
    }

    #[inline]
    fn read(
        &self,
        sel_lanes: &[usize],
        fallback: Fallback,
        base: &[u8],
        i: usize,
        out: &mut [Got],
    ) {
        for (o, &l) in out.iter_mut().zip(sel_lanes) {
            let hit = self.lanes[l].as_ref().and_then(|s| {
                count!(TRANSLATIONS);
                s.coords.binary_search(&(i as u32)).ok().map(|p| s.vals[p])
            });
            *o = match hit {
                Some(value) => Got::Overlay {
                    thought: self.id,
                    value,
                },
                None => match fallback {
                    Fallback::Base => Got::Base(base_lane(base, i, l)),
                    Fallback::Absent => Got::Absent,
                },
            };
        }
    }
}

// ------------------------------------------------------------- populations

/// `popcount(planes[0] & planes[1] & … )` through one mask-risc program.
fn and_count_program(n: u16) -> Program {
    let mut ops = vec![MaskOp::And {
        a: MOp::Plane(0),
        b: MOp::Plane(1),
        dst: 0,
    }];
    for p in 2..n {
        ops.push(MaskOp::And {
            a: MOp::Scratch(0),
            b: MOp::Plane(p),
            dst: 0,
        });
    }
    Program::new(
        ops,
        Terminal::Count {
            mask: MOp::Scratch(0),
        },
    )
}

/// `popcount(planes[0] ^ planes[1])`: Hamming distance of two coverages.
fn xor_count_program() -> Program {
    Program::new(
        vec![MaskOp::Xor {
            a: MOp::Plane(0),
            b: MOp::Plane(1),
            dst: 0,
        }],
        Terminal::Count {
            mask: MOp::Scratch(0),
        },
    )
}

/// `popcount(OR of n planes)`: the session-wide population.
fn or_count_program(n: u16) -> Program {
    if n == 1 {
        return Program::new(
            vec![],
            Terminal::Count {
                mask: MOp::Plane(0),
            },
        );
    }
    let mut ops = vec![MaskOp::Or {
        a: MOp::Plane(0),
        b: MOp::Plane(1),
        dst: 0,
    }];
    for p in 2..n {
        ops.push(MaskOp::Or {
            a: MOp::Scratch(0),
            b: MOp::Plane(p),
            dst: 0,
        });
    }
    Program::new(
        ops,
        Terminal::Count {
            mask: MOp::Scratch(0),
        },
    )
}

fn run_count(p: &Program, planes: &[&[u64]]) -> usize {
    let pl = Planes {
        n_rows: N,
        masks: planes,
        lanes: &[],
    };
    let mut s = Scratch::for_program(p, N).expect("scratch");
    match execute(p, &pl, &mut s, None).expect("execute") {
        Value::Count(c) => c,
        v => panic!("not a count: {v:?}"),
    }
}

fn scalar_and(planes: &[&[u64]]) -> usize {
    (0..N).filter(|&i| planes.iter().all(|p| bit(p, i))).count()
}

fn scalar_xor(a: &[u64], b: &[u64]) -> usize {
    (0..N).filter(|&i| bit(a, i) != bit(b, i)).count()
}

/// The planes and value lanes a mask-risc program names (D-HOLO-H14c-6):
/// the operand selection the program carries, read from the program itself.
/// Parsed from the program's `Debug` form (`Plane(n)`, `lane: n`) so it
/// covers every op and terminal variant without restating them; a lane field
/// that is not a literal is not counted.
fn lane_footprint(
    p: &Program,
) -> (
    std::collections::BTreeSet<u16>,
    std::collections::BTreeSet<u16>,
) {
    let text = format!("{p:?}");
    let grab = |key: &str, end: char| {
        text.match_indices(key)
            .filter_map(|(i, _)| {
                let rest = &text[i + key.len()..];
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                let ok = rest[num.len()..].starts_with(end) || end == ' ';
                (ok && !num.is_empty()).then(|| num.parse().ok()).flatten()
            })
            .collect::<std::collections::BTreeSet<u16>>()
    };
    (grab("Plane(", ')'), grab("lane: ", ' '))
}

// ------------------------------------------------------------- measurement

fn proc_status(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Minor page faults of this process so far (`/proc/self/stat` field 10).
fn minflt() -> u64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = s.rsplit(')').next().unwrap_or("");
    after
        .split_whitespace()
        .nth(7)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

struct Mem {
    vm_kb: u64,
    rss_kb: u64,
    flt: u64,
}

fn mem() -> Mem {
    Mem {
        vm_kb: proc_status("VmSize:"),
        rss_kb: proc_status("VmRSS:"),
        flt: minflt(),
    }
}

/// Median of 7 batches, ns per op.
fn time_batches<F: FnMut()>(ops_per_batch: usize, mut before: impl FnMut(), mut f: F) -> f64 {
    let mut v = Vec::with_capacity(7);
    f();
    for _ in 0..7 {
        before();
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_nanos() as f64 / ops_per_batch as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[3]
}

/// Touch a 128 MiB buffer so the next batch starts cache-cold.
fn evict(buf: &mut [u64]) {
    let mut s = 0u64;
    for w in buf.iter_mut().step_by(8) {
        *w = w.wrapping_add(1);
        s = s.wrapping_add(*w);
    }
    black_box(s);
}

fn csv(fields: &[String]) {
    println!("csv,{}", fields.join(","));
}

const READS: usize = 8_192;

fn addrs_hot() -> Vec<usize> {
    let set: Vec<usize> = (0..256).map(|j| (mix(0x407 ^ j) as usize) % N).collect();
    (0..READS).map(|j| set[j % 256]).collect()
}

fn addrs_cold() -> Vec<usize> {
    (0..READS)
        .map(|j| (mix(0xC01D ^ j as u64) as usize) % N)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let only = args.get(1).map(String::as_str).unwrap_or("all");
    if only == "memcell" {
        memory_cell(
            &args[2],
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            &args[5],
        );
        return;
    }
    println!("tier: {}", tier());
    println!(
        "host: MemTotal {} kB, batches 7 (median), reads per batch {READS}",
        proc_status_global("MemTotal:")
    );
    println!("csv,section,arm,pattern,density,T,U,k,cache,metric,value");
    let base = base_rows();
    let mut evict_buf = vec![0u64; 128 << 20 >> 3];
    if only == "all" || only == "main" {
        main_grid(&base, &mut evict_buf);
    }
    if only == "all" || only == "thoughts" {
        thought_axis(&base, &mut evict_buf);
    }
    if only == "all" || only == "spatial" {
        spatial_axis(&base, &mut evict_buf);
    }
    if only == "all" || only == "memory" {
        memory_grid();
    }
}

fn proc_status_global(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

const DENSITIES: [f64; 6] = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0];
const KS: [usize; 6] = [1, 2, 4, 8, 16, 31];

#[allow(clippy::too_many_arguments)]
fn row(
    section: &str,
    arm: &str,
    p: Pattern,
    d: f64,
    t: usize,
    u: usize,
    k: usize,
    cache: &str,
    metric: &str,
    value: f64,
) {
    csv(&[
        section.into(),
        arm.into(),
        p.name().into(),
        format!("{d}"),
        t.to_string(),
        u.to_string(),
        k.to_string(),
        cache.into(),
        metric.into(),
        format!("{value:.3}"),
    ]);
}

/// Point projection (A), populations (B) and the terminal (C) over one
/// thought, uniform addresses, every (density, U, k).
fn main_grid(base: &[u8], ev: &mut [u64]) {
    let hot = addrs_hot();
    let cold = addrs_cold();
    for &d in &DENSITIES {
        for &u in &[1usize, 2, 4, 8, 16, 31] {
            // U is crossed at {k, 8, 31} with U >= k
            let ks: Vec<usize> = KS
                .iter()
                .copied()
                .filter(|&k| k <= u && (u == k || u == 8 || u == 31))
                .collect();
            let w = writes(Pattern::Uniform, d, 0, u);
            let c = vec![ThoughtC::seal(&w, Commit::Lazy)];
            let b = ThoughtB::seal(&w);
            let s = ThoughtS::seal(&w);
            for &k in &ks {
                let lanes = touched_lanes(u)[..k].to_vec();
                let sel = Selection {
                    thought: 0,
                    lanes: lanes.clone(),
                    fallback: Fallback::Base,
                };
                let res_ns = {
                    let t = Instant::now();
                    for _ in 0..1000 {
                        black_box(resolve(black_box(&sel), &c, base));
                    }
                    t.elapsed().as_nanos() as f64 / 1000.0
                };
                row(
                    "A",
                    "C",
                    Pattern::Uniform,
                    d,
                    1,
                    u,
                    k,
                    "-",
                    "resolve_ns",
                    res_ns,
                );
                let r = resolve(&sel, &c, base);
                let mut out = vec![Got::Absent; k];
                for (cache, addrs) in [("hot", &hot), ("cold", &cold)] {
                    let cold_run = cache == "cold";
                    let ns_c = time_batches(
                        READS,
                        || {
                            if cold_run {
                                evict(ev)
                            }
                        },
                        || {
                            for &i in addrs.iter() {
                                r.read(black_box(i), &mut out);
                                black_box(&out);
                            }
                        },
                    );
                    let ns_b = time_batches(
                        READS,
                        || {
                            if cold_run {
                                evict(ev)
                            }
                        },
                        || {
                            for &i in addrs.iter() {
                                b.read(&lanes, Fallback::Base, base, black_box(i), &mut out);
                                black_box(&out);
                            }
                        },
                    );
                    let ns_s = time_batches(
                        READS,
                        || {
                            if cold_run {
                                evict(ev)
                            }
                        },
                        || {
                            for &i in addrs.iter() {
                                s.read(&lanes, Fallback::Base, base, black_box(i), &mut out);
                                black_box(&out);
                            }
                        },
                    );
                    row(
                        "A",
                        "C",
                        Pattern::Uniform,
                        d,
                        1,
                        u,
                        k,
                        cache,
                        "read_ns",
                        ns_c,
                    );
                    row(
                        "A",
                        "B",
                        Pattern::Uniform,
                        d,
                        1,
                        u,
                        k,
                        cache,
                        "read_ns",
                        ns_b,
                    );
                    row(
                        "A",
                        "S",
                        Pattern::Uniform,
                        d,
                        1,
                        u,
                        k,
                        cache,
                        "read_ns",
                        ns_s,
                    );
                }
            }
            if u >= 2 {
                // populations over the first two touched tables, aperture = all
                let l = touched_lanes(u);
                let pa = &c[0].tables[l[0]].as_ref().unwrap().present;
                let pb = &c[0].tables[l[1]].as_ref().unwrap().present;
                let ap = vec![u64::MAX; WORDS];
                let and3 = and_count_program(3);
                let xor = xor_count_program();
                let planes3: [&[u64]; 3] = [pa, pb, &ap];
                let planes2: [&[u64]; 2] = [pa, pb];
                let ns_and = time_batches(
                    WORDS,
                    || {},
                    || {
                        black_box(run_count(&and3, black_box(&planes3)));
                    },
                );
                let ns_xor = time_batches(
                    WORDS,
                    || {},
                    || {
                        black_box(run_count(&xor, black_box(&planes2)));
                    },
                );
                row(
                    "B",
                    "C",
                    Pattern::Uniform,
                    d,
                    1,
                    u,
                    2,
                    "hot",
                    "shared_ns_per_word",
                    ns_and,
                );
                row(
                    "B",
                    "C",
                    Pattern::Uniform,
                    d,
                    1,
                    u,
                    2,
                    "hot",
                    "hamming_ns_per_word",
                    ns_xor,
                );
                row(
                    "B",
                    "C",
                    Pattern::Uniform,
                    d,
                    1,
                    u,
                    2,
                    "-",
                    "shared_count",
                    run_count(&and3, &planes3) as f64,
                );
            }
            if u == 31 {
                // explicit terminal: rows built for 1,024 addresses
                let addrs: Vec<usize> = (0..1024).map(|j| (mix(0x7E ^ j) as usize) % N).collect();
                let ns = time_batches(
                    1024,
                    || {},
                    || {
                        for &i in &addrs {
                            black_box(materialize_row(&c[0], base, black_box(i)));
                        }
                    },
                );
                row("C", "C", Pattern::Uniform, d, 1, u, 31, "hot", "row_ns", ns);
            }
        }
    }
}

/// Point reads across thoughts and the session OR, at k = 2, U = 31.
fn thought_axis(base: &[u8], ev: &mut [u64]) {
    for &d in &[0.01, 0.1] {
        for &tn in &[1usize, 8, 64] {
            let cs: Vec<ThoughtC> = (0..tn)
                .map(|t| ThoughtC::seal(&writes(Pattern::Uniform, d, t, 31), Commit::Lazy))
                .collect();
            let lanes = touched_lanes(31)[..2].to_vec();
            let resolved: Vec<Resolved> = (0..tn)
                .map(|t| {
                    resolve(
                        &Selection {
                            thought: t,
                            lanes: lanes.clone(),
                            fallback: Fallback::Base,
                        },
                        &cs,
                        base,
                    )
                })
                .collect();
            let pairs: Vec<(usize, usize)> = (0..READS)
                .map(|j| {
                    let z = mix(0x7407 ^ j as u64);
                    ((z as usize) % tn, ((z >> 32) as usize) % N)
                })
                .collect();
            let mut out = vec![Got::Absent; 2];
            for cache in ["hot", "cold"] {
                let cold_run = cache == "cold";
                let ns = time_batches(
                    READS,
                    || {
                        if cold_run {
                            evict(ev)
                        }
                    },
                    || {
                        for &(t, i) in &pairs {
                            resolved[t].read(black_box(i), &mut out);
                            black_box(&out);
                        }
                    },
                );
                row(
                    "A",
                    "C",
                    Pattern::Uniform,
                    d,
                    tn,
                    31,
                    2,
                    cache,
                    "read_ns",
                    ns,
                );
            }
            let l0 = lanes[0];
            let planes: Vec<&[u64]> = cs
                .iter()
                .map(|c| c.tables[l0].as_ref().unwrap().present.as_slice())
                .collect();
            let prog = or_count_program(tn as u16);
            let ns = time_batches(
                WORDS,
                || {},
                || {
                    black_box(run_count(&prog, black_box(&planes)));
                },
            );
            row(
                "B",
                "C",
                Pattern::Uniform,
                d,
                tn,
                31,
                1,
                "hot",
                "session_or_ns_per_word",
                ns,
            );
        }
    }
}

/// k = 2, U = 31, one thought, every spatial pattern.
fn spatial_axis(base: &[u8], ev: &mut [u64]) {
    let hot = addrs_hot();
    let cold = addrs_cold();
    for &d in &[0.01, 0.1] {
        for p in [
            Pattern::Uniform,
            Pattern::Clustered,
            Pattern::Contiguous,
            Pattern::Morton,
            Pattern::Disjoint,
            Pattern::Overlapping,
        ] {
            let w = writes(p, d, 0, 31);
            let c = vec![ThoughtC::seal(&w, Commit::Lazy)];
            let b = ThoughtB::seal(&w);
            let lanes = touched_lanes(31)[..2].to_vec();
            let r = resolve(
                &Selection {
                    thought: 0,
                    lanes: lanes.clone(),
                    fallback: Fallback::Base,
                },
                &c,
                base,
            );
            let mut out = vec![Got::Absent; 2];
            for (cache, addrs) in [("hot", &hot), ("cold", &cold)] {
                let cold_run = cache == "cold";
                let ns_c = time_batches(
                    READS,
                    || {
                        if cold_run {
                            evict(ev)
                        }
                    },
                    || {
                        for &i in addrs.iter() {
                            r.read(black_box(i), &mut out);
                            black_box(&out);
                        }
                    },
                );
                let ns_b = time_batches(
                    READS,
                    || {
                        if cold_run {
                            evict(ev)
                        }
                    },
                    || {
                        for &i in addrs.iter() {
                            b.read(&lanes, Fallback::Base, base, black_box(i), &mut out);
                            black_box(&out);
                        }
                    },
                );
                row("A", "C", p, d, 1, 31, 2, cache, "read_ns", ns_c);
                row("A", "B", p, d, 1, 31, 2, cache, "read_ns", ns_b);
            }
            let pa = &c[0].tables[lanes[0]].as_ref().unwrap().present;
            let pb = &c[0].tables[lanes[1]].as_ref().unwrap().present;
            let planes2: [&[u64]; 2] = [pa, pb];
            let ap = vec![u64::MAX; WORDS];
            let planes3: [&[u64]; 3] = [pa, pb, &ap];
            row(
                "B",
                "C",
                p,
                d,
                1,
                31,
                2,
                "-",
                "shared_count",
                run_count(&and_count_program(3), &planes3) as f64,
            );
            row(
                "B",
                "C",
                p,
                d,
                1,
                31,
                2,
                "-",
                "hamming",
                run_count(&xor_count_program(), &planes2) as f64,
            );
            row(
                "B",
                "C",
                p,
                d,
                1,
                31,
                2,
                "-",
                "pop_a",
                run_count(&or_count_program(1), &[pa.as_slice()]) as f64,
            );
        }
    }
}

/// Resident cost of C-lazy, C-eager and S, per (T, density, pattern), U = 31.
/// Each (cell, arm) runs in its own child process: in one process the
/// allocator hands freed pages of the previous cell to the next one, and the
/// RSS and fault deltas read zero.
fn memory_grid() {
    let exe = std::env::current_exe().expect("own path");
    for p in [Pattern::Uniform, Pattern::Clustered] {
        for &d in &[0.001, 0.01, 0.1, 1.0] {
            for &tn in &[1usize, 8, 64] {
                for arm in ["sizes", "C-lazy", "C-eager", "S"] {
                    let out = std::process::Command::new(&exe)
                        .args(["memcell", p.name(), &d.to_string(), &tn.to_string(), arm])
                        .output()
                        .expect("child");
                    print!("{}", String::from_utf8_lossy(&out.stdout));
                }
            }
        }
    }
}

fn memory_cell(p: &str, d: f64, tn: usize, arm: &str) {
    let p = [Pattern::Uniform, Pattern::Clustered]
        .into_iter()
        .find(|x| x.name() == p)
        .expect("pattern");
    let ws: Vec<Writes> = (0..tn).map(|t| writes(p, d, t, 31)).collect();
    let dirty: usize = ws
        .iter()
        .map(|w| w.lanes.iter().map(|(_, v)| v.len()).sum::<usize>())
        .sum();
    let tables = tn * 31;
    if arm == "sizes" {
        row(
            "M",
            "-",
            p,
            d,
            tn,
            31,
            0,
            "-",
            "logical_bytes",
            (tables * N * LANE) as f64,
        );
        row(
            "M",
            "-",
            p,
            d,
            tn,
            31,
            0,
            "-",
            "payload_bytes",
            (dirty * LANE) as f64,
        );
        row(
            "M",
            "-",
            p,
            d,
            tn,
            31,
            0,
            "-",
            "presence_bytes",
            (tables * WORDS * 8) as f64,
        );
        row(
            "M",
            "-",
            p,
            d,
            tn,
            31,
            0,
            "-",
            "sparse_persist_bytes",
            (dirty * (4 + LANE)) as f64,
        );
        return;
    }
    let m0 = mem();
    let t = Instant::now();
    let held: (Vec<ThoughtC>, Vec<ThoughtS>) = match arm {
        "C-lazy" => (
            ws.iter().map(|w| ThoughtC::seal(w, Commit::Lazy)).collect(),
            vec![],
        ),
        "C-eager" => (
            ws.iter()
                .map(|w| ThoughtC::seal(w, Commit::Eager))
                .collect(),
            vec![],
        ),
        _ => (vec![], ws.iter().map(ThoughtS::seal).collect()),
    };
    let seal_ns = t.elapsed().as_nanos() as f64;
    let m1 = mem();
    black_box(&held);
    let delta = |a: u64, b: u64| ((b as f64 - a as f64) * 1024.0).max(0.0);
    row(
        "M",
        arm,
        p,
        d,
        tn,
        31,
        0,
        "-",
        "virtual_delta_bytes",
        delta(m0.vm_kb, m1.vm_kb),
    );
    row(
        "M",
        arm,
        p,
        d,
        tn,
        31,
        0,
        "-",
        "rss_delta_bytes",
        delta(m0.rss_kb, m1.rss_kb),
    );
    row(
        "M",
        arm,
        p,
        d,
        tn,
        31,
        0,
        "-",
        "minor_faults",
        (m1.flt - m0.flt) as f64,
    );
    row(
        "M",
        arm,
        p,
        d,
        tn,
        31,
        0,
        "-",
        "seal_ns_per_dirty",
        if dirty > 0 {
            seal_ns / dirty as f64
        } else {
            seal_ns
        },
    );
}

// ------------------------------------------------------------- oracle tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A small fixture: thought 0 touches 31 lanes, uniform 10 %.
    fn fixture(u: usize) -> (Vec<u8>, Writes, Vec<ThoughtC>) {
        let base = base_rows();
        let w = writes(Pattern::Uniform, 0.1, 0, u);
        let c = vec![ThoughtC::seal(&w, Commit::Lazy)];
        (base, w, c)
    }

    fn reference(w: &Writes, base: &[u8], l: usize, i: usize, fb: Fallback) -> Got {
        for (ll, dirty) in &w.lanes {
            if *ll == l && dirty.binary_search(&(i as u32)).is_ok() {
                return Got::Overlay {
                    thought: w.thought as u16,
                    value: overlay_value(w.thought, l, i),
                };
            }
        }
        match fb {
            Fallback::Base => Got::Base(base_lane(base, i, l)),
            Fallback::Absent => Got::Absent,
        }
    }

    /// Oracles 1 and 2: every selected lane at every address equals the
    /// reference, on C, B and S.
    #[test]
    fn every_arm_reads_the_reference_at_every_address() {
        let (base, w, c) = fixture(31);
        let b = ThoughtB::seal(&w);
        let s = ThoughtS::seal(&w);
        let lanes = touched_lanes(31)[..4].to_vec();
        for fb in [Fallback::Base, Fallback::Absent] {
            let r = resolve(
                &Selection {
                    thought: 0,
                    lanes: lanes.clone(),
                    fallback: fb,
                },
                &c,
                &base,
            );
            let mut oc = vec![Got::Absent; 4];
            let mut ob = vec![Got::Absent; 4];
            let mut os = vec![Got::Absent; 4];
            for i in 0..N {
                r.read(i, &mut oc);
                b.read(&lanes, fb, &base, i, &mut ob);
                s.read(&lanes, fb, &base, i, &mut os);
                for (j, &l) in lanes.iter().enumerate() {
                    let want = reference(&w, &base, l, i, fb);
                    assert_eq!(oc[j], want, "C lane {l} at {i}");
                    assert_eq!(ob[j], want, "B lane {l} at {i}");
                    assert_eq!(os[j], want, "S lane {l} at {i}");
                }
            }
        }
    }

    /// Oracle 5 (and the core falsifier): a read consults exactly the
    /// requested tables, whatever the thought touched.
    #[test]
    fn tables_consulted_equal_requested_lanes_for_every_touched_count() {
        let base = base_rows();
        for u in [2usize, 8, 31] {
            let w = writes(Pattern::Uniform, 0.1, 0, u);
            let c = vec![ThoughtC::seal(&w, Commit::Lazy)];
            assert_eq!(c[0].touched(), u, "the fixture must touch {u} tables");
            let lanes = touched_lanes(u)[..2].to_vec();
            let r = resolve(
                &Selection {
                    thought: 0,
                    lanes,
                    fallback: Fallback::Absent,
                },
                &c,
                &base,
            );
            let mut out = vec![Got::Absent; 2];
            reset();
            for i in 0..1000 {
                r.read(i, &mut out);
            }
            let [consulted, presence, _, rows, tr] = counters();
            assert_eq!(consulted, 2 * 1000, "U = {u}");
            assert_eq!(presence, 2 * 1000, "U = {u}");
            assert_eq!(rows, 0);
            assert_eq!(tr, 0);
        }
    }

    /// Oracle 3: an overlay of sixteen zero bytes is not `Absent`.
    #[test]
    fn a_zero_overlay_is_not_absent() {
        let base = base_rows();
        let l = touched_lanes(1)[0];
        let w = Writes {
            thought: 0,
            lanes: vec![(l, vec![5])],
        };
        let mut c = vec![ThoughtC::seal(&w, Commit::Lazy)];
        c[0].tables[l].as_mut().unwrap().values[5] = [0; LANE];
        let r = resolve(
            &Selection {
                thought: 0,
                lanes: vec![l],
                fallback: Fallback::Absent,
            },
            &c,
            &base,
        );
        let mut out = [Got::Absent];
        r.read(5, &mut out);
        assert_eq!(
            out[0],
            Got::Overlay {
                thought: 0,
                value: [0; LANE]
            }
        );
        r.read(6, &mut out);
        assert_eq!(out[0], Got::Absent);
    }

    /// Oracle 4: overlay beats base; base only when the selection asks.
    #[test]
    fn overlay_precedes_base_and_fallback_is_explicit() {
        let (base, w, c) = fixture(4);
        let l = touched_lanes(4)[0];
        let dirty = &w.lanes[0].1;
        let hit = dirty[0] as usize;
        let miss = (0..N)
            .find(|i| dirty.binary_search(&(*i as u32)).is_err())
            .unwrap();
        let mut out = [Got::Absent];
        let rb = resolve(
            &Selection {
                thought: 0,
                lanes: vec![l],
                fallback: Fallback::Base,
            },
            &c,
            &base,
        );
        rb.read(hit, &mut out);
        assert!(matches!(out[0], Got::Overlay { .. }));
        rb.read(miss, &mut out);
        assert_eq!(out[0], Got::Base(base_lane(&base, miss, l)));
        let ra = resolve(
            &Selection {
                thought: 0,
                lanes: vec![l],
                fallback: Fallback::Absent,
            },
            &c,
            &base,
        );
        ra.read(miss, &mut out);
        assert_eq!(out[0], Got::Absent);
    }

    /// Oracles 6 and 7: mask-risc populations equal a scalar bit loop.
    #[test]
    fn populations_equal_the_scalar_reference() {
        for p in [
            Pattern::Uniform,
            Pattern::Clustered,
            Pattern::Disjoint,
            Pattern::Overlapping,
        ] {
            let w = writes(p, 0.1, 0, 4);
            let c = ThoughtC::seal(&w, Commit::Lazy);
            let l = touched_lanes(4);
            let pa = &c.tables[l[0]].as_ref().unwrap().present;
            let pb = &c.tables[l[1]].as_ref().unwrap().present;
            let ap: Vec<u64> = (0..WORDS).map(|j| mix(0xA9 ^ j as u64)).collect();
            let three: [&[u64]; 3] = [pa, pb, &ap];
            assert_eq!(
                run_count(&and_count_program(3), &three),
                scalar_and(&three),
                "{p:?}"
            );
            let two: [&[u64]; 2] = [pa, pb];
            assert_eq!(
                run_count(&xor_count_program(), &two),
                scalar_xor(pa, pb),
                "{p:?}"
            );
            let or = (0..N).filter(|&i| bit(pa, i) || bit(pb, i)).count();
            assert_eq!(run_count(&or_count_program(2), &two), or, "{p:?}");
            // weight of one plane equals its dirty count
            assert_eq!(
                run_count(&or_count_program(1), &[pa.as_slice()]),
                w.lanes[0].1.len()
            );
        }
        // the disjoint pattern really is disjoint, the overlapping one overlaps
        let wd = writes(Pattern::Disjoint, 0.5, 0, 4);
        let cd = ThoughtC::seal(&wd, Commit::Lazy);
        let l = touched_lanes(4);
        let (a, b) = (
            &cd.tables[l[0]].as_ref().unwrap().present,
            &cd.tables[l[1]].as_ref().unwrap().present,
        );
        assert_eq!(scalar_and(&[a.as_slice(), b.as_slice()]), 0);
        assert!(scalar_xor(a, b) > 0);
    }

    /// Oracle 6 (C path): a population query loads no value.
    #[test]
    fn a_population_query_loads_no_value() {
        let (_, _, c) = fixture(4);
        let l = touched_lanes(4);
        let pa = &c[0].tables[l[0]].as_ref().unwrap().present;
        let pb = &c[0].tables[l[1]].as_ref().unwrap().present;
        reset();
        black_box(run_count(
            &and_count_program(2),
            &[pa.as_slice(), pb.as_slice()],
        ));
        assert_eq!(counters(), [0; 5]);
    }

    /// Oracle 11: point reads build no row; the terminal builds exactly the
    /// rows asked for, and equals base plus every overlay.
    #[test]
    fn rows_are_built_only_by_the_terminal() {
        let (base, w, c) = fixture(31);
        let r = resolve(
            &Selection {
                thought: 0,
                lanes: touched_lanes(31)[..2].to_vec(),
                fallback: Fallback::Base,
            },
            &c,
            &base,
        );
        let mut out = vec![Got::Absent; 2];
        reset();
        for i in 0..4096 {
            r.read(i, &mut out);
        }
        assert_eq!(counters()[3], 0);
        for i in 0..7 {
            let row = materialize_row(&c[0], &base, i);
            for l in 0..LANES {
                let want = match reference(&w, &base, l, i, Fallback::Base) {
                    Got::Overlay { value, .. } | Got::Base(value) => value,
                    _ => unreachable!(),
                };
                assert_eq!(&row[l * LANE..(l + 1) * LANE], &want);
            }
        }
        assert_eq!(counters()[3], 7);
    }

    /// Oracle 12: two thoughts at one coordinate keep their own values.
    #[test]
    fn two_thoughts_at_one_coordinate_stay_apart() {
        let base = base_rows();
        let l = touched_lanes(1)[0];
        let c: Vec<ThoughtC> = (0..2)
            .map(|t| {
                ThoughtC::seal(
                    &Writes {
                        thought: t,
                        lanes: vec![(l, vec![9])],
                    },
                    Commit::Lazy,
                )
            })
            .collect();
        let mut out = [Got::Absent];
        for t in 0..2 {
            resolve(
                &Selection {
                    thought: t,
                    lanes: vec![l],
                    fallback: Fallback::Absent,
                },
                &c,
                &base,
            )
            .read(9, &mut out);
            assert_eq!(
                out[0],
                Got::Overlay {
                    thought: t as u16,
                    value: overlay_value(t, l, 9)
                }
            );
        }
        assert_ne!(overlay_value(0, l, 9), overlay_value(1, l, 9));
    }

    /// Oracle 13: a denied address loads nothing.
    #[test]
    fn a_denied_address_loads_nothing() {
        let (base, _, c) = fixture(4);
        let r = resolve(
            &Selection {
                thought: 0,
                lanes: touched_lanes(4)[..2].to_vec(),
                fallback: Fallback::Base,
            },
            &c,
            &base,
        );
        let mut ap = vec![0u64; WORDS];
        ap[0] = 1; // only address 0 admitted
        let mut out = vec![Got::Absent; 2];
        reset();
        r.read_under(&ap, 1, &mut out);
        assert_eq!(out, vec![Got::Denied; 2]);
        assert_eq!(
            counters()[2],
            0,
            "no value may be loaded for a denied address"
        );
        r.read_under(&ap, 0, &mut out);
        assert!(out.iter().all(|g| !matches!(g, Got::Denied)));
    }

    /// The footprint instrument reads exactly the planes and lanes named.
    #[test]
    fn lane_footprint_reads_the_program_selection() {
        let (planes, lanes) = lane_footprint(&and_count_program(3));
        assert_eq!(planes.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(lanes.is_empty());
        let p = Program::new(
            vec![lance_graph_mask_risc::MaskOp::Pred {
                pred: lance_graph_mask_risc::Pred::GtI32 { lane: 7, t: 3 },
                under: Some(MOp::Plane(4)),
                dst: 0,
            }],
            Terminal::MaskedSumI32 {
                mask: MOp::Scratch(0),
                lane: 12,
            },
        );
        let (planes, lanes) = lane_footprint(&p);
        assert_eq!(planes.into_iter().collect::<Vec<_>>(), vec![4]);
        assert_eq!(lanes.into_iter().collect::<Vec<_>>(), vec![7, 12]);
    }

    /// The fixtures do what their names say (anti-vacuity).
    #[test]
    fn fixtures_have_the_stated_shape() {
        let u = writes(Pattern::Uniform, 0.01, 0, 2);
        let n = u.lanes[0].1.len();
        assert!((400..900).contains(&n), "uniform 1 % of 65,536: {n}");
        let cl = writes(Pattern::Clustered, 0.01, 0, 2);
        assert!(cl.lanes[0].1.chunks(64).all(|r| r[63] - r[0] == 63));
        let ct = writes(Pattern::Contiguous, 0.01, 0, 2);
        let v = &ct.lanes[0].1;
        assert_eq!(v.len(), 655);
        assert_eq!(v[v.len() - 1] - v[0], 654);
        let mo = writes(Pattern::Morton, 0.01, 0, 2);
        let blocks: std::collections::BTreeSet<u32> =
            mo.lanes[0].1.iter().map(|i| i / 256).collect();
        assert!(blocks.len() <= 6, "morton: {} blocks", blocks.len());
        assert_eq!(writes(Pattern::Uniform, 0.0, 0, 2).lanes[0].1.len(), 0);
        assert_eq!(writes(Pattern::Uniform, 1.0, 0, 2).lanes[0].1.len(), N);
    }
}
