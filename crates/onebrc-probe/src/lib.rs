//! `onebrc-probe` — 1BRC (One Billion Row Challenge) substrate probe.
//!
//! Measures groupby-aggregate throughput (the classic 1BRC workload: parse
//! `station;temp` lines, aggregate min/max/sum/count per station) as a
//! stand-in for the V3 substrate's own aggregation paths, at container
//! scale (100M rows). This crate is standalone and workspace-excluded (see
//! root `Cargo.toml`); it ships two baseline lanes:
//!
//! - **Lane A** (`lane_a_scalar`) — single-thread scalar baseline.
//! - **Lane C** (`lane_c_threads`) — `std::thread` parallel baseline,
//!   newline-aligned chunk split + commutative merge.
//!
//! - **Lane B** (`lane_b::lane_b_simd`, feature `lane-b`) — `ndarray::simd`
//!   vectorized `;`/`\n` scan, scalar parse.
//! - **Lane D** (`lane_d::lane_d_ractor`, feature `lane-d`) — `ractor`
//!   actor-per-worker over the same `chunk_bounds` split as Lane C.
//! - **Lane E** (`lane_e::lane_e_kanban`, feature `lane-e`) — kanban-scheduled
//!   batches: a shared `AtomicUsize` batch queue, one fresh `KanbanActor` per
//!   batch driven through the full Rubicon forward arc
//!   (Planning->CognitiveWork->Evaluation->Commit) around the actual work.
//!   Measures the V3 kanban scheduling/journaling tax (E-D isolates the
//!   journaling cost; fine-grained batching prices per-card scheduling).
//! - **Lanes F/R** (`lane_f::{lane_f_morton, lane_r_radix}`, std-only, no
//!   feature) — the substrate-native lane and its honest control: station
//!   identity → Morton tile address → SoA-shaped flat accumulators (F);
//!   identical pipeline with a plain-radix slot (R). F−R isolates the
//!   addressing tax; R−C prices flat-table-vs-BTreeMap.
//! - **Lane G** (`lane_g::lane_g_kanban_soa`, feature `lane-g`) — the
//!   kanban-update write path: the same Morton-tile 64K SoA held as OWNED
//!   state by shard mailbox actors (prefix-routed morsel casts, every
//!   applied batch witnessed with a `KanbanMove`). G−F prices witnessed
//!   streamed ownership vs private-merge-at-end; the `shards` sweep is
//!   the 64K-concurrent-SoA-vs-Morton-tile ownership ledger.
//!
//! ## Reference inventory
//!
//! Techniques below are REIMPLEMENTED from reading automataIA/1brc-rs (a
//! reference to study, never a dependency — see README §1 for the full
//! inventory + file pointers):
//!
//! - Newline-aligned chunk splitting for parallel work distribution
//!   (`chunk_bounds`, mirrors the reference's function of the same name).
//! - Integer (never float) temperature parsing in the hot loop.
//! - `min`/`max`/`sum`/`count` per-station `Stats` aggregate with a
//!   commutative `merge`.
//!
//! NOT reimplemented here (left to Lane B / follow-up): mmap zero-copy
//! file access, `FxHash` + `raw_entry_mut` hashing, the merykitty SWAR
//! branchless parser, and SIMD semicolon-finding (`pulp` / `wide`).

pub mod gen;
#[cfg(feature = "lane-b")]
pub mod lane_b;
#[cfg(feature = "lane-d")]
pub mod lane_d;
#[cfg(feature = "lane-e")]
pub mod lane_e;
pub mod lane_f;
#[cfg(feature = "lane-g")]
pub mod lane_g;
#[cfg(feature = "lane-h")]
pub mod lane_h;
#[cfg(feature = "lane-i")]
pub mod lane_i;
pub mod sha256;

#[cfg(feature = "lane-b")]
pub use lane_b::lane_b_simd;
#[cfg(feature = "lane-d")]
pub use lane_d::lane_d_ractor;
#[cfg(feature = "lane-e")]
pub use lane_e::lane_e_kanban;
pub use lane_f::{lane_f_morton, lane_r_radix};
#[cfg(feature = "lane-g")]
pub use lane_g::{lane_g_kanban_soa, lane_g_kanban_soa_with_morsel};
#[cfg(feature = "lane-h")]
pub use lane_h::{lane_h_orchestrated, lane_h_orchestrated_with};
#[cfg(feature = "lane-i")]
pub use lane_i::{lane_i_batch_pipeline, lane_i_batch_pipeline_with};

use std::collections::BTreeMap;

/// Per-station aggregate: min/max/sum/count, in tenths-of-a-degree.
///
/// `merge` is commutative and associative — the workspace's owned-microcopy
/// / gated-commutative-merge borrow-strategy rule (see
/// `.claude/rules/borrow-strategy.md`, "Multiple writers -> BUNDLE (majority
/// vote, commutative)"): each Lane-C worker computes an OWNED `Stats` value
/// independently (never a shared `&mut` reference across threads), and
/// `merge` combines two owned values into one via min/max/sum/count — never
/// a raw `=` assignment onto shared state. Merge order never matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub min: i32,
    pub max: i32,
    pub sum: i64,
    pub count: u32,
}

impl Stats {
    /// A fresh aggregate seeded with one observation.
    pub fn single(tenths: i32) -> Self {
        Self {
            min: tenths,
            max: tenths,
            sum: tenths as i64,
            count: 1,
        }
    }

    /// Fold one more observation into this (owned) aggregate.
    pub fn observe(&mut self, tenths: i32) {
        if tenths < self.min {
            self.min = tenths;
        }
        if tenths > self.max {
            self.max = tenths;
        }
        self.sum += tenths as i64;
        self.count += 1;
    }

    /// Commutative, associative merge of two owned aggregates. See the
    /// struct-level doc comment for why this is the only legal write-back
    /// shape for multi-writer (Lane C) accumulation.
    pub fn merge(&mut self, other: &Stats) {
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.sum += other.sum;
        self.count += other.count;
    }

    /// Mean, in whole degrees (tenths / 10).
    pub fn mean_tenths(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f64 / self.count as f64
        }
    }
}

/// Manual integer parse of a `[-]D[D].D` temperature field into tenths of a
/// degree — no float in the hot loop. `bytes` must be exactly the temp
/// field (no leading/trailing `;`/`\n`), matching the `gen::gen` output
/// format and the merykitty-SWAR-parser family's "parse to an integer,
/// never to f64" shape (see module doc "Reference inventory").
fn parse_temp_tenths(bytes: &[u8]) -> i32 {
    let mut i = 0usize;
    let neg = bytes[0] == b'-';
    if neg {
        i += 1;
    }
    let mut val: i32 = 0;
    while bytes[i] != b'.' {
        val = val * 10 + (bytes[i] - b'0') as i32;
        i += 1;
    }
    i += 1; // skip '.'
    val = val * 10 + (bytes[i] - b'0') as i32;
    if neg {
        -val
    } else {
        val
    }
}

/// Lane A — single-thread scalar baseline. One pass over `data`, byte-wise
/// scan for `;` and `\n`, integer temp parse, `BTreeMap<String, Stats>`
/// accumulation (owned per-station microcopies — see `Stats::merge` doc).
pub fn lane_a_scalar(data: &[u8]) -> BTreeMap<String, Stats> {
    let mut map: BTreeMap<String, Stats> = BTreeMap::new();
    let len = data.len();
    let mut i = 0usize;
    while i < len {
        let name_start = i;
        while data[i] != b';' {
            i += 1;
        }
        let name = std::str::from_utf8(&data[name_start..i]).expect("station name is valid utf8");
        i += 1; // skip ';'
        let temp_start = i;
        while data[i] != b'\n' {
            i += 1;
        }
        let tenths = parse_temp_tenths(&data[temp_start..i]);
        i += 1; // skip '\n'

        match map.get_mut(name) {
            Some(stats) => stats.observe(tenths),
            None => {
                map.insert(name.to_string(), Stats::single(tenths));
            }
        }
    }
    map
}

/// Split `data` into `n` byte ranges aligned on `\n` boundaries — each
/// `(start, end)` is a `[start, end)` half-open range that always ends
/// immediately after a newline (or at `data.len()`), so no record straddles
/// a chunk boundary. Mirrors automataIA/1brc-rs's `chunk_bounds` function
/// (reimplemented, not vendored — see README §1).
pub fn chunk_bounds(data: &[u8], n: usize) -> Vec<(usize, usize)> {
    let len = data.len();
    if n <= 1 || len == 0 {
        return vec![(0, len)];
    }
    let mut bounds = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..n {
        if i == n - 1 {
            bounds.push((start, len));
            break;
        }
        let target = (len / n) * (i + 1);
        let mut end = target.min(len);
        while end < len && data[end] != b'\n' {
            end += 1;
        }
        if end < len {
            end += 1; // include the newline itself in this chunk
        }
        bounds.push((start, end));
        start = end;
    }
    bounds
}

/// Commutative merge of N owned per-worker maps into one — the multi-writer
/// BUNDLE step (see `Stats::merge` doc); order of the input `Vec` never
/// affects the result.
fn merge_maps(maps: Vec<BTreeMap<String, Stats>>) -> BTreeMap<String, Stats> {
    let mut out: BTreeMap<String, Stats> = BTreeMap::new();
    for m in maps {
        for (name, stats) in m {
            match out.get_mut(&name) {
                Some(existing) => existing.merge(&stats),
                None => {
                    out.insert(name, stats);
                }
            }
        }
    }
    out
}

/// Lane C — `std::thread` parallel baseline. Splits `data` into `workers`
/// newline-aligned chunks (`chunk_bounds`), each worker runs `lane_a_scalar`
/// on its OWN slice producing an owned `BTreeMap<String, Stats>` (no shared
/// `&mut` state across threads — per-worker microcopies), then merges all
/// worker maps via the commutative `Stats::merge` (order-independent
/// BUNDLE; see struct-level doc on `Stats`).
pub fn lane_c_threads(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let bounds = chunk_bounds(data, workers);
    let results: Vec<BTreeMap<String, Stats>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                scope.spawn(move || lane_a_scalar(slice))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("lane C worker panicked"))
            .collect()
    });
    merge_maps(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_commutative_and_associative() {
        let a = Stats {
            min: -50,
            max: 100,
            sum: 500,
            count: 10,
        };
        let b = Stats {
            min: -80,
            max: 60,
            sum: -200,
            count: 5,
        };
        let c = Stats {
            min: 0,
            max: 40,
            sum: 120,
            count: 3,
        };

        let mut ab = a;
        ab.merge(&b);
        let mut ba = b;
        ba.merge(&a);
        assert_eq!(ab, ba, "merge must be commutative");

        let mut ab_c = ab;
        ab_c.merge(&c);
        let mut a_bc = a;
        let mut bc = b;
        bc.merge(&c);
        a_bc.merge(&bc);
        assert_eq!(ab_c, a_bc, "merge must be associative");
    }

    #[test]
    fn parse_temp_tenths_examples() {
        assert_eq!(parse_temp_tenths(b"0.0"), 0);
        assert_eq!(parse_temp_tenths(b"5.3"), 53);
        assert_eq!(parse_temp_tenths(b"-5.3"), -53);
        assert_eq!(parse_temp_tenths(b"99.9"), 999);
        assert_eq!(parse_temp_tenths(b"-99.9"), -999);
        assert_eq!(parse_temp_tenths(b"12.0"), 120);
    }

    #[test]
    fn chunk_bounds_covers_data_exactly_and_ends_on_newlines() {
        let data = b"aa;1.0\nbb;2.0\ncc;3.0\ndd;4.0\nee;5.0\n".to_vec();
        let bounds = chunk_bounds(&data, 3);
        assert_eq!(bounds.first().unwrap().0, 0);
        assert_eq!(bounds.last().unwrap().1, data.len());
        // Every boundary (except the very end) must land right after a '\n'.
        for &(_, end) in &bounds {
            if end < data.len() {
                assert_eq!(data[end - 1], b'\n');
            }
        }
        // Ranges must be contiguous, non-overlapping.
        for w in bounds.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    /// Lane A and Lane C must agree byte-for-byte on aggregate output —
    /// the correctness spot check for the parallel split + merge path.
    #[test]
    fn lane_a_and_lane_c_agree_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_{}.txt", std::process::id()));
        let result = gen::gen(&path, 100_000, 42).expect("gen");
        assert_eq!(result.rows, 100_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = lane_a_scalar(&data);
        let c = lane_c_threads(&data, 4);
        assert_eq!(a, c, "lane A and lane C must produce identical aggregates");
        assert!(!a.is_empty());
    }

    /// Same seed => same corpus bytes (checksum equality) — the recipe
    /// contract (`rows`, `seed`, `sha256`) must be reproducible.
    #[test]
    fn generator_is_deterministic() {
        let dir = std::env::temp_dir();
        let p1 = dir.join(format!("onebrc_probe_det_1_{}.txt", std::process::id()));
        let p2 = dir.join(format!("onebrc_probe_det_2_{}.txt", std::process::id()));
        let r1 = gen::gen(&p1, 10_000, 42).expect("gen 1");
        let r2 = gen::gen(&p2, 10_000, 42).expect("gen 2");
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
        assert_eq!(
            r1.sha256_hex, r2.sha256_hex,
            "same seed must produce same sha256"
        );
    }

    /// Lane A and Lane B must agree byte-for-byte on aggregate output over
    /// a generated corpus — the correctness spot check for the SIMD
    /// delimiter scan.
    #[cfg(feature = "lane-b")]
    #[test]
    fn lane_b_agrees_with_lane_a_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_b_{}.txt", std::process::id()));
        let result = gen::gen(&path, 50_000, 7).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = lane_a_scalar(&data);
        let b = lane_b_simd(&data);
        assert_eq!(a, b, "lane A and lane B must produce identical aggregates");
        assert!(!a.is_empty());
    }

    /// Hand-built corpus, crafted so at least one record's `;`/`\n` land in
    /// DIFFERENT 32-byte SIMD blocks (block0=[0,32), block1=[32,64),
    /// tail=[64,..)) — exercises the cross-iteration `line_start` /
    /// `pending_semi` carry in `lane_b_simd`, not just the common in-block
    /// case a random generated corpus would mostly hit.
    #[cfg(feature = "lane-b")]
    #[test]
    fn lane_b_handles_records_that_straddle_32_byte_block_boundaries() {
        let long_name = "N".repeat(22);
        let mut corpus = String::new();
        corpus.push_str("Ab;1.0\n"); // fully inside block0
        corpus.push_str(&format!("{long_name};9.9\n")); // straddles block0/block1
        corpus.push_str("Zz;3.3\n"); // fully inside block1
        corpus.push_str("QqRrSsTt;2.2\n"); // fully inside block1
        corpus.push_str("Uu;4.4\n"); // fully inside block1
        corpus.push_str("Vv;5.5\n"); // straddles block1/tail

        let data = corpus.as_bytes();
        assert!(
            data.len() > 64,
            "test corpus must span block0, block1, AND a tail region"
        );

        // Confirm (rather than assume) that at least one record's `;` and
        // `\n` land in different 32-byte blocks — otherwise this test
        // would silently degrade into testing only the non-crossing case.
        let find_all = |needle: u8| -> Vec<usize> {
            data.iter()
                .enumerate()
                .filter(|&(_, &b)| b == needle)
                .map(|(i, _)| i)
                .collect()
        };
        let semis = find_all(b';');
        let newlines = find_all(b'\n');
        assert_eq!(semis.len(), newlines.len());
        let crosses_a_block = semis
            .iter()
            .zip(newlines.iter())
            .any(|(&s, &n)| s / 32 != n / 32);
        assert!(
            crosses_a_block,
            "test corpus must contain a record whose `;`/`\\n` land in different 32-byte blocks"
        );

        let a = lane_a_scalar(data);
        let b = lane_b_simd(data);
        assert_eq!(
            a, b,
            "lane B must agree with lane A across straddled block boundaries"
        );

        let mut expected: BTreeMap<String, Stats> = BTreeMap::new();
        expected.insert("Ab".to_string(), Stats::single(10));
        expected.insert(long_name, Stats::single(99));
        expected.insert("Zz".to_string(), Stats::single(33));
        expected.insert("QqRrSsTt".to_string(), Stats::single(22));
        expected.insert("Uu".to_string(), Stats::single(44));
        expected.insert("Vv".to_string(), Stats::single(55));
        assert_eq!(
            b, expected,
            "lane B stats must match the hand-computed expectation"
        );
    }

    /// Lane A and Lane D must agree byte-for-byte on aggregate output —
    /// the correctness spot check for the ractor actor-per-worker path.
    /// Uses an odd worker count (3) on purpose, mirroring `chunk_bounds`'s
    /// own odd-split test coverage.
    #[cfg(feature = "lane-d")]
    #[test]
    fn lane_d_agrees_with_lane_a_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_d_{}.txt", std::process::id()));
        let result = gen::gen(&path, 50_000, 13).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = lane_a_scalar(&data);
        let d = lane_d_ractor(&data, 3);
        assert_eq!(a, d, "lane A and lane D must produce identical aggregates");
        assert!(!a.is_empty());
    }

    /// Lane A and Lane E must agree byte-for-byte on aggregate output — the
    /// correctness spot check for the kanban-scheduled-batches path. Uses
    /// `workers=3, batches=7` (odd, greater than workers) so the shared
    /// `AtomicUsize` batch queue actually round-robins multiple batches per
    /// puller, not just one batch per worker (the `batches == workers`
    /// degenerate case lane D already covers via its own test).
    #[cfg(feature = "lane-e")]
    #[test]
    fn lane_e_agrees_with_lane_a_on_generated_corpus() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_e_{}.txt", std::process::id()));
        let result = gen::gen(&path, 50_000, 21).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = lane_a_scalar(&data);
        let e = lane_e_kanban(&data, 3, 7);
        assert_eq!(a, e, "lane A and lane E must produce identical aggregates");
        assert!(!a.is_empty());
    }
}
