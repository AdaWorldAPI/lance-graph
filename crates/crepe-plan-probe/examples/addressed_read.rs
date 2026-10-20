//! Arm B's row read by strategy and demand selectivity, to place the switch
//! from exact ranges to a coalesced or full read. Run in release from this
//! crate's directory:
//!
//! ```text
//! cargo run --release --example addressed_read
//! ```
//!
//! The fixture is the one of `examples/addressed.rs`: `Answer(17, z)` over
//! `family`, `READ_N` (default 50,000) background edges with distinct `y`, and `k`
//! children of 17 spread over them, so `k` centers and `k` answers. Every
//! strategy is checked against that count before it is timed.
//!
//! Strategies, each a median of `READ_REPS` (default 9) warm runs, lanes
//! attested once. `READ_KS=1000,2000` restricts the sweep to those `k`:
//! - `exact`: the located ranges only (`ReadPolicy::EXACT`);
//! - `gap G`: ranges merged across gaps of at most `G` rows, then the extra
//!   rows dropped by their leading value (no full read);
//! - `full`: the whole table, then the same drop (`ReadPolicy::FULL`);
//! - `auto`: the pinned `ReadPolicy::DEFAULT`.

use std::time::{Duration, Instant};

use crepe_plan_probe::addressed::{AddressedRelation, Read, ReadPolicy, attest};
use crepe_plan_probe::lance_store::LanceStore;
use crepe_plan_probe::logic::{Labels, Rule, resolve};
use crepe_plan_probe::optimize::{Demand, Options, ResultKind};
use crepe_plan_probe::pipeline::{Outcome, evaluate_addressed_with};
use crepe_plan_probe::programs::family;
use mork_pathmap_probe::Semantics;

/// Background rows (`READ_N`, default 50,000).
fn n_rows() -> u32 {
    std::env::var("READ_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000)
}
const KIND: ResultKind = ResultKind::Count(Semantics::Bag);

/// Runs per strategy (`READ_REPS`, default 9).
fn reps() -> usize {
    std::env::var("READ_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Median time, the strategy of the side with the most centers, and the
/// ranges it handed to Lance.
async fn run(
    rules: &[Rule],
    d: &Demand,
    st: &LanceStore,
    lanes: &[AddressedRelation],
    policy: ReadPolicy,
    k: u64,
) -> (Duration, Read, usize) {
    let mut t = Vec::with_capacity(reps());
    let mut how = (Read::Ranges, 0);
    for _ in 0..reps() {
        let s = Instant::now();
        let (out, _, _, lookups) =
            evaluate_addressed_with(rules, d, KIND, Options::ALL, st, lanes, policy)
                .await
                .unwrap();
        t.push(s.elapsed());
        assert_eq!(out, Outcome::Count(k), "policy {policy:?} at k={k}");
        let widest = lookups
            .iter()
            .flatten()
            .max_by_key(|l| l.keys)
            .expect("a side");
        how = (widest.read, widest.ranges_read);
    }
    (median(t), how.0, how.1)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let plan = family::crepe_plan();
    let labels = Labels::of(&plan);
    let rules: Vec<Rule> = resolve(&plan, &labels)
        .into_iter()
        .map(|r| r.expect("admitted rule"))
        .collect();
    let rels = [
        labels.id("Parent").unwrap(),
        labels.id("Allowed").unwrap(),
        labels.id("Sponsor").unwrap(),
    ];
    let d = Demand {
        goal: labels.id("Answer").unwrap(),
        bound: vec![Some(17), None],
    };
    let n = n_rows();
    let gaps: [u32; 3] = [4, 16, 64];
    // `READ_KS=1000,2000` measures only those k.
    let ks: Vec<u32> = match std::env::var("READ_KS") {
        Ok(v) => v
            .split(',')
            .map(|k| k.trim().parse().expect("READ_KS"))
            .collect(),
        Err(_) => vec![
            0,
            1,
            8,
            n / 200,
            n / 100,
            n / 50,
            n / 33,
            n / 20,
            n / 14,
            n / 10,
            n / 2,
            n,
        ],
    };

    print!("{:>7} | {:>8}", "k", "exact");
    for g in gaps {
        print!(" {:>8}", format!("gap {g}"));
    }
    println!(
        " {:>8} | {:>8} {:>10} {:>7}",
        "full", "auto", "auto read", "ranges"
    );
    for k in ks {
        let mut parent: Vec<[u32; 2]> = (0..n).map(|i| [1_000_000 + i, 3_000_000 + i]).collect();
        let step = n.checked_div(k).unwrap_or(1);
        parent.extend((0..k).map(|j| [17, 1_000_000 + j * step]));
        let allowed: Vec<[u32; 2]> = (0..n).map(|i| [1_000_000 + i, 1_000_000 + i]).collect();
        let sponsor: Vec<[u32; 2]> = vec![[20, 20], [21, 21]];
        let dir = tempfile::tempdir().unwrap();
        let st = LanceStore::write(
            dir.path().to_str().unwrap(),
            &[
                (rels[0], "parent", &parent),
                (rels[1], "allowed", &allowed),
                (rels[2], "sponsor", &sponsor),
            ],
        )
        .await
        .unwrap();
        let mut lanes = Vec::new();
        for &r in &rels {
            lanes.push(attest(&st, r).await.unwrap());
        }
        let k64 = u64::from(k);

        let (exact, ..) = run(&rules, &d, &st, &lanes, ReadPolicy::EXACT, k64).await;
        print!("{:>7} | {:>8.3}", k, ms(exact));
        for g in gaps {
            let p = ReadPolicy {
                coalesce_gap: g,
                full_permille: u32::MAX,
            };
            let (t, ..) = run(&rules, &d, &st, &lanes, p, k64).await;
            print!(" {:>8.3}", ms(t));
        }
        let (full, ..) = run(&rules, &d, &st, &lanes, ReadPolicy::FULL, k64).await;
        let (auto, how, nr) = run(&rules, &d, &st, &lanes, ReadPolicy::DEFAULT, k64).await;
        println!(
            " {:>8.3} | {:>8.3} {:>10} {:>7}",
            ms(full),
            ms(auto),
            format!("{how:?}"),
            nr
        );
    }
}
