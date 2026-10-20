//! Arm A (scan semijoin) against Arm B (addressed semijoin) over Lance 13
//! tables, by demand selectivity. Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example addressed
//! ```
//!
//! The query is `Answer(17, z)` over the `family` program. `Parent` holds
//! `N = 50,000` background edges `[y, z]` with distinct `y`, every `y` is
//! `Allowed`, and `x = 17` has `k` children chosen among those `y`. So the
//! semijoin centers are exactly `k` and the answer is `k` rows.
//!
//! Modes, each a median of `REPS` runs:
//! - `full`: semijoin off; every unrestricted side is read in full.
//! - `arm A`: the scan semijoin; each side reads its key column in full,
//!   builds the address mask and reads the masked rows.
//! - `arm B warm`: the addressed semijoin with the lanes already attested.
//! - `attest`: reading the keys of the three relations and attesting them
//!   (Arm B's one-time cost `P`; `arm B cold` = `attest` + `arm B warm`).
//!
//! Break-even: Arm B's attest pays for itself after `ceil(P / S)` queries,
//! where `S` is Arm A's per-query time minus Arm B warm's.

use std::time::{Duration, Instant};

use crepe_plan_probe::addressed::{AddressedRelation, attest};
use crepe_plan_probe::lance_store::{LanceStore, SideIo};
use crepe_plan_probe::logic::{Labels, Rule, resolve};
use crepe_plan_probe::optimize::{Demand, Options, ResultKind};
use crepe_plan_probe::pipeline::{Outcome, evaluate_addressed, evaluate_lance};
use crepe_plan_probe::programs::family;
use mork_pathmap_probe::Semantics;

const N: u32 = 50_000;
const REPS: usize = 9;
const NO_SEMIJOIN: Options = Options {
    propagate: true,
    dedup: true,
    semijoin: false,
};
const KIND: ResultKind = ResultKind::Count(Semantics::Bag);

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[derive(Default, Clone, Copy)]
struct Read {
    rows: usize,
    bytes: usize,
}

fn read_of(io: &[Vec<SideIo>]) -> Read {
    let mut r = Read::default();
    for s in io.iter().flatten() {
        r.rows += s.rows.rows_scanned;
        r.bytes += s.rows.bytes_read;
        if let Some(c) = s.mask_column {
            r.rows += c.rows_scanned;
            r.bytes += c.bytes_read;
        }
    }
    r
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Mode {
    time: Duration,
    prepare: Duration,
    read: Read,
}

async fn lance_mode(rules: &[Rule], d: &Demand, opts: Options, st: &LanceStore, k: u64) -> Mode {
    let mut times = Vec::new();
    let mut preps = Vec::new();
    let mut read = Read::default();
    for _ in 0..REPS {
        let t = Instant::now();
        let (out, rep, io) = evaluate_lance(rules, d, KIND, opts, st).await.unwrap();
        times.push(t.elapsed());
        preps.push(rep.costs.prepare);
        assert_eq!(out, Outcome::Count(k));
        read = read_of(&io);
    }
    Mode {
        time: median(times),
        prepare: median(preps),
        read,
    }
}

async fn arm_b(
    rules: &[Rule],
    d: &Demand,
    st: &LanceStore,
    lanes: &[AddressedRelation],
    k: u64,
) -> Mode {
    let mut times = Vec::new();
    let mut preps = Vec::new();
    let mut read = Read::default();
    for _ in 0..REPS {
        let t = Instant::now();
        let (out, rep, io, _) = evaluate_addressed(rules, d, KIND, Options::ALL, st, lanes)
            .await
            .unwrap();
        times.push(t.elapsed());
        preps.push(rep.costs.prepare);
        assert_eq!(out, Outcome::Count(k));
        read = read_of(&io);
    }
    Mode {
        time: median(times),
        prepare: median(preps),
        read,
    }
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
    let ks: [u32; 7] = [0, 1, 8, N / 100, N / 10, N / 2, N];

    println!(
        "{:>7} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} | {:>6}",
        "k",
        "full ms",
        "A ms",
        "B ms",
        "attest",
        "full rows",
        "A rows",
        "B rows",
        "A prep",
        "B prep",
        "B bytes",
        "N*"
    );
    for k in ks {
        let mut parent: Vec<[u32; 2]> = (0..N).map(|i| [100_000 + i, 300_000 + i]).collect();
        // x = 17 sits below every background y, so the restricted range is
        // the table's first k rows; spread the chosen children over the
        // background instead of taking a prefix.
        let step = if k == 0 { 1 } else { N / k };
        parent.extend((0..k).map(|j| [17, 100_000 + j * step]));
        let allowed: Vec<[u32; 2]> = (0..N).map(|i| [100_000 + i, 100_000 + i]).collect();
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

        let mut attests = Vec::new();
        let mut lanes = Vec::new();
        for _ in 0..REPS {
            let t = Instant::now();
            let mut l = Vec::new();
            for &r in &rels {
                l.push(attest(&st, r).await.unwrap());
            }
            attests.push(t.elapsed());
            lanes = l;
        }
        let p = median(attests);

        let full = lance_mode(&rules, &d, NO_SEMIJOIN, &st, u64::from(k)).await;
        let a = lance_mode(&rules, &d, Options::ALL, &st, u64::from(k)).await;
        let b = arm_b(&rules, &d, &st, &lanes, u64::from(k)).await;
        let saving = a.time.saturating_sub(b.time);
        let break_even = if saving.is_zero() {
            "never".to_string()
        } else {
            (p.as_nanos().div_ceil(saving.as_nanos())).to_string()
        };
        println!(
            "{:>7} | {:>9.3} {:>9.3} {:>9.3} {:>9.3} | {:>9} {:>9} {:>9} | {:>9.3} {:>9.3} {:>9} | {:>6}",
            k,
            ms(full.time),
            ms(a.time),
            ms(b.time),
            ms(p),
            full.read.rows,
            a.read.rows,
            b.read.rows,
            ms(a.prepare),
            ms(b.prepare),
            b.read.bytes,
            break_even
        );
    }
}
