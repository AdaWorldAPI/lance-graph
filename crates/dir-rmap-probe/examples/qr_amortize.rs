//! Amortization of a masked count over several QR tiles (arm Q).
//!
//! ```text
//! cargo run --release --example qr_amortize            # 65,000 users
//! cargo run --release --example qr_amortize -- 8000
//! ```
//!
//! One question ("how many users hold role r", or "are effectively in group
//! g") over `k` tiles of 256 users, selected by a tile mask spread across
//! the population, with rows inside each tile selected by a row mask (all
//! rows, or interlaced: even rows in even tiles, odd rows in odd tiles).
//! Routes, all checked equal before timing:
//!
//! * `page`: resolve each selected page on first touch, then fold its
//!   resolved roles over the active rows;
//! * `warm`: the same fold over pages already resolved;
//! * `masked`: [`Qr::count_masked`]: each needed group's direct members are
//!   read once and set bits only in selected tiles; popcount of tile AND row
//!   mask. No page is resolved.
//! * `runs (cold)`: [`Runs::count`] with the per-slot tile pointer tables
//!   built inside the timed call; `runs (warm)`: tables already built. Only
//!   the selected tiles' runs are visited.

use dir_rmap_probe::World;
use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::focus::Row;
use dir_rmap_probe::prep::Prepared;
use dir_rmap_probe::qr::{Page, Qr, Runs};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ALL: Row = [!0; 4];
const EVEN: Row = [0x5555_5555_5555_5555; 4];
const ODD: Row = [0xAAAA_AAAA_AAAA_AAAA; 4];

fn bit(r: &Row, i: usize) -> bool {
    r[i >> 6] >> (i & 63) & 1 == 1
}

fn median(reps: usize, mut f: impl FnMut()) -> Duration {
    let mut t: Vec<Duration> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed()
        })
        .collect();
    t.sort();
    t[reps / 2]
}

fn fmt(d: Duration) -> String {
    if d >= Duration::from_millis(1) {
        "FAILED GATE (ms)".into()
    } else {
        format!("{:.1} us", d.as_secs_f64() * 1e6)
    }
}

/// `k` tiles spread evenly over `pages`.
fn tiles(pages: usize, k: usize) -> Vec<u64> {
    let mut m = vec![0u64; pages.div_ceil(64)];
    for j in 0..k {
        let p = j * pages / k;
        m[p >> 6] |= 1 << (p & 63);
    }
    m
}

fn selected(t: &[u64], pages: usize) -> Vec<usize> {
    (0..pages)
        .filter(|&p| t[p >> 6] >> (p & 63) & 1 == 1)
        .collect()
}

/// Count via resolved pages: `role` = Some(r) for a role question, else a
/// group slot `t` tested on the effective row.
fn fold(qr: &Qr, pages: &[(usize, &Page)], rows: &[Row], q: Q) -> u64 {
    let mut n = 0;
    for &(p, page) in pages {
        for i in 0..256 {
            if bit(&rows[p], i) {
                n += u64::from(match q {
                    Q::Role(r) => page.roles[i] >> r & 1 == 1,
                    Q::Group(t) => bit(&qr.effective(&page.member[i]), t),
                });
            }
        }
    }
    n
}

#[derive(Clone, Copy)]
enum Q {
    Role(u32),
    Group(usize),
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map_or(65_000, |s| s.parse().expect("user count"));
    let mut spec = Spec::small();
    spec.users = users;
    let w = World::new(Fixture::generate(spec));
    let b: Prepared = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let np = qr.pages();
    let all: Vec<Page> = (0..np).map(|p| qr.page(&b, p).unwrap()).collect();

    // The role carried by the most slots, and the group reached from the
    // most slots (the widest questions).
    let r = (0..64)
        .max_by_key(|&r| qr.role_slots[r].iter().map(|x| x.count_ones()).sum::<u32>())
        .unwrap();
    let g = *qr
        .ids
        .iter()
        .max_by_key(|&&g| {
            qr.slots_reaching(g)
                .unwrap()
                .iter()
                .map(|x| x.count_ones())
                .sum::<u32>()
        })
        .unwrap();
    let pop = |row: &Row| row.iter().map(|x| x.count_ones()).sum::<u32>();
    let questions = [
        ("role", qr.role_slots[r], Q::Role(r as u32)),
        (
            "group",
            qr.slots_reaching(g).unwrap(),
            Q::Group(qr.slot(g).unwrap()),
        ),
    ];
    let row_sets: [(&str, Vec<Row>); 2] = [
        ("all rows", vec![ALL; np]),
        (
            "interlaced",
            (0..np)
                .map(|p| if p % 2 == 0 { EVEN } else { ODD })
                .collect(),
        ),
    ];
    println!(
        "users {}, tiles {np}, security slots {}",
        b.users,
        qr.ids.len()
    );
    let mut scratch = vec![[0u64; 4]; np];
    for (qname, slots, q) in questions {
        let members: usize = qr
            .ids
            .iter()
            .enumerate()
            .filter(|&(s, _)| bit(&slots, s))
            .map(|(_, &g)| b.group_users.row(g as usize).len())
            .sum();
        println!(
            "\n{qname}: {} slots, {members} direct memberships read by the masked route",
            pop(&slots)
        );
        for (rname, rows) in &row_sets {
            println!("  {rname}:");
            println!(
                "    {:>4} {:>18} {:>12} {:>12} {:>12} {:>12}",
                "k", "page (first touch)", "warm page", "masked", "runs cold", "runs warm"
            );
            for k in [1, 2, 4, 8, 16, 32, 64, 128, np] {
                let k = k.min(np);
                let t = tiles(np, k);
                let sel = selected(&t, np);
                let warm: Vec<(usize, &Page)> = sel.iter().map(|&p| (p, &all[p])).collect();
                let want = fold(&qr, &warm, rows, q);
                let got = qr.count_masked(&b, &slots, &t, rows, &mut scratch).unwrap();
                assert_eq!(got, want, "{qname} {rname} k={k}");
                let t_page = median(9, || {
                    let fresh: Vec<Page> = sel.iter().map(|&p| qr.page(&b, p).unwrap()).collect();
                    let v: Vec<(usize, &Page)> = sel.iter().copied().zip(fresh.iter()).collect();
                    black_box(fold(&qr, &v, rows, q));
                });
                let t_warm = median(9, || {
                    black_box(fold(&qr, &warm, rows, q));
                });
                let t_mask = median(9, || {
                    black_box(qr.count_masked(&b, &slots, &t, rows, &mut scratch).unwrap());
                });
                let mut runs = Runs::new(&qr);
                assert_eq!(runs.count(&qr, &b, &slots, &t, rows).unwrap(), want);
                let t_cold = median(9, || {
                    let mut runs = Runs::new(&qr);
                    black_box(runs.count(&qr, &b, &slots, &t, rows).unwrap());
                });
                let t_runs = median(9, || {
                    black_box(runs.count(&qr, &b, &slots, &t, rows).unwrap());
                });
                println!(
                    "    {k:>4} {:>18} {:>12} {:>12} {:>12} {:>12}",
                    fmt(t_page),
                    fmt(t_warm),
                    fmt(t_mask),
                    fmt(t_cold),
                    fmt(t_runs),
                );
            }
        }
    }
}
