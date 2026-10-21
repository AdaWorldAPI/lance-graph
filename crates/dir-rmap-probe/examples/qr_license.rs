//! Chained population masking over license groups (arm Q).
//!
//! ```text
//! cargo run --release --example qr_license            # 65,000 users
//! cargo run --release --example qr_license -- 8000
//! ```
//!
//! Question: per license group, how many users hold the license (directly or
//! through nesting), hold a given role, and are NOT in a given location?
//! Every filter is a population (one bit per user, by tile and row); the
//! chain is AND / AND-NOT by position; the answer is a count per license.
//! The per-user route (resolved pages, one effective row per user, bit
//! tests) is the reference and is timed for comparison. Both are checked
//! equal before timing.

use dir_rmap_probe::World;
use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::focus::Row;
use dir_rmap_probe::prep::Ordinals;
use dir_rmap_probe::qr::{Page, Pop, Qr};
use std::hint::black_box;
use std::time::{Duration, Instant};

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
        "FAILED GATE (millisecond scale)".into()
    } else {
        format!("{:.1} us", d.as_secs_f64() * 1e6)
    }
}

fn bit(r: &Row, i: usize) -> bool {
    r[i >> 6] >> (i & 63) & 1 == 1
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map_or(65_000, |s| s.parse().expect("user count"));
    let mut spec = Spec::small();
    spec.users = users;
    spec.licenses = 16;
    let w = World::new(Fixture::generate(spec));
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let of_kind = |k: Kind| -> Vec<u16> {
        (0..w.f.groups.len())
            .filter(|&j| w.f.kind[j] == k)
            .map(|j| ord.group[j])
            .collect()
    };
    let licenses = of_kind(Kind::License);
    let reach = |g: u16| qr.slots_reaching(g).unwrap();
    let role = (0..64)
        .max_by_key(|&r| qr.role_slots[r].iter().map(|x| x.count_ones()).sum::<u32>())
        .unwrap();
    let loc = *of_kind(Kind::Location)
        .iter()
        .max_by_key(|&&g| b.group_users.row(g as usize).len())
        .unwrap();
    let (loc_slots, role_slots) = (reach(loc), qr.role_slots[role]);
    let lic_slots: Vec<Row> = licenses.iter().map(|&g| reach(g)).collect();
    let lic_bits: Vec<usize> = licenses.iter().map(|&g| qr.slot(g).unwrap()).collect();

    // The chained-mask route.
    let masked = || -> Vec<u64> {
        let chain: Pop = qr
            .population(&b, &role_slots)
            .unwrap()
            .and_not(&qr.population(&b, &loc_slots).unwrap());
        lic_slots
            .iter()
            .map(|s| qr.population(&b, s).unwrap().count_and(&chain))
            .collect()
    };
    // The per-user route over resolved pages.
    let pages: Vec<Page> = (0..qr.pages()).map(|p| qr.page(&b, p).unwrap()).collect();
    let (loc_bit, users_n) = (qr.slot(loc).unwrap(), b.users);
    let per_user = || -> Vec<u64> {
        let mut out = vec![0u64; licenses.len()];
        for u in 0..users_n {
            let page = &pages[u / 256];
            let i = u % 256;
            if page.roles[i] >> role & 1 == 0 {
                continue;
            }
            let eff = qr.effective(&page.member[i]);
            if bit(&eff, loc_bit) {
                continue;
            }
            for (o, &s) in out.iter_mut().zip(&lic_bits) {
                *o += u64::from(bit(&eff, s));
            }
        }
        out
    };
    let got = masked();
    assert_eq!(got, per_user(), "masked chain vs per-user");

    let role_pop = qr.population(&b, &role_slots).unwrap();
    let loc_pop = qr.population(&b, &loc_slots).unwrap();
    let chain = role_pop.clone().and_not(&loc_pop);
    let lic_pops: Vec<Pop> = lic_slots
        .iter()
        .map(|s| qr.population(&b, s).unwrap())
        .collect();
    let t_role = median(15, || {
        black_box(qr.population(&b, &role_slots).unwrap());
    });
    let t_loc = median(15, || {
        black_box(qr.population(&b, &loc_slots).unwrap());
    });
    let t_lic = median(15, || {
        for s in &lic_slots {
            black_box(qr.population(&b, s).unwrap());
        }
    });
    let t_chain = median(51, || {
        black_box(role_pop.clone().and_not(black_box(&loc_pop)));
    });
    let t_fold = median(51, || {
        for p in &lic_pops {
            black_box(p.count_and(black_box(&chain)));
        }
    });
    let t_all = median(15, || {
        black_box(masked());
    });
    let t_user = median(5, || {
        black_box(per_user());
    });

    let members = |s: &Row| -> usize {
        qr.ids
            .iter()
            .enumerate()
            .filter(|&(k, _)| bit(s, k))
            .map(|(_, &g)| b.group_users.row(g as usize).len())
            .sum()
    };
    println!(
        "users {}, tiles {}, {} licenses; role {role} holders {}, location holders {}, chain {}",
        b.users,
        qr.pages(),
        licenses.len(),
        role_pop.count(),
        loc_pop.count(),
        chain.count()
    );
    println!("per license (role AND-NOT location): {got:?}");
    println!(
        "memberships read: role {}, location {}, licenses {}",
        members(&role_slots),
        members(&loc_slots),
        lic_slots.iter().map(members).sum::<usize>()
    );
    println!("chained masks:");
    println!("  build role population        {}", fmt(t_role));
    println!("  build location population    {}", fmt(t_loc));
    println!("  build 16 license populations {}", fmt(t_lic));
    println!("  chain role AND-NOT location  {}", fmt(t_chain));
    println!("  16 x (license AND chain) count {}", fmt(t_fold));
    println!("  whole question, end to end   {}", fmt(t_all));
    println!("per-user route over resolved pages (reference):");
    println!("  whole question               {}", fmt(t_user));
    println!(
        "population bytes {} each",
        qr.pages() * std::mem::size_of::<Row>()
    );
}
