//! Arm Q, chained population masking: license groups filtered over every
//! user's memberships. Each filter is a population (one bit per user,
//! positioned by tile and row); the chain is AND / AND-NOT by position and
//! the answer is a count. Every count must equal a per-user check against
//! dir-sim's own walk of effective groups.

use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::prep::{Chain, Ordinals};
use dir_rmap_probe::qr::Qr;
use dir_rmap_probe::{World, a_groups};
use ogar_dir_core::Guid128;
use std::collections::{HashMap, HashSet};

#[test]
fn license_counts_by_chained_masks_equal_dir_sim() {
    let mut spec = Spec::small();
    spec.users = 3000;
    spec.licenses = 16;
    let w = World::new(Fixture::generate(spec));
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let of_kind = |k: Kind| -> Vec<usize> {
        (0..w.f.groups.len())
            .filter(|&j| w.f.kind[j] == k)
            .collect()
    };
    let licenses = of_kind(Kind::License);
    let locations = of_kind(Kind::Location);
    assert_eq!(licenses.len(), 16);

    // Per user, from dir-sim: effective security groups and roles.
    let grants: HashMap<Guid128, u64> = w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
        *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
        m
    });
    let users: Vec<(u16, HashSet<Guid128>, u64)> =
        w.f.users
            .iter()
            .enumerate()
            .map(|(i, ug)| {
                let eff: HashSet<Guid128> = a_groups(&v, ug, Chain::Security).into_iter().collect();
                let roles = eff
                    .iter()
                    .fold(0, |a, g| a | grants.get(g).copied().unwrap_or(0));
                (ord.user[i], eff, roles)
            })
            .collect();

    // The filters: a widely held role, and the most populated location.
    let role = (0..64)
        .max_by_key(|&r| users.iter().filter(|u| u.2 >> r & 1 == 1).count())
        .unwrap();
    let loc = *locations
        .iter()
        .max_by_key(|&&j| {
            users
                .iter()
                .filter(|u| u.1.contains(&w.f.groups[j]))
                .count()
        })
        .unwrap();
    let pop_of = |j: usize| {
        qr.population(&b, &qr.slots_reaching(ord.group[j]).unwrap())
            .unwrap()
    };
    let role_pop = qr.population(&b, &qr.role_slots[role]).unwrap();
    let loc_pop = pop_of(loc);
    // role holders outside the location: the chain every license is
    // filtered through.
    // Anti-vacuity: the location removes some role holders.
    assert!(role_pop.count_and(&loc_pop).unwrap() > 0);
    let chain = role_pop.and_not(&loc_pop).unwrap();

    let mut any = qr.population(&b, &[0; 4]).unwrap();
    let (mut nonzero, mut distinct) = (0, HashSet::new());
    for &l in &licenses {
        let lp = pop_of(l);
        let got = lp.count_and(&chain).unwrap();
        let want = users
            .iter()
            .filter(|(_, eff, roles)| {
                eff.contains(&w.f.groups[l])
                    && roles >> role & 1 == 1
                    && !eff.contains(&w.f.groups[loc])
            })
            .count() as u64;
        assert_eq!(got, want, "license {l}");
        // The same answer by building the chain explicitly.
        assert_eq!(lp.clone().and(&chain).unwrap().count(), want);
        nonzero += usize::from(got > 0);
        distinct.insert(got);
        for (a, t) in any.tiles.iter_mut().zip(&lp.tiles) {
            for k in 0..4 {
                a[k] |= t[k];
            }
        }
    }
    // Users with no license at all: everyone AND-NOT any license.
    let unlicensed = qr.everyone().and_not(&any).unwrap().count();
    let want = users
        .iter()
        .filter(|(_, eff, _)| licenses.iter().all(|&l| !eff.contains(&w.f.groups[l])))
        .count() as u64;
    assert_eq!(unlicensed, want);
    assert_eq!(qr.everyone().count(), b.users as u64);
    // Anti-vacuity: the chain filters something, licenses differ, and some
    // users are unlicensed.
    assert!(chain_filters(&qr, &b, role));
    assert!(nonzero > 8 && distinct.len() > 4, "{nonzero} {distinct:?}");
    assert!(unlicensed > 0 && unlicensed < b.users as u64);
}

fn chain_filters(qr: &Qr, b: &dir_rmap_probe::prep::Prepared, role: usize) -> bool {
    let n = qr.population(b, &qr.role_slots[role]).unwrap().count();
    n > 0 && n < b.users as u64
}
