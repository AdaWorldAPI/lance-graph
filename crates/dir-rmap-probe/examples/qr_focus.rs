//! Cold and warm cost of a team focus with bound populations (arm Q),
//! against B+.
//!
//! ```text
//! cargo run --release --example qr_focus            # 65,000 users
//! cargo run --release --example qr_focus -- 8000
//! ```
//!
//! Question, three methods over one focus (the security team with the most
//! members): per license (16), members holding the license and role r,
//! outside location L; members holding no license; members holding r.
//!
//! * Q cold: bind the register, bind the focus, build the pointer tables and
//!   the 18 populations in the focus's tiles, check versions, fold the three
//!   methods through lance-graph-relalg (populations lent as `Rel::Plane`).
//! * Q warm: version check + the three folds over the bound populations.
//! * Q repair: after one membership removal and one nesting removal, rebind
//!   the register and repair the 18 populations and the focus, against
//!   rebuilding them.
//! * B+: compile; then per focus user `groups_of` + `roles_of` (its query
//!   API), and the populations B+ can build (`members_of`).
//!
//! Every route is checked equal before timing. Phases are reported apart; a
//! millisecond phase is a failed gate, never a number.

use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::focus::Row;
use dir_rmap_probe::pop::{Pop, TeamFocus};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals, Prepared};
use dir_rmap_probe::qr::{Qr, Runs};
use dir_rmap_probe::{Edit, World};
use lance_graph_mask_risc::Planes;
use lance_graph_relalg::{Answer, Measure, Query, Rel, run};
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

fn show(name: &str, d: Duration) {
    if d >= Duration::from_millis(1) {
        println!("  {name:<44} FAILED GATE (millisecond scale)");
    } else {
        println!("  {name:<44} {:>8.1} us", d.as_secs_f64() * 1e6);
    }
}

fn bit(r: &[u64], i: usize) -> bool {
    r[i >> 6] >> (i & 63) & 1 == 1
}

/// Planes: 0 focus, 1 role, 2 location, 3.. licenses.
fn methods(n_rows: usize, focus: &TeamFocus, pops: &[Pop]) -> Vec<u64> {
    let mut masks: Vec<&[u64]> = vec![focus.rows.plane(n_rows)];
    masks.extend(pops.iter().map(|p| p.plane(n_rows)));
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &[],
    };
    let count = |rel: Rel| match run(
        &Query {
            rel,
            measure: Measure::Count,
        },
        &planes,
        None,
    )
    .unwrap()
    .answer
    {
        Answer::Count(n) => n,
        a => panic!("{a:?}"),
    };
    let lic = (pops.len() - 2) as u16;
    let mut out: Vec<u64> = (0..lic)
        .map(|i| {
            count(Rel::And(vec![
                Rel::Plane(0),
                Rel::Plane(3 + i),
                Rel::Plane(1),
                Rel::Not(Box::new(Rel::Plane(2))),
            ]))
        })
        .collect();
    let any = Rel::Or((0..lic).map(|i| Rel::Plane(3 + i)).collect());
    out.push(count(Rel::And(vec![
        Rel::Plane(0),
        Rel::Not(Box::new(any)),
    ])));
    out.push(count(Rel::And(vec![Rel::Plane(0), Rel::Plane(1)])));
    out
}

/// The same three methods as [`methods`], folded directly over the bound
/// populations (no relalg program): the reference for relalg's per-query
/// cost.
fn methods_native(focus: &TeamFocus, pops: &[Pop]) -> Vec<u64> {
    let chain = focus
        .rows
        .clone()
        .and(&pops[0])
        .unwrap()
        .and_not(&pops[1])
        .unwrap();
    let mut out: Vec<u64> = pops[2..]
        .iter()
        .map(|l| l.count_and(&chain).unwrap())
        .collect();
    let mut any = pops[2].clone();
    for l in &pops[3..] {
        any = any.or(l).unwrap();
    }
    out.push(focus.rows.clone().and_not(&any).unwrap().count());
    out.push(focus.rows.count_and(&pops[0]).unwrap());
    out
}

struct Q {
    role: usize,
    team: u16,
    loc: u16,
    lic: Vec<u16>,
}

impl Q {
    fn slots(&self, qr: &Qr) -> Vec<Row> {
        let mut v = vec![
            qr.role_slots[self.role],
            qr.slots_reaching(self.loc).unwrap(),
        ];
        v.extend(self.lic.iter().map(|&g| qr.slots_reaching(g).unwrap()));
        v
    }
}

fn cold(q: &Q, b: &Prepared, w: &dir_rmap_probe::prep::Witness) -> Vec<u64> {
    let qr = Qr::bind(b, w).unwrap();
    let focus = qr.team_focus(b, q.team).unwrap();
    let mut runs = Runs::new(&qr);
    let pops: Vec<Pop> = q
        .slots(&qr)
        .iter()
        .map(|s| qr.population_in(b, s, &focus, &mut runs).unwrap())
        .collect();
    for p in &pops {
        p.check(w).unwrap();
    }
    methods(b.users, &focus, &pops)
}

/// B+ per focus user, through its own query API.
fn bplus(q: &Q, c: &Compiled, focus_users: &[u16], scratch: &mut [u64]) -> Vec<u64> {
    let n = q.lic.len();
    let mut out = vec![0u64; n + 2];
    for &u in focus_users {
        c.groups_of(u, Chain::Security, scratch);
        let role = c.roles_of(u) >> q.role & 1 == 1;
        let in_loc = bit(scratch, q.loc as usize);
        let mut any = false;
        for (i, &l) in q.lic.iter().enumerate() {
            let has = bit(scratch, l as usize);
            any |= has;
            out[i] += u64::from(has && role && !in_loc);
        }
        out[n] += u64::from(!any);
        out[n + 1] += u64::from(role);
    }
    out
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map_or(65_000, |s| s.parse().expect("user count"));
    let mut spec = Spec::small();
    spec.users = users;
    spec.licenses = 16;
    let mut w = World::new(Fixture::generate(spec));
    let ord = {
        let v = w.view();
        Ordinals::of(&w.f, &v)
    };
    let kinds = |w: &World, k: Kind| -> Vec<usize> {
        (0..w.f.groups.len())
            .filter(|&j| w.f.kind[j] == k)
            .collect()
    };
    let direct_n = |w: &World, j: usize| w.direct.iter().filter(|&&(_, g)| g as usize == j).count();
    let team_f = *kinds(&w, Kind::Team)
        .iter()
        .filter(|&&j| w.f.security[j] == Some(true))
        .max_by_key(|&&j| direct_n(&w, j))
        .unwrap();
    let loc_f = *kinds(&w, Kind::Location)
        .iter()
        .max_by_key(|&&j| direct_n(&w, j))
        .unwrap();
    let wit = w.witness();
    let b = w.prepare();
    let qr = Qr::bind(&b, &wit).unwrap();
    let role = (0..64)
        .max_by_key(|&r| qr.population(&b, &qr.role_slots[r]).unwrap().count())
        .unwrap();
    let q = Q {
        role,
        team: ord.group[team_f],
        loc: ord.group[loc_f],
        lic: kinds(&w, Kind::License)
            .iter()
            .map(|&j| ord.group[j])
            .collect(),
    };

    // Q, built once.
    let focus = qr.team_focus(&b, q.team).unwrap();
    let mut runs = Runs::new(&qr);
    let pops: Vec<Pop> = q
        .slots(&qr)
        .iter()
        .map(|s| qr.population_in(&b, s, &focus, &mut runs).unwrap())
        .collect();
    let want = methods(b.users, &focus, &pops);
    assert_eq!(methods_native(&focus, &pops), want);
    assert_eq!(cold(&q, &b, &wit), want);

    // B+.
    let c = Compiled::compile(w.prepare());
    let mut focus_users = Vec::new();
    for p in 0..qr.pages() {
        if focus.tiles[p >> 6] >> (p & 63) & 1 == 1 {
            for i in 0..256 {
                if bit(&focus.rows.tiles[p], i) {
                    focus_users.push((p * 256 + i) as u16);
                }
            }
        }
    }
    let mut scratch = vec![0u64; b.groups.div_ceil(64)];
    assert_eq!(
        bplus(&q, &c, &focus_users, &mut scratch),
        want,
        "B+ per user"
    );
    let mut ubits = vec![0u64; b.users.div_ceil(64)];

    // Timings.
    let t_prepare = median(5, || {
        black_box(w.prepare());
    });
    let t_register = median(15, || {
        black_box(Qr::bind(&b, &wit).unwrap());
    });
    let t_focus = median(15, || {
        black_box(qr.team_focus(&b, q.team).unwrap());
    });
    let slots = q.slots(&qr);
    let t_pops = median(15, || {
        let mut runs = Runs::new(&qr);
        for s in &slots {
            black_box(qr.population_in(&b, s, &focus, &mut runs).unwrap());
        }
    });
    let t_pops_warm = median(15, || {
        for s in &slots {
            black_box(qr.population_in(&b, s, &focus, &mut runs).unwrap());
        }
    });
    let t_native = median(51, || {
        black_box(methods_native(&focus, &pops));
    });
    let t_check = median(51, || {
        for p in &pops {
            black_box(p.check(&wit)).unwrap();
        }
    });
    let t_methods = median(15, || {
        black_box(methods(b.users, &focus, &pops));
    });
    let t_cold = median(9, || {
        black_box(cold(&q, &b, &wit));
    });
    let t_compile = median(5, || {
        black_box(Compiled::compile(w.prepare()));
    });
    let t_bplus = median(15, || {
        black_box(bplus(&q, &c, &focus_users, &mut scratch));
    });
    let t_bplus_pop = median(15, || {
        for &g in q.lic.iter().chain([&q.loc, &q.team]) {
            c.members_of(g, Chain::Security, &mut ubits);
            black_box(&ubits);
        }
    });

    // Repair: one membership removal (a focus member) and one nesting
    // removal, both in the security chain.
    let sec = |w: &World, g: u32| w.f.security[g as usize] == Some(true);
    let (ru, rg) = *w
        .direct
        .iter()
        .find(|&&(u, g)| sec(&w, g) && focus_users.contains(&ord.user[u as usize]))
        .unwrap();
    let (nc, np) = *w
        .nested
        .iter()
        .find(|&&(c, p)| sec(&w, c) && sec(&w, p))
        .unwrap();
    w.apply(&[Edit::RemoveDirect(ru, rg), Edit::RemoveNested(nc, np)]);
    let wit1 = w.witness();
    let b1 = w.prepare();
    let touched = [ord.user[ru as usize]];
    let qr1 = Qr::bind(&b1, &wit1).unwrap();
    let slots1 = q.slots(&qr1);
    let focus_slots = (
        qr.slots_reaching(q.team).unwrap(),
        qr1.slots_reaching(q.team).unwrap(),
    );
    let repair_all = |pops: &mut Vec<Pop>, f: &mut Pop| {
        for (p, (s0, s1)) in pops.iter_mut().zip(slots.iter().zip(&slots1)) {
            p.repair(&qr, &qr1, &b1, (s0, s1), &touched, Some(&focus.tiles))
                .unwrap();
        }
        f.repair(
            &qr,
            &qr1,
            &b1,
            (&focus_slots.0, &focus_slots.1),
            &touched,
            None,
        )
        .unwrap();
    };
    let mut rp = pops.clone();
    let mut rf = focus.rows.clone();
    repair_all(&mut rp, &mut rf);
    let focus1 = qr1.team_focus(&b1, q.team).unwrap();
    assert_eq!(rf, focus1.rows, "repaired focus");
    for (p, s) in rp.iter().zip(&slots1) {
        // A fresh population of the new version, kept to the focus's tiles.
        let mut fresh = qr1.population(&b1, s).unwrap();
        for (i, t) in fresh.tiles.iter_mut().enumerate() {
            if focus.tiles[i >> 6] >> (i & 63) & 1 == 0 {
                *t = [0; 4];
            }
        }
        fresh.kept = Some(focus.tiles.clone());
        assert_eq!(*p, fresh, "repaired pop");
    }
    let t_rebind = median(15, || {
        black_box(Qr::bind(&b1, &wit1).unwrap());
    });
    let t_repair = median(15, || {
        let mut rp = pops.clone();
        let mut rf = focus.rows.clone();
        repair_all(&mut rp, &mut rf);
        black_box((rp, rf));
    });
    let t_rebuild = median(15, || {
        let f = qr1.team_focus(&b1, q.team).unwrap();
        let mut runs1 = Runs::new(&qr1);
        for s in &slots1 {
            black_box(qr1.population_in(&b1, s, &f, &mut runs1).unwrap());
        }
    });
    let t_clone = median(15, || {
        black_box((pops.clone(), focus.rows.clone()));
    });

    println!(
        "users {}, tiles {}, team members {}, focus tiles {}, 18 populations, pointer tables {} slots",
        b.users,
        qr.pages(),
        focus.rows.count(),
        focus.tiles_selected(),
        runs.built
    );
    println!("answers {want:?}");
    println!("Q cold (per version):");
    show("prepare (arm B CSR, shared with B+)", t_prepare);
    show("bind register", t_register);
    show("bind team focus", t_focus);
    show("pointer tables + 18 populations in focus", t_pops);
    show("  of which populations (tables warm)", t_pops_warm);
    show("version checks (18)", t_check);
    show("3 methods, relalg folds (18 queries)", t_methods);
    show("all of the above but prepare", t_cold);
    println!("Q warm (bound populations reused):");
    show("version checks + 3 methods (relalg)", t_check + t_methods);
    show(
        "version checks + 3 methods (native Pop folds)",
        t_check + t_native,
    );
    println!("Q after an edit (1 membership + 1 nesting removed):");
    show("rebind register", t_rebind);
    show(
        "repair 18 populations + focus",
        t_repair.saturating_sub(t_clone),
    );
    show("rebuild focus + tables + 18 populations", t_rebuild);
    println!("B+:");
    show("prepare + compile", t_compile);
    show(
        &format!(
            "per focus user ({}), groups_of + roles_of",
            focus_users.len()
        ),
        t_bplus,
    );
    show(
        "members_of populations (16 licenses + loc + team)",
        t_bplus_pop,
    );
}
