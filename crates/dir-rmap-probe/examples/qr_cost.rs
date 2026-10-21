//! Cost of arm Q (the DR256 register) against the compiled closure (B+).
//!
//! ```text
//! cargo run --release --example qr_cost            # 8,000 users
//! cargo run --release --example qr_cost -- 65000
//! ```
//!
//! The group side is `Spec::small`'s (about 160 security groups, so the whole
//! chain fits one register); only the user count grows. Every route is
//! checked equal to dir-sim's own walk before it is timed. Preparation
//! (prepare, compile, bind) is reported per phase, never folded into a
//! per-user figure.

use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals};
use dir_rmap_probe::qr::Qr;
use dir_rmap_probe::{World, a_groups};
use ogar_dir_core::Guid128;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn per_user(n: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        f();
        t.push(s.elapsed());
    }
    median(t).as_nanos() as f64 / n as f64
}

fn once(reps: usize, mut f: impl FnMut()) -> Duration {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        f();
        t.push(s.elapsed());
    }
    median(t)
}

/// P0: a millisecond result is a failed gate, never a measurement.
fn show(name: &str, d: Duration) {
    if d >= Duration::from_millis(1) {
        println!("  {name:<22} FAILED GATE (millisecond scale)");
    } else {
        println!("  {name:<22} {:>8.1} us", d.as_secs_f64() * 1e6);
    }
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map_or(8_000, |s| s.parse().expect("user count"));
    let mut spec = Spec::small();
    spec.users = users;
    let w = World::new(Fixture::generate(spec));
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let by_guid: HashMap<Guid128, u64> =
        w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
            *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
            m
        });

    let witness = w.witness();
    let b = w.prepare();
    let c = Compiled::compile(w.prepare());
    let q = Qr::bind(&b, &witness).expect("security chain fits 256 slots");
    for (i, ug) in w.f.users.iter().enumerate() {
        let want = a_groups(&v, ug, Chain::Security)
            .iter()
            .fold(0u64, |a, g| a | by_guid.get(g).copied().unwrap_or(0));
        let uo = ord.user[i];
        assert_eq!(c.roles_of(uo), want, "B+ {i}");
        assert_eq!(q.roles_of(uo), want, "Q resolved {i}");
        assert_eq!(q.roles_live(uo), want, "Q live {i}");
    }

    let t_prepare = once(5, || {
        black_box(w.prepare());
    });
    let t_compile = once(5, || {
        black_box(Compiled::compile(w.prepare()));
    });
    let t_bind = once(5, || {
        black_box(Qr::bind(&b, &witness).unwrap());
    });

    let n = w.f.users.len();
    let reps = 15;
    let t_fold = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(c.roles_of(black_box(uo)));
        }
    });
    let t_q = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(q.roles_of(black_box(uo)));
        }
    });
    let t_live = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(q.roles_live(black_box(uo)));
        }
    });
    let t_eff = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(q.effective(black_box(uo)));
        }
    });

    println!(
        "users {n}, groups {}, security slots {}, translations {}, all answers checked",
        v.groups_len(),
        q.ids.len(),
        q.translations
    );
    println!("preparation, once per version:");
    show("prepare", t_prepare);
    show("prepare + compile (B+)", t_compile);
    show("bind (Q)", t_bind);
    println!("per user (median of runs):");
    println!("  B+ fold                   {t_fold:>7.1} ns");
    println!("  Q roles, resolved at bind {t_q:>7.1} ns");
    println!("  Q roles, row x roles      {t_live:>7.1} ns");
    println!("  Q effective row (M x R)   {t_eff:>7.1} ns");
    println!("register bytes {}", q.bytes());
}
