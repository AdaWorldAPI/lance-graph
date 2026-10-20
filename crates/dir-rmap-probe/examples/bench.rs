//! D-RMAP-DIR-1 benchmark: arms A (dir-sim View), D (crepe), B (CSR),
//! B+ (compiled closure) and F (256-slot focus), on one synthetic IAM
//! directory. Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example bench            # 8,000 users
//! cargo run --release --example bench -- 65000   # large fixture
//! ```
//!
//! Every timed answer is checked against arm A (and D where it applies)
//! before it is timed. Medians of repeated runs; per-operation figures are
//! a timed batch divided by its size. "Cold" means after touching 64 MiB of
//! unrelated memory, so the prepared structures are out of L1/L2.

use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::focus::{Focus, FocusUsers, holds};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals, bit, words};
use dir_rmap_probe::{Edit, World, a_groups, a_members, focus_for, oracle, templates};
use lance_graph_dir_sim::{GroupOrdinal, UserOrdinal};
use ndarray::simd::{eq_u8_to_mask, mask_and_assign, popcount_batch_u64};
use ogar_dir_core::Guid128;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Median wall time of `reps` runs of `f`.
fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> Duration {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        black_box(f());
        t.push(s.elapsed());
    }
    median(t)
}

fn ns(d: Duration, n: usize) -> f64 {
    d.as_nanos() as f64 / n as f64
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Touch 64 MiB so the next run starts with cold caches.
fn evict(buf: &mut [u64]) {
    for (i, x) in buf.iter_mut().enumerate() {
        *x = x.wrapping_add(i as u64);
    }
    black_box(&buf);
}

fn cache_sizes() -> String {
    let mut out = Vec::new();
    for i in 0..4 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
        let (Ok(l), Ok(t), Ok(s)) = (
            std::fs::read_to_string(format!("{base}/level")),
            std::fs::read_to_string(format!("{base}/type")),
            std::fs::read_to_string(format!("{base}/size")),
        ) else {
            continue;
        };
        out.push(format!("L{} {} {}", l.trim(), t.trim(), s.trim()));
    }
    out.join(", ")
}

fn cpu() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_default()
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("user count"))
        .unwrap_or(8_000);
    println!("# D-RMAP-DIR-1, {users} users\n");
    println!("cpu: {}  ", cpu());
    println!("caches: {}  ", cache_sizes());
    println!(
        "avx512f: {}, avx2: {}\n",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    let mut evict_buf = vec![0u64; 8 << 20];

    // ---- construction ---------------------------------------------------
    let s = Instant::now();
    let fx = Fixture::generate(Spec::iam(users));
    let t_gen = s.elapsed();
    let s = Instant::now();
    let mut w = World::new(fx);
    let t_snap = s.elapsed();
    let v0 = w.version;
    let (n_users, n_groups) = (w.view().users_len(), w.view().groups_len());
    let t_b = time(5, || w.prepare());
    let b = w.prepare();
    let t_c = time(5, || {
        let p = w.prepare();
        let s = Instant::now();
        let c = Compiled::compile(p);
        (s.elapsed(), c.passes)
    });
    // `time` measured prepare + compile; compile alone is measured inside.
    let mut compile_only = Vec::new();
    for _ in 0..5 {
        let p = w.prepare();
        let s = Instant::now();
        black_box(Compiled::compile(p));
        compile_only.push(s.elapsed());
    }
    let t_compile = median(compile_only);
    let c = Compiled::compile(w.prepare());
    let all_templates = templates(&w.f);
    let s = Instant::now();
    let d = oracle::run(&w.f, &w.direct, &w.nested, &all_templates);
    let t_d = s.elapsed();
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    println!("## Construction\n");
    println!("| step | time |\n|---|---|");
    println!(
        "| fixture generation | {:.1} ms |",
        t_gen.as_secs_f64() * 1e3
    );
    println!(
        "| dir-sim snapshot (`observe`: Snapshot::build) | {:.1} ms |",
        t_snap.as_secs_f64() * 1e3
    );
    println!(
        "| B: CSR prepare (incl. GUID translation) | {:.2} ms |",
        t_b.as_secs_f64() * 1e3
    );
    println!(
        "| B+: closure compile, 3 chains x up/down (B excluded) | {:.2} ms |",
        t_compile.as_secs_f64() * 1e3
    );
    println!("| B + B+ together | {:.2} ms |", t_c.as_secs_f64() * 1e3);
    println!(
        "| D: crepe oracle, all relations | {:.1} ms |",
        t_d.as_secs_f64() * 1e3
    );
    println!(
        "\n{n_users} users, {n_groups} groups, {} direct rows, {} nesting pairs, {} groups > 256 direct members; closure passes (up, down) all/sec/mail: {:?}\n",
        w.direct.len(),
        w.nested.len(),
        w.f.large_groups(),
        c.passes
    );

    // ---- memory -----------------------------------------------------------
    let gw = words(n_groups);
    println!("## Memory\n");
    println!("| structure | bytes |\n|---|---|");
    println!("| B: CSR tables + planes | {} |", b.bytes());
    println!(
        "| B+: closure rows (3 chains x up/down) + role table | {} |",
        c.closure_bytes()
    );
    println!(
        "| one dense closure row ({gw} words) / one chain, up only | {} / {} |",
        gw * 8,
        n_groups * gw * 8
    );
    let (_, g256) = focus_for(&all_templates, 256);
    let focus_ids: Vec<u16> = g256.iter().map(|&g| ord.group[g as usize]).collect();
    let f256 = Focus::bind(&c, &focus_ids);
    let fu256 = FocusUsers::bind(&c, &focus_ids);
    println!(
        "| F (256 slots): projected rows + reverse map | {} |",
        f256.bytes()
    );
    println!(
        "| F+users (256 slots): + one row per user | {} |",
        fu256.bytes()
    );
    println!("| a 256 x 256 Boolean tile (not built) | 8192 |\n");

    // ---- correctness at scale ---------------------------------------------
    // Arm A scans every membership row per call, so it is measured on a
    // sample: 2,000 users, or 500 above 20,000 users.
    let a_n = if users > 20_000 { 500 } else { 2_000 };
    let sample: Vec<usize> = (0..w.f.users.len())
        .step_by((w.f.users.len() / a_n).max(1))
        .collect();
    let mut row = vec![0u64; gw];
    for &i in &sample {
        c.groups_of(ord.user[i], Chain::All, &mut row);
        let got: Vec<Guid128> = dir_rmap_probe::prep::ones(&row)
            .map(|g| v.group_guid(GroupOrdinal(g as u16)).unwrap())
            .collect();
        let mut a = a_groups(&v, &w.f.users[i], Chain::All);
        a.sort_unstable();
        let mut got = got;
        got.sort_unstable();
        assert_eq!(got, a, "B+ != A for user {i}");
    }
    let total_b: usize = (0..w.f.users.len())
        .map(|i| {
            c.groups_of(ord.user[i], Chain::All, &mut row);
            popcount_batch_u64(&row) as usize
        })
        .sum();
    assert_eq!(
        total_b,
        d.member.len(),
        "B+ and the oracle disagree on membership size"
    );
    println!(
        "Checked: B+ = A on {} users; B+ = D on all {} memberships.\n",
        sample.len(),
        d.member.len()
    );

    // ---- W1 direct membership --------------------------------------------
    let mut rng = dir_rmap_probe::fixture::Rng::new(7);
    let pairs: Vec<(usize, usize)> = (0..10_000)
        .map(|_| (rng.below(w.f.users.len()), rng.below(w.f.groups.len())))
        .collect();
    let a1 = time(9, || {
        pairs
            .iter()
            .filter(|&&(u, g)| v.is_member(&w.f.users[u], &w.f.groups[g]))
            .count()
    });
    let b1 = time(9, || {
        pairs
            .iter()
            .filter(|&&(u, g)| b.is_member(ord.user[u], ord.group[g]))
            .count()
    });
    // ---- W2 user -> effective groups --------------------------------------
    let mut queue = Vec::new();
    let a2 = time(3, || {
        sample
            .iter()
            .map(|&i| a_groups(&v, &w.f.users[i], Chain::All).len())
            .sum::<usize>()
    });
    let all_users: Vec<usize> = (0..w.f.users.len()).collect();
    let mut prow = vec![0u64; gw];
    let b2 = time(9, || {
        for &i in &all_users {
            b.groups_of(ord.user[i], Chain::All, &mut prow, &mut queue);
        }
        prow[0]
    });
    let c2 = time(9, || {
        for &i in &all_users {
            c.groups_of(ord.user[i], Chain::All, &mut prow);
        }
        prow[0]
    });
    let c2e = time(9, || {
        let mut n = 0;
        for &i in &all_users {
            c.groups_of(ord.user[i], Chain::All, &mut prow);
            let out: Vec<Guid128> = dir_rmap_probe::prep::ones(&prow)
                .map(|g| v.group_guid(GroupOrdinal(g as u16)).unwrap())
                .collect();
            n += out.len();
        }
        n
    });
    // ---- W3 group -> transitive members -----------------------------------
    let uw = words(n_users);
    let (mut reached, mut members) = (vec![0u64; gw], vec![0u64; uw]);
    let gsample: Vec<usize> = (0..w.f.groups.len()).collect();
    let a3 = time(3, || {
        gsample
            .iter()
            .map(|&j| a_members(&v, &w.f.groups[j], Chain::All).len())
            .sum::<usize>()
    });
    let b3 = time(9, || {
        for &j in &gsample {
            b.members_of(
                ord.group[j],
                Chain::All,
                &mut reached,
                &mut members,
                &mut queue,
            );
        }
        members[0]
    });
    let c3 = time(9, || {
        for &j in &gsample {
            c.members_of(ord.group[j], Chain::All, &mut members);
        }
        members[0]
    });
    // ---- W4 roles and SIDs -------------------------------------------------
    let by_guid: HashMap<Guid128, u64> =
        w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
            *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
            m
        });
    let a4 = time(3, || {
        sample
            .iter()
            .map(|&i| {
                a_groups(&v, &w.f.users[i], Chain::Security)
                    .iter()
                    .fold(0u64, |a, g| a | by_guid.get(g).copied().unwrap_or(0))
            })
            .fold(0, |a, r| a ^ r)
    });
    for &i in &sample {
        let a = a_groups(&v, &w.f.users[i], Chain::Security)
            .iter()
            .fold(0u64, |a, g| a | by_guid.get(g).copied().unwrap_or(0));
        assert_eq!(c.roles_of(ord.user[i]), a, "B+ roles != A");
    }
    let b4 = time(9, || {
        all_users
            .iter()
            .map(|&i| b.roles_of(ord.user[i], &mut prow, &mut queue))
            .fold(0, |a, r| a ^ r)
    });
    let c4 = time(9, || {
        all_users
            .iter()
            .map(|&i| c.roles_of(ord.user[i]))
            .fold(0, |a, r| a ^ r)
    });
    let c4s = time(9, || {
        for &i in &all_users {
            c.groups_of(ord.user[i], Chain::Security, &mut prow);
        }
        prow[0]
    });

    println!("## Hot workloads (per operation, warm)\n");
    println!("| workload | A (dir-sim) | B (CSR) | B+ (closure) | B+ / A |\n|---|---|---|---|---|");
    let row_fmt = |name: &str, a: f64, bb: Option<f64>, cc: f64| {
        println!(
            "| {name} | {a:.0} ns | {} | {cc:.0} ns | {:.0}x faster |",
            bb.map_or("-".into(), |x| format!("{x:.0} ns")),
            a / cc
        );
    };
    row_fmt(
        "W1 direct membership",
        ns(a1, pairs.len()),
        Some(ns(b1, pairs.len())),
        ns(b1, pairs.len()),
    );
    row_fmt(
        "W2 user -> effective groups (plane)",
        ns(a2, sample.len()),
        Some(ns(b2, all_users.len())),
        ns(c2, all_users.len()),
    );
    row_fmt(
        "W2 same, emitted as GUIDs",
        ns(a2, sample.len()),
        None,
        ns(c2e, all_users.len()),
    );
    row_fmt(
        "W3 group -> transitive members",
        ns(a3, gsample.len()),
        Some(ns(b3, gsample.len())),
        ns(c3, gsample.len()),
    );
    row_fmt(
        "W4 roles via SID chain",
        ns(a4, sample.len()),
        Some(ns(b4, all_users.len())),
        ns(c4, all_users.len()),
    );
    println!(
        "| W4 SIDs only (security closure OR) | - | - | {:.0} ns | |",
        ns(c4s, all_users.len())
    );
    println!(
        "\nA columns are measured on {} sampled users / all {} groups; B and B+ on all users.\n",
        sample.len(),
        gsample.len()
    );

    // ---- W5 role x location: B+ vs F ---------------------------------------
    println!("## W5: role x location questions, B+ vs F\n");
    println!(
        "One evaluation = one user, every template question in the focus. N = evaluations reusing one prepared focus (users cycled). F prepare = bind + projection of every group's closure row. Totals measured directly.\n"
    );
    println!(
        "| focus groups | questions | A eval | B+ eval | F eval | F+users eval | F prepare | F+users prepare | B+ N=1 / F / F+u | N=10 | N=100 | N=1000 | N=10000 |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for k in [16usize, 64, 128, 256] {
        let (ts, gs) = focus_for(&all_templates, k);
        let groups: Vec<u16> = gs.iter().map(|&g| ord.group[g as usize]).collect();
        let tq: Vec<(usize, usize)> = ts
            .iter()
            .map(|&(r, l)| {
                (
                    ord.group[r as usize] as usize,
                    ord.group[l as usize] as usize,
                )
            })
            .collect();
        let f = Focus::bind(&c, &groups);
        let fu = FocusUsers::bind(&c, &groups);
        let masks: Vec<_> = tq
            .iter()
            .map(|&(r, l)| f.all_of(&[r as u16, l as u16]).unwrap())
            .collect();
        let tq_guid: Vec<(Guid128, Guid128)> = ts
            .iter()
            .map(|&(r, l)| (w.f.groups[r as usize], w.f.groups[l as usize]))
            .collect();
        // Equal answers on every user before timing.
        for &i in &all_users {
            let u = ord.user[i];
            c.groups_of(u, Chain::All, &mut prow);
            let fr = f.row_of(&c, u);
            for (q, &(r, l)) in tq.iter().enumerate() {
                let want = bit(&prow, r) && bit(&prow, l);
                assert_eq!(holds(&fr, &masks[q]), want);
                assert_eq!(holds(&fu.rows[u as usize], &masks[q]), want);
            }
        }
        let users_ord: Vec<u16> = all_users.iter().map(|&i| ord.user[i]).collect();
        let eval_bp = |n: usize, prow: &mut Vec<u64>| {
            let mut hits = 0usize;
            for e in 0..n {
                let u = users_ord[e % users_ord.len()];
                c.groups_of(u, Chain::All, prow);
                for &(r, l) in &tq {
                    hits += usize::from(bit(prow, r) && bit(prow, l));
                }
            }
            hits
        };
        let eval_f = |f: &Focus, n: usize| {
            let mut hits = 0usize;
            for e in 0..n {
                let u = users_ord[e % users_ord.len()];
                let r = f.row_of(&c, u);
                for m in &masks {
                    hits += usize::from(holds(&r, m));
                }
            }
            hits
        };
        let eval_fu = |fu: &FocusUsers, n: usize| {
            let mut hits = 0usize;
            for e in 0..n {
                let r = &fu.rows[users_ord[e % users_ord.len()] as usize];
                for m in &masks {
                    hits += usize::from(holds(r, m));
                }
            }
            hits
        };
        let m = users_ord.len();
        let t_a = time(3, || {
            sample
                .iter()
                .map(|&i| {
                    let a = a_groups(&v, &w.f.users[i], Chain::All);
                    tq_guid
                        .iter()
                        .filter(|(r, l)| a.binary_search(r).is_ok() && a.binary_search(l).is_ok())
                        .count()
                })
                .sum::<usize>()
        });
        let t_bp = time(9, || eval_bp(m, &mut prow));
        let t_f = time(9, || eval_f(&f, m));
        let t_fu = time(9, || eval_fu(&fu, m));
        let t_fprep = time(9, || Focus::bind(&c, &groups));
        let t_fuprep = time(5, || FocusUsers::bind(&c, &groups));
        let mut totals = Vec::new();
        for n in [1usize, 10, 100, 1_000, 10_000] {
            let reps = if n <= 100 { 51 } else { 9 };
            let bp = time(reps, || eval_bp(n, &mut prow));
            let ff = time(reps, || {
                let f = Focus::bind(&c, &groups);
                eval_f(&f, n)
            });
            let fuu = time(reps, || {
                let fu = FocusUsers::bind(&c, &groups);
                eval_fu(&fu, n)
            });
            totals.push(format!("{:.1} / {:.1} / {:.1} µs", us(bp), us(ff), us(fuu)));
        }
        println!(
            "| {} | {} | {:.0} ns | {:.0} ns | {:.0} ns | {:.0} ns | {:.1} µs | {:.1} µs | {} |",
            gs.len(),
            ts.len(),
            ns(t_a, sample.len()),
            ns(t_bp, m),
            ns(t_f, m),
            ns(t_fu, m),
            us(t_fprep),
            us(t_fuprep),
            totals.join(" | ")
        );
    }
    // Cold vs warm for one evaluation, 256-slot focus.
    {
        let (ts, gs) = focus_for(&all_templates, 256);
        let groups: Vec<u16> = gs.iter().map(|&g| ord.group[g as usize]).collect();
        let f = Focus::bind(&c, &groups);
        let masks: Vec<_> = ts
            .iter()
            .map(|&(r, l)| {
                f.all_of(&[ord.group[r as usize], ord.group[l as usize]])
                    .unwrap()
            })
            .collect();
        let tq: Vec<(usize, usize)> = ts
            .iter()
            .map(|&(r, l)| {
                (
                    ord.group[r as usize] as usize,
                    ord.group[l as usize] as usize,
                )
            })
            .collect();
        let mut cold_bp = Vec::new();
        let mut cold_f = Vec::new();
        for e in 0..41 {
            let u = ord.user[(e * 197) % w.f.users.len()];
            evict(&mut evict_buf);
            let s = Instant::now();
            c.groups_of(u, Chain::All, &mut prow);
            black_box(
                tq.iter()
                    .filter(|&&(r, l)| bit(&prow, r) && bit(&prow, l))
                    .count(),
            );
            cold_bp.push(s.elapsed());
            evict(&mut evict_buf);
            let s = Instant::now();
            let r = f.row_of(&c, u);
            black_box(masks.iter().filter(|m| holds(&r, m)).count());
            cold_f.push(s.elapsed());
        }
        println!(
            "\nCold single evaluation (256-slot focus, {} questions, after evicting 64 MiB): B+ {:.0} ns, F {:.0} ns.\n",
            ts.len(),
            us(median(cold_bp)) * 1e3,
            us(median(cold_f)) * 1e3
        );
    }

    // ---- W6 team membership x attribute -----------------------------------
    let mut lane = vec![0u8; n_users];
    for (i, &a) in w.f.attr.iter().enumerate() {
        lane[ord.user[i] as usize] = a;
    }
    let teams: Vec<usize> = w.f.teams.iter().map(|&t| t as usize).collect();
    let mut attr_mask = vec![0u64; uw];
    let a6 = time(5, || {
        let mut n = 0;
        for &t in &teams {
            for g in a_members(&v, &w.f.groups[t], Chain::All) {
                let o = v.user_ordinal(&g).unwrap().0 as usize;
                n += usize::from(lane[o] == 3);
            }
        }
        n
    });
    let c6 = time(9, || {
        let mut n = 0;
        for &t in &teams {
            c.members_of(ord.group[t], Chain::All, &mut members);
            eq_u8_to_mask(&lane, 3, &mut attr_mask);
            mask_and_assign(&mut members, &attr_mask);
            n += popcount_batch_u64(&members);
        }
        n
    });
    let check_a = {
        let mut n = 0;
        for &t in &teams {
            for g in a_members(&v, &w.f.groups[t], Chain::All) {
                let o = v.user_ordinal(&g).unwrap().0 as usize;
                n += usize::from(lane[o] == 3);
            }
        }
        n as u64
    };
    let check_c = {
        let mut n = 0;
        for &t in &teams {
            c.members_of(ord.group[t], Chain::All, &mut members);
            eq_u8_to_mask(&lane, 3, &mut attr_mask);
            mask_and_assign(&mut members, &attr_mask);
            n += popcount_batch_u64(&members);
        }
        n
    };
    assert_eq!(check_a, check_c);
    println!("## W6: team transitive members AND attribute = value\n");
    println!(
        "{} teams: A {:.1} µs per team, B+ with `eq_u8_to_mask` + AND + popcount {:.1} µs per team. F does not apply: the focus selects groups, the attribute is per user.\n",
        teams.len(),
        us(a6) / teams.len() as f64,
        us(c6) / teams.len() as f64
    );

    // ---- W7/W8 overlay: reconciliation and refresh ------------------------
    let c0 = Compiled::compile(w.prepare());
    let mut rng = dir_rmap_probe::fixture::Rng::new(11);
    let mut edits = Vec::new();
    for _ in 0..(w.f.users.len() / 100).max(1) {
        let u = rng.below(w.f.users.len()) as u32;
        let t = w.f.teams[rng.below(w.f.teams.len())];
        if w.direct.binary_search(&(u, t)).is_err() {
            edits.push(Edit::AddDirect(u, t));
        }
    }
    let removals: Vec<(u32, u32)> = w
        .direct
        .iter()
        .step_by(97)
        .copied()
        .take(edits.len())
        .collect();
    edits.extend(removals.iter().map(|&(u, g)| Edit::RemoveDirect(u, g)));
    let membership_edits = edits.len();
    let nest = (w.f.teams[0], w.f.roles[5]);
    edits.push(Edit::AddNested(nest.0, nest.1));
    w.apply(&edits);
    let v1 = w.version;
    let view0 = w.st.view(v0).unwrap();
    let view1 = w.view();
    let ord1 = Ordinals::of(&w.f, &view1);
    let full1 = Compiled::compile(w.prepare());
    // A: both versions, on every user up to 20,000 users, else on the
    // sample (time extrapolated to every user, labelled).
    let a7_users: Vec<usize> = if users > 20_000 {
        sample.clone()
    } else {
        all_users.clone()
    };
    let a7s = Instant::now();
    let mut changed_a = 0;
    for &i in &a7_users {
        let u = &w.f.users[i];
        changed_a += usize::from(view0.groups_transitive(u) != view1.groups_transitive(u));
    }
    let a7 = a7s
        .elapsed()
        .mul_f64(w.f.users.len() as f64 / a7_users.len() as f64);
    let (mut r0, mut r1) = (vec![0u64; gw], vec![0u64; gw]);
    let c7 = time(9, || {
        let mut changed = 0;
        for i in 0..w.f.users.len() {
            c0.groups_of(ord.user[i], Chain::All, &mut r0);
            full1.groups_of(ord1.user[i], Chain::All, &mut r1);
            changed += usize::from(r0 != r1);
        }
        changed
    });
    let mut changed_c = 0;
    for &i in &a7_users {
        c0.groups_of(ord.user[i], Chain::All, &mut r0);
        full1.groups_of(ord1.user[i], Chain::All, &mut r1);
        changed_c += usize::from(r0 != r1);
    }
    assert_eq!(changed_a, changed_c, "reconciliation disagrees");
    // Refresh paths.
    let t_full_b = time(5, || w.prepare());
    let t_full_c = time(3, || Compiled::compile(w.prepare()));
    let t_direct = time(9, || {
        let mut p = w.prepare();
        let s = Instant::now();
        p.set_direct(&ord1, &w.direct);
        s.elapsed()
    });
    let mut set_direct_only = Vec::new();
    for _ in 0..9 {
        let mut p = w.prepare();
        let s = Instant::now();
        p.set_direct(&ord1, &w.direct);
        set_direct_only.push(s.elapsed());
        black_box(p);
    }
    let mut inc = Vec::new();
    for _ in 0..9 {
        let mut c = Compiled::compile({
            let mut p = w.prepare();
            p.nested
                .retain(|&x| x != (ord1.group[nest.0 as usize], ord1.group[nest.1 as usize]));
            let (u, d) = {
                let mut up: Vec<(u16, u16)> = p.nested.clone();
                let mut dn: Vec<(u16, u16)> = p.nested.iter().map(|&(c, p)| (p, c)).collect();
                (
                    dir_rmap_probe::prep::Csr::build(p.groups, &mut up),
                    dir_rmap_probe::prep::Csr::build(p.groups, &mut dn),
                )
            };
            p.up = u;
            p.down = d;
            p
        });
        let s = Instant::now();
        c.add_nesting(ord1.group[nest.0 as usize], ord1.group[nest.1 as usize]);
        inc.push(s.elapsed());
        for chain in [Chain::All, Chain::Security, Chain::Mail] {
            assert_eq!(
                c.up[chain as usize], full1.up[chain as usize],
                "incremental != full"
            );
        }
    }
    let t_rebind = time(9, || {
        Focus::bind(&full1, &focus_ids_for(&full1, &all_templates, &ord1))
    });
    println!("## W7/W8: overlay ({membership_edits} membership edits + 1 nesting edge)\n");
    println!("| step | time |\n|---|---|");
    println!(
        "| W7 reconciliation, all users, A (2 versions) | {:.1} ms{} ({changed_a} of {} checked users changed) |",
        a7.as_secs_f64() * 1e3,
        if a7_users.len() < w.f.users.len() {
            ", extrapolated from the sample"
        } else {
            ""
        },
        a7_users.len()
    );
    println!(
        "| W7 reconciliation, all users, B+ (2 compiled versions) | {:.1} ms |",
        c7.as_secs_f64() * 1e3
    );
    println!(
        "| W8 B full rebuild | {:.2} ms |",
        t_full_b.as_secs_f64() * 1e3
    );
    println!(
        "| W8 B + B+ full recompile | {:.2} ms |",
        t_full_c.as_secs_f64() * 1e3
    );
    println!(
        "| W8 membership edits: rebuild the two membership CSRs only (closures untouched) | {:.2} ms |",
        median(set_direct_only).as_secs_f64() * 1e3
    );
    let _ = t_direct;
    println!(
        "| W8 one nesting edge: incremental closure insertion | {:.1} µs |",
        us(median(inc))
    );
    println!(
        "| W8 F (256 slots) rebind after the change | {:.1} µs |\n",
        us(t_rebind)
    );

    // ---- W9 snapshot rebuild with shifted ordinals ------------------------
    let mut fx2 = Fixture::generate(Spec::iam(users));
    fx2.users.insert(0, Guid128([0; 16]));
    fx2.attr.insert(0, 0);
    for e in fx2.direct.iter_mut() {
        e.0 += 1;
    }
    let s = Instant::now();
    let w2 = World::new(fx2);
    let t_snap2 = s.elapsed();
    let ord2 = Ordinals::of(&w2.f, &w2.view());
    let shifted = (0..w.f.users.len())
        .filter(|&i| ord2.user[i + 1] != ord.user[i])
        .count();
    let stale = c.b.check(&w2.witness()).is_err();
    assert!(stale, "a prepared structure must refuse a rebuilt snapshot");
    let t_rebuild = time(3, || Compiled::compile(w2.prepare()));
    println!("## W9: snapshot rebuild with one user inserted first\n");
    println!(
        "{shifted} of {} user ordinals moved; the old B+ refuses (witness: version and size). Snapshot {:.1} ms, B + B+ rebuild {:.2} ms. Remapping old ordinals was not built.\n",
        w.f.users.len(),
        t_snap2.as_secs_f64() * 1e3,
        t_rebuild.as_secs_f64() * 1e3
    );
    let _ = v1;
    let _ = UserOrdinal(0);
}

fn focus_ids_for(c: &Compiled, templates: &[(u32, u32)], ord: &Ordinals) -> Vec<u16> {
    let _ = c;
    let (_, gs) = focus_for(templates, 256);
    gs.iter().map(|&g| ord.group[g as usize]).collect()
}
