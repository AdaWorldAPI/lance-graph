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
use dir_rmap_probe::focus::{Focus, FocusLazy, FocusUsers, Row, holds};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals, Witness, bit, ones, words};
use dir_rmap_probe::{
    Edit, World, a_groups, a_members, composite_groups, conjunctions, focus_for, oracle, templates,
};
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
    let (_, a_b) = counted(|| w.prepare());
    let p = w.prepare();
    let (cc, a_comp) = counted(|| Compiled::compile(p));
    drop(cc);
    println!(
        "| allocations: B prepare {} ({} allocated, {} retained); B+ compile {} ({} allocated, {} retained) | |",
        a_b.allocs,
        kib(a_b.bytes as isize),
        kib(a_b.retained),
        a_comp.allocs,
        kib(a_comp.bytes as isize),
        kib(a_comp.retained)
    );
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
    let f256 = Focus::bind(&c, Chain::All, &focus_ids, &w.witness()).unwrap();
    let fu256 = FocusUsers::bind(&c, Chain::All, &focus_ids, &w.witness()).unwrap();
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

    w5_all(&w, &c, &ord, &mut evict_buf, &sample);
    sweep(users, &mut evict_buf);

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
    let rebind_ids = focus_ids_for(&full1, &all_templates, &ord1);
    let t_rebind = time(9, || {
        Focus::bind(&full1, Chain::All, &rebind_ids, &w.witness())
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

// ---------------------------------------------------------------------------
// Allocation accounting: a counting global allocator. Every allocation the
// process makes goes through it; `counted` reports the allocations, bytes
// allocated and bytes still live (retained) across one closure.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged; the counters are side effects only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size(), Relaxed);
        LIVE.fetch_add(l.size() as isize, Relaxed);
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size(), Relaxed);
        LIVE.fetch_add(l.size() as isize, Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Relaxed);
        // SAFETY: `p` came from this allocator with layout `l`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new, Relaxed);
        LIVE.fetch_add(new as isize - l.size() as isize, Relaxed);
        // SAFETY: forwarded unchanged; `p` came from this allocator.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Clone, Copy, Default)]
struct Acct {
    allocs: usize,
    bytes: usize,
    retained: isize,
}

/// Run `f` once, counting what it allocates; the result is kept alive while
/// `retained` is read, so retained = what the result holds.
fn counted<T>(f: impl FnOnce() -> T) -> (T, Acct) {
    let (a0, b0, l0) = (
        ALLOCS.load(Relaxed),
        BYTES.load(Relaxed),
        LIVE.load(Relaxed),
    );
    let t = f();
    let a = Acct {
        allocs: ALLOCS.load(Relaxed) - a0,
        bytes: BYTES.load(Relaxed) - b0,
        retained: LIVE.load(Relaxed) - l0,
    };
    (t, a)
}

fn kib(b: isize) -> String {
    if b.abs() < 10_240 {
        format!("{b} B")
    } else {
        format!("{:.1} KiB", b as f64 / 1024.0)
    }
}

// ---------------------------------------------------------------------------
// W5: questions over group ordinals. A question of one group is a
// membership bit (a materialized composite); of two, a conjunction.

/// The longest prefix of `qs` whose groups fit in `k` slots, and those
/// groups (first-seen order).
fn take_k(qs: &[Vec<u16>], k: usize) -> (Vec<Vec<u16>>, Vec<u16>) {
    let mut groups: Vec<u16> = Vec::new();
    let mut kept = Vec::new();
    for q in qs {
        let new = q.iter().filter(|g| !groups.contains(g)).count();
        if groups.len() + new > k {
            continue;
        }
        for &g in q {
            if !groups.contains(&g) {
                groups.push(g);
            }
        }
        kept.push(q.clone());
    }
    (kept, groups)
}

/// B+: the user's full closure row (every group), then the question bits.
fn e_bp(c: &Compiled, u: u16, qs: &[Vec<u16>], row: &mut [u64]) -> usize {
    c.groups_of(u, Chain::All, row);
    qs.iter()
        .filter(|q| q.iter().all(|&g| bit(row, g as usize)))
        .count()
}

/// B+ reading only the closure words that hold a question group: the global
/// closure, question-aware, with no focus.
fn e_bpw(c: &Compiled, u: u16, qs: &[Vec<u16>], wds: &[usize], acc: &mut [u64]) -> usize {
    for &k in wds {
        acc[k] = 0;
    }
    for &d in c.b.user_groups.row(u as usize) {
        let r = c.row(Chain::All, d as usize);
        for &k in wds {
            acc[k] |= r[k];
        }
    }
    qs.iter()
        .filter(|q| q.iter().all(|&g| bit(acc, g as usize)))
        .count()
}

/// A question compiled onto focus slots: the slots it needs set.
type Slots = Vec<u8>;

#[inline]
fn slot_bit(r: &Row, s: u8) -> bool {
    r[(s >> 6) as usize] >> (s & 63) & 1 == 1
}

#[inline]
fn answer(r: &Row, qs: &[Slots]) -> usize {
    qs.iter()
        .filter(|q| q.iter().all(|&s| slot_bit(r, s)))
        .count()
}

/// F: the user's focus row (OR of its direct groups' projected rows), then
/// one 4-word mask test per question on the 256-bit row.
fn e_f(f: &Focus, c: &Compiled, u: u16, masks: &[Row]) -> usize {
    let r = f.row_of(c, u);
    masks.iter().filter(|m| holds(&r, m)).count()
}

fn e_fu(fu: &FocusUsers, u: u16, masks: &[Row]) -> usize {
    let r = &fu.rows[u as usize];
    masks.iter().filter(|m| holds(r, m)).count()
}

/// F with the questions as slot bits instead of masks: the same work per
/// question as B+'s bit tests, on the focus row.
fn e_f_bits(f: &Focus, c: &Compiled, u: u16, qs: &[Slots]) -> usize {
    answer(&f.row_of(c, u), qs)
}

fn slots_of(f: &Focus, qs: &[Vec<u16>]) -> Vec<Slots> {
    qs.iter()
        .map(|q| q.iter().map(|&g| f.slot(g).unwrap() as u8).collect())
        .collect()
}

/// Per-evaluation figures of one (workload, k).
struct Evals {
    bp: f64,
    bpw: f64,
    f: f64,
    fu: f64,
}

/// Run `n` evaluations of each arm, users cycled.
#[allow(clippy::too_many_arguments)]
fn time_evals(
    c: &Compiled,
    users: &[u16],
    qs: &[Vec<u16>],
    wds: &[usize],
    f: &Focus,
    fu: &FocusUsers,
    masks: &[Row],
    n: usize,
    reps: usize,
) -> (Evals, [usize; 4]) {
    let mut row = vec![0u64; c.w];
    let mut acc = vec![0u64; c.w];
    let pick = |e: usize| users[e % users.len()];
    let mut allocs = [0usize; 4];
    let ((), a) = counted(|| {
        for e in 0..n {
            black_box(e_bp(c, pick(e), qs, &mut row));
        }
    });
    allocs[0] = a.allocs;
    let ((), a) = counted(|| {
        for e in 0..n {
            black_box(e_bpw(c, pick(e), qs, wds, &mut acc));
        }
    });
    allocs[1] = a.allocs;
    let ((), a) = counted(|| {
        for e in 0..n {
            black_box(e_f(f, c, pick(e), masks));
        }
    });
    allocs[2] = a.allocs;
    let ((), a) = counted(|| {
        for e in 0..n {
            black_box(e_fu(fu, pick(e), masks));
        }
    });
    allocs[3] = a.allocs;
    let bp = time(reps, || {
        (0..n)
            .map(|e| e_bp(c, pick(e), qs, &mut row))
            .sum::<usize>()
    });
    let bpw = time(reps, || {
        (0..n)
            .map(|e| e_bpw(c, pick(e), qs, wds, &mut acc))
            .sum::<usize>()
    });
    let tf = time(reps, || {
        (0..n).map(|e| e_f(f, c, pick(e), masks)).sum::<usize>()
    });
    let tfu = time(reps, || {
        (0..n).map(|e| e_fu(fu, pick(e), masks)).sum::<usize>()
    });
    (
        Evals {
            bp: ns(bp, n),
            bpw: ns(bpw, n),
            f: ns(tf, n),
            fu: ns(tfu, n),
        },
        allocs,
    )
}

/// Median cold time (after evicting 64 MiB) of one evaluation per arm.
#[allow(clippy::too_many_arguments)]
fn time_cold(
    c: &Compiled,
    users: &[u16],
    qs: &[Vec<u16>],
    wds: &[usize],
    f: &Focus,
    fu: &FocusUsers,
    masks: &[Row],
    evict_buf: &mut [u64],
) -> [f64; 4] {
    let mut row = vec![0u64; c.w];
    let mut acc = vec![0u64; c.w];
    let mut t: [Vec<Duration>; 4] = Default::default();
    for e in 0..31 {
        let u = users[(e * 197) % users.len()];
        evict(evict_buf);
        let s = Instant::now();
        black_box(e_bp(c, u, qs, &mut row));
        t[0].push(s.elapsed());
        evict(evict_buf);
        let s = Instant::now();
        black_box(e_bpw(c, u, qs, wds, &mut acc));
        t[1].push(s.elapsed());
        evict(evict_buf);
        let s = Instant::now();
        black_box(e_f(f, c, u, masks));
        t[2].push(s.elapsed());
        evict(evict_buf);
        let s = Instant::now();
        black_box(e_fu(fu, u, masks));
        t[3].push(s.elapsed());
    }
    t.map(|v| ns(median(v), 1))
}

/// Distinct closure-word indices holding a focus group.
fn words_of(groups: &[u16]) -> Vec<usize> {
    let mut k: Vec<usize> = groups.iter().map(|&g| g as usize >> 6).collect();
    k.sort_unstable();
    k.dedup();
    k
}

/// One W5 workload over `all_qs`, for each focus size.
#[allow(clippy::too_many_arguments)]
fn w5(
    title: &str,
    c: &Compiled,
    witness: &Witness,
    fresh: &dyn Fn() -> Compiled,
    edges: &[(u16, u16)],
    edited: &[u16],
    all_qs: &[Vec<u16>],
    evict_buf: &mut [u64],
) {
    let users: Vec<u16> = (0..c.b.users).map(|u| u as u16).collect();
    let m = users.len();
    println!("### {title}\n");
    let mut evals_tbl = Vec::new();
    let mut acct_tbl = Vec::new();
    let mut amort_tbl = Vec::new();
    for k in [16usize, 64, 128, 256] {
        let (qs, groups) = take_k(all_qs, k);
        if qs.is_empty() {
            continue;
        }
        let wds = words_of(&groups);
        // Preparation, phase by phase, counted.
        let (mut f, a_slots) = counted(|| Focus::slots(c, Chain::All, &groups, witness).unwrap());
        let ((), a_proj) = counted(|| f.project(c));
        let mut fref = Focus::slots(c, Chain::All, &groups, witness).unwrap();
        let ((), a_ref) = counted(|| fref.project_by_test(c));
        let masks: Vec<Row> = qs.iter().map(|q| f.all_of(q).unwrap()).collect();
        let sl = slots_of(&f, &qs);
        let (fu, a_users) = counted(|| {
            FocusUsers::from_focus(c, Focus::bind(c, Chain::All, &groups, witness).unwrap())
        });
        // Every arm agrees with B+ on every user before anything is timed.
        let mut row = vec![0u64; c.w];
        let mut acc = vec![0u64; c.w];
        for &u in &users {
            let want = e_bp(c, u, &qs, &mut row);
            assert_eq!(e_bpw(c, u, &qs, &wds, &mut acc), want);
            assert_eq!(e_f(&f, c, u, &masks), want);
            assert_eq!(e_fu(&fu, u, &masks), want);
            assert_eq!(e_f_bits(&f, c, u, &sl), want);
        }
        let t_bits = time(9, || {
            (0..m)
                .map(|e| e_f_bits(&f, c, users[e], &sl))
                .sum::<usize>()
        });
        for g in 0..c.b.groups {
            assert_eq!(f.projected(g), fref.projected(g));
        }
        let t_slots = time(15, || {
            Focus::slots(c, Chain::All, &groups, witness).unwrap()
        });
        let t_proj = {
            let mut v = Vec::new();
            for _ in 0..15 {
                let mut g = Focus::slots(c, Chain::All, &groups, witness).unwrap();
                let s = Instant::now();
                g.project(c);
                v.push(s.elapsed());
                black_box(g);
            }
            median(v)
        };
        let t_ref = {
            let mut v = Vec::new();
            for _ in 0..7 {
                let mut g = Focus::slots(c, Chain::All, &groups, witness).unwrap();
                let s = Instant::now();
                g.project_by_test(c);
                v.push(s.elapsed());
            }
            median(v)
        };
        let t_users = {
            let mut v = Vec::new();
            for _ in 0..9 {
                let g = Focus::bind(c, Chain::All, &groups, witness).unwrap();
                let s = Instant::now();
                black_box(FocusUsers::from_focus(c, g));
                v.push(s.elapsed());
            }
            median(v)
        };
        // Lazy: cache allocation, then first touch of every user, then warm.
        let (mut fl, a_lazy) = counted(|| {
            FocusLazy::from_focus(c, Focus::bind(c, Chain::All, &groups, witness).unwrap())
        });
        let t_lazy_alloc = time(15, || {
            FocusLazy::from_focus(c, Focus::bind(c, Chain::All, &groups, witness).unwrap())
        })
        .saturating_sub(time(15, || {
            Focus::bind(c, Chain::All, &groups, witness).unwrap()
        }));
        let s = Instant::now();
        for &u in &users {
            black_box(fl.row(c, u));
        }
        let t_lazy_first = s.elapsed();
        assert_eq!(fl.filled, m);
        let t_lazy_warm = time(9, || {
            users
                .iter()
                .map(|&u| {
                    let r = fl.row(c, u);
                    masks.iter().filter(|mm| holds(&r, mm)).count()
                })
                .sum::<usize>()
        });
        // Refresh after membership edits (the edited users' rows).
        let mut fu2 =
            FocusUsers::from_focus(c, Focus::bind(c, Chain::All, &groups, witness).unwrap());
        let ((), a_ru) = counted(|| fu2.refresh_users(c, edited));
        let t_ru = time(15, || fu2.refresh_users(c, edited));
        let line = |phase: &str, t: f64, a: Acct| {
            format!(
                "| {} | {phase} | {t:.1} µs | {} | {} | {} |",
                groups.len(),
                a.allocs,
                kib(a.bytes as isize),
                kib(a.retained)
            )
        };
        // Refresh after one nesting pair: B+, then F, then F+users, for a
        // narrow edge and a wide one (a child with a large downward closure).
        let mut nest_rows = Vec::new();
        for &edge in edges {
            let (mut t_bn, mut t_fn, mut t_fun) = (Vec::new(), Vec::new(), Vec::new());
            let (mut a_bn, mut a_fn, mut a_fun) =
                (Acct::default(), Acct::default(), Acct::default());
            for _ in 0..5 {
                let mut cc = fresh();
                let mut fx = Focus::bind(&cc, Chain::All, &groups, witness).unwrap();
                let mut fux = FocusUsers::from_focus(
                    &cc,
                    Focus::bind(&cc, Chain::All, &groups, witness).unwrap(),
                );
                let s = Instant::now();
                let ((), a) = counted(|| cc.add_nesting(edge.0, edge.1));
                t_bn.push(s.elapsed());
                a_bn = a;
                let s = Instant::now();
                let ((), a) = counted(|| fx.refresh_nesting(&cc, edge.0));
                t_fn.push(s.elapsed());
                a_fn = a;
                let s = Instant::now();
                let ((), a) = counted(|| fux.refresh_nesting(&cc, edge.0));
                t_fun.push(s.elapsed());
                a_fun = a;
                let check = Focus::bind(&cc, Chain::All, &groups, witness).unwrap();
                for g in 0..cc.b.groups {
                    assert_eq!(fx.projected(g), check.projected(g), "refresh != rebind");
                }
                for &u in &users {
                    assert_eq!(fux.rows[u as usize], check.row_of(&cc, u), "users refresh");
                }
            }
            let refreshed = ones(c.down_row(Chain::All, edge.0 as usize)).count();
            nest_rows.push(line(
            &format!("refresh after 1 nesting pair (child closure {refreshed} groups): B+ closure insertion"),
            us(median(t_bn)),
            a_bn,
        ));
            nest_rows.push(line(
                "  … F re-projects those groups",
                us(median(t_fn)),
                a_fn,
            ));
            nest_rows.push(line(
                "  … F+users: re-project + recompute their direct users",
                us(median(t_fun)),
                a_fun,
            ));
        }
        // Evaluations.
        let (ev, allocs) = time_evals(c, &users, &qs, &wds, &f, &fu, &masks, m, 9);
        let cold = time_cold(c, &users, &qs, &wds, &f, &fu, &masks, evict_buf);
        evals_tbl.push(format!(
            "| {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} / {:.0} / {:.0} / {:.0} | {:?} |",
            groups.len(),
            qs.len(),
            wds.len(),
            ev.bp,
            ev.bpw,
            ev.f,
            ns(t_bits, m),
            ev.fu,
            ns(t_lazy_warm, m),
            cold[0],
            cold[1],
            cold[2],
            cold[3],
            allocs
        ));
        acct_tbl.push(line(
            "focus construction (slots + reverse map)",
            us(t_slots),
            a_slots,
        ));
        acct_tbl.push(line("closure projection (down rows)", us(t_proj), a_proj));
        acct_tbl.push(line(
            "closure projection by test (reference)",
            us(t_ref),
            a_ref,
        ));
        acct_tbl.push(line("eager per-user rows, all users", us(t_users), {
            let mut a = a_users;
            // `a_users` includes a bind; subtract its slots and projection.
            a.allocs -= a_slots.allocs + a_proj.allocs;
            a.bytes -= a_slots.bytes + a_proj.bytes;
            a.retained -= a_slots.retained + a_proj.retained;
            a
        }));
        acct_tbl.push(line(
            &format!(
                "lazy per-user rows: allocate the cache (then first touch {:.0} ns per user, filled on use)",
                ns(t_lazy_first, m)
            ),
            us(t_lazy_alloc),
            {
                let mut a = a_lazy;
                a.allocs -= a_slots.allocs + a_proj.allocs;
                a.bytes -= a_slots.bytes + a_proj.bytes;
                a.retained -= a_slots.retained + a_proj.retained;
                a
            },
        ));
        acct_tbl.push(line(
            &format!(
                "refresh after membership edits ({} users), eager rows",
                edited.len()
            ),
            us(t_ru),
            a_ru,
        ));
        acct_tbl.extend(nest_rows);
        // Amortization: computed crossover and measured crossover.
        let p_bind = ns(t_slots, 1);
        let p_proj = ns(t_proj, 1);
        let p_users = ns(t_users, 1);
        let nstar = |p: f64, base: f64, e: f64| {
            if base > e {
                format!("{:.0}", p / (base - e))
            } else {
                "never".into()
            }
        };
        let mut measured = |arm: u8| -> String {
            for n in [
                1usize, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
                50_000,
            ] {
                let reps = if n <= 1_000 { 21 } else { 5 };
                let base = time(reps, || {
                    (0..n)
                        .map(|e| e_bp(c, users[e % m], &qs, &mut row))
                        .sum::<usize>()
                });
                let t = match arm {
                    0 => time(reps, || {
                        let f = Focus::bind(c, Chain::All, &groups, witness).unwrap();
                        (0..n)
                            .map(|e| e_f(&f, c, users[e % m], &masks))
                            .sum::<usize>()
                    }),
                    _ => time(reps, || {
                        let fu = FocusUsers::bind(c, Chain::All, &groups, witness).unwrap();
                        (0..n)
                            .map(|e| e_fu(&fu, users[e % m], &masks))
                            .sum::<usize>()
                    }),
                };
                if t <= base {
                    return format!("{n}");
                }
            }
            "> 50000".into()
        };
        // Rebuilding a focus for each question: bind + project + 1 evaluation.
        let t_rebuild = time(9, || {
            (0..20)
                .map(|e| {
                    let f = Focus::bind(c, Chain::All, &groups, witness).unwrap();
                    e_f(&f, c, users[(e * 31) % m], &masks)
                })
                .sum::<usize>()
        });
        amort_tbl.push(format!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {} | {} | {} | {} | {:.1} µs |",
            groups.len(),
            p_bind / 1e3,
            p_proj / 1e3,
            p_users / 1e3,
            ev.bp,
            ev.bpw,
            ev.f,
            ev.fu,
            nstar(p_bind + p_proj, ev.bp, ev.f),
            nstar(p_bind + p_proj, ev.bpw, ev.f),
            nstar(p_bind + p_proj + p_users, ev.bp, ev.fu),
            measured(0),
            measured(1),
            us(t_rebuild) / 20.0,
        ));
    }
    println!(
        "Per evaluation (one user, every question), warm, users cycled; cold = after evicting 64 MiB. ns. `words` = closure words that hold a focus group. Allocations counted over {m} evaluations per arm (B+, B+w, F, F+u).\n"
    );
    println!(
        "| focus groups | questions | words | B+ | B+w | F | F (slot bits) | F+users | F lazy (warm) | cold B+ / B+w / F / F+u | allocs |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &evals_tbl {
        println!("{r}");
    }
    println!(
        "\nPreparation accounting (each phase alone; bytes = allocated, retained = still live after the phase):\n"
    );
    println!(
        "| focus groups | phase | time | allocs | bytes | retained |\n|---|---|---|---|---|---|"
    );
    for r in &acct_tbl {
        println!("{r}");
    }
    println!(
        "\nAmortization. T_B+(N) = N·e_B+; T_F(N) = P_bind + P_proj + N·e_F; T_F+u(N) = P_bind + P_proj + P_users + N·e_F+u. B+ compile is shared by every arm and excluded. N* = P / (e_B+ − e_F), computed; \"measured\" = smallest N in {{1, 2, 5, ..., 50000}} at which the directly timed total of F (bind included) is ≤ the timed B+ total. Rebuild = bind + project + one evaluation, per question.\n"
    );
    println!(
        "| focus groups | P_bind µs | P_proj µs | P_users µs | e_B+ ns | e_B+w ns | e_F ns | e_F+u ns | N* F vs B+ | N* F vs B+w | N* F+u vs B+ | measured F | measured F+u | rebuild per question |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &amort_tbl {
        println!("{r}");
    }
    println!();
}

/// W5 on the three question shapes.
fn w5_all(w: &World, c: &Compiled, ord: &Ordinals, evict_buf: &mut [u64], _sample: &[usize]) {
    println!("## W5: role x location and composite questions, B+ vs F\n");
    let witness = w.witness();
    let fresh = || Compiled::compile(w.prepare());
    let f = &w.f;
    // The nesting pair used for the refresh figures: a team into the first
    // role of the focus (absent before).
    let edge = (
        ord.group[f.teams[0] as usize],
        ord.group[f.roles[0] as usize],
    );
    assert!(c.b.nested.binary_search(&edge).is_err());
    let edited: Vec<u16> = (0..(f.users.len() / 100).max(1))
        .map(|i| ord.user[(i * 97) % f.users.len()])
        .collect();
    // (a) Policy conjunctions asked over the composite fixture.
    let conj: Vec<Vec<u16>> = conjunctions(f)
        .iter()
        .map(|&(r, l)| vec![ord.group[r as usize], ord.group[l as usize]])
        .collect();
    let edge_c = (edge.0, conj[0][0]);
    w5(
        "W5a: policy conjunction (in role R and in location L), composite fixture",
        c,
        &witness,
        &fresh,
        &[edge_c, (wide_child(c, edge_c.1), edge_c.1)],
        &edited,
        &conj,
        evict_buf,
    );
    // (b) Materialized composites: one membership bit per question.
    let comp: Vec<Vec<u16>> = composite_groups(f)
        .iter()
        .map(|&g| vec![ord.group[g as usize]])
        .collect();
    w5(
        "W5b: materialized composite membership (in composite C), composite fixture",
        c,
        &witness,
        &fresh,
        &[
            (edge.0, comp[0][0]),
            (wide_child(c, comp[0][0]), comp[0][0]),
        ],
        &edited,
        &comp,
        evict_buf,
    );
    // (c) Policy conjunctions on the fixture with no composites.
    let wc = World::new(Fixture::generate(
        Spec::iam(f.users.len()).without_composites(),
    ));
    let vc = wc.view();
    let ordc = Ordinals::of(&wc.f, &vc);
    let cc = Compiled::compile(wc.prepare());
    let qc: Vec<Vec<u16>> = conjunctions(&wc.f)
        .iter()
        .map(|&(r, l)| vec![ordc.group[r as usize], ordc.group[l as usize]])
        .collect();
    // B+ on this fixture equals the oracle (membership count) and dir-sim
    // on a sample, before W5c is timed.
    let d = oracle::run(&wc.f, &wc.direct, &wc.nested, &[]);
    let mut row = vec![0u64; cc.w];
    let total: usize = (0..wc.f.users.len())
        .map(|i| {
            cc.groups_of(ordc.user[i], Chain::All, &mut row);
            popcount_batch_u64(&row) as usize
        })
        .sum();
    assert_eq!(total, d.member.len());
    for i in (0..wc.f.users.len()).step_by(37) {
        cc.groups_of(ordc.user[i], Chain::All, &mut row);
        let mut got: Vec<Guid128> = ones(&row)
            .map(|g| vc.group_guid(GroupOrdinal(g as u16)).unwrap())
            .collect();
        got.sort_unstable();
        let mut a = a_groups(&vc, &wc.f.users[i], Chain::All);
        a.sort_unstable();
        assert_eq!(got, a);
    }
    let editedc: Vec<u16> = (0..(wc.f.users.len() / 100).max(1))
        .map(|i| ordc.user[(i * 97) % wc.f.users.len()])
        .collect();
    let freshc = || Compiled::compile(wc.prepare());
    let edgec = (ordc.group[wc.f.teams[0] as usize], qc[0][0]);
    println!(
        "Fixture without composites: {} groups, {} conjunction questions; B+ = oracle on all {} memberships, = dir-sim on a sample.\n",
        vc.groups_len(),
        qc.len(),
        d.member.len()
    );
    w5(
        "W5c: policy conjunction, fixture without composites",
        &cc,
        &wc.witness(),
        &freshc,
        &[edgec, (wide_child(&cc, edgec.1), edgec.1)],
        &editedc,
        &qc,
        evict_buf,
    );
}

// ---------------------------------------------------------------------------
// Working-set sweep: more groups, so the B+ closure outgrows L2.

const SWEEP_QUESTIONS: usize = 150;

fn sweep(users: usize, evict_buf: &mut [u64]) {
    let factors: Vec<usize> = std::env::var("DIR_RMAP_SWEEP")
        .unwrap_or_else(|_| "1,2,4,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("sweep factor"))
        .collect();
    println!(
        "## Working-set sweep ({users} users, conjunction questions, focus of 64 groups, {SWEEP_QUESTIONS} questions)\n"
    );
    println!(
        "Caches: {}. A focus row is 32 B; a closure row is `words`·8 B. \"touched\" = bytes read by one evaluation (direct rows + output); \"set\" = bytes of rows read when every user is evaluated in turn (distinct direct groups × row size). Hot = one user repeated; warm = users cycled; cold = after evicting 64 MiB. ns per evaluation.\n",
        cache_sizes()
    );
    println!(
        "| groups | questions | F masks | words | closure, one chain up | B+ set | F set | F+u set | avg direct groups | touched B+ / B+w / F | hot B+ / F | warm B+ / B+w / F / F+u | cold B+ / B+w / F / F+u | compile ms | F prep µs | N* F vs B+ | N* F vs B+w |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for factor in factors {
        let w = World::new(Fixture::generate(Spec::iam_scaled(users, factor)));
        let v = w.view();
        let ord = Ordinals::of(&w.f, &v);
        let witness = w.witness();
        let s = Instant::now();
        let c = Compiled::compile(w.prepare());
        let t_comp = s.elapsed();
        let all: Vec<Vec<u16>> = conjunctions(&w.f)
            .iter()
            .map(|&(r, l)| vec![ord.group[r as usize], ord.group[l as usize]])
            .collect();
        // A fixed number of questions, so the rows differ only in group
        // count (the first 64-group focus holds a different number of
        // conjunctions at each scale).
        let (mut qs, groups) = take_k(&all, 64);
        assert!(qs.len() >= SWEEP_QUESTIONS, "{} questions", qs.len());
        qs.truncate(SWEEP_QUESTIONS);
        let wds = words_of(&groups);
        let f = Focus::bind(&c, Chain::All, &groups, &witness).unwrap();
        let fu =
            FocusUsers::from_focus(&c, Focus::bind(&c, Chain::All, &groups, &witness).unwrap());
        let masks: Vec<Row> = qs.iter().map(|q| f.all_of(q).unwrap()).collect();
        let users_o: Vec<u16> = (0..c.b.users).map(|u| u as u16).collect();
        let mut row = vec![0u64; c.w];
        for &u in &users_o {
            let want = e_bp(&c, u, &qs, &mut row);
            assert_eq!(e_f(&f, &c, u, &masks), want);
            assert_eq!(e_fu(&fu, u, &masks), want);
        }
        let n = c.b.groups;
        let total_deg: usize = users_o
            .iter()
            .map(|&u| c.b.user_groups.row(u as usize).len())
            .sum();
        let avg = total_deg as f64 / users_o.len() as f64;
        let mut distinct = vec![false; n];
        for &u in &users_o {
            for &d in c.b.user_groups.row(u as usize) {
                distinct[d as usize] = true;
            }
        }
        let dg = distinct.iter().filter(|&&x| x).count();
        let set_bp = dg * c.w * 8;
        let set_f = dg * 32;
        let set_fu = users_o.len() * 32;
        let touched = (
            (avg + 1.0) * c.w as f64 * 8.0,
            avg * wds.len() as f64 * 8.0,
            avg * 32.0 + masks.len() as f64 * 32.0,
        );
        let (ev, _) = time_evals(&c, &users_o, &qs, &wds, &f, &fu, &masks, users_o.len(), 7);
        let one = [users_o[users_o.len() / 2]];
        let (hot, _) = time_evals(&c, &one, &qs, &wds, &f, &fu, &masks, 20_000, 7);
        let cold = time_cold(&c, &users_o, &qs, &wds, &f, &fu, &masks, evict_buf);
        let t_prep = time(9, || {
            Focus::bind(&c, Chain::All, &groups, &witness).unwrap()
        });
        let p = ns(t_prep, 1);
        let nstar = |base: f64, e: f64| {
            if base > e {
                format!("{:.0}", p / (base - e))
            } else {
                "never".into()
            }
        };
        println!(
            "| {n} | {} | {} | {} | {} | {} | {} | {} | {avg:.1} | {} / {} / {} | {:.0} / {:.0} | {:.0} / {:.0} / {:.0} / {:.0} | {:.0} / {:.0} / {:.0} / {:.0} | {:.0} | {:.1} | {} | {} |",
            qs.len(),
            kib((masks.len() * 32) as isize),
            c.w,
            kib((n * c.w * 8) as isize),
            kib(set_bp as isize),
            kib(set_f as isize),
            kib(set_fu as isize),
            kib(touched.0 as isize),
            kib(touched.1 as isize),
            kib(touched.2 as isize),
            hot.bp,
            hot.f,
            ev.bp,
            ev.bpw,
            ev.f,
            ev.fu,
            cold[0],
            cold[1],
            cold[2],
            cold[3],
            t_comp.as_secs_f64() * 1e3,
            us(t_prep),
            nstar(ev.bp, ev.f),
            nstar(ev.bpw, ev.f),
        );
    }
    println!();
}

/// The group with the largest downward closure that is not yet nested in
/// `parent`: the widest refresh a single nesting pair into `parent` causes.
fn wide_child(c: &Compiled, parent: u16) -> u16 {
    (0..c.b.groups as u16)
        .filter(|&g| g != parent && c.b.nested.binary_search(&(g, parent)).is_err())
        .max_by_key(|&g| ones(c.down_row(Chain::All, g as usize)).count())
        .unwrap()
}
