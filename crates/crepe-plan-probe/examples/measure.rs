//! Stage costs of the spike, each measured on its own, with and without the
//! rewrites. Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example measure
//! ```
//!
//! The query is `Answer(17, z)` over the `family` program. Each size `n` is the
//! number of `Parent` rows. `x = 17` has `k = 8` children; the other parents
//! are random, so the shared domain grows with `n` while the answer does not.
//! Parity is checked against datafrog at every size and against the nested-loop
//! reference up to 10,000 rows. Timings are medians of `REPS` runs.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use crepe_plan_probe::logic::{Labels, Rule, resolve};
use crepe_plan_probe::mork_api::{admit, query_multi};
use crepe_plan_probe::native::{StarPlan, Store, bindings, joint, prepare, prepare_semijoin, star};
use crepe_plan_probe::optimize::{Demand, Options, ResultKind, optimize};
use crepe_plan_probe::programs::family;
use crepe_plan_probe::reference;
use lance_graph_relalg::{Measure, Rel, normalize};
use mork_pathmap_probe::Semantics;

const REPS: usize = 31;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> Duration {
    let mut v = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        black_box(f());
        v.push(t.elapsed());
    }
    median(v)
}

/// `(Parent, Allowed, Sponsor)` rows.
type Rows = (Vec<[u32; 2]>, Vec<[u32; 2]>, Vec<[u32; 2]>);

fn data(n: usize) -> Rows {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) as u32
    };
    let mut parent = Vec::with_capacity(n);
    for c in 0..8u32 {
        parent.push([17, 100 + c]);
    }
    let people = (n as u32 / 2).max(64);
    while parent.len() < n {
        let a = 1000 + next() % people;
        let b = 1000 + next() % people;
        parent.push([a, b]);
    }
    for c in 0..8u32 {
        for g in 0..3u32 {
            parent.push([100 + c, 1000 + (c * 3 + g) % people]);
        }
    }
    let allowed: Vec<[u32; 2]> = (0..8u32)
        .filter(|c| c % 2 == 0)
        .map(|c| [100 + c, 100 + c])
        .chain((0..people / 4).map(|i| [1000 + i, 1000 + i]))
        .collect();
    let sponsor: Vec<[u32; 2]> = (0..n as u32 / 4).map(|i| [i, i]).collect();
    (parent, allowed, sponsor)
}

fn main() {
    let tier = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "baseline x86-64 (compile-time features)"
    };
    println!("# crepe-plan-probe measure, release, compile-time target features: {tier}");
    println!(
        "n,opts,plan_ns,resolve_ns,optimize_ns,map_admit_ns,normalize_ns,prepare_us,mask_ns,terminal_us,rows_prepared,rules,programs,mask_ops,answers,parity"
    );
    for n in [100usize, 1_000, 10_000, 50_000] {
        let (parent, allowed, sponsor) = data(n);
        let plan_t = time(family::crepe_plan);
        let plan = family::crepe_plan();
        let labels = Labels::of(&plan);
        let resolve_t = time(|| resolve(&plan, &labels));
        let rules: Vec<Rule> = resolve(&plan, &labels)
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        let st = Store {
            relations: vec![
                (labels.id("Parent").unwrap(), &parent, 1),
                (labels.id("Allowed").unwrap(), &allowed, 1),
                (labels.id("Sponsor").unwrap(), &sponsor, 1),
            ],
        };
        let demand = Demand {
            goal: labels.id("Answer").unwrap(),
            bound: vec![Some(17), None],
        };
        let kind = ResultKind::Bindings(Semantics::Set);

        // Oracles.
        let df = datafrog_answer(&parent, &allowed);
        let reference = (n <= 10_000).then(|| {
            let rows = |rel| {
                let (r, _) = st.view(rel);
                r.iter()
                    .map(|x| {
                        x[..labels.arity(rel)]
                            .iter()
                            .map(|&v| u64::from(v))
                            .collect()
                    })
                    .collect()
            };
            reference::eval(&rules, demand.goal, &demand.bound, &rows)
                .into_keys()
                .collect::<Vec<_>>()
        });

        let no_semijoin = Options {
            semijoin: false,
            ..Options::ALL
        };
        for (name, opts) in [
            ("all", Options::ALL),
            ("no-semijoin", no_semijoin),
            ("none", Options::NONE),
        ] {
            let optimize_t = time(|| optimize(&rules, &demand, kind, opts));
            let opt = optimize(&rules, &demand, kind, opts);
            let map_t = time(|| {
                opt.rules
                    .iter()
                    .map(|(_, r)| {
                        let q = query_multi(r);
                        admit(&q).unwrap();
                        star(&q).unwrap()
                    })
                    .collect::<Vec<StarPlan>>()
            });
            let stars: Vec<StarPlan> = opt
                .rules
                .iter()
                .map(|(_, r)| star(&query_multi(r)).unwrap())
                .collect();
            let rels: Vec<Rel> = stars
                .iter()
                .map(|s| Rel::And((0..s.sides.len() as u16).map(Rel::Plane).collect()))
                .collect();
            let normalize_t = time(|| rels.iter().map(normalize).collect::<Vec<_>>());
            let prep = |s: &StarPlan| {
                if opts.semijoin {
                    prepare_semijoin(s, &st)
                } else {
                    prepare(s, &st)
                }
            };
            let prepare_t = time(|| stars.iter().map(prep).collect::<Vec<_>>());
            let preps: Vec<_> = stars.iter().map(prep).collect();
            let gens = |s: &StarPlan| -> Vec<u64> { vec![1; s.sides.len()] };
            let mask_t = time(|| {
                stars
                    .iter()
                    .zip(&preps)
                    .map(|(s, p)| joint(p, s, &gens(s), Measure::Count, None).unwrap().0)
                    .collect::<Vec<_>>()
            });
            let mut programs = 0;
            let mut ops = 0;
            for (s, p) in stars.iter().zip(&preps) {
                let (run, prog) = joint(p, s, &gens(s), Measure::Count, None).unwrap();
                programs += run.programs;
                ops += prog.map_or(0, |p| p.ops.len());
            }
            let answer = || {
                let mut out: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
                for (s, p) in stars.iter().zip(&preps) {
                    for (row, m) in bindings(p, s, &gens(s), false).unwrap() {
                        *out.entry(row).or_insert(0) += m;
                    }
                }
                out.retain(|row, _| {
                    row.iter()
                        .zip(&opt.post_filter)
                        .all(|(v, f)| f.is_none_or(|f| f == *v))
                });
                out.into_keys().collect::<Vec<_>>()
            };
            let terminal_t = time(answer);
            let got = answer();
            let parity = got == df && reference.as_ref().is_none_or(|r| *r == got);
            println!(
                "{n},{name},{},{},{},{},{},{:.1},{},{:.1},{},{},{programs},{ops},{},{}",
                plan_t.as_nanos(),
                resolve_t.as_nanos(),
                optimize_t.as_nanos(),
                map_t.as_nanos(),
                normalize_t.as_nanos(),
                prepare_t.as_secs_f64() * 1e6,
                mask_t.as_nanos(),
                terminal_t.as_secs_f64() * 1e6,
                preps.iter().map(|p| p.rows_prepared).sum::<usize>(),
                opt.rules.len(),
                got.len(),
                if parity { "ok" } else { "MISMATCH" },
            );
            assert!(parity, "answers differ from the oracles at n = {n}, {name}");
        }
    }
}

/// `Answer(17, z)` by datafrog (set), the family rule only.
fn datafrog_answer(parent: &[[u32; 2]], allowed: &[[u32; 2]]) -> Vec<Vec<u64>> {
    use datafrog::Relation;
    let by_y: Relation<(u32, u32)> = parent.iter().map(|&[a, b]| (b, a)).collect();
    let by_parent: Relation<(u32, u32)> = parent.iter().map(|&[a, b]| (a, b)).collect();
    let allowed: Relation<(u32, ())> = allowed.iter().map(|&[a, _]| (a, ())).collect();
    let xz: Relation<(u32, (u32, u32))> =
        Relation::from_join(&by_y, &by_parent, |&y, &x, &z| (y, (x, z)));
    let ans: Relation<(u32, u32)> = Relation::from_join(&xz, &allowed, |_, &(x, z), _| (x, z));
    ans.elements
        .iter()
        .filter(|(a, _)| *a == 17)
        .map(|&(a, b)| vec![u64::from(a), u64::from(b)])
        .collect()
}
