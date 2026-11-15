//! Deterministic baselines for XLOG's triangle inner-pairing choice (selectivity pass,
//! `try_reorder_triangle_with_owner`), measured BEFORE any learned model.
//! No model is trained or integrated here. Pre-registered falsifiers F3 / F4 from the
//! feasibility report are the questions this example answers with numbers.
//!
//!   cargo run --release --example triangle_pairing_baselines
//!
//! Candidates: SharedX = join(R_xy,R_xz) on x, SharedY = join(R_xy,R_yz) on y,
//! SharedZ = join(R_yz,R_xz) on z. Exact label = size of the inner join
//! = sum_v degA(v)*degB(v), a fold over per-value degree counts (no join rows built).
//! The materialized brute-force join exists only as a test oracle (`oracle_check`).
use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use xlog_core::{RelId, ScalarType};
use ndarray::hpc::codec8::TiesEven;
use ndarray::hpc::codec9::{Codec9Runtime, EpiLane9};
use xlog_relcomp_probe::pm_ledger::PmLedger;
use xlog_stats::{ColumnStats, StatsManager};

// ------------------------------------------------------------------ rng
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

struct Zipf(Vec<f64>);
impl Zipf {
    fn new(n: usize, s: f64) -> Self {
        let mut acc = 0.0;
        let mut c = Vec::with_capacity(n);
        for r in 0..n {
            acc += 1.0 / ((r + 1) as f64).powf(s);
            c.push(acc);
        }
        let t = acc;
        c.iter_mut().for_each(|x| *x /= t);
        Zipf(c)
    }
    fn sample(&self, rng: &mut Rng) -> u32 {
        let u = rng.f64();
        self.0.partition_point(|&p| p < u).min(self.0.len() - 1) as u32
    }
}

// ------------------------------------------------------------------ families
#[derive(Clone, Copy, PartialEq)]
enum Family {
    Uniform,
    ZipfAll,
    ZipfY,
    StarHub,
    Cluster,
    AntiCorr,
    TwinHubY, // twin only: hub values on the shared variable y
}
impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Uniform => "uniform",
            Family::ZipfAll => "zipf_all",
            Family::ZipfY => "zipf_y",
            Family::StarHub => "star_hub",
            Family::Cluster => "bipartite_cluster",
            Family::AntiCorr => "anticorr_zipf",
            Family::TwinHubY => "twin_hub_y",
        }
    }
    fn split(self) -> &'static str {
        match self {
            Family::Cluster | Family::AntiCorr => "heldout",
            _ => "train",
        }
    }
}
const FAMILIES: [Family; 6] = [
    Family::Uniform,
    Family::ZipfAll,
    Family::ZipfY,
    Family::StarHub,
    Family::Cluster,
    Family::AntiCorr,
];

struct Ctx {
    v: usize,
    zipf: Zipf,
    zipf_k: Zipf,
}
impl Ctx {
    fn new(v: usize) -> Self {
        Ctx { v, zipf: Zipf::new(v, 1.1), zipf_k: Zipf::new(16, 1.0) }
    }
}

/// role: 0 = R_xy (x,y), 1 = R_yz (y,z), 2 = R_xz (x,z). var: 0 x, 1 y, 2 z.
fn sample_var(f: Family, var: usize, role: usize, rng: &mut Rng, c: &Ctx) -> u32 {
    let v = c.v;
    match f {
        Family::Uniform => rng.below(v),
        Family::ZipfAll => c.zipf.sample(rng),
        Family::ZipfY => {
            if var == 1 {
                c.zipf.sample(rng)
            } else {
                rng.below(v)
            }
        }
        Family::StarHub => {
            if var == 2 && rng.f64() < 0.5 {
                rng.below(4)
            } else {
                rng.below(v)
            }
        }
        Family::TwinHubY => {
            if var == 1 && rng.f64() < 0.5 {
                rng.below(4)
            } else {
                rng.below(v)
            }
        }
        Family::AntiCorr => {
            if var == 0 {
                let r = c.zipf.sample(rng);
                if role == 0 { r } else { (v as u32 - 1) - r }
            } else {
                rng.below(v)
            }
        }
        Family::Cluster => unreachable!(),
    }
}

fn sample_pair(f: Family, role: usize, rng: &mut Rng, c: &Ctx) -> (u32, u32) {
    if f == Family::Cluster {
        let cs = c.v / 16;
        if rng.f64() < 0.9 {
            let k = c.zipf_k.sample(rng) as usize;
            (
                (k * cs) as u32 + rng.below(cs),
                (k * cs) as u32 + rng.below(cs),
            )
        } else {
            (rng.below(c.v), rng.below(c.v))
        }
    } else {
        let (a, b) = [(0, 1), (1, 2), (0, 2)][role];
        (sample_var(f, a, role, rng, c), sample_var(f, b, role, rng, c))
    }
}

/// Exactly `n` distinct pairs (so cardinalities are exact and equal-size instances
/// really are equal-size); falls back to fewer if the family cannot supply them.
fn gen_rel(f: Family, role: usize, n: usize, rng: &mut Rng, c: &Ctx) -> Vec<(u32, u32)> {
    let mut seen = HashSet::with_capacity(n * 2);
    let mut out = Vec::with_capacity(n);
    let mut tries = 0usize;
    while out.len() < n && tries < n * 400 {
        tries += 1;
        let p = sample_pair(f, role, rng, c);
        if seen.insert(((p.0 as u64) << 32) | p.1 as u64) {
            out.push(p);
        }
    }
    out
}

// ------------------------------------------------------------------ degrees + exact fold
struct Deg {
    d: [[Vec<u32>; 2]; 3], // [relation][column]
}
impl Deg {
    fn zero(v: usize) -> Self {
        Deg { d: std::array::from_fn(|_| std::array::from_fn(|_| vec![0u32; v])) }
    }
    fn build(rels: &[Vec<(u32, u32)>; 3], v: usize) -> Self {
        let mut g = Deg::zero(v);
        for (i, r) in rels.iter().enumerate() {
            for &(a, b) in r {
                g.d[i][0][a as usize] += 1;
                g.d[i][1][b as usize] += 1;
            }
        }
        g
    }
}
fn fold(a: &[u32], b: &[u32]) -> u64 {
    a.iter().zip(b).map(|(&x, &y)| x as u64 * y as u64).sum()
}
/// exact inner-join size by PHYSICAL shared variable [x, y, z].
/// physical layout: P0=(x,y), P1=(y,z), P2=(x,z).
fn exact_phys(g: &Deg) -> [u64; 3] {
    [
        fold(&g.d[0][0], &g.d[2][0]),
        fold(&g.d[0][1], &g.d[1][0]),
        fold(&g.d[1][1], &g.d[2][1]),
    ]
}
fn distinct(d: &[u32]) -> u64 {
    d.iter().filter(|&&x| x > 0).count() as u64
}

/// Test oracle only: materialize the join rows, compare with the fold.
fn oracle_check() -> usize {
    let c = Ctx::new(48);
    let mut checked = 0;
    for f in FAMILIES {
        for seed in 0..5u64 {
            let mut rng = Rng(0xABCD ^ seed.wrapping_mul(7919));
            let rels: [Vec<(u32, u32)>; 3] =
                std::array::from_fn(|r| gen_rel(f, r, 150, &mut rng, &c));
            let ex = exact_phys(&Deg::build(&rels, 48));
            // x: P0.x == P2.x ; y: P0.y == P1.y ; z: P1.z == P2.z
            let mut jx = Vec::new();
            let mut jy = Vec::new();
            let mut jz = Vec::new();
            for &(a0, b0) in &rels[0] {
                for &(a2, b2) in &rels[2] {
                    if a0 == a2 {
                        jx.push((a0, b0, b2));
                    }
                }
                for &(a1, b1) in &rels[1] {
                    if b0 == a1 {
                        jy.push((a0, b0, b1));
                    }
                }
            }
            for &(a1, b1) in &rels[1] {
                for &(a2, b2) in &rels[2] {
                    if b1 == b2 {
                        jz.push((a1, b1, a2));
                    }
                }
            }
            assert_eq!(ex[0], jx.len() as u64, "fold != brute force (x) {}", f.name());
            assert_eq!(ex[1], jy.len() as u64, "fold != brute force (y) {}", f.name());
            assert_eq!(ex[2], jz.len() as u64, "fold != brute force (z) {}", f.name());
            checked += 3;
        }
    }
    checked
}

// ------------------------------------------------------------------ arms
/// Exactly the optimizer's argmin: start SharedY, replace on strictly-less X, then Z.
/// Index: 0 = SharedX, 1 = SharedY, 2 = SharedZ.
fn argmin_opt(e: [u64; 3]) -> usize {
    let mut best = (1usize, e[1]);
    if e[0] < best.1 {
        best = (0, e[0]);
    }
    if e[2] < best.1 {
        best = (2, e[2]);
    }
    best.0
}
const LBL: [&str; 3] = ["X", "Y", "Z"];

/// slot (xy=0,yz=1,xz=2) of physical relation with physical vars (u,w) under labels pi.
fn slot_of(pi: [usize; 3], u: usize, w: usize) -> (usize, bool) {
    let (a, b) = (pi[u], pi[w]);
    let (lo, hi) = (a.min(b), a.max(b));
    let s = match (lo, hi) {
        (0, 1) => 0,
        (1, 2) => 1,
        _ => 2,
    };
    (s, a > b)
}
const PHYS_VARS: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

struct Est {
    est: [u64; 3], // label-indexed [X,Y,Z]
}
/// Build the StatsManager the optimizer would see and ask it the three questions
/// exactly as `try_reorder_triangle_with_owner` does.
fn stats_estimates(
    cards: [u64; 3],
    dists: &[[u64; 2]; 3],
    pi: [usize; 3],
    ids: [u32; 3],
    colstats: bool,
    reg_order: [usize; 3],
) -> Est {
    let mut sm = StatsManager::new();
    let mut slot_rel = [RelId(0); 3];
    for &p in &reg_order {
        let (s, swapped) = slot_of(pi, PHYS_VARS[p].0, PHYS_VARS[p].1);
        let id = RelId(ids[p]);
        slot_rel[s] = id;
        sm.register_relation(id);
        sm.update_cardinality(id, cards[p]);
        if colstats {
            for j in 0..2 {
                let pc = if swapped { 1 - j } else { j };
                let mut cs = ColumnStats::new(j, ScalarType::U32);
                cs.distinct_estimate = dists[p][pc];
                sm.add_column_stats(id, cs);
            }
        }
    }
    let (xy, yz, xz) = (slot_rel[0], slot_rel[1], slot_rel[2]);
    let est_y = sm.estimate_join_cardinality(xy, yz, &[1], &[0]);
    let est_x = sm.estimate_join_cardinality(xy, xz, &[0], &[0]);
    let est_z = sm.estimate_join_cardinality(yz, xz, &[1], &[1]);
    Est { est: [est_x, est_y, est_z] }
}

const PERMS: [[usize; 3]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

fn inv(pi: [usize; 3]) -> [usize; 3] {
    let mut r = [0; 3];
    for (v, &l) in pi.iter().enumerate() {
        r[l] = v;
    }
    r
}

struct Trial {
    canon_a0: usize, // canonical = physical shared variable
    canon_a1: usize,
    canon_c: usize,
    cost_a0: u64,
    cost_a1: u64,
    cost_c: u64,
}

fn run_trial(
    rels: &[Vec<(u32, u32)>; 3],
    v: usize,
    pi: [usize; 3],
    rng: &mut Rng,
    shuffle: bool,
) -> Trial {
    // (ii) fact load order: statistics are re-derived from the (shuffled) fact lists.
    let mut r = rels.clone();
    if shuffle {
        r.iter_mut().for_each(|x| rng.shuffle(x));
    }
    let g = Deg::build(&r, v);
    let cards = [r[0].len() as u64, r[1].len() as u64, r[2].len() as u64];
    let dists: [[u64; 2]; 3] =
        std::array::from_fn(|i| [distinct(&g.d[i][0]), distinct(&g.d[i][1])]);
    // (iii) RelId assignment + registration order.
    let mut ids = [0u32; 3];
    let mut pool = [7u32, 123, 4096];
    if shuffle {
        rng.shuffle(&mut pool);
    } else {
        pool = [0, 1, 2];
    }
    ids.copy_from_slice(&pool);
    let mut reg = [0usize, 1, 2];
    if shuffle {
        rng.shuffle(&mut reg);
    }
    let e0 = stats_estimates(cards, &dists, pi, ids, false, reg);
    let e1 = stats_estimates(cards, &dists, pi, ids, true, reg);
    let ex_p = exact_phys(&g);
    let ip = inv(pi);
    let ex_lbl = [ex_p[ip[0]], ex_p[ip[1]], ex_p[ip[2]]];
    let (a0, a1, c) = (argmin_opt(e0.est), argmin_opt(e1.est), argmin_opt(ex_lbl));
    Trial {
        canon_a0: ip[a0],
        canon_a1: ip[a1],
        canon_c: ip[c],
        cost_a0: ex_lbl[a0],
        cost_a1: ex_lbl[a1],
        cost_c: ex_lbl[c],
    }
}

// ------------------------------------------------------------------ stats helpers
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}
fn ratio(chosen: u64, best: u64) -> f64 {
    (chosen.max(1)) as f64 / (best.max(1)) as f64
}

#[derive(Default)]
struct Agg {
    n: usize,
    reg: [Vec<f64>; 5], // a0, a1, c, rand, always-Y
    abs: [Vec<f64>; 5],
    hit: [usize; 5],
    tie_min_exact: usize,
    collapsed: [usize; 2],      // a0, a1: min estimate attained by >1 candidate
    all_equal: [usize; 2],      // all three estimates equal
    fell_to_y: [usize; 2],      // collapsed and chosen == Y
    perm_changed: [usize; 3],   // a0 a1 c: canonical pairing changed in any trial
    perm_costly: [usize; 3],    // ...and the exact cost of the choice changed
}

fn main() {
    assert!(!cfg!(debug_assertions), "release build only");
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.to_string()))
        .unwrap_or_default();
    println!("# triangle pairing baselines (no model trained or integrated)");
    println!(
        "# cpu: {} | target_arch={} | avx2={} avx512f={} | built with target-cpu={}",
        cpu.trim(),
        std::env::consts::ARCH,
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("avx512f"),
        option_env!("RUSTFLAGS").unwrap_or("(default)")
    );
    println!("# xlog rev bff6da66, xlog-stats::StatsManager::estimate_join_cardinality, optimizer.rs:3395-3404 reproduced");

    let oc = oracle_check();
    println!("# oracle: {oc} fold-vs-materialized-join equalities hold (V=48,n=150, 6 families x 5 seeds x 3 pairings)");

    const V: usize = 4096;
    let c = Ctx::new(V);
    println!("# V={V} values per variable; n per relation 6000..24000; seeds 0..11 per family; seed%3==0 => equal sizes");
    println!("family\tsplit\tseed\tn_facts\tn_xy\tn_yz\tn_xz\tN_est_x\tN_est_y\tN_est_z\tC_est_x\tC_est_y\tC_est_z\texact_x\texact_y\texact_z\tchoice_a0\tchoice_a1\tchoice_c\tregret_a0\tregret_a1\tregret_c\tcollapsed_a0\tcollapsed_a1\tperm_changed_a0\tperm_changed_a1\tperm_changed_c");

    let mut aggs: Vec<(Family, Agg)> = FAMILIES.iter().map(|&f| (f, Agg::default())).collect();
    for (fi, &f) in FAMILIES.iter().enumerate() {
        for seed in 0..12u64 {
            let mut rng = Rng(0x5EED_0000 + (fi as u64) * 1000 + seed);
            let base = [6000usize, 10000, 16000, 24000][(seed % 4) as usize];
            let ns: [usize; 3] = if seed % 3 == 0 {
                [base; 3]
            } else {
                let fac = [[0.5, 1.0, 2.0], [2.0, 0.5, 1.0], [1.0, 2.0, 0.5], [0.5, 2.0, 1.0]]
                    [(seed % 4) as usize];
                [
                    (base as f64 * fac[0]) as usize,
                    (base as f64 * fac[1]) as usize,
                    (base as f64 * fac[2]) as usize,
                ]
            };
            let rels: [Vec<(u32, u32)>; 3] =
                std::array::from_fn(|r| gen_rel(f, r, ns[r].min(30000), &mut rng, &c));
            let cards = [rels[0].len(), rels[1].len(), rels[2].len()];
            let g = Deg::build(&rels, V);
            let dists: [[u64; 2]; 3] =
                std::array::from_fn(|i| [distinct(&g.d[i][0]), distinct(&g.d[i][1])]);
            let exact = exact_phys(&g); // identity labels: [X,Y,Z]
            let id = [0usize, 1, 2];
            let e0 = stats_estimates(
                [cards[0] as u64, cards[1] as u64, cards[2] as u64],
                &dists, id, [0, 1, 2], false, [0, 1, 2],
            );
            let e1 = stats_estimates(
                [cards[0] as u64, cards[1] as u64, cards[2] as u64],
                &dists, id, [0, 1, 2], true, [0, 1, 2],
            );
            let (a0, a1, cc) = (argmin_opt(e0.est), argmin_opt(e1.est), argmin_opt(exact));
            let best = *exact.iter().min().unwrap();
            let rand_pick = (Rng(0xFACE ^ seed ^ (fi as u64) << 8).next() % 3) as usize;
            let picks = [a0, a1, cc, rand_pick, 1usize];
            // permutation falsifier: 6 variable relabelings, each with reshuffled
            // fact order, random RelIds and random registration order.
            let mut changed = [false; 3];
            let mut costly = [false; 3];
            let base_t = run_trial(&rels, V, [0, 1, 2], &mut rng, false);
            let base_canon = [base_t.canon_a0, base_t.canon_a1, base_t.canon_c];
            let base_cost = [base_t.cost_a0, base_t.cost_a1, base_t.cost_c];
            for pi in PERMS {
                let t = run_trial(&rels, V, pi, &mut rng, true);
                let cn = [t.canon_a0, t.canon_a1, t.canon_c];
                let co = [t.cost_a0, t.cost_a1, t.cost_c];
                for k in 0..3 {
                    if cn[k] != base_canon[k] {
                        changed[k] = true;
                    }
                    if co[k] != base_cost[k] {
                        costly[k] = true;
                    }
                }
            }
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}",
                f.name(), f.split(), seed, cards.iter().sum::<usize>(), cards[0], cards[1], cards[2],
                e0.est[0], e0.est[1], e0.est[2], e1.est[0], e1.est[1], e1.est[2],
                exact[0], exact[1], exact[2], LBL[a0], LBL[a1], LBL[cc],
                ratio(exact[a0], best), ratio(exact[a1], best), ratio(exact[cc], best),
                tie_min(e0.est), tie_min(e1.est), changed[0], changed[1], changed[2]
            );
            let a = &mut aggs[fi].1;
            a.n += 1;
            for (k, &p) in picks.iter().enumerate() {
                a.reg[k].push(ratio(exact[p], best));
                a.abs[k].push((exact[p] - best) as f64);
                if exact[p] == best {
                    a.hit[k] += 1;
                }
            }
            if tie_min(exact) {
                a.tie_min_exact += 1;
            }
            for (k, e) in [e0.est, e1.est].iter().enumerate() {
                if tie_min(*e) {
                    a.collapsed[k] += 1;
                    if argmin_opt(*e) == 1 {
                        a.fell_to_y[k] += 1;
                    }
                }
                if e[0] == e[1] && e[1] == e[2] {
                    a.all_equal[k] += 1;
                }
            }
            for k in 0..3 {
                a.perm_changed[k] += changed[k] as usize;
                a.perm_costly[k] += costly[k] as usize;
            }
        }
    }

    println!("\n## per-family summary (regret = exact inner size of chosen / best; 1.000 = optimal)");
    println!("arms: a0 = optimizer argmin, cardinalities only; a1 = optimizer argmin, exact per-column distinct counts supplied; c = exact-degree fold argmin; r = seeded random; Y = always SharedY");
    println!("family\tsplit\tn\tmean_regret[a0,a1,c,r,Y]\tmedian_a0\tp90_a0\tmax_a0\tmax_a1\tmean_abs_regret_a0\toptimal_hits[a0,a1,c,r,Y]\texact_ties\tcollapsed[a0,a1]\tall_equal[a0,a1]\tfell_to_Y[a0,a1]\tperm_changed[a0,a1,c]\tperm_costly[a0,a1,c]");
    let mut tot = Agg::default();
    for (f, a) in &aggs {
        println!(
            "{}\t{}\t{}\t[{:.3},{:.3},{:.3},{:.3},{:.3}]\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.0}\t{:?}\t{}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}",
            f.name(), f.split(), a.n,
            mean(&a.reg[0]), mean(&a.reg[1]), mean(&a.reg[2]), mean(&a.reg[3]), mean(&a.reg[4]),
            pct(&a.reg[0], 0.5), pct(&a.reg[0], 0.9), pct(&a.reg[0], 1.0), pct(&a.reg[1], 1.0),
            mean(&a.abs[0]), a.hit, a.tie_min_exact, a.collapsed, a.all_equal, a.fell_to_y,
            a.perm_changed, a.perm_costly
        );
        tot.n += a.n;
        for k in 0..5 {
            tot.reg[k].extend(&a.reg[k]);
            tot.hit[k] += a.hit[k];
        }
        for k in 0..2 {
            tot.collapsed[k] += a.collapsed[k];
            tot.all_equal[k] += a.all_equal[k];
            tot.fell_to_y[k] += a.fell_to_y[k];
        }
        for k in 0..3 {
            tot.perm_changed[k] += a.perm_changed[k];
            tot.perm_costly[k] += a.perm_costly[k];
        }
    }
    for split in ["train", "heldout"] {
        let mut regs: [Vec<f64>; 5] = Default::default();
        for (f, a) in &aggs {
            if f.split() == split {
                for k in 0..5 {
                    regs[k].extend(&a.reg[k]);
                }
            }
        }
        println!(
            "split {split}: mean regret [a0,a1,c,r,Y] = [{:.3},{:.3},{:.3},{:.3},{:.3}]  max a0 = {:.3}",
            mean(&regs[0]), mean(&regs[1]), mean(&regs[2]), mean(&regs[3]), mean(&regs[4]),
            pct(&regs[0], 1.0)
        );
    }
    println!(
        "ALL n={} mean regret [a0,a1,c,r,Y]=[{:.3},{:.3},{:.3},{:.3},{:.3}] optimal hits {:?} collapsed(min-tie) a0/a1 = {:?} all_equal = {:?} fell_to_Y = {:?} perm_changed[a0,a1,c] = {:?} perm_costly = {:?}",
        tot.n, mean(&tot.reg[0]), mean(&tot.reg[1]), mean(&tot.reg[2]), mean(&tot.reg[3]), mean(&tot.reg[4]),
        tot.hit, tot.collapsed, tot.all_equal, tot.fell_to_y, tot.perm_changed, tot.perm_costly
    );

    // ---------------------------------------------------------------- twin for F4
    println!("\n## F4 TWIN: same cardinalities, changed degree distribution on R_yz");
    {
        let mut rng = Rng(0x7117);
        let n = 12000;
        let r0 = gen_rel(Family::Uniform, 0, n, &mut rng, &c);
        let r1u = gen_rel(Family::Uniform, 1, n, &mut rng, &c);
        let r1h = gen_rel(Family::StarHub, 1, n, &mut rng, &c);
        let r2 = gen_rel(Family::Uniform, 2, n, &mut rng, &c);
        for (name, r1) in [("uniform R_yz", r1u), ("hub-on-z R_yz", r1h)] {
            let rels = [r0.clone(), r1, r2.clone()];
            let g = Deg::build(&rels, V);
            let ex = exact_phys(&g);
            let cards = [rels[0].len() as u64, rels[1].len() as u64, rels[2].len() as u64];
            let dists: [[u64; 2]; 3] =
                std::array::from_fn(|i| [distinct(&g.d[i][0]), distinct(&g.d[i][1])]);
            let e0 = stats_estimates(cards, &dists, [0, 1, 2], [0, 1, 2], false, [0, 1, 2]);
            println!(
                "{name}: cards={cards:?} exact(x,y,z)={ex:?} choice_c={} est_nocol={:?} choice_a0={}",
                LBL[argmin_opt(ex)], e0.est, LBL[argmin_opt(e0.est)]
            );
        }
    }

    // ---------------------------------------------------------------- costs
    println!("\n## COST (release; one-time costs in us, per-decision in ns/us; any ms-scale per-decision result is a FAILED GATE)");
    for (name, f) in [("uniform", Family::Uniform), ("zipf_all", Family::ZipfAll)] {
        let mut rng = Rng(0xC057);
        let rels: [Vec<(u32, u32)>; 3] =
            std::array::from_fn(|r| gen_rel(f, r, 10000, &mut rng, &c));
        let nf: usize = rels.iter().map(|r| r.len()).sum();
        // exact statistic, from scratch: count + fold
        let iters = 300;
        let t = Instant::now();
        for _ in 0..iters {
            black_box(Deg::build(black_box(&rels), V));
        }
        let build_ns = t.elapsed().as_nanos() as f64 / iters as f64;
        let g = Deg::build(&rels, V);
        let fi = 20000;
        let t = Instant::now();
        for _ in 0..fi {
            black_box(exact_phys(black_box(&g)));
        }
        let fold_ns = t.elapsed().as_nanos() as f64 / fi as f64;
        // argmin over StatsManager estimates (existing optimizer decision cost)
        let cards = [rels[0].len() as u64, rels[1].len() as u64, rels[2].len() as u64];
        let mut sm = StatsManager::new();
        for i in 0..3u32 {
            sm.register_relation(RelId(i));
            sm.update_cardinality(RelId(i), cards[i as usize]);
        }
        let ai = 2_000_000;
        let t = Instant::now();
        let mut acc = 0usize;
        for i in 0..ai {
            let (a, b, cz) = (RelId(black_box(0)), RelId(black_box(1)), RelId(black_box(2)));
            let _ = i;
            let est = [
                sm.estimate_join_cardinality(a, cz, &[0], &[0]),
                sm.estimate_join_cardinality(a, b, &[1], &[0]),
                sm.estimate_join_cardinality(b, cz, &[1], &[1]),
            ];
            acc += argmin_opt(black_box(est));
        }
        black_box(acc);
        let argmin_ns = t.elapsed().as_nanos() as f64 / ai as f64;
        // incremental maintenance of the three exact sizes under inserts
        let it = 100;
        let t = Instant::now();
        let mut final_ex = [0u64; 3];
        for _ in 0..it {
            let mut gd = Deg::zero(V);
            let mut ex = [0u64; 3];
            for &(a, b) in &rels[0] {
                ex[0] += gd.d[2][0][a as usize] as u64;
                ex[1] += gd.d[1][0][b as usize] as u64;
                gd.d[0][0][a as usize] += 1;
                gd.d[0][1][b as usize] += 1;
            }
            for &(a, b) in &rels[1] {
                ex[1] += gd.d[0][1][a as usize] as u64;
                ex[2] += gd.d[2][1][b as usize] as u64;
                gd.d[1][0][a as usize] += 1;
                gd.d[1][1][b as usize] += 1;
            }
            for &(a, b) in &rels[2] {
                ex[0] += gd.d[0][0][a as usize] as u64;
                ex[2] += gd.d[1][1][b as usize] as u64;
                gd.d[2][0][a as usize] += 1;
                gd.d[2][1][b as usize] += 1;
            }
            final_ex = black_box(ex);
        }
        let incr_ns = t.elapsed().as_nanos() as f64 / it as f64;
        assert_eq!(final_ex, exact_phys(&g), "incremental maintenance != fold");
        println!(
            "{name}: facts={nf} | from-scratch degree build {:.1} us ({:.2} ns/fact) | fold (3 pairings) {:.0} ns | build+fold per decision {:.1} us | optimizer argmin (3 estimate calls) {:.0} ns | incremental exact maintenance {:.2} ns/insert ({:.1} us for all {} inserts, final == fold: ok) | degree memory {} KiB",
            build_ns / 1000.0, build_ns / nf as f64, fold_ns, (build_ns + fold_ns) / 1000.0,
            argmin_ns, incr_ns / nf as f64, incr_ns / 1000.0, nf,
            6 * V * 4 / 1024
        );
    }
    codec9_delta_arm(&c);
    add_pm_arm(&c);
}

fn tie_min(e: [u64; 3]) -> bool {
    let m = *e.iter().min().unwrap();
    e.iter().filter(|&&x| x == m).count() > 1
}

// ------------------------------------------------------------------ Codec9 delta arm
// Lab arm (D-XLOG-C9-0): per-key degree DELTAS carried as Codec9 lanes.
// One lane per (relation, column, value): the signed net change of that key's
// degree in one batch, -255..=255; a key the batch never touched is the -0
// codeword (never touched), distinct from a touched +0 (net zero). Merge on
// read adds each touched lane to the base degree. Two write policies:
//   escape     |delta| > 255 is not encoded: the exact delta is applied by a
//              side path and counted (Codec9 itself saturates silently);
//   saturating the codec's own clamp, as shipped.
// Checked against exact degrees after every batch, and against the exact-fold
// pairing choice (arm c) at the end of the stream. Correctness only; the
// timings this example prints elsewhere are not this arm's.

/// The relations of one (family, seed) trial, generated exactly as `main` does.
fn trial_rels(f: Family, fi: usize, seed: u64, c: &Ctx) -> [Vec<(u32, u32)>; 3] {
    let mut rng = Rng(0x5EED_0000 + (fi as u64) * 1000 + seed);
    let base = [6000usize, 10000, 16000, 24000][(seed % 4) as usize];
    let ns: [usize; 3] = if seed % 3 == 0 {
        [base; 3]
    } else {
        let fac = [[0.5, 1.0, 2.0], [2.0, 0.5, 1.0], [1.0, 2.0, 0.5], [0.5, 2.0, 1.0]]
            [(seed % 4) as usize];
        [
            (base as f64 * fac[0]) as usize,
            (base as f64 * fac[1]) as usize,
            (base as f64 * fac[2]) as usize,
        ]
    };
    std::array::from_fn(|r| gen_rel(f, r, ns[r].min(30000), &mut rng, c))
}

/// One event: relation, fact, +1 insert / -1 retract.
type Event = (usize, (u32, u32), i32);

/// Inserts (shuffled), then retract-and-reinsert of a quarter of the facts
/// (shuffled pairs, so a batch can net a key to zero), then retraction of
/// another quarter. Returns the events and the final multiset.
fn trial_stream(rels: &[Vec<(u32, u32)>; 3], seed: u64) -> (Vec<Event>, [Vec<(u32, u32)>; 3]) {
    let mut rng = Rng(0xC0DE_C900 ^ seed);
    let mut ins: Vec<Event> = Vec::new();
    for (r, rel) in rels.iter().enumerate() {
        ins.extend(rel.iter().map(|&t| (r, t, 1)));
    }
    rng.shuffle(&mut ins);
    let mut churn: Vec<Event> = Vec::new();
    let mut gone: Vec<Event> = Vec::new();
    for (i, &(r, t, _)) in ins.iter().enumerate() {
        match i % 4 {
            0 => {
                churn.push((r, t, -1));
                churn.push((r, t, 1));
            }
            1 => gone.push((r, t, -1)),
            _ => {}
        }
    }
    rng.shuffle(&mut churn);
    // a retract must not precede its own insert within the churn phase: the
    // multiset stays non-negative per fact only if each pair keeps order.
    churn.sort_by_key(|e| e.2 == 1); // all retracts, then all reinserts
    rng.shuffle(&mut gone);
    let mut finals: [Vec<(u32, u32)>; 3] = Default::default();
    for (i, &(r, t, _)) in ins.iter().enumerate() {
        if i % 4 != 1 {
            finals[r].push(t);
        }
    }
    let mut ev = ins;
    ev.extend(churn);
    ev.extend(gone);
    (ev, finals)
}

#[derive(Default, Clone, Copy)]
struct C9Stats {
    batches: usize,
    lanes_touched: usize,
    touched_zero: usize,
    overflow: usize,
    max_abs: i64,
    escape_mismatch: usize,
    sat_mismatch: usize,
    sat_negative: usize,
}

fn deg_u32(d: &[[Vec<i64>; 2]; 3]) -> Option<Deg> {
    let mut g = Deg::zero(d[0][0].len());
    for r in 0..3 {
        for col in 0..2 {
            for (k, &v) in d[r][col].iter().enumerate() {
                g.d[r][col][k] = u32::try_from(v).ok()?;
            }
        }
    }
    Some(g)
}

fn run_c9(events: &[Event], v: usize, batch: usize) -> (C9Stats, Deg, Option<Deg>) {
    let codec = Codec9Runtime::<TiesEven>::new(1.0).expect("scale 1 is valid");
    let zero = || -> [[Vec<i64>; 2]; 3] { std::array::from_fn(|_| std::array::from_fn(|_| vec![0i64; v])) };
    let (mut exact, mut esc, mut sat) = (zero(), zero(), zero());
    let mut delta = zero();
    let mut lanes: [[Vec<EpiLane9>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![EpiLane9::NEVER_TOUCHED; v]));
    let mut touched: Vec<(usize, usize, usize)> = Vec::new();
    let mut st = C9Stats::default();
    for chunk in events.chunks(batch) {
        st.batches += 1;
        for &(r, (a, b), w) in chunk {
            for (col, key) in [(0usize, a as usize), (1, b as usize)] {
                if delta[r][col][key] == 0 && lanes[r][col][key].never_touched() {
                    touched.push((r, col, key));
                }
                delta[r][col][key] += i64::from(w);
                exact[r][col][key] += i64::from(w);
                // mark touched even if the running delta returns to 0
                lanes[r][col][key] = EpiLane9::from_codeword(0x100).expect("+0");
            }
        }
        // encode: every touched key becomes a touched lane (+0 if net zero)
        for &(r, col, key) in &touched {
            let d = delta[r][col][key];
            st.lanes_touched += 1;
            st.max_abs = st.max_abs.max(d.abs());
            if d == 0 {
                st.touched_zero += 1;
            }
            let lane = codec.encode_touched(d as f32).expect("finite");
            lanes[r][col][key] = lane;
            // saturating arm: the codec's own clamp, applied as is
            sat[r][col][key] += i64::from(lane.value().expect("touched"));
            // escape arm: refuse to encode past the codec's range
            if d.abs() > 255 {
                st.overflow += 1;
                esc[r][col][key] += d;
            }
        }
        // merge on read for the escape arm, over every lane: never-touched
        // lanes are skipped by the codeword itself, not by a second mask
        for r in 0..3 {
            for col in 0..2 {
                for (key, lane) in lanes[r][col].iter().enumerate() {
                    if let Some(x) = lane.value() {
                        if delta[r][col][key].abs() <= 255 {
                            esc[r][col][key] += i64::from(x);
                        }
                    }
                }
            }
        }
        for &(r, col, key) in &touched {
            delta[r][col][key] = 0;
            lanes[r][col][key] = EpiLane9::NEVER_TOUCHED;
        }
        touched.clear();
        if esc != exact {
            st.escape_mismatch += 1;
        }
        if sat != exact {
            st.sat_mismatch += 1;
        }
    }
    if sat.iter().flatten().flatten().any(|&x| x < 0) {
        st.sat_negative = 1;
    }
    let exact_deg = deg_u32(&exact).expect("exact degrees are non-negative");
    assert!(deg_u32(&esc).is_some_and(|g| g.d == exact_deg.d), "escape arm != exact at the end");
    (st, exact_deg, deg_u32(&sat))
}

fn codec9_delta_arm(c: &Ctx) {
    println!("\n## Codec9 delta arm (D-XLOG-C9-0): degree deltas as Codec9 lanes, scale 1");
    println!("batch\tfamily\ttrials\tbatches\tlanes_touched\ttouched_zero\toverflow_lanes\tmax_abs_delta\tescape_batches_wrong\tsat_batches_wrong\tsat_trials_negative\tregret_escape_max\tregret_sat_max\tsat_choice_changed");
    let v = c.v;
    for batch in [256usize, 4096, 65536] {
        let mut all = C9Stats::default();
        for (fi, &f) in FAMILIES.iter().enumerate() {
            let mut t = C9Stats::default();
            let (mut reg_esc, mut reg_sat, mut changed, mut trials) = (1.0f64, 1.0f64, 0usize, 0usize);
            for seed in 0..12u64 {
                let rels = trial_rels(f, fi, seed, c);
                let (events, finals) = trial_stream(&rels, seed);
                let (st, exact_deg, sat_deg) = run_c9(&events, v, batch);
                // the stream's end state is the final multiset, built independently
                assert_eq!(exact_deg.d, Deg::build(&finals, v).d, "stream bookkeeping");
                let exact = exact_phys(&exact_deg);
                let best = *exact.iter().min().unwrap();
                let pick_exact = argmin_opt(exact);
                reg_esc = reg_esc.max(ratio(exact[pick_exact], best));
                match &sat_deg {
                    Some(g) => {
                        let p = argmin_opt(exact_phys(g));
                        reg_sat = reg_sat.max(ratio(exact[p], best));
                        changed += usize::from(p != pick_exact);
                    }
                    None => changed += 1,
                }
                trials += 1;
                t.batches += st.batches;
                t.lanes_touched += st.lanes_touched;
                t.touched_zero += st.touched_zero;
                t.overflow += st.overflow;
                t.max_abs = t.max_abs.max(st.max_abs);
                t.escape_mismatch += st.escape_mismatch;
                t.sat_mismatch += st.sat_mismatch;
                t.sat_negative += st.sat_negative;
            }
            println!(
                "{batch}\t{}\t{trials}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{changed}",
                f.name(), t.batches, t.lanes_touched, t.touched_zero, t.overflow, t.max_abs,
                t.escape_mismatch, t.sat_mismatch, t.sat_negative, reg_esc, reg_sat
            );
            all.batches += t.batches;
            all.overflow += t.overflow;
            all.escape_mismatch += t.escape_mismatch;
            all.sat_mismatch += t.sat_mismatch;
        }
        println!(
            "# batch {batch}: {} batches, {} overflow lanes, escape wrong in {} batches, saturating wrong in {} batches",
            all.batches, all.overflow, all.escape_mismatch, all.sat_mismatch
        );
    }
}

// ------------------------------------------------------------------ add_pm arm
// Lab arm (D-XLOG-PM-1): the same streams, with every degree held as a
// Codec9 P × M lane and summed by `add_pm` (`PmLedger`). Per batch, each
// event is added into a delta ledger; the batch delta is then absorbed into
// the degree ledger 64 keys per `add_pm`. A sum past ±255 escapes to an
// exact wide value and is demoted when it returns. Three counts are kept
// apart: carries (an addition left ±255, even if a later one brings it
// back), batch overflows (keys whose net batch delta is wide), and final
// escalations (degrees wide at the end of the stream).

/// Inserts (shuffled), then retraction of 90 % of them (shuffled): hub
/// degrees climb past 255 and fall back, so wide degrees must demote.
fn drain_stream(rels: &[Vec<(u32, u32)>; 3], seed: u64) -> (Vec<Event>, [Vec<(u32, u32)>; 3]) {
    let mut rng = Rng(0xD7A1_0000 ^ seed);
    let mut ins: Vec<Event> = Vec::new();
    for (r, rel) in rels.iter().enumerate() {
        ins.extend(rel.iter().map(|&t| (r, t, 1)));
    }
    rng.shuffle(&mut ins);
    let mut gone: Vec<Event> = Vec::new();
    let mut finals: [Vec<(u32, u32)>; 3] = Default::default();
    for (i, &(r, t, _)) in ins.iter().enumerate() {
        if i % 10 == 0 {
            finals[r].push(t);
        } else {
            gone.push((r, t, -1));
        }
    }
    rng.shuffle(&mut gone);
    let mut ev = ins;
    ev.extend(gone);
    (ev, finals)
}

#[derive(Default, Clone, Copy)]
struct PmStats {
    batches: usize,
    delta_carries: u64,
    delta_demotions: u64,
    batch_overflow_keys: usize,
    deg_carries: u64,
    deg_demotions: u64,
    max_wide: usize,
    final_wide: usize,
    batches_wrong: usize,
}

type Ledgers = [[PmLedger; 2]; 3];

fn ledger_deg(l: &Ledgers, v: usize) -> Option<Deg> {
    let mut g = Deg::zero(v);
    for r in 0..3 {
        for col in 0..2 {
            for k in 0..v {
                g.d[r][col][k] = u32::try_from(l[r][col].value(k).unwrap_or(0)).ok()?;
            }
        }
    }
    Some(g)
}

fn run_pm(events: &[Event], v: usize, batch: usize) -> (PmStats, Deg, Option<Deg>) {
    let new = || -> Ledgers { std::array::from_fn(|_| std::array::from_fn(|_| PmLedger::new(v))) };
    let (mut deg, mut delta) = (new(), new());
    let mut exact: [[Vec<i64>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![0i64; v]));
    let mut st = PmStats::default();
    for chunk in events.chunks(batch) {
        st.batches += 1;
        for &(r, (a, b), w) in chunk {
            for (col, key) in [(0usize, a as usize), (1, b as usize)] {
                delta[r][col].add(key, i64::from(w));
                exact[r][col][key] += i64::from(w);
            }
        }
        let mut wide_now = 0;
        let mut wrong = false;
        for r in 0..3 {
            for col in 0..2 {
                st.batch_overflow_keys += delta[r][col].escalated();
                deg[r][col].absorb(&delta[r][col]);
                delta[r][col].clear();
                wide_now += deg[r][col].escalated();
                for k in 0..v {
                    if deg[r][col].value(k).unwrap_or(0) != exact[r][col][k] {
                        wrong = true;
                    }
                }
            }
        }
        st.max_wide = st.max_wide.max(wide_now);
        st.batches_wrong += usize::from(wrong);
    }
    for r in 0..3 {
        for col in 0..2 {
            st.delta_carries += delta[r][col].carries;
            st.delta_demotions += delta[r][col].demotions;
            st.deg_carries += deg[r][col].carries;
            st.deg_demotions += deg[r][col].demotions;
            st.final_wide += deg[r][col].escalated();
        }
    }
    let exact_deg = deg_u32(&exact).expect("exact degrees are non-negative");
    (st, exact_deg, ledger_deg(&deg, v))
}

fn add_pm_arm(c: &Ctx) {
    println!(
        "\n## add_pm arm (D-XLOG-PM-1): degrees as P x M lanes, add_pm with exact wide escape"
    );
    println!(
        "stream\tbatch\tfamily\ttrials\tbatches\tdelta_carries\tdelta_demotions\tbatch_overflow_keys\tdeg_carries\tdeg_demotions\tmax_wide\tfinal_wide\tbatches_wrong\tregret_max\tchoice_changed"
    );
    let v = c.v;
    type Stream = fn(&[Vec<(u32, u32)>; 3], u64) -> (Vec<Event>, [Vec<(u32, u32)>; 3]);
    let streams: [(&str, Stream); 2] = [("churn", trial_stream), ("drain", drain_stream)];
    for (sname, stream) in streams {
        for batch in [256usize, 4096, 65536] {
            let mut all = PmStats::default();
            let mut all_changed = 0;
            for (fi, &f) in FAMILIES.iter().enumerate() {
                let mut t = PmStats::default();
                let (mut regret, mut changed, mut trials) = (1.0f64, 0usize, 0usize);
                for seed in 0..12u64 {
                    let rels = trial_rels(f, fi, seed, c);
                    let (events, finals) = stream(&rels, seed);
                    let (st, exact_deg, pm_deg) = run_pm(&events, v, batch);
                    assert_eq!(exact_deg.d, Deg::build(&finals, v).d, "stream bookkeeping");
                    let pm_deg = pm_deg.expect("ledger degrees are non-negative");
                    assert_eq!(pm_deg.d, exact_deg.d, "add_pm degrees != exact at the end");
                    let exact = exact_phys(&exact_deg);
                    let best = *exact.iter().min().unwrap();
                    let pick_exact = argmin_opt(exact);
                    let pick_pm = argmin_opt(exact_phys(&pm_deg));
                    regret = regret.max(ratio(exact[pick_pm], best));
                    changed += usize::from(pick_pm != pick_exact);
                    trials += 1;
                    t.batches += st.batches;
                    t.delta_carries += st.delta_carries;
                    t.delta_demotions += st.delta_demotions;
                    t.batch_overflow_keys += st.batch_overflow_keys;
                    t.deg_carries += st.deg_carries;
                    t.deg_demotions += st.deg_demotions;
                    t.max_wide = t.max_wide.max(st.max_wide);
                    t.final_wide += st.final_wide;
                    t.batches_wrong += st.batches_wrong;
                }
                println!(
                    "{sname}\t{batch}\t{}\t{trials}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{changed}",
                    f.name(),
                    t.batches,
                    t.delta_carries,
                    t.delta_demotions,
                    t.batch_overflow_keys,
                    t.deg_carries,
                    t.deg_demotions,
                    t.max_wide,
                    t.final_wide,
                    t.batches_wrong,
                    regret
                );
                all.batches += t.batches;
                all.delta_carries += t.delta_carries;
                all.batch_overflow_keys += t.batch_overflow_keys;
                all.deg_carries += t.deg_carries;
                all.deg_demotions += t.deg_demotions;
                all.final_wide += t.final_wide;
                all.batches_wrong += t.batches_wrong;
                all_changed += changed;
            }
            println!(
                "# {sname} batch {batch}: {} batches, delta carries {}, batch overflow keys {}, degree carries {}, degree demotions {}, final wide {}, batches wrong {}, pairing changed {}",
                all.batches,
                all.delta_carries,
                all.batch_overflow_keys,
                all.deg_carries,
                all.deg_demotions,
                all.final_wide,
                all.batches_wrong,
                all_changed
            );
        }
    }
}
