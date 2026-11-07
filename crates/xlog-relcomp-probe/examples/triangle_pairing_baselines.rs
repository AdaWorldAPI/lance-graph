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
}

fn tie_min(e: [u64; 3]) -> bool {
    let m = *e.iter().min().unwrap();
    e.iter().filter(|&&x| x == m).count() > 1
}
