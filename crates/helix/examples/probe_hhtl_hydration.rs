//! PROBE-HHTL-HYDRATION — does a SEMANTIC address, pushed through the shipped
//! `HHTL place -> helix geometry` seam, land semantically-near items spatially
//! near? (Operator: "the only question is HHTL and helix to give semantics a
//! spatial perturbation hydration".)
//!
//! **What reading the code changed about the question.** `ResidueEncoder::encode`
//! computes `start_idx` from `place` ALONE and `end_idx` from `n` ALONE, and
//! neither ever sees a vector:
//!
//! ```text
//! start_idx = quantize(aligned_for_place(place))       // place only
//! end_idx   = quantize(aligned_for_residue(n, total))  // n only
//! ```
//!
//! So the residue cannot "recover direction the place discarded" — it is a
//! deterministic function of the address, exactly as the canon states (*phase is
//! convention, not data; magnitude is the only stored bits*). **Hydration is
//! therefore an ADDRESSING question, not an encoding one:** the geometry is free
//! and fixed; only the address assignment can carry meaning.
//!
//! **The seam under test is shipped and named** —
//! [`CurveRuler::from_hhtl(path, depth)`], documented as taking the `NiblePath`
//! packed form without importing the HHTL type. This probe supplies it a
//! semantic address and asks whether the geometry inherits the semantics.
//!
//! **Arms** (all on shipped helix code, one target, one null):
//!   - `place-only`   — circular distance between the two `start_offset`s.
//!   - `hydrated`     — Euclidean distance between the two `HemispherePoint`s
//!                      whose rank `n` is the item's position in HHTL-address
//!                      order (near addresses -> near curve positions).
//!   - `shuffled-addr`— the SAME pipeline over a shuffled lemma->path map. If
//!                      this scores like `hydrated`, the address never reached
//!                      the geometry and nothing was hydrated.
//!
//! **Target (a), spatial reconstruction — proved before any semantics claim.**
//! Ground truth is `1 - cosine` on the FULL-dimension real vectors. Asking
//! whether the geometry tracks *meaning* (target b) is only worth doing if it
//! first tracks *position*.
//!
//! **Floor.** A K-replicate permutation null over the lemma->address map, not a
//! parametric bound: `jc` pillar 5 (Jirak) establishes these samples are weakly
//! dependent, so a classical IID sigma would UNDERSTATE the floor; the pillar's
//! output is that citation, never a threshold. The permutation carries the
//! dependence structure itself. Effective independent unit = the LEMMA.
//!
//! **Named capacity limit, derived not guessed.** `from_place` reduces the whole
//! HHTL path to `place % 17`, so the place anchor has exactly **17**
//! distinguishable arc starts at any depth; `from_hhtl` folds `depth` in to
//! break same-path ties. A mod-17 integer map floors at `D* = 1/17` (measured:
//! `jc/examples/probe_stride_discrepancy.rs`). The probe asserts the collapse
//! explicitly rather than letting it surprise a later reader.
//!
//! Real bytes only (Rule 23). Deterministic SplitMix64, seed 0x9E3779B97F4A7C15.
//!
//! **Target (b), taxonomy — runs ONLY when the extra args are supplied, and is
//! only meaningful because (a) held.** Ground truth becomes the WordNet
//! shared-ancestor depth for the lemma pair, so the question changes from *does
//! the geometry track POSITION* to *does it track MEANING*. The address, the
//! geometry, and the permutation null are identical; only the target moves,
//! which is what makes the two numbers comparable.
//!
//! The WordNet walk is reproduced here rather than imported: helix is a
//! standalone crate (own `[workspace]`, git-sourced ndarray) and must not gain a
//! dependency on the planner to run a probe.
//!
//! ```text
//! # target (a) only:
//! cargo run --release --manifest-path crates/helix/Cargo.toml \
//!   --example probe_hhtl_hydration -- <emb.f32>
//! # targets (a) AND (b):
//! cargo run --release --manifest-path crates/helix/Cargo.toml \
//!   --example probe_hhtl_hydration -- <emb.f32> <vocab.txt> <wordnet31_isa_v2.tsv>
//! ```
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use helix::curve_ruler::CurveRuler;
use helix::placement::HemispherePoint;
use ndarray::simd::kmeans;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const N_LEMMAS: usize = 3000;
const N_PAIRS: usize = 40_000;
const N_PERM: usize = 200;
const KMEANS_ITERS: usize = 12;
/// `NiblePath` is nibble-based (`FAN_OUT = 16`), so the cascade is 16-ary to
/// match the address format exactly — one k-means level per nibble.
const FAN: usize = 16;
const LEVELS: u8 = 3;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt()).max(1e-300)
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
    let mut r = vec![0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na * nb).sqrt().max(1e-300)
}

fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn nearest(v: &[f32], cents: &[Vec<f32>]) -> usize {
    let mut best = (f32::INFINITY, 0usize);
    for (i, c) in cents.iter().enumerate() {
        let d = sq_l2(v, c);
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

/// Build a `LEVELS`-deep 16-ary cascade over FULL-dimension vectors and return
/// each row's packed `NiblePath` value. The packing mirrors
/// `NiblePath::root(n0).child(n1).child(n2)` exactly — `path = (path << 4) | nibble`
/// — reproduced here rather than imported, because helix deliberately does NOT
/// depend on the contract crate (its own doc: "WITHOUT importing the HHTL type").
fn semantic_paths(rows: &[Vec<f32>], dim: usize) -> Vec<u64> {
    let mut path = vec![0u64; rows.len()];
    // Groups at the current level: indices into `rows`.
    let mut groups: Vec<Vec<usize>> = vec![(0..rows.len()).collect()];
    for _ in 0..LEVELS {
        let mut next: Vec<Vec<usize>> = Vec::new();
        for g in &groups {
            let members: Vec<Vec<f32>> = g.iter().map(|&i| rows[i].clone()).collect();
            let cents = if members.len() >= FAN {
                kmeans(&members, FAN, dim, KMEANS_ITERS)
            } else {
                // Under-populated node: every member takes nibble 0 so the path
                // stays well-formed and ancestry is preserved.
                vec![members.first().cloned().unwrap_or_else(|| vec![0.0; dim]); FAN]
            };
            let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); FAN];
            for &i in g {
                let nib = nearest(&rows[i], &cents);
                path[i] = (path[i] << 4) | nib as u64;
                buckets[nib].push(i);
            }
            next.extend(buckets);
        }
        groups = next;
    }
    path
}

/// Circular distance between two mod-17 arc starts (the place anchor's own metric).
fn circ17(a: u8, b: u8) -> f64 {
    let d = (i32::from(a) - i32::from(b)).abs();
    f64::from(d.min(17 - d))
}

/// Euclidean distance between two hemisphere points.
fn hemi_dist(a: &HemispherePoint, b: &HemispherePoint) -> f64 {
    let (ax, az, ay) = a.cartesian();
    let (bx, bz, by) = b.cartesian();
    ((ax - bx).powi(2) + (az - bz).powi(2) + (ay - by).powi(2)).sqrt()
}

/// Rank each item by its HHTL address, then lift that rank onto the hemisphere.
/// THIS is the hydration step: near addresses -> near ranks -> near curve
/// positions, via the shipped golden-angle placement.
fn hydrate(paths: &[u64], depth: u8) -> (Vec<HemispherePoint>, Vec<u8>) {
    let total = paths.len();
    let mut order: Vec<usize> = (0..total).collect();
    // Tie-break on index so the ordering is deterministic for equal paths.
    order.sort_by(|&a, &b| paths[a].cmp(&paths[b]).then(a.cmp(&b)));
    let mut rank = vec![0usize; total];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    let points = (0..total)
        .map(|i| HemispherePoint::lift(rank[i], total))
        .collect();
    let starts = paths
        .iter()
        .map(|&p| CurveRuler::from_hhtl(p, depth).start_offset())
        .collect();
    (points, starts)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(path_arg) = argv.first().cloned() else {
        eprintln!(
            "usage: probe_hhtl_hydration <emb.f32>\n\n\
             Requires REAL embedding bytes (Rule 23). Format: [u32 n][u32 dim]\n\
             + n*dim f32 LE. Reference input (all-MiniLM-L6-v2 word embeddings,\n\
             30522 x 384, model.safetensors sha256\n\
             53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db).\n\n\
             Optional target (b): <emb.f32> <vocab.txt> <wordnet31_isa_v2.tsv>"
        );
        std::process::exit(2);
    };
    let buf = std::fs::read(&path_arg).unwrap_or_else(|e| panic!("read {path_arg}: {e}"));
    assert!(buf.len() > 8, "{path_arg}: too short for a header");
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(
        buf.len(),
        8 + n * dim * 4,
        "{path_arg}: header/size mismatch"
    );
    let all: Vec<f32> = buf[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(n >= N_LEMMAS, "need >= {N_LEMMAS} rows, file has {n}");
    println!("source: {n} x {dim} real embedding rows; using {N_LEMMAS}");

    // Target (b) needs lemma NAMES, so when vocab + rail are supplied the sample
    // is drawn from rows that are BOTH a whole-word vocab token and a WordNet
    // noun. Target (a) is then measured on that same sample, so the two targets
    // are comparable rather than measured on different populations.
    let taxonomy = (argv.len() >= 3).then(|| {
        let vocab: Vec<String> = std::fs::read_to_string(&argv[1])
            .unwrap_or_else(|e| panic!("read {}: {e}", argv[1]))
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(
            vocab.len(),
            n,
            "vocab has {} entries but the matrix has {n} rows - not the same tokenizer",
            vocab.len()
        );
        (vocab, WordNet::load(&argv[2]))
    });

    let mut rng = SplitMix64(SEED);
    let candidates: Vec<usize> = match &taxonomy {
        Some((vocab, wn)) => (0..n)
            .filter(|&i| {
                let w = &vocab[i];
                w.len() > 2
                    && w.chars().all(|c| c.is_ascii_lowercase())
                    && wn.senses.contains_key(w)
            })
            .collect(),
        None => (0..n).collect(),
    };
    assert!(
        candidates.len() >= N_LEMMAS,
        "only {} usable rows, need >= {N_LEMMAS}",
        candidates.len()
    );
    let mut taken = vec![false; candidates.len()];
    let mut pick = Vec::with_capacity(N_LEMMAS);
    while pick.len() < N_LEMMAS {
        let k = rng.below(candidates.len());
        if !taken[k] {
            taken[k] = true;
            pick.push(candidates[k]);
        }
    }
    if taxonomy.is_some() {
        println!(
            "target (b) enabled: sampling from {} vocab-AND-wordnet-noun rows",
            candidates.len()
        );
    }
    let rows: Vec<Vec<f32>> = pick
        .iter()
        .map(|&i| all[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // ── the semantic address ────────────────────────────────────────────────
    let paths = semantic_paths(&rows, dim);
    let distinct: std::collections::HashSet<u64> = paths.iter().copied().collect();
    println!(
        "semantic addresses: {} distinct {LEVELS}-nibble paths over {N_LEMMAS} lemmas",
        distinct.len()
    );
    assert!(
        distinct.len() > FAN,
        "address assignment is degenerate ({} distinct paths) - the cascade \
         collapsed and nothing semantic is being carried",
        distinct.len()
    );

    // ── DERIVED CAPACITY LIMIT: the place anchor is `place % 17` ────────────
    // Two addresses differing only above the modulus MUST collide. Asserted so
    // the collapse is a stated property, never a later surprise.
    {
        let a = CurveRuler::from_place(5).start_offset();
        let b = CurveRuler::from_place(5 + 17 * 4096).start_offset();
        assert_eq!(
            a, b,
            "from_place is not mod-17 - this probe's model is wrong"
        );
        let anchors: std::collections::HashSet<u8> = paths
            .iter()
            .map(|&p| CurveRuler::from_hhtl(p, LEVELS).start_offset())
            .collect();
        println!(
            "place anchors in use: {}/17 (mod-17 ceiling, independent of depth)",
            anchors.len()
        );
    }

    let (points, starts) = hydrate(&paths, LEVELS);

    // ── pairs + ground truth ────────────────────────────────────────────────
    let mut pairs = Vec::with_capacity(N_PAIRS);
    let mut truth = Vec::with_capacity(N_PAIRS);
    while pairs.len() < N_PAIRS {
        let (i, j) = (rng.below(N_LEMMAS), rng.below(N_LEMMAS));
        if i == j {
            continue;
        }
        pairs.push((i, j));
        truth.push(1.0 - cosine(&rows[i], &rows[j]));
    }

    let d_place: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| circ17(starts[i], starts[j]))
        .collect();
    let d_hydr: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| hemi_dist(&points[i], &points[j]))
        .collect();

    let rho_place = spearman(&d_place, &truth);
    let rho_hydr = spearman(&d_hydr, &truth);

    // ── PERMUTATION NULL over the lemma -> address map ──────────────────────
    // Shuffling the ADDRESS (not the geometry) is the right null: the geometry
    // is deterministic either way, so only the assignment is under test.
    let mut null = Vec::with_capacity(N_PERM);
    let mut shuffled = paths.clone();
    for _ in 0..N_PERM {
        for k in (1..shuffled.len()).rev() {
            shuffled.swap(k, rng.below(k + 1));
        }
        let (p_s, _) = hydrate(&shuffled, LEVELS);
        let d: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| hemi_dist(&p_s[i], &p_s[j]))
            .collect();
        null.push(spearman(&d, &truth));
    }
    let mean = null.iter().sum::<f64>() / N_PERM as f64;
    let sd = (null.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / N_PERM as f64).sqrt();
    assert!(
        sd > 1e-9,
        "permutation null is degenerate (sd={sd:.2e}) - the shuffle is not \
         perturbing the statistic, so no floor is being measured"
    );
    let z = (rho_hydr - mean) / sd;
    let beats = null.iter().filter(|r| **r >= rho_hydr).count();

    println!("\nSpearman rho( geometry distance , 1 - cosine on full-dim ):");
    println!("  place-only (circular mod-17)   rho {rho_place:+.4}");
    println!("  hydrated   (hemisphere point)  rho {rho_hydr:+.4}");
    println!(
        "\npermutation null over the lemma->address map (K={N_PERM}): \
         mean {mean:+.5}  sd {sd:.5}  range [{:+.4}, {:+.4}]",
        null.iter().cloned().fold(f64::INFINITY, f64::min),
        null.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\n--- verdict (target a: spatial reconstruction) ---");
    println!("  hydrated z {z:+.2}   {beats}/{N_PERM} permutations >= observed");
    if beats == 0 && z > 3.0 {
        println!(
            "\n  HYDRATION CARRIES. A semantic HHTL address pushed through the\n  \
             shipped from_hhtl -> lift seam puts semantically-near items\n  \
             spatially near, above a shuffled-address floor. Target (a) holds,\n  \
             so asking target (b) - whether the geometry tracks TAXONOMY, not\n  \
             just position - is now worth running."
        );
    } else {
        println!(
            "\n  HYDRATION DOES NOT CARRY at this address assignment. The\n  \
             geometry is deterministic and identical under shuffle, so a null\n  \
             here means the ADDRESS never reached it - not that the geometry is\n  \
             wrong. First suspect is the mod-17 collapse of the place anchor\n  \
             (printed above); second is the rank assignment, which is the only\n  \
             other channel from address to curve position. Target (b) must NOT\n  \
             be run until (a) holds - it would measure the same silence."
        );
    }
    println!(
        "\n  Floor is a permutation, not a parametric sigma: jc pillar 5 (Jirak)\n  \
         establishes weak dependence, so a classical IID bound would understate\n  \
         it. The permutation carries the dependence structure itself."
    );

    // ── TARGET (b): does the SAME geometry track TAXONOMY? ──────────────────
    let Some((vocab, wn)) = taxonomy else { return };
    let carried_a = beats == 0 && z > 3.0;
    assert!(
        carried_a,
        "target (a) did not hold (z={z:.2}, {beats}/{N_PERM}) - target (b) must \
         NOT be interpreted, it would measure the same silence"
    );

    let names: Vec<&str> = pick.iter().map(|&i| vocab[i].as_str()).collect();
    let mut cache: HashMap<u32, u32> = HashMap::new();
    let truth_b: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| {
            wn.shared_depth(names[i], names[j], &mut cache)
                .unwrap_or(0.0)
        })
        .collect();
    let distinct_b: HashSet<u64> = truth_b.iter().map(|d| d.to_bits()).collect();
    assert!(
        distinct_b.len() >= 3,
        "wordnet depth takes only {} distinct values - no ground-truth spread",
        distinct_b.len()
    );

    // Geometry distance predicts SMALL wordnet depth (far apart => shallow
    // shared ancestor), so the expected sign is NEGATIVE. Reported as-is.
    let rho_b_place = spearman(&d_place, &truth_b);
    let rho_b_hydr = spearman(&d_hydr, &truth_b);
    let mut null_b = Vec::with_capacity(N_PERM);
    let mut shuf_b = paths.clone();
    for _ in 0..N_PERM {
        for k in (1..shuf_b.len()).rev() {
            shuf_b.swap(k, rng.below(k + 1));
        }
        let (p_s, _) = hydrate(&shuf_b, LEVELS);
        let d: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| hemi_dist(&p_s[i], &p_s[j]))
            .collect();
        null_b.push(spearman(&d, &truth_b));
    }
    let mean_b = null_b.iter().sum::<f64>() / N_PERM as f64;
    let sd_b = (null_b
        .iter()
        .map(|r| (r - mean_b) * (r - mean_b))
        .sum::<f64>()
        / N_PERM as f64)
        .sqrt();
    assert!(sd_b > 1e-9, "target (b) null is degenerate (sd={sd_b:.2e})");
    let z_b = (rho_b_hydr - mean_b) / sd_b;
    let beats_b = null_b
        .iter()
        .filter(|r| r.abs() >= rho_b_hydr.abs())
        .count();

    println!("\n--- verdict (target b: taxonomy) ---");
    println!(
        "  ground truth: wordnet shared-ancestor depth, {} distinct values",
        distinct_b.len()
    );
    println!("  place-only   rho {rho_b_place:+.4}");
    println!("  hydrated     rho {rho_b_hydr:+.4}   z {z_b:+.2}   {beats_b}/{N_PERM} |perm| >= |observed|");
    println!(
        "  null: mean {mean_b:+.5}  sd {sd_b:.5}   (two-sided: geometry distance\n           should predict SHALLOW shared ancestors, so a NEGATIVE rho is the\n           hypothesis-consistent direction)"
    );
    if beats_b == 0 && z_b.abs() > 3.0 {
        println!(
            "\n  TAXONOMY TRACKS TOO. The same address->geometry seam that carried\n  \
             POSITION also separates items by hypernym relatedness above the\n  \
             shuffled-address floor. This is the measurement 'wordnet IS HHTL'\n  \
             was asking for - stated at the strength the margin supports, not\n  \
             more."
        );
    } else {
        println!(
            "\n  TAXONOMY DOES NOT TRACK. The geometry carries POSITION (target a\n  \
             held) but not hypernym structure above the floor. That is a real\n  \
             and useful split: the address is spatially faithful and\n  \
             taxonomically blind, so 'wordnet IS HHTL' is NOT supported by the\n  \
             same seam that supports spatial hydration."
        );
    }
}

// ── target (b): WordNet taxonomy ─────────────────────────────────────────────
// Reproduced, not imported: helix is standalone and must not depend on the
// planner to run a probe. Nouns only, matching the rail's own scope.

use std::collections::{HashMap, HashSet};

struct WordNet {
    senses: HashMap<String, Vec<u32>>,
    parents: HashMap<u32, Vec<u32>>,
}

impl WordNet {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}\nBuild the rail first."));
        let mut senses: HashMap<String, Vec<u32>> = HashMap::new();
        let mut parents: HashMap<u32, Vec<u32>> = HashMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            assert_eq!(c.len(), 7, "rail schema changed: {} columns", c.len());
            if c[1] != "n" {
                continue;
            }
            let (Ok(off), Ok(hyp)) = (c[3].parse::<u32>(), c[6].parse::<u32>()) else {
                continue;
            };
            senses.entry(c[0].to_string()).or_default().push(off);
            parents.entry(off).or_default().push(hyp);
        }
        for v in senses.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        WordNet { senses, parents }
    }

    fn ancestors(&self, s: u32) -> HashMap<u32, u32> {
        let mut out: HashMap<u32, u32> = HashMap::new();
        let mut frontier = vec![s];
        let mut seen: HashSet<u32> = HashSet::from([s]);
        let mut depth = 0u32;
        while !frontier.is_empty() && depth < 32 {
            depth += 1;
            let mut next = Vec::new();
            for nd in frontier.drain(..) {
                for &p in self.parents.get(&nd).map_or(&[][..], |v| v.as_slice()) {
                    if seen.insert(p) {
                        out.insert(p, depth);
                        next.push(p);
                    }
                }
            }
            frontier = next;
        }
        out
    }

    fn root_distance(&self, s: u32) -> u32 {
        self.ancestors(s).values().copied().max().unwrap_or(0)
    }

    /// Root-distance of the DEEPEST common ancestor over all sense pairs.
    /// Higher = the two lemmas share a more specific ancestor.
    fn shared_depth(&self, a: &str, b: &str, cache: &mut HashMap<u32, u32>) -> Option<f64> {
        let (sa, sb) = (self.senses.get(a)?, self.senses.get(b)?);
        let mut best = 0u32;
        for &x in sa {
            let ax = self.ancestors(x);
            for &y in sb {
                if x == y {
                    continue;
                }
                for k in self.ancestors(y).keys() {
                    if ax.contains_key(k) {
                        let d = *cache.entry(*k).or_insert_with(|| self.root_distance(*k));
                        best = best.max(d);
                    }
                }
            }
        }
        Some(f64::from(best))
    }
}
