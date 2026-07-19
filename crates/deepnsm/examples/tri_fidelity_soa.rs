//! The graph reasoning about itself: ONE 512-byte SoA node carries THREE
//! co-resident distance representations at three fidelities, and a
//! Rate-Distortion-Optimizing arbiter self-selects the cheapest lane that
//! is confident enough per query.
//!
//! This is the DeepNSM realization of the operator's directive -- store all
//! three versions in the 512-byte node and "let the Graph reason about
//! itself." It composes three already-recorded findings into one runnable
//! artifact:
//!
//!   - `E-FREQ-IS-COSINE-REPLACEMENT-1`  -- the FULL lane (raw 8-genre
//!     frequency vector) IS the semantic metric; L2 over it is the ground
//!     truth the DeepNSM 4096x4096 distance matrix materializes.
//!   - `E-IMPLICIT-MORTON-TILE-1`        -- the address does TWO jobs at two
//!     fidelities: pure shift/mask on the byte is ROUTING (coarse); ADC over
//!     the stored centroid codebook is the METRIC (fine). The 64k pairwise
//!     table is never materialized.
//!   - `E-DEEPNSM-FACET-BLIND-CONVERGENCE-1` -- frequency is a HEADER
//!     (routing) coordinate, not a payload (metric) one.
//!
//! ## The 512-byte node (canon: `key(16) | edges(16) | value(480)`)
//!
//! ```text
//! key[0..4]   classid (0 = default/bootstrap)
//! key[4..6]   HEEL = one 256:256 Morton tile  (x = leaf over genres 0..4,
//!                                               y = leaf over genres 4..8)
//! key[6..10]  HIP, TWIG  -- reserved (the finer mipmap tiers; unpopulated
//!                           on the 8-genre committed demo, see boundary)
//! key[10..16] identity (bootstrap address)
//! value[0..8] FULL lane: the 8-genre vector, i8-quantized (the 4096^2 GT)
//! value[16..32] L4 facet: classid | palette pairs -- here the SAME HEEL
//!                           tile bytes reused as the ADC codebook index
//! ```
//!
//! The Morton lane (address-only distance) and the palette lane (ADC over
//! the codebook) READ THE SAME two stored bytes -- the address IS the
//! codebook index. Only the distance FUNCTION differs. That co-residence is
//! the whole point: the graph holds routing-fidelity and metric-fidelity in
//! one row and picks per query.
//!
//! ## Self-reasoning (the RDO cascade: route -> re-rank -> verify)
//!
//! The cheap lanes ROUTE; the exact lane VERIFIES. Per `E-IMPLICIT-MORTON-
//! TILE-1`, address-only distance is ROUTING, not a final metric -- so the
//! graph never *answers* from the coarse lane; it uses it to PRUNE, then
//! spends exact FULL evals only on the survivors (the HHTL/CAM-PQ shape):
//!
//!   1. MORTON  (address-only shift/mask, no codebook load) shortlists a
//!      wide candidate net -- the net width scales with the lane's coarseness.
//!   2. PALETTE (ADC over the 2 KB codebook) re-ranks the shortlist down to a
//!      handful.
//!   3. FULL    (raw i8 L2, the ground truth) verifies the survivors exactly.
//!
//! The graph spends exact evals only on the routed survivors, and the
//! per-lane divergence vs FULL is its own self-knowledge (where it can trust
//! the cheap read). Reported: per-lane Spearman rho vs FULL (self-audit), the
//! MORTON shortlist recall, the cascade's agreement with brute-force FULL, and
//! the exact-L2 evals saved.
//!
//! KILL gates (regressions, not discoveries):
//!   - PALETTE rho vs FULL >= 0.85 (ADC preserves the metric ordering).
//!   - MORTON rho vs FULL  >  FLAT-relabel rho (hierarchy beats no-hierarchy).
//!   - cascade agreement with brute-force FULL >= 0.90, at a real eval saving.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example tri_fidelity_soa
//! ```

use std::collections::HashSet;

const DIMS: usize = 8;
const PM_START: usize = 17;
/// Words used (top-N by rank = highest frequency; deterministic slice).
const N: usize = 384;

struct WordVec {
    lemma: String,
    dims: [f64; DIMS],
}

fn load(csv_path: &str) -> Vec<WordVec> {
    let text =
        std::fs::read_to_string(csv_path).unwrap_or_else(|e| panic!("cannot read {csv_path}: {e}"));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < PM_START + DIMS {
            continue;
        }
        let lemma = f[1].to_ascii_lowercase();
        if !seen.insert(lemma.clone()) {
            continue;
        }
        let mut dims = [0.0f64; DIMS];
        let mut ok = true;
        for d in 0..DIMS {
            match f[PM_START + d].trim().parse::<f64>() {
                Ok(v) => dims[d] = (1.0 + v).ln(),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            out.push(WordVec { lemma, dims });
        }
    }
    out
}

fn zscore(words: &mut [WordVec]) {
    let n = words.len() as f64;
    for d in 0..DIMS {
        let mean = words.iter().map(|w| w.dims[d]).sum::<f64>() / n;
        let var = words
            .iter()
            .map(|w| (w.dims[d] - mean).powi(2))
            .sum::<f64>()
            / n;
        let sd = var.sqrt();
        let sd = if sd < 1e-12 { 1.0 } else { sd };
        for w in words.iter_mut() {
            w.dims[d] = (w.dims[d] - mean) / sd;
        }
    }
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// A 4-level 4-ary (`256 = 4^4`) hierarchical codebook over a `sub`-dim
/// subspace: 256 leaf centroids whose byte address encodes ancestry
/// (coarse -> fine), per the OGAR `256 = 4^4` centroid-tile canon. Returns
/// `(leaf_byte_per_point, leaf_centroids)`.
fn hier_codebook(points: &[Vec<f64>], sub: usize) -> (Vec<u8>, Vec<Vec<f64>>) {
    let mut leaf_of = vec![0u8; points.len()];
    let mut centroids: Vec<Vec<f64>> = Vec::new();
    fn centroid(points: &[Vec<f64>], idx: &[usize], sub: usize) -> Vec<f64> {
        let mut c = vec![0.0; sub];
        for &i in idx {
            for d in 0..sub {
                c[d] += points[i][d];
            }
        }
        let n = idx.len().max(1) as f64;
        for v in &mut c {
            *v /= n;
        }
        c
    }
    // Deterministic 4-means: seed on the max-variance axis (quartile
    // centers), Lloyd-iterate, no RNG.
    fn kmeans4(points: &[Vec<f64>], idx: &[usize], sub: usize) -> Vec<Vec<usize>> {
        if idx.len() <= 4 {
            let mut g = vec![Vec::new(); 4];
            for (t, &i) in idx.iter().enumerate() {
                g[t % 4].push(i);
            }
            return g;
        }
        let ax = (0..sub)
            .max_by(|&a, &b| {
                let va = variance(points, idx, a);
                let vb = variance(points, idx, b);
                va.partial_cmp(&vb).unwrap()
            })
            .unwrap();
        let mut order = idx.to_vec();
        order.sort_by(|&a, &b| points[a][ax].partial_cmp(&points[b][ax]).unwrap());
        let q = order.len() / 4;
        let mut cent: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                let pick = order[((2 * k + 1) * q / 2).min(order.len() - 1)];
                points[pick][..sub].to_vec()
            })
            .collect();
        let mut groups = vec![Vec::new(); 4];
        for _ in 0..8 {
            groups = vec![Vec::new(); 4];
            for &i in idx {
                let best = (0..4)
                    .min_by(|&a, &b| {
                        l2(&points[i][..sub], &cent[a])
                            .partial_cmp(&l2(&points[i][..sub], &cent[b]))
                            .unwrap()
                    })
                    .unwrap();
                groups[best].push(i);
            }
            for k in 0..4 {
                if !groups[k].is_empty() {
                    cent[k] = centroid(points, &groups[k], sub);
                }
            }
        }
        for k in 0..4 {
            if groups[k].is_empty() {
                let big = (0..4).max_by_key(|&t| groups[t].len()).unwrap();
                if let Some(v) = groups[big].pop() {
                    groups[k].push(v);
                }
            }
        }
        groups
    }
    fn variance(points: &[Vec<f64>], idx: &[usize], d: usize) -> f64 {
        let n = idx.len().max(1) as f64;
        let m = idx.iter().map(|&i| points[i][d]).sum::<f64>() / n;
        idx.iter().map(|&i| (points[i][d] - m).powi(2)).sum::<f64>() / n
    }
    fn build(
        points: &[Vec<f64>],
        idx: Vec<usize>,
        path: Vec<u8>,
        sub: usize,
        leaf_of: &mut [u8],
        centroids: &mut Vec<Vec<f64>>,
    ) {
        if path.len() == 4 {
            let byte = path[0] * 64 + path[1] * 16 + path[2] * 4 + path[3];
            for &i in &idx {
                leaf_of[i] = byte;
            }
            // keep centroids indexed by byte
            while centroids.len() <= byte as usize {
                centroids.push(vec![0.0; sub]);
            }
            centroids[byte as usize] =
                centroid(points, if idx.is_empty() { &[] } else { &idx }, sub);
            return;
        }
        for (k, g) in kmeans4(points, &idx, sub).into_iter().enumerate() {
            let mut np = path.clone();
            np.push(k as u8);
            let gg = if g.is_empty() { vec![idx[0]] } else { g };
            build(points, gg, np, sub, leaf_of, centroids);
        }
    }
    centroids.resize(256, vec![0.0; sub]);
    build(
        points,
        (0..points.len()).collect(),
        Vec::new(),
        sub,
        &mut leaf_of,
        &mut centroids,
    );
    (leaf_of, centroids)
}

/// A 512-byte SoA node carrying the three co-resident distance lanes
/// (canon: `key(16) | edges(16) | value(480)`).
struct TriNode {
    key: [u8; 16],
    /// The canon edge block (12 in-family + 4 out-of-family slots). Reserved
    /// and zeroed here -- this demo exercises the value lanes, not edges --
    /// but kept so the node is the real 16|16|480 = 512-byte layout.
    #[allow(dead_code)]
    edges: [u8; 16],
    value: [u8; 480],
}
const _: () = assert!(core::mem::size_of::<TriNode>() == 512);

impl TriNode {
    /// HEEL tile x-axis byte (leaf over genres 0..4) -- the Morton/palette
    /// index for the first subspace.
    fn heel_x(&self) -> u8 {
        self.key[4]
    }
    /// HEEL tile y-axis byte (leaf over genres 4..8) -- second subspace.
    fn heel_y(&self) -> u8 {
        self.key[5]
    }
    /// The FULL lane: i8-quantized 8-genre vector (value[0..8]).
    fn full_i8(&self) -> [i8; DIMS] {
        let mut v = [0i8; DIMS];
        for (slot, &b) in v.iter_mut().zip(self.value.iter()) {
            *slot = b as i8;
        }
        v
    }
}

/// Address-only Morton distance on one 256-byte axis: first-divergence 2-bit
/// group, Morton offset scaled by `4^-level`. Pure shift/mask, no centroid
/// load. (Per `E-IMPLICIT-MORTON-TILE-1`: this is ROUTING fidelity.)
fn morton_axis(a: u8, b: u8) -> f64 {
    for level in 0..4 {
        let sh = (3 - level) * 2;
        let na = (a >> sh) & 3;
        let nb = (b >> sh) & 3;
        if na != nb {
            let (ax, ay) = (na & 1, (na >> 1) & 1);
            let (bx, by) = (nb & 1, (nb >> 1) & 1);
            let off =
                (((ax as i32 - bx as i32).pow(2) + (ay as i32 - by as i32).pow(2)) as f64).sqrt();
            return 4.0f64.powi(-level) * off;
        }
    }
    0.0
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(x: &[f64]) -> Vec<f64> {
        let mut ord: Vec<usize> = (0..x.len()).collect();
        ord.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        let mut r = vec![0.0; x.len()];
        for (p, &i) in ord.iter().enumerate() {
            r[i] = p as f64;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let num: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let da: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
    let db: f64 = rb.iter().map(|x| (x - mb).powi(2)).sum::<f64>().sqrt();
    if da * db == 0.0 {
        0.0
    } else {
        num / (da * db)
    }
}

fn main() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let mut words = load(csv);
    words.truncate(N);
    let n = words.len();
    assert!(n >= 64, "need >= 64 words, got {n}");
    zscore(&mut words);

    // Two 4-dim subspaces -> two hierarchical 4^4 codebooks = one 256:256
    // Morton tile per word. (The real facet is 6 subspaces over 96 dims; the
    // committed 8-genre demo carries 2 -- see the boundary note in the run
    // report. The STRUCTURE -- co-resident lanes + RDO -- is identical.)
    let sub_x: Vec<Vec<f64>> = words.iter().map(|w| w.dims[0..4].to_vec()).collect();
    let sub_y: Vec<Vec<f64>> = words.iter().map(|w| w.dims[4..8].to_vec()).collect();
    let (leaf_x, cent_x) = hier_codebook(&sub_x, 4);
    let (leaf_y, cent_y) = hier_codebook(&sub_y, 4);

    // Materialize the 512-byte nodes: HEEL = (leaf_x : leaf_y), value = i8 vec.
    let nodes: Vec<TriNode> = (0..n)
        .map(|i| {
            let mut key = [0u8; 16];
            key[4] = leaf_x[i];
            key[5] = leaf_y[i];
            key[10] = (i & 0xff) as u8; // bootstrap identity (default basin)
            key[11] = ((i >> 8) & 0xff) as u8;
            let mut value = [0u8; 480];
            for (d, slot) in value.iter_mut().take(DIMS).enumerate() {
                // i8 quantize the z-scored vector (clamp +-3 sigma -> +-127)
                let q = (words[i].dims[d] / 3.0 * 127.0)
                    .round()
                    .clamp(-127.0, 127.0);
                *slot = q as i8 as u8;
            }
            TriNode {
                key,
                edges: [0u8; 16],
                value,
            }
        })
        .collect();

    // The three distance lanes -- all reading the ONE node.
    let d_full = |a: usize, b: usize| -> f64 {
        let (fa, fb) = (nodes[a].full_i8(), nodes[b].full_i8());
        let va: Vec<f64> = fa.iter().map(|&x| x as f64).collect();
        let vb: Vec<f64> = fb.iter().map(|&x| x as f64).collect();
        l2(&va, &vb)
    };
    let d_palette = |a: usize, b: usize| -> f64 {
        // ADC: per-subspace centroid L2 (loads the 2 KB codebook, no 64k table)
        let dx = l2(
            &cent_x[nodes[a].heel_x() as usize],
            &cent_x[nodes[b].heel_x() as usize],
        );
        let dy = l2(
            &cent_y[nodes[a].heel_y() as usize],
            &cent_y[nodes[b].heel_y() as usize],
        );
        (dx * dx + dy * dy).sqrt()
    };
    let d_morton = |a: usize, b: usize| -> f64 {
        // address-only: Morton offset on each axis byte, no centroid load
        let dx = morton_axis(nodes[a].heel_x(), nodes[b].heel_x());
        let dy = morton_axis(nodes[a].heel_y(), nodes[b].heel_y());
        (dx * dx + dy * dy).sqrt()
    };

    // Self-audit: per-lane Spearman rho vs the FULL ground truth.
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let vf: Vec<f64> = pairs.iter().map(|&(i, j)| d_full(i, j)).collect();
    let vp: Vec<f64> = pairs.iter().map(|&(i, j)| d_palette(i, j)).collect();
    let vm: Vec<f64> = pairs.iter().map(|&(i, j)| d_morton(i, j)).collect();
    let rho_p = spearman(&vp, &vf);
    let rho_m = spearman(&vm, &vf);

    // FLAT-relabel contrast for the Morton lane (destroy ancestry<->geometry):
    // relabel leaves by sorted centroid coord; morton distance loses signal.
    let mut order_x: Vec<usize> = (0..256).collect();
    order_x.sort_by(|&a, &b| cent_x[a][0].partial_cmp(&cent_x[b][0]).unwrap());
    let mut flat_x = [0u8; 256];
    for (nb, &c) in order_x.iter().enumerate() {
        flat_x[c] = nb as u8;
    }
    let vm_flat: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| {
            let dx = morton_axis(
                flat_x[nodes[i].heel_x() as usize],
                flat_x[nodes[j].heel_x() as usize],
            );
            let dy = morton_axis(nodes[i].heel_y(), nodes[j].heel_y());
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    let rho_m_flat = spearman(&vm_flat, &vf);

    // The RDO cascade: the cheap lanes ROUTE (shortlist), the exact lane
    // VERIFIES. Per E-IMPLICIT-MORTON-TILE-1, address-only distance is
    // ROUTING, not a final metric -- so the graph never *answers* from the
    // coarse lane; it uses it to PRUNE, then spends exact FULL evals only on
    // the survivors. This is the HHTL/CAM-PQ cascade shape.
    // The router width scales with the router's coarseness: the address lane
    // is rho~0.52 on 8-genre data, so it needs a generous net to retain
    // recall (the evals are free -- shift/mask, no codebook load). The
    // production 6x256:256 lane (higher rho) narrows this. The FULL-eval
    // budget is fixed by K_PALETTE, independent of the net width.
    const K_MORTON: usize = 96; // MORTON shortlist width (address, no load)
    const K_PALETTE: usize = 8; // PALETTE re-rank width (ADC, 2 KB load)
    let shortlist = |q: usize, k: usize, dist: &dyn Fn(usize, usize) -> f64| -> Vec<usize> {
        let mut cand: Vec<(usize, f64)> = (0..n)
            .filter(|&c| c != q)
            .map(|c| (c, dist(q, c)))
            .collect();
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.into_iter().take(k).map(|(c, _)| c).collect()
    };
    let argmin_over = |q: usize, cands: &[usize], dist: &dyn Fn(usize, usize) -> f64| -> usize {
        *cands
            .iter()
            .min_by(|&&a, &&b| dist(q, a).partial_cmp(&dist(q, b)).unwrap())
            .unwrap()
    };

    let mut recall_morton = 0usize; // true NN survived the MORTON shortlist
    let mut agree = 0usize; // cascade answer == brute-force FULL answer
    let mut full_evals = 0usize; // exact-L2 evaluations spent (vs n-1 brute)
    let mut sample = None;
    for q in 0..n {
        let brute = argmin_over(q, &(0..n).filter(|&c| c != q).collect::<Vec<_>>(), &d_full);
        // stage 1: MORTON routes to a shortlist (cheap, no codebook load)
        let s_morton = shortlist(q, K_MORTON, &d_morton);
        if s_morton.contains(&brute) {
            recall_morton += 1;
        }
        // stage 2: PALETTE re-ranks the shortlist by ADC (loads codebook)
        let mut s_pal = s_morton.clone();
        s_pal.sort_by(|&a, &b| d_palette(q, a).partial_cmp(&d_palette(q, b)).unwrap());
        s_pal.truncate(K_PALETTE);
        // stage 3: FULL verifies the tiny survivor set (exact, K_PALETTE evals)
        full_evals += s_pal.len();
        let answer = argmin_over(q, &s_pal, &d_full);
        if answer == brute {
            agree += 1;
        }
        if sample.is_none() && brute != usize::MAX {
            sample = Some((q, answer, brute));
        }
    }
    let agree_frac = agree as f64 / n as f64;
    let recall_frac = recall_morton as f64 / n as f64;
    let brute_evals = n * (n - 1);
    let eval_saving = 1.0 - full_evals as f64 / brute_evals as f64;

    // ---- report ----
    println!("tri-fidelity SoA: {n} words, one 512-byte node each (key 16 | edges 16 | value 480)");
    println!("  stored: HEEL 256:256 tile (leaf_x:leaf_y) in key[4..6]; i8 8-genre vector in value[0..8]");
    println!();
    println!("SELF-AUDIT -- per-lane Spearman rho vs FULL (the 4096^2 ground truth):");
    println!("  FULL    (raw i8 L2, ground truth)            rho = 1.0000");
    println!("  PALETTE (ADC over 2 KB centroid codebook)    rho = {rho_p:.4}");
    println!("  MORTON  (address-only shift/mask, hierarch.) rho = {rho_m:.4}");
    println!("  MORTON  (address-only, FLAT-relabel)         rho = {rho_m_flat:.4}  <- ancestry contrast");
    println!();
    println!(
        "SELF-OPTIMIZE -- the RDO cascade (MORTON routes -> PALETTE re-ranks -> FULL verifies):"
    );
    println!(
        "  MORTON shortlist recall (true NN in top-{K_MORTON}): {recall_morton}/{n}  ({:.1}%)",
        100.0 * recall_frac
    );
    println!(
        "  cascade answer == brute-force FULL NN:           {agree}/{n}  ({:.1}%)",
        100.0 * agree_frac
    );
    println!(
        "  exact-L2 evals: {full_evals} vs {brute_evals} brute  ->  {:.1}% saved",
        100.0 * eval_saving
    );
    if let Some((q, ans, _)) = sample {
        println!(
            "  e.g. nearest to '{}' -> '{}' (routed by address, verified by exact L2)",
            words[q].lemma, words[ans].lemma
        );
    }
    println!();

    // KILL gates (regression guards, not discoveries).
    let mut fail = Vec::new();
    if rho_p < 0.85 {
        fail.push(format!(
            "PALETTE rho {rho_p:.4} < 0.85 (ADC lost the metric ordering)"
        ));
    }
    if rho_m <= rho_m_flat {
        fail.push(format!(
            "MORTON rho {rho_m:.4} <= FLAT {rho_m_flat:.4} (hierarchy gave no routing signal)"
        ));
    }
    if agree_frac < 0.90 {
        fail.push(format!(
            "cascade agreement {agree_frac:.4} < 0.90 (routing dropped the true NN)"
        ));
    }
    if eval_saving <= 0.0 {
        fail.push("cascade saved no exact-L2 evals (no self-optimization)".to_string());
    }
    if fail.is_empty() {
        println!("KILL GATES: all pass -- the graph routes with its own address lane and");
        println!(
            "verifies with its own exact lane, matching brute force at a fraction of the cost."
        );
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }

    println!();
    println!("BOUNDARY: committed 8-genre data -> a 2x256:256 tile (2 subspaces). The");
    println!("production facet is 6x256:256 over 96 subgenres (gitignored codebook_pq.bin),");
    println!("which sharpens PALETTE/MORTON rho; the tri-lane co-residence + RDO structure");
    println!(
        "is identical. See EPIPHANIES E-IMPLICIT-MORTON-TILE-1, E-FREQ-IS-COSINE-REPLACEMENT-1."
    );
}
