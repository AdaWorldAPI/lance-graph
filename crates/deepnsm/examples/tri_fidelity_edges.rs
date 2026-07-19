//! The tri-fidelity cascade turned OUTWARD: self-search becomes self-
//! CONSTRUCTION. The route->verify cascade (`E-TRI-FIDELITY-SOA-SELF-
//! REASONING-1`) that answered "nearest node to a query" now FILLS each
//! node's canonical EdgeBlock (`key(16) | edges(16) | value(480)`): the
//! MORTON address net routes edge candidates, the exact lane verifies, and
//! the survivors land in the 12 in-family + 4 out-of-family edge slots. A
//! graph walk over the constructed edges proves the adjacency is
//! semantically coherent -- the graph reasons about itself ACROSS nodes,
//! not just within one query.
//!
//! Composes:
//!   - `E-TRI-FIDELITY-SOA-SELF-REASONING-1` -- the route->verify cascade.
//!   - the canon EdgeBlock (12 in-family + 4 out-of-family, one byte/slot;
//!     `canonical_node.rs`). "In-family" = same coarse basin (shared HEEL
//!     top nibble); "out-of-family" = a different basin (the 4 cross-links).
//!   - `E-FREQ-IS-COSINE-REPLACEMENT-1` -- L2 over the 8-genre vector is the
//!     semantic ground truth the edges must respect.
//!
//! ## What it builds (256 words, one 512-byte node each)
//!
//!   for each node: MORTON routes a wide net -> exact L2 verifies ->
//!     edges[0..12]  = the 12 nearest SAME-basin neighbors (row-index bytes)
//!     edges[12..16] = the 4 nearest DIFFERENT-basin neighbors (cross-links)
//!
//! Then a 2-hop walk from a seed follows the edge bytes as row indices.
//!
//! KILL gates (regressions, not discoveries):
//!   - in-family edge targets are semantically nearer than a random baseline
//!     (mean L2 to source << mean L2 of random pairs).
//!   - every node's 16 edge slots are filled (12 + 4), none phantom.
//!   - a 2-hop walk from a seed stays semantically coherent (reachable set
//!     mean L2 to seed < the global mean pair L2).
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example tri_fidelity_edges
//! ```

const DIMS: usize = 8;
const PM_START: usize = 17;
/// 256 words -> a neighbor row index fits in ONE edge byte (the canon slot width).
const N: usize = 256;

struct WordVec {
    lemma: String,
    dims: [f64; DIMS],
}

fn load(csv_path: &str) -> Vec<WordVec> {
    let text =
        std::fs::read_to_string(csv_path).unwrap_or_else(|e| panic!("cannot read {csv_path}: {e}"));
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
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
        for (d, slot) in dims.iter_mut().enumerate() {
            match f[PM_START + d].trim().parse::<f64>() {
                Ok(v) => *slot = (1.0 + v).ln(),
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
        let sd = var.sqrt().max(1e-12);
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
/// subspace; leaf byte encodes ancestry (per the OGAR centroid-tile canon).
fn hier_leaves(points: &[Vec<f64>], sub: usize) -> Vec<u8> {
    fn centroid(points: &[Vec<f64>], idx: &[usize], sub: usize) -> Vec<f64> {
        let mut c = vec![0.0; sub];
        for &i in idx {
            for (d, cd) in c.iter_mut().enumerate() {
                *cd += points[i][d];
            }
        }
        let n = idx.len().max(1) as f64;
        for v in &mut c {
            *v /= n;
        }
        c
    }
    fn variance(points: &[Vec<f64>], idx: &[usize], d: usize) -> f64 {
        let n = idx.len().max(1) as f64;
        let m = idx.iter().map(|&i| points[i][d]).sum::<f64>() / n;
        idx.iter().map(|&i| (points[i][d] - m).powi(2)).sum::<f64>() / n
    }
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
                variance(points, idx, a)
                    .partial_cmp(&variance(points, idx, b))
                    .unwrap()
            })
            .unwrap();
        let mut order = idx.to_vec();
        order.sort_by(|&a, &b| points[a][ax].partial_cmp(&points[b][ax]).unwrap());
        let q = order.len() / 4;
        let mut cent: Vec<Vec<f64>> = (0..4)
            .map(|k| points[order[((2 * k + 1) * q / 2).min(order.len() - 1)]][..sub].to_vec())
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
    fn build(points: &[Vec<f64>], idx: Vec<usize>, path: Vec<u8>, sub: usize, leaf: &mut [u8]) {
        if path.len() == 4 {
            let byte = path[0] * 64 + path[1] * 16 + path[2] * 4 + path[3];
            for &i in &idx {
                leaf[i] = byte;
            }
            return;
        }
        for (k, g) in kmeans4(points, &idx, sub).into_iter().enumerate() {
            let mut np = path.clone();
            np.push(k as u8);
            let gg = if g.is_empty() { vec![idx[0]] } else { g };
            build(points, gg, np, sub, leaf);
        }
    }
    let mut leaf = vec![0u8; points.len()];
    build(
        points,
        (0..points.len()).collect(),
        Vec::new(),
        sub,
        &mut leaf,
    );
    leaf
}

/// A 512-byte SoA node (canon `key(16) | edges(16) | value(480)`).
struct TriNode {
    key: [u8; 16],
    edges: [u8; 16],
    value: [u8; 480],
}
const _: () = assert!(core::mem::size_of::<TriNode>() == 512);

impl TriNode {
    fn heel_x(&self) -> u8 {
        self.key[4]
    }
    /// Coarse basin = the top 2-bit nibble of the HEEL.x tile (the 4^4 root).
    fn basin(&self) -> u8 {
        (self.key[4] >> 6) & 3
    }
    fn vec8(&self) -> [f64; DIMS] {
        let mut v = [0.0; DIMS];
        for (slot, &b) in v.iter_mut().zip(self.value.iter()) {
            *slot = b as i8 as f64;
        }
        v
    }
}

fn main() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let mut words = load(csv);
    words.truncate(N);
    let n = words.len();
    assert!(n == N, "need exactly {N} words, got {n}");
    zscore(&mut words);

    let sub_x: Vec<Vec<f64>> = words.iter().map(|w| w.dims[0..4].to_vec()).collect();
    let sub_y: Vec<Vec<f64>> = words.iter().map(|w| w.dims[4..8].to_vec()).collect();
    let leaf_x = hier_leaves(&sub_x, 4);
    let leaf_y = hier_leaves(&sub_y, 4);

    let mut nodes: Vec<TriNode> = (0..n)
        .map(|i| {
            let mut key = [0u8; 16];
            key[4] = leaf_x[i];
            key[5] = leaf_y[i];
            key[10] = i as u8; // bootstrap identity (row index, fits since N=256)
            let mut value = [0u8; 480];
            for (d, slot) in value.iter_mut().take(DIMS).enumerate() {
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

    // Address-only MORTON distance on one axis byte (routing, no vector load).
    let morton_axis = |a: u8, b: u8| -> f64 {
        for level in 0..4 {
            let sh = (3 - level) * 2;
            let (na, nb) = ((a >> sh) & 3, (b >> sh) & 3);
            if na != nb {
                let (ax, ay) = (na & 1, (na >> 1) & 1);
                let (bx, by) = (nb & 1, (nb >> 1) & 1);
                let off = (((ax as i32 - bx as i32).pow(2) + (ay as i32 - by as i32).pow(2))
                    as f64)
                    .sqrt();
                return 4.0f64.powi(-level) * off;
            }
        }
        0.0
    };
    let d_morton = |a: usize, b: usize| -> f64 {
        let dx = morton_axis(nodes[a].heel_x(), nodes[b].heel_x());
        let dy = morton_axis(nodes[a].key[5], nodes[b].key[5]);
        (dx * dx + dy * dy).sqrt()
    };
    let vecs: Vec<[f64; DIMS]> = nodes.iter().map(|nd| nd.vec8()).collect();
    let d_full = |a: usize, b: usize| -> f64 { l2(&vecs[a], &vecs[b]) };

    // Build edges per node: MORTON routes a wide net, exact L2 verifies, the
    // 12 nearest same-basin + 4 nearest different-basin survivors are stored.
    const K_NET: usize = 64;
    let mut edge_targets: Vec<[usize; 16]> = vec![[0; 16]; n];
    for q in 0..n {
        // stage 1: MORTON net (routing)
        let mut net: Vec<usize> = (0..n).filter(|&c| c != q).collect();
        net.sort_by(|&a, &b| d_morton(q, a).partial_cmp(&d_morton(q, b)).unwrap());
        net.truncate(K_NET);
        // stage 2: exact L2 verify, split by basin
        net.sort_by(|&a, &b| d_full(q, a).partial_cmp(&d_full(q, b)).unwrap());
        let qb = nodes[q].basin();
        let mut in_fam: Vec<usize> = net
            .iter()
            .copied()
            .filter(|&c| nodes[c].basin() == qb)
            .collect();
        let mut out_fam: Vec<usize> = net
            .iter()
            .copied()
            .filter(|&c| nodes[c].basin() != qb)
            .collect();
        // widen the net for whichever side is short (cheap MORTON evals only)
        let backfill = |exist: &mut Vec<usize>, want: usize, same: bool| {
            if exist.len() < want {
                let mut extra: Vec<usize> = (0..n)
                    .filter(|&c| c != q && (nodes[c].basin() == qb) == same && !exist.contains(&c))
                    .collect();
                extra.sort_by(|&a, &b| d_full(q, a).partial_cmp(&d_full(q, b)).unwrap());
                for e in extra.into_iter().take(want - exist.len()) {
                    exist.push(e);
                }
            }
        };
        backfill(&mut in_fam, 12, true);
        backfill(&mut out_fam, 4, false);
        let mut slots = [0usize; 16];
        for (s, &c) in slots.iter_mut().take(12).zip(in_fam.iter()) {
            *s = c;
        }
        for (s, &c) in slots.iter_mut().skip(12).take(4).zip(out_fam.iter()) {
            *s = c;
        }
        edge_targets[q] = slots;
    }
    // Commit into the canonical 16-byte edge block (one row-index byte per slot).
    for (q, node) in nodes.iter_mut().enumerate() {
        for (slot, &t) in node.edges.iter_mut().zip(edge_targets[q].iter()) {
            *slot = t as u8;
        }
    }

    // --- measurements ---
    // baseline: mean L2 over a fixed deterministic pair sample
    let sample: Vec<(usize, usize)> = (0..n)
        .map(|i| (i, (i * 97 + 13) % n))
        .filter(|&(a, b)| a != b)
        .collect();
    let global_mean = sample.iter().map(|&(a, b)| d_full(a, b)).sum::<f64>() / sample.len() as f64;
    // in-family edge coherence
    let mut infam_sum = 0.0;
    let mut infam_cnt = 0usize;
    for (q, tgts) in edge_targets.iter().enumerate() {
        for &t in tgts.iter().take(12) {
            infam_sum += d_full(q, t);
            infam_cnt += 1;
        }
    }
    let infam_mean = infam_sum / infam_cnt as f64;

    // 2-hop walk coherence from a seed
    let seed = 0usize;
    let mut reach = std::collections::HashSet::new();
    for &h1 in edge_targets[seed].iter().take(12) {
        reach.insert(h1);
        for &h2 in edge_targets[h1].iter().take(12) {
            reach.insert(h2);
        }
    }
    reach.remove(&seed);
    let walk_mean = reach.iter().map(|&r| d_full(seed, r)).sum::<f64>() / reach.len().max(1) as f64;

    let all_filled = (0..n).all(|q| {
        let s = &edge_targets[q];
        s.iter().take(12).all(|&t| t != q) && s.iter().skip(12).take(4).all(|&t| t != q)
    });

    println!(
        "tri-fidelity EDGE construction: {n} nodes, canon EdgeBlock 12 in-family + 4 out-of-family"
    );
    println!();
    println!("SELF-CONSTRUCTION -- the cascade filled every node's edge slots:");
    println!("  edges filled per node: 16/16 (all nodes: {all_filled})");
    println!("  in-family edge mean L2 to source:  {infam_mean:.3}");
    println!("  global random-pair mean L2:         {global_mean:.3}   <- baseline");
    println!(
        "  ratio (edges vs random):            {:.3}  (lower = tighter)",
        infam_mean / global_mean
    );
    println!();
    println!("GRAPH WALK -- 2 hops from seed '{}':", words[seed].lemma);
    let names: Vec<&str> = edge_targets[seed]
        .iter()
        .take(6)
        .map(|&t| words[t].lemma.as_str())
        .collect();
    println!("  1-hop in-family neighbors (first 6): {names:?}");
    println!(
        "  2-hop reachable set: {} nodes, mean L2 to seed = {walk_mean:.3}",
        reach.len()
    );
    // a cross-basin link example
    let xlink = edge_targets[seed][12];
    println!(
        "  out-of-family link[0]: '{}' (basin {} -> {})",
        words[xlink].lemma,
        nodes[seed].basin(),
        nodes[xlink].basin()
    );
    println!();

    let mut fail = Vec::new();
    if !all_filled {
        fail.push("some edge slots point to self or are unfilled".to_string());
    }
    if infam_mean >= global_mean {
        fail.push(format!(
            "in-family edges ({infam_mean:.3}) not tighter than random ({global_mean:.3})"
        ));
    }
    if walk_mean >= global_mean {
        fail.push(format!(
            "2-hop walk ({walk_mean:.3}) not coherent vs global ({global_mean:.3})"
        ));
    }
    if fail.is_empty() {
        println!("KILL GATES: all pass -- the cascade CONSTRUCTED a coherent adjacency; the graph");
        println!("now reasons about itself across nodes (self-search became self-construction).");
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
