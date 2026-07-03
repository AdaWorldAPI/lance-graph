//! Cross-perturbation / covariance probe: project COCA-4096 onto the 64×64 tile,
//! overlay the SPO co-occurrence seeds (ngrams.info v_the_n + n_n), and measure
//! whether there is exploitable 2D covariance structure.
//!
//! Answers the operator's question in three numbers:
//!   (1) SPECTRAL GAP  — a CRUDE power-iteration probe for low-rank structure.
//!       CAVEAT: the normalized-adjacency spectrum is mixed-sign and this solver's
//!       eigenvalue ORDERING is unreliable — do NOT read the λ gap as proof. The
//!       robust evidence for 2D structure is (3): a projection can only beat the
//!       random baseline if low-rank structure genuinely exists.
//!   (2) RANK-PROJECTION edge covariance — with (x,y)=(rank%64, rank/64), how
//!       far apart do co-occurring words land? Result ≈ the random baseline
//!       (mean‖Δ‖ ≈ 0.52·64 ≈ 33) ⇒ rank layout is semantically FLAT.
//!   (3) SPECTRAL-PROJECTION edge covariance — snap the top-2 eigenvectors onto
//!       the 64×64 grid; mean edge length collapses ~1.6× (|Δx| ~3.7×) ⇒ the
//!       cross-covariance is real and exploitable ⇒ the Cam4096 reorder is worth it.
//!       This edge-length collapse is the LOAD-BEARING result, not the λ gap.
//!
//! Licensed ngram data read from a local path (argv[1], default /tmp/sources/coca),
//! never committed. Dense 4096² f32 adjacency (~67 MB, RAM only).

use deepnsm::Vocabulary;
use std::path::{Path, PathBuf};

const N: usize = 4096;
const SIDE: usize = 64;

fn rank_of(v: &Vocabulary, w: &str) -> Option<usize> {
    v.tokenize(w)
        .iter()
        .find(|t| t.is_known())
        .map(|t| t.rank_or_default() as usize)
}

fn matvec(adj: &[f32], x: &[f32], y: &mut [f32]) {
    for i in 0..N {
        let row = &adj[i * N..i * N + N];
        let mut s = 0f32;
        for j in 0..N {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Power iteration with Gram-Schmidt deflation against already-found eigenvectors.
fn eig(adj: &[f32], found: &[Vec<f32>], iters: usize) -> (Vec<f32>, f32) {
    let mut v = vec![0f32; N];
    for (i, x) in v.iter_mut().enumerate() {
        *x = (i as f32 * 0.618_034).fract() - 0.5; // deterministic start
    }
    for p in found {
        let d = dot(&v, p);
        for i in 0..N {
            v[i] -= d * p[i];
        }
    }
    normalize(&mut v);
    let mut y = vec![0f32; N];
    let mut lambda = 0f32;
    for _ in 0..iters {
        matvec(adj, &v, &mut y);
        for p in found {
            let d = dot(&y, p);
            for i in 0..N {
                y[i] -= d * p[i];
            }
        }
        lambda = dot(&v, &y);
        v.copy_from_slice(&y);
        normalize(&mut v);
    }
    (v, lambda)
}

/// Weighted covariance of edge displacement (Δx,Δy) + mean edge length.
fn edge_cov(edges: &[(usize, usize, f32)], pos: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut sw = 0f64;
    let (mut mx, mut my) = (0f64, 0f64);
    for &(a, b, w) in edges {
        let dx = (pos[b].0 - pos[a].0) as f64;
        let dy = (pos[b].1 - pos[a].1) as f64;
        mx += w as f64 * dx.abs();
        my += w as f64 * dy.abs();
        sw += w as f64;
    }
    mx /= sw;
    my /= sw;
    let (mut vxx, mut vyy, mut vxy, mut mlen) = (0f64, 0f64, 0f64, 0f64);
    for &(a, b, w) in edges {
        let dx = (pos[b].0 - pos[a].0).abs() as f64;
        let dy = (pos[b].1 - pos[a].1).abs() as f64;
        vxx += w as f64 * (dx - mx) * (dx - mx);
        vyy += w as f64 * (dy - my) * (dy - my);
        vxy += w as f64 * (dx - mx) * (dy - my);
        mlen += w as f64 * (dx * dx + dy * dy).sqrt();
    }
    let corr = vxy / (vxx.sqrt() * vyy.sqrt()).max(1e-9);
    (mx as f32, my as f32, corr as f32, (mlen / sw) as f32)
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let vocab = Vocabulary::load(&Path::new(manifest).join("word_frequency")).expect("COCA");
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/sources/coca".to_string()),
    );

    // ── build the SPO co-occurrence graph (symmetric, freq-weighted) ──
    let mut adj = vec![0f32; N * N];
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut ingest = |file: &str, ca: usize, cb: usize, minf: usize| {
        if let Ok(t) = std::fs::read_to_string(dir.join(file)) {
            for line in t.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < minf {
                    continue;
                }
                let Ok(w) = f[1].parse::<f32>() else { continue };
                if let (Some(a), Some(b)) = (rank_of(&vocab, &f[ca].to_lowercase()), rank_of(&vocab, &f[cb].to_lowercase())) {
                    if a != b {
                        adj[a * N + b] += w;
                        adj[b * N + a] += w;
                        edges.push((a, b, w));
                    }
                }
            }
        }
    };
    ingest("v_the_n.txt", 2, 4, 5); // verb·noun
    ingest("n_n.txt", 2, 3, 4); // noun·noun
    println!("SPO co-occurrence graph: {} edges over {N} COCA nodes", edges.len());

    // degree-normalize: D^-1/2 A D^-1/2 (eigvec0 ~ trivial; eigvec1,2 = semantic axes)
    let mut deg = vec![0f32; N];
    for i in 0..N {
        deg[i] = adj[i * N..i * N + N].iter().sum::<f32>().sqrt();
    }
    for i in 0..N {
        if deg[i] == 0.0 {
            continue;
        }
        for j in 0..N {
            if deg[j] != 0.0 {
                adj[i * N + j] /= deg[i] * deg[j];
            }
        }
    }

    // ── (1) spectral gap: top 4 eigenvalues ──
    let mut evs: Vec<Vec<f32>> = Vec::new();
    let mut lambdas = Vec::new();
    for _ in 0..4 {
        let (v, l) = eig(&adj, &evs, 150);
        lambdas.push(l);
        evs.push(v);
    }
    println!("\n── (1) SPECTRAL GAP — CRUDE solver, ordering UNRELIABLE (see (3)) ──");
    println!("  λ={:.4} {:.4} {:.4} {:.4}  (raw Rayleigh quotients; do NOT read as a gap)",
        lambdas[0], lambdas[1], lambdas[2], lambdas[3]);

    // ── (2) rank projection: (x,y) = (rank%64, rank/64) ──
    let rank_pos: Vec<(f32, f32)> = (0..N).map(|r| ((r % SIDE) as f32, (r / SIDE) as f32)).collect();
    let (rx, ry, rc, rlen) = edge_cov(&edges, &rank_pos);
    println!("\n── (2) RANK PROJECTION edge covariance ──");
    println!("  mean|Δx|={rx:.1} mean|Δy|={ry:.1}  corr(Δx,Δy)={rc:+.3}  mean‖Δ‖={rlen:.1} cells");

    // ── (3) spectral projection: snap eigvec1,eigvec2 onto the 64×64 grid ──
    // rank words along e1 → x band, along e2 → y band (quantile snap).
    let mut ex: Vec<usize> = (0..N).collect();
    ex.sort_by(|&a, &b| evs[1][a].partial_cmp(&evs[1][b]).unwrap());
    let mut ey: Vec<usize> = (0..N).collect();
    ey.sort_by(|&a, &b| evs[2][a].partial_cmp(&evs[2][b]).unwrap());
    let mut sem_pos = vec![(0f32, 0f32); N];
    for (band, &w) in ex.iter().enumerate() {
        sem_pos[w].0 = (band * SIDE / N) as f32;
    }
    for (band, &w) in ey.iter().enumerate() {
        sem_pos[w].1 = (band * SIDE / N) as f32;
    }
    let (sx, sy, sc, slen) = edge_cov(&edges, &sem_pos);
    println!("\n── (3) SPECTRAL PROJECTION edge covariance (Cam4096-style reorder) ──");
    println!("  mean|Δx|={sx:.1} mean|Δy|={sy:.1}  corr(Δx,Δy)={sc:+.3}  mean‖Δ‖={slen:.1} cells");

    println!("\n── VERDICT ──");
    println!("  co-occurring words: rank layout ‖Δ‖={rlen:.1} → spectral layout ‖Δ‖={slen:.1}  ({:.1}× {})",
        (rlen / slen.max(1e-3)),
        if slen < rlen { "TIGHTER — the covariance is real and exploitable" } else { "no gain" });
}
