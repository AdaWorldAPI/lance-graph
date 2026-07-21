//! `palette_distance_recall` — measure near-neighbor recall of the REAL
//! `lance_graph_contract::distance` palette code (`[u8; 6]` CamPqCode byte-L1)
//! against Jina-v3 ground-truth nearest-neighbors, over the whole 19,869-word
//! academic vocabulary.
//!
//! This uses the SHIPPED distance function (`Distance::distance` on `[u8; 6]`),
//! not a home-rolled cosine-on-reconstruction. The palette is Morton-ordered
//! upstream (each byte = a 4⁴ DN path, high bits = coarse ancestry), so byte-L1
//! is a meaningful metric. Codes + queries + true-NN are produced offline
//! (`scratchpad/encode_palette.py`, Jina embeddings) and read here as raw LE bytes.
//!
//! ```sh
//! PAL_DIR=/path/to/scratchpad \
//!   cargo run -p lance-graph-contract --example palette_distance_recall
//! ```

use lance_graph_contract::distance::Distance;
use std::fs;

fn read_u32s(path: &str) -> Vec<u32> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let dir = std::env::var("PAL_DIR").unwrap_or_else(|_| {
        "/tmp/claude-0/-home-user/02f99e02-de80-5ce5-b368-b1a15f3a164f/scratchpad".to_string()
    });
    let Ok(meta) = fs::read_to_string(format!("{dir}/pal_meta.txt")) else {
        println!(
            "palette_distance_recall: no encoded data at {dir} — skipping.\n\
             Generate it with scratchpad/encode_palette.py (Jina embeddings of a vocab)\n\
             then set PAL_DIR. The example demonstrates recall via the shipped\n\
             lance_graph_contract::distance byte-L1 on [u8; 6] CamPqCode."
        );
        return;
    };
    let mut it = meta.split_whitespace();
    let n: usize = it.next().unwrap().parse().unwrap();
    let m: usize = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();
    let k: usize = it.next().unwrap().parse().unwrap();
    assert_eq!(m, 6, "this example expects the 6-byte CamPqCode");

    let raw = fs::read(format!("{dir}/pal_codes.u8")).expect("pal_codes.u8");
    assert_eq!(raw.len(), n * 6);
    // codes as [u8; 6] per word (the shipped CamPqCode carrier).
    let codes: Vec<[u8; 6]> = raw
        .chunks_exact(6)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
        .collect();
    let queries = read_u32s(&format!("{dir}/pal_queries.u32"));
    let truenn = read_u32s(&format!("{dir}/pal_truenn.u32"));
    assert_eq!(queries.len(), q);
    assert_eq!(truenn.len(), q * k);

    // recall@k: for each query word, does the REAL contract::distance byte-L1
    // over the palette codes recover its true Jina nearest-neighbors?
    let mut hits = 0usize;
    for (qi, &qw) in queries.iter().enumerate() {
        let qc = codes[qw as usize];
        // top-k nearest by palette byte-L1 (contract::distance)
        let mut scored: Vec<(u32, usize)> = codes
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as u32 != qw)
            .map(|(i, c)| (qc.distance(c), i))
            .collect();
        scored.select_nth_unstable_by(k, |a, b| a.0.cmp(&b.0));
        let got: std::collections::HashSet<u32> =
            scored[..k].iter().map(|&(_, i)| i as u32).collect();
        for j in 0..k {
            if got.contains(&truenn[qi * k + j]) {
                hits += 1;
            }
        }
    }
    let recall = hits as f64 / (q * k) as f64;

    println!("palette_distance_recall — REAL lance_graph_contract::distance on [u8; 6] CamPqCode");
    println!("  vocab {n} words · {q} queries · recall@{k} vs Jina-v3 true NN");
    println!(
        "  code footprint: {} bytes = {:.1} kB (6 B/word)",
        n * 6,
        (n * 6) as f64 / 1024.0
    );
    println!("  recall@{k} (contract::distance byte-L1, Morton-ordered palette): {recall:.3}");

    // gate: the real palette distance must beat the routing-address baseline (0.002)
    // by a wide margin — the codec is a metric, the address is not.
    let green = recall > 0.05;
    println!(
        "\n[{}] G1 palette byte-L1 recovers real near-neighbors (>{:.3} routing floor)",
        if green { "PASS" } else { "FAIL" },
        0.05
    );
    assert!(green, "palette distance recall below the routing floor");
    println!("ALL GATES GREEN");
}
