//! D-ARW-0 §16 — bounded address-locality falsifier.
//!
//! This is deliberately NOT a production bridge and NOT an address-identity proof.
//! It asks the smaller question that source archaeology exposed first:
//!
//! > Does the raw 12-bit COCA frequency-rank id carry useful 64×64 spatial
//! > locality when read row-major or when read as a Morton code?
//!
//! The oracle is external to either candidate mapping: real COCA n-gram
//! co-occurrence relations (`v_the_n.txt` + `n_n.txt`, ngrams.info /
//! english-corpora.org; licensed data supplied locally, never committed).
//!
//! Arms:
//! - A2: raw COCA rank → row-major `(x = id % 64, y = id / 64)`.
//! - A3: raw COCA rank IS a Morton code → decode through the canonical
//!   `lance_graph_contract::facet::FacetTier::morton` primitive.
//! - A4: 32 fixed seeded random bijections → sabotage baseline.
//!
//! Why this precedes the larger §16 identity probe:
//! `Vocabulary` defines the 12-bit id as COCA frequency rank. The older
//! `gridlake_spo_covariance.rs` probe found row-major rank spatially flat and
//! a semantic reorder useful. If A2 and A3 both look like the permutation
//! baseline here, the direct `codebook_id → cell` reading is killed and the
//! next honest candidate becomes:
//!
//!     semantic placement / CAM reorder → 64×64 field → Morton addressing
//!
//! rather than `Morton(raw_frequency_rank)`.
//!
//! Fences:
//! - no P64/CE64 production wiring;
//! - no evidence, ReasoningBand, Revision, or Rubicon mutation;
//! - a locality signal is NOT exact historical address identity;
//! - missing licensed n-gram input is a hard error, never NaN / silent skip.
//!
//! Run:
//! `cargo run --release --manifest-path crates/deepnsm/Cargo.toml \
!    --example d_arw_0_address_locality -- /path/to/coca-ngram-samples`

use deepnsm::Vocabulary;
use lance_graph_contract::facet::FacetTier;
use std::error::Error;
use std::path::{Path, PathBuf};

const N: usize = 4096;
const SIDE: usize = 64;
const SABOTAGE_ARMS: usize = 32;
const SABOTAGE_SEED: u64 = 0xD_A7A_0A11_5EED_2026;

#[derive(Clone, Copy, Debug)]
struct Edge {
    a: usize,
    b: usize,
    weight: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    total_weight: f64,
    mean_euclid: f64,
    mean_manhattan: f64,
    near1: f64,
    near2: f64,
    near4: f64,
    near8: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Baseline {
    mean_euclid: f64,
    sd_euclid: f64,
    mean_near4: f64,
    sd_near4: f64,
}

fn rank_of(vocab: &Vocabulary, word: &str) -> Option<usize> {
    vocab
        .tokenize(word)
        .iter()
        .find(|t| t.is_known())
        .map(|t| t.rank_or_default() as usize)
        .filter(|&r| r < N)
}

fn load_edges(vocab: &Vocabulary, dir: &Path) -> Result<Vec<Edge>, Box<dyn Error>> {
    let mut edges = Vec::new();

    let mut ingest = |file: &str,
                      word_a_col: usize,
                      word_b_col: usize,
                      min_fields: usize|
     -> Result<(), Box<dyn Error>> {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!(
                "required licensed n-gram input is missing/unreadable: {} ({e})",
                path.display()
            )
        })?;

        for line in text.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < min_fields {
                continue;
            }
            let Ok(weight) = fields[1].parse::<f64>() else {
                continue;
            };
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }

            let Some(wa) = fields.get(word_a_col) else {
                continue;
            };
            let Some(wb) = fields.get(word_b_col) else {
                continue;
            };

            if let (Some(a), Some(b)) = (
                rank_of(vocab, &wa.to_lowercase()),
                rank_of(vocab, &wb.to_lowercase()),
            ) {
                if a != b {
                    edges.push(Edge { a, b, weight });
                }
            }
        }
        Ok(())
    };

    // ngrams.info sample formats already used by the older gridlake probes.
    ingest("v_the_n.txt", 2, 4, 5)?; // verb · "the" · noun
    ingest("n_n.txt", 2, 3, 4)?; // noun · noun

    if edges.is_empty() {
        return Err("no in-vocabulary relation edges were loaded; probe cannot run".into());
    }

    Ok(edges)
}

fn row_major_positions() -> Vec<(u8, u8)> {
    (0..N)
        .map(|id| ((id % SIDE) as u8, (id / SIDE) as u8))
        .collect()
}

/// Build the inverse lookup by ENUMERATING the canonical Morton encoder.
///
/// This avoids a second hand-written Morton implementation. For every
/// `(x,y) ∈ [0,64)²`, `FacetTier::morton` emits exactly one 12-bit code.
/// The returned vector therefore answers: if a raw COCA id were already a
/// Morton code, which 64×64 cell would it name?
fn morton_decode_positions() -> Vec<(u8, u8)> {
    let mut out = vec![(0u8, 0u8); N];
    let mut seen = vec![false; N];

    for y in 0..SIDE {
        for x in 0..SIDE {
            let code = FacetTier {
                lo: x as u8,
                hi: y as u8,
            }
            .morton() as usize;

            assert!(code < N, "6-bit × 6-bit Morton must fit 12 bits");
            assert!(!seen[code], "canonical Morton must be bijective on 64×64");
            seen[code] = true;
            out[code] = (x as u8, y as u8);
        }
    }

    assert!(seen.into_iter().all(|v| v));
    out
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A fixed random BIJECTION from rank id → grid cell.
///
/// This is the sabotage arm. It preserves cardinality and exact identity
/// count, while destroying any spatial meaning carried by the original
/// assignment.
fn permutation_positions(seed: u64) -> Vec<(u8, u8)> {
    let mut cells: Vec<usize> = (0..N).collect();
    let mut state = seed;

    for i in (1..N).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        cells.swap(i, j);
    }

    cells
        .into_iter()
        .map(|cell| ((cell % SIDE) as u8, (cell / SIDE) as u8))
        .collect()
}

fn score(edges: &[Edge], pos: &[(u8, u8)]) -> Metrics {
    let mut out = Metrics::default();

    for edge in edges {
        let (ax, ay) = pos[edge.a];
        let (bx, by) = pos[edge.b];

        let dx = f64::from(ax.abs_diff(bx));
        let dy = f64::from(ay.abs_diff(by));
        let euclid = (dx * dx + dy * dy).sqrt();
        let manhattan = dx + dy;
        let chebyshev = dx.max(dy);

        out.total_weight += edge.weight;
        out.mean_euclid += edge.weight * euclid;
        out.mean_manhattan += edge.weight * manhattan;
        if chebyshev <= 1.0 {
            out.near1 += edge.weight;
        }
        if chebyshev <= 2.0 {
            out.near2 += edge.weight;
        }
        if chebyshev <= 4.0 {
            out.near4 += edge.weight;
        }
        if chebyshev <= 8.0 {
            out.near8 += edge.weight;
        }
    }

    let w = out.total_weight;
    assert!(w > 0.0);
    out.mean_euclid /= w;
    out.mean_manhattan /= w;
    out.near1 /= w;
    out.near2 /= w;
    out.near4 /= w;
    out.near8 /= w;
    out
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

fn sabotage_baseline(edges: &[Edge]) -> Baseline {
    let mut euclid = Vec::with_capacity(SABOTAGE_ARMS);
    let mut near4 = Vec::with_capacity(SABOTAGE_ARMS);

    for arm in 0..SABOTAGE_ARMS {
        let seed = SABOTAGE_SEED ^ (arm as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let m = score(edges, &permutation_positions(seed));
        euclid.push(m.mean_euclid);
        near4.push(m.near4);
    }

    let (mean_euclid, sd_euclid) = mean_sd(&euclid);
    let (mean_near4, sd_near4) = mean_sd(&near4);
    Baseline {
        mean_euclid,
        sd_euclid,
        mean_near4,
        sd_near4,
    }
}

fn z_better_shorter(candidate: f64, random_mean: f64, random_sd: f64) -> f64 {
    if random_sd == 0.0 {
        0.0
    } else {
        (random_mean - candidate) / random_sd
    }
}

fn z_better_more(candidate: f64, random_mean: f64, random_sd: f64) -> f64 {
    if random_sd == 0.0 {
        0.0
    } else {
        (candidate - random_mean) / random_sd
    }
}

fn print_metrics(name: &str, m: Metrics, base: Baseline) {
    let z_len = z_better_shorter(m.mean_euclid, base.mean_euclid, base.sd_euclid);
    let z_near4 = z_better_more(m.near4, base.mean_near4, base.sd_near4);

    println!(
        "{name:<18}  mean‖Δ‖={:>7.3}  meanL1={:>7.3}  near≤1={:>7.3}%  near≤2={:>7.3}%  near≤4={:>7.3}%  near≤8={:>7.3}%  z_len={:+6.2}  z_near4={:+6.2}",
        m.mean_euclid,
        m.mean_manhattan,
        100.0 * m.near1,
        100.0 * m.near2,
        100.0 * m.near4,
        100.0 * m.near8,
        z_len,
        z_near4,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let vocab =
        Vocabulary::load(&Path::new(manifest).join("word_frequency")).expect("load COCA vocabulary");

    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/sources/coca".to_string()),
    );

    let edges = load_edges(&vocab, &dir)?;
    let row = score(&edges, &row_major_positions());
    let morton = score(&edges, &morton_decode_positions());
    let random = sabotage_baseline(&edges);

    println!("D-ARW-0 §16 raw-codebook address locality probe");
    println!("corpus: {} weighted COCA n-gram relation rows", edges.len());
    println!(
        "random sabotage (n={}): mean‖Δ‖={:.3}±{:.3}, near≤4={:.3}%±{:.3}%",
        SABOTAGE_ARMS,
        random.mean_euclid,
        random.sd_euclid,
        100.0 * random.mean_near4,
        100.0 * random.sd_near4,
    );
    println!();
    print_metrics("A2 row-major", row, random);
    print_metrics("A3 Morton-decode", morton, random);

    let row_len_z = z_better_shorter(row.mean_euclid, random.mean_euclid, random.sd_euclid);
    let row_near_z = z_better_more(row.near4, random.mean_near4, random.sd_near4);
    let morton_len_z =
        z_better_shorter(morton.mean_euclid, random.mean_euclid, random.sd_euclid);
    let morton_near_z = z_better_more(morton.near4, random.mean_near4, random.sd_near4);

    println!("\nVERDICT");
    if row_len_z <= 1.0 && row_near_z <= 1.0 {
        println!(
            "  A2 NO-BUY signal: raw COCA frequency rank is within one sabotage σ on both locality readings."
        );
    } else {
        println!(
            "  A2 SIGNAL ONLY: row-major rank beats the sabotage envelope on at least one reading; this is locality, not address identity."
        );
    }

    if morton_len_z <= 1.0 && morton_near_z <= 1.0 {
        println!(
            "  A3 NO-BUY signal: treating raw COCA frequency rank as a Morton code is within one sabotage σ on both locality readings."
        );
        println!(
            "  Next candidate is semantic/CAM placement → 64×64 field → Morton addressing, not Morton(raw rank)."
        );
    } else {
        println!(
            "  A3 SIGNAL ONLY: Morton(raw rank) beats the sabotage envelope on at least one reading; exact historical identity is STILL unproven."
        );
    }

    println!(
        "  HARD FENCE: this probe cannot authorize PerturbationDto→P64 wiring, CE64 reinterpretation, evidence changes, Revision, or Rubicon."
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_morton_decode_is_complete_bijection() {
        let pos = morton_decode_positions();
        assert_eq!(pos.len(), N);

        let mut seen = vec![false; N];
        for (x, y) in pos {
            let linear = y as usize * SIDE + x as usize;
            assert!(linear < N);
            assert!(!seen[linear]);
            seen[linear] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn sabotage_is_deterministic_and_bijective() {
        let a = permutation_positions(SABOTAGE_SEED);
        let b = permutation_positions(SABOTAGE_SEED);
        assert_eq!(a, b);

        let mut seen = vec![false; N];
        for (x, y) in a {
            let linear = y as usize * SIDE + x as usize;
            assert!(!seen[linear]);
            seen[linear] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn morton_known_corners_round_trip_through_certified_encoder() {
        let pos = morton_decode_positions();
        for &(x, y) in &[(0u8, 0u8), (1, 0), (0, 1), (63, 63), (17, 42)] {
            let code = FacetTier { lo: x, hi: y }.morton() as usize;
            assert_eq!(pos[code], (x, y));
        }
    }

    #[test]
    fn locality_score_prefers_a_tight_mapping_over_a_far_mapping() {
        let edges = [Edge {
            a: 0,
            b: 1,
            weight: 1.0,
        }];

        let mut tight = row_major_positions();
        tight[0] = (10, 10);
        tight[1] = (11, 10);

        let mut far = tight.clone();
        far[1] = (63, 63);

        let mt = score(&edges, &tight);
        let mf = score(&edges, &far);
        assert!(mt.mean_euclid < mf.mean_euclid);
        assert!(mt.near1 > mf.near1);
    }
}
