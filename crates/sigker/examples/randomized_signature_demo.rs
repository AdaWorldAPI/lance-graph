//! Demonstrate the randomized-signature universality property: encodings
//! with the SAME random seed are reproducible across runs, and the cosine
//! similarity tracks path similarity smoothly.
//!
//! Run:
//!   cargo run --manifest-path crates/sigker/Cargo.toml \
//!             --example randomized_signature_demo --release

use sigker::randomized::RandomizedSignatureBuilder;

fn main() {
    let builder = RandomizedSignatureBuilder::new(2, 64, 0xADA_F00D);

    // Three reference paths.
    let line = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
    let arc = vec![vec![0.0, 0.0], vec![1.0, 0.5], vec![2.0, 0.5], vec![3.0, 0.0]];
    let zigzag = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, -1.0], vec![3.0, 1.0]];

    let s_line = builder.encode(&line);
    let s_arc = builder.encode(&arc);
    let s_zigzag = builder.encode(&zigzag);

    println!("Randomized signature demo — 3 reference paths in ℝ²");
    println!("Carrier dim: {}", s_line.dim());
    println!();
    println!("Self-similarity (sanity check, all should be 1.0):");
    println!("  line   ⋅ line   = {:.6}", s_line.cosine(&s_line));
    println!("  arc    ⋅ arc    = {:.6}", s_arc.cosine(&s_arc));
    println!("  zigzag ⋅ zigzag = {:.6}", s_zigzag.cosine(&s_zigzag));
    println!();
    println!("Cross-similarity:");
    println!("  line   ⋅ arc    = {:.6}", s_line.cosine(&s_arc));
    println!("  line   ⋅ zigzag = {:.6}", s_line.cosine(&s_zigzag));
    println!("  arc    ⋅ zigzag = {:.6}", s_arc.cosine(&s_zigzag));
    println!();

    // Reparametrization invariance — the universal-approximation theorem implies
    // that subdividing a path linearly should yield very similar signatures.
    let line_subdivided = vec![
        vec![0.0, 0.0],
        vec![0.5, 0.0],
        vec![1.0, 0.0],
        vec![1.5, 0.0],
        vec![2.0, 0.0],
        vec![2.5, 0.0],
        vec![3.0, 0.0],
    ];
    let s_line_sub = builder.encode(&line_subdivided);
    println!("Reparametrization (line vs subdivided line):");
    println!("  cosine = {:.6}  (should be very close to 1.0)", s_line.cosine(&s_line_sub));
    println!();

    // Determinism across builder instantiations with the same seed.
    let builder2 = RandomizedSignatureBuilder::new(2, 64, 0xADA_F00D);
    let s_line_b2 = builder2.encode(&line);
    let max_diff: f64 = s_line
        .state
        .iter()
        .zip(s_line_b2.state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("Determinism check (same seed, different builder instance):");
    println!("  max |Δ| across all {} dims = {:.2e}", s_line.dim(), max_diff);
}
