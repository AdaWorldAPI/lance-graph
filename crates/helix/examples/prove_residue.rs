//! The probe (workspace convention: the example IS the benchmark/proof).
//!
//! Run: `cargo run --manifest-path crates/helix/Cargo.toml --example prove_residue`
use helix::{prove, DistanceLut, ResidueEncoder};

fn main() {
    // 1. The 2-D golden-spiral hemisphere discrepancy proof (Open Item #1).
    let proof = prove();
    proof.report();

    // 2. Demo: encode residues at one place, show metric-safe L1 distances.
    let enc = ResidueEncoder::new(4096);
    let lut = DistanceLut::linear();
    let place = 0x1234u64;
    let base = enc.encode(place, 1700);
    println!("\nresidue edges at place 0x{place:x} (N=4096), L1 from n=1700:");
    for n in [1700usize, 1701, 1750, 2500, 4000] {
        let e = enc.encode(place, n);
        let d = base.distance_adaptive(&e, &lut);
        println!("  n={n:<4} -> edge {:?}  L1={d}", e.to_bytes());
    }
}
