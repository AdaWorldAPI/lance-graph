//! Diagnostic for the D-PFP-1 run-1 classification (harness defect vs data).
//! Uses only the pre-registered constants; prints, per stimulus, the energy
//! mass, the max, how many rows clear THETA, and the top-3 after `think`.
//! Not part of the verdict.

use perturbationsfeld_probe::helpers::SplitMix64;
use perturbationsfeld_probe::prereg::*;
use perturbationsfeld_probe::{fire, JINA_V5};
use thinking_engine::codebook_index::CodebookIndex;
use thinking_engine::engine::ThinkingEngine;
fn main() {
    let lens = JINA_V5;
    let mut eng = ThinkingEngine::new(lens.table.to_vec());
    println!("size {} floor {}", eng.size, eng.floor);
    let idx: Vec<u16> = lens
        .index
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let vocab = idx.len();
    let cb = CodebookIndex::new(idx, 256, "x".into());
    let mut rng = SplitMix64(STIMULUS_SEED);
    for s in 0..4 {
        let toks: Vec<u32> = (0..M).map(|_| rng.below(vocab as u64) as u32).collect();
        let ids = cb.lookup_many(&toks);
        eng.reset();
        eng.perturb(&ids);
        let pre: f32 = eng.energy.iter().sum();
        eng.cycle();
        let one: f32 = eng.energy.iter().sum();
        let p = fire(&mut eng, &ids);
        let sum: f32 = p.energy.iter().sum();
        let nz = p.energy.iter().filter(|&&e| e > 0.0).count();
        let mx = p.energy.iter().cloned().fold(0.0f32, f32::max);
        let above = p.energy.iter().filter(|&&e| e > THETA).count();
        println!("stim {s}: ids {:?} pre {pre} after1cycle {one} | think: sum {sum} nz {nz} max {mx} >θ {above} cycles {} top {:?}", ids, p.cycle_count, &p.top_k[..3]);
    }
    // row stats for one id
    let t = lens.table;
    let r = 10usize;
    let row = &t[r * 256..(r + 1) * 256];
    println!(
        "row 10: max {} diag {} count>floor {}",
        row.iter().max().unwrap(),
        row[r],
        row.iter().filter(|&&v| v > eng.floor).count()
    );
}
