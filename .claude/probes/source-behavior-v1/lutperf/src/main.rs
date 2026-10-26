//! Usage: lutperf63 ARTIFACTS_BIN
//! Applies the first SourceParity TANH table of the Amendment 6.2 artifact to 16 MiB of int8 codes:
//! (a) scalar `table[code ^ 0x80]`; (b) the same 256-entry lookup composed from ndarray's
//! `U8x64::permute_bytes` (64-entry lookup), `cmpgt_mask` and `mask_blend` — four permutes and three
//! blends per 64 codes. Asserts both outputs are identical, then reports median ns/elem of 7 runs.
use ndarray::simd::U8x64;
use std::time::Instant;

fn median_ns(mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..7).map(|_| { let a = Instant::now(); f(); a.elapsed().as_nanos() as f64 }).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[3]
}

fn main() {
    let path = std::env::args().nth(1).expect("ARTIFACTS_BIN");
    let blob = std::fs::read(path).expect("read artifact");
    let table: [u8; 256] = blob[14 + 64..14 + 64 + 256].try_into().unwrap();
    let n = 1usize << 24;
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let input: Vec<u8> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 56) as u8 }).collect();
    let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
    let t_scalar = median_ns(|| {
        for (o, &c) in a.iter_mut().zip(&input) { *o = table[(c ^ 0x80) as usize]; }
        std::hint::black_box(&a);
    });
    let blk: Vec<U8x64> = (0..4).map(|j| U8x64::from_slice(&table[64 * j..64 * j + 64])).collect();
    let (x80, t63, t127, t191) = (U8x64::splat(0x80), U8x64::splat(63), U8x64::splat(127), U8x64::splat(191));
    let t_simd = median_ns(|| {
        for (o, c) in b.chunks_exact_mut(64).zip(input.chunks_exact(64)) {
            let idx = U8x64::from_slice(c) ^ x80;
            let r0 = blk[0].permute_bytes(idx);
            let r1 = blk[1].permute_bytes(idx);
            let r2 = blk[2].permute_bytes(idx);
            let r3 = blk[3].permute_bytes(idx);
            let lo = U8x64::mask_blend(idx.cmpgt_mask(t63), r0, r1);
            let hi = U8x64::mask_blend(idx.cmpgt_mask(t191), r2, r3);
            U8x64::mask_blend(idx.cmpgt_mask(t127), lo, hi).copy_to_slice(o);
        }
        std::hint::black_box(&b);
    });
    let mism = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    println!("lut,N={n},scalar_ns_per_elem={:.3},ndarray_u8x64_ns_per_elem={:.3},mismatches={mism},vbmi_path={}",
             t_scalar / n as f64, t_simd / n as f64, std::is_x86_feature_detected!("avx512vbmi"));
    assert_eq!(mism, 0, "ndarray-composed lookup disagrees with the scalar table");
}
