//! requantize_i32_to_i8 (ndarray src/simd_fixedpoint.rs:237-284) over lengths that exercise the scalar tail
//! (len % 16 != 0) and mixed full blocks, per-tensor and per-channel, against an element-wise reference
//! built from multiply_by_quantized_multiplier_i32 + output offset + clamp (TFLM order).
use ndarray::simd::{multiply_by_quantized_multiplier_i32, requantize_i32_to_i8};
fn main() {
    let mut s: u64 = 0x243F6A8885A308D3;
    let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let (mut cells, mut bad, mut cases) = (0u64, 0u64, 0u64);
    for &len in &[1usize, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257] {
        for &ch in &[1usize, 3, 5, 16] {
            if len % ch != 0 { continue; }
            for rep in 0..200 {
                let acc: Vec<i32> = (0..len).map(|i| match (rep + i) % 9 {
                    0 => i32::MIN, 1 => i32::MAX, 2 => 0, 3 => -1, _ => next() as i32 >> (next() % 24) }).collect();
                let mult: Vec<i32> = (0..ch).map(|_| (1 << 30) + (next() % (1 << 30)) as i32).collect();
                let shift: Vec<i32> = (0..ch).map(|_| -((next() % 12) as i32)).collect();
                let (off, lo, hi) = ((next() % 256) as i32 - 128, -128, 127);
                let mut out = vec![0i8; len];
                requantize_i32_to_i8(&acc, &mult, &shift, off, lo, hi, &mut out);
                for i in 0..len {
                    let c = i % ch;
                    let want = multiply_by_quantized_multiplier_i32(acc[i], mult[c], shift[c]).wrapping_add(off).clamp(lo, hi) as i8;
                    cells += 1; if out[i] != want { bad += 1; }
                }
                cases += 1;
            }
        }
    }
    println!("REQUANT_TAIL cases={cases} cells={cells} mismatches={bad}");
}
