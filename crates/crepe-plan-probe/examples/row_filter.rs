//! The row filter at preparation, two ways, over the same rows: the scalar
//! test (`binary_search` in a sorted needle set) and `ndarray::simd`'s strided
//! compare, one pass per needle OR-ed into a mask, over the row bytes and
//! over a contiguous column (what a column-major store would hand it).
//! Release, medians of 31. Not wired into `prepare`: over the row-major store
//! the strided pass is slower than the scalar test at every needle count
//! (plan §2c).
//!
//! ```text
//! cargo run --release --example row_filter
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use mork_pathmap_probe::join::for_each_set;
use ndarray::simd::{eq_u32_strided_to_mask, eq_u32_to_mask, mask_or_assign};

/// The rows' bytes, borrowed: `[u32; 2]` has no padding and `u8` has no
/// alignment requirement, so the view covers exactly the rows' memory.
fn row_bytes(rows: &[[u32; 2]]) -> &[u8] {
    // SAFETY: `rows` is a live, initialized slice of `[u32; 2]`; its
    // `size_of_val` bytes are readable for the borrow's lifetime, and every
    // bit pattern is a valid `u8`.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) }
}

// `eq_u32_strided_to_mask` reads each field as little-endian.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "row_bytes hands native-endian u32 bytes to a little-endian kernel"
);

/// Mask over `rows` of the rows whose column `col` is one of `needles`: one
/// `ndarray::simd` strided compare per needle (16 lanes per step), OR-ed
/// into the mask. `needles` empty gives the empty mask.
fn column_in_mask(rows: &[[u32; 2]], col: usize, needles: &[u32]) -> Vec<u64> {
    let n = rows.len();
    let words = n.div_ceil(64);
    let mut mask = vec![0u64; words];
    let mut hit = vec![0u64; words];
    let bytes = row_bytes(rows);
    for &v in needles {
        eq_u32_strided_to_mask(bytes, col * 4, 8, n, v, &mut hit);
        mask_or_assign(&mut mask, &hit);
    }
    mask
}

/// The rows a mask selects, in row order.
fn rows_under(rows: &[[u32; 2]], mask: &[u64]) -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    for_each_set(mask, rows.len(), |i| out.push(rows[i]));
    out
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> Duration {
    let mut v = Vec::with_capacity(31);
    for _ in 0..31 {
        let t = Instant::now();
        black_box(f());
        v.push(t.elapsed());
    }
    median(v)
}

/// The strided filter equals the scalar one on row counts around word
/// boundaries, with repeated and absent needles.
fn check_boundaries() {
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 40) as u32
    };
    for n in [0usize, 1, 15, 16, 17, 63, 64, 65, 200] {
        let rows: Vec<[u32; 2]> = (0..n).map(|_| [next(), next()]).collect();
        for needles in [
            vec![],
            vec![3],
            vec![3, 3, 7],
            vec![1000],
            (0..40).collect(),
        ] {
            for col in [0, 1] {
                let got = rows_under(&rows, &column_in_mask(&rows, col, &needles));
                let want: Vec<[u32; 2]> = rows
                    .iter()
                    .copied()
                    .filter(|r| needles.contains(&r[col]))
                    .collect();
                assert_eq!(got, want, "n={n} col={col} needles={needles:?}");
            }
        }
    }
}

fn main() {
    check_boundaries();
    println!("n,needles,kept,scalar_us,ndarray_us,compare_us,collect_us,contiguous_us");
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) as u32
    };
    for n in [1_000usize, 50_000] {
        let rows: Vec<[u32; 2]> = (0..n).map(|_| [next() % 20_000, next()]).collect();
        for k in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let mut needles: Vec<u32> = (0..k as u32).map(|i| i * 97 % 20_000).collect();
            needles.sort_unstable();
            needles.dedup();
            let scalar = || -> Vec<[u32; 2]> {
                rows.iter()
                    .copied()
                    .filter(|r| needles.binary_search(&r[0]).is_ok())
                    .collect()
            };
            let simd = || rows_under(&rows, &column_in_mask(&rows, 0, &needles));
            assert_eq!(scalar(), simd());
            let mask = column_in_mask(&rows, 0, &needles);
            // What a column-major store would give: the same compares over a
            // contiguous u32 column (built once here, outside the timing).
            let column: Vec<u32> = rows.iter().map(|r| r[0]).collect();
            let contiguous = || {
                let mut m = vec![0u64; n.div_ceil(64)];
                let mut hit = vec![0u64; n.div_ceil(64)];
                for &v in &needles {
                    eq_u32_to_mask(&column, v, &mut hit);
                    mask_or_assign(&mut m, &hit);
                }
                m
            };
            assert_eq!(contiguous(), mask);
            println!(
                "{n},{k},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                scalar().len(),
                time(scalar).as_secs_f64() * 1e6,
                time(simd).as_secs_f64() * 1e6,
                time(|| column_in_mask(&rows, 0, &needles)).as_secs_f64() * 1e6,
                time(|| rows_under(&rows, &mask)).as_secs_f64() * 1e6,
                time(contiguous).as_secs_f64() * 1e6,
            );
        }
    }
}
