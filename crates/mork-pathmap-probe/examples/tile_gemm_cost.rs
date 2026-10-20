//! D-RMAP-2 tile arm: Boolean composition and triangle counts on one
//! 256 x 256 focus map, bit matrix vs BF16 16x16 tile GEMM.
//!
//! Run: `cd crates/mork-pathmap-probe && cargo run --release --example tile_gemm_cost`.
//! Release, warm caches, medians of `REPS`. Every timed answer is checked
//! against the triple-loop oracle first.

use std::hint::black_box;
use std::time::{Duration, Instant};

use mork_pathmap_probe::tile_gemm::{
    Bf16Staged, BitMat, N, fixture, oracle_square, oracle_triangles, threshold, tier,
    triangles_from_square,
};

const REPS: usize = 7;

fn time<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut ts = Vec::with_capacity(REPS);
    let mut last = None;
    for _ in 0..REPS {
        let t = Instant::now();
        let v = black_box(f());
        ts.push(t.elapsed());
        last = Some(v);
    }
    ts.sort();
    (ts[REPS / 2], last.unwrap())
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn main() {
    println!("bf16 tier: {} | release, medians of {REPS}", tier());
    println!(
        "case, edges, triangles, transpose_us, bit_compose_us, bit_counts_us, bit_triangles_us, bf16_stage_us, bf16_square_packed_us, bf16_square_rowmajor_us, threshold_us, bf16_triangles_us, bit_bytes, bf16_bytes"
    );
    for (label, p, hubs, seed) in [
        ("1%", 10u32, 0usize, 1u64),
        ("10%", 100, 0, 2),
        ("50%", 500, 0, 3),
        ("1% + 4 hubs", 10, 4, 4),
    ] {
        let a = fixture(p, hubs, seed);
        let want_sq = oracle_square(&a);
        let want_t = oracle_triangles(&a);

        let (t_tr, at) = time(|| black_box(&a).transpose());
        let (t_bc, c) = time(|| black_box(&a).compose(black_box(&at), black_box(&a)));
        let mut want_bits = BitMat::empty();
        for i in 0..N {
            for j in 0..N {
                if want_sq[i * N + j] > 0 {
                    want_bits.set(i, j);
                }
            }
        }
        assert_eq!(c, want_bits);
        let (t_bq, bq) = time(|| black_box(&a).square_counts(black_box(&at)));
        assert_eq!(bq, want_sq);
        let (t_bt, tri) = time(|| black_box(&a).triangles(black_box(&at)));
        assert_eq!(tri, want_t);

        let (t_st, staged) = time(|| Bf16Staged::stage(black_box(&a)));
        let mut sq = vec![0f32; N * N];
        let (t_sq, _) = time(|| {
            black_box(&staged).square(black_box(&mut sq));
        });
        assert!(sq.iter().zip(&want_sq).all(|(g, &w)| *g == w as f32));
        let mut sq2 = vec![0f32; N * N];
        let (t_sq2, _) = time(|| {
            black_box(&staged).square_rowmajor(black_box(&mut sq2));
        });
        assert_eq!(sq2, sq);
        let (t_th, cb) = time(|| threshold(black_box(&sq)));
        assert_eq!(cb, want_bits);
        let (t_gt, tri2) = time(|| triangles_from_square(black_box(&a), black_box(&sq)));
        assert_eq!(tri2, want_t);

        println!(
            "{label}, {}, {want_t}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {}, {}",
            a.edges(),
            us(t_tr),
            us(t_bc),
            us(t_bq),
            us(t_bt),
            us(t_st),
            us(t_sq),
            us(t_sq2),
            us(t_th),
            us(t_gt),
            2 * N * N / 8,
            staged.bytes()
        );
    }
}
