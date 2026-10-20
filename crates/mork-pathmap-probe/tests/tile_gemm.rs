//! The D-RMAP-2 tile arm against a triple-loop oracle.

use mork_pathmap_probe::tile_gemm::{
    Bf16Staged, BitMat, N, fixture, oracle_square, oracle_triangles, threshold,
    triangles_from_square,
};

const CASES: [(u32, usize, u64); 4] = [(10, 0, 1), (100, 0, 2), (500, 0, 3), (10, 4, 4)];

#[test]
fn bit_and_bf16_composition_equal_the_oracle() {
    for (p, hubs, seed) in CASES {
        let a = fixture(p, hubs, seed);
        let at = a.transpose();
        let want = oracle_square(&a);
        let mut want_bits = BitMat::empty();
        for i in 0..N {
            for j in 0..N {
                if want[i * N + j] > 0 {
                    want_bits.set(i, j);
                }
            }
        }
        assert_eq!(a.compose(&at, &a), want_bits, "bit arm p={p} hubs={hubs}");
        assert_eq!(a.square_counts(&at), want, "bit counts p={p} hubs={hubs}");
        let staged = Bf16Staged::stage(&a);
        let mut sq = vec![0f32; N * N];
        staged.square(&mut sq);
        // Exact walk counts, not only their support.
        for (got, &w) in sq.iter().zip(&want) {
            assert_eq!(*got, w as f32, "bf16 count p={p} hubs={hubs}");
        }
        assert_eq!(
            threshold(&sq),
            want_bits,
            "bf16 threshold p={p} hubs={hubs}"
        );
        // Not degenerate: two hops reach something one hop does not.
        assert_ne!(want_bits, a, "p={p}");
    }
}

#[test]
fn triangle_counts_agree_across_arms() {
    for (p, hubs, seed) in CASES {
        let a = fixture(p, hubs, seed);
        let at = a.transpose();
        let want = oracle_triangles(&a);
        assert_eq!(a.triangles(&at), want, "bit arm p={p} hubs={hubs}");
        let mut sq = vec![0f32; N * N];
        Bf16Staged::stage(&a).square(&mut sq);
        assert_eq!(
            triangles_from_square(&a, &sq),
            want,
            "bf16 arm p={p} hubs={hubs}"
        );
    }
    // Silence twin: a DAG (edges only i < j) has no directed cycle.
    let dag = BitMat::from_edges(
        (0..N).flat_map(|i| (i + 1..(i + 4).min(N)).map(move |j| (i as u8, j as u8))),
    );
    assert!(dag.edges() > 0);
    assert_eq!(dag.triangles(&dag.transpose()), 0);
    assert_eq!(oracle_triangles(&dag), 0);
}

/// The counts carry multiplicity the Boolean result drops: some pair is
/// reached by more than one walk, so a threshold is a real loss of
/// information, not a no-op.
#[test]
fn walk_counts_exceed_one_where_support_does_not_show_it() {
    let a = fixture(100, 0, 2);
    let mut sq = vec![0f32; N * N];
    Bf16Staged::stage(&a).square(&mut sq);
    assert!(sq.iter().any(|&c| c > 1.0));
}

/// An asymmetric relation, so a composition that forgets the transpose is
/// caught.
#[test]
fn composition_respects_direction() {
    let a = BitMat::from_edges([(0, 1), (1, 2)]);
    let c = a.compose(&a.transpose(), &a);
    assert!(c.get(0, 2));
    assert!(!c.get(2, 0));
    assert_eq!(c.edges(), 1);
}

/// The reported BF16 size counts every buffer the staged object keeps: the
/// 256 × 256 matrix, the packed B blocks and the row-major B blocks, each
/// 256 × 256 BF16 values.
#[test]
fn bf16_bytes_count_every_retained_buffer() {
    let staged = Bf16Staged::stage(&BitMat::from_edges([(0, 1), (1, 2)]));
    assert_eq!(staged.bytes(), 3 * 256 * 256 * 2);
}
