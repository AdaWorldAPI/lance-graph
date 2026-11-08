//! Neighbor aggregation over the virtual CSR: stage costs per realization.
//! Every timed result is checked against the CSR oracle first and on every
//! timed run. Units: us per batch.
//!
//! With `--features mkl` the MKL arm runs too; set
//! `MKL_THREADING_LAYER=SEQUENTIAL` so it is single-threaded like the rest.

use std::hint::black_box;
use std::time::Instant;

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::fixture::{bf16_value, Fixture, FAR_EVERY};
use dr_isa_stream_lab::gemm::Mode;
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use dr_isa_stream_lab::spmm::{aggregate, oracle};
use dr_isa_stream_lab::vcsr::{ExplicitCsr, VirtualCsr};

const RUNS: usize = 21;

fn median_ns(mut f: impl FnMut()) -> u128 {
    f();
    let mut t: Vec<u128> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos()
        })
        .collect();
    t.sort_unstable();
    t[RUNS / 2]
}

fn us(ns: u128) -> f64 {
    ns as f64 / 1000.0
}

fn main() {
    println!("tile tier: {}", ndarray::simd::bf16_tile_gemm_tier());
    let n = 65_536;
    let fx = Fixture {
        n,
        props: 16,
        seed: 0x5C5,
    };
    let fields: Vec<u8> = (0..16).collect();
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();

    // (label, edges): the default local graph, the same degree with a
    // long-range edge everywhere, and a community graph whose edges stay in
    // the identity's own tile at degree 64 (25% block density).
    let local = |far_every: usize| -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|i| fx.neighbors_with(i, far_every).map(move |j| (i, j)))
            .collect()
    };
    let community: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| {
            let base = i - i % 256;
            (0..64usize).map(move |k| (i, base + (i * 7 + k * 4 + k / 64) % 256))
        })
        .collect();
    let cases = [
        (format!("local, far_every={FAR_EVERY}"), local(FAR_EVERY)),
        ("local, far_every=1".to_string(), local(1)),
        ("community, degree 64 in own tile".to_string(), community),
    ];
    for (label, edges) in cases {
        let build_ns = median_ns(|| {
            black_box(VirtualCsr::from_edges(SPACE_A, n, edges.iter().copied()));
        });
        let g = VirtualCsr::from_edges(SPACE_A, n, edges.iter().copied());
        let csr = ExplicitCsr::from_edges(n, edges.iter().copied());
        let want: Vec<Vec<f32>> = oracle(&fx, &csr, &(0..16).collect::<Vec<_>>())
            .into_iter()
            .map(|r| r.into_iter().map(|v| v as f32).collect())
            .collect();

        let run = |mode| {
            let (y, r) = aggregate(&world, &binding, &socket, &g, &fields, mode).unwrap();
            assert_eq!(y, want, "{mode:?}: wrong answer");
            r
        };
        let rd = run(Mode::Direct);
        let direct_ns = median_ns(|| {
            black_box(run(Mode::Direct));
        });
        let rs = run(Mode::Staged);
        let staged_ns = median_ns(|| {
            black_box(run(Mode::Staged));
        });

        // Like-for-like scalar baseline: the explicit CSR over the same lanes.
        let lane_vals: Vec<&[u16]> = (0..16).map(|p| lanes.row(p).unwrap().values()).collect();
        let scalar = || {
            let mut y = vec![vec![0f32; n]; 16];
            for i in 0..n {
                for &j in csr.row(i) {
                    for (p, yp) in y.iter_mut().enumerate() {
                        yp[i] += bf16_value(lane_vals[p][j]);
                    }
                }
            }
            assert_eq!(y, want, "scalar CSR: wrong answer");
            y
        };
        let scalar_ns = median_ns(|| {
            black_box(scalar());
        });

        println!(
            "\n{label}: nnz={} blocks={} ({:.1}% of tile pairs) block bytes={}",
            g.nnz(),
            g.blocks(),
            100.0 * g.blocks() as f64 / 65_536.0,
            g.bytes()
        );
        println!("  build virtual CSR (cold)        {:>9.1} us", us(build_ns));
        println!(
            "  tile GEMM, A in place           {:>9.1} us  gemms={} A-staged={} B",
            us(direct_ns),
            rd.gemms,
            rd.a_staged_bytes
        );
        println!(
            "  tile GEMM, A staged             {:>9.1} us  gemms={} A-staged={} B",
            us(staged_ns),
            rs.gemms,
            rs.a_staged_bytes
        );
        #[cfg(feature = "mkl")]
        {
            use dr_isa_stream_lab::spmm::aggregate_mkl;
            let mkl = || {
                let (y, r) = aggregate_mkl(&world, &binding, &socket, &g, &fields).unwrap();
                assert_eq!(y, want, "MKL: wrong answer");
                r
            };
            let rm = mkl();
            let mkl_ns = median_ns(|| {
                black_box(mkl());
            });
            println!(
                "  MKL bf16 GEMM, A in place       {:>9.1} us  gemms={}",
                us(mkl_ns),
                rm.gemms
            );
        }
        #[cfg(not(feature = "mkl"))]
        println!("  MKL bf16 GEMM                   not built (enable --features mkl)");
        println!(
            "  scalar CSR over the same lanes  {:>9.1} us",
            us(scalar_ns)
        );
    }
}
