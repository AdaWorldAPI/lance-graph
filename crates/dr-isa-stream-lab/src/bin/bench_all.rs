//! Every arm this lab tests, on one host, in one table. Each result is
//! checked against its oracle before timing and on every timed run; a
//! mismatch aborts the run. Prints markdown (the committed `BENCH.md`).
//!
//! Warm medians of `RUNS` runs. "cold" rows are one-off preparation.
//! "reference" rows are kept to show where cost goes; they are not gates.

use std::hint::black_box;
use std::time::Instant;

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::cube::{fold, WhereMasks};
use dr_isa_stream_lab::fixture::{bf16_value, Fixture, FAR_EVERY, MASKS};
use dr_isa_stream_lab::gemm::{sums, sums_whole, Mode, SelectionB};
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use dr_isa_stream_lab::spmm::{aggregate, aggregate_cblas};
use dr_isa_stream_lab::vcsr::{ExplicitCsr, VirtualCsr};
use dr_isa_stream_lab::{oracle, spmm};

const RUNS: usize = 21;
const N: usize = 65_536;

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

fn row(section: &str, arm: &str, ns: u128, note: &str) {
    let (v, unit) = if ns < 10_000 {
        (ns as f64, "ns")
    } else if ns < 10_000_000 {
        (ns as f64 / 1e3, "µs")
    } else {
        (ns as f64 / 1e6, "ms")
    };
    println!("| {section} | {arm} | {v:.1} {unit} | {note} |");
}

fn main() {
    println!(
        "Host tile tier: `{}`. n = {N}, 16 lanes, {RUNS} warm runs (median).\n",
        ndarray::simd::bf16_tile_gemm_tier()
    );
    println!("| Workload | Arm | Time | Notes |");
    println!("|---|---|---|---|");

    let fx = Fixture {
        n: N,
        props: 16,
        seed: 0xA11,
    };
    let fields: Vec<u8> = (0..16).collect();
    let slab = Bf16Lanes::slab_in_order(&fx).unwrap();
    let scattered = Bf16Lanes::scattered(&fx).unwrap();
    let world = World::new();
    let s_slab = Socket::in_order(&slab, 16);
    let s_scat = Socket::in_order(&scattered, 16);
    let binding = world.bind(&s_slab, &fields, SPACE_A).unwrap();
    let b_scat = world.bind(&s_scat, &fields, SPACE_A).unwrap();

    // ── WHERE: SUM(lane p) WHERE mask m, 16 x 16 ──────────────────────────
    let w = "WHERE 16×16";
    let bind_ns = median_ns(|| {
        black_box(world.bind(&s_slab, &fields, SPACE_A).unwrap());
    });
    row(w, "bind (cold)", bind_ns, "DT64 lane binding, 16 fields");
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let want = oracle::sums(&fx, &(0..16).collect::<Vec<_>>());
    let want_f: Vec<[f32; MASKS]> = want.iter().map(|r| r.map(|v| v as f32)).collect();
    let counts_ns = median_ns(|| {
        black_box(masks.counts());
    });
    row(
        w,
        "16 WHERE counts (popcount)",
        counts_ns,
        "the cube's fold terminal",
    );
    let stage_ns = median_ns(|| {
        black_box(SelectionB::stage(&masks));
    });
    row(
        w,
        "stage selection B (cold)",
        stage_ns,
        "once per mask set, 2 MiB",
    );
    let sel = SelectionB::stage(&masks);
    let whole = |b, s: &Socket<'_>, mode| {
        let (y, r) = sums_whole(&world, b, s, &sel, &fields, mode).unwrap();
        assert_eq!(y, want_f);
        r
    };
    let r = whole(&binding, &s_slab, Mode::Direct);
    let ns = median_ns(|| {
        black_box(whole(&binding, &s_slab, Mode::Direct));
    });
    row(
        w,
        "tile GEMM, whole lane, A in place",
        ns,
        &format!("{} call, A staged {} B", r.gemms, r.a_staged_bytes),
    );
    let ns = median_ns(|| {
        black_box(whole(&binding, &s_slab, Mode::Staged));
    });
    row(
        w,
        "tile GEMM, whole lane, A staged (slab)",
        ns,
        "A copied once",
    );
    let ns = median_ns(|| {
        black_box(whole(&b_scat, &s_scat, Mode::Staged));
    });
    row(
        w,
        "tile GEMM, whole lane, A staged (scattered)",
        ns,
        "lanes in separate allocations",
    );

    // ndarray's CBLAS-compatible GEMM over the whole universe in one call.
    // At n = 65,536 the padded pitch equals K, so the slab is already a dense
    // 16 x 65,536 row-major matrix and is passed in place.
    let a_rows = slab.row(0).unwrap();
    assert_eq!(a_rows.pitch, N, "pitch == K: the slab is a dense A");
    let a_dense = &a_rows.slab[..16 * N];
    let mut b_rm = vec![0u16; N * MASKS];
    for m in 0..MASKS {
        for (wi, &word) in masks.words(m).iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                b_rm[(wi * 64 + bits.trailing_zeros() as usize) * MASKS + m] = 0x3F80;
                bits &= bits - 1;
            }
        }
    }
    let cblas = || {
        let mut c = vec![0f32; 16 * MASKS];
        ndarray::backend::cblas_gemm_bf16bf16f32(a_dense, &b_rm, &mut c, 16, MASKS, N);
        let y: Vec<[f32; MASKS]> = c.chunks(MASKS).map(|r| r.try_into().unwrap()).collect();
        assert_eq!(y, want_f);
        y
    };
    let ns = median_ns(|| {
        black_box(cblas());
    });
    row(
        w,
        "ndarray cblas_gemm_bf16bf16f32, whole 64K",
        ns,
        "1 call; A is the slab (pitch = K); B row-major",
    );
    let ns = median_ns(|| {
        let (y, _) = sums(&world, &binding, &s_slab, &masks, &fields, Mode::Direct).unwrap();
        assert_eq!(y, want_f);
    });
    row(
        w,
        "reference: tile GEMM per 32-identity block",
        ns,
        "2,048 calls, B restaged per block",
    );
    let ns = median_ns(|| {
        assert_eq!(
            fold(&world, &binding, &s_slab, &masks, &fields).unwrap(),
            want
        );
    });
    row(w, "reference: cube per-bit sum fold", ns, "scalar gather");

    // ── Graph: Y[p][i] = sum over row(i) of lane_p[j] ─────────────────────
    let local = |far_every: usize| -> Vec<(usize, usize)> {
        (0..N)
            .flat_map(|i| fx.neighbors_with(i, far_every).map(move |j| (i, j)))
            .collect()
    };
    let community: Vec<(usize, usize)> = (0..N)
        .flat_map(|i| {
            let base = i - i % 256;
            (0..64usize).map(move |k| (i, base + (i * 7 + k * 4 + k / 64) % 256))
        })
        .collect();
    let lane_vals: Vec<&[u16]> = (0..16).map(|p| slab.row(p).unwrap().values()).collect();
    for (label, edges) in [
        (format!("graph deg 8, 1/{FAR_EVERY} far"), local(FAR_EVERY)),
        ("graph deg 8, all far".to_string(), local(1)),
        ("graph deg 64, own tile".to_string(), community),
    ] {
        let g = VirtualCsr::from_edges(SPACE_A, N, edges.iter().copied());
        let csr = ExplicitCsr::from_edges(N, edges.iter().copied());
        let want: Vec<Vec<f32>> = spmm::oracle(&fx, &csr, &(0..16).collect::<Vec<_>>())
            .into_iter()
            .map(|r| r.into_iter().map(|v| v as f32).collect())
            .collect();
        let shape = format!(
            "nnz {}, {} blocks ({:.1}% of tile pairs)",
            g.nnz(),
            g.blocks(),
            100.0 * g.blocks() as f64 / 65_536.0
        );
        let ns = median_ns(|| {
            black_box(VirtualCsr::from_edges(SPACE_A, N, edges.iter().copied()));
        });
        row(&label, "build virtual CSR (cold)", ns, &shape);
        let tile = |mode| {
            let (y, r) = aggregate(&world, &binding, &s_slab, &g, &fields, mode).unwrap();
            assert_eq!(y, want);
            r
        };
        let r = tile(Mode::Direct);
        let ns = median_ns(|| {
            black_box(tile(Mode::Direct));
        });
        row(
            &label,
            "tile GEMM, A in place",
            ns,
            &format!("{} calls", r.gemms),
        );
        let ns = median_ns(|| {
            black_box(tile(Mode::Staged));
        });
        row(&label, "tile GEMM, A staged", ns, "");
        let nat = || {
            let (y, r) = aggregate_cblas(&world, &binding, &s_slab, &g, &fields).unwrap();
            assert_eq!(y, want);
            r
        };
        let r = nat();
        let ns = median_ns(|| {
            black_box(nat());
        });
        row(
            &label,
            "ndarray cblas_gemm_bf16bf16f32",
            ns,
            &format!("{} calls, A staged (no lda)", r.gemms),
        );
        let ns = median_ns(|| {
            let mut y = vec![vec![0f32; N]; 16];
            for i in 0..N {
                for &j in csr.row(i) {
                    for (p, yp) in y.iter_mut().enumerate() {
                        yp[i] += bf16_value(lane_vals[p][j]);
                    }
                }
            }
            assert_eq!(y, want);
        });
        row(
            &label,
            "scalar CSR over the same lanes",
            ns,
            "non-GEMM baseline",
        );
    }
}
