//! Stage costs for one binding feeding both arms. Every timed result is
//! checked against the fixture oracle; a mismatch aborts the run, because a
//! fast wrong answer is not a measurement (AMX tile state can corrupt under
//! host contention, ndarray `.claude/AMX_GOTCHAS.md` Gotcha 14).
//!
//! Units: ns per operation, us per batch. Rows marked "reference" are
//! oracle-grade paths kept for comparison; they are not gates.

use std::hint::black_box;
use std::time::Instant;

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::cube::{fold, WhereMasks};
use dr_isa_stream_lab::fixture::{Fixture, MASKS};
use dr_isa_stream_lab::gemm::{sums, sums_whole, Mode, SelectionB};
use dr_isa_stream_lab::oracle;
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};

const RUNS: usize = 41;

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
    println!("tier: {}", ndarray::simd::bf16_tile_gemm_tier());
    for (n, props) in [(65_536usize, 16usize), (65_536, 32)] {
        let fx = Fixture {
            n,
            props,
            seed: 0xBE7C,
        };
        let fields: Vec<u8> = (0..props as u8).collect();
        let want = oracle::sums(&fx, &(0..props).collect::<Vec<_>>());
        let want_f: Vec<[f32; MASKS]> = want.iter().map(|r| r.map(|v| v as f32)).collect();

        let slab = Bf16Lanes::slab_in_order(&fx).unwrap();
        let scattered = Bf16Lanes::scattered(&fx).unwrap();
        let world = World::new();
        let s_slab = Socket::in_order(&slab, props);
        let s_scat = Socket::in_order(&scattered, props);

        let bind_ns = median_ns(|| {
            black_box(world.bind(&s_slab, &fields, SPACE_A).unwrap());
        });
        let binding = world.bind(&s_slab, &fields, SPACE_A).unwrap();
        let b_scat = world.bind(&s_scat, &fields, SPACE_A).unwrap();
        let masks = WhereMasks::from_fixture(&fx, SPACE_A);
        let counts_ns = median_ns(|| {
            black_box(masks.counts());
        });
        let stage_ns = median_ns(|| {
            black_box(SelectionB::stage(&masks));
        });
        let sel = SelectionB::stage(&masks);

        let check = |label: &str, run: &dyn Fn() -> (Vec<[f32; MASKS]>, usize, usize)| {
            let (got, gemms, staged) = run();
            assert_eq!(got, want_f, "{label}: wrong answer");
            let ns = median_ns(|| {
                let (got, _, _) = run();
                assert_eq!(got, want_f, "{label}: wrong answer while timing");
                black_box(got);
            });
            (ns, gemms, staged)
        };
        let whole = |b: &_, s: &Socket<'_>, mode| {
            let (got, r) = sums_whole(&world, b, s, &sel, &fields, mode).unwrap();
            (got, r.gemms, r.a_staged_bytes)
        };
        let blocked = |mode| {
            let (got, r) = sums(&world, &binding, &s_slab, &masks, &fields, mode).unwrap();
            (got, r.gemms, r.a_staged_bytes)
        };
        let wd = check("whole direct", &|| whole(&binding, &s_slab, Mode::Direct));
        let ws = check("whole staged slab", &|| {
            whole(&binding, &s_slab, Mode::Staged)
        });
        let wc = check("whole staged scattered", &|| {
            whole(&b_scat, &s_scat, Mode::Staged)
        });
        let bd = check("blocked direct", &|| blocked(Mode::Direct));

        assert_eq!(
            fold(&world, &binding, &s_slab, &masks, &fields).unwrap(),
            want
        );
        let fold_ns = median_ns(|| {
            black_box(fold(&world, &binding, &s_slab, &masks, &fields).unwrap());
        });

        println!(
            "\nn={n} props={props} masks={MASKS} selected={} lanes={} B selection={} B",
            masks.counts().iter().sum::<u32>(),
            slab.resident_bytes(),
            sel.bytes()
        );
        println!("  bind, cold, once                     {bind_ns:>9} ns");
        println!(
            "  ARM A: 16 WHERE counts (popcount)    {:>9.1} us",
            us(counts_ns)
        );
        println!(
            "  stage selection B, cold, per masks   {:>9.1} us",
            us(stage_ns)
        );
        println!(
            "  ARM B whole lane, A in place         {:>9.1} us  gemms={} A-staged={} B",
            us(wd.0),
            wd.1,
            wd.2
        );
        println!(
            "  ARM B whole lane, A staged (slab)    {:>9.1} us  gemms={} A-staged={} B",
            us(ws.0),
            ws.1,
            ws.2
        );
        println!(
            "  ARM B whole lane, A staged (scatter) {:>9.1} us  gemms={} A-staged={} B",
            us(wc.0),
            wc.1,
            wc.2
        );
        println!(
            "  reference: ARM B per-32-block calls  {:>9.1} us  gemms={} (B restaged per block)",
            us(bd.0),
            bd.1
        );
        println!(
            "  reference: ARM A per-bit sum fold    {:>9.1} us",
            us(fold_ns)
        );
    }
}
