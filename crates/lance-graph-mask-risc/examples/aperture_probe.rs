//! Membership masks as the Self of the row: a 12-byte V3 facet seen through an
//! APERTURE (the bytes consulted), not uniformly random bit planes.
//!
//! One 64k-row V3 tile, 16-byte records: classid (4 B) + facet (12 B = 6 rails
//! `u8:u8`). Rail 0 is the row address itself (`row >> 8 : row & 0xFF`, the
//! exact address of a 64k table); rails 1..5 are content: a coarse bucket
//! (`row >> 12`), then low-cardinality palette bytes hashed from the row.
//!
//! For each aperture (`care` bytes of `Pred::MatchFacetStrided`):
//! - the cost of computing the mask (`Count`, tiled path), per tile and per row;
//! - the mask's SHAPE: survivors and the number of contiguous runs. One run
//!   means the aperture selected an address interval, which the planner may
//!   lower to `Pred::Range` (the range fold touches no word at all);
//! - for one-run apertures, the `Range` fold's cost for the same count.
//!
//! Then the amortization: once an aperture mask is computed it is a resident
//! plane, and every later query that composes it pays only the ternlog fold
//! over resident words, not the aperture again.
//!
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example aperture_probe`

use std::time::Instant;

use lance_graph_mask_risc::{
    execute_extent, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Scratch, StridedRef, Terminal, Value,
};

const N: usize = 1 << 16;
const STRIDE: usize = 16;
const FACET_AT: usize = 4;

fn median<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        last = Some(std::hint::black_box(f()));
        ts.push(t.elapsed().as_nanos() as f64);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[reps / 2], last.expect("reps > 0"))
}

fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    (x ^ (x >> 29)).wrapping_mul(0xc4ce_b9fe_1a85_ec53)
}

fn facet(r: usize) -> [u8; 12] {
    let h = mix(r as u64);
    [
        (r >> 8) as u8,          // rail 0 hi: row address
        r as u8,                 // rail 0 lo
        (r >> 12) as u8,         // rail 1 hi: coarse bucket (16 buckets)
        (h & 0x0F) as u8,        // rail 1 lo: palette 16
        ((h >> 8) & 0x0F) as u8, // rail 2: palette 16
        ((h >> 16) & 0x03) as u8,
        ((h >> 24) & 0x07) as u8,
        (h >> 32) as u8,
        (h >> 40) as u8,
        (h >> 48) as u8,
        (h >> 56) as u8,
        ((h >> 20) & 0x01) as u8,
    ]
}

fn runs(mask: &[u64], n: usize) -> usize {
    let mut runs = 0;
    let mut prev = false;
    for r in 0..n {
        let b = mask[r / 64] >> (r % 64) & 1 == 1;
        if b && !prev {
            runs += 1;
        }
        prev = b;
    }
    runs
}

fn s(i: u16) -> Operand {
    Operand::Scratch(i)
}

fn main() {
    let mut bytes = vec![0u8; N * STRIDE];
    for r in 0..N {
        bytes[r * STRIDE..r * STRIDE + 4].copy_from_slice(&0x0902_0001u32.to_le_bytes());
        bytes[r * STRIDE + FACET_AT..r * STRIDE + 16].copy_from_slice(&facet(r));
    }
    let lanes = [LaneRef::Strided(StridedRef {
        bytes: &bytes,
        first_offset: FACET_AT,
        stride: STRIDE,
        records: N,
    })];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let probe_row = 0x4A37;
    let pat = facet(probe_row);
    let care = |idx: &[usize]| {
        let mut c = [0u8; 12];
        for &i in idx {
            c[i] = 0xFF;
        }
        c
    };
    let apertures: [(&str, [u8; 12]); 6] = [
        ("rail0.hi (address prefix)", care(&[0])),
        ("rail0 (exact address)", care(&[0, 1])),
        ("rail1.hi (coarse bucket)", care(&[2])),
        ("rail1.lo (palette, content)", care(&[3])),
        ("rail0.hi + rail1.lo", care(&[0, 3])),
        ("all 12 bytes", [0xFF; 12]),
    ];
    let reps = 201;
    println!("64k-row tile, 16-byte records; probe row {probe_row:#06x}");
    println!(
        "{:>28} {:>8} {:>6} {:>10} {:>8} {:>10}",
        "aperture", "survive", "runs", "match ns", "ns/row", "range ns"
    );
    let mut cached = Vec::new();
    for (name, c) in apertures {
        let pred = Pred::MatchFacetStrided {
            lane: 0,
            pattern: pat,
            care: c,
        };
        let op = MaskOp::Pred {
            pred,
            under: None,
            dst: 0,
        };
        let count = Program::new(vec![op], Terminal::Count { mask: s(0) });
        let keep = Program::new(vec![op], Terminal::Keep { mask: s(0) });
        let mut sc = Scratch::for_program(&count, N).expect("scratch");
        let (t, v) = median(reps, || {
            execute_extent(&count, &planes, &Foreign::NONE, &mut sc, Out::None, 0..N).unwrap()
        });
        let mut m = vec![0u64; words_for(N)];
        let mut sk = Scratch::for_program(&keep, N).expect("scratch");
        execute_extent(
            &keep,
            &planes,
            &Foreign::NONE,
            &mut sk,
            Out::Mask(&mut m),
            0..N,
        )
        .unwrap();
        let pop: usize = m.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(v, Value::Count(pop));
        let nr = runs(&m, N);
        let range = if nr == 1 {
            let lo = (0..N).find(|&r| m[r / 64] >> (r % 64) & 1 == 1).unwrap();
            let p = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::Range {
                        lo: lo as u32,
                        hi: (lo + pop) as u32,
                    },
                    under: None,
                    dst: 0,
                }],
                Terminal::Count { mask: s(0) },
            );
            let (tr, vr) = median(reps, || {
                execute_extent(
                    &p,
                    &planes,
                    &Foreign::NONE,
                    &mut Scratch::new(0, 0),
                    Out::None,
                    0..N,
                )
                .unwrap()
            });
            assert_eq!(vr, Value::Count(pop));
            format!("{tr:>10.0}")
        } else {
            format!("{:>10}", "-")
        };
        println!(
            "{name:>28} {pop:>8} {nr:>6} {t:>10.0} {:>8.2} {range}",
            t / N as f64
        );
        cached.push(m);
    }

    // Amortization: compose a cached aperture mask with two more resident
    // planes, versus recomputing the aperture inside the same program.
    let bucket = &cached[2]; // rail1.hi
    let palette = &cached[3]; // rail1.lo
    let other: Vec<u64> = (0..words_for(N)).map(|i| mix(i as u64 ^ 0xABCD)).collect();
    let masks: [&[u64]; 3] = [bucket, palette, &other];
    let hot = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    let p = |i| Operand::Plane(i);
    let chain = Program::new(
        vec![
            MaskOp::And {
                a: p(0),
                b: p(1),
                dst: 0,
            },
            MaskOp::AndNot {
                a: s(0),
                b: p(2),
                dst: 1,
            },
        ],
        Terminal::Count { mask: s(1) },
    );
    let (t_hot, v_hot) = median(reps, || {
        execute_extent(
            &chain,
            &hot,
            &Foreign::NONE,
            &mut Scratch::new(0, 0),
            Out::None,
            0..N,
        )
        .unwrap()
    });
    let pred = |c| MaskOp::Pred {
        pred: Pred::MatchFacetStrided {
            lane: 0,
            pattern: pat,
            care: c,
        },
        under: None,
        dst: 0,
    };
    let mut p1 = pred(care(&[2]));
    let mut p2 = pred(care(&[3]));
    if let MaskOp::Pred { dst, .. } = &mut p1 {
        *dst = 0;
    }
    if let MaskOp::Pred { dst, .. } = &mut p2 {
        *dst = 1;
    }
    let cold = Program::new(
        vec![
            p1,
            p2,
            MaskOp::And {
                a: s(0),
                b: s(1),
                dst: 2,
            },
            MaskOp::AndNot {
                a: s(2),
                b: p(2),
                dst: 3,
            },
        ],
        Terminal::Count { mask: s(3) },
    );
    let mut sc = Scratch::for_program(&cold, N).expect("scratch");
    let (t_cold, v_cold) = median(reps, || {
        execute_extent(&cold, &hot, &Foreign::NONE, &mut sc, Out::None, 0..N).unwrap()
    });
    assert_eq!(v_hot, v_cold);
    println!(
        "\nbucket & palette & !other: cached masks {t_hot:.0} ns, apertures recomputed {t_cold:.0} ns ({:.1}x)",
        t_cold / t_hot
    );
}
