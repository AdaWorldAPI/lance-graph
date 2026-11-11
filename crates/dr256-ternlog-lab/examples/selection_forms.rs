//! Measures three forms of the same slab planes: the contiguous words the
//! fold reads, the borrowed per-row `Selection` view, and one owned
//! `WideFieldMask` per row (the measured alternative, built here only).
//! Every arm's answer is checked against a row-at-a-time oracle first.

use dr256_ternlog_lab::*;
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::Scope;
use lance_graph_mask_risc::LaneRef;
use std::hint::black_box;
use std::time::Instant;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
/// 3 of 4 rows are present.
fn present(i: usize) -> bool {
    !mix(i as u64).is_multiple_of(4)
}
/// 7 of 8 rows are in the reader's scope.
fn in_scope(i: usize) -> bool {
    !mix(i as u64 ^ 0xA5A5).is_multiple_of(8)
}

// One heap allocation per row: position 255 forces the four-chunk form.
fn owned(sel: Selection<'_>) -> WideFieldMask {
    let mut m = WideFieldMask::EMPTY.with(255);
    let w = m.words_mut();
    w.fill(0);
    w[..sel.words().len()].copy_from_slice(sel.words());
    m
}

fn fmt(ns: f64) -> String {
    if ns < 10_000.0 {
        format!("{ns:.0}ns")
    } else {
        format!("{:.2}\u{b5}s", ns / 1_000.0)
    }
}

/// Median and minimum of 31 samples, each `k` calls under one clock read.
fn measure(n: usize, arm: &str, k: u32, want: u64, f: &mut dyn FnMut() -> u64) -> f64 {
    assert_eq!(f(), want, "arm {arm}");
    for _ in 0..5 {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(31);
    for _ in 0..31 {
        let t = Instant::now();
        for _ in 0..k {
            black_box(f());
        }
        samples.push(t.elapsed().as_nanos() as f64 / f64::from(k));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "n={n} arm={arm} median={} min={}",
        fmt(samples[15]),
        fmt(samples[0])
    );
    samples[15]
}

fn sum3(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((a, b), c)| u64::from((a & b & c).count_ones()))
        .sum()
}

fn main() {
    let world = World::new();
    for n in [65_536usize, 1_000_000] {
        let k = if n == 65_536 { 200 } else { 20 };
        let id: Vec<u32> = (0..n as u32).collect();
        let age: Vec<i32> = (0..n as i32).map(|i| 20 + (i % 60)).collect();
        let table = Table::new(
            PERSON,
            1,
            n,
            1,
            Scope { id: 7 },
            &[
                (person::ID, LaneRef::U32(&id)),
                (person::AGE, LaneRef::I32(&age)),
            ],
        );
        let mask = select(&world.view, PERSON, &["person_id", "age"]).unwrap();
        let prepared = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
        let want = (0..n).filter(|&i| present(i) && in_scope(i)).count() as u64;
        let slabs = prepared.slabs();

        // The owned form, built once here; building it again is the build arm.
        let build = || -> Vec<[Vec<WideFieldMask>; 3]> {
            (0..slabs)
                .map(|s| {
                    let one = |p: usize| prepared.plane(s, p).selections().map(owned).collect();
                    [one(0), one(1), one(2)]
                })
                .collect()
        };
        let owned_form = build();
        let values: usize = owned_form
            .iter()
            .map(|planes| planes.iter().map(Vec::len).sum::<usize>())
            .sum();
        let selections: usize = (0..slabs)
            .flat_map(|s| (0..3).map(move |p| (s, p)))
            .map(|(s, p)| prepared.plane(s, p).masks())
            .sum();
        assert_eq!(values, selections);
        println!(
            "n={n} slabs={slabs} selections={selections} contiguous_bytes={} owned_allocations={values} owned_bytes={}",
            prepared.plane_bytes(),
            values * (std::mem::size_of::<WideFieldMask>() + 32)
        );

        let mut fold = Fold::count_and3().unwrap();
        measure(n, "fold/contiguous", k, want, &mut || {
            fold.count(&world, &table, &prepared).unwrap()
        });

        let contig = measure(n, "words/contiguous", k, want, &mut || {
            (0..slabs)
                .map(|s| {
                    sum3(
                        &prepared.plane(s, 0).words,
                        &prepared.plane(s, 1).words,
                        &prepared.plane(s, 2).words,
                    )
                })
                .sum()
        });

        let view = measure(n, "words/borrowed_view", k, want, &mut || {
            let mut total = 0u64;
            for s in 0..slabs {
                let (a, b, c) = (
                    prepared.plane(s, 0),
                    prepared.plane(s, 1),
                    prepared.plane(s, 2),
                );
                for y in 0..a.masks() {
                    let (x, y1, z) = (a.mask(y).unwrap(), b.mask(y).unwrap(), c.mask(y).unwrap());
                    total += sum3(x.words(), y1.words(), z.words());
                }
            }
            total
        });

        let wide = measure(n, "words/owned_wide", k, want, &mut || {
            let mut total = 0u64;
            for planes in &owned_form {
                for ((x, y1), z) in planes[0].iter().zip(&planes[1]).zip(&planes[2]) {
                    total += sum3(x.words(), y1.words(), z.words());
                }
            }
            total
        });

        // Three reusable contiguous buffers per slab, allocated outside the timing.
        let mut buffers: Vec<[Vec<u64>; 3]> = (0..slabs)
            .map(|s| {
                let len = prepared.plane(s, 0).words.len();
                [vec![0; len], vec![0; len], vec![0; len]]
            })
            .collect();
        let gather = measure(n, "gather/owned_wide", k, want, &mut || {
            for (planes, bufs) in owned_form.iter().zip(buffers.iter_mut()) {
                for p in 0..3 {
                    let len = bufs[p].len();
                    for (y, m) in planes[p].iter().enumerate() {
                        let lo = y * MASK_WORDS;
                        let take = MASK_WORDS.min(len - lo);
                        bufs[p][lo..lo + take].copy_from_slice(&m.words()[..take]);
                    }
                }
            }
            buffers.iter().map(|b| sum3(&b[0], &b[1], &b[2])).sum()
        });

        measure(n, "build/owned_wide", 1, values as u64, &mut || {
            build()
                .iter()
                .map(|planes| planes.iter().map(Vec::len).sum::<usize>() as u64)
                .sum()
        });

        println!(
            "n={n} ratios borrowed_view/contiguous={:.2} owned_wide/contiguous={:.2} gather/contiguous={:.2}",
            view / contig,
            wide / contig,
            gather / contig
        );
    }
}
