//! The resolved world's byte rule: a multi-byte integer read from raw bytes is
//! little-endian. `Cmp::EqU32Strided(0x1234_5678)` is a NUMBER; the strided
//! lane under it is BYTES, and the bytes `[0x78, 0x56, 0x34, 0x12]` are what
//! that number looks like on the substrate.
//!
//! The fixtures are written as literal byte vectors, never with
//! `to_ne_bytes`/`to_le_bytes`, so the test means the same thing on every
//! host: it cannot pass merely because the CI machine is little-endian. Each
//! fixture also stores the byte-swapped spelling `[0x12, 0x34, 0x56, 0x78]`
//! in other rows, which a big-endian reader would match instead.
//!
//! `Register128`'s word encoding has its own known-vector test in the owning
//! crate (`register128::tests::words_round_trip_little_endian`); here it is
//! only used as the production encoder of one record, to show that what the
//! contract writes and what the strided kernel reads agree.

use lance_graph_contract::register128::Register128;
use lance_graph_mask_risc::{
    execute, materialize_rows, reference_execute, words_for, Foreign, LaneRef, Out, Planes,
    Scratch, StridedRef, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

const V: u32 = 0x1234_5678;
const LE: [u8; 4] = [0x78, 0x56, 0x34, 0x12];
const BE: [u8; 4] = [0x12, 0x34, 0x56, 0x78];

/// `n` records of `stride` bytes. Rows `0`, `17` hold the LE spelling, row
/// `1` the BE spelling, everything else zero. The two counts differ on
/// purpose: with equal counts a big-endian reader in the reference
/// interpreter would still report the same `Count` and slip through. `n = 20` puts rows in
/// both the 16-lane group path and the scalar tail of the kernel.
fn fixture(stride: usize) -> Vec<u8> {
    let n = 20;
    let mut b = vec![0u8; n * stride];
    for (row, word) in [(0, LE), (17, LE), (1, BE)] {
        b[row * stride..row * stride + 4].copy_from_slice(&word);
    }
    b
}

/// Run `cmp` on the strided lane through both mask-risc readers — the
/// executor (ndarray's kernel) and the row-by-row reference — and return
/// the matching rows, asserting the two agree.
fn rows(bytes: &[u8], stride: usize, cmp: Cmp) -> Vec<usize> {
    let n = bytes.len() / stride;
    let p = lower(&Query {
        filter: Filter::cmp(Col(0), cmp),
        agg: Agg::Rows,
    })
    .unwrap();
    let lanes = [LaneRef::Strided(StridedRef {
        bytes,
        first_offset: 0,
        stride,
        records: n,
    })];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&p, n).unwrap();
    let mut out = vec![0u64; words_for(n)];
    lance_graph_mask_risc::execute_into(
        &p,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut out),
    )
    .unwrap();
    let got = materialize_rows(&out, n);

    // The reference interpreter decodes the same bytes independently.
    let count = lower(&Query {
        filter: Filter::cmp(Col(0), cmp),
        agg: Agg::Count,
    })
    .unwrap();
    let mut s2 = Scratch::for_program(&count, n).unwrap();
    let exec = execute(&count, &planes, &mut s2, None).unwrap();
    let refr = reference_execute(&count, &planes, None).unwrap();
    assert_eq!(exec, refr, "executor and reference disagree");
    assert_eq!(exec, Value::Count(got.len()));
    got
}

/// The numeric value `0x12345678` matches the rows whose bytes are
/// `[0x78, 0x56, 0x34, 0x12]`, and not the rows that spell it big-endian.
/// Covered for a contiguous lane (stride 4) and a lane inside wider
/// records (stride 16).
#[test]
fn eq_u32_strided_reads_canonical_le_bytes() {
    for stride in [4, 16] {
        let b = fixture(stride);
        assert_eq!(rows(&b, stride, Cmp::EqU32Strided(V)), vec![0, 17], "stride {stride}");
        // Can-fire: a big-endian reader would match row 1 instead.
        assert_eq!(
            rows(&b, stride, Cmp::EqU32Strided(V.swap_bytes())),
            vec![1],
            "stride {stride}"
        );
        // `!=` is the complement over the same canonical reading.
        let ne = rows(&b, stride, Cmp::NeU32Strided(V));
        assert_eq!(ne.len(), 18); // 20 rows minus the two LE matches
        assert!(!ne.contains(&0) && !ne.contains(&17));
    }
}

/// The contract's own encoder (`Register128::from_words`) writes the bytes the
/// strided kernel reads back as the same number: one byte rule on both sides
/// of the boundary.
#[test]
fn register128_encoding_is_read_back_by_the_strided_kernel() {
    let r = Register128::from_words([V, 0, 0, 0]);
    assert_eq!(r.0[..4], LE);
    let mut b = vec![0u8; 20 * 16];
    b[5 * 16..6 * 16].copy_from_slice(&r.0);
    assert_eq!(rows(&b, 16, Cmp::EqU32Strided(V)), vec![5]);
}
