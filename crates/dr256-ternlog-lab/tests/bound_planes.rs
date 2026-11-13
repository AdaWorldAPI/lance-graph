//! Bit-planes derived through per-slab bindings: they hold the bits, they are
//! refused when their frame or row order is not the binding's.
//! Guards that the benchmark's checks
//! (`.claude/board/entries/2026-11-13-wx-calibrated-plane-admission.md`, Cost)
//! are not vacuous.

use dr256_ternlog_lab::bit_planes::BitPlanes;
use dr256_ternlog_lab::bound_planes::{range_program, BoundBitPlanes};
use dr256_ternlog_lab::{person, select, Plane, Refused, Table, World, PERSON};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::LaneRefusal;
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};

const N: usize = 721 * 1440;
const LO: u8 = 145;
const HI: u8 = 183;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn codes() -> Vec<i32> {
    (0..N as u64).map(|i| (mix(i) % 256) as i32).collect()
}

fn oracle(codes: &[i32]) -> u64 {
    codes
        .iter()
        .filter(|&&c| i32::from(LO) < c && c < i32::from(HI))
        .count() as u64
}

fn table<'a>(key: &'a [u32], codes: &'a [i32]) -> Table<'a> {
    Table::new(
        PERSON,
        1,
        codes.len(),
        1,
        Scope { id: 7 },
        &[
            (person::ID, LaneRef::U32(key)),
            (person::AGE, LaneRef::I32(codes)),
        ],
    )
}

/// FAILS IF: the sixteen-slab count differs from the row loop, the query
/// selects nothing or nearly everything (a vacuous answer), or the resident
/// bytes are not eight planes of the slabs' words.
#[test]
fn sixteen_slabs_count_equals_the_row_loop() {
    let codes = codes();
    let key = vec![0u32; N];
    let table = table(&key, &codes);
    let world = World::new();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let bp = BoundBitPlanes::derive(&world, &table, &fields, person::AGE).unwrap();
    assert_eq!(bp.slabs(), 16);
    let program = range_program(LO, HI).unwrap();
    let mut scratch = Scratch::for_program(&program, dr256_ternlog_lab::SLAB_ROWS).unwrap();
    let got = bp.count(&world, &table, &program, &mut scratch).unwrap();
    let want = oracle(&codes);
    assert_eq!(got, want);
    assert!(0 < want && want < (N / 5) as u64, "selected {want}");
    let words: usize = (0..16).map(|s| table.slab_rows(s).div_ceil(64)).sum();
    assert_eq!(bp.bytes(), 8 * words * 8);
}

/// FAILS IF: the per-slab planes, executed unchecked, count differently from
/// the same program over the eight contiguous planes of the whole column.
#[test]
fn planes_equal_the_unbound_slicing() {
    let codes = codes();
    let key = vec![0u32; N];
    let table = table(&key, &codes);
    let world = World::new();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let bp = BoundBitPlanes::derive(&world, &table, &fields, person::AGE).unwrap();
    let program = range_program(LO, HI).unwrap();
    let mut slab_scratch = Scratch::for_program(&program, dr256_ternlog_lab::SLAB_ROWS).unwrap();
    let per_slab = bp.count_unchecked(&program, &mut slab_scratch).unwrap();

    let codes8: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
    let flat = BitPlanes::of(&codes8);
    let masks: Vec<&[u64]> = (0..8).map(|k| flat.planes[k].as_slice()).collect();
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let mut scratch = Scratch::for_program(&program, N).unwrap();
    let contiguous = match execute_into(
        &program,
        &planes,
        &Foreign {
            planes: &[],
            lanes: &[],
        },
        &mut scratch,
        Out::None,
    )
    .unwrap()
    {
        Value::Count(c) => c as u64,
        other => panic!("not a count: {other:?}"),
    };
    assert_eq!(per_slab, contiguous);
    assert_eq!(per_slab, oracle(&codes));
}

/// FAILS IF: planes derived before the source moved are still admitted after
/// it moved, by either `check` or `count`, or a fresh derivation does not
/// count correctly.
#[test]
fn a_moved_source_frame_refuses_the_resident_planes() {
    let codes = codes();
    let key = vec![0u32; N];
    let mut table = table(&key, &codes);
    let world = World::new();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let bp = BoundBitPlanes::derive(&world, &table, &fields, person::AGE).unwrap();
    let program = range_program(LO, HI).unwrap();
    let mut scratch = Scratch::for_program(&program, dr256_ternlog_lab::SLAB_ROWS).unwrap();
    assert!(bp.check(&world, &table).is_ok());

    table.advance_source();
    assert!(matches!(
        bp.check(&world, &table),
        Err(Refused::Lane(LaneRefusal::Frame(_)))
    ));
    assert!(matches!(
        bp.count(&world, &table, &program, &mut scratch),
        Err(Refused::Lane(LaneRefusal::Frame(_)))
    ));

    let fresh = BoundBitPlanes::derive(&world, &table, &fields, person::AGE).unwrap();
    assert_eq!(
        fresh.count(&world, &table, &program, &mut scratch).unwrap(),
        oracle(&codes)
    );
}

/// FAILS IF: a plane in another slab's row order is admitted, or the refusal
/// does not name the slab and plane it found.
#[test]
fn a_plane_from_another_row_order_is_refused() {
    let codes = codes();
    let key = vec![0u32; N];
    let table = table(&key, &codes);
    let world = World::new();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let mut bp = BoundBitPlanes::derive(&world, &table, &fields, person::AGE).unwrap();
    assert!(bp.check(&world, &table).is_ok());
    bp.set_plane(
        0,
        3,
        Plane::from_fn(table.space(1), table.slab_rows(0), |_| true),
    );
    assert_eq!(
        bp.check(&world, &table),
        Err(Refused::Misaligned { slab: 0, plane: 3 })
    );
}
