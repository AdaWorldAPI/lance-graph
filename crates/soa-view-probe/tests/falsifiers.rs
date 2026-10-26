//! D-NFR-6 runtime falsifiers. The compile-time ones (no write during a live
//! view, no view outliving its owner, no second holder) are the
//! `compile_fail` examples in `src/lib.rs`.

use lance_graph_contract::NodeRowPacket;
use lance_graph_contract::canonical_node::NODE_ROW_STRIDE;
use lance_graph_contract::soa_envelope::SoaEnvelope;
use soa_view_probe::fixture::{CLASS_A, CLASS_B, rows};
use soa_view_probe::{FacetFold, FacetSlot, set_value_facet};

const SIZES: [usize; 8] = [1, 2, 63, 64, 65, 1000, 4095, 4096];

fn slot0() -> FacetSlot {
    FacetSlot::value(0).unwrap()
}

fn facet(row: &lance_graph_contract::NodeRow) -> (u32, &[u8]) {
    (
        u32::from_le_bytes(row.value[..4].try_into().unwrap()),
        &row.value[4..16],
    )
}

/// Row-loop oracle over the typed rows, not the bytes: count of `class`,
/// count matching (class, first payload byte < 128), payload byte sum.
fn oracle(r: &[lance_graph_contract::NodeRow], class: u32) -> (u64, u64, i64) {
    let hit: Vec<_> = r.iter().map(facet).filter(|(c, _)| *c == class).collect();
    (
        hit.len() as u64,
        hit.iter().filter(|(_, p)| p[0] < 128).count() as u64,
        hit.iter()
            .flat_map(|(_, p)| p.iter())
            .map(|&b| i64::from(b))
            .sum(),
    )
}

/// FAILS IF: an operation through the packet disagrees with the same ndarray
/// call on the raw bytes, or with the row loop, at any size.
#[test]
fn packet_operations_equal_direct_ndarray_and_the_row_loop() {
    for n in SIZES {
        let owner = rows(n);
        let view = NodeRowPacket::new(&owner, 1);
        let w = n.div_ceil(64);
        for class in [CLASS_A, CLASS_B] {
            let mut pattern = [0u8; 16];
            pattern[..4].copy_from_slice(&class.to_le_bytes());
            let mut care = [0u8; 16];
            care[..4].fill(0xFF);
            care[4] = 0x80; // first payload byte < 128

            let (mut sel, mut hit) = (vec![0u64; w], vec![0u64; w]);
            view.classid_mask(slot0(), class, &mut sel);
            view.facet_match_mask(slot0(), &pattern, &care, &mut hit);
            let got = (
                ndarray::simd::popcount_batch_u64(&sel),
                ndarray::simd::popcount_batch_u64(&hit),
                view.payload_byte_sum(slot0(), &sel),
            );
            assert_eq!(got, oracle(&owner, class), "n={n} class={class:#x}");

            let bytes = view.as_le_bytes();
            let at = slot0().offset();
            let (mut dsel, mut dhit) = (vec![0u64; w], vec![0u64; w]);
            ndarray::simd::eq_u32_strided_to_mask(bytes, at, NODE_ROW_STRIDE, n, class, &mut dsel);
            ndarray::simd::ternary_match_strided16_to_mask(
                bytes,
                at,
                NODE_ROW_STRIDE,
                n,
                &pattern,
                &care,
                &mut dhit,
            );
            let dsum = ndarray::simd::masked_strided_group_sum(
                bytes,
                at + 4,
                NODE_ROW_STRIDE,
                n,
                12,
                1,
                &dsel,
            );
            assert_eq!((sel, hit, Some(got.2)), (dsel, dhit, dsum), "n={n}");
        }
    }
}

/// FAILS IF: the fixture is vacuous (one class only, or the payload filter
/// keeps all or none).
#[test]
fn the_fixture_is_not_vacuous() {
    let owner = rows(4096);
    let (a, a_lo, _) = oracle(&owner, CLASS_A);
    let (b, _, _) = oracle(&owner, CLASS_B);
    assert_eq!(a + b, 4096);
    assert!(b > 1000 && a > 1000, "a={a} b={b}");
    assert!(a_lo > 0 && a_lo < a);
}

/// FAILS IF: a view taken after the owner wrote reports the old bytes. There
/// is no stale view to refuse: the old one cannot exist across the write.
#[test]
fn a_view_taken_after_a_write_sees_the_write() {
    let mut owner = rows(130);
    let (a, _, _) = oracle(&owner, CLASS_A);
    let mut facet = [0u8; 16];
    facet[..4].copy_from_slice(&CLASS_B.to_le_bytes());
    for row in owner.iter_mut() {
        set_value_facet(row, 0, facet);
    }
    let view = NodeRowPacket::new(&owner, 2);
    let mut sel = vec![0u64; 3];
    view.classid_mask(slot0(), CLASS_A, &mut sel);
    assert!(a > 0);
    assert_eq!(ndarray::simd::popcount_batch_u64(&sel), 0);
    view.classid_mask(slot0(), CLASS_B, &mut sel);
    assert_eq!(ndarray::simd::popcount_batch_u64(&sel), 130);
    assert_eq!(view.payload_byte_sum(slot0(), &sel), 0);
}

/// FAILS IF: the facet geometry drifts: slot 0 is the key, value facet 0 sits
/// at byte 32, and there are exactly 32 slots of 16 bytes.
#[test]
fn facet_slots_are_the_canonical_16_byte_grid() {
    assert_eq!(FacetSlot::KEY.offset(), 0);
    assert_eq!(slot0().offset(), 32);
    assert!(FacetSlot::new(31).is_some());
    assert!(FacetSlot::new(32).is_none());
    assert!(FacetSlot::value(29).is_some());
    assert!(FacetSlot::value(30).is_none());
    let owner = rows(3);
    let view = NodeRowPacket::new(&owner, 7);
    assert!(view.verify_layout().is_ok());
    assert_eq!(view.row_stride(), 512);
    // Key facet: the minted classid is in key bytes 0..4.
    let mut sel = [0u64; 1];
    view.classid_mask(FacetSlot::KEY, CLASS_A, &mut sel);
    assert_eq!(sel[0], 0b111);
}

/// D-NFR-1 escaped reuse. A mask computed through a view is an owned
/// `Vec<u64>`: it outlives the borrow, so the borrow no longer protects it.
/// The packet's `cycle()` is the stamp a reader would naturally keep beside
/// it, but it is whatever the caller passed to `NodeRowPacket::new`, not a
/// record of writes. FAILS IF: the escaped mask still answers correctly after
/// the owner writes (the predicate input did not change), or if the cycle
/// stamp differs between the two views (it would then be a witness after all).
#[test]
fn an_escaped_mask_is_not_witnessed_by_the_packet_cycle() {
    let mut owner = rows(130);
    let kept = {
        let view = NodeRowPacket::new(&owner, 5);
        let mut sel = vec![0u64; 3];
        view.classid_mask(slot0(), CLASS_A, &mut sel);
        sel
    }; // borrow released; `kept` escapes
    let (a, _, _) = oracle(&owner, CLASS_A);
    assert!(a > 0 && a < 130, "fixture must mix classes");

    let flip = owner
        .iter()
        .position(|r| facet(r).0 == CLASS_B)
        .expect("a CLASS_B row");
    owner[flip].value[..4].copy_from_slice(&CLASS_A.to_le_bytes());

    let view = NodeRowPacket::new(&owner, 5); // same caller stamp
    assert_eq!(view.cycle(), 5);
    let mut fresh = vec![0u64; 3];
    view.classid_mask(slot0(), CLASS_A, &mut fresh);
    assert_ne!(kept, fresh, "the escaped mask still matches");
    assert_eq!(
        ndarray::simd::popcount_batch_u64(&fresh),
        ndarray::simd::popcount_batch_u64(&kept) + 1
    );
}
