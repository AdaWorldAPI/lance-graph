//! `lance-graph-xlog` against an independent scalar oracle, plus the alias
//! gates of plan `d-xlog-native-v1.md` §6 (N1, N5).

use std::collections::BTreeMap;

use lance_graph_dt64::dr::{LocalMap, MapRefusal, Width};
use lance_graph_xlog::{bag_count, ids, nonzero, support, Bounds, NativeRefusal};

/// r(X, Y) and s(Y, Z), with repeated Y on both sides and a Y with no partner.
const R: [(u32, u32); 6] = [(1, 2), (5, 2), (1, 3), (9, 7), (4, 8), (1, 2)];
const S: [(u32, u32); 5] = [(2, 4), (2, 6), (3, 4), (7, 8), (9, 1)];

fn y_map() -> LocalMap {
    let mut ys: Vec<u64> = R
        .iter()
        .map(|t| u64::from(t.1))
        .chain(S.iter().map(|t| u64::from(t.0)))
        .collect();
    ys.sort_unstable();
    ys.dedup();
    LocalMap::prepare(Width::W64, &ys).unwrap()
}

/// Every (x, y, z) witness, enumerated: per-Y derivation counts.
fn scalar(b: Bounds) -> BTreeMap<u64, i64> {
    let mut out = BTreeMap::new();
    for &(x, y) in &R {
        for &(y2, z) in &S {
            let keep = y == y2 && b.left.is_none_or(|v| v == x) && b.right.is_none_or(|v| v == z);
            if keep {
                *out.entry(u64::from(y)).or_insert(0) += 1;
            }
        }
    }
    out
}

const BOUNDS: [Bounds; 4] = [
    Bounds {
        left: None,
        right: None,
    },
    Bounds {
        left: Some(1),
        right: None,
    },
    Bounds {
        left: None,
        right: Some(4),
    },
    Bounds {
        left: Some(1),
        right: Some(6),
    },
];

#[test]
fn bag_count_matches_enumeration_under_every_bound() {
    let mid = y_map();
    for b in BOUNDS {
        let got: BTreeMap<u64, i64> = nonzero(&mid, &bag_count(b, &R, &S, &mid).unwrap())
            .into_iter()
            .collect();
        assert_eq!(got, scalar(b), "{b:?}");
    }
    // The fixture keeps multiplicity: (1, 2) twice in R.
    assert_eq!(scalar(BOUNDS[0])[&2], 6);
}

#[test]
fn support_is_the_set_of_counted_y() {
    let mid = y_map();
    for b in BOUNDS {
        let s = support(b, &R, &S, &mid).unwrap();
        let want: Vec<u64> = scalar(b).into_keys().collect();
        assert_eq!(ids(&mid, &s.answer), want, "{b:?}");
        assert!(!ids(&mid, &s.forward).is_empty());
    }
    // Y = 8 has a left partner and no right one; Y = 9 the reverse.
    let s = support(BOUNDS[0], &R, &S, &mid).unwrap();
    assert!(ids(&mid, &s.forward).contains(&8) && !ids(&mid, &s.answer).contains(&8));
    assert!(ids(&mid, &s.backward).contains(&9) && !ids(&mid, &s.answer).contains(&9));
}

#[test]
fn a_y_outside_the_map_is_refused() {
    let mid = LocalMap::prepare(Width::W64, &[2, 3]).unwrap();
    assert_eq!(
        bag_count(BOUNDS[0], &R, &S, &mid).err(),
        Some(NativeRefusal::Domain(MapRefusal::Unmapped))
    );
}

#[test]
fn and_is_the_native_symbol_and_the_onnx_one() {
    type Bin = fn(&[u64], &[u64], &mut [u64]);
    assert!(std::ptr::fn_addr_eq(
        lance_graph_xlog::and as Bin,
        ndarray::simd::mask_and as Bin
    ));
    assert!(std::ptr::fn_addr_eq(
        lance_graph_xlog::and as Bin,
        lance_graph_onnx::and as Bin
    ));
}
