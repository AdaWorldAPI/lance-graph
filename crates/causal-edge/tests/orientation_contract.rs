//! D-CGW-1: CE64 bits 43..45 carry one meaning everywhere, the orientation
//! (`causal_edge::orientation::Orientation`). These pin the code table, the
//! accessor and both writers to that one table, and field isolation 43..45
//! against its neighbours 40..42 (Pearl) and 46..49 (inference mantissa).
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::edge::InferenceType;
use causal_edge::orientation::Orientation;
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::CausalEdge64;

#[test]
fn code_table_is_one_to_one_over_three_bits() {
    let mut seen = [false; 8];
    for o in Orientation::ALL {
        let c = o.code();
        assert!(c < 8, "{o:?} code {c} does not fit 3 bits");
        assert!(!seen[c as usize], "{o:?} shares code {c}");
        seen[c as usize] = true;
        assert_eq!(Orientation::from_code(c), o, "round trip of {o:?}");
    }
    assert!(
        seen.iter().all(|&s| s),
        "every 3-bit code names an orientation"
    );
    // Only the low three bits are read.
    assert_eq!(
        Orientation::from_code(0b1010),
        Orientation::from_code(0b010)
    );
}

#[test]
fn zero_is_the_core_argument() {
    assert_eq!(Orientation::from_code(0), Orientation::Core);
    assert_eq!(CausalEdge64(0).orientation(), Orientation::Core);
}

#[test]
fn accessor_and_setter_agree_for_every_code() {
    for o in Orientation::ALL {
        let mut e = CausalEdge64(0);
        e.set_orientation(o);
        assert_eq!(e.orientation(), o);
        assert_eq!(
            (e.0 >> 43) & 0b111,
            u64::from(o.code()),
            "{o:?} lands on 43..45"
        );
    }
}

#[test]
fn both_packers_write_the_same_table() {
    for o in Orientation::ALL {
        let v2 = CausalEdge64::pack_v2(
            1,
            2,
            3,
            200,
            100,
            CausalMask::SPO,
            o,
            PlasticityState::ALL_HOT,
        );
        assert_eq!(v2.orientation(), o, "pack_v2 {o:?}");
        #[allow(deprecated)]
        let v1 = CausalEdge64::pack(
            1,
            2,
            3,
            200,
            100,
            CausalMask::SPO,
            o,
            InferenceType::Deduction,
            PlasticityState::ALL_HOT,
            0,
        );
        assert_eq!(v1.orientation(), o, "pack {o:?}");
    }
}

#[test]
fn orientation_is_isolated_from_its_neighbours() {
    // Every bit outside 43..45 set: writing the orientation must not move any.
    let outside = !(0b111u64 << 43);
    for o in Orientation::ALL {
        let mut e = CausalEdge64(outside);
        e.set_orientation(o);
        assert_eq!(e.0 & outside, outside, "{o:?} disturbed a neighbour");
        assert_eq!(e.orientation(), o);
    }
    // And the neighbours' setters must not move the orientation.
    for o in Orientation::ALL {
        let mut e = CausalEdge64(0);
        e.set_orientation(o);
        e.set_causal_mask(CausalMask::SPO);
        let e = e.with_inference_mantissa(-6);
        assert_eq!(e.orientation(), o, "{o:?} moved by a neighbour write");
        assert_eq!(e.causal_mask(), CausalMask::SPO);
        assert_eq!(e.inference_mantissa(), -6);
    }
}
