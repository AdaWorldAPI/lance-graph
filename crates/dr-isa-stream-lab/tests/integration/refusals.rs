//! Coordinate space, frame and mask-space refusals, and why they matter.

use dr_isa_stream_lab::bind::{World, SPACE_A, SPACE_OTHER};
use dr_isa_stream_lab::cube::{fold, ArmRefusal, WhereMasks};
use dr_isa_stream_lab::fixture::{bf16_exact, Fixture, MASKS};
use dr_isa_stream_lab::gemm::{sums, sums_whole, GemmRefusal, Mode, SelectionB};
use dr_isa_stream_lab::oracle;
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use lance_graph_dt64::dr::{FrameRefusal, FrameRevisions};
use lance_graph_dt64::LaneRefusal;

fn fx() -> Fixture {
    Fixture {
        n: 1024,
        props: 16,
        seed: 21,
    }
}

/// A lane in another row order is refused at bind, before any read.
#[test]
fn a_lane_in_another_row_order_is_refused_at_bind() {
    let fx = fx();
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let mut socket = Socket::in_order(&lanes, 16);
    socket.spaces[5] = SPACE_OTHER;
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    match world.bind(&socket, &fields, SPACE_A) {
        Err(LaneRefusal::CoordinateSpace { field: 5, .. }) => {}
        other => panic!("expected a coordinate-space refusal, got {other:?}"),
    }
}

/// The refusal is load-bearing: the reordered lane, read positionally, gives
/// a wrong answer that the oracle catches.
#[test]
fn without_the_refusal_a_reordered_lane_reads_wrong() {
    let fx = fx();
    let mut lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    lanes.row_mut(5).reverse(); // field 5 now in reverse row order
    let socket = Socket::in_order(&lanes, 16); // ...but declared in SPACE_A
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let want: Vec<[f32; MASKS]> = oracle::sums(&fx, &(0..16).collect::<Vec<_>>())
        .iter()
        .map(|r| r.map(|v| v as f32))
        .collect();
    let (got, _) = sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap();
    assert_ne!(got[5], want[5], "the reversed lane must read wrong");
    assert_eq!(got[4], want[4], "neighbouring lanes are unaffected");
}

/// A source that changed after binding refuses both arms before they read.
#[test]
fn a_stale_frame_refuses_both_arms() {
    let fx = fx();
    let mut lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = {
        let socket = Socket::in_order(&lanes, 16);
        world.bind(&socket, &fields, SPACE_A).unwrap()
    };
    lanes.frames_mut().advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let socket = Socket::in_order(&lanes, 16);
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let stale = LaneRefusal::Frame(FrameRefusal::SourceChanged);
    assert_eq!(
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Arm(ArmRefusal::Lane(stale.clone()))
    );
    assert_eq!(
        fold(&world, &binding, &socket, &masks, &fields).unwrap_err(),
        ArmRefusal::Lane(stale)
    );
    // A fresh binding at the new frame is accepted again.
    let rebound = world.bind(&socket, &fields, SPACE_A).unwrap();
    assert!(sums(&world, &rebound, &socket, &masks, &fields, Mode::Direct).is_ok());
}

/// Masks computed in another row order are refused by both arms.
#[test]
fn masks_in_another_row_order_are_refused() {
    let fx = fx();
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_OTHER);
    let refusal = ArmRefusal::MaskSpace {
        masks: SPACE_OTHER,
        selection: SPACE_A,
    };
    assert_eq!(
        fold(&world, &binding, &socket, &masks, &fields).unwrap_err(),
        refusal
    );
    assert_eq!(
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Arm(refusal.clone())
    );
    let sel = SelectionB::stage(&masks);
    assert_eq!(
        sums_whole(&world, &binding, &socket, &sel, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Arm(refusal)
    );
}

/// Sanity of the refusal pair: equal spaces are accepted.
#[test]
fn matching_spaces_are_accepted() {
    let fx = fx();
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    assert!(fold(&world, &binding, &socket, &masks, &fields).is_ok());
    assert!(bf16_exact(1).is_some());
}
