//! Refusals: what the binding must not answer.

use foundry_parity_lab::arm_b::{demanded_fields, Answer, ArmRefusal, Guards, ItemQuery};
use foundry_parity_lab::bind::{World, COST0, READER, SPACE_A, STOCK0};
use foundry_parity_lab::fixture::{case, store_id, CHEESE, ITEMS, STORES};
use foundry_parity_lab::oracle::Oracle;
use foundry_parity_lab::source::{Network, Order};
use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::{LaneRefusal, LaneSource};

use crate::common::{coarse, open};

#[test]
fn a_stale_frame_is_refused_and_a_rebind_sees_the_restock() {
    let world = World::new();
    let mut net = Network::ingest(Order::Identity);
    let fields = demanded_fields(0, false, false);
    let before = world.bind(&net, &fields, SPACE_A, READER).expect("binds");
    net.restock();
    let stale = ItemQuery::open(0, &before, &world.store, &world.view, net.frames(), &net);
    assert!(
        matches!(
            stale,
            Err(ArmRefusal::Lane(LaneRefusal::Frame(
                FrameRefusal::SourceChanged
            )))
        ),
        "{:?}",
        stale.err()
    );
    let after = open(&world, &net, 0, false, false).expect("rebinds");
    let b = after.answers(1, Guards::ALL).expect("answers");
    let o = Oracle {
        restocked: true,
        closed: &[],
    };
    assert_eq!(
        b[case::RESTOCKED],
        coarse(o.can_cook(case::RESTOCKED, 0, 1))
    );
    assert_eq!(
        b[case::RESTOCKED],
        Answer::Known {
            feasible: true,
            max: 30
        }
    );
    assert_eq!(after.network_capacity().unwrap(), o.network_capacity(0));
    assert!(o.network_capacity(0) > Oracle::BASELINE.network_capacity(0));
}

#[test]
fn a_lane_serving_another_scope_is_refused() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let r = world.bind(
        &net,
        &demanded_fields(0, false, false),
        SPACE_A,
        Scope { id: 99 },
    );
    assert!(matches!(r, Err(LaneRefusal::Scope { .. })), "{r:?}");
}

#[test]
fn another_row_order_is_refused_under_the_canonical_selection() {
    let world = World::new();
    let net = Network::ingest(Order::Reversed);
    let r = world.bind(&net, &demanded_fields(0, false, false), SPACE_A, READER);
    assert!(
        matches!(r, Err(LaneRefusal::CoordinateSpace { .. })),
        "{r:?}"
    );
}

#[test]
fn another_row_order_answers_identically_by_global_identity() {
    let world = World::new();
    let net = Network::ingest(Order::Reversed);
    let o = Oracle::BASELINE;
    for m in 0..ITEMS {
        let query = open(&world, &net, m, false, false).expect("binds in its own space");
        // The ordinal moved; the identity did not.
        assert_eq!(
            query.store_at(STORES - 1 - case::REORDERED),
            store_id(case::REORDERED)
        );
        let b = query.answers(2, Guards::ALL).expect("answers");
        for s in 0..STORES {
            assert_eq!(b[s], coarse(o.can_cook(s, m, 2)), "store {s} item {m}");
        }
        assert_eq!(query.network_capacity().unwrap(), o.network_capacity(m));
    }
}

#[test]
fn a_field_the_source_does_not_map_is_refused_not_read_as_empty() {
    let world = World::new();
    let mut net = Network::ingest(Order::Identity);
    net.unbound = Some((STOCK0 + CHEESE) as u8);
    let r = world.bind(&net, &demanded_fields(0, false, false), SPACE_A, READER);
    assert!(matches!(r, Err(LaneRefusal::Unbound { .. })), "{r:?}");
}

#[test]
fn a_question_needing_an_unbound_field_is_refused() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let query = open(&world, &net, 0, false, false).expect("binds without cost lanes");
    let r = query.network_cost(1);
    assert!(
        matches!(r, Err(ArmRefusal::NotBound { field }) if usize::from(field) >= COST0),
        "{r:?}"
    );
    assert!(matches!(
        query.cook_mask(0, Guards::ALL),
        Err(ArmRefusal::Quantity { q: 0 })
    ));
}
