//! Arm C: per-store ingredient cost as one INT8 GEMM, against the oracle.

use foundry_parity_lab::arm_b::Guards;
use foundry_parity_lab::arm_c::{
    cost_fields, demanded_categories, CostQuery, CostRefusal, PreparedDemand,
};
use foundry_parity_lab::bind::{World, READER};
use foundry_parity_lab::fixture::{case, ITEMS, MENU_REV, STORES};
use foundry_parity_lab::oracle::Oracle;
use foundry_parity_lab::source::{Network, Order};
use lance_graph_dt64::LaneSource;

use crate::common::open;

fn cost_query<'s>(world: &World, net: &'s Network) -> Result<CostQuery<'s>, CostRefusal> {
    let binding = world.bind(net, &cost_fields(), net.order().space(), READER)?;
    CostQuery::open(&binding, &world.store, &world.view, net.frames(), net)
}

#[test]
fn per_store_cost_matches_the_oracle_for_every_store_item_and_quantity() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let demand = PreparedDemand::prepare(MENU_REV);
    let query = cost_query(&world, &net).expect("binds and stages");
    let o = Oracle::BASELINE;
    for q in [1, 3, 128] {
        let c = query.per_store(&demand, q).expect("answers");
        for s in 0..STORES {
            for m in 0..ITEMS {
                assert_eq!(c[s][m], o.store_cost(s, m, q), "store {s} item {m} q {q}");
            }
        }
    }
    // Anti-vacuity: costs genuinely vary by store, so agreement is informative.
    let c = query.per_store(&demand, 1).unwrap();
    let mut distinct: Vec<i64> = (0..STORES).filter_map(|s| c[s][0]).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(distinct.len() > 20, "{} distinct costs", distinct.len());
    assert_eq!(
        c[case::HIDDEN],
        [None; ITEMS],
        "the denied store has no cost"
    );
    assert_eq!(
        (0..STORES).filter(|&s| c[s][0].is_some()).count(),
        STORES - 1
    );
    // The materialized null model agrees too.
    let objects = foundry_parity_lab::arm_a::Objects::build(false);
    assert_eq!(objects.store_costs(1), c);
}

#[test]
fn per_store_cost_sums_to_arm_bs_network_cost_over_feasible_stores() {
    // Two independent arms: arm B's MaskedSum over CanCook, arm C's GEMM
    // restricted to the same feasibility mask.
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let demand = PreparedDemand::prepare(MENU_REV);
    let costs = cost_query(&world, &net).unwrap();
    for m in 0..ITEMS {
        let cook = open(&world, &net, m, true, false).unwrap();
        for q in [1, 3] {
            let mask = cook.cook_mask(q, Guards::ALL).unwrap();
            let c = costs.per_store(&demand, q).unwrap();
            let via_gemm: i64 = (0..STORES)
                .filter(|&r| mask >> r & 1 == 1)
                .map(|r| c[(cook.store_at(r) - 1000) as usize][m].unwrap())
                .sum();
            assert_eq!(via_gemm, cook.network_cost(q).unwrap(), "item {m} q {q}");
        }
    }
}

#[test]
fn only_demanded_cost_lanes_are_bound_and_a_is_staged_once() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let fields = cost_fields();
    let query = cost_query(&world, &net).unwrap();
    assert_eq!(fields.len(), demanded_categories().len() + 1);
    // 23 of the 64 categories are demanded by some item, plus `storeId`.
    assert_eq!(fields.len(), 24);
    assert_eq!(
        net.borrows.get(),
        fields.len() as u64,
        "one borrow per bound lane"
    );
    assert_eq!(query.staged_bytes(), 64 * 64);
    let demand = PreparedDemand::prepare(MENU_REV);
    for q in 1..=5 {
        query.per_store(&demand, q).unwrap();
    }
    assert_eq!(
        net.borrows.get(),
        fields.len() as u64,
        "answers reuse the staged A"
    );
}

#[test]
fn a_cost_that_does_not_fit_u8_is_refused_not_truncated() {
    let world = World::new();
    let k = demanded_categories()[0];
    let mut net = Network::ingest(Order::Identity);
    net.set_unit_cost(10, k, 300);
    match cost_query(&world, &net) {
        Err(CostRefusal::NotU8 { row, value, .. }) => assert_eq!((row, value), (10, 300)),
        other => panic!("expected NotU8, got {:?}", other.err()),
    }
    // Silent half: the largest u8 is admitted and answered exactly.
    let mut net = Network::ingest(Order::Identity);
    net.set_unit_cost(10, k, 255);
    let c = cost_query(&world, &net)
        .expect("255 fits u8")
        .per_store(&PreparedDemand::prepare(MENU_REV), 1)
        .unwrap();
    let expected: i64 = (0..ITEMS)
        .map(|m| {
            foundry_parity_lab::fixture::recipe(m)
                .demand
                .iter()
                .map(|&(kk, d)| {
                    let price = if kk == k {
                        255
                    } else {
                        foundry_parity_lab::fixture::unit_cost(10, kk)
                    };
                    i64::from(d) * i64::from(price)
                })
                .sum::<i64>()
        })
        .sum();
    assert_eq!((0..ITEMS).map(|m| c[10][m].unwrap()).sum::<i64>(), expected);
}

#[test]
fn a_stale_binding_is_refused_after_a_price_change() {
    let world = World::new();
    let mut net = Network::ingest(Order::Identity);
    let binding = world
        .bind(&net, &cost_fields(), net.order().space(), READER)
        .unwrap();
    net.set_unit_cost(3, demanded_categories()[0], 99);
    let r = CostQuery::open(&binding, &world.store, &world.view, net.frames(), &net);
    assert!(
        matches!(
            r,
            Err(CostRefusal::Lane(lance_graph_dt64::LaneRefusal::Frame(_)))
        ),
        "{:?}",
        r.err()
    );
}

#[test]
fn a_demand_band_prepared_for_another_menu_is_refused() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let query = cost_query(&world, &net).unwrap();
    let other = PreparedDemand::prepare(MENU_REV + 1);
    assert_eq!(
        query.per_store(&other, 1).err(),
        Some(CostRefusal::MenuRevision {
            prepared: MENU_REV + 1,
            asked: MENU_REV
        })
    );
    let demand = PreparedDemand::prepare(MENU_REV);
    for q in [0, 129] {
        assert_eq!(
            query.per_store(&demand, q).err(),
            Some(CostRefusal::Quantity { q })
        );
    }
}

#[test]
fn another_row_order_costs_identically_by_identity() {
    let world = World::new();
    let net = Network::ingest(Order::Reversed);
    let c = cost_query(&world, &net)
        .expect("binds in its own space")
        .per_store(&PreparedDemand::prepare(MENU_REV), 2)
        .unwrap();
    let o = Oracle::BASELINE;
    for s in 0..STORES {
        for m in 0..ITEMS {
            assert_eq!(c[s][m], o.store_cost(s, m, 2), "store {s} item {m}");
        }
    }
}
