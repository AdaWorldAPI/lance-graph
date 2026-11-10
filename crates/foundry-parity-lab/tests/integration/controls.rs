//! Negative controls: bypass one load-bearing guard and the answer is wrong.

use foundry_parity_lab::arm_b::{Answer, Guards};
use foundry_parity_lab::bind::World;
use foundry_parity_lab::fixture::case;
use foundry_parity_lab::oracle::{Cook, Oracle};
use foundry_parity_lab::source::{Network, Order};

use crate::common::{coarse, open};

#[test]
fn bypassing_the_known_guard_turns_unknown_stock_into_a_false_yes() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let query = open(&world, &net, 0, false, false).unwrap();
    let guarded = query.answers(1, Guards::ALL).unwrap();
    let bypass = query
        .answers(
            1,
            Guards {
                known: false,
                ..Guards::ALL
            },
        )
        .unwrap();
    let truth = coarse(Oracle::BASELINE.can_cook(case::UNKNOWN_CHEESE, 0, 1));
    assert_eq!(guarded[case::UNKNOWN_CHEESE], truth);
    assert!(matches!(
        bypass[case::UNKNOWN_CHEESE],
        Answer::Known { feasible: true, .. }
    ));
    assert_ne!(
        bypass[case::UNKNOWN_CHEESE],
        truth,
        "the stale 1000 is read as stock"
    );
    // Silent half: the guard changes nothing for stores whose stock is known.
    for s in 0..64 {
        if !matches!(Oracle::BASELINE.can_cook(s, 0, 1), Cook::Unknown { .. }) {
            assert_eq!(bypass[s], guarded[s], "store {s}");
        }
    }
}

#[test]
fn bypassing_authorization_reveals_the_denied_store() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let query = open(&world, &net, 0, false, false).unwrap();
    let bypass = query
        .answers(
            1,
            Guards {
                authorization: false,
                ..Guards::ALL
            },
        )
        .unwrap();
    assert_eq!(
        Oracle::BASELINE.can_cook(case::HIDDEN, 0, 1),
        Cook::NotAuthorized
    );
    assert!(
        matches!(bypass[case::HIDDEN], Answer::Known { .. }),
        "{:?}",
        bypass[case::HIDDEN]
    );
    let guarded = query.answers(1, Guards::ALL).unwrap();
    let differing = (0..64).filter(|&s| bypass[s] != guarded[s]).count();
    assert_eq!(differing, 1, "authorization hides exactly the denied store");
}

#[test]
fn unknown_is_not_zero() {
    // The unknown store's answer is neither "infeasible" nor "max 0".
    let o = Oracle::BASELINE.can_cook(case::UNKNOWN_CHEESE, 0, 1);
    assert!(matches!(o, Cook::Unknown { .. }));
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let b = open(&world, &net, 0, false, false)
        .unwrap()
        .answers(1, Guards::ALL)
        .unwrap();
    assert_eq!(b[case::UNKNOWN_CHEESE], Answer::Unknown);
}

#[test]
fn the_scenario_is_oracle_only_and_lowers_capacity() {
    // Level S: closing store 12 for the window. The DR path has no overlay
    // contract, so only the scalar reference answers; canonical data is not
    // touched (the baseline is recomputed after and is unchanged).
    let base = Oracle::BASELINE;
    let closed = Oracle {
        restocked: false,
        closed: &[11],
    };
    let before = base.network_capacity(0);
    let after = closed.network_capacity(0);
    let lost = match base.can_cook(11, 0, 1) {
        Cook::Known { max, .. } => i64::from(max),
        _ => 0,
    };
    assert!(
        lost > 0,
        "store 12 must contribute for the scenario to be informative"
    );
    assert_eq!(before - after, lost);
    assert_eq!(base.network_capacity(0), before);
}
