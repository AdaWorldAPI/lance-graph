//! Integration tests for `bound`: a thought bound to a DT64 `LocalMap`.

use std::collections::BTreeSet;

use lance_graph_dt64::dr::{LocalMap, Width};
use thought_git_land_lab::bound::{Admitted, BoundThought, MapStamp, SpaceRefusal};
use thought_git_probe::fixtures::{Outcome, cell};

/// `n` distinct ids in a deterministic, unsorted order, all above 2^40 so a
/// small id is smaller than every one of them.
fn ids(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let id = (1u64 << 40) + (state >> 32);
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

fn map(width: Width, ids: &[u64]) -> LocalMap {
    LocalMap::prepare(width, ids).expect("ids are distinct and fit the tile")
}

fn bound(map: &LocalMap) -> BoundThought {
    BoundThought::new(
        1,
        map,
        8,
        vec![(cell(3, 2), 30), (cell(39, 7), 31)],
        Outcome::Commit,
    )
    .expect("writes are inside the map")
}

/// FAILS IF: a thought is refused under the very map it was written under, or
/// `admit` returns a thought other than the one built.
#[test]
fn a_thought_is_admitted_under_its_own_map() {
    let base = ids(40, 1);
    let m = map(Width::W64, &base);
    let t = bound(&m);
    assert_eq!(t.stamp().admits(&m), Ok(Admitted::Same));
    let got = t.admit(&m).expect("admitted");
    assert_eq!(got.writes, vec![(cell(3, 2), 30), (cell(39, 7), 31)]);
    assert_eq!(got.id, 1);
}

/// FAILS IF: an append is refused, the appended count is wrong, or appending
/// an id smaller than every existing id is treated as a renumbering.
#[test]
fn a_pure_append_is_admitted() {
    let base = ids(40, 2);
    let m1 = map(Width::W64, &base);
    let t = bound(&m1);
    let mut longer = base.clone();
    longer.extend(ids(6, 99));
    longer.push(5); // smaller than every existing id
    assert_eq!(longer.len(), 47);
    let m2 = map(Width::W64, &longer);
    assert_eq!(
        t.stamp().admits(&m2),
        Ok(Admitted::Extended { appended: 7 })
    );
    assert!(t.admit(&m2).is_ok());
}

/// FAILS IF: a swap inside the written prefix is admitted, or a swap beyond
/// the written length is refused (only the written prefix matters).
#[test]
fn a_reorder_is_refused() {
    let base = ids(40, 3);
    let t = bound(&map(Width::W64, &base));

    let mut swapped = base.clone();
    swapped.swap(3, 17);
    assert_eq!(
        t.stamp().admits(&map(Width::W64, &swapped)),
        Err(SpaceRefusal::Reordered)
    );

    let mut ext_a = base.clone();
    ext_a.extend(ids(7, 77));
    let mut ext_b = ext_a.clone();
    ext_b.swap(45, 46);
    assert_ne!(ext_a, ext_b);
    for ext in [&ext_a, &ext_b] {
        assert_eq!(
            t.stamp().admits(&map(Width::W64, ext)),
            Ok(Admitted::Extended { appended: 7 })
        );
    }
}

/// FAILS IF: a map with fewer ids than the thought was written under is
/// admitted, or the refusal reports the wrong lengths.
#[test]
fn a_removal_is_refused() {
    let base = ids(40, 4);
    let t = bound(&map(Width::W64, &base));
    let shorter = map(Width::W64, &base[..39]);
    assert_eq!(
        t.stamp().admits(&shorter),
        Err(SpaceRefusal::Shrunk {
            written: 40,
            current: 39
        })
    );
}

/// FAILS IF: the same ids prepared for another tile width are admitted.
#[test]
fn a_width_change_is_refused() {
    let base = ids(40, 5);
    let t = bound(&map(Width::W64, &base));
    assert_eq!(
        t.stamp().admits(&map(Width::W256, &base)),
        Err(SpaceRefusal::WidthChanged {
            written: Width::W64,
            current: Width::W256
        })
    );
}

/// FAILS IF: a map of the same length with one id replaced is admitted.
#[test]
fn a_replaced_id_is_refused() {
    let base = ids(40, 6);
    let t = bound(&map(Width::W64, &base));
    let mut replaced = base.clone();
    replaced[10] = 7; // fresh, not in the original
    assert_eq!(
        t.stamp().admits(&map(Width::W64, &replaced)),
        Err(SpaceRefusal::Reordered)
    );
}

/// FAILS IF: a write outside the map, an out-of-range lane or a repeated cell
/// panics or is accepted, or the boundary cell (last row, last lane) is
/// refused.
#[test]
fn writes_outside_the_map_are_refused_not_panicked() {
    let m = map(Width::W64, &ids(40, 7));
    let build = |writes| BoundThought::new(2, &m, 8, writes, Outcome::Commit);

    assert_eq!(
        build(vec![(cell(40, 0), 1)]).unwrap_err(),
        SpaceRefusal::RowOutOfMap { row: 40, len: 40 }
    );
    assert_eq!(
        build(vec![(cell(0, 8), 1)]).unwrap_err(),
        SpaceRefusal::LaneOutOfRange { lane: 8, lanes: 8 }
    );
    assert_eq!(
        build(vec![(cell(1, 1), 1), (cell(1, 1), 2)]).unwrap_err(),
        SpaceRefusal::DuplicateCell(cell(1, 1))
    );
    let ok = build(vec![(cell(39, 7), 9)]).expect("last row, last lane");
    assert_eq!(ok.lanes(), 8);
}

/// FAILS IF: the bound space is capped at the fixture's 12 x 4 space, so a
/// write at row 30, lane 5 is refused (`Thought::new` would panic there).
#[test]
fn the_bound_space_is_not_the_fixture_space() {
    let m = map(Width::W64, &ids(40, 8));
    let t = BoundThought::new(3, &m, 8, vec![(cell(30, 5), 42)], Outcome::Commit)
        .expect("inside the map");
    let got = t.admit(&m).expect("admitted");
    assert!(got.writes.contains(&(cell(30, 5), 42)));
    assert_eq!(t.stamp(), MapStamp::of(&m));
}
