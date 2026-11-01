//! DR addressing in `lance_graph_dt64::dr`: the frame witness and guards
//! (lifted from `dr-math-binding-probe`'s binding tests), the global → local
//! map and addressing at both tile widths, and the module boundary.

use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_dt64::dr::{
    address, coordinates, rotate_yz, Focus, FrameRefusal, FrameRevisions, FrameStore, GuardRefusal,
    Guards, LocalMap, MapRefusal, Scope, Width,
};

fn revisions() -> FrameRevisions {
    FrameRevisions {
        source: 7,
        relation_map: 11,
        coordinates: 3,
    }
}
const FOCUS: Focus = Focus { revision: 5 };
const SCOPE: Scope = Scope { id: 100 };

fn guards(store: &FrameStore) -> Guards {
    Guards {
        frame: store.seal(),
        focus: FOCUS,
        scope: SCOPE,
    }
}

/// Each moved revision is its own refusal, and a re-seal at equal numbers is
/// still a new frame.
#[test]
fn every_frame_revision_is_a_separate_refusal() {
    let r = revisions();
    let cases = [
        (
            FrameRevisions { source: 8, ..r },
            FrameRefusal::SourceChanged,
        ),
        (
            FrameRevisions {
                relation_map: 12,
                ..r
            },
            FrameRefusal::MapChanged,
        ),
        (
            FrameRevisions {
                coordinates: 4,
                ..r
            },
            FrameRefusal::CoordinatesChanged,
        ),
        (r, FrameRefusal::FrameResealed),
    ];
    for (moved, expected) in cases {
        let mut store = FrameStore::new(r);
        let frame = store.seal();
        assert_eq!(store.check(&frame), Ok(()));
        store.advance(moved);
        assert_eq!(store.check(&frame), Err(expected));
    }
}

/// The seal is store-issued identity, not equality of numbers.
#[test]
fn two_stores_at_equal_revisions_do_not_alias() {
    let first = FrameStore::new(revisions());
    let second = FrameStore::new(revisions());
    assert_eq!(first.revisions(), second.revisions());
    assert!(!first.seal().same_issue(&second.seal()));
    assert!(first.seal().same_issue(&first.seal()));
    assert_eq!(first.check(&second.seal()), Err(FrameRefusal::ForeignFrame));
}

/// Focus is motion and scope is policy: each is its own guard, and moving
/// either leaves the frame valid.
#[test]
fn focus_and_scope_are_guards_apart_from_the_frame() {
    let store = FrameStore::new(revisions());
    let g = guards(&store);
    assert_eq!(
        g.check(&store, (Focus { revision: 6 }, SCOPE)),
        Err(GuardRefusal::FocusChanged)
    );
    assert_eq!(
        g.check(&store, (FOCUS, Scope { id: 101 })),
        Err(GuardRefusal::ScopeChanged)
    );
    assert_eq!(store.check(&g.frame), Ok(()));
    assert_eq!(g.check(&store, (FOCUS, SCOPE)), Ok(()));
    let other = FrameStore::new(revisions());
    assert_eq!(
        g.check(&other, (FOCUS, SCOPE)),
        Err(GuardRefusal::Frame(FrameRefusal::ForeignFrame))
    );
}

/// The local map is per tile and takes the tile's width: a DR64 tile refuses
/// 65 identities, a DR256 tile holds 200. Lookups are counted; the inverse
/// read is not.
#[test]
fn a_local_map_holds_one_tile_of_either_width() {
    let ids: Vec<u64> = (0..200).map(|i| 1_000 + 7 * i).collect();
    assert_eq!(
        LocalMap::prepare(Width::W64, &ids[..65]).unwrap_err(),
        MapRefusal::TooMany
    );
    assert!(LocalMap::prepare(Width::W64, &ids[..64]).is_ok());

    let map = LocalMap::prepare(Width::W256, &ids).unwrap();
    assert_eq!(map.len(), 200);
    assert_eq!(map.local(1_000 + 7 * 199), Ok(199));
    assert_eq!(map.local(3), Err(MapRefusal::Unmapped));
    assert_eq!(map.lookups(), 2);
    assert_eq!(map.global(199), Some(1_000 + 7 * 199));
    assert_eq!(map.lookups(), 2, "the inverse is the table read, uncounted");

    assert_eq!(
        LocalMap::prepare(Width::W256, &[5, 9, 5]).unwrap_err(),
        MapRefusal::Duplicate
    );
    // The digest names the assignment: same ids in the same order agree,
    // another order does not.
    let a = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let b = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let c = LocalMap::prepare(Width::W64, &[3, 2, 1]).unwrap();
    assert_eq!(a.digest(), b.digest());
    assert_ne!(a.digest(), c.digest());
}

/// Addressing reads through the contract's Morton code at the tile's width:
/// (200, 17) is a DR256 cell and outside a DR64 tile. Rotation swaps the
/// axes and is its own inverse.
#[test]
fn addresses_follow_the_tile_width_and_rotation_is_an_involution() {
    let code = address(Width::W256, 200, 17).unwrap();
    assert_eq!(code, Morton8x8::from_xy(200, 17));
    assert_eq!(coordinates(Width::W256, code), Ok((200, 17)));
    assert_eq!(address(Width::W64, 200, 17), Err(MapRefusal::OutOfTile));
    assert_eq!(coordinates(Width::W64, code), Err(MapRefusal::OutOfTile));
    assert_eq!(
        address(Width::W64, 63, 63).map(|c| (c.x(), c.y())),
        Ok((63, 63))
    );

    let rotated = rotate_yz(code);
    assert_eq!((rotated.x(), rotated.y()), (17, 200));
    assert_eq!(rotate_yz(rotated), code);
}

/// "DR routes, DT interprets": the DR module reads nothing from the binding
/// side of the crate.
#[test]
fn dr_does_not_depend_on_the_binding_side() {
    let src = include_str!("../src/dr.rs");
    assert!(
        !src.contains("crate::"),
        "dr.rs must not name the crate root"
    );
    assert!(!src.contains("super::"), "dr.rs must not name its parent");
}

// ------------------------------------------- MapTable (D-DT64-7 O5, §1e)

use lance_graph_dt64::dr::{MapRead, MapTable};

fn map(ids: &[u64]) -> LocalMap {
    LocalMap::prepare(Width::W64, ids).unwrap()
}

#[test]
fn a_map_is_read_only_under_the_frame_it_was_sealed_under() {
    let mut t = MapTable::new(7, 11);
    t.advance(7, 11, vec![(1, Some(map(&[10, 20, 30])))]);
    let old = t.seal();
    assert_eq!(t.map(&old, 1).unwrap().local(20), Ok(1));
    t.advance(7, 11, vec![(1, Some(map(&[30, 20, 10])))]);
    assert_eq!(
        t.map(&old, 1).unwrap_err(),
        MapRead::Stale(FrameRefusal::CoordinatesChanged)
    );
    let now = t.seal();
    assert_eq!(t.map(&now, 1).unwrap().local(30), Ok(0));
    assert_eq!(t.map(&now, 2).unwrap_err(), MapRead::NoSuchTile(2));
}

#[test]
fn an_advance_that_changes_nothing_still_reseals() {
    let mut t = MapTable::new(7, 11);
    t.advance(7, 11, vec![(1, Some(map(&[10, 20])))]);
    let old = t.seal();
    let before = t.revisions();
    t.advance(7, 11, vec![]);
    assert_eq!(t.revisions(), before, "nothing moved");
    assert_eq!(
        t.map(&old, 1).unwrap_err(),
        MapRead::Stale(FrameRefusal::FrameResealed)
    );
}

#[test]
fn the_coordinates_revision_is_a_function_of_the_held_maps() {
    let at = |tiles: Vec<(u32, &[u64])>| {
        let mut t = MapTable::new(7, 11);
        t.advance(
            7,
            11,
            tiles.into_iter().map(|(k, v)| (k, Some(map(v)))).collect(),
        );
        t.revisions().coordinates
    };
    let base = at(vec![(1, &[10, 20]), (2, &[30, 40])]);
    assert_eq!(base, at(vec![(1, &[10, 20]), (2, &[30, 40])]), "same maps");
    assert_ne!(base, at(vec![(1, &[20, 10]), (2, &[30, 40])]), "reordered");
    assert_ne!(
        base,
        at(vec![(1, &[30, 40]), (2, &[10, 20])]),
        "tiles swapped"
    );
    assert_ne!(base, at(vec![(1, &[10, 20])]), "tile removed");
}

#[test]
fn a_removed_tile_is_gone_and_another_tables_frame_is_foreign() {
    let mut a = MapTable::new(7, 11);
    a.advance(7, 11, vec![(1, Some(map(&[10]))), (2, Some(map(&[20])))]);
    a.advance(7, 11, vec![(2, None)]);
    let f = a.seal();
    assert!(a.map(&f, 1).is_ok());
    assert_eq!(a.map(&f, 2).unwrap_err(), MapRead::NoSuchTile(2));

    let mut b = MapTable::new(7, 11);
    b.advance(7, 11, vec![(1, Some(map(&[10])))]);
    assert_eq!(a.revisions(), b.revisions(), "equal numbers");
    assert_eq!(
        a.map(&b.seal(), 1).unwrap_err(),
        MapRead::Stale(FrameRefusal::ForeignFrame)
    );
}
