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

/// RE-PINNED (D-DR-VMAP). Was: an old frame was refused with
/// `Stale(CoordinatesChanged)` after the table advanced. Now the table is an
/// append-only log and a frame carries its version, so the old frame still
/// reads the layout it was sealed under, and the new frame reads the new one.
#[test]
fn a_map_is_read_under_the_version_its_frame_was_sealed_at() {
    let mut t = MapTable::new(7, 11);
    t.advance(7, 11, vec![(1, Some(map(&[10, 20, 30])))]);
    let old = t.seal();
    assert_eq!(t.map(&old, 1).unwrap().local(20), Ok(1));
    t.advance(7, 11, vec![(1, Some(map(&[30, 20, 10])))]);
    // Old assertion, now false: Stale(CoordinatesChanged).
    assert_eq!(t.map(&old, 1).unwrap().local(20), Ok(1));
    assert_eq!(t.map(&old, 1).unwrap().local(30), Ok(2), "old layout");
    let now = t.seal();
    assert_eq!(t.map(&now, 1).unwrap().local(30), Ok(0), "new layout");
    assert_eq!(t.map(&now, 2).unwrap_err(), MapRead::NoSuchTile(2));
}

/// RE-PINNED (D-DR-VMAP). Was: the old frame was refused with
/// `Stale(FrameResealed)`. A reseal is still a distinct frame (identity), but
/// it is no longer a refusal: the old frame reads, `FrameStore::check` still
/// calls it non-current.
#[test]
fn an_advance_that_changes_nothing_still_reseals() {
    let mut t = MapTable::new(7, 11);
    t.advance(7, 11, vec![(1, Some(map(&[10, 20])))]);
    let old = t.seal();
    let before = t.revisions();
    t.advance(7, 11, vec![]);
    assert_eq!(t.revisions(), before, "nothing moved");
    let new = t.seal();
    assert!(!old.same_issue(&new), "a reseal is a new frame");
    assert!(new.version() > old.version());
    // Old assertion, now false: Stale(FrameResealed).
    assert_eq!(t.map(&old, 1).unwrap().local(20), Ok(1));
    assert_eq!(t.map(&new, 1).unwrap().local(20), Ok(1));
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
    // Same maps in the same order under another tile name: only the tile id
    // in the digest tells these apart.
    assert_ne!(
        base,
        at(vec![(1, &[10, 20]), (3, &[30, 40])]),
        "tile renamed"
    );
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

// ------------------------------------------- locate / extends (D-CODEGRAPH-0 wide closure)

mod locate {
    use lance_graph_dt64::dr::{LocalMap, LocateRefusal, MapRead, MapTable, TileId, Width};

    fn table(tiles: Vec<(TileId, Vec<u64>)>) -> MapTable {
        let mut t = MapTable::new(1, 1);
        let changes = tiles
            .into_iter()
            .map(|(id, g)| (id, LocalMap::prepare(Width::W64, &g).ok()))
            .collect();
        t.advance(1, 1, changes);
        t
    }

    /// Fails if the ordinal is the tile id instead of the index in the list,
    /// or the local is not the position within the holding tile.
    #[test]
    fn locate_finds_ordinal_and_local_across_two_tiles() {
        let t = table(vec![(7, vec![10, 20, 30]), (3, vec![40, 50])]);
        let f = t.seal();
        assert_eq!(t.locate(&f, &[7, 3], 20), Ok((0, 1)));
        assert_eq!(t.locate(&f, &[7, 3], 50), Ok((1, 1)));
        // List order defines the ordinal.
        assert_eq!(t.locate(&f, &[3, 7], 50), Ok((0, 1)));
    }

    /// Fails if the first hit wins instead of the overlap being refused.
    #[test]
    fn locate_refuses_an_id_in_two_tiles() {
        let t = table(vec![(1, vec![5, 6]), (2, vec![6, 7])]);
        let f = t.seal();
        assert_eq!(t.locate(&f, &[1, 2], 6), Err(LocateRefusal::Ambiguous));
        // Non-vacuous: a listing that excludes the second tile resolves.
        assert_eq!(t.locate(&f, &[1], 6), Ok((0, 1)));
    }

    /// Fails if an absent id resolves to anything, or a stale frame is read.
    #[test]
    fn locate_refuses_an_absent_id() {
        let t = table(vec![(1, vec![5, 6]), (2, vec![8])]);
        let f = t.seal();
        assert_eq!(t.locate(&f, &[1, 2], 99), Err(LocateRefusal::Unmapped));
        assert_eq!(
            t.locate(&f, &[1, 9], 5),
            Err(LocateRefusal::Read(MapRead::NoSuchTile(9)))
        );
    }

    fn map(width: Width, g: &[u64]) -> LocalMap {
        LocalMap::prepare(width, g).expect("fixture")
    }

    /// Fails if appending (even an id smaller than the existing ones) is not
    /// accepted, or if equality is not accepted as a trivial extension.
    #[test]
    fn extends_true_for_pure_prefix_extension() {
        let old = map(Width::W64, &[30, 20]);
        assert!(map(Width::W64, &[30, 20, 10]).extends(&old));
        assert!(old.extends(&old));
    }

    /// Fails if a reorder of the same ids counts as an extension.
    #[test]
    fn extends_false_for_reorder() {
        let old = map(Width::W64, &[1, 2]);
        assert!(!map(Width::W64, &[2, 1, 3]).extends(&old));
    }

    /// Fails if dropping an id counts as an extension.
    #[test]
    fn extends_false_for_removal() {
        let old = map(Width::W64, &[1, 2, 3]);
        assert!(!map(Width::W64, &[1, 2]).extends(&old));
        assert!(!map(Width::W64, &[1, 3]).extends(&old));
    }

    /// Fails if the width is ignored.
    #[test]
    fn extends_false_for_width_change() {
        let old = map(Width::W64, &[1, 2]);
        assert!(!map(Width::W256, &[1, 2, 3]).extends(&old));
    }
}

// ------------------------------------------- D-DR-VMAP falsifiers F1-F6

mod vmap {
    use super::map;
    use lance_graph_dt64::dr::{
        Axis, FrameRefusal, MapRead, MapRefusal, MapTable, RouteLog, RouteVersion, VersionClock,
        Width,
    };
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct Counting;

    thread_local! {
        // Const-initialised and drop-free, so touching it inside the
        // allocator never allocates. Per thread: parallel tests cannot
        // pollute the count.
        static ALLOCS: Cell<u64> = const { Cell::new(0) };
    }

    fn bump() {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
    }

    // SAFETY: forwards every call unchanged to the system allocator.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            bump();
            System.alloc(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            System.dealloc(p, l)
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            bump();
            System.realloc(p, l, n)
        }
    }

    #[global_allocator]
    static A: Counting = Counting;

    /// F1 replay. Fails (disable: `map` ignores `frame.version` and reads the
    /// last event) because f1 would then see the v2 layout.
    #[test]
    fn f1_every_id_replays_under_its_own_version() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30])))]);
        let f1 = t.seal();
        t.advance(1, 1, vec![(1, Some(map(&[30, 20, 10])))]);
        let f2 = t.seal();
        let v1 = [(10, 0), (20, 1), (30, 2)];
        let v2 = [(10, 2), (20, 1), (30, 0)];
        for (id, l) in v1 {
            assert_eq!(t.map(&f1, 1).unwrap().local(id), Ok(l));
        }
        for (id, l) in v2 {
            assert_eq!(t.map(&f2, 1).unwrap().local(id), Ok(l));
        }
        // Anti-vacuity: the two versions disagree on 10 and 30.
        assert_ne!(v1[0].1, v2[0].1);
        assert_ne!(v1[2].1, v2[2].1);
        // The numeric read agrees with the frame read.
        assert_eq!(t.diagnose_at(f1.version(), 1).unwrap().local(10), Ok(0));
        assert_eq!(t.diagnose_at(f2.version(), 1).unwrap().local(10), Ok(2));
    }

    /// F2 extension is a length at a version. Fails (disable: drop the
    /// `>= len` guard in `TileMap::local`) because 30 would resolve at v1;
    /// fails if an extension mints a generation.
    #[test]
    fn f2_an_extension_is_a_length_not_a_copy() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10, 20])))]);
        let f1 = t.seal();
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30])))]);
        let f2 = t.seal();
        let at1 = t.map(&f1, 1).unwrap();
        assert_eq!(at1.len(), 2);
        assert_eq!(at1.local(30), Err(MapRefusal::Unmapped));
        assert_eq!(at1.global(2), None);
        assert_eq!(at1.local(20), Ok(1));
        let at2 = t.map(&f2, 1).unwrap();
        assert_eq!(at2.len(), 3);
        assert_eq!(at2.local(30), Ok(2));
        assert_eq!(at2.global(2), Some(30));
        assert_eq!(t.generation_count(1), 1, "no generation for an extension");
        assert_eq!(t.event_count(1), 2, "one event per version");
        assert!(at2.extends(at1) && !at1.extends(at2));
    }

    /// F3 history survives a replacement and a removal. Fails (disable:
    /// `remove` deletes the tile instead of logging `None`) because f1 and f2
    /// would stop resolving.
    #[test]
    fn f3_a_removal_is_an_event_not_an_erasure() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30])))]);
        let f1 = t.seal();
        t.advance(1, 1, vec![(1, Some(map(&[30, 20, 10])))]);
        let f2 = t.seal();
        t.advance(1, 1, vec![(1, None)]);
        let f3 = t.seal();
        assert_eq!(t.map(&f1, 1).unwrap().local(10), Ok(0));
        assert_eq!(t.map(&f2, 1).unwrap().local(10), Ok(2));
        assert_eq!(t.map(&f3, 1).unwrap_err(), MapRead::NoSuchTile(1));
        assert_eq!(t.generation_count(1), 2, "replacement kept both layouts");
        // Re-adding after a removal is a new generation, and f3 stays gone.
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30])))]);
        assert_eq!(t.generation_count(1), 3);
        assert_eq!(t.map(&f3, 1).unwrap_err(), MapRead::NoSuchTile(1));
    }

    /// F4 a foreign frame is refused even at equal revisions and equal
    /// version. Fails (disable: `issued` skips the store identity check)
    /// because the foreign frame would read.
    #[test]
    fn f4_a_foreign_frame_is_refused_at_equal_version() {
        let mut a = MapTable::new(1, 1);
        let mut b = MapTable::new(1, 1);
        a.advance(1, 1, vec![(1, Some(map(&[10])))]);
        b.advance(1, 1, vec![(1, Some(map(&[10])))]);
        assert_eq!(a.revisions(), b.revisions());
        let fb = b.seal();
        assert_eq!(fb.version(), a.seal().version(), "equal versions");
        assert_eq!(
            a.map(&fb, 1).unwrap_err(),
            MapRead::Stale(FrameRefusal::ForeignFrame)
        );
        assert!(b.map(&fb, 1).is_ok(), "its own table reads it");
    }

    /// F5 a lookup allocates nothing. Fails (disable: `map` returns an owned
    /// prefix copy) because the counter would move.
    #[test]
    fn f5_map_and_local_allocate_nothing() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30])))]);
        let f1 = t.seal();
        t.advance(1, 1, vec![(1, Some(map(&[10, 20, 30, 40])))]);
        let f2 = t.seal();
        let before = ALLOCS.with(Cell::get);
        let mut hits = 0u32;
        for i in 0..10_000u64 {
            let frame = if i % 2 == 0 { &f1 } else { &f2 };
            if let Ok(view) = t.map(frame, 1) {
                if view.local(20 + (i % 2) * 20).is_ok() {
                    hits += 1;
                }
            }
        }
        let after = ALLOCS.with(Cell::get);
        assert_eq!(after - before, 0, "lookups allocated");
        // Anti-vacuity: the loop really resolved ids (20 under both, 40 only
        // under f2), so a skipped body cannot pass.
        assert_eq!(hits, 10_000 / 2 + 10_000 / 2);
        // And the counter can move: building a map allocates.
        let b2 = ALLOCS.with(Cell::get);
        let _ = map(&[1, 2, 3]);
        assert!(ALLOCS.with(Cell::get) > b2);
    }

    /// F6 silence twin: an advance with no changes adds no event and no
    /// generation, yet takes a version. Fails (disable: push an event per
    /// tile on every advance) because the counts would move.
    #[test]
    fn f6_an_empty_advance_adds_a_version_and_no_history() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10, 20]))), (2, Some(map(&[5])))]);
        let f_old = t.seal();
        let (g, e) = (t.generation_count(1), t.event_count(1));
        let (g2, e2) = (t.generation_count(2), t.event_count(2));
        let v = t.advance(1, 1, vec![]);
        let f_new = t.seal();
        assert_eq!((t.generation_count(1), t.event_count(1)), (g, e));
        assert_eq!((t.generation_count(2), t.event_count(2)), (g2, e2));
        assert_eq!(f_new.version(), v);
        assert!(v > f_old.version());
        assert!(!f_old.same_issue(&f_new));
        for tile in [1, 2] {
            assert_eq!(
                t.map(&f_old, tile).unwrap().digest(),
                t.map(&f_new, tile).unwrap().digest()
            );
        }
    }

    /// Q1: a RouteLog and a MapTable on one clock advance ONE sequence, so a
    /// frame's version also answers `routed_at`. Fails if the two keep
    /// separate counters (versions would collide / not interleave).
    #[test]
    fn a_shared_clock_makes_one_version_sequence() {
        let clock = VersionClock::new();
        let mut log = RouteLog::with_clock(clock.clone());
        let mut t = MapTable::with_clock(1, 1, clock.clone());
        let id = Axis::new(Width::W64, 1);
        let r1 = log.route(3, 9, id).unwrap();
        let v1 = t.advance(1, 1, vec![(1, Some(map(&[10])))]);
        let r2 = log.route(3, 8, id).unwrap();
        assert_eq!([r1.0, v1.0, r2.0], [1, 2, 3], "one interleaved sequence");
        let f = t.seal();
        assert_eq!(log.routed_at(f.version(), 3), Ok(Some(9)));
        assert_eq!(log.routed_at(r2, 3), Ok(Some(8)));
        assert_eq!(log.current_version(), t.current_version());
        // Separate clocks do not interleave (silence twin).
        let mut solo = RouteLog::new();
        assert_eq!(solo.route(3, 9, id).unwrap(), RouteVersion(1));
        // A clone of a log forks the clock.
        let mut copy = log.clone();
        copy.route(4, 1, id).unwrap();
        assert_eq!(log.current_version(), clock.current());
        assert!(copy.current_version() > log.current_version());
    }

    /// Q2: a version past the end is a typed refusal, not a clamp.
    #[test]
    fn diagnose_at_refuses_a_version_past_the_end() {
        let mut t = MapTable::new(1, 1);
        t.advance(1, 1, vec![(1, Some(map(&[10])))]);
        let cur = t.current_version();
        assert!(t.diagnose_at(cur, 1).is_ok());
        let future = RouteVersion(cur.0 + 1);
        assert_eq!(
            t.diagnose_at(future, 1).unwrap_err(),
            MapRead::FutureVersion {
                asked: future,
                current: cur
            }
        );
        assert_eq!(
            t.diagnose_at(RouteVersion(0), 1).unwrap_err(),
            MapRead::NoSuchTile(1),
            "before the tile existed"
        );
    }

    /// P9 (E1): a foreign frame's version, read numerically, must not yield an
    /// executable map. `map(frame)` refuses the frame; `diagnose_at` can still
    /// show the number's layout in THIS table, but only as a `MapDiagnostic`,
    /// which is not a `TileMap` and so cannot reach `MidMap` executors.
    #[test]
    fn p9_equal_version_foreign_frame_is_not_executable() {
        let mut a = MapTable::new(1, 1);
        let mut b = MapTable::new(1, 1);
        a.advance(1, 1, vec![(1, Some(map(&[10, 20])))]);
        b.advance(1, 1, vec![(1, Some(map(&[20, 10])))]);
        let fb = b.seal();
        assert_eq!(fb.version(), a.seal().version(), "equal versions");
        // The executable read refuses the foreign frame.
        assert_eq!(
            a.map(&fb, 1).unwrap_err(),
            MapRead::Stale(FrameRefusal::ForeignFrame)
        );
        // The issuer answers differently from A's layout at that number:
        // the numeric read in A is a different layout than B's frame means.
        assert_eq!(b.map(&fb, 1).unwrap().local(10), Ok(1));
        let d = a.diagnose_at(fb.version(), 1).unwrap();
        assert_eq!(d.local(10), Ok(0), "A's layout, not the frame's");
        assert_ne!(d.digest(), b.map(&fb, 1).unwrap().digest());
    }
}
