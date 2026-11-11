//! R5: an `Intake` holds local positions (the pointer table) looked up once
//! under one `MapTable` frame. D-DT64-6 D8 says such a held copy is valid only
//! while its frame is current: a relation-map move must discard it.
//!
//! Control (a): the table has not advanced, so the intake is reused.
//! Control (b): the relation map advances (source revision, tiles, tuples and
//! row order unchanged); reusing the old intake must be refused. Today it is
//! answered, because `MapTable::map` replays every frame the table issued.

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::{Bounds, Intake};

const TILE: u32 = 0;
const SOURCE: u64 = 1;
const R: [(u32, u32); 2] = [(10, 1), (11, 2)];
const S: [(u32, u32); 3] = [(1, 20), (2, 21), (1, 22)];

fn table() -> MapTable {
    let mut t = MapTable::new(SOURCE, 1);
    t.advance(
        SOURCE,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap()))],
    );
    t
}

#[test]
fn an_intake_is_reused_while_its_frame_is_current() {
    let table = table();
    let frame = table.seal();
    let intake = Intake::new(&table, &frame, TILE, &R, &S).unwrap();
    let first = intake.bag_count(&table, Bounds::default()).unwrap();
    // Non-vacuous: Y=1 has 1 left x 2 right partners, Y=2 has 1 x 1.
    assert_eq!(first[..3], [2, 1, 0]);
    assert_eq!(table.revisions(), frame.revisions());
    let again = intake.bag_count(&table, Bounds::default()).unwrap();
    assert_eq!(again, first);
    // The same artifact: still read through the layout it was built under.
    assert_eq!(
        intake.map(&table).unwrap().digest(),
        table.map(&frame, TILE).unwrap().digest()
    );
}

#[test]
fn an_intake_is_refused_after_the_relation_map_moves() {
    let mut table = table();
    let frame = table.seal();
    let intake = Intake::new(&table, &frame, TILE, &R, &S).unwrap();
    assert!(intake.bag_count(&table, Bounds::default()).is_ok());

    // M1 -> M2. Same source revision, no tile change, same tuples.
    table.advance(SOURCE, 2, vec![]);
    let (then, now) = (frame.revisions(), table.revisions());
    assert_eq!(then.source, now.source, "fixture: source unchanged");
    assert_eq!(then.coordinates, now.coordinates, "fixture: layout unchanged");
    assert_ne!(then.relation_map, now.relation_map, "fixture: map moved");

    let reused = intake.bag_count(&table, Bounds::default());
    assert!(
        reused.is_err(),
        "an intake made under relation map {} answered under relation map {}: {:?}",
        then.relation_map,
        now.relation_map,
        reused
    );
}
