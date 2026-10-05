//! `Destination { space, ordinal }` — bound computation PR 2.
//!
//! A destination is harvested from CAM: it names an ordinal that already
//! exists in a field's codebook. It is never minted here, it holds no text,
//! and renaming the ordinal's label does not change it.

use lance_graph_report::boundary::{BoundaryCounters, CamLabels, Destination, MemKv};
use lance_graph_report::ids::FieldId;

const REGION: FieldId = FieldId(3);

fn seeded() -> (CamLabels, MemKv) {
    let mut cam = CamLabels::default();
    let mut kv = MemKv::default();
    cam.canonicalize(REGION, ["North", "South", "North", "East"], &mut kv);
    (cam, kv)
}

#[test]
fn a_destination_is_built_from_an_existing_ordinal() {
    let (cam, _kv) = seeded();
    assert_eq!(cam.domain(REGION), 3);

    let d = cam.destination(REGION, 1).expect("ordinal 1 exists");
    assert_eq!(d.space(), REGION);
    assert_eq!(d.ordinal(), 1);
}

#[test]
fn an_ordinal_outside_the_domain_is_refused() {
    let (cam, _kv) = seeded();
    // Two-sided: the last ordinal is accepted, the next one is not.
    assert!(cam.destination(REGION, 2).is_some());
    assert_eq!(cam.destination(REGION, 3), None);
    assert_eq!(cam.destination(REGION, u32::MAX), None);
}

#[test]
fn an_unknown_space_is_refused() {
    let (cam, _kv) = seeded();
    assert_eq!(cam.destination(FieldId(99), 0), None);
}

#[test]
fn a_label_lookup_names_the_same_destination_and_never_mints() {
    let (cam, _kv) = seeded();
    let inserts = BoundaryCounters::get(&cam.counters.cam_insertions);

    let by_label = cam.destination_of(REGION, "South").expect("South exists");
    assert_eq!(Some(by_label), cam.destination(REGION, 1));

    assert_eq!(
        cam.destination_of(REGION, "West"),
        None,
        "unknown label is refused"
    );
    assert_eq!(cam.domain(REGION), 3, "no ordinal was minted");
    assert_eq!(
        BoundaryCounters::get(&cam.counters.cam_insertions),
        inserts,
        "no CAM insertion"
    );
}

#[test]
fn renaming_the_label_does_not_change_the_destination() {
    let (mut cam, mut kv) = seeded();
    let before = cam.destination_of(REGION, "North").expect("North exists");

    assert!(cam.rename(REGION, before.ordinal(), "Nordland", &mut kv));

    assert_eq!(cam.destination_of(REGION, "Nordland"), Some(before));
    assert_eq!(cam.destination(REGION, before.ordinal()), Some(before));
    assert_eq!(
        cam.destination_of(REGION, "North"),
        None,
        "old label no longer resolves"
    );
    assert_eq!(cam.domain(REGION), 3);
}

#[test]
fn a_destination_is_two_fixed_width_words_and_no_text() {
    assert_eq!(core::mem::size_of::<Destination>(), 8);
    let (cam, _kv) = seeded();
    let d = cam.destination(REGION, 0).unwrap();
    let copy = d; // Copy, no clone of any payload
    assert_eq!(d, copy);
}
