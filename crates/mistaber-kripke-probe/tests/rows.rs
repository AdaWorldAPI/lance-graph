//! The HHTL16 node rows: identity and parents as addresses, classid on the
//! lanes, and the refusals that name their population.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use lance_graph_contract::canonical_node::NodeRow;
use mistaber_kripke_probe::rows::{
    COLUMNS, Concepts, MAX_PARENTS, Population, UNBOUND, address, columns, identity, write_columns,
};
use mistaber_kripke_probe::{Golden, Kb, Pop, Refusal, intake, intake_under, parse_golden};

fn goldens() -> Vec<Golden> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens");
    let mut out: Vec<Golden> = fs::read_dir(&dir)
        .expect("goldens/")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .map(|p| parse_golden(&fs::read_to_string(p).expect("read")).expect("parse"))
        .collect();
    out.sort_by(|a, b| a.fixture_id.cmp(&b.fixture_id));
    out
}

fn fact(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_owned()).collect()
}

fn f1() -> Golden {
    goldens()
        .into_iter()
        .find(|g| g.fixture_id == "F1-modes")
        .expect("F1-modes")
}

/// FAILS IF identity or parents are not written to the rails, are dropped or
/// extra, or x and y are swapped.
#[test]
fn rows_round_trip_identity_and_parents() {
    let all = goldens();
    assert_eq!(all.len(), 11);
    let mut parent_columns = 0;
    for g in &all {
        let (i, names) = intake(&g.facts).expect(&g.fixture_id);
        for (pop, p) in [
            (Pop::World, &i.worlds),
            (Pop::Proposition, &i.props),
            (Pop::Rule, &i.rules),
        ] {
            for (o, row) in p.rows.iter().enumerate() {
                let id = identity(&p.lanes, row);
                assert_eq!(id, address(o as u16), "{} {pop:?} {o}", g.fixture_id);
                assert_ne!(id, UNBOUND);
                assert_eq!(
                    names.address_of(pop, names.name(pop, id)),
                    Some(id),
                    "{} {pop:?} {o}",
                    g.fixture_id
                );
                let r0 = p.lanes.get(row, 0).expect("rail 0");
                let r1 = p.lanes.get(row, 1).expect("rail 1");
                assert_eq!(p.lanes.get(row, 2), None);
                let cols = columns(&p.lanes, row);
                assert_eq!(cols.len(), COLUMNS);
                for (k, c) in cols.iter().enumerate() {
                    assert_eq!(r0.0[k], c.x(), "{} {pop:?} {o} x{k}", g.fixture_id);
                    assert_eq!(r1.0[k], c.y(), "{} {pop:?} {o} y{k}", g.fixture_id);
                }
                let held: BTreeSet<u16> = cols[1..]
                    .iter()
                    .filter(|a| **a != UNBOUND)
                    .map(|a| a.code())
                    .collect();
                let want: BTreeSet<u16> = if pop == Pop::World {
                    let me = names.name(Pop::World, id);
                    g.facts
                        .iter()
                        .filter(|f| f[0] == "accessible" && f[1] == me)
                        .map(|f| names.address_of(Pop::World, &f[2]).expect("parent").code())
                        .collect()
                } else {
                    BTreeSet::new()
                };
                assert_eq!(held, want, "{} {pop:?} {o}", g.fixture_id);
                // The unbound tail is contiguous: nothing past the parents.
                let bound = cols[1..].iter().filter(|a| **a != UNBOUND).count();
                assert_eq!(bound, want.len());
                assert!(cols[1 + bound..].iter().all(|a| *a == UNBOUND));
                parent_columns += held.len();
            }
        }
    }
    // Anti-vacuity: some world really has parents.
    assert!(parent_columns > 0);
}

fn parent_fixture(parents: usize) -> Vec<Vec<String>> {
    let mut facts = vec![fact(&["world", "c"])];
    for i in 0..parents {
        let w = format!("w{i}");
        facts.push(fact(&["world", &w]));
        facts.push(fact(&["accessible", "c", &w]));
        facts.push(fact(&["asserts", &w, "p"]));
    }
    facts
}

/// FAILS IF a 16th parent is truncated instead of refused, or if 15 are not
/// all carried.
#[test]
fn a_16th_parent_is_refused() {
    assert_eq!(MAX_PARENTS, 15);
    let (kb, names) = Kb::load(&parent_fixture(15)).expect("15 parents fit");
    let atoms = kb
        .solve()
        .materialize_atoms(&names, mistaber_kripke_probe::Projection::Set);
    let via = atoms
        .iter()
        .filter(|a| a.starts_with("holds_via(inherited(") && a.ends_with(",p,c)"))
        .count();
    assert_eq!(via, 15);
    assert_eq!(
        Kb::load(&parent_fixture(16)).err(),
        Some(Refusal::TooManyParents { parents: 16 })
    );
}

/// FAILS IF a classid reaches the key, the edges or the rails, or the lanes
/// do not carry the concept they were resolved under.
#[test]
fn the_classid_lives_on_the_lanes_not_in_the_rows() {
    let other = Concepts {
        world: 0x7F21,
        proposition: 0x7F22,
        rule: 0x7F23,
    };
    let g = f1();
    let (a, _) = intake_under(&g.facts, Concepts::LAB).unwrap();
    let (b, _) = intake_under(&g.facts, other).unwrap();
    let pops = |i: &mistaber_kripke_probe::Intake| {
        [i.worlds.rows.len(), i.props.rows.len(), i.rules.rows.len()]
    };
    assert_eq!(pops(&a), pops(&b));
    let pairs: [(&Population, &Population, u16, u16); 3] = [
        (&a.worlds, &b.worlds, Concepts::LAB.world, other.world),
        (
            &a.props,
            &b.props,
            Concepts::LAB.proposition,
            other.proposition,
        ),
        (&a.rules, &b.rules, Concepts::LAB.rule, other.rule),
    ];
    for (pa, pb, ca, cb) in pairs {
        assert_eq!(pa.lanes.concept(), ca);
        assert_eq!(pb.lanes.concept(), cb);
        assert_ne!(ca, cb);
        assert!(!pa.rows.is_empty());
        let r0 = pa.lanes.rail_range(0).unwrap();
        let r1 = pa.lanes.rail_range(1).unwrap();
        for (ra, rb) in pa.rows.iter().zip(&pb.rows) {
            assert_eq!(ra.key.as_bytes(), rb.key.as_bytes());
            assert_eq!(ra.edges.as_bytes(), rb.edges.as_bytes());
            assert_eq!(ra.value, rb.value);
            assert_eq!(ra.key.as_bytes(), &[0u8; 16]);
            assert_eq!(ra.edges.as_bytes(), &[0u8; 16]);
            for (at, byte) in ra.value.iter().enumerate() {
                if !r0.contains(&at) && !r1.contains(&at) {
                    assert_eq!(*byte, 0, "value byte {at}");
                }
            }
        }
    }
    for (c, name) in [(Concepts::LAB, "lab"), (other, "other")] {
        let (i, names) = intake_under(&g.facts, c).unwrap();
        let kb = Kb::assemble(i).expect(name);
        let got = kb
            .solve()
            .materialize_atoms(&names, mistaber_kripke_probe::Projection::Set);
        let mut want = g.atoms.clone();
        want.sort();
        assert_eq!(got, want, "{name}");
    }
}

/// FAILS IF the 64 limit is enforced anywhere but the local map, or a
/// duplicate or an unknown address is not named by its population.
#[test]
fn local_map_refusals_name_their_population() {
    let facts: Vec<Vec<String>> = (0..65)
        .map(|i| fact(&["world", &format!("w{i}")]))
        .collect();
    let (i, _) = intake(&facts).expect("intake does not enforce 64");
    assert_eq!(i.worlds.rows.len(), 65);
    assert_eq!(Kb::assemble(i).err(), Some(Refusal::TooMany("world")));

    let g = f1();
    let (mut i, _) = intake(&g.facts).unwrap();
    let again: NodeRow = i.worlds.rows[0];
    i.worlds.rows.push(again);
    assert_eq!(Kb::assemble(i).err(), Some(Refusal::Duplicate("world")));

    let (mut i, _) = intake(&g.facts).unwrap();
    let lanes = i.worlds.lanes;
    write_columns(&lanes, &mut i.worlds.rows[0], &[address(0), address(200)]);
    assert_eq!(Kb::assemble(i).err(), Some(Refusal::Unmapped("world")));
}
