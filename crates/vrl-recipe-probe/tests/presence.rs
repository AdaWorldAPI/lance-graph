//! EXO-294 (decision B): a value is ABSENT when its presence bit is 0 and it
//! is 0, as in codec9 / DR512. An abort over an absent value cannot decide,
//! so the row is neither kept nor dropped as if the value were a real 0.

mod common;

use common::{all_fields, bind, witness, Source, LABELS, VIEW_CLASS};
use lance_graph_dt64::LaneRefusal;
use vrl_recipe_probe::{interpret_present, Event, Recipe, Refusal, Rows};

fn compile(src: &str) -> Recipe {
    Recipe::compile(src, &common::View::new(), VIEW_CLASS)
        .unwrap_or_else(|e| panic!("{src}: {e:?}"))
}

fn events(rows: &Rows<'_>) -> (Vec<Event>, Vec<Vec<String>>) {
    let ev = (0..rows.len())
        .map(|r| {
            rows.names()
                .iter()
                .enumerate()
                .map(|(c, n)| {
                    (
                        n.clone(),
                        i64::from(rows.get(r, c).as_i32().expect("an i32 column")),
                    )
                })
                .collect()
        })
        .collect();
    let absent = (0..rows.len())
        .map(|r| {
            rows.names()
                .iter()
                .enumerate()
                .filter(|(c, _)| !rows.present(r, *c))
                .map(|(_, n)| n.clone())
                .collect()
        })
        .collect();
    (ev, absent)
}

/// The 3-row case, by hand: amount is absent, a real 0, and 5.
/// `if .a < 3 { abort }` drops the real 0, keeps 5, and cannot decide the
/// absent row: one kept, one unknown. Read as a 0, the absent row would
/// have been dropped (two dropped, none unknown), which is what this rules
/// out.
#[test]
fn an_absent_value_is_neither_kept_nor_dropped() {
    let mut source = Source::from_lanes([vec![1, 2, 3], vec![0; 3], vec![0; 3], vec![9, 0, 5]]);
    source.set_absent(3, [0]);
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    let r = compile(".a = .properties.amount.value\nif .a < 3 { abort }");
    assert_eq!(r.count(&w).unwrap(), 1);
    assert_eq!(r.count_unknown(&w).unwrap(), 1);
    assert_eq!(r.sum(&w, "a").unwrap(), 5);
    let rows = r.materialize_rows(&w).unwrap();
    assert_eq!(rows.positions(), &[2]);
    // Anti-vacuity: the same lane with the row present at value 0 drops it.
    let all = Source::from_lanes([vec![1, 2, 3], vec![0; 3], vec![0; 3], vec![0, 0, 5]]);
    let wa = bind(&all, &all_fields());
    let wa = witness(&wa, &all);
    assert_eq!(
        (r.count(&wa).unwrap(), r.count_unknown(&wa).unwrap()),
        (1, 0)
    );
}

/// A decided, true abort drops the row even when another abort over an
/// absent value cannot decide: Kleene's `F AND U = F`.
#[test]
fn a_decided_true_abort_drops_a_row_another_abort_cannot_decide() {
    let mut source = Source::from_lanes([vec![1, 2], vec![2, 0], vec![0; 2], vec![0, 0]]);
    source.set_absent(3, [0, 1]);
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    // Row 0: status 2 > 1, decided and true, so dropped. Row 1: status 0,
    // decided and false; amount absent, so unknown.
    let r = compile(
        ".a = .properties.amount.value\nif .a < 3 { abort }\nif .properties.status.value > 1 { abort }",
    );
    assert_eq!((r.count(&w).unwrap(), r.count_unknown(&w).unwrap()), (0, 1));
}

/// Absent needs BOTH presence 0 and value 0. A presence-0 row holding a
/// nonzero value is a value (a negative under DR512's sign convention), so
/// it is decided like any other.
#[test]
fn presence_zero_with_a_nonzero_value_is_a_value() {
    let mut source = Source::from_lanes([vec![1, 2], vec![0; 2], vec![0; 2], vec![-4, 0]]);
    let plane = source.presence[3].as_mut().unwrap();
    plane[0] &= !0b11; // both rows presence 0; row 0 keeps -4, row 1 is 0
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    let r = compile(".a = .properties.amount.value\nif .a > 0 { abort }");
    // Row 0 (-4, presence 0) is decided: not > 0, kept. Row 1 is absent.
    assert_eq!((r.count(&w).unwrap(), r.count_unknown(&w).unwrap()), (1, 1));
    assert_eq!(r.sum(&w, "a").unwrap(), -4);
}

const PROGRAMS: &[&str] = &[
    ".n = .key\n.a = .properties.amount.value",
    ".a = .properties.amount.value\nif .a < 0 { abort }",
    ".a = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".a = .properties.amount.value\nif .properties.region.value == 0 { abort }\nif .a >= 200 { abort }",
    ".r = .properties.region.value\n.a = .properties.amount.value\nif .properties.status.value > 2 { abort }\nif .a <= 10 { abort }\nif .r != 1 { abort }",
];

/// At widths 1..1000 with about a fifth of status, region and amount absent,
/// and some presence-0 rows left holding a nonzero value, the recipe's kept
/// count, unknown count, sum, rows and per-cell presence equal the
/// presence-aware interpreter's.
#[test]
fn the_three_valued_recipe_equals_the_interpreter_at_every_width() {
    let (mut unknown_total, mut absent_kept) = (0, 0);
    for &n in &[1usize, 63, 64, 65, 300, 1000] {
        let mut source = Source::random(n, 7 + n as u64);
        for lane in 1..=3 {
            let rows: Vec<usize> = (0..n).filter(|r| (r * 7 + lane * 3) % 5 == 0).collect();
            source.set_absent(lane, rows);
            // Presence 0 with a nonzero value: still a value.
            for r in (0..n).filter(|r| (r * 11 + lane) % 13 == 0) {
                if source.lanes[lane][r] != 0 {
                    let plane = source.presence[lane].as_mut().unwrap();
                    plane[r / 64] &= !(1u64 << (r % 64));
                }
            }
        }
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        let lanes = source.i32_lanes();
        let presence = source.i32_presence();
        for src in PROGRAMS {
            let r = compile(src);
            let want = interpret_present(src, &LABELS[..4], &lanes, Some(&presence)).unwrap();
            let rows = r.materialize_rows(&w).unwrap();
            let (got, absent) = events(&rows);
            assert_eq!(got, want.events, "{src} at {n}: rows");
            assert_eq!(absent, want.absent, "{src} at {n}: absent cells");
            assert_eq!(
                r.count(&w).unwrap() as usize,
                want.events.len(),
                "{src} at {n}"
            );
            assert_eq!(
                r.count_unknown(&w).unwrap() as usize,
                want.unknown,
                "{src} at {n}"
            );
            if let Some(a) = rows.names().iter().position(|n| n == "a") {
                let sum: i64 = want.events.iter().map(|e| e["a"]).sum();
                assert_eq!(r.sum(&w, "a").unwrap(), sum, "{src} at {n}: sum");
                absent_kept += (0..rows.len()).filter(|&i| !rows.present(i, a)).count();
            }
            unknown_total += want.unknown;
        }
    }
    // Anti-vacuity: some rows are unknown, and some kept rows carry an
    // absent output.
    assert!(unknown_total > 100, "unknown rows: {unknown_total}");
    assert!(
        absent_kept > 50,
        "kept rows with an absent output: {absent_kept}"
    );
}

/// A field the recipe reads with no presence plane is refused, never read as
/// all present. A field the recipe does not read may lack one.
#[test]
fn a_read_field_without_a_presence_plane_is_refused() {
    let mut source = Source::random(64, 3);
    source.presence[3] = None;
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    assert_eq!(
        compile(".a = .properties.amount.value")
            .count(&w)
            .unwrap_err(),
        Refusal::Binding(LaneRefusal::PresenceUnbound { field: 3 })
    );
    assert_eq!(compile(".n = .key").count(&w).unwrap(), 64);
}

/// A presence plane that does not span the lane's rows is refused before
/// folding.
#[test]
fn a_presence_plane_of_the_wrong_length_is_refused() {
    let mut source = Source::random(130, 3);
    source.presence[3].as_mut().unwrap().pop();
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    assert_eq!(
        compile(".a = .properties.amount.value")
            .count(&w)
            .unwrap_err(),
        Refusal::PresenceLength {
            field: 3,
            words: 2,
            need: 3
        }
    );
}
