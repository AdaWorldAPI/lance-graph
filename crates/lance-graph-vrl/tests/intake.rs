//! A key lookup (XVERWEIS) as a recipe over an intake table: the table
//! resolves keys to positions, the recipe projects the fields, text cells
//! come back as codes and are decoded through the table's code table.

use lance_graph_intake::{IntakeRefusal, ShapeConfig, Table};
use lance_graph_pg_row::PgRowRefusal;
use lance_graph_vrl::{Datum, Recipe, Refusal, Witness};

const CLASS: u32 = 0x2B07_0001;

fn table() -> Table {
    let rows: Vec<Vec<Option<&str>>> = vec![
        vec![Some("1"), Some("north"), Some("10")],
        vec![Some("2"), Some("south"), Some("-20")],
        vec![Some("3"), Some("east"), None],
    ];
    let config = ShapeConfig {
        key: Some("id".to_owned()),
        ..ShapeConfig::default()
    };
    Table::from_rows("orders", CLASS, &["id", "region", "qty"], &rows, &config).unwrap()
}

fn witness(t: &Table) -> Witness<'_, lance_graph_intake::IntakeView, lance_graph_intake::Owner> {
    let w = t.witness();
    Witness {
        binding: w.binding,
        store: w.store,
        view: w.view,
        frames: w.frames,
        source: w.source,
    }
}

fn recipe(t: &Table, src: &str) -> Recipe {
    let w = t.witness();
    let class = lance_graph_dt64::view_class(w.binding.class());
    Recipe::compile(src, w.view, class).unwrap()
}

#[test]
fn a_lookup_reads_the_rows_of_its_keys_in_list_order() {
    let t = table();
    let r = recipe(
        &t,
        ".id = .properties.id.value\n.region = .properties.region.value\n.qty = .properties.qty.value",
    );
    let positions = t.live_positions_for_keys(&[3, 1, 3]).unwrap();
    let rows = r.rows_at(&witness(&t), &positions, &t.frame()).unwrap();
    assert_eq!(rows.len(), 3);
    let col = |n: &str| rows.names().iter().position(|x| x == n).unwrap();
    let (id, region, qty) = (col("id"), col("region"), col("qty"));
    let ids: Vec<i64> = (0..3)
        .map(|i| i64::from(rows.get(i, id).as_i32().unwrap()))
        .collect();
    assert_eq!(ids, [3, 1, 3]);
    let names: Vec<String> = (0..3)
        .map(|i| match rows.get(i, region) {
            Datum::U32(code) => t.code_key("region", code).unwrap(),
            other @ Datum::I32(_) => panic!("region is a text column, got {other:?}"),
        })
        .collect();
    assert_eq!(names, ["east", "north", "east"]);
    assert_eq!(rows.get(1, qty).as_i32(), Some(10));
    assert_eq!(rows.get(0, region).as_i32(), None, "a code is not a number");
    assert!(rows.present(1, qty));
    assert!(!rows.present(0, qty), "key 3 has no qty");
}

#[test]
fn positions_resolved_before_an_append_are_refused() {
    let mut t = table();
    let r = recipe(&t, ".region = .properties.region.value");
    let positions = t.live_positions_for_keys(&[2]).unwrap();
    let under = t.frame();
    assert!(r.rows_at(&witness(&t), &positions, &under).is_ok());
    t.append(&[vec![Some("4"), Some("west"), Some("5")]])
        .unwrap();
    assert!(matches!(
        r.rows_at(&witness(&t), &positions, &under)
            .map(|rows| rows.len()),
        Err(Refusal::Binding(lance_graph_dt64::LaneRefusal::Frame(_)))
    ));
    let fresh = t.live_positions_for_keys(&[2]).unwrap();
    assert_eq!(
        r.rows_at(&witness(&t), &fresh, &t.frame()).unwrap().len(),
        1
    );
}

#[test]
fn a_lookup_refuses_drop_terms_and_positions_past_the_lanes() {
    let t = table();
    let filtered = recipe(
        &t,
        ".q = .properties.qty.value\nif .properties.qty.value < 0 { abort }",
    );
    let positions = t.live_positions_for_keys(&[1]).unwrap();
    assert_eq!(
        filtered
            .rows_at(&witness(&t), &positions, &t.frame())
            .map(|rows| rows.len()),
        Err(Refusal::FiltersAtPositions)
    );
    let r = recipe(&t, ".q = .properties.qty.value");
    assert!(matches!(
        r.rows_at(&witness(&t), &[7], &t.frame())
            .map(|rows| rows.len()),
        Err(Refusal::PositionOutOfRange { at: 7, n_rows: 3 })
    ));
}

#[test]
fn a_text_column_is_projected_but_never_compared() {
    let t = table();
    let compared = recipe(
        &t,
        ".r = .properties.region.value\nif .properties.region.value == 1 { abort }",
    );
    assert!(matches!(
        compared.count(&witness(&t)),
        Err(Refusal::NotI32 { .. })
    ));
    let projected = recipe(&t, ".r = .properties.region.value");
    assert_eq!(projected.count(&witness(&t)).unwrap(), 3);
}

/// A deleted key is never read: its key is refused when it is resolved, and
/// positions resolved before the delete are refused under their old frame.
#[test]
fn a_deleted_key_is_never_read() {
    let mut t = table();
    let r = recipe(&t, ".region = .properties.region.value");
    let before = t.live_positions_for_keys(&[2]).unwrap();
    let under = t.frame();
    t.delete_key(2).unwrap();
    assert_eq!(
        t.live_positions_for_keys(&[2]),
        Err(IntakeRefusal::PgRow(PgRowRefusal::DeadPosition { at: 1 }))
    );
    assert!(matches!(
        r.rows_at(&witness(&t), &before, &under)
            .map(|rows| rows.len()),
        Err(Refusal::Binding(lance_graph_dt64::LaneRefusal::Frame(_)))
    ));
    let live = t.live_positions_for_keys(&[1, 3]).unwrap();
    assert_eq!(r.rows_at(&witness(&t), &live, &t.frame()).unwrap().len(), 2);
}

fn resolved(t: &Table, src: &str) -> Recipe {
    let frame = t.frame();
    recipe(t, src)
        .resolve_text(&frame, |iri, text| t.code_for(&frame, iri, text))
        .unwrap()
}

/// A text literal is resolved to the column's code under the table's frame,
/// then compared as a code: `==` and `!=` drop the rows they name, and a
/// string the column never coded matches no row.
#[test]
fn a_text_term_compares_codes_resolved_under_the_frame() {
    let t = table();
    let w = witness(&t);
    let count = |src: &str| resolved(&t, src).count(&w).unwrap();
    let r = ".r = .properties.region.value\n";
    assert_eq!(count(&format!("{r}if .r == \"north\" {{ abort }}")), 2);
    assert_eq!(count(&format!("{r}if .r != \"north\" {{ abort }}")), 1);
    assert_eq!(count(&format!("{r}if .r == \"west\" {{ abort }}")), 3);
    assert_eq!(count(&format!("{r}if .r != \"west\" {{ abort }}")), 0);
    let rows = resolved(&t, &format!("{r}if .r == \"south\" {{ abort }}"))
        .materialize_rows(&w)
        .unwrap();
    let names: Vec<String> = (0..rows.len())
        .map(|i| {
            t.code_key("region", rows.get(i, 0).as_code().unwrap())
                .unwrap()
        })
        .collect();
    assert_eq!(names, ["north", "east"]);
}

/// Text has no order, a literal must be resolved before preparing, and only
/// a code lane is compared with text.
#[test]
fn a_text_term_refuses_order_an_unresolved_literal_and_a_number_lane() {
    let t = table();
    let w = witness(&t);
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    assert!(matches!(
        Recipe::compile(
            ".r = .properties.region.value\nif .r < \"m\" { abort }",
            t.witness().view,
            class
        ),
        Err(Refusal::OrderOnText { .. })
    ));
    let unresolved = recipe(
        &t,
        ".r = .properties.region.value\nif .r == \"north\" { abort }",
    );
    assert_eq!(unresolved.count(&w), Err(Refusal::TextUnresolved));
    let on_number = resolved(&t, ".q = .properties.qty.value\nif .q == \"10\" { abort }");
    assert!(matches!(on_number.count(&w), Err(Refusal::NotCode { .. })));
}

/// Literals resolved before a write are refused after it, since the write
/// may have coded the string; resolved again, the recipe reads the new code.
#[test]
fn literals_resolved_before_an_append_are_refused() {
    let mut t = table();
    let src = ".r = .properties.region.value\nif .r == \"west\" { abort }";
    let before = resolved(&t, src);
    let p = before.prepare_count(3).unwrap();
    t.append(&[vec![Some("4"), Some("west"), Some("5")]])
        .unwrap();
    let mut scratch = p.scratch().unwrap();
    assert!(matches!(
        before.run_into(
            &p,
            &witness(&t),
            &mut scratch,
            lance_graph_mask_risc::Out::None
        ),
        Err(Refusal::Binding(lance_graph_dt64::LaneRefusal::Frame(_)))
    ));
    // Prepared fresh at the new row count, only the frame check stands
    // between the stale literal and a wrong count (4: the west row kept).
    assert!(matches!(
        before.count(&witness(&t)),
        Err(Refusal::Binding(lance_graph_dt64::LaneRefusal::Frame(_)))
    ));
    assert_eq!(resolved(&t, src).count(&witness(&t)).unwrap(), 3);
}
