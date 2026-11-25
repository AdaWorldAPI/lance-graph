//! Data as config, class as config: detection, the hand-config escape hatch,
//! shape observation, and the PostgreSQL gate over two generated tables.
//!
//! The gate answers every query from the table and compares it with the rows
//! the model derives from the input text (always), and with PostgreSQL through
//! `pg-parity-harness` when a server answers. A SKIP prints `SKIP`; with
//! `PG_ROW_REQUIRE_PG=1` it fails the test. A SKIP is not a pass of the
//! PostgreSQL half.

use std::collections::BTreeMap;

use lance_graph_contract::pg_row::{write_cell, AbsentAs, LaneEncoding, PgColumn, PgType};
use lance_graph_intake::pg::{write_selection, CollectRows, CopyText};
use lance_graph_intake::{
    detect_column, detect_key, detections, ColumnConfig, Datum, IntakeRefusal, ShapeConfig,
    SourceProfile, Table, Verdict,
};
use lance_graph_pg_row::{PgRowRefusal, PgRows};
use lance_graph_quack::bind::{table as draft, BindError};
use lance_graph_quack::{Cmp, Col, Filter};
use pg_parity_harness::{multiset_diff, Pg, Row, Skip};

const fn col(ty: PgType, encoding: LaneEncoding, absent: AbsentAs) -> PgColumn {
    PgColumn {
        ty,
        encoding,
        absent,
    }
}

const BOOL: PgColumn = col(PgType::Bool, LaneEncoding::Bool, AbsentAs::Null);
const INT4: PgColumn = col(PgType::Int4, LaneEncoding::Signed, AbsentAs::Null);
const INT8_U: PgColumn = col(PgType::Int8, LaneEncoding::Unsigned, AbsentAs::Null);
const TEXT: PgColumn = col(PgType::Text, LaneEncoding::TextCode, AbsentAs::Null);
const TEXT_EMPTY: PgColumn = col(PgType::Text, LaneEncoding::TextCode, AbsentAs::EmptyText);
const TIMESTAMP: PgColumn = col(
    PgType::Timestamp(None),
    LaneEncoding::EpochSeconds,
    AbsentAs::Null,
);
const DATE: PgColumn = col(PgType::Date, LaneEncoding::EpochDays, AbsentAs::Null);

type Grid = Vec<Vec<Option<&'static str>>>;

fn small() -> Grid {
    vec![
        vec![Some("1"), Some("10")],
        vec![Some("2"), Some("20")],
        vec![Some("3"), None],
    ]
}

fn small_table(config: &ShapeConfig) -> Result<Table, IntakeRefusal> {
    Table::from_rows("small", 0x2B03_0001, &["id", "n"], &small(), config)
}

fn cells(rows: &[Vec<Option<String>>]) -> Vec<Vec<Option<&str>>> {
    rows.iter()
        .map(|row| row.iter().map(|cell| cell.as_deref()).collect())
        .collect()
}

fn text(c: &str) -> Option<String> {
    Some(c.to_owned())
}

/// FAILS IF: `t` / `f` cells are not read as a boolean column.
#[test]
fn detects_bool() {
    let d = detect_column(&[Some("t"), Some("f"), None, Some("t")]);
    assert_eq!(d.col, BOOL);
    assert_eq!((d.support, d.total, d.witness), (3, 3, None));
}

/// FAILS IF: cells that fit `i32` are not a signed integer, or a leading
/// zero or a plus sign (which PostgreSQL would not print) is accepted.
#[test]
fn detects_int4() {
    let d = detect_column(&[
        Some("1"),
        Some("-5"),
        Some("2147483647"),
        Some("-2147483648"),
    ]);
    assert_eq!(d.col, INT4);
    assert_eq!((d.support, d.total, d.witness), (4, 4, None));
    assert_eq!(detect_column(&[Some("007")]).col, TEXT);
    assert_eq!(detect_column(&[Some("+5")]).col, TEXT);
}

/// FAILS IF: a value over `i32::MAX` that fits `u32` is not a bigint with
/// an unsigned lane, or the row that broke `i32` is not named.
#[test]
fn detects_int8_unsigned_over_i32() {
    let d = detect_column(&[Some("5"), Some("2147483648"), Some("4294967295")]);
    assert_eq!(d.col, INT8_U);
    assert_eq!(d.witness, Some(1));
    assert_eq!((d.support, d.total), (3, 3));
}

/// FAILS IF: a value over `u32::MAX`, or a negative beside a large value,
/// stays an integer, or the first row that did not fit is not named.
#[test]
fn integer_past_a_lane_is_text_with_its_witness() {
    let d = detect_column(&[Some("7"), Some("4294967296")]);
    assert_eq!(d.col, TEXT);
    assert_eq!(d.witness, Some(1));
    let d = detect_column(&[Some("-1"), Some("3000000000")]);
    assert_eq!(d.col, TEXT);
    assert_eq!(d.witness, Some(0));
}

/// FAILS IF: `YYYY-MM-DD HH:MM:SS` cells are not a timestamp without time
/// zone, or one whose epoch does not fit `u32` stays a timestamp.
#[test]
fn detects_timestamp() {
    let d = detect_column(&[
        Some("2020-01-02 03:04:05"),
        Some("1970-01-01 00:00:00"),
        None,
    ]);
    assert_eq!(d.col, TIMESTAMP);
    assert_eq!((d.support, d.total, d.witness), (2, 2, None));
    let d = detect_column(&[Some("2020-01-02 03:04:05"), Some("2200-01-01 00:00:00")]);
    assert_eq!(d.col, TEXT);
    assert_eq!(d.witness, Some(1));
    assert_eq!(detect_column(&[Some("2020-01-02 03:04:05.5")]).col, TEXT);
}

/// FAILS IF: `YYYY-MM-DD` cells are not a date, or an impossible date stays
/// a date.
#[test]
fn detects_date() {
    let d = detect_column(&[Some("2000-02-29"), Some("1970-01-01")]);
    assert_eq!(d.col, DATE);
    assert_eq!((d.support, d.total, d.witness), (2, 2, None));
    let d = detect_column(&[Some("2000-02-29"), Some("2023-02-30")]);
    assert_eq!(d.col, TEXT);
    assert_eq!(d.witness, Some(1));
}

/// FAILS IF: anything else is not text with NULL as its absent spelling.
#[test]
fn detects_text() {
    let d = detect_column(&[Some("printer"), Some("vpn"), Some("12x")]);
    assert_eq!(d.col, TEXT);
    assert_eq!((d.support, d.total, d.witness), (3, 3, None));
}

/// FAILS IF: an observed `''` makes detection read `''` as absent. Only a
/// declaration may (`ColumnConfig::absent`); detection keeps `''` as a value.
#[test]
fn an_observed_empty_string_is_present_text() {
    assert_eq!(detect_column(&[Some("a"), Some(""), Some("b")]).col, TEXT);
    let mixed = detect_column(&[Some("a"), Some(""), None]);
    assert_eq!(mixed.col, TEXT);
    assert_eq!((mixed.support, mixed.total), (2, 2));
    assert_eq!(detect_column(&[Some("a"), Some("b"), None]).col, TEXT);
    assert_eq!(detect_column(&[Some("1"), None]).col, INT4);
}

/// FAILS IF: a column with no value at all has support, or a type other than
/// text with NULL absent.
#[test]
fn an_all_null_column_is_text_with_no_support() {
    let d = detect_column(&[None, None]);
    assert_eq!(d.col, TEXT);
    assert_eq!((d.support, d.total, d.witness), (0, 0, None));
    assert_eq!(detect_column(&[]).col, TEXT);
}

/// FAILS IF: the key is not the first integer column named `id`, then the
/// first integer column, that is non-NULL and distinct.
#[test]
fn detects_the_key() {
    let names = ["n", "id", "m"];
    let rows: Vec<Vec<Option<&str>>> = vec![
        vec![Some("5"), Some("1"), Some("9")],
        vec![Some("6"), Some("2"), Some("9")],
    ];
    let detect = |rows: &[Vec<Option<&str>>]| -> Vec<_> {
        (0..names.len())
            .map(|k| detect_column(&rows.iter().map(|r| r[k]).collect::<Vec<_>>()))
            .collect()
    };
    assert_eq!(detect_key(&names, &detect(&rows), &rows), Some(1));

    let dup: Vec<Vec<Option<&str>>> = vec![
        vec![Some("5"), Some("1"), Some("9")],
        vec![Some("6"), Some("1"), Some("8")],
    ];
    assert_eq!(detect_key(&names, &detect(&dup), &dup), Some(0));

    let nulls: Vec<Vec<Option<&str>>> = vec![
        vec![Some("5"), None, Some("9")],
        vec![Some("5"), Some("1"), Some("9")],
    ];
    assert_eq!(detect_key(&names, &detect(&nulls), &nulls), None);
}

/// FAILS IF: a config that forces a column to text does not win over the
/// detection, is not reported as differing, or the read does not print the
/// text.
#[test]
fn config_forcing_text_differs_and_reads_as_text() {
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            encoding: Some(LaneEncoding::TextCode),
            absent: None,
        },
    );
    let table = small_table(&config).expect("table");
    let n = &table.shape()[1];
    assert_eq!(n.col, TEXT);
    assert_eq!(n.detected.as_ref().expect("detected").col, INT4);
    assert_eq!(n.verdict, Verdict::Differs { detected: INT4 });
    assert_eq!(table.shape()[0].verdict, Verdict::DetectedOnly);

    let all = table.select_all().expect("select");
    assert_eq!(all.types, ["integer", "text"]);
    assert_eq!(
        all.rows,
        vec![
            vec![text("1"), text("10")],
            vec![text("2"), text("20")],
            vec![text("3"), None],
        ]
    );
}

/// FAILS IF: a config equal to the detection is reported as differing, or an
/// unconfigured column is reported as confirmed.
#[test]
fn config_equal_to_detection_agrees() {
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Int4),
            encoding: Some(LaneEncoding::Signed),
            absent: None,
        },
    );
    let table = small_table(&config).expect("table");
    assert_eq!(table.shape()[1].verdict, Verdict::Agrees);
    assert_eq!(table.shape()[0].verdict, Verdict::DetectedOnly);
}

fn declare(config: &mut ShapeConfig, name: &str, col: PgColumn) {
    config.columns.insert(
        name.to_owned(),
        ColumnConfig {
            ty: Some(col.ty),
            encoding: Some(col.encoding),
            absent: Some(col.absent),
        },
    );
}

/// FAILS IF: a table whose every column is declared still detects a column,
/// reports a verdict other than `Declared`, or reads differently from the
/// same rows taken by discovery. The discovery run detecting both columns is
/// the twin that shows the counter counts.
#[test]
fn declared_columns_are_not_detected_and_read_as_discovered() {
    let before = detections();
    let discovered = small_table(&ShapeConfig::default()).expect("discovered");
    assert_eq!(detections() - before, 2);

    let mut config = ShapeConfig::default();
    declare(&mut config, "id", INT4);
    declare(&mut config, "n", INT4);
    config.key = Some("id".to_owned());
    let before = detections();
    let declared = small_table(&config).expect("declared");
    assert_eq!(detections() - before, 0);

    for c in declared.shape() {
        assert_eq!(c.verdict, Verdict::Declared, "{}", c.name);
        assert_eq!(c.detected, None, "{}", c.name);
    }
    let cols = |t: &Table| t.shape().iter().map(|c| c.col).collect::<Vec<_>>();
    assert_eq!(cols(&declared), cols(&discovered));
    let (d, s) = (
        declared.select_all().expect("declared read"),
        discovered.select_all().expect("discovered read"),
    );
    assert_eq!(d.types, s.types);
    assert_eq!(d.rows, s.rows);
}

/// FAILS IF: a column that gives only some of its values skips detection;
/// only a column with all three is declared.
#[test]
fn partly_configured_column_is_still_detected() {
    let mut config = ShapeConfig::default();
    declare(&mut config, "id", INT4);
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Int4),
            encoding: Some(LaneEncoding::Signed),
            absent: None,
        },
    );
    let before = detections();
    let table = small_table(&config).expect("table");
    assert_eq!(detections() - before, 1);
    assert_eq!(table.shape()[0].verdict, Verdict::Declared);
    assert_eq!(table.shape()[1].verdict, Verdict::Agrees);
}

/// FAILS IF: skipping detection skips validation: a cell that does not fit
/// its declared column is accepted at build or at append.
#[test]
fn declared_column_still_validates_every_cell() {
    let mut config = ShapeConfig::default();
    declare(&mut config, "id", INT4);
    declare(&mut config, "n", INT4);
    config.key = Some("id".to_owned());
    let rows: Grid = vec![vec![Some("1"), Some("10")], vec![Some("2"), Some("x")]];
    let refused = Table::from_rows("small", 0x2B03_0001, &["id", "n"], &rows, &config);
    assert!(
        matches!(&refused, Err(IntakeRefusal::Cell { column, row: 1, .. }) if column == "n"),
        "{:?}",
        refused.err()
    );

    let mut table = small_table(&config).expect("table");
    let refused = table.append(&[vec![Some("4"), Some("x")]]);
    assert!(
        matches!(refused, Err(IntakeRefusal::ShapeChanged { ref column, .. }) if column == "n"),
        "{refused:?}"
    );
    assert_eq!(table.select_all().expect("read").rows.len(), 3);
}

/// FAILS IF: a config entry or key naming no column is accepted.
#[test]
fn config_naming_no_column_is_refused() {
    let mut config = ShapeConfig::default();
    config
        .columns
        .insert("nope".to_owned(), ColumnConfig::default());
    assert_eq!(
        small_table(&config).err(),
        Some(IntakeRefusal::UnknownColumn("nope".to_owned()))
    );
    let config = ShapeConfig {
        key: Some("nope".to_owned()),
        ..ShapeConfig::default()
    };
    assert_eq!(
        small_table(&config).err(),
        Some(IntakeRefusal::UnknownColumn("nope".to_owned()))
    );
}

/// FAILS IF: a cell that does not parse under a hand-given shape is read
/// anyway, or a hand-given shape the lane cannot hold is admitted.
#[test]
fn config_that_cannot_parse_a_cell_is_refused() {
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Bool),
            encoding: Some(LaneEncoding::Bool),
            absent: None,
        },
    );
    assert_eq!(
        small_table(&config).err(),
        Some(IntakeRefusal::Cell {
            column: "n".to_owned(),
            row: 0,
            text: "10".to_owned(),
        })
    );

    // A declared type brings an encoding the contract admits for it, so
    // digits declared as text are text values, not integers.
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            ..ColumnConfig::default()
        },
    );
    let declared = small_table(&config).expect("digits declared as text");
    assert_eq!(declared.shape()[1].col, TEXT);
    assert_eq!(
        declared.select_all().expect("rows").rows[0][1].as_deref(),
        Some("10")
    );

    // A hand-given pair the lane cannot hold is refused.
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            encoding: Some(LaneEncoding::Signed),
            absent: None,
        },
    );
    assert!(matches!(
        small_table(&config).err(),
        Some(IntakeRefusal::Column { ref name, .. }) if name == "n"
    ));
}

/// FAILS IF: some column has no verdict or more than one.
#[test]
fn every_column_has_one_verdict() {
    let table = small_table(&ShapeConfig::default()).expect("table");
    let count =
        |pick: fn(&Verdict) -> bool| table.shape().iter().filter(|c| pick(&c.verdict)).count();
    assert_eq!(
        count(|v| matches!(v, Verdict::Agrees))
            + count(|v| matches!(v, Verdict::Differs { .. }))
            + count(|v| matches!(v, Verdict::DetectedOnly)),
        table.shape().len()
    );
    assert_eq!(table.key_name(), "id");
}

/// FAILS IF: rows without a key, with a repeated key, or too wide are placed.
#[test]
fn placement_refusals() {
    let config = ShapeConfig::default();
    let dup: Grid = vec![vec![Some("1")], vec![Some("1")]];
    assert_eq!(
        Table::from_rows("t", 0x2B03_0002, &["id"], &dup, &config).err(),
        Some(IntakeRefusal::NoKey),
        "a column with a repeat is not detected as the key"
    );
    let keyed = ShapeConfig {
        key: Some("id".to_owned()),
        ..ShapeConfig::default()
    };
    assert_eq!(
        Table::from_rows("t", 0x2B03_0002, &["id"], &dup, &keyed).err(),
        Some(IntakeRefusal::DuplicateKey(1))
    );
    let words: Grid = vec![vec![Some("a")], vec![Some("b")]];
    assert_eq!(
        Table::from_rows("t", 0x2B03_0003, &["w"], &words, &config).err(),
        Some(IntakeRefusal::NoKey)
    );
    let ragged: Grid = vec![vec![Some("1"), Some("2")], vec![Some("3")]];
    assert_eq!(
        Table::from_rows("t", 0x2B03_0004, &["a", "b"], &ragged, &config).err(),
        Some(IntakeRefusal::Ragged { row: 1, cells: 1 })
    );
    let names: Vec<String> = (0..256).map(|k| format!("c{k}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert_eq!(
        Table::from_rows("t", 0x2B03_0005, &refs, &[], &config).err(),
        Some(IntakeRefusal::Width { columns: 256 })
    );
}

/// FAILS IF: rows of the same shape are refused, or a new row that turns an
/// integer column into text is accepted.
#[test]
fn observe_reports_a_changed_shape() {
    let table = small_table(&ShapeConfig::default()).expect("table");
    let same: Grid = vec![vec![Some("4"), Some("30")], vec![Some("5"), None]];
    assert_eq!(table.observe(&same), Ok(()));
    let changed: Grid = vec![vec![Some("4"), Some("30")], vec![Some("5"), Some("x")]];
    assert_eq!(
        table.observe(&changed),
        Err(IntakeRefusal::ShapeChanged {
            column: "n".to_owned(),
            was: INT4,
            now: TEXT,
        })
    );
}

/// FAILS IF: a batch that fits the shape is reported as a change because it
/// is judged on its own: an all-NULL batch, or only small values in a column
/// built as int8.
#[test]
fn observe_judges_new_rows_against_the_built_shape() {
    let table = small_table(&ShapeConfig::default()).expect("table");
    let nulls: Grid = vec![vec![Some("6"), None], vec![Some("7"), None]];
    assert_eq!(table.observe(&nulls), Ok(()), "all-NULL batch fits");

    let wide: Grid = vec![
        vec![Some("1"), Some("3000000000")],
        vec![Some("2"), Some("4")],
    ];
    let refs = ["id", "n"];
    let big = Table::from_rows("w", 0x2B03_0009, &refs, &wide, &ShapeConfig::default())
        .expect("int8 table");
    assert_eq!(
        big.shape()[1].col.ty,
        lance_graph_contract::pg_row::PgType::Int8
    );
    let small: Grid = vec![vec![Some("3"), Some("5")]];
    assert_eq!(
        big.observe(&small),
        Ok(()),
        "small values fit an int8 column"
    );
}

/// FAILS IF: a new batch with more or fewer cells per row than the table's
/// columns passes `observe`.
#[test]
fn observe_refuses_ragged_rows() {
    let table = small_table(&ShapeConfig::default()).expect("table");
    let short: Grid = vec![vec![Some("4")]];
    assert_eq!(
        table.observe(&short),
        Err(IntakeRefusal::Ragged { row: 0, cells: 1 })
    );
    let long: Grid = vec![
        vec![Some("4"), Some("30")],
        vec![Some("5"), Some("6"), Some("x")],
    ];
    assert_eq!(
        table.observe(&long),
        Err(IntakeRefusal::Ragged { row: 1, cells: 3 })
    );
}

/// FAILS IF: a configured `varchar(n)` admits a longer cell at intake or in
/// `observe`, or refuses one of exactly `n` characters.
#[test]
fn a_configured_varchar_bound_is_enforced() {
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(lance_graph_contract::pg_row::PgType::Varchar(Some(3))),
            encoding: Some(lance_graph_contract::pg_row::LaneEncoding::TextCode),
            absent: None,
        },
    );
    let fits: Grid = vec![vec![Some("1"), Some("abc")], vec![Some("2"), Some("äöü")]];
    let table = Table::from_rows("v", 0x2B03_000A, &["id", "n"], &fits, &config).expect("fits");
    assert_eq!(table.observe(&[vec![Some("3"), Some("xyz")]]), Ok(()));
    assert!(matches!(
        table.observe(&[vec![Some("3"), Some("abcd")]]),
        Err(IntakeRefusal::ShapeChanged { .. })
    ));
    let long: Grid = vec![vec![Some("1"), Some("abcd")]];
    assert!(matches!(
        Table::from_rows("v", 0x2B03_000B, &["id", "n"], &long, &config),
        Err(IntakeRefusal::Cell { row: 0, .. })
    ));
}

/// FAILS IF: a deleted key is read, kept by a scan, or counted; an unknown
/// key is not refused; or a bag of keys loses its repeats or its order.
#[test]
fn a_deleted_key_is_refused_and_absent_from_a_scan() {
    let mut table = small_table(&ShapeConfig::default()).expect("table");
    assert_eq!(table.len(), 3);
    table.delete_key(2).expect("delete");
    assert_eq!(table.len(), 2);
    assert_eq!(
        table.rows_for_keys(&[2]),
        Err(IntakeRefusal::PgRow(PgRowRefusal::DeadPosition { at: 1 }))
    );
    assert_eq!(table.rows_for_keys(&[9]), Err(IntakeRefusal::UnknownKey(9)));
    assert_eq!(table.delete_key(9), Err(IntakeRefusal::UnknownKey(9)));

    let all = table.select_all().expect("select");
    assert_eq!(
        all.rows,
        vec![vec![text("1"), text("10")], vec![text("3"), None]]
    );
    let bag = table.rows_for_keys(&[3, 1, 3]).expect("bag");
    assert_eq!(
        bag.rows,
        vec![
            vec![text("3"), None],
            vec![text("1"), text("10")],
            vec![text("3"), None],
        ]
    );
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn spell(ty: PgType, encoding: LaneEncoding, value: u32) -> String {
    let mut out = String::new();
    write_cell(
        &col(ty, encoding, AbsentAs::Null),
        true,
        value,
        None,
        &mut out,
    )
    .expect("spell");
    out
}

const TICKET_COLUMNS: [&str; 8] = [
    "id", "title", "note", "open", "owner_id", "priority", "created", "due",
];
const AGENT_COLUMNS: [&str; 2] = ["id", "name"];
const AGENT_NAMES: [&str; 8] = ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hal"];

/// 300 tickets with NULLs, `''` and negatives. `title` mixes NULL and `''`;
/// `note` has `''` and no NULL; no other column has `''`. Ticket owners run
/// past the agent ids, so some have no parent.
fn tickets() -> (Vec<Vec<Option<String>>>, Vec<bool>) {
    let mut x = 0x7E57_u64;
    let titles = ["printer", "vpn", "", "login", "mail"];
    let notes = ["", "call back", "waiting"];
    let mut rows = Vec::new();
    let mut deleted = Vec::new();
    for id in 1..=300i64 {
        let r = xorshift(&mut x);
        let title =
            (!r.is_multiple_of(7)).then(|| titles[(r >> 3) as usize % titles.len()].to_owned());
        let note = Some(notes[(r >> 5) as usize % notes.len()].to_owned());
        let open = Some(if r >> 7 & 1 == 1 { "t" } else { "f" }.to_owned());
        let owner = (!r.is_multiple_of(5))
            .then(|| (1 + (r >> 9) % (AGENT_NAMES.len() as u64 + 3)).to_string());
        let priority = (!r.is_multiple_of(11)).then(|| (((r >> 11) % 9) as i64 - 4).to_string());
        let created = Some(spell(
            PgType::Timestamp(None),
            LaneEncoding::EpochSeconds,
            1_600_000_000 + (r >> 13) as u32 % 100_000_000,
        ));
        let due = (!r.is_multiple_of(4)).then(|| {
            spell(
                PgType::Date,
                LaneEncoding::EpochDays,
                18_000 + (r >> 17) as u32 % 4_000,
            )
        });
        rows.push(vec![
            Some(id.to_string()),
            title,
            note,
            open,
            owner,
            priority,
            created,
            due,
        ]);
        deleted.push(r.is_multiple_of(13));
    }
    (rows, deleted)
}

fn agents() -> Vec<Vec<Option<String>>> {
    AGENT_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| vec![Some((i + 1).to_string()), Some((*name).to_owned())])
        .collect()
}

fn int(row: &[Option<String>], k: usize) -> Option<i64> {
    row[k].as_deref().and_then(|c| c.parse().ok())
}

fn same(name: &str, expected: &[Row], ours: &PgRows) {
    if let Some(diff) = multiset_diff(expected, &ours.rows) {
        panic!("{name}: model oracle differs: {diff}");
    }
}

fn sql_literal(cell: &Option<String>) -> String {
    match cell {
        None => "NULL".to_owned(),
        Some(t) => format!("'{}'", t.replace('\'', "''")),
    }
}

/// `CREATE TABLE` and `INSERT` of the live rows; column types come from the
/// detected shape, except that an empty-absent text column is
/// `NOT NULL DEFAULT ''`.
fn create(table: &Table, rows: &[Vec<Option<String>>], deleted: &[bool]) -> String {
    let columns: Vec<String> = table
        .shape()
        .iter()
        .map(|c| {
            let ty = if c.col.absent == AbsentAs::EmptyText {
                "text NOT NULL DEFAULT ''".to_owned()
            } else {
                c.col.ty.gdesc()
            };
            format!("{} {ty}", c.name)
        })
        .collect();
    let live: Vec<String> = rows
        .iter()
        .zip(deleted)
        .filter(|(_, &gone)| !gone)
        .map(|(row, _)| {
            let values: Vec<String> = row.iter().map(sql_literal).collect();
            format!("({})", values.join(", "))
        })
        .collect();
    format!(
        "CREATE TABLE {name} ({columns});\nINSERT INTO {name} VALUES {live};\n",
        name = table.name(),
        columns = columns.join(", "),
        live = live.join(",\n"),
    )
}

/// FAILS IF: any answer of the intake table differs from the model rows
/// derived from the input text, or from PostgreSQL's answer to the same
/// query, in rows or in `\gdesc` types, or a shape is detected wrongly.
#[test]
fn intake_equals_postgresql() {
    let (ticket_rows, ticket_deleted) = tickets();
    let agent_rows = agents();

    let ticket = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .expect("ticket table");
    assert!(ticket
        .shape()
        .iter()
        .all(|c| c.verdict == Verdict::DetectedOnly));
    gate(
        ticket,
        agent_table(&agent_rows),
        &ticket_rows,
        &ticket_deleted,
        &agent_rows,
        &agent_gone(),
    );
}

fn agent_table(agent_rows: &[Vec<Option<String>>]) -> Table {
    Table::from_rows(
        "agent",
        0x2B02_0001,
        &AGENT_COLUMNS,
        &cells(agent_rows),
        &ShapeConfig::default(),
    )
    .expect("agent table")
}

fn agent_gone() -> Vec<bool> {
    let mut deleted = vec![false; AGENT_NAMES.len()];
    deleted[2] = true;
    deleted
}

/// Delete the marked rows, answer the four cases from the tables and compare
/// each with the model (and PostgreSQL when a server answers). Returns the
/// four answers: select-all, where, selection bag, left join.
fn gate(
    mut ticket: Table,
    mut agent: Table,
    ticket_rows: &[Vec<Option<String>>],
    ticket_deleted: &[bool],
    agent_rows: &[Vec<Option<String>>],
    agent_deleted: &[bool],
) -> [PgRows; 4] {
    let shapes: Vec<PgColumn> = ticket.shape().iter().map(|c| c.col).collect();
    assert_eq!(
        shapes,
        [INT4, TEXT, TEXT, BOOL, INT4, INT4, TIMESTAMP, DATE]
    );
    assert_eq!(ticket.key_name(), "id");
    assert_eq!(agent.shape()[0].col, INT4);
    assert_eq!(agent.shape()[1].col, TEXT);

    for (r, &gone) in ticket_deleted.iter().enumerate() {
        if gone {
            ticket.delete_key(r as u64 + 1).expect("delete ticket");
        }
    }
    for (r, &gone) in agent_deleted.iter().enumerate() {
        if gone {
            agent.delete_key(r as u64 + 1).expect("delete agent");
        }
    }
    let live: Vec<usize> = (0..ticket_rows.len())
        .filter(|&r| !ticket_deleted[r])
        .collect();
    assert_eq!(ticket.len(), live.len());

    let model_row = |r: usize| -> Row { ticket_rows[r].clone() };

    let all = ticket.select_all().expect("select all");
    let model_all: Vec<Row> = live.iter().map(|&r| model_row(r)).collect();
    assert_eq!(all.rows.len(), live.len());
    same("select-all", &model_all, &all);

    let filter = Filter::or([
        Filter::cmp(Col(4), Cmp::EqI32(2)),
        Filter::negate(Filter::cmp(Col(5), Cmp::LtI32(0))),
    ]);
    let where_rows = ticket.select_where(&filter).expect("where");
    let model_where: Vec<Row> = live
        .iter()
        .copied()
        .filter(|&r| {
            let owner = int(&ticket_rows[r], 4).map(|o| o == 2);
            let not_negative = int(&ticket_rows[r], 5).map(|p| p >= 0);
            owner == Some(true) || not_negative == Some(true)
        })
        .map(model_row)
        .collect();
    same("where-k3", &model_where, &where_rows);
    assert_eq!(
        ticket.count_where(&filter).expect("count"),
        model_where.len()
    );
    assert!(
        live.iter()
            .any(|&r| int(&ticket_rows[r], 4).is_none() && int(&ticket_rows[r], 5).is_none()),
        "the fixture has rows where the filter is UNKNOWN"
    );
    assert!(
        model_where.len() < live.len() && model_where.len() * 5 > live.len(),
        "the filter keeps a real subset"
    );

    let mut bag_rows: Vec<usize> = live.iter().copied().step_by(5).collect();
    bag_rows.extend(live.iter().copied().step_by(11));
    let bag_keys: Vec<u64> = bag_rows.iter().map(|&r| r as u64 + 1).collect();
    let picked = ticket.rows_for_keys(&bag_keys).expect("bag");
    let model_bag: Vec<Row> = bag_rows.iter().map(|&r| model_row(r)).collect();
    assert!(bag_rows.len() > live.len() / 5, "the bag repeats keys");
    same("selection-bag", &model_bag, &picked);

    let joined = ticket.left_join("owner_id", &agent).expect("left join");
    let live_agents: BTreeMap<i64, &Vec<Option<String>>> = agent_rows
        .iter()
        .zip(agent_deleted)
        .filter(|(_, &gone)| !gone)
        .filter_map(|(row, _)| int(row, 0).map(|id| (id, row)))
        .collect();
    let mut missing = 0;
    let model_join: Vec<Row> = live
        .iter()
        .map(|&r| {
            let mut row = model_row(r);
            match int(&ticket_rows[r], 4).and_then(|o| live_agents.get(&o)) {
                Some(parent) => row.extend(parent.iter().cloned()),
                None => {
                    missing += 1;
                    row.extend([None, None]);
                }
            }
            row
        })
        .collect();
    assert!(missing > 0, "some tickets have no live owner");
    same("left-join", &model_join, &joined);
    let out = [
        all.clone(),
        where_rows.clone(),
        picked.clone(),
        joined.clone(),
    ];

    let empty_notes = all
        .rows
        .iter()
        .filter(|r| r[2].as_deref() == Some(""))
        .count();
    let null_titles = all.rows.iter().filter(|r| r[1].is_none()).count();
    let empty_titles = all
        .rows
        .iter()
        .filter(|r| r[1].as_deref() == Some(""))
        .count();
    let negatives = all
        .rows
        .iter()
        .filter(|r| r[5].as_deref().is_some_and(|p| p.starts_with('-')))
        .count();
    assert!(empty_notes > 0 && null_titles > 0 && empty_titles > 0 && negatives > 0);

    let pg = match Pg::connect() {
        Ok(pg) => pg,
        Err(Skip(why)) => {
            assert!(
                std::env::var_os("PG_ROW_REQUIRE_PG").is_none(),
                "PG_ROW_REQUIRE_PG is set and PostgreSQL is unavailable: {why}"
            );
            println!("SKIP (PostgreSQL half): {why}");
            return out;
        }
    };
    println!("PostgreSQL {}", pg.version());
    pg.script(
        &(create(&ticket, ticket_rows, ticket_deleted)
            + &create(&agent, agent_rows, agent_deleted)),
    )
    .expect("create");
    let bag_ids: Vec<String> = bag_keys.iter().map(u64::to_string).collect();
    let cases: [(&str, String, &PgRows); 4] = [
        ("select-all", "SELECT * FROM ticket".to_owned(), &all),
        (
            "where-k3",
            "SELECT * FROM ticket WHERE owner_id = 2 OR NOT (priority < 0)".to_owned(),
            &where_rows,
        ),
        (
            "selection-bag",
            format!(
                "SELECT t.* FROM unnest(ARRAY[{}]) AS u(id) JOIN ticket t USING (id)",
                bag_ids.join(",")
            ),
            &picked,
        ),
        (
            "left-join",
            "SELECT t.*, a.* FROM ticket t LEFT JOIN agent a ON a.id = t.owner_id".to_owned(),
            &joined,
        ),
    ];
    for (name, sql, ours) in cases {
        let theirs = pg.query(&sql).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(theirs.types, ours.types, "{name}: types");
        if let Some(diff) = multiset_diff(&theirs.rows, &ours.rows) {
            panic!("{name}: PostgreSQL differs: {diff}");
        }
        println!("PASS {name}: {} rows", ours.rows.len());
    }
    out
}

/// A table whose columns are the one-shot table's merged shapes, handed in as
/// hand config, so batches of any size are judged against the same shape.
fn full_config(table: &Table) -> ShapeConfig {
    let mut config = ShapeConfig::default();
    for c in table.shape() {
        config.columns.insert(
            c.name.clone(),
            ColumnConfig {
                ty: Some(c.col.ty),
                encoding: Some(c.col.encoding),
                absent: Some(c.col.absent),
            },
        );
    }
    config.key = Some(table.key_name().to_owned());
    config
}

/// The filter of the gate's where case: `owner_id = 2 OR NOT (priority < 0)`.
fn owner_or_not_negative() -> Filter {
    Filter::or([
        Filter::cmp(Col(4), Cmp::EqI32(2)),
        Filter::negate(Filter::cmp(Col(5), Cmp::LtI32(0))),
    ])
}

/// FAILS IF: a text column's strings are held anywhere but DT's `CodeTable`,
/// code 0 maps to a key, an old code is renumbered by an append, a repeated
/// string takes a new code, or a frame from before an append reads the new
/// generation.
#[test]
fn text_codes_live_in_dt() {
    let rows: Grid = vec![
        vec![Some("1"), Some("a")],
        vec![Some("2"), Some("b")],
        vec![Some("3"), None],
    ];
    let mut table = Table::from_rows(
        "w",
        0x2B04_0001,
        &["id", "w"],
        &rows,
        &ShapeConfig::default(),
    )
    .expect("table");
    assert_eq!(
        table.code_key("w", 0),
        None,
        "code 0 is absent, never a key"
    );
    assert_eq!(table.code_key("w", 1), Some("a".to_owned()));
    assert_eq!(table.code_key("w", 2), Some("b".to_owned()));
    assert_eq!(table.code_key("w", 3), None);
    assert_eq!(
        table.code_key("id", 1),
        None,
        "an integer column has no codes"
    );

    let old = table.frame();
    table
        .append(&[vec![Some("4"), Some("c")], vec![Some("5"), Some("a")]])
        .expect("append");
    assert_eq!(table.code_key("w", 3), Some("c".to_owned()));
    assert_eq!(
        table.code_key("w", 4),
        None,
        "a repeated string takes no new code"
    );
    assert_eq!(
        table.code_key("w", 1),
        Some("a".to_owned()),
        "old codes stay"
    );
    assert_eq!(table.code_key_at(&old, "w", 1), Some("a".to_owned()));
    assert_eq!(
        table.code_key_at(&old, "w", 3),
        None,
        "the old frame reads the old generation"
    );
    assert_eq!(
        table.select_all().expect("select").rows,
        vec![
            vec![text("1"), text("a")],
            vec![text("2"), text("b")],
            vec![text("3"), None],
            vec![text("4"), text("c")],
            vec![text("5"), text("a")],
        ]
    );
}

/// FAILS IF: ticket rows fed in three appends (100, 150, 50) answer any of
/// the four cases differently from the model, from PostgreSQL, or from the
/// same rows built in one `from_rows`; if an append writes rows other than
/// the ones it was given; or if the tile count the appends touch is not the
/// one the positions imply (one extended, one extended, one extended and one
/// added).
#[test]
fn append_equals_postgresql_on_the_union() {
    let (ticket_rows, ticket_deleted) = tickets();
    let agent_rows = agents();
    let agent_gone = agent_gone();

    let one_shot = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .expect("one-shot table");
    assert_eq!(one_shot.rows_written(), ticket_rows.len() as u64);

    let mut appended = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &[],
        &full_config(&one_shot),
    )
    .expect("empty table");
    assert!(appended.is_empty());
    assert_eq!(appended.rows_written(), 0);

    let (mut start, mut touched) = (0, Vec::new());
    for count in [100usize, 150, 50] {
        let before = appended.rows_written();
        let report = appended
            .append(&cells(&ticket_rows[start..start + count]))
            .expect("append");
        assert_eq!(report.rows, count);
        assert_eq!(appended.rows_written() - before, count as u64);
        touched.push(report.tiles_touched);
        start += count;
    }
    assert_eq!(touched, [1, 1, 2]);
    assert_eq!(appended.rows_written(), ticket_rows.len() as u64);

    let from_appends = gate(
        appended,
        agent_table(&agent_rows),
        &ticket_rows,
        &ticket_deleted,
        &agent_rows,
        &agent_gone,
    );
    let from_one_shot = gate(
        one_shot,
        agent_table(&agent_rows),
        &ticket_rows,
        &ticket_deleted,
        &agent_rows,
        &agent_gone,
    );
    assert_eq!(from_appends, from_one_shot);
}

/// FAILS IF: an existing key, a key repeated in the batch, a NULL key, a
/// ragged row or a cell that does not fit is accepted, or a refused append
/// leaves any trace: the length, the reads or the rows written change, or a
/// later valid append is affected.
#[test]
fn append_refuses() {
    let mut table = small_table(&ShapeConfig::default()).expect("table");
    let before = table.select_all().expect("select");
    let written = table.rows_written();

    let refused: [(&[Vec<Option<&str>>], IntakeRefusal); 5] = [
        (
            &[vec![Some("1"), Some("5")]],
            IntakeRefusal::DuplicateKey(1),
        ),
        (
            &[vec![Some("7"), Some("1")], vec![Some("7"), Some("2")]],
            IntakeRefusal::DuplicateKey(7),
        ),
        (&[vec![None, Some("1")]], IntakeRefusal::NoKey),
        (
            &[vec![Some("9")]],
            IntakeRefusal::Ragged { row: 0, cells: 1 },
        ),
        (
            &[vec![Some("8"), Some("5")], vec![Some("9"), Some("x")]],
            IntakeRefusal::ShapeChanged {
                column: "n".to_owned(),
                was: INT4,
                now: TEXT,
            },
        ),
    ];
    for (rows, refusal) in refused {
        assert_eq!(table.append(rows), Err(refusal));
        assert_eq!(table.len(), 3);
        assert_eq!(table.select_all().expect("select"), before);
        assert_eq!(table.rows_written(), written);
    }

    let report = table
        .append(&[vec![Some("4"), Some("40")]])
        .expect("a valid append after refusals");
    assert_eq!((report.rows, report.tiles_touched), (1, 1));
    assert_eq!(table.len(), 4);
    assert_eq!(
        table.rows_for_keys(&[4, 1]).expect("keys").rows,
        vec![vec![text("4"), text("40")], vec![text("1"), text("10")]]
    );
}

/// FAILS IF: `select_ref` does not run the predicate exactly once, a warm
/// `rows_of` runs one, the kept rows differ between reads, or differ from
/// `select_where` on the same filter.
#[test]
fn a_kept_selection_is_reused_without_predicate_work() {
    let (ticket_rows, _) = tickets();
    let mut table = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .expect("table");
    let filter = owner_or_not_negative();

    let runs = table.predicate_runs();
    let kept = table.select_ref(&filter).expect("keep");
    assert_eq!(table.predicate_runs(), runs + 1);
    assert!(
        !kept.is_empty() && kept.len() < ticket_rows.len(),
        "a real subset"
    );

    let first = table.rows_of(&kept).expect("first read");
    let second = table.rows_of(&kept).expect("second read");
    let third = table.rows_of(&kept).expect("third read");
    assert_eq!(
        table.predicate_runs(),
        runs + 1,
        "warm reads run no predicate"
    );
    assert_eq!(first, second);
    assert_eq!(second, third);
    assert_eq!(first.rows.len(), kept.len());
    assert_eq!(table.count_of(&kept), Ok(kept.len()));
    assert_eq!(table.predicate_runs(), runs + 1);

    assert_eq!(first, table.select_where(&filter).expect("select_where"));
    assert_eq!(table.predicate_runs(), runs + 2, "select_where runs one");
}

/// FAILS IF: a selection kept before an append or a delete still reads (or
/// counts), a selection kept after one is refused, or the refusal is anything
/// but `Stale`.
#[test]
fn a_kept_selection_is_refused_after_append_or_delete() {
    let (ticket_rows, _) = tickets();
    let mut table = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .expect("table");
    let filter = owner_or_not_negative();

    let before_append = table.select_ref(&filter).expect("keep");
    assert!(table.rows_of(&before_append).is_ok());
    let mut extra = ticket_rows[0].clone();
    extra[0] = text("301");
    table.append(&cells(&[extra])).expect("append");
    assert_eq!(table.rows_of(&before_append), Err(IntakeRefusal::Stale));
    assert_eq!(table.count_of(&before_append), Err(IntakeRefusal::Stale));

    let after_append = table.select_ref(&filter).expect("keep again");
    let rows = table.rows_of(&after_append).expect("a new ref reads");
    assert_eq!(rows.rows.len(), after_append.len());

    table.delete_key(2).expect("delete");
    assert_eq!(table.rows_of(&after_append), Err(IntakeRefusal::Stale));
    assert_eq!(table.count_of(&after_append), Err(IntakeRefusal::Stale));
    let after_delete = table.select_ref(&filter).expect("keep after delete");
    assert!(table.rows_of(&after_delete).is_ok());
}

/// FAILS IF: a kept bag of keys loses a repeat, reorders its keys, runs a
/// predicate, differs from `rows_for_keys`, or accepts an unknown key.
#[test]
fn a_key_bag_ref_keeps_order_and_repeats() {
    let mut table = small_table(&ShapeConfig::default()).expect("table");
    let runs = table.predicate_runs();
    let kept = table.select_keys_ref(&[3, 1, 3]).expect("keep");
    assert_eq!(kept.len(), 3);
    let rows = table.rows_of(&kept).expect("read");
    assert_eq!(
        rows.rows,
        vec![
            vec![text("3"), None],
            vec![text("1"), text("10")],
            vec![text("3"), None],
        ]
    );
    assert_eq!(rows, table.rows_for_keys(&[3, 1, 3]).expect("bag"));
    assert_eq!(table.predicate_runs(), runs);
    assert_eq!(
        table.select_keys_ref(&[1, 9]).err(),
        Some(IntakeRefusal::UnknownKey(9))
    );
}

fn ticket_table() -> (Table, Vec<Vec<Option<String>>>) {
    let (ticket_rows, _) = tickets();
    let table = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .expect("table");
    (table, ticket_rows)
}

/// FAILS IF: the first WHERE under a data version folds no validity plane or
/// not one per column, or a later WHERE (count, select, another filter) under
/// the same version folds again, or a warm answer differs from the first.
#[test]
fn a_warm_where_folds_no_validity() {
    let (table, _) = ticket_table();
    let filter = owner_or_not_negative();
    let columns = TICKET_COLUMNS.len() as u64;
    assert_eq!(table.validity_folds(), 0);

    let first = table.count_where(&filter).expect("first count");
    assert_eq!(table.validity_folds(), columns);
    assert!(first > 0 && first < 300, "a real subset");

    assert_eq!(table.count_where(&filter), Ok(first));
    let selected = table.select_where(&filter).expect("select");
    assert_eq!(selected.rows.len(), first);
    assert_eq!(table.select_where(&filter), Ok(selected));
    let other = Filter::cmp(Col(5), Cmp::LtI32(0));
    table.count_where(&other).expect("another filter");
    assert_eq!(table.validity_folds(), columns, "no fold after the first");
}

/// FAILS IF: validity planes kept before an append or a delete are used after
/// it (the answer misses the new row or still counts the deleted one), or the
/// recompute folds other than one plane per column, or folds again on the
/// second WHERE under the new version.
#[test]
fn validity_is_recomputed_after_append_or_delete() {
    let (mut table, ticket_rows) = ticket_table();
    let filter = owner_or_not_negative();
    let columns = TICKET_COLUMNS.len() as u64;

    let base = table.count_where(&filter).expect("count");
    assert_eq!(table.validity_folds(), columns);

    let mut extra = ticket_rows[0].clone();
    extra[0] = text("301");
    extra[4] = text("2");
    table.append(&cells(&[extra])).expect("append");
    assert_eq!(table.count_where(&filter), Ok(base + 1));
    assert_eq!(table.validity_folds(), 2 * columns);
    assert_eq!(table.count_where(&filter), Ok(base + 1));
    let kept = table.select_where(&filter).expect("select");
    assert_eq!(kept.rows.len(), base + 1);
    assert!(kept.rows.iter().any(|r| r[0].as_deref() == Some("301")));
    assert_eq!(table.validity_folds(), 2 * columns);

    table.delete_key(301).expect("delete");
    assert_eq!(table.count_where(&filter), Ok(base));
    assert_eq!(table.validity_folds(), 3 * columns);
    assert_eq!(table.count_where(&filter), Ok(base));
    assert_eq!(table.validity_folds(), 3 * columns);
}

/// FAILS IF: the K3 WHERE of the PostgreSQL gate fixture answers differently
/// through kept validity planes than on the cold call that folded them, or
/// than the model rows derived from the input text.
#[test]
fn kept_validity_does_not_change_answers() {
    let (mut table, ticket_rows) = ticket_table();
    let (_, deleted) = tickets();
    for (r, &gone) in deleted.iter().enumerate() {
        if gone {
            table.delete_key(r as u64 + 1).expect("delete");
        }
    }
    let filter = owner_or_not_negative();
    let model: Vec<Row> = (0..ticket_rows.len())
        .filter(|&r| !deleted[r])
        .filter(|&r| {
            let owner = int(&ticket_rows[r], 4).map(|o| o == 2);
            let not_negative = int(&ticket_rows[r], 5).map(|p| p >= 0);
            owner == Some(true) || not_negative == Some(true)
        })
        .map(|r| ticket_rows[r].clone())
        .collect();

    let cold = table.select_where(&filter).expect("cold");
    let folds = table.validity_folds();
    same("kept-validity cold", &model, &cold);
    let warm = table.select_where(&filter).expect("warm");
    assert_eq!(table.validity_folds(), folds, "the warm call folds nothing");
    assert_eq!(warm, cold);
    same("kept-validity warm", &model, &warm);
    assert_eq!(table.count_where(&filter), Ok(model.len()));
    let kept = table.select_ref(&filter).expect("keep");
    assert_eq!(table.validity_folds(), folds);
    assert_eq!(table.rows_of(&kept).expect("rows"), cold);
}

fn live_tickets() -> (Table, Vec<Vec<Option<String>>>, Vec<bool>) {
    let (rows, deleted) = tickets();
    let mut ticket = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&rows),
        &ShapeConfig::default(),
    )
    .expect("ticket table");
    for (r, &gone) in deleted.iter().enumerate() {
        if gone {
            ticket.delete_key(r as u64 + 1).expect("delete");
        }
    }
    (ticket, rows, deleted)
}

fn pg_or_skip() -> Option<Pg> {
    match Pg::connect() {
        Ok(pg) => Some(pg),
        Err(Skip(why)) => {
            assert!(
                std::env::var_os("PG_ROW_REQUIRE_PG").is_none(),
                "PG_ROW_REQUIRE_PG is set and PostgreSQL is unavailable: {why}"
            );
            println!("SKIP (PostgreSQL half): {why}");
            None
        }
    }
}

/// FAILS IF: a literal bound through Quack's membrane keeps a different
/// multiset of rows than PostgreSQL's `WHERE`, if `title = ''` (a present
/// empty string in a NULL-absent column) keeps a NULL title, or if an
/// integer literal binds to a boolean field.
#[test]
fn a_bound_literal_equals_postgresql() {
    let (mut ticket, rows, deleted) = live_tickets();
    let live = |r: usize| !deleted[r];
    type Keep = fn(&[Option<String>]) -> bool;
    let cases: [(&str, lance_graph_quack::bind::Draft, Keep); 7] = [
        (
            "title = 'vpn'",
            draft("ticket").where_eq("title", "vpn"),
            |r| r[1].as_deref() == Some("vpn"),
        ),
        ("title = ''", draft("ticket").where_eq("title", ""), |r| {
            r[1].as_deref() == Some("")
        }),
        (
            "title <> 'printer'",
            draft("ticket").where_cmp("title", lance_graph_quack::bind::Op::Ne, "printer"),
            |r| r[1].as_deref().is_some_and(|t| t != "printer"),
        ),
        (
            "title LIKE 'l%'",
            draft("ticket").where_like("title", "l%"),
            |r| r[1].as_deref().is_some_and(|t| t.starts_with('l')),
        ),
        (
            "priority < 0",
            draft("ticket").where_cmp("priority", lance_graph_quack::bind::Op::Lt, 0i32),
            |r| int(r, 5).is_some_and(|p| p < 0),
        ),
        ("open = true", draft("ticket").where_eq("open", true), |r| {
            r[3].as_deref() == Some("t")
        }),
        (
            "open <> true",
            draft("ticket").where_cmp("open", lance_graph_quack::bind::Op::Ne, true),
            |r| r[3].as_deref() == Some("f"),
        ),
    ];
    let mut ours = Vec::new();
    for (sql, d, keep) in cases {
        let q = ticket.bind(&d).expect("binds");
        let kept = ticket.select_bound(&q).expect("select");
        let got = ticket.rows_of(&kept).expect("rows");
        let model: Vec<Row> = (0..rows.len())
            .filter(|&r| live(r) && keep(&rows[r]))
            .map(|r| rows[r].clone())
            .collect();
        assert!(
            !model.is_empty() && model.len() < ticket.len(),
            "{sql}: a real subset"
        );
        same(sql, &model, &got);
        ours.push((sql, got));
    }
    assert!(matches!(
        ticket.bind(&draft("ticket").where_eq("open", 1i32)),
        Err(IntakeRefusal::Bind(BindError::KindMismatch { .. }))
    ));

    let Some(pg) = pg_or_skip() else { return };
    pg.script(&create(&ticket, &rows, &deleted))
        .expect("create");
    for (sql, got) in ours {
        let theirs = pg
            .query(&format!("SELECT * FROM ticket WHERE {sql}"))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(theirs.types, got.types);
        if let Some(diff) = multiset_diff(&theirs.rows, &got.rows) {
            panic!("{sql}: PostgreSQL differs: {diff}");
        }
        println!("PASS {sql}: {} rows", got.rows.len());
    }
}

/// FAILS IF: on a text column holding present `''` and no NULL, `= ''`,
/// `<> x` or a `LIKE` pattern that matches `''` keeps other rows than
/// PostgreSQL, which keeps `''` as a value.
#[test]
fn present_empty_text_compares_as_postgresql() {
    let (mut ticket, rows, deleted) = live_tickets();
    assert_eq!(ticket.shape()[2].col.absent, AbsentAs::Null);
    type Keep = fn(&str) -> bool;
    let cases: [(&str, lance_graph_quack::bind::Draft, Keep); 4] = [
        ("note = ''", draft("ticket").where_eq("note", ""), |n| {
            n.is_empty()
        }),
        (
            "note <> 'waiting'",
            draft("ticket").where_cmp("note", lance_graph_quack::bind::Op::Ne, "waiting"),
            |n| n != "waiting",
        ),
        (
            "note LIKE '%'",
            draft("ticket").where_like("note", "%"),
            |_| true,
        ),
        (
            "note LIKE 'w%'",
            draft("ticket").where_like("note", "w%"),
            |n| n.starts_with('w'),
        ),
    ];
    let mut ours = Vec::new();
    for (sql, d, keep) in cases {
        let q = ticket.bind(&d).expect("binds");
        let kept = ticket.select_bound(&q).expect("select");
        let got = ticket.rows_of(&kept).expect("rows");
        let model: Vec<Row> = (0..rows.len())
            .filter(|&r| !deleted[r] && rows[r][2].as_deref().is_some_and(keep))
            .map(|r| rows[r].clone())
            .collect();
        assert!(!model.is_empty(), "{sql}: the fixture has rows");
        same(sql, &model, &got);
        ours.push((sql, got));
    }
    assert!(
        ours.iter()
            .any(|(_, g)| g.rows.iter().any(|r| r[2].as_deref() == Some(""))),
        "empty notes are kept"
    );

    let Some(pg) = pg_or_skip() else { return };
    pg.script(&create(&ticket, &rows, &deleted))
        .expect("create");
    for (sql, got) in ours {
        let theirs = pg
            .query(&format!("SELECT * FROM ticket WHERE {sql}"))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        if let Some(diff) = multiset_diff(&theirs.rows, &got.rows) {
            panic!("{sql}: PostgreSQL differs: {diff}");
        }
        println!("PASS {sql}: {} rows", got.rows.len());
    }
}

/// FAILS IF: a query bound before an append is still executed after it, or
/// a query bound again after the append is refused.
#[test]
fn a_bound_query_is_refused_after_the_data_version_moves() {
    let (mut ticket, _, _) = live_tickets();
    let q = ticket
        .bind(&draft("ticket").where_eq("title", "vpn"))
        .expect("binds");
    ticket.select_bound(&q).expect("current");
    ticket
        .append(&[vec![
            Some("1001"),
            Some("vpn"),
            Some("call back"),
            Some("t"),
            Some("1"),
            Some("0"),
            Some("2021-01-01 00:00:00"),
            None,
        ]])
        .expect("append");
    assert!(matches!(ticket.select_bound(&q), Err(IntakeRefusal::Stale)));
    let again = ticket
        .bind(&draft("ticket").where_eq("title", "vpn"))
        .expect("rebinds");
    let kept = ticket.select_bound(&again).expect("rebound");
    let got = ticket.rows_of(&kept).expect("rows");
    assert!(got.rows.iter().any(|r| r[0].as_deref() == Some("1001")));
}

/// FAILS IF: the PostgreSQL egress adapter writes other cells, order or
/// repeats than `rows_of`, spells an absent cell differently, or writes a
/// stale or foreign selection.
#[test]
fn write_selection_equals_rows_of() {
    let (mut ticket, rows, deleted) = live_tickets();
    let keys: Vec<u64> = (0..rows.len())
        .filter(|&r| !deleted[r])
        .map(|r| r as u64 + 1)
        .step_by(7)
        .collect();
    let mut bag = keys.clone();
    bag.extend(keys.iter().rev().step_by(2));
    let kept = ticket.select_keys_ref(&bag).expect("keep");
    let expect = ticket.rows_of(&kept).expect("rows_of");
    assert!(
        expect.rows.iter().any(|r| r[1].is_none()),
        "a NULL title is written"
    );
    assert!(
        expect.rows.iter().any(|r| r[2].as_deref() == Some("")),
        "an absent note is ''"
    );

    let mut sink = CollectRows::new();
    assert_eq!(
        write_selection(&ticket, &kept, &mut sink).expect("write"),
        bag.len()
    );
    assert_eq!(sink.rows, expect.rows);
    let mut copy = CopyText::new();
    write_selection(&ticket, &kept, &mut copy).expect("copy");
    assert_eq!(copy.out.iter().filter(|&&b| b == b'\n').count(), bag.len());

    let (other, _, _) = live_tickets();
    let foreign = {
        let mut other = other;
        other.select_keys_ref(&keys).expect("other keep")
    };
    assert!(matches!(
        write_selection(&ticket, &foreign, &mut CollectRows::new()),
        Err(IntakeRefusal::Stale)
    ));
    ticket.delete_key(keys[0]).expect("delete");
    assert!(matches!(
        write_selection(&ticket, &kept, &mut CollectRows::new()),
        Err(IntakeRefusal::Stale)
    ));
}

// ---- declared types and the absent policy --------------------------------

fn field_rows() -> Vec<Vec<Option<&'static str>>> {
    vec![
        vec![Some("1"), Some("")],
        vec![Some("2"), None],
        vec![Some("3"), Some("abc")],
        vec![Some("4"), Some("")],
    ]
}

fn field_table(absent: Option<AbsentAs>) -> Table {
    let mut config = ShapeConfig::default();
    config.columns.insert(
        "field".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            encoding: None,
            absent,
        },
    );
    Table::from_rows("f", 0x2B0A_0001, &["id", "field"], &field_rows(), &config).expect("table")
}

/// `COUNT(*)`, `COUNT(field)`, `field = ''`, `field IS NULL`, `field = 'abc'`.
fn field_answers(t: &mut Table) -> [usize; 5] {
    let present = t
        .count_where(&Filter::cmp(Col(1), Cmp::NeU32(u32::MAX)))
        .expect("count(field)");
    let mut eq = |text: &str| {
        let q = t.bind(&draft("f").where_eq("field", text)).expect("bind");
        let kept = t.select_bound(&q).expect("select");
        t.count_of(&kept).expect("count")
    };
    let (empty, abc) = (eq(""), eq("abc"));
    [t.len(), present, empty, t.len() - present, abc]
}

/// FAILS IF: a declared nullable text column loses the difference between a
/// present `''` and NULL, or any of the five answers differs from
/// PostgreSQL 16.15 on the same rows.
#[test]
fn fixture_a_nullable_text_keeps_empty_and_null_apart() {
    let mut t = field_table(None);
    assert_eq!(t.shape()[1].col, TEXT);
    let ours = field_answers(&mut t);
    assert_eq!(ours, [4, 3, 2, 1, 1]);

    let Some(pg) = pg_or_skip() else { return };
    pg.script(
        "DROP TABLE IF EXISTS f; CREATE TABLE f (id integer, field text);\
         INSERT INTO f VALUES (1, ''), (2, NULL), (3, 'abc'), (4, '');",
    )
    .expect("create");
    let one = |sql: &str| -> usize {
        pg.query(sql).expect("query").rows[0][0]
            .as_deref()
            .expect("count")
            .parse()
            .expect("number")
    };
    let theirs = [
        one("SELECT count(*) FROM f"),
        one("SELECT count(field) FROM f"),
        one("SELECT count(*) FROM f WHERE field = ''"),
        one("SELECT count(*) FROM f WHERE field IS NULL"),
        one("SELECT count(*) FROM f WHERE field = 'abc'"),
    ];
    assert_eq!(ours, theirs, "PostgreSQL 16.15");
}

/// FAILS IF: a declared empty-as-absent policy is not followed: `''` and
/// NULL must be one absent value, spelled `''`. This is the declared,
/// lossy policy; it is not compared with PostgreSQL.
#[test]
fn fixture_b_declared_empty_as_absent_follows_the_policy() {
    let mut t = field_table(Some(AbsentAs::EmptyText));
    assert_eq!(t.shape()[1].col, TEXT_EMPTY);
    assert_eq!(field_answers(&mut t), [4, 1, 3, 3, 1]);
    let spelled: Vec<Option<String>> = t
        .select_all()
        .expect("rows")
        .rows
        .into_iter()
        .map(|r| r[1].clone())
        .collect();
    assert_eq!(
        spelled,
        [Some(""), Some(""), Some("abc"), Some("")].map(|s| s.map(str::to_owned))
    );
}

const TYPED_COLUMNS: [&str; 5] = ["id", "t", "b", "i", "d"];

fn typed_config() -> ShapeConfig {
    let mut config = ShapeConfig::default();
    for (name, ty) in [
        ("t", PgType::Text),
        ("b", PgType::Bool),
        ("i", PgType::Int4),
        ("d", PgType::Date),
    ] {
        config.columns.insert(
            name.to_owned(),
            ColumnConfig {
                ty: Some(ty),
                ..ColumnConfig::default()
            },
        );
    }
    config
}

/// FAILS IF: a declared type is replaced by what the cells suggest (digits in
/// a text column, `false` in a boolean one), `''`, `false` or `0` is not
/// kept as a present value, an append changes the declared class, an
/// incompatible cell is admitted, or the rows differ from PostgreSQL 16.15
/// with the same declared types.
#[test]
fn fixture_c_declared_types_hold_their_values() {
    let rows: Vec<Vec<Option<&str>>> = vec![
        vec![
            Some("1"),
            Some("0"),
            Some("false"),
            Some("0"),
            Some("2024-01-02"),
        ],
        vec![Some("2"), Some(""), Some("true"), Some("5"), None],
        vec![
            Some("3"),
            Some("7"),
            Some("false"),
            None,
            Some("1999-12-31"),
        ],
    ];
    let mut t = Table::from_rows("typed", 0x2B0A_0002, &TYPED_COLUMNS, &rows, &typed_config())
        .expect("declared");
    let shapes: Vec<PgColumn> = t.shape().iter().skip(1).map(|c| c.col).collect();
    assert_eq!(shapes, [TEXT, BOOL, INT4, DATE]);
    // `''` is not an integer, so detection already reads `t` as text.
    assert_eq!(t.shape()[1].verdict, Verdict::Agrees);
    assert!(
        matches!(t.shape()[2].verdict, Verdict::Differs { .. }),
        "false declared as boolean"
    );
    assert_eq!(t.len(), 3);

    let present = |t: &Table, k: u16| {
        // `x <> <unused value>` is TRUE on every present cell and UNKNOWN on
        // an absent one, so its count is COUNT(x).
        let cmp = if k == 3 {
            Cmp::NeI32(i32::MIN)
        } else {
            Cmp::NeU32(u32::MAX)
        };
        t.count_where(&Filter::cmp(Col(k), cmp)).expect("count")
    };
    assert_eq!(
        [
            present(&t, 1),
            present(&t, 2),
            present(&t, 3),
            present(&t, 4)
        ],
        [3, 3, 2, 2]
    );
    let q = t.bind(&draft("typed").where_eq("i", 0i32)).expect("bind");
    let kept = t.select_bound(&q).expect("select");
    assert_eq!(
        t.rows_of(&kept).expect("rows").rows[0][0].as_deref(),
        Some("1")
    );
    let q = t.bind(&draft("typed").where_eq("t", "")).expect("bind");
    let kept = t.select_bound(&q).expect("select");
    assert_eq!(t.count_of(&kept).expect("count"), 1);

    let before = t.select_all().expect("rows");
    t.append(&[vec![Some("4"), Some("8"), Some("t"), Some("1"), None]])
        .expect("append keeps the class");
    assert_eq!(t.shape()[1].col, TEXT);
    let refused = t.append(&[vec![Some("5"), Some("x"), Some("f"), Some("abc"), None]]);
    assert!(
        matches!(refused, Err(IntakeRefusal::ShapeChanged { ref column, .. }) if column == "i")
    );
    let refused = t.append(&[vec![Some("5"), Some("x"), Some("maybe"), Some("1"), None]]);
    assert!(
        matches!(refused, Err(IntakeRefusal::ShapeChanged { ref column, .. }) if column == "b")
    );
    assert_eq!(t.len(), 4);
    let after = t.select_all().expect("rows");
    assert_eq!(&after.rows[..3], &before.rows[..]);

    let Some(pg) = pg_or_skip() else { return };
    pg.script(
        "DROP TABLE IF EXISTS typed;\
         CREATE TABLE typed (id integer, t text, b boolean, i integer, d date);\
         INSERT INTO typed VALUES (1, '0', 'false', 0, '2024-01-02'),\
           (2, '', 'true', 5, NULL), (3, '7', 'false', NULL, '1999-12-31'),\
           (4, '8', 't', 1, NULL);",
    )
    .expect("create");
    let theirs = pg.query("SELECT * FROM typed").expect("query");
    assert_eq!(theirs.types, after.types);
    if let Some(diff) = multiset_diff(&theirs.rows, &after.rows) {
        panic!("typed: PostgreSQL differs: {diff}");
    }
}

/// FAILS IF: a read by key or of a kept selection needs `&mut Table`
/// (`identity[index]` is a read), or two reads through shared references
/// see different rows. A `&mut self` read does not compile here.
#[test]
fn reads_go_through_a_shared_table() {
    let mut table = small_table(&ShapeConfig::default()).expect("table");
    let kept = table.select_keys_ref(&[3, 1]).expect("keep");
    let (a, b): (&Table, &Table) = (&table, &table);
    let by_key = a.rows_for_keys(&[3, 1]).expect("by key");
    let by_ref = b.rows_of(&kept).expect("kept");
    assert_eq!(by_key, by_ref);
    assert_eq!(
        by_key.rows,
        vec![vec![text("3"), None], vec![text("1"), text("10")],]
    );
}

/// One registered profile: `id`, `region`, `seats`, named by source IRIs
/// that do not mention any table.
fn license_profile() -> std::sync::Arc<SourceProfile> {
    let mut config = ShapeConfig {
        key: Some("id".to_owned()),
        ..ShapeConfig::default()
    };
    config.columns.insert(
        "region".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            encoding: Some(LaneEncoding::TextCode),
            absent: Some(AbsentAs::Null),
        },
    );
    SourceProfile::new(
        0x2B09_0001,
        &[
            ("id", "graph:license/id"),
            ("region", "graph:license/region"),
            ("seats", "graph:license/prepaidUnits"),
        ],
        config,
    )
    .expect("profile")
}

/// FAILS IF: two tenants of one profile do not share its class and field
/// IRIs, share a frame store or a code table, accept each other's frame, or
/// decode each other's codes; or a table whose columns are not the
/// profile's labels is accepted.
#[test]
fn tenants_share_a_profile_but_never_a_frame_or_a_code() {
    let profile = license_profile();
    let cols = ["id", "region", "seats"];
    let a = Table::from_rows_in(
        "tenant_a",
        &profile,
        &cols,
        &[
            vec![Some("1"), Some("north"), Some("10")],
            vec![Some("2"), Some("south"), Some("5")],
        ],
    )
    .expect("a");
    let b = Table::from_rows_in(
        "tenant_b",
        &profile,
        &cols,
        &[
            vec![Some("1"), Some("south"), Some("3")],
            vec![Some("2"), Some("east"), None],
        ],
    )
    .expect("b");
    assert_eq!(
        std::sync::Arc::strong_count(&profile),
        3,
        "one profile, two tenants"
    );

    let iris = |t: &Table| -> Vec<String> {
        let w = t.witness();
        let class = lance_graph_dt64::view_class(w.binding.class());
        lance_graph_contract::class_view::ClassView::fields(w.view, class)
            .iter()
            .map(|f| f.predicate_iri.clone())
            .collect()
    };
    assert_eq!(iris(&a), iris(&b));
    assert_eq!(iris(&a)[2], "graph:license/prepaidUnits");

    assert_eq!(
        b.witness().frames.check(&a.frame()),
        Err(lance_graph_dt64::dr::FrameRefusal::ForeignFrame)
    );
    assert!(a.witness().frames.check(&a.frame()).is_ok());

    // Each tenant numbers its own strings: code 1 is `north` in a, `south`
    // in b, and a's frame decodes nothing in b.
    assert_eq!(a.code_key("region", 1).as_deref(), Some("north"));
    assert_eq!(b.code_key("region", 1).as_deref(), Some("south"));
    assert_eq!(b.code_key_at(&a.frame(), "region", 1), None);
    assert_eq!(
        b.code_for(&a.frame(), "graph:license/region", "south"),
        None
    );
    assert_eq!(
        b.code_for(&b.frame(), "graph:license/region", "south"),
        Some(1)
    );

    assert!(matches!(
        Table::from_rows_in("c", &profile, &["id", "seats", "region"], &[]),
        Err(IntakeRefusal::ProfileMismatch { .. })
    ));
}

/// FAILS IF: a profile whose two fields share a predicate IRI is accepted;
/// both would read the first field's lane (codex on #2194).
#[test]
fn a_profile_refuses_two_fields_with_one_iri() {
    assert!(matches!(
        SourceProfile::new(
            0x2B09_0002,
            &[("id", "source:id"), ("other", "source:id")],
            ShapeConfig::default()
        ),
        Err(IntakeRefusal::DuplicateIri(iri)) if iri == "source:id"
    ));
    assert!(SourceProfile::new(
        0x2B09_0002,
        &[("id", "source:id"), ("other", "source:other")],
        ShapeConfig::default()
    )
    .is_ok());
}

/// The ticket rows as typed values, by column: integers and booleans as
/// values, text as text, and the timestamp and date in their own spelling.
fn typed(row: &[Option<String>]) -> Vec<Option<Datum<'_>>> {
    row.iter()
        .enumerate()
        .map(|(k, c)| {
            c.as_deref().map(|t| match TICKET_COLUMNS[k] {
                "id" | "owner_id" | "priority" => Datum::Int(t.parse().unwrap()),
                "open" => Datum::Bool(t == "t"),
                _ => Datum::Text(t),
            })
        })
        .collect()
}

/// FAILS IF: typed rows appended with `append_values` answer PostgreSQL's
/// queries differently from the same rows appended as text.
#[test]
fn typed_append_equals_text_append() {
    let (ticket_rows, ticket_deleted) = tickets();
    let agent_rows = agents();
    let agent_gone = agent_gone();
    let one_shot = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .unwrap();
    let mut typed_table = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &[],
        &full_config(&one_shot),
    )
    .unwrap();
    for chunk in ticket_rows.chunks(120) {
        let rows: Vec<_> = chunk.iter().map(|r| typed(r)).collect();
        assert_eq!(typed_table.append_values(&rows).unwrap().rows, chunk.len());
    }
    let from_values = gate(
        typed_table,
        agent_table(&agent_rows),
        &ticket_rows,
        &ticket_deleted,
        &agent_rows,
        &agent_gone,
    );
    let from_text = gate(
        one_shot,
        agent_table(&agent_rows),
        &ticket_rows,
        &ticket_deleted,
        &agent_rows,
        &agent_gone,
    );
    assert_eq!(from_values, from_text);
}

/// FAILS IF: a typed cell that does not fit its declared column is placed
/// (an integer out of range or into a boolean or text-code lane, a boolean
/// into an integer lane, text that is not an integer), a ragged row is
/// accepted, or a refused append changes the table; or `check_values`
/// disagrees with `append_values`.
#[test]
fn typed_append_refuses_what_does_not_fit() {
    let (ticket_rows, _) = tickets();
    let one_shot = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &ShapeConfig::default(),
    )
    .unwrap();
    let mut t = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &[],
        &full_config(&one_shot),
    )
    .unwrap();
    let good = || -> Vec<Option<Datum<'static>>> {
        vec![
            Some(Datum::Int(9001)),
            Some(Datum::Text("vpn")),
            None,
            Some(Datum::Bool(true)),
            None,
            Some(Datum::Int(2)),
            None,
            None,
        ]
    };
    let with = |k: usize, d: Datum<'static>| {
        let mut r = good();
        r[k] = Some(d);
        r
    };
    for bad in [
        with(5, Datum::Int(i64::from(i32::MAX) + 1)),
        with(3, Datum::Int(1)),
        with(1, Datum::Int(1)),
        with(5, Datum::Bool(true)),
        with(5, Datum::Text("many")),
    ] {
        assert!(matches!(
            t.check_values(std::slice::from_ref(&bad)),
            Err(IntakeRefusal::Cell { .. })
        ));
        assert!(matches!(
            t.append_values(&[bad]),
            Err(IntakeRefusal::Cell { .. })
        ));
    }
    assert!(matches!(
        t.append_values(&[good()[..7].to_vec()]),
        Err(IntakeRefusal::Ragged { row: 0, cells: 7 })
    ));
    assert_eq!(t.len(), 0);
    assert_eq!(t.check_values(&[good()]), Ok(()));
    assert_eq!(t.len(), 0, "check_values appends nothing");
    assert_eq!(t.append_values(&[good()]).unwrap().rows, 1);
    assert_eq!(t.len(), 1);
}

/// FAILS IF: `positions_bound` answers a query bound before an append or a
/// delete, or bound against another table, instead of refusing it; the
/// current query is the control.
#[test]
fn positions_bound_refuses_a_stale_or_foreign_query() {
    let (mut ticket, _) = ticket_table();
    let q = ticket
        .bind(&draft("ticket").where_eq("priority", 1i32))
        .unwrap();
    let now = ticket.positions_bound(&q).unwrap();
    assert!(!now.is_empty());
    let kept = ticket.select_bound(&q).unwrap();
    assert_eq!(now.len(), ticket.count_of(&kept).unwrap());

    let (other, _) = ticket_table();
    let foreign = other
        .bind(&draft("ticket").where_eq("priority", 1i32))
        .unwrap();
    assert!(ticket.positions_bound(&foreign).is_err());

    ticket.delete_key(1).unwrap();
    assert_eq!(ticket.positions_bound(&q), Err(IntakeRefusal::Stale));
    let q = ticket
        .bind(&draft("ticket").where_eq("priority", 1i32))
        .unwrap();
    let row = vec![
        Some("9001".to_owned()),
        None,
        None,
        None,
        None,
        Some("1".to_owned()),
        None,
        None,
    ];
    ticket.append(&cells(&[row])).unwrap();
    assert_eq!(ticket.positions_bound(&q), Err(IntakeRefusal::Stale));
}
