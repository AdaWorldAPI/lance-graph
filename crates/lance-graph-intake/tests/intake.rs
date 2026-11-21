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
use lance_graph_intake::{
    detect_column, detect_key, ColumnConfig, IntakeRefusal, ShapeConfig, Table, Verdict,
};
use lance_graph_pg_row::{PgRowRefusal, PgRows};
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

/// FAILS IF: `''` with no NULL cell is not absent-as-empty, or `''` beside a
/// NULL cell is, or a non-text column takes the empty spelling.
#[test]
fn empty_strings_decide_the_absent_spelling() {
    assert_eq!(
        detect_column(&[Some("a"), Some(""), Some("b")]).col,
        TEXT_EMPTY
    );
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
    assert_eq!(n.detected.col, INT4);
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
            absent: Some(AbsentAs::Null),
        },
    );
    let table = small_table(&config).expect("table");
    assert_eq!(table.shape()[1].verdict, Verdict::Agrees);
    assert_eq!(table.shape()[0].verdict, Verdict::DetectedOnly);
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

    let mut config = ShapeConfig::default();
    config.columns.insert(
        "n".to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            ..ColumnConfig::default()
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
    let mut agent_deleted = vec![false; AGENT_NAMES.len()];
    agent_deleted[2] = true;
    let agent_rows = agents();

    let config = ShapeConfig::default();
    let mut ticket = Table::from_rows(
        "ticket",
        0x2B01_0001,
        &TICKET_COLUMNS,
        &cells(&ticket_rows),
        &config,
    )
    .expect("ticket table");
    let mut agent = Table::from_rows(
        "agent",
        0x2B02_0001,
        &AGENT_COLUMNS,
        &cells(&agent_rows),
        &config,
    )
    .expect("agent table");

    let shapes: Vec<PgColumn> = ticket.shape().iter().map(|c| c.col).collect();
    assert_eq!(
        shapes,
        [INT4, TEXT, TEXT_EMPTY, BOOL, INT4, INT4, TIMESTAMP, DATE]
    );
    assert!(ticket
        .shape()
        .iter()
        .all(|c| c.verdict == Verdict::DetectedOnly));
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
        .zip(&agent_deleted)
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
            return;
        }
    };
    println!("PostgreSQL {}", pg.version());
    pg.script(
        &(create(&ticket, &ticket_rows, &ticket_deleted)
            + &create(&agent, &agent_rows, &agent_deleted)),
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
}
