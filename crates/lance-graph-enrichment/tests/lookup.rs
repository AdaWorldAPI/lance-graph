//! Vector's enrichment lookups answered by an intake table through Quack:
//! direct trait calls, and the same lookup made by VRL itself.

use std::collections::HashMap;

use enrichment::{Case, Condition, Error, Table as _, TableRegistry};
use lance_graph_enrichment::QuackTable;
use lance_graph_intake::{ShapeConfig, Table};
use vrl::compiler::runtime::Runtime;
use vrl::compiler::state::ExternalEnv;
use vrl::compiler::{compile_with_external, CompileConfig, TargetValue, TimeZone};
use vrl::value::{ObjectMap, Secrets, Value};

fn hosts() -> Table {
    let rows: Vec<Vec<Option<&str>>> = vec![
        vec![Some("1"), Some("a1"), Some("eu"), Some("40"), Some("t")],
        vec![Some("2"), Some("b2"), Some("us"), Some("75"), Some("f")],
        vec![Some("3"), Some("c3"), Some("eu"), None, Some("t")],
    ];
    Table::from_rows(
        "hosts",
        0x2B0E_0001,
        &["key", "host", "region", "cpu", "active"],
        &rows,
        &ShapeConfig::default(),
    )
    .unwrap()
}

fn eq<'a>(field: &'a str, value: Value) -> Condition<'a> {
    Condition::Equals { field, value }
}

fn row(pairs: &[(&str, Value)]) -> ObjectMap {
    pairs
        .iter()
        .map(|(k, v)| ((*k).into(), v.clone()))
        .collect()
}

/// FAILS IF: a lookup returns other rows than the table holds for the
/// condition, a value comes back as another type than its lane (text as
/// text, integer as integer, boolean as boolean, absent as null), or `select`
/// returns other columns.
#[test]
fn lookups_return_the_rows_quack_selects_as_typed_values() {
    let t = QuackTable::new(hosts()).unwrap();
    let a1 = t
        .find_table_row(
            Case::Sensitive,
            &[eq("host", Value::from("a1"))],
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        a1,
        row(&[
            ("key", Value::Integer(1)),
            ("host", Value::from("a1")),
            ("region", Value::from("eu")),
            ("cpu", Value::Integer(40)),
            ("active", Value::Boolean(true)),
        ])
    );
    let eu = t
        .find_table_rows(
            Case::Sensitive,
            &[eq("region", Value::from("eu"))],
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(eu.len(), 2);
    assert_eq!(eu[1]["cpu"], Value::Null, "an absent cell is null");
    let both = t
        .find_table_rows(
            Case::Sensitive,
            &[
                eq("region", Value::from("eu")),
                eq("active", Value::Boolean(true)),
            ],
            Some(&["host".to_owned()]),
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        both,
        [
            row(&[("host", Value::from("a1"))]),
            row(&[("host", Value::from("c3"))])
        ]
    );
    let by_int = t
        .find_table_rows(
            Case::Sensitive,
            &[eq("cpu", Value::Integer(75))],
            Some(&["host".to_owned()]),
            None,
            None,
        )
        .unwrap();
    assert_eq!(by_int, [row(&[("host", Value::from("b2"))])]);
}

/// FAILS IF: a lookup that is not plain equality, or names a field the table
/// lacks, is answered instead of refused, or a single-row lookup does not
/// distinguish none from several.
#[test]
fn what_is_not_equality_is_refused() {
    let t = QuackTable::new(hosts()).unwrap();
    let find = |case, c: &[Condition<'_>]| t.find_table_row(case, c, None, None, None);
    assert_eq!(
        find(Case::Sensitive, &[eq("host", Value::from("zz"))]),
        Err(Error::NoRowsFound)
    );
    assert_eq!(
        find(Case::Sensitive, &[eq("region", Value::from("eu"))]),
        Err(Error::MoreThanOneRowFound)
    );
    assert_eq!(
        find(Case::Sensitive, &[eq("nope", Value::Integer(1))]),
        Err(Error::MissingDatasetFields {
            fields: vec!["nope".into()]
        })
    );
    let now = chrono::DateTime::from_timestamp(0, 0).unwrap();
    assert_eq!(
        find(
            Case::Sensitive,
            &[Condition::FromDate {
                field: "host",
                from: now
            }]
        ),
        Err(Error::OnlyEqualityConditionAllowed)
    );
    assert!(matches!(
        find(Case::Insensitive, &[eq("host", Value::from("A1"))]),
        Err(Error::FailedToEncodeValue { .. })
    ));
    assert!(matches!(
        t.find_table_row(
            Case::Sensitive,
            &[eq("host", Value::from("a1"))],
            None,
            Some(&Value::from("*")),
            None
        ),
        Err(Error::FailedToEncodeValue { .. })
    ));
}

/// FAILS IF: VRL's own `get_enrichment_table_record`, compiled against
/// Vector's registry holding the intake table, returns another record than
/// the direct lookup.
#[test]
fn vrl_reads_the_intake_table_through_the_registry() {
    let registry = TableRegistry::default();
    let mut tables: HashMap<String, Box<dyn enrichment::Table + Send + Sync>> = HashMap::new();
    tables.insert("hosts".into(), Box::new(QuackTable::new(hosts()).unwrap()));
    registry.load(tables);
    let mut config = CompileConfig::default();
    config.set_custom(registry.clone());
    let program = compile_with_external(
        r#". = get_enrichment_table_record!("hosts", {"host": .host}, select: ["region", "cpu"])"#,
        &enrichment::vrl_functions(),
        &ExternalEnv::default(),
        config,
    )
    .unwrap_or_else(|d| {
        panic!(
            "{:?}",
            d.iter()
                .map(|x| (
                    x.code,
                    x.message.clone(),
                    x.labels
                        .iter()
                        .map(|l| l.message.clone())
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>()
        )
    })
    .program;
    registry.finish_load();
    let mut target = TargetValue {
        value: Value::Object(row(&[("host", Value::from("b2"))])),
        metadata: Value::Object(ObjectMap::new()),
        secrets: Secrets::default(),
    };
    Runtime::default()
        .resolve(&mut target, &program, &TimeZone::default())
        .unwrap();
    assert_eq!(
        target.value,
        Value::Object(row(&[
            ("region", Value::from("us")),
            ("cpu", Value::Integer(75))
        ]))
    );
}
