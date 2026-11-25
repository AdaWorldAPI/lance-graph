//! The glue, the route and the convergence of the two paths: the same VRL
//! program run in-process by lance-graph-vrl (the polyfill) and run by
//! Vector, whose output lands through the route.

use std::sync::Arc;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgType};
use lance_graph_intake::{ColumnConfig, ShapeConfig, SourceProfile, Table};
use lance_graph_vector::{
    output_profile, vector_remap, Ack, Cell, Event, GlueRefusal, Refusal, VectorRoute, EVENT_ID,
};
use lance_graph_vrl::{Datum, Recipe, Witness};

const SOURCE: u32 = 0x2B0A_0001;
const OUTPUT: u32 = 0x2B0A_0002;

/// Normalize: rename region to host and amount to amt, drop negative amounts.
const PROGRAM: &str = ".host = .properties.region.value
.amt = .properties.amount.value
if .properties.amount.value < 0 { abort }";

fn declared(ty: PgType, encoding: LaneEncoding) -> ColumnConfig {
    ColumnConfig {
        ty: Some(ty),
        encoding: Some(encoding),
        absent: Some(AbsentAs::Null),
    }
}

fn source_profile(declare_region: bool) -> Arc<SourceProfile> {
    let mut config = ShapeConfig {
        key: Some("id".to_owned()),
        ..ShapeConfig::default()
    };
    config
        .columns
        .insert("id".into(), declared(PgType::Int4, LaneEncoding::Signed));
    config.columns.insert(
        "amount".into(),
        declared(PgType::Int4, LaneEncoding::Signed),
    );
    if declare_region {
        config.columns.insert(
            "region".into(),
            declared(PgType::Text, LaneEncoding::TextCode),
        );
    }
    SourceProfile::new(
        SOURCE,
        &[
            ("id", "dd:host/id"),
            ("region", "dd:host/region"),
            ("amount", "dd:host/cpu_milli"),
        ],
        config,
    )
    .unwrap()
}

/// The raw events, as Datadog would send them.
fn raw() -> Vec<Vec<Option<&'static str>>> {
    vec![
        vec![Some("1"), Some("north"), Some("12")],
        vec![Some("2"), Some("south"), Some("-3")],
        vec![Some("3"), Some("north"), Some("7")],
        vec![Some("4"), Some("east"), Some("40")],
    ]
}

/// What Vector emits for `raw()` under `vector_remap(PROGRAM, ..)`: the
/// dropped event is gone and the projection leaves only `.lg` and the outputs.
/// Written by hand, so it is an oracle independent of the polyfill. An absent
/// amount is left out: how upstream VRL treats `null < 0` is the S1
/// differential's question, not this one.
const VECTOR_OUT: &str = r#"{"lg":{"event_id":"e1","schema_rev":"1"},"host":"north","amt":12}
{"lg":{"event_id":"e3","schema_rev":"1"},"host":"north","amt":7}
{"lg":{"event_id":"e4","schema_rev":"1"},"host":"east","amt":40}
"#;

fn recipe(profile: &Arc<SourceProfile>) -> (Recipe, Table) {
    let t = Table::from_rows_in("raw", profile, &["id", "region", "amount"], &raw()).unwrap();
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    let r = Recipe::compile(PROGRAM, t.witness().view, class).unwrap();
    (r, t)
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

fn route() -> VectorRoute {
    let profile = source_profile(true);
    let (r, _) = recipe(&profile);
    let out = output_profile(&r, &profile, OUTPUT).unwrap();
    VectorRoute::new("tenant_a", &out, "1").unwrap()
}

/// `(host, amt)` per row of a table under the output profile, read through a
/// VRL projection over its lanes.
fn landed(t: &Table) -> Vec<(String, Option<i32>)> {
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    let r = Recipe::compile(
        ".h = .properties.host.value\n.a = .properties.amt.value",
        t.witness().view,
        class,
    )
    .unwrap();
    let rows = r.materialize_rows(&witness(t)).unwrap();
    let col = |n: &str| rows.names().iter().position(|x| x == n).unwrap();
    (0..rows.len())
        .map(|i| {
            let host = match rows.get(i, col("h")) {
                Datum::U32(c) => t.code_key("host", c).unwrap(),
                other => panic!("host is text, got {other:?}"),
            };
            let amt = rows
                .present(i, col("a"))
                .then(|| rows.get(i, col("a")).as_i32().unwrap());
            (host, amt)
        })
        .collect()
}

/// FAILS IF: the output profile does not carry the source fields' IRIs
/// under the program's output names, or an output over an undeclared source
/// field is accepted.
#[test]
fn the_program_declares_the_profile_its_output_lands_in() {
    let profile = source_profile(true);
    let (r, _) = recipe(&profile);
    let out = output_profile(&r, &profile, OUTPUT).unwrap();
    let fields: Vec<(&str, &str)> = out.fields().collect();
    assert_eq!(
        fields,
        [
            ("amt", "dd:host/cpu_milli"),
            ("host", "dd:host/region"),
            EVENT_ID,
            ("lg.seq", "lg:seq"),
        ]
    );

    let undeclared = source_profile(false);
    let (r, _) = recipe(&undeclared);
    assert_eq!(
        output_profile(&r, &undeclared, OUTPUT).map(|_| ()),
        Err(GlueRefusal::Undeclared {
            output: "host".into(),
            field: "region".into()
        })
    );
}

/// FAILS IF: Vector's output lands as other rows than the polyfill computes
/// in-process for the same program over the same raw events.
#[test]
fn vector_output_and_the_polyfill_converge() {
    let profile = source_profile(true);
    let (r, raw_table) = recipe(&profile);
    let poly = r.materialize_rows(&witness(&raw_table)).unwrap();
    let col = |n: &str| poly.names().iter().position(|x| x == n).unwrap();
    let polyfill: Vec<(String, Option<i32>)> = (0..poly.len())
        .map(|i| {
            let host = match poly.get(i, col("host")) {
                Datum::U32(c) => raw_table.code_key("region", c).unwrap(),
                other => panic!("{other:?}"),
            };
            let amt = poly
                .present(i, col("amt"))
                .then(|| poly.get(i, col("amt")).as_i32().unwrap());
            (host, amt)
        })
        .collect();

    let mut route = route();
    assert_eq!(route.receive(VECTOR_OUT).status(), 200);
    let landed = landed(route.table());
    assert_eq!(landed, polyfill);
    assert_eq!(
        landed,
        [
            ("north".to_owned(), Some(12)),
            ("north".to_owned(), Some(7)),
            ("east".to_owned(), Some(40)),
        ]
    );
}

/// FAILS IF: a retried batch appends again, a replay with a misfit value is
/// acknowledged, a repeat inside one batch is appended twice, or a wrong
/// schema revision or an unknown field appends anything or answers other
/// than 400.
#[test]
fn a_retry_changes_nothing_and_a_refusal_appends_nothing() {
    let mut route = route();
    assert_eq!(
        route.receive(VECTOR_OUT),
        Ack::Committed {
            appended: 3,
            duplicates: 0
        }
    );
    let retry = route.receive(VECTOR_OUT);
    assert_eq!(
        retry,
        Ack::Committed {
            appended: 0,
            duplicates: 3
        }
    );
    assert_eq!(retry.status(), 200);
    assert_eq!(route.table().len(), 3);

    let repeat = r#"{"lg":{"event_id":"e9","schema_rev":"1"},"host":"west","amt":1}
{"lg":{"event_id":"e9","schema_rev":"1"},"host":"west","amt":1}"#;
    assert_eq!(
        route.receive(repeat),
        Ack::Committed {
            appended: 1,
            duplicates: 1
        }
    );

    let bad_replay = r#"{"lg":{"event_id":"e1","schema_rev":"1"},"host":"north","amt":"many"}"#;
    assert!(matches!(
        route.receive(bad_replay),
        Ack::Refused(Refusal::Intake(_))
    ));

    let stale = r#"{"lg":{"event_id":"e10","schema_rev":"2"},"host":"west","amt":1}"#;
    let refused = route.receive(stale);
    assert_eq!(refused.status(), 400);
    assert!(matches!(
        refused,
        Ack::Refused(Refusal::SchemaRev { line: 1, .. })
    ));
    let unknown = r#"{"lg":{"event_id":"e11","schema_rev":"1"},"cpu":1}"#;
    assert!(matches!(
        route.receive(unknown),
        Ack::Refused(Refusal::UnknownField { .. })
    ));
    let misfit = r#"{"lg":{"event_id":"e12","schema_rev":"1"},"host":"x","amt":"many"}"#;
    assert!(matches!(
        route.receive(misfit),
        Ack::Refused(Refusal::Intake(_))
    ));
    let nested = r#"{"lg":{"event_id":"e13","schema_rev":"1"},"host":{"name":"x"}}"#;
    assert!(matches!(
        route.receive(nested),
        Ack::Refused(Refusal::NotScalar { line: 1, .. })
    ));
    assert_eq!(route.table().len(), 4, "refused batches appended nothing");
}

/// FAILS IF: typed events handed over by a sink inside Vector land other
/// than the same events through NDJSON.
#[test]
fn typed_events_land_as_ndjson_does() {
    let mut typed = route();
    let events = [
        Event {
            event_id: "e1",
            schema_rev: Some("1"),
            fields: vec![("host", Cell::Text("north")), ("amt", Cell::Int(12))],
        },
        Event {
            event_id: "e3",
            schema_rev: Some("1"),
            fields: vec![("host", Cell::Text("north")), ("amt", Cell::Int(7))],
        },
        Event {
            event_id: "e4",
            schema_rev: Some("1"),
            fields: vec![("host", Cell::Text("east")), ("amt", Cell::Number("40"))],
        },
    ];
    assert_eq!(
        typed.receive_events(&events),
        Ack::Committed {
            appended: 3,
            duplicates: 0
        }
    );
    let mut json = route();
    json.receive(VECTOR_OUT);
    assert_eq!(landed(typed.table()), landed(json.table()));

    let absent = [Event {
        event_id: "e5",
        schema_rev: Some("1"),
        fields: vec![("host", Cell::Text("west")), ("amt", Cell::Absent)],
    }];
    typed.receive_events(&absent);
    json.receive(r#"{"lg":{"event_id":"e5","schema_rev":"1"},"host":"west","amt":null}"#);
    assert_eq!(landed(typed.table())[3], ("west".to_owned(), None));
    assert_eq!(landed(typed.table()), landed(json.table()));
}

/// FAILS IF: the landed table is not queryable through Quack's binder: the
/// route's table is an intake table, bound through its BindStore.
#[test]
fn quack_reads_what_vector_landed() {
    use lance_graph_quack::bind::{table as draft, Op};
    let mut route = route();
    route.receive(VECTOR_OUT);
    let t = route.table_mut();
    let count = |t: &mut Table, d| {
        let q = t.bind(&d).unwrap();
        let r = t.select_bound(&q).unwrap();
        t.count_of(&r).unwrap()
    };
    assert_eq!(count(t, draft("tenant_a").where_eq("host", "north")), 2);
    assert_eq!(
        count(t, draft("tenant_a").where_cmp("amt", Op::Gt, 20i32)),
        1
    );
}

/// FAILS IF: the remap Vector runs does not end in a projection onto `.lg`
/// and the recipe's outputs, or the route accepts a remap output that still
/// carries the source object it read from (so the projection is not needed).
#[test]
fn the_vector_remap_projects_onto_the_outputs() {
    let profile = source_profile(true);
    let (r, _) = recipe(&profile);
    assert_eq!(
        vector_remap(PROGRAM, &r),
        format!("{PROGRAM}\n. = {{\"lg\": .lg, \"amt\": .amt, \"host\": .host}}\n")
    );

    let unprojected = r#"{"lg":{"event_id":"e1","schema_rev":"1"},"host":"north","amt":12,"properties":{"region":{"value":"north"},"amount":{"value":12}}}"#;
    let mut route = route();
    assert!(matches!(
        route.receive(unprojected),
        Ack::Refused(Refusal::NotScalar { line: 1, .. })
    ));
    assert_eq!(route.table().len(), 0);
}
