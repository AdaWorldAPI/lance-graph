//! Presence, case by case, through both engines: an integer, a boolean and a
//! text field holding a value, its zero (`0`, `false`, `""`), an explicit
//! `null` or no key at all, under each comparison the recipe lowers.
//!
//! Where the value is present, the two engines must agree. Where it is
//! absent, the recipe leaves the row unknown and upstream reads `null`; each
//! such pair is pinned in [`ABSENT`], and the admission gate refuses every
//! program whose answer depends on it (Q1).
//!
//! Run with `--features upstream-vector` or `--features upstream-o2`.
#![cfg(any(feature = "upstream-vector", feature = "upstream-o2"))]

#[cfg(all(feature = "upstream-o2", not(feature = "upstream-vector")))]
use vrl_o2 as vrl;
#[cfg(feature = "upstream-vector")]
use vrl_vector as vrl;

use std::collections::BTreeMap;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgType};
use lance_graph_intake::{ColumnConfig, ShapeConfig, Table};
use lance_graph_vrl::{Declared, DeclaredKind, Recipe, Refusal, Witness};
use vrl::compiler::runtime::{Runtime, Terminate};
use vrl::compiler::state::ExternalEnv;
use vrl::compiler::{compile_with_external, CompileConfig, TargetValue, TimeZone};
use vrl::value::kind::{Collection, Field};
use vrl::value::{Kind, ObjectMap, Secrets, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Keep,
    Drop,
    Unknown,
}
use Outcome::*;

/// The programs, each over one field.
const PROGRAMS: &[(&str, &str)] = &[
    ("n", "if .properties.n.value == 0 { abort }"),
    ("n", "if .properties.n.value != 0 { abort }"),
    ("n", "if .properties.n.value > 0 { abort }"),
    ("b", "if .properties.b.value { abort }"),
    ("b", "if !.properties.b.value { abort }"),
    ("b", "if .properties.b.value == false { abort }"),
    ("t", "if .properties.t.value == \"\" { abort }"),
    ("t", "if .properties.t.value != \"\" { abort }"),
];

/// The cases for a field: a value, its zero, and absent.
fn cases(field: &str) -> [(&'static str, Option<&'static str>); 3] {
    match field {
        "n" => [("value", Some("7")), ("zero", Some("0")), ("absent", None)],
        "b" => [("value", Some("t")), ("zero", Some("f")), ("absent", None)],
        _ => [("value", Some("x")), ("zero", Some("")), ("absent", None)],
    }
}

/// Upstream's outcome for an absent value, per program: as no key, and as an
/// explicit `null`. The recipe answers `Unknown` for every one.
///
/// Missing and `null` are read alike. An order compare or a bare truth test
/// over an absent value is a runtime error upstream, which is `Unknown` as
/// the recipe has it: they agree. Every `==`/`!=` diverges: upstream compares
/// `null` as a value (`null == 0` is false, `null != 0` true), which is Q1.
const ABSENT: &[(&str, Outcome, Outcome)] = &[
    ("if .properties.n.value == 0 { abort }", Keep, Keep),
    ("if .properties.n.value != 0 { abort }", Drop, Drop),
    ("if .properties.n.value > 0 { abort }", Unknown, Unknown),
    ("if .properties.b.value { abort }", Unknown, Unknown),
    ("if !.properties.b.value { abort }", Unknown, Unknown),
    ("if .properties.b.value == false { abort }", Keep, Keep),
    ("if .properties.t.value == \"\" { abort }", Keep, Keep),
    ("if .properties.t.value != \"\" { abort }", Drop, Drop),
];

fn declared(ty: PgType, encoding: LaneEncoding) -> ColumnConfig {
    ColumnConfig {
        ty: Some(ty),
        encoding: Some(encoding),
        absent: Some(AbsentAs::Null),
    }
}

/// A one-row table holding `cell` in `field` and a value in the others.
fn table(field: &str, cell: Option<&str>) -> Table {
    let mut config = ShapeConfig {
        key: Some("id".to_owned()),
        ..ShapeConfig::default()
    };
    config
        .columns
        .insert("id".into(), declared(PgType::Int4, LaneEncoding::Signed));
    config
        .columns
        .insert("n".into(), declared(PgType::Int4, LaneEncoding::Signed));
    config
        .columns
        .insert("b".into(), declared(PgType::Bool, LaneEncoding::Bool));
    config
        .columns
        .insert("t".into(), declared(PgType::Text, LaneEncoding::TextCode));
    let mut row = vec![Some("1"), Some("5"), Some("t"), Some("y")];
    let k = ["id", "n", "b", "t"]
        .iter()
        .position(|c| *c == field)
        .unwrap();
    row[k] = cell;
    Table::from_rows("m", 0x2B0F_0001, &["id", "n", "b", "t"], &[row], &config).unwrap()
}

fn recipe_outcome(t: &Table, src: &str) -> Outcome {
    let w = t.witness();
    let class = lance_graph_dt64::view_class(w.binding.class());
    let frame = t.frame();
    let r = Recipe::compile(src, w.view, class)
        .unwrap()
        .resolve_text(&frame, |iri, text| t.code_for(&frame, iri, text))
        .unwrap();
    let witness = Witness {
        binding: w.binding,
        store: w.store,
        view: w.view,
        frames: w.frames,
        source: w.source,
    };
    let rows = r.materialize_rows(&witness).unwrap();
    if rows.len() == 1 {
        Keep
    } else if rows.unknown() == 1 {
        Unknown
    } else {
        Drop
    }
}

fn external() -> ExternalEnv {
    let value = |k: Kind| {
        let mut m: BTreeMap<Field, Kind> = BTreeMap::new();
        m.insert("value".into(), k);
        Kind::object(Collection::from(m))
    };
    let mut props: BTreeMap<Field, Kind> = BTreeMap::new();
    props.insert("n".into(), value(Kind::integer()));
    props.insert("b".into(), value(Kind::boolean()));
    props.insert("t".into(), value(Kind::bytes()));
    let mut top: BTreeMap<Field, Kind> = BTreeMap::new();
    top.insert("properties".into(), Kind::object(Collection::from(props)));
    ExternalEnv::new_with_kind(
        Kind::object(Collection::from(top)),
        Kind::object(Collection::empty()),
    )
}

/// The event for `field` holding `cell`; `None` as no key or as `null`.
fn event(field: &str, cell: Option<&str>, null: bool) -> Value {
    let typed = |f: &str, v: &str| match f {
        "n" => Value::Integer(v.parse().unwrap()),
        "b" => Value::Boolean(v == "t"),
        _ => Value::from(v),
    };
    let mut props = ObjectMap::new();
    for (f, other) in [("n", "5"), ("b", "t"), ("t", "y")] {
        let mut v = ObjectMap::new();
        let cell = if f == field { cell } else { Some(other) };
        match cell {
            Some(c) => {
                v.insert("value".into(), typed(f, c));
            }
            None if null => {
                v.insert("value".into(), Value::Null);
            }
            None => {}
        }
        props.insert(f.into(), Value::Object(v));
    }
    let mut top = ObjectMap::new();
    top.insert("properties".into(), Value::Object(props));
    Value::Object(top)
}

fn upstream_outcome(src: &str, ev: Value) -> Outcome {
    let program = compile_with_external(
        src,
        &vrl::stdlib::all(),
        &external(),
        CompileConfig::default(),
    )
    .unwrap_or_else(|d| panic!("{src}: {:?}", d.iter().map(|x| x.code).collect::<Vec<_>>()))
    .program;
    let mut t = TargetValue {
        value: ev,
        metadata: Value::Object(ObjectMap::new()),
        secrets: Secrets::default(),
    };
    match Runtime::default().resolve(&mut t, &program, &TimeZone::default()) {
        Ok(_) => Keep,
        Err(Terminate::Abort(_)) => Drop,
        Err(_) => Unknown,
    }
}

/// FAILS IF: the two engines decide a present value (including `0`, `false`
/// and `""`) differently; upstream reads an explicit `null` and a missing
/// key differently; the recipe decides an absent value at all; or an absent
/// case differs from [`ABSENT`].
#[test]
fn presence_is_decided_alike_where_present_and_pinned_where_absent() {
    let mut absent = Vec::new();
    for (field, src) in PROGRAMS {
        for (case, cell) in cases(field) {
            let t = table(field, cell);
            let ours = recipe_outcome(&t, src);
            let missing = upstream_outcome(src, event(field, cell, false));
            let null = upstream_outcome(src, event(field, cell, true));
            eprintln!(
                "{src:45} {case:7} recipe {ours:?} upstream missing {missing:?} null {null:?}"
            );
            if cell.is_some() {
                assert_eq!(ours, missing, "{src} on {case}");
            } else {
                assert_eq!(ours, Unknown, "{src}: the recipe decides an absent value");
                absent.push((*src, missing, null));
            }
        }
    }
    let pinned: Vec<_> = ABSENT.to_vec();
    assert_eq!(absent, pinned);
}

/// FAILS IF: the gate admits a program of [`PROGRAMS`] over the field
/// declared optional (each one's answer depends on how absent is read), or
/// refuses it over the field declared present.
#[test]
fn the_gate_refuses_every_program_whose_answer_depends_on_absence() {
    for (field, src) in PROGRAMS {
        let t = table(field, cases(field)[1].1);
        let w = t.witness();
        let class = lance_graph_dt64::view_class(w.binding.class());
        let kind = |f: &str| match f {
            "b" => DeclaredKind::Boolean,
            "t" => DeclaredKind::Text,
            _ => DeclaredKind::Integer,
        };
        let schema = |optional: bool| -> Vec<Declared> {
            ["id", "n", "b", "t"]
                .iter()
                .map(|f| Declared::new(f, kind(f), optional && f == field))
                .collect()
        };
        let refused = Recipe::admit(src, w.view, class, &schema(true)).map(|_| ());
        assert!(
            matches!(
                refused,
                Err(Refusal::AmbiguousPresence { .. } | Refusal::Upstream { .. })
            ),
            "{src}: {refused:?}"
        );
        assert!(
            Recipe::admit(src, w.view, class, &schema(false)).is_ok(),
            "{src}"
        );
    }
}
