//! The VRL recipe on the existing witnessed path, against the naive VRL
//! interpreter, at every width.

mod common;

use common::{all_fields, bind, witness, Source, CODE, LABELS, VIEW_CLASS};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::LaneRefusal;
use vrl_recipe_probe::{interpret, Event, Recipe, Refusal, Rows};

/// Positional rows as events, for comparison with the interpreter only.
fn events(rows: &Rows<'_>) -> Vec<Event> {
    (0..rows.len())
        .map(|r| {
            rows.names()
                .iter()
                .enumerate()
                .map(|(c, n)| (n.clone(), i64::from(rows.get(r, c))))
                .collect()
        })
        .collect()
}

fn compile(src: &str) -> Recipe {
    Recipe::compile(src, &common::View::new(), VIEW_CLASS)
        .unwrap_or_else(|e| panic!("{src}: {e:?}"))
}

/// Programs inside the subset, each exercising a different part of it.
const PROGRAMS: &[&str] = &[
    ".name = .key\n.amount = .properties.amount.value",
    ".name = .key\n.amount = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".id = .key\nif .properties.region.value == 0 { abort }\nif .properties.status.value == 3 { abort }",
    ".orig = .key\n.key = .properties.region.value\n.reg = .key\nif .key != 1 { abort }",
    ".a = .key; .b = .properties.amount.value; del(.a)",
    ".a = .properties.status.value; del(.key); if .a == 0 { abort }",
    ".s = .properties.status.value\n.amt = .properties.amount.value\nif .s != 1 { abort }",
];

/// The same recipe answers as the interpreter does over materialized events,
/// at widths 0, 1, 63, 64, 65, 256, 512 and 1000: rows in row order, names,
/// the count, and the sum of every output.
#[test]
fn recipe_equals_the_interpreter_at_every_width() {
    let mut kept_any = false;
    for &n in &[0usize, 1, 63, 64, 65, 256, 512, 1000] {
        let source = Source::random(n, 0x7e51_0000 + n as u64);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        for (i, src) in PROGRAMS.iter().enumerate() {
            let recipe = compile(src);
            let want = interpret(src, &LABELS[..4], &source.i32_lanes()).unwrap();
            let rows = recipe.materialize_rows(&w).unwrap();
            assert_eq!(
                rows.names(),
                recipe.outputs().as_slice(),
                "n {n} p{i} names"
            );
            assert!(
                rows.positions().windows(2).all(|p| p[0] < p[1]),
                "row order"
            );
            assert_eq!(events(&rows), want, "n {n} program {i}");
            assert_eq!(
                recipe.count(&w).unwrap(),
                want.len() as u64,
                "n {n} p{i} count"
            );
            for out in recipe.outputs() {
                let total: i64 = want.iter().map(|e| e[out]).sum();
                assert_eq!(recipe.sum(&w, out).unwrap(), total, "n {n} p{i} sum {out}");
            }
            kept_any |= !want.is_empty() && want.len() < n;
        }
    }
    assert!(
        kept_any,
        "no program both kept and dropped rows: the filters are not exercised"
    );
}

/// The filters are load-bearing: each filtered program keeps strictly fewer
/// rows than the same program without its `if` lines.
#[test]
fn every_filter_drops_rows() {
    let source = Source::random(512, 9);
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    for src in PROGRAMS.iter().filter(|p| p.contains("if ")) {
        let unfiltered: String = src
            .split(['\n', ';'])
            .filter(|l| !l.trim_start().starts_with("if "))
            .collect::<Vec<_>>()
            .join("\n");
        let kept = compile(src).count(&w).unwrap();
        let all = compile(&unfiltered).count(&w).unwrap();
        assert!(kept < all && all == 512, "{src}: kept {kept} of {all}");
    }
}

/// The recipe asks for exactly the fields it reads, outputs and filters.
#[test]
fn the_recipe_binds_exactly_the_fields_it_reads() {
    let r = compile(".n = .key\nif .properties.region.value == 1 { abort }");
    assert_eq!(r.fields().count(), 2);
    assert!(r.fields().has(0) && r.fields().has(2) && !r.fields().has(3));
}

/// Each construct that computes a value is refused, by name, and never
/// approximated; each unknown path is refused as such.
#[test]
fn computing_constructs_are_refused_by_name() {
    let cases: &[(&str, Refusal)] = &[
        (
            ".a = parse_json(.key)",
            Refusal::Computes("function parse_json".into()),
        ),
        (".t = now()", Refusal::Computes("function now".into())),
        (".a = .key + 1", Refusal::Computes("arithmetic".into())),
        (".a = \"x\"", Refusal::Computes("string literal".into())),
        (".a = 5", Refusal::Computes("integer literal".into())),
        (
            "if .key > 3 { abort }",
            Refusal::Computes("comparison >".into()),
        ),
        (
            "if .key == 3 { .a = .key }",
            Refusal::Computes("if body other than abort".into()),
        ),
        (
            "to_string(.key)",
            Refusal::Computes("function to_string".into()),
        ),
        (
            ".properties.x.value = .key",
            Refusal::Computes("assignment into properties".into()),
        ),
        (".a = .nope", Refusal::UnknownPath(".nope".into())),
        (
            ".a = .properties.nope.value",
            Refusal::UnknownPath(".properties.nope.value".into()),
        ),
        ("del(.nope)", Refusal::UnknownPath(".nope".into())),
    ];
    for (src, want) in cases {
        let got = Recipe::compile(src, &common::View::new(), VIEW_CLASS).unwrap_err();
        assert_eq!(got, *want, "{src}");
    }
}

/// After the source writes (its frame advances), every terminal of a recipe
/// is refused by DT64's frame check before any lane is borrowed. A recipe is
/// read only against the frame it was bound under.
#[test]
fn a_moved_frame_is_refused_at_every_terminal() {
    let mut source = Source::random(64, 3);
    let world = bind(&source, &all_fields());
    let r = compile(".n = .key");
    assert_eq!(r.count(&witness(&world, &source)).unwrap(), 64);
    source.advance();
    let w = witness(&world, &source);
    let stale = |e: Refusal| e == Refusal::Binding(LaneRefusal::Frame(FrameRefusal::SourceChanged));
    assert!(stale(r.count(&w).unwrap_err()), "count");
    assert!(stale(r.sum(&w, "n").unwrap_err()), "sum");
    assert!(stale(r.materialize_rows(&w).unwrap_err()), "materialize");
}

/// A recipe reading a field the binding does not hold is refused, and a
/// field that is not an `i32` lane is refused, both before folding.
#[test]
fn unbound_and_non_i32_fields_are_refused() {
    let source = Source::random(64, 4);
    let narrow = bind(&source, &WideFieldMask::from_positions(&[0]));
    let r = compile(".n = .key\n.a = .properties.amount.value");
    assert_eq!(
        r.count(&witness(&narrow, &source)).unwrap_err(),
        Refusal::NotBound { field: 3 }
    );
    let world = bind(&source, &all_fields());
    let c = compile(".c = .properties.code.value");
    assert_eq!(
        c.count(&witness(&world, &source)).unwrap_err(),
        Refusal::NotI32 { field: CODE }
    );
}
