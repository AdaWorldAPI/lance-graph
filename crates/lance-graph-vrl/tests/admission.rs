//! S2: the admission gate. A program runs only if upstream VRL compiles it
//! against the declared schema and the recipe lowers it.
//!
//! Run with `--features upstream-vector` or `--features upstream-o2`.
#![cfg(any(feature = "upstream-vector", feature = "upstream-o2"))]

mod common;

use common::programs::PRESENCE;
use common::{View, VIEW_CLASS};
use lance_graph_vrl::{Declared, DeclaredKind, Recipe, Refusal};

fn cpu_view() -> View {
    View::with_labels(&["key".to_owned(), "cpu".to_owned()])
}

fn cpu(optional: bool) -> Vec<Declared> {
    vec![
        Declared::new("key", DeclaredKind::Integer, false),
        Declared::new("cpu", DeclaredKind::Integer, optional),
    ]
}

const OVER_CPU: &str = ".k = .key\nif .properties.cpu.value > 3 { abort }";

/// FAILS IF: an order compare over an optional field is admitted, or the
/// same program over a non-optional integer is refused or lowers to another
/// recipe than `compile` gives.
#[test]
fn an_order_compare_is_admitted_only_over_a_non_optional_field() {
    assert_eq!(
        Recipe::admit(OVER_CPU, &cpu_view(), VIEW_CLASS, &cpu(true)).map(|_| ()),
        Err(Refusal::Upstream { codes: vec![100] })
    );
    let admitted = Recipe::admit(OVER_CPU, &cpu_view(), VIEW_CLASS, &cpu(false)).unwrap();
    let compiled = Recipe::compile(OVER_CPU, &cpu_view(), VIEW_CLASS).unwrap();
    assert_eq!(admitted.outputs(), compiled.outputs());
    assert_eq!(admitted.fields(), compiled.fields());
}

/// FAILS IF: upstream admitting a program bypasses the lowering, or the
/// lowering is consulted before upstream so a program upstream cannot
/// compile reports a lowering refusal instead of its E-code.
#[test]
fn both_sides_must_admit() {
    // Upstream compiles arithmetic over a non-optional integer; the recipe
    // does not compute.
    assert!(matches!(
        Recipe::admit(".k = .key + 1", &cpu_view(), VIEW_CLASS, &cpu(false)),
        Err(Refusal::Computes(_))
    ));
    // No such function: upstream's E105, not the lowering's refusal.
    assert_eq!(
        Recipe::admit(
            ".k = no_such_fn(.key)",
            &cpu_view(),
            VIEW_CLASS,
            &cpu(false)
        )
        .map(|_| ()),
        Err(Refusal::Upstream { codes: vec![105] })
    );
}

/// FAILS IF: with the presence lanes declared optional, the gate admits a
/// `presence.rs` program whose answer depends on how an absent value is
/// read, or refuses one whose answer does not.
///
/// This ties S2 to S1's measurement. Upstream refuses programs 1, 3 and 4
/// (an order compare over an optional field, E100). The gate refuses program
/// 2 (`!=` over an optional field: upstream reads absent as `null`, the
/// recipe as unknown; Q1, refused until a presence-aware lowering). Only
/// program 0, which compares nothing, is admitted, so no S1 divergence class
/// reaches an admitted program.
#[test]
fn the_gate_admits_exactly_the_presence_programs_without_ambiguous_compares() {
    let schema = [
        Declared::new("key", DeclaredKind::Integer, false),
        Declared::new("status", DeclaredKind::Integer, true),
        Declared::new("region", DeclaredKind::Integer, true),
        Declared::new("amount", DeclaredKind::Integer, true),
        Declared::new("code", DeclaredKind::Integer, true),
    ];
    let mut admitted = Vec::new();
    let mut ambiguous = Vec::new();
    for (i, src) in PRESENCE.iter().enumerate() {
        match Recipe::admit(src, &View::new(), VIEW_CLASS, &schema) {
            Ok(_) => admitted.push(i),
            Err(Refusal::Upstream { codes }) => assert_eq!(codes, [100], "p{i}"),
            Err(Refusal::AmbiguousPresence { field }) => ambiguous.push((i, field)),
            Err(other) => panic!("p{i}: {other:?}"),
        }
    }
    assert_eq!(admitted, [0]);
    assert_eq!(ambiguous, [(2, "status".to_owned())]);
}

/// FAILS IF: an equality, a text equality or a truth test over an optional
/// field is admitted, including through an alias; or the same test over a
/// non-optional field, or an equality over another field, is refused.
#[test]
fn ambiguous_presence_is_refused_and_only_it() {
    let view = View::with_labels(&["key".to_owned(), "cpu".to_owned()]);
    let admit = |src: &str, optional: bool| {
        Recipe::admit(src, &view, VIEW_CLASS, &cpu(optional)).map(|_| ())
    };
    let refused = Err(Refusal::AmbiguousPresence {
        field: "cpu".into(),
    });
    for src in [
        ".k = .key\nif .properties.cpu.value == 3 { abort }",
        ".k = .key\nif .properties.cpu.value != 3 { abort }",
        ".c = .properties.cpu.value\nif .c == 3 { abort }",
    ] {
        assert_eq!(admit(src, true), refused, "{src}");
        assert_eq!(admit(src, false), Ok(()), "{src}");
    }
    assert_eq!(admit(".k = .key\nif .key == 3 { abort }", true), Ok(()));
}
