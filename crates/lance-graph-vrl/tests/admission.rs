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

/// FAILS IF: with the presence lanes declared optional, the gate admits
/// another set of `presence.rs` programs than those without an order
/// compare over an optional field.
///
/// This ties S2 to S1's measurement. The admitted programs are 0 and 2. The
/// S1 divergence class Drop → Unknown (upstream failing at an order compare)
/// occurs only in refused programs. The class left on admitted programs is
/// `==`/`!=` over an absent value (program 2), which is Q1.
#[test]
fn the_gate_admits_exactly_the_presence_programs_without_fallible_compares() {
    let schema = [
        Declared::new("key", DeclaredKind::Integer, false),
        Declared::new("status", DeclaredKind::Integer, true),
        Declared::new("region", DeclaredKind::Integer, true),
        Declared::new("amount", DeclaredKind::Integer, true),
        Declared::new("code", DeclaredKind::Integer, true),
    ];
    let mut admitted = Vec::new();
    for (i, src) in PRESENCE.iter().enumerate() {
        match Recipe::admit(src, &View::new(), VIEW_CLASS, &schema) {
            Ok(_) => admitted.push(i),
            Err(Refusal::Upstream { codes }) => assert_eq!(codes, [100], "p{i}"),
            Err(other) => panic!("p{i}: {other:?}"),
        }
    }
    assert_eq!(admitted, [0, 2]);
}
