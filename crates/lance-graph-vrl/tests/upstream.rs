//! S1: the upstream VRL differential. Every program of `recipe.rs` and
//! `presence.rs` runs twice: as the recipe, in-fold over lanes, and as
//! upstream VRL, one event per row, compiled against a typed `ExternalEnv`.
//! Each row's outcome is compared: KEEP ↔ `Ok`, DROP ↔ `abort`, UNKNOWN ↔ a
//! runtime error. A kept row's outputs are compared value by value.
//!
//! Where the two engines decide one row differently, the difference is a
//! pinned divergence: it must appear, with its count, in [`DIVERGENCES`].
//! An unlisted difference, or a listed one that no longer happens, fails.
//!
//! Run with `--features upstream-vector` (Vector's vrl, 0.39.0 at 7738f74) or
//! `--features upstream-o2` (OpenObserve's, 0.31.0).
#![cfg(any(feature = "upstream-vector", feature = "upstream-o2"))]

mod common;

use std::collections::BTreeMap;

use common::{all_fields, bind, witness, Source, VIEW_CLASS};
use lance_graph_vrl::Recipe;

#[cfg(all(feature = "upstream-o2", not(feature = "upstream-vector")))]
use vrl_o2 as vrl;
#[cfg(feature = "upstream-vector")]
use vrl_vector as vrl;

use vrl::compiler::runtime::{Runtime, Terminate};
use vrl::compiler::state::ExternalEnv;
use vrl::compiler::{compile_with_external, CompileConfig, Program, TargetValue, TimeZone};
use vrl::value::kind::{Collection, Field};
use vrl::value::{Kind, ObjectMap, Secrets, Value};

/// The pin under test, as it appears in [`DIVERGENCES`].
#[cfg(feature = "upstream-vector")]
const PIN: &str = "vector";
#[cfg(all(feature = "upstream-o2", not(feature = "upstream-vector")))]
const PIN: &str = "o2";

/// `recipe.rs`'s programs: every lane present.
const RECIPE: &[&str] = &[
    ".name = .key\n.amount = .properties.amount.value",
    ".name = .key\n.amount = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".id = .key\nif .properties.region.value == 0 { abort }\nif .properties.status.value == 3 { abort }",
    ".orig = .key\n.key = .properties.region.value\n.reg = .key\nif .key != 1 { abort }",
    ".a = .key; .b = .properties.amount.value; del(.a)",
    ".a = .properties.status.value; del(.key); if .a == 0 { abort }",
    ".s = .properties.status.value\n.amt = .properties.amount.value\nif .s != 1 { abort }",
    ".id = .key\nif .key < 1010 { abort }\nif .key >= 1050 { abort }",
    ".a = .properties.amount.value\nif .a <= 0 { abort }\nif .properties.status.value > 1 { abort }",
    ".r = .properties.region.value\nif .r < 1 { abort }\nif .r >= 2 { abort }",
];

/// `presence.rs`'s programs: lanes with absent values.
const PRESENCE: &[&str] = &[
    ".n = .key\n.a = .properties.amount.value",
    ".a = .properties.amount.value\nif .a < 0 { abort }",
    ".a = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".a = .properties.amount.value\nif .properties.region.value == 0 { abort }\nif .a >= 200 { abort }",
    ".r = .properties.region.value\n.a = .properties.amount.value\nif .properties.status.value > 2 { abort }\nif .a <= 10 { abort }\nif .r != 1 { abort }",
];

/// One row's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Keep,
    Drop,
    Unknown,
}

/// The pinned divergences: (pin, suite, program index, polyfill, upstream,
/// rows). The presence suite runs at 300 rows.
///
/// - `Unknown → Keep` / `Unknown → Drop`: an absent value under `==` or `!=`.
///   The recipe leaves the row undecided; upstream compares null like any
///   value (`null != 2` is true, `null == 0` false). Q1 of the census plan
///   decides whether a `VrlCompat` mode adopts this.
/// - `Drop → Unknown`: upstream stops at the first statement that errors,
///   so an absent value under an order compare hides a later `abort` the
///   recipe decides. The recipe's three-valued rule lets a decided abort win.
const DIVERGENCES: &[(&str, &str, usize, Outcome, Outcome, usize)] = &[
    // `if .properties.status.value != 2 { abort }`: absent status aborts.
    ("vector", "presence", 2, Outcome::Unknown, Outcome::Drop, 60),
    // `if .properties.region.value == 0 { abort }`: absent region passes.
    ("vector", "presence", 3, Outcome::Unknown, Outcome::Keep, 40),
    // `.status > 2` or `.a <= 10` errors before `.r != 1` aborts.
    ("vector", "presence", 4, Outcome::Drop, Outcome::Unknown, 79),
    // `if .r != 1 { abort }`: absent region aborts.
    ("vector", "presence", 4, Outcome::Unknown, Outcome::Drop, 31),
    // OpenObserve's 0.31.0 decides every row as 0.39.0 does: no drift.
    ("o2", "presence", 2, Outcome::Unknown, Outcome::Drop, 60),
    ("o2", "presence", 3, Outcome::Unknown, Outcome::Keep, 40),
    ("o2", "presence", 4, Outcome::Drop, Outcome::Unknown, 79),
    ("o2", "presence", 4, Outcome::Unknown, Outcome::Drop, 31),
];

/// The typed target: every lane a non-optional integer, as the recipe
/// binds it. Without it, every order compare is refused with E100.
fn external() -> ExternalEnv {
    let value = |k: Kind| {
        let mut m: BTreeMap<Field, Kind> = BTreeMap::new();
        m.insert("value".into(), k);
        Kind::object(Collection::from(m))
    };
    let mut props: BTreeMap<Field, Kind> = BTreeMap::new();
    for f in ["status", "region", "amount"] {
        props.insert(f.into(), value(Kind::integer()));
    }
    let mut top: BTreeMap<Field, Kind> = BTreeMap::new();
    top.insert("key".into(), Kind::integer());
    top.insert("properties".into(), Kind::object(Collection::from(props)));
    ExternalEnv::new_with_kind(
        Kind::object(Collection::from(top)),
        Kind::object(Collection::empty()),
    )
}

fn compile(src: &str) -> Program {
    let fns = vrl::stdlib::all();
    match compile_with_external(src, &fns, &external(), CompileConfig::default()) {
        Ok(r) => r.program,
        Err(d) => panic!(
            "{PIN}: upstream refuses {src:?}: {:?}",
            d.iter().map(|x| x.code).collect::<Vec<_>>()
        ),
    }
}

/// Row `r` as the event Vector would see: an absent value (presence 0 and
/// value 0, EXO-294) is a missing `value` key. Presence 0 with a nonzero
/// value is a value.
fn event(lanes: &[&[i32]], present: &[&[u64]], r: usize) -> Value {
    let has = |l: usize| present[l][r / 64] >> (r % 64) & 1 == 1 || lanes[l][r] != 0;
    let mut props = ObjectMap::new();
    for (l, name) in [(1, "status"), (2, "region"), (3, "amount")] {
        let mut v = ObjectMap::new();
        if has(l) {
            v.insert("value".into(), Value::Integer(lanes[l][r].into()));
        }
        props.insert(name.into(), Value::Object(v));
    }
    let mut top = ObjectMap::new();
    if has(0) {
        top.insert("key".into(), Value::Integer(lanes[0][r].into()));
    }
    top.insert("properties".into(), Value::Object(props));
    Value::Object(top)
}

/// Upstream's outcome for one event, and the kept event's outputs.
fn upstream(program: &Program, ev: Value, outputs: &[&str]) -> (Outcome, Vec<Option<i64>>) {
    let mut t = TargetValue {
        value: ev,
        metadata: Value::Object(ObjectMap::new()),
        secrets: Secrets::default(),
    };
    match Runtime::default().resolve(&mut t, program, &TimeZone::default()) {
        Ok(_) => {
            let out = outputs
                .iter()
                .copied()
                .map(|o| match &t.value {
                    Value::Object(m) => match m.get(o) {
                        Some(Value::Integer(i)) => Some(*i),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            (Outcome::Keep, out)
        }
        Err(Terminate::Abort(_)) => (Outcome::Drop, vec![]),
        Err(_) => (Outcome::Unknown, vec![]),
    }
}

/// The recipe's outcome for row `r` alone, and the kept row's outputs.
fn polyfill(
    recipe: &Recipe,
    lanes: &[&[i32]],
    present: &[&[u64]],
    r: usize,
) -> (Outcome, Vec<Option<i64>>) {
    let mut one = Source::from_lanes(std::array::from_fn(|l| vec![lanes[l][r]]));
    for (l, plane) in present.iter().enumerate() {
        if plane[r / 64] >> (r % 64) & 1 == 0 {
            one.presence[l].as_mut().unwrap()[0] &= !1;
        }
    }
    let world = bind(&one, &all_fields());
    let w = witness(&world, &one);
    let rows = recipe.materialize_rows(&w).unwrap();
    if rows.len() == 1 {
        let out = (0..rows.names().len())
            .map(|c| {
                rows.present(0, c)
                    .then(|| i64::from(rows.get(0, c).as_i32().unwrap()))
            })
            .collect();
        (Outcome::Keep, out)
    } else if rows.unknown() == 1 {
        (Outcome::Unknown, vec![])
    } else {
        (Outcome::Drop, vec![])
    }
}

type Tally = BTreeMap<(usize, Outcome, Outcome), usize>;

/// Run one suite over `source`; panic on any kept row whose outputs differ.
fn run(suite: &str, programs: &[&str], source: &Source) -> (Tally, [usize; 3]) {
    let lanes = source.i32_lanes();
    let present = source.i32_presence();
    let n = lanes[0].len();
    let mut tally = Tally::new();
    let mut seen = [0; 3];
    for (i, src) in programs.iter().enumerate() {
        let recipe = Recipe::compile(src, &common::View::new(), VIEW_CLASS).unwrap();
        let outputs = recipe.outputs();
        let program = compile(src);
        for r in 0..n {
            let (p, p_out) = polyfill(&recipe, &lanes, &present, r);
            let (u, u_out) = upstream(&program, event(&lanes, &present, r), &outputs);
            seen[u as usize] += 1;
            if p != u {
                let absent =
                    (0..4).any(|l| present[l][r / 64] >> (r % 64) & 1 == 0 && lanes[l][r] == 0);
                assert!(
                    absent,
                    "{PIN} {suite} p{i} row {r}: {p:?} vs {u:?} with every value present"
                );
                *tally.entry((i, p, u)).or_default() += 1;
            } else if p == Outcome::Keep {
                assert_eq!(
                    p_out, u_out,
                    "{PIN} {suite} p{i} row {r}: outputs {outputs:?}"
                );
            }
        }
    }
    (tally, seen)
}

fn pinned(suite: &str) -> Tally {
    DIVERGENCES
        .iter()
        .filter(|d| d.0 == PIN && d.1 == suite)
        .map(|d| ((d.2, d.3, d.4), d.5))
        .collect()
}

/// FAILS IF: upstream, compiled against the typed target, decides any row of
/// a fully present source differently from the recipe, or keeps it with
/// other outputs; or the suite never both keeps and drops.
#[test]
fn present_lanes_agree_with_upstream_row_by_row() {
    for &n in &[1usize, 64, 65, 256] {
        let source = Source::random(n, 0x7e51_0000 + n as u64);
        let (tally, seen) = run("recipe", RECIPE, &source);
        assert_eq!(tally, pinned("recipe"), "{PIN} recipe at {n}");
        if n == 256 {
            assert!(
                seen[0] > 0 && seen[1] > 0,
                "{PIN}: keeps {} drops {}",
                seen[0],
                seen[1]
            );
        }
    }
}

/// FAILS IF: a row with absent values is decided differently from the
/// recipe in a way not listed in [`DIVERGENCES`], a listed divergence no
/// longer happens or changes its count, or upstream never errors on an
/// absent value (the UNKNOWN ↔ error correspondence is not exercised).
#[test]
fn absent_values_agree_with_upstream_or_are_pinned() {
    let n = 300;
    let mut source = Source::random(n, 7 + n as u64);
    for lane in 1..=3 {
        let rows: Vec<usize> = (0..n).filter(|r| (r * 7 + lane * 3) % 5 == 0).collect();
        source.set_absent(lane, rows);
        for r in (0..n).filter(|r| (r * 11 + lane) % 13 == 0) {
            if source.lanes[lane][r] != 0 {
                let plane = source.presence[lane].as_mut().unwrap();
                plane[r / 64] &= !(1u64 << (r % 64));
            }
        }
    }
    let (tally, seen) = run("presence", PRESENCE, &source);
    assert!(
        seen[2] > 0,
        "{PIN}: upstream never errored on an absent value"
    );
    assert_eq!(tally, pinned("presence"), "{PIN} presence divergences");
}

/// FAILS IF: a program here is no longer one of its suite's programs, so the
/// differential silently stops covering what the suite tests.
#[test]
fn the_programs_are_the_suites_programs() {
    for (file, programs) in [
        (include_str!("recipe.rs"), RECIPE),
        (include_str!("presence.rs"), PRESENCE),
    ] {
        for p in programs {
            let quoted = format!("{p:?}");
            assert!(file.contains(&quoted), "not in its suite: {quoted}");
        }
    }
}
