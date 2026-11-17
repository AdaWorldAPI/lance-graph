//! The recipe against the naive VRL interpreter, at every width.

use vrl_recipe_probe::{interpret, Recipe, Refusal, Store};

struct Lcg(u64);
impl Lcg {
    fn below(&mut self, k: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % k as u64) as i64
    }
}

fn store(n: usize, seed: u64) -> Store {
    let mut l = Lcg(seed);
    let keys = (0..n as i64).map(|i| 1000 + i).collect();
    let status = (0..n).map(|_| l.below(4)).collect();
    let region = (0..n).map(|_| l.below(3)).collect();
    let amount = (0..n).map(|_| l.below(500) - 100).collect();
    Store::new(
        keys,
        vec![("status", status), ("region", region), ("amount", amount)],
    )
}

/// Programs inside the subset, each exercising a different part of it.
const PROGRAMS: &[&str] = &[
    // select + rename
    ".name = .key\n.amount = .properties.amount.value",
    // filter by equality and inequality
    ".name = .key\n.amount = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".id = .key\nif .properties.region.value == 0 { abort }\nif .properties.status.value == 3 { abort }",
    // re-binding: .key overwritten, later reads see the new value
    ".orig = .key\n.key = .properties.region.value\n.reg = .key\nif .key != 1 { abort }",
    // delete an output and a source name
    ".a = .key; .b = .properties.amount.value; del(.a)",
    ".a = .properties.status.value; del(.key); if .a == 0 { abort }",
    // filter on an output alias
    ".s = .properties.status.value\n.amt = .properties.amount.value\nif .s != 1 { abort }",
];

/// The same recipe answers as the interpreter does over materialized events,
/// at widths 0, 1, 63, 64, 65, 256, 512 and 1000: rows in row order, the
/// count, and the sum of every output.
#[test]
fn recipe_equals_the_interpreter_at_every_width() {
    let mut kept_any = false;
    for &n in &[0usize, 1, 63, 64, 65, 256, 512, 1000] {
        let s = store(n, 0x7e51_0000 + n as u64);
        for (i, src) in PROGRAMS.iter().enumerate() {
            let recipe = Recipe::compile(src, &s).unwrap_or_else(|e| panic!("p{i}: {e:?}"));
            let want = interpret(src, &s).unwrap();
            let got = recipe.materialize_rows(&s).unwrap();
            assert_eq!(got, want, "n {n} program {i}");
            assert_eq!(
                recipe.count(&s).unwrap(),
                want.len() as u64,
                "n {n} p{i} count"
            );
            for out in recipe.outputs() {
                let total: i64 = want.iter().map(|e| e[out]).sum();
                assert_eq!(recipe.sum(&s, out).unwrap(), total, "n {n} p{i} sum {out}");
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
    let s = store(512, 9);
    for src in PROGRAMS.iter().filter(|p| p.contains("if ")) {
        let unfiltered: String = src
            .split(['\n', ';'])
            .filter(|l| !l.trim_start().starts_with("if "))
            .collect::<Vec<_>>()
            .join("\n");
        let kept = Recipe::compile(src, &s).unwrap().count(&s).unwrap();
        let all = Recipe::compile(&unfiltered, &s).unwrap().count(&s).unwrap();
        assert!(kept < all && all == 512, "{src}: kept {kept} of {all}");
    }
}

/// The view holds exactly the lanes the outputs read.
#[test]
fn the_view_holds_the_output_lanes() {
    let s = store(8, 1);
    let r = Recipe::compile(".n = .key\n.a = .properties.amount.value", &s).unwrap();
    assert_eq!(r.view().count(), 2);
    assert!(r.view().has(0) && r.view().has(3) && !r.view().has(1));
}

/// Each construct that computes a value is refused, by name, and never
/// approximated; each unknown path is refused as such.
#[test]
fn computing_constructs_are_refused_by_name() {
    let s = store(8, 2);
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
        assert_eq!(Recipe::compile(src, &s).unwrap_err(), *want, "{src}");
    }
}

/// A recipe compiled before a write is refused after it, at every terminal.
#[test]
fn a_recipe_is_refused_after_the_store_moves() {
    let mut s = store(64, 3);
    let r = Recipe::compile(".n = .key", &s).unwrap();
    assert_eq!(r.count(&s).unwrap(), 64);
    s.bump();
    let stale = Refusal::Stale {
        compiled: 1,
        store: 2,
    };
    assert_eq!(r.count(&s).unwrap_err(), stale);
    assert_eq!(r.sum(&s, "n").unwrap_err(), stale);
    assert_eq!(r.materialize_rows(&s).unwrap_err(), stale);
    assert_eq!(
        Recipe::compile(".n = .key", &s).unwrap().count(&s).unwrap(),
        64
    );
}
