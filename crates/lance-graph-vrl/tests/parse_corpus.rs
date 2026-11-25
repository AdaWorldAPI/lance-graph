//! The parser's output over a fixed corpus: every program the suites use,
//! plus lexical edge cases. `tests/parse_corpus.out` is the committed
//! answer; a change to the parser that changes any line fails here.
//! Regenerate deliberately with `BLESS=1`.

use lance_graph_vrl::parse;

const CORPUS: &str = include_str!("parse_corpus.vrl");

fn programs() -> Vec<&'static str> {
    CORPUS
        .strip_suffix('\n')
        .unwrap_or(CORPUS)
        .split("\n%%%%\n")
        .collect()
}

fn render() -> String {
    programs()
        .iter()
        .map(|p| format!("{p:?}\n  => {:?}\n", parse(p)))
        .collect()
}

/// FAILS IF: the parser gives any corpus program another statement list or
/// another refusal than the committed one.
#[test]
fn the_corpus_parses_as_committed() {
    let got = render();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parse_corpus.out");
    if std::env::var_os("BLESS").is_some() {
        std::fs::write(path, &got).unwrap();
    }
    let want = std::fs::read_to_string(path).unwrap();
    for (g, w) in got.lines().zip(want.lines()) {
        assert_eq!(g, w);
    }
    assert_eq!(got.lines().count(), want.lines().count());
    assert!(programs().len() > 80, "corpus: {}", programs().len());
}
