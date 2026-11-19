//! The expectations below are a hand-written table from reading spaCy 3.8.18's
//! `matcher.pyx` and `schemas.py`, not output of a spaCy run.

use lance_graph_quack::bind::Pattern;
use lance_graph_spacy::{Attr, Cond, LengthOp, Quant, Refusal, Split, TokenPattern, TokenSpec};
use serde_json::{json, Value};

fn parse(pattern: Value) -> TokenPattern {
    TokenPattern::parse(&pattern).expect("the pattern parses")
}

fn spans(pattern: Value, tokens: &[&str]) -> Vec<(usize, usize)> {
    parse(pattern).spans(tokens)
}

fn quants(op: &str) -> Result<Vec<Quant>, Refusal> {
    TokenPattern::parse(&json!([{ "OP": op }])).map(|p| p.specs[0].quants.clone())
}

/// FAILS IF: an `OP` expands to anything other than what `_get_operators`
/// returns (matcher.pyx:1171-1172, 1176-1177, 1181-1200), or a malformed `OP`
/// is accepted, or the schema-rejected `"1"` is.
#[test]
fn the_operators_expand_as_spacy_does() {
    use Quant::{One, Zero, ZeroOne, ZeroPlus};
    assert_eq!(quants("+"), Ok(vec![One, ZeroPlus]));
    assert_eq!(quants("*"), Ok(vec![ZeroPlus]));
    assert_eq!(quants("?"), Ok(vec![ZeroOne]));
    assert_eq!(quants("!"), Ok(vec![Zero]));
    assert_eq!(quants("{2}"), Ok(vec![One, One]));
    assert_eq!(quants("{1,3}"), Ok(vec![One, ZeroOne, ZeroOne]));
    assert_eq!(quants("{,2}"), Ok(vec![ZeroOne, ZeroOne]));
    assert_eq!(quants("{2,}"), Ok(vec![One, One, ZeroPlus]));
    // `{0}`: head and tail are both empty (matcher.pyx:1198-1200), so the spec
    // yields no node at all.
    assert_eq!(quants("{0}"), Ok(vec![]));

    let no_op = parse(json!([{ "ORTH": "a" }]));
    assert_eq!(no_op.specs[0].quants, vec![One]);

    for bad in ["{3,1}", "{,}", "1", "x"] {
        assert_eq!(
            quants(bad),
            Err(Refusal::BadOp {
                index: 0,
                op: bad.to_string()
            }),
            "{bad}"
        );
    }
}

/// FAILS IF: `spans` returns anything but every matching `(start, end)`, in
/// order and without duplicates, for the quantifiers.
#[test]
fn spans_are_every_match() {
    let tokens = ["a", "b", "b", "c"];

    assert_eq!(
        spans(json!([{"ORTH": "a"}, {"ORTH": "b", "OP": "+"}]), &tokens),
        vec![(0, 2), (0, 3)]
    );
    assert_eq!(
        spans(json!([{"ORTH": "b", "OP": "*"}, {"ORTH": "c"}]), &tokens),
        vec![(1, 4), (2, 4), (3, 4)]
    );
    assert_eq!(
        spans(json!([{}]), &tokens),
        vec![(0, 1), (1, 2), (2, 3), (3, 4)]
    );
    assert_eq!(
        spans(json!([{"OP": "+"}]), &tokens),
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4)
        ]
    );

    // `?` as the final node: from start 0, `a` advances (get_action ONE,
    // non-final, matcher.pyx:697-699), then `b?` on token 1 is ZERO_ONE, match,
    // final: MATCH_DOUBLE (matcher.pyx:718-723), which emits the match without
    // token 1 (length 1, only because length > 0, transition_states
    // matcher.pyx:517-526) and with it (length 2, matcher.pyx:532-538). The state
    // then dies, so the second `b` is not reached from start 0. No other start
    // begins with `a`.
    assert_eq!(
        spans(json!([{"ORTH": "a"}, {"ORTH": "b", "OP": "?"}]), &tokens),
        vec![(0, 1), (0, 2)]
    );

    // `!` is ONE with `is_match` negated (matcher.pyx:690-692): it consumes
    // exactly one token that does not satisfy the spec. From start 0, `a`
    // advances; token 1 `b` is not `a`, so the negated match is true and the
    // final node returns MATCH with length 2 (matcher.pyx:693-696). No other
    // start begins with `a`.
    assert_eq!(
        spans(json!([{"ORTH": "a"}, {"ORTH": "a", "OP": "!"}]), &tokens),
        vec![(0, 2)]
    );
    // Alone: the tokens that are not `b`, one each.
    assert_eq!(
        spans(json!([{"ORTH": "b", "OP": "!"}]), &tokens),
        vec![(0, 1), (3, 4)]
    );
    // A `!` with no condition: get_is_match is true for every token with no
    // attribute to compare (matcher.pyx:747-755), the negation is false, so
    // nothing matches.
    assert_eq!(spans(json!([{"OP": "!"}]), &tokens), vec![]);

    // The silent twins: no tokens, no span; a spec that never holds, no span.
    assert_eq!(spans(json!([{"OP": "+"}]), &[]), vec![]);
    assert_eq!(spans(json!([{"ORTH": "z"}]), &tokens), vec![]);
}

/// FAILS IF: a condition reads the wrong attribute or compares wrongly, a
/// negated set is not negated, or LENGTH counts bytes instead of characters.
#[test]
fn conditions() {
    let tokens = ["a", "bb", "ccc", "\u{e9}"];

    assert_eq!(
        spans(
            json!([{"LOWER": "admin"}]),
            &["ADMIN", "Admin", "admin", "root"]
        ),
        vec![(0, 1), (1, 2), (2, 3)]
    );
    // LOWER is compared as written: an upper-case operand never holds.
    assert_eq!(spans(json!([{"LOWER": "Admin"}]), &["Admin"]), vec![]);

    assert_eq!(
        spans(json!([{"ORTH": {"IN": ["a", "b"]}}]), &["a", "c", "b"]),
        vec![(0, 1), (2, 3)]
    );
    assert_eq!(
        spans(json!([{"ORTH": {"NOT_IN": ["a", "b"]}}]), &["a", "c", "b"]),
        vec![(1, 2)]
    );

    assert_eq!(spans(json!([{"LENGTH": 2}]), &tokens), vec![(1, 2)]);
    assert_eq!(spans(json!([{"LENGTH": {"==": 2}}]), &tokens), vec![(1, 2)]);
    assert_eq!(
        spans(json!([{"LENGTH": {"!=": 2}}]), &tokens),
        vec![(0, 1), (2, 3), (3, 4)]
    );
    assert_eq!(
        spans(json!([{"LENGTH": {">=": 2}}]), &tokens),
        vec![(1, 2), (2, 3)]
    );
    assert_eq!(
        spans(json!([{"LENGTH": {"<=": 2}}]), &tokens),
        vec![(0, 1), (1, 2), (3, 4)]
    );
    assert_eq!(spans(json!([{"LENGTH": {">": 2}}]), &tokens), vec![(2, 3)]);
    assert_eq!(
        spans(json!([{"LENGTH": {"<": 2}}]), &tokens),
        vec![(0, 1), (3, 4)]
    );
    assert_eq!(
        spans(json!([{"LENGTH": {"IN": [1, 3]}}]), &tokens),
        vec![(0, 1), (2, 3), (3, 4)]
    );
    assert_eq!(
        spans(json!([{"LENGTH": {"NOT_IN": [1, 3]}}]), &tokens),
        vec![(1, 2)]
    );
    // "é" is one character, two bytes.
    assert_eq!(spans(json!([{"LENGTH": 1}]), &tokens), vec![(0, 1), (3, 4)]);

    // Conditions in one spec all hold; keys are read in any case.
    assert_eq!(
        spans(json!([{"ORTH": {"IN": ["a", "bb"]}, "LENGTH": 2}]), &tokens),
        vec![(1, 2)]
    );
    let upper = parse(json!([{"ORTH": "a"}]));
    assert_eq!(parse(json!([{"orth": "a"}])), upper);
    assert_eq!(parse(json!([{"Text": "a"}])), upper);
    assert_eq!(
        upper,
        TokenPattern {
            specs: vec![TokenSpec {
                conds: vec![Cond::Text {
                    attr: Attr::Orth,
                    equals: "a".to_string()
                }],
                quants: vec![Quant::One],
            }]
        }
    );
    assert_eq!(
        parse(json!([{"LENGTH": {">=": 2}}])).specs[0].conds,
        vec![Cond::Length {
            op: LengthOp::Ge,
            n: 2
        }]
    );
}

/// FAILS IF: a key spaCy has is accepted or reported under another name, an
/// unknown key is called unsupported, a malformed pattern is accepted, or a
/// supported pattern is refused.
#[test]
fn what_spacy_has_and_this_does_not_is_refused_by_name() {
    let unsupported = |spec: Value, key: &str| {
        assert_eq!(
            TokenPattern::parse(&json!([{}, spec])),
            Err(Refusal::Unsupported {
                index: 1,
                key: key.to_string()
            }),
            "{key}"
        );
    };
    unsupported(json!({"POS": "NOUN"}), "POS");
    unsupported(json!({"LEMMA": "run"}), "LEMMA");
    unsupported(json!({"DEP": "nsubj"}), "DEP");
    unsupported(json!({"ORTH": {"REGEX": "a+"}}), "REGEX");
    unsupported(json!({"ORTH": {"FUZZY": "abc"}}), "FUZZY");
    unsupported(json!({"IS_ALPHA": true}), "IS_ALPHA");
    unsupported(json!({"_": {"foo": 1}}), "_");

    assert_eq!(
        TokenPattern::parse(&json!([{"FOO": "x"}])),
        Err(Refusal::UnknownKey {
            index: 0,
            key: "FOO".to_string()
        })
    );
    assert_eq!(
        TokenPattern::parse(&json!({"ORTH": "a"})),
        Err(Refusal::NotAList)
    );
    assert_eq!(
        TokenPattern::parse(&json!([{"ORTH": "a"}, "b"])),
        Err(Refusal::NotAnObject { index: 1 })
    );
    assert_eq!(TokenPattern::parse(&json!([])), Err(Refusal::Empty));
    assert_eq!(
        TokenPattern::parse(&json!([{"LENGTH": "2"}])),
        Err(Refusal::BadValue {
            index: 0,
            key: "LENGTH".to_string()
        })
    );

    // Silent twin: a pattern of supported keys parses.
    assert!(TokenPattern::parse(&json!([
        {"LOWER": "a", "LENGTH": {">=": 1}},
        {"ORTH": {"NOT_IN": ["x"]}, "OP": "{1,2}"},
        {"OP": "!"}
    ]))
    .is_ok());
}

/// FAILS IF: the pattern `[{"ORTH": "admin"}, {"OP": "+"}]` under a `.` split
/// and `Pattern::like("admin.%")` disagree on any listed value, or the table
/// stops holding both a true and a false.
#[test]
fn a_prefix_pattern_equals_like() {
    let split = Split { separator: '.' };
    let pattern = parse(json!([{"ORTH": "admin"}, {"OP": "+"}]));
    let like = Pattern::like("admin.%").expect("a finished pattern");

    let mut seen_true = false;
    let mut seen_false = false;
    for value in [
        "admin",
        "admin.",
        "admin.user",
        "admin.user.x",
        "adminx",
        "admin..x",
        "x.admin",
        "",
        "Admin.user",
    ] {
        let from_pattern = pattern.matches_value(&split, value);
        assert_eq!(from_pattern, like.matches(value), "{value:?}");
        seen_true |= from_pattern;
        seen_false |= !from_pattern;
    }
    assert!(seen_true && seen_false);
}

/// FAILS IF: an empty part between or after separators is dropped, or the
/// empty value has a token.
#[test]
fn the_split_keeps_empty_parts() {
    let split = Split { separator: '.' };
    assert_eq!(split.tokens("a..b"), vec!["a", "", "b"]);
    assert_eq!(split.tokens("a."), vec!["a", ""]);
    assert_eq!(split.tokens(".a"), vec!["", "a"]);
    assert_eq!(split.tokens("."), vec!["", ""]);
    assert_eq!(split.tokens("a"), vec!["a"]);
    assert!(split.tokens("").is_empty());
}
