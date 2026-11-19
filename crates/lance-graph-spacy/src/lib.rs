//! A reading of spaCy's `Matcher` token-pattern FORMAT, as a front end.
//!
//! It parses a pattern (JSON, as spaCy writes it: a list of token-spec
//! objects) and answers whether the tokens of a value match. It holds no
//! document, no tokenizer rules, no vocabulary and no statistical model. A
//! token here is one part of a value under a declared [`Split`].
//!
//! It follows spaCy 3.8.18's `spacy/matcher/matcher.pyx` (and the token-pattern
//! schema in `spacy/schemas.py`) by reading. It was NOT run against spaCy: there
//! is no differential oracle yet, so every equivalence stated below is a claim
//! from source, cited by file:line, not a measurement.
//!
//! Supported: the keys `ORTH`, `TEXT`, `LOWER`, `LENGTH` and `OP`; for the text
//! keys a string, `IN` or `NOT_IN`; for `LENGTH` an integer, `IN`, `NOT_IN` or
//! `==` `!=` `>=` `<=` `>` `<`; every `OP` spaCy's `_get_operators` accepts
//! except the undocumented `"1"`.
//!
//! Refused, by name ([`Refusal`]): every other token-spec key spaCy has (the
//! annotation attributes, the `IS_*` and `LIKE_*` flags, `_` extensions) and
//! the predicates `REGEX`, `FUZZY`, `FUZZY1`..`FUZZY9`, `IS_SUBSET`,
//! `IS_SUPERSET`, `INTERSECTS`; unknown keys; malformed values and operators.
//! `greedy` filtering and match ids are not read: [`TokenPattern::spans`] is
//! spaCy's result without `greedy` (matcher.pyx:282-284), minus the match id.

use std::borrow::Cow;

use serde_json::{Map, Value};

/// Upper bound on a `{n,m}` repeat count, so a hostile pattern cannot ask for
/// an unbounded node list. spaCy has no such bound.
const MAX_REPEAT: usize = 4096;

/// Token-spec keys spaCy's schema has (schemas.py:264-300) that this crate
/// refuses. Taken from the schema's `TokenPattern` fields (map section A.2).
const UNSUPPORTED_KEYS: &[&str] = &[
    "POS",
    "TAG",
    "MORPH",
    "DEP",
    "LEMMA",
    "SHAPE",
    "ENT_TYPE",
    "ENT_IOB",
    "ENT_ID",
    "ENT_KB_ID",
    "NORM",
    "SPACY",
    "IS_ALPHA",
    "IS_ASCII",
    "IS_DIGIT",
    "IS_LOWER",
    "IS_UPPER",
    "IS_TITLE",
    "IS_PUNCT",
    "IS_SPACE",
    "IS_BRACKET",
    "IS_QUOTE",
    "IS_LEFT_PUNCT",
    "IS_RIGHT_PUNCT",
    "IS_CURRENCY",
    "IS_STOP",
    "IS_SENT_START",
    "SENT_START",
    "LIKE_NUM",
    "LIKE_URL",
    "LIKE_EMAIL",
    "_",
];

/// Predicates spaCy has (matcher.pyx:1065-1088) that this crate refuses.
const UNSUPPORTED_PREDICATES: &[&str] = &[
    "REGEX",
    "FUZZY",
    "FUZZY1",
    "FUZZY2",
    "FUZZY3",
    "FUZZY4",
    "FUZZY5",
    "FUZZY6",
    "FUZZY7",
    "FUZZY8",
    "FUZZY9",
    "IS_SUBSET",
    "IS_SUPERSET",
    "INTERSECTS",
];

/// How a value is cut into tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub separator: char,
}

impl Split {
    /// The parts of `value` between separators, empty parts kept (`"a..b"` is
    /// `["a", "", "b"]`, `"a."` is `["a", ""]`). The empty value has no token.
    pub fn tokens<'v>(&self, value: &'v str) -> Vec<&'v str> {
        if value.is_empty() {
            Vec::new()
        } else {
            value.split(self.separator).collect()
        }
    }
}

/// The token attribute a condition reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attr {
    Orth,
    Lower,
    Length,
}

/// How `LENGTH` is compared with a number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengthOp {
    Eq,
    Ne,
    Ge,
    Le,
    Gt,
    Lt,
}

/// One condition on one attribute of a token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cond {
    Text {
        attr: Attr,
        equals: String,
    },
    TextIn {
        attr: Attr,
        values: Vec<String>,
        negated: bool,
    },
    Length {
        op: LengthOp,
        n: u64,
    },
    LengthIn {
        values: Vec<u64>,
        negated: bool,
    },
}

/// spaCy's `ONE`, `ZERO_ONE`, `ZERO_PLUS` and `ZERO` (matcher.pxd:23-29).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Quant {
    One,
    ZeroOne,
    ZeroPlus,
    Zero,
}

/// One token spec of a pattern. `quants` is what `_get_operators` expands the
/// spec's `OP` to (matcher.pyx:1169-1203); a spec with no condition matches any
/// token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpec {
    pub conds: Vec<Cond>,
    pub quants: Vec<Quant>,
}

/// A parsed token pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenPattern {
    pub specs: Vec<TokenSpec>,
}

/// What [`TokenPattern::parse`] refused, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    NotAList,
    NotAnObject { index: usize },
    Empty,
    UnknownKey { index: usize, key: String },
    Unsupported { index: usize, key: String },
    BadOp { index: usize, op: String },
    BadValue { index: usize, key: String },
}

impl TokenPattern {
    /// Reads a pattern as spaCy writes it: a JSON list of token-spec objects.
    ///
    /// Stricter than spaCy in three places, all refusals: an empty `{}` is the
    /// only falsy spec accepted (spaCy treats any falsy spec as "any token",
    /// matcher.pyx:836-838, while its schema rejects them); a non-object is
    /// `NotAnObject` (matcher.pyx:840-841 raises E154 only for a truthy one);
    /// an unknown predicate name inside a value is `UnknownKey` (spaCy warns
    /// W035 and skips it, matcher.pyx:1115-1119; the schema forbids it).
    /// The empty list is refused as spaCy does (E012, matcher.pyx:122-123;
    /// `min_length=1`, schemas.py:317).
    pub fn parse(pattern: &Value) -> Result<TokenPattern, Refusal> {
        let list = pattern.as_array().ok_or(Refusal::NotAList)?;
        if list.is_empty() {
            return Err(Refusal::Empty);
        }
        let mut specs = Vec::with_capacity(list.len());
        for (index, item) in list.iter().enumerate() {
            let object = item.as_object().ok_or(Refusal::NotAnObject { index })?;
            specs.push(parse_spec(index, object)?);
        }
        Ok(TokenPattern { specs })
    }

    /// Every `(start, end)`, `start < end`, such that `tokens[start..end]`
    /// matches the whole pattern, sorted and without duplicates. This is
    /// spaCy's result without `greedy` (matcher.pyx:282-284) minus the match id:
    /// duplicates are dropped and zero-length matches skipped (matcher.pyx:409-426).
    ///
    /// The expanded node list is searched directly. The node rules are those
    /// of `get_action` (matcher.pyx:690-735) as `transition_states` applies them
    /// (matcher.pyx:466-562) and `finish_states` closes them (matcher.pyx:596-621):
    /// `One` consumes one satisfying token; `ZeroOne` consumes zero or one
    /// (RETRY_ADVANCE and MATCH_DOUBLE keep both branches); `ZeroPlus` consumes
    /// zero or more (RETRY_EXTEND, MATCH_EXTEND, MATCH_REJECT); `Zero` (`!`) is
    /// `One` with `is_match` negated (matcher.pyx:690-692): it consumes exactly
    /// one token that does NOT satisfy the conditions. A trailing run of
    /// `ZeroOne`/`ZeroPlus` nodes may end empty; a trailing `One` or `Zero` may not.
    pub fn spans(&self, tokens: &[&str]) -> Vec<(usize, usize)> {
        let n = tokens.len();
        let satisfied: Vec<Vec<bool>> = self
            .specs
            .iter()
            .map(|spec| {
                tokens
                    .iter()
                    .map(|token| spec.conds.iter().all(|cond| cond.holds(token)))
                    .collect()
            })
            .collect();
        let nodes: Vec<(usize, &Quant)> = self
            .specs
            .iter()
            .enumerate()
            .flat_map(|(k, spec)| spec.quants.iter().map(move |quant| (k, quant)))
            .collect();

        let mut spans = Vec::new();
        let mut cur = vec![false; n + 1];
        let mut next = vec![false; n + 1];
        for start in 0..n {
            cur.fill(false);
            cur[start] = true;
            for &(k, quant) in &nodes {
                step(quant, &satisfied[k], &cur, &mut next);
                std::mem::swap(&mut cur, &mut next);
            }
            spans.extend(
                (start + 1..=n)
                    .filter(|&end| cur[end])
                    .map(|end| (start, end)),
            );
        }
        spans
    }

    /// Whether the pattern matches ALL of `tokens`. False for no token.
    pub fn covers(&self, tokens: &[&str]) -> bool {
        !tokens.is_empty() && self.spans(tokens).contains(&(0, tokens.len()))
    }

    /// Whether the pattern matches the whole of `value` under `split`.
    pub fn matches_value(&self, split: &Split, value: &str) -> bool {
        self.covers(&split.tokens(value))
    }
}

impl Cond {
    fn holds(&self, token: &str) -> bool {
        match self {
            Cond::Text { attr, equals } => {
                text_attr(attr, token).is_some_and(|text| text.as_ref() == equals.as_str())
            }
            Cond::TextIn {
                attr,
                values,
                negated,
            } => {
                let found = text_attr(attr, token)
                    .is_some_and(|text| values.iter().any(|v| v.as_str() == text.as_ref()));
                found != *negated
            }
            Cond::Length { op, n } => {
                let len = length(token);
                match op {
                    LengthOp::Eq => len == *n,
                    LengthOp::Ne => len != *n,
                    LengthOp::Ge => len >= *n,
                    LengthOp::Le => len <= *n,
                    LengthOp::Gt => len > *n,
                    LengthOp::Lt => len < *n,
                }
            }
            Cond::LengthIn { values, negated } => values.contains(&length(token)) != *negated,
        }
    }
}

/// `LENGTH` is `len(token.text)` in spaCy (vocab.pyx:229 sets `lex.length`,
/// lexeme.pxd:77-78 reads it): Python code points, here Unicode scalar values.
fn length(token: &str) -> u64 {
    token.chars().count() as u64
}

/// The text an attribute reads. `Lower` is Rust's `to_lowercase`; spaCy's
/// `lex.lower` comes from a language's lexical-attribute getters
/// (vocab.pyx:234-236), which were NOT read: equal for ASCII, not verified
/// for other scripts. `Length` is a number, never text: `parse` does not
/// produce a text condition on it, and one built by hand never equals a string.
fn text_attr<'t>(attr: &Attr, token: &'t str) -> Option<Cow<'t, str>> {
    match attr {
        Attr::Orth => Some(Cow::Borrowed(token)),
        Attr::Lower => Some(Cow::Owned(token.to_lowercase())),
        Attr::Length => None,
    }
}

/// One node: from the positions reachable in `cur`, the positions reachable in
/// `next` after the node. `satisfied[p]` says token `p` meets the node's spec.
fn step(quant: &Quant, satisfied: &[bool], cur: &[bool], next: &mut [bool]) {
    let n = satisfied.len();
    match quant {
        Quant::One | Quant::Zero => {
            let wanted = *quant == Quant::One;
            next.fill(false);
            for p in 0..n {
                if cur[p] && satisfied[p] == wanted {
                    next[p + 1] = true;
                }
            }
        }
        Quant::ZeroOne => {
            next.copy_from_slice(cur);
            for p in 0..n {
                if cur[p] && satisfied[p] {
                    next[p + 1] = true;
                }
            }
        }
        Quant::ZeroPlus => {
            for p in 0..=n {
                next[p] = cur[p] || (p > 0 && next[p - 1] && satisfied[p - 1]);
            }
        }
    }
}

fn bad_value(index: usize, key: &str) -> Refusal {
    Refusal::BadValue {
        index,
        key: key.to_string(),
    }
}

fn parse_spec(index: usize, spec: &Map<String, Value>) -> Result<TokenSpec, Refusal> {
    let mut conds = Vec::new();
    let mut quants: Option<Vec<Quant>> = None;
    for (key, value) in spec {
        // Keys are upper-cased as spaCy does (matcher.pyx:856, 1099-1103,
        // 1174-1175); the schema alone is case-sensitive (schemas.py:304-305).
        let upper = key.to_uppercase();
        match upper.as_str() {
            "OP" => {
                // Two OP keys differing in case: spaCy keeps the last in dict
                // order (matcher.pyx:1174-1175); the order is not ours to pick.
                if quants.is_some() {
                    return Err(bad_value(index, key));
                }
                quants = Some(parse_op(index, value)?);
            }
            // TEXT is ORTH (matcher.pyx:861-862).
            "ORTH" | "TEXT" => text_conds(index, key, Attr::Orth, value, &mut conds)?,
            "LOWER" => text_conds(index, key, Attr::Lower, value, &mut conds)?,
            "LENGTH" => length_conds(index, key, value, &mut conds)?,
            other
                if UNSUPPORTED_KEYS.contains(&other) || UNSUPPORTED_PREDICATES.contains(&other) =>
            {
                return Err(Refusal::Unsupported {
                    index,
                    key: key.clone(),
                });
            }
            _ => {
                return Err(Refusal::UnknownKey {
                    index,
                    key: key.clone(),
                })
            }
        }
    }
    Ok(TokenSpec {
        conds,
        quants: quants.unwrap_or_else(|| vec![Quant::One]),
    })
}

fn parse_op(index: usize, value: &Value) -> Result<Vec<Quant>, Refusal> {
    let Some(op) = value.as_str() else {
        return Err(Refusal::BadOp {
            index,
            op: value.to_string(),
        });
    };
    operators(op).ok_or_else(|| Refusal::BadOp {
        index,
        op: op.to_string(),
    })
}

/// `_get_operators` (matcher.pyx:1169-1203), `None` where it raises.
fn operators(op: &str) -> Option<Vec<Quant>> {
    // matcher.pyx:1171-1172. Its `"1"` entry is left out: the schema rejects it
    // (schemas.py:244-251, an enum of `+ * ? !` and a brace regex), so it only
    // works with validation off.
    match op {
        "*" => return Some(vec![Quant::ZeroPlus]),
        "+" => return Some(vec![Quant::One, Quant::ZeroPlus]),
        "?" => return Some(vec![Quant::ZeroOne]),
        "!" => return Some(vec![Quant::Zero]),
        _ => {}
    }
    if !(op.starts_with('{') && op.ends_with('}')) {
        return None;
    }
    let inner = &op[1..op.len() - 1];
    let (n, m) = match inner.split_once(',') {
        Some((_, m)) if m.contains(',') => return None,
        Some(parts) => parts,
        None => (inner, inner),
    };
    // spaCy tests `isdecimal` (Unicode digits); only ASCII digits are read here.
    let is_number = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if (n.is_empty() && m.is_empty())
        || (!n.is_empty() && !is_number(n))
        || (!m.is_empty() && !is_number(m))
    {
        return None;
    }
    let count = |s: &str| -> Option<usize> {
        if s.is_empty() {
            Some(0)
        } else {
            s.parse::<usize>().ok().filter(|c| *c <= MAX_REPEAT)
        }
    };
    let n_count = count(n)?;
    let mut quants = vec![Quant::One; n_count];
    if m.is_empty() {
        quants.push(Quant::ZeroPlus);
    } else {
        let m_count = count(m)?;
        if n_count > m_count {
            return None;
        }
        quants.extend(vec![Quant::ZeroOne; m_count - n_count]);
    }
    Some(quants)
}

fn strings(operand: &Value) -> Option<Vec<String>> {
    operand
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect()
}

fn integers(operand: &Value) -> Option<Vec<u64>> {
    operand.as_array()?.iter().map(|v| v.as_u64()).collect()
}

/// What a predicate name inside a value is refused as. A comparison on a text
/// attribute has no field in the string schema (schemas.py:175-208).
fn refuse_predicate(index: usize, key: &str, name: &str) -> Refusal {
    if UNSUPPORTED_PREDICATES.contains(&name.to_uppercase().as_str()) {
        Refusal::Unsupported {
            index,
            key: name.to_string(),
        }
    } else if ["==", "!=", ">=", "<=", ">", "<"].contains(&name) {
        bad_value(index, key)
    } else {
        Refusal::UnknownKey {
            index,
            key: name.to_string(),
        }
    }
}

/// A string, or `{"IN": [...]}` / `{"NOT_IN": [...]}`. An empty `{}` adds no
/// condition, as in spaCy (matcher.pyx:875-876 skips a dict, and none of its
/// predicates exist).
fn text_conds(
    index: usize,
    key: &str,
    attr: Attr,
    value: &Value,
    out: &mut Vec<Cond>,
) -> Result<(), Refusal> {
    match value {
        Value::String(equals) => out.push(Cond::Text {
            attr,
            equals: equals.clone(),
        }),
        Value::Object(predicates) => {
            for (name, operand) in predicates {
                let upper = name.to_uppercase();
                if upper == "IN" || upper == "NOT_IN" {
                    let values = strings(operand).ok_or_else(|| bad_value(index, key))?;
                    out.push(Cond::TextIn {
                        attr: attr.clone(),
                        values,
                        negated: upper == "NOT_IN",
                    });
                } else {
                    return Err(refuse_predicate(index, key, name));
                }
            }
        }
        _ => return Err(bad_value(index, key)),
    }
    Ok(())
}

/// An integer, or an object of `IN` / `NOT_IN` and the six comparisons.
/// Integers only and non-negative: spaCy's schema also takes floats and
/// negatives (schemas.py:221-232, 256); both are refused here.
fn length_conds(
    index: usize,
    key: &str,
    value: &Value,
    out: &mut Vec<Cond>,
) -> Result<(), Refusal> {
    match value {
        Value::Number(_) => {
            let n = value.as_u64().ok_or_else(|| bad_value(index, key))?;
            out.push(Cond::Length {
                op: LengthOp::Eq,
                n,
            });
        }
        Value::Object(predicates) => {
            for (name, operand) in predicates {
                let op = match name.as_str() {
                    "==" => Some(LengthOp::Eq),
                    "!=" => Some(LengthOp::Ne),
                    ">=" => Some(LengthOp::Ge),
                    "<=" => Some(LengthOp::Le),
                    ">" => Some(LengthOp::Gt),
                    "<" => Some(LengthOp::Lt),
                    _ => None,
                };
                if let Some(op) = op {
                    let n = operand.as_u64().ok_or_else(|| bad_value(index, key))?;
                    out.push(Cond::Length { op, n });
                    continue;
                }
                let upper = name.to_uppercase();
                if upper == "IN" || upper == "NOT_IN" {
                    let values = integers(operand).ok_or_else(|| bad_value(index, key))?;
                    out.push(Cond::LengthIn {
                        values,
                        negated: upper == "NOT_IN",
                    });
                } else {
                    return Err(refuse_predicate(index, key, name));
                }
            }
        }
        _ => return Err(bad_value(index, key)),
    }
    Ok(())
}
