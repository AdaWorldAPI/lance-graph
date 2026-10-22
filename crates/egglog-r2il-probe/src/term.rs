//! The probe's term language, parsed back from egglog's rendered terms.
//!
//! egglog hands terms out as text (`TermDag::to_string`, e.g.
//! `(Mul (Num 8) (Var "c"))`). The probe reads them back into [`Tm`] so that
//! "closed" and the lowering walk operate on a structure, not on substrings.
//! A term egglog renders in a shape this parser does not know (for example a
//! `let` for a shared subterm) parses to `None` and is skipped, never guessed.

use std::fmt;

/// `(datatype Term (Num i64) (Var String) (Def String :cost 1000) (Mul Term Term))`.
///
/// `Var` is the target variable, the only one an answer may mention. `Def`
/// is every other variable: a name the premises define, priced so that
/// egglog's own AST-size extraction prefers any expansion of it over the
/// name itself (see `crate::encode::HEADER`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tm {
    /// `(Num k)`.
    Num(i64),
    /// `(Var "v")`: the target variable.
    Var(String),
    /// `(Def "v")`: a non-target variable, defined by a premise.
    Def(String),
    /// `(Mul l r)`.
    Mul(Box<Tm>, Box<Tm>),
}

impl Tm {
    /// Parses one rendered term. `None` when the text is not exactly one term
    /// of the four constructors above.
    pub fn parse(text: &str) -> Option<Tm> {
        let toks = tokenize(text)?;
        let mut pos = 0;
        let tm = parse_tm(&toks, &mut pos)?;
        (pos == toks.len()).then_some(tm)
    }

    /// Number of constructor nodes (the default AST-size cost, up to the
    /// constant per-leaf weight).
    pub fn size(&self) -> usize {
        match self {
            Tm::Num(_) | Tm::Var(_) | Tm::Def(_) => 1,
            Tm::Mul(l, r) => 1 + l.size() + r.size(),
        }
    }

    /// Variable occurrences other than `target`, in left-to-right order.
    pub fn foreign_vars(&self, target: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_foreign(target, &mut out);
        out
    }

    fn collect_foreign(&self, target: &str, out: &mut Vec<String>) {
        match self {
            Tm::Num(_) => {}
            Tm::Var(v) | Tm::Def(v) => {
                if v != target {
                    out.push(v.clone());
                }
            }
            Tm::Mul(l, r) => {
                l.collect_foreign(target, out);
                r.collect_foreign(target, out);
            }
        }
    }
}

/// Canonical rendering, byte-identical to egglog's for these four
/// constructors: `(Mul (Num 8) (Var "c"))`.
impl fmt::Display for Tm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tm::Num(k) => write!(f, "(Num {k})"),
            Tm::Var(v) => write!(f, "(Var {v:?})"),
            Tm::Def(v) => write!(f, "(Def {v:?})"),
            Tm::Mul(l, r) => write!(f, "(Mul {l} {r})"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Tok {
    Open,
    Close,
    Atom(String),
    Str(String),
}

fn tokenize(text: &str) -> Option<Vec<Tok>> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' => {
                chars.next();
                out.push(Tok::Open);
            }
            ')' => {
                chars.next();
                out.push(Tok::Close);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                loop {
                    match chars.next()? {
                        '"' => break,
                        '\\' => s.push(chars.next()?),
                        ch => s.push(ch),
                    }
                }
                out.push(Tok::Str(s));
            }
            _ => {
                let mut s = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == '(' || ch == ')' || ch == '"' || ch.is_whitespace() {
                        break;
                    }
                    s.push(ch);
                    chars.next();
                }
                out.push(Tok::Atom(s));
            }
        }
    }
    Some(out)
}

fn parse_tm(toks: &[Tok], pos: &mut usize) -> Option<Tm> {
    if toks.get(*pos)? != &Tok::Open {
        return None;
    }
    *pos += 1;
    let Tok::Atom(head) = toks.get(*pos)? else {
        return None;
    };
    *pos += 1;
    let tm = match head.as_str() {
        "Num" => {
            let Tok::Atom(n) = toks.get(*pos)? else {
                return None;
            };
            *pos += 1;
            Tm::Num(n.parse().ok()?)
        }
        "Var" | "Def" => {
            let Tok::Str(s) = toks.get(*pos)? else {
                return None;
            };
            *pos += 1;
            if head == "Var" {
                Tm::Var(s.clone())
            } else {
                Tm::Def(s.clone())
            }
        }
        "Mul" => {
            let l = parse_tm(toks, pos)?;
            let r = parse_tm(toks, pos)?;
            Tm::Mul(Box::new(l), Box::new(r))
        }
        _ => return None,
    };
    if toks.get(*pos)? != &Tok::Close {
        return None;
    }
    *pos += 1;
    Some(tm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: a rendered term stops round-tripping, or a malformed one is
    /// accepted (trailing text, unknown head, unbalanced parens).
    #[test]
    fn parse_round_trips_and_rejects_malformed() {
        let t = Tm::parse(r#"(Mul (Num -8) (Var "c"))"#).expect("parses");
        assert_eq!(t.to_string(), r#"(Mul (Num -8) (Var "c"))"#);
        assert_eq!(t.size(), 3);
        assert_eq!(t.foreign_vars("c"), Vec::<String>::new());
        assert_eq!(t.foreign_vars("x"), vec!["c".to_string()]);
        for bad in [
            "",
            "(Num 8) (Num 9)",
            "(Frob 1)",
            "(Mul (Num 8))",
            r#"(Var c)"#,
            "(Num 8",
        ] {
            assert_eq!(Tm::parse(bad), None, "{bad:?}");
        }
    }
}
