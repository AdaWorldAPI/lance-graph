//! Method declarations loaded at run time.
//!
//! A [`MethodDecl`] is a [`MethodSpec`] whose strings are owned, parsed from
//! text after the binary was compiled. It is not executed here and has no
//! evaluator of its own: [`MethodTable::prepare_with`] lends each declaration
//! to [`MethodTable::prepare`] as a borrowed spec, so a loaded method goes
//! through the same binding, claim, law and lowering checks as a compiled one
//! and runs as the same Quack program. Preparation copies out the numbers it
//! keeps, so the declarations and the text can be dropped afterwards.
//!
//! The grammar is deliberately small. One declaration per line; `#` starts a
//! comment line:
//!
//! ```text
//! <concept>::<label> key=<32 hex> schema=<16 hex> reading=<u16> := <folds> [WHERE <conds>]
//!
//! folds := fold { "," fold }
//! fold  := COUNT(*) | SUM(<field>) | GROUP_SUM(<val> BY <key> IN <groups>)
//! conds := cond { AND cond }
//! cond  := <field> = <literal> | <field> != <literal>
//! ```
//!
//! Every loaded body is a pure query finished by [`Law::Values`]. Anything
//! else is refused by name ([`DeclError::Unsupported`]): `OR`, `NOT`,
//! parentheses in a condition, ordering comparisons, `IS`, arithmetic, any
//! other aggregate (including the NARS law), and a reference to another
//! method (the ledger's next step). The concept is resolved through OGAR's
//! codebook, as [`crate::call::resolve_call`] does; the declaration's seal and
//! reading are checked against the glove at preparation, never trusted here.
use lance_graph_ogar::ogar_vocab::canonical_concept_id;

use crate::method::{
    Cond, EffectClass, Fold, Glove, Law, MethodBody, MethodKey, MethodSpec, MethodTable, ReadingId,
    SchemaSeal,
};

/// A method declared in text: a [`MethodSpec`] with owned storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDecl {
    /// Display label; resolves to the key, never used as one.
    pub label: String,
    /// Canonical concept id.
    pub concept: u16,
    pub key: MethodKey,
    pub schema: SchemaSeal,
    pub reading: ReadingId,
    filter: Vec<DeclCond>,
    folds: Vec<DeclFold>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclCond {
    field: String,
    literal: String,
    negate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeclFold {
    Count,
    Sum(String),
    GroupSum {
        key: String,
        val: String,
        groups: u32,
    },
}

impl MethodDecl {
    fn conds(&self) -> Vec<Cond<'_>> {
        self.filter
            .iter()
            .map(|c| {
                let (field, literal) = (c.field.as_str(), c.literal.as_str());
                if c.negate {
                    Cond::Ne { field, literal }
                } else {
                    Cond::Eq { field, literal }
                }
            })
            .collect()
    }

    fn folds(&self) -> Vec<Fold<'_>> {
        self.folds
            .iter()
            .map(|f| match f {
                DeclFold::Count => Fold::Count,
                DeclFold::Sum(field) => Fold::Sum(field),
                DeclFold::GroupSum { key, val, groups } => Fold::GroupSum {
                    key,
                    val,
                    groups: *groups,
                },
            })
            .collect()
    }

    /// The spec this declaration is, borrowing `conds` and `folds` (its own
    /// conditions and folds as views).
    fn spec<'a>(&self, conds: &'a [Cond<'a>], folds: &'a [Fold<'a>]) -> MethodSpec<'a> {
        MethodSpec {
            key: self.key,
            concept: self.concept,
            schema: self.schema,
            reading: self.reading,
            effect: EffectClass::Pure,
            body: MethodBody::Query {
                filter: conds,
                folds,
                law: Law::Values,
            },
        }
    }
}

impl MethodTable {
    /// [`MethodTable::prepare`] over compiled specs and loaded declarations
    /// together. Keys and labels are checked across both lists by the same
    /// rule as within one, so a loaded method cannot shadow a compiled one.
    #[must_use]
    pub fn prepare_with(
        glove: &Glove<'_>,
        compiled: &[(&str, MethodSpec<'_>)],
        declared: &[MethodDecl],
    ) -> Self {
        let views: Vec<(Vec<Cond<'_>>, Vec<Fold<'_>>)> =
            declared.iter().map(|d| (d.conds(), d.folds())).collect();
        let mut specs: Vec<(&str, MethodSpec<'_>)> = compiled.to_vec();
        specs.extend(
            declared
                .iter()
                .zip(&views)
                .map(|(d, (c, f))| (d.label.as_str(), d.spec(c, f))),
        );
        Self::prepare(glove, &specs)
    }
}

/// Why a declaration did not load. `line` is 1-based.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclError {
    /// The line is not of the grammar's form.
    Syntax { line: usize, msg: String },
    /// The line uses a form the grammar names and refuses.
    Unsupported { line: usize, what: String },
    /// OGAR's codebook has no concept of that name.
    UnknownConcept { line: usize, name: String },
}

/// Parse every declaration in `text`.
///
/// # Errors
///
/// The first line that does not load, with its reason.
pub fn parse_decls(text: &str) -> Result<Vec<MethodDecl>, DeclError> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(parse_line(line, i + 1)?);
    }
    Ok(out)
}

fn ident(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn parse_line(line: &str, n: usize) -> Result<MethodDecl, DeclError> {
    let syntax = |msg: &str| DeclError::Syntax {
        line: n,
        msg: msg.to_owned(),
    };
    let (head, body) = line
        .split_once(":=")
        .ok_or_else(|| syntax("expected `:=`"))?;
    let mut words = head.split_whitespace();
    let name = words.next().ok_or_else(|| syntax("expected a name"))?;
    let (concept, label) = name
        .split_once("::")
        .filter(|(c, l)| ident(c) && ident(l))
        .ok_or_else(|| syntax("expected `<concept>::<label>`"))?;
    let (mut key, mut schema, mut reading) = (None, None, None);
    for w in words {
        let (k, v) = w
            .split_once('=')
            .ok_or_else(|| syntax("expected `name=value`"))?;
        let dup = match k {
            "key" => key
                .replace(hex::<16>(v).ok_or_else(|| syntax("key is 32 hex digits"))?)
                .is_some(),
            "schema" => schema
                .replace(hex::<8>(v).ok_or_else(|| syntax("schema is 16 hex digits"))?)
                .is_some(),
            "reading" => reading
                .replace(v.parse::<u16>().map_err(|_| syntax("reading is a u16"))?)
                .is_some(),
            _ => return Err(syntax("unknown attribute")),
        };
        if dup {
            return Err(syntax("attribute given twice"));
        }
    }
    let (Some(key), Some(schema), Some(reading)) = (key, schema, reading) else {
        return Err(syntax("expected key=, schema= and reading="));
    };
    let concept_id = canonical_concept_id(concept).ok_or_else(|| DeclError::UnknownConcept {
        line: n,
        name: concept.to_owned(),
    })?;
    let toks = tokenize(body).map_err(|m| syntax(&m))?;
    let (folds, filter) = Parser {
        toks: &toks,
        at: 0,
        line: n,
    }
    .body()?;
    Ok(MethodDecl {
        label: label.to_owned(),
        concept: concept_id,
        key: MethodKey(key),
        schema: SchemaSeal(u64::from_be_bytes(schema)),
        reading: ReadingId(reading),
        filter,
        folds,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Word(String),
    Str(String),
    Punct(&'static str),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    const PUNCT: [&str; 13] = [
        "!=", "<=", ">=", "<>", "(", ")", ",", "*", "=", "<", ">", "+", "/",
    ];
    let mut out = Vec::new();
    let mut rest = s;
    'outer: loop {
        rest = rest.trim_start();
        let Some(c) = rest.chars().next() else {
            return Ok(out);
        };
        for p in PUNCT {
            if let Some(r) = rest.strip_prefix(p) {
                out.push(Tok::Punct(p));
                rest = r;
                continue 'outer;
            }
        }
        if c == '"' {
            let end = rest[1..].find('"').ok_or("unterminated string")?;
            out.push(Tok::Str(rest[1..=end].to_owned()));
            rest = &rest[end + 2..];
            continue;
        }
        let len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'))
            .unwrap_or(rest.len());
        if len == 0 {
            return Err(format!("unexpected `{c}`"));
        }
        out.push(Tok::Word(rest[..len].to_owned()));
        rest = &rest[len..];
    }
}

struct Parser<'t> {
    toks: &'t [Tok],
    at: usize,
    line: usize,
}

impl<'t> Parser<'t> {
    fn syntax(&self, msg: &str) -> DeclError {
        DeclError::Syntax {
            line: self.line,
            msg: msg.to_owned(),
        }
    }

    fn unsupported(&self, what: &str) -> DeclError {
        DeclError::Unsupported {
            line: self.line,
            what: what.to_owned(),
        }
    }

    fn peek(&self) -> Option<&'t Tok> {
        self.toks.get(self.at)
    }

    fn peek_at(&self, k: usize) -> Option<&'t Tok> {
        self.toks.get(self.at + k)
    }

    fn next(&mut self) -> Option<&'t Tok> {
        let t = self.toks.get(self.at);
        self.at += 1;
        t
    }

    fn is_word(t: Option<&Tok>, w: &str) -> bool {
        matches!(t, Some(Tok::Word(x)) if x.eq_ignore_ascii_case(w))
    }

    fn punct(&mut self, p: &str) -> Result<(), DeclError> {
        match self.next() {
            Some(Tok::Punct(q)) if *q == p => Ok(()),
            _ => Err(self.syntax(&format!("expected `{p}`"))),
        }
    }

    fn field(&mut self) -> Result<String, DeclError> {
        match self.next().cloned() {
            Some(Tok::Word(w)) if ident(&w) && !is_keyword(&w) => Ok(w),
            _ => Err(self.syntax("expected a field name")),
        }
    }

    fn body(mut self) -> Result<(Vec<DeclFold>, Vec<DeclCond>), DeclError> {
        let mut folds = vec![self.fold()?];
        while matches!(self.peek(), Some(Tok::Punct(","))) {
            self.at += 1;
            folds.push(self.fold()?);
        }
        let mut filter = Vec::new();
        if Self::is_word(self.peek(), "WHERE") {
            self.at += 1;
            filter.push(self.cond()?);
            loop {
                match self.peek() {
                    None => break,
                    t if Self::is_word(t, "AND") => {
                        self.at += 1;
                        filter.push(self.cond()?);
                    }
                    Some(Tok::Word(w)) if w.eq_ignore_ascii_case("OR") => {
                        return Err(self.unsupported(w))
                    }
                    _ => return Err(self.syntax("expected AND or the end")),
                }
            }
        }
        if self.peek().is_some() {
            return Err(self.syntax("unexpected text after the body"));
        }
        Ok((folds, filter))
    }

    fn fold(&mut self) -> Result<DeclFold, DeclError> {
        let Some(Tok::Word(name)) = self.next().cloned() else {
            return Err(self.syntax("expected an aggregate"));
        };
        if !matches!(self.peek(), Some(Tok::Punct("("))) {
            return Err(self.syntax("expected `(` after the aggregate"));
        }
        let known = ["COUNT", "SUM", "GROUP_SUM"]
            .iter()
            .find(|k| name.eq_ignore_ascii_case(k));
        let Some(&known) = known else {
            return Err(self.unsupported(&name));
        };
        self.at += 1;
        // A call inside the parentheses names another method.
        let close = self.toks[self.at..]
            .iter()
            .position(|t| *t == Tok::Punct(")"))
            .map_or(self.toks.len(), |c| self.at + c);
        if self.toks[self.at..close]
            .windows(2)
            .any(|w| matches!(w, [Tok::Word(_), Tok::Punct("(")]))
        {
            return Err(self.unsupported("method reference"));
        }
        let fold = match known {
            "COUNT" => {
                self.punct("*")?;
                DeclFold::Count
            }
            "SUM" => DeclFold::Sum(self.operand()?),
            _ => {
                let val = self.operand()?;
                if !Self::is_word(self.next(), "BY") {
                    return Err(self.syntax("expected BY"));
                }
                let key = self.operand()?;
                if !Self::is_word(self.next(), "IN") {
                    return Err(self.syntax("expected IN"));
                }
                let groups = match self.next() {
                    Some(Tok::Word(w)) => w.parse::<u32>().ok().filter(|&g| g > 0),
                    _ => None,
                }
                .ok_or_else(|| self.syntax("IN takes a positive group count"))?;
                DeclFold::GroupSum { key, val, groups }
            }
        };
        self.punct(")")?;
        Ok(fold)
    }

    /// A field inside an aggregate; an operator after it is arithmetic.
    fn operand(&mut self) -> Result<String, DeclError> {
        let f = self.field()?;
        if let Some(Tok::Punct(p @ ("*" | "+" | "/"))) = self.peek() {
            return Err(self.unsupported(p));
        }
        Ok(f)
    }

    fn cond(&mut self) -> Result<DeclCond, DeclError> {
        match self.peek() {
            Some(Tok::Punct("(")) => return Err(self.unsupported("(")),
            Some(Tok::Word(w)) if w.eq_ignore_ascii_case("NOT") => return Err(self.unsupported(w)),
            _ => {}
        }
        if matches!(self.peek_at(1), Some(Tok::Punct("("))) {
            return Err(self.unsupported("method reference"));
        }
        let field = self.field()?;
        let negate = match self.next() {
            Some(Tok::Punct("=")) => false,
            Some(Tok::Punct("!=")) => true,
            Some(Tok::Punct(p)) => return Err(self.unsupported(p)),
            Some(Tok::Word(w)) if w.eq_ignore_ascii_case("IS") => return Err(self.unsupported(w)),
            _ => return Err(self.syntax("expected `=` or `!=`")),
        };
        let literal = match self.next() {
            Some(Tok::Word(w)) if !is_keyword(w) => w.clone(),
            Some(Tok::Str(s)) => s.clone(),
            _ => return Err(self.syntax("expected a literal")),
        };
        Ok(DeclCond {
            field,
            literal,
            negate,
        })
    }
}

fn is_keyword(w: &str) -> bool {
    ["WHERE", "AND", "OR", "NOT", "BY", "IN", "IS"]
        .iter()
        .any(|k| w.eq_ignore_ascii_case(k))
}
