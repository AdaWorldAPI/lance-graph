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

use crate::gremlin::{compose, GremlinRefusal, Pred, Step};
use crate::method::{
    Cond, EffectClass, Fold, Glove, Law, MethodBody, MethodKey, MethodRef, MethodSpec, MethodTable,
    ReadingId, SchemaSeal,
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
enum DeclCond {
    Field {
        field: String,
        literal: String,
        negate: bool,
    },
    /// A predicate method, by label; resolved at preparation.
    Ref(String),
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
            .map(|c| match c {
                DeclCond::Field {
                    field,
                    literal,
                    negate: true,
                } => Cond::Ne { field, literal },
                DeclCond::Field { field, literal, .. } => Cond::Eq { field, literal },
                DeclCond::Ref(label) => Cond::Method(MethodRef::Label(label)),
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

    /// What the method may do. A declaration can only express a pure query
    /// or a pure predicate: no grammar form loads an effect, so loading
    /// grants no authority.
    #[must_use]
    pub fn effect(&self) -> EffectClass {
        EffectClass::Pure
    }

    /// The spec this declaration is, borrowing `conds` and `folds` (its own
    /// conditions and folds as views). No folds is a predicate.
    fn spec<'a>(&self, conds: &'a [Cond<'a>], folds: &'a [Fold<'a>]) -> MethodSpec<'a> {
        MethodSpec {
            key: self.key,
            concept: self.concept,
            schema: self.schema,
            reading: self.reading,
            effect: self.effect(),
            body: if folds.is_empty() {
                MethodBody::Select { filter: conds }
            } else {
                MethodBody::Query {
                    filter: conds,
                    folds,
                    law: Law::Values,
                }
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

/// The largest `GROUP_SUM … IN n` a loaded declaration may ask for. Each
/// call allocates an `n`-slot `i64` sink (here 512 KiB), so text from outside
/// the build must not choose `n` freely. A policy pin, not a measurement: a
/// compiled spec is not bound by it, and a wider domain needs its own case.
pub const MAX_DECLARED_GROUPS: u32 = 1 << 16;

/// Why a declaration did not load. `line` is 1-based.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclError {
    /// The line is not of the grammar's form.
    Syntax { line: usize, msg: String },
    /// The line uses a form the grammar names and refuses.
    Unsupported { line: usize, what: String },
    /// OGAR's codebook has no concept of that name.
    UnknownConcept { line: usize, name: String },
    /// A Gremlin step that changes the world. No declaration can carry one.
    Effect { line: usize, step: String },
    /// The steps parse, but break the composer's anchor contract.
    Refused {
        line: usize,
        refusal: GremlinRefusal,
    },
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
    let (folds, filter) = if body.trim_start().starts_with("V(") {
        traversal(body.trim(), concept_id, n)?
    } else {
        let toks = tokenize(body).map_err(|m| syntax(&m))?;
        Parser {
            toks: &toks,
            at: 0,
            line: n,
        }
        .body()?
    };
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
                    Some(Tok::Word(w)) => w
                        .parse::<u32>()
                        .ok()
                        .filter(|&g| (1..=MAX_DECLARED_GROUPS).contains(&g)),
                    _ => None,
                }
                .ok_or_else(|| {
                    self.syntax(&format!(
                        "IN takes a group count in 1..={MAX_DECLARED_GROUPS}"
                    ))
                })?;
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
        Ok(DeclCond::Field {
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

// ---------------------------------------------------------------- Gremlin form

/// Steps that change the world (TinkerPop 3.7.7). Refused as effects, never
/// as syntax, so a caller can tell "not allowed" from "not understood".
const EFFECT_STEPS: [&str; 13] = [
    "addV",
    "addE",
    "property",
    "drop",
    "mergeV",
    "mergeE",
    "sideEffect",
    "aggregate",
    "store",
    "sack",
    "inject",
    "io",
    "call",
];

/// `V(concept).step(args)…`, parsed into typed [`Step`]s and composed by
/// [`compose`]; the result is copied into owned conditions and folds.
fn traversal(
    body: &str,
    concept: u16,
    n: usize,
) -> Result<(Vec<DeclFold>, Vec<DeclCond>), DeclError> {
    let syntax = |msg: &str| DeclError::Syntax {
        line: n,
        msg: msg.to_owned(),
    };
    let unsupported = |what: &str| DeclError::Unsupported {
        line: n,
        what: what.to_owned(),
    };
    let mut steps = Vec::new();
    for (name, args) in calls(body).ok_or_else(|| syntax("expected `name(args).name(args)…`"))? {
        let arity = |k: usize| {
            let a = split_args(args);
            if a.len() == k {
                Ok(a)
            } else {
                Err(syntax(&format!("{name} takes {k} argument(s)")))
            }
        };
        let field = |s: &str| {
            if ident(s) {
                Ok(s.to_owned())
            } else {
                Err(syntax("expected a field name"))
            }
        };
        if EFFECT_STEPS.contains(&name) {
            return Err(DeclError::Effect {
                line: n,
                step: name.to_owned(),
            });
        }
        steps.push(match name {
            "V" => {
                let [c] = arity(1)?[..] else { unreachable!() };
                Step::V(
                    canonical_concept_id(c).ok_or_else(|| DeclError::UnknownConcept {
                        line: n,
                        name: c.to_owned(),
                    })?,
                )
            }
            "has" => {
                let [f, p] = arity(2)?[..] else {
                    unreachable!()
                };
                field(f)?;
                let pred = match calls(p) {
                    Some(c) if c.len() == 1 => {
                        let (pn, pa) = c[0];
                        let [lit] = split_args(pa)[..] else {
                            return Err(syntax("a predicate takes one literal"));
                        };
                        match pn {
                            "eq" => Pred::Eq(literal(lit).ok_or_else(|| syntax("literal"))?),
                            "neq" => Pred::Neq(literal(lit).ok_or_else(|| syntax("literal"))?),
                            other => return Err(unsupported(other)),
                        }
                    }
                    Some(_) => return Err(syntax("one predicate")),
                    None => Pred::Eq(literal(p).ok_or_else(|| syntax("expected a literal"))?),
                };
                Step::Has { field: f, pred }
            }
            "where" => {
                let [inner] = arity(1)?[..] else {
                    unreachable!()
                };
                match calls(inner).as_deref() {
                    Some([(label, "")]) if !is_step_name(label) => {
                        Step::Where(MethodRef::Label(label))
                    }
                    Some([(other, _), ..]) => return Err(unsupported(other)),
                    _ => return Err(syntax("where takes a method reference")),
                }
            }
            "values" => {
                let [f] = arity(1)?[..] else { unreachable!() };
                field(f)?;
                Step::Values(f)
            }
            "out" | "in" | "both" => {
                let [r] = arity(1)?[..] else { unreachable!() };
                field(r)?;
                match name {
                    "out" => Step::Out(r),
                    "in" => Step::In(r),
                    _ => Step::Both(r),
                }
            }
            "dedup" | "count" | "sum" => {
                if !args.trim().is_empty() {
                    return Err(syntax(&format!("{name} takes no arguments")));
                }
                match name {
                    "dedup" => Step::Dedup,
                    "count" => Step::Count,
                    _ => Step::Sum,
                }
            }
            other => return Err(unsupported(other)),
        });
    }
    let composed =
        compose(concept, &steps).map_err(|refusal| DeclError::Refused { line: n, refusal })?;
    let filter = composed
        .filter()
        .iter()
        .map(|c| match *c {
            Cond::Eq { field, literal } => DeclCond::Field {
                field: field.to_owned(),
                literal: literal.to_owned(),
                negate: false,
            },
            Cond::Ne { field, literal } => DeclCond::Field {
                field: field.to_owned(),
                literal: literal.to_owned(),
                negate: true,
            },
            Cond::Method(MethodRef::Label(l)) => DeclCond::Ref(l.to_owned()),
            Cond::Method(MethodRef::Key(_)) => unreachable!("text names methods by label"),
        })
        .collect();
    let folds = composed
        .folds()
        .iter()
        .map(|f| match *f {
            Fold::Count => DeclFold::Count,
            Fold::Sum(field) => DeclFold::Sum(field.to_owned()),
            Fold::GroupSum { key, val, groups } => DeclFold::GroupSum {
                key: key.to_owned(),
                val: val.to_owned(),
                groups,
            },
        })
        .collect();
    Ok((folds, filter))
}

/// A Gremlin step name: refused if used as a method label in `where`.
fn is_step_name(s: &str) -> bool {
    EFFECT_STEPS.contains(&s)
        || [
            "V", "has", "where", "values", "out", "in", "both", "dedup", "count", "sum", "not",
            "and", "or",
        ]
        .contains(&s)
}

/// `a(x).b(y)` → `[("a","x"), ("b","y")]`; arguments keep nested parentheses.
/// `None` if the text is not a dot-separated chain of calls.
fn calls(s: &str) -> Option<Vec<(&str, &str)>> {
    let mut out = Vec::new();
    let mut rest = s.trim();
    loop {
        let open = rest.find('(')?;
        let name = rest[..open].trim();
        if !ident(name) {
            return None;
        }
        let (mut depth, mut quoted) = (0usize, false);
        let mut close = None;
        for (i, c) in rest[open..].char_indices() {
            match c {
                '"' => quoted = !quoted,
                _ if quoted => {}
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close?;
        out.push((name, rest[open + 1..close].trim()));
        rest = rest[close + 1..].trim_start();
        if rest.is_empty() {
            return Some(out);
        }
        rest = rest.strip_prefix('.')?;
    }
}

/// Top-level comma split; empty input is no arguments.
fn split_args(s: &str) -> Vec<&str> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (mut depth, mut start, mut quoted) = (0usize, 0usize, false);
    for (i, c) in s.char_indices() {
        match c {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

/// A bare word or a double-quoted string.
fn literal(s: &str) -> Option<&str> {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        return Some(inner);
    }
    (!s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')))
    .then_some(s)
}
