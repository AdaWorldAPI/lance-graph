//! VRL recipes over the register contract (not Magic Sets / crepe rewrites, not
//! the NARS tactic recipes): the VRL read/rename/filter subset, lowered onto the
//! existing witnessed path (D-VRL-RECIPE-0). Moved here from the
//! `vrl-recipe-probe` lab, which keeps the reference interpreter and the
//! parity tests.
//!
//! Whatever the population size, the answer has one shape:
//! `identity | select {name = identity.key; identity.properties.f.value}`.
//! So the stored answer is a recipe, never rows: which fields to bind, the
//! comparison terms that drop rows, and the output aliases. It adds no authority
//! and no kernel of its own; it is the front of what D-DR-SELECT-K-SV-1/2
//! (#2039, #2040) already ship:
//!
//! - **Witness.** Every terminal reads its lanes through one
//!   [`LaneBinding::borrow_all`]: DT64 checks the frame (issuer, revisions,
//!   reseal), route, basis, mapping, scope and row order before a lane is
//!   borrowed. A recipe run after the source's frame moved is refused there,
//!   never read against new lanes.
//! - **Fold.** The population and every terminal are one mask-risc
//!   [`Program`]: `Range` for every row, one `i32` compare per `abort`
//!   (`EqI32` `NeI32` `LtI32` `LeI32` `GtI32` `GeI32`), `AndNot` to drop it, then `Count`, `MaskedSumI32` or `Keep`. The computation is
//!   mask-risc's, on `ndarray::simd`.
//! - **Rows.** [`Recipe::materialize_rows`] is the only row producer, and it
//!   is positional ([`Rows`]): the names once, the borrowed lanes, and a `u32`
//!   position table. No string or event is built per row.
//!
//! ## Which "recipe" this is
//!
//! Four things in this workspace are called recipes; they do not share code
//! or types, and a consumer should know which one it takes:
//!
//! - **This crate (`lance-graph-vrl`):** a VRL read/rename/filter program over
//!   one class's lanes, run as a mask-risc fold and answered as positional
//!   [`Rows`]. Consumers: report-rs's workbook XVERWEIS and, later, live log
//!   pipelines (Vector VRL).
//! - **Magic Sets / crepe** (`crepe-plan-probe`, a lab; a `lance-graph-crepe`
//!   crate would be its consumer-facing home): rewrites of recursive Datalog
//!   plans. Not a row projection, and not reached from here.
//! - **The planner's NARS recipes** (`lance-graph-planner`: `recipe_kernels`,
//!   `recipes`, `recipe_dispatch`): the tactic runbooks of the thinking stack.
//!   Not a data read.
//! - **xlog-cube's Route C recipe** (XLOG `crates/xlog-cube`, EXO-396): a
//!   non-recursive derived Datalog head kept as its body and substituted into
//!   readers, instead of being re-added as a relation. Same principle as this
//!   crate (the stored answer is a recipe, never rows); different language,
//!   no shared code.
//!
//! Lanes are read as contiguous slices: `i32` for any field, and `u32` (a
//! text column's codes, decoded by the consumer through its code table) for
//! a field that is only projected, never compared or summed. A strided lane
//! (for example a field read in place from 512-byte SoA rows) is refused
//! with [`Refusal::NotI32`], because mask-risc's strided predicates are
//! `u32` only. A consumer holds its own contiguous lanes until that changes.
//!
//! Rows come two ways: [`Recipe::materialize_rows`] folds the drop terms
//! over every row, and [`Recipe::rows_at`] reads the rows at positions the
//! source resolved from keys (a lookup such as XVERWEIS), under the frame
//! they were resolved in.
//!
//! The VRL subset, one statement per line (or `;`):
//!
//! - `.out = .key`, `.out = .other_out`, `.out = .properties.f.value`
//! - `del(.out)`
//! - `if .path <op> 3 { abort }` with `<op>` one of `==` `!=` `<` `<=` `>` `>=`
//!   (signed `i32`); a range is two aborts, since `&&` and `||` are refused
//! - `if .path == "text" { abort }` and `!=` on a text column: the literal is
//!   resolved to the column's code under a frame ([`Recipe::resolve_text`])
//!   before preparing; order comparisons on text are refused, since codes
//!   have no order
//!
//! A name is a field label of the class's [`ClassView`]; `key` is the field
//! labelled `key`. Output semantics, stated because VRL mutates the whole
//! event: the names the program assigned and did not delete afterwards.
//! Paths are resolved once, at compile time, with VRL's in-place mutation:
//! after `.key = .properties.x.value`, a later `.key` reads `x`.
//!
//! Everything that computes a value is refused by name: function calls other
//! than `del` (`parse_json`, `now`, `to_string`, ...), arithmetic, string or
//! integer literals assigned as values, any other comparison, `&&`/`||`, and
//! an `if` whose body is not `abort`.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_dt64::dr::{Frame, FrameStore};
use lance_graph_dt64::{BindStore, LaneBinding, LaneRefusal, PresenceSource};
use lance_graph_mask_risc::{
    execute_into, words_for, ExecError, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};

/// Why a program or a terminal is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The construct computes a value instead of reading one.
    Computes(String),
    /// A path that is neither a bound name nor `.properties.<label>.value`.
    UnknownPath(String),
    /// DT64 refused the witness; nothing was borrowed.
    Binding(LaneRefusal),
    /// The binding does not hold a field the recipe reads.
    NotBound { field: u8 },
    /// A field the recipe compares or sums is not a contiguous `i32` lane,
    /// or a projected field is neither a contiguous `i32` nor `u32` lane (a
    /// strided lane is refused; see the crate docs).
    NotI32 { field: u8 },
    /// The recipe reads no field, so it names no population.
    NoLanes,
    /// The recipe reads more fields than [`MAX_READS`].
    TooManyReads { reads: usize },
    /// A [`Prepared`] program run by a recipe that did not prepare it.
    NotThisRecipe,
    /// The witness is bound for another class than the recipe was compiled
    /// for.
    OtherClass,
    /// A field the recipe reads names another predicate in the witness's
    /// view than it did at compile time: the same position, another field.
    OtherField { field: u8 },
    /// The frame's row count is not the one the program was prepared for.
    RowsChanged { prepared: usize, now: usize },
    /// A field's presence plane does not span the lane's rows (`⌈n/64⌉`
    /// words).
    PresenceLength {
        field: u8,
        words: usize,
        need: usize,
    },
    /// mask-risc refused the program or its operands.
    Exec(ExecError),
    /// A recipe names a field it does not read. Not reachable through
    /// [`Recipe::compile`].
    NotARead { field: u8 },
    /// An order comparison (`<`, `<=`, `>`, `>=`) against a text literal: a
    /// text column's codes are dictionary addresses with no order.
    OrderOnText { op: String },
    /// The recipe has text terms and [`Recipe::resolve_text`] was not run.
    TextUnresolved,
    /// A field compared with a text literal is not a code (`u32`) lane.
    NotCode { field: u8 },
    /// [`Recipe::rows_at`] was given a recipe with a drop term.
    FiltersAtPositions,
    /// A lane read by [`Recipe::rows_at`] does not hold as many rows as the
    /// first read lane.
    LaneLength { field: u8, len: usize, need: usize },
    /// A position passed to [`Recipe::rows_at`] lies past the lanes.
    PositionOutOfRange { at: u32, n_rows: usize },
}

/// The most fields one recipe reads. The lanes are held in a fixed array of
/// this size, so the accepting path allocates nothing.
pub const MAX_READS: usize = 16;

/// The witnessed lanes, `len` of them, in `reads` order.
#[derive(Clone, Copy)]
struct Lanes<'a> {
    lanes: [LaneRef<'a>; MAX_READS],
    /// Each lane's presence plane (EXO-294), the mask-risc plane of the same
    /// index.
    presence: [&'a [u64]; MAX_READS],
    len: usize,
}

impl<'a> Lanes<'a> {
    fn get(&self) -> &[LaneRef<'a>] {
        &self.lanes[..self.len]
    }

    fn planes(&self) -> &[&'a [u64]] {
        &self.presence[..self.len]
    }
}

/// A recipe's program lowered for one row count: the cold step, done once.
#[derive(Clone, Debug)]
pub struct Prepared {
    program: Program,
    n_rows: usize,
    /// The preparing recipe's identity: its compiled content. Reads and drop
    /// terms alone do not tell recipes apart (two recipes can read the same
    /// fields and drop the same rows yet map an output name to different
    /// fields), so the output map is part of it. Equal recipes share
    /// prepared programs; nothing is counted.
    recipe: RecipeKey,
    /// The field a sum terminal folds; it must be an `i32` lane.
    sum: Option<u8>,
    /// The frame the text literals were resolved under, if the recipe has
    /// text terms; the program is refused once it is not current.
    frame: Option<Frame>,
}

/// What makes two compiled recipes the same recipe, compared exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecipeKey {
    class: ClassId,
    idents: Vec<String>,
    drop: Vec<(u8, Cmp, i32)>,
    text: Vec<(u8, bool, String)>,
    out: Vec<(String, u8)>,
}

/// Text literals resolved to codes under one frame, in `text` order.
#[derive(Clone, Debug)]
struct Codes {
    frame: Frame,
    of: Vec<Option<u32>>,
}

impl Prepared {
    /// Owned scratch sized for this program (tile-capped by mask-risc). The
    /// cold step; reuse it across [`Recipe::run_into`] calls.
    pub fn scratch(&self) -> Result<Scratch<'static>, Refusal> {
        Scratch::for_program(&self.program, self.n_rows).map_err(Refusal::Exec)
    }

    /// Scratch over the caller's reusable `buf`, sized for this program: the
    /// borrowed form of [`Prepared::scratch`].
    pub fn scratch_over<'b>(&self, buf: &'b mut [u64]) -> Result<Scratch<'b>, Refusal> {
        Scratch::over_for_program(buf, &self.program, self.n_rows).map_err(Refusal::Exec)
    }

    /// The row count this program was prepared for.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
}

/// Everything a terminal reads through: the existing DT64 witness.
pub struct Witness<'a, V: ?Sized, S> {
    pub binding: &'a LaneBinding,
    pub store: &'a BindStore,
    pub view: &'a V,
    pub frames: &'a FrameStore,
    pub source: &'a S,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Path {
    Name(String),
    Prop(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Assign(String, Path),
    Del(String),
    /// Abort the event when `path <cmp> value` holds.
    Abort {
        path: Path,
        cmp: Cmp,
        value: i32,
    },
    /// Abort the event when `path` equals (`eq`) or differs from a text
    /// literal. Only `==` and `!=`: a text column's codes have no order.
    AbortText {
        path: Path,
        eq: bool,
        text: String,
    },
}

/// A comparison of a lane value against an `i32` constant, signed. Each one
/// lowers to exactly one mask-risc `i32` predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn parse(op: &str) -> Option<Cmp> {
        Some(match op {
            "==" => Cmp::Eq,
            "!=" => Cmp::Ne,
            "<" => Cmp::Lt,
            "<=" => Cmp::Le,
            ">" => Cmp::Gt,
            ">=" => Cmp::Ge,
            _ => return None,
        })
    }

    /// Whether `a <cmp> b` holds, signed: the reference semantics of one abort.
    pub fn holds(self, a: i64, b: i64) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }

    /// The one mask-risc predicate this comparison is.
    fn pred(self, lane: u16, t: i32) -> Pred {
        match self {
            Cmp::Eq => Pred::EqI32 { lane, v: t },
            Cmp::Ne => Pred::NeI32 { lane, v: t },
            Cmp::Lt => Pred::LtI32 { lane, t },
            Cmp::Le => Pred::LeI32 { lane, t },
            Cmp::Gt => Pred::GtI32 { lane, t },
            Cmp::Ge => Pred::GeI32 { lane, t },
        }
    }
}

fn path(s: &str) -> Result<Path, Refusal> {
    let s = s.trim();
    let unknown = || Refusal::UnknownPath(s.to_string());
    let rest = s.strip_prefix('.').ok_or_else(unknown)?;
    let segs: Vec<&str> = rest.split('.').collect();
    let ident = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    match segs.as_slice() {
        [name] if ident(name) && *name != "properties" => Ok(Path::Name((*name).to_string())),
        ["properties", f, "value"] if ident(f) => Ok(Path::Prop((*f).to_string())),
        _ => Err(unknown()),
    }
}

/// The construct a value expression computes with, if it is not a path.
fn computes(rhs: &str) -> Option<String> {
    let rhs = rhs.trim();
    if let Some(i) = rhs.find('(') {
        return Some(format!("function {}", rhs[..i].trim()));
    }
    if rhs.starts_with('"') {
        return Some("string literal".into());
    }
    if rhs.parse::<i64>().is_ok() {
        return Some("integer literal".into());
    }
    if rhs.contains(['+', '-', '*', '/', '%']) {
        return Some("arithmetic".into());
    }
    None
}

/// Parse a recipe's source into its statements. [`Recipe::compile`] lowers
/// them; a reference interpreter may evaluate them independently.
pub fn parse(src: &str) -> Result<Vec<Stmt>, Refusal> {
    let mut out = Vec::new();
    for raw in src.split(['\n', ';']) {
        let s = raw.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if let Some(rest) = s.strip_prefix("if ") {
            let (cond, body) = rest
                .split_once('{')
                .ok_or_else(|| Refusal::Computes("if without a block".into()))?;
            let block = body.trim().strip_suffix('}').map(str::trim);
            if block != Some("abort") {
                return Err(Refusal::Computes("if body other than abort".into()));
            }
            if let Some((left, quoted)) = cond.split_once('"') {
                let text = quoted
                    .trim_end()
                    .strip_suffix('"')
                    .filter(|t| !t.contains(['"', '\\']))
                    .ok_or_else(|| Refusal::Computes(format!("text literal in {}", cond.trim())))?;
                let toks: Vec<&str> = left.split_whitespace().collect();
                let [p, op] = toks.as_slice() else {
                    return Err(Refusal::Computes(format!("condition {}", cond.trim())));
                };
                let eq = match *op {
                    "==" => true,
                    "!=" => false,
                    other if Cmp::parse(other).is_some() => {
                        return Err(Refusal::OrderOnText {
                            op: other.to_string(),
                        })
                    }
                    other => return Err(Refusal::Computes(format!("comparison {other}"))),
                };
                out.push(Stmt::AbortText {
                    path: path(p)?,
                    eq,
                    text: text.to_string(),
                });
                continue;
            }
            let toks: Vec<&str> = cond.split_whitespace().collect();
            let [p, op, v] = toks.as_slice() else {
                return Err(Refusal::Computes(format!("condition {}", cond.trim())));
            };
            let cmp =
                Cmp::parse(op).ok_or_else(|| Refusal::Computes(format!("comparison {op}")))?;
            let value = v
                .parse::<i32>()
                .map_err(|_| Refusal::Computes(format!("non-i32 comparison {v}")))?;
            out.push(Stmt::Abort {
                path: path(p)?,
                cmp,
                value,
            });
            continue;
        }
        if let Some(inner) = s.strip_prefix("del(").and_then(|x| x.strip_suffix(')')) {
            match path(inner)? {
                Path::Name(n) => out.push(Stmt::Del(n)),
                Path::Prop(_) => return Err(Refusal::Computes("del of a property".into())),
            }
            continue;
        }
        if let Some((lhs, rhs)) = s.split_once('=') {
            if let Some(c) = computes(rhs) {
                return Err(Refusal::Computes(c));
            }
            let Path::Name(name) = path(lhs)? else {
                return Err(Refusal::Computes("assignment into properties".into()));
            };
            out.push(Stmt::Assign(name, path(rhs)?));
            continue;
        }
        if let Some(c) = computes(s) {
            return Err(Refusal::Computes(c));
        }
        return Err(Refusal::Computes(format!("statement {s}")));
    }
    Ok(out)
}

/// The stored answer: no rows, only what to read and where.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// Every field the recipe reads (outputs and filters), by position.
    fields: WideFieldMask,
    /// The same fields, ascending: lane `i` of the program is `reads[i]`.
    reads: Vec<u8>,
    /// Rows dropped where `field <cmp> value`, one term per `abort`.
    drop: Vec<(u8, Cmp, i32)>,
    /// Rows dropped where a code field equals (`true`) or differs from a
    /// text literal.
    text: Vec<(u8, bool, String)>,
    /// The text literals' codes, set by [`Recipe::resolve_text`].
    codes: Option<Codes>,
    /// Output name -> field, in name order.
    out: Vec<(String, u8)>,
    /// The class the recipe was compiled for.
    class: ClassId,
    /// The predicate each read field named at compile time, in `reads`
    /// order: a witness must bind the same fields, not just the positions.
    idents: Vec<String>,
}

impl Recipe {
    /// Compile `src` against `class`'s fields in `view`. Every path is
    /// resolved here, once; nothing is resolved per row.
    pub fn compile<V: ClassView + ?Sized>(
        src: &str,
        view: &V,
        class: ClassId,
    ) -> Result<Self, Refusal> {
        let labels = view.fields(class);
        let by_label = |l: &str| {
            labels
                .iter()
                .position(|f| f.label == l)
                .and_then(|p| u8::try_from(p).ok())
        };
        let mut bound: BTreeMap<String, u8> = BTreeMap::new();
        if let Some(k) = by_label("key") {
            bound.insert("key".into(), k);
        }
        let mut assigned: BTreeMap<String, u8> = BTreeMap::new();
        let resolve = |bound: &BTreeMap<String, u8>, p: &Path| -> Result<u8, Refusal> {
            match p {
                Path::Name(n) => bound
                    .get(n)
                    .copied()
                    .ok_or_else(|| Refusal::UnknownPath(format!(".{n}"))),
                Path::Prop(f) => by_label(f)
                    .filter(|_| f != "key")
                    .ok_or_else(|| Refusal::UnknownPath(format!(".properties.{f}.value"))),
            }
        };
        let mut drop = Vec::new();
        let mut text_drop: Vec<(u8, bool, String)> = Vec::new();
        // The first field the program reads: the population when every output
        // is deleted again and nothing is filtered.
        let mut anchor: Option<u8> = None;
        for stmt in parse(src)? {
            match stmt {
                Stmt::Assign(name, p) => {
                    let field = resolve(&bound, &p)?;
                    anchor.get_or_insert(field);
                    bound.insert(name.clone(), field);
                    assigned.insert(name, field);
                }
                Stmt::Del(name) => {
                    if bound.remove(&name).is_none() {
                        return Err(Refusal::UnknownPath(format!(".{name}")));
                    }
                    assigned.remove(&name);
                }
                Stmt::Abort { path, cmp, value } => {
                    let field = resolve(&bound, &path)?;
                    anchor.get_or_insert(field);
                    drop.push((field, cmp, value));
                }
                Stmt::AbortText { path, eq, text } => {
                    let field = resolve(&bound, &path)?;
                    anchor.get_or_insert(field);
                    text_drop.push((field, eq, text));
                }
            }
        }
        let mut reads: Vec<u8> = assigned
            .values()
            .chain(drop.iter().map(|(f, _, _)| f))
            .chain(text_drop.iter().map(|(f, _, _)| f))
            .copied()
            .collect();
        reads.sort_unstable();
        reads.dedup();
        if reads.is_empty() {
            reads.extend(anchor);
        }
        if reads.len() > MAX_READS {
            return Err(Refusal::TooManyReads { reads: reads.len() });
        }
        let idents = reads
            .iter()
            .map(|&f| labels[usize::from(f)].predicate_iri.clone())
            .collect();
        Ok(Self {
            fields: WideFieldMask::from_positions(&reads),
            reads,
            drop,
            text: text_drop,
            codes: None,
            out: assigned.into_iter().collect(),
            class,
            idents,
        })
    }

    /// Resolve the text terms' literals to codes under `frame`: `code_of`
    /// answers the code a field (by predicate IRI) gives a string under that
    /// frame, `None` when the field has never seen it. Preparing a recipe
    /// with text terms needs this first, and a program prepared after it is
    /// refused once `frame` is no longer current, because a later write may
    /// give the string a code.
    ///
    /// A string with no code matches no row: `==` drops nothing, `!=` drops
    /// every row that holds a value.
    pub fn resolve_text(
        &self,
        frame: &Frame,
        code_of: impl Fn(&str, &str) -> Option<u32>,
    ) -> Result<Recipe, Refusal> {
        let mut of = Vec::with_capacity(self.text.len());
        for (field, _, text) in &self.text {
            let at = usize::from(self.lane(*field)?);
            of.push(code_of(&self.idents[at], text));
        }
        let mut r = self.clone();
        r.codes = Some(Codes {
            frame: frame.clone(),
            of,
        });
        Ok(r)
    }

    /// The drop terms, one per `abort`, in source order: rows where
    /// `field <cmp> value` holds are dropped. For an independent lowering
    /// (the r2il parity test), not for execution.
    pub fn terms(&self) -> &[(u8, Cmp, i32)] {
        &self.drop
    }

    /// The field an output name reads, if the recipe assigns it.
    pub fn output_field(&self, name: &str) -> Option<u8> {
        self.out.iter().find(|(n, _)| n == name).map(|(_, f)| *f)
    }

    /// Every field the recipe reads: what the caller binds through DT64.
    pub fn fields(&self) -> &WideFieldMask {
        &self.fields
    }

    /// The output names, in order.
    pub fn outputs(&self) -> Vec<&str> {
        self.out.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// The program lane of `field`. Every field a recipe names is in `reads`
    /// by construction (outputs and terms are what `reads` is built from), so
    /// the refusal is not reachable through `compile`; it stands in for a
    /// silent fallback to lane 0.
    fn lane(&self, field: u8) -> Result<u16, Refusal> {
        self.reads
            .iter()
            .position(|&f| f == field)
            .map(|at| at as u16)
            .ok_or(Refusal::NotARead { field })
    }

    /// The recipe's lanes, borrowed under ONE DT64 check, in `reads` order,
    /// into a fixed array: the accepting path allocates nothing. Refused
    /// before any borrow when the witness fails.
    /// `sum` is the field a sum terminal folds, if any: it and every field a
    /// drop term compares must be an `i32` lane. A field that is only
    /// projected may also be a `u32` lane (a text column's codes).
    fn lanes<'a, V, S>(&self, w: &Witness<'a, V, S>, sum: Option<u8>) -> Result<Lanes<'a>, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        if self.reads.is_empty() {
            return Err(Refusal::NoLanes);
        }
        // DT64 checks the witness against its own class and basis; this checks
        // that its class and fields are the ones the recipe was compiled for.
        if lance_graph_dt64::view_class(w.binding.class()) != self.class {
            return Err(Refusal::OtherClass);
        }
        let fields = w.view.fields(self.class);
        for (&f, iri) in self.reads.iter().zip(&self.idents) {
            if fields.get(usize::from(f)).map(|r| &r.predicate_iri) != Some(iri) {
                return Err(Refusal::OtherField { field: f });
            }
        }
        let mut lanes = [LaneRef::I32(&[]); MAX_READS];
        let mut presence: [&[u64]; MAX_READS] = [&[]; MAX_READS];
        let mut held = [false; MAX_READS];
        let bound = w
            .binding
            .borrow_all_present(w.store, w.view, w.frames, w.source)
            .map_err(Refusal::Binding)?;
        for item in bound {
            let (field, lane, plane) =
                match item {
                    Ok(triple) => triple,
                    Err(
                        LaneRefusal::Vanished { field } | LaneRefusal::PresenceUnbound { field },
                    ) if !self.reads.contains(&field) => continue,
                    Err(e) => return Err(Refusal::Binding(e)),
                };
            if let Some(i) = self.reads.iter().position(|&f| f == field) {
                let compared = sum == Some(field) || self.drop.iter().any(|t| t.0 == field);
                let coded = self.text.iter().any(|t| t.0 == field);
                match lane {
                    LaneRef::I32(_) if coded => return Err(Refusal::NotCode { field }),
                    LaneRef::I32(_) => {}
                    LaneRef::U32(_) if !compared => {}
                    _ => return Err(Refusal::NotI32 { field }),
                }
                lanes[i] = lane;
                presence[i] = plane;
                held[i] = true;
            }
        }
        if let Some(i) = held[..self.reads.len()].iter().position(|h| !h) {
            return Err(Refusal::NotBound {
                field: self.reads[i],
            });
        }
        Ok(Lanes {
            lanes,
            presence,
            len: self.reads.len(),
        })
    }

    /// Lower the population and `terminal` for a frame of `n_rows` rows.
    /// The cold step: it builds the program once. Run it with
    /// [`Recipe::run_into`] as often as the frame stays `n_rows` rows.
    fn prepare(
        &self,
        n_rows: usize,
        terminal: Terminal,
        sum: Option<u8>,
    ) -> Result<Prepared, Refusal> {
        Ok(Prepared {
            program: self.program(n_rows, terminal)?,
            n_rows,
            recipe: RecipeKey {
                class: self.class,
                idents: self.idents.clone(),
                drop: self.drop.clone(),
                text: self.text.clone(),
                out: self.out.clone(),
            },
            sum,
            frame: self.codes.as_ref().map(|c| c.frame.clone()),
        })
    }

    /// Prepare `Count`: how many rows the recipe keeps.
    pub fn prepare_count(&self, n_rows: usize) -> Result<Prepared, Refusal> {
        self.prepare(
            n_rows,
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
            None,
        )
    }

    /// Prepare `MaskedSumI32` of `output` over the kept rows.
    pub fn prepare_sum(&self, n_rows: usize, output: &str) -> Result<Prepared, Refusal> {
        let field = self
            .out
            .iter()
            .find(|(n, _)| n == output)
            .map(|(_, f)| *f)
            .ok_or_else(|| Refusal::UnknownPath(format!(".{output}")))?;
        self.prepare(
            n_rows,
            Terminal::MaskedSumI32 {
                mask: Operand::Scratch(0),
                lane: self.lane(field)?,
            },
            Some(field),
        )
    }

    /// Prepare the count of UNKNOWN rows: rows no term drops but some term
    /// cannot decide, because a field it reads is absent (presence 0, value
    /// 0). Neither kept nor dropped (EXO-294).
    pub fn prepare_unknown(&self, n_rows: usize) -> Result<Prepared, Refusal> {
        self.prepare(
            n_rows,
            Terminal::Count {
                mask: Operand::Scratch(UNKNOWN),
            },
            None,
        )
    }

    /// Prepare `Keep`: the kept rows' mask, written to the caller's
    /// [`Out::Mask`] of `⌈n/64⌉` words (mask-risc tiles its scratch, so a
    /// kept mask is never read back from a slot).
    pub fn prepare_keep(&self, n_rows: usize) -> Result<Prepared, Refusal> {
        self.prepare(
            n_rows,
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
            None,
        )
    }

    /// Run a prepared program over the witnessed lanes, into the caller's
    /// `scratch` (sized by [`Prepared::scratch`] or
    /// `Scratch::over_for_program`) and `out`. The accepting path: one DT64
    /// check, one execution, no allocation.
    ///
    /// Refused when `p` was prepared by another recipe or for another row
    /// count: its lane numbering or its every-row range would not describe
    /// these lanes.
    pub fn run_into<'a, V, S>(
        &self,
        p: &Prepared,
        w: &Witness<'a, V, S>,
        scratch: &mut Scratch<'_>,
        out: Out<'_>,
    ) -> Result<Value, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        self.exec(p, w, scratch, out).map(|(v, _)| v)
    }

    fn exec<'a, V, S>(
        &self,
        p: &Prepared,
        w: &Witness<'a, V, S>,
        scratch: &mut Scratch<'_>,
        out: Out<'_>,
    ) -> Result<(Value, Lanes<'a>), Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        let k = &p.recipe;
        if k.class != self.class
            || k.idents != self.idents
            || k.drop != self.drop
            || k.text != self.text
            || k.out != self.out
        {
            return Err(Refusal::NotThisRecipe);
        }
        if let Some(f) = &p.frame {
            LaneBinding::check_positions_frame(w.frames, f).map_err(Refusal::Binding)?;
        }
        let lanes = self.lanes(w, p.sum)?;
        let v = self.run_on(p, &lanes, scratch, out)?;
        Ok((v, lanes))
    }

    /// Run `p` over lanes already borrowed under one witness check.
    fn run_on(
        &self,
        p: &Prepared,
        lanes: &Lanes<'_>,
        scratch: &mut Scratch<'_>,
        out: Out<'_>,
    ) -> Result<Value, Refusal> {
        let n_rows = lanes.get()[0].len();
        if n_rows != p.n_rows {
            return Err(Refusal::RowsChanged {
                prepared: p.n_rows,
                now: n_rows,
            });
        }
        let need = words_for(n_rows);
        for (i, plane) in lanes.planes().iter().enumerate() {
            if plane.len() != need {
                return Err(Refusal::PresenceLength {
                    field: self.reads[i],
                    words: plane.len(),
                    need,
                });
            }
        }
        let planes = Planes {
            n_rows,
            masks: lanes.planes(),
            lanes: lanes.get(),
        };
        execute_into(&p.program, &planes, &Foreign::NONE, scratch, out).map_err(Refusal::Exec)
    }

    /// The witnessed row count, for preparing: one DT64 check, no fold.
    pub fn n_rows<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<usize, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        Ok(self.lanes(w, None)?.get()[0].len())
    }

    /// How many rows the recipe keeps: mask-risc `Count`, no row is built.
    /// Convenience: prepares and sizes scratch per call; the allocation-free
    /// form is [`Recipe::prepare_count`] + [`Recipe::run_into`].
    pub fn count<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<u64, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        let p = self.prepare_count(self.n_rows(w)?)?;
        let mut scratch = p.scratch()?;
        match self.run_into(&p, w, &mut scratch, Out::None)? {
            Value::Count(c) => Ok(c as u64),
            other => unreachable!("Count returned {other:?}"),
        }
    }

    /// How many rows are UNKNOWN: no term drops them, and some term cannot
    /// decide because a field it reads is absent. Convenience, as
    /// [`Recipe::count`].
    pub fn count_unknown<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<u64, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        let p = self.prepare_unknown(self.n_rows(w)?)?;
        let mut scratch = p.scratch()?;
        match self.run_into(&p, w, &mut scratch, Out::None)? {
            Value::Count(c) => Ok(c as u64),
            other => unreachable!("Count returned {other:?}"),
        }
    }

    /// The sum of an output over the kept rows: mask-risc `MaskedSumI32`.
    /// Convenience, as [`Recipe::count`].
    pub fn sum<'a, V, S>(&self, w: &Witness<'a, V, S>, output: &str) -> Result<i64, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        let p = self.prepare_sum(self.n_rows(w)?, output)?;
        let mut scratch = p.scratch()?;
        match self.run_into(&p, w, &mut scratch, Out::None)? {
            Value::SumI64(s) => Ok(s),
            other => unreachable!("MaskedSumI32 returned {other:?}"),
        }
    }

    /// The only place rows are produced, positionally: mask-risc `Keep` into
    /// one mask, then the kept positions in row order over the borrowed
    /// lanes. The mask is temporary and takes `⌈n_rows/64⌉` words even when
    /// no row survives; the returned position table is O(rows kept).
    pub fn materialize_rows<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<Rows<'a>, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        let p = self.prepare_keep(self.n_rows(w)?)?;
        let mut scratch = p.scratch()?;
        // Exactly sized: `resize` on an empty Vec rounds up to 4 words.
        let mut mask = vec![0u64; words_for(p.n_rows)];
        let (_, lanes) = self.exec(&p, w, &mut scratch, Out::Mask(&mut mask))?;
        // The rows a drop term could not decide: a second fold over the lanes
        // already borrowed, so the caller learns of them without another
        // witness check or a separate `count_unknown`.
        // Both programs are the same population program with another
        // terminal, so the keep program's scratch fits it.
        let u = self.prepare_unknown(p.n_rows)?;
        let unknown = match self.run_on(&u, &lanes, &mut scratch, Out::None)? {
            Value::Count(c) => c as u64,
            other => unreachable!("Count returned {other:?}"),
        };
        let frame = w.binding.frame().clone();
        let kept: u32 = mask.iter().map(|w| w.count_ones()).sum();
        let mut positions = Vec::with_capacity(kept as usize);
        for (w, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                positions.push((w * 64) as u32 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        self.rows(frame, lanes, positions, unknown)
    }

    /// The rows at `positions`, in list order, repeats kept: a key lookup,
    /// with the positions resolved by the source (for intake,
    /// `Table::live_positions_for_keys`) under `under`. The positions are
    /// admitted only while `under` is current, as
    /// `LaneBinding::borrow_all_at` admits them, and only inside the lanes.
    /// One frame check and one binding check, as in [`Recipe::run_into`].
    /// Liveness is the source's: positions from intake's
    /// `live_positions_for_keys` are live under the frame they came with, and a
    /// delete moves the frame; positions from anywhere else are not checked.
    ///
    /// A projection: a recipe with a drop term is refused
    /// ([`Refusal::FiltersAtPositions`]), because dropping a looked-up row
    /// would shift every later key's answer. No mask is built.
    pub fn rows_at<'a, V, S>(
        &self,
        w: &Witness<'a, V, S>,
        positions: &[u32],
        under: &Frame,
    ) -> Result<Rows<'a>, Refusal>
    where
        V: ClassView + ?Sized,
        S: PresenceSource<Lane<'a> = LaneRef<'a>>,
    {
        if !self.drop.is_empty() || !self.text.is_empty() {
            return Err(Refusal::FiltersAtPositions);
        }
        // The frame check `LaneBinding::borrow_all_at` makes, shared with it,
        // ahead of the one full binding check in `lanes`.
        LaneBinding::check_positions_frame(w.frames, under).map_err(Refusal::Binding)?;
        let lanes = self.lanes(w, None)?;
        let n_rows = lanes.get()[0].len();
        // `exec` checks these shapes on the fold path; a lookup reads the
        // lanes directly, so it checks them here before any position.
        let need = words_for(n_rows);
        for (i, (lane, plane)) in lanes.get().iter().zip(lanes.planes()).enumerate() {
            let field = self.reads[i];
            if lane.len() != n_rows {
                return Err(Refusal::LaneLength {
                    field,
                    len: lane.len(),
                    need: n_rows,
                });
            }
            if plane.len() != need {
                return Err(Refusal::PresenceLength {
                    field,
                    words: plane.len(),
                    need,
                });
            }
        }
        if let Some(&at) = positions.iter().find(|&&p| p as usize >= n_rows) {
            return Err(Refusal::PositionOutOfRange { at, n_rows });
        }
        // No drop term, so no row is undecided.
        self.rows(under.clone(), lanes, positions.to_vec(), 0)
    }

    fn rows<'a>(
        &self,
        frame: Frame,
        lanes: Lanes<'a>,
        positions: Vec<u32>,
        unknown: u64,
    ) -> Result<Rows<'a>, Refusal> {
        let mut cols = Vec::with_capacity(self.out.len());
        let mut planes = Vec::with_capacity(self.out.len());
        for (_, f) in &self.out {
            let at = usize::from(self.lane(*f)?);
            cols.push(lanes.get()[at]);
            planes.push(lanes.planes()[at]);
        }
        Ok(Rows {
            frame,
            names: self.out.iter().map(|(n, _)| n.clone()).collect(),
            lanes: cols,
            presence: planes,
            positions,
            unknown,
        })
    }

    /// The population program, three-valued (EXO-294). Each term is a
    /// compare `t` over its field, and the field's ABSENT rows are
    /// `a = (value == 0) & !presence`. A row is
    ///
    /// - KEPT (slot [`KEPT`]) when every term is decided and false:
    ///   `AND (!t & !a)`;
    /// - DROPPED (slot [`DROPPED`]) when some term is decided and true:
    ///   `OR (t & !a)`;
    /// - UNKNOWN (slot [`UNKNOWN`]) otherwise.
    ///
    /// This is Kleene's AND of `NOT term` over the aborts. `terminal` reads
    /// one of the three slots.
    fn program(&self, n_rows: usize, terminal: Terminal) -> Result<Program, Refusal> {
        let hi = u32::try_from(n_rows).map_err(|_| {
            Refusal::Exec(ExecError::RangeOutOfBounds {
                lo: 0,
                hi: u32::MAX,
                n_rows,
            })
        })?;
        let all = |dst| MaskOp::Pred {
            pred: Pred::Range { lo: 0, hi },
            under: None,
            dst,
        };
        let s = Operand::Scratch;
        let mut ops = vec![
            all(KEPT),
            MaskOp::Pred {
                pred: Pred::Range { lo: 0, hi: 0 },
                under: None,
                dst: DROPPED,
            },
        ];
        let mut terms = Vec::with_capacity(self.drop.len() + self.text.len());
        let term = |pred| {
            vec![MaskOp::Pred {
                pred,
                under: None,
                dst: TERM,
            }]
        };
        for &(field, cmp, t) in &self.drop {
            let lane = self.lane(field)?;
            terms.push((lane, term(cmp.pred(lane, t)), Pred::EqI32 { lane, v: 0 }));
        }
        if !self.text.is_empty() {
            let codes = self.codes.as_ref().ok_or(Refusal::TextUnresolved)?;
            for ((field, eq, _), code) in self.text.iter().zip(&codes.of) {
                let lane = self.lane(*field)?;
                // Code 0 is the row type's absence, never a stored string:
                // comparing with it is a presence test, decided at every
                // row, so no row is left unknown by it.
                if *code == Some(0) {
                    let presence = Operand::Plane(lane);
                    let ops = if *eq {
                        vec![
                            all(TERM),
                            MaskOp::AndNot {
                                a: s(TERM),
                                b: presence,
                                dst: TERM,
                            },
                        ]
                    } else {
                        vec![MaskOp::And {
                            a: presence,
                            b: presence,
                            dst: TERM,
                        }]
                    };
                    terms.push((lane, ops, Pred::Range { lo: 0, hi: 0 }));
                    continue;
                }
                let pred = match (code, eq) {
                    (Some(v), true) => Pred::EqU32 { lane, v: *v },
                    (Some(v), false) => Pred::NeU32 { lane, v: *v },
                    // No row holds a string the field never coded.
                    (None, true) => Pred::Range { lo: 0, hi: 0 },
                    (None, false) => Pred::Range { lo: 0, hi },
                };
                terms.push((lane, term(pred), Pred::EqU32 { lane, v: 0 }));
            }
        }
        for (lane, term, zero) in terms {
            ops.extend(term);
            ops.extend([
                MaskOp::Pred {
                    pred: zero,
                    under: None,
                    dst: ABSENT,
                },
                // absent = (value == 0) & !presence; plane `lane` is the
                // field's presence plane.
                MaskOp::AndNot {
                    a: s(ABSENT),
                    b: Operand::Plane(lane),
                    dst: ABSENT,
                },
                MaskOp::AndNot {
                    a: s(KEPT),
                    b: s(TERM),
                    dst: KEPT,
                },
                MaskOp::AndNot {
                    a: s(KEPT),
                    b: s(ABSENT),
                    dst: KEPT,
                },
                MaskOp::AndNot {
                    a: s(TERM),
                    b: s(ABSENT),
                    dst: TERM,
                },
                MaskOp::Or {
                    a: s(DROPPED),
                    b: s(TERM),
                    dst: DROPPED,
                },
            ]);
        }
        ops.extend([
            all(UNKNOWN),
            MaskOp::AndNot {
                a: s(UNKNOWN),
                b: s(KEPT),
                dst: UNKNOWN,
            },
            MaskOp::AndNot {
                a: s(UNKNOWN),
                b: s(DROPPED),
                dst: UNKNOWN,
            },
        ]);
        Ok(Program {
            ops,
            terminal,
            scratch_slots: u32::from(SLOTS),
        })
    }
}

/// Scratch slots of the population program.
const KEPT: u16 = 0;
const DROPPED: u16 = 1;
const TERM: u16 = 2;
const ABSENT: u16 = 3;
const UNKNOWN: u16 = 4;
const SLOTS: u16 = 5;

/// Positional rows: `names[c]` labels column `c`, and row `r` of column `c`
/// is `lanes[c][positions[r]]`, read in place from the borrowed lane.
#[derive(Debug)]
pub struct Rows<'a> {
    /// The seal the positions were taken under. Positions held past this
    /// call are re-read through `LaneBinding::borrow_all_at` under this
    /// frame, which refuses them once the source has moved.
    frame: Frame,
    /// Rows neither kept nor dropped: a drop term read an absent value.
    unknown: u64,
    names: Vec<String>,
    lanes: Vec<LaneRef<'a>>,
    /// Each column's presence plane, borrowed beside its lane.
    presence: Vec<&'a [u64]>,
    positions: Vec<u32>,
}

impl<'a> Rows<'a> {
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// How many rows are neither in these rows nor dropped: a drop term read
    /// an absent value and could not decide (EXO-294). Zero from
    /// [`Recipe::rows_at`], which takes no drop terms.
    pub fn unknown(&self) -> u64 {
        self.unknown
    }

    /// The column names, once.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The source position of each kept row (the `$i` table).
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Row `r`, column `c`, read from the lane: a signed value, or a text
    /// column's code (decoded through its code table by the consumer).
    pub fn get(&self, r: usize, c: usize) -> Datum {
        let p = self.positions[r] as usize;
        match self.lanes[c] {
            LaneRef::I32(l) => Datum::I32(l[p]),
            LaneRef::U32(l) => Datum::U32(l[p]),
            _ => unreachable!("lanes() admits i32 and u32 lanes only"),
        }
    }

    /// Whether row `r`, column `c` holds a value: false when it is absent
    /// (presence 0 and value 0, EXO-294). An absent cell reads 0 in
    /// [`Rows::get`].
    pub fn present(&self, r: usize, c: usize) -> bool {
        let p = self.positions[r] as usize;
        bit(self.presence[c], p) || !matches!(self.get(r, c), Datum::I32(0) | Datum::U32(0))
    }
}

/// One cell of [`Rows`]: the lane's value as stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Datum {
    I32(i32),
    /// A text column's code.
    U32(u32),
}

impl Datum {
    /// The signed value; `None` for a text code, which is not a number.
    pub fn as_i32(self) -> Option<i32> {
        match self {
            Datum::I32(v) => Some(v),
            Datum::U32(_) => None,
        }
    }

    /// A text column's code; `None` for a signed value.
    pub fn as_code(self) -> Option<u32> {
        match self {
            Datum::U32(c) => Some(c),
            Datum::I32(_) => None,
        }
    }
}

fn bit(plane: &[u64], row: usize) -> bool {
    plane[row / 64] >> (row % 64) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_abort_block_must_close_on_its_own_brace() {
        assert!(parse("if .key == 1 { abort }").is_ok());
        for src in ["if .key == 1 { abort", "if .key == 1 { abort }}"] {
            assert!(parse(src).is_err(), "{src:?} parsed");
        }
    }

    use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};

    struct View(Vec<FieldRef>);

    impl ClassView for View {
        fn fields(&self, _: ClassId) -> &[FieldRef] {
            &self.0
        }
        fn template(&self, _: ClassId) -> DisplayTemplate {
            DisplayTemplate::Detail
        }
        fn dolce_category_id(&self, _: ClassId) -> u8 {
            0
        }
    }

    fn view(n: usize) -> View {
        let mut labels = vec!["key".to_owned()];
        labels.extend((1..n).map(|i| format!("f{i}")));
        View(
            labels
                .iter()
                .map(|l| FieldRef::new(format!("t:{l}"), l.as_str()))
                .collect(),
        )
    }

    fn props(n: usize) -> String {
        (1..n)
            .map(|i| format!(".o{i} = .properties.f{i}.value"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn reading_more_fields_than_the_lane_array_holds_is_refused() {
        let v = view(MAX_READS + 2);
        assert!(Recipe::compile(&props(MAX_READS + 1), &v, 1).is_ok());
        assert_eq!(
            Recipe::compile(&props(MAX_READS + 2), &v, 1).unwrap_err(),
            Refusal::TooManyReads {
                reads: MAX_READS + 1
            }
        );
    }

    #[test]
    fn a_deleted_output_still_reads_the_field_that_anchors_the_population() {
        let v = view(3);
        let r = Recipe::compile(".id = .key\ndel(.id)", &v, 1).unwrap();
        assert!(r.outputs().is_empty());
        assert_eq!(r.fields().count(), 1);
        let kept = Recipe::compile(".id = .key", &v, 1).unwrap();
        assert_eq!(kept.fields().count(), 1);
    }
}
