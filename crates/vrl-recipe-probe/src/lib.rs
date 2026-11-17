//! The VRL read/rename/filter subset, lowered onto the existing witnessed
//! path (D-VRL-RECIPE-0, lab only).
//!
//! Whatever the population size, the answer has one shape:
//! `identity | select {name = identity.key; identity.properties.f.value}`.
//! So the stored answer is a recipe, never rows: which fields to bind, the
//! `== v` keep and drop terms, and the output aliases. It adds no authority
//! and no kernel of its own; it is the front of what D-DR-SELECT-K-SV-1/2
//! (#2039, #2040) already ship:
//!
//! - **Witness.** Every terminal reads its lanes through one
//!   [`LaneBinding::borrow_all`]: DT64 checks the frame (issuer, revisions,
//!   reseal), route, basis, mapping, scope and row order before a lane is
//!   borrowed. A recipe run after the source's frame moved is refused there,
//!   never read against new lanes.
//! - **Fold.** The population and every terminal are one mask-risc
//!   [`Program`]: `Range` for every row, `EqI32` per term, `And` / `AndNot`
//!   to combine, then `Count`, `MaskedSumI32` or `Keep`. The computation is
//!   mask-risc's, on `ndarray::simd`.
//! - **Rows.** [`Recipe::materialize_rows`] is the only row producer, and it
//!   is positional ([`Rows`]): the names once, the borrowed lanes, and a `u32`
//!   position table. No string or event is built per row; [`Event`] exists
//!   only for the reference interpreter.
//!
//! The VRL subset, one statement per line (or `;`):
//!
//! - `.out = .key`, `.out = .other_out`, `.out = .properties.f.value`
//! - `del(.out)`
//! - `if .path == 3 { abort }`, `if .path != 3 { abort }`
//!
//! A name is a field label of the class's [`ClassView`]; `key` is the field
//! labelled `key`. Output semantics, stated because VRL mutates the whole
//! event: the names the program assigned and did not delete afterwards.
//! Paths are resolved once, at compile time, with VRL's in-place mutation:
//! after `.key = .properties.x.value`, a later `.key` reads `x`.
//!
//! Everything that computes a value is refused by name: function calls other
//! than `del` (`parse_json`, `now`, `to_string`, ...), arithmetic, string or
//! integer literals assigned as values, comparisons other than `==`/`!=`, and
//! an `if` whose body is not `abort`.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_dt64::dr::FrameStore;
use lance_graph_dt64::{BindStore, LaneBinding, LaneRefusal, LaneSource};
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
    /// A field the recipe reads is not an `i32` lane.
    NotI32 { field: u8 },
    /// The recipe reads no field, so it names no population.
    NoLanes,
    /// mask-risc refused the program or its operands.
    Exec(ExecError),
}

/// One event of the reference interpreter: names repeated per row, which is
/// exactly what the recipe's [`Rows`] avoid.
pub type Event = BTreeMap<String, i64>;

/// Everything a terminal reads through: the existing DT64 witness.
pub struct Witness<'a, V: ?Sized, S> {
    pub binding: &'a LaneBinding,
    pub store: &'a BindStore,
    pub view: &'a V,
    pub frames: &'a FrameStore,
    pub source: &'a S,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Path {
    Name(String),
    Prop(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Stmt {
    Assign(String, Path),
    Del(String),
    /// Abort the event when `path == value` (`eq`) or `path != value`.
    Abort {
        path: Path,
        eq: bool,
        value: i32,
    },
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

fn parse(src: &str) -> Result<Vec<Stmt>, Refusal> {
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
            if body.trim().trim_end_matches('}').trim() != "abort" {
                return Err(Refusal::Computes("if body other than abort".into()));
            }
            let toks: Vec<&str> = cond.split_whitespace().collect();
            let [p, op, v] = toks.as_slice() else {
                return Err(Refusal::Computes(format!("condition {}", cond.trim())));
            };
            let eq = match *op {
                "==" => true,
                "!=" => false,
                other => return Err(Refusal::Computes(format!("comparison {other}"))),
            };
            let value = v
                .parse::<i32>()
                .map_err(|_| Refusal::Computes(format!("non-i32 comparison {v}")))?;
            out.push(Stmt::Abort {
                path: path(p)?,
                eq,
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
    /// Rows kept only where `field == value`.
    keep_eq: Vec<(u8, i32)>,
    /// Rows dropped where `field == value`.
    drop_eq: Vec<(u8, i32)>,
    /// Output name -> field, in name order.
    out: Vec<(String, u8)>,
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
        let (mut keep_eq, mut drop_eq) = (Vec::new(), Vec::new());
        for stmt in parse(src)? {
            match stmt {
                Stmt::Assign(name, p) => {
                    let field = resolve(&bound, &p)?;
                    bound.insert(name.clone(), field);
                    assigned.insert(name, field);
                }
                Stmt::Del(name) => {
                    if bound.remove(&name).is_none() {
                        return Err(Refusal::UnknownPath(format!(".{name}")));
                    }
                    assigned.remove(&name);
                }
                Stmt::Abort { path, eq, value } => {
                    let field = resolve(&bound, &path)?;
                    if eq {
                        drop_eq.push((field, value));
                    } else {
                        keep_eq.push((field, value));
                    }
                }
            }
        }
        let mut reads: Vec<u8> = assigned
            .values()
            .chain(keep_eq.iter().map(|(f, _)| f))
            .chain(drop_eq.iter().map(|(f, _)| f))
            .copied()
            .collect();
        reads.sort_unstable();
        reads.dedup();
        Ok(Self {
            fields: WideFieldMask::from_positions(&reads),
            reads,
            keep_eq,
            drop_eq,
            out: assigned.into_iter().collect(),
        })
    }

    /// Every field the recipe reads: what the caller binds through DT64.
    pub fn fields(&self) -> &WideFieldMask {
        &self.fields
    }

    /// The output names, in order.
    pub fn outputs(&self) -> Vec<&str> {
        self.out.iter().map(|(n, _)| n.as_str()).collect()
    }

    fn lane(&self, field: u8) -> u16 {
        self.reads.iter().position(|&f| f == field).unwrap_or(0) as u16
    }

    /// The recipe's lanes, borrowed under ONE DT64 check, in `reads` order.
    /// Refused before any borrow when the witness fails.
    fn lanes<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<Vec<LaneRef<'a>>, Refusal>
    where
        V: ClassView + ?Sized,
        S: LaneSource<Lane<'a> = LaneRef<'a>>,
    {
        if self.reads.is_empty() {
            return Err(Refusal::NoLanes);
        }
        let mut lanes: Vec<Option<LaneRef<'a>>> = vec![None; self.reads.len()];
        let bound = w
            .binding
            .borrow_all(w.store, w.view, w.frames, w.source)
            .map_err(Refusal::Binding)?;
        for item in bound {
            let (field, lane) = match item {
                Ok(pair) => pair,
                Err(LaneRefusal::Vanished { field }) if !self.reads.contains(&field) => continue,
                Err(e) => return Err(Refusal::Binding(e)),
            };
            if let Some(i) = self.reads.iter().position(|&f| f == field) {
                if !matches!(lane, LaneRef::I32(_)) {
                    return Err(Refusal::NotI32 { field });
                }
                lanes[i] = Some(lane);
            }
        }
        lanes
            .into_iter()
            .zip(&self.reads)
            .map(|(l, &field)| l.ok_or(Refusal::NotBound { field }))
            .collect()
    }

    /// Run the population program with `terminal` over the witnessed lanes.
    /// `keep` receives a `Keep` terminal's mask (`⌈n/64⌉` words): mask-risc
    /// tiles its scratch on wide planes, so a kept mask is written to a caller
    /// buffer, never read back from a slot.
    fn run<'a, V, S>(
        &self,
        w: &Witness<'a, V, S>,
        terminal: Terminal,
        keep: Option<&mut Vec<u64>>,
    ) -> Result<(Value, Vec<LaneRef<'a>>), Refusal>
    where
        V: ClassView + ?Sized,
        S: LaneSource<Lane<'a> = LaneRef<'a>>,
    {
        let lanes = self.lanes(w)?;
        let n_rows = lanes[0].len();
        let program = self.program(n_rows, terminal)?;
        let mut scratch = Scratch::for_program(&program, n_rows).map_err(Refusal::Exec)?;
        let planes = Planes {
            n_rows,
            masks: &[],
            lanes: &lanes,
        };
        let out = match keep {
            Some(buf) => {
                // Exactly sized: `resize` on an empty Vec rounds up to 4 words.
                *buf = vec![0u64; words_for(n_rows)];
                Out::Mask(buf.as_mut_slice())
            }
            None => Out::None,
        };
        let v = execute_into(&program, &planes, &Foreign::NONE, &mut scratch, out)
            .map_err(Refusal::Exec)?;
        Ok((v, lanes))
    }

    /// How many rows the recipe keeps: mask-risc `Count`, no row is built.
    pub fn count<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<u64, Refusal>
    where
        V: ClassView + ?Sized,
        S: LaneSource<Lane<'a> = LaneRef<'a>>,
    {
        let count = Terminal::Count {
            mask: Operand::Scratch(0),
        };
        match self.run(w, count, None)?.0 {
            Value::Count(c) => Ok(c as u64),
            other => unreachable!("Count returned {other:?}"),
        }
    }

    /// The sum of an output over the kept rows: mask-risc `MaskedSumI32`.
    pub fn sum<'a, V, S>(&self, w: &Witness<'a, V, S>, output: &str) -> Result<i64, Refusal>
    where
        V: ClassView + ?Sized,
        S: LaneSource<Lane<'a> = LaneRef<'a>>,
    {
        let field = self
            .out
            .iter()
            .find(|(n, _)| n == output)
            .map(|(_, f)| *f)
            .ok_or_else(|| Refusal::UnknownPath(format!(".{output}")))?;
        let sum = Terminal::MaskedSumI32 {
            mask: Operand::Scratch(0),
            lane: self.lane(field),
        };
        match self.run(w, sum, None)?.0 {
            Value::SumI64(s) => Ok(s),
            other => unreachable!("MaskedSumI32 returned {other:?}"),
        }
    }

    /// The only place rows are produced, positionally: mask-risc `Keep` into
    /// one mask, then the kept positions in row order over the borrowed
    /// lanes. O(rows kept) for the position table; nothing else grows with
    /// rows.
    pub fn materialize_rows<'a, V, S>(&self, w: &Witness<'a, V, S>) -> Result<Rows<'a>, Refusal>
    where
        V: ClassView + ?Sized,
        S: LaneSource<Lane<'a> = LaneRef<'a>>,
    {
        let mut mask = Vec::new();
        let keep = Terminal::Keep {
            mask: Operand::Scratch(0),
        };
        let (_, lanes) = self.run(w, keep, Some(&mut mask))?;
        let kept: u32 = mask.iter().map(|w| w.count_ones()).sum();
        let mut positions = Vec::with_capacity(kept as usize);
        for (w, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                positions.push((w * 64) as u32 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        let lane_of = |field: u8| match lanes[usize::from(self.lane(field))] {
            LaneRef::I32(s) => s,
            _ => unreachable!("lanes() admits i32 lanes only"),
        };
        Ok(Rows {
            names: self.out.iter().map(|(n, _)| n.clone()).collect(),
            lanes: self.out.iter().map(|(_, f)| lane_of(*f)).collect(),
            positions,
        })
    }

    /// The population program: every row, then each term and-ed or
    /// and-not-ed in, then `terminal` over slot 0.
    fn program(&self, n_rows: usize, terminal: Terminal) -> Result<Program, Refusal> {
        let hi = u32::try_from(n_rows).map_err(|_| {
            Refusal::Exec(ExecError::RangeOutOfBounds {
                lo: 0,
                hi: u32::MAX,
                n_rows,
            })
        })?;
        let mut ops = vec![MaskOp::Pred {
            pred: Pred::Range { lo: 0, hi },
            under: None,
            dst: 0,
        }];
        for (terms, keep) in [(&self.keep_eq, true), (&self.drop_eq, false)] {
            for &(field, v) in terms {
                ops.push(MaskOp::Pred {
                    pred: Pred::EqI32 {
                        lane: self.lane(field),
                        v,
                    },
                    under: None,
                    dst: 1,
                });
                let (a, b) = (Operand::Scratch(0), Operand::Scratch(1));
                ops.push(if keep {
                    MaskOp::And { a, b, dst: 0 }
                } else {
                    MaskOp::AndNot { a, b, dst: 0 }
                });
            }
        }
        let scratch_slots = if ops.len() > 1 { 2 } else { 1 };
        Ok(Program {
            ops,
            terminal,
            scratch_slots,
        })
    }
}

/// Positional rows: `names[c]` labels column `c`, and row `r` of column `c`
/// is `lanes[c][positions[r]]`, read in place from the borrowed lane.
#[derive(Debug)]
pub struct Rows<'a> {
    names: Vec<String>,
    lanes: Vec<&'a [i32]>,
    positions: Vec<u32>,
}

impl<'a> Rows<'a> {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The column names, once.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The source position of each kept row (the `$i` table).
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Row `r`, column `c`, read from the lane.
    pub fn get(&self, r: usize, c: usize) -> i32 {
        self.lanes[c][self.positions[r] as usize]
    }
}

/// The naive reference: materialize every row of `lanes` (labelled by
/// `labels`, `key` among them) as a VRL event, run the program statement by
/// statement, and project each surviving event onto the names the program
/// assigned and did not delete. Shares only the parser with [`Recipe`];
/// evaluation is independent and reads no DT64 or mask-risc code.
pub fn interpret(src: &str, labels: &[&str], lanes: &[&[i32]]) -> Result<Vec<Event>, Refusal> {
    let prog = parse(src)?;
    let n = lanes.first().map_or(0, |l| l.len());
    let mut out = Vec::new();
    'rows: for r in 0..n {
        let mut top: BTreeMap<String, i64> = BTreeMap::new();
        let mut props: BTreeMap<&str, i64> = BTreeMap::new();
        for (l, lane) in labels.iter().zip(lanes) {
            if *l == "key" {
                top.insert("key".into(), i64::from(lane[r]));
            } else {
                props.insert(l, i64::from(lane[r]));
            }
        }
        let mut assigned: Vec<String> = Vec::new();
        let read = |top: &BTreeMap<String, i64>, p: &Path| -> Result<i64, Refusal> {
            match p {
                Path::Name(n) => top
                    .get(n)
                    .copied()
                    .ok_or_else(|| Refusal::UnknownPath(format!(".{n}"))),
                Path::Prop(f) => props
                    .get(f.as_str())
                    .copied()
                    .ok_or_else(|| Refusal::UnknownPath(format!(".properties.{f}.value"))),
            }
        };
        for stmt in &prog {
            match stmt {
                Stmt::Assign(name, p) => {
                    let v = read(&top, p)?;
                    top.insert(name.clone(), v);
                    if !assigned.contains(name) {
                        assigned.push(name.clone());
                    }
                }
                Stmt::Del(name) => {
                    if top.remove(name).is_none() {
                        return Err(Refusal::UnknownPath(format!(".{name}")));
                    }
                    assigned.retain(|a| a != name);
                }
                Stmt::Abort { path, eq, value } => {
                    if (read(&top, path)? == i64::from(*value)) == *eq {
                        continue 'rows;
                    }
                }
            }
        }
        out.push(assigned.iter().map(|a| (a.clone(), top[a])).collect());
    }
    Ok(out)
}
