//! A stored late-materialization recipe, compiled from the read/rename/filter
//! subset of Vector's VRL (D-VRL-RECIPE-0, lab only).
//!
//! Whatever the population size, the answer has one shape:
//! `identity | select {name = identity.key; identity.properties.f.value}`.
//! So the stored answer is the recipe, never the rows: the store version it
//! was compiled against, the population terms, the participating fields as a
//! [`WideFieldMask`], and the output aliases. A terminal evaluates the
//! population into one mask of `n.div_ceil(64)` words and reads lanes in
//! place at its set bits. Only [`Recipe::materialize_rows`] produces rows,
//! and they are positional ([`Rows`]): the names once, one borrowed lane per
//! output, and a `u32` position table. No string and no event is built per
//! row; the probe's [`Event`] type exists only for the reference interpreter.
//!
//! Every fold is the masking polyfill in `ndarray::simd`
//! (`eq_i32_to_mask`, `mask_and_assign`, `mask_andnot_assign`,
//! `masked_sum_i32`, `popcount_batch_u64`); the probe adds no kernel of its
//! own. VRL is edge syntax: its statements resolve onto those existing
//! words, so neither a `dt64::vrl` module nor an ndarray `simd_vrl` module is
//! needed (D-VRL-RECIPE-0).
//!
//! The VRL subset, one statement per line (or `;`):
//!
//! - `.out = .key`, `.out = .other_out`, `.out = .properties.f.value`
//! - `del(.out)`
//! - `if .path == 3 { abort }`, `if .path != 3 { abort }`
//!
//! Output semantics, stated because VRL mutates the whole event: the result
//! is the set of names the program assigned and did not delete afterwards.
//! A path is resolved once, at compile time, so `.a = .key` followed by
//! `.key = .properties.x.value` and `.b = .key` binds `b` to `x`, as VRL's
//! in-place mutation does.
//!
//! Everything that computes a value is refused by name: function calls other
//! than `del` (`parse_json`, `now`, `to_string`, ...), arithmetic, string or
//! integer literals assigned as values, comparisons other than `==`/`!=`, and
//! an `if` whose body is not `abort`. Those are computations, not reads, and
//! a recipe must not absorb them.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::WideFieldMask;
use ndarray::simd::{
    eq_i32_to_mask, mask_and_assign, mask_andnot_assign, masked_sum_i32, popcount_batch_u64,
};

/// Why a program or a terminal is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The construct computes a value instead of reading one.
    Computes(String),
    /// A path that is neither a bound name nor `.properties.<field>.value`.
    UnknownPath(String),
    /// The recipe was compiled against another store version.
    Stale { compiled: u64, store: u64 },
    /// More fields than a `u8` lane position can name.
    TooManyFields,
}

/// A columnar store: lane 0 is `key`, lane `i > 0` is property `fields[i]`.
pub struct Store {
    version: u64,
    n: usize,
    fields: Vec<String>,
    lanes: Vec<Vec<i32>>,
}

impl Store {
    /// `props` are `(name, values)`; every lane has `keys.len()` rows.
    pub fn new(keys: Vec<i32>, props: Vec<(&str, Vec<i32>)>) -> Self {
        let n = keys.len();
        let mut fields = vec!["key".to_string()];
        let mut lanes = vec![keys];
        for (name, values) in props {
            assert_eq!(values.len(), n, "lane {name} has the wrong length");
            fields.push(name.to_string());
            lanes.push(values);
        }
        Self {
            version: 1,
            n,
            fields,
            lanes,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// A write: every recipe compiled before it is refused afterwards.
    pub fn bump(&mut self) {
        self.version += 1;
    }

    fn field(&self, name: &str) -> Option<u8> {
        self.fields
            .iter()
            .skip(1)
            .position(|f| f == name)
            .and_then(|i| u8::try_from(i + 1).ok())
    }
}

/// One event of the reference interpreter: names repeated per row, which is
/// exactly what the recipe's [`Rows`] avoid.
pub type Event = BTreeMap<String, i64>;

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
    version: u64,
    /// Rows kept only where `lane == value`.
    keep_eq: Vec<(u8, i32)>,
    /// Rows dropped where `lane == value`.
    drop_eq: Vec<(u8, i32)>,
    /// The participating fields, by lane position.
    view: WideFieldMask,
    /// Output name -> lane, in name order.
    out: Vec<(String, u8)>,
}

impl Recipe {
    /// Compile `src` against `store`'s schema. Every path is resolved here,
    /// once; nothing is resolved per row.
    pub fn compile(src: &str, store: &Store) -> Result<Self, Refusal> {
        if store.fields.len() > 256 {
            return Err(Refusal::TooManyFields);
        }
        let mut bound: BTreeMap<String, u8> = BTreeMap::from([("key".to_string(), 0)]);
        let mut assigned: BTreeMap<String, u8> = BTreeMap::new();
        let resolve = |bound: &BTreeMap<String, u8>, p: &Path| -> Result<u8, Refusal> {
            match p {
                Path::Name(n) => bound
                    .get(n)
                    .copied()
                    .ok_or_else(|| Refusal::UnknownPath(format!(".{n}"))),
                Path::Prop(f) => store
                    .field(f)
                    .ok_or_else(|| Refusal::UnknownPath(format!(".properties.{f}.value"))),
            }
        };
        let (mut keep_eq, mut drop_eq) = (Vec::new(), Vec::new());
        for stmt in parse(src)? {
            match stmt {
                Stmt::Assign(name, p) => {
                    let lane = resolve(&bound, &p)?;
                    bound.insert(name.clone(), lane);
                    assigned.insert(name, lane);
                }
                Stmt::Del(name) => {
                    if bound.remove(&name).is_none() {
                        return Err(Refusal::UnknownPath(format!(".{name}")));
                    }
                    assigned.remove(&name);
                }
                Stmt::Abort { path, eq, value } => {
                    let lane = resolve(&bound, &path)?;
                    if eq {
                        drop_eq.push((lane, value));
                    } else {
                        keep_eq.push((lane, value));
                    }
                }
            }
        }
        let positions: Vec<u8> = assigned.values().copied().collect();
        Ok(Self {
            version: store.version,
            keep_eq,
            drop_eq,
            view: WideFieldMask::from_positions(&positions),
            out: assigned.into_iter().collect(),
        })
    }

    /// The participating fields.
    pub fn view(&self) -> &WideFieldMask {
        &self.view
    }

    /// The output names, in order.
    pub fn outputs(&self) -> Vec<&str> {
        self.out.iter().map(|(n, _)| n.as_str()).collect()
    }

    fn check(&self, store: &Store) -> Result<(), Refusal> {
        if self.version != store.version {
            return Err(Refusal::Stale {
                compiled: self.version,
                store: store.version,
            });
        }
        Ok(())
    }

    /// The kept rows as a mask of `n.div_ceil(64)` words, built by the
    /// polyfill: each term packs `lane == value` into a scratch mask and is
    /// and-ed or and-not-ed in. Two allocations at most, both `⌈n/64⌉` words;
    /// none per row.
    fn mask(&self, store: &Store) -> Result<Vec<u64>, Refusal> {
        self.check(store)?;
        let n = store.n;
        let words = n.div_ceil(64);
        let mut m = vec![u64::MAX; words];
        if !n.is_multiple_of(64) {
            if let Some(last) = m.last_mut() {
                *last = (1u64 << (n % 64)) - 1;
            }
        }
        if self.keep_eq.is_empty() && self.drop_eq.is_empty() {
            return Ok(m);
        }
        let mut scratch = vec![0u64; words];
        for &(lane, value) in &self.keep_eq {
            eq_i32_to_mask(&store.lanes[lane as usize], value, &mut scratch);
            mask_and_assign(&mut m, &scratch);
        }
        for &(lane, value) in &self.drop_eq {
            eq_i32_to_mask(&store.lanes[lane as usize], value, &mut scratch);
            mask_andnot_assign(&mut m, &scratch);
        }
        Ok(m)
    }

    /// How many rows the recipe keeps: a popcount, no row is built.
    pub fn count(&self, store: &Store) -> Result<u64, Refusal> {
        Ok(popcount_batch_u64(&self.mask(store)?))
    }

    /// The sum of an output over the kept rows, read in place.
    pub fn sum(&self, store: &Store, output: &str) -> Result<i64, Refusal> {
        let lane = self
            .out
            .iter()
            .find(|(n, _)| n == output)
            .map(|(_, l)| *l)
            .ok_or_else(|| Refusal::UnknownPath(format!(".{output}")))?;
        let m = self.mask(store)?;
        Ok(masked_sum_i32(&store.lanes[lane as usize], &m))
    }

    /// The only place rows are produced, positionally: the output names once,
    /// one borrowed lane per output, and the kept positions in row order.
    /// O(rows kept) for the position table; nothing else grows with rows.
    pub fn materialize_rows<'s>(&'s self, store: &'s Store) -> Result<Rows<'s>, Refusal> {
        let m = self.mask(store)?;
        let mut positions = Vec::with_capacity(popcount_batch_u64(&m) as usize);
        for (w, &word) in m.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                positions.push((w * 64) as u32 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        Ok(Rows {
            names: self.out.iter().map(|(n, _)| n.as_str()).collect(),
            lanes: self
                .out
                .iter()
                .map(|(_, l)| store.lanes[*l as usize].as_slice())
                .collect(),
            positions,
        })
    }
}

/// Positional rows: `names[c]` labels column `c`, and row `r` of column `c`
/// is `lanes[c][positions[r]]`, read in place.
#[derive(Debug)]
pub struct Rows<'s> {
    names: Vec<&'s str>,
    lanes: Vec<&'s [i32]>,
    positions: Vec<u32>,
}

impl<'s> Rows<'s> {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The column names, once.
    pub fn names(&self) -> &[&'s str] {
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

/// The naive reference: materialize every row as a VRL event
/// (`{key, properties: {f: {value}}}`), run the program statement by
/// statement, and project each surviving event onto the names the program
/// assigned and did not delete. Shares only the parser with [`Recipe`];
/// evaluation is independent.
pub fn interpret(src: &str, store: &Store) -> Result<Vec<Event>, Refusal> {
    let prog = parse(src)?;
    let mut out = Vec::new();
    'rows: for r in 0..store.n {
        let mut top: BTreeMap<String, i64> =
            BTreeMap::from([("key".to_string(), i64::from(store.lanes[0][r]))]);
        let props: BTreeMap<&str, i64> = store
            .fields
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, f)| (f.as_str(), i64::from(store.lanes[i][r])))
            .collect();
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
