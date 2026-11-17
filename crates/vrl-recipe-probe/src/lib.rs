//! A stored late-materialization recipe, compiled from the read/rename/filter
//! subset of Vector's VRL (D-VRL-RECIPE-0, lab only).
//!
//! Whatever the population size, the answer has one shape:
//! `identity | select {name = identity.key; identity.properties.f.value}`.
//! So the stored answer is the recipe, never the rows: the store version it
//! was compiled against, the population terms, the participating fields as a
//! [`WideFieldMask`], and the output aliases. A terminal evaluates the
//! population into one mask of `n.div_ceil(64)` words and reads lanes in
//! place at its set bits. Only [`Recipe::materialize_rows`] writes rows out,
//! and it costs O(rows kept).
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
    lanes: Vec<Vec<i64>>,
}

impl Store {
    /// `props` are `(name, values)`; every lane has `keys.len()` rows.
    pub fn new(keys: Vec<i64>, props: Vec<(&str, Vec<i64>)>) -> Self {
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

/// One materialized output event.
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
        value: i64,
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
                .parse::<i64>()
                .map_err(|_| Refusal::Computes(format!("non-integer comparison {v}")))?;
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
    keep_eq: Vec<(u8, i64)>,
    /// Rows dropped where `lane == value`.
    drop_eq: Vec<(u8, i64)>,
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

    /// The kept rows as a mask of `n.div_ceil(64)` words: the one allocation
    /// a fold terminal makes. Each term is folded word by word over its lane.
    fn mask(&self, store: &Store) -> Result<Vec<u64>, Refusal> {
        self.check(store)?;
        let n = store.n;
        let mut m = vec![u64::MAX; n.div_ceil(64)];
        if !n.is_multiple_of(64) {
            if let Some(last) = m.last_mut() {
                *last = (1u64 << (n % 64)) - 1;
            }
        }
        for (w, word) in m.iter_mut().enumerate() {
            let base = w * 64;
            let end = (base + 64).min(n);
            let eq_bits = |lane: u8, value: i64| -> u64 {
                let l = &store.lanes[lane as usize][base..end];
                l.iter()
                    .enumerate()
                    .fold(0u64, |acc, (i, &x)| acc | (u64::from(x == value) << i))
            };
            for &(lane, value) in &self.keep_eq {
                *word &= eq_bits(lane, value);
            }
            for &(lane, value) in &self.drop_eq {
                *word &= !eq_bits(lane, value);
            }
        }
        Ok(m)
    }

    /// How many rows the recipe keeps: a popcount, no row is built.
    pub fn count(&self, store: &Store) -> Result<u64, Refusal> {
        Ok(self
            .mask(store)?
            .iter()
            .map(|w| u64::from(w.count_ones()))
            .sum())
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
        let values = &store.lanes[lane as usize];
        let mut total = 0i64;
        for (w, &word) in m.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                total = total.wrapping_add(values[w * 64 + bits.trailing_zeros() as usize]);
                bits &= bits - 1;
            }
        }
        Ok(total)
    }

    /// The only place rows are written: one event per kept row, in row order,
    /// each holding exactly the outputs. O(rows kept).
    pub fn materialize_rows(&self, store: &Store) -> Result<Vec<Event>, Refusal> {
        let m = self.mask(store)?;
        let mut rows = Vec::new();
        for (w, &word) in m.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let r = w * 64 + bits.trailing_zeros() as usize;
                rows.push(
                    self.out
                        .iter()
                        .map(|(name, lane)| (name.clone(), store.lanes[*lane as usize][r]))
                        .collect(),
                );
                bits &= bits - 1;
            }
        }
        Ok(rows)
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
            BTreeMap::from([("key".to_string(), store.lanes[0][r])]);
        let props: BTreeMap<&str, i64> = store
            .fields
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, f)| (f.as_str(), store.lanes[i][r]))
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
                    if (read(&top, path)? == *value) == *eq {
                        continue 'rows;
                    }
                }
            }
        }
        out.push(assigned.iter().map(|a| (a.clone(), top[a])).collect());
    }
    Ok(out)
}
