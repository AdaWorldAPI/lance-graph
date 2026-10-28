//! NSL neighbor selection (H1 / W1 of the NSL research plan).
//!
//! Reproduces what NSL's `pack_nbrs` chooses for each seed, without copying
//! neighbor features: every example is stored once in a [`FeatureTable`] and a
//! [`NeighborBatch`] refers to rows by ordinal. [`compat_lines`] turns batches
//! back into the per-neighbor records NSL would have written, which is how the
//! tests compare against the frozen oracle in `data/nsl_golden.txt`.
//!
//! Semantics kept from NSL (`tools/graph_utils.py`, `tools/pack_nbrs.py`):
//! - edges are directed; a missing weight is 1.0; duplicates keep the maximum;
//! - only neighbors that have an example are selected;
//! - order is (weight, neighbor ID) descending, comparing the f64 the TSV parse
//!   produced, with `0.0 == -0.0`; the emitted weight is that f64 cast to f32;
//! - a seed's own (labeled) example wins over an unlabeled one with the same ID;
//! - `cap = Some(0)` selects nothing, `None` is unlimited.
//!
//! Departures, each refused rather than approximated:
//! - a NaN weight: NSL's order then depends on input order;
//! - weight text Rust cannot parse exactly as Python's `float` does (e.g. `1_0`);
//! - a feature read whose `index_seal` differs from the one the table was
//!   loaded under ([`FocusLease`]).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Why an input or a read was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum NslError {
    /// A TSV line has fewer than two columns.
    MalformedLine { line: usize },
    /// The weight column is not a number in the accepted grammar.
    BadWeight { line: usize, text: String },
    /// A NaN weight on `src -> dst`.
    NanWeight { src: String, dst: String },
    /// The same example ID appears twice in one table.
    DuplicateExample { id: String },
    /// A read named a different seal than the table was loaded under.
    StaleSeal {
        loaded: SourceRef,
        requested: SourceRef,
    },
}

/// Identity of the data a feature table was loaded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRef {
    pub dataset_id: u64,
    pub version: u64,
    pub index_seal: u64,
}

/// One example: the three features the fixtures carry.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub id: String,
    pub tag: String,
    pub x: f32,
}

/// Every example stored once, seeds first. Row ordinals are stable.
#[derive(Debug, Default)]
pub struct FeatureTable {
    ids: Vec<String>,
    tags: Vec<String>,
    xs: Vec<f32>,
    seeds: usize,
    row_of: HashMap<String, u32>,
}

impl FeatureTable {
    /// Builds the table. A seed and an unlabeled example with the same ID
    /// keep the seed's features, as `pack_nbrs.lookup_ex` does.
    pub fn new(seeds: &[Example], unlabeled: &[Example]) -> Result<Self, NslError> {
        let mut t = FeatureTable::default();
        for e in seeds {
            if t.row_of.contains_key(&e.id) {
                return Err(NslError::DuplicateExample { id: e.id.clone() });
            }
            t.push(e);
        }
        t.seeds = t.ids.len();
        for e in unlabeled {
            match t.row_of.get(&e.id) {
                Some(&r) if (r as usize) < t.seeds => {}
                Some(_) => return Err(NslError::DuplicateExample { id: e.id.clone() }),
                None => t.push(e),
            }
        }
        Ok(t)
    }

    fn push(&mut self, e: &Example) {
        self.row_of.insert(e.id.clone(), self.ids.len() as u32);
        self.ids.push(e.id.clone());
        self.tags.push(e.tag.clone());
        self.xs.push(e.x);
    }

    pub fn rows(&self) -> usize {
        self.ids.len()
    }
    pub fn seed_rows(&self) -> std::ops::Range<u32> {
        0..self.seeds as u32
    }
    pub fn row(&self, id: &str) -> Option<u32> {
        self.row_of.get(id).copied()
    }
    pub fn id(&self, row: u32) -> &str {
        &self.ids[row as usize]
    }
    pub fn tag(&self, row: u32) -> &str {
        &self.tags[row as usize]
    }
    pub fn x(&self, row: u32) -> f32 {
        self.xs[row as usize]
    }
}

/// A directed weighted graph keyed by node ID, duplicates collapsed to the
/// maximum weight.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    out: HashMap<String, HashMap<String, f64>>,
}

impl Graph {
    /// Parses NSL's TSV form: `src<TAB>dst[<TAB>weight]`, extra columns ignored.
    pub fn from_tsv(text: &str) -> Result<Self, NslError> {
        let mut g = Graph::default();
        for (i, line) in text.lines().enumerate() {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 2 {
                return Err(NslError::MalformedLine { line: i + 1 });
            }
            let w = match cols.get(2) {
                Some(t) => parse_weight(t).ok_or_else(|| NslError::BadWeight {
                    line: i + 1,
                    text: (*t).to_string(),
                })?,
                None => 1.0,
            };
            g.add_edge(cols[0], cols[1], w)?;
        }
        Ok(g)
    }

    /// `graph_utils.add_edge`: keep the larger weight.
    pub fn add_edge(&mut self, src: &str, dst: &str, w: f64) -> Result<(), NslError> {
        if w.is_nan() {
            return Err(NslError::NanWeight {
                src: src.into(),
                dst: dst.into(),
            });
        }
        let t = self.out.entry(src.to_string()).or_default();
        match t.get_mut(dst) {
            Some(old) if w > *old => *old = w,
            Some(_) => {}
            None => {
                t.insert(dst.to_string(), w);
            }
        }
        Ok(())
    }

    /// `graph_utils.add_undirected_edges`: add every reversed edge.
    pub fn symmetrize(&mut self) {
        let edges: Vec<(String, String, f64)> = self
            .out
            .iter()
            .flat_map(|(s, t)| t.iter().map(move |(d, w)| (s.clone(), d.clone(), *w)))
            .collect();
        for (s, d, w) in edges {
            self.add_edge(&d, &s, w)
                .expect("weights were checked on insert");
        }
    }

    fn out_edges(&self, src: &str) -> impl Iterator<Item = (&str, f64)> {
        self.out
            .get(src)
            .into_iter()
            .flat_map(|t| t.iter().map(|(d, w)| (d.as_str(), *w)))
    }
}

/// Python `float()` for the inputs both accept identically. Underscores,
/// which Python accepts and Rust does not, are refused.
pub fn parse_weight(text: &str) -> Option<f64> {
    let t = text.trim();
    if t.contains('_') {
        return None;
    }
    t.parse::<f64>().ok()
}

/// One selected neighbor: a row reference, not a copy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selected {
    pub row: u32,
    pub weight: f64,
}

/// The neighbors chosen for one seed, in NSL order.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborBatch {
    pub seed: u32,
    pub selected: Vec<Selected>,
}

/// NSL order: weight descending, then neighbor ID descending. Python's tuple
/// comparison treats `0.0` and `-0.0` as equal, so `partial_cmp`, not
/// `total_cmp`; NaN never reaches here.
fn nsl_order(table: &FeatureTable, a: &Selected, b: &Selected) -> Ordering {
    b.weight
        .partial_cmp(&a.weight)
        .expect("NaN refused at insert")
        .then_with(|| table.id(b.row).cmp(table.id(a.row)))
}

/// Selects every seed's neighbors, as `pack_nbrs._join_examples` does.
pub fn select(graph: &Graph, table: &FeatureTable, cap: Option<usize>) -> Vec<NeighborBatch> {
    table
        .seed_rows()
        .map(|seed| {
            let mut sel: Vec<Selected> = graph
                .out_edges(table.id(seed))
                .filter_map(|(d, weight)| table.row(d).map(|row| Selected { row, weight }))
                .collect();
            sel.sort_by(|a, b| nsl_order(table, a, b));
            if let Some(c) = cap {
                sel.truncate(c);
            }
            NeighborBatch {
                seed,
                selected: sel,
            }
        })
        .collect()
}

fn f32_hex(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

/// The records NSL's packer would write, in the golden-file line format.
pub fn compat_lines(fixture: &str, table: &FeatureTable, batches: &[NeighborBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let mut keys: Vec<String> =
            vec!["NL_num_nbrs".into(), "id".into(), "tag".into(), "x".into()];
        for i in 0..b.selected.len() {
            for f in ["id", "tag", "weight", "x"] {
                keys.push(format!("NL_nbr_{i}_{f}"));
            }
        }
        keys.sort();
        out.push(format!(
            "seed {fixture} {} {} {}",
            table.id(b.seed),
            b.selected.len(),
            keys.join(",")
        ));
        for (i, s) in b.selected.iter().enumerate() {
            out.push(format!(
                "nbr {fixture} {} {i} {} {} {} {}",
                table.id(b.seed),
                table.id(s.row),
                f32_hex(s.weight as f32),
                table.tag(s.row),
                f32_hex(table.x(s.row))
            ));
        }
    }
    out
}

/// A feature table loaded once per source and shared by every reader.
///
/// The first `get` loads it and records the [`SourceRef`]; later calls return
/// the same table. A call naming a different seal for the same table is
/// refused before any feature is read.
#[derive(Default)]
pub struct FocusLease {
    slot: OnceLock<(SourceRef, Arc<FeatureTable>)>,
}

impl FocusLease {
    pub fn get(
        &self,
        source: SourceRef,
        load: impl FnOnce() -> FeatureTable,
    ) -> Result<Arc<FeatureTable>, NslError> {
        let (loaded, table) = self.slot.get_or_init(|| (source, Arc::new(load())));
        if *loaded != source {
            return Err(NslError::StaleSeal {
                loaded: *loaded,
                requested: source,
            });
        }
        Ok(Arc::clone(table))
    }
}

#[cfg(test)]
mod tests;
