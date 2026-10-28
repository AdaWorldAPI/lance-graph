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

/// A directed weighted graph over a label codebook: every node ID is stored
/// once, sorted, and an edge names its endpoints by rank in that order. One
/// edge per (source, target); duplicates keep the maximum, and between equal
/// weights the first read, as `graph_utils.add_edge` does.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    labels: Vec<String>,
    edges: Vec<(u32, u32, f64)>,
    offsets: Vec<u32>,
}

impl Graph {
    /// Parses NSL's TSV form: `src<TAB>dst[<TAB>weight]`, extra columns ignored.
    pub fn from_tsv(text: &str) -> Result<Self, NslError> {
        let mut code: HashMap<&str, u32> = HashMap::new();
        let mut first: Vec<&str> = Vec::new();
        let mut raw: Vec<(u32, u32, f64)> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let mut cols = line.split('\t');
            let (src, dst) = match (cols.next(), cols.next()) {
                (Some(s), Some(d)) => (s, d),
                _ => return Err(NslError::MalformedLine { line: i + 1 }),
            };
            let w = match cols.next() {
                Some(t) => parse_weight(t).ok_or_else(|| NslError::BadWeight {
                    line: i + 1,
                    text: t.to_string(),
                })?,
                None => 1.0,
            };
            if w.is_nan() {
                return Err(NslError::NanWeight {
                    src: src.into(),
                    dst: dst.into(),
                });
            }
            let mut ends = [0u32; 2];
            for (k, l) in [src, dst].into_iter().enumerate() {
                ends[k] = *code.entry(l).or_insert_with(|| {
                    first.push(l);
                    (first.len() - 1) as u32
                });
            }
            let (s, d) = (ends[0], ends[1]);
            raw.push((s, d, w));
        }
        let mut order: Vec<u32> = (0..first.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| first[a as usize].cmp(first[b as usize]));
        let mut rank = vec![0u32; first.len()];
        for (r, &o) in order.iter().enumerate() {
            rank[o as usize] = r as u32;
        }
        for e in &mut raw {
            e.0 = rank[e.0 as usize];
            e.1 = rank[e.1 as usize];
        }
        let labels = order
            .iter()
            .map(|&o| first[o as usize].to_string())
            .collect();
        Ok(Graph::from_coded(labels, raw))
    }

    /// Collapses coded edges in read order. The sort is stable, so for one
    /// pair the first read comes first and only a strictly larger weight
    /// replaces it.
    fn from_coded(labels: Vec<String>, mut raw: Vec<(u32, u32, f64)>) -> Self {
        raw.sort_by_key(|e| (e.0, e.1));
        let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
        for e in raw {
            match edges.last_mut() {
                Some(l) if (l.0, l.1) == (e.0, e.1) => {
                    if e.2 > l.2 {
                        l.2 = e.2;
                    }
                }
                _ => edges.push(e),
            }
        }
        let mut offsets = vec![0u32; labels.len() + 1];
        for e in &edges {
            offsets[e.0 as usize + 1] += 1;
        }
        for i in 0..labels.len() {
            offsets[i + 1] += offsets[i];
        }
        Graph {
            labels,
            edges,
            offsets,
        }
    }

    /// `graph_utils.add_undirected_edges`: every reversed edge, after the
    /// originals, so an existing edge wins a tie.
    pub fn symmetrize(&mut self) {
        let mut raw = self.edges.clone();
        raw.extend(self.edges.iter().map(|&(s, d, w)| (d, s, w)));
        *self = Graph::from_coded(std::mem::take(&mut self.labels), raw);
    }

    /// The code of a node ID: its rank among all labels.
    pub fn code(&self, label: &str) -> Option<u32> {
        self.labels
            .binary_search_by(|l| l.as_str().cmp(label))
            .ok()
            .map(|c| c as u32)
    }

    pub fn label(&self, code: u32) -> &str {
        &self.labels[code as usize]
    }

    fn out_edges(&self, code: u32) -> &[(u32, u32, f64)] {
        let c = code as usize;
        &self.edges[self.offsets[c] as usize..self.offsets[c + 1] as usize]
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

/// NSL order on `(weight, label code)`: weight descending, then ID
/// descending. Codes are ranks in sorted label order, so comparing codes is
/// comparing IDs. Python's tuple comparison treats `0.0` and `-0.0` as equal,
/// so `partial_cmp`, not `total_cmp`; NaN never reaches here.
fn nsl_order(a: &(f64, u32, u32), b: &(f64, u32, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .expect("NaN refused at parse")
        .then_with(|| b.1.cmp(&a.1))
}

/// The graph frozen against one feature table: per seed row, the neighbors
/// that have an example, already in NSL order (CSR). Built once; a query is a
/// borrowed slice.
#[derive(Debug, Clone, PartialEq)]
pub struct Adjacency {
    offsets: Vec<u32>,
    sel: Vec<Selected>,
}

impl Adjacency {
    /// Joins `graph` to `table` and sorts every seed's neighbors, as
    /// `pack_nbrs._join_examples` does before slicing to `max_nbrs`. Each
    /// label is looked up in the table once, not once per edge.
    pub fn build(graph: &Graph, table: &FeatureTable) -> Self {
        let row_of_code: Vec<Option<u32>> = graph.labels.iter().map(|l| table.row(l)).collect();
        let mut offsets = Vec::with_capacity(table.seeds + 1);
        let mut sel = Vec::new();
        let mut scratch: Vec<(f64, u32, u32)> = Vec::new();
        offsets.push(0);
        for seed in table.seed_rows() {
            scratch.clear();
            if let Some(c) = graph.code(table.id(seed)) {
                scratch.extend(
                    graph
                        .out_edges(c)
                        .iter()
                        .filter_map(|&(_, d, w)| row_of_code[d as usize].map(|row| (w, d, row))),
                );
            }
            scratch.sort_by(nsl_order);
            sel.extend(
                scratch
                    .iter()
                    .map(|&(weight, _, row)| Selected { row, weight }),
            );
            offsets.push(sel.len() as u32);
        }
        Adjacency { offsets, sel }
    }

    /// The first `cap` neighbors of a seed row (`None` = all), borrowed.
    pub fn neighbors(&self, seed: u32, cap: Option<usize>) -> &[Selected] {
        let (a, b) = (
            self.offsets[seed as usize] as usize,
            self.offsets[seed as usize + 1] as usize,
        );
        let all = &self.sel[a..b];
        &all[..cap.map_or(all.len(), |c| c.min(all.len()))]
    }

    pub fn seeds(&self) -> u32 {
        (self.offsets.len() - 1) as u32
    }
}

/// Every seed's batch, owned, for the compatibility serialization.
pub fn select(adj: &Adjacency, cap: Option<usize>) -> Vec<NeighborBatch> {
    (0..adj.seeds())
        .map(|seed| NeighborBatch {
            seed,
            selected: adj.neighbors(seed, cap).to_vec(),
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

pub mod loss;

#[cfg(test)]
mod eval;
#[cfg(test)]
mod tests;
