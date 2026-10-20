//! The Schnellverschluss: the one module that names `mork` and `pathmap`.
//!
//! In: [`JoinInput`] (borrowed Lance rows + generations). Out: [`Answers`]
//! (owned `[u32; 3]` rows) or a typed [`Refusal`]. Nothing upstream-typed is
//! returned, so a backend can be replaced or deleted by editing this file and
//! `Cargo.toml` only.
//!
//! Both upstream indexes are COPIES. [`bind`] encodes the borrowed rows into a
//! PathMap byte trie (directly, or through MORK's S-expression parser) and the
//! index owns those bytes. That is the translation cost the probe measures; it
//! is not zero-copy, and nothing here claims it is. A stale copy is caught by
//! the generation witness ([`Refusal::Stale`]), not by a lifetime.

use std::time::{Duration, Instant};

use mork::space::Space;
use pathmap::PathMap;
use pathmap::zipper::{ZipperIteration, ZipperPath, ZipperValues};

use crate::join::{JoinExec, JoinTable};
use crate::{Answers, JoinInput, Semantics};

/// A physical implementation of the fixed query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// MORK's MM2 `exec` over its own `Space`. Which join MORK runs is fixed at
    /// compile time by its `leapfrog` feature (this crate's `leapfrog` feature);
    /// see [`mork_join`].
    Mork,
    /// PathMap byte trie, read through read-only zippers.
    PathMapZipper,
    /// The versioned relationship table in [`crate::join`]: both inputs laid
    /// out by the shared variable's slot, intersected by one mask-risc `And`.
    /// A PROBE of the co-addressed producer §7 of the plan asks for, not an
    /// admitted production path.
    Native,
}

/// Why a backend did not answer. Every refusal is returned, never papered over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The backend stores a set (`PathMap<()>`, a MORK `Space`), so it cannot
    /// report multiplicities. Asking it for [`Semantics::Bag`] is refused rather
    /// than answered with every count collapsed to 1.
    SetOnly(Backend),
    /// No admitted native join exists for this input shape. Reading `S` at
    /// positions chosen by `R`'s values is the shape the 2026-10-16 no-gather
    /// ruling forbids. [`Backend::Native`] answers only after
    /// [`crate::join::JoinTable::prepare`] has laid both inputs out at one
    /// coordinate; this variant stays for shapes that table does not cover.
    NoAdmittedNativeJoin,
    /// The index was built from other generations than the caller's current
    /// views.
    Stale {
        bound: (u64, u64),
        current: (u64, u64),
    },
    /// The upstream index rejected the encoded input.
    Upstream(String),
}

/// Which join the compiled MORK runs for a conjunctive body.
pub const fn mork_join() -> &'static str {
    if cfg!(feature = "leapfrog") {
        "leapfrog"
    } else {
        "product-zipper"
    }
}

// Proof that the leapfrog module is compiled when the feature is on: this
// fails to build if `mork::leapfrog` is absent.
#[cfg(feature = "leapfrog")]
#[allow(dead_code)]
type LeapfrogCursor<'a> =
    mork::leapfrog::SubtermCursor<pathmap::zipper::ReadZipperUntracked<'a, 'static, ()>>;

/// Answers what a backend can do before anything is built.
pub fn capability(backend: Backend, semantics: Semantics) -> Result<(), Refusal> {
    match (backend, semantics) {
        // Native keeps multiplicities, so it admits both semantics.
        (Backend::Native, _) => Ok(()),
        (_, Semantics::Bag) => Err(Refusal::SetOnly(backend)),
        (_, Semantics::Set) => Ok(()),
    }
}

enum Index {
    Mork(Box<Space>),
    Path(PathMap<()>),
    Native(Box<(JoinTable, JoinExec)>, Semantics),
}

/// An upstream index built from one pair of generations.
pub struct BoundIndex {
    backend: Backend,
    generations: (u64, u64),
    index: Index,
    /// Rows handed to the upstream builder (duplicates included).
    pub rows_in: usize,
    /// Wall time of the encoding and index build.
    pub build_time: Duration,
}

/// Builds the backend's index from the borrowed rows. This is where the
/// translation cost is paid.
pub fn bind(
    backend: Backend,
    input: &JoinInput<'_>,
    semantics: Semantics,
) -> Result<BoundIndex, Refusal> {
    capability(backend, semantics)?;
    let start = Instant::now();
    let index = match backend {
        Backend::Mork => Index::Mork(Box::new(mork_space(input)?)),
        Backend::PathMapZipper => Index::Path(path_trie(input)),
        Backend::Native => {
            let table = JoinTable::prepare(input.r, input.s);
            let exec = JoinExec::new(&table);
            Index::Native(Box::new((table, exec)), semantics)
        }
    };
    Ok(BoundIndex {
        backend,
        generations: input.generations(),
        index,
        rows_in: input.r.rows.len() + input.s.rows.len(),
        build_time: start.elapsed(),
    })
}

impl BoundIndex {
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Answers the query against the views the caller holds now. Refuses if
    /// they are not the generations this index was built from.
    pub fn answers(&mut self, current: &JoinInput<'_>) -> Result<Answers, Refusal> {
        let now = current.generations();
        if now != self.generations {
            return Err(Refusal::Stale {
                bound: self.generations,
                current: now,
            });
        }
        match &mut self.index {
            Index::Mork(space) => mork_answers(space),
            Index::Path(trie) => Ok(path_answers(trie)),
            Index::Native(te, semantics) => {
                let (table, exec) = &mut **te;
                exec.answers(table, now, None, *semantics)
                    .map_err(|e| Refusal::Upstream(format!("{e:?}")))
            }
        }
    }
}

// ---- MORK ------------------------------------------------------------------

/// Symbols are `n<id>`, so the text round trip is exact.
fn mork_space(input: &JoinInput<'_>) -> Result<Space, Refusal> {
    let mut text = Vec::with_capacity(16 * (input.r.rows.len() + input.s.rows.len()));
    for (name, rows) in [("R", input.r.rows), ("S", input.s.rows)] {
        for &[a, b] in rows {
            text.extend_from_slice(format!("({name} n{a} n{b})\n").as_bytes());
        }
    }
    let mut space = Space::new();
    space.add_all_sexpr(&text).map_err(Refusal::Upstream)?;
    Ok(space)
}

const MORK_QUERY: &[u8] = b"(exec 0 (, (R $x $y) (S $y $z)) (, (Q $x $y $z)))\n";

/// MM2 `exec` writes its answers into the space as `Q` facts. They are read
/// back and removed again, so the index holds only `R` and `S` between calls.
fn mork_answers(space: &mut Space) -> Result<Answers, Refusal> {
    space.add_all_sexpr(MORK_QUERY).map_err(Refusal::Upstream)?;
    space.metta_calculus(1);
    let mut dump = Vec::new();
    space.dump_all_sexpr(&mut dump).map_err(Refusal::Upstream)?;
    let mut out = Answers::default();
    let mut produced = Vec::new();
    for line in dump.split(|&b| b == b'\n') {
        if let Some(rest) = line.strip_prefix(b"(Q ") {
            let row = parse_q(rest)
                .ok_or_else(|| Refusal::Upstream(String::from_utf8_lossy(line).into_owned()))?;
            out.rows.insert(row, 1);
            produced.extend_from_slice(line);
            produced.push(b'\n');
        }
    }
    if !produced.is_empty() {
        space
            .remove_all_sexpr(&produced)
            .map_err(Refusal::Upstream)?;
    }
    Ok(out)
}

fn parse_q(rest: &[u8]) -> Option<[u32; 3]> {
    let body = rest.strip_suffix(b")")?;
    let mut ids = body.split(|&b| b == b' ').map(|tok| {
        std::str::from_utf8(tok.strip_prefix(b"n")?)
            .ok()?
            .parse::<u32>()
            .ok()
    });
    let row = [ids.next()??, ids.next()??, ids.next()??];
    ids.next().is_none().then_some(row)
}

// ---- PathMap ---------------------------------------------------------------

const R_TAG: u8 = b'R';
const S_TAG: u8 = b'S';

/// `R(x, y)` is keyed `R ‖ y ‖ x` and `S(y, z)` is keyed `S ‖ y ‖ z`
/// (big-endian), so both are ordered by the shared variable.
fn path_trie(input: &JoinInput<'_>) -> PathMap<()> {
    let mut trie = PathMap::new();
    let mut key = [0u8; 9];
    for (tag, rows, swap) in [(R_TAG, input.r.rows, true), (S_TAG, input.s.rows, false)] {
        key[0] = tag;
        for &[a, b] in rows {
            let (first, second) = if swap { (b, a) } else { (a, b) };
            key[1..5].copy_from_slice(&first.to_be_bytes());
            key[5..9].copy_from_slice(&second.to_be_bytes());
            trie.insert(key, ());
        }
    }
    trie
}

/// For every `R ‖ y ‖ x`, walks the `S ‖ y` subtrie. Oracle code: the seek into
/// `S` at a position chosen by `R`'s value is exactly what lance-graph's
/// admitted path may not do.
fn path_answers(trie: &PathMap<()>) -> Answers {
    let mut out = Answers::default();
    let mut r = trie.read_zipper_at_path([R_TAG]);
    while r.to_next_val() {
        let p = r.path();
        let y = u32::from_be_bytes(p[0..4].try_into().unwrap());
        let x = u32::from_be_bytes(p[4..8].try_into().unwrap());
        let mut prefix = [S_TAG; 5];
        prefix[1..5].copy_from_slice(&y.to_be_bytes());
        let mut s = trie.read_zipper_at_path(prefix);
        while s.to_next_val() {
            let z = u32::from_be_bytes(s.path()[0..4].try_into().unwrap());
            out.rows.insert([x, y, z], 1);
        }
    }
    out
}

// ---- PathMap as a relationship index (D-RMAP-1A focus probe) ----------------

/// A PathMap trie over `(p, s, o)` facts, keyed `p ‖ s ‖ o` big-endian, each
/// leaf holding the canonical row number of its fact.
///
/// It is an INDEX, not the facts: the canonical rows stay in the caller's
/// slice and the leaf is a row reference. The key bytes are a copy of `p`, `s`
/// and `o` (12 bytes per fact), which [`SpogTrie::key_bytes`] reports; that is
/// the cost of a byte trie and the reason the native sorted run in
/// [`crate::rmap`] carries row references only.
pub struct SpogTrie(PathMap<u32>);

fn spog_key(p: u32, s: u32, o: u32) -> [u8; 12] {
    let mut k = [0u8; 12];
    k[0..4].copy_from_slice(&p.to_be_bytes());
    k[4..8].copy_from_slice(&s.to_be_bytes());
    k[8..12].copy_from_slice(&o.to_be_bytes());
    k
}

impl SpogTrie {
    /// Index `facts` (`[s, p, o]` rows). Row `i` becomes the leaf of its key.
    pub fn build(facts: &[[u32; 3]]) -> Self {
        let mut t = PathMap::new();
        for (row, &[s, p, o]) in facts.iter().enumerate() {
            t.insert(spog_key(p, s, o), row as u32);
        }
        SpogTrie(t)
    }

    /// Number of indexed facts (a full walk: PathMap's `val_count` is O(n)).
    pub fn len(&self) -> usize {
        self.0.val_count()
    }

    /// Whether the index holds no fact.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lower bound on what the index stores: its key bytes. Node and
    /// allocator overhead are not counted (MORK installs a global allocator,
    /// so this probe cannot count allocations; see ISS-MORK-LIBRARY-GLOBAL-ALLOCATOR).
    pub fn key_bytes(&self) -> usize {
        self.len() * 12
    }

    /// Prefix seek to `p ‖ s`, then every `(o, row)` under it in `o` order.
    pub fn objects(&self, p: u32, s: u32, mut f: impl FnMut(u32, u32)) {
        let mut prefix = [0u8; 8];
        prefix[0..4].copy_from_slice(&p.to_be_bytes());
        prefix[4..8].copy_from_slice(&s.to_be_bytes());
        let mut z = self.0.read_zipper_at_path(prefix);
        while z.to_next_val() {
            let o = u32::from_be_bytes(z.path()[0..4].try_into().unwrap());
            let row = *z.val().expect("to_next_val stops on a value");
            f(o, row);
        }
    }

    /// The facts whose `(p, s)` is in `pairs`, as a trie that shares the
    /// subtries it keeps with `self` (PathMap's `restrict`): the focus's
    /// outgoing relationships, cut out without re-inserting a fact.
    pub fn restrict_to(&self, pairs: impl IntoIterator<Item = (u32, u32)>) -> SpogTrie {
        let mut keep = PathMap::new();
        for (p, s) in pairs {
            let mut k = [0u8; 8];
            k[0..4].copy_from_slice(&p.to_be_bytes());
            k[4..8].copy_from_slice(&s.to_be_bytes());
            keep.insert(k, 0u32);
        }
        SpogTrie(self.0.restrict(&keep))
    }
}

/// A PathMap posting index keyed `term ‖ doc` big-endian (inverted index).
pub struct PostingTrie(PathMap<()>);

impl PostingTrie {
    /// Index `(term, doc)` pairs.
    pub fn build(pairs: &[[u32; 2]]) -> Self {
        let mut t = PathMap::new();
        for &[term, doc] in pairs {
            let mut k = [0u8; 8];
            k[0..4].copy_from_slice(&term.to_be_bytes());
            k[4..8].copy_from_slice(&doc.to_be_bytes());
            t.insert(k, ());
        }
        PostingTrie(t)
    }

    /// Every doc of `term`, in doc order (one prefix seek, then a walk).
    pub fn docs(&self, term: u32, mut f: impl FnMut(u32)) {
        let mut z = self.0.read_zipper_at_path(term.to_be_bytes());
        while z.to_next_val() {
            f(u32::from_be_bytes(z.path()[0..4].try_into().unwrap()));
        }
    }

    /// Docs of both terms, merged from two ordered zipper walks.
    pub fn intersect(&self, a: u32, b: u32) -> Vec<u32> {
        let mut za = self.0.read_zipper_at_path(a.to_be_bytes());
        let mut zb = self.0.read_zipper_at_path(b.to_be_bytes());
        let read = |p: &[u8]| u32::from_be_bytes(p[0..4].try_into().unwrap());
        let mut out = Vec::new();
        let (mut ha, mut hb) = (za.to_next_val(), zb.to_next_val());
        while ha && hb {
            let (x, y) = (read(za.path()), read(zb.path()));
            if x == y {
                out.push(x);
                ha = za.to_next_val();
                hb = zb.to_next_val();
            } else if x < y {
                ha = za.to_next_val();
            } else {
                hb = zb.to_next_val();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RelationView;

    fn lines(space: &Space) -> usize {
        let mut dump = Vec::new();
        space.dump_all_sexpr(&mut dump).unwrap();
        dump.split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .count()
    }

    /// The MORK index holds exactly the `R` and `S` facts before and after a
    /// query: the `Q` answers `exec` writes are removed again.
    #[test]
    fn mork_query_leaves_only_the_input_facts() {
        let r = [[1, 10], [2, 10]];
        let s = [[10, 5]];
        let inp = JoinInput {
            r: RelationView {
                rows: &r,
                generation: 1,
            },
            s: RelationView {
                rows: &s,
                generation: 1,
            },
        };
        let mut idx = bind(Backend::Mork, &inp, Semantics::Set).unwrap();
        let Index::Mork(space) = &idx.index else {
            panic!("mork index")
        };
        assert_eq!(lines(space), 3);
        assert_eq!(idx.answers(&inp).unwrap().rows.len(), 2);
        let Index::Mork(space) = &idx.index else {
            panic!("mork index")
        };
        assert_eq!(lines(space), 3);
    }
}
