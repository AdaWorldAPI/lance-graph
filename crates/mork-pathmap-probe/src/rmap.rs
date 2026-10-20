//! D-RMAP-1A probe: a focus-bound RelationshipMap over existing ordered indexes.
//!
//! STATUS: PROBE. Plan: `.claude/plans/d-rmap-separate-tables-thought-palettes-v1.md`
//! (§8 is this probe's record). Nothing here is a store, an executor or a
//! contract type; every type is a stand-in for a D-RMAP-1A name.
//!
//! The question: can a scattered global neighbourhood be cut out of an
//! existing ordered index once, bound to 256 local focus slots, and then
//! evaluated many times by the native mask executor, instead of re-walking the
//! index on every evaluation?
//!
//! Layers, and which D-RMAP-1A table each stands for:
//!
//! - **Canonical facts** (`&[Fact]`, RelationFacts). Never copied, never
//!   reordered. Every prepared form refers to a fact by its row number.
//! - **Ordered indexes** (RelationOrder): [`SortedRun`], a permutation of row
//!   numbers sorted by `(p, s, o)` (and its `(p, o, s)` twin), and the PathMap
//!   trie [`SpogTrie`]. Both answer "`(p, s)` → objects" by prefix seek.
//! - **Focus** ([`Focus`], FocusBinding): slot `0..256` → global identity, with
//!   a generation. Identities are never rewritten to fit the focus.
//! - **Prepared relationship operands**: [`Csr`] (row offsets per slot, local
//!   targets, the canonical row of every edge) and [`Tile`] (a 256 × 256
//!   Boolean plane per predicate). Both are DERIVED: they hold slot ordinals
//!   and row references, never fact payload.
//! - **Native execution**: [`Tile`] hops run as `lance-graph-mask-risc`
//!   programs over planes that share one coordinate space — the tile's address
//!   is `(target << 8) | source`, the `u8:u8` rail of a 64 Ki-row table, and
//!   the frontier is broadcast into the same space. No op reads at a position
//!   taken from another lane's value (no `MaskOp::Gather`; the 2026-10-16
//!   ruling).
//!
//! Where the 2026-10-16 ruling bites, and why it does not here: turning a
//! global target into a slot ordinal IS an address translation. It happens
//! once, in preparation ([`Csr::prepare`]), counted, outside the fold — the
//! form H14c §1 rule 6 allows ("any address translation is explicit, counted,
//! and outside the direct-position path"). Whether a preparation stage may do
//! this as a matter of policy is the open legal question D-RMAP-1 §6 names; the
//! probe does not settle it, it measures it.
//!
//! Semantics are Boolean reachability inside the focus. A target outside the
//! focus is returned in [`ScopedReach::escaped`], never folded into
//! `reached` and never dropped; [`ScopedReach::global`] refuses while any
//! escape exists, because a path through an out-of-focus node can return into
//! the focus. Bag counts and path counts are not answered here.

use std::collections::BTreeSet;

use lance_graph_mask_risc::{
    Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, Terminal, Value,
    execute, execute_into, words_for,
};

pub use crate::boundary::SpogTrie;

/// One canonical fact `[s, p, o]`.
pub type Fact = [u32; 3];

/// Focus width: local slots `0..256`.
pub const SLOTS: usize = 256;
/// Words of a 256-bit slot plane.
pub const SLOT_WORDS: usize = SLOTS / 64;
/// Rows of a 256 × 256 tile: the `u8:u8` rail.
pub const TILE_ROWS: usize = SLOTS * SLOTS;
/// Words of a tile plane (8 KiB).
pub const TILE_WORDS: usize = TILE_ROWS / 64;

// ---- validity --------------------------------------------------------------

/// What a prepared operand was built against. A probe-local stand-in for the
/// D-RMAP-1A §7 binding witness: `facts` ↔ the facts' `DurableRef`, `seal` ↔
/// the index seal, `focus` ↔ the focus generation, `scope` ↔ the
/// authorization receipt. A thought's identity is checked separately
/// ([`ThoughtDelta::thought`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub facts: u64,
    pub seal: u64,
    pub focus: u64,
    pub scope: u64,
}

/// Why a prepared operand may not be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The facts it was built from changed.
    FactsChanged,
    /// The index it navigated was resealed (append, compaction, rebuild).
    IndexResealed,
    /// A focus slot now names a different identity.
    FocusChanged,
    /// The scope or permissions changed.
    ScopeChanged,
    /// A thought delta belongs to another thought incarnation.
    ThoughtMismatch,
    /// Two masks over different row orderings were combined.
    OrderingMismatch,
    /// The answer depends on out-of-focus nodes ([`ScopedReach::global`]).
    IncompleteFocus,
}

impl Witness {
    /// Refuse when any component differs; the first difference is reported.
    pub fn check(&self, current: &Witness) -> Result<(), Refusal> {
        if self.facts != current.facts {
            return Err(Refusal::FactsChanged);
        }
        if self.seal != current.seal {
            return Err(Refusal::IndexResealed);
        }
        if self.focus != current.focus {
            return Err(Refusal::FocusChanged);
        }
        if self.scope != current.scope {
            return Err(Refusal::ScopeChanged);
        }
        Ok(())
    }
}

// ---- ordered index ---------------------------------------------------------

/// A sorted run over canonical facts: row numbers ordered by `(p, s, o)`, or
/// by `(p, o, s)` for the reverse direction. Holds `u32` row references only;
/// the facts stay in the borrowed slice.
pub struct SortedRun<'f> {
    facts: &'f [Fact],
    perm: Vec<u32>,
    reverse: bool,
    /// Index seal: which build this is. A rebuild gets a new seal.
    pub seal: u64,
}

impl<'f> SortedRun<'f> {
    fn key(f: &Fact, reverse: bool) -> (u32, u32, u32) {
        if reverse {
            (f[1], f[2], f[0])
        } else {
            (f[1], f[0], f[2])
        }
    }

    /// Sort row numbers by `(p, s, o)` (`reverse = false`) or `(p, o, s)`.
    pub fn build(facts: &'f [Fact], reverse: bool, seal: u64) -> Self {
        let mut perm: Vec<u32> = (0..facts.len() as u32).collect();
        perm.sort_unstable_by_key(|&r| Self::key(&facts[r as usize], reverse));
        SortedRun {
            facts,
            perm,
            reverse,
            seal,
        }
    }

    /// The rows whose `(p, first)` matches: a borrowed range of the run.
    pub fn range(&self, p: u32, first: u32) -> &[u32] {
        let k = |r: &u32| {
            let (kp, k1, _) = Self::key(&self.facts[*r as usize], self.reverse);
            (kp, k1)
        };
        let lo = self.perm.partition_point(|r| k(r) < (p, first));
        let hi = self.perm.partition_point(|r| k(r) <= (p, first));
        &self.perm[lo..hi]
    }

    /// `(other end, row)` for every fact under `(p, first)`.
    pub fn neighbours(&self, p: u32, first: u32, mut f: impl FnMut(u32, u32)) {
        let other = if self.reverse { 0 } else { 2 };
        for &r in self.range(p, first) {
            f(self.facts[r as usize][other], r);
        }
    }

    /// Index bytes: the permutation (the facts are not counted; they are not
    /// the index's).
    pub fn bytes(&self) -> usize {
        self.perm.len() * 4
    }
}

/// What the focus discovery and the reference oracle navigate.
pub trait Navigate {
    /// Every `(o, row)` with fact `(s, p, o)`.
    fn objects(&self, p: u32, s: u32, f: &mut dyn FnMut(u32, u32));
}

impl Navigate for SortedRun<'_> {
    fn objects(&self, p: u32, s: u32, f: &mut dyn FnMut(u32, u32)) {
        debug_assert!(!self.reverse);
        self.neighbours(p, s, f);
    }
}

impl Navigate for SpogTrie {
    fn objects(&self, p: u32, s: u32, f: &mut dyn FnMut(u32, u32)) {
        SpogTrie::objects(self, p, s, f);
    }
}

// ---- focus -----------------------------------------------------------------

/// FocusBinding: slot → global identity, and the sorted reverse lookup.
#[derive(Clone, Debug)]
pub struct Focus {
    ids: Vec<u32>,
    by_id: Vec<(u32, u8)>,
    /// Bumped by every change of a slot's identity.
    pub generation: u64,
}

impl Focus {
    /// Bind up to 256 distinct identities, in slot order.
    pub fn bind(ids: &[u32], generation: u64) -> Self {
        assert!(ids.len() <= SLOTS, "a focus holds at most 256 identities");
        let mut by_id: Vec<(u32, u8)> =
            ids.iter().enumerate().map(|(s, &g)| (g, s as u8)).collect();
        by_id.sort_unstable();
        assert!(
            by_id.windows(2).all(|w| w[0].0 != w[1].0),
            "focus identities must be distinct"
        );
        Focus {
            ids: ids.to_vec(),
            by_id,
            generation,
        }
    }

    /// Slot of a global identity, if focused.
    pub fn slot_of(&self, id: u32) -> Option<u8> {
        self.by_id
            .binary_search_by_key(&id, |&(g, _)| g)
            .ok()
            .map(|i| self.by_id[i].1)
    }

    /// Global identity of a slot.
    pub fn id(&self, slot: u8) -> u32 {
        self.ids[slot as usize]
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Replace the identity of the given slots; returns a new generation.
    pub fn replace(&self, changes: &[(u8, u32)]) -> Focus {
        let mut ids = self.ids.clone();
        for &(s, g) in changes {
            ids[s as usize] = g;
        }
        Focus::bind(&ids, self.generation + 1)
    }

    /// Bytes of the binding (slot table + reverse lookup).
    pub fn bytes(&self) -> usize {
        self.ids.len() * 4 + self.by_id.len() * 8
    }
}

/// Discover a focus: breadth-first from `seeds` over the index, through
/// `preds`, until `cap` identities are bound. The order of discovery is the
/// slot order.
pub fn discover(index: &dyn Navigate, seeds: &[u32], preds: &[u32], cap: usize) -> Focus {
    let cap = cap.min(SLOTS);
    let mut seen = BTreeSet::new();
    let mut order = Vec::with_capacity(cap);
    let mut queue = std::collections::VecDeque::new();
    for &s in seeds {
        if order.len() < cap && seen.insert(s) {
            order.push(s);
            queue.push_back(s);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &p in preds {
            index.objects(p, s, &mut |o, _| {
                if order.len() < cap && seen.insert(o) {
                    order.push(o);
                    queue.push_back(o);
                }
            });
        }
        if order.len() >= cap {
            break;
        }
    }
    Focus::bind(&order, 1)
}

// ---- scoped reachability ---------------------------------------------------

/// A 256-bit set of focus slots.
pub type SlotSet = [u64; SLOT_WORDS];

pub fn slot_set(slots: impl IntoIterator<Item = u8>) -> SlotSet {
    let mut s = [0u64; SLOT_WORDS];
    for x in slots {
        s[(x >> 6) as usize] |= 1 << (x & 63);
    }
    s
}

pub fn slots_of(set: &SlotSet) -> Vec<u8> {
    (0..SLOTS)
        .filter(|&i| set[i >> 6] >> (i & 63) & 1 == 1)
        .map(|i| i as u8)
        .collect()
}

/// Boolean reachability within the focus, plus every out-of-focus target an
/// in-focus node reached within the hop budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedReach {
    /// Slots reachable in `0..=hops` steps (seeds included).
    pub reached: SlotSet,
    /// Global identities outside the focus that a reached node points at
    /// within the budget. Exact, never truncated.
    pub escaped: BTreeSet<u32>,
}

impl ScopedReach {
    /// The answer to the GLOBAL question, which equals the scoped one only if
    /// nothing escaped.
    pub fn global(&self) -> Result<&SlotSet, Refusal> {
        if self.escaped.is_empty() {
            Ok(&self.reached)
        } else {
            Err(Refusal::IncompleteFocus)
        }
    }
}

/// [`ScopedReach`] without the escaped identities: the slots they leave from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedMask {
    pub reached: SlotSet,
    /// Slots, within the hop budget, with an edge out of the focus.
    pub escape_sources: SlotSet,
}

impl ScopedMask {
    pub fn global(&self) -> Result<&SlotSet, Refusal> {
        if self.escape_sources == [0; SLOT_WORDS] {
            Ok(&self.reached)
        } else {
            Err(Refusal::IncompleteFocus)
        }
    }
}

/// Reachability by walking the index on every call: the direct arm. Every hop
/// seeks the index again and translates every target.
pub fn reach_direct(
    index: &dyn Navigate,
    focus: &Focus,
    preds: &[u32],
    seeds: &SlotSet,
    hops: usize,
) -> ScopedReach {
    let mut reached = *seeds;
    let mut frontier = *seeds;
    let mut escaped = BTreeSet::new();
    for _ in 0..hops {
        let mut next = [0u64; SLOT_WORDS];
        for s in slots_of(&frontier) {
            let g = focus.id(s);
            for &p in preds {
                index.objects(p, g, &mut |o, _| match focus.slot_of(o) {
                    Some(t) => next[(t >> 6) as usize] |= 1 << (t & 63),
                    None => {
                        escaped.insert(o);
                    }
                });
            }
        }
        for w in 0..SLOT_WORDS {
            frontier[w] = next[w] & !reached[w];
            reached[w] |= next[w];
        }
        if frontier == [0; SLOT_WORDS] {
            break;
        }
    }
    ScopedReach { reached, escaped }
}

/// [`reach_direct`] with the same demand as [`Csr::reach_sources`]: escapes
/// as source slots, not identities. Still seeks the index and translates
/// every target on every call.
pub fn reach_direct_sources(
    index: &dyn Navigate,
    focus: &Focus,
    preds: &[u32],
    seeds: &SlotSet,
    hops: usize,
) -> ScopedMask {
    let mut reached = *seeds;
    let mut frontier = *seeds;
    let mut escape_sources = [0u64; SLOT_WORDS];
    for _ in 0..hops {
        let mut next = [0u64; SLOT_WORDS];
        for s in slots_of(&frontier) {
            let g = focus.id(s);
            for &p in preds {
                index.objects(p, g, &mut |o, _| match focus.slot_of(o) {
                    Some(t) => next[(t >> 6) as usize] |= 1 << (t & 63),
                    None => escape_sources[(s >> 6) as usize] |= 1 << (s & 63),
                });
            }
        }
        for w in 0..SLOT_WORDS {
            frontier[w] = next[w] & !reached[w];
            reached[w] |= next[w];
        }
        if frontier == [0; SLOT_WORDS] {
            break;
        }
    }
    ScopedMask {
        reached,
        escape_sources,
    }
}

// ---- prepared: CSR ---------------------------------------------------------

/// One predicate's edges out of the focus, compressed by source slot.
/// `dst[k]` is a local target slot; `row[k]` is the canonical fact.
pub struct CsrPred {
    pub p: u32,
    offsets: [u32; SLOTS + 1],
    dst: Vec<u8>,
    row: Vec<u32>,
    /// `(source slot, global target, row)` for every edge that leaves the
    /// focus, sorted by source slot.
    escaped: Vec<(u8, u32, u32)>,
    /// Slots with at least one edge in `escaped`.
    has_escape: SlotSet,
}

/// Prepared CSR over the focus for a set of predicates.
pub struct Csr {
    pub preds: Vec<CsrPred>,
    pub witness: Witness,
}

impl Csr {
    /// One seek per `(predicate, slot)`; every target translated once.
    // A slot number is the coordinate, not a loop counter.
    #[expect(clippy::needless_range_loop)]
    pub fn prepare(index: &dyn Navigate, focus: &Focus, preds: &[u32], witness: Witness) -> Csr {
        let mut out = Vec::with_capacity(preds.len());
        for &p in preds {
            let mut offsets = [0u32; SLOTS + 1];
            let mut dst = Vec::new();
            let mut row = Vec::new();
            let mut escaped = Vec::new();
            for s in 0..SLOTS {
                offsets[s] = dst.len() as u32;
                if s < focus.len() {
                    index.objects(p, focus.id(s as u8), &mut |o, r| match focus.slot_of(o) {
                        Some(t) => {
                            dst.push(t);
                            row.push(r);
                        }
                        None => escaped.push((s as u8, o, r)),
                    });
                }
            }
            offsets[SLOTS] = dst.len() as u32;
            out.push(CsrPred {
                p,
                offsets,
                dst,
                row,
                has_escape: slot_set(escaped.iter().map(|e| e.0)),
                escaped,
            });
        }
        Csr {
            preds: out,
            witness,
        }
    }

    /// Edges inside the focus, over all predicates.
    pub fn edges(&self) -> usize {
        self.preds.iter().map(|c| c.dst.len()).sum()
    }

    /// Edges leaving the focus.
    pub fn escapes(&self) -> usize {
        self.preds.iter().map(|c| c.escaped.len()).sum()
    }

    /// Derived bytes (offsets, slot targets, row references, escape list).
    pub fn bytes(&self) -> usize {
        self.preds
            .iter()
            .map(|c| (SLOTS + 1) * 4 + c.dst.len() + c.row.len() * 4 + c.escaped.len() * 12)
            .sum()
    }

    /// The canonical rows behind the in-focus edges of `p` from `slot`.
    pub fn rows(&self, p: u32, slot: u8) -> &[u32] {
        let c = self
            .preds
            .iter()
            .find(|c| c.p == p)
            .expect("predicate prepared");
        let s = slot as usize;
        &c.row[c.offsets[s] as usize..c.offsets[s + 1] as usize]
    }

    /// Slots that have an out-of-focus edge under any of `preds`.
    pub fn escape_plane(&self, preds: &[u32]) -> SlotSet {
        let mut e = [0u64; SLOT_WORDS];
        for c in self.preds.iter().filter(|c| preds.contains(&c.p)) {
            for (w, h) in e.iter_mut().zip(c.has_escape) {
                *w |= h;
            }
        }
        e
    }

    /// The out-of-focus identities reached from `sources` under `preds`,
    /// materialized on demand (the identities [`ScopedReach::escaped`] holds).
    pub fn escape_ids(&self, sources: &SlotSet, preds: &[u32]) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        self.escapes_from(sources, preds, &mut out);
        out
    }

    /// Reachability without materializing escaped identities: which reached
    /// slots have an edge out of the focus within the hop budget. As exact as
    /// [`Csr::reach`]; [`Csr::escape_ids`] turns the sources into identities.
    pub fn reach_sources(
        &self,
        current: &Witness,
        preds: &[u32],
        seeds: &SlotSet,
        hops: usize,
    ) -> Result<ScopedMask, Refusal> {
        self.witness.check(current)?;
        let esc_plane = self.escape_plane(preds);
        let mut reached = *seeds;
        let mut frontier = *seeds;
        let mut escape_sources = [0u64; SLOT_WORDS];
        for _ in 0..hops {
            let mut next = [0u64; SLOT_WORDS];
            for w in 0..SLOT_WORDS {
                escape_sources[w] |= frontier[w] & esc_plane[w];
            }
            for c in self.preds.iter().filter(|c| preds.contains(&c.p)) {
                for s in slots_of(&frontier) {
                    let s = s as usize;
                    for &t in &c.dst[c.offsets[s] as usize..c.offsets[s + 1] as usize] {
                        next[(t >> 6) as usize] |= 1 << (t & 63);
                    }
                }
            }
            for w in 0..SLOT_WORDS {
                frontier[w] = next[w] & !reached[w];
                reached[w] |= next[w];
            }
            if frontier == [0; SLOT_WORDS] {
                break;
            }
        }
        Ok(ScopedMask {
            reached,
            escape_sources,
        })
    }

    fn escapes_from(&self, frontier: &SlotSet, preds: &[u32], into: &mut BTreeSet<u32>) {
        for c in self.preds.iter().filter(|c| preds.contains(&c.p)) {
            for &(s, o, _) in &c.escaped {
                if frontier[(s >> 6) as usize] >> (s & 63) & 1 == 1 {
                    into.insert(o);
                }
            }
        }
    }

    /// Reachability over the prepared CSR. Refuses a stale witness.
    pub fn reach(
        &self,
        current: &Witness,
        preds: &[u32],
        seeds: &SlotSet,
        hops: usize,
    ) -> Result<ScopedReach, Refusal> {
        self.witness.check(current)?;
        let mut reached = *seeds;
        let mut frontier = *seeds;
        let mut escaped = BTreeSet::new();
        for _ in 0..hops {
            self.escapes_from(&frontier, preds, &mut escaped);
            let mut next = [0u64; SLOT_WORDS];
            for c in self.preds.iter().filter(|c| preds.contains(&c.p)) {
                for s in slots_of(&frontier) {
                    let s = s as usize;
                    for &t in &c.dst[c.offsets[s] as usize..c.offsets[s + 1] as usize] {
                        next[(t >> 6) as usize] |= 1 << (t & 63);
                    }
                }
            }
            for w in 0..SLOT_WORDS {
                frontier[w] = next[w] & !reached[w];
                reached[w] |= next[w];
            }
            if frontier == [0; SLOT_WORDS] {
                break;
            }
        }
        Ok(ScopedReach { reached, escaped })
    }
}

// ---- prepared: tile --------------------------------------------------------

/// Address of edge `source → target` in a tile plane.
pub fn tile_addr(source: u8, target: u8) -> usize {
    ((target as usize) << 8) | source as usize
}

/// 256 × 256 Boolean adjacency per predicate, addressed `(target, source)`,
/// built from a [`Csr`] (so it adds no navigation of its own).
pub struct Tile {
    pub preds: Vec<u32>,
    planes: Vec<Box<[u64]>>,
    pub witness: Witness,
}

fn plane() -> Box<[u64]> {
    vec![0u64; TILE_WORDS].into_boxed_slice()
}

fn set(plane: &mut [u64], i: usize) {
    plane[i >> 6] |= 1 << (i & 63);
}

impl Tile {
    pub fn from_csr(csr: &Csr) -> Tile {
        let mut planes = Vec::with_capacity(csr.preds.len());
        for c in &csr.preds {
            let mut pl = plane();
            for s in 0..SLOTS {
                for &t in &c.dst[c.offsets[s] as usize..c.offsets[s + 1] as usize] {
                    set(&mut pl, tile_addr(s as u8, t));
                }
            }
            planes.push(pl);
        }
        Tile {
            preds: csr.preds.iter().map(|c| c.p).collect(),
            planes,
            witness: csr.witness,
        }
    }

    /// Plane bytes (8 KiB per predicate).
    pub fn bytes(&self) -> usize {
        self.planes.len() * TILE_WORDS * 8
    }

    pub fn plane(&self, p: u32) -> &[u64] {
        let i = self
            .preds
            .iter()
            .position(|&q| q == p)
            .expect("predicate in tile");
        &self.planes[i]
    }
}

/// A thought's sparse speculative change to the shared tile: edges it adds
/// and edges it retracts. The shared tile is never written.
pub struct ThoughtDelta {
    /// The thought incarnation this delta belongs to (stand-in for
    /// `ThoughtCommitId`, D-RMAP-1A §7).
    pub thought: u64,
    pub p: u32,
    add: Box<[u64]>,
    del: Box<[u64]>,
    edits: usize,
}

impl ThoughtDelta {
    pub fn new(thought: u64, p: u32, add: &[(u8, u8)], del: &[(u8, u8)]) -> Self {
        let (mut a, mut d) = (plane(), plane());
        for &(s, t) in add {
            set(&mut a, tile_addr(s, t));
        }
        for &(s, t) in del {
            set(&mut d, tile_addr(s, t));
        }
        ThoughtDelta {
            thought,
            p,
            add: a,
            del: d,
            edits: add.len() + del.len(),
        }
    }

    /// Bytes as stored sparsely (two bytes per edit) and as the two planes
    /// the executor reads.
    pub fn bytes_sparse(&self) -> usize {
        self.edits * 2
    }

    pub fn bytes_planes(&self) -> usize {
        2 * TILE_WORDS * 8
    }
}

/// Executor state for tile hops: the broadcast frontier, the hop result and
/// the mask-risc scratch, reused across calls (no allocation per hop).
pub struct TileExec {
    frontier: Box<[u64]>,
    out: Box<[u64]>,
    scratch: Scratch<'static>,
}

impl Default for TileExec {
    fn default() -> Self {
        Self::new()
    }
}

impl TileExec {
    pub fn new() -> Self {
        TileExec {
            frontier: plane(),
            out: plane(),
            // Up to two scratch slots for every program built below.
            scratch: Scratch::new(words_for(TILE_ROWS), 2),
        }
    }

    /// The hop program over `n` predicate planes (masks `0..n`), an optional
    /// thought delta (masks `n` = add, `n + 1` = del) and the broadcast
    /// frontier (the last mask). Every operand is a plane over the same
    /// `(target, source)` address space.
    pub fn program(n: usize, delta: bool) -> Program {
        let f = Operand::Plane((n + if delta { 2 } else { 0 }) as u16);
        let mut ops = Vec::new();
        let mut acc = Operand::Plane(0);
        for i in 1..n {
            ops.push(MaskOp::Or {
                a: acc,
                b: Operand::Plane(i as u16),
                dst: 0,
            });
            acc = Operand::Scratch(0);
        }
        if delta {
            ops.push(MaskOp::Or {
                a: acc,
                b: Operand::Plane(n as u16),
                dst: 0,
            });
            ops.push(MaskOp::AndNot {
                a: Operand::Scratch(0),
                b: Operand::Plane(n as u16 + 1),
                dst: 0,
            });
            acc = Operand::Scratch(0);
        }
        ops.push(MaskOp::And {
            a: acc,
            b: f,
            dst: 1,
        });
        Program::new(
            ops,
            Terminal::Keep {
                mask: Operand::Scratch(1),
            },
        )
    }

    /// One hop: `next[t] = ∃ s. frontier[s] ∧ edge(s, t)`, run by mask-risc.
    fn hop(
        &mut self,
        planes: &[&[u64]],
        delta: Option<&ThoughtDelta>,
        frontier: &SlotSet,
    ) -> SlotSet {
        // Broadcast: frontier[s] at every (t, s). Derived, rewritten per hop.
        for t in 0..SLOTS {
            self.frontier[t * SLOT_WORDS..(t + 1) * SLOT_WORDS].copy_from_slice(frontier);
        }
        let mut masks: Vec<&[u64]> = planes.to_vec();
        if let Some(d) = delta {
            masks.push(&d.add);
            masks.push(&d.del);
        }
        masks.push(&self.frontier);
        let prog = Self::program(planes.len(), delta.is_some());
        let pl = Planes {
            n_rows: TILE_ROWS,
            masks: &masks,
            lanes: &[],
        };
        let v = execute_into(
            &prog,
            &pl,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(&mut self.out),
        )
        .expect("tile program is valid");
        debug_assert!(matches!(v, Value::Mask(_)));
        let mut next = [0u64; SLOT_WORDS];
        for t in 0..SLOTS {
            let row = &self.out[t * SLOT_WORDS..(t + 1) * SLOT_WORDS];
            if row.iter().any(|&w| w != 0) {
                next[t >> 6] |= 1 << (t & 63);
            }
        }
        next
    }

    /// Reachability over the tile, optionally through a thought's delta.
    /// Escapes come from the CSR the tile was built from.
    #[expect(clippy::too_many_arguments)]
    pub fn reach(
        &mut self,
        tile: &Tile,
        csr: &Csr,
        current: &Witness,
        delta: Option<&ThoughtDelta>,
        thought: Option<u64>,
        preds: &[u32],
        seeds: &SlotSet,
        hops: usize,
    ) -> Result<ScopedReach, Refusal> {
        tile.witness.check(current)?;
        csr.witness.check(current)?;
        if let Some(d) = delta {
            if Some(d.thought) != thought {
                return Err(Refusal::ThoughtMismatch);
            }
            assert!(preds.contains(&d.p), "delta predicate is not in the query");
        }
        let planes: Vec<&[u64]> = preds.iter().map(|&p| tile.plane(p)).collect();
        let mut reached = *seeds;
        let mut frontier = *seeds;
        let mut escaped = BTreeSet::new();
        for _ in 0..hops {
            csr.escapes_from(&frontier, preds, &mut escaped);
            let next = self.hop(&planes, delta, &frontier);
            for w in 0..SLOT_WORDS {
                frontier[w] = next[w] & !reached[w];
                reached[w] |= next[w];
            }
            if frontier == [0; SLOT_WORDS] {
                break;
            }
        }
        Ok(ScopedReach { reached, escaped })
    }
}

impl TileExec {
    /// [`TileExec::reach`] without materializing escaped identities.
    #[expect(clippy::too_many_arguments)]
    pub fn reach_sources(
        &mut self,
        tile: &Tile,
        csr: &Csr,
        current: &Witness,
        delta: Option<&ThoughtDelta>,
        thought: Option<u64>,
        preds: &[u32],
        seeds: &SlotSet,
        hops: usize,
    ) -> Result<ScopedMask, Refusal> {
        tile.witness.check(current)?;
        csr.witness.check(current)?;
        if let Some(d) = delta {
            if Some(d.thought) != thought {
                return Err(Refusal::ThoughtMismatch);
            }
            assert!(preds.contains(&d.p), "delta predicate is not in the query");
        }
        let esc_plane = csr.escape_plane(preds);
        let planes: Vec<&[u64]> = preds.iter().map(|&p| tile.plane(p)).collect();
        let mut reached = *seeds;
        let mut frontier = *seeds;
        let mut escape_sources = [0u64; SLOT_WORDS];
        for _ in 0..hops {
            for w in 0..SLOT_WORDS {
                escape_sources[w] |= frontier[w] & esc_plane[w];
            }
            let next = self.hop(&planes, delta, &frontier);
            for w in 0..SLOT_WORDS {
                frontier[w] = next[w] & !reached[w];
                reached[w] |= next[w];
            }
            if frontier == [0; SLOT_WORDS] {
                break;
            }
        }
        Ok(ScopedMask {
            reached,
            escape_sources,
        })
    }
}

/// A focus's dimensions loaded into 256-slot register lanes: a COPY of the
/// canonical values (12 bytes per slot), made once per focus. The other arm
/// ([`box_count`]) copies nothing and folds over the canonical lanes.
pub struct FocusLanes {
    pub lat: Vec<i32>,
    pub lon: Vec<i32>,
    pub class: Vec<u32>,
    /// The focus generation the copy was taken from.
    pub generation: u64,
}

impl FocusLanes {
    pub fn load(places: &Places, ids: &[u32], generation: u64) -> Self {
        FocusLanes {
            lat: ids.iter().map(|&i| places.lat[i as usize]).collect(),
            lon: ids.iter().map(|&i| places.lon[i as usize]).collect(),
            class: ids.iter().map(|&i| places.class[i as usize]).collect(),
            generation,
        }
    }

    /// Canonical payload bytes copied.
    pub fn bytes(&self) -> usize {
        self.lat.len() * 12
    }

    /// The [`BoxQuery`] count over the register lanes, by mask-risc.
    /// Refuses a copy taken under another focus generation.
    pub fn count(
        &self,
        generation: u64,
        q: &BoxQuery,
        scratch: &mut Scratch<'_>,
    ) -> Result<usize, Refusal> {
        if generation != self.generation {
            return Err(Refusal::FocusChanged);
        }
        let n = self.lat.len();
        let mut all = vec![u64::MAX; words_for(n)];
        if !n.is_multiple_of(64) {
            *all.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
        }
        let masks = [all.as_slice()];
        let lanes = [
            LaneRef::I32(&self.lat),
            LaneRef::I32(&self.lon),
            LaneRef::U32(&self.class),
        ];
        let pl = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        match execute(&box_program(q), &pl, scratch, None).expect("box program is valid") {
            Value::Count(c) => Ok(c),
            v => unreachable!("Count terminal returned {v:?}"),
        }
    }
}

// ---- incremental rebinding -------------------------------------------------

/// Rebind replaced slots in a prepared CSR + tile without rebuilding the
/// others. A changed slot's outgoing edges come from one forward seek each.
/// Edges INTO a newly focused identity need no reverse index: from an
/// unchanged member they were escapes before (the identity was outside the
/// focus), so re-resolving the escape list finds them; from another changed
/// slot, that slot's own forward seek finds them. Returns the new CSR and
/// tile.
// A slot number is the coordinate, not a loop counter.
#[expect(clippy::needless_range_loop)]
pub fn rebind(
    forward: &SortedRun<'_>,
    old: &Focus,
    new: &Focus,
    csr: &Csr,
    witness: Witness,
) -> (Csr, Tile) {
    let changed: Vec<u8> = (0..new.len())
        .filter(|&s| old.id(s as u8) != new.id(s as u8))
        .map(|s| s as u8)
        .collect();
    let is_changed = slot_set(changed.iter().copied());
    let hit = |s: u8| is_changed[(s >> 6) as usize] >> (s & 63) & 1 == 1;
    let mut preds = Vec::with_capacity(csr.preds.len());
    for c in &csr.preds {
        // Per source slot: keep unchanged edges, drop those touching a changed
        // slot, then add the re-derived ones.
        let mut adj: Vec<Vec<(u8, u32)>> = vec![Vec::new(); SLOTS];
        let mut esc: Vec<(u8, u32, u32)> = Vec::new();
        for s in 0..new.len() {
            if hit(s as u8) {
                continue;
            }
            let lo = c.offsets[s] as usize;
            let hi = c.offsets[s + 1] as usize;
            for k in lo..hi {
                if !hit(c.dst[k]) {
                    adj[s].push((c.dst[k], c.row[k]));
                }
            }
        }
        for &(s, o, r) in &c.escaped {
            // An old escape may now land on a newly focused identity.
            if hit(s) {
                continue;
            }
            match new.slot_of(o) {
                Some(t) => adj[s as usize].push((t, r)),
                None => esc.push((s, o, r)),
            }
        }
        for &s in &changed {
            // Outgoing edges of the new identity.
            forward.neighbours(c.p, new.id(s), |o, r| match new.slot_of(o) {
                Some(t) => adj[s as usize].push((t, r)),
                None => esc.push((s, o, r)),
            });
        }
        // An edge from an unchanged member to the OLD identity of a changed
        // slot that is no longer focused becomes an escape.
        for s in 0..new.len() {
            if hit(s as u8) {
                continue;
            }
            let lo = c.offsets[s] as usize;
            let hi = c.offsets[s + 1] as usize;
            for k in lo..hi {
                let t = c.dst[k];
                if hit(t) {
                    let g = old.id(t);
                    match new.slot_of(g) {
                        Some(t2) => adj[s].push((t2, c.row[k])),
                        None => esc.push((s as u8, g, c.row[k])),
                    }
                }
            }
        }
        let mut offsets = [0u32; SLOTS + 1];
        let mut dst = Vec::new();
        let mut row = Vec::new();
        for s in 0..SLOTS {
            offsets[s] = dst.len() as u32;
            adj[s].sort_unstable();
            adj[s].dedup();
            for &(t, r) in &adj[s] {
                dst.push(t);
                row.push(r);
            }
        }
        offsets[SLOTS] = dst.len() as u32;
        esc.sort_unstable();
        esc.dedup();
        preds.push(CsrPred {
            p: c.p,
            offsets,
            dst,
            row,
            has_escape: slot_set(esc.iter().map(|e| e.0)),
            escaped: esc,
        });
    }
    let csr = Csr { preds, witness };
    let tile = Tile::from_csr(&csr);
    (csr, tile)
}

impl Csr {
    /// The prepared relation as a set, for comparing two preparations.
    pub fn edge_set(&self) -> BTreeSet<(u32, u8, u8, u32)> {
        let mut out = BTreeSet::new();
        for c in &self.preds {
            for s in 0..SLOTS {
                for k in c.offsets[s] as usize..c.offsets[s + 1] as usize {
                    out.insert((c.p, s as u8, c.dst[k], c.row[k]));
                }
            }
        }
        out
    }

    pub fn escape_set(&self) -> BTreeSet<(u32, u8, u32, u32)> {
        self.preds
            .iter()
            .flat_map(|c| c.escaped.iter().map(move |&(s, o, r)| (c.p, s, o, r)))
            .collect()
    }
}

// ---- dimensions over canonical lanes --------------------------------------

/// Which row order a mask's bit positions follow. Two orders over the same
/// 65 536 rows both use 16-bit positions; equal positions mean nothing across
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// The canonical row order of the lanes.
    Canonical,
    /// A spatial (Morton) order, by index build.
    Morton(u64),
}

/// A focus as a presence plane over a row order: the zero-copy form for typed
/// dimensions, which stay in their canonical lanes.
pub struct FocusMask {
    pub words: Vec<u64>,
    pub ordering: Ordering,
    pub generation: u64,
}

impl FocusMask {
    pub fn over(rows: usize, ids: &[u32], ordering: Ordering, generation: u64) -> Self {
        let mut words = vec![0u64; words_for(rows)];
        for &i in ids {
            words[(i >> 6) as usize] |= 1 << (i & 63);
        }
        FocusMask {
            words,
            ordering,
            generation,
        }
    }
}

/// Canonical place lanes: latitude and longitude in micro-degrees, and a class.
pub struct Places {
    pub lat: Vec<i32>,
    pub lon: Vec<i32>,
    pub class: Vec<u32>,
}

/// A bounding box and a class: the demanded count.
#[derive(Clone, Copy, Debug)]
pub struct BoxQuery {
    pub lat: (i32, i32),
    pub lon: (i32, i32),
    pub class: u32,
}

/// The mask-risc program for [`BoxQuery`]: five gated predicates, then Count.
pub fn box_program(q: &BoxQuery) -> Program {
    let gate = |i: u16| Some(Operand::Scratch(i));
    Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GeI32 {
                    lane: 0,
                    t: q.lat.0,
                },
                under: Some(Operand::Plane(0)),
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LeI32 {
                    lane: 0,
                    t: q.lat.1,
                },
                under: gate(0),
                dst: 1,
            },
            MaskOp::Pred {
                pred: Pred::GeI32 {
                    lane: 1,
                    t: q.lon.0,
                },
                under: gate(1),
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LeI32 {
                    lane: 1,
                    t: q.lon.1,
                },
                under: gate(0),
                dst: 1,
            },
            MaskOp::Pred {
                pred: Pred::EqU32 {
                    lane: 2,
                    v: q.class,
                },
                under: gate(1),
                dst: 0,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

/// Count focused places matching `q`, natively, over the canonical lanes.
/// Refuses a focus mask built in another row order than the lanes'.
pub fn box_count(
    places: &Places,
    focus: &FocusMask,
    q: &BoxQuery,
    scratch: &mut Scratch<'_>,
) -> Result<usize, Refusal> {
    if focus.ordering != Ordering::Canonical {
        return Err(Refusal::OrderingMismatch);
    }
    let masks = [focus.words.as_slice()];
    let lanes = [
        LaneRef::I32(&places.lat),
        LaneRef::I32(&places.lon),
        LaneRef::U32(&places.class),
    ];
    let pl = Planes {
        n_rows: places.lat.len(),
        masks: &masks,
        lanes: &lanes,
    };
    match execute(&box_program(q), &pl, scratch, None).expect("box program is valid") {
        Value::Count(n) => Ok(n),
        v => unreachable!("Count terminal returned {v:?}"),
    }
}

/// The same count by reading each focused identity's lanes in turn.
pub fn box_count_by_id(places: &Places, ids: &[u32], q: &BoxQuery) -> usize {
    ids.iter()
        .filter(|&&i| {
            let i = i as usize;
            (q.lat.0..=q.lat.1).contains(&places.lat[i])
                && (q.lon.0..=q.lon.1).contains(&places.lon[i])
                && places.class[i] == q.class
        })
        .count()
}

/// Morton order of the places: row numbers sorted by interleaved quantized
/// coordinates. An index of `u32` row numbers; the coordinates are not copied.
pub fn morton_order(places: &Places) -> Vec<u32> {
    let q = |v: i32, lo: i64, span: i64| (((v as i64 - lo) * 65_535) / span) as u32;
    let key = |i: usize| {
        let x = q(places.lon[i], -180_000_000, 360_000_000);
        let y = q(places.lat[i], -90_000_000, 180_000_000);
        let mut k = 0u64;
        for b in 0..16 {
            k |= (((x >> b) & 1) as u64) << (2 * b);
            k |= (((y >> b) & 1) as u64) << (2 * b + 1);
        }
        k
    };
    let mut perm: Vec<u32> = (0..places.lat.len() as u32).collect();
    perm.sort_unstable_by_key(|&i| key(i as usize));
    perm
}

// ---- membership planes (map × identity) ------------------------------------

/// Map × identity: one 256-bit plane per selected predicate over the focus
/// slots. Built from posting lists through the focus binding.
pub fn membership_planes(postings: &[&[u32]], focus: &Focus) -> Vec<SlotSet> {
    postings
        .iter()
        .map(|docs| slot_set(docs.iter().filter_map(|&d| focus.slot_of(d))))
        .collect()
}

/// Intersect two membership planes with mask-risc: And, then Keep. Returns
/// the focused identities, in slot order.
pub fn intersect_planes(
    a: &SlotSet,
    b: &SlotSet,
    focus: &Focus,
    scratch: &mut Scratch<'_>,
) -> Vec<u32> {
    let masks = [a.as_slice(), b.as_slice()];
    let pl = Planes {
        n_rows: SLOTS,
        masks: &masks,
        lanes: &[],
    };
    let prog = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let mut out = [0u64; SLOT_WORDS];
    execute_into(&prog, &pl, &Foreign::NONE, scratch, Out::Mask(&mut out)).expect("valid");
    let mut ids: Vec<u32> = slots_of(&out).into_iter().map(|s| focus.id(s)).collect();
    ids.sort_unstable();
    ids
}

/// The count of the same intersection, as And then Count (POPCNT).
pub fn intersect_count(a: &SlotSet, b: &SlotSet, scratch: &mut Scratch<'_>) -> usize {
    let masks = [a.as_slice(), b.as_slice()];
    let pl = Planes {
        n_rows: SLOTS,
        masks: &masks,
        lanes: &[],
    };
    let prog = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    match execute(&prog, &pl, scratch, None).expect("valid") {
        Value::Count(n) => n,
        v => unreachable!("{v:?}"),
    }
}

/// Merge-intersect two sorted posting slices (the ordered-plane arm).
pub fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
}

// ---- deterministic fixtures -------------------------------------------------

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A sparse typed graph over `n` identities and `preds` predicates. Identities
/// are scattered: community membership is a hidden permutation of the ids, so
/// a community's members are spread over the whole id space. `local` is the
/// share of edges that stay inside their community (of size `community`).
pub fn graph(n: u32, preds: u32, degree: u32, community: u32, local: f64, seed: u64) -> Vec<Fact> {
    let mut rng = Rng(seed);
    // Hidden position <-> id bijection.
    let mut pos_to_id: Vec<u32> = (0..n).collect();
    for i in (1..n as usize).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        pos_to_id.swap(i, j);
    }
    let mut id_to_pos = vec![0u32; n as usize];
    for (p, &id) in pos_to_id.iter().enumerate() {
        id_to_pos[id as usize] = p as u32;
    }
    let mut facts = Vec::new();
    for s in 0..n {
        let c = id_to_pos[s as usize] / community;
        for p in 0..preds {
            for _ in 0..degree {
                let o = if (rng.below(1_000_000) as f64) < local * 1e6 {
                    let pos = c * community + rng.below(community as u64) as u32;
                    pos_to_id[pos.min(n - 1) as usize]
                } else {
                    rng.below(n as u64) as u32
                };
                if o != s {
                    facts.push([s, p, o]);
                }
            }
        }
    }
    facts.sort_unstable();
    facts.dedup();
    // Canonical order is not the index order.
    for i in (1..facts.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        facts.swap(i, j);
    }
    facts
}

/// Places with scattered coordinates and 16 classes.
pub fn places(n: usize, seed: u64) -> Places {
    let mut rng = Rng(seed);
    let mut p = Places {
        lat: Vec::with_capacity(n),
        lon: Vec::with_capacity(n),
        class: Vec::with_capacity(n),
    };
    for _ in 0..n {
        p.lat.push(rng.below(180_000_000) as i32 - 90_000_000);
        p.lon.push(rng.below(360_000_000) as i32 - 180_000_000);
        p.class.push(rng.below(16) as u32);
    }
    p
}
