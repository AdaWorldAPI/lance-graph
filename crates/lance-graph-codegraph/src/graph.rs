//! The codegraph: artifact rows interned into symbol positions (one 64- or
//! 256-position tile, or several 256-position tiles) and folded into six
//! relations of an xlog [`Cube`].

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::Path;

use lance_graph_dt64::dr::{Axis, LocalMap, MapRefusal, MapTable, TileId, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, EdgeOp, RelId, StagedEdges};
use lance_graph_xlog::wide::{WideRows, WideStats};

use crate::artifact::{
    self, CrateSource, Interner, Lanes, StagedNames, SymbolStage, Symbols, NO_SPAN, P_CALLS,
    P_CRATE, P_DEFINES, P_IMPLEMENTS, P_MAY, P_PUBLIC, P_REFERENCES, P_SYNTHETIC, P_UNRESOLVED,
};
use crate::artifact_bin::{CrateFile, Header};

/// A symbol's position: its interning index, which is also its cube coordinate.
pub type Pos = u32;

/// Why a directory could not become a [`Codegraph`].
#[derive(Debug)]
pub enum IntakeRefusal {
    /// A directory or file could not be read.
    Io(std::io::Error),
    /// The file's `#version` line is not 2 (0 when absent or unparsable).
    Version {
        /// The version found.
        found: u32,
    },
    /// The `#schema` line, or a required column, is missing.
    Schema {
        /// The missing column (or `#schema`).
        missing: String,
    },
    /// A data line whose field count differs from the schema, or whose
    /// span bounds are not numbers; 1-based line within its file.
    Arity {
        /// The offending line.
        line: usize,
    },
    /// The tile map was refused.
    Map(MapRefusal),
    /// The cube was refused.
    Cube(CubeRefusal),
    /// A `<crate>.rows.bin` file was refused.
    Bin(crate::artifact_bin::BinRefusal),
    /// A `<crate>.residuals.tsv` file was refused.
    Residual(crate::artifact::ResidualRefusal),
    /// A row names a placeholder symbol (a `?` path segment): not an
    /// identity, so the fact is refused. 0-based row of the read lane set.
    Placeholder {
        /// The offending row.
        row: usize,
        /// The placeholder symbol.
        name: String,
    },
}

/// Whether the extractor finished a crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStatus {
    /// The crate's CRATE row says `COMPLETE`.
    Complete,
    /// Any other CRATE status.
    Partial,
}

/// `CALLS`: (caller, callee) for resolved calls.
pub const CALLS: RelId = 0;
/// `NODE`: (x, x) for every position.
pub const NODE: RelId = 1;
/// `MAY`: (caller, Trait::method) for may-dispatch rows.
pub const MAY: RelId = 2;
/// `IMPL`: (impl fn, Trait::method).
pub const IMPL: RelId = 3;
/// `RCONF`: (callee, caller), the confirmed reverse edges.
pub const RCONF: RelId = 4;
/// `RPOSS`: the possible reverse edges, including dispatch and `?dynamic`.
pub const RPOSS: RelId = 5;

/// Positions per full tile of a multi-tile store.
const TILE_POSITIONS: usize = 256;
const NONE: u32 = u32::MAX;
const DYNAMIC: &str = "?dynamic";

/// Position masks, one bit per position, `words()` words each.
#[derive(Debug, Default)]
struct Masks {
    fns: Vec<u64>,
    public: Vec<u64>,
    impl_subjects: Vec<u64>,
    synthetic: Vec<u64>,
    referenced: Vec<u64>,
    called: Vec<u64>,
    may_targets: Vec<u64>,
    dynamic_callers: Vec<u64>,
    stored_owner: Vec<u64>,
}

/// Symbol positions over the shared interner. A position is the symbol's
/// first-seen index among the row symbols (the cube coordinate); the interner
/// also holds texts that are not symbols (grades, files, full MAY objects),
/// so the two numberings are kept apart: `of_id` maps an interner id to its
/// position (`NONE` when the text is not a symbol), `id_of` is the inverse.
/// Query time does one interner lookup and one array read, never a name map.
///
/// Positions are only ever assigned through a [`PosStage`] and adopted with
/// [`Positions::absorb`], so a refused refresh leaves this table untouched.
#[derive(Debug, Default)]
struct Positions {
    of_id: Vec<u32>,
    id_of: Vec<u32>,
    /// Interner id of the first `::` segment of each position's name (the
    /// owning crate), computed once when the position is assigned so the
    /// `stored_owner` mask never splits strings.
    owner: Vec<u32>,
    last_owner: u32,
}

/// Reading a position: the committed table, or a stage over it.
trait PosAt {
    /// The position of interner id `id`, `None` when it is not a symbol.
    fn at(&self, id: u32) -> Option<Pos>;
    /// Positions assigned.
    fn len(&self) -> usize;
}

impl PosAt for Positions {
    fn at(&self, id: u32) -> Option<Pos> {
        self.of_id.get(id as usize).copied().filter(|&p| p != NONE)
    }
    fn len(&self) -> usize {
        self.id_of.len()
    }
}

/// New positions over a borrowed [`Positions`], which is never written: the
/// stage holds the appended `(id, owner)` pairs and its own `last_owner`
/// until [`Positions::absorb`].
struct PosStage<'a> {
    base: &'a Positions,
    ids: Vec<u32>,
    owners: Vec<u32>,
    of_new: HashMap<u32, Pos>,
    last_owner: u32,
}

/// What a [`PosStage`] assigned, without the borrow.
struct StagedPositions {
    base_len: usize,
    ids: Vec<u32>,
    owners: Vec<u32>,
    last_owner: u32,
}

impl PosAt for PosStage<'_> {
    fn at(&self, id: u32) -> Option<Pos> {
        self.base.at(id).or_else(|| self.of_new.get(&id).copied())
    }
    fn len(&self) -> usize {
        self.base.len() + self.ids.len()
    }
}

impl<'a> PosStage<'a> {
    fn new(base: &'a Positions) -> Self {
        Self {
            base,
            ids: Vec::new(),
            owners: Vec::new(),
            of_new: HashMap::new(),
            last_owner: base.last_owner,
        }
    }

    /// Assign `id` the next position unless it already has one.
    fn assign(&mut self, id: u32, symbols: &mut impl Interner) -> Pos {
        if let Some(p) = self.at(id) {
            return p;
        }
        let p = self.len() as Pos;
        self.of_new.insert(id, p);
        self.ids.push(id);
        let owner = {
            let name = symbols.name(id);
            match name.find("::") {
                None => id,
                Some(cut) => {
                    let pre = &name[..cut];
                    if self.last_owner != NONE && symbols.name(self.last_owner) == pre {
                        self.last_owner
                    } else if let Some(o) = symbols.get(pre) {
                        o
                    } else {
                        let owned = pre.to_string();
                        symbols.intern(&owned)
                    }
                }
            }
        };
        self.last_owner = owner;
        self.owners.push(owner);
        p
    }

    fn finish(self) -> StagedPositions {
        StagedPositions {
            base_len: self.base.len(),
            ids: self.ids,
            owners: self.owners,
            last_owner: self.last_owner,
        }
    }
}

impl Positions {
    fn new() -> Self {
        Self {
            last_owner: NONE,
            ..Self::default()
        }
    }

    /// Adopt what a stage over this table assigned. Cannot fail.
    fn absorb(&mut self, staged: StagedPositions) {
        debug_assert_eq!(
            staged.base_len,
            self.id_of.len(),
            "positions changed under the stage"
        );
        for id in staged.ids {
            if self.of_id.len() <= id as usize {
                self.of_id.resize(id as usize + 1, NONE);
            }
            self.of_id[id as usize] = self.id_of.len() as Pos;
            self.id_of.push(id);
        }
        self.owner.extend(staged.owners);
        self.last_owner = staged.last_owner;
    }
}

/// The queryable codegraph. Owns the artifact lanes, the symbol interner and
/// position table, the tile map and the cube; relations were dropped after
/// the cube took them.
pub struct Codegraph {
    lanes: Lanes,
    symbols: Symbols,
    pos: Positions,
    /// Status by crate-name interner id.
    crates: HashMap<u32, RunStatus>,
    /// The reservation made at open; refresh grows only inside it.
    axis: Axis,
    table: MapTable,
    cube: Cube,
    masks: Masks,
    rconf: OnceCell<(WideRows, WideStats)>,
    rposs: OnceCell<(WideRows, WideStats)>,
    /// Bumped by every committed refresh and every refold: the fence a
    /// [`PreparedRefresh`] is checked against. Local and in memory only; not
    /// an identity, a version or a persisted counter.
    generation: u64,
    /// Test-only: make the next refresh's validation refuse at a chosen
    /// point, so the falsifiers can show a refusal changes nothing.
    #[cfg(test)]
    inject: Option<Inject>,
}

/// Where a test makes a refresh's validation refuse.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inject {
    /// As if `Cube::stage` refused the batch.
    Stage,
    /// As if `Cube::stage_closure` refused.
    Closure,
}

fn set(words: &mut [u64], p: Pos) {
    if let Some(w) = words.get_mut(p as usize / 64) {
        *w |= 1 << (p % 64);
    }
}

/// Assign positions to the symbols of `rows` in row order. Symbols already
/// positioned keep theirs, so running this over a lane set that gained rows
/// only ever appends. Returns whether a row had an `UNRESOLVED_CALL` (the
/// caller then adds the `?dynamic` position last).
fn resolve(
    lanes: &Lanes,
    rows: std::ops::Range<usize>,
    symbols: &mut impl Interner,
    pos: &mut PosStage<'_>,
) -> bool {
    let mut any_dynamic = false;
    for i in rows {
        let s = lanes.subject[i];
        match lanes.pred[i] {
            P_DEFINES | P_CALLS | P_PUBLIC | P_IMPLEMENTS | P_SYNTHETIC | P_REFERENCES => {
                pos.assign(s, symbols);
                pos.assign(lanes.object[i], symbols);
            }
            P_MAY => {
                pos.assign(s, symbols);
                pos.assign(lanes.may_target[i], symbols);
            }
            // The caller is a symbol; the object ("dynamic") is not.
            P_UNRESOLVED => {
                pos.assign(s, symbols);
                any_dynamic = true;
            }
            _ => {}
        }
    }
    any_dynamic
}

/// Add the `?dynamic` position (idempotent).
fn add_dynamic(symbols: &mut impl Interner, pos: &mut PosStage<'_>) {
    let d = symbols.intern(DYNAMIC);
    pos.assign(d, symbols);
}

fn dynamic_pos(symbols: &impl Interner, pos: &impl PosAt) -> Option<Pos> {
    symbols.get(DYNAMIC).and_then(|id| pos.at(id))
}

fn crate_statuses(lanes: &Lanes, symbols: &Symbols) -> HashMap<u32, RunStatus> {
    let complete = symbols.get("COMPLETE");
    let mut crates = HashMap::new();
    for i in 0..lanes.len() {
        if lanes.pred[i] == P_CRATE {
            let st = if Some(lanes.object[i]) == complete {
                RunStatus::Complete
            } else {
                RunStatus::Partial
            };
            crates.insert(lanes.subject[i], st);
        }
    }
    crates
}

/// The five edge relations (everything but `NODE`, which is left empty) of
/// the given rows, in row order. Intake and refresh both build edges here, so
/// they agree. Every symbol of these rows must already be positioned.
fn edges_of(
    lanes: &Lanes,
    rows: impl Iterator<Item = usize>,
    pos: &impl PosAt,
    dyn_pos: Option<Pos>,
) -> [Vec<(u32, u32)>; 6] {
    let mut rel: [Vec<(u32, u32)>; 6] = Default::default();
    let p = |id: u32| pos.at(id).unwrap_or(NONE);
    for i in rows {
        match lanes.pred[i] {
            P_CALLS => {
                let (s, o) = (p(lanes.subject[i]), p(lanes.object[i]));
                rel[CALLS as usize].push((s, o));
                rel[RCONF as usize].push((o, s));
                rel[RPOSS as usize].push((o, s));
            }
            P_IMPLEMENTS => {
                let (s, o) = (p(lanes.subject[i]), p(lanes.object[i]));
                rel[IMPL as usize].push((s, o));
                rel[RPOSS as usize].push((s, o));
            }
            P_MAY => {
                let (s, o) = (p(lanes.subject[i]), p(lanes.may_target[i]));
                rel[MAY as usize].push((s, o));
                rel[RPOSS as usize].push((o, s));
            }
            P_UNRESOLVED => {
                if let Some(d) = dyn_pos {
                    rel[RPOSS as usize].push((d, p(lanes.subject[i])));
                }
            }
            _ => {}
        }
    }
    rel
}

/// Position masks of a lane set.
fn masks_of(
    lanes: &Lanes,
    pos: &Positions,
    n_ids: usize,
    crates: &HashMap<u32, RunStatus>,
) -> Masks {
    let words = pos.len().div_ceil(64);
    let mut m = Masks {
        fns: vec![0; words],
        public: vec![0; words],
        impl_subjects: vec![0; words],
        synthetic: vec![0; words],
        referenced: vec![0; words],
        called: vec![0; words],
        may_targets: vec![0; words],
        dynamic_callers: vec![0; words],
        stored_owner: vec![0; words],
    };
    let p = |id: u32| pos.at(id).unwrap_or(NONE);
    for i in 0..lanes.len() {
        let (s, o) = (lanes.subject[i], lanes.object[i]);
        match lanes.pred[i] {
            P_DEFINES => set(&mut m.fns, p(s)),
            P_PUBLIC => set(&mut m.public, p(s)),
            P_SYNTHETIC => set(&mut m.synthetic, p(s)),
            P_REFERENCES => set(&mut m.referenced, p(o)),
            P_CALLS => set(&mut m.called, p(o)),
            P_IMPLEMENTS => set(&mut m.impl_subjects, p(s)),
            P_MAY => set(&mut m.may_targets, p(lanes.may_target[i])),
            P_UNRESOLVED => set(&mut m.dynamic_callers, p(s)),
            _ => {}
        }
    }
    let mut is_crate = vec![false; n_ids];
    for &c in crates.keys() {
        if let Some(f) = is_crate.get_mut(c as usize) {
            *f = true;
        }
    }
    for (q, &owner) in pos.owner.iter().enumerate() {
        if is_crate.get(owner as usize).copied().unwrap_or(false) {
            set(&mut m.stored_owner, q as Pos);
        }
    }
    m
}

/// Sorted edge keys `(src << 32) | dst` of one relation.
fn keys(edges: &[(u32, u32)]) -> Vec<u64> {
    let mut k: Vec<u64> = edges
        .iter()
        .map(|&(s, d)| (u64::from(s) << 32) | u64::from(d))
        .collect();
    k.sort_unstable();
    k
}

fn run_len(v: &[u64], i: usize) -> usize {
    let x = v[i];
    v[i..].iter().take_while(|&&y| y == x).count()
}

/// Multiset difference of two sorted key lists in one merge pass: each key
/// is logged `old - new` times into `gone` or `new - old` times into `came`,
/// ascending. A call graph is a bag: two rows for one `(src, dst)` are two
/// call sites.
fn diff_sorted(old: &[u64], new: &[u64], gone: &mut Vec<u64>, came: &mut Vec<u64>) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < old.len() || j < new.len() {
        let order = match (old.get(i), new.get(j)) {
            (Some(a), Some(b)) => a.cmp(b),
            (Some(_), None) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Greater,
        };
        match order {
            std::cmp::Ordering::Equal => {
                let (ro, rn) = (run_len(old, i), run_len(new, j));
                let key = old[i];
                if ro > rn {
                    gone.extend(std::iter::repeat_n(key, ro - rn));
                } else {
                    came.extend(std::iter::repeat_n(key, rn - ro));
                }
                i += ro;
                j += rn;
            }
            std::cmp::Ordering::Less => {
                let r = run_len(old, i);
                gone.extend(std::iter::repeat_n(old[i], r));
                i += r;
            }
            std::cmp::Ordering::Greater => {
                let r = run_len(new, j);
                came.extend(std::iter::repeat_n(new[j], r));
                j += r;
            }
        }
    }
}

/// What one [`Codegraph::refresh_crate`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefreshStats {
    /// Edges logged as ADD, over the five edge relations (not `NODE`).
    pub edges_added: u64,
    /// Edges logged as RETIRE, over the five edge relations.
    pub edges_retired: u64,
    /// Symbols interned past the last position.
    pub symbols_appended: u64,
    /// Closure rows recomputed (retired edges); 0 when no closure was cached.
    pub rows_recomputed: u64,
}

/// Why a refresh was refused.
#[derive(Debug)]
pub enum RefreshRefusal {
    /// The crate's file could not be read or parsed.
    Intake(IntakeRefusal),
    /// The cube refused an edge, a rebase, or a closure refresh.
    Cube(CubeRefusal),
    /// The new symbols do not fit the reserved tiles; open the directory
    /// afresh. Nothing was changed.
    NeedsReopen {
        /// Why growth in place is impossible.
        reason: &'static str,
    },
    /// A prepared refresh was offered to a graph that changed since it was
    /// prepared. Nothing was changed; prepare again against the graph as it
    /// is now.
    Stale {
        /// The graph generation the change was prepared against.
        prepared_at: u64,
        /// The graph's generation now.
        current: u64,
    },
}

/// A refresh prepared and validated against one graph generation, not yet
/// applied ([`Codegraph::refresh_crate`] is prepare + commit). Owns
/// everything it needs, so the graph is not borrowed while it is held.
pub(crate) struct PreparedRefresh {
    generation: u64,
    names: StagedNames,
    positions: StagedPositions,
    grown: Option<(TileId, LocalMap)>,
    staged: StagedEdges,
    closures: [Option<Option<(WideRows, WideStats)>>; 2],
    kid: u32,
    fresh: Lanes,
    stats: RefreshStats,
}

impl Codegraph {
    /// Read every crate of `dir` (`*.rows.bin`, or `*.rows.tsv` when no
    /// binary file exists for that crate) and build the graph.
    pub fn open(dir: &Path) -> Result<Self, IntakeRefusal> {
        let mut symbols = Symbols::default();
        let lanes = artifact::read_dir(dir, &mut symbols)?;
        artifact::refuse_placeholders(&lanes, 0..lanes.len(), &symbols)?;
        let mut pos = Positions::new();
        let staged = {
            let mut stage = PosStage::new(&pos);
            if resolve(&lanes, 0..lanes.len(), &mut symbols, &mut stage) {
                add_dynamic(&mut symbols, &mut stage);
            }
            stage.finish()
        };
        pos.absorb(staged);
        let crates = crate_statuses(&lanes, &symbols);
        let n = pos.len();
        let mut rel = edges_of(&lanes, 0..lanes.len(), &pos, dynamic_pos(&symbols, &pos));
        rel[NODE as usize] = (0..n as u32).map(|p| (p, p)).collect();
        let m = masks_of(&lanes, &pos, symbols.len(), &crates);

        // Tile t holds the global ids [t*256, min(n, (t+1)*256)). Every tile
        // but the last is full, so tile t's local j sits at cube coordinate
        // t*256 + j == the global id == the symbol's position. The masks and
        // `symbol()` rely on coordinate == position.
        let axis = Axis::for_survivors(n);
        let mut changes = Vec::new();
        let mut tile_ids: Vec<TileId> = Vec::new();
        for t in 0..axis.tiles {
            let lo = t as usize * TILE_POSITIONS;
            let hi = n.min(lo + TILE_POSITIONS);
            debug_assert!(
                axis.tiles == 1 || t + 1 == axis.tiles || hi - lo == TILE_POSITIONS,
                "only the last tile may be partly filled"
            );
            let ids: Vec<u64> = (lo as u64..hi as u64).collect();
            let map = LocalMap::prepare(axis.tile, &ids).map_err(IntakeRefusal::Map)?;
            changes.push((t, Some(map)));
            tile_ids.push(t);
        }
        debug_assert!(axis.tiles == 1 || axis.tile == Width::W256);
        let mut table = MapTable::new(1, 1);
        table.advance(1, 1, changes);
        let frame = table.seal();
        let slices: Vec<&[(u32, u32)]> = rel.iter().map(Vec::as_slice).collect();
        let cube = if axis.tiles == 1 {
            Cube::new(&table, &frame, 0, &slices)
        } else {
            Cube::new_tiled(&table, &frame, &tile_ids, &slices)
        }
        .map_err(IntakeRefusal::Cube)?;

        Ok(Self {
            lanes,
            symbols,
            pos,
            crates,
            axis,
            table,
            cube,
            masks: m,
            rconf: OnceCell::new(),
            rposs: OnceCell::new(),
            generation: 0,
            #[cfg(test)]
            inject: None,
        })
    }

    /// Re-read `<krate>.rows.bin` (or `<krate>.rows.tsv` when no binary file
    /// exists) from `dir` and bring the resident graph up
    /// to date without rebuilding it ("sparse write, merge on read").
    ///
    /// * **Digest skip.** The FNV-1a digest of every crate file is recorded
    ///   when it is read (at open, at each successful refresh). A file whose
    ///   digest equals the recorded one is unchanged, which is no delta:
    ///   the call returns `RefreshStats::default()` at once, without parsing
    ///   and without moving the crate's rows. (Before this, an unchanged
    ///   crate cost 25-36 ms to find nothing.)
    /// * Rows are attributed to their crate file by the `krate` lane
    ///   ([`Lanes::krate`]); the crate's old rows are dropped from the lanes
    ///   and the new file's rows appended. Only that one file is parsed, into
    ///   the graph's shared interner (allocating for new names only).
    /// * Positions never renumber. A vanished symbol keeps its position (and
    ///   its `NODE` edge) with no rows; new symbols are appended. Growth must
    ///   fit the positions reserved at open (`Axis::for_survivors(n).reserved()`):
    ///   a new tile, or a W64 tile that would become W256, is
    ///   [`RefreshRefusal::NeedsReopen`] and changes nothing. Growth inside
    ///   the reservation extends the last tile's map under a new frame
    ///   (`MapTable::advance` + `seal`) and `Cube::rebase`s onto it.
    /// * Edges: the crate's old edges and its new edges are compared per
    ///   relation as multisets of sorted `(src << 32) | dst` keys in one merge
    ///   pass (a call graph is a bag: two call sites of one pair are two
    ///   edges). Every other crate's rows are in both the old and the new
    ///   graph, so this equals the whole-graph difference: an edge a
    ///   *different* crate still produces is never retired and a second call
    ///   site is never lost. No string is hashed or compared. The difference is
    ///   logged with `retire_edge` / `add_edge`; the resident lanes are
    ///   untouched. **Apply order** (so the next tile/row is predictable):
    ///   relation by relation in id order; within a relation, retirements
    ///   then additions, each ascending by `(src tile, src, dst)` (tile =
    ///   src / 256, so plain `(src, dst)` order); `NODE` adds for appended
    ///   symbols ascending by id; the appended ids' global-to-local lookups
    ///   (inside `add_edge` and the map `prepare`) are therefore ascending.
    /// * A cached `RCONF`/`RPOSS` closure is brought up to date with
    ///   `Cube::refresh_closure` only when that relation's log was empty
    ///   before this call (its precondition is a closure of the *base*
    ///   lanes; a cached closure already includes the log). Otherwise the
    ///   cache is dropped and rebuilt on next read by `closure_wide`, which
    ///   folds base + log. No cache means nothing to refresh.
    /// * Masks and crate statuses are recomputed from the lanes.
    ///
    /// **Atomic: prepare → validate → commit.** Every step that can refuse
    /// (reading the file, interning, assigning positions, the reservation,
    /// the grown tile map, placing every logged edge, the closure refresh)
    /// runs against staged state: a [`SymbolStage`] and a [`PosStage`] over
    /// the graph's interner and positions, a grown [`LocalMap`] the table has
    /// not yet been given, and a batch [`Cube::stage`] checked without
    /// changing the cube. A refusal therefore changes nothing: not the
    /// interner, the positions, `last_owner`, the table (no version is
    /// issued and no frame sealed), the cube, its logs, the lanes, the masks
    /// or the digests. Only then does the commit run, and it has no error
    /// path: the table advances and seals, the cube adopts the frame and the
    /// batch, the stages are absorbed, the closures and lanes replaced.
    pub fn refresh_crate(
        &mut self,
        dir: &Path,
        krate: &str,
    ) -> Result<RefreshStats, RefreshRefusal> {
        match self.prepare_refresh(dir, krate)? {
            None => Ok(RefreshStats::default()),
            Some(prepared) => self.commit_refresh(prepared),
        }
    }

    /// Prepare and validate a refresh of `krate` without changing the graph:
    /// `None` when the file is unchanged (digest skip), otherwise the
    /// complete, checked change as an owned [`PreparedRefresh`]. Dropping it
    /// discards the change; nothing was written.
    pub(crate) fn prepare_refresh(
        &self,
        dir: &Path,
        krate: &str,
    ) -> Result<Option<PreparedRefresh>, RefreshRefusal> {
        // An unchanged file is no delta. The digest recorded when the crate
        // was last read (at open or at a successful refresh) decides, before
        // any parsing. A binary file carries that digest in its 64-byte
        // header, so the common "nothing happened" refresh reads 64 bytes; a
        // text file costs one read and one pass over the bytes. A crate first
        // read from text and later refreshed from binary has a different
        // digest and takes the full path.
        let mut syms = SymbolStage::new(&self.symbols);
        let mut fresh = Lanes::default();
        // The residuals belong to the same import as the rows but live in
        // their own file, so they can change while the rows do not. Read them
        // first: the unchanged-rows shortcut may only skip a refresh whose
        // residuals are unchanged too. A refusal here leaves the graph as is.
        let residuals = artifact::read_residuals(dir, krate).map_err(RefreshRefusal::Intake)?;
        let unchanged = |digest| {
            self.symbols.get(krate).is_some_and(|k| {
                self.lanes.digest_of(k) == Some(digest)
                    && self.lanes.residuals_of(k) == Some(residuals)
            })
        };
        match artifact::crate_source(dir, krate) {
            Some(CrateSource::Bin(path)) => {
                let mut f = std::fs::File::open(&path)
                    .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Io(e)))?;
                let header = Header::read(&mut f)
                    .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Io(e)))?
                    .map_err(|r| RefreshRefusal::Intake(IntakeRefusal::Bin(r)))?;
                if unchanged(header.digest) {
                    return Ok(None);
                }
                drop(f);
                let buf = std::fs::read(&path)
                    .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Io(e)))?;
                let file = CrateFile::parse(buf)
                    .map_err(|r| RefreshRefusal::Intake(IntakeRefusal::Bin(r)))?;
                artifact::read_bin_as(&file, krate, &mut syms, &mut fresh)
                    .map_err(RefreshRefusal::Intake)?;
            }
            Some(CrateSource::Tsv(path)) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Io(e)))?;
                let file_digest = artifact::digest(text.as_bytes());
                if unchanged(file_digest) {
                    return Ok(None);
                }
                artifact::read_text_as(&text, krate, &mut syms, &mut fresh)
                    .map_err(RefreshRefusal::Intake)?;
            }
            None => {
                return Err(RefreshRefusal::Intake(IntakeRefusal::Io(
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no {krate}.rows.bin or {krate}.rows.tsv in the directory"),
                    ),
                )));
            }
        }
        artifact::refuse_placeholders(&fresh, 0..fresh.len(), &syms)
            .map_err(RefreshRefusal::Intake)?;
        // ---- prepare: everything below borrows the graph immutably ----
        let kid = syms.intern(krate);
        fresh.set_residuals(kid, residuals);

        // Edges of the crate's old rows, before any new position exists. Every
        // other crate's rows are in both the old and the new lane set, so the
        // whole-graph multiset difference equals this crate's old rows against
        // its new rows.
        let old_dyn = dynamic_pos(&syms, &self.pos);
        let old_e = edges_of(
            &self.lanes,
            (0..self.lanes.len()).filter(|&i| self.lanes.krate[i] == kid),
            &self.pos,
            old_dyn,
        );

        // New symbols come only from the fresh rows (all other rows are
        // already positioned), appended in fresh-row order, `?dynamic` last:
        // the same order a rebuilt lane set would intern in.
        let n_old = self.pos.len();
        let mut pos = PosStage::new(&self.pos);
        if resolve(&fresh, 0..fresh.len(), &mut syms, &mut pos) {
            add_dynamic(&mut syms, &mut pos);
        }
        let n_new = pos.len();
        if n_new > self.axis.reserved() {
            return Err(RefreshRefusal::NeedsReopen {
                reason: "new symbols exceed the positions reserved at open (a new tile, or a W64 tile becoming W256)",
            });
        }

        let new_e = edges_of(&fresh, 0..fresh.len(), &pos, dynamic_pos(&syms, &pos));
        // The batch in apply order: relation by relation in id order; NODE
        // adds for the appended positions; otherwise retirements then
        // additions, each ascending by (src, dst).
        let mut ops: Vec<(RelId, EdgeOp, u32, u32)> = Vec::new();
        let mut stats = RefreshStats {
            symbols_appended: (n_new - n_old) as u64,
            ..RefreshStats::default()
        };
        for r in 0..6usize {
            let rel = r as RelId;
            if rel == NODE {
                ops.extend((n_old as u32..n_new as u32).map(|p| (NODE, EdgeOp::Add, p, p)));
                continue;
            }
            let (mut gone, mut came) = (Vec::new(), Vec::new());
            diff_sorted(&keys(&old_e[r]), &keys(&new_e[r]), &mut gone, &mut came);
            stats.edges_retired += gone.len() as u64;
            stats.edges_added += came.len() as u64;
            let split = |k: u64| ((k >> 32) as u32, k as u32);
            ops.extend(
                gone.into_iter()
                    .map(split)
                    .map(|(s, d)| (rel, EdgeOp::Retire, s, d)),
            );
            ops.extend(
                came.into_iter()
                    .map(split)
                    .map(|(s, d)| (rel, EdgeOp::Add, s, d)),
            );
        }

        // The last tile's grown map (ids ascending), not yet in the table.
        let last = self.axis.tiles.saturating_sub(1);
        let grown = if n_new > n_old {
            let lo = last as u64 * TILE_POSITIONS as u64;
            let ids: Vec<u64> = (lo..n_new as u64).collect();
            Some(
                LocalMap::prepare(self.axis.tile, &ids)
                    .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Map(e)))?,
            )
        } else {
            None
        };

        // ---- validate: the cube checks the batch without changing ----
        let staged = self
            .stage_batch(grown.as_ref().map(|m| (last, m)), &ops)
            .map_err(RefreshRefusal::Cube)?;
        let changed = !staged.is_empty();
        // A cached closure is refreshed only when its relation's log was
        // empty (its precondition is a closure of the base); otherwise it is
        // dropped and rebuilt on next read. Computed now, so a refusal here
        // also changes nothing.
        let mut closures: [Option<Option<(WideRows, WideStats)>>; 2] = [None, None];
        if changed {
            for (slot, rel) in [(0usize, RCONF), (1, RPOSS)] {
                let cell = if rel == RCONF {
                    &self.rconf
                } else {
                    &self.rposs
                };
                if let Some(old) = cell.get() {
                    closures[slot] = Some(if self.cube.delta_len(rel) == 0 {
                        Some(
                            self.stage_closure(rel, &old.0, &staged)
                                .map_err(RefreshRefusal::Cube)?,
                        )
                    } else {
                        None
                    });
                }
            }
        }
        Ok(Some(PreparedRefresh {
            generation: self.generation,
            names: syms.into_names(),
            positions: pos.finish(),
            grown: grown.map(|m| (last, m)),
            staged,
            closures,
            kid,
            fresh,
            stats,
        }))
    }

    /// Apply a [`PreparedRefresh`]. The one check comes first and changes
    /// nothing: the graph must still be at the generation the change was
    /// prepared against ([`RefreshRefusal::Stale`]); a change prepared
    /// against an older graph would hand out positions the graph has since
    /// given to other symbols. After the check there is no error path.
    pub(crate) fn commit_refresh(
        &mut self,
        prepared: PreparedRefresh,
    ) -> Result<RefreshStats, RefreshRefusal> {
        if prepared.generation != self.generation {
            return Err(RefreshRefusal::Stale {
                prepared_at: prepared.generation,
                current: self.generation,
            });
        }
        let PreparedRefresh {
            names,
            positions,
            grown,
            staged,
            mut closures,
            kid,
            fresh,
            mut stats,
            ..
        } = prepared;

        // ---- commit: no error path below ----
        self.generation += 1;
        self.symbols.absorb(names);
        self.pos.absorb(positions);
        let frame = match grown {
            Some((last, map)) => {
                let rev = self.table.revisions();
                self.table
                    .advance(rev.source + 1, rev.relation_map, vec![(last, Some(map))]);
                self.table.seal()
            }
            None => self.cube.frame().clone(),
        };
        let changed = !staged.is_empty();
        self.cube.commit(&self.table, &frame, staged);
        for (slot, cell) in [(0usize, &mut self.rconf), (1, &mut self.rposs)] {
            match closures[slot].take() {
                // Prepared against the cache that was there: replace it.
                Some(Some((rows, st))) => {
                    cell.take();
                    stats.rows_recomputed += st.rows_recomputed;
                    let _ = cell.set((rows, st));
                }
                // The cache was there but could not be refreshed: drop it.
                Some(None) => {
                    cell.take();
                }
                // No cache at preparation. A read between prepare and commit
                // may have filled one (`closure` / `impact` take `&self`) from
                // the pre-commit edges; the edges change now, so it is stale.
                None if changed => {
                    cell.take();
                }
                None => {}
            }
        }

        // Drop the crate's old rows, then append the fresh ones (the crate
        // moves to the end of the lanes) and adopt the new file digest.
        self.lanes.drop_crate(kid);
        self.lanes.append(fresh);
        self.crates = crate_statuses(&self.lanes, &self.symbols);
        self.masks = masks_of(&self.lanes, &self.pos, self.symbols.len(), &self.crates);
        Ok(stats)
    }

    /// [`Cube::stage`], or the refusal a test injected in its place.
    fn stage_batch(
        &self,
        grown: Option<(TileId, &LocalMap)>,
        ops: &[(RelId, EdgeOp, u32, u32)],
    ) -> Result<StagedEdges, CubeRefusal> {
        #[cfg(test)]
        if self.inject == Some(Inject::Stage) {
            return Err(CubeRefusal::Renumbered(0));
        }
        self.cube.stage(&self.table, grown, ops)
    }

    /// [`Cube::stage_closure`], or the refusal a test injected in its place.
    fn stage_closure(
        &self,
        rel: RelId,
        old: &WideRows,
        staged: &StagedEdges,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        #[cfg(test)]
        if self.inject == Some(Inject::Closure) {
            return Err(CubeRefusal::FixpointBound);
        }
        self.cube.stage_closure(rel, old, staged)
    }

    /// Absorb every relation's delta log into its base lanes (explicit only;
    /// nothing calls this on a threshold). Cached closures stay valid: the
    /// base now equals the effective edges they were computed from.
    pub fn refold(&mut self) -> Result<(), CubeRefusal> {
        // A prepared refresh staged its closures against the logs; folding
        // them into the base makes it stale.
        self.generation += 1;
        for rel in 0..6 {
            self.cube.refold(rel)?;
        }
        Ok(())
    }

    /// The position of `symbol`, if it is a symbol of the graph: one interner
    /// lookup, then one array read.
    pub fn pos(&self, symbol: &str) -> Option<Pos> {
        self.symbols.get(symbol).and_then(|id| self.pos.at(id))
    }

    /// The symbol at position `p`.
    pub fn symbol(&self, p: Pos) -> Option<&str> {
        self.id_of(p).map(|id| self.symbols.name(id))
    }

    /// Interner id of the symbol at position `p`.
    pub(crate) fn id_of(&self, p: Pos) -> Option<u32> {
        self.pos.id_of.get(p as usize).copied()
    }

    /// Position of interner id `id`, when that text is a symbol.
    pub(crate) fn pos_of_id(&self, id: u32) -> Option<Pos> {
        self.pos.at(id)
    }

    /// The artifact lanes, for the crate's own folds.
    pub(crate) fn lanes(&self) -> &Lanes {
        &self.lanes
    }

    /// Stored crates whose CRATE row is not `COMPLETE`.
    pub(crate) fn partial_crates(&self) -> u32 {
        self.crates
            .values()
            .filter(|&&s| s == RunStatus::Partial)
            .count() as u32
    }

    /// Names of the crates with a CRATE row: the first path segment of the
    /// symbols each crate defines.
    pub(crate) fn stored_crates(&self) -> std::collections::BTreeSet<&str> {
        self.crates
            .keys()
            .map(|&id| self.symbols.name(id))
            .collect()
    }

    /// Residual counts of the crate named `krate`, and whether no residuals
    /// file was read for it. A name the store did not read under is reported
    /// as having no file: unknown, never clean.
    pub(crate) fn residuals_of_crate(&self, krate: &str) -> (artifact::ResidualCounts, bool) {
        match self
            .symbols
            .get(krate)
            .and_then(|id| self.lanes.residuals_of(id))
        {
            Some(Some(r)) => (r, false),
            _ => (artifact::ResidualCounts::default(), true),
        }
    }

    /// Whether any stored crate's CRATE row is not `COMPLETE`.
    pub(crate) fn any_partial(&self) -> bool {
        self.crates.values().any(|&s| s == RunStatus::Partial)
    }

    /// Mask words per position mask: `n.div_ceil(64)`.
    pub fn words(&self) -> usize {
        self.pos.len().div_ceil(64)
    }

    /// A mask with the bits of `symbols` set; unknown symbols are ignored.
    pub fn mask_of(&self, symbols: &[&str]) -> Vec<u64> {
        let mut w = vec![0u64; self.words()];
        for s in symbols {
            if let Some(p) = self.pos(s) {
                set(&mut w, p);
            }
        }
        w
    }

    /// The extractor status of `krate`, from its CRATE row.
    pub fn crate_status(&self, krate: &str) -> Option<RunStatus> {
        let id = self.symbols.get(krate)?;
        self.crates.get(&id).copied()
    }

    /// Data rows read.
    pub fn rows(&self) -> usize {
        self.lanes.len()
    }

    /// Predicate of `row`; panics only on an out-of-range row, like indexing.
    pub fn row_predicate(&self, row: usize) -> &str {
        self.lanes.predicate_name(row, &self.symbols)
    }

    /// Subject of `row`.
    pub fn row_subject(&self, row: usize) -> &str {
        self.symbols.name(self.lanes.subject[row])
    }

    /// Object of `row`, verbatim from the artifact.
    pub fn row_object(&self, row: usize) -> &str {
        self.symbols.name(self.lanes.object[row])
    }

    /// `(file, start, end)` of `row`, `None` when the row has no site (`-`).
    pub fn site(&self, row: usize) -> Option<(&str, u32, u32)> {
        let (f, s, e) = (
            self.symbols.name(*self.lanes.file.get(row)?),
            self.lanes.start[row],
            self.lanes.end[row],
        );
        if f == "-" || s == NO_SPAN || e == NO_SPAN {
            None
        } else {
            Some((f, s, e))
        }
    }

    /// The cube holding the six relations.
    pub fn cube(&self) -> &Cube {
        &self.cube
    }

    /// The cube, mutably (views are built through `&mut`).
    pub fn cube_mut(&mut self) -> &mut Cube {
        &mut self.cube
    }

    /// The cube and the map table it reads under, borrowed together: views are
    /// built through `&mut Cube` and every view constructor needs the table.
    pub(crate) fn cube_and_table_mut(&mut self) -> (&mut Cube, &MapTable) {
        (&mut self.cube, &self.table)
    }

    /// The tile map table the cube reads under.
    pub fn table(&self) -> &MapTable {
        &self.table
    }

    /// Subjects of DEFINES rows.
    pub fn fns(&self) -> &[u64] {
        &self.masks.fns
    }
    /// Subjects of PUBLIC rows.
    pub fn public(&self) -> &[u64] {
        &self.masks.public
    }
    /// Subjects of IMPLEMENTS rows.
    pub fn impl_subjects(&self) -> &[u64] {
        &self.masks.impl_subjects
    }
    /// Subjects of SYNTHETIC rows.
    pub fn synthetic(&self) -> &[u64] {
        &self.masks.synthetic
    }
    /// Objects of REFERENCES rows.
    pub fn referenced(&self) -> &[u64] {
        &self.masks.referenced
    }
    /// Objects of CALLS_RESOLVED rows.
    pub fn called(&self) -> &[u64] {
        &self.masks.called
    }
    /// `Trait::method` objects of MAY_DISPATCH rows.
    pub fn may_targets(&self) -> &[u64] {
        &self.masks.may_targets
    }
    /// Subjects of UNRESOLVED_CALL rows.
    pub fn dynamic_callers(&self) -> &[u64] {
        &self.masks.dynamic_callers
    }
    /// Symbols whose first path segment names a crate with a CRATE row.
    pub fn stored_owner(&self) -> &[u64] {
        &self.masks.stored_owner
    }

    /// `edge⁺` of [`RCONF`] or [`RPOSS`], computed once and cached. Any other
    /// relation is `UnknownRel`. Any number of positions closes.
    pub fn closure(&self, rel: RelId) -> Result<&(WideRows, WideStats), CubeRefusal> {
        let cell = match rel {
            RCONF => &self.rconf,
            RPOSS => &self.rposs,
            other => return Err(CubeRefusal::UnknownRel(other)),
        };
        if let Some(c) = cell.get() {
            return Ok(c);
        }
        let computed = self.cube.closure_wide(&self.table, rel)?;
        Ok(cell.get_or_init(|| computed))
    }
}

/// P1 falsifiers: a refused refresh changes nothing, a retry equals the
/// refresh that never failed, and multiplicities survive. They live here
/// (not in `tests/`) because the refusals are injected through the
/// test-only [`Inject`] hook and the snapshot reads private state.
#[cfg(test)]
mod atomic {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Everything a refresh may touch, in a comparable form: interner,
    /// positions (incl. `last_owner`), lanes and digests, crate statuses,
    /// table version, revisions and per-tile history, the cube's frame and
    /// every delta log, masks, and the cached closures.
    pub(super) fn snapshot(g: &Codegraph) -> String {
        let names: Vec<&str> = (0..g.symbols.len() as u32)
            .map(|i| g.symbols.name(i))
            .collect();
        let mut crates: Vec<(u32, RunStatus)> = g.crates.iter().map(|(&k, &v)| (k, v)).collect();
        crates.sort_by_key(|c| c.0);
        let tiles: Vec<(usize, usize)> = (0..g.axis.tiles)
            .map(|t| (g.table.generation_count(t), g.table.event_count(t)))
            .collect();
        let logs: Vec<_> = (0..6).map(|r| g.cube.delta_log(r).cloned()).collect();
        format!(
            "gen {}\n{names:?}\n{:?}\n{:?}\n{crates:?}\n{:?} {:?} {tiles:?}\n{:?}\n{logs:?}\n{:?}\n{:?}\n{:?}",
            g.generation,
            g.pos,
            g.lanes,
            g.table.current_version(),
            g.table.revisions(),
            g.cube.frame(),
            g.masks,
            g.rconf.get(),
            g.rposs.get(),
        )
    }

    fn dir(tag: &str) -> PathBuf {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("cg_atomic_{}_{tag}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("temp dir");
        d
    }

    const HEAD: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";

    /// v1: `m::a` calls `m::b` at two sites. v2: three sites, and a new
    /// symbol `m::c` that also calls `m::b` (growth inside the reservation).
    fn rows(v: u8) -> String {
        let mut s = String::from(HEAD);
        let mut def = |f: &str, at: u32| {
            s.push_str(&format!(
                "DEFINES\t{f}\t{f}\tCOMPILER\tm/src/lib.rs\t{at}\t{}\n",
                at + 1
            ));
        };
        def("m::a", 0);
        def("m::b", 10);
        if v == 2 {
            def("m::c", 20);
        }
        let sites: &[(&str, u32)] = if v == 1 {
            &[("m::a", 2), ("m::a", 4)]
        } else {
            &[("m::a", 2), ("m::a", 4), ("m::a", 6), ("m::c", 22)]
        };
        for &(f, at) in sites {
            s.push_str(&format!(
                "CALLS_RESOLVED\t{f}\tm::b\tCOMPILER\tm/src/lib.rs\t{at}\t{}\n",
                at + 1
            ));
        }
        s
    }

    /// A graph opened at v1 with both closures cached (so the refresh has a
    /// closure to stage), and the directory now holding v2.
    fn primed(tag: &str) -> (PathBuf, Codegraph) {
        let d = dir(tag);
        fs::write(d.join("m.rows.tsv"), rows(1)).expect("write");
        let g = Codegraph::open(&d).expect("open");
        g.closure(RCONF).expect("rconf");
        g.closure(RPOSS).expect("rposs");
        fs::write(d.join("m.rows.tsv"), rows(2)).expect("write");
        (d, g)
    }

    /// The snapshot can fire: a successful refresh changes it. Without this
    /// the "unchanged" assertions below could pass on a blind snapshot.
    #[test]
    fn a_successful_refresh_changes_the_snapshot() {
        let (d, mut g) = primed("fires");
        let before = snapshot(&g);
        let st = g.refresh_crate(&d, "m").expect("refresh");
        assert!(st.symbols_appended > 0 && st.edges_added > 0, "{st:?}");
        assert_ne!(snapshot(&g), before);
    }

    /// Red if a refused reservation leaves anything behind: the interner
    /// entries of the new names, a position, `last_owner` (disable: absorb
    /// the staged names and positions before the reservation check).
    #[test]
    fn a_refused_reservation_changes_nothing() {
        let d = dir("reopen");
        let big = |n: usize| {
            let mut s = String::from(HEAD);
            for i in 0..n {
                s.push_str(&format!(
                    "DEFINES\tbig::f{i}\tbig::f{i}\tCOMPILER\tbig/src/lib.rs\t{i}\t{}\n",
                    i + 1
                ));
            }
            s
        };
        fs::write(d.join("big.rows.tsv"), big(64)).expect("write");
        let mut g = Codegraph::open(&d).expect("open");
        fs::write(d.join("big.rows.tsv"), big(65)).expect("write");
        // Anti-vacuity: the refused file carries a name the interner lacks.
        assert_eq!(g.symbols.get("big::f64"), None);
        let before = snapshot(&g);
        match g.refresh_crate(&d, "big") {
            Err(RefreshRefusal::NeedsReopen { .. }) => {}
            other => panic!("expected NeedsReopen, got {other:?}"),
        }
        assert_eq!(g.symbols.get("big::f64"), None, "interner leak");
        assert_eq!(snapshot(&g), before);
    }

    /// Red if a refusal from `Cube::stage` leaves the table advanced, a
    /// frame sealed or anything else changed (disable: advance and seal the
    /// table before staging the batch).
    #[test]
    fn an_injected_stage_refusal_changes_nothing() {
        let (d, mut g) = primed("stage");
        let before = snapshot(&g);
        let v = g.table.current_version();
        g.inject = Some(Inject::Stage);
        match g.refresh_crate(&d, "m") {
            Err(RefreshRefusal::Cube(_)) => {}
            other => panic!("expected the injected Cube refusal, got {other:?}"),
        }
        assert_eq!(g.table.current_version(), v, "a version was issued");
        assert_eq!(snapshot(&g), before);
    }

    /// Red if a refusal while staging a closure refresh comes after any
    /// commit step (disable: absorb the staged names and positions before
    /// the closure block).
    #[test]
    fn an_injected_closure_refusal_changes_nothing() {
        let (d, mut g) = primed("closure");
        // Anti-vacuity: both closures are cached over empty logs, so the
        // refresh does stage them.
        assert!(g.rconf.get().is_some() && g.cube.delta_len(RCONF) == 0);
        let before = snapshot(&g);
        g.inject = Some(Inject::Closure);
        assert!(matches!(
            g.refresh_crate(&d, "m"),
            Err(RefreshRefusal::Cube(_))
        ));
        assert_eq!(snapshot(&g), before);
    }

    /// Red if a retry after a refusal differs from the refresh that never
    /// failed: a second version, a doubled ADD, a doubled name (disable:
    /// advance and seal the table before staging). And red if the bag is
    /// lost: `m::b` has four call sites, as in a clean open of v2 (disable:
    /// deduplicate the keys before `diff_sorted`).
    #[test]
    fn a_retry_after_a_refusal_equals_the_refresh_that_never_failed() {
        let (d, mut g) = primed("retry");
        let (_, mut twin) = primed("twin");
        twin.refresh_crate(&d, "m").expect("twin refresh");
        for inject in [Inject::Stage, Inject::Closure] {
            g.inject = Some(inject);
            assert!(g.refresh_crate(&d, "m").is_err());
        }
        g.inject = None;
        g.refresh_crate(&d, "m").expect("retry");
        assert_eq!(snapshot(&g), snapshot(&twin));

        let clean_dir = dir("clean");
        fs::write(clean_dir.join("m.rows.tsv"), rows(2)).expect("write");
        let mut clean = Codegraph::open(&clean_dir).expect("open clean");
        let (pb, cb) = (g.pos("m::b").expect("b"), clean.pos("m::b").expect("b"));
        let (got, want) = (g.callers(pb).expect("q"), clean.callers(cb).expect("q"));
        assert_eq!(want.sites, 4, "fixture: four call sites of m::b");
        assert_eq!(got.sites, want.sites, "bag multiplicity");
    }
}

/// P4 lab falsifiers F-LAND and F-DISCARD (operator decision 2026-11-02):
/// two branches over one W0 base, each a revision of one source; landing
/// recomputes against the current W0; discarding leaves W0 untouched. Lab
/// only: no Thought-Git wiring, no persistence, no routing.
#[cfg(test)]
mod publication;

#[cfg(test)]
mod lifecycle {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    const HEAD: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";

    pub(super) fn dir(tag: &str) -> PathBuf {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("cg_lifecycle_{}_{tag}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("temp dir");
        d
    }

    /// Rows of one source: `defs` defined, `calls` as (caller, callee) sites.
    fn rows(file: &str, defs: &[&str], calls: &[(&str, &str)]) -> String {
        let mut s = String::from(HEAD);
        for (i, f) in defs.iter().enumerate() {
            s.push_str(&format!(
                "DEFINES\t{f}\t{f}\tCOMPILER\t{file}\t{i}\t{}\n",
                i + 1
            ));
        }
        for (i, (a, b)) in calls.iter().enumerate() {
            let at = 100 + i;
            s.push_str(&format!(
                "CALLS_RESOLVED\t{a}\t{b}\tCOMPILER\t{file}\t{at}\t{}\n",
                at + 1
            ));
        }
        s
    }

    // W0: m::a calls m::b at two sites; n::c calls m::b once.
    pub(super) fn m_v1() -> String {
        rows(
            "m.rs",
            &["m::a", "m::b"],
            &[("m::a", "m::b"), ("m::a", "m::b")],
        )
    }
    pub(super) fn n_v1() -> String {
        rows("n.rs", &["n::c"], &[("n::c", "m::b")])
    }
    // Branch A revises source m: a new symbol m::x calling m::b.
    pub(super) fn m_v2() -> String {
        rows(
            "m.rs",
            &["m::a", "m::b", "m::x"],
            &[("m::a", "m::b"), ("m::a", "m::b"), ("m::x", "m::b")],
        )
    }
    // Branch B revises source n: a new symbol n::y; n::c now calls it too.
    pub(super) fn n_v2() -> String {
        rows(
            "n.rs",
            &["n::c", "n::y"],
            &[("n::c", "m::b"), ("n::c", "n::y"), ("n::y", "m::b")],
        )
    }

    pub(super) fn store(tag: &str, m: &str, n: &str) -> PathBuf {
        let d = dir(tag);
        fs::write(d.join("m.rows.tsv"), m).expect("write");
        fs::write(d.join("n.rows.tsv"), n).expect("write");
        d
    }

    fn names(g: &Codegraph, mask: &[u64]) -> Vec<String> {
        let mut v: Vec<String> = (0..mask.len() * 64)
            .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
            .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
            .collect();
        v.sort();
        v
    }

    /// Every symbol's callers (names + call-site count), callees and
    /// impact, by name, so two graphs with different positions compare.
    pub(super) fn answers(g: &mut Codegraph) -> BTreeMap<String, String> {
        let all: Vec<String> = (0..g.words() * 64)
            .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
            .collect();
        let mut out = BTreeMap::new();
        for name in all {
            let p = g.pos(&name).expect("pos");
            let c = g.callers(p).expect("callers");
            let e = g.callees(p).expect("callees");
            let i = g.impact(p).expect("impact");
            out.insert(
                name,
                format!(
                    "callers {:?}/{:?}/{:?} sites {} callees {:?}/{:?}/{:?} impact {:?}/{:?}/{:?}",
                    names(g, &c.graded.confirmed),
                    names(g, &c.graded.possible),
                    names(g, &c.graded.unknown),
                    c.sites,
                    names(g, &e.graded.confirmed),
                    names(g, &e.graded.possible),
                    names(g, &e.graded.unknown),
                    names(g, &i.confirmed),
                    names(g, &i.possible),
                    names(g, &i.unknown),
                ),
            );
        }
        out
    }

    /// F-LAND. Branch A (source m) and branch B (source n) are prepared
    /// against the same W0; both mint a new symbol, so both claim the next
    /// free position. After A lands, B's old preparation is refused (it would
    /// hand A's position to another symbol) and B is recomputed against the
    /// current W0. The result equals a clean open of both revised sources,
    /// call-site counts included. Red if a stale preparation is applied
    /// (disable: drop the generation check in `commit_refresh`).
    #[test]
    fn f_land_recomputes_a_second_branch_against_the_current_w0() {
        let w0 = store("w0", &m_v1(), &n_v1());
        let branch_a = store("a", &m_v2(), &n_v1());
        let branch_b = store("b", &m_v1(), &n_v2());
        let mut g = Codegraph::open(&w0).expect("open W0");
        let base_positions = g.pos.len();

        let a = g
            .prepare_refresh(&branch_a, "m")
            .expect("A")
            .expect("A changes");
        let b_old = g
            .prepare_refresh(&branch_b, "n")
            .expect("B")
            .expect("B changes");
        // The collision the fence exists for: both preparations start their
        // new symbols at the same free position.
        assert_eq!(a.positions.base_len, base_positions);
        assert_eq!(b_old.positions.base_len, base_positions);
        assert!(!a.positions.ids.is_empty() && !b_old.positions.ids.is_empty());

        g.commit_refresh(a).expect("land A");
        let x = g.pos("m::x").expect("m::x landed");
        assert_eq!(
            x as usize, base_positions,
            "A's new symbol took the free position"
        );

        let before = super::atomic::snapshot(&g);
        match g.commit_refresh(b_old) {
            Err(RefreshRefusal::Stale { .. }) => {}
            other => panic!("a stale preparation must be refused, got {other:?}"),
        }
        assert_eq!(
            super::atomic::snapshot(&g),
            before,
            "the refusal changed nothing"
        );

        let b = g
            .prepare_refresh(&branch_b, "n")
            .expect("B again")
            .expect("B changes");
        g.commit_refresh(b).expect("land B");
        let y = g.pos("n::y").expect("n::y landed");
        assert_ne!(x, y, "distinct symbols, distinct positions");

        let both = store("oracle", &m_v2(), &n_v2());
        let mut clean = Codegraph::open(&both).expect("clean open");
        assert_eq!(answers(&mut g), answers(&mut clean));
        // Anti-vacuity: the oracle sees the real multiplicities.
        let pb = clean.pos("m::b").expect("m::b");
        assert_eq!(
            clean.callers(pb).expect("q").sites,
            5,
            "2 + 1 + 1 + 1 call sites"
        );
    }

    /// Codex P2 on #1924. A closure cache filled by a read between prepare and
    /// commit (`impact` takes `&self`) was built from the pre-commit edges.
    /// Red if commit keeps it (disable: drop the `None if changed` arm in
    /// `commit_refresh`): `impact` would then answer from the old graph.
    #[test]
    fn a_closure_cached_between_prepare_and_commit_is_not_served_stale() {
        let w0 = store("w0", &m_v1(), &n_v1());
        let branch_a = store("a", &m_v2(), &n_v1());
        let mut g = Codegraph::open(&w0).expect("open W0");
        assert!(g.rconf.get().is_none() && g.rposs.get().is_none());
        let a = g
            .prepare_refresh(&branch_a, "m")
            .expect("A")
            .expect("A changes");
        let b = g.pos("m::b").expect("m::b");
        let before = g.impact(b).expect("impact");
        // Anti-vacuity: the read did fill a cache the preparation did not see.
        assert!(g.rconf.get().is_some() || g.rposs.get().is_some());
        g.commit_refresh(a).expect("commit");
        let after = g.impact(b).expect("impact");
        let x = g.pos("m::x").expect("m::x");
        let has = |m: &[u64]| {
            m.get(x as usize / 64)
                .is_some_and(|w| w >> (x % 64) & 1 == 1)
        };
        assert!(!has(&before.confirmed), "m::x did not exist before");
        assert!(
            has(&after.confirmed),
            "impact of m::b must include its new caller m::x"
        );
        let mut clean = Codegraph::open(&branch_a).expect("clean");
        assert_eq!(answers(&mut g), answers(&mut clean));
    }

    /// F-DISCARD. A prepared branch that is dropped leaves W0 exactly as it
    /// was: interner, positions, `last_owner`, lanes, digests, the map
    /// table's version and history, the cube's frame and logs, masks,
    /// cached closures and the graph generation. The codegraph holds no
    /// `BindStore`, so DT64 routing is untouched by construction.
    #[test]
    fn f_discard_leaves_w0_unchanged() {
        let w0 = store("w0", &m_v1(), &n_v1());
        let branch_a = store("a", &m_v2(), &n_v1());
        let g = Codegraph::open(&w0).expect("open W0");
        g.closure(RCONF).expect("rconf");
        g.closure(RPOSS).expect("rposs");
        let before = super::atomic::snapshot(&g);
        let version = g.table.current_version();

        let a = g
            .prepare_refresh(&branch_a, "m")
            .expect("A")
            .expect("A changes");
        // Anti-vacuity: the discarded change was real (a new symbol, edges).
        assert!(!a.positions.ids.is_empty() && !a.staged.is_empty());
        drop(a);

        assert_eq!(g.table.current_version(), version, "no version issued");
        assert_eq!(super::atomic::snapshot(&g), before);

        // Twin: committing the same change does move W0, so the snapshot
        // comparison above can fail.
        let mut twin = Codegraph::open(&w0).expect("open twin");
        let a = twin
            .prepare_refresh(&branch_a, "m")
            .expect("A")
            .expect("A changes");
        twin.commit_refresh(a).expect("commit");
        assert_ne!(
            super::atomic::snapshot(&twin),
            super::atomic::snapshot(&Codegraph::open(&w0).expect("w0"))
        );
    }
}
