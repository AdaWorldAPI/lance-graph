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
    self, Interner, Lanes, SymbolStage, Symbols, NO_SPAN, P_CALLS, P_CRATE, P_DEFINES,
    P_IMPLEMENTS, P_MAY, P_PUBLIC, P_REFERENCES, P_SYNTHETIC, P_UNRESOLVED,
};

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
}

impl Codegraph {
    /// Read every `*.rows.tsv` in `dir` and build the graph.
    pub fn open(dir: &Path) -> Result<Self, IntakeRefusal> {
        let mut symbols = Symbols::default();
        let lanes = artifact::read_dir(dir, &mut symbols)?;
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
            #[cfg(test)]
            inject: None,
        })
    }

    /// Re-read `<krate>.rows.tsv` from `dir` and bring the resident graph up
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
        let text = artifact::read_crate_text(dir, krate).map_err(RefreshRefusal::Intake)?;
        // An unchanged file is no delta. The digest recorded when the crate
        // was last read (at open or at a successful refresh) decides, before
        // any parsing, so the common "nothing happened" refresh costs one
        // read and one pass over the bytes.
        let file_digest = artifact::digest(text.as_bytes());
        if let Some(k) = self.symbols.get(krate) {
            if self.lanes.digest_of(k) == Some(file_digest) {
                return Ok(RefreshStats::default());
            }
        }

        // ---- prepare: everything below borrows the graph immutably ----
        let mut syms = SymbolStage::new(&self.symbols);
        let mut fresh = Lanes::default();
        artifact::read_text_as(&text, krate, &mut syms, &mut fresh)
            .map_err(RefreshRefusal::Intake)?;
        let kid = syms.intern(krate);

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
        let names = syms.into_names();
        let positions = pos.finish();

        // ---- commit: no error path below ----
        self.symbols.absorb(names);
        self.pos.absorb(positions);
        let frame = match grown {
            Some(map) => {
                let rev = self.table.revisions();
                self.table
                    .advance(rev.source + 1, rev.relation_map, vec![(last, Some(map))]);
                self.table.seal()
            }
            None => self.cube.frame().clone(),
        };
        self.cube.commit(&self.table, &frame, staged);
        for (slot, cell) in [(0usize, &mut self.rconf), (1, &mut self.rposs)] {
            if let Some(next) = closures[slot].take() {
                cell.take();
                if let Some((rows, st)) = next {
                    stats.rows_recomputed += st.rows_recomputed;
                    let _ = cell.set((rows, st));
                }
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
