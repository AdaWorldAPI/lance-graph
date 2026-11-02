//! The codegraph: artifact rows interned into symbol positions (one 64- or
//! 256-position tile, or several 256-position tiles) and folded into six
//! relations of an xlog [`Cube`].

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::Path;

use lance_graph_dt64::dr::{Axis, LocalMap, MapRefusal, MapTable, TileId, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, RelId};
use lance_graph_xlog::wide::{WideRows, WideStats};

use crate::artifact::{self, Lanes, NO_SPAN};

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
#[derive(Default)]
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

/// The queryable codegraph. Owns the artifact lanes, the symbol table, the
/// tile map and the cube; relations were dropped after the cube took them.
pub struct Codegraph {
    lanes: Lanes,
    names: Vec<String>,
    index: HashMap<String, Pos>,
    crates: HashMap<String, RunStatus>,
    /// The reservation made at open; refresh grows only inside it.
    axis: Axis,
    table: MapTable,
    cube: Cube,
    masks: Masks,
    rconf: OnceCell<(WideRows, WideStats)>,
    rposs: OnceCell<(WideRows, WideStats)>,
}

fn intern(names: &mut Vec<String>, index: &mut HashMap<String, Pos>, s: &str) -> Pos {
    if let Some(&p) = index.get(s) {
        return p;
    }
    let p = names.len() as Pos;
    names.push(s.to_string());
    index.insert(s.to_string(), p);
    p
}

/// The `Trait::method` part of a MAY_DISPATCH object.
fn dispatch_target(object: &str) -> &str {
    object.split(" via ").next().unwrap_or(object)
}

fn set(words: &mut [u64], p: Pos) {
    words[p as usize / 64] |= 1 << (p % 64);
}

/// Row subject/object ids after interning; `NONE` when a row has no symbol.
struct Resolved {
    sid: Vec<u32>,
    oid: Vec<u32>,
    dyn_id: Option<u32>,
}

/// Intern every row's symbols into `names`/`idx` in row order. Symbols already
/// interned keep their position, so running this over a lane set that gained
/// or lost rows only ever appends.
fn resolve(lanes: &Lanes, names: &mut Vec<String>, idx: &mut HashMap<String, Pos>) -> Resolved {
    let rows = lanes.predicate.len();
    let mut sid = vec![NONE; rows];
    let mut oid = vec![NONE; rows];
    let mut any_dynamic = false;
    for i in 0..rows {
        let (s, o) = (lanes.subject[i].as_str(), lanes.object[i].as_str());
        match lanes.predicate[i].as_str() {
            "DEFINES" | "CALLS_RESOLVED" | "PUBLIC" | "IMPLEMENTS" | "SYNTHETIC" | "REFERENCES" => {
                sid[i] = intern(names, idx, s);
                oid[i] = intern(names, idx, o);
            }
            "MAY_DISPATCH" => {
                sid[i] = intern(names, idx, s);
                oid[i] = intern(names, idx, dispatch_target(o));
            }
            // The caller is a symbol; the object ("dynamic") is not.
            "UNRESOLVED_CALL" => {
                sid[i] = intern(names, idx, s);
                any_dynamic = true;
            }
            _ => {}
        }
    }
    let dyn_id = any_dynamic.then(|| intern(names, idx, DYNAMIC));
    Resolved { sid, oid, dyn_id }
}

fn crate_statuses(lanes: &Lanes) -> HashMap<String, RunStatus> {
    let mut crates = HashMap::new();
    for i in 0..lanes.predicate.len() {
        if lanes.predicate[i] == "CRATE" {
            let st = if lanes.object[i] == "COMPLETE" {
                RunStatus::Complete
            } else {
                RunStatus::Partial
            };
            crates.insert(lanes.subject[i].clone(), st);
        }
    }
    crates
}

/// The five edge relations (everything but `NODE`, which is left empty) of a
/// lane set. Intake and refresh both build edges here, so they agree.
fn edges_of(lanes: &Lanes, r: &Resolved) -> [Vec<(u32, u32)>; 6] {
    let mut rel: [Vec<(u32, u32)>; 6] = Default::default();
    for i in 0..lanes.predicate.len() {
        let (s, o) = (r.sid[i], r.oid[i]);
        match lanes.predicate[i].as_str() {
            "CALLS_RESOLVED" => {
                rel[CALLS as usize].push((s, o));
                rel[RCONF as usize].push((o, s));
                rel[RPOSS as usize].push((o, s));
            }
            "IMPLEMENTS" => {
                rel[IMPL as usize].push((s, o));
                rel[RPOSS as usize].push((s, o));
            }
            "MAY_DISPATCH" => {
                rel[MAY as usize].push((s, o));
                rel[RPOSS as usize].push((o, s));
            }
            "UNRESOLVED_CALL" => {
                if let Some(d) = r.dyn_id {
                    rel[RPOSS as usize].push((d, s));
                }
            }
            _ => {}
        }
    }
    rel
}

/// Position masks of a lane set over `names`.
fn masks_of(
    lanes: &Lanes,
    r: &Resolved,
    names: &[String],
    crates: &HashMap<String, RunStatus>,
) -> Masks {
    let words = names.len().div_ceil(64);
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
    for i in 0..lanes.predicate.len() {
        let (s, o) = (r.sid[i], r.oid[i]);
        match lanes.predicate[i].as_str() {
            "DEFINES" => set(&mut m.fns, s),
            "PUBLIC" => set(&mut m.public, s),
            "SYNTHETIC" => set(&mut m.synthetic, s),
            "REFERENCES" => set(&mut m.referenced, o),
            "CALLS_RESOLVED" => set(&mut m.called, o),
            "IMPLEMENTS" => set(&mut m.impl_subjects, s),
            "MAY_DISPATCH" => set(&mut m.may_targets, o),
            "UNRESOLVED_CALL" => set(&mut m.dynamic_callers, s),
            _ => {}
        }
    }
    for (p, name) in names.iter().enumerate() {
        let owner = name.split("::").next().unwrap_or(name);
        if crates.contains_key(owner) {
            set(&mut m.stored_owner, p as Pos);
        }
    }
    m
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
        let lanes = artifact::read_dir(dir)?;
        let mut names: Vec<String> = Vec::new();
        let mut idx: HashMap<String, Pos> = HashMap::new();
        let res = resolve(&lanes, &mut names, &mut idx);
        let crates = crate_statuses(&lanes);
        let n = names.len();
        let mut rel = edges_of(&lanes, &res);
        rel[NODE as usize] = (0..n as u32).map(|p| (p, p)).collect();
        let m = masks_of(&lanes, &res, &names, &crates);

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
            names,
            index: idx,
            crates,
            axis,
            table,
            cube,
            masks: m,
            rconf: OnceCell::new(),
            rposs: OnceCell::new(),
        })
    }

    /// Re-read `<krate>.rows.tsv` from `dir` and bring the resident graph up
    /// to date without rebuilding it ("sparse write, merge on read").
    ///
    /// * Rows are attributed to their crate file by the `krate` lane
    ///   ([`Lanes::krate`]); the crate's old rows are dropped from the lanes
    ///   and the new file's rows appended.
    /// * Positions never renumber. A vanished symbol keeps its position (and
    ///   its `NODE` edge) with no rows; new symbols are appended. Growth must
    ///   fit the positions reserved at open (`Axis::for_survivors(n).reserved()`):
    ///   a new tile, or a W64 tile that would become W256, is
    ///   [`RefreshRefusal::NeedsReopen`] and changes nothing. Growth inside
    ///   the reservation extends the last tile's map under a new frame
    ///   (`MapTable::advance` + `seal`) and `Cube::rebase`s onto it.
    /// * Edges: the whole graph's edges before and after are compared per
    ///   relation as multisets (a call graph is a bag: two call sites of one
    ///   pair are two edges), so an edge a *different* crate still produces
    ///   is never retired and a second call site is never lost. The difference is
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
    /// Refusals before the first cube mutation leave the graph unchanged; a
    /// `Cube` refusal afterwards (not expected) may leave it partly updated.
    pub fn refresh_crate(
        &mut self,
        dir: &Path,
        krate: &str,
    ) -> Result<RefreshStats, RefreshRefusal> {
        let fresh = artifact::read_crate(dir, krate).map_err(RefreshRefusal::Intake)?;
        let mut new_lanes = self.lanes.without_crate(krate);
        new_lanes.append(fresh);

        let mut names = self.names.clone();
        let mut idx = self.index.clone();
        let old_res = resolve(&self.lanes, &mut names, &mut idx);
        let new_res = resolve(&new_lanes, &mut names, &mut idx);
        let (n_old, n_new) = (self.names.len(), names.len());
        if n_new > self.axis.reserved() {
            return Err(RefreshRefusal::NeedsReopen {
                reason: "new symbols exceed the positions reserved at open (a new tile, or a W64 tile becoming W256)",
            });
        }

        let old_e = edges_of(&self.lanes, &old_res);
        let new_e = edges_of(&new_lanes, &new_res);
        let mut retires: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut adds: Vec<Vec<(u32, u32)>> = Vec::new();
        for r in 0..6 {
            // Multiset diff: a call graph is a bag, two rows for one
            // (src, dst) are two call sites. BTreeMap iterates ascending.
            let mut o: std::collections::BTreeMap<(u32, u32), usize> =
                std::collections::BTreeMap::new();
            let mut n: std::collections::BTreeMap<(u32, u32), usize> =
                std::collections::BTreeMap::new();
            for &e in &old_e[r] {
                *o.entry(e).or_insert(0) += 1;
            }
            for &e in &new_e[r] {
                *n.entry(e).or_insert(0) += 1;
            }
            let mut gone: Vec<(u32, u32)> = Vec::new();
            let mut came: Vec<(u32, u32)> = Vec::new();
            for (&e, &oc) in &o {
                let nc = n.get(&e).copied().unwrap_or(0);
                for _ in nc..oc {
                    gone.push(e);
                }
            }
            for (&e, &nc) in &n {
                let oc = o.get(&e).copied().unwrap_or(0);
                for _ in oc..nc {
                    came.push(e);
                }
            }
            retires.push(gone);
            adds.push(came);
        }
        let crates = crate_statuses(&new_lanes);
        let masks = masks_of(&new_lanes, &new_res, &names, &crates);

        let pending = [
            self.cube.delta_len(RCONF) > 0,
            self.cube.delta_len(RPOSS) > 0,
        ];
        let mut stats = RefreshStats {
            symbols_appended: (n_new - n_old) as u64,
            ..RefreshStats::default()
        };

        // Extend the last tile's map (ids ascending) and adopt the new frame.
        if n_new > n_old {
            let last = self.axis.tiles.saturating_sub(1);
            let lo = last as u64 * TILE_POSITIONS as u64;
            let ids: Vec<u64> = (lo..n_new as u64).collect();
            let map = LocalMap::prepare(self.axis.tile, &ids)
                .map_err(|e| RefreshRefusal::Intake(IntakeRefusal::Map(e)))?;
            let rev = self.table.revisions();
            self.table
                .advance(rev.source + 1, rev.relation_map, vec![(last, Some(map))]);
            let frame = self.table.seal();
            self.cube
                .rebase(&self.table, &frame)
                .map_err(RefreshRefusal::Cube)?;
        }

        for r in 0..6usize {
            let rel = r as RelId;
            if rel == NODE {
                for p in n_old as u32..n_new as u32 {
                    self.cube
                        .add_edge(&self.table, NODE, p, p)
                        .map_err(RefreshRefusal::Cube)?;
                }
                continue;
            }
            for &(s, d) in &retires[r] {
                self.cube
                    .retire_edge(&self.table, rel, s, d)
                    .map_err(RefreshRefusal::Cube)?;
                stats.edges_retired += 1;
            }
            for &(s, d) in &adds[r] {
                self.cube
                    .add_edge(&self.table, rel, s, d)
                    .map_err(RefreshRefusal::Cube)?;
                stats.edges_added += 1;
            }
        }

        let changed = stats.edges_added + stats.edges_retired > 0 || n_new > n_old;
        for (rel, pend) in [(RCONF, pending[0]), (RPOSS, pending[1])] {
            let cell = if rel == RCONF {
                &mut self.rconf
            } else {
                &mut self.rposs
            };
            if !changed {
                continue;
            }
            if let Some(old) = cell.take() {
                if !pend {
                    let (rows, st) = self
                        .cube
                        .refresh_closure(&self.table, rel, &old.0)
                        .map_err(RefreshRefusal::Cube)?;
                    stats.rows_recomputed += st.rows_recomputed;
                    let _ = cell.set((rows, st));
                }
            }
        }

        self.lanes = new_lanes;
        self.names = names;
        self.index = idx;
        self.crates = crates;
        self.masks = masks;
        Ok(stats)
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

    /// The position of `symbol`, if interned.
    pub fn pos(&self, symbol: &str) -> Option<Pos> {
        self.index.get(symbol).copied()
    }

    /// The symbol at position `p`.
    pub fn symbol(&self, p: Pos) -> Option<&str> {
        self.names.get(p as usize).map(String::as_str)
    }

    /// Mask words per position mask: `n.div_ceil(64)`.
    pub fn words(&self) -> usize {
        self.names.len().div_ceil(64)
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
        self.crates.get(krate).copied()
    }

    /// Data rows read.
    pub fn rows(&self) -> usize {
        self.lanes.predicate.len()
    }

    /// Predicate of `row`; panics only on an out-of-range row, like indexing.
    pub fn row_predicate(&self, row: usize) -> &str {
        &self.lanes.predicate[row]
    }

    /// Subject of `row`.
    pub fn row_subject(&self, row: usize) -> &str {
        &self.lanes.subject[row]
    }

    /// Object of `row`, verbatim from the artifact.
    pub fn row_object(&self, row: usize) -> &str {
        &self.lanes.object[row]
    }

    /// `(file, start, end)` of `row`, `None` when the row has no site (`-`).
    pub fn site(&self, row: usize) -> Option<(&str, u32, u32)> {
        let (f, s, e) = (
            self.lanes.file.get(row)?,
            self.lanes.start[row],
            self.lanes.end[row],
        );
        if f == "-" || s == NO_SPAN || e == NO_SPAN {
            None
        } else {
            Some((f.as_str(), s, e))
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
