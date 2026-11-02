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

impl Codegraph {
    /// Read every `*.rows.tsv` in `dir` and build the graph.
    pub fn open(dir: &Path) -> Result<Self, IntakeRefusal> {
        let lanes = artifact::read_dir(dir)?;
        let rows = lanes.predicate.len();

        // Pass 1: intern symbols in first-seen row order; remember each row's
        // subject and object ids in lanes (NONE when the row has no symbol).
        let mut names: Vec<String> = Vec::new();
        let mut idx: HashMap<String, Pos> = HashMap::new();
        let mut sid = vec![NONE; rows];
        let mut oid = vec![NONE; rows];
        let mut any_dynamic = false;
        let mut crates = HashMap::new();
        for i in 0..rows {
            let (s, o) = (lanes.subject[i].as_str(), lanes.object[i].as_str());
            match lanes.predicate[i].as_str() {
                "DEFINES" | "CALLS_RESOLVED" | "PUBLIC" | "IMPLEMENTS" | "SYNTHETIC"
                | "REFERENCES" => {
                    sid[i] = intern(&mut names, &mut idx, s);
                    oid[i] = intern(&mut names, &mut idx, o);
                }
                "MAY_DISPATCH" => {
                    sid[i] = intern(&mut names, &mut idx, s);
                    oid[i] = intern(&mut names, &mut idx, dispatch_target(o));
                }
                // The caller is a symbol; the object ("dynamic") is not.
                "UNRESOLVED_CALL" => {
                    sid[i] = intern(&mut names, &mut idx, s);
                    any_dynamic = true;
                }
                "CRATE" => {
                    let st = if o == "COMPLETE" {
                        RunStatus::Complete
                    } else {
                        RunStatus::Partial
                    };
                    crates.insert(s.to_string(), st);
                }
                _ => {}
            }
        }
        let dyn_id = any_dynamic.then(|| intern(&mut names, &mut idx, DYNAMIC));
        let n = names.len();
        let words = n.div_ceil(64);

        // Pass 2: relations and masks.
        let mut rel: [Vec<(u32, u32)>; 6] = Default::default();
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
        rel[NODE as usize] = (0..n as u32).map(|p| (p, p)).collect();
        for i in 0..rows {
            let (s, o) = (sid[i], oid[i]);
            match lanes.predicate[i].as_str() {
                "DEFINES" => set(&mut m.fns, s),
                "PUBLIC" => set(&mut m.public, s),
                "SYNTHETIC" => set(&mut m.synthetic, s),
                "REFERENCES" => set(&mut m.referenced, o),
                "CALLS_RESOLVED" => {
                    set(&mut m.called, o);
                    rel[CALLS as usize].push((s, o));
                    rel[RCONF as usize].push((o, s));
                    rel[RPOSS as usize].push((o, s));
                }
                "IMPLEMENTS" => {
                    set(&mut m.impl_subjects, s);
                    rel[IMPL as usize].push((s, o));
                    rel[RPOSS as usize].push((s, o));
                }
                "MAY_DISPATCH" => {
                    set(&mut m.may_targets, o);
                    rel[MAY as usize].push((s, o));
                    rel[RPOSS as usize].push((o, s));
                }
                "UNRESOLVED_CALL" => {
                    set(&mut m.dynamic_callers, s);
                    if let Some(d) = dyn_id {
                        rel[RPOSS as usize].push((d, s));
                    }
                }
                _ => {}
            }
        }
        for (p, name) in names.iter().enumerate() {
            let owner = name.split("::").next().unwrap_or(name);
            if crates.contains_key(owner) {
                set(&mut m.stored_owner, p as Pos);
            }
        }

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

        let index = idx.into_iter().map(|(k, v)| (k, v)).collect();
        Ok(Self {
            lanes,
            names,
            index,
            crates,
            table,
            cube,
            masks: m,
            rconf: OnceCell::new(),
            rposs: OnceCell::new(),
        })
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
