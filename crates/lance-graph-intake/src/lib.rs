//! Data as config, class as config: PostgreSQL-shaped rows in, PostgreSQL-shaped
//! answers out, with no lane, tile, class or binding in the caller's hands.
//!
//! A consumer hands in column names and cells as PostgreSQL prints them (text,
//! or NULL). [`Table::from_rows`] then:
//!
//! 1. detects each column's PostgreSQL type, lane encoding and absent spelling
//!    from the cells, and the key column ([`detect`]);
//! 2. lets hand config win over detection, with a diff verdict per column
//!    ([`config`]);
//! 3. builds the class from the merged fields (a `ClassView`), registers it in a
//!    `BindStore` and routes it on a DR position;
//! 4. places the rows ([`Table::append`] on an empty table): one `u32` lane and
//!    one presence plane per column and a live plane, owned by this crate's
//!    source, the key to position map in DR W256 tiles (`MapTable`), and every
//!    text column's strings in a DT `CodeTable`;
//! 5. binds, and answers through `lance_graph_pg_row::Reader`, which reads by
//!    the bound basis and consults no `ClassView`.
//!
//! The position of a row is its row index; a key is resolved to it through
//! `MapTable::locate`, never through a hash map. A `Reader` borrows the source
//! and the column specs, so it is built inside each method and never stored.
//!
//! A key is the column's integer value as `u64`: an unsigned column as is, a
//! signed one sign-extended (`-1i64 as u64`).
//!
//! Ownership: DT owns string to code (one code space per text column), DR owns
//! key to position. This crate holds neither. [`Table::append`] and
//! [`Table::delete_key`] advance the data version, which refuses every
//! [`SelectionRef`] kept before.

#![forbid(unsafe_code)]

#[cfg(all(
    not(debug_assertions),
    any(feature = "disable-intake-signed", feature = "disable-intake-live")
))]
compile_error!("disable-intake-* features are for disable runs in debug builds only");

mod bind;
pub mod config;
pub mod detect;
pub mod pg;
pub use bind::BoundQuery;

use lance_graph_mask_risc::LaneRef;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::pg_row::{
    AbsentAs, CodeText, LaneEncoding, PgColumn, PgType, PgTypeRefusal,
};
use lance_graph_dt64::dr::{
    CodeSpaceId, CodeTable, Frame, FrameStore, KeptId, LocalMap, LocateRefusal, MapTable, Scope,
    TileId, ValueKey, Width,
};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, PresenceSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_pg_row::{read_bag, BagSide, ColumnSpec, LiveSource, PgRowRefusal, PgRows, Reader};

pub use config::{merge, ColumnConfig, ColumnShape, Shape, ShapeConfig, Verdict};
pub use detect::{detect_column, detect_key, parse_epoch_days, parse_epoch_seconds, Detected};

const READER: Scope = Scope { id: 1 };
const ROUTE_POSITION: u16 = 1;
const TILE_WIDTH: usize = 256;
const MAX_COLUMNS: usize = 255;
const RELATION_MAP: u64 = 1;

/// Why intake, a placement or a read refused. Nothing is returned with a
/// refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum IntakeRefusal {
    /// A config entry or the configured key names no column.
    UnknownColumn(String),
    /// A column's merged shape is not admitted.
    Column {
        name: String,
        refusal: PgTypeRefusal,
    },
    /// A cell does not parse under its column's merged shape.
    Cell {
        column: String,
        row: usize,
        text: String,
    },
    /// No key column: none configured or detected, it has a NULL cell, or it
    /// is not an integer.
    NoKey,
    /// Two rows carry this key.
    DuplicateKey(u64),
    /// More columns than a binding's field mask holds.
    Width { columns: usize },
    /// The DT64 check refused (class registration, route, bind or read).
    Lane(LaneRefusal),
    /// The polyfill read refused.
    PgRow(PgRowRefusal),
    /// A column's merged shape differs from the one it had.
    ShapeChanged {
        column: String,
        was: PgColumn,
        now: PgColumn,
    },
    /// A row has another number of cells than there are columns.
    Ragged { row: usize, cells: usize },
    /// No row holds this key.
    UnknownKey(u64),
    /// The key map or the code table could not be built or read.
    Placement(String),
    /// A selection could not be kept or read back.
    Keep(String),
    /// A [`SelectionRef`] was kept under an older data version than the
    /// table's current one.
    Stale,
    /// A query could not be bound through `lance_graph_quack::bind`, or a
    /// bound query was not admitted; bind it again.
    Bind(lance_graph_quack::bind::BindError),
}

impl From<LaneRefusal> for IntakeRefusal {
    fn from(refusal: LaneRefusal) -> Self {
        IntakeRefusal::Lane(refusal)
    }
}

impl From<PgRowRefusal> for IntakeRefusal {
    fn from(refusal: PgRowRefusal) -> Self {
        IntakeRefusal::PgRow(refusal)
    }
}

/// The inverse of `write_cell` for one cell: the lane value, or `None` when
/// the text is not a value of the column. `code_of` admits a text cell and
/// returns its code; it is called for text columns only.
fn lane_value(col: &PgColumn, text: &str, code_of: &mut dyn FnMut(&str) -> u32) -> Option<u32> {
    match col.encoding {
        LaneEncoding::Bool => match text {
            // PostgreSQL's output spelling, and its input spelling of the
            // same values: a declared boolean reads both.
            "t" | "true" => Some(1),
            "f" | "false" => Some(0),
            _ => None,
        },
        LaneEncoding::Signed => {
            let v = detect::parse_int(text)?;
            match col.ty {
                PgType::Int2 => i16::try_from(v).ok().map(|x| i32::from(x) as u32),
                _ => i32::try_from(v).ok().map(|x| x as u32),
            }
        }
        LaneEncoding::Unsigned => u32::try_from(detect::parse_int(text)?).ok(),
        LaneEncoding::EpochSeconds => parse_epoch_seconds(text),
        LaneEncoding::EpochDays => parse_epoch_days(text),
        LaneEncoding::TextCode => {
            // A declared `varchar(n)` holds at most `n` characters, as
            // PostgreSQL counts them; a longer cell does not fit the column.
            if let PgType::Varchar(Some(n)) = col.ty {
                if text.chars().count() > n as usize {
                    return None;
                }
            }
            Some(code_of(text))
        }
    }
}

/// A key from a lane value, under an integer encoding.
fn key_of(encoding: LaneEncoding, value: u32) -> Option<u64> {
    match encoding {
        LaneEncoding::Signed => Some(i64::from(value as i32) as u64),
        LaneEncoding::Unsigned => Some(u64::from(value)),
        _ => None,
    }
}

fn bit(plane: &[u64], at: usize) -> bool {
    plane.get(at / 64).is_some_and(|w| w >> (at % 64) & 1 == 1)
}

/// The ascending positions of the set bits of `mask`.
fn positions_of(mask: &[u64]) -> Vec<u32> {
    let mut positions = Vec::new();
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            positions.push((w * 64) as u32 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }
    positions
}

/// The code of `text` in one space: its code at `frame` when DT holds one,
/// else the next fresh code after `base` (codes start at 1; 0 is absent). A
/// fresh string is recorded in `fresh` in code order. `memo` answers repeats
/// within one batch without asking DT again.
fn admit_text(
    codes: &CodeTable<u32>,
    frame: &Frame,
    space: &CodeSpaceId,
    base: u32,
    memo: &mut BTreeMap<String, u32>,
    fresh: &mut Vec<String>,
    text: &str,
) -> u32 {
    if let Some(&code) = memo.get(text) {
        return code;
    }
    let code = match codes.code(frame, space, &ValueKey(text.to_owned())) {
        Ok(code) => code,
        Err(_) => {
            fresh.push(text.to_owned());
            base + fresh.len() as u32
        }
    };
    memo.insert(text.to_owned(), code);
    code
}

/// The selection number under which the owner lends a kept selection.
const KEPT_SEL: u16 = u16::MAX;

/// The source owner: lanes, presence and live planes, the key map and the
/// selections it issues. It holds the `MapTable`, so its frame store is the
/// one the key map is sealed under.
struct Owner {
    class: u32,
    space: CoordSpace,
    iris: Vec<String>,
    lanes: Vec<OwnedLane>,
    presence: Vec<Vec<u64>>,
    live: Vec<u64>,
    rows: usize,
    table: MapTable,
    frame: Frame,
    tiles: Vec<TileId>,
    /// A kept selection lent as selection [`KEPT_SEL`] while one read is
    /// open: its positions are read from DR's keep log in place, never copied.
    lend: Cell<Option<KeptId>>,
}

/// Clears [`Owner::lend`] when dropped, so a read that fails, or a sink that
/// panics, leaves no kept selection lent.
struct Lend<'a>(&'a Cell<Option<KeptId>>);

impl<'a> Lend<'a> {
    fn new(cell: &'a Cell<Option<KeptId>>, id: KeptId) -> Self {
        cell.set(Some(id));
        Lend(cell)
    }
}

impl Drop for Lend<'_> {
    fn drop(&mut self) {
        self.0.set(None);
    }
}

/// A column's lane as the owner holds it. A signed column is an `i32` lane,
/// so Quack's signed comparisons (`LtI32`, `EqI32`, ...) read it as written;
/// every other encoding is a `u32` lane.
enum OwnedLane {
    U32(Vec<u32>),
    I32(Vec<i32>),
}

impl OwnedLane {
    fn new(encoding: LaneEncoding, raw: Vec<u32>) -> OwnedLane {
        match encoding {
            LaneEncoding::Signed if !cfg!(feature = "disable-intake-signed") => {
                OwnedLane::I32(raw.into_iter().map(|v| v as i32).collect())
            }
            _ => OwnedLane::U32(raw),
        }
    }

    fn push(&mut self, value: u32) {
        match self {
            OwnedLane::U32(v) => v.push(value),
            OwnedLane::I32(v) => v.push(value as i32),
        }
    }

    /// The lane value at `at`, as the `u32` bits `write_cell` reads.
    fn value(&self, at: usize) -> u32 {
        match self {
            OwnedLane::U32(v) => v[at],
            OwnedLane::I32(v) => v[at] as u32,
        }
    }

    fn lane_ref(&self) -> LaneRef<'_> {
        match self {
            OwnedLane::U32(v) => LaneRef::U32(v),
            OwnedLane::I32(v) => LaneRef::I32(v),
        }
    }
}

impl LaneSource for Owner {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.class || rev != SchemaRev(1) {
            return None;
        }
        let lane = self
            .iris
            .iter()
            .position(|iri| *iri == basis.predicate_iri)?;
        Some(LaneAddr {
            space: self.space,
            lane: u16::try_from(lane).ok()?,
            scope: READER,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        self.lanes
            .get(usize::from(addr.lane))
            .map(OwnedLane::lane_ref)
    }

    fn frames(&self) -> &FrameStore {
        self.table.frames()
    }
}

impl PresenceSource for Owner {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        self.presence.get(usize::from(addr.lane)).map(Vec::as_slice)
    }
}

impl SelectionSource for Owner {
    fn selection_of(&self, sel: u16, _frame: &Frame) -> Option<SelAddr> {
        (sel == KEPT_SEL && self.lend.get().is_some()).then_some(SelAddr {
            space: self.space,
            sel,
            scope: READER,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        if addr.sel != KEPT_SEL {
            return None;
        }
        let id = self.lend.get()?;
        self.table
            .kept::<Vec<u32>>(&self.frame, id)
            .ok()
            .map(Vec::as_slice)
    }
}

impl LiveSource for Owner {
    fn borrow_live(&self, space: CoordSpace) -> Option<&[u64]> {
        (space == self.space).then_some(self.live.as_slice())
    }
}

/// The class built from the detected fields. Used at registration and bind
/// only; reads go by the bound reference.
struct IntakeView {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl ClassView for IntakeView {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }

    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }

    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// One text column's code space, read at one frame. Built per read, beside the
/// column specs that point at it.
struct SpaceText<'a> {
    codes: &'a CodeTable<u32>,
    frame: &'a Frame,
    space: &'a CodeSpaceId,
}

impl CodeText for SpaceText<'_> {
    fn text(&self, code: u32) -> Option<&str> {
        self.codes
            .key(self.frame, self.space, code)
            .ok()
            .map(|key| key.0.as_str())
    }
}

/// New rows as lane values, and the strings DT does not hold yet.
struct Placed {
    /// `[column][row]`: the lane value, `None` for an absent cell.
    cells: Vec<Vec<Option<u32>>>,
    /// `[column]`: the fresh strings, in code order after the column's
    /// existing codes.
    fresh: Vec<Vec<String>>,
}

/// What one [`Table::append`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendReport {
    /// Rows appended.
    pub rows: usize,
    /// Tiles extended or added.
    pub tiles_touched: usize,
    /// The data version after the append.
    pub version: u64,
}

/// A selection kept under the data version it was made at.
///
/// It reads back only while that version is the table's current one: an
/// append or a delete refuses it with [`IntakeRefusal::Stale`]. A read of a
/// stale reference under its own snapshot is not offered: lanes are
/// append-only, but the live plane after a delete is not versioned per row,
/// so an old selection would not replay what it kept.
#[derive(Clone, Debug)]
pub struct SelectionRef {
    id: KeptId,
    frame: Frame,
    len: usize,
}

impl SelectionRef {
    /// Positions kept.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A table of rows: detected shape, registered class, placed rows and one
/// binding.
pub struct Table {
    name: String,
    shape: Vec<ColumnShape>,
    key: usize,
    /// Every text column's strings, one code space per column.
    codes: CodeTable<u32>,
    /// `Some` for a text column.
    spaces: Vec<Option<CodeSpaceId>>,
    /// Codes assigned per column; they are `1..=code_counts[k]`.
    code_counts: Vec<u32>,
    /// The data version: it moves with every append and every delete.
    source: u64,
    rows_written: Cell<u64>,
    predicate_runs: Cell<u64>,
    /// The validity planes of every column, with the frame they were folded
    /// under. They depend only on the lanes and their presence, which change
    /// only with the data version, so they are folded once per version.
    validity: RefCell<Option<(Frame, Vec<Vec<u64>>)>>,
    validity_folds: Cell<u64>,
    owner: Owner,
    view: IntakeView,
    store: BindStore,
    binding: LaneBinding,
}

impl Table {
    /// Detect, merge config, build the class, register, bind an empty table,
    /// then [`Table::append`] the rows.
    pub fn from_rows(
        name: &str,
        class: u32,
        columns: &[&str],
        rows: &[Vec<Option<&str>>],
        config: &ShapeConfig,
    ) -> Result<Table, IntakeRefusal> {
        if columns.len() > MAX_COLUMNS {
            return Err(IntakeRefusal::Width {
                columns: columns.len(),
            });
        }
        let shaped = Shape::build(columns, rows, config)?;
        let key = shaped.key_column(columns, rows, config)?;
        let shape = shaped.columns;

        let map_table = MapTable::new(1, RELATION_MAP);
        let frame = map_table.seal();
        let codes = CodeTable::new(map_table.frames());

        let space = CoordSpace {
            population: u64::from(class),
            coordinates: 1,
        };
        let iris: Vec<String> = shape
            .iter()
            .map(|c| format!("intake:{name}/{}", c.name))
            .collect();
        let spaces: Vec<Option<CodeSpaceId>> = iris
            .iter()
            .zip(&shape)
            .map(|(iri, c)| {
                (c.col.encoding == LaneEncoding::TextCode).then(|| CodeSpaceId {
                    class,
                    property: iri.clone(),
                })
            })
            .collect();
        let owner = Owner {
            class,
            space,
            iris: iris.clone(),
            lanes: shape
                .iter()
                .map(|column| OwnedLane::new(column.col.encoding, Vec::new()))
                .collect(),
            presence: vec![Vec::new(); shape.len()],
            live: Vec::new(),
            rows: 0,
            table: map_table,
            frame,
            tiles: Vec::new(),
            lend: Cell::new(None),
        };

        let view = IntakeView {
            class: (class >> 16) as ClassId,
            fields: iris
                .iter()
                .zip(&shape)
                .map(|(iri, c)| FieldRef::new(iri.clone(), c.name.clone()))
                .collect(),
        };
        let mut store = BindStore::new();
        store
            .register(
                &view,
                class,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: shape.len(),
                    extent: Extent::DR64_CUBE,
                },
                Vec::new(),
            )
            .map_err(|refusal| IntakeRefusal::Lane(LaneRefusal::Bind(refusal)))?;
        let version = store
            .route(ROUTE_POSITION, class)
            .map_err(|refusal| IntakeRefusal::Lane(LaneRefusal::Bind(refusal)))?;
        let fields: Vec<u8> = (0..shape.len()).map(|k| k as u8).collect();
        let binding = store.bind_lanes(
            &view,
            owner.table.frames(),
            &owner,
            LaneRequest {
                route: RouteAt {
                    position: ROUTE_POSITION,
                    version,
                },
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&fields),
                frame: &owner.frame,
                scope: READER,
                selection: owner.space,
            },
        )?;

        let mut table = Table {
            name: name.to_owned(),
            code_counts: vec![0; shape.len()],
            shape,
            key,
            codes,
            spaces,
            source: 1,
            rows_written: Cell::new(0),
            predicate_runs: Cell::new(0),
            validity: RefCell::new(None),
            validity_folds: Cell::new(0),
            owner,
            view,
            store,
            binding,
        };
        table.append_rows(rows)?;
        Ok(table)
    }

    /// Every column's merged shape with its verdict against the detection.
    pub fn shape(&self) -> &[ColumnShape] {
        &self.shape
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Live rows.
    pub fn len(&self) -> usize {
        self.owner
            .live
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rows placed by `from_rows` and `append` so far: one per row, never per
    /// row already in the table.
    #[doc(hidden)]
    pub fn rows_written(&self) -> u64 {
        self.rows_written.get()
    }

    /// Predicate executions (`keep_where` or `count_where`) the facade has
    /// run. Reading a kept selection runs none.
    #[doc(hidden)]
    pub fn predicate_runs(&self) -> u64 {
        self.predicate_runs.get()
    }

    /// Validity planes folded so far: one per column each time the data
    /// version moved and a WHERE ran. A WHERE under an unchanged version adds
    /// none.
    #[doc(hidden)]
    pub fn validity_folds(&self) -> u64 {
        self.validity_folds.get()
    }

    /// The frame the table's lanes and codes are currently sealed under.
    #[doc(hidden)]
    pub fn frame(&self) -> Frame {
        self.owner.frame.clone()
    }

    /// The string `code` meant in `column` as of `frame`, read from DT's
    /// `CodeTable`. `None` for code 0, an unknown code or a non-text column.
    #[doc(hidden)]
    pub fn code_key_at(&self, frame: &Frame, column: &str, code: u32) -> Option<String> {
        let k = self.shape.iter().position(|c| c.name == column)?;
        let space = self.spaces.get(k)?.as_ref()?;
        self.codes
            .key(frame, space, code)
            .ok()
            .map(|key| key.0.clone())
    }

    /// [`Table::code_key_at`] under the current frame.
    #[doc(hidden)]
    pub fn code_key(&self, column: &str, code: u32) -> Option<String> {
        self.code_key_at(&self.owner.frame, column, code)
    }

    fn texts(&self) -> Vec<Option<SpaceText<'_>>> {
        self.spaces
            .iter()
            .map(|space| {
                space.as_ref().map(|space| SpaceText {
                    codes: &self.codes,
                    frame: &self.owner.frame,
                    space,
                })
            })
            .collect()
    }

    fn specs<'a>(&'a self, texts: &'a [Option<SpaceText<'a>>]) -> Vec<ColumnSpec<'a>> {
        self.shape
            .iter()
            .zip(texts)
            .map(|(column, text)| ColumnSpec {
                name: &column.name,
                col: column.col,
                codes: text.as_ref().map(|t| t as &dyn CodeText),
            })
            .collect()
    }

    fn reader<'s, 'c>(
        &'s self,
        specs: &'c [ColumnSpec<'c>],
    ) -> Result<Reader<'s, 'c>, IntakeRefusal> {
        Ok(Reader::open(
            &self.binding,
            &self.store,
            self.owner.table.frames(),
            &self.owner,
            specs,
        )?)
    }

    /// Run `f` with the validity plane of every column for the current data
    /// version. The planes are folded on the first call under a frame and
    /// reused until an append or a delete moves it.
    fn with_validity<R>(
        &self,
        reader: &Reader<'_, '_>,
        f: impl FnOnce(&[&[u64]]) -> Result<R, PgRowRefusal>,
    ) -> Result<R, IntakeRefusal> {
        let current = self
            .validity
            .borrow()
            .as_ref()
            .is_some_and(|(frame, _)| self.owner.table.frames().check(frame).is_ok());
        if !current {
            let planes = reader.validity_planes()?;
            self.validity_folds
                .set(self.validity_folds.get() + planes.len() as u64);
            *self.validity.borrow_mut() = Some((self.owner.frame.clone(), planes));
        }
        let kept = self.validity.borrow();
        let planes = kept
            .as_ref()
            .map(|(_, planes)| planes.as_slice())
            .ok_or_else(|| IntakeRefusal::Keep("validity planes missing".to_owned()))?;
        let views: Vec<&[u64]> = planes.iter().map(Vec::as_slice).collect();
        Ok(f(&views)?)
    }

    /// SELECT * : every live row, ascending position.
    pub fn select_all(&self) -> Result<PgRows, IntakeRefusal> {
        let texts = self.texts();
        let specs = self.specs(&texts);
        let reader = self.reader(&specs)?;
        Ok(reader.rows_in(&self.owner.live)?)
    }

    /// SELECT * WHERE filter (Quack Filter over `Col(k)` = column `k`),
    /// three-valued.
    pub fn select_where(
        &self,
        filter: &lance_graph_quack::Filter,
    ) -> Result<PgRows, IntakeRefusal> {
        let texts = self.texts();
        let specs = self.specs(&texts);
        let reader = self.reader(&specs)?;
        self.predicate_runs.set(self.predicate_runs.get() + 1);
        let kept = self.with_validity(&reader, |v| reader.keep_where_valid(filter, v))?;
        Ok(reader.rows_in(&kept)?)
    }

    /// `COUNT(*)` of the live rows `filter` holds for.
    pub fn count_where(&self, filter: &lance_graph_quack::Filter) -> Result<usize, IntakeRefusal> {
        let texts = self.texts();
        let specs = self.specs(&texts);
        let reader = self.reader(&specs)?;
        self.predicate_runs.set(self.predicate_runs.get() + 1);
        self.with_validity(&reader, |v| reader.count_where_valid(filter, v))
    }

    /// The position of the row holding `key`, or `None` when no row does.
    fn locate(&self, key: u64) -> Result<Option<u32>, IntakeRefusal> {
        match self
            .owner
            .table
            .locate(&self.owner.frame, &self.owner.tiles, key)
        {
            Ok((ordinal, local)) => ordinal
                .checked_mul(TILE_WIDTH as u32)
                .and_then(|base| base.checked_add(u32::from(local)))
                .map(Some)
                .ok_or_else(|| IntakeRefusal::Placement("position overflow".to_owned())),
            Err(LocateRefusal::Unmapped) => Ok(None),
            Err(other) => Err(IntakeRefusal::Placement(format!("{other:?}"))),
        }
    }

    fn position_of(&self, key: u64) -> Result<u32, IntakeRefusal> {
        self.locate(key)?.ok_or(IntakeRefusal::UnknownKey(key))
    }

    fn is_live(&self, position: u32) -> bool {
        bit(&self.owner.live, position as usize)
    }

    /// The positions of these keys, in list order, duplicates kept; an
    /// unknown key is refused.
    fn positions_for_keys(&self, keys: &[u64]) -> Result<Vec<u32>, IntakeRefusal> {
        // Each distinct key is located once per call and reused for its
        // repeats. A sorted Vec of this call's keys is local to the call, so it
        // is not a second key registry (D-REGC-7): the key map stays the DR tiles.
        let mut distinct = keys.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut resolved: Vec<(u64, u32)> = Vec::with_capacity(distinct.len());
        for key in distinct {
            resolved.push((key, self.position_of(key)?));
        }
        let mut positions = Vec::with_capacity(keys.len());
        for &key in keys {
            let at = resolved
                .binary_search_by_key(&key, |&(k, _)| k)
                .map_err(|_| IntakeRefusal::UnknownKey(key))?;
            positions.push(resolved[at].1);
        }
        Ok(positions)
    }

    /// The rows of these keys, in list order, duplicates kept (a bag); an
    /// unknown or deleted key is refused.
    ///
    /// A read: the positions live on this call and are read in place
    /// (`identity[index]`); nothing is written to the table.
    pub fn rows_for_keys(&self, keys: &[u64]) -> Result<PgRows, IntakeRefusal> {
        let positions = self.positions_for_keys(keys)?;
        let texts = self.texts();
        let specs = self.specs(&texts);
        Ok(self.reader(&specs)?.rows_at(&positions)?)
    }

    fn read_selection(&self, sel: u16) -> Result<PgRows, IntakeRefusal> {
        let texts = self.texts();
        let specs = self.specs(&texts);
        let (reader, positions) = Reader::open_selected(
            &self.binding,
            &self.store,
            self.owner.table.frames(),
            &self.owner,
            sel,
            &specs,
        )?;
        Ok(reader.rows_at(positions)?)
    }

    /// Keep `positions` under the current frame.
    fn keep_positions(&mut self, positions: Vec<u32>) -> Result<SelectionRef, IntakeRefusal> {
        let len = positions.len();
        let frame = self.owner.frame.clone();
        let id = self
            .owner
            .table
            .keep(&frame, positions)
            .map_err(|refusal| IntakeRefusal::Keep(format!("{refusal:?}")))?;
        Ok(SelectionRef { id, frame, len })
    }

    /// Run `filter` once, keep the positions it holds for under the current
    /// data version, and return the handle. [`Table::rows_of`] then reads
    /// them back with no predicate work.
    pub fn select_ref(
        &mut self,
        filter: &lance_graph_quack::Filter,
    ) -> Result<SelectionRef, IntakeRefusal> {
        let positions = {
            let texts = self.texts();
            let specs = self.specs(&texts);
            let reader = self.reader(&specs)?;
            self.predicate_runs.set(self.predicate_runs.get() + 1);
            let kept = self.with_validity(&reader, |v| reader.keep_where_valid(filter, v))?;
            positions_of(&kept)
        };
        self.keep_positions(positions)
    }

    /// Keep the positions of these keys, in list order, repeats kept, as
    /// [`Table::rows_for_keys`] resolves them. An unknown key is refused.
    pub fn select_keys_ref(&mut self, keys: &[u64]) -> Result<SelectionRef, IntakeRefusal> {
        let positions = self.positions_for_keys(keys)?;
        self.keep_positions(positions)
    }

    /// The kept positions of `r`, when `r` was kept under the table's current
    /// data version.
    fn kept_of(&self, r: &SelectionRef) -> Result<&Vec<u32>, IntakeRefusal> {
        self.owner
            .table
            .frames()
            .check(&r.frame)
            .map_err(|_| IntakeRefusal::Stale)?;
        self.owner
            .table
            .kept::<Vec<u32>>(&r.frame, r.id)
            .map_err(|refusal| IntakeRefusal::Keep(format!("{refusal:?}")))
    }

    /// The rows of a kept selection, in its order. The selection must have
    /// been kept under the current data version; no predicate runs.
    pub fn rows_of(&self, r: &SelectionRef) -> Result<PgRows, IntakeRefusal> {
        self.kept_of(r)?;
        let _lend = Lend::new(&self.owner.lend, r.id);
        self.read_selection(KEPT_SEL)
    }

    /// How many positions a kept selection holds, under the same currency
    /// check as [`Table::rows_of`].
    pub fn count_of(&self, r: &SelectionRef) -> Result<usize, IntakeRefusal> {
        Ok(self.kept_of(r)?.len())
    }

    /// Re-seal the frame after the table advanced, and bind again under it:
    /// registration and route are unchanged. Cold; once per append or delete.
    fn rebind(&mut self) -> Result<(), IntakeRefusal> {
        self.owner.frame = self.owner.table.seal();
        let fields: Vec<u8> = (0..self.shape.len()).map(|k| k as u8).collect();
        self.binding = self.store.bind_lanes(
            &self.view,
            self.owner.table.frames(),
            &self.owner,
            LaneRequest {
                route: self.binding.route(),
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&fields),
                frame: &self.owner.frame,
                scope: READER,
                selection: self.owner.space,
            },
        )?;
        Ok(())
    }

    /// Advance the data version once, with the given tile changes.
    fn advance_data(&mut self, changes: Vec<(TileId, Option<LocalMap>)>) {
        self.source += 1;
        self.owner.table.advance(self.source, RELATION_MAP, changes);
    }

    /// Mark a row deleted: its live bit is cleared; its values stay, and
    /// reads refuse or skip it. The data version advances, so a kept
    /// selection from before is refused.
    pub fn delete_key(&mut self, key: u64) -> Result<(), IntakeRefusal> {
        let at = self.position_of(key)? as usize;
        if !cfg!(feature = "disable-intake-live") {
            self.owner.live[at / 64] &= !(1u64 << (at % 64));
        }
        self.advance_data(Vec::new());
        self.rebind()
    }

    /// LEFT JOIN `right` ON right.key = self.`fk`: every live row of self,
    /// ascending, with right's columns or -0 (NULL / '') where the fk is NULL
    /// or names no live right row.
    pub fn left_join(&self, fk: &str, right: &Table) -> Result<PgRows, IntakeRefusal> {
        let f = self
            .shape
            .iter()
            .position(|c| c.name == fk)
            .ok_or_else(|| IntakeRefusal::UnknownColumn(fk.to_owned()))?;
        let encoding = self.shape[f].col.encoding;
        if key_of(encoding, 0).is_none() {
            return Err(IntakeRefusal::Column {
                name: fk.to_owned(),
                refusal: PgTypeRefusal::EncodingMismatch {
                    ty: self.shape[f].col.ty,
                    encoding,
                },
            });
        }

        let left: Vec<u32> = (0..self.owner.rows)
            .filter(|&p| bit(&self.owner.live, p))
            .map(|p| p as u32)
            .collect();
        // The fk is read in place at each left position (`identity[index]
        // .value[fk]`); no copy of the column is made.
        let fk_at = |p: u32| -> Option<u64> {
            let at = p as usize;
            bit(&self.owner.presence[f], at)
                .then(|| key_of(encoding, self.owner.lanes[f].value(at)))
                .flatten()
        };

        // Each distinct fk value is located once per call; see `rows_for_keys`.
        let mut distinct: Vec<u64> = left.iter().filter_map(|&p| fk_at(p)).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut resolved: Vec<(u64, Option<u32>)> = Vec::with_capacity(distinct.len());
        for key in distinct {
            let position = right.locate(key)?.filter(|&p| right.is_live(p));
            resolved.push((key, position));
        }

        let mut positions = vec![0u32; left.len()];
        let mut present = vec![0u64; left.len().div_ceil(64)];
        for (i, &p) in left.iter().enumerate() {
            let found = fk_at(p)
                .and_then(|k| resolved.binary_search_by_key(&k, |&(k, _)| k).ok())
                .and_then(|at| resolved[at].1);
            if let Some(position) = found {
                positions[i] = position;
                present[i / 64] |= 1 << (i % 64);
            }
        }

        let left_texts = self.texts();
        let right_texts = right.texts();
        let left_specs = self.specs(&left_texts);
        let right_specs = right.specs(&right_texts);
        let left_reader = self.reader(&left_specs)?;
        let right_reader = right.reader(&right_specs)?;
        Ok(read_bag(&[
            BagSide {
                reader: &left_reader,
                positions: &left,
                present: None,
            },
            BagSide {
                reader: &right_reader,
                positions: &positions,
                present: Some(&present),
            },
        ])?)
    }

    /// Check new rows against the shape this table was built with. `Ok(())`
    /// when every cell of every new row fits its column's merged shape (an
    /// all-NULL batch, or small values in an `int8` column, fit);
    /// `ShapeChanged` for the first column with a cell that does not, with
    /// the shape detected for that column over the new batch. A new shape is
    /// a new class: the caller supplies a new classid through `from_rows`.
    pub fn observe(&self, rows: &[Vec<Option<&str>>]) -> Result<(), IntakeRefusal> {
        if let Some((row, cells)) = rows
            .iter()
            .enumerate()
            .find(|(_, cells)| cells.len() != self.shape.len())
        {
            return Err(IntakeRefusal::Ragged {
                row,
                cells: cells.len(),
            });
        }
        for (k, was) in self.shape.iter().enumerate() {
            let fits = rows.iter().all(|row| match row.get(k).copied().flatten() {
                None => true,
                Some("") if was.col.absent == AbsentAs::EmptyText => true,
                Some(text) => lane_value(&was.col, text, &mut |_: &str| 0u32).is_some(),
            });
            if !fits {
                let cells: Vec<Option<&str>> = rows
                    .iter()
                    .map(|row| row.get(k).copied().flatten())
                    .collect();
                return Err(IntakeRefusal::ShapeChanged {
                    column: was.name.clone(),
                    was: was.col,
                    now: detect::detect_column(&cells).col,
                });
            }
        }
        Ok(())
    }

    /// The key column's name.
    pub fn key_name(&self) -> &str {
        &self.shape[self.key].name
    }

    /// Append rows. Every row must fit the table's merged shape (as
    /// [`Table::observe`] judges it) and carry a key no row holds; otherwise
    /// the call refuses and the table is unchanged. A new shape is a new
    /// class, so a column is never widened in place.
    ///
    /// Positions continue from the rows already placed: the last tile is
    /// extended and further tiles are added. New strings take the next codes
    /// of their column's code space; old codes never renumber. The data
    /// version advances once, so a [`SelectionRef`] kept before is refused.
    pub fn append(&mut self, rows: &[Vec<Option<&str>>]) -> Result<AppendReport, IntakeRefusal> {
        self.observe(rows)?;
        self.append_rows(rows)
    }

    /// Place rows that have not been checked against the shape: a cell that
    /// does not fit is a `Cell` refusal. Nothing changes until every check
    /// has passed.
    fn append_rows(&mut self, rows: &[Vec<Option<&str>>]) -> Result<AppendReport, IntakeRefusal> {
        let n = rows.len();
        if n == 0 {
            return Ok(AppendReport {
                rows: 0,
                tiles_touched: 0,
                version: self.source,
            });
        }
        let n0 = self.owner.rows;
        let total = n0
            .checked_add(n)
            .filter(|&t| u32::try_from(t).is_ok())
            .ok_or_else(|| {
                IntakeRefusal::Placement("more rows than a position holds".to_owned())
            })?;

        let placed = self.place(rows)?;
        let keys = self.new_keys(&placed)?;
        let (changes, new_tiles) = self.tile_changes(&keys)?;
        let tiles_touched = changes.len();

        // Past the last refusal. DT first: its advance is the only step that
        // can still refuse, and it refuses without changing anything.
        let mut code_changes = Vec::new();
        for (k, fresh) in placed.fresh.iter().enumerate() {
            let Some(space) = self.spaces[k].as_ref().filter(|_| !fresh.is_empty()) else {
                continue;
            };
            let base = self.code_counts[k];
            let mut entries = Vec::with_capacity(base as usize + fresh.len());
            for code in 1..=base {
                let key = self
                    .codes
                    .key(&self.owner.frame, space, code)
                    .map_err(|refusal| IntakeRefusal::Placement(format!("{refusal:?}")))?;
                entries.push((key.clone(), code));
            }
            for (i, text) in fresh.iter().enumerate() {
                entries.push((ValueKey(text.clone()), base + 1 + i as u32));
            }
            code_changes.push((space.clone(), entries));
        }
        if !code_changes.is_empty() {
            self.owner
                .table
                .advance_codes(&mut self.codes, code_changes)
                .map_err(|refusal| IntakeRefusal::Placement(format!("{refusal:?}")))?;
            for (count, fresh) in self.code_counts.iter_mut().zip(&placed.fresh) {
                *count += fresh.len() as u32;
            }
        }

        let words = total.div_ceil(64);
        for (k, lane) in self.owner.lanes.iter_mut().enumerate() {
            for cell in &placed.cells[k] {
                lane.push(cell.unwrap_or(0));
            }
        }
        for (k, plane) in self.owner.presence.iter_mut().enumerate() {
            plane.resize(words, 0);
            for (r, cell) in placed.cells[k].iter().enumerate() {
                if cell.is_some() {
                    let at = n0 + r;
                    plane[at / 64] |= 1 << (at % 64);
                }
            }
        }
        self.owner.live.resize(words, 0);
        for at in n0..total {
            self.owner.live[at / 64] |= 1 << (at % 64);
        }
        self.owner.rows = total;
        self.owner.tiles.extend(new_tiles);

        self.advance_data(changes);
        self.rebind()?;
        self.rows_written.set(self.rows_written.get() + n as u64);
        Ok(AppendReport {
            rows: n,
            tiles_touched,
            version: self.source,
        })
    }

    /// New rows to lane values. NULL is absent (presence 0, value 0), and so
    /// is `''` under `AbsentAs::EmptyText`. A text cell is admitted through
    /// DT: its code at the current frame, else the next fresh one.
    fn place(&self, rows: &[Vec<Option<&str>>]) -> Result<Placed, IntakeRefusal> {
        let columns = self.shape.len();
        let mut cells = vec![vec![None; rows.len()]; columns];
        let mut memos: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); columns];
        let mut fresh: Vec<Vec<String>> = vec![Vec::new(); columns];
        for (r, row) in rows.iter().enumerate() {
            for (k, (column, space)) in self.shape.iter().zip(&self.spaces).enumerate() {
                let Some(text) = row.get(k).copied().flatten() else {
                    continue;
                };
                if text.is_empty() && column.col.absent == AbsentAs::EmptyText {
                    continue;
                }
                let base = self.code_counts[k];
                let (memo, new) = (&mut memos[k], &mut fresh[k]);
                let mut code_of = |t: &str| match space {
                    Some(space) => {
                        admit_text(&self.codes, &self.owner.frame, space, base, memo, new, t)
                    }
                    None => 0,
                };
                let value = lane_value(&column.col, text, &mut code_of).ok_or_else(|| {
                    IntakeRefusal::Cell {
                        column: column.name.clone(),
                        row: r,
                        text: text.to_owned(),
                    }
                })?;
                cells[k][r] = Some(value);
            }
        }
        Ok(Placed { cells, fresh })
    }

    /// The key of every new row, from the key column. A NULL key is refused,
    /// and so is a key repeated in the batch or held by an existing row.
    fn new_keys(&self, placed: &Placed) -> Result<Vec<u64>, IntakeRefusal> {
        let encoding = self.shape[self.key].col.encoding;
        let keys = placed.cells[self.key]
            .iter()
            .map(|&cell| {
                cell.and_then(|value| key_of(encoding, value))
                    .ok_or(IntakeRefusal::NoKey)
            })
            .collect::<Result<Vec<u64>, IntakeRefusal>>()?;
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(IntakeRefusal::DuplicateKey(pair[0]));
        }
        for &key in &keys {
            if self.locate(key)?.is_some() {
                return Err(IntakeRefusal::DuplicateKey(key));
            }
        }
        Ok(keys)
    }

    /// The tile changes that place `keys` after the rows already held: the
    /// last tile extended up to its width, then new tiles. Also the ids of
    /// the tiles added.
    #[allow(clippy::type_complexity)]
    fn tile_changes(
        &self,
        keys: &[u64],
    ) -> Result<(Vec<(TileId, Option<LocalMap>)>, Vec<TileId>), IntakeRefusal> {
        let placement = |detail: String| IntakeRefusal::Placement(detail);
        let mut changes = Vec::new();
        let mut added = Vec::new();
        let mut rest = keys;
        if let Some(&last) = self.owner.tiles.last() {
            let old = self
                .owner
                .table
                .map(&self.owner.frame, last)
                .map_err(|refusal| placement(format!("{refusal:?}")))?;
            let take = TILE_WIDTH.saturating_sub(old.len()).min(rest.len());
            if take > 0 {
                let mut globals = old.globals().to_vec();
                globals.extend_from_slice(&rest[..take]);
                let map = LocalMap::prepare(Width::W256, &globals)
                    .map_err(|refusal| placement(format!("{refusal:?}")))?;
                if !map.view().extends(old) {
                    return Err(placement(
                        "a tile extension renumbered a position".to_owned(),
                    ));
                }
                changes.push((last, Some(map)));
                rest = &rest[take..];
            }
        }
        for chunk in rest.chunks(TILE_WIDTH) {
            let tile = TileId::try_from(self.owner.tiles.len() + added.len() + 1)
                .map_err(|_| placement("more tiles than a tile id holds".to_owned()))?;
            let map = LocalMap::prepare(Width::W256, chunk)
                .map_err(|refusal| placement(format!("{refusal:?}")))?;
            added.push(tile);
            changes.push((tile, Some(map)));
        }
        Ok((changes, added))
    }
}
