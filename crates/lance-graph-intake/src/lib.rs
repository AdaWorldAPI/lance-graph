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
//! 4. places the rows: one `u32` lane and one presence plane per column and a
//!    live plane, owned by this crate's source, and the key to position map in
//!    DR W256 tiles (`MapTable`);
//! 5. binds once, and answers through `lance_graph_pg_row::Reader`, which reads
//!    by the bound basis and consults no `ClassView`.
//!
//! The position of a row is its row index; a key is resolved to it through
//! `MapTable::locate`, never through a hash map. A `Reader` borrows the source
//! and the column specs, so it is built inside each method and never stored.
//!
//! A key is the column's integer value as `u64`: an unsigned column as is, a
//! signed one sign-extended (`-1i64 as u64`).

#![forbid(unsafe_code)]

#[cfg(all(
    not(debug_assertions),
    any(feature = "disable-intake-signed", feature = "disable-intake-live")
))]
compile_error!("disable-intake-* features are for disable runs in debug builds only");

pub mod config;
pub mod detect;

use lance_graph_mask_risc::LaneRef;
use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::pg_row::{
    AbsentAs, CodeText, LaneEncoding, PgColumn, PgType, PgTypeRefusal,
};
use lance_graph_dt64::dr::{
    Frame, FrameStore, LocalMap, LocateRefusal, MapTable, Scope, TileId, Width,
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
const PLACEHOLDER_CODE: &str = "<code 0>";

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
    /// The key map could not be built or read.
    Placement(String),
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

/// A text column's codes: code 0 is reserved for absent.
struct Codes {
    table: Vec<String>,
    index: BTreeMap<String, u32>,
}

impl Codes {
    fn new() -> Codes {
        Codes {
            table: vec![PLACEHOLDER_CODE.to_owned()],
            index: BTreeMap::new(),
        }
    }

    fn code(&mut self, text: &str) -> u32 {
        if let Some(&code) = self.index.get(text) {
            return code;
        }
        let code = self.table.len() as u32;
        self.table.push(text.to_owned());
        self.index.insert(text.to_owned(), code);
        code
    }
}

/// The inverse of `write_cell` for one cell: the lane value, or `None` when
/// the text is not a value of the column.
fn lane_value(col: &PgColumn, text: &str, codes: &mut Codes) -> Option<u32> {
    match col.encoding {
        LaneEncoding::Bool => match text {
            "t" => Some(1),
            "f" => Some(0),
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
        LaneEncoding::TextCode => Some(codes.code(text)),
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
    selections: Vec<Vec<u32>>,
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
        (usize::from(sel) < self.selections.len()).then_some(SelAddr {
            space: self.space,
            sel,
            scope: READER,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.selections
            .get(usize::from(addr.sel))
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

struct Cells {
    lanes: Vec<Vec<u32>>,
    presence: Vec<Vec<u64>>,
    codes: Vec<Vec<String>>,
}

/// Rows to lanes, presence planes and code tables, by each column's merged
/// shape. NULL is absent (presence 0, value 0), and so is `''` under
/// `AbsentAs::EmptyText`.
fn place_cells(shape: &[ColumnShape], rows: &[Vec<Option<&str>>]) -> Result<Cells, IntakeRefusal> {
    let n = rows.len();
    let words = n.div_ceil(64);
    let mut lanes = vec![vec![0u32; n]; shape.len()];
    let mut presence = vec![vec![0u64; words]; shape.len()];
    let mut codes: Vec<Codes> = shape.iter().map(|_| Codes::new()).collect();
    for (r, row) in rows.iter().enumerate() {
        for (k, column) in shape.iter().enumerate() {
            let Some(text) = row.get(k).copied().flatten() else {
                continue;
            };
            if text.is_empty() && column.col.absent == AbsentAs::EmptyText {
                continue;
            }
            let value = lane_value(&column.col, text, &mut codes[k]).ok_or_else(|| {
                IntakeRefusal::Cell {
                    column: column.name.clone(),
                    row: r,
                    text: text.to_owned(),
                }
            })?;
            lanes[k][r] = value;
            presence[k][r / 64] |= 1 << (r % 64);
        }
    }
    Ok(Cells {
        lanes,
        presence,
        codes: codes.into_iter().map(|c| c.table).collect(),
    })
}

/// The key of every row, from the key column's lane. A NULL key is refused.
fn key_values(
    column: &ColumnShape,
    lane: &[u32],
    presence: &[u64],
) -> Result<Vec<u64>, IntakeRefusal> {
    lane.iter()
        .enumerate()
        .map(|(r, &value)| {
            bit(presence, r)
                .then(|| key_of(column.col.encoding, value))
                .flatten()
                .ok_or(IntakeRefusal::NoKey)
        })
        .collect()
}

/// A table of rows: detected shape, registered class, placed rows and one
/// binding.
pub struct Table {
    name: String,
    config: ShapeConfig,
    shape: Vec<ColumnShape>,
    key: usize,
    codes: Vec<Vec<String>>,
    owner: Owner,
    store: BindStore,
    binding: LaneBinding,
}

impl Table {
    /// Detect, merge config, build the class, place rows, register, bind.
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
        let n = u32::try_from(rows.len())
            .map_err(|_| IntakeRefusal::Placement("more rows than a position holds".to_owned()))?;
        let shaped = Shape::build(columns, rows, config)?;
        let key = shaped.key_column(columns, rows, config)?;
        let shape = shaped.columns;
        let cells = place_cells(&shape, rows)?;
        let keys = key_values(&shape[key], &cells.lanes[key], &cells.presence[key])?;

        let mut sorted = keys.clone();
        sorted.sort_unstable();
        if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(IntakeRefusal::DuplicateKey(pair[0]));
        }

        let mut table = MapTable::new(1, 1);
        let mut tiles = Vec::new();
        let mut changes = Vec::new();
        for (i, chunk) in keys.chunks(TILE_WIDTH).enumerate() {
            let tile = TileId::try_from(i + 1).map_err(|_| {
                IntakeRefusal::Placement("more tiles than a tile id holds".to_owned())
            })?;
            let map = LocalMap::prepare(Width::W256, chunk)
                .map_err(|refusal| IntakeRefusal::Placement(format!("{refusal:?}")))?;
            tiles.push(tile);
            changes.push((tile, Some(map)));
        }
        table.advance(1, 1, changes);
        let frame = table.seal();

        let space = CoordSpace {
            population: u64::from(class),
            coordinates: 1,
        };
        let iris: Vec<String> = shape
            .iter()
            .map(|c| format!("intake:{name}/{}", c.name))
            .collect();
        let mut live = vec![0u64; rows.len().div_ceil(64)];
        for r in 0..rows.len() {
            live[r / 64] |= 1 << (r % 64);
        }
        let owner = Owner {
            class,
            space,
            iris: iris.clone(),
            lanes: cells
                .lanes
                .into_iter()
                .zip(&shape)
                .map(|(raw, column)| OwnedLane::new(column.col.encoding, raw))
                .collect(),
            presence: cells.presence,
            live,
            rows: n as usize,
            table,
            frame,
            tiles,
            selections: Vec::new(),
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

        Ok(Table {
            name: name.to_owned(),
            config: config.clone(),
            shape,
            key,
            codes: cells.codes,
            owner,
            store,
            binding,
        })
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

    fn specs(&self) -> Vec<ColumnSpec<'_>> {
        self.shape
            .iter()
            .zip(&self.codes)
            .map(|(column, codes)| ColumnSpec {
                name: &column.name,
                col: column.col,
                codes: (column.col.encoding == LaneEncoding::TextCode)
                    .then_some(codes as &dyn CodeText),
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

    /// SELECT * : every live row, ascending position.
    pub fn select_all(&self) -> Result<PgRows, IntakeRefusal> {
        let specs = self.specs();
        let reader = self.reader(&specs)?;
        Ok(reader.rows_in(&self.owner.live)?)
    }

    /// SELECT * WHERE filter (Quack Filter over `Col(k)` = column `k`),
    /// three-valued.
    pub fn select_where(
        &self,
        filter: &lance_graph_quack::Filter,
    ) -> Result<PgRows, IntakeRefusal> {
        let specs = self.specs();
        let reader = self.reader(&specs)?;
        let kept = reader.keep_where(filter)?;
        Ok(reader.rows_in(&kept)?)
    }

    /// `COUNT(*)` of the live rows `filter` holds for.
    pub fn count_where(&self, filter: &lance_graph_quack::Filter) -> Result<usize, IntakeRefusal> {
        let specs = self.specs();
        let reader = self.reader(&specs)?;
        Ok(reader.count_where(filter)?)
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

    /// The rows of these keys, in list order, duplicates kept (a bag); an
    /// unknown or deleted key is refused.
    pub fn rows_for_keys(&mut self, keys: &[u64]) -> Result<PgRows, IntakeRefusal> {
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

        self.owner.selections.clear();
        self.owner.selections.push(positions);
        let result = self.read_selection(0);
        self.owner.selections.clear();
        result
    }

    fn read_selection(&self, sel: u16) -> Result<PgRows, IntakeRefusal> {
        let specs = self.specs();
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

    /// Mark a row deleted: its live bit is cleared; its values stay, and
    /// reads refuse or skip it.
    pub fn delete_key(&mut self, key: u64) -> Result<(), IntakeRefusal> {
        let at = self.position_of(key)? as usize;
        if !cfg!(feature = "disable-intake-live") {
            self.owner.live[at / 64] &= !(1u64 << (at % 64));
        }
        Ok(())
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
        let fk_keys: Vec<Option<u64>> = left
            .iter()
            .map(|&p| {
                let at = p as usize;
                bit(&self.owner.presence[f], at)
                    .then(|| key_of(encoding, self.owner.lanes[f].value(at)))
                    .flatten()
            })
            .collect();

        // Each distinct fk value is located once per call; see `rows_for_keys`.
        let mut distinct: Vec<u64> = fk_keys.iter().flatten().copied().collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut resolved: Vec<(u64, Option<u32>)> = Vec::with_capacity(distinct.len());
        for key in distinct {
            let position = right.locate(key)?.filter(|&p| right.is_live(p));
            resolved.push((key, position));
        }

        let mut positions = vec![0u32; left.len()];
        let mut present = vec![0u64; left.len().div_ceil(64)];
        for (i, key) in fk_keys.iter().enumerate() {
            let found = key
                .and_then(|k| resolved.binary_search_by_key(&k, |&(k, _)| k).ok())
                .and_then(|at| resolved[at].1);
            if let Some(position) = found {
                positions[i] = position;
                present[i / 64] |= 1 << (i % 64);
            }
        }

        let left_specs = self.specs();
        let right_specs = right.specs();
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
        for (k, was) in self.shape.iter().enumerate() {
            let mut scratch = Codes::new();
            let fits = rows.iter().all(|row| match row.get(k).copied().flatten() {
                None => true,
                Some("") if was.col.absent == AbsentAs::EmptyText => true,
                Some(text) => lane_value(&was.col, text, &mut scratch).is_some(),
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
}
