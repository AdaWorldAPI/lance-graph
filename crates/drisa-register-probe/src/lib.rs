//! D-DRISA-1 (EXO-153): a sealed DR frame read as a logical register file over
//! borrowed canonical SoA planes. LAB ONLY: no public contract, no new store,
//! no transport, no change to `dt64`, `mask-risc` or the contract.
//!
//! The question under test is whether a witnessed DR frame can play the role
//! of a register file: typed operands name coordinates (a frame, its tiles,
//! a field position, an identity selection), and are resolved to borrowed
//! slices of the canonical SoA only when an instruction executes.
//!
//! What this crate owns, and nothing more:
//! - [`Soa`]: a stand-in for the canonical owner of the values. It records
//!   which source it is, its value revision, and the coordinate assignment
//!   its rows are laid out under (the per-tile map digests of the frame it
//!   was built from).
//! - [`RegisterFile`]: a DESCRIPTOR. It holds a frame, the tile digests it was
//!   bound under, a scope, the demanded field positions, the value revision
//!   (only if values were demanded) and one map-only register: the identity
//!   selection, routed from global ids once. It holds no reference into the
//!   SoA and no copy of any value.
//! - [`MethodOp`]: a typed method over registers, compiled once to an
//!   unchanged `mask-risc` [`Program`].
//!
//! Execution resolves the operands into [`Planes`] that BORROW the SoA's own
//! slices, then calls `lance_graph_mask_risc::execute`. There is no ALU here.
//!
//! The selection register is the one owned buffer (one bit per row). It is a
//! map-only derived value: it depends on the frame and the tile digests, never
//! on field values, so a value write leaves it valid while a value-dependent
//! method must refuse until its value witness is re-taken.

use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::{
    Frame, FrameRevisions, LocalMap, MapRead, MapTable, Scope, TileId, Width,
};
use lance_graph_dt64::ValueDemand;
use lance_graph_mask_risc::{
    execute, execute_into, ExecError, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};

/// Every way a register operand can be invalid. A refusal is always typed:
/// no path returns an empty result or zero in place of a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The frame is not the table's current seal. `then` and `now` are the
    /// revisions, reported, never used as identity.
    StaleFrame {
        then: FrameRevisions,
        now: FrameRevisions,
    },
    /// The table could not read a tile under the frame.
    Map(MapRead),
    /// The SoA belongs to another source than the frame's.
    SourceMismatch { frame: u64, soa: u64 },
    /// The SoA's rows are laid out under a different coordinate assignment
    /// than the one the register was bound under (a permuted or foreign map).
    CoordinateMismatch { tile: TileId },
    /// The reader's scope is not the one the register was bound for.
    ScopeChanged,
    /// A field position past the class basis.
    UnknownField(u8),
    /// A field position the scope does not authorize.
    Unauthorized(u8),
    /// A method names a field the register file did not bind.
    FieldNotBound(u8),
    /// A selection names a global id no bound tile holds.
    UnmappedIdentity(u64),
    /// A selection names a global id twice.
    DuplicateIdentity(u64),
    /// A value-dependent method, but values were not demanded at bind.
    ValuesNotBound,
    /// The SoA's values changed since the register took its value witness,
    /// or the SoA is another incarnation (rebuilt) with a restarted revision.
    StaleValues {
        bound: ValueWitness,
        now: ValueWitness,
    },
    /// A compiled method was resolved against another register file's lane
    /// schema: its lane indices would name different fields here.
    SchemaMismatch,
    /// A method reads the derived register, but none was derived.
    NoDerivedRegister,
    /// `mask-risc` refused the program.
    Exec(ExecError),
}

/// The identity of one value state: which SoA incarnation and which revision
/// of it. The revision alone restarts at zero when a SoA is rebuilt, so it
/// cannot tell a rebuilt SoA from the one a witness was taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueWitness {
    pub incarnation: u64,
    pub revision: u64,
}

/// Every built SoA gets a fresh incarnation, never reused in a process.
static NEXT_INCARNATION: AtomicU64 = AtomicU64::new(1);

/// The canonical owner's values: one row per DR position, tile-major
/// (`row = tile_ordinal * width + local`). Not part of the register file.
#[derive(Debug)]
pub struct Soa {
    source: u64,
    incarnation: u64,
    revision: u64,
    width: Width,
    layout: Vec<(TileId, u64)>,
    n_rows: usize,
    presence: Vec<u64>,
    fields: Vec<Vec<i32>>,
}

impl Soa {
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The value witness a register takes and later compares.
    pub fn witness(&self) -> ValueWitness {
        ValueWitness {
            incarnation: self.incarnation,
            revision: self.revision,
        }
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// The address of a field lane: lets a test show the executed planes
    /// borrow these exact bytes.
    pub fn field_ptr(&self, field: u8) -> *const i32 {
        self.fields[usize::from(field)].as_ptr()
    }

    pub fn presence_ptr(&self) -> *const u64 {
        self.presence.as_ptr()
    }

    /// Write one value by row. Any write bumps the value revision.
    pub fn write(&mut self, row: usize, field: u8, value: i32) {
        self.fields[usize::from(field)][row] = value;
        self.revision += 1;
    }

    /// The row of `global` under `table`'s current frame, by the same layout
    /// rule the SoA was built with. For writes in tests only.
    pub fn row_of(&self, table: &MapTable, global: u64) -> Option<usize> {
        let frame = table.seal();
        let w = self.width.positions();
        for (ordinal, (tile, _)) in self.layout.iter().enumerate() {
            let map = table.map(&frame, *tile).ok()?;
            if let Ok(local) = map.local(global) {
                return Some(ordinal * w + usize::from(local));
            }
        }
        None
    }
}

/// Independent truth: identities and their values, keyed by global id. The
/// oracle reads only this, never a map, a position or a register.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// `(global, present, values)`; absent identities have no row data.
    pub rows: Vec<(u64, bool, Vec<i32>)>,
    pub field_count: usize,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Fixture {
    /// `n` identities with pseudo-random ids, ~90% present, values in
    /// `-100..100` derived from (id, field).
    pub fn new(n: usize, field_count: usize, seed: u64) -> Fixture {
        let rows = (0..n)
            .map(|i| {
                let id = splitmix(seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
                let present = !splitmix(id ^ 0x55).is_multiple_of(10);
                let values = (0..field_count)
                    .map(|f| {
                        (splitmix(id ^ (f as u64 + 1).wrapping_mul(0x9E37)) % 200) as i32 - 100
                    })
                    .collect();
                (id, present, values)
            })
            .collect();
        Fixture { rows, field_count }
    }

    fn get(&self, global: u64) -> Option<&(u64, bool, Vec<i32>)> {
        self.rows.iter().find(|r| r.0 == global)
    }

    /// Oracle: present identities among `selection`.
    pub fn count(&self, selection: &[u64]) -> usize {
        let index: std::collections::BTreeMap<u64, usize> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.0, i))
            .collect();
        selection
            .iter()
            .filter(|g| index.get(g).is_some_and(|&i| self.rows[i].1))
            .count()
    }

    /// Oracle: present identities among `selection` with `field > t`, and the
    /// sum of `sum_field` over them.
    pub fn where_gt(&self, selection: &[u64], field: u8, t: i32, sum_field: u8) -> (usize, i64) {
        let index: std::collections::BTreeMap<u64, usize> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.0, i))
            .collect();
        let mut n = 0;
        let mut s = 0i64;
        for g in selection {
            if let Some(&i) = index.get(g) {
                let r = &self.rows[i];
                if r.1 && r.2[usize::from(field)] > t {
                    n += 1;
                    s += i64::from(r.2[usize::from(sum_field)]);
                }
            }
        }
        (n, s)
    }

    /// Apply a value write to the truth (mirrors [`Soa::write`]).
    pub fn set(&mut self, global: u64, field: u8, value: i32) {
        if let Some(r) = self.rows.iter_mut().find(|r| r.0 == global) {
            r.2[usize::from(field)] = value;
        }
    }

    pub fn ids(&self) -> Vec<u64> {
        self.rows.iter().map(|r| r.0).collect()
    }

    #[doc(hidden)]
    pub fn present(&self, global: u64) -> bool {
        self.get(global).is_some_and(|r| r.1)
    }
}

/// Lay the fixture's identities out over `tiles` tiles of `width`, each tile's
/// local order permuted by `perm_seed`, and append the maps to a fresh table.
pub fn build_table(
    fixture: &Fixture,
    width: Width,
    source: u64,
    perm_seed: u64,
) -> (MapTable, Vec<TileId>) {
    let w = width.positions();
    let ids = fixture.ids();
    let mut table = MapTable::new(source, 1);
    let mut changes = Vec::new();
    let mut tiles = Vec::new();
    for (t, chunk) in ids.chunks(w).enumerate() {
        let mut order: Vec<u64> = chunk.to_vec();
        // Fisher-Yates by a seeded stream: the local order is not the id order.
        let mut r = perm_seed ^ (t as u64);
        for i in (1..order.len()).rev() {
            r = splitmix(r);
            order.swap(i, (r % (i as u64 + 1)) as usize);
        }
        let tile = 1000 + t as TileId;
        changes.push((
            tile,
            Some(LocalMap::prepare(width, &order).expect("fits the tile")),
        ));
        tiles.push(tile);
    }
    table.advance(source, 1, changes);
    (table, tiles)
}

/// Build the canonical SoA for `fixture` under `table`'s current frame.
pub fn build_soa(fixture: &Fixture, table: &MapTable, tiles: &[TileId], width: Width) -> Soa {
    let frame = table.seal();
    let w = width.positions();
    let n_rows = tiles.len() * w;
    let mut presence = vec![0u64; n_rows.div_ceil(64)];
    let mut fields = vec![vec![0i32; n_rows]; fixture.field_count];
    let mut layout = Vec::new();
    for (ordinal, &tile) in tiles.iter().enumerate() {
        let map = table.map(&frame, tile).expect("tile in table");
        layout.push((tile, map.digest()));
        for (local, &g) in map.globals().iter().enumerate() {
            // An id the fixture does not hold is mapped but absent: it has a
            // position and no row data.
            let Some((_, present, values)) = fixture.get(g) else {
                continue;
            };
            let row = ordinal * w + local;
            if *present {
                presence[row / 64] |= 1 << (row % 64);
                for (f, v) in values.iter().enumerate() {
                    fields[f][row] = *v;
                }
            }
        }
    }
    Soa {
        source: table.revisions().source,
        incarnation: NEXT_INCARNATION.fetch_add(1, Ordering::Relaxed),
        revision: 0,
        width,
        layout,
        n_rows,
        presence,
        fields,
    }
}

/// Global → local lookups served by every tile map of `table` (DR's own
/// counter): the routing cost a register file is meant to pay once.
pub fn lookups(table: &MapTable, tiles: &[TileId]) -> u64 {
    let frame = table.seal();
    tiles
        .iter()
        .filter_map(|&t| table.map(&frame, t).ok())
        .map(|m| m.lookups())
        .sum()
}

/// Which checks a register file runs. All on by default; a lab disable run
/// switches exactly one off to show the oracle or a refusal test turns red.
#[derive(Debug, Clone, Copy)]
pub struct Checks {
    pub frame: bool,
    pub coordinates: bool,
    pub values: bool,
    pub scope: bool,
    pub presence: bool,
}

impl Checks {
    pub const ALL: Checks = Checks {
        frame: true,
        coordinates: true,
        values: true,
        scope: true,
        presence: true,
    };
}

/// The register file: a descriptor over a sealed frame. See the crate docs.
#[derive(Debug)]
pub struct RegisterFile {
    frame: Frame,
    width: Width,
    tiles: Vec<(TileId, u64)>,
    scope: Scope,
    source: u64,
    /// Demanded field positions; lane `i` of the executed planes is
    /// `demanded[i]`.
    demanded: Vec<u8>,
    value_rev: Option<ValueWitness>,
    /// The map-only identity selection register, one bit per row.
    selection: Vec<u64>,
    /// A derived, value-dependent predicate register: `(value witness it was
    /// computed at, mask)`. Valid only while the SoA has that witness.
    derived: Option<(ValueWitness, Vec<u64>)>,
    n_rows: usize,
    checks: Checks,
}

/// A typed method over the register file. It names field positions and
/// constants; it never names a row, a value or a reconstructed object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodOp {
    /// How many selected identities are present.
    Count,
    /// Whether any selected identity is present.
    Exists,
    /// How many selected, present identities have `field > t`.
    CountWhereGt { field: u8, t: i32 },
    /// Σ `sum` over selected, present identities with `field > t`.
    SumWhereGt { field: u8, t: i32, sum: u8 },
    /// Count of the derived predicate register (see
    /// [`RegisterFile::derive_gt`]): reads no lane, but the register itself
    /// was computed from values, so it is value-scoped.
    CountDerived,
}

impl MethodOp {
    pub fn demand(self) -> ValueDemand {
        match self {
            MethodOp::Count | MethodOp::Exists => ValueDemand::MapOnly,
            MethodOp::CountWhereGt { .. }
            | MethodOp::SumWhereGt { .. }
            | MethodOp::CountDerived => ValueDemand::Values,
        }
    }
}

/// A method compiled against one register file: an unchanged `mask-risc`
/// program plus the demand it was compiled for.
#[derive(Debug, Clone)]
pub struct Compiled {
    program: Program,
    demand: ValueDemand,
    /// Reads the derived predicate register (plane 2).
    derived: bool,
    /// The lane schema (demanded fields, in lane order) the program's lane
    /// indices were resolved against. Checked before execution.
    schema: Vec<u8>,
}

impl RegisterFile {
    /// Bind registers under `frame`.
    ///
    /// `class_fields` is the class basis size, `authorized` the scope's field
    /// mask, `demanded` the fields the methods will read. The selection is
    /// routed here, once: every id is located in exactly one bound tile.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        table: &MapTable,
        frame: &Frame,
        tiles: &[TileId],
        width: Width,
        scope: Scope,
        class_fields: usize,
        authorized: &WideFieldMask,
        demanded: &[u8],
        demand: ValueDemand,
        selection_ids: &[u64],
        soa: &Soa,
        checks: Checks,
    ) -> Result<RegisterFile, Refusal> {
        current(table, frame)?;
        let mut bound = Vec::with_capacity(tiles.len());
        for &tile in tiles {
            let map = table.map(frame, tile).map_err(Refusal::Map)?;
            bound.push((tile, map.digest()));
        }
        for &f in demanded {
            if usize::from(f) >= class_fields {
                return Err(Refusal::UnknownField(f));
            }
            if !authorized.has(f) {
                return Err(Refusal::Unauthorized(f));
            }
        }
        let w = width.positions();
        let n_rows = tiles.len() * w;
        let mut selection = vec![0u64; n_rows.div_ceil(64)];
        let maps: Vec<_> = tiles
            .iter()
            .map(|&t| table.map(frame, t).map_err(Refusal::Map))
            .collect::<Result<_, _>>()?;
        for &g in selection_ids {
            let mut row = None;
            for (ordinal, map) in maps.iter().enumerate() {
                if let Ok(local) = map.local(g) {
                    row = Some(ordinal * w + usize::from(local));
                    break;
                }
            }
            let row = row.ok_or(Refusal::UnmappedIdentity(g))?;
            let bit = 1u64 << (row % 64);
            if selection[row / 64] & bit != 0 {
                return Err(Refusal::DuplicateIdentity(g));
            }
            selection[row / 64] |= bit;
        }
        let file = RegisterFile {
            frame: frame.clone(),
            width,
            tiles: bound,
            scope,
            source: table.revisions().source,
            demanded: demanded.to_vec(),
            value_rev: None,
            selection,
            derived: None,
            n_rows,
            checks,
        };
        file.check_soa(soa)?;
        Ok(RegisterFile {
            value_rev: matches!(demand, ValueDemand::Values).then_some(soa.witness()),
            ..file
        })
    }

    /// Re-take the value witness without re-routing anything: the selection
    /// register and the tile digests are kept as they are.
    pub fn rebind_values(&mut self, table: &MapTable, soa: &Soa) -> Result<(), Refusal> {
        if self.checks.frame {
            current(table, &self.frame)?;
        }
        self.check_soa(soa)?;
        self.value_rev = Some(soa.witness());
        Ok(())
    }

    fn check_soa(&self, soa: &Soa) -> Result<(), Refusal> {
        if soa.source != self.source {
            return Err(Refusal::SourceMismatch {
                frame: self.source,
                soa: soa.source,
            });
        }
        if self.checks.coordinates {
            if soa.width != self.width || soa.layout.len() != self.tiles.len() {
                return Err(Refusal::CoordinateMismatch {
                    tile: self.tiles.first().map_or(0, |t| t.0),
                });
            }
            for (have, want) in soa.layout.iter().zip(&self.tiles) {
                if have != want {
                    return Err(Refusal::CoordinateMismatch { tile: want.0 });
                }
            }
        }
        Ok(())
    }

    /// Compile a method against this register file: field positions resolve
    /// to lane indices once, here.
    pub fn compile(&self, op: MethodOp) -> Result<Compiled, Refusal> {
        let lane = |f: u8| -> Result<u16, Refusal> {
            self.demanded
                .iter()
                .position(|&d| d == f)
                .map(|i| i as u16)
                .ok_or(Refusal::FieldNotBound(f))
        };
        let sel = if self.checks.presence {
            MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            }
        } else {
            // Disable run only: the selection without the presence plane.
            MaskOp::Or {
                a: Operand::Plane(1),
                b: Operand::Plane(1),
                dst: 0,
            }
        };
        let program = match op {
            MethodOp::Count => Program::new(
                vec![sel],
                Terminal::Count {
                    mask: Operand::Scratch(0),
                },
            ),
            MethodOp::Exists => Program::new(
                vec![sel],
                Terminal::Any {
                    mask: Operand::Scratch(0),
                },
            ),
            MethodOp::CountWhereGt { field, t } => Program::new(
                vec![
                    sel,
                    MaskOp::Pred {
                        pred: Pred::GtI32 {
                            lane: lane(field)?,
                            t,
                        },
                        under: Some(Operand::Scratch(0)),
                        dst: 1,
                    },
                ],
                Terminal::Count {
                    mask: Operand::Scratch(1),
                },
            ),
            MethodOp::CountDerived => Program::new(
                vec![],
                Terminal::Count {
                    mask: Operand::Plane(2),
                },
            ),
            MethodOp::SumWhereGt { field, t, sum } => Program::new(
                vec![
                    sel,
                    MaskOp::Pred {
                        pred: Pred::GtI32 {
                            lane: lane(field)?,
                            t,
                        },
                        under: Some(Operand::Scratch(0)),
                        dst: 1,
                    },
                ],
                Terminal::MaskedSumI32 {
                    mask: Operand::Scratch(1),
                    lane: lane(sum)?,
                },
            ),
        };
        Ok(Compiled {
            program,
            demand: op.demand(),
            derived: matches!(op, MethodOp::CountDerived),
            schema: self.demanded.clone(),
        })
    }

    /// A scratch sized for `compiled` over this register file.
    pub fn scratch(&self, compiled: &Compiled) -> Result<Scratch<'static>, Refusal> {
        Scratch::for_program(&compiled.program, self.n_rows).map_err(Refusal::Exec)
    }

    /// Revalidate every register, then hand the operands to `f` as planes
    /// that borrow the SoA. Nothing is copied.
    pub fn with_planes<R>(
        &self,
        table: &MapTable,
        soa: &Soa,
        scope_now: Scope,
        demand: ValueDemand,
        f: impl FnOnce(&Planes<'_>) -> R,
    ) -> Result<R, Refusal> {
        if self.checks.frame {
            current(table, &self.frame)?;
        }
        if self.checks.scope && scope_now != self.scope {
            return Err(Refusal::ScopeChanged);
        }
        self.check_soa(soa)?;
        if matches!(demand, ValueDemand::Values) {
            let bound = self.value_rev.ok_or(Refusal::ValuesNotBound)?;
            if self.checks.values && bound != soa.witness() {
                return Err(Refusal::StaleValues {
                    bound,
                    now: soa.witness(),
                });
            }
        }
        // Plane 2 is the derived register when one exists; otherwise the
        // selection stands in so every plane spans `n_rows` (no program
        // compiled before `derive_gt` names plane 2).
        let derived: &[u64] = self.derived.as_ref().map_or(&self.selection, |d| &d.1);
        let masks: [&[u64]; 3] = [&soa.presence, &self.selection, derived];
        // Up to four bound lanes, borrowed in place; no Vec on this path.
        let mut lanes: [LaneRef<'_>; 4] = [LaneRef::I32(&[]); 4];
        let k = self.demanded.len().min(4);
        for (slot, &field) in lanes.iter_mut().zip(&self.demanded) {
            *slot = LaneRef::I32(&soa.fields[usize::from(field)]);
        }
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &masks,
            lanes: &lanes[..k],
        };
        Ok(f(&planes))
    }

    /// Execute a compiled method. The only arithmetic is `mask-risc`'s.
    pub fn run(
        &self,
        table: &MapTable,
        soa: &Soa,
        scope_now: Scope,
        compiled: &Compiled,
        scratch: &mut Scratch<'_>,
    ) -> Result<Value, Refusal> {
        if compiled.schema != self.demanded {
            return Err(Refusal::SchemaMismatch);
        }
        if compiled.derived {
            let (at, _) = self.derived.as_ref().ok_or(Refusal::NoDerivedRegister)?;
            if self.checks.values && *at != soa.witness() {
                return Err(Refusal::StaleValues {
                    bound: *at,
                    now: soa.witness(),
                });
            }
        }
        self.with_planes(table, soa, scope_now, compiled.demand, |planes| {
            execute(&compiled.program, planes, scratch, None)
        })?
        .map_err(Refusal::Exec)
    }

    /// Compute the derived register `selected ∧ present ∧ field > t` through
    /// `mask-risc` (a `Keep` into an owned mask), stamped with the SoA's value
    /// revision. Requires values to be bound and current.
    pub fn derive_gt(
        &mut self,
        table: &MapTable,
        soa: &Soa,
        scope_now: Scope,
        field: u8,
        t: i32,
    ) -> Result<(), Refusal> {
        let compiled = self.compile(MethodOp::CountWhereGt { field, t })?;
        let program = Program::new(
            compiled.program.ops.clone(),
            Terminal::Keep {
                mask: Operand::Scratch(1),
            },
        );
        let mut scratch = Scratch::for_program(&program, self.n_rows).map_err(Refusal::Exec)?;
        let mut words = vec![0u64; self.n_rows.div_ceil(64)];
        self.derived = None;
        self.with_planes(table, soa, scope_now, ValueDemand::Values, |planes| {
            execute_into(
                &program,
                planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(&mut words),
            )
        })?
        .map_err(Refusal::Exec)?;
        self.derived = Some((soa.witness(), words));
        Ok(())
    }

    /// Bytes this register file owns: the selection register and, if
    /// derived, the predicate register. Never a field value.
    pub fn owned_bytes(&self) -> usize {
        self.selection.len() * 8 + self.derived.as_ref().map_or(0, |d| d.1.len() * 8)
    }
}

fn current(table: &MapTable, frame: &Frame) -> Result<(), Refusal> {
    let now = table.seal();
    if frame.same_issue(&now) {
        Ok(())
    } else {
        Err(Refusal::StaleFrame {
            then: frame.revisions(),
            now: now.revisions(),
        })
    }
}

/// Arm A of the bench: the same programs over planes the caller assembles
/// directly from the SoA and a caller-built selection. No register checks.
pub fn run_direct(
    program: &Program,
    soa: &Soa,
    selection: &[u64],
    demanded: &[u8],
    scratch: &mut Scratch<'_>,
) -> Result<Value, ExecError> {
    let masks: [&[u64]; 2] = [&soa.presence, selection];
    let mut lanes: [LaneRef<'_>; 4] = [LaneRef::I32(&[]); 4];
    let k = demanded.len().min(4);
    for (slot, &field) in lanes.iter_mut().zip(demanded) {
        *slot = LaneRef::I32(&soa.fields[usize::from(field)]);
    }
    let planes = Planes {
        n_rows: soa.n_rows,
        masks: &masks,
        lanes: &lanes[..k],
    };
    execute(program, &planes, scratch, None)
}

impl Compiled {
    pub fn program(&self) -> &Program {
        &self.program
    }
}

impl RegisterFile {
    /// The selection register's words (read-only), for arm A of the bench.
    pub fn selection(&self) -> &[u64] {
        &self.selection
    }
}
