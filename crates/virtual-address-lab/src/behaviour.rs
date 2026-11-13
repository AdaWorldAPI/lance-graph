//! ON -> WHEN -> demanded operands, composed from existing authorities only
//! (plan section 17). No dispatcher, evaluator of effects or write-back.
//!
//! - **Whose behaviour**: the classid the position routes to, at the captured
//!   route version and now ([`behaviour`], [`admit_routed`]). `BindStore::admit`
//!   alone checks a reference against the store, not against a route.
//! - **WHEN**: a two-valued guard of `Res::I32` comparisons, over lanes the
//!   source lends under a `LaneBinding`, lowered by Quack and run by
//!   mask-RISC ([`guard_mask`]). The `SetExpr -> Filter` translation and the
//!   reading of a guard lane as a class field position are this lab's; no
//!   library seam does either. A mask-plane guard (`Sym`) and a Kleene guard
//!   are refused: nothing issues the planes they need.
//! - **Operands**: read through a binding made over the admitted method's
//!   `reads` only ([`operands`]); the binding refuses every other field.

use std::cell::Cell;

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Cmp as SetCmp, Logic, Res, SetExpr};
use lance_graph_dt64::{
    CoordSpace, Guard, LaneBinding, MethodBinding, MethodRef, Resolved, SchemaRev,
};
use lance_graph_mask_risc::{execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

use crate::grid::{Grid, World};
use crate::method::Method;
use crate::view::Refused;

/// Three separate counts. `effects` stays 0: nothing here runs an effect.
#[derive(Debug, Default)]
pub struct Ledgers {
    /// Guard programs run.
    pub guard_runs: Cell<u64>,
    /// Source rows a guard program scanned (rows x guard lanes).
    pub guard_rows: Cell<u64>,
    /// Operand values read for a DO.
    pub operands: Cell<u64>,
    /// Effects executed.
    pub effects: Cell<u64>,
}

/// The behaviour of the position: the methods of the classid it routes to at
/// the world's captured route version. Reads no lane and no value.
pub fn behaviour(world: &World) -> Result<Resolved<'_, '_>, Refused> {
    world
        .store
        .resolve_at(
            &world.view,
            world.route.version,
            world.route.position,
            SchemaRev(1),
        )
        .map_err(Refused::Bind)
}

/// Admit `r` for the world's position: its class must be the one the
/// position routed to at the captured version AND routes to now (the
/// comparison R2IL's `check_route` makes), then `BindStore::admit`.
pub fn admit_routed<'w>(world: &'w World, r: &MethodRef) -> Result<&'w MethodBinding, Refused> {
    let store = &world.store;
    let position = world.route.position;
    let captured = store
        .routed_at(world.route.version, position)
        .map_err(Refused::Bind)?;
    let now = store
        .routed_at(store.current_version(), position)
        .map_err(Refused::Bind)?;
    if captured != Some(r.class) || now != Some(r.class) {
        return Err(Refused::NotRoutedBehaviour {
            position,
            routed: now,
            offered: r.class,
        });
    }
    store.admit(r).map_err(Refused::Bind)
}

/// The class fields a guard reads, ascending and distinct. A guard lane is
/// read as a class field position (this lab's reading).
fn guard_fields(e: &SetExpr, out: &mut Vec<u8>) -> Result<(), Refused> {
    match e {
        SetExpr::Sym(sym) => return Err(Refused::GuardPlaneUnissued { sym: *sym }),
        SetExpr::All | SetExpr::Empty => return Err(Refused::GuardUnsupported("constant set")),
        SetExpr::Res(Res::U32 { .. }) => return Err(Refused::GuardUnsupported("unsigned lane")),
        SetExpr::Res(Res::I32 { lane, .. }) => {
            let field = u8::try_from(*lane).map_err(|_| Refused::OutsideTheView)?;
            if !out.contains(&field) {
                out.push(field);
            }
        }
        SetExpr::Not(x) => guard_fields(x, out)?,
        SetExpr::And(cs) | SetExpr::Or(cs) => {
            for c in cs {
                guard_fields(c, out)?;
            }
        }
    }
    Ok(())
}

/// The guard as a Quack filter over planes whose lane `i` is `fields[i]`.
fn to_filter(e: &SetExpr, fields: &[u8]) -> Filter {
    match e {
        SetExpr::Res(Res::I32 { lane, op, t }) => {
            let idx = fields
                .iter()
                .position(|&f| u16::from(f) == *lane)
                .expect("collected by guard_fields");
            let cmp = match op {
                SetCmp::Lt => Cmp::LtI32(*t),
                SetCmp::Le => Cmp::LeI32(*t),
                SetCmp::Gt => Cmp::GtI32(*t),
                SetCmp::Ge => Cmp::GeI32(*t),
                SetCmp::Eq => Cmp::EqI32(*t),
                SetCmp::Ne => Cmp::NeI32(*t),
            };
            Filter::Cmp(Col(idx as u16), cmp)
        }
        SetExpr::Not(x) => Filter::Not(Box::new(to_filter(x, fields))),
        SetExpr::And(cs) => Filter::And(cs.iter().map(|c| to_filter(c, fields)).collect()),
        SetExpr::Or(cs) => Filter::Or(cs.iter().map(|c| to_filter(c, fields)).collect()),
        // `guard_fields` refused these before any filter is built.
        SetExpr::Sym(_) | SetExpr::All | SetExpr::Empty | SetExpr::Res(Res::U32 { .. }) => {
            unreachable!("refused by guard_fields")
        }
    }
}

/// Where a guard holds: one bit per SOURCE row, with the binding it was
/// evaluated under. A membership set: it carries no multiplicity.
#[derive(Debug)]
pub struct GuardMask {
    bits: Vec<u64>,
    rows: usize,
    binding: LaneBinding,
}

impl GuardMask {
    /// Whether the guard holds at source row `position`. The binding's check
    /// runs first, so a mask evaluated under a frame the source has left, or
    /// asked of another source, is refused, never read.
    pub fn holds(&self, world: &World, grid: &Grid, position: u32) -> Result<bool, Refused> {
        self.binding
            .check(&world.store, &world.view, grid.frame_store(), grid)
            .map_err(Refused::Lane)?;
        let p = position as usize;
        if p >= self.rows {
            return Err(Refused::PositionOutsideLane { at: 0, position });
        }
        Ok(self.bits[p / 64] >> (p % 64) & 1 == 1)
    }

    /// Set bits: distinct source rows where the guard holds.
    pub fn rows_held(&self) -> u64 {
        self.bits.iter().map(|w| u64::from(w.count_ones())).sum()
    }
}

/// Evaluate `guard` over the source's own lanes, in row order `selection`.
/// Quack lowers it, mask-RISC runs it; the lanes are lent under a binding made
/// over the guard's fields only.
pub fn guard_mask(
    world: &World,
    grid: &Grid,
    guard: &Guard,
    selection: CoordSpace,
    ledgers: &Ledgers,
) -> Result<GuardMask, Refused> {
    if guard.logic != Logic::TwoValued {
        return Err(Refused::GuardUnsupported("three-valued logic"));
    }
    let mut fields = Vec::new();
    guard_fields(&guard.expr, &mut fields)?;
    let binding = world
        .bind_fields(grid, &WideFieldMask::from_positions(&fields), selection)
        .map_err(Refused::Lane)?;
    let mut lanes = Vec::with_capacity(fields.len());
    for &f in &fields {
        lanes.push(
            binding
                .lane(&world.store, &world.view, grid.frame_store(), grid, f)
                .map_err(Refused::Lane)?,
        );
    }
    if !lanes.iter().all(|l| matches!(l, LaneRef::I32(_))) {
        return Err(Refused::LaneKind);
    }
    let rows = grid.len();
    let planes = Planes {
        n_rows: rows,
        masks: &[],
        lanes: &lanes,
    };
    let query = Query {
        filter: to_filter(&guard.expr, &fields),
        agg: Agg::Rows,
    };
    let lowered = |e: String| Refused::GuardLower(e);
    let program = lower(&query).map_err(|e| lowered(format!("{e:?}")))?;
    let mut scratch =
        Scratch::for_program(&program, rows).map_err(|e| lowered(format!("{e:?}")))?;
    let mut bits = vec![0u64; words_for(rows)];
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut bits),
    )
    .map_err(|e| lowered(format!("{e:?}")))?;
    ledgers.guard_runs.set(ledgers.guard_runs.get() + 1);
    ledgers
        .guard_rows
        .set(ledgers.guard_rows.get() + (rows * fields.len()) as u64);
    Ok(GuardMask {
        bits,
        rows,
        binding,
    })
}

/// The operands the admitted method declares, at `ordinal` of the method's
/// selection, with the binding they were read through. The binding is made
/// over `admitted.reads` ONLY, so it refuses every other field.
pub fn operands(
    world: &World,
    grid: &Grid,
    admitted: &MethodBinding,
    method: Method,
    ordinal: usize,
    ledgers: &Ledgers,
) -> Result<(LaneBinding, Vec<(u8, i32)>), Refused> {
    let binding = world
        .bind_fields(grid, &admitted.reads, grid.space())
        .map_err(Refused::Lane)?;
    let mut values = Vec::with_capacity(admitted.reads.count() as usize);
    for field in (0..=u8::MAX).filter(|&f| admitted.reads.has(f)) {
        let (lane, positions) = binding
            .selected(
                &world.store,
                &world.view,
                grid.frame_store(),
                grid,
                field,
                method.selection,
            )
            .map_err(Refused::Lane)?;
        let LaneRef::I32(lane) = lane else {
            return Err(Refused::LaneKind);
        };
        let position = *positions.get(ordinal).ok_or(Refused::OutsideTheView)?;
        let value = *lane
            .get(position as usize)
            .ok_or(Refused::PositionOutsideLane {
                at: ordinal,
                position,
            })?;
        ledgers.operands.set(ledgers.operands.get() + 1);
        values.push((field, value));
    }
    Ok((binding, values))
}
