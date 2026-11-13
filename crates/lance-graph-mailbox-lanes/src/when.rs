//! WHEN over one publication of a mailbox (D-VAS-C3 to C6).
//!
//! A guard of an admitted method, evaluated over the mailbox's own columns:
//!
//! 1. Quack turns the guard's set into a filter
//!    ([`lance_graph_quack::set::lower_set`]).
//! 2. A guard lane is a field position of the routed class (DT64 refuses any
//!    other at `register`). The lanes are bound through DT64 over EXACTLY the
//!    fields the guard names, under this publication's frame, and lent by the
//!    socket: the guard reads the owner's arrays in place.
//! 3. Quack lowers the filter, mask-risc runs it. This module evaluates
//!    nothing itself.
//!
//! The answer is a membership mask over the lane's rows (the populated
//! prefix). It carries the binding it was evaluated under, so it is refused
//! after a re-route and under any other publication. It says nothing about a
//! bag: a selection that lists a row twice still has one bit.
//!
//! Refused by name, not answered:
//!
//! - **A mask plane** ([`SetExpr::Sym`](lance_graph_contract::set_algebra::SetExpr::Sym)).
//!   No authority issues a plane with a frame, a row order and a population
//!   for a mailbox; the library readings of `Sym` disagree (a plane index in
//!   `set_lower`, a digest or a field elsewhere). D-VAS-C5 is open.
//! - **A three-valued guard.** Quack evaluates one given a validity plane per
//!   nullable lane; the mailbox states none for its columns, and reading
//!   "every populated row is valid" would be this module's assertion.
//!   D-VAS-C6 is open for the mailbox.
//! - **A comparison the column's kind does not support.** `meta` and the two
//!   cycle stamps are `u32` (equality and inequality); `edges` and `temporal`
//!   are `u64` and no residual compares a 64-bit lane. The executor refuses
//!   the kind itself.

use lance_graph_contract::class_view::{ClassView, WideFieldMask};
use lance_graph_contract::set_algebra::Logic;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{
    BindStore, Guard, LaneBinding, LaneRefusal, LaneRequest, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{execute_into, words_for, ExecError, Foreign, Out, Planes, Scratch};
use lance_graph_quack::set::{lower_set, SetLowerError, SetLowered};
use lance_graph_quack::{lower, Agg, LowerError, Query};

use crate::MailboxLanes;

/// Why a guard was not evaluated, or an answer not given.
#[derive(Debug, Clone, PartialEq)]
pub enum WhenRefusal {
    /// The guard's set could not be spelled as a filter.
    Set(SetLowerError),
    /// The guard names a mask plane; this socket issues none.
    PlaneUnissued { plane: u16 },
    /// The guard is three-valued; the mailbox states no validity.
    ValidityUnstated,
    /// The guard names a lane no field position can be.
    FieldOutOfRange { lane: u16 },
    /// DT64 or the socket refused the lanes (route, frame, basis, scope).
    Lane(LaneRefusal),
    /// Quack could not lower the filter.
    Lower(LowerError),
    /// mask-risc refused the program (for example the lane kind).
    Exec(ExecError),
    /// The row is not in the lane.
    RowOutsideLane { row: u32, rows: usize },
}

/// What a guard is evaluated for: the routed position, the schema revision
/// and the reader scope the caller asserts.
#[derive(Debug, Clone, Copy)]
pub struct WhenRequest {
    pub route: RouteAt,
    pub rev: SchemaRev,
    pub scope: Scope,
}

/// Where a guard holds: one bit per row of the lane, with the binding it was
/// evaluated under.
#[derive(Debug)]
pub struct Fired {
    bits: Vec<u64>,
    rows: usize,
    fields: Vec<u8>,
    binding: LaneBinding,
}

impl Fired {
    /// Rows the guard was evaluated over (the populated prefix).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Distinct rows where the guard holds.
    pub fn count(&self) -> u64 {
        self.bits.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// The class fields the guard read, ascending. Empty for a constant
    /// guard, which reads none.
    pub fn fields(&self) -> &[u8] {
        &self.fields
    }

    /// Whether the guard holds at `row`. The binding is checked first, so an
    /// answer is refused after a re-route, under another publication and for
    /// another source.
    pub fn holds<V: ClassView + ?Sized, const N: usize>(
        &self,
        store: &BindStore,
        view: &V,
        lanes: &MailboxLanes<'_, N>,
        row: u32,
    ) -> Result<bool, WhenRefusal> {
        self.binding
            .check(store, view, lanes.frame_store(), lanes)
            .map_err(WhenRefusal::Lane)?;
        let r = row as usize;
        if r >= self.rows {
            return Err(WhenRefusal::RowOutsideLane {
                row,
                rows: self.rows,
            });
        }
        Ok(self.bits[r / 64] >> (r % 64) & 1 == 1)
    }
}

/// Evaluate `guard` over the columns `lanes` publishes, for the class
/// `request.route` routes to.
pub fn when<V: ClassView + ?Sized, const N: usize>(
    lanes: &MailboxLanes<'_, N>,
    store: &BindStore,
    view: &V,
    request: WhenRequest,
    guard: &Guard,
) -> Result<Fired, WhenRefusal> {
    let lowered = lower_set(&guard.expr, guard.logic).map_err(WhenRefusal::Set)?;
    let set = match &lowered {
        SetLowered::Filter(set) => Some(set),
        SetLowered::All | SetLowered::Empty => None,
    };
    if let Some(&plane) = set.and_then(|s| s.planes().first()) {
        return Err(WhenRefusal::PlaneUnissued { plane });
    }
    if set.is_some() && guard.logic != Logic::TwoValued {
        return Err(WhenRefusal::ValidityUnstated);
    }
    let mut fields = Vec::new();
    for &lane in set.map_or(&[][..], |s| s.lanes()) {
        fields.push(u8::try_from(lane).map_err(|_| WhenRefusal::FieldOutOfRange { lane })?);
    }

    // The binding is made for a constant guard too: the answer is tied to
    // the route and the publication either way.
    let frame = lanes.seal();
    let binding = store
        .bind_lanes(
            view,
            lanes.frame_store(),
            lanes,
            LaneRequest {
                route: request.route,
                rev: request.rev,
                fields: &WideFieldMask::from_positions(&fields),
                frame: &frame,
                scope: request.scope,
                selection: lanes.space(),
            },
        )
        .map_err(WhenRefusal::Lane)?;

    let rows = lanes.rows();
    let mut bits = vec![0u64; words_for(rows)];
    match (&lowered, set) {
        (_, Some(set)) => {
            let mut lent = Vec::with_capacity(fields.len());
            for &field in &fields {
                lent.push(
                    binding
                        .lane(store, view, lanes.frame_store(), lanes, field)
                        .map_err(WhenRefusal::Lane)?,
                );
            }
            // Column `i` of the filter reads `set.lanes()[i]`, the order the
            // lanes were lent in.
            let filter = set.two_valued(set.lanes()).map_err(WhenRefusal::Set)?;
            let program = lower(&Query {
                filter,
                agg: Agg::Rows,
            })
            .map_err(WhenRefusal::Lower)?;
            let planes = Planes {
                n_rows: rows,
                masks: &[],
                lanes: &lent,
            };
            let mut scratch = Scratch::for_program(&program, rows).map_err(WhenRefusal::Exec)?;
            execute_into(
                &program,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(&mut bits),
            )
            .map_err(WhenRefusal::Exec)?;
        }
        (SetLowered::All, None) => {
            for r in 0..rows {
                bits[r / 64] |= 1 << (r % 64);
            }
        }
        _ => {}
    }
    Ok(Fired {
        bits,
        rows,
        fields,
        binding,
    })
}
