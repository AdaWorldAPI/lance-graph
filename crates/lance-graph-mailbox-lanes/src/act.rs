//! ON, WHEN and DO for one publication of a mailbox, and the owner's
//! write-back (D-VAS-D2, D3, D4).
//!
//! Every step is an existing authority's; this module only orders them.
//!
//! 1. **Admission.** [`BindStore::admit_routed`]: the method must be the
//!    routed class's own. Nothing is read from the mailbox before it passes.
//! 2. **WHEN.** [`crate::when::when`]: Quack lowers the guard, mask-risc runs
//!    it over the columns the socket lends. The kept rows are the rows the
//!    method fires on.
//! 3. **Operands.** The lanes are bound through DT64 over the method's
//!    `reads` only.
//! 4. **DO.** `lance_graph_r2il::binding`: the effects are prepared and run
//!    as R2IL folds through mask-risc, with the guard's mask as the gate
//!    plane, so every fold reads only rows the guard keeps. An execution
//!    permit for the exact method reference is required.
//! 5. **Write-back.** A fold result is one scalar per written field. For
//!    each one whose field names a column the owner lets a method write, the
//!    result becomes an [`Intent`]: column, value, the fired rows as one bit
//!    per row, and the owner's cycle at publication. The write goes through
//!    the owner's own cycle-aware mutator, `MailboxSoA::write_row`, which
//!    refuses a write made for another cycle. A cell that already holds the
//!    value is not written.
//! 6. **The next ON.** [`cascade`] turns the columns a round really changed
//!    into the next round's ON, until a round changes nothing or the
//!    caller's bound is reached.
//!
//! # The read-to-write boundary
//!
//! The publication ends before any write, so every frame, binding and lent
//! lane of the read is gone by construction; only an owned, row-bounded
//! [`Intent`] remains. Nothing ties that intent to the bytes it was computed
//! from: the owner's setters and public columns change data without moving
//! its cycle, and the mailbox has no revision that every write moves.
//!
//! So a write is only made inside one exclusive borrow of the owner that
//! also covers the read it rests on:
//!
//! - [`cascade`] holds `&mut` on the owner from a round's read to its
//!   writes.
//! - [`apply`] takes an intent as a statement of what the caller expects.
//!   Under its own exclusive borrow it runs the intent's method again and
//!   writes only when the method asks for exactly that write now. An intent
//!   whose guard rows or value no longer hold is refused
//!   ([`ApplyRefusal::Stale`]) and nothing is written.
//!
//! No revision counter, log or copy of the read columns is kept.
//!
//! # What a method means here
//!
//! DT64 states ON as a field mask and WHEN as a guard over the population.
//! This module reads them as: a change of an ON field makes the method a
//! candidate; the guard's rows are where it fires; an effect's scalar is
//! written to its target field at every fired row. A reading that fires only
//! on the row that changed is not implemented and no contract states it.
//!
//! # Limits
//!
//! - Writable columns: `meta` and `temporal`. The two cycle stamps are the
//!   owner's; a method that targets one is refused whole
//!   ([`ActRefusal::NotWritable`]). `edges` is lendable and not written here.
//!   A field naming a mailbox column the socket cannot lend at all (`sigma`,
//!   `qualia`, ...) refuses the method too
//!   ([`ActRefusal::UnsupportedColumn`]).
//! - Folds over the mailbox's lanes: predicates on `u32` columns
//!   (`IntEqual`), Boolean combinations and counts. `SUM`, `MIN` and `MAX`
//!   need an `i32` lane, which the mailbox does not have; R2IL refuses them.
//! - A field that names no mailbox column at all (a method's scratch mask, an
//!   intermediate scalar) is never written; its scalars are returned in
//!   [`Acted::unwritten`].
//! - Rounds are evaluated against one publication each: every method of a
//!   round reads the state the round started with; writes land afterwards,
//!   in ascending [`MethodId`] order, so on one cell the later method wins.

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::class_view::ClassView;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{Focus, Guards, Scope};
use lance_graph_dt64::{
    BindStore, LaneRefusal, LaneRequest, MethodId, MethodRef, Refusal, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{Foreign, Planes};
use lance_graph_r2il::binding::{
    prepare, prepare_gated, run, run_gated, ExecPermit, ExecRefusal, FieldLanes, Gate, R2ilLens,
    Receipt, Resolution,
};
use lance_graph_r2il::FoldDialect;

use crate::when::{when, WhenRefusal, WhenRequest};
use crate::{Column, ColumnRefusal, MailboxLanes};

/// What a method is run under: the routed position, the class's schema
/// revision, the reader scope and focus the caller asserts, the R2IL lens
/// the effects' words are read under, and the execution permit.
#[derive(Debug, Clone, Copy)]
pub struct ActContext<'a> {
    pub route: RouteAt,
    pub rev: SchemaRev,
    pub scope: Scope,
    pub focus: Focus,
    pub lens: R2ilLens,
    pub permit: &'a ExecPermit,
}

/// Why a method did not act.
#[derive(Debug, Clone, PartialEq)]
pub enum ActRefusal {
    /// The position does not route to the method's class, or the store
    /// refused the reference or the class.
    Admission(Refusal),
    /// The guard could not be evaluated.
    When(WhenRefusal),
    /// DT64 or the socket refused the operand lanes.
    Lane(LaneRefusal),
    /// R2IL refused to prepare or run the effects.
    Exec(ExecRefusal),
    /// An effect targets a mailbox column a method may not write.
    NotWritable { field: u8, column: Column },
    /// An effect targets a mailbox column of a kind the socket does not
    /// handle. It is a column of the owner, so it is not a scratch field.
    UnsupportedColumn { field: u8, kind: &'static str },
    /// A result does not fit the column it would be written to.
    ValueOutOfRange { field: u8, value: i64 },
}

/// A write a method asks the owner to make: `value` into `column` at every
/// row of `rows`, for the cycle the owner was in when the method read. Owned
/// and bounded by the row count (one bit per row); it borrows nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    method: MethodRef,
    mailbox: MailboxId,
    cycle: u32,
    column: Column,
    value: u64,
    rows: Vec<u64>,
    n_rows: usize,
}

impl Intent {
    /// The column the write targets.
    pub fn column(&self) -> Column {
        self.column
    }

    /// The value written at every fired row.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// The same write: method, owner, column, value and rows. The cycle is
    /// left to the owner's own check.
    fn same_write(&self, other: &Intent) -> bool {
        self.method == other.method
            && self.mailbox == other.mailbox
            && self.column == other.column
            && self.value == other.value
            && self.n_rows == other.n_rows
            && self.rows == other.rows
    }
}

/// What one admitted method did during one publication.
#[derive(Debug, Clone, PartialEq)]
pub struct Acted {
    /// The writes it asks for, in ascending field order.
    pub intents: Vec<Intent>,
    /// The rows it fired on, ascending.
    pub rows: Vec<u32>,
    /// Rows the guard program ran over (0 for an unguarded method).
    pub guard_rows: u64,
    /// R2IL's receipt: plan, frame, steps, programs, scalar outputs.
    pub receipt: Receipt,
    /// Scalar outputs whose field names no mailbox column.
    pub unwritten: Vec<(u8, i64)>,
}

/// The column a class field names, if it names one a method may write.
fn writable(iri: &str, field: u8) -> Result<Option<Column>, ActRefusal> {
    match Column::from_iri(iri) {
        Ok(column @ (Column::Meta | Column::Temporal)) => Ok(Some(column)),
        Ok(column) => Err(ActRefusal::NotWritable { field, column }),
        Err(ColumnRefusal::UnsupportedKind(kind)) => {
            Err(ActRefusal::UnsupportedColumn { field, kind })
        }
        // No mailbox column at all: a scratch field of the class.
        Err(ColumnRefusal::Unknown) => Ok(None),
    }
}

/// Run `method` over the columns `lanes` publishes. Writes nothing: the
/// answer is what the method asks the owner to write.
pub fn act<V: ClassView + ?Sized, const N: usize>(
    lanes: &MailboxLanes<'_, N>,
    store: &BindStore,
    view: &V,
    ctx: &ActContext<'_>,
    method: &MethodRef,
) -> Result<Acted, ActRefusal> {
    let admitted = store
        .admit_routed(ctx.route, method)
        .map_err(ActRefusal::Admission)?;
    let class = store
        .resolve_at(view, ctx.route.version, ctx.route.position, ctx.rev)
        .map_err(ActRefusal::Admission)?;
    let iri = |field: u8| {
        class
            .fields
            .get(usize::from(field))
            .map_or("", |f| f.predicate_iri.as_str())
    };
    // Refuse a forbidden target before anything is read or run.
    for effect in &admitted.effects {
        writable(iri(effect.target), effect.target)?;
    }

    let fired = match &admitted.when {
        Some(guard) => Some(
            when(
                lanes,
                store,
                view,
                WhenRequest {
                    route: ctx.route,
                    rev: ctx.rev,
                    scope: ctx.scope,
                },
                guard,
            )
            .map_err(ActRefusal::When)?,
        ),
        None => None,
    };

    // Operands: bound over the method's `reads`, and nothing else.
    let frames = lanes.frame_store();
    let frame = lanes.seal();
    let binding = store
        .bind_lanes(
            view,
            frames,
            lanes,
            LaneRequest {
                route: ctx.route,
                rev: ctx.rev,
                fields: &admitted.reads,
                frame: &frame,
                scope: ctx.scope,
                selection: lanes.space(),
            },
        )
        .map_err(ActRefusal::Lane)?;
    let mut lent = Vec::new();
    let mut field_lanes = vec![None; class.fields.len()];
    for field in (0..=u8::MAX).filter(|&f| admitted.reads.has(f)) {
        let lane = binding
            .lane(store, view, frames, lanes, field)
            .map_err(ActRefusal::Lane)?;
        if let Some(slot) = field_lanes.get_mut(usize::from(field)) {
            *slot = u16::try_from(lent.len()).ok();
        }
        lent.push(lane);
    }

    // DO: R2IL over mask-risc, gated by the guard's rows.
    let rows = lanes.rows();
    let masks: Vec<&[u64]> = fired.iter().map(|f| f.bits()).collect();
    let planes = Planes {
        n_rows: rows,
        masks: &masks,
        lanes: &lent,
    };
    let foreign = Foreign::NONE;
    let mut dialect = FoldDialect::new(&planes, &foreign);
    let guards = Guards {
        frame,
        focus: ctx.focus,
        scope: ctx.scope,
    };
    let resolution = Resolution {
        method,
        schema_rev: ctx.rev,
        route: (ctx.route.version, ctx.route.position),
        guards: &guards,
    };
    let now = (ctx.focus, ctx.scope);
    let fields = FieldLanes {
        lanes: &field_lanes,
    };
    let receipt = match &method.guard {
        Some(witness) if fired.is_some() => {
            let plan = prepare_gated(store, view, ctx.lens, &resolution, frames, now)
                .map_err(ActRefusal::Exec)?;
            let gate = Gate {
                plane: 0,
                guard: witness.clone(),
            };
            run_gated(
                &plan,
                store,
                frames,
                now,
                ctx.permit,
                fields,
                &mut dialect,
                &gate,
            )
        }
        _ => {
            let plan = prepare(store, view, ctx.lens, &resolution, frames, now)
                .map_err(ActRefusal::Exec)?;
            run(&plan, store, frames, now, ctx.permit, fields, &mut dialect)
        }
    }
    .map_err(ActRefusal::Exec)?;

    // The fired rows: the guard's, or every row for an unguarded method.
    let mut bits = vec![0u64; rows.div_ceil(64)];
    match &fired {
        Some(f) => bits.copy_from_slice(f.bits()),
        None => (0..rows).for_each(|r| bits[r / 64] |= 1 << (r % 64)),
    }
    let fired_rows: Vec<u32> = (0..rows as u32)
        .filter(|&r| bits[r as usize / 64] >> (r % 64) & 1 == 1)
        .collect();

    let (mailbox, cycle) = lanes.owner_cycle();
    let (mut intents, mut unwritten) = (Vec::new(), Vec::new());
    for &(field, value) in &receipt.output.values {
        let Some(column) = writable(iri(field), field)? else {
            unwritten.push((field, value));
            continue;
        };
        let fits = match column {
            Column::Meta => u32::try_from(value).map(u64::from).ok(),
            _ => u64::try_from(value).ok(),
        };
        intents.push(Intent {
            method: method.clone(),
            mailbox,
            cycle,
            column,
            value: fits.ok_or(ActRefusal::ValueOutOfRange { field, value })?,
            rows: bits.clone(),
            n_rows: rows,
        });
    }
    Ok(Acted {
        intents,
        rows: fired_rows,
        guard_rows: fired.as_ref().map_or(0, |f| f.rows() as u64),
        receipt,
        unwritten,
    })
}

/// Why an intent was not handed to the owner at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ApplyRefusal {
    /// The intent was computed from another mailbox.
    ForeignOwner { intent: MailboxId, owner: MailboxId },
    /// The intent's method no longer asks for this write: its guard keeps
    /// other rows, or it computes another value, than when the intent was
    /// made. Nothing was written.
    Stale { column: Column },
    /// The intent's method does not run against the owner as it is now.
    /// Nothing was written.
    NotReproduced(ActRefusal),
}

/// What the owner did with an intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Applied {
    /// Cells the owner wrote (each stamped with the intent's cycle).
    pub changed: u32,
    /// Cells that already held the value; not written.
    pub unchanged: u32,
    /// Written cells whose row had another write-cycle stamp before: the
    /// owner stamps `last_write_cycle` on every accepted write, so these are
    /// changes of that column.
    pub stamped: u32,
    /// The owner's refusal that stopped the intent, if any. Cells before it
    /// were written; none after it.
    pub refused: Option<WriteOutcome>,
}

/// Hand `intent` to its owner, if its method still asks for it.
///
/// The intent was computed under a publication that has ended, and the owner
/// may have changed since without moving its cycle. So the method is run
/// again here, under this call's exclusive borrow of the owner, and the
/// intent is written only when one of the writes the method asks for now is
/// the same write (method, column, value, rows). Otherwise it is refused and
/// nothing is written.
///
/// Every write goes through `MailboxSoA::write_row` with the cycle the
/// intent was computed in, so the owner still refuses it once it has moved
/// to another cycle.
///
/// # Errors
///
/// [`ApplyRefusal`]; in every case no cell and no stamp has changed.
pub fn apply<V: ClassView + ?Sized, const N: usize>(
    owner: &mut MailboxSoA<N>,
    store: &BindStore,
    view: &V,
    ctx: &ActContext<'_>,
    intent: &Intent,
) -> Result<Applied, ApplyRefusal> {
    if owner.mailbox_id() != intent.mailbox {
        return Err(ApplyRefusal::ForeignOwner {
            intent: intent.mailbox,
            owner: owner.mailbox_id(),
        });
    }
    let now = {
        let lanes = MailboxLanes::new(owner.published(), ctx.scope);
        act(&lanes, store, view, ctx, &intent.method).map_err(ApplyRefusal::NotReproduced)?
    };
    if !now.intents.iter().any(|i| i.same_write(intent)) {
        return Err(ApplyRefusal::Stale {
            column: intent.column,
        });
    }
    Ok(write(owner, intent))
}

/// Write `intent` to the owner. The caller holds the owner exclusively since
/// the read the intent was computed from.
fn write<const N: usize>(owner: &mut MailboxSoA<N>, intent: &Intent) -> Applied {
    let mut applied = Applied::default();
    for row in 0..intent.n_rows.min(owner.populated()) {
        if intent.rows[row / 64] >> (row % 64) & 1 == 0 {
            continue;
        }
        let (current, cell) = match intent.column {
            Column::Meta => (
                u64::from(owner.meta[row].0),
                WriteCell {
                    // `act` checked the value fits the column.
                    meta: Some(MetaWord(intent.value as u32)),
                    ..WriteCell::default()
                },
            ),
            _ => (
                owner.temporal[row],
                WriteCell {
                    temporal: Some(intent.value),
                    ..WriteCell::default()
                },
            ),
        };
        if current == intent.value {
            applied.unchanged += 1;
            continue;
        }
        let restamped = owner.last_write_cycle[row] != intent.cycle;
        match owner.write_row(row, intent.cycle, &cell) {
            WriteOutcome::Accepted => {
                applied.changed += 1;
                applied.stamped += u32::from(restamped);
            }
            refused => {
                applied.refused = Some(refused);
                break;
            }
        }
    }
    applied
}

/// One method of one round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fire {
    pub method: MethodId,
    /// The rows it fired on, ascending.
    pub rows: Vec<u32>,
    /// The value of its first write (0 when it asked for none).
    pub value: u64,
    /// What the owner did with its writes, summed.
    pub applied: Applied,
    /// Rows its guard program ran over.
    pub guard_rows: u64,
    /// Lanes the socket lent for it: its guard's fields, then its `reads`.
    pub lanes_lent: u64,
    /// Effect steps R2IL executed.
    pub steps: usize,
    /// mask-risc programs R2IL finalized.
    pub programs: usize,
}

/// One round: the methods ON the changed columns, in ascending id order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Round {
    pub fired: Vec<Fire>,
}

/// A cascade's report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cascade {
    pub rounds: Vec<Round>,
    /// The last round changed nothing. `false` means the bound was reached
    /// with changes still pending.
    pub settled: bool,
}

/// Run the methods a change of `changed` columns triggers, apply their
/// writes, and repeat with the columns those writes really changed.
///
/// Stops when a round changes no cell (`settled`), or after `max_rounds`
/// rounds. A write that leaves a cell as it was triggers nothing. A write
/// that moves a row's `last_write_cycle` stamp counts as a change of that
/// column too.
///
/// # Errors
///
/// The first [`ActRefusal`]; rounds before it have been applied.
pub fn cascade<V: ClassView + ?Sized, const N: usize>(
    owner: &mut MailboxSoA<N>,
    store: &BindStore,
    view: &V,
    ctx: &ActContext<'_>,
    changed: &[Column],
    max_rounds: usize,
) -> Result<Cascade, ActRefusal> {
    let mut changed = changed.to_vec();
    let mut rounds = Vec::new();
    while !changed.is_empty() && rounds.len() < max_rounds {
        // Read: one publication for the whole round.
        let acted: Vec<(MethodId, u64, Acted)> = {
            let lanes = MailboxLanes::new(owner.published(), ctx.scope);
            let class = store
                .resolve_at(view, ctx.route.version, ctx.route.position, ctx.rev)
                .map_err(ActRefusal::Admission)?;
            let mut ids: Vec<MethodId> = Vec::new();
            for (field, basis) in class.fields.iter().enumerate() {
                let Ok(field) = u8::try_from(field) else {
                    break;
                };
                if changed.iter().any(|c| c.iri() == basis.predicate_iri) {
                    ids.extend(class.methods_on(field).map(|m| m.id));
                }
            }
            ids.sort_unstable();
            ids.dedup();
            let mut acted = Vec::with_capacity(ids.len());
            for id in ids {
                let reference = store
                    .method_ref(class.class, id)
                    .map_err(ActRefusal::Admission)?;
                let before = lanes.lends();
                let done = act(&lanes, store, view, ctx, &reference)?;
                acted.push((id, lanes.lends() - before, done));
            }
            acted
        };
        // Write: the publication is over.
        let mut round = Round::default();
        let mut next = Vec::new();
        for (method, lanes_lent, done) in acted {
            let mut applied = Applied::default();
            for intent in &done.intents {
                // `owner` has been borrowed exclusively since the round's
                // read; only this round's earlier writes lie in between.
                let one = write(owner, intent);
                applied.changed += one.changed;
                applied.unchanged += one.unchanged;
                applied.stamped += one.stamped;
                applied.refused = applied.refused.or(one.refused);
                if one.changed > 0 && !next.contains(&intent.column) {
                    next.push(intent.column);
                }
                // The owner's stamp is a column a method can be ON.
                if one.stamped > 0 && !next.contains(&Column::LastWriteCycle) {
                    next.push(Column::LastWriteCycle);
                }
            }
            round.fired.push(Fire {
                method,
                rows: done.rows,
                value: done.intents.first().map_or(0, Intent::value),
                applied,
                guard_rows: done.guard_rows,
                lanes_lent,
                steps: done.receipt.steps_executed,
                programs: done.receipt.programs_run,
            });
        }
        rounds.push(round);
        changed = next;
    }
    Ok(Cascade {
        rounds,
        settled: changed.is_empty(),
    })
}
