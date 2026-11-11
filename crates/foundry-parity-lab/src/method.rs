//! G6 (P2, part): `CanCook` as a registered DT64 method, executed from its
//! `MethodRef`.
//!
//! DT64 binds behaviour and executes nothing. Here the store holds one
//! `MethodBinding` per (item, quantity) whose WHEN guard states "this store can
//! make `q` of item `m`" as a [`SetExpr`] over the class's field positions:
//! mask fields as `Sym(field)`, value fields as residuals on lane `field`. The
//! executor admits the reference, binds exactly the method's `reads`, renames
//! field positions to the bound planes, and lowers the guard through relalg
//! ONCE per binding in `prepare` (`run_with_program`, keeping the returned
//! mask-RISC program). `run` only executes that program. The lab writes no
//! lowering of its own; the rename is the only lab step.
//!
//! The reader scope (the IAM stand-in) is NOT part of the method. It is
//! conjoined at execution, so the registered behaviour stays the same for
//! every reader (gap G2 is where it is supplied for real).
//!
//! The quantity is a constant of the guard, so it is part of the method's
//! identity ([`method_id`]): DT64 guards carry no parameters. Only the
//! quantities in [`QS`] are registered.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Cmp, Logic, Res, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    Guard, LaneRefusal, MethodBinding, MethodId, MethodRef, Refusal, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{
    execute_into, tile_words_for, ExecError, Foreign, LaneRef, Out, Planes, Program, Scratch,
};
use lance_graph_relalg::{rel_from_set, run_with_program, Measure, Query, RelError};

use crate::bind::{World, EQUIP0, KNOWN0, OFFERED0, READER, STAFF0, STOCK0, STORE_ID};
use crate::fixture::{self as fx, DENIED_STORE, ITEMS, STORES, STORE_ID_BASE};
use crate::source::{Network, Plane};
use lance_graph_dt64::LaneSource;

/// The quantities a `CanCook` method is registered for.
pub const QS: [i32; 2] = [1, 3];

/// The method id of `CanCook(item m, quantity q)`.
pub fn method_id(m: usize, q: i32) -> MethodId {
    MethodId(((m as u32) << 8) | q as u32)
}

/// "This store can make `q` of item `m`", over field positions.
pub fn can_cook_guard(m: usize, q: i32) -> SetExpr {
    let r = fx::recipe(m);
    let mut all = vec![
        SetExpr::Sym((OFFERED0 + m) as u32),
        SetExpr::Sym((EQUIP0 + r.equipment) as u32),
    ];
    for &(k, d) in &r.demand {
        all.push(SetExpr::Sym((KNOWN0 + k) as u32));
        all.push(SetExpr::i32((STOCK0 + k) as u16, Cmp::Ge, d * q));
    }
    // staff · per_staff ≥ q  ⇔  staff ≥ ⌈q / per_staff⌉ for positive per_staff.
    let staff = (q + r.per_staff - 1) / r.per_staff;
    all.push(SetExpr::i32((STAFF0 + r.skill) as u16, Cmp::Ge, staff));
    SetExpr::And(all)
}

/// The field positions a guard names.
fn guard_fields(e: &SetExpr, out: &mut Vec<u8>) {
    match e {
        SetExpr::Sym(f) => out.push(*f as u8),
        SetExpr::Res(Res::I32 { lane, .. }) | SetExpr::Res(Res::U32 { lane, .. }) => {
            out.push(*lane as u8)
        }
        SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().for_each(|c| guard_fields(c, out)),
        SetExpr::Not(c) => guard_fields(c, out),
        SetExpr::All | SetExpr::Empty => {}
    }
}

/// The binding of `CanCook(m, q)`: reads exactly the fields its guard names.
pub fn can_cook_method(m: usize, q: i32) -> MethodBinding {
    let guard = can_cook_guard(m, q);
    let mut fields = Vec::new();
    guard_fields(&guard, &mut fields);
    let reads = WideFieldMask::from_positions(&fields);
    MethodBinding {
        id: method_id(m, q),
        sig_rev: SigRev(1),
        name: format!("CanCook(item {m}, q {q})"),
        on: reads.clone(),
        when: Some(Guard::new(guard, Logic::TwoValued)),
        reads,
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// Every registered `CanCook` method.
pub fn methods() -> Vec<MethodBinding> {
    (0..ITEMS)
        .flat_map(|m| QS.into_iter().map(move |q| can_cook_method(m, q)))
        .collect()
}

/// Why a method could not be executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MethodRefusal {
    /// The store did not admit the reference.
    Admit(Refusal),
    Lane(LaneRefusal),
    /// relalg refused to lower the guard.
    Lower(RelError),
    /// The guard was answered from its description alone, so there is no
    /// program to reuse; never expected for a CanCook guard.
    Symbolic,
    /// The guard lowered into more than one fold, so its last program alone
    /// is not the answer and cannot be reused. A tripwire for a change in
    /// relalg's lowering: today every `SetExpr` lowers to one fold under
    /// `Keep` (pinned by `every_guard_shape_lowers_to_one_reusable_fold`).
    Folds {
        programs: usize,
    },
    /// The lowered program could not run.
    Exec(ExecError),
    /// The method has no WHEN guard to execute.
    NoGuard,
    /// The guard names a field the method does not read.
    NotBound {
        field: u8,
    },
    /// A bound plane had another kind than the guard uses it as.
    PlaneKind {
        field: u8,
    },
    /// `storeId` was not bound.
    NoIdentity,
}

impl From<LaneRefusal> for MethodRefusal {
    fn from(r: LaneRefusal) -> Self {
        MethodRefusal::Lane(r)
    }
}

/// The stores (bit = store row, by identity) that satisfy the guard, and how
/// many.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Executed {
    pub stores: u64,
    pub count: u64,
}

/// Where each bound field sits in the relalg planes.
#[derive(Clone, Copy)]
enum Slot {
    Mask(u32),
    I32(u16),
    U32(u16),
}

fn rename(e: &SetExpr, slots: &[Option<Slot>; 256]) -> Result<SetExpr, MethodRefusal> {
    Ok(match e {
        SetExpr::Sym(f) => match slots[*f as usize] {
            Some(Slot::Mask(i)) => SetExpr::Sym(i),
            Some(_) => return Err(MethodRefusal::PlaneKind { field: *f as u8 }),
            None => return Err(MethodRefusal::NotBound { field: *f as u8 }),
        },
        SetExpr::Res(Res::I32 { lane, op, t }) => match slots[*lane as usize] {
            Some(Slot::I32(i)) => SetExpr::i32(i, *op, *t),
            Some(_) => return Err(MethodRefusal::PlaneKind { field: *lane as u8 }),
            None => return Err(MethodRefusal::NotBound { field: *lane as u8 }),
        },
        SetExpr::Res(Res::U32 { lane, eq, v }) => match slots[*lane as usize] {
            Some(Slot::U32(i)) => SetExpr::Res(Res::U32 {
                lane: i,
                eq: *eq,
                v: *v,
            }),
            Some(_) => return Err(MethodRefusal::PlaneKind { field: *lane as u8 }),
            None => return Err(MethodRefusal::NotBound { field: *lane as u8 }),
        },
        SetExpr::And(cs) => SetExpr::And(
            cs.iter()
                .map(|c| rename(c, slots))
                .collect::<Result<_, _>>()?,
        ),
        SetExpr::Or(cs) => SetExpr::Or(
            cs.iter()
                .map(|c| rename(c, slots))
                .collect::<Result<_, _>>()?,
        ),
        SetExpr::Not(c) => SetExpr::negated(rename(c, slots)?),
        SetExpr::All => SetExpr::All,
        SetExpr::Empty => SetExpr::Empty,
    })
}

/// A method made ready once per binding: admitted, its reads bound and
/// borrowed, its guard renamed onto the bound planes and lowered by relalg
/// once into a mask-RISC program. [`PreparedMethod::run`] only executes that
/// program, once per question.
pub struct PreparedMethod<'s> {
    masks: Vec<&'s [u64]>,
    lanes: Vec<LaneRef<'s>>,
    ids: &'s [u32],
    program: Program,
    scratch: std::cell::RefCell<Scratch<'static>>,
}

impl<'s> PreparedMethod<'s> {
    /// Admit `r`, bind the method's reads plus `storeId`, and rename its
    /// guard (conjoined with the reader scope when `authorization`).
    pub fn prepare(
        world: &World,
        source: &'s Network,
        r: &MethodRef,
        authorization: bool,
    ) -> Result<PreparedMethod<'s>, MethodRefusal> {
        let binding = world.store.admit(r).map_err(MethodRefusal::Admit)?;
        let guard = binding.when.as_ref().ok_or(MethodRefusal::NoGuard)?;
        let mut fields: Vec<u8> = (0..=255u8).filter(|&f| binding.reads.has(f)).collect();
        fields.push(STORE_ID as u8);
        fields.sort_unstable();
        fields.dedup();
        let lanes_bound = world.bind(source, &fields, source.order().space(), READER)?;

        let mut masks: Vec<&'s [u64]> = Vec::new();
        let mut lanes: Vec<LaneRef<'s>> = Vec::new();
        let mut slots: [Option<Slot>; 256] = [None; 256];
        let mut ids: Option<&'s [u32]> = None;
        for lane in lanes_bound.borrow_all(&world.store, &world.view, source.frames(), source)? {
            let (field, plane) = lane?;
            let at = usize::from(field);
            slots[at] = Some(match plane {
                Plane::Mask(w) => {
                    masks.push(w);
                    Slot::Mask((masks.len() - 1) as u32)
                }
                Plane::I32(v) => {
                    lanes.push(LaneRef::I32(v));
                    Slot::I32((lanes.len() - 1) as u16)
                }
                Plane::U32(v) => {
                    if at == STORE_ID {
                        ids = Some(v);
                    }
                    lanes.push(LaneRef::U32(v));
                    Slot::U32((lanes.len() - 1) as u16)
                }
            });
        }
        let ids = ids.ok_or(MethodRefusal::NoIdentity)?;
        let mut query = rename(&guard.expr, &slots)?;
        if authorization {
            let Some(Slot::U32(id_lane)) = slots[STORE_ID] else {
                return Err(MethodRefusal::NoIdentity);
            };
            query = SetExpr::And(vec![query, SetExpr::ne_u32(id_lane, DENIED_STORE)]);
        }
        let planes = Planes {
            n_rows: STORES,
            masks: &masks,
            lanes: &lanes,
        };
        let mut keep = [0u64; 1];
        let (lowered, program) = run_with_program(
            &Query {
                rel: rel_from_set(&query),
                measure: Measure::Keep,
            },
            &planes,
            Some(&mut keep[..]),
        )
        .map_err(MethodRefusal::Lower)?;
        let program = program.ok_or(MethodRefusal::Symbolic)?;
        // relalg keeps only the LAST fold's program. Re-running it alone is the
        // whole answer only when the lowering was one fold.
        if lowered.programs != 1 {
            return Err(MethodRefusal::Folds {
                programs: lowered.programs,
            });
        }
        let scratch = std::cell::RefCell::new(Scratch::new(
            tile_words_for(STORES),
            program.scratch_slots as usize,
        ));
        Ok(PreparedMethod {
            masks,
            lanes,
            ids,
            program,
            scratch,
        })
    }

    /// One question: the stores satisfying the guard, and how many.
    pub fn run(&self) -> Result<Executed, MethodRefusal> {
        let planes = Planes {
            n_rows: STORES,
            masks: &self.masks,
            lanes: &self.lanes,
        };
        let mut keep = [0u64; 1];
        execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch.borrow_mut(),
            Out::Mask(&mut keep[..]),
        )
        .map_err(MethodRefusal::Exec)?;
        let count = u64::from(keep[0].count_ones());
        // Positions become stores only by reading `storeId` at the set bits.
        let mut stores = 0u64;
        let mut w = keep[0];
        while w != 0 {
            let row = w.trailing_zeros() as usize;
            stores |= 1 << (self.ids[row] - STORE_ID_BASE);
            w &= w - 1;
        }
        Ok(Executed { stores, count })
    }
}

/// [`PreparedMethod::prepare`] then one [`PreparedMethod::run`].
pub fn execute_can_cook(
    world: &World,
    source: &Network,
    r: &MethodRef,
    authorization: bool,
) -> Result<Executed, MethodRefusal> {
    PreparedMethod::prepare(world, source, r, authorization)?.run()
}
