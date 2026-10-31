//! The prepared native form of an R1 plan ([`crate::binding`]): the plan's
//! folds are run through [`FoldDialect`] once, cold, and the mask-risc
//! programs it finalizes are kept as [`CompiledProgram`]s. A hot
//! [`run_native`] executes those programs and nothing else.
//!
//! [`crate::binding::run`] re-interprets every step on every call. This
//! module is its successor for plans whose folds do not depend on each other:
//! [`prepare_native`] refuses, by name, every plan whose answer would need a
//! step re-evaluated per run.
//!
//! # What a native plan accepts
//!
//! Each field the plan writes has an origin:
//!
//! - `Const`: derived only from `Arg::Const`;
//! - `Data`: a lane, or a value derived from lanes, with no fold result in it;
//! - `Fold(k)`: the result of the k-th finalized program, or a value computed
//!   from one.
//!
//! A fold fixes the data it reads when its program is built, so a value that
//! depends on a fold result cannot be frozen. A mask computed from a fold
//! result is allowed to exist (it is only a field id in the plan's
//! intermediates), but:
//!
//! - a fold that consumes a `Fold`-origin value is refused
//!   ([`ExecRefusal::FoldResultFeedsFold`]);
//! - a step that is not a fold and computes a scalar from a `Fold`-origin
//!   value is refused ([`ExecRefusal::ScalarAfterFold`]);
//! - a plan that writes more scalar fields than [`MAX_NATIVE_OUTPUTS`] is
//!   refused ([`ExecRefusal::TooManyOutputs`]).
//!
//! # What is checked when
//!
//! [`prepare_native`]: the guards and the route, then every step through
//! [`FoldDialect`] with the argument resolution, compare-range check and
//! result-arity check of [`crate::binding::run`].
//!
//! [`run_native`]: the guards, the route and the execution permit, in the
//! order [`crate::binding::run`] checks them, then the planes' shape. The
//! permit is a lab permission, not a production authorization. `run_native`
//! allocates nothing.

use std::collections::BTreeMap;

use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{Arg, BindStore, MethodRef, RouteVersion};
use lance_graph_mask_risc::{
    execute_compiled, scratch_words_for, tile_words_for, CompiledProgram, Foreign, Out, Planes,
    Scratch, Terminal, Value,
};
use ogar_loco::{Dialect, FnIndex};
use ogar_r2il::{ALL, ANY, KEY_RUNS, MAX, MIN, SUM};

use crate::binding::{
    check_compare, check_route, lane, ExecPermit, ExecRefusal, FieldLanes, Prepared, LANE_COMPARES,
};
use crate::{FoldDialect, FoldError, Val, POP_COUNT};

/// The most scalar fields a [`NativeReceipt`] carries.
pub const MAX_NATIVE_OUTPUTS: usize = 8;

/// The opcodes that finalize a mask-risc program and push its scalar result.
const FOLD_OPCODES: [FnIndex; 7] = [POP_COUNT, SUM, MIN, MAX, ANY, ALL, KEY_RUNS];

/// Where a value comes from; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Const,
    Data,
    Fold(usize),
}

/// One finalized fold program, kept for the hot path.
#[derive(Debug, Clone)]
struct ProgramPlan {
    cp: CompiledProgram,
    /// The index of the step that produced the program.
    effect: usize,
    /// The fold opcode, for reporting an unexpected result shape.
    opcode: FnIndex,
    /// Scratch slots the program touches.
    slots: usize,
    /// The scalar an empty `MIN`/`MAX` yields, as the dialect reports it.
    empty: i64,
    /// The index into the receipt's values that holds this program's result,
    /// or `None` when a later step overwrote the field.
    out: Option<usize>,
}

/// The planes' shape a native plan was prepared over:
/// `(n_rows, lanes, masks)`.
type Shape = (usize, usize, usize);

fn shape_of(planes: &Planes<'_>) -> Shape {
    (planes.n_rows, planes.lanes.len(), planes.masks.len())
}

/// A prepared plan whose folds are compiled mask-risc programs. Built by
/// [`prepare_native`], executed by [`run_native`].
#[derive(Debug, Clone)]
pub struct NativePlan {
    prepared: Prepared,
    programs: Vec<ProgramPlan>,
    shape: Shape,
    /// The receipt's values before any program runs: the fixed scalars are
    /// filled in, the fold slots are zero until their program has run.
    template: [(u8, i64); MAX_NATIVE_OUTPUTS],
    outputs: usize,
    intermediates: Vec<u8>,
    tile_words: usize,
    scratch_words: usize,
}

impl NativePlan {
    /// The plan's identity, the one [`Prepared::plan_id`] carries.
    pub fn plan_id(&self) -> u64 {
        self.prepared.plan_id
    }

    /// The method reference the plan was prepared for.
    pub fn method(&self) -> &MethodRef {
        &self.prepared.method
    }

    /// The route the plan was prepared under.
    pub fn route(&self) -> (RouteVersion, u16) {
        self.prepared.route
    }

    /// How many mask-risc programs one run executes.
    pub fn program_count(&self) -> usize {
        self.programs.len()
    }

    /// For each program, the index of the step that produced it, in
    /// execution order.
    pub fn program_effects(&self) -> Vec<usize> {
        self.programs.iter().map(|p| p.effect).collect()
    }

    /// How many scalar fields a run reports.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The written fields whose final value is a mask or an address, which a
    /// receipt never carries. Static for the plan.
    pub fn intermediates(&self) -> &[u8] {
        &self.intermediates
    }

    /// Words of scratch a run needs.
    pub fn scratch_words(&self) -> usize {
        self.scratch_words
    }
}

/// The caller-owned scratch words [`run_native`] carves for each program.
/// Allocated once by [`NativeScratch::for_plan`].
#[derive(Debug, Clone)]
pub struct NativeScratch {
    words: Vec<u64>,
}

impl NativeScratch {
    /// Scratch sized for `plan`.
    pub fn for_plan(plan: &NativePlan) -> Self {
        Self {
            words: vec![0u64; plan.scratch_words],
        }
    }
}

/// What one native run leaves behind. It names the plan, the frame and the
/// scalar answer; it holds no heap data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeReceipt {
    pub plan_id: u64,
    pub frame: FrameRevisions,
    /// mask-risc programs the run executed.
    pub programs_run: usize,
    /// How many entries of `values` are filled.
    pub len: usize,
    /// `(field, value)`, ascending by field; entries past `len` are unused.
    pub values: [(u8, i64); MAX_NATIVE_OUTPUTS],
}

impl NativeReceipt {
    /// The filled `(field, value)` entries, ascending by field.
    pub fn values(&self) -> &[(u8, i64)] {
        &self.values[..self.len]
    }
}

/// Run `plan`'s steps once through [`FoldDialect`] over `planes` and keep the
/// programs its folds finalize, cold.
///
/// # Errors
///
/// An [`ExecRefusal`]: the guard, route, argument, compare and fold refusals
/// of [`crate::binding::run`], and the named refusals in the module docs.
pub fn prepare_native(
    plan: &Prepared,
    store: &BindStore,
    frames: &FrameStore,
    now: (Focus, Scope),
    fields: FieldLanes<'_>,
    planes: &Planes<'_>,
) -> Result<NativePlan, ExecRefusal> {
    plan.guards
        .check(frames, now)
        .map_err(ExecRefusal::Guards)?;
    check_route(store, plan.method.class, plan.route)?;
    let mut dialect = FoldDialect::new(planes, &Foreign::NONE);
    let mut values: BTreeMap<u8, (Val, Origin)> = BTreeMap::new();
    let mut stack: Vec<Val> = Vec::new();
    let mut programs: Vec<ProgramPlan> = Vec::new();
    for (i, step) in plan.steps.iter().enumerate() {
        let mut all_const = true;
        let mut tainted: Option<usize> = None;
        for &a in &step.args {
            let (v, origin) = match a {
                Arg::Field(p) => match values.get(&p) {
                    Some(&held) => held,
                    None => (lane(planes, fields, i, p)?, Origin::Data),
                },
                Arg::Const(c) => (Val::Scalar(c), Origin::Const),
            };
            match origin {
                Origin::Const => {}
                Origin::Data => all_const = false,
                Origin::Fold(k) => {
                    all_const = false;
                    tainted = tainted.or(Some(k));
                }
            }
            stack.push(v);
        }
        if tainted.is_some() && FOLD_OPCODES.contains(&step.opcode) {
            return Err(ExecRefusal::FoldResultFeedsFold { effect: i });
        }
        if LANE_COMPARES.contains(&step.opcode) {
            check_compare(&stack, i)?;
        }
        let before = dialect.programs_run();
        dialect
            .call(step.opcode, [0, 0, 0], &mut stack)
            .map_err(|error| ExecRefusal::Fold { effect: i, error })?;
        if stack.len() != 1 {
            return Err(ExecRefusal::ResultArity {
                effect: i,
                left: stack.len(),
            });
        }
        let out = stack.pop().expect("length checked");
        let origin = if dialect.programs_run() > before {
            if tainted.is_some() {
                return Err(ExecRefusal::FoldResultFeedsFold { effect: i });
            }
            let program = dialect
                .last_program()
                .ok_or(ExecRefusal::Fold {
                    effect: i,
                    error: FoldError::Unimplemented(step.opcode),
                })?
                .clone();
            let empty = match program.terminal {
                Terminal::MaskedMinI32 { .. } => i64::MAX,
                Terminal::MaskedMaxI32 { .. } => i64::MIN,
                _ => 0,
            };
            let slots = program.scratch_slots as usize;
            programs.push(ProgramPlan {
                cp: CompiledProgram::new(program),
                effect: i,
                opcode: step.opcode,
                slots,
                empty,
                out: None,
            });
            Origin::Fold(programs.len() - 1)
        } else if let Some(k) = tainted {
            if matches!(out, Val::Scalar(_)) {
                return Err(ExecRefusal::ScalarAfterFold { effect: i });
            }
            Origin::Fold(k)
        } else if all_const {
            Origin::Const
        } else {
            Origin::Data
        };
        values.insert(step.target, (out, origin));
    }
    let mut template = [(0u8, 0i64); MAX_NATIVE_OUTPUTS];
    let mut intermediates = Vec::new();
    let mut outputs = 0usize;
    for (field, (v, origin)) in values {
        let Val::Scalar(x) = v else {
            intermediates.push(field);
            continue;
        };
        if outputs < MAX_NATIVE_OUTPUTS {
            match origin {
                Origin::Fold(k) => {
                    template[outputs] = (field, 0);
                    programs[k].out = Some(outputs);
                }
                Origin::Const => template[outputs] = (field, x),
                Origin::Data => return Err(ExecRefusal::DataScalarNotPrepared { field }),
            }
        }
        outputs += 1;
    }
    if outputs > MAX_NATIVE_OUTPUTS {
        return Err(ExecRefusal::TooManyOutputs { outputs });
    }
    let tile_words = tile_words_for(planes.n_rows);
    let max_slots = programs.iter().map(|p| p.slots).max().unwrap_or(0);
    let scratch_words =
        scratch_words_for(tile_words, max_slots).ok_or_else(|| ExecRefusal::Fold {
            effect: programs.last().map_or(0, |p| p.effect),
            error: FoldError::Exec(lance_graph_mask_risc::ExecError::ScratchBufferTooSmall {
                need_words: usize::MAX,
                have_words: 0,
            }),
        })?;
    Ok(NativePlan {
        prepared: plan.clone(),
        programs,
        shape: shape_of(planes),
        template,
        outputs,
        intermediates,
        tile_words,
        scratch_words,
    })
}

/// Execute a native plan, hot: the guards, the route and the permit are
/// checked as [`crate::binding::run`] checks them, the planes' shape is
/// compared with the one the plan was prepared over, and the plan's programs
/// run in step order. Nothing is allocated.
///
/// # Errors
///
/// An [`ExecRefusal`]; nothing has executed when a check before the first
/// program fails.
pub fn run_native(
    np: &NativePlan,
    store: &BindStore,
    frames: &FrameStore,
    now: (Focus, Scope),
    permit: &ExecPermit,
    planes: &Planes<'_>,
    scratch: &mut NativeScratch,
) -> Result<NativeReceipt, ExecRefusal> {
    let plan = &np.prepared;
    plan.guards
        .check(frames, now)
        .map_err(ExecRefusal::Guards)?;
    check_route(store, plan.method.class, plan.route)?;
    if !permit.methods.contains(&plan.method) {
        return Err(ExecRefusal::NotPermitted);
    }
    let shape = shape_of(planes);
    if shape != np.shape {
        return Err(ExecRefusal::PlanesChanged {
            recorded: np.shape,
            now: shape,
        });
    }
    let mut values = np.template;
    for p in &np.programs {
        let fold = |error| ExecRefusal::Fold {
            effect: p.effect,
            error: FoldError::Exec(error),
        };
        let mut s = Scratch::over(&mut scratch.words, np.tile_words, p.slots).map_err(fold)?;
        let v = execute_compiled(
            &p.cp.compiled(),
            planes,
            &Foreign::NONE,
            &mut s,
            Out::None,
            0..planes.n_rows,
        )
        .map_err(fold)?;
        let x = match v {
            Value::Count(n) => i64::try_from(n).unwrap_or(i64::MAX),
            Value::SumI64(n) => n,
            Value::Bool(b) => i64::from(b),
            Value::OptI32(o) => o.map_or(p.empty, i64::from),
            _ => {
                return Err(ExecRefusal::Fold {
                    effect: p.effect,
                    error: FoldError::WrongOperandKind(p.opcode),
                })
            }
        };
        if let Some(o) = p.out {
            values[o].1 = x;
        }
    }
    Ok(NativeReceipt {
        plan_id: plan.plan_id,
        frame: frames.revisions(),
        programs_run: np.programs.len(),
        len: np.outputs,
        values,
    })
}
