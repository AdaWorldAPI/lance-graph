//! An optional R2IL reading of a DT64 [`MethodBinding`] (D-DYNBAG §19.3 R1,
//! corrected by §20): when a consumer asks for it, an admitted binding's
//! ordered [`Effect`]s are bound to R2IL opcodes and run through
//! [`FoldDialect`].
//!
//! **This is not the path every execution takes.** A consumer executes its
//! own prepared native plan (Quack, mask-risc, `ndarray::simd`) without
//! consulting any codegraph. R2IL is the addressable, inspectable reading
//! behind execution; this module shows that the same binding can also be
//! run through that reading, and nothing more.
//!
//! The lifecycle is Tract's: [`prepare`] resolves once, cold, and binds the
//! opcodes; [`run`] executes the bound plan, hot, rechecking only the frame,
//! focus, scope, route and execution permit. `run` never reads the
//! `ClassView`, OGAR's vocabulary or the store's lens again. DT64 stays
//! non-executing; mask-risc runs every fold; no opcode, interpreter or
//! planner is added.
//!
//! A [`Receipt`] is bounded by the plan's step count. It names the plan, the
//! frame and the answer. It is not a row trace, not the plan's rewrite
//! history, and not the logical provenance of the answer.
//!
//! # How a word becomes a call
//!
//! An [`AlgebraRef`] names a field position under a lens classid. It is
//! **never** read as an [`FnIndex`] because the two happen to be `u8`.
//! [`prepare`] checks each effect before binding it:
//!
//! 1. the lens is the one this seam executes, and it is one of two readings:
//!    - the hardcoded polyfill lens [`POLYFILL_LENS`] (`0xFFFF :: 0xF000`),
//!      where the POSITION binds (see 3);
//!    - a lens whose concept half is OGAR's R2IL fold reading
//!      ([`ogar_r2il::CONCEPT_R2IL_FOLD`], provisional in OGAR: a lab lens,
//!      never a persisted identity), where the LABEL binds;
//! 2. the lens is bound in the store and its basis matches its `ClassView`
//!    at the stated schema revision ([`BindStore::resolve_algebra`]);
//! 3. the word becomes an opcode:
//!    - under [`POLYFILL_LENS`], word `p` is [`POLYFILL`]`[p]`; the view's
//!      label at `p` must be one of that position's aliases (an R2IL, SQL or
//!      Quack name), and a label belonging to another position is refused,
//!      never followed;
//!    - under the R2IL fold reading, the label resolves to exactly one opcode
//!      by OGAR's own vocabulary ([`R2ILVocabulary`]), so the position is
//!      free and only the name binds;
//! 4. the opcode is in the subset this seam translates faithfully
//!    ([`EXECUTABLE`]);
//! 5. the opcode's stack arity equals the effect's argument count (which
//!    DT64's registration already ties to the lens's declared arity) and it
//!    pushes one result: by the polyfill table under [`POLYFILL_LENS`], by
//!    OGAR's vocabulary under the R2IL fold reading.
//!
//! # What is checked when
//!
//! [`prepare`]: the frame, focus and scope guards; the route at the captured
//! version and now; the class basis; [`BindStore::admit`]; the binding's
//! WHEN and truth, which this reading cannot honour and so refuses; then
//! every effect as above.
//!
//! [`run`]: the guards and the route again, and an explicit execution permit
//! for the exact method reference. `admit` succeeding is not authorization.
//!
//! Arguments: `Arg::Field(p)` is the value an earlier effect wrote to `p`,
//! else the resident lane the caller bound `p` to ([`FieldLanes`]), whose
//! kind is read from the lane itself. `Arg::Const(c)` is pushed as is. A
//! constant compared against a lane must fit that lane's type; the dialect
//! would otherwise saturate it, so the seam refuses it first.
//!
//! The answer is the value each written field holds after the last effect.
//! A field that still holds a mask slot or an address is an intermediate
//! (a slot names dialect scratch and means nothing after the run); it is
//! listed in [`Executed::intermediates`], never surfaced as a value.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::ClassView;
use lance_graph_contract::ogar_codebook::classid_concept;
use lance_graph_contract::set_rules::Truth;
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, Effect, MethodRef, Refusal, RouteVersion, SchemaRev,
};
use lance_graph_mask_risc::{LaneKind, LaneRef, Planes};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, ALL, ANY, CONCEPT_R2IL_FOLD, KEY_RUNS, MAX, MIN, SUM};

use crate::polyfill::{POLYFILL, POLYFILL_LENS};
use crate::{
    Addr, FoldDialect, FoldError, Val, INT_ADD, INT_AND, INT_EQUAL, INT_MULT, INT_NOT, INT_OR,
    INT_SUB, INT_S_DIV, INT_S_LESS, INT_XOR, POP_COUNT,
};

/// The opcodes this seam translates faithfully: each takes its operands from
/// the stack only (no immediate bytes), pushes exactly one value, and has a
/// `FoldDialect` arm. `NUMBER`, `LOAD` and `RANGE` read immediates an
/// [`Effect`] cannot carry; `GROUP_SUM` and `KEEP` write a sink and push
/// nothing; everything else is unimplemented in the dialect. They are
/// refused by name ([`ExecRefusal::NotExecutable`]).
pub const EXECUTABLE: [FnIndex; 17] = [
    INT_ADD, INT_SUB, INT_MULT, INT_S_DIV, INT_EQUAL, INT_S_LESS, INT_AND, INT_OR, INT_XOR,
    INT_NOT, POP_COUNT, SUM, MIN, MAX, ANY, ALL, KEY_RUNS,
];

/// The compares whose constant operand the dialect narrows to the lane type.
pub(crate) const LANE_COMPARES: [FnIndex; 2] = [INT_EQUAL, INT_S_LESS];

/// Which resident lane each class field is read from: `lanes[p]` is the
/// index into `Planes::lanes` for field position `p`, or `None` when the
/// field has no resident lane. The consumer's field-to-lane binding; nothing
/// is inferred from positions.
#[derive(Debug, Clone, Copy)]
pub struct FieldLanes<'a> {
    pub lanes: &'a [Option<u16>],
}

/// The R2IL lens this seam executes, at its schema revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2ilLens {
    pub lens: u32,
    pub schema_rev: SchemaRev,
}

/// Explicit, limited permission to execute exact method references. This is
/// a lab permission, not a production authorization: who issues it is not
/// decided. `BindStore::admit` succeeding is never a substitute for it.
#[derive(Debug, Clone, Default)]
pub struct ExecPermit {
    pub methods: Vec<MethodRef>,
}

/// What a method was resolved under: the class's schema revision, where it
/// was routed, and the guards of the read.
#[derive(Debug, Clone)]
pub struct Resolution<'a> {
    pub method: &'a MethodRef,
    pub schema_rev: SchemaRev,
    pub route: (RouteVersion, u16),
    pub guards: &'a Guards,
}

/// One bound step of a prepared plan: the opcode the word resolved to, the
/// field it writes and its arguments, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub opcode: FnIndex,
    pub target: u8,
    pub args: Vec<Arg>,
}

/// A method resolved once, cold: every check that needs the `ClassView`,
/// OGAR's vocabulary or the store's lens is done, and the opcodes are bound.
/// [`run`] executes it without consulting any of them again.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub method: MethodRef,
    pub lens: R2ilLens,
    pub route: (RouteVersion, u16),
    pub guards: Guards,
    pub steps: Vec<Step>,
    /// FNV-1a over the method reference, the lens and every step: the plan's
    /// identity, carried into each [`Receipt`].
    pub plan_id: u64,
}

/// A run's answer: the final scalar of every field the plan writes,
/// ascending by position, and the written fields whose final value was a
/// mask or an address (an intermediate; a slot names dialect scratch and
/// means nothing after the run).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Executed {
    pub values: Vec<(u8, i64)>,
    pub intermediates: Vec<u8>,
}

/// What one run leaves behind: bounded by the plan's step count, never by
/// the row count. It names the plan, the frame it ran under and the answer;
/// it is not a trace of rows and not a proof of how the answer was derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub plan_id: u64,
    pub method: MethodRef,
    pub frame: FrameRevisions,
    pub steps_executed: usize,
    /// mask-risc programs the run finalized (one per fold step).
    pub programs_run: usize,
    pub output: Executed,
}

/// Why an invocation did not run. Every arm is named; none is an empty
/// answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecRefusal {
    /// The frame, focus or scope the method was resolved under moved.
    Guards(GuardRefusal),
    /// The route at the captured version, or now, does not name the
    /// method's class.
    RouteChanged {
        position: u16,
        captured: Option<u32>,
        now: Option<u32>,
        class: u32,
    },
    /// The store refused the class, the lens or the method reference.
    Binding(Refusal),
    /// The reference is admitted but not permitted to execute.
    NotPermitted,
    /// The binding has a WHEN guard; this seam does not evaluate guards.
    GuardNotExecutable,
    /// The binding is three-valued; the dialect has no UNKNOWN rows.
    ThreeValuedNotExecutable,
    /// An effect names a lens other than the R2IL lens.
    WrongLens { effect: usize, lens: u32 },
    /// The lens is neither [`POLYFILL_LENS`] nor a lens whose concept half
    /// is OGAR's R2IL fold reading.
    NotR2ilReading { lens: u32 },
    /// Under [`POLYFILL_LENS`], the view names the word with a label that is
    /// not an alias of its position.
    NotAnAlias {
        effect: usize,
        word: AlgebraRef,
        label: String,
    },
    /// The word's position is outside the lens's basis.
    WordOutOfBasis { effect: usize, word: AlgebraRef },
    /// The word's label is not an R2IL opcode name, or names more than one.
    UnknownWord { effect: usize, label: String },
    /// The opcode is not in [`EXECUTABLE`].
    NotExecutable { effect: usize, opcode: FnIndex },
    /// OGAR's stack arity differs from the effect's argument count. The
    /// lens's declared arity is reported; registration ties it to the
    /// argument count.
    Arity {
        effect: usize,
        ogar: Option<u8>,
        declared: Option<u8>,
        args: usize,
    },
    /// OGAR does not say the opcode pushes exactly one result.
    NoResult { effect: usize, opcode: FnIndex },
    /// A field argument has neither an earlier value nor a resident lane.
    FieldUnbound { effect: usize, field: u8 },
    /// The field's lane index is outside the planes, or its kind is not one
    /// the dialect addresses.
    LaneUnusable { effect: usize, field: u8, lane: u16 },
    /// A constant compared against a lane does not fit the lane's type.
    ConstOutOfRange { effect: usize, value: i64 },
    /// The dialect refused the call.
    Fold { effect: usize, error: FoldError },
    /// The call left other than one value on the stack.
    ResultArity { effect: usize, left: usize },
    /// A fold step consumes a value computed from an earlier fold's result;
    /// a native plan would freeze that value into the compiled ops.
    FoldResultFeedsFold { effect: usize },
    /// A step that is not a fold computes a scalar from an earlier fold's
    /// result; a native plan would have to re-evaluate it on every run.
    ScalarAfterFold { effect: usize },
    /// A scalar output read from the planes without a fold. A native plan
    /// would store it as a fixed value and miss a later change in the lanes.
    DataScalarNotPrepared { field: u8 },
    /// The plan writes more scalar fields than a native receipt carries.
    TooManyOutputs { outputs: usize },
    /// The planes' shape differs from the one the native plan was prepared
    /// over: `(n_rows, lanes, masks)`.
    PlanesChanged {
        recorded: (usize, usize, usize),
        now: (usize, usize, usize),
    },
}

/// Resolve `r`'s method into a [`Prepared`] plan, cold. See the module docs
/// for every check. Execution permission is not checked here: preparing a
/// plan runs nothing.
///
/// # Errors
///
/// An [`ExecRefusal`] naming the first check that failed.
pub fn prepare<V: ClassView + ?Sized>(
    store: &BindStore,
    view: &V,
    lens: R2ilLens,
    r: &Resolution<'_>,
    frames: &FrameStore,
    now: (Focus, Scope),
) -> Result<Prepared, ExecRefusal> {
    r.guards.check(frames, now).map_err(ExecRefusal::Guards)?;
    check_route(store, r.method.class, r.route)?;
    store
        .resolve(view, r.method.class, r.schema_rev)
        .map_err(ExecRefusal::Binding)?;
    let binding = store.admit(r.method).map_err(ExecRefusal::Binding)?;
    if binding.when.is_some() {
        return Err(ExecRefusal::GuardNotExecutable);
    }
    if binding.truth == Truth::ThreeValued {
        return Err(ExecRefusal::ThreeValuedNotExecutable);
    }
    if lens.lens != POLYFILL_LENS && classid_concept(lens.lens) != CONCEPT_R2IL_FOLD {
        return Err(ExecRefusal::NotR2ilReading { lens: lens.lens });
    }
    let steps = binding
        .effects
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Ok(Step {
                opcode: opcode(store, view, lens, i, e)?,
                target: e.target,
                args: e.args.clone(),
            })
        })
        .collect::<Result<Vec<_>, ExecRefusal>>()?;
    let plan_id = plan_id(r.method, lens, &steps);
    Ok(Prepared {
        method: r.method.clone(),
        lens,
        route: r.route,
        guards: r.guards.clone(),
        steps,
        plan_id,
    })
}

/// Run a prepared plan, hot: the guards and the route are checked again
/// (each a comparison or one route lookup), the permit is checked, and the
/// bound opcodes run through `dialect`. Nothing is re-resolved.
///
/// # Errors
///
/// An [`ExecRefusal`]; nothing has executed when a check before the first
/// step fails.
pub fn run(
    plan: &Prepared,
    store: &BindStore,
    frames: &FrameStore,
    now: (Focus, Scope),
    permit: &ExecPermit,
    fields: FieldLanes<'_>,
    dialect: &mut FoldDialect<'_>,
) -> Result<Receipt, ExecRefusal> {
    plan.guards
        .check(frames, now)
        .map_err(ExecRefusal::Guards)?;
    check_route(store, plan.method.class, plan.route)?;
    if !permit.methods.contains(&plan.method) {
        return Err(ExecRefusal::NotPermitted);
    }
    dialect.discard_pending();
    let result = run_steps(plan, frames, fields, dialect);
    dialect.discard_pending();
    result
}

/// [`run`]'s body after the checks; [`run`] clears the dialect's pending
/// state on both sides of it, on success and on error.
fn run_steps(
    plan: &Prepared,
    frames: &FrameStore,
    fields: FieldLanes<'_>,
    dialect: &mut FoldDialect<'_>,
) -> Result<Receipt, ExecRefusal> {
    let programs_before = dialect.programs_run();
    let mut values: BTreeMap<u8, Val> = BTreeMap::new();
    let mut stack: Vec<Val> = Vec::new();
    for (i, step) in plan.steps.iter().enumerate() {
        for &a in &step.args {
            let v = match a {
                Arg::Field(p) => match values.get(&p) {
                    Some(&v) => v,
                    None => lane(dialect.planes, fields, i, p)?,
                },
                Arg::Const(c) => Val::Scalar(c),
            };
            stack.push(v);
        }
        if LANE_COMPARES.contains(&step.opcode) {
            check_compare(&stack, i)?;
        }
        call_step(dialect, &plan.steps, i, &mut stack)
            .map_err(|error| ExecRefusal::Fold { effect: i, error })?;
        if stack.len() != 1 {
            return Err(ExecRefusal::ResultArity {
                effect: i,
                left: stack.len(),
            });
        }
        let out = stack.pop().expect("length checked");
        values.insert(step.target, out);
    }
    let mut output = Executed {
        values: Vec::new(),
        intermediates: Vec::new(),
    };
    for (field, v) in values {
        match v {
            Val::Scalar(x) => output.values.push((field, x)),
            _ => output.intermediates.push(field),
        }
    }
    Ok(Receipt {
        plan_id: plan.plan_id,
        method: plan.method.clone(),
        frame: frames.revisions(),
        steps_executed: plan.steps.len(),
        programs_run: dialect.programs_run() - programs_before,
        output,
    })
}

/// Call step `i`. A field a later step reads again keeps its value, so a
/// step that takes one as an operand goes through
/// [`FoldDialect::call_shared`], which never gates that operand's predicate
/// in place.
pub(crate) fn call_step(
    dialect: &mut FoldDialect<'_>,
    steps: &[Step],
    i: usize,
    stack: &mut Vec<Val>,
) -> Result<(), FoldError> {
    let step = &steps[i];
    let read_later = step.args.iter().any(|a| match a {
        Arg::Field(p) => steps[i + 1..]
            .iter()
            .any(|s| s.args.contains(&Arg::Field(*p))),
        Arg::Const(_) => false,
    });
    if read_later {
        dialect.call_shared(step.opcode, [0, 0, 0], stack)
    } else {
        dialect.call(step.opcode, [0, 0, 0], stack)
    }
}

/// The route at the captured version, and now, must both name `class`.
pub(crate) fn check_route(
    store: &BindStore,
    class: u32,
    route: (RouteVersion, u16),
) -> Result<(), ExecRefusal> {
    let (version, position) = route;
    let captured = store
        .routed_at(version, position)
        .map_err(ExecRefusal::Binding)?;
    let now = store
        .routed_at(store.current_version(), position)
        .map_err(ExecRefusal::Binding)?;
    if captured == Some(class) && now == Some(class) {
        Ok(())
    } else {
        Err(ExecRefusal::RouteChanged {
            position,
            captured,
            now,
            class,
        })
    }
}

/// FNV-1a over the method reference's identity fields, the lens and the
/// steps. Prepared plans refuse guarded bindings, so the guard is absent.
fn plan_id(method: &MethodRef, lens: R2ilLens, steps: &[Step]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&method.class.to_le_bytes());
    eat(&method.method.0.to_le_bytes());
    eat(&method.sig_rev.0.to_le_bytes());
    eat(&lens.lens.to_le_bytes());
    eat(&lens.schema_rev.0.to_le_bytes());
    for s in steps {
        eat(&[s.opcode.0, s.target]);
        eat(&(s.args.len() as u64).to_le_bytes());
        for a in &s.args {
            match a {
                Arg::Field(p) => eat(&[0, *p]),
                Arg::Const(c) => {
                    eat(&[1]);
                    eat(&c.to_le_bytes());
                }
            }
        }
    }
    h
}

/// The opcode an effect's word names, after every check in the module docs.
fn opcode<V: ClassView + ?Sized>(
    store: &BindStore,
    view: &V,
    lens: R2ilLens,
    effect: usize,
    e: &Effect,
) -> Result<FnIndex, ExecRefusal> {
    if e.algebra.lens != lens.lens {
        return Err(ExecRefusal::WrongLens {
            effect,
            lens: e.algebra.lens,
        });
    }
    let read = store
        .resolve_algebra(view, lens.lens, lens.schema_rev)
        .map_err(ExecRefusal::Binding)?;
    let label = read
        .fields
        .get(usize::from(e.algebra.word))
        .ok_or(ExecRefusal::WordOutOfBasis {
            effect,
            word: e.algebra,
        })?
        .label
        .as_str();
    if lens.lens == POLYFILL_LENS {
        return frozen_opcode(effect, e, label);
    }
    let mut named = (0..=u8::MAX)
        .map(FnIndex)
        .filter(|&f| R2ILVocabulary.name(f) == Some(label));
    let f = match (named.next(), named.next()) {
        (Some(f), None) => f,
        _ => {
            return Err(ExecRefusal::UnknownWord {
                effect,
                label: label.to_string(),
            })
        }
    };
    if !EXECUTABLE.contains(&f) {
        return Err(ExecRefusal::NotExecutable { effect, opcode: f });
    }
    let ogar = R2ILVocabulary.stack_arity(f);
    let declared = store.arity(e.algebra);
    // `BindStore::register` already refuses an effect whose argument count
    // differs from the lens's declared arity, so `declared == args` here;
    // the independent check is OGAR's arity against both.
    if ogar.map(usize::from) != Some(e.args.len()) {
        return Err(ExecRefusal::Arity {
            effect,
            ogar,
            declared,
            args: e.args.len(),
        });
    }
    if R2ILVocabulary.pushes_result(f) != Some(true) {
        return Err(ExecRefusal::NoResult { effect, opcode: f });
    }
    Ok(f)
}

/// The opcode at a polyfill position: the position binds, the view's label
/// must be one of its aliases, and the table's arity must match.
fn frozen_opcode(effect: usize, e: &Effect, label: &str) -> Result<FnIndex, ExecRefusal> {
    let word = POLYFILL
        .get(usize::from(e.algebra.word))
        .ok_or(ExecRefusal::WordOutOfBasis {
            effect,
            word: e.algebra,
        })?;
    if !word.aliases.contains(&label) {
        return Err(ExecRefusal::NotAnAlias {
            effect,
            word: e.algebra,
            label: label.to_string(),
        });
    }
    if usize::from(word.arity) != e.args.len() {
        return Err(ExecRefusal::Arity {
            effect,
            ogar: None,
            declared: Some(word.arity),
            args: e.args.len(),
        });
    }
    Ok(word.opcode)
}

/// The address of the resident lane field `p` is bound to.
pub(crate) fn lane(
    planes: &Planes<'_>,
    fields: FieldLanes<'_>,
    effect: usize,
    field: u8,
) -> Result<Val, ExecRefusal> {
    let idx = fields
        .lanes
        .get(usize::from(field))
        .copied()
        .flatten()
        .ok_or(ExecRefusal::FieldUnbound { effect, field })?;
    let unusable = ExecRefusal::LaneUnusable {
        effect,
        field,
        lane: idx,
    };
    let kind = match planes.lanes.get(usize::from(idx)) {
        Some(LaneRef::I32(_)) => LaneKind::I32,
        Some(LaneRef::U32(_)) => LaneKind::U32,
        Some(LaneRef::U64(_)) => LaneKind::U64,
        _ => return Err(unusable),
    };
    Ok(Val::Address(Addr::Lane { idx, kind }))
}

/// A lane compare's constant must fit the lane's type: the dialect narrows
/// it with a saturating conversion, which would change the predicate.
pub(crate) fn check_compare(stack: &[Val], effect: usize) -> Result<(), ExecRefusal> {
    let [.., a, b] = stack else {
        return Ok(());
    };
    let (kind, value) = match (*a, *b) {
        (Val::Address(Addr::Lane { kind, .. }), Val::Scalar(c))
        | (Val::Scalar(c), Val::Address(Addr::Lane { kind, .. })) => (kind, c),
        _ => return Ok(()),
    };
    let fits = match kind {
        LaneKind::I32 => i32::try_from(value).is_ok(),
        LaneKind::U32 => u32::try_from(value).is_ok(),
        LaneKind::U64 => u64::try_from(value).is_ok(),
        _ => false,
    };
    if fits {
        Ok(())
    } else {
        Err(ExecRefusal::ConstOutOfRange { effect, value })
    }
}
