//! **W0C — the row bridge: a merged operation carried as loco PROGRAM DATA,
//! executed by the ONE tile-fused evaluator, with no population crossing.**
//!
//! The capstone thesis under test: *"we have already built the instruction
//! set; stop re-implementing programs as Rust enum variants."* The candidate
//! operation is the factored join predicate AND a local compare, folded to a
//! count — `COUNT(*) WHERE p.country = v THROUGH l.partner_id AND l.amount > t`.
//! Today it exists three times as enums: `mask_risc::{Pred, Terminal}`,
//! `quack::{Filter, Agg}`, and (behind Panama) `LgjOpDesc`. Here it exists
//! ONCE more, as bytes:
//!
//! ```text
//! (EQ_VIA:fk,key,v) (GT_I32:lane,t) (AND) (COUNT)
//! ```
//!
//! and the question is whether those bytes can reach the substrate through
//! machinery that ALREADY exists — `ogar_loco::Interpreter`, a `Vocabulary`
//! above `DOMAIN_FLOOR` under its own classid, a `Dialect` — with the result
//! bit-identical to the native path and NOTHING proportional to rows written
//! outside the executor's tile-local scratch.
//!
//! # The seam, and why it is not a second interpreter
//!
//! loco refuses `FOR_EACH` / `FOR_RANGE` (`interpret.rs`), so it structurally
//! cannot sweep rows — and it must not. The dialect's stack therefore never
//! holds a population: its `Value` is a mask-risc [`Operand`], a SLOT NAME.
//! Interpreting the body BUILDS a [`Program`]; [`execute_into`] then runs it
//! tile-fused. loco composes, mask-risc executes, `ndarray::simd` computes.
//! A per-row R2IL replay (the `zipper_hop_parity` shape) would be scalar and
//! is the wrong seam for this question; it is deliberately not what this
//! probe does.
//!
//! # What is measured (the capstone's list)
//!
//! - logical op count: calls in the body (4);
//! - physical facade passes: `Program::ops.len()` after the dialect's
//!   survivor-gating peephole (2 — equal to quack's lowering, checked);
//! - population-sized bytes written by the loco side: 0 by construction
//!   (the dialect owns a `Vec<MaskOp>` and nothing else);
//! - tile-local scratch: `Scratch::for_program`, the same the native path uses;
//! - allocations: measured with a counting allocator at two row counts;
//!   the loco side's bytes must be IDENTICAL at 1,000 and 65,536 rows;
//! - final result: `Value::Count` equal to quack's program AND a scalar walk;
//! - code surface for operation #2: `GT_I32` is one `match` arm in
//!   [`MaskFold::call`] and one row in the vocabulary — no enum variant, no
//!   ABI symbol, no status code.
//!
//! # Disable table (each run red-then-green against the committed probe)
//!
//! | assertion | disable | observed |
//! |---|---|---|
//! | 2 passes, gated second op | peephole condition forced false | `logical_calls…` + `gating_only…` red (3 ops, no `under`) |
//! | malformed body refused | `pop().unwrap_or(Scratch(0))` | `malformed_bodies…` red |
//! | only the LAST ungated pred is folded | drop the `under @ None` guard | `gating_only…` red (re-gated an already-gated op) |
//! | join semantics, not shape | `EQ_VIA` emits `EqU32` on the fk lane | both count tests red |
//!
//! Two of the four patch anchors had been moved by `cargo fmt` between write
//! and run; the `assert s.count(old) == 1` guard caught both, so no green run
//! was recorded as evidence for a disable that never applied.
//!
//! # Honest limits
//!
//! Immediates are the call's value bytes (`u8`); a compare literal past 255
//! goes through loco's constant pool (`CONSTANT`), which this probe does not
//! exercise. The vocabulary lives in this test, not in a crate: promoting it
//! is the decision this probe informs, not one it takes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    execute_into, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, Terminal,
    Value,
};
use lance_graph_quack::{Agg, Cmp, Col, Filter, ForeignLane, Query};
use ogar_loco::program::Program as LocoProgram;
use ogar_loco::vocabulary::conformance::validate;
use ogar_loco::{
    Call, Dialect, FnIndex, FunctionBody, Interpreter, LaneShape, Vocabulary, DOMAIN_FLOOR,
};

// ── allocation counter ─────────────────────────────────────────────────────

struct Counting;
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn bytes_now() -> usize {
    BYTES.load(Ordering::Relaxed)
}

// ── the vocabulary: the mask-fold classid's opcodes above DOMAIN_FLOOR ─────

/// `EQ_VIA:fk,key,v` — the factored join leaf. Pushes a mask.
const EQ_VIA: FnIndex = FnIndex(DOMAIN_FLOOR);
/// `GT_I32:lane,t` — a local compare. Pushes a mask. **Operation #2** — the
/// one added to measure the code surface of a second operation.
const GT_I32: FnIndex = FnIndex(DOMAIN_FLOOR + 1);
/// `COUNT` — the demanded sink. Pops one mask, pushes nothing.
const COUNT: FnIndex = FnIndex(DOMAIN_FLOOR + 2);
/// Boolean AND is the SHARED CORE's `0x20`, not this vocabulary's: pops two
/// masks, pushes one. Nothing to declare — `Vocabulary::stack_arity` answers
/// it from the core below the floor.
const AND: FnIndex = FnIndex::AND;

/// The vocabulary is a TABLE, not behaviour: arity / pushes per opcode.
struct MaskFoldVocabulary;

impl Vocabulary for MaskFoldVocabulary {
    fn domain_stack_arity(&self, f: FnIndex) -> Option<u8> {
        match f {
            EQ_VIA | GT_I32 => Some(0),
            COUNT => Some(1),
            _ => None,
        }
    }
    fn domain_body_refs(&self, _f: FnIndex) -> u8 {
        0
    }
    fn domain_pushes_result(&self, f: FnIndex) -> Option<bool> {
        match f {
            EQ_VIA | GT_I32 => Some(true),
            COUNT => Some(false),
            _ => None,
        }
    }
    fn domain_name(&self, f: FnIndex) -> Option<&'static str> {
        match f {
            EQ_VIA => Some("EQ_VIA"),
            GT_I32 => Some("GT_I32"),
            COUNT => Some("COUNT"),
            _ => None,
        }
    }
}

// ── the dialect: interpretation BUILDS a mask-risc Program ─────────────────

#[derive(Debug, PartialEq, Eq)]
enum FoldError {
    Unknown(FnIndex),
    Underflow(FnIndex),
    TwoTerminals,
}

/// Owns the ops it emits and nothing sized by rows. Its stack values are
/// slot NAMES (`Operand::Scratch`), never words.
#[derive(Default)]
struct MaskFold {
    ops: Vec<MaskOp>,
    terminal: Option<Terminal>,
    next_slot: u16,
}

impl MaskFold {
    fn fresh(&mut self) -> u16 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Survivor gating, the loco-side twin of quack's in-place emission: if
    /// `b` was produced by the op emitted LAST and that op is an ungated
    /// `Pred`, gate it under `a` instead of spending a facade pass on AND.
    fn and(&mut self, a: Operand, b: Operand, stack: &mut Vec<Operand>) {
        if let (
            Operand::Scratch(bs),
            Some(MaskOp::Pred {
                under: under @ None,
                dst,
                ..
            }),
        ) = (b, self.ops.last_mut())
        {
            if *dst == bs {
                *under = Some(a);
                stack.push(b);
                return;
            }
        }
        let dst = self.fresh();
        self.ops.push(MaskOp::And { a, b, dst });
        stack.push(Operand::Scratch(dst));
    }
}

impl Dialect for MaskFold {
    type Value = Operand;
    type Error = FoldError;

    fn truthy(&self, _: &Operand) -> bool {
        // A mask's truth is its population — a question the EXECUTOR answers.
        // This probe's bodies never branch, so the answer is never read.
        false
    }
    fn repeat_count(&self, _: &Operand) -> u32 {
        0
    }

    fn call(&mut self, f: FnIndex, v: [u8; 3], stack: &mut Vec<Operand>) -> Result<(), FoldError> {
        match f {
            EQ_VIA => {
                let dst = self.fresh();
                self.ops.push(MaskOp::Pred {
                    pred: Pred::EqU32Via {
                        fk: u16::from(v[0]),
                        key: u16::from(v[1]),
                        v: u32::from(v[2]),
                    },
                    under: None,
                    dst,
                });
                stack.push(Operand::Scratch(dst));
            }
            // Operation #2: this arm and one vocabulary row are its whole surface.
            GT_I32 => {
                let dst = self.fresh();
                self.ops.push(MaskOp::Pred {
                    pred: Pred::GtI32 {
                        lane: u16::from(v[0]),
                        t: i32::from(v[1] as i8),
                    },
                    under: None,
                    dst,
                });
                stack.push(Operand::Scratch(dst));
            }
            AND => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                self.and(a, b, stack);
            }
            COUNT => {
                let mask = stack.pop().ok_or(FoldError::Underflow(f))?;
                if self.terminal.replace(Terminal::Count { mask }).is_some() {
                    return Err(FoldError::TwoTerminals);
                }
            }
            other => return Err(FoldError::Unknown(other)),
        }
        Ok(())
    }
}

/// The body under test, as data. Quads: three value bytes per call.
fn join_count_body(fk: u8, key: u8, v: u8, lane: u8, t: i8) -> LocoProgram {
    let calls = [
        Call::with_values(EQ_VIA, [fk, key, v]),
        Call::with_values(GT_I32, [lane, t as u8, 0]),
        Call::new(AND),
        Call::new(COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("4 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

/// loco interprets the body; the dialect's residue IS the mask-risc program.
fn lower_via_loco(body: &LocoProgram) -> Program {
    let vocab = validate(MaskFoldVocabulary).expect("vocabulary conforms");
    let mut it = Interpreter::new(&vocab, body, MaskFold::default());
    it.run().expect("straight-line body runs");
    assert!(it.stack().is_empty(), "COUNT consumed the last mask");
    // `Interpreter` exposes the dialect read-only; the residue is copied out
    // (op-count sized, never row sized — the allocation test pins that).
    let d = it.dialect();
    Program::new(d.ops.clone(), d.terminal.expect("terminal set"))
}

// ── fixture ────────────────────────────────────────────────────────────────

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

struct Tables {
    fk: Vec<u32>,
    amount: Vec<i32>,
    country: Vec<u32>,
}

impl Tables {
    fn seeded(n: usize, foreign_rows: usize, seed: u64) -> Self {
        let mut s = seed;
        // fk deliberately overshoots the foreign table on some rows: the
        // zero-fallback for an out-of-range fk is part of the contract.
        let fk = (0..n)
            .map(|_| (lcg(&mut s) % (foreign_rows as u64 + 3)) as u32)
            .collect();
        let amount = (0..n).map(|_| (lcg(&mut s) % 200) as i32 - 100).collect();
        let country = (0..foreign_rows)
            .map(|_| (lcg(&mut s) % 6) as u32)
            .collect();
        Self {
            fk,
            amount,
            country,
        }
    }
    fn lanes(&self) -> [LaneRef<'_>; 2] {
        [LaneRef::U32(&self.fk), LaneRef::I32(&self.amount)]
    }
    fn foreign_lanes(&self) -> [LaneRef<'_>; 1] {
        [LaneRef::U32(&self.country)]
    }
    /// The scalar truth, written without the substrate.
    fn oracle(&self, v: u32, t: i32) -> usize {
        self.fk
            .iter()
            .zip(&self.amount)
            .filter(|(&fk, &a)| self.country.get(fk as usize) == Some(&v) && a > t)
            .count()
    }
}

fn run(program: &Program, tables: &Tables) -> Value {
    let lanes = tables.lanes();
    let planes = Planes {
        n_rows: tables.fk.len(),
        masks: &[],
        lanes: &lanes,
    };
    let flanes = tables.foreign_lanes();
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mut scratch = Scratch::for_program(program, planes.n_rows).expect("scratch");
    execute_into(program, &planes, &foreign, &mut scratch, Out::None).expect("executes")
}

const ROWS: [usize; 6] = [1, 63, 64, 130, 1000, 65_536];
const V: u8 = 3;
const T: i8 = 17;

// ── tests ──────────────────────────────────────────────────────────────────

/// FAILS IF: the loco-carried body, quack's lowering, and the scalar walk
/// disagree on the count at any row boundary — the bit-identity the capstone
/// demands, on the factored join predicate specifically.
#[test]
fn a_join_count_carried_as_loco_bytes_matches_quack_and_the_oracle() {
    let body = join_count_body(0, 0, V, 1, T);
    let via_loco = lower_via_loco(&body);
    let via_quack = lance_graph_quack::lower(&Query {
        filter: Filter::and([
            Filter::eq_u32_via(Col(0), ForeignLane(0), u32::from(V)),
            Filter::cmp(Col(1), Cmp::GtI32(i32::from(T))),
        ]),
        agg: Agg::Count,
    })
    .expect("quack lowers");
    for (i, &n) in ROWS.iter().enumerate() {
        let t = Tables::seeded(n, 40, 11 + i as u64);
        let expect = t.oracle(u32::from(V), i32::from(T));
        assert_eq!(run(&via_loco, &t), Value::Count(expect), "loco path, n={n}");
        assert_eq!(
            run(&via_quack, &t),
            Value::Count(expect),
            "quack path, n={n}"
        );
    }
}

/// FAILS IF: the loco side spends MORE facade passes than the native
/// lowering, or the pinned counts drift silently.
///
/// MEASURED (2026-09-22): 4 logical calls → **2** physical ops via loco,
/// **3** via `quack::lower`. quack's `emit_gated` emits the second conjunct
/// gated under the first (`Pred{under: Scratch(0)}`) and THEN still spends an
/// `And{a:0,b:1}` — redundant, since a gated `b` is already a subset of `a`.
/// The eight-line dialect peephole does not. This is a finding about the
/// native path, pinned here so it cannot drift unnoticed; fixing quack is its
/// own wave with its own falsifier, not a drive-by in a probe.
#[test]
fn logical_calls_fold_to_the_same_physical_pass_count_as_quack() {
    let body = join_count_body(0, 0, V, 1, T);
    let via_loco = lower_via_loco(&body);
    let via_quack = lance_graph_quack::lower(&Query {
        filter: Filter::and([
            Filter::eq_u32_via(Col(0), ForeignLane(0), u32::from(V)),
            Filter::cmp(Col(1), Cmp::GtI32(i32::from(T))),
        ]),
        agg: Agg::Count,
    })
    .expect("quack lowers");
    let logical = body.entry().calls().filter(|c| !c.is_nop()).count();
    println!(
        "W0C: logical calls={logical} loco ops={} quack ops={} scratch slots loco={} quack={}",
        via_loco.ops.len(),
        via_quack.ops.len(),
        via_loco.scratch_slots,
        via_quack.scratch_slots
    );
    assert_eq!(logical, 4);
    assert_eq!(
        via_loco.ops.len(),
        2,
        "two facade passes: EqU32Via, GtI32 gated under it"
    );
    assert!(
        via_loco.ops.len() <= via_quack.ops.len(),
        "never more passes than the native lowering"
    );
    // Pinned two-sided so a quack improvement forces a deliberate re-pin here
    // rather than leaving this comment describing a gap that closed.
    assert_eq!(
        via_quack.ops.len(),
        3,
        "quack::lower's redundant trailing AND (see doc)"
    );
    // Anti-vacuity: the gating actually happened — the second op reads the first.
    assert!(matches!(
        via_loco.ops[1],
        MaskOp::Pred {
            under: Some(Operand::Scratch(0)),
            ..
        }
    ));
}

/// FAILS IF: anything the loco side allocates scales with rows. The bytes
/// spent building the program from bytes must be IDENTICAL at 1,000 and
/// 65,536 rows — the body has no row count, so the number cannot move.
/// The executor's own scratch is sized per program (tile-local) and measured
/// separately, as the native path is.
#[test]
fn the_loco_side_allocates_nothing_proportional_to_rows() {
    // The counter is process-global and the harness runs tests in parallel,
    // so other tests' allocations can only ADD to a delta. The minimum over a
    // few repetitions is therefore the uncontaminated figure.
    let mut per_n = Vec::new();
    for &n in &[1_000usize, 65_536] {
        let t = Tables::seeded(n, 40, 5);
        let body = join_count_body(0, 0, V, 1, T);
        let (mut loco_bytes, mut exec_bytes) = (usize::MAX, usize::MAX);
        let mut v = Value::Count(0);
        for _ in 0..8 {
            let before = bytes_now();
            let program = lower_via_loco(&body);
            loco_bytes = loco_bytes.min(bytes_now() - before);
            let before = bytes_now();
            v = run(&program, &t);
            exec_bytes = exec_bytes.min(bytes_now() - before);
        }
        println!("W0C: n={n} loco-side bytes={loco_bytes} executor bytes={exec_bytes} -> {v:?}");
        per_n.push((loco_bytes, exec_bytes));
    }
    assert_eq!(
        per_n[0].0, per_n[1].0,
        "loco-side allocation is independent of row count"
    );
    assert!(per_n[0].0 > 0, "the counter is live");
    // The executor's scratch grows with rows only up to the tile: two slots
    // of tile-local words, never a population (65,536 rows would be 16 KiB
    // of mask per slot if it were).
    let tile_cap = lance_graph_mask_risc::TILE_WORDS * 8 * 2;
    assert!(
        per_n[1].1 <= tile_cap + 64,
        "executor scratch stays tile-local: {} <= {tile_cap}",
        per_n[1].1
    );
    assert!(per_n[1].1 < 65_536 / 8, "not a population-sized mask");
}

/// FAILS IF: a body whose stack discipline is broken is silently accepted.
/// loco owns arity (the vocabulary table); the dialect owns underflow. Both
/// refusals must be reachable — a guard that cannot fire is not a guard.
#[test]
fn malformed_bodies_are_refused_not_approximated() {
    let vocab = validate(MaskFoldVocabulary).expect("vocabulary conforms");
    // AND with one operand: the shared core's arity is 2, so the dialect sees
    // an underflow on the second pop.
    let calls = [
        Call::with_values(EQ_VIA, [0, 0, V]),
        Call::new(AND),
        Call::new(COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let p = LocoProgram {
        functions: vec![body],
    };
    let mut it = Interpreter::new(&vocab, &p, MaskFold::default());
    assert!(matches!(
        it.run(),
        Err(ogar_loco::RunError::Dialect(FoldError::Underflow(AND)))
    ));
    // An opcode this vocabulary does not cover is refused by NAME.
    let calls = [Call::new(FnIndex(DOMAIN_FLOOR + 7))];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let p = LocoProgram {
        functions: vec![body],
    };
    let mut it = Interpreter::new(&vocab, &p, MaskFold::default());
    assert!(it.run().is_err());
}

/// FAILS IF: the survivor-gating peephole rewrites an op it must not — a
/// mask consumed by AND that is NOT the last emitted op keeps its pass, and
/// the result is still exact. `(EQ_VIA) (GT) (GT) (AND) (AND)`: the inner
/// AND gates the third op under the second; the outer AND finds the last op
/// already gated and must spend a real AND.
#[test]
fn gating_only_folds_the_last_ungated_pred() {
    let calls = [
        Call::with_values(EQ_VIA, [0, 0, V]),
        Call::with_values(GT_I32, [1, T as u8, 0]),
        Call::with_values(GT_I32, [1, (T - 40) as u8, 0]),
        Call::new(AND),
        Call::new(AND),
        Call::new(COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let p = LocoProgram {
        functions: vec![body],
    };
    let program = lower_via_loco(&p);
    assert_eq!(program.ops.len(), 4, "3 preds + 1 real AND");
    assert!(matches!(program.ops[3], MaskOp::And { .. }));
    let t = Tables::seeded(1000, 40, 3);
    // amount > T && amount > T-40 == amount > T
    assert_eq!(
        run(&program, &t),
        Value::Count(t.oracle(u32::from(V), i32::from(T)))
    );
}
