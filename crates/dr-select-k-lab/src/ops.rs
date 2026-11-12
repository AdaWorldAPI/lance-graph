//! The three operations over a bound selection, per admissible view.
//!
//! Folds run through mask-RISC over the staged K-local lanes (lane 0 = `a`,
//! lane 1 = `b`); the sparse-positions arm is a plain loop.

use crate::bind::{Bound, Order, Residual, View};
use crate::counters::Counters;
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Scratch, Terminal, Value,
};

/// Lane index of `a` in every program of this file.
const LANE_A: u16 = 0;
/// Lane index of `b` in every program of this file.
const LANE_B: u16 = 1;

/// Run `program` over two I32 lanes and at most one input mask plane.
fn run(
    program: &Program,
    scratch: &mut Scratch<'_>,
    n_rows: usize,
    mask: Option<&[u64]>,
    a: &[i32],
    b: &[i32],
) -> Value {
    let lanes = [LaneRef::I32(a), LaneRef::I32(b)];
    let masks: [&[u64]; 1] = [mask.unwrap_or(&[])];
    let planes = Planes {
        n_rows,
        masks: if mask.is_some() {
            &masks[..]
        } else {
            &masks[..0]
        },
        lanes: &lanes,
    };
    execute_into(program, &planes, &Foreign::NONE, scratch, Out::None)
        .expect("mask-RISC rejected a fixed lab program (lab bug)")
}

/// `count(b > t)` program, gated by plane 0 when `gated`.
fn count_b_gt_program(t: i32, gated: bool) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::GtI32 { lane: LANE_B, t },
            under: if gated { Some(Operand::Plane(0)) } else { None },
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

/// `sum(a)` over plane 0.
fn sum_a_program() -> Program {
    Program::new(
        Vec::new(),
        Terminal::MaskedSumI32 {
            mask: Operand::Plane(0),
            lane: LANE_A,
        },
    )
}

/// Rewrite the threshold of a [`count_b_gt_program`] in place, no allocation.
fn set_threshold(program: &mut Program, t: i32) {
    if let MaskOp::Pred { pred, .. } = &mut program.ops[0] {
        *pred = Pred::GtI32 { lane: LANE_B, t };
    }
}

/// The operations the lab compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `count(residual AND b > t)`.
    CountBGt { t: i32 },
    /// `sum(a)` over the residual, as `i64`.
    SumA,
    /// The first `limit` guids of the residual in the declared key order.
    /// `order` names the contract (which tie rule), never `Order::Source`.
    TopByKey { limit: usize, order: Order },
}

/// Why a view may not answer an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inadmissible {
    /// The operation needs one declared key order and the binding witnesses
    /// another, or none. A mask or ascending positions keep the local order
    /// only, so no view can supply an order the binding does not have.
    OrderNotWitnessed { want: Order, have: Order },
}

/// Whether `view` over `bound` can lawfully answer `op`. `CountBGt` and
/// `SumA` are order-free and always admissible. `TopByKey` is admissible
/// only when the binding witnesses exactly the requested order (same tie
/// rule), for every view; nothing in this crate sorts at query time.
pub fn admissible(op: Op, bound: &Bound, view: View<'_>) -> Result<(), Inadmissible> {
    let _ = view;
    match op {
        Op::CountBGt { .. } | Op::SumA => Ok(()),
        Op::TopByKey { order, .. } => {
            if order != Order::Source && bound.order() == order {
                Ok(())
            } else {
                Err(Inadmissible::OrderNotWitnessed {
                    want: order,
                    have: bound.order(),
                })
            }
        }
    }
}

/// Build the residual `a >= t` over the K local rows in the form of `like`:
/// `Residual::All` is returned unchanged (it ignores `t`), `Positions`
/// yields ascending local positions, `Mask` yields `ceil(K/64)` words.
/// Bumps `conversions` by one and `bytes_copied` by the bytes written, for
/// the two stored forms.
pub fn residual_a_ge(bound: &Bound, t: i32, like: &Residual, counters: &Counters) -> Residual {
    let a = &bound.staged().a;
    match like {
        Residual::All => Residual::All,
        Residual::Positions(_) => {
            let pos: Vec<u32> = (0..a.len() as u32)
                .filter(|&i| a[i as usize] >= t)
                .collect();
            Counters::bump(&counters.conversions, 1);
            Counters::bump(&counters.bytes_copied, (pos.len() * 4) as u64);
            Residual::Positions(pos)
        }
        Residual::Mask(_) => {
            let mut words = vec![0u64; words_for(a.len())];
            for (w, chunk) in words.iter_mut().zip(a.chunks(64)) {
                let mut bits = 0u64;
                for (j, &v) in chunk.iter().enumerate() {
                    bits |= u64::from(v >= t) << j;
                }
                *w = bits;
            }
            Counters::bump(&counters.conversions, 1);
            Counters::bump(&counters.bytes_copied, (words.len() * 8) as u64);
            Residual::Mask(words)
        }
    }
}

/// Compiled mask-RISC programs and scratch for one binding: built once,
/// reused by every fold over it.
pub struct PreparedFolds {
    k: usize,
    /// `count(b > t)` under plane 0; `t` is rewritten per call.
    count_masked: Program,
    /// `count(b > t)` with no gate (view ALL); `t` is rewritten per call.
    count_all: Program,
    /// `sum(a)` over plane 0.
    sum_masked: Program,
    /// One scratch slot, enough for both count programs. The programs differ
    /// only in `t`, so the slot count never changes and it is reused as is.
    scratch: Scratch<'static>,
    /// mask-RISC has no implicit ALL for a masked sum; this is the borrowed
    /// constant plane: `ceil(K/64)` words, all ones, tail bits zero.
    ones: Vec<u64>,
}

impl PreparedFolds {
    /// Prepare the programs for `bound` (sized for its K rows).
    pub fn new(bound: &Bound) -> PreparedFolds {
        let k = bound.k();
        let count_masked = count_b_gt_program(0, true);
        let count_all = count_b_gt_program(0, false);
        let sum_masked = sum_a_program();
        let scratch = Scratch::for_program(&count_masked, k)
            .expect("scratch for a one-slot lab program (lab bug)");
        let mut ones = vec![u64::MAX; words_for(k)];
        if !k.is_multiple_of(64) {
            if let Some(last) = ones.last_mut() {
                *last = (1u64 << (k % 64)) - 1;
            }
        }
        PreparedFolds {
            k,
            count_masked,
            count_all,
            sum_masked,
            scratch,
            ones,
        }
    }

    /// `count(residual AND b > t)`. Bumps `operand_rows` by the rows handed
    /// to the fold: K for `All` and `Mask`, the position count for
    /// `Positions`.
    pub fn count_b_gt(
        &mut self,
        bound: &Bound,
        view: View<'_>,
        t: i32,
        counters: &Counters,
    ) -> u64 {
        let st = bound.staged();
        match view {
            View::All => {
                Counters::bump(&counters.operand_rows, self.k as u64);
                if self.k == 0 {
                    return 0;
                }
                set_threshold(&mut self.count_all, t);
                match run(
                    &self.count_all,
                    &mut self.scratch,
                    self.k,
                    None,
                    &st.a,
                    &st.b,
                ) {
                    Value::Count(n) => n as u64,
                    _ => unreachable!("Count terminal returns Value::Count"),
                }
            }
            View::Mask(m) => {
                Counters::bump(&counters.operand_rows, self.k as u64);
                if self.k == 0 {
                    return 0;
                }
                set_threshold(&mut self.count_masked, t);
                match run(
                    &self.count_masked,
                    &mut self.scratch,
                    self.k,
                    Some(m),
                    &st.a,
                    &st.b,
                ) {
                    Value::Count(n) => n as u64,
                    _ => unreachable!("Count terminal returns Value::Count"),
                }
            }
            View::Positions(p) => {
                Counters::bump(&counters.operand_rows, p.len() as u64);
                p.iter().filter(|&&i| st.b[i as usize] > t).count() as u64
            }
        }
    }

    /// `sum(a)` over the residual. Same counting as [`Self::count_b_gt`].
    pub fn sum_a(&mut self, bound: &Bound, view: View<'_>, counters: &Counters) -> i64 {
        let st = bound.staged();
        match view {
            View::All | View::Mask(_) => {
                Counters::bump(&counters.operand_rows, self.k as u64);
                if self.k == 0 {
                    return 0;
                }
                let m: &[u64] = match view {
                    View::Mask(m) => m,
                    _ => &self.ones,
                };
                match run(
                    &self.sum_masked,
                    &mut self.scratch,
                    self.k,
                    Some(m),
                    &st.a,
                    &st.b,
                ) {
                    Value::SumI64(s) => s,
                    _ => unreachable!("MaskedSumI32 returns Value::SumI64"),
                }
            }
            View::Positions(p) => {
                Counters::bump(&counters.operand_rows, p.len() as u64);
                p.iter().map(|&i| i64::from(st.a[i as usize])).sum()
            }
        }
    }

    /// The first `limit` guids of the residual in the declared `order`.
    /// Appends to `out` (cleared first). Returns `Err` exactly when
    /// [`admissible`] does.
    pub fn top_by_key(
        &mut self,
        bound: &Bound,
        view: View<'_>,
        limit: usize,
        order: Order,
        out: &mut Vec<u128>,
        counters: &Counters,
    ) -> Result<(), Inadmissible> {
        admissible(Op::TopByKey { limit, order }, bound, view)?;
        out.clear();
        let guid = &bound.staged().guid;
        // Local order is key order (witnessed above); never sorted.
        match view {
            View::All => out.extend(guid.iter().take(limit).copied()),
            View::Positions(p) => out.extend(p.iter().take(limit).map(|&i| guid[i as usize])),
            View::Mask(m) => {
                'words: for (wi, &word) in m.iter().enumerate() {
                    let mut w = word;
                    while w != 0 {
                        if out.len() >= limit {
                            break 'words;
                        }
                        let idx = wi * 64 + w.trailing_zeros() as usize;
                        if let Some(&g) = guid.get(idx) {
                            out.push(g);
                        }
                        w &= w - 1;
                    }
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        // Rows read = guids emitted (rows actually examined for the output).
        Counters::bump(&counters.operand_rows, out.len() as u64);
        Ok(())
    }
}

/// The resident baseline: the same answers folded over all N source rows
/// with an N-bit selection plane, no binding. `selected` is one bit per
/// source row (`key in range AND a >= t_a`), built by the caller once.
pub mod baseline {
    use crate::counters::Counters;
    use crate::source::Source;
    use lance_graph_mask_risc::{Scratch, Value};

    /// `count(selected AND b > t)` over N rows. The program and its scratch
    /// are built inside the call, so that cost is in the measured time.
    pub fn count_b_gt(source: &Source, selected: &[u64], t: i32, counters: &Counters) -> u64 {
        let n = source.n_rows();
        Counters::bump(&counters.operand_rows, n as u64);
        if n == 0 {
            return 0;
        }
        let (_, _, a, b) = source.lanes();
        let program = super::count_b_gt_program(t, true);
        let mut scratch = Scratch::for_program(&program, n)
            .expect("scratch for a one-slot lab program (lab bug)");
        match super::run(&program, &mut scratch, n, Some(selected), a, b) {
            Value::Count(c) => c as u64,
            _ => unreachable!("Count terminal returns Value::Count"),
        }
    }

    /// `sum(a)` over the selected source rows. Program and scratch are built
    /// inside the call, as for [`count_b_gt`].
    pub fn sum_a(source: &Source, selected: &[u64], counters: &Counters) -> i64 {
        let n = source.n_rows();
        Counters::bump(&counters.operand_rows, n as u64);
        if n == 0 {
            return 0;
        }
        let (_, _, a, b) = source.lanes();
        let program = super::sum_a_program();
        let mut scratch =
            Scratch::for_program(&program, n).expect("scratch for a lab program (lab bug)");
        match super::run(&program, &mut scratch, n, Some(selected), a, b) {
            Value::SumI64(s) => s,
            _ => unreachable!("MaskedSumI32 returns Value::SumI64"),
        }
    }
}
