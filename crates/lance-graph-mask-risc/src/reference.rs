//! The scalar oracle — the same [`Program`] evaluated ONE ROW AT A TIME in
//! plain Rust, with no SIMD facade anywhere in this file (law L4: a test
//! greps the source). Every executor is diffed against it on every backend.
//!
//! It also owns [`validate`], the ONE spelling of the rules a program must
//! satisfy before it runs; the executor calls the same function, so both
//! sides refuse the same bad program with the same [`ExecError`].

use crate::ir::{
    LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, MASKED_SUM_I32_MAX_ROWS,
};
use crate::value::{ExecError, LaneKind, Value};
use crate::words_for;

fn kind_of(lane: &LaneRef<'_>) -> LaneKind {
    match lane {
        LaneRef::I32(_) => LaneKind::I32,
        LaneRef::U32(_) => LaneKind::U32,
        LaneRef::U64(_) => LaneKind::U64,
    }
}

fn pred_lane_and_kind(pred: Pred) -> (u16, LaneKind) {
    match pred {
        Pred::GtI32 { lane, .. }
        | Pred::LtI32 { lane, .. }
        | Pred::GeI32 { lane, .. }
        | Pred::LeI32 { lane, .. }
        | Pred::EqI32 { lane, .. }
        | Pred::NeI32 { lane, .. } => (lane, LaneKind::I32),
        Pred::EqU32 { lane, .. } | Pred::NeU32 { lane, .. } | Pred::MatchU32 { lane, .. } => {
            (lane, LaneKind::U32)
        }
        Pred::MatchU64 { lane, .. } => (lane, LaneKind::U64),
    }
}

fn check_operand(p: &Program, planes: &Planes<'_>, o: Operand) -> Result<(), ExecError> {
    match o {
        Operand::Plane(i) if usize::from(i) >= planes.masks.len() => {
            Err(ExecError::PlaneOutOfRange(i))
        }
        Operand::Scratch(i) if u32::from(i) >= p.scratch_slots => Err(ExecError::ScratchTooSmall {
            need: u32::from(i) + 1,
            have: p.scratch_slots as usize,
        }),
        _ => Ok(()),
    }
}

fn check_lane(planes: &Planes<'_>, lane: u16, expected: LaneKind) -> Result<(), ExecError> {
    match planes.lanes.get(usize::from(lane)) {
        None => Err(ExecError::LaneOutOfRange(lane)),
        Some(l) if kind_of(l) != expected => Err(ExecError::LaneKind {
            lane,
            expected,
            found: kind_of(l),
        }),
        Some(_) => Ok(()),
    }
}

/// The validation rules, in the order both the executor and the oracle apply
/// them: (1) plane and lane lengths, then a dirty plane tail; (2) every op in
/// program order, operands in field order; (3) the terminal's mask, its lanes,
/// the sum bound, the blend destination. `out_len` is the caller's `out`
/// slice length, if any.
pub(crate) fn validate(
    p: &Program,
    planes: &Planes<'_>,
    out_len: Option<usize>,
) -> Result<(), ExecError> {
    let n = planes.n_rows;
    let words = words_for(n);
    for m in planes.masks {
        if m.len() != words {
            return Err(ExecError::LenMismatch {
                what: "plane",
                expected: words,
                found: m.len(),
            });
        }
    }
    for l in planes.lanes {
        if l.len() != n {
            return Err(ExecError::LenMismatch {
                what: "lane",
                expected: n,
                found: l.len(),
            });
        }
    }
    if !n.is_multiple_of(64) {
        for (i, m) in planes.masks.iter().enumerate() {
            if m[n / 64] >> (n % 64) != 0 {
                // `masks.len()` was bounded by the plane index type when the
                // caller built `Planes`; a wider slice cannot be addressed.
                return Err(ExecError::PlaneTail(u16::try_from(i).unwrap_or(u16::MAX)));
            }
        }
    }
    for op in &p.ops {
        match *op {
            MaskOp::Pred { pred, under, dst } => {
                if let Some(u) = under {
                    check_operand(p, planes, u)?;
                    if u == Operand::Scratch(dst) {
                        return Err(ExecError::GateAliasesDst { dst });
                    }
                }
                let (lane, kind) = pred_lane_and_kind(pred);
                check_lane(planes, lane, kind)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
            }
            MaskOp::And { a, b, dst }
            | MaskOp::Or { a, b, dst }
            | MaskOp::Xor { a, b, dst }
            | MaskOp::AndNot { a, b, dst } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, b)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
            }
            MaskOp::Not { a, dst } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
            }
            MaskOp::Ternlog { a, b, c, dst, .. } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, b)?;
                check_operand(p, planes, c)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
            }
        }
    }
    match p.terminal {
        Terminal::Count { mask }
        | Terminal::Any { mask }
        | Terminal::All { mask }
        | Terminal::Keep { mask } => check_operand(p, planes, mask),
        Terminal::MaskedSumI32 { mask, lane } => {
            check_operand(p, planes, mask)?;
            check_lane(planes, lane, LaneKind::I32)?;
            if n > MASKED_SUM_I32_MAX_ROWS {
                return Err(ExecError::SumRowBound { n_rows: n });
            }
            Ok(())
        }
        Terminal::MaskedMinI32 { mask, lane } | Terminal::MaskedMaxI32 { mask, lane } => {
            check_operand(p, planes, mask)?;
            check_lane(planes, lane, LaneKind::I32)
        }
        Terminal::BlendI32 { mask, then, els } => {
            check_operand(p, planes, mask)?;
            check_lane(planes, then, LaneKind::I32)?;
            check_lane(planes, els, LaneKind::I32)?;
            match out_len {
                None => Err(ExecError::BlendNeedsOut),
                Some(len) if len != n => Err(ExecError::LenMismatch {
                    what: "out",
                    expected: n,
                    found: len,
                }),
                Some(_) => Ok(()),
            }
        }
    }
}

fn plane_bit(planes: &Planes<'_>, i: u16, row: usize) -> bool {
    (planes.masks[usize::from(i)][row / 64] >> (row % 64)) & 1 == 1
}

fn i32_at(planes: &Planes<'_>, lane: u16, row: usize) -> i32 {
    match planes.lanes[usize::from(lane)] {
        LaneRef::I32(v) => v[row],
        _ => 0,
    }
}

fn eval_pred(planes: &Planes<'_>, pred: Pred, row: usize) -> bool {
    let u32_at = |lane: u16| match planes.lanes[usize::from(lane)] {
        LaneRef::U32(v) => v[row],
        _ => 0,
    };
    let u64_at = |lane: u16| match planes.lanes[usize::from(lane)] {
        LaneRef::U64(v) => v[row],
        _ => 0,
    };
    match pred {
        Pred::GtI32 { lane, t } => i32_at(planes, lane, row) > t,
        Pred::LtI32 { lane, t } => i32_at(planes, lane, row) < t,
        Pred::GeI32 { lane, t } => i32_at(planes, lane, row) >= t,
        Pred::LeI32 { lane, t } => i32_at(planes, lane, row) <= t,
        Pred::EqI32 { lane, v } => i32_at(planes, lane, row) == v,
        Pred::NeI32 { lane, v } => i32_at(planes, lane, row) != v,
        Pred::EqU32 { lane, v } => u32_at(lane) == v,
        Pred::NeU32 { lane, v } => u32_at(lane) != v,
        Pred::MatchU32 {
            lane,
            pattern,
            care,
        } => (u32_at(lane) ^ pattern) & care == 0,
        Pred::MatchU64 {
            lane,
            pattern,
            care,
        } => (u64_at(lane) ^ pattern) & care == 0,
    }
}

/// The oracle's state: one `Vec<bool>` per scratch slot, one entry per row.
struct Rows {
    slots: Vec<Vec<bool>>,
}

impl Rows {
    fn bit(&self, planes: &Planes<'_>, o: Operand, row: usize) -> bool {
        match o {
            Operand::Plane(i) => plane_bit(planes, i, row),
            Operand::Scratch(i) => self.slots[usize::from(i)][row],
        }
    }

    fn set(&mut self, dst: u16, row: usize, v: bool) {
        self.slots[usize::from(dst)][row] = v;
    }
}

fn run(p: &Program, planes: &Planes<'_>) -> Rows {
    let n = planes.n_rows;
    let mut rows = Rows {
        slots: (0..p.scratch_slots).map(|_| vec![false; n]).collect(),
    };
    for op in &p.ops {
        for row in 0..n {
            let v = match *op {
                MaskOp::Pred {
                    pred,
                    under,
                    dst: _,
                } => under.is_none_or(|u| rows.bit(planes, u, row)) && eval_pred(planes, pred, row),
                MaskOp::And { a, b, .. } => rows.bit(planes, a, row) & rows.bit(planes, b, row),
                MaskOp::Or { a, b, .. } => rows.bit(planes, a, row) | rows.bit(planes, b, row),
                MaskOp::Xor { a, b, .. } => rows.bit(planes, a, row) ^ rows.bit(planes, b, row),
                MaskOp::AndNot { a, b, .. } => rows.bit(planes, a, row) & !rows.bit(planes, b, row),
                MaskOp::Not { a, .. } => !rows.bit(planes, a, row),
                MaskOp::Ternlog { imm, a, b, c, .. } => {
                    let idx = (u8::from(rows.bit(planes, a, row)) << 2)
                        | (u8::from(rows.bit(planes, b, row)) << 1)
                        | u8::from(rows.bit(planes, c, row));
                    (imm >> idx) & 1 == 1
                }
            };
            let dst = match *op {
                MaskOp::Pred { dst, .. }
                | MaskOp::And { dst, .. }
                | MaskOp::Or { dst, .. }
                | MaskOp::Xor { dst, .. }
                | MaskOp::AndNot { dst, .. }
                | MaskOp::Not { dst, .. }
                | MaskOp::Ternlog { dst, .. } => dst,
            };
            rows.set(dst, row, v);
        }
    }
    rows
}

/// Evaluate `p` over `planes` row by row; `out` is the destination a
/// [`Terminal::BlendI32`] writes. Same validation, same [`Value`], same
/// [`ExecError`] as the executor.
pub fn reference_execute(
    p: &Program,
    planes: &Planes<'_>,
    out: Option<&mut [i32]>,
) -> Result<Value, ExecError> {
    validate(p, planes, out.as_deref().map(<[i32]>::len))?;
    let n = planes.n_rows;
    let rows = run(p, planes);
    let rows = &rows;
    let survivors = |mask: Operand| (0..n).filter(move |&r| rows.bit(planes, mask, r));
    Ok(match p.terminal {
        Terminal::Count { mask } => Value::Count(survivors(mask).count()),
        Terminal::Any { mask } => Value::Bool(survivors(mask).next().is_some()),
        Terminal::All { mask } => Value::Bool(survivors(mask).count() == n),
        Terminal::MaskedSumI32 { mask, lane } => Value::SumI64(
            survivors(mask)
                .map(|r| i64::from(i32_at(planes, lane, r)))
                .sum(),
        ),
        Terminal::MaskedMinI32 { mask, lane } => {
            Value::OptI32(survivors(mask).map(|r| i32_at(planes, lane, r)).min())
        }
        Terminal::MaskedMaxI32 { mask, lane } => {
            Value::OptI32(survivors(mask).map(|r| i32_at(planes, lane, r)).max())
        }
        Terminal::BlendI32 { mask, then, els } => {
            if let Some(o) = out {
                for (r, slot) in o.iter_mut().enumerate() {
                    let pick = if rows.bit(planes, mask, r) { then } else { els };
                    *slot = i32_at(planes, pick, r);
                }
            }
            Value::Blended
        }
        Terminal::Keep { mask } => Value::Mask(mask),
    })
}

/// Every scratch slot's FINAL contents after `p` runs, packed LSB-first with
/// tail bits zero — the shape an executor's `Scratch` holds, built here by
/// hand so a `Keep` result and every intermediate can be diffed word by word.
pub fn reference_scratch(p: &Program, planes: &Planes<'_>) -> Result<Vec<Vec<u64>>, ExecError> {
    validate(p, planes, None)?;
    let n = planes.n_rows;
    let rows = run(p, planes);
    Ok(rows
        .slots
        .iter()
        .map(|slot| {
            let mut words = vec![0u64; words_for(n)];
            for (r, &b) in slot.iter().enumerate() {
                if b {
                    words[r / 64] |= 1u64 << (r % 64);
                }
            }
            words
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const P0: Operand = Operand::Plane(0);
    const S0: Operand = Operand::Scratch(0);

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 11
    }

    struct Fx {
        n: usize,
        mask: Vec<u64>,
        i32s: Vec<i32>,
        u32s: Vec<u32>,
    }

    impl Fx {
        fn new(n: usize) -> Self {
            let mut s = 99u64 ^ n as u64;
            let mut mask: Vec<u64> = (0..words_for(n))
                .map(|_| lcg(&mut s) & lcg(&mut s))
                .collect();
            if !n.is_multiple_of(64) {
                mask[words_for(n) - 1] &= (1u64 << (n % 64)) - 1;
            }
            let i32s = (0..n).map(|_| (lcg(&mut s) % 2000) as i32 - 1000).collect();
            let u32s = (0..n).map(|_| (lcg(&mut s) % 64) as u32).collect();
            Self {
                n,
                mask,
                i32s,
                u32s,
            }
        }

        fn with<R>(&self, f: impl FnOnce(&Planes<'_>) -> R) -> R {
            let masks: [&[u64]; 1] = [&self.mask];
            let lanes = [LaneRef::I32(&self.i32s), LaneRef::U32(&self.u32s)];
            f(&Planes {
                n_rows: self.n,
                masks: &masks,
                lanes: &lanes,
            })
        }
    }

    /// FAILS IF: the oracle ever names the SIMD facade — law L4. The needle is
    /// concatenated so this test's own text cannot match itself.
    #[test]
    fn the_oracle_has_no_facade_token() {
        let needle = ["nd", "array"].concat();
        assert!(!include_str!("reference.rs").contains(&needle));
    }

    /// FAILS IF: a complement sets phantom bits — over 70 rows scratch word 1
    /// may carry only its 6 live bits.
    #[test]
    fn packed_scratch_obeys_the_tail_law() {
        let zero = vec![0u64; 2];
        let masks: [&[u64]; 1] = [&zero];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        let p = Program::new(
            vec![MaskOp::Not { a: P0, dst: 0 }],
            Terminal::Keep { mask: S0 },
        );
        let slots = reference_scratch(&p, &planes).unwrap_or_default();
        assert_eq!(slots, vec![vec![u64::MAX, 0b11_1111]]);
    }

    /// FAILS IF: a gated predicate is not `pred & gate` row for row — and the
    /// fixture is vacuous (both the gate and the predicate must be selective).
    #[test]
    fn a_gated_predicate_equals_the_ungated_one_under_the_gate() {
        let fx = Fx::new(130);
        fx.with(|planes| {
            let pred = Pred::GtI32 { lane: 0, t: 600 };
            let gated = Program::new(
                vec![MaskOp::Pred { pred, under: Some(P0), dst: 0 }],
                Terminal::Count { mask: S0 },
            );
            let manual = Program::new(
                vec![MaskOp::Pred { pred, under: None, dst: 0 }, MaskOp::And { a: S0, b: P0, dst: 1 }],
                Terminal::Count { mask: Operand::Scratch(1) },
            );
            let g = reference_execute(&gated, planes, None);
            assert_eq!(g, reference_execute(&manual, planes, None));
            let ungated = Program::new(vec![MaskOp::Pred { pred, under: None, dst: 0 }], Terminal::Count { mask: S0 });
            let u = reference_execute(&ungated, planes, None);
            assert!(matches!((g, u), (Ok(Value::Count(a)), Ok(Value::Count(b))) if a > 0 && a < b && b * 3 < 130));
        });
    }

    /// FAILS IF: any terminal disagrees with an independent spelling over the
    /// raw fixture data (`filter().count()`, `iter().sum()`, …).
    #[test]
    fn every_terminal_matches_an_independent_spelling() {
        let fx = Fx::new(130);
        fx.with(|planes| {
            let pred = MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: -600 },
                under: None,
                dst: 0,
            };
            let sel: Vec<usize> = (0..fx.n).filter(|&r| fx.i32s[r] < -600).collect();
            assert!(sel.len() * 3 < fx.n && !sel.is_empty());
            let run = |t: Terminal, out: Option<&mut [i32]>| {
                reference_execute(&Program::new(vec![pred], t), planes, out)
            };
            assert_eq!(
                run(Terminal::Count { mask: S0 }, None),
                Ok(Value::Count(sel.len()))
            );
            assert_eq!(run(Terminal::Any { mask: S0 }, None), Ok(Value::Bool(true)));
            assert_eq!(
                run(Terminal::All { mask: S0 }, None),
                Ok(Value::Bool(false))
            );
            let sum: i64 = sel.iter().map(|&r| i64::from(fx.i32s[r])).sum();
            assert_eq!(
                run(Terminal::MaskedSumI32 { mask: S0, lane: 0 }, None),
                Ok(Value::SumI64(sum))
            );
            let min = sel.iter().map(|&r| fx.i32s[r]).min();
            let max = sel.iter().map(|&r| fx.i32s[r]).max();
            assert_eq!(
                run(Terminal::MaskedMinI32 { mask: S0, lane: 0 }, None),
                Ok(Value::OptI32(min))
            );
            assert_eq!(
                run(Terminal::MaskedMaxI32 { mask: S0, lane: 0 }, None),
                Ok(Value::OptI32(max))
            );
            let mut out = vec![0i32; fx.n];
            assert_eq!(
                run(
                    Terminal::BlendI32 {
                        mask: S0,
                        then: 0,
                        els: 0
                    },
                    Some(&mut out)
                ),
                Ok(Value::Blended)
            );
            assert_eq!(out, fx.i32s);
            assert_eq!(run(Terminal::Keep { mask: P0 }, None), Ok(Value::Mask(P0)));
            let empty = Program::new(
                vec![MaskOp::AndNot {
                    a: P0,
                    b: P0,
                    dst: 0,
                }],
                Terminal::MaskedMinI32 { mask: S0, lane: 0 },
            );
            assert_eq!(
                reference_execute(&empty, planes, None),
                Ok(Value::OptI32(None))
            );
            let all = Program::new(
                vec![
                    MaskOp::Or {
                        a: P0,
                        b: P0,
                        dst: 0,
                    },
                    MaskOp::Not { a: S0, dst: 0 },
                    MaskOp::Or {
                        a: S0,
                        b: P0,
                        dst: 0,
                    },
                ],
                Terminal::All { mask: S0 },
            );
            assert_eq!(reference_execute(&all, planes, None), Ok(Value::Bool(true)));
        });
    }

    /// FAILS IF: `All` over zero rows is not vacuously true or `Any` not false.
    #[test]
    fn empty_planes_reduce_to_the_identities() {
        let planes = Planes {
            n_rows: 0,
            masks: &[&[][..]],
            lanes: &[],
        };
        let all = Program::new(vec![], Terminal::All { mask: P0 });
        let any = Program::new(vec![], Terminal::Any { mask: P0 });
        assert_eq!(
            reference_execute(&all, &planes, None),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            reference_execute(&any, &planes, None),
            Ok(Value::Bool(false))
        );
    }

    /// FAILS IF: any refusal is missing or carries the wrong payload — one
    /// program per `ExecError` arm the oracle can produce.
    #[test]
    fn every_error_arm_is_reachable_with_its_exact_payload() {
        let fx = Fx::new(70);
        fx.with(|planes| {
            let e = |ops: Vec<MaskOp>, t: Terminal| {
                reference_execute(&Program::new(ops, t), planes, None)
            };
            assert_eq!(
                e(
                    vec![MaskOp::Not {
                        a: Operand::Plane(7),
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::PlaneOutOfRange(7))
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::LaneOutOfRange(9))
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::EqU32 { lane: 0, v: 0 },
                        under: None,
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::LaneKind {
                    lane: 0,
                    expected: LaneKind::U32,
                    found: LaneKind::I32
                })
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 0, t: 0 },
                        under: Some(S0),
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::GateAliasesDst { dst: 0 })
            );
            assert_eq!(
                e(
                    vec![],
                    Terminal::BlendI32 {
                        mask: P0,
                        then: 0,
                        els: 0
                    }
                ),
                Err(ExecError::BlendNeedsOut)
            );
            let mut short = vec![0i32; 3];
            assert_eq!(
                reference_execute(
                    &Program::new(
                        vec![],
                        Terminal::BlendI32 {
                            mask: P0,
                            then: 0,
                            els: 0
                        }
                    ),
                    planes,
                    Some(&mut short)
                ),
                Err(ExecError::LenMismatch {
                    what: "out",
                    expected: 70,
                    found: 3
                })
            );
            let lying = Program {
                ops: vec![],
                terminal: Terminal::Any {
                    mask: Operand::Scratch(4),
                },
                scratch_slots: 1,
            };
            assert_eq!(
                reference_execute(&lying, planes, None),
                Err(ExecError::ScratchTooSmall { need: 5, have: 1 })
            );
        });
        let short_plane = vec![0u64; 1];
        let masks: [&[u64]; 1] = [&short_plane];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::Any { mask: P0 }),
                &planes,
                None
            ),
            Err(ExecError::LenMismatch {
                what: "plane",
                expected: 2,
                found: 1
            })
        );
        let dirty = vec![0u64, 1u64 << 10];
        let masks: [&[u64]; 1] = [&dirty];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::Any { mask: P0 }),
                &planes,
                None
            ),
            Err(ExecError::PlaneTail(0))
        );
        let short_lane = vec![0i32; 3];
        let lanes = [LaneRef::I32(&short_lane)];
        let planes = Planes {
            n_rows: 70,
            masks: &[],
            lanes: &lanes,
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::MaskedSumI32 { mask: S0, lane: 0 }),
                &planes,
                None
            ),
            Err(ExecError::LenMismatch {
                what: "lane",
                expected: 70,
                found: 3
            })
        );
    }

    /// FAILS IF: validation is not in program order — the FIRST bad op is
    /// the one reported, even when a later op is wrong in a "worse" way.
    #[test]
    fn the_first_bad_op_in_program_order_is_reported() {
        let fx = Fx::new(70);
        fx.with(|planes| {
            let p = Program::new(
                vec![
                    MaskOp::Not {
                        a: Operand::Plane(3),
                        dst: 0,
                    },
                    MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 1,
                    },
                ],
                Terminal::Any { mask: S0 },
            );
            assert_eq!(
                reference_execute(&p, planes, None),
                Err(ExecError::PlaneOutOfRange(3))
            );
            let q = Program::new(
                vec![
                    MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 1,
                    },
                    MaskOp::Not {
                        a: Operand::Plane(3),
                        dst: 0,
                    },
                ],
                Terminal::Any { mask: S0 },
            );
            assert_eq!(
                reference_execute(&q, planes, None),
                Err(ExecError::LaneOutOfRange(9))
            );
        });
    }
}
