//! The fuser — a Boolean tree over mask operands becomes a chain of
//! [`MaskOp::Ternlog`]s by truth-table evaluation. Any subtree over at most
//! three distinct leaves is ONE ternlog; a wider tree is reduced from its
//! larger child inward, each reduction replacing a subtree by the scratch
//! slot that holds it, until the remainder fits in one table.
//!
//! Semantics only: which instruction realizes an immediate is `ndarray`'s
//! business (law 3 of the crate doc).

use crate::ir::{MaskOp, Operand, Program, Terminal};

/// A Boolean expression over mask operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolExpr {
    /// A resident plane or a scratch slot the caller already filled.
    Leaf(Operand),
    /// `l & r`.
    And(Box<BoolExpr>, Box<BoolExpr>),
    /// `l | r`.
    Or(Box<BoolExpr>, Box<BoolExpr>),
    /// `l ^ r`.
    Xor(Box<BoolExpr>, Box<BoolExpr>),
    /// `!e`.
    Not(Box<BoolExpr>),
}

/// The lowered fragment: the ops to append, the operand holding the result,
/// and the first scratch slot still free after them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fused {
    /// Ternlogs in evaluation order.
    pub ops: Vec<MaskOp>,
    /// Where the expression's value ends up (the leaf itself for a bare leaf).
    pub result: Operand,
    /// `first_free_slot` plus the slots this fragment consumed, or `None`
    /// when the fragment used the LAST addressable slot. Saturating to
    /// `u16::MAX` here would name a slot this fragment already wrote, and a
    /// caller chaining `fuse(e2, f.next_slot)` would silently clobber it —
    /// the same one-short bug `ir.rs` records for `Program::new`'s count.
    pub next_slot: Option<u16>,
}

/// Why an expression could not be lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseError {
    /// The chain needed a scratch slot past `u16::MAX`.
    SlotOverflow,
}

/// The VPTERNLOG immediate of a three-input function: bit `(a<<2)|(b<<1)|c`
/// is `f(a, b, c)`.
pub fn ternlog_imm(f: impl Fn(bool, bool, bool) -> bool) -> u8 {
    let mut imm = 0u8;
    for idx in 0..8u8 {
        if f(idx & 4 != 0, idx & 2 != 0, idx & 1 != 0) {
            imm |= 1 << idx;
        }
    }
    imm
}

fn distinct_leaves(e: &BoolExpr, out: &mut Vec<Operand>) {
    match e {
        BoolExpr::Leaf(o) => {
            if !out.contains(o) {
                out.push(*o);
            }
        }
        BoolExpr::And(l, r) | BoolExpr::Or(l, r) | BoolExpr::Xor(l, r) => {
            distinct_leaves(l, out);
            distinct_leaves(r, out);
        }
        BoolExpr::Not(i) => distinct_leaves(i, out),
    }
}

fn leaf_count(e: &BoolExpr) -> usize {
    let mut v = Vec::new();
    distinct_leaves(e, &mut v);
    v.len()
}

/// Evaluate `e` with each distinct leaf bound to its position in `leaves`.
fn eval(e: &BoolExpr, leaves: &[Operand], vals: [bool; 3]) -> bool {
    match e {
        BoolExpr::Leaf(o) => leaves
            .iter()
            .position(|l| l == o)
            .is_some_and(|i| vals[i.min(2)]),
        BoolExpr::And(l, r) => eval(l, leaves, vals) & eval(r, leaves, vals),
        BoolExpr::Or(l, r) => eval(l, leaves, vals) | eval(r, leaves, vals),
        BoolExpr::Xor(l, r) => eval(l, leaves, vals) ^ eval(r, leaves, vals),
        BoolExpr::Not(i) => !eval(i, leaves, vals),
    }
}

struct Lowering {
    ops: Vec<MaskOp>,
    next: u32,
}

impl Lowering {
    fn alloc(&mut self) -> Result<u16, FuseError> {
        let slot = u16::try_from(self.next).map_err(|_| FuseError::SlotOverflow)?;
        self.next += 1;
        Ok(slot)
    }

    fn lower(&mut self, e: &BoolExpr) -> Result<Operand, FuseError> {
        if let BoolExpr::Leaf(o) = e {
            return Ok(*o);
        }
        let mut leaves = Vec::new();
        distinct_leaves(e, &mut leaves);
        if leaves.len() <= 3 {
            let imm = ternlog_imm(|a, b, c| eval(e, &leaves, [a, b, c]));
            // Missing inputs repeat the LAST leaf: the table was evaluated
            // over `leaves` only, so the padded position is a don't-care.
            let last = leaves[leaves.len() - 1];
            let a = leaves[0];
            let b = *leaves.get(1).unwrap_or(&last);
            let c = *leaves.get(2).unwrap_or(&last);
            let dst = self.alloc()?;
            self.ops.push(MaskOp::Ternlog { imm, a, b, c, dst });
            return Ok(Operand::Scratch(dst));
        }
        // Too wide: lower the larger child to a slot, substitute, and retry —
        // every step removes at least one distinct leaf, so this terminates.
        let reduced = match e {
            BoolExpr::Not(inner) => BoolExpr::Not(Box::new(BoolExpr::Leaf(self.lower(inner)?))),
            BoolExpr::And(l, r) | BoolExpr::Or(l, r) | BoolExpr::Xor(l, r) => {
                let left_is_bigger = leaf_count(l) >= leaf_count(r);
                let (big, small) = if left_is_bigger { (l, r) } else { (r, l) };
                let x = Box::new(BoolExpr::Leaf(self.lower(big)?));
                let (nl, nr) = if left_is_bigger {
                    (x, small.clone())
                } else {
                    (small.clone(), x)
                };
                match e {
                    BoolExpr::And(..) => BoolExpr::And(nl, nr),
                    BoolExpr::Or(..) => BoolExpr::Or(nl, nr),
                    _ => BoolExpr::Xor(nl, nr),
                }
            }
            BoolExpr::Leaf(o) => BoolExpr::Leaf(*o),
        };
        self.lower(&reduced)
    }
}

/// Lower `expr` into ternlogs whose destinations start at `first_free_slot`.
pub fn fuse(expr: &BoolExpr, first_free_slot: u16) -> Result<Fused, FuseError> {
    let mut l = Lowering {
        ops: Vec::new(),
        next: u32::from(first_free_slot),
    };
    let result = l.lower(expr)?;
    let next_slot = u16::try_from(l.next).ok();
    Ok(Fused {
        ops: l.ops,
        result,
        next_slot,
    })
}

/// [`fuse`] wrapped into a whole [`Program`]; `terminal_of` receives the
/// operand holding the expression's value.
pub fn fuse_program(
    expr: &BoolExpr,
    first_free_slot: u16,
    terminal_of: impl FnOnce(Operand) -> Terminal,
) -> Result<Program, FuseError> {
    let f = fuse(expr, first_free_slot)?;
    Ok(Program::new(f.ops, terminal_of(f.result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{LaneRef, Planes};
    use crate::reference::reference_execute;
    use crate::value::Value;

    fn leaf(i: u16) -> Box<BoolExpr> {
        Box::new(BoolExpr::Leaf(Operand::Plane(i)))
    }

    /// FAILS IF: a three-leaf `WHERE` costs more than one mask pass — F-B1.
    #[test]
    fn a_three_leaf_where_is_one_ternlog() {
        let e = BoolExpr::Or(Box::new(BoolExpr::And(leaf(0), leaf(1))), leaf(2));
        let p = fuse_program(&e, 0, |m| Terminal::Count { mask: m }).unwrap_or_else(|_| {
            Program::new(
                vec![],
                Terminal::Count {
                    mask: Operand::Plane(0),
                },
            )
        });
        assert_eq!(p.op_histogram().mask_passes(), 1);
        assert_eq!(p.op_histogram().ternlog, 1);
        assert!(
            matches!(p.ops[0], MaskOp::Ternlog { imm: 0xEA, .. }),
            "{:?}",
            p.ops[0]
        );
    }

    /// FAILS IF: `ternlog_imm` indexes the table wrongly — F-B4. The expected
    /// value is derived by a separate bit-serial loop AND equals the facade's
    /// `AND2_ANDNOT` convention (`0x40`).
    #[test]
    fn immediates_match_a_bit_serial_derivation() {
        let mut expected = 0u8;
        for idx in 0..8u8 {
            let (a, b, c) = ((idx >> 2) & 1 == 1, (idx >> 1) & 1 == 1, idx & 1 == 1);
            if a && b && !c {
                expected |= 1 << idx;
            }
        }
        assert_eq!(ternlog_imm(|a, b, c| a & b & !c), expected);
        assert_eq!(expected, 0x40);
        assert_eq!(ternlog_imm(|a, b, c| a & b & c), 0x80);
        assert_eq!(ternlog_imm(|a, b, _| a & b), 0xC0);
        assert_eq!(ternlog_imm(|a, b, c| a | b | c), 0xFE);
    }

    /// FAILS IF: a five-leaf tree is not reduced larger-child-first into two
    /// ternlogs whose second reads the first's slot.
    #[test]
    fn five_leaves_fuse_to_two_chained_ternlogs() {
        let e = BoolExpr::Or(
            Box::new(BoolExpr::And(
                Box::new(BoolExpr::And(leaf(0), leaf(1))),
                leaf(2),
            )),
            Box::new(BoolExpr::And(leaf(3), leaf(4))),
        );
        let f = fuse(&e, 5).unwrap_or(Fused {
            ops: vec![],
            result: Operand::Plane(0),
            next_slot: None,
        });
        assert_eq!(f.ops.len(), 2, "{:?}", f.ops);
        assert!(matches!(
            f.ops[0],
            MaskOp::Ternlog {
                imm: 0x80,
                dst: 5,
                ..
            }
        ));
        assert!(matches!(
            f.ops[1],
            MaskOp::Ternlog {
                a: Operand::Scratch(5),
                dst: 6,
                ..
            }
        ));
        assert_eq!(f.result, Operand::Scratch(6));
        assert_eq!(f.next_slot, Some(7));
    }

    /// FAILS IF: a bare leaf mints an op or a slot.
    #[test]
    fn a_bare_leaf_is_free() {
        let f = fuse(&BoolExpr::Leaf(Operand::Scratch(3)), 9).unwrap_or(Fused {
            ops: vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: 0,
            }],
            result: Operand::Plane(0),
            next_slot: None,
        });
        assert!(f.ops.is_empty());
        assert_eq!(f.result, Operand::Scratch(3));
        assert_eq!(f.next_slot, Some(9));
    }

    /// FAILS IF: the padded input leaks into a two-leaf table — the bit for
    /// `c = 0` and `c = 1` must agree at every `(a, b)`.
    #[test]
    fn padding_is_a_dont_care() {
        let e = BoolExpr::Xor(leaf(0), leaf(1));
        let f = fuse(&e, 0).unwrap_or(Fused {
            ops: vec![],
            result: Operand::Plane(0),
            next_slot: None,
        });
        let MaskOp::Ternlog { imm, b, c, .. } = f.ops[0] else {
            panic!("not a ternlog")
        };
        assert_eq!(b, c, "the third input repeats the last leaf");
        for ab in 0..4u8 {
            let idx0 = ab << 1;
            assert_eq!((imm >> idx0) & 1, (imm >> (idx0 | 1)) & 1);
        }
        assert_eq!(imm, 0x3C);
    }

    /// FAILS IF: slot exhaustion is not reported.
    #[test]
    fn slot_overflow_is_an_error() {
        let e = BoolExpr::Or(
            Box::new(BoolExpr::And(
                Box::new(BoolExpr::And(leaf(0), leaf(1))),
                leaf(2),
            )),
            Box::new(BoolExpr::And(leaf(3), leaf(4))),
        );
        assert_eq!(
            fuse(&e, u16::MAX).map(|f| f.ops.len()),
            Err(FuseError::SlotOverflow)
        );
        // The last-slot arm SUCCEEDS (both ternlogs fit) but has no next free
        // slot. Saturating would report 65,535 — a slot it just wrote.
        let last = fuse(&e, u16::MAX - 1);
        assert!(last.is_ok());
        assert_eq!(last.map(|f| f.next_slot), Ok(None));
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 11
    }

    fn random_expr(seed: &mut u64, depth: u32) -> BoolExpr {
        let pick = lcg(seed) % if depth == 0 { 1 } else { 5 };
        match pick {
            0 => BoolExpr::Leaf(Operand::Plane((lcg(seed) % 5) as u16)),
            1 => BoolExpr::And(
                Box::new(random_expr(seed, depth - 1)),
                Box::new(random_expr(seed, depth - 1)),
            ),
            2 => BoolExpr::Or(
                Box::new(random_expr(seed, depth - 1)),
                Box::new(random_expr(seed, depth - 1)),
            ),
            3 => BoolExpr::Xor(
                Box::new(random_expr(seed, depth - 1)),
                Box::new(random_expr(seed, depth - 1)),
            ),
            _ => BoolExpr::Not(Box::new(random_expr(seed, depth - 1))),
        }
    }

    fn direct(e: &BoolExpr, planes: &Planes<'_>, row: usize) -> bool {
        match e {
            BoolExpr::Leaf(Operand::Plane(i)) => {
                (planes.masks[usize::from(*i)][row / 64] >> (row % 64)) & 1 == 1
            }
            BoolExpr::Leaf(Operand::Scratch(_)) => false,
            BoolExpr::And(l, r) => direct(l, planes, row) & direct(r, planes, row),
            BoolExpr::Or(l, r) => direct(l, planes, row) | direct(r, planes, row),
            BoolExpr::Xor(l, r) => direct(l, planes, row) ^ direct(r, planes, row),
            BoolExpr::Not(i) => !direct(i, planes, row),
        }
    }

    /// FAILS IF: any fused chain computes a different function than the tree
    /// it came from — 50 seeded trees over five planes, counted through the
    /// oracle against a direct per-row evaluation. The chains must actually
    /// exercise the wide path (asserted: at least one tree needs ≥ 2 ops).
    #[test]
    fn fused_chains_are_semantically_equal_to_their_trees() {
        let n = 130;
        let mut seed = 0xF00Du64;
        let planes_data: Vec<Vec<u64>> = (0..5)
            .map(|_| {
                let mut m: Vec<u64> = (0..3).map(|_| lcg(&mut seed) & lcg(&mut seed)).collect();
                m[2] &= (1u64 << (n % 64)) - 1;
                m
            })
            .collect();
        let masks: Vec<&[u64]> = planes_data.iter().map(|m| m.as_slice()).collect();
        let lanes: [LaneRef<'_>; 0] = [];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut wide = 0;
        for _ in 0..50 {
            let e = random_expr(&mut seed, 4);
            let p = fuse_program(&e, 0, |m| Terminal::Count { mask: m })
                .unwrap_or_else(|_| panic!("fuse failed for {e:?}"));
            if p.ops.len() >= 2 {
                wide += 1;
            }
            let want = (0..n).filter(|&r| direct(&e, &planes, r)).count();
            assert_eq!(
                reference_execute(&p, &planes, None),
                Ok(Value::Count(want)),
                "{e:?} → {:?}",
                p.ops
            );
        }
        assert!(wide >= 5, "only {wide} trees needed the reduction path");
    }
}
