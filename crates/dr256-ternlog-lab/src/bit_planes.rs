//! A one-byte code lane held as eight resident bit-planes: plane `k` holds
//! bit `k` of every row's code. Lab only.
//!
//! Row `i` is bit `i % 64` of word `i / 64`; bits past `rows` are zero.

use lance_graph_mask_risc::{BoolExpr, MaskOp, Operand, Program, Terminal};

/// The planes `(a, b, c)` of the band ternlog: the top three bits of the code.
pub const BAND_BITS: [usize; 3] = [7, 6, 5];

/// Eight bit-planes of one byte-per-row code lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPlanes {
    /// Row count every plane spans.
    pub rows: usize,
    /// `planes[k]` holds bit `k` of every row's code.
    pub planes: [Vec<u64>; 8],
}

impl BitPlanes {
    /// Slice `codes` into eight bit-planes.
    pub fn of(codes: &[u8]) -> BitPlanes {
        let words = codes.len().div_ceil(64);
        let mut planes: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; words]);
        for (i, &code) in codes.iter().enumerate() {
            for (k, plane) in planes.iter_mut().enumerate() {
                if (code >> k) & 1 == 1 {
                    plane[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        BitPlanes {
            rows: codes.len(),
            planes,
        }
    }

    /// Total bytes held by the eight planes.
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(|p| p.len() * 8).sum()
    }
}

/// The operand naming bit-plane `k`, with plane index `k` holding bit `k`.
pub fn plane_op(k: usize) -> Operand {
    Operand::Plane(k as u16)
}

/// The expression `code > t` over the eight planes; `plane(k)` is the operand
/// holding bit `k`. `None` when no code satisfies it (`t == 255`).
pub fn gt(t: u8, plane: impl Fn(usize) -> Operand + Copy) -> Option<BoolExpr> {
    let mut g: Option<BoolExpr> = None;
    for k in 0..8 {
        let x = BoolExpr::Leaf(plane(k));
        if (t >> k) & 1 == 0 {
            g = Some(match g {
                None => x,
                Some(g) => BoolExpr::Or(Box::new(x), Box::new(g)),
            });
        } else {
            g = g.map(|g| BoolExpr::And(Box::new(x), Box::new(g)));
        }
    }
    g
}

/// The expression `code < t` over the eight planes; `None` when no code
/// satisfies it (`t == 0`).
pub fn lt(t: u8, plane: impl Fn(usize) -> Operand + Copy) -> Option<BoolExpr> {
    let mut l: Option<BoolExpr> = None;
    for k in 0..8 {
        let nx = BoolExpr::Not(Box::new(BoolExpr::Leaf(plane(k))));
        if (t >> k) & 1 == 1 {
            l = Some(match l {
                None => nx,
                Some(l) => BoolExpr::Or(Box::new(nx), Box::new(l)),
            });
        } else {
            l = l.map(|l| BoolExpr::And(Box::new(nx), Box::new(l)));
        }
    }
    l
}

/// `a & b`.
pub fn and(a: BoolExpr, b: BoolExpr) -> BoolExpr {
    BoolExpr::And(Box::new(a), Box::new(b))
}

/// Count of rows in `region` whose band (`code >> 5`) is in `subset`: one
/// ternlog over planes 7, 6, 5 with immediate `subset`, then an AND with
/// `region`. Plane index `k` must hold bit `k`; `region` is the caller's plane
/// index (by convention 8).
pub fn band_program(subset: u8, region: Operand) -> Program {
    Program::new(
        vec![
            MaskOp::Ternlog {
                imm: subset,
                a: Operand::Plane(7),
                b: Operand::Plane(6),
                c: Operand::Plane(5),
                dst: 0,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: region,
                dst: 1,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: an unsatisfiable bound yields an expression, or a
    /// satisfiable one yields none.
    #[test]
    fn gt_and_lt_are_none_exactly_at_the_ends() {
        for t in 0..=255u8 {
            assert_eq!(gt(t, plane_op).is_none(), t == 255, "gt {t}");
            assert_eq!(lt(t, plane_op).is_none(), t == 0, "lt {t}");
        }
    }

    /// FAILS IF: the planes of a small code list are not its bits.
    #[test]
    fn planes_hold_the_bits() {
        let bp = BitPlanes::of(&[0b0000_0001, 0b1000_0000, 0xFF]);
        assert_eq!(bp.planes[0][0], 0b101);
        assert_eq!(bp.planes[7][0], 0b110);
        assert_eq!(bp.planes[3][0], 0b100);
        assert_eq!(bp.bytes(), 8 * 8);
    }
}
