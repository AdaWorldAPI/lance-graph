//! Lane laws: a value lane declared as a rule over a key lane, never stored.
//!
//! A row is `identity[index].key : identity[index].value` (EXO-332). When the
//! value follows from the key by a rule, the value lane does not exist as
//! data: `identity[index].value = law(identity[index].key)`, computed from the
//! key lane where it sits. Three stacked tables
//!
//! ```text
//! 1 2 3 4 5    9 5 7 8 4    5 8 5 3 8
//! 6 7 8 9 0    4 0 2 3 9    0 3 0 8 3
//! ```
//!
//! are one law (`+5 mod 10`) over three key rows, and no value row is kept.
//!
//! # The register contract
//!
//! The op set is closed and small, like an ISA register contract: every op is
//! an affine map `x ↦ (mul·x + add) mod m` over one modulus `m` (`0` means
//! `2^32`, plain `u32` wrap). Affine maps compose to affine maps, so a stack of
//! laws (one pipe per law) folds to ONE op when it is declared: reading through
//! a stack of N laws costs one multiply-add, and a stack that folds to the
//! identity costs nothing. Laws over different moduli do not compose and are
//! refused rather than chained.
//!
//! A law is a property of a lane, not a query operation: it adds no planner or
//! V4 opcode. What is NOT here: lookups through a stored table, and anything
//! that needs a second lane as input.
//!
//! Zero dependencies, `Copy`, no allocation.

/// One affine op `x ↦ (mul·x + add) mod m`, the modulus given by the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LawOp {
    pub mul: u32,
    pub add: u32,
}

impl LawOp {
    /// `x ↦ x + add`.
    pub const fn add(add: u32) -> Self {
        Self { mul: 1, add }
    }

    /// `x ↦ mul·x`.
    pub const fn mul(mul: u32) -> Self {
        Self { mul, add: 0 }
    }
}

/// Why a law was not declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawRefusal {
    /// An op's constant is not below the program's modulus.
    OutOfModulus { modulus: u32, value: u32 },
    /// Two programs over different moduli do not compose.
    ModulusMismatch { left: u32, right: u32 },
}

/// A stack of laws over one modulus, folded to a single affine op as it is
/// built. `modulus == 0` means `2^32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LawProgram {
    modulus: u32,
    op: LawOp,
}

impl LawProgram {
    /// The identity over `modulus` (`0` = `2^32`).
    pub const fn identity(modulus: u32) -> Self {
        Self {
            modulus,
            op: LawOp { mul: 1, add: 0 },
        }
    }

    pub const fn modulus(&self) -> u32 {
        self.modulus
    }

    /// The folded op.
    pub const fn op(&self) -> LawOp {
        self.op
    }

    /// True when reading through the program returns the key unchanged.
    pub fn is_identity(&self) -> bool {
        let m = self.m();
        u64::from(self.op.mul) % m == 1 % m && u64::from(self.op.add) % m == 0
    }

    fn m(&self) -> u64 {
        if self.modulus == 0 {
            1 << 32
        } else {
            u64::from(self.modulus)
        }
    }

    /// This program, then `op`: folded into one op.
    pub fn then(self, op: LawOp) -> Result<Self, LawRefusal> {
        let m = self.m();
        for value in [op.mul, op.add] {
            if self.modulus != 0 && value >= self.modulus {
                return Err(LawRefusal::OutOfModulus {
                    modulus: self.modulus,
                    value,
                });
            }
        }
        // op ∘ self: x ↦ op.mul·(self.mul·x + self.add) + op.add.
        let (a1, b1) = (u64::from(self.op.mul), u64::from(self.op.add));
        let (a2, b2) = (u64::from(op.mul), u64::from(op.add));
        let mul = (a2 * a1) % m;
        let add = ((a2 * b1) % m + b2) % m;
        Ok(Self {
            modulus: self.modulus,
            op: LawOp {
                mul: mul as u32,
                add: add as u32,
            },
        })
    }

    /// This program, then `next`: one program over the same modulus.
    pub fn compose(self, next: LawProgram) -> Result<Self, LawRefusal> {
        if self.modulus != next.modulus {
            return Err(LawRefusal::ModulusMismatch {
                left: self.modulus,
                right: next.modulus,
            });
        }
        self.then(next.op)
    }

    /// The value for `key`, or `None` when `key` is not below the modulus
    /// (outside the law's domain).
    #[inline]
    pub fn apply(&self, key: u32) -> Option<u32> {
        let m = self.m();
        let x = u64::from(key);
        if x >= m {
            return None;
        }
        Some(((u64::from(self.op.mul) * x % m + u64::from(self.op.add)) % m) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: [([u32; 5], [u32; 5]); 3] = [
        ([1, 2, 3, 4, 5], [6, 7, 8, 9, 0]),
        ([9, 5, 7, 8, 4], [4, 0, 2, 3, 9]),
        ([5, 8, 5, 3, 8], [0, 3, 0, 8, 3]),
    ];

    /// FAILS IF: one law does not produce all three value rows from their key
    /// rows.
    #[test]
    fn one_law_reads_three_tables() {
        let law = LawProgram::identity(10).then(LawOp::add(5)).unwrap();
        for (keys, values) in ROWS {
            let got: Vec<u32> = keys.iter().map(|&k| law.apply(k).unwrap()).collect();
            assert_eq!(got, values);
        }
    }

    /// FAILS IF: a stack does not fold to one op, or a stack that cancels is
    /// not the identity.
    #[test]
    fn a_stack_folds_to_one_op() {
        let plus5 = LawProgram::identity(10).then(LawOp::add(5)).unwrap();
        let twice = plus5.compose(plus5).unwrap();
        assert!(twice.is_identity());
        let thrice = twice.compose(plus5).unwrap();
        assert_eq!(thrice, plus5);
        assert!(!plus5.is_identity());

        let affine = LawProgram::identity(10)
            .then(LawOp::mul(3))
            .unwrap()
            .then(LawOp::add(7))
            .unwrap();
        assert_eq!(affine.op(), LawOp { mul: 3, add: 7 });
        for k in 0..10 {
            assert_eq!(affine.apply(k), Some((3 * k + 7) % 10));
        }
    }

    /// FAILS IF: a key outside the modulus is answered, a constant outside it
    /// is admitted, or two moduli are chained.
    #[test]
    fn domains_and_moduli_are_refused() {
        let plus5 = LawProgram::identity(10).then(LawOp::add(5)).unwrap();
        assert_eq!(plus5.apply(10), None);
        assert_eq!(
            LawProgram::identity(10).then(LawOp::add(12)),
            Err(LawRefusal::OutOfModulus {
                modulus: 10,
                value: 12
            })
        );
        assert_eq!(
            plus5.compose(LawProgram::identity(7)),
            Err(LawRefusal::ModulusMismatch { left: 10, right: 7 })
        );
        let wrap = LawProgram::identity(0).then(LawOp::add(1)).unwrap();
        assert_eq!(wrap.apply(u32::MAX), Some(0));
    }
}
