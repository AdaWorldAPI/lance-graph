//! A row bijection, and the one rewrite it licenses.

/// Why a position list is not a bijection of `0..n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotBijective {
    /// The list does not have `n` entries.
    Length {
        expected: usize,
        got: usize,
    },
    OutOfRange {
        at: usize,
        position: u32,
    },
    Duplicate {
        at: usize,
        position: u32,
    },
}

/// A bijection of `0..n`: view row `i` is physical row `positions[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation(Vec<u32>);

impl Permutation {
    /// Accepts `positions` only if every row of `0..n` is listed once.
    pub fn try_new(positions: &[u32], n: usize) -> Result<Permutation, NotBijective> {
        if positions.len() != n {
            return Err(NotBijective::Length {
                expected: n,
                got: positions.len(),
            });
        }
        let mut seen = vec![false; n];
        for (at, &position) in positions.iter().enumerate() {
            let slot = seen
                .get_mut(position as usize)
                .ok_or(NotBijective::OutOfRange { at, position })?;
            if *slot {
                return Err(NotBijective::Duplicate { at, position });
            }
            *slot = true;
        }
        Ok(Permutation(positions.to_vec()))
    }

    pub fn reversal(n: usize) -> Permutation {
        Permutation((0..n as u32).rev().collect())
    }

    pub fn positions(&self) -> &[u32] {
        &self.0
    }

    /// `P(M)`: bit `i` of the result is bit `positions[i]` of `mask`. A test
    /// and baseline helper; it builds a mask.
    pub fn apply(&self, mask: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; mask.len()];
        for (i, &p) in self.0.iter().enumerate() {
            let p = p as usize;
            if mask[p / 64] >> (p % 64) & 1 == 1 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }
}

/// The permutation three Boolean operands share, if they share one: the only
/// case in which `f(P a, P b, P c)` may be rewritten as `P f(a, b, c)`.
/// `None` for an operand means "in physical order".
pub fn shared<'p>(operands: [Option<&'p Permutation>; 3]) -> Result<Option<&'p Permutation>, ()> {
    let [a, b, c] = operands;
    if a == b && b == c {
        Ok(a)
    } else {
        Err(())
    }
}
