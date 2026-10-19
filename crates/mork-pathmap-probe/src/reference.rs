//! Independent oracle: a nested loop over the borrowed rows. Names no upstream
//! type and shares no code with [`crate::boundary`].

use crate::{Answers, JoinInput, Semantics};

/// `Q(x, y, z) :- R(x, y), S(y, z)` by nested loop.
pub fn answers(input: &JoinInput<'_>, semantics: Semantics) -> Answers {
    let mut out = Answers::default();
    for &[x, y] in input.r.rows {
        for &[y2, z] in input.s.rows {
            if y == y2 {
                *out.rows.entry([x, y, z]).or_insert(0) += 1;
            }
        }
    }
    match semantics {
        Semantics::Bag => out,
        Semantics::Set => out.to_set(),
    }
}
