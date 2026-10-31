//! The hardcoded relational polyfill as a positional lens (D-DT64-7 §1a, §1b).
//!
//! Hardcoded behaviour is addressed `0xFFFF :: 0xF###`: left side `0xFFFF`,
//! right side a behaviour lens in a concern range. [`POLYFILL_LENS`] is
//! `0xFFFF :: 0xF000`, the relational / mask-risc polyfill.
//!
//! **Position binds; labels are aliases.** Word `p` of this lens is
//! [`POLYFILL`]`[p]`, whatever any view calls it. A view over the lens
//! supplies its language's names for the positions (an R2IL opcode name, a
//! SQL keyword, a Quack variant), and each name is checked against the
//! aliases fixed at that position. A name that belongs to another position,
//! or to none, is refused; it never selects a different word.
//!
//! An alias states that the word computes the same thing as the named
//! operation over a non-empty selection. Over an empty selection SQL answers
//! `NULL` for `SUM`, `MIN` and `MAX`, while this reading answers the fold's
//! identity (`0`, `i64::MAX`, `i64::MIN`); a three-valued binding is refused
//! by [`crate::binding::prepare`], so `NULL` never reaches it.
//!
//! The positions are the [`crate::binding::EXECUTABLE`] opcodes in that
//! order. The opcode, arity and R2IL name of each entry are checked against
//! OGAR's `R2ILVocabulary` by test, not at run time.

use lance_graph_contract::ogar_codebook::compose_classid;
use ogar_loco::FnIndex;
use ogar_r2il::{ALL, ANY, KEY_RUNS, MAX, MIN, SUM};

use crate::{
    INT_ADD, INT_AND, INT_EQUAL, INT_MULT, INT_NOT, INT_OR, INT_SUB, INT_S_DIV, INT_S_LESS,
    INT_XOR, POP_COUNT,
};

/// The left side of every hardcoded behaviour classid.
pub const HARDCODED: u16 = 0xFFFF;

/// The relational / mask-risc polyfill lens, `0xFFFF :: 0xF000`.
pub const POLYFILL_LENS: u32 = compose_classid(HARDCODED, 0xF000);

/// One word of the polyfill lens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenWord {
    /// The fold-dialect opcode the word runs as.
    pub opcode: FnIndex,
    /// The number of stack operands it takes; it always pushes one value.
    pub arity: u8,
    /// Every name a view may give this position. The first is the R2IL
    /// opcode name.
    pub aliases: &'static [&'static str],
}

const fn w(opcode: FnIndex, arity: u8, aliases: &'static [&'static str]) -> FrozenWord {
    FrozenWord {
        opcode,
        arity,
        aliases,
    }
}

/// The polyfill lens, by position.
pub const POLYFILL: [FrozenWord; 17] = [
    w(INT_ADD, 2, &["IntAdd", "+"]),
    w(INT_SUB, 2, &["IntSub", "-"]),
    w(INT_MULT, 2, &["IntMult", "*"]),
    w(INT_S_DIV, 2, &["IntSDiv", "/"]),
    w(INT_EQUAL, 2, &["IntEqual", "=", "Cmp::EqI32", "Cmp::EqU32"]),
    w(INT_S_LESS, 2, &["IntSLess", "<", "Cmp::LtI32"]),
    w(INT_AND, 2, &["IntAnd", "AND", "Filter::and"]),
    w(INT_OR, 2, &["IntOr", "OR", "Filter::or"]),
    w(INT_XOR, 2, &["IntXor", "XOR"]),
    w(INT_NOT, 1, &["IntNot", "NOT", "Filter::negate"]),
    w(POP_COUNT, 1, &["PopCount", "COUNT", "Agg::Count"]),
    w(SUM, 2, &["SUM", "Agg::SumI32"]),
    w(MIN, 2, &["MIN", "Agg::MinI32"]),
    w(MAX, 2, &["MAX", "Agg::MaxI32"]),
    w(ANY, 1, &["ANY", "EXISTS", "Agg::Any"]),
    w(ALL, 1, &["ALL", "Agg::All"]),
    w(
        KEY_RUNS,
        2,
        &["KEY_RUNS", "COUNT DISTINCT", "Agg::CountDistinctOrderedU32"],
    ),
];

/// The lens's declared arities, in position order, for
/// `BindStore::declare_algebra`.
#[must_use]
pub fn polyfill_arities() -> Vec<u8> {
    POLYFILL.iter().map(|w| w.arity).collect()
}

/// The position whose aliases include `label`, if exactly one does.
#[must_use]
pub fn position_of(label: &str) -> Option<u8> {
    let mut hits = POLYFILL
        .iter()
        .enumerate()
        .filter(|(_, w)| w.aliases.contains(&label));
    match (hits.next(), hits.next()) {
        (Some((p, _)), None) => u8::try_from(p).ok(),
        _ => None,
    }
}
