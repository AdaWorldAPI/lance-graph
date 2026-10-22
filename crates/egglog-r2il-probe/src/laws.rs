//! The probe's two arithmetic laws and their domain guards. The whole table;
//! nothing else is admitted.
//!
//! | law        | rewrite                                                        | Exact | Wrapping | Checked | Ieee |
//! |------------|----------------------------------------------------------------|-------|----------|---------|------|
//! | MulAssoc   | `(Mul (Num x) (Mul (Num y) t)) => (Mul (Mul (Num x) (Num y)) t)` | admit | admit    | refuse  | refuse |
//! | ConstFold  | `(Mul (Num x) (Num y)) => (Num (* x y))`                       | admit | refuse   | refuse  | refuse |
//!
//! The domains are the contract's own
//! ([`lance_graph_contract::set_rules::Arith`], set_rules.rs:34); the
//! contract's catalogue is neither copied nor extended here.

use lance_graph_contract::set_rules::Arith;

/// The two laws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LawId {
    /// `x * (y * t) = (x * y) * t`.
    MulAssoc,
    /// `Num x * Num y = Num (x * y)`.
    ConstFold,
}

/// A law admitted in a domain: the legal domain is part of the id, so a rule
/// cannot be named without the arithmetic it is valid in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId {
    /// Which law.
    pub law: LawId,
    /// The arithmetic domain it was admitted under.
    pub domain: Arith,
}

/// Why a law is not admitted in a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawRefusal {
    /// Re-association is not a theorem of this arithmetic.
    ///
    /// * `Checked`: an overflow is an error the caller sees, and
    ///   `set_rules.rs:40-42` says it in one line: "A rewrite that removes an
    ///   overflow the original would report changes the answer." Reassociating
    ///   `2 * (2 * (2 * c))` into `(2 * 2 * 2) * c` can move or remove the
    ///   point at which the checked product leaves `i64`.
    /// * `Ieee`: floating-point multiplication is not associative.
    NotAssociative(Arith),
    /// Constant folding through egglog's `*` is not this domain's multiply.
    ///
    /// egglog's `i64` `*` is `a.checked_mul(b)` (egglog `src/sort/i64.rs:31`):
    /// on overflow the primitive FAILS (no panic, no wrap) and the rule simply
    /// does not fire. That matches no domain except an exact one that never
    /// overflows:
    ///
    /// * `Wrapping`: the fold would need two's-complement `wrapping_mul`, which
    ///   egglog does not offer, so the folded constant could be absent where
    ///   the wrapping program has a value.
    /// * `Checked`: the fold would hide an overflow the original reports.
    /// * `Ieee`: not an integer domain.
    NotExactFold(Arith),
}

/// Admits `law` in `domain`, or says why not.
pub fn admit(law: LawId, domain: Arith) -> Result<RuleId, LawRefusal> {
    match (law, domain) {
        (LawId::MulAssoc, Arith::Exact | Arith::Wrapping) => Ok(RuleId { law, domain }),
        (LawId::MulAssoc, d @ (Arith::Checked | Arith::Ieee)) => Err(LawRefusal::NotAssociative(d)),
        (LawId::ConstFold, Arith::Exact) => Ok(RuleId { law, domain }),
        (LawId::ConstFold, d @ (Arith::Wrapping | Arith::Checked | Arith::Ieee)) => {
            Err(LawRefusal::NotExactFold(d))
        }
    }
}

/// Both laws against one domain, in table order.
pub fn admit_all(domain: Arith) -> (Vec<RuleId>, Vec<(LawId, LawRefusal)>) {
    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    for law in [LawId::MulAssoc, LawId::ConstFold] {
        match admit(law, domain) {
            Ok(id) => admitted.push(id),
            Err(why) => refused.push((law, why)),
        }
    }
    (admitted, refused)
}

/// The egglog `rewrite` for an admitted law, in the `arith` ruleset.
///
/// ConstFold's RHS `(Num (* x y))` uses the `i64` `*` primitive, which is
/// `checked_mul`: an overflowing fold does not fire, it does not panic.
pub fn rewrite_text(id: RuleId) -> &'static str {
    match id.law {
        LawId::MulAssoc => {
            "(rewrite (Mul (Num x) (Mul (Num y) t)) (Mul (Mul (Num x) (Num y)) t) :ruleset arith)"
        }
        LawId::ConstFold => "(rewrite (Mul (Num x) (Num y)) (Num (* x y)) :ruleset arith)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: the table drifts from the spec: every cell is checked, so a
    /// guard that admits too much or too little in any domain fails.
    #[test]
    fn the_whole_table() {
        use Arith::*;
        use LawId::*;
        let want = [
            (MulAssoc, Exact, true),
            (MulAssoc, Wrapping, true),
            (MulAssoc, Checked, false),
            (MulAssoc, Ieee, false),
            (ConstFold, Exact, true),
            (ConstFold, Wrapping, false),
            (ConstFold, Checked, false),
            (ConstFold, Ieee, false),
        ];
        for (law, dom, ok) in want {
            assert_eq!(admit(law, dom).is_ok(), ok, "{law:?} in {dom:?}");
        }
        assert_eq!(
            admit(MulAssoc, Checked),
            Err(LawRefusal::NotAssociative(Checked))
        );
        assert_eq!(
            admit(ConstFold, Wrapping),
            Err(LawRefusal::NotExactFold(Wrapping))
        );
    }
}
