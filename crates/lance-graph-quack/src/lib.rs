//! A columnar query surface whose **operators are masking ops**.
//!
//! # What this is, and the shape it deliberately does NOT have
//!
//! DuckDB's operator set — scan, filter, project, aggregate — expressed so
//! that every operator LOWERS to a [`Program`] and is executed by the one
//! evaluator in `lance-graph-mask-risc`, which sits on `ndarray::simd`'s
//! masking algebra.
//!
//! What a columnar engine normally grows, and what is absent here on purpose:
//!
//! | the usual shape | why it is absent |
//! |---|---|
//! | an expression interpreter (`Expr` tree walked per batch) | a filter IS a `Pred`; walking a tree per batch is the second evaluator |
//! | a row iterator / `next()` volcano loop | the unit is a mask over `n_rows`, never a row |
//! | a per-operator kernel library | every operator is a `MaskOp` composition; a new operator is a new LOWERING, never a new kernel |
//! | a physical-plan `dyn Operator` chain | a plan is a `Program` — one flat op list, one terminal |
//!
//! The rule that keeps it honest: **this crate may build a [`Program`] and
//! must never evaluate one.** `execute` is called by the consumer, on a
//! scratch the consumer owns. A single `match` over operators here that
//! computed anything would be the duplicate evaluator the whole arc exists to
//! avoid.
//!
//! # Status
//!
//! Scaffold. The lowering surface below is real and tested; the operator set
//! is deliberately the minimum that proves the shape — **filter and count** —
//! because an operator without a falsifier is a claim. Join, group-by and
//! projection land one at a time, each with the differential that shows it
//! agrees with an independent reading.

#![forbid(unsafe_code)]

use lance_graph_mask_risc::{MaskOp, Operand, Pred, Program, Terminal};

/// A column reference — an index into [`Planes::lanes`](lance_graph_mask_risc::Planes).
///
/// Not a name: name resolution is the catalogue's job and this crate has no
/// catalogue. A consumer that has names resolves them before it gets here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Col(pub u16);

/// A predicate over one column — the filter's whole vocabulary.
///
/// One variant per masking op that produces a mask from a value lane. That is
/// not a coincidence and not a coding convenience: **the query language's
/// predicate set IS the masking algebra's predicate set**, so a predicate this
/// crate cannot spell is one the substrate cannot run, and adding one here
/// without adding it below would be the first crack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// `col == v` over a signed lane.
    EqI32(i32),
    /// `col != v`.
    NeI32(i32),
    /// `col < v`.
    LtI32(i32),
    /// `col <= v`.
    LeI32(i32),
    /// `col > v`.
    GtI32(i32),
    /// `col >= v`.
    GeI32(i32),
    /// `col == v` over an unsigned lane, exact bitwise.
    EqU32(u32),
    /// `col != v`.
    NeU32(u32),
    /// `(col ^ pattern) & care == 0` — the ternary match, which SQL has no
    /// spelling for and the substrate has had all along.
    MatchU32 {
        /// The bits to compare.
        pattern: u32,
        /// Which bits participate; zero means "don't care".
        care: u32,
    },
}

/// A filter expression: predicates over columns, composed with AND/OR/NOT.
///
/// Deliberately a tree HERE and never at run time — it is lowered once into a
/// flat [`Program`] and the tree is gone before anything executes.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// A leaf comparison on one column.
    Cmp(Col, Cmp),
    /// Every child must hold.
    And(Vec<Filter>),
    /// Some child must hold.
    Or(Vec<Filter>),
    /// The child must not hold.
    Not(Box<Filter>),
}

impl Filter {
    /// `col <cmp>` — the leaf builder, so call sites read as the query does.
    pub fn cmp(col: Col, cmp: Cmp) -> Self {
        Filter::Cmp(col, cmp)
    }

    /// Conjunction.
    pub fn and(parts: impl IntoIterator<Item = Filter>) -> Self {
        Filter::And(parts.into_iter().collect())
    }

    /// Disjunction.
    pub fn or(parts: impl IntoIterator<Item = Filter>) -> Self {
        Filter::Or(parts.into_iter().collect())
    }

    /// Negation.
    ///
    /// Named `negate` rather than `not`: an inherent `not` shadows
    /// `std::ops::Not` and reads ambiguously at a call site that also uses the
    /// operator. Clippy's `should_implement_trait` is right here.
    pub fn negate(inner: Filter) -> Self {
        Filter::Not(Box::new(inner))
    }
}

/// Why a query could not be lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LowerError {
    /// An `And`/`Or` with no children. Refused rather than folded to a
    /// constant: an empty conjunction is `true` and an empty disjunction is
    /// `false`, and a caller that built one by accident wants to hear about it
    /// rather than receive whichever identity this crate happened to pick.
    EmptyJunction,
    /// The program would need more scratch slots than a `u16` can name.
    TooManySlots {
        /// The count that overflowed.
        needed: usize,
    },
}

impl core::fmt::Display for LowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LowerError::EmptyJunction => {
                write!(f, "an AND/OR with no children has no non-arbitrary meaning")
            }
            LowerError::TooManySlots { needed } => {
                write!(f, "needs {needed} scratch slots; the address space is u16")
            }
        }
    }
}

/// What a query asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// How many rows survive the filter.
    Count,
    /// Whether any row survives.
    Any,
    /// Whether every row survives.
    All,
    /// Σ over a signed lane, restricted to surviving rows.
    SumI32(Col),
    /// min over surviving rows.
    MinI32(Col),
    /// max over surviving rows.
    MaxI32(Col),
}

/// One query: a filter and what to ask of the rows that pass it.
///
/// # Why the filter is NOT optional
///
/// A bare `SELECT count(*)` — every row, no predicate — has no cheap spelling
/// here, and the reason is worth recording rather than papering over: the IR
/// has **no `Fill`/const op**, so "all rows" cannot be written into a scratch
/// slot directly, and a slot that no op has written is refused at validation
/// (`ExecError::ScratchReadBeforeWrite`).
///
/// A first draft of this crate lowered it as `Pred::NeU32{lane: 0, v: 0}`
/// followed by `Ternlog{imm: 0xFF}` — the ternlog ignores its inputs, so the
/// predicate existed only to make slot 0 written. That is a **latent
/// lane-kind bug**: it typechecks only when lane 0 happens to be `U32`, and
/// fails with a kind error on any schema where it is not. It is recorded here
/// instead of shipped.
///
/// The honest spellings, for whoever closes this: a tautology over a
/// caller-named lane (`NOT p OR p`, two preds and an `Or`, lane-kind correct
/// by construction), or an unfiltered `Terminal` reading a resident mask plane
/// the caller supplies. Both are real; neither is guessable from this API as
/// it stands, so the type refuses rather than picks.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// The filter. Required — see the type's own doc.
    pub filter: Filter,
    /// The aggregate.
    pub agg: Agg,
}

/// Lower a query to a [`Program`].
///
/// # The allocation rule
///
/// Slots are assigned by a strict post-order walk, and a junction's children
/// are folded left-to-right into the FIRST child's slot. So a filter of depth
/// `d` costs `d + 1` slots, never one per leaf — which is what keeps a wide
/// conjunction from asking for a scratch arena proportional to its width.
///
/// # Errors
///
/// [`LowerError::EmptyJunction`] for an `And`/`Or` with no children;
/// [`LowerError::TooManySlots`] if the walk needs more than `u16::MAX + 1`.
pub fn lower(q: &Query) -> Result<Program, LowerError> {
    let mut ops = Vec::new();
    let mut high_water = 0usize;

    lower_filter(&q.filter, 0, &mut ops, &mut high_water)?;
    let mask = Operand::Scratch(0);

    let terminal = match q.agg {
        Agg::Count => Terminal::Count { mask },
        Agg::Any => Terminal::Any { mask },
        Agg::All => Terminal::All { mask },
        Agg::SumI32(c) => Terminal::MaskedSumI32 { mask, lane: c.0 },
        Agg::MinI32(c) => Terminal::MaskedMinI32 { mask, lane: c.0 },
        Agg::MaxI32(c) => Terminal::MaskedMaxI32 { mask, lane: c.0 },
    };

    let slots = high_water.max(1);
    let scratch_slots =
        u32::try_from(slots).map_err(|_| LowerError::TooManySlots { needed: slots })?;

    Ok(Program {
        ops,
        terminal,
        scratch_slots,
    })
}

/// Lower one filter node into `dst`, tracking the highest slot touched.
fn lower_filter(
    f: &Filter,
    dst: u16,
    ops: &mut Vec<MaskOp>,
    high_water: &mut usize,
) -> Result<(), LowerError> {
    *high_water = (*high_water).max(usize::from(dst) + 1);
    match f {
        Filter::Cmp(col, cmp) => {
            ops.push(MaskOp::Pred {
                pred: pred_of(*col, *cmp),
                under: None,
                dst,
            });
            Ok(())
        }
        Filter::Not(inner) => {
            lower_filter(inner, dst, ops, high_water)?;
            ops.push(MaskOp::Not {
                a: Operand::Scratch(dst),
                dst,
            });
            Ok(())
        }
        Filter::And(parts) | Filter::Or(parts) => {
            let (first, rest) = parts.split_first().ok_or(LowerError::EmptyJunction)?;
            lower_filter(first, dst, ops, high_water)?;
            // The next slot up is scratch for each sibling in turn, so width
            // costs one slot, not one per child.
            let tmp = dst.checked_add(1).ok_or(LowerError::TooManySlots {
                needed: usize::from(dst) + 2,
            })?;
            let is_and = matches!(f, Filter::And(_));
            for part in rest {
                lower_filter(part, tmp, ops, high_water)?;
                ops.push(if is_and {
                    MaskOp::And {
                        a: Operand::Scratch(dst),
                        b: Operand::Scratch(tmp),
                        dst,
                    }
                } else {
                    MaskOp::Or {
                        a: Operand::Scratch(dst),
                        b: Operand::Scratch(tmp),
                        dst,
                    }
                });
            }
            Ok(())
        }
    }
}

/// One comparison → one `Pred`. Exhaustive, and the compiler keeps it so.
fn pred_of(col: Col, cmp: Cmp) -> Pred {
    let lane = col.0;
    match cmp {
        Cmp::EqI32(v) => Pred::EqI32 { lane, v },
        Cmp::NeI32(v) => Pred::NeI32 { lane, v },
        Cmp::LtI32(t) => Pred::LtI32 { lane, t },
        Cmp::LeI32(t) => Pred::LeI32 { lane, t },
        Cmp::GtI32(t) => Pred::GtI32 { lane, t },
        Cmp::GeI32(t) => Pred::GeI32 { lane, t },
        Cmp::EqU32(v) => Pred::EqU32 { lane, v },
        Cmp::NeU32(v) => Pred::NeU32 { lane, v },
        Cmp::MatchU32 { pattern, care } => Pred::MatchU32 {
            lane,
            pattern,
            care,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_mask_risc::{execute, scratch_words_for, LaneRef, Planes, Scratch, Value};

    const N: usize = 1000;

    /// Two lanes: a signed value lane and an unsigned class lane.
    fn fixture() -> (Vec<i32>, Vec<u32>) {
        let vals: Vec<i32> = (0..N as i32).map(|i| (i * 7) % 401 - 200).collect();
        let classes: Vec<u32> = (0..N as u32).map(|i| i % 5).collect();
        (vals, classes)
    }

    /// The INDEPENDENT reading — a plain row loop that shares no code with the
    /// lowering. This is the oracle: a filter tree walked per row, which is
    /// exactly the shape this crate refuses to ship, written here because a
    /// test oracle is the one place it is licensed.
    fn oracle(f: &Filter, vals: &[i32], classes: &[u32], row: usize) -> bool {
        match f {
            Filter::Cmp(col, cmp) => {
                let v = vals[row];
                let c = classes[row];
                match cmp {
                    Cmp::EqI32(x) => v == *x,
                    Cmp::NeI32(x) => v != *x,
                    Cmp::LtI32(x) => v < *x,
                    Cmp::LeI32(x) => v <= *x,
                    Cmp::GtI32(x) => v > *x,
                    Cmp::GeI32(x) => v >= *x,
                    Cmp::EqU32(x) => c == *x,
                    Cmp::NeU32(x) => c != *x,
                    Cmp::MatchU32 { pattern, care } => (c ^ *pattern) & *care == 0,
                }
                .then_some(*col)
                .is_some()
            }
            Filter::And(ps) => ps.iter().all(|p| oracle(p, vals, classes, row)),
            Filter::Or(ps) => ps.iter().any(|p| oracle(p, vals, classes, row)),
            Filter::Not(p) => !oracle(p, vals, classes, row),
        }
    }

    fn run(q: &Query, vals: &[i32], classes: &[u32]) -> Value {
        let program = lower(q).expect("lowers");
        let lanes = [LaneRef::I32(vals), LaneRef::U32(classes)];
        let planes = Planes {
            n_rows: vals.len(),
            masks: &[],
            lanes: &lanes,
        };
        let words = vals.len().div_ceil(64);
        let slots = program.scratch_slots as usize;
        let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
        let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
        execute(&program, &planes, &mut scratch, None).expect("runs")
    }

    /// FAILS IF: the lowering and an independent per-row reading of the same
    /// filter disagree.
    ///
    /// This is the whole claim of the crate — that a query expressed as
    /// masking ops answers what the query means — so it is checked against an
    /// oracle that never sees a `Program`.
    #[test]
    fn every_lowered_filter_agrees_with_an_independent_per_row_reading() {
        let (vals, classes) = fixture();
        let cases: Vec<(&str, Filter)> = vec![
            ("one leaf", Filter::cmp(Col(0), Cmp::GtI32(0))),
            ("u32 leaf", Filter::cmp(Col(1), Cmp::EqU32(2))),
            (
                "and of two",
                Filter::and([
                    Filter::cmp(Col(0), Cmp::GtI32(-50)),
                    Filter::cmp(Col(1), Cmp::NeU32(0)),
                ]),
            ),
            (
                "or of two",
                Filter::or([
                    Filter::cmp(Col(0), Cmp::LtI32(-150)),
                    Filter::cmp(Col(1), Cmp::EqU32(4)),
                ]),
            ),
            ("not", Filter::negate(Filter::cmp(Col(0), Cmp::GeI32(0)))),
            (
                "and of four — the width case",
                Filter::and([
                    Filter::cmp(Col(0), Cmp::GtI32(-180)),
                    Filter::cmp(Col(0), Cmp::LtI32(180)),
                    Filter::cmp(Col(1), Cmp::NeU32(3)),
                    Filter::cmp(Col(1), Cmp::NeU32(1)),
                ]),
            ),
            (
                "nested — or inside and, with a not",
                Filter::and([
                    Filter::cmp(Col(0), Cmp::GeI32(-100)),
                    Filter::or([
                        Filter::cmp(Col(1), Cmp::EqU32(1)),
                        Filter::negate(Filter::cmp(Col(0), Cmp::GtI32(100))),
                    ]),
                ]),
            ),
            (
                "ternary match — no SQL spelling, the substrate had it all along",
                Filter::cmp(
                    Col(1),
                    Cmp::MatchU32 {
                        pattern: 0b100,
                        care: 0b110,
                    },
                ),
            ),
        ];

        for (label, f) in cases {
            let expected = (0..N).filter(|&r| oracle(&f, &vals, &classes, r)).count();
            let q = Query {
                filter: f,
                agg: Agg::Count,
            };
            assert_eq!(
                run(&q, &vals, &classes),
                Value::Count(expected),
                "{label}: the lowered program and the per-row oracle disagree"
            );
            // Anti-vacuity: agreement on "nothing" or "everything" would hold
            // for a lowering that ignored the filter entirely.
            assert!(
                expected > 0 && expected < N,
                "{label} selects {expected}/{N} — a degenerate case proves nothing"
            );
        }
    }

    /// FAILS IF: a junction allocates a slot per child.
    ///
    /// Children fold left-to-right into the first child's slot, so width is
    /// free and only DEPTH costs. Without this a 64-wide conjunction would ask
    /// for a 64-slot arena — a scratch allocation proportional to the query's
    /// text rather than to its shape.
    #[test]
    fn a_wide_conjunction_costs_one_extra_slot_not_one_per_child() {
        let wide = Filter::and((0..32).map(|i| Filter::cmp(Col(0), Cmp::NeI32(i))));
        let p = lower(&Query {
            filter: wide,
            agg: Agg::Count,
        })
        .expect("lowers");
        assert_eq!(
            p.scratch_slots, 2,
            "32 children must cost depth-2, not 32 slots"
        );

        // Paired: DEPTH does cost, or the constant above would be vacuous.
        let deep = Filter::and([
            Filter::cmp(Col(0), Cmp::GtI32(0)),
            Filter::or([
                Filter::cmp(Col(1), Cmp::EqU32(1)),
                Filter::and([
                    Filter::cmp(Col(0), Cmp::LtI32(50)),
                    Filter::cmp(Col(1), Cmp::NeU32(2)),
                ]),
            ]),
        ]);
        let pd = lower(&Query {
            filter: deep,
            agg: Agg::Count,
        })
        .expect("lowers");
        assert!(
            pd.scratch_slots > p.scratch_slots,
            "depth must cost more than width: deep={} wide={}",
            pd.scratch_slots,
            p.scratch_slots
        );
    }

    /// FAILS IF: an empty AND/OR is folded to a constant instead of refused.
    ///
    /// An empty conjunction is `true` and an empty disjunction is `false`, so
    /// whichever identity the code picked would silently be the answer to a
    /// query the caller built by accident.
    #[test]
    fn an_empty_junction_is_refused_rather_than_folded_to_an_identity() {
        for f in [Filter::And(vec![]), Filter::Or(vec![])] {
            assert_eq!(
                lower(&Query {
                    filter: f,
                    agg: Agg::Count
                }),
                Err(LowerError::EmptyJunction)
            );
        }
    }

    /// FAILS IF: the aggregates do not read the filter's mask.
    ///
    /// `Count`/`Any`/`All`/`Sum`/`Min`/`Max` over the SAME filter must be
    /// mutually consistent with the per-row oracle — an aggregate wired to the
    /// wrong operand would still return a plausible number.
    #[test]
    fn every_aggregate_reads_the_same_filter_mask() {
        let (vals, classes) = fixture();
        let f = Filter::and([
            Filter::cmp(Col(0), Cmp::GtI32(-100)),
            Filter::cmp(Col(1), Cmp::EqU32(3)),
        ]);
        let rows: Vec<usize> = (0..N).filter(|&r| oracle(&f, &vals, &classes, r)).collect();
        assert!(
            !rows.is_empty() && rows.len() < N,
            "fixture must be a proper subset"
        );

        let sum: i64 = rows.iter().map(|&r| i64::from(vals[r])).sum();
        let min = rows.iter().map(|&r| vals[r]).min();
        let max = rows.iter().map(|&r| vals[r]).max();

        let q = |agg| Query {
            filter: f.clone(),
            agg,
        };
        assert_eq!(
            run(&q(Agg::Count), &vals, &classes),
            Value::Count(rows.len())
        );
        assert_eq!(run(&q(Agg::Any), &vals, &classes), Value::Bool(true));
        assert_eq!(run(&q(Agg::All), &vals, &classes), Value::Bool(false));
        assert_eq!(
            run(&q(Agg::SumI32(Col(0))), &vals, &classes),
            Value::SumI64(sum)
        );
        assert_eq!(
            run(&q(Agg::MinI32(Col(0))), &vals, &classes),
            Value::OptI32(min)
        );
        assert_eq!(
            run(&q(Agg::MaxI32(Col(0))), &vals, &classes),
            Value::OptI32(max)
        );
    }
}
