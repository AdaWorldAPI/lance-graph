//! Scheduler rewrite falsifier: a fold may distribute over lane arithmetic only
//! when that arithmetic's row semantics survive the rewrite.
//!
//! The tempting rewrite
//!
//! `SUM(wrapping_i32(a + b))  ->  SUM(a) + SUM(b)`
//!
//! is NOT generally semantics-preserving. The lane expression is 32-bit
//! wrapping arithmetic, while `Terminal::MaskedSumI32` widens each selected
//! row to `i64` before accumulating. If any selected row overflows its `i32`
//! add, distributing the fold silently changes the program.
//!
//! This is the first scheduler law: expression folding needs a proof that the
//! row operator is exact on every selected row (or a separately defined
//! widened operator). A Rayon-shaped "associative reduce" argument is not
//! enough, because it ignores the semantic boundary between row arithmetic
//! and the widened terminal fold.

use lance_graph_mask_risc::{execute, LaneRef, Operand, Planes, Program, Scratch, Terminal, Value};

/// Execute the shipped widened `MaskedSumI32` over an all-selected lane.
///
/// The mask is resident input, not an intermediate population: this helper
/// introduces no mask operation and therefore isolates only the reduction law
/// the scheduler wants to rewrite.
fn sum_all(lane: &[i32]) -> i64 {
    let words = lane.len().div_ceil(64);
    let mut mask = vec![u64::MAX; words];
    if let Some(last) = mask.last_mut() {
        let rem = lane.len() % 64;
        if rem != 0 {
            *last = (1u64 << rem) - 1;
        }
    }

    let masks: [&[u64]; 1] = [&mask];
    let lanes = [LaneRef::I32(lane)];
    let planes = Planes {
        n_rows: lane.len(),
        masks: &masks,
        lanes: &lanes,
    };
    let program = Program::new(
        vec![],
        Terminal::MaskedSumI32 {
            mask: Operand::Plane(0),
            lane: 0,
        },
    );
    let mut scratch = Scratch::for_program(&program, lane.len()).expect("addressable");
    match execute(&program, &planes, &mut scratch, None).expect("sum executes") {
        Value::SumI64(v) => v,
        other => panic!("MaskedSumI32 has a fixed result shape, got {other:?}"),
    }
}

/// FAILS IF a scheduler treats `SUM(a+b)` as distributive without preserving
/// the row-level `i32` wrapping semantics.
///
/// Row 0 is the entire counterexample: `i32::MAX + 1` wraps to `i32::MIN`.
/// The literal program therefore contributes `-2^31`; distributing the
/// widened fold contributes `2^31`. Both are individually valid i64 sums,
/// so no terminal overflow guard can rescue the rewrite.
#[test]
fn wrapping_row_overflow_forbids_distributing_sum_over_add() {
    let a = [i32::MAX, 0];
    let b = [1, 0];
    let derived: Vec<i32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();

    let literal = sum_all(&derived);
    let distributed = sum_all(&a) + sum_all(&b);

    assert_eq!(literal, i64::from(i32::MIN), "literal wrapping lane");
    assert_eq!(
        distributed,
        i64::from(i32::MAX) + 1,
        "widened independent folds"
    );
    assert_ne!(
        literal, distributed,
        "the rewrite must remain illegal without a per-row no-overflow proof"
    );
}

/// Positive control: when every selected row's add is exact in `i32`, the
/// same distribution is valid. This keeps the negative result narrow: SUM is
/// not intrinsically non-distributive; the missing scheduler fact is the row
/// arithmetic's overflow domain.
#[test]
fn distributing_sum_over_add_is_valid_when_every_row_add_is_exact() {
    let a: [i32; 4] = [10, -7, 2, 100];
    let b: [i32; 4] = [3, 11, -5, -40];
    let derived: Vec<i32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();

    for (&x, &y) in a.iter().zip(b.iter()) {
        assert!(
            x.checked_add(y).is_some(),
            "fixture must stay inside the exact i32 domain"
        );
    }

    let literal = sum_all(&derived);
    let distributed = sum_all(&a) + sum_all(&b);

    assert_eq!(literal, distributed);
    assert_eq!(literal, 74);
}
