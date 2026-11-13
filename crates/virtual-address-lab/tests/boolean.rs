//! H5, H6, H8: pushing a row bijection through Boolean algebra and through a
//! permutation-invariant terminal, and the controls where that is refused.

use lance_graph_mask_risc::{
    execute_into, Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal,
};
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::perm::{shared, NotBijective, Permutation};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::route::{plan_count_eq, plan_rows_eq, plan_sum_where_eq};
use virtual_address_lab::view::{Trace, VirtualTable};

const N: usize = 1_003;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn mask(seed: u64, one_in: u64) -> Vec<u64> {
    let mut words = vec![0u64; N.div_ceil(64)];
    for i in (0..N).filter(|&i| mix(seed ^ i as u64).is_multiple_of(one_in)) {
        words[i / 64] |= 1u64 << (i % 64);
    }
    words
}

/// A shuffle of `0..N`, from sorting by a hash.
fn shuffle(seed: u64) -> Permutation {
    let mut rows: Vec<u32> = (0..N as u32).collect();
    rows.sort_by_key(|&r| mix(seed ^ u64::from(r)));
    Permutation::try_new(&rows, N).unwrap()
}

fn ternlog(imm: u8, a: &[u64], b: &[u64], c: &[u64]) -> Vec<u64> {
    let program = Program::new(
        vec![MaskOp::Ternlog {
            imm,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let masks: [&[u64]; 3] = [a, b, c];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let mut out = vec![0u64; N.div_ceil(64)];
    let mut scratch = Scratch::for_program(&program, N).unwrap();
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut out),
    )
    .unwrap();
    out
}

fn pop(m: &[u64]) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

/// H5. FAILS IF: for a shared bijection the two sides differ at any of the
/// 256 immediates, or a count changes under it.
#[test]
fn a_shared_bijection_commutes_with_every_ternlog() {
    let (a, b, c) = (mask(1, 2), mask(2, 3), mask(3, 5));
    for p in [Permutation::reversal(N), shuffle(77)] {
        let (pa, pb, pc) = (p.apply(&a), p.apply(&b), p.apply(&c));
        assert_ne!(pa, a, "the bijection moves something");
        let mut moved = 0;
        for imm in 0..=255u8 {
            let inside = ternlog(imm, &pa, &pb, &pc);
            let plain = ternlog(imm, &a, &b, &c);
            assert_eq!(inside, p.apply(&plain), "imm {imm:#04x}");
            assert_eq!(pop(&inside), pop(&plain), "imm {imm:#04x}");
            moved += usize::from(inside != plain);
        }
        assert!(
            moved > 200,
            "only {moved} immediates distinguish the orders"
        );
        assert_eq!(shared([Some(&p), Some(&p), Some(&p)]), Ok(Some(&p)));
    }
    assert_eq!(shared([None, None, None]), Ok(None));
}

/// H6. FAILS IF: a control that must differ does not, or the rewrite is
/// offered for operands that do not share one bijection, or a position list
/// that is not a bijection is accepted as one.
#[test]
fn the_rewrite_is_refused_where_it_is_not_lawful() {
    let (a, b, c) = (mask(1, 2), mask(2, 3), mask(3, 5));
    let p = shuffle(77);
    let and3 = 0x80u8;

    // Only A permuted: no rewrite, and the answer is another one.
    assert_eq!(shared([Some(&p), None, None]), Err(()));
    let plain = ternlog(and3, &a, &b, &c);
    let mixed = ternlog(and3, &p.apply(&a), &b, &c);
    assert_ne!(mixed, plain);
    assert_ne!(mixed, p.apply(&plain));
    assert_ne!(pop(&mixed), pop(&plain), "even the count differs");
    // Two different bijections.
    let q = Permutation::reversal(N);
    assert_eq!(shared([Some(&p), Some(&q), Some(&p)]), Err(()));

    // A selection mask left in the other row order.
    let selection = mask(9, 4);
    let lawful = ternlog(and3, &p.apply(&a), &p.apply(&selection), &p.apply(&c));
    let stale = ternlog(and3, &p.apply(&a), &selection, &p.apply(&c));
    assert_ne!(pop(&stale), pop(&lawful));

    // Not bijections: a duplicate, a gap (short list), a row out of range.
    let mut rows: Vec<u32> = (0..N as u32).collect();
    rows[5] = 6;
    assert_eq!(
        Permutation::try_new(&rows, N),
        Err(NotBijective::Duplicate { at: 6, position: 6 })
    );
    assert_eq!(
        Permutation::try_new(&rows[..N - 1], N),
        Err(NotBijective::Length {
            expected: N,
            got: N - 1
        })
    );
    rows[5] = N as u32;
    assert_eq!(
        Permutation::try_new(&rows, N),
        Err(NotBijective::OutOfRange {
            at: 5,
            position: N as u32
        })
    );
    // A bag is not a set: the same rows with one listed twice count more.
    let bag = [3u32, 3, 8];
    assert_ne!(bag.len(), 2, "a mask of {{3, 8}} would count 2");
}

/// H8 and the order-sensitive control. FAILS IF: a row-reversing recipe
/// changes the program of a count or a sum, or the reversal is dropped for an
/// output in row order.
#[test]
fn a_row_reversal_is_dropped_for_a_fold_and_kept_for_rows() {
    let grid = Grid::new(8, N, |l, p| (mix((l * 7919 + p) as u64) % 9) as i32);
    let world = World::new(8);
    let binding = world.bind(&grid).unwrap();
    let trace = Trace::default();
    let plain = VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &trace).unwrap();
    let turned = VirtualTable::bound(&world, &binding, &grid, Recipe::ROTATE90, &trace).unwrap();

    let want_sum: i64 = (0..N)
        .filter(|&p| grid.cell(3, p) == 4)
        .map(|p| i64::from(grid.cell(1, p)))
        .sum();
    let want_count = (0..N).filter(|&p| grid.cell(3, p) == 4).count();
    assert!(want_count > 50 && want_count < N / 2);

    let mut a = plan_sum_where_eq(&plain, 1, 3, 4).unwrap();
    let mut b = plan_sum_where_eq(&turned, 1, 3, 4).unwrap();
    assert_eq!(a.program, b.program);
    assert!(b.reversed && b.reversal_cancelled && !a.reversed);
    assert_eq!(a.sum(&plain), Ok(want_sum));
    assert_eq!(b.sum(&turned), Ok(want_sum));
    let mut count = plan_count_eq(&turned, 3, 4).unwrap();
    assert!(count.reversal_cancelled);
    assert_eq!(count.count(&turned), Ok(want_count));
    assert_eq!(
        trace.transforms.get() + trace.gathers.get() + trace.copied.get(),
        0
    );

    // Rows: the reversal stays attached, and is applied per demanded row.
    let mut rows = plan_rows_eq(&turned, 3, 4).unwrap();
    assert!(rows.reversed && !rows.reversal_cancelled);
    let mut words = vec![0u64; N.div_ceil(64)];
    let view = rows.rows(&turned, &mut words).unwrap();
    assert_eq!(view.count(), want_count);
    let mut asymmetric = false;
    for r in 0..N {
        let want = grid.cell(3, N - 1 - r) == 4;
        assert_eq!(view.has(r), want, "view row {r}");
        asymmetric |= want != (grid.cell(3, r) == 4);
    }
    assert!(asymmetric, "the selection is not its own mirror image");
    assert_eq!(
        trace.transforms.get(),
        0,
        "no mask was reversed to answer has()"
    );
    // A consumer that insists on a buffer in view order pays one transform.
    let moved = view.to_view_order(&trace);
    assert_eq!(trace.transforms.get(), 1);
    assert!((0..N).all(|r| (moved[r / 64] >> (r % 64) & 1 == 1) == view.has(r)));
}
