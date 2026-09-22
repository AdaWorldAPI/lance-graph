//! Rotation is a coordinate system, not a mask op.
//!
//! Toy: values `A B / C D` stored ONCE as one lane in row-major order. A
//! rotation or transpose is an address map ρ over that storage; nothing is
//! copied, no rotated buffer, no reordered mask. Two consequences are
//! proved here, on the existing IR and the existing T1 words only:
//!
//! 1. For any bijection ρ on rows, `ρ(A & B) = ρ(A) & ρ(B)` (likewise `|`,
//!    `^`), so predicates and masks are computed ONCE in canonical
//!    coordinates and the coordinate transform is absorbed at the consumer:
//!    a reducer (`Count`, `MaskedSumI32`) is invariant under ρ and needs no
//!    view at all; an order-sensitive sink reads bit `ρ(i)` through the
//!    map. `MaskOp::Rotate` does not exist and must not.
//! 2. When the consumer's coordinates are a stride/offset over the same
//!    bytes (a column of the 2×2, i.e. the transpose or a 90° rotation),
//!    the predicate itself reads through that projection —
//!    `ndarray::simd::eq_u32_strided_to_mask` over the canonical storage —
//!    and its mask equals the canonical mask read through ρ. Same bytes,
//!    different coordinates, fold directly.

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{
    scratch_words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal,
    Value,
};
use ndarray::simd::eq_u32_strided_to_mask;

/// `A B / C D` = `[1, 2, 3, 4]`, row-major, the ONLY copy of the values.
const CANON: [i32; 4] = [1, 2, 3, 4];

/// Row-major index of cell `(r, c)` in a 2×2.
fn rm(r: usize, c: usize) -> usize {
    r * 2 + c
}

/// 90° clockwise: `A B / C D` → `C A / D B`… the addendum's example is
/// `A B / C D` → `B D / A C` (counter-clockwise): new `(r, c)` reads old
/// `(c, 1 - r)`.
fn rot_ccw(r: usize, c: usize) -> usize {
    rm(c, 1 - r)
}

/// Transpose: `A B / C D` → `A C / B D`: new `(r, c)` reads old `(c, r)`.
fn transpose(r: usize, c: usize) -> usize {
    rm(c, r)
}

/// Read a canonical mask through an address map — the VIEW, computed per
/// bit on demand, never stored.
fn view(mask: u64, map: fn(usize, usize) -> usize) -> u64 {
    let mut v = 0u64;
    for r in 0..2 {
        for c in 0..2 {
            if mask >> map(r, c) & 1 == 1 {
                v |= 1 << rm(r, c);
            }
        }
    }
    v
}

fn run(program: &Program, planes: &Planes<'_>) -> (u64, Value) {
    let slots = program.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(1, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, 1, slots).expect("carves");
    let mut kept = [0u64; 1];
    let out = if matches!(program.terminal, Terminal::Keep { .. }) {
        Out::Mask(&mut kept)
    } else {
        Out::None
    };
    let v = execute_into(program, planes, &Foreign::NONE, &mut scratch, out).expect("runs");
    (kept[0], v)
}

const S0: Operand = Operand::Scratch(0);
const S1: Operand = Operand::Scratch(1);
const S2: Operand = Operand::Scratch(2);

/// `v > 1` AND `v != 3` over the canonical lane; the kept mask.
fn and_program(terminal: Terminal) -> Program {
    Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 1 },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::NeI32 { lane: 0, v: 3 },
                under: None,
                dst: 1,
            },
            MaskOp::And {
                a: S0,
                b: S1,
                dst: 2,
            },
        ],
        terminal,
    )
}

#[test]
fn a_rotation_commutes_with_the_mask_algebra_so_operands_are_never_rotated() {
    let lanes = [LaneRef::I32(&CANON)];
    let planes = Planes {
        n_rows: 4,
        masks: &[],
        lanes: &lanes,
    };
    // The two predicates, each kept, in canonical coordinates — once.
    let (a, _) = run(
        &Program::new(
            vec![MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 1 },
                under: None,
                dst: 0,
            }],
            Terminal::Keep { mask: S0 },
        ),
        &planes,
    );
    let (b, _) = run(
        &Program::new(
            vec![MaskOp::Pred {
                pred: Pred::NeI32 { lane: 0, v: 3 },
                under: None,
                dst: 0,
            }],
            Terminal::Keep { mask: S0 },
        ),
        &planes,
    );
    let (ab, _) = run(&and_program(Terminal::Keep { mask: S2 }), &planes);
    assert_eq!(a, 0b1110, "v > 1 selects B C D");
    assert_eq!(b, 0b1011, "v != 3 selects A B D");
    assert_eq!(ab, 0b1010, "B D");
    for map in [rot_ccw as fn(usize, usize) -> usize, transpose] {
        // ρ(A & B) = ρ(A) & ρ(B): the view of the conjunction is the
        // conjunction of the views. No operand was ever rotated.
        assert_eq!(view(ab, map), view(a, map) & view(b, map));
        assert_eq!(view(a | b, map), view(a, map) | view(b, map));
        assert_eq!(view(a ^ b, map), view(a, map) ^ view(b, map));
    }
    // The rotated matrix is `B D / A C`, so the kept cells B, D sit at
    // (0,0) and (0,1): the view reads 0b0011 — and the answer came from the
    // canonical mask through the map, not from a rotated buffer.
    assert_eq!(view(ab, rot_ccw), 0b0011);
    // The transpose is `A C / B D`: B, D are its second row, (1,0) and (1,1).
    assert_eq!(view(ab, transpose), 0b1100);
    let (_, count) = run(&and_program(Terminal::Count { mask: S2 }), &planes);
    let (_, sum) = run(
        &and_program(Terminal::MaskedSumI32 { mask: S2, lane: 0 }),
        &planes,
    );
    // A reducer is invariant under ρ: it consumes the canonical mask
    // directly; the coordinate system never reaches it.
    assert_eq!(count, Value::Count(2));
    assert_eq!(sum, Value::SumI64(2 + 4));
    assert_eq!(view(ab, rot_ccw).count_ones(), 2);
    assert_eq!(view(ab, transpose).count_ones(), 2);
}

#[test]
fn a_column_view_is_a_strided_read_of_the_same_bytes_not_a_copy() {
    // The same four values as LE bytes, stored once.
    let vals: [u32; 4] = [1, 2, 3, 4];
    let mut bytes = [0u8; 16];
    for (i, v) in vals.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    // Canonical: row-major predicate `v == 2` on the contiguous lane.
    let mut canon = [0u64; 1];
    eq_u32_strided_to_mask(&bytes, 0, 4, 4, 2, &mut canon);
    assert_eq!(canon[0], 0b0010, "B");

    // Transpose `A C / B D`: its row r IS canonical column r — a strided
    // read at offset 4r, stride 8, count 2. Two rows, two reads, no copy.
    let mut t = 0u64;
    for r in 0..2 {
        let mut w = [0u64; 1];
        eq_u32_strided_to_mask(&bytes, 4 * r, 8, 2, 2, &mut w);
        t |= w[0] << (2 * r);
    }
    assert_eq!(
        t,
        view(canon[0], transpose),
        "strided read == canonical mask through ρ"
    );

    // Rotation `B D / A C`: its row r is canonical column 1 - r, top-down —
    // the same strided read at offset 4(1 - r).
    let mut rot = 0u64;
    for r in 0..2 {
        let mut w = [0u64; 1];
        eq_u32_strided_to_mask(&bytes, 4 * (1 - r), 8, 2, 2, &mut w);
        rot |= w[0] << (2 * r);
    }
    assert_eq!(rot, view(canon[0], rot_ccw));
    assert_eq!(rot, 0b0001, "B is now at (0,0)");
    // A count over either view equals the canonical count: the fold never
    // needed the view.
    assert_eq!(t.count_ones(), canon[0].count_ones());
    assert_eq!(rot.count_ones(), canon[0].count_ones());
}
