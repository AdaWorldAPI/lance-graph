//! D-RPF-0 (`.claude/plans/2026-10-08-resident-projection-fold-mask-v1.md`):
//! a CE64 field predicate over resident `NodeRow` bytes, with no extracted
//! lane.
//!
//! The four `CausalEdge64` words of `ValueTenant::MaterializedEdges` (32 B,
//! `u64` LE each) are read in place through two 16-byte strided windows: edges
//! 0 and 1 in the first, edges 2 and 3 in the second. Edge `k` is matched in
//! window `k / 2`, half `k % 2`, with `care` zero on the other edge's eight
//! bytes, so no window reaches outside the tenant.
//!
//! Two independent spellings of each field meet here:
//! - the predicate is built from `causal_edge::isa::Field::span()`, the ISA's
//!   field table;
//! - the oracle is the CE64 named accessors (`causal_mask`,
//!   `inference_mantissa`, `epistemic_raw5`), which never consult that table.
//!
//! Tenant offsets come from `ValueTenant::value_offset()`; no literal offset.

use causal_edge::isa::Field;
use causal_edge::CausalEdge64;
use lance_graph_contract::canonical_node::{ValueTenant, NODE_ROW_STRIDE, VALUE_SLAB_ROW_OFFSET};
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::reference::reference_execute_into;
use lance_graph_mask_risc::{
    words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, StridedRef, Terminal,
    Value,
};

/// Edges per `MaterializedEdges` tenant.
const EDGES: usize = 4;
/// 3 tiles of 512 rows plus a ragged tail.
const N: usize = 3 * 512 + 37;

/// SplitMix64: all 64 output bits vary. (A `>> 11` LCG leaves bits 53..63 —
/// the witness and epistemic fields — zero in every word, which made the
/// epistemic tests vacuous until the selectivity guard caught it.)
fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Row offset of the `MaterializedEdges` tenant.
fn edges_row_offset() -> usize {
    VALUE_SLAB_ROW_OFFSET + ValueTenant::MaterializedEdges.value_offset()
}

/// `(window index, byte offset of edge k inside its 16-byte window)`.
fn placement(k: usize) -> (usize, usize) {
    (k / 2, 8 * (k % 2))
}

/// Row-relative byte offset where window `w` starts.
fn window_offset(w: usize) -> usize {
    edges_row_offset() + 16 * w
}

/// `N` rows of random bytes: every tenant, the key and the edge block are
/// noise, so a predicate that reads anything but its cared bits shows up.
fn fixture(seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut bytes = vec![0u8; N * NODE_ROW_STRIDE];
    for chunk in bytes.as_chunks_mut::<8>().0 {
        *chunk = lcg(&mut s).to_le_bytes();
    }
    bytes
}

fn edge_at(bytes: &[u8], row: usize, k: usize) -> CausalEdge64 {
    let o = row * NODE_ROW_STRIDE + edges_row_offset() + 8 * k;
    CausalEdge64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"))
}

/// The oracle: read the field through the CE64 accessor, not the span table.
fn accessor(field: Field, e: CausalEdge64) -> u64 {
    match field {
        Field::Pearl => e.causal_mask() as u64,
        Field::Inference => (e.inference_mantissa() as u8 & 0x0F) as u64,
        Field::Epistemic => e.epistemic_raw5() as u64,
        other => panic!("no accessor oracle wired for {other:?}"),
    }
}

/// `(window, pattern, care)` for "edge `k`'s `field` equals `value`", built
/// from `field_care` so a test can hand in a deliberately wrong care.
fn pattern_with_care(
    k: usize,
    value: u64,
    field_care: u64,
    shift: u32,
) -> (usize, [u8; 16], [u8; 16]) {
    let (w, half) = placement(k);
    let mut pattern = [0u8; 16];
    let mut care = [0u8; 16];
    pattern[half..half + 8].copy_from_slice(&((value << shift) & field_care).to_le_bytes());
    care[half..half + 8].copy_from_slice(&field_care.to_le_bytes());
    (w, pattern, care)
}

fn field_pattern(k: usize, field: Field, value: u64) -> (usize, [u8; 16], [u8; 16]) {
    pattern_with_care(k, value, field.mask(), field.span().0)
}

/// The two windows as strided lanes over the same row bytes.
fn lanes(bytes: &[u8]) -> [LaneRef<'_>; 2] {
    let view = |w: usize| {
        LaneRef::Strided(StridedRef {
            bytes,
            first_offset: window_offset(w),
            stride: NODE_ROW_STRIDE,
            records: N,
        })
    };
    [view(0), view(1)]
}

/// Run `ops` with a `Keep` of the last-written slot, executor and oracle, and
/// return the mask (both paths must agree).
fn run_keep(bytes: &[u8], ops: Vec<MaskOp>, result_slot: u16) -> Vec<u64> {
    let lanes = lanes(bytes);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let p = Program::new(
        ops,
        Terminal::Keep {
            mask: Operand::Scratch(result_slot),
        },
    );
    let mut got = vec![0u64; words_for(N)];
    let mut want = vec![0u64; words_for(N)];
    let mut s = Scratch::for_program(&p, N).expect("addressable");
    let g = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::Mask(&mut got));
    let r = reference_execute_into(&p, &planes, &Foreign::NONE, Out::Mask(&mut want));
    assert_eq!(g, r, "executor vs oracle result");
    assert!(
        matches!(g, Ok(Value::Mask(_))),
        "expected a kept mask, got {g:?}"
    );
    assert_eq!(got, want, "executor vs oracle mask");
    got
}

fn single(lane: usize, pattern: [u8; 16], care: [u8; 16]) -> Vec<MaskOp> {
    vec![MaskOp::Pred {
        pred: Pred::MatchFacet16Strided {
            lane: lane as u16,
            pattern,
            care,
        },
        under: None,
        dst: 0,
    }]
}

fn bit(mask: &[u64], row: usize) -> bool {
    mask[row / 64] >> (row % 64) & 1 == 1
}

fn expected(bytes: &[u8], pred: impl Fn(CausalEdge64) -> bool, k: usize) -> Vec<u64> {
    let mut m = vec![0u64; words_for(N)];
    for row in 0..N {
        if pred(edge_at(bytes, row, k)) {
            m[row / 64] |= 1 << (row % 64);
        }
    }
    m
}

fn popcount(m: &[u64]) -> usize {
    m.iter().map(|w| w.count_ones() as usize).sum()
}

const FIELDS: [Field; 3] = [Field::Pearl, Field::Inference, Field::Epistemic];

/// FAILS IF: a window leaves the tenant, or the tenant is not four `u64`s.
#[test]
fn both_windows_stay_inside_the_materialized_edges_tenant() {
    assert_eq!(ValueTenant::MaterializedEdges.byte_len(), 8 * EDGES);
    let lo = edges_row_offset();
    let hi = lo + ValueTenant::MaterializedEdges.byte_len();
    for k in 0..EDGES {
        let (w, half) = placement(k);
        let start = window_offset(w);
        assert!(
            start >= lo && start + 16 <= hi,
            "window {w} leaves the tenant"
        );
        assert_eq!(
            start + half,
            lo + 8 * k,
            "edge {k} is not where its half points"
        );
    }
}

/// FAILS IF: the in-place predicate disagrees with the CE64 accessor for any
/// edge, field or value. Anti-vacuity: each field has at least one value that
/// admits a strict, non-empty minority of rows.
#[test]
fn equality_on_each_field_of_each_edge_equals_the_accessor() {
    let bytes = fixture(0xD5F0);
    for k in 0..EDGES {
        for field in FIELDS {
            let width = field.span().1;
            let mut selective = false;
            for value in 0..(1u64 << width) {
                let (w, p, c) = field_pattern(k, field, value);
                let got = run_keep(&bytes, single(w, p, c), 0);
                let want = expected(&bytes, |e| accessor(field, e) == value, k);
                assert_eq!(got, want, "edge {k} {field:?} == {value}");
                let n = popcount(&got);
                selective |= n > 0 && n * 3 < N;
            }
            assert!(
                selective,
                "edge {k} {field:?}: no value is a selective filter"
            );
        }
    }
}

/// FAILS IF: a care that names the wrong bits still agrees with the accessor.
/// Shifting the care up by one bit must disagree for some value.
#[test]
fn a_care_shifted_by_one_bit_disagrees_with_the_accessor() {
    let bytes = fixture(0xD5F1);
    let k = 1;
    for field in FIELDS {
        let (shift, width) = field.span();
        let mut disagreed = false;
        for value in 0..(1u64 << width) {
            let (w, p, c) = pattern_with_care(k, value, field.mask() << 1, shift + 1);
            let got = run_keep(&bytes, single(w, p, c), 0);
            let want = expected(&bytes, |e| accessor(field, e) == value, k);
            disagreed |= got != want;
        }
        assert!(
            disagreed,
            "{field:?}: a wrong-bit care was indistinguishable"
        );
    }
}

/// FAILS IF: anything outside the cared bits changes the mask. Every byte of
/// every row is rewritten except the cared field of the edge under test.
#[test]
fn bytes_outside_the_cared_field_do_not_move_the_mask() {
    let base = fixture(0xD5F2);
    let mut noise = fixture(0x7E57);
    for k in 0..EDGES {
        for field in FIELDS {
            let keep = field.mask();
            for row in 0..N {
                let o = row * NODE_ROW_STRIDE + edges_row_offset() + 8 * k;
                let a = u64::from_le_bytes(base[o..o + 8].try_into().expect("8 bytes"));
                let b = u64::from_le_bytes(noise[o..o + 8].try_into().expect("8 bytes"));
                noise[o..o + 8].copy_from_slice(&((a & keep) | (b & !keep)).to_le_bytes());
            }
            let value = 1u64;
            let (w, p, c) = field_pattern(k, field, value);
            assert_eq!(
                run_keep(&base, single(w, p, c), 0),
                run_keep(&noise, single(w, p, c), 0),
                "edge {k} {field:?}: bytes outside the field moved the mask"
            );
            assert_ne!(base, noise, "the noise fixture must differ from the base");
        }
    }
}

/// `x >= t` on an n-bit field as a disjoint union of at most n + 1 ternary
/// patterns: `x == t`, plus, for every bit of `t` that is 0, the prefix of `t`
/// above it with that bit set and everything below don't-care.
fn ge_patterns(t: u64, width: u32) -> Vec<(u64, u64)> {
    let full = (1u64 << width) - 1;
    let mut out = vec![(t, full)];
    for i in 0..width {
        if t >> i & 1 == 0 {
            let above = full & !((1u64 << (i + 1)) - 1);
            out.push(((t & above) | (1 << i), above | (1 << i)));
        }
    }
    out
}

/// FAILS IF: the threshold-as-pattern-union disagrees with `raw5 >= t` on any
/// edge, or uses more than width + 1 patterns.
#[test]
fn epistemic_threshold_as_a_pattern_union_equals_the_accessor() {
    let bytes = fixture(0xD5F3);
    let field = Field::Epistemic;
    let (shift, width) = field.span();
    for k in 0..EDGES {
        for t in 0..(1u64 << width) {
            let pats = ge_patterns(t, width);
            assert!(pats.len() <= width as usize + 1);
            let (w, _) = placement(k);
            let mut ops = Vec::new();
            for (i, &(v, c)) in pats.iter().enumerate() {
                let (_, p, care) = pattern_with_care(k, v, c << shift, shift);
                ops.push(MaskOp::Pred {
                    pred: Pred::MatchFacet16Strided {
                        lane: w as u16,
                        pattern: p,
                        care,
                    },
                    under: None,
                    dst: (i + 1) as u16,
                });
            }
            // Fold the pattern masks into slot 0 with Or.
            ops.push(MaskOp::Or {
                a: Operand::Scratch(1),
                b: Operand::Scratch(1),
                dst: 0,
            });
            for i in 2..=pats.len() {
                ops.push(MaskOp::Or {
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(i as u16),
                    dst: 0,
                });
            }
            let got = run_keep(&bytes, ops, 0);
            let want = expected(&bytes, |e| u64::from(e.epistemic_raw5()) >= t, k);
            assert_eq!(got, want, "edge {k}: raw5 >= {t}");
            if t == 0 {
                assert_eq!(popcount(&got), N, "raw5 >= 0 admits every row");
            }
        }
    }
}

/// FAILS IF: the threshold rows are not where `bit` says, i.e. the row/bit
/// convention the helpers above assume is wrong (LSB-first per word).
#[test]
fn the_mask_is_lsb_first_per_word() {
    let bytes = fixture(0xD5F4);
    let (w, p, c) = field_pattern(0, Field::Pearl, 0b101);
    let got = run_keep(&bytes, single(w, p, c), 0);
    for row in 0..N {
        assert_eq!(
            bit(&got, row),
            edge_at(&bytes, row, 0).causal_mask() as u64 == 0b101,
            "row {row}"
        );
    }
}
