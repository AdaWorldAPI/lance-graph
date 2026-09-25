//! Differential tests for the strided lane (W3 of 3): [`Pred::EqU32Strided`],
//! [`Pred::NeU32Strided`], [`Pred::MatchFacetStrided`] and
//! [`Terminal::MaskedStridedGroupSum`], executor vs the row-at-a-time oracle,
//! over a `NodeRow`-shaped (512-byte stride) fixture built from raw bytes —
//! per the crate's `LaneRef::Strided` doc: a field VIEW over unchanged
//! record bytes, never an extracted column.
//!
//! The line numbers cited below are where each signature was read while the
//! executor and oracle were being written; treat them as pointers, not pins.
//!
//! Semantics pinned (from `src/ir.rs`'s doc comments on the four items
//! above, and `src/value.rs`'s `ExecError`/`Value` variants):
//! - `EqU32Strided`/`NeU32Strided`: `u32::from_le_bytes` of the 4 bytes at
//!   `first_offset + i*stride`, compared exactly.
//! - `MatchFacetStrided`: `(b[k] ^ pattern[k]) & care[k] == 0` for all
//!   `k < 12`, at the same per-record offset.
//! - `MaskedStridedGroupSum`: for every masked row, decode `groups`
//!   consecutive little-endian UNSIGNED integers of `group_bytes` bytes each
//!   (`groups * group_bytes` bytes total, starting at the view's per-record
//!   offset) and add ALL of them into ONE running `i64` total — not one
//!   bucket per group. `Value::StridedSum(None)` is defined for a total that
//!   leaves `i64`; that arm is not exercised here (constructing an
//!   `i64`-overflowing fixture cheaply is not attempted — the doc says so
//!   rather than a fabricated test faking it, per the worker rule against
//!   inventing coverage for a case not actually driven).
//! - Validation (shared by both paths, read at `src/reference.rs`):
//!   - `check_lane` (`reference.rs:125-135`) refuses a lane that exists but
//!     is not `LaneKind::Strided` with `ExecError::LaneKind`.
//!   - the generic per-lane length check (`reference.rs:342-350`) refuses
//!     `lane.len() != n_rows` with `ExecError::LenMismatch{what:"lane",..}` —
//!     `LaneRef::Strided::len()` returns `records` (`src/ir.rs:60`), so this
//!     is exactly "records != n_rows".
//!   - `check_strided` (`reference.rs:169-190`) refuses a view whose last
//!     record's field-of-width-`width` end exceeds `bytes.len()` with
//!     `ExecError::StridedOutOfBounds{lane, need, have}`; `need` is
//!     `first_offset + (records-1)*stride + width` (saturating to
//!     `usize::MAX` on overflow).
//!   - `MaskedStridedGroupSum`'s own arm (`reference.rs:493-505`) refuses
//!     `group_bytes` outside `1..=4` with `ExecError::StridedGroupWidth`
//!     BEFORE consulting `check_strided`.

use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
// read at src/exec.rs:641-654 (`execute`), :906-914 (`execute_into`),
// :949-956 (`execute_extent`), :170-204/:194 (`Scratch::for_program`).
use lance_graph_mask_risc::reference::{reference_execute_into, reference_scratch};
// read at src/reference.rs:849-855 (`reference_execute` — unused here, its
// sibling below takes the `Foreign` + `Out` this file needs), :862-877
// (`reference_execute_into`), :1111-1113/:1116-1120 (`reference_scratch` /
// `reference_scratch_with_foreign`).
use lance_graph_mask_risc::{
    words_for, ExecError, Foreign, LaneKind, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    StridedRef, Terminal, Value,
};
// `words_for`, `ExecError`, `Foreign`, `LaneKind`, `LaneRef`, `MaskOp`,
// `Operand`, `Out`, `Planes`, `Pred`, `Program`, `StridedRef`, `Terminal`,
// `Value` are all re-exported at `src/lib.rs:124-140`.

const RECORD_STRIDE: usize = 512;
/// `TILE_WORDS` (`src/exec.rs:96`) * 64 — one execution tile in rows. Used to
/// pick a fixture spanning "3 tiles + a ragged tail".
const TILE_ROWS: usize = 8 * 64;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// A `NodeRow`-shaped buffer: `n` records of `RECORD_STRIDE` bytes each, one
/// `Vec<u8>`. classid `u32` LE lives at `+0`, a 12-byte facet at `+4`, drawn
/// so both an equality and a partial-care match select a non-trivial subset.
/// An independent `i32` lane (contiguous, ordinary) is carried alongside for
/// the mixed-lane tests.
struct Fixture {
    n: usize,
    bytes: Vec<u8>,
    i32s: Vec<i32>,
}

const CLASSIDS: [u32; 4] = [0x1000_0000, 0x2000_0001, 0x3000_0002, 0x4000_0003];

impl Fixture {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = seed ^ (n as u64);
        let mut bytes = vec![0u8; n * RECORD_STRIDE];
        for i in 0..n {
            let base = i * RECORD_STRIDE;
            for k in 0..RECORD_STRIDE {
                bytes[base + k] = (lcg(&mut s) & 0xFF) as u8;
            }
            let cid = CLASSIDS[(lcg(&mut s) as usize) % CLASSIDS.len()];
            bytes[base..base + 4].copy_from_slice(&cid.to_le_bytes());
            // leave +4..+16 (the facet) at its random draw — the fixture's
            // partial-care selectivity comes from that randomness, not from
            // a planted value, so it must be checked non-trivial per test.
        }
        let i32s: Vec<i32> = (0..n).map(|_| (lcg(&mut s) % 2000) as i32 - 1000).collect();
        Self { n, bytes, i32s }
    }

    fn classid_view(&self) -> StridedRef<'_> {
        StridedRef {
            bytes: &self.bytes,
            first_offset: 0,
            stride: RECORD_STRIDE,
            records: self.n,
        }
    }

    fn facet_view(&self) -> StridedRef<'_> {
        StridedRef {
            bytes: &self.bytes,
            first_offset: 4,
            stride: RECORD_STRIDE,
            records: self.n,
        }
    }

    fn classid_at(&self, row: usize) -> u32 {
        let base = row * RECORD_STRIDE;
        u32::from_le_bytes(self.bytes[base..base + 4].try_into().unwrap())
    }

    fn facet_at(&self, row: usize) -> [u8; 12] {
        let base = row * RECORD_STRIDE + 4;
        self.bytes[base..base + 12].try_into().unwrap()
    }

    /// The classid column, extracted into an ordinary contiguous `Vec<u32>` —
    /// used ONLY by the cross-path equivalence test, which must build it from
    /// the SAME bytes the strided view reads.
    fn classid_column(&self) -> Vec<u32> {
        (0..self.n).map(|r| self.classid_at(r)).collect()
    }
}

fn words(n: usize) -> usize {
    words_for(n)
}

/// A fresh, whole-population `Scratch` for `p` over `f`.
fn scratch(p: &Program, f: &Fixture) -> Scratch<'static> {
    Scratch::for_program(p, f.n).expect("addressable")
}

/// Run `p` (whole population, `Out::None`) on both paths and assert they
/// agree. Returns the shared `Value` for the caller's own assertions.
fn run_none(f: &Fixture, planes: &Planes<'_>, p: &Program, name: &str) -> Value {
    let mut s = scratch(p, f);
    let got = execute_into(p, planes, &Foreign::NONE, &mut s, Out::None);
    let want = reference_execute_into(p, planes, &Foreign::NONE, Out::None);
    assert_eq!(got, want, "{name} @ n={}: executor vs oracle", f.n);
    want.unwrap_or_else(|e| panic!("{name} @ n={}: expected Ok, got {e:?}", f.n))
}

/// Independent (of both paths) count of rows whose classid equals `target`,
/// reading bytes directly.
fn expected_eq_count(f: &Fixture, target: u32) -> usize {
    (0..f.n).filter(|&r| f.classid_at(r) == target).count()
}

fn expected_facet_count(f: &Fixture, pattern: [u8; 12], care: [u8; 12]) -> usize {
    (0..f.n)
        .filter(|&r| {
            let b = f.facet_at(r);
            (0..12).all(|k| (b[k] ^ pattern[k]) & care[k] == 0)
        })
        .count()
}

/// Sum of ALL `groups` little-endian unsigned `group_bytes`-wide fields of
/// every SELECTED record's facet, widened to `i64` — the independent
/// hand-computation `MaskedStridedGroupSum` is checked against.
fn expected_group_sum(
    f: &Fixture,
    groups: u8,
    group_bytes: u8,
    selected: impl Fn(usize) -> bool,
) -> i64 {
    let mut total: i64 = 0;
    for r in 0..f.n {
        if !selected(r) {
            continue;
        }
        let facet = f.facet_at(r);
        for g in 0..usize::from(groups) {
            let off = g * usize::from(group_bytes);
            let mut v: u64 = 0;
            for k in 0..usize::from(group_bytes) {
                v |= u64::from(facet[off + k]) << (8 * k);
            }
            total += v as i64;
        }
    }
    total
}

const ROWS: [usize; 3] = [0, 70, 3 * TILE_ROWS + 37];

// ─────────────────────────────────────────────────────────────────────────
// 1. EqU32Strided / NeU32Strided on the classid view.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the executor disagrees with the oracle on either predicate, or
/// the fixture is vacuous (0 < eq_count < n for n large enough to make that
/// meaningful), or `eq_count + ne_count != n` (the two-valued complement law
/// `lib.rs` documents — no NULL in this substrate).
#[test]
fn eq_and_ne_u32_strided_partition_the_population() {
    for n in ROWS {
        let f = Fixture::new(n, 101);
        let target = CLASSIDS[0];
        let want = expected_eq_count(&f, target);
        if n >= 8 {
            assert!(
                want > 0 && want < n,
                "n={n}: classid=={target:#x} is not a selective fixture ({want}/{n})"
            );
        }
        let lanes = [LaneRef::Strided(f.classid_view())];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let eq_p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let ne_p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::NeU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let eq_v = run_none(&f, &planes, &eq_p, "EqU32Strided");
        let ne_v = run_none(&f, &planes, &ne_p, "NeU32Strided");
        assert_eq!(
            eq_v,
            Value::Count(want),
            "n={n}: eq count vs independent computation"
        );
        let ne_c = match ne_v {
            Value::Count(c) => c,
            other => panic!("n={n}: expected Count, got {other:?}"),
        };
        assert_eq!(want + ne_c, n, "n={n}: eq_count + ne_count must equal n");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 2. Cross-path equivalence: the strided eq equals the same predicate over
//    an extracted contiguous `LaneRef::U32` column built from the SAME
//    bytes.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: reading the classid field through the strided view produces a
/// different population than reading the identical bytes through an
/// ordinary contiguous `U32` lane.
#[test]
fn strided_eq_matches_the_extracted_contiguous_column() {
    for n in ROWS {
        let f = Fixture::new(n, 103);
        let target = CLASSIDS[1];
        let column = f.classid_column();
        let strided_lanes = [LaneRef::Strided(f.classid_view())];
        let contig_lanes = [LaneRef::U32(&column)];
        let strided_planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &strided_lanes,
        };
        let contig_planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &contig_lanes,
        };
        let strided_p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let contig_p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32 { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let strided_v = run_none(&f, &strided_planes, &strided_p, "strided eq");
        // The contiguous side is diffed too (its own executor/oracle pair),
        // then the two VALUES are compared cross-path.
        let contig_v = run_none(&f, &contig_planes, &contig_p, "contiguous eq");
        assert_eq!(
            strided_v, contig_v,
            "n={n}: strided view and extracted column must select the same rows"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 3. MatchFacetStrided with a partial care mask.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the executor disagrees with the oracle, the partial-care match
/// is vacuous, or an all-zero care mask fails to match every row (it must —
/// `care == 0` means no bit is inspected).
#[test]
fn match_facet_strided_partial_and_all_wildcard_care() {
    for n in ROWS {
        let f = Fixture::new(n, 107);
        // A care mask that inspects roughly half the bits of the first 6
        // bytes and none of the rest — a genuine PARTIAL match, not full
        // equality and not a no-op.
        let mut care = [0u8; 12];
        for c in care.iter_mut().take(6) {
            *c = 0x0F;
        }
        // Any concrete byte pattern; `n == 0` has no row to draw one from, so
        // fall back to all-zero (unreachable by any predicate at n == 0
        // anyway — the loop below is empty).
        let pattern = if n == 0 { [0u8; 12] } else { f.facet_at(0) };
        let want = expected_facet_count(&f, pattern, care);
        if n >= 8 {
            assert!(
                want < n,
                "n={n}: a 48-bit-wide partial care that matches every row is not informative"
            );
        }
        let lanes = [LaneRef::Strided(f.facet_view())];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::MatchFacetStrided {
                    lane: 0,
                    pattern,
                    care,
                },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let v = run_none(&f, &planes, &p, "MatchFacetStrided partial");
        assert_eq!(v, Value::Count(want), "n={n}: partial-care count");

        // Silence twin: care == 0 matches every row.
        let all_wild = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::MatchFacetStrided {
                    lane: 0,
                    pattern,
                    care: [0u8; 12],
                },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let v_all = run_none(&f, &planes, &all_wild, "MatchFacetStrided all-wild");
        assert_eq!(
            v_all,
            Value::Count(n),
            "n={n}: care==0 must match every row"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 4. Gated: a scratch mask from an I32 predicate, compared against a manual
//    AND of two independently-materialized masks.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the gated program's count differs from the executor, the
/// oracle, or a manual word-wise AND of the two ungated masks — proving the
/// gate is a real intersection, not decoration.
#[test]
fn gated_strided_predicate_equals_the_and_of_the_two_separate_masks() {
    for n in [70usize, 3 * TILE_ROWS + 37] {
        let f = Fixture::new(n, 109);
        let target = CLASSIDS[2];
        let lanes = [LaneRef::Strided(f.classid_view()), LaneRef::I32(&f.i32s)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let gate_only = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::GtI32 { lane: 1, t: 0 },
                under: None,
                dst: 0,
            }],
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        let eq_only = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        // Materialize both masks through the ORACLE's own scratch mirror
        // (`reference_scratch`), independent of the executor's bit packing.
        let gate_bits = reference_scratch(&gate_only, &planes).expect("gate scratch")[0].clone();
        let eq_bits = reference_scratch(&eq_only, &planes).expect("eq scratch")[0].clone();
        assert_eq!(gate_bits.len(), words(n));
        let and_count: u32 = gate_bits
            .iter()
            .zip(eq_bits.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum();

        let gated = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 1, t: 0 },
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: Pred::EqU32Strided { lane: 0, v: target },
                    under: Some(Operand::Scratch(0)),
                    dst: 1,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(1),
            },
        );
        let v = run_none(&f, &planes, &gated, "gated EqU32Strided");
        assert_eq!(
            v,
            Value::Count(and_count as usize),
            "n={n}: gated count must equal the manual AND of the two masks"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 5. Two strided views over the SAME Vec, plus an I32 lane, in one program.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the executor disagrees with the oracle on a program that reads
/// the classid view at `+0` and the facet view at `+4` of the SAME backing
/// buffer alongside an ordinary I32 lane, or the composed relation is
/// vacuous.
#[test]
fn two_strided_views_over_one_buffer_plus_an_i32_lane() {
    for n in [70usize, 3 * TILE_ROWS + 37] {
        let f = Fixture::new(n, 113);
        let target = CLASSIDS[3];
        let mut care = [0u8; 12];
        care[0] = 0xFF;
        let pattern = f.facet_at(0);
        let lanes = [
            LaneRef::Strided(f.classid_view()),
            LaneRef::Strided(f.facet_view()),
            LaneRef::I32(&f.i32s),
        ];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let want = (0..n)
            .filter(|&r| {
                f.classid_at(r) == target
                    && (f.facet_at(r)[0] ^ pattern[0]) & care[0] == 0
                    && f.i32s[r] > -1000
            })
            .count();
        let p = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::EqU32Strided { lane: 0, v: target },
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: Pred::MatchFacetStrided {
                        lane: 1,
                        pattern,
                        care,
                    },
                    under: Some(Operand::Scratch(0)),
                    dst: 1,
                },
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 2, t: -1000 },
                    under: Some(Operand::Scratch(1)),
                    dst: 2,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(2),
            },
        );
        let v = run_none(&f, &planes, &p, "two strided views + i32");
        assert_eq!(v, Value::Count(want), "n={n}: composed relation");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 6. MaskedStridedGroupSum over the facet, as 6×u16 and as 3×u32.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the executor disagrees with the oracle, or either disagrees
/// with a hand-computed sum, for the facet read as `6 * u16` or `3 * u32`.
/// Also checked over `execute_extent`'s WHOLE-population extent (`0..n`),
/// which `execute_into` itself is defined to be
/// (`execute_extent(.., 0..n_rows)` per `src/exec.rs:906-914`), and over a
/// PARTIAL extent (the middle third), which must equal the hand-computed sum
/// over the selected rows inside that extent only — a sum merges by
/// addition, so the partial extent is admitted.
#[test]
fn masked_strided_group_sum_as_u16x6_and_u32x3() {
    for n in [70usize, 3 * TILE_ROWS + 37] {
        let f = Fixture::new(n, 127);
        // Select roughly half the rows via an ordinary I32 predicate, so the
        // sum is over a non-trivial, non-total subset.
        let lanes = [LaneRef::Strided(f.facet_view()), LaneRef::I32(&f.i32s)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let selected = |r: usize| f.i32s[r] >= 0;
        let selected_count = (0..n).filter(|&r| selected(r)).count();
        if n >= 8 {
            assert!(
                selected_count > 0 && selected_count < n,
                "n={n}: the selecting predicate is not informative"
            );
        }
        for &(groups, group_bytes) in &[(6u8, 2u8), (3u8, 4u8)] {
            let want = expected_group_sum(&f, groups, group_bytes, selected);
            let p = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::GeI32 { lane: 1, t: 0 },
                    under: None,
                    dst: 0,
                }],
                Terminal::MaskedStridedGroupSum {
                    mask: Operand::Scratch(0),
                    lane: 0,
                    groups,
                    group_bytes,
                },
            );
            let mut s = scratch(&p, &f);
            let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
            let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
            assert_eq!(
                got, oracle,
                "n={n} groups={groups} group_bytes={group_bytes}: executor vs oracle"
            );
            assert_eq!(
                got,
                Ok(Value::StridedSum(Some(want))),
                "n={n} groups={groups} group_bytes={group_bytes}: vs hand-computed sum"
            );

            // The whole-population extent: `execute_extent(.., 0..n)` IS
            // `execute_into` by construction.
            let mut s2 = scratch(&p, &f);
            let got_ext = execute_extent(&p, &planes, &Foreign::NONE, &mut s2, Out::None, 0..n);
            assert_eq!(
                got_ext, got,
                "n={n} groups={groups} group_bytes={group_bytes}: whole-extent vs execute_into"
            );

            // A partial extent: only rows in `lo..hi` contribute.
            let (lo, hi) = (n / 3, 2 * n / 3);
            let want_part = expected_group_sum(&f, groups, group_bytes, |r| {
                (lo..hi).contains(&r) && selected(r)
            });
            assert_ne!(want_part, want, "n={n}: the extent must exclude something");
            let mut s3 = scratch(&p, &f);
            let got_part = execute_extent(&p, &planes, &Foreign::NONE, &mut s3, Out::None, lo..hi);
            assert_eq!(
                got_part,
                Ok(Value::StridedSum(Some(want_part))),
                "n={n} groups={groups} group_bytes={group_bytes}: partial extent {lo}..{hi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 7. Falsifier: the strided read is IN PLACE — mutating one selected
//    record's classid byte changes the count by exactly one.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: the strided view was copied rather than read in place — a copy
/// would not observe the mutation, and the count would not move.
#[test]
fn mutating_one_record_in_place_changes_the_count_by_exactly_one() {
    let n = 3 * TILE_ROWS + 37;
    let mut f = Fixture::new(n, 131);
    let target = CLASSIDS[0];
    let before = expected_eq_count(&f, target);
    // Find a row NOT currently equal to `target` and flip its classid to it —
    // guaranteed to exist since CLASSIDS has 4 distinct values and `before <
    // n` for a real fixture (asserted below).
    assert!(before < n, "fixture must contain a non-target row");
    let flip_row = (0..n).find(|&r| f.classid_at(r) != target).unwrap();
    {
        let lanes = [LaneRef::Strided(f.classid_view())];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let v = run_none(&f, &planes, &p, "before mutation");
        assert_eq!(v, Value::Count(before));
    }
    let base = flip_row * RECORD_STRIDE;
    f.bytes[base..base + 4].copy_from_slice(&target.to_le_bytes());
    let after = expected_eq_count(&f, target);
    assert_eq!(after, before + 1);
    {
        let lanes = [LaneRef::Strided(f.classid_view())];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: target },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let v = run_none(&f, &planes, &p, "after mutation");
        assert_eq!(
            v,
            Value::Count(before + 1),
            "the count must move by exactly one"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 8. Refusals, identical from both paths — plus the silence twin.
// ─────────────────────────────────────────────────────────────────────────

/// FAILS IF: an out-of-bounds strided view, a records/n_rows mismatch, a
/// strided predicate on a non-strided lane, or an out-of-range
/// `group_bytes` is accepted by either path, refused with a different
/// error, or refused identically to a DIFFERENT wrong reason than the exact
/// one named; also checks the positive twin (an exactly-sized buffer is
/// accepted).
#[test]
fn strided_refusals_match_identically_and_the_exact_length_is_accepted() {
    let n = 70usize;
    let f = Fixture::new(n, 137);

    // (a) bytes one short of what the classid view over `n` records needs.
    {
        let need = (n - 1) * RECORD_STRIDE + 4; // first_offset=0, width=4
        let short = &f.bytes[..f.bytes.len().min(need) - 1];
        assert!(
            short.len() < need,
            "fixture bug: `short` is not actually short"
        );
        let view = StridedRef {
            bytes: short,
            first_offset: 0,
            stride: RECORD_STRIDE,
            records: n,
        };
        let lanes = [LaneRef::Strided(view)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: 0 },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let want = Err(ExecError::StridedOutOfBounds {
            lane: 0,
            need,
            have: short.len(),
        });
        let mut s = scratch(&p, &f);
        let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
        let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
        assert_eq!(got, want, "executor: bytes one short");
        assert_eq!(oracle, want, "oracle: bytes one short");
    }

    // (b) records = n - 1 while n_rows = n: LenMismatch on the lane length.
    {
        let view = StridedRef {
            bytes: &f.bytes,
            first_offset: 0,
            stride: RECORD_STRIDE,
            records: n - 1,
        };
        let lanes = [LaneRef::Strided(view)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: 0 },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let want = Err(ExecError::LenMismatch {
            what: "lane",
            expected: n,
            found: n - 1,
        });
        let mut s = scratch(&p, &f);
        let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
        let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
        assert_eq!(got, want, "executor: records != n_rows");
        assert_eq!(oracle, want, "oracle: records != n_rows");
    }

    // (c) EqU32Strided naming a plain U32 lane: LaneKind.
    {
        let column = f.classid_column();
        let lanes = [LaneRef::U32(&column)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided { lane: 0, v: 0 },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let want = Err(ExecError::LaneKind {
            lane: 0,
            expected: LaneKind::Strided,
            found: LaneKind::U32,
        });
        let mut s = scratch(&p, &f);
        let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
        let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
        assert_eq!(got, want, "executor: EqU32Strided on a U32 lane");
        assert_eq!(oracle, want, "oracle: EqU32Strided on a U32 lane");
    }

    // (d) group_bytes 0 and 5: StridedGroupWidth, for both paths.
    for &group_bytes in &[0u8, 5u8] {
        let lanes = [LaneRef::Strided(f.facet_view())];
        let p = Program::new(
            vec![],
            Terminal::MaskedStridedGroupSum {
                mask: Operand::Plane(0),
                lane: 0,
                groups: 1,
                group_bytes,
            },
        );
        // `validate`'s terminal arm (`reference.rs:493-505`) checks
        // `check_operand`/`written_slots.readable` on `mask` BEFORE the
        // `group_bytes` range check, so `Plane(0)` must name a REAL plane —
        // otherwise the case would fire `PlaneOutOfRange` instead of the
        // `StridedGroupWidth` this test means to isolate.
        let real_plane = vec![0u64; words(n)];
        let masks_store: [&[u64]; 1] = [&real_plane];
        let planes = Planes {
            n_rows: n,
            masks: &masks_store,
            lanes: &lanes,
        };
        let want = Err(ExecError::StridedGroupWidth { group_bytes });
        let mut s = scratch(&p, &f);
        let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
        let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
        assert_eq!(got, want, "executor: group_bytes={group_bytes}");
        assert_eq!(oracle, want, "oracle: group_bytes={group_bytes}");
    }

    // Silence twin: an EXACTLY-sized buffer is accepted (no refusal at all).
    {
        let need = (n - 1) * RECORD_STRIDE + 4;
        let exact = &f.bytes[..need];
        let view = StridedRef {
            bytes: exact,
            first_offset: 0,
            stride: RECORD_STRIDE,
            records: n,
        };
        let lanes = [LaneRef::Strided(view)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32Strided {
                    lane: 0,
                    v: CLASSIDS[0],
                },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let mut s = scratch(&p, &f);
        let got = execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None);
        let oracle = reference_execute_into(&p, &planes, &Foreign::NONE, Out::None);
        assert_eq!(got, oracle, "exact-length buffer: executor vs oracle");
        assert!(
            got.is_ok(),
            "exact-length buffer must be accepted, got {got:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 9. Ternlog over three strided predicates: map → fold through one table.
// ─────────────────────────────────────────────────────────────────────────

/// `table[imm](a, b, c)` for one row, Intel VPTERNLOG index convention
/// `(a << 2) | (b << 1) | c` — the independent oracle for the combination.
fn ternlog_bit(imm: u8, a: bool, b: bool, c: bool) -> bool {
    let idx = (usize::from(a) << 2) | (usize::from(b) << 1) | usize::from(c);
    (imm >> idx) & 1 == 1
}

/// FAILS IF: three masks produced by strided field views (two over the
/// classid at `+0`, one over the facet at `+4`, all borrowing ONE buffer) do
/// not combine through `MaskOp::Ternlog` into the count a row-by-row
/// evaluation of the same truth table gives — for an even table (majority,
/// `0xE8`) and an odd one (its negation, `0x17`), whose tail bits the
/// executor must clear before `Count` reads them — or the result is vacuous.
#[test]
fn ternlog_combines_three_strided_views_into_one_count() {
    for n in [70usize, 3 * TILE_ROWS + 37] {
        let f = Fixture::new(n, 139);
        let mut care = [0u8; 12];
        care[0] = 0x80;
        let pattern = [0u8; 12];
        let lanes = [
            LaneRef::Strided(f.classid_view()),
            LaneRef::Strided(f.facet_view()),
        ];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let a = |r: usize| f.classid_at(r) == CLASSIDS[0];
        let b = |r: usize| (f.facet_at(r)[0] & 0x80) == 0;
        let c = |r: usize| f.classid_at(r) != CLASSIDS[1];
        for imm in [0xE8u8, 0x17] {
            let want = (0..n)
                .filter(|&r| ternlog_bit(imm, a(r), b(r), c(r)))
                .count();
            assert!(
                want > 0 && want < n,
                "n={n} imm={imm:#04x}: vacuous table result"
            );
            let p = Program::new(
                vec![
                    MaskOp::Pred {
                        pred: Pred::EqU32Strided {
                            lane: 0,
                            v: CLASSIDS[0],
                        },
                        under: None,
                        dst: 0,
                    },
                    MaskOp::Pred {
                        pred: Pred::MatchFacetStrided {
                            lane: 1,
                            pattern,
                            care,
                        },
                        under: None,
                        dst: 1,
                    },
                    MaskOp::Pred {
                        pred: Pred::NeU32Strided {
                            lane: 0,
                            v: CLASSIDS[1],
                        },
                        under: None,
                        dst: 2,
                    },
                    MaskOp::Ternlog {
                        imm,
                        a: Operand::Scratch(0),
                        b: Operand::Scratch(1),
                        c: Operand::Scratch(2),
                        dst: 3,
                    },
                ],
                Terminal::Count {
                    mask: Operand::Scratch(3),
                },
            );
            let v = run_none(&f, &planes, &p, "ternlog over strided views");
            assert_eq!(v, Value::Count(want), "n={n} imm={imm:#04x}");
        }
    }
}
