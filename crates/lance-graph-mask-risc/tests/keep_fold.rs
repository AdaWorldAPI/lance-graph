//! `Program::fused_keep`: the same Boolean-chain collapse `fused_ternlog`
//! folds to `Count`/`Any` (at most three resident planes, one 8-bit ternlog
//! table), but consumed by `Terminal::Keep` and written straight into the
//! caller's `Out::Mask` in ONE `ndarray::simd::mask_ternlog` pass — no
//! scratch slot carved, no membership word written, unless the caller
//! demands the result some other way (`Out::None`), in which case the
//! ordinary tiled path runs instead.
//!
//! Every case below checks the fused path against a TILED TWIN of the exact
//! same relation — the same ops, with an extra copy forced into a slot at
//! [`FUSED_SLOT_CAP`], which `fused_keep`'s fixed-size validation bitmap
//! cannot mark, so the twin runs the ordinary scratch-writing path — and
//! several cases also check against the crate's independent row-at-a-time
//! oracle (`reference_execute_into` / `reference_scratch`).

use lance_graph_mask_risc::{
    execute_extent, reference_execute_into, reference_scratch, scratch_words_for, words_for,
    Foreign, Lowering, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal, Value,
    FUSED_SLOT_CAP,
};

/// A tiny, fast, deterministic PRNG — no new dev-dependency needed for it.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A resident plane of `n` rows, `~1/modulus` density, tail bits past `n`
/// left at their initial zero (only rows `0..n` are ever written).
fn random_plane(n: usize, seed: &mut u64, modulus: u64) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in 0..n {
        if splitmix64(seed).is_multiple_of(modulus) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

/// A garbage fill for a caller's `Out::Mask` buffer that is NOT all-zero and
/// NOT uniform across words, so a write that misses a word, or bleeds past
/// the extent it was asked for, is visible instead of accidentally matching.
fn garbage(words: usize) -> Vec<u64> {
    const PATTERN: u64 = 0xA5A5_5A5A_F00F_0FF0;
    (0..words)
        .map(|w| PATTERN.rotate_left((w % 64) as u32))
        .collect()
}

/// A random Boolean chain over at most three planes (`Operand::Plane(0..3)`),
/// `len` ops of `And`/`Or`/`Xor`/`AndNot`/`Not`/`Ternlog` with slot reuse and
/// derived operands — the same generator shape `tests/program_collapse.rs`
/// uses for its `fused_ternlog` random-chain coverage, reused here for
/// `fused_keep`. Returns the ops and the LAST WRITTEN slot (what the `Keep`
/// terminal of the test below names).
fn random_chain(seed: &mut u64, len: usize) -> (Vec<MaskOp>, u16) {
    let mut written: Vec<u16> = Vec::new();
    let mut ops = Vec::new();
    let pick = |seed: &mut u64, written: &Vec<u16>| {
        if !written.is_empty() && splitmix64(seed).is_multiple_of(2) {
            Operand::Scratch(written[(splitmix64(seed) as usize) % written.len()])
        } else {
            Operand::Plane((splitmix64(seed) % 3) as u16)
        }
    };
    let mut last = 0u16;
    for _ in 0..len {
        let (x, y, z) = (
            pick(seed, &written),
            pick(seed, &written),
            pick(seed, &written),
        );
        let dst = (splitmix64(seed) % 6) as u16;
        ops.push(match splitmix64(seed) % 6 {
            0 => MaskOp::And { a: x, b: y, dst },
            1 => MaskOp::Or { a: x, b: y, dst },
            2 => MaskOp::Xor { a: x, b: y, dst },
            3 => MaskOp::AndNot { a: x, b: y, dst },
            4 => MaskOp::Not { a: x, dst },
            _ => MaskOp::Ternlog {
                imm: (splitmix64(seed) & 0xFF) as u8,
                a: x,
                b: y,
                c: z,
                dst,
            },
        });
        if !written.contains(&dst) {
            written.push(dst);
        }
        last = dst;
    }
    (ops, last)
}

/// The tiled twin of a `fused_keep`-eligible chain: the SAME ops, plus one
/// extra self-copy (`x = x | x`) landing in a slot at [`FUSED_SLOT_CAP`] —
/// past the fixed-size on-stack bitmap `Program::fused_keep`'s symbolic
/// interpreter validates against, so this program computes the identical
/// relation but is refused by `fused_keep` and runs the ordinary
/// scratch-writing tiled path instead.
fn force_tiled(ops: &[MaskOp], last: u16) -> Program {
    let mut ops = ops.to_vec();
    let far = FUSED_SLOT_CAP as u16;
    ops.push(MaskOp::Or {
        a: Operand::Scratch(last),
        b: Operand::Scratch(last),
        dst: far,
    });
    Program::new(
        ops,
        Terminal::Keep {
            mask: Operand::Scratch(far),
        },
    )
}

/// Partial and whole extents over `[0, n)`: the whole population, empty
/// ranges at both ends and (for larger `n`) an interior point, `[0, k)` for
/// several `k` both word-aligned and not, `[k, n)` with an unaligned `k` (the
/// `hi == n` case that also exercises the population's own tail word), and
/// (for a large enough `n`) single-word ranges at an aligned and an
/// unaligned start.
fn extents(n: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(0, n), (0, 0), (n, n)];
    for &k in &[1usize, 5, 63, 64, 65, 127, 128, 129, 191, 193] {
        if k < n {
            v.push((0, k));
            v.push((k, n));
        }
    }
    if n >= 4 {
        v.push((n / 4, 3 * n / 4));
    }
    if n > 130 {
        v.push((64, 65));
        v.push((70, 71));
        v.push((n / 2, n / 2));
    }
    v.retain(|&(lo, hi)| lo <= hi && hi <= n);
    v.sort_unstable();
    v.dedup();
    v
}

/// FAILS IF: for any random Boolean chain (1..=6 ops, at most three resident
/// planes) over any of the listed row counts and any extent — the whole
/// population, an unaligned `lo`, an unaligned `hi < n_rows`, a single word,
/// an empty (`lo == hi`) range, or `hi == n_rows` with an unaligned `lo` —
/// the fused `Keep` write into a pre-poisoned `Out::Mask` buffer differs, in
/// even one word, from the tiled twin's write into an identically
/// pre-poisoned buffer of its own. Also fails if the recogniser admits a
/// chain it should not (`fused_keep().is_none()` on the original) or
/// declines to force the twin onto the tiled path
/// (`fused_keep().is_some()` on the twin), or if either lowering disagrees
/// with `compile().lowering()`.
#[test]
fn fused_keep_matches_tiled_keep_over_random_chains_and_extents() {
    let ns = [1usize, 63, 64, 65, 200, 1000, 64 * 300 + 17];
    let mut top_seed = 0xC0FF_EE15_u64;
    for &n in &ns {
        let mut seed = splitmix64(&mut top_seed) ^ n as u64;
        let (m0, m1, m2) = (
            random_plane(n, &mut seed, 2),
            random_plane(n, &mut seed, 3),
            random_plane(n, &mut seed, 5),
        );
        let masks: [&[u64]; 3] = [&m0, &m1, &m2];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let words = words_for(n);
        for _ in 0..25 {
            let len = 1 + (splitmix64(&mut seed) % 6) as usize;
            let (ops, last) = random_chain(&mut seed, len);
            let p = Program::new(
                ops.clone(),
                Terminal::Keep {
                    mask: Operand::Scratch(last),
                },
            );
            assert!(
                p.fused_keep().is_some(),
                "n={n} len={len}: a plain Boolean chain over <=3 planes must fuse"
            );
            assert!(
                matches!(p.compile().lowering(), Lowering::TernlogKeep(_)),
                "n={n} len={len}: lowering must be TernlogKeep"
            );

            let tp = force_tiled(&ops, last);
            assert!(
                tp.fused_keep().is_none(),
                "n={n} len={len}: the forced-tiled twin must NOT fuse"
            );
            assert!(
                matches!(tp.compile().lowering(), Lowering::Tiled),
                "n={n} len={len}: the forced-tiled twin's lowering must be Tiled"
            );

            for (lo, hi) in extents(n) {
                let mut out_fused = garbage(words);
                let mut out_tiled = out_fused.clone();

                // The fused path needs no scratch at all under `Out::Mask`.
                let mut sc_fused = Scratch::new(0, 0);
                let v_fused = execute_extent(
                    &p,
                    &planes,
                    &Foreign::NONE,
                    &mut sc_fused,
                    Out::Mask(&mut out_fused),
                    lo..hi,
                )
                .unwrap_or_else(|e| panic!("fused keep n={n} len={len} [{lo},{hi}): {e:?}"));

                let mut sc_tiled = Scratch::for_program(&tp, n).expect("tiled scratch");
                let v_tiled = execute_extent(
                    &tp,
                    &planes,
                    &Foreign::NONE,
                    &mut sc_tiled,
                    Out::Mask(&mut out_tiled),
                    lo..hi,
                )
                .unwrap_or_else(|e| panic!("tiled keep n={n} len={len} [{lo},{hi}): {e:?}"));

                assert_eq!(
                    out_fused, out_tiled,
                    "n={n} len={len} [{lo},{hi}): fused vs tiled Out::Mask mismatch"
                );
                // The two programs' `Keep` terminals deliberately name
                // DIFFERENT slots (that is what forces the twin onto the
                // tiled path), so `Value::Mask`'s wrapped `Operand` differs
                // by construction; the shape (both `Value::Mask`) still
                // must agree, and the byte comparison above is the load-
                // bearing check.
                assert!(matches!(v_fused, Value::Mask(_)));
                assert!(matches!(v_tiled, Value::Mask(_)));
            }
        }
    }
}

/// FAILS IF: over the whole population, the fused `Keep` write disagrees
/// with the crate's independent row-at-a-time oracle (`reference_execute_into`)
/// for any of a handful of ternlog immediates over three planes, including
/// an odd one (whose dead tail bits the oracle and the fused path must both
/// leave clear).
#[test]
fn fused_keep_matches_reference_execute() {
    let mut seed = 0xFEED_u64;
    for n in [1usize, 63, 64, 65, 200, 1000] {
        let (m0, m1, m2) = (
            random_plane(n, &mut seed, 2),
            random_plane(n, &mut seed, 3),
            random_plane(n, &mut seed, 5),
        );
        let masks: [&[u64]; 3] = [&m0, &m1, &m2];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let words = words_for(n);
        for imm in [0x96u8, 0xE4, 0x01, 0xFE, 0x69, 0x00, 0xFF] {
            let p = Program::new(
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
            assert!(p.fused_keep().is_some(), "n={n} imm={imm:#04x}");

            let mut want = vec![u64::MAX; words];
            reference_execute_into(&p, &planes, &Foreign::NONE, Out::Mask(&mut want))
                .unwrap_or_else(|e| panic!("oracle n={n} imm={imm:#04x}: {e:?}"));

            let mut got = garbage(words);
            let mut sc = Scratch::new(0, 0);
            let v = execute_extent(
                &p,
                &planes,
                &Foreign::NONE,
                &mut sc,
                Out::Mask(&mut got),
                0..n,
            )
            .unwrap_or_else(|e| panic!("fused keep n={n} imm={imm:#04x}: {e:?}"));
            assert_eq!(v, Value::Mask(Operand::Scratch(0)));
            assert_eq!(got, want, "n={n} imm={imm:#04x}: fused vs oracle mismatch");
        }
    }
}

/// FAILS IF: a `fused_keep`-eligible program needs a non-empty `Scratch`
/// when the caller passes `Out::Mask` — either a zero-slot `Scratch::new(0,
/// 0)` is rejected, or a correctly-sized but POISONED scratch buffer is
/// touched (any word changed from its `u64::MAX` poison) by the fused path.
#[test]
fn fused_keep_needs_no_scratch_with_out_mask() {
    let n = 400usize;
    let mut seed = 0x00C0_FFEE_u64;
    let (m0, m1, m2) = (
        random_plane(n, &mut seed, 2),
        random_plane(n, &mut seed, 3),
        random_plane(n, &mut seed, 5),
    );
    let masks: [&[u64]; 3] = [&m0, &m1, &m2];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (pa, pb, pc) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
    let p = Program::new(
        vec![
            MaskOp::And {
                a: pa,
                b: pb,
                dst: 0,
            },
            MaskOp::Ternlog {
                imm: 0x96,
                a: Operand::Scratch(0),
                b: pc,
                c: pa,
                dst: 1,
            },
        ],
        Terminal::Keep {
            mask: Operand::Scratch(1),
        },
    );
    assert!(p.fused_keep().is_some());
    let words = words_for(n);

    let mut want = vec![0u64; words];
    reference_execute_into(&p, &planes, &Foreign::NONE, Out::Mask(&mut want)).expect("oracle");

    // A zero-slot, zero-word scratch: the fused path must never consult it.
    let mut got = garbage(words);
    let mut sc = Scratch::new(0, 0);
    assert_eq!(sc.slots(), 0);
    assert_eq!(sc.words(), 0);
    let v = execute_extent(
        &p,
        &planes,
        &Foreign::NONE,
        &mut sc,
        Out::Mask(&mut got),
        0..n,
    )
    .expect("a zero-slot scratch must suffice for a fused Keep under Out::Mask");
    assert_eq!(v, Value::Mask(Operand::Scratch(1)));
    assert_eq!(got, want);

    // A correctly-sized carved scratch. `Scratch::over` zero-fills what it
    // carves (a Keep program still requires scratch, for the `Out::None`
    // shape), so a poisoned buffer cannot survive the carve itself. The
    // observable instead: every carved slot still reads all-zero after the
    // run. The tiled path would leave the kept mask in slot 1.
    let mut buf = vec![u64::MAX; 8 * words.max(1)];
    let mut sc2 = Scratch::over_for_program(&mut buf, &p, n).expect("carve");
    let mut got2 = garbage(words);
    execute_extent(
        &p,
        &planes,
        &Foreign::NONE,
        &mut sc2,
        Out::Mask(&mut got2),
        0..n,
    )
    .expect("carved scratch, fused keep");
    assert_eq!(got2, want);
    assert!(
        want.iter().any(|&w| w != 0),
        "fixture: the kept mask must be non-empty"
    );
    for i in 0..sc2.slots() {
        let slot = sc2.slot(i as u16).expect("carved slot");
        assert!(
            slot.iter().all(|&w| w == 0),
            "the fused Keep path must never write into the caller's scratch (slot {i})"
        );
    }
}

/// FAILS IF: with `n_rows == 65` (a population whose last word carries only
/// one live row) and an ODD ternlog table (a bare `Not` of a resident
/// plane), the fused `Keep` write leaves any of the dead tail bits of the
/// last word set, or disagrees word-for-word with the tiled twin over the
/// whole population.
#[test]
fn fused_keep_writes_the_whole_last_word_and_clears_its_tail() {
    let n = 65usize;
    let mut seed = 0xB0B0_u64;
    let m0 = random_plane(n, &mut seed, 2);
    let masks: [&[u64]; 1] = [&m0];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let ops = vec![MaskOp::Not {
        a: Operand::Plane(0),
        dst: 0,
    }];
    let p = Program::new(
        ops.clone(),
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let f = p.fused_keep().expect("a bare Not of a plane must collapse");
    assert_eq!(f.imm & 1, 1, "NOT's own table must be odd (f(0,0,0) = 1)");
    assert!(matches!(p.compile().lowering(), Lowering::TernlogKeep(_)));

    let words = words_for(n);
    assert_eq!(words, 2, "n=65 must span exactly two words");
    let mut out = garbage(words);
    let v = execute_extent(
        &p,
        &planes,
        &Foreign::NONE,
        &mut Scratch::new(0, 0),
        Out::Mask(&mut out),
        0..n,
    )
    .expect("fused keep");
    assert_eq!(v, Value::Mask(Operand::Scratch(0)));

    let live = n % 64;
    assert_eq!(
        out[1] >> live,
        0,
        "dead tail bits (row {live}..64 of the last word) must be cleared"
    );
    // The whole first word is written exactly — no garbage bit survives.
    assert_eq!(out[0], !m0[0]);

    let tp = force_tiled(&ops, 0);
    let mut out_tiled = garbage(words);
    let mut sc = Scratch::for_program(&tp, n).expect("tiled scratch");
    execute_extent(
        &tp,
        &planes,
        &Foreign::NONE,
        &mut sc,
        Out::Mask(&mut out_tiled),
        0..n,
    )
    .expect("tiled keep");
    assert_eq!(out, out_tiled, "fused vs tiled last-word mismatch");
}

/// FAILS IF: a chain that reads a FOURTH distinct resident plane is admitted
/// by `fused_keep` (it cannot fit one 3-input table), or if its lowering is
/// anything but `Tiled`, or if the tiled path it falls back to computes the
/// wrong `Keep` result.
#[test]
fn a_fourth_plane_declines_to_tiled() {
    let n = 500usize;
    let mut seed = 0x4EAF_u64;
    let plane_vecs: Vec<Vec<u64>> = (0..4)
        .map(|i| random_plane(n, &mut seed, 2 + i as u64))
        .collect();
    let masks: [&[u64]; 4] = [
        &plane_vecs[0],
        &plane_vecs[1],
        &plane_vecs[2],
        &plane_vecs[3],
    ];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (p0, p1, p2, p3) = (
        Operand::Plane(0),
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
    );
    let p = Program::new(
        vec![
            MaskOp::And {
                a: p0,
                b: p1,
                dst: 0,
            },
            MaskOp::Or {
                a: p2,
                b: p3,
                dst: 1,
            },
            MaskOp::Xor {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        Terminal::Keep {
            mask: Operand::Scratch(2),
        },
    );
    assert!(
        p.fused_keep().is_none(),
        "a fourth distinct plane must never fuse"
    );
    assert!(matches!(p.compile().lowering(), Lowering::Tiled));
    assert!(p.requires_scratch());

    let words = words_for(n);
    let mut want = vec![0u64; words];
    reference_execute_into(&p, &planes, &Foreign::NONE, Out::Mask(&mut want)).expect("oracle");

    let mut out = garbage(words);
    let mut sc = Scratch::for_program(&p, n).expect("scratch");
    execute_extent(
        &p,
        &planes,
        &Foreign::NONE,
        &mut sc,
        Out::Mask(&mut out),
        0..n,
    )
    .expect("tiled keep");
    assert_eq!(out, want, "the four-plane chain's tiled Keep is wrong");
}

/// FAILS IF: a program eligible for `fused_keep` fails, errors, or produces
/// the wrong `Keep` result when the caller passes `Out::None` with a
/// whole-width (single-tile) scratch instead of `Out::Mask` — the shape that
/// must run the ordinary tiled path and leave its answer in the named
/// scratch slot, per `Scratch::slot`'s own contract.
#[test]
fn keep_with_out_none_still_runs_tiled_and_fills_the_slot() {
    let n = 300usize;
    let mut seed = 0x0E1_u64;
    let (m0, m1) = (random_plane(n, &mut seed, 2), random_plane(n, &mut seed, 3));
    let masks: [&[u64]; 2] = [&m0, &m1];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let p = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    assert!(
        p.fused_keep().is_some(),
        "eligible for the fused path when Out::Mask is offered"
    );

    let words = words_for(n);
    let mut buf = vec![0u64; scratch_words_for(words, 1).expect("size fits")];
    let mut sc = Scratch::over(&mut buf, words, 1).expect("whole-width, single-tile scratch");
    let v = execute_extent(&p, &planes, &Foreign::NONE, &mut sc, Out::None, 0..n)
        .expect("Out::None must still run the tiled Keep path");
    assert_eq!(v, Value::Mask(Operand::Scratch(0)));

    let want = reference_scratch(&p, &planes).expect("oracle scratch");
    let got = sc
        .slot(0)
        .expect("slot 0 must be readable after the tiled run");
    assert_eq!(got, want[0].as_slice());
}
