//! Constants — direct transcode of `qtc.h` macros + `fib.c` static tables.
//!
//! All values match the C reference bit-for-bit (modulo the f64 representation
//! of derived irrationals, which the tests verify to machine epsilon).

use crate::types::TilingDesc;

/// Golden ratio φ = (1 + √5) / 2.
///
/// Pisot-Vijayaraghavan property: φ is an algebraic integer whose conjugate
/// ψ = (1 − √5)/2 has |ψ| < 1. This is the algebraic reason why the
/// Fibonacci hierarchy never collapses (paper Thm 2 + Cor 4).
pub const PHI: f64 = 1.618_033_988_749_894_8;

/// 1/φ ≈ 0.618_033_988_749_894_8 — the canonical Fibonacci cut-and-project
/// slope. `qc_word_tiling` uses this; `qc_word_tiling_alpha` accepts arbitrary
/// alpha for the multi-tiling engine.
pub const INV_PHI: f64 = 0.618_033_988_749_894_8;

/// Maximum hierarchy depth (levels 0..=9 → 10 entries).
///
/// Matches `QTC_MAX_HIER = 10` in the C reference. The Fibonacci hierarchy
/// `never collapses` (Thm 2), so this bound is operational, not theoretical.
pub const MAX_HIER: usize = 10;

/// Number of usable n-gram levels above level 0 (1..=9 → 9 entries).
pub const N_LEVELS: usize = 9;

/// Fibonacci phrase lengths per hierarchy level.
///
/// Level k spans `HIER_WORD_LENS[k]` words. The sequence is
/// `{F_3, F_4, ..., F_{12}} = {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}` —
/// the first 10 Fibonacci numbers ≥ 2.
pub const HIER_WORD_LENS: [usize; MAX_HIER] = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144];

/// Words per encoding level (including escape/unigram levels 0–1).
///
/// `LEVEL_WORDS[k]` matches `QTM_LEVEL_WORDS[k]` in the C reference.
pub const LEVEL_WORDS: [usize; 12] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];

/// Number of n-gram codebooks (unigram .. 144-gram).
pub const N_CODEBOOKS: usize = 11;

/// Total number of multi-tilings in the v5.6 engine.
///
/// 12 golden-ratio phases + 6 quadratic-irrational tilings (√58−7 ×2,
/// noble-5 ×2, √13−3 ×2) + 18 greedy-search-discovered alphas (paired
/// at phase 0.0 and 0.5).
pub const N_TILINGS: usize = 36;

/// Fibonacci-only tiling count: just the 12 golden-ratio phases.
pub const N_TILINGS_FIB: usize = 12;

/// Noble-5: continued fraction `[0; 1, 1, 1, 1, 2, 1̄]`.
///
/// `alpha = 62 * (99 − √5) / 9796 ≈ 0.612_429_949_5`. Hierarchy reaches
/// level 5 (21-gram) before collapsing — useful as an A/B baseline against
/// the non-collapsing Fibonacci.
#[inline]
#[must_use]
pub fn noble5_alpha() -> f64 {
    62.0 * (99.0 - 5.0_f64.sqrt()) / 9796.0
}

/// √58 − 7: continued fraction `[0; 1, 1, 1, 1, 1, 1, 14, ...]`.
/// Closest quadratic irrational to 1/φ; reaches 55-gram level.
#[inline]
#[must_use]
pub fn sqrt58_alpha() -> f64 {
    58.0_f64.sqrt() - 7.0
}

/// √13 − 3: continued fraction `[0; 1, 1, 1, 1, 6, ...]`.
/// L-density 0.606 — broadest 3-gram / 8-gram coverage; reaches 8-gram.
#[inline]
#[must_use]
pub fn sqrt13_alpha() -> f64 {
    13.0_f64.sqrt() - 3.0
}

/// Build the canonical 36-tiling descriptor table.
///
/// Direct transcode of `qtm_get_tiling_descs` in `fib.c`. Names match the
/// C reference so cross-checks against published results stay legible.
///
/// Layout (indices):
/// - `0..12`     — 12 golden-ratio phases, golden-spaced (φ-iterate of phase).
/// - `12..14`    — √58 − 7 phases at 0.0 and 0.5.
/// - `14..16`    — noble-5 phases at 0.0 and 0.5.
/// - `16..18`    — √13 − 3 phases at 0.0 and 0.5.
/// - `18..20`    — α = 0.502 (far-out, massive trigram/5-gram gain).
/// - `20..36`    — eight near-golden alphas in `[0.612, 0.622]`, paired phases.
#[must_use]
pub fn tiling_descs() -> [TilingDesc; N_TILINGS] {
    let a58 = sqrt58_alpha();
    let an5 = noble5_alpha();
    let a13 = sqrt13_alpha();

    let mut descs = [TilingDesc {
        alpha: 0.0,
        phase: 0.0,
        name: "",
    }; N_TILINGS];

    // Tier 1: 12 golden-ratio phases, golden-spaced.
    for i in 0..12 {
        descs[i] = TilingDesc {
            alpha: INV_PHI,
            phase: ((i as f64) * INV_PHI).rem_euclid(1.0),
            name: "golden",
        };
    }

    // Tier 2: √58 − 7 phases.
    descs[12] = TilingDesc {
        alpha: a58,
        phase: 0.0,
        name: "sqrt58",
    };
    descs[13] = TilingDesc {
        alpha: a58,
        phase: 0.5,
        name: "sqrt58",
    };

    // Tier 3: noble-5 phases.
    descs[14] = TilingDesc {
        alpha: an5,
        phase: 0.0,
        name: "noble5",
    };
    descs[15] = TilingDesc {
        alpha: an5,
        phase: 0.5,
        name: "noble5",
    };

    // Tier 4: √13 − 3 phases.
    descs[16] = TilingDesc {
        alpha: a13,
        phase: 0.0,
        name: "sqrt13",
    };
    descs[17] = TilingDesc {
        alpha: a13,
        phase: 0.5,
        name: "sqrt13",
    };

    // Tier 5: greedy-search-discovered alphas (enwik8 calibration in the
    // C reference). Names mirror the upstream labels for traceability.
    descs[18] = TilingDesc {
        alpha: 0.502,
        phase: 0.0,
        name: "opt-0.502",
    };
    descs[19] = TilingDesc {
        alpha: 0.502,
        phase: 0.5,
        name: "opt-0.502",
    };

    descs[20] = TilingDesc {
        alpha: 0.6190,
        phase: 0.0,
        name: "opt-0.619",
    };
    descs[21] = TilingDesc {
        alpha: 0.6190,
        phase: 0.5,
        name: "opt-0.619",
    };

    descs[22] = TilingDesc {
        alpha: 0.6170,
        phase: 0.0,
        name: "opt-0.617",
    };
    descs[23] = TilingDesc {
        alpha: 0.6170,
        phase: 0.5,
        name: "opt-0.617",
    };

    descs[24] = TilingDesc {
        alpha: 0.6160,
        phase: 0.0,
        name: "opt-0.616",
    };
    descs[25] = TilingDesc {
        alpha: 0.6160,
        phase: 0.5,
        name: "opt-0.616",
    };

    descs[26] = TilingDesc {
        alpha: 0.6200,
        phase: 0.0,
        name: "opt-0.620",
    };
    descs[27] = TilingDesc {
        alpha: 0.6200,
        phase: 0.5,
        name: "opt-0.620",
    };

    descs[28] = TilingDesc {
        alpha: 0.6140,
        phase: 0.0,
        name: "opt-0.614",
    };
    descs[29] = TilingDesc {
        alpha: 0.6140,
        phase: 0.5,
        name: "opt-0.614",
    };

    descs[30] = TilingDesc {
        alpha: 0.6210,
        phase: 0.0,
        name: "opt-0.621",
    };
    descs[31] = TilingDesc {
        alpha: 0.6210,
        phase: 0.5,
        name: "opt-0.621",
    };

    descs[32] = TilingDesc {
        alpha: 0.6220,
        phase: 0.0,
        name: "opt-0.622",
    };
    descs[33] = TilingDesc {
        alpha: 0.6220,
        phase: 0.5,
        name: "opt-0.622",
    };

    descs[34] = TilingDesc {
        alpha: 0.6120,
        phase: 0.0,
        name: "opt-0.612",
    };
    descs[35] = TilingDesc {
        alpha: 0.6120,
        phase: 0.5,
        name: "opt-0.612",
    };

    descs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_matches_irrational_to_machine_epsilon() {
        let recomputed = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((PHI - recomputed).abs() < 1e-15);
        assert!((INV_PHI - 1.0 / recomputed).abs() < 1e-15);
    }

    #[test]
    fn hier_word_lens_are_fibonacci() {
        // F_3..F_12 = 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
        for w in HIER_WORD_LENS.windows(3) {
            assert_eq!(w[0] + w[1], w[2], "Fibonacci recurrence");
        }
        assert_eq!(HIER_WORD_LENS[0], 2);
        assert_eq!(HIER_WORD_LENS[MAX_HIER - 1], 144);
    }

    #[test]
    fn tiling_descs_have_canonical_shape() {
        let d = tiling_descs();
        // 12 golden phases.
        for i in 0..12 {
            assert!((d[i].alpha - INV_PHI).abs() < 1e-15);
            assert_eq!(d[i].name, "golden");
            assert!(d[i].phase >= 0.0 && d[i].phase < 1.0);
        }
        // Quadratic irrationals.
        assert_eq!(d[12].name, "sqrt58");
        assert_eq!(d[14].name, "noble5");
        assert_eq!(d[16].name, "sqrt13");
        // Greedy-discovered.
        assert_eq!(d[18].name, "opt-0.502");
        assert!((d[18].alpha - 0.502).abs() < 1e-15);
    }
}
