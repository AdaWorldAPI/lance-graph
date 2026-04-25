//! Role keys — canonical deterministic `[start:stop]`-slice VSA role bindings.
//!
//! Each role owns a **disjoint contiguous slice** of the 10,000-dim VSA
//! space (compatible with `CrystalFingerprint::Binary16K`). Only the bits
//! in that slice are set to a deterministic pseudo-random pattern seeded
//! from FNV-64 of the label; all other bits are zero. This is the
//! VSA-native `[start:stop]` addressing convention — **not** scattered bits.
//!
//! Because the slices are disjoint, XOR-binding a value with a role key
//! only affects that role's slice, and bundles of different role-bindings
//! don't contaminate each other.
//!
//! ## Space layout (10,000 total dims)
//!
//! ```text
//! [    0 .. 2000)   SUBJECT_KEY
//! [ 2000 .. 4000)   PREDICATE_KEY
//! [ 4000 .. 6000)   OBJECT_KEY
//! [ 6000 .. 7500)   MODIFIER_KEY
//! [ 7500 .. 9000)   CONTEXT_KEY
//! [ 9000 .. 9200)   TEMPORAL_KEY
//! [ 9200 .. 9400)   KAUSAL_KEY
//! [ 9400 .. 9500)   MODAL_KEY
//! [ 9500 .. 9650)   LOKAL_KEY
//! [ 9650 .. 9750)   INSTRUMENT_KEY
//! [ 9750 .. 9780)   BENEFICIARY_KEY
//! [ 9780 .. 9810)   GOAL_KEY
//! [ 9810 .. 9840)   SOURCE_KEY
//! [ 9840 .. 9910)   Finnish 15 cases — ~4-5 dims each
//! [ 9910 .. 9970)   12 tense keys — 5 dims each
//! [ 9970 .. 10000)  7 NARS inference keys — ~4 dims each
//! ```

use std::sync::LazyLock;

use super::finnish::FinnishCase;
use super::inference::NarsInference;

/// VSA vector width in `u64` words. Matches `ndarray::hpc::vsa::VSA_WORDS`.
/// 157 × 64 = 10,048 bits, covering the 10,000 VSA dims with 48 slack bits.
pub const VSA_WORDS: usize = 157;

/// VSA vector width in dimensions (bits actually used).
pub const VSA_DIMS: usize = 10_000;

// NOTE: The `Vsa10k = [u64; 157]` bitpacked type alias, `VSA_ZERO` constant,
// and `RoleKey::{bind, unbind, recovery_margin}` methods that existed in
// early 2026-04-21 session work were REMOVED in the cleanup commit
// `cd5c049...`. That code used GF(2)/XOR algebra which is the
// Binary16K Hamming-comparison format, not the real-valued VSA bundling
// format. Correct VSA substrate is `Vsa10kF32 = Box<[f32; 10_000]>`
// (existing) or `Vsa16kF32 = Box<[f32; 16_384]>` (pending rescale), with
// element-wise multiply/add via existing `crystal::fingerprint::{vsa_bind,
// vsa_bundle, vsa_cosine}`.
//
// See `CHANGELOG.md` § VSA format switches and
// `.claude/knowledge/vsa-switchboard-architecture.md` for the full
// three-layer architecture. Role keys (this module) are Layer-2 catalogue
// ONLY — identity slice boundaries. Algebra lives in Layer-1 `crystal/`.

/// A role key owns a contiguous slice of the VSA space.
/// Outside the slice, **all bits are zero**.
pub struct RoleKey {
    pub words: Box<[u64; VSA_WORDS]>,
    pub slice_start: usize,
    pub slice_end: usize,
    pub label: &'static str,
}

impl RoleKey {
    /// Dim range of this role's slice.
    pub fn slice_range(&self) -> std::ops::Range<usize> {
        self.slice_start..self.slice_end
    }

    /// Width of this role's slice in dimensions.
    pub fn slice_width(&self) -> usize {
        self.slice_end - self.slice_start
    }

    // NOTE: `bind/unbind/recovery_margin` methods removed in cleanup commit
    // `cd5c049...` (see CHANGELOG.md). Those operated on a hallucinated
    // `Vsa10k = [u64; 157]` bitpacked carrier with GF(2)/XOR algebra —
    // the wrong substrate for lossless role bundling. Correct algebra
    // is element-wise multiply/add on `Vsa10kF32`/`Vsa16kF32` via existing
    // `crystal::fingerprint::{vsa_bind, vsa_bundle, vsa_cosine}`.
    //
    // Role keys are a Layer-2 catalogue (slice boundaries for a domain);
    // algebra is Layer-1 on the switchboard carrier. See
    // `.claude/knowledge/vsa-switchboard-architecture.md`.

    /// Generate a deterministic role key: pseudo-random bits in `[start..end)`,
    /// zeros everywhere else. Seeded from FNV-64 of the label.
    fn generate(label: &'static str, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= VSA_DIMS);
        let mut words = Box::new([0u64; VSA_WORDS]);
        let seed = fnv64(label);
        for dim in start..end {
            let mut state = seed.wrapping_add(dim as u64);
            let bit = lcg_next(&mut state) & 1;
            if bit == 1 {
                let word = dim / 64;
                let offset = dim % 64;
                words[word] |= 1u64 << offset;
            }
        }
        Self { words, slice_start: start, slice_end: end, label }
    }
}

// NOTE: `vsa_xor`, `vsa_similarity`, `word_slice_mask`, and
// `slice_matching_bits` free functions removed in cleanup commit
// `cd5c049...` (see CHANGELOG.md). They operated on the hallucinated
// `Vsa10k = [u64; 157]` bitpacked carrier with GF(2)/XOR algebra.
// Correct VSA operations are in `crystal::fingerprint`:
//   `vsa_bind` (element-wise multiply on `[f32; 10_000]`)
//   `vsa_bundle` (element-wise add)
//   `vsa_cosine` (similarity)

fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1);
    *state
}

// ---------------------------------------------------------------------------
// SPO core roles
// ---------------------------------------------------------------------------

pub static SUBJECT_KEY:   LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("SUBJECT",      0, 2000));
pub static PREDICATE_KEY: LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("PREDICATE", 2000, 4000));
pub static OBJECT_KEY:    LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("OBJECT",    4000, 6000));
pub static MODIFIER_KEY:  LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("MODIFIER",  6000, 7500));
pub static CONTEXT_KEY:   LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("CONTEXT",   7500, 9000));

// ---------------------------------------------------------------------------
// TEKAMOLO slots
// ---------------------------------------------------------------------------

pub static TEMPORAL_KEY: LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("TEMPORAL", 9000, 9200));
pub static KAUSAL_KEY:   LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("KAUSAL",   9200, 9400));
pub static MODAL_KEY:    LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("MODAL",    9400, 9500));
pub static LOKAL_KEY:    LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("LOKAL",    9500, 9650));

// ---------------------------------------------------------------------------
// Future-ready roles (CausalityFlow not extended yet)
// ---------------------------------------------------------------------------

pub static INSTRUMENT_KEY:  LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("INSTRUMENT",  9650, 9750));
pub static BENEFICIARY_KEY: LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("BENEFICIARY", 9750, 9780));
pub static GOAL_KEY:        LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("GOAL",        9780, 9810));
pub static SOURCE_KEY:      LazyLock<RoleKey> = LazyLock::new(|| RoleKey::generate("SOURCE",      9810, 9840));

// ---------------------------------------------------------------------------
// Finnish 15 cases — [9840 .. 9910), 70 dims / 15 cases ≈ 4-5 dims each.
// First 10 cases get 5 dims; remaining 5 cases get 4 dims. Total = 50 + 20 = 70.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const FINNISH_START: usize = 9840;
#[allow(dead_code)]
const FINNISH_END:   usize = 9910;

/// Inclusive-exclusive slice boundaries for each of the 15 Finnish cases,
/// indexed by `FinnishCase as u8`. Widths: first 10 cases = 5 dims, last 5 = 4 dims.
const FINNISH_SLICES: [(usize, usize); 15] = [
    (9840, 9845), // Nominative
    (9845, 9850), // Genitive
    (9850, 9855), // Accusative
    (9855, 9860), // Partitive
    (9860, 9865), // Inessive
    (9865, 9870), // Elative
    (9870, 9875), // Illative
    (9875, 9880), // Adessive
    (9880, 9885), // Ablative
    (9885, 9890), // Allative
    (9890, 9894), // Essive
    (9894, 9898), // Translative
    (9898, 9902), // Instructive
    (9902, 9906), // Abessive
    (9906, 9910), // Comitative
];

const FINNISH_LABELS: [&str; 15] = [
    "FI_NOMINATIVE", "FI_GENITIVE", "FI_ACCUSATIVE", "FI_PARTITIVE",
    "FI_INESSIVE",   "FI_ELATIVE",  "FI_ILLATIVE",
    "FI_ADESSIVE",   "FI_ABLATIVE", "FI_ALLATIVE",
    "FI_ESSIVE",     "FI_TRANSLATIVE", "FI_INSTRUCTIVE",
    "FI_ABESSIVE",   "FI_COMITATIVE",
];

static FINNISH_KEYS: LazyLock<[RoleKey; 15]> = LazyLock::new(|| {
    core::array::from_fn(|i| {
        let (s, e) = FINNISH_SLICES[i];
        RoleKey::generate(FINNISH_LABELS[i], s, e)
    })
});

pub fn finnish_case_key(case: FinnishCase) -> &'static RoleKey {
    &FINNISH_KEYS[case as usize]
}

// ---------------------------------------------------------------------------
// 12 tense keys — [9910 .. 9970), 60 dims / 12 = 5 dims each.
// ---------------------------------------------------------------------------

/// Tense key, 12 variants, each owning 5 dims of the VSA space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tense {
    Present = 0,
    Past = 1,
    Future = 2,
    PresentContinuous = 3,
    PastContinuous = 4,
    FutureContinuous = 5,
    Perfect = 6,
    Pluperfect = 7,
    FuturePerfect = 8,
    Habitual = 9,
    Potential = 10,
    Imperative = 11,
}

const TENSE_START: usize = 9910;
#[allow(dead_code)]
const TENSE_END:   usize = 9970;
const TENSE_WIDTH: usize = 5;

const TENSE_LABELS: [&str; 12] = [
    "T_PRESENT", "T_PAST", "T_FUTURE",
    "T_PRESENT_CONTINUOUS", "T_PAST_CONTINUOUS", "T_FUTURE_CONTINUOUS",
    "T_PERFECT", "T_PLUPERFECT", "T_FUTURE_PERFECT",
    "T_HABITUAL", "T_POTENTIAL", "T_IMPERATIVE",
];

static TENSE_KEYS: LazyLock<[RoleKey; 12]> = LazyLock::new(|| {
    core::array::from_fn(|i| {
        let s = TENSE_START + i * TENSE_WIDTH;
        let e = s + TENSE_WIDTH;
        RoleKey::generate(TENSE_LABELS[i], s, e)
    })
});

pub fn tense_key(tense: Tense) -> &'static RoleKey {
    &TENSE_KEYS[tense as usize]
}

// ---------------------------------------------------------------------------
// 7 NARS inference keys — [9970 .. 10000), 30 dims / 7 ≈ 4 dims each.
// First 2 get 5 dims, remaining 5 get 4 dims. Total = 10 + 20 = 30.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const NARS_START: usize = 9970;
#[allow(dead_code)]
const NARS_END:   usize = 10_000;

const NARS_SLICES: [(usize, usize); 7] = [
    (9970, 9975), // Deduction
    (9975, 9980), // Induction
    (9980, 9984), // Abduction
    (9984, 9988), // Revision
    (9988, 9992), // Synthesis
    (9992, 9996), // Extrapolation
    (9996, 10_000), // CounterfactualSynthesis
];

const NARS_LABELS: [&str; 7] = [
    "N_DEDUCTION", "N_INDUCTION", "N_ABDUCTION",
    "N_REVISION", "N_SYNTHESIS", "N_EXTRAPOLATION",
    "N_COUNTERFACTUAL",
];

static NARS_KEYS: LazyLock<[RoleKey; 7]> = LazyLock::new(|| {
    core::array::from_fn(|i| {
        let (s, e) = NARS_SLICES[i];
        RoleKey::generate(NARS_LABELS[i], s, e)
    })
});

pub fn nars_inference_key(inf: NarsInference) -> &'static RoleKey {
    let idx = match inf {
        NarsInference::Deduction               => 0,
        NarsInference::Induction               => 1,
        NarsInference::Abduction               => 2,
        NarsInference::Revision                => 3,
        NarsInference::Synthesis               => 4,
        NarsInference::Extrapolation           => 5,
        NarsInference::CounterfactualSynthesis => 6,
    };
    &NARS_KEYS[idx]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect every (start, end, label) from every defined role key.
    fn all_slices() -> Vec<(usize, usize, &'static str)> {
        let mut v: Vec<(usize, usize, &'static str)> = Vec::new();
        for k in [
            &*SUBJECT_KEY, &*PREDICATE_KEY, &*OBJECT_KEY,
            &*MODIFIER_KEY, &*CONTEXT_KEY,
            &*TEMPORAL_KEY, &*KAUSAL_KEY, &*MODAL_KEY, &*LOKAL_KEY,
            &*INSTRUMENT_KEY, &*BENEFICIARY_KEY, &*GOAL_KEY, &*SOURCE_KEY,
        ] {
            v.push((k.slice_start, k.slice_end, k.label));
        }
        for k in FINNISH_KEYS.iter() { v.push((k.slice_start, k.slice_end, k.label)); }
        for k in TENSE_KEYS.iter()   { v.push((k.slice_start, k.slice_end, k.label)); }
        for k in NARS_KEYS.iter()    { v.push((k.slice_start, k.slice_end, k.label)); }
        v
    }

    #[test]
    fn all_slices_disjoint() {
        let mut slices = all_slices();
        slices.sort_by_key(|(s, _, _)| *s);
        for pair in slices.windows(2) {
            let (s0, e0, l0) = pair[0];
            let (s1, _e1, l1) = pair[1];
            assert!(
                e0 <= s1,
                "slice overlap: {l0} [{s0}..{e0}) vs {l1} [{s1}..)"
            );
        }
    }

    #[test]
    fn all_slices_within_vsa_dims() {
        for (s, e, label) in all_slices() {
            assert!(s < e, "empty slice for {label}");
            assert!(e <= VSA_DIMS, "slice {label} ends at {e} > {VSA_DIMS}");
        }
    }

    #[test]
    fn role_key_bits_only_in_slice() {
        // SUBJECT_KEY owns [0..2000). Every bit >= dim 2000 must be zero.
        let k = &*SUBJECT_KEY;
        for dim in 0..(VSA_WORDS * 64) {
            let word = dim / 64;
            let offset = dim % 64;
            let bit = (k.words[word] >> offset) & 1;
            if dim < k.slice_start || dim >= k.slice_end {
                assert_eq!(
                    bit, 0,
                    "SUBJECT_KEY bit set outside slice at dim {dim}"
                );
            }
        }

        // Spot-check KAUSAL_KEY [9200..9400).
        let k = &*KAUSAL_KEY;
        for dim in 0..(VSA_WORDS * 64) {
            let word = dim / 64;
            let offset = dim % 64;
            let bit = (k.words[word] >> offset) & 1;
            if dim < k.slice_start || dim >= k.slice_end {
                assert_eq!(
                    bit, 0,
                    "KAUSAL_KEY bit set outside slice at dim {dim}"
                );
            }
        }
    }

    #[test]
    fn role_keys_deterministic() {
        // Re-generate from scratch; compare to the static instance.
        let a = RoleKey::generate("SUBJECT", 0, 2000);
        let b = RoleKey::generate("SUBJECT", 0, 2000);
        assert_eq!(a.words.as_ref(), b.words.as_ref());
        assert_eq!(a.words.as_ref(), SUBJECT_KEY.words.as_ref());
    }

    #[test]
    fn finnish_case_lookup_covers_all_15() {
        let all = [
            FinnishCase::Nominative, FinnishCase::Genitive, FinnishCase::Accusative,
            FinnishCase::Partitive,  FinnishCase::Inessive, FinnishCase::Elative,
            FinnishCase::Illative,   FinnishCase::Adessive, FinnishCase::Ablative,
            FinnishCase::Allative,   FinnishCase::Essive,   FinnishCase::Translative,
            FinnishCase::Instructive, FinnishCase::Abessive, FinnishCase::Comitative,
        ];
        for case in all {
            let k = finnish_case_key(case);
            assert!(k.slice_start >= FINNISH_START);
            assert!(k.slice_end <= FINNISH_END);
            assert!(k.slice_width() >= 4);
        }
    }

    #[test]
    fn nars_inference_lookup_covers_all_7() {
        let all = [
            NarsInference::Deduction, NarsInference::Induction,
            NarsInference::Abduction, NarsInference::Revision,
            NarsInference::Synthesis, NarsInference::Extrapolation,
            NarsInference::CounterfactualSynthesis,
        ];
        for inf in all {
            let k = nars_inference_key(inf);
            assert!(k.slice_start >= NARS_START);
            assert!(k.slice_end <= NARS_END);
            assert!(k.slice_width() >= 4);
        }
    }

    // Tests for the RoleKey-as-operator family (bind/unbind/recovery_margin,
    // vsa_xor, vsa_similarity) were REMOVED in cleanup commit `cd5c049...`
    // along with the methods they covered. See CHANGELOG.md § VSA format
    // switches. Correct algebra tests live on the carrier in
    // `crystal::fingerprint` (existing: vsa_bind/bundle/superpose/cosine).

    #[test]
    fn tense_lookup_covers_all_12() {
        let all = [
            Tense::Present, Tense::Past, Tense::Future,
            Tense::PresentContinuous, Tense::PastContinuous, Tense::FutureContinuous,
            Tense::Perfect, Tense::Pluperfect, Tense::FuturePerfect,
            Tense::Habitual, Tense::Potential, Tense::Imperative,
        ];
        for t in all {
            let k = tense_key(t);
            assert_eq!(k.slice_width(), TENSE_WIDTH);
            assert!(k.slice_start >= TENSE_START);
            assert!(k.slice_end <= TENSE_END);
        }
    }
}
