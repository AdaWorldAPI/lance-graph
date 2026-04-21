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

/// 10K-dim VSA vector carrier, bit-packed in 157 × u64 words.
///
/// Chosen as `[u64; VSA_WORDS]` (not `Box<[...]>`) so it stays stack-friendly
/// for XOR binding in the hot path. Matches the layout consumed by
/// `ndarray::hpc::vsa::*` without requiring an ndarray import here.
pub type Vsa10k = [u64; VSA_WORDS];

/// Zero vector — identity element for XOR bundling.
pub const VSA_ZERO: Vsa10k = [0u64; VSA_WORDS];

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

    /// Bind `content` into this role's slice via slice-masked XOR.
    ///
    /// `bind` returns a vector that is **zero outside** `[slice_start..slice_end)`
    /// and equals `(content ^ role_key)` inside the slice. Slice-masking is
    /// enforced at bind time (not left as caller discipline) so XOR
    /// superposition of multiple bindings is lossless per role:
    ///
    /// ```text
    /// let bundle = xor(SUBJECT.bind(s_content), OBJECT.bind(o_content));
    /// SUBJECT.unbind(&bundle) == SUBJECT.bind(s_content) (content inside SUBJECT slice)
    /// OBJECT.unbind(&bundle)  == OBJECT.bind(o_content)  (content inside OBJECT slice)
    /// ```
    ///
    /// Self-inverse within the slice: unbinding twice recovers the
    /// slice-masked content.
    pub fn bind(&self, content: &Vsa10k) -> Vsa10k {
        let mut out = [0u64; VSA_WORDS];
        let first_word = self.slice_start / 64;
        let last_word = match self.slice_end {
            0 => 0,
            e => (e - 1) / 64,
        };
        for w in first_word..=last_word {
            let mask = word_slice_mask(w, self.slice_start, self.slice_end);
            out[w] = (content[w] & mask) ^ self.words[w];
        }
        out
    }

    /// Unbind: slice-masked XOR with this role key.
    ///
    /// Given a bundle produced by XOR-superposing slice-masked bindings
    /// for multiple roles, `unbind` recovers this role's slice-masked
    /// content: `key.unbind(&bundle) == content & slice_mask`.
    pub fn unbind(&self, bundle: &Vsa10k) -> Vsa10k {
        let mut out = [0u64; VSA_WORDS];
        let first_word = self.slice_start / 64;
        let last_word = match self.slice_end {
            0 => 0,
            e => (e - 1) / 64,
        };
        for w in first_word..=last_word {
            let mask = word_slice_mask(w, self.slice_start, self.slice_end);
            out[w] = (bundle[w] & mask) ^ self.words[w];
        }
        out
    }

    /// Hamming similarity between two vectors restricted to this role's
    /// slice. Returns the fraction of matching bits in `[0, 1]`.
    ///
    /// Used after `unbind` to measure how cleanly the expected content
    /// was recovered from the bundle — the per-role likelihood term in
    /// the active-inference free-energy computation.
    pub fn recovery_margin(&self, unbound: &Vsa10k, expected: &Vsa10k) -> f32 {
        let width = self.slice_width();
        if width == 0 {
            return 0.0;
        }
        let matches = slice_matching_bits(
            unbound, expected, self.slice_start, self.slice_end,
        );
        matches as f32 / width as f32
    }

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

/// Bundle two VSA vectors via XOR superposition. Commutative, associative,
/// self-inverse — the free algebraic structure on which role binding sits.
///
/// For majority-vote bundling (statistical superposition preserving ratios),
/// callers use `ndarray::hpc::vsa::vsa_bundle` directly; this zero-dep
/// contract carries only the XOR form.
pub fn vsa_xor(a: &Vsa10k, b: &Vsa10k) -> Vsa10k {
    let mut out = [0u64; VSA_WORDS];
    for w in 0..VSA_WORDS {
        out[w] = a[w] ^ b[w];
    }
    out
}

/// Hamming similarity on the full vector in `[0, 1]` (matching bits / 10_000).
pub fn vsa_similarity(a: &Vsa10k, b: &Vsa10k) -> f32 {
    let matches = slice_matching_bits(a, b, 0, VSA_DIMS);
    matches as f32 / VSA_DIMS as f32
}

/// Bitmask selecting the bits of word `w` that lie in `[start..end)`.
/// Returns 0 if the word is entirely outside the slice.
#[inline]
fn word_slice_mask(w: usize, start: usize, end: usize) -> u64 {
    let lo = (w * 64).max(start);
    let hi = ((w + 1) * 64).min(end);
    if hi <= lo {
        return 0;
    }
    let lo_bit = lo - w * 64;
    let hi_bit = hi - w * 64;
    if hi_bit == 64 {
        !0u64 << lo_bit
    } else {
        ((1u64 << hi_bit) - 1) & !((1u64 << lo_bit) - 1)
    }
}

#[inline]
fn slice_matching_bits(
    a: &Vsa10k,
    b: &Vsa10k,
    start: usize,
    end: usize,
) -> u32 {
    if start >= end {
        return 0;
    }
    let first_word = start / 64;
    let last_word = (end - 1) / 64;
    let mut matches: u32 = 0;
    for w in first_word..=last_word {
        let mask = word_slice_mask(w, start, end);
        if mask == 0 {
            continue;
        }
        let differing = ((a[w] ^ b[w]) & mask).count_ones();
        let width = mask.count_ones();
        matches += width - differing;
    }
    matches
}

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

    // ── RoleKey-as-operator tests ──────────────────────────────────────

    fn mk_content(seed: u64) -> Vsa10k {
        let mut v = [0u64; VSA_WORDS];
        let mut s = seed.max(1);
        for w in 0..VSA_WORDS {
            s = s.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(w as u64 + 1);
            v[w] = s;
        }
        v
    }

    #[test]
    fn bind_unbind_is_self_inverse_within_slice() {
        // bind slice-masks content to [0..2000). unbind returns the
        // slice-masked content. The recovery margin inside the slice
        // must be exactly 1.0.
        let content = mk_content(0xDEAD_BEEF);
        let bound = SUBJECT_KEY.bind(&content);
        let recovered = SUBJECT_KEY.unbind(&bound);
        let m = SUBJECT_KEY.recovery_margin(&recovered, &content);
        assert!(
            (m - 1.0).abs() < 1e-6,
            "bind→unbind inside SUBJECT slice must recover, got margin {m}"
        );
    }

    #[test]
    fn bind_zero_outside_slice() {
        // bind's output is zero outside the slice regardless of content.
        let content = mk_content(0xFFFF_FFFF_FFFF_FFFF);
        let bound = OBJECT_KEY.bind(&content);
        for dim in 0..(VSA_WORDS * 64) {
            let in_slice = dim >= OBJECT_KEY.slice_start && dim < OBJECT_KEY.slice_end;
            let word = dim / 64;
            let offset = dim % 64;
            let bit = (bound[word] >> offset) & 1;
            if !in_slice {
                assert_eq!(
                    bit, 0,
                    "bind output nonzero outside OBJECT slice at dim {dim}"
                );
            }
        }
    }

    #[test]
    fn different_role_keys_do_not_contaminate_each_other() {
        // Bind S-content into SUBJECT slice, bind O-content into OBJECT slice,
        // XOR-superpose. Unbinding with SUBJECT_KEY recovers S-content
        // within the SUBJECT slice; unbinding with OBJECT_KEY recovers
        // O-content within the OBJECT slice. Slice-masking guarantees
        // disjoint superposition at the substrate level.
        let s_content = mk_content(0x1111);
        let o_content = mk_content(0x2222);

        let s_bound = SUBJECT_KEY.bind(&s_content);
        let o_bound = OBJECT_KEY.bind(&o_content);
        let superposed = vsa_xor(&s_bound, &o_bound);

        let s_recovered = SUBJECT_KEY.unbind(&superposed);
        let s_margin = SUBJECT_KEY.recovery_margin(&s_recovered, &s_content);
        assert!(
            (s_margin - 1.0).abs() < 1e-6,
            "SUBJECT recovery after XOR-superposed OBJECT must be exact, got {s_margin}"
        );

        let o_recovered = OBJECT_KEY.unbind(&superposed);
        let o_margin = OBJECT_KEY.recovery_margin(&o_recovered, &o_content);
        assert!(
            (o_margin - 1.0).abs() < 1e-6,
            "OBJECT recovery after XOR-superposed SUBJECT must be exact, got {o_margin}"
        );
    }

    #[test]
    fn recovery_margin_is_one_for_identical_content_within_slice() {
        let content = mk_content(0xABCD);
        let m = SUBJECT_KEY.recovery_margin(&content, &content);
        assert!((m - 1.0).abs() < 1e-6, "identical content must yield margin 1.0, got {m}");
    }

    #[test]
    fn cross_role_superposition_of_five_roles_all_recover() {
        // The real use case: bind content per role for S/P/O + TEMPORAL + LOKAL,
        // XOR-superpose into one trajectory. Each role recovers its own
        // content losslessly.
        let s = mk_content(0x1111);
        let p = mk_content(0x2222);
        let o = mk_content(0x3333);
        let t = mk_content(0x4444);
        let l = mk_content(0x5555);

        let bundle = [
            SUBJECT_KEY.bind(&s),
            PREDICATE_KEY.bind(&p),
            OBJECT_KEY.bind(&o),
            TEMPORAL_KEY.bind(&t),
            LOKAL_KEY.bind(&l),
        ]
        .into_iter()
        .fold([0u64; VSA_WORDS], |acc, b| vsa_xor(&acc, &b));

        for (key, original) in [
            (&*SUBJECT_KEY, s),
            (&*PREDICATE_KEY, p),
            (&*OBJECT_KEY, o),
            (&*TEMPORAL_KEY, t),
            (&*LOKAL_KEY, l),
        ] {
            let recovered = key.unbind(&bundle);
            let m = key.recovery_margin(&recovered, &original);
            assert!(
                (m - 1.0).abs() < 1e-6,
                "{} recovery from 5-role bundle must be exact, got {m}",
                key.label
            );
        }
    }

    #[test]
    fn recovery_margin_in_unit_interval() {
        let a = mk_content(0xAAAA_AAAA_AAAA_AAAA);
        let b = mk_content(0x5555_5555_5555_5555);
        for key in [&*SUBJECT_KEY, &*PREDICATE_KEY, &*OBJECT_KEY, &*LOKAL_KEY] {
            let m = key.recovery_margin(&a, &b);
            assert!(
                (0.0..=1.0).contains(&m),
                "recovery_margin for {} must be in [0, 1], got {m}",
                key.label
            );
        }
    }

    #[test]
    fn vsa_similarity_range() {
        let a = mk_content(0xF00D);
        let b = a;
        assert!((vsa_similarity(&a, &b) - 1.0).abs() < 1e-6);
        let c = [!0u64; VSA_WORDS];
        let zero = [0u64; VSA_WORDS];
        // Full-inverse similarity: 48 slack bits are 1 vs 0 on mismatch,
        // plus the 10_000 real dims are all-1 vs all-0. Sim over first 10_000
        // bits is 0.
        let sim = vsa_similarity(&c, &zero);
        assert!(sim < 0.01, "all-ones vs all-zeros similarity should be ~0, got {sim}");
    }

    #[test]
    fn vsa_xor_is_self_inverse() {
        let a = mk_content(0x1234);
        let b = mk_content(0x5678);
        let ab = vsa_xor(&a, &b);
        let recovered = vsa_xor(&ab, &b);
        assert_eq!(recovered, a, "XOR twice must recover original");
    }

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
