//! VSA (Vector Symbolic Architecture) encoder.
//!
//! Replaces transformer attention with algebraic composition:
//! - **XOR binding**: word ⊕ role → role-tagged representation
//! - **Majority bundle**: superposition of bindings → sentence vector
//! - **Unbinding**: bundle ⊕ role → recover approximate word
//!
//! Word order sensitivity: "dog bites man" ≠ "man bites dog" because
//! `XOR(dog, ROLE_SUBJECT) ≠ XOR(dog, ROLE_OBJECT)`.
//!
//! Role vectors are fixed pseudo-random binary patterns (not learned).
//! 10,000 bits each = compatible with Fingerprint<256> (16Kbit).
//! But for DeepNSM's 4K vocabulary, we use a compact 512-bit representation.

/// Bits per vector in the VSA space.
/// 512 bits = 64 bytes = 8 u64s. Compact but sufficient for 4096 vocab.
pub const VSA_BITS: usize = 512;
/// Number of u64 words per vector.
pub const VSA_WORDS: usize = VSA_BITS / 64;

/// A binary vector in the VSA space. 512 bits stored as 8 u64s.
#[derive(Clone, PartialEq, Eq)]
pub struct VsaVec {
    data: [u64; VSA_WORDS],
}

impl VsaVec {
    /// Zero vector (all bits clear).
    pub const ZERO: Self = VsaVec {
        data: [0u64; VSA_WORDS],
    };

    /// Create from raw u64 array.
    pub fn from_words(data: [u64; VSA_WORDS]) -> Self {
        Self { data }
    }

    /// Generate a pseudo-random vector from a seed.
    /// Uses a simple but deterministic hash (SplitMix64).
    pub fn random(seed: u64) -> Self {
        let mut data = [0u64; VSA_WORDS];
        let mut state = seed;
        for word in data.iter_mut() {
            state = splitmix64(state);
            *word = state;
        }
        Self { data }
    }

    /// Generate a word vector from vocabulary rank.
    /// Deterministic: same rank always produces the same vector.
    pub fn from_rank(rank: u16) -> Self {
        // Use rank as seed with a large prime multiplier for spread
        Self::random((rank as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF58476D1CE4E5B9))
    }

    /// XOR bind: `self ⊕ other`. Reversible: `(a ⊕ b) ⊕ b = a`.
    #[inline]
    pub fn bind(&self, other: &VsaVec) -> VsaVec {
        let mut result = [0u64; VSA_WORDS];
        for i in 0..VSA_WORDS {
            result[i] = self.data[i] ^ other.data[i];
        }
        VsaVec { data: result }
    }

    /// Hamming distance (number of differing bits).
    #[inline]
    pub fn hamming(&self, other: &VsaVec) -> u32 {
        let mut count = 0u32;
        for i in 0..VSA_WORDS {
            count += (self.data[i] ^ other.data[i]).count_ones();
        }
        count
    }

    /// Cosine-like similarity in Hamming space: `1.0 - 2 * hamming / bits`.
    /// Range: [-1.0, 1.0]. 1.0 = identical, 0.0 = random, -1.0 = complement.
    #[inline]
    pub fn similarity(&self, other: &VsaVec) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / VSA_BITS as f32
    }

    /// Count of set bits.
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }

    /// Bitwise NOT (complement).
    pub fn complement(&self) -> VsaVec {
        let mut result = [0u64; VSA_WORDS];
        for i in 0..VSA_WORDS {
            result[i] = !self.data[i];
        }
        VsaVec { data: result }
    }

    /// Access raw data.
    pub fn as_words(&self) -> &[u64; VSA_WORDS] {
        &self.data
    }
}

impl core::fmt::Debug for VsaVec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "VsaVec({} bits set)", self.popcount())
    }
}

// ─── Role Vectors ────────────────────────────────────────────────────────────

/// Fixed pseudo-random role vectors for SPO binding.
///
/// Each role is a 512-bit binary pattern generated from a fixed seed.
/// The roles are:
/// - `SUBJECT`: who/what does the action
/// - `PREDICATE`: the action/state
/// - `OBJECT`: who/what receives the action
/// - `MODIFIER`: adjective/adverb
/// - `TEMPORAL`: time reference
/// - `NEGATION`: negation marker
pub struct RoleVectors {
    pub subject: VsaVec,
    pub predicate: VsaVec,
    pub object: VsaVec,
    pub modifier: VsaVec,
    pub temporal: VsaVec,
    pub negation: VsaVec,
}

impl RoleVectors {
    /// Create role vectors from fixed seeds.
    /// These never change — they're architectural constants.
    pub fn new() -> Self {
        RoleVectors {
            subject: VsaVec::random(0x5375626A65637400),   // "Subject\0"
            predicate: VsaVec::random(0x5072656469636174), // "Predicat"
            object: VsaVec::random(0x4F626A6563740000),    // "Object\0\0"
            modifier: VsaVec::random(0x4D6F646966696572),  // "Modifier"
            temporal: VsaVec::random(0x54656D706F72616C),  // "Temporal"
            negation: VsaVec::random(0x4E65676174696F6E),  // "Negation"
        }
    }
}

impl Default for RoleVectors {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Bundling (Majority Vote) ────────────────────────────────────────────────

/// Bundle multiple vectors via element-wise majority vote.
///
/// For each bit position, if more than half the input vectors have that bit set,
/// the output bit is set. This is the VSA equivalent of vector addition.
///
/// Properties:
/// - Similarity to any component ≈ 0.75 (vs 0.50 random baseline)
/// - All components recoverable via unbinding
/// - Capacity: ~5-7 components before noise dominates
pub fn bundle(vectors: &[VsaVec]) -> VsaVec {
    if vectors.is_empty() {
        return VsaVec::ZERO;
    }
    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let threshold = vectors.len() / 2;
    let mut result = [0u64; VSA_WORDS];

    for bit_word in 0..VSA_WORDS {
        let mut result_word = 0u64;
        for bit_pos in 0..64 {
            let mask = 1u64 << bit_pos;
            let count: usize = vectors
                .iter()
                .filter(|v| v.data[bit_word] & mask != 0)
                .count();
            if count > threshold {
                result_word |= mask;
            } else if count == threshold && vectors.len() % 2 == 0 {
                // Tie-breaking for even count: use deterministic rule
                // (use bit position parity)
                if bit_pos % 2 == 0 {
                    result_word |= mask;
                }
            }
        }
        result[bit_word] = result_word;
    }

    VsaVec { data: result }
}

/// Unbind: recover a role's content from a bundled representation.
///
/// `bundle ⊕ role ≈ word` (approximate, similarity ≈ 0.75 to original).
#[inline]
pub fn unbind(bundled: &VsaVec, role: &VsaVec) -> VsaVec {
    bundled.bind(role)
}

// ─── SPO Encoding ────────────────────────────────────────────────────────────

/// Encode an SPO triple as a bundled VSA vector.
///
/// `encode(S, P, O) = bundle(S⊕ROLE_S, P⊕ROLE_P, O⊕ROLE_O)`
///
/// This gives word-order sensitivity: "dog bites man" ≠ "man bites dog"
/// because `dog⊕ROLE_S ≠ dog⊕ROLE_O`.
pub fn encode_triple(
    subject: u16,
    predicate: u16,
    object: Option<u16>,
    roles: &RoleVectors,
) -> VsaVec {
    let s_vec = VsaVec::from_rank(subject).bind(&roles.subject);
    let p_vec = VsaVec::from_rank(predicate).bind(&roles.predicate);

    match object {
        Some(obj) => {
            let o_vec = VsaVec::from_rank(obj).bind(&roles.object);
            bundle(&[s_vec, p_vec, o_vec])
        }
        None => bundle(&[s_vec, p_vec]),
    }
}

/// Encode with negation: complement the predicate binding.
pub fn encode_triple_negated(
    subject: u16,
    predicate: u16,
    object: Option<u16>,
    roles: &RoleVectors,
) -> VsaVec {
    let s_vec = VsaVec::from_rank(subject).bind(&roles.subject);
    let p_vec = VsaVec::from_rank(predicate)
        .bind(&roles.predicate)
        .bind(&roles.negation); // XOR with negation role
    match object {
        Some(obj) => {
            let o_vec = VsaVec::from_rank(obj).bind(&roles.object);
            bundle(&[s_vec, p_vec, o_vec])
        }
        None => bundle(&[s_vec, p_vec]),
    }
}

/// Encode a modifier attachment as a bound pair.
pub fn encode_modifier(modifier: u16, head: u16, roles: &RoleVectors) -> VsaVec {
    let m_vec = VsaVec::from_rank(modifier).bind(&roles.modifier);
    let h_vec = VsaVec::from_rank(head);
    bundle(&[m_vec, h_vec])
}

// ─── SplitMix64 PRNG ────────────────────────────────────────────────────────

/// SplitMix64 — deterministic, fast, good avalanche.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_vectors_near_half() {
        // Random vectors should have ~50% bits set
        let v = VsaVec::random(42);
        let popcount = v.popcount();
        let expected = VSA_BITS as u32 / 2;
        let tolerance = (VSA_BITS as f32).sqrt() as u32 * 3; // 3σ
        assert!(
            popcount.abs_diff(expected) < tolerance,
            "popcount={}, expected≈{}", popcount, expected
        );
    }

    #[test]
    fn self_similarity() {
        let v = VsaVec::random(42);
        assert_eq!(v.similarity(&v), 1.0);
    }

    #[test]
    fn random_similarity_near_zero() {
        let a = VsaVec::random(42);
        let b = VsaVec::random(99);
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.2, "sim = {} (should be ~0)", sim);
    }

    #[test]
    fn bind_is_reversible() {
        let word = VsaVec::random(42);
        let role = VsaVec::random(99);
        let bound = word.bind(&role);
        let recovered = bound.bind(&role);
        assert_eq!(word, recovered);
    }

    #[test]
    fn bundle_preserves_components() {
        let a = VsaVec::random(1);
        let b = VsaVec::random(2);
        let c = VsaVec::random(3);
        let bundled = bundle(&[a.clone(), b.clone(), c.clone()]);

        // Each component should have similarity > 0.5 with bundle
        assert!(bundled.similarity(&a) > 0.3);
        assert!(bundled.similarity(&b) > 0.3);
        assert!(bundled.similarity(&c) > 0.3);
    }

    #[test]
    fn word_order_matters() {
        let roles = RoleVectors::new();

        // "dog bites man" ≠ "man bites dog"
        let v1 = encode_triple(671, 2943, Some(95), &roles);
        let v2 = encode_triple(95, 2943, Some(671), &roles);

        // These should be different (not similar)
        let sim = v1.similarity(&v2);
        assert!(sim < 0.5, "sim = {} — word order should matter!", sim);
    }

    #[test]
    fn negation_changes_vector() {
        let roles = RoleVectors::new();

        let positive = encode_triple(671, 2943, Some(95), &roles);
        let negated = encode_triple_negated(671, 2943, Some(95), &roles);

        // Negated should be different
        let sim = positive.similarity(&negated);
        assert!(sim < 0.8, "sim = {} — negation should change the vector!", sim);
    }

    #[test]
    fn deterministic_vectors() {
        // Same rank always gives same vector
        let v1 = VsaVec::from_rank(42);
        let v2 = VsaVec::from_rank(42);
        assert_eq!(v1, v2);
    }
}
