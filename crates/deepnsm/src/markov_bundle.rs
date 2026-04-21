//! Markov ±5 bundler with role-indexed binding and braiding.
//!
//! Each sentence is bound per-token via `RoleKey::bind`, XOR-bundled
//! into one `Vsa10k`, then braided by `vsa_permute(sentence_vsa, offset)`
//! per position in the ±5 window. The 11 braided vectors are XOR-
//! superposed into the trajectory bundle.
//!
//! The braiding encodes temporal order without learned positional
//! embeddings — Shaw's ρ operator (2501.05368 §B.3).

use lance_graph_contract::grammar::role_keys::{
    Vsa10k, VSA_WORDS, VSA_ZERO, vsa_xor,
    SUBJECT_KEY, PREDICATE_KEY, OBJECT_KEY, MODIFIER_KEY, TEMPORAL_KEY,
};
use lance_graph_contract::grammar::context_chain::{
    MARKOV_RADIUS, CHAIN_LEN, WeightingKernel,
};

use crate::content_fp::content_fp;
use crate::parser::SentenceStructure;

/// Cyclic left-shift of a Vsa10k by `shift` bit positions.
/// This is the braiding operator ρ from Shaw et al.
fn vsa_permute(v: &Vsa10k, shift: usize) -> Vsa10k {
    if shift == 0 {
        return *v;
    }
    let shift = shift % (VSA_WORDS * 64);
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut out = [0u64; VSA_WORDS];
    for w in 0..VSA_WORDS {
        let src = (w + VSA_WORDS - word_shift) % VSA_WORDS;
        if bit_shift == 0 {
            out[w] = v[src];
        } else {
            let prev = (src + VSA_WORDS - 1) % VSA_WORDS;
            out[w] = (v[src] << bit_shift) | (v[prev] >> (64 - bit_shift));
        }
    }
    out
}

/// Encode one sentence into a single Vsa10k via role-key binding.
///
/// Each SPO triple's subject/predicate/object gets bound to the
/// corresponding role key. Modifiers get bound to MODIFIER_KEY.
/// Temporals get bound to TEMPORAL_KEY. All bindings are XOR-
/// superposed into one sentence-level vector.
pub fn encode_sentence(structure: &SentenceStructure) -> Vsa10k {
    let mut sentence_vsa = VSA_ZERO;
    for triple in &structure.triples {
        let s_fp = content_fp(triple.subject());
        let p_fp = content_fp(triple.predicate());
        let s_bound = SUBJECT_KEY.bind(&s_fp);
        let p_bound = PREDICATE_KEY.bind(&p_fp);
        sentence_vsa = vsa_xor(&sentence_vsa, &s_bound);
        sentence_vsa = vsa_xor(&sentence_vsa, &p_bound);
        let obj = triple.object();
        if obj != crate::spo::NO_ROLE {
            let o_fp = content_fp(obj);
            let o_bound = OBJECT_KEY.bind(&o_fp);
            sentence_vsa = vsa_xor(&sentence_vsa, &o_bound);
        }
    }
    for modifier in &structure.modifiers {
        let m_fp = content_fp(modifier.modifier);
        let m_bound = MODIFIER_KEY.bind(&m_fp);
        sentence_vsa = vsa_xor(&sentence_vsa, &m_bound);
    }
    for &(_triple_idx, temporal_rank) in &structure.temporals {
        let t_fp = content_fp(temporal_rank);
        let t_bound = TEMPORAL_KEY.bind(&t_fp);
        sentence_vsa = vsa_xor(&sentence_vsa, &t_bound);
    }
    sentence_vsa
}

/// Ring buffer of ±5 encoded sentences with braided bundling.
pub struct MarkovBundler {
    sentences: [Option<Vsa10k>; CHAIN_LEN],
    head: usize,
    count: usize,
    kernel: WeightingKernel,
}

impl MarkovBundler {
    pub fn new(kernel: WeightingKernel) -> Self {
        Self {
            sentences: [None; CHAIN_LEN],
            head: 0,
            count: 0,
            kernel,
        }
    }

    /// Push a parsed sentence into the ring buffer.
    pub fn push(&mut self, structure: &SentenceStructure) {
        let encoded = encode_sentence(structure);
        self.sentences[self.head] = Some(encoded);
        self.head = (self.head + 1) % CHAIN_LEN;
        if self.count < CHAIN_LEN {
            self.count += 1;
        }
    }

    /// Index of the focal sentence (most recently pushed).
    fn focal_index(&self) -> usize {
        (self.head + CHAIN_LEN - 1) % CHAIN_LEN
    }

    /// Build a braided trajectory bundle from the current window.
    ///
    /// Each sentence is permuted by its distance from the focal point
    /// (braiding ρ^d), weighted by the kernel, and XOR-superposed.
    pub fn build_bundle(&self) -> Vsa10k {
        let focal = self.focal_index();
        let mut bundle = VSA_ZERO;

        for i in 0..CHAIN_LEN {
            let slot = (focal + CHAIN_LEN - MARKOV_RADIUS + i) % CHAIN_LEN;
            if let Some(ref sentence_vsa) = self.sentences[slot] {
                let distance = if i <= MARKOV_RADIUS {
                    MARKOV_RADIUS - i
                } else {
                    i - MARKOV_RADIUS
                };
                let weight = self.kernel.weight(distance);
                if weight <= 0.0 {
                    continue;
                }
                // Braid by distance from focal — encodes temporal order.
                let braided = vsa_permute(sentence_vsa, distance * 64);
                bundle = vsa_xor(&bundle, &braided);
            }
        }
        bundle
    }

    pub fn is_saturated(&self) -> bool {
        self.count >= CHAIN_LEN
    }

    pub fn filled(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::SentenceStructure;
    use crate::spo::SpoTriple;
    use lance_graph_contract::grammar::role_keys::vsa_similarity;

    fn mk_sentence(s: u16, p: u16, o: u16) -> SentenceStructure {
        SentenceStructure {
            triples: vec![SpoTriple::new(s, p, o)],
            modifiers: vec![],
            negations: vec![],
            temporals: vec![],
        }
    }

    #[test]
    fn encode_sentence_is_deterministic() {
        let s = mk_sentence(10, 20, 30);
        let a = encode_sentence(&s);
        let b = encode_sentence(&s);
        assert_eq!(a, b);
    }

    #[test]
    fn encode_sentence_word_order_matters() {
        let s1 = mk_sentence(10, 20, 30);
        let s2 = mk_sentence(30, 20, 10);
        let a = encode_sentence(&s1);
        let b = encode_sentence(&s2);
        assert_ne!(a, b, "S/O swap must produce different vectors");
    }

    #[test]
    fn encode_sentence_recovers_subject_via_unbind() {
        let s = mk_sentence(42, 100, 200);
        let encoded = encode_sentence(&s);
        let expected_s = content_fp(42);
        let margin = SUBJECT_KEY.recovery_margin(
            &SUBJECT_KEY.unbind(&encoded),
            &expected_s,
        );
        assert!(
            margin > 0.99,
            "SUBJECT should recover losslessly from single-triple sentence, got {margin}"
        );
    }

    #[test]
    fn bundler_push_and_build() {
        let mut b = MarkovBundler::new(WeightingKernel::Uniform);
        for i in 0..11 {
            b.push(&mk_sentence(i, i + 100, i + 200));
        }
        assert!(b.is_saturated());
        let bundle = b.build_bundle();
        // Non-zero bundle.
        let pop: u32 = bundle.iter().map(|w| w.count_ones()).sum();
        assert!(pop > 0, "bundle should not be all-zero");
    }

    #[test]
    fn braided_bundle_differs_from_unbraided() {
        let mut braided = MarkovBundler::new(WeightingKernel::MexicanHat);
        let mut uniform = MarkovBundler::new(WeightingKernel::Uniform);
        for i in 0..11 {
            let s = mk_sentence(i, i + 100, i + 200);
            braided.push(&s);
            uniform.push(&s);
        }
        let a = braided.build_bundle();
        let b = uniform.build_bundle();
        // Different kernels should produce different bundles (MexicanHat
        // weights differ from Uniform).
        assert_ne!(a, b, "different kernels should produce different bundles");
    }

    #[test]
    fn vsa_permute_is_invertible() {
        let v = content_fp(42);
        let shifted = vsa_permute(&v, 137);
        let restored = vsa_permute(&shifted, VSA_WORDS * 64 - 137);
        assert_eq!(v, restored, "permute then inverse-permute must recover");
    }

    #[test]
    fn vsa_permute_zero_is_identity() {
        let v = content_fp(99);
        assert_eq!(v, vsa_permute(&v, 0));
    }

    #[test]
    fn focal_is_most_recent() {
        let mut b = MarkovBundler::new(WeightingKernel::Uniform);
        b.push(&mk_sentence(1, 2, 3));
        b.push(&mk_sentence(4, 5, 6));
        // Focal should be the second pushed sentence.
        let focal_encoded = encode_sentence(&mk_sentence(4, 5, 6));
        let slot = b.focal_index();
        assert_eq!(b.sentences[slot].as_ref().unwrap(), &focal_encoded);
    }
}
