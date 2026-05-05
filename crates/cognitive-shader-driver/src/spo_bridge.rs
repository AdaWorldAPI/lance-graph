//! `spo_bridge` — canonical **3D** SPO triple → BindSpace projection.
//!
//! Every input format reduces to a stream of `(S, P, O)` triples + a
//! per-triple [`SpoWitness`]. This module is the single place that stream
//! becomes BindSpace SoA rows. The reducer is intentionally small: callers
//! do their own string→id work, choose a witness flavour (`asserted` /
//! `literal` / `derived` / `scenario`), and call [`project_into`].
//!
//! ## Dimensional position
//!
//! ```text
//! 1D Redis tree (branch:twig:leaf) ──┐
//! 2D SQL row / blasgraph cell ────────┤
//! 3D Cypher / GQL / SPARQL / Gremlin ─┤── stream of (S, P, O) ──▶ project_into
//! 3D NARS triplet <a-->b>. %f; c%  ───┤                              │
//! DeepNSM Markov ±5 (centre, MARKOV, neighbour) ────┘                ▼
//!                                                              BindSpace SoA row
//!                                                                    │
//! 4D cognitive shader sweeps row, emits cycle_fingerprint ◀──────────┘
//! 5D Gaussian splat hydration writes scenario triples back through this same reducer
//! ```
//!
//! ## Reused canonical types — no parallel definitions
//!
//! - [`MetaWord`] / [`MetaFilter`] from `lance_graph_contract::cognitive_shader`
//!   pack `thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6)`.
//! - [`InferenceType`] from `lance_graph_contract::nars` carries the NARS
//!   reasoning kind. Stored on the witness for downstream shader routing.
//! - [`BindSpace`] SoA columns from [`crate::bindspace`].
//!
//! ## Fingerprint expansion convention
//!
//! Each u32 id deterministically expands to `[u64; 256]` (16 K bits) via
//! splitmix64 stream seeded by the id. Three planes are written per row:
//!
//! - `topic`   = expand(subject)
//! - `angle`   = expand(predicate)
//! - `content` = expand(subject) XOR expand(object)
//!
//! The `cycle` plane is **not** written here — it is emitted by the
//! cognitive shader during dispatch, not during ingestion.

use crate::bindspace::{BindSpace, WORDS_PER_FP};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::nars::InferenceType;

/// Subject-Predicate-Object triple identified by host-assigned u32 ids.
///
/// String → id mapping is the host's responsibility (catalog, ontology
/// resolver, label registry, ...). The reducer only needs three integers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpoTriple {
    pub subject: u32,
    pub predicate: u32,
    pub object: u32,
}

/// Per-triple witness — NARS truth, thinking style, edge bits, temporal
/// position, expert/relation index, entity type, and the inference kind
/// that produced the triple.
///
/// Use the const constructors ([`literal`](SpoWitness::literal),
/// [`asserted`](SpoWitness::asserted), [`derived`](SpoWitness::derived),
/// [`scenario`](SpoWitness::scenario)) and override individual fields
/// rather than constructing from scratch — the defaults document the
/// intended truth/confidence semantics.
#[derive(Copy, Clone, Debug)]
pub struct SpoWitness {
    /// Quantized NARS frequency in [0, 255]. 128 ≈ 0.5.
    pub nars_freq: u8,
    /// Quantized NARS confidence in [0, 255]. 0 = no evidence; 255 = literal.
    pub nars_conf: u8,
    /// Thinking style ordinal in [0, 63]. 0 = systematic.
    pub thinking: u8,
    /// Awareness rung in [0, 15]. 0 = surface.
    pub awareness: u8,
    /// Free-energy class in [0, 63]. 0 = no constraint.
    pub free_energy: u8,
    /// CausalEdge64 packed bits.
    pub edge: u64,
    /// Cycle / sequence position. 0 = atemporal.
    pub temporal: u64,
    /// Expert / relation index.
    pub expert: u16,
    /// Foundry Object Type instance (0 = untyped).
    pub entity_type: u16,
    /// NARS inference kind that produced this triple.
    pub inference: InferenceType,
}

impl SpoWitness {
    /// "Literal fact" — direct insert (Cypher CREATE, raw SPO insert,
    /// 1D Redis tree leaf). Frequency and confidence both saturated.
    pub const fn literal() -> Self {
        Self {
            nars_freq: 255,
            nars_conf: 255,
            thinking: 0,
            awareness: 0,
            free_energy: 0,
            edge: 0,
            temporal: 0,
            expert: 0,
            entity_type: 0,
            inference: InferenceType::Deduction,
        }
    }

    /// "Asserted by source" — Cypher MATCH default, SPARQL triple in a
    /// trusted dataset, GQL pattern match. Frequency ≈ 0.9, confidence ≈ 0.5.
    pub const fn asserted() -> Self {
        Self {
            nars_freq: 230,
            nars_conf: 128,
            thinking: 0,
            awareness: 0,
            free_energy: 0,
            edge: 0,
            temporal: 0,
            expert: 0,
            entity_type: 0,
            inference: InferenceType::Deduction,
        }
    }

    /// "Derived from inference" — output of NARS deduction / induction /
    /// abduction. Frequency ≈ 0.5, confidence ≈ 0.25 (room to grow with
    /// more evidence via Revision).
    pub const fn derived() -> Self {
        Self {
            nars_freq: 128,
            nars_conf: 64,
            thinking: 0,
            awareness: 0,
            free_energy: 0,
            edge: 0,
            temporal: 0,
            expert: 0,
            entity_type: 0,
            inference: InferenceType::Induction,
        }
    }

    /// "Scenario / forecast" — 5D Gaussian splat hydration, Chronos
    /// projection, counterfactual branch. Confidence intentionally low
    /// so promotion gates reject without explicit re-confirmation.
    pub const fn scenario() -> Self {
        Self {
            nars_freq: 128,
            nars_conf: 16,
            thinking: 0,
            awareness: 0,
            free_energy: 0,
            edge: 0,
            temporal: 0,
            expert: 0,
            entity_type: 0,
            inference: InferenceType::Synthesis,
        }
    }
}

/// Project one SPO triple + witness into row `row` of `bs`.
///
/// Writes:
/// - `fingerprints.topic[row]`   ← `expand(subject)`
/// - `fingerprints.angle[row]`   ← `expand(predicate)`
/// - `fingerprints.content[row]` ← `expand(subject)` XOR `expand(object)`
/// - `meta[row]` ← packed `MetaWord(thinking, awareness, nars_freq, nars_conf, free_energy)`
/// - `edges[row]` ← `witness.edge`
/// - `temporal[row]` ← `witness.temporal`
/// - `expert[row]` ← `witness.expert`
/// - `entity_type[row]` ← `witness.entity_type`
///
/// The `cycle` plane and `qualia` column are **not** touched — those are
/// emitted by the cognitive shader during dispatch, not during ingestion.
///
/// `witness.inference` is currently advisory: it is preserved for callers
/// that want to log it alongside the row (no BindSpace column is dedicated
/// to it yet). Future revisions may pack it into the free-energy / expert
/// fields once the convention stabilises.
///
/// # Panics
/// Panics if `row >= bs.len`.
pub fn project_into(bs: &mut BindSpace, row: usize, triple: SpoTriple, witness: SpoWitness) {
    assert!(
        row < bs.len,
        "spo_bridge::project_into row {row} out of range for BindSpace len {}",
        bs.len
    );

    // 1. Fingerprint planes — deterministic splitmix64 expansion.
    let s_fp = expand_to_fingerprint(triple.subject as u64);
    let p_fp = expand_to_fingerprint(triple.predicate as u64);
    let o_fp = expand_to_fingerprint(triple.object as u64);

    let mut content = [0u64; WORDS_PER_FP];
    for (i, w) in content.iter_mut().enumerate() {
        *w = s_fp[i] ^ o_fp[i];
    }
    bs.fingerprints.set_content(row, &content);

    let topic_offset = row * WORDS_PER_FP;
    bs.fingerprints.topic[topic_offset..topic_offset + WORDS_PER_FP]
        .copy_from_slice(&s_fp);
    bs.fingerprints.angle[topic_offset..topic_offset + WORDS_PER_FP]
        .copy_from_slice(&p_fp);

    // 2. Packed metadata word — the cheapest prefilter.
    let meta = MetaWord::new(
        witness.thinking,
        witness.awareness,
        witness.nars_freq,
        witness.nars_conf,
        witness.free_energy,
    );
    bs.meta.set(row, meta);

    // 3. CausalEdge64 + temporal + expert + entity_type.
    bs.edges.set(row, witness.edge);
    bs.temporal[row] = witness.temporal;
    bs.expert[row] = witness.expert;
    bs.entity_type[row] = witness.entity_type;

    // qualia and cycle planes intentionally untouched — they are emitted
    // by the cognitive shader during dispatch, not during ingestion.
}

/// Deterministic 256×u64 fingerprint from a 64-bit seed via splitmix64.
///
/// No allocations beyond the array, well-distributed across words. The
/// `topic` plane uses subject id as seed; `angle` uses predicate; the
/// `content` plane XORs subject and object expansions.
pub fn expand_to_fingerprint(seed: u64) -> [u64; WORDS_PER_FP] {
    let mut out = [0u64; WORDS_PER_FP];
    let mut state = seed;
    for w in out.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        *w = z;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindspace::{BindSpace, FLOATS_PER_VSA, QUALIA_DIMS};

    #[test]
    fn project_writes_all_fingerprint_planes() {
        let mut bs = BindSpace::zeros(4);
        project_into(
            &mut bs,
            1,
            SpoTriple { subject: 100, predicate: 200, object: 300 },
            SpoWitness::asserted(),
        );

        // All three planes should be non-zero (splitmix64 expansion).
        assert!(bs.fingerprints.content_row(1).iter().any(|&w| w != 0));
        assert!(bs.fingerprints.topic_row(1).iter().any(|&w| w != 0));
        assert!(bs.fingerprints.angle_row(1).iter().any(|&w| w != 0));
    }

    #[test]
    fn content_plane_equals_topic_xor_object_expansion() {
        let mut bs = BindSpace::zeros(2);
        project_into(
            &mut bs,
            0,
            SpoTriple { subject: 7, predicate: 13, object: 19 },
            SpoWitness::asserted(),
        );

        let topic = bs.fingerprints.topic_row(0).to_vec();
        let object_expansion = expand_to_fingerprint(19);
        let content = bs.fingerprints.content_row(0);
        for i in 0..WORDS_PER_FP {
            assert_eq!(content[i], topic[i] ^ object_expansion[i]);
        }
    }

    #[test]
    fn project_packs_meta_word_with_witness_fields() {
        let mut bs = BindSpace::zeros(2);
        let mut witness = SpoWitness::literal();
        witness.thinking = 31;
        witness.awareness = 5;
        witness.free_energy = 10;
        project_into(
            &mut bs,
            0,
            SpoTriple { subject: 1, predicate: 2, object: 3 },
            witness,
        );
        let meta = bs.meta.get(0);
        assert_eq!(meta.thinking(), 31);
        assert_eq!(meta.awareness(), 5);
        assert_eq!(meta.nars_f(), 255);
        assert_eq!(meta.nars_c(), 255);
        assert_eq!(meta.free_e(), 10);
    }

    #[test]
    fn project_writes_temporal_expert_edge_entity_type() {
        let mut bs = BindSpace::zeros(3);
        let mut witness = SpoWitness::asserted();
        witness.edge = 0xDEAD_BEEF_CAFE_BABE;
        witness.temporal = 12345;
        witness.expert = 7;
        witness.entity_type = 42;
        project_into(
            &mut bs,
            2,
            SpoTriple { subject: 1, predicate: 2, object: 3 },
            witness,
        );
        assert_eq!(bs.edges.get(2), 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(bs.temporal[2], 12345);
        assert_eq!(bs.expert[2], 7);
        assert_eq!(bs.entity_type[2], 42);
    }

    #[test]
    fn project_does_not_touch_other_rows() {
        let mut bs = BindSpace::zeros(4);
        project_into(
            &mut bs,
            1,
            SpoTriple { subject: 1, predicate: 2, object: 3 },
            SpoWitness::asserted(),
        );
        for row in [0usize, 2, 3] {
            assert!(
                bs.fingerprints.content_row(row).iter().all(|&w| w == 0),
                "content row {row} should be untouched"
            );
            assert!(
                bs.fingerprints.topic_row(row).iter().all(|&w| w == 0),
                "topic row {row} should be untouched"
            );
            assert!(
                bs.fingerprints.angle_row(row).iter().all(|&w| w == 0),
                "angle row {row} should be untouched"
            );
            assert_eq!(bs.meta.get(row), MetaWord::default(), "meta row {row}");
            assert_eq!(bs.edges.get(row), 0, "edges row {row}");
            assert_eq!(bs.temporal[row], 0, "temporal row {row}");
        }
    }

    #[test]
    fn project_does_not_touch_cycle_or_qualia() {
        let mut bs = BindSpace::zeros(2);
        project_into(
            &mut bs,
            0,
            SpoTriple { subject: 1, predicate: 2, object: 3 },
            SpoWitness::literal(),
        );
        // cycle plane (Vsa16kF32 carrier) stays zeros — only the shader
        // emits cycle_fingerprint.
        assert!(bs.fingerprints.cycle_row(0).iter().all(|&v| v == 0.0));
        // qualia stays zeros — callers fill via separate write if they
        // have a vector.
        for q in bs.qualia.row(0) {
            assert_eq!(*q, 0.0);
        }
        assert_eq!(bs.fingerprints.cycle.len(), 2 * FLOATS_PER_VSA);
        assert_eq!(bs.qualia.0.len(), 2 * QUALIA_DIMS);
    }

    #[test]
    fn expand_is_deterministic() {
        let a = expand_to_fingerprint(42);
        let b = expand_to_fingerprint(42);
        assert_eq!(a, b);
    }

    #[test]
    fn expand_diverges_for_adjacent_seeds() {
        let a = expand_to_fingerprint(42);
        let b = expand_to_fingerprint(43);
        assert_ne!(a, b);
        let differ = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        // splitmix64 should diversify almost every word for adjacent seeds.
        assert!(
            differ > 240,
            "splitmix64 expansion should diversify across words ({differ}/256 differed)"
        );
    }

    #[test]
    fn witness_truth_flavours_have_expected_quantiles() {
        // Sanity: literal saturates, derived sits at midpoint frequency
        // with low confidence, scenario stays low-confidence so promotion
        // gates can reject it.
        assert_eq!(SpoWitness::literal().nars_freq, 255);
        assert_eq!(SpoWitness::literal().nars_conf, 255);
        assert!(SpoWitness::asserted().nars_freq > 200);
        assert!(SpoWitness::asserted().nars_conf < 200);
        assert_eq!(SpoWitness::derived().nars_freq, 128);
        assert!(SpoWitness::derived().nars_conf < SpoWitness::asserted().nars_conf);
        assert!(SpoWitness::scenario().nars_conf < SpoWitness::derived().nars_conf);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn project_panics_on_oob_row() {
        let mut bs = BindSpace::zeros(2);
        project_into(
            &mut bs,
            5,
            SpoTriple { subject: 1, predicate: 2, object: 3 },
            SpoWitness::asserted(),
        );
    }
}
