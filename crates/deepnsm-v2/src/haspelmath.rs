//! Executable Haspelmath semantic-map probe at the DeepNSM-v2 / CE64 / hydration seam.
//!
//! This is a TEST HARNESS, not a universal grammar parser or an ontology mint.
//! Haspelmath (2003), "The Geometry of Grammatical Meaning", pp. 212-219:
//! grammatical functions are cross-linguistically distinguished; the proposed
//! map is a network; attested multi-function distributions are hypothesized to
//! occupy contiguous regions. Spatial positions and line lengths are NOT
//! semantic distances. See .arxiv/2026-10_haspelmath-deepnsm-ce64-harness.md.
//!
//! Critical contract: DeepNSM WordId is u16; CausalEdge64 S/P/O are u8.
//! Never truncate WordId. The CE64 projection below has a reversible, PER VERSE
//! local palette. The high 40 bits remain zero: syntax is not NARS confidence,
//! Pearl causation, evidence independence or epistemic maturity.

use crate::fsm::{parse_readings, PosSet, Reading};
use crate::hydrate::{hydrate, Hydration};
use crate::spo::Spo;
use crate::vocab::{PaletteVocab, WordId};
use causal_edge::CausalEdge64;
use lance_graph_contract::hhtl::NiblePath;
use std::collections::HashMap;

/// An explicitly named function from Haspelmath's Fig. 8.1 (p. 213).
/// Ordinals are harness-local mask coordinates, never a CE64 or ISA codebook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DativeFunction {
    Direction = 0,
    Recipient = 1,
    Experiencer = 2,
    Purpose = 3,
    Beneficiary = 4,
    PredicativePossessor = 5,
    ExternalPossessor = 6,
    Judicantis = 7,
}

impl DativeFunction {
    #[must_use]
    pub const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

// Undirected structural links as printed in Fig. 8.1.
// They are hypotheses about neighbors, NOT embedding distances or causal edges.
const NEIGHBORS: [u8; 8] = [
    (1 << 1) | (1 << 3),               // direction: recipient, purpose
    (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5),
    1 << 1,                             // experiencer: recipient
    1 << 0,                             // purpose: direction
    (1 << 1) | (1 << 6) | (1 << 7), // beneficiary
    (1 << 1) | (1 << 6),           // predicative possessor
    (1 << 4) | (1 << 5),           // external possessor
    1 << 4,                             // judicantis: beneficiary
];

/// Test continuity on the INDUCED graph, not along paths through unlicensed
/// functions. An incomplete sample cannot falsify the universal hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapVerdict {
    Connected,
    Disconnected,
    Incomplete,
}

#[must_use]
pub fn continuity(functions: u8, coverage_complete: bool) -> MapVerdict {
    if !coverage_complete {
        return MapVerdict::Incomplete;
    }
    if functions == 0 {
        return MapVerdict::Disconnected;
    }
    let mut seen = 1u8 << functions.trailing_zeros();
    loop {
        let mut next = seen;
        for i in 0..8 {
            if seen & (1 << i) != 0 {
                next |= NEIGHBORS[i] & functions;
            }
        }
        if next == seen {
            break;
        }
        seen = next;
    }
    if seen == functions {
        MapVerdict::Connected
    } else {
        MapVerdict::Disconnected
    }
}

/// One independently cited FUNCTION DISTRIBUTION for a grammatical form.
/// This is not a truth value; same evidence_root on two attestations is still
/// one underlying source. "Complete" must be warranted externally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormAttestation {
    pub word: WordId,
    pub functions: u8,
    pub coverage_complete: bool,
    pub evidence_root: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum HarnessError {
    UnalignedVerses,
    UnknownWord,
    MissingEvidenceRoot,
    DisconnectedCompleteDistribution,
    ParserOverflow,
    UnaddressedTriples(usize),
    LocalPaletteOverflow,
}

/// Bind surfaces to an existing vocabulary and supplied lexical reading sets.
/// Token segmentation, language-specific PoS decisions and morphology remain
/// upstream responsibilities (COCA/WordNet/UD or the appropriate parser).
pub fn bind_tokens(
    vocab: &PaletteVocab,
    annotated: &[(&str, PosSet)],
) -> Result<Vec<Reading>, HarnessError> {
    annotated
        .iter()
        .map(|(surface, pos)| {
            vocab
                .id(&surface.to_lowercase())
                .map(|id| Reading::new(id, *pos))
                .ok_or(HarnessError::UnknownWord)
        })
        .collect()
}

/// CE64's grammar-only payload has no truth/causality/witness claims.
/// The three indices refer to the per-path palette in HarnessOutput.
#[derive(Clone, Copy)]
pub struct AddressedSyntax {
    pub path: NiblePath,
    pub order: u32,
    pub word: CausalEdge64,
}

pub struct HarnessOutput {
    pub hydration: Hydration,
    pub encoded: Vec<AddressedSyntax>,
    pub local_palettes: HashMap<NiblePath, Vec<WordId>>,
    /// Uncertain triples stay associated with their exact input verse ordinal.
    /// They are NEVER passed as certain givens to hydrate().
    pub alternatives: Vec<(usize, Spo)>,
    pub attestations: Vec<(usize, FormAttestation, MapVerdict)>,
    pub ambiguous_tokens: usize,
}

impl HarnessOutput {
    /// Reconstruct u16 global vocabulary identity, using the SAME path as the
    /// edge. Treating CE64 low bytes as global WordIds would be lossy.
    #[must_use]
    pub fn decode(&self, edge: &AddressedSyntax) -> Option<Spo> {
        let palette = self.local_palettes.get(&edge.path)?;
        Some(Spo::new(
            *palette.get(edge.word.s_idx() as usize)?,
            *palette.get(edge.word.p_idx() as usize)?,
            *palette.get(edge.word.o_idx() as usize)?,
        ))
    }
}

/// Encode already-hydrated CERTAIN triples, with one reversible local u8
/// codebook per verse address. This is deliberately a reference implementation,
/// not a throughput claim: each lookup scans at most 256 entries.
/// An actual writer must persist the palette with its sealed-version identity.
fn encode_hydration(
    hydration: &Hydration,
) -> Result<(Vec<AddressedSyntax>, HashMap<NiblePath, Vec<WordId>>), HarnessError> {
    let mut palettes: HashMap<NiblePath, Vec<WordId>> = HashMap::new();
    let mut encoded = Vec::with_capacity(hydration.triples.len());
    for triple in &hydration.triples {
        let palette = palettes.entry(triple.path).or_default();
        let mut idx = [0u8; 3];
        for (i, id) in [
            triple.spo.subject,
            triple.spo.predicate,
            triple.spo.object,
        ]
        .into_iter()
        .enumerate()
        {
            let index = if let Some(existing) = palette.iter().position(|&x| x == id) {
                existing
            } else {
                if palette.len() == 256 {
                    return Err(HarnessError::LocalPaletteOverflow);
                }
                palette.push(id);
                palette.len() - 1
            };
            idx[i] = index as u8;
        }
        let mut word = CausalEdge64::ZERO;
        word.set_s_idx(idx[0]);
        word.set_p_idx(idx[1]);
        word.set_o_idx(idx[2]);
        encoded.push(AddressedSyntax {
            path: triple.path,
            order: triple.order,
            word,
        });
    }
    Ok((encoded, palettes))
}

/// Executable seam: known/readable tokens -> existing multi-reading FSM ->
/// CERTAIN SPO -> existing hydrate.rs -> reversible syntax-only CE64 view.
///
/// The semantic map checks ATTESTED form distributions, not token-level word
/// similarity. Neither a connected map nor a grammatical fit proves a causal
/// or NARS inference. Refuses mismatched verse walks and address loss.
pub fn run_harness(
    markers: &[(u16, u16)],
    verses: &[Vec<Reading>],
    form_evidence: &[Vec<FormAttestation>],
    basin: u8,
) -> Result<HarnessOutput, HarnessError> {
    if markers.len() != verses.len() || markers.len() != form_evidence.len() {
        return Err(HarnessError::UnalignedVerses);
    }
    let mut attestations = Vec::new();
    for (verse, forms) in form_evidence.iter().enumerate() {
        for &record in forms {
            if record.evidence_root == 0 {
                return Err(HarnessError::MissingEvidenceRoot);
            }
            let verdict = continuity(record.functions, record.coverage_complete);
            if verdict == MapVerdict::Disconnected {
                return Err(HarnessError::DisconnectedCompleteDistribution);
            }
            attestations.push((verse, record, verdict));
        }
    }
    let mut certain = Vec::with_capacity(verses.len());
    let mut alternatives = Vec::new();
    let mut ambiguous_tokens = 0;
    for (verse, tokens) in verses.iter().enumerate() {
        let mut input = tokens.clone();
        input.push(Reading::stop());
        let parsed = parse_readings(&input);
        if parsed.overflow_flushes != 0 {
            return Err(HarnessError::ParserOverflow);
        }
        ambiguous_tokens += parsed.ambiguous.len();
        alternatives.extend(parsed.alternative.into_iter().map(|spo| (verse, spo)));
        certain.push(parsed.certain);
    }
    let hydration = hydrate(markers, &certain, basin);
    if hydration.unaddressed != 0 {
        return Err(HarnessError::UnaddressedTriples(hydration.unaddressed));
    }
    let (encoded, local_palettes) = encode_hydration(&hydration)?;
    Ok(HarnessOutput {
        hydration,
        encoded,
        local_palettes,
        alternatives,
        attestations,
        ambiguous_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::Pos;

    fn one(id: WordId, pos: Pos) -> Reading {
        Reading::new(id, PosSet::single(pos))
    }

    #[test]
    fn haspelmath_to_and_zu_are_connected_but_unrelated_pair_is_not() {
        use DativeFunction as F;
        let english_to = F::Direction.bit()
            | F::Recipient.bit()
            | F::Purpose.bit()
            | F::Experiencer.bit();
        assert_eq!(continuity(english_to, true), MapVerdict::Connected);
        assert_eq!(
            continuity(F::Direction.bit() | F::Purpose.bit(), true),
            MapVerdict::Connected
        );
        assert_eq!(
            continuity(F::Purpose.bit() | F::Judicantis.bit(), true),
            MapVerdict::Disconnected
        );
        assert_eq!(
            continuity(F::Purpose.bit() | F::Judicantis.bit(), false),
            MapVerdict::Incomplete
        );
    }

    #[test]
    fn real_fsm_then_hydrate_then_ce64_roundtrips_global_ids() {
        let forms = vec![vec![FormAttestation {
            word: 42,
            functions: DativeFunction::Direction.bit() | DativeFunction::Purpose.bit(),
            coverage_complete: true,
            evidence_root: 0xABC,
        }]];
        let v = vec![vec![
            one(0x0101, Pos::Noun),
            one(0x1234, Pos::Verb),
            one(0xFFFF, Pos::Noun),
        ]];
        let output = run_harness(&[(1, 1)], &v, &forms, 3).unwrap();
        assert_eq!(output.hydration.unaddressed, 0);
        assert_eq!(output.encoded.len(), 1);
        assert_eq!(output.hydration.triples.len(), 1);
        let e = &output.encoded[0];
        assert_eq!(output.decode(e), Some(Spo::new(0x0101, 0x1234, 0xFFFF)));
        assert_eq!(e.order, output.hydration.triples[0].order);
        assert_eq!(e.path, output.hydration.triples[0].path);
        assert_eq!(e.word.0 >> 24, 0, "no fabricated truth, Pearl or maturity");
    }

    #[test]
    fn incomplete_map_can_be_stored_as_unresolved_evidence() {
        let att = FormAttestation {
            word: 7,
            functions: DativeFunction::Purpose.bit() | DativeFunction::Judicantis.bit(),
            coverage_complete: false,
            evidence_root: 1,
        };
        let r = run_harness(&[(1, 1)], &[vec![]], &[vec![att]], 1).unwrap();
        assert_eq!(r.attestations[0].2, MapVerdict::Incomplete);
        assert!(r.encoded.is_empty());
        assert_eq!(r.hydration.verse_nodes(), 1);
        assert_eq!(
            run_harness(
                &[(1, 1)],
                &[vec![]],
                &[vec![FormAttestation { coverage_complete: true, ..att }]],
                1,
            ).err(),
            Some(HarnessError::DisconnectedCompleteDistribution)
        );
    }

    #[test]
    fn ambiguity_is_retained_not_minted_as_a_given() {
        let tokens = vec![
            one(9, Pos::Det),
            one(1, Pos::Noun),
            Reading::new(2, PosSet::single(Pos::Noun).with(Pos::Verb)),
            one(3, Pos::Noun),
        ];
        let mut with_stop = tokens.clone();
        with_stop.push(Reading::stop());
        let oracle = parse_readings(&with_stop);
        let r = run_harness(&[(1, 1)], &[tokens], &[vec![]], 2).unwrap();
        assert_eq!(r.ambiguous_tokens, oracle.ambiguous.len());
        assert_eq!(r.alternatives, oracle.alternative.into_iter().map(|s| (0, s)).collect::<Vec<_>>());
        assert_eq!(r.hydration.triples.iter().map(|t| t.spo).collect::<Vec<_>>(), oracle.certain);
    }

    #[test]
    fn local_palette_refuses_overflow_instead_of_truncating() {
        let triples: Vec<Spo> = (0..86u16)
            .map(|i| Spo::new(i * 3, i * 3 + 1, i * 3 + 2))
            .collect();
        let h = hydrate(&[(1, 1)], &[triples], 1);
        assert_eq!(
            encode_hydration(&h).err(),
            Some(HarnessError::LocalPaletteOverflow)
        );
    }

    #[test]
    fn refuses_unaligned_walk_and_missing_evidence_provenance() {
        assert_eq!(
            run_harness(&[(1, 1)], &[], &[], 1).err(),
            Some(HarnessError::UnalignedVerses)
        );
        let forms = vec![vec![FormAttestation {
            word: 1,
            functions: DativeFunction::Recipient.bit(),
            coverage_complete: true,
            evidence_root: 0,
        }]];
        assert_eq!(
            run_harness(&[(1, 1)], &[vec![]], &forms, 1).err(),
            Some(HarnessError::MissingEvidenceRoot)
        );
    }
}
