//! The **second codebook**: the OBO Ontology vocabulary, alongside COCA.
//!
//! DeepNSM's own [`Vocabulary`](crate::vocabulary::Vocabulary) is the academic
//! frequency codebook — COCA ranks in a 12-bit space. This module is the other
//! one a reasoning caller needs: the public OBO biomedical reference
//! (`ConceptDomain::Ontology`, `0x03XX`), read through the contract's zero-dep
//! wire mirror.
//!
//! # No new dependency, and that is the point
//!
//! `lance-graph-contract` is already a hard dep of this crate (the canonical
//! `RoleKeySlice` constants), so reaching the ontology costs nothing here. The
//! alternative — deping the producer `ogar-obo` — is what the plug-and-play
//! posture exists to avoid, and it would pull an OBO bake into a crate whose
//! job is distributional semantics.
//!
//! # Two codebooks, two address spaces — never silently merged
//!
//! | codebook | address | width | source |
//! |---|---|---|---|
//! | COCA | frequency rank | 12-bit (`VOCAB_SIZE`) | [`crate::vocabulary`] |
//! | Ontology | canonical concept id | 16-bit (`0x03XX`) | the OGAR mint, mirrored |
//!
//! A rank and a concept id are not interchangeable and this module does not
//! offer a conversion. Fusing them into one integer space would make
//! `rank == concept` collisions unnoticeable, and the two spaces have entirely
//! different owners: a rank moves when the corpus is re-counted, a concept id
//! is minted once and never moves.
//!
//! # Why this could not be written before 2026-08-22
//!
//! `concepts_in_domain(ConceptDomain::Ontology)` returned **empty**. The OBO
//! concept ids lived only in the producer crates, so the shared codebook had
//! nothing in that domain — and an empty enumeration reads exactly like "this
//! domain has nothing to reason about". The operator's ruling that the domains
//! are minted in `ogar-vocab` is what gave this module something to return.
//!
//! # WordNet: the same law, a second register (restored 2026-10-04)
//!
//! [`WordNetRail`] reads the WordNet 3.1 is-a rail (`wordnet31_isa_v2.tsv`,
//! codebook release `v0.1.0-codebooks-2026-07-26`, WordNet License). A synset
//! offset is an IDENTITY exactly like an ontology concept id: minted by
//! WordNet, never moved, with no distance semantics. So the register offers
//! lookup, ancestry and membership — `is_a` and [`ConceptMask`] — and no
//! distance of any kind. That is the line `48405aa2` drew (CAM-PQ is
//! prohibited for ontologies), kept here: the hypernym chain is a parent/child
//! ADDRESS (the HHTL reading), never a metric.
//!
//! A mask is a set of synsets; a sense is covered when it or any ancestor is in
//! the set. That expresses categories that cut ACROSS the taxonomy — water
//! animals = {aquatic_mammal, aquatic_vertebrate, cephalopod} covers whale,
//! dolphin, shark, manta and octopus, whose only common ancestor is `animal`.
//! Senses are explicit: `dolphin` sense 1 is a fish and `octopus` sense 1 is
//! seafood, so a mask asks about a SENSE, never a word.

use lance_graph_contract::ogar_codebook::{concepts_in_domain, ConceptDomain};
use std::collections::{HashMap, HashSet, VecDeque};

/// One ontology concept: its canonical name and the id it is minted at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OntologyConcept {
    /// Canonical concept name as minted (`"mondo"`, `"uberon"`, `"bfo"`, …).
    pub name: &'static str,
    /// The canonical hi-u16 concept id (`0x03XX`).
    pub concept_id: u16,
}

/// The whole Ontology vocabulary, in codebook order.
///
/// Derived from the mirror on every call — never cached into a second table,
/// so it cannot drift from the mint the way a local copy would.
#[must_use]
pub fn ontology_vocabulary() -> Vec<OntologyConcept> {
    concepts_in_domain(ConceptDomain::Ontology)
        .map(|(name, concept_id)| OntologyConcept { name, concept_id })
        .collect()
}

/// Resolve a concept name to its id, or `None` if the Ontology domain does not
/// mint it — a refusal, never a guess.
#[must_use]
pub fn concept_id(name: &str) -> Option<u16> {
    concepts_in_domain(ConceptDomain::Ontology)
        .find(|(n, _)| *n == name)
        .map(|(_, id)| id)
}

/// The inverse: which ontology concept a `0x03XX` id names, or `None` outside
/// the domain.
#[must_use]
pub fn concept_at(concept_id: u16) -> Option<&'static str> {
    concepts_in_domain(ConceptDomain::Ontology)
        .find(|(_, id)| *id == concept_id)
        .map(|(n, _)| n)
}

// ═══════════════════════════════════════════════════════════════════════════
// WordNet — an identity register over synset offsets
// ═══════════════════════════════════════════════════════════════════════════

/// A WordNet synset: its offset AND its part of speech. An offset is unique
/// only inside one part-of-speech data file, so a noun and a verb can share
/// one (PR #1321 review); the pair is the identity. Never a coordinate in a
/// metric space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Synset {
    /// The part of speech (`'n'`, `'v'`, `'a'`, `'r'`, …).
    pub pos: char,
    /// The offset inside that part of speech's data file.
    pub offset: u32,
}

/// One sense of a word: its synset and WordNet's own sense rank (1-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sense {
    /// The synset this sense names.
    pub synset: Synset,
    /// WordNet's sense number for the lemma (1 = most frequent).
    pub sense_num: u16,
}

/// Why a rail line was refused. A malformed line is an error, never skipped:
/// a reader that silently drops lines reports a smaller taxonomy as if it were
/// the whole one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RailError {
    /// The line does not have exactly 7 tab-separated columns.
    Arity { line: usize, cols: usize },
    /// A sense number or synset offset is not a number.
    Number { line: usize },
}

/// The WordNet is-a rail: senses by `(lemma, pos)`, hypernym edges by synset.
/// All senses and every hypernym edge are kept (multiple inheritance included);
/// nothing is collapsed to a first sense.
#[derive(Debug, Default)]
pub struct WordNetRail {
    senses: HashMap<(String, char), Vec<Sense>>,
    parents: HashMap<Synset, Vec<Synset>>,
    names: HashMap<Synset, String>,
}

impl WordNetRail {
    /// Parse the 7-column rail (`word pos sense_num synset kind hypernym
    /// hypernym_offset`). `#` lines are comments. Exact arity, per the
    /// release manifest: a line with any other column count is refused.
    pub fn parse(text: &str) -> Result<Self, RailError> {
        let mut rail = Self::default();
        for (n, line) in text.lines().enumerate() {
            let line_no = n + 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() != 7 {
                return Err(RailError::Arity {
                    line: line_no,
                    cols: c.len(),
                });
            }
            let num = |s: &str| {
                s.parse::<u32>()
                    .map_err(|_| RailError::Number { line: line_no })
            };
            let pos = c[1]
                .chars()
                .next()
                .ok_or(RailError::Number { line: line_no })?;
            let sense_num =
                u16::try_from(num(c[2])?).map_err(|_| RailError::Number { line: line_no })?;
            let synset = Synset {
                pos,
                offset: num(c[3])?,
            };
            let senses = rail.senses.entry((c[0].to_string(), pos)).or_default();
            if !senses.iter().any(|s| s.synset == synset) {
                senses.push(Sense { synset, sense_num });
                senses.sort_by_key(|s| s.sense_num);
            }
            rail.names.entry(synset).or_insert_with(|| c[0].to_string());
            if !c[6].is_empty() {
                // Hypernym edges never cross parts of speech, so the parent
                // shares the row's part of speech.
                let parent = Synset {
                    pos,
                    offset: num(c[6])?,
                };
                // The hypernym column carries the parent's canonical name.
                rail.names.insert(parent, c[5].to_string());
                let ps = rail.parents.entry(synset).or_default();
                if !ps.contains(&parent) {
                    ps.push(parent);
                }
            }
        }
        Ok(rail)
    }

    /// Every sense of `word` as part of speech `pos` (`'n'`, `'v'`, `'a'`,
    /// `'r'`), in sense order. Empty when the rail does not know the word.
    #[must_use]
    pub fn senses(&self, word: &str, pos: char) -> &[Sense] {
        self.senses
            .get(&(word.to_string(), pos))
            .map_or(&[], Vec::as_slice)
    }

    /// The synset of one numbered sense, or `None`.
    #[must_use]
    pub fn synset(&self, word: &str, pos: char, sense_num: u16) -> Option<Synset> {
        self.senses(word, pos)
            .iter()
            .find(|s| s.sense_num == sense_num)
            .map(|s| s.synset)
    }

    /// A synset's name: the hypernym name the rail gives it, else the first
    /// word read for it.
    #[must_use]
    pub fn name(&self, synset: Synset) -> Option<&str> {
        self.names.get(&synset).map(String::as_str)
    }

    /// Every ancestor of `synset` through all hypernym edges, nearest first,
    /// each once. The synset itself is not included.
    #[must_use]
    pub fn ancestors(&self, synset: Synset) -> Vec<Synset> {
        let mut out = Vec::new();
        let mut seen: HashSet<Synset> = HashSet::from([synset]);
        let mut queue: VecDeque<Synset> = VecDeque::from([synset]);
        while let Some(s) = queue.pop_front() {
            for &p in self.parents.get(&s).map_or(&[][..], Vec::as_slice) {
                if seen.insert(p) {
                    out.push(p);
                    queue.push_back(p);
                }
            }
        }
        out
    }

    /// `synset` is `ancestor` or lies below it.
    #[must_use]
    pub fn is_a(&self, synset: Synset, ancestor: Synset) -> bool {
        synset == ancestor || self.ancestors(synset).contains(&ancestor)
    }
}

/// A category as a set of synsets: a sense is covered when it or any of its
/// ancestors is in the set. Membership only — two senses are never "close",
/// they are inside or outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptMask {
    roots: Vec<Synset>,
}

impl ConceptMask {
    /// A mask over the given root synsets.
    #[must_use]
    pub fn new(roots: impl IntoIterator<Item = Synset>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
        }
    }

    /// The root synsets of the mask.
    #[must_use]
    pub fn roots(&self) -> &[Synset] {
        &self.roots
    }

    /// `synset` falls under one of the mask's roots.
    #[must_use]
    pub fn covers(&self, rail: &WordNetRail, synset: Synset) -> bool {
        self.roots.iter().any(|&r| rail.is_a(synset, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The OBO half is superseded, not dropped: `98de36eb` retracted the 14
    /// 0x03XX mirror rows as never minted by ogar-vocab, so the Ontology
    /// domain is reserved with zero rows. The lookups answer empty and refuse,
    /// which is the honest state. When ogar-vocab mints 0x03 rows, this test
    /// fails and the original round-trip tests come back
    /// (`git show 48405aa2^:crates/deepnsm/src/ontology_vocab.rs`).
    #[test]
    fn the_ontology_domain_is_reserved_and_answers_empty() {
        assert!(ontology_vocabulary().is_empty());
        assert_eq!(concept_id("mondo"), None);
        assert_eq!(concept_at(0x0301), None);
    }

    /// Real rows from the WordNet 3.1 rail (WordNet License): the chains of
    /// whale, dolphin, shark, manta and octopus, all senses that appear.
    const EXCERPT: &str = include_str!("wordnet_excerpt.tsv");

    fn rail() -> WordNetRail {
        WordNetRail::parse(EXCERPT).expect("excerpt parses")
    }

    fn sense(r: &WordNetRail, w: &str, n: u16) -> Synset {
        r.synset(w, 'n', n)
            .unwrap_or_else(|| panic!("{w} sense {n}"))
    }

    fn named(r: &WordNetRail, name: &str) -> Synset {
        r.synset(name, 'n', 1)
            .or_else(|| r.names.iter().find(|(_, n)| *n == name).map(|(s, _)| *s))
            .unwrap_or_else(|| panic!("no synset named {name}"))
    }

    /// The hypernym chain is an address: whale (sense 2) reaches `animal`
    /// through cetacean and aquatic_mammal, nearest first.
    #[test]
    fn ancestry_walks_every_hypernym_nearest_first() {
        let r = rail();
        let whale = sense(&r, "whale", 2);
        let up: Vec<&str> = r
            .ancestors(whale)
            .iter()
            .filter_map(|&s| r.name(s))
            .collect();
        assert_eq!(&up[..3], &["cetacean", "aquatic_mammal", "placental"]);
        assert!(up.contains(&"animal"));
        assert!(r.is_a(whale, named(&r, "mammal")));
        assert!(!r.is_a(whale, named(&r, "fish")));
    }

    /// A mask cuts across the taxonomy. Aquatic mammals take whale and
    /// dolphin; cartilaginous fish take shark and manta; neither takes the
    /// other's members. Water animals need three roots from three branches,
    /// because the only common ancestor is `animal`.
    #[test]
    fn masks_cut_across_the_taxonomy() {
        let r = rail();
        let whale = sense(&r, "whale", 2);
        let dolphin = sense(&r, "dolphin", 2);
        let shark = sense(&r, "shark", 1);
        let manta = sense(&r, "manta", 2);
        let octopus = sense(&r, "octopus", 2);
        let mammals = ConceptMask::new([named(&r, "aquatic_mammal")]);
        let cartilaginous = ConceptMask::new([named(&r, "cartilaginous_fish")]);
        let water = ConceptMask::new([
            named(&r, "aquatic_mammal"),
            named(&r, "aquatic_vertebrate"),
            named(&r, "cephalopod"),
        ]);
        for s in [whale, dolphin] {
            assert!(mammals.covers(&r, s) && !cartilaginous.covers(&r, s));
        }
        for s in [shark, manta] {
            assert!(cartilaginous.covers(&r, s) && !mammals.covers(&r, s));
        }
        assert!(!mammals.covers(&r, octopus) && !cartilaginous.covers(&r, octopus));
        for s in [whale, dolphin, shark, manta, octopus] {
            assert!(water.covers(&r, s));
        }
        // The aquatic_vertebrate root alone does not reach the whale: WordNet
        // files whales under mammal, not under aquatic_vertebrate.
        assert!(!ConceptMask::new([named(&r, "aquatic_vertebrate")]).covers(&r, whale));
    }

    /// A mask asks about a sense, never a word: dolphin sense 1 is a
    /// percoid fish and octopus sense 1 is seafood.
    #[test]
    fn a_mask_reads_the_sense_not_the_word() {
        let r = rail();
        let mammals = ConceptMask::new([named(&r, "aquatic_mammal")]);
        let water = ConceptMask::new([
            named(&r, "aquatic_mammal"),
            named(&r, "aquatic_vertebrate"),
            named(&r, "cephalopod"),
        ]);
        assert!(!mammals.covers(&r, sense(&r, "dolphin", 1)));
        assert!(
            water.covers(&r, sense(&r, "dolphin", 1)),
            "a fish is a water animal"
        );
        assert!(
            !water.covers(&r, sense(&r, "octopus", 1)),
            "seafood is food"
        );
        assert_eq!(r.senses("dolphin", 'n').len(), 2);
    }

    /// Exact arity: a 6-column line is refused, never skipped.
    #[test]
    fn a_noun_and_a_verb_with_the_same_offset_stay_distinct() {
        // Same offset 00000123 in the noun file and the verb file: the verb
        // must not inherit the noun's ancestors.
        let text = "dog\tn\t1\t00000123\tisa\tanimal\t00000999\n\
                    run\tv\t1\t00000123\tisa\tmove\t00000555\n";
        let r = WordNetRail::parse(text).expect("parses");
        let dog = r.synset("dog", 'n', 1).expect("dog");
        let run = r.synset("run", 'v', 1).expect("run");
        assert_ne!(dog, run);
        assert_eq!(dog.offset, run.offset);
        let animal = Synset {
            pos: 'n',
            offset: 999,
        };
        assert!(r.is_a(dog, animal));
        assert!(
            !r.is_a(run, animal),
            "the verb inherited the noun's ancestor"
        );
        assert!(!ConceptMask::new([animal]).covers(&r, run));
        assert_eq!(r.ancestors(run).len(), 1);
    }

    #[test]
    fn a_line_with_the_wrong_arity_is_refused() {
        let bad = "whale\tn\t2\t02065397\tisa\tcetacean";
        assert_eq!(
            WordNetRail::parse(bad).unwrap_err(),
            RailError::Arity { line: 1, cols: 6 }
        );
        assert!(WordNetRail::parse("# comment only\n").is_ok());
    }
}
