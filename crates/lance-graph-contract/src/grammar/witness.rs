//! `witness` — typed **construction licenses** from source-language grammar
//! witnesses (treebank-agnostic: PROIEL, UD, BHSA…).
//!
//! The design rule (D-SCI-1 Phase 2): a witness does NOT vote on the English
//! answer — it describes the **attested clause geometry** of the source
//! passage, and the constraint engine confirms / weakens / stays silent.
//! Candidate *generation* may use English evidence OR witness construction
//! evidence; candidate *elimination* uses all available typed constraints.
//!
//! Three structural findings from the live PROIEL probe are baked into these
//! types (recorded in `E-SCI-1-WITNESS-CONSTRUCTION-LICENSE-1`):
//!
//! 1. **Dependency outranks case.** In Acts 3:22 "him" is `αὐτοῦ` — GENITIVE,
//!    relation `obl` — because ἀκούω governs the genitive of person. The rule
//!    "accusative = object" would have missed the canonical example itself.
//!    So [`ClauseSignature`] carries dependency **relations**; morphological
//!    case is supporting evidence, never the classifier.
//! 2. **Voice is not a binary.** `ἀκούσεσθε` is future MIDDLE. Flattening
//!    Greek voice into active|passive would import Greek morphology while
//!    quietly projecting it back into an English binary — a scholarly-looking
//!    corruption factory. Hence [`VoiceClass`] (deponent variants reserved
//!    until a deponency lexicon lands — reserve, don't reclaim).
//! 3. **Missing text is not negative evidence.** KJV Acts 7:37 has "him shall
//!    ye hear" (Textus Receptus); the critical text PROIEL annotates does not.
//!    [`WitnessDisposition::TextAbsent`] is a first-class outcome — textual
//!    tradition belongs in the evidence address ([`ClauseSignature::edition`]),
//!    not in a footnote — and it never eliminates a candidate.

/// Voice of a witnessed predicate — deliberately richer than the English
/// active/passive binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceClass {
    Active,
    Passive,
    /// Greek middle (e.g. the future middle `ἀκούσεσθε` of active ἀκούω).
    Middle,
    /// Middle-deponent — RESERVED: emitting this requires lexicon knowledge
    /// of the lemma's deponency; parsers emit [`Middle`](Self::Middle) until
    /// the deponency lexicon lands.
    MiddleDeponent,
    /// Passive-deponent — RESERVED (same gate as `MiddleDeponent`).
    PassiveDeponent,
    /// Underdetermined by the annotation (e.g. PROIEL `e` = middle-or-passive),
    /// or no voice attested (non-finite / non-verb).
    Ambiguous,
}

impl VoiceClass {
    /// Map a PROIEL morphology voice code (position 5 of the 10-char string)
    /// into a voice class: `a`ctive, `m`iddle, `p`assive, `e`ither → Ambiguous.
    #[must_use]
    pub fn from_proiel_code(c: char) -> Self {
        match c {
            'a' => Self::Active,
            'm' => Self::Middle,
            'p' => Self::Passive,
            _ => Self::Ambiguous,
        }
    }

    /// Voices under which a fronted-argument ACTIVE canonicalization is
    /// licensed (the fronted argument stays the object/oblique of an agentive
    /// subject). Passive is NOT in this set — a passive witness licenses a
    /// different candidate class (Phase 2b).
    #[must_use]
    pub fn licenses_agentive(self) -> bool {
        matches!(self, Self::Active | Self::Middle | Self::MiddleDeponent)
    }
}

/// How a witness relates to a committed (or candidate) English reading.
///
/// The gate rule: a commitment must never be `Contradicted` **without
/// surfacing the conflict**; `TextAbsent` / `AlignmentUnknown` never block —
/// a missing witness must never prevent the English substrate from
/// generating or keeping a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WitnessDisposition {
    /// A matching clause exists and its geometry licenses the construction.
    Confirmed,
    /// A matching clause exists; its geometry neither licenses nor conflicts
    /// (translations reorder freely — absence of fronting in the source does
    /// NOT contradict fronting in English).
    Compatible,
    /// A matching clause exists and its geometry conflicts with the reading.
    /// RESERVED in Phase 2: no classifier emits this yet — ordering alone can
    /// never justify it, and role-conflict detection needs the lexical bridge.
    Contradicted,
    /// The witness's text does not contain the clause: textual-tradition
    /// difference (TR vs critical) or the passage lies outside the witness
    /// corpus. First-class evidence, never an elimination.
    TextAbsent,
    /// The clause could not be matched (e.g. the English predicate has no
    /// entry in the seed lexical bridge). Honest ignorance, never a veto.
    AlignmentUnknown,
}

/// The attested geometry of ONE source-language clause — what the witness
/// actually says, independent of any English candidate. Built per finite (or
/// participial) predicate token from a dependency treebank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseSignature {
    /// Passage address, witness-native (e.g. `ACTS 3.22`).
    pub citation: String,
    /// Which edition/tradition this witness annotates (e.g. `PROIEL-greek-nt`,
    /// a critical text — NOT the Textus Receptus the KJV translates). The
    /// tradition is part of the evidence address.
    pub edition: String,
    /// Ordinal of this clause among the verse's predicates.
    pub clause_index: u16,
    /// The predicate lemma, witness-native script (e.g. `ἀκούω`).
    pub predicate_lemma: String,
    /// Whether a dependent with relation `sub` is expressed (Greek pro-drop:
    /// an unexpressed subject lives in the verb's person morphology).
    pub subject_expressed: bool,
    /// Dependency relations of argument dependents appearing BEFORE the
    /// predicate (the fronted field), e.g. `["obl"]` for `αὐτοῦ ἀκούσεσθε`.
    pub fronted_relations: Vec<String>,
    /// Dependency relations of all argument dependents of this predicate.
    pub argument_relations: Vec<String>,
    /// The predicate's voice class.
    pub voice: VoiceClass,
}

impl ClauseSignature {
    /// Does this clause attest a **fronted argument** (an `obj` or `obl`
    /// dependent preceding its predicate)? Dependency-first by design:
    /// `obl` is included precisely because government verbs (ἀκούω +
    /// genitive) surface their patient as oblique — case would lie here.
    #[must_use]
    pub fn has_fronted_argument(&self) -> bool {
        self.fronted_relations
            .iter()
            .any(|r| r == "obj" || r == "obl")
    }

    /// The Phase-2 license: fronted argument + agentive voice ⇒ an English
    /// object-fronted ACTIVE candidate is licensed for this passage.
    #[must_use]
    pub fn licenses_fronted_object_active(&self) -> bool {
        self.has_fronted_argument() && self.voice.licenses_agentive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acts 3:22 `αὐτοῦ ἀκούσεσθε` — the government-verb warning shot: the
    /// fronted patient is GENITIVE/`obl` and the verb is future MIDDLE, yet
    /// the fronted-object-active license must fire.
    #[test]
    fn government_verb_obl_and_middle_still_license() {
        let sig = ClauseSignature {
            citation: "ACTS 3.22".into(),
            edition: "PROIEL-greek-nt".into(),
            clause_index: 2,
            predicate_lemma: "ἀκούω".into(),
            subject_expressed: false, // pro-drop: subject in 2p verb morphology
            fronted_relations: vec!["obl".into()],
            argument_relations: vec!["obl".into(), "obj".into()],
            voice: VoiceClass::from_proiel_code('m'),
        };
        assert_eq!(sig.voice, VoiceClass::Middle);
        assert!(
            sig.has_fronted_argument(),
            "obl counts — dependency, not case"
        );
        assert!(sig.licenses_fronted_object_active());
    }

    /// A passive witness does NOT license the agentive fronted-object reading.
    #[test]
    fn passive_witness_does_not_license_agentive() {
        let sig = ClauseSignature {
            citation: "X 1.1".into(),
            edition: "test".into(),
            clause_index: 0,
            predicate_lemma: "x".into(),
            subject_expressed: true,
            fronted_relations: vec!["obj".into()],
            argument_relations: vec!["obj".into()],
            voice: VoiceClass::Passive,
        };
        assert!(!sig.licenses_fronted_object_active());
    }

    /// No fronting attested → no license (but that is Compatible, never
    /// Contradicted — translations reorder freely).
    #[test]
    fn unfronted_clause_does_not_license() {
        let sig = ClauseSignature {
            citation: "X 1.2".into(),
            edition: "test".into(),
            clause_index: 0,
            predicate_lemma: "x".into(),
            subject_expressed: true,
            fronted_relations: vec![],
            argument_relations: vec!["obj".into()],
            voice: VoiceClass::Active,
        };
        assert!(!sig.licenses_fronted_object_active());
    }

    #[test]
    fn proiel_voice_codes_map_honestly() {
        assert_eq!(VoiceClass::from_proiel_code('a'), VoiceClass::Active);
        assert_eq!(VoiceClass::from_proiel_code('m'), VoiceClass::Middle);
        assert_eq!(VoiceClass::from_proiel_code('p'), VoiceClass::Passive);
        // PROIEL 'e' = middle-or-passive → Ambiguous, never silently active.
        assert_eq!(VoiceClass::from_proiel_code('e'), VoiceClass::Ambiguous);
        assert_eq!(VoiceClass::from_proiel_code('-'), VoiceClass::Ambiguous);
        assert!(!VoiceClass::Passive.licenses_agentive());
        assert!(VoiceClass::Middle.licenses_agentive());
    }
}
