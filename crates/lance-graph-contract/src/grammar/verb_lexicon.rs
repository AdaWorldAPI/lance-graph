//! `verb_lexicon` — verb surface form → [`VerbFamily`] + [`Tense`], the
//! **consumer** of the 144-cell [`verb_table`](crate::grammar::verb_table)
//! archetypes. It closes the archetype path the table opened:
//!
//! ```text
//!   verb string  ──classify──▶  (VerbFamily, Tense)
//!                                      │  verb_table::base_prior + tense_modifier
//!                                      ▼
//!                                  SlotPrior  ──argmax──▶  TekamoloSlot
//! ```
//!
//! Where `verb_table` answers *"given a family and tense, which TEKAMOLO slots
//! does a predicate expect?"*, this module answers *"what family and tense is
//! THIS verb?"* — so a downstream relation extractor reads a **table cell**
//! (never a hand-rolled connective list) to type its edge and pick the
//! adverbial slot the relation fills. This is the piece the D-SCI-1 extractor
//! was missing: PR #841 shipped the sparse-edge *shape* with a flat verb list;
//! this makes the verb resolve to its archetype so the edge carries a
//! `(VerbFamily, TekamoloSlot)` reading, not just "there was a verb here."
//!
//! Deterministic, zero-dep, no model: a fixed lexicon + regular English
//! morphology (`-ed` / `-ing` / `-s`, with the `-e` and doubled-consonant
//! restorations). Unknown words return `None` — the extractor emits no typed
//! edge for them, which is exactly what keeps the graph sparse.
//!
//! Copulas (`is` / `are` / `was` …) are intentionally NOT in the family
//! lexicon: they are inheritance links (`Copula::Inh`, is-a), handled by the
//! consumer via [`is_copula`], not a verb family. Causal connectives
//! (`because` / `therefore` …) are likewise surfaced via [`is_causal_cue`] so
//! the consumer can emit `Copula::Impl` for reasoned causality.
//!
//! The family→verb assignments are **starter** knowledge (grammar-landscape.md
//! §3 semantics), same status as `verb_table`'s starter priors: a future PR
//! replaces them with corpus-derived frequencies. Marked here so no consumer
//! reads a specific verb→family mapping as ground truth.

use crate::grammar::role_keys::Tense;
use crate::grammar::tekamolo::TekamoloSlot;
use crate::grammar::verb_table::{base_prior, tense_modifier, SlotPrior, VerbFamily};

/// The base (present-tense) lemma → family table. Grouped by the twelve
/// `verb_table` families, seeded from their semantic profiles
/// (`base_prior`): Change verbs (temporal+modal), Action verbs
/// (kausal+temporal), State verbs (modal), etc. Starter — tune empirically.
///
/// Entries are BASE forms (`cause`, not `caused`); [`classify_verb`] strips
/// regular inflection to a base candidate before lookup.
const FAMILY_LEXICON: &[(&str, VerbFamily)] = &[
    // ── Becomes — Change verb: high Temporal + Modal ──
    ("become", VerbFamily::Becomes),
    ("grow", VerbFamily::Becomes),
    ("emerge", VerbFamily::Becomes),
    ("develop", VerbFamily::Becomes),
    ("evolve", VerbFamily::Becomes),
    ("mature", VerbFamily::Becomes),
    ("ripen", VerbFamily::Becomes),
    ("awaken", VerbFamily::Becomes),
    ("arise", VerbFamily::Becomes),
    ("bloom", VerbFamily::Becomes),
    // ── Causes — Action verb: high Kausal + Instrument ──
    ("cause", VerbFamily::Causes),
    ("produce", VerbFamily::Causes),
    ("generate", VerbFamily::Causes),
    ("create", VerbFamily::Causes),
    ("trigger", VerbFamily::Causes),
    ("induce", VerbFamily::Causes),
    ("yield", VerbFamily::Causes),
    ("spark", VerbFamily::Causes),
    ("bring", VerbFamily::Causes),
    ("effect", VerbFamily::Causes),
    // ── Supports — State verb: high Modal, low Temporal ──
    ("support", VerbFamily::Supports),
    ("hold", VerbFamily::Supports),
    ("sustain", VerbFamily::Supports),
    ("uphold", VerbFamily::Supports),
    ("bear", VerbFamily::Supports),
    ("carry", VerbFamily::Supports),
    ("maintain", VerbFamily::Supports),
    ("confirm", VerbFamily::Supports),
    ("corroborate", VerbFamily::Supports),
    ("affirm", VerbFamily::Supports),
    // ── Contradicts — State verb: high Modal + Kausal ──
    ("contradict", VerbFamily::Contradicts),
    ("oppose", VerbFamily::Contradicts),
    ("deny", VerbFamily::Contradicts),
    ("refute", VerbFamily::Contradicts),
    ("negate", VerbFamily::Contradicts),
    ("conflict", VerbFamily::Contradicts),
    ("dispute", VerbFamily::Contradicts),
    ("challenge", VerbFamily::Contradicts),
    ("undermine", VerbFamily::Contradicts),
    ("counter", VerbFamily::Contradicts),
    // ── Refines — State verb: high Modal, moderate Kausal ──
    ("refine", VerbFamily::Refines),
    ("clarify", VerbFamily::Refines),
    ("sharpen", VerbFamily::Refines),
    ("improve", VerbFamily::Refines),
    ("adjust", VerbFamily::Refines),
    ("tune", VerbFamily::Refines),
    ("polish", VerbFamily::Refines),
    ("hone", VerbFamily::Refines),
    ("qualify", VerbFamily::Refines),
    ("temper", VerbFamily::Refines),
    // ── Grounds — State verb: high Lokal + Modal ──
    ("ground", VerbFamily::Grounds),
    ("rest", VerbFamily::Grounds),
    ("base", VerbFamily::Grounds),
    ("root", VerbFamily::Grounds),
    ("anchor", VerbFamily::Grounds),
    ("found", VerbFamily::Grounds),
    ("situate", VerbFamily::Grounds),
    ("locate", VerbFamily::Grounds),
    ("embed", VerbFamily::Grounds),
    ("dwell", VerbFamily::Grounds),
    // ── Abstracts — Change verb: high Modal + Temporal ──
    ("abstract", VerbFamily::Abstracts),
    ("generalize", VerbFamily::Abstracts),
    ("summarize", VerbFamily::Abstracts),
    ("distill", VerbFamily::Abstracts),
    ("simplify", VerbFamily::Abstracts),
    ("conceptualize", VerbFamily::Abstracts),
    ("idealize", VerbFamily::Abstracts),
    ("model", VerbFamily::Abstracts),
    // ── Enables — Discovery/enablement: high Kausal + Lokal ──
    ("enable", VerbFamily::Enables),
    ("allow", VerbFamily::Enables),
    ("permit", VerbFamily::Enables),
    ("facilitate", VerbFamily::Enables),
    ("empower", VerbFamily::Enables),
    ("afford", VerbFamily::Enables),
    ("unlock", VerbFamily::Enables),
    ("admit", VerbFamily::Enables),
    // ── Prevents — Action verb: high Kausal + Temporal ──
    ("prevent", VerbFamily::Prevents),
    ("block", VerbFamily::Prevents),
    ("hinder", VerbFamily::Prevents),
    ("impede", VerbFamily::Prevents),
    ("inhibit", VerbFamily::Prevents),
    ("deter", VerbFamily::Prevents),
    ("forbid", VerbFamily::Prevents),
    ("resist", VerbFamily::Prevents),
    ("obstruct", VerbFamily::Prevents),
    ("avoid", VerbFamily::Prevents),
    // ── Transforms — Action verb: high Kausal + Temporal + Instrument ──
    ("transform", VerbFamily::Transforms),
    ("convert", VerbFamily::Transforms),
    ("alter", VerbFamily::Transforms),
    ("translate", VerbFamily::Transforms),
    ("transmute", VerbFamily::Transforms),
    ("reshape", VerbFamily::Transforms),
    ("remake", VerbFamily::Transforms),
    ("adapt", VerbFamily::Transforms),
    // ── Mirrors — Change verb: high Temporal + Modal + Lokal ──
    ("mirror", VerbFamily::Mirrors),
    ("reflect", VerbFamily::Mirrors),
    ("resemble", VerbFamily::Mirrors),
    ("echo", VerbFamily::Mirrors),
    ("parallel", VerbFamily::Mirrors),
    ("imitate", VerbFamily::Mirrors),
    ("mimic", VerbFamily::Mirrors),
    ("correspond", VerbFamily::Mirrors),
    ("match", VerbFamily::Mirrors),
    // ── Dissolves — Change verb: high Temporal + Modal ──
    ("dissolve", VerbFamily::Dissolves),
    ("dissipate", VerbFamily::Dissolves),
    ("fade", VerbFamily::Dissolves),
    ("vanish", VerbFamily::Dissolves),
    ("collapse", VerbFamily::Dissolves),
    ("crumble", VerbFamily::Dissolves),
    ("disintegrate", VerbFamily::Dissolves),
    ("decay", VerbFamily::Dissolves),
    ("erode", VerbFamily::Dissolves),
    ("melt", VerbFamily::Dissolves),
];

/// Copula surface forms — the `to be` paradigm. A copula is an inheritance
/// (is-a) link, NOT a verb family; the consumer emits `Copula::Inh` for these.
const COPULA_BE: &[&str] = &[
    "is", "are", "was", "were", "be", "been", "being", "am", "isnt", "arent", "wasnt", "werent",
];

/// Causal connectives — reasoned-causality cues. The consumer emits
/// `Copula::Impl` (reasoned implication, non-transitive) when one links two
/// clauses. Distinct from the `Causes` verb family (a lexical predicate).
const CAUSAL_CUES: &[&str] = &[
    "because",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "so",
    "accordingly",
    "wherefore",
];

/// Is `word` a form of the copula `to be`? (Case-insensitive.)
#[must_use]
pub fn is_copula(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    COPULA_BE.contains(&w.as_str())
}

/// Is `word` a causal connective (`because` / `therefore` / …)? (Case-insensitive.)
#[must_use]
pub fn is_causal_cue(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    CAUSAL_CUES.contains(&w.as_str())
}

/// Look up a BASE lemma directly (no morphology). Exposed for consumers that
/// have already lemmatised.
#[must_use]
pub fn family_of_lemma(lemma: &str) -> Option<VerbFamily> {
    let l = lemma.to_ascii_lowercase();
    FAMILY_LEXICON
        .iter()
        .find(|(v, _)| *v == l.as_str())
        .map(|(_, f)| *f)
}

/// Candidate (lemma, tense) readings for a surface form, in priority order.
/// Regular English inflection only — `-ing`/`-ed`/`-es`/`-s` with the `-e`
/// restoration (`caus` → `cause`) and single doubled-consonant collapse
/// (`running` → `run`). The surface form itself is tried first (irregulars and
/// already-base forms), as present tense.
fn lemma_candidates(w: &str) -> Vec<(String, Tense)> {
    let mut out: Vec<(String, Tense)> = Vec::new();
    let push = |s: String, t: Tense, out: &mut Vec<(String, Tense)>| {
        if !s.is_empty() && !out.iter().any(|(cs, _)| *cs == s) {
            out.push((s, t));
        }
    };
    // Bare surface form (irregular past / present base): try as present.
    push(w.to_string(), Tense::Present, &mut out);

    // -ing → present continuous.
    if let Some(stem) = w.strip_suffix("ing") {
        push(stem.to_string(), Tense::PresentContinuous, &mut out);
        push(format!("{stem}e"), Tense::PresentContinuous, &mut out); // caus(ing) → cause
        if let Some(collapsed) = collapse_double(stem) {
            push(collapsed, Tense::PresentContinuous, &mut out); // runn(ing) → run
        }
    }
    // -ed → past.
    if let Some(stem) = w.strip_suffix("ed") {
        push(stem.to_string(), Tense::Past, &mut out);
        push(format!("{stem}e"), Tense::Past, &mut out); // caus(ed) → cause
        if let Some(collapsed) = collapse_double(stem) {
            push(collapsed, Tense::Past, &mut out); // stopp(ed) → stop
        }
    }
    // -es → present (3rd person after sibilant: passes → pass, dissolves handled by -s).
    if let Some(stem) = w.strip_suffix("es") {
        push(stem.to_string(), Tense::Present, &mut out); // passes → pass
        push(format!("{stem}e"), Tense::Present, &mut out); // dissolv(es) → dissolve
    }
    // -s → present (3rd person singular).
    if let Some(stem) = w.strip_suffix('s') {
        push(stem.to_string(), Tense::Present, &mut out); // grounds → ground
    }
    out
}

/// If `stem` ends in a doubled consonant (`runn`, `stopp`), return it with one
/// removed (`run`, `stop`). Vowels are not treated as doublings worth
/// collapsing (`agree` stays `agree`).
fn collapse_double(stem: &str) -> Option<String> {
    let b = stem.as_bytes();
    let n = b.len();
    if n >= 2 && b[n - 1] == b[n - 2] && b[n - 1].is_ascii_alphabetic() && !is_vowel(b[n - 1]) {
        Some(stem[..n - 1].to_string())
    } else {
        None
    }
}

fn is_vowel(c: u8) -> bool {
    matches!(c.to_ascii_lowercase(), b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Classify a verb surface form into its `(VerbFamily, Tense)` archetype
/// coordinate. Returns `None` for non-verbs, copulas, and unknown words —
/// the consumer emits no typed relational edge for those (sparsity by default).
///
/// ```
/// use lance_graph_contract::grammar::verb_lexicon::classify_verb;
/// use lance_graph_contract::grammar::verb_table::VerbFamily;
/// use lance_graph_contract::grammar::role_keys::Tense;
/// assert_eq!(classify_verb("caused"), Some((VerbFamily::Causes, Tense::Past)));
/// assert_eq!(classify_verb("grounds"), Some((VerbFamily::Grounds, Tense::Present)));
/// assert_eq!(classify_verb("dissolving"), Some((VerbFamily::Dissolves, Tense::PresentContinuous)));
/// assert_eq!(classify_verb("the"), None);
/// assert_eq!(classify_verb("is"), None); // copula → is_copula, not a family
/// ```
#[must_use]
pub fn classify_verb(word: &str) -> Option<(VerbFamily, Tense)> {
    let lower = word.to_ascii_lowercase();
    for (lemma, tense) in lemma_candidates(&lower) {
        if let Some(family) = family_of_lemma(&lemma) {
            return Some((family, tense));
        }
    }
    None
}

/// The dominant TEKAMOLO slot of a slot prior — the adverbial role a predicate
/// of this profile most expects to fill. Deterministic argmax over the five
/// axes; ties resolve to the earliest axis (Temporal < Kausal < Modal < Lokal
/// < Instrument), matching [`TekamoloSlot`] declaration order.
#[must_use]
pub fn dominant_slot(prior: SlotPrior) -> TekamoloSlot {
    let axes = [
        (prior.temporal, TekamoloSlot::Temporal),
        (prior.kausal, TekamoloSlot::Kausal),
        (prior.modal, TekamoloSlot::Modal),
        (prior.lokal, TekamoloSlot::Lokal),
        (prior.instrument, TekamoloSlot::Instrument),
    ];
    let mut best = axes[0];
    for &cand in &axes[1..] {
        if cand.0 > best.0 {
            best = cand;
        }
    }
    best.1
}

/// The TEKAMOLO slot a `(family, tense)` predicate fills — the full archetype
/// read: `dominant_slot(base_prior(family).combine(tense_modifier(tense)))`.
/// This is the single call a relation extractor makes to type an edge's
/// adverbial role from the `verb_table` cell.
///
/// ```
/// use lance_graph_contract::grammar::verb_lexicon::slot_for;
/// use lance_graph_contract::grammar::verb_table::VerbFamily;
/// use lance_graph_contract::grammar::role_keys::Tense;
/// use lance_graph_contract::grammar::tekamolo::TekamoloSlot;
/// assert_eq!(slot_for(VerbFamily::Causes, Tense::Present), TekamoloSlot::Kausal);
/// assert_eq!(slot_for(VerbFamily::Grounds, Tense::Present), TekamoloSlot::Lokal);
/// assert_eq!(slot_for(VerbFamily::Becomes, Tense::Present), TekamoloSlot::Temporal);
/// ```
#[must_use]
pub fn slot_for(family: VerbFamily, tense: Tense) -> TekamoloSlot {
    dominant_slot(base_prior(family).combine(tense_modifier(tense)))
}

/// One-shot convenience: a verb surface form → its `(family, tense, slot)`
/// reading, or `None` if it is not a known verb. This is exactly what the
/// D-SCI-1 relation extractor needs per detected verb: the family types the
/// relation, the slot names the adverbial role the edge fills.
#[must_use]
pub fn read_verb(word: &str) -> Option<(VerbFamily, Tense, TekamoloSlot)> {
    let (family, tense) = classify_verb(word)?;
    Some((family, tense, slot_for(family, tense)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_regular_and_irregular_inflections() {
        assert_eq!(
            classify_verb("cause"),
            Some((VerbFamily::Causes, Tense::Present))
        );
        assert_eq!(
            classify_verb("caused"),
            Some((VerbFamily::Causes, Tense::Past))
        );
        assert_eq!(
            classify_verb("causing"),
            Some((VerbFamily::Causes, Tense::PresentContinuous))
        );
        assert_eq!(
            classify_verb("supports"),
            Some((VerbFamily::Supports, Tense::Present))
        );
        assert_eq!(
            classify_verb("dissolving"),
            Some((VerbFamily::Dissolves, Tense::PresentContinuous))
        );
        assert_eq!(
            classify_verb("grounds"),
            Some((VerbFamily::Grounds, Tense::Present))
        );
    }

    #[test]
    fn collapses_doubled_consonant_past() {
        // "embedded" → strip -ed → "embedd" → collapse → "embed".
        assert_eq!(
            classify_verb("embedded"),
            Some((VerbFamily::Grounds, Tense::Past))
        );
    }

    #[test]
    fn non_verbs_copulas_and_cues_are_not_families() {
        assert_eq!(classify_verb("the"), None);
        assert_eq!(classify_verb("truth"), None);
        assert_eq!(classify_verb("is"), None);
        assert_eq!(classify_verb("because"), None);
        assert!(is_copula("is"));
        assert!(is_copula("Were"));
        assert!(!is_copula("cause"));
        assert!(is_causal_cue("therefore"));
        assert!(is_causal_cue("Because"));
        assert!(!is_causal_cue("cause"));
    }

    #[test]
    fn every_family_has_at_least_one_verb_that_round_trips() {
        for family in VerbFamily::ALL {
            let has = FAMILY_LEXICON
                .iter()
                .any(|(v, f)| *f == family && classify_verb(v).map(|(cf, _)| cf) == Some(family));
            assert!(has, "family {family:?} has no round-tripping verb");
        }
    }

    #[test]
    fn dominant_slot_matches_family_semantics() {
        // The archetype read: each family's dominant slot from its base prior.
        assert_eq!(
            slot_for(VerbFamily::Causes, Tense::Present),
            TekamoloSlot::Kausal
        );
        assert_eq!(
            slot_for(VerbFamily::Grounds, Tense::Present),
            TekamoloSlot::Lokal
        );
        assert_eq!(
            slot_for(VerbFamily::Becomes, Tense::Present),
            TekamoloSlot::Temporal
        );
        assert_eq!(
            slot_for(VerbFamily::Contradicts, Tense::Present),
            TekamoloSlot::Modal
        );
        assert_eq!(
            slot_for(VerbFamily::Supports, Tense::Present),
            TekamoloSlot::Modal
        );
    }

    #[test]
    fn dominant_slot_ties_resolve_to_earliest_axis() {
        assert_eq!(dominant_slot(SlotPrior::uniform()), TekamoloSlot::Temporal);
    }

    #[test]
    fn read_verb_gives_family_tense_and_slot() {
        assert_eq!(
            read_verb("caused"),
            Some((VerbFamily::Causes, Tense::Past, TekamoloSlot::Kausal))
        );
        assert_eq!(read_verb("nonverb"), None);
    }

    #[test]
    fn imperative_potential_shift_modal_but_family_still_classifies() {
        // Tense modulation is applied by slot_for via tense_modifier; here we
        // only check that the classifier itself is tense-robust for the forms
        // the extractor sees. (Imperative/Potential need aux context the flat
        // classifier does not see, so they arrive via the table, not here.)
        assert_eq!(
            classify_verb("prevented"),
            Some((VerbFamily::Prevents, Tense::Past))
        );
    }
}
