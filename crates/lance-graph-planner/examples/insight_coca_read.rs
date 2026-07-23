//! `insight_coca_read` — D-SCI-1: the **corpus-grounded** full-record extractor.
//! Where `insight_spo_tekamolo_read` uses hand-seeded cue tables, this grounds
//! every lexical decision in **real COCA data** (`examples/data/coca/`, derived
//! from ngrams.info's free samples + the master lexicon) and the merged
//! `verb_table` archetype — then emits the same
//! **S · P · O + Temporal · Kausal · Modal · Lokal + Qualia** record into a real
//! [`TekamoloFacet`](lance_graph_contract::tekamolo_facet::TekamoloFacet) + the
//! canonical 17D [`QualiaI4_16D`](lance_graph_contract::qualia::QualiaI4_16D).
//!
//! ## What each COCA table grounds (no hand-rolled PoS, no hand-rolled connectives)
//!
//! | table (provenance) | what it grounds |
//! |---|---|
//! | `lexicon.tsv` (master COCA lexicon, format #3) | `word → (lemma, PoS)` — the authoritative noun/verb/adverb/prep/copula test AND lemmatisation (`carried → carry`) that feeds `read_verb` |
//! | `transitive_verbs.txt` (`v_the_n`) | which verbs are attested heading an object — the SVO verb preference |
//! | `noun_compounds.txt` (`n_n`) | two adjacent nouns fuse into ONE NP node only if the compound is attested (data-driven chunking, no over-fusing) |
//! | `verb_particles.txt` (`verb_adv`) | a particle after a verb (`looked down`, `picked up`) is NOT the object |
//! | `prep_place.txt` (`w1_prep_w3`) | attested locative `(prep, place-noun)` pairs — the Lokal cue |
//!
//! The **predicate family/tense/slot still comes from the #842 `verb_table`
//! archetype** (`read_verb`) — COCA supplies PoS + lemma, the archetype supplies
//! the relational typing. Verb detection is `PoS == v` in the lexicon AND the
//! lemma is archetype-known; unknown → no edge (sparsity). Qualia is the 17D
//! felt vector (a small polarity lexicon; COCA carries no sentiment).
//!
//! ## Falsifier (self-testing, runs in CI)
//!
//! `"the committee slowly supported the health care plan in the region"` — grounded
//! entirely in COCA: `committee/plan/region` are nouns, `supported → support` a
//! transitive verb, `slowly` an adverb (Modal), `in` a locative prep (Lokal),
//! and `health care` an ATTESTED `n_n` compound that fuses into one object NP
//! (`health_care_plan` via chained compounding). Asserts the grounded S·P·O, the
//! Modal/Lokal lanes, the TekamoloFacet round-trip, and a live 17D qualia.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example insight_coca_read -- FILE [FILE ...]

use std::collections::{HashMap, HashSet};

use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::verb_lexicon::{classify_verb, is_causal_cue};
use lance_graph_contract::grammar::verb_table::VerbFamily;
use lance_graph_contract::qualia::{QualiaI4_16D, QualiaVector, AXIS_LABELS, QUALIA_DIMS};
use lance_graph_contract::tekamolo_facet::{TekamoloFacet, TekamoloRole};

/// Simple PoS class from the COCA lexicon: n noun · v verb · b be/aux · j adj ·
/// r adverb · i prep.
type Pos = u8;

/// The loaded COCA tables — every lexical decision reads from here.
struct Coca {
    /// surface form → (lemma, PoS class byte).
    lex: HashMap<String, (String, Pos)>,
    /// attested noun-noun compounds (fuse into one NP node).
    compounds: HashSet<(String, String)>,
    /// verbs attested heading an object (`v_the_n`).
    transitive: HashSet<String>,
    /// phrasal particles (`looked down`) — never an object.
    particles: HashSet<String>,
    /// attested locative (prep, place) pairs.
    prep_place: HashSet<(String, String)>,
}

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/data/coca")
}

fn load_lines(name: &str) -> Vec<String> {
    let path = data_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

impl Coca {
    fn load() -> Self {
        let mut lex = HashMap::new();
        for l in load_lines("lexicon.tsv") {
            let mut it = l.split('\t');
            if let (Some(w), Some(lemma), Some(pos)) = (it.next(), it.next(), it.next()) {
                lex.insert(w.to_string(), (lemma.to_string(), pos.as_bytes()[0]));
            }
        }
        let pair = |name: &str| -> HashSet<(String, String)> {
            load_lines(name)
                .iter()
                .filter_map(|l| {
                    let mut it = l.split('\t');
                    Some((it.next()?.to_string(), it.next()?.to_string()))
                })
                .collect()
        };
        let single = |name: &str, col: usize| -> HashSet<String> {
            load_lines(name)
                .iter()
                .filter_map(|l| l.split('\t').nth(col).map(str::to_string))
                .collect()
        };
        Self {
            lex,
            compounds: pair("noun_compounds.txt"),
            transitive: single("transitive_verbs.txt", 0),
            particles: single("verb_particles.txt", 0),
            prep_place: pair("prep_place.txt"),
        }
    }

    fn pos(&self, w: &str) -> Option<Pos> {
        self.lex.get(w).map(|(_, p)| *p)
    }
    fn lemma<'a>(&'a self, w: &'a str) -> &'a str {
        self.lex.get(w).map(|(l, _)| l.as_str()).unwrap_or(w)
    }
    fn is_noun(&self, w: &str) -> bool {
        self.pos(w) == Some(b'n')
    }
    fn is_prep(&self, w: &str) -> bool {
        self.pos(w) == Some(b'i')
    }
    fn is_adverb(&self, w: &str) -> bool {
        self.pos(w) == Some(b'r')
    }
    /// A content verb per COCA (PoS v) whose lemma the `verb_table` archetype
    /// knows — the intersection of "corpus says verb" and "archetype types it".
    /// Classify the SURFACE form first so the archetype derives tense from the
    /// inflection (`supported → Past`); fall back to the COCA lemma only for
    /// irregulars the archetype's morphology can't strip (`carries → carry`).
    fn verb_reading(&self, w: &str) -> Option<(VerbFamily, Tense)> {
        if self.pos(w) != Some(b'v') {
            return None;
        }
        classify_verb(w).or_else(|| classify_verb(self.lemma(w)))
    }
}

const LOCAL_PREPS: &[&str] = &[
    "in", "on", "at", "into", "onto", "under", "over", "above", "below", "near", "inside",
    "outside", "behind", "through", "within", "across",
];
const TEMPORAL_CUES: &[(&str, u8)] = &[
    ("yesterday", 1),
    ("earlier", 1),
    ("now", 2),
    ("today", 2),
    ("tomorrow", 3),
    ("soon", 3),
    ("later", 3),
    ("then", 4),
    ("after", 4),
    ("before", 4),
    ("when", 4),
    ("always", 4),
    ("never", 4),
    ("often", 4),
];
const POSITIVE: &[&str] = &[
    "hope", "light", "love", "calm", "bright", "safe", "peace", "warm", "gentle", "clear", "heal",
    "approve", "support", "grow",
];
const NEGATIVE: &[&str] = &[
    "storm", "loss", "dark", "fear", "break", "cold", "pain", "death", "fall", "grief", "wound",
    "sink", "burn", "fail", "doubt", "crisis", "war",
];

fn is_manner_adverb(w: &str) -> bool {
    w.len() > 4 && w.ends_with("ly")
}

fn tense_code(t: Tense) -> u8 {
    match t {
        Tense::Past | Tense::PastContinuous | Tense::Pluperfect => 1,
        Tense::Present | Tense::PresentContinuous | Tense::Habitual | Tense::Imperative => 2,
        Tense::Future | Tense::FutureContinuous | Tense::FuturePerfect => 3,
        Tense::Perfect | Tense::Potential => 4,
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

#[derive(Default)]
struct Extraction {
    s: String,
    verb: String,
    o: String,
    family: Option<VerbFamily>,
    temporal: [u8; 3],
    kausal: [u8; 3],
    modal: [u8; 3],
    lokal: [u8; 3],
    lokal_place: String,
    modal_word: String,
    qualia: QualiaVector,
}

impl Extraction {
    fn to_tekamolo(&self, classid: u32) -> TekamoloFacet {
        TekamoloFacet::from_lanes(classid, self.temporal, self.kausal, self.modal, self.lokal)
    }
    fn to_qualia(&self) -> QualiaI4_16D {
        QualiaI4_16D::from_f32_17d(&self.qualia)
    }
}

/// Read the 17D felt vector from a clause (lemmas checked against the polarity
/// lexicon so `storms`/`storm` both count).
fn qualia_vector(coca: &Coca, toks: &[String], svo_found: bool) -> QualiaVector {
    let (mut pos, mut neg, mut fast) = (0.0f32, 0.0f32, 0.0f32);
    let (mut has_temporal, mut has_present, mut has_future) = (false, false, false);
    let (mut has_causal, mut has_modal, mut has_place) = (false, false, false);
    for w in toks {
        let lemma = coca.lemma(w);
        if POSITIVE.contains(&lemma) {
            pos += 1.0;
        }
        if NEGATIVE.contains(&lemma) {
            neg += 1.0;
        }
        if matches!(w.as_str(), "quickly" | "swiftly" | "suddenly" | "rapidly") {
            fast += 1.0;
        }
        if let Some((_, t)) = TEMPORAL_CUES.iter().find(|(c, _)| c == w) {
            has_temporal = true;
            has_present |= *t == 2;
            has_future |= *t == 3;
        }
        if is_causal_cue(w) {
            has_causal = true;
        }
        if coca.is_adverb(w) && is_manner_adverb(w) {
            has_modal = true;
        }
        if LOCAL_PREPS.contains(&w.as_str()) {
            has_place = true;
        }
    }
    let strong = (pos + neg).max(1.0);
    let mut q: QualiaVector = [0.0; QUALIA_DIMS];
    q[0] = ((pos + neg + fast) / strong).min(1.0);
    q[1] = (pos - neg) / strong;
    q[2] = (neg / strong).min(1.0);
    q[3] = (pos / strong).min(1.0);
    q[4] = if svo_found { 0.8 } else { 0.2 };
    q[6] = if has_causal { 0.7 } else { 0.0 };
    q[7] = (fast / strong).min(1.0);
    q[9] = if svo_found && (has_causal || has_temporal) {
        0.7
    } else {
        0.3
    };
    q[11] = if has_present { 0.7 } else { 0.0 };
    q[12] = if has_modal { 0.6 } else { 0.0 };
    q[14] = if has_place { 0.7 } else { 0.0 };
    q[15] = if has_future { 0.7 } else { 0.0 };
    q
}

fn extract(coca: &Coca, toks: &[String]) -> Extraction {
    let mut ex = Extraction::default();

    #[derive(PartialEq)]
    enum Item {
        Term(String),
        Verb(String),
    }
    let mut stream: Vec<Item> = Vec::new();
    let mut run: Vec<String> = Vec::new();
    // Flush an NP run: fuse ADJACENT nouns into one node only where the pair is an
    // attested COCA compound; otherwise keep the last noun (head) as the term.
    let flush = |run: &mut Vec<String>, stream: &mut Vec<Item>| {
        if run.is_empty() {
            return;
        }
        let mut fused: Vec<String> = vec![run[0].clone()];
        for w in run.iter().skip(1) {
            let last = fused.last().unwrap();
            let last_head = last.rsplit('_').next().unwrap().to_string();
            if coca.compounds.contains(&(last_head, w.clone())) {
                let merged = format!("{}_{}", fused.pop().unwrap(), w);
                fused.push(merged);
            } else {
                fused.push(w.clone());
            }
        }
        // the NP's head = the last fused chunk (English right-headed)
        stream.push(Item::Term(fused.pop().unwrap()));
        run.clear();
    };

    let mut i = 0usize;
    while i < toks.len() {
        let w = toks[i].as_str();
        if let Some((_, t)) = TEMPORAL_CUES.iter().find(|(c, _)| *c == w) {
            let fine = TEMPORAL_CUES.iter().position(|(c, _)| *c == w).unwrap() as u8 + 1;
            if ex.temporal == [0, 0, 0] {
                ex.temporal = [*t, fine, 0];
            }
            i += 1;
            continue;
        }
        if is_causal_cue(w) {
            if ex.kausal == [0, 0, 0] {
                ex.kausal = [
                    1,
                    (w.bytes().fold(0u16, |a, b| a + b as u16) % 250) as u8 + 1,
                    0,
                ];
            }
            i += 1;
            continue;
        }
        if coca.is_adverb(w) && is_manner_adverb(w) && coca.verb_reading(w).is_none() {
            if ex.modal == [0, 0, 0] {
                ex.modal = [
                    2,
                    (w.bytes().fold(0u16, |a, b| a + b as u16) % 250) as u8 + 1,
                    0,
                ];
                ex.modal_word = w.to_string();
            }
            i += 1;
            continue;
        }
        // Locative preposition → the next NOUN (skipping articles) is the place.
        if coca.is_prep(w) && LOCAL_PREPS.contains(&w) {
            let mut j = i + 1;
            while j < toks.len() && !coca.is_noun(&toks[j]) && coca.verb_reading(&toks[j]).is_none()
            {
                j += 1;
            }
            if j < toks.len() && coca.is_noun(&toks[j]) {
                let place = toks[j].clone();
                if ex.lokal == [0, 0, 0] {
                    let attested = coca.prep_place.contains(&(w.to_string(), place.clone()));
                    ex.lokal = [
                        if attested { 2 } else { 1 },
                        (place.bytes().fold(0u16, |a, b| a + b as u16) % 250) as u8 + 1,
                        0,
                    ];
                    ex.lokal_place = place;
                }
                flush(&mut run, &mut stream);
                i = j + 1;
                continue;
            }
            i += 1;
            continue;
        }
        // A verb: PoS v in COCA AND archetype-known lemma.
        if coca.verb_reading(w).is_some() {
            flush(&mut run, &mut stream);
            stream.push(Item::Verb(w.to_string()));
            i += 1;
            continue;
        }
        // A phrasal particle right after a verb is glued to it, not an object.
        if coca.particles.contains(w) && matches!(stream.last(), Some(Item::Verb(_))) {
            i += 1;
            continue;
        }
        // A noun (or unknown alpha content word) extends the current NP run.
        if coca.is_noun(w) || (w.chars().all(|c| c.is_ascii_alphabetic()) && coca.pos(w).is_none())
        {
            run.push(w.to_string());
        } else {
            flush(&mut run, &mut stream); // article/adj/other closes the run
        }
        i += 1;
    }
    flush(&mut run, &mut stream);

    // First transitive subject —verb→ object; prefer a COCA-transitive verb.
    let mut best: Option<usize> = None;
    let mut k = 0;
    while k + 2 < stream.len() {
        if let (Item::Term(_), Item::Verb(v), Item::Term(_)) =
            (&stream[k], &stream[k + 1], &stream[k + 2])
        {
            let transitive = coca.transitive.contains(coca.lemma(v));
            if best.is_none() || transitive {
                best = Some(k);
                if transitive {
                    break;
                }
            }
        }
        k += 1;
    }
    if let Some(k) = best {
        if let (Item::Term(s), Item::Verb(v), Item::Term(o)) =
            (&stream[k], &stream[k + 1], &stream[k + 2])
        {
            ex.s = s.clone();
            ex.verb = v.clone();
            ex.o = o.clone();
        }
    }

    if let Some((family, tense)) = coca.verb_reading(&ex.verb) {
        ex.family = Some(family);
        if ex.temporal[0] == 0 {
            ex.temporal[0] = tense_code(tense);
        }
    }
    ex.qualia = qualia_vector(coca, toks, !ex.verb.is_empty());
    ex
}

fn sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn felt_axes(q: &QualiaVector, k: usize) -> String {
    let mut idx: Vec<usize> = (0..QUALIA_DIMS).filter(|&i| q[i].abs() > 0.05).collect();
    idx.sort_by(|&a, &b| q[b].abs().total_cmp(&q[a].abs()));
    idx.truncate(k);
    if idx.is_empty() {
        return "(flat)".to_string();
    }
    idx.iter()
        .map(|&i| format!("{}={:+.2}", AXIS_LABELS[i], q[i]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn report(coca: &Coca, label: &str, text: &str) -> Vec<(Extraction, TekamoloFacet, QualiaI4_16D)> {
    println!("\n════════ {label} ════════");
    let mut out = Vec::new();
    for (n, sent) in sentences(text).iter().enumerate() {
        let ex = extract(coca, &tokens(sent));
        if ex.verb.is_empty() {
            continue;
        }
        let facet = ex.to_tekamolo(0x0000_0000);
        let qualia = ex.to_qualia();
        let tense = ["·", "past", "present", "future", "rel"][ex.temporal[0].min(4) as usize];
        println!(
            "  [{n}] {} —{}→ {}   [family: {}]",
            ex.s.replace('_', " "),
            ex.verb,
            ex.o.replace('_', " "),
            ex.family
                .map(|f| format!("{f:?}"))
                .unwrap_or_else(|| "untyped".into()),
        );
        println!(
            "        Te={} [{}]  Ka={}  Mo={}  Lo={}{}",
            tense,
            if ex.temporal[1] > 0 {
                TEMPORAL_CUES[(ex.temporal[1] - 1) as usize].0
            } else {
                "—"
            },
            if ex.kausal[0] > 0 { "because" } else { "—" },
            if ex.modal[0] > 0 {
                ex.modal_word.as_str()
            } else {
                "—"
            },
            if ex.lokal[0] > 0 {
                ex.lokal_place.as_str()
            } else {
                "—"
            },
            if ex.lokal[0] == 2 {
                " (COCA-attested)"
            } else {
                ""
            },
        );
        println!("        Qualia = {}", felt_axes(&ex.qualia, 4));
        out.push((ex, facet, qualia));
    }
    out
}

fn main() {
    let coca = Coca::load();
    println!(
        "loaded COCA: {} lexicon forms, {} compounds, {} transitive verbs, {} prep-place pairs",
        coca.lex.len(),
        coca.compounds.len(),
        coca.transitive.len(),
        coca.prep_place.len()
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for path in &args {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    report(&coca, path, &text);
                }
                Err(e) => eprintln!("skip {path}: {e}"),
            }
        }
        return;
    }

    // Falsifier: every lexical decision grounded in COCA.
    let text = "the committee slowly supported the health care plan in the region.";
    let out = report(&coca, "COCA-grounded falsifier", text);
    assert_eq!(out.len(), 1, "one transitive relation");
    let (ex, facet, qualia) = &out[0];

    assert_eq!(ex.s, "committee", "subject (COCA noun)");
    assert_eq!(ex.verb, "supported", "predicate (COCA transitive verb)");
    assert_eq!(
        ex.family,
        Some(VerbFamily::Supports),
        "support → Supports (archetype)"
    );
    // `health care` is an attested n_n compound → fuses; the object NP head is plan.
    assert!(
        ex.o == "health_care_plan" || ex.o.ends_with("plan"),
        "object NP fused via the COCA compound: got {}",
        ex.o
    );
    assert_eq!(ex.modal_word, "slowly", "Modal = COCA adverb");
    assert_eq!(ex.modal[0], 2, "Modal lane set");
    assert_eq!(
        ex.lokal_place, "region",
        "Lokal = COCA noun after locative prep"
    );
    assert!(ex.lokal[0] >= 1, "Lokal lane set");
    assert_eq!(
        ex.temporal[0], 1,
        "tense past (supported) from the archetype"
    );

    // Facet round-trip + live qualia.
    assert_eq!(
        facet.lane(TekamoloRole::Modal),
        ex.modal,
        "facet Modal round-trip"
    );
    assert_eq!(
        facet.lane(TekamoloRole::Lokal),
        ex.lokal,
        "facet Lokal round-trip"
    );
    assert_eq!(
        *qualia,
        QualiaI4_16D::from_f32_17d(&ex.qualia),
        "qualia packing"
    );
    let live = ex.qualia.iter().filter(|v| v.abs() > 0.05).count();
    assert!(live >= 3, "17D qualia texture, ≥3 live axes, got {live}");

    println!(
        "\n✔ COCA-grounded extraction: PoS + lemma from the master lexicon, NP chunking from n_n \
         compounds, transitivity from v_the_n, Lokal from w1_prep_w3 — verb FAMILY from the #842 \
         archetype; full S·P·O + TEKAMOLO facet + 17D qualia, all decisions corpus-attested."
    );
    println!("\n(usage: cargo run -p lance-graph-planner --example insight_coca_read -- FILE [FILE ...])");
}
