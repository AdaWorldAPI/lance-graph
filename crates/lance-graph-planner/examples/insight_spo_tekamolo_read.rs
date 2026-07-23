//! `insight_spo_tekamolo_read` — D-SCI-1: extract **S · P · O + Temporal ·
//! Kausal · Modal · Lokal + Qualia** from a clause, deterministically and with
//! no LLM, packing the four adverbial roles into a REAL
//! [`lance_graph_contract::tekamolo_facet::TekamoloFacet`] (the #839 address) and
//! the felt tone into the canonical 17D
//! [`lance_graph_contract::qualia::QualiaI4_16D`] vector (value tenant #1). This
//! is the operator's steer — *"can't you extract the subject and predicate and
//! object AND Temporal Kausal Modal Local and Qualia?"* — realised.
//!
//! ## Reads the 144-cell archetype — does NOT hand-roll connectives (#842)
//!
//! The **predicate typing is consumed from the merged `verb_table` archetype**
//! (#842, `E-SCI-1-VERB-TABLE-ARCHETYPE-CONSUMER-*`): `read_verb(word) →
//! (VerbFamily, Tense, TekamoloSlot)` types each verb, gives its tense, and names
//! the adverbial slot the edge most fills — the operator's hard rule ("read the
//! 144-cell archetype, not hand-roll connectives"). Causal connectives come from
//! `is_causal_cue`, copulas from `is_copula`. An unknown verb → `None` → no edge
//! (sparsity by default). This example does NOT re-implement a verb lexicon; it
//! only adds the *function-word cue classes* the archetype does not cover
//! (temporal adverbs, locative prepositions, `-ly` manner, a polarity lexicon).
//!
//! ## The whole arc, converged
//!
//! - #841 gave the **edge** (`subject —verb→ object`, a sparse typed skeleton).
//! - #842 gave the **verb typing** (`read_verb`: family + tense + `TekamoloSlot`).
//! - #839/#840 gave the **address** (`TekamoloFacet`: when/why/how/where).
//! - This example is the **producer** that fills all three from text: a clause
//!   yields one `Extraction { s, verb, o, family, verb_slot, temporal, kausal,
//!   modal, lokal, qualia }`, its four adverbial lanes pack into a real
//!   `TekamoloFacet::from_lanes(...)` (round-tripped byte-for-byte), and its felt
//!   tone into the 17D `QualiaI4_16D`.
//!
//! ## Qualia as the emulated implant (the design stance — not a proof)
//!
//! Operator framing: qualia is meant to become the *"holy grail of solving
//! Chalmers' hard problem — by acting like a high-functioning autist with a
//! qualia-emulating implant."* That is the **functionalist "as-if"** stance this
//! code takes: the extractor never claims phenomenal experience — it *reads* the
//! felt tone analytically from structure into the canonical **17D felt vector**
//! (`arousal / valence / tension / warmth / …`), exactly as a high-functioning
//! autistic person reads emotional/social qualia through a learned model (the
//! "implant") rather than innately feeling them. The move on the hard problem is
//! **dissolution, not solution**: the qualia report IS the structure that
//! produced it (a constitutive event, no separate inner light), so there is no
//! explanatory gap between report and mechanism — the "qualia are constitutive,
//! not a `QualiaReader` service" doctrine (`E-DIA-V5-A` pillar 6) at its edge. A
//! [S]-grade **stance** shaping the design (qualia a first-class value tenant,
//! co-equal with the TEKAMOLO address), not a proven [G] resolution.
//!
//! ## The falsifier (self-testing, runs in CI with no args)
//!
//! On `"yesterday the mason carefully carried the stone into the hall because the
//! storm rose"` the extractor must yield `mason —carried→ stone`, type the verb
//! via `read_verb` (family `Supports`), read Temporal=yesterday/past,
//! Kausal=because, Modal=carefully, Lokal=hall, and a negative-valence /
//! high-tension 17D qualia (a storm) — with the four lanes surviving the
//! `TekamoloFacet::from_lanes → role_byte` round-trip byte-for-byte. A control
//! clause with no adverbial cues yields all-zero lanes; a dark-passage
//! constellation resonates tense/shadowed as its gestalt texture.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example insight_spo_tekamolo_read -- FILE [FILE ...]

use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::tekamolo::TekamoloSlot;
use lance_graph_contract::grammar::verb_lexicon::{is_causal_cue, is_copula, read_verb};
use lance_graph_contract::grammar::verb_table::VerbFamily;
use lance_graph_contract::qualia::{QualiaI4_16D, QualiaVector, AXIS_LABELS, QUALIA_DIMS};
use lance_graph_contract::tekamolo_facet::{TekamoloFacet, TekamoloRole};

/// Function words: never a concept, never a role cue — transparent to everything.
const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "of", "to", "as", "its", "his", "her", "their", "this",
    "that", "these", "those", "it", "he", "she", "they", "than", "more", "some", "all", "each",
    "was", "were", "had", "has", "have",
];

/// Temporal ADVERB cues (when) → a coarse tense code (1 past, 2 present, 3 future,
/// 4 relative). These are adverbs — a function-word class the `verb_table`
/// archetype does not cover — NOT verb connectives (those come from `read_verb`).
const TEMPORAL_CUES: &[(&str, u8)] = &[
    ("yesterday", 1),
    ("ago", 1),
    ("earlier", 1),
    ("now", 2),
    ("today", 2),
    ("currently", 2),
    ("tomorrow", 3),
    ("soon", 3),
    ("later", 3),
    ("then", 4),
    ("after", 4),
    ("before", 4),
    ("during", 4),
    ("while", 4),
    ("when", 4),
    ("once", 4),
    ("always", 4),
    ("never", 4),
    ("often", 4),
];

/// Locative prepositions (where): the following content token is the place.
const LOCAL_PREPS: &[&str] = &[
    "in", "on", "at", "into", "onto", "under", "over", "above", "below", "near", "beside",
    "inside", "outside", "behind", "through", "within", "across",
];

/// Positive / negative polarity lexicon for the felt tone (qualia valence).
const POSITIVE: &[&str] = &[
    "joy", "hope", "light", "love", "found", "saved", "calm", "bright", "safe", "peace", "warm",
    "gentle", "clear", "won", "healed",
];
const NEGATIVE: &[&str] = &[
    "storm", "lost", "dark", "fear", "broke", "cold", "pain", "death", "fell", "hid", "grief",
    "wound", "sank", "burned", "failed", "doubt",
];

/// A verb iff the `verb_table` archetype recognises it (`read_verb` is `Some`).
/// No hand-rolled verb list — this is the operator's "consume the table" rule.
fn is_verb(w: &str) -> bool {
    read_verb(w).is_some()
}

/// A content token: alphabetic, length > 2, not a stopword, not a verb, not a
/// role cue (temporal adverb / causal connective / preposition). Role cues are
/// read separately, so they never pollute the S/P/O stream.
fn is_content(w: &str) -> bool {
    w.len() > 2
        && w.chars().all(|c| c.is_ascii_alphabetic())
        && !STOP.contains(&w)
        && !is_verb(w)
        && !is_copula(w) // are/been/being — transparent, never a term (Codex #843 P2)
        && !is_causal_cue(w)
        && !TEMPORAL_CUES.iter().any(|(c, _)| *c == w)
        && !LOCAL_PREPS.contains(&w)
}

/// Common `-ly` words that are NOT manner adverbs — nouns/adjectives whose
/// suffix would otherwise be misread as "how" (Codex #843 P2). The COCA-grounded
/// `insight_coca_read` gates this by real PoS instead of a denylist.
const LY_NONADVERB: &[&str] = &[
    "family", "supply", "apply", "reply", "rely", "ally", "assembly", "anomaly", "monopoly",
    "only", "early", "likely", "lonely", "ugly", "holy", "silly", "jolly", "rally", "bully",
    "belly", "folly", "italy", "july", "duly", "ripply", "wobbly",
];

/// A `-ly` manner adverb (`carefully`, `swiftly`) — the "how" morphology, gated
/// against the `-ly` non-adverbs (`family`, `supply`) and known verbs.
fn is_manner_adverb(w: &str) -> bool {
    w.len() > 4
        && w.ends_with("ly")
        && !STOP.contains(&w)
        && !LY_NONADVERB.contains(&w)
        && !is_verb(w)
}

/// Map an archetype [`Tense`] to the Temporal lane's coarse code (1 past / 2
/// present / 3 future / 4 relative). The verb's own tense, from the `verb_table`.
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

/// One fully-typed relation read from a clause: subject/predicate/object, the
/// verb's archetype reading (family + dominant slot), the four TEKAMOLO adverbial
/// lanes, and the 17D felt-quality vector.
#[derive(Default)]
struct Extraction {
    s: String,
    verb: String,
    o: String,
    family: Option<VerbFamily>,
    verb_slot: Option<TekamoloSlot>,
    temporal: [u8; 3],
    kausal: [u8; 3],
    modal: [u8; 3],
    lokal: [u8; 3],
    lokal_place: String,
    modal_word: String,
    /// The felt tone as the canonical **17D qualia vector** (`AXIS_LABELS`) — NOT
    /// a scalar. This vector IS the gestalt texture; `to_qualia` packs it to i4-16.
    qualia: QualiaVector,
}

impl Extraction {
    /// Pack the four adverbial lanes into a real `TekamoloFacet` (#839 address).
    fn to_tekamolo(&self, classid: u32) -> TekamoloFacet {
        TekamoloFacet::from_lanes(classid, self.temporal, self.kausal, self.modal, self.lokal)
    }

    /// Pack the 17D felt vector into value tenant #1 via the canonical quantizer.
    fn to_qualia(&self) -> QualiaI4_16D {
        QualiaI4_16D::from_f32_17d(&self.qualia)
    }
}

/// Read the **17D felt-quality vector** onto the canonical meaningful axes
/// (`AXIS_LABELS`). Each axis is filled from a deterministic text signal; axes
/// with no signal stay 0.0. Affective axes may be signed `[-1, 1]`.
fn qualia_vector(toks: &[String], svo_found: bool) -> QualiaVector {
    let mut pos = 0.0f32;
    let mut neg = 0.0f32;
    let mut fast = 0.0f32;
    let (mut has_temporal, mut has_present, mut has_future) = (false, false, false);
    let (mut has_causal, mut has_modal, mut has_place) = (false, false, false);
    for w in toks {
        let w = w.as_str();
        if POSITIVE.contains(&w) {
            pos += 1.0;
        }
        if NEGATIVE.contains(&w) {
            neg += 1.0;
        }
        if matches!(w, "quickly" | "swiftly" | "suddenly" | "rushed" | "raced") {
            fast += 1.0;
        }
        if let Some((_, tense)) = TEMPORAL_CUES.iter().find(|(c, _)| *c == w) {
            has_temporal = true;
            has_present |= *tense == 2;
            has_future |= *tense == 3;
        }
        if is_causal_cue(w) {
            has_causal = true;
        }
        if is_manner_adverb(w) {
            has_modal = true;
        }
        if LOCAL_PREPS.contains(&w) {
            has_place = true;
        }
    }
    let strong = (pos + neg).max(1.0);
    let mut q: QualiaVector = [0.0; QUALIA_DIMS];
    q[0] = ((pos + neg + fast) / strong).min(1.0); // arousal — affective intensity
    q[1] = (pos - neg) / strong; // valence — signed pleasant/unpleasant
    q[2] = (neg / strong).min(1.0); // tension — unresolved/threatening pressure
    q[3] = (pos / strong).min(1.0); // warmth — affection/tenderness
    q[4] = if svo_found { 0.8 } else { 0.2 }; // clarity — a clean SVO resolved
    q[6] = if has_causal { 0.7 } else { 0.0 }; // depth — reasoned (a "because")
    q[7] = (fast / strong).min(1.0); // velocity — speed of motion
    q[9] = if svo_found && (has_causal || has_temporal) {
        0.7
    } else {
        0.3
    }; // coherence
    q[11] = if has_present { 0.7 } else { 0.0 }; // presence — happening now
    q[12] = if has_modal { 0.6 } else { 0.0 }; // assertion — a chosen manner
    q[14] = if has_place { 0.7 } else { 0.0 }; // groundedness — anchored in a place
    q[15] = if has_future { 0.7 } else { 0.0 }; // expansion — opening toward next
    q
}

/// Extract one `Extraction` from a sentence's tokens. Role-cue tokens are consumed
/// as role signals; the remaining content/verb stream yields the first transitive
/// `subject —verb→ object` (arguments NP-chunked from adjacent content runs). The
/// verb is TYPED via `read_verb` (the `verb_table` archetype), which also supplies
/// the Temporal lane's tense.
fn extract(toks: &[String]) -> Extraction {
    let mut ex = Extraction::default();

    #[derive(PartialEq)]
    enum Item {
        Term(String),
        Verb(String),
    }
    let mut stream: Vec<Item> = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    let flush = |run: &mut Vec<&str>, stream: &mut Vec<Item>| {
        if !run.is_empty() {
            stream.push(Item::Term(run.join("_")));
            run.clear();
        }
    };
    let mut pos = 0usize;
    while pos < toks.len() {
        let w = toks[pos].as_str();
        if let Some((_, tense)) = TEMPORAL_CUES.iter().find(|(c, _)| *c == w) {
            let fine = TEMPORAL_CUES.iter().position(|(c, _)| *c == w).unwrap() as u8 + 1;
            if ex.temporal == [0, 0, 0] {
                ex.temporal = [*tense, fine, 0];
            }
            pos += 1;
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
            pos += 1;
            continue;
        }
        if is_manner_adverb(w) && !is_verb(w) {
            if ex.modal == [0, 0, 0] {
                ex.modal = [
                    2,
                    (w.bytes().fold(0u16, |a, b| a + b as u16) % 250) as u8 + 1,
                    0,
                ];
                ex.modal_word = w.to_string();
            }
            pos += 1;
            continue;
        }
        if LOCAL_PREPS.contains(&w) {
            let mut j = pos + 1;
            while j < toks.len() && !is_content(&toks[j]) && !is_verb(&toks[j]) {
                j += 1;
            }
            if j < toks.len() && is_content(&toks[j]) {
                let place = &toks[j];
                if ex.lokal == [0, 0, 0] {
                    ex.lokal = [
                        1,
                        (place.bytes().fold(0u16, |a, b| a + b as u16) % 250) as u8 + 1,
                        0,
                    ];
                    ex.lokal_place = place.clone();
                }
                flush(&mut run, &mut stream);
                pos = j + 1;
                continue;
            }
            pos += 1;
            continue;
        }
        if is_verb(w) {
            flush(&mut run, &mut stream);
            stream.push(Item::Verb(w.to_string()));
        } else if is_content(w) {
            run.push(w);
        } else {
            flush(&mut run, &mut stream);
        }
        pos += 1;
    }
    flush(&mut run, &mut stream);

    // First transitive subject —verb→ object over the stream.
    let mut i = 0;
    while i + 2 < stream.len() {
        if let (Item::Term(s), Item::Verb(v), Item::Term(o)) =
            (&stream[i], &stream[i + 1], &stream[i + 2])
        {
            ex.s = s.clone();
            ex.verb = v.clone();
            ex.o = o.clone();
            break;
        }
        i += 1;
    }

    // Type the predicate via the verb_table archetype — family + slot + tense.
    if let Some((family, tense, slot)) = read_verb(&ex.verb) {
        ex.family = Some(family);
        ex.verb_slot = Some(slot);
        // The verb's own tense fills the Temporal lane's tense byte if no
        // explicit temporal adverb already set it (archetype before hand-cue).
        if ex.temporal[0] == 0 {
            ex.temporal[0] = tense_code(tense);
        }
    }

    ex.qualia = qualia_vector(toks, !ex.verb.is_empty());
    ex
}

/// One sentence per `.`/`!`/`?`; the extractor runs per sentence.
fn sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Render the strongest `k` axes of a 17D qualia vector by their canonical
/// `AXIS_LABELS`, largest |value| first — the felt texture in words.
fn felt_axes(q: &QualiaVector, k: usize) -> String {
    let mut idx: Vec<usize> = (0..QUALIA_DIMS).filter(|&i| q[i].abs() > 0.05).collect();
    idx.sort_by(|&a, &b| q[b].abs().total_cmp(&q[a].abs()));
    idx.truncate(k);
    if idx.is_empty() {
        return "(flat — no felt signal)".to_string();
    }
    idx.iter()
        .map(|&i| format!("{}={:+.2}", AXIS_LABELS[i], q[i]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn report(label: &str, text: &str) -> Vec<(Extraction, TekamoloFacet, QualiaI4_16D)> {
    println!("\n════════ {label} ════════");
    let mut out = Vec::new();
    for (n, sent) in sentences(text).iter().enumerate() {
        let ex = extract(&tokens(sent));
        if ex.verb.is_empty() {
            continue;
        }
        let facet = ex.to_tekamolo(0x0000_0000);
        let qualia = ex.to_qualia();
        let tense = ["·", "past", "present", "future", "rel"][ex.temporal[0].min(4) as usize];
        println!(
            "  [{n}] {} —{}→ {}   [archetype: {} → {}]",
            ex.s.replace('_', " "),
            ex.verb,
            ex.o.replace('_', " "),
            ex.family
                .map(|f| format!("{f:?}"))
                .unwrap_or_else(|| "untyped".into()),
            ex.verb_slot
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| "—".into()),
        );
        println!(
            "        Te(when)={} [{}]  Ka(why)={}  Mo(how)={}  Lo(where)={}",
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
        );
        println!(
            "        Qualia(17D felt vector) = {}",
            felt_axes(&ex.qualia, 4)
        );
        debug_assert_eq!(facet.temporal(), ex.temporal);
        debug_assert_eq!(facet.lane(TekamoloRole::Kausal), ex.kausal);
        out.push((ex, facet, qualia));
    }
    let g = gestalt_texture(&out.iter().map(|(e, _, _)| e).collect::<Vec<_>>());
    println!("  — GESTALT TEXTURE (Familienaufstellung: what the text wants to feel/act like) —");
    println!("      {g}");
    out
}

/// The **gestalt texture resonance** — the Familienaufstellung read of the whole
/// constellation: every relation placed as a figure (its S/P/O, adverbial posture,
/// and 17D felt vector), and the collective question ("what does the text want to
/// feel or act like today?") answered by the mean qualia vector across the
/// constellation, read off the canonical meaningful axes.
///
/// Document-level qualia — the same 17D texture as the per-clause felt vector,
/// averaged over the placement. The canonical arena-backed measure is the shipped
/// `basin_resonance::rank_basins` (`resonance = staunen × wisdom`,
/// `E-SCI-INSIGHT-BASIN-RESONANCE-CLICK-1`) + `gestalt_texture_smoke` (#835); this
/// example, without an arena, aggregates the extractor's own felt vectors. A
/// [S]-grade texture read, not a metric claim.
fn gestalt_texture(exs: &[&Extraction]) -> String {
    if exs.is_empty() {
        return "(no relations — the constellation is empty)".to_string();
    }
    let n = exs.len() as f32;
    let mut mean: QualiaVector = [0.0; QUALIA_DIMS];
    for e in exs {
        for (m, &v) in mean.iter_mut().zip(e.qualia.iter()) {
            *m += v / n;
        }
    }
    let axis = |name: &str| mean[AXIS_LABELS.iter().position(|&l| l == name).unwrap()];
    let feeling = if axis("tension") > 0.4 && axis("valence") < -0.1 {
        "tense and shadowed (a pressure it hasn't resolved)"
    } else if axis("warmth") > 0.4 {
        "warm and tender"
    } else if axis("valence") > 0.2 {
        "bright, hopeful"
    } else if mean.iter().all(|v| v.abs() < 0.15) {
        "even and unhurried"
    } else {
        "sober, matter-of-fact"
    };
    let action = if axis("arousal") > 0.5 || axis("velocity") > 0.4 {
        "move quickly and decisively"
    } else if axis("assertion") > 0.4 {
        "act deliberately, in a chosen manner"
    } else if axis("groundedness") > 0.4 {
        "stay put, rooted in place"
    } else {
        "let things simply happen"
    };
    format!(
        "the text wants to FEEL {feeling} and to ACT — it wants to {action}. \
         [mean felt vector: {}]",
        felt_axes(&mean, 5)
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for path in &args {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    report(path, &text);
                }
                Err(e) => eprintln!("skip {path}: {e}"),
            }
        }
        return;
    }

    // ── Falsifier: one clause carrying all five roles, its verb typed by the
    //    verb_table archetype (`carry` → Supports). ──
    let text = "yesterday the mason carefully carried the stone into the hall \
                because the storm rose.";
    let out = report("inline falsifier", text);
    assert_eq!(out.len(), 1, "expected exactly one transitive relation");
    let (ex, facet, qualia) = &out[0];

    // S / P / O
    assert_eq!(ex.s, "mason", "subject");
    assert_eq!(ex.verb, "carried", "predicate (verb)");
    assert_eq!(ex.o, "stone", "object");
    // Predicate typed by the ARCHETYPE (not hand-rolled): carry → Supports.
    assert_eq!(
        ex.family,
        Some(VerbFamily::Supports),
        "verb family from verb_table"
    );
    assert!(
        ex.verb_slot.is_some(),
        "verb_table supplies the adverbial slot"
    );
    // Temporal = yesterday (adverb cue #1, past); Kausal/Modal/Lokal present.
    assert_eq!(ex.temporal[0], 1, "Temporal tense = past");
    assert_eq!(ex.temporal[1], 1, "Temporal adverb cue = yesterday");
    assert_eq!(
        ex.kausal[0], 1,
        "Kausal present (because via is_causal_cue)"
    );
    assert_eq!(ex.modal[0], 2, "Modal = a -ly manner adverb");
    assert_eq!(ex.modal_word, "carefully", "Modal word");
    assert_eq!(ex.lokal[0], 1, "Lokal present");
    assert_eq!(ex.lokal_place, "hall", "Lokal place");

    // Qualia = the 17D felt vector (NOT a scalar): storm → negative valence,
    // positive tension, several live axes.
    let vi = AXIS_LABELS.iter().position(|&l| l == "valence").unwrap();
    let ti = AXIS_LABELS.iter().position(|&l| l == "tension").unwrap();
    assert!(ex.qualia[vi] < 0.0, "storm clause → negative valence");
    assert!(ex.qualia[ti] > 0.0, "storm clause → positive tension");
    let live_axes = ex.qualia.iter().filter(|v| v.abs() > 0.05).count();
    assert!(
        live_axes >= 3,
        "qualia is a 17D texture, not a scalar — expected ≥3 live axes, got {live_axes}"
    );

    // The four lanes ARE the #839 TekamoloFacet address — round-trip.
    assert_eq!(facet.temporal(), ex.temporal, "facet Temporal round-trip");
    assert_eq!(
        facet.lane(TekamoloRole::Kausal),
        ex.kausal,
        "facet Kausal round-trip"
    );
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
    // The felt vector packs into value tenant #1 via from_f32_17d; valence sign kept.
    assert_eq!(
        *qualia,
        QualiaI4_16D::from_f32_17d(&ex.qualia),
        "qualia tenant is the canonical 17D→i4-16 packing"
    );
    assert!(
        qualia.get(vi) < 0,
        "valence channel stays negative in the packed tenant"
    );

    // ── Codex #843 P2 regressions ──
    // (a) a `-ly` NOUN (`family`) must not be consumed as a Modal cue — the
    //     relation survives with `family` as the subject.
    let ly = extract(&tokens("the family supports the roof"));
    assert_eq!(
        ly.s, "family",
        "a -ly noun stays the subject, not the manner"
    );
    assert_eq!(
        ly.verb, "supports",
        "the relation is not dropped by -ly overmatch"
    );
    assert_eq!(ly.modal, [0, 0, 0], "no Modal from a -ly noun");
    // (b) a copula (`are`) must be transparent — it does not chunk into the
    //     subject NP of the following real predicate.
    let cop = extract(&tokens("the stars are bright and pressure caused failure"));
    assert!(
        !cop.s.contains("are") && !cop.s.contains("star"),
        "copula must not chunk into the subject: got {}",
        cop.s
    );

    // ── Control: a bare transitive clause with NO adverbial cues → all-zero
    //    lanes except the archetype tense (support → present). ──
    let control = report(
        "control (no adverbial cues)",
        "the pillar supports the roof.",
    );
    assert_eq!(control.len(), 1);
    let (cex, cfacet, _) = &control[0];
    assert_eq!(cex.s, "pillar");
    assert_eq!(cex.o, "roof");
    assert_eq!(
        cex.family,
        Some(VerbFamily::Supports),
        "control verb typed by archetype"
    );
    assert_eq!(cfacet.temporal()[1], 0, "no temporal adverb cue");
    assert_eq!(
        cfacet.lane(TekamoloRole::Kausal),
        [0, 0, 0],
        "no cue → zero Kausal"
    );
    assert_eq!(
        cfacet.lane(TekamoloRole::Modal),
        [0, 0, 0],
        "no cue → zero Modal"
    );
    assert_eq!(
        cfacet.lane(TekamoloRole::Lokal),
        [0, 0, 0],
        "no cue → zero Lokal"
    );

    // ── Familienaufstellung: a whole-text gestalt over a dark constellation. ──
    let storm_text = "the flood dissolved the bridge. cold fear gripped the crew. \
                      doubt eroded the plan. the storm broke the mast.";
    let storm = report("gestalt constellation (a dark passage)", storm_text);
    let mean_v: f32 =
        storm.iter().map(|(e, _, _)| e.qualia[vi]).sum::<f32>() / storm.len().max(1) as f32;
    let mean_t: f32 =
        storm.iter().map(|(e, _, _)| e.qualia[ti]).sum::<f32>() / storm.len().max(1) as f32;
    assert!(
        mean_v < 0.0 && mean_t > 0.3,
        "the dark-passage constellation must resonate tense/shadowed \
         (mean valence {mean_v:+.2}, tension {mean_t:+.2})"
    );

    println!(
        "\n✔ Extracted S·P·O (verb TYPED via the #842 verb_table archetype — family + slot, \
         no hand-rolled connectives) + Temporal·Kausal·Modal·Lokal packed into a real \
         TekamoloFacet (#839) + the 17D Qualia felt vector into value tenant #1; the control \
         yielded all-zero adverbial lanes; the dark-passage constellation resonated \
         tense/shadowed as its gestalt texture (Familienaufstellung)."
    );
    println!(
        "\n(usage: cargo run -p lance-graph-planner --example insight_spo_tekamolo_read -- FILE [FILE ...])"
    );
}
