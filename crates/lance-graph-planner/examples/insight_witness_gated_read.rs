//! `insight_witness_gated_read` — D-SCI-1 Phase 2: **witness-gated candidate
//! generation** — the Greek treebank as a typed construction license.
//!
//! The witness does NOT vote on the English answer. It describes the attested
//! clause geometry of the source passage ([`ClauseSignature`]); the English
//! side mints bounded candidates; the constraint engine confirms, weakens, or
//! stays silent ([`WitnessDisposition`]). The relation:
//!
//! ```text
//! candidate generation:   English evidence OR witness construction evidence
//! candidate elimination:  all available typed constraints
//! ```
//!
//! Phase 2 licenses exactly ONE new English candidate class beyond Phase 1's
//! case-decided pronouns: the **fronted NP object** (`A prophet shall the
//! Lord your God raise up` — O AUX S V with a noun, not a pronoun, fronted).
//! English case morphology cannot decide an NP, so this candidate commits
//! ONLY when the passage's witness clause licenses object-fronted-active
//! (fronted `obj`/`obl` dependent + agentive voice). Pronoun-case commitments
//! (Phase 1) stay witness-INDEPENDENT — the witness adds a receipt, never a
//! veto: `TextAbsent` / `AlignmentUnknown` never block.
//!
//! Dependency-first (the probe's warning shots, `grammar::witness` docs):
//! ἀκούω governs the GENITIVE (`αὐτοῦ` is `obl`, not accusative `obj`) and
//! its future is MIDDLE — so the license reads relations + [`VoiceClass`],
//! never raw case, never an active/passive binary.
//!
//! Data (gitignored, never committed):
//!   - COCA lexicon → `examples/data/coca/` (Release `coca-codebook-v2`)
//!   - PROIEL Greek NT → `examples/data/proiel/greek-nt.xml`
//!     (github.com/proiel/proiel-treebank, CC BY-NC-SA) or `$PROIEL_GREEK_NT`
//!
//! Usage: cargo run -p lance-graph-planner --example insight_witness_gated_read

use std::collections::HashMap;
use std::path::PathBuf;

use lance_graph_contract::codegen_spine::Triple;
use lance_graph_contract::grammar::clause_cues::{
    is_modal_aux, modal_tense, pronoun_case, PronounCase,
};
use lance_graph_contract::grammar::witness::{ClauseSignature, VoiceClass, WitnessDisposition};

const EDITION: &str = "PROIEL-greek-nt (critical text — NOT the TR the KJV translates)";

/// Seed lexical bridge English-verb-lemma → Greek lemma, for clause MATCHING
/// only (never translation). Tiny by design: the future Strong's-number bridge
/// replaces it; an English predicate absent here yields `AlignmentUnknown` —
/// honest ignorance, never a veto.
const VERB_BRIDGE: &[(&str, &str)] = &[
    ("hear", "ἀκούω"),
    ("raise", "ἀνίστημι"),
    ("say", "λέγω"),
    ("give", "δίδωμι"),
    ("send", "πέμπω"),
];

// ── COCA basin (PoS + lemma, as in the sibling examples) ────────────────────

fn dir(env: &str, sub: &str) -> PathBuf {
    if let Ok(d) = std::env::var(env) {
        return PathBuf::from(d);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/data")
        .join(sub)
}

struct Basins {
    lex: HashMap<String, (String, u8)>,
}

impl Basins {
    fn load() -> Result<Self, String> {
        let coca = dir("COCA_CODEBOOK_DIR", "coca").join("lexicon.tsv");
        let txt = std::fs::read_to_string(&coca).map_err(|_| {
            format!(
                "missing COCA codebook (Release `coca-codebook-v2`) — {}",
                coca.display()
            )
        })?;
        let mut lex = HashMap::new();
        for l in txt.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let c: Vec<&str> = l.split('\t').collect();
            // Guard the first byte: a row with an empty PoS field must be
            // skipped, never panic (CodeRabbit #849 hardening).
            if let (3.., Some(&pos)) = (c.len(), c.get(2).and_then(|s| s.as_bytes().first())) {
                lex.insert(c[0].to_string(), (c[1].to_string(), pos));
            }
        }
        Ok(Self { lex })
    }
    fn pos(&self, w: &str) -> Option<u8> {
        self.lex.get(w).map(|(_, p)| *p)
    }
    fn lemma<'a>(&'a self, w: &'a str) -> &'a str {
        self.lex.get(w).map(|(l, _)| l.as_str()).unwrap_or(w)
    }
}

// ── PROIEL → clause signatures (one per verb token, dependency-first) ───────

fn attr<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("{name}=\"");
    let start = line.find(&pat)? + pat.len();
    let end = line[start..].find('"')? + start;
    Some(&line[start..end])
}

/// Parse the PROIEL XML (token-per-line format) into per-citation clause
/// signatures: one [`ClauseSignature`] per verb token, its arguments = the
/// verb's DIRECT dependents (`head-id` == verb id), fronted = dependents
/// appearing before the verb. Voice from morphology position 5.
fn load_witness(path: &PathBuf) -> Result<HashMap<String, Vec<ClauseSignature>>, String> {
    let txt = std::fs::read_to_string(path).map_err(|_| {
        format!(
            "missing PROIEL Greek NT — expected {} \
             (github.com/proiel/proiel-treebank greek-nt.xml, or $PROIEL_GREEK_NT)",
            path.display()
        )
    })?;
    // (id, citation, lemma, pos, morph, head, relation, order)
    struct Tok {
        id: String,
        citation: String,
        lemma: String,
        is_verb: bool,
        voice: VoiceClass,
        head: String,
        relation: String,
        order: usize,
    }
    let mut out: HashMap<String, Vec<ClauseSignature>> = HashMap::new();
    let mut sent: Vec<Tok> = Vec::new();
    let flush = |sent: &mut Vec<Tok>, out: &mut HashMap<String, Vec<ClauseSignature>>| {
        for v in sent.iter().filter(|t| t.is_verb) {
            let deps: Vec<&Tok> = sent.iter().filter(|t| t.head == v.id).collect();
            let arg = |t: &Tok| matches!(t.relation.as_str(), "obj" | "obl" | "sub" | "xobj");
            let sig = ClauseSignature {
                citation: v.citation.clone(),
                edition: EDITION.to_string(),
                clause_index: 0, // stamped below, per citation
                predicate_lemma: v.lemma.clone(),
                subject_expressed: deps.iter().any(|t| t.relation == "sub"),
                fronted_relations: deps
                    .iter()
                    .filter(|t| arg(t) && t.order < v.order)
                    .map(|t| t.relation.clone())
                    .collect(),
                argument_relations: deps
                    .iter()
                    .filter(|t| arg(t))
                    .map(|t| t.relation.clone())
                    .collect(),
                voice: v.voice,
            };
            let list = out.entry(v.citation.clone()).or_default();
            let mut sig = sig;
            sig.clause_index = list.len() as u16;
            list.push(sig);
        }
        sent.clear();
    };
    for line in txt.lines() {
        let l = line.trim_start();
        if l.starts_with("<token ") {
            // Empty tokens (ellipsis) carry no form — skip them as clause heads
            // but they can still be heads of dependents; keep them if lemma'd.
            let (Some(id), Some(cit)) = (attr(l, "id"), attr(l, "citation-part")) else {
                continue;
            };
            let lemma = attr(l, "lemma").unwrap_or("").to_string();
            let pos = attr(l, "part-of-speech").unwrap_or("");
            let morph = attr(l, "morphology").unwrap_or("----------");
            let order = sent.len();
            sent.push(Tok {
                id: id.to_string(),
                citation: cit.to_string(),
                lemma,
                is_verb: pos.starts_with('V'),
                voice: VoiceClass::from_proiel_code(morph.chars().nth(4).unwrap_or('-')),
                head: attr(l, "head-id").unwrap_or("").to_string(),
                relation: attr(l, "relation").unwrap_or("").to_string(),
                order,
            });
        } else if l.starts_with("</sentence>") {
            flush(&mut sent, &mut out);
        }
    }
    flush(&mut sent, &mut out);
    Ok(out)
}

// ── witness matching: (citation, English predicate) → disposition + receipt ─

fn consult(
    witness: &HashMap<String, Vec<ClauseSignature>>,
    citation: &str,
    english_verb_lemma: &str,
) -> (WitnessDisposition, Option<ClauseSignature>) {
    // Corpus scope outranks bridge ignorance: a passage outside the witness
    // corpus (e.g. an OT citation vs a Greek-NT treebank) is TextAbsent no
    // matter what the seed bridge covers.
    let Some(clauses) = witness.get(citation) else {
        return (WitnessDisposition::TextAbsent, None);
    };
    let Some(greek) = VERB_BRIDGE
        .iter()
        .find(|(e, _)| *e == english_verb_lemma)
        .map(|(_, g)| *g)
    else {
        return (WitnessDisposition::AlignmentUnknown, None);
    };
    match clauses.iter().find(|c| c.predicate_lemma == greek) {
        Some(c) if c.licenses_fronted_object_active() => {
            (WitnessDisposition::Confirmed, Some(c.clone()))
        }
        // No fronting attested ≠ contradiction — translations reorder freely.
        Some(c) => (WitnessDisposition::Compatible, Some(c.clone())),
        // The verse exists in the witness but carries no such predicate:
        // textual-tradition difference (the TR clause the critical text lacks).
        None => (WitnessDisposition::TextAbsent, None),
    }
}

// ── English candidate generators ────────────────────────────────────────────

fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Phase-1 pronoun-case right-corner scan (witness-INDEPENDENT — case decides).
fn pronoun_fronted(b: &Basins, toks: &[String]) -> Option<Triple> {
    let mut it = toks.iter();
    let o = it
        .next()
        .filter(|w| pronoun_case(w) == Some(PronounCase::Accusative))?;
    let _m = it.next().filter(|w| modal_tense(w).is_some())?;
    let s = it
        .next()
        .filter(|w| pronoun_case(w) == Some(PronounCase::Nominative) || b.pos(w) == Some(b'n'))?;
    let v = it
        .next()
        .filter(|w| b.pos(w) == Some(b'v') && !is_modal_aux(w))?;
    Some(Triple {
        s: s.clone(),
        p: v.clone(),
        o: o.clone(),
        f: 1.0,
        c: 0.9,
    })
}

/// Phase-2 fronted-NP candidate: `[det] N … MODAL … N* V` — English morphology
/// CANNOT decide this (nouns carry no case), so the caller commits it only
/// under a witness license. Object = last noun before the modal; subject =
/// last noun between modal and verb; verb = first COCA verb after the modal.
fn np_fronted_candidate(b: &Basins, toks: &[String]) -> Option<Triple> {
    let m = toks.iter().position(|w| is_modal_aux(w))?;
    let o = toks[..m].iter().rev().find(|w| b.pos(w) == Some(b'n'))?;
    let v_rel = toks[m + 1..]
        .iter()
        .position(|w| b.pos(w) == Some(b'v') && !is_modal_aux(w))?;
    let v = &toks[m + 1 + v_rel];
    let s = toks[m + 1..m + 1 + v_rel]
        .iter()
        .rev()
        .find(|w| b.pos(w) == Some(b'n'))?;
    Some(Triple {
        s: s.clone(),
        p: v.clone(),
        o: o.clone(),
        f: 1.0,
        c: 0.8, // NP fronting: below the case-decided 0.9 until witnessed
    })
}

// ── the falsifier ───────────────────────────────────────────────────────────

struct Case {
    citation: &'static str,
    clause: &'static str,
}

fn main() {
    let b = match Basins::load() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let xml = if let Ok(p) = std::env::var("PROIEL_GREEK_NT") {
        PathBuf::from(p)
    } else {
        dir("PROIEL_DIR", "proiel").join("greek-nt.xml")
    };
    let witness = match load_witness(&xml) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    println!(
        "loaded: COCA {} entries · PROIEL witness {} cited passages",
        b.lex.len(),
        witness.len()
    );

    let cases = [
        // Phase-1 pronoun commitment; witness clause EXISTS in the critical
        // text at 3:22 (αὐτοῦ ἀκούσεσθε — fronted obl, future middle).
        Case {
            citation: "ACTS 3.22",
            clause: "him shall ye hear",
        },
        // Same English clause cited at 7:37 — present in the TR the KJV
        // translates, ABSENT from the critical text: TextAbsent, surfaced.
        Case {
            citation: "ACTS 7.37",
            clause: "him shall ye hear",
        },
        // Outside the witness corpus entirely (OT vs Greek-NT treebank).
        Case {
            citation: "DEUT 6.13",
            clause: "him shalt thou serve",
        },
        // The Phase-2 recall hole: fronted NP — English case cannot decide;
        // the Greek clause (προφήτην obj fronted + θεός sub + future ACTIVE)
        // licenses object-fronted-active. Attested at BOTH 3:22 and 7:37.
        Case {
            citation: "ACTS 7.37",
            clause: "a prophet shall the lord your god raise up unto you",
        },
        // Negative control: NP-fronted surface shape whose witness gives no
        // license (no carry-verb bridge entry → AlignmentUnknown → no commit).
        Case {
            citation: "ACTS 3.22",
            clause: "the shepherd shall the lamb carry",
        },
    ];

    let mut committed: Vec<(String, Triple, WitnessDisposition)> = Vec::new();
    for c in &cases {
        let toks = tokens(c.clause);
        println!("\n「{}」 @ {}", c.clause, c.citation);
        // Phase 1: case-decided pronoun fronting (witness-independent).
        if let Some(t) = pronoun_fronted(&b, &toks) {
            let (disp, receipt) = consult(&witness, c.citation, b.lemma(&t.p));
            println!(
                "  commit (case-decided): {} —{}→ {}   [witness: {disp:?}]",
                t.s, t.p, t.o
            );
            if let Some(r) = receipt {
                println!(
                    "    receipt: clause #{} `{}` fronted={:?} voice={:?} subj_expressed={}",
                    r.clause_index,
                    r.predicate_lemma,
                    r.fronted_relations,
                    r.voice,
                    r.subject_expressed
                );
            }
            assert_ne!(
                disp,
                WitnessDisposition::Contradicted,
                "never silently contradicted"
            );
            committed.push((c.citation.into(), t, disp));
            continue;
        }
        // Phase 2: fronted-NP candidate — commits ONLY under a witness license.
        if let Some(t) = np_fronted_candidate(&b, &toks) {
            let (disp, receipt) = consult(&witness, c.citation, b.lemma(&t.p));
            if disp == WitnessDisposition::Confirmed {
                let r = receipt.expect("Confirmed always carries its licensing clause");
                println!(
                    "  commit (witness-licensed NP-fronting): {} —{}→ {}",
                    t.s, t.p, t.o
                );
                println!(
                    "    receipt: {} clause #{} `{}` fronted={:?} voice={:?} — {}",
                    r.citation,
                    r.clause_index,
                    r.predicate_lemma,
                    r.fronted_relations,
                    r.voice,
                    r.edition
                );
                committed.push((c.citation.into(), t, disp));
            } else {
                println!(
                    "  candidate held (NP fronting needs a license): {} —{}→ {}  [witness: {disp:?}]",
                    t.s, t.p, t.o
                );
            }
            continue;
        }
        println!("  (no candidate — honest incompleteness)");
    }

    // ── assertions: ChatGPT-revised gate ────────────────────────────────────
    // 1) Acts 3:22 pronoun commit is witness-CONFIRMED via the government-verb
    //    clause (fronted obl + middle voice — dependency-first, not case).
    let a322 = committed
        .iter()
        .find(|(c, t, _)| c == "ACTS 3.22" && t.o == "him")
        .unwrap();
    assert_eq!(a322.2, WitnessDisposition::Confirmed);
    assert_eq!(a322.1.s, "ye");
    // 2) Acts 7:37 same clause: committed (English evidence suffices) with the
    //    textual-tradition difference SURFACED as TextAbsent, never a veto.
    let a737 = committed
        .iter()
        .find(|(c, t, _)| c == "ACTS 7.37" && t.o == "him")
        .unwrap();
    assert_eq!(a737.2, WitnessDisposition::TextAbsent);
    // 3) OT citation outside the witness corpus: committed + TextAbsent.
    let deut = committed.iter().find(|(c, _, _)| c == "DEUT 6.13").unwrap();
    assert_eq!(deut.2, WitnessDisposition::TextAbsent);
    // 4) THE RECALL TEST: the fronted-NP clause commits with correct roles,
    //    licensed by the Greek clause geometry (baseline Phase 1: a miss).
    let np = committed.iter().find(|(_, t, _)| t.o == "prophet").unwrap();
    assert_eq!(
        np.1.s, "god",
        "postverbal-subject recovery: god —raise→ prophet"
    );
    assert_eq!(np.1.p, "raise");
    assert_eq!(np.2, WitnessDisposition::Confirmed);
    // 5) The negative control did NOT commit (AlignmentUnknown gates NP
    //    fronting — but note it held the candidate, it did not erase it).
    assert!(
        !committed.iter().any(|(_, t, _)| t.o == "lamb"),
        "unlicensed NP fronting must hold, not commit"
    );

    println!(
        "\n✔ Phase 2: the witness issues typed construction licenses, never answers. \
         Dependency outranks case (αὐτοῦ is genitive `obl` under ἀκούω — and still licenses); \
         voice is a class, not a binary (future middle ἀκούσεσθε licenses agentive); \
         missing text is evidence, not elimination (KJV's TR clause at Acts 7:37 surfaces as \
         TextAbsent; OT citations outside the Greek-NT corpus likewise). English case-decided \
         commitments stay witness-independent; the fronted-NP class commits ONLY under a \
         license — and `a prophet shall the lord your god raise up` goes from Phase-1 miss \
         to witness-licensed `god —raise→ prophet`, with the Greek clause as receipt."
    );
}
