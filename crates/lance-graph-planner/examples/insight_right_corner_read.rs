//! `insight_right_corner_read` — D-SCI-1 Phase 1: **delayed clause commitment**
//! for fronted-argument ("Yoda-style" / KJV OSV) clauses.
//!
//! The bug this fixes is not a missing passive rule — it is premature
//! commitment: the left-corner SVO scan binds the first noun as subject, so
//! `him shall ye hear` (Deut 18:15 / Acts 3:22) either fails or binds wrong.
//! Phase 1 instead carries an **incomplete clause hypothesis** (an
//! `AwaitingClause`) and commits only when the right-corner lexical predicate
//! closes the frame:
//!
//! ```text
//! him     shall     ye      hear
//! O       AUX       S       V(right corner)
//! fronted awaiting  subject predicate  → canonical: ye —hear→ him (Future)
//! ```
//!
//! Decidability comes from **surviving case morphology** (the Core's new
//! [`clause_cues`] catalogue): an accusative pronoun (`him`/`thee`/`them`…)
//! clause-initially CANNOT be a subject, and the KJV preserves MORE case than
//! modern English (`ye`=nom / `you`=acc, `thou`/`thee`). Eroded forms
//! (`you`/`it`/`her`) are [`PronounCase::Ambiguous`] and NEVER commit on case
//! alone — the honest boundary.
//!
//! **Active canonicalization, not passive:** the stored semantic edge is
//! `S —V→ O` in active form (`Hear{agent: ye, theme: him}`); a passive
//! rewrite would be an orthopedic intermediate, never the stored meaning.
//! The clause **tense reads off the modal** (`shall`→Future) because the
//! right-corner verb surfaces as a bare infinitive.
//!
//! Phase 2 (auxiliary-chain passive generator) and Phase 3 (parallelism
//! eliminators) build on this; the whole-corpus falsifier is Phase 4.
//!
//! Data: same two-basin store as `insight_reason_wired` (Release assets,
//! gitignored — skips cleanly if absent).
//!
//! Usage: cargo run -p lance-graph-planner --example insight_right_corner_read -- [FILE ...]

use std::collections::HashMap;
use std::path::PathBuf;

use lance_graph_contract::codegen_spine::Triple;
use lance_graph_contract::grammar::clause_cues::{
    is_modal_aux, modal_tense, pronoun_case, PronounCase,
};
use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::verb_lexicon::read_verb;
use lance_graph_contract::tekamolo_facet::TekamoloFacet;

// ── data loading (same two-basin store as insight_reason_wired) ─────────────

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
                "missing Release data (COCA codebook `coca-codebook-v2`) — \
                 expected {}",
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

// ── the incomplete clause hypothesis (right-corner state) ───────────────────

/// The carried incomplete constituent: what has been seen, what is awaited.
/// Commitment happens only when `predicate` closes the frame at the right
/// corner — until then nothing is bound.
#[derive(Debug, Default)]
struct AwaitingClause {
    /// The fronted argument (an unambiguous accusative pronoun, Phase 1).
    fronted: Option<String>,
    /// The finite modal auxiliary (the left bracket) + the tense it projects.
    modal: Option<(String, Tense)>,
    /// The probable subject (unambiguous nominative pronoun, or a COCA noun).
    subject: Option<String>,
    /// The right-corner lexical predicate.
    predicate: Option<String>,
}

/// One committed clause reading: surface order + the canonical ACTIVE edge.
struct RightCornerReason {
    surface: String,
    canonical: Triple,
    tense: Tense,
    facet: TekamoloFacet,
}

fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn tense_code(t: Tense) -> u8 {
    match t {
        Tense::Past | Tense::PastContinuous | Tense::Pluperfect => 1,
        Tense::Present | Tense::PresentContinuous | Tense::Habitual | Tense::Imperative => 2,
        Tense::Future | Tense::FutureContinuous | Tense::FuturePerfect => 3,
        _ => 4, // Potential / Perfect — the "other" temporal coarse tier
    }
}

/// Leading discourse connectives that may precede a fronted clause
/// (`FOR unto thee will I pray`) — skipped, never part of the frame.
const CONNECTIVES: &[&str] = &["for", "and", "but", "then", "therefore", "yea", "now", "so"];

/// Prepositions that may head a fronted PP argument (`UNTO thee will I pray`).
/// The fronted argument is then the PP's accusative object. Deliberately ONLY
/// the recipient pair: locative/source preps (`upon`/`of`/`in`…) mark the
/// Lokal lane, not an object — the whole-KJV re-run showed `of them shall ye
/// buy` (them = source) and `upon thee shall he offer` (thee = place) would
/// otherwise commit role-wrong edges. Precision over recall, as ever.
const FRONT_PREPS: &[&str] = &["unto", "to"];

/// Phase-1 delayed-commitment scan: recognise `[CONN*] [PREP] O AUX S V` and
/// canonicalise to the active `S —V→ O`. Returns `None` (no premature binding)
/// unless every awaited slot fills — and NEVER commits on an
/// [`PronounCase::Ambiguous`] fronted form. The leading-connective and
/// fronted-PP handling is the codex-review fix: `for unto thee will I pray`
/// (Ps 5:2) previously only parsed by line-wrapping luck.
fn right_corner(b: &Basins, toks: &[String]) -> Option<RightCornerReason> {
    // Skip discourse connectives, then at most ONE preposition — and only when
    // it immediately precedes an unambiguous accusative pronoun (the PP object
    // is the fronted argument; anything else is not a case-decided front).
    let mut start = 0usize;
    while start < toks.len() && CONNECTIVES.contains(&toks[start].as_str()) {
        start += 1;
    }
    if start < toks.len()
        && FRONT_PREPS.contains(&toks[start].as_str())
        && toks
            .get(start + 1)
            .is_some_and(|w| pronoun_case(w) == Some(PronounCase::Accusative))
    {
        start += 1;
    }
    let mut aw = AwaitingClause::default();
    for w in &toks[start..] {
        match (&aw.fronted, &aw.modal, &aw.subject) {
            // Awaiting the fronted argument: only an UNAMBIGUOUS accusative
            // pronoun opens a right-corner hypothesis (case decides; eroded
            // forms never do).
            (None, _, _) => match pronoun_case(w) {
                Some(PronounCase::Accusative) => aw.fronted = Some(w.clone()),
                Some(PronounCase::Ambiguous) => return None, // you/it/her — case can't decide
                _ => return None, // nominative or non-pronoun start → not a fronted clause
            },
            // Awaiting the finite auxiliary (the left bracket).
            (Some(_), None, _) => {
                let t = modal_tense(w)?;
                aw.modal = Some((w.clone(), t));
            }
            // Awaiting the subject: unambiguous nominative pronoun, or a COCA noun.
            (Some(_), Some(_), None) => {
                let nom = pronoun_case(w) == Some(PronounCase::Nominative);
                let noun = b.pos(w) == Some(b'n');
                if nom || noun {
                    aw.subject = Some(w.clone());
                } else {
                    return None;
                }
            }
            // Awaiting the right-corner lexical predicate: a COCA verb or an
            // archetype-known verb closes the frame.
            (Some(_), Some(_), Some(_)) => {
                let verbish = b.pos(w) == Some(b'v')
                    || read_verb(w).or_else(|| read_verb(b.lemma(w))).is_some();
                if verbish && !is_modal_aux(w) {
                    aw.predicate = Some(w.clone());
                    break; // frame closed
                }
                return None;
            }
        }
    }
    let (o, (_, tense), s, v) = (aw.fronted?, aw.modal?, aw.subject?, aw.predicate?);
    let facet = TekamoloFacet::from_lanes(0, [tense_code(tense), 0, 0], [0; 3], [0; 3], [0; 3]);
    Some(RightCornerReason {
        surface: format!("O({o}) AUX S({s}) V({v})"),
        canonical: Triple {
            s,
            p: v,
            o,
            f: 1.0,
            c: 0.9,
        },
        tense,
        facet,
    })
}

/// The plain left-corner SVO scan (control — the two machines coexist; the
/// right-corner pass only fires on case-decided fronted clauses).
fn left_corner(b: &Basins, toks: &[String]) -> Option<Triple> {
    let is_verb = |w: &str| {
        b.pos(w) == Some(b'v') && read_verb(w).or_else(|| read_verb(b.lemma(w))).is_some()
    };
    let is_noun = |w: &str| b.pos(w) == Some(b'n');
    for i in 0..toks.len() {
        if !is_verb(&toks[i]) {
            continue;
        }
        let s = toks[..i].iter().rev().find(|w| is_noun(w));
        let o = toks[i + 1..].iter().find(|w| is_noun(w));
        if let (Some(s), Some(o)) = (s, o) {
            return Some(Triple {
                s: s.clone(),
                p: toks[i].clone(),
                o: o.clone(),
                f: 1.0,
                c: 0.9,
            });
        }
    }
    None
}

fn report(b: &Basins, label: &str, text: &str) -> (Vec<RightCornerReason>, Vec<Triple>) {
    println!("\n════════ {label} ════════");
    let (mut rc, mut lc) = (Vec::new(), Vec::new());
    for sent in text
        .split(['.', ';', ':', '?', '!', ',', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let toks = tokens(sent);
        if let Some(r) = right_corner(b, &toks) {
            println!("  「{sent}」");
            println!("    surface:   {}", r.surface);
            println!(
                "    canonical: {} —{}→ {}   [tense: {:?} (from modal) · Te lane {:?}]",
                r.canonical.s,
                r.canonical.p,
                r.canonical.o,
                r.tense,
                r.facet.temporal(),
            );
            rc.push(r);
        } else if let Some(t) = left_corner(b, &toks) {
            println!("  「{sent}」  left-corner: {} —{}→ {}", t.s, t.p, t.o);
            lc.push(t);
        } else {
            println!("  「{sent}」  (no committed parse — honest incompleteness)");
        }
    }
    (rc, lc)
}

fn main() {
    let b = match Basins::load() {
        Ok(b) => b,
        Err(h) => {
            eprintln!("{h}");
            return;
        }
    };
    println!("loaded COCA lexicon: {} entries", b.lex.len());

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for p in &args {
            match std::fs::read_to_string(p) {
                Ok(t) => {
                    report(&b, p, &t);
                }
                Err(e) => eprintln!("skip {p}: {e}"),
            }
        }
        return;
    }

    // ── Falsifier ────────────────────────────────────────────────────────────
    // 1) The KJV fronted clause (Deut 18:15 / Acts 3:22) — case-decided OSV.
    // 2) A plain SVO control — must stay on the left-corner machine.
    // 3) An eroded-case front (`you`) — must NOT commit on case alone.
    // 4) Codex #849 regression: fronted PP after a connective + `:` splitting
    //    (Ps 5:2 `…my God: for unto thee will I pray` — previously reachable
    //    only by line-wrapping luck).
    // 5) Codex #849 regression: `?` terminates a unit — two fronted clauses in
    //    one string must yield TWO commitments, not one.
    let (rc, lc) = report(
        &b,
        "right-corner falsifier",
        "him shall ye hear. the shepherd carries the lamb. you shall hear them. \
         hearken unto my god: for unto thee will i pray. \
         him shall ye hear? them shall he serve?",
    );

    // 1) `him shall ye hear` canonicalises ACTIVE with roles recovered by case.
    let r = &rc[0];
    assert_eq!(
        r.canonical.s, "ye",
        "subject recovered from nominative `ye`"
    );
    assert_eq!(
        r.canonical.p, "hear",
        "right-corner predicate closes the frame"
    );
    assert_eq!(
        r.canonical.o, "him",
        "fronted accusative `him` is the object"
    );
    assert_eq!(r.tense, Tense::Future, "tense reads off the modal `shall`");
    assert_eq!(
        r.facet.temporal()[0],
        3,
        "Future in the Temporal coarse tier"
    );

    // 2) The SVO control stayed left-corner (the two machines coexist).
    assert_eq!(lc.len(), 1, "one left-corner control");
    assert_eq!(lc[0].s, "shepherd");
    assert_eq!(lc[0].o, "lamb");

    // 3) `you shall hear them` did NOT right-corner-commit: `you` is
    //    case-eroded (Ambiguous), so the clause falls through to honest
    //    incompleteness — no parse beats a WRONG parse (premature certainty
    //    is the bug Phase 1 exists to remove).
    assert!(
        !rc.iter().any(|r| r.canonical.o == "you"),
        "eroded `you` must never be committed as a fronted object on case alone"
    );

    // 4) Fronted PP (Ps 5:2): the `:` split isolates `for unto thee will i
    //    pray`; the connective + preposition are skipped; `thee` (accusative)
    //    fronts, `i` (nominative) is the subject — by structure, not luck.
    let pp = rc
        .iter()
        .find(|r| r.canonical.o == "thee")
        .expect("`for unto thee will i pray` must right-corner-commit");
    assert_eq!(pp.canonical.s, "i");
    assert_eq!(pp.canonical.p, "pray");
    assert_eq!(pp.tense, Tense::Future, "tense reads off the modal `will`");

    // 5) `?` is a unit boundary: BOTH fronted questions commit.
    assert!(
        rc.iter()
            .any(|r| r.canonical.o == "them" && r.canonical.s == "he" && r.canonical.p == "serve"),
        "the second `?`-terminated clause must commit (he —serve→ them)"
    );
    assert_eq!(
        rc.len(),
        4,
        "four right-corner commitments: hear/pray + the two question clauses"
    );

    println!(
        "\n✔ Phase 1: delayed clause commitment — the fronted clause is carried as an \
         incomplete hypothesis (AwaitingClause) and committed only when the right-corner \
         predicate closes the frame; roles recovered by surviving case morphology \
         (KJV ye/you, thou/thee — MORE case than modern English); canonical form is \
         ACTIVE (the passive would be orthopedic, never the stored meaning); tense reads \
         off the modal. Eroded forms (you/it/her) never commit on case alone. \
         Phase 2 = auxiliary-chain passive generator; Phase 3 = parallelism eliminators; \
         Phase 4 = whole-corpus replay gate."
    );
    println!("\n(usage: cargo run -p lance-graph-planner --example insight_right_corner_read -- FILE [FILE ...])");
}
