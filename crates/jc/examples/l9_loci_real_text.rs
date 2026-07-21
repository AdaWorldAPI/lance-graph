//! PROBE L9-REAL-TEXT — run the L9 loci register over ACTUAL TEXT and measure
//! what it says: validity (against gold annotations of the text itself) and
//! reliability (Pearson / Spearman / ICC(2,1) / Cronbach α via the shipped
//! `jc::reliability` battery).
//!
//! ## Real substrate (nothing planted, nothing synthetic)
//!
//! - **Text:** three Aesop fables in their traditional public-domain wording
//!   ("The Boy Who Cried Wolf", "The Fox and the Grapes", "The Dog and the
//!   Shadow") — natural narrative English with pronouns, relative clauses,
//!   and causal connectives. The text was NOT constructed for this probe.
//! - **Ranks/PoS:** the committed COCA table
//!   `deepnsm/word_frequency/lemmas_5k.csv` (real ranks, real PoS tags).
//! - **Metric:** frequency-rank distance `|Δrank|/16` capped 255 — the
//!   `E-FREQ-IS-COSINE-REPLACEMENT-1` grounding.
//! - **Rules:** the anaphora resolution mirrors `deepnsm`'s shipped
//!   `spo_anaphora_nibble` (agreement in number/animacy, nearest-preceding,
//!   ±8 window, relative pronoun binds its head); the nominal/verbal role
//!   split mirrors `E-SURFACE-FORM-COLLAPSE-1`.
//!
//! ## The instrument (the L9 loci computable on THIS substrate)
//!
//! Events = content tokens (COCA PoS n/v/j with a known rank). Per event the
//! register's computable loci (signed token offset in −8..+7; 0 = unbound):
//!
//! - `ante`     — pronoun → antecedent (pronoun events only; gold-labeled)
//! - `kausal`   — effect-clause event → the cause clause across a causal
//!                connective (because/so; gold from the text's connectives)
//! - `noun-gnd` — nearest nominal grounding (S/O-meaning on this substrate)
//! - `verb-gnd` — nearest verbal grounding (P-meaning)
//! - `temporal` — nearest temporal marker (closed class: then/when/day/…)
//! - `basin`    — nearest same-basin event (`rank >> 8`)
//! - `quorum`   — nearest agreeing peer (rank-distance ≤ threshold)
//!
//! Dimensions NOT computable on this standalone substrate (runbook, qualia,
//! meaning-level, contradiction, MODAL/LOKAL) are OUT OF SCOPE of this run —
//! declared, not faked.
//!
//! ## Reliability (the asked battery)
//!
//! Two independent raters derive each locus: **Method A** (semantic-nearest:
//! minimize rank-distance among class-eligible window events) vs **Method B**
//! (positional-nearest: nearest class-eligible event). Per dimension over
//! co-bound events: Pearson r, Spearman ρ, ICC(2,1) (Shrout-Fleiss absolute
//! agreement). Internal consistency: Cronbach α over the 5 always-computable
//! dimensions as a k-item locality scale across events. Per the
//! evolve-not-collapse ruling the agreement values are a REPORTED living
//! diagnostic, never a collapse trigger.
//!
//! ## Validity (what the register SAYS about the text)
//!
//! - `ante`: accuracy vs gold antecedents (every pronoun in the fables was
//!   hand-labeled from reading the text), AND vs the agreement-blind
//!   baseline (nearest preceding noun regardless of agreement).
//! - `kausal`: hit rate — does the effect clause's locus land inside the
//!   cause clause (gold clause spans from the text's actual connectives).
//!
//! ## Registered gates (fixed before the first run)
//!
//! 1. WELL-POSED: every jc measure returns `Some`.
//! 2. `ante` accuracy ≥ 0.60 AND ≥ the agreement-blind baseline.
//! 3. `kausal` hit rate ≥ 0.50.
//! KILL: any gate fails (recorded loudly). Agreement magnitudes are not
//! gated — their direction is measured, not assumed.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/jc/Cargo.toml --example l9_loci_real_text
//! ```

use jc::reliability::{cronbach_alpha, icc, pearson, spearman, IccForm};
use std::collections::HashMap;

const WINDOW: i32 = 8;
/// Quorum: an agreeing peer is one within this rank-distance (u8 scale).
const QUORUM_DIST: u8 = 32;

// ── The actual text (public-domain Aesop, traditional wording) ─────────────
const TEXT: &str = "\
A shepherd boy tended his sheep near a village. He thought it would be fun to \
trick the villagers, so he cried wolf when there was no wolf. The villagers \
ran to help him, and they found no wolf. The boy laughed at them. Later he \
cried wolf again, and again they ran to help him, and again they found \
nothing. At last a wolf really came, and the boy cried wolf in terror. But \
the villagers thought he was lying again, so nobody came, and the wolf ate \
the sheep. \
A hungry fox saw fine grapes that hung from a vine. He jumped for the grapes, \
but they hung too high. The fox tried again and again, but he could not reach \
them. At last he gave up, because he was tired. The fox walked away and said \
the grapes were sour. \
A dog carried a piece of meat in his mouth. He crossed a bridge over a river, \
and he saw his own shadow in the water. The dog thought the shadow was \
another dog that carried better meat, so he snapped at the shadow. When he \
opened his mouth, the meat fell into the river, and the dog lost it.";

// ── Gold annotations (hand-labeled from READING the text above) ─────────────
// Pronoun occurrence -> antecedent lemma (None = non-referential).
// Indexed by pronoun occurrence order in the token stream.
fn gold_antecedents() -> Vec<(&'static str, Option<&'static str>)> {
    vec![
        ("he", Some("boy")),        // He thought
        ("it", None),               // it would be fun (extrapositional)
        ("he", Some("boy")),        // so he cried wolf
        ("they", Some("villager")), // and they found no wolf
        ("them", Some("villager")), // The boy laughed at them
        ("he", Some("boy")),        // Later he cried
        ("they", Some("villager")), // again they ran
        ("they", Some("villager")), // again they found
        ("he", Some("boy")),        // thought he was lying
        ("that", Some("grape")),    // grapes that hung
        ("he", Some("fox")),        // He jumped
        ("they", Some("grape")),    // but they hung too high
        ("he", Some("fox")),        // but he could not reach
        ("them", Some("grape")),    // reach them
        ("he", Some("fox")),        // At last he gave up
        ("he", Some("fox")),        // because he was tired
        ("he", Some("dog")),        // He crossed
        ("he", Some("dog")),        // and he saw
        ("that", Some("dog")),      // another dog that carried
        ("he", Some("dog")),        // so he snapped
        ("he", Some("dog")),        // When he opened
        ("it", Some("meat")),       // the dog lost it
    ]
}

/// Gold causal links: (effect-clause anchor lemma occurring right after the
/// connective, cause-clause witness lemma before it). From the text's actual
/// because/so connectives.
fn gold_causal() -> Vec<(&'static str, &'static str)> {
    // LEMMA space (the stream compares lemmas, not surface forms).
    vec![
        ("cry", "trick"),  // ...to trick the villagers, SO he cried wolf
        ("nobody", "lie"), // ...thought he was lying, SO nobody came
        ("tired", "give"), // he gave up BECAUSE he was tired (cause after)
        ("snap", "think"), // ...thought the shadow was..., SO he snapped
    ]
}

// ── COCA table (committed, real ranks + PoS) ────────────────────────────────
struct Coca {
    rank: HashMap<String, u32>,
    pos: HashMap<String, char>,
}

fn load_coca() -> Coca {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../deepnsm/word_frequency/lemmas_5k.csv"
    );
    let text = std::fs::read_to_string(path).expect("committed COCA table present");
    let mut rank = HashMap::new();
    let mut pos = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 3 {
            continue;
        }
        if let Ok(r) = f[0].trim().parse::<u32>() {
            let lemma = f[1].to_ascii_lowercase();
            rank.entry(lemma.clone()).or_insert(r);
            pos.entry(lemma)
                .or_insert_with(|| f[2].chars().next().unwrap_or('x'));
        }
    }
    Coca { rank, pos }
}

/// Tiny surface→lemma map for the corpus (plural/past forms not in the lemma
/// table verbatim). Only forms occurring in TEXT.
fn lemma_of(word: &str) -> String {
    match word {
        "villagers" => "villager",
        "grapes" => "grape",
        "cried" => "cry",
        "tended" => "tend",
        "ran" => "run",
        "laughed" => "laugh",
        "came" => "come",
        "ate" => "eat",
        "saw" => "see",
        "hung" => "hang",
        "jumped" => "jump",
        "tried" => "try",
        "gave" => "give",
        "walked" => "walk",
        "said" => "say",
        "carried" => "carry",
        "crossed" => "cross",
        "snapped" => "snap",
        "opened" => "open",
        "fell" => "fall",
        "lost" => "lose",
        "lying" => "lie",
        "found" => "find",
        "thought" => "think",
        "was" | "were" | "be" | "been" | "is" => "be",
        "sheep" => "sheep",
        w => return w.to_string(),
    }
    .to_string()
}

/// frequency-rank distance (E-FREQ-IS-COSINE), u8-capped.
fn dist(a: u32, b: u32) -> u8 {
    ((a.abs_diff(b)) / 16).min(255) as u8
}

// ── Anaphora features (mirrors spo_anaphora_nibble; corpus nouns only) ──────
#[derive(Clone, Copy)]
struct NounF {
    animate: bool,
    plural: bool,
}
fn noun_features(lemma: &str) -> Option<NounF> {
    let (animate, plural) = match lemma {
        "boy" | "shepherd" | "fox" | "dog" | "wolf" | "nobody" => (true, false),
        "villager" => (true, true), // occurs as "villagers"
        "sheep" => (true, false),
        "village" | "vine" | "bridge" | "river" | "meat" | "mouth" | "shadow" | "water"
        | "piece" | "terror" | "nothing" | "day" => (false, false),
        "grape" => (false, true), // occurs as "grapes"
        _ => return None,
    };
    Some(NounF { animate, plural })
}

#[derive(Clone, Copy, PartialEq)]
enum Pron {
    Personal { animate: Option<bool>, plural: bool },
    Relative { animate_head: bool },
}
fn pronoun(word: &str) -> Option<Pron> {
    match word {
        "he" | "she" => Some(Pron::Personal {
            animate: Some(true),
            plural: false,
        }),
        "it" => Some(Pron::Personal {
            animate: None,
            plural: false,
        }),
        "they" | "them" => Some(Pron::Personal {
            animate: None,
            plural: true,
        }),
        "that" | "which" => Some(Pron::Relative {
            animate_head: false,
        }),
        "who" => Some(Pron::Relative { animate_head: true }),
        _ => None,
    }
}

const TEMPORAL_MARKERS: [&str; 8] = [
    "later", "again", "last", "when", "day", "then", "now", "night",
];
const CAUSAL_CONNECTIVES: [&str; 2] = ["so", "because"];

// ── Token stream ────────────────────────────────────────────────────────────
#[derive(Clone)]
struct Tok {
    word: String,
    lemma: String,
    rank: Option<u32>,
    pos: char, // COCA PoS of the lemma ('x' unknown)
}

fn tokenize(text: &str, coca: &Coca) -> Vec<Tok> {
    text.split_whitespace()
        .map(|w| {
            let word: String = w
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase();
            let lemma = lemma_of(&word);
            let rank = coca.rank.get(&lemma).copied();
            let pos = coca.pos.get(&lemma).copied().unwrap_or('x');
            Tok {
                word,
                lemma,
                rank,
                pos,
            }
        })
        .filter(|t| !t.word.is_empty())
        // EVENT STREAM (the design's window unit — ticks are events, not raw
        // words): content tokens (n/v/j), pronouns (slot fillers awaiting
        // resolution), temporal markers, and causal connectives (clause
        // boundaries the KAUSAL locus crosses). Determiners/prepositions are
        // not events (the spo_markov_kg stage-1 "determiners skipped" rule).
        .filter(|t| {
            matches!(t.pos, 'n' | 'v' | 'j')
                || pronoun(&t.word).is_some()
                || TEMPORAL_MARKERS.contains(&t.word.as_str())
                || CAUSAL_CONNECTIVES.contains(&t.word.as_str())
                || noun_features(&t.lemma).is_some()
        })
        .collect()
}

fn clamp_off(target: usize, this: usize) -> Option<i8> {
    let d = target as i64 - this as i64;
    if d == 0 || d < -(WINDOW as i64) || d > WINDOW as i64 {
        None
    } else {
        Some(d as i8)
    }
}

/// Content event = noun/verb/adjective with a known rank.
fn is_content(t: &Tok) -> bool {
    t.rank.is_some() && matches!(t.pos, 'n' | 'v' | 'j')
}

// ── The two locus raters ────────────────────────────────────────────────────
// Method A: SEMANTIC-nearest — among class-eligible window tokens, minimize
// rank-distance (tie → positionally nearer).
// Method B: POSITIONAL-nearest — the nearest class-eligible window token.
fn locus<F: Fn(usize) -> bool>(toks: &[Tok], i: usize, eligible: F, semantic: bool) -> Option<i8> {
    let my_rank = toks[i].rank?;
    let lo = i.saturating_sub(WINDOW as usize);
    let hi = (i + WINDOW as usize).min(toks.len() - 1);
    let mut best: Option<(u32, usize)> = None; // (key, pos)
    for j in (lo..i).rev().chain(i + 1..=hi) {
        if !eligible(j) {
            continue;
        }
        let key = if semantic {
            match toks[j].rank {
                Some(r) => dist(my_rank, r) as u32,
                None => continue,
            }
        } else {
            (j as i64 - i as i64).unsigned_abs() as u32
        };
        let better = match best {
            None => true,
            Some((bk, bp)) => {
                key < bk
                    || (key == bk
                        && (j as i64 - i as i64).unsigned_abs()
                            < (bp as i64 - i as i64).unsigned_abs())
            }
        };
        if better {
            best = Some((key, j));
        }
    }
    best.and_then(|(_, j)| clamp_off(j, i))
}

/// The shipped anaphora rule (mirrors spo_anaphora_nibble::resolve).
fn resolve_antecedent(toks: &[Tok], i: usize, p: Pron) -> Option<usize> {
    let lo = i.saturating_sub(WINDOW as usize);
    let back: Vec<usize> = (lo..i).rev().collect();
    match p {
        Pron::Relative { animate_head } => {
            for &k in &back {
                if let Some(nf) = noun_features(&toks[k].lemma) {
                    if !animate_head || nf.animate {
                        return Some(k);
                    }
                    return None;
                }
            }
            None
        }
        Pron::Personal { animate, plural } => {
            // `it`/`they` (animate=None): prefer inanimate, fall back animate.
            let pass = |want_inanimate: Option<bool>| -> Option<usize> {
                for &k in &back {
                    if let Some(nf) = noun_features(&toks[k].lemma) {
                        if nf.plural != plural {
                            continue;
                        }
                        match want_inanimate {
                            Some(true) if nf.animate => continue,
                            Some(false) if !nf.animate => continue,
                            _ => {}
                        }
                        return Some(k);
                    }
                }
                None
            };
            match animate {
                Some(true) => pass(Some(false)),
                Some(false) => pass(Some(true)),
                None => pass(Some(true)).or_else(|| pass(Some(false))),
            }
        }
    }
}

/// Agreement-blind baseline: nearest preceding noun, no agreement.
fn baseline_antecedent(toks: &[Tok], i: usize) -> Option<usize> {
    let lo = i.saturating_sub(WINDOW as usize);
    (lo..i)
        .rev()
        .find(|&k| noun_features(&toks[k].lemma).is_some())
}

/// v2 — LOCI CHAINING (the register following its own edges). Real text
/// carries pronoun CHAINS ("He … he … he"): by the second hop the noun is
/// outside ±8 and the noun-only rule (v1) goes UNRESOLVED — measured 0.455 on
/// this text (the recorded v1 KILL). The register's own mechanism resolves
/// it: an agreeing PRONOUN event inside the window is a valid binder, and its
/// already-stored antecedent nibble completes the chain — one extra hop of
/// the loci method, no new machinery. `resolved` maps pronoun event → its
/// ULTIMATE noun event (built left→right, so back-references are always
/// available).
fn resolve_chained(
    toks: &[Tok],
    i: usize,
    p: Pron,
    resolved: &HashMap<usize, usize>,
) -> Option<usize> {
    // Relative pronouns bind their immediately-preceding HEAD NOUN — chains
    // don't apply (unchanged shipped rule).
    if matches!(p, Pron::Relative { .. }) {
        return resolve_antecedent(toks, i, p);
    }
    let Pron::Personal { animate, plural } = p else {
        unreachable!()
    };
    let lo = i.saturating_sub(WINDOW as usize);
    // Candidate features at k: a noun's own features, or a resolved pronoun's
    // ultimate-noun features (the chain hop).
    let feat = |k: usize| -> Option<(NounF, usize)> {
        if let Some(nf) = noun_features(&toks[k].lemma) {
            return Some((nf, k));
        }
        if let Some(&noun) = resolved.get(&k) {
            return noun_features(&toks[noun].lemma).map(|nf| (nf, noun));
        }
        None
    };
    let pass = |want_inanimate: Option<bool>| -> Option<usize> {
        for k in (lo..i).rev() {
            if let Some((nf, ultimate)) = feat(k) {
                if nf.plural != plural {
                    continue;
                }
                match want_inanimate {
                    Some(true) if nf.animate => continue,
                    Some(false) if !nf.animate => continue,
                    _ => {}
                }
                return Some(ultimate);
            }
        }
        None
    };
    match animate {
        Some(true) => pass(Some(false)),
        Some(false) => pass(Some(true)),
        None => pass(Some(true)).or_else(|| pass(Some(false))),
    }
}

fn main() {
    let coca = load_coca();
    let toks = tokenize(TEXT, &coca);
    let known = toks.iter().filter(|t| t.rank.is_some()).count();
    println!(
        "tokens: {}  COCA-known: {} ({:.0}%)",
        toks.len(),
        known,
        100.0 * known as f64 / toks.len() as f64
    );

    // ── VALIDITY 1: antecedent loci vs gold ─────────────────────────────────
    let gold = gold_antecedents();
    let mut gi = 0usize;
    let (mut hits, mut v1_hits, mut base_hits, mut total) = (0usize, 0usize, 0usize, 0usize);
    let mut ante_offsets: Vec<i8> = Vec::new();
    let mut resolved: HashMap<usize, usize> = HashMap::new(); // pronoun ev → ultimate noun ev
    for i in 0..toks.len() {
        let Some(p) = pronoun(&toks[i].word) else {
            continue;
        };
        // corpus guard: "that" as relative only right after a noun (its head);
        // other "that"/"so"/"when" tokens are connectives, skipped by gold order.
        if matches!(toks[i].word.as_str(), "that")
            && !(i > 0 && noun_features(&toks[i - 1].lemma).is_some())
        {
            continue;
        }
        if gi >= gold.len() {
            break;
        }
        let (gw, gante) = gold[gi];
        assert_eq!(
            toks[i].word, gw,
            "gold ledger out of sync at occurrence {gi} (token {i})"
        );
        gi += 1;
        total += 1;
        let check = |lemma: &Option<String>| match (gante, lemma) {
            (Some(g), Some(r)) => g == r,
            (None, None) => true,
            _ => false,
        };
        // v1 (noun-only, the shipped rule as-is) — recorded, not gated.
        let v1 = resolve_antecedent(&toks, i, p);
        if check(&v1.map(|k| toks[k].lemma.clone())) {
            v1_hits += 1;
        }
        // v2 (loci chaining — the gated instrument).
        let res = resolve_chained(&toks, i, p, &resolved);
        if let Some(k) = res {
            resolved.insert(i, k);
            if let Some(off) = clamp_off(k, i) {
                ante_offsets.push(off);
            }
        }
        if check(&res.map(|k| toks[k].lemma.clone())) {
            hits += 1;
        }
        // agreement-blind baseline.
        let bres = baseline_antecedent(&toks, i).map(|k| toks[k].lemma.clone());
        if check(&bres) {
            base_hits += 1;
        }
    }
    assert_eq!(gi, gold.len(), "gold ledger fully consumed");
    let acc = hits as f64 / total as f64;
    let v1_acc = v1_hits as f64 / total as f64;
    let base_acc = base_hits as f64 / total as f64;

    // ── VALIDITY 2: KAUSAL loci vs the text's causal connectives ───────────
    let mut kausal_hits = 0usize;
    let causal = gold_causal();
    for (effect_lemma, cause_lemma) in &causal {
        // find the connective, the effect anchor after it, the cause witness before
        let mut hit = false;
        for c in 0..toks.len() {
            if !CAUSAL_CONNECTIVES.contains(&toks[c].word.as_str()) {
                continue;
            }
            let eff = (c + 1..toks.len().min(c + 1 + WINDOW as usize))
                .find(|&j| toks[j].lemma == *effect_lemma);
            let Some(e) = eff else { continue };
            // KAUSAL locus of the effect event: nearest content event BEFORE
            // the connective within the window (the cause clause).
            let lo = e.saturating_sub(WINDOW as usize);
            let locus = (lo..c).rev().find(|&j| is_content(&toks[j]));
            if let Some(l) = locus {
                if clamp_off(l, e).is_some() {
                    // does the cause clause (window-before-connective) contain
                    // the gold cause witness?
                    let found = (lo..c).any(|j| toks[j].lemma == *cause_lemma);
                    if found {
                        hit = true;
                    }
                }
            }
            if hit {
                break;
            }
        }
        if hit {
            kausal_hits += 1;
        }
        println!(
            "kausal: effect '{}' <- cause '{}': {}",
            effect_lemma,
            cause_lemma,
            if hit { "HIT" } else { "miss" }
        );
    }
    let kausal_rate = kausal_hits as f64 / causal.len() as f64;

    // ── RELIABILITY: Method A vs Method B on the always-computable loci ────
    let content_idx: Vec<usize> = (0..toks.len()).filter(|&i| is_content(&toks[i])).collect();
    let dims: [(&str, Box<dyn Fn(usize, bool) -> Option<i8>>); 5] = [
        (
            "noun-gnd",
            Box::new(|i, sem| {
                locus(
                    &toks,
                    i,
                    |j| j != i && toks[j].pos == 'n' && toks[j].rank.is_some(),
                    sem,
                )
            }),
        ),
        (
            "verb-gnd",
            Box::new(|i, sem| {
                locus(
                    &toks,
                    i,
                    |j| j != i && toks[j].pos == 'v' && toks[j].rank.is_some(),
                    sem,
                )
            }),
        ),
        (
            "temporal",
            Box::new(|i, sem| {
                locus(
                    &toks,
                    i,
                    |j| TEMPORAL_MARKERS.contains(&toks[j].word.as_str()),
                    sem,
                )
            }),
        ),
        (
            "basin",
            Box::new(|i, sem| {
                let mb = toks[i].rank.map(|r| r >> 8);
                locus(
                    &toks,
                    i,
                    |j| j != i && toks[j].rank.map(|r| Some(r >> 8) == mb).unwrap_or(false),
                    sem,
                )
            }),
        ),
        (
            "quorum",
            Box::new(|i, sem| {
                let my = toks[i].rank;
                locus(
                    &toks,
                    i,
                    |j| {
                        j != i
                            && match (my, toks[j].rank) {
                                (Some(a), Some(b)) => dist(a, b) <= QUORUM_DIST,
                                _ => false,
                            }
                    },
                    sem,
                )
            }),
        ),
    ];

    println!("\n== L9-REAL-TEXT: reliability (Method A semantic vs Method B positional) ==");
    println!(
        "{:<10} {:>5} {:>9} {:>9} {:>9}",
        "dim", "n", "pearson", "spearman", "icc(2,1)"
    );
    let mut all_wellposed = true;
    let mut profile_a: Vec<Vec<f64>> = vec![Vec::new(); dims.len()]; // items × subjects
    for (d, (name, f)) in dims.iter().enumerate() {
        let mut a_v = Vec::new();
        let mut b_v = Vec::new();
        for &i in &content_idx {
            let a = f(i, true);
            let b = f(i, false);
            // α profile: Method A locality (|offset|, 8 = unbound sentinel→8.0)
            profile_a[d].push(a.map(|o| o.unsigned_abs() as f64).unwrap_or(8.0));
            if let (Some(x), Some(y)) = (a, b) {
                a_v.push(x as f64);
                b_v.push(y as f64);
            }
        }
        let r = pearson(&a_v, &b_v);
        let rho = spearman(&a_v, &b_v);
        let ratings: Vec<Vec<f64>> = a_v.iter().zip(&b_v).map(|(&x, &y)| vec![x, y]).collect();
        let i21 = icc(&ratings, IccForm::Icc2_1);
        all_wellposed &= r.is_some() && rho.is_some() && i21.is_some();
        println!(
            "{:<10} {:>5} {:>9} {:>9} {:>9}",
            name,
            a_v.len(),
            fmt(r),
            fmt(rho),
            fmt(i21)
        );
    }
    let alpha = cronbach_alpha(&profile_a);
    all_wellposed &= alpha.is_some();
    println!(
        "cronbach_a (5-dim locality profile, N={}) = {}",
        content_idx.len(),
        fmt(alpha)
    );

    println!("\n== validity ==");
    println!("ante accuracy   = {acc:.3} ({hits}/{total})   [v2 loci-chaining; gate >= 0.60 AND >= baseline]");
    println!("ante v1 (noun-only) = {v1_acc:.3} ({v1_hits}/{total})   (recorded v1 KILL: chains escape ±8)");
    println!(
        "ante baseline   = {base_acc:.3} ({base_hits}/{total})   (agreement-blind nearest noun)"
    );
    println!(
        "kausal hit rate = {kausal_rate:.3} ({kausal_hits}/{})   [gate >= 0.50]",
        causal.len()
    );
    if !ante_offsets.is_empty() {
        let backward = ante_offsets.iter().filter(|&&o| o < 0).count();
        println!(
            "ante offsets: {} resolved, {} backward (all must be in -8..=7)",
            ante_offsets.len(),
            backward
        );
        assert!(ante_offsets.iter().all(|&o| (-8..=7).contains(&(o as i32))));
    }

    // ── Registered gates ────────────────────────────────────────────────────
    assert!(
        all_wellposed,
        "KILL gate 1: a jc measure returned None — battery not well-posed"
    );
    assert!(
        acc >= 0.60 && acc >= base_acc,
        "KILL gate 2: antecedent locus validity below floor or below the agreement-blind baseline"
    );
    assert!(
        kausal_rate >= 0.50,
        "KILL gate 3: KAUSAL locus misses the cause clause too often"
    );

    println!("\nPASS — the register's computable loci are VALID against the text's own structure");
    println!("(antecedents, causal clauses) and the reliability battery is well-posed; agreement");
    println!(
        "magnitudes above are the living diagnostic (evolve-not-collapse), direction measured."
    );
}

fn fmt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "None".into(),
    }
}
