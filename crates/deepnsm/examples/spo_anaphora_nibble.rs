//! The nibble anaphora edge: a SIGNED 4-bit offset (-8..+7) from a pronoun
//! token to its antecedent within the +-8 window, so the SPO fact-building
//! spine (`E-SPO-MARKOV-KG-SPINE-1`) fills a role slot with the
//! ANTECEDENT's lemma-centroid instead of the opaque pronoun. This is the
//! coreference leg of the operator's endgame ("resolving sentences for
//! relativPronomen/anaphora"); TEKAMOLO already exists
//! (`lance_graph_contract::grammar::tekamolo`) and is NOT rebuilt here.
//!
//! ## Why a nibble
//!
//! An antecedent within +-8 tokens fits a SIGNED nibble: `-8..+7`. Negative
//! = the antecedent is BEHIND (the normal case, "the dog ... it"); the range
//! is the ±8 window the operator named. The nibble is an EDGE on the pronoun
//! node pointing back to the resolved noun -- the same "edge = signed offset"
//! shape as the Morton motion codes (`E-X265-MORTON-SHIFT-1`), here over
//! token positions instead of pixels.
//!
//! ## Resolution (a STATED heuristic, not full coreference)
//!
//!   - PERSONAL (he/she/it/they): nearest preceding noun within 8 that AGREES
//!     in number (they=plural) and animacy (he/she=animate; it prefers
//!     inanimate, falls back to animate). Cross-sentence allowed (recency).
//!   - RELATIVE (that/which/who): the nearest preceding noun (the head it
//!     modifies), same window; `who` requires an animate head.
//!   - No agreeing antecedent in +-8 -> UNRESOLVED (pleonastic "it rained"),
//!     nibble = sentinel 0, slot stays the pronoun.
//!
//! This demonstrates the MECHANISM (a nibble edge resolves a pronoun to an
//! antecedent by a stated rule, and the SPO slot is rewritten to the
//! antecedent's lemRank centroid). Full coreference (world knowledge,
//! salience models) is out of scope -- the honest boundary, same as
//! `E-SURFACE-FORM-COLLAPSE-1`'s S/O simplification.
//!
//! KILL gates (regressions, not discoveries):
//!   - every labeled pronoun resolves to its expected antecedent.
//!   - every resolved nibble is in `-8..=7` and points BACKWARD (< 0).
//!   - the pleonastic pronoun is left UNRESOLVED (nibble 0).
//!   - a resolved pronoun's SPO slot is rewritten to the antecedent lemma.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example spo_anaphora_nibble
//! ```

use std::collections::HashMap;

/// A content noun's agreement features.
#[derive(Clone, Copy)]
struct Noun {
    animate: bool,
    plural: bool,
}

/// The pronoun classes the resolver handles.
#[derive(Clone, Copy, PartialEq)]
enum Pron {
    Personal { animate: Option<bool>, plural: bool }, // he/she/it/they
    Relative { animate_head: bool },                  // that/which/who
}

fn noun_features(lemma: &str) -> Option<Noun> {
    match lemma {
        "man" | "dog" | "girl" | "bird" => Some(Noun {
            animate: true,
            plural: false,
        }),
        "girls" | "men" | "dogs" => Some(Noun {
            animate: true,
            plural: true,
        }),
        "book" | "car" | "house" => Some(Noun {
            animate: false,
            plural: false,
        }),
        _ => None,
    }
}

fn pronoun(lemma: &str) -> Option<Pron> {
    match lemma {
        "he" | "she" => Some(Pron::Personal {
            animate: Some(true),
            plural: false,
        }),
        "it" => Some(Pron::Personal {
            animate: None,
            plural: false,
        }), // prefers inanimate
        "they" => Some(Pron::Personal {
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

/// Load `lemma -> rank` (the gridlake centroid address of each resolved noun).
fn load_ranks(csv_path: &str) -> HashMap<String, u32> {
    let text = std::fs::read_to_string(csv_path).unwrap_or_default();
    let mut m = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        if let Ok(r) = f[0].trim().parse::<u32>() {
            m.entry(f[1].to_ascii_lowercase()).or_insert(r);
        }
    }
    m
}

const WINDOW: i32 = 8;

/// Pleonastic (non-referential) "it": detected by its PREDICATE, not the
/// window -- "it rained" / "it snowed" have no antecedent however near a noun
/// sits. This is the honest rule (recency alone cannot tell referential from
/// pleonastic "it"); the weather/existential verb is the signal.
fn is_pleonastic_it(stream: &[&str], pos: usize) -> bool {
    stream[pos] == "it"
        && matches!(
            stream.get(pos + 1),
            Some(&("rained" | "rain" | "snowed" | "snow"))
        )
}

/// Resolve a pronoun at `pos` in `stream` to an antecedent noun index within
/// the +-8 window. Returns `Some(antecedent_pos)` or `None` (unresolved).
fn resolve(stream: &[&str], pos: usize, p: Pron) -> Option<usize> {
    // scan backward, nearest first, at most WINDOW tokens
    let lo = pos.saturating_sub(WINDOW as usize);
    let mut cands: Vec<usize> = (lo..pos).rev().collect();
    // relative pronoun binds the IMMEDIATELY preceding noun (its head)
    if let Pron::Relative { animate_head } = p {
        for &k in &cands {
            if let Some(nf) = noun_features(stream[k]) {
                if !animate_head || nf.animate {
                    return Some(k);
                }
                return None; // head found but wrong animacy
            }
        }
        return None;
    }
    // personal pronoun: nearest AGREEING noun; for `it`, prefer inanimate then animate
    let Pron::Personal { animate, plural } = p else {
        return None;
    };
    let agrees = |nf: &Noun, want_anim: Option<bool>| -> bool {
        nf.plural == plural && want_anim.is_none_or(|a| nf.animate == a)
    };
    // pass 1: strict animacy (it -> inanimate)
    let want = animate.or(Some(false).filter(|_| animate.is_none()));
    for &k in &cands {
        if let Some(nf) = noun_features(stream[k]) {
            if agrees(&nf, want) {
                return Some(k);
            }
        }
    }
    // pass 2 (only `it`): fall back to any number-agreeing noun (animate ok)
    if animate.is_none() {
        cands = (lo..pos).rev().collect();
        for &k in &cands {
            if let Some(nf) = noun_features(stream[k]) {
                if nf.plural == plural {
                    return Some(k);
                }
            }
        }
    }
    None
}

fn main() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let ranks = load_ranks(csv);

    // Synthetic corpus (COCA lemmas). `.` marks sentence ends (kept in-stream
    // so recency crosses boundaries, as anaphora does). Each pronoun's
    // EXPECTED antecedent is labeled for the KILL gate.
    let stream: Vec<&str> = vec![
        "the", "man", "read", "a", "book", ".", // 0..6
        "he", "liked", "it", ".", // 6:he->man, 8:it->book
        "the", "girls", "played", ".", // 10..14
        "they", "won", ".", // 14:they->girls
        "the", "car", "that", "broke", ".", // 18..23  20:that->car
        "it", "rained", ".", // 23:it-> UNRESOLVED (pleonastic; book is >8 back)
    ];
    // expected: stream position of pronoun -> Some(antecedent lemma) / None
    // (that@19 binds its head car@18; it@22 is pleonastic "it rained")
    let expected: &[(usize, Option<&str>)] = &[
        (6, Some("man")),
        (8, Some("book")),
        (14, Some("girls")),
        (19, Some("car")),
        (22, None),
    ];

    let rank_of = |w: &str| {
        ranks
            .get(w)
            .map(|r| r.to_string())
            .unwrap_or_else(|| "?".into())
    };

    println!("nibble anaphora edges (+-8 signed offset), then SPO-slot rewrite to the antecedent");
    println!();
    let mut resolved: HashMap<usize, (usize, i32)> = HashMap::new();
    for (pos, &tok) in stream.iter().enumerate() {
        if let Some(p) = pronoun(tok) {
            if is_pleonastic_it(&stream, pos) {
                println!(
                    "  '{tok}'@{pos} --nibble  0--> (pleonastic 'it {}')       [unresolved]",
                    stream[pos + 1]
                );
                continue;
            }
            match resolve(&stream, pos, p) {
                Some(ant) => {
                    let nibble = ant as i32 - pos as i32; // negative = behind
                    resolved.insert(pos, (ant, nibble));
                    println!(
                        "  '{tok}'@{pos} --nibble {nibble:+}--> '{}'@{ant} (r{})   [resolved]",
                        stream[ant],
                        rank_of(stream[ant])
                    );
                }
                None => {
                    println!("  '{tok}'@{pos} --nibble  0--> (none in +-8)          [unresolved / pleonastic]");
                }
            }
        }
    }
    println!();

    // SPO-slot rewrite: a resolved pronoun in a role slot is replaced by the
    // antecedent's lemma-centroid. Demo on "he liked it" -> (man, like, book).
    println!("SPO-slot rewrite (the spine fills the slot with the antecedent, not the pronoun):");
    let subj = resolved.get(&6).map(|&(a, _)| stream[a]).unwrap_or("he");
    let obj = resolved.get(&8).map(|&(a, _)| stream[a]).unwrap_or("it");
    println!(
        "  'he liked it'  ->  ({subj} r{}) --like--> ({obj} r{})   [pronouns dissolved into centroids]",
        rank_of(subj),
        rank_of(obj)
    );
    println!();

    // KILL gates.
    let mut fail = Vec::new();
    for &(pos, exp) in expected {
        match (resolved.get(&pos), exp) {
            (Some(&(ant, nib)), Some(lem)) => {
                if stream[ant] != lem {
                    fail.push(format!(
                        "@{pos} resolved to '{}' expected '{lem}'",
                        stream[ant]
                    ));
                }
                if !(-8..=7).contains(&nib) || nib >= 0 {
                    fail.push(format!("@{pos} nibble {nib} out of range or not backward"));
                }
            }
            (None, None) => {} // correct: pleonastic unresolved
            (Some(&(ant, _)), None) => {
                fail.push(format!(
                    "@{pos} resolved to '{}' but expected UNRESOLVED",
                    stream[ant]
                ));
            }
            (None, Some(lem)) => fail.push(format!("@{pos} unresolved but expected '{lem}'")),
        }
    }
    if subj != "man" || obj != "book" {
        fail.push(format!(
            "SPO rewrite gave ({subj},like,{obj}), expected (man,like,book)"
        ));
    }
    if fail.is_empty() {
        println!("KILL GATES: all pass -- nibble edges resolve pronouns to their antecedents");
        println!("within +-8, the pleonastic 'it' stays unresolved, and the SPO slots are");
        println!("rewritten to the antecedents' lemRank centroids (pronouns dissolved).");
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
