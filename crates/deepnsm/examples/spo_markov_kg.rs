//! The fact-building spine: text -> SPO 2^3 facts -> +-5 CausalEdge window
//! fill -> Markov/NARS reasoning fills the remaining -> stored as a knowledge
//! graph. The affect layer (qualia sign = irony/sarcasm) is deferred; this
//! example proves the SPINE that layer sits on.
//!
//! This is the DeepNSM realization of the operator's endgame, spine-first:
//!
//!   "SPO 2^3 represents actual facts that get filled by CausalEdge context
//!    from +5/-5, and then reasoning fills the remaining with Markov chain
//!    context building ... reasoning about supporting basins ... stored as
//!    knowledge graph."
//!
//! It composes three already-recorded findings into one runnable pipeline:
//!
//!   - `E-SURFACE-FORM-COLLAPSE-1` -- the SPO 2^3 role mask collapses a token
//!     to a (lemma, role) fact slot (Subject/Object nominal, Predicate verbal).
//!   - the Markov +-5 trajectory / `temporal.rs` stream -- the CausalEdge
//!     context window that lands DIRECTLY-supported facts.
//!   - `E-ARM-DISCOVERY-REASONING-BRIDGE-1` + `I-SUBSTRATE-MARKOV` (the NARS
//!     revision arc IS the Markov trajectory) -- transitive deduction +
//!     property inheritance down the `is_a` basin fill the facts NO window
//!     ever stated.
//!
//! ## The three stages
//!
//!   1. COLLAPSE: each sentence's tokens -> (lemma, Role) via a tiny PoS FSM
//!      (determiners skipped; copula/verb = Predicate; nouns = Subject before
//!      the predicate, Object after).
//!   2. +-5 FILL: for each Predicate token, the Subject is the nearest noun
//!      within 5 to its left and the Object the nearest noun within 5 to its
//!      right -- the CausalEdge (S -predicate-> O), landed from the window.
//!      Each direct fact carries a NARS truth <freq=1.0, conf=0.9>.
//!   3. MARKOV/NARS FILL: the remaining facts are DEDUCED, not read --
//!      transitive `is_a` (dog is_a pet, pet is_a animal |- dog is_a animal)
//!      and property inheritance down the `is_a` basin (dog is_a pet, pet
//!      need food |- dog need food). NARS deduction truth: f = f1*f2,
//!      c = c1*c2*0.9 (monotone weaker than any premise).
//!
//! The result is an AriGraph-shaped knowledge graph: nodes carry their COCA
//! `lemRank` (the frequency-centroid gridlake address from
//! `E-FREQ-IS-COSINE-REPLACEMENT-1`), edges carry NARS truth. A query reads
//! direct + deduced facts, deduced strictly weaker.
//!
//! Honest boundary: a small SYNTHETIC corpus of COCA lemmas (the real
//! ngrams.info COCA windows are licensed / gitignored -- see
//! `gridlake_spo_ngrams`). The corpus is the demonstrator; the PIPELINE (+-5
//! fill -> NARS gap-fill -> store) is the claim, KILL-gated on a fact being
//! DEDUCED that no window stated.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example spo_markov_kg
//! ```

use std::collections::HashMap;

/// A NARS truth value: frequency (evidence ratio) and confidence.
#[derive(Clone, Copy)]
struct Truth {
    f: f64,
    c: f64,
}
impl Truth {
    /// NARS deduction: chaining two beliefs weakens both terms.
    fn deduce(self, other: Truth) -> Truth {
        Truth {
            f: self.f * other.f,
            c: self.c * other.c * 0.9,
        }
    }
}

/// The SPO role a collapsed token fills. Subject and Object are the same
/// nominal class at collapse time (`Noun`); which one a noun fills is
/// resolved by POSITION relative to the predicate at +-5 fill time (before
/// the predicate = Subject, after = Object), per the S/O nominal simplification
/// in `E-SURFACE-FORM-COLLAPSE-1`.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    Noun,
    Predicate,
    Skip,
}

/// The 6-word predicate vocabulary of the synthetic corpus, each mapped to
/// its lemma. `is`/`is_a` is the taxonomy copula; the rest are properties.
fn predicate_lemma(tok: &str) -> Option<&'static str> {
    match tok {
        "is" => Some("is_a"),
        "runs" | "run" => Some("run"),
        "needs" | "need" => Some("need"),
        "eats" | "eat" => Some("eat"),
        _ => None,
    }
}

fn role_of(tok: &str) -> Role {
    match tok {
        "the" | "a" | "an" => Role::Skip,
        _ if predicate_lemma(tok).is_some() => Role::Predicate,
        _ => Role::Noun, // resolved to Subject/Object by position at fill time
    }
}

/// Load `lemma -> rank` from the committed COCA frequency table (the
/// gridlake address each SPO node lands on).
fn load_ranks(csv_path: &str) -> HashMap<String, u32> {
    let text = std::fs::read_to_string(csv_path).unwrap_or_default();
    let mut m = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        if let Ok(rank) = f[0].trim().parse::<u32>() {
            m.entry(f[1].to_ascii_lowercase()).or_insert(rank);
        }
    }
    m
}

fn main() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let ranks = load_ranks(csv);

    // A small SYNTHETIC corpus of COCA lemmas. The chain dog -> pet -> animal
    // is deliberately never stated whole; the graph must DEDUCE (dog is_a
    // animal) and (dog need food).
    let corpus: [&[&str]; 6] = [
        &["the", "dog", "is", "a", "pet"],
        &["a", "pet", "is", "an", "animal"],
        &["the", "cat", "is", "an", "animal"], // direct, for contrast
        &["a", "pet", "needs", "food"],
        &["the", "dog", "runs"],
        &["the", "cat", "eats", "food"],
    ];
    const WINDOW: usize = 5;

    // Stage 1+2: COLLAPSE + +-5 FILL. Facts keyed (subject, predicate) -> obj.
    // `is_a` facts and property facts share the store; obj = "" for
    // predicate-only facts (the dog runs).
    let mut direct: Vec<(String, String, String, Truth)> = Vec::new();
    for sent in corpus.iter() {
        // collapse tokens to (lemma, role), keeping position
        let toks: Vec<(&str, Role)> = sent.iter().map(|&t| (t, role_of(t))).collect();
        for (pi, &(ptok, prole)) in toks.iter().enumerate() {
            if prole != Role::Predicate {
                continue;
            }
            let pred = predicate_lemma(ptok).unwrap();
            // Subject = nearest noun within WINDOW to the left
            let subj = (0..pi)
                .rev()
                .filter(|&k| pi - k <= WINDOW)
                .find(|&k| toks[k].1 == Role::Noun)
                .map(|k| toks[k].0);
            // Object = nearest noun within WINDOW to the right
            let obj = ((pi + 1)..toks.len())
                .filter(|&k| k - pi <= WINDOW)
                .find(|&k| toks[k].1 == Role::Noun)
                .map(|k| toks[k].0);
            if let Some(s) = subj {
                let o = obj.unwrap_or("").to_string();
                direct.push((s.to_string(), pred.to_string(), o, Truth { f: 1.0, c: 0.9 }));
            }
        }
    }

    // Stage 3: MARKOV/NARS FILL -- deduce facts no window stated.
    // (a) transitive is_a: s is_a m, m is_a o  |-  s is_a o
    // (b) property inheritance: s is_a m, m PRED o  |-  s PRED o  (PRED != is_a)
    let is_a: Vec<(String, String, Truth)> = direct
        .iter()
        .filter(|(_, p, o, _)| p == "is_a" && !o.is_empty())
        .map(|(s, _, o, t)| (s.clone(), o.clone(), *t))
        .collect();
    let stated: std::collections::HashSet<(String, String, String)> = direct
        .iter()
        .map(|(s, p, o, _)| (s.clone(), p.clone(), o.clone()))
        .collect();

    let mut deduced: Vec<(String, String, String, Truth)> = Vec::new();
    // (a) transitive is_a (single hop of closure; the chain here is length 2)
    for (s, m, t1) in is_a.iter() {
        for (m2, o, t2) in is_a.iter() {
            if m == m2 && s != o {
                let key = (s.clone(), "is_a".to_string(), o.clone());
                if !stated.contains(&key) && !deduced.iter().any(|(a, _, c, _)| a == s && c == o) {
                    deduced.push((s.clone(), "is_a".to_string(), o.clone(), t1.deduce(*t2)));
                }
            }
        }
    }
    // (b) property inheritance down the is_a basin
    for (s, m, t1) in is_a.iter() {
        for (bs, bp, bo, t2) in direct.iter() {
            if bp != "is_a" && bs == m {
                let key = (s.clone(), bp.clone(), bo.clone());
                if !stated.contains(&key)
                    && !deduced
                        .iter()
                        .any(|(a, p, c, _)| a == s && p == bp && c == bo)
                {
                    deduced.push((s.clone(), bp.clone(), bo.clone(), t1.deduce(*t2)));
                }
            }
        }
    }

    // Store: the AriGraph-shaped knowledge graph (adjacency + truth), nodes
    // grounded on their COCA lemRank (gridlake address).
    let rank_of = |w: &str| {
        ranks
            .get(w)
            .map(|r| r.to_string())
            .unwrap_or_else(|| "?".into())
    };

    println!("SPO -> +-5 fill -> Markov/NARS fill -> knowledge graph");
    println!(
        "  synthetic corpus: {} sentences, window +-{WINDOW}",
        corpus.len()
    );
    println!();
    println!("STAGE 2 -- DIRECT facts landed from the +-5 CausalEdge window:");
    for (s, p, o, t) in &direct {
        let obj = if o.is_empty() {
            "(intrans)"
        } else {
            o.as_str()
        };
        println!(
            "  ({s} r{}) --{p}--> ({obj}{})  <f={:.2} c={:.2}>",
            rank_of(s),
            if o.is_empty() {
                String::new()
            } else {
                format!(" r{}", rank_of(o))
            },
            t.f,
            t.c
        );
    }
    println!();
    println!("STAGE 3 -- DEDUCED facts (Markov/NARS; NO window stated these):");
    for (s, p, o, t) in &deduced {
        println!(
            "  ({s}) --{p}--> ({o})  <f={:.2} c={:.2}>  [reasoned]",
            t.f, t.c
        );
    }
    println!();

    // Query the graph: "what is a dog?" -> direct + deduced, deduced weaker.
    let query = "dog";
    println!("QUERY  what is '{query}'?  (is_a closure, direct first)");
    let mut answers: Vec<(&String, Truth, bool)> = Vec::new();
    for (s, p, o, t) in &direct {
        if s == query && p == "is_a" && !o.is_empty() {
            answers.push((o, *t, false));
        }
    }
    for (s, p, o, t) in &deduced {
        if s == query && p == "is_a" {
            answers.push((o, *t, true));
        }
    }
    answers.sort_by(|a, b| b.1.c.partial_cmp(&a.1.c).unwrap());
    for (o, t, reasoned) in &answers {
        println!(
            "  {query} is_a {o}  <c={:.2}>{}",
            t.c,
            if *reasoned {
                "  [reasoned, not read]"
            } else {
                "  [direct]"
            }
        );
    }
    println!();

    // KILL gates (regression guards).
    let mut fail = Vec::new();
    if direct.is_empty() {
        fail.push("no DIRECT facts landed from the +-5 window".to_string());
    }
    let deduced_is_a_animal = deduced
        .iter()
        .any(|(s, p, o, _)| s == "dog" && p == "is_a" && o == "animal");
    if !deduced_is_a_animal {
        fail.push(
            "(dog is_a animal) was NOT deduced -- Markov/NARS gap-fill did not fire".to_string(),
        );
    }
    let inherited = deduced
        .iter()
        .any(|(s, p, o, _)| s == "dog" && p == "need" && o == "food");
    if !inherited {
        fail.push("(dog need food) was NOT inherited down the is_a basin".to_string());
    }
    // deduced strictly weaker than any direct fact (NARS monotonicity)
    let min_direct_c = direct.iter().map(|(_, _, _, t)| t.c).fold(1.0f64, f64::min);
    let max_deduced_c = deduced
        .iter()
        .map(|(_, _, _, t)| t.c)
        .fold(0.0f64, f64::max);
    if !deduced.is_empty() && max_deduced_c >= min_direct_c {
        fail.push(format!(
            "deduced confidence {max_deduced_c:.2} not < direct {min_direct_c:.2} (NARS monotonicity broken)"
        ));
    }
    if fail.is_empty() {
        println!("KILL GATES: all pass -- the +-5 window lands facts, NARS reasoning fills the");
        println!("basins no window stated, and the graph answers with direct + reasoned truth.");
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
    println!();
    println!("DEFERRED (the affect layer): qualia -8/+8 sign = ironic/sarcastic inversion of");
    println!("the SPO literal valence, stored as a COMMITTED contradiction (triangle_bridge::");
    println!("qualia_distance is the hook). Gated on a labeled ironic corpus -- CONJECTURE, not");
    println!("fabricated here. See the endgame north-star on the board.");
}
