//! `insight_read` — D-SCI-1 first cut: run the D-SCI-INSIGHT readers over a REAL
//! plain-text document, no LLM. Where `insight_overlap_smoke` (R&J) and
//! `gestalt_texture_smoke` (Tagore/Rumi) fed hand-compressed concept-lines, this
//! reads an actual text file and extracts its own concept-KG deterministically:
//!
//!   1. tokenize → content words (len > 2, not a stopword);
//!   2. FREQUENCY-SALIENCE term extraction — keep the top-`k` most frequent
//!      content words as the concept vocabulary (this is the "turn the colorblind
//!      finding positive" step: salience picks the terms, bounding the arena so
//!      `close_transitive` terminates);
//!   3. ±`window` co-occurrence — an `Inh` edge between consecutive salient
//!      concepts within a small positional window (the text's own adjacency);
//!   4. the shipped readers: `rank_epiphany_attractors` (THEME), `rank_basins`
//!      (GESTALT TEXTURE — staunen/wisdom + coherence-vs-evidence basins),
//!      `extract_main_insights` (the ranked MAIN INSIGHTS with explained reasons).
//!
//! Deterministic and bounded: same file → same insights, no network, no model.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example insight_read -- FILE [FILE ...]
//! With no args it runs a tiny inline corpus (so it is self-testing in CI).

use std::collections::HashMap;

use lance_graph_planner::nars::{
    extract_main_insights, rank_basins, rank_epiphany_attractors, staunen, wisdom, BeliefArena,
    CStmt, Copula, InsightConfig, InsightKind, InsightReason, ResonanceConfig, Snapshot, Stamp,
    TruthValue,
};

/// A modest English stoplist — enough to keep real prose from drowning the
/// salient concepts in function words. Not exhaustive; deterministic.
const STOP: &[&str] = &[
    "the",
    "and",
    "was",
    "for",
    "with",
    "his",
    "her",
    "she",
    "you",
    "him",
    "had",
    "have",
    "has",
    "are",
    "were",
    "that",
    "this",
    "but",
    "not",
    "all",
    "who",
    "they",
    "them",
    "their",
    "our",
    "your",
    "its",
    "from",
    "then",
    "than",
    "into",
    "out",
    "off",
    "over",
    "under",
    "again",
    "once",
    "here",
    "there",
    "when",
    "where",
    "what",
    "which",
    "how",
    "why",
    "any",
    "each",
    "few",
    "more",
    "most",
    "some",
    "such",
    "own",
    "same",
    "too",
    "very",
    "can",
    "will",
    "just",
    "would",
    "could",
    "should",
    "now",
    "one",
    "two",
    "upon",
    "unto",
    "shall",
    "may",
    "might",
    "must",
    "did",
    "does",
    "done",
    "been",
    "being",
    "himself",
    "herself",
    "myself",
    "itself",
    "yourself",
    "about",
    "before",
    "after",
    "above",
    "below",
    "between",
    "through",
    "during",
    "without",
    "within",
    "along",
    "around",
    "down",
    "yet",
    "nor",
    "because",
    "while",
    "though",
    "although",
    "however",
    "thus",
    "hence",
    "therefore",
    "let",
    "like",
    "still",
    "even",
    "ever",
    "never",
    "always",
    "said",
    "say",
    "says",
    "came",
    "come",
    "went",
    "goes",
    "get",
    "got",
    "make",
    "made",
    "man",
    "men",
    "old",
    "new",
    "way",
    "day",
    "see",
    "saw",
    "look",
    "looked",
    "know",
    "knew",
    // High-frequency low-content words that leak into salience on real prose
    // (D-SCI-1 finding on Tagore/Siddhartha): keep them out of the vocabulary.
    "also",
    "much",
    "many",
    "every",
    "everything",
    "something",
    "nothing",
    "anything",
    "become",
    "became",
    "cannot",
    "could",
    "would",
    "keep",
    "kept",
    "call",
    "called",
    "end",
    "put",
    "take",
    "took",
    "give",
    "gave",
    "find",
    "found",
    "feel",
    "felt",
    "want",
    "wanted",
    "seem",
    "seemed",
    "thing",
    "things",
    "part",
    "long",
    "great",
    "good",
    "little",
    "another",
    "other",
    "others",
    "myself",
    "himself",
    "away",
    "back",
    "off",
    "onto",
];

struct Interner {
    map: HashMap<String, u16>,
    names: Vec<String>,
}
impl Interner {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }
    fn id(&mut self, w: &str) -> u16 {
        if let Some(&i) = self.map.get(w) {
            return i;
        }
        let i = self.names.len() as u16;
        self.map.insert(w.to_string(), i);
        self.names.push(w.to_string());
        i
    }
    fn name(&self, id: u16) -> &str {
        &self.names[id as usize]
    }
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

/// MathML/LaTeX-mangle fragments that PDF text extraction leaves behind on
/// math-heavy papers (D-SCI-1 real-paper finding on the JAR OWL-reasoning paper:
/// `msub`, equation refs, figure numbers dominated the salient vocabulary). Kept
/// out of the concept vocabulary so the readers see terms, not notation.
const MATH_FRAG: &[&str] = &[
    "msub",
    "msup",
    "mrow",
    "mfrac",
    "mtable",
    "mtd",
    "mtr",
    "mstyle",
    "mtext",
    "mspace",
    "mpadded",
    "mphantom",
    "mover",
    "munder",
    "mroot",
    "mfenced",
    "msqrt",
    "mmultiscripts",
];

/// Deterministic content-word tokenisation. Drops all-numeric tokens (equation
/// refs / figure numbers) and MathML fragments — PDF-extraction noise that would
/// otherwise dominate frequency salience on academic papers.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| {
            w.len() > 2
                && !STOP.contains(&w.as_str())
                && !MATH_FRAG.contains(&w.as_str())
                && !w.chars().all(|c| c.is_ascii_digit())
        })
        .collect()
}

/// Build + close a concept-KG from a salient (pos, concept) stream: ±`window`
/// co-occurrence `Inh` edges. `exclude` drops one concept entirely (its edges are
/// never observed) — the ablation used by the centre finder.
fn close_from(salient: &[(usize, u16)], window: usize, exclude: Option<u16>) -> BeliefArena {
    let mut arena = BeliefArena::new();
    let mut src = 0u32;
    for i in 0..salient.len() {
        let (pos_i, ci) = salient[i];
        if exclude == Some(ci) {
            continue;
        }
        for &(pos_j, cj) in salient.iter().skip(i + 1) {
            if pos_j - pos_i > window {
                break;
            }
            if ci != cj && exclude != Some(cj) {
                arena.observe(inh(ci, cj), TruthValue::new(0.9, 0.9), Stamp::source(src));
                src = src.wrapping_add(1);
            }
        }
    }
    arena.close_transitive(512);
    arena
}

/// Read one document into a bounded concept-KG: top-`k` frequency-salient terms
/// as the vocabulary, ±`window` co-occurrence between salient terms as `Inh`
/// edges. Returns (arena, interner, ordered top terms, salient stream).
#[allow(clippy::type_complexity)]
fn read_kg(
    text: &str,
    k: usize,
    window: usize,
) -> (
    BeliefArena,
    Interner,
    Vec<(String, usize)>,
    Vec<(usize, u16)>,
) {
    let toks = tokens(text);

    // Frequency-salience: pick the top-k content words as the concept vocabulary.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in &toks {
        *freq.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut ranked: Vec<(String, usize)> = freq.iter().map(|(w, &c)| (w.to_string(), c)).collect();
    // Deterministic: by count desc, then word asc.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    let vocab: HashMap<&str, ()> = ranked.iter().map(|(w, _)| (w.as_str(), ())).collect();

    // ±window co-occurrence between salient terms, in text order.
    let mut intern = Interner::new();
    for (w, _) in &ranked {
        intern.id(w); // stable ids for the vocabulary, salience order
    }
    let salient: Vec<(usize, u16)> = toks
        .iter()
        .enumerate()
        .filter(|(_, t)| vocab.contains_key(t.as_str()))
        .map(|(pos, t)| (pos, intern.id(t)))
        .collect();

    let arena = close_from(&salient, window, None);
    (arena, intern, ranked, salient)
}

fn report(label: &str, text: &str) {
    const K: usize = 48;
    const WINDOW: usize = 4;
    let (arena, intern, ranked, salient) = read_kg(text, K, WINDOW);
    let snap = Snapshot::of(&arena, 0.0);

    println!("\n════════ {label} ════════");
    println!(
        "  {} salient terms, {} beliefs, gestalt: staunen={:.3} wisdom={:.3} coherence={:.3}",
        ranked.len(),
        arena.entries().len(),
        staunen(&snap),
        wisdom(&snap),
        snap.coherence,
    );

    println!("  — THEME (epiphany attractors, top 6 by density) —");
    for e in rank_epiphany_attractors(&arena, 2).iter().take(6) {
        println!(
            "      {:>14}  rate={:.3}  ({}/{})",
            intern.name(e.subject),
            e.rate,
            e.epiphanies,
            e.attempts
        );
    }

    println!("  — GESTALT TEXTURE (top resonance basins) —");
    for b in rank_basins(&arena, &ResonanceConfig::default())
        .iter()
        .take(6)
    {
        println!(
            "      {:>14}  resonance={:.3}  wisdom={:.3}  evidence={:.3}  {:?}",
            intern.name(b.concept),
            b.resonance,
            b.wisdom,
            b.evidence,
            b.kind
        );
    }

    println!("  — MAIN INSIGHTS (ranked, with reason) —");
    for mi in extract_main_insights(&arena, &InsightConfig::default())
        .iter()
        .take(8)
    {
        let what = match mi.kind {
            InsightKind::CoreTheme => format!("THEME  {}", intern.name(mi.focus.s)),
            InsightKind::Bridge => format!("BRIDGE {}", intern.name(mi.focus.s)),
            InsightKind::Conclusion => format!(
                "CONCLUDE {} → {}",
                intern.name(mi.focus.s),
                intern.name(mi.focus.p)
            ),
        };
        let why = match mi.reason {
            InsightReason::DenseBasin {
                epiphanies,
                attempts,
            } => format!("dense basin {epiphanies}/{attempts}"),
            InsightReason::StrongDerivation { expectation } => {
                format!(
                    "strong derivation exp={expectation:.3} (ladder {} deep)",
                    mi.rung
                )
            }
            InsightReason::MiddleTerm { bridges } => format!("middle term of {bridges} claims"),
        };
        println!("      [{:.3}] {what:<28} — {why}", mi.strength);
    }

    // — CENTRE (ablation) — "the smallest set of relations whose removal makes
    //   the remaining statements mean something else" (the House text's own
    //   definition of the centre). For each salient concept, remove it, re-close,
    //   and measure how many conclusions OVER THE SURVIVING TERMS collapse. This
    //   is the inverse of reach_out_integrate (which measures what ADDING a bridge
    //   composes).
    //
    //   Crucially we compare the SURVIVING-term derivation SETS, not raw counts
    //   (Codex #836 P2): a conclusion that merely mentions the removed term `c`
    //   (e.g. losing `A→C` when ablating endpoint `A`) is not a change to the
    //   remaining graph — only its trivial disappearance. Counting those biases
    //   the centre toward frequent endpoints/hubs. So `lost(c)` = derived
    //   statements NOT touching `c` that existed before but vanish after removal
    //   (e.g. losing `A→C` when ablating the MIDDLE term `B` — a real collapse of
    //   the surviving graph). Normalized by the surviving base derivations.
    let base_survivor = |c: u16| -> Vec<CStmt> {
        arena
            .entries()
            .iter()
            .filter(|b| b.rung >= 1 && b.stmt.s != c && b.stmt.p != c)
            .map(|b| b.stmt)
            .collect()
    };
    let mut impact: Vec<(u16, f32, usize)> = Vec::new();
    for (i, _) in ranked.iter().enumerate() {
        let c = i as u16; // interner ids are salience order 0..ranked.len()
        let ablated = close_from(&salient, WINDOW, Some(c));
        let ablated_set: std::collections::HashSet<CStmt> = ablated
            .entries()
            .iter()
            .filter(|b| b.rung >= 1)
            .map(|b| b.stmt)
            .collect();
        let surviving = base_survivor(c); // base derivations not touching c
        let base_n = surviving.len().max(1);
        let lost = surviving
            .iter()
            .filter(|s| !ablated_set.contains(s))
            .count();
        impact.push((c, lost as f32 / base_n as f32, lost));
    }
    impact.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!("  — CENTRE (ablation: fraction of SURVIVING-term conclusions lost) —");
    for (c, frac, lost) in impact.iter().take(6) {
        println!(
            "      {:>14}  collapse={:.3}  ({} conclusions)",
            intern.name(*c),
            frac,
            lost
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        // Inline fallback so the example is self-testing without a corpus.
        let sample = "the mind seeks truth through patient thought; the seeker doubts \
             the mind, the mind doubts the seeker, and truth waits in the doubt. \
             patient thought finds truth; the restless mind finds only more thought.";
        report("inline sample", sample);
        println!(
            "\n(usage: cargo run -p lance-graph-planner --example insight_read -- FILE [FILE ...])"
        );
        return;
    }
    for path in &args {
        match std::fs::read_to_string(path) {
            Ok(text) => report(path, &text),
            Err(e) => eprintln!("skip {path}: {e}"),
        }
    }
}
