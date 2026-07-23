//! `insight_archetype_read` — D-SCI-1, the archetype side: a verb-mediated
//! relation extractor that **types every edge through the `verb_table`
//! archetypes**, not a flat connective list. Where `insight_relation_read`
//! (#841) proved the sparse-edge *shape* the centre-finder needs — a verb
//! makes an edge, any verb — this proves the edge's *reading*: each verb
//! resolves to its `(VerbFamily, Tense)` cell and the dominant `TekamoloSlot`
//! that cell expects, via `lance_graph_contract::grammar::verb_lexicon`.
//!
//! ## Why (the operator's requirement)
//!
//! "The relation extractor MUST consume `verb_table`, not hand-roll
//! connectives." The 144-cell table (12 families × 12 tenses) is the archetype
//! surface; `verb_lexicon::read_verb` is its consumer: `verb string → family →
//! base_prior·tense_modifier cell → argmax → TekamoloSlot`. So `caused` types
//! its edge Kausal, `rests` / `grounds` type theirs Lokal, `becomes` types its
//! Temporal — the adverbial role is READ FROM THE CELL, deterministically, no
//! model. The verb id is still carried as the articulation label; the family +
//! slot are the typed reading a downstream planner/reader can dispatch on.
//!
//! ## The falsifier (self-testing, runs in CI with no args)
//!
//! Two disjoint mini-corpora, extracted identically:
//!   * a CAUSAL text (`pressure caused failure`, `heat produced pressure`, …)
//!     → the plurality TEKAMOLO slot across its typed edges is **Kausal**;
//!   * a GROUNDING text (`theory rests on axioms`, `logic grounds proof`, …)
//!     → the plurality slot is **Lokal**.
//!
//! The asserted claim is that the slot typing is *discriminative* — the same
//! extractor over different relation semantics yields different dominant slots,
//! which is only possible if the edges genuinely consume the archetype cells.
//! A decorative/constant typing would fail this.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example insight_archetype_read -- FILE [FILE ...]
//! With no args it runs the inline falsifier corpora and asserts the separation.

use std::collections::HashMap;

use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::tekamolo::TekamoloSlot;
use lance_graph_contract::grammar::verb_lexicon::{is_copula, read_verb};
use lance_graph_contract::grammar::verb_table::VerbFamily;
use lance_graph_planner::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};

/// Minimal function-word stoplist — the point here is the typed edges, not
/// vocabulary tuning.
const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with", "as",
    "its", "his", "her", "their", "this", "that", "these", "those", "it", "then", "than", "into",
    "from", "only", "more", "some", "any", "all", "each", "upon", "onto",
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

/// One extracted, archetype-typed relation: subject → object across a verb,
/// with the verb's family + TEKAMOLO slot READ from the verb_table cell.
struct TypedEdge {
    s: u16,
    p: u16,
    family: VerbFamily,
    slot: TekamoloSlot,
    verb: String,
}

fn is_word(w: &str) -> bool {
    w.len() > 2 && !STOP.contains(&w)
}

/// Extract verb-mediated typed edges. Per sentence: track the last content noun
/// as the pending subject; on a verb (classified through the lexicon), remember
/// it; on the next content noun, emit `subject → object` typed by the verb's
/// archetype, then chain (object seeds the next subject). Copulas (`is`/`are`)
/// are inheritance links with no family/slot — recorded as `Inh` but not typed
/// (they are not verb-family predicates). Non-verb, non-noun tokens are skipped.
fn extract(text: &str, intern: &mut Interner) -> Vec<TypedEdge> {
    let mut edges = Vec::new();
    for sentence in text.split(['.', ';', '?', '!', '\n']) {
        let toks: Vec<String> = sentence
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        let mut subject: Option<u16> = None;
        let mut pending: Option<(VerbFamily, Tense, TekamoloSlot, String)> = None;
        let mut pending_copula = false;

        for tok in &toks {
            if let Some((fam, tense, slot)) = read_verb(tok) {
                // A typed verb — arm it for the next noun (overrides any pending
                // copula; "last verb wins", the FSM's recency tie-break).
                pending = Some((fam, tense, slot, tok.clone()));
                pending_copula = false;
            } else if is_copula(tok) {
                pending_copula = true;
                pending = None;
            } else if is_word(tok) {
                let n = intern.id(tok);
                if let Some((fam, tense, slot, verb)) = pending.take() {
                    if let Some(s) = subject {
                        if s != n {
                            let _ = tense; // tense folded into slot already
                            edges.push(TypedEdge {
                                s,
                                p: n,
                                family: fam,
                                slot,
                                verb,
                            });
                        }
                    }
                    subject = Some(n); // serial-verb chain
                } else if pending_copula {
                    pending_copula = false;
                    // Copula link recorded via extract's caller as Inh; here we
                    // only chain the subject (no typed family).
                    subject = Some(n);
                } else {
                    subject = Some(n); // a noun with no verb yet — the subject
                }
            }
            // else: a stopword / unknown short token — skipped (keeps it sparse).
        }
    }
    edges
}

/// Build the composable arena skeleton (`Inh`, so `close_transitive` chains it)
/// from the typed edges, discarding the typing — the typing is the *reading*,
/// the `Inh` skeleton is what the shipped centre-finder ablates.
fn arena_of(edges: &[TypedEdge]) -> BeliefArena {
    let mut arena = BeliefArena::new();
    for (i, e) in edges.iter().enumerate() {
        arena.observe(
            CStmt {
                s: e.s,
                cop: Copula::Inh,
                p: e.p,
            },
            TruthValue::new(0.9, 0.9),
            Stamp::source(i as u32),
        );
    }
    arena.close_transitive(256);
    arena
}

/// Plurality TEKAMOLO slot across a set of typed edges (deterministic; ties
/// resolve to the slot with the lower enum discriminant via the fold order).
fn plurality_slot(edges: &[TypedEdge]) -> Option<TekamoloSlot> {
    let order = [
        TekamoloSlot::Temporal,
        TekamoloSlot::Kausal,
        TekamoloSlot::Modal,
        TekamoloSlot::Lokal,
        TekamoloSlot::Instrument,
    ];
    let mut counts: HashMap<TekamoloSlot, usize> = HashMap::new();
    for e in edges {
        *counts.entry(e.slot).or_insert(0) += 1;
    }
    order
        .into_iter()
        .max_by_key(|s| counts.get(s).copied().unwrap_or(0))
        .filter(|s| counts.get(s).copied().unwrap_or(0) > 0)
}

fn report(label: &str, text: &str) -> (Vec<TypedEdge>, Interner) {
    let mut intern = Interner::new();
    let edges = extract(text, &mut intern);
    let arena = arena_of(&edges);

    println!("\n════════ {label} ════════");
    println!(
        "  {} typed edges, {} beliefs after closure",
        edges.len(),
        arena.entries().len()
    );
    let mut slot_counts: HashMap<TekamoloSlot, usize> = HashMap::new();
    for e in &edges {
        *slot_counts.entry(e.slot).or_insert(0) += 1;
    }
    println!("  — TYPED RELATIONS (subject —verb[family/slot]→ object) —");
    for e in edges.iter().take(10) {
        println!(
            "      {:>10} —{:<9}[{:?}/{:?}]→ {}",
            intern.name(e.s),
            e.verb,
            e.family,
            e.slot,
            intern.name(e.p)
        );
    }
    println!(
        "  — TEKAMOLO SLOT DISTRIBUTION — plurality={:?}",
        plurality_slot(&edges)
    );
    (edges, intern)
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

    // ── Falsifier: the archetype typing must be DISCRIMINATIVE ──
    // Causal relations classify Kausal; grounding relations classify Lokal.
    // Object words chosen to NOT collide with the verb lexicon (a POS-free
    // lexicon treats a noun/verb-ambiguous word like "collapse" as a verb).
    let causal = "pressure caused failure. failure caused damage. \
         heat produced pressure. stress generated failure. \
         friction induced heat.";
    let grounding = "theory rests on axioms. proof rests on theory. \
         logic grounds proof. reason anchors logic. \
         evidence roots reason.";

    let (causal_edges, _) = report("CAUSAL corpus", causal);
    let (grounding_edges, _) = report("GROUNDING corpus", grounding);

    let causal_slot = plurality_slot(&causal_edges);
    let grounding_slot = plurality_slot(&grounding_edges);

    println!("\n════════ FALSIFIER ════════");
    println!("  causal plurality slot   = {causal_slot:?}  (expect Kausal)");
    println!("  grounding plurality slot = {grounding_slot:?}  (expect Lokal)");

    assert!(
        !causal_edges.is_empty() && !grounding_edges.is_empty(),
        "both corpora must extract typed edges"
    );
    assert_eq!(
        causal_slot,
        Some(TekamoloSlot::Kausal),
        "causal relations must type Kausal (verb_table Causes/Prevents → Kausal)"
    );
    assert_eq!(
        grounding_slot,
        Some(TekamoloSlot::Lokal),
        "grounding relations must type Lokal (verb_table Grounds → Lokal)"
    );
    assert_ne!(
        causal_slot, grounding_slot,
        "the typing must be discriminative — same extractor, different dominant slot"
    );
    println!("  ✓ archetype typing is discriminative — the extractor consumes verb_table");
}
