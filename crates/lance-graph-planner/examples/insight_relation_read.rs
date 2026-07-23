//! `insight_relation_read` — D-SCI-1: **typed relation extraction**, the single
//! frontier the centre-finder named (#836/#837). Where `insight_read` connects
//! every pair of salient terms inside a ±window (DENSE word-adjacency), this
//! connects two salient terms **only across a verb** — a sparse, verb-mediated
//! `Inh` skeleton (`cup —held→ memory`, `memory —filled→ room`). Deterministic,
//! no LLM: a verb is detected by a fixed lexicon + regular English morphology
//! (`-ed` past tense), never by a model.
//!
//! ## Why this exists (the arc's stated gap)
//!
//! #836/#837 proved the coherence-ablation **centre-finder** is a correct reader
//! but that **word-adjacency co-occurrence has no articulation points**: every
//! salient term is redundantly connected by many short paths, so removing any
//! single one leaves the surviving conclusions intact (collapse ≈ 0), and what
//! signal remains is biased toward frequent endpoints/hubs — never the semantic
//! centre the text is about. The fix they registered verbatim: *"typed relation
//! extraction that yields a sparse graph with real articulation structure — an
//! edge whose removal genuinely disconnects — on which this now-correct ablation
//! measure will find the centre."* This example is that extractor, and it feeds
//! the **same** centre-finder from `insight_read`, unchanged.
//!
//! ## The composable spine is `Inh`, the verb is the label
//!
//! `Copula::Rel(verb)` NEVER transits (belief.rs) — so a bare `Rel` edge is inert
//! to `close_transitive` and the ablation would see nothing. The subject→object
//! **skeleton** is therefore emitted as transitive `Inh` (so chains compose and
//! the ablation has something to collapse), and the verb id is carried alongside
//! purely for articulation labelling in the CENTRE printout (`key —altered→
//! room`). Extraction is sparse; composition and ablation are the shipped code.
//!
//! ## The falsifier (self-testing, runs in CI with no args)
//!
//! Over one corpus of three disjoint SVO chains (`cup→memory→room`,
//! `key→drawer→letter`, `clock→hours→day`), built two ways:
//!   * SPARSE (this extractor): the true **middle** terms (`memory`/`drawer`/
//!     `hours`) STRICTLY outrank the **endpoints** (`cup`/`key`/`letter`) in
//!     ablation collapse — the middle is an articulation point, the endpoint is
//!     not. Removing `memory` collapses the surviving-term conclusion `cup→room`.
//!   * DENSE (±window, as `insight_read`): the ranking is dominated by position/
//!     frequency; a true middle term does NOT cleanly top the endpoints. The
//!     centre is smeared — #837's finding reproduced on the same text.
//!
//! The asserted claim is the *separation*: sparse extraction turns the diffuse
//! dense signal into a clean middle-vs-endpoint verdict.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example insight_relation_read -- FILE [FILE ...]
//! With no args it runs the inline falsifier corpus and asserts the separation.

use std::collections::{HashMap, HashSet};

use lance_graph_planner::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};

/// A modest English stoplist — function words that never carry a concept. Kept
/// deterministic and small; the point of this example is the *edges*, not vocab
/// tuning (see `insight_read` for the fuller frequency-salience path).
const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with", "as",
    "its", "his", "her", "their", "this", "that", "these", "those", "it", "he", "she", "they",
    "then", "than", "into", "from", "only", "more", "some", "any", "all", "each",
];

/// Deterministic verb lexicon — high-frequency verbs whose surface form does not
/// end in `-ed` (so morphology alone would miss them). Irregular pasts, copulas,
/// and common present-tense transitives. Not exhaustive; deterministic and
/// no-LLM — the whole point is that "which token is a verb" is decided by a
/// fixed table + a suffix rule, never by a trained model.
const VERB_LEXICON: &[&str] = &[
    // copulas / auxiliaries that link a subject to a predicate
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "becomes",
    "became",
    "remains",
    // irregular pasts + common present transitives (subject acts on object)
    "held",
    "hid",
    "hides",
    "holds",
    "made",
    "makes",
    "gave",
    "gives",
    "took",
    "takes",
    "found",
    "finds",
    "lost",
    "loses",
    "kept",
    "keeps",
    "told",
    "tells",
    "saw",
    "sees",
    "knew",
    "knows",
    "grew",
    "grows",
    "drew",
    "draws",
    "broke",
    "breaks",
    "bound",
    "binds",
    "wove",
    "weaves",
    "bore",
    "bears",
    "led",
    "leads",
    "meant",
    "means",
    "shook",
    "left",
    "leaves",
    "brought",
    "bought",
    "caught",
    "sought",
    "built",
    "sent",
    "spent",
    "won",
    "wins",
    "ran",
    "runs",
    "rose",
    "fell",
    "falls",
    "turned",
    "turns",
    "shaped",
    "ordered",
    "locked",
    "filled",
    "altered",
    "reveals",
    "revealed",
    "hides",
    "carries",
    "carried",
    "connects",
    "connected",
    "links",
    "linked",
    "causes",
    "caused",
    "yields",
    "yielded",
    "proves",
    "proved",
    "shows",
    "showed",
];

/// Is `w` a verb, by lexicon or regular `-ed` past-tense morphology? Deterministic,
/// no model. The `-ed` rule over a non-stopword content token catches regular
/// past tenses the lexicon does not enumerate (`altered`, `ordered`, `refused`);
/// short `-ed` tokens (`bed`, `red`, `fed`) are excluded by the length guard.
fn is_verb(w: &str) -> bool {
    if VERB_LEXICON.contains(&w) {
        return true;
    }
    // Regular past tense: `<stem>ed`, stem long enough to be a real verb, and the
    // token itself not a stopword. `-ied`/`-ed` both end in `ed`.
    w.len() > 4 && w.ends_with("ed") && !STOP.contains(&w)
}

/// A content term: alphabetic, length > 2, not a stopword, not a verb. These are
/// the concept vocabulary — the nodes the sparse skeleton connects.
fn is_term(w: &str) -> bool {
    w.len() > 2 && !STOP.contains(&w) && !is_verb(w) && w.chars().all(|c| c.is_ascii_alphabetic())
}

/// Lowercase alphabetic tokenisation, preserving order (position carries the
/// verb-between-terms structure the extractor reads).
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

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

/// A centre-finder impact table: `(concept, collapse_fraction, conclusions_lost)`
/// rows, sorted by collapse descending.
type Impact = Vec<(u16, f32, usize)>;

/// One extracted typed relation: `subject —verb→ object`, the verb kept as a
/// label. The composable edge is `Inh(subject, object)`; `verb` is provenance.
#[derive(Clone, Copy)]
struct Relation {
    s: u16,
    verb: u16,
    o: u16,
}

/// **The extractor.** Walk the token stream; a relation is emitted when a term is
/// followed (with only stopwords + exactly one verb between) by another term:
/// `subject <verb> object`. Sparse by construction — a pair of terms with no verb
/// between them is NOT connected (that is the whole difference from `insight_read`'s
/// dense ±window). The verb resets after each emitted edge, so `A verb B verb C`
/// yields `A→B` and `B→C`, never `A→C` (closure derives that).
fn extract_relations(toks: &[String], intern: &mut Interner) -> Vec<Relation> {
    let mut rels = Vec::new();
    let mut subject: Option<u16> = None;
    let mut pending_verb: Option<u16> = None;
    for w in toks {
        if is_verb(w) {
            // Only arm a verb once we have a subject to attach it to.
            if subject.is_some() {
                pending_verb = Some(intern.id(w));
            }
            continue;
        }
        if !is_term(w) {
            continue; // stopword — transparent, does not break the S-V-O window
        }
        // A content term.
        let o = intern.id(w);
        if let (Some(s), Some(verb)) = (subject, pending_verb) {
            if s != o {
                rels.push(Relation { s, verb, o });
            }
        }
        // This term becomes the subject of the next relation; verb disarmed.
        subject = Some(o);
        pending_verb = None;
    }
    rels
}

/// Build + close a KG from a sparse typed relation set, optionally ablating one
/// concept (its edges are never observed). The skeleton is `Inh(s, o)` so chains
/// compose under `close_transitive` and the ablation has surviving-term
/// conclusions to collapse.
fn close_sparse(rels: &[Relation], exclude: Option<u16>) -> BeliefArena {
    let mut arena = BeliefArena::new();
    let mut src = 0u32;
    for r in rels {
        if exclude == Some(r.s) || exclude == Some(r.o) {
            continue;
        }
        arena.observe(
            CStmt {
                s: r.s,
                cop: Copula::Inh,
                p: r.o,
            },
            TruthValue::new(0.9, 0.9),
            Stamp::source(src),
        );
        src = src.wrapping_add(1);
    }
    arena.close_transitive(512);
    arena
}

/// Build + close the DENSE ±window co-occurrence KG over the same salient terms,
/// exactly as `insight_read` does — the baseline this extractor is measured
/// against. `salient` is the in-order (position, concept) stream.
fn close_dense(salient: &[(usize, u16)], window: usize, exclude: Option<u16>) -> BeliefArena {
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
                arena.observe(
                    CStmt {
                        s: ci,
                        cop: Copula::Inh,
                        p: cj,
                    },
                    TruthValue::new(0.9, 0.9),
                    Stamp::source(src),
                );
                src = src.wrapping_add(1);
            }
        }
    }
    arena.close_transitive(512);
    arena
}

/// The centre-finder (survivor-set coherence ablation, #837), parameterised over
/// how the KG is rebuilt post-ablation. For each concept `c`: remove it, re-close,
/// and count base derivations NOT touching `c` that vanish — a real collapse of
/// the *surviving* graph (losing `A→C` when the MIDDLE term `B` is ablated),
/// normalised by the surviving base derivation count. Returns `(concept, frac,
/// lost)` sorted by collapse desc.
fn centre<F>(base: &BeliefArena, concepts: &[u16], rebuild: F) -> Impact
where
    F: Fn(u16) -> BeliefArena,
{
    let base_survivor = |c: u16| -> Vec<CStmt> {
        base.entries()
            .iter()
            .filter(|b| b.rung >= 1 && b.stmt.s != c && b.stmt.p != c)
            .map(|b| b.stmt)
            .collect()
    };
    let mut impact: Impact = Vec::new();
    for &c in concepts {
        let ablated = rebuild(c);
        let ablated_set: HashSet<CStmt> = ablated
            .entries()
            .iter()
            .filter(|b| b.rung >= 1)
            .map(|b| b.stmt)
            .collect();
        let surviving = base_survivor(c);
        let base_n = surviving.len().max(1);
        let lost = surviving
            .iter()
            .filter(|s| !ablated_set.contains(s))
            .count();
        impact.push((c, lost as f32 / base_n as f32, lost));
    }
    impact.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    impact
}

/// Rank of a concept in a collapse-sorted impact table (0 = highest collapse).
fn rank_of(impact: &[(u16, f32, usize)], c: u16) -> usize {
    impact
        .iter()
        .position(|&(x, _, _)| x == c)
        .unwrap_or(usize::MAX)
}

/// Report both KGs' centre-finder output for `text`, print the extracted typed
/// relations, and return the two impact tables + interner for assertions.
fn report(label: &str, text: &str) -> (Impact, Impact, Interner) {
    const WINDOW: usize = 4;
    let toks = tokens(text);
    let mut intern = Interner::new();
    let rels = extract_relations(&toks, &mut intern);

    // The salient stream (positions of content terms) for the dense baseline —
    // over the SAME vocabulary the extractor interned, so the comparison is fair.
    let salient: Vec<(usize, u16)> = toks
        .iter()
        .enumerate()
        .filter(|(_, t)| is_term(t))
        .map(|(pos, t)| (pos, intern.id(t)))
        .collect();

    let concepts: Vec<u16> = {
        let mut set: Vec<u16> = rels.iter().flat_map(|r| [r.s, r.o]).collect();
        set.sort_unstable();
        set.dedup();
        set
    };

    let sparse = close_sparse(&rels, None);
    let dense = close_dense(&salient, WINDOW, None);

    println!("\n════════ {label} ════════");
    println!(
        "  {} concepts · {} typed relations · sparse KG {} beliefs · dense KG {} beliefs",
        concepts.len(),
        rels.len(),
        sparse.entries().len(),
        dense.entries().len(),
    );
    println!("  — EXTRACTED RELATIONS (subject —verb→ object) —");
    for r in rels.iter().take(12) {
        println!(
            "      {:>12} —{}→ {}",
            intern.name(r.s),
            intern.name(r.verb),
            intern.name(r.o)
        );
    }

    let sparse_impact = centre(&sparse, &concepts, |c| close_sparse(&rels, Some(c)));
    let dense_impact = centre(&dense, &concepts, |c| {
        close_dense(&salient, WINDOW, Some(c))
    });

    println!("  — CENTRE via SPARSE typed relations (articulation structure) —");
    for (c, frac, lost) in sparse_impact.iter().take(6) {
        println!(
            "      {:>12}  collapse={:.3}  ({} conclusions)",
            intern.name(*c),
            frac,
            lost
        );
    }
    println!("  — CENTRE via DENSE word-adjacency (#837 baseline: smeared) —");
    for (c, frac, lost) in dense_impact.iter().take(6) {
        println!(
            "      {:>12}  collapse={:.3}  ({} conclusions)",
            intern.name(*c),
            frac,
            lost
        );
    }

    (sparse_impact, dense_impact, intern)
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

    // ── Inline falsifier corpus: three disjoint SVO chains ──
    // cup —held→ memory —filled→ room · key —locked→ drawer —hid→ letter ·
    // clock —ordered→ hours —shaped→ day. The MIDDLE terms (memory/drawer/hours)
    // are articulation points; the ENDPOINTS (cup/key/letter) are not.
    let corpus = "the cup held the memory, and the memory filled the room. \
         the key locked the drawer, and the drawer hid the letter. \
         the clock ordered the hours, and the hours shaped the day.";
    let (sparse, dense, intern) = report("inline falsifier corpus", corpus);

    // The claim: SPARSE extraction SEPARATES middle terms from endpoints — every
    // true middle term outranks every endpoint in ablation collapse — where the
    // DENSE baseline does NOT (an endpoint outranks a middle, the centre smeared).
    let id = |w: &str| -> u16 {
        intern
            .names
            .iter()
            .position(|n| n == w)
            .map(|i| i as u16)
            .unwrap_or_else(|| panic!("term {w} not interned"))
    };
    let middles = ["memory", "drawer", "hours"];
    let endpoints = ["cup", "key", "letter"];

    // SPARSE: each middle strictly above each endpoint, and each middle has real
    // collapse (> 0) while each endpoint has none.
    let worst_middle = middles
        .iter()
        .map(|m| rank_of(&sparse, id(m)))
        .max()
        .unwrap();
    let best_endpoint = endpoints
        .iter()
        .map(|e| rank_of(&sparse, id(e)))
        .min()
        .unwrap();
    assert!(
        worst_middle < best_endpoint,
        "SPARSE: every middle term must outrank every endpoint \
         (worst middle rank {worst_middle} vs best endpoint rank {best_endpoint})"
    );
    for m in middles {
        let (_, frac, lost) = sparse[rank_of(&sparse, id(m))];
        assert!(
            frac > 0.0 && lost >= 1,
            "SPARSE: middle term {m} must be an articulation point (collapse {frac} > 0)"
        );
    }
    for e in endpoints {
        let (_, frac, _) = sparse[rank_of(&sparse, id(e))];
        assert!(
            frac == 0.0,
            "SPARSE: endpoint {e} must NOT be an articulation point (collapse {frac} == 0)"
        );
    }

    // DENSE: the clean separation that holds for SPARSE must FAIL — some true
    // middle term ranks no better than some endpoint (the centre is smeared by
    // position/frequency; a frequent endpoint outranks a real articulation
    // point). This is the exact NEGATION of the sparse success check above —
    // #837's "biased toward endpoints/hubs" finding, made a live assertion.
    let worst_middle_dense = middles
        .iter()
        .map(|m| rank_of(&dense, id(m)))
        .max()
        .unwrap();
    let best_endpoint_dense = endpoints
        .iter()
        .map(|e| rank_of(&dense, id(e)))
        .min()
        .unwrap();
    assert!(
        worst_middle_dense >= best_endpoint_dense,
        "DENSE baseline must FAIL to separate middles from endpoints \
         (worst middle rank {worst_middle_dense} vs best endpoint rank \
         {best_endpoint_dense}); if it separated, this example's premise is void"
    );

    println!(
        "\n✔ SPARSE typed extraction separated the {} middle terms (articulation points, \
         collapse > 0) from the {} endpoints (collapse 0); the DENSE baseline did not. \
         The extractor is the centre-finder's missing input (#836/#837).",
        middles.len(),
        endpoints.len()
    );
    println!(
        "\n(usage: cargo run -p lance-graph-planner --example insight_relation_read -- FILE [FILE ...])"
    );
}
