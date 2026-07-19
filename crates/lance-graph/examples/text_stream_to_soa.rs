//! `text_stream_to_soa` — the S07 golden slice: a plain-text file → knowledge
//! graph + SoA, **no LLM**. It demonstrates the whole thesis on real text:
//!
//! ```text
//! text → COCA FSM (no LLM) → SPO stream → TripletGraph
//!      → ±5 Markov context → NARS reasoning → SpoFacet 6×(8:8) → 512-B node size
//! ```
//!
//! Run (default = the bundled public-domain Aesop sample):
//! ```sh
//! cargo run -p lance-graph --example text_stream_to_soa
//! cargo run -p lance-graph --example text_stream_to_soa -- /path/to/book.txt
//! ```
//!
//! Two-sixes note (do not conflate): the **CAM-PQ 6** is the 6 PQ *subspaces* of
//! ONE word's 96-D distributional vector (`deepnsm::codebook`); the **SpoFacet 6**
//! is the 6 *slots* of one triple (S·P·O + episodic-witness). The word→palette
//! `(basin:identity)` encoder is a documented gap — here we use the reversible
//! byte-split stand-in (`rank>>8, rank&0xFF`) until `cam_codes.bin` ships.

use std::path::{Path, PathBuf};

use deepnsm::parser;
use deepnsm::spo::NO_ROLE;
use deepnsm::vocabulary::Vocabulary;

use lance_graph::graph::arigraph::markov_soa::SpoRanks;
use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph_contract::awareness_facet::SpoFacet;
use lance_graph_contract::canonical_node::NODE_ROW_STRIDE;

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let input = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(manifest).join("examples/data/aesop_fables.txt"));
    let text =
        std::fs::read_to_string(&input).unwrap_or_else(|e| panic!("read {}: {e}", input.display()));

    // ── Stage 1: COCA vocab (no LLM) — the 4096-word rank table. ──
    let vocab_dir = Path::new(manifest).join("../deepnsm/word_frequency");
    let vocab = Vocabulary::load(&vocab_dir).expect("load COCA vocabulary");

    // ── Stream: per-sentence 6-state PoS FSM → SPO triples. ──
    let mut graph = TripletGraph::new();
    let mut ranks: Vec<SpoRanks> = Vec::new(); // the Markov side-stream (1:1 with triples)
    let mut sentences = 0usize;
    let mut tokens_total = 0usize;
    // Canonical-surface-fidelity detector: word → (canonical-matched, total). A
    // symbol whose canonical surface accounts for <15% of the tokens that resolved
    // to it is tokenizer-MANUFACTURED — the lemma table (`jean`→`jeans`) or the
    // suffix-strip fallback (`Boxer`→`box`) invented it. This is the working
    // codebook-resolution leg (E-CODEBOOK-OOV-SURFACE-FIDELITY-1): unlike the
    // genre-vector over-representation filter — which FALSE-POSITIVES on genuine
    // over-represented theme words (`animal`/`farm`/`mouse`) — this metric clears
    // them (canonical present) and flags the manufactured symbols. It is a SUPERSET
    // detector: it also flags benign lemma-only inflection (`tried`→`try`, base form
    // absent). Separating the proper-noun collapses (`Jean`→`jeans`) from benign
    // inflection needs the capitalization signal, which the tokenizer DESTROYS at
    // `split_words` (deepnsm `vocabulary.rs:344` lowercases every char) — so the
    // finer split needs case-preserving NER upstream, or the LLM tail. As a bonus
    // it doubles as a corpus-integrity check: it exposed a mislabeled corpus.
    let mut surface_fid: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    // Naming heuristic (escape hatch #1): mid-sentence Capitalized tokens that
    // don't resolve to a direct common word are NAMED ENTITIES (`Jean`, `Boxer`,
    // `Napoleon`) — kept out of the lossy lemma fallback so they no longer
    // collapse to `jeans`/`box`. `deepnsm::parser::named_entities` surfaces them;
    // here we histogram them (the "capitalized-within-sentence + histogram = a
    // name" signal). They carry identity by surface string, not COCA rank.
    let mut name_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    // Grammar-heuristic knob (literature: OIE stopword filtering — OPIEC 1904.12324,
    // surface-fact linking 2310.14909). `CONTENT_ONLY=1` drops triples whose subject
    // or object is a top-`FUNCTION_CUTOFF` COCA rank (function words / pronouns) — the
    // "content-word grammar heuristic". Testable hypothesis: the symbol layer becomes
    // content words and the surface noise (e.g. `jeans`, `it`) drops.
    let content_only = std::env::var("CONTENT_ONLY").is_ok();
    const FUNCTION_CUTOFF: u16 = 150;

    for (ts, sentence) in text.split(['.', '!', '?']).enumerate() {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        sentences += 1;
        let toks = vocab.tokenize(s); // word → COCA token
        tokens_total += toks.len();
        for tok in &toks {
            if let Some(r) = tok.rank {
                let w = vocab.word(r);
                let e = surface_fid.entry(w.to_string()).or_insert((0, 0));
                e.1 += 1;
                if tok.surface == w {
                    e.0 += 1; // canonical surface literally present (already lowercased)
                }
            }
        }
        for (_pos, name) in parser::named_entities(&toks) {
            *name_freq.entry(name).or_default() += 1;
        }
        let structure = parser::parse(&toks); // FSM → SPO
        for t in &structure.triples {
            // Stage 1→2 adapter: index-triple → String-triple via vocab.word().
            // NB: Triplet::new order is (subject, OBJECT, relation, ts).
            let subj = vocab.word(t.subject()).to_string();
            let pred = vocab.word(t.predicate()).to_string();
            let obj = if t.has_object() {
                vocab.word(t.object()).to_string()
            } else {
                String::new()
            };
            // `TripletGraph::add_triplets` silently drops the "free" relation (and
            // empty preds). Mirror that skip here so the graph, the Markov stream,
            // and the facet count stay consistent for exactly those sentences
            // (e.g. Aesop's "freed" → lemma "free").
            if pred == "free" || pred.is_empty() {
                continue;
            }
            // Content-word grammar heuristic: skip function-word S/O (top ranks).
            if content_only
                && (t.subject() < FUNCTION_CUTOFF
                    || (t.has_object() && t.object() < FUNCTION_CUTOFF))
            {
                continue;
            }
            graph.add_triplets(&[Triplet::new(&subj, &obj, &pred, ts as u64)]);
            ranks.push(SpoRanks {
                s: t.subject(),
                p: t.predicate(),
                o: t.object(),
            });
        }
    }
    let n_triples = ranks.len();

    // ── Stage 3: ±5 Markov context window over the SPO stream. ──
    // Production primitive: `markov_soa::SoaWavePrimer::project` (radius = 5).
    // Here we walk the window directly over the SpoRanks side-stream: a Markov
    // "link" = two nearby triples sharing a role (subject/object continuity) —
    // this is the structural temporal causality (braiding), not a learned weight.
    const RADIUS: usize = 5;
    let mut context_links = 0usize;
    for focal in 0..n_triples {
        let lo = focal.saturating_sub(RADIUS);
        let hi = (focal + RADIUS + 1).min(n_triples);
        let a = &ranks[focal];
        for other in lo..hi {
            if other == focal {
                continue;
            }
            let b = &ranks[other];
            // Subject continuity is always real; object continuity only counts
            // when the object is a real role, not the intransitive NO_ROLE (0xFFF)
            // sentinel (else two objectless clauses "link" on the sentinel).
            let obj_link =
                (a.o != NO_ROLE && (a.o == b.o || a.o == b.s)) || (b.o != NO_ROLE && a.s == b.o);
            if a.s == b.s || obj_link {
                context_links += 1;
            }
        }
    }

    // ── Coherence-length probe (H1, "Three Sentences Are All You Need",
    // 2106.01793): for every focal triple, at what *forward* distance d does a
    // role-continuity partner sit? The histogram's cumulative curve is the
    // text's coherence length — locally-coherent fables saturate within a
    // sentence or two; long-range narratives keep accruing links out to d=20.
    // This is precisely what makes the ±5 window a *projection* parameter
    // (read-time), not a structural one (E-MARKOV-TEMPORAL-STREAM-1): the
    // committed graph (deductions / contradictions / Q) is byte-identical at
    // every radius; only this curve moves. Testable claim: frac_within_5 (how
    // much a ±5 window captures) is high for fables, lower for novels.
    const SCAN_MAX: usize = 20;
    let mut link_hist = [0usize; SCAN_MAX + 1];
    for focal in 0..n_triples {
        let a = &ranks[focal];
        let hi = (focal + SCAN_MAX + 1).min(n_triples);
        for (other, b) in ranks.iter().enumerate().take(hi).skip(focal + 1) {
            let obj_link =
                (a.o != NO_ROLE && (a.o == b.o || a.o == b.s)) || (b.o != NO_ROLE && a.s == b.o);
            if a.s == b.s || obj_link {
                link_hist[other - focal] += 1;
            }
        }
    }
    let total_hist: usize = link_hist.iter().sum();
    let cum_at = |r: usize| -> usize { link_hist.iter().take(r + 1).sum() };
    let frac_within_5 = if total_hist > 0 {
        cum_at(RADIUS) as f64 / total_hist as f64
    } else {
        0.0
    };
    // Coherence length L90 = smallest forward distance capturing ≥90% of links.
    let mut coherence_len = SCAN_MAX;
    for d in 1..=SCAN_MAX {
        if total_hist > 0 && cum_at(d) * 10 >= total_hist * 9 {
            coherence_len = d;
            break;
        }
    }

    // ── Stage 4: NARS ambiguity resolution over the committed graph. ──
    let deductions = graph.infer_deductions(); // 2-hop A→B, B→C ⇒ A→C
    let contradictions = graph.detect_contradictions(0.5); // same S+O, different relation

    // ── Multilayer readout: symbolic vocabulary + Leiden basins + paradox. ──
    // The "layers" of the communication: the surface SPO, the recurring symbols
    // (which words the text orbits), the community/basin partition + its
    // modularity Q (low Q = one center of gravity; high Q = distinct themes),
    // and the NARS contradictions (where the text says two things at once).
    let communities = graph.communities();
    let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for t in &graph.triplets {
        *freq.entry(t.subject.as_str()).or_default() += 1;
        if !t.object.is_empty() {
            *freq.entry(t.object.as_str()).or_default() += 1;
        }
    }
    let mut symbols: Vec<(&str, usize)> = freq.into_iter().collect();
    symbols.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    // Optional: emit every surfaced symbol + its corpus frequency as TSV for the
    // codebook distributional-outlier filter (`EMIT_SYMBOLS=<path>`). The filter
    // joins these against `deepnsm/word_frequency/forms_5k.csv` (COCA base rank +
    // per-genre columns) to flag OOV-collapse sinks (e.g. `jeans`) that the
    // rank-based grammar heuristic misses — the "use the codebook to resolve what
    // you found" leg, zero-LLM.
    if let Ok(path) = std::env::var("EMIT_SYMBOLS") {
        let mut tsv = String::from("symbol\tfreq\n");
        for (w, n) in &symbols {
            tsv.push_str(&format!("{w}\t{n}\n"));
        }
        std::fs::write(&path, tsv).unwrap_or_else(|e| panic!("write {path}: {e}"));
        eprintln!("[EMIT_SYMBOLS] wrote {} symbols → {path}", symbols.len());
    }
    symbols.truncate(6);

    // ── Stage 5: SPO → SpoFacet 6×(8:8). Byte-split shim for rank→(basin:identity). ──
    let pair = |i: u16| ((i >> 8) as u8, (i & 0xFF) as u8);
    let mut facet_bytes = 0usize;
    let mut facet_roundtrip_ok = true;
    for t in &ranks {
        let f = SpoFacet::from_rails([
            pair(t.s),
            pair(t.p),
            pair(t.o),
            (0, 0),
            (0, 0),
            (0, 0), // no witness triple yet (A1 base design = 3 SPO + 3 episodic-witness)
        ]);
        // The 6×(8:8) reading must be loss-free (register roundtrip).
        if SpoFacet::from_register(f.to_register()) != f {
            facet_roundtrip_ok = false;
        }
        facet_bytes += f.to_register().len();
    }

    // ── Size proof: the COLD 512-byte NodeRow SoA (the persisted/calcified KG). ──
    let n_nodes = graph.triplets.len(); // deduped triples → committed nodes
    let cold_bytes = n_nodes * NODE_ROW_STRIDE; // 512 B/row
    let mb = |b: usize| b as f64 / 1_048_576.0;

    println!("── text_stream_to_soa : {} ──", input.display());
    println!("input        : {sentences} sentences, {tokens_total} COCA tokens (NO LLM, FSM only)");
    println!("stage1 FSM   : {n_triples} SPO triples extracted");
    println!("stage2 graph : {n_nodes} committed triples in TripletGraph (deduped)");
    println!("stage3 markov: ±{RADIUS} window → {context_links} role-continuity links");
    println!(
        "stage4 nars  : {} deductions (2-hop), {} contradictions",
        deductions.len(),
        contradictions.len()
    );
    println!(
        "stage5 facet : {n_triples} SpoFacets, {facet_bytes} bytes (6×(8:8)), roundtrip_ok={facet_roundtrip_ok}"
    );
    println!("── multilayer communication ──");
    let symline = symbols
        .iter()
        .map(|(w, n)| format!("{w}×{n}"))
        .collect::<Vec<_>>()
        .join("  ");
    println!("symbols      : {symline}");
    println!(
        "basins       : {} Leiden communities, modularity Q={:.3} ({} entities)",
        communities.num_communities,
        communities.modularity,
        communities.entities.len()
    );
    let hist_head = (1..=8)
        .map(|d| format!("{}", link_hist[d]))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "coherence    : L90={coherence_len} (forward links to capture 90%), \
         frac(±{RADIUS})={frac_within_5:.2}, hist[d=1..8]={hist_head}"
    );
    // Manufactured-symbol readout: canonical surface accounts for <15% of the
    // tokens that resolved to this symbol → tokenizer-manufactured (lemma/OOV/
    // proper-noun collapse). SUPERSET (also catches benign lemma-only inflection);
    // the proper-noun subset that needs case-preservation is a documented follow-up.
    let mut sinks: Vec<(&String, usize, f64)> = surface_fid
        .iter()
        .filter(|(_, &(_, total))| total >= 5)
        .map(|(w, &(matched, total))| (w, total, 1.0 - matched as f64 / total as f64))
        .filter(|&(_, _, manufactured)| manufactured > 0.85)
        .collect();
    sinks.sort_by(|a, b| b.2.total_cmp(&a.2).then(b.1.cmp(&a.1)));
    let sinkline = sinks
        .iter()
        .take(6)
        .map(|(w, total, m)| format!("{w}({total},{:.0}%)", m * 100.0))
        .collect::<Vec<_>>()
        .join("  ");
    println!(
        "manufactured : {} symbols >85% tokenizer-manufactured (canonical surface absent); top: {}",
        sinks.len(),
        if sinkline.is_empty() {
            "(none)".into()
        } else {
            sinkline
        }
    );
    let mut names: Vec<(&String, usize)> = name_freq.iter().map(|(w, n)| (w, *n)).collect();
    names.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let nameline = names
        .iter()
        .take(6)
        .map(|(w, n)| format!("{w}×{n}"))
        .collect::<Vec<_>>()
        .join("  ");
    println!(
        "names        : {} named-entity types (capitalized mid-sentence, no lemma collapse); top: {}",
        names.len(),
        if nameline.is_empty() {
            "(none)".into()
        } else {
            nameline
        }
    );
    println!(
        "paradox      : {} contradictions; sample:",
        contradictions.len()
    );
    for &(i, j) in contradictions.iter().take(3) {
        let a = &graph.triplets[i];
        let b = &graph.triplets[j];
        println!(
            "  \"{} {} {}\"  ⟂  \"{} {} {}\"",
            a.subject, a.relation, a.object, b.subject, b.relation, b.object
        );
    }
    println!("── size (COLD NodeRow SoA, 512 B/row = the persisted KG) ──");
    println!(
        "this run     : {n_nodes} nodes × {NODE_ROW_STRIDE} B = {cold_bytes} B ({:.4} MiB)",
        mb(cold_bytes)
    );
    println!(
        "budget       : 64k × 512 B = {:.0} MiB ; 256k × 512 B = {:.0} MiB",
        mb(64 * 1024 * 512),
        mb(256 * 1024 * 512)
    );
    println!(
        "note         : the ractor MailboxSoA (HOT working set) is ~6.2 KB/row (3 identity \
         planes); 512 B is the COLD NodeRow (three-tier model — this is what you store)."
    );
}
