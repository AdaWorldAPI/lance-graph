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
    let text = std::fs::read_to_string(&input)
        .unwrap_or_else(|e| panic!("read {}: {e}", input.display()));

    // ── Stage 1: COCA vocab (no LLM) — the 4096-word rank table. ──
    let vocab_dir = Path::new(manifest).join("../deepnsm/word_frequency");
    let vocab = Vocabulary::load(&vocab_dir).expect("load COCA vocabulary");

    // ── Stream: per-sentence 6-state PoS FSM → SPO triples. ──
    let mut graph = TripletGraph::new();
    let mut ranks: Vec<SpoRanks> = Vec::new(); // the Markov side-stream (1:1 with triples)
    let mut sentences = 0usize;
    let mut tokens_total = 0usize;

    for (ts, sentence) in text.split(['.', '!', '?']).enumerate() {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        sentences += 1;
        let toks = vocab.tokenize(s); // word → COCA token
        tokens_total += toks.len();
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
            if a.s == b.s || a.o == b.o || a.o == b.s || a.s == b.o {
                context_links += 1;
            }
        }
    }

    // ── Stage 4: NARS ambiguity resolution over the committed graph. ──
    let deductions = graph.infer_deductions(); // 2-hop A→B, B→C ⇒ A→C
    let contradictions = graph.detect_contradictions(0.5); // same S+O, different relation

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
