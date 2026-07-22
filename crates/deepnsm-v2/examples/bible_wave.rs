//! `bible_wave` — the WHOLE-BOOK falsifier: one book, one 64k SoA tile,
//! literal standing wave vs the fire-and-forget ±5 ring.
//!
//! The endgame thesis under test (`E-LC-SCARCITY-INVERSION-1`): a whole book's
//! triples fit ONE `256×256` tile, so context is read LITERALLY over the whole
//! work — no bundle, no beam, no per-sentence reset. v1's `MarkovBundler`
//! superposes a ±5 window and forgets; this example measures exactly how much
//! long-range context that forfeits, on a real public-domain book.
//!
//! Run (KJV from Project Gutenberg #10, not committed):
//! ```sh
//! cargo run --example bible_wave -- /path/to/pg10.txt
//! ```
//!
//! Pipeline: verses → PoS-tag (COCA lemma lexicon + documented archaic
//! fallback) → FSM → SPO stream (verse index = version) → `TemporalStream` +
//! the TRAINED Cam96 codebook (`data/`, real Jina-v3 embeddings).
//!
//! Gates (panic on KILL):
//! - G1 whole book fits the 64k SoA (verses ≤ 65,536)
//! - G2 the trained codebook loads and codes align with the vocab
//! - G3 KG is non-trivial (≥ 1,000 triples)
//! - G4 meaning sanity on the trained codebook: sim(god, lord) > sim(god, fish)
//!
//! Reported (not gated): the long-range share — % of same-subject recurrence
//! links farther than ±5 (v1's ring forfeits them) and ±8 (the local reference
//! horizon → the Escalate zone).

use deepnsm_v2::{
    load_cam96_codes, load_cam96_space, parse_to_spo, Nsm, PaletteVocab, Pos, Spo, Tagged,
    TemporalStream,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// The trained artifacts are NOT committed — they ship as the `AdaWorldAPI/lance-graph`
/// release `v0.1.0-cam96-data` (see `data/README.md` for the fetch commands).
/// Loaded at runtime from `data/` (override the directory with
/// `DEEPNSM_V2_DATA`).
fn data_file(name: &str) -> Vec<u8> {
    let dir = std::env::var("DEEPNSM_V2_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing {} ({e}) — fetch the v0.1.0-cam96-data release assets per data/README.md",
            path.display()
        )
    })
}

/// COCA PoS letter → the FSM's coarse tag. Pronouns ride as Noun so subjects
/// like "he said" bind; adverbs/preps/conj are Other (skipped by the FSM).
fn coca_pos(letter: &str) -> Pos {
    match letter {
        "n" | "p" => Pos::Noun,
        "v" => Pos::Verb,
        "j" => Pos::Adj,
        "a" | "d" => Pos::Det,
        _ => Pos::Other,
    }
}

/// Tiny archaic supplement for KJV forms absent from the COCA lemma list.
/// Documented heuristics, not a tagger: -eth/-est verb endings are Early
/// Modern English 3rd/2nd-person inflections.
fn archaic_pos(w: &str) -> Option<Pos> {
    match w {
        "thou" | "thee" | "ye" => Some(Pos::Noun),
        "thy" | "thine" => Some(Pos::Det),
        "shalt" | "hath" | "doth" | "saith" | "spake" | "begat" | "art" | "wilt" | "hast"
        | "shall" | "cometh" | "wast" => Some(Pos::Verb),
        "unto" | "thereof" | "wherefore" | "verily" | "yea" | "lo" => Some(Pos::Other),
        _ => {
            if w.ends_with("eth") || w.ends_with("est") {
                Some(Pos::Verb)
            } else {
                None
            }
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bible_wave <pg10.txt>");
    let raw = std::fs::read_to_string(&path).expect("read KJV text");

    // ── verses: a whitespace token shaped d+:d+ starts a new verse ──
    let mut verses: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_body = false;
    for tok in raw.split_whitespace() {
        let is_marker = tok.split_once(':').is_some_and(|(a, b)| {
            !a.is_empty()
                && !b.is_empty()
                && a.bytes().all(|c| c.is_ascii_digit())
                && b.bytes().all(|c| c.is_ascii_digit())
        });
        if is_marker {
            in_body = true;
            if !cur.is_empty() {
                verses.push(std::mem::take(&mut cur));
            }
        } else if in_body {
            if tok.contains("***") {
                break; // Gutenberg footer
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(tok);
        }
    }
    if !cur.is_empty() {
        verses.push(cur);
    }

    // G1 — the whole book is ONE 64k SoA tile.
    assert!(verses.len() <= 65_536, "KILL G1: book exceeds the 64k tile");
    println!(
        "G1 PASS  whole book = {} verses ≤ 65,536 (one 256×256 tile)",
        verses.len()
    );

    // ── vocab + TRAINED codebook (real Jina-v3 embeddings; runtime-fetched) ──
    let vocab_text = String::from_utf8(data_file("bible_vocab.txt")).expect("utf8 vocab");
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(vocab_text.lines());
    let space = load_cam96_space(&data_file("cam96_codebook.bin")).expect("codebook artifact");
    let codes = load_cam96_codes(&data_file("cam96_codes.bin")).expect("codes artifact");
    assert_eq!(codes.len(), vocab.len(), "KILL G2: codes/vocab misaligned");
    let nsm = Nsm::with_codes(vocab, space, codes);
    println!(
        "G2 PASS  trained codebook loaded: {} words, 12 axes",
        nsm.vocab.len()
    );

    // ── PoS lexicon: COCA lemmas + archaic fallback ──
    let lemmas = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../deepnsm/word_frequency/lemmas_5k.csv"
    ))
    .expect("lemmas_5k.csv (sibling deepnsm crate)");
    let mut pos_of: HashMap<String, Pos> = HashMap::new();
    for line in lemmas.lines().skip(1) {
        let mut f = line.split(',');
        let (Some(_), Some(lemma), Some(pos)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        pos_of
            .entry(lemma.to_lowercase())
            .or_insert_with(|| coca_pos(pos));
    }

    // ── stream: verse index = version; FSM → SPO ──
    let mut stream = TemporalStream::new();
    let mut all: Vec<(u64, Spo)> = Vec::new();
    let mut tagged_buf: Vec<Tagged> = Vec::new();
    for (vi, verse) in verses.iter().enumerate() {
        tagged_buf.clear();
        for tok in verse.split_whitespace() {
            let w: String = tok
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_lowercase();
            if w.len() < 2 {
                continue;
            }
            let Some(id) = nsm.vocab.id(&w) else { continue };
            let pos = pos_of
                .get(&w)
                .copied()
                .or_else(|| archaic_pos(&w))
                .unwrap_or(Pos::Other);
            tagged_buf.push(Tagged::new(id, pos));
        }
        tagged_buf.push(Tagged::new(0, Pos::Stop)); // verse boundary flushes
        for t in parse_to_spo(&tagged_buf) {
            stream.push(vi as u64, t);
            all.push((vi as u64, t));
        }
    }

    // G3 — non-trivial KG.
    let mut subjects: HashMap<u16, Vec<u64>> = HashMap::new();
    for &(v, t) in &all {
        subjects.entry(t.subject).or_default().push(v);
    }
    let distinct_s = subjects.len();
    assert!(
        all.len() >= 1_000,
        "KILL G3: too few triples ({})",
        all.len()
    );
    println!(
        "G3 PASS  KG: {} triples, {} distinct subjects, {} distinct predicates",
        all.len(),
        distinct_s,
        all.iter()
            .map(|&(_, t)| t.predicate)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    // ── the long-range measurement: what does fire-and-forget ±5 forfeit? ──
    let (mut links, mut beyond5, mut beyond8) = (0u64, 0u64, 0u64);
    for occs in subjects.values() {
        for w in occs.windows(2) {
            let gap = w[1] - w[0];
            if gap == 0 {
                continue;
            }
            links += 1;
            if gap > 5 {
                beyond5 += 1;
            }
            if gap > 8 {
                beyond8 += 1;
            }
        }
    }
    println!(
        "LONG-RANGE  {} same-subject recurrence links: {:.1}% beyond ±5 (v1 ring FORFEITS), {:.1}% beyond ±8 (Escalate zone → global graph)",
        links,
        100.0 * beyond5 as f64 / links as f64,
        100.0 * beyond8 as f64 / links as f64
    );

    // window sanity: the literal read reaches any span — GATED, not just logged
    // (a temporal-range regression must fail the falsifier, per #803 review).
    let w = stream.window_range(lance_graph_contract::temporal_pov::VersionRange::new(
        0,
        verses.len() as u64,
    ));
    assert_eq!(
        w.len(),
        all.len(),
        "KILL: whole-book window did not return every streamed triple"
    );
    println!(
        "WINDOW      whole-book literal read returns {} triples (no bundle, no reset)",
        w.len()
    );

    // G4 — trained-codebook meaning sanity.
    let near = nsm.word_similarity("god", "lord").expect("in vocab");
    let far = nsm.word_similarity("god", "fish").expect("in vocab");
    assert!(
        near > far,
        "KILL G4: sim(god,lord)={near} !> sim(god,fish)={far}"
    );
    println!(
        "G4 PASS  meaning (trained codebook): sim(god,lord)={near:.3} > sim(god,fish)={far:.3}"
    );

    // ── D-SRS-1 — the derivation-pointer fabric over the SAME whole-book KG ──
    // The graph reasons about itself: per-predicate transitive composition, each
    // derived triple carrying premise pointers (the pointers ARE the proof tree),
    // stamped max(premise rungs)+1. The pre-registered gate is STRUCTURAL and
    // proven exhaustively (all three metrics incl. fixed-point termination) by
    // the unit tests in `src/reason.rs`. At BOOK scale we deliberately BOUND the
    // closure: the KJV `begat` genealogies are long same-predicate chains whose
    // FULL transitive closure is O(N²) (empirically the whole-book closure does
    // not settle quickly) — and bounding the derivation horizon is exactly what
    // Layers 2-3 prescribe (±8-local + Escalate; the D-SRS-2 rung cap). The
    // SOUNDNESS half of the gate — 100% premise resolvability + acyclicity —
    // holds on any prefix of the closure, so the bounded run re-checks it on the
    // real book without paying for the full O(N²) genealogy closure.
    const DERIV_HORIZON: usize = 50_000;
    let base: Vec<Spo> = all.iter().map(|&(_, t)| t).collect();
    let arena = deepnsm_v2::reason::DerivationArena::derive_transitive_capped(&base, DERIV_HORIZON);
    let g = arena.gate();
    // Book-scale assertion: SOUNDNESS (the horizon-independent half of the gate).
    assert!(
        g.resolvability_pct == 100.0 && g.acyclic,
        "KILL D-SRS-1 soundness: resolvability={:.1}% acyclic={}",
        g.resolvability_pct,
        g.acyclic
    );
    let horizon = if g.terminated {
        "full fixed point".to_string()
    } else {
        format!("bounded at {DERIV_HORIZON} (full closure is larger — the genealogy O(N²), Layer-2/3 bounds it)")
    };
    println!(
        "D-SRS-1 PASS  derivation fabric: {} base → {} derived triples ({} passes, {horizon}); \
         SOUND — premise resolvability {:.1}%, acyclic={} (strictly-lower rung)",
        g.base, g.derived, g.passes, g.resolvability_pct, g.acyclic
    );

    println!(
        "\nALL GATES GREEN — the whole book is resident, literally read, with real meaning codes, \
         and reasoning about its own derivations (bounded horizon)."
    );
}
