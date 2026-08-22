//! End-to-end over the REAL corpus: spawn the TOC tree, hydrate the triple
//! stream over it, promote basins at TOC node keys.
//!
//! ```sh
//! cargo run --release --example toc_hydrate -- /path/to/pg10.txt
//! ```
//!
//! Loading and tagging mirror `bible_wave` exactly (same vocab, same trained
//! Cam96 codebook, same COCA lemma lexicon + archaic fallback) so this cannot
//! silently disagree with the pipeline that already runs.

use deepnsm_v2::basin::basin_self_code;
use deepnsm_v2::codebook::{load_cam96_codes, load_cam96_space};
use deepnsm_v2::corpus::split_verses_detailed;
use deepnsm_v2::fsm::{parse_to_spo, Pos, Tagged};
use deepnsm_v2::hydrate::hydrate;
use deepnsm_v2::lexicon::{normalise, Lexicon};
use deepnsm_v2::promote::{key_at, promote, read_lane, row_of};
use deepnsm_v2::spo::Spo;
use deepnsm_v2::vocab::PaletteVocab;
use lance_graph_contract::hhtl::NiblePath;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const BASIN: u8 = 3;
// A KJV corpus concept has NO mint — that is a real gap, not a detail. This
// borrows OSINT-V3 for ONE property only: it resolves to the V3 tail, so the
// key can carry `leaf` and read its own identity back. It says nothing about
// what a scripture node IS; the actual address needs an OGAR mint.
//
// The previous stand-in here was `0x0301_0000` (MONDO's block), which has no
// registry entry and therefore resolves to the V1 tail. `key_at` now refuses
// that outright — see its doc for how silently it used to succeed.
const CLASSID: u32 = lance_graph_contract::canonical_node::NodeGuid::CLASSID_OSINT_V3;

fn data_file(name: &str) -> Vec<u8> {
    let dir = std::env::var("DEEPNSM_V2_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: toc_hydrate <pg10.txt>");
    let text = std::fs::read_to_string(&path).expect("corpus");

    // ── verses + their (chapter, verse) markers, from ONE walk ──
    // This used to be two walks: `split_verses` for the text, and a separate
    // `split_whitespace().filter(is_verse_marker)` over the RAW text for the
    // markers. The second walk did not apply the footer trim, so any `d+:d+`
    // in the Gutenberg license would have been admitted as a verse marker and
    // de-aligned every index after it. On the shipped file it happened to be
    // clean (measured: 0 markers outside the body); nothing guaranteed it.
    // `CorpusSplit::markers` is aligned with `verses` by construction.
    let split = split_verses_detailed(&text);
    let (verses, markers) = (split.verses, split.markers);
    println!(
        "corpus      {} verses, {} markers",
        verses.len(),
        markers.len()
    );
    assert_eq!(
        markers.len(),
        verses.len(),
        "marker/verse misalignment — the join key would be wrong"
    );

    // ── vocab + trained codebook (the DOCUMENTED artifacts) ──
    let vocab_text = String::from_utf8(data_file("bible_vocab.txt")).expect("utf8");
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(vocab_text.lines());
    let space = load_cam96_space(&data_file("cam96_codebook.bin")).expect("codebook");
    let codes = load_cam96_codes(&data_file("cam96_codes.bin")).expect("codes");
    assert_eq!(codes.len(), vocab.len(), "codes/vocab misaligned");
    println!("codebook    {} words, {} codes", vocab.len(), codes.len());

    // ── PoS lexicon: COCA lemmas + FORMS + archaic fallback ──
    // One lexicon, in the crate. Reading `lemmas_5k.csv` alone left every
    // inflected verb (`created`, `made`, `called`) untagged; see
    // `deepnsm_v2::lexicon` for the measurement.
    let lexicon = Lexicon::from_coca(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../deepnsm/word_frequency/lemmas_5k.csv"
        ))
        .expect("lemmas_5k.csv"),
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../deepnsm/word_frequency/word_forms.csv"
        ))
        .expect("word_forms.csv"),
    );
    println!(
        "lexicon     {} lemmas + {} forms",
        lexicon.sizes().0,
        lexicon.sizes().1
    );

    // ── per-verse SPO, same tagging as bible_wave ──
    let mut per_verse: Vec<Vec<Spo>> = Vec::with_capacity(verses.len());
    let mut tagged: Vec<Tagged> = Vec::new();
    for verse in &verses {
        tagged.clear();
        for tok in verse.split_whitespace() {
            let Some(w) = normalise(tok) else { continue };
            let Some(id) = vocab.id(&w) else { continue };
            tagged.push(Tagged::new(id, lexicon.pos(&w)));
        }
        tagged.push(Tagged::new(0, Pos::Stop));
        per_verse.push(parse_to_spo(&tagged));
    }
    let total: usize = per_verse.iter().map(Vec::len).sum();
    println!("triples     {total} from {} verses", verses.len());

    // ── SPAWN + HYDRATE ──
    let h = hydrate(&markers, &per_verse, BASIN);
    println!(
        "tree        {} nodes ({} books, {} verse nodes)",
        h.entries.len(),
        h.toc.book_count(),
        h.verse_nodes()
    );
    println!(
        "hydrated    {} addressed · {} barren verses · {} unaddressed",
        h.triples.len(),
        h.barren_verses,
        h.unaddressed
    );
    assert_eq!(
        h.triples.len() + h.unaddressed,
        total,
        "triples vanished in the join"
    );

    let chain = h.chain();
    let scan = h.scanpath();
    println!(
        "chain       {} links, {} cross-verse ({:.1}%)",
        chain.len(),
        h.cross_verse_links(),
        100.0 * h.cross_verse_links() as f64 / chain.len().max(1) as f64
    );
    println!("scanpath    {} entries (residue over the tree)", scan.len());

    // Every address must be a minted verse node.
    let minted: HashSet<NiblePath> = h
        .entries
        .iter()
        .filter(|e| e.level == deepnsm_v2::toc::TocLevel::Verse)
        .map(|e| e.path)
        .collect();
    let stray = h
        .triples
        .iter()
        .filter(|t| !minted.contains(&t.path))
        .count();
    println!("addressing  {stray} triples at an unminted address");
    assert_eq!(
        stray, 0,
        "a triple is at an address nothing can ascend from"
    );

    // Keys round-trip: fold a path into tiers, read it back through the canon.
    let mut checked = 0usize;
    for e in h.entries.iter().take(2000) {
        let k = key_at(CLASSID, e.path, 0, 1).expect("V3 classid mints");
        assert_eq!(
            NiblePath::from_guid_prefix_v3(&k).prefix(e.path.depth()),
            Some(e.path),
            "key does not carry its own address back"
        );
        checked += 1;
    }
    println!("keys        {checked} round-tripped through the v3 fold");

    // ── BASINS from the REAL codebook, promoted at TOC keys ──
    let mut by_subject: HashMap<u16, Vec<(u16, u16)>> = HashMap::new();
    for t in &h.triples {
        by_subject
            .entry(t.spo.subject)
            .or_default()
            .push((t.spo.predicate, t.spo.object));
    }
    let mut promoted = 0usize;
    let mut with_code = 0usize;
    let mut first_verse: HashMap<u16, NiblePath> = HashMap::new();
    for t in &h.triples {
        first_verse.entry(t.spo.subject).or_insert(t.path);
    }
    // The entry a path addresses, indexed once. Scanning `h.entries` inside the
    // loop was O(subjects x entries) — 702 x 32,357 — and only tolerable while
    // the loop was capped.
    let entry_at: HashMap<NiblePath, &deepnsm_v2::toc::TocEntry> =
        h.entries.iter().map(|e| (e.path, e)).collect();

    // Deterministic order. `HashMap::iter` is randomised per run, so the old
    // `.take(64)` promoted a DIFFERENT 64 subjects each time — which is fine
    // for proving the path and useless for anything that has to be reproduced.
    let mut subjects: Vec<u16> = by_subject.keys().copied().collect();
    subjects.sort_unstable();

    let mut rows = Vec::new();
    let mut no_members = 0usize;
    let mut no_code = 0usize;
    for subject in subjects {
        let edges = &by_subject[&subject];
        let members: Vec<[u8; 12]> = edges
            .iter()
            .filter_map(|(_, o)| codes.get(*o as usize).copied())
            .collect();
        if members.is_empty() {
            no_members += 1;
            continue;
        }
        let Some(code) = basin_self_code(&space, subject, &members, edges) else {
            no_code += 1;
            continue;
        };
        if code.self_code != [0u8; 12] {
            with_code += 1;
        }
        let path = first_verse[&subject];
        let entry = entry_at
            .get(&path)
            .expect("the address came from a minted node");
        // Identity is the SUBJECT, not a positional counter. The counter made
        // the key depend on iteration order — two runs would address the same
        // basin differently. The subject is the basin's own name, so the key
        // is stable across runs and across corpus growth, and the lane
        // round-trip below becomes a real check that key and payload agree.
        let row = promote(
            entry,
            &row_of(&code, 0, verses.len() as u64),
            CLASSID,
            u32::from(subject),
        )
        .expect("CLASSID resolves to the V3 tail, so the mint cannot be refused");
        assert_eq!(read_lane(&row).subject, subject, "lane round-trip");
        assert_eq!(
            row.key.identity_v2(),
            subject,
            "key identity must be the basin's own subject"
        );
        rows.push(row);
        promoted += 1;
    }

    // Every promoted basin must occupy its OWN address. Basins that first
    // appear in the same verse share a path, so the identity half is what
    // separates them — this is the assertion that proves it does.
    let distinct: HashSet<_> = rows.iter().map(|r| r.key).collect();
    assert_eq!(distinct.len(), rows.len(), "two basins collided on one key");
    let shared_path = rows.len()
        - rows
            .iter()
            .map(|r| NiblePath::from_guid_prefix_v3(&r.key))
            .collect::<HashSet<_>>()
            .len();
    println!("skipped     {no_members} with no coded member, {no_code} with no self-code");
    println!(
        "collision   {} distinct keys / {} rows · {shared_path} rows share a verse address",
        distinct.len(),
        rows.len()
    );
    println!(
        "basins      {} subjects total · {promoted} promoted at TOC keys · {with_code} carry a real Cam96 self-code",
        by_subject.len()
    );
    println!(
        "rows        {} NodeRows, {} bytes each",
        rows.len(),
        core::mem::size_of::<lance_graph_contract::canonical_node::NodeRow>()
    );
    println!("\nEND-TO-END GREEN over the real corpus.");
}
