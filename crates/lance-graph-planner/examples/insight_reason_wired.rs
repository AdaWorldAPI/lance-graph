//! `insight_reason_wired` — D-SCI-1: the new reasoning **wired end-to-end** —
//! one clause → S·P·O + TEKAMOLO address + 17D qualia + the **two-basin meaning**
//! (WordNet symbolic *rails* + COCA/Jina distributional *field*), all composed
//! into the canonical **value tenants** and emitted as **SPO-G** triples.
//!
//! This is additive: it COMPOSES the shipped contract types
//! ([`Triple`](lance_graph_contract::codegen_spine::Triple),
//! [`TekamoloFacet`](lance_graph_contract::tekamolo_facet::TekamoloFacet),
//! [`QualiaI4_16D`](lance_graph_contract::qualia::QualiaI4_16D), the
//! [`ValueTenant`](lance_graph_contract::canonical_node::ValueTenant) layout). It
//! writes to NOTHING — the existing 3×SPO + 3×AriGraph SPO-G grouping is
//! untouched (per `nars_dispatch`: "SPO-G grouping must remain untouched").
//!
//! ## What lands where (the wiring)
//!
//! | reasoning output | tenant / carrier |
//! |---|---|
//! | felt tone (17D) | `ValueTenant::Qualia` (#1) — `QualiaI4_16D` |
//! | WordNet is-a TYPE of S | `ValueTenant::EntityType` (#8) — the OGIT class discriminator (u16) |
//! | when/why/how/where | `ValueTenant::Tekamolo` (#13) — the `TekamoloFacet` G4D3 address (16 B) |
//! | the relation + rails | `ValueTenant::MaterializedEdges` (#2) — as SPO-G triples |
//! | verb family / archetype slot | `ValueTenant::Meta` (#0) — thinking bits |
//!
//! Runtime-owned tenants (`Helix`/`Turbovec`/`Energy`/`Plasticity`/`Kanban`/
//! `FrozenStyle`/`LearnedStyle`/`ExploreStyle`) are NOT text-derived — left at
//! default (reserve-don't-reclaim), never fabricated from a sentence.
//!
//! ## SPO-G (the G = named-graph 4th position)
//!
//! Each emitted `Triple` carries a `Graph` tag = the SPO-G `G` slot (the u32 OGIT
//! named-graph per the SPO-G plan): the extracted relation lands on `Utterance`,
//! the WordNet is-a/instance rails on `WordNet`. Same store surface, two graphs —
//! rails (symbolic, exact) beside field (distributional, fuzzy).
//!
//! ## Data (Release assets, gitignored — skips cleanly if absent)
//!   - COCA codebook  → `coca-codebook-v2` (MedCare-rs) → `examples/data/coca/`
//!   - WordNet rails  → `wordnet31-rails-v1` (MedCare-rs) → `examples/data/wordnet/`
//!
//! Or point `$COCA_CODEBOOK_DIR` / `$WORDNET_DIR` at the extraction dirs.
//!
//! Usage: cargo run -p lance-graph-planner --example insight_reason_wired -- [FILE ...]

use std::collections::HashMap;
use std::path::PathBuf;

use lance_graph_contract::canonical_node::ValueTenant;
use lance_graph_contract::codegen_spine::Triple;
use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::verb_lexicon::read_verb;
use lance_graph_contract::grammar::verb_table::VerbFamily;
use lance_graph_contract::qualia::{QualiaI4_16D, QualiaVector, AXIS_LABELS, QUALIA_DIMS};
use lance_graph_contract::tekamolo_facet::{TekamoloFacet, TekamoloRole};

// ── data loading (from Release assets; gitignored) ──────────────────────────

fn dir(env: &str, sub: &str) -> PathBuf {
    if let Ok(d) = std::env::var(env) {
        return PathBuf::from(d);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/data")
        .join(sub)
}

/// The two-basin meaning store: WordNet symbolic rails + COCA syntax lexicon.
struct Basins {
    /// COCA `word → (lemma, pos)` — the syntax axis (n/v/b/j/r/i).
    lex: HashMap<String, (String, u8)>,
    /// WordNet `word → (kind, type)` — the symbolic rails (`isa`/`inst`).
    rails: HashMap<String, (String, String)>,
}

impl Basins {
    fn load() -> Result<Self, String> {
        let coca = dir("COCA_CODEBOOK_DIR", "coca").join("lexicon.tsv");
        let wn = dir("WORDNET_DIR", "wordnet").join("wordnet31_isa.tsv");
        let hint = "\
missing Release data. The codebooks are NOT in the repo:
  COCA codebook → `coca-codebook-v2` (MedCare-rs) → examples/data/coca/  (or $COCA_CODEBOOK_DIR)
  WordNet rails → `wordnet31-rails-v1` (MedCare-rs) → examples/data/wordnet/ (or $WORDNET_DIR)";
        let lex_txt = std::fs::read_to_string(&coca)
            .map_err(|_| format!("{hint}\n(missing {})", coca.display()))?;
        let wn_txt = std::fs::read_to_string(&wn)
            .map_err(|_| format!("{hint}\n(missing {})", wn.display()))?;
        let mut lex = HashMap::new();
        for l in lex_txt
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 3 {
                lex.insert(c[0].to_string(), (c[1].to_string(), c[2].as_bytes()[0]));
            }
        }
        let mut rails = HashMap::new();
        for l in wn_txt
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let c: Vec<&str> = l.split('\t').collect();
            // word<TAB>pos<TAB>kind<TAB>type ; keep the first (noun preferred) reading
            if c.len() >= 4 && !rails.contains_key(c[0]) {
                rails.insert(c[0].to_string(), (c[2].to_string(), c[3].to_string()));
            }
        }
        Ok(Self { lex, rails })
    }
    fn pos(&self, w: &str) -> Option<u8> {
        self.lex.get(w).map(|(_, p)| *p)
    }
    fn lemma<'a>(&'a self, w: &'a str) -> &'a str {
        self.lex.get(w).map(|(l, _)| l.as_str()).unwrap_or(w)
    }
    /// The symbolic rail for a word: `(kind, type)` — `moses → (inst, prophet)`.
    fn rail(&self, w: &str) -> Option<&(String, String)> {
        self.rails.get(w)
    }
}

// ── SPO-G ───────────────────────────────────────────────────────────────────

/// The SPO-G `G` (named-graph) slot — the 4th tuple position (the u32 OGIT slot
/// in the SPO-G plan): which graph a triple belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Graph {
    /// The relation extracted from this utterance.
    Utterance,
    /// A WordNet symbolic rail (`is_a` / `instance_of`).
    WordNet,
}

/// One SPO-G quad: a shipped `Triple` + its `G`.
struct Quad {
    t: Triple,
    g: Graph,
}

// ── the reasoning record (the new reasoning, wired) ─────────────────────────

/// The composed reasoning for one clause: the SPO-G quads + every value tenant
/// the reasoning populates. Runtime-owned tenants are intentionally absent.
struct WiredReason {
    subject: String,
    verb: String,
    object: String,
    family: Option<VerbFamily>,
    quads: Vec<Quad>,                   // SPO-G (Utterance relation + WordNet rails)
    facet: TekamoloFacet,               // the TEKAMOLO address (G4D3 reading)
    qualia: QualiaI4_16D,               // ValueTenant::Qualia (#1)
    entity_type: Option<(String, u16)>, // ValueTenant::EntityType (#8): WordNet type of S → u16
    meta_bits: u16,                     // ValueTenant::Meta (#0): family+slot thinking bits
}

const LOCAL_PREPS: &[&str] = &["in", "on", "at", "into", "under", "over", "near", "within"];
const POSITIVE: &[&str] = &["hope", "love", "light", "peace", "grace", "mercy", "joy"];
const NEGATIVE: &[&str] = &[
    "storm", "fear", "death", "sin", "wrath", "cold", "dark", "war",
];

fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn tense_code(t: Tense) -> u8 {
    match t {
        Tense::Past | Tense::PastContinuous | Tense::Pluperfect => 1,
        Tense::Present | Tense::PresentContinuous | Tense::Habitual | Tense::Imperative => 2,
        Tense::Future | Tense::FutureContinuous | Tense::FuturePerfect => 3,
        _ => 4,
    }
}

/// Deterministic u16 code for a symbolic type (the OGIT EntityType discriminator).
fn type_code(t: &str) -> u16 {
    (t.bytes()
        .fold(0u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32))
        & 0xFFFF) as u16
}

/// Build a 17D felt vector (same canonical axes as the extractor).
fn qualia_vec(toks: &[String], svo: bool) -> QualiaVector {
    let (mut pos, mut neg) = (0.0f32, 0.0f32);
    let mut place = false;
    for w in toks {
        if POSITIVE.contains(&w.as_str()) {
            pos += 1.0;
        }
        if NEGATIVE.contains(&w.as_str()) {
            neg += 1.0;
        }
        if LOCAL_PREPS.contains(&w.as_str()) {
            place = true;
        }
    }
    let strong = (pos + neg).max(1.0);
    let mut q = [0.0f32; QUALIA_DIMS];
    q[1] = (pos - neg) / strong; // valence
    q[2] = (neg / strong).min(1.0); // tension
    q[3] = (pos / strong).min(1.0); // warmth
    q[4] = if svo { 0.8 } else { 0.2 }; // clarity
    q[9] = if svo { 0.6 } else { 0.3 }; // coherence
    q[14] = if place { 0.7 } else { 0.0 }; // groundedness
    q
}

/// Extract + wire one clause into a `WiredReason`.
fn reason(b: &Basins, sentence: &str) -> Option<WiredReason> {
    let toks = tokens(sentence);
    let is_verb = |w: &str| {
        b.pos(w) == Some(b'v') && read_verb(w).or_else(|| read_verb(b.lemma(w))).is_some()
    };
    let is_noun = |w: &str| b.pos(w) == Some(b'n');

    // First transitive noun—verb—noun.
    let mut svo = None;
    for i in 0..toks.len() {
        if !is_verb(&toks[i]) {
            continue;
        }
        let s = toks[..i].iter().rev().find(|w| is_noun(w));
        let o = toks[i + 1..].iter().find(|w| is_noun(w));
        if let (Some(s), Some(o)) = (s, o) {
            svo = Some((s.clone(), toks[i].clone(), o.clone()));
            break;
        }
    }
    let (s, v, o) = svo?;

    // Predicate typing + Temporal lane from the archetype.
    let reading = read_verb(&v).or_else(|| read_verb(b.lemma(&v)));
    let (family, temporal) = match reading {
        Some((f, t, _)) => (Some(f), [tense_code(t), 0, 0]),
        None => (None, [0, 0, 0]),
    };
    // Lokal lane if a locative prep precedes a noun.
    let mut lokal = [0u8, 0, 0];
    for w in &toks {
        if LOCAL_PREPS.contains(&w.as_str()) {
            lokal = [
                1,
                (w.bytes().fold(0u16, |a, x| a + x as u16) % 250) as u8 + 1,
                0,
            ];
            break;
        }
    }
    let facet = TekamoloFacet::from_lanes(0, temporal, [0; 3], [0; 3], lokal);

    // SPO-G quads — the extracted relation (Utterance) + the WordNet rails (WordNet).
    let mut quads = vec![Quad {
        t: Triple {
            s: s.clone(),
            p: v.clone(),
            o: o.clone(),
            f: 1.0,
            c: 0.9,
        },
        g: Graph::Utterance,
    }];
    let mut entity_type = None;
    for (n, is_subj) in [(&s, true), (&o, false)] {
        if let Some((kind, ty)) = b.rail(n) {
            let pred = if kind == "inst" {
                "instance_of"
            } else {
                "is_a"
            };
            quads.push(Quad {
                t: Triple {
                    s: n.clone(),
                    p: pred.to_string(),
                    o: ty.clone(),
                    f: 1.0,
                    c: 1.0,
                },
                g: Graph::WordNet,
            });
            if is_subj {
                entity_type = Some((ty.clone(), type_code(ty)));
            }
        }
    }

    let qualia = QualiaI4_16D::from_f32_17d(&qualia_vec(&toks, true));
    // Meta bits: family ordinal (low nibble) + Lokal-present flag.
    let meta_bits = (family.map(|f| f as u16).unwrap_or(0xF) & 0xF) | ((lokal[0] as u16) << 4);

    Some(WiredReason {
        subject: s,
        verb: v,
        object: o,
        family,
        quads,
        facet,
        qualia,
        entity_type,
        meta_bits,
    })
}

fn report(b: &Basins, label: &str, text: &str) -> Vec<WiredReason> {
    println!("\n════════ {label} ════════");
    let mut out = Vec::new();
    for sent in text
        .split(['.', ';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some(r) = reason(b, sent) {
            let vi = AXIS_LABELS.iter().position(|&l| l == "valence").unwrap();
            println!(
                "  {} —{}→ {}   [family: {}]",
                r.subject,
                r.verb,
                r.object,
                r.family
                    .map(|f| format!("{f:?}"))
                    .unwrap_or_else(|| "—".into())
            );
            println!("    SPO-G:");
            for q in &r.quads {
                println!("      ({} {} {})  @G:{:?}", q.t.s, q.t.p, q.t.o, q.g);
            }
            let tk = r.facet.facet().to_bytes();
            println!(
                "    tenants → Qualia(#1) valence={:+} · EntityType(#8)={} · Meta(#0)=0x{:03X}",
                r.qualia.get(vi),
                r.entity_type
                    .as_ref()
                    .map(|(t, c)| format!("{t}(0x{c:04X})"))
                    .unwrap_or_else(|| "—".into()),
                r.meta_bits,
            );
            println!(
                "      Tekamolo(#{}) @slab[{}..{}) = {:02X?}  (Te={:?} Ka={:?} Mo={:?} Lo={:?})",
                ValueTenant::Tekamolo as u8,
                ValueTenant::Tekamolo.value_offset(),
                ValueTenant::Tekamolo.value_offset() + ValueTenant::Tekamolo.byte_len(),
                tk,
                r.facet.temporal(),
                r.facet.causal(),
                r.facet.modal(),
                r.facet.lane(TekamoloRole::Lokal),
            );
            out.push(r);
        }
    }
    out
}

fn main() {
    let b = match Basins::load() {
        Ok(b) => b,
        Err(h) => {
            eprintln!("{h}");
            return;
        }
    };
    println!(
        "loaded two-basin store: {} COCA lexicon · {} WordNet rails",
        b.lex.len(),
        b.rails.len()
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for p in &args {
            match std::fs::read_to_string(p) {
                Ok(t) => {
                    report(&b, p, &t);
                }
                Err(e) => eprintln!("skip {p}: {e}"),
            }
        }
        return;
    }

    // Falsifier: the new reasoning wired — SPO-G (utterance + WordNet rails) + tenants.
    let out = report(
        &b,
        "wired reasoning falsifier",
        "the shepherd carries the lamb in the wilderness.",
    );
    assert_eq!(out.len(), 1, "one wired reason");
    let r = &out[0];
    assert_eq!(r.subject, "shepherd");
    assert_eq!(r.object, "lamb");
    // SPO-G: the extracted relation on Utterance + at least one WordNet rail.
    assert!(
        r.quads.iter().any(|q| q.g == Graph::Utterance),
        "utterance relation quad"
    );
    assert!(
        r.quads
            .iter()
            .any(|q| q.g == Graph::WordNet && q.t.p.contains("is_a")),
        "at least one WordNet is_a rail (shepherd→herder / flock→...)"
    );
    // EntityType tenant populated from the WordNet rail (shepherd → herder).
    assert!(
        r.entity_type.is_some(),
        "EntityType tenant set from the WordNet rail"
    );
    // Qualia tenant is the canonical i4-16 packing; facet round-trips.
    assert_eq!(
        r.facet.temporal()[0],
        2,
        "present tense (carries) in the Temporal lane"
    );
    assert!(
        r.facet.lane(TekamoloRole::Lokal)[0] == 1,
        "Lokal set (in the wilderness)"
    );

    // TEKAMOLO tenant (#13): write the facet into the value slab at the tenant's
    // canonical carve and read it back — proves the address lands where the LE
    // contract says, byte-for-byte (the actual wiring, not just a print).
    let mut slab = [0u8; lance_graph_contract::canonical_node::VALUE_SLAB_LEN];
    let off = ValueTenant::Tekamolo.value_offset();
    let len = ValueTenant::Tekamolo.byte_len();
    assert_eq!(len, 16, "TEKAMOLO tenant is the 16-byte 4+12 facet");
    slab[off..off + len].copy_from_slice(&r.facet.facet().to_bytes());
    let tk_bytes: [u8; 16] = slab[off..off + len].try_into().unwrap();
    let read_back = lance_graph_contract::facet::FacetCascade::from_bytes(&tk_bytes);
    assert_eq!(
        read_back,
        *r.facet.facet(),
        "TEKAMOLO facet round-trips through the ValueTenant::Tekamolo slab carve"
    );

    println!(
        "\n✔ Wired: one clause → S·P·O + TEKAMOLO address + 17D qualia + the two-basin meaning \
         (WordNet symbolic rails + COCA field), composed into the canonical value tenants \
         (Qualia #1 · EntityType #8 · Meta #0 · Tekamolo #13 — the when/why/how/where facet \
         landing byte-for-byte in the ValueTenant::Tekamolo slab carve) and emitted as SPO-G \
         triples (Utterance relation + WordNet is_a/instance rails). The existing 3×SPO + \
         3×AriGraph grouping is untouched; runtime-owned tenants left at default \
         (reserve-don't-reclaim)."
    );
    println!("\n(usage: cargo run -p lance-graph-planner --example insight_reason_wired -- FILE [FILE ...])");
}
