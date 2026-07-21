//! PROBE PARTOF-ISA-vs-PALETTE256 — the two sanctioned `6×(8:8)` readings of
//! the SAME 12 bytes, measured on real substrate: WHAT DOES EACH HOLD?
//!
//! le-contract §3 offers two 6×(8:8) carvings of one payload:
//!   - **L1 `part_of:is_a`** — the V3 mereology:taxonomy rails: DISCRETE
//!     one-byte refs into a basin/type codebook. Holds STRUCTURE (exact
//!     hierarchy, transitive is_a, part membership).
//!   - **L4 `palette256²`** — CAM-PQ centroid pairs: each byte pair indexes
//!     the 256×256 palette distance table. Holds SIMILARITY (continuous,
//!     graded, metric).
//!
//! Same shape, same 12 bytes — but they are NOT interchangeable. This probe
//! computes both readings for a real vocabulary and asks the jc battery
//! whether they AGREE (Pearson / Spearman / ICC / Cronbach), and reports what
//! capacity each holds. Per `E-AWARENESS-TENANTS-EVOLVE-NOT-COLLAPSE-1` a LOW
//! agreement is the KEEP-BOTH result — the two readings are diverse-redundant
//! siblings measuring different facets, not duplicates to collapse.
//!
//! ## Real substrate (nothing planted)
//!
//! - **Vocabulary + palette256²:** the content nouns of the three Aesop fables
//!   (the merged `l9_loci_real_text` corpus) with their real COCA ranks
//!   (`deepnsm/word_frequency/lemmas_5k.csv`). The palette pair is
//!   `(rank>>8, rank&0xFF)`; the palette-table distance is the freq-rank
//!   distance `|Δrank|/16` (`E-FREQ-IS-COSINE-REPLACEMENT-1`) — the same table
//!   the merged probe used.
//! - **part_of:is_a:** objectively-true English taxonomy for those nouns
//!   (dog IS-A animal; animal IS-A creature; grape IS-A food …), the kind
//!   `deepnsm::spo_markov_kg` extracts from copular sentences (`"is"→"is_a"`).
//!   These are verifiable facts, not planted structure; the taxonomic distance
//!   is is_a-graph path length (transitive), normalised to the u8 table scale.
//!
//! ## What "holds" means, measured
//!
//! - **palette256²** resolution: how finely can it separate the vocabulary by
//!   its metric (distinct quantised distances observed).
//! - **part_of:is_a** resolution: how many taxonomic classes it distinguishes
//!   exactly (is_a roots reached).
//! - **Cross-agreement:** over all vocabulary PAIRS, the palette256² distance
//!   vs the is_a taxonomic distance — Pearson/Spearman/ICC. LOW ⇒ they hold
//!   different facets (both kept). Cronbach α over the two as a 2-item scale.
//!
//! ## Validity anchor (what each SAYS, against a gold contrast)
//!
//! Gold "same-kind" pairs (both animals; both people; both foods) SHOULD read
//! CLOSE under is_a (they share a hypernym) — is_a's job. Under palette256²
//! (frequency centroid) same-kind pairs are NOT systematically close (dog and
//! wolf have very different COCA ranks). The probe measures both: is_a
//! same-kind accuracy vs palette256² same-kind accuracy — showing WHICH
//! reading holds taxonomy and which holds frequency-similarity.
//!
//! ## Registered gates (fixed before first run)
//!
//! 1. WELL-POSED: every jc measure returns `Some`.
//! 2. is_a holds taxonomy: gold same-kind pairs read CLOSER than cross-kind
//!    under is_a (separation ≥ 0.20 on the [0,1] closeness scale).
//! 3. The two readings are NOT redundant: cross-reading Spearman |ρ| < 0.70
//!    (if they agreed strongly, one would be redundant — they do not).
//! KILL: any gate fails (recorded). Agreement magnitude is reported, its
//! direction measured not assumed.
//!
//! ## Run
//! ```bash
//! cargo run --manifest-path crates/jc/Cargo.toml --example partof_isa_vs_palette256
//! ```

use jc::reliability::{cronbach_alpha, icc, pearson, spearman, IccForm};
use std::collections::HashMap;

// ── Real vocabulary: the fable content nouns + true is_a hypernyms ──────────
// (lemma, hypernym) — objectively-true English taxonomy, the copula facts
// deepnsm's spo_markov_kg extracts ("a dog is an animal" -> dog is_a animal).
// Roots (animal/person/food/place/thing) chain up to `entity`.
const ISA: [(&str, &str); 26] = [
    ("dog", "animal"),
    ("fox", "animal"),
    ("wolf", "animal"),
    ("sheep", "animal"),
    ("bird", "animal"),
    ("animal", "creature"),
    ("boy", "person"),
    ("shepherd", "person"),
    ("villager", "person"),
    ("person", "creature"),
    ("creature", "entity"),
    ("grape", "food"),
    ("meat", "food"),
    ("food", "thing"),
    ("village", "place"),
    ("bridge", "place"),
    ("river", "place"),
    ("vine", "plant"),
    ("water", "substance"),
    ("plant", "thing"),
    ("substance", "thing"),
    ("place", "thing"),
    ("shadow", "thing"),
    ("mouth", "part"),
    ("part", "thing"),
    ("thing", "entity"),
];

/// The content nouns we score (leaves of the taxonomy — real fable entities).
const VOCAB: [&str; 16] = [
    "dog", "fox", "wolf", "sheep", "bird", "boy", "shepherd", "villager", "grape", "meat",
    "village", "bridge", "river", "vine", "water", "shadow",
];

/// Gold same-kind partition (the true kinds) — for the taxonomy-holding anchor.
fn kind(lemma: &str) -> &'static str {
    match lemma {
        "dog" | "fox" | "wolf" | "sheep" | "bird" => "animal",
        "boy" | "shepherd" | "villager" => "person",
        "grape" | "meat" => "food",
        "village" | "bridge" | "river" | "vine" | "water" | "shadow" => "place-thing",
        _ => "other",
    }
}

// ── COCA (committed real ranks) ─────────────────────────────────────────────
fn load_ranks() -> HashMap<String, u32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../deepnsm/word_frequency/lemmas_5k.csv"
    );
    let text = std::fs::read_to_string(path).expect("committed COCA table present");
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

// ── palette256² reading + its table distance (freq-is-cosine) ───────────────
fn palette_pair(rank: u32) -> (u8, u8) {
    ((rank >> 8) as u8, (rank & 0xFF) as u8)
}
fn palette_dist(a: u32, b: u32) -> u8 {
    ((a.abs_diff(b)) / 16).min(255) as u8
}

// ── part_of:is_a reading + its taxonomic (graph path) distance ──────────────
fn hyper_chain(lemma: &str, isa: &HashMap<&str, &str>) -> Vec<String> {
    let mut chain = vec![lemma.to_string()];
    let mut cur = lemma;
    for _ in 0..12 {
        match isa.get(cur) {
            Some(&h) => {
                chain.push(h.to_string());
                cur = h;
            }
            None => break,
        }
    }
    chain
}

/// Taxonomic distance = graph path length via the least common hypernym,
/// normalised to the u8 table scale (so it is comparable to palette_dist).
fn isa_dist(a: &str, b: &str, isa: &HashMap<&str, &str>) -> u8 {
    if a == b {
        return 0;
    }
    let ca = hyper_chain(a, isa);
    let cb = hyper_chain(b, isa);
    // depth of each to the least common ancestor.
    for (ia, ta) in ca.iter().enumerate() {
        if let Some(ib) = cb.iter().position(|tb| tb == ta) {
            // path = ia + ib hops; scale so 1 hop ~ 32 (a sibling pair = 64).
            return ((ia + ib) as u32 * 32).min(255) as u8;
        }
    }
    255 // no common ancestor (disjoint roots) — maximally far
}

fn main() {
    let ranks = load_ranks();
    let isa: HashMap<&str, &str> = ISA.iter().copied().collect();

    // resolve vocab ranks (all present in COCA 5k)
    let vocab: Vec<(&str, u32)> = VOCAB
        .iter()
        .filter_map(|&w| ranks.get(w).map(|&r| (w, r)))
        .collect();
    println!(
        "vocab: {}/{} content nouns resolved in COCA 5k",
        vocab.len(),
        VOCAB.len()
    );

    // ── what each HOLDS: resolution ─────────────────────────────────────────
    // palette256² distinct quantised distances observed over all pairs.
    let mut pal_vals = std::collections::BTreeSet::new();
    let mut isa_vals = std::collections::BTreeSet::new();
    let (mut pal_pairs, mut isa_pairs) = (Vec::new(), Vec::new());
    // gold same-kind vs cross-kind closeness under each reading.
    let (mut isa_same, mut isa_cross, mut pal_same, mut pal_cross) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for i in 0..vocab.len() {
        for j in (i + 1)..vocab.len() {
            let (wa, ra) = vocab[i];
            let (wb, rb) = vocab[j];
            let pd = palette_dist(ra, rb) as f64;
            let id = isa_dist(wa, wb, &isa) as f64;
            pal_vals.insert(palette_dist(ra, rb));
            isa_vals.insert(isa_dist(wa, wb, &isa));
            pal_pairs.push(pd);
            isa_pairs.push(id);
            // closeness = 1 - d/255 ∈ [0,1]
            let (pc, ic) = (1.0 - pd / 255.0, 1.0 - id / 255.0);
            if kind(wa) == kind(wb) {
                isa_same.push(ic);
                pal_same.push(pc);
            } else {
                isa_cross.push(ic);
                pal_cross.push(pc);
            }
        }
    }

    let pal0 = palette_pair(vocab[0].1);
    println!(
        "\npalette256²: e.g. '{}' r{} -> pair ({},{}); {} distinct quantised distances over {} pairs (continuous metric — holds SIMILARITY)",
        vocab[0].0, vocab[0].1, pal0.0, pal0.1, pal_vals.len(), pal_pairs.len()
    );
    // is_a distinct roots = taxonomy classes it distinguishes exactly.
    let roots: std::collections::BTreeSet<String> = vocab
        .iter()
        .map(|(w, _)| hyper_chain(w, &isa).last().unwrap().clone())
        .collect();
    println!(
        "part_of:is_a: {} distinct hypernym paths, {} exact taxonomic root(s) [{}]; {} distinct path-distances (discrete structure — holds TAXONOMY)",
        vocab.len(),
        roots.len(),
        roots.iter().cloned().collect::<Vec<_>>().join(","),
        isa_vals.len()
    );

    // ── the jc battery: do the two readings agree over the pairs? ───────────
    let r = pearson(&pal_pairs, &isa_pairs);
    let rho = spearman(&pal_pairs, &isa_pairs);
    let ratings: Vec<Vec<f64>> = pal_pairs
        .iter()
        .zip(&isa_pairs)
        .map(|(&p, &i)| vec![p, i])
        .collect();
    let i21 = icc(&ratings, IccForm::Icc2_1);
    let alpha = cronbach_alpha(&[pal_pairs.clone(), isa_pairs.clone()]);

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let isa_sep = mean(&isa_same) - mean(&isa_cross);
    let pal_sep = mean(&pal_same) - mean(&pal_cross);

    println!(
        "\n== palette256²  vs  part_of:is_a  (over {} vocab pairs) ==",
        pal_pairs.len()
    );
    println!("cross-reading pearson_r   = {}", fmt(r));
    println!(
        "cross-reading spearman_ρ  = {}   [gate |ρ| < 0.70 → NOT redundant]",
        fmt(rho)
    );
    println!("cross-reading icc(2,1)    = {}", fmt(i21));
    println!("cronbach_α (2-item)       = {}", fmt(alpha));
    println!("\n-- what each HOLDS (gold same-kind vs cross-kind closeness) --");
    println!(
        "part_of:is_a  same-kind {:.3} vs cross-kind {:.3}  → separation {:+.3}  [gate ≥ 0.20: taxonomy]",
        mean(&isa_same), mean(&isa_cross), isa_sep
    );
    println!(
        "palette256²   same-kind {:.3} vs cross-kind {:.3}  → separation {:+.3}  (frequency, NOT taxonomy)",
        mean(&pal_same), mean(&pal_cross), pal_sep
    );

    // ── registered gates ────────────────────────────────────────────────────
    assert!(
        r.is_some() && rho.is_some() && i21.is_some() && alpha.is_some(),
        "KILL gate 1: a jc measure returned None — battery not well-posed"
    );
    assert!(
        isa_sep >= 0.20,
        "KILL gate 2: part_of:is_a does not hold taxonomy (same-kind not closer than cross-kind by the margin)"
    );
    assert!(
        rho.unwrap().abs() < 0.70,
        "KILL gate 3: the two readings agree too strongly — one would be redundant (they should hold DIFFERENT facets)"
    );

    println!("\nPASS — the two 6×(8:8) readings hold DIFFERENT facets of the same 12 bytes:");
    println!(
        "part_of:is_a holds exact taxonomy (same-kind separation {:+.3}); palette256² holds",
        isa_sep
    );
    println!(
        "frequency-centroid similarity (separation {:+.3}, taxonomy-blind); cross-reading ρ={}",
        pal_sep,
        fmt(rho)
    );
    println!("→ diverse-redundant siblings, BOTH kept (evolve-not-collapse). jc measured it, did not assume it.");
}

fn fmt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "None".into(),
    }
}
