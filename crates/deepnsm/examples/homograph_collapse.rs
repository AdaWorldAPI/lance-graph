//! Homograph collapse — the SPO 2³ role mask as a superposition collapse operator.
//!
//! CLAIM. A surface word form is a SUPERPOSITION of readings: it maps to a SET
//! of `(lemma, PoS)` senses. The SPO 2³ role mask (Subject / Predicate / Object)
//! is the COLLAPSE OPERATOR — fixing which grammatical role a token fills selects
//! one reading, collapsing the superposition to a UNIQUE lemma, hence a UNIQUE
//! `lemRank` (the frequency-centroid address in the COCA-ranked vocabulary). So
//! the SAME surface string lands on DIFFERENT centroids depending on its slot.
//! This grounds "hold the meaning of all words in parallel, collapse by role"
//! on real data.
//!
//! Minimal collapse rule (a SIMPLIFICATION — see HONEST BOUNDARY below):
//!   S → expects a nominal ('n');  P → expects a verbal ('v');  O → nominal ('n').
//! Given `(surface, role)`, keep the senses whose PoS matches the role's class;
//! the collapse is UNIQUE iff exactly one sense survives.
//!
//! Data: `word_frequency/word_forms.csv` (COCA), header
//! `lemRank,lemma,PoS,lemFreq,wordFreq,word`. Std-only, hand-parsed.
//!
//! Run: cargo run --manifest-path crates/deepnsm/Cargo.toml --example homograph_collapse

use std::collections::HashMap;

/// One reading of a surface word: a `(lemma, PoS)` sense addressed by its
/// lemma-frequency rank. `lem_rank` IS the centroid address — the thing a role
/// collapse selects out of the superposition.
#[derive(PartialEq, Eq)]
struct Sense {
    lemma: String,
    pos: char,
    lem_rank: u32,
}

/// A grammatical slot in the SPO 2³ mask. Each slot fixes the PoS *class* the
/// token must fill: S and O are nominal, P is verbal. Fixing the role is the
/// COLLAPSE OPERATOR that selects one reading of a homograph.
#[derive(Clone, Copy)]
enum Role {
    Subject,
    Predicate,
    Object,
}

impl Role {
    /// The PoS letter this role's class expects: 'v' (verbal) for P, else 'n'.
    fn expected_pos(self) -> char {
        match self {
            Role::Predicate => 'v',
            Role::Subject | Role::Object => 'n',
        }
    }

    fn label(self) -> &'static str {
        match self {
            Role::Subject => "S",
            Role::Predicate => "P",
            Role::Object => "O",
        }
    }
}

/// Result of applying a role mask to a surface word's sense superposition.
#[derive(Clone, Copy)]
enum Collapse<'a> {
    /// Superposition collapsed to a single reading: `(lemma, lemRank centroid)`.
    Unique(&'a str, u32),
    /// More than one sense of the expected class — the role mask ALONE is
    /// insufficient (the same-PoS residue, e.g. `found` = find.v / found.v).
    Ambiguous(usize),
    /// No sense of the expected class fills this role.
    None,
}

/// Apply a role mask: keep only senses whose PoS matches the role's class, then
/// report whether the superposition collapsed to a UNIQUE frequency-centroid.
fn collapse(senses: &[Sense], role: Role) -> Collapse<'_> {
    let want = role.expected_pos();
    let mut hits = senses.iter().filter(|s| s.pos == want);
    match (hits.next(), hits.next()) {
        (Some(s), None) => Collapse::Unique(s.lemma.as_str(), s.lem_rank),
        (None, _) => Collapse::None,
        (Some(_), Some(_)) => Collapse::Ambiguous(senses.iter().filter(|s| s.pos == want).count()),
    }
}

/// Parse `word_forms.csv` into `lowercased surface → Vec<Sense>` (identical
/// tuples deduplicated). Hand-split on ',' — deepnsm is std-only, zero-dep.
fn parse_forms(csv: &str) -> HashMap<String, Vec<Sense>> {
    let mut map: HashMap<String, Vec<Sense>> = HashMap::new();
    for line in csv.lines().skip(1) {
        // lemRank,lemma,PoS,lemFreq,wordFreq,word
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            continue;
        }
        let Ok(lem_rank) = f[0].parse::<u32>() else {
            continue;
        };
        let Some(pos) = f[2].chars().next() else {
            continue;
        };
        let sense = Sense {
            lemma: f[1].to_string(),
            pos,
            lem_rank,
        };
        let bucket = map.entry(f[5].to_ascii_lowercase()).or_default();
        if !bucket.contains(&sense) {
            bucket.push(sense);
        }
    }
    map
}

/// One line of a demonstration: how a role mask collapses the superposition.
fn describe(senses: &[Sense], role: Role) -> String {
    let lab = role.label();
    let pos = role.expected_pos();
    let class = if pos == 'v' { "verbal" } else { "nominal" };
    match collapse(senses, role) {
        Collapse::Unique(lemma, rank) => {
            format!("{lab} → {class:<7} → {lemma}.{pos} → centroid r{rank}")
        }
        Collapse::Ambiguous(n) => {
            format!("{lab} → {class:<7} → AMBIGUOUS: {n} senses of class '{pos}' (mask alone fails)")
        }
        Collapse::None => format!("{lab} → {class:<7} → (no {class} reading present)"),
    }
}

/// Demonstrate + KILL-GATE one cross-PoS homograph. The P-slot must collapse to
/// a unique verbal centroid, the S/O-slot to a unique nominal centroid, and the
/// two addresses must DIFFER — else the superposition did not split under role
/// masking and the probe is falsified.
fn demonstrate(map: &HashMap<String, Vec<Sense>>, surface: &str) {
    let Some(senses) = map.get(surface) else {
        println!(
            "  ✗ KILL: {surface} did not collapse to two distinct unique centroids under role masking"
        );
        std::process::exit(1);
    };

    let bundle: Vec<String> = senses
        .iter()
        .map(|s| format!("({}, {}, r{})", s.lemma, s.pos, s.lem_rank))
        .collect();
    println!("  '{surface}'  superposition = {{ {} }}", bundle.join(", "));
    for role in [Role::Subject, Role::Predicate, Role::Object] {
        println!("      {}", describe(senses, role));
    }

    // KILL GATE: P-collapse and S/O-collapse each UNIQUE and DIFFERENT.
    let p = collapse(senses, Role::Predicate);
    let so = collapse(senses, Role::Subject); // O is identical (also nominal)
    if let (Collapse::Unique(_, pr), Collapse::Unique(_, sr)) = (p, so) {
        if pr != sr {
            println!("      ✓ same surface, DISTINCT centroids by role: P r{pr} ≠ S/O r{sr}\n");
            return;
        }
    }
    println!(
        "  ✗ KILL: {surface} did not collapse to two distinct unique centroids under role masking"
    );
    std::process::exit(1);
}

fn main() {
    let csv = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/word_frequency/word_forms.csv"
    ))
    .expect("read word_forms.csv");
    let map = parse_forms(&csv);

    println!("── HOMOGRAPH COLLAPSE · the SPO 2³ role mask as a superposition collapse operator ──");
    println!("  A surface word is a SUPERPOSITION of (lemma, PoS) readings. Fixing the SPO slot it");
    println!("  fills is the COLLAPSE OPERATOR: role → PoS class → one lemma → one lemRank centroid.");
    println!(
        "  mask:  {} & {} → nominal   |   {} → verbal\n",
        Role::Subject.label(),
        Role::Object.label(),
        Role::Predicate.label()
    );

    // ── census: how much of the vocabulary is a collapsible superposition? ──
    let distinct = map.len();
    let mut ambiguous = 0usize;
    let mut cross_pos = 0usize;
    let mut verb_noun = 0usize;
    let mut collapsible = 0usize;
    for senses in map.values() {
        if senses.len() > 1 {
            ambiguous += 1;
            let mut letters: Vec<char> = senses.iter().map(|s| s.pos).collect();
            letters.sort_unstable();
            letters.dedup();
            if letters.len() >= 2 {
                cross_pos += 1;
            }
        }
        let has_v = senses.iter().any(|s| s.pos == 'v');
        let has_n = senses.iter().any(|s| s.pos == 'n');
        if has_v && has_n {
            verb_noun += 1;
            if let (Collapse::Unique(_, pr), Collapse::Unique(_, sr)) = (
                collapse(senses, Role::Predicate),
                collapse(senses, Role::Subject),
            ) {
                if pr != sr {
                    collapsible += 1;
                }
            }
        }
    }
    let same_pos = ambiguous - cross_pos;
    let pct = 100.0 * ambiguous as f64 / distinct as f64;
    let collaps_pct = 100.0 * collapsible as f64 / verb_noun as f64;

    println!("── AMBIGUITY CENSUS · word_frequency/word_forms.csv ──");
    println!("  distinct surface forms                       : {distinct}");
    println!("  ambiguous (>1 (lemma,PoS) sense)             : {ambiguous}  ({pct:.2}% of surfaces)");
    println!("    ├─ CROSS-PoS (senses span ≥2 PoS letters)  : {cross_pos}   ← collapsible by a role mask");
    println!("    └─ same-PoS residue (e.g. found=find.v/found.v): {same_pos}   ← role mask ALONE cannot split");
    println!("  verb∩noun homographs (≥1 verb AND ≥1 noun)   : {verb_noun}");
    println!("    └─ collapse to distinct v/n centroids under P-vs-S/O : {collapsible}  ({collaps_pct:.0}%)\n");

    // ── the three cross-PoS demonstrations (also the kill gate) ──
    println!("── DEMONSTRATION · same surface, different SPO slot → different centroid ──");
    for surface in ["thought", "left", "means"] {
        demonstrate(&map, surface);
    }

    // ── synthesis + honest boundary ──
    println!("── SYNTHESIS ──");
    println!("  Each surface word is held as the SUPERPOSITION of its readings. The SPO 2³ role");
    println!("  mask is the collapse operator: the slot a token fills fixes one PoS class → one");
    println!("  lemma → one lemRank, a unique frequency-centroid. 'thought' addresses r53 as a");
    println!("  Predicate but r692 as a Subject/Object — same string, different centroid, chosen");
    println!("  by role alone; likewise left (r152/r4559) and means (r130/r1392). Across COCA,");
    println!("  {collapsible} verb∩noun homographs collapse this way.\n");
    println!("  HONEST BOUNDARY: the S/O→nominal, P→verbal mapping is a SIMPLIFICATION. Real");
    println!("  syntactic role assignment is the parser's job — a full grammar decides which slot");
    println!("  a token actually fills, and resolves the same-PoS residue (found=find.v/found.v)");
    println!("  this mask cannot split. This probe demonstrates the COLLAPSE MECHANISM — a role");
    println!("  mask disambiguating a homograph to a unique centroid — not full parsing. It shows");
    println!("  the 2³ SPO mask acting as a homograph-collapse operator on real surface-form data.");
}
