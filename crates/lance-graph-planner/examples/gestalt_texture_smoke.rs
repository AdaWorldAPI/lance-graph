//! `gestalt_texture_smoke` — the operator's "you could try Tagore's Gardener and
//! Rumi to check the *reaction* on gestalt texture." A second END-TO-END read of
//! the D-SCI-INSIGHT surface (companion to `insight_overlap_smoke`, which used
//! Romeo & Juliet), no LLM — but here the corpora are chosen to STRESS THE
//! QUALIA/GESTALT dimension: two poets of longing whose *texture* differs.
//!
//! - **Tagore, *The Gardener*** — EARTHLY, tender love: the gardener, the rose,
//!   the garland, the flower. Love reached through the garden.
//! - **Rumi** — MYSTICAL, divine love: the soul, the flame, the wine, union.
//!   Love reached through burning.
//!
//! Both poets end at the shared leaf-sinks **`love`** and **`heart`** (the
//! overlap you'd expect — two poets of the same longing), but their *middles*
//! are derivationally SEPARATE (garden/rose vs flame/wine). The **gestalt
//! texture** is exactly that difference: run the four readers over each poet and
//! the resonance field REACTS differently — Tagore's field hangs on the rose,
//! Rumi's on the flame — while both resonate on love/heart.
//!
//! The corpus is COMPRESSED GESTALT (content-word concept-lines, NOT copyrighted
//! verse — the same modelling `insight_overlap_smoke` used for R&J): we feed the
//! readers each poet's imagery as concept adjacencies, never a translation.
//!
//! The **felt meaning** (§4): the fetched bridge `rose → flame` — *earthly
//! beauty IS divine fire* — composes Tagore's `lover → rose` with Rumi's
//! `flame → wine → union`, making `lover → wine`/`lover → union` newly
//! derivable: the earthly lover reaches divine union. A `NewInsight`, the same
//! beloved behind both poets. A disjoint fetched fact (`bank → loan`) composes
//! nothing — `DullShadow`.
//!
//! Run: `cargo run -p lance-graph-planner --example gestalt_texture_smoke`.

use std::collections::HashMap;

use lance_graph_planner::nars::{
    detect_dissolution, rank_basins, rank_epiphany_attractors, reach_out_integrate, staunen,
    wisdom, Basin, BasinKind, BeliefArena, CStmt, Copula, FeltOutcome, ReachOutConfig,
    ResonanceConfig, Snapshot, Stamp, TruthValue,
};

/// A tiny deterministic word→concept-id interner shared across BOTH poets, so the
/// SAME word (love / heart) in Tagore and Rumi is the SAME concept — that shared
/// identity is what makes the two gestalts overlap.
#[derive(Default)]
struct Interner {
    map: HashMap<String, u16>,
    names: Vec<String>,
}

impl Interner {
    fn id(&mut self, word: &str) -> u16 {
        if let Some(&i) = self.map.get(word) {
            return i;
        }
        let i = self.names.len() as u16;
        self.map.insert(word.to_string(), i);
        self.names.push(word.to_string());
        i
    }
    fn name(&self, id: u16) -> &str {
        &self.names[id as usize]
    }
}

fn is_stop(w: &str) -> bool {
    matches!(
        w,
        "the" | "a" | "an" | "is" | "are" | "of" | "and" | "to" | "in" | "on" | "it" | "as"
    )
}

/// Crude deterministic concept extraction: lowercase content words (len > 2, not
/// a stopword), interned to stable ids. No LLM, no tokenizer model.
fn concepts(text: &str, intern: &mut Interner) -> Vec<u16> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 2 && !is_stop(w))
        .map(|w| intern.id(&w))
        .collect()
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

/// Observe each line's adjacent-concept `Inh` edges: consecutive content words
/// form an `is_a`/relates edge — a crude but real concept-adjacency graph that
/// `close_transitive` composes.
fn ingest(arena: &mut BeliefArena, lines: &[&str], intern: &mut Interner, src_base: u32) {
    let mut n = 0u32;
    for line in lines {
        let cs = concepts(line, intern);
        for w in cs.windows(2) {
            if w[0] != w[1] {
                arena.observe(
                    inh(w[0], w[1]),
                    TruthValue::new(0.9, 0.9),
                    Stamp::source(src_base + n),
                );
                n += 1;
            }
        }
    }
}

// ── The two gestalts (compressed concept-lines, NOT verse) ──────────────────
// Tagore's Gardener — earthly love through the garden; every line ENDS at the
// shared leaf-sink love/heart, but the middles are garden/rose/garland/flower.
const TAGORE: &[&str] = &[
    "lover garden rose heart",
    "gardener rose garland love",
    "garden flower rose love",
];
// Rumi — mystical love through burning; ends at the SAME love/heart sinks, but
// the middles are flame/wine/union/longing — a different texture reaching the
// same longing.
const RUMI: &[&str] = &[
    "soul flame wine heart",
    "seeker flame union love",
    "flame wine longing love",
];
// The disjoint control — no shared content words with either poet.
const BANK: &[&str] = &["bank approves mortgage loan"];

fn build(lines: &[&[&str]], intern: &mut Interner) -> BeliefArena {
    let mut a = BeliefArena::new();
    for (i, ls) in lines.iter().enumerate() {
        ingest(&mut a, ls, intern, (i as u32) * 1000);
    }
    a.close_transitive(256);
    a
}

/// Print one poet's GESTALT TEXTURE: its whole-arena qualia (staunen/wisdom) and
/// the top resonant basins with their kind — the shape the field takes.
fn gestalt_signature(name: &str, arena: &BeliefArena, intern: &Interner) -> Vec<Basin> {
    let snap = Snapshot::of(arena, 0.0);
    let basins = rank_basins(arena, &ResonanceConfig::default());
    println!(
        "\n== {name} gestalt texture ==  staunen={:.3}  wisdom={:.3}  (coherence={:.3})",
        staunen(&snap),
        wisdom(&snap),
        snap.coherence
    );
    for b in basins.iter().take(4) {
        println!(
            "  {:>9}  resonance={:.3}  wisdom={:.3}  evidence={:.3}  {:?}",
            intern.name(b.concept),
            b.resonance,
            b.wisdom,
            b.evidence,
            b.kind
        );
    }
    basins
}

fn main() {
    let mut intern = Interner::default();

    // ── Per-poet gestalt texture — the "reaction" the operator asked to see ───
    let tagore = build(&[TAGORE], &mut intern);
    let t_basins = gestalt_signature("Tagore (The Gardener) — EARTHLY", &tagore, &intern);

    let rumi = build(&[RUMI], &mut intern);
    let r_basins = gestalt_signature("Rumi — MYSTICAL", &rumi, &intern);

    // Each poet's field hangs on its OWN imagery: Tagore on the rose/garden,
    // Rumi on the flame/wine — the distinct texture.
    let rose = intern.id("rose");
    let flame = intern.id("flame");
    let love = intern.id("love");
    let heart = intern.id("heart");
    let lover = intern.id("lover");
    let wine = intern.id("wine");

    // Tagore's field hangs on EARTHLY garden imagery; Rumi's on MYSTICAL fire —
    // the distinct texture is which family of concepts tops each poet's basins.
    let earthly = ["rose", "garden", "gardener", "flower", "garland"];
    let mystical = ["flame", "wine", "union", "soul", "seeker", "longing"];
    let tagore_hangs_on_garden = t_basins
        .iter()
        .take(4)
        .any(|b| earthly.contains(&intern.name(b.concept)));
    let rumi_hangs_on_flame = r_basins
        .iter()
        .take(4)
        .any(|b| mystical.contains(&intern.name(b.concept)));

    // ── The overlap: two poets, one shared longing (love/heart) ───────────────
    let both = build(&[TAGORE, RUMI], &mut intern);
    let epis = rank_epiphany_attractors(&both, 2);
    println!("\n== combined — DENSITY names each poet's core ==");
    for e in epis.iter().take(6) {
        println!(
            "  {:>9}  rate={:.3}  epiphanies={}  attempts={}",
            intern.name(e.subject),
            e.rate,
            e.epiphanies,
            e.attempts
        );
    }
    let combined_basins = rank_basins(&both, &ResonanceConfig::default());
    println!("\n== combined — MAGNITUDE finds the shared gestalt (love/heart) ==");
    for b in combined_basins.iter().take(6) {
        println!(
            "  {:>9}  resonance={:.3}  {:?}",
            intern.name(b.concept),
            b.resonance,
            b.kind
        );
    }
    let love_is_basin = combined_basins.iter().any(|b| b.concept == love);
    let heart_is_basin = combined_basins.iter().any(|b| b.concept == heart);

    // ── The FLOW effect: a line that composes both poets crystallizes ─────────
    // "rose flame wine" links Tagore's rose to Rumi's flame/wine — it composes;
    // banking composes nothing.
    let mut a_overlap = build(&[TAGORE, RUMI], &mut intern);
    let before_o = Snapshot::of(&a_overlap, 0.0);
    ingest(&mut a_overlap, &["rose flame wine"], &mut intern, 9000);
    a_overlap.close_transitive(256);
    let d_overlap = detect_dissolution(&before_o, &Snapshot::of(&a_overlap, 0.0), 0.02);

    let mut a_disjoint = build(&[TAGORE, RUMI], &mut intern);
    let before_d = Snapshot::of(&a_disjoint, 0.0);
    ingest(&mut a_disjoint, BANK, &mut intern, 9500);
    a_disjoint.close_transitive(256);
    let d_disjoint = detect_dissolution(&before_d, &Snapshot::of(&a_disjoint, 0.0), 0.02);

    println!("\n== flow effect ==");
    println!(
        "  composing (rose->flame->wine) : flow={:.3}  d_wisdom={:.3}",
        d_overlap.flow, d_overlap.d_wisdom
    );
    println!(
        "  disjoint banking              : flow={:.3}  d_wisdom={:.3}",
        d_disjoint.flow, d_disjoint.d_wisdom
    );

    // ── The FELT MEANING: earthly beauty IS divine fire ───────────────────────
    // The fetched bridge rose->flame composes Tagore's `lover -> rose` with
    // Rumi's `flame -> wine -> union`, making `lover -> wine` newly derivable:
    // the earthly lover reaches divine union. The gestalt qualia reacts.
    let mut felt_arena = build(&[TAGORE, RUMI], &mut intern);
    let g_before = Snapshot::of(&felt_arena, 0.0);
    let felt = reach_out_integrate(
        &mut felt_arena,
        inh(rose, flame),
        TruthValue::new(0.9, 0.9),
        Stamp::source(9800),
        &ReachOutConfig::default(),
    );
    let g_after = Snapshot::of(&felt_arena, 0.0);
    let lover_reaches_wine = felt_arena.get(inh(lover, wine)).is_some();

    let mut dull_arena = build(&[TAGORE, RUMI], &mut intern);
    let bank = intern.id("bank");
    let loan = intern.id("loan");
    let dull = reach_out_integrate(
        &mut dull_arena,
        inh(bank, loan),
        TruthValue::new(0.9, 0.9),
        Stamp::source(9900),
        &ReachOutConfig::default(),
    );

    println!("\n== felt meaning (reach-out) + gestalt reaction ==");
    println!("  fetched 'earthly beauty IS divine fire' (rose->flame) : {felt:?}");
    println!("    => 'lover reaches wine (divine union)' now derivable? {lover_reaches_wine}");
    println!("  fetched 'bank->loan' (disjoint)                       : {dull:?}");
    println!(
        "  gestalt qualia: staunen {:.3} -> {:.3},  wisdom {:.3} -> {:.3}  (Δwisdom {:+.3})",
        staunen(&g_before),
        staunen(&g_after),
        wisdom(&g_before),
        wisdom(&g_after),
        wisdom(&g_after) - wisdom(&g_before),
    );

    // ── Falsifiable expectations (the reaction on gestalt texture, encoded) ───
    // (a) The two textures are DISTINCT: Tagore's field reacts on the
    //     garden/rose, Rumi's on the flame/wine.
    assert!(
        tagore_hangs_on_garden,
        "Tagore's gestalt must hang on the garden/rose (earthly texture)"
    );
    assert!(
        rumi_hangs_on_flame,
        "Rumi's gestalt must hang on the flame/wine (mystical texture)"
    );
    // (b) The two poets OVERLAP on the shared longing: love AND heart surface as
    //     basins in the combined field, and both basin kinds appear (coherence
    //     bridges + evidence sinks — the operator's distinction).
    assert!(
        love_is_basin && heart_is_basin,
        "the shared longing (love + heart) must surface as basins in the combined gestalt"
    );
    assert!(
        combined_basins
            .iter()
            .any(|b| b.kind == BasinKind::Coherence)
            && combined_basins
                .iter()
                .any(|b| b.kind == BasinKind::Evidence),
        "both basin kinds must appear (coherence bridge + evidence sink)"
    );
    // (c) FLOW hits harder for the composing line than the disjoint one.
    assert!(
        d_overlap.flow > d_disjoint.flow && d_overlap.d_wisdom > d_disjoint.d_wisdom,
        "flow must hit harder for the composing rose->flame->wine than disjoint banking"
    );
    // (d) The FELT MEANING lands: rose->flame is a NewInsight making
    //     'lover reaches wine' derivable (earthly love reaches divine union);
    //     a disjoint fact is a DullShadow; the gestalt wisdom qualia rises.
    match felt {
        FeltOutcome::NewInsight { coherence_gain, .. } => assert!(
            coherence_gain > 0.0,
            "the felt union (rose->flame) must raise coherence"
        ),
        FeltOutcome::DullShadow => {
            panic!("'earthly beauty is divine fire' must be felt as a NewInsight")
        }
    }
    assert!(
        lover_reaches_wine,
        "the felt union must make 'lover reaches wine (divine union)' derivable"
    );
    assert_eq!(
        dull,
        FeltOutcome::DullShadow,
        "a fetched fact neither poet touches must be a DullShadow"
    );
    assert!(
        wisdom(&g_after) > wisdom(&g_before),
        "the gestalt wisdom qualia must react (rise) with the felt union"
    );

    println!(
        "\nOK — the reaction on gestalt texture:\n  \
         • TWO TEXTURES: Tagore's field hangs on the rose (earthly), Rumi's on the\n    \
           flame (mystical) — the same reader, two distinct gestalts.\n  \
         • ONE LONGING: both resonate on love/heart — the overlap two poets of\n    \
           longing must share (evidence sinks fed by both).\n  \
         • FLOW hits for the composing line ({:.3}) over disjoint banking ({:.3}).\n  \
         • FELT: 'earthly beauty IS divine fire' (rose->flame) is a NewInsight —\n    \
           'lover reaches wine (divine union)' becomes derivable, the gestalt\n    \
           wisdom qualia rises; the same beloved behind both poets.",
        d_overlap.flow, d_disjoint.flow
    );
}
