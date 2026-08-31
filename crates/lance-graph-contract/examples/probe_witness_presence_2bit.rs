//! PROBE — 2-bit presence masks as the cheap sibling of the 24×i4 witness
//! tenant (fox / mammal / wombat).
//!
//! Question under test: can a derived presence projection of the A9 register
//! answer the triptych's shape-2 questions (downstream inheritance /
//! reachability) at free-AND cost, pruning before the nibble tier — and does
//! the SECOND bit (orientation) discriminate beyond plain presence?
//!
//! # Semantics correction this probe encodes (read before citing)
//!
//! In the A9 register the nibble SIGN is **orientation** (− = before /
//! antecedent, + = after / consequent), NEVER valence ("Loci, not
//! magnitudes", operator-locked). Support-vs-contradiction lives in WHICH
//! locus is bound (Quorum=14, Contradiction=15, SupportedBy=9, Supports=10).
//! So the faithful 2-bit presence is:
//!
//!   before_mask  = { slot | nibble < 0 }   (anchored into what came earlier)
//!   after_mask   = { slot | nibble > 0 }   (bound to what comes later)
//!
//! and valence-tier questions are constant locus-group STENCILS intersected
//! onto presence — free ANDs, no third mask.
//!
//! # What is fixture-shaped vs transferable
//!
//! Density numbers below are FIXTURE-shaped (no production witness corpus
//! exists in-repo). The transferable measurements are mechanism-level: the
//! prune arithmetic, the false-positive accounting, the 1-bit-vs-2-bit
//! discrimination counts, and the fuse() plumbing through the shipped
//! `EvidenceMask` operators (difference / is_subset_of — D-MAR-1).
//!
//! Run: `cargo run -p lance-graph-contract --example probe_witness_presence_2bit`

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, WITNESS_LOCI};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::fusion::{fuse, FusionOutcome};
use lance_graph_contract::revision::{
    CodebookId, EvidenceMask, GrammarId, HorizonId, InterpretiveHorizon, LanguageId, LensId,
    QuestionId,
};

/// The 2-bit presence projection: (before_mask, after_mask). Derived, never
/// stored — cast-reproducible from the register, so under the zero-copy law
/// its only legitimate persistent home is the (separate) stencil arena.
fn presence_2bit(w: &CausalWitnessFacet) -> (WideFieldMask, WideFieldMask) {
    let mut before: Vec<u8> = Vec::new();
    let mut after: Vec<u8> = Vec::new();
    for slot in 0..WITNESS_LOCI {
        let v = w.get(slot);
        if v < 0 {
            before.push(slot as u8);
        } else if v > 0 {
            after.push(slot as u8);
        }
    }
    (
        WideFieldMask::from_positions(&before),
        WideFieldMask::from_positions(&after),
    )
}

/// 1-bit presence = before ∪ after (what plain presence would keep).
fn presence_1bit(before: &WideFieldMask, after: &WideFieldMask) -> WideFieldMask {
    before.union(after)
}

/// Orphan-rule-clean newtype so fuse() can run on WideFieldMask-backed
/// evidence without touching src/. A real `impl EvidenceMask for
/// WideFieldMask` in revision.rs is one small additive change, unlocked by
/// D-MAR-1 (the trait needs difference + is_subset_of) — recorded, not done.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Presence(WideFieldMask);

impl EvidenceMask for Presence {
    fn empty() -> Self {
        Presence(WideFieldMask::EMPTY)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn union(&self, other: &Self) -> Self {
        Presence(self.0.union(&other.0))
    }
    fn intersection(&self, other: &Self) -> Self {
        Presence(self.0.intersect(&other.0))
    }
    fn difference(&self, other: &Self) -> Self {
        Presence(self.0.difference(&other.0))
    }
    fn is_subset_of(&self, other: &Self) -> bool {
        self.0.is_subset_of(&other.0)
    }
}

fn horizon(
    id: u64,
    claims: Presence,
    independent: Presence,
    inherited: Presence,
) -> InterpretiveHorizon<u64, Presence> {
    InterpretiveHorizon {
        id: HorizonId(id),
        awareness: 0,
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: claims,
        independent_roots: independent,
        inherited_roots: inherited,
        unresolved_tension: Presence::empty(),
        revision_index: 0,
    }
}

fn mask_of(slots: &[Locus]) -> WideFieldMask {
    let v: Vec<u8> = slots.iter().map(|&l| l as u8).collect();
    WideFieldMask::from_positions(&v)
}

fn main() {
    // ── Fixture stream ─────────────────────────────────────────────────────
    // pos 0 animal · 1 mammal · 2 bird · 3 fox · 4 wombat
    // pos 5 obs-fox-fur · 6 claim-fox-is-bird · 7 obs-wombat-pouch
    //
    // Taxonomy anchors point BACKWARD (parents occur earlier → before-mask);
    // observations and preserved dissent occur LATER (→ after-mask).
    let mammal = CausalWitnessFacet::ZERO
        .with(Locus::BasinAnchor, -1) // → animal
        .with(Locus::Supports, 2); // ↑ supports fox (later)
    let fox = CausalWitnessFacet::ZERO
        .with(Locus::BasinAnchor, -2) // → mammal
        .with(Locus::SupportedBy, 2) // ← obs-fox-fur (later)
        .with(Locus::Quorum, 1) // agreeing peer: wombat
        .with(Locus::Contradiction, 3); // preserved dissent: claim-fox-is-bird
    let wombat = CausalWitnessFacet::ZERO
        .with(Locus::BasinAnchor, -3) // → mammal
        .with(Locus::SupportedBy, 3) // ← obs-wombat-pouch
        .with(Locus::Quorum, -1); // agreeing peer: fox
                                  // A corrupted fox: inheritance chain broken (BasinAnchor dropped).
    let fox_corrupt = CausalWitnessFacet::ZERO
        .with(Locus::SupportedBy, 2)
        .with(Locus::Quorum, 1);

    // Spread population: 64 deterministic row-witnesses with varied binding
    // (fixture-shaped density; rules, not randomness).
    let mut population: Vec<CausalWitnessFacet> = vec![mammal, fox, wombat];
    // Sign policy mirrors the register's documented semantics: Kausal,
    // BasinAnchor and Antecedent are orientation-PINNED (causes and parents
    // lie backward, always −); Temporal, Quorum and SupportedBy are
    // orientation-FREE (peers, time references and evidence occur on both
    // sides — fox's Quorum is +1 while wombat's is −1 in the core above).
    //
    // GENERATOR PROVENANCE — two failed designs, both caught by this probe's
    // own can-fire assert (§3), kept here so they are not re-invented:
    //   1. every locus sign fixed → sign is a function of locus → the 2nd
    //      bit is mathematically redundant (0/210 pairs split);
    //   2. sign from (i/k)%2 on the SAME counter driving membership (i%k)
    //      → sign arithmetically entangled with membership (e.g. positive
    //      Temporal ⟺ i≡0 mod 4 ⟺ BasinAnchor bound), so within every
    //      1-bit-confusable class the sign was constant — again 0/210.
    // The orientation axis must be INDEPENDENT of the membership axes: a
    // mixed hash of (i, locus) supplies it, deterministic and divisor-free.
    fn free_sign(i: u32, locus: Locus) -> i8 {
        let mut z = (i as u64) ^ ((locus as u64) << 32) ^ 0x9E37_79B9_7F4A_7C15;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        if (z >> 63) == 0 {
            1
        } else {
            -1
        }
    }
    for i in 0..61u32 {
        let mut w = CausalWitnessFacet::ZERO.with(Locus::Kausal, -(((i % 7) + 1) as i8));
        if i % 2 == 0 {
            let mag = ((i % 5) + 1) as i8;
            w = w.with(Locus::Temporal, mag * free_sign(i, Locus::Temporal));
        }
        if i % 3 == 0 {
            w = w.with(Locus::Antecedent, -1); // pinned backward
        }
        if i % 4 == 0 {
            w = w.with(Locus::BasinAnchor, -(((i % 6) + 1) as i8)); // pinned backward
        }
        if i % 5 == 0 {
            let mag = ((i % 4) + 1) as i8;
            w = w.with(Locus::SupportedBy, mag * free_sign(i, Locus::SupportedBy));
        }
        if i % 6 == 0 {
            w = w.with(Locus::Quorum, free_sign(i, Locus::Quorum));
        }
        if i % 16 == 0 {
            w = w.with(Locus::Contradiction, ((i % 3) + 1) as i8);
        }
        population.push(w);
    }
    let n = population.len();

    // ── 1. Density (fixture-shaped) ────────────────────────────────────────
    let mut sum_1bit = 0u32;
    let mut sum_before = 0u32;
    let mut sum_after = 0u32;
    for w in &population {
        let (b, a) = presence_2bit(w);
        sum_before += b.count();
        sum_after += a.count();
        sum_1bit += presence_1bit(&b, &a).count();
    }
    println!("== 1. presence density (N={n}, 24 loci, FIXTURE-shaped) ==");
    println!(
        "mean bound/24: {:.2}  (before {:.2}, after {:.2}) → density {:.1}%",
        sum_1bit as f64 / n as f64,
        sum_before as f64 / n as f64,
        sum_after as f64 / n as f64,
        100.0 * sum_1bit as f64 / (n as f64 * 24.0)
    );

    // ── 2. Prune: "anchored into a taxonomy basin, antecedent-oriented" ────
    // Presence-tier stencil: BasinAnchor bound in the BEFORE mask. Nibble
    // tier then confirms the exact offset (here: resolves within the stream).
    let anchor_stencil = mask_of(&[Locus::BasinAnchor]);
    let mut presence_pass = 0usize;
    let mut nibble_pass = 0usize;
    for w in &population {
        let (b, _a) = presence_2bit(w);
        if !b.intersect(&anchor_stencil).is_empty() {
            presence_pass += 1;
            // nibble tier (only survivors pay this):
            if w.at(Locus::BasinAnchor) < 0 {
                nibble_pass += 1;
            }
        }
    }
    let pruned = n - presence_pass;
    println!("\n== 2. ancestor-anchor sweep (presence tier prunes, nibble tier decides) ==");
    println!(
        "presence-tier survivors {presence_pass}/{n} → pruned {:.1}% before any nibble read",
        100.0 * pruned as f64 / n as f64
    );
    println!(
        "nibble-tier confirms {nibble_pass}/{presence_pass} survivors (false-positive rate {:.1}%)",
        100.0 * (presence_pass - nibble_pass) as f64 / presence_pass.max(1) as f64
    );
    // Falsifiability pair: the prune must discriminate on this fixture.
    assert!(
        pruned > 0,
        "can-fire: the presence tier must prune something"
    );
    assert!(
        presence_pass > 0,
        "can-stay-silent: it must not prune everything"
    );

    // ── 3. Second-bit discrimination: 1-bit-equal pairs the 2-bit splits ───
    let mut one_bit_confusable = 0usize;
    let mut split_by_orientation = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let (bi, ai) = presence_2bit(&population[i]);
            let (bj, aj) = presence_2bit(&population[j]);
            if presence_1bit(&bi, &ai) == presence_1bit(&bj, &aj) {
                one_bit_confusable += 1;
                if bi != bj || ai != aj {
                    split_by_orientation += 1;
                }
            }
        }
    }
    println!("\n== 3. what the SECOND bit buys (orientation) ==");
    println!(
        "pairs identical under 1-bit presence: {one_bit_confusable}; of those, split by the 2nd bit: {split_by_orientation}"
    );
    // Honest both ways: the 2nd bit must discriminate somewhere AND not everywhere.
    assert!(
        split_by_orientation > 0,
        "can-fire: orientation must add information"
    );
    assert!(
        split_by_orientation < one_bit_confusable || one_bit_confusable == 0,
        "can-stay-silent: some 1-bit-equal pairs legitimately stay equal"
    );

    // ── 4. Valence via constant stencils (no third mask) ───────────────────
    let contra_stencil = mask_of(&[Locus::Contradiction]);
    let (fox_b, fox_a) = presence_2bit(&fox);
    let (wom_b, wom_a) = presence_2bit(&wombat);
    let fox_dissent = !presence_1bit(&fox_b, &fox_a)
        .intersect(&contra_stencil)
        .is_empty();
    let wom_dissent = !presence_1bit(&wom_b, &wom_a)
        .intersect(&contra_stencil)
        .is_empty();
    println!("\n== 4. valence as locus-group stencil ∧ presence ==");
    println!("fox carries preserved dissent: {fox_dissent}; wombat: {wom_dissent}");
    assert!(
        fox_dissent && !wom_dissent,
        "stencil discriminates fox from wombat"
    );

    // ── 5. fuse() on presence-derived masks (shape-2 plumbing) ─────────────
    // Thesis "fox is a mammal": independently grounded (SupportedBy) AND
    // inherited (BasinAnchor). Antithesis "fox is a bird": inherited only.
    let fox_supported =
        Presence(presence_1bit(&fox_b, &fox_a).intersect(&mask_of(&[Locus::SupportedBy])));
    let fox_anchored = Presence(presence_1bit(&fox_b, &fox_a).intersect(&anchor_stencil));
    let thesis = horizon(
        1,
        Presence(mask_of(&[Locus::BasinAnchor])),
        fox_supported.clone(),
        fox_anchored.clone(),
    );
    let antithesis = horizon(
        2,
        Presence(mask_of(&[Locus::Contradiction])),
        Presence::empty(), // no independent grounding for fox-is-bird
        Presence(mask_of(&[Locus::Contradiction])),
    );
    let contradiction = Presence(contra_stencil.clone());
    let receipt = fuse(&thesis, &antithesis, &contradiction);
    println!("\n== 5. fuse() over WideFieldMask-backed presence (D-MAR-1 operators live) ==");
    println!("fox-is-mammal vs fox-is-bird → {:?}", receipt.outcome);
    assert_eq!(receipt.outcome, FusionOutcome::ThesisSurvives);

    // Sanity that the shipped operators, not stand-ins, carried the run:
    assert!(fox_supported.is_subset_of(&Presence(presence_1bit(&fox_b, &fox_a))));
    assert!(!Presence(fox_b.clone()).is_subset_of(&Presence(fox_a.clone())));

    // ── 6. Containment falsifier on the inheritance chain ──────────────────
    // The fixture's rule: a taxonomy-anchored child is before-bound at
    // BasinAnchor. The corrupted fox breaks it — the check must FIRE.
    let (cor_b, _) = presence_2bit(&fox_corrupt);
    let intact = !fox_b.intersect(&anchor_stencil).is_empty()
        && !wom_b.intersect(&anchor_stencil).is_empty();
    let corrupt_detected = cor_b.intersect(&anchor_stencil).is_empty();
    println!("\n== 6. chain check: intact taxonomy passes, corrupted witness fires ==");
    println!("fox+wombat anchored: {intact}; corrupted fox detected: {corrupt_detected}");
    assert!(intact && corrupt_detected);

    println!("\nPROBE GREEN — all falsifiability pairs held.");
}
