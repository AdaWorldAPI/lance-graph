//! PROBE D-CSW-1 (leg 1) — standing-wave rung stability + escalation cascade
//! over an ordered text stream (`.claude/plans/causal-rung-standing-wave-v1.md`
//! §4/§6).
//!
//! ## The claim under test
//!
//! On a temporally-ordered stream of SPO facts, (a) the **standing wave** —
//! per-rung reading stability across the stream — separates CAUSAL pairs from
//! COINCIDENTAL pairs better than a single-cycle reading AND better than the
//! p64 baseline (CausalEdge64-style 3×u8 SPO cram = 16-fold rank aliasing, no
//! witness arm, no wave); and (b) the **escalation cascade** (§0.5.2: rungs
//! cost-ordered by selected-plane count; wave stability gates escalation, the
//! full counterfactual contrast runs only on survivors) prunes a registered
//! fraction of pairs below the top rung WITHOUT destroying separation.
//!
//! ## Registered gates (fixed BEFORE the first run — see consts below)
//!
//! PASS requires ALL of:
//!   1. `auc_wave >= auc_single + MARGIN_VS_SINGLE`
//!   2. `auc_wave >= auc_p64   + MARGIN_VS_P64`
//!   3. cascade: `pruned_fraction >= MIN_PRUNED` AND
//!      `auc_cascade >= auc_wave - MAX_AUC_DROP`
//! KILL: any gate fails (recorded either way; a failed cascade gate forces
//! re-ordering or dropping the cascade, per the plan's §6 kill column).
//!
//! ## Honest boundary (leg 1 of 2)
//!
//! - **Stream order stands in for `temporal.rs` versions.** Each tick is one
//!   sentence-commit yielding one `SpoTriple` — the plan's "text as a stream
//!   of temporal updates". Binding to real `temporal.rs`/Lance versions is
//!   **leg 2** (needs the lance-graph core build; protoc-gated, CI-side).
//!   Leg 1 tests the CLAIM's mechanism: stream order supplies orientation
//!   (instances are collected cause-BEFORE-effect only), so causal discovery
//!   here never searches over orderings.
//! - **Labels are PLANTED.** Ground-truth causality cannot be read off
//!   unlabeled text; planted structure is the standard falsifier (house
//!   precedent: `spo_markov_kg`'s demonstrator corpus). Causal pairs fire
//!   cause→effect with a consistent mechanism (stable predicate/object
//!   context, small jitter); coincidental pairs co-occur at a comparable
//!   rate but in RANDOM order with independently drawn contexts. The PIPELINE
//!   and the separation mechanism are the claim — external validity on wild
//!   corpora is NOT claimed by this leg.
//! - **Real shipped types, real table.** `deepnsm::spo::{SpoTriple,
//!   WordDistanceMatrix}` + `distance_per_role`; the 4096² table is built via
//!   `WordDistanceMatrix::build` from log-rank distance (the
//!   `E-FREQ-IS-COSINE-REPLACEMENT-1` grounding: COCA frequency rank distance
//!   as the cosine replacement) — NOT a `sin()` fixture.
//! - **SpoFacet correspondence.** A 12-bit rank maps to the larger-codebook
//!   palette pair `(basin, identity) = (rank >> 8, rank & 0xFF)` (the
//!   operator's "palette256² could even work for larger codebooks"); the
//!   probe mirrors the `awareness_facet::SpoFacet` register convention
//!   `rail k = (b[2k], b[2k+1])` (3 semantic + 3 witness pairs) and asserts
//!   the round-trip once. std-only mirror per deepnsm example convention
//!   (`causal_edge_v3_facet` precedent); the contract type is not depended on.
//!
//! ## Reading definition (registered)
//!
//! For a candidate pair (A→B), an INSTANCE is `A` as subject at tick t and
//! `B` as subject at t' ∈ (t, t+W]. The rung-m reading of instance i is the
//! **weakest-selected-plane** consistency of the CURRENT effect triple vs the
//! WITNESS (the pair's previous effect triple — the AriGraph arm):
//! `reading = 1 - max(selected per-role distances)/255` — a conjunction of
//! planes is as consistent as its least consistent plane. The top rung
//! (SPO=0b111, all three planes) IS the counterfactual contrast: semantic
//! triple vs witnessed history. Standing-wave stability per rung =
//! `mean(readings) - std(readings)`; single-cycle baseline = last reading
//! only; p64 baseline = ranks aliased to 8 bits (`rank % 256`) before the
//! table read, single-cycle, full-SPO only.
//!
//! ## v1 KILL (recorded) → v2 fixture revision (gates UNCHANGED)
//!
//! The first run (v1 fixture) tripped KILL gate 1 — but by a **fixture
//! ceiling**, not a mechanism verdict: ALL four arms scored AUC = 1.000
//! (wave/single/p64/cascade) and pruning = 0.000. Diagnosis: (a) the log-rank
//! table (`|ln i − ln j|·30`) is nearly flat over the upper ranks, so even
//! random contexts read ≥ 0.68 — above the escalation gate, no pruning, no
//! discrimination; (b) the planted mechanism was EXACT-repeat, and exact
//! repeats stay consistent under ANY deterministic aliasing, so the p64 cram
//! lost nothing. A fixture in which every method is perfect cannot falsify
//! their differences (the tesseract noise-fixture lesson). v2 changes the
//! FIXTURE only — every registered gate/margin/Θ is byte-identical to v1:
//!
//! 1. **Linear rank distance** `d = |i−j|/16` (frequency-rank distance as the
//!    cosine replacement, full 0..255 dynamic range).
//! 2. **25% disrupted causal contexts** — the effect event still fires but in
//!    a random context (messy text): a SINGLE reading is now unreliable; the
//!    wave has to earn its margin by averaging.
//! 3. **Residue-aligned coincidental context pools** (8 contexts per pair,
//!    spaced 512 = 2·256 ranks): distinct full-width words that ALIAS TO ONE
//!    id under the 3×u8 cram — the 16-fold-aliasing false-consistency failure
//!    made observable (the corruption-is-observable test discipline), while
//!    full-width sees 8 genuinely different contexts.
//!
//! ## v2 KILL (recorded) → v3 statistic correction (gates/margins UNCHANGED)
//!
//! v2 killed on gate 1 with REAL numbers: `auc_wave 0.715 < auc_single 0.875`
//! — while `auc_p64 = 0.375` (worse than chance: the residue-aligned pools
//! made the 16-fold-aliasing false-consistency observable — the full-width
//! half of the claim HELD). Diagnosis: the registered v1/v2 wave statistic
//! `stability = mean − std` is **refuted under bursty contamination** — the
//! 25% disrupted causal contexts explode the causal pairs' std, so the
//! variance penalty punishes exactly the pairs it should protect. A standing
//! wave is the component that RECURS; the robust estimator of recurrence
//! under ≤50% contamination is a **persistence fraction**, not mean−std
//! (breakdown-point argument). v3 replaces ONLY the statistic:
//! `persistence = fraction of readings ≥ STRONG_READING (0.8)` — one new
//! registered constant, same gates, same margins, same Θ, same fixture.
//! v2's kill stands on the board as the negative finding about mean−std.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example probe_dcsw1_standing_wave
//! ```

use deepnsm::spo::{SpoTriple, WordDistanceMatrix};

// ── Registered gates (fixed before first run; do not tune post-hoc) ─────────
const MARGIN_VS_SINGLE: f64 = 0.05;
const MARGIN_VS_P64: f64 = 0.05;
const MIN_PRUNED: f64 = 0.40;
const MAX_AUC_DROP: f64 = 0.02;
/// Escalation gate: a pair pays for the next rung only while its best
/// current-level stability clears this.
const THETA_ESCALATE: f64 = 0.5;
/// Precedence window W (ticks): effect must follow cause within W.
const WINDOW: usize = 5;

// ── Deterministic corpus (LCG; no wall-clock, no OS randomness) ─────────────
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG — deterministic across runs/platforms.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// Planted causal pairs: (cause rank, effect rank, mechanism predicate rank,
/// mechanism object rank). Ranks are COCA-style 12-bit ids spread over the
/// 4096 vocabulary; the mechanism context is CONSISTENT per pair (with small
/// jitter), which is what a real causal mechanism leaves in text.
const CAUSAL: [(u16, u16, u16, u16); 12] = [
    (40, 900, 210, 1300),
    (55, 950, 230, 1350),
    (70, 1000, 250, 1400),
    (85, 1050, 270, 1450),
    (100, 1100, 290, 1500),
    (115, 1150, 310, 1550),
    (130, 1200, 330, 1600),
    (145, 1250, 350, 1650),
    (160, 1300, 370, 1700),
    (175, 1350, 390, 1750),
    (190, 1400, 410, 1800),
    (205, 1450, 430, 1850),
];

/// Coincidental pairs: co-occur at a comparable rate, RANDOM order, contexts
/// drawn independently per occurrence (no mechanism).
const COINCIDENTAL: [(u16, u16); 12] = [
    (2000, 2900),
    (2050, 2950),
    (2100, 3000),
    (2150, 3050),
    (2200, 3100),
    (2250, 3150),
    (2300, 3200),
    (2350, 3250),
    (2400, 3300),
    (2450, 3350),
    (2500, 3400),
    (2550, 3450),
];

fn jitter(rank: u16, rng: &mut Lcg) -> u16 {
    // 10% chance: drift to a near rank neighbour (realistic lexical variation).
    if rng.chance(10) {
        let d = (rng.below(7) as i32) - 3;
        (rank as i32 + d).clamp(0, 4094) as u16
    } else {
        rank
    }
}

fn random_rank(rng: &mut Lcg) -> u16 {
    rng.below(4095) as u16
}

/// The ordered stream: each tick is one sentence-commit → one SpoTriple.
fn build_stream(rng: &mut Lcg) -> Vec<SpoTriple> {
    let mut stream = Vec::new();
    for _round in 0..40 {
        for &(cause, effect, mech_p, mech_o) in &CAUSAL {
            // cause fires in a random context...
            stream.push(SpoTriple::new(cause, random_rank(rng), random_rank(rng)));
            // distractor gap 0..2
            for _ in 0..rng.below(3) {
                stream.push(SpoTriple::new(
                    random_rank(rng),
                    random_rank(rng),
                    random_rank(rng),
                ));
            }
            // ...effect follows within the window with p=0.85, via its MECHANISM
            // — but 25% of firings are context-DISRUPTED (v2: the event still
            // happens, its stated context is noise), so one reading is
            // unreliable and the wave must average.
            if rng.chance(85) {
                let (p, o) = if rng.chance(25) {
                    (random_rank(rng), random_rank(rng))
                } else {
                    (jitter(mech_p, rng), jitter(mech_o, rng))
                };
                stream.push(SpoTriple::new(effect, p, o));
            }
        }
        for &(a, b) in &COINCIDENTAL {
            // same co-occurrence rate, random ORDER; the follower's context is
            // drawn from a RESIDUE-ALIGNED pool (v2): 8 distinct full-width
            // contexts spaced 512 ranks — all colliding to ONE id under the
            // 3×u8 cram (rank % 256), so p64 sees false consistency while
            // full width sees 8 different contexts.
            let (first, second) = if rng.chance(50) { (a, b) } else { (b, a) };
            stream.push(SpoTriple::new(first, random_rank(rng), random_rank(rng)));
            for _ in 0..rng.below(3) {
                stream.push(SpoTriple::new(
                    random_rank(rng),
                    random_rank(rng),
                    random_rank(rng),
                ));
            }
            if rng.chance(85) {
                let k = rng.below(8) as u16;
                let pool_p = (a % 256) + 300 + k * 512; // residue class fixed
                let pool_o = (b % 256) + 44 + k * 512;
                stream.push(SpoTriple::new(second, pool_p.min(4094), pool_o.min(4094)));
            }
        }
        // background traffic
        for _ in 0..6 {
            stream.push(SpoTriple::new(
                random_rank(rng),
                random_rank(rng),
                random_rank(rng),
            ));
        }
    }
    stream
}

// ── Rung machinery ──────────────────────────────────────────────────────────

/// The 8 canonical masks (O1: canonical constants — Pearl's ladder does not
/// vary per class). Bit 2=S, 1=P, 0=O, matching `causal_edge::pearl::CausalMask`.
const RUNGS: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

fn rung_cost(mask: u8) -> u32 {
    mask.count_ones()
}

/// Weakest-selected-plane reading (registered): 1 - max(selected dists)/255.
fn reading(mask: u8, d: (u8, u8, u8)) -> f64 {
    let mut worst = 0u8;
    if mask & 0b100 != 0 {
        worst = worst.max(d.0);
    }
    if mask & 0b010 != 0 {
        worst = worst.max(d.1);
    }
    if mask & 0b001 != 0 {
        worst = worst.max(d.2);
    }
    if mask == 0 {
        return 0.0; // the prior rung carries no plane evidence
    }
    1.0 - (worst as f64) / 255.0
}

/// v3 registered wave statistic: the PERSISTENCE FRACTION — how often the
/// relationship re-manifests strongly across the window. Robust to ≤50%
/// bursty contamination (breakdown-point argument); replaces the refuted
/// v1/v2 `mean − std` (see header, v2 KILL).
const STRONG_READING: f64 = 0.8;

fn stability(readings: &[f64]) -> f64 {
    if readings.is_empty() {
        return 0.0;
    }
    let strong = readings.iter().filter(|&&r| r >= STRONG_READING).count();
    strong as f64 / readings.len() as f64
}

/// Collect the instance sequence for a directed pair (A→B): per instance the
/// per-role distances of the current effect triple vs the WITNESS (previous
/// effect triple for this pair). First instance has no witness → skipped.
fn instance_readings(
    stream: &[SpoTriple],
    a: u16,
    b: u16,
    matrix: &WordDistanceMatrix,
    alias_p64: bool,
) -> Vec<(u8, u8, u8)> {
    let look = |t: &SpoTriple| -> SpoTriple {
        if alias_p64 {
            // CausalEdge64-style 3×u8 cram: 16-fold rank aliasing.
            SpoTriple::new(t.subject() % 256, t.predicate() % 256, t.object() % 256)
        } else {
            *t
        }
    };
    let mut out = Vec::new();
    let mut witness: Option<SpoTriple> = None;
    let mut i = 0usize;
    while i < stream.len() {
        if stream[i].subject() == a {
            // effect must FOLLOW the cause within the window — stream order
            // supplies orientation; no ordering search exists anywhere here.
            let hi = (i + WINDOW).min(stream.len() - 1);
            if let Some(j) = (i + 1..=hi).find(|&j| stream[j].subject() == b) {
                let eff = look(&stream[j]);
                if let Some(w) = witness {
                    out.push(eff.distance_per_role(&w, matrix));
                }
                witness = Some(eff);
                i = j; // do not double-count overlapping windows
            }
        }
        i += 1;
    }
    out
}

/// Mann-Whitney AUC: P(causal score > coincidental score) (+0.5 per tie).
fn auc(causal: &[f64], coincidental: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &c in causal {
        for &k in coincidental {
            if c > k {
                wins += 1.0;
            } else if (c - k).abs() < 1e-12 {
                wins += 0.5;
            }
        }
    }
    wins / (causal.len() * coincidental.len()) as f64
}

/// The escalation cascade (registered): climb cost levels 1→2→3; at each
/// level take the best stability among that level's rungs; escalate only if
/// it clears THETA_ESCALATE. Returns (score = stability at the last level
/// paid for, reached_top).
fn cascade(per_rung: &[(u8, f64)]) -> (f64, bool) {
    let mut score = 0.0;
    for level in 1..=3u32 {
        let best = per_rung
            .iter()
            .filter(|(m, _)| rung_cost(*m) == level)
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        score = best;
        if level == 3 {
            return (score, true);
        }
        if best < THETA_ESCALATE {
            return (score, false); // resolved below the counterfactual rung
        }
    }
    (score, false)
}

fn main() {
    // SpoFacet register-convention mirror (asserted once): rail k = (b[2k], b[2k+1]).
    {
        let reg: [u8; 12] = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61];
        let rails: [(u8, u8); 6] = core::array::from_fn(|k| (reg[2 * k], reg[2 * k + 1]));
        let back: Vec<u8> = rails.iter().flat_map(|&(x, y)| [x, y]).collect();
        assert_eq!(back, reg, "SpoFacet rail convention must round-trip");
        // larger-codebook pair carving: rank -> (basin, identity)
        let rank = 1300u16;
        assert_eq!(((rank >> 8) as u8, (rank & 0xFF) as u8), (5, 20));
    }

    // The real table (v2): LINEAR frequency-rank distance — the freq-is-cosine
    // grounding with full 0..255 dynamic range (|Δrank| up to 4095, /16).
    let matrix = WordDistanceMatrix::build(|i, j| ((i.abs_diff(j) as u32) / 16).min(255) as u8);

    let mut rng = Lcg(0x5EED_CA5CADE);
    let stream = build_stream(&mut rng);
    println!("stream: {} sentence-commit ticks", stream.len());

    let pairs: Vec<(u16, u16, bool)> = CAUSAL
        .iter()
        .map(|&(c, e, _, _)| (c, e, true))
        .chain(COINCIDENTAL.iter().map(|&(a, b)| (a, b, false)))
        .collect();

    let mut wave_c = Vec::new(); // full standing wave, SPO rung
    let mut wave_k = Vec::new();
    let mut single_c = Vec::new(); // single-cycle (last reading), SPO rung
    let mut single_k = Vec::new();
    let mut p64_c = Vec::new(); // p64 baseline: aliased + single-cycle + SPO
    let mut p64_k = Vec::new();
    let mut casc_c = Vec::new(); // cascade score
    let mut casc_k = Vec::new();
    let mut pruned = 0usize;

    for &(a, b, is_causal) in &pairs {
        let inst = instance_readings(&stream, a, b, &matrix, false);
        let spo_readings: Vec<f64> = inst.iter().map(|&d| reading(0b111, d)).collect();

        let wave = stability(&spo_readings);
        let single = spo_readings.last().copied().unwrap_or(0.0);

        let inst64 = instance_readings(&stream, a, b, &matrix, true);
        let p64 = inst64
            .iter()
            .map(|&d| reading(0b111, d))
            .next_back()
            .unwrap_or(0.0);

        let per_rung: Vec<(u8, f64)> = RUNGS
            .iter()
            .map(|&m| {
                let rs: Vec<f64> = inst.iter().map(|&d| reading(m, d)).collect();
                (m, stability(&rs))
            })
            .collect();
        let (casc_score, reached_top) = cascade(&per_rung);
        if !reached_top {
            pruned += 1;
        }

        if is_causal {
            wave_c.push(wave);
            single_c.push(single);
            p64_c.push(p64);
            casc_c.push(casc_score);
        } else {
            wave_k.push(wave);
            single_k.push(single);
            p64_k.push(p64);
            casc_k.push(casc_score);
        }
        println!(
            "pair ({a:>4},{b:>4}) {}: instances={:>2} wave={wave:>6.3} single={single:>6.3} p64={p64:>6.3} cascade={casc_score:>6.3}{}",
            if is_causal { "CAUSAL" } else { "COINC " },
            inst.len(),
            if reached_top { "" } else { "  [pruned]" },
        );
    }

    let auc_wave = auc(&wave_c, &wave_k);
    let auc_single = auc(&single_c, &single_k);
    let auc_p64 = auc(&p64_c, &p64_k);
    let auc_cascade = auc(&casc_c, &casc_k);
    let pruned_fraction = pruned as f64 / pairs.len() as f64;

    println!("\n== D-CSW-1 leg 1 ==");
    println!("auc_wave     = {auc_wave:.3}");
    println!("auc_single   = {auc_single:.3}   (gate: wave >= single + {MARGIN_VS_SINGLE})");
    println!("auc_p64      = {auc_p64:.3}   (gate: wave >= p64 + {MARGIN_VS_P64})");
    println!("auc_cascade  = {auc_cascade:.3}   (gate: >= wave - {MAX_AUC_DROP})");
    println!("pruned       = {pruned_fraction:.3}   (gate: >= {MIN_PRUNED})");

    // Registered gates — a failure here is the KILL, recorded loudly.
    assert!(
        auc_wave >= auc_single + MARGIN_VS_SINGLE,
        "KILL gate 1: standing wave does not beat single-cycle by the registered margin"
    );
    assert!(
        auc_wave >= auc_p64 + MARGIN_VS_P64,
        "KILL gate 2: standing wave does not beat the p64 8-bit-alias baseline by the registered margin"
    );
    assert!(
        pruned_fraction >= MIN_PRUNED,
        "KILL gate 3a: escalation cascade prunes less than the registered fraction"
    );
    assert!(
        auc_cascade >= auc_wave - MAX_AUC_DROP,
        "KILL gate 3b: cascade pruning destroys separation"
    );
    println!("\nPASS — all registered gates green (leg 1; temporal.rs binding = leg 2).");
}
