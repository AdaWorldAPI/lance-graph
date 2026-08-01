//! PROBE-CLAM-VS-HELIX-RESIDUE, leg 0 — **the intake check that must run before
//! the residue sweep**, and it kills the sweep as specified.
//!
//! ## Why this exists
//!
//! `PROBE-WORDNET-44-ACTIVATION` (merged, `E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1`)
//! measured that a 4⁴ address carries real taxonomic ancestry and that the
//! sub-nibble rung is worth **2.47 WordNet hops** the 16-ary `NiblePath` cannot
//! express. The operator then fixed the division of labour: the fold is the
//! ADDRESS, **CLAM is the established CALCULATOR**, and **HHTL + helix residue**
//! is the alternative calculator — *"exact address, exact spacial location /
//! perturbation"*.
//!
//! The queued hypothesis was: a finer address makes more of the answer
//! deterministic PLACE and less of it stored RESIDUE, so residue should shrink
//! as address granularity rises. Two failure modes were **pre-registered** before
//! any measurement. This probe runs both. One is confirmed and it is fatal to the
//! sweep; the other is NOT CONFIRMED — which retracts a suspicion without
//! licensing any design claim in its place (see the ⚠ under H2).
//!
//! ## H1 — the intake is STRUCTURE-BLIND (pre-registered failure mode (a))
//!
//! [`CurveRuler::from_hhtl(path, depth)`] is `from_place(path + depth)`, and
//! [`CurveRuler::from_place`] is `place % 17`. The address therefore enters the
//! ruler **as a number, never as a hierarchy** — `from_hhtl` has no parameter
//! that could carry a carving, so how the byte's bits group into levels cannot
//! influence the output. The measurable half: the entire difference between the
//! two carvings, as the ruler sees it, is a CONSTANT rotation with ZERO per-cell
//! variance; a cell-dependent shift would falsify it.
//!
//! (An earlier version of this gate compared SORTED histograms. That was wrong:
//! sorting discards which cell maps to which offset, and the unsorted histograms
//! genuinely differ, so the gate passed while hiding that every cell had moved —
//! codex P1 on #876.)
//!
//! **Consequence: the residue-vs-granularity sweep cannot produce a signal, and a
//! null from it would say nothing about 4⁴.** Reporting "4⁴ doesn't help the
//! calculator" from that null would have been a false negative caused entirely by
//! the intake. This is an INTAKE finding, exactly as pre-registered.
//!
//! ## H2 — stride 4 vs the φ stride 11 (pre-registered (b): NOT CONFIRMED)
//!
//! The suspicion was: `gcd(4,17) = 1` buys *coverage*, not *low discrepancy*;
//! stride 4's first four steps are `0,4,8,12` (one column of the 4×4 raster,
//! clumped) while the φ stride over 17 is **11** (`17/φ = 10.51`, and the
//! workspace's C5 pins `11/17` as the proven golden step). Since HHTL terminates
//! early, only short prefixes are ever consumed — so short-prefix uniformity is
//! what matters, and stride 4 looked like a raster wearing a φ-spiral's docstring.
//!
//! The measurement does NOT confirm it: after 4 steps stride 4 gives gaps
//! `[4,4,4,5]` — near-perfect tiling of the 17-circle — because `4·4 = 16 ≈ 17`.
//! Stride 11 is better at n=2..3 and worse at n=4..6.
//!
//! **⚠ The design conclusion is WITHDRAWN (codex P1 on #876).** The premise that
//! a 4-ary tier consumes four ruler steps is not wired into anything that ships:
//! `ResidueEncoder::encode` reads only `start_offset` (n=1, `residue.rs:157`),
//! `index`/`arc` appear solely in `walk_spectrum` and this crate's tests, and the
//! shipped router `NiblePath` is `FAN_OUT = 16`. So this RETRACTS a suspicion; it
//! does not establish that the constants are "matched to 4-ary" — that would need
//! a consumer that reads 4 steps, and none currently exists.
//!
//! ## H3 — the intake blindness is FIXABLE, not fundamental
//!
//! A per-tier intake (one ruler seed per 2-bit group, folded with its level)
//! preserves ancestry by construction: two cells sharing a k-level address prefix
//! share their first k seeds — so per-tier agreement equals the eligible
//! population exactly, and the gate asserts that equality as a CONSTRUCTION
//! CHECK, not as evidence. The informative number is the FLAT rate. The finding
//! lands as a *design direction* rather than a dead end.
//!
//! Run: `cargo run --manifest-path crates/helix/Cargo.toml --example probe_hhtl_intake_blindness`

use helix::curve_ruler::CurveRuler;

const CELLS: usize = 256;

// ── gate harness ────────────────────────────────────────────────────────────

static FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn gate(name: &str, pass: bool, detail: String) {
    println!(
        "  [{}] {name}\n        {detail}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Start-offset histogram over all 256 cells for a carving of the given depth.
/// `depth` is what `NiblePath` would report: 4 for the 4-ary (4×2-bit) address,
/// 2 for the 16-ary (2×4-bit) address. Same cells, same numbers, different
/// *claimed* structure.
fn start_histogram(depth: u8) -> [usize; 17] {
    let mut h = [0usize; 17];
    for cell in 0..CELLS {
        let r = CurveRuler::from_hhtl(cell as u64, depth);
        h[r.start_offset() as usize] += 1;
    }
    h
}

/// Gap structure of the first `n` points of `(stride·k) mod 17`, as
/// (max_gap − min_gap). Lower = more uniform. This is the three-distance view:
/// a low-discrepancy prefix leaves near-equal gaps around the circle.
fn prefix_gap_spread(stride: u8, n: usize) -> (u8, Vec<u8>) {
    let m = CurveRuler::MODULUS;
    let mut pts: Vec<u8> = (0..n)
        .map(|k| ((stride as u32 * k as u32) % m as u32) as u8)
        .collect();
    pts.sort_unstable();
    pts.dedup();
    let mut gaps: Vec<u8> = Vec::with_capacity(pts.len());
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let d = (b as i16 - a as i16).rem_euclid(m as i16) as u8;
        gaps.push(if d == 0 { m } else { d });
    }
    let spread = gaps.iter().max().copied().unwrap_or(0) - gaps.iter().min().copied().unwrap_or(0);
    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    (spread, sorted)
}

/// Levels of shared prefix between two 4-ary addresses (0..=4).
fn shared_levels_4ary(a: u8, b: u8) -> usize {
    for k in 0..4 {
        let shift = 2 * (4 - k) - 2;
        if (a >> shift) != (b >> shift) {
            return k;
        }
    }
    4
}

/// FLAT intake (what ships): the whole address collapses to one seed.
fn flat_seed(cell: u8) -> u8 {
    CurveRuler::from_hhtl(cell as u64, 4).start_offset()
}

/// PER-TIER intake (the proposed fix): one seed per 2-bit group, folded with its
/// level so the same group value at different depths starts elsewhere — the same
/// `depth` fold `from_hhtl` already applies, but applied PER LEVEL instead of once.
fn tiered_seeds(cell: u8) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (level, slot) in out.iter_mut().enumerate() {
        let shift = 2 * (4 - 1 - level);
        let group = (cell >> shift) & 0b11;
        *slot = CurveRuler::from_hhtl(group as u64, level as u8).start_offset();
    }
    out
}

fn shared_seed_prefix(a: [u8; 4], b: [u8; 4]) -> usize {
    (0..4).take_while(|&i| a[i] == b[i]).count()
}

fn main() {
    println!("PROBE-HHTL-INTAKE-BLINDNESS (PROBE-CLAM-VS-HELIX-RESIDUE, leg 0)");
    println!(
        "helix CurveRuler: MODULUS={} STRIDE={}  (17 = 4²+1 — the comma)\n",
        CurveRuler::MODULUS,
        CurveRuler::STRIDE
    );

    // ── H1: the carving cannot reach the ruler ──────────────────────────────
    //
    // The FIRST version of this gate sorted both histograms and compared
    // multisets. That was wrong twice over (codex P1): sorting discards which
    // cell maps to which offset, and the UNSORTED histograms genuinely differ
    // (the peak sits at index 4 vs index 2), so a "shared aggregate
    // distribution" gate passed while hiding that every cell's start moved.
    //
    // The real argument is not aggregate at all. It has two halves:
    //
    //   (i)  TYPE-LEVEL: `from_hhtl(path: u64, depth: u8)` has no parameter
    //        that could carry a carving. How the byte's bits group into levels
    //        is not an input, so it cannot influence the output. No measurement
    //        can establish this and none should pretend to — it is stated.
    //   (ii) EMPIRICAL, and this is what the gate measures: the entire
    //        difference between the two carvings, as the ruler sees it, is a
    //        CONSTANT rotation with ZERO per-cell variance. If the ruler
    //        carried any per-cell structure, the shift would vary by cell.
    //        A varying shift is the falsifier.
    let h4 = start_histogram(4); // "4-ary" — 4 levels × 2 bits
    let h16 = start_histogram(2); // "16-ary" — 2 levels × 4 bits
    let shifts: Vec<u8> = (0..CELLS)
        .map(|cell| {
            let a = CurveRuler::from_hhtl(cell as u64, 4).start_offset() as i16;
            let b = CurveRuler::from_hhtl(cell as u64, 2).start_offset() as i16;
            (a - b).rem_euclid(CurveRuler::MODULUS as i16) as u8
        })
        .collect();
    let first = shifts[0];
    let constant_shift = shifts.iter().all(|&s| s == first);
    gate(
        "H1 the carving cannot reach the ruler (constant-shift, zero variance)",
        constant_shift,
        format!(
            "per-cell start-offset shift between the two carvings is CONSTANT at \
             {first} for all {CELLS} cells (variance 0) — the depth term moves every \
             cell identically and the carving contributes nothing. Falsifier: any \
             cell-dependent shift.\n        Unsorted histograms (shown because \
             sorting them would hide exactly this): 4-ary(depth 4) {h4:?}; \
             16-ary(depth 2) {h16:?} — a rotation, not a reshaping.\n        \
             Type-level half (stated, not measured): `from_hhtl(path: u64, depth: u8)` \
             takes no carving parameter, so 4⁴ level structure has no channel into \
             `from_place(p) = p % 17`."
        ),
    );

    // ── H2: prefix uniformity, shipped stride 4 vs the φ stride 11 ──────────
    let phi_stride = 11u8; // nearest integer to 17/φ = 10.5066 (workspace C5)
    let mut rows = Vec::new();
    for n in 2..=6 {
        let (d4, g4) = prefix_gap_spread(CurveRuler::STRIDE, n);
        let (d11, g11) = prefix_gap_spread(phi_stride, n);
        rows.push(format!(
            "n={n}: stride4 spread={d4} gaps={g4:?} | stride11(φ) spread={d11} gaps={g11:?}"
        ));
    }
    let (d4_at4, g4_at4) = prefix_gap_spread(CurveRuler::STRIDE, 4);
    let (d11_at4, _) = prefix_gap_spread(phi_stride, 4);
    // ⚠ CLAIM DOWNGRADED (codex P1). The first version asserted the shipped
    // constants are "MATCHED to the 4-ary use case". That premise — that a
    // 4-ary tier consumes 4 ruler steps — is NOT WIRED anywhere that ships:
    //   • `ResidueEncoder::encode` reads ONLY `start_offset` (residue.rs:157),
    //     i.e. n = 1;
    //   • `CurveRuler::index` / `arc` appear only in `walk_spectrum` and this
    //     crate's own tests;
    //   • the shipped router `NiblePath` is FAN_OUT = 16, not 4.
    // So the n=4 comparison is a property of a HYPOTHETICAL 4-step consumer.
    // What survives is narrower and still worth recording: the pre-registered
    // suspicion (stride 4 clumps where φ would not) is NOT CONFIRMED. That is a
    // retraction of a concern, not a vindication of a design.
    gate(
        "H2 the \"stride 4 clumps\" suspicion is NOT CONFIRMED (design claim withdrawn)",
        d4_at4 <= d11_at4,
        format!(
            "{}\n        At n=4, stride 4 gives gaps {g4_at4:?} (spread {d4_at4}) vs \
             φ-stride 11 spread {d11_at4} — 4·4 = 16 ≈ 17, so four steps of 4 nearly \
             tile the circle.\n        ⚠ SCOPE: no shipped path consumes a 4-step \
             prefix. `ResidueEncoder::encode` reads only `start_offset` (n=1); \
             `index`/`arc` are used solely by `walk_spectrum` and crate tests; the \
             shipped `NiblePath` is FAN_OUT=16. This therefore RETRACTS the \
             pre-registered suspicion and does NOT establish that the constants are \
             matched to 4-ary — that would need a consumer that reads 4 steps, which \
             does not currently exist.",
            rows.join("\n        ")
        ),
    );

    // ── H3: a per-tier intake preserves ancestry; the flat one destroys it ──
    let mut flat_agree = 0usize;
    let mut tier_agree = 0usize;
    let mut pairs = 0usize;
    // NOTE: iterate in u16. `CELLS as u8` is `256 as u8` == 0, which silently
    // made this loop empty on the first run — the gate reported 0 pairs and
    // correctly FAILED rather than passing vacuously.
    for a16 in 0u16..CELLS as u16 {
        for b16 in (a16 + 1)..CELLS as u16 {
            let (a, b) = (a16 as u8, b16 as u8);
            let k = shared_levels_4ary(a, b);
            if k == 0 {
                continue; // not eligible: no shared address level to preserve
            }
            // COUNT ONLY ELIGIBLE PAIRS, over the FULL population.
            //
            // Two successive labelling defects landed here, both caught in review:
            //   1. (codex P2) the counter incremented BEFORE this eligibility
            //      check, so "4,662 pairs sharing ≥1 address level" was really
            //      every thinned pair;
            //   2. (CodeRabbit) the corrected 1,152 was the eligible THINNED
            //      SAMPLE, not the eligible POPULATION — a second population-label
            //      error stacked on the fix for the first.
            // A deterministic `(a+b) % 7` thinning caused both and bought nothing:
            // 256·255/2 = 32,640 pairs is trivial to enumerate exhaustively. The
            // thinning is removed, so there is no sample to mislabel.
            pairs += 1;
            // FLAT: one seed each. "Ancestry preserved" would mean cells sharing a
            // prefix share the seed — which mod 17 cannot arrange.
            if flat_seed(a) == flat_seed(b) {
                flat_agree += 1;
            }
            // PER-TIER: cells sharing k levels must share their first k seeds.
            if shared_seed_prefix(tiered_seeds(a), tiered_seeds(b)) >= k {
                tier_agree += 1;
            }
        }
    }
    // Per-tier agreement is TRUE BY CONSTRUCTION for every eligible pair, so it
    // should equal the eligible population exactly; asserting that is the
    // construction check, and the informative number is the FLAT rate.
    let flat_rate = flat_agree as f64 / pairs as f64;

    // DECLARED NULL MODEL (replaces a hand-tuned `flat_rate < 0.25`, which
    // 24.9 % would have passed while the text claimed ≈1/17 — CodeRabbit).
    // H0: "the flat seed carries no ancestry information." Under H0 its
    // agreement rate among ELIGIBLE pairs (sharing ≥1 address level) must equal
    // its rate among ALL pairs, because ancestry would be irrelevant to it.
    // Both are computed exhaustively; the gate is that they COINCIDE.
    let (mut all_pairs, mut all_agree) = (0usize, 0usize);
    for a16 in 0u16..CELLS as u16 {
        for b16 in (a16 + 1)..CELLS as u16 {
            all_pairs += 1;
            if flat_seed(a16 as u8) == flat_seed(b16 as u8) {
                all_agree += 1;
            }
        }
    }
    let null_rate = all_agree as f64 / all_pairs as f64;
    gate(
        "H3 blindness is FIXABLE at the intake (per-tier seeds keep ancestry)",
        tier_agree == pairs && flat_rate <= null_rate && pairs > 100,
        format!(
            "eligible population = {pairs} pairs sharing ≥1 address level \
             (EXHAUSTIVE over all {all_pairs} = 256·255/2 cell pairs — no sampling, \
             no thinning).\n        FLAT intake: {flat_agree}/{pairs} = {:.4}% among \
             ELIGIBLE pairs vs {all_agree}/{all_pairs} = {:.4}% among ALL pairs \
             (H0 = \"the flat seed carries no ancestry information\", so the \
             eligible rate should not EXCEED the base rate). Δ = {:+.4} pp — the \
             eligible rate is BELOW the base rate, so the flat seed gives no \
             ancestry lift; it is in fact mildly ancestry-AVERSE, because cells \
             sharing an address prefix are numerically close and therefore less \
             likely to be congruent mod 17. Gate = no lift (falsifier: an eligible \
             rate materially above the base rate, which would mean the flat seed \
             DOES carry ancestry). No hand-picked ceiling.\n        PER-TIER intake: {tier_agree}/{pairs} = \
             100% BY CONSTRUCTION (each 2-bit group seeds its own ruler) — asserted \
             as a construction check, not as evidence.\n        The fix is to feed \
             the ruler the address's HIERARCHY, not its integer.",
            100.0 * flat_rate,
            100.0 * null_rate,
            100.0 * (flat_rate - null_rate)
        ),
    );

    let failures = FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "\n{}",
        if failures == 0 {
            "ALL GATES GREEN".to_string()
        } else {
            format!("{failures} GATE(S) FAILED")
        }
    );
    println!(
        "\nVERDICT: the residue-vs-granularity sweep is NOT RUN — H1 shows it cannot\n\
         carry signal through the current intake. That is a finding about the\n\
         CALCULATOR'S INTAKE, not about the 4⁴ address, whose ancestry result stands\n\
         (E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1). H2 RETRACTS a suspicion it\n\
         does not replace with a design claim — no shipped path reads a 4-step prefix.\n\
         H3 names the unblock and reports the flat rate over the FULL population\n\
         (360/8064 = 4.5%, ≈ the 1/17 ≈ 5.9% chance level), not a ratio."
    );
    if failures > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H1: the carving reaches the ruler only as a constant rotation.
    #[test]
    fn carving_difference_is_a_constant_shift() {
        let shifts: Vec<u8> = (0..CELLS)
            .map(|c| {
                let a = CurveRuler::from_hhtl(c as u64, 4).start_offset() as i16;
                let b = CurveRuler::from_hhtl(c as u64, 2).start_offset() as i16;
                (a - b).rem_euclid(CurveRuler::MODULUS as i16) as u8
            })
            .collect();
        assert_eq!(shifts.len(), CELLS);
        assert!(
            shifts.iter().all(|&s| s == 2),
            "expected constant shift of 2"
        );
    }

    /// Falsifier for the above: a genuinely structure-carrying seed must NOT
    /// produce a constant shift. Guards against the gate passing for any seed.
    #[test]
    fn a_structure_carrying_seed_would_not_be_constant_shift() {
        let shifts: std::collections::HashSet<u8> = (0..CELLS)
            .map(|c| {
                let t = tiered_seeds(c as u8);
                (t[0] as i16 - t[3] as i16).rem_euclid(CurveRuler::MODULUS as i16) as u8
            })
            .collect();
        assert!(shifts.len() > 1, "per-tier seeds must vary by cell");
    }

    /// H2: the exact recorded gap structure, so a constants change is caught.
    #[test]
    fn prefix_gap_spread_matches_recorded_values() {
        assert_eq!(prefix_gap_spread(4, 4), (1, vec![4, 4, 4, 5]));
        assert_eq!(prefix_gap_spread(11, 4).0, 5);
        assert_eq!(prefix_gap_spread(4, 2).0, 9);
        assert_eq!(prefix_gap_spread(11, 2).0, 5);
    }

    /// The shipped constants this probe reasons about.
    #[test]
    fn shipped_constants_are_what_the_probe_assumes() {
        assert_eq!(CurveRuler::MODULUS, 17);
        assert_eq!(CurveRuler::STRIDE, 4);
        // 17 = 4² + 1 — the comma. Without the +1, stride 4 mod 16 covers 4/16.
        let covered: std::collections::HashSet<u8> = (0..16u8).map(|k| (4 * k) % 16).collect();
        assert_eq!(covered.len(), 4, "stride 4 over 16 retraces one column");
    }

    /// H3: the recorded population and agreement counts.
    #[test]
    fn h3_population_and_agreement_counts() {
        let (mut eligible, mut flat, mut tier) = (0usize, 0usize, 0usize);
        for a16 in 0u16..CELLS as u16 {
            for b16 in (a16 + 1)..CELLS as u16 {
                let (a, b) = (a16 as u8, b16 as u8);
                let k = shared_levels_4ary(a, b);
                if k == 0 {
                    continue;
                }
                eligible += 1;
                if flat_seed(a) == flat_seed(b) {
                    flat += 1;
                }
                if shared_seed_prefix(tiered_seeds(a), tiered_seeds(b)) >= k {
                    tier += 1;
                }
            }
        }
        assert_eq!(eligible, 8064, "eligible population");
        assert_eq!(flat, 360, "flat-intake ancestry agreements");
        assert_eq!(
            tier, eligible,
            "per-tier agreement is total by construction"
        );
    }

    #[test]
    fn shared_levels_is_a_prefix_measure() {
        assert_eq!(shared_levels_4ary(0b00_00_00_00, 0b00_00_00_11), 3);
        assert_eq!(shared_levels_4ary(0b01_00_00_00, 0b10_00_00_00), 0);
        assert_eq!(shared_levels_4ary(0b11_01_10_00, 0b11_01_10_00), 4);
    }

    #[test]
    fn start_histogram_covers_every_cell() {
        assert_eq!(start_histogram(4).iter().sum::<usize>(), CELLS);
    }
}
