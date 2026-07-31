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
//! sweep; the other is REFUTED, and its refutation vindicates the shipped design.
//!
//! ## H1 — the intake is STRUCTURE-BLIND (pre-registered failure mode (a))
//!
//! [`CurveRuler::from_hhtl(path, depth)`] is `from_place(path + depth)`, and
//! [`CurveRuler::from_place`] is `place % 17`. The address therefore enters the
//! ruler **as a number, never as a hierarchy**. Feed the same 256 cells carved
//! 4-ary (4 levels × 2 bits) or 16-ary (2 levels × 4 bits) and the ruler receives
//! the same 256 integers, so it produces the same start-offset multiset — the
//! carvings are indistinguishable to it.
//!
//! **Consequence: the residue-vs-granularity sweep cannot produce a signal, and a
//! null from it would say nothing about 4⁴.** Reporting "4⁴ doesn't help the
//! calculator" from that null would have been a false negative caused entirely by
//! the intake. This is an INTAKE finding, exactly as pre-registered.
//!
//! ## H2 — stride 4 vs the φ stride 11 (pre-registered failure mode (b), REFUTED)
//!
//! The suspicion was: `gcd(4,17) = 1` buys *coverage*, not *low discrepancy*;
//! stride 4's first four steps are `0,4,8,12` (one column of the 4×4 raster,
//! clumped) while the φ stride over 17 is **11** (`17/φ = 10.51`, and the
//! workspace's C5 pins `11/17` as the proven golden step). Since HHTL terminates
//! early, only short prefixes are ever consumed — so short-prefix uniformity is
//! what matters, and stride 4 looked like a raster wearing a φ-spiral's docstring.
//!
//! **The measurement refutes it at exactly the prefix length a 4-ary tier
//! consumes.** After 4 steps, stride 4 gives gaps `[4,4,4,5]` — near-perfect
//! tiling of the 17-circle — because `4·4 = 16 ≈ 17`. The comma (`17 = 4² + 1`)
//! is what makes a 4-step prefix land almost uniformly. Stride 11 is better at
//! n=2..3 and worse at n=4..6. The shipped constants are *matched to the 4-ary
//! use case*, not sloppy about it.
//!
//! ## H3 — the intake blindness is FIXABLE, not fundamental
//!
//! A per-tier intake (one ruler seed per 2-bit group, folded with its level)
//! preserves ancestry by construction: two cells sharing a k-level address prefix
//! share their first k seeds. The flat intake destroys that. H3 measures both, so
//! the finding lands as a *design direction* rather than a dead end.
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

    // ── H1: the two carvings are indistinguishable to the ruler ─────────────
    let h4 = start_histogram(4); // 4-ary, 4 levels × 2 bits
    let h16 = start_histogram(2); // 16-ary, 2 levels × 4 bits
    let mut s4: Vec<usize> = h4.to_vec();
    let mut s16: Vec<usize> = h16.to_vec();
    s4.sort_unstable();
    s16.sort_unstable();
    let identical_multiset = s4 == s16;
    gate(
        "H1 intake is STRUCTURE-BLIND (the sweep-killer)",
        identical_multiset,
        format!(
            "start-offset histograms over the SAME 256 cells:\n          \
             4-ary(depth 4): {h4:?}\n          16-ary(depth 2): {h16:?}\n        \
             identical multiset = {identical_multiset} (a pure rotation). \
             `from_place(p) = p % 17` consumes the address as a NUMBER, so 4⁴ \
             ancestry never reaches the ruler — a residue-vs-granularity sweep \
             would return a null for a reason unrelated to 4⁴."
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
    // The 4-ary tier consumes exactly ARITY=4 steps. That is the prefix that must
    // be uniform. Falsifier for the REFUTATION: stride 4 worse than φ at n=4.
    gate(
        "H2 shipped stride 4 is MATCHED to the 4-step prefix (my suspicion REFUTED)",
        d4_at4 <= d11_at4,
        format!(
            "{}\n        At n=4 — the prefix a 4-ary tier actually consumes — \
             stride 4 gives gaps {g4_at4:?} (spread {d4_at4}) vs φ-stride 11 spread \
             {d11_at4}. 4·4 = 16 ≈ 17, so four steps of 4 nearly tile the circle: \
             the comma is what makes the 4-step prefix land uniformly. \
             Pre-registered suspicion (\"stride 4 is a raster wearing a φ-spiral's \
             docstring\") does NOT hold at the lengths early termination reads.",
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
            if !(a16 as usize + b16 as usize).is_multiple_of(7) {
                continue; // deterministic thinning; still thousands of pairs
            }
            let (a, b) = (a16 as u8, b16 as u8);
            pairs += 1;
            let k = shared_levels_4ary(a, b);
            // FLAT: one seed each. "Ancestry preserved" would mean cells sharing a
            // prefix share the seed — which mod 17 cannot arrange.
            if k > 0 && flat_seed(a) == flat_seed(b) {
                flat_agree += 1;
            }
            // PER-TIER: cells sharing k levels must share their first k seeds.
            if k > 0 && shared_seed_prefix(tiered_seeds(a), tiered_seeds(b)) >= k {
                tier_agree += 1;
            }
        }
    }
    gate(
        "H3 blindness is FIXABLE at the intake (per-tier seeds keep ancestry)",
        tier_agree > flat_agree * 4 && pairs > 100,
        format!(
            "over {pairs} sampled cell pairs sharing ≥1 address level: \
             FLAT intake preserves ancestry in {flat_agree} (mod 17 scrambles it); \
             PER-TIER intake preserves it in {tier_agree} — by construction, since \
             each 2-bit group seeds its own ruler. The fix is to feed the ruler the \
             address's HIERARCHY, not its integer value."
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
         (E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1). H2 additionally clears the\n\
         shipped constants of the discrepancy suspicion. H3 names the unblock."
    );
    if failures > 0 {
        std::process::exit(1);
    }
}
