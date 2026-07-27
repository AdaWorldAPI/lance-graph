//! PROBE-STRIDE-DISCREPANCY — settle "which stride is what" with the metric
//! pillar 3 is already proven against, instead of a downstream proxy.
//!
//! **Why this exists.** This session compared strides (`Base17`'s
//! `GOLDEN_STEP=11`, `11/17 × φ`, the Pythagorean comma, the Quintenzirkel) by
//! correlating downstream embedding distances — a noisy proxy for a question
//! that is decided upstream, in number theory, by **Weyl equidistribution**.
//! `jc::weyl` already holds both the metric (star-discrepancy `D*`) and an
//! ABSOLUTE gate (Ostrowski `2/N`, tight for φ because every continued-fraction
//! partial quotient is 1), so this probe calls
//! [`jc::weyl::star_discrepancy`] rather than re-deriving it.
//!
//! **The claim it settles.** `(i·11) mod 17` is an INTEGER relabel: `11/17` is
//! RATIONAL, so its Weyl sequence is periodic with period 17 and visits only 17
//! distinct points — `D*` floors near `1/17` however large `N` grows. A real
//! stride from the same pair, `frac(11/17·φ)`, is irrational and equidistributes.
//! `gcd(11,17)=1` buys full COVERAGE of the 17 cells — the Quintenzirkel
//! property, exactly as `gcd(7,12)=1` closes the circle of fifths — and nothing
//! more. Coverage is the precondition; low discrepancy is the payload.
//!
//! **Falsifiers.** (1) Rational strides must FLOOR — `D*` must not fall like
//! `1/N`. (2) Irrational strides must actually improve with `N`, or the
//! comparison is measuring noise. (3) φ must beat the Quintenzirkel control and
//! clear Ostrowski at `N=144` — pillar 3's own pass criterion. If (3) fails,
//! this harness contradicts a shipped proof and the harness is wrong.
//!
//! No embeddings, no codebooks, no significance claim — `D*` is deterministic,
//! so `I-NOISE-FLOOR-JIRAK` does not apply here (it governs the sampled
//! correlations elsewhere, which this probe deliberately does not use).
//!
//! ```text
//! cargo run --release --manifest-path crates/jc/Cargo.toml \
//!   --example probe_stride_discrepancy
//! ```

use jc::weyl::{PHI_INV, QUINTENZIRKEL, star_discrepancy};

fn main() {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let frac = |x: f64| x - x.floor();
    // The Pythagorean comma as a stride in octaves: log2(531441/524288).
    let comma = (531_441.0_f64 / 524_288.0).log2();

    // (name, stride, is_rational)
    let strides: [(&str, f64, bool); 7] = [
        ("phi^-1 (golden)", PHI_INV, false),
        ("Quintenzirkel log2(3/2)", QUINTENZIRKEL, false),
        ("11/17 (Base17 GOLDEN_STEP)", 11.0 / 17.0, true),
        ("4/17 (helix CurveRuler)", 4.0 / 17.0, true),
        ("frac(11/17 * phi)", frac(11.0 / 17.0 * phi), false),
        ("Pythagorean comma (log2)", comma, false),
        ("frac(17/phi)", frac(17.0 / phi), false),
    ];

    println!(
        "{:<28} {:>11} {:>11} {:>12} {:>12}",
        "stride", "value", "D*(N=144)", "D*(N=10000)", "N=144/10000"
    );
    println!("{}", "-".repeat(80));
    let mut rational = Vec::new();
    let mut irrational = Vec::new();
    for (name, s, is_rat) in strides {
        let d144 = star_discrepancy(144, s);
        let d10k = star_discrepancy(10_000, s);
        let improves = d10k < d144 * 0.5;
        println!(
            "{name:<28} {s:>11.7} {d144:>11.6} {d10k:>12.6} {:>12.2}",
            d144 / d10k
        );
        if is_rat {
            rational.push((name, d10k));
        } else {
            irrational.push((name, d10k, improves));
        }
    }

    // (1) A periodic sequence cannot equidistribute.
    for (name, d) in &rational {
        assert!(
            *d > 0.01,
            "{name}: rational stride reached D*={d:.6} at N=10000 - a sequence \
             with period 17 cannot equidistribute, so this harness is wrong"
        );
    }
    // (2) Irrational strides must improve, or the table is noise.
    for (name, _, improves) in &irrational {
        assert!(improves, "{name}: irrational stride did NOT improve with N");
    }
    // (3) Agreement with the shipped pillar-3 criterion.
    let (d_phi, d_quint) = (
        star_discrepancy(144, PHI_INV),
        star_discrepancy(144, QUINTENZIRKEL),
    );
    assert!(
        d_phi < d_quint,
        "phi ({d_phi:.6}) did not beat the Quintenzirkel ({d_quint:.6}) at N=144 \
         - contradicts jc pillar 3; trust the pillar, fix this probe"
    );
    assert!(
        d_phi < 2.0 / 144.0,
        "phi missed the Ostrowski 2/N bound ({d_phi:.6} vs {:.6})",
        2.0 / 144.0
    );

    println!(
        "\n--- verdict ---\n  \
         RATIONAL strides floor. 11/17 and 4/17 visit 17 points forever, so D*\n  \
         stalls near 1/17 = {:.4} instead of falling with N. gcd(step,17)=1 buys\n  \
         full COVERAGE of those 17 cells and nothing else - so any readout that\n  \
         is symmetric over all 17 (Base17's `l1`) cannot see the step at all.\n  \
         That is the relabel result, in its proper form: a discrepancy statement,\n  \
         not an empirical correlation.\n\n  \
         IRRATIONAL strides equidistribute. phi^-1 is optimal and clears the\n  \
         Ostrowski 2/N bound; a real stride built from the same 11/17 pair does\n  \
         not close, and the residue it leaves is where information can live.\n  \
         Circle (closes, no comma) vs spiral (does not, comma is the payload).",
        1.0 / 17.0
    );
}
