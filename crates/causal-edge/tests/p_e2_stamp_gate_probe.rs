//! P-E2: count each source once (pre-registered in
//! `.claude/board/entries/2026-10-22-coresearch-pal256-evidence-probes.md`).
//!
//! The production revision (`Opcode::Revision.truth_code`) pools evidence
//! weights from four scalars. It has no stamp, so revising a belief with
//! evidence it already holds raises confidence. This probe measures that,
//! then tests a candidate gate in the ONA shape (overlap check before
//! revision; on overlap, keep the higher-confidence belief instead of
//! pooling). The gate is local to this test. Where its stamp lives (G5) is
//! an owner decision; nothing here touches CE64 bits.
//!
//! Pre-registered conditions:
//! - fires: ungated revision of duplicated evidence raises `c` above the
//!   single-source code (the measured defect);
//! - pass: under the gate, duplicates give Δc = 0 and the posterior sd
//!   ratio (duplicated / single) is 1.00 ± 0.05;
//! - silent twin: under the gate, k disjoint sources give exactly the codes
//!   ungated revision gives, so the gate does not freeze genuine evidence.

use causal_edge::Opcode;

/// A belief with the set of sources it was derived from (one bit each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Belief {
    f: u8,
    c: u8,
    stamp: u64,
}

fn revise_ungated(a: Belief, b: Belief) -> Belief {
    let (f, c) = Opcode::Revision.truth_code(a.f, a.c, b.f, b.c);
    Belief {
        f,
        c,
        stamp: a.stamp | b.stamp,
    }
}

/// ONA shape: pool only disjoint evidence; on overlap keep the stronger.
fn revise_gated(a: Belief, b: Belief) -> Belief {
    if a.stamp & b.stamp != 0 {
        if b.c > a.c {
            b
        } else {
            a
        }
    } else {
        revise_ungated(a, b)
    }
}

/// Beta posterior sd rebuilt from codes, horizon k = 1 (Jeffreys).
fn posterior_sd(b: Belief) -> f64 {
    let c = f64::from(b.c);
    let w = c / (255.0 - c);
    let w_pos = w * f64::from(b.f) / 255.0;
    let (a, bb) = (w_pos + 0.5, w - w_pos + 0.5);
    (a * bb / ((a + bb) * (a + bb) * (a + bb + 1.0))).sqrt()
}

const SOURCE: Belief = Belief {
    f: 200,
    c: 128,
    stamp: 1,
};

fn fold_duplicates(k: usize, revise: fn(Belief, Belief) -> Belief) -> Belief {
    (1..k).fold(SOURCE, |acc, _| revise(acc, SOURCE))
}

#[test]
fn ungated_revision_double_counts_a_duplicated_source() {
    for k in [2usize, 4, 8] {
        let b = fold_duplicates(k, revise_ungated);
        let ratio = posterior_sd(b) / posterior_sd(SOURCE);
        println!(
            "ungated k={k}: c {} -> {}, sd ratio {ratio:.3}",
            SOURCE.c, b.c
        );
        assert!(
            b.c > SOURCE.c,
            "k={k}: the defect is gone; re-register P-E2 (c stayed {})",
            b.c
        );
        assert!(
            ratio < 0.95,
            "k={k}: duplicates no longer shrink the posterior"
        );
    }
}

#[test]
fn gated_revision_counts_a_duplicated_source_once() {
    for k in [1usize, 2, 4, 8] {
        let b = fold_duplicates(k, revise_gated);
        let ratio = posterior_sd(b) / posterior_sd(SOURCE);
        println!("gated k={k}: c {}, sd ratio {ratio:.3}", b.c);
        assert_eq!(b.c, SOURCE.c, "k={k}: duplicate raised confidence");
        assert!((ratio - 1.0).abs() <= 0.05, "k={k}: sd ratio {ratio}");
    }
}

#[test]
fn gated_revision_still_pools_disjoint_sources() {
    let fs = [200u8, 180, 220, 150, 240, 170, 210, 190];
    for k in [2usize, 4, 8] {
        let sources: Vec<Belief> = (0..k)
            .map(|i| Belief {
                f: fs[i],
                c: 128,
                stamp: 1 << i,
            })
            .collect();
        let ungated = sources[1..]
            .iter()
            .fold(sources[0], |a, &s| revise_ungated(a, s));
        let gated = sources[1..]
            .iter()
            .fold(sources[0], |a, &s| revise_gated(a, s));
        println!("disjoint k={k}: c {} (f {})", gated.c, gated.f);
        assert_eq!(gated, ungated, "k={k}: the gate changed disjoint pooling");
        assert!(gated.c > 128, "k={k}: disjoint evidence did not raise c");
    }
}

/// A shared premise reached by two derivation paths overlaps on the stamp,
/// so the gate refuses to pool it, while two genuinely disjoint roots pool.
#[test]
fn shared_premise_is_not_pooled_but_disjoint_roots_are() {
    let premise = Belief {
        f: 200,
        c: 128,
        stamp: 1,
    };
    let other = Belief {
        f: 200,
        c: 128,
        stamp: 2,
    };
    let via_a = revise_gated(premise, Belief { stamp: 4, ..other });
    let via_b = revise_gated(premise, Belief { stamp: 8, ..other });
    let shared = revise_gated(via_a, via_b);
    let disjoint = revise_gated(
        via_a,
        Belief {
            stamp: 16 | 32,
            ..via_b
        },
    );
    println!("shared c {}, disjoint c {}", shared.c, disjoint.c);
    assert_eq!(shared.c, via_a.c.max(via_b.c));
    assert!(disjoint.c > shared.c);
}
