//! D-NALP-2: the query operations that need no new byte, each checked against
//! a row-at-a-time oracle over the UNSORTED input beliefs.

use lance_graph_contract::ordered_lane::OrderedLaneWitness;
use lance_graph_mask_risc::Lowering;
use lance_graph_nal::{nal, Belief, Beliefs, Copula, NalError, Statement};

const CLASS: u32 = 0x0E01_0001;

/// Deterministic beliefs: SplitMix64 over 40 terms, 4 copulas, every truth code.
fn fixture(n: usize) -> Vec<Belief> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..n)
        .map(|_| {
            let r = next();
            Belief {
                s: (r % 40) as u16,
                copula: Copula::ALL[((r >> 8) % 4) as usize],
                p: ((r >> 16) % 40) as u16,
                f: (r >> 24) as u8,
                c: (r >> 32) as u8,
            }
        })
        .collect()
}

fn sealed(input: &[Belief]) -> Beliefs {
    Beliefs::seal(CLASS, input, 7).expect("seal")
}

fn statements() -> Vec<Statement> {
    let mut v = Vec::new();
    for s in [0u16, 3, 17, 39, 40, 999] {
        for copula in Copula::ALL {
            for p in [0u16, 5, 39, 41] {
                v.push(Statement { s, copula, p });
            }
        }
    }
    v
}

fn same(b: &Belief, st: Statement) -> bool {
    b.s == st.s && b.copula == st.copula && b.p == st.p
}

/// FAILS IF: the statement range is bound on the wrong tiles, or the compiled
/// fold counts anything but the bound range.
#[test]
fn question_count_equals_the_row_oracle() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let mut hits = 0;
    for st in statements() {
        let q = nal::Question::bind(&bound, st).unwrap();
        let mut scratch = q.scratch(&bound).unwrap();
        let want = input.iter().filter(|b| same(b, st)).count();
        assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{st:?}");
        // The prepared program answers again, unchanged.
        assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{st:?}");
        hits += usize::from(want > 0);
    }
    // Anti-vacuity: the statement set hits and misses.
    assert!(hits > 5 && hits < statements().len(), "hits = {hits}");
}

/// FAILS IF: the question folds through anything but the fused range, i.e.
/// the coordinate bound did not reach mask-risc as a range.
#[test]
fn question_lowers_to_the_fused_range() {
    let beliefs = sealed(&fixture(2_000));
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let q = nal::Question::bind(
        &bound,
        Statement {
            s: 3,
            copula: Copula::Inheritance,
            p: 5,
        },
    )
    .unwrap();
    assert!(
        matches!(q.lowering(), Lowering::Range(_)),
        "{:?}",
        q.lowering()
    );
}

/// FAILS IF: choice is not the highest-confidence belief of the statement, or
/// reports one for a statement with no belief.
#[test]
fn choice_is_the_highest_confidence_belief() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let (mut some, mut none) = (0, 0);
    for st in statements() {
        let want = input.iter().filter(|b| same(b, st)).map(|b| b.c).max();
        let got = nal::Choice::of(&bound, st);
        assert_eq!(got.map(|b| b.c), want, "{st:?}");
        if let Some(b) = got {
            assert!(same(&b, st));
            assert!(input.contains(&b), "choice must be a stored belief");
            some += 1;
        } else {
            none += 1;
        }
    }
    assert!(some > 0 && none > 0);
}

/// FAILS IF: the confidence threshold is off by one, inverted, or ignores the
/// statement.
#[test]
fn truth_at_least_equals_the_row_oracle() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let mut partial = 0;
    for st in statements() {
        for t in [0u8, 1, 100, 204, 255] {
            let q = nal::TruthAtLeast::bind(&bound, st, t).unwrap();
            let mut scratch = q.scratch(&bound).unwrap();
            let all = input.iter().filter(|b| same(b, st)).count();
            let want = input.iter().filter(|b| same(b, st) && b.c >= t).count();
            assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{st:?} t={t}");
            partial += usize::from(want > 0 && want < all);
        }
    }
    // Anti-vacuity: some thresholds cut inside a statement.
    assert!(partial > 0);
}

/// FAILS IF: support over a subject prefix miscounts.
#[test]
fn support_counts_every_belief_about_a_subject() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    for s in [0u16, 3, 39, 40] {
        let q = nal::Support::bind(&bound, s).unwrap();
        let mut scratch = q.scratch(&bound).unwrap();
        let want = input.iter().filter(|b| b.s == s).count();
        assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "s={s}");
    }
}

/// FAILS IF: the gated confidence predicate over a subject prefix (which is not
/// one interval) miscounts.
#[test]
fn truth_at_least_across_a_subject_equals_the_row_oracle() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    for s in [0u16, 3, 39, 40] {
        for t in [0u8, 128, 204, 255] {
            let q = nal::TruthAtLeastAcross::bind(&bound, s, t).unwrap();
            let mut scratch = q.scratch(&bound).unwrap();
            let want = input.iter().filter(|b| b.s == s && b.c >= t).count();
            assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "s={s} t={t}");
        }
    }
}

/// FAILS IF: the empty and the full population answer wrongly.
#[test]
fn empty_and_full_populations() {
    let empty = sealed(&[]);
    let b = empty.bind(&empty.witness()).unwrap();
    let st = Statement {
        s: 1,
        copula: Copula::Similarity,
        p: 2,
    };
    let q = nal::Question::bind(&b, st).unwrap();
    assert_eq!(q.count(&b, &mut q.scratch(&b).unwrap()).unwrap(), 0);
    assert_eq!(nal::Choice::of(&b, st), None);

    let one = vec![
        Belief {
            s: 1,
            copula: Copula::Similarity,
            p: 2,
            f: 9,
            c: 200
        };
        300
    ];
    let full = sealed(&one);
    let b = full.bind(&full.witness()).unwrap();
    let q = nal::TruthAtLeast::bind(&b, st, 200).unwrap();
    assert_eq!(q.count(&b, &mut q.scratch(&b).unwrap()).unwrap(), 300);
    let q = nal::TruthAtLeast::bind(&b, st, 201).unwrap();
    assert_eq!(q.count(&b, &mut q.scratch(&b).unwrap()).unwrap(), 0);
}

/// FAILS IF: a witness from another seal binds. A stale witness must make the
/// coordinate path unavailable, never answer from the wrong order.
#[test]
fn a_stale_witness_refuses_to_bind() {
    let beliefs = sealed(&fixture(1_000));
    let forged = OrderedLaneWitness::forged(7, beliefs.witness().n_rows(), 0xDEAD);
    assert!(matches!(beliefs.bind(&forged), Err(NalError::Witness(_))));
}
