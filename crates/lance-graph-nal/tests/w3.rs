//! D-NALP-3: revision as one grouped cross-moment fold over the weight and
//! frequency lanes, checked against an independent scalar oracle (ONA's
//! `Truth_Revision`, transcribed from `src/Truth.c` at
//! `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a`) and against the values
//! OpenNARS printed for `revision_points.nal` (`oracles.md`).

use lance_graph_nal::{
    nal, weight, Belief, Beliefs, Copula, Evidence, NalError, Revised, Statement, HORIZON,
    MAX_CONFIDENCE_CODE, WEIGHT_SCALE,
};

const CLASS: u32 = 0x0E01_0001;

// --- the independent oracle -------------------------------------------------

/// ONA's truth, in reals.
#[derive(Debug, Clone, Copy)]
struct Truth {
    f: f64,
    c: f64,
}

/// ONA `Truth_c2w` with `TRUTH_EVIDENTIAL_HORIZON = 1.0` (`Config.h:168`).
fn c2w(c: f64) -> f64 {
    c / (1.0 - c)
}

/// ONA `Truth_w2c`.
fn w2c(w: f64) -> f64 {
    w / (w + 1.0)
}

/// ONA `Truth_Revision` (`Truth.c:46-75`), `MAX_CONFIDENCE = 0.99`
/// (`Config.h:172`): the pairwise law, folded left over the evidence.
fn ona_revision(a: Truth, b: Truth) -> Truth {
    if a.c == 0.0 {
        return b;
    }
    if b.c == 0.0 {
        return a;
    }
    let (w1, w2) = (c2w(a.c), c2w(b.c));
    let w = w1 + w2;
    Truth {
        f: ((w1 * a.f + w2 * b.f) / w).min(1.0),
        c: w2c(w).max(a.c).max(b.c).min(0.99),
    }
}

fn code(x: f64) -> u8 {
    (x * 255.0).round() as u8
}

fn real(k: u8) -> f64 {
    f64::from(k) / 255.0
}

/// The oracle answer for evidence given in codes, folded pairwise.
fn oracle(evidence: &[(u8, u8)]) -> Option<(u8, u8)> {
    let mut it = evidence.iter().filter(|e| e.1 > 0);
    let first = it.next()?;
    let t = it.fold(
        Truth {
            f: real(first.0),
            c: real(first.1),
        },
        |acc, &(f, c)| {
            ona_revision(
                acc,
                Truth {
                    f: real(f),
                    c: real(c),
                },
            )
        },
    );
    Some((code(t.f), code(t.c)))
}

// --- fixtures ---------------------------------------------------------------

fn st(s: u16) -> Statement {
    Statement {
        s,
        copula: Copula::Inheritance,
        p: s + 100,
    }
}

fn belief(s: u16, id: u16, f: f64, c: f64) -> Belief {
    let t = st(s);
    Belief {
        s: t.s,
        copula: t.copula,
        p: t.p,
        id,
        f: code(f),
        c: code(c),
    }
}

fn revise(beliefs: &Beliefs, s: Statement) -> Result<Option<Revised>, NalError> {
    let b = beliefs.bind(&beliefs.witness()).unwrap();
    let r = nal::Revision::bind(&b, s)?;
    r.revise(&b, &mut r.scratch(&b).unwrap())
}

fn within_one(got: u8, want: u8) -> bool {
    got.abs_diff(want) <= 1
}

// --- agreement --------------------------------------------------------------

/// The three pairs of `revision_points.nal` (`<a1 --> b1>.` is `(1.0, 0.9)`).
/// `(f, c)` in reals.
type Fc = (f64, f64);

/// Two inputs and the revised truth OpenNARS printed.
const POINTS: [(Fc, Fc, Fc); 3] = [
    ((1.0, 0.9), (0.1, 0.6), (0.87, 0.91)),
    ((0.8, 0.5), (0.3, 0.7), (0.45, 0.77)),
    ((0.0, 0.9), (1.0, 0.9), (0.50, 0.95)),
];

/// FAILS IF: the compiled revision disagrees with ONA's pairwise law by more
/// than one code, or with the truth OpenNARS printed (two decimals) by more
/// than the printing plus one code.
#[test]
fn revision_agrees_with_ona_and_opennars() {
    let roots = [[1u64], [2], [3], [4], [5], [6]];
    let mut evidence = Vec::new();
    for (i, &(a, b, _)) in POINTS.iter().enumerate() {
        let s = i as u16;
        evidence.push(Evidence {
            belief: belief(s, 0, a.0, a.1),
            stamp: &roots[2 * i],
        });
        evidence.push(Evidence {
            belief: belief(s, 1, b.0, b.1),
            stamp: &roots[2 * i + 1],
        });
    }
    let beliefs = Beliefs::seal_evidence(CLASS, &evidence, 3).unwrap();
    for (i, &(a, b, printed)) in POINTS.iter().enumerate() {
        let got = revise(&beliefs, st(i as u16)).unwrap().unwrap();
        let input = [(code(a.0), code(a.1)), (code(b.0), code(b.1))];
        let (of, oc) = oracle(&input).unwrap();
        assert!(within_one(got.f, of), "point {i}: f {} vs ona {of}", got.f);
        assert!(within_one(got.c, oc), "point {i}: c {} vs ona {oc}", got.c);
        let tol = 0.005 + 1.0 / 255.0;
        assert!((real(got.f) - printed.0).abs() <= tol, "point {i}: {got:?}");
        assert!((real(got.c) - printed.1).abs() <= tol, "point {i}: {got:?}");
        assert_eq!(got.rows, 2);
        assert_eq!(got.duplicates, 0);
    }
}

/// SplitMix64.
fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// FAILS IF: on any statement the fold's sums differ from the exact integer
/// sums of the stored weights (`Σw`, `Σw·f`), or its truth differs from ONA's
/// pairwise law by more than one code.
///
/// The agreement with ONA holds on this fixture; where ONA's running
/// confidence saturates mid-fold the two laws part, see
/// `saturated_evidence_is_the_exact_sum_where_onas_fold_depends_on_order`.
#[test]
fn revision_is_the_exact_integer_sum_and_agrees_on_every_statement() {
    let mut next = rng(0xC0FF_EE00);
    let n_stmt = 60u16;
    let mut rows: Vec<(Belief, [u64; 2])> = Vec::new();
    let mut root = 1u64;
    for s in 0..n_stmt {
        // 0..=5 pieces of evidence per statement; statement 7 has none.
        let k = if s == 7 { 0 } else { (next() % 6) as u16 };
        for id in 0..k {
            let r = next();
            let mut b = belief(s, id, 0.0, 0.0);
            b.f = (r >> 8) as u8;
            b.c = ((r >> 16) % 255) as u8; // never 255
            rows.push((b, [root, root + 1]));
            root += 2;
        }
    }
    let evidence: Vec<Evidence<'_>> = rows
        .iter()
        .map(|(b, stamp)| Evidence { belief: *b, stamp })
        .collect();
    let beliefs = Beliefs::seal_evidence(CLASS, &evidence, 9).unwrap();
    let (mut multi, mut none) = (0, 0);
    for s in 0..n_stmt {
        let mine: Vec<&Belief> = rows.iter().map(|r| &r.0).filter(|b| b.s == s).collect();
        let got = revise(&beliefs, st(s)).unwrap();
        let w: i64 = mine.iter().map(|b| i64::from(weight(b.c).unwrap())).sum();
        let wf: i128 = mine
            .iter()
            .map(|b| i128::from(weight(b.c).unwrap()) * i128::from(b.f))
            .sum();
        let want = oracle(&mine.iter().map(|b| (b.f, b.c)).collect::<Vec<_>>());
        match (got, want) {
            (Some(g), Some((of, oc))) => {
                assert_eq!(g.w, w, "s{s}");
                assert_eq!(g.wf, wf, "s{s}");
                assert_eq!(g.rows, mine.len() as u64, "s{s}");
                assert!(within_one(g.f, of), "s{s}: f {} vs {of}", g.f);
                assert!(within_one(g.c, oc), "s{s}: c {} vs {oc}", g.c);
                multi += usize::from(mine.len() >= 3);
            }
            (None, None) => none += 1,
            other => panic!("s{s}: {other:?}"),
        }
    }
    // Anti-vacuity: statements with three or more pieces, and empty ones.
    assert!(multi >= 10 && none >= 1, "multi {multi}, none {none}");
}

/// FAILS IF: confidence is not capped at ONA's `MAX_CONFIDENCE` (0.99).
#[test]
fn confidence_is_capped_where_ona_caps_it() {
    let roots: Vec<[u64; 1]> = (1..=20).map(|r| [r]).collect();
    let evidence: Vec<Evidence<'_>> = roots
        .iter()
        .enumerate()
        .map(|(i, stamp)| Evidence {
            belief: belief(0, i as u16, 1.0, 0.9),
            stamp,
        })
        .collect();
    let beliefs = Beliefs::seal_evidence(CLASS, &evidence, 1).unwrap();
    let got = revise(&beliefs, st(0)).unwrap().unwrap();
    assert_eq!(got.c, MAX_CONFIDENCE_CODE);
    assert_eq!(
        oracle(&[(255, code(0.9)); 20]).unwrap().1,
        MAX_CONFIDENCE_CODE
    );
    // Non-vacuity: uncapped, twenty pieces at 0.9 give a confidence above it.
    let w = 20.0 * c2w(real(code(0.9)));
    assert!(code(w2c(w)) > MAX_CONFIDENCE_CODE);
    assert_eq!((WEIGHT_SCALE, HORIZON), (1 << 16, 1));
}

/// Ten positive and ten negative pieces at `c = 230`, each with its own root.
fn mixed(positive_first: bool) -> Vec<(u8, u8)> {
    let (pos, neg) = ([(255u8, 230u8); 10], [(0u8, 230u8); 10]);
    if positive_first {
        pos.iter().chain(neg.iter()).copied().collect()
    } else {
        neg.iter().chain(pos.iter()).copied().collect()
    }
}

/// A documented divergence, not an agreement. Once the running confidence
/// reaches ONA's 0.99 cap, ONA's pairwise law turns the capped confidence back
/// into a bounded weight for the next step, so its answer depends on the order
/// the evidence arrives in. The fold sums every weight once and caps only the
/// result, so it is order-independent: the exact NAL revision `w = Σwᵢ`.
///
/// FAILS IF: the fold depends on the sealed order, stops being the exact
/// weighted mean, or ONA's pairwise fold stops disagreeing with itself on this
/// input (then the fixture no longer reaches saturation and pins nothing).
#[test]
fn saturated_evidence_is_the_exact_sum_where_onas_fold_depends_on_order() {
    let roots: Vec<[u64; 1]> = (1..=20).map(|r| [r]).collect();
    let mut answers = Vec::new();
    for positive_first in [true, false] {
        let evidence: Vec<Evidence<'_>> = mixed(positive_first)
            .iter()
            .zip(&roots)
            .enumerate()
            .map(|(i, (&(f, c), stamp))| Evidence {
                belief: Belief {
                    f,
                    c,
                    ..belief(0, i as u16, 0.0, 0.0)
                },
                stamp,
            })
            .collect();
        let beliefs = Beliefs::seal_evidence(CLASS, &evidence, 1).unwrap();
        let got = revise(&beliefs, st(0)).unwrap().unwrap();
        answers.push((got.f, got.c));
    }
    // Order-independent, and the weighted mean of equal weights at 255 and 0.
    assert_eq!(answers[0], answers[1]);
    assert_eq!(answers[0], (128, MAX_CONFIDENCE_CODE));
    // ONA's pairwise fold disagrees with itself on the same multiset.
    assert_eq!(oracle(&mixed(true)), Some((104, MAX_CONFIDENCE_CODE)));
    assert_eq!(oracle(&mixed(false)), Some((151, MAX_CONFIDENCE_CODE)));
}

// --- evidential independence ------------------------------------------------

/// The backup's counterexample: stamp 10 on statement 0 twice.
fn repeated() -> [(Belief, &'static [u64]); 3] {
    [
        (belief(0, 0, 1.0, 0.9), &[10]),
        (belief(0, 1, 0.1, 0.6), &[11]),
        (belief(0, 2, 1.0, 0.9), &[10]),
    ]
}

fn seal(rows: &[(Belief, &[u64])]) -> Result<Beliefs, NalError> {
    let ev: Vec<Evidence<'_>> = rows
        .iter()
        .map(|&(belief, stamp)| Evidence { belief, stamp })
        .collect();
    Beliefs::seal_evidence(CLASS, &ev, 2)
}

/// FAILS IF: the same evidence (same stamp, same truth) is counted twice.
/// Counting it twice would give c = 0.951, not OpenNARS's 0.91.
#[test]
fn an_exact_duplicate_counts_once() {
    let rows = repeated();
    let got = revise(&seal(&rows).unwrap(), st(0)).unwrap().unwrap();
    let once = revise(&seal(&rows[..2]).unwrap(), st(0)).unwrap().unwrap();
    assert_eq!(got.duplicates, 1);
    assert_eq!(got.rows, 2);
    assert_eq!(
        (got.f, got.c, got.w, got.wf),
        (once.f, once.c, once.w, once.wf)
    );
    // Non-vacuity: summing all three rows inflates the confidence.
    let inflated = oracle(&[(255, code(0.9)), (code(0.1), code(0.6)), (255, code(0.9))])
        .unwrap()
        .1;
    assert!(inflated >= got.c + 8, "{inflated} vs {}", got.c);
}

/// FAILS IF: partially overlapping evidence, or one stamp carrying two truth
/// values, is revised instead of refused. Neither has an exact revision.
#[test]
fn overlapping_evidence_is_refused() {
    let partial = [
        (belief(0, 0, 1.0, 0.9), &[1u64, 2][..]),
        (belief(0, 1, 0.1, 0.6), &[2, 3][..]),
    ];
    assert_eq!(
        revise(&seal(&partial).unwrap(), st(0)),
        Err(NalError::DependentEvidence { rows: 1 })
    );
    let disagreeing = [
        (belief(0, 0, 1.0, 0.9), &[7u64][..]),
        (belief(0, 1, 0.2, 0.9), &[7][..]),
    ];
    assert_eq!(
        revise(&seal(&disagreeing).unwrap(), st(0)),
        Err(NalError::DependentEvidence { rows: 1 })
    );
    // The same stamp on two DIFFERENT statements is not overlap.
    let across = [
        (belief(0, 0, 1.0, 0.9), &[5u64][..]),
        (belief(1, 0, 0.1, 0.6), &[5][..]),
    ];
    let b = seal(&across).unwrap();
    assert!(revise(&b, st(0)).unwrap().is_some());
    assert!(revise(&b, st(1)).unwrap().is_some());
}

/// FAILS IF: evidence without provenance is revised. A population sealed
/// without stamps answers the W2 queries but refuses revision.
#[test]
fn evidence_without_provenance_is_refused() {
    let plain = Beliefs::seal(CLASS, &[belief(0, 0, 1.0, 0.9), belief(0, 1, 0.1, 0.6)], 1).unwrap();
    assert_eq!(
        revise(&plain, st(0)),
        Err(NalError::MissingProvenance { rows: 2 })
    );
    let mixed = [
        (belief(0, 0, 1.0, 0.9), &[1u64][..]),
        (belief(0, 1, 0.1, 0.6), &[][..]),
    ];
    assert_eq!(
        revise(&seal(&mixed).unwrap(), st(0)),
        Err(NalError::MissingProvenance { rows: 1 })
    );
}

/// FAILS IF: a stamp carrying one root twice is accepted. ONA's
/// reasoningbench answer #9 (`c = 0.815686`, stamp `[10,2,11,2,1]`,
/// `oracles.md`) carries root 2 twice; ONA itself drops such events at input
/// (`Memory.c:501`, `Stamp_hasDuplicate`).
#[test]
fn a_stamp_with_a_repeated_root_is_refused_at_seal() {
    let rows = [(belief(0, 0, 1.0, 0.815686), &[10u64, 2, 11, 2, 1][..])];
    assert!(matches!(seal(&rows), Err(NalError::InvalidStamp { .. })));
    let rows = [(belief(0, 0, 1.0, 0.815686), &[10u64, 2, 11, 1][..])];
    assert!(seal(&rows).is_ok());
}

/// FAILS IF: a confidence of 1 (unbounded weight) is stored instead of refused.
#[test]
fn unbounded_weight_is_refused_at_write() {
    let mut b = belief(0, 0, 1.0, 0.9);
    b.c = 255;
    assert!(matches!(
        seal(&[(b, &[1u64][..])]),
        Err(NalError::UnboundedWeight { .. })
    ));
    assert_eq!(weight(255), None);
    assert_eq!(weight(254), Some(254 * WEIGHT_SCALE));
    assert_eq!(weight(0), Some(0));
}

/// FAILS IF: no evidence answers a truth. An absent statement and one with
/// only zero-confidence evidence both answer `None`.
#[test]
fn no_evidence_answers_none() {
    let rows = [(belief(0, 0, 1.0, 0.0), &[1u64][..])];
    let b = seal(&rows).unwrap();
    assert_eq!(revise(&b, st(0)), Ok(None));
    assert_eq!(revise(&b, st(5)), Ok(None));
}

/// FAILS IF: a revision bound on one population runs on another.
#[test]
fn a_prepared_revision_refuses_another_population() {
    let a = seal(&repeated()).unwrap();
    // Same size, different keys: the second belief of statement 0 moves to
    // statement 1, so the kept range names other rows there.
    let mut other = repeated();
    other[1].0 = belief(1, 1, 0.1, 0.6);
    let b = seal(&other).unwrap();
    assert_eq!(a.witness().n_rows(), b.witness().n_rows());
    let (ba, bb) = (a.bind(&a.witness()).unwrap(), b.bind(&b.witness()).unwrap());
    let r = nal::Revision::bind(&ba, st(0)).unwrap();
    assert!(matches!(
        r.revise(&bb, &mut r.scratch(&bb).unwrap()),
        Err(NalError::WrongPopulation)
    ));
    assert!(r.revise(&ba, &mut r.scratch(&ba).unwrap()).is_ok());
}
