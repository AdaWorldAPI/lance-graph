//! D-NALP-2: the query operations that need no new byte, each checked against
//! a row-at-a-time oracle over the UNSORTED input beliefs.

use lance_graph_contract::ordered_lane::OrderedLaneWitness;
use lance_graph_mask_risc::Lowering;
use lance_graph_nal::{nal, Aperture, Belief, Beliefs, Copula, NalError, Statement};

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
        .map(|i| {
            let r = next();
            Belief {
                s: (r % 40) as u16,
                copula: Copula::ALL[((r >> 8) % 4) as usize],
                p: ((r >> 16) % 40) as u16,
                id: i as u16,
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
    b.statement() == st
}

/// FAILS IF: the statement aperture cares the wrong bytes, or the compiled
/// fold counts anything but the bound range.
#[test]
fn question_count_equals_the_row_oracle() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let mut hits = 0;
    for st in statements() {
        let q = nal::Question::bind(&bound, Aperture::statement(CLASS, st)).unwrap();
        let mut scratch = q.scratch(&bound).unwrap();
        let want = input.iter().filter(|b| same(b, st)).count();
        assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{st:?}");
        // The kept program answers again, unchanged.
        assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{st:?}");
        hits += usize::from(want > 0);
    }
    // Anti-vacuity: the statement set hits and misses.
    assert!(hits > 5 && hits < statements().len(), "hits = {hits}");
}

/// FAILS IF: the aperture byte count does not decide the selection: the class
/// (4 bytes), the subject (6) and the statement (10) of one key must nest, and
/// another class must select nothing.
#[test]
fn the_aperture_width_is_the_selection() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let b = beliefs.bind(&beliefs.witness()).unwrap();
    let st = input[17].statement();
    let key = lance_graph_nal::guid(CLASS, st.s, st.copula, st.p, 0);
    let count = |a: Aperture| {
        let q = nal::Question::bind(&b, a).unwrap();
        q.count(&b, &mut q.scratch(&b).unwrap()).unwrap()
    };
    let class = count(Aperture { key, bytes: 4 });
    let subject = count(Aperture { key, bytes: 6 });
    let statement = count(Aperture { key, bytes: 10 });
    assert_eq!(class, input.len());
    assert_eq!(subject, input.iter().filter(|x| x.s == st.s).count());
    assert_eq!(statement, input.iter().filter(|x| same(x, st)).count());
    assert!(class > subject && subject > statement && statement > 0);
    let other = lance_graph_nal::guid(CLASS + 1, st.s, st.copula, st.p, 0);
    assert_eq!(
        count(Aperture {
            key: other,
            bytes: 4
        }),
        0
    );
    // Zero bytes cared selects every row.
    assert_eq!(
        count(Aperture {
            key: other,
            bytes: 0
        }),
        input.len()
    );
}

/// FAILS IF: the question folds through anything but the fused range, i.e.
/// the coordinate bound did not reach mask-risc as a range.
#[test]
fn question_lowers_to_the_fused_range() {
    let beliefs = sealed(&fixture(2_000));
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let st = Statement {
        s: 3,
        copula: Copula::Inheritance,
        p: 5,
    };
    let q = nal::Question::bind(&bound, Aperture::statement(CLASS, st)).unwrap();
    assert!(
        matches!(q.lowering(), Lowering::Range(_)),
        "{:?}",
        q.lowering()
    );
}

/// FAILS IF: choice is not the highest confidence of the selection, or reports
/// one for an empty selection.
#[test]
fn choice_is_the_highest_confidence() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let (mut some, mut none) = (0, 0);
    for st in statements() {
        let want = input
            .iter()
            .filter(|b| same(b, st))
            .map(|b| i32::from(b.c))
            .max();
        let q = nal::Choice::bind(&bound, Aperture::statement(CLASS, st)).unwrap();
        let got = q.max(&bound, &mut q.scratch(&bound).unwrap()).unwrap();
        assert_eq!(got, want, "{st:?}");
        if want.is_some() {
            some += 1;
        } else {
            none += 1;
        }
    }
    assert!(some > 0 && none > 0);
}

/// FAILS IF: the confidence threshold is off by one, inverted, ignores the
/// selection, or reads outside the selection's range.
#[test]
fn truth_at_least_equals_the_row_oracle() {
    let input = fixture(5_000);
    let beliefs = sealed(&input);
    let bound = beliefs.bind(&beliefs.witness()).unwrap();
    let mut partial = 0;
    let selections = statements()
        .into_iter()
        .map(|st| {
            (
                Aperture::statement(CLASS, st),
                Box::new(move |b: &Belief| same(b, st)) as Box<dyn Fn(&Belief) -> bool>,
            )
        })
        .chain([0u16, 3, 39, 40].into_iter().map(|s| {
            (
                Aperture::subject(CLASS, s),
                Box::new(move |b: &Belief| b.s == s) as Box<dyn Fn(&Belief) -> bool>,
            )
        }));
    for (a, sel) in selections {
        for t in [0u8, 1, 100, 204, 255] {
            let q = nal::TruthAtLeast::bind(&bound, a, t).unwrap();
            let mut scratch = q.scratch(&bound).unwrap();
            let all = input.iter().filter(|b| sel(b)).count();
            let want = input.iter().filter(|b| sel(b) && b.c >= t).count();
            assert_eq!(q.count(&bound, &mut scratch).unwrap(), want, "{a:?} t={t}");
            partial += usize::from(want > 0 && want < all);
        }
    }
    // Anti-vacuity: some thresholds cut inside a selection.
    assert!(partial > 0);
}

/// FAILS IF: the empty and the full population answer wrongly.
#[test]
fn empty_and_full_populations() {
    let st = Statement {
        s: 1,
        copula: Copula::Similarity,
        p: 2,
    };
    let a = Aperture::statement(CLASS, st);
    let empty = sealed(&[]);
    let b = empty.bind(&empty.witness()).unwrap();
    let q = nal::Question::bind(&b, a).unwrap();
    assert_eq!(q.count(&b, &mut q.scratch(&b).unwrap()).unwrap(), 0);
    let q = nal::Choice::bind(&b, a).unwrap();
    assert_eq!(q.max(&b, &mut q.scratch(&b).unwrap()).unwrap(), None);

    let one: Vec<Belief> = (0..300)
        .map(|i| Belief {
            s: 1,
            copula: Copula::Similarity,
            p: 2,
            id: i,
            f: 9,
            c: 200,
        })
        .collect();
    let full = sealed(&one);
    let b = full.bind(&full.witness()).unwrap();
    let q = nal::TruthAtLeast::bind(&b, a, 200).unwrap();
    assert_eq!(q.count(&b, &mut q.scratch(&b).unwrap()).unwrap(), 300);
    let q = nal::TruthAtLeast::bind(&b, a, 201).unwrap();
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
