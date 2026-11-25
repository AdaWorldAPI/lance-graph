//! The exact-decimal contract, checked against independent exact oracles
//! (decimal text and i128 rationals), never against `f64`.

use exact_decimal_probe::*;
use lance_graph_contract::class_view::ClassView;
use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgType};
use lance_graph_intake::{ColumnConfig, ShapeConfig, Table};
use lance_graph_mask_risc::{words_for, Out};
use lance_graph_vrl::{Recipe, Witness};
use std::collections::BTreeMap;

/// `"1.20"` and `"1.2"` are one number; trailing zeros of an integer are a
/// coarser exponent; a negative zero is the established zero.
#[test]
fn canonical_reads_one_number_per_value() {
    assert_eq!(canonical("1.20"), canonical("1.2"));
    assert_eq!(canonical("1.2").unwrap(), (12, Exp(-1)));
    assert_eq!(canonical("25000").unwrap(), (25, Exp(3)));
    assert_eq!(canonical("-0.00").unwrap(), (0, Exp(0)));
    assert_eq!(canonical("007.50").unwrap(), (75, Exp(-1)));
    for bad in ["", ".", "1e3", "1.2.3", "abc", "1,5", "--1"] {
        assert!(
            matches!(canonical(bad), Err(DecimalRefusal::Malformed(_))),
            "{bad:?} accepted"
        );
    }
}

/// Padding is exact, extra zero digits are exact, a nonzero digit past the
/// exponent is refused, and the i32 bounds are exact at scale 2.
#[test]
fn parse_exact_pads_refuses_rounding_and_overflow() {
    let s2 = Exp::scale(2);
    assert_eq!(parse_exact("1.2", s2), Ok(120));
    assert_eq!(parse_exact("1.200", s2), Ok(120));
    assert_eq!(parse_exact("-0.50", s2), Ok(-50));
    assert_eq!(parse_exact("-0", s2), Ok(0));
    assert!(matches!(
        parse_exact("1.205", s2),
        Err(DecimalRefusal::NotExact { .. })
    ));
    assert_eq!(parse_exact("21474836.47", s2), Ok(i32::MAX));
    assert_eq!(parse_exact("-21474836.48", s2), Ok(i32::MIN));
    for over in ["21474836.48", "-21474836.49"] {
        assert!(
            matches!(parse_exact(over, s2), Err(DecimalRefusal::Overflow { .. })),
            "{over}"
        );
    }
    // A coarser exponent: thousands.
    assert_eq!(parse_exact("25000", Exp(3)), Ok(25));
    assert!(matches!(
        parse_exact("25001", Exp(3)),
        Err(DecimalRefusal::NotExact { .. })
    ));
}

/// Presentation precision is its own parameter: 250 at exponent 0 shows as
/// 250.00, and cutting a nonzero digit is refused rather than rounded.
#[test]
fn presentation_is_independent_of_scale() {
    assert_eq!(format(250, Exp(0), 2).unwrap(), "250.00");
    assert_eq!(format(125, Exp(-2), 2).unwrap(), "1.25");
    assert_eq!(format(125, Exp(-2), 4).unwrap(), "1.2500");
    assert_eq!(format(-5, Exp(-2), 2).unwrap(), "-0.05");
    assert_eq!(format(0, Exp(-2), 2).unwrap(), "0.00");
    assert_eq!(format(25, Exp(3), 0).unwrap(), "25000");
    assert_eq!(format(1200, Exp(-3), 1).unwrap(), "1.2");
    assert!(matches!(
        format(125, Exp(-2), 1),
        Err(DecimalRefusal::PresentationCuts { .. })
    ));
}

/// An `f64` converts only under the named policy, and the result says so;
/// a binary artefact (0.1 + 0.2) is refused at scale 2, not rounded.
#[test]
fn f64_conversion_is_an_explicit_policy() {
    let s2 = Exp::scale(2);
    let c = from_f64(37.5, s2, F64Policy::ShortestRoundTrip).unwrap();
    assert_eq!(c.coefficient, 3750);
    assert_eq!(c.provenance, Provenance::F64ShortestRoundTrip);
    assert_eq!(
        from_lexical("37.5", s2).unwrap().provenance,
        Provenance::Lexical
    );
    assert!(matches!(
        from_f64(0.1 + 0.2, s2, F64Policy::ShortestRoundTrip),
        Err(DecimalRefusal::NotExact { .. })
    ));
    assert_eq!(
        from_f64(f64::NAN, s2, F64Policy::ShortestRoundTrip),
        Err(DecimalRefusal::NotFinite)
    );
}

/// `c/10^s <op> L` compared exactly as rationals.
fn oracle(c: i64, exp: Exp, op: Op, literal: &str) -> bool {
    let (lc, le) = canonical(literal).unwrap();
    // Both sides in units of 10^m, m = min exponent.
    let m = exp.0.min(le.0);
    let lhs = i128::from(c) * 10i128.pow((exp.0 - m) as u32);
    let rhs = lc * 10i128.pow((le.0 - m) as u32);
    match op {
        Op::Eq => lhs == rhs,
        Op::Ne => lhs != rhs,
        Op::Lt => lhs < rhs,
        Op::Le => lhs <= rhs,
        Op::Gt => lhs > rhs,
        Op::Ge => lhs >= rhs,
    }
}

fn holds(t: Threshold, c: i32) -> bool {
    match t {
        Threshold::Never => false,
        Threshold::EveryValue => true,
        Threshold::Coefficient(o, v) => match o {
            Op::Eq => c == v,
            Op::Ne => c != v,
            Op::Lt => c < v,
            Op::Le => c <= v,
            Op::Gt => c > v,
            Op::Ge => c >= v,
        },
    }
}

/// Every operator, against literals on, between and far past the
/// representable coefficients, agrees with the exact rational oracle for
/// every coefficient in range. Anti-vacuity: some literal is not
/// representable, and every outcome kind occurs.
#[test]
fn every_threshold_equals_the_exact_rational_oracle() {
    let literals = [
        "1.205",
        "1.20",
        "1.2",
        "-0.005",
        "0",
        "-0",
        "2.5",
        "-1.995",
        "0.009",
        "99999999999",
        "-99999999999",
        "3",
    ];
    let ops = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge];
    let mut kinds = [false; 3];
    for exp in [Exp::scale(2), Exp::scale(0), Exp(1)] {
        for lit in literals {
            for op in ops {
                let t = threshold(op, lit, exp).unwrap();
                kinds[match t {
                    Threshold::Coefficient(..) => 0,
                    Threshold::Never => 1,
                    Threshold::EveryValue => 2,
                }] = true;
                for c in -400..=400 {
                    assert_eq!(
                        holds(t, c),
                        oracle(i64::from(c), exp, op, lit),
                        "{c} at {exp:?} {op:?} {lit}"
                    );
                }
                for c in [i32::MIN, i32::MIN + 1, i32::MAX - 1, i32::MAX] {
                    assert_eq!(
                        holds(t, c),
                        oracle(i64::from(c), exp, op, lit),
                        "{c} at {exp:?} {op:?} {lit}"
                    );
                }
            }
        }
    }
    assert_eq!(kinds, [true; 3], "every outcome kind must occur");
}

/// Sealing a slice removes redundant powers of ten before choosing the
/// width: whole amounts declared at scale 2 seal coarser and fit a narrower
/// width than at the field's fixed scale. A slice never seals finer than its
/// field's declared resolution.
#[test]
fn sealing_removes_redundant_powers_of_ten_before_the_width() {
    let s2 = Exp::scale(2);
    let mixed = seal(
        &[
            Some("250.00"),
            Some("100"),
            Some("37.5"),
            None,
            Some("0.00"),
        ],
        s2,
    )
    .unwrap();
    assert_eq!(mixed.exp, Exp(-1));
    assert_eq!(mixed.coefficients, [2500, 1000, 375, 0, 0]);
    assert_eq!(mixed.present, [true, true, true, false, true]);
    assert_eq!(mixed.width, Some(PmWidth::Pm17));

    let whole = seal(&[Some("250.00"), Some("100"), Some("1000")], s2).unwrap();
    assert_eq!(whole.exp, Exp(1));
    assert_eq!(whole.coefficients, [25, 10, 100]);
    assert_eq!(whole.width, Some(PmWidth::Pm9));
    // The same values at the field's fixed scale 2 need PM25.
    let fixed_max = ["250.00", "100", "1000"]
        .iter()
        .map(|t| parse_exact(t, s2).unwrap().unsigned_abs())
        .max()
        .unwrap();
    assert_eq!(
        PmWidth::for_magnitude(u64::from(fixed_max)),
        Some(PmWidth::Pm25)
    );

    assert!(matches!(
        seal(&[Some("0.001")], s2),
        Err(DecimalRefusal::NotExact { .. })
    ));
    assert_eq!(seal(&[None, Some("0")], s2).unwrap().exp, Exp(0));
}

/// The exact rational value of a cell, at exponent `t`.
fn at(text: &str, t: Exp) -> i128 {
    let (c, e) = canonical(text).unwrap();
    c * 10i128.pow((e.0 - t.0) as u32)
}

/// Per-slice N, S, Q from one mask-risc pass, combined by aligning the
/// aggregates, equal the statistics of every value rescaled one by one. `n`
/// counts established values (an explicit 0.00 counts, a blank does not),
/// and the alignments are one per slice, not one per value.
#[test]
fn slice_statistics_combine_without_rescaling_the_values() {
    let s3 = Exp::scale(3);
    let slices: [&[Option<&str>]; 3] = [
        &[Some("250.00"), Some("100"), None, Some("1000"), Some("-50")],
        &[
            Some("37.5"),
            Some("-12.25"),
            Some("0.00"),
            None,
            Some("0.125"),
        ],
        &[
            Some("-1"),
            Some("2.5"),
            Some("40"),
            Some("0.5"),
            Some("-0.75"),
        ],
    ];
    let sealed: Vec<SealedSlice> = slices.iter().map(|s| seal(s, s3).unwrap()).collect();
    assert!(
        sealed.windows(2).any(|w| w[0].exp != w[1].exp),
        "slices must differ in exponent"
    );
    let mut coef = Vec::new();
    let mut slice_of = Vec::new();
    let mut present = Vec::new();
    for (k, sl) in sealed.iter().enumerate() {
        coef.extend(&sl.coefficients);
        present.extend(&sl.present);
        slice_of.extend(std::iter::repeat_n(k as u32, sl.coefficients.len()));
    }
    let mut plane = vec![0u64; words_for(coef.len())];
    for (i, &p) in present.iter().enumerate() {
        if p {
            plane[i / 64] |= 1 << (i % 64);
        }
    }
    let sums = slice_sums(&coef, &plane, &slice_of, sealed.len()).unwrap();
    let stats: Vec<SliceStats> = sealed
        .iter()
        .zip(&sums)
        .map(|(sl, s)| SliceStats {
            exp: sl.exp,
            sums: *s,
        })
        .collect();
    let got = combine(&stats).unwrap();

    let all: Vec<&str> = slices
        .iter()
        .flat_map(|s| s.iter().flatten().copied())
        .collect();
    let t = got.exp;
    let want_s: i128 = all.iter().map(|v| at(v, t)).sum();
    let want_q: i128 = all.iter().map(|v| at(v, t) * at(v, t)).sum();
    assert_eq!(
        got.n,
        all.len() as u64,
        "established values, blanks excluded"
    );
    assert_eq!(got.n, 13);
    assert_eq!(got.s, want_s);
    assert_eq!(got.q, want_q as u128);
    let n = all.len() as i128;
    assert_eq!(
        got.variance_parts(),
        Some((n * want_q - want_s * want_s, n * n))
    );
    let differing = stats.iter().filter(|s| s.exp != t).count() as u32;
    assert_eq!(got.alignments, differing);
    assert!(got.alignments < got.n as u32);
}

/// Aligning aggregates that leave i128 is refused, not wrapped.
#[test]
fn an_alignment_past_i128_is_refused() {
    use lance_graph_mask_risc::PowerSums;
    let wide = SliceStats {
        exp: Exp(40),
        sums: PowerSums {
            n: 1,
            sum: 1,
            sum_sq: 1,
        },
    };
    let fine = SliceStats {
        exp: Exp(-40),
        sums: PowerSums {
            n: 1,
            sum: 1,
            sum_sq: 1,
        },
    };
    assert_eq!(
        combine(&[wide, fine]),
        Err(DecimalRefusal::AlignmentOverflow)
    );
}

const CLASS: u32 = 0x2B08_0001;

struct Declared(BTreeMap<String, Exp>);

impl FieldExps for Declared {
    fn exp_of(&self, predicate_iri: &str) -> Option<Exp> {
        self.0.get(predicate_iri).copied()
    }
}

/// An intake table with an i32 coefficient column `amount` at scale 2: the
/// cells are the coefficients the exact parser gives.
fn table() -> (Table, String) {
    let s2 = Exp::scale(2);
    let amounts = [
        Some("123.45"),
        None,
        Some("0.00"),
        Some("-0.50"),
        Some("1.20"),
    ];
    let coef: Vec<Option<String>> = amounts
        .iter()
        .map(|a| a.map(|t| parse_exact(t, s2).unwrap().to_string()))
        .collect();
    let ids: Vec<String> = (1..=amounts.len()).map(|i| i.to_string()).collect();
    let rows: Vec<Vec<Option<&str>>> = ids
        .iter()
        .zip(&coef)
        .map(|(id, c)| vec![Some(id.as_str()), c.as_deref()])
        .collect();
    let mut config = ShapeConfig {
        key: Some("id".into()),
        ..ShapeConfig::default()
    };
    config.columns.insert(
        "amount".into(),
        ColumnConfig {
            ty: Some(PgType::Int4),
            encoding: Some(LaneEncoding::Signed),
            absent: Some(AbsentAs::Null),
        },
    );
    let t = Table::from_rows("amounts", CLASS, &["id", "amount"], &rows, &config).unwrap();
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    let iri = t
        .witness()
        .view
        .fields(class)
        .iter()
        .find(|f| f.label == "amount")
        .unwrap()
        .predicate_iri
        .clone();
    (t, iri)
}

fn witness(t: &Table) -> Witness<'_, lance_graph_intake::IntakeView, lance_graph_intake::Owner> {
    let w = t.witness();
    Witness {
        binding: w.binding,
        store: w.store,
        view: w.view,
        frames: w.frames,
        source: w.source,
    }
}

fn decimal_recipe(t: &Table, iri: &str, filters: &[(Op, &str)], at: Exp) -> DecimalRecipe {
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    let exps = Declared(BTreeMap::from([(iri.to_string(), at)]));
    let filters: Vec<DecimalFilter<'_>> = filters
        .iter()
        .map(|&(op, literal)| DecimalFilter {
            path: ".a",
            predicate_iri: iri,
            op,
            literal,
        })
        .collect();
    DecimalRecipe::compile(
        ".a = .properties.amount.value",
        &filters,
        &exps,
        t.witness().view,
        class,
    )
    .unwrap()
}

/// COUNT(rows) counts every row; COUNT(established values) counts the rows
/// that hold a value, the explicit 0.00 included and the blank excluded.
#[test]
fn rows_and_established_values_are_counted_apart() {
    let (t, iri) = table();
    let w = witness(&t);
    let class = lance_graph_dt64::view_class(t.witness().binding.class());
    let rows = Recipe::compile(".a = .properties.amount.value", t.witness().view, class).unwrap();
    assert_eq!(rows.count(&w).unwrap(), 5, "COUNT(rows)");
    let s2 = Exp::scale(2);
    let exps = Declared(BTreeMap::from([(iri.clone(), s2)]));
    // A filter that never holds decides every row with a value and leaves
    // the blank undecided: COUNT(values).
    let values = decimal_recipe(&t, &iri, &[(Op::Lt, "-99999999999")], s2);
    assert_eq!(values.count(&w, &exps).unwrap(), 4, "COUNT(values)");
    assert_eq!(values.count_unknown(&w, &exps).unwrap(), 1, "the blank");
}

/// A decimal literal is resolved at the field's exponent and the filter
/// runs on coefficients: dropping `amount >= 1.205` keeps 1.20, 0.00 and
/// -0.50, drops 123.45 and leaves the blank undecided.
#[test]
fn a_decimal_filter_runs_on_coefficients() {
    let (t, iri) = table();
    let w = witness(&t);
    let s2 = Exp::scale(2);
    let exps = Declared(BTreeMap::from([(iri.clone(), s2)]));
    let r = decimal_recipe(&t, &iri, &[(Op::Ge, "1.205")], s2);
    assert_eq!(r.count(&w, &exps).unwrap(), 3);
    assert_eq!(r.count_unknown(&w, &exps).unwrap(), 1);
    let eq = decimal_recipe(&t, &iri, &[(Op::Eq, "1.2")], s2);
    assert_eq!(eq.count(&w, &exps).unwrap(), 3, "1.2 == 1.20 drops one row");
}

/// A prepared program resolved at one exponent is refused under a fresh
/// declaration with another, although the class, the field identities and
/// the row count are unchanged; under its own exponent it runs.
#[test]
fn a_prepared_filter_refuses_a_changed_exponent_on_the_same_witness() {
    let (t, iri) = table();
    let w = witness(&t);
    let s2 = Exp::scale(2);
    let r = decimal_recipe(&t, &iri, &[(Op::Ge, "1.205")], s2);
    let p = r.prepare_count(5).unwrap();
    let mut scratch = p.scratch().unwrap();
    let same = Declared(BTreeMap::from([(iri.clone(), s2)]));
    assert!(r.run_into(&p, &w, &same, &mut scratch, Out::None).is_ok());
    let other = Declared(BTreeMap::from([(iri.clone(), Exp::scale(3))]));
    assert_eq!(
        r.run_into(&p, &w, &other, &mut scratch, Out::None),
        Err(RecipeRefusal::ExpChanged {
            predicate_iri: iri.clone(),
            resolved: s2,
            now: Exp::scale(3),
        })
    );
    assert!(matches!(
        r.count(&w, &other),
        Err(RecipeRefusal::ExpChanged { .. })
    ));
    let none = Declared(BTreeMap::new());
    assert!(matches!(
        r.count(&w, &none),
        Err(RecipeRefusal::Undeclared { .. })
    ));
    // The VRL recipe alone would have run: the exponent is what refuses.
    assert!(r.recipe.count(&w).is_ok());
}

/// Rows with values, established zeros and absent cells, cut into slices at
/// unaligned boundaries, one slice empty: the bounds-based passes against
/// the slice-id-lane pass of #2182.
fn sliced_rows() -> (Vec<i32>, Vec<u64>, Vec<usize>) {
    let n = 1000;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut coef = Vec::with_capacity(n);
    let mut plane = vec![0u64; words_for(n)];
    for i in 0..n {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = (x >> 33) as i64;
        let (c, present) = match r % 5 {
            0 => (0, false),
            1 => (0, true),
            _ => ((r % 200_001 - 100_000) as i32, true),
        };
        coef.push(c);
        if present {
            plane[i / 64] |= 1 << (i % 64);
        }
    }
    (coef, plane, vec![0, 3, 130, 130, 517, 999, 1000])
}

/// FAILS IF: N, S or Q of any slice taken by its boundaries differs from
/// the grouped pass keyed by a slice-id lane, an empty slice is not empty,
/// or a boundary inside a word leaks rows across slices.
#[test]
fn slice_boundaries_equal_the_slice_id_lane() {
    let (coef, plane, bounds) = sliced_rows();
    let mut slice_of = Vec::with_capacity(coef.len());
    for (k, w) in bounds.windows(2).enumerate() {
        slice_of.extend(std::iter::repeat_n(k as u32, w[1] - w[0]));
    }
    let slices = bounds.len() - 1;
    let want = slice_sums(&coef, &plane, &slice_of, slices).unwrap();
    assert!(
        want.iter().filter(|s| s.n > 0).count() >= 4,
        "non-trivial slices"
    );
    assert_eq!(want[2].n, 0, "the empty slice");

    let zeros = vec![0u32; coef.len()];
    let got = slice_sums_by_bounds(&coef, &plane, &bounds, &zeros).unwrap();
    assert_eq!(got, want);
    let ns = slice_counts_sums(&coef, &plane, &bounds).unwrap();
    let want_ns: Vec<(u64, i64)> = want.iter().map(|s| (s.n, s.sum)).collect();
    assert_eq!(ns, want_ns);
}

/// FAILS IF: boundaries that do not start at 0, end at the row count, or
/// that decrease are accepted.
#[test]
fn slice_boundaries_out_of_order_are_refused() {
    let (coef, plane, _) = sliced_rows();
    let zeros = vec![0u32; coef.len()];
    for bad in [vec![1, 1000], vec![0, 999], vec![0, 500, 400, 1000], vec![]] {
        assert!(slice_counts_sums(&coef, &plane, &bad).is_err(), "{bad:?}");
        assert!(
            slice_sums_by_bounds(&coef, &plane, &bad, &zeros).is_err(),
            "{bad:?}"
        );
    }
}

/// Slices that seal at PM9, PM17 and past it, each with an established
/// zero, a negative value and an absent cell, across a word boundary.
fn width_slices() -> Vec<SealedSlice> {
    let s2 = Exp::scale(2);
    let long = |big: &'static str| -> Vec<Option<&'static str>> {
        (0..70)
            .map(|i| match i % 7 {
                0 => None,
                1 => Some("0"),
                2 => Some("-3"),
                3 => Some(big),
                _ => Some("12"),
            })
            .collect()
    };
    [long("255"), long("-65535"), long("70000.25")]
        .iter()
        .map(|cells| seal(cells, s2).unwrap())
        .collect()
}

/// FAILS IF: a slice is not stored at the narrowest width that holds it,
/// or reads back other values, another presence or another order than it
/// was sealed with.
#[test]
fn a_sealed_slice_round_trips_at_its_physical_width() {
    let sealed = width_slices();
    let widths: Vec<&str> = sealed
        .iter()
        .map(|s| PhysicalSlice::store(s).width())
        .collect();
    assert_eq!(widths, ["PM9", "PM17", "i32"]);
    for s in &sealed {
        let p = PhysicalSlice::store(s);
        assert_eq!(p.load(), *s, "{}", p.width());
        for i in 0..s.coefficients.len() {
            let want = s.present[i].then_some(s.coefficients[i]);
            assert_eq!(p.get(i), want, "{} value {i}", p.width());
        }
        let bits: Vec<bool> = (0..s.coefficients.len())
            .map(|i| p.presence()[i / 64] >> (i % 64) & 1 == 1)
            .collect();
        assert_eq!(bits, s.present, "{}: presence is P ∨ M ≠ 0", p.width());
    }
}

/// FAILS IF: an established zero is stored as untouched, an absent cell as
/// `+0`, a negative value with the positive polarity, or a `-0` is ever
/// written.
#[test]
fn p_times_m_keeps_absent_and_established_zero_apart() {
    let s = seal(&[None, Some("0"), Some("-7"), Some("7")], Exp::scale(2)).unwrap();
    let p = PhysicalSlice::store(&s);
    let Lanes::Pm9 { mag, pos } = &p.lanes else {
        panic!("PM9 expected, got {}", p.width())
    };
    let pos = pos[0];
    let lane = |i: usize| (pos >> i & 1 == 1, mag[i]);
    assert_eq!(lane(0), (false, 0), "absent: untouched");
    assert_eq!(lane(1), (true, 0), "established +0");
    assert_eq!(lane(2), (false, 7), "-7");
    assert_eq!(lane(3), (true, 7), "+7");
    assert_eq!(p.presence(), vec![0b1110]);
}

/// FAILS IF: a stored PM9 block is not Abakus's representation: adding
/// the untouched block must give it back, and subtracting it from itself
/// must leave `+0` exactly where a value was established.
#[test]
fn a_pm9_block_is_abakus_arithmetic_input() {
    use ndarray::hpc::abakus::pm::{add_pm, sub_pm, Pm};
    let s = &width_slices()[0];
    let p = PhysicalSlice::store(s);
    assert_eq!(p.width(), "PM9");
    let blocks: Vec<Pm> = (0..s.coefficients.len().div_ceil(64))
        .map(|w| p.pm_block(w).unwrap())
        .collect();
    assert_eq!(blocks.len(), 2, "the slice crosses a block");
    for b in &blocks {
        assert_eq!(add_pm(*b, Pm::untouched()).exact(), Some(*b));
        let zero = sub_pm(*b, *b).exact().expect("no carry");
        assert_eq!(zero.mag.to_array(), [0u8; 64]);
        assert_eq!(zero.pos, ndarray::hpc::abakus::pm::touched(*b));
    }
}

/// FAILS IF: PM9 storage costs more than one magnitude byte and one
/// polarity bit per value, or the resident Abakus register is mistaken for
/// the storage layout (it is padded to its SIMD alignment).
#[test]
fn pm9_is_stored_packed_not_as_resident_registers() {
    let s = &width_slices()[0];
    let p = PhysicalSlice::store(s);
    let n = s.coefficients.len();
    assert_eq!(p.bytes(), DESCRIPTOR_BYTES + n + n.div_ceil(64) * 8);
    let register = std::mem::size_of::<ndarray::hpc::abakus::pm::Pm>();
    assert!(register >= 72, "a Pm register holds 64 bytes and a u64");
    println!("resident Pm register: {register} bytes");
}

/// Columns shaped like the three sources the plan names, each with the
/// exponent a hand declaration gives it.
fn decimal_samples() -> Vec<(&'static str, Vec<Option<&'static str>>, Exp)> {
    vec![
        // Log lines: request durations in milliseconds, one absent.
        (
            "log duration_ms",
            vec![Some("12.5"), Some("3.25"), None, Some("100"), Some("0.75")],
            Exp::scale(2),
        ),
        // JSON telemetry: CPU percent with one decimal, negative drift.
        (
            "json cpu_pct",
            vec![Some("37.5"), Some("0.0"), Some("99.9"), Some("-0.1")],
            Exp::scale(1),
        ),
        // Graph-report-shaped: storage in GB with a trailing-zero scale.
        (
            "graph storage_gb",
            vec![Some("250.00"), Some("1.50"), Some("0.25"), None],
            Exp::scale(2),
        ),
    ]
}

/// FAILS IF: a decimal column of a log, JSON or Graph-report shape does not
/// detect as a decimal at the coarsest exponent that reads every value, its
/// range or support is wrong, or the declaration disagrees with detection.
#[test]
fn decimal_text_detects_as_an_exact_decimal_and_agrees_with_its_declaration() {
    for (name, cells, declared) in decimal_samples() {
        let detected = detect_decimal(&cells);
        let DecimalReading::Decimal {
            exp,
            min,
            max,
            provenance,
            support,
            ..
        } = detected.clone()
        else {
            panic!("{name}: {detected:?}")
        };
        assert_eq!(exp, declared, "{name}");
        assert_eq!(provenance, Provenance::Lexical);
        let present: Vec<i32> = cells
            .iter()
            .flatten()
            .map(|t| parse_exact(t, exp).unwrap())
            .collect();
        assert_eq!(
            (min, max),
            (
                *present.iter().min().unwrap(),
                *present.iter().max().unwrap()
            ),
            "{name}"
        );
        assert_eq!(support, present.len(), "{name}");
        let merged =
            merge_decimal(&cells, &detected, Some(DecimalDecl::Decimal(declared))).unwrap();
        assert_eq!(
            merged,
            (DecimalColumn::Decimal(declared), DecimalVerdict::Agrees),
            "{name}"
        );
    }
}

/// FAILS IF: the same column, today, detects as anything but text in
/// intake: the decimal reading is an addition, not a change to intake.
#[test]
fn intake_still_detects_decimal_text_as_text() {
    for (name, cells, _) in decimal_samples() {
        let d = lance_graph_intake::detect_column(&cells);
        assert_eq!(d.col.ty, PgType::Text, "{name}: {:?}", d.col);
    }
}

/// FAILS IF: a column with a binary-float artefact is read as a decimal,
/// a short decimal is mistaken for one, or the artefact is rounded instead
/// of refused. An explicit `f64` declaration is what such a column gets.
#[test]
fn a_float_artefact_column_is_refused_as_decimal() {
    let cells = [Some("0.1"), Some("0.30000000000000004"), Some("0.2")];
    let detected = detect_decimal(&cells);
    assert_eq!(
        detected,
        DecimalReading::Refused {
            row: 1,
            why: DecimalDetectRefusal::FloatArtefact
        }
    );
    let merged = merge_decimal(&cells, &detected, None).unwrap();
    assert_eq!(merged.0, DecimalColumn::Text, "never silently a decimal");
    let f64_decl = merge_decimal(&cells, &detected, Some(DecimalDecl::F64)).unwrap();
    assert_eq!(f64_decl.0, DecimalColumn::F64);
    assert!(matches!(f64_decl.1, DecimalVerdict::Differs { .. }));
    // Fifteen significant digits round-trip through f64, so they are not
    // an artefact; this one is refused only because it leaves `i32`.
    assert_eq!(
        detect_decimal(&[Some("0.123456789012345")]),
        DecimalReading::Refused {
            row: 0,
            why: DecimalDetectRefusal::Overflow
        }
    );
}

/// FAILS IF: a declaration coarser than the data rounds instead of
/// refusing, exponent notation or overflow detects as a decimal, or an
/// integer column claims the decimal reading.
#[test]
fn decimal_detection_never_rounds() {
    let cells = [Some("1.25"), Some("2.5")];
    let detected = detect_decimal(&cells);
    assert!(matches!(
        merge_decimal(&cells, &detected, Some(DecimalDecl::Decimal(Exp::scale(1)))),
        Err(DecimalRefusal::NotExact { .. })
    ));
    assert_eq!(
        detect_decimal(&[Some("1.5"), Some("1e3")]),
        DecimalReading::Refused {
            row: 1,
            why: DecimalDetectRefusal::Malformed
        }
    );
    assert_eq!(
        detect_decimal(&[Some("0.5"), Some("3000000000")]),
        DecimalReading::Refused {
            row: 1,
            why: DecimalDetectRefusal::Overflow
        }
    );
    assert_eq!(
        detect_decimal(&[Some("12"), None, Some("-4")]),
        DecimalReading::NotDecimal
    );
}
