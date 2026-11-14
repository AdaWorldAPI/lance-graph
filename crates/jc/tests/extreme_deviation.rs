//! Range and largest standardized deviation from folded summaries.
//!
//! The inputs come from the two substrate folds (`masked_group_power_sums_i32`
//! and `masked_argminmax_i32`); the expected values come from an independent
//! scalar oracle written in this file that loops over the selected rows in
//! `f64` and never calls the functions under test.

use jc::stats::{extreme_deviation_from_power_sums, range_from_extrema, ExtremeDeviation};
use ndarray::simd::{masked_argminmax_i32, masked_group_power_sums_i32, PowerSums};

type Arg = (usize, i32);

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn mask_words_for(len: usize) -> usize {
    len.div_ceil(64)
}

fn mask_all(len: usize) -> Vec<u64> {
    let mut m = vec![0u64; mask_words_for(len)];
    for i in 0..len {
        m[i / 64] |= 1u64 << (i % 64);
    }
    m
}

fn selected(mask: &[u64], i: usize) -> bool {
    (mask[i / 64] >> (i % 64)) & 1 == 1
}

fn fold(values: &[i32], mask: &[u64]) -> (PowerSums, Option<(Arg, Arg)>) {
    let keys = vec![0u32; values.len()];
    let mut out = [PowerSums::default()];
    masked_group_power_sums_i32(mask, &keys, values, &mut out);
    let extrema = masked_argminmax_i32(values, mask);
    (out[0], extrema)
}

#[derive(Debug)]
struct Oracle {
    range: i64,
    deviation: f64,
    at: usize,
    value: i32,
    mean: f64,
    std_dev: f64,
}

/// Plain `f64` loops over the selected rows. `None` when fewer than two rows
/// are selected or all selected values are equal.
fn oracle(values: &[i32], mask: &[u64]) -> Option<Oracle> {
    let rows: Vec<(usize, i32)> = (0..values.len())
        .filter(|&i| selected(mask, i))
        .map(|i| (i, values[i]))
        .collect();
    let n = rows.len();
    if n < 2 {
        return None;
    }
    let mut min = rows[0];
    let mut max = rows[0];
    let mut total_f = 0.0f64;
    let mut total_i = 0i128;
    for &(i, v) in &rows {
        if v < min.1 {
            min = (i, v);
        }
        if v > max.1 {
            max = (i, v);
        }
        total_f += f64::from(v);
        total_i += i128::from(v);
    }
    if min.1 == max.1 {
        return None;
    }
    let nf = n as f64;
    let mean = total_f / nf;
    let mut ss = 0.0f64;
    for &(_, v) in &rows {
        let d = f64::from(v) - mean;
        ss += d * d;
    }
    let std_dev = (ss / (nf - 1.0)).sqrt();
    let below = mean - f64::from(min.1);
    let above = f64::from(max.1) - mean;
    // Exact tie test in integers: x̄ − min == max − x̄  <=>  (min + max)·n == 2·Σx.
    let exact_tie = (i128::from(min.1) + i128::from(max.1)) * (n as i128) == 2 * total_i;
    let take_min = if exact_tie {
        min.0 < max.0
    } else {
        below > above
    };
    let (at, value, far) = if take_min {
        (min.0, min.1, below)
    } else {
        (max.0, max.1, above)
    };
    Some(Oracle {
        range: i64::from(max.1) - i64::from(min.1),
        deviation: far / std_dev,
        at,
        value,
        mean,
        std_dev,
    })
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

fn random_values(rng: &mut SplitMix64, n: usize, full_range: bool) -> Vec<i32> {
    (0..n)
        .map(|_| {
            if full_range {
                rng.next_u64() as u32 as i32
            } else {
                rng.below(2001) as i32 - 1000
            }
        })
        .collect()
}

fn random_mask(rng: &mut SplitMix64, len: usize) -> Vec<u64> {
    let mut m = vec![0u64; mask_words_for(len)];
    let density = 30 + rng.below(70); // percent
    for i in 0..len {
        if rng.below(100) < density {
            m[i / 64] |= 1u64 << (i % 64);
        }
    }
    let count: u32 = m.iter().map(|w| w.count_ones()).sum();
    if count < 2 {
        m[0] |= 0b11;
    }
    m
}

fn from_all_rows(values: &[i32]) -> Option<ExtremeDeviation> {
    let mask = mask_all(values.len());
    let (sums, mm) = fold(values, &mask);
    let (min, max) = mm?;
    extreme_deviation_from_power_sums(&sums, min, max)
}

#[test]
fn range_is_exact_at_the_i32_extremes() {
    assert_eq!(
        range_from_extrema((0, i32::MIN), (1, i32::MAX)),
        Some(4_294_967_295),
        "full i32 span must be exact, not wrapped"
    );
    assert_eq!(
        range_from_extrema((3, -7), (9, -7)),
        Some(0),
        "equal extrema give a zero range"
    );
    assert_eq!(
        range_from_extrema((0, 5), (1, 4)),
        None,
        "min above max is not the extrema of any population"
    );
}

#[test]
fn matches_the_scalar_oracle_on_random_populations() {
    let mut rng = SplitMix64::new(SEED);
    let mut compared = 0usize;
    let mut skipped = 0usize;
    for case in 0..300 {
        let n = 2 + rng.below(499) as usize;
        let full = case % 2 == 1;
        let values = random_values(&mut rng, n, full);
        let mask = random_mask(&mut rng, n);
        let (sums, mm) = fold(&values, &mask);
        let expected = oracle(&values, &mask);
        let Some(exp) = expected else {
            skipped += 1;
            if let Some((min, max)) = mm {
                assert_eq!(
                    extreme_deviation_from_power_sums(&sums, min, max),
                    None,
                    "case {case}: zero-variance population must give None"
                );
            }
            continue;
        };
        let (min, max) = mm.expect("selected rows exist, so extrema exist");
        let got = extreme_deviation_from_power_sums(&sums, min, max)
            .unwrap_or_else(|| panic!("case {case}: expected Some for n={n}, full={full}"));
        assert_eq!(got.range, exp.range, "case {case}: range");
        assert_eq!(got.at, exp.at, "case {case}: row of the deviation");
        assert_eq!(got.value, exp.value, "case {case}: value at that row");
        assert!(
            close(got.deviation, exp.deviation, 1e-9),
            "case {case}: deviation {} vs oracle {}",
            got.deviation,
            exp.deviation
        );
        assert!(
            close(got.mean, exp.mean, 1e-9),
            "case {case}: mean {} vs oracle {}",
            got.mean,
            exp.mean
        );
        assert!(
            close(got.std_dev, exp.std_dev, 1e-9),
            "case {case}: std_dev {} vs oracle {}",
            got.std_dev,
            exp.std_dev
        );
        compared += 1;
    }
    assert!(
        compared >= 250,
        "only {compared} cases compared ({skipped} skipped); the check would be vacuous"
    );
}

#[test]
fn masked_out_rows_do_not_count() {
    let mut rng = SplitMix64::new(SEED ^ 3);
    for excluded_value in [i32::MAX, i32::MIN] {
        let n = 200usize;
        let mut values = random_values(&mut rng, n, false);
        let excluded_row = 77usize;
        values[excluded_row] = excluded_value;
        let mut mask = mask_all(n);
        mask[excluded_row / 64] &= !(1u64 << (excluded_row % 64));

        let (sums, mm) = fold(&values, &mask);
        let (min, max) = mm.expect("selected rows exist");
        let got = extreme_deviation_from_power_sums(&sums, min, max).expect("Some");
        let exp = oracle(&values, &mask).expect("oracle Some");
        assert_ne!(
            got.at, excluded_row,
            "must not name the excluded row (value {excluded_value})"
        );
        assert_eq!(got.at, exp.at, "row vs oracle over selected rows");
        assert_eq!(got.value, exp.value, "value vs oracle over selected rows");
        assert_eq!(got.range, exp.range, "range excludes the masked-out value");
        assert!(
            close(got.deviation, exp.deviation, 1e-9),
            "deviation vs oracle over selected rows"
        );
    }
}

#[test]
fn degenerate_inputs_are_none() {
    assert_eq!(
        extreme_deviation_from_power_sums(&PowerSums::default(), (0, 0), (1, 5)),
        None,
        "n == 0 must give None"
    );

    let values = [42i32, 1, 2, 3];
    let mask = [0b0001u64];
    let (sums, mm) = fold(&values, &mask);
    assert_eq!(sums.n, 1, "fixture selects exactly one row");
    let (min, max) = mm.expect("one selected row has extrema");
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, min, max),
        None,
        "n == 1 has no sample standard deviation"
    );

    let equal = [7i32, 7, 7, 7];
    let (sums, mm) = fold(&equal, &mask_all(4));
    let (min, max) = mm.expect("extrema exist");
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, min, max),
        None,
        "all values equal (extrema at the same row) must give None"
    );
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, (0, 7), (3, 7)),
        None,
        "all values equal (extrema at different rows) must give None"
    );
}

#[test]
fn inconsistent_inputs_are_none() {
    // Population [0, 10, 5, 5]: mean 5.
    let values = [0i32, 10, 5, 5];
    let (sums, mm) = fold(&values, &mask_all(4));
    let (min, max) = mm.expect("extrema exist");
    assert!(
        extreme_deviation_from_power_sums(&sums, min, max).is_some(),
        "the consistent inputs must be accepted"
    );

    assert_eq!(
        extreme_deviation_from_power_sums(&sums, (0, 10), (1, 0)),
        None,
        "case (a): min above max"
    );
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, (0, 0), (1, 2)),
        None,
        "case (b): max below the mean"
    );
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, (0, 7), (1, 10)),
        None,
        "case (c): min above the mean"
    );
    assert_eq!(
        extreme_deviation_from_power_sums(&sums, (0, 0), (0, 10)),
        None,
        "case (d): both extrema at one row with different values"
    );

    // Case (e): sums of [0, 0, 0, 1] with a max from a much wider population.
    let narrow = PowerSums {
        n: 4,
        sum: 1,
        sum_sq: 1,
    };
    assert_eq!(
        extreme_deviation_from_power_sums(&narrow, (0, 0), (3, 1000)),
        None,
        "case (e): the mean lies inside [min, max], so the (n-1)/sqrt(n) bound is what must reject this"
    );
}

#[test]
fn equal_tails_pick_the_lower_row() {
    // The two cases together distinguish "lower row" from "always min" and
    // from "always max": the first must return the max, the second the min.
    let a = from_all_rows(&[10, 0, 5, 5]).expect("Some");
    assert_eq!(a.at, 0, "[10,0,5,5]: tails equal, lower row is the max");
    assert_eq!(a.value, 10, "[10,0,5,5]: value at the lower row");

    let b = from_all_rows(&[0, 10, 5, 5]).expect("Some");
    assert_eq!(b.at, 0, "[0,10,5,5]: tails equal, lower row is the min");
    assert_eq!(b.value, 0, "[0,10,5,5]: value at the lower row");
}

#[test]
fn unequal_tails_pick_the_further_one_regardless_of_row() {
    let cases: [([i32; 5], usize, i32, &str); 4] = [
        ([0, 0, 0, 0, 100], 4, 100, "max further, at the higher row"),
        (
            [100, 100, 100, 100, 0],
            4,
            0,
            "min further, at the higher row",
        ),
        ([100, 0, 0, 0, 0], 0, 100, "max further, at the lower row"),
        (
            [0, 100, 100, 100, 100],
            0,
            0,
            "min further, at the lower row",
        ),
    ];
    for (values, at, value, what) in cases {
        let d = from_all_rows(&values).unwrap_or_else(|| panic!("{what}: expected Some"));
        assert_eq!(d.at, at, "{what}: row");
        assert_eq!(d.value, value, "{what}: value");
    }
}

#[test]
fn deviation_never_exceeds_its_bound() {
    let mut rng = SplitMix64::new(SEED ^ 8);
    for case in 0..300 {
        let n = 2 + rng.below(499) as usize;
        let values = random_values(&mut rng, n, case % 2 == 1);
        let mask = random_mask(&mut rng, n);
        let (sums, mm) = fold(&values, &mask);
        let Some((min, max)) = mm else { continue };
        let Some(d) = extreme_deviation_from_power_sums(&sums, min, max) else {
            continue;
        };
        let m = sums.n as f64;
        let bound = (m - 1.0) / m.sqrt();
        assert!(d.deviation > 0.0, "case {case}: deviation must be positive");
        assert!(
            d.deviation <= bound * (1.0 + 1e-9),
            "case {case}: deviation {} above bound {bound}",
            d.deviation
        );
    }

    for n in [2usize, 3, 10, 100] {
        let mut values = vec![3i32; n];
        values[n / 2] = 10;
        let d = from_all_rows(&values).expect("Some");
        let bound = (n as f64 - 1.0) / (n as f64).sqrt();
        assert!(
            (d.deviation - bound).abs() <= 1e-12 * bound,
            "n={n}: deviation {} must attain the bound {bound}",
            d.deviation
        );
    }
}

#[test]
fn full_i32_span_does_not_overflow() {
    let values = [i32::MIN, i32::MAX, 0, 0];
    let d = from_all_rows(&values).expect("Some at the full i32 span");
    assert_eq!(d.range, 4_294_967_295, "range at the full span");
    assert!(d.deviation.is_finite(), "deviation must be finite");
    let exp = oracle(&values, &mask_all(4)).expect("oracle Some");
    assert_eq!(d.at, exp.at, "row vs oracle");
    assert_eq!(d.value, exp.value, "value vs oracle");
    assert!(
        close(d.deviation, exp.deviation, 1e-9),
        "deviation {} vs oracle {}",
        d.deviation,
        exp.deviation
    );
    assert!(close(d.mean, exp.mean, 1e-9), "mean vs oracle");
    assert!(close(d.std_dev, exp.std_dev, 1e-9), "std_dev vs oracle");
}

fn merge_min(a: Option<Arg>, b: Option<Arg>) -> Option<Arg> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if y.1 < x.1 || (y.1 == x.1 && y.0 < x.0) {
            y
        } else {
            x
        }),
    }
}

fn merge_max(a: Option<Arg>, b: Option<Arg>) -> Option<Arg> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if y.1 > x.1 || (y.1 == x.1 && y.0 < x.0) {
            y
        } else {
            x
        }),
    }
}

#[test]
fn tiles_merge_to_the_same_answer() {
    let mut rng = SplitMix64::new(SEED ^ 10);
    let len = 300usize;
    // Few distinct values so the same extreme value recurs across pieces.
    let values: Vec<i32> = (0..len).map(|_| rng.below(5) as i32).collect();
    let mut mask = vec![0u64; mask_words_for(len)];
    for i in 0..len {
        if rng.below(100) < 90 {
            mask[i / 64] |= 1u64 << (i % 64);
        }
    }

    let (whole_sums, whole_mm) = fold(&values, &mask);
    let (wmin, wmax) = whole_mm.expect("extrema exist");
    let whole = extreme_deviation_from_power_sums(&whole_sums, wmin, wmax).expect("Some");

    let bounds = [(0usize, 128usize), (128, 256), (256, 300)];
    let mut total = PowerSums::default();
    let mut min: Option<Arg> = None;
    let mut max: Option<Arg> = None;
    let mut piece_mins = Vec::new();
    let mut piece_maxs = Vec::new();
    for &(from, to) in &bounds {
        let words = &mask[from / 64..mask_words_for(to)];
        let (sums, mm) = fold(&values[from..to], words);
        total.n += sums.n;
        total.sum += sums.sum;
        total.sum_sq += sums.sum_sq;
        if let Some((pmin, pmax)) = mm {
            let pmin = (pmin.0 + from, pmin.1);
            let pmax = (pmax.0 + from, pmax.1);
            piece_mins.push(pmin);
            piece_maxs.push(pmax);
            min = merge_min(min, Some(pmin));
            max = merge_max(max, Some(pmax));
        }
    }

    let tie_min = piece_mins
        .iter()
        .filter(|p| Some(p.1) == min.map(|m| m.1))
        .count();
    let tie_max = piece_maxs
        .iter()
        .filter(|p| Some(p.1) == max.map(|m| m.1))
        .count();
    assert!(
        tie_min >= 2 || tie_max >= 2,
        "fixture must contain an extreme value tied across pieces (min ties {tie_min}, max ties {tie_max})"
    );

    assert_eq!(total, whole_sums, "power sums add field by field");
    assert_eq!(min, Some(wmin), "merged minimum equals the single fold's");
    assert_eq!(max, Some(wmax), "merged maximum equals the single fold's");
    let merged =
        extreme_deviation_from_power_sums(&total, min.unwrap(), max.unwrap()).expect("Some");
    assert_eq!(merged, whole, "merged summaries give the same result");
}

#[test]
fn the_row_is_returned_unchanged() {
    // The function does not interpret the row: it only returns the one it was given.
    let base = 1_000_000usize;
    let mut rng = SplitMix64::new(SEED ^ 11);
    let random = random_values(&mut rng, 120, false);
    let populations: [Vec<i32>; 2] = [random, vec![10, 0, 5, 5]];
    for (k, values) in populations.iter().enumerate() {
        let mask = mask_all(values.len());
        let (sums, mm) = fold(values, &mask);
        let (min, max) = mm.expect("extrema exist");
        let plain = extreme_deviation_from_power_sums(&sums, min, max).expect("Some");
        let shifted =
            extreme_deviation_from_power_sums(&sums, (min.0 + base, min.1), (max.0 + base, max.1))
                .expect("Some");
        assert_eq!(
            shifted.at,
            plain.at + base,
            "population {k}: row moves by the base"
        );
        let mut back = shifted;
        back.at -= base;
        assert_eq!(
            back, plain,
            "population {k}: every other field is identical"
        );
    }
}
