//! D-RU-1 (U1): a batch answered from shared power-sum passes equals every
//! request answered on its own, costs fewer traversals, values read and mask
//! bytes, and keeps the integer sums exact.

use jc::reliability::pearson;
use jc::stats::anova_one_way;
use jc_batch_probe::{answer_batched, answer_separately, Answer, BatchError, Request, Stat, Table};
use lance_graph_mask_risc::LaneRef;
use lance_graph_relalg::{Cmp, Rel};

/// Statistic lanes 0..7 (`I32`), a selection lane 7 (`I32`), a category lane
/// 8 (`U32`), the group lane 9 (`U32`, `GROUPS` values) and a zero lane 10.
const STAT_LANES: u16 = 7;
const SEL: u16 = 7;
const CAT: u16 = 8;
const GROUP: u16 = 9;
const ZERO: u16 = 10;
const GROUPS: usize = 8;
const VERSION: u64 = 42;

struct Fixture {
    n: usize,
    stats: Vec<Vec<i32>>,
    sel: Vec<i32>,
    cat: Vec<u32>,
    group: Vec<u32>,
    zero: Vec<u32>,
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

impl Fixture {
    /// `n` rows; lane `j` is correlated with lane 0 by `j`, plus noise, so
    /// the correlations differ from lane to lane.
    fn new(n: usize, seed: u64) -> Self {
        let mut s = seed;
        let base: Vec<i32> = (0..n)
            .map(|_| (lcg(&mut s) % 2_001) as i32 - 1_000)
            .collect();
        let stats = (0..STAT_LANES as usize)
            .map(|j| {
                base.iter()
                    .map(|&b| b * j as i32 + (lcg(&mut s) % 401) as i32 - 200)
                    .collect()
            })
            .collect();
        Fixture {
            n,
            stats,
            sel: (0..n).map(|_| (lcg(&mut s) % 100) as i32).collect(),
            cat: (0..n).map(|_| (lcg(&mut s) % 4) as u32).collect(),
            group: (0..n)
                .map(|_| (lcg(&mut s) % GROUPS as u64) as u32)
                .collect(),
            zero: vec![0; n],
        }
    }

    fn lanes(&self) -> Vec<LaneRef<'_>> {
        let mut l: Vec<LaneRef<'_>> = self.stats.iter().map(|v| LaneRef::I32(v)).collect();
        l.push(LaneRef::I32(&self.sel));
        l.push(LaneRef::U32(&self.cat));
        l.push(LaneRef::U32(&self.group));
        l.push(LaneRef::U32(&self.zero));
        l
    }

    fn table<'a>(&self, lanes: &'a [LaneRef<'a>]) -> Table<'a> {
        Table {
            version: VERSION,
            n_rows: self.n,
            lanes,
            group: GROUP,
            groups: GROUPS,
            zero: ZERO,
        }
    }

    /// The rows a selection keeps, computed directly (not through relalg).
    fn rows(&self, which: Which) -> Vec<usize> {
        (0..self.n)
            .filter(|&i| match which {
                Which::Low => self.sel[i] < 60,
                Which::LowCat => self.sel[i] < 60 && self.cat[i] == 2,
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum Which {
    Low,
    LowCat,
}

/// `sel < 60`.
fn low() -> Rel {
    Rel::i32(SEL, Cmp::Lt, 60)
}

/// `sel < 60 AND cat == 2`, written in two orders.
fn low_cat(flipped: bool) -> Rel {
    if flipped {
        Rel::And(vec![Rel::eq_u32(CAT, 2), low()])
    } else {
        Rel::And(vec![low(), Rel::eq_u32(CAT, 2)])
    }
}

/// A full battery over one selection: means, variances and ANOVAs of every
/// lane; covariance, Pearson and R² of every pair; and the regression line
/// of every pair in both directions.
fn battery(mut select: impl FnMut() -> Rel) -> Vec<Request> {
    let mut r = Vec::new();
    let mut push = |stat| {
        r.push(Request {
            version: VERSION,
            select: select(),
            stat,
        })
    };
    for x in 0..STAT_LANES {
        push(Stat::Mean(x));
        push(Stat::Variance(x));
        push(Stat::Anova(x));
    }
    for x in 0..STAT_LANES {
        for y in (x + 1)..STAT_LANES {
            push(Stat::Covariance(x, y));
            push(Stat::Pearson(x, y));
            push(Stat::RSquared(x, y));
            push(Stat::Regression(x, y));
            push(Stat::Regression(y, x));
        }
    }
    r
}

/// Two selections, the second written in two orders across its requests.
fn requests() -> Vec<Request> {
    let mut r = battery(low);
    let mut flip = false;
    r.extend(battery(move || {
        flip = !flip;
        low_cat(flip)
    }));
    r
}

/// FAILS IF: any batched answer differs from the request answered on its own
/// (bit for bit; the integer sums are exact, so equality is the contract).
/// Anti-vacuity: the battery has defined answers of every kind.
#[test]
fn batched_answers_equal_separate_answers_exactly() {
    let fx = Fixture::new(5_000, 0xBA7C);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs = requests();
    let (sep, _) = answer_separately(&t, &reqs).unwrap();
    let (bat, _, _) = answer_batched(&t, &reqs).unwrap();
    assert_eq!(sep.len(), reqs.len());
    assert_eq!(bat, sep);
    let defined = |f: fn(&Answer) -> bool| bat.iter().any(f);
    assert!(defined(|a| matches!(a, Answer::Scalar(Some(_)))));
    assert!(defined(|a| matches!(a, Answer::Line(Some(_)))));
    assert!(defined(|a| matches!(a, Answer::Anova(Some(_)))));
}

/// FAILS IF: the batch does not reduce to 2 selections and 21 pair passes
/// per selection, or does not cost less than the separate path in programs,
/// rows traversed, values read and mask bytes.
#[test]
fn batched_cost_is_lower_in_traversals_values_and_masks() {
    let fx = Fixture::new(5_000, 0xC057);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs = requests();
    let (_, sep) = answer_separately(&t, &reqs).unwrap();
    let (_, bat, red) = answer_batched(&t, &reqs).unwrap();
    let pairs = (STAT_LANES as usize) * (STAT_LANES as usize - 1) / 2;
    assert_eq!(red.selections, 2);
    assert_eq!(red.cross_passes, 2 * pairs);
    assert_eq!(red.power_passes, 0, "every lane is covered by a pair");

    let n = fx.n as u64;
    assert_eq!(sep.programs, 2 * reqs.len());
    assert_eq!(sep.selection_passes, reqs.len());
    assert_eq!(bat.selection_passes, 2);
    assert_eq!(bat.sum_passes, 2 * pairs);
    assert_eq!(bat.rows_traversed, (2 + 2 * pairs) as u64 * n);
    assert_eq!(bat.mask_bytes, 2 * (fx.n.div_ceil(64) * 8) as u64);
    assert!(
        bat.rows_traversed * 5 < sep.rows_traversed,
        "{bat:?} vs {sep:?}"
    );
    assert!(bat.values_read * 5 < sep.values_read, "{bat:?} vs {sep:?}");
    assert!(bat.mask_bytes * 100 < sep.mask_bytes, "{bat:?} vs {sep:?}");
}

fn rel_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

/// FAILS IF: a batched statistic disagrees with an independent computation
/// over the selected rows in `f64` (JC's slice functions and two-pass
/// formulas; rows selected directly, not through relalg).
#[test]
fn batched_answers_agree_with_an_independent_f64_oracle() {
    let fx = Fixture::new(5_000, 0x0C1E);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs = requests();
    let (bat, _, _) = answer_batched(&t, &reqs).unwrap();
    let half = reqs.len() / 2;
    let mut checked = 0;
    for (i, (r, a)) in reqs.iter().zip(&bat).enumerate() {
        let rows = fx.rows(if i < half { Which::Low } else { Which::LowCat });
        let col = |l: u16| -> Vec<f64> {
            rows.iter()
                .map(|&k| f64::from(fx.stats[l as usize][k]))
                .collect()
        };
        match (r.stat, a) {
            (Stat::Mean(x), Answer::Scalar(Some(m))) => {
                let v = col(x);
                assert!(rel_close(*m, v.iter().sum::<f64>() / v.len() as f64));
                checked += 1;
            }
            (Stat::Variance(x), Answer::Scalar(Some(s2))) => {
                let v = col(x);
                let mu = v.iter().sum::<f64>() / v.len() as f64;
                let want =
                    v.iter().map(|z| (z - mu) * (z - mu)).sum::<f64>() / (v.len() - 1) as f64;
                assert!(rel_close(*s2, want), "{s2} vs {want}");
                checked += 1;
            }
            (Stat::Pearson(x, y), Answer::Scalar(Some(rr))) => {
                let want = pearson(&col(x), &col(y)).unwrap();
                assert!(rel_close(*rr, want), "{rr} vs {want}");
                checked += 1;
            }
            (Stat::Anova(x), Answer::Anova(Some(an))) => {
                let mut g = vec![Vec::new(); GROUPS];
                for &k in &rows {
                    g[fx.group[k] as usize].push(f64::from(fx.stats[x as usize][k]));
                }
                let want = anova_one_way(&g).unwrap();
                assert!(rel_close(an.f, want.f), "{} vs {}", an.f, want.f);
                checked += 1;
            }
            _ => {}
        }
    }
    assert!(checked > 50, "checked {checked}");
}

/// FAILS IF: the sums are not exact at the `i32` extremes. Every value is
/// `i32::MIN` or `i32::MAX`, so `Σx²` passes `u64::MAX` and an `i64`
/// accumulator of `Σx·y` would overflow (checked below); the batched mean
/// and variance must equal the exact `i128` values.
#[test]
fn sums_stay_exact_at_the_i32_extremes() {
    let n = 4_096;
    let mut fx = Fixture::new(n, 0xE7E7);
    for (j, lane) in fx.stats.iter_mut().enumerate() {
        for (i, v) in lane.iter_mut().enumerate() {
            *v = if (i + j) % 3 == 0 { i32::MIN } else { i32::MAX };
        }
    }
    fx.sel = vec![0; n]; // every row selected by `low()`
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs: Vec<Request> = (0..STAT_LANES)
        .flat_map(|x| {
            [
                Stat::Mean(x),
                Stat::Variance(x),
                Stat::Covariance(x, (x + 1) % STAT_LANES),
            ]
        })
        .map(|stat| Request {
            version: VERSION,
            select: low(),
            stat,
        })
        .collect();
    let (bat, _, _) = answer_batched(&t, &reqs).unwrap();
    let (sep, _) = answer_separately(&t, &reqs).unwrap();
    assert_eq!(bat, sep);

    // Anti-vacuity: the naive accumulators really overflow on this data.
    let x0 = &fx.stats[0];
    let sq_u64 = x0.iter().try_fold(0u64, |a, &v| {
        a.checked_add((i64::from(v) * i64::from(v)) as u64)
    });
    assert!(
        sq_u64.is_none(),
        "Σx² must exceed u64 for the fixture to bite"
    );
    let xy_i64 = x0.iter().zip(&fx.stats[1]).try_fold(0i64, |a, (&p, &q)| {
        a.checked_add(i64::from(p) * i64::from(q))
    });
    assert!(
        xy_i64.is_none(),
        "Σxy must exceed i64 for the fixture to bite"
    );

    for (r, a) in reqs.iter().zip(&bat) {
        let x = match r.stat {
            Stat::Mean(x) | Stat::Variance(x) => x,
            _ => continue,
        };
        let v = &fx.stats[x as usize];
        let n128 = n as i128;
        let s: i128 = v.iter().map(|&z| i128::from(z)).sum();
        let s2: i128 = v.iter().map(|&z| i128::from(z) * i128::from(z)).sum();
        let want = match r.stat {
            Stat::Mean(_) => s as f64 / n as f64,
            _ => (n128 * s2 - s * s) as f64 / (n128 * (n128 - 1)) as f64,
        };
        assert_eq!(*a, Answer::Scalar(Some(want)), "{:?}", r.stat);
    }
}

/// FAILS IF: the exact path loses the accuracy a naive one-pass `f64`
/// formula loses. Lanes sit at an offset of 10⁹ with a small spread; the
/// naive `Σxy/n − x̄ȳ` correlation is visibly wrong there (anti-vacuity), and
/// the exact one matches the two-pass reference.
#[test]
fn exact_sums_keep_accuracy_where_a_naive_f64_pass_loses_it() {
    let n = 20_000;
    let mut fx = Fixture::new(n, 0xACC0);
    let mut s = 7u64;
    let off = 1_000_000_000;
    fx.stats[0] = (0..n).map(|_| off + (lcg(&mut s) % 1_000) as i32).collect();
    fx.stats[1] = fx.stats[0]
        .iter()
        .map(|&x| x + (lcg(&mut s) % 300) as i32)
        .collect();
    fx.sel = vec![0; n];
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs = [Request {
        version: VERSION,
        select: low(),
        stat: Stat::Pearson(0, 1),
    }];
    let (bat, _, _) = answer_batched(&t, &reqs).unwrap();
    let Answer::Scalar(Some(exact)) = bat[0] else {
        panic!("defined")
    };
    let x: Vec<f64> = fx.stats[0].iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = fx.stats[1].iter().map(|&v| f64::from(v)).collect();
    let two_pass = pearson(&x, &y).unwrap();
    let nf = n as f64;
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let syy: f64 = y.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    let naive = (nf * sxy - sx * sy) / ((nf * sxx - sx * sx).sqrt() * (nf * syy - sy * sy).sqrt());
    assert!(
        (naive - two_pass).abs() > 1e-3,
        "the fixture must defeat the naive formula: {naive} vs {two_pass}"
    );
    assert!((exact - two_pass).abs() < 1e-12, "{exact} vs {two_pass}");
}

/// FAILS IF: a request built for another version is answered, by either path.
#[test]
fn a_request_for_another_version_is_refused() {
    let fx = Fixture::new(500, 0x7E57);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let mut reqs = requests();
    reqs[3].version = VERSION + 1;
    let want = Err(BatchError::VersionMismatch {
        request: 3,
        request_version: VERSION + 1,
        table_version: VERSION,
    });
    assert_eq!(answer_batched(&t, &reqs).map(|_| ()), want);
    assert_eq!(answer_separately(&t, &reqs).map(|_| ()), want);
}

/// FAILS IF: two selections equal after normalization are evaluated twice,
/// or two different selections are merged. The second selection is written
/// in two atom orders across its requests.
#[test]
fn selections_are_shared_after_normalization_and_kept_apart_otherwise() {
    let fx = Fixture::new(1_000, 0x5E15);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs = requests();
    assert!(reqs.iter().any(|r| r.select == low_cat(true)));
    assert!(reqs.iter().any(|r| r.select == low_cat(false)));
    let (_, cost, red) = answer_batched(&t, &reqs).unwrap();
    assert_eq!(red.selections, 2);
    assert_eq!(cost.selection_passes, 2);
}

/// FAILS IF: a lane no pair request covers is not given its own pass, or
/// that pass answers differently from the separate path.
#[test]
fn a_lane_only_univariate_requests_need_gets_one_pass() {
    let fx = Fixture::new(2_000, 0x51A6);
    let lanes = fx.lanes();
    let t = fx.table(&lanes);
    let reqs: Vec<Request> = [
        Stat::Pearson(0, 1),
        Stat::Mean(4),
        Stat::Variance(4),
        Stat::Anova(4),
        Stat::Mean(0),
    ]
    .into_iter()
    .map(|stat| Request {
        version: VERSION,
        select: low(),
        stat,
    })
    .collect();
    let (bat, cost, red) = answer_batched(&t, &reqs).unwrap();
    let (sep, _) = answer_separately(&t, &reqs).unwrap();
    assert_eq!(bat, sep);
    assert_eq!((red.cross_passes, red.power_passes), (1, 1));
    assert_eq!(cost.sum_passes, 2);
}
