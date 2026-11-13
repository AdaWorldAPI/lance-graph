//! Statistics over a slab stack: exact integer folds, the quantized form,
//! admission, and the two shapes the slab fold does not carry (a one-byte
//! lane, a grouped fold), run through mask-RISC directly.
//! Pass conditions: `.claude/board/entries/2026-11-13-weather-dr256-ternlog.md`
//! H3, H4, H6 and the u8 item.
//!
//! The lab has one node class. Its `I32` field (`person::AGE`) carries the
//! measured value here and its `U32` field the row id or the station key; no
//! class is added for the probe.

use dr256_ternlog_lab::*;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::LaneRefusal;
use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, GroupFold, GroupKey, LaneRef, MaskOp, Operand, Out, Planes,
    Pred, Program, Scratch, StridedRef, Terminal, Value,
};
use std::collections::BTreeMap;
use weather_poc::floor::calibrate;

const READER: Scope = Scope { id: 7 };
const GRID: usize = 721 * 1440;
const VALUE: u8 = person::AGE;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn region(i: usize) -> bool {
    !mix(i as u64 ^ 0x11).is_multiple_of(3)
}
fn valid(i: usize) -> bool {
    !mix(i as u64 ^ 0x22).is_multiple_of(50)
}

struct Obs {
    key: Vec<u32>,
    value: Vec<i32>,
}
impl Obs {
    fn table(&self) -> Table<'_> {
        Table::new(
            PERSON,
            1,
            self.value.len(),
            1,
            READER,
            &[
                (person::ID, LaneRef::U32(&self.key)),
                (VALUE, LaneRef::I32(&self.value)),
            ],
        )
    }
}

/// Tenths of a degree in -500..=500: about a thousand distinct values over a
/// million rows, so nearly every value is a duplicate.
fn tenths(n: usize) -> Obs {
    Obs {
        key: (0..n as u64)
            .map(|i| (mix(i ^ 0x33) % 413) as u32)
            .collect(),
        value: (0..n as u64)
            .map(|i| (mix(i) % 1001) as i32 - 500)
            .collect(),
    }
}

#[derive(Debug, PartialEq)]
struct Stats {
    count: u64,
    sum: i64,
    min: Option<i32>,
    max: Option<i32>,
}

fn oracle(obs: &Obs, lo: i32, hi: i32) -> Stats {
    let mut sum = 0i128;
    let (mut count, mut min, mut max) = (0u64, None::<i32>, None::<i32>);
    for (i, &v) in obs.value.iter().enumerate() {
        if region(i) && valid(i) && lo < v && v < hi {
            count += 1;
            sum += i128::from(v);
            min = Some(min.map_or(v, |m| m.min(v)));
            max = Some(max.map_or(v, |m| m.max(v)));
        }
    }
    Stats {
        count,
        sum: i64::try_from(sum).unwrap(),
        min,
        max,
    }
}

fn folded(obs: &Obs, lo: i32, hi: i32) -> Stats {
    let world = World::new();
    let table = obs.table();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let prepared = prepare_nodes(&world, &table, &fields, region, valid).unwrap();
    let run = |agg| {
        Fold::between(&prepared, VALUE, lo, hi, agg)
            .unwrap()
            .answer(&world, &table, &prepared)
            .unwrap()
    };
    let (Answer::Count(count), Answer::Sum(sum), Answer::Min(min), Answer::Max(max)) = (
        run(Agg::Count),
        run(Agg::Sum(VALUE)),
        run(Agg::Min(VALUE)),
        run(Agg::Max(VALUE)),
    ) else {
        panic!("an answer of another kind");
    };
    Stats {
        count,
        sum,
        min,
        max,
    }
}

#[test]
fn exact_statistics_equal_the_scalar_oracle_on_a_ragged_grid() {
    let obs = tenths(GRID);
    for (lo, hi) in [(0, 100), (-501, 501), (-500, -498), (499, 501)] {
        let want = oracle(&obs, lo, hi);
        assert!(want.count > 0, "({lo}, {hi}) selects nothing");
        assert_eq!(folded(&obs, lo, hi), want, "({lo}, {hi})");
    }
    let all = oracle(&obs, -501, 501);
    let some = oracle(&obs, 0, 100);
    assert!(
        some.count * 5 < all.count,
        "the predicate must exclude most rows"
    );
    assert_ne!(some.sum, all.sum);
}

#[test]
fn an_empty_selection_has_no_minimum_and_no_maximum() {
    let obs = tenths(SLAB_ROWS + 77);
    let got = folded(&obs, 500, 501);
    assert_eq!(
        got,
        Stats {
            count: 0,
            sum: 0,
            min: None,
            max: None
        }
    );
}

#[test]
fn duplicate_rows_count_once_each() {
    // The same 100,000 observations three times over: a bag, not a set.
    let once = tenths(100_000);
    let thrice = Obs {
        key: once.key.repeat(3),
        value: once.value.repeat(3),
    };
    assert_eq!(folded(&once, 0, 100), oracle(&once, 0, 100));
    // Row positions differ per copy, so region and valid differ; compare
    // against the oracle over the tripled bag, and against 3x under full
    // planes where position plays no part.
    assert_eq!(folded(&thrice, 0, 100), oracle(&thrice, 0, 100));

    let world = World::new();
    let stats = |obs: &Obs| {
        let table = obs.table();
        let fields = select(&world.view, PERSON, &["age"]).unwrap();
        let prepared = prepare_nodes(&world, &table, &fields, |_| true, |_| true).unwrap();
        let run = |agg| {
            Fold::between(&prepared, VALUE, 0, 100, agg)
                .unwrap()
                .answer(&world, &table, &prepared)
                .unwrap()
        };
        (run(Agg::Count), run(Agg::Sum(VALUE)), run(Agg::Min(VALUE)))
    };
    let (Answer::Count(c1), Answer::Sum(s1), m1) = stats(&once) else {
        panic!()
    };
    let (Answer::Count(c3), Answer::Sum(s3), m3) = stats(&thrice) else {
        panic!()
    };
    assert!(c1 > 0);
    assert_eq!((c3, s3, m3), (3 * c1, 3 * s1, m1));
}

#[test]
fn the_mean_rounds_to_tenths_with_ties_toward_positive_infinity() {
    // Java `Math.round`, which the 1BRC reference uses: floor(x + 0.5).
    let reference = |sum: i64, count: u64| (sum as f64 / count as f64 + 0.5).floor() as i64;
    for (sum, count, want) in [
        (3, 2, 2),   //  1.5 ->  2
        (-3, 2, -1), // -1.5 -> -1
        (-5, 2, -2), // -2.5 -> -2
        (5, 2, 3),   //  2.5 ->  3
        (-1, 3, 0),  // -0.33 -> 0
        (-2, 3, -1), // -0.67 -> -1
        (7, 1, 7),
        (0, 9, 0),
    ] {
        assert_eq!(mean_round_half_up(sum, count), Some(want), "{sum}/{count}");
        assert_eq!(reference(sum, count), want, "the reference, {sum}/{count}");
    }
    assert_eq!(mean_round_half_up(5, 0), None);
}

#[test]
fn folds_over_codes_are_exact_in_code_space_and_decode_through_the_floor() {
    // A synthetic field in kelvin: a meridional gradient plus noise.
    let kelvin: Vec<f64> = (0..GRID)
        .map(|i| {
            let lat = (i / 1440) as f64 / 720.0; // 0 at one pole, 1 at the other
            250.0
                + 50.0 * (std::f64::consts::PI * lat).sin()
                + (mix(i as u64) % 2000) as f64 / 100.0
        })
        .collect();
    let floor = calibrate(&kelvin).unwrap();
    let codes = Obs {
        key: vec![0; GRID],
        value: kelvin
            .iter()
            .map(|&k| i32::from(floor.quantize(k)))
            .collect(),
    };
    let (lo, hi) = (
        i32::from(floor.quantize(273.15)),
        i32::from(floor.quantize(283.15)),
    );
    assert!(lo + 2 < hi);
    let got = folded(&codes, lo, hi);
    assert_eq!(got, oracle(&codes, lo, hi));
    assert!(got.count > 1000);

    // Decoding is affine in the code, so the mean of the decoded values is
    // the decoded mean code; a sum over codes is a lawful accumulator.
    let version = floor.floor_version();
    let (mut direct, mut n) = (0f64, 0u64);
    for (i, &c) in codes.value.iter().enumerate() {
        if region(i) && valid(i) && lo < c && c < hi {
            direct += floor.decode(c as u8, version).unwrap();
            n += 1;
        }
    }
    let (f_lo, f_hi) = floor.bounds();
    let mean_code = got.sum as f64 / got.count as f64;
    let decoded = f_lo + (mean_code + 0.5) / 256.0 * (f_hi - f_lo);
    assert_eq!(n, got.count);
    assert!((direct / n as f64 - decoded).abs() < 1e-9);

    // Another floor version does not decode these codes.
    assert_eq!(floor.decode(got.min.unwrap() as u8, version + 1), None);
}

#[test]
fn a_stale_frame_and_a_foreign_plane_refuse_the_statistics_folds() {
    let world = World::new();
    let obs = tenths(SLAB_ROWS + 500);
    let mut table = obs.table();
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let mut prepared = prepare_nodes(&world, &table, &fields, region, valid).unwrap();
    for agg in [
        Agg::Count,
        Agg::Sum(VALUE),
        Agg::Min(VALUE),
        Agg::Max(VALUE),
    ] {
        let mut fold = Fold::between(&prepared, VALUE, 0, 100, agg).unwrap();
        fold.answer(&world, &table, &prepared).unwrap();

        let other = Plane::from_fn(table.space(1), table.slab_rows(0), region);
        let own = prepared.plane(0, 0).clone();
        prepared.set_plane(0, 0, other);
        assert_eq!(
            fold.answer(&world, &table, &prepared),
            Err(Refused::Misaligned { slab: 0, plane: 0 }),
            "{agg:?}"
        );
        prepared.set_plane(0, 0, own);
        fold.answer(&world, &table, &prepared).unwrap();
    }
    // A field the selection does not hold cannot be aggregated.
    assert!(matches!(
        Fold::between(&prepared, VALUE, 0, 100, Agg::Sum(person::ID)),
        Err(Refused::FieldNotSelected { field: person::ID })
    ));

    table.advance_source();
    for agg in [Agg::Count, Agg::Sum(VALUE), Agg::Max(VALUE)] {
        let mut fold = Fold::between(&prepared, VALUE, 0, 100, agg).unwrap();
        assert!(
            matches!(
                fold.answer(&world, &table, &prepared),
                Err(Refused::Lane(LaneRefusal::Frame(_)))
            ),
            "{agg:?} read a frame the source has moved past"
        );
    }
}

// ------------------------------------------------- outside the slab fold

fn plane_of(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut words = vec![0u64; n.div_ceil(64)];
    for i in (0..n).filter(|&i| set(i)) {
        words[i / 64] |= 1 << (i % 64);
    }
    words
}

fn run(
    program: &Program,
    n: usize,
    mask: &[u64],
    lanes: &[LaneRef<'_>],
    out: Out<'_>,
) -> Result<Value, ExecError> {
    let masks = [mask];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes,
    };
    let mut scratch = Scratch::for_program(program, n)?;
    let foreign = Foreign {
        planes: &[],
        lanes: &[],
    };
    execute_into(program, &planes, &foreign, &mut scratch, out)
}

#[test]
fn a_one_byte_lane_sums_in_place_but_has_no_range_predicate() {
    let n = GRID;
    let codes: Vec<u8> = (0..n as u64).map(|i| (mix(i) % 256) as u8).collect();
    let mask = plane_of(n, region);
    let lane = [LaneRef::Strided(StridedRef {
        bytes: &codes,
        first_offset: 0,
        stride: 1,
        records: n,
    })];
    let sum = Program::new(
        vec![],
        Terminal::MaskedStridedGroupSum {
            mask: Operand::Plane(0),
            lane: 0,
            groups: 1,
            group_bytes: 1,
        },
    );
    let want: i64 = (0..n)
        .filter(|&i| region(i))
        .map(|i| i64::from(codes[i]))
        .sum();
    assert_eq!(
        run(&sum, n, &mask, &lane, Out::None),
        Ok(Value::StridedSum(Some(want)))
    );

    // The ordered compares and the extrema read `I32` lanes only.
    let gt = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::GtI32 { lane: 0, t: 100 },
            under: None,
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    assert!(matches!(
        run(&gt, n, &mask, &lane, Out::None),
        Err(ExecError::LaneKind { .. })
    ));
    let min = Program::new(
        vec![],
        Terminal::MaskedMinI32 {
            mask: Operand::Plane(0),
            lane: 0,
        },
    );
    assert!(matches!(
        run(&min, n, &mask, &lane, Out::None),
        Err(ExecError::LaneKind { .. })
    ));
}

#[test]
fn per_station_statistics_are_grouped_folds_over_a_key_lane() {
    // The 1BRC shape: 413 stations, a million readings, one row each.
    const STATIONS: usize = 413;
    let obs = tenths(GRID);
    let mask = plane_of(GRID, valid);
    let lanes = [LaneRef::U32(&obs.key), LaneRef::I32(&obs.value)];

    let mut want: BTreeMap<u32, (i64, i64, i64, i64)> = BTreeMap::new();
    for i in (0..GRID).filter(|&i| valid(i)) {
        let v = i64::from(obs.value[i]);
        let e = want.entry(obs.key[i]).or_insert((0, 0, i64::MAX, i64::MIN));
        *e = (e.0 + 1, e.1 + v, e.2.min(v), e.3.max(v));
    }
    assert_eq!(want.len(), STATIONS);

    let grouped = |terminal: Terminal| {
        let mut out = vec![0i64; STATIONS];
        run(
            &Program::new(vec![], terminal),
            GRID,
            &mask,
            &lanes,
            Out::I64(&mut out),
        )
        .unwrap();
        out
    };
    let reduce = |fold| {
        grouped(Terminal::GroupReduce {
            mask: Operand::Plane(0),
            key: GroupKey::Lane(0),
            fold,
        })
    };
    let count = reduce(GroupFold::Count);
    let min = reduce(GroupFold::MinI32(1));
    let max = reduce(GroupFold::MaxI32(1));
    let sum = grouped(Terminal::GroupSumI32 {
        mask: Operand::Plane(0),
        key: 0,
        val: 1,
    });
    for (&k, &w) in &want {
        let k = k as usize;
        assert_eq!((count[k], sum[k], min[k], max[k]), w, "station {k}");
    }
    assert_eq!(
        count.iter().sum::<i64>() as usize,
        (0..GRID).filter(|&i| valid(i)).count()
    );
}
