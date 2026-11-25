//! D-PM-TILE falsifiers: each part equals a scalar oracle, the combined
//! answer equals the fold of the whole population in any tile order, and a
//! part is refused once its tile moved or when presented with another tile.

use lance_graph_dt64::dr::FrameRefusal;
use pm_tile_fold_probe::fixture::{self, Tile, FIELD, READER, ROWS};
use pm_tile_fold_probe::{combine, fold_lane, fold_tile, CombineRefusal, FoldRefusal, Stats, TilePart};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn codes(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n).map(|_| (xorshift(&mut s) % 512) as u16).collect()
}

fn selection(blocks: usize, seed: u64) -> Vec<u64> {
    let mut s = seed;
    (0..blocks).map(|_| xorshift(&mut s)).collect()
}

/// The statistics of the selected codewords, row by row.
fn oracle(codes: &[u16], sel: &[u64]) -> Stats {
    let mut st = Stats::default();
    for (i, &c) in codes.iter().enumerate() {
        if sel[i / 64] >> (i % 64) & 1 == 1 {
            if let Some(v) = fixture::value_of(c) {
                st.n += 1;
                st.s += v;
                st.q += (v * v) as u64;
                st.a += v.unsigned_abs();
            }
        }
    }
    st
}

fn part(tile: &Tile, sel: &[u64]) -> TilePart {
    let (store, route) = fixture::store();
    let view = fixture::View::new();
    let b = pm_tile_fold_probe::bind_field(&store, &view, tile, route, ROWS, READER, FIELD).expect("bind");
    fold_tile(&b, &store, lance_graph_dt64::LaneSource::frames(tile), tile, FIELD, sel).expect("fold")
}

/// Tiles of unequal sizes, the last ones not a whole block.
fn tiles() -> Vec<(Vec<u16>, Vec<u64>, Tile)> {
    [64usize, 1000, 4096, 77, 1]
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let c = codes(n, 11 + k as u64);
            let t = Tile::from_codes(&c, 1);
            let sel = selection(t.blocks(), 31 + k as u64);
            (c, sel, t)
        })
        .collect()
}

/// FAILS IF: a part differs from the row-by-row oracle, including on a tile
/// padded to a whole block.
#[test]
fn each_part_equals_the_oracle() {
    for (c, sel, t) in tiles() {
        assert_eq!(part(&t, &sel).stats_unwitnessed(), oracle(&c, &sel), "{} rows", c.len());
    }
}

/// FAILS IF: combining parts differs from folding the whole population, or
/// depends on the order of the parts.
#[test]
fn combined_parts_equal_the_whole_in_any_order() {
    let ts = tiles();
    let parts: Vec<TilePart> = ts.iter().map(|(_, sel, t)| part(t, sel)).collect();
    let want = ts
        .iter()
        .map(|(c, sel, _)| oracle(c, sel))
        .fold(Stats::default(), |a, b| a.checked_add(b).unwrap());
    assert!(want.n > 1000, "the fixture selects a real population");
    let mut order: Vec<usize> = (0..ts.len()).collect();
    for rot in 0..ts.len() {
        order.rotate_left(1);
        let presented: Vec<_> = order
            .iter()
            .map(|&i| (&parts[i], lance_graph_dt64::LaneSource::frames(&ts[i].2)))
            .collect();
        assert_eq!(combine(&presented), Ok(want), "rotation {rot}");
    }
}

/// FAILS IF: a part folded before its tile was relaid still combines.
#[test]
fn a_part_of_a_relaid_tile_is_refused() {
    let mut ts = tiles();
    let parts: Vec<TilePart> = ts.iter().map(|(_, sel, t)| part(t, sel)).collect();
    let fresh = codes(1000, 99);
    ts[1].2.relay(&fresh);
    let presented: Vec<_> = parts
        .iter()
        .zip(&ts)
        .map(|(p, (_, _, t))| (p, lance_graph_dt64::LaneSource::frames(t)))
        .collect();
    assert_eq!(
        combine(&presented),
        Err(CombineRefusal::Frame {
            tile: 1,
            refusal: FrameRefusal::SourceChanged
        })
    );
    // Re-folding the relaid tile combines again, with the new values.
    let refolded = part(&ts[1].2, &ts[1].1);
    assert_eq!(refolded.stats_unwitnessed(), oracle(&fresh, &ts[1].1));
    let mut presented = presented;
    presented[1].0 = &refolded;
    assert!(combine(&presented).is_ok());
}

/// FAILS IF: a part combines when presented with a tile that did not fold it,
/// even one at equal revisions.
#[test]
fn a_part_presented_with_another_tile_is_refused() {
    let ts = tiles();
    let p0 = part(&ts[0].2, &ts[0].1);
    let other = lance_graph_dt64::LaneSource::frames(&ts[2].2);
    assert_eq!(
        combine(&[(&p0, other)]),
        Err(CombineRefusal::Frame {
            tile: 0,
            refusal: FrameRefusal::ForeignFrame
        })
    );
}

/// FAILS IF: a lane whose plane or selection describes other rows is folded.
#[test]
fn a_mismatched_shape_is_refused() {
    let t = Tile::from_codes(&codes(128, 3), 1);
    assert!(matches!(fold_lane(&t.mags, &t.pos, &[u64::MAX]), Err(FoldRefusal::Shape { .. })));
    assert!(matches!(fold_lane(&t.mags[..100], &t.pos, &[u64::MAX; 2]), Err(FoldRefusal::Shape { .. })));
    assert!(fold_lane(&t.mags, &t.pos, &[u64::MAX; 2]).is_ok());
}

/// FAILS IF: an overflow wraps instead of refusing, or variance is not exact.
#[test]
fn overflow_refuses_and_variance_is_exact() {
    let big = Stats { n: 1, s: i64::MAX, q: 0, a: 0 };
    assert_eq!(big.checked_add(Stats { n: 1, s: 1, q: 0, a: 0 }), None);
    // Untouched only: no values, variance undefined.
    let u = Tile::from_codes(&[0u16; 64], 1);
    let st = fold_lane(&u.mags, &u.pos, &[u64::MAX]).unwrap();
    assert_eq!((st, st.variance_parts()), (Stats::default(), None));
    // 63 × +200 and one +201: variance 63 / 4096.
    let mut c = vec![256 + 200u16; 64];
    c[9] = 256 + 201;
    let t = Tile::from_codes(&c, 1);
    assert_eq!(fold_lane(&t.mags, &t.pos, &[u64::MAX]).unwrap().variance_parts(), Some((63, 4096)));
}
