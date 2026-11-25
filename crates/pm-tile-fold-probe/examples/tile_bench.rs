//! D-PM-TILE: cost of folding per sealed tile and combining, against one
//! fold over the same rows with no socket. ns per 64 lanes (best of 7), and
//! ns per part for the combine.

use std::hint::black_box;
use std::time::Instant;

use pm_tile_fold_probe::fixture::{self, Tile, FIELD, READER, ROWS};
use pm_tile_fold_probe::{bind_field, combine, fold_lane, fold_tile};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn best(mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_nanos() as f64);
    }
    best
}

fn main() {
    println!(
        "D-PM-TILE  avx512f={} avx2={}  (best of 7)",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "tiles", "lanes/tile", "fold+check", "plain fold", "combine", "ns/part"
    );
    let total_blocks = 16 * 1024; // 1M lanes in every row
    for tiles_n in [1usize, 16, 256, 4096] {
        let blocks = total_blocks / tiles_n;
        let mut s = 7u64;
        let tiles: Vec<Tile> = (0..tiles_n)
            .map(|_| {
                let codes: Vec<u16> = (0..blocks * 64)
                    .map(|_| (xorshift(&mut s) % 512) as u16)
                    .collect();
                Tile::from_codes(&codes, 1)
            })
            .collect();
        let sels: Vec<Vec<u64>> = tiles
            .iter()
            .map(|t| (0..t.blocks()).map(|_| xorshift(&mut s)).collect())
            .collect();
        let (store, route) = fixture::store();
        let view = fixture::View::new();
        let bindings: Vec<_> = tiles
            .iter()
            .map(|t| bind_field(&store, &view, t, route, ROWS, READER, FIELD).expect("bind"))
            .collect();

        let fold_all = || {
            tiles
                .iter()
                .zip(&bindings)
                .zip(&sels)
                .map(|((t, b), sel)| {
                    fold_tile(
                        b,
                        &store,
                        lance_graph_dt64::LaneSource::frames(t),
                        t,
                        FIELD,
                        sel,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let parts = fold_all();
        let presented: Vec<_> = parts
            .iter()
            .zip(&tiles)
            .map(|(p, t)| (p, lance_graph_dt64::LaneSource::frames(t)))
            .collect();
        let whole = combine(&presented).unwrap();
        let plain: pm_tile_fold_probe::Stats = tiles
            .iter()
            .zip(&sels)
            .map(|(t, sel)| fold_lane(&t.mags, &t.pos, sel).unwrap())
            .fold(Default::default(), |a, b| a.checked_add(b).unwrap());
        assert_eq!(whole, plain);

        let per = (tiles_n * blocks) as f64;
        let t_fold = best(|| {
            black_box(fold_all());
        });
        let t_plain = best(|| {
            for (t, sel) in tiles.iter().zip(&sels) {
                black_box(fold_lane(&t.mags, &t.pos, sel).unwrap());
            }
        });
        let t_comb = best(|| {
            black_box(combine(&presented).unwrap());
        });
        println!(
            "{:>8} {:>10} {:>12.2} {:>12.2} {:>12.0} {:>12.1}",
            tiles_n,
            blocks * 64,
            t_fold / per,
            t_plain / per,
            t_comb,
            t_comb / tiles_n as f64
        );
    }
    println!("(fold+check and plain fold in ns per 64 lanes; combine in ns total)");
}
