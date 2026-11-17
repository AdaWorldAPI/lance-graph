//! Cost of `MapTable::locate` (global id -> (tile ordinal, local)) as the
//! number of listed tiles grows. `locate` reads every listed tile so that an id
//! held by two of them is refused, so its cost is linear in the tile count.
//!
//! Run: `cargo run --release -p lance-graph-dt64 --example locate_cost`
//!
//! A report, not an assertion on a threshold. Every located id is checked
//! (`global(local)` of the returned tile is the id), and an id held by no tile
//! is refused as `Unmapped`.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_dt64::dr::{LocalMap, LocateRefusal, MapTable, Width};

fn main() {
    println!("tiles  ids     ns/locate  ns/locate/tile  ns/local(one tile)");
    for tiles in [1usize, 4, 16, 64, 256] {
        // Tile t holds the 256 globals t*256 .. t*256+255, in a scrambled local
        // order (the lookup index is sorted inside LocalMap, the locals are not).
        let mut table = MapTable::new(1, 1);
        let ids: Vec<u32> = (0..tiles as u32).map(|t| 100 + t).collect();
        let maps = ids
            .iter()
            .enumerate()
            .map(|(t, &tile)| {
                let g: Vec<u64> = (0..256u64)
                    .map(|i| (t as u64) * 256 + (i * 167) % 256)
                    .collect();
                (tile, Some(LocalMap::prepare(Width::W256, &g).unwrap()))
            })
            .collect();
        table.advance(1, 1, maps);
        let frame = table.seal();
        let n = tiles * 256;

        // Correctness, outside the timed loop.
        for g in 0..n as u64 {
            let (ord, local) = table.locate(&frame, &ids, g).unwrap();
            let map = table.map(&frame, ids[ord as usize]).unwrap();
            assert_eq!(map.global(local), Some(g), "locate {g} at {tiles} tiles");
        }
        assert_eq!(
            table.locate(&frame, &ids, n as u64),
            Err(LocateRefusal::Unmapped)
        );

        let mut ts = Vec::new();
        for _ in 0..21 {
            let t0 = Instant::now();
            for g in 0..n as u64 {
                black_box(table.locate(&frame, &ids, black_box(g)).unwrap());
            }
            ts.push(t0.elapsed().as_nanos() as f64 / n as f64);
        }
        ts.sort_by(f64::total_cmp);
        let per = ts[10];

        // The per-tile floor: one LocalMap::local over the first tile.
        let map = table.map(&frame, ids[0]).unwrap();
        let mut ls = Vec::new();
        for _ in 0..21 {
            let t0 = Instant::now();
            for g in 0..256u64 {
                black_box(map.local(black_box(g)).unwrap());
            }
            ls.push(t0.elapsed().as_nanos() as f64 / 256.0);
        }
        ls.sort_by(f64::total_cmp);
        println!(
            "{tiles:<6} {n:<7} {per:<10.1} {:<15.1} {:.1}",
            per / tiles as f64,
            ls[10]
        );
    }
}
