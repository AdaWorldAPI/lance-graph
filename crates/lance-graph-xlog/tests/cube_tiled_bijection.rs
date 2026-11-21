//! D-REGC-8: the cube's coordinates across several tiles, before and after
//! logged edges and a refold, and what placing a key costs.
//!
//! Three `W256` tiles hold 600 scattered ids in a shuffled local order, so a
//! wide coordinate (`tile_ordinal * 256 + local`) is never the id itself and
//! is not sorted by id. For every id the test checks:
//!
//! - exactly one coordinate;
//! - no two ids share a coordinate;
//! - the coordinate decodes back to the id through its tile's pointer table.
//!
//! An id no tile holds has no coordinate. A coordinate comes from the tile
//! maps alone, so the last part of the test pins that `add_edge`,
//! `retire_edge` and `refold` do not rebase or renumber them: the row order
//! of the relation moves, the coordinates do not. Each key placement costs one counted lookup per listed
//! tile (`MapTable::locate` reads every listed tile, to refuse an id held
//! twice).

use std::collections::BTreeMap;

use lance_graph_dt64::dr::{Frame, LocalMap, LocateRefusal, MapTable, TileId, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, SetExpr};

const TILES: [TileId; 3] = [4, 5, 6];
const SIZES: [usize; 3] = [256, 256, 88];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// 600 distinct ids, shuffled, split across the three tiles in that order.
fn ids(l: &mut Lcg) -> Vec<Vec<u64>> {
    let mut all: Vec<u64> = Vec::new();
    while all.len() < SIZES.iter().sum::<usize>() {
        let id = 1_000_000 + l.next() % 50_000_000;
        if !all.contains(&id) {
            all.push(id);
        }
    }
    let mut out = Vec::new();
    let mut at = 0;
    for n in SIZES {
        out.push(all[at..at + n].to_vec());
        at += n;
    }
    out
}

fn table(tiles: &[Vec<u64>]) -> (MapTable, Frame) {
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        TILES
            .iter()
            .zip(tiles)
            .map(|(&id, g)| (id, Some(LocalMap::prepare(Width::W256, g).unwrap())))
            .collect(),
    );
    let f = t.seal();
    (t, f)
}

fn lookups(t: &MapTable, f: &Frame) -> u64 {
    TILES
        .iter()
        .map(|&tile| t.map(f, tile).unwrap().lookups())
        .sum()
}

/// Each id's coordinate in `cube`, read through a one-id set.
fn coordinates(cube: &mut Cube, t: &MapTable, all: &[u64]) -> BTreeMap<u64, usize> {
    let mut out = BTreeMap::new();
    for &id in all {
        let set = cube.add_set(t, &[id as u32]).expect("held");
        let mask = cube.set_mask(t, &SetExpr::Set(set)).expect("mask");
        let bits: Vec<usize> = (0..mask.len() * 64)
            .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
            .collect();
        assert_eq!(bits.len(), 1, "id {id} has exactly one coordinate");
        out.insert(id, bits[0]);
    }
    out
}

#[test]
fn coordinates_are_a_bijection_across_tiles_and_survive_edges_and_refold() {
    let mut l = Lcg(0xD2E6_0008);
    let tiles = ids(&mut l);
    let all: Vec<u64> = tiles.concat();
    let (t, f) = table(&tiles);
    let pick = |l: &mut Lcg| all[(l.next() % all.len() as u64) as usize] as u32;
    let mut edges: Vec<(u32, u32)> = (0..900).map(|_| (pick(&mut l), pick(&mut l))).collect();
    edges.extend_from_within(..20); // duplicates: a bag
    let mut cube = Cube::new_tiled(&t, &f, &TILES, &[edges.as_slice()]).unwrap();

    // One lookup per listed tile for every placed key.
    let before = lookups(&t, &f);
    let coord = coordinates(&mut cube, &t, &all);
    assert_eq!(
        lookups(&t, &f) - before,
        (all.len() * TILES.len()) as u64,
        "locate reads every listed tile once per key"
    );

    // Injective, and decodes back through the tile's pointer table.
    let mut seen = BTreeMap::new();
    for (&id, &p) in &coord {
        assert!(seen.insert(p, id).is_none(), "coordinate {p} shared");
        let (ordinal, local) = (p / 256, (p % 256) as u8);
        let back = t.map(&f, TILES[ordinal]).unwrap().global(local);
        assert_eq!(back, Some(id), "global(local) == id at {p}");
    }
    let crossing = coord.values().filter(|&&p| p >= 512).count();
    assert_eq!(crossing, 88, "the third tile is used");
    assert!(
        coord.iter().any(|(&id, &p)| p as u64 != id),
        "a coordinate is not the id"
    );

    // An id no tile holds has no coordinate.
    assert!(matches!(
        cube.add_set(&t, &[7]),
        Err(CubeRefusal::Locate(LocateRefusal::Unmapped))
    ));

    // Edges logged, retired and folded: the row order moves, coordinates do not.
    let (a, b) = (pick(&mut l), pick(&mut l));
    cube.add_edge(&t, 0, a, b).unwrap();
    cube.retire_edge(&t, 0, edges[0].0, edges[0].1).unwrap();
    assert_eq!(
        coordinates(&mut cube, &t, &all),
        coord,
        "after logged edges"
    );
    cube.refold(0).unwrap();
    assert_eq!(coordinates(&mut cube, &t, &all), coord, "after refold");
}
