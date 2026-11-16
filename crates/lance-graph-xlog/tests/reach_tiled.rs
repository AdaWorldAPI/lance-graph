//! `Cube::reach` on a tiled cube (D-XLOG-W5-D). A tiled cube's lanes hold wide
//! coordinates (`tile_ordinal * 256 + local`); a stored set, a set expression
//! and `reach` read the same coordinates, so the answer mask is indexed by
//! wide coordinate. Checked against a BFS over an adjacency list built here,
//! which never calls the code under test.

use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;

use lance_graph_dt64::dr::{Frame, LocalMap, MapTable, TileId, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, SetExpr};

/// Three `W256` tiles holding 0..256, 256..512, 512..694 in ascending order,
/// so global id `g` has wide coordinate `g`.
const IDS: u32 = 694;
const TILES: [TileId; 3] = [1, 2, 3];

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, k: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % u64::from(k)) as u32
    }
}

fn table() -> (MapTable, Frame) {
    let ranges: [Range<u64>; 3] = [0..256, 256..512, 512..u64::from(IDS)];
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        TILES
            .iter()
            .zip(ranges)
            .map(|(&id, r)| {
                let g: Vec<u64> = r.collect();
                (id, Some(LocalMap::prepare(Width::W256, &g).unwrap()))
            })
            .collect(),
    );
    let f = t.seal();
    (t, f)
}

/// `seed ∪ {y : x ->+ y, every step landing in target}`.
fn bfs(edges: &BTreeSet<(u32, u32)>, seed: &[u32], target: &BTreeSet<u32>) -> BTreeSet<u32> {
    let mut adj = vec![Vec::new(); IDS as usize];
    for &(a, b) in edges {
        adj[a as usize].push(b);
    }
    let mut seen: BTreeSet<u32> = seed.iter().copied().collect();
    let mut q: VecDeque<u32> = seed.iter().copied().collect();
    while let Some(x) = q.pop_front() {
        for &y in &adj[x as usize] {
            if target.contains(&y) && seen.insert(y) {
                q.push_back(y);
            }
        }
    }
    seen
}

fn positions(mask: &[u64]) -> BTreeSet<u32> {
    (0..mask.len() * 64)
        .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
        .map(|p| p as u32)
        .collect()
}

fn sample(l: &mut Lcg, k: usize) -> Vec<u32> {
    (0..k).map(|_| l.below(IDS)).collect()
}

/// Random graphs with and without a ring through all three tiles, random seed
/// and target sets: the tiled reach equals the BFS, position for position,
/// both from `Cube::reach` and from `SetExpr::Reach` under `set_mask`, and
/// the answer crosses tile boundaries.
///
/// Disable it catches: placing set ids by the first tile's local only (ids in
/// tiles 2 and 3 are refused or land on the wrong position), and a reach that
/// stops at the first tile's block.
#[test]
fn tiled_reach_matches_bfs_across_tiles() {
    let mut l = Lcg(0x7111_ed00);
    let (t, f) = table();
    for (case, ring, chords) in [(0, true, 300usize), (1, false, 900), (2, false, 200)] {
        let mut edges: BTreeSet<(u32, u32)> =
            (0..chords).map(|_| (l.below(IDS), l.below(IDS))).collect();
        if ring {
            edges.extend((0..IDS).map(|i| (i, (i + 1) % IDS)));
        }
        let v: Vec<(u32, u32)> = edges.iter().copied().collect();
        let mut cube = Cube::new_tiled(&t, &f, &TILES, &[v.as_slice()]).unwrap();
        let seed_ids = sample(&mut l, 5);
        let target_ids: Vec<u32> = (0..IDS).filter(|_| l.below(3) != 0).collect();
        let seed = SetExpr::Set(cube.add_set(&t, &seed_ids).unwrap());
        let target = SetExpr::Set(cube.add_set(&t, &target_ids).unwrap());
        let want = bfs(&edges, &seed_ids, &target_ids.iter().copied().collect());

        let (mask, _) = cube.reach(&t, 0, &seed, &target).unwrap();
        assert_eq!(positions(&mask), want, "case {case}: reach");
        let expr = SetExpr::reach(0, seed.clone(), target.clone());
        assert_eq!(
            positions(&cube.set_mask(&t, &expr).unwrap()),
            want,
            "case {case}: set_mask"
        );
        let tiles_hit: BTreeSet<u32> = want.iter().map(|p| p / 256).collect();
        assert!(
            want.len() > seed_ids.len(),
            "case {case}: reach adds nothing"
        );
        if ring {
            assert_eq!(tiles_hit.len(), 3, "case {case}: {tiles_hit:?}");
        }
    }
}

/// A stored set on a tiled cube places each id at its wide coordinate, and
/// an id no tile holds is refused, as `new_tiled` refuses it.
#[test]
fn tiled_sets_place_ids_at_wide_coordinates() {
    let (t, f) = table();
    let mut cube = Cube::new_tiled(&t, &f, &TILES, &[&[(0, 1)]]).unwrap();
    let s = SetExpr::Set(cube.add_set(&t, &[3, 300, 693]).unwrap());
    assert_eq!(
        positions(&cube.set_mask(&t, &s).unwrap()),
        BTreeSet::from([3, 300, 693])
    );
    assert!(matches!(
        cube.add_set(&t, &[IDS]),
        Err(CubeRefusal::Locate(_))
    ));
}
