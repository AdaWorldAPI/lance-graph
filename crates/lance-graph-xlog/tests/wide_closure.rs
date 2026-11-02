//! Wide closure (`Cube::closure_wide`, `refresh_closure`, `refold`, `rebase`)
//! against an independent per-node DFS over an adjacency list built here.
//! The reference never calls the code under test. Fixtures come from a tiny
//! deterministic LCG, so nothing depends on `rand` or on files outside the
//! repository.

use std::collections::BTreeSet;
use std::ops::Range;

use lance_graph_dt64::dr::{Frame, LocalMap, LocateRefusal, MapTable, TileId, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal};
use lance_graph_xlog::wide::{WideRows, WideStats};

/// Positions in one block (one dt64 tile).
const BLOCK: usize = 256;

/// The three-tile fixture: tile ids, the global ids each holds, and the
/// reserved wide positions (`256 * 3`).
const TILES: [TileId; 3] = [11, 22, 33];
const N_RES: usize = 768;
/// Ids actually in use in the three-tile fixture: 0..256, 256..512, 512..600.
const N_USED: u32 = 600;

/// 64-bit LCG (Knuth's MMIX constants); high bits only.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform-ish value in `0..k` (`k > 0`).
    fn below(&mut self, k: u32) -> u32 {
        (self.next() % u64::from(k)) as u32
    }
}

/// A table whose tiles are `W256` maps holding the given global ids in
/// ascending order, so a global id `g` in tile ordinal `k` has local position
/// `g - 256 * k` and wide coordinate `g`.
fn make_table(tiles: &[(TileId, Range<u64>)]) -> (MapTable, Frame) {
    let mut t = MapTable::new(1, 1);
    let changes: Vec<(TileId, Option<LocalMap>)> = tiles
        .iter()
        .map(|(id, r)| {
            let globals: Vec<u64> = r.clone().collect();
            (*id, Some(LocalMap::prepare(Width::W256, &globals).unwrap()))
        })
        .collect();
    t.advance(1, 1, changes);
    let f = t.seal();
    (t, f)
}

fn three_tile_table() -> (MapTable, Frame) {
    make_table(&[
        (TILES[0], 0..256),
        (TILES[1], 256..512),
        (TILES[2], 512..u64::from(N_USED)),
    ])
}

/// Reference reachability by DFS from every node over an adjacency list:
/// `out[x][y]` is true when `y` is reachable from `x` in one or more steps.
fn reference(n: usize, edges: &[(u32, u32)]) -> Vec<Vec<bool>> {
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a as usize].push(b as usize);
    }
    let mut out = vec![vec![false; n]; n];
    for x in 0..n {
        let mut stack: Vec<usize> = adj[x].clone();
        while let Some(y) = stack.pop() {
            if !out[x][y] {
                out[x][y] = true;
                stack.extend(adj[y].iter().copied());
            }
        }
    }
    out
}

/// Row `x` as dense words (`blocks * 4`).
fn dense(rows: &WideRows, x: usize) -> Vec<u64> {
    let mut v = vec![0u64; rows.blocks() as usize * 4];
    rows.row_into(x as u32, &mut v);
    v
}

fn bit(row: &[u64], y: usize) -> bool {
    row.get(y / 64).is_some_and(|w| w >> (y % 64) & 1 == 1)
}

/// Every row of `a` equals the same row of `b`, read through `row_into`.
fn assert_rows_eq(a: &WideRows, b: &WideRows, what: &str) {
    assert_eq!(a.n(), b.n(), "{what}: n");
    assert_eq!(a.blocks(), b.blocks(), "{what}: blocks");
    for x in 0..a.n() as usize {
        assert_eq!(dense(a, x), dense(b, x), "{what}: row {x}");
    }
}

/// Every `(x, y)` of the reserved range agrees with the reference (positions
/// beyond it must read as unreachable). Returns the number of reachable
/// pairs.
fn assert_matches_ref(rows: &WideRows, reference: &[Vec<bool>], what: &str) -> u64 {
    assert_eq!(rows.n() as usize, reference.len(), "{what}: n");
    let mut pairs = 0u64;
    for (x, ref_row) in reference.iter().enumerate() {
        let d = dense(rows, x);
        for (y, &want) in ref_row.iter().enumerate() {
            assert_eq!(bit(&d, y), want, "{what}: ({x},{y})");
            pairs += u64::from(want);
        }
    }
    pairs
}

/// Blocks a sparse layout must store: one per `(row, block)` holding a
/// reachable position.
fn blocks_in_use(reference: &[Vec<bool>]) -> u64 {
    let mut count = 0u64;
    for row in reference {
        for chunk in row.chunks(BLOCK) {
            count += u64::from(chunk.iter().any(|&b| b));
        }
    }
    count
}

fn random_edges(l: &mut Lcg, n_used: u32, k: usize) -> BTreeSet<(u32, u32)> {
    let mut s = BTreeSet::new();
    while s.len() < k {
        s.insert((l.below(n_used), l.below(n_used)));
    }
    s
}

fn vec_of(s: &BTreeSet<(u32, u32)>) -> Vec<(u32, u32)> {
    s.iter().copied().collect()
}

fn tiled_cube(
    table: &MapTable,
    frame: &Frame,
    tiles: &[TileId],
    edges: &BTreeSet<(u32, u32)>,
) -> Cube {
    let v = vec_of(edges);
    Cube::new_tiled(table, frame, tiles, &[v.as_slice()]).unwrap()
}

/// With at most 64 positions in one `W64`-sized block, the wide kernel and
/// the narrow `Cube::closure` give the same bits, word for word, and both
/// agree with the reference.
///
/// Disable it catches: dropping the `andnot` in the wide kernel
/// (`mask_andnot(next, full, delta)` in `run_fixpoint`), so a delta
/// re-enters positions already in `full` (rows_composed and, on cycles, the
/// fixpoint bound diverge from the narrow kernel).
#[test]
fn wide_matches_narrow_bit_for_bit_up_to_64() {
    let mut l = Lcg(0x5eed_0001);
    for trial in 0..20u32 {
        let n = if trial == 19 {
            64
        } else {
            1 + l.below(64) as usize
        };
        let edges: Vec<(u32, u32)> = (0..2 * n)
            .map(|_| (l.below(n as u32), l.below(n as u32)))
            .collect();
        let (table, frame) = make_table(&[(7, 0..n as u64)]);
        let cube = Cube::new(&table, &frame, 7, &[edges.as_slice()]).unwrap();
        let (narrow, _) = cube.closure(&table, 0).unwrap();
        let (wide, _) = cube.closure_wide(&table, 0).unwrap();
        let reference = reference(n, &edges);
        assert_eq!(wide.n() as usize, n, "trial {trial}");
        assert_eq!(reference.len(), n);
        for (x, ref_row) in reference.iter().enumerate() {
            let d = dense(&wide, x);
            assert_eq!(narrow.rows()[x], d[0], "trial {trial} row {x} word 0");
            assert!(d[1..].iter().all(|&w| w == 0), "trial {trial} row {x} tail");
            for (y, &want) in ref_row.iter().enumerate() {
                let w = wide.contains(x as u32, y as u32);
                assert_eq!(narrow.contains(x as u8, y as u8), w, "t{trial} ({x},{y})");
                assert_eq!(want, w, "t{trial} ref ({x},{y})");
            }
        }
    }
}

/// 694 positions across three `W256` tiles (256 + 256 + 182), edges that form
/// a ring through all three tiles plus random chords, and a sparser random
/// graph without the ring: the wide closure equals the reference for every
/// `(x, y)` of the 768 reserved positions, and `rows_composed` equals the
/// number of reachable pairs.
///
/// Disable it catches: packing only block 0 of each row (every cross-tile
/// pair goes missing), or composing through only the first block of a delta.
#[test]
fn three_tile_graph_with_cross_tile_cycles_matches_bfs() {
    let mut l = Lcg(0x5eed_0002);
    let ids: u32 = 694;
    let (table, frame) = make_table(&[(1, 0..256), (2, 256..512), (3, 512..694)]);
    for (name, ring, chords) in [("ring+chords", true, 1500usize), ("sparse", false, 800)] {
        let mut edges: BTreeSet<(u32, u32)> = random_edges(&mut l, ids, chords);
        if ring {
            edges.extend((0..ids).map(|i| (i, (i + 1) % ids)));
        }
        let cube = tiled_cube(&table, &frame, &[1, 2, 3], &edges);
        let (rows, stats) = cube.closure_wide(&table, 0).unwrap();
        let reference = reference(N_RES, &vec_of(&edges));
        let pairs = assert_matches_ref(&rows, &reference, name);
        assert_eq!(stats.rows_composed, pairs, "{name}: rows_composed");
        assert_eq!(stats.blocks_reserved, (N_RES * 3) as u64, "{name}");
        assert_eq!(stats.blocks_allocated, blocks_in_use(&reference), "{name}");
        // The ring makes every id reach every id, across all tiles.
        if ring {
            assert_eq!(pairs, u64::from(ids) * u64::from(ids), "{name}: all pairs");
        } else {
            assert!(
                pairs > 0 && pairs < u64::from(ids) * u64::from(ids),
                "{name}"
            );
        }
    }
}

/// Block presence: edges that stay inside one tile allocate only that tile's
/// block, and the cross blocks cost nothing. The twin adds one edge from tile
/// 0 to tile 1 and the count must grow to what the reference says.
///
/// Disable it catches: allocating every block of a row (the first count
/// becomes `rows * 2`), and, for the twin, a presence bit that is never set
/// (the count would not grow).
#[test]
fn absent_blocks_allocate_nothing() {
    let mut l = Lcg(0x5eed_0003);
    let (table, frame) = make_table(&[(1, 0..256), (2, 256..512)]);
    let mut edges = BTreeSet::new();
    while edges.len() < 300 {
        edges.insert((l.below(256), l.below(256)));
    }
    while edges.len() < 600 {
        edges.insert((256 + l.below(256), 256 + l.below(256)));
    }
    let cube = tiled_cube(&table, &frame, &[1, 2], &edges);
    let (rows, stats) = cube.closure_wide(&table, 0).unwrap();
    let reference_a = reference(512, &vec_of(&edges));
    assert_matches_ref(&rows, &reference_a, "in-tile");
    let rows_with_reach = reference_a.iter().filter(|r| r.iter().any(|&b| b)).count() as u64;
    assert!(rows_with_reach > 0, "fixture must reach something");
    // Each reaching row stores exactly one block (its own tile's); no cross.
    assert_eq!(stats.blocks_allocated, rows_with_reach);
    assert_eq!(stats.blocks_allocated, blocks_in_use(&reference_a));
    assert_eq!(stats.blocks_reserved, 512 * 2);
    for x in 0..512 {
        let d = dense(&rows, x);
        let cross = if x < 256 { 256..512 } else { 0..256 };
        assert!(cross.into_iter().all(|y| !bit(&d, y)), "row {x} crossed");
    }

    // Anti-vacuity: one cross-tile edge must make blocks appear.
    let mut with_cross = edges.clone();
    with_cross.insert((7, 300));
    let cube = tiled_cube(&table, &frame, &[1, 2], &with_cross);
    let (rows_b, stats_b) = cube.closure_wide(&table, 0).unwrap();
    let reference_b = reference(512, &vec_of(&with_cross));
    assert_matches_ref(&rows_b, &reference_b, "cross");
    assert!(stats_b.blocks_allocated > stats.blocks_allocated);
    assert_eq!(stats_b.blocks_allocated, blocks_in_use(&reference_b));
}

/// One checked step of the refresh script: the refreshed closure equals the
/// closure of a fresh cube built from the effective edge set (every row, via
/// `row_into`) and the reference; `rows_recomputed` equals `|A|`, the rows
/// that are a retired source or reach one in the base closure.
#[allow(clippy::too_many_arguments)]
fn check_step(
    label: &str,
    cube: &Cube,
    table: &MapTable,
    frame: &Frame,
    old: &WideRows,
    base: &BTreeSet<(u32, u32)>,
    ref_base: &[Vec<bool>],
    eff: &BTreeSet<(u32, u32)>,
) -> WideStats {
    let (new, stats) = cube.refresh_closure(table, 0, old).unwrap();
    let fresh = tiled_cube(table, frame, &TILES, eff);
    let (rebuilt, _) = fresh.closure_wide(table, 0).unwrap();
    assert_rows_eq(&new, &rebuilt, label);
    assert_matches_ref(&new, &reference(N_RES, &vec_of(eff)), label);
    let retired_src: BTreeSet<usize> = base.difference(eff).map(|e| e.0 as usize).collect();
    let expected = (0..N_RES)
        .filter(|x| retired_src.contains(x) || retired_src.iter().any(|&a| ref_base[*x][a]))
        .count() as u64;
    assert_eq!(stats.rows_recomputed, expected, "{label}: rows_recomputed");
    stats
}

/// A scripted sequence on one cube, checked after every step: add only,
/// retire only, mixed, and an add followed by a retire of the same new edge.
/// An add-only step recomputes no row (can stay silent); a retire step
/// recomputes some rows but not all of them (not vacuous).
///
/// Disable it catches: skipping the affected-rows recompute on retire and
/// reusing the old rows (the retire steps then differ from the rebuild and
/// `rows_recomputed` is 0), or recomputing every row on every refresh (the
/// add-only `== 0` and the `|A|` equality fail).
#[test]
fn refresh_equals_rebuild_after_every_step() {
    let mut l = Lcg(0x5eed_0004);
    let (table, frame) = three_tile_table();
    let base = random_edges(&mut l, N_USED, 650);
    let base_v = vec_of(&base);
    let ref_base = reference(N_RES, &base_v);
    let mut cube = tiled_cube(&table, &frame, &TILES, &base);
    let (old, _) = cube.closure_wide(&table, 0).unwrap();
    let mut eff = base.clone();

    // Step 1: add only.
    for _ in 0..40 {
        let e = (l.below(N_USED), l.below(N_USED));
        cube.add_edge(&table, 0, e.0, e.1).unwrap();
        eff.insert(e);
    }
    assert!(eff.len() > base.len(), "step 1 must add something new");
    let s1 = check_step(
        "add-only", &cube, &table, &frame, &old, &base, &ref_base, &eff,
    );
    assert_eq!(s1.rows_recomputed, 0, "an add-only step recomputes no row");

    // Step 2: retire only, from the base.
    let mut retired = 0;
    for _ in 0..15 {
        let e = base_v[l.below(base_v.len() as u32) as usize];
        cube.retire_edge(&table, 0, e.0, e.1).unwrap();
        retired += usize::from(eff.remove(&e));
    }
    assert!(retired > 0, "step 2 must retire a base edge");
    let s2 = check_step(
        "retire-only",
        &cube,
        &table,
        &frame,
        &old,
        &base,
        &ref_base,
        &eff,
    );
    assert!(
        s2.rows_recomputed > 0,
        "a retire recomputes the rows it can reach"
    );
    assert!(s2.rows_recomputed < N_RES as u64, "but not every row");

    // Step 3: mixed.
    for _ in 0..10 {
        let e = (l.below(N_USED), l.below(N_USED));
        cube.add_edge(&table, 0, e.0, e.1).unwrap();
        eff.insert(e);
        let cur = vec_of(&eff);
        let r = cur[l.below(cur.len() as u32) as usize];
        cube.retire_edge(&table, 0, r.0, r.1).unwrap();
        eff.remove(&r);
    }
    let s3 = check_step("mixed", &cube, &table, &frame, &old, &base, &ref_base, &eff);
    assert!(s3.rows_recomputed < N_RES as u64);

    // Step 4: add then retire the same new edge: nothing changes.
    let fresh_edge = loop {
        let e = (l.below(N_USED), l.below(N_USED));
        if !eff.contains(&e) && !base.contains(&e) {
            break e;
        }
    };
    cube.add_edge(&table, 0, fresh_edge.0, fresh_edge.1)
        .unwrap();
    cube.retire_edge(&table, 0, fresh_edge.0, fresh_edge.1)
        .unwrap();
    let s4 = check_step(
        "cancel", &cube, &table, &frame, &old, &base, &ref_base, &eff,
    );
    assert_eq!(
        s4.rows_recomputed, s3.rows_recomputed,
        "a cancelled edge adds no work"
    );

    // Cancel on its own: a log of only add-then-retire leaves the closure
    // bit for bit and recomputes nothing.
    let mut quiet = tiled_cube(&table, &frame, &TILES, &base);
    quiet
        .add_edge(&table, 0, fresh_edge.0, fresh_edge.1)
        .unwrap();
    quiet
        .retire_edge(&table, 0, fresh_edge.0, fresh_edge.1)
        .unwrap();
    assert_eq!(quiet.delta_len(0), 2);
    let (same, s) = quiet.refresh_closure(&table, 0, &old).unwrap();
    assert_rows_eq(&same, &old, "cancel-only");
    assert_eq!(s.rows_recomputed, 0);
}

/// The log is merged on read and folded only when asked: `closure_wide`
/// leaves `delta_len` alone, `refold` empties it, and the closure is the
/// same before and after.
///
/// Disable it catches: a `refold` that forgets to empty the log (delta_len
/// stays > 0) or one that drops the retired edges (rows differ), and a
/// `closure_wide` that folds on its own (delta_len would hit 0 early).
#[test]
fn refold_is_explicit_and_preserves_the_closure() {
    let mut l = Lcg(0x5eed_0005);
    let (table, frame) = three_tile_table();
    let base = random_edges(&mut l, N_USED, 650);
    let base_v = vec_of(&base);
    let mut cube = tiled_cube(&table, &frame, &TILES, &base);
    let mut eff = base.clone();
    for _ in 0..20 {
        let e = (l.below(N_USED), l.below(N_USED));
        cube.add_edge(&table, 0, e.0, e.1).unwrap();
        eff.insert(e);
    }
    for _ in 0..10 {
        let e = base_v[l.below(base_v.len() as u32) as usize];
        cube.retire_edge(&table, 0, e.0, e.1).unwrap();
        eff.remove(&e);
    }
    assert_eq!(cube.delta_len(0), 30);
    let (before, _) = cube.closure_wide(&table, 0).unwrap();
    assert_eq!(cube.delta_len(0), 30, "reading must not fold");
    assert_matches_ref(&before, &reference(N_RES, &vec_of(&eff)), "before refold");

    cube.refold(0).unwrap();
    assert_eq!(cube.delta_len(0), 0, "refold empties the log");
    let (after, _) = cube.closure_wide(&table, 0).unwrap();
    assert_rows_eq(&before, &after, "refold");
    assert_matches_ref(&after, &reference(N_RES, &vec_of(&eff)), "after refold");
}

/// A pure extension of a tile map (ids appended under a new frame) is
/// accepted by `rebase`, after which an appended id can be used. A reordered
/// map is refused as `Renumbered(tile)` and the cube keeps answering under
/// its previous frame.
///
/// Disable it catches: a `rebase` that skips the `extends` check (the
/// reorder is accepted), or one that adopts the frame before checking (the
/// cube is left on the refused frame and the last assertions fail).
#[test]
fn rebase_accepts_prefix_extension_and_refuses_reorder() {
    let globals_b: Vec<u64> = (100..200).collect();
    let (mut table, frame_a) = make_table(&[(10, 0..100), (20, 100..200)]);
    let edges = [(0u32, 1u32), (1, 150), (150, 199), (199, 0), (120, 121)];
    let mut cube = Cube::new_tiled(&table, &frame_a, &[10, 20], &[edges.as_slice()]).unwrap();
    let (rows_a, _) = cube.closure_wide(&table, 0).unwrap();

    // Extension: tile 20 gains ids 5000 and 5001 at the end.
    let mut extended = globals_b.clone();
    extended.extend([5000, 5001]);
    table.advance(
        2,
        2,
        vec![(20, Some(LocalMap::prepare(Width::W256, &extended).unwrap()))],
    );
    let frame_b = table.seal();
    assert_eq!(cube.rebase(&table, &frame_b), Ok(()));
    let (rows_b, _) = cube.closure_wide(&table, 0).unwrap();
    assert_rows_eq(&rows_a, &rows_b, "existing positions keep their rows");
    // 5000 is local 100 of tile ordinal 1: wide coordinate 356. 199 is 355.
    cube.add_edge(&table, 0, 199, 5000).unwrap();
    let (rows_c, _) = cube.closure_wide(&table, 0).unwrap();
    assert!(bit(&dense(&rows_c, 355), 356), "appended id is addressable");

    // Reorder: the SAME table advances again, tile 20 swapping its first two
    // ids. (Previously a second, unrelated table was used and the cube's
    // private snapshot supplied the old layout; the cube now reads its old
    // layout through the table's index, so the comparison must stay within
    // one table. A foreign table's frame is refused as `Map(Stale)`.)
    let mut swapped = extended.clone();
    swapped.swap(0, 1);
    table.advance(
        3,
        3,
        vec![(20, Some(LocalMap::prepare(Width::W256, &swapped).unwrap()))],
    );
    let frame_c = table.seal();
    assert_eq!(
        cube.rebase(&table, &frame_c),
        Err(CubeRefusal::Renumbered(20))
    );
    // Still on frame_b: the same table answers, and the logged edge stays.
    let (rows_d, _) = cube.closure_wide(&table, 0).unwrap();
    assert_rows_eq(&rows_c, &rows_d, "refused rebase leaves the cube alone");
    assert_eq!(cube.delta_len(0), 1);
}

/// An id that no tile of the cube holds is refused by `add_edge` and
/// `retire_edge`, on either endpoint, and nothing is logged: growth is a new
/// frame (see `rebase`), never an in-place resize.
///
/// Disable it catches: placing an unknown id at a default coordinate (the
/// call succeeds and the log grows), or logging the source before the
/// destination is checked (delta_len becomes 1 after a refused call).
#[test]
fn growth_is_a_new_frame() {
    let (table, frame) = make_table(&[(10, 0..100), (20, 100..200)]);
    let edges = [(0u32, 1u32), (1, 150)];
    let mut tiled = Cube::new_tiled(&table, &frame, &[10, 20], &[edges.as_slice()]).unwrap();
    let (before, _) = tiled.closure_wide(&table, 0).unwrap();
    for (a, b) in [(5000, 1), (1, 5000), (5000, 5001)] {
        assert_eq!(
            tiled.add_edge(&table, 0, a, b),
            Err(CubeRefusal::Locate(LocateRefusal::Unmapped)),
            "add ({a},{b})"
        );
        assert_eq!(
            tiled.retire_edge(&table, 0, a, b),
            Err(CubeRefusal::Locate(LocateRefusal::Unmapped)),
            "retire ({a},{b})"
        );
    }
    assert_eq!(tiled.delta_len(0), 0);
    let (after, _) = tiled.closure_wide(&table, 0).unwrap();
    assert_rows_eq(&before, &after, "tiled");

    // The single-tile cube refuses the same way, through its one map.
    let (table1, frame1) = make_table(&[(7, 0..100)]);
    let e1 = [(0u32, 1u32)];
    let mut single = Cube::new(&table1, &frame1, 7, &[e1.as_slice()]).unwrap();
    assert!(matches!(
        single.add_edge(&table1, 0, 1, 5000),
        Err(CubeRefusal::Domain(_))
    ));
    assert!(matches!(
        single.retire_edge(&table1, 0, 5000, 1),
        Err(CubeRefusal::Domain(_))
    ));
    assert_eq!(single.delta_len(0), 0);
    // Anti-vacuity: a known pair is accepted by the same call.
    single.add_edge(&table1, 0, 1, 2).unwrap();
    assert_eq!(single.delta_len(0), 1);
}
