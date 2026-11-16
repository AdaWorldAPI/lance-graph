//! D-XLOG-W5-B: stored one-column sets, expressions over them, and
//! [`Cube::reach`]. The acceptance target is the G1c witness for XLOG's
//! `replay_reachable` (`crates/xlog-coverage-probe/tests/witness.rs`,
//! `p4_replay_witness`): frontier {1,2,3}, widened {1}, edge 1->2->3->4,
//! blocked {3}; replay = {1,2,3} (4 is reached by an edge but is not on the
//! frontier), rollback = {3}.

use lance_graph_dt64::dr::{LocalMap, MapRead, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, SetExpr};

const TILE: u32 = 5;

fn table(ids: &[u32]) -> MapTable {
    let mut all: Vec<u64> = ids.iter().map(|&i| u64::from(i)).collect();
    all.sort_unstable();
    all.dedup();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

/// The global ids of a position mask, sorted.
fn ids(t: &MapTable, mask: &[u64]) -> Vec<u64> {
    let map = t.map(&t.seal(), TILE).unwrap();
    let globals = map.globals();
    let mut out: Vec<u64> = (0..globals.len())
        .filter(|&p| mask.get(p / 64).is_some_and(|w| w >> (p % 64) & 1 == 1))
        .map(|p| globals[p])
        .collect();
    out.sort_unstable();
    out
}

fn and(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(x, y)| x & y).collect()
}

struct Fixture {
    t: MapTable,
    cube: Cube,
    frontier: SetExpr,
    widened: SetExpr,
    blocked: SetExpr,
}

/// The G1c witness, with `blocked` as given.
fn fixture(blocked: &[u32]) -> Fixture {
    let edge: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4)];
    let t = table(&[1, 2, 3, 4]);
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[edge]).unwrap();
    let frontier = SetExpr::Set(cube.add_set(&t, &[1, 2, 3]).unwrap());
    let widened = SetExpr::Set(cube.add_set(&t, &[1]).unwrap());
    let blocked = SetExpr::Set(cube.add_set(&t, blocked).unwrap());
    Fixture {
        t,
        cube,
        frontier,
        widened,
        blocked,
    }
}

#[test]
fn replay_reachable_matches_the_g1c_witness() {
    let f = fixture(&[3]);
    let seed = SetExpr::and(f.frontier.clone(), f.widened.clone());
    // promoted = frontier ∧ widened = {1}. With an empty target, reach
    // returns its seed and nothing else.
    let empty = SetExpr::and_not(seed.clone(), seed.clone());
    let (seed_only, _) = f.cube.reach(&f.t, 0, &seed, &empty).unwrap();
    assert_eq!(ids(&f.t, &seed_only), [1]);

    let (replay, stats) = f.cube.reach(&f.t, 0, &seed, &f.frontier).unwrap();
    assert_eq!(
        ids(&f.t, &replay),
        [1, 2, 3],
        "node 4 is not on the frontier"
    );
    // Rounds 1 and 2 add 2 and 3; round 3 composes 3's edge to 4, which the
    // frontier filter removes, and finds nothing new.
    assert_eq!(stats.rounds, 3);

    let (blocked, _) = f
        .cube
        .reach(
            &f.t,
            0,
            &f.blocked,
            &SetExpr::and_not(f.blocked.clone(), f.blocked.clone()),
        )
        .unwrap();
    assert_eq!(ids(&f.t, &and(&replay, &blocked)), [3], "rollback_hit");
}

#[test]
fn blocking_an_unreached_node_leaves_rollback_empty() {
    let f = fixture(&[4]);
    let seed = SetExpr::and(f.frontier.clone(), f.widened.clone());
    let (replay, _) = f.cube.reach(&f.t, 0, &seed, &f.frontier).unwrap();
    let nothing = SetExpr::and_not(f.blocked.clone(), f.blocked.clone());
    let (blocked, _) = f.cube.reach(&f.t, 0, &f.blocked, &nothing).unwrap();
    assert_eq!(ids(&f.t, &blocked), [4]);
    assert!(ids(&f.t, &and(&replay, &blocked)).is_empty());
}

/// The target filter applies to the step's destination, not its source: on
/// this fixture every edge's source is on the frontier, so a source filter
/// would let 4 in.
#[test]
fn the_filter_is_on_the_target_column() {
    let edge: &[(u32, u32)] = &[(1, 2), (2, 4), (4, 3)];
    let t = table(&[1, 2, 3, 4]);
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[edge]).unwrap();
    let seed = SetExpr::Set(cube.add_set(&t, &[1]).unwrap());
    // 2 and 3 are targets; 4 is not, so 3 is only reachable through 4.
    let target = SetExpr::Set(cube.add_set(&t, &[2, 3]).unwrap());
    let (got, _) = cube.reach(&t, 0, &seed, &target).unwrap();
    assert_eq!(ids(&t, &got), [1, 2]);
}

/// A deterministic xorshift, so the differential is reproducible.
fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// `reach(S)` against two oracles over random graphs of up to 64 positions:
/// a breadth-first search that shares no code with the cube, and the cube's
/// own `closure` on the target-filtered edges, `S ∪ row_OR(closure(E_F), S)`.
#[test]
fn reach_agrees_with_bfs_and_with_closure_on_filtered_edges() {
    let mut nonempty_growth = 0;
    for seed_val in 1..=40u64 {
        let mut next = rng(seed_val.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n = 2 + (next() % 63) as u32;
        let ids_all: Vec<u32> = (0..n).collect();
        let t = table(&ids_all);
        let m = (next() % (3 * u64::from(n))) as usize;
        let edges: Vec<(u32, u32)> = (0..m)
            .map(|_| {
                (
                    (next() % u64::from(n)) as u32,
                    (next() % u64::from(n)) as u32,
                )
            })
            .collect();
        let seed: Vec<u32> = ids_all
            .iter()
            .copied()
            .filter(|_| next().is_multiple_of(5))
            .collect();
        let target: Vec<u32> = ids_all
            .iter()
            .copied()
            .filter(|_| !next().is_multiple_of(3))
            .collect();

        // BFS oracle.
        let mut want: Vec<u32> = seed.clone();
        let mut frontier = seed.clone();
        while let Some(x) = frontier.pop() {
            for &(a, b) in &edges {
                if a == x && target.contains(&b) && !want.contains(&b) {
                    want.push(b);
                    frontier.push(b);
                }
            }
        }
        want.sort_unstable();
        if want.len() > seed.len() {
            nonempty_growth += 1;
        }

        let filtered: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|(_, b)| target.contains(b))
            .collect();
        let mut cube = Cube::new(&t, &t.seal(), TILE, &[&edges, &filtered]).unwrap();
        let s = SetExpr::Set(cube.add_set(&t, &seed).unwrap());
        let tg = SetExpr::Set(cube.add_set(&t, &target).unwrap());
        let (got, _) = cube.reach(&t, 0, &s, &tg).unwrap();
        let got_ids: Vec<u64> = ids(&t, &got);
        let want64: Vec<u64> = want.iter().map(|&x| u64::from(x)).collect();
        assert_eq!(got_ids, want64, "bfs, seed {seed_val}");

        // closure oracle: S ∪ row_OR(closure(E_F), S). Global id = local here.
        let (closure, _) = cube.closure(&t, 1).unwrap();
        let mut via = 0u64;
        for &x in &seed {
            via |= 1 << x;
            via |= closure.rows()[x as usize];
        }
        assert_eq!(got[0], via, "closure differential, seed {seed_val}");
    }
    // Anti-vacuity: the recursion did real work on most graphs.
    assert!(
        nonempty_growth >= 20,
        "only {nonempty_growth} graphs grew past the seed"
    );
}

/// Composing expressions and evaluating them runs no mask-risc program.
#[test]
fn reach_runs_no_program() {
    let f = fixture(&[3]);
    let seed = SetExpr::and(f.frontier.clone(), f.widened.clone());
    let before = f.cube.stats();
    let _ = SetExpr::or(
        seed.clone(),
        SetExpr::and_not(f.frontier.clone(), f.blocked.clone()),
    );
    f.cube.reach(&f.t, 0, &seed, &f.frontier).unwrap();
    assert_eq!(f.cube.stats(), before);
    assert_eq!(before.programs, 0);
}

#[test]
fn refusals() {
    let f = fixture(&[3]);
    let seed = SetExpr::and(f.frontier.clone(), f.widened.clone());
    // A set this cube never stored.
    assert_eq!(
        f.cube.reach(&f.t, 0, &SetExpr::Set(99), &f.frontier),
        Err(CubeRefusal::UnknownSet(99))
    );
    assert_eq!(
        f.cube.reach(
            &f.t,
            0,
            &seed,
            &SetExpr::and(f.frontier.clone(), SetExpr::Set(7))
        ),
        Err(CubeRefusal::UnknownSet(7))
    );
    // A relation this cube never held.
    assert_eq!(
        f.cube.reach(&f.t, 9, &seed, &f.frontier),
        Err(CubeRefusal::UnknownRel(9))
    );
    // A table that never issued the cube's frame: the sets were placed under
    // the cube's frame and are not read under another.
    let alien = table(&[1, 2, 3, 4]);
    assert!(matches!(
        f.cube.reach(&alien, 0, &seed, &f.frontier),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
    // An id outside the map.
    let mut f = fixture(&[3]);
    assert!(matches!(
        f.cube.add_set(&f.t, &[1000]),
        Err(CubeRefusal::Domain(_))
    ));
}

/// Advancing the table keeps the cube's frame readable, and the answer
/// identical, exactly as for relations.
#[test]
fn an_advanced_table_keeps_the_answer() {
    let mut f = fixture(&[3]);
    let seed = SetExpr::and(f.frontier.clone(), f.widened.clone());
    let before = f.cube.reach(&f.t, 0, &seed, &f.frontier).unwrap().0;
    f.t.advance(1, 1, vec![]);
    assert_eq!(f.cube.reach(&f.t, 0, &seed, &f.frontier).unwrap().0, before);
}
