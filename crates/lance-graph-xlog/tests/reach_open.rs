//! Three-valued reachability with open nodes, written with the existing
//! [`Cube::reach`] and [`SetExpr`]: no new API.
//!
//! A node is *open* when its outgoing relations may exist but are not known
//! (masked, partial or uncovered evidence). The rule is sentinel-rs's
//! `sentinel_csi::reason` (EXO-298): a breadth-first search from `from` that
//! reaches open nodes but never expands them. `to` is
//! - `Reachable` when the search reaches it,
//! - `Unknown` when it does not, but the search reached an open node
//!   (`from` included),
//! - `Unreachable` otherwise.
//!
//! As set expressions over stored sets `P = {from}`, `T = {to}`, `C`
//! (closed) and `O` (open):
//! - `R1 = reach(P ∧ C, C)`: the closed nodes reached through closed nodes.
//! - `Reachable` iff `to ∈ reach(R1, C ∨ T)`: one more step from `R1` may
//!   land on `to` even when `to` is open.
//! - `Unknown` iff `(P ∨ reach(R1, O)) ∧ O` is non-empty. `reach(R1, O)`
//!   also expands open nodes, but its steps land only in `O`, so it adds no
//!   closed node and the set is empty exactly when no node of `R1` has an
//!   edge to an open node.
//!
//! `from == to` is left out: `reason` never tests `from` against `to`, so
//! that case is Reachable only through a cycle, and the seeds of `reach`
//! include `from`. A caller with that question asks it separately.

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, SetExpr};
use std::collections::{BTreeSet, VecDeque};

const TILE: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    Reachable,
    Unreachable,
    Unknown,
}

fn table(n: u32) -> MapTable {
    let all: Vec<u64> = (0..u64::from(n)).collect();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// The rule as a breadth-first search, written after `sentinel_csi::reason`
/// and sharing no code with the cube.
fn oracle(edges: &[(u32, u32)], open: &BTreeSet<u32>, from: u32, to: u32) -> Reach {
    let mut seen = BTreeSet::from([from]);
    let mut todo = VecDeque::from([from]);
    let mut hit_open = false;
    while let Some(x) = todo.pop_front() {
        if open.contains(&x) {
            hit_open = true;
            continue;
        }
        for &(a, b) in edges {
            if a == x && seen.insert(b) {
                if b == to {
                    return Reach::Reachable;
                }
                todo.push_back(b);
            }
        }
    }
    if hit_open {
        Reach::Unknown
    } else {
        Reach::Unreachable
    }
}

fn has(mask: &[u64], p: u32) -> bool {
    mask[(p / 64) as usize] >> (p % 64) & 1 == 1
}

/// The rule as three expressions over stored sets. `all` is stored because
/// a `SetExpr` has no complement of the universe.
fn cube_reach(
    t: &MapTable,
    edges: &[(u32, u32)],
    n: u32,
    open: &BTreeSet<u32>,
    from: u32,
    to: u32,
) -> Reach {
    let mut cube = Cube::new(t, &t.seal(), TILE, &[edges]).unwrap();
    let all: Vec<u32> = (0..n).collect();
    let open_ids: Vec<u32> = open.iter().copied().collect();
    let a = SetExpr::Set(cube.add_set(t, &all).unwrap());
    let o = SetExpr::Set(cube.add_set(t, &open_ids).unwrap());
    let p = SetExpr::Set(cube.add_set(t, &[from]).unwrap());
    let tt = SetExpr::Set(cube.add_set(t, &[to]).unwrap());
    let c = SetExpr::and_not(a, o.clone());

    let r1 = SetExpr::reach(0, SetExpr::and(p.clone(), c.clone()), c.clone());
    let to_reached = SetExpr::and(
        SetExpr::reach(0, r1.clone(), SetExpr::or(c, tt.clone())),
        tt,
    );
    if cube
        .set_mask(t, &to_reached)
        .unwrap()
        .iter()
        .any(|&w| w != 0)
    {
        debug_assert!(has(&cube.set_mask(t, &to_reached).unwrap(), to));
        return Reach::Reachable;
    }
    let hit = SetExpr::and(SetExpr::or(p, SetExpr::reach(0, r1, o.clone())), o);
    if cube.set_mask(t, &hit).unwrap().iter().any(|&w| w != 0) {
        Reach::Unknown
    } else {
        Reach::Unreachable
    }
}

/// The cube expressions agree with the breadth-first rule over random
/// graphs, open sets and endpoints, and every answer occurs.
#[test]
fn open_node_reach_matches_the_breadth_first_rule() {
    let mut seen = [0usize; 3];
    for s in 1..=400u64 {
        let mut next = rng(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n = 2 + (next() % 63) as u32;
        let t = table(n);
        let m = (next() % (2 * u64::from(n))) as usize;
        let edges: Vec<(u32, u32)> = (0..m)
            .map(|_| {
                (
                    (next() % u64::from(n)) as u32,
                    (next() % u64::from(n)) as u32,
                )
            })
            .collect();
        let open: BTreeSet<u32> = (0..n).filter(|_| next().is_multiple_of(6)).collect();
        let from = (next() % u64::from(n)) as u32;
        let to = (next() % u64::from(n)) as u32;
        if from == to {
            continue;
        }
        let want = oracle(&edges, &open, from, to);
        let got = cube_reach(&t, &edges, n, &open, from, to);
        assert_eq!(
            got, want,
            "seed {s}: n={n} from={from} to={to} open={open:?}"
        );
        seen[want as usize] += 1;
    }
    // Anti-vacuity: all three answers are exercised.
    assert!(seen.iter().all(|&k| k >= 20), "answers seen {seen:?}");
}

/// The cases that tell the rule apart from plain reachability: an open node
/// is reached but not expanded, an open `to` is still reached, and an open
/// `from` gives Unknown.
#[test]
fn open_nodes_are_reached_but_not_expanded() {
    // 0 -> 1 -> 2, 0 -> 3.
    let edges = [(0, 1), (1, 2), (0, 3)];
    let t = table(4);
    let open = |ids: &[u32]| ids.iter().copied().collect::<BTreeSet<u32>>();
    let cases: [(&[u32], u32, u32, Reach); 6] = [
        (&[], 0, 2, Reach::Reachable),
        (&[1], 0, 2, Reach::Unknown), // 1 is reached, its edge to 2 is not walked
        (&[2], 0, 2, Reach::Reachable), // an open `to` is still reached
        (&[3], 0, 2, Reach::Reachable), // an open node off the path changes nothing
        (&[3], 1, 0, Reach::Unreachable),
        (&[0], 0, 2, Reach::Unknown), // an open `from` is never expanded
    ];
    for (o, from, to, want) in cases {
        let o = open(o);
        assert_eq!(
            oracle(&edges, &o, from, to),
            want,
            "oracle {o:?} {from}->{to}"
        );
        assert_eq!(
            cube_reach(&t, &edges, 4, &o, from, to),
            want,
            "cube {o:?} {from}->{to}"
        );
    }
}
