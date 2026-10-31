//! `trace` returns the derivation semi-naive evaluation performed.

use std::collections::{BTreeSet, VecDeque};

use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Scope,
};
use xlog_relcomp_probe::recursive::{closure, support, trace};
use xlog_relcomp_probe::{AddressMap, SLOTS, register};

const N: Axis = Axis {
    class: 30,
    kind: AxisKind::Identity,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

const DIAMOND: &[(u32, u32)] = &[(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)];
const CYCLE: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 1), (3, 4)];

struct Run {
    map: AddressMap,
    edge: [u64; SLOTS],
    path: [u64; SLOTS],
}

fn run(edges: &[(u32, u32)]) -> Run {
    let map = AddressMap::new(N, edges.iter().flat_map(|&(a, b)| [a, b])).unwrap();
    let edge = register(edges, &map, &map).unwrap();
    let st = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let g = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    let sig = MethodSignature::presence(40, 0, N, N);
    let path = closure(&st, &g, (FOCUS, SCOPE), sig, &edge, map.valid())
        .unwrap()
        .rows;
    Run { map, edge, path }
}

/// BFS shortest path length in edges from `x` to `y` (at least 1).
fn bfs_len(edges: &[(u32, u32)], x: u32, y: u32) -> Option<usize> {
    let mut dist = std::collections::BTreeMap::new();
    let mut q = VecDeque::new();
    for &(a, b) in edges {
        if a == x && !dist.contains_key(&b) {
            dist.insert(b, 1usize);
            q.push_back(b);
        }
    }
    while let Some(u) = q.pop_front() {
        let d = dist[&u];
        for &(a, b) in edges {
            if a == u && !dist.contains_key(&b) {
                dist.insert(b, d + 1);
                q.push_back(b);
            }
        }
    }
    dist.get(&y).copied()
}

fn check_all(edges: &[(u32, u32)], label: &str) {
    let r = run(edges);
    let valid = r.map.valid();
    for x in 0..r.map.len() {
        for y in 0..r.map.len() {
            let holds = r.path[x] >> y & 1 == 1;
            let t = trace(&r.edge, valid, x, y);
            if !holds {
                assert!(t.is_none(), "{label}: ({x},{y}) does not hold");
                continue;
            }
            let chain = t.unwrap_or_else(|| panic!("{label}: ({x},{y}) holds, trace is None"));
            // Fails if the chain does not start at x or end at y.
            assert_eq!(chain.first().unwrap().0, x, "{label}: start");
            assert_eq!(chain.last().unwrap().1, y, "{label}: end");
            // Fails if consecutive facts do not share endpoints.
            for w in chain.windows(2) {
                assert_eq!(w[0].1, w[1].0, "{label}: link");
            }
            // Fails if a fact is not a bit of the register.
            let sup = support(&r.edge, &r.path, valid, x, y);
            for &(u, v) in &chain {
                assert_eq!(r.edge[u] >> v & 1, 1, "{label}: ({u},{v}) not in edge");
                // Fails if the chain uses a fact outside support.
                assert_eq!(sup[u] >> v & 1, 1, "{label}: ({u},{v}) not in support");
            }
            // Fails if the first-derivation round is not the shortest distance.
            let (ix, iy) = (r.map.ids(1 << x)[0], r.map.ids(1 << y)[0]);
            assert_eq!(
                Some(chain.len()),
                bfs_len(edges, ix, iy),
                "{label}: length of ({ix},{iy})"
            );
        }
    }
}

#[test]
fn diamond_trace_takes_the_smallest_intermediate() {
    let r = run(DIAMOND);
    let s = |id| r.map.slot(id).unwrap();
    // Fails if the tie between 1->2->4 and 1->3->4 is not broken by smallest z.
    let got = trace(&r.edge, r.map.valid(), s(1), s(5)).unwrap();
    assert_eq!(got, vec![(s(1), s(2)), (s(2), s(4)), (s(4), s(5))]);
}

#[test]
fn a_base_fact_traces_to_itself() {
    let r = run(DIAMOND);
    let s = |id| r.map.slot(id).unwrap();
    // Fails if a round-1 pair recurses instead of returning the fact.
    assert_eq!(
        trace(&r.edge, r.map.valid(), s(4), s(5)).unwrap(),
        vec![(s(4), s(5))]
    );
}

#[test]
fn diamond_traces_are_valid_derivations() {
    check_all(DIAMOND, "diamond");
}

#[test]
fn cycle_traces_are_valid_derivations() {
    check_all(CYCLE, "cycle");
}

#[test]
fn a_pair_that_does_not_hold_has_no_trace() {
    let r = run(DIAMOND);
    let s = |id| r.map.slot(id).unwrap();
    // Fails if trace invents a chain against edge direction.
    assert_eq!(trace(&r.edge, r.map.valid(), s(5), s(1)), None);
    // Fails if an out-of-range slot is not refused.
    assert_eq!(trace(&r.edge, r.map.valid(), SLOTS, 0), None);
}

#[test]
fn deterministic_differential() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let mut checked = BTreeSet::new();
    for case in 0..200 {
        let n = 2 + next() % 40;
        let m = next() % (3 * n);
        let mut dedup: Vec<(u32, u32)> = Vec::new();
        for _ in 0..m {
            let e = (next() % n, next() % n);
            if !dedup.contains(&e) {
                dedup.push(e);
            }
        }
        if dedup.is_empty() {
            continue;
        }
        check_all(&dedup, &format!("case {case}"));
        checked.insert(case);
    }
    assert!(checked.len() > 150);
}
