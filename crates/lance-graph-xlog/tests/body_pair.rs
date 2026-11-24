//! EXO-396 reworked: a body's demanded two-column projection as presence
//! over existing coordinates (`Cube::body_pair`), and `X != Z` as a
//! coordinate constraint inside the same walk. No tuple is built.
//!
//! Two independent oracles: a nested-loop enumeration over global ids that
//! shares no code with the cube, and the sets written out by hand for the
//! XLOG fixtures (`sg_result`, `fof`).

use std::collections::BTreeSet;

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Atom, BodyRefusal, Cube, PairAnswer};

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

fn atom(rel: u32, a: u8, b: u8) -> Atom {
    Atom { rel, vars: [a, b] }
}

fn ids_of(rels: &[Vec<(u32, u32)>]) -> Vec<u32> {
    let mut ids: Vec<u32> = rels.iter().flatten().flat_map(|&(a, b)| [a, b]).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// `{ (b[u], b[v]) }` over every binding of `0..k` that satisfies every atom
/// and every distinctness pair, by nested loops over global ids.
fn oracle(
    rels: &[Vec<(u32, u32)>],
    atoms: &[Atom],
    k: usize,
    distinct: &[[u8; 2]],
    [u, v]: [u8; 2],
) -> BTreeSet<(u32, u32)> {
    let dom = ids_of(rels);
    let sets: Vec<BTreeSet<(u32, u32)>> =
        rels.iter().map(|r| r.iter().copied().collect()).collect();
    let mut out = BTreeSet::new();
    let mut b = vec![0u32; k];
    #[allow(clippy::too_many_arguments)]
    fn go(
        i: usize,
        k: usize,
        dom: &[u32],
        b: &mut Vec<u32>,
        sets: &[BTreeSet<(u32, u32)>],
        atoms: &[Atom],
        distinct: &[[u8; 2]],
        uv: (usize, usize),
        out: &mut BTreeSet<(u32, u32)>,
    ) {
        if i == k {
            let ok = atoms.iter().all(|a| {
                sets[a.rel as usize].contains(&(b[a.vars[0] as usize], b[a.vars[1] as usize]))
            }) && distinct
                .iter()
                .all(|&[x, y]| b[x as usize] != b[y as usize]);
            if ok {
                out.insert((b[uv.0], b[uv.1]));
            }
            return;
        }
        for &x in dom {
            b[i] = x;
            go(i + 1, k, dom, b, sets, atoms, distinct, uv, out);
        }
    }
    go(
        0,
        k,
        &dom,
        &mut b,
        &sets,
        atoms,
        distinct,
        (usize::from(u), usize::from(v)),
        &mut out,
    );
    out
}

fn cube_of(rels: &[Vec<(u32, u32)>]) -> (MapTable, Cube) {
    let t = table(&ids_of(rels));
    let slices: Vec<&[(u32, u32)]> = rels.iter().map(Vec::as_slice).collect();
    let cube = Cube::new(&t, &t.seal(), TILE, &slices).unwrap();
    (t, cube)
}

/// The presence as global-id pairs: decoded at the edge, for the comparison.
fn decode(t: &MapTable, a: &PairAnswer) -> BTreeSet<(u32, u32)> {
    let map = t.map(&t.seal(), TILE).unwrap();
    let g = map.globals();
    let mut out = BTreeSet::new();
    for x in 0..g.len() {
        for y in 0..g.len() {
            if a.contains(x, y) {
                out.insert((g[x] as u32, g[y] as u32));
            }
        }
    }
    out
}

fn check(rels: &[Vec<(u32, u32)>], atoms: &[Atom], k: usize, distinct: &[[u8; 2]], what: &str) {
    let (t, cube) = cube_of(rels);
    for u in 0..k as u8 {
        for v in 0..k as u8 {
            if u == v {
                continue;
            }
            let want = oracle(rels, atoms, k, distinct, [u, v]);
            let got = cube.body_pair(&t, atoms, k, distinct, [u, v]).unwrap();
            assert_eq!(decode(&t, &got), want, "{what}: demand ({u},{v})");
            assert_eq!(got.pairs, want.len() as u64, "{what}: pair count ({u},{v})");
        }
    }
}

fn triangle_rels() -> Vec<Vec<(u32, u32)>> {
    vec![
        vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
        ],
        vec![(2, 3), (2, 4), (3, 4), (6, 7)],
        vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)],
    ]
}

#[test]
fn every_demand_of_the_triangle_equals_the_oracle() {
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    check(&triangle_rels(), &tri, 3, &[], "tri");
}

#[test]
fn every_demand_of_the_four_cycle_equals_the_oracle() {
    let e = vec![
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (1, 3),
        (2, 5),
        (5, 1),
        (3, 5),
    ];
    let cyc = [atom(0, 0, 1), atom(0, 1, 2), atom(0, 2, 3), atom(0, 3, 0)];
    check(std::slice::from_ref(&e), &cyc, 4, &[], "cyc");
    check(&[e], &cyc, 4, &[[0, 2], [1, 3]], "cyc distinct");
}

#[test]
fn a_chain_with_distinctness_equals_the_oracle() {
    let r = vec![(1, 2), (2, 1), (2, 3), (3, 4), (4, 2), (3, 3), (1, 1)];
    let chain = [atom(0, 0, 1), atom(0, 1, 2), atom(0, 2, 3)];
    check(std::slice::from_ref(&r), &chain, 4, &[], "chain");
    check(std::slice::from_ref(&r), &chain, 4, &[[0, 2]], "chain 0!=2");
    check(&[r], &chain, 4, &[[0, 3], [1, 2]], "chain two pairs");
}

/// Two triangles whose supports multiply to more pairs than exist: the
/// projection keeps the pairing a support product loses.
#[test]
fn the_projection_keeps_the_pairing() {
    let rels = vec![
        vec![(0, 1), (3, 4)],
        vec![(1, 2), (4, 5)],
        vec![(0, 2), (3, 5)],
    ];
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    let (t, cube) = cube_of(&rels);
    let got = cube.body_pair(&t, &tri, 3, &[], [0, 2]).unwrap();
    // p(X, Z) :- tri(X, Y, Z): (0,2) and (3,5); the product of the X and Z
    // supports {0,3} × {2,5} would add (0,5) and (3,2).
    assert_eq!(decode(&t, &got), BTreeSet::from([(0, 2), (3, 5)]));
    let sup = cube.body_answer(&t, &tri, 3).unwrap();
    let xs = sup.support[0].iter().map(|w| w.count_ones()).sum::<u32>();
    let zs = sup.support[2].iter().map(|w| w.count_ones()).sum::<u32>();
    assert_eq!(xs * zs, 4, "the support product is the wrong answer here");
}

/// `fof(X, Z) :- friend(X, Y), friend(Y, Z), X != Z.` (EXO-396 oracle).
#[test]
fn fof_clears_the_diagonal_inside_the_walk() {
    let friend = vec![(1, 2), (2, 1), (2, 3), (3, 4)];
    let fof = [atom(0, 0, 1), atom(0, 1, 2)];
    let (t, cube) = cube_of(&[friend]);
    let got = cube.body_pair(&t, &fof, 3, &[[0, 2]], [0, 2]).unwrap();
    // Joined: (1,1) (1,3) (2,2) (2,4); X != Z drops (1,1) and (2,2).
    assert_eq!(decode(&t, &got), BTreeSet::from([(1, 3), (2, 4)]));
    // Negative twin: without the constraint the diagonal is there.
    let all = cube.body_pair(&t, &fof, 3, &[], [0, 2]).unwrap();
    assert_eq!(
        decode(&t, &all),
        BTreeSet::from([(1, 1), (1, 3), (2, 2), (2, 4)])
    );
}

/// `sg_result` (EXO-396 oracle): rule 1 demands (X, Y) of
/// `parent(X, P), parent(Y, P)`; rule 2 unfolds the recipe
/// `sg_witness(W, X, Y, Z) :- parent(W, X), sg(X, Y), parent_rev(Y, Z),
/// all_child_pairs(Z, W)` and demands (W, Z). `sg_witness` is never stored.
#[test]
fn sg_result_is_two_demanded_projections_ored() {
    let parent = vec![(1, 3), (6, 3), (7, 4), (3, 5), (4, 5)];
    let parent_rev = vec![(3, 1), (3, 6), (4, 7), (5, 3), (5, 4)];
    let acp = vec![(7, 1), (1, 7), (1, 6), (9, 9)];
    // sg's fixpoint, by hand: siblings, then (1,7) (7,1) (6,7) (7,6).
    let sg = vec![
        (1, 1),
        (1, 6),
        (6, 1),
        (6, 6),
        (7, 7),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (1, 7),
        (7, 1),
        (6, 7),
        (7, 6),
    ];
    let rels = vec![parent, sg, parent_rev, acp];
    let (t, cube) = cube_of(&rels);
    let rule1 = [atom(0, 0, 2), atom(0, 1, 2)];
    let a = cube.body_pair(&t, &rule1, 3, &[], [0, 1]).unwrap();
    // W=0, X=1, Y=2, Z=3.
    let rule2 = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 2, 3), atom(3, 3, 0)];
    let b = cube.body_pair(&t, &rule2, 4, &[], [0, 3]).unwrap();
    assert_eq!(decode(&t, &b), BTreeSet::from([(1, 7), (6, 1), (7, 1)]));
    let ored = PairAnswer {
        words: a.words,
        rows: a.rows.iter().zip(&b.rows).map(|(x, y)| x | y).collect(),
        pairs: 0,
    };
    let want = BTreeSet::from([
        (1, 1),
        (1, 6),
        (6, 1),
        (6, 6),
        (7, 7),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (1, 7),
        (7, 1),
    ]);
    assert_eq!(decode(&t, &ored), want);
}

#[test]
fn a_projection_builds_no_rows() {
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    let (t, cube) = cube_of(&triangle_rels());
    let before = cube.stats().rows_materialized;
    cube.body_pair(&t, &tri, 3, &[[0, 2]], [0, 2]).unwrap();
    assert_eq!(cube.stats().rows_materialized, before);
}

#[test]
fn bad_demands_and_pairs_are_refused() {
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    let (t, cube) = cube_of(&triangle_rels());
    assert_eq!(
        cube.body_pair(&t, &tri, 3, &[], [1, 1]),
        Err(BodyRefusal::SameDemand)
    );
    assert_eq!(
        cube.body_pair(&t, &tri, 3, &[[2, 2]], [0, 1]),
        Err(BodyRefusal::SameDistinct { pair: 0 })
    );
    assert!(matches!(
        cube.body_pair(&t, &tri, 3, &[], [0, 7]),
        Err(BodyRefusal::VarOutOfRange { var: 7, .. })
    ));
}

#[test]
fn an_empty_body_has_an_empty_projection() {
    let rels = vec![vec![(1, 2)], vec![(3, 4)]];
    let chain = [atom(0, 0, 1), atom(1, 1, 2)];
    let (t, cube) = cube_of(&rels);
    let got = cube.body_pair(&t, &chain, 3, &[], [0, 2]).unwrap();
    assert_eq!(got.pairs, 0);
    assert!(decode(&t, &got).is_empty());
}

/// Two independent components, `r(X, Y), s(Z, W)`, demanded across them:
/// the projection is the full product of the two supports, a Cartesian law
/// held as presence. The top state splits, so the walk pinned at `X` starts
/// from a state the first plan never reached.
#[test]
fn a_demand_across_independent_components_is_their_product() {
    let rels = vec![vec![(1, 2), (3, 4)], vec![(5, 6), (7, 8), (7, 9)]];
    let body = [atom(0, 0, 1), atom(1, 2, 3)];
    check(&rels, &body, 4, &[], "split");
    let (t, cube) = cube_of(&rels);
    let got = cube.body_pair(&t, &body, 4, &[], [0, 2]).unwrap();
    assert_eq!(
        decode(&t, &got),
        BTreeSet::from([(1, 5), (1, 7), (3, 5), (3, 7)])
    );
}
