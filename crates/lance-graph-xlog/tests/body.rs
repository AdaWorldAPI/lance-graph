//! D-XLOG-W7 items 1 and 3: conjunctive bodies over the cube, cyclic ones
//! included, against a nested-loop oracle that shares no code with the
//! cube. The fixtures are XLOG's own (`test_wcoj_recursive_dispatch.rs`,
//! `test_wcoj_clique_dispatch.rs` at `007bb19`).

use std::collections::BTreeSet;

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Atom, BodyRefusal, Cube, CubeRefusal, MAX_BODY_VARS};

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

/// Every binding of `0..k` over `dom` that satisfies every atom, by nested
/// loops over global ids.
fn oracle(rels: &[Vec<(u32, u32)>], atoms: &[Atom], k: usize, dom: &[u32]) -> Vec<Vec<u32>> {
    let sets: Vec<BTreeSet<(u32, u32)>> =
        rels.iter().map(|r| r.iter().copied().collect()).collect();
    let mut out = Vec::new();
    let mut b = vec![0u32; k];
    fn go(
        v: usize,
        k: usize,
        dom: &[u32],
        b: &mut Vec<u32>,
        sets: &[BTreeSet<(u32, u32)>],
        atoms: &[Atom],
        out: &mut Vec<Vec<u32>>,
    ) {
        if v == k {
            if atoms.iter().all(|a| {
                sets[a.rel as usize].contains(&(b[a.vars[0] as usize], b[a.vars[1] as usize]))
            }) {
                out.push(b.clone());
            }
            return;
        }
        for &x in dom {
            b[v] = x;
            go(v + 1, k, dom, b, sets, atoms, out);
        }
    }
    go(0, k, dom, &mut b, &sets, atoms, &mut out);
    out
}

/// The cube's answer, checked against the oracle: count, support per
/// variable, and the witness set. Returns the count.
fn check(rels: &[Vec<(u32, u32)>], atoms: &[Atom], k: usize, what: &str) -> u64 {
    let mut ids: Vec<u32> = rels.iter().flatten().flat_map(|&(a, b)| [a, b]).collect();
    ids.sort_unstable();
    ids.dedup();
    let t = table(&ids);
    let slices: Vec<&[(u32, u32)]> = rels.iter().map(Vec::as_slice).collect();
    let cube = Cube::new(&t, &t.seal(), TILE, &slices).unwrap();
    let want = oracle(rels, atoms, k, &ids);
    let got = cube.body_answer(&t, atoms, k).unwrap();
    assert_eq!(got.count, want.len() as u64, "{what}: count");

    let map = t.map(&t.seal(), TILE).unwrap();
    let globals = map.globals();
    for v in 0..k {
        let mut sup: Vec<u64> = (0..globals.len())
            .filter(|&p| got.support[v][p / 64] >> (p % 64) & 1 == 1)
            .map(|p| globals[p])
            .collect();
        sup.sort_unstable();
        let mut w: Vec<u64> = want.iter().map(|r| u64::from(r[v])).collect();
        w.sort_unstable();
        w.dedup();
        assert_eq!(sup, w, "{what}: support of variable {v}");
    }

    let rows = cube.materialize_witnesses(&t, atoms, k).unwrap();
    assert_eq!(rows.len(), want.len() * k, "{what}: witness width");
    let got_set: BTreeSet<Vec<u64>> = rows.chunks(k).map(<[u64]>::to_vec).collect();
    let want_set: BTreeSet<Vec<u64>> = want
        .iter()
        .map(|r| r.iter().map(|&x| u64::from(x)).collect())
        .collect();
    assert_eq!(got_set.len(), want.len(), "{what}: witnesses are distinct");
    assert_eq!(got_set, want_set, "{what}: witness set");
    got.count
}

/// XLOG's `triangle_inputs`: `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`.
#[test]
fn xlog_triangle_fixture() {
    let e1 = vec![
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (6, 7),
    ];
    let e2 = vec![(2, 3), (2, 4), (3, 4), (6, 7)];
    let e3 = vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)];
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    let n = check(&[e1, e2, e3], &tri, 3, "tri");
    // (1,2,3), (1,2,4), (1,3,4), (2,3,4), (5,6,7).
    assert_eq!(n, 5);
}

/// The four-cycle `cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z),
/// e4(Z, W)`: the last atom closes back on `W`.
#[test]
fn four_cycle_closes_on_the_first_variable() {
    let e: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (4, 1), (2, 5), (5, 6), (6, 9)];
    let cyc = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 2, 3), atom(3, 3, 0)];
    let rels = vec![e.clone(), e.clone(), e.clone(), e];
    // The four rotations of 1→2→3→4→1; the chain 2→5→6→9 does not close.
    assert_eq!(check(&rels, &cyc, 4, "cyc"), 4);
}

/// `clique5` over ten relations, one per vertex pair, as in XLOG's
/// `CLIQUE5_SRC`, on a graph holding exactly one 5-clique plus noise.
#[test]
fn clique5_over_ten_relations() {
    let mut e: Vec<(u32, u32)> = Vec::new();
    for a in 10..15u32 {
        for b in a + 1..15 {
            e.push((a, b));
        }
    }
    e.extend([(10, 20), (20, 21), (11, 21), (12, 20)]);
    let mut atoms = Vec::new();
    let mut rels = Vec::new();
    for a in 0..5u8 {
        for b in a + 1..5 {
            atoms.push(atom(rels.len() as u32, a, b));
            rels.push(e.clone());
        }
    }
    assert_eq!(check(&rels, &atoms, 5, "clique5"), 1);
}

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// Random relations under cyclic, acyclic, star and disconnected bodies.
#[test]
fn random_bodies_match_the_nested_loop_oracle() {
    let shapes: [(&str, Vec<Atom>, usize); 6] = [
        ("tri", vec![atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)], 3),
        (
            "tri-rev",
            vec![atom(0, 1, 0), atom(1, 2, 1), atom(2, 2, 0)],
            3,
        ),
        (
            "cyc",
            vec![atom(0, 0, 1), atom(1, 1, 2), atom(2, 2, 3), atom(0, 3, 0)],
            4,
        ),
        (
            "chain3",
            vec![atom(0, 0, 1), atom(1, 1, 2), atom(2, 2, 3)],
            4,
        ),
        ("star", vec![atom(0, 0, 1), atom(1, 0, 2), atom(2, 0, 3)], 4),
        ("two-parts", vec![atom(0, 0, 1), atom(1, 2, 3)], 4),
    ];
    let mut nonzero = 0;
    for s in 1..=60u64 {
        let mut next = rng(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let dom = 4 + (next() % 9) as u32;
        let rels: Vec<Vec<(u32, u32)>> = (0..3)
            .map(|_| {
                (0..(next() % 3 + 1) * u64::from(dom))
                    .map(|_| {
                        (
                            100 + (next() % u64::from(dom)) as u32,
                            100 + (next() % u64::from(dom)) as u32,
                        )
                    })
                    .collect()
            })
            .collect();
        for (name, atoms, k) in &shapes {
            if check(&rels, atoms, *k, &format!("{name} seed {s}")) > 0 {
                nonzero += 1;
            }
        }
    }
    // Anti-vacuity: most cases have witnesses, so equality is not 0 == 0.
    assert!(nonzero >= 200, "only {nonzero} of 360 cases had witnesses");
}

#[test]
fn malformed_bodies_are_refused() {
    let e = vec![(1u32, 2u32), (2, 3)];
    let t = table(&[1, 2, 3]);
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let ans = |atoms: &[Atom], k: usize| cube.body_answer(&t, atoms, k).unwrap_err();
    assert_eq!(ans(&[atom(0, 0, 1)], 0), BodyRefusal::Arity(0));
    assert_eq!(
        ans(&[atom(0, 0, 1)], MAX_BODY_VARS + 1),
        BodyRefusal::Arity(MAX_BODY_VARS + 1)
    );
    assert_eq!(
        ans(&[atom(0, 0, 2)], 2),
        BodyRefusal::VarOutOfRange { atom: 0, var: 2 }
    );
    assert_eq!(
        ans(&[atom(0, 1, 1)], 2),
        BodyRefusal::SameVariable { atom: 0 }
    );
    assert_eq!(ans(&[atom(0, 0, 1)], 3), BodyRefusal::Unused(2));
    assert_eq!(
        ans(&[atom(7, 0, 1)], 2),
        BodyRefusal::Cube(CubeRefusal::UnknownRel(7))
    );
}

/// The count runs no mask-risc program and builds no row; only
/// `materialize_witnesses` counts rows.
#[test]
fn the_count_builds_no_rows() {
    let e = vec![(1u32, 2u32), (2, 3), (1, 3)];
    let t = table(&[1, 2, 3]);
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e, &e, &e]).unwrap();
    let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
    assert_eq!(cube.body_answer(&t, &tri, 3).unwrap().count, 1);
    assert_eq!(cube.stats().rows_materialized, 0);
    assert_eq!(cube.stats().programs, 0);
    assert_eq!(
        cube.materialize_witnesses(&t, &tri, 3).unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(cube.stats().rows_materialized, 1);
}
