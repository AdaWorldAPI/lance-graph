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

/// Repeated atoms over the same variable pair are one constraint stated
/// several times; more of them than variables must not overflow the per-leaf
/// row list.
#[test]
fn repeated_atoms_between_one_pair() {
    let e = vec![(1u32, 2u32), (2, 3), (1, 3), (3, 1)];
    let rels = vec![e.clone(), e];
    let atoms: Vec<Atom> = (0..9).map(|i| atom(i % 2, 0, 1)).collect();
    assert_eq!(check(&rels, &atoms, 2, "nine copies"), 4);
}

// ---------------------------------------------------------------------------
// D-XLOG-W7-2 step 3: the cube owns each relation's pair plane and lends it
// ---------------------------------------------------------------------------

mod lent_planes {
    use std::collections::BTreeSet;

    use lance_graph_dt64::dr::{LocalMap, MapTable, PairDir, PairRefusal, Width};
    use lance_graph_dt64::{relation_rows, RelationRefusal};
    use lance_graph_xlog::cube::Cube;

    use super::{atom, oracle, TILE};

    fn table(ids: &[u64]) -> MapTable {
        let mut t = MapTable::new(1, 1);
        t.advance(
            1,
            1,
            vec![(TILE, Some(LocalMap::prepare(Width::W256, ids).unwrap()))],
        );
        t
    }

    /// The pairs a lent plane holds, as global ids, read in `dir`.
    fn pairs(cube: &Cube, t: &MapTable, rel: u32, dir: PairDir) -> BTreeSet<(u64, u64)> {
        let f = t.seal();
        let g = t.map(&f, TILE).unwrap().globals().to_vec();
        let rows = relation_rows(cube, rel, cube.frame(), dir).unwrap();
        let mut out = BTreeSet::new();
        for (at, &a) in g.iter().enumerate() {
            let row = rows.row(at).unwrap();
            for (other, &b) in g.iter().enumerate() {
                if row[other / 64] >> (other % 64) & 1 == 1 {
                    out.insert(match dir {
                        PairDir::Forward => (a, b),
                        PairDir::Backward => (b, a),
                    });
                }
            }
        }
        out
    }

    fn set(edges: &[(u32, u32)]) -> BTreeSet<(u64, u64)> {
        edges
            .iter()
            .map(|&(a, b)| (u64::from(a), u64::from(b)))
            .collect()
    }

    /// The lent plane equals the relation's effective edges, in both
    /// directions, after every kind of change: logged add, logged retire,
    /// refold. A stale plane would still show the state before the change.
    #[test]
    fn the_lent_plane_follows_adds_retires_and_refolds() {
        let t = table(&[1, 2, 3, 4, 5]);
        let e = [(1u32, 2u32), (2, 3), (3, 1)];
        let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
        let both = |c: &Cube, want: &[(u32, u32)], what: &str| {
            assert_eq!(pairs(c, &t, 0, PairDir::Forward), set(want), "{what}: fwd");
            assert_eq!(pairs(c, &t, 0, PairDir::Backward), set(want), "{what}: bwd");
        };
        both(&cube, &e, "built");
        cube.add_edge(&t, 0, 3, 4).unwrap();
        both(&cube, &[(1, 2), (2, 3), (3, 1), (3, 4)], "after add");
        cube.retire_edge(&t, 0, 2, 3).unwrap();
        both(&cube, &[(1, 2), (3, 1), (3, 4)], "after retire");
        cube.refold(0).unwrap();
        both(&cube, &[(1, 2), (3, 1), (3, 4)], "after refold");
        cube.add_edge(&t, 0, 4, 5).unwrap();
        both(&cube, &[(1, 2), (3, 1), (3, 4), (4, 5)], "add after refold");
    }

    /// A body answered before and after a change agrees with the oracle both
    /// times: the triangle exists, loses an edge, and comes back.
    #[test]
    fn a_body_reads_the_current_plane_after_a_change() {
        let t = table(&[1, 2, 3]);
        let e = [(1u32, 2u32), (2, 3), (1, 3)];
        let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e, &e, &e]).unwrap();
        let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
        let count = |c: &Cube| c.body_answer(&t, &tri, 3).unwrap().count;
        let want = |r1: &[(u32, u32)]| {
            oracle(&[e.to_vec(), r1.to_vec(), e.to_vec()], &tri, 3, &[1, 2, 3]).len() as u64
        };
        assert_eq!((count(&cube), want(&e)), (1, 1));
        cube.retire_edge(&t, 1, 2, 3).unwrap();
        assert_eq!((count(&cube), want(&[(1, 2), (1, 3)])), (0, 0));
        cube.add_edge(&t, 1, 2, 3).unwrap();
        assert_eq!(count(&cube), 1);
    }

    /// Two reads with no change between them lend the same words: the plane
    /// is built once per state of the relation, never per call. A body call
    /// in between does not rebuild it either.
    #[test]
    fn an_unchanged_relation_is_not_rebuilt() {
        let t = table(&[1, 2, 3]);
        let e = [(1u32, 2u32), (2, 3), (1, 3)];
        let cube = Cube::new(&t, &t.seal(), TILE, &[&e, &e, &e]).unwrap();
        let f = cube.frame().clone();
        let first = relation_rows(&cube, 0, &f, PairDir::Forward)
            .unwrap()
            .as_words()
            .as_ptr();
        let tri = [atom(0, 0, 1), atom(1, 1, 2), atom(2, 0, 2)];
        assert_eq!(cube.body_answer(&t, &tri, 3).unwrap().count, 1);
        let again = relation_rows(&cube, 0, &f, PairDir::Forward)
            .unwrap()
            .as_words()
            .as_ptr();
        assert_eq!(first, again);
    }

    /// The socket's refusals through the cube: an unknown relation, and a
    /// plane asked for under the frame the cube has left after a rebase. The
    /// rebased cube lends a plane under its new frame, holding the same pairs.
    #[test]
    fn unknown_relations_and_left_frames_are_refused() {
        let mut t = table(&[1, 2, 3]);
        let e = [(1u32, 2u32), (2, 3)];
        let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
        let old = cube.frame().clone();
        assert_eq!(
            relation_rows(&cube, 9, &old, PairDir::Forward).unwrap_err(),
            RelationRefusal::NoRelation(9)
        );
        assert!(relation_rows(&cube, 0, &old, PairDir::Forward).is_ok());
        t.advance(
            2,
            1,
            vec![(
                TILE,
                Some(LocalMap::prepare(Width::W256, &[1, 2, 3, 4]).unwrap()),
            )],
        );
        let new = t.seal();
        cube.rebase(&t, &new).unwrap();
        assert_eq!(
            relation_rows(&cube, 0, &old, PairDir::Forward).unwrap_err(),
            RelationRefusal::Pair(PairRefusal::ForeignFrame)
        );
        assert_eq!(pairs(&cube, &t, 0, PairDir::Forward), set(&e));
    }

    /// A witnessed reach taken straight over the cube through the relation
    /// socket: hops follow the cube's effective edges after a logged add and
    /// retire, the path is made of held pairs, and a witness or a frame from
    /// before a change is refused.
    #[test]
    fn a_witnessed_reach_over_the_cube_follows_its_edges() {
        use lance_graph_dt64::{reach_path, relation_reach, ReachBuf, ReachRefusal, UNREACHED};
        let mut t = table(&[1, 2, 3, 4, 5]);
        let e = [(1u32, 2u32), (2, 3), (3, 4)];
        let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
        let f = cube.frame().clone();
        let (mut reached, mut hop, mut scratch) = ([0u64; 4], [0u16; 256], [0u64; 8]);
        let seed = [1u64, 0, 0, 0];
        let mut go = |cube: &Cube, f, hop: &mut [u16; 256]| {
            relation_reach(
                cube,
                0,
                f,
                PairDir::Forward,
                &seed,
                None,
                &mut ReachBuf {
                    reached: &mut reached,
                    hop,
                    scratch: &mut scratch,
                },
            )
        };
        go(&cube, &f, &mut hop).unwrap();
        assert_eq!(hop[..5], [0, 1, 2, 3, UNREACHED]);
        let before = hop;

        // 1 -> 5 -> 4 is added and 2 -> 3 retired: 4 is now two hops away
        // and 3 is not reached at all.
        cube.add_edge(&t, 0, 1, 5).unwrap();
        cube.add_edge(&t, 0, 5, 4).unwrap();
        cube.retire_edge(&t, 0, 2, 3).unwrap();
        go(&cube, &f, &mut hop).unwrap();
        assert_eq!(hop[..5], [0, 1, UNREACHED, 2, 1]);
        let mut path = [0u32; 8];
        let n = reach_path(&cube, 0, &f, PairDir::Forward, &hop, 3, &mut path).unwrap();
        assert_eq!(path[..n], [0, 4, 3]);
        // The witness from before the change names a hop that is gone.
        assert_eq!(
            reach_path(&cube, 0, &f, PairDir::Forward, &before, 3, &mut path),
            Err(ReachRefusal::WitnessMismatch(2))
        );

        // After a rebase the old frame is refused and the new one answers.
        t.advance(
            2,
            1,
            vec![(
                TILE,
                Some(LocalMap::prepare(Width::W256, &[1, 2, 3, 4, 5, 6]).unwrap()),
            )],
        );
        let new = t.seal();
        cube.rebase(&t, &new).unwrap();
        assert_eq!(
            go(&cube, &f, &mut hop),
            Err(ReachRefusal::Relation(RelationRefusal::Pair(
                PairRefusal::ForeignFrame
            )))
        );
        go(&cube, &new, &mut hop).unwrap();
        assert_eq!(hop[..5], [0, 1, UNREACHED, 2, 1]);
    }
}
