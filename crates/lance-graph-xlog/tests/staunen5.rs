//! D-STAUNEN-5 P0 and P1 (`.claude/plans/d-staunen-5-resident-recursive-plasticity-v1.md`).
//!
//! P0: a seeded episode generator and a scalar reference closure. The hidden
//! graph `G*` lives in [`fixtures::World`] only; [`fixtures::Episode`], which
//! later phases hand to the policies, has no field that can reach it.
//!
//! P1: [`Cube::closure`] against the scalar oracle, bit for bit, on cyclic
//! and acyclic graphs up to local position 63, with its work counted.

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal};

const TILE: u32 = 3;

/// The scalar reference: `edge⁺` as one row mask per local position, by a
/// breadth-first search from every source. Shares no code with the cube.
mod oracle {
    pub fn closure(n: usize, edges: &[(u8, u8)]) -> Vec<u64> {
        let mut succ = vec![Vec::new(); n];
        for &(a, b) in edges {
            succ[a as usize].push(b as usize);
        }
        (0..n)
            .map(|s| {
                let mut seen = 0u64;
                let mut stack: Vec<usize> = succ[s].clone();
                while let Some(v) = stack.pop() {
                    if seen >> v & 1 == 0 {
                        seen |= 1 << v;
                        stack.extend(&succ[v]);
                    }
                }
                seen
            })
            .collect()
    }

    pub fn pairs(rows: &[u64]) -> u64 {
        rows.iter().map(|r| u64::from(r.count_ones())).sum()
    }
}

/// A frame over the ids `0..n`, so global id = local position.
fn table(n: u32) -> MapTable {
    let ids: Vec<u64> = (0..u64::from(n)).collect();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W64, &ids).unwrap()))],
    );
    t
}

fn resident(n: u32, edges: &[(u8, u8)]) -> (Vec<u64>, lance_graph_xlog::cube::ClosureStats, Cube) {
    let t = table(n);
    let tuples: Vec<(u32, u32)> = edges
        .iter()
        .map(|&(a, b)| (u32::from(a), u32::from(b)))
        .collect();
    let rels: [&[(u32, u32)]; 1] = [&tuples];
    let cube = Cube::new(&t, &t.seal(), TILE, &rels).unwrap();
    let (closure, stats) = cube.closure(&t, 0).unwrap();
    (closure.rows().to_vec(), stats, cube)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self, m: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % u64::from(m)) as u32
    }
}

// ---------------------------------------------------------------- P1

/// The fixed fixture: a 3-cycle with an exit. `0,1,2` reach `0..=3`
/// (themselves included), `3` reaches nothing.
#[test]
fn the_fixed_cycle_has_exactly_twelve_pairs() {
    let edges = [(0, 1), (1, 2), (2, 0), (2, 3)];
    let (rows, stats, _) = resident(4, &edges);
    assert_eq!(rows, oracle::closure(4, &edges));
    assert_eq!(oracle::pairs(&rows), 12);
    let has = |x: usize, y: usize| rows[x] >> y & 1 == 1;
    assert!(has(0, 0) && has(1, 1) && has(2, 2));
    assert!(!has(3, 3));
    assert_eq!(stats.rows_composed, 12);
}

/// Random cyclic and acyclic graphs, every size up to 64 ids, so local
/// position 63 is exercised. Bit for bit against the oracle; the work
/// invariant holds on every one.
#[test]
fn resident_closure_matches_the_oracle_bit_for_bit() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut cyclic, mut acyclic, mut top) = (0, 0, 0);
    for case in 0..400u32 {
        let n = if case % 4 == 0 { 64 } else { rng.next(64) + 1 };
        let dag = case % 2 == 0;
        let m = rng.next(3 * n) + 1;
        let edges: Vec<(u8, u8)> = (0..m)
            .filter_map(|_| {
                let (a, b) = (rng.next(n) as u8, rng.next(n) as u8);
                match dag {
                    true if a < b => Some((a, b)),
                    true => None,
                    false => Some((a, b)),
                }
            })
            .collect();
        if edges.is_empty() {
            continue;
        }
        let want = oracle::closure(n as usize, &edges);
        let (rows, stats, cube) = resident(n, &edges);
        assert_eq!(rows, want, "case {case}, n {n}, edges {edges:?}");
        assert_eq!(stats.rows_composed, oracle::pairs(&want), "case {case}");
        assert_eq!(cube.stats().relations_added, 0, "case {case}");
        let has_cycle = (0..n as usize).any(|x| want[x] >> x & 1 == 1);
        cyclic += usize::from(has_cycle);
        acyclic += usize::from(!has_cycle);
        top += usize::from(want.iter().any(|r| r >> 63 & 1 == 1));
    }
    // Anti-vacuity: both shapes occur, and position 63 is reached.
    assert!(
        cyclic > 50 && acyclic > 50,
        "{cyclic} cyclic, {acyclic} acyclic"
    );
    assert!(top > 20, "position 63 reached in only {top} cases");
}

/// A long chain needs one round per step; a cycle of 64 needs 64 rounds and
/// must still terminate.
#[test]
fn the_longest_paths_terminate() {
    let chain: Vec<(u8, u8)> = (0..63).map(|i| (i, i + 1)).collect();
    let (rows, stats, _) = resident(64, &chain);
    assert_eq!(rows, oracle::closure(64, &chain));
    assert_eq!(oracle::pairs(&rows), 63 * 64 / 2);

    let mut ring = chain.clone();
    ring.push((63, 0));
    let (rows, stats_ring, _) = resident(64, &ring);
    assert!(rows.iter().all(|&r| r == u64::MAX));
    assert!(stats_ring.rounds <= 64 && stats.rounds <= 64);
}

#[test]
fn an_empty_relation_has_an_empty_closure() {
    let (rows, stats, _) = resident(5, &[]);
    assert_eq!(rows, vec![0; 5]);
    assert_eq!(stats.rows_composed, 0);
}

/// More than one word per row is not admitted: the closure refuses instead
/// of reading a 65th position.
#[test]
fn a_map_wider_than_64_is_refused() {
    let ids: Vec<u64> = (0..65).collect();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids).unwrap()))],
    );
    let tuples = [(0u32, 64u32)];
    let rels: [&[(u32, u32)]; 1] = [&tuples];
    let cube = Cube::new(&t, &t.seal(), TILE, &rels).unwrap();
    assert!(matches!(
        cube.closure(&t, 0),
        Err(CubeRefusal::ClosureWidth(65))
    ));
    assert!(matches!(
        cube.closure(&t, 1),
        Err(CubeRefusal::UnknownRel(1))
    ));
}

// ---------------------------------------------------------------- P0

/// The episode generator. `World` holds the hidden graph; `Episode` is what
/// a policy may read.
mod fixtures {
    use super::{oracle, Rng};

    /// Read only by the generator and by external scoring.
    pub struct World {
        pub n: usize,
        pub g_star: Vec<(u8, u8)>,
    }

    pub struct Stimulus {
        pub pair: (u8, u8),
        /// Not in the closure of `base`.
        pub novel: bool,
        /// Indices into `candidates` whose activation alone closes `pair`.
        pub closing: Vec<usize>,
    }

    /// No field reaches `G*`.
    pub struct Episode {
        pub n: usize,
        pub base: Vec<(u8, u8)>,
        pub candidates: Vec<(u8, u8)>,
        pub stimuli: Vec<Stimulus>,
    }

    fn reaches(n: usize, edges: &[(u8, u8)], (a, b): (u8, u8)) -> bool {
        oracle::closure(n, edges)[a as usize] >> b & 1 == 1
    }

    /// `G*` is a random graph over `n` ids. Two planted edges in series,
    /// `(p,q),(q,r)`, plus `missing` random ones are withheld from `base`;
    /// the candidates are every withheld edge plus `distractors` edges absent
    /// from `G*`. Stimuli are pairs reachable in `G*`, some novel.
    pub fn generate(seed: u64, n: usize, distractors: usize) -> (World, Episode) {
        let mut rng = Rng(seed | 1);
        let nn = n as u32;
        let (p, q, r) = (0u8, 1u8, 2u8);
        let mut g_star = vec![(p, q), (q, r)];
        while g_star.len() < 2 * n {
            let e = (rng.next(nn) as u8, rng.next(nn) as u8);
            // Keep p→q and q→r as the only routes into q and r, so the
            // planted pair (p,r) needs both withheld edges.
            if e.1 == q || e.1 == r || g_star.contains(&e) {
                continue;
            }
            g_star.push(e);
        }
        let mut withheld = vec![(p, q), (q, r)];
        while withheld.len() < 5 {
            let e = g_star[2 + rng.next(nn * 2 - 2) as usize];
            if !withheld.contains(&e) {
                withheld.push(e);
            }
        }
        let base: Vec<_> = g_star
            .iter()
            .filter(|e| !withheld.contains(e))
            .copied()
            .collect();
        let mut candidates = withheld.clone();
        while candidates.len() < withheld.len() + distractors {
            let e = (rng.next(nn) as u8, rng.next(nn) as u8);
            // A distractor into q or r could close the planted pair alone.
            if e.1 != q && e.1 != r && !g_star.contains(&e) && !candidates.contains(&e) {
                candidates.push(e);
            }
        }
        let truth = oracle::closure(n, &g_star);
        let mut stimuli = Vec::new();
        for a in 0..n as u8 {
            for b in 0..n as u8 {
                if truth[a as usize] >> b & 1 == 0 {
                    continue;
                }
                let novel = !reaches(n, &base, (a, b));
                let closing = (0..candidates.len())
                    .filter(|&i| {
                        novel && {
                            let mut e = base.clone();
                            e.push(candidates[i]);
                            reaches(n, &e, (a, b))
                        }
                    })
                    .collect();
                stimuli.push(Stimulus {
                    pair: (a, b),
                    novel,
                    closing,
                });
            }
        }
        (
            World { n, g_star },
            Episode {
                n,
                base,
                candidates,
                stimuli,
            },
        )
    }
}

#[test]
fn episodes_are_deterministic_per_seed() {
    let (w1, e1) = fixtures::generate(7, 16, 4);
    let (w2, e2) = fixtures::generate(7, 16, 4);
    assert_eq!(w1.g_star, w2.g_star);
    assert_eq!(e1.base, e2.base);
    assert_eq!(e1.candidates, e2.candidates);
    let (w3, _) = fixtures::generate(8, 16, 4);
    assert_ne!(w1.g_star, w3.g_star);
}

/// Every stream has a one-edge-closable novel stimulus and a novel stimulus
/// no single candidate closes; every `closing` list is exactly right.
#[test]
fn novel_stimuli_are_classified_by_the_oracle() {
    for seed in 1..40u64 {
        let (w, e) = fixtures::generate(seed, 16, 4);
        let novel: Vec<_> = e.stimuli.iter().filter(|s| s.novel).collect();
        assert!(novel.iter().any(|s| !s.closing.is_empty()), "seed {seed}");
        assert!(
            novel.iter().any(|s| s.closing.is_empty()),
            "seed {seed}: no novel stimulus needs two edges"
        );
        // The planted pair needs both withheld edges.
        let planted = e.stimuli.iter().find(|s| s.pair == (0, 2)).unwrap();
        assert!(planted.novel && planted.closing.is_empty(), "seed {seed}");
        for s in &e.stimuli {
            assert!(s.novel || s.closing.is_empty());
            for (i, &c) in e.candidates.iter().enumerate() {
                let mut edges = e.base.clone();
                edges.push(c);
                let closes = oracle::closure(e.n, &edges)[s.pair.0 as usize] >> s.pair.1 & 1 == 1;
                assert_eq!(s.novel && closes, s.closing.contains(&i), "seed {seed}");
            }
        }
        // Withheld edges are in G*; distractors are not.
        assert!(e.candidates[..5].iter().all(|c| w.g_star.contains(c)));
        assert!(e.candidates[5..].iter().all(|c| !w.g_star.contains(c)));
        assert_eq!(w.n, e.n);
    }
}

/// The `relations_added == 0` gate above can fire: adding a relation counts.
#[test]
fn adding_a_relation_is_counted() {
    let t = table(4);
    let tuples = [(0u32, 1u32)];
    let rels: [&[(u32, u32)]; 1] = [&tuples];
    let mut cube = Cube::new(&t, &t.seal(), TILE, &rels).unwrap();
    assert_eq!(cube.stats().relations_added, 0);
    cube.add_relation(&t, &[(1, 2)]).unwrap();
    assert_eq!(cube.stats().relations_added, 1);
    let (c, _) = cube.closure(&t, 1).unwrap();
    assert!(c.contains(1, 2) && !c.contains(0, 1));
    assert_eq!(cube.stats().relations_added, 1);
}
