//! Cost of `Cube::body_answer` for the triangle and the four-cycle over one
//! tile: `cargo run --release -p lance-graph-xlog --example body_cost`.
//! Every count is checked against an edge walk that shares no code with the
//! cube.

use std::collections::BTreeSet;
use std::time::Instant;

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Atom, Cube};

const TILE: u32 = 5;

/// A body, its variable count, its distinctness pairs and the demanded pair.
type Demand = (&'static str, Vec<Atom>, usize, Vec<[u8; 2]>, [u8; 2]);

/// Witnesses of `atoms` by walking relation 0's edges and extending through
/// adjacency lists; every atom is checked on the full binding.
fn oracle(rels: &[Vec<(u32, u32)>], atoms: &[Atom], k: usize) -> u64 {
    let sets: Vec<BTreeSet<(u32, u32)>> =
        rels.iter().map(|r| r.iter().copied().collect()).collect();
    let mut out = vec![Vec::<u32>::new(); 256];
    for &(x, y) in &sets[atoms[1].rel as usize] {
        out[x as usize].push(y);
    }
    let mut n = 0;
    for &(x0, x1) in &sets[atoms[0].rel as usize] {
        for &x2 in &out[x1 as usize] {
            let ok = |b: &[u32]| {
                atoms.iter().all(|a| {
                    sets[a.rel as usize].contains(&(b[a.vars[0] as usize], b[a.vars[1] as usize]))
                })
            };
            if k == 3 {
                n += u64::from(ok(&[x0, x1, x2]));
            } else {
                n += (0..256).filter(|&x3| ok(&[x0, x1, x2, x3])).count() as u64;
            }
        }
    }
    n
}

fn main() {
    let mut s = 0x7EA1_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let all: Vec<u64> = (0..256).collect();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    let a = |rel, x, y| Atom { rel, vars: [x, y] };
    println!("edges/rel  shape  count  median_us");
    for m in [512usize, 2048, 8192] {
        let rels: Vec<Vec<(u32, u32)>> = (0..4)
            .map(|_| {
                (0..m)
                    .map(|_| ((next() % 256) as u32, (next() % 256) as u32))
                    .collect()
            })
            .collect();
        let slices: Vec<&[(u32, u32)]> = rels.iter().map(Vec::as_slice).collect();
        let cube = Cube::new(&t, &t.seal(), TILE, &slices).unwrap();
        let shapes: [(&str, Vec<Atom>, usize); 2] = [
            ("tri", vec![a(0, 0, 1), a(1, 1, 2), a(2, 0, 2)], 3),
            (
                "cyc",
                vec![a(0, 0, 1), a(1, 1, 2), a(2, 2, 3), a(3, 3, 0)],
                4,
            ),
        ];
        for (name, atoms, k) in shapes {
            let mut ts = Vec::new();
            let mut count = 0;
            for _ in 0..31 {
                let t0 = Instant::now();
                count = std::hint::black_box(cube.body_answer(&t, &atoms, k).unwrap()).count;
                ts.push(t0.elapsed().as_nanos() as f64 / 1e3);
            }
            assert_eq!(count, oracle(&rels, &atoms, k), "{name} at {m} edges");
            ts.sort_by(f64::total_cmp);
            println!("{m:<10} {name:<6} {count:<6} {:.1}", ts[15]);
        }
        // Cube::body_pair: the demanded two-column projection as presence.
        // Checked against the witnesses projected at the edge.
        let pairs: [Demand; 4] = [
            (
                "tri→XZ",
                vec![a(0, 0, 1), a(1, 1, 2), a(2, 0, 2)],
                3,
                vec![],
                [0, 2],
            ),
            (
                "cyc→WY",
                vec![a(0, 0, 1), a(1, 1, 2), a(2, 2, 3), a(3, 3, 0)],
                4,
                vec![],
                [0, 2],
            ),
            (
                "sg2→WZ",
                vec![a(0, 0, 1), a(1, 1, 2), a(2, 2, 3), a(3, 3, 0)],
                4,
                vec![],
                [0, 3],
            ),
            (
                "fof→XZ",
                vec![a(0, 0, 1), a(0, 1, 2)],
                3,
                vec![[0, 2]],
                [0, 2],
            ),
        ];
        for (name, atoms, k, distinct, demand) in pairs {
            let mut ts = Vec::new();
            let mut got = 0;
            for _ in 0..31 {
                let t0 = Instant::now();
                got =
                    std::hint::black_box(cube.body_pair(&t, &atoms, k, &distinct, demand).unwrap())
                        .pairs();
                ts.push(t0.elapsed().as_nanos() as f64 / 1e3);
            }
            let w = cube.materialize_witnesses(&t, &atoms, k).unwrap();
            let want: BTreeSet<(u64, u64)> = w
                .chunks(k)
                .filter(|r| {
                    distinct
                        .iter()
                        .all(|&[x, y]| r[x as usize] != r[y as usize])
                })
                .map(|r| (r[demand[0] as usize], r[demand[1] as usize]))
                .collect();
            assert_eq!(got, want.len() as u64, "{name} at {m} edges");
            ts.sort_by(f64::total_cmp);
            println!("{m:<10} {name:<6} {got:<6} {:.1}", ts[15]);
        }
    }
}
