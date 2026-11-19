//! Cost of `relation_reach` over one lent plane, in nanoseconds per call.
//! Every answer is asserted against a breadth-first walk before timing.
//!
//! `cargo run --release -p lance-graph-dt64 --example reach_cost`

use std::hint::black_box;
use std::time::Instant;

use lance_graph_dt64::dr::{Axis, MapTable, PairDir, PairPlane, Width};
use lance_graph_dt64::{relation_reach, ReachBuf, RelNo, RelationSource, UNREACHED};

struct Owner(PairPlane);

impl RelationSource for Owner {
    fn borrow_relation(&self, _: RelNo) -> Option<&PairPlane> {
        Some(&self.0)
    }
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn bfs(edges: &[(usize, usize)], p: usize) -> Vec<u16> {
    let mut dist = vec![UNREACHED; p];
    let mut queue = std::collections::VecDeque::from([0usize]);
    dist[0] = 0;
    while let Some(x) = queue.pop_front() {
        for &(a, b) in edges {
            if a == x && dist[b] == UNREACHED {
                dist[b] = dist[x] + 1;
                queue.push_back(b);
            }
        }
    }
    dist
}

fn main() {
    println!("positions\tedges\treached\trounds\tns_per_call");
    for tiles in [1u32, 4, 16] {
        let p = 256 * tiles as usize;
        let w = p / 64;
        for per_node in [1usize, 4] {
            let mut s = 11u64;
            let mut edges: Vec<(usize, usize)> =
                (0..p - 1).step_by(3).map(|i| (i, i + 1)).collect();
            for _ in 0..p * per_node {
                edges.push((
                    (splitmix(&mut s) as usize) % p,
                    (splitmix(&mut s) as usize) % p,
                ));
            }
            let f = MapTable::new(1, 1).seal();
            let axis = Axis::new(Width::W256, tiles);
            let mut plane = PairPlane::new(&f, axis, axis);
            for &(a, b) in &edges {
                plane.insert(a, b).unwrap();
            }
            let o = Owner(plane);
            let mut seed = vec![0u64; w];
            seed[0] = 1;
            let (mut reached, mut hop, mut scratch) =
                (vec![0u64; w], vec![0u16; p], vec![0u64; 2 * w]);
            let mut call = |seed: &[u64]| {
                relation_reach(
                    &o,
                    0,
                    &f,
                    PairDir::Forward,
                    seed,
                    None,
                    &mut ReachBuf {
                        reached: &mut reached,
                        hop: &mut hop,
                        scratch: &mut scratch,
                    },
                )
                .unwrap()
            };
            let stats = call(&seed);
            let mut times = Vec::new();
            for _ in 0..41 {
                let t0 = Instant::now();
                for _ in 0..50 {
                    black_box(call(black_box(&seed)));
                }
                times.push(t0.elapsed().as_nanos() / 50);
            }
            times.sort_unstable();
            assert_eq!(hop, bfs(&edges, p), "reach differs from the walk");
            println!(
                "{p}\t{}\t{}\t{}\t{}",
                edges.len(),
                stats.reached,
                stats.rounds,
                times[20]
            );
        }
    }
}
