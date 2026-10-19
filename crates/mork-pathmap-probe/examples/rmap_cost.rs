//! Preparation versus reuse for the D-RMAP focus probe.
//! Run: `cd crates/mork-pathmap-probe && cargo run --release --example rmap_cost`
//!
//! Every per-evaluation figure is a median over repeated runs of a real loop
//! (evaluation `i` uses seed slot `i % 256`, so no two consecutive evaluations
//! are identical). Totals for large N are MEASURED by running the loop, up to
//! a time cap; a capped arm says so and reports how far it got. The crossover
//! line is ARITHMETIC over measured inputs, labelled as such.

use std::time::{Duration, Instant};

use lance_graph_mask_risc::{Scratch, words_for};
use mork_pathmap_probe::boundary::PostingTrie;
use mork_pathmap_probe::rmap::*;

const HOPS: usize = 3;
const REPS: usize = 5;
const CAP: Duration = Duration::from_secs(20);

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median wall time of `f`, in microseconds.
fn time_us(mut f: impl FnMut()) -> f64 {
    median(
        (0..REPS)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect(),
    )
}

/// Run `eval(i)` for i in 0..n, stopping at the cap. Returns (evals done,
/// seconds, checksum).
fn run_n(n: usize, mut eval: impl FnMut(usize) -> usize) -> (usize, f64, usize) {
    let t = Instant::now();
    let mut sum = 0usize;
    for i in 0..n {
        sum = sum.wrapping_add(eval(i));
        if i & 1023 == 1023 && t.elapsed() > CAP {
            return (i + 1, t.elapsed().as_secs_f64(), sum);
        }
    }
    (n, t.elapsed().as_secs_f64(), sum)
}

fn count(r: &ScopedReach) -> usize {
    slots_of(&r.reached).len() + r.escaped.len()
}

fn graph_section(label: &str, preds: &[u32], degree: u32, community: u32, local: f64) {
    println!(
        "\n## SPOG focus, {label}: 65 536 identities, {} predicates, degree {degree} per predicate, community {community}, local share {local}",
        preds.len()
    );
    let facts = graph(65_536, preds.len() as u32, degree, community, local, 0x5EED);
    println!(
        "facts: {}  (canonical payload {} B, never copied)",
        facts.len(),
        facts.len() * 12
    );

    let t = Instant::now();
    let run = SortedRun::build(&facts, false, 1);
    let run_build = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let trie = SpogTrie::build(&facts);
    let trie_build = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "index build: sorted run {run_build:.1} ms ({} B of row refs) | PathMap trie {trie_build:.1} ms (>= {} B of key bytes; node overhead not measurable here)",
        run.bytes(),
        trie.key_bytes()
    );

    let disc_run = time_us(|| {
        std::hint::black_box(discover(&run, &[17], preds, SLOTS));
    });
    let disc_trie = time_us(|| {
        std::hint::black_box(discover(&trie, &[17], preds, SLOTS));
    });
    let focus = discover(&run, &[17], preds, SLOTS);
    let wit = Witness {
        facts: 1,
        seal: 1,
        focus: focus.generation,
        scope: 1,
    };
    let prep_csr = time_us(|| {
        std::hint::black_box(Csr::prepare(&run, &focus, preds, wit));
    });
    let csr = Csr::prepare(&run, &focus, preds, wit);
    let prep_tile = time_us(|| {
        std::hint::black_box(Tile::from_csr(&csr));
    });
    let tile = Tile::from_csr(&csr);
    let pairs: Vec<(u32, u32)> = preds
        .iter()
        .flat_map(|&p| focus.ids().iter().map(move |&s| (p, s)))
        .collect();
    let restrict = time_us(|| {
        std::hint::black_box(trie.restrict_to(pairs.iter().copied()));
    });
    println!(
        "focus discovery (BFS to 256): run {disc_run:.1} us | trie {disc_trie:.1} us\n\
         prepare: CSR {prep_csr:.1} us ({} in-focus edges, {} escapes, {} B) | tile from CSR {prep_tile:.1} us ({} B) | trie restrict {restrict:.1} us (shares nodes; bytes not measurable)\n\
         focus binding: {} B",
        csr.edges(),
        csr.escapes(),
        csr.bytes(),
        tile.bytes(),
        focus.bytes()
    );

    // Per-evaluation: 3-hop scoped reach from seed slot i % 256.
    let mut ex = TileExec::new();
    let seeds: Vec<SlotSet> = (0..SLOTS).map(|s| slot_set([s as u8])).collect();
    // Agreement check before any timing.
    for s in [0usize, 77, 255] {
        let d = reach_direct(&run, &focus, preds, &seeds[s], HOPS);
        assert_eq!(csr.reach(&wit, preds, &seeds[s], HOPS).unwrap(), d);
        assert_eq!(
            ex.reach(&tile, &csr, &wit, None, None, preds, &seeds[s], HOPS)
                .unwrap(),
            d
        );
        assert_eq!(reach_direct(&trie, &focus, preds, &seeds[s], HOPS), d);
    }
    let per = |f: &mut dyn FnMut(usize) -> usize| {
        median(
            (0..REPS)
                .map(|_| {
                    let t = Instant::now();
                    let mut x = 0;
                    for i in 0..1024 {
                        x += f(i);
                    }
                    std::hint::black_box(x);
                    t.elapsed().as_secs_f64() * 1e6 / 1024.0
                })
                .collect(),
        )
    };
    let e_run = per(&mut |i| count(&reach_direct(&run, &focus, preds, &seeds[i % SLOTS], HOPS)));
    let e_trie = per(&mut |i| count(&reach_direct(&trie, &focus, preds, &seeds[i % SLOTS], HOPS)));
    let e_csr = per(&mut |i| count(&csr.reach(&wit, preds, &seeds[i % SLOTS], HOPS).unwrap()));
    let e_tile = per(&mut |i| {
        count(
            &ex.reach(
                &tile,
                &csr,
                &wit,
                None,
                None,
                preds,
                &seeds[i % SLOTS],
                HOPS,
            )
            .unwrap(),
        )
    });
    println!(
        "per evaluation, escapes as IDENTITIES (3-hop scoped reach, median of {REPS} x 1024): direct over run {e_run:.2} us | direct over trie {e_trie:.2} us | CSR {e_csr:.2} us | tile via mask-risc {e_tile:.2} us"
    );
    let cnt = |m: &ScopedMask| slots_of(&m.reached).len() + slots_of(&m.escape_sources).len();
    for s in [0usize, 77, 255] {
        let d = reach_direct_sources(&run, &focus, preds, &seeds[s], HOPS);
        assert_eq!(csr.reach_sources(&wit, preds, &seeds[s], HOPS).unwrap(), d);
    }
    let m_run = per(&mut |i| {
        cnt(&reach_direct_sources(
            &run,
            &focus,
            preds,
            &seeds[i % SLOTS],
            HOPS,
        ))
    });
    let m_trie = per(&mut |i| {
        cnt(&reach_direct_sources(
            &trie,
            &focus,
            preds,
            &seeds[i % SLOTS],
            HOPS,
        ))
    });
    let m_csr = per(&mut |i| {
        cnt(&csr
            .reach_sources(&wit, preds, &seeds[i % SLOTS], HOPS)
            .unwrap())
    });
    let m_tile = per(&mut |i| {
        cnt(&ex
            .reach_sources(
                &tile,
                &csr,
                &wit,
                None,
                None,
                preds,
                &seeds[i % SLOTS],
                HOPS,
            )
            .unwrap())
    });
    println!(
        "per evaluation, escapes as SOURCE SLOTS (same reach; identities on demand): direct over run {m_run:.2} us | direct over trie {m_trie:.2} us | CSR {m_csr:.2} us | tile via mask-risc {m_tile:.2} us"
    );

    // Measured totals over N evaluations, escapes as source slots.
    println!(
        "N, direct_run_s, csr_total_s (discover+prepare+N), tile_total_s (discover+prepare+tile+N)  [escapes as source slots]"
    );
    let fixed_csr = (disc_run + prep_csr) / 1e6;
    let fixed_tile = fixed_csr + prep_tile / 1e6;
    for &n in &[1usize, 8, 64, 1024, 65_536, 1_000_000] {
        let (dn, ds, _) = run_n(n, |i| {
            cnt(&reach_direct_sources(
                &run,
                &focus,
                preds,
                &seeds[i % SLOTS],
                HOPS,
            ))
        });
        let (cn, cs, _) = run_n(n, |i| {
            cnt(&csr
                .reach_sources(&wit, preds, &seeds[i % SLOTS], HOPS)
                .unwrap())
        });
        let (tn, ts, _) = run_n(n, |i| {
            cnt(&ex
                .reach_sources(
                    &tile,
                    &csr,
                    &wit,
                    None,
                    None,
                    preds,
                    &seeds[i % SLOTS],
                    HOPS,
                )
                .unwrap())
        });
        let cap = |done: usize| {
            if done < n {
                format!(" (CAPPED at {done})")
            } else {
                String::new()
            }
        };
        println!(
            "{n}, {ds:.4}{}, {:.4}{}, {:.4}{}",
            cap(dn),
            fixed_csr + cs,
            cap(cn),
            fixed_tile + ts,
            cap(tn)
        );
    }
    let cross = |base: f64, prep_us: f64, per_us: f64| {
        if base > per_us {
            format!("{:.1}", prep_us / (base - per_us))
        } else {
            "never (not cheaper per evaluation)".into()
        }
    };
    println!(
        "break-even reuse count, ARITHMETIC from the measured medians above (vs direct over run): identities mode CSR {} | tile {} ; source-slot mode CSR {} | tile {}",
        cross(e_run, disc_run + prep_csr, e_csr),
        cross(e_run, disc_run + prep_csr + prep_tile, e_tile),
        cross(m_run, disc_run + prep_csr, m_csr),
        cross(m_run, disc_run + prep_csr + prep_tile, m_tile)
    );

    // Focus churn.
    let other = discover(&run, &[40_000], preds, SLOTS);
    println!("rebind k slots (incremental vs full re-prepare of CSR + tile), us:");
    for k in [1usize, 8, 16, 128, 256] {
        let changes: Vec<(u8, u32)> = (0..k)
            .zip(other.ids())
            .filter(|(_, g)| focus.slot_of(**g).is_none())
            .map(|(s, &g)| ((SLOTS - 1 - s) as u8, g))
            .collect();
        let new = focus.replace(&changes);
        let nw = Witness {
            focus: new.generation,
            ..wit
        };
        let inc = time_us(|| {
            std::hint::black_box(rebind(&run, &focus, &new, &csr, nw));
        });
        let full = time_us(|| {
            let c = Csr::prepare(&run, &new, preds, nw);
            std::hint::black_box(Tile::from_csr(&c));
        });
        println!(
            "  k={k:>3} ({} replaced): incremental {inc:.1} | full {full:.1}",
            changes.len()
        );
    }

    // Thoughts sharing the tile, each with a sparse delta of 16 edits.
    println!("thought contexts over one shared tile (delta: 8 added + 8 retracted edges each):");
    let edges: Vec<(u32, u8, u8, u32)> = csr.edge_set().into_iter().filter(|e| e.0 == 0).collect();
    for &t in &[1usize, 8, 64] {
        let mut rng = Rng(t as u64);
        let deltas: Vec<ThoughtDelta> = (0..t)
            .map(|k| {
                let add: Vec<(u8, u8)> = (0..8)
                    .map(|_| (rng.below(256) as u8, rng.below(256) as u8))
                    .collect();
                let del: Vec<(u8, u8)> = (0..8)
                    .map(|_| {
                        let e = edges[rng.below(edges.len() as u64) as usize];
                        (e.1, e.2)
                    })
                    .collect();
                ThoughtDelta::new(k as u64, 0, &add, &del)
            })
            .collect();
        let per_thought = per(&mut |i| {
            let d = &deltas[i % t];
            count(
                &ex.reach(
                    &tile,
                    &csr,
                    &wit,
                    Some(d),
                    Some(d.thought),
                    &[0],
                    &seeds[i % SLOTS],
                    HOPS,
                )
                .unwrap(),
            )
        });
        println!(
            "  {t:>2} thoughts: per evaluation {per_thought:.2} us | per-thought memory {} B sparse, {} B as executor planes | shared tile {} B once",
            deltas[0].bytes_sparse(),
            deltas[0].bytes_planes(),
            tile.bytes()
        );
    }
}

fn places_section() {
    println!("\n## Places: 65 536 canonical rows, focus of 256");
    let places = places(65_536, 3);
    let morton = morton_order(&places);
    println!(
        "Morton order index: {} B of row refs (coordinates not copied)",
        morton.len() * 4
    );
    let mut rng = Rng(11);
    let mut scattered: Vec<u32> = Vec::new();
    while scattered.len() < SLOTS {
        let i = rng.below(65_536) as u32;
        if !scattered.contains(&i) {
            scattered.push(i);
        }
    }
    let clustered: Vec<u32> = morton[30_000..30_000 + SLOTS].to_vec();
    let mut scratch = Scratch::new(words_for(65_536), 2);
    let qs: Vec<BoxQuery> = (0..64)
        .map(|k| BoxQuery {
            lat: (-90_000_000 + k * 2_000_000, 90_000_000),
            lon: (-180_000_000, 180_000_000 - k * 4_000_000),
            class: (k as u32) % 16,
        })
        .collect();
    for (name, ids) in [("scattered", &scattered), ("Morton-clustered", &clustered)] {
        let bind = time_us(|| {
            std::hint::black_box(FocusMask::over(65_536, ids, Ordering::Canonical, 1));
        });
        let fm = FocusMask::over(65_536, ids, Ordering::Canonical, 1);
        let live = fm.words.iter().filter(|&&w| w != 0).count();
        for q in &qs {
            assert_eq!(
                box_count(&places, &fm, q, &mut scratch).unwrap(),
                box_count_by_id(&places, ids, q)
            );
        }
        let native = time_us(|| {
            for q in &qs {
                std::hint::black_box(box_count(&places, &fm, q, &mut scratch).unwrap());
            }
        }) / qs.len() as f64;
        let by_id = time_us(|| {
            for q in &qs {
                std::hint::black_box(box_count_by_id(&places, ids, q));
            }
        }) / qs.len() as f64;
        let load = time_us(|| {
            std::hint::black_box(FocusLanes::load(&places, ids, 1));
        });
        let lanes = FocusLanes::load(&places, ids, 1);
        let mut small = Scratch::new(SLOT_WORDS, 2);
        for q in &qs {
            assert_eq!(
                lanes.count(1, q, &mut small).unwrap(),
                box_count_by_id(&places, ids, q)
            );
        }
        let reg = time_us(|| {
            for q in &qs {
                std::hint::black_box(lanes.count(1, q, &mut small).unwrap());
            }
        }) / qs.len() as f64;
        println!(
            "{name}: bind mask {bind:.2} us (8 KiB plane, {live} live words of 1024) | bbox+class count: mask-risc over canonical lanes {native:.2} us (0 B copied) | read each id's lanes {by_id:.2} us (0 B copied) | register lanes: load {load:.2} us ({} B COPIED), then mask-risc over 256 rows {reg:.2} us",
            lanes.bytes()
        );
    }
}

fn postings_section() {
    println!("\n## Inverted index: 256 terms over 65 536 docs, ~40 docs per term");
    let mut postings: Vec<Vec<u32>> = vec![vec![17, 21, 39], vec![21, 39, 82]];
    let mut rng = Rng(9);
    for _ in 0..254 {
        let mut v: Vec<u32> = (0..40).map(|_| rng.below(65_536) as u32).collect();
        v.sort_unstable();
        v.dedup();
        postings.push(v);
    }
    let pairs: Vec<[u32; 2]> = postings
        .iter()
        .enumerate()
        .flat_map(|(t, d)| d.iter().map(move |&x| [t as u32, x]))
        .collect();
    let trie = PostingTrie::build(&pairs);
    // Focus: the docs of terms 2..8 (a scope that fits 256 slots).
    let mut scope: Vec<u32> = postings[2..8].iter().flatten().copied().collect();
    scope.sort_unstable();
    scope.dedup();
    let focus = Focus::bind(&scope, 1);
    let lists: Vec<&[u32]> = postings[2..8].iter().map(|v| v.as_slice()).collect();
    let prep = time_us(|| {
        std::hint::black_box(membership_planes(&lists, &focus));
    });
    let planes = membership_planes(&lists, &focus);
    let mut scratch = Scratch::new(SLOT_WORDS, 2);
    let pairs6: Vec<(usize, usize)> = (0..6)
        .flat_map(|a| (0..6).map(move |b| (a, b)))
        .filter(|(a, b)| a < b)
        .collect();
    for &(a, b) in &pairs6 {
        let want = intersect_sorted(lists[a], lists[b]);
        assert_eq!(trie.intersect(a as u32 + 2, b as u32 + 2), want);
        assert_eq!(
            intersect_planes(&planes[a], &planes[b], &focus, &mut scratch),
            want
        );
    }
    let n = pairs6.len() as f64;
    let sorted = time_us(|| {
        for &(a, b) in &pairs6 {
            std::hint::black_box(intersect_sorted(lists[a], lists[b]));
        }
    }) / n;
    let tri = time_us(|| {
        for &(a, b) in &pairs6 {
            std::hint::black_box(trie.intersect(a as u32 + 2, b as u32 + 2));
        }
    }) / n;
    let keep = time_us(|| {
        for &(a, b) in &pairs6 {
            std::hint::black_box(intersect_planes(
                &planes[a],
                &planes[b],
                &focus,
                &mut scratch,
            ));
        }
    }) / n;
    let cnt = time_us(|| {
        for &(a, b) in &pairs6 {
            std::hint::black_box(intersect_count(&planes[a], &planes[b], &mut scratch));
        }
    }) / n;
    println!(
        "focus {} docs, membership planes prepared in {prep:.2} us ({} B)\n\
         per intersection: sorted merge {sorted:.3} us | PathMap zipper merge {tri:.3} us | mask-risc And+Keep -> ids {keep:.3} us | mask-risc And+Count {cnt:.3} us",
        focus.len(),
        planes.len() * SLOT_WORDS * 8
    );
}

fn main() {
    println!("D-RMAP focus probe — release, one machine, medians of {REPS} repetitions");
    let three = [0u32, 1, 2];
    graph_section("sparse", &three, 2, 512, 0.9);
    graph_section("sparse, degree 8", &three, 8, 512, 0.9);
    graph_section("dense in-community", &[0], 48, 256, 1.0);
    places_section();
    postings_section();
}
