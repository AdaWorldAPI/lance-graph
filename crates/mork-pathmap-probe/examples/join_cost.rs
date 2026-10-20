//! Cost of `Q(x, y, z) :- R(x, y), S(y, z)` over a shared, versioned
//! relationship table vs the PathMap and MORK indexes.
//!
//! Run: `cd crates/mork-pathmap-probe && cargo run --release --example join_cost`
//! (add `--features leapfrog` for MORK's leapfrog join). One machine, warm
//! caches, medians of `REPS` runs. Every answer is checked against the
//! nested-loop oracle before anything is timed.

use std::time::{Duration, Instant};

use mork_pathmap_probe::join::{JoinExec, JoinTable, S_KEY, filter_plane};
use mork_pathmap_probe::{
    Backend, JoinInput, RelationView, Semantics, bind, boundary::mork_join, reference,
};

const REPS: usize = 5;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut ts = Vec::with_capacity(REPS);
    let mut last = None;
    for _ in 0..REPS {
        let t = Instant::now();
        let v = f();
        ts.push(t.elapsed());
        last = Some(v);
    }
    (median(ts), last.unwrap())
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// The `cost.rs` fixture: y in a small domain, one z per y.
fn small_domain(n: usize) -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
    let domain = (n as u32 / 4).max(1);
    let mut s = 7u32;
    let r = (0..n as u32)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            [i, s % domain]
        })
        .collect();
    let s = (0..domain).map(|y| [y, y + 1_000_000]).collect();
    (r, s)
}

/// Two datasets with ids spread over `u32`: half of each side's y values are
/// shared, S fans out 3 z per row.
fn heterogeneous(n: usize) -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
    let ny = (n / 4) as u32;
    let ys: Vec<u32> = (0..ny + ny / 2)
        .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(17))
        .collect();
    let (ry, sy) = (&ys[..ny as usize], &ys[(ny / 2) as usize..]);
    let mut st = 99u64;
    let mut next = || {
        st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (st >> 33) as u32
    };
    let r = (0..n)
        .map(|_| [next(), ry[next() as usize % ry.len()]])
        .collect();
    let s = (0..n / 2)
        .flat_map(|_| {
            let y = sy[next() as usize % sy.len()];
            let z = next();
            (0..3u32).map(move |k| [y, z.wrapping_add(k)])
        })
        .collect();
    (r, s)
}

fn section(label: &str, r: &[[u32; 2]], s: &[[u32; 2]]) {
    let inp = JoinInput {
        r: RelationView {
            rows: r,
            generation: 1,
        },
        s: RelationView {
            rows: s,
            generation: 1,
        },
    };
    let want_set = reference::answers(&inp, Semantics::Set);
    let want_bag = reference::answers(&inp, Semantics::Bag);
    println!(
        "\n## {label}: |R| {} |S| {}, {} set answers, {} bag answers",
        r.len(),
        s.len(),
        want_set.rows.len(),
        want_bag.total()
    );

    // Upstream arms: build once, query once.
    let mut rows = Vec::new();
    for backend in [Backend::PathMapZipper, Backend::Mork] {
        let (build, mut idx) = time(|| bind(backend, &inp, Semantics::Set).unwrap());
        let (query, got) = time(|| idx.answers(&inp).unwrap());
        assert_eq!(got, want_set, "{backend:?}");
        rows.push((format!("{backend:?}"), build, query));
    }

    // Native: prepare, then answer.
    let (prep, table) = time(|| JoinTable::prepare(inp.r, inp.s));
    let mut exec = JoinExec::new(&table);
    let (q_set, got) = time(|| exec.answers(&table, (1, 1), None, Semantics::Set).unwrap());
    assert_eq!(got, want_set);
    let (q_bag, got) = time(|| exec.answers(&table, (1, 1), None, Semantics::Bag).unwrap());
    assert_eq!(got, want_bag);
    let (q_count, n) = time(|| exec.bag_count(&table, &[1, 1], None).unwrap());
    assert_eq!(n, want_bag.total());
    let (q_joint, _) = time(|| exec.joint(&table, None).unwrap().len());
    let (q_stream, streamed) = time(|| {
        let mut k = 0u64;
        exec.for_each_answer(&table, (1, 1), |row| {
            k = k.wrapping_add(u64::from(row[0] ^ row[2])) + 1
        })
        .unwrap();
        k
    });
    let _ = streamed;
    rows.push(("Native set".into(), prep, q_set));

    println!("arm, build_us, query_us (set answers into a BTreeMap)");
    for (name, b, q) in &rows {
        println!("{name}, {:.1}, {:.1}", us(*b), us(*q));
    }
    println!(
        "native extras: bag answers {:.1} us | answers streamed, no map {:.1} us | bag count (no rows) {:.2} us | joint plane alone (mask-risc And+Keep) {:.2} us",
        us(q_bag),
        us(q_stream),
        us(q_count),
        us(q_joint)
    );
    println!(
        "native table: {} B (domain {} slots, {} B; plane {} B per side)",
        table.bytes(),
        table.domain.len(),
        table.domain.bytes(),
        table.words() * 8
    );

    // Amortization, ARITHMETIC from the medians above: build + N * query.
    println!("N, pathmap_ms, mork_ms, native_set_ms, native_count_ms  [arithmetic]");
    for n in [1u32, 8, 64, 1024] {
        let tot = |b: Duration, q: Duration| (us(b) + f64::from(n) * us(q)) / 1e3;
        println!(
            "{n}, {:.2}, {:.2}, {:.2}, {:.3}",
            tot(rows[0].1, rows[0].2),
            tot(rows[1].1, rows[1].2),
            tot(prep, q_set),
            tot(prep, q_count)
        );
    }

    // 64 thoughts, each restricting y to its own slice (1 in 8), over the one
    // shared table: measured, not arithmetic.
    let filters: Vec<Vec<u64>> = (0..64u32)
        .map(|t| filter_plane(&table, |y| y.wrapping_mul(2_246_822_519) >> 29 == t % 8))
        .collect();
    let (thoughts, total) = time(|| {
        filters
            .iter()
            .map(|f| exec.bag_count(&table, &[1, 1], Some(f)).unwrap())
            .sum::<u64>()
    });
    let mut check = 0u64;
    for t in 0..64u32 {
        check += want_bag
            .rows
            .iter()
            .filter(|(row, _)| row[1].wrapping_mul(2_246_822_519) >> 29 == t % 8)
            .map(|(_, &c)| u64::from(c))
            .sum::<u64>();
    }
    assert_eq!(total, check);
    println!(
        "64 thoughts over one table, each a 1/8 y filter, bag count: {:.1} us total ({:.2} us per thought; filter plane {} B per thought)",
        us(thoughts),
        us(thoughts) / 64.0,
        table.words() * 8
    );

    // One dataset resealed: re-prepare S only vs the whole table.
    let s2: Vec<[u32; 2]> = s.iter().step_by(2).copied().collect();
    let v2 = RelationView {
        rows: &s2,
        generation: 2,
    };
    let (side, _) = time(|| {
        let mut t = table.clone();
        t.replace_side(1, v2, S_KEY).unwrap();
        t
    });
    let (full, _) = time(|| JoinTable::prepare(inp.r, v2));
    let (clone, _) = time(|| table.clone());
    println!(
        "S resealed: replace S side {:.1} us (includes a {:.1} us table clone) | full prepare {:.1} us",
        us(side),
        us(clone),
        us(full)
    );
}

fn main() {
    println!("mork join: {} | release, medians of {REPS}", mork_join());
    for n in [1_000usize, 10_000, 50_000] {
        let (r, s) = small_domain(n);
        section(&format!("cost.rs fixture, n = {n}"), &r, &s);
    }
    let (r, s) = heterogeneous(50_000);
    section("heterogeneous, ids over u32, S fan-out 3", &r, &s);
}
