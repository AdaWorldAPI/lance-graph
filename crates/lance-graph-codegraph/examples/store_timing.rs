//! Per-call timing over any ruff_charon store directory (default: the rev1
//! fixture), in nanoseconds; open and the first closure builds are cold and
//! reported in microseconds, not gated. Usage: `store_timing [dir] [symbol]`.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use lance_graph_codegraph::graph::{RCONF, RPOSS};
use lance_graph_codegraph::Codegraph;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rev1"),
        PathBuf::from,
    );
    let t = Instant::now();
    let mut g = Codegraph::open(&dir).expect("store opens");
    println!(
        "open: {} us (cold, reported, not gated)",
        t.elapsed().as_micros()
    );
    let n = (0u32..).take_while(|&p| g.symbol(p).is_some()).count();
    println!("symbols: {n}, mask words: {}", g.words());
    let probe = args.next().and_then(|s| g.pos(&s)).unwrap_or_else(|| {
        // The function with the most confirmed callers is a dense probe.
        (0..n as u32)
            .max_by_key(|&p| {
                g.called()
                    .get(p as usize / 64)
                    .map_or(0, |w| w >> (p % 64) & 1)
            })
            .unwrap_or(0)
    });
    println!("probe: {}", g.symbol(probe).unwrap_or("?"));
    for (name, rel) in [("RCONF", RCONF), ("RPOSS", RPOSS)] {
        let t = Instant::now();
        let (_, stats) = g.closure(rel).expect("closure");
        println!(
            "closure {name}: {} us (cold, reported, not gated); rounds {}, pairs {}, blocks {}/{}",
            t.elapsed().as_micros(),
            stats.rounds,
            stats.rows_composed,
            stats.blocks_allocated,
            stats.blocks_reserved
        );
    }
    const N: u32 = 10_000;
    let time = |label: &str, f: &mut dyn FnMut()| {
        for _ in 0..1000 {
            f();
        }
        let t = Instant::now();
        for _ in 0..N {
            f();
        }
        println!(
            "{label}: {:.1} ns/call",
            t.elapsed().as_nanos() as f64 / f64::from(N)
        );
    };
    time("impact", &mut || {
        black_box(g.impact(probe).expect("impact"));
    });
    time("references", &mut || {
        black_box(g.references(probe));
    });
    time("callers", &mut || {
        black_box(g.callers(probe).expect("callers"));
    });
    // Refresh every crate in the store from its own unchanged file: the
    // whole intake-side diff runs and finds nothing to log.
    let crates: Vec<String> = std::fs::read_dir(&dir)
        .expect("store dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_str()
                .and_then(|n| n.strip_suffix(".rows.tsv"))
                .map(str::to_string)
        })
        .collect();
    for k in &crates {
        let t = Instant::now();
        let lanes = lance_graph_codegraph::artifact::read_crate(&dir, k).expect("read");
        println!(
            "  of which reading {k}.rows.tsv ({} rows): {} us",
            lanes.predicate.len(),
            t.elapsed().as_micros()
        );
        let t = Instant::now();
        let stats = g.refresh_crate(&dir, k).expect("refresh");
        println!(
            "refresh_crate {k} (unchanged): {} us; {stats:?}",
            t.elapsed().as_micros()
        );
    }
    let i = g.impact(probe).expect("impact");
    let pop = |m: &[u64]| m.iter().map(|w| w.count_ones()).sum::<u32>();
    println!(
        "impact sizes: confirmed {}, possible {}, unknown {}",
        pop(&i.confirmed),
        pop(&i.possible),
        pop(&i.unknown)
    );
}
