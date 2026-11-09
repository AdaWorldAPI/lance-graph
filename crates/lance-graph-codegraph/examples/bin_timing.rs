//! Timing gate for the binary rows format (plan `d-codegraph-0-binary-lanes-v1`
//! §5): an unchanged `refresh_crate` against a `rows.bin` store, and the intake
//! of one whole crate from `rows.bin` (read, verify, fill lanes) beside the
//! same crate read from `rows.tsv`. N runs each, pre-registered at 41; median
//! and range in microseconds. Run in release.
//!
//! Usage: `bin_timing <dir> <crate>` where `<dir>` holds `<crate>.rows.bin`
//! and `<crate>.rows.tsv` (as `ruff_charon import` writes them).

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use lance_graph_codegraph::artifact::{self, Lanes, Symbols};
use lance_graph_codegraph::artifact_bin::CrateFile;
use lance_graph_codegraph::Codegraph;

const N: usize = 41;

fn stats(name: &str, mut ns: Vec<u128>) {
    ns.sort_unstable();
    let us = |v: u128| v as f64 / 1000.0;
    println!(
        "{name}: median {:.2} us, min {:.2} us, max {:.2} us (N={})",
        us(ns[ns.len() / 2]),
        us(ns[0]),
        us(ns[ns.len() - 1]),
        ns.len()
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: bin_timing <dir> <crate>"));
    let krate = args.next().expect("usage: bin_timing <dir> <crate>");
    let bin = dir.join(format!("{krate}.rows.bin"));
    let tsv = dir.join(format!("{krate}.rows.tsv"));

    let mut g = Codegraph::open(&dir).expect("store opens");
    let rows = {
        let file = CrateFile::parse(std::fs::read(&bin).expect("bin")).expect("parse");
        file.n_rows()
    };
    println!("crate {krate}: {rows} rows");

    let mut unchanged = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let s = g.refresh_crate(&dir, &krate).expect("refresh");
        unchanged.push(t.elapsed().as_nanos());
        black_box(s);
    }
    stats("refresh_crate (unchanged, rows.bin)", unchanged);

    let mut bin_intake = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let file = CrateFile::parse(std::fs::read(&bin).expect("bin")).expect("parse");
        let mut syms = Symbols::default();
        let mut lanes = Lanes::default();
        artifact::read_bin_as(&file, &krate, &mut syms, &mut lanes).expect("lanes");
        bin_intake.push(t.elapsed().as_nanos());
        black_box(lanes.len());
    }
    stats(
        "changed-crate intake (rows.bin: read+verify+lanes)",
        bin_intake,
    );

    let (mut r, mut v, mut l) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..N {
        let t = Instant::now();
        let buf = std::fs::read(&bin).expect("bin");
        r.push(t.elapsed().as_nanos());
        let t = Instant::now();
        let file = CrateFile::parse(buf).expect("parse");
        v.push(t.elapsed().as_nanos());
        let t = Instant::now();
        let mut syms = Symbols::default();
        let mut lanes = Lanes::default();
        artifact::read_bin_as(&file, &krate, &mut syms, &mut lanes).expect("lanes");
        l.push(t.elapsed().as_nanos());
        black_box(lanes.len());
    }
    stats("  of which fs::read", r);
    stats("  of which CrateFile::parse (verify)", v);
    stats("  of which read_bin_as (intern + lanes)", l);

    let mut tsv_intake = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let text = std::fs::read_to_string(&tsv).expect("tsv");
        let mut syms = Symbols::default();
        let mut lanes = Lanes::default();
        artifact::read_text_as(&text, &krate, &mut syms, &mut lanes).expect("lanes");
        tsv_intake.push(t.elapsed().as_nanos());
        black_box(lanes.len());
    }
    stats("baseline intake (rows.tsv: read+parse+lanes)", tsv_intake);
}
