//! Prints the verdict table of one rows file, the cold cost, and one proof.
//!
//! `cargo run --release --example report -- tests/data/m0.defuse.tsv [file:line]`
//!
//! The times are reported, not gates: the derivation runs once per program,
//! never per query.

use std::time::Instant;

use cg_defuse_probe::{derive, fixpoint, nesting_class, parse};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("rows file");
    let want = args.next();
    let facts = parse(&std::fs::read_to_string(&path).expect("read")).expect("rows");
    let t = Instant::now();
    let d = derive(&facts, 64).expect("derive");
    let cold = t.elapsed();
    let t = Instant::now();
    let plain = fixpoint(&facts);
    let plain_cost = t.elapsed();
    println!("rows\t{path}");
    println!("sites\t{}\titerations\t{}", d.verdicts.len(), d.iterations);
    println!(
        "cold_egglog_us\t{}\tplain_fixpoint_us\t{}\t(reported, not gates)",
        cold.as_micros(),
        plain_cost.as_micros()
    );
    println!("file\tline\top\tfunction\tderived\tfixpoint\tnesting\tproof_bytes");
    for (v, p) in d.verdicts.iter().zip(&plain) {
        println!(
            "{}\t{}\t{}\t{}\t{:?}\t{:?}\t{:?}\t{}",
            v.site.file,
            v.site.line,
            v.site.op,
            v.site.func,
            v.class,
            p,
            nesting_class(&v.site),
            v.proof.as_ref().map_or(0, String::len)
        );
    }
    if let Some(w) = want {
        for v in &d.verdicts {
            if format!("{}:{}", v.site.file, v.site.line) == w {
                println!(
                    "\nproof for {w} ({:?}):\n{}",
                    v.class,
                    v.proof.as_deref().unwrap_or("none")
                );
                break;
            }
        }
    }
}
