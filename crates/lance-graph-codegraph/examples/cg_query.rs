//! Read-only command-line access to the four Codegraph queries, for agents and
//! people who would otherwise have to write Rust to ask one question.
//!
//! ```text
//! cargo run --release --example cg_query -- <store_dir> <src_root> symbols [SUBSTRING]
//! cargo run --release --example cg_query -- <store_dir> <src_root> callers|callees|impact|refs SYMBOL
//! ```
//!
//! Output is tab-separated, one fact per line. Every result line carries its
//! grade (`CONFIRMED`, `POSSIBLE`, `UNKNOWN`); a refusal is printed verbatim
//! with `REFUSED` and exit code 3, never as an empty answer. Callers and impact of
//! a `PUBLIC` symbol always add an `UNKNOWN external-callers-unobserved` line, and
//! every graded answer lists the files the store observed. `src_root` is the
//! directory the store's file paths are relative to; it is used only to turn a
//! byte offset into a 1-based line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lance_graph_codegraph::Codegraph;

fn line_of(src_root: &Path, file: &str, offset: u32) -> String {
    match std::fs::read(src_root.join(file)) {
        Ok(bytes) => {
            let end = (offset as usize).min(bytes.len());
            (bytes[..end].iter().filter(|&&b| b == b'\n').count() + 1).to_string()
        }
        Err(_) => "?".to_string(),
    }
}

fn names(g: &Codegraph, mask: &[u64]) -> Vec<String> {
    let mut out = Vec::new();
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros();
            bits &= bits - 1;
            if let Some(s) = g.symbol((w as u32) * 64 + b) {
                out.push(s.to_string());
            }
        }
    }
    out.sort();
    out
}

fn bit(mask: &[u64], p: u32) -> bool {
    mask.get((p / 64) as usize)
        .is_some_and(|w| w >> (p % 64) & 1 == 1)
}

/// Files that contributed at least one row. Anything else (tests, examples,
/// other crates) was not observed by this store.
fn observed_files(g: &Codegraph) -> Vec<String> {
    let mut files: Vec<String> = (0..g.rows())
        .filter_map(|r| g.site(r).map(|(f, _, _)| f.to_string()))
        .collect();
    files.sort();
    files.dedup();
    files
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: cg_query <store_dir> <src_root> symbols [SUBSTR] | callers|callees|impact|refs SYMBOL");
        return ExitCode::from(2);
    }
    let (store, src, cmd) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        args[2].as_str(),
    );
    let mut g = match Codegraph::open(&store) {
        Ok(g) => g,
        Err(e) => {
            println!("REFUSED\topen\t{e:?}");
            return ExitCode::from(3);
        }
    };
    if cmd == "symbols" {
        let sub = args.get(3).map(String::as_str).unwrap_or("");
        let mut p = 0u32;
        while let Some(s) = g.symbol(p) {
            if s.contains(sub) {
                println!("symbol\t{p}\t{s}");
            }
            p += 1;
        }
        return ExitCode::SUCCESS;
    }
    let Some(sym) = args.get(3) else {
        eprintln!("{cmd} needs a SYMBOL");
        return ExitCode::from(2);
    };
    let Some(pos) = g.pos(sym) else {
        println!("UNKNOWN\tsymbol-not-in-store\t{sym}");
        return ExitCode::from(4);
    };
    let graded = match cmd {
        "callers" => g.callers(pos).map(|c| (c.graded, Some(c.sites))),
        "callees" => g.callees(pos).map(|c| (c.graded, Some(c.sites))),
        "impact" => g.impact(pos).map(|i| (i, None)),
        "refs" => {
            let s = g.references(pos);
            for (w, &word) in s.rows.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let b = bits.trailing_zeros();
                    bits &= bits - 1;
                    let row = w * 64 + b as usize;
                    let at = g
                        .site(row)
                        .map(|(f, st, _)| format!("{f}:{}", line_of(&src, f, st)))
                        .unwrap_or_else(|| "-".into());
                    println!(
                        "ref\t{}\t{}\t{at}",
                        g.row_predicate(row),
                        g.row_subject(row)
                    );
                }
            }
            println!("sites\tconfirmed={}\tpossible={}", s.confirmed, s.possible);
            return ExitCode::SUCCESS;
        }
        _ => {
            eprintln!("unknown command {cmd}");
            return ExitCode::from(2);
        }
    };
    match graded {
        Ok((gr, sites)) => {
            for (grade, mask) in [
                ("CONFIRMED", &gr.confirmed),
                ("POSSIBLE", &gr.possible),
                ("UNKNOWN", &gr.unknown),
            ] {
                for n in names(&g, mask) {
                    println!("{cmd}\t{grade}\t{n}");
                }
            }
            if let Some(s) = sites {
                println!("sites\t{s}");
            }
            // The store only sees the crates it was built from. A public symbol can
            // have callers anywhere else, so an empty answer is "not observed", never
            // "none". This is printed on every answer, not only on empty ones.
            if cmd != "callees" && bit(g.public(), pos) {
                println!("{cmd}\tUNKNOWN\texternal-callers-unobserved\t{sym} is PUBLIC");
            }
            for f in observed_files(&g) {
                println!("coverage\tobserved-file\t{f}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("REFUSED\t{cmd}\t{e:?}");
            ExitCode::from(3)
        }
    }
}
