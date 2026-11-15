//! Reconcile every `r#"..."#` literal of the given files.
//! Usage: reconcile [--out <tsv path>] <file.rs>...
//!
//! Prints the ladder counts, the reasons behind refusals and missing oracle
//! answers, and every rule on which the cube and the reference evaluator
//! disagree. With `--out`, writes one TSV row per rule (and one per program
//! that has no rule): file, literal index, head, rung, host_loop, reason.
use std::collections::BTreeMap;
use std::fmt::Write as _;

use xlog_coverage_probe::{literals, reconcile, Rung, DEFAULT_SEEDS};

const NOT_BINARY: &str = "is not a binary relation in the cube";

fn flat(s: &str) -> String {
    s.chars()
        .map(|c| if matches!(c, '\t' | '\n' | '\r') { ' ' } else { c })
        .collect()
}

fn shown(s: &str) -> String {
    const MAX: usize = 200;
    let s = flat(s);
    if s.chars().count() > MAX {
        let cut: String = s.chars().take(MAX).collect();
        format!("{cut}...")
    } else {
        s
    }
}

fn print_counts(title: &str, counts: BTreeMap<String, usize>) {
    println!("{title}");
    if counts.is_empty() {
        println!("  (none)");
        return;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (why, n) in counts {
        println!("  {n:4}  {}", shown(&why));
    }
}

fn main() {
    let mut out_path: Option<String> = None;
    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            out_path = Some(args.next().unwrap_or_else(|| {
                eprintln!("--out needs a path");
                std::process::exit(2)
            }));
        } else {
            files.push(a);
        }
    }

    let (mut literal_count, mut compiled, mut rule_count) = (0usize, 0usize, 0usize);
    let mut by_rung: BTreeMap<Rung, usize> = BTreeMap::new();
    let mut native_tested = 0usize;
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let mut unavailable: BTreeMap<String, usize> = BTreeMap::new();
    let mut disagreements: Vec<String> = Vec::new();
    let mut tsv = String::new();

    for path in &files {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("{path}: {e}");
            std::process::exit(1)
        });
        for (index, lit) in literals(&text).into_iter().enumerate() {
            literal_count += 1;
            let rec = reconcile(lit, &DEFAULT_SEEDS);
            if let Some(rung) = rec.program_rung {
                *by_rung.entry(rung).or_default() += 1;
                if rung == Rung::Compiled {
                    compiled += 1;
                    *unavailable
                        .entry(format!("plan refused: {}", rec.program_reason))
                        .or_default() += 1;
                }
                let _ = writeln!(
                    tsv,
                    "{path}\t{index}\t\t{}\tfalse\t{}",
                    rung.label(),
                    flat(&rec.program_reason)
                );
                continue;
            }
            compiled += 1;
            for r in &rec.rules {
                rule_count += 1;
                *by_rung.entry(r.rung).or_default() += 1;
                if r.rung == Rung::OracleAgrees && !r.host_loop {
                    native_tested += 1;
                }
                match r.rung {
                    Rung::Refused => {
                        let key = if r.reason.contains(NOT_BINARY) {
                            "reads a relation that is not binary".to_string()
                        } else {
                            r.reason.clone()
                        };
                        *refusals.entry(key).or_default() += 1;
                    }
                    Rung::LoweredUnverified => {
                        *unavailable.entry(r.reason.clone()).or_default() += 1;
                    }
                    Rung::OracleDisagrees => disagreements.push(format!(
                        "{path} literal {index} head {}: {}",
                        r.head,
                        flat(&r.reason)
                    )),
                    _ => {}
                }
                let _ = writeln!(
                    tsv,
                    "{path}\t{index}\t{}\t{}\t{}\t{}",
                    r.head,
                    r.rung.label(),
                    r.host_loop,
                    flat(&r.reason)
                );
            }
        }
    }

    println!(
        "files {}, literals {literal_count}, programs-compiled {compiled}, rules {rule_count}",
        files.len()
    );
    let count = |r: Rung| by_rung.get(&r).copied().unwrap_or(0);
    println!("{:20} programs {}", Rung::Observed.label(), count(Rung::Observed));
    println!("{:20} programs {}", Rung::Compiled.label(), count(Rung::Compiled));
    for rung in [
        Rung::Refused,
        Rung::LoweredUnverified,
        Rung::OracleDisagrees,
        Rung::OracleAgrees,
    ] {
        println!("{:20} rules    {}", rung.label(), count(rung));
    }
    println!("native-tested (oracle-agrees and not host_loop): {native_tested}");
    println!("production-admitted: 0 (no production caller; xlog-cube is a probe)");

    print_counts("refusal reasons:", refusals);
    print_counts("oracle unavailable (or plan refused):", unavailable);

    println!("oracle disagreements: {}", disagreements.len());
    for d in &disagreements {
        println!("  {d}");
    }

    if let Some(path) = out_path {
        let mut text = String::from("file\tliteral\thead\trung\thost_loop\treason\n");
        text.push_str(&tsv);
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }
}
