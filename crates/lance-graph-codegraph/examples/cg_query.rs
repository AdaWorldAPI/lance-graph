//! Read-only command-line access to the four Codegraph queries, for agents and
//! people who would otherwise have to write Rust to ask one question.
//!
//! ```text
//! cargo run --release --example cg_query -- <store_dir> <src_root> symbols [SUBSTRING]
//! cargo run --release --example cg_query -- <store_dir> <src_root> callers|callees|impact|refs SYMBOL
//! cargo run --release --example cg_query -- <store_dir> <src_root> effects SYMBOL | --kind EFFECT:KIND | --reach SYMBOL (--kind EFFECT:KIND | --class LIST)
//! cargo run --release --example cg_query -- <store_dir> <src_root> impact-report SYMBOL \
//!     [--deps a,b | --consumers-closed] [--targets CRATE=lib,tests,examples,benches]... \
//!     [--exclude CRATE=REGION]... [--no-freshness]
//! ```
//!
//! `effects SYMBOL` and `effects --kind EFFECT:KIND` (for example `ASSERT:DivisionByZero`,
//! `DEREF_WRITE:Arg`) print the rows of `<crate>.effects.tsv`: what a function body witnesses.
//! A function without a row is printed as `NONE-WITNESSED`, which is not a purity verdict, and a
//! crate without an effects file as `CRATE-NOT-READ`. `--reach SYMBOL` joins the effect rows with
//! the call rows: the effects in bodies SYMBOL is or reaches, each with its grade. `--class` takes
//! a list of `division,remainder,shift,write`. Every known reason the answer may be missing
//! effects is a `LIMIT` line (a crate without an effects file, an incomplete harvest, dynamic
//! calls), and an answer with a limit and no hit is printed as `UNKNOWN`, never as none.
//!
//! `impact-report` (alias `report`) prints [`Codegraph::impact_report`]: whether the symbol has a
//! position, the impact masks, the stored crates, and every coverage gap of the declared scope,
//! one `GAP` line each. Nothing is assumed: a crate without `--targets` is reported as undeclared
//! (`--targets '*=lib,tests'` declares every stored crate), consumers are undeclared without
//! `--deps` (`--deps ""` declares none) or `--consumers-closed`, and `--exclude` names code the
//! extraction did not translate (an opaque module, a feature left off).
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

use lance_graph_codegraph::effects::{Effect, EffectAnswer, EffectClass, EffectKind, Effects};
use lance_graph_codegraph::impact_report::{Consumers, Scope, SymbolState, Targets};
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

fn effect_line(src: &Path, e: &Effect) -> String {
    let (effect, kind) = e.kind.columns();
    let site = match &e.site {
        Some(s) => format!(
            "{}\t{}\t{}..{}",
            s.file,
            line_of(src, &s.file, s.start),
            s.start,
            s.end
        ),
        None => "-\t-\t-".to_string(),
    };
    format!("effect\t{effect}\t{kind}\t{}\t{site}", e.subject)
}

fn parse_classes(list: &str) -> Option<Vec<EffectClass>> {
    list.split(',')
        .map(|c| match c {
            "division" => Some(EffectClass::Division),
            "remainder" => Some(EffectClass::Remainder),
            "shift" => Some(EffectClass::Shift),
            "write" => Some(EffectClass::Write),
            _ => None,
        })
        .collect()
}

fn parse_kind(spec: &str) -> Option<EffectKind> {
    match spec.split_once(':') {
        Some(("ASSERT", k)) if !k.is_empty() => Some(EffectKind::Assert(k.to_string())),
        Some(("DEREF_WRITE", "Arg")) => Some(EffectKind::DerefWriteArg),
        Some(("DEREF_WRITE", "Local")) => Some(EffectKind::DerefWriteLocal),
        _ => None,
    }
}

/// `effects SYMBOL`, `effects --kind EFFECT:KIND` or `effects --reach SYMBOL (--kind EFFECT:KIND | --class LIST)`. Exit 4 when the symbol's crate has no
/// effects file; 0 otherwise, also for a function without a row.
fn effects(store: &Path, src: &Path, args: &[String]) -> ExitCode {
    let fx = match Effects::open(store) {
        Ok(fx) => fx,
        Err(e) => {
            println!("REFUSED\teffects\t{e:?}");
            return ExitCode::from(3);
        }
    };
    let read: Vec<&str> = fx.crates().collect();
    match args {
        [flag, spec] if flag == "--kind" => {
            let Some(kind) = parse_kind(spec) else {
                eprintln!("--kind needs ASSERT:<tag>, DEREF_WRITE:Arg or DEREF_WRITE:Local");
                return ExitCode::from(2);
            };
            let mut n = 0u32;
            for e in fx.with_kind(&kind) {
                println!("{}", effect_line(src, e));
                n += 1;
            }
            println!("effects\t{n}\tcrates-read={}", read.join(","));
            ExitCode::SUCCESS
        }
        [flag, sym, how, spec] if flag == "--reach" && (how == "--kind" || how == "--class") => {
            let one_kind = if how == "--kind" {
                parse_kind(spec)
            } else {
                None
            };
            let classes = if how == "--class" {
                parse_classes(spec)
            } else {
                None
            };
            if one_kind.is_none() && classes.is_none() {
                eprintln!("--kind needs ASSERT:<tag>, DEREF_WRITE:Arg or DEREF_WRITE:Local; --class needs a list of division,remainder,shift,write");
                return ExitCode::from(2);
            }
            let keep = |k: &EffectKind| match (&one_kind, &classes) {
                (Some(want), _) => k == want,
                (_, Some(cs)) => k.class().is_some_and(|c| cs.contains(&c)),
                _ => false,
            };
            let g = match Codegraph::open(store) {
                Ok(g) => g,
                Err(e) => {
                    println!("REFUSED\topen\t{e:?}");
                    return ExitCode::from(3);
                }
            };
            match g.effects_reached(sym, &fx, keep) {
                Err(e) => {
                    println!("REFUSED\teffects\t{e:?}");
                    ExitCode::from(3)
                }
                Ok(None) => {
                    println!("UNKNOWN\tsymbol-not-in-store\t{sym}");
                    ExitCode::from(4)
                }
                Ok(Some(r)) => {
                    for h in &r.hits {
                        println!("reach\t{:?}\t{}", h.grade, effect_line(src, h.effect));
                    }
                    for l in &r.limits {
                        println!("LIMIT\t{l:?}");
                    }
                    for k in &read {
                        if let Some(i) = fx.identity(k) {
                            println!("effects-file\t{k}\tfnv64={:016x}\trows={}", i.fnv64, i.rows);
                        }
                    }
                    // A partial answer with no hit is "unknown", never "none".
                    let status = match (r.partial(), r.hits.is_empty()) {
                        (true, true) => "UNKNOWN (partial: see LIMIT lines)",
                        (true, false) => "PARTIAL (see LIMIT lines)",
                        (false, _) => "no known limit (not a proof of absence)",
                    };
                    println!("reached\t{}\t{status}", r.hits.len());
                    ExitCode::SUCCESS
                }
            }
        }
        [sym] if !sym.starts_with("--") => match fx.of(sym) {
            EffectAnswer::Witnessed(rows) => {
                for e in rows {
                    println!("{}", effect_line(src, e));
                }
                ExitCode::SUCCESS
            }
            EffectAnswer::NoneWitnessed => {
                println!("NONE-WITNESSED\t{sym}\tnot a purity verdict");
                ExitCode::SUCCESS
            }
            EffectAnswer::CrateNotRead => {
                println!("CRATE-NOT-READ\t{sym}\tcrates-read={}", read.join(","));
                ExitCode::from(4)
            }
        },
        _ => {
            eprintln!(
                "effects needs a SYMBOL, --kind EFFECT:KIND, or --reach SYMBOL (--kind EFFECT:KIND | --class LIST)"
            );
            ExitCode::from(2)
        }
    }
}

fn parse_targets(list: &str) -> Option<Targets> {
    let mut t = Targets::default();
    for k in list.split(',') {
        match k {
            "lib" => t.lib = true,
            "tests" => t.tests = true,
            "examples" => t.examples = true,
            "benches" => t.benches = true,
            _ => return None,
        }
    }
    Some(t)
}

/// `impact-report SYMBOL [flags]`. The flags declare what the rows cannot say.
fn impact_report(g: &Codegraph, store: &Path, src: &Path, sym: &str, flags: &[String]) -> ExitCode {
    let mut scope = Scope {
        store: Some(store.to_path_buf()),
        source_root: Some(src.to_path_buf()),
        ..Scope::default()
    };
    let pair = |v: Option<&String>| {
        v.and_then(|v| v.split_once('='))
            .map(|(a, b)| (a.to_string(), b.to_string()))
    };
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match f.as_str() {
            "--no-freshness" => {
                scope.store = None;
                scope.source_root = None;
            }
            "--consumers-closed" | "--deps" if scope.consumers != Consumers::NotDeclared => {
                eprintln!("--deps and --consumers-closed are given once and exclude each other");
                return ExitCode::from(2);
            }
            "--consumers-closed" => scope.consumers = Consumers::AllInStore,
            "--deps" => match it.next() {
                Some(v) if !v.starts_with("--") => {
                    scope.consumers = Consumers::Declared(
                        v.split(',')
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect(),
                    );
                }
                _ => {
                    eprintln!("--deps needs a value: a comma-separated list, or \"\" for none");
                    return ExitCode::from(2);
                }
            },
            "--targets" => match pair(it.next()) {
                Some((k, list)) => match parse_targets(&list) {
                    Some(t) if k == "*" => {
                        scope
                            .targets
                            .extend(g.crate_list().into_iter().map(|(c, _)| (c, t)));
                    }
                    Some(t) => scope.targets.push((k, t)),
                    None => {
                        eprintln!("bad --targets list {list}");
                        return ExitCode::from(2);
                    }
                },
                None => {
                    eprintln!("--targets needs CRATE=LIST");
                    return ExitCode::from(2);
                }
            },
            "--exclude" => match pair(it.next()) {
                Some((k, region)) => match scope.excluded.iter_mut().find(|(n, _)| *n == k) {
                    Some((_, r)) => r.push(region),
                    None => scope.excluded.push((k, vec![region])),
                },
                None => {
                    eprintln!("--exclude needs CRATE=REGION");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let r = match g.impact_report(sym, &scope) {
        Ok(r) => r,
        Err(e) => {
            println!("REFUSED\timpact-report\t{e:?}");
            return ExitCode::from(3);
        }
    };
    let found = match r.symbol {
        SymbolState::Found(p) => {
            println!("symbol\tFOUND\t{p}\t{sym}");
            true
        }
        SymbolState::Missing { owner_stored } => {
            println!("symbol\tMISSING\towner_stored={owner_stored}\t{sym}");
            false
        }
    };
    for (grade, mask) in [
        ("CONFIRMED", &r.graded.confirmed),
        ("POSSIBLE", &r.graded.possible),
        ("UNKNOWN", &r.graded.unknown),
    ] {
        for n in names(g, mask) {
            println!("impact\t{grade}\t{n}");
        }
    }
    for (c, st) in &r.observed {
        println!("observed\t{c}\t{st:?}");
    }
    for gap in &r.gaps {
        println!("GAP\t{gap:?}");
    }
    println!("closed_in_scope\t{}", r.closed_in_scope());
    // A missing symbol is not an empty impact: exit 4, as the other queries do.
    if found {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(4)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: cg_query <store_dir> <src_root> symbols [SUBSTR] | sweep | callers|callees|impact|refs SYMBOL | effects SYMBOL|--kind EFFECT:KIND | impact-report SYMBOL [--deps a,b | --consumers-closed] [--targets CRATE=LIST]... [--exclude CRATE=REGION]... [--no-freshness]");
        return ExitCode::from(2);
    }
    let (store, src, cmd) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        args[2].as_str(),
    );
    if cmd == "effects" {
        return effects(&store, &src, &args[3..]);
    }
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
    if cmd == "sweep" {
        // Runs `impact` and `callers` for every position in one process and counts
        // answers, refusals (by kind) and empty sets. Measures where the cube's
        // limits bind on this store; prints nothing per symbol.
        let started = std::time::Instant::now();
        let (mut ok, mut empty) = (0u32, 0u32);
        let mut refused: std::collections::BTreeMap<String, u32> = Default::default();
        let mut p = 0u32;
        while g.symbol(p).is_some() {
            for which in ["impact", "callers"] {
                let r = match which {
                    "impact" => g.impact(p).map(|i| {
                        i.confirmed
                            .iter()
                            .chain(&i.possible)
                            .chain(&i.unknown)
                            .any(|w| *w != 0)
                    }),
                    _ => g.callers(p).map(|c| c.sites > 0),
                };
                match r {
                    Ok(true) => ok += 1,
                    Ok(false) => empty += 1,
                    Err(e) => {
                        let kind = format!("{e:?}");
                        let kind = kind
                            .split(['(', '{', ' '])
                            .take(2)
                            .collect::<Vec<_>>()
                            .join(" ");
                        *refused.entry(format!("{which} {kind}")).or_default() += 1;
                    }
                }
            }
            p += 1;
        }
        println!(
            "sweep\tpositions={p}\tanswered={ok}\tempty={empty}\tms={}",
            started.elapsed().as_millis()
        );
        for (k, n) in refused {
            println!("sweep\tREFUSED\t{k}\t{n}");
        }
        return ExitCode::SUCCESS;
    }
    if cmd == "impact-report" || cmd == "report" {
        let Some(sym) = args.get(3) else {
            eprintln!("{cmd} needs a SYMBOL");
            return ExitCode::from(2);
        };
        return impact_report(&g, &store, &src, sym, &args[4..]);
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
