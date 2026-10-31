//! D-SAG-6-1 lab probe: can ra_ap index a workspace on stable, how fast,
//! how complete is its MIR lowering. Read-only; prints counts and timings.
//!
//! Output is `key<TAB>value` lines. `sample` lines show the first three
//! errors of each MIR lowering error kind. The sysroot must be set
//! (`RustLibSource::Discover`, needs the `rust-src` component): without it
//! `push`, `collect` and `is_empty` are unresolved and coverage drops to 31%.
#![allow(deprecated)]
use ra_ap_hir::{AssocItem, Crate, DefWithBody, Module, ModuleDef};
use ra_ap_ide::{
    AnalysisHost, FilePosition, GotoImplementationConfig, MonikerResult, StaticIndex,
    SymbolInformationKind, VendoredLibrariesConfig,
};
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::CargoConfig;
use std::time::Instant;

fn walk(db: &dyn ra_ap_hir::db::HirDatabase, m: Module, out: &mut Vec<DefWithBody>) {
    for d in m.declarations(db) {
        match d {
            ModuleDef::Function(f) => {
                if f.has_body(db) {
                    out.push(DefWithBody::Function(f))
                }
            }
            ModuleDef::Const(c) => out.push(DefWithBody::Const(c)),
            ModuleDef::Static(s) => out.push(DefWithBody::Static(s)),
            ModuleDef::Trait(t) => {
                for it in t.items(db) {
                    if let AssocItem::Function(f) = it
                        && f.has_body(db)
                    {
                        out.push(DefWithBody::Function(f))
                    }
                }
            }
            _ => {}
        }
    }
    for i in m.impl_defs(db) {
        for it in i.items(db) {
            if let AssocItem::Function(f) = it
                && f.has_body(db)
            {
                out.push(DefWithBody::Function(f))
            }
        }
    }
    for c in m.children(db) {
        walk(db, c, out)
    }
}

fn main() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(std::env::args().nth(1).expect("workspace root"));
    let cargo = CargoConfig {
        all_targets: false,
        sysroot: Some(ra_ap_project_model::RustLibSource::Discover),
        ..Default::default()
    };
    // MACROS=1 runs the build scripts (`load_out_dirs_from_check`, i.e. a
    // `cargo check` of the target) and expands proc macros through the
    // toolchain's `rust-analyzer-proc-macro-srv`. Without it, derives and
    // attribute macros stay unexpanded.
    let macros = std::env::var("MACROS").is_ok();
    let load = LoadCargoConfig {
        load_out_dirs_from_check: macros,
        with_proc_macro_server: if macros {
            ProcMacroServerChoice::Sysroot
        } else {
            ProcMacroServerChoice::None
        },
        prefill_caches: false,
        num_worker_threads: 4,
        proc_macro_processes: 1,
    };
    let t = Instant::now();
    let (db, _vfs, _pm) = load_workspace_at(&root, &cargo, &load, &|_| {})?;
    println!("load_workspace_ms\t{}", t.elapsed().as_millis());
    let host = AnalysisHost::with_database(db);

    let t = Instant::now();
    let (n_local, mut bodies) = ra_ap_hir::attach_db(host.raw_database(), || {
        let db = host.raw_database();
        let mut out = Vec::new();
        let local: Vec<Crate> = Crate::all(db)
            .into_iter()
            .filter(|c| c.origin(db).is_local())
            .collect();
        for c in &local {
            walk(db, c.root_module(db), &mut out);
        }
        (local.len(), out)
    });
    println!(
        "local_crates\t{n_local}\nbodies\t{}\nwalk_ms\t{}",
        bodies.len(),
        t.elapsed().as_millis()
    );

    let t = Instant::now();
    let (ok, err, kinds) = ra_ap_hir::attach_db(host.raw_database(), || {
        let db = host.raw_database();
        let (mut ok, mut err) = (0usize, 0usize);
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for b in bodies.drain(..) {
            match b.run_mir_body(db) {
                Ok(()) => ok += 1,
                Err(e) => {
                    err += 1;
                    let s = format!("{e:?}");
                    let k = s
                        .split(|c: char| !c.is_alphanumeric())
                        .next()
                        .unwrap_or("")
                        .to_string();
                    let e2 = kinds.entry(k.clone()).or_default();
                    *e2 += 1;
                    if *e2 <= 3 {
                        println!("sample\t{k}\t{}", s.chars().take(300).collect::<String>());
                    }
                }
            }
        }
        (ok, err, kinds)
    });
    println!(
        "mir_ok\t{ok}\nmir_err\t{err}\nmir_ms\t{}",
        t.elapsed().as_millis()
    );
    for (k, n) in kinds.iter().rev().take(12) {
        println!("mir_err_kind\t{k}\t{n}");
    }

    let t = Instant::now();
    let analysis = host.analysis();
    let idx = StaticIndex::compute(&analysis, VendoredLibrariesConfig::Excluded, false);
    let files = idx.files.len();
    let occ: usize = idx.files.iter().map(|f| f.tokens.len()).sum();
    let (mut toks, mut defs, mut refs) = (0usize, 0usize, 0usize);
    let all: Vec<_> = idx.tokens.iter().collect();
    for (_, d) in &all {
        toks += 1;
        if d.definition.is_some() {
            defs += 1
        }
        refs += d.references.len();
    }
    // DUMP=1: one `sym` line per indexed symbol (name, kind, definition range,
    // reference count), for reading what the index witnessed on a fixture.
    if std::env::var("DUMP").is_ok() {
        for (_, d) in &all {
            let Some(name) = d.display_name.clone() else {
                continue;
            };
            let def = d
                .definition
                .map(|r| format!("{:?}", r.range))
                .unwrap_or_default();
            // The moniker is rust-analyzer's crate-qualified path for the
            // definition: a candidate semantic identity, not a stable one
            // (it changes when the item moves).
            let moniker = match &d.moniker {
                Some(MonikerResult::Moniker(m)) => m.identifier.to_string(),
                Some(MonikerResult::Local { .. }) => "local".into(),
                None => "none".into(),
            };
            // For traits, trait methods and structs: how many impls rust-analyzer
            // finds, including derive-generated impls (not filtered).
            let impls = match (d.kind, d.definition) {
                (
                    SymbolInformationKind::Trait
                    | SymbolInformationKind::TraitMethod
                    | SymbolInformationKind::Struct,
                    Some(r),
                ) => {
                    let pos = FilePosition {
                        file_id: r.file_id,
                        offset: r.range.start(),
                    };
                    let cfg = GotoImplementationConfig {
                        filter_adjacent_derive_implementations: false,
                    };
                    match analysis.goto_implementation(&cfg, pos) {
                        Ok(Some(info)) => info.info.len().to_string(),
                        Ok(None) => "0".into(),
                        Err(_) => "cancelled".into(),
                    }
                }
                _ => "-".into(),
            };
            // FNV-1a over the definition body's text: says whether two
            // occurrences carry the same source, never that they are the
            // same symbol.
            let body = d
                .definition_body
                .and_then(|r| {
                    let text = analysis.file_text(r.file_id).ok()?;
                    let t = text.get(usize::from(r.range.start())..usize::from(r.range.end()))?;
                    Some(format!(
                        "{:016x}",
                        t.bytes().fold(0xcbf29ce484222325u64, |h, b| {
                            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
                        })
                    ))
                })
                .unwrap_or_else(|| "none".into());
            println!(
                "sym\t{name}\tkind={:?}\tdef={def}\trefs={}\tmoniker={moniker}\timpls={impls}\tbody={body}",
                d.kind,
                d.references.len()
            );
        }
    }
    println!(
        "static_index_ms\t{}\nfiles\t{files}\noccurrences\t{occ}\ntokens\t{toks}\ntokens_with_definition\t{defs}\nreference_rows\t{refs}",
        t.elapsed().as_millis()
    );
    Ok(())
}
