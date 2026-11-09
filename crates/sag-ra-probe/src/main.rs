//! D-SAG-6-1 lab probe: can ra_ap index a workspace on stable, how fast,
//! how complete is its MIR lowering. Read-only; prints counts and timings.
//!
//! Output is `key<TAB>value` lines. `sample` lines show the first three
//! errors of each MIR lowering error kind. The sysroot must be set
//! (`RustLibSource::Discover`, needs the `rust-src` component): without it
//! `push`, `collect` and `is_empty` are unresolved and coverage drops to 31%.
#![allow(deprecated)]
use ra_ap_hir::{AssocItem, Crate, DefWithBody, Module, ModuleDef, ScopeDef};
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

/// A symbol's identity: crate name, then each descriptor's name followed by
/// a suffix for its kind (`/` namespace, `#` type, `.` term, `().` method,
/// `[]` type parameter, `()` parameter, `!` macro, `:` meta).
fn identity(id: &ra_ap_ide::MonikerIdentifier) -> String {
    use ra_ap_ide::MonikerDescriptorKind as K;
    let parts: Vec<String> = id
        .description
        .iter()
        .map(|d| {
            let suffix = match d.desc {
                K::Namespace => "/",
                K::Type => "#",
                K::Term => ".",
                K::Method => "().",
                K::TypeParameter => "[]",
                K::Parameter => "()",
                K::Macro => "!",
                K::Meta => ":",
            };
            format!("{}{suffix}", d.name)
        })
        .collect();
    format!("{}::{}", id.crate_name, parts.join("::"))
}

fn main() -> anyhow::Result<()> {
    let root = std::fs::canonicalize(std::env::args().nth(1).expect("workspace root"))?;
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
    let (db, vfs, _pm) = load_workspace_at(&root, &cargo, &load, &|_| {})?;
    println!("load_workspace_ms\t{}", t.elapsed().as_millis());
    let host = AnalysisHost::with_database(db);

    // RESOLVE=a::b::c,d::e: resolve each item path in this workspace the
    // way the compiler does, following `use` re-exports, and print the
    // canonical path of what it names. This is the witness that an old path
    // still reaches an item after a move.
    if let Ok(paths) = std::env::var("RESOLVE") {
        ra_ap_hir::attach_db(host.raw_database(), || {
            let db = host.raw_database();
            for p in paths.split(',').filter(|p| !p.is_empty()) {
                println!("resolve\t{p}\t{}", resolve_path(db, p));
            }
        });
    }

    // IMPLS=1: one `impl` row per impl of every local ADT and trait, from the
    // same `Impl::all_for_type` / `all_for_trait` calls `goto_implementation`
    // makes, so each counted impl is listed with what it is:
    //   impl <type or trait> <trait path | inherent> <self type> <impl crate> <items> <file:begin-end>
    // The location is the impl block's source range (lines, 1-based), or
    // `none` for an impl with no source (a builtin derive).
    if std::env::var("IMPLS").is_ok() {
        let rows = ra_ap_hir::attach_db(host.raw_database(), || {
            let db = host.raw_database();
            let mut rows = Vec::new();
            for c in Crate::all(db)
                .into_iter()
                .filter(|c| c.origin(db).is_local())
            {
                list_impls(db, c.root_module(db), &mut rows);
            }
            rows
        });
        let analysis = host.analysis();
        for (row, loc) in rows {
            let at = loc
                .and_then(|(fid, range)| {
                    let file = std::path::Path::new(vfs.file_path(fid).as_path()?.as_str())
                        .strip_prefix(&root)
                        .ok()?
                        .display()
                        .to_string();
                    let li = analysis.file_line_index(fid).ok()?;
                    Some(format!(
                        "{file}:{}-{}",
                        li.line_col(range.start()).line + 1,
                        li.line_col(range.end()).line + 1
                    ))
                })
                .unwrap_or_else(|| "none".into());
            println!("{row}\t{at}");
        }
    }

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

    // NO_MIR=1 skips MIR lowering, which the GRAPH rows do not read.
    if std::env::var("NO_MIR").is_ok() {
        bodies.clear();
    }
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
            // File (relative to the workspace root) and 1-based lines of the
            // name and of the whole definition: the join key against Charon,
            // whose spans are 1-based lines in the same files.
            let pos = |r: ra_ap_ide::FileRange| {
                let file = vfs
                    .file_path(r.file_id)
                    .as_path()
                    .and_then(|a| {
                        std::path::Path::new(a.as_str())
                            .strip_prefix(&root)
                            .ok()
                            .map(|p| p.display().to_string())
                    })
                    .unwrap_or_else(|| "external".into());
                let li = analysis.file_line_index(r.file_id).ok()?;
                let (b, e) = (li.line_col(r.range.start()), li.line_col(r.range.end()));
                Some((file, b.line + 1, e.line + 1))
            };
            let at = d
                .definition
                .and_then(pos)
                .map(|(f, l, _)| format!("{f}:{l}"))
                .unwrap_or_else(|| "none".into());
            let lines = d
                .definition_body
                .and_then(pos)
                .map(|(_, b, e)| format!("{b}-{e}"))
                .unwrap_or_else(|| "none".into());
            // The moniker is rust-analyzer's crate-qualified path for the
            // definition: a candidate semantic identity, not a stable one
            // (it changes when the item moves).
            let moniker = match &d.moniker {
                Some(MonikerResult::Moniker(m)) => m.identifier.to_string(),
                Some(MonikerResult::Local { .. }) => "local".into(),
                None => "none".into(),
            };
            // For traits, trait methods and structs: how many impls rust-analyzer
            // finds, including derive-generated impls (not filtered), and where
            // each one is: `file:begin-end:description`, `;`-separated. The
            // locations let a comparison pair impls one by one instead of
            // comparing two counts.
            let mut impl_at = Vec::new();
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
                        Ok(Some(info)) => {
                            for nav in &info.info {
                                let file = vfs
                                    .file_path(nav.file_id)
                                    .as_path()
                                    .and_then(|a| {
                                        std::path::Path::new(a.as_str())
                                            .strip_prefix(&root)
                                            .ok()
                                            .map(|p| p.display().to_string())
                                    })
                                    .unwrap_or_else(|| "external".into());
                                let lines = analysis
                                    .file_line_index(nav.file_id)
                                    .map(|li| {
                                        format!(
                                            "{}-{}",
                                            li.line_col(nav.full_range.start()).line + 1,
                                            li.line_col(nav.full_range.end()).line + 1
                                        )
                                    })
                                    .unwrap_or_else(|_| "?".into());
                                let desc = nav
                                    .description
                                    .as_deref()
                                    .unwrap_or("")
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" ")
                                    .replace(';', ",");
                                impl_at.push(format!("{file}:{lines}:{desc}"));
                            }
                            info.info.len().to_string()
                        }
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
                "sym\t{name}\tkind={:?}\tdef={def}\trefs={}\tmoniker={moniker}\timpls={impls}\tbody={body}\tat={at}\tlines={lines}\timpl_at={}",
                d.kind,
                d.references.len(),
                impl_at.join(";")
            );
        }
    }
    // XREFS=1: one `occ` line per occurrence of every symbol rust-analyzer
    // gives a moniker: `occ <moniker> <kind> <file> <start> <end> def|ref`.
    // Offsets are byte offsets into the file; files outside the workspace
    // root are skipped. These are the witnessed facts a Kythe producer reads
    // (tools/kythe_entries.py); nothing here is Kythe-specific.
    if std::env::var("XREFS").is_ok() {
        let rel = |file_id| {
            vfs.file_path(file_id).as_path().and_then(|a| {
                std::path::Path::new(a.as_str())
                    .strip_prefix(&root)
                    .ok()
                    .map(|p| p.display().to_string())
            })
        };
        for (_, d) in &all {
            let Some(MonikerResult::Moniker(m)) = &d.moniker else {
                continue;
            };
            // `MonikerIdentifier`'s Display drops each descriptor's kind, so
            // `struct Foo` and `fn Foo` would print the same name. Keep the
            // kind as a SCIP-style suffix so distinct symbols stay distinct.
            let moniker = identity(&m.identifier);
            for r in &d.references {
                let Some(file) = rel(r.range.file_id) else {
                    continue;
                };
                println!(
                    "occ\t{moniker}\t{:?}\t{file}\t{}\t{}\t{}",
                    d.kind,
                    u32::from(r.range.range.start()),
                    u32::from(r.range.range.end()),
                    if r.is_definition { "def" } else { "ref" }
                );
            }
        }
    }
    // GRAPH=1: the typed rows the codegraph (`crates/lance-graph-codegraph`)
    // reads. Every row is a fact rust-analyzer witnessed; nothing is inferred
    // here. Offsets are byte offsets into the file, paths are relative to the
    // workspace root, and files outside the root are skipped.
    //   file   <path> <fnv64 of the text>                 (always every file)
    //   node   <id> <kind> <file> <name s> <name e> <body s> <body e> <body fnv64>
    //   occ    <id> <file> <s> <e> <def|ref|use|pubuse>
    //   call   <caller id> <callee id> <callee kind> <file> <s> <e>
    //   impl   <trait id|inherent> <self id> <file> <s> <e>
    //   macro  <file> <s> <e>        a macro call with arguments: references
    //                                inside it may be missing (KYTHE.md)
    //   diag   <code> <file> <s> <e> <message>
    //   import <target id|unresolved> <file> <use s> <use e> <leaf s> <leaf e>
    //          <alias|-> <pub|priv> <glob|name>
    // `<id>` is the kind-preserving identity of `identity()`. `use` / `pubuse`
    // mark a reference inside a `use` item (private / any visibility).
    // FILES=a.rs,b.rs restricts every row except `file` to what an incremental
    // update of those files needs: rows located in them, plus `occ` and `call`
    // rows whose target is defined in them.
    if std::env::var("GRAPH").is_ok() {
        use ra_ap_syntax::{AstNode, ast, ast::HasName, ast::HasVisibility};
        use std::collections::{BTreeSet, HashMap};
        let rel = |file_id| {
            vfs.file_path(file_id).as_path().and_then(|a| {
                std::path::Path::new(a.as_str())
                    .strip_prefix(&root)
                    .ok()
                    .map(|p| p.display().to_string())
            })
        };
        let only: Option<BTreeSet<String>> = std::env::var("FILES").ok().map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        });
        let keep = |f: &str| only.as_ref().is_none_or(|o| o.contains(f));
        let fnv = |t: &str| {
            t.bytes().fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            })
        };
        // Identity and kind of every definition the index holds, by the
        // position of its name, so call targets resolve to the same id.
        let mut by_pos: HashMap<(ra_ap_ide::FileId, u32), (String, SymbolInformationKind)> =
            HashMap::new();
        let mut def_file: HashMap<String, String> = HashMap::new();
        for (_, d) in &all {
            let (Some(MonikerResult::Moniker(m)), Some(def)) = (&d.moniker, d.definition) else {
                continue;
            };
            let id = identity(&m.identifier);
            by_pos.insert(
                (def.file_id, u32::from(def.range.start())),
                (id.clone(), d.kind),
            );
            if let Some(f) = rel(def.file_id) {
                def_file.insert(id, f);
            }
        }
        // The id each reference position resolves to, for `import` rows.
        let mut occ_at: HashMap<(ra_ap_ide::FileId, u32), String> = HashMap::new();
        for (_, d) in &all {
            let Some(MonikerResult::Moniker(m)) = &d.moniker else {
                continue;
            };
            let id = identity(&m.identifier);
            for r in &d.references {
                occ_at.insert(
                    (r.range.file_id, u32::from(r.range.range.start())),
                    id.clone(),
                );
            }
        }
        let target_kept = |id: &str| def_file.get(id).is_some_and(|f| keep(f));
        let mut parsed: HashMap<ra_ap_ide::FileId, ra_ap_syntax::SourceFile> = HashMap::new();
        for f in &idx.files {
            let Some(path) = rel(f.file_id) else { continue };
            let Ok(text) = analysis.file_text(f.file_id) else {
                continue;
            };
            println!("file\t{path}\t{:016x}", fnv(&text));
            if !keep(&path) {
                continue;
            }
            let Ok(tree) = analysis.parse(f.file_id) else {
                continue;
            };
            for mc in tree.syntax().descendants().filter_map(ast::MacroCall::cast) {
                let has_args = mc.token_tree().is_some_and(|tt| {
                    tt.syntax()
                        .descendants_with_tokens()
                        .any(|e| e.kind() == ra_ap_syntax::SyntaxKind::IDENT)
                });
                if has_args {
                    let r = mc.syntax().text_range();
                    println!(
                        "macro\t{path}\t{}\t{}",
                        u32::from(r.start()),
                        u32::from(r.end())
                    );
                }
            }
            // One `import` row per leaf of every `use` tree:
            //   import <target id|unresolved> <file> <use s> <use e> <leaf s> <leaf e>
            //          <alias|-> <pub|priv> <glob|name>
            // The target is what rust-analyzer resolved the leaf's last
            // segment to (for a glob: the module the glob reads).
            for u in tree.syntax().descendants().filter_map(ast::Use::cast) {
                let vis = if u.visibility().is_some() {
                    "pub"
                } else {
                    "priv"
                };
                let ur = u.syntax().text_range();
                for t in u.syntax().descendants().filter_map(ast::UseTree::cast) {
                    if t.use_tree_list().is_some() {
                        continue;
                    }
                    let glob = t.star_token().is_some();
                    let Some(seg) = t.path().and_then(|p| p.segment()) else {
                        continue;
                    };
                    let Some(nr) = seg.name_ref() else { continue };
                    let lr = nr.syntax().text_range();
                    let target = occ_at
                        .get(&(f.file_id, u32::from(lr.start())))
                        .cloned()
                        .unwrap_or_else(|| "unresolved".into());
                    let alias = t
                        .rename()
                        .and_then(|r| r.name())
                        .map(|n| n.text().to_string())
                        .unwrap_or_else(|| "-".into());
                    println!(
                        "import\t{target}\t{path}\t{}\t{}\t{}\t{}\t{alias}\t{vis}\t{}",
                        u32::from(ur.start()),
                        u32::from(ur.end()),
                        u32::from(lr.start()),
                        u32::from(lr.end()),
                        if glob { "glob" } else { "name" }
                    );
                }
            }
            parsed.insert(f.file_id, tree);
            let cfg = ra_ap_ide::DiagnosticsConfig::test_sample();
            if let Ok(diags) =
                analysis.full_diagnostics(&cfg, ra_ap_ide::AssistResolveStrategy::None, f.file_id)
            {
                for dg in diags {
                    if dg.range.file_id != f.file_id {
                        continue;
                    }
                    println!(
                        "diag\t{}\t{path}\t{}\t{}\t{}",
                        dg.code.as_str(),
                        u32::from(dg.range.range.start()),
                        u32::from(dg.range.range.end()),
                        dg.message.replace(['\t', '\n'], " ")
                    );
                }
            }
        }
        let role =
            |file_id: ra_ap_ide::FileId, at: ra_ap_ide::TextSize, parsed: &mut HashMap<_, _>| {
                let tree: &ra_ap_syntax::SourceFile = parsed
                    .entry(file_id)
                    .or_insert_with(|| analysis.parse(file_id).expect("parse"));
                let tok = tree.syntax().token_at_offset(at).right_biased()?;
                let u = tok.parent_ancestors().find_map(ast::Use::cast)?;
                Some(if u.visibility().is_some() {
                    "pubuse"
                } else {
                    "use"
                })
            };
        let cfg_calls = ra_ap_ide::CallHierarchyConfig {
            exclude_tests: false,
            ra_fixture: ra_ap_ide::RaFixtureConfig::default(),
        };
        for (_, d) in &all {
            let Some(MonikerResult::Moniker(m)) = &d.moniker else {
                continue;
            };
            let id = identity(&m.identifier);
            if let Some(def) = d.definition
                && let Some(file) = rel(def.file_id)
                && keep(&file)
            {
                let (bs, be, digest) = match d.definition_body {
                    Some(b) => {
                        let text = analysis.file_text(b.file_id).ok();
                        let body = text.as_deref().and_then(|t| {
                            t.get(usize::from(b.range.start())..usize::from(b.range.end()))
                        });
                        (
                            u32::from(b.range.start()),
                            u32::from(b.range.end()),
                            body.map(|t| format!("{:016x}", fnv(t)))
                                .unwrap_or("none".into()),
                        )
                    }
                    None => (0, 0, "none".into()),
                };
                println!(
                    "node\t{id}\t{:?}\t{file}\t{}\t{}\t{bs}\t{be}\t{digest}",
                    d.kind,
                    u32::from(def.range.start()),
                    u32::from(def.range.end())
                );
            }
            for r in &d.references {
                let Some(file) = rel(r.range.file_id) else {
                    continue;
                };
                if !(keep(&file) || target_kept(&id)) {
                    continue;
                }
                let kind = if r.is_definition {
                    "def"
                } else {
                    role(r.range.file_id, r.range.range.start(), &mut parsed).unwrap_or("ref")
                };
                println!(
                    "occ\t{id}\t{file}\t{}\t{}\t{kind}",
                    u32::from(r.range.range.start()),
                    u32::from(r.range.range.end())
                );
            }
            // Calls out of every function-like definition in the workspace:
            // rust-analyzer's call hierarchy, which resolves each callee the
            // way the IDE does. A call through a trait bound or `dyn`
            // resolves to the trait's method declaration (kind
            // `TraitMethod`), which names a candidate set, not a callee.
            let fn_like = matches!(
                d.kind,
                SymbolInformationKind::Function
                    | SymbolInformationKind::Method
                    | SymbolInformationKind::StaticMethod
                    | SymbolInformationKind::TraitMethod
            );
            let Some(def) = d.definition else { continue };
            let Some(caller_file) = rel(def.file_id) else {
                continue;
            };
            if !fn_like {
                continue;
            }
            let pos = FilePosition {
                file_id: def.file_id,
                offset: def.range.start(),
            };
            let Ok(Some(items)) = analysis.outgoing_calls(&cfg_calls, pos) else {
                continue;
            };
            for it in items {
                let target = it
                    .target
                    .focus_range
                    .and_then(|fr| by_pos.get(&(it.target.file_id, u32::from(fr.start()))));
                let (callee, ckind) = match target {
                    Some((cid, k)) => (cid.clone(), format!("{k:?}")),
                    None => (format!("unknown:{}", it.target.name), "Unknown".to_string()),
                };
                for fr in it.ranges {
                    let Some(file) = rel(fr.file_id) else {
                        continue;
                    };
                    if !(keep(&file) || keep(&caller_file) || target_kept(&callee)) {
                        continue;
                    }
                    println!(
                        "call\t{id}\t{callee}\t{ckind}\t{file}\t{}\t{}",
                        u32::from(fr.range.start()),
                        u32::from(fr.range.end())
                    );
                }
            }
        }
        // Trait implementations: one row per impl block of a local trait or
        // local type, with the ids the rows above use.
        let impls = ra_ap_hir::attach_db(host.raw_database(), || {
            let db = host.raw_database();
            let mut out = Vec::new();
            for c in Crate::all(db)
                .into_iter()
                .filter(|c| c.origin(db).is_local())
            {
                graph_impls(db, c.root_module(db), &mut out);
            }
            out
        });
        let mut seen = BTreeSet::new();
        for (trait_pos, self_pos, loc) in impls {
            let Some((file_id, range)) = loc else {
                continue;
            };
            let Some(file) = rel(file_id) else { continue };
            let name = |p: Option<(ra_ap_ide::FileId, u32)>, none: &str| {
                p.and_then(|p| by_pos.get(&p).map(|(id, _)| id.clone()))
                    .unwrap_or_else(|| none.into())
            };
            let tr = name(trait_pos, "inherent");
            let st = name(self_pos, "unknown");
            if !(keep(&file) || target_kept(&tr) || target_kept(&st)) {
                continue;
            }
            let row = format!(
                "impl\t{tr}\t{st}\t{file}\t{}\t{}",
                u32::from(range.start()),
                u32::from(range.end())
            );
            if seen.insert(row.clone()) {
                println!("{row}");
            }
        }
    }
    println!(
        "static_index_ms\t{}\nfiles\t{files}\noccurrences\t{occ}\ntokens\t{toks}\ntokens_with_definition\t{defs}\nreference_rows\t{refs}",
        t.elapsed().as_millis()
    );
    Ok(())
}

/// Resolve `crate::m1::…::item` through module scopes, which include `use`
/// imports, so a re-export at an old path resolves to the item it names.
/// Returns the item's canonical `crate::module::name` path, `unresolved`, or
/// `ambiguous` (one name in several namespaces, e.g. a function and a module).
fn resolve_path(db: &dyn ra_ap_hir::db::HirDatabase, path: &str) -> String {
    let segs: Vec<&str> = path.split("::").collect();
    let Some((&krate, rest)) = segs.split_first() else {
        return "unresolved".into();
    };
    let Some(c) = Crate::all(db).into_iter().find(|c| {
        c.origin(db).is_local()
            && c.display_name(db)
                .is_some_and(|n| n.crate_name().as_str() == krate)
    }) else {
        return "unresolved".into();
    };
    let mut module = c.root_module(db);
    let Some((&last, mods)) = rest.split_last() else {
        return "unresolved".into();
    };
    for m in mods {
        let next = module
            .scope(db, None)
            .into_iter()
            .find_map(|(n, d)| match d {
                ScopeDef::ModuleDef(ModuleDef::Module(sub)) if n.as_str() == *m => Some(sub),
                _ => None,
            });
        let Some(next) = next else {
            return "unresolved".into();
        };
        module = next;
    }
    let hits: Vec<ModuleDef> = module
        .scope(db, None)
        .into_iter()
        .filter_map(|(n, d)| match d {
            ScopeDef::ModuleDef(def) if n.as_str() == last => Some(def),
            _ => None,
        })
        .collect();
    match hits.as_slice() {
        [] => "unresolved".into(),
        [def] => canonical(db, *def).unwrap_or_else(|| "unresolved".into()),
        _ => "ambiguous".into(),
    }
}

/// `crate::module::…::name`, the same spelling as rust-analyzer's moniker for
/// a free item.
fn canonical(db: &dyn ra_ap_hir::db::HirDatabase, def: ModuleDef) -> Option<String> {
    let module = def.module(db)?;
    let krate = module
        .krate(db)
        .display_name(db)?
        .crate_name()
        .as_str()
        .to_owned();
    let mut segs = vec![krate];
    for m in module.path_to_root(db).into_iter().rev() {
        if let Some(n) = m.name(db) {
            segs.push(n.as_str().to_owned());
        }
    }
    segs.push(def.name(db)?.as_str().to_owned());
    Some(segs.join("::"))
}

type ImplRow = (String, Option<(ra_ap_ide::FileId, ra_ap_ide::TextRange)>);

fn impl_row(
    db: &dyn ra_ap_hir::db::HirDatabase,
    owner: &str,
    imp: ra_ap_hir::Impl,
    rows: &mut Vec<ImplRow>,
) {
    use ra_ap_hir::HasSource;
    let tr = imp
        .trait_(db)
        .and_then(|t| canonical(db, ModuleDef::Trait(t)))
        .unwrap_or_else(|| "inherent".into());
    let self_ty = imp
        .self_ty(db)
        .as_adt()
        .and_then(|a| canonical(db, ModuleDef::Adt(a)))
        .unwrap_or_else(|| "non-adt".into());
    let krate = imp
        .module(db)
        .krate(db)
        .display_name(db)
        .map(|n| n.crate_name().as_str().to_owned())
        .unwrap_or_else(|| "?".into());
    let loc = imp.source(db).map(|src| {
        let fr = src.syntax().original_file_range_rooted(db);
        (fr.file_id.file_id(db), fr.range)
    });
    rows.push((
        format!(
            "impl\t{owner}\t{tr}\t{self_ty}\t{krate}\t{}",
            imp.items(db).len()
        ),
        loc,
    ));
}

fn list_impls(db: &dyn ra_ap_hir::db::HirDatabase, m: Module, rows: &mut Vec<ImplRow>) {
    for d in m.declarations(db) {
        match d {
            ModuleDef::Adt(a) => {
                let Some(owner) = canonical(db, d) else {
                    continue;
                };
                for imp in ra_ap_hir::Impl::all_for_type(db, a.ty(db)) {
                    impl_row(db, &owner, imp, rows);
                }
            }
            ModuleDef::Trait(t) => {
                let Some(owner) = canonical(db, d) else {
                    continue;
                };
                for imp in ra_ap_hir::Impl::all_for_trait(db, t) {
                    impl_row(db, &owner, imp, rows);
                }
            }
            _ => {}
        }
    }
    for c in m.children(db) {
        list_impls(db, c, rows)
    }
}

/// The name position of a trait or ADT, in the form `GRAPH`'s `by_pos` keys.
type NamePos = Option<(ra_ap_ide::FileId, u32)>;

fn name_pos<N: ra_ap_syntax::ast::HasName>(
    db: &dyn ra_ap_hir::db::HirDatabase,
    src: Option<ra_ap_hir::InFile<N>>,
) -> NamePos {
    use ra_ap_syntax::AstNode;
    let src = src?;
    let name = src.value.name()?;
    let fr = src.with_value(name.syntax()).original_file_range_rooted(db);
    Some((fr.file_id.file_id(db), u32::from(fr.range.start())))
}

/// `GRAPH`'s impl rows: (trait name, self type name, impl block range) for
/// every impl of a local ADT or trait. Duplicates (an impl of a local trait
/// for a local type is found from both) are removed by the caller.
fn graph_impls(
    db: &dyn ra_ap_hir::db::HirDatabase,
    m: Module,
    out: &mut Vec<(
        NamePos,
        NamePos,
        Option<(ra_ap_ide::FileId, ra_ap_ide::TextRange)>,
    )>,
) {
    use ra_ap_hir::HasSource;
    let mut push = |imp: ra_ap_hir::Impl| {
        let tr = imp.trait_(db).and_then(|t| name_pos(db, t.source(db)));
        let st = imp.self_ty(db).as_adt().and_then(|a| match a {
            ra_ap_hir::Adt::Struct(s) => name_pos(db, s.source(db)),
            ra_ap_hir::Adt::Enum(e) => name_pos(db, e.source(db)),
            ra_ap_hir::Adt::Union(u) => name_pos(db, u.source(db)),
        });
        let loc = imp.source(db).map(|src| {
            let fr = src.syntax().original_file_range_rooted(db);
            (fr.file_id.file_id(db), fr.range)
        });
        out.push((tr, st, loc));
    };
    for d in m.declarations(db) {
        match d {
            ModuleDef::Adt(a) => {
                for imp in ra_ap_hir::Impl::all_for_type(db, a.ty(db)) {
                    push(imp);
                }
            }
            ModuleDef::Trait(t) => {
                for imp in ra_ap_hir::Impl::all_for_trait(db, t) {
                    push(imp);
                }
            }
            _ => {}
        }
    }
    for c in m.children(db) {
        graph_impls(db, c, out)
    }
}
