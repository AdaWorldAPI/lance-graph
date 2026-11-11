//! Coverage-aware impact (D-CODEGRAPH-RECON-2, Gate A), over the merged `impact_report` API.
//!
//! `suite_a` runs real Charon artifacts (`tests/data/rev1`: `cg_lib` and its consumer `cg_app`,
//! extracted by ruff_charon from `crates/sag-ra-probe/fixtures/codegraph/rev1`). `suite_b` runs
//! synthetic stores with exact rows, each gate with a can-fire test and a can-stay-silent twin.
//! `narrowing` pins that each declared fact removes the gap it answers.

mod suite_a {
    use std::path::{Path, PathBuf};

    use lance_graph_codegraph::impact_report::{
        Consumers, Gap, ImpactReport, Scope, SymbolState, Targets,
    };
    use lance_graph_codegraph::Codegraph;

    const BOTH: &[&str] = &["cg_lib", "cg_app"];

    fn data() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rev1")
    }

    fn source_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../sag-ra-probe/fixtures/codegraph/rev1")
    }

    /// A directory under the system temp dir, removed when the test ends.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let dir =
                std::env::temp_dir().join(format!("lgcg_impact_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Tmp(dir)
        }
    }

    impl std::ops::Deref for Tmp {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A store holding only `crates`, copied from rev1.
    fn store(name: &str, crates: &[&str]) -> Tmp {
        let dir = Tmp::new(name);
        for c in crates {
            for ext in ["rows.tsv", "files.tsv", "residuals.tsv"] {
                std::fs::copy(
                    data().join(format!("{c}.{ext}")),
                    dir.join(format!("{c}.{ext}")),
                )
                .unwrap();
            }
        }
        dir
    }

    /// The fullest scope: every target extracted for every crate in `crates`, consumers
    /// declared, freshness checked.
    fn full(store: &Path, root: &Path, crates: &[&str], deps: &[&str]) -> Scope {
        Scope {
            store: Some(store.to_path_buf()),
            source_root: Some(root.to_path_buf()),
            targets: crates
                .iter()
                .map(|c| (c.to_string(), Targets::ALL))
                .collect(),
            consumers: Consumers::Declared(deps.iter().map(|d| d.to_string()).collect()),
            excluded: vec![],
        }
    }

    fn a_report(dir: &Path, symbol: &str, scope: &Scope) -> ImpactReport {
        let r = Codegraph::open(dir)
            .unwrap()
            .impact_report(symbol, scope)
            .unwrap();
        assert!(
            matches!(r.symbol, SymbolState::Found(_)),
            "{symbol} not found: {:?}",
            r.symbol
        );
        r
    }

    fn confirmed(dir: &Path, r: &ImpactReport) -> Vec<String> {
        let g = Codegraph::open(dir).unwrap();
        let mut out = Vec::new();
        for (w, &word) in r.graded.confirmed.iter().enumerate() {
            for b in 0..64 {
                if word >> b & 1 == 1 {
                    out.push(g.symbol((w * 64 + b) as u32).unwrap().to_string());
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn no_callers_inside_a_fully_observed_scope_has_no_gap() {
        // cg_app::run is public, but nothing depends on cg_app, so the declared scope is complete.
        let dir = store("complete", BOTH);
        let root = source_root();
        let r = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert!(confirmed(&dir, &r).is_empty());
        assert_eq!(r.gaps, vec![]);
        assert!(r.closed_in_scope());
    }

    #[test]
    fn a_public_symbol_with_unobserved_consumers_reports_missing_coverage() {
        let dir = store("lib_only", &["cg_lib"]);
        let root = source_root();
        let r = a_report(
            &dir,
            "cg_lib::util::double",
            &full(&dir, &root, &["cg_lib"], &["cg_app"]),
        );
        // The consumer has no position, so it is not in the answer: it is a gap, never an entry.
        assert!(r.gaps.contains(&Gap::ConsumersNotInStore {
            crates: vec!["cg_app".into()]
        }));
        assert!(!r.closed_in_scope());
        // Undeclared consumers of a public symbol are a gap too.
        let undeclared = Scope {
            consumers: Consumers::NotDeclared,
            ..full(&dir, &root, &["cg_lib"], &[])
        };
        assert!(a_report(&dir, "cg_lib::util::double", &undeclared)
            .gaps
            .iter()
            .any(|g| matches!(g, Gap::ExternalConsumersUnobserved { .. })));
    }

    #[test]
    fn library_only_extraction_reports_unobserved_test_callers() {
        let dir = store("targets", BOTH);
        let root = source_root();
        let lib_only = Scope {
            targets: BOTH
                .iter()
                .map(|c| (c.to_string(), Targets::LIB_ONLY))
                .collect(),
            ..full(&dir, &root, BOTH, &[])
        };
        assert!(a_report(&dir, "cg_app::run", &lib_only)
            .gaps
            .iter()
            .any(|g| matches!(g, Gap::TargetsNotExtracted { missing, .. } if missing.tests)));
        assert!(
            !a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]))
                .gaps
                .iter()
                .any(|g| matches!(g, Gap::TargetsNotExtracted { .. }))
        );
    }

    #[test]
    fn dispatch_only_callers_report_unresolved_dispatch() {
        // cg_lib::area_of dispatches Shape::area, which Rect implements; nothing calls Rect::area
        // directly, so every caller is possible only.
        let dir = store("dispatch", BOTH);
        let root = source_root();
        let r = a_report(
            &dir,
            "cg_lib::shapes::impl_Shape_for_Rect::area",
            &full(&dir, &root, BOTH, &[]),
        );
        assert!(confirmed(&dir, &r).is_empty());
        assert!(
            r.gaps
                .iter()
                .any(|g| matches!(g, Gap::TraitDispatch { possible } if *possible > 0)),
            "{:?}",
            r.gaps
        );
        // Twin: util::triple's only caller (macro_use::checked) is reached by resolved calls only.
        // (util::double would not do: Square::area calls it, and area_of reaches Square::area
        // through dispatch, so area_of is a legitimate possible impact of double.)
        let d = a_report(&dir, "cg_lib::util::triple", &full(&dir, &root, BOTH, &[]));
        assert!(
            !d.gaps
                .iter()
                .any(|g| matches!(g, Gap::TraitDispatch { .. })),
            "{:?}",
            d.gaps
        );
        assert!(!confirmed(&dir, &d).is_empty());
    }

    #[test]
    fn a_changed_source_file_reports_stale_coverage() {
        let dir = store("stale", BOTH);
        let root = Tmp::new("src");
        for c in ["cg_lib", "cg_app"] {
            copy_dir(&source_root().join(c), &root.join(c));
        }
        // cg_app depends on cg_lib, so cg_lib's sources decide the impact of a cg_lib symbol.
        let scope = full(&dir, &root, BOTH, &["cg_app"]);
        assert!(!a_report(&dir, "cg_lib::util::double", &scope)
            .gaps
            .iter()
            .any(|g| matches!(g, Gap::StaleSource { .. })));
        let f = root.join("cg_lib/src/util.rs");
        let mut text = std::fs::read(&f).unwrap();
        text.extend_from_slice(b"\n// edited after extraction\n");
        std::fs::write(&f, text).unwrap();
        let stale = Gap::StaleSource {
            krate: "cg_lib".into(),
            files: vec!["cg_lib/src/util.rs".into()],
        };
        let r = a_report(&dir, "cg_lib::util::double", &scope);
        assert!(r.gaps.contains(&stale), "{:?}", r.gaps);
        // Declared scope narrows it: nothing depends on cg_app, so no cg_lib code can call
        // cg_app::run and cg_lib's edit cannot move that answer.
        let none = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert!(!none.gaps.contains(&stale), "{:?}", none.gaps);
        // Without declared consumers any stored crate may call the public cg_app::run, so the
        // same edit is a gap again.
        let undeclared = Scope {
            consumers: Consumers::NotDeclared,
            ..full(&dir, &root, BOTH, &[])
        };
        let open = a_report(&dir, "cg_app::run", &undeclared);
        assert!(open.gaps.contains(&stale), "{:?}", open.gaps);
        // No source root: freshness is not checked, and that is a gap, not a pass.
        let unchecked = Scope {
            store: None,
            source_root: None,
            ..full(&dir, &root, BOTH, &[])
        };
        assert!(a_report(&dir, "cg_app::run", &unchecked)
            .gaps
            .contains(&Gap::FreshnessUnchecked));
    }

    #[test]
    fn a_symbol_absent_from_the_store_is_not_an_empty_impact() {
        let dir = store("missing", &["cg_lib"]);
        let root = source_root();
        let r = Codegraph::open(&dir)
            .unwrap()
            .impact_report("cg_app::run", &full(&dir, &root, &["cg_lib"], &[]))
            .unwrap();
        // cg_app is not a stored crate: nothing of it was read.
        assert_eq!(
            r.symbol,
            SymbolState::Missing {
                owner_stored: false
            }
        );
        assert_eq!(
            r.observed
                .iter()
                .map(|(c, _)| c.as_str())
                .collect::<Vec<_>>(),
            vec!["cg_lib"]
        );
        assert!(!r.closed_in_scope());
    }

    #[test]
    fn adding_the_consumer_crate_upgrades_missing_coverage_to_confirmed() {
        let root = source_root();
        let lib = store("before", &["cg_lib"]);
        let before = a_report(
            &lib,
            "cg_lib::util::double",
            &full(&lib, &root, &["cg_lib"], &["cg_app"]),
        );
        assert!(!confirmed(&lib, &before).contains(&"cg_app::run".to_string()));
        assert!(before.gaps.contains(&Gap::ConsumersNotInStore {
            crates: vec!["cg_app".into()]
        }));

        let both = store("after", BOTH);
        let after = a_report(
            &both,
            "cg_lib::util::double",
            &full(&both, &root, BOTH, &["cg_app"]),
        );
        assert!(
            confirmed(&both, &after).contains(&"cg_app::run".to_string()),
            "{:?}",
            confirmed(&both, &after)
        );
        assert!(!after
            .gaps
            .iter()
            .any(|g| matches!(g, Gap::ConsumersNotInStore { .. })));
        // The unchanged position-mask answer is a superset of what it was.
        assert!(confirmed(&lib, &before)
            .iter()
            .all(|s| confirmed(&both, &after).contains(s)));
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            if p.is_dir() {
                copy_dir(&p, &to.join(e.file_name()));
            } else {
                std::fs::copy(&p, to.join(e.file_name())).unwrap();
            }
        }
    }

    /// Fails if extraction residuals are dropped from the impact answer: a body ruff_charon could
    /// not represent may call the symbol, so a crate with residuals is a gap even when its CRATE row
    /// is `COMPLETE`. The twin is `no_callers_inside_a_fully_observed_scope_has_no_gap` (rev1 has
    /// none).
    #[test]
    fn extraction_residuals_are_a_coverage_gap() {
        let dir = store("residuals", BOTH);
        let root = source_root();
        let res = dir.join("cg_app.residuals.tsv");
        let mut t = std::fs::read_to_string(&res).unwrap();
        t += "BODY_NOT_STRUCTURED\tcg_app::run\n";
        std::fs::write(&res, t).unwrap();
        let r = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert!(
            matches!(
                r.gaps.as_slice(),
                [Gap::HarvestIncomplete {
                    krate,
                    residuals,
                    partial: false,
                    no_residuals_file: false
                }] if krate == "cg_app" && residuals.total() == 1
            ),
            "{:?}",
            r.gaps
        );
        assert!(!r.closed_in_scope());
    }

    /// Fails if a crate read without a residuals file counts as fully covered: nothing is known
    /// about what its harvest missed.
    #[test]
    fn a_missing_residuals_file_is_a_coverage_gap() {
        let dir = store("no_residuals", BOTH);
        let root = source_root();
        std::fs::remove_file(dir.join("cg_app.residuals.tsv")).unwrap();
        let r = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert!(
            matches!(
                r.gaps.as_slice(),
                [Gap::HarvestIncomplete {
                    krate,
                    no_residuals_file: true,
                    ..
                }] if krate == "cg_app"
            ),
            "{:?}",
            r.gaps
        );
    }

    /// Fails if a crate read without a CRATE row is left out of the report: nothing states that its
    /// extraction completed, so it is a partial crate, and its residual and freshness checks still run.
    #[test]
    fn a_crate_without_a_crate_row_is_partial_not_absent() {
        let dir = store("no_crate_row", BOTH);
        let root = source_root();
        let rows = dir.join("cg_app.rows.tsv");
        let text = std::fs::read_to_string(&rows).unwrap();
        let kept: Vec<&str> = text.lines().filter(|l| !l.starts_with("CRATE\t")).collect();
        assert_eq!(
            kept.len() + 1,
            text.lines().count(),
            "exactly one CRATE row"
        );
        std::fs::write(&rows, kept.join("\n") + "\n").unwrap();
        let r = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert!(
            r.observed.iter().any(|(c, _)| c == "cg_app"),
            "{:?}",
            r.observed
        );
        assert!(
            matches!(
                r.gaps.as_slice(),
                [Gap::HarvestIncomplete {
                    krate,
                    partial: true,
                    ..
                }] if krate == "cg_app"
            ),
            "{:?}",
            r.gaps
        );
    }

    /// Fails if a symbol whose rows a refresh removed still answers `Found` with an empty impact: its
    /// position survives the refresh, but it is no longer in the artifacts.
    #[test]
    fn a_symbol_removed_by_a_refresh_is_not_found() {
        let dir = store("removed", BOTH);
        let root = source_root();
        let mut g = Codegraph::open(&dir).unwrap();
        let scope = full(&dir, &root, BOTH, &[]);
        assert!(matches!(
            g.impact_report("cg_app::run", &scope).unwrap().symbol,
            SymbolState::Found(_)
        ));
        let rows = dir.join("cg_app.rows.tsv");
        let text = std::fs::read_to_string(&rows).unwrap();
        let kept: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with('#') || l.starts_with("CRATE\t"))
            .collect();
        std::fs::write(&rows, kept.join("\n") + "\n").unwrap();
        g.refresh_crate(&dir, "cg_app").unwrap();
        assert!(g.pos("cg_app::run").is_some(), "the position is retained");
        assert!(matches!(
            g.impact_report("cg_app::run", &scope).unwrap().symbol,
            SymbolState::Missing { .. }
        ));
    }

    /// Fails if a file-list row without a hash passes as a successful freshness check: that source
    /// file was never compared.
    #[test]
    fn a_malformed_file_list_row_is_a_freshness_gap() {
        let dir = store("bad_files", BOTH);
        let root = source_root();
        let files = dir.join("cg_app.files.tsv");
        let mut t = std::fs::read_to_string(&files).unwrap();
        t += "cg_app/src/extra.rs\n";
        std::fs::write(&files, t).unwrap();
        let r = a_report(&dir, "cg_app::run", &full(&dir, &root, BOTH, &[]));
        assert_eq!(
            r.gaps,
            vec![Gap::InvalidFileList {
                krate: "cg_app".into()
            }]
        );
    }
}

mod suite_b {
    use std::path::PathBuf;

    use lance_graph_codegraph::impact_report::{Consumers, Gap, Scope, SymbolState, Targets};
    use lance_graph_codegraph::Codegraph;
    use lance_graph_contract::hash::fnv1a;

    pub const HEADER: &str =
        "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
    pub const RES_HEADER: &str = "#version 1\n#schema reason\tsubject\n";
    pub const FILES_HEADER: &str = "#version 1\n#schema path\tfnv64\n";

    fn fnv64_hex(bytes: &[u8]) -> String {
        format!("{:016x}", fnv1a(bytes))
    }

    /// One crate of a store: its rows (predicate, subject, object) and the source
    /// files it was extracted from (path, contents).
    pub struct Krate<'a> {
        pub name: &'a str,
        pub rows: &'a [(&'a str, &'a str, &'a str)],
        pub files: &'a [(&'a str, &'a str)],
    }

    /// A store directory plus a source root holding each crate's files.
    pub struct Fixture {
        pub dir: PathBuf,
        pub store: PathBuf,
        pub src: PathBuf,
    }

    impl Fixture {
        pub fn new(name: &str, crates: &[Krate<'_>]) -> Fixture {
            let dir =
                std::env::temp_dir().join(format!("lgcg_impact_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let (store, src) = (dir.join("store"), dir.join("src"));
            std::fs::create_dir_all(&store).unwrap();
            std::fs::create_dir_all(&src).unwrap();
            let fx = Fixture { dir, store, src };
            for k in crates {
                fx.add(k);
            }
            fx
        }

        /// Write one crate's rows, empty residuals and files table.
        pub fn add(&self, k: &Krate<'_>) {
            let file = k.files.first().map_or("f.rs", |f| f.0);
            let mut t = String::from(HEADER);
            for (p, s, o) in k.rows {
                t += &format!("{p}\t{s}\t{o}\tCOMPILER\t{file}\t1\t2\n");
            }
            t += &format!("CRATE\t{}\tCOMPLETE\tCOMPILER\t-\t-\t-\n", k.name);
            std::fs::write(self.store.join(format!("{}.rows.tsv", k.name)), t).unwrap();
            std::fs::write(
                self.store.join(format!("{}.residuals.tsv", k.name)),
                RES_HEADER,
            )
            .unwrap();
            let mut f = String::from(FILES_HEADER);
            for (path, body) in k.files {
                let at = self.src.join(path);
                std::fs::create_dir_all(at.parent().unwrap()).unwrap();
                std::fs::write(at, body).unwrap();
                f += &format!("{path}\t{}\n", fnv64_hex(body.as_bytes()));
            }
            std::fs::write(self.store.join(format!("{}.files.tsv", k.name)), f).unwrap();
        }

        pub fn open(&self) -> Codegraph {
            Codegraph::open(&self.store).unwrap()
        }

        /// Every stored crate fully extracted, freshness checked against `src`.
        pub fn scope(&self, crates: &[&str]) -> Scope {
            Scope {
                store: Some(self.store.clone()),
                source_root: Some(self.src.clone()),
                targets: crates
                    .iter()
                    .map(|c| (c.to_string(), Targets::ALL))
                    .collect(),
                consumers: Consumers::NotDeclared,
                excluded: Vec::new(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// `lib::entry` is public and calls `lib::used`; `lib::helper` is private
    /// and has no caller.
    pub const LIB: &[(&str, &str, &str)] = &[
        ("DEFINES", "lib::entry", "lib::entry"),
        ("PUBLIC", "lib::entry", "lib::entry"),
        ("DEFINES", "lib::used", "lib::used"),
        ("DEFINES", "lib::helper", "lib::helper"),
        ("CALLS_RESOLVED", "lib::entry", "lib::used"),
    ];

    pub fn lib_crate() -> Krate<'static> {
        Krate {
            name: "lib",
            rows: LIB,
            files: &[(
                "lib.rs",
                "pub fn entry() { used() }\nfn used() {}\nfn helper() {}\n",
            )],
        }
    }

    pub fn has(g: &Codegraph, mask: &[u64], symbol: &str) -> bool {
        g.pos(symbol).is_some_and(|p| {
            mask.get(p as usize / 64)
                .is_some_and(|w| w >> (p % 64) & 1 == 1)
        })
    }

    pub fn empty(mask: &[u64]) -> bool {
        mask.iter().all(|&w| w == 0)
    }

    /// Case 1: A private symbol with no caller, in a crate whose every target was
    /// extracted, whose harvest left no residual and whose sources are unchanged:
    /// the empty answer is closed in scope. Anti-vacuity: `lib::used` does have a
    /// caller, so the impact query is live, not merely empty.
    #[test]
    fn no_callers_in_a_fully_observed_scope_is_closed() {
        let fx = Fixture::new("closed", &[lib_crate()]);
        let g = fx.open();
        let r = g.impact_report("lib::helper", &fx.scope(&["lib"])).unwrap();
        assert!(
            empty(&r.graded.confirmed) && empty(&r.graded.possible) && empty(&r.graded.unknown)
        );
        assert_eq!(r.gaps, Vec::<Gap>::new());
        assert!(r.closed_in_scope());

        let used = g.impact_report("lib::used", &fx.scope(&["lib"])).unwrap();
        assert!(has(&g, &used.graded.confirmed, "lib::entry"));
    }

    /// Case 2: A public symbol with no observed caller: the masks stay exactly what
    /// `impact` returns (no fabricated unknown), and the missing external
    /// callers are a typed gap. `Coverage::no_known_gap()` is true for this
    /// store, and the report is still not closed: no-known-gap is not global
    /// completeness.
    #[test]
    fn a_public_symbol_reports_its_unobserved_external_consumers() {
        let fx = Fixture::new("public", &[lib_crate()]);
        let g = fx.open();
        assert!(g.coverage().no_known_gap());
        let pos = g.pos("lib::entry").unwrap();
        let r = g.impact_report("lib::entry", &fx.scope(&["lib"])).unwrap();
        assert_eq!(r.graded, g.impact(pos).unwrap());
        assert!(
            empty(&r.graded.unknown),
            "missing coverage must not become a position"
        );
        assert_eq!(
            r.gaps,
            vec![Gap::ExternalConsumersUnobserved {
                owner: "lib".into(),
                observed_consumers: vec![],
            }]
        );
        assert!(!r.closed_in_scope());

        // Silence twin: the caller declares the consumer set closed.
        let closed = Scope {
            consumers: Consumers::AllInStore,
            ..fx.scope(&["lib"])
        };
        assert!(g
            .impact_report("lib::entry", &closed)
            .unwrap()
            .closed_in_scope());
    }

    /// Case 3: Test-target callers: with only the lib target extracted, the
    /// test-only caller has no position, so the report names the missing
    /// targets. With the test target extracted (its rows present, declared ALL),
    /// the caller is confirmed and the gap disappears.
    #[test]
    fn test_callers_absent_from_a_lib_extraction_are_a_gap() {
        let fx = Fixture::new("tests", &[lib_crate()]);
        let lib_only = Scope {
            targets: vec![("lib".into(), Targets::LIB_ONLY)],
            ..fx.scope(&["lib"])
        };
        let r = fx.open().impact_report("lib::helper", &lib_only).unwrap();
        assert!(empty(&r.graded.confirmed));
        assert_eq!(
            r.gaps,
            vec![Gap::TargetsNotExtracted {
                krate: "lib".into(),
                missing: Targets {
                    lib: false,
                    tests: true,
                    examples: true,
                    benches: true,
                },
            }]
        );

        // An undeclared crate is not assumed complete.
        let undeclared = Scope {
            targets: vec![],
            ..fx.scope(&["lib"])
        };
        let r = fx.open().impact_report("lib::helper", &undeclared).unwrap();
        assert_eq!(
            r.gaps,
            vec![Gap::TargetsUndeclared {
                krate: "lib".into()
            }]
        );

        // The test target extracted: the caller is a confirmed position.
        let mut rows = LIB.to_vec();
        rows.push((
            "DEFINES",
            "lib::tests::helper_works",
            "lib::tests::helper_works",
        ));
        rows.push(("CALLS_RESOLVED", "lib::tests::helper_works", "lib::helper"));
        let fx2 = Fixture::new(
            "tests_all",
            &[Krate {
                name: "lib",
                rows: &rows,
                files: lib_crate().files,
            }],
        );
        let g = fx2.open();
        let r = g
            .impact_report("lib::helper", &fx2.scope(&["lib"]))
            .unwrap();
        assert!(has(&g, &r.graded.confirmed, "lib::tests::helper_works"));
        assert_eq!(r.gaps, Vec::<Gap>::new());
    }

    /// Case 4: Unresolved dispatch: `lib::cb` is referenced (its address is taken) and
    /// `lib::runner` makes a call the compiler could not name, so a dynamic call
    /// may reach `cb`. Silence twin: `lib::used` is neither an impl nor
    /// referenced, so the same dynamic call is not a gap for it.
    #[test]
    fn an_unresolved_dispatch_that_may_reach_the_symbol_is_a_gap() {
        let mut rows = LIB.to_vec();
        rows.extend([
            ("DEFINES", "lib::cb", "lib::cb"),
            ("DEFINES", "lib::runner", "lib::runner"),
            ("REFERENCES", "lib::entry", "lib::cb"),
            ("UNRESOLVED_CALL", "lib::runner", "dynamic"),
        ]);
        let fx = Fixture::new(
            "dyn",
            &[Krate {
                name: "lib",
                rows: &rows,
                files: lib_crate().files,
            }],
        );
        let g = fx.open();
        let r = g.impact_report("lib::cb", &fx.scope(&["lib"])).unwrap();
        assert!(
            r.gaps.iter().any(|x| matches!(
                x,
                Gap::DynamicCalls {
                    store_dynamic_calls: 1,
                    ..
                }
            )),
            "{:?}",
            r.gaps
        );

        let quiet = g.impact_report("lib::used", &fx.scope(&["lib"])).unwrap();
        assert!(!quiet
            .gaps
            .iter()
            .any(|x| matches!(x, Gap::DynamicCalls { .. })));
    }

    /// Case 5: Stale source: a recorded file changed after extraction. Twins: the
    /// unchanged tree has no gap; a missing files table and a missing source
    /// root are their own gaps, never "fresh".
    #[test]
    fn a_stale_source_revision_is_a_gap() {
        let fx = Fixture::new("stale", &[lib_crate()]);
        let g = fx.open();
        assert!(g
            .impact_report("lib::helper", &fx.scope(&["lib"]))
            .unwrap()
            .gaps
            .is_empty());

        // The file is edited, not deleted: its digest differs, so it is stale, not missing.
        std::fs::write(
            fx.src.join("lib.rs"),
            "pub fn entry() { used(); helper() }\n",
        )
        .unwrap();
        let r = g.impact_report("lib::helper", &fx.scope(&["lib"])).unwrap();
        assert_eq!(
            r.gaps,
            vec![Gap::StaleSource {
                krate: "lib".into(),
                files: vec!["lib.rs".into()],
            }]
        );

        let unchecked = Scope {
            source_root: None,
            ..fx.scope(&["lib"])
        };
        assert_eq!(
            g.impact_report("lib::helper", &unchecked).unwrap().gaps,
            vec![Gap::FreshnessUnchecked]
        );

        std::fs::remove_file(fx.store.join("lib.files.tsv")).unwrap();
        assert_eq!(
            g.impact_report("lib::helper", &fx.scope(&["lib"]))
                .unwrap()
                .gaps,
            vec![Gap::FilesUnrecorded {
                krate: "lib".into()
            }]
        );
    }

    /// Case 6: A truly missing symbol is not "no callers": it is `Missing`, never
    /// closed, and says whether its crate was read at all.
    #[test]
    fn a_missing_symbol_is_missing_not_unaffected() {
        let fx = Fixture::new("missing", &[lib_crate()]);
        let g = fx.open();
        let scope = fx.scope(&["lib"]);
        let r = g.impact_report("lib::nope", &scope).unwrap();
        assert_eq!(r.symbol, SymbolState::Missing { owner_stored: true });
        assert!(!r.closed_in_scope());
        let r = g.impact_report("other::thing", &scope).unwrap();
        assert_eq!(
            r.symbol,
            SymbolState::Missing {
                owner_stored: false
            }
        );
        assert!(!r.closed_in_scope());
        // Twin: a present symbol with no callers is Found.
        assert!(matches!(
            g.impact_report("lib::helper", &scope).unwrap().symbol,
            SymbolState::Found(_)
        ));
    }

    /// Case 7: Adding a consumer crate upgrades the answer: before, `lib::entry` has no
    /// confirmed caller and the external callers are unobserved; after the `app`
    /// crate's compiler facts are stored, `app::main` is a confirmed position and
    /// the gap names `app` as an observed consumer. The gap stays until the
    /// consumer set is declared closed, and `app` must itself be declared.
    #[test]
    fn adding_a_consumer_crate_confirms_its_callers() {
        let fx = Fixture::new("consumer", &[lib_crate()]);
        let before = fx
            .open()
            .impact_report("lib::entry", &fx.scope(&["lib"]))
            .unwrap();
        assert!(empty(&before.graded.confirmed));

        fx.add(&Krate {
            name: "app",
            rows: &[
                ("DEFINES", "app::main", "app::main"),
                ("CALLS_RESOLVED", "app::main", "lib::entry"),
            ],
            files: &[("main.rs", "fn main() { lib::entry() }\n")],
        });
        let g = fx.open();
        let after = g
            .impact_report("lib::entry", &fx.scope(&["lib", "app"]))
            .unwrap();
        assert!(has(&g, &after.graded.confirmed, "app::main"));
        assert_eq!(
            after.gaps,
            vec![Gap::ExternalConsumersUnobserved {
                owner: "lib".into(),
                observed_consumers: vec!["app".into()],
            }]
        );

        let closed = Scope {
            consumers: Consumers::AllInStore,
            ..fx.scope(&["lib", "app"])
        };
        assert!(g
            .impact_report("lib::entry", &closed)
            .unwrap()
            .closed_in_scope());

        let app_undeclared = Scope {
            consumers: Consumers::AllInStore,
            ..fx.scope(&["lib"])
        };
        assert_eq!(
            g.impact_report("lib::entry", &app_undeclared).unwrap().gaps,
            vec![Gap::TargetsUndeclared {
                krate: "app".into()
            }]
        );
    }

    /// Case 8: An incomplete harvest of the owning crate (a residual, a missing
    /// residuals file, or a PARTIAL crate row) is a gap: absent rows prove
    /// nothing. Twin: the same store with empty residuals has no such gap (test 1).
    #[test]
    fn an_incomplete_harvest_of_the_owner_is_a_gap() {
        let fx = Fixture::new("harvest", &[lib_crate()]);
        std::fs::write(
            fx.store.join("lib.residuals.tsv"),
            format!("{RES_HEADER}BODY_NOT_STRUCTURED\tlib::entry\n"),
        )
        .unwrap();
        let r = fx
            .open()
            .impact_report("lib::helper", &fx.scope(&["lib"]))
            .unwrap();
        assert!(
            matches!(r.gaps.as_slice(), [Gap::HarvestIncomplete { krate, residuals, partial: false, no_residuals_file: false }]
            if krate == "lib" && residuals.total() == 1),
            "{:?}",
            r.gaps
        );

        std::fs::remove_file(fx.store.join("lib.residuals.tsv")).unwrap();
        let r = fx
            .open()
            .impact_report("lib::helper", &fx.scope(&["lib"]))
            .unwrap();
        assert!(
            matches!(
                r.gaps.as_slice(),
                [Gap::HarvestIncomplete {
                    no_residuals_file: true,
                    ..
                }]
            ),
            "{:?}",
            r.gaps
        );
    }

    /// Case 9: Regions the extraction did not translate (an opaque module, a
    /// cargo feature left off) are a gap for a found symbol and for a missing
    /// one. A symbol defined only in an uncompiled feature module is missing from
    /// the store without being absent from the crate.
    #[test]
    fn excluded_regions_are_a_gap_for_found_and_missing_symbols() {
        let fx = Fixture::new("excluded", &[lib_crate()]);
        let g = fx.open();
        let excluded = Scope {
            excluded: vec![("lib".into(), vec!["feature iam".into()])],
            ..fx.scope(&["lib"])
        };
        let gap = Gap::RegionsExcluded {
            krate: "lib".into(),
            regions: vec!["feature iam".into()],
        };
        let r = g.impact_report("lib::helper", &excluded).unwrap();
        assert_eq!(r.gaps, vec![gap.clone()]);
        assert!(!r.closed_in_scope());
        let r = g.impact_report("lib::authorize", &excluded).unwrap();
        assert_eq!(r.symbol, SymbolState::Missing { owner_stored: true });
        assert_eq!(r.gaps, vec![gap]);
        // Silence twins: no exclusion for this crate, or an empty list, leaves the
        // found symbol closed and the missing one without gaps.
        for silent in [
            fx.scope(&["lib"]),
            Scope {
                excluded: vec![
                    ("lib".into(), Vec::new()),
                    ("other".into(), vec!["x".into()]),
                ],
                ..fx.scope(&["lib"])
            },
        ] {
            assert!(g
                .impact_report("lib::helper", &silent)
                .unwrap()
                .closed_in_scope());
            assert!(g
                .impact_report("lib::authorize", &silent)
                .unwrap()
                .gaps
                .is_empty());
        }
    }

    /// Case 10: A missing symbol carries its crate's target gaps. Extracting
    /// only the library cannot show that a symbol is absent from the tests.
    #[test]
    fn a_missing_symbol_carries_its_crate_target_gaps() {
        let fx = Fixture::new("missing-targets", &[lib_crate()]);
        let g = fx.open();
        let lib_only = Scope {
            targets: vec![("lib".into(), Targets::LIB_ONLY)],
            ..fx.scope(&["lib"])
        };
        let r = g.impact_report("lib::only_in_tests", &lib_only).unwrap();
        assert!(
            matches!(
                r.gaps.as_slice(),
                [Gap::TargetsNotExtracted { krate, missing }] if krate == "lib" && missing.tests
            ),
            "{:?}",
            r.gaps
        );
        // A symbol outside every stored crate reports no crate gap: nothing of
        // its crate was read, which `owner_stored: false` already says.
        assert!(g
            .impact_report("other::x", &lib_only)
            .unwrap()
            .gaps
            .is_empty());
    }

    /// Case 11: A `.rs` file under a directory the extraction read from, but not
    /// itself recorded, is a gap: its code (a feature-gated module, a binary
    /// root) has no rows. Files in the root and non-Rust files are not scanned.
    #[test]
    fn unread_source_files_under_a_read_directory_are_a_gap() {
        let k = Krate {
            name: "lib",
            rows: &[
                ("DEFINES", "lib::entry", "lib::entry"),
                ("DEFINES", "lib::helper", "lib::helper"),
            ],
            files: &[("crate/lib.rs", "fn entry() {}\nfn helper() {}\n")],
        };
        let fx = Fixture::new("unread", &[k]);
        let g = fx.open();
        let scope = fx.scope(&["lib"]);
        // Silence twin: everything under crate/ is recorded; a non-Rust file and a
        // root-level file are not scanned.
        std::fs::write(fx.src.join("crate/data.json"), "{}").unwrap();
        std::fs::write(fx.src.join("top.rs"), "fn x() {}").unwrap();
        assert!(g
            .impact_report("lib::helper", &scope)
            .unwrap()
            .closed_in_scope());
        std::fs::create_dir_all(fx.src.join("crate/gated")).unwrap();
        std::fs::write(fx.src.join("crate/iam.rs"), "pub fn authorize() {}").unwrap();
        std::fs::write(fx.src.join("crate/gated/deep.rs"), "fn d() {}").unwrap();
        let r = g.impact_report("lib::helper", &scope).unwrap();
        assert_eq!(
            r.gaps,
            vec![Gap::SourceFilesUnread {
                krate: "lib".into(),
                files: vec!["crate/gated/deep.rs".into(), "crate/iam.rs".into()],
            }]
        );
        // A symbol missing from the store may be defined in an unread file.
        let r = g.impact_report("lib::authorize", &scope).unwrap();
        assert_eq!(r.symbol, SymbolState::Missing { owner_stored: true });
        assert!(
            matches!(r.gaps.as_slice(), [Gap::SourceFilesUnread { files, .. }] if files.len() == 2),
            "{:?}",
            r.gaps
        );
    }

    /// Case 12: A private symbol reached by a public caller is affected by that
    /// caller's external consumers, so the external-consumer gap fires for it
    /// too. A private symbol no public function reaches stays closed.
    #[test]
    fn a_private_symbol_behind_a_public_caller_has_unobserved_consumers() {
        let fx = Fixture::new("frontier", &[lib_crate()]);
        let g = fx.open();
        let r = g.impact_report("lib::used", &fx.scope(&["lib"])).unwrap();
        assert!(has(&g, &r.graded.confirmed, "lib::entry"));
        assert!(
            matches!(r.gaps.as_slice(), [Gap::ExternalConsumersUnobserved { owner, .. }] if owner == "lib"),
            "{:?}",
            r.gaps
        );
        assert!(!r.closed_in_scope());
        // Silence twins: nothing public reaches `lib::helper`; and a closed
        // consumer set closes `lib::used`.
        assert!(g
            .impact_report("lib::helper", &fx.scope(&["lib"]))
            .unwrap()
            .closed_in_scope());
        let closed = Scope {
            consumers: Consumers::AllInStore,
            ..fx.scope(&["lib"])
        };
        assert!(g
            .impact_report("lib::used", &closed)
            .unwrap()
            .closed_in_scope());
    }

    /// Case 13: With the consumer set declared closed, every stored crate is a
    /// possible consumer. One whose call into the symbol sits in an excluded
    /// region contributes no position, and its exclusion must still be reported.
    #[test]
    fn a_closed_consumer_set_checks_consumers_without_observed_edges() {
        let app = Krate {
            name: "app",
            rows: &[("DEFINES", "app::main", "app::main")],
            files: &[("app.rs", "fn main() {}\n")],
        };
        let fx = Fixture::new("closed-consumers", &[lib_crate(), app]);
        let g = fx.open();
        let base = Scope {
            consumers: Consumers::AllInStore,
            ..fx.scope(&["lib", "app"])
        };
        // Silence twin: every stored crate fully covered.
        assert!(g
            .impact_report("lib::entry", &base)
            .unwrap()
            .closed_in_scope());
        let excluded = Scope {
            excluded: vec![("app".into(), vec!["feature plugins".into()])],
            ..base
        };
        let r = g.impact_report("lib::entry", &excluded).unwrap();
        assert_eq!(
            r.gaps,
            vec![Gap::RegionsExcluded {
                krate: "app".into(),
                regions: vec!["feature plugins".into()],
            }]
        );
    }

    /// Case 14: A missing symbol in a crate whose harvest is incomplete carries
    /// `HarvestIncomplete`: the failed harvest may have dropped its definition.
    #[test]
    fn a_missing_symbol_carries_its_crate_harvest_gap() {
        let fx = Fixture::new("missing-harvest", &[lib_crate()]);
        let scope = fx.scope(&["lib"]);
        assert!(fx
            .open()
            .impact_report("lib::nope", &scope)
            .unwrap()
            .gaps
            .is_empty());
        std::fs::write(
            fx.store.join("lib.residuals.tsv"),
            format!("{RES_HEADER}BODY_NOT_STRUCTURED\tlib::nope\n"),
        )
        .unwrap();
        let r = fx.open().impact_report("lib::nope", &scope).unwrap();
        assert!(
            matches!(r.gaps.as_slice(), [Gap::HarvestIncomplete { krate, .. }] if krate == "lib"),
            "{:?}",
            r.gaps
        );
    }
}

/// Each declared fact removes exactly the gap it answers: every test shows the gap present
/// first, then absent once the fact is supplied.
mod narrowing {
    use super::suite_b::{lib_crate, Fixture, Krate};
    use lance_graph_codegraph::impact_report::{Consumers, Gap, Scope, Targets};

    #[test]
    fn adding_a_declared_consumer_narrows_the_gap() {
        let fx = Fixture::new("narrow_consumer", &[lib_crate()]);
        let declared = |fx: &Fixture, crates: &[&str]| Scope {
            consumers: Consumers::Declared(vec!["app".into()]),
            ..fx.scope(crates)
        };
        let r = fx
            .open()
            .impact_report("lib::entry", &declared(&fx, &["lib"]))
            .unwrap();
        assert!(
            r.gaps.contains(&Gap::ConsumersNotInStore {
                crates: vec!["app".into()]
            }),
            "{:?}",
            r.gaps
        );

        fx.add(&Krate {
            name: "app",
            rows: &[
                ("DEFINES", "app::main", "app::main"),
                ("CALLS_RESOLVED", "app::main", "lib::entry"),
            ],
            files: &[("main.rs", "fn main() { lib::entry() }\n")],
        });
        let r = fx
            .open()
            .impact_report("lib::entry", &declared(&fx, &["lib", "app"]))
            .unwrap();
        assert!(
            !r.gaps
                .iter()
                .any(|g| matches!(g, Gap::ConsumersNotInStore { .. })),
            "{:?}",
            r.gaps
        );
    }

    #[test]
    fn declaring_test_targets_narrows_the_gap() {
        let fx = Fixture::new("narrow_targets", &[lib_crate()]);
        let g = fx.open();
        let lib_only = Scope {
            targets: vec![("lib".into(), Targets::LIB_ONLY)],
            ..fx.scope(&["lib"])
        };
        let r = g.impact_report("lib::helper", &lib_only).unwrap();
        assert!(
            r.gaps.iter().any(|x| matches!(
                x,
                Gap::TargetsNotExtracted { krate, missing } if krate == "lib" && missing.tests
            )),
            "{:?}",
            r.gaps
        );

        let all = Scope {
            targets: vec![("lib".into(), Targets::ALL)],
            ..fx.scope(&["lib"])
        };
        let r = g.impact_report("lib::helper", &all).unwrap();
        assert!(
            !r.gaps.iter().any(|x| matches!(
                x,
                Gap::TargetsNotExtracted { .. } | Gap::TargetsUndeclared { .. }
            )),
            "{:?}",
            r.gaps
        );
    }

    #[test]
    fn removing_an_excluded_region_narrows_the_gap() {
        let fx = Fixture::new("narrow_excluded", &[lib_crate()]);
        let g = fx.open();
        let excluded = Scope {
            excluded: vec![("lib".into(), vec!["feature iam".into()])],
            ..fx.scope(&["lib"])
        };
        let r = g.impact_report("lib::helper", &excluded).unwrap();
        assert!(
            r.gaps.contains(&Gap::RegionsExcluded {
                krate: "lib".into(),
                regions: vec!["feature iam".into()],
            }),
            "{:?}",
            r.gaps
        );

        let none = Scope {
            excluded: vec![],
            ..fx.scope(&["lib"])
        };
        let r = g.impact_report("lib::helper", &none).unwrap();
        assert!(
            !r.gaps
                .iter()
                .any(|x| matches!(x, Gap::RegionsExcluded { .. })),
            "{:?}",
            r.gaps
        );
    }
}
