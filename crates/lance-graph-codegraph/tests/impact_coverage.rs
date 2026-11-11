//! Coverage-aware impact (D-CODEGRAPH-RECON-2, Gate A).
//!
//! `Codegraph::impact` answers over symbol positions; a caller the store never
//! extracted has none. These tests pin that `impact_report` keeps the impact
//! masks unchanged and reports each kind of missing coverage as a typed gap,
//! never as a fabricated position in `Graded.unknown`. Each gate has a
//! can-fire test and a can-stay-silent twin on a non-trivial store.

use std::path::PathBuf;

use lance_graph_codegraph::artifact::fnv64_hex;
use lance_graph_codegraph::impact_report::{Gap, Scope, SymbolState, Targets};
use lance_graph_codegraph::Codegraph;

const HEADER: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
const RES_HEADER: &str = "#version 1\n#schema reason\tsubject\n";
const FILES_HEADER: &str = "#version 1\n#schema path\tfnv64\n";

/// One crate of a store: its rows (predicate, subject, object) and the source
/// files it was extracted from (path, contents).
struct Krate<'a> {
    name: &'a str,
    rows: &'a [(&'a str, &'a str, &'a str)],
    files: &'a [(&'a str, &'a str)],
}

/// A store directory plus a source root holding each crate's files.
struct Fixture {
    dir: PathBuf,
    store: PathBuf,
    src: PathBuf,
}

impl Fixture {
    fn new(name: &str, crates: &[Krate<'_>]) -> Fixture {
        let dir = std::env::temp_dir().join(format!("lgcg_impact_{name}_{}", std::process::id()));
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
    fn add(&self, k: &Krate<'_>) {
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
            std::fs::write(self.src.join(path), body).unwrap();
            f += &format!("{path}\t{}\n", fnv64_hex(body.as_bytes()));
        }
        std::fs::write(self.store.join(format!("{}.files.tsv", k.name)), f).unwrap();
    }

    fn open(&self) -> Codegraph {
        Codegraph::open(&self.store).unwrap()
    }

    /// Every stored crate fully extracted, freshness checked against `src`.
    fn scope(&self, crates: &[&str]) -> Scope {
        Scope {
            store: Some(self.store.clone()),
            source_root: Some(self.src.clone()),
            targets: crates
                .iter()
                .map(|c| (c.to_string(), Targets::ALL))
                .collect(),
            consumers_closed: false,
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
const LIB: &[(&str, &str, &str)] = &[
    ("DEFINES", "lib::entry", "lib::entry"),
    ("PUBLIC", "lib::entry", "lib::entry"),
    ("DEFINES", "lib::used", "lib::used"),
    ("DEFINES", "lib::helper", "lib::helper"),
    ("CALLS_RESOLVED", "lib::entry", "lib::used"),
];

fn lib_crate() -> Krate<'static> {
    Krate {
        name: "lib",
        rows: LIB,
        files: &[(
            "lib.rs",
            "pub fn entry() { used() }\nfn used() {}\nfn helper() {}\n",
        )],
    }
}

fn has(g: &Codegraph, mask: &[u64], symbol: &str) -> bool {
    g.pos(symbol).is_some_and(|p| {
        mask.get(p as usize / 64)
            .is_some_and(|w| w >> (p % 64) & 1 == 1)
    })
}

fn empty(mask: &[u64]) -> bool {
    mask.iter().all(|&w| w == 0)
}

/// 1. A private symbol with no caller, in a crate whose every target was
/// extracted, whose harvest left no residual and whose sources are unchanged:
/// the empty answer is closed in scope. Anti-vacuity: `lib::used` does have a
/// caller, so the impact query is live, not merely empty.
#[test]
fn no_callers_in_a_fully_observed_scope_is_closed() {
    let fx = Fixture::new("closed", &[lib_crate()]);
    let g = fx.open();
    let r = g.impact_report("lib::helper", &fx.scope(&["lib"])).unwrap();
    assert!(empty(&r.graded.confirmed) && empty(&r.graded.possible) && empty(&r.graded.unknown));
    assert_eq!(r.gaps, Vec::<Gap>::new());
    assert!(r.closed_in_scope());

    let used = g.impact_report("lib::used", &fx.scope(&["lib"])).unwrap();
    assert!(has(&g, &used.graded.confirmed, "lib::entry"));
}

/// 2. A public symbol with no observed caller: the masks stay exactly what
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
        consumers_closed: true,
        ..fx.scope(&["lib"])
    };
    assert!(g
        .impact_report("lib::entry", &closed)
        .unwrap()
        .closed_in_scope());
}

/// 3. Test-target callers: with only the lib target extracted, the
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

/// 4. Unresolved dispatch: `lib::cb` is referenced (its address is taken) and
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
        r.gaps
            .contains(&Gap::UnresolvedDispatch { dynamic_calls: 1 }),
        "{:?}",
        r.gaps
    );

    let quiet = g.impact_report("lib::used", &fx.scope(&["lib"])).unwrap();
    assert!(!quiet
        .gaps
        .iter()
        .any(|x| matches!(x, Gap::UnresolvedDispatch { .. })));
}

/// 5. Stale source: a recorded file changed after extraction. Twins: the
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

/// 6. A truly missing symbol is not "no callers": it is `Missing`, never
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

/// 7. Adding a consumer crate upgrades the answer: before, `lib::entry` has no
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
        consumers_closed: true,
        ..fx.scope(&["lib", "app"])
    };
    assert!(g
        .impact_report("lib::entry", &closed)
        .unwrap()
        .closed_in_scope());

    let app_undeclared = Scope {
        consumers_closed: true,
        ..fx.scope(&["lib"])
    };
    assert_eq!(
        g.impact_report("lib::entry", &app_undeclared).unwrap().gaps,
        vec![Gap::TargetsUndeclared {
            krate: "app".into()
        }]
    );
}

/// 8. An incomplete harvest of the owning crate (a residual, a missing
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
