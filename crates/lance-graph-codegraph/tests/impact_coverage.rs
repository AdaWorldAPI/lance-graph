//! Coverage-aware impact (D-CODEGRAPH-RECON-2 Gate A), over real Charon artifacts.
//!
//! `tests/data/rev1` holds `cg_lib` and its consumer `cg_app`, extracted by ruff_charon from
//! `crates/sag-ra-probe/fixtures/codegraph/rev1`. Each case distinguishes a different reason an
//! impact answer can be incomplete, and each has a twin where the reason is absent and the gap
//! must not appear.

use std::path::{Path, PathBuf};

use lance_graph_codegraph::impact_report::{
    CoverageGap, Dependents, ExtractionScope, ImpactAnswer, ImpactReport, Targets,
};
use lance_graph_codegraph::Codegraph;

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
        let dir = std::env::temp_dir().join(format!("lgcg_impact_{name}_{}", std::process::id()));
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

/// The fullest scope: tests extracted, dependents declared, freshness checked.
fn full<'a>(store: &'a Path, root: &'a Path, deps: &'a [&'a str]) -> ExtractionScope<'a> {
    ExtractionScope {
        targets: Targets::LibraryAndTests,
        dependents: Dependents::Declared(deps),
        sources: Some((store, root)),
    }
}

fn report(dir: &Path, symbol: &str, scope: &ExtractionScope<'_>) -> ImpactReport {
    match Codegraph::open(dir)
        .unwrap()
        .impact_report(symbol, scope)
        .unwrap()
    {
        ImpactAnswer::Found(r) => r,
        other => panic!("{symbol} not found: {other:?}"),
    }
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
    let dir = store("complete", &["cg_lib", "cg_app"]);
    let root = source_root();
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert!(confirmed(&dir, &r).is_empty());
    assert_eq!(r.gaps, vec![]);
    assert!(r.no_known_gap());
}

#[test]
fn a_public_symbol_with_unobserved_consumers_reports_missing_coverage() {
    let dir = store("lib_only", &["cg_lib"]);
    let root = source_root();
    let r = report(
        &dir,
        "cg_lib::util::double",
        &full(&dir, &root, &["cg_app"]),
    );
    // The consumer has no position, so it is not in the answer: it is a gap, never an entry.
    assert!(r.gaps.contains(&CoverageGap::ConsumersNotInStore {
        crates: vec!["cg_app".into()]
    }));
    assert!(!r.no_known_gap());
    // Undeclared dependents of a public symbol are a gap too.
    let undeclared = ExtractionScope {
        dependents: Dependents::NotDeclared,
        ..full(&dir, &root, &[])
    };
    assert!(report(&dir, "cg_lib::util::double", &undeclared)
        .gaps
        .contains(&CoverageGap::ConsumersNotDeclared));
}

#[test]
fn library_only_extraction_reports_unobserved_test_callers() {
    let dir = store("targets", &["cg_lib", "cg_app"]);
    let root = source_root();
    let lib_only = ExtractionScope {
        targets: Targets::LibraryOnly,
        ..full(&dir, &root, &[])
    };
    assert!(report(&dir, "cg_app::run", &lib_only)
        .gaps
        .contains(&CoverageGap::TestTargetsNotExtracted));
    assert!(!report(&dir, "cg_app::run", &full(&dir, &root, &[]))
        .gaps
        .contains(&CoverageGap::TestTargetsNotExtracted));
}

#[test]
fn dispatch_only_callers_report_unresolved_dispatch() {
    // cg_lib::area_of dispatches Shape::area, which Rect implements; nothing calls Rect::area
    // directly, so every caller is possible only.
    let dir = store("dispatch", &["cg_lib", "cg_app"]);
    let root = source_root();
    let r = report(
        &dir,
        "cg_lib::shapes::impl_Shape_for_Rect::area",
        &full(&dir, &root, &[]),
    );
    assert!(confirmed(&dir, &r).is_empty());
    assert!(
        r.gaps
            .iter()
            .any(|g| matches!(g, CoverageGap::UnresolvedDispatch { possible } if *possible > 0)),
        "{:?}",
        r.gaps
    );
    // Twin: util::triple's only caller (macro_use::checked) is reached by resolved calls only.
    // (util::double would not do: Square::area calls it, and area_of reaches Square::area
    // through dispatch, so area_of is a legitimate possible impact of double.)
    let d = report(&dir, "cg_lib::util::triple", &full(&dir, &root, &[]));
    assert!(
        !d.gaps
            .iter()
            .any(|g| matches!(g, CoverageGap::UnresolvedDispatch { .. })),
        "{:?}",
        d.gaps
    );
    assert!(!confirmed(&dir, &d).is_empty());
}

#[test]
fn a_changed_source_file_reports_stale_coverage() {
    let dir = store("stale", &["cg_lib", "cg_app"]);
    let root = Tmp::new("src");
    for c in ["cg_lib", "cg_app"] {
        copy_dir(&source_root().join(c), &root.join(c));
    }
    assert!(!report(&dir, "cg_app::run", &full(&dir, &root, &[]))
        .gaps
        .iter()
        .any(|g| matches!(g, CoverageGap::StaleSource { .. })));
    let f = root.join("cg_lib/src/util.rs");
    let mut text = std::fs::read(&f).unwrap();
    text.extend_from_slice(b"\n// edited after extraction\n");
    std::fs::write(&f, text).unwrap();
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert!(
        r.gaps.contains(&CoverageGap::StaleSource {
            files: vec!["cg_lib/src/util.rs".into()]
        }),
        "{:?}",
        r.gaps
    );
    // No source root: freshness is not checked, and that is a gap, not a pass.
    let unchecked = ExtractionScope {
        sources: None,
        ..full(&dir, &root, &[])
    };
    assert!(report(&dir, "cg_app::run", &unchecked)
        .gaps
        .contains(&CoverageGap::FreshnessNotChecked));
}

#[test]
fn a_symbol_absent_from_the_store_is_not_an_empty_impact() {
    let dir = store("missing", &["cg_lib"]);
    let root = source_root();
    let a = Codegraph::open(&dir)
        .unwrap()
        .impact_report("cg_app::run", &full(&dir, &root, &[]))
        .unwrap();
    assert_eq!(
        a,
        ImpactAnswer::SymbolNotInStore {
            symbol: "cg_app::run".into(),
            observed: vec!["cg_lib".into()]
        }
    );
}

#[test]
fn adding_the_consumer_crate_upgrades_missing_coverage_to_confirmed() {
    let root = source_root();
    let lib = store("before", &["cg_lib"]);
    let before = report(
        &lib,
        "cg_lib::util::double",
        &full(&lib, &root, &["cg_app"]),
    );
    assert!(!confirmed(&lib, &before).contains(&"cg_app::run".to_string()));
    assert!(before.gaps.contains(&CoverageGap::ConsumersNotInStore {
        crates: vec!["cg_app".into()]
    }));

    let both = store("after", &["cg_lib", "cg_app"]);
    let after = report(
        &both,
        "cg_lib::util::double",
        &full(&both, &root, &["cg_app"]),
    );
    assert!(
        confirmed(&both, &after).contains(&"cg_app::run".to_string()),
        "{:?}",
        confirmed(&both, &after)
    );
    assert!(!after
        .gaps
        .iter()
        .any(|g| matches!(g, CoverageGap::ConsumersNotInStore { .. })));
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
    let dir = store("residuals", &["cg_lib", "cg_app"]);
    let root = source_root();
    let res = dir.join("cg_app.residuals.tsv");
    let mut t = std::fs::read_to_string(&res).unwrap();
    t += "BODY_NOT_STRUCTURED\tcg_app::run\n";
    std::fs::write(&res, t).unwrap();
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert_eq!(
        r.gaps,
        vec![CoverageGap::Residuals {
            krate: "cg_app".into(),
            count: 1
        }]
    );
    assert!(!r.no_known_gap());
}

/// Fails if a crate read without a residuals file counts as fully covered: nothing is known
/// about what its harvest missed.
#[test]
fn a_missing_residuals_file_is_a_coverage_gap() {
    let dir = store("no_residuals", &["cg_lib", "cg_app"]);
    let root = source_root();
    std::fs::remove_file(dir.join("cg_app.residuals.tsv")).unwrap();
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert_eq!(
        r.gaps,
        vec![CoverageGap::ResidualsNotRead {
            krate: "cg_app".into()
        }]
    );
}

/// Fails if a crate read without a CRATE row is left out of the report: nothing states that its
/// extraction completed, so it is a partial crate, and its residual and freshness checks still run.
#[test]
fn a_crate_without_a_crate_row_is_partial_not_absent() {
    let dir = store("no_crate_row", &["cg_lib", "cg_app"]);
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
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert!(
        r.observed.iter().any(|(c, _)| c == "cg_app"),
        "{:?}",
        r.observed
    );
    assert_eq!(
        r.gaps,
        vec![CoverageGap::PartialCrate {
            krate: "cg_app".into()
        }]
    );
}

/// Fails if a symbol whose rows a refresh removed still answers `Found` with an empty impact: its
/// position survives the refresh, but it is no longer in the artifacts.
#[test]
fn a_symbol_removed_by_a_refresh_is_not_found() {
    let dir = store("removed", &["cg_lib", "cg_app"]);
    let root = source_root();
    let mut g = Codegraph::open(&dir).unwrap();
    let scope = full(&dir, &root, &[]);
    assert!(matches!(
        g.impact_report("cg_app::run", &scope).unwrap(),
        ImpactAnswer::Found(_)
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
        g.impact_report("cg_app::run", &scope).unwrap(),
        ImpactAnswer::SymbolNotInStore { .. }
    ));
}

/// Fails if a file-list row without a hash passes as a successful freshness check: that source
/// file was never compared.
#[test]
fn a_malformed_file_list_row_is_a_freshness_gap() {
    let dir = store("bad_files", &["cg_lib", "cg_app"]);
    let root = source_root();
    let files = dir.join("cg_app.files.tsv");
    let mut t = std::fs::read_to_string(&files).unwrap();
    t += "cg_app/src/extra.rs\n";
    std::fs::write(&files, t).unwrap();
    let r = report(&dir, "cg_app::run", &full(&dir, &root, &[]));
    assert_eq!(
        r.gaps,
        vec![CoverageGap::InvalidFileList {
            krate: "cg_app".into()
        }]
    );
}
