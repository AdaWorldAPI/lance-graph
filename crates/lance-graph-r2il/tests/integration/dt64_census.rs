//! D-DT64-AUTH census (plan `.claude/plans/d-dt64-method-authority-census-v1.md`).
//!
//! `dt64-authority-census.tsv` lists every production behaviour lens (`LENS`)
//! and every method identity or dispatch path DT64 does not see (`LAB`). The
//! sweep is exact in both directions: a lens mint missing from the table, and
//! a listed row whose token no longer occurs, both fail.

use std::fs;
use std::path::{Path, PathBuf};

use lance_graph_dt64::{AlgebraRef, BindStore, Extent, SchemaRev};
use lance_graph_r2il::polyfill::{polyfill_arities, POLYFILL_LENS};

struct Row {
    kind: String,
    name: String,
    file: String,
    token: String,
    lens: String,
    rev: String,
    arities: String,
    status: String,
}

fn rows(text: &str) -> Vec<Row> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 10, "census row must have 10 columns: {line:?}");
        out.push(Row {
            kind: cols[0].to_string(),
            name: cols[1].to_string(),
            file: cols[2].to_string(),
            token: cols[3].to_string(),
            lens: cols[4].to_string(),
            rev: cols[5].to_string(),
            arities: cols[6].to_string(),
            status: cols[7].to_string(),
        });
    }
    out
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn table() -> Vec<Row> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("dt64-authority-census.tsv");
    rows(&fs::read_to_string(path).expect("census table readable"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            walk(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// Every `(repo-relative file, trimmed line)` that mints a behaviour lens.
fn lens_mints(root: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let crates = root.join("crates");
    let Ok(entries) = fs::read_dir(&crates) else {
        return found;
    };
    for entry in entries.flatten() {
        let mut files = Vec::new();
        walk(&entry.path().join("src"), &mut files);
        for f in files {
            let Ok(text) = fs::read_to_string(&f) else {
                continue;
            };
            let rel = f
                .strip_prefix(root)
                .expect("under root")
                .to_string_lossy()
                .replace('\\', "/");
            for line in text.lines() {
                let t = line.trim();
                if t.contains("compose_classid(HARDCODED")
                    && (t.starts_with("pub const") || t.starts_with("const"))
                {
                    found.push((rel.clone(), t.to_string()));
                }
            }
        }
    }
    found.sort();
    found
}

/// Mints with no LENS row.
fn unlisted(mints: &[(String, String)], rows: &[Row]) -> Vec<(String, String)> {
    mints
        .iter()
        .filter(|(f, t)| {
            !rows
                .iter()
                .any(|r| r.kind == "LENS" && &r.file == f && &r.token == t)
        })
        .cloned()
        .collect()
}

/// LENS rows with no matching mint.
fn stale_lenses(mints: &[(String, String)], rows: &[Row]) -> Vec<String> {
    rows.iter()
        .filter(|r| r.kind == "LENS")
        .filter(|r| !mints.iter().any(|(f, t)| f == &r.file && t == &r.token))
        .map(|r| format!("stale lens row: {} — remove its row", r.name))
        .collect()
}

/// LAB rows whose file is missing or no longer contains the token.
fn stale_exceptions(root: &Path, rows: &[Row]) -> Vec<String> {
    rows.iter()
        .filter(|r| r.kind == "LAB")
        .filter(|r| {
            fs::read_to_string(root.join(&r.file))
                .map(|t| !t.contains(&r.token))
                .unwrap_or(true)
        })
        .map(|r| format!("stale exception: {} — remove its row", r.name))
        .collect()
}

fn parse_lens(s: &str) -> u32 {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("lens is hex")
}

fn parse_arities(s: &str) -> Vec<u8> {
    s.split(',')
        .map(|n| n.parse().expect("arity is decimal"))
        .collect()
}

/// Red if a lens mint exists that the table does not list, or a LENS row
/// names a mint that is gone.
#[test]
fn every_lens_mint_is_listed_and_every_listed_lens_exists() {
    let rows = table();
    let mints = lens_mints(&root());
    let lens_rows = rows.iter().filter(|r| r.kind == "LENS").count();
    assert!(!mints.is_empty());
    assert_eq!(mints.len(), lens_rows, "mints: {mints:?}");
    assert!(unlisted(&mints, &rows).is_empty());
    assert!(stale_lenses(&mints, &rows).is_empty());
}

/// Red if a word is appended to or changed in `POLYFILL` without updating the
/// row (the store refuses a second binding of the lens: `AlgebraImmutable`).
#[test]
fn the_polyfill_row_matches_the_live_declaration() {
    let rows = table();
    let row = rows
        .iter()
        .find(|r| r.kind == "LENS" && r.name == "POLYFILL_LENS")
        .expect("POLYFILL_LENS row");
    assert_eq!(row.status, "CANONICAL");
    assert_eq!(parse_lens(&row.lens), POLYFILL_LENS);
    let arities = parse_arities(&row.arities);
    assert_eq!(arities, polyfill_arities());
    let rev: u32 = row.rev.parse().expect("rev is decimal");

    let mut s = BindStore::new();
    s.declare_algebra(
        POLYFILL_LENS,
        SchemaRev(rev),
        Extent::DR64_CUBE,
        arities.clone(),
    )
    .expect("declare with the row");
    for (p, a) in arities.iter().enumerate() {
        let word = AlgebraRef {
            lens: POLYFILL_LENS,
            word: u8::try_from(p).expect("position fits u8"),
        };
        assert_eq!(s.arity(word), Some(*a));
    }
    let past = AlgebraRef {
        lens: POLYFILL_LENS,
        word: u8::try_from(arities.len()).expect("len fits u8"),
    };
    assert_eq!(s.arity(past), None);
    s.declare_algebra(
        POLYFILL_LENS,
        SchemaRev(rev),
        Extent::DR64_CUBE,
        polyfill_arities(),
    )
    .expect("live declaration is identical to the row");
}

/// Red if a LAB row's file or token is gone (the deviation was removed).
#[test]
fn every_lab_exception_still_deviates() {
    let stale = stale_exceptions(&root(), &table());
    assert!(stale.is_empty(), "{stale:?}");
}

/// C1: red if `unlisted` cannot see a mint missing from the table.
#[test]
fn a_lens_mint_missing_from_the_table_is_caught() {
    let rows = table();
    let mut mints = lens_mints(&root());
    mints.push((
        "crates/x/src/y.rs".to_string(),
        "pub const NEW_LENS: u32 = compose_classid(HARDCODED, 0xF200);".to_string(),
    ));
    let missing = unlisted(&mints, &rows);
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].0, "crates/x/src/y.rs");
}

/// C2: red if `stale_exceptions` cannot see a row whose token is gone.
#[test]
fn a_stale_exception_is_caught() {
    let row = Row {
        kind: "LAB".to_string(),
        name: "ghost".to_string(),
        file: "crates/lance-graph-r2il/src/polyfill.rs".to_string(),
        token: "this token does not occur anywhere 0xDEAD".to_string(),
        lens: "-".to_string(),
        rev: "-".to_string(),
        arities: "-".to_string(),
        status: "LAB".to_string(),
    };
    let stale = stale_exceptions(&root(), &[row]);
    assert_eq!(stale.len(), 1);
    assert!(stale[0].contains("ghost"));
}

/// C3 silence: red if the real tree and table disagree, or if the table is
/// too small for silence to mean anything.
#[test]
fn the_current_tree_is_green() {
    let rows = table();
    let mints = lens_mints(&root());
    assert!(unlisted(&mints, &rows).is_empty());
    assert!(stale_lenses(&mints, &rows).is_empty());
    assert!(stale_exceptions(&root(), &rows).is_empty());
    assert!(rows.iter().filter(|r| r.kind == "LAB").count() >= 1);
    assert!(rows.iter().filter(|r| r.kind == "LENS").count() >= 2);
}
