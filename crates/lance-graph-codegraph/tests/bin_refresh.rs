//! Intake and refresh over `*.rows.bin` stores, against the golden file pair
//! (`tests/data/golden/cg_lib.rows.{bin,tsv}`, the same rows in both forms).
//! Ids differ between a bin-read and a TSV-read graph (the interning order
//! differs), so graphs are compared by symbol LABEL, never by id.

use std::fs;
use std::path::PathBuf;

use lance_graph_codegraph::artifact_bin::BinRefusal;
use lance_graph_codegraph::{Codegraph, IntakeRefusal, Pos, RefreshRefusal, RefreshStats};

const BIN: &str = "cg_lib.rows.bin";
const TSV: &str = "cg_lib.rows.tsv";

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/golden")
        .join(name)
}

/// A fresh temp dir, unique per call (tests run in parallel).
fn temp_dir(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cg_bin_refresh_{}_{tag}_{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// A store holding only the golden `.rows.bin`.
fn bin_store(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::copy(golden(BIN), dir.join(BIN)).expect("copy bin");
    dir
}

/// A store holding only the golden `.rows.tsv`.
fn tsv_store(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::copy(golden(TSV), dir.join(TSV)).expect("copy tsv");
    dir
}

fn names(g: &Codegraph, mask: &[u64]) -> Vec<String> {
    let mut v: Vec<String> = (0..mask.len() * 64)
        .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
        .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
        .collect();
    v.sort();
    v
}

fn all_names(g: &Codegraph) -> Vec<String> {
    let mut v: Vec<String> = (0..g.words() * 64)
        .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
        .collect();
    v.sort();
    v
}

/// One line per symbol with every graded answer as sorted LABELS.
fn answers(g: &mut Codegraph) -> Vec<String> {
    let mut out = Vec::new();
    for name in all_names(g) {
        let p = g.pos(&name).expect("pos");
        let c = g.callers(p).expect("callers");
        let e = g.callees(p).expect("callees");
        let i = g.impact(p).expect("impact");
        let r = g.references(p);
        out.push(format!(
            "{name} | callers {:?} {:?} {:?} sites {} | callees {:?} {:?} {:?} sites {} | \
             impact {:?} {:?} {:?} | refs {} {}",
            names(g, &c.graded.confirmed),
            names(g, &c.graded.possible),
            names(g, &c.graded.unknown),
            c.sites,
            names(g, &e.graded.confirmed),
            names(g, &e.graded.possible),
            names(g, &e.graded.unknown),
            e.sites,
            names(g, &i.confirmed),
            names(g, &i.possible),
            names(g, &i.unknown),
            r.confirmed,
            r.possible,
        ));
    }
    out.sort();
    out
}

fn callers_of(g: &mut Codegraph, name: &str) -> Vec<String> {
    let p = g.pos(name).expect("pos");
    let c = g.callers(p).expect("callers");
    names(g, &c.graded.confirmed)
}

const CALLER_ROW: &str =
    "CALLS_RESOLVED\tcg_lib::shapes::impl_Shape_for_Square::area\tcg_lib::util::double\t";

/// Red if the bin intake and the TSV intake build different graphs: a lane
/// filled wrongly, a symbol translated through the wrong table slot, or a
/// grade/file/predicate mapped to the wrong code (disable: translate subject
/// ids with the object table offset by one).
#[test]
fn bin_and_tsv_open_to_the_same_graph() {
    let mut from_bin = Codegraph::open(&bin_store("f1_bin")).expect("open bin");
    let mut from_tsv = Codegraph::open(&tsv_store("f1_tsv")).expect("open tsv");

    assert_eq!(from_bin.rows(), from_tsv.rows());
    assert_eq!(all_names(&from_bin), all_names(&from_tsv));
    assert!(all_names(&from_bin).len() > 10, "fixture is not trivial");
    for row in 0..from_bin.rows() {
        assert_eq!(from_bin.row_predicate(row), from_tsv.row_predicate(row));
        assert_eq!(from_bin.row_subject(row), from_tsv.row_subject(row));
        assert_eq!(from_bin.row_object(row), from_tsv.row_object(row));
        assert_eq!(from_bin.site(row), from_tsv.site(row), "site of row {row}");
    }
    assert_eq!(
        from_bin.crate_status("cg_lib"),
        from_tsv.crate_status("cg_lib")
    );

    // Anti-vacuity: the comparison below must cover a non-empty caller set.
    assert_eq!(
        callers_of(&mut from_bin, "cg_lib::util::double"),
        vec!["cg_lib::shapes::impl_Shape_for_Square::area".to_string()]
    );

    assert_eq!(answers(&mut from_bin), answers(&mut from_tsv));
}

/// Red if an unchanged bin refresh reads past the 64-byte header: the body is
/// overwritten (its digest would no longer match) yet the refresh must still
/// answer "no delta" (disable: parse the whole file before comparing digests).
#[test]
fn an_unchanged_bin_refresh_reads_only_the_header() {
    let dir = bin_store("f2");
    let mut g = Codegraph::open(&dir).expect("open");
    let before = answers(&mut g);

    let path = dir.join(BIN);
    let mut bytes = fs::read(&path).expect("read");
    assert!(bytes.len() > 64);
    for b in &mut bytes[64..] {
        *b = 0xEE;
    }
    fs::write(&path, &bytes).expect("write");

    let stats = g.refresh_crate(&dir, "cg_lib").expect("refresh");
    assert_eq!(stats, RefreshStats::default());
    assert_eq!(answers(&mut g), before);
}

/// Red if a changed header digest is trusted without verifying the body
/// (disable: skip `CrateFile::parse` in `prepare_refresh`, or compare only
/// the header). Also checks the refusal changed nothing.
#[test]
fn a_changed_bin_body_is_verified() {
    let dir = bin_store("f2_twin");
    let mut g = Codegraph::open(&dir).expect("open");
    let before = answers(&mut g);

    let path = dir.join(BIN);
    let mut bytes = fs::read(&path).expect("read");
    bytes[16] ^= 0x01; // the header digest now differs from the recorded one
    for b in &mut bytes[64..] {
        *b = 0xEE;
    }
    fs::write(&path, &bytes).expect("write");

    match g.refresh_crate(&dir, "cg_lib") {
        Err(RefreshRefusal::Intake(IntakeRefusal::Bin(BinRefusal::Digest { .. }))) => {}
        other => panic!("expected a digest refusal, got {other:?}"),
    }
    assert_eq!(answers(&mut g), before);
}

/// Red if a directory holding both files reads the TSV (disable: prefer
/// `.rows.tsv` in `read_dir`). The TSV copy lacks one call row, so the two
/// stores answer differently.
#[test]
fn bin_is_preferred_over_tsv() {
    let tsv = fs::read_to_string(golden(TSV)).expect("read tsv");
    let kept: Vec<&str> = tsv.lines().filter(|l| !l.starts_with(CALLER_ROW)).collect();
    assert_eq!(
        kept.len() + 1,
        tsv.lines().count(),
        "exactly one row removed"
    );
    let reduced = format!("{}\n", kept.join("\n"));

    let both = temp_dir("pref_both");
    fs::copy(golden(BIN), both.join(BIN)).expect("copy bin");
    fs::write(both.join(TSV), &reduced).expect("write reduced tsv");

    let reduced_only = temp_dir("pref_tsv");
    fs::write(reduced_only.join(TSV), &reduced).expect("write reduced tsv");

    let mut g_both = Codegraph::open(&both).expect("open both");
    let mut g_bin = Codegraph::open(&bin_store("pref_bin")).expect("open bin");
    let mut g_tsv = Codegraph::open(&reduced_only).expect("open reduced tsv");

    // The removed row really changes an answer.
    assert_eq!(g_tsv.rows() + 1, g_bin.rows());
    assert!(callers_of(&mut g_tsv, "cg_lib::util::double").is_empty());
    assert!(!callers_of(&mut g_bin, "cg_lib::util::double").is_empty());
    assert_ne!(answers(&mut g_tsv), answers(&mut g_bin));

    // The mixed directory is the bin graph, not the TSV one.
    assert_eq!(g_both.rows(), g_bin.rows());
    assert_eq!(answers(&mut g_both), answers(&mut g_bin));
}

/// Red if a store with no binary file stops opening (disable: make `read_dir`
/// list only `.rows.bin`).
#[test]
fn a_tsv_only_store_still_opens() {
    let g = Codegraph::open(&tsv_store("tsv_only")).expect("open tsv");
    assert!(g.rows() > 0);
    assert!(g.words() > 0);
    assert!(g.pos("cg_lib::util::double").is_some());
}
