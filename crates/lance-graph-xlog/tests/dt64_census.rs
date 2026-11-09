//! The XLOG row of the D-DT64-AUTH census must match the live declaration.
//!
//! The r2il census test owns the two-way sweep; this test exists because
//! `lance-graph-r2il` does not depend on `lance-graph-xlog` (no new
//! dependency edge). Red if `XLOG_WORDS` or its rev changes without the row.

use std::fs;
use std::path::Path;

use lance_graph_dt64::{AlgebraRef, BindStore};
use lance_graph_xlog::lens::{self, XLOG_LENS, XLOG_WORDS};

#[test]
fn the_xlog_row_matches_the_live_declaration() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../lance-graph-r2il/dt64-authority-census.tsv");
    let text = fs::read_to_string(path).expect("census table readable");
    let mut found = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 10, "census row must have 10 columns: {line:?}");
        if cols[0] == "LENS" && cols[1] == "XLOG_LENS" {
            found.push(cols);
        }
    }
    assert_eq!(found.len(), 1);
    let cols = &found[0];
    let lens = u32::from_str_radix(cols[4].trim_start_matches("0x"), 16).expect("hex lens");
    assert_eq!(lens, XLOG_LENS);
    let arities: Vec<u8> = cols[6]
        .split(',')
        .map(|n| n.parse().expect("decimal arity"))
        .collect();
    let live: Vec<u8> = XLOG_WORDS.iter().map(|w| w.columns).collect();
    assert_eq!(arities, live);

    let mut store = BindStore::new();
    lens::declare(&mut store).expect("declare");
    for (p, a) in arities.iter().enumerate() {
        let word = AlgebraRef {
            lens: XLOG_LENS,
            word: u8::try_from(p).expect("position fits u8"),
        };
        assert_eq!(store.arity(word), Some(*a));
    }
    let past = AlgebraRef {
        lens: XLOG_LENS,
        word: u8::try_from(arities.len()).expect("len fits u8"),
    };
    assert_eq!(store.arity(past), None);
}
