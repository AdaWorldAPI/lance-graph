//! Load method declarations from a file at run time and call them.
//!
//! Build once, then run against any declarations file; a new method needs a
//! new line of text, not a rebuild:
//!
//! ```text
//! cargo build --release --example method_load
//! target/release/examples/method_load tests/fixtures/declared.methods 4097
//! ```
//!
//! The glove is SAP CATS with SAP's real claims, so a method reading
//! `duration` (claimed `Hypothesized`) is refused, and the refusal is printed.
//! Each method runs on both frontends, which must agree; the independent
//! oracle lives in `tests/declared.rs`.
use std::time::Instant;

use lance_graph_glove_parity::basis::{canonical_fields, SAP_FIELDS};
use lance_graph_glove_parity::decl::parse_decls;
use lance_graph_glove_parity::method::{Frontend, Glove, MethodTable, ReadingId, Universe};
use lance_graph_mask_risc::Planes;
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_sap::bind::{CatsBatch, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[BILLING][i] = Some(INDICATORS[(i * 7 + i / 3) % 4]);
        input[HOURS][i] = Some(HOURS_TEXT[i % 3]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: method_load <file> [rows]");
    let n: usize = args.next().map_or(4097, |s| s.parse().unwrap());

    let t0 = Instant::now();
    let text = std::fs::read_to_string(&path).expect("read declarations");
    let decls = match parse_decls(&text) {
        Ok(d) => d,
        Err(e) => {
            println!("refused: {e:?}");
            std::process::exit(2);
        }
    };
    let parsed = t0.elapsed();

    let batch = batch(n);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let glove = Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis: &basis,
        claims: SAP_FIELDS,
        binder: &cats,
        table: TABLE,
        reading: ReadingId(1),
        universe: Universe(42),
    };
    let t1 = Instant::now();
    let table = MethodTable::prepare_with(&glove, &[], &decls);
    let prepared = t1.elapsed();
    // Each declaration's label and key, taken before the text and the
    // declarations are dropped; calls go by key, so a duplicate label (which
    // `resolve` leaves unnamed on purpose) is still reported, not a panic.
    let methods: Vec<(String, _)> = decls.iter().map(|d| (d.label.clone(), d.key)).collect();
    drop((decls, text));

    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    println!("{path}: parsed in {parsed:?}, prepared in {prepared:?}, rows {n}");
    for (label, key) in methods {
        let q = table.call(key, &planes, Universe(42), Frontend::Quack);
        let r = table.call(key, &planes, Universe(42), Frontend::R2il);
        match (q, r) {
            (Ok(q), Ok(r)) => {
                assert_eq!(q, r, "{label}: frontends disagree");
                println!("  {label}: {q:?}");
            }
            (Err(e), _) | (_, Err(e)) => println!("  {label}: refused {e:?}"),
        }
    }
}
