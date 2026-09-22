use lance_graph_sap::{
    bind::CatsBatch,
    schema::{CatsSchema, FIELD_COUNT},
};

pub fn fixture(n: usize) -> [Vec<Option<&'static str>>; FIELD_COUNT] {
    let base = [
        Some("entry-1"),
        Some("SAP"),
        Some("BillableHours"),
        Some("100"),
        Some("2026-09-01T12:34:56Z"),
        Some("00000042"),
        Some("0000000123"),
        Some("WBS-1"),
        Some("000000000123"),
        Some("2026-09-01T00:00:00Z"),
        Some("8.50"),
        Some("DEV"),
        Some("Billable"),
        Some("2026-09-01T00:00:00Z"),
        Some("2026-09-01T02:00:00+02:00"),
        Some("Valid"),
        None,
        Some("true"),
        Some("fixture-hash"),
        Some("GDPR"),
        Some("fixture note"),
        None,
        None,
    ];
    std::array::from_fn(|i| vec![base[i]; n])
}

pub fn bind(input: &[Vec<Option<&str>>; FIELD_COUNT]) -> CatsBatch {
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}
