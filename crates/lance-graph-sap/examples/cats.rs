//! Run: cargo run --manifest-path crates/lance-graph-sap/Cargo.toml --example cats
use lance_graph_sap::{bind::CatsBatch, edge::*, query::CatsQuery, schema::CatsSchema};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<_> = include_str!("../fixtures/cats.txt")
        .lines()
        .map(|s| if s == "\\N" { None } else { Some(s) })
        .collect();
    let mut input: [Vec<_>; 23] = std::array::from_fn(|i| vec![fields[i]; 4]);
    input[5][3] = Some("00000007");
    input[11][2] = Some("OPS");
    input[10][2] = Some("0.125");
    input[20] = vec![Some("123456789012345678901234567890123456789012345678901234567890"); 4];
    let batch = CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )?;
    let mut query = CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30")?;
    let mut sums = vec![0; query.groups()];
    query
        .execute_into(&mut sums)
        .map_err(|e| format!("execution: {e:?}"))?;
    let mut kept = vec![0; batch.len().div_ceil(64)];
    query
        .select_into(&mut kept)
        .map_err(|e| format!("selection: {e:?}"))?;
    println!("Exact scale: {}", batch.scale());
    for total in activity_totals(&batch, &sums)? {
        println!("{} = {} hours", total.activity_type, total.hours);
    }
    let records = bapi_sink(&batch, &kept)?;
    println!(
        "{}: {} selected original assignments",
        BAPI_FUNCTION,
        records.len()
    );
    for record in records {
        println!("{record:?}");
    }
    Ok(())
}
