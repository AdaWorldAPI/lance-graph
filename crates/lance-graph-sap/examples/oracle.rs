use lance_graph_sap::{bind::CatsBatch, edge::*, schema::CatsSchema};
fn main() {
    let values: Vec<_> = include_str!("../fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let columns: [_; 23] = std::array::from_fn(|i| [values[i]]);
    let batch = CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| columns[i].as_slice()),
    )
    .unwrap();
    for value in csharp_fields(&batch, 0).unwrap() {
        println!("{}", value.as_deref().unwrap_or("\\N"));
    }
    let projection = ordered_hash_projection(&batch, 0, HashProfile::SmbMiddleware).unwrap();
    println!(
        "{projection}\n{}",
        hash_projection(&projection, b"fixture-key")
    );
}
